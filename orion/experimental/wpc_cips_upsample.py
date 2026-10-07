"""WPC-compressed CIPS support for U-Net transposed-convolution upsampling.

The U-Net decoder uses ``ConvTranspose2d(kernel_size=2, stride=2)``.  In CIPS,
the low-resolution input and high-resolution output have different channel
capacities per ciphertext.  This module builds the direct cyclic diagonals
between those two layouts, so the online path expands ciphertext groups
without decrypting, repacking, or encoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import weakref
from typing import Any

import numpy as np
import torch

from orion.backend.python.tensors import CipherTensor
from orion.experimental.wpc_cips_baseline import (
    CIPSConvCase,
    LAYOUT_CIPS,
    pack_tensor,
    unpack_output,
)
from orion.experimental.wpc_cips_layer import WPCCIPSConv2dPlan, _tuple2


def _is_power_of_two(value: int) -> bool:
    return int(value) > 0 and int(value) & (int(value) - 1) == 0


def _group_ranges(total: int, capacity: int) -> tuple[tuple[int, int], ...]:
    if int(total) <= 0 or int(capacity) <= 0:
        raise ValueError("channel count and capacity must be positive")
    return tuple(
        (start, min(int(total), start + int(capacity)))
        for start in range(0, int(total), int(capacity))
    )


@dataclass(frozen=True)
class CIPSTranspose2ConvCase:
    """Geometry for a 2x CIPS transposed convolution."""

    slots: int
    input_channels: int
    output_channels: int
    input_height: int
    input_width: int
    kernel_height: int = 2
    kernel_width: int = 2
    stride_height: int = 2
    stride_width: int = 2
    pad_height_before: int = 0
    pad_width_before: int = 0
    output_pad_height: int = 0
    output_pad_width: int = 0
    dilation_height: int = 1
    dilation_width: int = 1

    def __post_init__(self) -> None:
        dimensions = (
            self.slots,
            self.input_channels,
            self.output_channels,
            self.input_height,
            self.input_width,
            self.kernel_height,
            self.kernel_width,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("CIPS transposed-convolution dimensions must be positive")
        if not _is_power_of_two(self.slots):
            raise ValueError("slots must be a power of two")
        if not _is_power_of_two(self.input_height) or not _is_power_of_two(
            self.input_width
        ):
            raise ValueError("input height and width must be powers of two")
        if (int(self.kernel_height), int(self.kernel_width)) != (2, 2):
            raise ValueError("the WPC decoder path supports kernel size 2x2 only")
        if (int(self.stride_height), int(self.stride_width)) != (2, 2):
            raise ValueError("the WPC decoder path supports stride 2x2 only")
        if (int(self.pad_height_before), int(self.pad_width_before)) != (0, 0):
            raise ValueError("the WPC decoder path supports zero padding only")
        if (int(self.output_pad_height), int(self.output_pad_width)) != (0, 0):
            raise ValueError("the WPC decoder path supports zero output padding only")
        if (int(self.dilation_height), int(self.dilation_width)) != (1, 1):
            raise ValueError("the WPC decoder path supports dilation one only")
        if int(self.slots) % (int(self.input_height) * int(self.input_width)):
            raise ValueError("slots must be divisible by input height*width")
        if int(self.slots) % (int(self.output_height) * int(self.output_width)):
            raise ValueError("slots must be divisible by output height*width")

    @property
    def height(self) -> int:
        """Input height alias used by the shared plan's input packer."""

        return int(self.input_height)

    @property
    def width(self) -> int:
        """Input width alias used by the shared plan's input packer."""

        return int(self.input_width)

    @property
    def output_height(self) -> int:
        return int(
            (int(self.input_height) - 1) * int(self.stride_height)
            - 2 * int(self.pad_height_before)
            + int(self.dilation_height) * (int(self.kernel_height) - 1)
            + int(self.output_pad_height)
            + 1
        )

    @property
    def output_width(self) -> int:
        return int(
            (int(self.input_width) - 1) * int(self.stride_width)
            - 2 * int(self.pad_width_before)
            + int(self.dilation_width) * (int(self.kernel_width) - 1)
            + int(self.output_pad_width)
            + 1
        )

    @property
    def input_channel_capacity(self) -> int:
        return int(self.slots) // (int(self.input_height) * int(self.input_width))

    @property
    def output_channel_capacity(self) -> int:
        return int(self.slots) // (int(self.output_height) * int(self.output_width))

    @property
    def input_group_ranges(self) -> tuple[tuple[int, int], ...]:
        return _group_ranges(self.input_channels, self.input_channel_capacity)

    @property
    def output_group_ranges(self) -> tuple[tuple[int, int], ...]:
        return _group_ranges(self.output_channels, self.output_channel_capacity)

    @property
    def input_group_count(self) -> int:
        return len(self.input_group_ranges)

    @property
    def output_group_count(self) -> int:
        return len(self.output_group_ranges)

    @property
    def transform_count(self) -> int:
        return int(self.input_group_count * self.output_group_count)

    def local_case(self, output_group: int, input_group: int) -> CIPSConvCase:
        """Return the low-resolution input case used by shared input packing."""

        del output_group
        input_start, input_end = self.input_group_ranges[int(input_group)]
        channel_count = int(input_end - input_start)
        return CIPSConvCase(
            slots=int(self.slots),
            input_channels=channel_count,
            output_channels=channel_count,
            height=int(self.input_height),
            width=int(self.input_width),
            kernel_height=1,
            kernel_width=1,
            pad_height_before=0,
            pad_width_before=0,
        )

    def local_output_case(self, output_group: int) -> CIPSConvCase:
        output_start, output_end = self.output_group_ranges[int(output_group)]
        channel_count = int(output_end - output_start)
        return CIPSConvCase(
            slots=int(self.slots),
            input_channels=channel_count,
            output_channels=channel_count,
            height=int(self.output_height),
            width=int(self.output_width),
            kernel_height=1,
            kernel_width=1,
            pad_height_before=0,
            pad_width_before=0,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {key: int(value) for key, value in asdict(self).items()}
        result.update(
            {
                "output_height": int(self.output_height),
                "output_width": int(self.output_width),
                "input_channel_capacity": int(self.input_channel_capacity),
                "output_channel_capacity": int(self.output_channel_capacity),
                "input_group_ranges": [list(value) for value in self.input_group_ranges],
                "output_group_ranges": [list(value) for value in self.output_group_ranges],
                "transform_count": int(self.transform_count),
            }
        )
        return result


def input_slot(
    case: CIPSTranspose2ConvCase,
    local_channel: int,
    height: int,
    width: int,
) -> int:
    return int(
        (int(height) * int(case.input_width) + int(width))
        * int(case.input_channel_capacity)
        + int(local_channel)
    )


def output_slot(
    case: CIPSTranspose2ConvCase,
    local_channel: int,
    height: int,
    width: int,
) -> int:
    return int(
        (int(height) * int(case.output_width) + int(width))
        * int(case.output_channel_capacity)
        + int(local_channel)
    )


def build_transpose2_cips_group_transforms(
    weights: np.ndarray,
    case: CIPSTranspose2ConvCase,
) -> dict[str, dict[str, Any]]:
    """Build direct low-resolution-to-high-resolution CIPS diagonals.

    PyTorch stores transposed-convolution weights as
    ``[input_channel, output_channel, kernel_height, kernel_width]``.
    """

    kernel = np.asarray(weights, dtype=np.float64)
    expected = (
        int(case.input_channels),
        int(case.output_channels),
        int(case.kernel_height),
        int(case.kernel_width),
    )
    if tuple(kernel.shape) != expected:
        raise ValueError(f"weight tensor shape is {tuple(kernel.shape)}, expected {expected}")

    transforms: dict[str, dict[str, Any]] = {}
    for output_group, (output_start, output_end) in enumerate(
        case.output_group_ranges
    ):
        for input_group, (input_start, input_end) in enumerate(
            case.input_group_ranges
        ):
            diagonals: dict[int, np.ndarray] = {}
            for local_input, global_input in enumerate(
                range(int(input_start), int(input_end))
            ):
                for local_output, global_output in enumerate(
                    range(int(output_start), int(output_end))
                ):
                    for kernel_height in range(int(case.kernel_height)):
                        for kernel_width in range(int(case.kernel_width)):
                            weight = float(
                                kernel[
                                    global_input,
                                    global_output,
                                    kernel_height,
                                    kernel_width,
                                ]
                            )
                            if weight == 0.0:
                                continue
                            for source_height in range(int(case.input_height)):
                                for source_width in range(int(case.input_width)):
                                    target_height = int(
                                        source_height * int(case.stride_height)
                                        - int(case.pad_height_before)
                                        + kernel_height * int(case.dilation_height)
                                    )
                                    target_width = int(
                                        source_width * int(case.stride_width)
                                        - int(case.pad_width_before)
                                        + kernel_width * int(case.dilation_width)
                                    )
                                    if not (
                                        0 <= target_height < int(case.output_height)
                                        and 0 <= target_width < int(case.output_width)
                                    ):
                                        continue
                                    source_slot = input_slot(
                                        case,
                                        local_input,
                                        source_height,
                                        source_width,
                                    )
                                    target_slot = output_slot(
                                        case,
                                        local_output,
                                        target_height,
                                        target_width,
                                    )
                                    rotation = int(
                                        (source_slot - target_slot) % int(case.slots)
                                    )
                                    diagonal = diagonals.setdefault(
                                        rotation,
                                        np.zeros((int(case.slots),), dtype=np.float64),
                                    )
                                    diagonal[target_slot] += weight
            key = f"out{output_group}_in{input_group}"
            transforms[key] = {
                "key": key,
                "output_group": int(output_group),
                "input_group": int(input_group),
                "output_channel_range": [int(output_start), int(output_end)],
                "input_channel_range": [int(input_start), int(input_end)],
                "diagonals": {
                    int(rotation): diagonal
                    for rotation, diagonal in sorted(diagonals.items())
                    if bool(np.any(diagonal != 0.0))
                },
            }
    return transforms


def conv_transpose2d_reference(
    tensor: np.ndarray,
    weights: np.ndarray,
    case: CIPSTranspose2ConvCase,
) -> np.ndarray:
    """Independent clear oracle for the supported transposed convolution."""

    source = np.asarray(tensor, dtype=np.float64)
    kernel = np.asarray(weights, dtype=np.float64)
    expected_input = (
        int(case.input_channels),
        int(case.input_height),
        int(case.input_width),
    )
    expected_weight = (
        int(case.input_channels),
        int(case.output_channels),
        int(case.kernel_height),
        int(case.kernel_width),
    )
    if tuple(source.shape) != expected_input:
        raise ValueError(f"input tensor shape is {tuple(source.shape)}, expected {expected_input}")
    if tuple(kernel.shape) != expected_weight:
        raise ValueError(f"weight tensor shape is {tuple(kernel.shape)}, expected {expected_weight}")
    output = np.zeros(
        (int(case.output_channels), int(case.output_height), int(case.output_width)),
        dtype=np.float64,
    )
    for input_channel in range(int(case.input_channels)):
        for source_height in range(int(case.input_height)):
            for source_width in range(int(case.input_width)):
                source_value = float(source[input_channel, source_height, source_width])
                for output_channel in range(int(case.output_channels)):
                    for kernel_height in range(int(case.kernel_height)):
                        for kernel_width in range(int(case.kernel_width)):
                            target_height = int(
                                source_height * int(case.stride_height)
                                - int(case.pad_height_before)
                                + kernel_height * int(case.dilation_height)
                            )
                            target_width = int(
                                source_width * int(case.stride_width)
                                - int(case.pad_width_before)
                                + kernel_width * int(case.dilation_width)
                            )
                            if (
                                0 <= target_height < int(case.output_height)
                                and 0 <= target_width < int(case.output_width)
                            ):
                                output[output_channel, target_height, target_width] += (
                                    source_value
                                    * float(
                                        kernel[
                                            input_channel,
                                            output_channel,
                                            kernel_height,
                                            kernel_width,
                                        ]
                                    )
                                )
    return output


def pack_upsample_output(
    tensor: np.ndarray,
    case: CIPSTranspose2ConvCase,
) -> np.ndarray:
    values = np.asarray(tensor, dtype=np.float64)
    expected = (
        int(case.output_channels),
        int(case.output_height),
        int(case.output_width),
    )
    if tuple(values.shape) != expected:
        raise ValueError(f"output tensor shape is {tuple(values.shape)}, expected {expected}")
    return np.stack(
        [
            pack_tensor(
                values[start:end],
                case.local_output_case(output_group),
                LAYOUT_CIPS,
            )
            for output_group, (start, end) in enumerate(case.output_group_ranges)
        ]
    )


def unpack_upsample_output(
    messages: np.ndarray,
    case: CIPSTranspose2ConvCase,
) -> np.ndarray:
    values = np.asarray(messages, dtype=np.float64)
    expected = (int(case.output_group_count), int(case.slots))
    if tuple(values.shape) != expected:
        raise ValueError(f"packed output shape is {tuple(values.shape)}, expected {expected}")
    return np.concatenate(
        [
            unpack_output(
                values[output_group],
                case.local_output_case(output_group),
                LAYOUT_CIPS,
            )
            for output_group in range(int(case.output_group_count))
        ],
        axis=0,
    )


class WPCCIPSConvTranspose2dPlan(WPCCIPSConv2dPlan):
    """Installed compressed CIPS path for an Orion ``ConvTranspose2d``."""

    def __init__(
        self,
        layer: Any,
        *,
        input_shape: tuple[int, int, int, int] | torch.Size,
        slots: int,
    ) -> None:
        shape = tuple(int(value) for value in input_shape)
        if len(shape) != 4 or int(shape[0]) != 1:
            raise ValueError("WPC CIPS ConvTranspose2d requires input shape [1,C,H,W]")
        if int(shape[1]) != int(getattr(layer, "in_channels", -1)):
            raise ValueError("input shape channel count does not match ConvTranspose2d")
        kernel = _tuple2(getattr(layer, "kernel_size"))
        stride = _tuple2(getattr(layer, "stride"))
        padding = _tuple2(getattr(layer, "padding"))
        output_padding = _tuple2(getattr(layer, "output_padding"))
        dilation = _tuple2(getattr(layer, "dilation"))
        if int(getattr(layer, "groups", 1)) != 1:
            raise ValueError("the WPC decoder path supports groups=1 only")

        self._layer_ref = weakref.ref(layer)
        self.layer_name = str(getattr(layer, "name", layer.__class__.__name__))
        self.input_shape = shape
        self.case = CIPSTranspose2ConvCase(
            slots=int(slots),
            input_channels=int(getattr(layer, "in_channels")),
            output_channels=int(getattr(layer, "out_channels")),
            input_height=int(shape[2]),
            input_width=int(shape[3]),
            kernel_height=int(kernel[0]),
            kernel_width=int(kernel[1]),
            stride_height=int(stride[0]),
            stride_width=int(stride[1]),
            pad_height_before=int(padding[0]),
            pad_width_before=int(padding[1]),
            output_pad_height=int(output_padding[0]),
            output_pad_width=int(output_padding[1]),
            dilation_height=int(dilation[0]),
            dilation_width=int(dilation[1]),
        )
        self.output_shape = (
            1,
            int(self.case.output_channels),
            int(self.case.output_height),
            int(self.case.output_width),
        )
        self.scheme: Any | None = None
        self.level: int | None = None
        self.output_level: int | None = None
        self.compressed_transform_ids: dict[str, int] = {}
        self.full_control_transform_ids: dict[str, int] = {}
        self.online_transform_ids: dict[str, int] = {}
        self.transform_rows: dict[str, dict[str, Any]] = {}
        self.bias_plaintext: Any | None = None
        self.bias_plaintext_payload_bytes = 0
        self.include_full_control = False
        self.storage_mode = "compressed"
        self.compiled = False
        self.cleaned = False
        self.last_evaluation: dict[str, Any] = {}
        self.record_sequence = True

    @property
    def input_packing_signature(self) -> tuple[Any, ...]:
        return (
            "wpc_cips",
            int(self.case.slots),
            int(self.case.input_channels),
            int(self.case.input_height),
            int(self.case.input_width),
            tuple(tuple(value) for value in self.case.input_group_ranges),
        )

    @property
    def output_packing_signature(self) -> tuple[Any, ...]:
        return (
            "wpc_cips",
            int(self.case.slots),
            int(self.case.output_channels),
            int(self.case.output_height),
            int(self.case.output_width),
            tuple(tuple(value) for value in self.case.output_group_ranges),
        )

    def _layer(self) -> Any:
        layer = self._layer_ref()
        if layer is None:
            raise RuntimeError("the Orion ConvTranspose2d owning this plan no longer exists")
        return layer

    def _build_group_transforms(
        self, weight: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        return build_transpose2_cips_group_transforms(weight, self.case)

    def clear_reference(self, tensor: np.ndarray | torch.Tensor) -> np.ndarray:
        source = np.asarray(
            tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor,
            dtype=np.float64,
        )
        if tuple(source.shape) == self.input_shape:
            source = source[0]
        weight, bias = self._weight_bias()
        return conv_transpose2d_reference(source, weight, self.case) + bias[
            :, None, None
        ]

    def _bias_messages(self, bias: np.ndarray) -> np.ndarray:
        values = np.broadcast_to(
            bias[:, None, None],
            (
                int(self.case.output_channels),
                int(self.case.output_height),
                int(self.case.output_width),
            ),
        )
        return pack_upsample_output(values, self.case)

    def decrypt_unpack(self, value: CipherTensor) -> np.ndarray:
        if self.scheme is None:
            raise RuntimeError("WPC transposed-convolution plan has no scheme")
        if getattr(value, "_wpc_cips_packing_signature", None) != self.output_packing_signature:
            raise ValueError("output does not have this plan's high-resolution CIPS packing")
        decoded = np.asarray(self.scheme.decode(self.scheme.decrypt(value)), dtype=np.float64)
        return unpack_upsample_output(decoded, self.case)


__all__ = [
    "CIPSTranspose2ConvCase",
    "WPCCIPSConvTranspose2dPlan",
    "build_transpose2_cips_group_transforms",
    "conv_transpose2d_reference",
    "input_slot",
    "output_slot",
    "pack_upsample_output",
    "unpack_upsample_output",
]
