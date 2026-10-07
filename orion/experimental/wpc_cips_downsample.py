"""Stride-two WPC CIPS convolution and encrypted layout restoration.

WPC's stride-two Rotation-Padding convolution leaves results on a sparse
subset of the input CIPS grid.  The following one-level linear transform
compacts those values into the ordinary lower-resolution CIPS layout and may
merge channel groups made redundant by the smaller spatial dimensions.  No
decrypt, decode, clear repack, encode, or re-encrypt occurs at this boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import weakref
from typing import Any

import numpy as np
import torch

from orion.backend.python.tensors import CipherTensor
from orion.experimental.wpc_cips_baseline import (
    CIPSConvCase,
    LAYOUT_CIPS,
    pack_tensor,
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
class CIPSStride2ConvCase:
    """Geometry for sparse stride-two output followed by CIPS compaction."""

    slots: int
    input_channels: int
    output_channels: int
    input_height: int
    input_width: int
    kernel_height: int
    kernel_width: int
    pad_height_before: int
    pad_width_before: int
    stride_height: int = 2
    stride_width: int = 2

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
            raise ValueError("stride-two CIPS dimensions must be positive")
        if not _is_power_of_two(self.slots):
            raise ValueError("slots must be a power of two")
        if not _is_power_of_two(self.input_height) or not _is_power_of_two(
            self.input_width
        ):
            raise ValueError("input height and width must be powers of two")
        if (int(self.stride_height), int(self.stride_width)) != (2, 2):
            raise ValueError("this experimental CIPS path supports stride 2x2 only")
        if int(self.slots) % (int(self.input_height) * int(self.input_width)):
            raise ValueError("slots must be divisible by input height*width")
        if int(self.output_height) <= 0 or int(self.output_width) <= 0:
            raise ValueError("convolution has an empty output")
        if int(self.slots) % (int(self.output_height) * int(self.output_width)):
            raise ValueError("slots must be divisible by output height*width")

    @property
    def height(self) -> int:
        """Input height alias used by the shared compiler."""

        return int(self.input_height)

    @property
    def width(self) -> int:
        """Input width alias used by the shared compiler."""

        return int(self.input_width)

    @property
    def output_height(self) -> int:
        return int(
            math.floor(
                (
                    int(self.input_height)
                    + 2 * int(self.pad_height_before)
                    - (int(self.kernel_height) - 1)
                    - 1
                )
                / int(self.stride_height)
                + 1
            )
        )

    @property
    def output_width(self) -> int:
        return int(
            math.floor(
                (
                    int(self.input_width)
                    + 2 * int(self.pad_width_before)
                    - (int(self.kernel_width) - 1)
                    - 1
                )
                / int(self.stride_width)
                + 1
            )
        )

    @property
    def sparse_channel_capacity(self) -> int:
        return int(self.slots) // (int(self.input_height) * int(self.input_width))

    @property
    def compact_channel_capacity(self) -> int:
        return int(self.slots) // (int(self.output_height) * int(self.output_width))

    @property
    def input_group_ranges(self) -> tuple[tuple[int, int], ...]:
        return _group_ranges(self.input_channels, self.sparse_channel_capacity)

    @property
    def output_group_ranges(self) -> tuple[tuple[int, int], ...]:
        """Sparse output groups, still limited by the input-grid capacity."""

        return _group_ranges(self.output_channels, self.sparse_channel_capacity)

    @property
    def compact_output_group_ranges(self) -> tuple[tuple[int, int], ...]:
        return _group_ranges(self.output_channels, self.compact_channel_capacity)

    @property
    def input_group_count(self) -> int:
        return len(self.input_group_ranges)

    @property
    def output_group_count(self) -> int:
        return len(self.output_group_ranges)

    @property
    def compact_output_group_count(self) -> int:
        return len(self.compact_output_group_ranges)

    @property
    def transform_count(self) -> int:
        return int(self.input_group_count * self.output_group_count)

    def local_case(self, output_group: int, input_group: int) -> CIPSConvCase:
        input_start, input_end = self.input_group_ranges[int(input_group)]
        output_start, output_end = self.output_group_ranges[int(output_group)]
        return CIPSConvCase(
            slots=int(self.slots),
            input_channels=int(input_end - input_start),
            output_channels=int(output_end - output_start),
            height=int(self.input_height),
            width=int(self.input_width),
            kernel_height=int(self.kernel_height),
            kernel_width=int(self.kernel_width),
            pad_height_before=int(self.pad_height_before),
            pad_width_before=int(self.pad_width_before),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            key: int(value) for key, value in asdict(self).items()
        }
        result.update(
            {
                "output_height": int(self.output_height),
                "output_width": int(self.output_width),
                "sparse_channel_capacity": int(self.sparse_channel_capacity),
                "compact_channel_capacity": int(self.compact_channel_capacity),
                "input_group_ranges": [list(value) for value in self.input_group_ranges],
                "sparse_output_group_ranges": [
                    list(value) for value in self.output_group_ranges
                ],
                "compact_output_group_ranges": [
                    list(value) for value in self.compact_output_group_ranges
                ],
                "transform_count": int(self.transform_count),
            }
        )
        return result


def sparse_output_slot(
    case: CIPSStride2ConvCase,
    local_channel: int,
    output_height: int,
    output_width: int,
) -> int:
    """Slot occupied by a stride-two result before reshaping."""

    anchor_height = int(output_height) * int(case.stride_height)
    anchor_width = int(output_width) * int(case.stride_width)
    return int(
        (anchor_height * int(case.input_width) + anchor_width)
        * int(case.sparse_channel_capacity)
        + int(local_channel)
    )


def compact_output_slot(
    case: CIPSStride2ConvCase,
    local_channel: int,
    output_height: int,
    output_width: int,
) -> int:
    """Slot occupied after restoring the lower-resolution CIPS layout."""

    return int(
        (int(output_height) * int(case.output_width) + int(output_width))
        * int(case.compact_channel_capacity)
        + int(local_channel)
    )


def build_stride2_cips_group_transforms(
    weights: np.ndarray,
    case: CIPSStride2ConvCase,
) -> dict[str, dict[str, Any]]:
    """Build transforms whose active outputs remain on the sparse stride grid."""

    kernel = np.asarray(weights, dtype=np.float64)
    expected = (
        int(case.output_channels),
        int(case.input_channels),
        int(case.kernel_height),
        int(case.kernel_width),
    )
    if tuple(kernel.shape) != expected:
        raise ValueError(f"weight tensor shape is {tuple(kernel.shape)}, expected {expected}")

    transforms: dict[str, dict[str, Any]] = {}
    spatial_length = int(case.input_height) * int(case.input_width)
    capacity = int(case.sparse_channel_capacity)
    for output_group, (output_start, output_end) in enumerate(
        case.output_group_ranges
    ):
        for input_group, (input_start, input_end) in enumerate(
            case.input_group_ranges
        ):
            diagonals: dict[int, np.ndarray] = {}
            for local_output, global_output in enumerate(
                range(int(output_start), int(output_end))
            ):
                for local_input, global_input in enumerate(
                    range(int(input_start), int(input_end))
                ):
                    for kernel_height in range(int(case.kernel_height)):
                        for kernel_width in range(int(case.kernel_width)):
                            weight = float(
                                kernel[
                                    global_output,
                                    global_input,
                                    kernel_height,
                                    kernel_width,
                                ]
                            )
                            if weight == 0.0:
                                continue
                            flattened_offset = int(
                                (kernel_height - int(case.pad_height_before))
                                * int(case.input_width)
                                + kernel_width
                                - int(case.pad_width_before)
                            )
                            for output_height in range(int(case.output_height)):
                                for output_width in range(int(case.output_width)):
                                    output_slot = sparse_output_slot(
                                        case,
                                        local_output,
                                        output_height,
                                        output_width,
                                    )
                                    anchor_position = int(
                                        output_height
                                        * int(case.stride_height)
                                        * int(case.input_width)
                                        + output_width * int(case.stride_width)
                                    )
                                    source_position = int(
                                        (anchor_position + flattened_offset)
                                        % spatial_length
                                    )
                                    input_slot = int(
                                        source_position * capacity + local_input
                                    )
                                    rotation = int(
                                        (input_slot - output_slot) % int(case.slots)
                                    )
                                    diagonal = diagonals.setdefault(
                                        rotation,
                                        np.zeros((int(case.slots),), dtype=np.float64),
                                    )
                                    diagonal[output_slot] += weight
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


def rotation_padded_stride2_reference(
    tensor: np.ndarray,
    weights: np.ndarray,
    case: CIPSStride2ConvCase,
) -> np.ndarray:
    """Independent clear oracle using WPC's flattened Rotation Padding."""

    source = np.asarray(tensor, dtype=np.float64)
    kernel = np.asarray(weights, dtype=np.float64)
    expected_input = (
        int(case.input_channels),
        int(case.input_height),
        int(case.input_width),
    )
    expected_weight = (
        int(case.output_channels),
        int(case.input_channels),
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
    spatial_length = int(case.input_height) * int(case.input_width)
    for output_channel in range(int(case.output_channels)):
        for output_height in range(int(case.output_height)):
            for output_width in range(int(case.output_width)):
                anchor = int(
                    output_height
                    * int(case.stride_height)
                    * int(case.input_width)
                    + output_width * int(case.stride_width)
                )
                total = 0.0
                for input_channel in range(int(case.input_channels)):
                    for kernel_height in range(int(case.kernel_height)):
                        for kernel_width in range(int(case.kernel_width)):
                            offset = int(
                                (kernel_height - int(case.pad_height_before))
                                * int(case.input_width)
                                + kernel_width
                                - int(case.pad_width_before)
                            )
                            source_position = int((anchor + offset) % spatial_length)
                            source_height, source_width = divmod(
                                source_position, int(case.input_width)
                            )
                            total += float(
                                source[input_channel, source_height, source_width]
                                * kernel[
                                    output_channel,
                                    input_channel,
                                    kernel_height,
                                    kernel_width,
                                ]
                            )
                output[output_channel, output_height, output_width] = total
    return output


def pack_sparse_output(
    tensor: np.ndarray, case: CIPSStride2ConvCase
) -> np.ndarray:
    """Pack logical stride-two output into its sparse per-group layout."""

    values = np.asarray(tensor, dtype=np.float64)
    expected = (
        int(case.output_channels),
        int(case.output_height),
        int(case.output_width),
    )
    if tuple(values.shape) != expected:
        raise ValueError(f"output tensor shape is {tuple(values.shape)}, expected {expected}")
    messages: list[np.ndarray] = []
    for start, end in case.output_group_ranges:
        message = np.zeros((int(case.slots),), dtype=np.float64)
        for local_channel, global_channel in enumerate(range(start, end)):
            for height in range(int(case.output_height)):
                for width in range(int(case.output_width)):
                    message[
                        sparse_output_slot(case, local_channel, height, width)
                    ] = values[global_channel, height, width]
        messages.append(message)
    return np.stack(messages)


def unpack_sparse_output(
    messages: np.ndarray, case: CIPSStride2ConvCase
) -> np.ndarray:
    values = np.asarray(messages, dtype=np.float64)
    expected = (int(case.output_group_count), int(case.slots))
    if tuple(values.shape) != expected:
        raise ValueError(f"packed sparse shape is {tuple(values.shape)}, expected {expected}")
    output = np.empty(
        (int(case.output_channels), int(case.output_height), int(case.output_width)),
        dtype=np.float64,
    )
    for group, (start, end) in enumerate(case.output_group_ranges):
        for local_channel, global_channel in enumerate(range(start, end)):
            for height in range(int(case.output_height)):
                for width in range(int(case.output_width)):
                    output[global_channel, height, width] = values[
                        group, sparse_output_slot(case, local_channel, height, width)
                    ]
    return output


def pack_compact_output(
    tensor: np.ndarray, case: CIPSStride2ConvCase
) -> np.ndarray:
    """Pack logical output into ordinary lower-resolution CIPS groups."""

    values = np.asarray(tensor, dtype=np.float64)
    expected = (
        int(case.output_channels),
        int(case.output_height),
        int(case.output_width),
    )
    if tuple(values.shape) != expected:
        raise ValueError(f"output tensor shape is {tuple(values.shape)}, expected {expected}")
    messages: list[np.ndarray] = []
    for start, end in case.compact_output_group_ranges:
        message = np.zeros((int(case.slots),), dtype=np.float64)
        for local_channel, global_channel in enumerate(range(start, end)):
            for height in range(int(case.output_height)):
                for width in range(int(case.output_width)):
                    message[
                        compact_output_slot(case, local_channel, height, width)
                    ] = values[global_channel, height, width]
        messages.append(message)
    return np.stack(messages)


def unpack_compact_output(
    messages: np.ndarray, case: CIPSStride2ConvCase
) -> np.ndarray:
    values = np.asarray(messages, dtype=np.float64)
    expected = (int(case.compact_output_group_count), int(case.slots))
    if tuple(values.shape) != expected:
        raise ValueError(f"packed compact shape is {tuple(values.shape)}, expected {expected}")
    output = np.empty(
        (int(case.output_channels), int(case.output_height), int(case.output_width)),
        dtype=np.float64,
    )
    for group, (start, end) in enumerate(case.compact_output_group_ranges):
        for local_channel, global_channel in enumerate(range(start, end)):
            for height in range(int(case.output_height)):
                for width in range(int(case.output_width)):
                    output[global_channel, height, width] = values[
                        group, compact_output_slot(case, local_channel, height, width)
                    ]
    return output


def build_downsample_reshape_transforms(
    case: CIPSStride2ConvCase,
) -> dict[str, dict[str, Any]]:
    """Build sparse-to-compact CIPS permutation transforms."""

    transforms: dict[str, dict[str, Any]] = {}
    for output_group, (output_start, output_end) in enumerate(
        case.compact_output_group_ranges
    ):
        for input_group, (input_start, input_end) in enumerate(
            case.output_group_ranges
        ):
            overlap_start = max(int(output_start), int(input_start))
            overlap_end = min(int(output_end), int(input_end))
            if overlap_start >= overlap_end:
                continue
            diagonals: dict[int, np.ndarray] = {}
            for global_channel in range(overlap_start, overlap_end):
                sparse_channel = int(global_channel - int(input_start))
                compact_channel = int(global_channel - int(output_start))
                for height in range(int(case.output_height)):
                    for width in range(int(case.output_width)):
                        input_slot = sparse_output_slot(
                            case, sparse_channel, height, width
                        )
                        output_slot = compact_output_slot(
                            case, compact_channel, height, width
                        )
                        rotation = int((input_slot - output_slot) % int(case.slots))
                        diagonal = diagonals.setdefault(
                            rotation,
                            np.zeros((int(case.slots),), dtype=np.float64),
                        )
                        diagonal[output_slot] = 1.0
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
                },
            }
    return transforms


class WPCCIPSStride2Conv2dPlan(WPCCIPSConv2dPlan):
    """Installed WPC plan that emits a sparse stride-two CIPS result."""

    def __init__(
        self,
        layer: Any,
        *,
        input_shape: tuple[int, int, int, int] | torch.Size,
        slots: int,
    ) -> None:
        shape = tuple(int(value) for value in input_shape)
        if len(shape) != 4 or int(shape[0]) != 1:
            raise ValueError("WPC stride-two Conv2d requires input shape [1,C,H,W]")
        if int(shape[1]) != int(getattr(layer, "in_channels", -1)):
            raise ValueError("input shape channel count does not match Conv2d")
        kernel = _tuple2(getattr(layer, "kernel_size"))
        stride = _tuple2(getattr(layer, "stride"))
        padding = _tuple2(getattr(layer, "padding"))
        dilation = _tuple2(getattr(layer, "dilation"))
        if stride != (2, 2):
            raise ValueError("WPC stride-two CIPS plan requires stride 2x2")
        if dilation != (1, 1):
            raise ValueError("WPC stride-two CIPS plan supports dilation one only")
        if int(getattr(layer, "groups", 1)) != 1:
            raise ValueError("WPC stride-two CIPS plan supports groups=1 only")
        if any(int(size) % 2 == 0 for size in kernel):
            raise ValueError("WPC Rotation Padding requires odd kernels")
        if padding != (kernel[0] // 2, kernel[1] // 2):
            raise ValueError("WPC stride-two path requires symmetric same padding")

        self._layer_ref = weakref.ref(layer)
        self.layer_name = str(getattr(layer, "name", layer.__class__.__name__))
        self.input_shape = shape
        self.case = CIPSStride2ConvCase(
            slots=int(slots),
            input_channels=int(getattr(layer, "in_channels")),
            output_channels=int(getattr(layer, "out_channels")),
            input_height=int(shape[2]),
            input_width=int(shape[3]),
            kernel_height=int(kernel[0]),
            kernel_width=int(kernel[1]),
            pad_height_before=int(padding[0]),
            pad_width_before=int(padding[1]),
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
    def output_packing_signature(self) -> tuple[Any, ...]:
        return (
            "wpc_cips_stride2_sparse",
            int(self.case.slots),
            int(self.case.output_channels),
            int(self.case.input_height),
            int(self.case.input_width),
            int(self.case.output_height),
            int(self.case.output_width),
            (2, 2),
            tuple(tuple(value) for value in self.case.output_group_ranges),
        )

    def _build_group_transforms(
        self, weight: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        return build_stride2_cips_group_transforms(weight, self.case)

    def clear_reference(self, tensor: np.ndarray | torch.Tensor) -> np.ndarray:
        source = np.asarray(
            tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor,
            dtype=np.float64,
        )
        if tuple(source.shape) == self.input_shape:
            source = source[0]
        weight, bias = self._weight_bias()
        return rotation_padded_stride2_reference(source, weight, self.case) + bias[
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
        return pack_sparse_output(values, self.case)

    def decrypt_unpack(self, value: CipherTensor) -> np.ndarray:
        if self.scheme is None:
            raise RuntimeError("WPC stride-two plan has no scheme")
        if getattr(value, "_wpc_cips_packing_signature", None) != self.output_packing_signature:
            raise ValueError("output does not have this plan's sparse CIPS packing")
        decoded = np.asarray(self.scheme.decode(self.scheme.decrypt(value)), dtype=np.float64)
        return unpack_sparse_output(decoded, self.case)


class WPCCIPSDownsampleReshapePlan:
    """One-level encrypted sparse-to-compact CIPS layout transform."""

    def __init__(
        self,
        source_plan: WPCCIPSStride2Conv2dPlan,
        *,
        level: int,
        bsgs_ratio: float = 2.0,
    ) -> None:
        self._source_ref = weakref.ref(source_plan)
        self.case = source_plan.case
        self.level = int(level)
        if self.level <= 0:
            raise ValueError("downsample reshape requires one rescale level")
        self.output_level = int(self.level - 1)
        self.bsgs_ratio = float(bsgs_ratio)
        self.scheme: Any | None = None
        self.transform_ids: dict[str, int] = {}
        self.transform_rows: dict[str, dict[str, Any]] = {}
        self.compiled = False
        self.cleaned = False
        self.last_evaluation: dict[str, Any] = {}

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    @property
    def input_packing_signature(self) -> tuple[Any, ...]:
        source = self._source_ref()
        if source is None:
            raise RuntimeError("source stride-two plan no longer exists")
        return source.output_packing_signature

    @property
    def output_packing_signature(self) -> tuple[Any, ...]:
        return (
            "wpc_cips",
            int(self.case.slots),
            int(self.case.output_channels),
            int(self.case.output_height),
            int(self.case.output_width),
            tuple(tuple(value) for value in self.case.compact_output_group_ranges),
        )

    def compile(self, scheme: Any) -> dict[str, Any]:
        if self.compiled:
            raise RuntimeError("downsample reshape plan is already compiled")
        if self.cleaned:
            raise RuntimeError("a cleaned downsample reshape plan cannot be reused")
        source = self._source_ref()
        if source is None or not source.compiled:
            raise RuntimeError("compile the stride-two source plan first")
        if source.scheme is not scheme:
            raise ValueError("source and reshape plans must use the same scheme")
        if int(source.output_level) != int(self.level):
            raise ValueError(
                f"source output level {source.output_level} does not match reshape level {self.level}"
            )
        self.scheme = scheme
        rows = build_downsample_reshape_transforms(self.case)
        bytes_per_diagonal = int(
            (self.level + 1 + len(scheme.params.get_logp()))
            * int(scheme.params.get_ring_degree())
            * 8
        )
        try:
            for key, row in rows.items():
                diagonals = row["diagonals"]
                indices = [int(value) for value in sorted(diagonals)]
                flattened = np.concatenate(
                    [np.asarray(diagonals[index], dtype=np.float32) for index in indices]
                ).tolist()
                transform_id = int(
                    scheme.backend.GenerateLinearTransform(
                        indices,
                        flattened,
                        int(self.level),
                        float(self.bsgs_ratio),
                        "none",
                    )
                )
                scheme.lt_evaluator.generate_rotation_keys(transform_id)
                self.transform_ids[key] = transform_id
                self.transform_rows[key] = {
                    **{name: value for name, value in row.items() if name != "diagonals"},
                    "diagonal_count": int(len(diagonals)),
                    "full_qp_payload_bytes": int(len(diagonals) * bytes_per_diagonal),
                }
            self.compiled = True
            return self.storage_summary()
        except Exception:
            self.cleanup()
            raise

    def storage_summary(self) -> dict[str, Any]:
        return {
            "storage_mode": "ordinary_full_qp_permutation",
            "transform_count": int(len(self.transform_rows)),
            "diagonal_count": int(
                sum(int(row["diagonal_count"]) for row in self.transform_rows.values())
            ),
            "full_qp_payload_bytes": int(
                sum(
                    int(row["full_qp_payload_bytes"])
                    for row in self.transform_rows.values()
                )
            ),
        }

    def validate_consumer(self, consumer_plan: WPCCIPSConv2dPlan) -> None:
        if self.output_packing_signature != consumer_plan.input_packing_signature:
            raise ValueError("reshape output packing does not match consumer input packing")
        if int(self.output_level) != int(consumer_plan.level):
            raise ValueError("reshape output level does not match consumer plan level")

    def evaluate(self, value: CipherTensor) -> CipherTensor:
        if not self.compiled or self.scheme is None:
            raise RuntimeError("compile the downsample reshape plan before evaluation")
        if not isinstance(value, CipherTensor):
            raise TypeError("downsample reshape expects a grouped CipherTensor")
        if getattr(value, "_wpc_cips_packing_signature", None) != self.input_packing_signature:
            raise ValueError("reshape input does not have the sparse stride-two packing")
        if len(value.ids) != int(self.case.output_group_count):
            raise ValueError("reshape input ciphertext-group count is incorrect")
        for ciphertext_id in value.ids:
            current_level = int(
                self.scheme.backend.GetCiphertextLevel(int(ciphertext_id))
            )
            if current_level != int(self.level):
                raise ValueError(
                    f"reshape input level {current_level} does not match {self.level}"
                )

        backend = self.scheme.backend
        output_ids: list[int] = []
        accumulation_add_count = 0
        evaluation_count = 0
        for output_group in range(int(self.case.compact_output_group_count)):
            accumulated_id: int | None = None
            for input_group in range(int(self.case.output_group_count)):
                key = f"out{output_group}_in{input_group}"
                transform_id = self.transform_ids.get(key)
                if transform_id is None:
                    continue
                partial_id = int(
                    backend.EvaluateLinearTransform(
                        int(transform_id), int(value.ids[input_group])
                    )
                )
                evaluation_count += 1
                if accumulated_id is None:
                    accumulated_id = partial_id
                else:
                    backend.AddCiphertext(int(accumulated_id), int(partial_id))
                    backend.DeleteCiphertext(int(partial_id))
                    accumulation_add_count += 1
            if accumulated_id is None:
                raise RuntimeError(f"reshape output group {output_group} has no source")
            rescaled_id = int(self.scheme.evaluator.rescale(accumulated_id, in_place=False))
            backend.DeleteCiphertext(int(accumulated_id))
            output_ids.append(rescaled_id)

        output = CipherTensor(
            self.scheme,
            output_ids,
            torch.Size([int(self.case.compact_output_group_count), int(self.case.slots)]),
        )
        output._wpc_cips_packing_signature = self.output_packing_signature
        self.last_evaluation = {
            "transform_evaluation_count": int(evaluation_count),
            "ciphertext_accumulation_add_count": int(accumulation_add_count),
            "input_ciphertext_group_count": int(self.case.output_group_count),
            "output_ciphertext_group_count": int(
                self.case.compact_output_group_count
            ),
            "input_level": int(self.level),
            "output_level": int(self.output_level),
            "clear_repack_or_encode_count": 0,
        }
        return output

    def decrypt_unpack(self, value: CipherTensor) -> np.ndarray:
        if self.scheme is None:
            raise RuntimeError("downsample reshape has no scheme")
        if getattr(value, "_wpc_cips_packing_signature", None) != self.output_packing_signature:
            raise ValueError("reshape output does not have compact CIPS packing")
        decoded = np.asarray(self.scheme.decode(self.scheme.decrypt(value)), dtype=np.float64)
        return unpack_compact_output(decoded, self.case)

    def cleanup(self) -> None:
        if self.cleaned:
            return
        backend = getattr(self.scheme, "backend", None) if self.scheme is not None else None
        if backend is not None:
            for transform_id in list(self.transform_ids.values()):
                backend.DeleteLinearTransform(int(transform_id))
        self.transform_ids = {}
        self.compiled = False
        self.cleaned = True
