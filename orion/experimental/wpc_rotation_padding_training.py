"""Differentiable WPC Rotation Padding for checkpoint fine-tuning.

WPC's CIPS convolution treats the spatial plane as one flattened cyclic
sequence.  It is therefore not equivalent to PyTorch's zero padding or to
independent circular padding on the height and width axes.  This module
provides a clear PyTorch implementation that keeps the original Conv2d
parameter names, allowing an existing checkpoint to be converted, fine-tuned,
and loaded back into Orion without a state-dict translation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import nn

from orion.nn.linear import Conv2d as OrionConv2d


def _pair(value: int | Iterable[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    values = tuple(int(item) for item in value)
    if len(values) != 2:
        raise ValueError(f"expected a two-dimensional value, got {value!r}")
    return values


def rotation_padding_conv2d(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    padding: int | tuple[int, int],
) -> torch.Tensor:
    """Evaluate a stride-one convolution with flattened cyclic padding.

    For output position ``(h,w)`` and kernel position ``(kh,kw)``, the source
    spatial index is

    ``(h*W + w + (kh-PH)*W + (kw-PW)) mod (H*W)``.

    The implementation stacks the shifted flattened feature maps and performs
    one tensor contraction.  It is fully differentiable with respect to both
    input and weights.
    """

    if value.ndim != 4:
        raise ValueError("WPC Rotation Padding expects an NCHW input")
    if weight.ndim != 4:
        raise ValueError("WPC Rotation Padding expects OIHW weights")
    batch, input_channels, height, width = (int(item) for item in value.shape)
    output_channels, weight_channels, kernel_height, kernel_width = (
        int(item) for item in weight.shape
    )
    if input_channels != weight_channels:
        raise ValueError("input and weight channel counts differ")
    pad_height, pad_width = _pair(padding)
    if pad_height != kernel_height // 2 or pad_width != kernel_width // 2:
        raise ValueError("WPC Rotation Padding requires centered same padding")
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("WPC Rotation Padding requires odd kernel dimensions")

    flattened = value.reshape(batch, input_channels, height * width)
    shifted = torch.stack(
        [
            torch.roll(
                flattened,
                shifts=-int(
                    (kernel_row - pad_height) * width
                    + kernel_column
                    - pad_width
                ),
                dims=-1,
            )
            for kernel_row in range(kernel_height)
            for kernel_column in range(kernel_width)
        ],
        dim=2,
    )
    kernel = weight.reshape(output_channels, input_channels, -1)
    output = torch.einsum("ock,nckp->nop", kernel, shifted)
    output = output.reshape(batch, output_channels, height, width)
    if bias is not None:
        output = output + bias.reshape(1, output_channels, 1, 1)
    return output


class WPCRotationPaddingConv2d(nn.Conv2d):
    """A state-dict-compatible Conv2d using WPC flattened cyclic padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        padding: int | tuple[int, int],
        bias: bool = True,
    ) -> None:
        kernel = _pair(kernel_size)
        padding_pair = _pair(padding)
        if any(size % 2 == 0 for size in kernel):
            raise ValueError("WPC Rotation Padding requires odd kernels")
        if padding_pair != (kernel[0] // 2, kernel[1] // 2):
            raise ValueError("WPC Rotation Padding requires centered same padding")
        super().__init__(
            int(in_channels),
            int(out_channels),
            kernel,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bool(bias),
        )
        self.wpc_padding = padding_pair

    @classmethod
    def from_conv2d(cls, source: Any) -> "WPCRotationPaddingConv2d":
        kernel = _pair(source.kernel_size)
        stride = _pair(source.stride)
        padding = _pair(source.padding)
        dilation = _pair(source.dilation)
        if stride != (1, 1):
            raise ValueError("WPC training conversion supports stride-one Conv2d only")
        if dilation != (1, 1):
            raise ValueError("WPC training conversion supports dilation one only")
        if int(source.groups) != 1:
            raise ValueError("WPC training conversion supports groups=1 only")
        converted = cls(
            int(source.in_channels),
            int(source.out_channels),
            kernel,
            padding=padding,
            bias=source.bias is not None,
        ).to(device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            converted.weight.copy_(source.weight)
            if source.bias is not None and converted.bias is not None:
                converted.bias.copy_(source.bias)
        converted.weight.requires_grad_(source.weight.requires_grad)
        if source.bias is not None and converted.bias is not None:
            converted.bias.requires_grad_(source.bias.requires_grad)
        converted.train(source.training)
        return converted

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return rotation_padding_conv2d(
            value,
            self.weight,
            self.bias,
            padding=self.wpc_padding,
        )


@dataclass(frozen=True)
class RotationPaddingConversion:
    module_name: str
    input_channels: int
    output_channels: int
    kernel_size: tuple[int, int]
    padding: tuple[int, int]
    weight_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def convert_model_to_wpc_rotation_padding(
    model: nn.Module,
) -> list[RotationPaddingConversion]:
    """Replace eligible spatial Conv2d modules in-place.

    Only odd, same-padded, stride-one, dilation-one, groups-one convolutions
    larger than ``1x1`` are converted.  A U-Net output head therefore remains
    an ordinary ``1x1`` convolution.  State-dict keys and tensor values are
    unchanged by the conversion.
    """

    conversions: list[RotationPaddingConversion] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, WPCRotationPaddingConv2d):
                continue
            if isinstance(child, (nn.Conv2d, OrionConv2d)):
                kernel = _pair(child.kernel_size)
                padding = _pair(child.padding)
                eligible = bool(
                    kernel != (1, 1)
                    and kernel[0] % 2 == 1
                    and kernel[1] % 2 == 1
                    and padding == (kernel[0] // 2, kernel[1] // 2)
                )
                if eligible:
                    converted = WPCRotationPaddingConv2d.from_conv2d(child)
                    setattr(parent, child_name, converted)
                    conversions.append(
                        RotationPaddingConversion(
                            module_name=name,
                            input_channels=int(child.in_channels),
                            output_channels=int(child.out_channels),
                            kernel_size=kernel,
                            padding=padding,
                            weight_shape=tuple(int(item) for item in child.weight.shape),
                        )
                    )
                    continue
            visit(child, name)

    visit(model)
    return conversions


def rotation_padding_module_names(model: nn.Module) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, WPCRotationPaddingConv2d)
    ]


__all__ = [
    "RotationPaddingConversion",
    "WPCRotationPaddingConv2d",
    "convert_model_to_wpc_rotation_padding",
    "rotation_padding_conv2d",
    "rotation_padding_module_names",
]
