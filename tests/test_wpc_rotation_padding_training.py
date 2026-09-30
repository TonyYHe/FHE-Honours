from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from orion.experimental.wpc_cips_baseline import (
    CIPSConvCase,
    rotation_padded_reference,
)
from orion.experimental.wpc_rotation_padding_training import (
    WPCRotationPaddingConv2d,
    convert_model_to_wpc_rotation_padding,
    rotation_padding_conv2d,
    rotation_padding_module_names,
)
from orion.models.unet import UNet22PlusOutput


def test_rotation_padding_conv_matches_independent_numpy_oracle() -> None:
    generator = torch.Generator().manual_seed(20260930)
    value = torch.randn((2, 3, 4, 8), generator=generator, dtype=torch.float64)
    weight = torch.randn((4, 3, 3, 3), generator=generator, dtype=torch.float64)
    bias = torch.randn((4,), generator=generator, dtype=torch.float64)

    actual = rotation_padding_conv2d(value, weight, bias, padding=1)
    case = CIPSConvCase(
        slots=128,
        input_channels=3,
        output_channels=4,
        height=4,
        width=8,
        kernel_height=3,
        kernel_width=3,
        pad_height_before=1,
        pad_width_before=1,
    )
    expected = np.stack(
        [
            rotation_padded_reference(
                sample.detach().numpy(),
                weight.detach().numpy(),
                case,
            )
            + bias.detach().numpy()[:, None, None]
            for sample in value
        ]
    )
    assert np.allclose(actual.detach().numpy(), expected, rtol=0.0, atol=1e-12)


def test_rotation_padding_is_not_zero_padding_at_boundaries() -> None:
    value = torch.arange(1, 13, dtype=torch.float64).reshape(1, 1, 3, 4)
    weight = torch.ones((1, 1, 3, 3), dtype=torch.float64)
    rotation = rotation_padding_conv2d(value, weight, padding=1)
    zero = F.conv2d(value, weight, padding=1)
    assert not torch.equal(rotation, zero)
    assert float((rotation - zero).abs().max()) > 0.0


def test_rotation_padding_conv_is_differentiable() -> None:
    layer = WPCRotationPaddingConv2d(2, 3, 3, padding=1, bias=True)
    value = torch.randn((2, 2, 4, 4), requires_grad=True)
    loss = layer(value).square().mean()
    loss.backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert layer.bias is not None
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()


def test_unet_conversion_preserves_checkpoint_keys_and_weights() -> None:
    torch.manual_seed(7)
    model = UNet22PlusOutput(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        activation="silu",
        silu_degree=7,
    )
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    conversions = convert_model_to_wpc_rotation_padding(model)
    after = model.state_dict()

    assert len(conversions) == 18
    assert len(rotation_padding_module_names(model)) == 18
    assert set(after) == set(before)
    assert all(torch.equal(after[name], before[name]) for name in before)
    assert not isinstance(model.output, WPCRotationPaddingConv2d)
    assert tuple(model.output.kernel_size) == (1, 1)


def test_conversion_rejects_unsupported_stride_or_groups() -> None:
    with pytest.raises(ValueError, match="stride-one"):
        WPCRotationPaddingConv2d.from_conv2d(
            nn.Conv2d(2, 2, 3, stride=2, padding=1)
        )
    with pytest.raises(ValueError, match="groups=1"):
        WPCRotationPaddingConv2d.from_conv2d(
            nn.Conv2d(2, 2, 3, padding=1, groups=2)
        )
