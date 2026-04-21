from __future__ import annotations

import copy
import pytest
import torch
import orion
from orion.core import packing
from orion.models.unet import MiniUNet
from orion.nn.linear import ConvTranspose2d


PYTHON_BACKEND_CONFIG = {
    "ckks_params": {
        "LogN": 10,
        "LogQ": [45, 35, 35, 35, 35, 45],
        "LogP": [50],
        "LogScale": 35,
        "H": 64,
        "RingType": "Standard",
    },
    "orion": {
        "margin": 2,
        "embedding_method": "square",
        "backend": "python",
        "fuse_modules": True,
        "debug": False,
        "io_mode": "none",
    },
}


def _python_backend_config(*, embedding_method: str = "square") -> dict:
    config = copy.deepcopy(PYTHON_BACKEND_CONFIG)
    config["orion"]["embedding_method"] = str(embedding_method)
    return config


def _pad_to_fhe(matrix: torch.Tensor, fhe_shape: torch.Size) -> torch.Tensor:
    out = torch.zeros(fhe_shape, dtype=matrix.dtype)
    out[:, : matrix.shape[1], : matrix.shape[2], : matrix.shape[3]] = matrix
    return out


@pytest.mark.parametrize(
    ("seed", "spec"),
    [
        (
            0,
            {
                "in_channels": 4,
                "out_channels": 3,
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "output_padding": 0,
                "groups": 1,
                "input_shape": (1, 4, 4, 4),
                "input_gap": 2,
            },
        ),
        (
            1,
            {
                "in_channels": 4,
                "out_channels": 6,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 0,
                "groups": 2,
                "input_shape": (1, 4, 5, 7),
                "input_gap": 1,
            },
        ),
        (
            2,
            {
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 1,
                "groups": 1,
                "input_shape": (1, 4, 3, 5),
                "input_gap": 1,
            },
        ),
    ],
)
def test_conv_transpose2d_toeplitz_matches_pytorch_in_multiplexed_space(seed: int, spec: dict) -> None:
    torch.manual_seed(seed)

    layer = ConvTranspose2d(
        spec["in_channels"],
        spec["out_channels"],
        kernel_size=spec["kernel_size"],
        stride=spec["stride"],
        padding=spec["padding"],
        output_padding=spec["output_padding"],
        groups=spec["groups"],
        bias=True,
    )
    layer.eval()
    layer.init_orion_params()

    x = torch.randn(*spec["input_shape"])
    y = layer(x)

    layer.input_shape = x.shape
    layer.output_shape = y.shape
    layer.input_gap = int(spec["input_gap"])
    layer.output_gap = max(1, layer.input_gap // int(spec["stride"]))

    on_ci = (spec["in_channels"] + layer.input_gap ** 2 - 1) // (layer.input_gap ** 2)
    on_co = (spec["out_channels"] + layer.output_gap ** 2 - 1) // (layer.output_gap ** 2)
    layer.fhe_input_shape = torch.Size((1, on_ci, x.shape[2] * layer.input_gap, x.shape[3] * layer.input_gap))
    layer.fhe_output_shape = torch.Size((1, on_co, y.shape[2] * layer.output_gap, y.shape[3] * layer.output_gap))

    toeplitz = packing.construct_conv_transpose2d_toeplitz(layer)
    bias = packing.construct_conv_transpose2d_bias(layer)

    x_fhe = _pad_to_fhe(packing.multiplex(x, layer.input_gap), layer.fhe_input_shape)
    expected = _pad_to_fhe(packing.multiplex(y, layer.output_gap), layer.fhe_output_shape)
    actual = torch.tensor(toeplitz @ x_fhe.flatten().numpy(), dtype=torch.float32).reshape(layer.fhe_output_shape)
    actual = actual + bias.reshape(layer.fhe_output_shape)

    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.parametrize("embedding_method", ["square", "hybrid"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_mini_unet_runs_end_to_end_on_python_backend(embedding_method: str, batch_size: int) -> None:
    torch.manual_seed(0)

    scheme = orion.init_scheme(_python_backend_config(embedding_method=embedding_method))
    try:
        net = MiniUNet(in_channels=1, base_channels=4, out_channels=1)
        net.eval()

        x = torch.randn(batch_size, 1, 8, 8)
        clear = net(x)

        orion.fit(net, x)
        input_level = orion.compile(net)

        encrypted = orion.encrypt(orion.encode(x, input_level))
        net.he()
        fhe = net(encrypted).decrypt().decode()

        assert fhe.shape == clear.shape
        assert torch.allclose(fhe, clear, atol=1.0e-4, rtol=1.0e-4)
    finally:
        scheme.delete_scheme()


def test_python_encoder_preserves_explicit_level_zero() -> None:
    scheme = orion.init_scheme(_python_backend_config())
    try:
        encoded = orion.encode(torch.tensor([1.0]), level=0)
        assert encoded.level() == 0
    finally:
        scheme.delete_scheme()
