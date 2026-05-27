from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import orion
import orion.nn as on
from orion.core.orion import scheme


class MiniConcatUNet(on.Module):
    def __init__(self, *, in_channels: int = 1, base_dim: int = 16, out_channels: int = 1):
        super().__init__()
        hidden = int(base_dim)
        bottleneck = hidden * 2
        self.enc = on.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=True)
        self.down = on.Conv2d(hidden, bottleneck, kernel_size=3, stride=2, padding=1, bias=True)
        self.mid = on.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1, bias=True)
        self.up = on.ConvTranspose2d(bottleneck, hidden, kernel_size=2, stride=2, bias=True)
        self.cat = on.Concat(dim=1)
        self.out = on.Conv2d(hidden * 2, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        skip = self.enc(x)
        x = self.down(skip)
        x = self.mid(x)
        x = self.up(x)
        x = self.cat(x, skip)
        return self.out(x)


def _config(*, provider: bool) -> dict:
    orion_config = {
        "margin": 2,
        "embedding_method": "square",
        "backend": "lattigo",
        "fuse_modules": True,
        "debug": False,
        "io_mode": "none",
    }
    if bool(provider):
        orion_config["experimental_region_first"] = "generic_layout_dp"
    return {
        "ckks_params": {
            "LogN": 13,
            "LogQ": [45, 35, 35, 35, 35, 35, 35, 45],
            "LogP": [50],
            "LogScale": 35,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": orion_config,
    }


@pytest.mark.parametrize("provider", [False, True])
def test_lattigo_single_slot_layer_cache_mini_concat_unet_64_dim16(monkeypatch, provider: bool) -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")
    if os.environ.get("ORION_RUN_LATTIGO_MINI_UNET_SMOKE", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set ORION_RUN_LATTIGO_MINI_UNET_SMOKE=1 to run the 64x64 dim16 Lattigo smoke")

    monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    monkeypatch.setenv("ORION_SINGLE_SLOT_ENCODE_WORKERS", "2")
    monkeypatch.setenv("ORION_LATTIGO_STREAMING_LT", "0")
    monkeypatch.setenv("ORION_LATTIGO_MEMORY_BOUNDED_COMPILE", "0")
    monkeypatch.setenv("ORION_LATTIGO_MEMORY_BOUNDED_EVAL", "0")
    monkeypatch.setenv("ORION_CONCAT_FUSION", "1")

    orion.init_scheme(_config(provider=bool(provider)))
    try:
        torch.manual_seed(7)
        net = MiniConcatUNet(in_channels=1, base_dim=16, out_channels=1)
        net.eval()
        x = torch.randn(1, 1, 64, 64, dtype=torch.float32) * 0.05
        clear = net(x).detach()

        orion.fit(net, x)
        input_level = orion.compile(net)
        live_count = getattr(scheme.backend, "GetLiveLinearTransformCount", lambda: 0)
        assert int(live_count()) == 0

        encrypted = orion.encrypt(orion.encode(x, input_level))
        net.he()
        decoded = net(encrypted).decrypt().decode()

        assert int(live_count()) == 0
        assert decoded.shape == clear.shape
        assert torch.isfinite(decoded).all()
        assert float((decoded - clear).abs().max()) <= 5.0e-5
    finally:
        scheme.delete_scheme()
