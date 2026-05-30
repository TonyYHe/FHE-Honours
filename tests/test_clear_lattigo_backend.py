from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from orion.core.orion import Scheme


ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "orion" / "backend" / "clear_lattigo" / "clear_lattigo-linux.so"


def _build_clear_lattigo() -> None:
    if LIB_PATH.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_clear_lattigo.py"),
            "--output",
            str(LIB_PATH),
        ],
        cwd=str(ROOT),
        check=True,
    )


@pytest.fixture(scope="module", autouse=True)
def _clear_lattigo_library() -> None:
    _build_clear_lattigo()


def _config(backend: str = "clear_lattigo") -> dict:
    return {
        "ckks_params": {
            "logn": 4,
            "logq": [45, 30, 30],
            "logp": [31],
            "logscale": 30,
        },
        "orion": {
            "backend": backend,
            "embedding_method": "hybrid",
            "io_mode": "none",
        },
    }


def _scheme(backend: str = "clear_lattigo") -> Scheme:
    scheme = Scheme()
    scheme.init_scheme(_config(backend=backend))
    return scheme


def test_clear_lattigo_backend_explicit_encode_encrypt_lt_direction() -> None:
    scheme = _scheme()
    try:
        x = torch.arange(8, dtype=torch.float32)
        ct = scheme.encrypt(scheme.encode(x, level=2, scale=1 << 30))
        transform_id = scheme.backend.GenerateLinearTransform(
            [1],
            [1.0] * 8,
            2,
            1.0,
            "none",
        )

        out_pt = scheme.decrypt(
            type(ct)(scheme, scheme.backend.EvaluateLinearTransform(transform_id, ct.ids[0]), x.shape, x.shape)
        )
        out = scheme.decode(out_pt)

        assert torch.allclose(out, torch.roll(x, shifts=-1))
        assert scheme.backend.GetCiphertextLevel(ct.ids[0]) == 2
    finally:
        scheme.delete_scheme()


def test_clear_lattigo_unified_and_source_sum_apis() -> None:
    scheme = _scheme()
    try:
        x = torch.arange(8, dtype=torch.float32)
        ct = scheme.encrypt(scheme.encode(x, level=2, scale=1 << 30))
        idx0 = (ctypes.c_int * 1)(0)
        idx1 = (ctypes.c_int * 1)(1)
        idxs = (ctypes.POINTER(ctypes.c_int) * 2)(idx0, idx1)
        idx_lens = (ctypes.c_int * 2)(1, 1)
        data0 = (ctypes.c_float * 8)(*([2.0] * 8))
        data1 = (ctypes.c_float * 8)(*([1.0] * 8))
        data = (ctypes.POINTER(ctypes.c_float) * 2)(data0, data1)
        data_lens = (ctypes.c_int * 2)(8, 8)
        levels = (ctypes.c_int * 2)(2, 2)
        ids = scheme.backend.GenerateLinearTransformsUnified(
            2,
            idxs,
            idx_lens,
            data,
            data_lens,
            levels,
        )
        ids_array = (ctypes.c_int * len(ids))(*ids)
        outputs = scheme.backend.EvaluateLinearTransformsWithSharedCache(ids_array, len(ids), ct.ids[0])
        decoded = [
            scheme.decode(scheme.decrypt(type(ct)(scheme, output_id, x.shape, x.shape)))
            for output_id in outputs
        ]
        assert torch.allclose(decoded[0], x * 2.0)
        assert torch.allclose(decoded[1], torch.roll(x, shifts=-1))

        source_array = (ctypes.c_int * 1)(ct.ids[0])
        transform_array = (ctypes.c_int * 2)(ids[0], ids[1])
        target_array = (ctypes.c_int * 2)(0, 0)
        offsets_array = (ctypes.c_int * 2)(0, 2)
        summed = scheme.backend.EvaluateLinearTransformSourcesWithSharedCacheAdd(
            source_array,
            1,
            transform_array,
            target_array,
            offsets_array,
            2,
            1,
        )
        summed_out = scheme.decode(scheme.decrypt(type(ct)(scheme, summed[0], x.shape, x.shape)))
        assert torch.allclose(summed_out, x * 2.0 + torch.roll(x, shifts=-1))
    finally:
        scheme.delete_scheme()


def test_clear_lattigo_polynomial_and_bootstrap_metadata() -> None:
    scheme = _scheme()
    try:
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 0.5, -0.5])
        ct = scheme.encrypt(scheme.encode(x, level=1, scale=1 << 30))
        poly = scheme.backend.GenerateMonomial([1.0, 0.0, 2.0])
        out_id = scheme.backend.EvaluatePolynomial(ct.ids[0], poly, 1 << 30)
        out_ct = type(ct)(scheme, out_id, x.shape, x.shape)
        out = scheme.decode(scheme.decrypt(out_ct))
        assert torch.allclose(out, x * x + 2.0)

        boot_id = scheme.backend.Bootstrap(out_id, 8)
        assert scheme.backend.GetCiphertextLevel(boot_id) == scheme.params.get_max_level()
    finally:
        scheme.delete_scheme()


def test_lattigo_env_switch_loads_clear_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_LATTIGO_CLEAR_BACKEND", "1")
    scheme = _scheme(backend="lattigo")
    try:
        values = torch.arange(8, dtype=torch.float32)
        decoded = scheme.decode(scheme.decrypt(scheme.encrypt(scheme.encode(values))))
        assert torch.allclose(decoded, values)
    finally:
        scheme.delete_scheme()
        os.environ.pop("ORION_LATTIGO_CLEAR_BACKEND", None)
