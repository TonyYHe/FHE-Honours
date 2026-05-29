from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def test_lattigo_linear_transform_eval_recovers_missing_rotation_key() -> None:
    _require_lattigo()
    code = textwrap.dedent(
        """
        import torch

        import orion

        config = {
            "ckks_params": {
                "LogN": 6,
                "LogQ": [45, 35, 45],
                "LogP": [50],
                "LogScale": 35,
                "H": 64,
                "RingType": "Standard",
            },
            "orion": {
                "margin": 2,
                "embedding_method": "square",
                "backend": "lattigo",
                "fuse_modules": True,
                "debug": False,
                "io_mode": "none",
            },
        }

        scheme = orion.init_scheme(config)
        try:
            slots = int(scheme.params.get_slots())
            level = len(scheme.params.get_logq()) - 1
            transform_id = int(
                scheme.backend.GenerateLinearTransform(
                    [1],
                    [1.0] * slots,
                    level,
                    2.0,
                    "none",
                )
            )
            required_keys = list(scheme.backend.GetLinearTransformRotationKeys(transform_id))
            assert required_keys

            encoded = scheme.encode(torch.arange(slots, dtype=torch.float32) / 100.0, level)
            ciphertext = scheme.encrypt(encoded)

            scheme.backend.RemoveRotationKeys()
            output_id = int(
                scheme.backend.EvaluateLinearTransform(transform_id, int(ciphertext.ids[0]))
            )
            plaintext_id = int(scheme.backend.Decrypt(output_id))
            decoded = torch.tensor(scheme.backend.Decode(plaintext_id)[:slots])
            assert torch.isfinite(decoded).all()
        finally:
            scheme.delete_scheme()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lattigo_linear_transform_ids_are_not_reused_after_scheme_reset() -> None:
    _require_lattigo()
    code = textwrap.dedent(
        """
        import orion

        config = {
            "ckks_params": {
                "LogN": 6,
                "LogQ": [45, 35, 45],
                "LogP": [50],
                "LogScale": 35,
                "H": 64,
                "RingType": "Standard",
            },
            "orion": {
                "margin": 2,
                "embedding_method": "square",
                "backend": "lattigo",
                "fuse_modules": True,
                "debug": False,
                "io_mode": "none",
            },
        }

        def generate_transform_id(active_scheme):
            slots = int(active_scheme.params.get_slots())
            level = len(active_scheme.params.get_logq()) - 1
            return int(
                active_scheme.backend.GenerateLinearTransform(
                    [1],
                    [1.0] * slots,
                    level,
                    2.0,
                    "none",
                )
            )

        first_scheme = orion.init_scheme(config)
        try:
            first_id = generate_transform_id(first_scheme)
        finally:
            first_scheme.delete_scheme()

        second_scheme = orion.init_scheme(config)
        try:
            second_id = generate_transform_id(second_scheme)
            assert second_id != first_id, (first_id, second_id)
            assert int(second_scheme.backend.GetLiveLinearTransformCount()) == 1
        finally:
            second_scheme.delete_scheme()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lattigo_load_mode_reuses_cached_compile_plan(tmp_path) -> None:
    _require_lattigo()
    code = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path

        import torch

        import orion
        from orion.nn.linear import Conv2d

        tmp_path = Path(sys.argv[1])

        def config(io_mode):
            return {
                "ckks_params": {
                    "LogN": 6,
                    "LogQ": [45, 35, 45],
                    "LogP": [50],
                    "LogScale": 35,
                    "H": 64,
                    "RingType": "Standard",
                },
                "orion": {
                    "margin": 2,
                    "embedding_method": "square",
                    "backend": "lattigo",
                    "fuse_modules": True,
                    "debug": False,
                    "io_mode": str(io_mode),
                    "diags_path": str(tmp_path / "diags.h5"),
                    "keys_path": str(tmp_path / "keys.h5"),
                },
            }

        torch.manual_seed(17)
        weight = torch.randn(1, 1, 1, 1)
        x = torch.randn(1, 1, 2, 2)

        save_scheme = orion.init_scheme(config("save"))
        try:
            layer = Conv2d(1, 1, kernel_size=1, bias=False)
            layer.weight.data.copy_(weight)
            orion.fit(layer, x)
            save_level = orion.compile(layer)
        finally:
            save_scheme.delete_scheme()

        manifest = json.loads((tmp_path / "compile_manifest.json").read_text())
        assert manifest["bootstrap_plan"]["input_level"] == save_level

        load_scheme = orion.init_scheme(config("load"))
        try:
            layer = Conv2d(1, 1, kernel_size=1, bias=False)
            layer.weight.data.copy_(weight)
            orion.fit(layer, x)
            load_level = orion.compile(layer)
            assert load_level == save_level
            assert layer.transform_ids
        finally:
            load_scheme.delete_scheme()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
