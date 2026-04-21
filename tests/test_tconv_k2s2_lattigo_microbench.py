from __future__ import annotations

from pathlib import Path

import pytest

from orion.core import packing


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def test_tconv_k2s2_lattigo_microbench_parity() -> None:
    _require_lattigo()
    from orion.core.region_cir_replay import build_tconv_k2s2_lattigo_microbench

    payload = build_tconv_k2s2_lattigo_microbench()

    assert payload["status"] == "ok"
    assert payload["local_lattigo"] is True
    assert payload["unified_transform_group"] is True
    assert payload["pair_count"] == 2
    assert payload["input_phase_count"] == 4
    assert payload["source_pair_count"] == 2
    assert payload["output_bank_channels"] == 8
    assert payload["output_bank_count"] == 4
    assert payload["compact_source_rotation_count"] == 3
    assert payload["expansion_rotation_count"] == 2
    assert payload["mix_rotation_count"] == 15
    assert payload["rotation_count"] == 18
    assert payload["parity"]["exact"] is True
    assert payload["parity"]["max_abs"] <= payload["parity"]["tolerance"]


def test_tconv_k2s2_does_not_call_pack_conv2d(monkeypatch) -> None:
    _require_lattigo()
    from orion.core.region_cir_replay import build_tconv_k2s2_lattigo_microbench

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("tconv k2s2 microbench must not call pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
    payload = build_tconv_k2s2_lattigo_microbench()
    assert payload["status"] == "ok"
