from __future__ import annotations

from pathlib import Path

import pytest

from orion.core import packing
from orion.core.region_cir_replay import build_r18_stage2_lattigo_microbench


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def test_r18_stage2_lattigo_microbench_matches_scripts_cir_stats(monkeypatch) -> None:
    _require_lattigo()

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("R18 stage2 Lattigo microbench must not call pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)

    payload = build_r18_stage2_lattigo_microbench()

    assert payload["status"] == "ok"
    assert payload["network"] == "R18"
    assert payload["family"] == "stage2_same"
    assert payload["full_region"] is True
    assert payload["stats_from_execution"] == {
        "rotations": 84,
        "conjugations": 8,
        "ct_pt_mults": 5880,
        "adds": 5890,
    }
    assert payload["stats_match_scripts_cir"] is True
    assert payload["same_plan_certificate"] is True
    assert payload["parity"]["exact"] is True
    assert payload["parity"]["max_abs"] <= payload["parity"]["tolerance"]
