from __future__ import annotations

from pathlib import Path

import pytest

from orion.core import packing
from orion.core.region_cir_replay import (
    build_big_graph_lattigo_microbench,
    write_big_graph_lattigo_microbench,
)


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def test_r18_original_size_block_runs_through_lattigo_unified_transform(monkeypatch) -> None:
    _require_lattigo()

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("big-graph Lattigo microbench must not call pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)

    payload = build_big_graph_lattigo_microbench()

    assert payload["status"] == "ok"
    assert payload["network"] == "R18"
    assert payload["family"] == "stage1_same"
    assert payload["original_size_slot_domain"] is True
    assert payload["full_region"] is False
    assert payload["local_lattigo"] is True
    assert payload["unified_transform_group"] is True
    assert payload["uses_orion_dense_pack_conv2d"] is False
    assert payload["bank_count"] == 2
    assert payload["same_plan_certificate"] is True
    assert payload["parity"]["exact"] is True
    assert payload["parity"]["max_abs"] <= payload["parity"]["tolerance"]
    assert payload["publishable_lattigo_microbenchmark"] is True
    assert payload["publishability"]["lattigo_microbenchmark_publishable_count"] == 1


def test_big_graph_lattigo_microbench_artifact_is_written(tmp_path: Path) -> None:
    _require_lattigo()
    out = tmp_path / "orion_big_graph_lattigo_microbench.json"

    payload = write_big_graph_lattigo_microbench(out_path=out)

    assert out.exists()
    assert payload["status"] == "ok"
