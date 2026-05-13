from __future__ import annotations

from pathlib import Path

import pytest

from orion.core import packing
from orion.core.region_cir_replay import (
    DEFAULT_END_TO_END_ARTIFACT,
    build_original_size_cir_replay,
    write_original_size_cir_replay,
)


def _require_end_to_end_artifact() -> None:
    if not Path(DEFAULT_END_TO_END_ARTIFACT).exists():
        pytest.skip(f"scripts/cir artifact is not available: {DEFAULT_END_TO_END_ARTIFACT}")


def _row(payload: dict, replay_id: str) -> dict:
    for row in payload["rows"]:
        if row["replay_id"] == replay_id:
            return row
    raise AssertionError(f"missing replay row {replay_id}")


def test_original_size_cir_replay_matches_locked_scripts_cir_stats() -> None:
    _require_end_to_end_artifact()
    payload = build_original_size_cir_replay()

    assert payload["status"] == "ok"
    assert _row(payload, "r20_stage1_two_output_region")["replayed_cir_stats"] == {
        "rotations": 39,
        "conjugations": 0,
        "ct_pt_mults": 288,
        "adds": 290,
    }
    assert _row(payload, "r18_stage1_stage2_same_shape")["replayed_cir_stats"] == {
        "rotations": 572,
        "conjugations": 152,
        "ct_pt_mults": 34920,
        "adds": 35094,
    }
    assert _row(payload, "r34_stage1_stage2_same_shape")["replayed_cir_stats"] == {
        "rotations": 1928,
        "conjugations": 124,
        "ct_pt_mults": 48508,
        "adds": 48670,
    }


def test_original_size_replay_publishable_rows_have_execution_gates() -> None:
    _require_end_to_end_artifact()
    payload = build_original_size_cir_replay()
    publishable = [row for row in payload["rows"] if row["publishable"]]

    assert len(publishable) == 3
    assert payload["gates"]["all_stats_match_expected"] is True
    assert payload["gates"]["all_publishable_rows_executor_equivalent"] is True
    assert payload["gates"]["no_publishable_count_only_rows"] is True
    for row in publishable:
        assert row["stats_match_expected"] is True
        assert row["executor_equivalent"] is True
        assert row["same_plan_certificate"] is True
        assert row["parity"]["exact"] is True
        assert row["count_only"] is False


def test_r34_transition_replay_is_non_publishable_cost_surface() -> None:
    _require_end_to_end_artifact()
    payload = build_original_size_cir_replay()
    row = _row(payload, "r34_transition_compile_surface")

    assert row["stats_match_expected"] is True
    assert row["publishable"] is False
    assert row["count_only"] is True
    assert row["executor_equivalent"] is False
    assert row["parity"]["exact"] is False


def test_original_size_replay_does_not_call_orion_dense_pack_conv2d(monkeypatch) -> None:
    _require_end_to_end_artifact()

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("original-size CIR replay must not call pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)

    payload = build_original_size_cir_replay()

    assert payload["status"] == "ok"


def test_original_size_replay_artifact_is_written(tmp_path: Path) -> None:
    _require_end_to_end_artifact()
    out = tmp_path / "orion_original_size_cir_replay.json"

    payload = write_original_size_cir_replay(out_path=out)

    assert out.exists()
    assert payload["status"] == "ok"
