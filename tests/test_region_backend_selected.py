from __future__ import annotations

from orion.core import packing
from orion.core.region_experiments import (
    build_selected_region_backend_proof,
    run_selected_region_backend_case,
)


def _assert_selected_backend_case(row: dict) -> None:
    assert row["status"] == "ok"
    assert row["executor_equivalent"] is True
    assert row["same_plan_certificate"] is True
    assert row["parity"]["exact"] is True
    assert row["stats_from_execution"] == row["stats_from_plan"]
    assert row["backend_case"]["unified_transform_group"] is True
    assert row["backend_case"]["uses_real_region_masks"] is True
    assert row["backend_case"]["uses_orion_dense_pack_conv2d"] is False
    assert row["backend_case"]["materializer"] == "tile_local_region_lowering"


def test_selected_region_backends_execute_generated_masks_through_unified_transform_group(monkeypatch) -> None:
    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("selected region backend proof must not call pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)

    cases = [
        run_selected_region_backend_case(network="R20", region_id="stage1_two_output_region"),
        run_selected_region_backend_case(network="R18", region_id="stage1_stage2_same_shape"),
        run_selected_region_backend_case(network="R34", region_id="stage1_stage2_same_shape"),
    ]

    for row in cases:
        _assert_selected_backend_case(row)

    assert cases[0]["backend_case"]["hybrid"] is False
    assert cases[0]["backend_case"]["source_packing"]["kind"] == "input_replication"
    assert cases[1]["backend_case"]["hybrid"] is True
    assert cases[2]["backend_case"]["hybrid"] is True


def test_selected_region_backend_proof_payload_is_publishable() -> None:
    proof = build_selected_region_backend_proof()

    assert proof["status"] == "ok"
    assert "UnifiedTransformGroup" in proof["scope"]
    assert len(proof["cases"]) == 3
    for row in proof["cases"]:
        _assert_selected_backend_case(row)
