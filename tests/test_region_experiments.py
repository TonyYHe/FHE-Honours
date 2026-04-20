from __future__ import annotations

import json

from orion.core.region_experiments import build_region_experiments


def test_region_experiments_include_local_lattigo_metadata() -> None:
    payload = build_region_experiments(networks=("R20",))

    assert payload["status"] == "ok"
    assert payload["backend"]["local_lattigo_exists"] is True
    assert "replace github.com/realqhc/lattigo" in payload["backend"]["replace_line"]
    assert payload["backend"]["conjugate_available"] is True
    assert payload["backend"]["shared_lt_backend_reference"] == "st/fedb4ae UnifiedLinearTransform"


def test_r20_experiments_capture_isolated_and_multi_output_input_mult() -> None:
    payload = build_region_experiments(networks=("R20",))
    rows = {row["region_id"]: row for row in payload["experiments"]}

    assert rows["stage1_isolated_conv"]["delta"]["rotations"] == 0
    assert "no useful output banks" in rows["stage1_isolated_conv"]["evidence"]
    assert rows["stage1_two_output_region"]["input_mult_factor"] == 2
    assert rows["stage1_two_output_region"]["delta"]["rotations"] < 0
    assert rows["stage1_two_output_region"]["delta"]["ct_pt_mults"] == 0


def test_r18_r34_real_imag_experiments_are_publishable_except_transition_estimate() -> None:
    payload = build_region_experiments(networks=("R18", "R34"))
    rows = {row["region_id"]: row for row in payload["experiments"]}

    assert rows["stage1_stage2_same_shape"]["real_imag_hybrid"] is True
    assert rows["stage1_stage2_same_shape"]["publishable_executor_fact"] is True
    assert rows["stage2_stage3_stage4_transition_branch_regions"]["real_imag_hybrid"] is True
    assert rows["stage2_stage3_stage4_transition_branch_regions"]["publishable_executor_fact"] is False
    assert payload["non_publishable_count"] == 1


def test_region_experiment_payload_is_json_serializable() -> None:
    payload = build_region_experiments(networks=("R20", "R18", "R34"))

    dumped = json.dumps(payload)

    assert '"publishable_summary"' in dumped
    assert payload["publishable_summary"]["delta"]["rotations"] < 0
    assert payload["publishable_summary"]["delta"]["ct_pt_mults"] < 0
