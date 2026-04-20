from __future__ import annotations

import json

import pytest

from orion.core.region_experiments import (
    build_region_experiments,
    run_selected_region_backend_case,
    run_tiny_imaginary_unit_backend_case,
    run_tiny_mul_plain_backend_case,
    run_tiny_real_imag_hybrid_backend_case,
    run_tiny_unified_region_backend_case,
)


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


def test_tiny_unified_region_backend_case_runs_on_local_lattigo() -> None:
    result = run_tiny_unified_region_backend_case()

    assert result["status"] == "ok"
    assert result["output_count"] == 2
    assert result["max_abs"] <= 1.0e-4


def test_tiny_mul_plain_backend_case_runs_on_local_lattigo() -> None:
    result = run_tiny_mul_plain_backend_case()

    assert result["status"] == "ok"
    assert result["max_abs"] <= result["tolerance"]


def test_tiny_imaginary_unit_backend_case_runs_on_local_lattigo() -> None:
    result = run_tiny_imaginary_unit_backend_case()

    assert result["status"] == "ok"
    assert result["max_abs"] <= 1.0e-4
    assert result["pos_imag_error"] <= 1.0e-4
    assert result["neg_imag_error"] <= 1.0e-4


def test_tiny_real_imag_hybrid_backend_case_runs_with_conjugate_binding() -> None:
    result = run_tiny_real_imag_hybrid_backend_case()

    assert result["status"] == "ok"
    assert result["max_abs"] <= 1.0e-4
    assert result["conjugate_error"] <= 1.0e-4
    assert result["conjugate_available"] is True
    assert result["boundary_action"] == "insert_extract"


def test_selected_region_backend_cases_use_unified_transform_group() -> None:
    r20 = run_selected_region_backend_case(network="R20", region_id="stage1_two_output_region")
    r18 = run_selected_region_backend_case(network="R18", region_id="stage1_stage2_same_shape")
    r34 = run_selected_region_backend_case(network="R34", region_id="stage1_stage2_same_shape")

    assert r20["status"] == "ok"
    assert r20["backend_case"]["output_count"] == 2
    assert r20["backend_case"]["uses_real_region_masks"] is True
    assert r18["status"] == "ok"
    assert r18["backend_case"]["uses_real_region_masks"] is True
    assert r18["backend_case"]["hybrid"] is True
    assert r34["status"] == "ok"
    assert r34["backend_case"]["uses_real_region_masks"] is True
    assert r34["backend_case"]["hybrid"] is True
