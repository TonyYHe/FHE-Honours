from __future__ import annotations

import pytest

from orion.experimental.cir.runtime_group import (
    RegionFirstRuntimeGroup,
    build_r18_tiny_region_first_e2e_report,
    discover_r18_tiny_region_groups,
)


def test_r18_tiny_region_discovery_finds_four_runtime_groups() -> None:
    groups, audit = discover_r18_tiny_region_groups()

    assert len(groups) == 4
    assert all(isinstance(group, RegionFirstRuntimeGroup) for group in groups)
    assert {group.stage for group in groups} == {"stage1", "stage2", "stage3", "stage4"}
    assert audit["selected_region_count"] == 4
    assert all(group.depth > 0 for group in groups)
    assert all(group.conv_nodes for group in groups)


def test_r18_tiny_region_first_e2e_report_records_dense_vs_region_first_costs() -> None:
    payload = build_r18_tiny_region_first_e2e_report()

    assert payload["status"] == "partial"
    assert payload["network"] == "R18"
    assert payload["dataset"] == "tiny"
    assert payload["region_first"]["runtime_group_count"] == 4
    assert payload["dense"]["stats"] == {
        "rotations": 11454,
        "conjugations": 0,
        "ct_pt_mults": 177121,
        "adds": 177121,
    }
    assert payload["region_first"]["stats"] == {
        "rotations": 3996,
        "conjugations": 161,
        "ct_pt_mults": 108007,
        "adds": 108193,
    }
    assert payload["comparison"]["cost_score_speedup"] > 1.0
    assert payload["comparison"]["delta_region_first_minus_dense"]["score"] == pytest.approx(-130936.4)


def test_r18_tiny_region_first_e2e_report_has_claim_hygiene_and_depth_audit() -> None:
    payload = build_r18_tiny_region_first_e2e_report()

    assert payload["claim"]["selected_regions_use_region_first_runtime_group"] is True
    assert payload["claim"]["full_network_ckks"] is False
    assert payload["claim"]["full_runtime_publishable"] is False
    assert payload["fallback_audit"]["selected_region_hidden_fallback_count"] == 0
    assert payload["fallback_audit"]["selected_executable_regions_no_dense_pack_conv2d"] is True
    assert payload["fallback_audit"]["executable_region_count"] == 1
    assert payload["fallback_audit"]["fallback_count"] == 12
    assert payload["bootstrap_audit"]["status"] == "depths_declared_for_solver"
    assert payload["bootstrap_audit"]["region_depths"]["stage1"]["depth"] == 2
    assert payload["bootstrap_audit"]["region_depths"]["stage4"]["depth"] == 3
    for group in payload["region_first"]["groups"]:
        assert "insert_extract_before_relu_or_add" in group["boundary_actions"]
    assert payload["region_first"]["groups"][0]["executor_attached"] is True
