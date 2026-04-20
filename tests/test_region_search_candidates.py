from __future__ import annotations

from orion.core.region_lowering import build_region_search_candidates


def _case(payload: dict, network: str, region_id: str) -> dict:
    for case in payload["cases"]:
        if case["network"] == network and case["region_id"] == region_id:
            return case
    raise AssertionError(f"missing case {network}:{region_id}")


def _row(case: dict, candidate: str) -> dict:
    for row in case["rows"]:
        if row["candidate"] == candidate:
            return row
    raise AssertionError(f"missing candidate {candidate}")


def test_r20_isolated_rejects_input_mult_due_to_no_useful_output_banks() -> None:
    payload = build_region_search_candidates()
    case = _case(payload, "R20", "stage1_isolated_conv")
    row = _row(case, "input_mult_output_fold")

    assert row["legal"] is False
    assert row["reject_reason"] == "no_useful_output_banks"
    assert row["count_only"] is True
    assert case["selected"] == "baseline"


def test_r20_synthetic_two_output_selects_input_mult_output_fold() -> None:
    payload = build_region_search_candidates()
    case = _case(payload, "R20", "stage1_two_output_region")
    row = _row(case, "input_mult_output_fold")

    assert case["useful_output_banks"] == 2
    assert case["selected"] == "input_mult_output_fold"
    assert row["selected"] is True
    assert row["executor_equivalent"] is True
    assert row["same_plan_certificate"] is True
    assert row["parity"]["exact"] is True
    assert row["stats_from_execution"] == row["stats_from_plan"]


def test_r18_r34_same_shape_select_hybrid_when_score_wins() -> None:
    payload = build_region_search_candidates()

    for network in ("R18", "R34"):
        case = _case(payload, network, "stage1_stage2_same_shape")
        assert case["selected"] in {"real_imag_hybrid", "real_imag_hybrid_output_fold"}
        selected = _row(case, case["selected"])
        baseline = _row(case, "baseline")
        assert selected["score"] < baseline["score"]
        assert selected["executor_equivalent"] is True
        assert selected["same_plan_certificate"] is True
        assert selected["parity"]["exact"] is True


def test_selected_rows_are_never_count_only() -> None:
    payload = build_region_search_candidates()

    selected = [row for case in payload["cases"] for row in case["rows"] if row["selected"]]

    assert selected
    assert all(row["count_only"] is False for row in selected)
    assert all(row["stats_source"] == "generated_tile_local_masks" for row in selected)
