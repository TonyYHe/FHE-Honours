from __future__ import annotations

from pathlib import Path

import pytest

from orion.experimental.cir.report import build_region_first_pipeline_report, write_region_first_pipeline_report
from orion.experimental.cir.selector import build_region_first_full_selector, build_region_first_full_selector_summary


def _fake_lattigo_evidence() -> dict:
    return {
        "status": "ok",
        "scope": "test evidence",
        "network": "R18",
        "family": "stage1_same",
        "region_id": "stage1_same_block0",
        "full_region": False,
        "original_size_slot_domain": True,
        "local_lattigo": True,
        "unified_transform_group": True,
        "uses_orion_dense_pack_conv2d": False,
        "bank_count": 8,
        "stats_from_execution": {"rotations": 20, "conjugations": 8, "ct_pt_mults": 1080, "adds": 1089},
        "scripts_cir_full_block_stats": {"rotations": 20, "conjugations": 8, "ct_pt_mults": 1080, "adds": 1089},
        "parity": {"exact": True, "max_abs": 0.0001, "tolerance": 0.001},
        "publishable_lattigo_microbenchmark": True,
    }


def test_vendored_selector_reports_only_r18_r34() -> None:
    payload = build_region_first_full_selector()

    assert payload["status"] == "ok"
    assert {case["network"] for case in payload["cases"]} == {"R18", "R34"}
    assert all(case["network"] != "R20" for case in payload["cases"])
    assert payload["limitations"]["excluded_synthetic_rows"] == ["R20:stage1_two_output_region"]
    assert payload["summary"]["selected_non_orion_count"] == 2
    assert payload["summary"]["all_selected_parity_ok"] is True


def test_vendored_selector_aggregate_stats_match_reference() -> None:
    summary = build_region_first_full_selector_summary(build_region_first_full_selector())

    assert summary["model_summaries"]["resnet18_tiny_imagenet"]["ours"] == {
        "rotations": 572,
        "conjugations": 152,
        "ct_pt_mults": 34920,
        "adds": 35094,
    }
    assert summary["model_summaries"]["resnet34_imagenet"]["ours"] == {
        "rotations": 1928,
        "conjugations": 124,
        "ct_pt_mults": 48508,
        "adds": 48670,
    }
    assert summary["combined"]["ours"] == {
        "rotations": 2500,
        "conjugations": 276,
        "ct_pt_mults": 83428,
        "adds": 83764,
    }
    assert summary["combined"]["orion"] == {
        "rotations": 11694,
        "conjugations": 0,
        "ct_pt_mults": 190210,
        "adds": 190210,
    }
    assert summary["combined"]["delta_ours_minus_orion"]["score"] == pytest.approx(-183448.3)
    assert summary["combined"]["reduction_percent"]["rotations"] == pytest.approx(78.62151530699504)


def test_pipeline_report_attaches_backend_evidence_and_claim_hygiene() -> None:
    payload = build_region_first_pipeline_report(attach_lattigo=False, lattigo_evidence=_fake_lattigo_evidence())

    assert payload["status"] == "ok"
    assert payload["publishable_facts"]["full_network_cost"]["publishable"] is True
    assert payload["publishable_facts"]["full_network_cost"]["models"] == ["R18", "R34"]
    assert payload["publishable_facts"]["lattigo_backend"]["publishable"] is True
    evidence = payload["publishable_facts"]["lattigo_backend"]["evidence"]
    assert evidence["original_size_slot_domain"] is True
    assert evidence["bank_count"] == 8
    assert evidence["stats_from_execution"] == {"rotations": 20, "conjugations": 8, "ct_pt_mults": 1080, "adds": 1089}
    assert evidence["scripts_cir_full_block_stats"] == {"rotations": 20, "conjugations": 8, "ct_pt_mults": 1080, "adds": 1089}
    assert payload["claim_hygiene"]["excluded_synthetic_rows"] == ["R20:stage1_two_output_region"]
    assert payload["claim_hygiene"]["r20_in_main_claim"] is False
    assert payload["claim_hygiene"]["hidden_fallback_count"] == 0
    assert payload["claim_hygiene"]["full_ckks_runtime_claimed"] is False


def test_pipeline_report_artifact_is_written(tmp_path: Path) -> None:
    out = tmp_path / "orion_region_first_pipeline_report.json"

    payload = write_region_first_pipeline_report(out_path=out, attach_lattigo=False, lattigo_evidence=_fake_lattigo_evidence())

    assert out.exists()
    assert payload["status"] == "ok"
