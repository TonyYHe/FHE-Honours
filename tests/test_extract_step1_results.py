from __future__ import annotations

import csv
import json
from pathlib import Path

from tools import extract_step1_results as extract


def _profile(*, valid: bool = True, he_forward_s: float = 100.0) -> dict:
    major = {
        category: {"seconds": 0.0, "percent_of_he_forward": 0.0}
        for category in extract.MAJOR_CATEGORY_ORDER
    }
    major.update(
        {
            "online_encode": {"seconds": 20.0, "percent_of_he_forward": 20.0},
            "bootstrap": {"seconds": 50.0, "percent_of_he_forward": 50.0},
            "mvm_kernel": {"seconds": 10.0, "percent_of_he_forward": 10.0},
            "other_he_forward": {"seconds": 20.0, "percent_of_he_forward": 20.0},
        }
    )
    return {
        "schema_version": 2,
        "valid": valid,
        "validation_errors": [] if valid else ["clear backend is diagnostic only"],
        "runtime_fairness_mode": "single_slot_layer_cache",
        "measurement_scope": "one encrypted model forward",
        "denominator": "he_forward_s",
        "profile_count": 2,
        "measured_attempt_count": 2,
        "requested_measured_attempt_count": 2,
        "he_forward_s": he_forward_s,
        "online_encode_s": 20.0,
        "online_encode_pct_of_he_forward": 20.0,
        "major_wall_categories": major,
        "major_wall_categories_metadata": {"additive": True},
        "major_wall_categories_accounting": {
            "valid": True,
            "category_sum_s": he_forward_s,
            "closure_error_s": 0.0,
            "tolerance_s": 0.0001,
        },
        "operator_microprofile": {
            "lt_rotation": {"seconds": 12.0, "percent_of_he_forward": 12.0}
        },
        "operator_microprofile_metadata": {"additive": False},
        "operation_counts_mean_per_forward": {
            "transform_count": 7.0,
            "diag_terms": 90.0,
        },
    }


def _payload(*, clear: bool = False) -> dict:
    attempts = []
    for index, encode_s in enumerate((18.0, 22.0), start=1):
        profile = _profile(valid=not clear)
        profile["online_encode_s"] = encode_s
        profile["online_encode_pct_of_he_forward"] = encode_s
        attempts.append(
            {
                "attempt_index": index,
                "kind": "measured",
                "status": "ok",
                "timing_s": {"he_forward": 100.0},
                "step1_online_encode_profile": profile,
            }
        )
    return {
        "status": "ok",
        "network": "tiny_network",
        "model": "Tiny",
        "label": "Tiny test",
        "mode": "provider",
        "backend": "lattigo",
        "forward_runs": 2,
        "lattigo_runtime_env": {
            "ORION_LATTIGO_CLEAR_BACKEND": "1" if clear else "0",
            "ORION_SINGLE_SLOT_LAYER_CACHE": "1",
            "ORION_LATTIGO_STREAMING_LT": "0",
            "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT": "0",
        },
        "timing_s": {"compile": 3.5},
        "mae_vs_clear": {"shape_match": True, "mae": 1e-5},
        "step1_online_encode_profile": _profile(valid=not clear),
        "forward_attempts": attempts,
    }


def test_extract_record_accepts_real_fhe_and_computes_measured_variability(
    tmp_path: Path,
) -> None:
    record = extract.extract_record(tmp_path / "real.json", _payload())

    assert record["result_kind"] == "accepted_real_fhe"
    assert record["accepted_as_step1"] is True
    assert record["acceptance_errors"] == []
    assert record["measured_attempt_statistics"]["online_encode_mean_s"] == 20.0
    assert record["measured_attempt_statistics"]["online_encode_sample_std_s"] > 0.0


def test_clear_profile_is_structural_and_never_accepted(tmp_path: Path) -> None:
    record = extract.extract_record(tmp_path / "clear.json", _payload(clear=True))

    assert record["result_kind"] == "clear_structural"
    assert record["accepted_as_step1"] is False
    assert any("CLEAR_BACKEND" in error for error in record["acceptance_errors"])


def test_extractor_independently_rejects_non_closing_major_categories(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["step1_online_encode_profile"]["major_wall_categories"]["bootstrap"][
        "seconds"
    ] = 500.0

    record = extract.extract_record(tmp_path / "bad-accounting.json", payload)

    assert record["result_kind"] == "invalid_profile"
    assert any("do not close" in error for error in record["acceptance_errors"])


def test_extractor_independently_rejects_incorrect_major_percentage(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["step1_online_encode_profile"]["major_wall_categories"]["bootstrap"][
        "percent_of_he_forward"
    ] = 500.0

    record = extract.extract_record(tmp_path / "bad-percentage.json", payload)

    assert record["result_kind"] == "invalid_profile"
    assert any("percentages disagree" in error for error in record["acceptance_errors"])


def test_discovery_ignores_progress_checkpoints(tmp_path: Path) -> None:
    primary = tmp_path / "model.json"
    primary.write_text(json.dumps(_payload()), encoding="utf-8")
    (tmp_path / "model.forward0.progress_state.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )

    assert extract.discover_result_files([tmp_path]) == [primary.resolve()]


def test_write_outputs_produces_tables_and_excludes_clear_from_report(
    tmp_path: Path,
) -> None:
    records = [
        extract.extract_record(tmp_path / "real.json", _payload()),
        extract.extract_record(tmp_path / "clear.json", _payload(clear=True)),
    ]
    out_dir = tmp_path / "extracted"

    paths, plot_error = extract.write_outputs(
        records, [], out_dir, plots=False
    )

    assert plot_error is None
    assert {path.name for path in paths} == {
        "extracted_results.json",
        "runs.csv",
        "attempts.csv",
        "major_wall_categories.csv",
        "operator_microprofile.csv",
        "operation_counts.csv",
        "report.md",
    }
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Accepted real-FHE profiles: **1** of **2**" in report
    assert "clear_structural" in report
    with (out_dir / "runs.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["accepted_as_step1"] for row in rows] == ["True", "False"]
    assert rows[0]["major_closure_error_s"] == "0.0"
