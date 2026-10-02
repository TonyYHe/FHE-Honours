from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from orion.experimental.wpc_tradeoff_synthesis import (
    EvidenceValidationError,
    build_synthesis,
    classify_periodic_candidate,
    summarize_periodicity,
    summarize_step1,
)
from tools import synthesize_wpc_orion_tradeoff as cli


def _step1_record(network: str, model: str, mode: str, encode_pct: float) -> dict:
    he_forward = 100.0
    remaining = he_forward - encode_pct
    bootstrap = remaining * 0.6
    mvm = remaining * 0.3
    other = remaining - bootstrap - mvm
    categories = {
        "online_encode": {
            "seconds": encode_pct,
            "percent_of_he_forward": encode_pct,
        },
        "bootstrap": {
            "seconds": bootstrap,
            "percent_of_he_forward": bootstrap,
        },
        "mvm_kernel": {
            "seconds": mvm,
            "percent_of_he_forward": mvm,
        },
        "other_he_forward": {
            "seconds": other,
            "percent_of_he_forward": other,
        },
    }
    return {
        "accepted_as_step1": True,
        "network": network,
        "model": model,
        "mode": mode,
        "correctness": {"shape_match": True, "mae": 1e-7},
        "measured_attempt_statistics": {
            "he_forward_sample_std_s": 1.0,
            "online_encode_sample_std_s": 0.5,
            "online_encode_pct_sample_std": 0.25,
        },
        "profile": {
            "schema_version": 2,
            "valid": True,
            "profile_count": 3,
            "he_forward_s": he_forward,
            "online_encode_s": encode_pct,
            "online_encode_pct_of_he_forward": encode_pct,
            "major_wall_categories": categories,
            "major_wall_categories_metadata": {"additive": True},
            "major_wall_categories_accounting": {
                "valid": True,
                "tolerance_s": 1e-6,
            },
            "operator_microprofile_metadata": {"additive": False},
        },
    }


def _step1() -> dict:
    records = [
        _step1_record("resnet20_cifar10", "ResNet20", "dense", 20.0),
        _step1_record("u22_64_base32", "UNet22", "provider", 90.0),
        _step1_record("vgg16_imgnet", "VGG16-base16", "provider", 48.0),
    ]
    return {
        "schema_version": 1,
        "accepted_real_fhe_count": 3,
        "read_errors": [],
        "records": records,
    }


def _periodicity_files(
    tmp_path: Path,
    *,
    network: str,
    mode: str,
    periodic: bool,
) -> tuple[dict, Path]:
    module = "cat1_materialize_test" if periodic else "conv_test"
    base = {
        "schema_version": 2,
        "model": network,
        "mode": mode,
        "module_name": module,
        "operator_type": "UnifiedLinearTransform",
        "transform_id": f"provider:key:0:{module}",
        "wpc_slot_candidate": periodic,
        "encoded_qp_verification": (
            {"attempted": True, "passed": True} if periodic else {}
        ),
    }
    unique = {"record_type": "slot_periodicity_unique", **base}
    occurrence = {"record_type": "slot_periodicity_occurrence", **base}
    path = tmp_path / f"{network}.jsonl"
    path.write_text(
        json.dumps(unique) + "\n" + json.dumps(occurrence) + "\n",
        encoding="utf-8",
    )
    full = 100
    hybrid = 50 if periodic else 100
    aggregate = {
        "observed_count": 1,
        "nonzero_count": 1,
        "all_zero_count": 0,
        "periodic_count": 1 if periodic else 0,
        "count_coverage_pct": 100.0 if periodic else 0.0,
        "byte_coverage_pct": 50.0 if periodic else 0.0,
        "full_encoded_bytes": full,
        "hybrid_storage_bytes": hybrid,
        "partial_storage_compression_ratio": full / hybrid,
    }
    summary = {
        "schema_version": 2,
        "profile": "wpc_orion_slot_periodicity",
        "occurrence": aggregate,
        "unique_diagonal": aggregate,
        "metadata_complete_occurrence_count": 1,
        "metadata_complete_unique_count": 1,
        "logical_diagonal_payload_change_count": 0,
        "encoded_qp_verification": {
            "candidate_occurrence_count": 1 if periodic else 0,
            "attempted_occurrence_count": 1 if periodic else 0,
            "passed_occurrence_count": 1 if periodic else 0,
            "failed_occurrence_count": 0,
            "complete": periodic,
        },
    }
    return summary, path


def _accuracy(checkpoint: Path) -> dict:
    def row(*, dice: float, iou: float, loss: float, nonfinite: int) -> dict:
        return {
            "sample_count": 2115,
            "finite_sample_count": 2115 - nonfinite,
            "nonfinite_sample_count": nonfinite,
            "fully_finite": nonfinite == 0,
            "dice": dice,
            "iou": iou,
            "loss": loss,
        }

    return {
        "schema_version": 4,
        "status": "ok",
        "acceptance": {"valid": True, "all_samples": True},
        "configuration": {"eval_only": True},
        "dataset": {"name": "covid19", "validation_count": 2115},
        "metrics": {
            "native_zero_padding_checkpoint": row(
                dice=0.94, iou=0.89, loss=0.14, nonfinite=0
            ),
            "rotation_padding_before_finetune": row(
                dice=0.83, iou=0.72, loss=10.0, nonfinite=1
            ),
            "rotation_padding_best": row(
                dice=0.91, iou=0.84, loss=0.24, nonfinite=0
            ),
        },
        "outputs": {"best_checkpoint": str(checkpoint)},
    }


def _correctness(checkpoint_sha: str) -> dict:
    return {
        "schema_version": 1,
        "status": "ok",
        "acceptance": {"valid": True, "correct": True},
        "checkpoint": {"sha256": checkpoint_sha},
        "full_control": {"errors_vs_clear": {"dec1a": 1e-7, "dec1b": 2e-7}},
        "compressed": {
            "errors_vs_clear": {"dec1a": 1e-7, "dec1b": 2e-7},
            "final_max_abs_delta_vs_full_control": 0.0,
        },
        "compressed_global_stats": {
            "registered_transform_count": 56,
            "total_weight_plaintext_online_encode_calls": 0,
        },
        "storage": {"overall_full_to_stored_including_concat_ratio": 12.9},
    }


def _benchmark(checkpoint_sha: str) -> dict:
    return {
        "schema_version": 1,
        "status": "ok",
        "acceptance": {"valid": True, "matched": True},
        "checkpoint": {"sha256": checkpoint_sha},
        "workers": {"full": {"experiment": {"forward_runs": 10}}},
        "comparison": {
            "latency": {
                "full_forward_s": {"median": 2.0},
                "compressed_forward_s": {"median": 2.1},
                "compressed_over_full_median_ratio": 1.05,
                "compressed_decompression_s": {"median": 0.1},
                "compressed_decompression_pct_of_forward": {"median": 4.8},
                "compressed_bootstrap_s": {"median": 0.7},
                "compressed_activation_plus_bootstrap_pct_of_forward": {
                    "median": 35.0
                },
            },
            "memory": {
                "logical_full_resident_bytes": 1000,
                "logical_compressed_resident_bytes": 100,
                "logical_storage_compression_ratio": 10.0,
                "full_pre_online_rss_bytes": 2000,
                "compressed_pre_online_rss_bytes": 1500,
                "full_online_peak_rss_bytes": 2500,
                "compressed_online_peak_rss_bytes": 1600,
                "full_go_heap_inuse_after_compile_gc_bytes": 900,
                "compressed_go_heap_inuse_after_compile_gc_bytes": 300,
            },
            "correctness": {"max_abs_delta_between_isolated_outputs": 1e-8},
            "operations": {"match": True},
        },
    }


def _periodicity_rows(tmp_path: Path) -> tuple[list[dict], dict[str, tuple[dict, Path]]]:
    fixtures = {
        "resnet": _periodicity_files(
            tmp_path, network="resnet20_cifar10", mode="dense", periodic=False
        ),
        "unet": _periodicity_files(
            tmp_path, network="u22_64_base32", mode="provider", periodic=True
        ),
        "vgg": _periodicity_files(
            tmp_path, network="vgg16_imgnet", mode="provider", periodic=False
        ),
    }
    rows = []
    for key, network, display, mode, verify in (
        ("resnet", "resnet20_cifar10", "ResNet20", "dense", False),
        ("unet", "u22_64_base32", "U-Net22", "provider", True),
        ("vgg", "vgg16_imgnet", "VGG16", "provider", False),
    ):
        summary, path = fixtures[key]
        row, _ = summarize_periodicity(
            summary,
            path,
            network=network,
            display_name=display,
            expected_mode=mode,
            require_encoded_verification=verify,
        )
        rows.append(row)
    return rows, fixtures


def test_three_model_step1_matrix_is_validated() -> None:
    rows = summarize_step1(_step1())
    assert [row["model"] for row in rows] == ["ResNet20", "U-Net22", "VGG16"]
    assert rows[1]["online_encode_pct"] == 90.0


def test_candidate_classification_is_conservative() -> None:
    assert classify_periodic_candidate(
        {
            "module_name": "cat2_materialize_forward",
            "operator_type": "UnifiedLinearTransform",
        }
    ) == "structural"
    assert classify_periodic_candidate(
        {"module_name": "conv", "operator_type": "Conv2d"}
    ) == "learned_weight"
    assert classify_periodic_candidate(
        {"module_name": "mystery", "operator_type": "UnifiedLinearTransform"}
    ) == "unclassified"


def test_schema_one_census_is_accepted_when_no_encoded_verification_is_needed(
    tmp_path: Path,
) -> None:
    summary, path = _periodicity_files(
        tmp_path,
        network="resnet20_cifar10",
        mode="dense",
        periodic=False,
    )
    summary["schema_version"] = 1
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        record["schema_version"] = 1
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    row, _ = summarize_periodicity(
        summary,
        path,
        network="resnet20_cifar10",
        display_name="ResNet20",
        expected_mode="dense",
        require_encoded_verification=False,
    )

    assert row["periodic_count"] == 0
    with pytest.raises(EvidenceValidationError, match="requires census schema 2"):
        summarize_periodicity(
            summary,
            path,
            network="resnet20_cifar10",
            display_name="ResNet20",
            expected_mode="dense",
            require_encoded_verification=True,
        )


def test_synthesis_joins_scoped_evidence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rotation_padding_best.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows, _ = _periodicity_rows(tmp_path)
    result = build_synthesis(
        step1=_step1(),
        periodicity_rows=rows,
        accuracy=_accuracy(checkpoint),
        decoder_correctness=_correctness(checkpoint_sha),
        decoder_benchmark=_benchmark(checkpoint_sha),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_path=checkpoint,
    )
    assert result["status"] == "ok"
    assert (
        result["derived"]["selective_existing_orion_layout_hypothesis"]
        == "not_supported"
    )
    assert (
        result["derived"]["periodic_learned_weight_candidate_count_across_models"]
        == 0
    )
    assert result["trained_decoder_isolated_benchmark"][
        "latency_overhead_pct"
    ] == pytest.approx(5.0)


def test_synthesis_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rotation_padding_best.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows, _ = _periodicity_rows(tmp_path)
    with pytest.raises(EvidenceValidationError, match="hashes differ"):
        build_synthesis(
            step1=_step1(),
            periodicity_rows=rows,
            accuracy=_accuracy(checkpoint),
            decoder_correctness=_correctness(checkpoint_sha),
            decoder_benchmark=_benchmark("f" * 64),
            checkpoint_sha256=checkpoint_sha,
            checkpoint_path=checkpoint,
        )


def test_cli_writes_validated_tables_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "rotation_padding_best.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _, fixtures = _periodicity_rows(tmp_path)

    paths = {
        "step1": tmp_path / "step1.json",
        "accuracy": tmp_path / "accuracy.json",
        "correctness": tmp_path / "correctness.json",
        "benchmark": tmp_path / "benchmark.json",
    }
    payloads = {
        "step1": _step1(),
        "accuracy": _accuracy(checkpoint),
        "correctness": _correctness(checkpoint_sha),
        "benchmark": _benchmark(checkpoint_sha),
    }
    for key, path in paths.items():
        path.write_text(json.dumps(payloads[key]), encoding="utf-8")
    census_paths = {}
    for key, (summary, jsonl_path) in fixtures.items():
        summary_path = tmp_path / f"{key}.summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        census_paths[key] = (summary_path, jsonl_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthesize_wpc_orion_tradeoff.py",
            "--step1",
            str(paths["step1"]),
            "--resnet-summary",
            str(census_paths["resnet"][0]),
            "--resnet-jsonl",
            str(census_paths["resnet"][1]),
            "--unet-summary",
            str(census_paths["unet"][0]),
            "--unet-jsonl",
            str(census_paths["unet"][1]),
            "--vgg-summary",
            str(census_paths["vgg"][0]),
            "--vgg-jsonl",
            str(census_paths["vgg"][1]),
            "--full-validation",
            str(paths["accuracy"]),
            "--decoder-correctness",
            str(paths["correctness"]),
            "--decoder-benchmark",
            str(paths["benchmark"]),
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert cli.main() == 0
    result = json.loads((out_dir / "synthesis.json").read_text())
    assert result["status"] == "ok"
    assert len(result["outputs"]["plots"]) == 4
    assert all((out_dir / name).is_file() for name in result["outputs"]["plots"])
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "artifact_manifest.csv").is_file()
