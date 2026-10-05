from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path
import sys

import pytest

from orion.experimental.wpc_tradeoff_synthesis import (
    EvidenceValidationError,
    ACCURACY_GATES,
    DECODER_GATES,
    build_synthesis,
    classify_periodic_candidate,
    summarize_periodicity,
    summarize_step1,
    summarize_accuracy,
    summarize_decoder_correctness,
    summarize_decoder_benchmark,
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
    record = {
        "accepted_as_step1": True,
        "status": "ok", "backend": "lattigo", "clear_backend": False,
        "network": network,
        "model": model,
        "mode": mode,
        "correctness": {"shape_match": True, "mae": 1e-7},
        "measured_attempt_statistics": {
            "he_forward_mean_s": he_forward, "online_encode_mean_s": encode_pct,
            "online_encode_pct_mean": encode_pct,
            "he_forward_sample_std_s": 0.0,
            "online_encode_sample_std_s": 0.0,
            "online_encode_pct_sample_std": 0.0,
        },
        "profile": {
            "schema_version": 2,
            "valid": True,
            "profile_count": 3,
            "requested_measured_attempt_count": 3, "measured_attempt_count": 3,
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
    record["attempts"] = [{"kind": "measured", "status": "ok", "timing_s": {"he_forward": he_forward}, "profile": copy.deepcopy(record["profile"])} for _ in range(3)]
    return record


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


def _periodicity_files(tmp_path: Path, *, network: str, mode: str, periodic: bool) -> tuple[dict, Path]:
    from orion.experimental.wpc_periodicity import PeriodicityCollector
    collector = PeriodicityCollector()
    collector.record_payload(
        [1, 2, 1, 2] if periodic else [1, 2, 3, 4],
        metadata={
            "model": network, "mode": mode, "checkpoint_hash": "none:seed=0",
            "module_name": "cat1_materialize_test" if periodic else "conv_test",
            "operator_type": "UnifiedLinearTransform", "transform_id": "test",
            "block_row": 0, "block_col": 0, "diagonal_index": 0,
        },
        full_encoded_bytes=100,
        encoded_qp_verifier=lambda slots, periodicity, fmt, level: {
            "attempted": True, "passed": True, "status_code": 1,
            "full_polynomial_reconstruction_exact": True,
            "slot_period_t": periodicity.minimal_period,
            "evaluation_period_2t": 2 * periodicity.minimal_period,
        },
    )
    path = tmp_path / f"{network}.jsonl"
    return collector.flush(path), path


def _accuracy(checkpoint: Path) -> dict:
    def row(*, dice: float, iou: float, loss: float, nonfinite: int) -> dict:
        return {
            "sample_count": 2115,
            "finite_sample_count": 2115 - nonfinite,
            "nonfinite_sample_count": nonfinite,
            "fully_finite": nonfinite == 0,
            "metric_scope": "all_samples" if nonfinite == 0 else "finite_samples_only",
            "nonfinite_sample_indices": list(range(nonfinite)),
            "dice": dice,
            "iou": iou,
            "loss": loss,
        }

    return {
        "schema_version": 5,
        "status": "ok",
        "acceptance": dict.fromkeys(ACCURACY_GATES, True),
        "source_checkpoint": {"sha256": "a" * 64},
        "evaluated_checkpoint": {
            "path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_checkpoint_sha256": "a" * 64, "epoch": 5,
            "padding_semantics": "wpc_flattened_spatial_rotation_padding",
            "identity_policy": "sha256_of_exact_bytes_deserialized",
        },
        "configuration": {"eval_only": True},
        "dataset": {"name": "covid19", "validation_count": 2115, "train_count": 2048},
        "training": {"best_epoch": 5, "completed_epoch": 5, "history": [
            {"epoch": epoch, "post_epoch_training_audit": {"sample_count": 2048, "finite_sample_count": 2048, "nonfinite_sample_count": 0, "fully_finite": True}}
            for epoch in range(1, 6)]},
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
        "checkpoint_manifest": [
            {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "bytes": checkpoint.stat().st_size},
            *[{"path": str(checkpoint.parent / name), "sha256": "b" * 64, "bytes": 100}
              for name in ["rotation_padding_last.pt", *[f"rotation_padding_epoch_{epoch:04d}.pt" for epoch in range(6)]]],
        ],
    }


def _correctness(checkpoint_sha: str) -> dict:
    return {
        "schema_version": 2,
        "correctness_atol": 2e-3,
        "status": "ok",
        "acceptance": dict.fromkeys(DECODER_GATES, True),
        "checkpoint": {"sha256": checkpoint_sha},
        "full_control": {"errors_vs_clear": dict.fromkeys(("up1", "concat", "dec1a", "activation", "dec1b"), 2e-7), "operation_counters": {"rotation_total": 1200}},
        "compressed": {
            "errors_vs_clear": dict.fromkeys(("up1", "concat", "dec1a", "activation", "dec1b"), 2e-7),
            "final_max_abs_delta_vs_full_control": 0.0,
            "operation_counters": {"rotation_total": 1200}, "online_python_encode_call_count": 0,
        },
        "compressed_global_stats": {
            "registered_transform_count": 56,
            "total_weight_plaintext_online_encode_calls": 0,
            "current_materialized_full_payload_bytes": 0, "current_materialized_transform_count": 0, "peak_materialized_transform_count": 1,
        },
        "storage": {"overall_full_to_stored_including_concat_ratio": 10.2,
                    "full_weight_plus_bias_payload_bytes": 1000,
                    "stored_weight_plus_metadata_plus_bias_bytes": 80,
                    "concat_full_qp_payload_bytes": 20},
    }


def _benchmark(checkpoint_sha: str) -> dict:
    from orion.experimental.wpc_cips_trained_benchmark import WORKER_GATES, compare_trained_decoder_workers
    workers = {}
    for mode in ("full", "compressed"):
        compressed = mode == "compressed"
        per_forward = {"rotation_total": 1200, "linear_transform_rotation": 1200,
                       "direct_rotation": 0, "conjugation": 4}
        workers[mode] = {
            "status": "ok", "mode": mode, "seed": 17, "process": {"pid": 101 if compressed else 100},
            "checkpoint": {"sha256": checkpoint_sha},
            "experiment": {"forward_runs": 10, "atol": 2e-3},
            "acceptance": dict.fromkeys(WORKER_GATES, True),
            "measurements": {
                "forward_wall_s": [2.1 if compressed else 2.0] * 10,
                "decompression_s": [0.1 if compressed else 0.0] * 10,
                "transform_evaluate_s": [0.0] * 10,
                "activation_s": [0.035] * 10, "bootstrap_s": [0.7] * 10,
                "bootstrap_call_count": 10, "online_python_encode_call_count": 0,
                "operation_counters_per_forward": per_forward,
                "operation_counters_total": {key: value * 10 for key, value in per_forward.items()},
            },
            "correctness": {"correct": True, "max_abs_error": 1e-6,
                            "output_shape": [1, 3], "output_values": [1., 2., 3.]},
            "storage": {"learned_transform_count": 56, "logical_resident_total_bytes": 100 if compressed else 1000},
            "memory": {"pre_measured_gc": {"current_rss_bytes": 1500 if compressed else 2000},
                       "post_compile_gc": {"go": {"heap_inuse_bytes": 300 if compressed else 900}}},
            "backend": {"weight_plaintext_offline_encode_calls": 56, "weight_plaintext_online_encode_calls": 0,
                        "compressed_global_stats": {
                            "registered_transform_count": 56 if compressed else 0,
                            "total_weight_plaintext_online_encode_calls": 0,
                            "current_materialized_full_payload_bytes": 0, "current_materialized_transform_count": 0,
                            "peak_materialized_transform_count": 1 if compressed else 0}},
        }
    rss = {mode: {"peak_rss_by_phase": {"measured": 1600 if mode == "compressed" else 2500},
                  "sample_count_by_phase": {"measured": 5}} for mode in workers}
    comparison = compare_trained_decoder_workers(workers["full"], workers["compressed"], rss_samples=rss, atol=2e-3)
    return {"schema_version": 1, "status": "ok", "acceptance": comparison["acceptance"],
            "checkpoint": {"sha256": checkpoint_sha}, "workers": workers, "comparison": comparison}


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


@pytest.mark.parametrize("corruption", ["percentage", "category", "empty", "count", "std", "canonical"])
def test_step1_rejects_false_percentages_or_incomplete_measurements(corruption):
    payload = _step1()
    record = payload["records"][0]
    if corruption == "percentage":
        record["profile"]["online_encode_pct_of_he_forward"] = 99
        record["profile"]["major_wall_categories"]["online_encode"]["percent_of_he_forward"] = 99
    elif corruption == "category":
        record["profile"]["major_wall_categories"]["bootstrap"]["percent_of_he_forward"] = 999
    elif corruption == "empty":
        record["attempts"] = []
    elif corruption == "count":
        record["profile"]["profile_count"] = 4
    elif corruption == "std":
        record["measured_attempt_statistics"]["he_forward_sample_std_s"] = 99
    else:
        record["profile"]["he_forward_s"] = 1000
        record["profile"]["major_wall_categories_accounting"]["tolerance_s"] = 10000
    with pytest.raises(EvidenceValidationError):
        summarize_step1(payload)


@pytest.mark.parametrize("corruption", ["ratio", "median", "count", "empty", "gate", "memory", "operations", "output"])
def test_benchmark_statistics_are_recomputed_from_raw_workers(corruption):
    payload = _benchmark("a" * 64)
    comparison = payload["comparison"]
    if corruption == "ratio":
        comparison["latency"]["compressed_over_full_median_ratio"] = 2
    elif corruption in ("median", "count"):
        comparison["latency"]["compressed_forward_s"][corruption] = 99
    elif corruption == "empty":
        payload["workers"]["compressed"]["measurements"]["forward_wall_s"] = []
    elif corruption == "gate":
        del payload["acceptance"]["operation_counters_match_per_forward"]
    elif corruption == "memory":
        comparison["memory"]["logical_storage_compression_ratio"] = 999
    elif corruption == "operations":
        comparison["operations"]["compressed_per_forward"]["rotation_total"] += 1
    else:
        payload["workers"]["compressed"]["correctness"]["output_values"][0] += 1
    with pytest.raises(EvidenceValidationError):
        summarize_decoder_benchmark(payload)


@pytest.mark.parametrize("corruption", ["encode", "transforms", "error", "delta", "gate", "release"])
def test_decoder_flags_cannot_hide_inconsistent_measurements(corruption):
    payload = _correctness("a" * 64)
    if corruption == "encode":
        payload["compressed_global_stats"]["total_weight_plaintext_online_encode_calls"] = 999
    elif corruption == "transforms":
        payload["compressed_global_stats"]["registered_transform_count"] = 55
    elif corruption == "error":
        payload["compressed"]["errors_vs_clear"]["dec1b"] = 1
    elif corruption == "delta":
        payload["compressed"]["final_max_abs_delta_vs_full_control"] = 1
    elif corruption == "release":
        payload["compressed_global_stats"]["current_materialized_full_payload_bytes"] = 1
    else:
        del payload["acceptance"]["zero_online_python_encode_calls"]
    with pytest.raises(EvidenceValidationError):
        summarize_decoder_correctness(payload)


def test_accuracy_legacy_schema_requires_honest_evaluation_rerun(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = _accuracy(checkpoint)
    payload["schema_version"] = 4
    with pytest.raises(EvidenceValidationError, match="rerun eval-only"):
        summarize_accuracy(payload)


@pytest.mark.parametrize("corruption", ["source", "flag", "dice", "indices", "audit", "gate", "epoch"])
def test_accuracy_rejects_false_sample_accounting_and_provenance(tmp_path, corruption):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = _accuracy(checkpoint)
    if corruption == "source":
        payload["evaluated_checkpoint"]["source_checkpoint_sha256"] = "b" * 64
    elif corruption == "flag":
        payload["metrics"]["native_zero_padding_checkpoint"]["fully_finite"] = False
    elif corruption == "dice":
        payload["metrics"]["rotation_padding_best"]["dice"] = 2
    elif corruption == "indices":
        payload["metrics"]["rotation_padding_before_finetune"]["nonfinite_sample_indices"] = []
    elif corruption == "audit":
        payload["training"]["history"] = []
    elif corruption == "epoch":
        payload["evaluated_checkpoint"]["epoch"] = 1
    else:
        del payload["acceptance"]["evaluated_checkpoint_content_identity_recorded"]
    with pytest.raises(EvidenceValidationError):
        summarize_accuracy(payload)


def test_same_path_checkpoint_replacement_cannot_mix_stale_accuracy(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"first weights")
    accuracy = _accuracy(checkpoint)
    checkpoint.write_bytes(b"replacement weights")
    new_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows, _ = _periodicity_rows(tmp_path)
    with pytest.raises(EvidenceValidationError, match="hashes differ"):
        build_synthesis(step1=_step1(), periodicity_rows=rows, accuracy=accuracy,
                        decoder_correctness=_correctness(new_hash), decoder_benchmark=_benchmark(new_hash),
                        checkpoint_sha256=new_hash, checkpoint_path=checkpoint)


@pytest.mark.parametrize("corruption", ["coverage", "metadata", "failed_verification", "unique_model", "payload_change"])
def test_census_summary_is_reconciled_with_raw_records(tmp_path, corruption):
    summary, path = _periodicity_files(tmp_path, network="u22_64_base32", mode="provider", periodic=True)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "coverage":
        summary["occurrence"]["byte_coverage_pct"] = 99
    elif corruption == "metadata":
        records[0]["block_row"] = None
    elif corruption == "failed_verification":
        summary["encoded_qp_verification"]["failed_occurrence_count"] = 1
    elif corruption == "unique_model":
        records[0]["model"] = "wrong"
    else:
        records[1]["source_hash"] = "f" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(EvidenceValidationError):
        summarize_periodicity(summary, path, network="u22_64_base32", display_name="U-Net22",
                              expected_mode="provider", require_encoded_verification=True)


def test_checkpoint_copy_to_different_path_is_allowed_by_content_hash(tmp_path):
    source = tmp_path / "server-best.pt"
    source.write_bytes(b"weights")
    copied = tmp_path / "local-best.pt"
    copied.write_bytes(source.read_bytes())
    digest = hashlib.sha256(copied.read_bytes()).hexdigest()
    rows, _ = _periodicity_rows(tmp_path)
    result = build_synthesis(step1=_step1(), periodicity_rows=rows, accuracy=_accuracy(source),
                             decoder_correctness=_correctness(digest), decoder_benchmark=_benchmark(digest),
                             checkpoint_sha256=digest, checkpoint_path=copied)
    assert result["status"] == "ok"


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


@pytest.mark.parametrize("corruption", [None, "raw_mismatch", "raw_missing", "stale_ratio"])
def test_cli_writes_validated_tables_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str | None,
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
    benchmark = payloads["benchmark"]
    benchmark["artifacts"] = {}
    for mode, worker in benchmark["workers"].items():
        raw_path = tmp_path / f"{mode}.worker.json"
        raw_path.write_text(json.dumps(worker))
        benchmark["artifacts"][mode] = {"result": str(raw_path)}
        worker["correctness"].pop("output_values")
    if corruption == "raw_mismatch":
        benchmark["workers"]["full"]["seed"] += 1
    elif corruption == "raw_missing":
        (tmp_path / "full.worker.json").unlink()
    elif corruption == "stale_ratio":
        benchmark["comparison"]["latency"]["compressed_over_full_median_ratio"] = 2
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
    status = cli.main()
    if corruption is not None:
        assert status == 2
        assert not (out_dir / "report.md").exists()
        assert not (out_dir / "synthesis.json").exists()
        return
    assert status == 0
    result = json.loads((out_dir / "synthesis.json").read_text())
    assert result["status"] == "ok"
    assert len(result["outputs"]["plots"]) == 4
    assert all((out_dir / name).is_file() for name in result["outputs"]["plots"])
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "artifact_manifest.csv").is_file()
    assert "full_raw_worker" in result["input_artifacts"]
    assert "compressed_raw_worker" in result["input_artifacts"]
