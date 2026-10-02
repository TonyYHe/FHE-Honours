from __future__ import annotations

import copy

import pytest

from orion.experimental.wpc_cips_trained_benchmark import (
    TRAINED_DECODER_TRANSFORM_COUNT,
    compare_trained_decoder_workers,
)


def _worker(mode: str, pid: int) -> dict:
    compressed = mode == "compressed"
    return {
        "status": "ok",
        "mode": mode,
        "seed": 17,
        "process": {"pid": pid},
        "checkpoint": {"sha256": "a" * 64},
        "experiment": {
            "graph": "trained-decoder",
            "forward_runs": 3,
        },
        "measurements": {
            "forward_wall_s": (
                [1.0, 1.1, 0.9] if not compressed else [1.2, 1.3, 1.1]
            ),
            "decompression_s": (
                [0.0, 0.0, 0.0] if not compressed else [0.2, 0.2, 0.2]
            ),
            "activation_s": [0.1, 0.1, 0.1],
            "bootstrap_s": [0.5, 0.5, 0.5],
            "bootstrap_call_count": 3,
            "operation_counters_per_forward": {
                "rotation_total": 1200,
                "linear_transform_rotation": 1200,
                "direct_rotation": 0,
                "conjugation": 0,
            },
            "online_python_encode_call_count": 0,
        },
        "correctness": {
            "correct": True,
            "max_abs_error": 1.0e-6,
            "output_values": [1.0, 2.0, 3.0],
        },
        "storage": {
            "learned_transform_count": TRAINED_DECODER_TRANSFORM_COUNT,
            "logical_resident_total_bytes": 200 if compressed else 2000,
        },
        "memory": {
            "pre_measured_gc": {
                "current_rss_bytes": 600 if compressed else 1000
            },
            "post_compile_gc": {
                "go": {"heap_inuse_bytes": 300 if compressed else 900}
            },
        },
        "backend": {
            "weight_plaintext_offline_encode_calls": (
                TRAINED_DECODER_TRANSFORM_COUNT
            ),
            "weight_plaintext_online_encode_calls": 0,
            "compressed_global_stats": {
                "registered_transform_count": (
                    TRAINED_DECODER_TRANSFORM_COUNT if compressed else 0
                ),
                "total_weight_plaintext_online_encode_calls": 0,
                "current_materialized_full_payload_bytes": 0,
                "current_materialized_transform_count": 0,
                "peak_materialized_transform_count": 1 if compressed else 0,
            }
        },
    }


def _rss() -> dict:
    return {
        "full": {
            "peak_rss_by_phase": {"measured": 1100},
            "sample_count_by_phase": {"measured": 5},
        },
        "compressed": {
            "peak_rss_by_phase": {"measured": 700},
            "sample_count_by_phase": {"measured": 6},
        },
    }


def test_trained_decoder_comparison_closes_acceptance_gates() -> None:
    full = _worker("full", 100)
    compressed = _worker("compressed", 101)
    original_output = copy.deepcopy(compressed["correctness"]["output_values"])

    result = compare_trained_decoder_workers(
        full,
        compressed,
        rss_samples=_rss(),
        atol=2.0e-3,
    )

    assert result["acceptance"]["valid"] is True
    assert result["memory"]["logical_storage_compression_ratio"] == 10.0
    assert result["memory"]["online_peak_rss_difference_bytes"] == 400
    assert result["latency"]["compressed_over_full_median_ratio"] == 1.2
    assert result["latency"]["compressed_decompression_pct_of_forward"][
        "median"
    ] == pytest.approx(100.0 * 0.2 / 1.2)
    assert result["operations"]["match"] is True
    assert compressed["correctness"]["output_values"] == original_output


def test_trained_decoder_comparison_rejects_mismatched_checkpoint() -> None:
    full = _worker("full", 100)
    compressed = _worker("compressed", 101)
    compressed["checkpoint"]["sha256"] = "b" * 64

    result = compare_trained_decoder_workers(
        full,
        compressed,
        rss_samples=_rss(),
        atol=2.0e-3,
    )

    assert (
        result["acceptance"]["matched_checkpoint_configuration_and_seed"]
        is False
    )
    assert result["acceptance"]["valid"] is False


def test_trained_decoder_comparison_rejects_missing_measured_rss() -> None:
    rss = _rss()
    rss["compressed"]["sample_count_by_phase"] = {}

    result = compare_trained_decoder_workers(
        _worker("full", 100),
        _worker("compressed", 101),
        rss_samples=rss,
        atol=2.0e-3,
    )

    assert (
        result["acceptance"]["rss_sampling_captured_both_measured_phases"]
        is False
    )
    assert result["acceptance"]["valid"] is False
