from __future__ import annotations

import copy

import pytest

from orion.experimental.wpc_cips_benchmark import (
    compare_worker_payloads,
    percentile,
    summarize_samples,
)


def test_percentile_and_summary_are_deterministic() -> None:
    assert percentile([], 95) is None
    assert percentile([1], 95) == 1
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1], 101)

    summary = summarize_samples([1, 2, 3])
    assert summary["count"] == 3
    assert summary["mean"] == 2
    assert summary["median"] == 2
    assert summary["stdev"] == 1


def _worker(mode: str, pid: int) -> dict:
    compressed = mode == "compressed"
    return {
        "status": "ok",
        "mode": mode,
        "seed": 7,
        "process": {"pid": pid},
        "experiment": {"shape": [1, 12, 8, 8], "forward_runs": 3},
        "measurements": {
            "forward_wall_s": [1.0, 1.1, 0.9]
            if not compressed
            else [1.2, 1.3, 1.1],
            "decompression_s": [0.0, 0.0, 0.0]
            if not compressed
            else [0.2, 0.2, 0.2],
            "operation_counters_per_forward": {
                "rotation_total": 142,
                "linear_transform_rotation": 142,
                "direct_rotation": 0,
                "conjugation": 0,
            },
            "online_python_encode_call_count": 0,
        },
        "correctness": {
            "correct": True,
            "max_abs_error": 1e-8,
            "output_values": [1.0, 2.0, 3.0],
        },
        "storage": {
            "stored_weight_plus_metadata_plus_bias_bytes": (
                100 if compressed else 1000
            )
        },
        "memory": {
            "pre_measured_gc": {"current_rss_bytes": 500 if compressed else 900},
            "post_compile_gc": {
                "go": {"heap_inuse_bytes": 200 if compressed else 800}
            },
        },
        "backend": {
            "compressed_global_stats": {
                "registered_transform_count": 8 if compressed else 0,
                "total_weight_plaintext_online_encode_calls": 0,
                "current_materialized_full_payload_bytes": 0,
                "current_materialized_transform_count": 0,
                "peak_materialized_transform_count": 1 if compressed else 0,
            }
        },
    }


def test_matched_worker_comparison_closes_all_acceptance_gates() -> None:
    full = _worker("full", 100)
    compressed = _worker("compressed", 101)
    original = copy.deepcopy(compressed["correctness"]["output_values"])
    rss = {
        "full": {
            "peak_rss_by_phase": {"measured": 950},
            "sample_count_by_phase": {"measured": 4},
        },
        "compressed": {
            "peak_rss_by_phase": {"measured": 600},
            "sample_count_by_phase": {"measured": 5},
        },
    }

    result = compare_worker_payloads(
        full,
        compressed,
        rss_samples=rss,
        atol=1e-6,
    )

    assert result["acceptance"]["valid"] is True
    assert result["memory"]["logical_storage_compression_ratio"] == 10
    assert result["memory"]["online_peak_rss_difference_bytes"] == 350
    assert result["latency"]["compressed_over_full_median_ratio"] == 1.2
    assert result["latency"]["compressed_decompression_pct_of_forward"][
        "median"
    ] == pytest.approx(100 * 0.2 / 1.2)
    assert compressed["correctness"]["output_values"] == original
