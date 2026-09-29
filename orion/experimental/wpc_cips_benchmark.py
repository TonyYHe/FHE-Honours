"""Pure helpers for the isolated WPC CIPS resource benchmark."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


def percentile(values: Iterable[float], q: float) -> float | None:
    """Return a linearly interpolated percentile without a NumPy dependency."""

    samples = sorted(float(value) for value in values)
    if not samples:
        return None
    if not 0.0 <= float(q) <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * float(q) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return samples[lower]
    fraction = position - lower
    return samples[lower] * (1.0 - fraction) + samples[upper] * fraction


def summarize_samples(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "min": min(samples) if samples else None,
        "max": max(samples) if samples else None,
        "mean": statistics.fmean(samples) if samples else None,
        "median": statistics.median(samples) if samples else None,
        "p95": percentile(samples, 95.0),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def safe_ratio(numerator: float | int | None, denominator: float | int | None):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def compare_worker_payloads(
    full: dict[str, Any],
    compressed: dict[str, Any],
    *,
    rss_samples: dict[str, dict[str, Any]],
    atol: float,
) -> dict[str, Any]:
    """Build the matched comparison and its non-performance acceptance gates."""

    full_times = list(full["measurements"]["forward_wall_s"])
    compressed_times = list(compressed["measurements"]["forward_wall_s"])
    decompression_times = list(
        compressed["measurements"]["decompression_s"]
    )
    decompression_shares = [
        100.0 * float(decompress) / float(wall)
        for decompress, wall in zip(decompression_times, compressed_times)
        if float(wall) > 0.0
    ]
    full_summary = summarize_samples(full_times)
    compressed_summary = summarize_samples(compressed_times)
    decompression_summary = summarize_samples(decompression_times)
    decompression_share_summary = summarize_samples(decompression_shares)

    full_output = list(full["correctness"].get("output_values", []))
    compressed_output = list(
        compressed["correctness"].get("output_values", [])
    )
    if len(full_output) != len(compressed_output):
        output_delta = None
    else:
        output_delta = max(
            (abs(float(left) - float(right)) for left, right in zip(full_output, compressed_output)),
            default=0.0,
        )

    full_storage = full["storage"]
    compressed_storage = compressed["storage"]
    full_memory = full["memory"]
    compressed_memory = compressed["memory"]
    full_rss = rss_samples["full"]
    compressed_rss = rss_samples["compressed"]

    logical_full = int(full_storage["stored_weight_plus_metadata_plus_bias_bytes"])
    logical_compressed = int(
        compressed_storage["stored_weight_plus_metadata_plus_bias_bytes"]
    )
    full_pre_online = int(full_memory["pre_measured_gc"]["current_rss_bytes"])
    compressed_pre_online = int(
        compressed_memory["pre_measured_gc"]["current_rss_bytes"]
    )
    full_online_peak = full_rss.get("peak_rss_by_phase", {}).get("measured")
    compressed_online_peak = compressed_rss.get("peak_rss_by_phase", {}).get(
        "measured"
    )

    same_configuration = bool(
        full.get("experiment") == compressed.get("experiment")
        and full.get("seed") == compressed.get("seed")
    )
    operations_match = bool(
        full["measurements"]["operation_counters_per_forward"]
        == compressed["measurements"]["operation_counters_per_forward"]
    )
    acceptance = {
        "workers_completed_successfully": bool(
            full.get("status") == "ok" and compressed.get("status") == "ok"
        ),
        "workers_used_distinct_processes": int(full["process"]["pid"])
        != int(compressed["process"]["pid"]),
        "matched_configuration_and_seed": same_configuration,
        "full_worker_contains_no_compressed_transforms": bool(
            full["backend"]["compressed_global_stats"][
                "registered_transform_count"
            ]
            == 0
        ),
        "compressed_worker_contains_eight_compressed_transforms": bool(
            compressed["backend"]["compressed_global_stats"][
                "registered_transform_count"
            ]
            == 8
        ),
        "both_paths_match_clear_reference": bool(
            full["correctness"]["correct"]
            and compressed["correctness"]["correct"]
        ),
        "isolated_outputs_match_within_tolerance": bool(
            output_delta is not None and float(output_delta) <= float(atol)
        ),
        "operation_counters_match": operations_match,
        "zero_online_encode_calls": bool(
            full["measurements"]["online_python_encode_call_count"] == 0
            and compressed["measurements"]["online_python_encode_call_count"]
            == 0
            and compressed["backend"]["compressed_global_stats"][
                "total_weight_plaintext_online_encode_calls"
            ]
            == 0
        ),
        "compressed_materialization_released": bool(
            compressed["backend"]["compressed_global_stats"][
                "current_materialized_full_payload_bytes"
            ]
            == 0
            and compressed["backend"]["compressed_global_stats"][
                "current_materialized_transform_count"
            ]
            == 0
        ),
        "compressed_peak_is_one_transform": bool(
            compressed["backend"]["compressed_global_stats"][
                "peak_materialized_transform_count"
            ]
            == 1
        ),
        "logical_storage_is_reduced": logical_compressed < logical_full,
        "rss_sampling_captured_both_measured_phases": bool(
            int(full_rss.get("sample_count_by_phase", {}).get("measured", 0)) > 0
            and int(
                compressed_rss.get("sample_count_by_phase", {}).get(
                    "measured", 0
                )
            )
            > 0
        ),
    }
    acceptance["valid"] = all(acceptance.values())

    return {
        "latency": {
            "full_forward_s": full_summary,
            "compressed_forward_s": compressed_summary,
            "compressed_over_full_median_ratio": safe_ratio(
                compressed_summary["median"], full_summary["median"]
            ),
            "compressed_decompression_s": decompression_summary,
            "compressed_decompression_pct_of_forward": decompression_share_summary,
        },
        "memory": {
            "logical_full_storage_bytes": logical_full,
            "logical_compressed_storage_bytes": logical_compressed,
            "logical_storage_compression_ratio": safe_ratio(
                logical_full, logical_compressed
            ),
            "full_pre_online_rss_bytes": full_pre_online,
            "compressed_pre_online_rss_bytes": compressed_pre_online,
            "pre_online_rss_difference_bytes": full_pre_online
            - compressed_pre_online,
            "full_online_peak_rss_bytes": full_online_peak,
            "compressed_online_peak_rss_bytes": compressed_online_peak,
            "online_peak_rss_difference_bytes": (
                int(full_online_peak) - int(compressed_online_peak)
                if full_online_peak is not None and compressed_online_peak is not None
                else None
            ),
            "full_online_peak_delta_from_pre_online_bytes": (
                int(full_online_peak) - full_pre_online
                if full_online_peak is not None
                else None
            ),
            "compressed_online_peak_delta_from_pre_online_bytes": (
                int(compressed_online_peak) - compressed_pre_online
                if compressed_online_peak is not None
                else None
            ),
            "full_go_heap_inuse_after_compile_gc_bytes": full_memory[
                "post_compile_gc"
            ]["go"]["heap_inuse_bytes"],
            "compressed_go_heap_inuse_after_compile_gc_bytes": compressed_memory[
                "post_compile_gc"
            ]["go"]["heap_inuse_bytes"],
            "rss_sampling": rss_samples,
            "interpretation": (
                "RSS includes Python, Torch, keys, ciphertexts, Go allocator pages, "
                "and transform storage. Logical Q/P accounting isolates transform "
                "payloads; neither metric substitutes for the other."
            ),
        },
        "correctness": {
            "full_max_abs_error_vs_clear": full["correctness"]["max_abs_error"],
            "compressed_max_abs_error_vs_clear": compressed["correctness"][
                "max_abs_error"
            ],
            "max_abs_delta_between_isolated_outputs": output_delta,
            "atol": float(atol),
        },
        "operations": {
            "full_per_forward": full["measurements"][
                "operation_counters_per_forward"
            ],
            "compressed_per_forward": compressed["measurements"][
                "operation_counters_per_forward"
            ],
            "match": operations_match,
        },
        "acceptance": acceptance,
    }


__all__ = [
    "compare_worker_payloads",
    "percentile",
    "safe_ratio",
    "summarize_samples",
]
