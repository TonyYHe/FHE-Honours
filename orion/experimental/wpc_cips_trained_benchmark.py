"""Pure comparison helpers for the isolated trained-decoder benchmark."""

from __future__ import annotations

from typing import Any

from orion.experimental.wpc_cips_benchmark import (
    safe_ratio,
    summarize_samples,
)


TRAINED_DECODER_TRANSFORM_COUNT = 56


def compare_trained_decoder_workers(
    full: dict[str, Any],
    compressed: dict[str, Any],
    *,
    rss_samples: dict[str, dict[str, Any]],
    atol: float,
) -> dict[str, Any]:
    """Compare matched full/compressed checkpoint-decoder worker results."""

    full_times = [float(value) for value in full["measurements"]["forward_wall_s"]]
    compressed_times = [
        float(value) for value in compressed["measurements"]["forward_wall_s"]
    ]
    decompression_times = [
        float(value) for value in compressed["measurements"]["decompression_s"]
    ]
    activation_times = [
        float(value) for value in compressed["measurements"]["activation_s"]
    ]
    bootstrap_times = [
        float(value) for value in compressed["measurements"]["bootstrap_s"]
    ]
    decompression_shares = [
        100.0 * decompress / wall
        for decompress, wall in zip(decompression_times, compressed_times)
        if wall > 0.0
    ]
    activation_bootstrap_shares = [
        100.0 * (activation + bootstrap) / wall
        for activation, bootstrap, wall in zip(
            activation_times,
            bootstrap_times,
            compressed_times,
        )
        if wall > 0.0
    ]

    full_output = [float(value) for value in full["correctness"]["output_values"]]
    compressed_output = [
        float(value) for value in compressed["correctness"]["output_values"]
    ]
    output_delta = (
        max(
            (
                abs(left - right)
                for left, right in zip(full_output, compressed_output)
            ),
            default=0.0,
        )
        if len(full_output) == len(compressed_output)
        else None
    )

    full_storage = full["storage"]
    compressed_storage = compressed["storage"]
    logical_full = int(full_storage["logical_resident_total_bytes"])
    logical_compressed = int(compressed_storage["logical_resident_total_bytes"])
    full_memory = full["memory"]
    compressed_memory = compressed["memory"]
    full_rss = rss_samples["full"]
    compressed_rss = rss_samples["compressed"]
    full_pre_online = int(full_memory["pre_measured_gc"]["current_rss_bytes"])
    compressed_pre_online = int(
        compressed_memory["pre_measured_gc"]["current_rss_bytes"]
    )
    full_online_peak = full_rss.get("peak_rss_by_phase", {}).get("measured")
    compressed_online_peak = compressed_rss.get("peak_rss_by_phase", {}).get(
        "measured"
    )

    full_stats = full["backend"]["compressed_global_stats"]
    compressed_stats = compressed["backend"]["compressed_global_stats"]
    same_checkpoint = bool(
        full["checkpoint"]["sha256"] == compressed["checkpoint"]["sha256"]
    )
    same_configuration = bool(
        full.get("experiment") == compressed.get("experiment")
        and full.get("seed") == compressed.get("seed")
        and same_checkpoint
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
        "matched_checkpoint_configuration_and_seed": same_configuration,
        "full_worker_contains_no_compressed_transforms": int(
            full_stats["registered_transform_count"]
        )
        == 0,
        "compressed_worker_contains_56_compressed_transforms": int(
            compressed_stats["registered_transform_count"]
        )
        == TRAINED_DECODER_TRANSFORM_COUNT,
        "both_paths_match_the_same_clear_reference": bool(
            full["correctness"]["correct"]
            and compressed["correctness"]["correct"]
        ),
        "isolated_outputs_match_within_tolerance": bool(
            output_delta is not None and output_delta <= float(atol)
        ),
        "operation_counters_match_per_forward": operations_match,
        "trained_cheb7_and_real_bootstrap_execute_each_forward": bool(
            full["measurements"]["bootstrap_call_count"]
            == full["experiment"]["forward_runs"]
            and compressed["measurements"]["bootstrap_call_count"]
            == compressed["experiment"]["forward_runs"]
        ),
        "zero_online_python_and_weight_encode_calls": bool(
            full["measurements"]["online_python_encode_call_count"] == 0
            and compressed["measurements"]["online_python_encode_call_count"] == 0
            and full["backend"]["weight_plaintext_online_encode_calls"] == 0
            and compressed["backend"]["weight_plaintext_online_encode_calls"]
            == 0
            and compressed_stats["total_weight_plaintext_online_encode_calls"] == 0
        ),
        "both_workers_encoded_56_weight_transforms_offline": bool(
            full["backend"]["weight_plaintext_offline_encode_calls"]
            == TRAINED_DECODER_TRANSFORM_COUNT
            and compressed["backend"]["weight_plaintext_offline_encode_calls"]
            == TRAINED_DECODER_TRANSFORM_COUNT
        ),
        "compressed_materialization_released": bool(
            compressed_stats["current_materialized_full_payload_bytes"] == 0
            and compressed_stats["current_materialized_transform_count"] == 0
        ),
        "compressed_peak_is_one_transform": int(
            compressed_stats["peak_materialized_transform_count"]
        )
        == 1,
        "logical_resident_storage_is_reduced": logical_compressed < logical_full,
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
    acceptance["valid"] = bool(all(acceptance.values()))

    full_summary = summarize_samples(full_times)
    compressed_summary = summarize_samples(compressed_times)
    return {
        "latency": {
            "full_forward_s": full_summary,
            "compressed_forward_s": compressed_summary,
            "compressed_over_full_median_ratio": safe_ratio(
                compressed_summary["median"], full_summary["median"]
            ),
            "compressed_decompression_s": summarize_samples(
                decompression_times
            ),
            "compressed_decompression_pct_of_forward": summarize_samples(
                decompression_shares
            ),
            "compressed_activation_s": summarize_samples(activation_times),
            "compressed_bootstrap_s": summarize_samples(bootstrap_times),
            "compressed_activation_plus_bootstrap_pct_of_forward": (
                summarize_samples(activation_bootstrap_shares)
            ),
        },
        "memory": {
            "logical_full_resident_bytes": logical_full,
            "logical_compressed_resident_bytes": logical_compressed,
            "logical_storage_compression_ratio": safe_ratio(
                logical_full, logical_compressed
            ),
            "full_pre_online_rss_bytes": full_pre_online,
            "compressed_pre_online_rss_bytes": compressed_pre_online,
            "pre_online_rss_difference_bytes": (
                full_pre_online - compressed_pre_online
            ),
            "full_online_peak_rss_bytes": full_online_peak,
            "compressed_online_peak_rss_bytes": compressed_online_peak,
            "online_peak_rss_difference_bytes": (
                int(full_online_peak) - int(compressed_online_peak)
                if full_online_peak is not None
                and compressed_online_peak is not None
                else None
            ),
            "full_go_heap_inuse_after_compile_gc_bytes": full_memory[
                "post_compile_gc"
            ]["go"]["heap_inuse_bytes"],
            "compressed_go_heap_inuse_after_compile_gc_bytes": (
                compressed_memory["post_compile_gc"]["go"]["heap_inuse_bytes"]
            ),
            "rss_sampling": rss_samples,
            "interpretation": (
                "Logical storage counts learned Q/P transforms, bias plaintexts, "
                "metadata, and the shared full-Q/P concat permutation. RSS also "
                "includes Python, Torch, keys, ciphertexts, bootstrap state, and "
                "allocator pages."
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
    "TRAINED_DECODER_TRANSFORM_COUNT",
    "compare_trained_decoder_workers",
]
