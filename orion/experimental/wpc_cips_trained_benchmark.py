"""Pure comparison helpers for the isolated trained-decoder benchmark."""

from __future__ import annotations

import math
from typing import Any

from orion.experimental.wpc_cips_benchmark import (
    safe_ratio,
    summarize_samples,
)
from orion.experimental.wpc_evidence_validation import (
    EvidenceValidationError, finite, gates, integer, sha256,
)


TRAINED_DECODER_TRANSFORM_COUNT = 56
WORKER_GATES = {
    "valid", "correct_vs_clear", "independent_torch_oracle_matches_clear",
    "expected_transform_count", "checkpoint_activation_is_degree_seven",
    "bootstrap_uses_four_ciphertext_groups",
    "real_bootstrap_ran_for_every_measured_forward",
    "zero_online_python_encode_calls", "storage_mode_matches_request",
    "measured_forward_count_matches_request", "compressed_registry_matches_mode",
    "compressed_materialization_released",
}


def validate_worker(worker: dict[str, Any], *, mode: str, atol: float) -> None:
    """Check raw observations, not just the worker's success booleans."""
    try:
        gates(worker, name=f"{mode} worker", required=WORKER_GATES)
        if worker.get("status") != "ok" or worker.get("mode") != mode:
            raise EvidenceValidationError(f"{mode} worker status/mode mismatch")
        sha256(worker["checkpoint"]["sha256"], name=f"{mode} checkpoint")
        count = integer(worker["experiment"]["forward_runs"], name="forward runs", minimum=1)
        tolerance = finite(atol, name="correctness tolerance", minimum=0.0)
        if tolerance <= 0:
            raise EvidenceValidationError("correctness tolerance must be positive")
        if worker["experiment"]["atol"] != tolerance:
            raise EvidenceValidationError(f"{mode} worker correctness tolerance mismatch")
        measurements = worker["measurements"]
        keys = ("forward_wall_s", "decompression_s", "transform_evaluate_s", "activation_s", "bootstrap_s")
        for key in keys:
            values = measurements[key]
            if not isinstance(values, list) or len(values) != count:
                raise EvidenceValidationError(f"{mode}/{key}: expected {count} measured samples")
            for value in values:
                number = finite(value, name=f"{mode}/{key}", minimum=0.0)
                if key in ("forward_wall_s", "bootstrap_s") and number <= 0:
                    raise EvidenceValidationError(f"{mode}/{key} must be positive")
        for index, wall in enumerate(measurements["forward_wall_s"]):
            components = sum(measurements[key][index] for key in keys[1:])
            if components > wall + max(1e-6, wall * 1e-6):
                raise EvidenceValidationError(f"{mode} subtimers exceed forward at sample {index}")
        if mode == "full" and any(measurements["decompression_s"]):
            raise EvidenceValidationError("full worker reports decompression")
        shape = worker["correctness"]["output_shape"]
        values = worker["correctness"]["output_values"]
        if not isinstance(shape, list) or not shape or not isinstance(values, list):
            raise EvidenceValidationError(f"{mode} output shape/values are missing")
        size = math.prod(integer(dim, name="output dimension", minimum=1) for dim in shape)
        if len(values) != size:
            raise EvidenceValidationError(f"{mode} output length does not match shape")
        for value in values:
            finite(value, name=f"{mode} output")
        error = finite(worker["correctness"]["max_abs_error"], name=f"{mode} clear error", minimum=0.0)
        if worker["correctness"].get("correct") is not True or error > tolerance:
            raise EvidenceValidationError(f"{mode} output exceeds clear-reference tolerance")
        counters = measurements["operation_counters_per_forward"]
        for key in ("rotation_total", "linear_transform_rotation", "direct_rotation", "conjugation"):
            finite(counters[key], name=f"{mode}/{key}", minimum=0.0)
        if counters["rotation_total"] != counters["linear_transform_rotation"] + counters["direct_rotation"]:
            raise EvidenceValidationError(f"{mode} rotation counters do not reconcile")
        totals = measurements["operation_counters_total"]
        for key, value in counters.items():
            if totals[key] != value * count:
                raise EvidenceValidationError(f"{mode}/{key} operation totals do not reconcile")
        integer(worker["process"]["pid"], name="worker PID", minimum=1)
        for value in (worker["storage"]["logical_resident_total_bytes"],
                      worker["memory"]["pre_measured_gc"]["current_rss_bytes"],
                      worker["memory"]["post_compile_gc"]["go"]["heap_inuse_bytes"]):
            integer(value, name=f"{mode} memory bytes", minimum=1)
        if integer(worker["storage"]["learned_transform_count"], name="learned transforms") != TRAINED_DECODER_TRANSFORM_COUNT:
            raise EvidenceValidationError(f"{mode} worker has incorrect learned transform count")
        for key in ("bootstrap_call_count", "online_python_encode_call_count"):
            integer(measurements[key], name=f"{mode}/{key}")
        for key in ("weight_plaintext_offline_encode_calls", "weight_plaintext_online_encode_calls"):
            integer(worker["backend"][key], name=f"{mode}/{key}")
        for key, value in worker["backend"]["compressed_global_stats"].items():
            if key in ("registered_transform_count", "total_weight_plaintext_online_encode_calls",
                       "current_materialized_full_payload_bytes", "current_materialized_transform_count",
                       "peak_materialized_transform_count"):
                integer(value, name=f"{mode}/{key}")
    except (KeyError, TypeError) as exc:
        raise EvidenceValidationError(f"{mode} worker is missing required evidence: {exc}") from exc


def compare_trained_decoder_workers(
    full: dict[str, Any],
    compressed: dict[str, Any],
    *,
    rss_samples: dict[str, dict[str, Any]],
    atol: float,
) -> dict[str, Any]:
    """Compare matched full/compressed checkpoint-decoder worker results."""

    validate_worker(full, mode="full", atol=atol)
    validate_worker(compressed, mode="compressed", atol=atol)

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
        if full["correctness"]["output_shape"] == compressed["correctness"]["output_shape"]
        and len(full_output) == len(compressed_output)
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
        "raw_measurements_and_outputs_validated": True,
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
            and full_online_peak is not None and compressed_online_peak is not None
            and integer(full_online_peak, name="full measured RSS", minimum=1) > 0
            and integer(compressed_online_peak, name="compressed measured RSS", minimum=1) > 0
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
