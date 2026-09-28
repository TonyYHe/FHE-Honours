#!/usr/bin/env python3
"""Run a two-layer Orion Conv2d pipeline through the WPC CIPS planner.

The run checks actual layer installation and forward dispatch, CIPS ciphertext
handoff between layers, bias addition, level consumption, exact full-vs-
compressed Q/P behavior, and aggregate sequential-materialization accounting.
Timings are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import orion
from orion.experimental.wpc_cips_layer import WPC_GLOBAL_STATS_FIELDS
from orion.nn import Conv2d


DEFAULT_OUT = (
    REPO_ROOT
    / ".tmp/results/honours/14_wpc_cnn_layer_pipeline/"
    "two_conv_cips_compressed_qp.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate two chained Orion Conv2d layers using the opt-in WPC "
            "CIPS compressed-Q/P planner."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--atol", type=float, default=2e-6)
    return parser


def _config(logn: int) -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [55, 45, 45],
            "LogP": [60],
            "LogScale": 45,
            "H": 192,
            "RingType": "Standard",
        },
        "orion": {
            "backend": "lattigo",
            "embedding_method": "square",
            "io_mode": "none",
            "debug": False,
        },
    }


def _operation_counters(backend: Any) -> dict[str, int | None]:
    values = [int(value) for value in backend.GetOperationCounters()]
    return {
        "rotation_total": values[0] if len(values) > 0 else None,
        "linear_transform_rotation": values[1] if len(values) > 1 else None,
        "direct_rotation": values[2] if len(values) > 2 else None,
        "conjugation": values[3] if len(values) > 3 else None,
    }


def _make_layer(
    *,
    name: str,
    channels: int,
    level: int,
    bsgs_ratio: float,
    weight: np.ndarray,
    bias: np.ndarray,
) -> Conv2d:
    layer = Conv2d(
        int(channels),
        int(channels),
        3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        bsgs_ratio=float(bsgs_ratio),
        level=int(level),
    )
    layer.name = str(name)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    layer.init_orion_params()
    return layer


def _global_stats(backend: Any) -> dict[str, Any]:
    values = [int(value) for value in backend.GetWPCCompressedGlobalStats()]
    if len(values) != len(WPC_GLOBAL_STATS_FIELDS):
        raise RuntimeError("unexpected WPC global-statistics length")
    result: dict[str, Any] = {
        name: int(value) for name, value in zip(WPC_GLOBAL_STATS_FIELDS, values)
    }
    full_bytes = int(result["aggregate_full_payload_bytes"])
    compressed_bytes = int(result["aggregate_compressed_payload_bytes"])
    stored_bytes = int(result["aggregate_stored_payload_plus_metadata_bytes"])
    peak_bytes = int(result["peak_materialized_full_payload_bytes"])
    result["aggregate_payload_compression_ratio"] = (
        float(full_bytes / compressed_bytes) if compressed_bytes else None
    )
    result["aggregate_storage_compression_ratio_including_metadata"] = (
        float(full_bytes / stored_bytes) if stored_bytes else None
    )
    result["aggregate_full_to_sequential_peak_ratio"] = (
        float(full_bytes / peak_bytes) if peak_bytes else None
    )
    return result


def main() -> int:
    args = _parser().parse_args()
    slots = 1 << (int(args.logn) - 1)
    channel_capacity = int(slots // (int(args.height) * int(args.width)))
    if int(args.channels) <= channel_capacity:
        raise SystemExit(
            f"channels must exceed one-ciphertext capacity {channel_capacity}"
        )

    rng = np.random.default_rng(int(args.seed))
    input_values = rng.normal(
        0.0,
        0.4,
        size=(1, int(args.channels), int(args.height), int(args.width)),
    ).astype(np.float64)
    weights = [
        rng.normal(
            0.0,
            0.08,
            size=(int(args.channels), int(args.channels), 3, 3),
        ).astype(np.float64)
        for _ in range(2)
    ]
    biases = [
        rng.normal(0.0, 0.03, size=(int(args.channels),)).astype(np.float64)
        for _ in range(2)
    ]

    scheme = orion.init_scheme(_config(int(args.logn)))
    Conv2d.set_scheme(scheme)
    layers = [
        _make_layer(
            name="wpc_block_conv1",
            channels=int(args.channels),
            level=2,
            bsgs_ratio=float(args.bsgs_ratio),
            weight=weights[0],
            bias=biases[0],
        ),
        _make_layer(
            name="wpc_block_conv2",
            channels=int(args.channels),
            level=1,
            bsgs_ratio=float(args.bsgs_ratio),
            weight=weights[1],
            bias=biases[1],
        ),
    ]
    plans = []
    input_ciphertext = None
    full_first = None
    full_second = None
    compressed_first = None
    compressed_second = None
    try:
        compile_started = time.perf_counter()
        for layer in layers:
            plan = layer.install_wpc_cips_plan(
                (1, int(args.channels), int(args.height), int(args.width)),
                include_full_control=True,
                verify_exact_qp=True,
            )
            layer.he()
            plans.append(plan)
        compile_s = float(time.perf_counter() - compile_started)

        reference_first = plans[0].clear_reference(input_values)
        reference_second = plans[1].clear_reference(reference_first[None, ...])
        signatures_chain = bool(
            plans[0].output_packing_signature == plans[1].input_packing_signature
        )
        if not signatures_chain:
            raise RuntimeError("layer CIPS packing signatures do not chain")

        input_ciphertext = plans[0].encrypt_input(input_values)

        online_encode_call_count = 0
        original_encode = scheme.encoder.encode

        def counted_online_encode(*encode_args, **encode_kwargs):
            nonlocal online_encode_call_count
            online_encode_call_count += 1
            return original_encode(*encode_args, **encode_kwargs)

        scheme.encoder.encode = counted_online_encode
        try:
            scheme.backend.ResetOperationCounters()
            full_started = time.perf_counter()
            full_first = plans[0].evaluate(input_ciphertext, compressed=False)
            full_first_row = dict(plans[0].last_evaluation)
            full_second = plans[1].evaluate(full_first, compressed=False)
            full_second_row = dict(plans[1].last_evaluation)
            full_s = float(time.perf_counter() - full_started)
            full_counters = _operation_counters(scheme.backend)
            full_output = plans[1].decrypt_unpack(full_second)

            scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
            global_before_compressed = _global_stats(scheme.backend)
            scheme.backend.ResetOperationCounters()
            compressed_started = time.perf_counter()
            # Exercise the installed Orion Conv2d.forward dispatch, rather
            # than calling the plan directly.
            compressed_first = layers[0](input_ciphertext)
            compressed_first_row = dict(plans[0].last_evaluation)
            compressed_second = layers[1](compressed_first)
            compressed_second_row = dict(plans[1].last_evaluation)
            compressed_s = float(time.perf_counter() - compressed_started)
            compressed_counters = _operation_counters(scheme.backend)
            compressed_output = plans[1].decrypt_unpack(compressed_second)
        finally:
            scheme.encoder.encode = original_encode

        global_after_compressed = _global_stats(scheme.backend)
        compressed_stats = {
            plan.layer_name: plan.compressed_stats() for plan in plans
        }
        storage_by_layer = {
            plan.layer_name: plan.storage_summary() for plan in plans
        }
        full_layer_storage = int(
            sum(
                row["full_weight_plus_bias_payload_bytes"]
                for row in storage_by_layer.values()
            )
        )
        stored_layer_storage = int(
            sum(
                row["stored_weight_plus_metadata_plus_bias_bytes"]
                for row in storage_by_layer.values()
            )
        )
        layer_storage = {
            "full_weight_plus_bias_payload_bytes": full_layer_storage,
            "stored_weight_plus_metadata_plus_bias_bytes": stored_layer_storage,
            "compression_ratio_including_uncompressed_bias": float(
                full_layer_storage / stored_layer_storage
            ),
            "by_layer": storage_by_layer,
        }

        full_error = np.abs(full_output - reference_second)
        compressed_error = np.abs(compressed_output - reference_second)
        path_delta = np.abs(compressed_output - full_output)
        all_transform_rows = [
            row
            for plan in plans
            for row in plan.transform_rows.values()
        ]
        all_runtime_stats = [
            row
            for layer_rows in compressed_stats.values()
            for row in layer_rows.values()
        ]
        sequence = [
            *compressed_first_row["evaluation_sequence"],
            *compressed_second_row["evaluation_sequence"],
        ]
        exact_qp = bool(
            len(all_transform_rows) == 8
            and all(row["exact_qp_match"] is True for row in all_transform_rows)
        )
        all_released = bool(
            len(all_runtime_stats) == 8
            and all(
                int(row["evaluation_count"]) == 1
                and int(row["materialized_full_payload_bytes"]) == 0
                and int(row["weight_plaintext_offline_encode_calls"]) == 1
                and int(row["weight_plaintext_online_encode_calls"]) == 0
                for row in all_runtime_stats
            )
        )
        sequence_isolated = bool(
            len(sequence) == 8
            and all(
                int(row["current_materialized_bytes_before"]) == 0
                and int(row["current_materialized_bytes_after"]) == 0
                and int(row["current_materialized_transforms_after"]) == 0
                for row in sequence
            )
        )
        peak_one = bool(
            int(global_after_compressed["peak_materialized_transform_count"]) == 1
            and int(global_after_compressed["peak_materialized_full_payload_bytes"])
            == int(global_after_compressed["max_single_transform_full_payload_bytes"])
            and int(global_after_compressed["current_materialized_transform_count"])
            == 0
            and int(global_after_compressed["current_materialized_full_payload_bytes"])
            == 0
        )
        expected_transform_count = int(sum(plan.case.transform_count for plan in plans))
        storage_valid = bool(
            expected_transform_count == 8
            and int(global_after_compressed["registered_transform_count"]) == 8
            and int(
                global_after_compressed[
                    "total_weight_plaintext_offline_encode_calls"
                ]
            )
            == 8
            and int(
                global_after_compressed[
                    "total_weight_plaintext_online_encode_calls"
                ]
            )
            == 0
            and int(global_after_compressed["aggregate_compressed_payload_bytes"])
            < int(global_after_compressed["aggregate_full_payload_bytes"])
        )
        full_correct = bool(
            np.allclose(full_output, reference_second, rtol=0.0, atol=float(args.atol))
        )
        compressed_correct = bool(
            np.allclose(
                compressed_output,
                reference_second,
                rtol=0.0,
                atol=float(args.atol),
            )
        )
        operations_match = bool(full_counters == compressed_counters)
        accumulation_match = bool(
            sum(
                row["ciphertext_accumulation_add_count"]
                for row in (full_first_row, full_second_row)
            )
            == sum(
                row["ciphertext_accumulation_add_count"]
                for row in (compressed_first_row, compressed_second_row)
            )
            == 4
        )
        bias_adds_match = bool(
            sum(
                row["bias_plaintext_add_count"]
                for row in (full_first_row, full_second_row)
            )
            == sum(
                row["bias_plaintext_add_count"]
                for row in (compressed_first_row, compressed_second_row)
            )
            == 4
        )
        level_chain_valid = bool(
            full_first_row["input_level"] == 2
            and full_first_row["output_level"] == 1
            and full_second_row["input_level"] == 1
            and full_second_row["output_level"] == 0
            and compressed_first_row["input_level"] == 2
            and compressed_first_row["output_level"] == 1
            and compressed_second_row["input_level"] == 1
            and compressed_second_row["output_level"] == 0
        )
        accepted = bool(
            signatures_chain
            and exact_qp
            and all_released
            and sequence_isolated
            and peak_one
            and storage_valid
            and full_correct
            and compressed_correct
            and float(np.max(path_delta)) == 0.0
            and operations_match
            and accumulation_match
            and bias_adds_match
            and level_chain_valid
            and online_encode_call_count == 0
        )
        payload = {
            "schema_version": 1,
            "profile": "wpc_cips_orion_conv2d_two_layer_pipeline",
            "status": "ok" if accepted else "invalid",
            "timing_policy": "diagnostic_single_run_not_for_performance_claims",
            "seed": int(args.seed),
            "scope": {
                "orion_module": "orion.nn.Conv2d",
                "layer_count": 2,
                "convolution": "3x3_stride1_same_shape_rotation_padding",
                "channels_per_layer": int(args.channels),
                "spatial_shape": [int(args.height), int(args.width)],
                "levels": [2, 1],
                "output_level": 0,
                "group_matrix_per_layer": [2, 2],
                "compressed_transform_count": 8,
                "limitations": [
                    "no activation or bootstrap between convolutions",
                    "no stride, downsample reshaping, dilation, or grouped convolution",
                    "uncompressed bias plaintexts remain resident",
                    "validation-only full transforms coexist during this runner",
                    "logical plaintext accounting rather than process RSS",
                    "no trained-model accuracy or performance claim",
                ],
            },
            "ckks": {
                "logn": int(args.logn),
                "slots": int(slots),
                "logq": [55, 45, 45],
                "logp": [60],
            },
            "compile_s": compile_s,
            "layer_plans": {
                plan.layer_name: {
                    "case": plan.case.to_dict(),
                    "level": int(plan.level),
                    "output_level": int(plan.output_level),
                    "transform_rows": plan.transform_rows,
                }
                for plan in plans
            },
            "storage": layer_storage,
            "global_stats_before_compressed_pipeline": global_before_compressed,
            "global_stats_after_compressed_pipeline": global_after_compressed,
            "full_control_pipeline": {
                "correct": full_correct,
                "max_abs_error": float(np.max(full_error)),
                "mean_abs_error": float(np.mean(full_error)),
                "evaluate_s": full_s,
                "operation_counters": full_counters,
                "layers": [full_first_row, full_second_row],
            },
            "compressed_orion_forward_pipeline": {
                "correct": compressed_correct,
                "max_abs_error": float(np.max(compressed_error)),
                "mean_abs_error": float(np.mean(compressed_error)),
                "max_abs_delta_vs_full_control": float(np.max(path_delta)),
                "evaluate_s": compressed_s,
                "operation_counters": compressed_counters,
                "operation_counters_match_full_control": operations_match,
                "layers": [compressed_first_row, compressed_second_row],
                "online_python_encode_call_count": int(online_encode_call_count),
            },
            "acceptance": {
                "orion_conv2d_forward_dispatch_exercised": True,
                "cips_output_to_input_signature_chains": signatures_chain,
                "eight_group_transforms_exact_qp": exact_qp,
                "full_and_compressed_two_layer_outputs_correct": bool(
                    full_correct and compressed_correct
                ),
                "compressed_output_exactly_matches_full_control": bool(
                    float(np.max(path_delta)) == 0.0
                ),
                "operation_counters_match": operations_match,
                "ciphertext_accumulation_counts_match": accumulation_match,
                "bias_plaintext_add_counts_match": bias_adds_match,
                "level_chain_2_to_1_to_0_valid": level_chain_valid,
                "zero_online_python_encode_calls": online_encode_call_count == 0,
                "all_compressed_transforms_released": all_released,
                "materialization_isolated_between_transforms": sequence_isolated,
                "peak_materialization_is_one_transform": peak_one,
                "aggregate_storage_accounting_valid": storage_valid,
                "valid": accepted,
            },
        }
    finally:
        for value in (
            compressed_second,
            compressed_first,
            full_second,
            full_first,
            input_ciphertext,
        ):
            release = getattr(value, "release", None)
            if callable(release):
                release()
        for layer in layers:
            layer.remove_wpc_cips_plan()
        scheme.delete_scheme()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(f"\nresult: {out_path}")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
