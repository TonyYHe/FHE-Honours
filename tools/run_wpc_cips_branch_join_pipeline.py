#!/usr/bin/env python3
"""Validate WPC CIPS residual and channel-concat joins with real Orion modules."""

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
from orion.experimental.wpc_cips_branches import WPCCIPSConcatPlan, WPCCIPSResidualAddPlan
from orion.experimental.wpc_cips_layer import WPC_GLOBAL_STATS_FIELDS
from orion.nn import Add, Concat, Conv2d


DEFAULT_OUT = (
    REPO_ROOT
    / ".tmp/results/honours/18_wpc_branch_joins/"
    "residual_concat_pipeline.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate encrypted WPC CIPS residual and channel-concat joins."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--residual-channels", type=int, default=12)
    parser.add_argument("--side-channels", type=int, default=4)
    parser.add_argument("--output-channels", type=int, default=8)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--atol", type=float, default=2e-6)
    return parser


def _config(logn: int) -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [55, 45, 45, 45],
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


def _make_layer(
    *,
    name: str,
    input_channels: int,
    output_channels: int,
    level: int,
    bsgs_ratio: float,
    weight: np.ndarray,
    bias: np.ndarray,
) -> Conv2d:
    layer = Conv2d(
        int(input_channels),
        int(output_channels),
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


def _operation_counters(backend: Any) -> dict[str, int | None]:
    values = [int(value) for value in backend.GetOperationCounters()]
    return {
        "rotation_total": values[0] if len(values) > 0 else None,
        "linear_transform_rotation": values[1] if len(values) > 1 else None,
        "direct_rotation": values[2] if len(values) > 2 else None,
        "conjugation": values[3] if len(values) > 3 else None,
    }


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
    capacity = int(slots // (int(args.height) * int(args.width)))
    concat_channels = int(args.residual_channels + args.side_channels)
    if int(args.input_channels) <= capacity or int(args.residual_channels) <= capacity:
        raise SystemExit(
            f"input and residual channels must exceed one-group capacity {capacity}"
        )
    if int(args.side_channels) > capacity:
        raise SystemExit("side branch must fit in one ciphertext for this validation")
    if concat_channels % capacity:
        raise SystemExit(
            "default validation requires concatenated channels to fill complete groups"
        )

    rng = np.random.default_rng(int(args.seed))
    input_values = rng.normal(
        0.0,
        0.3,
        size=(1, int(args.input_channels), int(args.height), int(args.width)),
    ).astype(np.float64)
    layer_specs = [
        (
            "wpc_residual_left",
            int(args.input_channels),
            int(args.residual_channels),
            3,
        ),
        (
            "wpc_residual_right",
            int(args.input_channels),
            int(args.residual_channels),
            3,
        ),
        (
            "wpc_concat_side",
            int(args.input_channels),
            int(args.side_channels),
            3,
        ),
        (
            "wpc_join_consumer",
            int(concat_channels),
            int(args.output_channels),
            1,
        ),
    ]
    weights = [
        rng.normal(0.0, 0.06, size=(out_c, in_c, 3, 3)).astype(np.float64)
        for _, in_c, out_c, _ in layer_specs
    ]
    biases = [
        rng.normal(0.0, 0.02, size=(out_c,)).astype(np.float64)
        for _, _, out_c, _ in layer_specs
    ]

    scheme = orion.init_scheme(_config(int(args.logn)))
    Conv2d.set_scheme(scheme)
    layers = [
        _make_layer(
            name=name,
            input_channels=in_c,
            output_channels=out_c,
            level=level,
            bsgs_ratio=float(args.bsgs_ratio),
            weight=weight,
            bias=bias,
        )
        for (name, in_c, out_c, level), weight, bias in zip(
            layer_specs, weights, biases
        )
    ]
    residual_module = Add()
    residual_module.name = "wpc_residual_add"
    concat_module = Concat(dim=1, bsgs_ratio=float(args.bsgs_ratio))
    concat_module.name = "wpc_channel_concat"
    plans: list[Any] = []
    residual_plan: WPCCIPSResidualAddPlan | None = None
    concat_plan: WPCCIPSConcatPlan | None = None
    values_to_release: list[Any] = []
    payload: dict[str, Any] = {}
    accepted = False
    try:
        input_shape = (
            1,
            int(args.input_channels),
            int(args.height),
            int(args.width),
        )
        consumer_shape = (
            1,
            int(concat_channels),
            int(args.height),
            int(args.width),
        )
        compile_started = time.perf_counter()
        left_plan = layers[0].install_wpc_cips_plan(
            input_shape, include_full_control=True, verify_exact_qp=True
        )
        right_plan = layers[1].install_wpc_cips_plan(
            input_shape, include_full_control=True, verify_exact_qp=True
        )
        side_plan = layers[2].install_wpc_cips_plan(
            input_shape, include_full_control=True, verify_exact_qp=True
        )
        consumer_plan = layers[3].install_wpc_cips_plan(
            consumer_shape, include_full_control=True, verify_exact_qp=True
        )
        plans = [left_plan, right_plan, side_plan, consumer_plan]
        residual_plan = residual_module.install_wpc_cips_plan(left_plan, right_plan)
        concat_plan = concat_module.install_wpc_cips_plan(
            (residual_plan, side_plan),
            consumer_plan=consumer_plan,
        )
        for layer in layers:
            layer.he()
        residual_module.he()
        concat_module.he()
        compile_s = float(time.perf_counter() - compile_started)

        clear_left = left_plan.clear_reference(input_values)
        clear_right = right_plan.clear_reference(input_values)
        clear_side = side_plan.clear_reference(input_values)
        clear_residual = clear_left + clear_right
        clear_concat = np.concatenate((clear_residual, clear_side), axis=0)
        clear_final = consumer_plan.clear_reference(clear_concat[None, ...])

        input_ciphertext = left_plan.encrypt_input(input_values)
        values_to_release.append(input_ciphertext)
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
            full_left = left_plan.evaluate(input_ciphertext, compressed=False)
            full_right = right_plan.evaluate(input_ciphertext, compressed=False)
            full_side = side_plan.evaluate(input_ciphertext, compressed=False)
            values_to_release.extend((full_left, full_right, full_side))
            full_residual = residual_plan.evaluate(full_left, full_right)
            values_to_release.append(full_residual)
            full_residual_row = dict(residual_plan.last_evaluation)
            full_residual_output = residual_plan.decrypt_unpack(full_residual)
            full_concat = concat_plan.evaluate(full_residual, full_side)
            values_to_release.append(full_concat)
            full_concat_row = dict(concat_plan.last_evaluation)
            full_concat_output = concat_plan.decrypt_unpack(full_concat)
            full_final = consumer_plan.evaluate(full_concat, compressed=False)
            values_to_release.append(full_final)
            full_s = float(time.perf_counter() - full_started)
            full_counters = _operation_counters(scheme.backend)
            full_output = consumer_plan.decrypt_unpack(full_final)

            scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
            scheme.backend.ResetOperationCounters()
            compressed_started = time.perf_counter()
            compressed_left = layers[0](input_ciphertext)
            compressed_left_row = dict(left_plan.last_evaluation)
            compressed_right = layers[1](input_ciphertext)
            compressed_right_row = dict(right_plan.last_evaluation)
            compressed_side = layers[2](input_ciphertext)
            compressed_side_row = dict(side_plan.last_evaluation)
            values_to_release.extend(
                (compressed_left, compressed_right, compressed_side)
            )
            compressed_residual = residual_module(compressed_left, compressed_right)
            values_to_release.append(compressed_residual)
            compressed_residual_row = dict(residual_plan.last_evaluation)
            compressed_residual_output = residual_plan.decrypt_unpack(
                compressed_residual
            )
            compressed_concat = concat_module(compressed_residual, compressed_side)
            values_to_release.append(compressed_concat)
            compressed_concat_row = dict(concat_plan.last_evaluation)
            compressed_concat_output = concat_plan.decrypt_unpack(compressed_concat)
            compressed_final = layers[3](compressed_concat)
            compressed_consumer_row = dict(consumer_plan.last_evaluation)
            values_to_release.append(compressed_final)
            compressed_s = float(time.perf_counter() - compressed_started)
            compressed_counters = _operation_counters(scheme.backend)
            compressed_output = consumer_plan.decrypt_unpack(compressed_final)
        finally:
            scheme.encoder.encode = original_encode

        global_stats = _global_stats(scheme.backend)
        compressed_stats = {
            plan.layer_name: plan.compressed_stats() for plan in plans
        }
        runtime_rows = [
            row
            for layer_rows in compressed_stats.values()
            for row in layer_rows.values()
        ]
        transform_rows = [
            row for plan in plans for row in plan.transform_rows.values()
        ]
        sequence = [
            *compressed_left_row["evaluation_sequence"],
            *compressed_right_row["evaluation_sequence"],
            *compressed_side_row["evaluation_sequence"],
            *compressed_consumer_row["evaluation_sequence"],
        ]
        layer_storage = {plan.layer_name: plan.storage_summary() for plan in plans}
        full_weight_bias = int(
            sum(row["full_weight_plus_bias_payload_bytes"] for row in layer_storage.values())
        )
        stored_weight_bias = int(
            sum(
                row["stored_weight_plus_metadata_plus_bias_bytes"]
                for row in layer_storage.values()
            )
        )
        concat_storage = concat_plan.storage_summary()

        errors = {
            "full_residual_max_abs_error": float(
                np.max(np.abs(full_residual_output - clear_residual))
            ),
            "compressed_residual_max_abs_error": float(
                np.max(np.abs(compressed_residual_output - clear_residual))
            ),
            "full_concat_max_abs_error": float(
                np.max(np.abs(full_concat_output - clear_concat))
            ),
            "compressed_concat_max_abs_error": float(
                np.max(np.abs(compressed_concat_output - clear_concat))
            ),
            "full_final_max_abs_error": float(np.max(np.abs(full_output - clear_final))),
            "compressed_final_max_abs_error": float(
                np.max(np.abs(compressed_output - clear_final))
            ),
            "compressed_vs_full_final_max_abs_delta": float(
                np.max(np.abs(compressed_output - full_output))
            ),
        }
        expected_transforms = int(sum(plan.case.transform_count for plan in plans))
        exact_qp = bool(
            len(transform_rows) == expected_transforms
            and all(row["exact_qp_match"] is True for row in transform_rows)
        )
        runtime_released = bool(
            len(runtime_rows) == expected_transforms
            and all(
                int(row["evaluation_count"]) == 1
                and int(row["materialized_full_payload_bytes"]) == 0
                and int(row["weight_plaintext_offline_encode_calls"]) == 1
                and int(row["weight_plaintext_online_encode_calls"]) == 0
                for row in runtime_rows
            )
        )
        sequence_isolated = bool(
            len(sequence) == expected_transforms
            and all(
                int(row["current_materialized_bytes_before"]) == 0
                and int(row["current_materialized_bytes_after"]) == 0
                and int(row["current_materialized_transforms_after"]) == 0
                for row in sequence
            )
        )
        peak_one = bool(
            int(global_stats["peak_materialized_transform_count"]) == 1
            and int(global_stats["peak_materialized_full_payload_bytes"])
            == int(global_stats["max_single_transform_full_payload_bytes"])
            and int(global_stats["current_materialized_transform_count"]) == 0
            and int(global_stats["current_materialized_full_payload_bytes"]) == 0
        )
        signatures_valid = bool(
            left_plan.output_packing_signature == right_plan.output_packing_signature
            == residual_plan.output_packing_signature
            and concat_plan.output_packing_signature
            == consumer_plan.input_packing_signature
        )
        levels_valid = bool(
            compressed_left_row["input_level"] == 3
            and compressed_left_row["output_level"] == 2
            and compressed_right_row["output_level"] == 2
            and compressed_side_row["output_level"] == 2
            and compressed_residual_row["input_level"] == 2
            and compressed_residual_row["output_level"] == 2
            and compressed_concat_row["input_level"] == 2
            and compressed_concat_row["output_level"] == 1
            and compressed_consumer_row["input_level"] == 1
            and compressed_consumer_row["output_level"] == 0
        )
        join_counts_valid = bool(
            compressed_residual_row["ciphertext_add_count"] == 2
            and compressed_concat_row["input_branch_count"] == 2
            and compressed_concat_row["input_ciphertext_group_count"] == 3
            and compressed_concat_row["output_ciphertext_group_count"] == 2
            and compressed_concat_row["transform_evaluation_count"] == 3
            and compressed_concat_row["ciphertext_accumulation_add_count"] == 1
        )
        correctness = bool(
            all(value <= float(args.atol) for key, value in errors.items() if "delta" not in key)
            and errors["compressed_vs_full_final_max_abs_delta"] == 0.0
        )
        storage_valid = bool(
            int(global_stats["registered_transform_count"]) == expected_transforms
            and int(global_stats["total_weight_plaintext_offline_encode_calls"])
            == expected_transforms
            and int(global_stats["total_weight_plaintext_online_encode_calls"]) == 0
            and stored_weight_bias < full_weight_bias
        )
        acceptance = {
            "actual_orion_add_and_concat_modules_executed": bool(
                isinstance(residual_plan, WPCCIPSResidualAddPlan)
                and isinstance(concat_plan, WPCCIPSConcatPlan)
            ),
            "compressed_weight_qp_exactly_matches_full_control": exact_qp,
            "encrypted_residual_matches_clear": bool(
                errors["compressed_residual_max_abs_error"] <= float(args.atol)
            ),
            "encrypted_concat_matches_clear": bool(
                errors["compressed_concat_max_abs_error"] <= float(args.atol)
            ),
            "full_and_compressed_final_outputs_correct": correctness,
            "operation_counters_match": full_counters == compressed_counters,
            "packing_signatures_chain_without_clear_repack": signatures_valid,
            "residual_preserves_level_and_concat_consumes_one_level": levels_valid,
            "branch_join_operation_counts_valid": join_counts_valid,
            "zero_online_python_encode_calls": online_encode_call_count == 0,
            "branch_joins_report_zero_clear_repack_or_encode": bool(
                compressed_residual_row["clear_repack_or_encode_count"] == 0
                and compressed_concat_row["clear_repack_or_encode_count"] == 0
            ),
            "all_compressed_weight_materialization_released": runtime_released,
            "materialization_isolated_between_weight_transforms": sequence_isolated,
            "peak_weight_materialization_is_one_transform": peak_one,
            "storage_and_weight_encode_accounting_valid": storage_valid,
        }
        accepted = bool(all(acceptance.values()))
        acceptance["valid"] = accepted

        payload = {
            "schema_version": 1,
            "profile": "wpc_cips_residual_concat_branch_join_pipeline",
            "status": "ok" if accepted else "validation_failed",
            "seed": int(args.seed),
            "scope": {
                "pipeline": "three_compressed_branches_residual_add_channel_concat_compressed_consumer",
                "input_shape": list(input_values.shape),
                "residual_shape": list(clear_residual.shape),
                "side_shape": list(clear_side.shape),
                "concat_shape": list(clear_concat.shape),
                "final_shape": list(clear_final.shape),
                "compressed_weight_transform_count": expected_transforms,
                "concat_transform_count": len(concat_plan.transform_ids),
                "limitations": [
                    "functional branch block rather than a complete trained model",
                    "concat uses ordinary offline-encoded Q/P permutation plaintexts",
                    "all branches must share spatial shape, slots, scheme, and level",
                    "full-Q/P weight controls coexist only for correctness validation",
                    "logical plaintext accounting rather than process RSS",
                    "diagnostic single-run timing without performance claim",
                ],
            },
            "packing": {
                "residual_branch_signature": left_plan.output_packing_signature,
                "residual_output_signature": residual_plan.output_packing_signature,
                "side_branch_signature": side_plan.output_packing_signature,
                "concat_output_signature": concat_plan.output_packing_signature,
                "consumer_input_signature": consumer_plan.input_packing_signature,
            },
            "levels": {
                "branch_convolutions": [3, 2],
                "residual_add": [2, 2],
                "channel_concat": [2, 1],
                "consumer_convolution": [1, 0],
            },
            "branch_joins": {
                "residual": {
                    "full_path_evaluation": full_residual_row,
                    "compressed_path_evaluation": compressed_residual_row,
                },
                "concat": {
                    "transforms": concat_plan.transform_rows,
                    "storage": concat_storage,
                    "full_path_evaluation": full_concat_row,
                    "compressed_path_evaluation": compressed_concat_row,
                },
            },
            "periodicity_and_exact_qp": {
                plan.layer_name: plan.transform_rows for plan in plans
            },
            "storage": {
                "by_weight_layer": layer_storage,
                "full_weight_plus_bias_payload_bytes": full_weight_bias,
                "stored_weight_plus_metadata_plus_bias_bytes": stored_weight_bias,
                "weight_layer_storage_compression_ratio": float(
                    full_weight_bias / stored_weight_bias
                ),
                "ordinary_concat_qp_payload_bytes": int(
                    concat_storage["full_qp_payload_bytes"]
                ),
                "stored_weight_and_concat_bytes": int(
                    stored_weight_bias + concat_storage["full_qp_payload_bytes"]
                ),
            },
            "compressed_runtime_stats": compressed_stats,
            "global_stats_after_compressed_pipeline": global_stats,
            "errors": errors,
            "operation_counters": {
                "full_control": full_counters,
                "compressed": compressed_counters,
            },
            "timings_s": {
                "compile": compile_s,
                "full_control_pipeline": full_s,
                "compressed_pipeline": compressed_s,
            },
            "online_python_encode_call_count": int(online_encode_call_count),
            "acceptance": acceptance,
            "timing_policy": "diagnostic_single_run_not_for_performance_claims",
        }
    finally:
        for value in reversed(values_to_release):
            release = getattr(value, "release", None)
            if callable(release):
                release()
        concat_module.remove_wpc_cips_plan()
        residual_module.remove_wpc_cips_plan()
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
