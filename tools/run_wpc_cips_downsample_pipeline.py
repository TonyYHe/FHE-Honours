#!/usr/bin/env python3
"""Validate stride-two WPC CIPS convolution and encrypted layout reshaping.

The deterministic pipeline is:

    Conv2d(stride=2) -> sparse-to-compact reshape -> Conv2d(stride=1)

Both convolution layers use WPC compressed Q/P storage.  A separately encoded
full-Q/P control traverses the identical reshape transform.  Timings are
diagnostic single observations and are not a performance comparison.
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
from orion.experimental.wpc_cips_downsample import (
    WPCCIPSDownsampleReshapePlan,
    WPCCIPSStride2Conv2dPlan,
    build_downsample_reshape_transforms,
    pack_compact_output,
    pack_sparse_output,
    unpack_compact_output,
)
from orion.experimental.wpc_cips_layer import WPC_GLOBAL_STATS_FIELDS
from orion.nn import Conv2d


DEFAULT_OUT = (
    REPO_ROOT
    / ".tmp/results/honours/17_wpc_downsample_reshape/"
    "stride2_reshape_pipeline.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate WPC stride-two CIPS and its one-level encrypted reshape."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260929)
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
    channels: int,
    stride: int,
    level: int,
    bsgs_ratio: float,
    weight: np.ndarray,
    bias: np.ndarray,
) -> Conv2d:
    layer = Conv2d(
        int(channels),
        int(channels),
        3,
        stride=int(stride),
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


def _compact_clear_roundtrip(values: np.ndarray, plan: Any) -> bool:
    packed = pack_compact_output(values, plan.case)
    return bool(np.array_equal(unpack_compact_output(packed, plan.case), values))


def main() -> int:
    args = _parser().parse_args()
    slots = 1 << (int(args.logn) - 1)
    input_capacity = int(slots // (int(args.height) * int(args.width)))
    if int(args.channels) <= input_capacity:
        raise SystemExit(
            f"channels must exceed input-grid ciphertext capacity {input_capacity}"
        )
    if int(args.height) % 2 or int(args.width) % 2:
        raise SystemExit("height and width must be divisible by two")

    rng = np.random.default_rng(int(args.seed))
    input_values = rng.normal(
        0.0,
        0.35,
        size=(1, int(args.channels), int(args.height), int(args.width)),
    ).astype(np.float64)
    weights = [
        rng.normal(
            0.0,
            0.07,
            size=(int(args.channels), int(args.channels), 3, 3),
        ).astype(np.float64)
        for _ in range(2)
    ]
    biases = [
        rng.normal(0.0, 0.02, size=(int(args.channels),)).astype(np.float64)
        for _ in range(2)
    ]

    scheme = orion.init_scheme(_config(int(args.logn)))
    Conv2d.set_scheme(scheme)
    layers = [
        _make_layer(
            name="wpc_downsample_conv",
            channels=int(args.channels),
            stride=2,
            level=3,
            bsgs_ratio=float(args.bsgs_ratio),
            weight=weights[0],
            bias=biases[0],
        ),
        _make_layer(
            name="wpc_post_downsample_conv",
            channels=int(args.channels),
            stride=1,
            level=1,
            bsgs_ratio=float(args.bsgs_ratio),
            weight=weights[1],
            bias=biases[1],
        ),
    ]
    plans: list[Any] = []
    reshape: WPCCIPSDownsampleReshapePlan | None = None
    values_to_release: list[Any] = []
    payload: dict[str, Any] = {}
    accepted = False
    try:
        compile_started = time.perf_counter()
        first_plan = layers[0].install_wpc_cips_plan(
            (1, int(args.channels), int(args.height), int(args.width)),
            include_full_control=True,
            verify_exact_qp=True,
        )
        if not isinstance(first_plan, WPCCIPSStride2Conv2dPlan):
            raise RuntimeError("stride-two Conv2d did not install its stride-two WPC plan")
        second_shape = (
            1,
            int(args.channels),
            int(first_plan.case.output_height),
            int(first_plan.case.output_width),
        )
        second_plan = layers[1].install_wpc_cips_plan(
            second_shape,
            include_full_control=True,
            verify_exact_qp=True,
        )
        plans = [first_plan, second_plan]
        reshape = WPCCIPSDownsampleReshapePlan(
            first_plan,
            level=2,
            bsgs_ratio=float(args.bsgs_ratio),
        )
        reshape.compile(scheme)
        reshape.validate_consumer(second_plan)
        for layer in layers:
            layer.he()
        compile_s = float(time.perf_counter() - compile_started)

        clear_first = first_plan.clear_reference(input_values)
        clear_second = second_plan.clear_reference(clear_first[None, ...])
        sparse_clear = pack_sparse_output(clear_first, first_plan.case)
        compact_clear = pack_compact_output(clear_first, first_plan.case)
        reshape_rows = build_downsample_reshape_transforms(first_plan.case)
        clear_layout_valid = bool(
            sparse_clear.shape[0] == first_plan.case.output_group_count
            and compact_clear.shape[0] == first_plan.case.compact_output_group_count
            and _compact_clear_roundtrip(clear_first, first_plan)
            and len(reshape_rows) > 0
        )

        input_ciphertext = first_plan.encrypt_input(input_values)
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
            full_sparse = first_plan.evaluate(input_ciphertext, compressed=False)
            values_to_release.append(full_sparse)
            full_sparse_output = first_plan.decrypt_unpack(full_sparse)
            full_compact = reshape.evaluate(full_sparse)
            values_to_release.append(full_compact)
            full_reshape_row = dict(reshape.last_evaluation)
            full_compact_output = reshape.decrypt_unpack(full_compact)
            full_final = second_plan.evaluate(full_compact, compressed=False)
            values_to_release.append(full_final)
            full_s = float(time.perf_counter() - full_started)
            full_counters = _operation_counters(scheme.backend)
            full_output = second_plan.decrypt_unpack(full_final)

            scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
            scheme.backend.ResetOperationCounters()
            compressed_started = time.perf_counter()
            compressed_sparse = layers[0](input_ciphertext)
            values_to_release.append(compressed_sparse)
            compressed_first_row = dict(first_plan.last_evaluation)
            compressed_sparse_output = first_plan.decrypt_unpack(compressed_sparse)
            compressed_compact = reshape.evaluate(compressed_sparse)
            values_to_release.append(compressed_compact)
            compressed_reshape_row = dict(reshape.last_evaluation)
            compressed_compact_output = reshape.decrypt_unpack(compressed_compact)
            compressed_final = layers[1](compressed_compact)
            values_to_release.append(compressed_final)
            compressed_second_row = dict(second_plan.last_evaluation)
            compressed_s = float(time.perf_counter() - compressed_started)
            compressed_counters = _operation_counters(scheme.backend)
            compressed_output = second_plan.decrypt_unpack(compressed_final)
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
            *compressed_first_row["evaluation_sequence"],
            *compressed_second_row["evaluation_sequence"],
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
        reshape_storage = reshape.storage_summary()

        errors = {
            "full_sparse_max_abs_error": float(
                np.max(np.abs(full_sparse_output - clear_first))
            ),
            "compressed_sparse_max_abs_error": float(
                np.max(np.abs(compressed_sparse_output - clear_first))
            ),
            "full_reshape_max_abs_error": float(
                np.max(np.abs(full_compact_output - clear_first))
            ),
            "compressed_reshape_max_abs_error": float(
                np.max(np.abs(compressed_compact_output - clear_first))
            ),
            "full_final_max_abs_error": float(np.max(np.abs(full_output - clear_second))),
            "compressed_final_max_abs_error": float(
                np.max(np.abs(compressed_output - clear_second))
            ),
            "compressed_vs_full_final_max_abs_delta": float(
                np.max(np.abs(compressed_output - full_output))
            ),
        }
        expected_compressed_transforms = int(sum(plan.case.transform_count for plan in plans))
        exact_qp = bool(
            len(transform_rows) == expected_compressed_transforms
            and all(row["exact_qp_match"] is True for row in transform_rows)
        )
        runtime_released = bool(
            len(runtime_rows) == expected_compressed_transforms
            and all(
                int(row["evaluation_count"]) == 1
                and int(row["materialized_full_payload_bytes"]) == 0
                and int(row["weight_plaintext_offline_encode_calls"]) == 1
                and int(row["weight_plaintext_online_encode_calls"]) == 0
                for row in runtime_rows
            )
        )
        sequence_isolated = bool(
            len(sequence) == expected_compressed_transforms
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
        level_chain = bool(
            compressed_first_row["input_level"] == 3
            and compressed_first_row["output_level"] == 2
            and compressed_reshape_row["input_level"] == 2
            and compressed_reshape_row["output_level"] == 1
            and compressed_second_row["input_level"] == 1
            and compressed_second_row["output_level"] == 0
        )
        group_compaction = bool(
            compressed_reshape_row["input_ciphertext_group_count"] == 2
            and compressed_reshape_row["output_ciphertext_group_count"] == 1
        )
        signature_chain = bool(
            first_plan.output_packing_signature == reshape.input_packing_signature
            and reshape.output_packing_signature == second_plan.input_packing_signature
        )
        operation_match = full_counters == compressed_counters
        correctness = bool(
            all(value <= float(args.atol) for key, value in errors.items() if "delta" not in key)
            and errors["compressed_vs_full_final_max_abs_delta"] == 0.0
        )
        storage_valid = bool(
            int(global_stats["registered_transform_count"])
            == expected_compressed_transforms
            and int(global_stats["total_weight_plaintext_offline_encode_calls"])
            == expected_compressed_transforms
            and int(global_stats["total_weight_plaintext_online_encode_calls"]) == 0
            and stored_weight_bias < full_weight_bias
        )
        acceptance = {
            "clear_sparse_and_compact_layout_roundtrips_valid": clear_layout_valid,
            "stride2_plan_installed_on_actual_orion_conv2d": isinstance(
                first_plan, WPCCIPSStride2Conv2dPlan
            ),
            "compressed_weight_qp_exactly_matches_full_control": exact_qp,
            "encrypted_sparse_intermediate_matches_clear": bool(
                errors["compressed_sparse_max_abs_error"] <= float(args.atol)
            ),
            "encrypted_reshape_matches_clear": bool(
                errors["compressed_reshape_max_abs_error"] <= float(args.atol)
            ),
            "full_and_compressed_final_outputs_correct": correctness,
            "operation_counters_match": operation_match,
            "packing_signatures_chain_without_clear_repack": signature_chain,
            "reshape_compacts_two_ciphertexts_to_one": group_compaction,
            "reshape_consumes_exactly_one_level": level_chain,
            "zero_online_python_encode_calls": online_encode_call_count == 0,
            "reshape_reports_zero_clear_repack_or_encode": (
                compressed_reshape_row["clear_repack_or_encode_count"] == 0
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
            "profile": "wpc_cips_stride2_encrypted_reshape_pipeline",
            "status": "ok" if accepted else "validation_failed",
            "seed": int(args.seed),
            "scope": {
                "pipeline": "stride2_conv_sparse_cips_then_one_level_reshape_then_stride1_conv",
                "input_shape": list(input_values.shape),
                "downsample_output_shape": list(clear_first.shape),
                "stride": [2, 2],
                "rotation_padding": "flattened_spatial_cyclic",
                "source_ciphertext_groups": int(first_plan.case.output_group_count),
                "compact_ciphertext_groups": int(
                    first_plan.case.compact_output_group_count
                ),
                "compressed_weight_transform_count": expected_compressed_transforms,
                "reshape_transform_count": len(reshape.transform_ids),
                "limitations": [
                    "functional two-layer pipeline rather than a complete trained model",
                    "reshape uses ordinary offline-encoded Q/P permutation plaintexts",
                    "full-Q/P weight controls coexist only for correctness validation",
                    "logical plaintext accounting rather than process RSS",
                    "diagnostic single-run timing without performance claim",
                ],
            },
            "stride2_case": first_plan.case.to_dict(),
            "packing": {
                "stride2_input_signature": first_plan.input_packing_signature,
                "sparse_output_signature": first_plan.output_packing_signature,
                "compact_output_signature": reshape.output_packing_signature,
                "consumer_input_signature": second_plan.input_packing_signature,
            },
            "levels": {
                "stride2_convolution": [3, 2],
                "encrypted_reshape": [2, 1],
                "post_downsample_convolution": [1, 0],
            },
            "periodicity_and_exact_qp": {
                plan.layer_name: plan.transform_rows for plan in plans
            },
            "reshape": {
                "transforms": reshape.transform_rows,
                "storage": reshape_storage,
                "full_path_evaluation": full_reshape_row,
                "compressed_path_evaluation": compressed_reshape_row,
            },
            "storage": {
                "by_weight_layer": layer_storage,
                "full_weight_plus_bias_payload_bytes": full_weight_bias,
                "stored_weight_plus_metadata_plus_bias_bytes": stored_weight_bias,
                "weight_layer_storage_compression_ratio": float(
                    full_weight_bias / stored_weight_bias
                ),
                "ordinary_reshape_qp_payload_bytes": int(
                    reshape_storage["full_qp_payload_bytes"]
                ),
                "stored_weight_and_reshape_bytes": int(
                    stored_weight_bias + reshape_storage["full_qp_payload_bytes"]
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
        if reshape is not None:
            reshape.cleanup()
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
