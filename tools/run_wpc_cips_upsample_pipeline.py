#!/usr/bin/env python3
"""Validate compressed CIPS ConvTranspose2d in a decoder-style pipeline.

The deterministic pipeline is:

    ConvTranspose2d(kernel=2,stride=2) -> Conv2d(kernel=3,stride=1)

Both weight layers have compressed and full-Q/P controls.  The high-resolution
CIPS output of the transposed convolution is consumed directly: there is no
decrypt, clear repack, Encode, or re-encryption at the boundary.  Timings are
single diagnostic observations and are not performance claims.
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
from orion.experimental.wpc_cips_upsample import (
    WPCCIPSConvTranspose2dPlan,
    pack_upsample_output,
    unpack_upsample_output,
)
from orion.nn import Conv2d, ConvTranspose2d


DEFAULT_OUT = (
    REPO_ROOT
    / ".tmp/results/honours/19_wpc_upsample/"
    "conv_transpose2d_pipeline.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate WPC-compressed CIPS transposed-convolution upsampling."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--consumer-channels", type=int, default=8)
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


def _make_upsample(
    *,
    channels: int,
    level: int,
    bsgs_ratio: float,
    weight: np.ndarray,
    bias: np.ndarray,
) -> ConvTranspose2d:
    layer = ConvTranspose2d(
        int(channels),
        int(channels),
        2,
        stride=2,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        bsgs_ratio=float(bsgs_ratio),
        level=int(level),
    )
    layer.name = "wpc_decoder_upsample"
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    layer.init_orion_params()
    return layer


def _make_consumer(
    *,
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
    layer.name = "wpc_decoder_consumer"
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
    if int(args.height) <= 0 or int(args.width) <= 0:
        raise SystemExit("height and width must be positive")
    if slots % (4 * int(args.height) * int(args.width)):
        raise SystemExit("slots must be divisible by the high-resolution area")

    rng = np.random.default_rng(int(args.seed))
    input_values = rng.normal(
        0.0,
        0.35,
        size=(1, int(args.channels), int(args.height), int(args.width)),
    ).astype(np.float64)
    upsample_weight = rng.normal(
        0.0,
        0.07,
        size=(int(args.channels), int(args.channels), 2, 2),
    ).astype(np.float64)
    upsample_bias = rng.normal(
        0.0, 0.02, size=(int(args.channels),)
    ).astype(np.float64)
    consumer_weight = rng.normal(
        0.0,
        0.07,
        size=(int(args.consumer_channels), int(args.channels), 3, 3),
    ).astype(np.float64)
    consumer_bias = rng.normal(
        0.0, 0.02, size=(int(args.consumer_channels),)
    ).astype(np.float64)

    scheme = orion.init_scheme(_config(int(args.logn)))
    Conv2d.set_scheme(scheme)
    ConvTranspose2d.set_scheme(scheme)
    layers = [
        _make_upsample(
            channels=int(args.channels),
            level=2,
            bsgs_ratio=float(args.bsgs_ratio),
            weight=upsample_weight,
            bias=upsample_bias,
        ),
        _make_consumer(
            input_channels=int(args.channels),
            output_channels=int(args.consumer_channels),
            level=1,
            bsgs_ratio=float(args.bsgs_ratio),
            weight=consumer_weight,
            bias=consumer_bias,
        ),
    ]
    plans: list[Any] = []
    values_to_release: list[Any] = []
    payload: dict[str, Any] = {}
    accepted = False
    try:
        compile_started = time.perf_counter()
        upsample_plan = layers[0].install_wpc_cips_plan(
            tuple(input_values.shape),
            include_full_control=True,
            verify_exact_qp=True,
        )
        if not isinstance(upsample_plan, WPCCIPSConvTranspose2dPlan):
            raise RuntimeError("ConvTranspose2d did not install its WPC CIPS plan")
        consumer_plan = layers[1].install_wpc_cips_plan(
            tuple(upsample_plan.output_shape),
            include_full_control=True,
            verify_exact_qp=True,
        )
        plans = [upsample_plan, consumer_plan]
        for layer in layers:
            layer.he()
        compile_s = float(time.perf_counter() - compile_started)

        clear_upsample = upsample_plan.clear_reference(input_values)
        clear_final = consumer_plan.clear_reference(clear_upsample[None, ...])
        packed_clear = pack_upsample_output(clear_upsample, upsample_plan.case)
        clear_layout_roundtrip = bool(
            np.array_equal(
                unpack_upsample_output(packed_clear, upsample_plan.case),
                clear_upsample,
            )
        )

        input_ciphertext = upsample_plan.encrypt_input(input_values)
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
            full_upsample = upsample_plan.evaluate(input_ciphertext, compressed=False)
            values_to_release.append(full_upsample)
            full_upsample_output = upsample_plan.decrypt_unpack(full_upsample)
            full_final = consumer_plan.evaluate(full_upsample, compressed=False)
            values_to_release.append(full_final)
            full_s = float(time.perf_counter() - full_started)
            full_counters = _operation_counters(scheme.backend)
            full_output = consumer_plan.decrypt_unpack(full_final)

            scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
            scheme.backend.ResetOperationCounters()
            compressed_started = time.perf_counter()
            compressed_upsample = layers[0](input_ciphertext)
            values_to_release.append(compressed_upsample)
            compressed_upsample_row = dict(upsample_plan.last_evaluation)
            compressed_upsample_output = upsample_plan.decrypt_unpack(
                compressed_upsample
            )
            compressed_final = layers[1](compressed_upsample)
            values_to_release.append(compressed_final)
            compressed_consumer_row = dict(consumer_plan.last_evaluation)
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
            *compressed_upsample_row["evaluation_sequence"],
            *compressed_consumer_row["evaluation_sequence"],
        ]
        layer_storage = {plan.layer_name: plan.storage_summary() for plan in plans}
        full_weight_bias = int(
            sum(
                row["full_weight_plus_bias_payload_bytes"]
                for row in layer_storage.values()
            )
        )
        stored_weight_bias = int(
            sum(
                row["stored_weight_plus_metadata_plus_bias_bytes"]
                for row in layer_storage.values()
            )
        )

        errors = {
            "full_upsample_max_abs_error": float(
                np.max(np.abs(full_upsample_output - clear_upsample))
            ),
            "compressed_upsample_max_abs_error": float(
                np.max(np.abs(compressed_upsample_output - clear_upsample))
            ),
            "full_final_max_abs_error": float(
                np.max(np.abs(full_output - clear_final))
            ),
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
        level_chain = bool(
            compressed_upsample_row["input_level"] == 2
            and compressed_upsample_row["output_level"] == 1
            and compressed_consumer_row["input_level"] == 1
            and compressed_consumer_row["output_level"] == 0
        )
        group_expansion = bool(
            upsample_plan.case.input_group_count == 1
            and upsample_plan.case.output_group_count == 2
            and len(compressed_upsample.ids) == 2
        )
        signature_chain = bool(
            upsample_plan.output_packing_signature
            == consumer_plan.input_packing_signature
        )
        operation_match = full_counters == compressed_counters
        correctness = bool(
            all(
                value <= float(args.atol)
                for key, value in errors.items()
                if "delta" not in key
            )
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
            "clear_high_resolution_cips_roundtrip_valid": clear_layout_roundtrip,
            "wpc_plan_installed_on_actual_orion_convtranspose2d": isinstance(
                upsample_plan, WPCCIPSConvTranspose2dPlan
            ),
            "compressed_weight_qp_exactly_matches_full_control": exact_qp,
            "encrypted_upsample_matches_clear": bool(
                errors["compressed_upsample_max_abs_error"] <= float(args.atol)
            ),
            "full_and_compressed_final_outputs_correct": correctness,
            "operation_counters_match": operation_match,
            "packing_signatures_chain_without_clear_repack": signature_chain,
            "upsample_expands_one_ciphertext_group_to_two": group_expansion,
            "two_weight_layers_consume_exactly_two_levels": level_chain,
            "zero_online_python_encode_calls": online_encode_call_count == 0,
            "all_compressed_weight_materialization_released": runtime_released,
            "materialization_isolated_between_weight_transforms": sequence_isolated,
            "peak_weight_materialization_is_one_transform": peak_one,
            "storage_and_weight_encode_accounting_valid": storage_valid,
        }
        accepted = bool(all(acceptance.values()))
        acceptance["valid"] = accepted

        payload = {
            "schema_version": 1,
            "profile": "wpc_cips_convtranspose2d_decoder_pipeline",
            "status": "ok" if accepted else "validation_failed",
            "seed": int(args.seed),
            "scope": {
                "pipeline": "convtranspose2d_2x_upsample_then_stride1_conv",
                "input_shape": list(input_values.shape),
                "upsample_output_shape": list(clear_upsample.shape),
                "final_output_shape": list(clear_final.shape),
                "kernel": [2, 2],
                "stride": [2, 2],
                "input_ciphertext_groups": int(
                    upsample_plan.case.input_group_count
                ),
                "high_resolution_ciphertext_groups": int(
                    upsample_plan.case.output_group_count
                ),
                "compressed_weight_transform_count": expected_transforms,
                "limitations": [
                    "functional decoder block rather than a complete trained U-Net",
                    "supports the U-Net 2x2 stride-two zero-padding geometry only",
                    "full-Q/P controls coexist only for correctness validation",
                    "logical plaintext accounting rather than process RSS",
                    "diagnostic single-run timing without performance claim",
                ],
            },
            "upsample_case": upsample_plan.case.to_dict(),
            "packing": {
                "low_resolution_input_signature": upsample_plan.input_packing_signature,
                "high_resolution_output_signature": upsample_plan.output_packing_signature,
                "consumer_input_signature": consumer_plan.input_packing_signature,
            },
            "levels": {
                "transposed_convolution": [2, 1],
                "post_upsample_convolution": [1, 0],
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
