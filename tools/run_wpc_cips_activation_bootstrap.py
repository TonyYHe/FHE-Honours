#!/usr/bin/env python3
"""Validate Conv2d -> activation -> bootstrap -> Conv2d in WPC CIPS layout.

This is a correctness and accounting experiment.  It uses real Orion
``Conv2d`` and ``Quad`` modules, a real Lattigo bootstrap for every grouped
CIPS ciphertext, and the compressed-Q/P convolution path.  Single-run timing
fields are diagnostic and are not a performance comparison.
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
from orion.experimental.wpc_cips_activation import WPCCIPSActivationBootstrap
from orion.experimental.wpc_cips_layer import WPC_GLOBAL_STATS_FIELDS
from orion.nn import Conv2d


DEFAULT_OUT = (
    REPO_ROOT
    / ".tmp/results/honours/16_wpc_activation_bootstrap/"
    "two_conv_quad_bootstrap.json"
)

BOOTSTRAP_PROFILE_FIELDS = (
    "seq",
    "kind",
    "slots",
    "input_level",
    "output_level",
    "input_slots",
    "output_slots",
    "input_log_cols",
    "output_log_cols",
    "total_ns",
    "retrieve_ns",
    "copy_ns",
    "evaluator_bootstrap_ns",
    "postscale_ns",
    "push_ns",
    "input_degree",
    "output_degree",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--bound-headroom", type=float, default=1.25)
    parser.add_argument("--atol", type=float, default=2e-5)
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
        "boot_params": {"LogP": [61] * 8},
        "orion": {
            "margin": 1,
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


def _operation_counters(backend: Any) -> dict[str, int]:
    values = [int(value) for value in backend.GetOperationCounters()]
    names = (
        "rotation_total",
        "linear_transform_rotation",
        "direct_rotation",
        "conjugation",
    )
    if len(values) != len(names):
        raise RuntimeError("unexpected operation-counter length")
    return {name: value for name, value in zip(names, values)}


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


def _set_bootstrap_profile(backend: Any, enabled: bool) -> None:
    if not all(
        callable(getattr(backend, name, None))
        for name in (
            "EnableBootstrapProfile",
            "ResetBootstrapProfile",
            "GetBootstrapProfileCounters",
        )
    ):
        raise RuntimeError("Lattigo backend does not expose bootstrap profiling")
    if enabled:
        backend.ResetBootstrapProfile()
        backend.EnableBootstrapProfile(1)
    else:
        backend.EnableBootstrapProfile(0)


def _bootstrap_profile(backend: Any) -> dict[str, Any]:
    values = [int(value) for value in backend.GetBootstrapProfileCounters()]
    width = len(BOOTSTRAP_PROFILE_FIELDS)
    if len(values) % width:
        raise RuntimeError("malformed Lattigo bootstrap profile")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(values), width):
        row = {
            name: int(value)
            for name, value in zip(
                BOOTSTRAP_PROFILE_FIELDS,
                values[start : start + width],
            )
        }
        for name in (
            "total_ns",
            "retrieve_ns",
            "copy_ns",
            "evaluator_bootstrap_ns",
            "postscale_ns",
            "push_ns",
        ):
            row[name.replace("_ns", "_s")] = float(row[name] / 1_000_000_000)
        rows.append(row)
    return {
        "row_count": int(len(rows)),
        "rows": rows,
        "total_s": float(sum(row["total_s"] for row in rows)),
        "evaluator_bootstrap_s": float(
            sum(row["evaluator_bootstrap_s"] for row in rows)
        ),
    }


def _aggregate_storage(plans: list[Any]) -> dict[str, Any]:
    rows = {plan.layer_name: plan.storage_summary() for plan in plans}
    fields = (
        "full_weight_qp_payload_bytes",
        "resident_weight_qp_payload_bytes",
        "weight_metadata_bytes",
        "uncompressed_bias_q_payload_bytes",
        "full_weight_plus_bias_payload_bytes",
        "stored_weight_plus_metadata_plus_bias_bytes",
    )
    result: dict[str, Any] = {
        name: int(sum(int(row[name]) for row in rows.values())) for name in fields
    }
    result["transform_count"] = int(
        sum(int(row["transform_count"]) for row in rows.values())
    )
    result["full_to_stored_ratio"] = float(
        result["full_weight_plus_bias_payload_bytes"]
        / result["stored_weight_plus_metadata_plus_bias_bytes"]
    )
    result["by_layer"] = rows
    return result


def _release(value: Any) -> None:
    release = getattr(value, "release", None)
    if callable(release):
        release()


def main() -> int:
    args = _parser().parse_args()
    if float(args.bound_headroom) < 1.0:
        raise SystemExit("bound-headroom must be at least one")
    slots = 1 << (int(args.logn) - 1)
    channel_capacity = int(slots // (int(args.height) * int(args.width)))
    if int(args.channels) <= channel_capacity:
        raise SystemExit(
            f"channels must exceed one-ciphertext capacity {channel_capacity}"
        )

    rng = np.random.default_rng(int(args.seed))
    input_values = rng.normal(
        0.0,
        0.2,
        size=(1, int(args.channels), int(args.height), int(args.width)),
    ).astype(np.float64)
    weights = [
        rng.normal(
            0.0,
            0.035,
            size=(int(args.channels), int(args.channels), 3, 3),
        ).astype(np.float64)
        for _ in range(2)
    ]
    biases = [
        rng.normal(0.0, 0.01, size=(int(args.channels),)).astype(np.float64)
        for _ in range(2)
    ]

    scheme = orion.init_scheme(_config(int(args.logn)))
    Conv2d.set_scheme(scheme)
    layers = [
        _make_layer(
            name="wpc_pre_activation_conv",
            channels=int(args.channels),
            level=3,
            bsgs_ratio=float(args.bsgs_ratio),
            weight=weights[0],
            bias=biases[0],
        ),
        _make_layer(
            name="wpc_post_bootstrap_conv",
            channels=int(args.channels),
            level=3,
            bsgs_ratio=float(args.bsgs_ratio),
            weight=weights[1],
            bias=biases[1],
        ),
    ]
    plans: list[Any] = []
    bridge: WPCCIPSActivationBootstrap | None = None
    values_to_release: list[Any] = []
    payload: dict[str, Any]
    try:
        plan_compile_started = time.perf_counter()
        for layer in layers:
            plans.append(
                layer.install_wpc_cips_plan(
                    (1, int(args.channels), int(args.height), int(args.width)),
                    include_full_control=True,
                    verify_exact_qp=True,
                )
            )
            layer.he()
        plan_compile_s = float(time.perf_counter() - plan_compile_started)

        reference_first = plans[0].clear_reference(input_values)
        reference_activation = np.square(reference_first)
        reference_second = plans[1].clear_reference(
            reference_activation[None, ...]
        )
        bootstrap_bound = float(
            max(
                1.0,
                float(np.max(np.abs(reference_activation)))
                * float(args.bound_headroom),
            )
        )
        bridge = WPCCIPSActivationBootstrap.between_plans(
            plans[0],
            plans[1],
            bootstrap_bound=bootstrap_bound,
        )
        bridge_compile_started = time.perf_counter()
        bridge_compile = bridge.compile(scheme)
        bridge_compile_s = float(time.perf_counter() - bridge_compile_started)
        group_count = int(plans[0].case.input_group_count)
        expected_transform_count = int(
            sum(plan.case.transform_count for plan in plans)
        )

        input_ciphertext = plans[0].encrypt_input(input_values)
        values_to_release.append(input_ciphertext)
        online_encode_calls = 0
        original_encode = scheme.encoder.encode

        def counted_online_encode(*encode_args, **encode_kwargs):
            nonlocal online_encode_calls
            online_encode_calls += 1
            return original_encode(*encode_args, **encode_kwargs)

        scheme.encoder.encode = counted_online_encode
        try:
            bridge.clear_runtime_profile()
            _set_bootstrap_profile(scheme.backend, True)
            scheme.backend.ResetOperationCounters()
            full_started = time.perf_counter()
            full_first = plans[0].evaluate(input_ciphertext, compressed=False)
            values_to_release.append(full_first)
            full_first_row = dict(plans[0].last_evaluation)
            full_refreshed = bridge(full_first)
            values_to_release.append(full_refreshed)
            full_bridge_row = dict(bridge.last_evaluation)
            full_second = plans[1].evaluate(full_refreshed, compressed=False)
            values_to_release.append(full_second)
            full_second_row = dict(plans[1].last_evaluation)
            full_s = float(time.perf_counter() - full_started)
            full_counters = _operation_counters(scheme.backend)
            full_bootstrap_profile = _bootstrap_profile(scheme.backend)

            bridge.clear_runtime_profile()
            scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
            _set_bootstrap_profile(scheme.backend, True)
            scheme.backend.ResetOperationCounters()
            compressed_started = time.perf_counter()
            compressed_first = layers[0](input_ciphertext)
            values_to_release.append(compressed_first)
            compressed_first_row = dict(plans[0].last_evaluation)
            compressed_refreshed = bridge(compressed_first)
            values_to_release.append(compressed_refreshed)
            compressed_bridge_row = dict(bridge.last_evaluation)
            compressed_second = layers[1](compressed_refreshed)
            values_to_release.append(compressed_second)
            compressed_second_row = dict(plans[1].last_evaluation)
            compressed_s = float(time.perf_counter() - compressed_started)
            compressed_counters = _operation_counters(scheme.backend)
            compressed_bootstrap_profile = _bootstrap_profile(scheme.backend)
        finally:
            _set_bootstrap_profile(scheme.backend, False)
            scheme.encoder.encode = original_encode

        full_first_output = plans[0].decrypt_unpack(full_first)
        full_refreshed_output = plans[0].decrypt_unpack(full_refreshed)
        compressed_first_output = plans[0].decrypt_unpack(compressed_first)
        compressed_refreshed_output = plans[0].decrypt_unpack(compressed_refreshed)
        full_output = plans[1].decrypt_unpack(full_second)
        compressed_output = plans[1].decrypt_unpack(compressed_second)
        full_first_error = np.abs(full_first_output - reference_first)
        full_refresh_error = np.abs(full_refreshed_output - reference_activation)
        compressed_first_error = np.abs(compressed_first_output - reference_first)
        compressed_refresh_error = np.abs(
            compressed_refreshed_output - reference_activation
        )
        full_error = np.abs(full_output - reference_second)
        compressed_error = np.abs(compressed_output - reference_second)
        path_delta = np.abs(compressed_output - full_output)
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
        paths_match = bool(
            np.allclose(full_output, compressed_output, rtol=0.0, atol=1e-12)
        )

        global_stats = _global_stats(scheme.backend)
        storage = _aggregate_storage(plans)
        transform_rows = [
            row for plan in plans for row in plan.transform_rows.values()
        ]
        runtime_stats = [
            row for plan in plans for row in plan.compressed_stats().values()
        ]
        compressed_sequence = [
            *compressed_first_row["evaluation_sequence"],
            *compressed_second_row["evaluation_sequence"],
        ]
        exact_qp = bool(
            len(transform_rows) == int(expected_transform_count)
            and all(row["exact_qp_match"] is True for row in transform_rows)
        )
        transforms_released = bool(
            len(runtime_stats) == int(expected_transform_count)
            and all(
                int(row["evaluation_count"]) == 1
                and int(row["materialized_full_payload_bytes"]) == 0
                and int(row["weight_plaintext_online_encode_calls"]) == 0
                for row in runtime_stats
            )
        )
        sequence_isolated = bool(
            len(compressed_sequence) == int(expected_transform_count)
            and all(
                int(row["current_materialized_bytes_before"]) == 0
                and int(row["current_materialized_bytes_after"]) == 0
                and int(row["current_materialized_transforms_after"]) == 0
                for row in compressed_sequence
            )
        )
        peak_one = bool(
            int(global_stats["peak_materialized_transform_count"]) == 1
            and int(global_stats["peak_materialized_full_payload_bytes"])
            == int(global_stats["max_single_transform_full_payload_bytes"])
            and int(global_stats["current_materialized_transform_count"]) == 0
            and int(global_stats["current_materialized_full_payload_bytes"]) == 0
        )
        backend_bootstrap_valid = bool(
            full_bootstrap_profile["row_count"] == int(group_count)
            and compressed_bootstrap_profile["row_count"] == int(group_count)
            and all(
                int(row["slots"]) == int(slots)
                and int(row["input_level"]) == 0
                and int(row["output_level"]) == 3
                for profile in (
                    full_bootstrap_profile,
                    compressed_bootstrap_profile,
                )
                for row in profile["rows"]
            )
        )
        level_chain_valid = bool(
            all(
                first["input_level"] == 3
                and first["output_level"] == 2
                and middle["input_level"] == 2
                and middle["activation_output_level"] == 1
                and middle["bootstrap_output_level"] == 3
                and second["input_level"] == 3
                and second["output_level"] == 2
                for first, middle, second in (
                    (full_first_row, full_bridge_row, full_second_row),
                    (
                        compressed_first_row,
                        compressed_bridge_row,
                        compressed_second_row,
                    ),
                )
            )
        )
        signatures_preserved = bool(
            full_bridge_row["packing_signature_preserved"]
            and compressed_bridge_row["packing_signature_preserved"]
            and plans[0].output_packing_signature
            == plans[1].input_packing_signature
        )
        operations_match = bool(full_counters == compressed_counters)
        storage_valid = bool(
            int(global_stats["registered_transform_count"])
            == int(expected_transform_count)
            and int(global_stats["total_weight_plaintext_offline_encode_calls"])
            == int(expected_transform_count)
            and int(global_stats["total_weight_plaintext_online_encode_calls"]) == 0
            and int(storage["stored_weight_plus_metadata_plus_bias_bytes"])
            < int(storage["full_weight_plus_bias_payload_bytes"])
        )

        acceptance = {
            "all_group_transforms_exact_qp": exact_qp,
            "full_pipeline_correct_vs_clear": full_correct,
            "compressed_pipeline_correct_vs_clear": compressed_correct,
            "compressed_output_matches_full_control": paths_match,
            "quad_activation_executed": bool(
                full_bridge_row["activation_output_level"] == 1
                and compressed_bridge_row["activation_output_level"] == 1
            ),
            "real_lattigo_bootstrap_executed_for_each_group": backend_bootstrap_valid,
            "level_chain_3_to_2_to_1_to_bootstrap_3_to_2_valid": level_chain_valid,
            "cips_packing_signature_preserved_across_activation_bootstrap": signatures_preserved,
            "second_conv_accepts_refreshed_ciphertext_without_repacking": True,
            "zero_online_python_encode_calls": online_encode_calls == 0,
            "operation_counters_match": operations_match,
            "all_compressed_transforms_released": transforms_released,
            "materialization_isolated_between_transforms": sequence_isolated,
            "peak_materialization_is_one_transform": peak_one,
            "aggregate_storage_accounting_valid": storage_valid,
        }
        acceptance["valid"] = bool(all(acceptance.values()))

        payload = {
            "schema_version": 1,
            "profile": "wpc_cips_conv_quad_bootstrap_conv_pipeline",
            "status": "ok" if acceptance["valid"] else "invalid",
            "timing_policy": "single diagnostic execution; no performance claim",
            "seed": int(args.seed),
            "scope": {
                "pipeline": "Conv2d -> Quad -> Bootstrap -> Conv2d",
                "orion_modules": [
                    "orion.nn.Conv2d",
                    "orion.nn.Quad",
                    "orion.nn.Bootstrap",
                    "orion.nn.Conv2d",
                ],
                "convolution": "3x3_stride1_same_shape_rotation_padding",
                "channels": int(args.channels),
                "spatial_shape": [int(args.height), int(args.width)],
                "ciphertext_group_count": int(group_count),
                "group_matrix_per_convolution": [
                    int(group_count),
                    int(group_count),
                ],
                "compressed_transform_count": int(expected_transform_count),
                "limitations": [
                    "quadratic activation rather than a trained-model activation",
                    "one functional activation/bootstrap boundary",
                    "no stride, downsample reshaping, residual, or concatenation path",
                    "validation-only full transforms coexist with compressed transforms",
                    "single diagnostic timing run",
                    "no trained-model accuracy or performance claim",
                ],
            },
            "ckks": {
                "logn": int(args.logn),
                "slots": int(slots),
                "logq": [55, 45, 45, 45],
                "logp": [60],
                "bootstrap_logp": [61] * 8,
            },
            "compile": {
                "convolution_plans_s": float(plan_compile_s),
                "activation_bootstrap_bridge_s": float(bridge_compile_s),
                "activation_bootstrap_bridge": bridge_compile,
            },
            "bootstrap_range": {
                "reference_activation_min": float(np.min(reference_activation)),
                "reference_activation_max": float(np.max(reference_activation)),
                "symmetric_bound": float(bootstrap_bound),
                "headroom": float(args.bound_headroom),
            },
            "storage": storage,
            "compressed_global_stats": global_stats,
            "full_control_pipeline": {
                "correct": full_correct,
                "max_abs_error": float(np.max(full_error)),
                "mean_abs_error": float(np.mean(full_error)),
                "evaluate_s": float(full_s),
                "operation_counters": full_counters,
                "intermediate_correctness": {
                    "first_conv_max_abs_error": float(np.max(full_first_error)),
                    "activation_bootstrap_max_abs_error": float(
                        np.max(full_refresh_error)
                    ),
                },
                "first_conv": full_first_row,
                "activation_bootstrap": full_bridge_row,
                "second_conv": full_second_row,
                "backend_bootstrap_profile": full_bootstrap_profile,
            },
            "compressed_pipeline": {
                "correct": compressed_correct,
                "max_abs_error": float(np.max(compressed_error)),
                "mean_abs_error": float(np.mean(compressed_error)),
                "max_abs_delta_vs_full_control": float(np.max(path_delta)),
                "evaluate_s": float(compressed_s),
                "operation_counters": compressed_counters,
                "operation_counters_match_full_control": operations_match,
                "intermediate_correctness": {
                    "first_conv_max_abs_error": float(
                        np.max(compressed_first_error)
                    ),
                    "activation_bootstrap_max_abs_error": float(
                        np.max(compressed_refresh_error)
                    ),
                },
                "first_conv": compressed_first_row,
                "activation_bootstrap": compressed_bridge_row,
                "second_conv": compressed_second_row,
                "backend_bootstrap_profile": compressed_bootstrap_profile,
                "online_python_encode_call_count": int(online_encode_calls),
            },
            "acceptance": acceptance,
        }
    finally:
        for value in reversed(values_to_release):
            _release(value)
        if bridge is not None:
            bridge.cleanup()
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
    return 0 if payload["acceptance"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
