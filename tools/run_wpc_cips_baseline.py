#!/usr/bin/env python3
"""Run the matched WPC CIPS and compressed-Q/P correctness baseline.

This is an audit microbenchmark, not a performance result. It uses the same
input, weights, ring size, and clear rotate/multiply/accumulate evaluator for a
channel-first control and a channel-innermost CIPS layout. Every CIPS weight
message must pass exact Q/P copy-map reconstruction, then a real compressed
transform must decompress without Encode and match the full transform.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.backend.python.parameters import NewParameters
from orion.experimental.wpc_cips_baseline import (
    CIPSConvCase,
    LAYOUT_CHANNEL_FIRST,
    LAYOUT_CIPS,
    pack_tensor,
    rotation_padded_reference,
    run_clear_layout_comparison,
    unpack_output,
)
from orion.experimental.wpc_periodicity import (
    ENCODED_VERIFY_ENV,
    analyze_slot_periodicity,
    encoded_plaintext_bytes,
    encoded_qp_verifier_for_params,
    reset_global_periodicity_collector,
)


DEFAULT_OUT = (
    REPO_ROOT
    / ".tmp/results/honours/12_wpc_compressed_qp/cips_3x3_compressed_qp.json"
)

WPC_COMPRESSED_STATS_FIELDS = (
    "diagonal_count",
    "full_payload_bytes",
    "compressed_payload_bytes",
    "metadata_bytes",
    "stored_payload_plus_metadata_bytes",
    "min_slot_period",
    "max_slot_period",
    "last_decompress_nanoseconds",
    "last_evaluate_nanoseconds",
    "weight_plaintext_offline_encode_calls",
    "weight_plaintext_online_encode_calls",
    "decompression_count",
    "evaluation_count",
    "materialized_full_payload_bytes",
    "backend_schema_version",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a channel-first diagonal layout with WPC CIPS and "
            "Rotation Padding on one matched two-dimensional convolution."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-channels", type=int, default=4)
    parser.add_argument("--output-channels", type=int, default=4)
    parser.add_argument("--kernel-height", type=int, default=3)
    parser.add_argument("--kernel-width", type=int, default=3)
    parser.add_argument("--pad-height-before", type=int, default=None)
    parser.add_argument("--pad-width-before", type=int, default=None)
    parser.add_argument(
        "--skip-encoded-qp",
        action="store_true",
        help="Run only clear layout checks; output is not an accepted WPC baseline.",
    )
    parser.add_argument(
        "--skip-fhe-eval",
        action="store_true",
        help="Skip encrypted evaluation; output is not an accepted WPC baseline.",
    )
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    return parser


def _params(logn: int) -> NewParameters:
    return NewParameters(
        {
            "ckks_params": {
                "LogN": int(logn),
                "LogQ": [55, 45, 45],
                "LogP": [60],
                "LogScale": 45,
                "H": 192,
                "RingType": "Standard",
            },
            "orion": {
                "backend": "clear_lattigo",
                "io_mode": "none",
                "debug": False,
            },
        }
    )


def _decode_wpc_compressed_stats(values: list[int]) -> dict[str, Any]:
    if len(values) != len(WPC_COMPRESSED_STATS_FIELDS):
        raise RuntimeError(
            "unexpected WPC compressed-transform statistics length: "
            f"{len(values)} != {len(WPC_COMPRESSED_STATS_FIELDS)}"
        )
    result = {
        name: int(value)
        for name, value in zip(WPC_COMPRESSED_STATS_FIELDS, values)
    }
    full_bytes = int(result["full_payload_bytes"])
    compressed_bytes = int(result["compressed_payload_bytes"])
    stored_bytes = int(result["stored_payload_plus_metadata_bytes"])
    result["payload_compression_ratio"] = (
        float(full_bytes / compressed_bytes) if compressed_bytes else None
    )
    result["storage_compression_ratio_including_metadata"] = (
        float(full_bytes / stored_bytes) if stored_bytes else None
    )
    result["last_decompress_s"] = float(
        int(result["last_decompress_nanoseconds"]) / 1_000_000_000
    )
    result["last_evaluate_s"] = float(
        int(result["last_evaluate_nanoseconds"]) / 1_000_000_000
    )
    return result


def _encoded_qp_verify(
    diagonals: dict[int, np.ndarray],
    *,
    params: NewParameters,
    enabled: bool,
) -> dict[str, Any]:
    candidates = []
    for rotation, message in sorted(diagonals.items()):
        periodicity = analyze_slot_periodicity(message, payload_format="real")
        if periodicity.wpc_candidate:
            candidates.append((int(rotation), message, periodicity))

    if not enabled:
        return {
            "enabled": False,
            "candidate_count": int(len(candidates)),
            "attempted_count": 0,
            "passed_count": 0,
            "failed_count": 0,
            "complete": False,
            "failures": [],
        }

    os.environ[ENCODED_VERIFY_ENV] = "1"
    verifier = encoded_qp_verifier_for_params(params)
    if verifier is None:
        raise RuntimeError("encoded Q/P verifier did not initialize")
    level_q = int(params.get_max_level())
    results: list[dict[str, Any]] = []
    for rotation, message, periodicity in candidates:
        verification = dict(
            verifier(
                tuple(complex(float(value), 0.0) for value in message),
                periodicity,
                "real",
                level_q,
            )
        )
        results.append(
            {
                "rotation": int(rotation),
                "source_slot_period_t": int(periodicity.minimal_period),
                **verification,
            }
        )
    failures = [row for row in results if not bool(row.get("passed", False))]
    return {
        "enabled": True,
        "candidate_count": int(len(candidates)),
        "attempted_count": int(len(results)),
        "passed_count": int(len(results) - len(failures)),
        "failed_count": int(len(failures)),
        "complete": bool(len(results) == len(candidates) and not failures),
        "failures": failures,
        "records": results,
    }


def _real_fhe_evaluate(
    *,
    tensor: np.ndarray,
    reference: np.ndarray,
    case: CIPSConvCase,
    diagonals_by_layout: dict[str, dict[int, np.ndarray]],
    cips_slot_period: int,
    logn: int,
    bsgs_ratio: float,
    enabled: bool,
    atol: float = 1e-6,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "complete": False,
            "all_layouts_correct": False,
            "compressed_path_valid": False,
            "layouts": {},
        }

    import torch
    import orion

    config = {
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
    init_started = time.perf_counter()
    scheme = orion.init_scheme(config)
    init_s = float(time.perf_counter() - init_started)
    rows: dict[str, Any] = {}
    compressed_path: dict[str, Any] = {}
    try:
        required_methods = (
            "GenerateWPCCompressedLinearTransform",
            "DecompressWPCLinearTransform",
            "VerifyWPCDecompressedLinearTransformExact",
            "RemoveWPCDecompressedLinearTransform",
            "EvaluateWPCCompressedLinearTransform",
            "GetWPCCompressedLinearTransformStats",
        )
        missing = [
            name for name in required_methods if not hasattr(scheme.backend, name)
        ]
        if missing:
            raise RuntimeError(
                "Lattigo library is missing WPC compressed-Q/P APIs: "
                + ", ".join(missing)
            )

        transform_ids: dict[str, int] = {}
        compile_s: dict[str, float] = {}
        key_prepare_s: dict[str, float] = {}
        level_q = int(scheme.params.get_max_level())
        for layout in (LAYOUT_CHANNEL_FIRST, LAYOUT_CIPS):
            diagonals = diagonals_by_layout[layout]
            indices = [int(value) for value in sorted(diagonals)]
            flattened = np.concatenate(
                [np.asarray(diagonals[index], dtype=np.float32) for index in indices]
            ).tolist()
            started = time.perf_counter()
            transform_id = int(
                scheme.backend.GenerateLinearTransform(
                    indices,
                    flattened,
                    level_q,
                    float(bsgs_ratio),
                    "none",
                )
            )
            compile_s[layout] = float(time.perf_counter() - started)
            started = time.perf_counter()
            scheme.lt_evaluator.generate_rotation_keys(transform_id)
            key_prepare_s[layout] = float(time.perf_counter() - started)
            transform_ids[layout] = transform_id

        cips_diagonals = diagonals_by_layout[LAYOUT_CIPS]
        cips_indices = [int(value) for value in sorted(cips_diagonals)]
        cips_flattened = np.concatenate(
            [
                np.asarray(cips_diagonals[index], dtype=np.float32)
                for index in cips_indices
            ]
        ).tolist()
        started = time.perf_counter()
        compressed_transform_id = int(
            scheme.backend.GenerateWPCCompressedLinearTransform(
                cips_indices,
                cips_flattened,
                level_q,
                float(bsgs_ratio),
                int(cips_slot_period),
            )
        )
        compressed_generate_s = float(time.perf_counter() - started)
        started = time.perf_counter()
        scheme.lt_evaluator.generate_rotation_keys(compressed_transform_id)
        compressed_key_prepare_s = float(time.perf_counter() - started)

        stats_after_compression = _decode_wpc_compressed_stats(
            scheme.backend.GetWPCCompressedLinearTransformStats(
                compressed_transform_id
            )
        )
        started = time.perf_counter()
        manually_decompressed_count = int(
            scheme.backend.DecompressWPCLinearTransform(compressed_transform_id)
        )
        manual_decompress_call_s = float(time.perf_counter() - started)
        exact_qp_match = bool(
            int(
                scheme.backend.VerifyWPCDecompressedLinearTransformExact(
                    int(transform_ids[LAYOUT_CIPS]),
                    compressed_transform_id,
                )
            )
            == 1
        )
        stats_while_materialized = _decode_wpc_compressed_stats(
            scheme.backend.GetWPCCompressedLinearTransformStats(
                compressed_transform_id
            )
        )
        scheme.backend.RemoveWPCDecompressedLinearTransform(
            compressed_transform_id
        )
        stats_after_manual_release = _decode_wpc_compressed_stats(
            scheme.backend.GetWPCCompressedLinearTransformStats(
                compressed_transform_id
            )
        )

        ciphertext_ids: dict[str, int] = {}
        output_tensors: dict[str, np.ndarray] = {}
        for layout in (LAYOUT_CHANNEL_FIRST, LAYOUT_CIPS):
            packed = pack_tensor(tensor, case, layout)
            started = time.perf_counter()
            plaintext = scheme.encode(
                torch.tensor(packed, dtype=torch.float32),
                level=level_q,
            )
            encode_s = float(time.perf_counter() - started)
            started = time.perf_counter()
            ciphertext = scheme.encrypt(plaintext)
            encrypt_s = float(time.perf_counter() - started)
            ciphertext_ids[layout] = int(ciphertext.ids[0])
            scheme.backend.ResetOperationCounters()
            started = time.perf_counter()
            output_id = int(
                scheme.backend.EvaluateLinearTransform(
                    int(transform_ids[layout]),
                    int(ciphertext.ids[0]),
                )
            )
            evaluate_s = float(time.perf_counter() - started)
            counters = [int(value) for value in scheme.backend.GetOperationCounters()]
            started = time.perf_counter()
            decoded = np.asarray(
                scheme.backend.Decode(scheme.backend.Decrypt(output_id))[: int(case.slots)],
                dtype=np.float64,
            )
            decrypt_decode_s = float(time.perf_counter() - started)
            output = unpack_output(decoded, case, layout)
            output_tensors[layout] = output
            error = np.abs(output - reference)
            rows[layout] = {
                "correct": bool(
                    np.allclose(output, reference, rtol=0.0, atol=float(atol))
                ),
                "max_abs_error": float(np.max(error)),
                "mean_abs_error": float(np.mean(error)),
                "compile_encode_plaintexts_s": float(compile_s[layout]),
                "rotation_key_prepare_s": float(key_prepare_s[layout]),
                "input_encode_s": encode_s,
                "encrypt_s": encrypt_s,
                "evaluate_s": evaluate_s,
                "decrypt_decode_s": decrypt_decode_s,
                "operation_counters": {
                    "rotation_total": counters[0] if len(counters) > 0 else None,
                    "linear_transform_rotation": counters[1] if len(counters) > 1 else None,
                    "direct_rotation": counters[2] if len(counters) > 2 else None,
                    "conjugation": counters[3] if len(counters) > 3 else None,
                },
            }

        scheme.backend.ResetOperationCounters()
        started = time.perf_counter()
        compressed_output_id = int(
            scheme.backend.EvaluateWPCCompressedLinearTransform(
                compressed_transform_id,
                int(ciphertext_ids[LAYOUT_CIPS]),
            )
        )
        compressed_call_s = float(time.perf_counter() - started)
        compressed_counters = [
            int(value) for value in scheme.backend.GetOperationCounters()
        ]
        started = time.perf_counter()
        compressed_decoded = np.asarray(
            scheme.backend.Decode(
                scheme.backend.Decrypt(compressed_output_id)
            )[: int(case.slots)],
            dtype=np.float64,
        )
        compressed_decrypt_decode_s = float(time.perf_counter() - started)
        compressed_output = unpack_output(
            compressed_decoded,
            case,
            LAYOUT_CIPS,
        )
        compressed_error = np.abs(compressed_output - reference)
        compressed_full_delta = np.abs(
            compressed_output - output_tensors[LAYOUT_CIPS]
        )
        compressed_operation_counters = {
            "rotation_total": (
                compressed_counters[0] if len(compressed_counters) > 0 else None
            ),
            "linear_transform_rotation": (
                compressed_counters[1] if len(compressed_counters) > 1 else None
            ),
            "direct_rotation": (
                compressed_counters[2] if len(compressed_counters) > 2 else None
            ),
            "conjugation": (
                compressed_counters[3] if len(compressed_counters) > 3 else None
            ),
        }
        stats_after_evaluation = _decode_wpc_compressed_stats(
            scheme.backend.GetWPCCompressedLinearTransformStats(
                compressed_transform_id
            )
        )
        operation_counters_match = bool(
            compressed_operation_counters
            == rows[LAYOUT_CIPS]["operation_counters"]
        )
        rows["cips_compressed_qp"] = {
            "correct": bool(
                np.allclose(
                    compressed_output,
                    reference,
                    rtol=0.0,
                    atol=float(atol),
                )
            ),
            "max_abs_error": float(np.max(compressed_error)),
            "mean_abs_error": float(np.mean(compressed_error)),
            "max_abs_delta_vs_full_cips": float(np.max(compressed_full_delta)),
            "decompress_and_evaluate_call_s": compressed_call_s,
            "decrypt_decode_s": compressed_decrypt_decode_s,
            "operation_counters": compressed_operation_counters,
            "operation_counters_match_full_cips": operation_counters_match,
        }
        compressed_path_valid = bool(
            exact_qp_match
            and manually_decompressed_count
            == int(stats_after_compression["diagonal_count"])
            and int(stats_while_materialized["materialized_full_payload_bytes"])
            == int(stats_after_compression["full_payload_bytes"])
            and int(stats_after_manual_release["materialized_full_payload_bytes"])
            == 0
            and rows["cips_compressed_qp"]["correct"]
            and float(rows["cips_compressed_qp"]["max_abs_delta_vs_full_cips"])
            <= float(atol)
            and operation_counters_match
            and int(stats_after_evaluation["weight_plaintext_offline_encode_calls"])
            == 1
            and int(stats_after_evaluation["weight_plaintext_online_encode_calls"])
            == 0
            and int(stats_after_evaluation["materialized_full_payload_bytes"])
            == 0
            and int(stats_after_evaluation["compressed_payload_bytes"])
            < int(stats_after_evaluation["full_payload_bytes"])
        )
        compressed_path = {
            "valid": compressed_path_valid,
            "slot_period": int(cips_slot_period),
            "offline_generate_encode_compress_s": compressed_generate_s,
            "rotation_key_prepare_s": compressed_key_prepare_s,
            "manual_decompression_call_s": manual_decompress_call_s,
            "manual_decompressed_diagonal_count": manually_decompressed_count,
            "exact_qp_match_vs_full_transform": exact_qp_match,
            "stats_after_compression": stats_after_compression,
            "stats_while_manually_materialized": stats_while_materialized,
            "stats_after_manual_release": stats_after_manual_release,
            "stats_after_online_evaluation": stats_after_evaluation,
            "online_path": rows["cips_compressed_qp"],
            "online_encode_definition": (
                "weight-plaintext Encode calls inside decompression/evaluation; "
                "input ciphertext encoding is separate"
            ),
        }
    finally:
        scheme.delete_scheme()

    all_paths_correct = bool(
        len(rows) == 3 and all(bool(row["correct"]) for row in rows.values())
    )
    return {
        "enabled": True,
        "complete": bool(all_paths_correct and compressed_path.get("valid", False)),
        "all_layouts_correct": all_paths_correct,
        "compressed_path_valid": bool(compressed_path.get("valid", False)),
        "scheme_init_and_base_keys_s": init_s,
        "bsgs_ratio": float(bsgs_ratio),
        "correctness_atol": float(atol),
        "timing_interpretation": (
            "single diagnostic execution without warmup; not a performance comparison"
        ),
        "layouts": rows,
        "compressed_qp_storage_and_online_decompression": compressed_path,
    }


def main() -> int:
    args = _parser().parse_args()
    pad_height_before = (
        int(args.kernel_height) // 2
        if args.pad_height_before is None
        else int(args.pad_height_before)
    )
    pad_width_before = (
        int(args.kernel_width) // 2
        if args.pad_width_before is None
        else int(args.pad_width_before)
    )
    slots = 1 << (int(args.logn) - 1)
    case = CIPSConvCase(
        slots=int(slots),
        input_channels=int(args.input_channels),
        output_channels=int(args.output_channels),
        height=int(args.height),
        width=int(args.width),
        kernel_height=int(args.kernel_height),
        kernel_width=int(args.kernel_width),
        pad_height_before=int(pad_height_before),
        pad_width_before=int(pad_width_before),
    )
    params = _params(int(args.logn))
    level_q = int(params.get_max_level())
    level_p = int(len(params.get_logp()) - 1)
    full_bytes = encoded_plaintext_bytes(
        ring_degree=int(params.get_ring_degree()),
        level_q=level_q,
        level_p=level_p,
    )

    rng = np.random.default_rng(int(args.seed))
    tensor = rng.normal(
        0.0,
        0.5,
        size=(int(case.input_channels), int(case.height), int(case.width)),
    ).astype(np.float64)
    weights = rng.normal(
        0.0,
        0.25,
        size=(
            int(case.output_channels),
            int(case.input_channels),
            int(case.kernel_height),
            int(case.kernel_width),
        ),
    ).astype(np.float64)

    comparison = run_clear_layout_comparison(
        tensor,
        weights,
        case,
        full_encoded_bytes_per_diagonal=int(full_bytes),
    )
    diagonals_by_layout = {
        LAYOUT_CIPS: comparison["layouts"][LAYOUT_CIPS].pop("diagonals"),
        LAYOUT_CHANNEL_FIRST: comparison["layouts"][LAYOUT_CHANNEL_FIRST].pop(
            "diagonals"
        ),
    }
    cips_diagonals = diagonals_by_layout[LAYOUT_CIPS]
    cips_periodicity = comparison["layouts"][LAYOUT_CIPS]["periodicity"]
    cips_slot_periods = {
        int(value)
        for value in comparison["layouts"][LAYOUT_CIPS][
            "minimal_slot_period_by_rotation"
        ].values()
    }
    cips_common_slot_period = (
        next(iter(cips_slot_periods)) if len(cips_slot_periods) == 1 else 0
    )
    cips_all_nonzero_periodic = bool(
        int(cips_periodicity["periodic_count"])
        == int(cips_periodicity["nonzero_count"])
        == int(comparison["layouts"][LAYOUT_CIPS]["diagonal_count"])
        and cips_common_slot_period > 0
    )

    encoded_verification: dict[str, Any]
    try:
        encoded_verification = _encoded_qp_verify(
            cips_diagonals,
            params=params,
            enabled=not bool(args.skip_encoded_qp),
        )
    finally:
        reset_global_periodicity_collector()

    real_fhe = _real_fhe_evaluate(
        tensor=tensor,
        reference=rotation_padded_reference(tensor, weights, case),
        case=case,
        diagonals_by_layout=diagonals_by_layout,
        cips_slot_period=int(cips_common_slot_period),
        logn=int(args.logn),
        bsgs_ratio=float(args.bsgs_ratio),
        enabled=not bool(args.skip_fhe_eval),
    )

    two_dimensional_wrap_required = bool(
        int(case.kernel_width) > 1 and int(case.width) > 1
    )
    accepted = bool(
        comparison["valid"]
        and cips_all_nonzero_periodic
        and comparison["layouts"][LAYOUT_CIPS]["message_period_reconstruction_exact"]
        and encoded_verification["complete"]
        and real_fhe["complete"]
        and real_fhe["all_layouts_correct"]
        and real_fhe["compressed_path_valid"]
        and float(comparison["rotation_padding_vs_zero_padding_max_abs_delta"]) > 0.0
        and (
            not two_dimensional_wrap_required
            or bool(
                comparison["flattened_rotation_differs_from_independent_axis_wrap"]
            )
        )
    )
    payload = {
        "schema_version": 3,
        "profile": "wpc_cips_compressed_qp_functional_baseline",
        "status": "ok" if accepted else "invalid",
        "timing_policy": "correctness_and_accounting_only_not_for_performance_claims",
        "seed": int(args.seed),
        "ckks": {
            "logn": int(params.get_logn()),
            "slots": int(params.get_slots()),
            "logq": [int(value) for value in params.get_logq()],
            "logp": [int(value) for value in params.get_logp()],
            "level_q": level_q,
            "level_p": level_p,
            "full_encoded_bytes_per_diagonal": int(full_bytes),
        },
        "scope": {
            "convolution": "single_ciphertext_two_dimensional_kernel",
            "kernel_shape": [int(case.kernel_height), int(case.kernel_width)],
            "stride": [1, 1],
            "rotation_padding": (
                "algorithm2_flattened_spatial_cyclic_adjacent_same_channel"
            ),
            "wpc_compressed_qp_storage": (
                "one evaluation period per active Q/P limb with online copy-map "
                "materialization and no weight-plaintext Encode"
            ),
            "matches_wpc_paper": "Algorithms 1-2 and Figures 8-10 functional subset",
            "limitations": [
                "not an optimized homomorphic kernel",
                "no multi-ciphertext channel groups",
                "no downsample reshaping layer",
                "no performance claim",
            ],
        },
        "comparison": comparison,
        "cips_all_nonzero_messages_slot_periodic": cips_all_nonzero_periodic,
        "encoded_qp_verification": encoded_verification,
        "real_fhe_evaluation": real_fhe,
        "acceptance": {
            "both_layouts_match_rotation_padding_reference": bool(comparison["valid"]),
            "cips_all_nonzero_messages_slot_periodic": cips_all_nonzero_periodic,
            "cips_message_period_roundtrip_exact": bool(
                comparison["layouts"][LAYOUT_CIPS][
                    "message_period_reconstruction_exact"
                ]
            ),
            "cips_encoded_qp_roundtrip_complete": bool(encoded_verification["complete"]),
            "both_layouts_match_reference_under_real_fhe": bool(
                real_fhe["all_layouts_correct"]
            ),
            "compressed_qp_storage_and_online_decompression_valid": bool(
                real_fhe["compressed_path_valid"]
            ),
            "rotation_padding_boundary_change_observed": bool(
                float(comparison["rotation_padding_vs_zero_padding_max_abs_delta"]) > 0.0
            ),
            "two_dimensional_flattened_wrap_exercised": bool(
                comparison["flattened_rotation_differs_from_independent_axis_wrap"]
            ),
            "two_dimensional_flattened_wrap_required": (
                two_dimensional_wrap_required
            ),
            "valid": accepted,
        },
    }

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
