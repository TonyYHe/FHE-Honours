#!/usr/bin/env python3
"""Validate WPC compressed Q/P execution across ciphertext channel groups.

This is a correctness and lifecycle-accounting experiment. It compiles one
CIPS linear transform for every (output group, input group) pair, evaluates
the full-Q/P and compressed-Q/P paths, accumulates partial ciphertexts, and
checks that compressed transforms are materialized and released sequentially.
The recorded single-run timings are diagnostic and are not performance claims.
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
    CIPSMultiGroupConvCase,
    LAYOUT_CIPS,
    pack_tensor,
    rotation_padded_reference_multi_group,
    run_clear_cips_group_comparison,
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
    / ".tmp/results/honours/13_wpc_multigroup/cips_multigroup_compressed_qp.json"
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

WPC_GLOBAL_STATS_FIELDS = (
    "registered_transform_count",
    "aggregate_full_payload_bytes",
    "aggregate_compressed_payload_bytes",
    "aggregate_metadata_bytes",
    "aggregate_stored_payload_plus_metadata_bytes",
    "current_materialized_full_payload_bytes",
    "peak_materialized_full_payload_bytes",
    "current_materialized_transform_count",
    "peak_materialized_transform_count",
    "max_single_transform_full_payload_bytes",
    "total_weight_plaintext_offline_encode_calls",
    "total_weight_plaintext_online_encode_calls",
    "backend_schema_version",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate sequential WPC compressed-Q/P execution and ciphertext "
            "accumulation across multiple CIPS channel groups."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--output-channels", type=int, default=12)
    parser.add_argument("--kernel-height", type=int, default=3)
    parser.add_argument("--kernel-width", type=int, default=3)
    parser.add_argument("--pad-height-before", type=int, default=None)
    parser.add_argument("--pad-width-before", type=int, default=None)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument(
        "--skip-encoded-qp",
        action="store_true",
        help="Skip exact encoded Q/P verification; output cannot be accepted.",
    )
    parser.add_argument(
        "--skip-fhe-eval",
        action="store_true",
        help="Skip real FHE execution; output cannot be accepted.",
    )
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


def _decode_stats(values: list[int], fields: tuple[str, ...]) -> dict[str, Any]:
    if len(values) != len(fields):
        raise RuntimeError(
            f"unexpected statistics length: {len(values)} != {len(fields)}"
        )
    result: dict[str, Any] = {
        name: int(value) for name, value in zip(fields, values)
    }
    if "full_payload_bytes" in result:
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
    if "aggregate_full_payload_bytes" in result:
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


def _flatten_diagonals(diagonals: dict[int, np.ndarray]) -> tuple[list[int], list[float]]:
    indices = [int(value) for value in sorted(diagonals)]
    flattened = np.concatenate(
        [np.asarray(diagonals[index], dtype=np.float32) for index in indices]
    ).tolist()
    return indices, flattened


def _group_order(case: CIPSMultiGroupConvCase) -> list[str]:
    return [
        f"out{output_group}_in{input_group}"
        for output_group in range(int(case.output_group_count))
        for input_group in range(int(case.input_group_count))
    ]


def _common_slot_period(diagonals: dict[int, np.ndarray]) -> int:
    periods = {
        int(analyze_slot_periodicity(message, payload_format="real").minimal_period)
        for message in diagonals.values()
    }
    return next(iter(periods)) if len(periods) == 1 else 0


def _encoded_qp_verify_groups(
    diagonals_by_group: dict[str, dict[int, np.ndarray]],
    *,
    params: NewParameters,
    enabled: bool,
) -> dict[str, Any]:
    candidates: list[tuple[str, int, np.ndarray, Any]] = []
    for group_key, diagonals in diagonals_by_group.items():
        for rotation, message in sorted(diagonals.items()):
            periodicity = analyze_slot_periodicity(message, payload_format="real")
            if periodicity.wpc_candidate:
                candidates.append((group_key, int(rotation), message, periodicity))

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
    records: list[dict[str, Any]] = []
    for group_key, rotation, message, periodicity in candidates:
        verification = dict(
            verifier(
                tuple(complex(float(value), 0.0) for value in message),
                periodicity,
                "real",
                level_q,
            )
        )
        records.append(
            {
                "group_key": group_key,
                "rotation": int(rotation),
                "source_slot_period_t": int(periodicity.minimal_period),
                **verification,
            }
        )
    failures = [row for row in records if not bool(row.get("passed", False))]
    return {
        "enabled": True,
        "candidate_count": int(len(candidates)),
        "attempted_count": int(len(records)),
        "passed_count": int(len(records) - len(failures)),
        "failed_count": int(len(failures)),
        "complete": bool(len(records) == len(candidates) and not failures),
        "failures": failures,
        "records": records,
    }


def _operation_counters(backend: Any) -> dict[str, int | None]:
    values = [int(value) for value in backend.GetOperationCounters()]
    return {
        "rotation_total": values[0] if len(values) > 0 else None,
        "linear_transform_rotation": values[1] if len(values) > 1 else None,
        "direct_rotation": values[2] if len(values) > 2 else None,
        "conjugation": values[3] if len(values) > 3 else None,
    }


def _decode_output_groups(
    *,
    scheme: Any,
    output_ids: list[int],
    case: CIPSMultiGroupConvCase,
) -> np.ndarray:
    groups: list[np.ndarray] = []
    for output_group, output_id in enumerate(output_ids):
        local_case = case.local_case(output_group, 0)
        decoded = np.asarray(
            scheme.backend.Decode(scheme.backend.Decrypt(int(output_id)))[
                : int(case.slots)
            ],
            dtype=np.float64,
        )
        groups.append(unpack_output(decoded, local_case, LAYOUT_CIPS))
    return np.concatenate(groups, axis=0)


def _evaluate_group_matrix(
    *,
    backend: Any,
    case: CIPSMultiGroupConvCase,
    transform_ids: dict[str, int],
    input_ciphertext_ids: list[int],
    compressed: bool,
) -> tuple[list[int], int, list[dict[str, Any]]]:
    output_ids: list[int] = []
    accumulation_add_count = 0
    sequence: list[dict[str, Any]] = []
    for output_group in range(int(case.output_group_count)):
        accumulated_id: int | None = None
        for input_group in range(int(case.input_group_count)):
            key = f"out{output_group}_in{input_group}"
            if compressed:
                before = _decode_stats(
                    backend.GetWPCCompressedGlobalStats(),
                    WPC_GLOBAL_STATS_FIELDS,
                )
                partial_id = int(
                    backend.EvaluateWPCCompressedLinearTransform(
                        int(transform_ids[key]),
                        int(input_ciphertext_ids[input_group]),
                    )
                )
                after = _decode_stats(
                    backend.GetWPCCompressedGlobalStats(),
                    WPC_GLOBAL_STATS_FIELDS,
                )
                sequence.append(
                    {
                        "group_key": key,
                        "current_materialized_bytes_before": int(
                            before["current_materialized_full_payload_bytes"]
                        ),
                        "current_materialized_bytes_after": int(
                            after["current_materialized_full_payload_bytes"]
                        ),
                        "current_materialized_transforms_after": int(
                            after["current_materialized_transform_count"]
                        ),
                    }
                )
            else:
                partial_id = int(
                    backend.EvaluateLinearTransform(
                        int(transform_ids[key]),
                        int(input_ciphertext_ids[input_group]),
                    )
                )
            if accumulated_id is None:
                accumulated_id = partial_id
            else:
                backend.AddCiphertext(int(accumulated_id), int(partial_id))
                backend.DeleteCiphertext(int(partial_id))
                accumulation_add_count += 1
        if accumulated_id is None:
            raise RuntimeError(f"output group {output_group} had no partial results")
        output_ids.append(int(accumulated_id))
    return output_ids, int(accumulation_add_count), sequence


def _real_fhe_multigroup(
    *,
    tensor: np.ndarray,
    reference: np.ndarray,
    case: CIPSMultiGroupConvCase,
    diagonals_by_group: dict[str, dict[int, np.ndarray]],
    slot_period_by_group: dict[str, int],
    logn: int,
    bsgs_ratio: float,
    enabled: bool,
    atol: float = 1e-6,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "complete": False,
            "valid": False,
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
    try:
        required_methods = (
            "GenerateWPCCompressedLinearTransform",
            "DecompressWPCLinearTransform",
            "VerifyWPCDecompressedLinearTransformExact",
            "RemoveWPCDecompressedLinearTransform",
            "EvaluateWPCCompressedLinearTransform",
            "GetWPCCompressedLinearTransformStats",
            "ResetWPCCompressedGlobalMaterializationPeak",
            "GetWPCCompressedGlobalStats",
        )
        missing = [name for name in required_methods if not hasattr(scheme.backend, name)]
        if missing:
            raise RuntimeError(
                "Lattigo library is missing WPC multi-group APIs: "
                + ", ".join(missing)
            )

        level_q = int(scheme.params.get_max_level())
        order = _group_order(case)
        full_transform_ids: dict[str, int] = {}
        compressed_transform_ids: dict[str, int] = {}
        transform_rows: dict[str, dict[str, Any]] = {}
        for key in order:
            indices, flattened = _flatten_diagonals(diagonals_by_group[key])
            started = time.perf_counter()
            full_id = int(
                scheme.backend.GenerateLinearTransform(
                    indices,
                    flattened,
                    level_q,
                    float(bsgs_ratio),
                    "none",
                )
            )
            full_generate_s = float(time.perf_counter() - started)
            scheme.lt_evaluator.generate_rotation_keys(full_id)

            started = time.perf_counter()
            compressed_id = int(
                scheme.backend.GenerateWPCCompressedLinearTransform(
                    indices,
                    flattened,
                    level_q,
                    float(bsgs_ratio),
                    int(slot_period_by_group[key]),
                )
            )
            compressed_generate_s = float(time.perf_counter() - started)
            scheme.lt_evaluator.generate_rotation_keys(compressed_id)
            full_transform_ids[key] = full_id
            compressed_transform_ids[key] = compressed_id

            stats_after_compression = _decode_stats(
                scheme.backend.GetWPCCompressedLinearTransformStats(compressed_id),
                WPC_COMPRESSED_STATS_FIELDS,
            )
            manual_count = int(
                scheme.backend.DecompressWPCLinearTransform(compressed_id)
            )
            exact_qp_match = bool(
                int(
                    scheme.backend.VerifyWPCDecompressedLinearTransformExact(
                        full_id,
                        compressed_id,
                    )
                )
                == 1
            )
            stats_materialized = _decode_stats(
                scheme.backend.GetWPCCompressedLinearTransformStats(compressed_id),
                WPC_COMPRESSED_STATS_FIELDS,
            )
            scheme.backend.RemoveWPCDecompressedLinearTransform(compressed_id)
            stats_released = _decode_stats(
                scheme.backend.GetWPCCompressedLinearTransformStats(compressed_id),
                WPC_COMPRESSED_STATS_FIELDS,
            )
            transform_rows[key] = {
                "full_generate_encode_s": full_generate_s,
                "compressed_generate_encode_s": compressed_generate_s,
                "slot_period": int(slot_period_by_group[key]),
                "manual_decompressed_diagonal_count": manual_count,
                "exact_qp_match_vs_full_transform": exact_qp_match,
                "stats_after_compression": stats_after_compression,
                "stats_while_manually_materialized": stats_materialized,
                "stats_after_manual_release": stats_released,
            }

        global_after_registration = _decode_stats(
            scheme.backend.GetWPCCompressedGlobalStats(),
            WPC_GLOBAL_STATS_FIELDS,
        )

        # Keep the Python ciphertext wrappers alive for the entire group
        # evaluation. Their finalizers own and delete the backend IDs.
        input_ciphertexts: list[Any] = []
        input_ciphertext_ids: list[int] = []
        input_encode_encrypt_s = 0.0
        for input_group, (input_start, input_end) in enumerate(
            case.input_group_ranges
        ):
            local_case = case.local_case(0, input_group)
            packed = pack_tensor(
                tensor[input_start:input_end],
                local_case,
                LAYOUT_CIPS,
            )
            started = time.perf_counter()
            plaintext = scheme.encode(
                torch.tensor(packed, dtype=torch.float32),
                level=level_q,
            )
            ciphertext = scheme.encrypt(plaintext)
            input_encode_encrypt_s += float(time.perf_counter() - started)
            input_ciphertexts.append(ciphertext)
            input_ciphertext_ids.append(int(ciphertext.ids[0]))

        scheme.backend.ResetOperationCounters()
        started = time.perf_counter()
        full_output_ids, full_add_count, _ = _evaluate_group_matrix(
            backend=scheme.backend,
            case=case,
            transform_ids=full_transform_ids,
            input_ciphertext_ids=input_ciphertext_ids,
            compressed=False,
        )
        full_evaluate_s = float(time.perf_counter() - started)
        full_counters = _operation_counters(scheme.backend)
        full_output = _decode_output_groups(
            scheme=scheme,
            output_ids=full_output_ids,
            case=case,
        )

        scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
        global_before_online = _decode_stats(
            scheme.backend.GetWPCCompressedGlobalStats(),
            WPC_GLOBAL_STATS_FIELDS,
        )
        scheme.backend.ResetOperationCounters()
        started = time.perf_counter()
        compressed_output_ids, compressed_add_count, evaluation_sequence = (
            _evaluate_group_matrix(
                backend=scheme.backend,
                case=case,
                transform_ids=compressed_transform_ids,
                input_ciphertext_ids=input_ciphertext_ids,
                compressed=True,
            )
        )
        compressed_evaluate_s = float(time.perf_counter() - started)
        compressed_counters = _operation_counters(scheme.backend)
        compressed_output = _decode_output_groups(
            scheme=scheme,
            output_ids=compressed_output_ids,
            case=case,
        )
        global_after_online = _decode_stats(
            scheme.backend.GetWPCCompressedGlobalStats(),
            WPC_GLOBAL_STATS_FIELDS,
        )
        for key in order:
            transform_rows[key]["stats_after_online_evaluation"] = _decode_stats(
                scheme.backend.GetWPCCompressedLinearTransformStats(
                    compressed_transform_ids[key]
                ),
                WPC_COMPRESSED_STATS_FIELDS,
            )

        full_error = np.abs(full_output - reference)
        compressed_error = np.abs(compressed_output - reference)
        path_delta = np.abs(compressed_output - full_output)
        expected_add_count = int(
            case.output_group_count * max(0, case.input_group_count - 1)
        )
        all_exact = bool(
            all(
                bool(row["exact_qp_match_vs_full_transform"])
                and int(row["manual_decompressed_diagonal_count"])
                == int(row["stats_after_compression"]["diagonal_count"])
                and int(
                    row["stats_while_manually_materialized"][
                        "materialized_full_payload_bytes"
                    ]
                )
                == int(row["stats_after_compression"]["full_payload_bytes"])
                and int(
                    row["stats_after_manual_release"][
                        "materialized_full_payload_bytes"
                    ]
                )
                == 0
                for row in transform_rows.values()
            )
        )
        all_released = bool(
            all(
                int(
                    row["stats_after_online_evaluation"][
                        "materialized_full_payload_bytes"
                    ]
                )
                == 0
                and int(
                    row["stats_after_online_evaluation"]["evaluation_count"]
                )
                == 1
                and int(
                    row["stats_after_online_evaluation"][
                        "weight_plaintext_offline_encode_calls"
                    ]
                )
                == 1
                and int(
                    row["stats_after_online_evaluation"][
                        "weight_plaintext_online_encode_calls"
                    ]
                )
                == 0
                for row in transform_rows.values()
            )
        )
        sequence_released = bool(
            len(evaluation_sequence) == int(case.transform_count)
            and all(
                int(row["current_materialized_bytes_before"]) == 0
                and int(row["current_materialized_bytes_after"]) == 0
                and int(row["current_materialized_transforms_after"]) == 0
                for row in evaluation_sequence
            )
        )
        sequential_peak_valid = bool(
            int(global_after_online["current_materialized_full_payload_bytes"])
            == 0
            and int(global_after_online["current_materialized_transform_count"])
            == 0
            and int(global_after_online["peak_materialized_transform_count"])
            == 1
            and int(global_after_online["peak_materialized_full_payload_bytes"])
            == int(global_after_online["max_single_transform_full_payload_bytes"])
            and int(global_after_online["peak_materialized_full_payload_bytes"])
            < int(global_after_online["aggregate_full_payload_bytes"])
        )
        storage_valid = bool(
            int(global_after_online["aggregate_full_payload_bytes"])
            == sum(
                int(row["stats_after_compression"]["full_payload_bytes"])
                for row in transform_rows.values()
            )
            and int(global_after_online["aggregate_compressed_payload_bytes"])
            == sum(
                int(row["stats_after_compression"]["compressed_payload_bytes"])
                for row in transform_rows.values()
            )
            and int(global_after_online["aggregate_metadata_bytes"])
            == sum(
                int(row["stats_after_compression"]["metadata_bytes"])
                for row in transform_rows.values()
            )
            and int(global_after_online["registered_transform_count"])
            == int(case.transform_count)
            and int(global_after_online["aggregate_compressed_payload_bytes"])
            < int(global_after_online["aggregate_full_payload_bytes"])
            and int(
                global_after_online["total_weight_plaintext_offline_encode_calls"]
            )
            == int(case.transform_count)
            and int(
                global_after_online["total_weight_plaintext_online_encode_calls"]
            )
            == 0
        )
        full_correct = bool(
            np.allclose(full_output, reference, rtol=0.0, atol=float(atol))
        )
        compressed_correct = bool(
            np.allclose(compressed_output, reference, rtol=0.0, atol=float(atol))
        )
        counters_match = bool(full_counters == compressed_counters)
        additions_match = bool(
            full_add_count == compressed_add_count == expected_add_count
        )
        valid = bool(
            all_exact
            and all_released
            and sequence_released
            and sequential_peak_valid
            and storage_valid
            and full_correct
            and compressed_correct
            and float(np.max(path_delta)) == 0.0
            and counters_match
            and additions_match
        )
        return {
            "enabled": True,
            "complete": valid,
            "valid": valid,
            "scheme_init_and_base_keys_s": init_s,
            "input_group_encode_encrypt_s": input_encode_encrypt_s,
            "correctness_atol": float(atol),
            "transform_count": int(case.transform_count),
            "full_qp_path": {
                "correct": full_correct,
                "max_abs_error": float(np.max(full_error)),
                "mean_abs_error": float(np.mean(full_error)),
                "evaluate_and_accumulate_s": full_evaluate_s,
                "ciphertext_accumulation_add_count": int(full_add_count),
                "operation_counters": full_counters,
            },
            "compressed_qp_path": {
                "correct": compressed_correct,
                "max_abs_error": float(np.max(compressed_error)),
                "mean_abs_error": float(np.mean(compressed_error)),
                "max_abs_delta_vs_full_qp": float(np.max(path_delta)),
                "evaluate_decompress_accumulate_s": compressed_evaluate_s,
                "ciphertext_accumulation_add_count": int(compressed_add_count),
                "operation_counters": compressed_counters,
                "operation_counters_match_full_qp": counters_match,
            },
            "exact_qp_all_transforms": all_exact,
            "all_transforms_released_after_online_evaluation": all_released,
            "evaluation_sequence_released_between_transforms": sequence_released,
            "sequential_peak_limited_to_one_transform": sequential_peak_valid,
            "aggregate_storage_valid": storage_valid,
            "global_stats_after_registration": global_after_registration,
            "global_stats_before_online_evaluation": global_before_online,
            "global_stats_after_online_evaluation": global_after_online,
            "evaluation_sequence": evaluation_sequence,
            "transforms": transform_rows,
            "timing_interpretation": (
                "single diagnostic execution without warmup; not a performance comparison"
            ),
        }
    finally:
        scheme.delete_scheme()


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
    case = CIPSMultiGroupConvCase(
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
    if case.input_group_count < 2 or case.output_group_count < 2:
        raise SystemExit(
            "multi-group baseline requires at least two input and two output groups"
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

    clear = run_clear_cips_group_comparison(
        tensor,
        weights,
        case,
        full_encoded_bytes_per_diagonal=int(full_bytes),
    )
    diagonals_by_group: dict[str, dict[int, np.ndarray]] = {}
    for key, row in clear["transforms"].items():
        diagonals_by_group[key] = row.pop("diagonals")
    slot_period_by_group = {
        key: int(_common_slot_period(diagonals))
        for key, diagonals in diagonals_by_group.items()
    }
    all_group_periods_valid = bool(
        all(0 < period < int(case.slots) for period in slot_period_by_group.values())
        and bool(clear["all_transforms_proper_periodic"])
        and bool(clear["all_message_period_roundtrips_exact"])
    )
    total_diagonal_count = int(
        sum(int(row["diagonal_count"]) for row in clear["transforms"].values())
    )

    try:
        encoded_verification = _encoded_qp_verify_groups(
            diagonals_by_group,
            params=params,
            enabled=not bool(args.skip_encoded_qp),
        )
    finally:
        reset_global_periodicity_collector()

    reference = rotation_padded_reference_multi_group(tensor, weights, case)
    real_fhe = _real_fhe_multigroup(
        tensor=tensor,
        reference=reference,
        case=case,
        diagonals_by_group=diagonals_by_group,
        slot_period_by_group=slot_period_by_group,
        logn=int(args.logn),
        bsgs_ratio=float(args.bsgs_ratio),
        enabled=not bool(args.skip_fhe_eval),
    )
    accepted = bool(
        clear["correct"]
        and int(clear["transform_count"]) == int(case.transform_count)
        and all_group_periods_valid
        and encoded_verification["complete"]
        and int(encoded_verification["candidate_count"]) == total_diagonal_count
        and real_fhe["complete"]
        and real_fhe["valid"]
    )
    payload = {
        "schema_version": 1,
        "profile": "wpc_cips_multigroup_compressed_qp_functional_baseline",
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
            "convolution": "multi_ciphertext_channel_group_3x3_cips",
            "group_matrix": [
                int(case.output_group_count),
                int(case.input_group_count),
            ],
            "transform_count": int(case.transform_count),
            "online_lifecycle": (
                "sequential_decompress_evaluate_release_then_ciphertext_accumulate"
            ),
            "limitations": [
                "functional microbenchmark rather than a complete neural network",
                "comparison full transforms coexist only for correctness validation",
                "logical Q/P payload accounting rather than process RSS",
                "no downsample reshaping layer",
                "no performance claim",
            ],
        },
        "case": case.to_dict(),
        "clear_grouped_evaluation": clear,
        "slot_period_by_group": slot_period_by_group,
        "all_group_periods_valid": all_group_periods_valid,
        "total_group_diagonal_count": total_diagonal_count,
        "encoded_qp_verification": encoded_verification,
        "real_fhe_evaluation": real_fhe,
        "acceptance": {
            "clear_grouped_accumulation_matches_global_reference": bool(
                clear["correct"]
            ),
            "all_group_transforms_proper_periodic": all_group_periods_valid,
            "all_group_encoded_qp_roundtrips_complete": bool(
                encoded_verification["complete"]
                and int(encoded_verification["candidate_count"])
                == total_diagonal_count
            ),
            "all_group_qp_reconstructions_exact": bool(
                real_fhe.get("exact_qp_all_transforms", False)
            ),
            "full_and_compressed_fhe_outputs_correct": bool(
                real_fhe.get("complete", False)
            ),
            "sequential_peak_limited_to_one_transform": bool(
                real_fhe.get("sequential_peak_limited_to_one_transform", False)
            ),
            "zero_online_weight_plaintext_encode": bool(
                real_fhe.get("aggregate_storage_valid", False)
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
