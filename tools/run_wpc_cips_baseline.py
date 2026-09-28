#!/usr/bin/env python3
"""Run the matched WPC CIPS/Rotation-Padding correctness baseline.

This is an audit microbenchmark, not a performance result. It uses the same
input, weights, ring size, and clear rotate/multiply/accumulate evaluator for a
channel-first control and a channel-innermost CIPS layout. Every CIPS weight
message is additionally encoded by real Lattigo and must pass exact Q/P
copy-map reconstruction.
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


DEFAULT_OUT = REPO_ROOT / ".tmp/results/honours/11_wpc_cips_baseline/cips_baseline.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a channel-first diagonal layout with WPC CIPS and "
            "Rotation Padding on one matched vertical-convolution workload."
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
    parser.add_argument("--pad-height-before", type=int, default=None)
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
    try:
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
    finally:
        scheme.delete_scheme()

    return {
        "enabled": True,
        "complete": bool(len(rows) == 2),
        "all_layouts_correct": bool(
            len(rows) == 2 and all(bool(row["correct"]) for row in rows.values())
        ),
        "scheme_init_and_base_keys_s": init_s,
        "bsgs_ratio": float(bsgs_ratio),
        "correctness_atol": float(atol),
        "timing_interpretation": (
            "single diagnostic execution without warmup; not a performance comparison"
        ),
        "layouts": rows,
    }


def main() -> int:
    args = _parser().parse_args()
    pad_before = (
        int(args.kernel_height) // 2
        if args.pad_height_before is None
        else int(args.pad_height_before)
    )
    slots = 1 << (int(args.logn) - 1)
    case = CIPSConvCase(
        slots=int(slots),
        input_channels=int(args.input_channels),
        output_channels=int(args.output_channels),
        height=int(args.height),
        width=int(args.width),
        kernel_height=int(args.kernel_height),
        pad_height_before=int(pad_before),
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
            1,
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
    cips_all_nonzero_periodic = bool(
        int(cips_periodicity["periodic_count"])
        == int(cips_periodicity["nonzero_count"])
        == int(comparison["layouts"][LAYOUT_CIPS]["diagonal_count"])
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
        logn=int(args.logn),
        bsgs_ratio=float(args.bsgs_ratio),
        enabled=not bool(args.skip_fhe_eval),
    )

    accepted = bool(
        comparison["valid"]
        and cips_all_nonzero_periodic
        and comparison["layouts"][LAYOUT_CIPS]["message_period_reconstruction_exact"]
        and encoded_verification["complete"]
        and real_fhe["complete"]
        and real_fhe["all_layouts_correct"]
        and float(comparison["rotation_padding_vs_zero_padding_max_abs_delta"]) > 0.0
    )
    payload = {
        "schema_version": 1,
        "profile": "wpc_cips_rotation_padding_functional_baseline",
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
            "convolution": "single_ciphertext_vertical_kernel",
            "kernel_width": 1,
            "stride": [1, 1],
            "rotation_padding": "adjacent_same_channel_circular_height",
            "matches_wpc_paper": "Algorithms 1-2 and Figures 8-10 functional subset",
            "limitations": [
                "not an optimized homomorphic kernel",
                "no width-kernel row-boundary transform",
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
            "rotation_padding_boundary_change_observed": bool(
                float(comparison["rotation_padding_vs_zero_padding_max_abs_delta"]) > 0.0
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
