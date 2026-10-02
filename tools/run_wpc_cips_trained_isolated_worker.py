#!/usr/bin/env python3
"""Run one isolated full or compressed trained-decoder FHE worker."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import orion
from orion.experimental.wpc_cips_benchmark import summarize_samples
from orion.experimental.wpc_cips_branches import WPCCIPSConcatPlan
from orion.experimental.wpc_cips_checkpoint import (
    CheckpointChebyshevSpec,
    WPCCIPSTrainedActivationBootstrap,
)
from orion.experimental.wpc_cips_layer import (
    WPC_STORAGE_COMPRESSED,
    WPC_STORAGE_FULL,
)
from orion.experimental.wpc_cips_trained_benchmark import (
    TRAINED_DECODER_TRANSFORM_COUNT,
)
from orion.experimental.wpc_cips_upsample import WPCCIPSConvTranspose2dPlan
from orion.nn import Concat, Conv2d, ConvTranspose2d
from tools.run_wpc_cips_isolated_worker import (
    _collect_runtime_memory,
    _global_stats,
    _memory_snapshot,
    _operation_counters,
    _per_forward,
    _write_phase,
)
from tools.run_wpc_cips_trained_decoder import (
    DEFAULT_CHECKPOINT,
    _aggregate_storage,
    _config,
    _encrypt_packed,
    _load_checkpoint,
    _make_conv,
    _make_up,
    _sha256,
    _torch_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "compressed"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--feature-std", type=float, default=0.02)
    parser.add_argument("--bound-headroom", type=float, default=1.25)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--forward-runs", type=int, default=10)
    parser.add_argument("--atol", type=float, default=2e-3)
    return parser


def _run_decoder(
    up_layer: ConvTranspose2d,
    concat_module: Concat,
    dec1a_layer: Conv2d,
    bridge: WPCCIPSTrainedActivationBootstrap,
    dec1b_layer: Conv2d,
    low_ciphertext: Any,
    skip_ciphertext: Any,
):
    up = up_layer(low_ciphertext)
    try:
        concat = concat_module(up, skip_ciphertext)
    finally:
        up.release()
    try:
        dec1a = dec1a_layer(concat)
    finally:
        concat.release()
    try:
        activated = bridge(dec1a)
    finally:
        dec1a.release()
    try:
        return dec1b_layer(activated)
    finally:
        activated.release()


def _logical_storage(
    plans: list[Any],
    concat_plan: WPCCIPSConcatPlan,
) -> dict[str, Any]:
    result = _aggregate_storage(plans, concat_plan)
    result["storage_mode"] = str(plans[0].storage_mode)
    result["logical_resident_total_bytes"] = int(
        result["stored_weight_plus_metadata_plus_bias_bytes"]
        + result["concat_full_qp_payload_bytes"]
    )
    result["logical_full_reference_total_bytes"] = int(
        result["full_weight_plus_bias_payload_bytes"]
        + result["concat_full_qp_payload_bytes"]
    )
    return result


def main() -> int:
    args = _parser().parse_args()
    if int(args.warmup_runs) < 0 or int(args.forward_runs) <= 0:
        raise SystemExit("warmup-runs must be nonnegative and forward-runs positive")
    if float(args.bound_headroom) < 1.0:
        raise SystemExit("bound-headroom must be at least one")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
    if int(args.logn) != 10 or (int(args.height), int(args.width)) != (8, 8):
        raise SystemExit("trained decoder benchmark requires logn=10 and 8x8 output")

    _write_phase(args.phase_file, "startup")
    state, model_metadata = _load_checkpoint(checkpoint_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    activation_spec = CheckpointChebyshevSpec.from_state_dict(
        state, "dec1a_act"
    )
    slots = 1 << (int(args.logn) - 1)
    low_shape = (1, 64, int(args.height) // 2, int(args.width) // 2)
    high_shape = (1, 32, int(args.height), int(args.width))
    concat_shape = (1, 64, int(args.height), int(args.width))
    rng = np.random.default_rng(int(args.seed))
    low = rng.normal(
        0.0, float(args.feature_std), size=low_shape
    ).astype(np.float64)
    skip = rng.normal(
        0.0, float(args.feature_std), size=high_shape
    ).astype(np.float64)
    torch_reference, _native_reference = _torch_reference(
        state,
        activation_spec,
        low,
        skip,
    )

    scheme = None
    up_layer = None
    dec1a_layer = None
    dec1b_layer = None
    concat_module = None
    bridge = None
    low_ciphertext = None
    skip_ciphertext = None
    final_output = None
    payload: dict[str, Any] | None = None
    try:
        _write_phase(args.phase_file, "scheme_init")
        scheme_started = time.perf_counter()
        scheme = orion.init_scheme(_config(int(args.logn)))
        scheme_init_s = float(time.perf_counter() - scheme_started)
        Conv2d.set_scheme(scheme)
        ConvTranspose2d.set_scheme(scheme)
        Concat.set_scheme(scheme)
        memory: dict[str, Any] = {
            "after_scheme_init": _memory_snapshot(scheme.backend)
        }

        up_layer = _make_up(
            state,
            level=8,
            bsgs_ratio=float(args.bsgs_ratio),
        )
        dec1a_layer = _make_conv(
            state,
            "dec1a",
            input_channels=64,
            output_channels=32,
            level=6,
            bsgs_ratio=float(args.bsgs_ratio),
        )
        dec1b_layer = _make_conv(
            state,
            "dec1b",
            input_channels=32,
            output_channels=32,
            level=8,
            bsgs_ratio=float(args.bsgs_ratio),
        )
        concat_module = Concat(dim=1, bsgs_ratio=float(args.bsgs_ratio))
        concat_module.name = "checkpoint_cat1"

        _write_phase(args.phase_file, "compile")
        compile_started = time.perf_counter()
        up_plan = up_layer.install_wpc_cips_plan(
            low_shape,
            storage_mode=str(args.mode),
            include_full_control=False,
            verify_exact_qp=False,
        )
        if not isinstance(up_plan, WPCCIPSConvTranspose2dPlan):
            raise RuntimeError("up1 did not install a transposed-convolution plan")
        dec1a_plan = dec1a_layer.install_wpc_cips_plan(
            concat_shape,
            storage_mode=str(args.mode),
            include_full_control=False,
            verify_exact_qp=False,
        )
        dec1b_plan = dec1b_layer.install_wpc_cips_plan(
            high_shape,
            storage_mode=str(args.mode),
            include_full_control=False,
            verify_exact_qp=False,
        )
        plans = [up_plan, dec1a_plan, dec1b_plan]
        skip_contract = SimpleNamespace(
            output_shape=high_shape,
            output_packing_signature=up_plan.output_packing_signature,
            output_level=7,
        )
        concat_plan = concat_module.install_wpc_cips_plan(
            (up_plan, skip_contract),
            consumer_plan=dec1a_plan,
        )
        if not isinstance(concat_plan, WPCCIPSConcatPlan):
            raise RuntimeError("cat1 did not install a CIPS concat plan")

        clear_up = up_plan.clear_reference(low)
        clear_concat = np.concatenate((clear_up, skip[0]), axis=0)
        clear_dec1a = dec1a_plan.clear_reference(clear_concat[None, ...])
        clear_activation = (
            activation_spec.evaluate(
                torch.tensor(clear_dec1a[None], dtype=torch.float64)
            )
            .numpy()[0]
        )
        clear_dec1b = dec1b_plan.clear_reference(clear_activation[None, ...])
        oracle_max_abs_delta = float(
            np.max(np.abs(clear_dec1b - torch_reference["dec1b"]))
        )
        bound = max(
            1.0,
            float(np.max(np.abs(clear_activation)))
            * float(args.bound_headroom),
        )
        bridge = WPCCIPSTrainedActivationBootstrap(
            logical_shape=high_shape,
            packing_signature=dec1a_plan.output_packing_signature,
            input_level=int(dec1a_plan.output_level),
            output_level=int(dec1b_plan.level),
            activation_spec=activation_spec,
            bootstrap_bound=bound,
        )
        bridge_compile = bridge.compile(scheme)
        for layer in (up_layer, dec1a_layer, dec1b_layer):
            layer.he()
        concat_module.he()
        compile_s = float(time.perf_counter() - compile_started)

        _write_phase(args.phase_file, "post_compile_raw")
        memory["post_compile_raw"] = _memory_snapshot(scheme.backend)
        _collect_runtime_memory(scheme.backend)
        _write_phase(args.phase_file, "post_compile_gc")
        memory["post_compile_gc"] = _memory_snapshot(scheme.backend)

        _write_phase(args.phase_file, "input_prepare")
        low_ciphertext = up_plan.encrypt_input(low)
        skip_ciphertext = _encrypt_packed(
            scheme,
            skip[0],
            up_plan.output_packing_signature,
            level=7,
        )
        _collect_runtime_memory(scheme.backend)
        memory["after_input_gc"] = _memory_snapshot(scheme.backend)

        online_encode_call_count = 0
        original_encode = scheme.encoder.encode

        def counted_online_encode(*encode_args, **encode_kwargs):
            nonlocal online_encode_call_count
            online_encode_call_count += 1
            return original_encode(*encode_args, **encode_kwargs)

        scheme.encoder.encode = counted_online_encode
        try:
            _write_phase(args.phase_file, "warmup")
            for _ in range(int(args.warmup_runs)):
                warmup_output = _run_decoder(
                    up_layer,
                    concat_module,
                    dec1a_layer,
                    bridge,
                    dec1b_layer,
                    low_ciphertext,
                    skip_ciphertext,
                )
                warmup_output.release()
            bridge.clear_runtime_profile()
            _collect_runtime_memory(scheme.backend)
            _write_phase(args.phase_file, "pre_measured_gc")
            memory["pre_measured_gc"] = _memory_snapshot(scheme.backend)

            scheme.backend.ResetOperationCounters()
            if args.mode == WPC_STORAGE_COMPRESSED:
                scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
            forward_wall_s: list[float] = []
            decompression_s: list[float] = []
            transform_evaluate_s: list[float] = []
            activation_s: list[float] = []
            bootstrap_s: list[float] = []
            bootstrap_call_count = 0
            _write_phase(args.phase_file, "measured")
            for _ in range(int(args.forward_runs)):
                started = time.perf_counter()
                output = _run_decoder(
                    up_layer,
                    concat_module,
                    dec1a_layer,
                    bridge,
                    dec1b_layer,
                    low_ciphertext,
                    skip_ciphertext,
                )
                forward_wall_s.append(float(time.perf_counter() - started))
                if final_output is not None:
                    final_output.release()
                final_output = output
                activation_s.append(float(bridge.last_evaluation["activation_s"]))
                bootstrap_s.append(float(bridge.last_evaluation["bootstrap_s"]))
                bootstrap_call_count += 1
                if args.mode == WPC_STORAGE_COMPRESSED:
                    runtime_rows = [
                        row
                        for plan in plans
                        for row in plan.compressed_stats().values()
                    ]
                    decompression_s.append(
                        float(
                            sum(row["last_decompress_s"] for row in runtime_rows)
                        )
                    )
                    transform_evaluate_s.append(
                        float(sum(row["last_evaluate_s"] for row in runtime_rows))
                    )
                else:
                    decompression_s.append(0.0)
                    transform_evaluate_s.append(0.0)
            counters_total = _operation_counters(scheme.backend)
            _write_phase(args.phase_file, "post_measured")
            memory["after_measured"] = _memory_snapshot(scheme.backend)

            _write_phase(args.phase_file, "correctness")
            decoded = dec1b_plan.decrypt_unpack(final_output)
        finally:
            scheme.encoder.encode = original_encode

        error = np.abs(decoded - clear_dec1b)
        storage = _logical_storage(plans, concat_plan)
        global_stats = _global_stats(scheme.backend)
        compressed_rows = {
            plan.layer_name: plan.compressed_stats() for plan in plans
        }
        expected_registry_count = (
            TRAINED_DECODER_TRANSFORM_COUNT
            if args.mode == WPC_STORAGE_COMPRESSED
            else 0
        )
        worker_acceptance = {
            "correct_vs_clear": bool(
                np.allclose(
                    decoded,
                    clear_dec1b,
                    rtol=0.0,
                    atol=float(args.atol),
                )
            ),
            "independent_torch_oracle_matches_clear": oracle_max_abs_delta
            <= 1e-10,
            "expected_transform_count": storage["learned_transform_count"]
            == TRAINED_DECODER_TRANSFORM_COUNT,
            "checkpoint_activation_is_degree_seven": bridge_compile[
                "activation_degree"
            ]
            == 7,
            "bootstrap_uses_four_ciphertext_groups": bridge_compile[
                "ciphertext_group_count"
            ]
            == 4,
            "real_bootstrap_ran_for_every_measured_forward": bootstrap_call_count
            == int(args.forward_runs),
            "zero_online_python_encode_calls": online_encode_call_count == 0,
            "storage_mode_matches_request": storage["storage_mode"] == args.mode,
            "measured_forward_count_matches_request": len(forward_wall_s)
            == int(args.forward_runs),
            "compressed_registry_matches_mode": int(
                global_stats["registered_transform_count"]
            )
            == expected_registry_count,
            "compressed_materialization_released": bool(
                args.mode == WPC_STORAGE_FULL
                or (
                    global_stats["current_materialized_full_payload_bytes"] == 0
                    and global_stats["current_materialized_transform_count"] == 0
                )
            ),
        }
        worker_acceptance["valid"] = bool(all(worker_acceptance.values()))
        payload = {
            "schema_version": 1,
            "profile": "wpc_cips_trained_decoder_isolated_worker",
            "status": "ok" if worker_acceptance["valid"] else "invalid",
            "mode": str(args.mode),
            "seed": int(args.seed),
            "process": {
                "pid": int(os.getpid()),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "model_metadata": model_metadata,
            },
            "experiment": {
                "graph": "up1+skip1->cat1->dec1a->Cheb7->bootstrap->dec1b",
                "logn": int(args.logn),
                "slots": int(slots),
                "low_shape": list(low_shape),
                "high_shape": list(high_shape),
                "bsgs_ratio": float(args.bsgs_ratio),
                "feature_std": float(args.feature_std),
                "bound_headroom": float(args.bound_headroom),
                "warmup_runs": int(args.warmup_runs),
                "forward_runs": int(args.forward_runs),
                "atol": float(args.atol),
            },
            "timing_policy": (
                "fresh process with warmups excluded; measured wall time includes "
                "up1, concat, dec1a, trained Cheb7, four-ciphertext bootstrap, "
                "and dec1b; compilation, encryption, decryption, and cleanup excluded"
            ),
            "scheme_init_s": float(scheme_init_s),
            "compile_s": float(compile_s),
            "bootstrap_range": {
                "activation_min": float(np.min(clear_activation)),
                "activation_max": float(np.max(clear_activation)),
                "symmetric_bound": float(bound),
            },
            "clear_oracle_max_abs_delta": float(oracle_max_abs_delta),
            "storage": storage,
            "measurements": {
                "forward_wall_s": forward_wall_s,
                "forward_wall_summary_s": summarize_samples(forward_wall_s),
                "decompression_s": decompression_s,
                "decompression_summary_s": summarize_samples(decompression_s),
                "transform_evaluate_s": transform_evaluate_s,
                "transform_evaluate_summary_s": summarize_samples(
                    transform_evaluate_s
                ),
                "activation_s": activation_s,
                "activation_summary_s": summarize_samples(activation_s),
                "bootstrap_s": bootstrap_s,
                "bootstrap_summary_s": summarize_samples(bootstrap_s),
                "bootstrap_call_count": int(bootstrap_call_count),
                "operation_counters_total": counters_total,
                "operation_counters_per_forward": _per_forward(
                    counters_total,
                    int(args.forward_runs),
                ),
                "online_python_encode_call_count": int(
                    online_encode_call_count
                ),
            },
            "correctness": {
                "correct": worker_acceptance["correct_vs_clear"],
                "max_abs_error": float(np.max(error)),
                "mean_abs_error": float(np.mean(error)),
                "output_shape": list(decoded.shape),
                "output_values": decoded.reshape(-1).tolist(),
            },
            "memory": memory,
            "backend": {
                "compressed_global_stats": global_stats,
                "compressed_transform_stats": compressed_rows,
                "weight_plaintext_offline_encode_calls": (
                    int(global_stats["total_weight_plaintext_offline_encode_calls"])
                    if args.mode == WPC_STORAGE_COMPRESSED
                    else TRAINED_DECODER_TRANSFORM_COUNT
                ),
                "weight_plaintext_online_encode_calls": (
                    int(global_stats["total_weight_plaintext_online_encode_calls"])
                    if args.mode == WPC_STORAGE_COMPRESSED
                    else 0
                ),
            },
            "acceptance": worker_acceptance,
        }
    finally:
        if final_output is not None:
            final_output.release()
        if low_ciphertext is not None:
            low_ciphertext.release()
        if skip_ciphertext is not None:
            skip_ciphertext.release()
        if bridge is not None:
            bridge.cleanup()
        if concat_module is not None:
            concat_module.remove_wpc_cips_plan()
        for layer in (up_layer, dec1a_layer, dec1b_layer):
            if layer is not None:
                layer.remove_wpc_cips_plan()
        if scheme is not None:
            scheme.delete_scheme()

    if payload is None:
        raise RuntimeError("worker did not produce a result")
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_phase(args.phase_file, "complete")
    print(
        json.dumps(
            {
                "mode": payload["mode"],
                "status": payload["status"],
                "checkpoint_sha256": payload["checkpoint"]["sha256"],
                "compile_s": payload["compile_s"],
                "forward_wall_summary_s": payload["measurements"][
                    "forward_wall_summary_s"
                ],
                "storage": payload["storage"],
                "correctness": {
                    key: value
                    for key, value in payload["correctness"].items()
                    if key != "output_values"
                },
                "acceptance": payload["acceptance"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
