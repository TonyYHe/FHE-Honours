#!/usr/bin/env python3
"""Run one isolated full, compressed, or online-Encode decoder FHE worker."""

from __future__ import annotations

import argparse
import hashlib
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
    WPC_STORAGE_ONLINE,
    WPC_STORAGE_MODES,
)
from orion.experimental.wpc_decoder_geometry import (
    UNASSESSED_SECURITY_SCOPE, decoder_geometry, resolve_decoder_config,
    validate_parameter_manifest,
)
from orion.experimental.wpc_cips_upsample import WPCCIPSConvTranspose2dPlan
from orion.experimental.wpc_orion_layout_control import (
    OrionLayoutConvPlan, OrionAlignedConcatPlan,
    OrionLayoutTrainedActivationBootstrap, encrypt_native,
)
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
    _encrypt_packed,
    _load_checkpoint_identified,
    _make_conv,
    _make_up,
    _torch_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=WPC_STORAGE_MODES, required=True)
    parser.add_argument("--layout", choices=("cips", "native_orion"), default="cips")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--logn", type=int, default=None)
    parser.add_argument("--ckks-config", type=Path, help="explicit Orion CKKS JSON; otherwise use functional defaults")
    parser.add_argument("--verify-exact-qp", action="store_true", help="untimed exact Q/P control for the compressed worker")
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--feature-std", type=float, default=0.02)
    parser.add_argument("--bound-headroom", type=float, default=1.25)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--forward-runs", type=int, default=10)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--retain-sample-outputs", action="store_true",
                        help="retain every measured output for independent Stage-31 review (outside timed forwards)")
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
    native = args.layout == "native_orion"
    if native and args.mode == WPC_STORAGE_COMPRESSED:
        raise SystemExit("native control does not support WPC periodic compression")
    config, config_source = resolve_decoder_config(args)
    geometry = decoder_geometry(args.logn, args.height, args.width, len(config["ckks_params"]["LogQ"]) - 1)
    levels = geometry["levels"]
    expected_transform_count = geometry["learned_transform_count"]
    for name in ("feature_std", "bsgs_ratio", "atol", "bound_headroom"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise SystemExit(f"{name} must be finite and positive")
    if int(args.warmup_runs) < 0 or int(args.forward_runs) <= 0:
        raise SystemExit("warmup-runs must be nonnegative and forward-runs positive")
    if float(args.bound_headroom) < 1.0:
        raise SystemExit("bound-headroom must be at least one")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
    environment = {
        "ORION_LATTIGO_CLEAR_BACKEND": "0", "ORION_LATTIGO_STREAMING_LT": "0",
        "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT": "0", "ORION_WPC_PERIODICITY_PROFILE": "0",
        "ORION_SINGLE_SLOT_LAYER_CACHE": "0", "ORION_CPP_DIAG_BUILDER": "0",
        "ORION_DIRECT_PACK_WORKERS": "1",
        "ORION_SINGLE_SLOT_ENCODE_WORKERS": "1", "ORION_LATTIGO_COMPILE_WORKERS": "1",
    }
    os.environ.update(environment)

    _write_phase(args.phase_file, "startup")
    state, model_metadata, checkpoint_sha256 = _load_checkpoint_identified(checkpoint_path)
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
    plans = []
    concat_plan = None
    try:
        _write_phase(args.phase_file, "scheme_init")
        scheme_started = time.perf_counter()
        scheme = orion.init_scheme(config)
        shared_library_path = Path(scheme.backend.lib._name).resolve()
        shared_library_sha256 = hashlib.sha256(shared_library_path.read_bytes()).hexdigest()
        required_apis = ("ResetWPCBenchmarkEncodeCounters", "GetWPCBenchmarkEncodeCounters",
                         "GetWPCOnlineGlobalStats", "ResetWPCOnlineMaterializationPeak", "GetWPCParameterManifest")
        if native:
            required_apis += ("GetLinearTransformPayloadStats", "CopyWPCLayoutCiphertext")
        if any(not hasattr(scheme.backend, name) for name in required_apis):
            raise RuntimeError("decoder benchmark shared-library APIs missing; rebuild with tools/build_lattigo.py")
        scheme.backend.ResetWPCBenchmarkEncodeCounters()
        scheme_init_s = float(time.perf_counter() - scheme_started)
        Conv2d.set_scheme(scheme)
        ConvTranspose2d.set_scheme(scheme)
        Concat.set_scheme(scheme)
        memory: dict[str, Any] = {
            "after_scheme_init": _memory_snapshot(scheme.backend)
        }

        up_layer = _make_up(
            state,
            level=levels["up1"],
            bsgs_ratio=float(args.bsgs_ratio),
        )
        dec1a_layer = _make_conv(
            state,
            "dec1a",
            input_channels=64,
            output_channels=32,
            level=levels["dec1a"],
            bsgs_ratio=float(args.bsgs_ratio),
        )
        dec1b_layer = _make_conv(
            state,
            "dec1b",
            input_channels=32,
            output_channels=32,
            level=levels["dec1b"],
            bsgs_ratio=float(args.bsgs_ratio),
        )
        concat_module = Concat(dim=1, bsgs_ratio=float(args.bsgs_ratio))
        concat_module.name = "checkpoint_cat1"

        _write_phase(args.phase_file, "compile")
        compile_started = time.perf_counter()
        up_plan = (OrionLayoutConvPlan(up_layer, low_shape, scheme, storage_mode=str(args.mode), transpose=True)
                   if native else up_layer.install_wpc_cips_plan(
            low_shape,
            storage_mode=str(args.mode),
            include_full_control=False,
            verify_exact_qp=args.verify_exact_qp and args.mode == WPC_STORAGE_COMPRESSED,
        ))
        plans.append(up_plan)
        if not native and not isinstance(up_plan, WPCCIPSConvTranspose2dPlan):
            raise RuntimeError("up1 did not install a transposed-convolution plan")
        dec1a_plan = (OrionLayoutConvPlan(dec1a_layer, concat_shape, scheme, storage_mode=str(args.mode))
                      if native else dec1a_layer.install_wpc_cips_plan(
            concat_shape,
            storage_mode=str(args.mode),
            include_full_control=False,
            verify_exact_qp=args.verify_exact_qp and args.mode == WPC_STORAGE_COMPRESSED,
        ))
        plans.append(dec1a_plan)
        dec1b_plan = (OrionLayoutConvPlan(dec1b_layer, high_shape, scheme, storage_mode=str(args.mode))
                      if native else dec1b_layer.install_wpc_cips_plan(
            high_shape,
            storage_mode=str(args.mode),
            include_full_control=False,
            verify_exact_qp=args.verify_exact_qp and args.mode == WPC_STORAGE_COMPRESSED,
        ))
        plans = [up_plan, dec1a_plan, dec1b_plan]
        skip_contract = SimpleNamespace(
            output_shape=high_shape,
            output_packing_signature=up_plan.output_packing_signature,
            output_level=levels["skip1"],
        )
        concat_plan = (OrionAlignedConcatPlan(scheme, up_plan, skip_contract, dec1a_plan)
                       if native else concat_module.install_wpc_cips_plan(
            (up_plan, skip_contract),
            consumer_plan=dec1a_plan,
        ))
        if not native and not isinstance(concat_plan, WPCCIPSConcatPlan):
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
            float(np.max(np.abs(torch_reference["activation"])))
            * float(args.bound_headroom),
        )
        bridge_type = OrionLayoutTrainedActivationBootstrap if native else WPCCIPSTrainedActivationBootstrap
        bridge = bridge_type(
            logical_shape=high_shape,
            packing_signature=dec1a_plan.output_packing_signature,
            input_level=int(dec1a_plan.output_level),
            output_level=int(dec1b_plan.level),
            activation_spec=activation_spec,
            bootstrap_bound=bound,
        )
        bridge_compile = bridge.compile(scheme)
        parameter_manifest = json.loads(bytes(scheme.backend.GetWPCParameterManifest()).decode("utf-8"))
        validate_parameter_manifest(parameter_manifest, config, slots)
        print(json.dumps({"event": "decoder_compiled", "geometry": geometry,
                          "security_scope": UNASSESSED_SECURITY_SCOPE}), flush=True)
        for layer in (up_layer, dec1a_layer, dec1b_layer):
            layer.he()
        concat_module.he()
        compile_s = float(time.perf_counter() - compile_started)
        compile_encode_invocations = list(scheme.backend.GetWPCBenchmarkEncodeCounters())

        _write_phase(args.phase_file, "post_compile_raw")
        memory["post_compile_raw"] = _memory_snapshot(scheme.backend)
        _collect_runtime_memory(scheme.backend)
        _write_phase(args.phase_file, "post_compile_gc")
        memory["post_compile_gc"] = _memory_snapshot(scheme.backend)

        _write_phase(args.phase_file, "input_prepare")
        low_ciphertext = up_plan.encrypt_input(low)
        skip_ciphertext = (encrypt_native(scheme, skip, up_plan.output_packing_signature, levels["skip1"])
                           if native else _encrypt_packed(
            scheme,
            skip[0],
            up_plan.output_packing_signature,
            level=levels["skip1"],
        ))
        def run_decoder():
            return _run_decoder(
                up_plan if native else up_layer, concat_plan if native else concat_module,
                dec1a_plan if native else dec1a_layer, bridge,
                dec1b_plan if native else dec1b_layer, low_ciphertext, skip_ciphertext)
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
            # Traced correctness/lifecycle run is OUTSIDE performance samples.
            _write_phase(args.phase_file, "correctness_preflight")
            preflight = run_decoder()
            try:
                preflight_decoded = dec1b_plan.decrypt_unpack(preflight)
                preflight_error = float(np.max(np.abs(preflight_decoded - clear_dec1b)))
                if not np.isfinite(preflight_decoded).all() or preflight_error > float(args.atol):
                    raise RuntimeError("untimed decoder correctness preflight failed")
                lifecycle_trace = [plan.last_evaluation["evaluation_sequence"] for plan in plans]
            finally:
                preflight.release()
            for plan in plans:
                plan.record_sequence = False
            _write_phase(args.phase_file, "warmup")
            for _ in range(int(args.warmup_runs)):
                warmup_output = run_decoder()
                warmup_output.release()
            bridge.clear_runtime_profile()
            _collect_runtime_memory(scheme.backend)
            _write_phase(args.phase_file, "pre_measured_gc")
            memory["pre_measured_gc"] = _memory_snapshot(scheme.backend)

            scheme.backend.ResetOperationCounters()
            scheme.backend.ResetWPCBenchmarkEncodeCounters()
            scheme.backend.ResetWPCOnlineMaterializationPeak()
            if args.mode == WPC_STORAGE_COMPRESSED:
                scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
            forward_wall_s: list[float] = []
            decompression_s: list[float] = []
            online_encode_s: list[float] = []
            online_prepare_s: list[float] = []
            transform_encode_invocations: list[int] = []
            measured_output_errors: list[float] = []
            measured_output_values = []
            measured_operation_counters = []
            timed_lifecycle_records: list[int] = []
            transform_evaluate_s: list[float] = []
            activation_s: list[float] = []
            bootstrap_s: list[float] = []
            bootstrap_call_count = 0
            _write_phase(args.phase_file, "measured")
            for _ in range(int(args.forward_runs)):
                _write_phase(args.phase_file, "measured")
                before_encode = list(scheme.backend.GetWPCBenchmarkEncodeCounters())
                before_operations = _operation_counters(scheme.backend) if args.retain_sample_outputs else None
                started = time.perf_counter()
                output = run_decoder()
                forward_wall_s.append(float(time.perf_counter() - started))
                timed_lifecycle_records.append(sum(len(plan.last_evaluation["evaluation_sequence"]) for plan in plans))
                _write_phase(args.phase_file, "sample_validation")
                after_encode = list(scheme.backend.GetWPCBenchmarkEncodeCounters())
                transform_encode_invocations.append(int(sum(after_encode) - sum(before_encode)))
                if final_output is not None:
                    final_output.release()
                final_output = output
                activation_s.append(float(bridge.last_evaluation["activation_s"]))
                bootstrap_s.append(float(bridge.last_evaluation["bootstrap_s"]))
                bootstrap_call_count += 1
                sample_decoded = dec1b_plan.decrypt_unpack(output)
                sample_error = float(np.max(np.abs(sample_decoded - clear_dec1b)))
                if not np.isfinite(sample_decoded).all() or sample_error > float(args.atol):
                    raise RuntimeError("measured decoder output failed clear-reference validation")
                measured_output_errors.append(sample_error)
                if args.retain_sample_outputs:
                    measured_output_values.append(sample_decoded.reshape(-1).tolist())
                    after_operations = _operation_counters(scheme.backend)
                    measured_operation_counters.append({key: after_operations[key] - value
                                                        for key, value in before_operations.items()})
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
                    online_encode_s.append(0.0)
                    online_prepare_s.append(0.0)
                elif args.mode == WPC_STORAGE_ONLINE:
                    rows = [row for plan in plans for row in plan.online_stats().values()]
                    online_encode_s.append(sum(row["last_encode_nanoseconds"] for row in rows) / 1e9)
                    online_prepare_s.append(sum(row["last_prepare_nanoseconds"] for row in rows) / 1e9)
                    transform_evaluate_s.append(sum(row["last_evaluate_nanoseconds"] for row in rows) / 1e9)
                    decompression_s.append(0.0)
                else:
                    decompression_s.append(0.0)
                    transform_evaluate_s.append(sum(plan.last_evaluation["full_transform_evaluate_call_s"] for plan in plans))
                    online_encode_s.append(0.0)
                    online_prepare_s.append(0.0)
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
        online_global_values = list(scheme.backend.GetWPCOnlineGlobalStats())
        online_global_stats = dict(zip(("registered_transform_count", "current_materialized_bytes",
            "peak_materialized_bytes", "current_materialized_transforms", "peak_materialized_transforms"),
            map(int, online_global_values)))
        measured_encode_invocations = list(map(int, scheme.backend.GetWPCBenchmarkEncodeCounters()))
        compressed_rows = {
            plan.layer_name: plan.compressed_stats() for plan in plans
        }
        expected_registry_count = (
            expected_transform_count
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
            == expected_transform_count,
            "checkpoint_activation_is_degree_seven": bridge_compile[
                "activation_degree"
            ]
            == 7,
            "bootstrap_group_count_matches_geometry": bridge_compile[
                "ciphertext_group_count"
            ]
            == geometry["bootstrap_ciphertext_group_count"],
            "runtime_parameters_match_request": True,
            "exact_qp_verified_if_requested": bool(
                not args.verify_exact_qp or args.mode != WPC_STORAGE_COMPRESSED
                or all(row["exact_qp_match"] is True for plan in plans for row in plan.transform_rows.values())),
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
            "untimed_preflight_correct": preflight_error <= float(args.atol),
            "all_measured_outputs_correct": all(value <= float(args.atol) for value in measured_output_errors),
            "timed_lifecycle_tracing_disabled": all(plan.record_sequence is False for plan in plans),
            "actual_transform_encode_invocations_match_mode": all(
                value == (expected_transform_count if args.mode == WPC_STORAGE_ONLINE else 0)
                for value in transform_encode_invocations),
            "online_recipe_materialization_released": online_global_stats["current_materialized_bytes"] == 0
                and online_global_stats["current_materialized_transforms"] == 0,
        }
        worker_acceptance["valid"] = bool(all(worker_acceptance.values()))
        payload = {
            "schema_version": 3,
            "profile": ("wpc_native_orion_matched_decoder_worker" if native else "wpc_cips_trained_decoder_isolated_worker"),
            "layout": args.layout,
            "status": "ok" if worker_acceptance["valid"] else "invalid",
            "mode": str(args.mode),
            "seed": int(args.seed),
            "feature_sha256": hashlib.sha256(low.tobytes(order="C") + skip.tobytes(order="C")).hexdigest(),
            "shared_library": {"path": str(shared_library_path), "sha256": shared_library_sha256},
            "process": {
                "pid": int(os.getpid()),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": str(torch.__version__),
                "numpy": str(np.__version__),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "identity_policy": "sha256_of_exact_bytes_deserialized",
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
                "ckks_config": config,
                "configuration_source": config_source,
                "geometry": geometry,
                "verify_exact_qp": bool(args.verify_exact_qp),
                "security_scope": UNASSESSED_SECURITY_SCOPE,
                "timed_lifecycle_tracing": False,
                "runtime_environment": environment,
                "padding_semantics": "flattened_spatial_cyclic",
                "native_control_scope": ("square embedding; low gap 2, high gap 1; aligned concat; explicit blockwise policy; not automatic whole-model Orion" if native else None),
            },
            "timing_policy": (
                "fresh process with warmups excluded; measured wall time includes "
                "up1, concat, dec1a, trained Cheb7, geometry-derived ciphertext-group bootstrap, "
                "and dec1b; compilation, encryption, decryption, and cleanup excluded"
            ),
            "scheme_init_s": float(scheme_init_s),
            "compile_s": float(compile_s),
            "bootstrap_range": {
                "activation_min": float(np.min(torch_reference["activation"])),
                "activation_max": float(np.max(torch_reference["activation"])),
                "symmetric_bound": float(bound),
            },
            "clear_oracle_max_abs_delta": float(oracle_max_abs_delta),
            "runtime_parameter_manifest": parameter_manifest,
            "bootstrap_compile": bridge_compile,
            "qp_verification": {
                "requested": bool(args.verify_exact_qp),
                "applicable": args.mode == WPC_STORAGE_COMPRESSED,
                "rows": [dict(layer=plan.layer_name, transform=key, exact_qp_match=row["exact_qp_match"],
                              diagonal_count=row["diagonal_count"], reconstructed_diagonal_count=row["manual_decompressed_diagonal_count"])
                         for plan in plans for key, row in plan.transform_rows.items()]
                        if args.verify_exact_qp and args.mode == WPC_STORAGE_COMPRESSED else [],
                "retained_full_control_count": sum(len(plan.full_control_transform_ids) for plan in plans),
            },
            "storage": storage,
            "measurements": {
                "transform_evaluate_timer_scope": "Python-to-backend EvaluateLinearTransform call wall" if native or args.mode == WPC_STORAGE_FULL else "backend evaluation timer",
                "online_encode_timer_scope": "GenerateLinearTransform call wall (allocation and binding included)" if native else "backend lintrans.Encode only",
                "forward_wall_s": forward_wall_s,
                "online_encode_s": online_encode_s,
                "online_prepare_s": online_prepare_s,
                "online_encode_pct_of_forward": [100 * encode / wall for encode, wall in zip(online_encode_s, forward_wall_s)],
                "online_materialization_pct_of_forward": [100 * (encode + prepare) / wall for encode, prepare, wall in zip(online_encode_s, online_prepare_s, forward_wall_s)],
                "transform_encode_invocations": transform_encode_invocations,
                "measured_output_max_abs_errors": measured_output_errors,
                "timed_lifecycle_record_count": timed_lifecycle_records,
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
                "preflight_max_abs_error": preflight_error,
                "untimed_lifecycle_trace": lifecycle_trace,
                "mean_abs_error": float(np.mean(error)),
                "output_shape": list(decoded.shape),
                "output_values": decoded.reshape(-1).tolist(),
                "independent_clear_output_values": torch_reference["dec1b"].reshape(-1).tolist(),
                "independent_clear_output_sha256": hashlib.sha256(np.ascontiguousarray(torch_reference["dec1b"], dtype="<f8").tobytes()).hexdigest(),
            },
            "memory": memory,
            "backend": {
                "compressed_global_stats": global_stats,
                "compressed_transform_stats": compressed_rows,
                "online_recipe_global_stats": online_global_stats,
                "expected_max_single_transform_bytes": max(
                    row["full_payload_bytes"] for plan in plans for row in plan.transform_rows.values()),
                "compile_transform_encode_invocations": dict(zip(("ordinary", "compressed", "online_recipe"), map(int, compile_encode_invocations))),
                "measured_transform_encode_invocations": dict(zip(("ordinary", "compressed", "online_recipe"), measured_encode_invocations)),
                "encode_counter_scope": "actual successful lintrans.Encode invocations; one invocation per transform, not per diagonal; excludes input/bias/bootstrapping encoder internals",
                "native_materialized_transform_count_after_forward": sum(plan.current_materialized_count for plan in plans) if native else None,
                "native_transform_rows": {plan.layer_name: plan.transform_rows for plan in plans} if native else None,
                "weight_plaintext_offline_encode_calls": (
                    int(global_stats["total_weight_plaintext_offline_encode_calls"])
                    if args.mode == WPC_STORAGE_COMPRESSED
                    else (0 if args.mode == WPC_STORAGE_ONLINE else expected_transform_count)
                ),
                "weight_plaintext_online_encode_calls": (
                    int(global_stats["total_weight_plaintext_online_encode_calls"])
                    if args.mode == WPC_STORAGE_COMPRESSED
                    else (sum(transform_encode_invocations) if args.mode == WPC_STORAGE_ONLINE else 0)
                ),
            },
            "acceptance": worker_acceptance,
        }
        if args.retain_sample_outputs:
            from orion.experimental.wpc_orion_layout_control import NATIVE_PREPARATION_POLICY
            payload["experiment"]["retain_sample_outputs"] = True
            payload["experiment"]["native_preparation_policy"] = NATIVE_PREPARATION_POLICY if native else None
            payload["correctness"]["measured_output_values"] = measured_output_values
            payload["measurements"]["measured_operation_counters"] = measured_operation_counters
    finally:
        if final_output is not None:
            final_output.release()
        if low_ciphertext is not None:
            low_ciphertext.release()
        if skip_ciphertext is not None:
            skip_ciphertext.release()
        if bridge is not None:
            bridge.cleanup()
        if native:
            if concat_plan is not None:
                concat_plan.cleanup()
            for plan in plans:
                plan.cleanup()
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
                    if key not in ("output_values", "independent_clear_output_values", "measured_output_values")
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
