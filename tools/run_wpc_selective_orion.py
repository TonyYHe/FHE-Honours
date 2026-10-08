#!/usr/bin/env python3
"""Correctness gate for selective Q/P storage in unchanged Orion transforms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orion
from orion.core.packing import direct_diagonalize_conv2d
from orion.experimental.wpc_selective_orion import SelectiveOrionTransform
from tools.run_wpc_cips_isolated_worker import _config, _operation_counters


def source_hashes():
    paths = (
        "tools/run_wpc_selective_orion.py", "tools/run_wpc_selective_model.py",
        "tools/run_wpc_selective_orion_server.sh", "tools/run_lattigo_e2e_compare.py",
        "orion/experimental/wpc_selective_orion.py", "orion/experimental/wpc_periodicity.py",
        "orion/backend/lattigo/wpc_selective.go", "orion/backend/lattigo/wpc_compression.go",
        "orion/backend/lattigo/lineartransform.go", "orion/backend/lattigo/scheme.go",
        "orion/backend/lattigo/bindings.py", "orion/backend/python/lt_evaluator.py",
        "orion/core/packing.py", "orion/nn/linear.py",
    )
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}


def gate_cases(slots, seed):
    rng = np.random.default_rng(seed)
    layer = torch.nn.Conv2d(8, 8, 3, padding=1, bias=False)
    layer.on_weight = torch.tensor(rng.normal(0, .02, (8, 8, 3, 3)), dtype=torch.float32)
    layer.input_shape = layer.output_shape = torch.Size((1, 8, 8, 8))
    layer.input_gap = layer.output_gap = 1
    layer.fhe_input_shape = layer.fhe_output_shape = layer.input_shape
    native, _ = direct_diagonalize_conv2d(layer, layer.on_weight, slots, "square", False, allow_hybrid=False)
    for key, diagonals in sorted(native.items()):
        yield f"native_zero_padded_conv_block_{key[0]}_{key[1]}", diagonals, "ordinary Orion direct packer, deterministic random weights"
    yield "mixed_control", {
        0: np.full(slots, .125, dtype=np.float32),
        1: np.tile(rng.normal(0, .02, 8).astype(np.float32), slots//8),
        7: np.arange(slots, dtype=np.float32)/slots,
        -1: np.zeros(slots, dtype=np.float32),
    }, "synthetic mixed periodic/nonperiodic/zero diagonals; not a model census"
    yield "all_periodic_control", {
        0: np.full(slots, .125, dtype=np.float32),
        1: np.tile(rng.normal(0, .02, 8).astype(np.float32), slots//8),
        7: np.tile(rng.normal(0, .02, 32).astype(np.float32), slots//32),
    }, "synthetic all-periodic control; not a U-Net structural-transform census"


def decode(scheme, ciphertext_id):
    pt = scheme.backend.Decrypt(ciphertext_id)
    try:
        return np.asarray(scheme.backend.Decode(pt), dtype=np.float64)
    finally:
        scheme.backend.DeletePlaintext(pt)


def run_case(scheme, name, diagonals, description, *, runs, seed):
    backend, slots = scheme.backend, scheme.params.get_slots()
    indices = np.array(sorted(diagonals), dtype=np.int32)
    data = np.concatenate([np.asarray(diagonals[int(i)], dtype=np.float32) for i in indices])
    messages = np.random.default_rng(seed).normal(0, .02, slots).astype(np.float32)
    reference = sum(np.asarray(diagonals[int(i)], dtype=np.float32).astype(np.float64) *
                    np.roll(messages.astype(np.float64), -int(i)) for i in indices)
    control = int(backend.GenerateLinearTransform(indices.tolist(), data.tolist(), 2, 2.0, "none"))
    encrypted, pt = None, None
    policies = {}
    try:
        scheme.lt_evaluator.generate_rotation_keys(control)
        pt = scheme.encode(torch.tensor(messages.reshape(1, -1)), level=2)
        encrypted = scheme.encrypt(pt)
        backend.ResetOperationCounters()
        output = int(backend.EvaluateLinearTransform(control, int(encrypted.ids[0])))
        try:
            full_output = decode(scheme, output)
        finally:
            backend.DeleteCiphertext(output)
        full_operations = _operation_counters(backend)
        for policy in ("online", "hybrid"):
            plan = SelectiveOrionTransform(backend, indices, data, slots=slots, level=2, bsgs_ratio=2.0, policy=policy)
            samples = []
            try:
                for _ in range(runs):
                    backend.ResetOperationCounters()
                    output = None
                    try:
                        started = time.perf_counter()
                        plan.materialize(indices, data)
                        eval_started = time.perf_counter()
                        output = int(backend.EvaluateLinearTransform(plan.id, int(encrypted.ids[0])))
                        evaluate_s = time.perf_counter() - eval_started
                        wall_s = time.perf_counter() - started
                        # Exactness, decoding and stats are OUTSIDE diagnostic timing.
                        exact = int(backend.VerifyWPCSelectiveLinearTransformExact(control, plan.id)) == 1
                        actual = decode(scheme, output)
                        stats = plan.stats()
                        operations = _operation_counters(backend)
                        encode_s = stats["last_encode_nanoseconds"]/1e9
                        decompression_s = stats["last_decompress_nanoseconds"]/1e9
                        prepare_s = stats["last_prepare_nanoseconds"]/1e9
                        eligible_s = stats["last_eligible_encode_nanoseconds"]/1e9
                        residual_s = wall_s - evaluate_s - encode_s - decompression_s - prepare_s
                        samples.append({"lt_prepare_plus_evaluate_s": wall_s,
                            "backend_prepare_s": prepare_s, "backend_encode_s": encode_s,
                            "decompression_s": decompression_s, "evaluate_call_s": evaluate_s,
                            "python_ffi_other_s": residual_s,
                            "encode_pct_of_lt_prepare_plus_evaluate": 100*encode_s/wall_s,
                            "baseline_eligible_encode_time_coverage_pct": (100*eligible_s/encode_s if encode_s else None)
                                if policy == "online" else None,
                            "max_abs_error_vs_clear": float(np.max(np.abs(actual-reference))),
                            "max_abs_delta_vs_full": float(np.max(np.abs(actual-full_output))),
                            "exact_qp_match": exact, "operation_counters": operations,
                            "outputs": actual.tolist(), "stats_while_materialized": stats})
                    finally:
                        if output is not None:
                            backend.DeleteCiphertext(output)
                        plan.release()
                    samples[-1]["stats_after_release"] = plan.stats()
                stats = plan.stats()
                expected_embeds = (stats["diagonal_count"] - stats["compressed_count"]) * runs
                acceptance = {
                    "exact_qp_in_every_run": all(s["exact_qp_match"] for s in samples),
                    "clear_error_within_tolerance": all(s["max_abs_error_vs_clear"] <= 1e-6 for s in samples),
                    "full_and_selective_outputs_match": all(s["max_abs_delta_vs_full"] == 0 for s in samples),
                    "operations_unchanged": all(s["operation_counters"] == full_operations for s in samples),
                    "fallback_encode_count_correct": stats["online_embed_calls"] == expected_embeds,
                    "eligible_encodes_eliminated_in_hybrid": policy != "hybrid" or stats["online_eligible_embed_calls"] == 0,
                    "materialization_released": all(s["stats_after_release"]["materialized_bytes"] == 0 for s in samples),
                    "timers_close": all(s["python_ffi_other_s"] >= -1e-8 for s in samples),
                }
                policies[policy] = {"samples": samples, "stats": stats, "acceptance": acceptance}
            finally:
                plan.close()
    finally:
        if encrypted is not None:
            encrypted.release()
        if pt is not None:
            pt.release()
        backend.DeleteLinearTransform(control)
    return {"case": name, "description": description,
        "indices": indices.tolist(), "slot_payload_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "source_values": messages.tolist(), "diagonal_values": data.reshape(-1, slots).tolist(),
        "clear_reference": reference.tolist(), "full_output": full_output.tolist(),
        "full_operation_counters": full_operations, "policies": policies}


def run_gate(*, runs=3, seed=20261008):
    scheme = orion.init_scheme(_config(10))
    try:
        cases = [run_case(scheme, name, diagonals, description, runs=runs, seed=seed)
                 for name, diagonals, description in gate_cases(scheme.params.get_slots(), seed)]
    finally:
        scheme.delete_scheme()
    valid = all(all(p["acceptance"].values()) for case in cases for p in case["policies"].values())
    return {"schema_version": 1, "profile": "selective_unchanged_orion_transform_correctness",
        "status": "ok" if valid else "failed", "acceptance": {"all_case_gates_passed": valid},
        "configuration": _config(10), "seed": seed, "runs": runs, "cases": cases,
        "performance_claims_enabled": False, "security_assessed": False,
        "scope": "Orion-packed layer blocks plus synthetic controls; not whole-model O-hybrid impact",
        "timing_note": "Diagnostic LT preparation/evaluation only; no warmups, no independent-process CI, not HE-forward proportions"}


def review_gate(data):
    """Recompute numerical/counter/accounting evidence, without FHE execution.

    Encoded-polynomial equality remains a recorded backend gate: raw Q/P arrays
    are not bundled. This reviewer does not authenticate a hostile JSON author.
    """
    if data.get("schema_version") != 1 or data.get("status") != "ok" or not data.get("cases"):
        raise ValueError("gate schema/status/cases are invalid")
    if data.get("performance_claims_enabled") is not False or data.get("security_assessed") is not False:
        raise ValueError("incorrect claim scope")
    slots = 1 << (int(data["configuration"]["ckks_params"]["LogN"])-1)
    if data["configuration"] != _config(10) or data.get("acceptance") != {"all_case_gates_passed": True}:
        raise ValueError("gate configuration/acceptance differs from the fixed protocol")
    runs = data["runs"]
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("invalid measured run count")
    for case in data["cases"]:
        indices = np.asarray(case["indices"])
        diagonals = np.asarray(case["diagonal_values"], dtype=np.float32)
        from orion.experimental.wpc_selective_orion import _buffers
        _buffers(indices, diagonals, slots)
        if diagonals.shape != (indices.size, slots) or not np.isfinite(diagonals).all():
            raise ValueError("invalid diagonal data")
        if hashlib.sha256(diagonals.tobytes()).hexdigest() != case["slot_payload_sha256"]:
            raise ValueError("slot payload hash mismatch")
        source = np.asarray(case["source_values"], dtype=np.float64)
        if source.shape != (slots,) or not np.isfinite(source).all():
            raise ValueError("invalid source values")
        oracle = sum(row.astype(np.float64)*np.roll(source, -int(key)) for key, row in zip(indices, diagonals))
        if not np.array_equal(oracle, case["clear_reference"]):
            raise ValueError("clear oracle differs from raw recipes")
        full = np.asarray(case["full_output"])
        if full.shape != (slots,) or not np.isfinite(full).all() or np.max(np.abs(full-oracle)) > 1e-6:
            raise ValueError("full output invalid")
        if set(case["policies"]) != {"online", "hybrid"}:
            raise ValueError("missing storage policy")
        from orion.experimental.wpc_periodicity import analyze_slot_periodicity
        classifications = [analyze_slot_periodicity(row, payload_format="real") for row in diagonals]
        eligible_count = sum(c.wpc_candidate for c in classifications)
        bytes_per_diagonal = 2*slots*4*8  # fixed gate: Q level 2, P level 0
        compressed_bytes = sum(2*c.minimal_period*4*8 for c in classifications if c.wpc_candidate)
        for name, policy in case["policies"].items():
            if len(policy["samples"]) != runs or not policy["acceptance"] or not all(v is True for v in policy["acceptance"].values()):
                raise ValueError("policy gates/run count invalid")
            for index, sample in enumerate(policy["samples"], start=1):
                actual = np.asarray(sample["outputs"])
                if actual.shape != (slots,) or not np.isfinite(actual).all():
                    raise ValueError("invalid selective output")
                error, difference = float(np.max(np.abs(actual-oracle))), float(np.max(np.abs(actual-full)))
                if error > 1e-6 or difference != 0 or error != sample["max_abs_error_vs_clear"] or difference != sample["max_abs_delta_vs_full"]:
                    raise ValueError("stored correctness summary disagrees with raw outputs")
                if sample["operation_counters"] != case["full_operation_counters"] or sample["exact_qp_match"] is not True:
                    raise ValueError("operation or recorded Q/P exactness gate failed")
                stats = sample["stats_while_materialized"]
                after = sample["stats_after_release"]
                if (stats["eligible_count"] != eligible_count or stats["full_payload_bytes"] != indices.size*bytes_per_diagonal
                        or stats["eligible_full_payload_bytes"] != eligible_count*bytes_per_diagonal
                        or stats["compressed_payload_bytes"] != (compressed_bytes if name == "hybrid" else 0)):
                    raise ValueError("periodicity or Q/P byte accounting invalid")
                if stats["diagonal_count"] != indices.size or stats["materialization_count"] != index:
                    raise ValueError("materialization counts invalid")
                compressed = stats["eligible_count"] if name == "hybrid" else 0
                if stats["compressed_count"] != compressed or stats["online_embed_calls"] != index*(indices.size-compressed):
                    raise ValueError("fallback Encode counts invalid")
                if stats["offline_embed_calls"] != compressed or (name == "hybrid" and stats["online_eligible_embed_calls"] != 0):
                    raise ValueError("offline/online eligible Encode counts invalid")
                if stats["materialized_bytes"] != stats["full_payload_bytes"] or after["materialized_bytes"] != 0:
                    raise ValueError("materialization lifecycle invalid")
                if after != {**stats, "materialized_bytes": 0}:
                    raise ValueError("release changed counters beyond active materialization")
                for field, counter in (("backend_encode_s", "last_encode_nanoseconds"),
                    ("backend_prepare_s", "last_prepare_nanoseconds"), ("decompression_s", "last_decompress_nanoseconds")):
                    if not np.isfinite(sample[field]) or sample[field] < 0 or sample[field] != stats[counter]/1e9:
                        raise ValueError("backend timer summary invalid")
                total = sample["lt_prepare_plus_evaluate_s"]
                components = sum(sample[k] for k in ("backend_prepare_s", "backend_encode_s", "decompression_s", "evaluate_call_s", "python_ffi_other_s"))
                if not np.isfinite(total) or total <= 0 or abs(components-total) > 1e-8 or sample["python_ffi_other_s"] < -1e-8:
                    raise ValueError("diagnostic timers do not close")
                if abs(sample["encode_pct_of_lt_prepare_plus_evaluate"] - 100*sample["backend_encode_s"]/total) > 1e-8:
                    raise ValueError("Encode share disagrees with measured timers")
                encode_ns = stats["last_encode_nanoseconds"]
                coverage = (100*stats["last_eligible_encode_nanoseconds"]/encode_ns if encode_ns else None) if name == "online" else None
                actual_coverage = sample["baseline_eligible_encode_time_coverage_pct"]
                if ((coverage is None) != (actual_coverage is None)
                        or coverage is not None and abs(actual_coverage-coverage) > 1e-8):
                    raise ValueError("Encode-time coverage invalid")
            if policy["stats"] != policy["samples"][-1]["stats_after_release"]:
                raise ValueError("final policy statistics disagree with the last sample")
    return {"status": "ok", "reviewed_cases": len(data["cases"]),
        "scope": "raw outputs, recipes, counters and timer arithmetic checked; Q/P exactness is a recorded backend gate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out-dir", type=Path)
    group.add_argument("--review-dir", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.review_dir is not None:
        print(json.dumps(review_gate(json.loads((args.review_dir/"gate.json").read_text())), indent=2))
        return 0
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.out_dir.exists():
        parser.error("output directory exists; choose a fresh path")
    os.environ.update({"ORION_LATTIGO_CLEAR_BACKEND": "0", "ORION_LATTIGO_STREAMING_LT": "0",
        "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT": "0", "ORION_SINGLE_SLOT_ENCODE_WORKERS": "1",
        "ORION_DIRECT_PACK_WORKERS": "1", "ORION_WPC_SELECTIVE_POLICY": "off"})
    args.out_dir.mkdir(parents=True)
    from orion.experimental.wpc_periodicity import real_lattigo_library_path
    library = real_lattigo_library_path()
    sources = source_hashes()
    binary_sha = hashlib.sha256(library.read_bytes()).hexdigest()
    result = run_gate(runs=args.runs, seed=args.seed)
    review_gate(result)
    if source_hashes() != sources or hashlib.sha256(library.read_bytes()).hexdigest() != binary_sha:
        raise ValueError("source/backend changed during the gate")
    result["provenance"] = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "source_sha256": sources, "backend_sha256": binary_sha}
    path = args.out_dir/"gate.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "result": str(path),
        "cases": len(result["cases"]), "performance_claims_enabled": False}, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
