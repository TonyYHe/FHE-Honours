#!/usr/bin/env python3
"""Fresh-process online/hybrid diagnostic for Orion's ordinary dense cache.

Not a balanced performance benchmark or a CIPS/layout crossover experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.experimental.wpc_selective_orion import runtime_delta


def validate_pair(workers, *, atol):
    if not np.isfinite(atol) or atol <= 0 or set(workers) != {"online", "hybrid"}:
        raise ValueError("invalid tolerance/storage policy set")
    baseline = workers["online"]
    rows, operations, identities = {}, {}, {}
    for name, worker in workers.items():
        if worker.get("status") != "ok" or worker.get("mode") != "dense" or worker.get("selective_orion_policy") != name:
            raise ValueError("worker status/mode/policy invalid")
        for key in ("network", "model", "seed", "config", "clear"):
            if key not in worker or worker[key] != baseline[key]:
                raise ValueError(f"model/configuration/clear identity differs: {key}")
        compile_snapshot = worker["selective_orion_after_compile"]
        if compile_snapshot["transform_count"] == 0:
            raise ValueError("no selective cache transforms were compiled")
        if (compile_snapshot["totals"]["materialization_count"] != 0
                or compile_snapshot["totals"]["online_embed_calls"] != 0
                or compile_snapshot["totals"]["materialized_bytes"] != 0
                or any(r["policy"] != name for r in compile_snapshot["transforms"])):
            raise ValueError("compiled snapshot policy/counters invalid")
        identities[name] = sorted((r["module"], tuple(r["block"]), r["slot_payload_sha256"],
            r["stats"]["diagonal_count"], r["stats"]["eligible_count"], r["stats"]["full_payload_bytes"])
            for r in compile_snapshot["transforms"])
        samples, ops, previous = [], [], compile_snapshot
        for attempt in worker["forward_attempts"]:
            if attempt["status"] != "ok" or attempt["kind"] not in ("warmup", "measured"):
                raise ValueError("a requested forward failed")
            current = attempt["selective_orion_snapshot"]
            expected = runtime_delta(previous, current, forward_s=attempt["timing_s"]["he_forward"])
            if expected != attempt["selective_orion_profile"] or expected["materialized_bytes_after_forward"] != 0:
                raise ValueError("runtime counters/timers/lifecycle disagreed with raw snapshots")
            output, clear = np.asarray(attempt["decoded"]["values"]), np.asarray(worker["clear"]["values"])
            if (attempt["decoded"]["shape"] != worker["clear"]["shape"] or output.shape != clear.shape
                    or not output.size or not np.isfinite(clear).all()
                    or not np.isfinite(output).all() or np.max(np.abs(output-clear)) > atol):
                raise ValueError("decrypted model output exceeds recorded tolerance")
            expected_calls = sum(r["stats"]["diagonal_count"]-(r["stats"]["eligible_count"] if name == "hybrid" else 0)
                                 for r in current["transforms"])
            if expected["counter_delta"]["online_embed_calls"] != expected_calls:
                raise ValueError("not every registered transform was materialized exactly once")
            previous_counts = {(r["module"], tuple(r["block"])): r["stats"]["materialization_count"]
                               for r in previous["transforms"]}
            if (expected["counter_delta"]["materialization_count"] != current["transform_count"]
                    or any(r["stats"]["materialization_count"] != previous_counts[(r["module"], tuple(r["block"]))]+1
                           for r in current["transforms"])):
                raise ValueError("a registered transform was omitted or repeated")
            if name == "hybrid" and expected["counter_delta"]["online_eligible_embed_calls"] != 0:
                raise ValueError("hybrid encoded an eligible diagonal online")
            previous = current
            op = attempt["selective_orion_operations"]
            if (set(op) != {"conjugation", "direct_rotation", "linear_transform_rotation", "rotation_total"}
                    or any(type(v) is not int or v < 0 for v in op.values())
                    or op["rotation_total"] != op["direct_rotation"]+op["linear_transform_rotation"]):
                raise ValueError("invalid raw operation counters")
            if attempt["kind"] == "measured":
                samples.append({"he_forward_s": attempt["timing_s"]["he_forward"], **expected})
                ops.append(op)
        if len(samples) != worker["forward_runs"] or not samples:
            raise ValueError("measured forward count invalid")
        if sum(a["kind"] == "warmup" for a in worker["forward_attempts"]) != worker["warmup_runs"]:
            raise ValueError("warmup forward count invalid")
        rows[name], operations[name] = samples, ops
    if identities["online"] != identities["hybrid"] or operations["online"] != operations["hybrid"]:
        raise ValueError("diagonal identity or rotation/conjugation counts changed across policies")
    shares = {}
    for name, samples in rows.items():
        shares[name] = {key: float(np.mean([s[key] for s in samples])) for key in
            ("he_forward_s", "backend_encode_s", "backend_prepare_s", "decompression_s", "backend_encode_pct_of_he_forward")}
    encode = sum(s["backend_encode_s"] for s in rows["online"])
    eligible = sum(s["counter_delta"]["total_eligible_encode_nanoseconds"] for s in rows["online"])/1e9
    return {"status": "ok", "schema_version": 1, "network": baseline["network"], "samples": rows,
        "summary": shares, "correctness_atol": atol,
        "baseline_eligible_encode_time_coverage_pct": 100*eligible/encode if encode else None,
        "registered_transform_count": baseline["selective_orion_after_compile"]["transform_count"],
        "eligible_diagonal_count": baseline["selective_orion_after_compile"]["totals"]["eligible_count"],
        "scope": "registered ordinary dense-cache transforms; remaining model work is outside these backend timers",
        "performance_claims_enabled": False,
        "limitations": ["one fixed-order fresh process per policy, no confidence intervals",
            "same per-diagonal backend Embed implementation in both policies; not the historical batch-Encode baseline",
            "no complete-network CIPS comparison; no security assessment or dataset accuracy claim",
            "rotation/conjugation counters measured; individual multiplication/addition counts unmeasured"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out-dir", type=Path)
    group.add_argument("--review-dir", type=Path)
    parser.add_argument("--network", default="resnet20_cifar10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--forward-runs", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()
    if args.review_dir is not None:
        saved = json.loads((args.review_dir/"comparison.json").read_text())
        workers = {}
        for name in ("online", "hybrid"):
            path = args.review_dir/f"{name}.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != saved["raw_sha256"][name]:
                raise ValueError("raw worker hash changed")
            workers[name] = json.loads(path.read_text())
        recomputed = validate_pair(workers, atol=saved["correctness_atol"])
        if recomputed != {k:v for k,v in saved.items() if k not in ("raw_sha256", "provenance")}:
            raise ValueError("saved comparison differs from raw artifacts")
        print(json.dumps({"status": "ok", "raw_workers_checked": 2, "performance_claims_enabled": False}))
        return 0
    if (args.out_dir.exists() or args.forward_runs < 1 or args.warmup_runs < 0
            or not np.isfinite(args.atol) or args.atol <= 0):
        parser.error("choose a fresh output directory, valid run counts and a finite positive tolerance")
    args.out_dir.mkdir(parents=True)
    workers, hashes = {}, {}
    from orion.experimental.wpc_periodicity import real_lattigo_library_path
    from tools.run_wpc_selective_orion import source_hashes
    binary = real_lattigo_library_path()
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    sources = source_hashes()
    for policy in ("online", "hybrid"):
        env = dict(os.environ)
        env.update({"ORION_WPC_SELECTIVE_POLICY": policy, "ORION_LATTIGO_CLEAR_BACKEND": "0",
            "ORION_SINGLE_SLOT_LAYER_CACHE": "1", "ORION_SINGLE_SLOT_ENCODE_WORKERS": "1",
            "ORION_LATTIGO_STREAMING_LT": "0", "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT": "0",
            "ORION_WPC_PERIODICITY_PROFILE": "0", "PYTHONUNBUFFERED": "1"})
        path = args.out_dir/f"{policy}.json"
        command = [sys.executable, str(ROOT/"tools/run_lattigo_e2e_compare.py"), "--mode", "dense",
            "--backend", "lattigo", "--network", args.network, "--out", str(path.resolve()),
            "--seed", str(args.seed), "--warmup-runs", str(args.warmup_runs),
            "--forward-runs", str(args.forward_runs), "--io-mode", "none"]
        print(json.dumps({"event": "worker_start", "policy": policy, "command": command}), flush=True)
        with (args.out_dir/f"{policy}.log").open("w") as log:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if hashlib.sha256(binary.read_bytes()).hexdigest() != binary_sha or source_hashes() != sources:
            raise ValueError("source/backend changed during the experiment")
        workers[policy] = json.loads(path.read_text())
        hashes[policy] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = validate_pair(workers, atol=args.atol)
    result.update({"raw_sha256": hashes, "provenance": {"backend_sha256": binary_sha,
        "source_sha256": sources, "python": sys.version,
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}})
    (args.out_dir/"comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "ok", "summary": result["summary"], "result": str(args.out_dir/"comparison.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
