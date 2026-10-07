#!/usr/bin/env python3
"""Balanced three-way online-Encode/full-QP/compressed-QP decoder benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.experimental.wpc_online_encode_benchmark import (
    MODES, balanced_orders, compare_block, process_block_summary,
)
from tools.run_wpc_cips_trained_isolated_benchmark import _parser as worker_parser, _sample_worker
from orion.experimental.wpc_decoder_geometry import archive_config_input, decoder_geometry, resolve_decoder_config

IMPLEMENTATION_FILES = (
    "tools/run_wpc_online_encode_benchmark.py", "tools/run_wpc_cips_trained_isolated_worker.py",
    "tools/run_wpc_cips_trained_isolated_benchmark.py", "tools/run_wpc_cips_trained_decoder.py",
    "orion/experimental/wpc_online_encode_benchmark.py", "orion/experimental/wpc_cips_layer.py",
    "orion/experimental/wpc_cips_upsample.py", "orion/backend/lattigo/bindings.py",
    "orion/backend/lattigo/wpc_online_encode.go", "orion/backend/lattigo/wpc_compression.go",
    "orion/backend/lattigo/lineartransform.go", "orion/backend/lattigo/scheme.go",
    "orion/experimental/wpc_decoder_geometry.py", "orion/experimental/wpc_cips_trained_benchmark.py",
    "orion/backend/lattigo/wpc_parameter_manifest.go", "orion/backend/lattigo/bootstrapper.go",
    "lattigo/circuits/ckks/bootstrapping/parameters_literal.go", "lattigo/circuits/ckks/bootstrapping/parameters.go",
    "lattigo/circuits/ckks/bootstrapping/keys.go",
    "tools/run_wpc_decoder_feasibility.py", "tools/run_wpc_decoder_feasibility_server.sh",
    "configs/wpc_decoder_scale_functional.json", "orion/experimental/wpc_cips_checkpoint.py",
    "orion/experimental/wpc_cips_branches.py", "orion/experimental/wpc_cips_baseline.py",
    "orion/experimental/wpc_evidence_validation.py",
)


def implementation_hashes() -> dict[str, str]:
    return {relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in IMPLEMENTATION_FILES}

def parser() -> argparse.ArgumentParser:
    result = worker_parser()
    result.description = __doc__
    result.set_defaults(out_dir=REPO_ROOT / ".tmp/results/honours/28_wpc_online_encode_benchmark/server_run1",
        checkpoint=REPO_ROOT / "checkpoints/wpc_rotation_padding_covid19_cheb7_audited_restart/rotation_padding_best.pt")
    result.add_argument("--trial-blocks", type=int, default=6)
    result.add_argument("--order-seed", type=int, default=0)
    result.add_argument("--ci-resamples", type=int, default=10000)
    result.add_argument("--smoke", action="store_true", help="one correctness-only block; no confidence intervals/performance claims")
    return result


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    experiment = payload["blocks"][0].get("experiment", {})
    security_note = (f"**Security not assessed:** LogN={experiment.get('logn', 10)}, "
                     f"output shape {experiment.get('high_shape', [1, 32, 8, 8])}; neither a secure-deployment nor a complete-network benchmark.")
    lines = ["# Three-way CIPS online-Encode benchmark", "",
             f"Status: {payload['status']}. Independent process blocks: {len(payload['blocks'])}.", "",
             "Same checkpoint, exact feature values, CIPS layout, CKKS configuration, and homomorphic operations in all modes.", "",
             security_note, "",
             "| Mode | Forward (s) | Encode share | Preparation + Encode share | Decompression share | Logical resident (MiB) | Sampled online peak RSS (MiB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        row = summary["by_mode"][mode]
        mean = lambda key: row[key]["mean"]
        lines.append(f"| {mode} | {mean('forward_median_s'):.6f} | {mean('online_encode_median_pct'):.3f}% | "
                     f"{mean('online_materialization_median_pct'):.3f}% | {mean('decompression_median_pct'):.3f}% | "
                     f"{mean('logical_resident_bytes') / 2**20:.3f} | {mean('sampled_online_peak_rss_bytes') / 2**20:.3f} |")
    lines += ["", "Entries are means of per-process-block statistics; shares are within-forward ratios, not ratios of aggregate medians.", "",
              "| Paired latency ratio | Mean | 95% CI for mean |", "|---|---:|---:|"]
    for key, row in summary["paired_ratios"].items():
        ci = row["ci95_mean"]
        text = f"[{ci[0]:.6f}, {ci[1]:.6f}]" if ci else "not estimated (smoke)"
        lines.append(f"| {key} | {row['mean']:.6f} | {text} |")
    lines += ["", summary["uncertainty_method"] + ".", "",
              "Lifecycle tracing and output validation run outside timed forwards. Encode counters count actual successful linear-transform Encode invocations, not individual diagonals.", "",
              "Online mode stores compact float32 slot-period recipes; it expands them and allocates Q/P online. Bias and concat plaintexts are preencoded in all modes.", "",
              "Preparation and Encode are separate: neither has the same boundary as the historical Orion Step-1 layer-cache category. Do not subtract these numbers from the whole-model profiles.", "",
              "RSS is an externally sampled maximum, not a guaranteed transient allocation peak. Warmups are excluded.", "",
              ("This smoke block uses a fixed treatment order for correctness only." if payload["smoke"] else
               "Fresh-process order is balanced across all six treatment permutations."), "",
              "This isolates storage/materialization within CIPS. It does not establish the Orion-versus-WPC layout crossover.", ""]
    if payload["smoke"]:
        lines += ["**Smoke run only: no balanced-order performance claim or confidence interval.**", ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    config, source = resolve_decoder_config(args)
    geometry = decoder_geometry(args.logn, args.height, args.width, len(config["ckks_params"]["LogQ"]) - 1)
    for name in ("rss_sample_ms", "atol", "feature_std", "bsgs_ratio", "bound_headroom"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise SystemExit(f"{name} must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.max_worker_rss_mib, args.worker_timeout_s)):
        raise SystemExit("worker guards must be finite and nonnegative")
    if args.bound_headroom < 1:
        raise SystemExit("bound-headroom must be at least one")
    if args.warmup_runs < 0 or args.forward_runs <= 0 or args.rss_sample_ms <= 0 or args.atol <= 0:
        raise SystemExit("invalid forward/warmup count, RSS interval, or tolerance")
    orders = balanced_orders(args.trial_blocks, seed=args.order_seed, smoke=args.smoke)
    if args.ci_resamples < 100:
        raise SystemExit("ci-resamples must be at least 100")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    root = args.out_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"refusing to overwrite existing run directory: {root}; choose a fresh --out-dir")
    root.mkdir(parents=True, exist_ok=True)
    configuration_artifact = archive_config_input(config, source, root / "configuration_input.json")
    (root / "requested_run.json").write_text(json.dumps({"ckks_config": config, "configuration_source": source,
        "geometry": geometry, "security_assessed": False, "verify_exact_qp": args.verify_exact_qp,
        "resource_guards": {"max_worker_rss_mib": args.max_worker_rss_mib, "worker_timeout_s": args.worker_timeout_s}}, indent=2) + "\n")
    implementation = implementation_hashes()
    manifest = []
    blocks = []
    for index, order in enumerate(orders):
        directory = root / f"block_{index + 1:03d}"
        directory.mkdir()
        workers, sampling = {}, {}
        for mode in order:
            print(json.dumps({"event": "worker_start", "block": index + 1, "mode": mode, "order": order}), flush=True)
            worker, rss, command = _sample_worker(mode=mode, checkpoint=checkpoint, out_dir=directory, args=args)
            if worker["experiment"]["ckks_config"] != config or worker["experiment"]["geometry"] != geometry or worker["experiment"]["configuration_source"] != source:
                raise RuntimeError("worker configuration changed after the controller's request was frozen")
            workers[mode], sampling[mode] = worker, rss
            file = directory / f"{mode}.worker.json"
            worker_bytes = file.read_bytes()
            if json.loads(worker_bytes) != worker:
                raise RuntimeError("raw worker artifact changed after loading; refusing mismatched provenance")
            rss_file = directory / f"{mode}.rss.json"
            rss_file.write_text(json.dumps(rss, indent=2, allow_nan=False) + "\n")
            manifest.append({"path": str(file.relative_to(root)), "sha256": hashlib.sha256(worker_bytes).hexdigest(),
                             "size_bytes": file.stat().st_size, "command": command,
                             "rss_path": str(rss_file.relative_to(root)), "rss_sha256": hashlib.sha256(rss_file.read_bytes()).hexdigest()})
            print(json.dumps({"event": "worker_complete", "block": index + 1, "mode": mode}), flush=True)
        block = compare_block(workers, sampling, order=order, atol=args.atol)
        block["block_index"] = index + 1
        blocks.append(block)
        (directory / "comparison.json").write_text(json.dumps(block, indent=2, allow_nan=False) + "\n")
    summary = process_block_summary(blocks, smoke=args.smoke, resamples=args.ci_resamples, seed=args.order_seed)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)
    except subprocess.SubprocessError:
        commit, status = None, "unavailable"
    if implementation_hashes() != implementation:
        raise RuntimeError("implementation files changed during the run; preserve partial artifacts and rerun from a frozen checkout")
    payload = {"schema_version": 2, "profile": "wpc_cips_three_way_online_encode_process_blocks", "status": "ok",
               "smoke": args.smoke, "blocks": blocks, "summary": summary,
               "checkpoint": {"path": str(checkpoint), "sha256": blocks[0]["checkpoint_sha256"]},
               "provenance": {"repository_commit": commit, "repository_status": status, "implementation_sha256": implementation,
                              "worker_artifacts": manifest, "configuration_artifact": configuration_artifact},
               "acceptance": {"all_blocks_independently_validated": True, "checkpoint_and_features_consistent": True,
                              "balanced_order_or_explicit_smoke": summary["order_balanced"] or args.smoke, "valid": True}}
    (root / "comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (root / "process_blocks.csv").open("w", newline="") as handle:
        fields = ["block", "mode", "position", *blocks[0]["rows"][MODES[0]].keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block in blocks:
            for mode in MODES:
                writer.writerow({"block": block["block_index"], "mode": mode,
                                 "position": block["order"].index(mode) + 1, **block["rows"][mode]})
    (root / "comparison.md").write_text(render_report(payload))
    print(json.dumps({"status": "ok", "result": str(root / "comparison.json"),
                      "report": str(root / "comparison.md"), "smoke": args.smoke}, indent=2))
    return 0


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    try:
        code = main()
    except (ValueError, RuntimeError, KeyError) as error:
        print(f"THREE-WAY BENCHMARK FAILED: {error}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
