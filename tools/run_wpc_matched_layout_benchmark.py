#!/usr/bin/env python3
"""Stage 31: repeated five-treatment Orion/CIPS benchmark and offline review."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.experimental.wpc_decoder_geometry import archive_config_input, decoder_geometry, resolve_decoder_config
from orion.experimental.wpc_layout_gate import TREATMENTS
from orion.experimental.wpc_matched_layout_benchmark import balanced_orders, compare_block, summarize_blocks, COMPONENTS, RATIOS
from orion.experimental.wpc_orion_layout_control import native_signature, NATIVE_PREPARATION_POLICY
from orion.experimental.wpc_evidence_validation import finite, integer, sha256
from tools.run_wpc_cips_trained_isolated_benchmark import _sample_worker
from tools.run_wpc_layout_gate import SOURCE_FILES as GATE_SOURCES, parser as gate_parser, save, validate_request

SOURCE_FILES = (*GATE_SOURCES, "orion/experimental/wpc_matched_layout_benchmark.py",
                "tools/run_wpc_matched_layout_benchmark.py", "tools/run_wpc_matched_layout_server.sh")


def source_hashes():
    return {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def parser():
    result = gate_parser()
    result.description = __doc__
    result.set_defaults(forward_runs=10, warmup_runs=2,
                        out_dir=REPO_ROOT / ".tmp/results/honours/31_wpc_matched_layout_benchmark/server_run1")
    result.add_argument("--trial-blocks", type=int, default=10)
    result.add_argument("--order-seed", type=int, default=0)
    result.add_argument("--bootstrap-resamples", type=int, default=10000)
    result.add_argument("--bootstrap-seed", type=int, default=0)
    result.add_argument("--smoke", action="store_true", help="one block, no confidence intervals or performance conclusion")
    return result


def render_report(data):
    summary = data["summary"]
    lines = ["# Matched-function Orion–CIPS layout/storage benchmark", "",
             f"Status: {data['status']}. Fresh-process blocks: {summary['block_count']}. Smoke: {data['request']['smoke']}.", "",
             "Same checkpoint, exact synthetic features, flattened Rotation Padding, Cheb7, bootstrap and CKKS levels. Native preparation uses channel-pruned block regeneration; Stage-30 timings are not reused.", "",
             "| Treatment | Forward (s) | Prep + Encode (%) | Decompress (%) | Logical resident (MiB) | Sampled measured peak RSS (MiB) | LT rotations |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for treatment in TREATMENTS:
        row = summary["metrics"][treatment]
        mean = lambda key: row[key]["mean"]
        lines.append(f"| {treatment} | {mean('forward_median_s'):.6f} | "
                     f"{mean('online_prepare_s_pct') + mean('online_encode_s_pct'):.3f} | {mean('decompression_s_pct'):.3f} | "
                     f"{mean('logical_resident_bytes')/2**20:.3f} | {mean('sampled_measured_peak_rss_bytes')/2**20:.3f} | "
                     f"{summary['operations'][treatment]['linear_transform_rotation']} |")
    lines += ["", "Forward entries are means of within-process medians. Component shares are means of within-forward ratios. RSS is the mean of sampled worker maxima, not an allocation ceiling.", "",
              "| Paired forward ratio (numerator / denominator) | Mean | 95% CI for mean |",
              "|---|---:|---:|"]
    for left, right in RATIOS:
        label = f"{left}_over_{right}"
        row = summary["paired_latency_ratios"][label]
        interval = row["ci95_mean"]
        rendered = f"[{interval[0]:.6f}, {interval[1]:.6f}]" if interval else "not estimated (smoke)"
        lines.append(f"| {label.replace('_over_', ' / ')} | {row['mean']:.6f} | {rendered} |")
    lines += ["", "## Forward wall accounting", "",
              "Mean seconds per forward; components plus residual close to the mean wall, not the median above.", "",
              "| Treatment | Wall | Preparation | Encode | Decompress | LT evaluation | Activation | Bootstrap | Other |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for treatment in TREATMENTS:
        row = summary["metrics"][treatment]
        values = [row[key]["mean"] for key in ("forward_mean_s", *COMPONENTS, "other_forward_s")]
        lines.append(f"| {treatment} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", summary["inference"] + ".", "",
              "Ten-block Williams cycles balance treatment positions and ordered immediate predecessors; they do not enumerate all 120 treatment permutations. Warmups and the correctness preflight are excluded. Output validation and lifecycle tracing are outside timed forwards. Every measured output and operation-count sample is retained and independently checked.", "",
              "Native Encode includes the GenerateLinearTransform allocation/binding call; CIPS Encode is the narrower backend timer. Components are not interchangeable with historical Step-1 categories. Other includes concat, bias/rescale, wrappers and remaining call/allocation overhead; it is not a cryptographic operator measurement.", "",
              f"Maximum final-output delta across treatments: {summary['maximum_final_output_delta']:.3e}. Within-layout storage policies have identical instrumented operations; layout operation counts may differ. Individual additions/multiplications are not separately instrumented.", "",
              "**Scope:** functional, security-unassessed CKKS parameters, one checkpoint-derived decoder stage and synthetic internal tensors. This is not a complete-network crossover, secure deployment, dataset accuracy evaluation or a general claim about native Orion. The order-balanced process-block intervals describe this run and do not establish reproducibility across hosts or sessions.", "",
              "Raw outputs, timings, RSS/guard observations, exact configuration bytes and source/checkpoint/binary hashes are archived. Hashes establish content identity, not authenticity; checkpoint and binary bytes are not bundled.", ""]
    return "\n".join(lines)


def manifest_row(path, root):
    content = path.read_bytes()
    return {"path": str(path.relative_to(root)), "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def validate_frozen_request(request):
    if type(request["smoke"]) is not bool or request.get("security_assessed") is not False:
        raise ValueError("smoke/security scope must be explicit")
    orders = balanced_orders(len(request["orders"]), seed=request["order_seed"], smoke=request["smoke"])
    if request["orders"] != orders or request["native_preparation_policy"] != NATIVE_PREPARATION_POLICY:
        raise ValueError("frozen schedule/preparation policy mismatch")
    worker = request["worker_request"]
    if integer(worker["forward_runs"], name="measured forwards", minimum=1) < (1 if request["smoke"] else 3) or integer(worker["warmup_runs"], name="warmups") < (0 if request["smoke"] else 1):
        raise ValueError("frozen benchmark does not have required repetitions/warmups")
    if worker["verify_exact_qp"] is not True or worker["retain_sample_outputs"] is not True:
        raise ValueError("exact Q/P verification and every measured output are required")
    integer(request["bootstrap_resamples"], name="resamples", minimum=100)
    for value in request["resource_guards"].values():
        if finite(value, name="resource guard", minimum=0) <= 0:
            raise ValueError("positive resource guards required")
    return orders


def run(args):
    if args.review_dir is not None:
        return review(args.review_dir)
    orders = balanced_orders(args.trial_blocks, seed=args.order_seed, smoke=args.smoke)
    if args.forward_runs < (1 if args.smoke else 3) or args.warmup_runs < (0 if args.smoke else 1):
        raise ValueError("benchmark requires at least three measured forwards and one warmup; smoke permits one/zero")
    if args.bootstrap_resamples < 100 or not args.verify_exact_qp:
        raise ValueError("require exact Q/P verification and at least 100 bootstrap resamples")
    if any(not math.isfinite(v) or v <= 0 for v in (args.max_worker_rss_mib, args.worker_timeout_s, args.atol,
            args.feature_std, args.bsgs_ratio, args.bound_headroom, args.rss_sample_ms)) or args.bound_headroom < 1:
        raise ValueError("invalid correctness/resource/packing parameters")
    config, source = resolve_decoder_config(args)
    geometry = decoder_geometry(args.logn, args.height, args.width, len(config["ckks_params"]["LogQ"]) - 1)
    native_signature(geometry["low_shape"], geometry["slots"], 2)
    native_signature(geometry["high_shape"], geometry["slots"], 1)
    if 32 * args.height * args.width % geometry["slots"]:
        raise ValueError("native concat requires aligned branches")
    request = {"ckks_config": config, "configuration_source": source, "geometry": geometry, "orders": orders,
        "order_seed": args.order_seed, "smoke": args.smoke, "security_assessed": False,
        "bootstrap_resamples": args.bootstrap_resamples, "bootstrap_seed": args.bootstrap_seed,
        "native_preparation_policy": NATIVE_PREPARATION_POLICY,
        "resource_guards": {"max_worker_rss_mib": args.max_worker_rss_mib, "worker_timeout_s": args.worker_timeout_s},
        "worker_request": {"seed": args.seed, "bsgs_ratio": args.bsgs_ratio, "feature_std": args.feature_std,
            "bound_headroom": args.bound_headroom, "atol": args.atol, "forward_runs": args.forward_runs,
            "warmup_runs": args.warmup_runs, "verify_exact_qp": args.verify_exact_qp, "retain_sample_outputs": True}}
    validate_frozen_request(request)
    print(json.dumps(request, indent=2, allow_nan=False), flush=True)
    if args.plan_only:
        return 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        available = next((int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines() if line.startswith("MemAvailable:")), None)
        if available is None or args.max_worker_rss_mib * 2**20 > .8 * available:
            raise ValueError("RSS budget exceeds 80% of available host memory")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"missing checkpoint: {checkpoint}")
    root = args.out_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    provenance = {"implementation_sha256": source_hashes(),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "repository_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True),
        "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "configuration_artifact": archive_config_input(config, source, root / "configuration_input.json"), "artifacts": []}
    save(root / "requested_run.json", request)
    blocks = []
    try:
        for number, order in enumerate(orders, 1):
            directory = root / f"block_{number:03d}"
            directory.mkdir()
            workers, samples = {}, {}
            for treatment in order:
                layout, mode = treatment.split("/")
                destination = directory / layout
                destination.mkdir(exist_ok=True)
                print(json.dumps({"event": "worker_start", "block": number, "treatment": treatment, "order": order}), flush=True)
                worker, rss, command = _sample_worker(mode=mode, checkpoint=checkpoint, out_dir=destination,
                    args=args, worker_args=["--layout", layout, "--retain-sample-outputs"])
                validate_request(worker, rss, request)
                workers[treatment], samples[treatment] = worker, rss
                worker_path = destination / f"{mode}.worker.json"
                if json.loads(worker_path.read_bytes()) != worker:
                    raise ValueError("worker result changed after loading")
                save(destination / f"{mode}.rss.json", rss)
                for suffix in ("worker.json", "rss.json", "worker.log", "phase.json"):
                    provenance["artifacts"].append({**manifest_row(destination / f"{mode}.{suffix}", root), "command": command})
                print(json.dumps({"event": "worker_complete", "block": number, "treatment": treatment}), flush=True)
            block = compare_block(workers, samples, order=order, atol=args.atol,
                                  checkpoint_sha256=provenance["checkpoint"]["sha256"], config=config)
            save(directory / "block.json", block)
            provenance["artifacts"].append(manifest_row(directory / "block.json", root))
            blocks.append(block)
        summary = summarize_blocks(blocks, orders=orders, smoke=args.smoke,
                                   resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
        if source_hashes() != provenance["implementation_sha256"] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != provenance["checkpoint"]["sha256"]:
            raise ValueError("source/checkpoint changed during run")
        data = {"schema_version": 1, "profile": "wpc_orion_matched_layout_repeated_benchmark", "status": "ok",
                "request": request, "provenance": provenance, "summary": summary, "security_assessed": False}
        save(root / "comparison.json", data)
        (root / "comparison.md").write_text(render_report(data))
        review(root)
        print(f"MATCHED LAYOUT BENCHMARK COMPLETE: {root / 'comparison.md'}", flush=True)
        return 0
    except Exception as error:
        save(root / "comparison.failed.json", {"status": "failed", "error": str(error), "request": request,
                                              "completed_block_count": len(blocks), "provenance": provenance})
        raise


def review(directory):
    root = directory.expanduser().resolve()
    data = json.loads((root / "comparison.json").read_bytes())
    if data.get("schema_version") != 1 or data.get("profile") != "wpc_orion_matched_layout_repeated_benchmark" or data.get("status") != "ok" or data.get("security_assessed") is not False:
        raise ValueError("not a successful Stage-31 benchmark")
    request, provenance = data["request"], data["provenance"]
    orders = validate_frozen_request(request)
    expected = {f"block_{i:03d}/{layout}/{mode}.{suffix}" for i in range(1, len(orders) + 1)
                for layout, mode in (t.split("/") for t in TREATMENTS) for suffix in ("worker.json", "rss.json", "worker.log", "phase.json")}
    expected.update(f"block_{i:03d}/block.json" for i in range(1, len(orders) + 1))
    artifacts = provenance["artifacts"]
    if len(artifacts) != len(expected) or {r["path"] for r in artifacts} != expected or set(provenance["implementation_sha256"]) != set(SOURCE_FILES):
        raise ValueError("incomplete/duplicate artifact or source manifest")
    sha256(provenance["checkpoint"]["sha256"], name="frozen checkpoint hash")
    for digest in provenance["implementation_sha256"].values():
        sha256(digest, name="archived source hash")
    if provenance["configuration_artifact"]["path"] != "configuration_input.json":
        raise ValueError("configuration archive path mismatch")
    for row in [*artifacts, provenance["configuration_artifact"]]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or manifest_row(path, root) != {k: row[k] for k in ("path", "size_bytes", "sha256")}:
            raise ValueError(f"artifact path/checksum/size mismatch: {row['path']}")
    config_bytes = (root / provenance["configuration_artifact"]["path"]).read_bytes()
    if json.loads(config_bytes) != request["ckks_config"] or json.loads((root / "requested_run.json").read_bytes()) != request:
        raise ValueError("configuration/request evidence mismatch")
    if request["configuration_source"]["kind"] == "json_file" and hashlib.sha256(config_bytes).hexdigest() != request["configuration_source"]["sha256"]:
        raise ValueError("configuration source hash mismatch")
    blocks = []
    for number, order in enumerate(orders, 1):
        directory = root / f"block_{number:03d}"
        workers, samples = {}, {}
        for treatment in TREATMENTS:
            layout, mode = treatment.split("/")
            workers[treatment] = json.loads((directory / layout / f"{mode}.worker.json").read_bytes())
            samples[treatment] = json.loads((directory / layout / f"{mode}.rss.json").read_bytes())
            validate_request(workers[treatment], samples[treatment], request)
        block = compare_block(workers, samples, order=order, atol=request["worker_request"]["atol"],
                              checkpoint_sha256=provenance["checkpoint"]["sha256"], config=request["ckks_config"])
        if json.loads((directory / "block.json").read_bytes()) != block:
            raise ValueError("block summary does not match raw evidence")
        blocks.append(block)
    summary = summarize_blocks(blocks, orders=orders, smoke=request["smoke"],
                               resamples=request["bootstrap_resamples"], seed=request["bootstrap_seed"])
    if data["summary"] != summary or (root / "comparison.md").read_text() != render_report(data):
        raise ValueError("summary/report does not match raw evidence")
    current = source_hashes()
    print(json.dumps({"status": "ok", "reviewed_workers": 5 * len(blocks), "raw_artifacts_checked": len(artifacts),
        "current_source_hash_differences": [name for name, digest in provenance["implementation_sha256"].items() if current.get(name) != digest],
        "scope": "every measured output, timing/counter closure, hashes and paired-block statistics checked; binary/checkpoint bytes not bundled"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except (ValueError, RuntimeError, KeyError, TypeError, IndexError, OSError) as error:
        print(f"MATCHED LAYOUT BENCHMARK FAILED: {str(error).splitlines()[0]}", file=sys.stderr, flush=True)
        raise SystemExit(1)
