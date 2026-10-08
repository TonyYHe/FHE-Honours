#!/usr/bin/env python3
"""Stage 30: fresh-process matched-function Orion/CIPS correctness/resource gate."""
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
from orion.experimental.wpc_layout_gate import TREATMENTS, validate_layout_gate
from orion.experimental.wpc_orion_layout_control import native_signature
from tools.run_wpc_cips_trained_isolated_benchmark import _parser, _sample_worker
from tools.run_wpc_online_encode_benchmark import IMPLEMENTATION_FILES

SOURCE_FILES = tuple(dict.fromkeys((*IMPLEMENTATION_FILES,
    "tools/run_wpc_layout_gate.py", "tools/run_wpc_layout_gate_server.sh",
    "orion/experimental/wpc_layout_gate.py", "orion/experimental/wpc_orion_layout_control.py",
    "orion/core/packing.py", "orion/backend/python/lt_evaluator.py", "orion/backend/python/tensors.py",
    "orion/backend/lattigo/wpc_layout_control.go", "orion/backend/lattigo/tensors.go",
    "orion/backend/lattigo/evaluator.go")))


def source_hashes():
    return {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def parser():
    result = _parser()
    result.description = __doc__
    result.set_defaults(ckks_config=REPO_ROOT / "configs/wpc_decoder_scale_functional.json",
        checkpoint=REPO_ROOT / "checkpoints/wpc_rotation_padding_covid19_cheb7_audited_restart/rotation_padding_best.pt",
        height=16, width=16, warmup_runs=0, forward_runs=1, verify_exact_qp=True,
        max_worker_rss_mib=8192, worker_timeout_s=1800, atol=1e-5,
        out_dir=REPO_ROOT / ".tmp/results/honours/30_wpc_orion_layout_gate/server_run1")
    result.add_argument("--plan-only", action="store_true")
    result.add_argument("--review-dir", type=Path, help="independently check a transferred Stage-30 run; no FHE execution")
    return result


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def render_report(data):
    lines = ["# Matched-function Orion–CIPS decoder gate", "", f"Status: {data['status']}.", "",
        "Both layouts evaluate the same checkpoint, synthetic features, flattened Rotation Padding, trained Cheb7 and bootstrap at the same CKKS levels.", "",
        "| Layout / storage | Max clear error | Diagnostic forward (s) | Logical resident (MiB) | Sampled measured RSS (MiB) | LT rotations |",
        "|---|---:|---:|---:|---:|---:|"]
    for treatment in TREATMENTS:
        row = data["rows"][treatment]
        lines.append(f"| {treatment} | {row['maximum_error_vs_independent_clear']:.3e} | {row['diagnostic_forward_s']:.6f} | "
                     f"{row['logical_resident_bytes']/2**20:.3f} | {row['sampled_measured_peak_rss_bytes']/2**20:.3f} | {row['operation_counters']['linear_transform_rotation']} |")
    lines += ["", f"Maximum output delta versus CIPS/full: {data['maximum_output_delta_vs_cips_full']:.3e}.", "",
        "Instrumented rotation/conjugation counts must match across storage policies within each layout; they need not match between layouts. Individual addition/multiplication counts are not separately measured.", "",
        "Native Orion uses its ordinary square-embedding diagonal packers (low gap 2, high gap 1), with an opt-in cyclic-boundary adapter. Its aligned concat copies and drops ciphertexts by one level; CIPS uses encoded permutation transforms. Native online preparation rebuilds ordinary diagonals from a float32 kernel, not CIPS periodic recipes.", "",
        "Full-Q/P transform evaluation is now measured, rather than reported as an unmeasured zero. Native Encode is GenerateLinearTransform call wall time; CIPS online Encode is the narrower backend Encode timer. These shares are not interchangeable with historical Step-1 categories.", "",
        "**Scope:** one fixed-order diagnostic forward per fresh process, no warmups. No latency ranking, confidence interval, secure-deployment claim or complete-network/layout-crossover conclusion. RSS is sampled, not an enforced allocation ceiling. Logical storage excludes runtime object overhead; native online Q/P estimates are checked against the native full worker's actual coefficient arrays.", "",
        "Raw worker/RSS files, exact configuration bytes, source/binary hashes and checkpoint identity are retained in gate.json provenance.", ""]
    return "\n".join(lines)


def run(args):
    if args.review_dir is not None:
        return review(args.review_dir)
    config, source = resolve_decoder_config(args)
    geometry = decoder_geometry(args.logn, args.height, args.width, len(config["ckks_params"]["LogQ"]) - 1)
    if args.forward_runs != 1 or args.warmup_runs != 0 or not args.verify_exact_qp:
        raise ValueError("layout gate requires zero warmups, one forward and exact compressed-Q/P verification")
    if any(not math.isfinite(value) or value <= 0 for value in (args.max_worker_rss_mib, args.worker_timeout_s,
            args.atol, args.feature_std, args.bsgs_ratio, args.bound_headroom, args.rss_sample_ms)):
        raise ValueError("layout gate requires positive RSS and timeout budgets")
    if args.bound_headroom < 1:
        raise ValueError("bootstrap bound headroom must be at least one")
    native_signature(geometry["low_shape"], geometry["slots"], 2)
    native_signature(geometry["high_shape"], geometry["slots"], 1)
    if 32 * args.height * args.width % geometry["slots"]:
        raise ValueError("native concat requires ciphertext-aligned 32-channel branches")
    request = {"ckks_config": config, "configuration_source": source, "geometry": geometry,
        "order": list(TREATMENTS), "padding_semantics": "flattened_spatial_cyclic", "security_assessed": False,
        "resource_guards": {"max_worker_rss_mib": args.max_worker_rss_mib, "worker_timeout_s": args.worker_timeout_s}}
    request["worker_request"] = {"seed": args.seed, "bsgs_ratio": args.bsgs_ratio,
        "feature_std": args.feature_std, "bound_headroom": args.bound_headroom, "atol": args.atol,
        "forward_runs": args.forward_runs, "warmup_runs": args.warmup_runs, "verify_exact_qp": args.verify_exact_qp}
    print(json.dumps(request, indent=2, allow_nan=False), flush=True)
    if args.plan_only:
        return 0
    memory_info = Path("/proc/meminfo")
    if memory_info.is_file():
        available = next((int(line.split()[1]) * 1024 for line in memory_info.read_text().splitlines()
                          if line.startswith("MemAvailable:")), None)
        if available is None or args.max_worker_rss_mib * 2**20 > .8 * available:
            raise ValueError("RSS budget exceeds 80% of currently available host memory")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"missing checkpoint: {checkpoint}")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    root = args.out_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    provenance = {"implementation_sha256": source_hashes(),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "repository_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash},
        "configuration_artifact": archive_config_input(config, source, root / "configuration_input.json"),
        "worker_artifacts": []}
    save(root / "requested_run.json", request)
    workers, samples = {}, {}
    try:
        for treatment in TREATMENTS:
            layout, mode = treatment.split("/")
            directory = root / layout
            directory.mkdir(exist_ok=True)
            print(json.dumps({"event": "worker_start", "treatment": treatment}), flush=True)
            worker, rss, command = _sample_worker(mode=mode, checkpoint=checkpoint, out_dir=directory,
                args=args, worker_args=["--layout", layout])
            if worker["experiment"]["configuration_source"] != source:
                raise ValueError("worker configuration source changed after request was frozen")
            validate_request(worker, rss, request)
            workers[treatment], samples[treatment] = worker, rss
            worker_path, rss_path = directory / f"{mode}.worker.json", directory / f"{mode}.rss.json"
            if json.loads(worker_path.read_bytes()) != worker:
                raise ValueError("worker evidence changed after loading")
            save(rss_path, rss)
            for path in (worker_path, rss_path, directory / f"{mode}.worker.log", directory / f"{mode}.phase.json"):
                content = path.read_bytes()
                provenance["worker_artifacts"].append({"path": str(path.relative_to(root)), "size_bytes": len(content),
                                                     "sha256": hashlib.sha256(content).hexdigest(), "command": command})
            print(json.dumps({"event": "worker_complete", "treatment": treatment}), flush=True)
        data = validate_layout_gate(workers, samples, atol=args.atol, checkpoint_sha256=checkpoint_hash, config=config)
        if source_hashes() != provenance["implementation_sha256"] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
            raise ValueError("source/checkpoint changed during the run; preserve partial evidence and rerun from a frozen checkout")
        data.update(schema_version=1, profile="wpc_orion_matched_function_layout_gate", request=request, provenance=provenance,
                    performance_claims_enabled=False, security_assessed=False)
        save(root / "gate.json", data)
        (root / "gate.md").write_text(render_report(data))
        print(f"MATCHED LAYOUT GATE COMPLETE: {root / 'gate.json'}", flush=True)
        return 0
    except Exception as error:
        save(root / "gate.failed.json", {"status": "failed", "error": str(error), "request": request, "provenance": provenance})
        raise


def validate_request(worker, rss, request):
    if worker["seed"] != request["worker_request"]["seed"]:
        raise ValueError("worker seed differs from request")
    if any(worker["experiment"][key] != value for key, value in request["worker_request"].items() if key != "seed"):
        raise ValueError("worker arguments differ from frozen request")
    if worker["experiment"]["ckks_config"] != request["ckks_config"] or worker["experiment"]["geometry"] != request["geometry"]:
        raise ValueError("worker configuration/geometry differs from request")
    if worker["experiment"]["configuration_source"] != request["configuration_source"]:
        raise ValueError("worker configuration provenance differs from request")
    if any(rss["guards"][key] != value for key, value in request["resource_guards"].items()):
        raise ValueError("worker resource guards differ from request")


def review(directory):
    root = directory.expanduser().resolve()
    stored = json.loads((root / "gate.json").read_bytes())
    if stored.get("status") != "ok" or stored.get("schema_version") != 1:
        raise ValueError("not a successful Stage-30 gate")
    provenance = stored["provenance"]
    expected = {f"{layout}/{mode}.{suffix}" for layout, mode in (t.split("/") for t in TREATMENTS)
                for suffix in ("worker.json", "rss.json", "worker.log", "phase.json")}
    artifacts = provenance["worker_artifacts"]
    if len(artifacts) != len(expected) or {r["path"] for r in artifacts} != expected:
        raise ValueError("incomplete or duplicate artifact manifest")
    for row in [*artifacts, provenance["configuration_artifact"]]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact path escapes review directory")
        content = path.read_bytes()
        if len(content) != row["size_bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"raw artifact checksum/size mismatch: {row['path']}")
    config_bytes = (root / provenance["configuration_artifact"]["path"]).read_bytes()
    request = stored["request"]
    if json.loads(config_bytes) != request["ckks_config"] or json.loads((root / "requested_run.json").read_bytes()) != request:
        raise ValueError("archived configuration/request mismatch")
    if request["configuration_source"]["kind"] == "json_file" and hashlib.sha256(config_bytes).hexdigest() != request["configuration_source"]["sha256"]:
        raise ValueError("archived configuration does not match original file digest")
    workers, samples = {}, {}
    for treatment in TREATMENTS:
        layout, mode = treatment.split("/")
        workers[treatment] = json.loads((root / layout / f"{mode}.worker.json").read_bytes())
        samples[treatment] = json.loads((root / layout / f"{mode}.rss.json").read_bytes())
        validate_request(workers[treatment], samples[treatment], request)
    result = validate_layout_gate(workers, samples, atol=request["worker_request"]["atol"],
        checkpoint_sha256=provenance["checkpoint"]["sha256"], config=request["ckks_config"])
    if any(stored.get(key) != value for key, value in result.items()):
        raise ValueError("stored gate summary does not match independent raw-artifact recomputation")
    if stored.get("performance_claims_enabled") is not False or stored.get("security_assessed") is not False:
        raise ValueError("diagnostic gate must not claim performance or assessed security")
    if (root / "gate.md").read_text() != render_report(stored):
        raise ValueError("Markdown report differs from checked JSON")
    current = source_hashes()
    changes = [name for name, digest in provenance["implementation_sha256"].items() if current.get(name) != digest]
    print(json.dumps({"status": "ok", "reviewed_workers": 5, "raw_artifacts_checked": len(artifacts),
        "current_source_hash_differences": changes, "scope": "raw numerical evidence and archived hashes checked; checkpoint/binary bytes are not bundled"}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except (ValueError, RuntimeError, KeyError) as error:
        print(f"MATCHED LAYOUT GATE FAILED: {str(error).splitlines()[0]}", file=sys.stderr, flush=True)
        raise SystemExit(1)
