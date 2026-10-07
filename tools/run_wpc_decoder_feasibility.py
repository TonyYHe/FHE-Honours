#!/usr/bin/env python3
"""One untimed-audit/one-forward process per storage mode before scale timing.

This correctness/resource gate does not assess cryptographic security, dataset
accuracy, complete-network performance, or the Orion-versus-WPC layout tradeoff.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.experimental.wpc_decoder_geometry import decoder_geometry, resolve_decoder_config
from tools.run_wpc_online_encode_benchmark import parser as benchmark_parser, run


def parser():
    result = benchmark_parser()
    result.description = __doc__
    result.set_defaults(ckks_config=REPO_ROOT / "configs/wpc_decoder_scale_functional.json",
        height=16, width=16, trial_blocks=1, smoke=True, warmup_runs=0, forward_runs=1,
        verify_exact_qp=True, max_worker_rss_mib=8192, worker_timeout_s=1800,
        out_dir=REPO_ROOT / ".tmp/results/honours/29_wpc_decoder_scale_feasibility/server_run1")
    result.add_argument("--plan-only", action="store_true", help="print configuration/geometry without loading a checkpoint or creating FHE keys")
    return result


def main() -> int:
    args = parser().parse_args()
    config, source = resolve_decoder_config(args)
    geometry = decoder_geometry(args.logn, args.height, args.width, len(config["ckks_params"]["LogQ"]) - 1)
    if args.max_worker_rss_mib <= 0 or args.worker_timeout_s <= 0:
        raise ValueError("feasibility requires positive RSS and timeout watchdog budgets")
    if not args.smoke or args.trial_blocks != 1 or args.warmup_runs != 0 or args.forward_runs != 1 or not args.verify_exact_qp:
        raise ValueError("feasibility requires exactly one smoke block, zero warmups, one measured forward, and exact Q/P verification")
    request = {"profile": "wpc_decoder_scale_feasibility_request", "configuration_source": source,
               "ckks_config": config, "geometry": geometry, "security_assessed": False,
               "resource_guards": {"max_worker_rss_mib": args.max_worker_rss_mib,
                                   "worker_timeout_s": args.worker_timeout_s}}
    print(json.dumps(request, indent=2, allow_nan=False), flush=True)
    if args.plan_only:
        return 0
    # Leave headroom on Linux; this is an admission check, not a reservation.
    memory_info = Path("/proc/meminfo")
    if memory_info.is_file():
        available = next((int(line.split()[1]) * 1024 for line in memory_info.read_text().splitlines()
                          if line.startswith("MemAvailable:")), None)
        if available is None or args.max_worker_rss_mib * 2**20 > .8 * available:
            raise ValueError("RSS budget exceeds 80% of currently available host memory; choose a smaller budget or another machine")
    root = args.out_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"refusing to overwrite existing run directory: {root}")
    try:
        run(args)
    except (ValueError, RuntimeError, KeyError) as error:
        if root.is_dir():
            # New run directory belongs to this invocation; preserve all partial evidence.
            if not (root / "feasibility.json").exists() and (root / "requested_run.json").exists() and not (root / "comparison.json").exists():
                (root / "feasibility.json").write_text(json.dumps({"schema_version": 1, "status": "failed",
                    "error": str(error), "security_assessed": False, "request": request}, indent=2) + "\n")
        raise
    comparison = json.loads((root / "comparison.json").read_text())
    block = root / "block_001"
    workers = {mode: json.loads((block / f"{mode}.worker.json").read_text())
               for mode in ("online_encode", "full", "compressed")}
    samples = {mode: json.loads((block / f"{mode}.rss.json").read_text()) for mode in workers}
    acceptance = {
        "three_mode_outputs_independently_validated": comparison["acceptance"]["valid"] is True,
        "exact_compressed_qp_verified": len(workers["compressed"]["qp_verification"]["rows"]) == geometry["learned_transform_count"],
        "all_workers_within_sampled_rss_budget": all(s["total_sample_count"] > 0 and max(s["peak_rss_by_phase"].values()) <= args.max_worker_rss_mib * 2**20 for s in samples.values()),
        "all_workers_within_time_budget": all(s["elapsed_s"] <= args.worker_timeout_s and s["termination_reason"] is None and s["return_code"] == 0 for s in samples.values()),
        "no_performance_or_security_claim": comparison["smoke"] is True and comparison["summary"]["performance_claims_enabled"] is False,
    }
    acceptance["valid"] = all(acceptance.values())
    payload = {"schema_version": 1, "profile": "wpc_decoder_scale_correctness_resource_gate",
        "status": "ok" if acceptance["valid"] else "invalid", "request": request,
        "checkpoint": comparison["checkpoint"], "acceptance": acceptance,
        "runtime_parameter_manifest": workers["full"]["runtime_parameter_manifest"],
        "resource_observations": {mode: {"sampled_all_phase_peak_rss_bytes": max(s["peak_rss_by_phase"].values()),
            "elapsed_s": s["elapsed_s"]} for mode, s in samples.items()},
        "scope": "same-CIPS synthetic decoder feasibility only; no security, layout-crossover or performance claim"}
    (root / "feasibility.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"DECODER FEASIBILITY {'PASSED' if acceptance['valid'] else 'FAILED'}: {root / 'feasibility.json'}", flush=True)
    return 0 if acceptance["valid"] else 1


def _report_failure(error: Exception) -> None:
    message = str(error)
    print(f"DECODER FEASIBILITY FAILED: {message}", file=sys.stderr)
    # Worker context can be a long JSON tail; keep the cause visible to tail -n.
    headline = message.splitlines()[0] if message else type(error).__name__
    print(f"FAILURE SUMMARY: {headline}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        result = main()
    except (ValueError, RuntimeError, KeyError) as error:
        _report_failure(error)
        result = 1
    raise SystemExit(result)
