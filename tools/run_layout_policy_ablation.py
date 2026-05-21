#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.experimental.layout_policy_ablation import (
    attach_backend_runtime_anchors,
    attach_non_ckks_simulation,
    attach_python_runtime_anchors,
    attach_runtime_anchor,
    build_planner_ablation,
    normalize_policies,
    run_backend_runtime_anchors,
    run_non_ckks_layout_simulation,
    run_python_runtime_anchors,
    run_runtime_anchor,
)
from orion.experimental.layout_policy_ablation import PROVIDER_PRESSURE_SUMMARY_KEYS


DEFAULT_CACHE_ROOT = Path("/run/media/anakano/7TB/haloed-cache")
DEFAULT_OUT = REPO_ROOT / ".tmp" / "results" / "layout_policy_ablation_u22_64.csv"
SUMMARY_COLUMNS = [
    "policy",
    "metric_source",
    "relayouts",
    "halo_redundancy_ratio",
    "total_ciphertext_tiles",
    "stored_slots",
    "relayout_rotation_estimate",
    "relayout_mask_mult_estimate",
    "relayout_depth_estimate",
    "lt_bsgs_rotation_estimate",
    "compact_fallback_penalty_estimate",
    "bootstrap_proxy",
    "bootstrap_count",
    "he_forward_s",
    "mae",
    "dice",
    "speedup_vs_fixed_max",
    "runtime_status",
    "runtime_reason",
    "non_ckks_forward_s",
    "max_abs",
    "layout_alignment_ok",
    "objective",
    *PROVIDER_PRESSURE_SUMMARY_KEYS,
]


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(part.strip().lower() for part in str(value).split(",") if part.strip()))
    unknown = [mode for mode in modes if mode not in {"planner", "e2e", "simulate"}]
    if unknown:
        raise ValueError(f"unknown mode(s): {', '.join(unknown)}")
    if not modes:
        return ("planner",)
    if ("e2e" in modes or "simulate" in modes) and "planner" not in modes:
        return ("planner", *modes)
    return modes


def _write_outputs(payload: dict[str, Any], out_csv: Path) -> Path:
    out_csv = Path(out_csv)
    out_json = out_csv.with_suffix(".json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in payload.get("policies", []):
            writer.writerow(dict(row))
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_json


def run(args: argparse.Namespace) -> dict[str, Any]:
    modes = _parse_modes(str(args.mode))
    policies = normalize_policies(args.policies)
    payload = build_planner_ablation(network=str(args.network), policies=policies, slots=int(args.slots))
    payload["backend"] = str(args.backend)
    payload["mode"] = list(modes)
    payload["cache_root"] = str(args.cache_root)
    if "simulate" in modes:
        simulation = run_non_ckks_layout_simulation(payload, seed=int(args.sim_seed))
        payload = attach_non_ckks_simulation(payload, simulation)
    if "e2e" in modes:
        if str(args.backend) == "python":
            anchors = run_python_runtime_anchors(payload, seed=int(args.sim_seed))
            payload = attach_python_runtime_anchors(payload, anchors)
        elif str(args.backend) == "lattigo":
            anchors = run_backend_runtime_anchors(
                payload,
                backend=str(args.backend),
                cache_root=Path(args.cache_root),
                compile_timeout_s=int(args.compile_timeout_s),
                python=Path(args.python) if args.python is not None else None,
            )
            payload = attach_backend_runtime_anchors(payload, anchors)
        else:
            anchor = run_runtime_anchor(
                network=str(args.network),
                backend=str(args.backend),
                cache_root=Path(args.cache_root),
                compile_timeout_s=int(args.compile_timeout_s),
                python=Path(args.python) if args.python is not None else None,
            )
            payload = attach_runtime_anchor(payload, anchor)
    out_json = _write_outputs(payload, Path(args.out))
    payload["out_csv"] = str(Path(args.out))
    payload["out_json"] = str(out_json)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run U22 full-graph halo layout-policy planner ablation.")
    parser.add_argument(
        "--network",
        choices=("u22_64_base32", "u22_128_base32", "u22_256_base32"),
        default="u22_64_base32",
    )
    parser.add_argument("--policies", nargs="+", default=("fixed_max", "eager", "greedy", "dp"))
    parser.add_argument("--mode", default="planner", help="Comma-separated: planner, planner,simulate, or planner,e2e.")
    parser.add_argument("--backend", choices=("lattigo", "cheddar", "python"), default="lattigo")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--slots", type=int, default=32768)
    parser.add_argument("--compile-timeout-s", type=int, default=10800)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--sim-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(json.dumps({"status": payload.get("status"), "out_csv": payload["out_csv"], "out_json": payload["out_json"]}, indent=2))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
