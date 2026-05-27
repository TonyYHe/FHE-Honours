#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.auto_bootstrap import BootstrapSolver
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import layout_policy_ablation as lp
from orion.experimental.u22_phase1 import U22CompileRegistry
from orion.models.unet import UNet22
from orion.nn.module import Module


POLICIES = ("fixed_max", "always_fused", "dp")
IMAGE_SIZE = 224
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
BASE_CHANNELS = 32
ACTIVATION = "silu"
SILU_DEGREE = 7
RUNNER = REPO_ROOT / "tools" / "run_u22_base32_silu7_streaming_provider_e2e.py"
DEFAULT_RUN_ROOT_BASE = Path("/run/media/anakano/7TB/haloed-cache")
DEFAULT_OUT_CSV = REPO_ROOT / ".tmp" / "results" / "u22_224_base32_silu7_policy_time_rot_boot_relayout.csv"
DEFAULT_OUT_JSON = REPO_ROOT / ".tmp" / "results" / "u22_224_base32_silu7_policy_time_rot_boot_relayout.json"
DEFAULT_OUT_MD = REPO_ROOT / ".tmp" / "results" / "u22_224_base32_silu7_policy_time_rot_boot_relayout.md"
STRATEGY_LABELS = {
    "fixed_max": "Max-Re-Layout",
    "always_fused": "Always+Fusion",
    "dp": "HaloDP",
}

lp.LAYOUT_ESTIMATOR_DEFAULT = lp.LAYOUT_ESTIMATOR_TEMPLATE


class _DummyParams:
    def get_slots(self) -> int:
        return int(lp.DEFAULT_SLOTS)

    def get_debug_status(self) -> bool:
        return False

    def get_io_mode(self) -> str:
        return "none"

    def get_compile_save_resume(self) -> bool:
        return False


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {"status": "bad_json", "error": f"{type(exc).__name__}: {exc}"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_dag() -> NetworkDAG:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    Module.set_margin(2)
    scheme = SimpleNamespace(params=_DummyParams())

    model = UNet22(
        dataset="kvasir_polyp_256",
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
        activation=ACTIVATION,
        silu_degree=SILU_DEGREE,
    )
    model.eval()
    traced = OrionTracer().trace_model(model)
    sample = torch.randn((1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    StatsTracker(traced).propagate(sample)

    for module in traced.modules():
        if isinstance(module, Module):
            module.scheme = scheme
        if hasattr(module, "fit"):
            module.fit()
    for module in traced.modules():
        if hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if hasattr(module, "update_params"):
            module.update_params()

    dag = NetworkDAG(traced)
    dag.build_dag()
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None:
            module.name = str(node)
            module.scheme = scheme
    return dag


def _policy_bootstrap_count(policy: str) -> tuple[int, dict[str, Any]]:
    dag = _build_dag()
    registry = U22CompileRegistry.for_dag(
        dag,
        allowed_nodes=None,
        enable_conv_kernels=True,
        layout_policy=str(policy),
    )
    audit = registry.attach_to_dag(dag)
    dag.find_residuals()
    input_level, bootstraps, bootstrapper_slots = BootstrapSolver(
        SimpleNamespace(),
        dag,
        l_eff=len(lp.U22_E2E_LOGQ) - 1,
    ).solve()
    summary = dict(audit.get("graph_audit", {}).get("layout_policy_summary", {}) or {})
    return int(bootstraps), {
        "policy": str(policy),
        "source": "policy_aware_bootstrap_solver_after_registry_attach",
        "input_level": int(input_level),
        "bootstrapper_slots": [int(value) for value in bootstrapper_slots],
        "relayout_depth_estimate": int(summary.get("relayout_depth_estimate", 0) or 0),
        "fused_relayout": int(summary.get("producer_fused_materialization_count", 0) or 0)
        + int(summary.get("consumer_fused_relayout_count", 0) or 0),
    }


def _planner_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dag = _build_dag()
    edges = lp.build_edge_infos(dag, slots=int(lp.DEFAULT_SLOTS))
    rows: list[dict[str, Any]] = []
    bootstrap_details: list[dict[str, Any]] = []
    for policy in POLICIES:
        plan = lp.plan_policy(
            dag,
            edges,
            policy,
            slots=int(lp.DEFAULT_SLOTS),
            estimator=lp.LAYOUT_ESTIMATOR_TEMPLATE,
        )
        boot_count, boot_detail = _policy_bootstrap_count(str(policy))
        bootstrap_details.append(dict(boot_detail))
        rows.append(
            {
                "policy": str(policy),
                "rotation": int(plan.reported_rotation_estimate),
                "boot": int(boot_count),
                "explicit_relayout": int(plan.relayout_depth_estimate),
                "explicit_relayout_count": int(plan.relayouts),
                "fused_relayout": int(
                    plan.producer_fused_materialization_count + plan.consumer_fused_relayout_count
                ),
                "ct": int(plan.total_ciphertext_tiles),
            }
        )
    metadata = {
        "image_size": int(IMAGE_SIZE),
        "input_channels": int(INPUT_CHANNELS),
        "output_channels": int(OUTPUT_CHANNELS),
        "base_channels": int(BASE_CHANNELS),
        "activation": str(ACTIVATION),
        "silu_degree": int(SILU_DEGREE),
        "slots": int(lp.DEFAULT_SLOTS),
        "layout_estimator": str(lp.LAYOUT_ESTIMATOR_TEMPLATE),
        "node_count": int(len(dag.nodes)),
        "edge_count": int(len(dag.edges)),
        "bootstrap_source": "policy_aware_bootstrap_solver_after_registry_attach",
        "bootstrap_details": bootstrap_details,
    }
    return rows, metadata


def _forward_timing(payload: dict[str, Any], key: str) -> float | None:
    timing = (
        payload.get("measured_forward_mean_timing_s")
        or payload.get("forward_mean_timing_s")
        or payload.get("timing_s")
        or {}
    )
    if not isinstance(timing, dict):
        return None
    value = timing.get(key)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _result_paths(run_root: Path, policy: str) -> tuple[Path, Path]:
    case_dir = Path(run_root) / str(policy)
    return case_dir / "provider_e2e.json", case_dir / "provider_e2e.log"


def _runtime_row(run_root: Path, policy: str) -> dict[str, Any]:
    result_path, log_path = _result_paths(run_root, policy)
    payload = _read_json(result_path) or {}
    runner = payload.get("runner", {}) if isinstance(payload, dict) else {}
    return {
        "time_s": _forward_timing(payload, "he_forward"),
        "status": str(payload.get("status", "pending")) if isinstance(payload, dict) else "pending",
        "provider_mode": str(payload.get("provider_mode", "") if isinstance(payload, dict) else ""),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "elapsed_s": runner.get("elapsed_s") if isinstance(runner, dict) else None,
        "error": str(payload.get("error", "") if isinstance(payload, dict) else ""),
    }


def _combined_rows(planner_rows: list[dict[str, Any]], run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in planner_rows:
        policy = str(row["policy"])
        runtime = _runtime_row(run_root, policy)
        rows.append(
            {
                "policy": policy,
                "strategy": STRATEGY_LABELS.get(policy, policy),
                "time_s": "" if runtime["time_s"] is None else float(runtime["time_s"]),
                "rotation": int(row["rotation"]),
                "boot": int(row["boot"]),
                "ct": int(row["ct"]),
                "explicit_relayout_depth": int(row["explicit_relayout"]),
                "fused_relayout": int(row["fused_relayout"]),
                "provider_status": runtime["status"],
                "provider_mode": runtime["provider_mode"],
                "result_path": runtime["result_path"],
            }
        )
    return rows


def _format_int(value: Any) -> str:
    return f"{int(value):,}"


def _format_time(value: Any) -> str:
    if value == "" or value is None:
        return "-"
    return f"{float(value):.1f}"


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Strategy | Time (s) | Rot. | Boot. | CT | Explicit RL Depth | Fused RL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['strategy']} | "
            f"{_format_time(row['time_s'])} | "
            f"{_format_int(row['rotation'])} | "
            f"{_format_int(row['boot'])} | "
            f"{_format_int(row['ct'])} | "
            f"{_format_int(row['explicit_relayout_depth'])} | "
            f"{_format_int(row['fused_relayout'])} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(
    *,
    out_csv: Path,
    out_json: Path,
    out_md: Path | None,
    run_root: Path,
    planner_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _combined_rows(planner_rows, run_root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "strategy",
                "time_s",
                "rotation",
                "boot",
                "ct",
                "explicit_relayout_depth",
                "fused_relayout",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "policy": row["policy"],
                    "strategy": row["strategy"],
                    "time_s": row["time_s"],
                    "rotation": row["rotation"],
                    "boot": row["boot"],
                    "ct": row["ct"],
                    "explicit_relayout_depth": row["explicit_relayout_depth"],
                    "fused_relayout": row["fused_relayout"],
                }
            )
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_markdown(rows), encoding="utf-8")
    _write_json(
        out_json,
        {
            "status": "ok",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_root": str(run_root),
            "csv": str(out_csv),
            "markdown": "" if out_md is None else str(out_md),
            "metadata": metadata,
            "planner_rows": planner_rows,
            "rows": rows,
        },
    )
    return rows


def _status_ok(path: Path) -> bool:
    payload = _read_json(path)
    return isinstance(payload, dict) and payload.get("status") == "ok"


def _run_provider_policy(args: argparse.Namespace, policy: str) -> int:
    result_path, log_path = _result_paths(Path(args.run_root), policy)
    if _status_ok(result_path) and not bool(args.force):
        print(f"[{datetime.now().isoformat(timespec='seconds')}] skip {policy}: {result_path}", flush=True)
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.run_root) / "tmp" / str(policy)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_dir)
    env.setdefault("XDG_CACHE_HOME", str(Path(args.run_root) / "xdg-cache"))
    env["ORION_PACK_CONV_WORKERS"] = str(int(args.pack_workers))
    env["ORION_DIRECT_PACK_WORKERS"] = str(int(args.pack_workers))
    command = [
        "/usr/bin/time",
        "-v",
        sys.executable,
        str(RUNNER),
        "--run-one",
        "--size",
        "224x224",
        "--policy",
        str(policy),
        "--run-root",
        str(args.run_root),
        "--out",
        str(result_path),
        "--seed",
        str(args.seed),
        "--forward-runs",
        str(args.forward_runs),
        "--warmup-runs",
        str(args.warmup_runs),
    ]
    if bool(args.operator_breakdown):
        command.append("--operator-breakdown")
    else:
        command.append("--no-operator-breakdown")

    _write_json(
        result_path,
        {
            "status": "running",
            "network": "u22_224_224_base32_full",
            "size": "224x224",
            "policy": str(policy),
            "activation": {"kind": ACTIVATION, "silu_degree": SILU_DEGREE},
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "pack_workers": int(args.pack_workers),
            "command": " ".join(shlex.quote(part) for part in command),
        },
    )
    print(f"[{datetime.now().isoformat(timespec='seconds')}] start {policy}", flush=True)
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = int(proc.wait())
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] done {policy} rc={return_code} elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return return_code


def _default_run_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUN_ROOT_BASE / f"u22_224_base32_silu7_policy_streaming_provider_{timestamp}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 224x224 U22 base32 SiLU7 policy table with provider times."
    )
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--run-provider", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--pack-workers", type=int, default=4)
    parser.add_argument("--operator-breakdown", dest="operator_breakdown", action="store_true")
    parser.add_argument("--no-operator-breakdown", dest="operator_breakdown", action="store_false")
    parser.set_defaults(operator_breakdown=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_root = Path(args.run_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "tmp").mkdir(parents=True, exist_ok=True)

    planner_rows, metadata = _planner_rows()
    selected = set(str(policy) for policy in args.policies)
    planner_rows = [row for row in planner_rows if str(row["policy"]) in selected]
    _write_outputs(
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
        out_md=Path(args.out_md),
        run_root=Path(args.run_root),
        planner_rows=planner_rows,
        metadata=metadata,
    )

    if bool(args.run_provider):
        manifest = {
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "run_root": str(args.run_root),
            "out_csv": str(args.out_csv),
            "out_json": str(args.out_json),
            "policies": [str(policy) for policy in args.policies],
            "pack_workers": int(args.pack_workers),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(shlex.quote(part) for part in sys.argv),
        }
        _write_json(Path(args.run_root) / "manifest.json", manifest)
        for policy in args.policies:
            return_code = _run_provider_policy(args, str(policy))
            _write_outputs(
                out_csv=Path(args.out_csv),
                out_json=Path(args.out_json),
                out_md=Path(args.out_md),
                run_root=Path(args.run_root),
                planner_rows=planner_rows,
                metadata=metadata,
            )
            if int(return_code) != 0 and not bool(args.keep_going):
                return int(return_code)

    rows = _write_outputs(
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
        out_md=Path(args.out_md),
        run_root=Path(args.run_root),
        planner_rows=planner_rows,
        metadata=metadata,
    )
    print(Path(args.out_csv))
    print(Path(args.out_md))
    print(_render_markdown(rows), end="")
    for row in rows:
        print(
            f"{row['policy']}: time={row['time_s']} rotation={row['rotation']} "
            f"boot={row['boot']} ct={row['ct']} "
            f"explicit_relayout_depth={row['explicit_relayout_depth']} fused_relayout={row['fused_relayout']} "
            f"status={row['provider_status']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
