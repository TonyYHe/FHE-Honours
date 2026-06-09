#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.orion import scheme
from orion.experimental import layout_policy_ablation as lp
from orion.experimental.u22_phase1 import collect_layout_policy_provider_pressure
from orion.models.unet import UNet22PlusOutput
from orion.nn.module import Module


DEFAULT_POLICIES = ("dp_no_share_fold", "fixed_max_no_share", "always_no_share_producer_fused")
PROVIDER_SUFFIX = {
    "dp_no_share_fold": "dp_no_share_fold",
    "dp_noshare_fold": "dp_no_share_fold",
    "noshare_fold": "dp_no_share_fold",
    "fixed_max_no_share": "fixedmax_no_share",
    "fixedmax_no_share": "fixedmax_no_share",
    "fixed_noshare": "fixedmax_no_share",
    "fixedmax_noshare": "fixedmax_no_share",
    "fixed_max_no_share_fused": "fixedmax_no_share",
    "fixedmax_no_share_fused": "fixedmax_no_share",
    "fixed_noshare_fused": "fixedmax_no_share",
    "fixedmax_noshare_fused": "fixedmax_no_share",
    "fixed_max_no_share_unfused": "fixedmax_no_share_unfused",
    "fixedmax_no_share_unfused": "fixedmax_no_share_unfused",
    "fixed_noshare_unfused": "fixedmax_no_share_unfused",
    "fixedmax_noshare_unfused": "fixedmax_no_share_unfused",
    "always_no_share": "always_no_share_producer",
    "always_noshare": "always_no_share_producer",
    "always_relayout_no_share": "always_no_share_producer",
    "always_relayout_noshare": "always_no_share_producer",
    "always_no_share_producer": "always_no_share_producer",
    "always_noshare_producer": "always_no_share_producer",
    "always_relayout_no_share_producer": "always_no_share_producer",
    "always_relayout_noshare_producer": "always_no_share_producer",
    "always_no_share_producer_fused": "always_no_share_producer",
    "always_noshare_producer_fused": "always_no_share_producer",
    "always_relayout_no_share_producer_fused": "always_no_share_producer",
    "always_relayout_noshare_producer_fused": "always_no_share_producer",
    "always_no_share_fused": "always_no_share_fused",
    "always_noshare_fused": "always_no_share_fused",
    "always_relayout_no_share_fused": "always_no_share_fused",
    "always_relayout_noshare_fused": "always_no_share_fused",
    "always_no_share_unfused": "always_no_share_unfused",
    "always_noshare_unfused": "always_no_share_unfused",
    "always_relayout_no_share_unfused": "always_no_share_unfused",
    "always_relayout_noshare_unfused": "always_no_share_unfused",
}
SUMMARY_KEYS = (
    "relayouts",
    "relayout_rotation_estimate",
    "relayout_mask_mult_estimate",
    "relayout_depth_estimate",
    "producer_fused_materialization_count",
    "producer_fused_rotation_estimate",
    "consumer_fused_relayout_count",
    "consumer_fused_rotation_estimate",
    "lt_bsgs_rotation_estimate",
    "planner_rotation_cost_estimate",
    "reported_rotation_estimate",
    "ct_pt_mult_estimate",
    "total_ciphertext_tiles",
    "stored_slots",
    "objective",
)
PRESSURE_KEYS = (
    "provider_region_count",
    "native_halo_provider_region_count",
    "relayout_lt_region_count",
    "relayout_edge_count",
    "native_physical_relayout_edge_count",
    "compact_align_shared_edge_count",
    "output_relayout_edge_count",
    "relayout_kernel_count",
    "relayout_rotation_count",
    "relayout_mask_mult_count",
    "relayout_sparse_lt_count",
    "group_union_rotation_count",
    "transform_sum_rotation_count",
)


def _json_sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(_json_sanitize(key)): _json_sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_sanitize(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_sanitize(item())
        except Exception:
            pass
    return {"__type__": type(value).__name__, "repr": repr(value)}


def _noop_compile(self: Any) -> None:
    return None


def _patch_module_compile_noop(net: Any) -> dict[str, int]:
    patched = Counter()
    for module in net.modules():
        if isinstance(module, Module) and callable(getattr(module, "compile", None)):
            module.compile = MethodType(_noop_compile, module)
            patched[type(module).__name__] += 1
    return dict(sorted(patched.items()))


def _provider_mode(policy: str) -> str:
    key = str(policy).strip().lower().replace("-", "_")
    if key not in PROVIDER_SUFFIX:
        raise ValueError(f"unsupported policy {policy!r}")
    return f"u22_256_base32_layout_{PROVIDER_SUFFIX[key]}"


def _env_snapshot() -> dict[str, str]:
    keys = [
        "ORION_BOOTSTRAP_LAYOUT_REFINEMENT_MAX_ROUNDS",
        "ORION_BOOTSTRAP_LAYOUT_REFINEMENT",
        "ORION_BOOTSTRAP_LAYOUT_REFINEMENT_AUTO_TARGET",
        "ORION_BOOTSTRAP_LAYOUT_REFINEMENT_TARGET_BOOTSTRAPS",
        "ORION_LAYOUT_POLICY_RELAYOUT_KERNEL",
        "ORION_LAYOUT_POLICY_PROVIDER_NATIVE_HALO",
        "ORION_UNIFIED_LT_INDIVIDUAL_EVAL",
        "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS",
        "ORION_LATTIGO_UNIFIED_NO_BSGS",
    ]
    return {key: str(os.environ.get(key, "")) for key in keys}


def _iter_executor_objects(root: Any) -> Iterable[Any]:
    stack = [root]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("base_executor", "delegate", "executor"):
            child = getattr(current, attr, None)
            if child is not None:
                stack.append(child)


def _compile_plans_from_registry(registry: Any) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for group in tuple(getattr(registry, "groups", ()) or ()):
        for executor in _iter_executor_objects(getattr(group, "executor", None)):
            plan = getattr(executor, "compile_plan", None)
            if isinstance(plan, dict) and plan.get("edge_layouts"):
                plans.append(dict(plan))
    plans.sort(
        key=lambda plan: (
            -len(plan.get("edge_layouts", []) or []),
            str(plan.get("policy", "")),
        )
    )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for plan in plans:
        key = (
            str(plan.get("policy", "")),
            len(plan.get("edge_layouts", []) or []),
            len(plan.get("node_layouts", []) or []),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(plan)
    return deduped


def _primary_compile_plan(registry: Any) -> dict[str, Any]:
    plans = _compile_plans_from_registry(registry)
    if not plans:
        return {}
    return plans[0]


def _layout_tuple(row: dict[str, Any]) -> tuple[int, int, int, int, int, int, str]:
    layout = dict(row.get("selected_layout", {}) or {})
    return (
        int(layout.get("top_beta", 0) or 0),
        int(layout.get("bottom_beta", 0) or 0),
        int(layout.get("stride", 0) or 0),
        int(layout.get("gap", 0) or 0),
        int(layout.get("physical_top_beta", 0) or 0),
        int(layout.get("physical_bottom_beta", 0) or 0),
        str(row.get("physical_layout", "")),
    )


def _edge_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge": str(row.get("edge", "")),
        "source": str(row.get("source", "")),
        "target": str(row.get("target", "")),
        "op_kind": str(row.get("op_kind", "")),
        "layout_mode": str(row.get("layout_mode", "")),
        "physical_layout": str(row.get("physical_layout", "")),
        "source_physical_layout": str(row.get("source_physical_layout", "")),
        "relayout": bool(row.get("relayout", False)),
        "relayout_reason": str(row.get("relayout_reason", "")),
        "consumer_fused_relayout": bool(row.get("consumer_fused_relayout", False)),
        "selected_layout": dict(row.get("selected_layout", {}) or {}),
        "lt_bsgs_rotation_estimate": int(row.get("lt_bsgs_rotation_estimate", 0) or 0),
        "planner_rotation_cost_estimate": int(row.get("planner_rotation_cost_estimate", 0) or 0),
        "relayout_depth_estimate": int(row.get("relayout_depth_estimate", 0) or 0),
        "relayout_mask_mult_estimate": int(row.get("relayout_mask_mult_estimate", 0) or 0),
    }


def _node_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "node": str(row.get("node", "")),
        "physical_layout": str(row.get("physical_layout", "")),
        "producer_materialized_halo": bool(row.get("producer_materialized_halo", False)),
        "producer_materialized_halo_reason": str(row.get("producer_materialized_halo_reason", "")),
        "producer_fused_rotation_estimate": int(row.get("producer_fused_rotation_estimate", 0) or 0),
        "selected_layout": dict(row.get("selected_layout", {}) or {}),
    }


def _summarize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    summary = dict(plan.get("summary", {}) or {})
    edges = [dict(row) for row in plan.get("edge_layouts", []) or []]
    nodes = [dict(row) for row in plan.get("node_layouts", []) or []]
    relayout_edges = [row for row in edges if bool(row.get("relayout", False))]
    native_edges = [row for row in edges if str(row.get("physical_layout", "")) == "native_source_stripe"]
    compact_shared_edges = [
        row
        for row in edges
        if str(row.get("layout_mode", "")) in {"compact_align_shared", "compact_halo_shared", "compact_tconv_shared"}
    ]
    producer_nodes = [row for row in nodes if bool(row.get("producer_materialized_halo", False))]
    consumer_fused_edges = [row for row in edges if bool(row.get("consumer_fused_relayout", False))]
    by_mode = Counter(str(row.get("layout_mode", "")) for row in edges)
    by_physical = Counter(str(row.get("physical_layout", "")) for row in edges)
    layout_signature = Counter(_layout_tuple(row) for row in edges)
    bootstrap_refinement = dict(plan.get("bootstrap_aware_layout_refinement", {}) or {})
    return {
        "summary": {key: summary.get(key, "") for key in SUMMARY_KEYS},
        "edge_layout_count": int(len(edges)),
        "node_layout_count": int(len(nodes)),
        "relayout_edge_count": int(len(relayout_edges)),
        "native_source_stripe_edge_count": int(len(native_edges)),
        "compact_shared_edge_count": int(len(compact_shared_edges)),
        "consumer_fused_edge_count": int(len(consumer_fused_edges)),
        "producer_materialized_node_count": int(len(producer_nodes)),
        "layout_mode_counts": dict(sorted(by_mode.items())),
        "physical_layout_counts": dict(sorted(by_physical.items())),
        "layout_signature_counts": [
            {
                "top_beta": key[0],
                "bottom_beta": key[1],
                "stride": key[2],
                "gap": key[3],
                "physical_top_beta": key[4],
                "physical_bottom_beta": key[5],
                "physical_layout": key[6],
                "count": int(count),
            }
            for key, count in sorted(layout_signature.items(), key=lambda item: (item[0], item[1]))
        ],
        "relayout_edges": [_edge_brief(row) for row in relayout_edges],
        "native_source_stripe_edges": [_edge_brief(row) for row in native_edges],
        "compact_shared_edges": [_edge_brief(row) for row in compact_shared_edges],
        "consumer_fused_edges": [_edge_brief(row) for row in consumer_fused_edges],
        "producer_materialized_nodes": [_node_brief(row) for row in producer_nodes],
        "bootstrap_aware_layout_refinement": bootstrap_refinement,
    }


def _build_model(args: argparse.Namespace) -> tuple[UNet22PlusOutput, torch.Tensor]:
    torch.manual_seed(int(args.seed))
    torch.set_grad_enabled(False)
    net = UNet22PlusOutput(
        dataset="kvasir_polyp_256",
        in_channels=3,
        out_channels=1,
        base_channels=int(args.base_channels),
        activation="silu",
        silu_degree=int(args.silu_degree),
    )
    net.eval()
    x = torch.randn((1, 3, int(args.image_size), int(args.image_size)), dtype=torch.float32)
    return net, x


def _run_policy(args: argparse.Namespace, policy: str) -> dict[str, Any]:
    provider_mode = _provider_mode(policy)
    config = lp._runtime_config(backend=str(args.backend), provider_mode=provider_mode, logn=int(args.logn))
    started = time.perf_counter()
    row: dict[str, Any] = {
        "policy": str(policy),
        "provider_mode": str(provider_mode),
        "backend": str(args.backend),
        "status": "started",
    }
    try:
        scheme.init_scheme(config)
        Module.set_scheme(scheme)
        Module.set_margin(scheme.params.get_margin())
        net, x = _build_model(args)
        scheme.fit(net, x)
        compile_noop_patch: dict[str, int] = {}
        if bool(args.skip_layer_compile):
            compile_noop_patch = _patch_module_compile_noop(net)
        input_level = scheme.compile(net)
        registry = getattr(scheme, "region_first_registry", None)
        plan = _primary_compile_plan(registry)
        pressure = collect_layout_policy_provider_pressure(
            registry,
            backend=getattr(scheme, "backend", None),
            slots=int(scheme.params.get_slots()),
        )
        pressure_summary = dict((pressure or {}).get("summary", {}) or {})
        plan_summary = _summarize_plan(plan) if plan else {}
        row.update(
            {
                "status": "ok",
                "elapsed_s": float(time.perf_counter() - started),
                "input_level": int(input_level),
                "skip_layer_compile": bool(args.skip_layer_compile),
                "compile_noop_patch": compile_noop_patch,
                "attach_audit": dict(getattr(scheme, "region_first_attach_audit", {}) or {}),
                "compile_plan": plan,
                "plan_summary": plan_summary,
                "provider_pressure": pressure,
                "pressure_summary": {key: pressure_summary.get(key, "") for key in PRESSURE_KEYS},
            }
        )
        row.update({f"summary_{key}": plan_summary.get("summary", {}).get(key, "") for key in SUMMARY_KEYS})
        row.update({f"pressure_{key}": pressure_summary.get(key, "") for key in PRESSURE_KEYS})
        return row
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "elapsed_s": float(time.perf_counter() - started),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return row
    finally:
        try:
            scheme.delete_scheme()
        except Exception:
            pass


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "policy",
        "status",
        "elapsed_s",
        "input_level",
        "summary_reported_rotation_estimate",
        "summary_planner_rotation_cost_estimate",
        "summary_producer_fused_rotation_estimate",
        "summary_relayout_depth_estimate",
        "summary_relayouts",
        "summary_relayout_mask_mult_estimate",
        "summary_producer_fused_materialization_count",
        "summary_consumer_fused_relayout_count",
        "pressure_provider_region_count",
        "pressure_native_halo_provider_region_count",
        "pressure_relayout_edge_count",
        "pressure_native_physical_relayout_edge_count",
        "pressure_compact_align_shared_edge_count",
        "pressure_relayout_kernel_count",
        "pressure_relayout_rotation_count",
        "pressure_group_union_rotation_count",
        "pressure_transform_sum_rotation_count",
        "provider_mode",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile-level U22 224 policy relayout probe.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / ".tmp" / "results" / "u22_224_current_policy_compile_probe.json")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--backend", choices=("python", "lattigo"), default="python")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--silu-degree", type=int, default=7)
    parser.add_argument("--logn", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--auto-target", choices=("0", "1"), default="1")
    parser.add_argument("--target-bootstraps", default="")
    parser.add_argument("--skip-layer-compile", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("ORION_LAYOUT_POLICY_RELAYOUT_KERNEL", "0")
    os.environ.setdefault("ORION_LAYOUT_POLICY_PROVIDER_NATIVE_HALO", "1")
    os.environ.setdefault("ORION_UNIFIED_LT_INDIVIDUAL_EVAL", "1")
    os.environ.setdefault("ORION_UNIFIED_LT_SHARED_ROTATION_KEYS", "0")
    os.environ.setdefault("ORION_LATTIGO_UNIFIED_NO_BSGS", "0")
    os.environ["ORION_BOOTSTRAP_LAYOUT_REFINEMENT_MAX_ROUNDS"] = str(int(args.max_rounds))
    os.environ["ORION_BOOTSTRAP_LAYOUT_REFINEMENT_AUTO_TARGET"] = str(args.auto_target)
    if str(args.target_bootstraps).strip():
        os.environ["ORION_BOOTSTRAP_LAYOUT_REFINEMENT_TARGET_BOOTSTRAPS"] = str(args.target_bootstraps).strip()

    rows: list[dict[str, Any]] = []
    for policy in args.policies:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] compile {policy}", flush=True)
        row = _run_policy(args, str(policy))
        rows.append(row)
        print(
            json.dumps(
                {
                    "policy": row.get("policy"),
                    "status": row.get("status"),
                    "elapsed_s": row.get("elapsed_s"),
                    "reported_rotation": row.get("summary_reported_rotation_estimate"),
                    "relayout_depth": row.get("summary_relayout_depth_estimate"),
                    "relayouts": row.get("summary_relayouts"),
                    "producer_fused": row.get("summary_producer_fused_materialization_count"),
                    "native_physical_relayout_edges": row.get("pressure_native_physical_relayout_edge_count"),
                    "error": row.get("error", ""),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    payload = {
        "status": "ok" if all(str(row.get("status")) == "ok" for row in rows) else "error",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case": {
            "network": "u22_plus_output",
            "model_variant": "UNet22PlusOutput body plus explicit 1x1 output; input/output layouts are runtime boundaries",
            "image_size": int(args.image_size),
            "base_channels": int(args.base_channels),
            "activation": "silu",
            "silu_degree": int(args.silu_degree),
            "backend": str(args.backend),
            "logn": int(args.logn),
            "env": _env_snapshot(),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.csv if args.csv is not None else args.out.with_suffix(".csv")
    _write_csv(Path(csv_path), rows)
    print(str(args.out), flush=True)
    print(str(csv_path), flush=True)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
