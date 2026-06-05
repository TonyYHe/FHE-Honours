#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import orion.nn as on
from orion.core.fuser import Fuser
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import layout_policy_ablation as layout_planner
from orion.nn.module import Module


DEFAULT_IMAGE_SIZE = 256
DEFAULT_BASE_DIM = 8
DEFAULT_SLOTS = 32768
DEFAULT_POLICY = "dp"
DEFAULT_ESTIMATOR = "template"
DEFAULT_SILU_DEGREE = 7

layout_planner.LAYOUT_ESTIMATOR_DEFAULT = layout_planner.LAYOUT_ESTIMATOR_TEMPLATE


class _DummyParams:
    def __init__(self, slots: int) -> None:
        self._slots = int(slots)

    def get_slots(self) -> int:
        return int(self._slots)

    def get_embedding_method(self) -> str:
        return "hybrid"

    def get_debug_status(self) -> bool:
        return False

    def get_io_mode(self) -> str:
        return "none"

    def get_compile_save_resume(self) -> bool:
        return False


def _make_act(degree: int):
    return on.SiLU(degree=int(degree))


def _apply_act(module: Any, value: Any) -> Any:
    return module(value) if module is not None else value


class ConvDAGStressNet(on.Module):
    """Pure convolutional multi-branch DAG with heterogeneous halo demand."""

    def __init__(self, *, in_channels: int = 3, out_channels: int = 1, base_dim: int = 8, silu_degree: int = 7):
        super().__init__()
        b = int(base_dim)
        self.base_dim = int(b)
        self.silu_degree = int(silu_degree)

        self.stem = on.Conv2d(int(in_channels), b, kernel_size=3, padding=1, bias=True)
        self.stem_act = _make_act(int(silu_degree))

        # Same source, deliberately conflicting halo requirements:
        # 1x1 -> beta 0, 3x3 -> beta 1, 5x5/pool3 -> beta 2, 7x7 -> beta 3.
        self.b1 = on.Conv2d(b, b, kernel_size=1, bias=True)
        self.b3 = on.Conv2d(b, b, kernel_size=3, padding=1, bias=True)
        self.b5 = on.Conv2d(b, b, kernel_size=5, padding=2, bias=True)
        self.b7 = on.Conv2d(b, b, kernel_size=7, padding=3, bias=True)
        self.bp = on.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.cat0 = on.Concat(dim=1)
        self.mix0 = on.Conv2d(5 * b, 2 * b, kernel_size=1, bias=True)
        self.mix0_act = _make_act(int(silu_degree))

        self.down = on.AvgPool2d(kernel_size=2, stride=2)
        self.l3 = on.Conv2d(2 * b, 2 * b, kernel_size=3, padding=1, bias=True)
        self.l5 = on.Conv2d(2 * b, 2 * b, kernel_size=5, padding=2, bias=True)
        self.l7 = on.Conv2d(2 * b, 2 * b, kernel_size=7, padding=3, bias=True)
        self.cat1 = on.Concat(dim=1)
        self.mix1 = on.Conv2d(6 * b, 2 * b, kernel_size=1, bias=True)
        self.mix1_act = _make_act(int(silu_degree))

        self.up = on.ConvTranspose2d(2 * b, 2 * b, kernel_size=2, stride=2, bias=True)
        self.add = on.Add()

        self.c1 = on.Conv2d(2 * b, b, kernel_size=1, bias=True)
        self.c3 = on.Conv2d(2 * b, b, kernel_size=3, padding=1, bias=True)
        self.c5 = on.Conv2d(2 * b, b, kernel_size=5, padding=2, bias=True)
        self.c7 = on.Conv2d(2 * b, b, kernel_size=7, padding=3, bias=True)
        self.cat2 = on.Concat(dim=1)
        self.mix2 = on.Conv2d(4 * b, b, kernel_size=1, bias=True)
        self.mix2_act = _make_act(int(silu_degree))
        self.out = on.Conv2d(b, int(out_channels), kernel_size=1, bias=True)

    def forward(self, x):
        stem = _apply_act(self.stem_act, self.stem(x))

        high = self.cat0(
            self.b1(stem),
            self.b3(stem),
            self.b5(stem),
            self.b7(stem),
            self.bp(stem),
        )
        high = _apply_act(self.mix0_act, self.mix0(high))

        low = self.down(high)
        low = self.cat1(self.l3(low), self.l5(low), self.l7(low))
        low = _apply_act(self.mix1_act, self.mix1(low))

        fused = self.add(high, self.up(low))
        out = self.cat2(self.c1(fused), self.c3(fused), self.c5(fused), self.c7(fused))
        out = _apply_act(self.mix2_act, self.mix2(out))
        return self.out(out)


def _ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def _ct_count(shape: Any, *, slots: int) -> int:
    return max(1, _ceil_div(int(torch.Size(tuple(int(v) for v in shape)).numel()), int(slots)))


def _shape_text(value: Any, *, drop_batch: bool = False) -> str:
    shape = tuple(int(item) for item in tuple(value))
    if bool(drop_batch) and len(shape) > 1 and int(shape[0]) == 1:
        shape = shape[1:]
    return "x".join(str(int(item)) for item in shape)


def _layout_beta_text(layout: dict[str, Any]) -> str:
    top = int(layout.get("top_beta", 0) or 0)
    bottom = int(layout.get("bottom_beta", 0) or 0)
    return str(top) if top == bottom else f"{top}/{bottom}"


def _fmt_int(value: int | str) -> str:
    if value == "":
        return "-"
    return f"{int(value):,}"


def _max_layout(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    layouts = [dict(row.get(key, {}) or {}) for row in rows if row.get(key)]
    if not layouts:
        return {}
    return {
        "top_beta": max(int(layout.get("top_beta", 0) or 0) for layout in layouts),
        "bottom_beta": max(int(layout.get("bottom_beta", 0) or 0) for layout in layouts),
        "stride": max(int(layout.get("stride", 1) or 1) for layout in layouts),
        "gap": max(int(layout.get("gap", 1) or 1) for layout in layouts),
        "tile_count": max(int(layout.get("tile_count", 1) or 1) for layout in layouts),
        "stored_slots": max(int(layout.get("stored_slots", 0) or 0) for layout in layouts),
    }


def _transition_text(in_rows: list[dict[str, Any]], node_row: dict[str, Any]) -> str:
    parts: list[str] = []
    edge_count = sum(int(bool(row.get("relayout", False))) for row in in_rows)
    consumer_count = sum(int(bool(row.get("consumer_fused_relayout", False))) for row in in_rows)
    if edge_count:
        parts.append(f"edge:{edge_count}")
    if consumer_count:
        parts.append(f"consumer-fused:{consumer_count}")
    if bool(node_row.get("producer_materialized_halo", False)):
        parts.append("producer-halo")
    if bool(node_row.get("output_relayout", False)):
        parts.append("output")
    return ";".join(parts) if parts else "-"


def _build_dag(args: argparse.Namespace) -> NetworkDAG:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    Module.set_margin(2)
    scheme = SimpleNamespace(params=_DummyParams(slots=int(args.slots)))

    model = ConvDAGStressNet(
        in_channels=int(args.input_channels),
        out_channels=int(args.output_channels),
        base_dim=int(args.base_dim),
        silu_degree=int(args.silu_degree),
    )
    model.eval()
    traced = OrionTracer().trace_model(model)
    StatsTracker(traced).propagate(
        torch.randn((1, int(args.input_channels), int(args.image_size), int(args.image_size)), dtype=torch.float32)
    )
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
    Fuser(dag).fuse_modules()
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None:
            module.name = str(node)
            module.scheme = scheme
    return dag


def _detail_rows(args: argparse.Namespace, plan: dict[str, Any], dag: NetworkDAG) -> list[dict[str, Any]]:
    edge_rows_by_target: dict[str, list[dict[str, Any]]] = {}
    for row in plan.get("edge_layouts", []):
        edge_rows_by_target.setdefault(str(row["target"]), []).append(dict(row))
    node_rows = {str(row["node"]): dict(row) for row in plan.get("node_layouts", [])}
    original_nodes = [
        str(node)
        for node in dag.topological_sort()
        if str(node) != "x" and dag.nodes[node].get("module") is not None
    ]

    rows: list[dict[str, Any]] = []
    for index, node in enumerate(original_nodes, start=1):
        module = dag.nodes[node].get("module")
        if module is None:
            continue
        in_rows = edge_rows_by_target.get(str(node), [])
        node_row = node_rows.get(str(node), {})
        required = _max_layout(in_rows, "required_layout")
        selected_input = _max_layout(in_rows, "selected_layout")
        selected_output = dict(node_row.get("selected_layout", {}) or {})
        output_shape = tuple(int(v) for v in getattr(module, "output_shape"))
        fhe_output_shape = tuple(int(v) for v in getattr(module, "fhe_output_shape"))
        if not selected_output:
            selected_output = {
                "top_beta": 0,
                "bottom_beta": 0,
                "gap": int(getattr(module, "output_gap", 1)),
                "tile_count": _ct_count(fhe_output_shape, slots=int(args.slots)),
            }
        planner_rot = 0 if type(module).__name__ == "SiLU" else sum(
            int(row.get("planner_rotation_cost_estimate", 0) or 0) for row in in_rows
        )
        relayout_rot = (
            sum(int(row.get("relayout_rotation_estimate", 0) or 0) for row in in_rows)
            + int(node_row.get("relayout_rotation_estimate", 0) or 0)
        )
        relayout_mask = (
            sum(int(row.get("relayout_mask_mult_estimate", 0) or 0) for row in in_rows)
            + int(node_row.get("relayout_mask_mult_estimate", 0) or 0)
        )
        lt_mult = 0 if type(module).__name__ == "SiLU" else sum(int(row.get("lt_ct_pt_mult_estimate", 0) or 0) for row in in_rows)
        activation_mult = sum(int(row.get("activation_ct_mult_estimate", 0) or 0) for row in in_rows)
        rows.append(
            {
                "node_index": int(index),
                "node": str(node),
                "op_type": type(module).__name__,
                "source_nodes": ";".join(str(row.get("source", "")) for row in in_rows),
                "output_shape": _shape_text(output_shape, drop_batch=True),
                "output_gap": int(getattr(module, "output_gap", 0) or 0),
                "selected_output_ct": int(selected_output.get("tile_count", _ct_count(fhe_output_shape, slots=int(args.slots))) or 0),
                "required_beta": _layout_beta_text(required),
                "selected_input_beta": _layout_beta_text(selected_input),
                "selected_output_beta": _layout_beta_text(selected_output),
                "layout_transition": _transition_text(in_rows, node_row),
                "layout_mode": ";".join(sorted({str(row.get("layout_mode", "") or "") for row in in_rows if row.get("layout_mode")})),
                "physical_layout": str(node_row.get("physical_layout", "") or "packed_compact"),
                "edge_relayout_count": sum(int(bool(row.get("relayout", False))) for row in in_rows),
                "consumer_fused_count": sum(int(bool(row.get("consumer_fused_relayout", False))) for row in in_rows),
                "producer_materialized_halo": bool(node_row.get("producer_materialized_halo", False)),
                "rotation": int(planner_rot + relayout_rot),
                "ct_pt_mult": int(lt_mult + relayout_mask + activation_mult),
            }
        )
    return rows


def render_markdown(args: argparse.Namespace, summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    beta_counts = summary["required_beta_counts"]
    lines = [
        "# ConvDAGStressNet DP layout metadata",
        "",
        f"Pure convolutional multi-branch DAG, image `{int(args.image_size)}x{int(args.image_size)}`, base `{int(args.base_dim)}`, SiLU{int(args.silu_degree)}, policy `{args.policy}`.",
        "Planner-only metadata; no HE transform generation or FHE forward.",
        "",
        f"Summary: edges `{summary['edge_count']}`, nodes `{summary['node_count']}`, CT `{_fmt_int(summary['ct'])}`, rotations `{_fmt_int(summary['rotations'])}`, CT-PT mult `{_fmt_int(summary['ct_pt_mult'])}`, re-layouts `{_fmt_int(summary['relayouts'])}`.",
        f"Required beta distribution: {', '.join(f'beta {key}: {value}' for key, value in beta_counts.items())}.",
        "",
        "| # | node | op | sources | out | gap | CT | req beta | in beta | out beta | transition | mode | physical | R | M |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["node_index"]),
                    str(row["node"]),
                    str(row["op_type"]),
                    str(row["source_nodes"]) or "-",
                    str(row["output_shape"]),
                    str(row["output_gap"]),
                    _fmt_int(row["selected_output_ct"]),
                    str(row["required_beta"]),
                    str(row["selected_input_beta"]),
                    str(row["selected_output_beta"]),
                    str(row["layout_transition"]),
                    str(row["layout_mode"]) or "-",
                    str(row["physical_layout"]) or "-",
                    _fmt_int(row["rotation"]),
                    _fmt_int(row["ct_pt_mult"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _default_stem(args: argparse.Namespace) -> str:
    return f"convdag_stress_base{int(args.base_dim)}_{int(args.image_size)}_silu{int(args.silu_degree)}_{args.policy}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ConvDAGStressNet DP layout metadata.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--output-channels", type=int, default=1)
    parser.add_argument("--base-dim", type=int, default=DEFAULT_BASE_DIM)
    parser.add_argument("--silu-degree", type=int, default=DEFAULT_SILU_DEGREE)
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--estimator", default=DEFAULT_ESTIMATOR)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dag = _build_dag(args)
    plan = layout_planner.build_layout_policy_compile_plan(
        dag,
        policy=str(args.policy),
        slots=int(args.slots),
        estimator=str(args.estimator),
    )
    rows = _detail_rows(args, plan, dag)
    required_beta_counts = Counter(
        str(int(row["required_layout"]["top_beta"]))
        if int(row["required_layout"]["top_beta"]) == int(row["required_layout"]["bottom_beta"])
        else f"{row['required_layout']['top_beta']}/{row['required_layout']['bottom_beta']}"
        for row in plan.get("edge_layouts", [])
    )
    summary = {
        "policy": str(args.policy),
        "image_size": int(args.image_size),
        "base_dim": int(args.base_dim),
        "node_count": int(len(rows)),
        "edge_count": int(plan.get("edge_layout_count", 0) or 0),
        "ct": int(plan["summary"].get("total_ciphertext_tiles", 0) or 0),
        "rotations": int(plan["summary"].get("reported_rotation_estimate", 0) or 0),
        "ct_pt_mult": int(plan["summary"].get("ct_pt_mult_estimate", 0) or 0),
        "relayouts": int(plan["summary"].get("relayouts", 0) or 0),
        "relayout_depth": int(plan["summary"].get("relayout_depth_estimate", 0) or 0),
        "producer_materialized_halo_count": int(plan["summary"].get("producer_fused_materialization_count", 0) or 0),
        "consumer_fused_relayout_count": int(plan["summary"].get("consumer_fused_relayout_count", 0) or 0),
        "required_beta_counts": dict(sorted(required_beta_counts.items(), key=lambda item: int(str(item[0]).split("/")[0]))),
    }

    stem = _default_stem(args)
    out_csv = Path(args.out_csv) if args.out_csv is not None else REPO_ROOT / ".tmp" / "results" / f"{stem}_detail.csv"
    out_md = Path(args.out_md) if args.out_md is not None else REPO_ROOT / ".tmp" / "results" / f"{stem}_layout.md"
    out_json = Path(args.out_json) if args.out_json is not None else REPO_ROOT / ".tmp" / "results" / f"{stem}_summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(args, summary, rows), encoding="utf-8")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_csv)
    print(out_md)
    print(out_json)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
