#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental.layout_policy_ablation import (
    build_layout_policy_compile_plan,
    build_u22_dag,
    network_spec,
)
from orion.models.resnet import BasicBlock, ResNet


def _prepare_dag(model: Any, sample: torch.Tensor) -> NetworkDAG:
    traced = OrionTracer().trace_model(model)
    StatsTracker(traced).propagate(sample)
    dag = NetworkDAG(traced)
    dag.build_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()
    return dag


def _resnet_dag(name: str) -> NetworkDAG:
    if name == "r18_tiny_mock":
        layers = [2, 2, 2, 2]
    elif name == "r34_tiny_mock":
        layers = [3, 4, 6, 3]
    else:
        raise ValueError(f"unsupported ResNet diagnostic case: {name}")
    model = ResNet(
        "cifar10",
        BasicBlock,
        layers,
        [4, 8, 16, 32],
        {"kernel_size": 3, "stride": 1, "padding": 1},
        10,
    )
    return _prepare_dag(model, torch.randn((1, 3, 32, 32), dtype=torch.float32))


def _dag_for_case(case: str) -> NetworkDAG:
    if case.startswith("u22_"):
        return build_u22_dag(network_spec(case))
    if case in {"r18_tiny_real", "r34_imgnet_real"}:
        from tools import benchmark_node_specific_lattigo_provider_vs_dense as bench

        network = "r18_tiny" if case == "r18_tiny_real" else "r34_imgnet"
        bench._init_scheme(str(network), backend="lattigo")
        dag, _audit = bench._prepare_dag(str(network), provider=False)
        return dag
    return _resnet_dag(case)


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0) or 0) for row in rows))


def _summarize_case(case: str, *, slots: int, out_dir: Path, estimator: str) -> dict[str, Any]:
    dag = _dag_for_case(case)
    plan = build_layout_policy_compile_plan(dag, policy="dp", slots=int(slots), estimator=str(estimator))
    edges = [dict(row) for row in plan["edge_layouts"]]
    op_edges = [row for row in edges if str(row.get("op_kind")) not in {"add", "input"}]
    summary = dict(plan["summary"])
    row = {
        "case": case,
        "slots": int(slots),
        "layout_estimator": str(plan.get("layout_estimator", estimator)),
        "status": str(plan.get("status", "")),
        "edge_count": len(edges),
        "op_edge_count": len(op_edges),
        "conv_edge_count": sum(1 for item in edges if str(item.get("op_kind")) == "conv2d"),
        "tconv_edge_count": sum(1 for item in edges if str(item.get("op_kind")) == "conv_transpose2d"),
        "lt_bsgs_rotation_estimate": int(summary.get("lt_bsgs_rotation_estimate", 0) or 0),
        "lt_input_cross_rotation_estimate": _sum(edges, "lt_input_cross_rotation_estimate"),
        "lt_local_submatrix_rotation_estimate": _sum(edges, "lt_local_submatrix_rotation_estimate"),
        "lt_output_materialize_rotation_estimate": _sum(edges, "lt_output_materialize_rotation_estimate"),
        "lt_unfused_rotation_estimate": _sum(edges, "lt_unfused_rotation_estimate"),
        "lt_same_input_fusion_savings_estimate": _sum(edges, "lt_same_input_fusion_savings_estimate"),
        "lt_local_program_count_estimate": _sum(edges, "lt_local_program_count_estimate"),
        "lt_recovery_program_count_estimate": _sum(edges, "lt_recovery_program_count_estimate"),
        "relayout_rotation_estimate": int(summary.get("relayout_rotation_estimate", 0) or 0),
        "relayout_depth_estimate": int(summary.get("relayout_depth_estimate", 0) or 0),
        "producer_fused_rotation_estimate": int(summary.get("producer_fused_rotation_estimate", 0) or 0),
        "consumer_fused_rotation_estimate": int(summary.get("consumer_fused_rotation_estimate", 0) or 0),
        "total_ciphertext_tiles": int(summary.get("total_ciphertext_tiles", 0) or 0),
        "stored_slots": int(summary.get("stored_slots", 0) or 0),
    }
    top_edges = sorted(
        edges,
        key=lambda item: (
            int(item.get("lt_input_cross_rotation_estimate", 0) or 0)
            + int(item.get("lt_local_submatrix_rotation_estimate", 0) or 0),
            int(item.get("lt_recovery_program_count_estimate", 0) or 0),
        ),
        reverse=True,
    )[:12]
    payload = {"summary": row, "top_edges": top_edges, "plan_summary": summary}
    (out_dir / f"{case}_rho_hat_debug.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump count-only layout-policy rho_hat diagnostics.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["r18_tiny_mock", "r34_tiny_mock", "u22_64_base32", "u22_128_base32", "u22_256_base32"],
        help=(
            "Planner cases. Use r18_tiny_real/r34_imgnet_real for the full node-specific "
            "ResNet DAGs, or r18_tiny_mock/r34_tiny_mock for fast smoke tests."
        ),
    )
    parser.add_argument("--slots", type=int, default=32768)
    parser.add_argument("--estimator", default="auto", choices=["auto", "count_only", "template"])
    parser.add_argument("--out-dir", type=Path, default=Path(".tmp/results/layout_policy_rho_hat_debug"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _summarize_case(str(case), slots=int(args.slots), out_dir=out_dir, estimator=str(args.estimator))
        for case in args.cases
    ]
    csv_path = out_dir / "layout_policy_rho_hat_debug_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = out_dir / "layout_policy_rho_hat_debug_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(rows[0]) + " |\n")
        handle.write("| " + " | ".join("---" for _ in rows[0]) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row[key]) for key in rows[0]) + " |\n")
    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
