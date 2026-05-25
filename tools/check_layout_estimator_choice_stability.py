#!/usr/bin/env python3
"""Check whether template refinement changes U22 DP layout choices.

This diagnostic answers a narrow planner question:

  If DP first ranks candidates with the fast count-only estimator and then
  re-ranks close candidates with the cached unweighted-template estimator, does
  the chosen layout/re-layout/fusion plan change?

It builds synthetic U22 base32 DAGs at selected image sizes, runs the planner
with ``count_only`` and ``auto``, and reports edge/node choice deltas. No CKKS
backend or descriptor oracle is invoked.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from orion.experimental.layout_policy_ablation import (
    NetworkSpec,
    build_layout_policy_compile_plan,
    build_u22_dag,
)


DEFAULT_OUT_DIR = Path(".tmp/results/layout_estimator_choice_stability")


def _spec_for_size(size: int) -> NetworkSpec:
    return NetworkSpec(
        network=f"u22_{int(size)}_base32_estimator_stability",
        dataset="kvasir_polyp_256",
        image_size=int(size),
        input_channels=3,
        base_channels=32,
        provider_mode="u22_256_base32",
    )


def _layout_key(row: dict[str, Any], key: str = "selected_layout") -> tuple[int, int, int, int]:
    layout = dict(row.get(key, {}) or {})
    return (
        int(layout.get("top_beta", layout.get("alpha", 0)) or 0),
        int(layout.get("bottom_beta", layout.get("beta", 0)) or 0),
        int(layout.get("stride", 0) or 0),
        int(layout.get("gap", 0) or 0),
    )


def _edge_choice_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _layout_key(row),
        str(row.get("layout_mode", "")),
        str(row.get("physical_layout", "")),
        bool(row.get("relayout", False)),
        bool(row.get("consumer_fused_relayout", False)),
    )


def _node_choice_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _layout_key(row),
        str(row.get("physical_layout", "")),
        bool(row.get("output_relayout", False)),
        bool(row.get("producer_materialized_halo", False)),
    )


def _diff_edges(count_plan: dict[str, Any], auto_plan: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    count_edges = {str(row["edge"]): dict(row) for row in count_plan["edge_layouts"]}
    auto_edges = {str(row["edge"]): dict(row) for row in auto_plan["edge_layouts"]}
    rows: list[dict[str, Any]] = []
    for edge in sorted(set(count_edges) & set(auto_edges)):
        left = count_edges[edge]
        right = auto_edges[edge]
        choice_changed = _edge_choice_key(left) != _edge_choice_key(right)
        rotation_delta = int(right.get("lt_bsgs_rotation_estimate", 0) or 0) - int(
            left.get("lt_bsgs_rotation_estimate", 0) or 0
        )
        if not bool(choice_changed) and int(rotation_delta) == 0:
            continue
        rows.append(
            {
                "edge": edge,
                "op_kind": str(left.get("op_kind", "")),
                "source": str(left.get("source", "")),
                "target": str(left.get("target", "")),
                "shape": str(left.get("shape", "")),
                "choice_changed": bool(choice_changed),
                "count_layout": str(_layout_key(left)),
                "auto_layout": str(_layout_key(right)),
                "count_mode": str(left.get("layout_mode", "")),
                "auto_mode": str(right.get("layout_mode", "")),
                "count_physical": str(left.get("physical_layout", "")),
                "auto_physical": str(right.get("physical_layout", "")),
                "count_relayout": bool(left.get("relayout", False)),
                "auto_relayout": bool(right.get("relayout", False)),
                "count_estimator": str(left.get("lt_estimator", "")),
                "auto_estimator": str(right.get("lt_estimator", "")),
                "count_lt_rot": int(left.get("lt_bsgs_rotation_estimate", 0) or 0),
                "auto_lt_rot": int(right.get("lt_bsgs_rotation_estimate", 0) or 0),
                "rotation_delta_auto_minus_count": int(rotation_delta),
                "abs_rotation_delta": abs(int(rotation_delta)),
            }
        )
    rows.sort(key=lambda row: (bool(row["choice_changed"]), int(row["abs_rotation_delta"])), reverse=True)
    return rows[: int(limit)]


def _diff_nodes(count_plan: dict[str, Any], auto_plan: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    count_nodes = {str(row["node"]): dict(row) for row in count_plan["node_layouts"]}
    auto_nodes = {str(row["node"]): dict(row) for row in auto_plan["node_layouts"]}
    rows: list[dict[str, Any]] = []
    for node in sorted(set(count_nodes) & set(auto_nodes)):
        left = count_nodes[node]
        right = auto_nodes[node]
        choice_changed = _node_choice_key(left) != _node_choice_key(right)
        if not bool(choice_changed):
            continue
        rows.append(
            {
                "node": node,
                "count_layout": str(_layout_key(left)),
                "auto_layout": str(_layout_key(right)),
                "count_physical": str(left.get("physical_layout", "")),
                "auto_physical": str(right.get("physical_layout", "")),
                "count_output_relayout": bool(left.get("output_relayout", False)),
                "auto_output_relayout": bool(right.get("output_relayout", False)),
                "count_producer_materialized_halo": bool(left.get("producer_materialized_halo", False)),
                "auto_producer_materialized_halo": bool(right.get("producer_materialized_halo", False)),
            }
        )
    return rows[: int(limit)]


def _summary_for_size(size: int, *, limit: int) -> dict[str, Any]:
    dag = build_u22_dag(_spec_for_size(int(size)))
    count_plan = build_layout_policy_compile_plan(dag, policy="dp", estimator="count_only")
    auto_plan = build_layout_policy_compile_plan(dag, policy="dp", estimator="auto")
    edge_deltas = _diff_edges(count_plan, auto_plan, limit=int(limit))
    node_deltas = _diff_nodes(count_plan, auto_plan, limit=int(limit))
    count_edges = [dict(row) for row in count_plan["edge_layouts"]]
    auto_edges = [dict(row) for row in auto_plan["edge_layouts"]]
    changed_edge_count = sum(
        1
        for left, right in zip(count_edges, auto_edges, strict=True)
        if str(left.get("edge")) == str(right.get("edge")) and _edge_choice_key(left) != _edge_choice_key(right)
    )
    template_edge_count = sum(1 for row in auto_edges if str(row.get("lt_estimator")) == "template")
    return {
        "size": int(size),
        "count_summary": dict(count_plan["summary"]),
        "auto_summary": dict(auto_plan["summary"]),
        "edge_count": int(len(count_edges)),
        "node_count": int(len(auto_plan["node_layouts"])),
        "changed_edge_count": int(changed_edge_count),
        "changed_node_count": int(len(node_deltas)),
        "auto_template_edge_count": int(template_edge_count),
        "count_lt_rot": int(count_plan["summary"].get("lt_bsgs_rotation_estimate", 0) or 0),
        "auto_lt_rot": int(auto_plan["summary"].get("lt_bsgs_rotation_estimate", 0) or 0),
        "count_relayout_depth": int(count_plan["summary"].get("relayout_depth_estimate", 0) or 0),
        "auto_relayout_depth": int(auto_plan["summary"].get("relayout_depth_estimate", 0) or 0),
        "count_relayout_rot": int(count_plan["summary"].get("relayout_rotation_estimate", 0) or 0),
        "auto_relayout_rot": int(auto_plan["summary"].get("relayout_rotation_estimate", 0) or 0),
        "edge_deltas": edge_deltas,
        "node_deltas": node_deltas,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Layout Estimator Choice Stability",
        "",
        "Planner comparison: `count_only` DP versus `auto` DP, where `auto` refines close candidates with cached unweighted diagonal templates.",
        "",
        "| image | changed edges | changed nodes | auto template edges | count LT rot | auto LT rot | count relayout depth | auto relayout depth |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {image} | {changed_edges} | {changed_nodes} | {template_edges} | {count_lt} | {auto_lt} | {count_depth} | {auto_depth} |".format(
                image=f"{int(row['size'])}x{int(row['size'])}",
                changed_edges=row["changed_edge_count"],
                changed_nodes=row["changed_node_count"],
                template_edges=row["auto_template_edge_count"],
                count_lt=row["count_lt_rot"],
                auto_lt=row["auto_lt_rot"],
                count_depth=row["count_relayout_depth"],
                auto_depth=row["auto_relayout_depth"],
            )
        )
    lines.append("")
    lines.append("## Changed / largest-delta kernel edges")
    for item in payload["cases"]:
        lines.append("")
        lines.append(f"### {int(item['size'])}x{int(item['size'])}")
        edge_rows = list(item["edge_deltas"])
        if not edge_rows:
            lines.append("")
            lines.append("No edge-level choice or rotation-estimate deltas.")
            continue
        lines.append("")
        lines.append("| edge | op | shape | choice changed | count layout | auto layout | count mode | auto mode | count rot | auto rot |")
        lines.append("| --- | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: |")
        for row in edge_rows:
            lines.append(
                "| {edge} | {op_kind} | {shape} | {choice_changed} | {count_layout} | {auto_layout} | {count_mode} | {auto_mode} | {count_lt_rot} | {auto_lt_rot} |".format(
                    **row
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[192, 224, 256])
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    cases = [_summary_for_size(int(size), limit=int(args.limit)) for size in args.sizes]
    summary_rows = [
        {
            key: value
            for key, value in item.items()
            if key
            in {
                "size",
                "edge_count",
                "node_count",
                "changed_edge_count",
                "changed_node_count",
                "auto_template_edge_count",
                "count_lt_rot",
                "auto_lt_rot",
                "count_relayout_depth",
                "auto_relayout_depth",
                "count_relayout_rot",
                "auto_relayout_rot",
            }
        }
        for item in cases
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary_rows": summary_rows, "cases": cases}
    (out_dir / "layout_estimator_choice_stability.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "layout_estimator_choice_stability_summary.csv", summary_rows)
    edge_rows: list[dict[str, Any]] = []
    for item in cases:
        for row in item["edge_deltas"]:
            edge_rows.append({"size": int(item["size"]), **dict(row)})
    _write_csv(out_dir / "layout_estimator_choice_stability_edge_deltas.csv", edge_rows)
    _write_markdown(out_dir / "layout_estimator_choice_stability.md", payload)
    print(out_dir / "layout_estimator_choice_stability.md")
    print(out_dir / "layout_estimator_choice_stability_summary.csv")


if __name__ == "__main__":
    main()
