from __future__ import annotations

import math
from typing import Any

import torch

PHYSICAL_COMPACT = "packed_compact"
PHYSICAL_LOGICAL_HALO = "logical_halo_compact"

_LAYOUT_PRESERVING_MODULES = {
    "Activation",
    "Chebyshev",
    "ELU",
    "GELU",
    "Mish",
    "Quad",
    "ReLU",
    "SELU",
    "SiLU",
    "Sigmoid",
    "Softplus",
}


def _ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def _layout_top_beta(layout: dict[str, Any]) -> int:
    return max(0, int(dict(layout).get("top_beta", dict(layout).get("alpha", 0)) or 0))


def _layout_bottom_beta(layout: dict[str, Any]) -> int:
    return max(0, int(dict(layout).get("bottom_beta", dict(layout).get("beta", 0)) or 0))


def _shape_tuple(value: Any) -> tuple[int, int, int, int] | None:
    try:
        shape = tuple(int(item) for item in value)
    except Exception:
        return None
    return shape if len(shape) == 4 else None


def _compact_layout_for_row(row: dict[str, Any]) -> dict[str, int]:
    compact = dict(row.get("compact_layout", {}) or {})
    if compact:
        return {str(key): int(value) for key, value in compact.items()}

    selected = dict(row.get("selected_layout", {}) or {})
    shape = _shape_tuple(row.get("shape", ()))
    gap = max(1, int(selected.get("gap", 1) or 1))
    stride = max(1, int(selected.get("stride", 1) or 1))
    if shape is None:
        stored = int(selected.get("core_slots", selected.get("stored_slots", 0)) or 0)
        slots = max(1, int(row.get("slots", 32768) or 32768))
        return {
            "top_beta": 0,
            "bottom_beta": 0,
            "stride": int(stride),
            "gap": int(gap),
            "core_slots": int(stored),
            "stored_slots": int(stored),
            "tile_count": max(1, _ceil_div(int(stored), int(slots))),
        }

    _n, channels, height, width = shape
    phase = max(1, int(gap) * int(gap))
    channel_groups = _ceil_div(int(channels), int(phase))
    core_slots = int(channel_groups * int(height) * int(gap) * int(width) * int(gap))
    slots = max(1, int(row.get("slots", 32768) or 32768))
    return {
        "top_beta": 0,
        "bottom_beta": 0,
        "stride": int(stride),
        "gap": int(gap),
        "core_slots": int(core_slots),
        "stored_slots": int(core_slots),
        "tile_count": max(1, _ceil_div(int(core_slots), int(slots))),
    }


def _layout_has_halo(layout: dict[str, Any]) -> bool:
    return bool(_layout_top_beta(layout) > 0 or _layout_bottom_beta(layout) > 0)


def _fhe_shape_for_layout(row: dict[str, Any], layout: dict[str, Any]) -> torch.Size | None:
    shape = _shape_tuple(row.get("shape", ()))
    if shape is None:
        return None
    n, channels, height, width = shape
    gap = max(1, int(dict(layout).get("gap", 1) or 1))
    top_beta = _layout_top_beta(layout)
    bottom_beta = _layout_bottom_beta(layout)
    on_channels = int(math.ceil(int(channels) / float(int(gap) * int(gap))))
    return torch.Size(
        (
            int(n),
            int(on_channels),
            int(height * gap + (top_beta + bottom_beta) * gap),
            int(width * gap),
        )
    )


def _copy_row_with_compact_layout(row: dict[str, Any], *, source_layout: dict[str, Any] | None = None) -> dict[str, Any]:
    compact = _compact_layout_for_row(row)
    updated = dict(row)
    updated["selected_layout"] = dict(compact)
    updated["target_layout"] = dict(compact)
    updated["source_layout"] = dict(source_layout if source_layout is not None else compact)
    updated["physical_layout"] = PHYSICAL_COMPACT
    updated["source_physical_layout"] = PHYSICAL_COMPACT
    updated["target_physical_layout"] = PHYSICAL_COMPACT
    updated["relayout"] = False
    updated["relayout_reason"] = ""
    updated["relayout_rotation_estimate"] = 0
    updated["relayout_mask_mult_estimate"] = 0
    updated["relayout_depth_estimate"] = 0
    updated["consumer_fused_relayout"] = False
    updated["consumer_fused_rotation_estimate"] = 0
    updated["producer_materialized_halo"] = False
    updated["producer_materialized_halo_reason"] = ""
    updated["producer_fused_depth_estimate"] = 0
    if str(updated.get("layout_mode", "")) not in {"input", ""}:
        updated["layout_mode"] = "bootstrap_compact_boundary"
    return updated


def _copy_node_with_compact_layout(row: dict[str, Any]) -> dict[str, Any]:
    compact = _compact_layout_for_row(row)
    updated = dict(row)
    updated["selected_layout"] = dict(compact)
    updated["compact_layout"] = dict(compact)
    updated["physical_layout"] = PHYSICAL_COMPACT
    updated["output_relayout"] = False
    updated["output_relayout_reason"] = ""
    updated["producer_materialized_halo"] = False
    updated["producer_materialized_halo_reason"] = ""
    updated["producer_fused_depth_estimate"] = 0
    shape = _fhe_shape_for_layout(row, compact)
    if shape is not None:
        updated["fhe_shape"] = [int(value) for value in shape]
    return updated


def _iter_layout_policy_executors(network_dag: Any) -> list[Any]:
    executors: list[Any] = []
    seen: set[int] = set()
    for node in network_dag.nodes:
        module = network_dag.nodes[node].get("module")
        candidates = [
            getattr(getattr(module, "region_runtime", None), "executor", None),
            getattr(module, "layout_policy_add_runtime", None),
            getattr(module, "layout_policy_concat_runtime", None),
        ]
        stack = [candidate for candidate in candidates if candidate is not None]
        while stack:
            current = stack.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            current_dict = getattr(current, "__dict__", {})
            if isinstance(current_dict.get("compile_plan"), dict):
                executors.append(current)
            for attr in ("base_executor", "delegate", "_delegate", "executor"):
                child = current_dict.get(attr)
                if child is not None:
                    stack.append(child)
    return executors


def _layout_preserving_module(network_dag: Any, node: str) -> bool:
    module = network_dag.nodes[node].get("module") if node in network_dag.nodes else None
    return type(module).__name__ in _LAYOUT_PRESERVING_MODULES


def rewrite_layout_policy_plan_for_bootstrap_compression(
    compile_plan: dict[str, Any],
    network_dag: Any,
    *,
    bootstrap_nodes: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    node_rows = {str(row.get("node", "")): dict(row) for row in compile_plan.get("node_layouts", [])}
    if not node_rows:
        return dict(compile_plan), {"enabled": False, "reason": "no_layout_policy_node_rows"}

    compressed_nodes: set[str] = set()
    audit_nodes: list[dict[str, Any]] = []
    queue: list[str] = []
    force_compact = str(compile_plan.get("policy", "")) == "dp_no_share_fold"
    for node in sorted(str(value) for value in bootstrap_nodes):
        row = node_rows.get(str(node))
        if row is None:
            continue
        selected = dict(row.get("selected_layout", {}) or {})
        compact = _compact_layout_for_row(row)
        current_tiles = max(1, int(selected.get("tile_count", compact.get("tile_count", 1)) or 1))
        compact_tiles = max(1, int(compact.get("tile_count", current_tiles) or 1))
        already_compact = bool(
            not _layout_has_halo(selected)
            and str(row.get("physical_layout", PHYSICAL_COMPACT) or PHYSICAL_COMPACT) == PHYSICAL_COMPACT
        )
        if bool(already_compact):
            continue
        if int(current_tiles) <= int(compact_tiles) and not bool(force_compact):
            continue
        node_rows[str(node)] = _copy_node_with_compact_layout(row)
        compressed_nodes.add(str(node))
        queue.append(str(node))
        audit_nodes.append(
            {
                "node": str(node),
                "current_tile_count": int(current_tiles),
                "compact_tile_count": int(compact_tiles),
                "saved_ciphertexts_per_bootstrap": int(max(0, int(current_tiles - compact_tiles))),
                "forced_compact_boundary": bool(force_compact),
                "current_layout": dict(selected),
                "compact_layout": dict(compact),
            }
        )

    if not compressed_nodes:
        return dict(compile_plan), {"enabled": False, "reason": "no_bootstrap_ct_savings"}

    edge_rows = [dict(row) for row in compile_plan.get("edge_layouts", [])]
    while queue:
        source = queue.pop(0)
        source_layout = dict(node_rows[source].get("selected_layout", {}) or {})
        for row in edge_rows:
            if str(row.get("source", "")) != str(source):
                continue
            compact_edge = _copy_row_with_compact_layout(row, source_layout=source_layout)
            row.clear()
            row.update(compact_edge)
            target = str(row.get("target", ""))
            if (
                target
                and target not in compressed_nodes
                and target in node_rows
                and _layout_preserving_module(network_dag, target)
            ):
                target_row = node_rows[target]
                if _layout_has_halo(dict(target_row.get("selected_layout", {}) or {})):
                    node_rows[target] = _copy_node_with_compact_layout(target_row)
                    compressed_nodes.add(target)
                    queue.append(target)

    node_layouts = [
        dict(node_rows.get(str(row.get("node", "")), dict(row)))
        for row in compile_plan.get("node_layouts", [])
    ]
    summary = dict(compile_plan.get("summary", {}) or {})
    summary["bootstrap_layout_compression_count"] = int(len(audit_nodes))
    summary["bootstrap_layout_compression_saved_ciphertexts"] = int(
        sum(int(row["saved_ciphertexts_per_bootstrap"]) for row in audit_nodes)
    )
    updated_plan = {
        **dict(compile_plan),
        "edge_layouts": edge_rows,
        "node_layouts": node_layouts,
        "summary": summary,
        "bootstrap_layout_compression": {
            "enabled": True,
            "nodes": audit_nodes,
            "propagated_nodes": sorted(str(node) for node in compressed_nodes),
        },
    }
    return updated_plan, dict(updated_plan["bootstrap_layout_compression"])


def apply_bootstrap_layout_compression(network_dag: Any) -> dict[str, Any]:
    bootstrap_nodes = {
        str(node)
        for node in network_dag.nodes
        if bool(network_dag.nodes[node].get("bootstrap", False))
    }
    if not bootstrap_nodes:
        return {"enabled": False, "reason": "no_bootstrap_nodes"}

    executors = _iter_layout_policy_executors(network_dag)
    if not executors:
        return {"enabled": False, "reason": "no_layout_policy_executors"}

    compile_plan = dict(executors[0].compile_plan)
    updated_plan, audit = rewrite_layout_policy_plan_for_bootstrap_compression(
        compile_plan,
        network_dag,
        bootstrap_nodes=bootstrap_nodes,
    )
    if not bool(audit.get("enabled", False)):
        return audit

    for executor in executors:
        update = getattr(executor, "update_layout_policy_compile_plan", None)
        if callable(update):
            update(updated_plan)
        else:
            executor.compile_plan = dict(updated_plan)

    node_rows = {str(row.get("node", "")): dict(row) for row in updated_plan.get("node_layouts", [])}
    for node in audit.get("propagated_nodes", []):
        row = node_rows.get(str(node))
        if row is None or node not in network_dag.nodes:
            continue
        module = network_dag.nodes[node].get("module")
        if module is None:
            continue
        layout = dict(row.get("selected_layout", {}) or {})
        setattr(module, "layout_policy_output_layout", dict(layout))
        setattr(module, "layout_policy_output_row_offset", 0)
        setattr(module, "layout_policy_output_materialization", "bootstrap_compact")
        shape = _fhe_shape_for_layout(row, layout)
        if shape is not None:
            module.fhe_output_shape = shape

    network_dag.bootstrap_layout_compression_audit = dict(audit)
    return dict(audit)
