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
_MISSING = object()


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


def _layout_for_row_shape(
    row: dict[str, Any],
    *,
    top_beta: int,
    bottom_beta: int,
    stride: int,
) -> dict[str, int | bool]:
    shape = _shape_tuple(row.get("shape", ()))
    base = dict(row.get("selected_layout", row.get("required_layout", {})) or {})
    gap = max(1, int(base.get("gap", 1) or 1))
    slots = max(1, int(row.get("slots", 32768) or 32768))
    if shape is None:
        core_slots = int(base.get("core_slots", base.get("stored_slots", 0)) or 0)
        stored_slots = int(base.get("stored_slots", core_slots) or core_slots)
        tile_count = max(1, int(base.get("tile_count", _ceil_div(max(1, stored_slots), slots)) or 1))
    else:
        _n, channels, height, width = shape
        phase = max(1, int(gap) * int(gap))
        channel_groups = _ceil_div(int(channels), int(phase))
        row_width_slots = int(channel_groups * int(width) * int(gap))
        core_slots = int(row_width_slots * int(height) * int(gap))
        stored_slots = int(row_width_slots * (int(height) + int(top_beta) + int(bottom_beta)) * int(gap))
        tile_count = max(1, _ceil_div(int(stored_slots), int(slots)))
    return {
        "top_beta": int(top_beta),
        "bottom_beta": int(bottom_beta),
        "stride": max(1, int(stride)),
        "gap": int(gap),
        "core_slots": int(core_slots),
        "stored_slots": int(stored_slots),
        "tile_count": int(tile_count),
        "physical_top_beta": int(top_beta),
        "physical_bottom_beta": int(bottom_beta),
        "boundary_pruned": False,
    }


def _layout_covers(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        int(dict(left).get("gap", 1) or 1) == int(dict(right).get("gap", 1) or 1)
        and _layout_top_beta(left) >= _layout_top_beta(right)
        and _layout_bottom_beta(left) >= _layout_bottom_beta(right)
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


def _iter_layout_policy_bindings(network_dag: Any) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    seen: set[int] = set()
    for node in network_dag.nodes:
        module = network_dag.nodes[node].get("module")
        if module is None:
            continue
        region = getattr(module, "region_runtime", None)
        executor = getattr(region, "executor", None) if region is not None else None
        if executor is not None and id(executor) not in seen:
            seen.add(id(executor))
            if isinstance(getattr(executor, "compile_plan", None), dict):
                bindings.append(
                    {
                        "node": str(node),
                        "module": module,
                        "runtime": region,
                        "executor": executor,
                    }
                )
        for runtime_name in ("layout_policy_add_runtime", "layout_policy_concat_runtime"):
            runtime = getattr(module, runtime_name, None)
            if runtime is None or id(runtime) in seen:
                continue
            seen.add(id(runtime))
            if isinstance(getattr(runtime, "compile_plan", None), dict):
                bindings.append(
                    {
                        "node": str(node),
                        "module": module,
                        "runtime": None,
                        "executor": runtime,
                    }
                )
    return bindings


def _layout_policy_compile_plan_for_dag(network_dag: Any) -> dict[str, Any] | None:
    for binding in _iter_layout_policy_bindings(network_dag):
        plan = getattr(binding["executor"], "compile_plan", None)
        if isinstance(plan, dict) and plan.get("edge_layouts"):
            return dict(plan)
    return None


def bootstrap_aware_layout_refinement_applicable(network_dag: Any) -> bool:
    plan = _layout_policy_compile_plan_for_dag(network_dag)
    if not isinstance(plan, dict):
        return False
    return str(plan.get("policy", "")) == "dp_no_share_fold"


def _executor_relayout_depth(executor: Any) -> int:
    relayout_rows = tuple(getattr(executor, "relayout_rows", ()) or ())
    native_physical_rows = tuple(getattr(executor, "native_physical_relayout_rows", ()) or ())
    output_rows = tuple(getattr(executor, "output_relayout_rows", ()) or ())
    native_halo = bool(getattr(executor, "native_halo_input", False))
    return int(
        (len(relayout_rows) if bool(native_halo) else 2 * len(relayout_rows))
        + len(native_physical_rows)
        + len(output_rows)
    )


def _snapshot_layout_policy_depths(network_dag: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for binding in _iter_layout_policy_bindings(network_dag):
        runtime = binding.get("runtime")
        module = binding.get("module")
        executor = binding.get("executor")
        rows.append(
            {
                "node": str(binding.get("node", "")),
                "runtime": runtime,
                "module": module,
                "executor": executor,
                "runtime_depth": getattr(runtime, "depth", None) if runtime is not None else None,
                "runtime_solver_depth": getattr(runtime, "solver_depth", None) if runtime is not None else None,
                "module_depth": getattr(module, "depth", None) if module is not None else None,
                "relayout_depth": _executor_relayout_depth(executor),
            }
        )
    return rows


def _layout_policy_module_layout_snapshot(network_dag: Any, nodes: set[str] | None = None) -> list[dict[str, Any]]:
    selected_nodes = None if nodes is None else {str(node) for node in nodes}
    rows: list[dict[str, Any]] = []
    for node in network_dag.nodes:
        node = str(node)
        if selected_nodes is not None and node not in selected_nodes:
            continue
        module = network_dag.nodes[node].get("module")
        if module is None:
            continue
        rows.append(
            {
                "node": str(node),
                "module": module,
                "fhe_output_shape": getattr(module, "fhe_output_shape", _MISSING),
                "layout_policy_output_layout": getattr(module, "layout_policy_output_layout", _MISSING),
                "layout_policy_output_row_offset": getattr(module, "layout_policy_output_row_offset", _MISSING),
                "layout_policy_output_materialization": getattr(
                    module,
                    "layout_policy_output_materialization",
                    _MISSING,
                ),
            }
        )
    return rows


def _restore_layout_policy_module_layout_snapshot(snapshot: list[dict[str, Any]]) -> None:
    for row in snapshot:
        module = row.get("module")
        if module is None:
            continue
        for attr in (
            "fhe_output_shape",
            "layout_policy_output_layout",
            "layout_policy_output_row_offset",
            "layout_policy_output_materialization",
        ):
            value = row.get(attr, _MISSING)
            if value is _MISSING:
                try:
                    delattr(module, attr)
                except AttributeError:
                    pass
            else:
                setattr(module, attr, value)


def _apply_layout_policy_module_output_layouts(network_dag: Any, compile_plan: dict[str, Any]) -> list[dict[str, Any]]:
    changed_nodes: set[str] = set()
    for row in compile_plan.get("node_layouts", []):
        node = str(row.get("node", ""))
        if not node or node not in network_dag.nodes:
            continue
        layout = dict(row.get("selected_layout", {}) or {})
        if not _layout_has_halo(layout):
            continue
        module = network_dag.nodes[node].get("module")
        if module is None:
            continue
        changed_nodes.add(str(node))
    snapshot = _layout_policy_module_layout_snapshot(network_dag, changed_nodes)
    for row in compile_plan.get("node_layouts", []):
        node = str(row.get("node", ""))
        if node not in changed_nodes:
            continue
        module = network_dag.nodes[node].get("module")
        if module is None:
            continue
        layout = dict(row.get("selected_layout", {}) or {})
        shape = _fhe_shape_for_layout(dict(row), layout)
        if shape is not None:
            module.fhe_output_shape = shape
        gap = max(1, int(layout.get("gap", 1) or 1))
        top_beta = _layout_top_beta(layout)
        module.layout_policy_output_layout = dict(layout)
        module.layout_policy_output_row_offset = int(top_beta * gap)
        if bool(row.get("producer_materialized_halo", False)) or _layout_has_halo(layout):
            module.layout_policy_output_materialization = "fused_relayout"
    return snapshot


def restore_layout_policy_compile_plan(
    network_dag: Any,
    compile_plan: dict[str, Any],
    *,
    depth_snapshot: list[dict[str, Any]] | None = None,
    module_layout_snapshot: list[dict[str, Any]] | None = None,
) -> None:
    for binding in _iter_layout_policy_bindings(network_dag):
        executor = binding["executor"]
        update = getattr(executor, "update_layout_policy_compile_plan", None)
        if callable(update):
            update(dict(compile_plan))
        else:
            executor.compile_plan = dict(compile_plan)
    for row in depth_snapshot or []:
        runtime = row.get("runtime")
        module = row.get("module")
        if runtime is not None:
            if row.get("runtime_depth") is not None:
                runtime.depth = int(row["runtime_depth"])
            if row.get("runtime_solver_depth") is not None:
                runtime.solver_depth = int(row["runtime_solver_depth"])
        if module is not None and row.get("module_depth") is not None:
            module.depth = int(row["module_depth"])
    if module_layout_snapshot is not None:
        _restore_layout_policy_module_layout_snapshot(module_layout_snapshot)


def _refresh_layout_policy_depths(
    network_dag: Any,
    *,
    previous_depths: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    previous_by_executor = {id(row["executor"]): dict(row) for row in previous_depths}
    updates: list[dict[str, Any]] = []
    for binding in _iter_layout_policy_bindings(network_dag):
        runtime = binding.get("runtime")
        module = binding.get("module")
        executor = binding.get("executor")
        if runtime is None or module is None or executor is None:
            continue
        previous = previous_by_executor.get(id(executor), {})
        old_runtime_depth = getattr(runtime, "solver_depth", None)
        if old_runtime_depth is None:
            old_runtime_depth = getattr(runtime, "depth", None)
        if old_runtime_depth is None:
            continue
        old_relayout_depth = int(previous.get("relayout_depth", _executor_relayout_depth(executor)) or 0)
        base_depth = max(0, int(old_runtime_depth) - int(old_relayout_depth))
        new_relayout_depth = _executor_relayout_depth(executor)
        new_depth = int(base_depth + int(new_relayout_depth))
        runtime.depth = int(new_depth)
        runtime.solver_depth = int(new_depth)
        if hasattr(module, "set_depth"):
            module.set_depth(int(new_depth))
        else:
            module.depth = int(new_depth)
        updates.append(
            {
                "node": str(binding.get("node", "")),
                "base_depth": int(base_depth),
                "old_relayout_depth": int(old_relayout_depth),
                "new_relayout_depth": int(new_relayout_depth),
                "old_solver_depth": int(old_runtime_depth),
                "new_solver_depth": int(new_depth),
            }
        )
    return updates


def _rebuild_relayout_edges(edge_layouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in edge_layouts:
        if not bool(row.get("relayout", False)):
            continue
        layout = dict(row.get("selected_layout", {}) or {})
        rows.append(
            {
                "edge": str(row.get("edge", "")),
                "source": str(row.get("source", "")),
                "target": str(row.get("target", "")),
                "reason": str(row.get("relayout_reason", "")),
                "source_layout": dict(row.get("source_layout", {}) or {}),
                "selected_layout": dict(layout),
                "target_layout": dict(row.get("target_layout", layout) or layout),
                "rotation_estimate": int(row.get("relayout_rotation_estimate", 0) or 0),
                "mask_mult_estimate": int(row.get("relayout_mask_mult_estimate", 0) or 0),
                "sparse_lt_estimate": int(row.get("relayout_sparse_lt_estimate", 0) or 0),
                "depth_estimate": int(row.get("relayout_depth_estimate", 0) or 0),
            }
        )
    return rows


def _refresh_layout_policy_plan_summary(compile_plan: dict[str, Any]) -> dict[str, Any]:
    edge_layouts = [dict(row) for row in compile_plan.get("edge_layouts", [])]
    relayout_edges = _rebuild_relayout_edges(edge_layouts)
    output_nodes = [
        dict(row)
        for row in compile_plan.get("output_relayout_nodes", [])
        if bool(row.get("selected_layout", {}))
    ]
    summary = dict(compile_plan.get("summary", {}) or {})
    summary["relayouts"] = int(len(relayout_edges) + len(output_nodes))
    summary["relayout_rotation_estimate"] = int(
        sum(int(row.get("rotation_estimate", 0) or 0) for row in relayout_edges)
        + sum(int(row.get("rotation_estimate", 0) or 0) for row in output_nodes)
    )
    summary["relayout_mask_mult_estimate"] = int(
        sum(int(row.get("mask_mult_estimate", 0) or 0) for row in relayout_edges)
        + sum(int(row.get("mask_mult_estimate", 0) or 0) for row in output_nodes)
    )
    summary["relayout_depth_estimate"] = int(
        sum(int(row.get("depth_estimate", 0) or 0) for row in relayout_edges)
        + sum(int(row.get("depth_estimate", 0) or 0) for row in output_nodes)
    )
    summary["consumer_fused_relayout_count"] = int(
        sum(1 for row in edge_layouts if bool(row.get("consumer_fused_relayout", False)))
    )
    summary["consumer_fused_rotation_estimate"] = int(
        sum(int(row.get("consumer_fused_rotation_estimate", 0) or 0) for row in edge_layouts)
    )
    return {
        **dict(compile_plan),
        "edge_layouts": edge_layouts,
        "relayout_edges": relayout_edges,
        "relayout_edge_count": int(len(relayout_edges)),
        "output_relayout_nodes": output_nodes,
        "output_relayout_node_count": int(len(output_nodes)),
        "summary": summary,
    }


def _compact_physical_for_refined_source(row: dict[str, Any]) -> str:
    source_physical = str(row.get("source_physical_layout", "") or "")
    if source_physical in {PHYSICAL_COMPACT, PHYSICAL_LOGICAL_HALO}:
        return str(source_physical)
    source_layout = dict(row.get("source_layout", {}) or {})
    return PHYSICAL_LOGICAL_HALO if _layout_has_halo(source_layout) else PHYSICAL_COMPACT


def _rewrite_native_physical_relayout_row_for_boot_refinement(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("op_kind", "")) != "conv2d":
        return None
    if str(row.get("source", "")) == "x":
        return None
    if str(row.get("physical_layout", "")) != "native_source_stripe":
        return None
    source_physical = _compact_physical_for_refined_source(row)
    if source_physical not in {PHYSICAL_COMPACT, PHYSICAL_LOGICAL_HALO}:
        return None
    source_layout = dict(row.get("source_layout", {}) or {})
    selected_layout = dict(row.get("selected_layout", {}) or {})
    if source_physical != PHYSICAL_LOGICAL_HALO:
        return None
    if not _layout_has_halo(source_layout):
        return None
    if _layout_top_beta(source_layout) < _layout_top_beta(selected_layout):
        return None
    if _layout_bottom_beta(source_layout) < _layout_bottom_beta(selected_layout):
        return None
    updated = dict(row)
    updated["relayout"] = False
    updated["relayout_reason"] = ""
    updated["relayout_rotation_estimate"] = 0
    updated["relayout_mask_mult_estimate"] = 0
    updated["relayout_sparse_lt_estimate"] = 0
    updated["relayout_depth_estimate"] = 0
    updated["physical_layout"] = str(source_physical)
    updated["target_physical_layout"] = str(source_physical)
    updated["layout_mode"] = "boot_refined_compact_source"
    updated["consumer_fused_relayout"] = True
    updated["boot_refined_compact_source"] = True
    updated["boot_refinement_reason"] = "boot_interval_native_physical_relayout"
    updated["boot_refinement_original_physical_layout"] = "native_source_stripe"
    return updated


def _producer_beta_lift_allowed(module: Any | None) -> bool:
    return type(module).__name__ in {"AvgPool2d", "Conv2d", "ConvTranspose2d"}


def _pair_first(value: Any, default: int = 1) -> int:
    if isinstance(value, (tuple, list)):
        return int(value[0]) if value else int(default)
    return int(value if value is not None else default)


def _pair_max(value: Any, default: int = 1) -> int:
    if isinstance(value, (tuple, list)):
        return max((int(item) for item in value), default=int(default)) if value else int(default)
    return int(value if value is not None else default)


def _conv_halo_consume(module: Any | None) -> int:
    if module is None:
        return 0
    kernel = _pair_max(getattr(module, "kernel_size", 1), 1)
    padding = _pair_max(getattr(module, "padding", 0), 0)
    if int(padding) > 0:
        return max(0, int(kernel) // 2)
    return max(0, int(kernel) - 1)


def _compact_height_strip_fits_single_ct(*, row: dict[str, Any]) -> bool:
    shape = _shape_tuple(row.get("shape", ()))
    if shape is None:
        return False
    layout = dict(row.get("compact_layout", row.get("required_layout", row.get("selected_layout", {}))) or {})
    gap = max(1, int(layout.get("gap", 1) or 1))
    slots = max(1, int(row.get("slots", 32768) or 32768))
    _n, _channels, height, width = shape
    return bool(int(height) * int(gap) * int(width) * int(gap) <= int(slots))


def _input_demand_for_output_layout(
    module: Any | None,
    edge_row: dict[str, Any],
    output_layout: dict[str, Any],
) -> dict[str, Any]:
    if type(module).__name__ in {"AvgPool2d", "Conv2d"} and _compact_height_strip_fits_single_ct(row=edge_row):
        return dict(edge_row.get("required_layout", edge_row.get("selected_layout", {})) or {})
    if type(module).__name__ == "ConvTranspose2d":
        scale = max(1, _pair_first(getattr(module, "stride", 1), 1))
        top_beta = _ceil_div(_layout_top_beta(output_layout), int(scale))
        bottom_beta = _ceil_div(_layout_bottom_beta(output_layout), int(scale))
    elif type(module).__name__ == "AvgPool2d":
        stride = max(1, _pair_max(getattr(module, "stride", 1), 1))
        kernel = max(1, _pair_max(getattr(module, "kernel_size", 1), 1))
        consume = max(0, int(kernel) - int(stride))
        top_beta = _layout_top_beta(output_layout) * int(stride) + int(consume)
        bottom_beta = _layout_bottom_beta(output_layout) * int(stride) + int(consume)
    elif type(module).__name__ == "Conv2d":
        stride = max(1, _pair_max(getattr(module, "stride", 1), 1))
        consume = _conv_halo_consume(module)
        top_beta = _layout_top_beta(output_layout) * int(stride) + int(consume)
        bottom_beta = _layout_bottom_beta(output_layout) * int(stride) + int(consume)
    else:
        top_beta = _layout_top_beta(output_layout)
        bottom_beta = _layout_bottom_beta(output_layout)
    required = dict(edge_row.get("required_layout", {}) or {})
    return _layout_for_row_shape(
        edge_row,
        top_beta=max(_layout_top_beta(required), int(top_beta)),
        bottom_beta=max(_layout_bottom_beta(required), int(bottom_beta)),
        stride=max(1, int(required.get("stride", output_layout.get("stride", 1)) or 1)),
    )


def _semantic_output_layout(
    module: Any | None,
    edge_row: dict[str, Any],
    source_layout: dict[str, Any],
) -> dict[str, Any]:
    if type(module).__name__ == "ConvTranspose2d":
        scale = max(1, _pair_first(getattr(module, "stride", 1), 1))
        top_beta = _layout_top_beta(source_layout) * int(scale)
        bottom_beta = _layout_bottom_beta(source_layout) * int(scale)
    elif type(module).__name__ == "AvgPool2d":
        stride = max(1, _pair_max(getattr(module, "stride", 1), 1))
        kernel = max(1, _pair_max(getattr(module, "kernel_size", 1), 1))
        consume = max(0, int(kernel) - int(stride))
        top_beta = max(0, (_layout_top_beta(source_layout) - int(consume)) // int(stride))
        bottom_beta = max(0, (_layout_bottom_beta(source_layout) - int(consume)) // int(stride))
    elif type(module).__name__ == "Conv2d":
        stride = max(1, _pair_max(getattr(module, "stride", 1), 1))
        consume = _conv_halo_consume(module)
        top_beta = max(0, (_layout_top_beta(source_layout) - int(consume)) // int(stride))
        bottom_beta = max(0, (_layout_bottom_beta(source_layout) - int(consume)) // int(stride))
    else:
        top_beta = _layout_top_beta(source_layout)
        bottom_beta = _layout_bottom_beta(source_layout)
    return _layout_for_row_shape(
        edge_row,
        top_beta=int(top_beta),
        bottom_beta=int(bottom_beta),
        stride=max(1, int(source_layout.get("stride", 1) or 1)),
    )


def _actual_native_edge_ids(network_dag: Any) -> set[str]:
    edges: set[str] = set()
    for binding in _iter_layout_policy_bindings(network_dag):
        executor = binding["executor"]
        for row in tuple(getattr(executor, "native_physical_relayout_rows", ()) or ()):
            edges.add(str(row.get("edge", "")))
    return edges


def _bootstrap_edge_set(first_pass_audit: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(row.get("source", "")), str(row.get("target", "")))
        for row in first_pass_audit.get("boot_edges", [])
        if str(row.get("source", "")) and str(row.get("target", ""))
    }


def _bootstrap_upstream_intervals(
    network_dag: Any,
    first_pass_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return no-boot upstream components that feed each first-pass boot edge."""

    boot_edges = _bootstrap_edge_set(first_pass_audit)
    intervals: list[dict[str, Any]] = []
    for row in first_pass_audit.get("boot_edges", []):
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        if not source or source not in network_dag.nodes:
            continue
        visited: set[str] = set()
        stack = [source]
        interval_edges: set[tuple[str, str]] = set()
        while stack:
            node = str(stack.pop())
            if node in visited or node not in network_dag.nodes:
                continue
            visited.add(node)
            for pred in network_dag.predecessors(node):
                pred = str(pred)
                edge = (pred, node)
                if edge in boot_edges:
                    continue
                interval_edges.add(edge)
                if pred in network_dag.nodes:
                    stack.append(pred)
        intervals.append(
            {
                "boot_edge": (source, target),
                "nodes": visited,
                "edges": interval_edges,
            }
        )
    return intervals


def _candidate_interval_matches(
    *,
    row: dict[str, Any],
    intervals: list[dict[str, Any]],
    boot_edges: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    source = str(row.get("source", ""))
    target = str(row.get("target", ""))
    if not source or not target:
        return []
    if (source, target) in boot_edges:
        return []
    matches: list[dict[str, Any]] = []
    for interval in intervals:
        nodes = set(interval.get("nodes", set()) or set())
        edges = set(interval.get("edges", set()) or set())
        if target in nodes and (source, target) in edges:
            matches.append(interval)
    return matches


def _linear_interval_path(network_dag: Any, interval: dict[str, Any]) -> list[str] | None:
    nodes = set(str(node) for node in (interval.get("nodes", set()) or set()))
    if not nodes:
        return None
    sub_edges = [
        (str(source), str(target))
        for source, target in (interval.get("edges", set()) or set())
        if str(source) in nodes and str(target) in nodes
    ]
    preds = {node: 0 for node in nodes}
    succs = {node: 0 for node in nodes}
    for source, target in sub_edges:
        succs[source] += 1
        preds[target] += 1
    starts = [node for node in nodes if preds[node] == 0]
    ends = [node for node in nodes if succs[node] == 0]
    if len(starts) != 1 or len(ends) != 1:
        return None
    if any(count > 1 for count in preds.values()) or any(count > 1 for count in succs.values()):
        return None
    start = starts[0]
    path = [start]
    current = start
    edge_map = {source: target for source, target in sub_edges}
    while current in edge_map:
        current = edge_map[current]
        path.append(current)
    if set(path) != nodes:
        return None
    return path


def _rewrite_row_as_compact_source(
    row: dict[str, Any],
    *,
    source_layout: dict[str, Any],
    target_layout: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    updated = dict(row)
    updated["source_layout"] = dict(source_layout)
    updated["selected_layout"] = dict(target_layout)
    updated["target_layout"] = dict(target_layout)
    updated["physical_layout"] = PHYSICAL_LOGICAL_HALO if _layout_has_halo(target_layout) else PHYSICAL_COMPACT
    updated["source_physical_layout"] = PHYSICAL_LOGICAL_HALO if _layout_has_halo(source_layout) else PHYSICAL_COMPACT
    updated["target_physical_layout"] = str(updated["physical_layout"])
    updated["relayout"] = False
    updated["relayout_reason"] = ""
    updated["relayout_rotation_estimate"] = 0
    updated["relayout_mask_mult_estimate"] = 0
    updated["relayout_sparse_lt_estimate"] = 0
    updated["relayout_depth_estimate"] = 0
    updated["consumer_fused_relayout"] = True
    updated["consumer_fused_rotation_estimate"] = int(updated.get("consumer_fused_rotation_estimate", 0) or 0)
    updated["boot_refined_beta_lift"] = True
    updated["boot_refinement_reason"] = str(reason)
    if str(row.get("op_kind", "")) == "conv2d":
        updated["layout_mode"] = "compact_halo_shared" if _layout_covers(source_layout, target_layout) else "compact_align_shared"
        updated["provider_lt_grouping_mode"] = "individual"
        updated["native_halo_channel_fold_mode"] = "per_stripe"
    return updated


def _rewrite_node_as_beta_lift_producer(
    row: dict[str, Any],
    *,
    layout: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    updated = dict(row)
    updated["selected_layout"] = dict(layout)
    updated["physical_layout"] = PHYSICAL_LOGICAL_HALO if _layout_has_halo(layout) else PHYSICAL_COMPACT
    updated["output_relayout"] = False
    updated["output_relayout_reason"] = ""
    updated["producer_materialized_halo"] = bool(_layout_has_halo(layout))
    updated["producer_materialized_halo_reason"] = str(reason) if _layout_has_halo(layout) else ""
    updated["producer_fused_rotation_estimate"] = 0
    updated["producer_fused_depth_estimate"] = 0
    shape = _fhe_shape_for_layout(updated, layout)
    if shape is not None:
        updated["fhe_shape"] = [int(value) for value in shape]
    return updated


def _activation_transparent_single_native_beta_lift(
    network_dag: Any,
    *,
    path: list[str],
    path_edges: list[tuple[str, str]],
    start_index: int,
    producer_module: Any | None,
    lifted_edges: list[tuple[str, str]],
) -> bool:
    if type(producer_module).__name__ != "Conv2d":
        return False
    if len(lifted_edges) != 1:
        return False
    native_edge = tuple(lifted_edges[0])
    try:
        native_edge_index = path_edges.index(native_edge)
    except ValueError:
        return False
    if int(native_edge_index) <= int(start_index):
        return False
    activation_nodes = [str(node) for node in path[start_index + 1 : native_edge_index + 1]]
    if not activation_nodes:
        return False
    return all(_layout_preserving_module(network_dag, node) for node in activation_nodes)


def _pool_direct_single_native_beta_lift(
    network_dag: Any,
    *,
    path_edges: list[tuple[str, str]],
    start_index: int,
    producer_module: Any | None,
    lifted_edges: list[tuple[str, str]],
) -> bool:
    if type(producer_module).__name__ != "AvgPool2d":
        return False
    if len(lifted_edges) != 1:
        return False
    native_edge = tuple(lifted_edges[0])
    if native_edge != tuple(path_edges[int(start_index)]):
        return False
    _source, target = native_edge
    target_module = network_dag.nodes[str(target)].get("module") if str(target) in network_dag.nodes else None
    return type(target_module).__name__ == "Conv2d"


def _try_apply_beta_lift_candidate(
    network_dag: Any,
    compile_plan: dict[str, Any],
    interval: dict[str, Any],
    *,
    actual_native_edges: set[str],
) -> dict[str, Any] | None:
    path = _linear_interval_path(network_dag, interval)
    if path is None or len(path) < 2:
        return None
    plan_slots = max(1, int(compile_plan.get("slots", 32768) or 32768))
    edge_rows_by_edge = {
        str(row.get("edge", "")): {**dict(row), "slots": int(plan_slots)}
        for row in compile_plan.get("edge_layouts", [])
    }
    node_rows_by_node = {str(row.get("node", "")): dict(row) for row in compile_plan.get("node_layouts", [])}
    path_edges = [(path[index], path[index + 1]) for index in range(len(path) - 1)]
    boot_source, boot_target = interval.get("boot_edge", ("", ""))
    interval_edge_set = set(path_edges)
    for start_index in range(len(path_edges)):
        producer = str(path[start_index])
        if producer == "x":
            continue
        producer_module = network_dag.nodes[producer].get("module") if producer in network_dag.nodes else None
        if not _producer_beta_lift_allowed(producer_module):
            continue
        if len(list(network_dag.successors(producer))) != 1:
            continue
        producer_row = node_rows_by_node.get(producer)
        if producer_row is None:
            continue
        changed_output_nodes = set(path[start_index:-1])
        external_fanout = False
        for node in changed_output_nodes:
            for successor in network_dag.successors(node):
                if (str(node), str(successor)) not in interval_edge_set:
                    external_fanout = True
                    break
            if external_fanout:
                break
        if external_fanout:
            continue
        lifted_edges: list[tuple[str, str]] = []
        current_demand: dict[str, Any] | None = None
        current_source_layout: dict[str, Any] | None = None
        for edge_source, edge_target in reversed(path_edges[start_index:]):
            edge_id = f"{edge_source}->{edge_target}"
            row = edge_rows_by_edge.get(edge_id)
            if row is None:
                break
            if edge_id in actual_native_edges:
                if str(row.get("op_kind", "")) != "conv2d":
                    break
                if str(row.get("source", "")) == "x":
                    break
                if str(row.get("physical_layout", "")) != "native_source_stripe":
                    break
                selected = dict(row.get("selected_layout", {}) or {})
                required = dict(row.get("required_layout", selected) or selected)
                if not _layout_covers(selected, required):
                    break
                if int(selected.get("gap", 1) or 1) != int(required.get("gap", selected.get("gap", 1)) or 1):
                    break
                current_demand = (
                    selected
                    if current_demand is None
                    else _input_demand_for_output_layout(
                        network_dag.nodes[edge_target].get("module") if edge_target in network_dag.nodes else None,
                        row,
                        current_demand,
                    )
                )
                lifted_edges.append((edge_source, edge_target))
            elif current_demand is not None:
                current_demand = _input_demand_for_output_layout(
                    network_dag.nodes[edge_target].get("module") if edge_target in network_dag.nodes else None,
                    row,
                    current_demand,
                )
            if edge_source == producer:
                current_source_layout = current_demand
                break
        if current_source_layout is None:
            continue
        if len(lifted_edges) < 2:
            activation_transparent_single_native = _activation_transparent_single_native_beta_lift(
                network_dag,
                path=path,
                path_edges=path_edges,
                start_index=int(start_index),
                producer_module=producer_module,
                lifted_edges=lifted_edges,
            )
            pool_direct_single_native = _pool_direct_single_native_beta_lift(
                network_dag,
                path_edges=path_edges,
                start_index=int(start_index),
                producer_module=producer_module,
                lifted_edges=lifted_edges,
            )
            if not activation_transparent_single_native and not pool_direct_single_native:
                continue
        else:
            activation_transparent_single_native = False
            pool_direct_single_native = False
        if not _layout_has_halo(current_source_layout):
            continue
        updated_edge_rows = []
        live_layout = dict(current_source_layout)
        accepted_edges: list[dict[str, Any]] = []
        carried_edges: list[dict[str, Any]] = []
        path_edge_set = set(path_edges[start_index:])
        edge_index_by_pair = {pair: index for index, pair in enumerate(path_edges)}
        for row in compile_plan.get("edge_layouts", []):
            edge_source = str(row.get("source", ""))
            edge_target = str(row.get("target", ""))
            edge_id = str(row.get("edge", ""))
            if (edge_source, edge_target) in path_edge_set:
                input_layout = dict(live_layout)
                if edge_id in actual_native_edges:
                    required = dict(row.get("required_layout", row.get("selected_layout", {})) or {})
                    if not _layout_covers(input_layout, required):
                        accepted_edges = []
                        break
                    updated = _rewrite_row_as_compact_source(
                        dict(row),
                        source_layout=input_layout,
                        target_layout=input_layout,
                        reason="boot_interval_beta_lift",
                    )
                    accepted_edges.append(
                        {
                            "edge": str(edge_id),
                            "source": str(edge_source),
                            "target": str(edge_target),
                            "old_physical_layout": str(row.get("physical_layout", "")),
                            "new_physical_layout": str(updated.get("physical_layout", "")),
                        }
                    )
                    updated_edge_rows.append(updated)
                elif _layout_preserving_module(network_dag, edge_target):
                    updated = _rewrite_row_as_compact_source(
                        dict(row),
                        source_layout=input_layout,
                        target_layout=input_layout,
                        reason="boot_interval_activation_transparent_beta_lift",
                    )
                    carried_edges.append(
                        {
                            "edge": str(edge_id),
                            "source": str(edge_source),
                            "target": str(edge_target),
                            "op_kind": str(row.get("op_kind", "")),
                            "physical_layout": str(updated.get("physical_layout", "")),
                        }
                    )
                    updated_edge_rows.append(updated)
                else:
                    updated_edge_rows.append(dict(row))
                edge_index = int(edge_index_by_pair[(edge_source, edge_target)])
                if edge_index + 1 < len(path_edges):
                    next_source, next_target = path_edges[edge_index + 1]
                    next_row = edge_rows_by_edge.get(f"{next_source}->{next_target}", dict(row))
                    live_layout = _semantic_output_layout(
                        network_dag.nodes[edge_target].get("module") if edge_target in network_dag.nodes else None,
                        dict(next_row),
                        input_layout,
                    )
            else:
                updated_edge_rows.append(dict(row))
        live_layout = dict(current_source_layout)
        node_output_updates: dict[str, dict[str, Any]] = {str(producer): dict(current_source_layout)}
        for edge_index, (_edge_source, edge_target) in enumerate(path_edges[start_index:]):
            absolute_edge_index = int(start_index + edge_index)
            if absolute_edge_index + 1 >= len(path_edges):
                break
            next_source, next_target = path_edges[absolute_edge_index + 1]
            next_row = edge_rows_by_edge.get(f"{next_source}->{next_target}")
            if next_row is None:
                break
            live_layout = _semantic_output_layout(
                network_dag.nodes[edge_target].get("module") if edge_target in network_dag.nodes else None,
                dict(next_row),
                live_layout,
            )
            node_output_updates[str(edge_target)] = dict(live_layout)
        updated_node_rows = []
        for row in compile_plan.get("node_layouts", []):
            node = str(row.get("node", ""))
            layout = node_output_updates.get(node)
            if layout is None:
                updated_node_rows.append(dict(row))
                continue
            updated_node_rows.append(
                _rewrite_node_as_beta_lift_producer(
                    dict(row),
                    layout=layout,
                    reason="boot_interval_beta_lift",
                )
            )
        if len(accepted_edges) < 2 and not activation_transparent_single_native:
            if not pool_direct_single_native:
                continue
        candidate_kind = (
            "activation_transparent_beta_lift"
            if bool(activation_transparent_single_native)
            else "pool_direct_beta_lift"
            if bool(pool_direct_single_native)
            else "producer_beta_lift"
        )
        updated_plan = _refresh_layout_policy_plan_summary(
            {
                **dict(compile_plan),
                "edge_layouts": updated_edge_rows,
                "node_layouts": updated_node_rows,
            }
        )
        summary = dict(updated_plan.get("summary", {}) or {})
        summary["boot_refined_beta_lift_count"] = int(summary.get("boot_refined_beta_lift_count", 0) or 0) + 1
        updated_plan["summary"] = summary
        return {
            "kind": candidate_kind,
            "strategy": candidate_kind,
            "plan": updated_plan,
            "accepted": [
                {
                    "kind": candidate_kind,
                    "producer": str(producer),
                    "boot_edge": {"source": str(boot_source), "target": str(boot_target)},
                    "producer_layout": dict(current_source_layout),
                    "covered_edges": accepted_edges,
                    "carried_edges": carried_edges,
                    "covered_edge_count": int(len(accepted_edges)),
                }
            ],
            "rejected": [],
        }
    return None


def _boot_interval_audit_rows(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "boot_edge": {
                "source": str(interval.get("boot_edge", ("", ""))[0]),
                "target": str(interval.get("boot_edge", ("", ""))[1]),
            },
            "node_count": int(len(interval.get("nodes", set()) or set())),
            "edge_count": int(len(interval.get("edges", set()) or set())),
        }
        for interval in intervals
    ]


def _native_physical_relayout_refinement_candidate(
    compile_plan: dict[str, Any],
    *,
    intervals: list[dict[str, Any]],
    boot_edges: set[tuple[str, str]],
    actual_native_physical_edges: set[str],
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    edge_layouts: list[dict[str, Any]] = []
    for row in compile_plan.get("edge_layouts", []):
        current = dict(row)
        edge_id = str(current.get("edge", ""))
        source = str(current.get("source", ""))
        target = str(current.get("target", ""))
        matches = _candidate_interval_matches(
            row=current,
            intervals=intervals,
            boot_edges=boot_edges,
        )
        if edge_id in actual_native_physical_edges and matches:
            if len(matches) != 1:
                rejected.append(
                    {
                        "edge": str(edge_id),
                        "source": str(source),
                        "target": str(target),
                        "reason": "ambiguous_multiple_boot_intervals",
                        "boot_edges": [
                            {
                                "source": str(match.get("boot_edge", ("", ""))[0]),
                                "target": str(match.get("boot_edge", ("", ""))[1]),
                            }
                            for match in matches
                        ],
                    }
                )
                edge_layouts.append(current)
                continue
            rewritten = _rewrite_native_physical_relayout_row_for_boot_refinement(current)
            if rewritten is not None:
                boot_source, boot_target = matches[0].get("boot_edge", ("", ""))
                edge_layouts.append(rewritten)
                accepted.append(
                    {
                        "edge": str(edge_id),
                        "source": str(source),
                        "target": str(target),
                        "boot_edge": {
                            "source": str(boot_source),
                            "target": str(boot_target),
                        },
                        "op_kind": str(current.get("op_kind", "")),
                        "old_physical_layout": str(current.get("physical_layout", "")),
                        "new_physical_layout": str(rewritten.get("physical_layout", "")),
                        "old_relayout_depth": int(current.get("relayout_depth_estimate", 0) or 0),
                        "new_relayout_depth": 0,
                    }
                )
                continue
            rejected.append(
                {
                    "edge": str(edge_id),
                    "source": str(source),
                    "target": str(target),
                    "reason": "source_not_materialized_logical_halo_compact",
                }
            )
        edge_layouts.append(current)

    if not accepted:
        return {
            "enabled": False,
            "reason": "no_supported_boot_interval_native_physical_relayout",
            "rejected": rejected,
            "native_physical_relayout_edge_count": int(len(actual_native_physical_edges)),
            "boot_interval_count": int(len(intervals)),
        }

    updated_plan = _refresh_layout_policy_plan_summary(
        {
            **dict(compile_plan),
            "edge_layouts": edge_layouts,
        }
    )
    summary = dict(updated_plan.get("summary", {}) or {})
    summary["boot_refined_compact_source_count"] = int(len(accepted))
    updated_plan["summary"] = summary
    updated_plan["bootstrap_aware_layout_refinement"] = {
        "enabled": True,
        "accepted": accepted,
        "rejected": rejected,
    }
    return {
        "enabled": True,
        "kind": "native_physical_relayout",
        "strategy": "native_physical_relayout",
        "plan": updated_plan,
        "accepted": accepted,
        "rejected": rejected,
        "accepted_count": int(len(accepted)),
        "native_physical_relayout_edge_count": int(len(actual_native_physical_edges)),
        "boot_interval_count": int(len(intervals)),
    }


def _concat_output_beta_lift_refinement_candidates(
    network_dag: Any,
    compile_plan: dict[str, Any],
    *,
    intervals: list[dict[str, Any]],
    boot_edges: set[tuple[str, str]],
    actual_native_physical_edges: set[str],
) -> list[dict[str, Any]]:
    edge_rows_by_edge = {str(row.get("edge", "")): dict(row) for row in compile_plan.get("edge_layouts", [])}
    node_rows_by_node = {str(row.get("node", "")): dict(row) for row in compile_plan.get("node_layouts", [])}
    candidates: list[dict[str, Any]] = []
    for row in compile_plan.get("edge_layouts", []):
        current = dict(row)
        edge_id = str(current.get("edge", ""))
        source = str(current.get("source", ""))
        target = str(current.get("target", ""))
        if edge_id not in actual_native_physical_edges:
            continue
        if str(current.get("op_kind", "")) != "conv2d":
            continue
        if str(current.get("physical_layout", "")) != "native_source_stripe":
            continue
        if not source or not target or source not in network_dag.nodes or target not in network_dag.nodes:
            continue
        if type(network_dag.nodes[source].get("module")).__name__ != "Concat":
            continue
        if type(network_dag.nodes[target].get("module")).__name__ != "Conv2d":
            continue
        if len(list(network_dag.successors(source))) != 1:
            continue
        if (source, target) in boot_edges:
            continue
        matches = _candidate_interval_matches(
            row=current,
            intervals=intervals,
            boot_edges=boot_edges,
        )
        if len(matches) != 1:
            continue
        join_rows = [
            dict(join_row)
            for join_row in compile_plan.get("edge_layouts", [])
            if str(join_row.get("target", "")) == source and str(join_row.get("op_kind", "")) == "concat"
        ]
        if len(join_rows) < 2:
            continue
        join_safe = True
        for join_row in join_rows:
            selected = dict(join_row.get("selected_layout", {}) or {})
            target_physical = str(join_row.get("target_physical_layout", join_row.get("physical_layout", "")) or "")
            if _layout_has_halo(selected) or target_physical != PHYSICAL_COMPACT:
                join_safe = False
                break
        if not join_safe:
            continue
        selected = dict(current.get("selected_layout", {}) or {})
        required = dict(current.get("required_layout", selected) or selected)
        if not _layout_covers(selected, required):
            continue
        if int(selected.get("gap", 1) or 1) != int(required.get("gap", selected.get("gap", 1)) or 1):
            continue
        node_row = node_rows_by_node.get(source)
        if node_row is None:
            continue
        updated_edge_rows: list[dict[str, Any]] = []
        for candidate_row in compile_plan.get("edge_layouts", []):
            if str(candidate_row.get("edge", "")) != edge_id:
                updated_edge_rows.append(dict(candidate_row))
                continue
            updated = _rewrite_row_as_compact_source(
                dict(candidate_row),
                source_layout=selected,
                target_layout=selected,
                reason="boot_interval_concat_output_beta_lift",
            )
            updated["layout_mode"] = "concat_output_compact_halo_shared"
            updated["concat_output_beta_lift"] = True
            updated["concat_output_beta_lift_source"] = str(source)
            updated_edge_rows.append(updated)
        updated_node_rows: list[dict[str, Any]] = []
        for candidate_node in compile_plan.get("node_layouts", []):
            if str(candidate_node.get("node", "")) != source:
                updated_node_rows.append(dict(candidate_node))
                continue
            updated_node_rows.append(
                _rewrite_node_as_beta_lift_producer(
                    dict(candidate_node),
                    layout=selected,
                    reason="boot_interval_concat_output_beta_lift",
                )
            )
        updated_plan = _refresh_layout_policy_plan_summary(
            {
                **dict(compile_plan),
                "edge_layouts": updated_edge_rows,
                "node_layouts": updated_node_rows,
            }
        )
        summary = dict(updated_plan.get("summary", {}) or {})
        summary["boot_refined_concat_output_beta_lift_count"] = (
            int(summary.get("boot_refined_concat_output_beta_lift_count", 0) or 0) + 1
        )
        updated_plan["summary"] = summary
        boot_source, boot_target = matches[0].get("boot_edge", ("", ""))
        candidates.append(
            {
                "kind": "concat_output_beta_lift",
                "strategy": "concat_output_beta_lift",
                "plan": updated_plan,
                "accepted": [
                    {
                        "kind": "concat_output_beta_lift",
                        "producer": str(source),
                        "boot_edge": {"source": str(boot_source), "target": str(boot_target)},
                        "producer_layout": dict(selected),
                        "covered_edges": [
                            {
                                "edge": str(edge_id),
                                "source": str(source),
                                "target": str(target),
                                "old_physical_layout": str(current.get("physical_layout", "")),
                                "new_physical_layout": PHYSICAL_LOGICAL_HALO,
                            }
                        ],
                        "covered_edge_count": 1,
                    }
                ],
                "rejected": [],
            }
        )
    return candidates


def _single_successor(network_dag: Any, node: str, expected: str) -> bool:
    try:
        successors = [str(value) for value in network_dag.successors(str(node))]
    except Exception:
        return False
    return successors == [str(expected)]


def _single_predecessor(network_dag: Any, node: str) -> str | None:
    try:
        predecessors = [str(value) for value in network_dag.predecessors(str(node))]
    except Exception:
        return None
    if len(predecessors) != 1:
        return None
    return str(predecessors[0])


def _native_conv_row_beta_lift_supported(row: dict[str, Any]) -> bool:
    if str(row.get("op_kind", "")) != "conv2d":
        return False
    if str(row.get("source", "")) == "x":
        return False
    if str(row.get("physical_layout", "")) != "native_source_stripe":
        return False
    selected = dict(row.get("selected_layout", {}) or {})
    required = dict(row.get("required_layout", selected) or selected)
    if not _layout_has_halo(selected):
        return False
    if not _layout_covers(selected, required):
        return False
    return int(selected.get("gap", 1) or 1) == int(required.get("gap", selected.get("gap", 1)) or 1)


def _boot_boundary_beta_lift_refinement_candidates(
    network_dag: Any,
    compile_plan: dict[str, Any],
    *,
    intervals: list[dict[str, Any]],
    boot_edges: set[tuple[str, str]],
    actual_native_physical_edges: set[str],
) -> list[dict[str, Any]]:
    edge_rows_by_edge = {str(row.get("edge", "")): dict(row) for row in compile_plan.get("edge_layouts", [])}
    node_rows_by_node = {str(row.get("node", "")): dict(row) for row in compile_plan.get("node_layouts", [])}
    candidates: list[dict[str, Any]] = []
    for row in compile_plan.get("edge_layouts", []):
        current = dict(row)
        edge_id = str(current.get("edge", ""))
        source = str(current.get("source", ""))
        target = str(current.get("target", ""))
        if edge_id not in actual_native_physical_edges:
            continue
        if not _native_conv_row_beta_lift_supported(current):
            continue
        if not source or not target or source not in network_dag.nodes or target not in network_dag.nodes:
            continue
        if type(network_dag.nodes[target].get("module")).__name__ != "Conv2d":
            continue

        direct_boot_edge = (source, target) in boot_edges
        interval_matches = _candidate_interval_matches(
            row=current,
            intervals=intervals,
            boot_edges=boot_edges,
        )
        selected = dict(current.get("selected_layout", {}) or {})
        producer = source
        producer_reason = "boot_boundary_beta_lift"
        carried_edge_ids: list[str] = []
        carried_edges: list[dict[str, Any]] = []
        source_module = network_dag.nodes[source].get("module")
        kind = "boot_boundary_beta_lift"
        source_boot_edge: tuple[str, str] | None = (source, target) if direct_boot_edge else None

        if type(source_module).__name__ in {"AvgPool2d", "Conv2d"}:
            if not direct_boot_edge:
                continue
            if not _single_successor(network_dag, source, target):
                continue
            if node_rows_by_node.get(producer) is None:
                continue
            kind = "boot_boundary_pool_direct_beta_lift" if type(source_module).__name__ == "AvgPool2d" else kind
        elif _layout_preserving_module(network_dag, source):
            predecessor = _single_predecessor(network_dag, source)
            if predecessor is None or predecessor not in network_dag.nodes:
                continue
            predecessor_module = network_dag.nodes[predecessor].get("module")
            if type(predecessor_module).__name__ != "Conv2d":
                continue
            if not _single_successor(network_dag, predecessor, source):
                continue
            if not _single_successor(network_dag, source, target):
                continue
            producer = str(predecessor)
            source_boot_edge = (producer, source) if (producer, source) in boot_edges else source_boot_edge
            if source_boot_edge is None and not interval_matches:
                continue
            producer_row = node_rows_by_node.get(producer)
            source_node_row = node_rows_by_node.get(source)
            activation_edge_id = f"{producer}->{source}"
            activation_edge = edge_rows_by_edge.get(activation_edge_id)
            if producer_row is None or source_node_row is None or activation_edge is None:
                continue
            carried_edge_ids.append(activation_edge_id)
            kind = "boot_boundary_activation_beta_lift"
        else:
            continue

        if source_boot_edge is None and interval_matches:
            source_boot_edge = tuple(interval_matches[0].get("boot_edge", ("", "")))  # type: ignore[arg-type]
        if source_boot_edge is None:
            continue

        updated_edge_rows: list[dict[str, Any]] = []
        for candidate_row in compile_plan.get("edge_layouts", []):
            candidate_edge_id = str(candidate_row.get("edge", ""))
            if candidate_edge_id == edge_id:
                updated = _rewrite_row_as_compact_source(
                    dict(candidate_row),
                    source_layout=selected,
                    target_layout=selected,
                    reason=producer_reason,
                )
                updated["boot_boundary_beta_lift"] = True
                updated["boot_boundary_beta_lift_source"] = str(producer)
                updated_edge_rows.append(updated)
            elif candidate_edge_id in carried_edge_ids:
                updated = _rewrite_row_as_compact_source(
                    dict(candidate_row),
                    source_layout=selected,
                    target_layout=selected,
                    reason="boot_boundary_activation_beta_lift",
                )
                carried_edges.append(
                    {
                        "edge": str(candidate_edge_id),
                        "source": str(candidate_row.get("source", "")),
                        "target": str(candidate_row.get("target", "")),
                        "op_kind": str(candidate_row.get("op_kind", "")),
                        "physical_layout": str(updated.get("physical_layout", "")),
                    }
                )
                updated_edge_rows.append(updated)
            else:
                updated_edge_rows.append(dict(candidate_row))

        updated_node_rows: list[dict[str, Any]] = []
        lift_nodes = {str(producer)}
        if carried_edge_ids:
            lift_nodes.add(str(source))
        for candidate_node in compile_plan.get("node_layouts", []):
            if str(candidate_node.get("node", "")) not in lift_nodes:
                updated_node_rows.append(dict(candidate_node))
                continue
            updated_node_rows.append(
                _rewrite_node_as_beta_lift_producer(
                    dict(candidate_node),
                    layout=selected,
                    reason=producer_reason,
                )
            )

        updated_plan = _refresh_layout_policy_plan_summary(
            {
                **dict(compile_plan),
                "edge_layouts": updated_edge_rows,
                "node_layouts": updated_node_rows,
            }
        )
        summary = dict(updated_plan.get("summary", {}) or {})
        summary["boot_refined_boot_boundary_beta_lift_count"] = (
            int(summary.get("boot_refined_boot_boundary_beta_lift_count", 0) or 0) + 1
        )
        updated_plan["summary"] = summary
        boot_source, boot_target = source_boot_edge
        candidates.append(
            {
                "kind": str(kind),
                "strategy": str(kind),
                "plan": updated_plan,
                "accepted": [
                    {
                        "kind": str(kind),
                        "producer": str(producer),
                        "boot_edge": {"source": str(boot_source), "target": str(boot_target)},
                        "producer_layout": dict(selected),
                        "covered_edges": [
                            {
                                "edge": str(edge_id),
                                "source": str(source),
                                "target": str(target),
                                "old_physical_layout": str(current.get("physical_layout", "")),
                                "new_physical_layout": PHYSICAL_LOGICAL_HALO,
                            }
                        ],
                        "carried_edges": carried_edges,
                        "covered_edge_count": 1,
                    }
                ],
                "rejected": [],
            }
        )
    return candidates


def enumerate_bootstrap_aware_layout_refinement_candidates(
    network_dag: Any,
    first_pass_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build boot-aware relayout refinement candidates without applying them.

    The DP layout policy remains rotation-only.  This pass only rewrites
    executable provider boundaries that lie inside a first-pass no-boot
    interval feeding a bootstrap.  A relayout's depth contribution is interval
    pressure, not edge-adjacent pressure: any true relayout between the previous
    bootstrap boundary and the current bootstrap boundary consumes the same
    level budget.
    """

    compile_plan = _layout_policy_compile_plan_for_dag(network_dag)
    if not isinstance(compile_plan, dict):
        return {"enabled": False, "reason": "no_layout_policy_executors"}
    if str(compile_plan.get("policy", "")) != "dp_no_share_fold":
        return {"enabled": False, "reason": "policy_not_dp_no_share_fold"}

    boot_edges = _bootstrap_edge_set(first_pass_audit)
    if not boot_edges:
        return {"enabled": False, "reason": "no_first_pass_boot_edges"}
    intervals = _bootstrap_upstream_intervals(network_dag, first_pass_audit)
    if not intervals:
        return {"enabled": False, "reason": "no_bootstrap_intervals"}

    previous_depths = _snapshot_layout_policy_depths(network_dag)
    actual_native_physical_edges = _actual_native_edge_ids(network_dag)
    boot_interval_rows = _boot_interval_audit_rows(intervals)

    candidates: list[dict[str, Any]] = []
    for interval in intervals:
        beta_lift = _try_apply_beta_lift_candidate(
            network_dag,
            compile_plan,
            interval,
            actual_native_edges=actual_native_physical_edges,
        )
        if beta_lift is None:
            continue
        accepted = [dict(row) for row in beta_lift.get("accepted", [])]
        candidate_kind = str(beta_lift.get("kind", "producer_beta_lift"))
        candidate_strategy = str(beta_lift.get("strategy", candidate_kind))
        candidates.append(
            {
                "candidate_id": f"bootstrap_refine_beta_lift_{len(candidates) + 1}",
                "kind": candidate_kind,
                "strategy": candidate_strategy,
                "plan": dict(beta_lift["plan"]),
                "accepted": accepted,
                "rejected": list(beta_lift.get("rejected", [])),
                "accepted_count": int(
                    sum(int(row.get("covered_edge_count", 1) or 1) for row in accepted)
                ),
                "rotation_delta": 0,
                "boot_interval_count": int(len(intervals)),
                "boot_intervals": boot_interval_rows,
            }
        )
    for concat_lift in _concat_output_beta_lift_refinement_candidates(
        network_dag,
        compile_plan,
        intervals=intervals,
        boot_edges=boot_edges,
        actual_native_physical_edges=actual_native_physical_edges,
    ):
        accepted = [dict(row) for row in concat_lift.get("accepted", [])]
        candidate_kind = str(concat_lift.get("kind", "concat_output_beta_lift"))
        candidate_strategy = str(concat_lift.get("strategy", candidate_kind))
        candidates.append(
            {
                "candidate_id": f"bootstrap_refine_concat_output_{len(candidates) + 1}",
                "kind": candidate_kind,
                "strategy": candidate_strategy,
                "plan": dict(concat_lift["plan"]),
                "accepted": accepted,
                "rejected": list(concat_lift.get("rejected", [])),
                "accepted_count": int(
                    sum(int(row.get("covered_edge_count", 1) or 1) for row in accepted)
                ),
                "rotation_delta": 0,
                "boot_interval_count": int(len(intervals)),
                "boot_intervals": boot_interval_rows,
            }
        )
    for boot_lift in _boot_boundary_beta_lift_refinement_candidates(
        network_dag,
        compile_plan,
        intervals=intervals,
        boot_edges=boot_edges,
        actual_native_physical_edges=actual_native_physical_edges,
    ):
        accepted = [dict(row) for row in boot_lift.get("accepted", [])]
        candidate_kind = str(boot_lift.get("kind", "boot_boundary_beta_lift"))
        candidate_strategy = str(boot_lift.get("strategy", candidate_kind))
        candidates.append(
            {
                "candidate_id": f"bootstrap_refine_boot_boundary_{len(candidates) + 1}",
                "kind": candidate_kind,
                "strategy": candidate_strategy,
                "plan": dict(boot_lift["plan"]),
                "accepted": accepted,
                "rejected": list(boot_lift.get("rejected", [])),
                "accepted_count": int(
                    sum(int(row.get("covered_edge_count", 1) or 1) for row in accepted)
                ),
                "rotation_delta": 0,
                "boot_interval_count": int(len(intervals)),
                "boot_intervals": boot_interval_rows,
            }
        )

    native_candidate = _native_physical_relayout_refinement_candidate(
        compile_plan,
        intervals=intervals,
        boot_edges=boot_edges,
        actual_native_physical_edges=actual_native_physical_edges,
    )
    fallback_rejected = list(native_candidate.get("rejected", []))
    if bool(native_candidate.get("enabled", False)):
        candidates.append(
            {
                "candidate_id": f"bootstrap_refine_native_relayout_{len(candidates) + 1}",
                "kind": str(native_candidate.get("kind", "native_physical_relayout")),
                "strategy": str(native_candidate.get("strategy", "native_physical_relayout")),
                "plan": dict(native_candidate["plan"]),
                "accepted": list(native_candidate.get("accepted", [])),
                "rejected": fallback_rejected,
                "accepted_count": int(native_candidate.get("accepted_count", 0) or 0),
                "rotation_delta": 0,
                "boot_interval_count": int(len(intervals)),
                "boot_intervals": boot_interval_rows,
            }
        )

    if not candidates:
        return {
            "enabled": False,
            "reason": str(native_candidate.get("reason", "no_supported_boot_interval_native_physical_relayout")),
            "rejected": fallback_rejected,
            "native_physical_relayout_edge_count": int(len(actual_native_physical_edges)),
            "boot_interval_count": int(len(intervals)),
            "boot_intervals": boot_interval_rows,
            "previous_compile_plan": compile_plan,
            "previous_depths": previous_depths,
        }

    return {
        "enabled": True,
        "policy": str(compile_plan.get("policy", "")),
        "first_pass_bootstrap_count": int(first_pass_audit.get("bootstrap_count", 0) or 0),
        "candidate_count": int(len(candidates)),
        "candidates": candidates,
        "rejected": fallback_rejected,
        "native_physical_relayout_edge_count": int(len(actual_native_physical_edges)),
        "boot_interval_count": int(len(intervals)),
        "boot_intervals": boot_interval_rows,
        "previous_compile_plan": compile_plan,
        "previous_depths": previous_depths,
    }


def apply_bootstrap_aware_layout_refinement_candidate(
    network_dag: Any,
    candidate: dict[str, Any],
    *,
    first_pass_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one enumerated boot-aware refinement candidate."""

    updated_plan = dict(candidate.get("plan", {}) or {})
    if not isinstance(updated_plan, dict) or not updated_plan:
        return {
            "enabled": False,
            "reason": "candidate_missing_plan",
            "candidate_id": str(candidate.get("candidate_id", "")),
        }
    compile_plan = _layout_policy_compile_plan_for_dag(network_dag)
    if not isinstance(compile_plan, dict):
        return {
            "enabled": False,
            "reason": "no_layout_policy_executors",
            "candidate_id": str(candidate.get("candidate_id", "")),
        }
    previous_depths = _snapshot_layout_policy_depths(network_dag)
    true_relayout_count_before = int(sum(int(row.get("relayout_depth", 0) or 0) for row in previous_depths))
    previous_module_layouts = _apply_layout_policy_module_output_layouts(network_dag, updated_plan)

    for binding in _iter_layout_policy_bindings(network_dag):
        executor = binding["executor"]
        update = getattr(executor, "update_layout_policy_compile_plan", None)
        if callable(update):
            update(updated_plan)
        else:
            executor.compile_plan = dict(updated_plan)
    depth_updates = _refresh_layout_policy_depths(
        network_dag,
        previous_depths=previous_depths,
    )
    true_relayout_count_after = int(
        sum(_executor_relayout_depth(binding["executor"]) for binding in _iter_layout_policy_bindings(network_dag))
    )
    accepted = [dict(row) for row in candidate.get("accepted", [])]
    audit = {
        "enabled": True,
        "policy": str(updated_plan.get("policy", "")),
        "candidate_id": str(candidate.get("candidate_id", "")),
        "kind": str(candidate.get("kind", candidate.get("strategy", ""))),
        "strategy": str(candidate.get("strategy", candidate.get("kind", ""))),
        "first_pass_bootstrap_count": int((first_pass_audit or {}).get("bootstrap_count", 0) or 0),
        "accepted": accepted,
        "rejected": list(candidate.get("rejected", [])),
        "accepted_count": int(candidate.get("accepted_count", len(accepted)) or 0),
        "rotation_delta": int(candidate.get("rotation_delta", 0) or 0),
        "depth_updates": depth_updates,
        "true_relayout_kernel_depth_before": int(true_relayout_count_before),
        "true_relayout_kernel_depth_after": int(true_relayout_count_after),
        "boot_interval_count": int(candidate.get("boot_interval_count", 0) or 0),
        "boot_intervals": list(candidate.get("boot_intervals", [])),
        "previous_compile_plan": compile_plan,
        "previous_depths": previous_depths,
        "_previous_module_layouts": previous_module_layouts,
    }
    network_dag.bootstrap_layout_refinement_audit = {
        key: value for key, value in dict(audit).items() if key != "_previous_module_layouts"
    }
    return audit


def apply_bootstrap_aware_layout_refinement(
    network_dag: Any,
    first_pass_audit: dict[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper that applies the first enumerated candidate."""

    enumeration = enumerate_bootstrap_aware_layout_refinement_candidates(
        network_dag,
        first_pass_audit,
    )
    if not bool(enumeration.get("enabled", False)):
        network_dag.bootstrap_layout_refinement_audit = dict(enumeration)
        return enumeration
    candidates = list(enumeration.get("candidates", []) or [])
    if not candidates:
        audit = {
            **dict(enumeration),
            "enabled": False,
            "reason": "no_supported_boot_interval_native_physical_relayout",
        }
        network_dag.bootstrap_layout_refinement_audit = dict(audit)
        return audit
    audit = apply_bootstrap_aware_layout_refinement_candidate(
        network_dag,
        dict(candidates[0]),
        first_pass_audit=first_pass_audit,
    )
    return {
        **dict(audit),
        "candidate_count": int(enumeration.get("candidate_count", len(candidates)) or len(candidates)),
    }


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
