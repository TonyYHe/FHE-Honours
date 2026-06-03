from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import Any

import torch

PHYSICAL_COMPACT = "packed_compact"
PHYSICAL_LOGICAL_HALO = "logical_halo_compact"
PHYSICAL_NATIVE_SOURCE_STRIPE = "native_source_stripe"
NATIVE_OUTPUT_MATERIALIZATION = "native_halo_stripe"
BOOTSTRAP_AWARE_LAYOUT_REFINEMENT_POLICIES = {
    "dp_no_share_fold",
    "fixed_max_no_share_fused",
    "always_no_share_fused",
}

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


def _ordered_fx_input_names(fx_node: Any) -> tuple[str, ...]:
    names: list[str] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        name = getattr(value, "name", None)
        if name is not None:
            names.append(str(name))
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(getattr(fx_node, "args", None))
    visit(getattr(fx_node, "kwargs", None))
    return tuple(names)


def _ordered_join_input_sources(
    network_dag: Any,
    *,
    node: str,
    row_sources: list[str],
) -> list[str]:
    row_source_set = set(row_sources)
    fx_node = network_dag.nodes[str(node)].get("fx_node") if str(node) in getattr(network_dag, "nodes", {}) else None
    fx_inputs = list(_ordered_fx_input_names(fx_node))
    if len(fx_inputs) == len(row_sources) and set(fx_inputs) == row_source_set:
        return list(fx_inputs)
    fx_all_inputs = [str(value.name) for value in getattr(fx_node, "all_input_nodes", ()) or ()]
    if len(fx_all_inputs) == len(row_sources) and set(fx_all_inputs) == row_source_set:
        return list(fx_all_inputs)
    try:
        graph_predecessors = [str(value) for value in network_dag.predecessors(str(node))]
    except Exception:
        graph_predecessors = []
    if len(graph_predecessors) == len(row_sources) and set(graph_predecessors) == row_source_set:
        return list(graph_predecessors)
    return list(row_sources)


def _env_truthy(name: str, default: str = "0") -> bool:
    return str(os.environ.get(str(name), str(default))).strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _bootstrap_halo0_channel_aligned_stripe_enabled() -> bool:
    return _env_truthy("ORION_LAYOUT_POLICY_HALO0_CHANNEL_ALIGNED_STRIPE", "1")


def _ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def _bsgs_hat_from_diagonal_count(diagonal_count: int, *, slots: int) -> dict[str, int]:
    diagonals = max(0, int(diagonal_count))
    if int(diagonals) == 0:
        return {"rotations": 0, "baby": 0, "giant": 0}
    best: dict[str, int] | None = None
    n1 = 1
    while int(n1) < max(2, int(slots)):
        baby = min(int(diagonals), max(1, int(n1) - 1))
        giant = _ceil_div(int(diagonals), max(1, int(n1)))
        rotations = int(baby + giant)
        candidate = {"rotations": int(rotations), "baby": int(baby), "giant": int(giant)}
        if best is None or int(candidate["rotations"]) < int(best["rotations"]):
            best = candidate
        n1 *= 2
    return dict(best or {"rotations": int(diagonals), "baby": int(diagonals), "giant": 0})


def _layout_top_beta(layout: dict[str, Any]) -> int:
    return max(0, int(dict(layout).get("top_beta", dict(layout).get("alpha", 0)) or 0))


def _layout_bottom_beta(layout: dict[str, Any]) -> int:
    return max(0, int(dict(layout).get("bottom_beta", dict(layout).get("beta", 0)) or 0))


def _layout_physical_top_beta(layout: dict[str, Any]) -> int:
    values = dict(layout)
    return max(0, int(values.get("physical_top_beta", values.get("top_beta", values.get("alpha", 0))) or 0))


def _layout_physical_bottom_beta(layout: dict[str, Any]) -> int:
    values = dict(layout)
    return max(0, int(values.get("physical_bottom_beta", values.get("bottom_beta", values.get("beta", 0))) or 0))


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


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return int(parsed) if int(parsed) > 0 else int(default)


def _native_output_ct_count_for_node(row: dict[str, Any], compile_plan: dict[str, Any]) -> int:
    node = str(row.get("node", ""))
    counts = dict(compile_plan.get("_native_physical_output_ct_counts", {}) or {})
    for key in (
        "native_physical_output_ct_count",
        "native_output_ct_count",
        "native_output_ct_count_estimate",
    ):
        value = _positive_int(row.get(key, 0), 0)
        if int(value) > 0:
            return int(value)
    return _positive_int(counts.get(node, 0), 0)


def _fhe_shape_for_output_row(
    row: dict[str, Any],
    layout: dict[str, Any],
    compile_plan: dict[str, Any],
) -> torch.Size | None:
    if str(row.get("physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE:
        ct_count = _native_output_ct_count_for_node(row, compile_plan)
        if int(ct_count) > 0:
            slots = max(1, int(compile_plan.get("slots", row.get("slots", 32768)) or 32768))
            return torch.Size((int(ct_count), int(slots)))
    return _fhe_shape_for_layout(row, layout)


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


def _layout_storage_key(layout: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    values = dict(layout)
    return (
        _layout_top_beta(values),
        _layout_bottom_beta(values),
        _layout_physical_top_beta(values),
        _layout_physical_bottom_beta(values),
        max(1, int(values.get("stride", 1) or 1)),
        max(1, int(values.get("gap", 1) or 1)),
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
    return str(plan.get("policy", "")) in BOOTSTRAP_AWARE_LAYOUT_REFINEMENT_POLICIES


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
                "layout_policy_native_output_target_signature": getattr(
                    module,
                    "layout_policy_native_output_target_signature",
                    _MISSING,
                ),
                "layout_policy_concat_input_source_signatures": getattr(
                    module,
                    "layout_policy_concat_input_source_signatures",
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
            "layout_policy_native_output_target_signature",
            "layout_policy_concat_input_source_signatures",
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
        should_apply = bool(
            _layout_has_halo(layout)
            or str(row.get("physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE
            or bool(row.get("boot_refined_halo0_channel_aligned_stripe", False))
        )
        if not bool(should_apply):
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
        shape = _fhe_shape_for_output_row(dict(row), layout, compile_plan)
        if shape is not None:
            module.fhe_output_shape = shape
        gap = max(1, int(layout.get("gap", 1) or 1))
        top_beta = _layout_physical_top_beta(layout)
        module.layout_policy_output_layout = dict(layout)
        module.layout_policy_output_row_offset = int(top_beta * gap)
        if str(row.get("physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE:
            module.layout_policy_output_materialization = NATIVE_OUTPUT_MATERIALIZATION
            target_signature = row.get("native_halo_target_storage_signature") or ()
            module.layout_policy_native_output_target_signature = [
                [int(value) for value in item] for item in target_signature
            ]
            if type(module).__name__ == "Concat":
                input_rows = [
                    dict(edge_row)
                    for edge_row in compile_plan.get("edge_layouts", [])
                    if str(edge_row.get("target", "")) == str(node)
                    and str(edge_row.get("op_kind", "")) == "concat"
                ]
                input_rows_by_source = {
                    str(edge_row.get("source", "")): dict(edge_row) for edge_row in input_rows
                }
                row_sources = [str(edge_row.get("source", "")) for edge_row in input_rows]
                input_sources = _ordered_join_input_sources(
                    network_dag,
                    node=str(node),
                    row_sources=row_sources,
                )
                input_signatures = [
                    [
                        [int(value) for value in item]
                        for item in input_rows_by_source.get(str(source), {}).get(
                            "native_halo_source_storage_signature",
                            (),
                        )
                        or ()
                    ]
                    for source in input_sources
                ]
                expected_signature_count = len(getattr(module, "concat_input_shapes", ()) or ())
                missing_signature_sources = [
                    str(source)
                    for source, signature in zip(input_sources, input_signatures, strict=False)
                    if not signature
                ]
                if int(expected_signature_count) > 0 and len(input_signatures) != int(expected_signature_count):
                    raise ValueError(
                        f"layout-policy Concat native materialization for {node} expected "
                        f"{int(expected_signature_count)} input signatures, got {len(input_signatures)} "
                        f"from sources {list(input_sources)}; available row sources={sorted(input_rows_by_source)}"
                    )
                if missing_signature_sources:
                    raise ValueError(
                        f"layout-policy Concat native materialization for {node} is missing "
                        f"native source signatures for {missing_signature_sources}; "
                        f"available row sources={sorted(input_rows_by_source)}"
                    )
                module.layout_policy_concat_input_source_signatures = input_signatures
        elif bool(row.get("producer_materialized_halo", False)) or _layout_has_halo(layout):
            module.layout_policy_output_materialization = "fused_relayout"
            for attr in (
                "layout_policy_native_output_target_signature",
                "layout_policy_concat_input_source_signatures",
            ):
                try:
                    delattr(module, attr)
                except AttributeError:
                    pass
        else:
            try:
                delattr(module, "layout_policy_output_materialization")
            except AttributeError:
                pass
            for attr in (
                "layout_policy_native_output_target_signature",
                "layout_policy_concat_input_source_signatures",
            ):
                try:
                    delattr(module, attr)
                except AttributeError:
                    pass
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
        if module is None or executor is None:
            continue
        previous = previous_by_executor.get(id(executor), {})
        old_runtime_depth = None
        if runtime is not None:
            old_runtime_depth = getattr(runtime, "solver_depth", None)
            if old_runtime_depth is None:
                old_runtime_depth = getattr(runtime, "depth", None)
        if old_runtime_depth is None:
            old_runtime_depth = previous.get("module_depth", getattr(module, "depth", None))
        if old_runtime_depth is None:
            continue
        old_relayout_depth = int(previous.get("relayout_depth", _executor_relayout_depth(executor)) or 0)
        base_depth = max(0, int(old_runtime_depth) - int(old_relayout_depth))
        new_relayout_depth = _executor_relayout_depth(executor)
        new_depth = int(base_depth + int(new_relayout_depth))
        if runtime is not None:
            runtime.depth = int(new_depth)
            runtime.solver_depth = int(new_depth)
        if hasattr(executor, "assigned_depth"):
            executor.assigned_depth = int(new_depth)
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
    node_layouts = [dict(row) for row in compile_plan.get("node_layouts", [])]
    native_output_counts = dict(compile_plan.get("_native_physical_output_ct_counts", {}) or {})
    native_output_nodes: set[str] = set()
    sanitized_node_layouts: list[dict[str, Any]] = []
    for row in node_layouts:
        updated = dict(row)
        node = str(updated.get("node", ""))
        if str(updated.get("physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE:
            if node:
                native_output_nodes.add(node)
        else:
            for key in (
                "native_physical_output_ct_count",
                "native_output_ct_count",
                "native_output_ct_count_estimate",
            ):
                updated.pop(key, None)
            if str(updated.get("layout_policy_output_materialization", "")) == NATIVE_OUTPUT_MATERIALIZATION:
                updated.pop("layout_policy_output_materialization", None)
        sanitized_node_layouts.append(updated)
    node_layouts = sanitized_node_layouts
    native_output_counts = {
        str(node): int(count)
        for node, count in native_output_counts.items()
        if str(node) in native_output_nodes and _positive_int(count, 0) > 0
    }
    relayout_edges = _rebuild_relayout_edges(edge_layouts)
    output_nodes = [
        {
            "node": str(row.get("node", "")),
            "reason": str(row.get("output_relayout_reason", "")),
            "selected_layout": dict(row.get("selected_layout", {}) or {}),
            "rotation_estimate": int(row.get("relayout_rotation_estimate", 0) or 0),
            "mask_mult_estimate": int(row.get("relayout_mask_mult_estimate", 0) or 0),
            "depth_estimate": int(row.get("relayout_depth_estimate", 0) or 0),
        }
        for row in node_layouts
        if bool(row.get("output_relayout", False)) and bool(row.get("selected_layout", {}))
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
    producer_nodes = [
        row
        for row in node_layouts
        if bool(row.get("producer_materialized_halo", False))
    ]
    summary["producer_fused_materialization_count"] = int(len(producer_nodes))
    summary["producer_fused_rotation_estimate"] = int(
        sum(int(row.get("producer_fused_rotation_estimate", 0) or 0) for row in producer_nodes)
    )
    summary["consumer_fused_relayout_count"] = int(
        sum(1 for row in edge_layouts if bool(row.get("consumer_fused_relayout", False)))
    )
    summary["consumer_fused_rotation_estimate"] = int(
        sum(int(row.get("consumer_fused_rotation_estimate", 0) or 0) for row in edge_layouts)
    )
    summary["planner_rotation_cost_estimate"] = int(
        sum(
            int(row.get("planner_rotation_cost_estimate", row.get("lt_bsgs_rotation_estimate", 0)) or 0)
            for row in edge_layouts
        )
    )
    if str(compile_plan.get("policy", "")) != "orion_dense":
        summary["reported_rotation_estimate"] = int(
            int(summary["planner_rotation_cost_estimate"])
            + int(summary["relayout_rotation_estimate"])
            + int(summary["producer_fused_rotation_estimate"])
        )
    return {
        **dict(compile_plan),
        "edge_layouts": edge_layouts,
        "node_layouts": node_layouts,
        "_native_physical_output_ct_counts": native_output_counts,
        "relayout_edges": relayout_edges,
        "relayout_edge_count": int(len(relayout_edges)),
        "output_relayout_nodes": output_nodes,
        "output_relayout_node_count": int(len(output_nodes)),
        "summary": summary,
    }


def _plan_rotation_cost(compile_plan: dict[str, Any]) -> int:
    summary = dict(compile_plan.get("summary", {}) or {})
    edge_layouts = [dict(row) for row in compile_plan.get("edge_layouts", [])]
    if not edge_layouts and "reported_rotation_estimate" in summary:
        return int(summary.get("reported_rotation_estimate", 0) or 0)
    relayout_edges = _rebuild_relayout_edges(edge_layouts)
    output_nodes = [
        dict(row)
        for row in compile_plan.get("output_relayout_nodes", [])
        if bool(row.get("selected_layout", {}))
    ]
    producer_nodes = [
        dict(row)
        for row in compile_plan.get("node_layouts", [])
        if bool(row.get("producer_materialized_halo", False))
    ]
    return int(
        sum(
            int(row.get("planner_rotation_cost_estimate", row.get("lt_bsgs_rotation_estimate", 0)) or 0)
            for row in edge_layouts
        )
        + sum(int(row.get("rotation_estimate", 0) or 0) for row in relayout_edges)
        + sum(int(row.get("rotation_estimate", 0) or 0) for row in output_nodes)
        + sum(int(row.get("producer_fused_rotation_estimate", 0) or 0) for row in producer_nodes)
    )


def _layout_stored_slots(layout: dict[str, Any]) -> int:
    return max(1, int(dict(layout).get("stored_slots", dict(layout).get("core_slots", 1)) or 1))


def _layout_tile_count(layout: dict[str, Any], *, slots: int = 32768) -> int:
    layout = dict(layout)
    if "tile_count" in layout:
        return max(1, int(layout.get("tile_count", 1) or 1))
    return max(1, _ceil_div(_layout_stored_slots(layout), max(1, int(slots or 32768))))


def _layout_tile_count_delta(
    old_layout: dict[str, Any],
    new_layout: dict[str, Any],
    *,
    slots: int = 32768,
) -> int:
    return int(
        max(
            0,
            _layout_tile_count(new_layout, slots=int(slots))
            - _layout_tile_count(old_layout, slots=int(slots)),
        )
    )


def _plan_output_tile_count_delta(
    old_plan: dict[str, Any],
    new_plan: dict[str, Any],
) -> int:
    slots = max(1, int(dict(old_plan).get("slots", dict(new_plan).get("slots", 32768)) or 32768))
    old_rows = {
        str(row.get("node", "")): dict(row)
        for row in dict(old_plan).get("node_layouts", [])
        if str(row.get("node", ""))
    }
    delta = 0
    for new_row in dict(new_plan).get("node_layouts", []):
        node = str(dict(new_row).get("node", ""))
        if not node or node not in old_rows:
            continue
        old_layout = dict(old_rows[node].get("selected_layout", {}) or {})
        new_layout = dict(dict(new_row).get("selected_layout", {}) or {})
        delta += _layout_tile_count_delta(old_layout, new_layout, slots=slots)
    return int(delta)


def _annotate_costless_output_lift_candidate(
    compile_plan: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    annotated = dict(candidate)
    plan = dict(annotated.get("plan", {}) or {})
    rotation_delta = int(
        annotated.get("rotation_delta", _candidate_rotation_delta(compile_plan, plan)) or 0
    )
    output_tile_delta = int(
        annotated.get(
            "output_tile_delta",
            _plan_output_tile_count_delta(compile_plan, plan),
        )
        or 0
    )
    annotated["rotation_delta"] = int(rotation_delta)
    annotated["output_tile_delta"] = int(output_tile_delta)
    if int(output_tile_delta) <= 0 and int(rotation_delta) <= 0:
        annotated["allow_relayout_depth_unchanged"] = True
        annotated["allow_after_bootstrap_target"] = True
        annotated["candidate_priority"] = min(
            int(annotated.get("candidate_priority", 0) or 0),
            -1,
        )
    return annotated


def _producer_layout_growth_rotation_delta(
    compile_plan: dict[str, Any],
    *,
    node: str,
    old_layout: dict[str, Any],
    new_layout: dict[str, Any],
) -> int:
    slots = max(1, int(compile_plan.get("slots", 32768) or 32768))
    old_tiles = _layout_tile_count(old_layout, slots=slots)
    new_tiles = _layout_tile_count(new_layout, slots=slots)
    tile_delta = int(new_tiles - old_tiles)
    if int(tile_delta) <= 0:
        return 0
    incoming_cost = int(
        sum(
            int(row.get("planner_rotation_cost_estimate", row.get("lt_bsgs_rotation_estimate", 0)) or 0)
            for row in compile_plan.get("edge_layouts", [])
            if str(row.get("target", "")) == str(node)
        )
    )
    if int(incoming_cost) <= 0:
        return int(tile_delta)
    return int(math.ceil(float(incoming_cost) * float(tile_delta) / float(max(1, int(old_tiles)))))


def _producer_node_layout_growth_rotation_delta(
    network_dag: Any,
    compile_plan: dict[str, Any],
    *,
    node: str,
    old_row: dict[str, Any],
    new_layout: dict[str, Any],
) -> int:
    module = network_dag.nodes[str(node)].get("module") if str(node) in network_dag.nodes else None
    if not _producer_beta_lift_allowed(module):
        return 0
    return _producer_layout_growth_rotation_delta(
        compile_plan,
        node=str(node),
        old_layout=dict(old_row.get("selected_layout", {}) or {}),
        new_layout=dict(new_layout),
    )


def _candidate_rotation_delta(
    previous_plan: dict[str, Any],
    updated_plan: dict[str, Any],
    *,
    extra_rotation_delta: int = 0,
) -> int:
    return int(
        _plan_rotation_cost(updated_plan)
        - _plan_rotation_cost(previous_plan)
        + int(extra_rotation_delta)
    )


def _compact_physical_for_refined_source(row: dict[str, Any]) -> str:
    source_physical = str(row.get("source_physical_layout", "") or "")
    if source_physical in {PHYSICAL_COMPACT, PHYSICAL_LOGICAL_HALO}:
        return str(source_physical)
    source_layout = dict(row.get("source_layout", {}) or {})
    return PHYSICAL_LOGICAL_HALO if _layout_has_halo(source_layout) else PHYSICAL_COMPACT


def _rewrite_native_physical_relayout_row_for_boot_refinement(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("op_kind", "")) != "conv2d":
        return None
    if _native_source_stripe_compact_rewrite_blocked(row):
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


def _pair_values(value: Any, default: int = 1) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if not value:
            return int(default), int(default)
        if len(value) == 1:
            single = int(value[0])
            return single, single
        return int(value[0]), int(value[1])
    scalar = int(value if value is not None else default)
    return scalar, scalar


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


def _native_source_stripe_compact_rewrite_blocked(row: dict[str, Any]) -> bool:
    if bool(row.get("concat_explicit_native_materialization", False)) or bool(
        row.get("concat_native_runtime_materializer", False)
    ):
        return True
    if str(row.get("op_kind", "")) == "concat" and (
        str(row.get("physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE
        or str(row.get("source_physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE
        or str(row.get("target_physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE
    ):
        return True
    return bool(
        str(row.get("op_kind", "")) == "conv2d"
        and str(row.get("physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE
        and str(row.get("provider_lt_grouping_mode", "")) == "individual"
        and str(row.get("native_halo_channel_fold_mode", "")) == "per_stripe"
    )


def _rewrite_row_as_compact_source(
    row: dict[str, Any],
    *,
    source_layout: dict[str, Any],
    target_layout: dict[str, Any],
    reason: str,
    allow_protected_native_source: bool = False,
) -> dict[str, Any] | None:
    if _native_source_stripe_compact_rewrite_blocked(row) and not bool(allow_protected_native_source):
        return None
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


def _native_input_ct_count_from_edge(row: dict[str, Any]) -> int:
    for key in (
        "native_input_ct_count",
        "native_input_ct_count_estimate",
        "native_output_ct_count",
        "native_output_ct_count_estimate",
    ):
        value = _positive_int(row.get(key, 0), 0)
        if int(value) > 0:
            return int(value)
    layout = dict(row.get("selected_layout", {}) or {})
    return _positive_int(layout.get("tile_count", 0), 0)


def _parse_storage_signature(raw: Any) -> tuple[tuple[int, int, int, int], ...]:
    rows: list[tuple[int, int, int, int]] = []
    for item in tuple(raw or ()):
        try:
            h_start, h_end, channel_start, channel_count = (
                int(value) for value in tuple(item)
            )
        except Exception:
            return ()
        if int(h_end) <= int(h_start) or int(channel_count) <= 0:
            continue
        rows.append((int(h_start), int(h_end), int(channel_start), int(channel_count)))
    return tuple(rows)


def _serialise_storage_signature(
    signature: tuple[tuple[int, int, int, int], ...],
) -> list[list[int]]:
    return [[int(value) for value in item] for item in signature]


def _concat_native_materialize_rotation_estimate(
    network_dag: Any,
    *,
    concat_node: str,
    input_rows: list[dict[str, Any]],
    target_signature: tuple[tuple[int, int, int, int], ...],
    slots: int,
) -> int:
    module = network_dag.nodes[str(concat_node)].get("module") if str(concat_node) in network_dag.nodes else None
    input_shapes = tuple(getattr(module, "concat_input_shapes", ()) or ()) if module is not None else ()
    if not input_shapes:
        return int(len(target_signature))
    rows_by_source = {str(row.get("source", "")): dict(row) for row in input_rows}
    input_sources = _ordered_join_input_sources(
        network_dag,
        node=str(concat_node),
        row_sources=[str(row.get("source", "")) for row in input_rows],
    )
    total = 0
    channel_offset = 0
    for input_index, source in enumerate(input_sources):
        row = rows_by_source.get(str(source), {})
        source_signature = _parse_storage_signature(row.get("native_halo_source_storage_signature") or ())
        if not source_signature:
            return 0
        if int(input_index) < len(input_shapes):
            try:
                branch_channels = int(tuple(input_shapes[int(input_index)])[1])
            except Exception:
                branch_channels = 0
        else:
            branch_channels = 0
        block_pair_diagonal_counts: dict[tuple[int, int], int] = {}
        for source_block, (source_h0, source_h1, source_c0, source_count) in enumerate(source_signature):
            source_global_c0 = int(channel_offset + int(source_c0))
            source_global_c1 = int(source_global_c0 + int(source_count))
            for target_block, (target_h0, target_h1, target_c0, target_count) in enumerate(target_signature):
                h0 = max(int(source_h0), int(target_h0))
                h1 = min(int(source_h1), int(target_h1))
                c0 = max(int(source_global_c0), int(target_c0))
                c1 = min(int(source_global_c1), int(target_c0 + target_count))
                if int(h1) <= int(h0) or int(c1) <= int(c0):
                    continue
                key = (int(target_block), int(source_block))
                # A native concat materializer is a sparse permutation.  Use
                # the overlapping channel count as a conservative diagonal-set
                # proxy; exact payload construction still happens at runtime.
                block_pair_diagonal_counts[key] = int(block_pair_diagonal_counts.get(key, 0) + max(1, int(c1 - c0)))
        for diagonal_count in block_pair_diagonal_counts.values():
            total += int(_bsgs_hat_from_diagonal_count(int(diagonal_count), slots=int(slots))["rotations"])
        channel_offset += int(branch_channels)
    return int(total)


def _pair_tuple(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    try:
        values = tuple(int(item) for item in tuple(value))
    except Exception:
        values = tuple(int(item) for item in default)
    if len(values) == 0:
        return tuple(int(item) for item in default)
    if len(values) == 1:
        return (int(values[0]), int(values[0]))
    return (int(values[0]), int(values[1]))


def _recompute_normal_native_conv_rotation_estimate(
    network_dag: Any,
    row: dict[str, Any],
    *,
    source_storage_signature: tuple[tuple[int, int, int, int], ...],
    target_storage_signature: tuple[tuple[int, int, int, int], ...],
    slots: int,
) -> dict[str, int] | None:
    """Recompute the non-fused native Conv2d cost for an explicit concat trial."""

    target = str(row.get("target", ""))
    module = network_dag.nodes.get(target, {}).get("module") if target in getattr(network_dag, "nodes", {}) else None
    input_shape = _shape_tuple(row.get("shape", ()) or getattr(module, "input_shape", ()))
    output_shape = _shape_tuple(getattr(module, "output_shape", ()) or row.get("output_shape", ()))
    if module is None or input_shape is None or output_shape is None:
        return None
    try:
        from orion.experimental.layout_policy_ablation import _native_conv_plan_for_layouts
    except Exception:
        return None

    kernel = _pair_tuple(getattr(module, "kernel_size", (1, 1)), (1, 1))
    stride = _pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))
    padding = _pair_tuple(getattr(module, "padding", (0, 0)), (0, 0))
    dilation = _pair_tuple(getattr(module, "dilation", (1, 1)), (1, 1))
    groups = max(1, int(getattr(module, "groups", 1) or 1))
    input_layout = dict(row.get("selected_layout", {}) or {})
    output_layout = dict(row.get("target_layout", {}) or getattr(module, "layout_policy_output_layout", {}) or {})
    if not input_layout or not output_layout:
        return None

    edge = SimpleNamespace(
        op_kind="conv2d",
        source=str(row.get("source", "")),
        target=str(target),
        shape=tuple(int(value) for value in input_shape),
        output_shape=tuple(int(value) for value in output_shape),
        output_fhe_shape=tuple(int(value) for value in getattr(module, "fhe_output_shape", ()) or ()),
        kernel_size=tuple(int(value) for value in kernel),
        stride=tuple(int(value) for value in stride),
        padding=tuple(int(value) for value in padding),
        dilation=tuple(int(value) for value in dilation),
        groups=int(groups),
        input_channels=int(input_shape[1]),
        output_channels=int(output_shape[1]),
        slots=int(slots),
    )
    plan = _native_conv_plan_for_layouts(
        edge,
        input_layout=dict(input_layout),
        output_layout=dict(output_layout),
        source_storage_signature=tuple(source_storage_signature),
        target_storage_signature=tuple(target_storage_signature) or None,
        require_native_target_fit=bool(target_storage_signature),
    )
    if plan is None:
        return None
    return {
        "rotations": int(getattr(plan, "c_only_rotations", 0) or 0),
        "cb_shared_rotations": int(getattr(plan, "cb_shared_rotations", 0) or 0),
        "transforms": int(getattr(plan, "submatrix_program_count", 0) or 0),
        "input_ct_count": int(getattr(plan, "input_ct_count", 0) or 0),
        "output_ct_count": int(sum(int(value) for value in getattr(plan, "target_channel_group_counts", ()) or ())),
        "sharing_group_count": int(getattr(plan, "sharing_group_count", 0) or 0),
    }


def _rewrite_row_as_native_source(
    row: dict[str, Any],
    *,
    source_layout: dict[str, Any],
    target_layout: dict[str, Any],
    native_ct_count: int,
    reason: str,
) -> dict[str, Any]:
    updated = dict(row)
    updated["source_layout"] = dict(source_layout)
    updated["selected_layout"] = dict(target_layout)
    updated["target_layout"] = dict(target_layout)
    updated["physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["source_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["target_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["relayout"] = False
    updated["relayout_reason"] = ""
    updated["relayout_rotation_estimate"] = 0
    updated["relayout_mask_mult_estimate"] = 0
    updated["relayout_sparse_lt_estimate"] = 0
    updated["relayout_depth_estimate"] = 0
    updated["consumer_fused_relayout"] = False
    updated["consumer_fused_rotation_estimate"] = 0
    updated["boot_refined_beta_lift"] = True
    updated["boot_refinement_reason"] = str(reason)
    if int(native_ct_count) > 0:
        updated["native_input_ct_count_estimate"] = int(native_ct_count)
        updated["native_source_output_ct_count"] = int(native_ct_count)
    if str(row.get("op_kind", "")) == "conv2d":
        updated["layout_mode"] = "native_halo_stripe"
        updated["provider_lt_grouping_mode"] = "individual"
        updated["native_halo_channel_fold_mode"] = "per_stripe"
    elif str(row.get("layout_mode", "")):
        updated["layout_mode"] = str(row.get("layout_mode", ""))
    else:
        updated["layout_mode"] = "layout_preserving_native_halo_stripe"
    return updated


def _rewrite_row_as_native_compact_source(
    row: dict[str, Any],
    *,
    source_layout: dict[str, Any],
    target_layout: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    physical = PHYSICAL_LOGICAL_HALO if _layout_has_halo(source_layout) else PHYSICAL_COMPACT
    updated = dict(row)
    updated["source_layout"] = dict(source_layout)
    updated["selected_layout"] = dict(target_layout)
    updated["target_layout"] = dict(target_layout)
    updated["physical_layout"] = str(physical)
    updated["source_physical_layout"] = str(physical)
    updated["target_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["relayout"] = False
    updated["relayout_reason"] = ""
    updated["relayout_rotation_estimate"] = 0
    updated["relayout_mask_mult_estimate"] = 0
    updated["relayout_sparse_lt_estimate"] = 0
    updated["relayout_depth_estimate"] = 0
    updated["consumer_fused_relayout"] = True
    updated["consumer_fused_rotation_estimate"] = 0
    updated["boot_refined_beta_lift"] = True
    updated["boot_refinement_reason"] = str(reason)
    updated["native_compact_source_fused"] = True
    updated["layout_mode"] = "native_halo_stripe"
    updated["provider_lt_grouping_mode"] = "individual"
    updated["native_halo_channel_fold_mode"] = "per_stripe"
    return updated


def _source_is_concat_like(source: str) -> bool:
    return str(source).startswith("cat")


def _row_is_concat_like_source(row: dict[str, Any], network_dag: Any) -> bool:
    source = str(row.get("source", "") or "")
    if _source_is_concat_like(source):
        return True
    if source in getattr(network_dag, "nodes", {}):
        if type(network_dag.nodes[source].get("module")).__name__ == "Concat":
            return True
    return bool(
        bool(row.get("concat_fusion_runtime_estimate", False))
        or str(row.get("lt_estimator", "")) == "concat_fused_module_unified_plan"
        or bool(row.get("concat_output_beta_lift", False))
        or bool(row.get("local_concat_transitive_beta_lift", False))
        or str(row.get("layout_mode", "")) == "concat_output_compact_halo_shared"
    )


def _conv2d_module_halo0_channel_aligned_native_stats(
    network_dag: Any,
    compile_plan: dict[str, Any],
    row: dict[str, Any],
    *,
    compact_layout: dict[str, Any],
    input_layout: dict[str, Any] | None = None,
    node_rows_by_node: dict[str, dict[str, Any]],
) -> dict[str, int] | None:
    target = str(row.get("target", ""))
    if not target or target not in network_dag.nodes:
        return None
    module = network_dag.nodes[target].get("module")
    if type(module).__name__ != "Conv2d":
        return None
    input_shape = _shape_tuple(row.get("shape", ()))
    output_row = node_rows_by_node.get(str(target), {})
    output_shape = _shape_tuple(output_row.get("shape", ()))
    if output_shape is None:
        try:
            output_shape = _shape_tuple(getattr(module, "output_shape"))
        except Exception:
            output_shape = None
    if input_shape is None or output_shape is None:
        return None
    kernel = _pair_values(getattr(module, "kernel_size", 1), 1)
    stride = _pair_values(getattr(module, "stride", 1), 1)
    padding = _pair_values(getattr(module, "padding", 0), 0)
    dilation = _pair_values(getattr(module, "dilation", 1), 1)
    groups = max(1, int(getattr(module, "groups", 1) or 1))
    if (
        int(groups) != 1
        or int(kernel[0]) != int(kernel[1])
        or int(stride[0]) != int(stride[1])
        or int(padding[0]) != int(padding[1])
        or int(dilation[0]) != int(dilation[1])
    ):
        return None
    try:
        from orion.experimental.cir.native_halo_conv2d import NativeHaloConv2DSpec, native_halo_conv2d_plan
    except Exception:
        return None
    stats_input_layout = dict(input_layout) if input_layout is not None else dict(compact_layout)
    output_layout = dict(output_row.get("selected_layout", {}) or {})
    output_gap = max(1, int(output_layout.get("gap", compact_layout.get("gap", 1)) or 1))
    slots = max(1, int(compile_plan.get("slots", row.get("slots", 32768)) or 32768))
    c_in = max(1, int(getattr(module, "in_channels", input_shape[1]) or input_shape[1]))
    c_out = max(1, int(getattr(module, "out_channels", output_shape[1]) or output_shape[1]))
    spec = NativeHaloConv2DSpec(
        family_label=(
            f"bootstrap_refine_halo0_channel_aligned_{row.get('source', '')}_{target}"
            f"_{int(c_in)}x{int(input_shape[2])}x{int(input_shape[3])}"
            f"_to_{int(c_out)}x{int(output_shape[2])}x{int(output_shape[3])}"
        ),
        c_in=int(c_in),
        h_in=int(input_shape[2]),
        w_in=int(input_shape[3]),
        c_out=int(c_out),
        h_out=int(output_shape[2]),
        w_out=int(output_shape[3]),
        gap_in=max(1, int(stats_input_layout.get("gap", compact_layout.get("gap", 1)) or 1)),
        gap_out=int(output_gap),
        kernel=int(kernel[0]),
        stride=int(stride[0]),
        pad=int(padding[0]),
        dilation=int(dilation[0]),
        groups=int(groups),
        slot_count=int(slots),
        input_top_beta=_layout_top_beta(stats_input_layout),
        input_bottom_beta=_layout_bottom_beta(stats_input_layout),
        output_top_beta=0,
        output_bottom_beta=0,
        input_physical_top_beta=_layout_physical_top_beta(stats_input_layout),
        input_physical_bottom_beta=_layout_physical_bottom_beta(stats_input_layout),
        output_physical_top_beta=0,
        output_physical_bottom_beta=0,
    )
    try:
        plan = native_halo_conv2d_plan(
            spec,
            require_native_target_fit=False,
            channel_fold_mode="per_stripe",
        )
    except ValueError:
        return None
    native_output_ct_count = int(
        sum(
            int(plan.target_group_count_for_stripe(stripe))
            for stripe in getattr(plan, "stripes", ()) or ()
        )
    )
    return {
        "rotations": int(plan.c_only_rotations),
        "cb_shared_rotations": int(plan.cb_shared_rotations),
        "baby_rotations": int(sum(int(value) for value in plan.program_rotation_counts)),
        "giant_rotations": int(plan.shared_giant_rotations),
        "transform_count": int(plan.submatrix_program_count),
        "input_ct_count": int(plan.input_ct_count),
        "output_ct_count": int(native_output_ct_count),
        "compact_output_ct_count": int(plan.output_ct_count),
        "stripe_count": int(getattr(plan, "stripe_count", 0) or 0),
    }


def _rewrite_row_as_halo0_channel_aligned_native_stripe(
    row: dict[str, Any],
    *,
    compact_layout: dict[str, Any],
    native_stats: dict[str, int],
    reason: str,
) -> dict[str, Any]:
    updated = dict(row)
    updated["source_layout"] = dict(compact_layout)
    updated["selected_layout"] = dict(compact_layout)
    updated["target_layout"] = dict(compact_layout)
    updated["physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["source_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["target_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["relayout"] = False
    updated["relayout_reason"] = ""
    updated["relayout_rotation_estimate"] = 0
    updated["relayout_mask_mult_estimate"] = 0
    updated["relayout_sparse_lt_estimate"] = 0
    updated["relayout_depth_estimate"] = 0
    updated["consumer_fused_relayout"] = False
    updated["consumer_fused_rotation_estimate"] = 0
    updated["layout_mode"] = "native_halo_stripe"
    updated["provider_lt_grouping_mode"] = "individual"
    updated["native_halo_channel_fold_mode"] = "per_stripe"
    updated["native_halo_channel_aligned_stripe"] = True
    updated["native_halo_input_storage_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["native_halo_output_storage_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["boot_refined_halo0_channel_aligned_stripe"] = True
    updated["boot_refinement_reason"] = str(reason)
    updated["boot_refinement_original_physical_layout"] = str(row.get("physical_layout", ""))
    rotations = int(native_stats.get("rotations", 0) or 0)
    updated["planner_rotation_cost_estimate"] = int(rotations)
    updated["lt_bsgs_rotation_estimate"] = int(rotations)
    updated["lt_baby_rotation_estimate"] = int(native_stats.get("baby_rotations", rotations) or rotations)
    updated["lt_giant_rotation_estimate"] = int(native_stats.get("giant_rotations", 0) or 0)
    updated["lt_transform_count_estimate"] = int(native_stats.get("transform_count", 0) or 0)
    updated["native_c_only_rotation_estimate"] = int(rotations)
    updated["native_cb_shared_rotation_estimate"] = int(native_stats.get("cb_shared_rotations", rotations) or rotations)
    updated["native_plan_c_only_rotation_estimate"] = int(rotations)
    updated["native_plan_cb_shared_rotation_estimate"] = int(native_stats.get("cb_shared_rotations", rotations) or rotations)
    updated["native_shared_baby_rotation_estimate"] = int(native_stats.get("baby_rotations", rotations) or rotations)
    updated["native_shared_giant_rotation_estimate"] = int(native_stats.get("giant_rotations", 0) or 0)
    updated["native_input_ct_count_estimate"] = int(native_stats.get("input_ct_count", 0) or 0)
    updated["native_input_ct_count"] = int(native_stats.get("input_ct_count", 0) or 0)
    updated["native_output_ct_count_estimate"] = int(native_stats.get("output_ct_count", 0) or 0)
    updated["native_output_ct_count"] = int(native_stats.get("output_ct_count", 0) or 0)
    updated["native_ct_count_estimate"] = int(native_stats.get("input_ct_count", 0) or 0)
    updated["native_ct_count"] = int(native_stats.get("input_ct_count", 0) or 0)
    updated["native_submatrix_program_count_estimate"] = int(native_stats.get("transform_count", 0) or 0)
    updated["native_stripe_count_estimate"] = int(native_stats.get("stripe_count", 0) or 0)
    updated["native_halo_rotation_estimator"] = "native_halo_conv2d_plan_boot_interval_halo0"
    updated["native_halo_rotation_mode"] = "c_only"
    updated["native_halo_rotation_exact_compact_output"] = False
    updated["native_halo_rotation_search_surrogate"] = ""
    return updated


def _rewrite_row_as_halo0_channel_aligned_native_output(
    row: dict[str, Any],
    *,
    native_stats: dict[str, int],
    reason: str,
) -> dict[str, Any]:
    """Keep the native-source input layout and make only the Conv output beta0 native."""

    updated = dict(row)
    updated["physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["source_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["target_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["relayout"] = False
    updated["relayout_reason"] = ""
    updated["relayout_rotation_estimate"] = 0
    updated["relayout_mask_mult_estimate"] = 0
    updated["relayout_sparse_lt_estimate"] = 0
    updated["relayout_depth_estimate"] = 0
    updated["consumer_fused_relayout"] = False
    updated["consumer_fused_rotation_estimate"] = 0
    updated["layout_mode"] = "native_halo_stripe"
    updated["provider_lt_grouping_mode"] = "individual"
    updated["native_halo_channel_fold_mode"] = "per_stripe"
    updated["native_halo_input_storage_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["native_halo_output_storage_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["boot_refined_halo0_channel_aligned_stripe"] = True
    updated["boot_refined_halo0_native_output_only"] = True
    updated["boot_refinement_reason"] = str(reason)
    updated["boot_refinement_original_output_storage_layout"] = str(
        row.get("native_halo_output_storage_layout", "")
    )
    rotations = int(native_stats.get("rotations", 0) or 0)
    updated["planner_rotation_cost_estimate"] = int(rotations)
    updated["lt_bsgs_rotation_estimate"] = int(rotations)
    updated["lt_baby_rotation_estimate"] = int(native_stats.get("baby_rotations", rotations) or rotations)
    updated["lt_giant_rotation_estimate"] = int(native_stats.get("giant_rotations", 0) or 0)
    updated["lt_transform_count_estimate"] = int(native_stats.get("transform_count", 0) or 0)
    updated["native_c_only_rotation_estimate"] = int(rotations)
    updated["native_cb_shared_rotation_estimate"] = int(native_stats.get("cb_shared_rotations", rotations) or rotations)
    updated["native_plan_c_only_rotation_estimate"] = int(rotations)
    updated["native_plan_cb_shared_rotation_estimate"] = int(native_stats.get("cb_shared_rotations", rotations) or rotations)
    updated["native_shared_baby_rotation_estimate"] = int(native_stats.get("baby_rotations", rotations) or rotations)
    updated["native_shared_giant_rotation_estimate"] = int(native_stats.get("giant_rotations", 0) or 0)
    updated["native_input_ct_count_estimate"] = int(native_stats.get("input_ct_count", 0) or 0)
    updated["native_input_ct_count"] = int(native_stats.get("input_ct_count", 0) or 0)
    updated["native_output_ct_count_estimate"] = int(native_stats.get("output_ct_count", 0) or 0)
    updated["native_output_ct_count"] = int(native_stats.get("output_ct_count", 0) or 0)
    updated["native_ct_count_estimate"] = int(native_stats.get("input_ct_count", 0) or 0)
    updated["native_ct_count"] = int(native_stats.get("input_ct_count", 0) or 0)
    updated["native_submatrix_program_count_estimate"] = int(native_stats.get("transform_count", 0) or 0)
    updated["native_stripe_count_estimate"] = int(native_stats.get("stripe_count", 0) or 0)
    updated["native_halo_rotation_estimator"] = "native_halo_conv2d_plan_boot_boundary_output_only"
    updated["native_halo_rotation_mode"] = "c_only"
    updated["native_halo_rotation_exact_compact_output"] = False
    updated["native_halo_rotation_search_surrogate"] = ""
    return updated


def _rewrite_node_as_halo0_boot_boundary_output(
    row: dict[str, Any],
    *,
    compact_layout: dict[str, Any],
    native_ct_count: int,
    slots: int,
    reason: str,
) -> dict[str, Any]:
    updated = dict(row)
    updated["selected_layout"] = dict(compact_layout)
    updated["physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["output_relayout"] = False
    updated["output_relayout_reason"] = ""
    updated["producer_materialized_halo"] = False
    updated["producer_materialized_halo_reason"] = ""
    updated["layout_policy_output_materialization"] = NATIVE_OUTPUT_MATERIALIZATION
    updated["native_halo_output_storage_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["boot_refined_halo0_channel_aligned_stripe"] = True
    updated["boot_refinement_reason"] = str(reason)
    if int(native_ct_count) > 0:
        updated["native_physical_output_ct_count"] = int(native_ct_count)
        updated["native_output_ct_count"] = int(native_ct_count)
        updated["native_output_ct_count_estimate"] = int(native_ct_count)
        updated["fhe_shape"] = [int(native_ct_count), max(1, int(slots))]
    return updated


def _rewrite_layout_preserving_edge_as_halo0_compact(
    row: dict[str, Any],
    *,
    compact_layout: dict[str, Any],
    native_ct_count: int = 0,
    reason: str,
) -> dict[str, Any]:
    updated = dict(row)
    updated["source_layout"] = dict(compact_layout)
    updated["selected_layout"] = dict(compact_layout)
    updated["target_layout"] = dict(compact_layout)
    updated["physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["source_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["target_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["relayout"] = False
    updated["relayout_reason"] = ""
    updated["relayout_rotation_estimate"] = 0
    updated["relayout_mask_mult_estimate"] = 0
    updated["relayout_sparse_lt_estimate"] = 0
    updated["relayout_depth_estimate"] = 0
    updated["consumer_fused_relayout"] = False
    updated["consumer_fused_rotation_estimate"] = 0
    updated["boot_refined_halo0_channel_aligned_stripe"] = True
    updated["boot_refinement_reason"] = str(reason)
    if int(native_ct_count) > 0:
        updated["native_input_ct_count"] = int(native_ct_count)
        updated["native_input_ct_count_estimate"] = int(native_ct_count)
        updated["native_ct_count"] = int(native_ct_count)
        updated["native_ct_count_estimate"] = int(native_ct_count)
    return updated


def _row_is_native_source_stripe_relayout(row: dict[str, Any]) -> bool:
    selected = dict(row.get("selected_layout", {}) or {})
    required = dict(row.get("required_layout", selected) or selected)
    reason = str(row.get("relayout_reason", "") or "")
    source_physical = row.get("source_physical_layout", None)
    return bool(
        str(row.get("op_kind", "")) == "conv2d"
        and str(row.get("source", "")) != "x"
        and str(row.get("physical_layout", "")) == "native_source_stripe"
        and str(row.get("target_physical_layout", row.get("physical_layout", "")) or "") == "native_source_stripe"
        and source_physical is not None
        and str(source_physical or "") != "native_source_stripe"
        and _layout_has_halo(selected)
        and _layout_covers(selected, required)
        and int(selected.get("gap", 1) or 1) == int(required.get("gap", selected.get("gap", 1)) or 1)
        and ("native_source_stripe_relayout" in reason or "physical_source_stripe_relayout" in reason)
        and str(row.get("provider_lt_grouping_mode", "")) == "individual"
        and str(row.get("native_halo_channel_fold_mode", "")) == "per_stripe"
    )


def _row_is_already_native_source_boot_boundary(row: dict[str, Any]) -> bool:
    selected = dict(row.get("selected_layout", {}) or {})
    required = dict(row.get("required_layout", selected) or selected)
    return bool(
        str(row.get("op_kind", "")) == "conv2d"
        and str(row.get("source", "")) != "x"
        and str(row.get("layout_mode", "")) == "native_halo_stripe"
        and str(row.get("physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE
        and str(row.get("source_physical_layout", "") or "") == PHYSICAL_NATIVE_SOURCE_STRIPE
        and str(row.get("target_physical_layout", "") or "") == PHYSICAL_NATIVE_SOURCE_STRIPE
        and _layout_covers(selected, required)
        and int(selected.get("gap", 1) or 1) == int(required.get("gap", selected.get("gap", 1)) or 1)
        and str(row.get("provider_lt_grouping_mode", "")) == "individual"
        and str(row.get("native_halo_channel_fold_mode", "")) == "per_stripe"
    )


def _bootstrap_boundary_halo0_candidate_edges(
    network_dag: Any,
    edge_rows_by_edge: dict[str, dict[str, Any]],
    boot_edges: set[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Map conv input edges that immediately produce a booted source."""

    candidate_edges: dict[str, list[tuple[str, str]]] = {}
    for boot_source, boot_target in sorted((str(source), str(target)) for source, target in boot_edges):
        if not boot_source or boot_source not in getattr(network_dag, "nodes", {}):
            continue
        producer = str(boot_source)
        if _layout_preserving_module(network_dag, producer):
            predecessors = [str(value) for value in network_dag.predecessors(producer)]
            if len(predecessors) != 1:
                continue
            upstream = str(predecessors[0])
            if upstream not in getattr(network_dag, "nodes", {}):
                continue
            upstream_successors = [str(value) for value in network_dag.successors(upstream)]
            if len(upstream_successors) != 1 or set(upstream_successors) != {producer}:
                continue
            boot_source_successors = [str(value) for value in network_dag.successors(producer)]
            if any((producer, successor) not in boot_edges for successor in boot_source_successors):
                continue
            producer = upstream
        producer_module = network_dag.nodes[producer].get("module") if producer in network_dag.nodes else None
        if type(producer_module).__name__ != "Conv2d":
            continue
        producer_successors = [str(value) for value in network_dag.successors(producer)]
        if boot_source == producer:
            if any((producer, successor) not in boot_edges for successor in producer_successors):
                continue
        else:
            if len(producer_successors) != 1 or set(producer_successors) != {boot_source}:
                continue
        predecessors = [str(value) for value in network_dag.predecessors(producer)]
        if len(predecessors) != 1:
            continue
        edge_id = f"{predecessors[0]}->{producer}"
        row = edge_rows_by_edge.get(edge_id)
        if row is None or str(row.get("op_kind", "")) != "conv2d":
            continue
        candidate_edges.setdefault(edge_id, []).append((str(boot_source), str(boot_target)))
    return candidate_edges


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
    for key in (
        "native_physical_output_ct_count",
        "native_output_ct_count",
        "native_output_ct_count_estimate",
    ):
        updated.pop(key, None)
    if str(updated.get("layout_policy_output_materialization", "")) == NATIVE_OUTPUT_MATERIALIZATION:
        updated.pop("layout_policy_output_materialization", None)
    shape = _fhe_shape_for_layout(updated, layout)
    if shape is not None:
        updated["fhe_shape"] = [int(value) for value in shape]
    return updated


def _rewrite_node_as_native_output_producer(
    row: dict[str, Any],
    *,
    layout: dict[str, Any],
    native_ct_count: int,
    slots: int,
    reason: str,
) -> dict[str, Any]:
    updated = dict(row)
    updated["selected_layout"] = dict(layout)
    updated["physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
    updated["output_relayout"] = False
    updated["output_relayout_reason"] = ""
    updated["producer_materialized_halo"] = bool(_layout_has_halo(layout))
    updated["producer_materialized_halo_reason"] = str(reason) if _layout_has_halo(layout) else ""
    updated["producer_fused_rotation_estimate"] = 0
    updated["producer_fused_depth_estimate"] = 0
    updated["layout_policy_output_materialization"] = NATIVE_OUTPUT_MATERIALIZATION
    if int(native_ct_count) > 0:
        updated["native_physical_output_ct_count"] = int(native_ct_count)
        updated["fhe_shape"] = [int(native_ct_count), max(1, int(slots))]
    else:
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


def _conv2d_unit_stride_radius_one(module: Any | None) -> bool:
    if type(module).__name__ != "Conv2d":
        return False
    stride_h, stride_w = _pair_values(getattr(module, "stride", 1), 1)
    dilation_h, dilation_w = _pair_values(getattr(module, "dilation", 1), 1)
    kernel_h, kernel_w = _pair_values(getattr(module, "kernel_size", 1), 1)
    padding_h, padding_w = _pair_values(getattr(module, "padding", 0), 0)
    return bool(
        int(stride_h) == 1
        and int(stride_w) == 1
        and int(dilation_h) == 1
        and int(dilation_w) == 1
        and int(kernel_h) == 3
        and int(kernel_w) == 3
        and int(padding_h) == 1
        and int(padding_w) == 1
    )


def _producer_transitive_conv_conv_beta_lift(
    network_dag: Any,
    *,
    path: list[str],
    path_edges: list[tuple[str, str]],
    edge_rows_by_edge: dict[str, dict[str, Any]],
    start_index: int,
    producer_module: Any | None,
    lifted_edges: list[tuple[str, str]],
    producer_layout: dict[str, Any],
) -> bool:
    if type(producer_module).__name__ not in {"AvgPool2d", "ConvTranspose2d"}:
        return False
    if len(lifted_edges) != 2:
        return False
    indexed_edges: list[tuple[int, tuple[str, str]]] = []
    for edge in lifted_edges:
        native_edge = tuple(edge)
        try:
            native_edge_index = path_edges.index(native_edge)
        except ValueError:
            return False
        indexed_edges.append((int(native_edge_index), native_edge))
    indexed_edges.sort(key=lambda item: item[0])
    first_index, first_edge = indexed_edges[0]
    second_index, second_edge = indexed_edges[1]
    if int(first_index) != int(start_index):
        return False
    if int(second_index) <= int(first_index):
        return False
    first_consumer = str(first_edge[1])
    final_consumer = str(second_edge[1])
    middle_module = (
        network_dag.nodes[str(first_consumer)].get("module") if str(first_consumer) in network_dag.nodes else None
    )
    consumer_module = (
        network_dag.nodes[str(final_consumer)].get("module") if str(final_consumer) in network_dag.nodes else None
    )
    if not _conv2d_unit_stride_radius_one(middle_module):
        return False
    if not _conv2d_unit_stride_radius_one(consumer_module):
        return False
    carried_edge_ids: list[str] = []
    for edge_index in range(int(first_index) + 1, int(second_index)):
        carried_edge = path_edges[int(edge_index)]
        carried_edge_id = f"{carried_edge[0]}->{carried_edge[1]}"
        carried_row = edge_rows_by_edge.get(carried_edge_id)
        if carried_row is None:
            return False
        intermediate_target = str(path_edges[int(edge_index)][1])
        if not _layout_preserving_module(network_dag, intermediate_target):
            return False
        carried_edge_ids.append(str(carried_edge_id))

    first_edge_id = f"{first_edge[0]}->{first_edge[1]}"
    second_edge_id = f"{second_edge[0]}->{second_edge[1]}"
    first_row = edge_rows_by_edge.get(first_edge_id)
    second_row = edge_rows_by_edge.get(second_edge_id)
    if first_row is None or second_row is None:
        return False
    first_selected = dict(first_row.get("selected_layout", {}) or {})
    second_selected = dict(second_row.get("selected_layout", {}) or {})
    first_gap = int(first_selected.get("gap", 1) or 1)
    second_gap = int(second_selected.get("gap", 1) or 1)
    producer_gap = int(dict(producer_layout).get("gap", 1) or 1)
    if int(first_gap) != int(second_gap) or int(producer_gap) != int(second_gap):
        return False
    for carried_edge_id in carried_edge_ids:
        carried_row = edge_rows_by_edge.get(str(carried_edge_id))
        carried_selected = dict(carried_row.get("selected_layout", {}) or {}) if carried_row is not None else {}
        carried_required = dict(carried_row.get("required_layout", carried_selected) or carried_selected)
        if int(carried_selected.get("gap", 1) or 1) != int(second_gap):
            return False
        if int(carried_required.get("gap", carried_selected.get("gap", 1)) or 1) != int(second_gap):
            return False
        if not _layout_covers(second_selected, carried_required):
            return False
    if _layout_top_beta(producer_layout) != _layout_top_beta(second_selected) + 1:
        return False
    if _layout_bottom_beta(producer_layout) != _layout_bottom_beta(second_selected) + 1:
        return False

    middle_layout = _semantic_output_layout(
        middle_module,
        dict(second_row),
        dict(producer_layout),
    )
    if _layout_top_beta(middle_layout) != _layout_top_beta(second_selected):
        return False
    if _layout_bottom_beta(middle_layout) != _layout_bottom_beta(second_selected):
        return False
    return True


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
        activation_transparent_single_native = False
        pool_direct_single_native = False
        producer_transitive_conv_conv = False
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
            producer_transitive_conv_conv = _producer_transitive_conv_conv_beta_lift(
                network_dag,
                path=path,
                path_edges=path_edges,
                edge_rows_by_edge=edge_rows_by_edge,
                start_index=int(start_index),
                producer_module=producer_module,
                lifted_edges=lifted_edges,
                producer_layout=dict(current_source_layout),
            )
            if type(producer_module).__name__ in {"AvgPool2d", "ConvTranspose2d"} and not producer_transitive_conv_conv:
                continue
        if not _layout_has_halo(current_source_layout):
            continue
        allow_protected_compact_rewrite = bool(pool_direct_single_native or producer_transitive_conv_conv)
        if bool(producer_transitive_conv_conv):
            indexed_lifted_edges: list[tuple[int, tuple[str, str]]] = []
            for lifted_edge in lifted_edges:
                try:
                    lifted_index = path_edges.index(tuple(lifted_edge))
                except ValueError:
                    continue
                indexed_lifted_edges.append((int(lifted_index), tuple(lifted_edge)))
            indexed_lifted_edges.sort(key=lambda item: int(item[0]))
            if len(indexed_lifted_edges) < 2:
                continue
        updated_edge_rows = []
        live_layout = dict(current_source_layout)
        accepted_edges: list[dict[str, Any]] = []
        carried_edges: list[dict[str, Any]] = []
        rewrite_failed = False
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
                        reason=(
                            "boot_interval_transitive_beta_lift"
                            if bool(producer_transitive_conv_conv)
                            else "boot_interval_beta_lift"
                        ),
                        allow_protected_native_source=bool(allow_protected_compact_rewrite),
                    ) if not bool(allow_protected_compact_rewrite) else _rewrite_row_as_native_source(
                        dict(row),
                        source_layout=input_layout,
                        target_layout=input_layout,
                        native_ct_count=_native_input_ct_count_from_edge(dict(row)),
                        reason=(
                            "boot_interval_transitive_beta_lift"
                            if bool(producer_transitive_conv_conv)
                            else "boot_interval_beta_lift"
                        ),
                    )
                    if updated is None:
                        rewrite_failed = True
                        break
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
                    required = dict(row.get("required_layout", row.get("selected_layout", {})) or {})
                    if required and not _layout_covers(input_layout, required):
                        accepted_edges = []
                        break
                    updated = _rewrite_row_as_compact_source(
                        dict(row),
                        source_layout=input_layout,
                        target_layout=input_layout,
                        reason=(
                            "boot_interval_transitive_activation_carry"
                            if bool(producer_transitive_conv_conv)
                            else "boot_interval_activation_transparent_beta_lift"
                        ),
                        allow_protected_native_source=bool(allow_protected_compact_rewrite),
                    ) if not bool(allow_protected_compact_rewrite) else _rewrite_row_as_native_source(
                        dict(row),
                        source_layout=input_layout,
                        target_layout=input_layout,
                        native_ct_count=_native_input_ct_count_from_edge(dict(row)),
                        reason=(
                            "boot_interval_transitive_activation_carry"
                            if bool(producer_transitive_conv_conv)
                            else "boot_interval_activation_transparent_beta_lift"
                        ),
                    )
                    if updated is None:
                        rewrite_failed = True
                        break
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
        if rewrite_failed:
            continue
        candidate_kind = (
            "activation_transparent_beta_lift"
            if bool(activation_transparent_single_native)
            else "pool_direct_beta_lift"
            if bool(pool_direct_single_native)
            else "mvm_transitive_beta_lift"
            if bool(producer_transitive_conv_conv)
            else "producer_beta_lift"
        )
        protected_native_lift = bool(pool_direct_single_native or producer_transitive_conv_conv)
        native_count_by_source: dict[str, int] = {}
        if bool(protected_native_lift):
            for updated_edge in updated_edge_rows:
                if str(updated_edge.get("physical_layout", "")) != PHYSICAL_NATIVE_SOURCE_STRIPE:
                    continue
                count = _native_input_ct_count_from_edge(dict(updated_edge))
                if int(count) <= 0:
                    continue
                source_node = str(updated_edge.get("source", ""))
                if source_node:
                    native_count_by_source[str(source_node)] = int(count)
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
            producer_growth_delta = _producer_node_layout_growth_rotation_delta(
                network_dag,
                compile_plan,
                node=str(node),
                old_row=dict(row),
                new_layout=dict(layout),
            )
            if bool(protected_native_lift) and int(native_count_by_source.get(str(node), 0) or 0) > 0:
                updated_node = _rewrite_node_as_native_output_producer(
                    dict(row),
                    layout=layout,
                    native_ct_count=int(native_count_by_source[str(node)]),
                    slots=int(plan_slots),
                    reason=str(candidate_kind),
                )
            else:
                updated_node = _rewrite_node_as_beta_lift_producer(
                    dict(row),
                    layout=layout,
                    reason=(
                        "boot_interval_transitive_beta_lift"
                        if bool(producer_transitive_conv_conv)
                        else "boot_interval_beta_lift"
                    ),
                )
            if int(producer_growth_delta) > 0:
                updated_node["producer_fused_rotation_estimate"] = int(
                    int(row.get("producer_fused_rotation_estimate", 0) or 0)
                    + int(producer_growth_delta)
                )
            updated_node_rows.append(updated_node)
        if len(accepted_edges) < 2 and not activation_transparent_single_native:
            if not pool_direct_single_native:
                continue
        native_output_counts = dict(compile_plan.get("_native_physical_output_ct_counts", {}) or {})
        if bool(protected_native_lift):
            for node, count in native_count_by_source.items():
                if str(node) in node_output_updates and int(count) > 0:
                    native_output_counts[str(node)] = int(count)
        if bool(producer_transitive_conv_conv):
            for node in node_output_updates:
                if str(node) != str(producer):
                    native_output_counts.pop(str(node), None)
        updated_plan = _refresh_layout_policy_plan_summary(
            {
                **dict(compile_plan),
                "edge_layouts": updated_edge_rows,
                "node_layouts": updated_node_rows,
                "_native_physical_output_ct_counts": native_output_counts,
            }
        )
        summary = dict(updated_plan.get("summary", {}) or {})
        summary["boot_refined_beta_lift_count"] = int(summary.get("boot_refined_beta_lift_count", 0) or 0) + 1
        updated_plan["summary"] = summary
        result = {
            "kind": candidate_kind,
            "strategy": candidate_kind,
            "plan": updated_plan,
            "rotation_delta": _candidate_rotation_delta(compile_plan, updated_plan),
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
        if bool(producer_transitive_conv_conv):
            result["candidate_priority"] = 2
            result["require_bootstrap_count_unchanged"] = False
            result["require_bootstrap_shape_nonincrease"] = False
        return result
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
        "rotation_delta": _candidate_rotation_delta(compile_plan, updated_plan),
        "accepted": accepted,
        "rejected": rejected,
        "accepted_count": int(len(accepted)),
        "native_physical_relayout_edge_count": int(len(actual_native_physical_edges)),
        "boot_interval_count": int(len(intervals)),
    }


def _halo0_channel_aligned_stripe_refinement_candidates(
    network_dag: Any,
    compile_plan: dict[str, Any],
    *,
    intervals: list[dict[str, Any]],
    boot_edges: set[tuple[str, str]],
    actual_native_physical_edges: set[str],
) -> list[dict[str, Any]]:
    if not _bootstrap_halo0_channel_aligned_stripe_enabled():
        return []

    edge_rows_by_edge = {str(row.get("edge", "")): dict(row) for row in compile_plan.get("edge_layouts", [])}
    node_rows_by_node = {str(row.get("node", "")): dict(row) for row in compile_plan.get("node_layouts", [])}
    boot_boundary_candidate_edges = _bootstrap_boundary_halo0_candidate_edges(
        network_dag,
        edge_rows_by_edge,
        boot_edges,
    )
    updated_by_edge: dict[str, dict[str, Any]] = {}
    updated_by_node: dict[str, dict[str, Any]] = {}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for row in compile_plan.get("edge_layouts", []):
        current = dict(row)
        edge_id = str(current.get("edge", ""))
        source = str(current.get("source", ""))
        target = str(current.get("target", ""))
        already_native_candidate = bool(_row_is_already_native_source_boot_boundary(current))
        if not bool(already_native_candidate):
            continue
        boundary_boot_edges = list(boot_boundary_candidate_edges.get(edge_id, []) or [])
        if not boundary_boot_edges:
            rejected.append(
                {
                    "edge": edge_id,
                    "source": source,
                    "target": target,
                    "reason": "not_immediate_boot_boundary_last_hop",
                }
            )
            continue
        if not source or not target or source not in network_dag.nodes or target not in network_dag.nodes:
            continue
        if str(source) == "x" or _row_is_concat_like_source(current, network_dag):
            rejected.append({"edge": edge_id, "source": source, "target": target, "reason": "unsupported_source"})
            continue
        source_layout = dict(current.get("source_layout", {}) or {})
        selected = dict(current.get("selected_layout", {}) or {})
        required = dict(current.get("required_layout", selected) or selected)
        if int(selected.get("stride", 1) or 1) != int(required.get("stride", selected.get("stride", 1)) or 1):
            rejected.append({"edge": edge_id, "source": source, "target": target, "reason": "stride_mismatch"})
            continue
        compact_layout = _layout_for_row_shape(
            current,
            top_beta=0,
            bottom_beta=0,
            stride=max(1, int(selected.get("stride", 1) or 1)),
        )
        if int(compact_layout.get("gap", 1) or 1) != int(required.get("gap", compact_layout.get("gap", 1)) or 1):
            rejected.append({"edge": edge_id, "source": source, "target": target, "reason": "gap_mismatch"})
            continue
        target_output_row = node_rows_by_node.get(target)
        if target_output_row is None:
            rejected.append({"edge": edge_id, "source": source, "target": target, "reason": "target_node_row_missing"})
            continue
        native_stats = _conv2d_module_halo0_channel_aligned_native_stats(
            network_dag,
            compile_plan,
            current,
            compact_layout=dict(compact_layout),
            input_layout=dict(selected),
            node_rows_by_node=node_rows_by_node,
        )
        if native_stats is None:
            rejected.append({"edge": edge_id, "source": source, "target": target, "reason": "native_plan_unavailable"})
            continue
        native_output_ct_count = int(native_stats.get("output_ct_count", 0) or 0)
        if int(native_output_ct_count) <= 0:
            rejected.append({"edge": edge_id, "source": source, "target": target, "reason": "native_output_count_unavailable"})
            continue
        native_input_ct_count = int(native_stats.get("input_ct_count", 0) or 0)
        if int(native_input_ct_count) <= 0:
            rejected.append({"edge": edge_id, "source": source, "target": target, "reason": "native_input_count_unavailable"})
            continue
        rewritten = _rewrite_row_as_halo0_channel_aligned_native_output(
            current,
            native_stats=dict(native_stats),
            reason="boot_boundary_halo0_channel_aligned_output",
        )
        updated_by_edge[edge_id] = dict(rewritten)
        target_output_compact_layout: dict[str, Any] | None = None
        target_output_compact_layout = _compact_layout_for_row(target_output_row)
        updated_by_node[target] = _rewrite_node_as_halo0_boot_boundary_output(
            target_output_row,
            compact_layout=dict(target_output_compact_layout),
            native_ct_count=int(native_output_ct_count),
            slots=int(compile_plan.get("slots", current.get("slots", 32768)) or 32768),
            reason="boot_boundary_halo0_channel_aligned_stripe",
        )
        boot_source_nodes: list[dict[str, Any]] = []
        passthrough_edges: list[dict[str, Any]] = []
        carry_rewrite_failed = False
        target_shape = _shape_tuple(target_output_row.get("shape", ())) if target_output_row else None
        for edge_source, _edge_target in boundary_boot_edges:
            boot_source = str(edge_source)
            boot_source_row = node_rows_by_node.get(boot_source)
            if boot_source_row:
                boot_source_compact_layout = _compact_layout_for_row(boot_source_row)
                if boot_source != target:
                    passthrough_edge_id = f"{target}->{boot_source}"
                    passthrough_row = edge_rows_by_edge.get(passthrough_edge_id)
                    boot_source_shape = _shape_tuple(boot_source_row.get("shape", ()))
                    if (
                        target_output_compact_layout is None
                        or passthrough_row is None
                        or not _layout_preserving_module(network_dag, boot_source)
                        or target_shape != boot_source_shape
                        or _layout_storage_key(target_output_compact_layout)
                        != _layout_storage_key(boot_source_compact_layout)
                        or int(native_output_ct_count) <= 0
                    ):
                        rejected.append(
                            {
                                "edge": edge_id,
                                "source": source,
                                "target": target,
                                "boot_source": boot_source,
                                "reason": "unsafe_layout_preserving_boot_source_carry",
                            }
                        )
                        carry_rewrite_failed = True
                        break
                updated_by_node[boot_source] = _rewrite_node_as_halo0_boot_boundary_output(
                    boot_source_row,
                    compact_layout=dict(boot_source_compact_layout),
                    native_ct_count=int(native_output_ct_count),
                    slots=int(compile_plan.get("slots", current.get("slots", 32768)) or 32768),
                    reason="boot_boundary_halo0_channel_aligned_stripe",
                )
                boot_source_nodes.append(
                    {
                        "node": boot_source,
                        "old_physical_layout": str(boot_source_row.get("physical_layout", "")),
                        "new_physical_layout": PHYSICAL_NATIVE_SOURCE_STRIPE,
                        "old_layout": dict(boot_source_row.get("selected_layout", {}) or {}),
                        "new_layout": dict(boot_source_compact_layout),
                    }
                )
            passthrough_edge_id = f"{target}->{boot_source}"
            passthrough_row = edge_rows_by_edge.get(passthrough_edge_id)
            if passthrough_row is not None and target_output_compact_layout is not None:
                updated_by_edge[passthrough_edge_id] = _rewrite_layout_preserving_edge_as_halo0_compact(
                    passthrough_row,
                    compact_layout=dict(target_output_compact_layout),
                    native_ct_count=int(native_output_ct_count),
                    reason="boot_boundary_halo0_channel_aligned_stripe",
                )
                passthrough_edges.append(
                    {
                        "edge": passthrough_edge_id,
                        "old_physical_layout": str(passthrough_row.get("physical_layout", "")),
                        "new_physical_layout": PHYSICAL_NATIVE_SOURCE_STRIPE,
                    }
                )
        if carry_rewrite_failed:
            continue
        boot_source, boot_target = boundary_boot_edges[0]
        accepted.append(
            {
                "kind": "halo0_channel_aligned_native_output",
                "edge": edge_id,
                "source": source,
                "target": target,
                "boot_edge": {"source": str(boot_source), "target": str(boot_target)},
                "boot_edges": [
                    {"source": str(edge_source), "target": str(edge_target)}
                    for edge_source, edge_target in boundary_boot_edges
                ],
                "old_physical_layout": str(current.get("physical_layout", "")),
                "new_physical_layout": PHYSICAL_NATIVE_SOURCE_STRIPE,
                "old_source_physical_layout": str(current.get("source_physical_layout", "")),
                "new_source_physical_layout": PHYSICAL_NATIVE_SOURCE_STRIPE,
                "old_relayout_depth": int(current.get("relayout_depth_estimate", 0) or 0),
                "new_relayout_depth": 0,
                "old_rotation": int(current.get("planner_rotation_cost_estimate", 0) or 0),
                "new_rotation": int(rewritten.get("planner_rotation_cost_estimate", 0) or 0),
                "source_layout": dict(source_layout),
                "selected_layout": dict(selected if bool(already_native_candidate) else compact_layout),
                "output_layout": dict(target_output_compact_layout or compact_layout),
                "input_layout_preserved": True,
                "source_promoted_nodes": [],
                "source_promoted_edges": [],
                "updated_boot_source_nodes": boot_source_nodes,
                "updated_passthrough_edges": passthrough_edges,
            }
        )

    if not accepted:
        return []

    updated_edge_rows = [
        dict(updated_by_edge.get(str(row.get("edge", "")), dict(row)))
        for row in compile_plan.get("edge_layouts", [])
    ]
    updated_node_rows = [
        dict(updated_by_node.get(str(row.get("node", "")), dict(row)))
        for row in compile_plan.get("node_layouts", [])
    ]
    updated_plan = _refresh_layout_policy_plan_summary(
        {
            **dict(compile_plan),
            "edge_layouts": updated_edge_rows,
            "node_layouts": updated_node_rows,
        }
    )
    summary = dict(updated_plan.get("summary", {}) or {})
    summary["boot_refined_halo0_channel_aligned_stripe_count"] = int(len(accepted))
    updated_plan["summary"] = summary
    updated_plan["bootstrap_aware_layout_refinement"] = {
        "enabled": True,
        "accepted": accepted,
        "rejected": rejected,
    }
    return [
        {
            "kind": "halo0_channel_aligned_stripe",
            "strategy": "halo0_channel_aligned_stripe",
            "plan": updated_plan,
            "rotation_delta": _candidate_rotation_delta(compile_plan, updated_plan),
            "accepted": accepted,
            "rejected": rejected,
            "accepted_count": int(len(accepted)),
            "candidate_priority": -3,
            "allow_after_bootstrap_target": True,
            "require_bootstrap_count_unchanged": True,
            "require_bootstrap_shape_nonincrease": True,
            "boot_interval_count": int(len(intervals)),
            "boot_intervals": _boot_interval_audit_rows(intervals),
        }
    ]


def enumerate_final_boot_boundary_halo0_cleanup_candidate(
    network_dag: Any,
    final_bootstrap_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build the opt-in halo0 cleanup only from the current final boot edges.

    Unlike the normal refinement candidates, this is intentionally not part of
    the iterative search frontier.  Callers apply it after boot placement has
    reached its normal fixed point, then re-solve and either keep the cleanup or
    restore the previous compile plan.
    """

    if not _bootstrap_halo0_channel_aligned_stripe_enabled():
        return {"enabled": False, "reason": "halo0_channel_aligned_stripe_disabled"}

    compile_plan = _layout_policy_compile_plan_for_dag(network_dag)
    if not isinstance(compile_plan, dict):
        return {"enabled": False, "reason": "no_layout_policy_executors"}
    if str(compile_plan.get("policy", "")) not in BOOTSTRAP_AWARE_LAYOUT_REFINEMENT_POLICIES:
        return {
            "enabled": False,
            "policy": str(compile_plan.get("policy", "")),
            "reason": "policy_not_bootstrap_aware_refinement",
        }

    boot_edges = _bootstrap_edge_set(final_bootstrap_audit)
    if not boot_edges:
        return {
            "enabled": False,
            "policy": str(compile_plan.get("policy", "")),
            "reason": "no_final_boot_edges",
        }

    intervals = _bootstrap_upstream_intervals(network_dag, final_bootstrap_audit)
    previous_depths = _snapshot_layout_policy_depths(network_dag)
    actual_native_physical_edges = _actual_native_edge_ids(network_dag)
    candidates = _halo0_channel_aligned_stripe_refinement_candidates(
        network_dag,
        compile_plan,
        intervals=intervals,
        boot_edges=boot_edges,
        actual_native_physical_edges=actual_native_physical_edges,
    )
    if not candidates:
        return {
            "enabled": False,
            "policy": str(compile_plan.get("policy", "")),
            "reason": "no_final_boot_boundary_halo0_cleanup_candidates",
            "native_physical_relayout_edge_count": int(len(actual_native_physical_edges)),
            "final_boot_edge_count": int(len(boot_edges)),
            "boot_interval_count": int(len(intervals)),
            "boot_intervals": _boot_interval_audit_rows(intervals),
            "previous_compile_plan": compile_plan,
            "previous_depths": previous_depths,
        }

    candidate = dict(candidates[0])
    accepted = [dict(row) for row in candidate.get("accepted", [])]
    return {
        **candidate,
        "enabled": True,
        "candidate_id": "bootstrap_final_halo0_boundary_cleanup",
        "kind": "final_boot_boundary_halo0_cleanup",
        "strategy": "final_boot_boundary_halo0_cleanup",
        "accepted": accepted,
        "accepted_count": int(candidate.get("accepted_count", len(accepted)) or len(accepted)),
        "candidate_priority": int(candidate.get("candidate_priority", -3) or -3),
        "allow_after_bootstrap_target": True,
        "require_bootstrap_count_unchanged": True,
        "require_bootstrap_shape_nonincrease": True,
        "policy": str(compile_plan.get("policy", "")),
        "current_bootstrap_count": int(final_bootstrap_audit.get("bootstrap_count", 0) or 0),
        "native_physical_relayout_edge_count": int(len(actual_native_physical_edges)),
        "final_boot_edge_count": int(len(boot_edges)),
        "previous_compile_plan": compile_plan,
        "previous_depths": previous_depths,
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
        rewrite_failed = False
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
            if updated is None:
                rewrite_failed = True
                break
            updated["layout_mode"] = "concat_output_compact_halo_shared"
            updated["concat_output_beta_lift"] = True
            updated["concat_output_beta_lift_source"] = str(source)
            updated_edge_rows.append(updated)
        if rewrite_failed:
            continue
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
                "rotation_delta": _candidate_rotation_delta(compile_plan, updated_plan),
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


def _concat_transitive_consumer_chain(
    network_dag: Any,
    edge_rows_by_edge: dict[str, dict[str, Any]],
    *,
    first_conv: str,
) -> dict[str, Any] | None:
    first_module = network_dag.nodes[str(first_conv)].get("module") if str(first_conv) in network_dag.nodes else None
    if not _conv2d_unit_stride_radius_one(first_module):
        return None
    current = str(first_conv)
    carried_edge_ids: list[str] = []
    carried_edges: list[dict[str, Any]] = []
    while True:
        try:
            successors = [str(value) for value in network_dag.successors(str(current))]
        except Exception:
            return None
        if len(successors) != 1:
            return None
        successor = str(successors[0])
        edge_id = f"{current}->{successor}"
        edge_row = edge_rows_by_edge.get(edge_id)
        if edge_row is None:
            return None
        successor_module = network_dag.nodes[str(successor)].get("module") if str(successor) in network_dag.nodes else None
        if _layout_preserving_module(network_dag, successor):
            carried_edge_ids.append(str(edge_id))
            carried_edges.append(
                {
                    "edge": str(edge_id),
                    "source": str(current),
                    "target": str(successor),
                    "op_kind": str(edge_row.get("op_kind", "")),
                }
            )
            current = str(successor)
            continue
        if _conv2d_unit_stride_radius_one(successor_module):
            return {
                "final_conv": str(successor),
                "final_edge_id": str(edge_id),
                "final_edge_row": dict(edge_row),
                "carried_edge_ids": carried_edge_ids,
                "carried_edges": carried_edges,
            }
        return None


def _local_concat_transitive_beta_lift_refinement_candidates(
    network_dag: Any,
    compile_plan: dict[str, Any],
    *,
    intervals: list[dict[str, Any]],
    boot_edges: set[tuple[str, str]],
    actual_native_physical_edges: set[str],
) -> list[dict[str, Any]]:
    plan_slots = max(1, int(compile_plan.get("slots", 32768) or 32768))
    edge_rows = [{**dict(row), "slots": int(plan_slots)} for row in compile_plan.get("edge_layouts", [])]
    edge_rows_by_edge = {str(row.get("edge", "")): dict(row) for row in edge_rows}
    native_physical_edge_ids = {
        str(row.get("edge", ""))
        for row in edge_rows
        if str(row.get("edge", "")) and _row_is_native_source_stripe_relayout(row)
    } | {str(edge_id) for edge_id in actual_native_physical_edges}
    node_rows_by_node = {str(row.get("node", "")): dict(row) for row in compile_plan.get("node_layouts", [])}
    candidates: list[dict[str, Any]] = []
    for row in edge_rows:
        current = dict(row)
        edge_id = str(current.get("edge", ""))
        source = str(current.get("source", ""))
        target = str(current.get("target", ""))
        if edge_id not in native_physical_edge_ids:
            continue
        if (source, target) in boot_edges:
            continue
        if _candidate_interval_matches(row=current, intervals=intervals, boot_edges=boot_edges):
            continue
        if str(current.get("op_kind", "")) != "conv2d":
            continue
        if str(current.get("physical_layout", "")) != "native_source_stripe":
            continue
        if not source or not target or source not in network_dag.nodes or target not in network_dag.nodes:
            continue
        if type(network_dag.nodes[source].get("module")).__name__ != "Concat":
            continue
        if not _single_successor(network_dag, source, target):
            continue
        if node_rows_by_node.get(source) is None or node_rows_by_node.get(target) is None:
            continue
        join_rows = [
            dict(join_row)
            for join_row in edge_rows
            if str(join_row.get("target", "")) == source and str(join_row.get("op_kind", "")) == "concat"
        ]
        predecessor_set = {str(predecessor) for predecessor in network_dag.predecessors(source)}
        join_source_set = {str(join_row.get("source", "")) for join_row in join_rows}
        synthetic_join = False
        if len(predecessor_set) == 1:
            join_node = str(next(iter(predecessor_set), ""))
            if join_node == f"{source}_join" and join_node in network_dag.nodes:
                join_pred_sources = {
                    str(pred)[: -len("_fork")] if str(pred).endswith("_fork") else str(pred)
                    for pred in network_dag.predecessors(join_node)
                }
                synthetic_join = bool(
                    join_pred_sources == join_source_set
                    and {str(succ) for succ in network_dag.successors(join_node)} == {source}
                )
        if len(join_rows) < 2 or (join_source_set != predecessor_set and not synthetic_join):
            continue
        join_safe = True
        for join_row in join_rows:
            selected = dict(join_row.get("selected_layout", {}) or {})
            source_physical = str(join_row.get("source_physical_layout", PHYSICAL_COMPACT) or "")
            physical = str(join_row.get("physical_layout", "") or "")
            target_physical = str(join_row.get("target_physical_layout", join_row.get("physical_layout", "")) or "")
            if (
                _layout_has_halo(selected)
                or source_physical != PHYSICAL_COMPACT
                or physical != PHYSICAL_COMPACT
                or target_physical != PHYSICAL_COMPACT
            ):
                join_safe = False
                break
        if not join_safe:
            continue

        chain = _concat_transitive_consumer_chain(
            network_dag,
            edge_rows_by_edge,
            first_conv=str(target),
        )
        if chain is None:
            continue
        final_edge_id = str(chain.get("final_edge_id", ""))
        final_edge = dict(chain.get("final_edge_row", {}) or {})
        final_selected = dict(final_edge.get("selected_layout", {}) or {})
        final_required = dict(final_edge.get("required_layout", final_selected) or final_selected)
        if not _layout_has_halo(final_selected):
            continue
        if not _layout_covers(final_selected, final_required):
            continue
        if int(final_selected.get("gap", 1) or 1) != int(
            final_required.get("gap", final_selected.get("gap", 1)) or 1
        ):
            continue
        producer_layout = _input_demand_for_output_layout(
            network_dag.nodes[target].get("module"),
            current,
            final_selected,
        )
        if not _layout_has_halo(producer_layout):
            continue
        if int(producer_layout.get("gap", 1) or 1) != int(final_selected.get("gap", 1) or 1):
            continue
        if _layout_top_beta(producer_layout) != _layout_top_beta(final_selected) + 1:
            continue
        if _layout_bottom_beta(producer_layout) != _layout_bottom_beta(final_selected) + 1:
            continue
        current_required = dict(current.get("required_layout", current.get("selected_layout", {}) or {}) or {})
        if current_required and not _layout_covers(producer_layout, current_required):
            continue
        carried_supported = True
        for carried_edge_id in chain.get("carried_edge_ids", []) or []:
            carried_row = edge_rows_by_edge.get(str(carried_edge_id))
            if carried_row is None:
                carried_supported = False
                break
            carried_selected = dict(carried_row.get("selected_layout", {}) or {})
            carried_required = dict(carried_row.get("required_layout", carried_selected) or carried_selected)
            if int(carried_selected.get("gap", 1) or 1) != int(final_selected.get("gap", 1) or 1):
                carried_supported = False
                break
            if int(carried_required.get("gap", carried_selected.get("gap", 1)) or 1) != int(
                final_selected.get("gap", 1) or 1
            ):
                carried_supported = False
                break
            if not _layout_covers(final_selected, carried_required):
                carried_supported = False
                break
        if not carried_supported:
            continue

        carried_edge_ids = {str(value) for value in chain.get("carried_edge_ids", []) or []}
        updated_edge_rows: list[dict[str, Any]] = []
        accepted_edges: list[dict[str, Any]] = []
        carried_edges: list[dict[str, Any]] = []
        rewrite_failed = False
        for candidate_row in compile_plan.get("edge_layouts", []):
            candidate_edge_id = str(candidate_row.get("edge", ""))
            if candidate_edge_id == edge_id:
                updated = _rewrite_row_as_compact_source(
                    dict(candidate_row),
                    source_layout=producer_layout,
                    target_layout=producer_layout,
                    reason="local_concat_transitive_beta_lift",
                )
                if updated is None:
                    rewrite_failed = True
                    break
                updated["concat_output_beta_lift"] = True
                updated["local_concat_transitive_beta_lift"] = True
                updated["concat_output_beta_lift_source"] = str(source)
                accepted_edges.append(
                    {
                        "edge": str(edge_id),
                        "source": str(source),
                        "target": str(target),
                        "old_physical_layout": str(current.get("physical_layout", "")),
                        "new_physical_layout": str(updated.get("physical_layout", "")),
                        "new_layout_mode": str(updated.get("layout_mode", "")),
                    }
                )
                updated_edge_rows.append(updated)
            elif candidate_edge_id in carried_edge_ids:
                updated = _rewrite_row_as_compact_source(
                    dict(candidate_row),
                    source_layout=final_selected,
                    target_layout=final_selected,
                    reason="local_concat_transitive_activation_carry",
                )
                if updated is None:
                    rewrite_failed = True
                    break
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
            elif (
                candidate_edge_id == final_edge_id
                and candidate_edge_id in native_physical_edge_ids
                and str(candidate_row.get("physical_layout", "")) == "native_source_stripe"
            ):
                updated = _rewrite_row_as_compact_source(
                    dict(candidate_row),
                    source_layout=final_selected,
                    target_layout=final_selected,
                    reason="local_concat_transitive_beta_lift",
                )
                if updated is None:
                    rewrite_failed = True
                    break
                accepted_edges.append(
                    {
                        "edge": str(candidate_edge_id),
                        "source": str(candidate_row.get("source", "")),
                        "target": str(candidate_row.get("target", "")),
                        "old_physical_layout": str(candidate_row.get("physical_layout", "")),
                        "new_physical_layout": str(updated.get("physical_layout", "")),
                        "new_layout_mode": str(updated.get("layout_mode", "")),
                    }
                )
                updated_edge_rows.append(updated)
            else:
                updated_edge_rows.append(dict(candidate_row))
        if rewrite_failed:
            continue

        lift_nodes = {str(source), str(target)}
        for carried_edge_id in carried_edge_ids:
            carried_row = edge_rows_by_edge.get(str(carried_edge_id), {})
            carried_target = str(carried_row.get("target", ""))
            if carried_target:
                lift_nodes.add(str(carried_target))
        updated_node_rows: list[dict[str, Any]] = []
        for candidate_node in compile_plan.get("node_layouts", []):
            node = str(candidate_node.get("node", ""))
            if node == source:
                updated_node_rows.append(
                    _rewrite_node_as_beta_lift_producer(
                        dict(candidate_node),
                        layout=producer_layout,
                        reason="local_concat_transitive_beta_lift",
                    )
                )
            elif node in lift_nodes:
                producer_growth_delta = _producer_node_layout_growth_rotation_delta(
                    network_dag,
                    compile_plan,
                    node=str(node),
                    old_row=dict(candidate_node),
                    new_layout=dict(final_selected),
                )
                updated_node = _rewrite_node_as_beta_lift_producer(
                    dict(candidate_node),
                    layout=final_selected,
                    reason="local_concat_transitive_beta_lift",
                )
                if int(producer_growth_delta) > 0:
                    updated_node["producer_fused_rotation_estimate"] = int(
                        int(candidate_node.get("producer_fused_rotation_estimate", 0) or 0)
                        + int(producer_growth_delta)
                    )
                updated_node_rows.append(updated_node)
            else:
                updated_node_rows.append(dict(candidate_node))

        native_output_counts = dict(compile_plan.get("_native_physical_output_ct_counts", {}) or {})
        for node in lift_nodes:
            if str(node) != str(source):
                native_output_counts.pop(str(node), None)
        updated_plan = _refresh_layout_policy_plan_summary(
            {
                **dict(compile_plan),
                "edge_layouts": updated_edge_rows,
                "node_layouts": updated_node_rows,
                "_native_physical_output_ct_counts": native_output_counts,
            }
        )
        summary = dict(updated_plan.get("summary", {}) or {})
        summary["boot_refined_local_concat_transitive_beta_lift_count"] = (
            int(summary.get("boot_refined_local_concat_transitive_beta_lift_count", 0) or 0) + 1
        )
        updated_plan["summary"] = summary
        candidates.append(
            {
                "kind": "local_concat_transitive_beta_lift",
                "strategy": "local_concat_transitive_beta_lift",
                "plan": updated_plan,
                "rotation_delta": _candidate_rotation_delta(compile_plan, updated_plan),
                "candidate_priority": 2,
                "require_bootstrap_count_unchanged": False,
                "require_bootstrap_shape_nonincrease": False,
                "accepted": [
                    {
                        "kind": "local_concat_transitive_beta_lift",
                        "producer": str(source),
                        "producer_layout": dict(producer_layout),
                        "consumer_layout": dict(final_selected),
                        "covered_edges": accepted_edges,
                        "carried_edges": carried_edges,
                        "covered_edge_count": int(len(accepted_edges)),
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


def _node_region_executor(network_dag: Any, node: str) -> tuple[Any | None, Any | None]:
    module = network_dag.nodes[str(node)].get("module") if str(node) in network_dag.nodes else None
    runtime = getattr(module, "region_runtime", None) if module is not None else None
    executor = getattr(runtime, "executor", None) if runtime is not None else None
    return module, executor


def _restore_object_attrs(obj: Any, snapshot: dict[str, Any]) -> None:
    if obj is None:
        return
    for name, value in snapshot.items():
        if value is _MISSING:
            try:
                delattr(obj, name)
            except AttributeError:
                pass
        else:
            setattr(obj, name, value)


def _producer_native_output_plan_ct_count(
    network_dag: Any,
    *,
    producer: str,
    layout: dict[str, Any],
    native_ct_count: int,
    slots: int,
    allow_output_ct_superset: bool = False,
) -> int:
    module, executor = _node_region_executor(network_dag, str(producer))
    if module is None or executor is None:
        return 0
    base_executor = getattr(executor, "base_executor", executor)
    if not bool(
        getattr(base_executor, "native_halo_output_capable", False)
        or getattr(executor, "native_halo_output_capable", False)
    ):
        return 0
    refresh = getattr(base_executor, "_refresh_runtime_plan", None)
    if not callable(refresh):
        return 0

    gap = max(1, int(dict(layout).get("gap", 1) or 1))
    output_attrs = {
        "layout_policy_output_layout": dict(layout),
        "layout_policy_output_row_offset": int(_layout_physical_top_beta(layout) * gap),
        "layout_policy_output_materialization": NATIVE_OUTPUT_MATERIALIZATION,
        "fhe_output_shape": torch.Size([int(native_ct_count), max(1, int(slots))]),
    }
    module_snapshot = {name: getattr(module, name, _MISSING) for name in output_attrs}
    executor_snapshot = {
        name: getattr(base_executor, name, _MISSING)
        for name in (
            "native_plan",
            "_native_plan_require_target_fit",
            "rows",
            "cols",
            "slots",
            "output_shape",
            "fhe_output_shape",
        )
    }
    try:
        for name, value in output_attrs.items():
            setattr(module, name, value)
        refresh()
        tight_output = getattr(base_executor, "_uses_tight_compact_output", None)
        if not callable(tight_output):
            return 0
        if bool(tight_output()):
            return 0
        plan = getattr(base_executor, "native_plan", None)
        output_count = _positive_int(getattr(plan, "output_ct_count", 0), 0)
        if int(output_count) <= 0:
            return 0
        if bool(allow_output_ct_superset):
            if int(output_count) < int(native_ct_count):
                return 0
        elif int(output_count) != int(native_ct_count):
            return 0
        return int(output_count) if int(native_ct_count) > 0 else 0
    except Exception:
        return 0
    finally:
        _restore_object_attrs(module, module_snapshot)
        _restore_object_attrs(base_executor, executor_snapshot)


def _producer_native_output_plan_fits(
    network_dag: Any,
    *,
    producer: str,
    layout: dict[str, Any],
    native_ct_count: int,
    slots: int,
    allow_output_ct_superset: bool = False,
) -> bool:
    return (
        int(
            _producer_native_output_plan_ct_count(
                network_dag,
                producer=str(producer),
                layout=dict(layout),
                native_ct_count=int(native_ct_count),
                slots=int(slots),
                allow_output_ct_superset=bool(allow_output_ct_superset),
            )
        )
        > 0
    )


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
        rewrite_failed = False
        for candidate_row in compile_plan.get("edge_layouts", []):
            candidate_edge_id = str(candidate_row.get("edge", ""))
            if candidate_edge_id == edge_id:
                updated = _rewrite_row_as_compact_source(
                    dict(candidate_row),
                    source_layout=selected,
                    target_layout=selected,
                    reason=producer_reason,
                )
                if updated is None:
                    rewrite_failed = True
                    break
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
                if updated is None:
                    rewrite_failed = True
                    break
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
        if rewrite_failed:
            continue

        updated_node_rows: list[dict[str, Any]] = []
        lift_nodes = {str(producer)}
        if carried_edge_ids:
            lift_nodes.add(str(source))
        for candidate_node in compile_plan.get("node_layouts", []):
            if str(candidate_node.get("node", "")) not in lift_nodes:
                updated_node_rows.append(dict(candidate_node))
                continue
            node = str(candidate_node.get("node", ""))
            producer_growth_delta = _producer_node_layout_growth_rotation_delta(
                network_dag,
                compile_plan,
                node=str(node),
                old_row=dict(candidate_node),
                new_layout=dict(selected),
            )
            updated_node = _rewrite_node_as_beta_lift_producer(
                dict(candidate_node),
                layout=selected,
                reason=producer_reason,
            )
            if int(producer_growth_delta) > 0:
                updated_node["producer_fused_rotation_estimate"] = int(
                    int(candidate_node.get("producer_fused_rotation_estimate", 0) or 0)
                    + int(producer_growth_delta)
                )
            updated_node_rows.append(updated_node)

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
                "rotation_delta": _candidate_rotation_delta(compile_plan, updated_plan),
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


def _local_activation_beta_lift_refinement_candidates(
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
        if not _layout_preserving_module(network_dag, source):
            continue
        producer = _single_predecessor(network_dag, source)
        if producer is None or producer not in network_dag.nodes:
            continue
        if type(network_dag.nodes[producer].get("module")).__name__ != "Conv2d":
            continue
        if (source, target) in boot_edges or (producer, source) in boot_edges:
            continue
        if _candidate_interval_matches(
            row=current,
            intervals=intervals,
            boot_edges=boot_edges,
        ):
            continue
        if _single_predecessor(network_dag, target) != source:
            continue
        if not _single_successor(network_dag, producer, source):
            continue
        if not _single_successor(network_dag, source, target):
            continue

        selected = dict(current.get("selected_layout", {}) or {})
        activation_edge_id = f"{producer}->{source}"
        activation_edge = edge_rows_by_edge.get(activation_edge_id)
        producer_row = node_rows_by_node.get(producer)
        source_node_row = node_rows_by_node.get(source)
        if activation_edge is None or producer_row is None or source_node_row is None:
            continue
        activation_required = dict(
            activation_edge.get("required_layout", activation_edge.get("selected_layout", {}) or {}) or {}
        )
        if activation_required and not _layout_covers(selected, activation_required):
            continue
        if int(selected.get("gap", 1) or 1) != int(
            activation_required.get("gap", selected.get("gap", 1)) or 1
        ):
            continue

        updated_edge_rows: list[dict[str, Any]] = []
        carried_edges: list[dict[str, Any]] = []
        rewrite_failed = False
        for candidate_row in compile_plan.get("edge_layouts", []):
            candidate_edge_id = str(candidate_row.get("edge", ""))
            if candidate_edge_id == edge_id:
                updated = _rewrite_row_as_compact_source(
                    dict(candidate_row),
                    source_layout=selected,
                    target_layout=selected,
                    reason="local_activation_beta_lift",
                )
                if updated is None:
                    rewrite_failed = True
                    break
                updated["local_activation_beta_lift"] = True
                updated["local_activation_beta_lift_source"] = str(producer)
                updated_edge_rows.append(updated)
            elif candidate_edge_id == activation_edge_id:
                updated = _rewrite_row_as_compact_source(
                    dict(candidate_row),
                    source_layout=selected,
                    target_layout=selected,
                    reason="local_activation_beta_lift",
                )
                if updated is None:
                    rewrite_failed = True
                    break
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
        if rewrite_failed:
            continue

        updated_node_rows: list[dict[str, Any]] = []
        lift_nodes = {str(producer), str(source)}
        output_tile_delta = 0
        slots = max(1, int(compile_plan.get("slots", 32768) or 32768))
        for candidate_node in compile_plan.get("node_layouts", []):
            if str(candidate_node.get("node", "")) not in lift_nodes:
                updated_node_rows.append(dict(candidate_node))
                continue
            node = str(candidate_node.get("node", ""))
            output_tile_delta += _layout_tile_count_delta(
                dict(candidate_node.get("selected_layout", {}) or {}),
                dict(selected),
                slots=slots,
            )
            producer_growth_delta = _producer_node_layout_growth_rotation_delta(
                network_dag,
                compile_plan,
                node=str(node),
                old_row=dict(candidate_node),
                new_layout=dict(selected),
            )
            updated_node = _rewrite_node_as_beta_lift_producer(
                dict(candidate_node),
                layout=selected,
                reason="local_activation_beta_lift",
            )
            if int(producer_growth_delta) > 0:
                updated_node["producer_fused_rotation_estimate"] = int(
                    int(candidate_node.get("producer_fused_rotation_estimate", 0) or 0)
                    + int(producer_growth_delta)
                )
            updated_node_rows.append(updated_node)

        updated_plan = _refresh_layout_policy_plan_summary(
            {
                **dict(compile_plan),
                "edge_layouts": updated_edge_rows,
                "node_layouts": updated_node_rows,
            }
        )
        summary = dict(updated_plan.get("summary", {}) or {})
        summary["boot_refined_local_activation_beta_lift_count"] = (
            int(summary.get("boot_refined_local_activation_beta_lift_count", 0) or 0) + 1
        )
        updated_plan["summary"] = summary
        rotation_delta = _candidate_rotation_delta(compile_plan, updated_plan)
        allow_depth_unchanged = bool(int(output_tile_delta) <= 0 and int(rotation_delta) <= 0)
        candidates.append(
            {
                "kind": "local_activation_beta_lift",
                "strategy": "local_activation_beta_lift",
                "plan": updated_plan,
                "rotation_delta": int(rotation_delta),
                "candidate_priority": -1 if bool(allow_depth_unchanged) else 1,
                "allow_relayout_depth_unchanged": bool(allow_depth_unchanged),
                "allow_after_bootstrap_target": bool(allow_depth_unchanged),
                "output_tile_delta": int(output_tile_delta),
                "accepted": [
                    {
                        "kind": "local_activation_beta_lift",
                        "producer": str(producer),
                        "producer_layout": dict(selected),
                        "output_tile_delta": int(output_tile_delta),
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


def _local_native_output_beta_lift_refinement_candidates(
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
    slots = max(1, int(compile_plan.get("slots", 32768) or 32768))

    for row in compile_plan.get("edge_layouts", []):
        current = dict(row)
        edge_id = str(current.get("edge", ""))
        source = str(current.get("source", ""))
        target = str(current.get("target", ""))
        if edge_id not in actual_native_physical_edges:
            continue
        if not _native_source_stripe_compact_rewrite_blocked(current):
            continue
        if not _native_conv_row_beta_lift_supported(current):
            continue
        if not source or not target or source not in network_dag.nodes or target not in network_dag.nodes:
            continue
        if type(network_dag.nodes[target].get("module")).__name__ != "Conv2d":
            continue
        if (source, target) in boot_edges:
            continue
        if _candidate_interval_matches(row=current, intervals=intervals, boot_edges=boot_edges):
            continue
        if _single_predecessor(network_dag, target) != source:
            continue

        selected = dict(current.get("selected_layout", {}) or {})
        native_ct_count = _native_input_ct_count_from_edge(current)
        if int(native_ct_count) <= 0:
            continue

        producer = source
        carried_edge_ids: list[str] = []
        carried_edges: list[dict[str, Any]] = []
        kind = "local_direct_native_output_beta_lift"
        source_module = network_dag.nodes[source].get("module")
        if _layout_preserving_module(network_dag, source):
            predecessor = _single_predecessor(network_dag, source)
            if predecessor is None or predecessor not in network_dag.nodes:
                continue
            if type(network_dag.nodes[predecessor].get("module")).__name__ != "Conv2d":
                continue
            if not _single_successor(network_dag, predecessor, source):
                continue
            if not _single_successor(network_dag, source, target):
                continue
            activation_edge_id = f"{predecessor}->{source}"
            activation_edge = edge_rows_by_edge.get(activation_edge_id)
            if activation_edge is None:
                continue
            activation_required = dict(
                activation_edge.get("required_layout", activation_edge.get("selected_layout", {}) or {}) or {}
            )
            if activation_required and not _layout_covers(selected, activation_required):
                continue
            if int(selected.get("gap", 1) or 1) != int(
                activation_required.get("gap", selected.get("gap", 1)) or 1
            ):
                continue
            if (predecessor, source) in boot_edges:
                continue
            producer = str(predecessor)
            carried_edge_ids.append(str(activation_edge_id))
            kind = "local_activation_native_output_beta_lift"
        elif type(source_module).__name__ in {"AvgPool2d", "Conv2d", "ConvTranspose2d"}:
            if not _single_successor(network_dag, source, target):
                continue
        else:
            continue

        producer_row = node_rows_by_node.get(producer)
        source_node_row = node_rows_by_node.get(source)
        if producer_row is None or source_node_row is None:
            continue
        current_shape = _shape_tuple(current.get("shape", ()))
        for node_row in (producer_row, source_node_row):
            node_shape = _shape_tuple(dict(node_row).get("shape", ()))
            if current_shape is not None and node_shape is not None and tuple(node_shape) != tuple(current_shape):
                producer_row = None
                break
        if producer_row is None:
            continue
        if not _producer_native_output_plan_fits(
            network_dag,
            producer=str(producer),
            layout=selected,
            native_ct_count=int(native_ct_count),
            slots=int(slots),
        ):
            continue

        updated_edge_rows: list[dict[str, Any]] = []
        for candidate_row in compile_plan.get("edge_layouts", []):
            candidate_edge_id = str(candidate_row.get("edge", ""))
            if candidate_edge_id == edge_id:
                updated = _rewrite_row_as_native_source(
                    dict(candidate_row),
                    source_layout=selected,
                    target_layout=selected,
                    native_ct_count=int(native_ct_count),
                    reason=kind,
                )
                updated["local_native_output_beta_lift"] = True
                updated["local_native_output_beta_lift_source"] = str(producer)
                updated_edge_rows.append(updated)
            elif candidate_edge_id in carried_edge_ids:
                updated = _rewrite_row_as_native_source(
                    dict(candidate_row),
                    source_layout=selected,
                    target_layout=selected,
                    native_ct_count=int(native_ct_count),
                    reason=kind,
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

        lift_nodes = {str(producer), str(source)}
        updated_node_rows: list[dict[str, Any]] = []
        output_tile_delta = 0
        for candidate_node in compile_plan.get("node_layouts", []):
            node = str(candidate_node.get("node", ""))
            if node not in lift_nodes:
                updated_node_rows.append(dict(candidate_node))
                continue
            output_tile_delta += _layout_tile_count_delta(
                dict(candidate_node.get("selected_layout", {}) or {}),
                dict(selected),
                slots=slots,
            )
            updated_node_rows.append(
                _rewrite_node_as_native_output_producer(
                    dict(candidate_node),
                    layout=selected,
                    native_ct_count=int(native_ct_count),
                    slots=int(slots),
                    reason=kind,
                )
            )

        native_output_counts = dict(compile_plan.get("_native_physical_output_ct_counts", {}) or {})
        for node in sorted(lift_nodes):
            native_output_counts[str(node)] = int(native_ct_count)
        updated_plan = _refresh_layout_policy_plan_summary(
            {
                **dict(compile_plan),
                "edge_layouts": updated_edge_rows,
                "node_layouts": updated_node_rows,
                "_native_physical_output_ct_counts": native_output_counts,
            }
        )
        summary = dict(updated_plan.get("summary", {}) or {})
        summary["boot_refined_local_native_output_beta_lift_count"] = (
            int(summary.get("boot_refined_local_native_output_beta_lift_count", 0) or 0) + 1
        )
        updated_plan["summary"] = summary
        rotation_delta = _candidate_rotation_delta(compile_plan, updated_plan)
        candidates.append(
            {
                "kind": str(kind),
                "strategy": str(kind),
                "plan": updated_plan,
                "rotation_delta": int(rotation_delta),
                "candidate_priority": 0,
                "output_tile_delta": int(output_tile_delta),
                "require_bootstrap_count_unchanged": True,
                "require_bootstrap_shape_nonincrease": True,
                "accepted": [
                    {
                        "kind": str(kind),
                        "producer": str(producer),
                        "producer_layout": dict(selected),
                        "native_output_ct_count": int(native_ct_count),
                        "output_tile_delta": int(output_tile_delta),
                        "covered_edges": [
                            {
                                "edge": str(edge_id),
                                "source": str(source),
                                "target": str(target),
                                "old_physical_layout": str(current.get("physical_layout", "")),
                                "new_physical_layout": PHYSICAL_NATIVE_SOURCE_STRIPE,
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


def _batched_activation_native_output_beta_lift_refinement_candidates(
    network_dag: Any,
    compile_plan: dict[str, Any],
    *,
    intervals: list[dict[str, Any]],
    boot_edges: set[tuple[str, str]],
    actual_native_physical_edges: set[str],
) -> list[dict[str, Any]]:
    edge_rows_by_edge = {str(row.get("edge", "")): dict(row) for row in compile_plan.get("edge_layouts", [])}
    node_rows_by_node = {str(row.get("node", "")): dict(row) for row in compile_plan.get("node_layouts", [])}
    slots = max(1, int(compile_plan.get("slots", 32768) or 32768))

    units: list[dict[str, Any]] = []
    seen_edges: set[str] = set()
    for row in compile_plan.get("edge_layouts", []):
        current = dict(row)
        edge_id = str(current.get("edge", ""))
        source = str(current.get("source", ""))
        target = str(current.get("target", ""))
        if not edge_id or edge_id in seen_edges:
            continue
        if edge_id not in actual_native_physical_edges:
            continue
        if not _native_source_stripe_compact_rewrite_blocked(current):
            continue
        if not _native_conv_row_beta_lift_supported(current):
            continue
        if not source or not target or source not in network_dag.nodes or target not in network_dag.nodes:
            continue
        if type(network_dag.nodes[target].get("module")).__name__ != "Conv2d":
            continue
        if not _layout_preserving_module(network_dag, source):
            continue
        producer = _single_predecessor(network_dag, source)
        if producer is None or producer not in network_dag.nodes:
            continue
        if type(network_dag.nodes[producer].get("module")).__name__ != "Conv2d":
            continue
        if _single_predecessor(network_dag, target) != source:
            continue
        if not _single_successor(network_dag, producer, source):
            continue
        if not _single_successor(network_dag, source, target):
            continue

        interval_matches = _candidate_interval_matches(
            row=current,
            intervals=intervals,
            boot_edges=boot_edges,
        )
        if not interval_matches and (source, target) not in boot_edges and (producer, source) not in boot_edges:
            continue

        selected = dict(current.get("selected_layout", {}) or {})
        target_native_ct_count = _native_input_ct_count_from_edge(current)
        logical_ct_count = int(selected.get("tile_count", 0) or 0)
        if int(target_native_ct_count) <= 0 or int(logical_ct_count) <= 0:
            continue

        activation_edge_id = f"{producer}->{source}"
        activation_edge = edge_rows_by_edge.get(activation_edge_id)
        producer_row = node_rows_by_node.get(str(producer))
        source_node_row = node_rows_by_node.get(str(source))
        if activation_edge is None or producer_row is None or source_node_row is None:
            continue
        activation_required = dict(
            activation_edge.get("required_layout", activation_edge.get("selected_layout", {}) or {}) or {}
        )
        if activation_required and not _layout_covers(selected, activation_required):
            continue
        if int(selected.get("gap", 1) or 1) != int(
            activation_required.get("gap", selected.get("gap", 1)) or 1
        ):
            continue

        current_shape = _shape_tuple(current.get("shape", ()))
        shape_mismatch = False
        for node_row in (producer_row, source_node_row):
            node_shape = _shape_tuple(dict(node_row).get("shape", ()))
            if current_shape is not None and node_shape is not None and tuple(node_shape) != tuple(current_shape):
                shape_mismatch = True
                break
        if bool(shape_mismatch):
            continue
        native_output_ct_count = _producer_native_output_plan_ct_count(
            network_dag,
            producer=str(producer),
            layout=selected,
            native_ct_count=int(target_native_ct_count),
            slots=int(slots),
        )
        if int(native_output_ct_count) != int(target_native_ct_count):
            continue
        mode = "native_source"
        output_ct_count = int(native_output_ct_count)

        units.append(
            {
                "edge_id": str(edge_id),
                "source": str(source),
                "target": str(target),
                "producer": str(producer),
                "activation_edge_id": str(activation_edge_id),
                "selected": dict(selected),
                "native_ct_count": int(target_native_ct_count),
                "output_ct_count": int(output_ct_count),
                "mode": str(mode),
                "current": dict(current),
            }
        )
        seen_edges.add(str(edge_id))

    if len(units) < 2:
        return []

    node_updates: dict[str, tuple[dict[str, Any], int, str]] = {}
    for unit in units:
        layout = dict(unit["selected"])
        output_ct_count = int(unit["output_ct_count"])
        mode = str(unit["mode"])
        for node in (str(unit["producer"]), str(unit["source"])):
            existing = node_updates.get(str(node))
            if existing is not None:
                existing_layout, existing_count, existing_mode = existing
                if (
                    dict(existing_layout) != dict(layout)
                    or int(existing_count) != int(output_ct_count)
                    or str(existing_mode) != str(mode)
                ):
                    return []
            node_updates[str(node)] = (dict(layout), int(output_ct_count), str(mode))

    units_by_edge = {str(unit["edge_id"]): dict(unit) for unit in units}
    carried_by_edge = {str(unit["activation_edge_id"]): dict(unit) for unit in units}
    updated_edge_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for candidate_row in compile_plan.get("edge_layouts", []):
        candidate_edge_id = str(candidate_row.get("edge", ""))
        if candidate_edge_id in units_by_edge:
            unit = units_by_edge[candidate_edge_id]
            layout = dict(unit["selected"])
            native_ct_count = int(unit["native_ct_count"])
            if str(unit["mode"]) == "native_source":
                updated = _rewrite_row_as_native_source(
                    dict(candidate_row),
                    source_layout=layout,
                    target_layout=layout,
                    native_ct_count=int(native_ct_count),
                    reason="batched_activation_native_output_beta_lift",
                )
            else:
                updated = _rewrite_row_as_native_compact_source(
                    dict(candidate_row),
                    source_layout=layout,
                    target_layout=layout,
                    reason="batched_activation_native_output_beta_lift",
                )
            updated["batched_activation_native_output_beta_lift"] = True
            updated["batched_activation_native_output_beta_lift_source"] = str(unit["producer"])
            updated_edge_rows.append(updated)
            accepted.append(
                {
                    "kind": "batched_activation_native_output_beta_lift",
                    "producer": str(unit["producer"]),
                    "producer_layout": dict(layout),
                    "native_output_ct_count": int(unit["output_ct_count"]),
                    "mode": str(unit["mode"]),
                    "covered_edges": [
                        {
                            "edge": str(candidate_edge_id),
                            "source": str(unit["source"]),
                            "target": str(unit["target"]),
                            "old_physical_layout": str(dict(unit["current"]).get("physical_layout", "")),
                            "new_physical_layout": str(updated.get("physical_layout", "")),
                        }
                    ],
                    "carried_edges": [
                        {
                            "edge": str(unit["activation_edge_id"]),
                            "source": str(unit["producer"]),
                            "target": str(unit["source"]),
                            "op_kind": str(edge_rows_by_edge.get(str(unit["activation_edge_id"]), {}).get("op_kind", "")),
                            "physical_layout": PHYSICAL_NATIVE_SOURCE_STRIPE
                            if str(unit["mode"]) == "native_source"
                            else PHYSICAL_LOGICAL_HALO,
                        }
                    ],
                    "covered_edge_count": 1,
                }
            )
        elif candidate_edge_id in carried_by_edge:
            unit = carried_by_edge[candidate_edge_id]
            layout = dict(unit["selected"])
            native_ct_count = int(unit["native_ct_count"])
            if str(unit["mode"]) == "native_source":
                updated = _rewrite_row_as_native_source(
                    dict(candidate_row),
                    source_layout=layout,
                    target_layout=layout,
                    native_ct_count=int(native_ct_count),
                    reason="batched_activation_native_output_beta_lift",
                )
            else:
                updated = _rewrite_row_as_compact_source(
                    dict(candidate_row),
                    source_layout=layout,
                    target_layout=layout,
                    reason="batched_activation_native_output_beta_lift",
                )
                if updated is None:
                    return []
            updated["batched_activation_native_output_beta_lift"] = True
            updated_edge_rows.append(updated)
        else:
            updated_edge_rows.append(dict(candidate_row))

    updated_node_rows: list[dict[str, Any]] = []
    output_tile_delta = 0
    for candidate_node in compile_plan.get("node_layouts", []):
        node = str(candidate_node.get("node", ""))
        update = node_updates.get(str(node))
        if update is None:
            updated_node_rows.append(dict(candidate_node))
            continue
        layout, output_ct_count, mode = update
        output_tile_delta += _layout_tile_count_delta(
            dict(candidate_node.get("selected_layout", {}) or {}),
            dict(layout),
            slots=int(slots),
        )
        if str(mode) == "native_source":
            updated_node_rows.append(
                _rewrite_node_as_native_output_producer(
                    dict(candidate_node),
                    layout=dict(layout),
                    native_ct_count=int(output_ct_count),
                    slots=int(slots),
                    reason="batched_activation_native_output_beta_lift",
                )
            )
        else:
            updated_node_rows.append(
                _rewrite_node_as_beta_lift_producer(
                    dict(candidate_node),
                    layout=dict(layout),
                    reason="batched_activation_native_output_beta_lift",
                )
            )
    if int(output_tile_delta) > 0:
        return []

    native_output_counts = dict(compile_plan.get("_native_physical_output_ct_counts", {}) or {})
    for node, (_layout, output_ct_count, mode) in node_updates.items():
        if str(mode) == "native_source":
            native_output_counts[str(node)] = int(output_ct_count)

    updated_plan = _refresh_layout_policy_plan_summary(
        {
            **dict(compile_plan),
            "edge_layouts": updated_edge_rows,
            "node_layouts": updated_node_rows,
            "_native_physical_output_ct_counts": native_output_counts,
        }
    )
    summary = dict(updated_plan.get("summary", {}) or {})
    summary["boot_refined_batched_activation_native_output_beta_lift_count"] = (
        int(summary.get("boot_refined_batched_activation_native_output_beta_lift_count", 0) or 0) + 1
    )
    updated_plan["summary"] = summary
    return [
        {
            "kind": "batched_activation_native_output_beta_lift",
            "strategy": "batched_activation_native_output_beta_lift",
            "plan": updated_plan,
            "rotation_delta": _candidate_rotation_delta(compile_plan, updated_plan),
            "candidate_priority": -2,
            "output_tile_delta": int(output_tile_delta),
            "require_bootstrap_count_unchanged": False,
            "require_bootstrap_shape_nonincrease": True,
            "accepted": accepted,
            "rejected": [],
        }
    ]


def _explicit_concat_native_materialization_refinement_candidates(
    network_dag: Any,
    compile_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """Try replacing fused cat-conv with explicit native concat materialization.

    The default DP keeps concat lazy and fuses it into the consuming Conv2d.
    In individual-provider mode that fused row can be much more expensive than
    materializing the native concat once and then running a normal native Conv.
    This candidate is deliberately narrow and relies on the official trial
    BootstrapSolver to reject any depth-induced boot increase.
    """

    edge_rows = [dict(row) for row in compile_plan.get("edge_layouts", [])]
    node_rows = [dict(row) for row in compile_plan.get("node_layouts", [])]
    node_rows_by_node = {str(row.get("node", "")): dict(row) for row in node_rows}
    incoming_by_target: dict[str, list[dict[str, Any]]] = {}
    for row in edge_rows:
        incoming_by_target.setdefault(str(row.get("target", "")), []).append(dict(row))

    slots = max(1, int(compile_plan.get("slots", 32768) or 32768))
    candidates: list[dict[str, Any]] = []
    for current in edge_rows:
        source = str(current.get("source", ""))
        target = str(current.get("target", ""))
        edge_id = str(current.get("edge", ""))
        if str(current.get("op_kind", "")) != "conv2d":
            continue
        if not source or source not in getattr(network_dag, "nodes", {}):
            continue
        if type(network_dag.nodes[source].get("module")).__name__ != "Concat":
            continue
        if not bool(current.get("concat_native_runtime_materializer", False)):
            continue
        if str(current.get("physical_layout", "")) != PHYSICAL_NATIVE_SOURCE_STRIPE:
            continue
        if str(current.get("provider_lt_grouping_mode", "")) != "individual":
            continue
        if str(current.get("native_halo_channel_fold_mode", "")) != "per_stripe":
            continue

        input_rows = [dict(row) for row in incoming_by_target.get(str(source), []) if str(row.get("op_kind", "")) == "concat"]
        if not input_rows:
            continue
        if any(str(row.get("source_physical_layout", "")) != PHYSICAL_NATIVE_SOURCE_STRIPE for row in input_rows):
            continue
        if any(not _parse_storage_signature(row.get("native_halo_source_storage_signature") or ()) for row in input_rows):
            continue

        source_node_row = dict(node_rows_by_node.get(str(source), {}) or {})
        if str(source_node_row.get("physical_layout", "")) != PHYSICAL_NATIVE_SOURCE_STRIPE:
            continue
        target_signature = _parse_storage_signature(
            source_node_row.get("native_halo_target_storage_signature")
            or current.get("native_halo_source_storage_signature")
            or ()
        )
        if not target_signature:
            continue

        consumer_target_signature = _parse_storage_signature(
            current.get("native_halo_target_storage_signature")
            or current.get("consumer_native_source_stripe_target_signature")
            or ()
        )
        normal_stats = _recompute_normal_native_conv_rotation_estimate(
            network_dag,
            current,
            source_storage_signature=tuple(target_signature),
            target_storage_signature=tuple(consumer_target_signature),
            slots=int(slots),
        )
        if normal_stats is None:
            continue
        normal_rotation = _positive_int(normal_stats.get("rotations", 0), 0)
        fused_rotation = _positive_int(current.get("planner_rotation_cost_estimate", 0), 0)
        if int(normal_rotation) <= 0 or int(fused_rotation) <= 0:
            continue
        materialize_rotation = _concat_native_materialize_rotation_estimate(
            network_dag,
            concat_node=str(source),
            input_rows=input_rows,
            target_signature=tuple(target_signature),
            slots=int(slots),
        )
        if int(materialize_rotation) <= 0:
            materialize_rotation = int(len(target_signature))
        rotation_delta = int(int(normal_rotation) + int(materialize_rotation) - int(fused_rotation))
        if int(rotation_delta) >= 0:
            continue

        updated_edge_rows: list[dict[str, Any]] = []
        accepted_input_edges: list[dict[str, Any]] = []
        for row in edge_rows:
            row_edge_id = str(row.get("edge", ""))
            if row_edge_id == edge_id:
                updated = dict(row)
                for key in (
                    "concat_native_runtime_materializer",
                    "concat_native_runtime_materializer_reason",
                    "concat_fused_native_output_signature_projected",
                    "concat_fusion_runtime_estimate",
                    "concat_fusion_native_source_runtime_estimate",
                ):
                    updated.pop(key, None)
                updated["concat_explicit_native_materialization"] = True
                updated["concat_explicit_native_materialization_reason"] = (
                    "bootstrap_refine_explicit_concat_native_materialization"
                )
                updated["consumer_fused_relayout"] = False
                updated["consumer_fused_relayout_reason"] = ""
                updated["consumer_fused_rotation_estimate"] = 0
                updated["layout_mode"] = "native_halo_stripe"
                updated["physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
                updated["source_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
                updated["target_physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
                updated["lt_estimator"] = "native_halo_plan_runtime_signature"
                updated["planner_rotation_cost_estimate"] = int(normal_rotation)
                updated["lt_bsgs_rotation_estimate"] = int(normal_rotation)
                updated["lt_baby_rotation_estimate"] = int(normal_rotation)
                updated["lt_giant_rotation_estimate"] = 0
                updated["lt_local_submatrix_rotation_estimate"] = int(normal_rotation)
                updated["native_halo_rotation_estimator"] = "native_halo_plan_runtime_signature"
                updated["native_halo_rotation_mode"] = "c_only"
                updated["native_halo_runtime_signature_estimate"] = True
                updated["rotation_report_source"] = "explicit_concat_native_materialization_candidate"
                updated["rotation_eval_count_mode"] = "independent_transform_bsgs"
                updated_edge_rows.append(updated)
                continue
            if str(row.get("target", "")) == str(source) and str(row.get("op_kind", "")) == "concat":
                updated = dict(row)
                updated["concat_explicit_native_materialization"] = True
                updated["concat_explicit_native_materialization_reason"] = (
                    "bootstrap_refine_explicit_concat_native_materialization"
                )
                updated_edge_rows.append(updated)
                accepted_input_edges.append(
                    {
                        "edge": str(row_edge_id),
                        "source": str(row.get("source", "")),
                        "target": str(row.get("target", "")),
                        "native_source_signature_count": int(
                            len(_parse_storage_signature(row.get("native_halo_source_storage_signature") or ()))
                        ),
                    }
                )
                continue
            updated_edge_rows.append(dict(row))

        updated_node_rows: list[dict[str, Any]] = []
        for row in node_rows:
            if str(row.get("node", "")) != str(source):
                updated_node_rows.append(dict(row))
                continue
            updated = dict(row)
            updated["physical_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
            updated["output_relayout"] = True
            updated["output_relayout_reason"] = "bootstrap_refine_explicit_concat_native_materialization"
            updated["concat_explicit_native_materialization"] = True
            updated["producer_materialized_native_source_stripe"] = True
            updated["producer_native_source_stripe_reason"] = (
                "bootstrap_refine_explicit_concat_native_materialization"
            )
            updated["layout_policy_output_materialization"] = NATIVE_OUTPUT_MATERIALIZATION
            updated["native_halo_output_storage_layout"] = PHYSICAL_NATIVE_SOURCE_STRIPE
            updated["native_halo_target_storage_signature"] = _serialise_storage_signature(tuple(target_signature))
            updated["native_physical_output_ct_count"] = int(len(target_signature))
            updated["native_output_ct_count"] = int(len(target_signature))
            updated["native_output_ct_count_estimate"] = int(len(target_signature))
            updated["fhe_shape"] = [int(len(target_signature)), int(slots)]
            updated["relayout_rotation_estimate"] = int(materialize_rotation)
            updated["relayout_mask_mult_estimate"] = 0
            updated["relayout_depth_estimate"] = 1
            updated_node_rows.append(updated)

        native_output_counts = dict(compile_plan.get("_native_physical_output_ct_counts", {}) or {})
        native_output_counts[str(source)] = int(len(target_signature))
        updated_plan = _refresh_layout_policy_plan_summary(
            {
                **dict(compile_plan),
                "edge_layouts": updated_edge_rows,
                "node_layouts": updated_node_rows,
                "_native_physical_output_ct_counts": native_output_counts,
            }
        )
        summary = dict(updated_plan.get("summary", {}) or {})
        summary["boot_refined_explicit_concat_native_materialization_count"] = (
            int(summary.get("boot_refined_explicit_concat_native_materialization_count", 0) or 0) + 1
        )
        updated_plan["summary"] = summary
        candidates.append(
            {
                "kind": "explicit_concat_native_materialization",
                "strategy": "explicit_concat_native_materialization",
                "plan": updated_plan,
                "rotation_delta": int(rotation_delta),
                "candidate_priority": 1,
                "allow_relayout_depth_increase_without_boot_increase": True,
                "max_relayout_depth_increase_without_boot_increase": 1,
                "allow_after_bootstrap_target": True,
                "output_tile_delta": 0,
                "require_bootstrap_count_unchanged": True,
                "require_bootstrap_shape_nonincrease": True,
                "accepted": [
                    {
                        "kind": "explicit_concat_native_materialization",
                        "concat": str(source),
                        "consumer": str(target),
                        "edge": str(edge_id),
                        "fused_rotation": int(fused_rotation),
                        "normal_native_rotation": int(normal_rotation),
                        "normal_native_rotation_source": "recomputed_native_plan",
                        "normal_native_transform_count": int(normal_stats.get("transforms", 0) or 0),
                        "normal_native_source_signature_count": int(len(target_signature)),
                        "normal_native_target_signature_count": int(len(consumer_target_signature)),
                        "materialize_rotation_estimate": int(materialize_rotation),
                        "rotation_delta": int(rotation_delta),
                        "native_output_ct_count": int(len(target_signature)),
                        "input_edges": accepted_input_edges,
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
    if str(compile_plan.get("policy", "")) not in BOOTSTRAP_AWARE_LAYOUT_REFINEMENT_POLICIES:
        return {"enabled": False, "reason": "policy_not_bootstrap_aware_refinement"}

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
        beta_lift = _annotate_costless_output_lift_candidate(compile_plan, dict(beta_lift))
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
                "rotation_delta": int(
                    beta_lift.get(
                        "rotation_delta",
                        _candidate_rotation_delta(compile_plan, dict(beta_lift["plan"])),
                    )
                    or 0
                ),
                "candidate_priority": int(beta_lift.get("candidate_priority", 0) or 0),
                "allow_relayout_depth_unchanged": bool(
                    beta_lift.get("allow_relayout_depth_unchanged", False)
                ),
                "allow_after_bootstrap_target": bool(beta_lift.get("allow_after_bootstrap_target", False)),
                "output_tile_delta": int(beta_lift.get("output_tile_delta", 0) or 0),
                "require_bootstrap_count_unchanged": bool(
                    beta_lift.get("require_bootstrap_count_unchanged", False)
                ),
                "require_bootstrap_shape_nonincrease": bool(
                    beta_lift.get("require_bootstrap_shape_nonincrease", False)
                ),
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
        concat_lift = _annotate_costless_output_lift_candidate(compile_plan, dict(concat_lift))
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
                "rotation_delta": int(
                    concat_lift.get(
                        "rotation_delta",
                        _candidate_rotation_delta(compile_plan, dict(concat_lift["plan"])),
                    )
                    or 0
                ),
                "candidate_priority": int(concat_lift.get("candidate_priority", 0) or 0),
                "allow_relayout_depth_unchanged": bool(
                    concat_lift.get("allow_relayout_depth_unchanged", False)
                ),
                "allow_after_bootstrap_target": bool(concat_lift.get("allow_after_bootstrap_target", False)),
                "output_tile_delta": int(concat_lift.get("output_tile_delta", 0) or 0),
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
        boot_lift = _annotate_costless_output_lift_candidate(compile_plan, dict(boot_lift))
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
                "rotation_delta": int(
                    boot_lift.get(
                        "rotation_delta",
                        _candidate_rotation_delta(compile_plan, dict(boot_lift["plan"])),
                    )
                    or 0
                ),
                "candidate_priority": int(boot_lift.get("candidate_priority", 0) or 0),
                "allow_relayout_depth_unchanged": bool(
                    boot_lift.get("allow_relayout_depth_unchanged", False)
                ),
                "allow_after_bootstrap_target": bool(boot_lift.get("allow_after_bootstrap_target", False)),
                "output_tile_delta": int(boot_lift.get("output_tile_delta", 0) or 0),
                "boot_interval_count": int(len(intervals)),
                "boot_intervals": boot_interval_rows,
            }
        )
    for native_lift in _local_native_output_beta_lift_refinement_candidates(
        network_dag,
        compile_plan,
        intervals=intervals,
        boot_edges=boot_edges,
        actual_native_physical_edges=actual_native_physical_edges,
    ):
        native_lift = _annotate_costless_output_lift_candidate(compile_plan, dict(native_lift))
        accepted = [dict(row) for row in native_lift.get("accepted", [])]
        candidate_kind = str(native_lift.get("kind", "local_native_output_beta_lift"))
        candidate_strategy = str(native_lift.get("strategy", candidate_kind))
        candidates.append(
            {
                "candidate_id": f"bootstrap_refine_local_native_output_{len(candidates) + 1}",
                "kind": candidate_kind,
                "strategy": candidate_strategy,
                "plan": dict(native_lift["plan"]),
                "accepted": accepted,
                "rejected": list(native_lift.get("rejected", [])),
                "accepted_count": int(
                    sum(int(row.get("covered_edge_count", 1) or 1) for row in accepted)
                ),
                "rotation_delta": int(
                    native_lift.get(
                        "rotation_delta",
                        _candidate_rotation_delta(compile_plan, dict(native_lift["plan"])),
                    )
                    or 0
                ),
                "candidate_priority": int(native_lift.get("candidate_priority", 0) or 0),
                "allow_relayout_depth_unchanged": bool(
                    native_lift.get("allow_relayout_depth_unchanged", False)
                ),
                "allow_after_bootstrap_target": bool(native_lift.get("allow_after_bootstrap_target", False)),
                "output_tile_delta": int(native_lift.get("output_tile_delta", 0) or 0),
                "require_bootstrap_count_unchanged": True,
                "require_bootstrap_shape_nonincrease": True,
                "boot_interval_count": int(len(intervals)),
                "boot_intervals": boot_interval_rows,
            }
        )
    for batch_lift in _batched_activation_native_output_beta_lift_refinement_candidates(
        network_dag,
        compile_plan,
        intervals=intervals,
        boot_edges=boot_edges,
        actual_native_physical_edges=actual_native_physical_edges,
    ):
        batch_lift = _annotate_costless_output_lift_candidate(compile_plan, dict(batch_lift))
        accepted = [dict(row) for row in batch_lift.get("accepted", [])]
        candidate_kind = str(batch_lift.get("kind", "batched_activation_native_output_beta_lift"))
        candidate_strategy = str(batch_lift.get("strategy", candidate_kind))
        candidates.append(
            {
                "candidate_id": f"bootstrap_refine_batched_activation_native_output_{len(candidates) + 1}",
                "kind": candidate_kind,
                "strategy": candidate_strategy,
                "plan": dict(batch_lift["plan"]),
                "accepted": accepted,
                "rejected": list(batch_lift.get("rejected", [])),
                "accepted_count": int(
                    sum(int(row.get("covered_edge_count", 1) or 1) for row in accepted)
                ),
                "rotation_delta": int(
                    batch_lift.get(
                        "rotation_delta",
                        _candidate_rotation_delta(compile_plan, dict(batch_lift["plan"])),
                    )
                    or 0
                ),
                "candidate_priority": int(batch_lift.get("candidate_priority", -2) or -2),
                "allow_relayout_depth_unchanged": bool(
                    batch_lift.get("allow_relayout_depth_unchanged", False)
                ),
                "allow_after_bootstrap_target": bool(batch_lift.get("allow_after_bootstrap_target", False)),
                "output_tile_delta": int(batch_lift.get("output_tile_delta", 0) or 0),
                "require_bootstrap_count_unchanged": False,
                "require_bootstrap_shape_nonincrease": True,
                "boot_interval_count": int(len(intervals)),
                "boot_intervals": boot_interval_rows,
            }
        )
    for local_lift in _local_activation_beta_lift_refinement_candidates(
        network_dag,
        compile_plan,
        intervals=intervals,
        boot_edges=boot_edges,
        actual_native_physical_edges=actual_native_physical_edges,
    ):
        local_lift = _annotate_costless_output_lift_candidate(compile_plan, dict(local_lift))
        accepted = [dict(row) for row in local_lift.get("accepted", [])]
        candidate_kind = str(local_lift.get("kind", "local_activation_beta_lift"))
        candidate_strategy = str(local_lift.get("strategy", candidate_kind))
        candidates.append(
            {
                "candidate_id": f"bootstrap_refine_local_activation_{len(candidates) + 1}",
                "kind": candidate_kind,
                "strategy": candidate_strategy,
                "plan": dict(local_lift["plan"]),
                "accepted": accepted,
                "rejected": list(local_lift.get("rejected", [])),
                "accepted_count": int(
                    sum(int(row.get("covered_edge_count", 1) or 1) for row in accepted)
                ),
                "rotation_delta": int(
                    local_lift.get(
                        "rotation_delta",
                        _candidate_rotation_delta(compile_plan, dict(local_lift["plan"])),
                    )
                    or 0
                ),
                "candidate_priority": int(
                    local_lift["candidate_priority"]
                    if "candidate_priority" in local_lift
                    else 1
                ),
                "allow_relayout_depth_unchanged": bool(
                    local_lift.get("allow_relayout_depth_unchanged", False)
                ),
                "allow_after_bootstrap_target": bool(local_lift.get("allow_after_bootstrap_target", False)),
                "output_tile_delta": int(local_lift.get("output_tile_delta", 0) or 0),
                "require_bootstrap_count_unchanged": True,
                "require_bootstrap_shape_nonincrease": True,
                "boot_interval_count": int(len(intervals)),
                "boot_intervals": boot_interval_rows,
            }
        )
    for concat_lift in _local_concat_transitive_beta_lift_refinement_candidates(
        network_dag,
        compile_plan,
        intervals=intervals,
        boot_edges=boot_edges,
        actual_native_physical_edges=actual_native_physical_edges,
    ):
        concat_lift = _annotate_costless_output_lift_candidate(compile_plan, dict(concat_lift))
        accepted = [dict(row) for row in concat_lift.get("accepted", [])]
        candidate_kind = str(concat_lift.get("kind", "local_concat_transitive_beta_lift"))
        candidate_strategy = str(concat_lift.get("strategy", candidate_kind))
        candidates.append(
            {
                "candidate_id": f"bootstrap_refine_local_concat_{len(candidates) + 1}",
                "kind": candidate_kind,
                "strategy": candidate_strategy,
                "plan": dict(concat_lift["plan"]),
                "accepted": accepted,
                "rejected": list(concat_lift.get("rejected", [])),
                "accepted_count": int(
                    sum(int(row.get("covered_edge_count", 1) or 1) for row in accepted)
                ),
                "rotation_delta": int(
                    concat_lift.get(
                        "rotation_delta",
                        _candidate_rotation_delta(compile_plan, dict(concat_lift["plan"])),
                    )
                    or 0
                ),
                "candidate_priority": int(concat_lift.get("candidate_priority", 2) or 2),
                "allow_relayout_depth_unchanged": bool(
                    concat_lift.get("allow_relayout_depth_unchanged", False)
                ),
                "allow_after_bootstrap_target": bool(concat_lift.get("allow_after_bootstrap_target", False)),
                "output_tile_delta": int(concat_lift.get("output_tile_delta", 0) or 0),
                "require_bootstrap_count_unchanged": bool(
                    concat_lift.get("require_bootstrap_count_unchanged", False)
                ),
                "require_bootstrap_shape_nonincrease": bool(
                    concat_lift.get("require_bootstrap_shape_nonincrease", True)
                ),
                "boot_interval_count": int(len(intervals)),
                "boot_intervals": boot_interval_rows,
            }
        )
    for concat_materialize in _explicit_concat_native_materialization_refinement_candidates(
        network_dag,
        compile_plan,
    ):
        accepted = [dict(row) for row in concat_materialize.get("accepted", [])]
        candidate_kind = str(concat_materialize.get("kind", "explicit_concat_native_materialization"))
        candidate_strategy = str(concat_materialize.get("strategy", candidate_kind))
        candidates.append(
            {
                "candidate_id": f"bootstrap_refine_explicit_concat_{len(candidates) + 1}",
                "kind": candidate_kind,
                "strategy": candidate_strategy,
                "plan": dict(concat_materialize["plan"]),
                "accepted": accepted,
                "rejected": list(concat_materialize.get("rejected", [])),
                "accepted_count": int(
                    sum(int(row.get("covered_edge_count", 1) or 1) for row in accepted)
                ),
                "rotation_delta": int(
                    concat_materialize.get(
                        "rotation_delta",
                        _candidate_rotation_delta(compile_plan, dict(concat_materialize["plan"])),
                    )
                    or 0
                ),
                "candidate_priority": int(concat_materialize.get("candidate_priority", 1) or 1),
                "allow_relayout_depth_unchanged": bool(
                    concat_materialize.get("allow_relayout_depth_unchanged", False)
                ),
                "allow_relayout_depth_increase_without_boot_increase": bool(
                    concat_materialize.get("allow_relayout_depth_increase_without_boot_increase", True)
                ),
                "max_relayout_depth_increase_without_boot_increase": int(
                    concat_materialize.get("max_relayout_depth_increase_without_boot_increase", 1) or 1
                ),
                "allow_after_bootstrap_target": bool(concat_materialize.get("allow_after_bootstrap_target", True)),
                "output_tile_delta": int(concat_materialize.get("output_tile_delta", 0) or 0),
                "require_bootstrap_count_unchanged": bool(
                    concat_materialize.get("require_bootstrap_count_unchanged", True)
                ),
                "require_bootstrap_shape_nonincrease": bool(
                    concat_materialize.get("require_bootstrap_shape_nonincrease", True)
                ),
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
                "rotation_delta": int(
                    native_candidate.get(
                        "rotation_delta",
                        _candidate_rotation_delta(compile_plan, dict(native_candidate["plan"])),
                    )
                    or 0
                ),
                "boot_interval_count": int(len(intervals)),
                "boot_intervals": boot_interval_rows,
            }
        )

    if not candidates:
        return {
            "enabled": False,
            "policy": str(compile_plan.get("policy", "")),
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
        "candidate_priority": int(candidate.get("candidate_priority", 0) or 0),
        "allow_relayout_depth_unchanged": bool(candidate.get("allow_relayout_depth_unchanged", False)),
        "allow_relayout_depth_increase_without_boot_increase": bool(
            candidate.get("allow_relayout_depth_increase_without_boot_increase", False)
        ),
        "max_relayout_depth_increase_without_boot_increase": int(
            candidate.get("max_relayout_depth_increase_without_boot_increase", 0) or 0
        ),
        "allow_after_bootstrap_target": bool(candidate.get("allow_after_bootstrap_target", False)),
        "output_tile_delta": int(candidate.get("output_tile_delta", 0) or 0),
        "require_bootstrap_count_unchanged": bool(candidate.get("require_bootstrap_count_unchanged", False)),
        "require_bootstrap_shape_nonincrease": bool(candidate.get("require_bootstrap_shape_nonincrease", False)),
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
        key: value
        for key, value in dict(audit).items()
        if key not in {"_previous_module_layouts", "previous_compile_plan", "previous_depths"}
    }
    return audit


def apply_bootstrap_aware_layout_refinement(
    network_dag: Any,
    first_pass_audit: dict[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper that applies the lowest-rotation candidate."""

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
    selected = min(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            int(candidate.get("rotation_delta", 0) or 0),
            int(candidate.get("candidate_priority", 0) or 0),
            -int(candidate.get("accepted_count", 0) or 0),
            str(candidate.get("candidate_id", "")),
        ),
    )
    audit = apply_bootstrap_aware_layout_refinement_candidate(
        network_dag,
        dict(selected),
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
    policy = str(compile_plan.get("policy", ""))
    if policy in BOOTSTRAP_AWARE_LAYOUT_REFINEMENT_POLICIES:
        return dict(compile_plan), {
            "enabled": False,
            "policy": policy,
            "reason": "bootstrap_aware_policy_uses_refinement_final_halo0_cleanup",
        }

    node_rows = {str(row.get("node", "")): dict(row) for row in compile_plan.get("node_layouts", [])}
    if not node_rows:
        return dict(compile_plan), {"enabled": False, "reason": "no_layout_policy_node_rows"}

    compressed_nodes: set[str] = set()
    audit_nodes: list[dict[str, Any]] = []
    queue: list[str] = []
    for node in sorted(str(value) for value in bootstrap_nodes):
        row = node_rows.get(str(node))
        if row is None:
            continue
        module = network_dag.nodes.get(str(node), {}).get("module") if str(node) in getattr(network_dag, "nodes", {}) else None
        if (
            bool(row.get("concat_explicit_native_materialization", False))
            or (
                type(module).__name__ == "Concat"
                and str(row.get("physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE
            )
        ):
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
        if int(current_tiles) <= int(compact_tiles):
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
                "forced_compact_boundary": False,
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
            if _native_source_stripe_compact_rewrite_blocked(row):
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
