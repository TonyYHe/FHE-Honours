from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.models.unet import UNet22
from orion.nn.linear import Conv2d, ConvTranspose2d
from orion.nn.pooling import AvgPool2d


DEFAULT_SLOTS = 32768
RELAYOUT_ROTATION_WEIGHT = 6.0
LT_ROTATION_WEIGHT = 8.0
HALO_SLOT_WEIGHT = 1.0 / 32.0
TILE_WEIGHT = 4.0
BOOTSTRAP_PROXY_WEIGHT = 64.0
BOOTSTRAP_HALO_SLOT_DIVISOR = 4096.0
POLICY_ALIASES = {
    "fixed": "fixed_max",
    "fixedmax": "fixed_max",
    "fixed-max": "fixed_max",
    "fixed_max": "fixed_max",
    "eager": "eager",
    "eager_relayout": "eager",
    "eager-relayout": "eager",
    "greedy": "greedy",
    "greedy_local": "greedy",
    "greedy-local": "greedy",
    "dp": "dp",
    "dp_global": "dp",
    "dp-global": "dp",
}
POLICY_LABELS = {
    "fixed_max": "Fixed-Max-Halo",
    "eager": "Eager-Re-Layout",
    "greedy": "Greedy-Local",
    "dp": "DP-Global",
}


@dataclass(frozen=True)
class NetworkSpec:
    network: str
    dataset: str
    image_size: int
    input_channels: int
    base_channels: int
    provider_mode: str


@dataclass(frozen=True)
class LayoutState:
    alpha: int
    beta: int
    stride: int
    gap: int
    core_slots: int
    stored_slots: int
    tile_count: int

    @property
    def halo_slots(self) -> int:
        return max(0, int(self.stored_slots) - int(self.core_slots))

    def key(self) -> tuple[int, int, int, int]:
        return int(self.alpha), int(self.beta), int(self.stride), int(self.gap)

    def covers(self, other: "LayoutState") -> bool:
        return (
            int(self.gap) == int(other.gap)
            and int(self.alpha) >= int(other.alpha)
            and int(self.beta) >= int(other.beta)
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class EdgeInfo:
    source: str
    target: str
    shape: tuple[int, int, int, int]
    fhe_shape: tuple[int, int, int, int]
    requirement: LayoutState
    compact: LayoutState
    op_kind: str
    lt_rotation_base: int

    @property
    def edge_id(self) -> str:
        return f"{self.source}->{self.target}"


@dataclass(frozen=True)
class PolicyPlan:
    policy: str
    policy_label: str
    metric_source: str
    relayouts: int
    halo_redundancy_ratio: float
    total_ciphertext_tiles: int
    stored_slots: int
    relayout_rotation_estimate: int
    lt_bsgs_rotation_estimate: int
    bootstrap_proxy: int
    objective: float
    edge_layouts: tuple[dict[str, Any], ...]
    runtime_status: str = "planner_only"
    runtime_reason: str = ""
    bootstrap_count: int | None = None
    he_forward_s: float | None = None
    mae: float | None = None
    dice: float | None = None
    speedup_vs_fixed_max: float | None = None

    def summary_row(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "metric_source": self.metric_source,
            "relayouts": int(self.relayouts),
            "halo_redundancy_ratio": float(self.halo_redundancy_ratio),
            "total_ciphertext_tiles": int(self.total_ciphertext_tiles),
            "stored_slots": int(self.stored_slots),
            "relayout_rotation_estimate": int(self.relayout_rotation_estimate),
            "lt_bsgs_rotation_estimate": int(self.lt_bsgs_rotation_estimate),
            "bootstrap_proxy": int(self.bootstrap_proxy),
            "bootstrap_count": "" if self.bootstrap_count is None else int(self.bootstrap_count),
            "he_forward_s": "" if self.he_forward_s is None else float(self.he_forward_s),
            "mae": "" if self.mae is None else float(self.mae),
            "dice": "" if self.dice is None else float(self.dice),
            "speedup_vs_fixed_max": "" if self.speedup_vs_fixed_max is None else float(self.speedup_vs_fixed_max),
            "runtime_status": self.runtime_status,
            "runtime_reason": self.runtime_reason,
            "objective": float(self.objective),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.summary_row()
        payload["policy_label"] = self.policy_label
        payload["edge_layouts"] = [dict(row) for row in self.edge_layouts]
        return payload


def normalize_policy(value: str) -> str:
    key = str(value).strip().lower()
    if key not in POLICY_ALIASES:
        supported = ", ".join(sorted(POLICY_ALIASES))
        raise ValueError(f"unknown layout policy {value!r}; supported: {supported}")
    return POLICY_ALIASES[key]


def normalize_policies(values: Iterable[str]) -> tuple[str, ...]:
    policies = [normalize_policy(value) for value in values]
    return tuple(dict.fromkeys(policies))


def network_spec(network: str) -> NetworkSpec:
    normalized = str(network).strip().lower()
    if normalized == "u22_64_base32":
        return NetworkSpec(
            network="u22_64_base32",
            dataset="montgomery_lung_64",
            image_size=64,
            input_channels=1,
            base_channels=32,
            provider_mode="u22_64_base32",
        )
    if normalized == "u22_256_base32":
        return NetworkSpec(
            network="u22_256_base32",
            dataset="kvasir_polyp_256",
            image_size=256,
            input_channels=3,
            base_channels=32,
            provider_mode="u22_256_base32",
        )
    raise ValueError("layout-policy ablation supports u22_64_base32 and u22_256_base32")


def build_u22_dag(spec: NetworkSpec) -> NetworkDAG:
    torch.manual_seed(0)
    model = UNet22(dataset=str(spec.dataset), base_channels=int(spec.base_channels))
    traced = OrionTracer().trace_model(model)
    x = torch.randn((1, int(spec.input_channels), int(spec.image_size), int(spec.image_size)), dtype=torch.float32)
    StatsTracker(traced).propagate(x)
    dag = NetworkDAG(traced)
    dag.build_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()
    return dag


def _prod(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def _shape_tuple(value: Any) -> tuple[int, int, int, int]:
    shape = tuple(int(item) for item in value)
    if len(shape) != 4:
        raise ValueError(f"expected NCHW shape, received {shape}")
    return shape


def _layout_for_shape(
    *,
    shape: tuple[int, int, int, int],
    gap: int,
    alpha: int,
    beta: int,
    stride: int,
    slots: int,
) -> LayoutState:
    _n, channels, height, width = shape
    phase = max(1, int(gap) * int(gap))
    channel_groups = _ceil_div(int(channels), int(phase))
    core_slots = int(channel_groups * int(height) * int(gap) * int(width) * int(gap))
    stored_h = int(height) * int(gap) + int(alpha + beta) * int(gap)
    stored_slots = int(channel_groups * int(stored_h) * int(width) * int(gap))
    return LayoutState(
        alpha=int(alpha),
        beta=int(beta),
        stride=int(stride),
        gap=int(gap),
        core_slots=int(core_slots),
        stored_slots=int(stored_slots),
        tile_count=max(1, _ceil_div(int(stored_slots), int(slots))),
    )


def _consumer_requirement(module: Any | None) -> tuple[int, int, int, str, int]:
    if isinstance(module, Conv2d) and not isinstance(module, ConvTranspose2d):
        kernel = tuple(int(value) for value in getattr(module, "kernel_size", (1, 1)))
        stride = tuple(int(value) for value in getattr(module, "stride", (1, 1)))
        pad = tuple(int(value) for value in getattr(module, "padding", (0, 0)))
        radius = max(0, max(kernel) // 2 if max(pad) > 0 else max(kernel) - 1)
        return int(radius), int(radius), int(stride[0]), "conv2d", max(0, int(kernel[0] * kernel[1] - 1))
    if isinstance(module, AvgPool2d):
        kernel = tuple(int(value) for value in getattr(module, "kernel_size", (1, 1)))
        stride = tuple(int(value) for value in getattr(module, "stride", (1, 1)))
        halo = max(0, int(max(kernel) - max(stride)))
        return int(halo), int(halo), int(stride[0]), "avgpool2d", max(0, int(kernel[0] * kernel[1] - 1))
    if isinstance(module, ConvTranspose2d):
        kernel = tuple(int(value) for value in getattr(module, "kernel_size", (1, 1)))
        stride = tuple(int(value) for value in getattr(module, "stride", (1, 1)))
        return 0, 0, int(stride[0]), "conv_transpose2d", max(0, int(kernel[0] * kernel[1] - 1))
    return 0, 0, 1, type(module).__name__.lower() if module is not None else "input", 0


def _edge_shapes(dag: NetworkDAG, source: str, target: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int]:
    source_module = dag.nodes[source].get("module")
    target_module = dag.nodes[target].get("module")
    if source_module is not None and hasattr(source_module, "output_shape"):
        shape = _shape_tuple(getattr(source_module, "output_shape"))
        fhe_shape = _shape_tuple(getattr(source_module, "fhe_output_shape", getattr(source_module, "output_shape")))
        gap = int(getattr(source_module, "output_gap", 1))
        return shape, fhe_shape, gap
    if target_module is not None and hasattr(target_module, "input_shape"):
        shape = _shape_tuple(getattr(target_module, "input_shape"))
        fhe_shape = _shape_tuple(getattr(target_module, "fhe_input_shape", getattr(target_module, "input_shape")))
        gap = int(getattr(target_module, "input_gap", 1))
        return shape, fhe_shape, gap
    raise ValueError(f"cannot infer edge shape for {source}->{target}")


def build_edge_infos(dag: NetworkDAG, *, slots: int = DEFAULT_SLOTS) -> tuple[EdgeInfo, ...]:
    edges: list[EdgeInfo] = []
    for source, target in dag.edges:
        target_module = dag.nodes[target].get("module")
        shape, fhe_shape, gap = _edge_shapes(dag, str(source), str(target))
        alpha, beta, stride, op_kind, lt_base = _consumer_requirement(target_module)
        compact = _layout_for_shape(shape=shape, gap=int(gap), alpha=0, beta=0, stride=1, slots=int(slots))
        requirement = _layout_for_shape(
            shape=shape,
            gap=int(gap),
            alpha=int(alpha),
            beta=int(beta),
            stride=int(stride),
            slots=int(slots),
        )
        edges.append(
            EdgeInfo(
                source=str(source),
                target=str(target),
                shape=shape,
                fhe_shape=fhe_shape,
                requirement=requirement,
                compact=compact,
                op_kind=str(op_kind),
                lt_rotation_base=int(lt_base),
            )
        )
    topo_index = {str(node): index for index, node in enumerate(dag.topological_sort())}
    return tuple(sorted(edges, key=lambda edge: (topo_index.get(edge.source, 10**9), topo_index.get(edge.target, 10**9))))


def _same_layout(left: LayoutState, right: LayoutState) -> bool:
    return left.key() == right.key()


def _max_layout(edges: Sequence[EdgeInfo], *, slots: int) -> LayoutState:
    if not edges:
        raise ValueError("cannot build max layout for empty edge set")
    edge = edges[0]
    alpha = max(int(item.requirement.alpha) for item in edges)
    beta = max(int(item.requirement.beta) for item in edges)
    stride = max(int(item.requirement.stride) for item in edges)
    return _layout_for_shape(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        alpha=int(alpha),
        beta=int(beta),
        stride=int(stride),
        slots=int(slots),
    )


def _edge_row(edge: EdgeInfo, layout: LayoutState, *, relayout: bool, relayout_reason: str, lt_rotations: int) -> dict[str, Any]:
    return {
        "edge": edge.edge_id,
        "source": edge.source,
        "target": edge.target,
        "op_kind": edge.op_kind,
        "shape": [int(value) for value in edge.shape],
        "fhe_shape": [int(value) for value in edge.fhe_shape],
        "required_layout": edge.requirement.to_dict(),
        "selected_layout": layout.to_dict(),
        "relayout": bool(relayout),
        "relayout_reason": str(relayout_reason),
        "lt_bsgs_rotation_estimate": int(lt_rotations),
    }


def _lt_rotations(edge: EdgeInfo, layout: LayoutState) -> int:
    return int(edge.lt_rotation_base) * int(layout.tile_count)


def _relayout_rotations(layouts: Iterable[LayoutState]) -> int:
    return int(sum(max(1, int(layout.tile_count)) * max(1, int(layout.alpha + layout.beta + 1)) for layout in layouts))


def _halo_slots_for_rows(edge_rows: Iterable[dict[str, Any]]) -> int:
    return int(
        sum(
            max(0, int(row["selected_layout"]["stored_slots"]) - int(row["selected_layout"]["core_slots"]))
            for row in edge_rows
        )
    )


def _edge_linear_cost(row: dict[str, Any]) -> float:
    layout = dict(row["selected_layout"])
    halo_slots = max(0, int(layout["stored_slots"]) - int(layout["core_slots"]))
    return (
        float(row["lt_bsgs_rotation_estimate"]) * LT_ROTATION_WEIGHT
        + float(halo_slots) * HALO_SLOT_WEIGHT
        + float(layout["tile_count"]) * TILE_WEIGHT
    )


def _relayout_linear_cost(layout: LayoutState) -> float:
    return float(_relayout_rotations((layout,))) * RELAYOUT_ROTATION_WEIGHT


def _policy_linear_cost(edge_rows: Iterable[dict[str, Any]], relayout_layouts: Iterable[LayoutState]) -> float:
    return float(sum(_edge_linear_cost(row) for row in edge_rows)) + float(
        sum(_relayout_linear_cost(layout) for layout in relayout_layouts)
    )


def _finalize_policy(
    *,
    policy: str,
    edge_rows: list[dict[str, Any]],
    relayout_layouts: list[LayoutState],
) -> PolicyPlan:
    stored_slots = int(sum(int(row["selected_layout"]["stored_slots"]) for row in edge_rows))
    core_slots = int(sum(int(row["selected_layout"]["core_slots"]) for row in edge_rows))
    tile_count = int(sum(int(row["selected_layout"]["tile_count"]) for row in edge_rows))
    relayout_rotation_estimate = _relayout_rotations(relayout_layouts)
    lt_rotation_estimate = int(sum(int(row["lt_bsgs_rotation_estimate"]) for row in edge_rows))
    halo_slots = int(stored_slots - core_slots)
    bootstrap_proxy = int(
        math.ceil(
            (
                float(relayout_rotation_estimate)
                + float(lt_rotation_estimate)
                + float(halo_slots) / BOOTSTRAP_HALO_SLOT_DIVISOR
                + float(tile_count)
            )
            / 64.0
        )
    )
    redundancy = 0.0 if int(core_slots) == 0 else float(stored_slots - core_slots) / float(core_slots)
    objective = (
        float(relayout_rotation_estimate) * RELAYOUT_ROTATION_WEIGHT
        + float(lt_rotation_estimate) * LT_ROTATION_WEIGHT
        + float(halo_slots) * HALO_SLOT_WEIGHT
        + float(tile_count) * TILE_WEIGHT
        + float(bootstrap_proxy) * BOOTSTRAP_PROXY_WEIGHT
    )
    return PolicyPlan(
        policy=str(policy),
        policy_label=POLICY_LABELS[str(policy)],
        metric_source="planner_estimate",
        relayouts=int(len(relayout_layouts)),
        halo_redundancy_ratio=float(redundancy),
        total_ciphertext_tiles=int(tile_count),
        stored_slots=int(stored_slots),
        relayout_rotation_estimate=int(relayout_rotation_estimate),
        lt_bsgs_rotation_estimate=int(lt_rotation_estimate),
        bootstrap_proxy=int(bootstrap_proxy),
        objective=float(objective),
        edge_layouts=tuple(edge_rows),
    )


def _align_add_inputs(dag: NetworkDAG, rows_by_edge: dict[str, dict[str, Any]], relayout_layouts: list[LayoutState], *, slots: int) -> None:
    for node in dag.topological_sort():
        module = dag.nodes[node].get("module")
        if type(module).__name__ != "Add":
            continue
        incoming = [f"{source}->{node}" for source in dag.predecessors(node)]
        layouts = [LayoutState(**rows_by_edge[edge_id]["selected_layout"]) for edge_id in incoming if edge_id in rows_by_edge]
        if len(layouts) < 2 or len({layout.key() for layout in layouts}) == 1:
            continue
        target = max(layouts, key=lambda item: (int(item.alpha + item.beta), int(item.stored_slots), int(item.stride)))
        for edge_id in incoming:
            if edge_id not in rows_by_edge:
                continue
            previous = LayoutState(**rows_by_edge[edge_id]["selected_layout"])
            if _same_layout(previous, target):
                continue
            rows_by_edge[edge_id]["selected_layout"] = target.to_dict()
            rows_by_edge[edge_id]["relayout"] = True
            rows_by_edge[edge_id]["relayout_reason"] = "add_input_alignment"
            rows_by_edge[edge_id]["lt_bsgs_rotation_estimate"] = int(
                rows_by_edge[edge_id]["lt_bsgs_rotation_estimate"]
            )
            relayout_layouts.append(target)


def _plan_fixed_max(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    global_alpha = max(int(edge.requirement.alpha) for edge in edges)
    global_beta = max(int(edge.requirement.beta) for edge in edges)
    rows: list[dict[str, Any]] = []
    for edge in edges:
        layout = _layout_for_shape(
            shape=edge.shape,
            gap=int(edge.compact.gap),
            alpha=int(global_alpha),
            beta=int(global_beta),
            stride=max(1, int(edge.requirement.stride)),
            slots=int(slots),
        )
        rows.append(_edge_row(edge, layout, relayout=False, relayout_reason="", lt_rotations=_lt_rotations(edge, layout)))
    rows_by_edge = {str(row["edge"]): row for row in rows}
    relayout_layouts: list[LayoutState] = []
    _align_add_inputs(dag, rows_by_edge, relayout_layouts, slots=int(slots))
    return _finalize_policy(policy="fixed_max", edge_rows=list(rows_by_edge.values()), relayout_layouts=relayout_layouts)


def _plan_eager(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    rows: list[dict[str, Any]] = []
    relayout_layouts: list[LayoutState] = []
    for edge in edges:
        layout = edge.requirement
        relayout = not _same_layout(layout, edge.compact)
        if bool(relayout):
            relayout_layouts.append(layout)
        rows.append(
            _edge_row(
                edge,
                layout,
                relayout=bool(relayout),
                relayout_reason="consumer_min_layout" if bool(relayout) else "",
                lt_rotations=_lt_rotations(edge, layout),
            )
        )
    rows_by_edge = {str(row["edge"]): row for row in rows}
    _align_add_inputs(dag, rows_by_edge, relayout_layouts, slots=int(slots))
    return _finalize_policy(policy="eager", edge_rows=list(rows_by_edge.values()), relayout_layouts=relayout_layouts)


def _plan_greedy(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    current_by_source: dict[str, LayoutState] = {}
    rows: list[dict[str, Any]] = []
    relayout_layouts: list[LayoutState] = []
    for edge in edges:
        current = current_by_source.get(edge.source, edge.compact)
        if current.covers(edge.requirement):
            layout = current
            relayout = False
            reason = ""
        else:
            layout = edge.requirement
            current_by_source[edge.source] = layout
            relayout = not _same_layout(layout, edge.compact)
            reason = "greedy_local_consumer_min" if bool(relayout) else ""
            if bool(relayout):
                relayout_layouts.append(layout)
        rows.append(_edge_row(edge, layout, relayout=bool(relayout), relayout_reason=reason, lt_rotations=_lt_rotations(edge, layout)))
    rows_by_edge = {str(row["edge"]): row for row in rows}
    _align_add_inputs(dag, rows_by_edge, relayout_layouts, slots=int(slots))
    return _finalize_policy(policy="greedy", edge_rows=list(rows_by_edge.values()), relayout_layouts=relayout_layouts)


def _source_candidate_rows(edges: Sequence[EdgeInfo], layouts: Sequence[LayoutState], *, relayout_reason: str) -> tuple[list[dict[str, Any]], list[LayoutState]]:
    rows: list[dict[str, Any]] = []
    relayout_layouts: list[LayoutState] = []
    for edge, layout in zip(edges, layouts):
        relayout = not _same_layout(layout, edge.compact)
        if bool(relayout):
            relayout_layouts.append(layout)
        rows.append(
            _edge_row(
                edge,
                layout,
                relayout=bool(relayout),
                relayout_reason=str(relayout_reason) if bool(relayout) else "",
                lt_rotations=_lt_rotations(edge, layout),
            )
        )
    return rows, relayout_layouts


@dataclass(frozen=True)
class _FrontierState:
    score: float
    live_layouts: tuple[tuple[str, LayoutState], ...]
    edge_rows: tuple[dict[str, Any], ...]
    relayout_layouts: tuple[LayoutState, ...]

    def live_dict(self) -> dict[str, LayoutState]:
        return {str(node): layout for node, layout in self.live_layouts}


def _frontier_key(live: dict[str, LayoutState]) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    return tuple(sorted((str(node), layout.key()) for node, layout in live.items()))


def _frontier_live_items(live: dict[str, LayoutState]) -> tuple[tuple[str, LayoutState], ...]:
    return tuple(sorted(((str(node), layout) for node, layout in live.items()), key=lambda item: item[0]))


def _dedupe_layouts(layouts: Iterable[LayoutState]) -> tuple[LayoutState, ...]:
    deduped: dict[tuple[int, int, int, int], LayoutState] = {}
    for layout in layouts:
        deduped.setdefault(layout.key(), layout)
    return tuple(deduped.values())


def _source_layout_candidates(edges: Sequence[EdgeInfo], *, global_alpha: int, global_beta: int, slots: int) -> tuple[LayoutState, ...]:
    if not edges:
        return ()
    edge = edges[0]
    local_need = _max_layout(edges, slots=int(slots))
    fixed_like = _layout_for_shape(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        alpha=int(global_alpha),
        beta=int(global_beta),
        stride=max(1, int(local_need.stride)),
        slots=int(slots),
    )
    return _dedupe_layouts((edge.compact, local_need, fixed_like))


def _incoming_non_add_rows(edge: EdgeInfo, source_layout: LayoutState) -> tuple[list[dict[str, Any]], list[LayoutState]]:
    if source_layout.covers(edge.requirement):
        return [
            _edge_row(
                edge,
                source_layout,
                relayout=False,
                relayout_reason="",
                lt_rotations=_lt_rotations(edge, source_layout),
            )
        ], []
    layout = edge.requirement
    return [
        _edge_row(
            edge,
            layout,
            relayout=True,
            relayout_reason="dp_state_consumer_min",
            lt_rotations=_lt_rotations(edge, layout),
        )
    ], [layout]


def _incoming_add_options(incoming: Sequence[EdgeInfo], live: dict[str, LayoutState]) -> tuple[tuple[list[dict[str, Any]], list[LayoutState]], ...]:
    source_layouts = [live[edge.source] for edge in incoming]
    candidates: list[LayoutState] = []
    if incoming:
        candidates.append(incoming[0].compact)
    if source_layouts:
        candidates.append(
            max(
                source_layouts,
                key=lambda layout: (
                    int(layout.alpha + layout.beta),
                    int(layout.stride),
                    int(layout.stored_slots),
                ),
            )
        )
    options: list[tuple[list[dict[str, Any]], list[LayoutState]]] = []
    for target_layout in _dedupe_layouts(candidates):
        rows: list[dict[str, Any]] = []
        relayouts: list[LayoutState] = []
        for edge in incoming:
            source_layout = live[edge.source]
            relayout = not _same_layout(source_layout, target_layout)
            if bool(relayout):
                relayouts.append(target_layout)
            rows.append(
                _edge_row(
                    edge,
                    target_layout,
                    relayout=bool(relayout),
                    relayout_reason="dp_add_input_alignment" if bool(relayout) else "",
                    lt_rotations=_lt_rotations(edge, target_layout),
                )
            )
        options.append((rows, relayouts))
    return tuple(options)


def _plan_dp(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    edges_by_source: dict[str, list[EdgeInfo]] = {}
    edges_by_target: dict[str, list[EdgeInfo]] = {}
    for edge in edges:
        edges_by_source.setdefault(edge.source, []).append(edge)
        edges_by_target.setdefault(edge.target, []).append(edge)
    topo = [str(node) for node in dag.topological_sort()]
    topo_index = {str(node): index for index, node in enumerate(topo)}
    last_consumer_index = {
        str(source): max(topo_index[str(edge.target)] for edge in source_edges)
        for source, source_edges in edges_by_source.items()
    }
    global_alpha = max(int(edge.requirement.alpha) for edge in edges)
    global_beta = max(int(edge.requirement.beta) for edge in edges)

    states: dict[tuple[tuple[str, tuple[int, int, int, int]], ...], _FrontierState] = {
        (): _FrontierState(score=0.0, live_layouts=(), edge_rows=(), relayout_layouts=())
    }
    for node in topo:
        incoming = tuple(edges_by_target.get(str(node), ()))
        outgoing = tuple(edges_by_source.get(str(node), ()))
        node_index = int(topo_index[str(node)])
        next_states: dict[tuple[tuple[str, tuple[int, int, int, int]], ...], _FrontierState] = {}
        for state in states.values():
            live = state.live_dict()
            if any(edge.source not in live for edge in incoming):
                continue
            if incoming and type(dag.nodes[node].get("module")).__name__ == "Add":
                incoming_options = _incoming_add_options(incoming, live)
            else:
                rows: list[dict[str, Any]] = []
                relayouts: list[LayoutState] = []
                for edge in incoming:
                    edge_rows, edge_relayouts = _incoming_non_add_rows(edge, live[edge.source])
                    rows.extend(edge_rows)
                    relayouts.extend(edge_relayouts)
                incoming_options = ((rows, relayouts),)

            for incoming_rows, incoming_relayouts in incoming_options:
                next_live = dict(live)
                for source in list(next_live):
                    if int(last_consumer_index.get(str(source), -1)) <= int(node_index):
                        next_live.pop(str(source), None)
                output_candidates = _source_layout_candidates(
                    outgoing,
                    global_alpha=int(global_alpha),
                    global_beta=int(global_beta),
                    slots=int(slots),
                )
                if not output_candidates:
                    output_candidates = (None,)
                for output_layout in output_candidates:
                    candidate_live = dict(next_live)
                    if output_layout is not None:
                        candidate_live[str(node)] = output_layout
                    candidate_rows = tuple(list(state.edge_rows) + list(incoming_rows))
                    candidate_relayouts = tuple(list(state.relayout_layouts) + list(incoming_relayouts))
                    candidate_score = float(state.score) + _policy_linear_cost(incoming_rows, incoming_relayouts)
                    key = _frontier_key(candidate_live)
                    existing = next_states.get(key)
                    if existing is None or (
                        candidate_score,
                        len(candidate_relayouts),
                        _halo_slots_for_rows(candidate_rows),
                    ) < (
                        float(existing.score),
                        len(existing.relayout_layouts),
                        _halo_slots_for_rows(existing.edge_rows),
                    ):
                        next_states[key] = _FrontierState(
                            score=float(candidate_score),
                            live_layouts=_frontier_live_items(candidate_live),
                            edge_rows=candidate_rows,
                            relayout_layouts=candidate_relayouts,
                        )
        states = next_states
    if not states:
        raise RuntimeError("layout policy DP found no legal state")
    best_state = min(
        states.values(),
        key=lambda state: (
            _policy_linear_cost(state.edge_rows, state.relayout_layouts),
            len(state.relayout_layouts),
            _halo_slots_for_rows(state.edge_rows),
        ),
    )
    rows = [{**dict(row), "dp_state_planned": True} for row in best_state.edge_rows]
    rows.sort(key=lambda row: (topo_index.get(str(row["source"]), 10**9), topo_index.get(str(row["target"]), 10**9)))
    return _finalize_policy(policy="dp", edge_rows=rows, relayout_layouts=list(best_state.relayout_layouts))


def _plan_source_local_dp(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    by_source: dict[str, list[EdgeInfo]] = {}
    for edge in edges:
        by_source.setdefault(edge.source, []).append(edge)

    selected_rows: list[dict[str, Any]] = []
    selected_relayout_layouts: list[LayoutState] = []
    for source, source_edges in by_source.items():
        eager_rows, eager_relayouts = _source_candidate_rows(
            source_edges,
            [edge.requirement for edge in source_edges],
            relayout_reason="dp_consumer_min",
        )
        max_layout = _max_layout(source_edges, slots=int(slots))
        shared_rows, shared_relayouts = _source_candidate_rows(
            source_edges,
            [max_layout for _edge in source_edges],
            relayout_reason="dp_source_shared_layout",
        )
        eager_plan = _finalize_policy(policy="dp", edge_rows=eager_rows, relayout_layouts=eager_relayouts)
        shared_plan = _finalize_policy(policy="dp", edge_rows=shared_rows, relayout_layouts=shared_relayouts[:1] if shared_relayouts else [])
        if shared_plan.objective < eager_plan.objective:
            selected_rows.extend(shared_rows)
            if shared_relayouts:
                selected_relayout_layouts.append(max_layout)
        else:
            selected_rows.extend(eager_rows)
            selected_relayout_layouts.extend(eager_relayouts)

    topo_index = {str(node): index for index, node in enumerate(dag.topological_sort())}
    selected_rows.sort(key=lambda row: (topo_index.get(str(row["source"]), 10**9), topo_index.get(str(row["target"]), 10**9)))
    rows_by_edge = {str(row["edge"]): row for row in selected_rows}
    _align_add_inputs(dag, rows_by_edge, selected_relayout_layouts, slots=int(slots))
    return _finalize_policy(policy="dp", edge_rows=list(rows_by_edge.values()), relayout_layouts=selected_relayout_layouts)


def plan_policy(dag: NetworkDAG, edges: Sequence[EdgeInfo], policy: str, *, slots: int = DEFAULT_SLOTS) -> PolicyPlan:
    normalized = normalize_policy(policy)
    if normalized == "fixed_max":
        return _plan_fixed_max(dag, edges, slots=int(slots))
    if normalized == "eager":
        return _plan_eager(dag, edges, slots=int(slots))
    if normalized == "greedy":
        return _plan_greedy(dag, edges, slots=int(slots))
    if normalized == "dp":
        return _plan_dp(dag, edges, slots=int(slots))
    raise AssertionError(f"unreachable policy {policy!r}")


def build_planner_ablation(
    *,
    network: str = "u22_64_base32",
    policies: Sequence[str] = ("fixed_max", "eager", "greedy", "dp"),
    slots: int = DEFAULT_SLOTS,
) -> dict[str, Any]:
    spec = network_spec(str(network))
    dag = build_u22_dag(spec)
    edges = build_edge_infos(dag, slots=int(slots))
    normalized_policies = normalize_policies(policies)
    plans = [plan_policy(dag, edges, policy, slots=int(slots)) for policy in normalized_policies]
    fixed_objective = next((plan.objective for plan in plans if plan.policy == "fixed_max"), None)
    if fixed_objective is None:
        fixed_objective = plan_policy(dag, edges, "fixed_max", slots=int(slots)).objective
    with_speedups = []
    for plan in plans:
        speedup = None if float(plan.objective) == 0.0 else float(fixed_objective) / float(plan.objective)
        with_speedups.append(
            PolicyPlan(
                **{**plan.__dict__, "speedup_vs_fixed_max": speedup}
            )
        )
    return {
        "status": "ok",
        "network": spec.network,
        "dataset": spec.dataset,
        "base_channels": int(spec.base_channels),
        "slots": int(slots),
        "graph": {
            "node_count": int(len(dag.nodes)),
            "edge_count": int(len(dag.edges)),
            "nodes": [str(node) for node in dag.topological_sort()],
            "edges": [edge.edge_id for edge in edges],
        },
        "policies": [plan.to_dict() for plan in with_speedups],
    }


def _policy_add_inputs_aligned(policy_row: dict[str, Any]) -> bool:
    edge_rows = list(policy_row.get("edge_layouts", []))
    for add_node in ("add4", "add3", "add2", "add1"):
        incoming = [row for row in edge_rows if str(row.get("target")) == str(add_node)]
        if len(incoming) != 2:
            continue
        keys = {
            (
                int(row["selected_layout"]["alpha"]),
                int(row["selected_layout"]["beta"]),
                int(row["selected_layout"]["stride"]),
                int(row["selected_layout"]["gap"]),
            )
            for row in incoming
        }
        if len(keys) != 1:
            return False
    return True


def _dice_against_reference(left_logits: torch.Tensor, right_logits: torch.Tensor) -> float:
    left = (torch.sigmoid(left_logits.detach().cpu().to(dtype=torch.float32)) >= 0.5).to(dtype=torch.float32)
    right = (torch.sigmoid(right_logits.detach().cpu().to(dtype=torch.float32)) >= 0.5).to(dtype=torch.float32)
    dims = tuple(range(1, left.dim()))
    intersection = (left * right).sum(dim=dims)
    total = left.sum(dim=dims) + right.sum(dim=dims)
    return float(((2.0 * intersection + 1.0e-6) / (total + 1.0e-6)).mean().item())


def run_non_ckks_layout_simulation(
    planner_payload: dict[str, Any],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a clear PyTorch sanity pass while exercising planner policy metadata.

    This deliberately does not initialize an Orion scheme or backend. It is a
    shape/layout-policy simulation only: each policy reuses the same clear U22
    forward and verifies that planner-selected Add inputs remain layout-aligned.
    """

    spec = network_spec(str(planner_payload.get("network", "u22_64_base32")))
    torch.manual_seed(int(seed))
    model = UNet22(dataset=str(spec.dataset), base_channels=int(spec.base_channels))
    model.eval()
    x = torch.randn((1, int(spec.input_channels), int(spec.image_size), int(spec.image_size)), dtype=torch.float32)
    with torch.no_grad():
        reference = model(x).detach().cpu()
    policy_rows: dict[str, dict[str, Any]] = {}
    for policy in list(planner_payload.get("policies", [])):
        started = time.perf_counter()
        with torch.no_grad():
            observed = model(x).detach().cpu()
        forward_s = float(time.perf_counter() - started)
        diff = (observed.to(dtype=torch.float32) - reference.to(dtype=torch.float32)).abs()
        policy_rows[str(policy.get("policy"))] = {
            "status": "ok",
            "metric_source": "non_ckks_pytorch_sim",
            "forward_s": float(forward_s),
            "mae": float(diff.mean().item()),
            "max_abs": float(diff.max().item()),
            "dice": _dice_against_reference(observed, reference),
            "layout_alignment_ok": bool(_policy_add_inputs_aligned(policy)),
        }
    return {
        "status": "ok",
        "kind": "non_ckks_pytorch_layout_sim",
        "seed": int(seed),
        "input_shape": [1, int(spec.input_channels), int(spec.image_size), int(spec.image_size)],
        "reference_output_shape": [int(value) for value in reference.shape],
        "policies": policy_rows,
    }


Runner = Callable[..., subprocess.CompletedProcess[str]]


def run_runtime_anchor(
    *,
    network: str,
    backend: str,
    cache_root: Path,
    compile_timeout_s: int,
    python: Path | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    spec = network_spec(str(network))
    if spec.network != "u22_64_base32":
        return {
            "status": "skipped",
            "reason": "runtime anchor is only enabled for u22_64_base32 in v1",
        }
    repo_root = Path(__file__).resolve().parents[2]
    python_bin = Path(sys.executable if python is None else python)
    out_dir = Path(cache_root) / "layout_policy_ablation" / f"{spec.network}_{backend}_dp_provider_anchor"
    cmd = [
        str(python_bin),
        str(repo_root / "tools" / "run_unet22_medical_fhe_figure.py"),
        "--dataset",
        str(spec.dataset),
        "--backend",
        str(backend),
        "--mode",
        "provider",
        "--provider-mode",
        str(spec.provider_mode),
        "--sample-index",
        "0",
        "--sample-count",
        "1",
        "--seed",
        "0",
        "--out-dir",
        str(out_dir),
    ]
    started = time.perf_counter()
    try:
        completed = runner(
            cmd,
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(compile_timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "compile_timeout",
            "reason": f"runtime anchor exceeded {int(compile_timeout_s)}s",
            "elapsed_s": float(time.perf_counter() - started),
            "out_dir": str(out_dir),
            "stdout_tail": str(exc.stdout or "")[-1000:],
        }
    elapsed = float(time.perf_counter() - started)
    if int(completed.returncode) != 0:
        return {
            "status": "failed",
            "reason": f"runtime anchor exited {int(completed.returncode)}",
            "elapsed_s": float(elapsed),
            "out_dir": str(out_dir),
            "stdout_tail": str(completed.stdout or "")[-1000:],
        }
    payloads = sorted(Path(out_dir).glob("*_fhe_figure.json"))
    if not payloads:
        return {
            "status": "missing_payload",
            "reason": "runtime anchor completed but no *_fhe_figure.json was found",
            "elapsed_s": float(elapsed),
            "out_dir": str(out_dir),
        }
    payload = json.loads(payloads[-1].read_text(encoding="utf-8"))
    samples = list(payload.get("samples", []))
    he_forward = payload.get("timing_s", {}).get("he_forward_total")
    if he_forward is None and samples:
        he_forward = sum(float(sample.get("timing_s", {}).get("he_forward", 0.0)) for sample in samples)
    mae = None
    dice = None
    if samples:
        mae = samples[0].get("fhe_vs_pytorch_logits", {}).get("mae")
        dice = samples[0].get("fhe_vs_reference", {}).get("dice")
    return {
        "status": str(payload.get("status", "ok")),
        "reason": "",
        "elapsed_s": float(elapsed),
        "out_dir": str(out_dir),
        "payload_path": str(payloads[-1]),
        "he_forward_s": None if he_forward is None else float(he_forward),
        "mae": None if mae is None else float(mae),
        "dice": None if dice is None else float(dice),
        "bootstrap_count": None,
    }


def attach_runtime_anchor(planner_payload: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    payload = dict(planner_payload)
    policy_rows = []
    for row in list(payload.get("policies", [])):
        updated = dict(row)
        if str(updated.get("policy")) == "dp":
            updated["metric_source"] = "planner_estimate+runtime_anchor"
            updated["runtime_status"] = str(anchor.get("status", "unknown"))
            updated["runtime_reason"] = str(anchor.get("reason", ""))
            for key in ("bootstrap_count", "he_forward_s", "mae", "dice"):
                value = anchor.get(key)
                if value is not None:
                    updated[key] = value
        policy_rows.append(updated)
    payload["policies"] = policy_rows
    payload["runtime_anchor"] = dict(anchor)
    return payload


def attach_non_ckks_simulation(planner_payload: dict[str, Any], simulation: dict[str, Any]) -> dict[str, Any]:
    payload = dict(planner_payload)
    sim_by_policy = dict(simulation.get("policies", {}))
    policy_rows = []
    for row in list(payload.get("policies", [])):
        updated = dict(row)
        policy = str(updated.get("policy", ""))
        sim_row = dict(sim_by_policy.get(policy, {}))
        if sim_row:
            updated["metric_source"] = f"{updated.get('metric_source', 'planner_estimate')}+non_ckks_sim"
            updated["runtime_status"] = "non_ckks_sim_ok" if bool(sim_row.get("layout_alignment_ok", False)) else "non_ckks_sim_failed"
            updated["runtime_reason"] = "clear PyTorch simulation; no CKKS scheme initialized"
            updated["mae"] = float(sim_row.get("mae", 0.0))
            updated["dice"] = float(sim_row.get("dice", 0.0))
            updated["max_abs"] = float(sim_row.get("max_abs", 0.0))
            updated["non_ckks_forward_s"] = float(sim_row.get("forward_s", 0.0))
            updated["layout_alignment_ok"] = bool(sim_row.get("layout_alignment_ok", False))
        policy_rows.append(updated)
    payload["policies"] = policy_rows
    payload["non_ckks_simulation"] = dict(simulation)
    return payload
