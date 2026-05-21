from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
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
RELAYOUT_MASK_MULT_WEIGHT = 1.0
LT_ROTATION_WEIGHT = 8.0
HALO_SLOT_WEIGHT = 0.0
TILE_WEIGHT = 4.0
BOOTSTRAP_PROXY_WEIGHT = 64.0
BOOTSTRAP_HALO_SLOT_DIVISOR = math.inf
RELAYOUT_DEPTH_WEIGHT = 4096.0
RELAYOUT_NODE_STACK_DEPTH_WEIGHT = 65536.0
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
PROVIDER_PRESSURE_SUMMARY_KEYS = (
    "provider_region_count",
    "native_halo_provider_region_count",
    "relayout_lt_region_count",
    "relayout_edge_count",
    "output_relayout_edge_count",
    "relayout_kernel_count",
    "relayout_rotation_count",
    "relayout_mask_mult_count",
    "relayout_sparse_lt_count",
    "provider_input_block_cols",
    "compact_input_block_cols",
    "halo_input_block_cols",
    "extra_input_block_cols_vs_compact",
    "hybrid_pair_count",
    "hybrid_pair_rejected_count",
    "hybrid_pair_layout_strict_pair_count",
    "hybrid_pair_layout_covered_output_count",
    "diagonal_key_set_mismatch_count",
    "group_union_rotation_count",
    "transform_sum_rotation_count",
)


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
    output_shape: tuple[int, int, int, int] | None = None
    output_fhe_shape: tuple[int, int, int, int] | None = None
    kernel_size: tuple[int, int] = (1, 1)
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    output_padding: tuple[int, int] = (0, 0)
    groups: int = 1
    input_channels: int = 0
    output_channels: int = 0
    slots: int = DEFAULT_SLOTS

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
    relayout_mask_mult_estimate: int
    relayout_depth_estimate: int
    lt_bsgs_rotation_estimate: int
    compact_fallback_penalty_estimate: int
    bootstrap_proxy: int
    objective: float
    edge_layouts: tuple[dict[str, Any], ...]
    node_layouts: tuple[dict[str, Any], ...] = ()
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
            "relayout_mask_mult_estimate": int(self.relayout_mask_mult_estimate),
            "relayout_depth_estimate": int(self.relayout_depth_estimate),
            "lt_bsgs_rotation_estimate": int(self.lt_bsgs_rotation_estimate),
            "compact_fallback_penalty_estimate": int(self.compact_fallback_penalty_estimate),
            "bootstrap_proxy": int(self.bootstrap_proxy),
            "bootstrap_count": "" if self.bootstrap_count is None else int(self.bootstrap_count),
            "he_forward_s": "" if self.he_forward_s is None else float(self.he_forward_s),
            "mae": "" if self.mae is None else float(self.mae),
            "dice": "" if self.dice is None else float(self.dice),
            "speedup_vs_fixed_max": "" if self.speedup_vs_fixed_max is None else float(self.speedup_vs_fixed_max),
            "runtime_status": self.runtime_status,
            "runtime_reason": self.runtime_reason,
            "objective": float(self.objective),
            **{key: "" for key in PROVIDER_PRESSURE_SUMMARY_KEYS},
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.summary_row()
        payload["policy_label"] = self.policy_label
        payload["edge_layouts"] = [dict(row) for row in self.edge_layouts]
        payload["node_layouts"] = [dict(row) for row in self.node_layouts]
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
    if normalized == "u22_64_base8":
        return NetworkSpec(
            network="u22_64_base8",
            dataset="montgomery_lung_64",
            image_size=64,
            input_channels=1,
            base_channels=8,
            provider_mode="u22_64_base8",
        )
    if normalized == "u22_128_base32":
        return NetworkSpec(
            network="u22_128_base32",
            dataset="kvasir_polyp_256",
            image_size=128,
            input_channels=3,
            base_channels=32,
            provider_mode="u22_128_base32",
        )
    if normalized == "u22_128_base8":
        return NetworkSpec(
            network="u22_128_base8",
            dataset="kvasir_polyp_256",
            image_size=128,
            input_channels=3,
            base_channels=8,
            provider_mode="u22_128_base8",
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
    if normalized == "u22_256_base8":
        return NetworkSpec(
            network="u22_256_base8",
            dataset="kvasir_polyp_256",
            image_size=256,
            input_channels=3,
            base_channels=8,
            provider_mode="u22_256_base8",
        )
    raise ValueError(
        "layout-policy ablation supports u22_64_base8/u22_64_base32, "
        "u22_128_base8/u22_128_base32, and u22_256_base8/u22_256_base32"
    )


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


def _pair_tuple(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return int(value), int(value)
    items = tuple(int(item) for item in value)
    if len(items) == 1:
        return int(items[0]), int(items[0])
    if len(items) >= 2:
        return int(items[0]), int(items[1])
    return default


def _consumer_output_shapes(
    module: Any | None,
    fallback_shape: tuple[int, int, int, int],
    fallback_fhe_shape: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    if module is None:
        return fallback_shape, fallback_fhe_shape
    output_shape = fallback_shape
    output_fhe_shape = fallback_fhe_shape
    if hasattr(module, "output_shape"):
        maybe_shape = tuple(int(item) for item in getattr(module, "output_shape"))
        if len(maybe_shape) == 4:
            output_shape = maybe_shape
    if hasattr(module, "fhe_output_shape"):
        maybe_fhe_shape = tuple(int(item) for item in getattr(module, "fhe_output_shape"))
        if len(maybe_fhe_shape) == 4:
            output_fhe_shape = maybe_fhe_shape
    return output_shape, output_fhe_shape


def _consumer_op_params(
    module: Any | None,
    *,
    input_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    if module is None:
        return {
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "output_padding": (0, 0),
            "groups": 1,
            "input_channels": int(input_shape[1]),
            "output_channels": int(input_shape[1] if output_shape is None else output_shape[1]),
        }
    return {
        "kernel_size": _pair_tuple(getattr(module, "kernel_size", (1, 1)), (1, 1)),
        "stride": _pair_tuple(getattr(module, "stride", (1, 1)), (1, 1)),
        "padding": _pair_tuple(getattr(module, "padding", (0, 0)), (0, 0)),
        "dilation": _pair_tuple(getattr(module, "dilation", (1, 1)), (1, 1)),
        "output_padding": _pair_tuple(getattr(module, "output_padding", (0, 0)), (0, 0)),
        "groups": max(1, int(getattr(module, "groups", 1) or 1)),
        "input_channels": int(getattr(module, "in_channels", input_shape[1])),
        "output_channels": int(getattr(module, "out_channels", input_shape[1] if output_shape is None else output_shape[1])),
    }


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
    for successor in dag.successors(target):
        successor_module = dag.nodes[successor].get("module")
        if successor_module is not None and hasattr(successor_module, "input_shape"):
            shape = _shape_tuple(getattr(successor_module, "input_shape"))
            fhe_shape = _shape_tuple(getattr(successor_module, "fhe_input_shape", getattr(successor_module, "input_shape")))
            gap = int(getattr(successor_module, "input_gap", 1))
            return shape, fhe_shape, gap
    for predecessor in dag.predecessors(source):
        predecessor_module = dag.nodes[predecessor].get("module")
        if predecessor_module is not None and hasattr(predecessor_module, "output_shape"):
            shape = _shape_tuple(getattr(predecessor_module, "output_shape"))
            fhe_shape = _shape_tuple(getattr(predecessor_module, "fhe_output_shape", getattr(predecessor_module, "output_shape")))
            gap = int(getattr(predecessor_module, "output_gap", 1))
            return shape, fhe_shape, gap
    raise ValueError(f"cannot infer edge shape for {source}->{target}")


def build_edge_infos(dag: NetworkDAG, *, slots: int = DEFAULT_SLOTS) -> tuple[EdgeInfo, ...]:
    edges: list[EdgeInfo] = []
    for source, target in dag.edges:
        target_module = dag.nodes[target].get("module")
        shape, fhe_shape, gap = _edge_shapes(dag, str(source), str(target))
        alpha, beta, stride, op_kind, lt_base = _consumer_requirement(target_module)
        output_shape, output_fhe_shape = _consumer_output_shapes(target_module, shape, fhe_shape)
        op_params = _consumer_op_params(target_module, input_shape=shape, output_shape=output_shape)
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
                output_shape=output_shape,
                output_fhe_shape=output_fhe_shape,
                kernel_size=op_params["kernel_size"],
                stride=op_params["stride"],
                padding=op_params["padding"],
                dilation=op_params["dilation"],
                output_padding=op_params["output_padding"],
                groups=int(op_params["groups"]),
                input_channels=int(op_params["input_channels"]),
                output_channels=int(op_params["output_channels"]),
                slots=int(slots),
            ),
        )
    topo_index = {str(node): index for index, node in enumerate(dag.topological_sort())}
    return tuple(sorted(edges, key=lambda edge: (topo_index.get(edge.source, 10**9), topo_index.get(edge.target, 10**9))))


def _same_layout(left: LayoutState, right: LayoutState) -> bool:
    return left.key() == right.key()


def _same_physical_layout(left: LayoutState, right: LayoutState) -> bool:
    return (
        int(left.alpha) == int(right.alpha)
        and int(left.beta) == int(right.beta)
        and int(left.gap) == int(right.gap)
        and int(left.tile_count) == int(right.tile_count)
    )


def _side_after_downsample(side: int, *, consume: int, stride: int) -> int:
    return max(0, int(int(side) - int(consume)) // max(1, int(stride)))


def _conv_halo_consume(module: Any | None) -> int:
    if module is None:
        return 0
    kernel = _pair_tuple(getattr(module, "kernel_size", (1, 1)), (1, 1))
    padding = _pair_tuple(getattr(module, "padding", (0, 0)), (0, 0))
    if max(int(value) for value in padding) > 0:
        return max(0, max(int(value) for value in kernel) // 2)
    return max(0, max(int(value) for value in kernel) - 1)


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


def _edge_row(
    edge: EdgeInfo,
    layout: LayoutState,
    *,
    relayout: bool,
    relayout_reason: str,
    lt_rotations: int | None = None,
    layout_mode: str = "halo_local",
) -> dict[str, Any]:
    if str(layout_mode) == "compact_global_fallback":
        raise ValueError("compact global fallback is not a valid halo-local layout-policy edge")
    if lt_rotations is None:
        lt_rotations = _lt_rotations(edge, layout)
    lt_stats = _lt_rotation_stats(edge, layout)
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
        "relayout_rotation_estimate": int(_relayout_rotations((layout,)) if bool(relayout) else 0),
        "relayout_mask_mult_estimate": int(_relayout_mask_mults((layout,)) if bool(relayout) else 0),
        "lt_bsgs_rotation_estimate": int(lt_rotations),
        "compact_fallback_penalty_estimate": 0,
        "lt_bsgs_group_count_estimate": int(lt_stats["bsgs_groups"]),
        "lt_transform_count_estimate": int(lt_stats["transforms"]),
        "lt_baby_rotation_estimate": int(lt_stats["baby_rotations"]),
        "lt_giant_rotation_estimate": int(lt_stats["giant_rotations"]),
        "lt_input_channel_multiplier": int(lt_stats["input_channel_multiplier"]),
        "lt_output_channel_multiplier": int(lt_stats["output_channel_multiplier"]),
        "layout_mode": str(layout_mode),
    }


def _layout_one_channel_physical_shape(
    *,
    clear_shape: tuple[int, int, int, int],
    gap: int,
    alpha: int,
    beta: int,
) -> tuple[int, int]:
    _n, _c, height, width = clear_shape
    physical_h = int(height) * int(gap) + int(alpha + beta) * int(gap)
    physical_w = int(width) * int(gap)
    return max(1, int(physical_h)), max(1, int(physical_w))


def _one_channel_block(slot_index: int, *, slots: int) -> tuple[int, int]:
    return int(slot_index) // int(slots), int(slot_index) % int(slots)


@lru_cache(maxsize=None)
def _one_channel_lt_groups_cached(
    *,
    op_kind: str,
    input_h: int,
    input_w: int,
    output_h: int,
    output_w: int,
    input_phys_h: int,
    input_phys_w: int,
    output_phys_h: int,
    output_phys_w: int,
    input_gap: int,
    output_gap: int,
    alpha: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    slots: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    by_input_block: dict[int, dict[int, set[int]]] = {}

    def add_mapping(in_h: int, in_w: int, out_h: int, out_w: int) -> None:
        if not (0 <= int(in_h) < int(input_h) and 0 <= int(in_w) < int(input_w)):
            return
        if not (0 <= int(out_h) < int(output_h) and 0 <= int(out_w) < int(output_w)):
            return
        in_ph = (int(in_h) + int(alpha)) * int(input_gap)
        in_pw = int(in_w) * int(input_gap)
        out_ph = int(out_h) * int(output_gap)
        out_pw = int(out_w) * int(output_gap)
        if not (0 <= int(in_ph) < int(input_phys_h) and 0 <= int(in_pw) < int(input_phys_w)):
            return
        if not (0 <= int(out_ph) < int(output_phys_h) and 0 <= int(out_pw) < int(output_phys_w)):
            return
        in_index = int(in_ph) * int(input_phys_w) + int(in_pw)
        out_index = int(out_ph) * int(output_phys_w) + int(out_pw)
        in_block, in_slot = _one_channel_block(in_index, slots=int(slots))
        out_block, out_slot = _one_channel_block(out_index, slots=int(slots))
        diagonal = (int(in_slot) - int(out_slot)) % int(slots)
        if int(diagonal) == 0:
            return
        output_groups = by_input_block.setdefault(int(in_block), {})
        output_groups.setdefault(int(out_block), set()).add(int(diagonal))

    if str(op_kind) in {"conv2d", "avgpool2d"}:
        for out_h in range(int(output_h)):
            base_h = int(out_h) * int(stride_h) - int(pad_h)
            for out_w in range(int(output_w)):
                base_w = int(out_w) * int(stride_w) - int(pad_w)
                for kernel_y in range(int(kernel_h)):
                    in_h = int(base_h) + int(kernel_y) * int(dilation_h)
                    for kernel_x in range(int(kernel_w)):
                        in_w = int(base_w) + int(kernel_x) * int(dilation_w)
                        add_mapping(in_h, in_w, out_h, out_w)
    elif str(op_kind) == "conv_transpose2d":
        for in_h in range(int(input_h)):
            base_h = int(in_h) * int(stride_h) - int(pad_h)
            for in_w in range(int(input_w)):
                base_w = int(in_w) * int(stride_w) - int(pad_w)
                for kernel_y in range(int(kernel_h)):
                    out_h = int(base_h) + int(kernel_y) * int(dilation_h)
                    for kernel_x in range(int(kernel_w)):
                        out_w = int(base_w) + int(kernel_x) * int(dilation_w)
                        add_mapping(in_h, in_w, out_h, out_w)

    return tuple(
        tuple(
            tuple(sorted(int(diagonal) for diagonal in diagonals))
            for _output_block, diagonals in sorted(output_groups.items())
            if diagonals
        )
        for _input_block, output_groups in sorted(by_input_block.items())
        if output_groups
    )


def _powers_of_two_below_slots(slots: int) -> tuple[int, ...]:
    values: list[int] = []
    n1 = 1
    while int(n1) < int(slots):
        values.append(int(n1))
        n1 <<= 1
    return tuple(values or (1,))


def _bsgs_index(diag_indices: Iterable[int], *, slots: int, n1: int) -> tuple[set[int], set[int]]:
    rot_n1: set[int] = set()
    rot_n2: set[int] = set()
    slots = max(1, int(slots))
    n1 = max(1, int(n1))
    for value in diag_indices:
        rot = int(value) % int(slots)
        idx_n1 = int(((int(rot) // int(n1)) * int(n1)) % int(slots))
        idx_n2 = int(int(rot) % int(n1))
        rot_n1.add(int(idx_n1))
        rot_n2.add(int(idx_n2))
    return rot_n1, rot_n2


def _shared_bsgs_group_cost(
    diag_sets: Sequence[Sequence[int]],
    *,
    slots: int,
    repeated_transform_count: int,
) -> dict[str, int]:
    base_sets = [tuple(sorted({int(value) % int(slots) for value in diag_set})) for diag_set in diag_sets if diag_set]
    repeat = max(1, int(repeated_transform_count))
    if not base_sets:
        return {"rotations": 0, "baby_rotations": 0, "giant_rotations": 0, "n1": 1, "transforms": 0}

    best: dict[str, int] | None = None
    for n1 in _powers_of_two_below_slots(int(slots)):
        shared_baby: set[int] = set()
        giant_total = 0
        for diag_set in base_sets:
            rot_n1, rot_n2 = _bsgs_index(diag_set, slots=int(slots), n1=int(n1))
            shared_baby.update(int(value) for value in rot_n2 if int(value) != 0)
            giant_total += int(repeat) * sum(1 for value in rot_n1 if int(value) != 0)
        rotations = int(len(shared_baby) + int(giant_total))
        candidate = {
            "rotations": int(rotations),
            "baby_rotations": int(len(shared_baby)),
            "giant_rotations": int(giant_total),
            "n1": int(n1),
            "transforms": int(len(base_sets) * int(repeat)),
        }
        if best is None or (int(candidate["rotations"]), -int(candidate["n1"])) <= (
            int(best["rotations"]),
            -int(best["n1"]),
        ):
            best = candidate
    assert best is not None
    return best


def _lt_channel_multipliers(edge: EdgeInfo) -> tuple[int, int]:
    if str(edge.op_kind) in {"add", "input"}:
        return 0, 0
    input_channels = max(1, int(edge.input_channels or edge.shape[1]))
    if str(edge.op_kind) == "avgpool2d":
        return int(input_channels), 1
    if str(edge.op_kind) in {"conv2d", "conv_transpose2d"}:
        groups = max(1, int(edge.groups or 1))
        output_channels = max(1, int(edge.output_channels or (edge.output_shape or edge.shape)[1]))
        return int(input_channels), max(1, int(output_channels) // int(groups))
    return 0, 0


def _output_gap_for_edge(edge: EdgeInfo) -> int:
    if edge.output_fhe_shape is None or edge.output_shape is None:
        return max(1, int(edge.compact.gap))
    clear_h = max(1, int(edge.output_shape[2]))
    clear_w = max(1, int(edge.output_shape[3]))
    fhe_h = max(1, int(edge.output_fhe_shape[2]))
    fhe_w = max(1, int(edge.output_fhe_shape[3]))
    return max(1, min(max(1, int(round(fhe_h / clear_h))), max(1, int(round(fhe_w / clear_w)))))


def _lt_rotation_stats(edge: EdgeInfo, layout: LayoutState) -> dict[str, int]:
    if str(edge.op_kind) in {"add", "input"}:
        return {
            "bsgs_groups": 0,
            "transforms": 0,
            "baby_rotations": 0,
            "giant_rotations": 0,
            "input_channel_multiplier": 0,
            "output_channel_multiplier": 0,
            "rotations": 0,
        }
    if edge.output_shape is None or edge.output_fhe_shape is None:
        return {
            "bsgs_groups": 0,
            "transforms": 0,
            "baby_rotations": 0,
            "giant_rotations": 0,
            "input_channel_multiplier": 0,
            "output_channel_multiplier": 0,
            "rotations": 0,
        }
    input_phys_h, input_phys_w = _layout_one_channel_physical_shape(
        clear_shape=edge.shape,
        gap=int(layout.gap),
        alpha=int(layout.alpha),
        beta=int(layout.beta),
    )
    output_gap = _output_gap_for_edge(edge)
    one_channel_groups = _one_channel_lt_groups_cached(
        op_kind=str(edge.op_kind),
        input_h=int(edge.shape[2]),
        input_w=int(edge.shape[3]),
        output_h=int(edge.output_shape[2]),
        output_w=int(edge.output_shape[3]),
        input_phys_h=int(input_phys_h),
        input_phys_w=int(input_phys_w),
        output_phys_h=int(edge.output_fhe_shape[2]),
        output_phys_w=int(edge.output_fhe_shape[3]),
        input_gap=int(layout.gap),
        output_gap=int(output_gap),
        alpha=int(layout.alpha),
        kernel_h=int(edge.kernel_size[0]),
        kernel_w=int(edge.kernel_size[1]),
        stride_h=int(edge.stride[0]),
        stride_w=int(edge.stride[1]),
        pad_h=int(edge.padding[0]),
        pad_w=int(edge.padding[1]),
        dilation_h=int(edge.dilation[0]),
        dilation_w=int(edge.dilation[1]),
        slots=int(edge.slots),
    )
    input_multiplier, output_multiplier = _lt_channel_multipliers(edge)
    one_channel_bsgs_groups = 0
    one_channel_transforms = 0
    one_channel_baby = 0
    one_channel_giant = 0
    one_channel_rotations = 0
    for group in one_channel_groups:
        cost = _shared_bsgs_group_cost(
            group,
            slots=int(edge.slots),
            repeated_transform_count=int(output_multiplier),
        )
        one_channel_bsgs_groups += 1
        one_channel_transforms += int(cost["transforms"])
        one_channel_baby += int(cost["baby_rotations"])
        one_channel_giant += int(cost["giant_rotations"])
        one_channel_rotations += int(cost["rotations"])
    return {
        "bsgs_groups": int(one_channel_bsgs_groups) * int(input_multiplier),
        "transforms": int(one_channel_transforms) * int(input_multiplier),
        "baby_rotations": int(one_channel_baby) * int(input_multiplier),
        "giant_rotations": int(one_channel_giant) * int(input_multiplier),
        "input_channel_multiplier": int(input_multiplier),
        "output_channel_multiplier": int(output_multiplier),
        "rotations": int(one_channel_rotations) * int(input_multiplier),
    }


def _lt_rotations(edge: EdgeInfo, layout: LayoutState) -> int:
    return int(_lt_rotation_stats(edge, layout)["rotations"])


def _relayout_halo_side_count(layout: LayoutState) -> int:
    return int(max(1, int(layout.tile_count)) * (int(int(layout.alpha) > 0) + int(int(layout.beta) > 0)))


def _relayout_rotations(layouts: Iterable[LayoutState]) -> int:
    return int(sum(_relayout_halo_side_count(layout) for layout in layouts))


def _relayout_mask_mults(layouts: Iterable[LayoutState]) -> int:
    return int(sum(_relayout_halo_side_count(layout) for layout in layouts))


def _relayout_depth_units(layouts: Iterable[LayoutState]) -> int:
    return int(sum(1 for layout in layouts if int(layout.alpha) > 0 or int(layout.beta) > 0))


def _node_layout_row(
    node: str,
    layout: LayoutState,
    compact: LayoutState | None,
    *,
    relayout: bool,
    reason: str,
    producer_materialized_halo: bool = False,
    shape: Sequence[int] | None = None,
    fhe_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    return {
        "node": str(node),
        "shape": [] if shape is None else [int(value) for value in shape],
        "fhe_shape": [] if fhe_shape is None else [int(value) for value in fhe_shape],
        "selected_layout": layout.to_dict(),
        "compact_layout": {} if compact is None else compact.to_dict(),
        "output_relayout": bool(relayout),
        "output_relayout_reason": str(reason) if bool(relayout) else "",
        "producer_materialized_halo": bool(producer_materialized_halo),
        "producer_materialized_halo_reason": (
            str(reason) if bool(producer_materialized_halo) else ""
        ),
        "relayout_rotation_estimate": int(_relayout_rotations((layout,)) if bool(relayout) else 0),
        "relayout_mask_mult_estimate": int(_relayout_mask_mults((layout,)) if bool(relayout) else 0),
        "relayout_depth_estimate": int(_relayout_depth_units((layout,)) if bool(relayout) else 0),
    }


def _halo_slots_for_rows(edge_rows: Iterable[dict[str, Any]]) -> int:
    return int(
        sum(
            max(0, int(row["selected_layout"]["stored_slots"]) - int(row["selected_layout"]["core_slots"]))
            for row in edge_rows
        )
    )


def _halo_slot_tiebreak(edge_rows: Iterable[dict[str, Any]]) -> int:
    halo_slots = _halo_slots_for_rows(edge_rows)
    return int(halo_slots) if float(HALO_SLOT_WEIGHT) > 0.0 else -int(halo_slots)


def _edge_linear_cost(row: dict[str, Any]) -> float:
    layout = dict(row["selected_layout"])
    halo_slots = max(0, int(layout["stored_slots"]) - int(layout["core_slots"]))
    return (
        float(int(row["lt_bsgs_rotation_estimate"]) + int(row.get("compact_fallback_penalty_estimate", 0))) * LT_ROTATION_WEIGHT
        + float(halo_slots) * HALO_SLOT_WEIGHT
        + float(layout["tile_count"]) * TILE_WEIGHT
    )


def _relayout_linear_cost(layout: LayoutState) -> float:
    return (
        float(_relayout_rotations((layout,))) * RELAYOUT_ROTATION_WEIGHT
        + float(_relayout_mask_mults((layout,))) * RELAYOUT_MASK_MULT_WEIGHT
    )


def _policy_linear_cost(edge_rows: Iterable[dict[str, Any]], relayout_layouts: Iterable[LayoutState]) -> float:
    return float(sum(_edge_linear_cost(row) for row in edge_rows)) + float(
        sum(_relayout_linear_cost(layout) for layout in relayout_layouts)
    ) + float(
        _relayout_depth_units(relayout_layouts) * RELAYOUT_DEPTH_WEIGHT
    )


def _finalize_policy(
    *,
    policy: str,
    edge_rows: list[dict[str, Any]],
    relayout_layouts: list[LayoutState],
    node_layouts: list[dict[str, Any]] | None = None,
) -> PolicyPlan:
    node_layouts = list(node_layouts or [])
    stored_slots = int(sum(int(row["selected_layout"]["stored_slots"]) for row in edge_rows))
    core_slots = int(sum(int(row["selected_layout"]["core_slots"]) for row in edge_rows))
    tile_count = int(sum(int(row["selected_layout"]["tile_count"]) for row in edge_rows))
    relayout_rotation_estimate = _relayout_rotations(relayout_layouts)
    relayout_mask_mult_estimate = _relayout_mask_mults(relayout_layouts)
    relayout_depth_estimate = _relayout_depth_units(relayout_layouts)
    lt_rotation_estimate = int(sum(int(row["lt_bsgs_rotation_estimate"]) for row in edge_rows))
    compact_fallback_penalty_estimate = int(
        sum(int(row.get("compact_fallback_penalty_estimate", 0)) for row in edge_rows)
    )
    halo_slots = int(stored_slots - core_slots)
    bootstrap_proxy = int(
        math.ceil(
            (
                float(relayout_rotation_estimate)
                + float(relayout_mask_mult_estimate)
                + float(relayout_depth_estimate) * 64.0
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
        + float(relayout_mask_mult_estimate) * RELAYOUT_MASK_MULT_WEIGHT
        + float(lt_rotation_estimate + compact_fallback_penalty_estimate) * LT_ROTATION_WEIGHT
        + float(halo_slots) * HALO_SLOT_WEIGHT
        + float(tile_count) * TILE_WEIGHT
        + float(relayout_depth_estimate) * RELAYOUT_DEPTH_WEIGHT
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
        relayout_mask_mult_estimate=int(relayout_mask_mult_estimate),
        relayout_depth_estimate=int(relayout_depth_estimate),
        lt_bsgs_rotation_estimate=int(lt_rotation_estimate),
        compact_fallback_penalty_estimate=int(compact_fallback_penalty_estimate),
        bootstrap_proxy=int(bootstrap_proxy),
        objective=float(objective),
        edge_layouts=tuple(edge_rows),
        node_layouts=tuple(dict(row) for row in node_layouts),
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
            ),
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
    node_layouts: tuple[dict[str, Any], ...] = ()

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


def _fill_beta_to_tile_capacity(layout: LayoutState, *, shape: tuple[int, int, int, int], slots: int) -> LayoutState:
    target_tiles = max(1, int(layout.tile_count))
    _n, channels, height, width = shape
    gap = max(1, int(layout.gap))
    phase = max(1, int(gap) * int(gap))
    channel_groups = _ceil_div(int(channels), int(phase))
    row_width_slots = max(1, int(channel_groups) * int(width) * int(gap))
    max_stored_h = max(1, int(target_tiles) * int(slots) // int(row_width_slots))
    max_halo_rows = max(0, int(max_stored_h) - int(height) * int(gap))
    max_alpha_beta = int(max_halo_rows) // int(gap)
    beta = max(int(layout.beta), int(max_alpha_beta) - int(layout.alpha))
    filled = _layout_for_shape(
        shape=shape,
        gap=int(layout.gap),
        alpha=int(layout.alpha),
        beta=int(beta),
        stride=int(layout.stride),
        slots=int(slots),
    )
    while int(filled.tile_count) > int(target_tiles) and int(beta) > int(layout.beta):
        beta -= 1
        filled = _layout_for_shape(
            shape=shape,
            gap=int(layout.gap),
            alpha=int(layout.alpha),
            beta=int(beta),
            stride=int(layout.stride),
            slots=int(slots),
        )
    return filled


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
    capacity_fill = _fill_beta_to_tile_capacity(local_need, shape=edge.shape, slots=int(slots))
    fixed_capacity_fill = _fill_beta_to_tile_capacity(fixed_like, shape=edge.shape, slots=int(slots))
    reduced_alpha = int(global_alpha)
    reduced_beta = int(global_beta)
    target_tiles = max(int(edge.compact.tile_count), int(local_need.tile_count))
    while (int(reduced_alpha) > int(local_need.alpha) or int(reduced_beta) > int(local_need.beta)):
        candidate = _layout_for_shape(
            shape=edge.shape,
            gap=int(edge.compact.gap),
            alpha=int(reduced_alpha),
            beta=int(reduced_beta),
            stride=max(1, int(local_need.stride)),
            slots=int(slots),
        )
        if int(candidate.tile_count) <= int(target_tiles):
            break
        if int(reduced_alpha) >= int(reduced_beta) and int(reduced_alpha) > int(local_need.alpha):
            reduced_alpha -= 1
        elif int(reduced_beta) > int(local_need.beta):
            reduced_beta -= 1
        else:
            break
    capacity_safe = _layout_for_shape(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        alpha=int(reduced_alpha),
        beta=int(reduced_beta),
        stride=max(1, int(local_need.stride)),
        slots=int(slots),
    )
    return _dedupe_layouts((edge.compact, local_need, capacity_fill, capacity_safe, fixed_like, fixed_capacity_fill))


def _tconv_output_layout_candidates(
    module: Any | None,
    *,
    incoming_rows: Sequence[dict[str, Any]],
    outgoing: Sequence[EdgeInfo],
    slots: int,
) -> tuple[LayoutState, ...]:
    if not outgoing:
        return ()
    edge = outgoing[0]
    scale = max(1, int(_pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))[0]))
    layouts: list[LayoutState] = []
    for row in incoming_rows:
        source_layout = LayoutState(**dict(row["selected_layout"]))
        layouts.append(
            _layout_for_shape(
                shape=edge.shape,
                gap=int(edge.compact.gap),
                alpha=int(source_layout.alpha) * int(scale),
                beta=int(source_layout.beta) * int(scale),
                stride=max(1, int(source_layout.stride)),
                slots=int(slots),
            )
        )
    return _dedupe_layouts(layouts)


def _operator_output_layout_candidates(
    module: Any | None,
    *,
    incoming_rows: Sequence[dict[str, Any]],
    outgoing: Sequence[EdgeInfo],
    slots: int,
) -> tuple[LayoutState, ...]:
    if not outgoing:
        return ()
    edge = outgoing[0]
    layouts: list[LayoutState] = []
    for row in incoming_rows:
        source_layout = LayoutState(**dict(row["selected_layout"]))
        if isinstance(module, ConvTranspose2d):
            scale = max(1, int(_pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))[0]))
            alpha = int(source_layout.alpha) * int(scale)
            beta = int(source_layout.beta) * int(scale)
        elif isinstance(module, Conv2d):
            stride = max(1, int(_pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))[0]))
            consume = _conv_halo_consume(module)
            alpha = _side_after_downsample(source_layout.alpha, consume=int(consume), stride=int(stride))
            beta = _side_after_downsample(source_layout.beta, consume=int(consume), stride=int(stride))
        elif isinstance(module, AvgPool2d):
            stride_pair = _pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))
            kernel_pair = _pair_tuple(getattr(module, "kernel_size", (1, 1)), (1, 1))
            stride = max(1, int(stride_pair[0]))
            consume = max(0, int(kernel_pair[0]) - int(stride_pair[0]))
            alpha = _side_after_downsample(source_layout.alpha, consume=int(consume), stride=int(stride))
            beta = _side_after_downsample(source_layout.beta, consume=int(consume), stride=int(stride))
        else:
            alpha = int(source_layout.alpha)
            beta = int(source_layout.beta)
        layouts.append(
            _layout_for_shape(
                shape=edge.shape,
                gap=int(edge.compact.gap),
                alpha=int(alpha),
                beta=int(beta),
                stride=max(1, int(source_layout.stride)),
                slots=int(slots),
            )
        )
    return _dedupe_layouts(layouts)


def _dp_output_layout_candidates(
    module: Any | None,
    *,
    incoming_rows: Sequence[dict[str, Any]],
    outgoing: Sequence[EdgeInfo],
    global_alpha: int,
    global_beta: int,
    slots: int,
) -> tuple[LayoutState, ...]:
    if not outgoing:
        return ()
    if not incoming_rows:
        return _source_layout_candidates(
            outgoing,
            global_alpha=int(global_alpha),
            global_beta=int(global_beta),
            slots=int(slots),
        )
    if type(module).__name__ == "Add":
        return _dedupe_layouts(
            LayoutState(**dict(row["selected_layout"]))
            for row in incoming_rows
        )
    if isinstance(module, ConvTranspose2d):
        return _tconv_output_layout_candidates(
            module,
            incoming_rows=incoming_rows,
            outgoing=outgoing,
            slots=int(slots),
        )
    return _operator_output_layout_candidates(
        module,
        incoming_rows=incoming_rows,
        outgoing=outgoing,
        slots=int(slots),
    )


def _native_operator_output_layout(module: Any | None) -> bool:
    return isinstance(module, (AvgPool2d, Conv2d, ConvTranspose2d))


def _incoming_relayout_candidates(
    edge: EdgeInfo,
    *,
    global_alpha: int,
    global_beta: int,
    slots: int,
) -> tuple[LayoutState, ...]:
    del global_alpha, global_beta
    capacity_fill = _fill_beta_to_tile_capacity(edge.requirement, shape=edge.shape, slots=int(slots))
    return tuple(
        layout
        for layout in _dedupe_layouts((edge.requirement, capacity_fill))
        if layout.covers(edge.requirement)
    )


def _future_tconv_input_relayout_candidates(edge: EdgeInfo, *, slots: int) -> tuple[LayoutState, ...]:
    if str(edge.op_kind) != "conv_transpose2d":
        return ()
    halo = _layout_for_shape(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        alpha=max(1, int(edge.requirement.alpha)),
        beta=max(1, int(edge.requirement.beta)),
        stride=max(1, int(edge.requirement.stride)),
        slots=int(slots),
    )
    return _dedupe_layouts((halo, _fill_beta_to_tile_capacity(halo, shape=edge.shape, slots=int(slots))))


def _incoming_non_add_options(
    edge: EdgeInfo,
    source_layout: LayoutState,
    *,
    global_alpha: int,
    global_beta: int,
    slots: int,
) -> tuple[tuple[list[dict[str, Any]], list[LayoutState]], ...]:
    options: list[tuple[list[dict[str, Any]], list[LayoutState]]] = []
    if source_layout.covers(edge.requirement):
        options.append(
            (
                [
                    _edge_row(
                        edge,
                        source_layout,
                        relayout=False,
                        relayout_reason="",
                        lt_rotations=_lt_rotations(edge, source_layout),
                    )
                ],
                [],
            ),
        )
    if str(edge.source) == "x":
        return tuple(options)
    if source_layout.covers(edge.requirement):
        relayout_candidates = _future_tconv_input_relayout_candidates(edge, slots=int(slots))
    else:
        relayout_candidates = _incoming_relayout_candidates(
            edge,
            global_alpha=int(global_alpha),
            global_beta=int(global_beta),
            slots=int(slots),
        )
    for layout in relayout_candidates:
        if source_layout.covers(layout) and _same_physical_layout(source_layout, layout):
            continue
        options.append(
            (
                [
                    _edge_row(
                        edge,
                        layout,
                        relayout=True,
                        relayout_reason="dp_state_consumer_relayout",
                        lt_rotations=_lt_rotations(edge, layout),
                    )
                ],
                [layout],
            )
        )
    return tuple(options)


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
            relayout = not _same_physical_layout(source_layout, target_layout)
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
                incoming_options_work: tuple[tuple[list[dict[str, Any]], list[LayoutState]], ...] = (([], []),)
                for edge in incoming:
                    edge_options = _incoming_non_add_options(
                        edge,
                        live[edge.source],
                        global_alpha=int(global_alpha),
                        global_beta=int(global_beta),
                        slots=int(slots),
                    )
                    combined: list[tuple[list[dict[str, Any]], list[LayoutState]]] = []
                    for prefix_rows, prefix_relayouts in incoming_options_work:
                        for edge_rows, edge_relayouts in edge_options:
                            combined.append(
                                (
                                    [*prefix_rows, *edge_rows],
                                    [*prefix_relayouts, *edge_relayouts],
                                )
                            )
                    incoming_options_work = tuple(combined)
                incoming_options = incoming_options_work

            for incoming_rows, incoming_relayouts in incoming_options:
                next_live = dict(live)
                for source in list(next_live):
                    if int(last_consumer_index.get(str(source), -1)) <= int(node_index):
                        next_live.pop(str(source), None)
                module = dag.nodes[node].get("module")
                output_candidates = _dp_output_layout_candidates(
                    module,
                    incoming_rows=incoming_rows,
                    outgoing=outgoing,
                    global_alpha=int(global_alpha),
                    global_beta=int(global_beta),
                    slots=int(slots),
                )
                if not output_candidates:
                    output_candidates = (None,)
                for output_layout in output_candidates:
                    candidate_live = dict(next_live)
                    output_layout_rows = list(state.node_layouts)
                    output_relayouts: list[LayoutState] = []
                    if output_layout is not None:
                        candidate_live[str(node)] = output_layout
                        compact = outgoing[0].compact if outgoing else None
                        output_relayout = (
                            compact is not None
                            and not _same_physical_layout(output_layout, compact)
                            and str(node) != "x"
                            and type(module).__name__ != "Add"
                            and not _native_operator_output_layout(module)
                        )
                        if bool(output_relayout):
                            output_relayouts.append(output_layout)
                        output_layout_rows.append(
                            _node_layout_row(
                                str(node),
                                output_layout,
                                compact,
                                relayout=bool(output_relayout),
                                reason="dp_producer_materialized_halo" if bool(output_relayout) else "",
                                producer_materialized_halo=bool(output_relayout),
                                shape=outgoing[0].shape if outgoing else None,
                                fhe_shape=outgoing[0].fhe_shape if outgoing else None,
                            )
                        )
                    candidate_rows = tuple(list(state.edge_rows) + list(incoming_rows))
                    candidate_relayouts = tuple(
                        list(state.relayout_layouts)
                        + list(incoming_relayouts)
                        + list(output_relayouts)
                    )
                    local_relayout_depth = int(
                        _relayout_depth_units(incoming_relayouts)
                        + _relayout_depth_units(output_relayouts)
                    )
                    local_stack_depth = max(0, int(local_relayout_depth) - 1)
                    candidate_score = (
                        float(state.score)
                        + _policy_linear_cost(incoming_rows, incoming_relayouts)
                        + float(sum(_relayout_linear_cost(layout) for layout in output_relayouts))
                        + float(_relayout_depth_units(output_relayouts) * RELAYOUT_DEPTH_WEIGHT)
                        + float(local_stack_depth) * RELAYOUT_NODE_STACK_DEPTH_WEIGHT
                    )
                    key = _frontier_key(candidate_live)
                    existing = next_states.get(key)
                    if existing is None or (
                        candidate_score,
                        len(candidate_relayouts),
                        _halo_slot_tiebreak(candidate_rows),
                    ) < (
                        float(existing.score),
                        len(existing.relayout_layouts),
                        _halo_slot_tiebreak(existing.edge_rows),
                    ):
                        next_states[key] = _FrontierState(
                            score=float(candidate_score),
                            live_layouts=_frontier_live_items(candidate_live),
                            edge_rows=candidate_rows,
                            relayout_layouts=candidate_relayouts,
                            node_layouts=tuple(output_layout_rows),
                        )
        states = next_states
    if not states:
        raise RuntimeError("layout policy DP found no legal state")
    best_state = min(
        states.values(),
        key=lambda state: (
            float(state.score),
            len(state.relayout_layouts),
            _halo_slot_tiebreak(state.edge_rows),
        ),
    )
    rows = [{**dict(row), "dp_state_planned": True} for row in best_state.edge_rows]
    rows.sort(key=lambda row: (topo_index.get(str(row["source"]), 10**9), topo_index.get(str(row["target"]), 10**9)))
    return _finalize_policy(
        policy="dp",
        edge_rows=rows,
        relayout_layouts=list(best_state.relayout_layouts),
        node_layouts=list(best_state.node_layouts),
    )


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


def build_layout_policy_compile_plan(
    dag: NetworkDAG,
    *,
    policy: str = "dp",
    slots: int = DEFAULT_SLOTS,
) -> dict[str, Any]:
    edges = build_edge_infos(dag, slots=int(slots))
    plan = plan_policy(dag, edges, policy, slots=int(slots))
    edge_layouts = [dict(row) for row in plan.edge_layouts]
    node_layouts = [dict(row) for row in plan.node_layouts]
    relayout_edges = [
        {
            "edge": str(row["edge"]),
            "source": str(row["source"]),
            "target": str(row["target"]),
            "reason": str(row.get("relayout_reason", "")),
            "selected_layout": dict(row["selected_layout"]),
            "rotation_estimate": int(_relayout_rotations((LayoutState(**dict(row["selected_layout"])),))),
            "mask_mult_estimate": int(_relayout_mask_mults((LayoutState(**dict(row["selected_layout"])),))),
        }
        for row in edge_layouts
        if bool(row.get("relayout", False))
    ]
    output_relayout_nodes = [
        {
            "node": str(row["node"]),
            "reason": str(row.get("output_relayout_reason", "")),
            "selected_layout": dict(row["selected_layout"]),
            "rotation_estimate": int(row.get("relayout_rotation_estimate", 0)),
            "mask_mult_estimate": int(row.get("relayout_mask_mult_estimate", 0)),
            "depth_estimate": int(row.get("relayout_depth_estimate", 0)),
        }
        for row in node_layouts
        if bool(row.get("output_relayout", False))
    ]
    summary = plan.summary_row()
    summary["relayouts"] = int(len(relayout_edges) + len(output_relayout_nodes))
    summary["relayout_rotation_estimate"] = int(
        sum(int(row["rotation_estimate"]) for row in relayout_edges)
        + sum(int(row.get("rotation_estimate", 0)) for row in output_relayout_nodes)
    )
    summary["relayout_mask_mult_estimate"] = int(
        sum(int(row["mask_mult_estimate"]) for row in relayout_edges)
        + sum(int(row.get("mask_mult_estimate", 0)) for row in output_relayout_nodes)
    )
    summary["relayout_depth_estimate"] = int(
        sum(1 for row in relayout_edges if int(dict(row.get("selected_layout", {})).get("alpha", 0)) > 0 or int(dict(row.get("selected_layout", {})).get("beta", 0)) > 0)
        + sum(int(row.get("depth_estimate", 0)) for row in output_relayout_nodes)
    )
    return {
        "status": "ok",
        "policy": str(plan.policy),
        "policy_label": str(plan.policy_label),
        "metric_source": str(plan.metric_source),
        "slots": int(slots),
        "edge_layout_count": int(len(edge_layouts)),
        "relayout_edge_count": int(len(relayout_edges)),
        "output_relayout_node_count": int(len(output_relayout_nodes)),
        "summary": summary,
        "relayout_edges": relayout_edges,
        "output_relayout_nodes": output_relayout_nodes,
        "edge_layouts": edge_layouts,
        "node_layouts": node_layouts,
    }


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

U22_E2E_LOGQ = [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
U22_E2E_LOGP = [61, 61, 61]
U22_E2E_BOOT_LOGP = [61, 61, 61, 61, 61, 61, 61, 61]


def _runtime_config(*, backend: str, provider_mode: str, logn: int) -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": list(U22_E2E_LOGQ),
            "LogP": list(U22_E2E_LOGP),
            "LogScale": 40,
            "H": 192,
            "RingType": "standard",
        },
        "boot_params": {
            "LogP": list(U22_E2E_BOOT_LOGP),
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": str(backend),
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
            "experimental_region_first": str(provider_mode),
        },
    }


def _provider_mode_for_policy(spec: NetworkSpec, policy: str) -> str:
    normalized = normalize_policy(policy)
    if normalized == "dp":
        return str(spec.provider_mode)
    suffix = "fixedmax" if normalized == "fixed_max" else str(normalized)
    return f"{spec.provider_mode}_layout_{suffix}"


def _provider_pressure_summary(pressure: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(pressure, dict):
        return {}
    summary = pressure.get("summary", {})
    if not isinstance(summary, dict):
        return {}
    return {
        key: summary[key]
        for key in PROVIDER_PRESSURE_SUMMARY_KEYS
        if key in summary
    }


def _attach_provider_pressure(row: dict[str, Any], anchor: dict[str, Any]) -> None:
    pressure = anchor.get("provider_pressure")
    summary = dict(anchor.get("provider_pressure_summary", {}) or {})
    if not summary:
        summary = _provider_pressure_summary(pressure if isinstance(pressure, dict) else None)
    if isinstance(pressure, dict):
        row["provider_pressure"] = dict(pressure)
    if summary:
        row["provider_pressure_summary"] = dict(summary)
        for key in PROVIDER_PRESSURE_SUMMARY_KEYS:
            if key in summary:
                row[key] = summary[key]


def run_python_runtime_anchors(
    planner_payload: dict[str, Any],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    from orion.core.orion import scheme
    from orion.nn.module import Module

    spec = network_spec(str(planner_payload.get("network", "u22_64_base32")))
    if spec.network != "u22_64_base32":
        return {
            "status": "skipped",
            "reason": "python runtime anchors are only enabled for u22_64_base32 in v1",
            "policies": {},
        }
    torch.manual_seed(int(seed))
    x = torch.randn((1, int(spec.input_channels), int(spec.image_size), int(spec.image_size)), dtype=torch.float32)
    policy_anchors: dict[str, dict[str, Any]] = {}
    for policy_row in list(planner_payload.get("policies", [])):
        policy = normalize_policy(str(policy_row.get("policy", "dp")))
        if policy == "dp":
            policy_anchors[str(policy)] = {
                "status": "skipped",
                "reason": "dp uses the existing provider executable anchor; python layout runtime anchors cover non-dp lowering",
                "backend": "python",
                "provider_mode": str(spec.provider_mode),
            }
            continue
        provider_mode = _provider_mode_for_policy(spec, str(policy))
        config = _runtime_config(backend="python", provider_mode=str(provider_mode), logn=15)
        started = time.perf_counter()
        scheme.init_scheme(config)
        try:
            Module.set_scheme(scheme)
            Module.set_margin(scheme.params.get_margin())
            torch.manual_seed(int(seed))
            model = UNet22(dataset=str(spec.dataset), base_channels=int(spec.base_channels))
            model.eval()
            with torch.no_grad():
                reference = model(x).detach().cpu()
            scheme.fit(model, x)
            input_level = scheme.compile(model)
            attach_audit = dict(getattr(scheme, "region_first_attach_audit", {}) or {})
            model.he()
            forward_started = time.perf_counter()
            x_ct = scheme.encrypt(scheme.encode(x, int(input_level)))
            out_ct = model(x_ct)
            he_forward_s = float(time.perf_counter() - forward_started)
            from orion.experimental.u22_phase1 import collect_layout_policy_provider_pressure

            provider_pressure = collect_layout_policy_provider_pressure(
                getattr(scheme, "region_first_registry", None),
                backend=getattr(scheme, "backend", None),
                slots=int(scheme.params.get_slots()),
            )
            decoded = out_ct.decrypt().decode().detach().cpu()
            if torch.is_complex(decoded):
                decoded = decoded.real
            decoded = decoded.to(dtype=torch.float32)
            diff = (decoded - reference.to(dtype=torch.float32)).abs()
            for tensor in (out_ct, x_ct):
                release = getattr(tensor, "release", None)
                if callable(release):
                    release()
            policy_anchors[str(policy)] = {
                "status": "ok",
                "reason": "",
                "backend": "python",
                "provider_mode": str(provider_mode),
                "elapsed_s": float(time.perf_counter() - started),
                "he_forward_s": float(he_forward_s),
                "mae": float(diff.mean().item()),
                "max_abs": float(diff.max().item()),
                "dice": _dice_against_reference(decoded, reference),
                "bootstrap_count": None,
                "input_level": int(input_level),
                "attach_audit": attach_audit,
                "provider_pressure": provider_pressure,
                "provider_pressure_summary": _provider_pressure_summary(provider_pressure),
            }
        except Exception as exc:
            policy_anchors[str(policy)] = {
                "status": "failed",
                "reason": str(exc),
                "backend": "python",
                "provider_mode": str(provider_mode),
                "elapsed_s": float(time.perf_counter() - started),
            }
        finally:
            scheme.delete_scheme()
    return {
        "status": "ok" if all(str(row.get("status")) in {"ok", "skipped"} for row in policy_anchors.values()) else "failed",
        "kind": "orion_python_backend_runtime_anchor",
        "seed": int(seed),
        "input_shape": [int(value) for value in x.shape],
        "policies": policy_anchors,
    }


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
    return _run_provider_runtime_anchor(
        spec=spec,
        backend=str(backend),
        provider_mode=str(spec.provider_mode),
        cache_root=Path(cache_root),
        compile_timeout_s=int(compile_timeout_s),
        python=python,
        runner=runner,
        policy="dp",
    )


def _run_provider_runtime_anchor(
    *,
    spec: NetworkSpec,
    backend: str,
    provider_mode: str,
    cache_root: Path,
    compile_timeout_s: int,
    python: Path | None = None,
    runner: Runner = subprocess.run,
    policy: str = "dp",
) -> dict[str, Any]:
    if spec.network != "u22_64_base32":
        return {
            "status": "skipped",
            "reason": "runtime anchor is only enabled for u22_64_base32 in v1",
        }
    repo_root = Path(__file__).resolve().parents[2]
    python_bin = Path(sys.executable if python is None else python)
    safe_policy = normalize_policy(str(policy))
    out_dir = Path(cache_root) / "layout_policy_ablation" / f"{spec.network}_{backend}_{safe_policy}_provider_anchor"
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
        str(provider_mode),
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
            "policy": str(safe_policy),
            "provider_mode": str(provider_mode),
        }
    elapsed = float(time.perf_counter() - started)
    if int(completed.returncode) != 0:
        return {
            "status": "failed",
            "reason": f"runtime anchor exited {int(completed.returncode)}",
            "elapsed_s": float(elapsed),
            "out_dir": str(out_dir),
            "stdout_tail": str(completed.stdout or "")[-1000:],
            "policy": str(safe_policy),
            "provider_mode": str(provider_mode),
        }
    payloads = sorted(Path(out_dir).glob("*_fhe_figure.json"))
    if not payloads:
        return {
            "status": "missing_payload",
            "reason": "runtime anchor completed but no *_fhe_figure.json was found",
            "elapsed_s": float(elapsed),
            "out_dir": str(out_dir),
            "policy": str(safe_policy),
            "provider_mode": str(provider_mode),
        }
    payload = json.loads(payloads[-1].read_text(encoding="utf-8"))
    provider_pressure = payload.get("provider_pressure", {})
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
        "policy": str(safe_policy),
        "provider_mode": str(provider_mode),
        "he_forward_s": None if he_forward is None else float(he_forward),
        "mae": None if mae is None else float(mae),
        "dice": None if dice is None else float(dice),
        "bootstrap_count": None,
        "provider_pressure": provider_pressure if isinstance(provider_pressure, dict) else {},
        "provider_pressure_summary": _provider_pressure_summary(
            provider_pressure if isinstance(provider_pressure, dict) else None
        ),
    }


def run_backend_runtime_anchors(
    planner_payload: dict[str, Any],
    *,
    backend: str,
    cache_root: Path,
    compile_timeout_s: int,
    python: Path | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    spec = network_spec(str(planner_payload.get("network", "u22_64_base32")))
    if spec.network != "u22_64_base32":
        return {
            "status": "skipped",
            "reason": "backend runtime anchors are only enabled for u22_64_base32 in v1",
            "backend": str(backend),
            "policies": {},
        }
    if str(backend) != "lattigo":
        return {
            "status": "skipped",
            "reason": "non-dp layout-policy backend lowering is currently enabled for Lattigo only",
            "backend": str(backend),
            "policies": {},
        }
    policy_anchors: dict[str, dict[str, Any]] = {}
    for policy_row in list(planner_payload.get("policies", [])):
        policy = normalize_policy(str(policy_row.get("policy", "dp")))
        provider_mode = _provider_mode_for_policy(spec, str(policy))
        policy_anchors[str(policy)] = _run_provider_runtime_anchor(
            spec=spec,
            backend=str(backend),
            provider_mode=str(provider_mode),
            cache_root=Path(cache_root),
            compile_timeout_s=int(compile_timeout_s),
            python=python,
            runner=runner,
            policy=str(policy),
        )
    return {
        "status": "ok"
        if all(str(row.get("status")) in {"ok", "skipped"} for row in policy_anchors.values())
        else "failed",
        "kind": "orion_backend_layout_policy_runtime_anchors",
        "backend": str(backend),
        "policies": policy_anchors,
    }


def attach_runtime_anchor(planner_payload: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    payload = dict(planner_payload)
    backend = str(payload.get("backend", ""))
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
            _attach_provider_pressure(updated, anchor)
        elif backend == "lattigo":
            updated["runtime_status"] = "lattigo_layout_policy_runtime_anchor_not_run"
            updated["runtime_reason"] = "use run_backend_runtime_anchors to execute non-dp Lattigo layout-policy lowering"
        elif backend == "cheddar":
            updated["runtime_status"] = "cheddar_layout_policy_lowering_not_implemented"
            updated["runtime_reason"] = "non-dp executable layout-policy lowering is currently enabled for Lattigo only"
        policy_rows.append(updated)
    payload["policies"] = policy_rows
    payload["runtime_anchor"] = dict(anchor)
    return payload


def attach_backend_runtime_anchors(planner_payload: dict[str, Any], anchors: dict[str, Any]) -> dict[str, Any]:
    payload = dict(planner_payload)
    anchor_by_policy = dict(anchors.get("policies", {}))
    policy_rows = []
    for row in list(payload.get("policies", [])):
        updated = dict(row)
        policy = normalize_policy(str(updated.get("policy", "dp")))
        anchor = dict(anchor_by_policy.get(str(policy), {}))
        if anchor:
            updated["metric_source"] = f"{updated.get('metric_source', 'planner_estimate')}+runtime_anchor"
            updated["runtime_status"] = str(anchor.get("status", "unknown"))
            updated["runtime_reason"] = str(anchor.get("reason", ""))
            for key in ("bootstrap_count", "he_forward_s", "mae", "dice"):
                value = anchor.get(key)
                if value is not None:
                    updated[key] = value
            _attach_provider_pressure(updated, anchor)
        policy_rows.append(updated)
    payload["policies"] = policy_rows
    payload["runtime_anchors"] = dict(anchors)
    return payload


def attach_python_runtime_anchors(planner_payload: dict[str, Any], anchors: dict[str, Any]) -> dict[str, Any]:
    payload = dict(planner_payload)
    anchor_by_policy = dict(anchors.get("policies", {}))
    policy_rows = []
    for row in list(payload.get("policies", [])):
        updated = dict(row)
        policy = str(updated.get("policy", ""))
        anchor = dict(anchor_by_policy.get(policy, {}))
        if anchor:
            updated["metric_source"] = f"{updated.get('metric_source', 'planner_estimate')}+python_runtime_anchor"
            updated["runtime_status"] = str(anchor.get("status", "unknown"))
            updated["runtime_reason"] = str(anchor.get("reason", ""))
            for key in ("bootstrap_count", "he_forward_s", "mae", "dice", "max_abs"):
                value = anchor.get(key)
                if value is not None:
                    updated[key] = value
            _attach_provider_pressure(updated, anchor)
        policy_rows.append(updated)
    payload["policies"] = policy_rows
    payload["python_runtime_anchors"] = dict(anchors)
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
