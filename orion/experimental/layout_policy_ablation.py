from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.models.unet import UNet22
from orion.nn.activation import Activation, Chebyshev, Quad, ReLU
from orion.nn.linear import Conv2d, ConvTranspose2d
from orion.nn.pooling import AvgPool2d


DEFAULT_SLOTS = 32768
INPUT_CROSS_RECOVERY_ROTATION_MULTIPLIER = 2.0
DP_FRONTIER_STATE_LIMIT = 128
LAYOUT_ESTIMATOR_COUNT_ONLY = "count_only"
LAYOUT_ESTIMATOR_TEMPLATE = "template"
LAYOUT_ESTIMATOR_AUTO = "auto"
LAYOUT_ESTIMATOR_DEFAULT = os.environ.get("HALOED_LAYOUT_ESTIMATOR", LAYOUT_ESTIMATOR_COUNT_ONLY)
LAYOUT_ESTIMATOR_AUTO_RELATIVE_WINDOW = 0.15
LAYOUT_ESTIMATOR_AUTO_ABSOLUTE_WINDOW = 8192.0
TEMPLATE_ESTIMATOR_MAX_SLOT_MAPPINGS = int(os.environ.get("HALOED_TEMPLATE_ESTIMATOR_MAX_MAPPINGS", "4000000"))
PHYSICAL_COMPACT = "packed_compact"
PHYSICAL_LOGICAL_HALO = "logical_halo_compact"
PHYSICAL_NATIVE_SOURCE_STRIPE = "native_source_stripe"
POLICY_ALIASES = {
    "fixed": "fixed_max",
    "fixedmax": "fixed_max",
    "fixed-max": "fixed_max",
    "fixed_max": "fixed_max",
    "max": "fixed_max",
    "max_relayout": "fixed_max",
    "max-relayout": "fixed_max",
    "maximum_relayout": "fixed_max",
    "maximum-relayout": "fixed_max",
    "fixed_fused": "fixed_max_fused",
    "fixed-fused": "fixed_max_fused",
    "fixedmax_fused": "fixed_max_fused",
    "fixedmax-fused": "fixed_max_fused",
    "fixed_max_fused": "fixed_max_fused",
    "fixed-max-fused": "fixed_max_fused",
    "eager": "eager",
    "eager_relayout": "eager",
    "eager-relayout": "eager",
    "eager_fused": "eager_fused",
    "eager-fused": "eager_fused",
    "eager_relayout_fused": "eager_fused",
    "eager-relayout-fused": "eager_fused",
    "greedy": "greedy",
    "greedy_local": "greedy",
    "greedy-local": "greedy",
    "greedy_fused": "greedy_fused",
    "greedy-fused": "greedy_fused",
    "greedy_local_fused": "greedy_fused",
    "greedy-local-fused": "greedy_fused",
    "always": "always",
    "always_relayout": "always",
    "always-relayout": "always",
    "always_fused": "always_fused",
    "always-fused": "always_fused",
    "always_relayout_fused": "always_fused",
    "always-relayout-fused": "always_fused",
    "orion": "orion_dense",
    "dense": "orion_dense",
    "orion_dense": "orion_dense",
    "orion-dense": "orion_dense",
    "oriondense": "orion_dense",
    "no_halo": "orion_dense",
    "no-halo": "orion_dense",
    "nohalo": "orion_dense",
    "dp": "dp",
    "dp_global": "dp",
    "dp-global": "dp",
}
POLICY_LABELS = {
    "fixed_max": "Max-Re-Layout",
    "fixed_max_fused": "Fixed-Max-Halo+Fusion",
    "eager": "Eager-Re-Layout",
    "eager_fused": "Eager-Re-Layout+Fusion",
    "greedy": "Greedy-Max-Zero-Cycle",
    "greedy_fused": "Greedy-Local+Fusion",
    "always": "Always-Re-Layout",
    "always_fused": "Always-Re-Layout+Fusion",
    "orion_dense": "Orion-Dense-No-Halo",
    "dp": "DP-Global",
}
PROVIDER_PRESSURE_SUMMARY_KEYS = (
    "provider_region_count",
    "native_halo_provider_region_count",
    "relayout_lt_region_count",
    "relayout_edge_count",
    "output_relayout_edge_count",
    "compact_align_shared_edge_count",
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
    top_beta: int
    bottom_beta: int
    stride: int
    gap: int
    core_slots: int
    stored_slots: int
    tile_count: int
    physical_top_beta: int | None = None
    physical_bottom_beta: int | None = None
    boundary_pruned: bool = False

    def __post_init__(self) -> None:
        physical_top = int(self.top_beta) if self.physical_top_beta is None else int(self.physical_top_beta)
        physical_bottom = (
            int(self.bottom_beta) if self.physical_bottom_beta is None else int(self.physical_bottom_beta)
        )
        object.__setattr__(self, "top_beta", int(self.top_beta))
        object.__setattr__(self, "bottom_beta", int(self.bottom_beta))
        object.__setattr__(self, "stride", int(self.stride))
        object.__setattr__(self, "gap", int(self.gap))
        object.__setattr__(self, "core_slots", int(self.core_slots))
        object.__setattr__(self, "stored_slots", int(self.stored_slots))
        object.__setattr__(self, "tile_count", int(self.tile_count))
        object.__setattr__(self, "physical_top_beta", max(0, int(physical_top)))
        object.__setattr__(self, "physical_bottom_beta", max(0, int(physical_bottom)))
        object.__setattr__(
            self,
            "boundary_pruned",
            bool(self.boundary_pruned)
            or int(physical_top) < int(self.top_beta)
            or int(physical_bottom) < int(self.bottom_beta),
        )

    @property
    def halo_slots(self) -> int:
        return max(0, int(self.stored_slots) - int(self.core_slots))

    def key(self) -> tuple[int, int, int, int, int, int, int]:
        return (
            int(self.top_beta),
            int(self.bottom_beta),
            int(self.stride),
            int(self.gap),
            int(self.physical_top_beta or 0),
            int(self.physical_bottom_beta or 0),
            int(bool(self.boundary_pruned)),
        )

    def covers(self, other: "LayoutState") -> bool:
        return (
            int(self.gap) == int(other.gap)
            and int(self.top_beta) >= int(other.top_beta)
            and int(self.bottom_beta) >= int(other.bottom_beta)
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
    future_layouts: tuple[LayoutState, ...] = ()
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
    activation_ct_mult_depth: int = 0

    @property
    def edge_id(self) -> str:
        return f"{self.source}->{self.target}"


@dataclass(frozen=True)
class RotationEstimate:
    input_cross: int
    local_submatrix: int
    output_materialize: int
    bsgs_groups: int
    transforms: int
    baby_rotations: int
    giant_rotations: int
    input_channel_multiplier: int
    output_channel_multiplier: int
    local_programs: int
    recovery_programs: int
    rho_hat_per_program: int
    unfused_rotations: int
    same_input_fusion_savings: int
    ct_pt_mults: int
    estimator: str = LAYOUT_ESTIMATOR_COUNT_ONLY

    @property
    def rotations(self) -> int:
        return int(self.input_cross) + int(self.local_submatrix) + int(self.output_materialize)

    def to_lt_stats(self) -> dict[str, int]:
        return {
            "bsgs_groups": int(self.bsgs_groups),
            "transforms": int(self.transforms),
            "baby_rotations": int(self.baby_rotations),
            "giant_rotations": int(self.giant_rotations),
            "input_channel_multiplier": int(self.input_channel_multiplier),
            "output_channel_multiplier": int(self.output_channel_multiplier),
            "rotations": int(self.rotations),
            "input_cross_rotations": int(self.input_cross),
            "local_submatrix_rotations": int(self.local_submatrix),
            "output_materialize_rotations": int(self.output_materialize),
            "local_programs": int(self.local_programs),
            "recovery_programs": int(self.recovery_programs),
            "rho_hat_per_program": int(self.rho_hat_per_program),
            "unfused_rotations": int(self.unfused_rotations),
            "same_input_fusion_savings": int(self.same_input_fusion_savings),
            "ct_pt_mults": int(self.ct_pt_mults),
            "estimator": str(self.estimator),
        }


@dataclass(frozen=True)
class ExecutionCandidate:
    edge: EdgeInfo
    source_layout: LayoutState
    target_layout: LayoutState
    source_physical: str
    target_physical: str | None = None
    layout_mode: str = "halo_local"
    relayout: bool = False
    relayout_reason: str = ""
    consumer_fused_relayout: bool = False
    consumer_fused_rotation_estimate: int = 0


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
    producer_fused_materialization_count: int
    producer_fused_rotation_estimate: int
    consumer_fused_relayout_count: int
    consumer_fused_rotation_estimate: int
    lt_bsgs_rotation_estimate: int
    planner_rotation_cost_estimate: int
    reported_rotation_estimate: int
    lt_ct_pt_mult_estimate: int
    activation_ct_mult_estimate: int
    ct_pt_mult_estimate: int
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
            "producer_fused_materialization_count": int(self.producer_fused_materialization_count),
            "producer_fused_rotation_estimate": int(self.producer_fused_rotation_estimate),
            "consumer_fused_relayout_count": int(self.consumer_fused_relayout_count),
            "consumer_fused_rotation_estimate": int(self.consumer_fused_rotation_estimate),
            "compact_fallback_penalty_estimate": 0,
            "lt_bsgs_rotation_estimate": int(self.lt_bsgs_rotation_estimate),
            "planner_rotation_cost_estimate": int(self.planner_rotation_cost_estimate),
            "reported_rotation_estimate": int(self.reported_rotation_estimate),
            "lt_ct_pt_mult_estimate": int(self.lt_ct_pt_mult_estimate),
            "activation_ct_mult_estimate": int(self.activation_ct_mult_estimate),
            "ct_pt_mult_estimate": int(self.ct_pt_mult_estimate),
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
    top_beta: int,
    bottom_beta: int,
    stride: int,
    slots: int,
    physical_top_beta: int | None = None,
    physical_bottom_beta: int | None = None,
    boundary_pruned: bool | None = None,
) -> LayoutState:
    _n, channels, height, width = shape
    phase = max(1, int(gap) * int(gap))
    channel_groups = _ceil_div(int(channels), int(phase))
    core_slots = int(channel_groups * int(height) * int(gap) * int(width) * int(gap))
    if boundary_pruned is None:
        boundary_pruned = bool(int(top_beta) > 0 or int(bottom_beta) > 0)
    if physical_top_beta is None:
        physical_top_beta = 0 if bool(boundary_pruned) else int(top_beta)
    if physical_bottom_beta is None:
        physical_bottom_beta = 0 if bool(boundary_pruned) else int(bottom_beta)
    physical_top_beta = max(0, int(physical_top_beta))
    physical_bottom_beta = max(0, int(physical_bottom_beta))
    stored_h = int(height) * int(gap) + int(physical_top_beta + physical_bottom_beta) * int(gap)
    stored_slots = int(channel_groups * int(stored_h) * int(width) * int(gap))
    return LayoutState(
        top_beta=int(top_beta),
        bottom_beta=int(bottom_beta),
        stride=int(stride),
        gap=int(gap),
        core_slots=int(core_slots),
        stored_slots=int(stored_slots),
        tile_count=max(1, _ceil_div(int(stored_slots), int(slots))),
        physical_top_beta=int(physical_top_beta),
        physical_bottom_beta=int(physical_bottom_beta),
        boundary_pruned=bool(boundary_pruned),
    )


def _layout_physical_top_beta(layout: LayoutState) -> int:
    return max(0, int(layout.physical_top_beta if layout.physical_top_beta is not None else layout.top_beta))


def _layout_physical_bottom_beta(layout: LayoutState) -> int:
    return max(0, int(layout.physical_bottom_beta if layout.physical_bottom_beta is not None else layout.bottom_beta))


def _fhe_shape_for_layout(
    *,
    shape: tuple[int, int, int, int],
    layout: LayoutState,
) -> tuple[int, int, int, int]:
    n, channels, height, width = (int(value) for value in shape)
    gap = max(1, int(layout.gap))
    on_channels = _ceil_div(int(channels), int(gap * gap))
    on_height = int(
        height * gap
        + (_layout_physical_top_beta(layout) + _layout_physical_bottom_beta(layout)) * gap
    )
    on_width = int(width * gap)
    return int(n), int(on_channels), int(on_height), int(on_width)


def _edge_with_output_layout(edge: EdgeInfo, output_layout: LayoutState) -> EdgeInfo:
    output_shape = edge.output_shape or edge.shape
    return replace(
        edge,
        output_fhe_shape=_fhe_shape_for_layout(
            shape=output_shape,
            layout=output_layout,
        ),
    )


def _compact_height_strip_fits_single_ct(
    *,
    shape: tuple[int, int, int, int],
    gap: int,
    slots: int,
) -> bool:
    _n, _channels, height, width = shape
    physical_h = int(height) * int(gap)
    physical_w = int(width) * int(gap)
    return int(physical_h * physical_w) <= int(slots)


def _consumer_requirement(module: Any | None) -> tuple[int, int, int, str, int]:
    if isinstance(module, AvgPool2d):
        kernel = tuple(int(value) for value in getattr(module, "kernel_size", (1, 1)))
        stride = tuple(int(value) for value in getattr(module, "stride", (1, 1)))
        halo = max(0, int(max(kernel) - max(stride)))
        return int(halo), int(halo), int(stride[0]), "avgpool2d", max(0, int(kernel[0] * kernel[1] - 1))
    if isinstance(module, Conv2d) and not isinstance(module, ConvTranspose2d):
        kernel = tuple(int(value) for value in getattr(module, "kernel_size", (1, 1)))
        stride = tuple(int(value) for value in getattr(module, "stride", (1, 1)))
        pad = tuple(int(value) for value in getattr(module, "padding", (0, 0)))
        radius = max(0, max(kernel) // 2 if max(pad) > 0 else max(kernel) - 1)
        return int(radius), int(radius), int(stride[0]), "conv2d", max(0, int(kernel[0] * kernel[1] - 1))
    if isinstance(module, ConvTranspose2d):
        kernel = tuple(int(value) for value in getattr(module, "kernel_size", (1, 1)))
        stride = tuple(int(value) for value in getattr(module, "stride", (1, 1)))
        return 0, 0, int(stride[0]), "conv_transpose2d", max(0, int(kernel[0] * kernel[1] - 1))
    return 0, 0, 1, type(module).__name__.lower() if module is not None else "input", 0


def _layout_preserving_module_for_demand(module: Any | None) -> bool:
    return isinstance(module, (Activation, Chebyshev, Quad, ReLU))


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


def _activation_ct_mult_depth(module: Any | None) -> int:
    if module is None:
        return 0
    if isinstance(module, Quad):
        return 1
    if isinstance(module, Chebyshev):
        degree = max(1, int(getattr(module, "degree", 1)))
        return int(math.ceil(math.log2(degree)))
    if isinstance(module, Activation):
        degree = max(1, int(len(getattr(module, "coeffs", ()) or ()) - 1))
        return int(math.ceil(math.log2(degree)))
    if isinstance(module, ReLU):
        degrees = tuple(int(value) for value in getattr(module, "degrees", ()) or ())
        sign_cost = sum(int(math.ceil(math.log2(max(1, degree)))) for degree in degrees)
        return int(sign_cost + 2)
    return 0


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
        top_beta, bottom_beta, stride, op_kind, lt_base = _consumer_requirement(target_module)
        output_shape, output_fhe_shape = _consumer_output_shapes(target_module, shape, fhe_shape)
        op_params = _consumer_op_params(target_module, input_shape=shape, output_shape=output_shape)
        if (
            str(op_kind) in {"conv2d", "avgpool2d"}
            and (int(top_beta) > 0 or int(bottom_beta) > 0)
            and _compact_height_strip_fits_single_ct(shape=shape, gap=int(gap), slots=int(slots))
        ):
            top_beta = 0
            bottom_beta = 0
        compact = _layout_for_shape(shape=shape, gap=int(gap), top_beta=0, bottom_beta=0, stride=1, slots=int(slots))
        requirement = _layout_for_shape(
            shape=shape,
            gap=int(gap),
            top_beta=int(top_beta),
            bottom_beta=int(bottom_beta),
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
                activation_ct_mult_depth=int(_activation_ct_mult_depth(target_module)),
            ),
        )
    edges_by_id = {str(edge.edge_id): edge for edge in edges}
    incoming_by_target: dict[str, list[str]] = {}
    outgoing_by_source: dict[str, list[str]] = {}
    for edge in edges:
        incoming_by_target.setdefault(str(edge.target), []).append(str(edge.edge_id))
        outgoing_by_source.setdefault(str(edge.source), []).append(str(edge.edge_id))
    for node in reversed([str(value) for value in dag.topological_sort()]):
        module = dag.nodes[node].get("module")
        if not _layout_preserving_module_for_demand(module):
            continue
        outgoing = [edges_by_id[edge_id] for edge_id in outgoing_by_source.get(str(node), [])]
        if not outgoing:
            continue
        hard_layouts: list[LayoutState] = []
        optional_layouts: list[LayoutState] = []
        for edge in outgoing:
            optional_layouts.append(edge.requirement)
            optional_layouts.extend(tuple(edge.future_layouts))
            hard_layouts.extend(_future_tconv_input_relayout_candidates(edge, slots=int(slots)))
        if not hard_layouts and not optional_layouts:
            continue
        hard_top_beta = max([int(layout.top_beta) for layout in hard_layouts] or [0])
        hard_bottom_beta = max([int(layout.bottom_beta) for layout in hard_layouts] or [0])
        hard_stride = max([int(layout.stride) for layout in hard_layouts] or [1])
        optional_top_beta = max([int(layout.top_beta) for layout in optional_layouts] or [0])
        optional_bottom_beta = max([int(layout.bottom_beta) for layout in optional_layouts] or [0])
        optional_stride = max([int(layout.stride) for layout in optional_layouts] or [1])
        for edge_id in incoming_by_target.get(str(node), []):
            edge = edges_by_id[str(edge_id)]
            requirement = edge.requirement
            if hard_layouts:
                requirement = _layout_for_shape(
                    shape=edge.shape,
                    gap=int(edge.compact.gap),
                    top_beta=max(int(edge.requirement.top_beta), int(hard_top_beta)),
                    bottom_beta=max(int(edge.requirement.bottom_beta), int(hard_bottom_beta)),
                    stride=max(int(edge.requirement.stride), int(hard_stride)),
                    slots=int(slots),
                )
            propagated_optional = _layout_for_shape(
                shape=edge.shape,
                gap=int(edge.compact.gap),
                top_beta=max(int(requirement.top_beta), int(optional_top_beta)),
                bottom_beta=max(int(requirement.bottom_beta), int(optional_bottom_beta)),
                stride=max(int(requirement.stride), int(optional_stride)),
                slots=int(slots),
            )
            future_layouts = tuple(
                layout for layout in _dedupe_layouts((*tuple(edge.future_layouts), propagated_optional))
                if not _same_layout(layout, requirement)
            )
            if not _same_layout(requirement, edge.requirement) or tuple(edge.future_layouts) != future_layouts:
                edges_by_id[str(edge_id)] = replace(edge, requirement=requirement, future_layouts=future_layouts)
    edges = list(edges_by_id.values())
    topo_index = {str(node): index for index, node in enumerate(dag.topological_sort())}
    return tuple(sorted(edges, key=lambda edge: (topo_index.get(edge.source, 10**9), topo_index.get(edge.target, 10**9))))


def _same_layout(left: LayoutState, right: LayoutState) -> bool:
    return left.key() == right.key()


def _same_physical_layout(left: LayoutState, right: LayoutState) -> bool:
    return (
        _layout_physical_top_beta(left) == _layout_physical_top_beta(right)
        and _layout_physical_bottom_beta(left) == _layout_physical_bottom_beta(right)
        and int(left.gap) == int(right.gap)
        and int(left.tile_count) == int(right.tile_count)
    )


def _layout_with_stride(layout: LayoutState, stride: int) -> LayoutState:
    return LayoutState(
        top_beta=int(layout.top_beta),
        bottom_beta=int(layout.bottom_beta),
        stride=max(1, int(stride)),
        gap=int(layout.gap),
        core_slots=int(layout.core_slots),
        stored_slots=int(layout.stored_slots),
        tile_count=int(layout.tile_count),
        physical_top_beta=_layout_physical_top_beta(layout),
        physical_bottom_beta=_layout_physical_bottom_beta(layout),
        boundary_pruned=bool(layout.boundary_pruned),
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
    top_beta = max(int(item.requirement.top_beta) for item in edges)
    bottom_beta = max(int(item.requirement.bottom_beta) for item in edges)
    stride = max(int(item.requirement.stride) for item in edges)
    return _layout_for_shape(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        top_beta=int(top_beta),
        bottom_beta=int(bottom_beta),
        stride=int(stride),
        slots=int(slots),
    )


def _max_layout_states(
    layouts: Sequence[LayoutState],
    *,
    shape: tuple[int, int, int, int],
    gap: int,
    slots: int,
) -> LayoutState:
    if not layouts:
        return _layout_for_shape(shape=shape, gap=int(gap), top_beta=0, bottom_beta=0, stride=1, slots=int(slots))
    return _layout_for_shape(
        shape=shape,
        gap=int(gap),
        top_beta=max(int(layout.top_beta) for layout in layouts),
        bottom_beta=max(int(layout.bottom_beta) for layout in layouts),
        stride=max(int(layout.stride) for layout in layouts),
        slots=int(slots),
    )


def _input_demand_for_output_layout(
    module: Any | None,
    edge: EdgeInfo,
    output_demand: LayoutState | None,
    *,
    slots: int,
) -> LayoutState:
    if output_demand is None:
        return edge.requirement
    if isinstance(module, (AvgPool2d, Conv2d)) and _compact_height_strip_fits_single_ct(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        slots=int(slots),
    ):
        return edge.requirement
    if isinstance(module, ConvTranspose2d):
        scale = max(1, int(_pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))[0]))
        top_beta = _ceil_div(int(output_demand.top_beta), int(scale))
        bottom_beta = _ceil_div(int(output_demand.bottom_beta), int(scale))
    elif isinstance(module, AvgPool2d):
        stride_pair = _pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))
        kernel_pair = _pair_tuple(getattr(module, "kernel_size", (1, 1)), (1, 1))
        stride = max(1, int(stride_pair[0]))
        consume = max(0, int(kernel_pair[0]) - int(stride_pair[0]))
        top_beta = int(output_demand.top_beta) * int(stride) + int(consume)
        bottom_beta = int(output_demand.bottom_beta) * int(stride) + int(consume)
    elif isinstance(module, Conv2d):
        stride = max(1, int(_pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))[0]))
        consume = _conv_halo_consume(module)
        top_beta = int(output_demand.top_beta) * int(stride) + int(consume)
        bottom_beta = int(output_demand.bottom_beta) * int(stride) + int(consume)
    else:
        top_beta = int(output_demand.top_beta)
        bottom_beta = int(output_demand.bottom_beta)
    return _layout_for_shape(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        top_beta=max(int(edge.requirement.top_beta), int(top_beta)),
        bottom_beta=max(int(edge.requirement.bottom_beta), int(bottom_beta)),
        stride=max(int(edge.requirement.stride), int(output_demand.stride)),
        slots=int(slots),
    )


def _backward_max_demand_layouts(
    dag: NetworkDAG,
    edges: Sequence[EdgeInfo],
    *,
    slots: int,
) -> dict[str, LayoutState]:
    edges_by_source: dict[str, list[EdgeInfo]] = {}
    edges_by_target: dict[str, list[EdgeInfo]] = {}
    for edge in edges:
        edges_by_source.setdefault(str(edge.source), []).append(edge)
        edges_by_target.setdefault(str(edge.target), []).append(edge)
    edge_demand: dict[str, LayoutState] = {}
    topo = [str(node) for node in dag.topological_sort()]
    for node in reversed(topo):
        outgoing = tuple(edges_by_source.get(str(node), ()))
        if outgoing:
            first = outgoing[0]
            output_demand = _max_layout_states(
                [edge_demand.get(str(edge.edge_id), edge.requirement) for edge in outgoing],
                shape=first.shape,
                gap=int(first.compact.gap),
                slots=int(slots),
            )
        else:
            output_demand = None
        module = dag.nodes[node].get("module")
        for edge in edges_by_target.get(str(node), ()):
            edge_demand[str(edge.edge_id)] = _input_demand_for_output_layout(
                module,
                edge,
                output_demand,
                slots=int(slots),
            )
    return edge_demand


def _edge_row(
    edge: EdgeInfo,
    layout: LayoutState,
    *,
    relayout: bool,
    relayout_reason: str,
    lt_rotations: int | None = None,
    planner_rotation_cost: int | None = None,
    layout_mode: str = "halo_local",
    source_layout: LayoutState | None = None,
    source_physical_layout: str | None = None,
    physical_layout: str | None = None,
    consumer_fused_relayout: bool = False,
    consumer_fused_rotation_estimate: int = 0,
    producer_fused_relayout: bool = False,
    producer_fused_rotation_estimate: int = 0,
    estimator: str | None = None,
) -> dict[str, Any]:
    if str(layout_mode) == "compact_global_fallback":
        raise ValueError("compact global fallback is not a valid halo-local layout-policy edge")
    lt_stats = _lt_rotation_stats(edge, layout, estimator=estimator)
    if lt_rotations is None:
        lt_rotations = int(lt_stats["rotations"])
    if planner_rotation_cost is None:
        planner_rotation_cost = int(lt_rotations)
    lt_ct_pt_mults = int(lt_stats["ct_pt_mults"])
    activation_ct_mults = int(layout.tile_count) * int(edge.activation_ct_mult_depth)
    if bool(consumer_fused_relayout):
        lt_ct_pt_mults = max(int(lt_ct_pt_mults), int(_lt_ct_pt_mults(edge, edge.requirement)))
    if physical_layout is None:
        physical_layout = (
            PHYSICAL_NATIVE_SOURCE_STRIPE
            if str(layout_mode) == "native_halo_stripe"
            else (PHYSICAL_LOGICAL_HALO if int(layout.top_beta) > 0 or int(layout.bottom_beta) > 0 else PHYSICAL_COMPACT)
        )
    relayout_estimate = _relayout_transition_estimate(
        source_layout=source_layout,
        target_layout=layout,
        relayout=bool(relayout),
    )
    if (
        bool(relayout)
        and str(physical_layout) == PHYSICAL_NATIVE_SOURCE_STRIPE
        and int(relayout_estimate["depth_estimate"]) == 0
    ):
        relayout_estimate = {
            "rotation_count": int(_relayout_rotations((layout,))),
            "mask_mult_count": 0,
            "sparse_lt_count": int(max(1, int(layout.tile_count))),
            "depth_estimate": 1,
        }
    return {
        "edge": edge.edge_id,
        "source": edge.source,
        "target": edge.target,
        "op_kind": edge.op_kind,
        "shape": [int(value) for value in edge.shape],
        "fhe_shape": [int(value) for value in _fhe_shape_for_layout(shape=edge.shape, layout=layout)],
        "required_layout": edge.requirement.to_dict(),
        "future_layouts": [layout.to_dict() for layout in tuple(edge.future_layouts)],
        "source_layout": {} if source_layout is None else source_layout.to_dict(),
        "selected_layout": layout.to_dict(),
        "target_layout": layout.to_dict(),
        "relayout": bool(relayout),
        "relayout_reason": str(relayout_reason),
        "relayout_rotation_estimate": int(relayout_estimate["rotation_count"]),
        "relayout_mask_mult_estimate": int(relayout_estimate["mask_mult_count"]),
        "relayout_sparse_lt_estimate": int(relayout_estimate["sparse_lt_count"]),
        "relayout_depth_estimate": int(relayout_estimate["depth_estimate"]),
        "lt_bsgs_rotation_estimate": int(lt_rotations),
        "planner_rotation_cost_estimate": int(planner_rotation_cost),
        "lt_ct_pt_mult_estimate": int(lt_ct_pt_mults),
        "activation_ct_mult_estimate": int(activation_ct_mults),
        "lt_bsgs_group_count_estimate": int(lt_stats["bsgs_groups"]),
        "lt_transform_count_estimate": int(lt_stats["transforms"]),
        "lt_baby_rotation_estimate": int(lt_stats["baby_rotations"]),
        "lt_giant_rotation_estimate": int(lt_stats["giant_rotations"]),
        "lt_input_cross_rotation_estimate": int(lt_stats["input_cross_rotations"]),
        "lt_local_submatrix_rotation_estimate": int(lt_stats["local_submatrix_rotations"]),
        "lt_output_materialize_rotation_estimate": int(lt_stats["output_materialize_rotations"]),
        "lt_local_program_count_estimate": int(lt_stats["local_programs"]),
        "lt_recovery_program_count_estimate": int(lt_stats["recovery_programs"]),
        "lt_rho_hat_per_program_estimate": int(lt_stats["rho_hat_per_program"]),
        "lt_unfused_rotation_estimate": int(lt_stats["unfused_rotations"]),
        "lt_same_input_fusion_savings_estimate": int(lt_stats["same_input_fusion_savings"]),
        "lt_input_channel_multiplier": int(lt_stats["input_channel_multiplier"]),
        "lt_output_channel_multiplier": int(lt_stats["output_channel_multiplier"]),
        "lt_estimator": str(lt_stats.get("estimator", _normalize_layout_estimator(estimator))),
        "layout_mode": str(layout_mode),
        "source_physical_layout": str(source_physical_layout or ""),
        "target_physical_layout": str(physical_layout),
        "physical_layout": str(physical_layout),
        "consumer_fused_relayout": bool(consumer_fused_relayout),
        "consumer_fused_rotation_estimate": int(consumer_fused_rotation_estimate),
        "producer_fused_relayout": bool(producer_fused_relayout),
        "producer_fused_rotation_estimate": int(producer_fused_rotation_estimate),
    }


def _layout_one_channel_physical_shape(
    *,
    clear_shape: tuple[int, int, int, int],
    gap: int,
    top_beta: int,
    bottom_beta: int,
) -> tuple[int, int]:
    _n, _c, height, width = clear_shape
    physical_h = int(height) * int(gap) + int(top_beta + bottom_beta) * int(gap)
    physical_w = int(width) * int(gap)
    return max(1, int(physical_h)), max(1, int(physical_w))


def _layout_one_channel_physical_shape_for_layout(
    *,
    clear_shape: tuple[int, int, int, int],
    layout: LayoutState,
) -> tuple[int, int]:
    return _layout_one_channel_physical_shape(
        clear_shape=clear_shape,
        gap=int(layout.gap),
        top_beta=_layout_physical_top_beta(layout),
        bottom_beta=_layout_physical_bottom_beta(layout),
    )


def _one_channel_block(slot_index: int, *, slots: int) -> tuple[int, int]:
    return int(slot_index) // int(slots), int(slot_index) % int(slots)


def _one_channel_lt_mapping_tensors(
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
    top_beta: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    slots: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if str(op_kind) in {"conv2d", "avgpool2d"}:
        out_h, out_w, kernel_y, kernel_x = torch.meshgrid(
            torch.arange(int(output_h), dtype=torch.int64),
            torch.arange(int(output_w), dtype=torch.int64),
            torch.arange(int(kernel_h), dtype=torch.int64),
            torch.arange(int(kernel_w), dtype=torch.int64),
            indexing="ij",
        )
        in_h = out_h * int(stride_h) - int(pad_h) + kernel_y * int(dilation_h)
        in_w = out_w * int(stride_w) - int(pad_w) + kernel_x * int(dilation_w)
    elif str(op_kind) == "conv_transpose2d":
        in_h, in_w, kernel_y, kernel_x = torch.meshgrid(
            torch.arange(int(input_h), dtype=torch.int64),
            torch.arange(int(input_w), dtype=torch.int64),
            torch.arange(int(kernel_h), dtype=torch.int64),
            torch.arange(int(kernel_w), dtype=torch.int64),
            indexing="ij",
        )
        out_h = in_h * int(stride_h) - int(pad_h) + kernel_y * int(dilation_h)
        out_w = in_w * int(stride_w) - int(pad_w) + kernel_x * int(dilation_w)
    else:
        empty = torch.empty((0,), dtype=torch.int64)
        return empty, empty, empty

    in_h = in_h.reshape(-1)
    in_w = in_w.reshape(-1)
    out_h = out_h.reshape(-1)
    out_w = out_w.reshape(-1)
    valid = (
        (in_h >= 0)
        & (in_h < int(input_h))
        & (in_w >= 0)
        & (in_w < int(input_w))
        & (out_h >= 0)
        & (out_h < int(output_h))
        & (out_w >= 0)
        & (out_w < int(output_w))
    )
    if not bool(valid.any().item()):
        empty = torch.empty((0,), dtype=torch.int64)
        return empty, empty, empty

    in_ph = (in_h[valid] + int(top_beta)) * int(input_gap)
    in_pw = in_w[valid] * int(input_gap)
    out_ph = out_h[valid] * int(output_gap)
    out_pw = out_w[valid] * int(output_gap)
    physical_valid = (
        (in_ph >= 0)
        & (in_ph < int(input_phys_h))
        & (in_pw >= 0)
        & (in_pw < int(input_phys_w))
        & (out_ph >= 0)
        & (out_ph < int(output_phys_h))
        & (out_pw >= 0)
        & (out_pw < int(output_phys_w))
    )
    if not bool(physical_valid.any().item()):
        empty = torch.empty((0,), dtype=torch.int64)
        return empty, empty, empty

    in_index = in_ph[physical_valid] * int(input_phys_w) + in_pw[physical_valid]
    out_index = out_ph[physical_valid] * int(output_phys_w) + out_pw[physical_valid]
    in_block = torch.div(in_index, int(slots), rounding_mode="floor").to(dtype=torch.int64)
    out_block = torch.div(out_index, int(slots), rounding_mode="floor").to(dtype=torch.int64)
    diagonal = torch.remainder(
        torch.remainder(in_index, int(slots)) - torch.remainder(out_index, int(slots)),
        int(slots),
    )
    return in_block, out_block, diagonal.to(dtype=torch.int64)


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
    top_beta: int,
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
    in_block, out_block, diagonal = _one_channel_lt_mapping_tensors(
        op_kind=str(op_kind),
        input_h=int(input_h),
        input_w=int(input_w),
        output_h=int(output_h),
        output_w=int(output_w),
        input_phys_h=int(input_phys_h),
        input_phys_w=int(input_phys_w),
        output_phys_h=int(output_phys_h),
        output_phys_w=int(output_phys_w),
        input_gap=int(input_gap),
        output_gap=int(output_gap),
        top_beta=int(top_beta),
        kernel_h=int(kernel_h),
        kernel_w=int(kernel_w),
        stride_h=int(stride_h),
        stride_w=int(stride_w),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
        dilation_h=int(dilation_h),
        dilation_w=int(dilation_w),
        slots=int(slots),
    )
    nonzero = diagonal != 0
    if not bool(nonzero.any().item()):
        return ()

    output_block_count = max(1, _ceil_div(int(output_phys_h) * int(output_phys_w), int(slots)))
    keys = (
        (in_block[nonzero] * int(output_block_count) + out_block[nonzero]) * int(slots)
        + diagonal[nonzero]
    )
    by_input_block: dict[int, dict[int, list[int]]] = {}
    for key in torch.unique(keys).tolist():
        pair, diag = divmod(int(key), int(slots))
        source, target = divmod(int(pair), int(output_block_count))
        by_input_block.setdefault(int(source), {}).setdefault(int(target), []).append(int(diag))

    return tuple(
        tuple(
            tuple(sorted(int(diagonal) for diagonal in diagonals))
            for _output_block, diagonals in sorted(output_groups.items())
            if diagonals
        )
        for _input_block, output_groups in sorted(by_input_block.items())
        if output_groups
    )


@lru_cache(maxsize=None)
def _one_channel_lt_ct_pt_mult_count_cached(
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
    top_beta: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    slots: int,
) -> int:
    in_block, out_block, diagonal = _one_channel_lt_mapping_tensors(
        op_kind=str(op_kind),
        input_h=int(input_h),
        input_w=int(input_w),
        output_h=int(output_h),
        output_w=int(output_w),
        input_phys_h=int(input_phys_h),
        input_phys_w=int(input_phys_w),
        output_phys_h=int(output_phys_h),
        output_phys_w=int(output_phys_w),
        input_gap=int(input_gap),
        output_gap=int(output_gap),
        top_beta=int(top_beta),
        kernel_h=int(kernel_h),
        kernel_w=int(kernel_w),
        stride_h=int(stride_h),
        stride_w=int(stride_w),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
        dilation_h=int(dilation_h),
        dilation_w=int(dilation_w),
        slots=int(slots),
    )
    if int(diagonal.numel()) == 0:
        return 0

    output_block_count = max(1, _ceil_div(int(output_phys_h) * int(output_phys_w), int(slots)))
    keys = (
        (in_block * int(output_block_count) + out_block) * int(slots)
        + diagonal
    )
    return int(torch.unique(keys).numel())


@lru_cache(maxsize=None)
def _one_channel_lt_adjacency_counts_cached(
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
    top_beta: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    slots: int,
) -> tuple[int, ...]:
    in_block, out_block, _diagonal = _one_channel_lt_mapping_tensors(
        op_kind=str(op_kind),
        input_h=int(input_h),
        input_w=int(input_w),
        output_h=int(output_h),
        output_w=int(output_w),
        input_phys_h=int(input_phys_h),
        input_phys_w=int(input_phys_w),
        output_phys_h=int(output_phys_h),
        output_phys_w=int(output_phys_w),
        input_gap=int(input_gap),
        output_gap=int(output_gap),
        top_beta=int(top_beta),
        kernel_h=int(kernel_h),
        kernel_w=int(kernel_w),
        stride_h=int(stride_h),
        stride_w=int(stride_w),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
        dilation_h=int(dilation_h),
        dilation_w=int(dilation_w),
        slots=int(slots),
    )
    if int(in_block.numel()) == 0:
        return ()
    output_block_count = max(1, _ceil_div(int(output_phys_h) * int(output_phys_w), int(slots)))
    pair_keys = torch.unique(in_block * int(output_block_count) + out_block)
    source_blocks = torch.div(pair_keys, int(output_block_count), rounding_mode="floor")
    _sources, counts = torch.unique_consecutive(source_blocks, return_counts=True)
    return tuple(int(value) for value in counts.tolist())


@lru_cache(maxsize=None)
def _representative_shift_indices_cached(
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
    top_beta: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    slots: int,
) -> tuple[int, ...]:
    samples_h = tuple(dict.fromkeys((0, max(0, int(input_h) // 2), max(0, int(input_h) - 1))))
    samples_w = tuple(dict.fromkeys((0, max(0, int(input_w) // 2), max(0, int(input_w) - 1))))
    out_samples_h = tuple(dict.fromkeys((0, max(0, int(output_h) // 2), max(0, int(output_h) - 1))))
    out_samples_w = tuple(dict.fromkeys((0, max(0, int(output_w) // 2), max(0, int(output_w) - 1))))
    shifts: set[int] = set()

    def add_shift(in_h: int, in_w: int, out_h: int, out_w: int) -> None:
        if not (0 <= int(in_h) < int(input_h) and 0 <= int(in_w) < int(input_w)):
            return
        if not (0 <= int(out_h) < int(output_h) and 0 <= int(out_w) < int(output_w)):
            return
        in_ph = (int(in_h) + int(top_beta)) * int(input_gap)
        in_pw = int(in_w) * int(input_gap)
        out_ph = int(out_h) * int(output_gap)
        out_pw = int(out_w) * int(output_gap)
        if not (0 <= int(in_ph) < int(input_phys_h) and 0 <= int(in_pw) < int(input_phys_w)):
            return
        if not (0 <= int(out_ph) < int(output_phys_h) and 0 <= int(out_pw) < int(output_phys_w)):
            return
        in_index = int(in_ph) * int(input_phys_w) + int(in_pw)
        out_index = int(out_ph) * int(output_phys_w) + int(out_pw)
        shift = int((int(in_index % int(slots)) - int(out_index % int(slots))) % int(slots))
        if int(shift) != 0:
            shifts.add(int(shift))

    if str(op_kind) in {"conv2d", "avgpool2d"}:
        for out_h in out_samples_h:
            base_h = int(out_h) * int(stride_h) - int(pad_h)
            for out_w in out_samples_w:
                base_w = int(out_w) * int(stride_w) - int(pad_w)
                for kernel_y in range(int(kernel_h)):
                    in_h = int(base_h) + int(kernel_y) * int(dilation_h)
                    for kernel_x in range(int(kernel_w)):
                        in_w = int(base_w) + int(kernel_x) * int(dilation_w)
                        add_shift(in_h, in_w, int(out_h), int(out_w))
    elif str(op_kind) == "conv_transpose2d":
        for in_h in samples_h:
            base_h = int(in_h) * int(stride_h) - int(pad_h)
            for in_w in samples_w:
                base_w = int(in_w) * int(stride_w) - int(pad_w)
                for kernel_y in range(int(kernel_h)):
                    out_h = int(base_h) + int(kernel_y) * int(dilation_h)
                    for kernel_x in range(int(kernel_w)):
                        out_w = int(base_w) + int(kernel_x) * int(dilation_w)
                        add_shift(int(in_h), int(in_w), int(out_h), int(out_w))
    return tuple(sorted(shifts))


@lru_cache(maxsize=None)
def _sampled_source_shift_sets_cached(
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
    top_beta: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    slots: int,
) -> tuple[tuple[int, ...], ...]:
    samples_h = tuple(dict.fromkeys((0, max(0, int(input_h) // 2), max(0, int(input_h) - 1))))
    samples_w = tuple(dict.fromkeys((0, max(0, int(input_w) // 2), max(0, int(input_w) - 1))))
    out_samples_h = tuple(dict.fromkeys((0, max(0, int(output_h) // 2), max(0, int(output_h) - 1))))
    out_samples_w = tuple(dict.fromkeys((0, max(0, int(output_w) // 2), max(0, int(output_w) - 1))))
    by_source: dict[int, set[int]] = {}

    def add_shift(in_h: int, in_w: int, out_h: int, out_w: int) -> None:
        if not (0 <= int(in_h) < int(input_h) and 0 <= int(in_w) < int(input_w)):
            return
        if not (0 <= int(out_h) < int(output_h) and 0 <= int(out_w) < int(output_w)):
            return
        in_ph = (int(in_h) + int(top_beta)) * int(input_gap)
        in_pw = int(in_w) * int(input_gap)
        out_ph = int(out_h) * int(output_gap)
        out_pw = int(out_w) * int(output_gap)
        if not (0 <= int(in_ph) < int(input_phys_h) and 0 <= int(in_pw) < int(input_phys_w)):
            return
        if not (0 <= int(out_ph) < int(output_phys_h) and 0 <= int(out_pw) < int(output_phys_w)):
            return
        in_index = int(in_ph) * int(input_phys_w) + int(in_pw)
        out_index = int(out_ph) * int(output_phys_w) + int(out_pw)
        source_block, in_slot = _one_channel_block(in_index, slots=int(slots))
        _out_block, out_slot = _one_channel_block(out_index, slots=int(slots))
        shift = int((int(in_slot) - int(out_slot)) % int(slots))
        if int(shift) != 0:
            by_source.setdefault(int(source_block), set()).add(int(shift))

    if str(op_kind) in {"conv2d", "avgpool2d"}:
        for out_h in out_samples_h:
            base_h = int(out_h) * int(stride_h) - int(pad_h)
            for out_w in out_samples_w:
                base_w = int(out_w) * int(stride_w) - int(pad_w)
                for kernel_y in range(int(kernel_h)):
                    in_h = int(base_h) + int(kernel_y) * int(dilation_h)
                    for kernel_x in range(int(kernel_w)):
                        in_w = int(base_w) + int(kernel_x) * int(dilation_w)
                        add_shift(in_h, in_w, int(out_h), int(out_w))
    elif str(op_kind) == "conv_transpose2d":
        for in_h in samples_h:
            base_h = int(in_h) * int(stride_h) - int(pad_h)
            for in_w in samples_w:
                base_w = int(in_w) * int(stride_w) - int(pad_w)
                for kernel_y in range(int(kernel_h)):
                    out_h = int(base_h) + int(kernel_y) * int(dilation_h)
                    for kernel_x in range(int(kernel_w)):
                        out_w = int(base_w) + int(kernel_x) * int(dilation_w)
                        add_shift(int(in_h), int(in_w), int(out_h), int(out_w))
    return tuple(tuple(sorted(shifts)) for _source, shifts in sorted(by_source.items()) if shifts)


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
    if (str(edge.op_kind) in {"add", "concat", "input"}):
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


def _normalize_layout_estimator(estimator: str | None) -> str:
    value = str(estimator or LAYOUT_ESTIMATOR_DEFAULT).strip().lower().replace("-", "_")
    aliases = {
        "count": LAYOUT_ESTIMATOR_COUNT_ONLY,
        "countonly": LAYOUT_ESTIMATOR_COUNT_ONLY,
        "count_only": LAYOUT_ESTIMATOR_COUNT_ONLY,
        "closed_form": LAYOUT_ESTIMATOR_COUNT_ONLY,
        "template": LAYOUT_ESTIMATOR_TEMPLATE,
        "template_refine": LAYOUT_ESTIMATOR_TEMPLATE,
        "unweighted_template": LAYOUT_ESTIMATOR_TEMPLATE,
        "auto": LAYOUT_ESTIMATOR_AUTO,
        "auto_template": LAYOUT_ESTIMATOR_AUTO,
    }
    if value not in aliases:
        raise ValueError(f"unsupported layout estimator {estimator!r}")
    return aliases[value]


def _bsgs_hat_from_diagonal_count(diagonal_count: int, *, slots: int) -> dict[str, int]:
    """Count-only BSGS proxy used by the layout DP.

    The planner must not construct weighted Toeplitz submatrices or exact
    diagonal sets for every candidate layout. This proxy uses only the number of
    nonzero diagonals and chooses a baby-step width from the same power-of-two
    family as the descriptor tools.
    """

    diagonals = max(0, int(diagonal_count))
    if diagonals == 0:
        return {"rotations": 0, "baby": 0, "giant": 0, "n1": 1}
    best: dict[str, int] | None = None
    for n1 in _powers_of_two_below_slots(int(slots)):
        baby = min(int(diagonals), max(1, int(n1) - 1))
        giant = _ceil_div(int(diagonals), max(1, int(n1)))
        rotations = int(baby + giant)
        candidate = {"rotations": int(rotations), "baby": int(baby), "giant": int(giant), "n1": int(n1)}
        if best is None or (int(rotations), -int(n1)) < (int(best["rotations"]), -int(best["n1"])):
            best = candidate
    assert best is not None
    return dict(best)


def _rho_hat_from_diagonal_count(diagonal_count: int, *, slots: int) -> int:
    return int(_bsgs_hat_from_diagonal_count(int(diagonal_count), slots=int(slots))["rotations"])


def _conv_missing_halo_rows(edge: EdgeInfo, layout: LayoutState) -> int:
    if str(edge.op_kind) not in {"conv2d", "avgpool2d"}:
        return 0
    missing_top = max(0, int(edge.requirement.top_beta) - int(layout.top_beta))
    missing_bottom = max(0, int(edge.requirement.bottom_beta) - int(layout.bottom_beta))
    return int(missing_top + missing_bottom)


def _lt_ct_pt_mults(edge: EdgeInfo, layout: LayoutState) -> int:
    if (str(edge.op_kind) in {"add", "concat", "input"}):
        return 0
    if edge.output_shape is None or edge.output_fhe_shape is None:
        return 0

    input_phys_h, input_phys_w = _layout_one_channel_physical_shape_for_layout(
        clear_shape=edge.shape,
        layout=layout,
    )
    output_gap = _output_gap_for_edge(edge)
    output_phys_h = max(1, int(edge.output_shape[2]) * int(output_gap))
    output_phys_w = max(1, int(edge.output_shape[3]) * int(output_gap))
    input_multiplier, output_multiplier = _lt_channel_multipliers(edge)
    full_diags = max(1, int(edge.kernel_size[0]) * int(edge.kernel_size[1]))

    if _template_slot_mapping_count(edge) <= int(TEMPLATE_ESTIMATOR_MAX_SLOT_MAPPINGS):
        one_channel = _one_channel_lt_ct_pt_mult_count_cached(
            op_kind=str(edge.op_kind),
            input_h=int(edge.shape[2]),
            input_w=int(edge.shape[3]),
            output_h=int(edge.output_shape[2]),
            output_w=int(edge.output_shape[3]),
            input_phys_h=int(input_phys_h),
            input_phys_w=int(input_phys_w),
            output_phys_h=int(output_phys_h),
            output_phys_w=int(output_phys_w),
            input_gap=int(layout.gap),
            output_gap=int(output_gap),
            top_beta=_layout_physical_top_beta(layout),
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
    else:
        input_blocks = max(1, _ceil_div(int(input_phys_h) * int(input_phys_w), int(edge.slots)))
        output_blocks = max(1, _ceil_div(int(output_phys_h) * int(output_phys_w), int(edge.slots)))
        one_channel = int(input_blocks * output_blocks * int(full_diags))

    return int(int(one_channel) * max(1, int(input_multiplier)) * max(1, int(output_multiplier)))


def _count_only_rotation_estimate(edge: EdgeInfo, layout: LayoutState) -> RotationEstimate:
    if (str(edge.op_kind) in {"add", "concat", "input"}):
        return RotationEstimate(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    if edge.output_shape is None or edge.output_fhe_shape is None:
        return RotationEstimate(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    input_phys_h, input_phys_w = _layout_one_channel_physical_shape_for_layout(
        clear_shape=edge.shape,
        layout=layout,
    )
    output_gap = _output_gap_for_edge(edge)
    output_phys_h = max(1, int(edge.output_shape[2]) * int(output_gap))
    output_phys_w = max(1, int(edge.output_shape[3]) * int(output_gap))
    input_multiplier, output_multiplier = _lt_channel_multipliers(edge)
    input_blocks = max(1, _ceil_div(int(input_phys_h) * int(input_phys_w), int(edge.slots)))
    output_blocks = max(
        1,
        _ceil_div(
            int(output_phys_h) * int(output_phys_w),
            int(edge.slots),
        ),
    )
    bsgs_groups = int(input_blocks * output_blocks * max(1, int(input_multiplier)))
    full_diags = max(
        1,
        int(edge.kernel_size[0]) * int(edge.kernel_size[1]) - int(str(edge.op_kind) in {"conv2d", "avgpool2d"}),
    )

    missing_rows = _conv_missing_halo_rows(edge, layout)
    missing_diags = 0
    if int(missing_rows) > 0:
        missing_diags = min(
            int(full_diags),
            int(missing_rows) * max(1, int(edge.kernel_size[1])),
        )
    # Missing halo does not remove the local stencil work. It adds boundary
    # recovery/reassembly work on top of the local LT, which is exactly the
    # distinction the DP needs when comparing native halo against fused
    # cross-ciphertext handling.
    local_diags = int(full_diags)

    local_hat = _bsgs_hat_from_diagonal_count(int(local_diags), slots=int(edge.slots))
    recovery_hat = _bsgs_hat_from_diagonal_count(int(missing_diags), slots=int(edge.slots))
    local_rho = int(local_hat["rotations"])
    recovery_rho = int(recovery_hat["rotations"])
    local_baby = int(local_hat["baby"])
    local_giant = int(local_hat["giant"])
    recovery_baby = int(recovery_hat["baby"])
    recovery_giant = int(recovery_hat["giant"])
    repeated = max(1, int(output_multiplier))
    adjacency_counts: tuple[int, ...] = ()
    shift_indices: tuple[int, ...] = ()
    sampled_source_shift_sets: tuple[tuple[int, ...], ...] = ()
    if (
        str(edge.op_kind) in {"conv2d", "avgpool2d", "conv_transpose2d"}
        and _template_slot_mapping_count(edge) <= int(TEMPLATE_ESTIMATOR_MAX_SLOT_MAPPINGS)
    ):
        adjacency_counts = _one_channel_lt_adjacency_counts_cached(
            op_kind=str(edge.op_kind),
            input_h=int(edge.shape[2]),
            input_w=int(edge.shape[3]),
            output_h=int(edge.output_shape[2]),
            output_w=int(edge.output_shape[3]),
            input_phys_h=int(input_phys_h),
            input_phys_w=int(input_phys_w),
            output_phys_h=int(output_phys_h),
            output_phys_w=int(output_phys_w),
            input_gap=int(layout.gap),
            output_gap=int(output_gap),
            top_beta=_layout_physical_top_beta(layout),
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
        sampled_source_shift_sets = _sampled_source_shift_sets_cached(
            op_kind=str(edge.op_kind),
            input_h=int(edge.shape[2]),
            input_w=int(edge.shape[3]),
            output_h=int(edge.output_shape[2]),
            output_w=int(edge.output_shape[3]),
            input_phys_h=int(input_phys_h),
            input_phys_w=int(input_phys_w),
            output_phys_h=int(output_phys_h),
            output_phys_w=int(output_phys_w),
            input_gap=int(layout.gap),
            output_gap=int(output_gap),
            top_beta=_layout_physical_top_beta(layout),
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
        shift_indices = _representative_shift_indices_cached(
            op_kind=str(edge.op_kind),
            input_h=int(edge.shape[2]),
            input_w=int(edge.shape[3]),
            output_h=int(edge.output_shape[2]),
            output_w=int(edge.output_shape[3]),
            input_phys_h=int(input_phys_h),
            input_phys_w=int(input_phys_w),
            output_phys_h=int(output_phys_h),
            output_phys_w=int(output_phys_w),
            input_gap=int(layout.gap),
            output_gap=int(output_gap),
            top_beta=_layout_physical_top_beta(layout),
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
    local_programs = int(
        (sum(int(value) for value in adjacency_counts) if adjacency_counts else int(input_blocks * output_blocks))
        * max(1, int(input_multiplier))
    )
    recovery_programs = int(0 if int(missing_diags) == 0 else output_blocks * max(1, int(input_multiplier)))
    source_group_count_one_channel = int(len(adjacency_counts) if adjacency_counts else int(input_blocks))
    source_groups = int(source_group_count_one_channel * max(1, int(input_multiplier)))
    recovery_transforms_per_source = int(output_blocks * int(repeated))
    if sampled_source_shift_sets:
        local_rotations_one_channel = 0
        local_baby_one_channel = 0
        local_giant_one_channel = 0
        for source_index, fanout in enumerate(adjacency_counts):
            diag_set = sampled_source_shift_sets[min(int(source_index), len(sampled_source_shift_sets) - 1)]
            cost = _shared_bsgs_group_cost(
                (diag_set,),
                slots=int(edge.slots),
                repeated_transform_count=int(max(1, int(fanout)) * int(repeated)),
            )
            local_rotations_one_channel += int(cost["rotations"])
            local_baby_one_channel += int(cost["baby_rotations"])
            local_giant_one_channel += int(cost["giant_rotations"])
        local_rotations = int(local_rotations_one_channel * max(1, int(input_multiplier)))
        local_baby = int(local_baby_one_channel * max(1, int(input_multiplier)))
        local_giant = int(local_giant_one_channel * max(1, int(input_multiplier)))
        local_rho = int(
            max(
                (
                    _shared_bsgs_group_cost((diag_set,), slots=int(edge.slots), repeated_transform_count=1)[
                        "rotations"
                    ]
                    for diag_set in sampled_source_shift_sets
                ),
                default=int(local_rho),
            )
        )
    elif shift_indices:
        local_rotations_one_channel = 0
        local_baby_one_channel = 0
        local_giant_one_channel = 0
        for fanout in adjacency_counts:
            cost = _shared_bsgs_group_cost(
                (shift_indices,),
                slots=int(edge.slots),
                repeated_transform_count=int(max(1, int(fanout)) * int(repeated)),
            )
            local_rotations_one_channel += int(cost["rotations"])
            local_baby_one_channel += int(cost["baby_rotations"])
            local_giant_one_channel += int(cost["giant_rotations"])
        local_rotations = int(local_rotations_one_channel * max(1, int(input_multiplier)))
        local_baby = int(local_baby_one_channel * max(1, int(input_multiplier)))
        local_giant = int(local_giant_one_channel * max(1, int(input_multiplier)))
        single_cost = _shared_bsgs_group_cost((shift_indices,), slots=int(edge.slots), repeated_transform_count=1)
        local_rho = int(single_cost["rotations"])
    else:
        transforms_per_source = int(output_blocks * int(repeated))
        local_rotations = int(
            int(source_groups) * (int(local_baby) + int(transforms_per_source) * int(local_giant))
        )
    local_unfused = int(local_programs * int(local_rho) * int(repeated))
    input_cross_rotations = int(
        math.ceil(
            float(source_groups)
            * float(0 if int(recovery_programs) == 0 else int(recovery_baby) + int(recovery_transforms_per_source) * int(recovery_giant))
            * float(INPUT_CROSS_RECOVERY_ROTATION_MULTIPLIER)
        )
    )
    input_cross_unfused = int(recovery_programs * int(recovery_rho) * int(repeated))
    total_programs = int(local_programs + recovery_programs)
    transforms = int(total_programs * int(repeated))
    local_baby_total = int(local_baby if shift_indices else int(source_groups) * int(local_baby))
    input_cross_baby_total = int(
        math.ceil(
            float(0 if int(recovery_programs) == 0 else int(source_groups) * int(recovery_baby))
            * float(INPUT_CROSS_RECOVERY_ROTATION_MULTIPLIER)
        )
    )
    baby_rotations = int(local_baby_total + input_cross_baby_total)
    total_rotations = int(local_rotations + input_cross_rotations)
    giant_rotations = int(max(0, int(total_rotations) - int(baby_rotations)))
    unfused_rotations = int(local_unfused + math.ceil(float(input_cross_unfused) * float(INPUT_CROSS_RECOVERY_ROTATION_MULTIPLIER)))
    return RotationEstimate(
        input_cross=int(input_cross_rotations),
        local_submatrix=int(local_rotations),
        output_materialize=0,
        bsgs_groups=int(source_groups),
        transforms=int(transforms),
        baby_rotations=int(baby_rotations),
        giant_rotations=int(giant_rotations),
        input_channel_multiplier=int(input_multiplier),
        output_channel_multiplier=int(output_multiplier),
        local_programs=int(local_programs),
        recovery_programs=int(recovery_programs),
        rho_hat_per_program=int(max(int(local_rho), int(recovery_rho))),
        unfused_rotations=int(unfused_rotations),
        same_input_fusion_savings=int(max(0, int(unfused_rotations) - int(total_rotations))),
        ct_pt_mults=int(_lt_ct_pt_mults(edge, layout)),
        estimator=LAYOUT_ESTIMATOR_COUNT_ONLY,
    )


def _template_slot_mapping_count(edge: EdgeInfo) -> int:
    if str(edge.op_kind) in {"conv2d", "avgpool2d"}:
        output_shape = edge.output_shape or edge.shape
        return int(output_shape[2]) * int(output_shape[3]) * int(edge.kernel_size[0]) * int(edge.kernel_size[1])
    if str(edge.op_kind) == "conv_transpose2d":
        return int(edge.shape[2]) * int(edge.shape[3]) * int(edge.kernel_size[0]) * int(edge.kernel_size[1])
    return 0


def _template_rotation_estimate(edge: EdgeInfo, layout: LayoutState) -> RotationEstimate:
    """Cached unweighted-diagonal estimator for close-call DP candidates.

    This constructs only slot-offset sets for one logical channel template. It
    deliberately does not materialize plaintext weights or full Toeplitz
    matrices, so it is much cheaper than the descriptor oracle but still models
    exact BSGS baby/giant schedules for the candidate geometry.
    """

    count_only = _count_only_rotation_estimate(edge, layout)
    if str(edge.op_kind) not in {"conv2d", "avgpool2d", "conv_transpose2d"}:
        return count_only
    if _template_slot_mapping_count(edge) > int(TEMPLATE_ESTIMATOR_MAX_SLOT_MAPPINGS):
        return count_only
    if edge.output_shape is None:
        return count_only

    input_phys_h, input_phys_w = _layout_one_channel_physical_shape_for_layout(
        clear_shape=edge.shape,
        layout=layout,
    )
    output_gap = _output_gap_for_edge(edge)
    output_phys_h = max(1, int(edge.output_shape[2]) * int(output_gap))
    output_phys_w = max(1, int(edge.output_shape[3]) * int(output_gap))
    groups = _one_channel_lt_groups_cached(
        op_kind=str(edge.op_kind),
        input_h=int(edge.shape[2]),
        input_w=int(edge.shape[3]),
        output_h=int(edge.output_shape[2]),
        output_w=int(edge.output_shape[3]),
        input_phys_h=int(input_phys_h),
        input_phys_w=int(input_phys_w),
        output_phys_h=int(output_phys_h),
        output_phys_w=int(output_phys_w),
        input_gap=int(layout.gap),
        output_gap=int(output_gap),
        top_beta=_layout_physical_top_beta(layout),
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
    if not groups:
        return count_only

    input_multiplier, output_multiplier = _lt_channel_multipliers(edge)
    repeated = max(1, int(output_multiplier))
    local_rotations = 0
    local_baby = 0
    local_giant = 0
    transforms = 0
    unfused_local = 0
    local_programs = 0
    rho_values: list[int] = []
    for source_group in groups:
        source_cost = _shared_bsgs_group_cost(
            source_group,
            slots=int(edge.slots),
            repeated_transform_count=int(repeated),
        )
        local_rotations += int(source_cost["rotations"])
        local_baby += int(source_cost["baby_rotations"])
        local_giant += int(source_cost["giant_rotations"])
        transforms += int(source_cost["transforms"])
        local_programs += int(len(source_group))
        for diag_set in source_group:
            single = _shared_bsgs_group_cost(
                (diag_set,),
                slots=int(edge.slots),
                repeated_transform_count=1,
            )
            rho_values.append(int(single["rotations"]))
            unfused_local += int(single["rotations"]) * int(repeated)

    local_rotations *= max(1, int(input_multiplier))
    local_baby *= max(1, int(input_multiplier))
    local_giant *= max(1, int(input_multiplier))
    transforms *= max(1, int(input_multiplier))
    local_programs *= max(1, int(input_multiplier))
    unfused_local *= max(1, int(input_multiplier))

    input_cross = int(count_only.input_cross)
    total_rotations = int(local_rotations + input_cross)
    unfused = int(unfused_local + max(0, int(count_only.unfused_rotations) - int(count_only.local_submatrix)))
    return RotationEstimate(
        input_cross=int(input_cross),
        local_submatrix=int(local_rotations),
        output_materialize=int(count_only.output_materialize),
        bsgs_groups=int(len(groups) * max(1, int(input_multiplier))),
        transforms=int(transforms + int(count_only.recovery_programs) * int(repeated)),
        baby_rotations=int(local_baby),
        giant_rotations=int(local_giant),
        input_channel_multiplier=int(input_multiplier),
        output_channel_multiplier=int(output_multiplier),
        local_programs=int(local_programs),
        recovery_programs=int(count_only.recovery_programs),
        rho_hat_per_program=int(max(rho_values) if rho_values else count_only.rho_hat_per_program),
        unfused_rotations=int(unfused),
        same_input_fusion_savings=int(max(0, int(unfused) - int(total_rotations))),
        ct_pt_mults=int(_lt_ct_pt_mults(edge, layout)),
        estimator=LAYOUT_ESTIMATOR_TEMPLATE,
    )


def _lt_rotation_stats(edge: EdgeInfo, layout: LayoutState, *, estimator: str | None = None) -> dict[str, int]:
    mode = _normalize_layout_estimator(estimator)
    if mode == LAYOUT_ESTIMATOR_TEMPLATE:
        return _template_rotation_estimate(edge, layout).to_lt_stats()
    return _count_only_rotation_estimate(edge, layout).to_lt_stats()


def _lt_rotations(edge: EdgeInfo, layout: LayoutState, *, estimator: str | None = None) -> int:
    return int(_lt_rotation_stats(edge, layout, estimator=estimator)["rotations"])


def _relayout_halo_side_count(layout: LayoutState) -> int:
    return int(
        max(1, int(layout.tile_count))
        * (
            int(_layout_physical_top_beta(layout) > 0)
            + int(_layout_physical_bottom_beta(layout) > 0)
        )
    )


def _relayout_rotations(layouts: Iterable[LayoutState]) -> int:
    return int(sum(_relayout_halo_side_count(layout) for layout in layouts))


def _relayout_mask_mults(layouts: Iterable[LayoutState]) -> int:
    return int(sum(_relayout_halo_side_count(layout) for layout in layouts))


def _relayout_depth_units(layouts: Iterable[LayoutState]) -> int:
    return int(
        sum(
            1
            for layout in layouts
            if _layout_physical_top_beta(layout) > 0 or _layout_physical_bottom_beta(layout) > 0
        )
    )


def _relayout_transition_estimate(
    *,
    source_layout: LayoutState | None,
    target_layout: LayoutState,
    relayout: bool,
) -> dict[str, int]:
    if not bool(relayout):
        return {
            "rotation_count": 0,
            "mask_mult_count": 0,
            "sparse_lt_count": 0,
            "depth_estimate": 0,
        }
    if source_layout is None:
        rotations = _relayout_rotations((target_layout,))
        masks = _relayout_mask_mults((target_layout,))
        return {
            "rotation_count": int(rotations),
            "mask_mult_count": int(masks),
            "sparse_lt_count": 0,
            "depth_estimate": int(_relayout_depth_units((target_layout,))),
        }

    source_has_halo = bool(
        _layout_physical_top_beta(source_layout) > 0 or _layout_physical_bottom_beta(source_layout) > 0
    )
    target_has_halo = bool(
        _layout_physical_top_beta(target_layout) > 0 or _layout_physical_bottom_beta(target_layout) > 0
    )
    if not source_has_halo and target_has_halo:
        side_count = _relayout_halo_side_count(target_layout)
        return {
            "rotation_count": int(side_count),
            "mask_mult_count": int(side_count),
            "sparse_lt_count": 0,
            "depth_estimate": 1,
        }
    if source_has_halo and not target_has_halo:
        return {
            "rotation_count": 0,
            "mask_mult_count": int(max(1, int(source_layout.tile_count))),
            "sparse_lt_count": 0,
            "depth_estimate": 1,
        }
    if source_has_halo and target_has_halo:
        if _same_physical_layout(source_layout, target_layout):
            return {
                "rotation_count": 0,
                "mask_mult_count": 0,
                "sparse_lt_count": 0,
                "depth_estimate": 0,
            }
        side_count = _relayout_halo_side_count(target_layout)
        return {
            "rotation_count": int(side_count),
            "mask_mult_count": int(side_count),
            "sparse_lt_count": 1,
            "depth_estimate": 1,
        }
    return {
        "rotation_count": 0,
        "mask_mult_count": int(max(1, int(target_layout.tile_count))),
        "sparse_lt_count": 1,
        "depth_estimate": 1,
    }


def _node_layout_row(
    node: str,
    layout: LayoutState,
    compact: LayoutState | None,
    *,
    relayout: bool,
    reason: str,
    producer_materialized_halo: bool = False,
    producer_materialized_halo_reason: str | None = None,
    producer_fused_rotation_estimate: int = 0,
    physical_layout: str | None = None,
    shape: Sequence[int] | None = None,
    fhe_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    if physical_layout is None:
        physical_layout = PHYSICAL_LOGICAL_HALO if int(layout.top_beta) > 0 or int(layout.bottom_beta) > 0 else PHYSICAL_COMPACT
    if shape is not None:
        fhe_shape = _fhe_shape_for_layout(
            shape=tuple(int(value) for value in shape),
            layout=layout,
        )
    return {
        "node": str(node),
        "shape": [] if shape is None else [int(value) for value in shape],
        "fhe_shape": [] if fhe_shape is None else [int(value) for value in fhe_shape],
        "selected_layout": layout.to_dict(),
        "physical_layout": str(physical_layout),
        "compact_layout": {} if compact is None else compact.to_dict(),
        "output_relayout": bool(relayout),
        "output_relayout_reason": str(reason) if bool(relayout) else "",
        "producer_materialized_halo": bool(producer_materialized_halo),
        "producer_materialized_halo_reason": (
            str(producer_materialized_halo_reason if producer_materialized_halo_reason is not None else reason)
            if bool(producer_materialized_halo)
            else ""
        ),
        "producer_fused_rotation_estimate": int(producer_fused_rotation_estimate),
        "producer_fused_depth_estimate": 0,
        "relayout_rotation_estimate": int(_relayout_rotations((layout,)) if bool(relayout) else 0),
        "relayout_mask_mult_estimate": int(_relayout_mask_mults((layout,)) if bool(relayout) else 0),
        "relayout_depth_estimate": int(_relayout_depth_units((layout,)) if bool(relayout) else 0),
    }


def _producer_fused_materialization_estimate(
    module: Any | None,
    *,
    incoming: Sequence[EdgeInfo],
    incoming_rows: Sequence[dict[str, Any]],
    outgoing: Sequence[EdgeInfo],
    output_layout: LayoutState | None,
    slots: int,
) -> dict[str, Any]:
    if output_layout is None or not incoming or not incoming_rows or not outgoing:
        return {"enabled": False, "rotation_count": 0}
    if not _producer_fused_output_allowed(module):
        return {"enabled": False, "rotation_count": 0}
    if int(output_layout.top_beta) == 0 and int(output_layout.bottom_beta) == 0:
        return {"enabled": False, "rotation_count": 0}
    semantic = _operator_semantic_output_layout(
        module,
        LayoutState(**dict(incoming_rows[0]["selected_layout"])),
        outgoing[0],
        slots=int(slots),
    )
    grows_halo = (
        int(output_layout.gap) == int(semantic.gap)
        and (
            int(output_layout.top_beta) > int(semantic.top_beta)
            or int(output_layout.bottom_beta) > int(semantic.bottom_beta)
        )
    )
    if not bool(grows_halo):
        return {"enabled": False, "rotation_count": 0}
    # This is not a separate LT. The producer emits the requested boundary rows
    # as part of its existing transform; it may change diagonal work, but should
    # not be charged as an independent rotation/depth path.
    return {
        "enabled": True,
        "rotation_count": 0,
    }


def _halo_slots_for_rows(edge_rows: Iterable[dict[str, Any]]) -> int:
    return int(
        sum(
            max(0, int(row["selected_layout"]["stored_slots"]) - int(row["selected_layout"]["core_slots"]))
            for row in edge_rows
        )
    )


def _edge_linear_cost(row: dict[str, Any]) -> float:
    rotation_cost = int(row.get("planner_rotation_cost_estimate", row["lt_bsgs_rotation_estimate"]) or 0)
    activation_cost = int(row.get("activation_ct_mult_estimate", 0) or 0)
    return float(rotation_cost + activation_cost)


def _relayout_linear_cost(layout: LayoutState) -> float:
    return float(_relayout_rotations((layout,)))


def _policy_linear_cost(edge_rows: Iterable[dict[str, Any]], relayout_layouts: Iterable[LayoutState]) -> float:
    rows = list(edge_rows)
    relayouts = list(relayout_layouts)
    if rows:
        relayout_rotation_cost = sum(
            int(row.get("relayout_rotation_estimate", 0) or 0)
            for row in rows
            if bool(row.get("relayout", False))
        )
        return (
            float(sum(_edge_linear_cost(row) for row in rows))
            + float(relayout_rotation_cost)
        )
    return float(sum(_relayout_linear_cost(layout) for layout in relayouts))


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
    edge_relayout_rows = [row for row in edge_rows if bool(row.get("relayout", False))]
    relayout_rotation_estimate = int(
        sum(int(row.get("relayout_rotation_estimate", 0) or 0) for row in edge_relayout_rows)
    )
    relayout_mask_mult_estimate = int(
        sum(int(row.get("relayout_mask_mult_estimate", 0) or 0) for row in edge_relayout_rows)
    )
    relayout_depth_estimate = int(
        sum(int(row.get("relayout_depth_estimate", 0) or 0) for row in edge_relayout_rows)
    )
    producer_fused_rows = [
        row
        for row in node_layouts
        if bool(row.get("producer_materialized_halo", False))
    ]
    producer_fused_rotation_estimate = int(
        sum(int(row.get("producer_fused_rotation_estimate", 0) or 0) for row in producer_fused_rows)
    )
    consumer_fused_rows = [
        row
        for row in edge_rows
        if bool(row.get("consumer_fused_relayout", False))
    ]
    consumer_fused_rotation_estimate = int(
        sum(int(row.get("consumer_fused_rotation_estimate", 0) or 0) for row in consumer_fused_rows)
    )
    add_relayout_depth_estimate = int(
        sum(
            int(row.get("relayout_depth_estimate", 0) or 0)
            for row in edge_relayout_rows
            if _is_join_op(str(row.get("op_kind", "")))
        )
    )
    lt_rotation_estimate = int(sum(int(row["lt_bsgs_rotation_estimate"]) for row in edge_rows))
    unfused_lt_rotation_estimate = int(
        sum(int(row.get("lt_unfused_rotation_estimate", row["lt_bsgs_rotation_estimate"]) or 0) for row in edge_rows)
    )
    planner_rotation_cost_estimate = int(
        sum(int(row.get("planner_rotation_cost_estimate", row["lt_bsgs_rotation_estimate"]) or 0) for row in edge_rows)
    )
    reported_rotation_estimate = (
        int(unfused_lt_rotation_estimate)
        if str(policy) == "orion_dense"
        else int(planner_rotation_cost_estimate + relayout_rotation_estimate + producer_fused_rotation_estimate)
    )
    lt_ct_pt_mult_estimate = int(
        sum(int(row.get("lt_ct_pt_mult_estimate", 0) or 0) for row in edge_rows)
    )
    activation_ct_mult_estimate = int(
        sum(int(row.get("activation_ct_mult_estimate", 0) or 0) for row in edge_rows)
    )
    ct_pt_mult_estimate = int(lt_ct_pt_mult_estimate + relayout_mask_mult_estimate + activation_ct_mult_estimate)
    halo_slots = int(stored_slots - core_slots)
    bootstrap_proxy = 0
    redundancy = 0.0 if int(core_slots) == 0 else float(stored_slots - core_slots) / float(core_slots)
    objective = (
        float(relayout_rotation_estimate)
        + float(planner_rotation_cost_estimate)
        + float(producer_fused_rotation_estimate)
        + float(activation_ct_mult_estimate)
    )
    return PolicyPlan(
        policy=str(policy),
        policy_label=POLICY_LABELS[str(policy)],
        metric_source="planner_estimate",
        relayouts=int(len(edge_relayout_rows)),
        halo_redundancy_ratio=float(redundancy),
        total_ciphertext_tiles=int(tile_count),
        stored_slots=int(stored_slots),
        relayout_rotation_estimate=int(relayout_rotation_estimate),
        relayout_mask_mult_estimate=int(relayout_mask_mult_estimate),
        relayout_depth_estimate=int(relayout_depth_estimate),
        producer_fused_materialization_count=int(len(producer_fused_rows)),
        producer_fused_rotation_estimate=int(producer_fused_rotation_estimate),
        consumer_fused_relayout_count=int(len(consumer_fused_rows)),
        consumer_fused_rotation_estimate=int(consumer_fused_rotation_estimate),
        lt_bsgs_rotation_estimate=int(lt_rotation_estimate),
        planner_rotation_cost_estimate=int(planner_rotation_cost_estimate),
        reported_rotation_estimate=int(reported_rotation_estimate),
        lt_ct_pt_mult_estimate=int(lt_ct_pt_mult_estimate),
        activation_ct_mult_estimate=int(activation_ct_mult_estimate),
        ct_pt_mult_estimate=int(ct_pt_mult_estimate),
        bootstrap_proxy=int(bootstrap_proxy),
        objective=float(objective),
        edge_layouts=tuple(edge_rows),
        node_layouts=tuple(dict(row) for row in node_layouts),
    )


def _align_add_inputs(dag: NetworkDAG, rows_by_edge: dict[str, dict[str, Any]], relayout_layouts: list[LayoutState], *, slots: int) -> None:
    for node in dag.topological_sort():
        module = dag.nodes[node].get("module")
        if not _is_join_module(module):
            continue
        incoming = [f"{source}->{node}" for source in dag.predecessors(node)]
        layouts = [LayoutState(**rows_by_edge[edge_id]["selected_layout"]) for edge_id in incoming if edge_id in rows_by_edge]
        if len(layouts) < 2 or len({layout.key() for layout in layouts}) == 1:
            continue
        target = max(layouts, key=lambda item: (int(item.top_beta + item.bottom_beta), int(item.stored_slots), int(item.stride)))
        for edge_id in incoming:
            if edge_id not in rows_by_edge:
                continue
            previous = LayoutState(**rows_by_edge[edge_id]["selected_layout"])
            if _same_layout(previous, target):
                continue
            rows_by_edge[edge_id]["source_layout"] = previous.to_dict()
            rows_by_edge[edge_id]["selected_layout"] = target.to_dict()
            rows_by_edge[edge_id]["target_layout"] = target.to_dict()
            rows_by_edge[edge_id]["relayout"] = True
            rows_by_edge[edge_id]["relayout_reason"] = "add_input_alignment"
            estimate = _relayout_transition_estimate(
                source_layout=previous,
                target_layout=target,
                relayout=True,
            )
            rows_by_edge[edge_id]["relayout_rotation_estimate"] = int(estimate["rotation_count"])
            rows_by_edge[edge_id]["relayout_mask_mult_estimate"] = int(estimate["mask_mult_count"])
            rows_by_edge[edge_id]["relayout_sparse_lt_estimate"] = int(estimate["sparse_lt_count"])
            rows_by_edge[edge_id]["relayout_depth_estimate"] = int(estimate["depth_estimate"])
            rows_by_edge[edge_id]["lt_bsgs_rotation_estimate"] = int(
                rows_by_edge[edge_id]["lt_bsgs_rotation_estimate"]
            )
            relayout_layouts.append(target)


def _fixed_max_layout_for_edge(
    edge: EdgeInfo,
    *,
    global_alpha: int,
    global_beta: int,
    max_demand_layouts: dict[str, LayoutState] | None = None,
    slots: int,
) -> LayoutState:
    if max_demand_layouts is not None and str(edge.edge_id) in max_demand_layouts:
        return max_demand_layouts[str(edge.edge_id)]
    fixed_like = _layout_for_shape(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        top_beta=int(global_alpha),
        bottom_beta=int(global_beta),
        stride=max(1, int(edge.requirement.stride)),
        slots=int(slots),
    )
    return fixed_like


def _edge_requires_halo(edge: EdgeInfo) -> bool:
    return bool(int(edge.requirement.top_beta) > 0 or int(edge.requirement.bottom_beta) > 0)


def _choose_non_dp_input_layout(
    policy: str,
    edge: EdgeInfo,
    source_layout: LayoutState,
    *,
    global_alpha: int,
    global_beta: int,
    max_demand_layouts: dict[str, LayoutState] | None = None,
    slots: int,
) -> tuple[LayoutState, str]:
    if str(policy) == "fixed_max":
        layout = _fixed_max_layout_for_edge(
            edge,
            global_alpha=int(global_alpha),
            global_beta=int(global_beta),
            max_demand_layouts=max_demand_layouts,
            slots=int(slots),
        )
        return layout, "fixed_max_restore_halo"
    if str(policy) == "greedy":
        if source_layout.covers(edge.requirement):
            return _layout_with_stride(source_layout, max(int(source_layout.stride), int(edge.requirement.stride))), ""
        return (
            _fixed_max_layout_for_edge(
                edge,
                global_alpha=int(global_alpha),
                global_beta=int(global_beta),
                max_demand_layouts=max_demand_layouts,
                slots=int(slots),
            ),
            "greedy_restore_max_halo",
        )
    if str(policy) == "always":
        return edge.requirement, "always_relayout_to_consumer_requirement"
    return edge.requirement, "consumer_min_layout"


def _choose_non_dp_add_layout(
    policy: str,
    incoming: Sequence[EdgeInfo],
    live: dict[str, LayoutState],
    *,
    global_alpha: int,
    global_beta: int,
    max_demand_layouts: dict[str, LayoutState] | None = None,
    slots: int,
) -> LayoutState:
    if not incoming:
        raise ValueError("cannot choose Add layout without incoming edges")
    if str(policy) == "fixed_max":
        return _fixed_max_layout_for_edge(
            incoming[0],
            global_alpha=int(global_alpha),
            global_beta=int(global_beta),
            max_demand_layouts=max_demand_layouts,
            slots=int(slots),
        )
    if str(policy) == "greedy":
        live_layouts = [live[edge.source] for edge in incoming]
        required = _max_layout(incoming, slots=int(slots))
        if all(layout.covers(required) for layout in live_layouts):
            return _max_layout_states(
                live_layouts,
                shape=incoming[0].shape,
                gap=int(incoming[0].compact.gap),
                slots=int(slots),
            )
        return _fixed_max_layout_for_edge(
            incoming[0],
            global_alpha=int(global_alpha),
            global_beta=int(global_beta),
            max_demand_layouts=max_demand_layouts,
            slots=int(slots),
        )
    if str(policy) == "always":
        return _max_layout(incoming, slots=int(slots))
    return incoming[0].requirement


def _source_initial_layout(
    policy: str,
    outgoing: Sequence[EdgeInfo],
    *,
    global_alpha: int,
    global_beta: int,
    max_demand_layouts: dict[str, LayoutState] | None = None,
    slots: int,
) -> LayoutState | None:
    if not outgoing:
        return None
    if str(policy) in {"fixed_max", "greedy"}:
        return _fixed_max_layout_for_edge(
            outgoing[0],
            global_alpha=int(global_alpha),
            global_beta=int(global_beta),
            max_demand_layouts=max_demand_layouts,
            slots=int(slots),
        )
    local_need = _max_layout(outgoing, slots=int(slots))
    return local_need


def _no_halo_layout_for_edge(edge: EdgeInfo, *, slots: int) -> LayoutState:
    return _layout_for_shape(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        top_beta=0,
        bottom_beta=0,
        stride=max(1, int(edge.requirement.stride)),
        slots=int(slots),
    )


def _no_halo_layout_from_semantic(
    semantic: LayoutState,
    *,
    edge: EdgeInfo,
    slots: int,
) -> LayoutState:
    return _layout_for_shape(
        shape=edge.shape,
        gap=int(semantic.gap),
        top_beta=0,
        bottom_beta=0,
        stride=max(1, int(semantic.stride)),
        slots=int(slots),
    )


def _base_non_dp_policy(policy: str) -> str:
    normalized = str(policy)
    if normalized == "fixed_max_fused":
        return "fixed_max"
    if normalized == "eager_fused":
        return "eager"
    if normalized == "greedy_fused":
        return "greedy"
    if normalized == "always_fused":
        return "always"
    return normalized


def _non_dp_policy_uses_fusion(policy: str) -> bool:
    return str(policy) in {"fixed_max_fused", "eager_fused", "greedy_fused", "always_fused"}


def _layout_has_halo(layout: LayoutState) -> bool:
    return bool(int(layout.top_beta) > 0 or int(layout.bottom_beta) > 0)


def _layout_has_physical_halo(layout: LayoutState) -> bool:
    return bool(_layout_physical_top_beta(layout) > 0 or _layout_physical_bottom_beta(layout) > 0)


def _compact_packing_for_layout(layout: LayoutState) -> str:
    return PHYSICAL_LOGICAL_HALO if _layout_has_halo(layout) else PHYSICAL_COMPACT


def _join_source_layout_for_physical(
    edge: EdgeInfo,
    source_layout: LayoutState,
    source_physical: str,
) -> LayoutState:
    if str(source_physical) != PHYSICAL_COMPACT:
        return source_layout
    return _layout_for_shape(
        shape=edge.shape,
        gap=int(source_layout.gap),
        top_beta=0,
        bottom_beta=0,
        stride=int(source_layout.stride),
        slots=int(edge.slots),
    )


def _join_input_requires_relayout(
    source_layout: LayoutState,
    source_physical: str,
    target_layout: LayoutState,
) -> bool:
    if str(source_physical or "") == PHYSICAL_NATIVE_SOURCE_STRIPE:
        return True
    return not (source_layout.covers(target_layout) and _same_physical_layout(source_layout, target_layout))


def _is_join_module(module: Any | None) -> bool:
    return type(module).__name__ in {"Add", "Concat"}


def _is_join_op(op_kind: str) -> bool:
    return str(op_kind) in {"add", "concat"}


def _layout_for_join_output_shape(
    layout: LayoutState,
    *,
    outgoing: Sequence[EdgeInfo],
    slots: int,
) -> LayoutState:
    if not outgoing:
        return layout
    return _layout_for_shape(
        shape=outgoing[0].shape,
        gap=int(layout.gap),
        top_beta=int(layout.top_beta),
        bottom_beta=int(layout.bottom_beta),
        stride=int(layout.stride),
        slots=int(slots),
        physical_top_beta=_layout_physical_top_beta(layout),
        physical_bottom_beta=_layout_physical_bottom_beta(layout),
        boundary_pruned=bool(layout.boundary_pruned),
    )


def _non_dp_consumer_fused_row(
    policy: str,
    edge: EdgeInfo,
    *,
    source_layout: LayoutState,
    source_physical: str,
) -> dict[str, Any] | None:
    selected_layout = _layout_with_stride(
        source_layout,
        max(int(source_layout.stride), int(edge.requirement.stride)),
    )
    if (
        source_layout.covers(edge.requirement)
        and str(source_physical) == PHYSICAL_NATIVE_SOURCE_STRIPE
        and _conv_native_stripe_candidate_allowed(edge, source_layout)
    ):
        return _edge_row(
            edge,
            selected_layout,
            relayout=False,
            relayout_reason="",
            layout_mode="native_halo_stripe",
            source_layout=source_layout,
            source_physical_layout=str(source_physical),
            physical_layout=PHYSICAL_NATIVE_SOURCE_STRIPE,
            planner_rotation_cost=_consumer_fused_planner_rotation_cost(
                edge,
                selected_layout,
                layout_mode="native_halo_stripe",
            ),
        )
    if str(edge.op_kind) == "conv_transpose2d" and source_layout.covers(edge.requirement):
        physical_layout = PHYSICAL_LOGICAL_HALO if _layout_has_halo(selected_layout) else PHYSICAL_COMPACT
        return _edge_row(
            edge,
            selected_layout,
            relayout=False,
            relayout_reason="",
            layout_mode="compact_tconv_shared",
            source_layout=source_layout,
            source_physical_layout=str(source_physical),
            physical_layout=str(physical_layout),
            consumer_fused_relayout=True,
            consumer_fused_rotation_estimate=0,
            planner_rotation_cost=_consumer_fused_planner_rotation_cost(
                edge,
                selected_layout,
                layout_mode="compact_tconv_shared",
            ),
        )
    if not _compact_align_shared_allowed(edge, source_layout, str(source_physical)):
        return None
    if source_layout.covers(edge.requirement):
        if not _compact_halo_shared_allowed(edge, source_layout, str(source_physical)):
            return None
        layout_mode = "compact_halo_shared"
    else:
        layout_mode = "compact_align_shared"
    physical_layout = PHYSICAL_LOGICAL_HALO if _layout_has_halo(selected_layout) else PHYSICAL_COMPACT
    return _edge_row(
        edge,
        selected_layout,
        relayout=False,
        relayout_reason="",
        layout_mode=str(layout_mode),
        source_layout=source_layout,
        source_physical_layout=str(source_physical),
        physical_layout=str(physical_layout),
        consumer_fused_relayout=True,
        consumer_fused_rotation_estimate=0,
        planner_rotation_cost=_consumer_fused_planner_rotation_cost(
            edge,
            selected_layout,
            layout_mode=str(layout_mode),
        ),
    )


def _non_dp_producer_fused_output_preference(
    policy: str,
    module: Any | None,
    *,
    incoming_rows: Sequence[dict[str, Any]],
    outgoing: Sequence[EdgeInfo],
    live: dict[str, LayoutState],
    edges_by_target: dict[str, list[EdgeInfo]],
    global_alpha: int,
    global_beta: int,
    max_demand_layouts: dict[str, LayoutState] | None = None,
    slots: int,
) -> tuple[LayoutState, str] | None:
    if not incoming_rows or len(outgoing) != 1:
        return None
    if not _producer_fused_output_allowed(module):
        return None
    edge = outgoing[0]
    semantic = _operator_semantic_output_layout(
        module,
        LayoutState(**dict(incoming_rows[0]["selected_layout"])),
        edge,
        slots=int(slots),
    )
    if _is_join_op(str(edge.op_kind)):
        add_incoming = tuple(edges_by_target.get(str(edge.target), ()))
        if not add_incoming:
            return None
        temp_live = dict(live)
        temp_live[str(edge.source)] = semantic
        if any(str(item.source) not in temp_live for item in add_incoming):
            return None
        target_layout = _choose_non_dp_add_layout(
            _base_non_dp_policy(policy),
            add_incoming,
            temp_live,
            global_alpha=int(global_alpha),
            global_beta=int(global_beta),
            max_demand_layouts=max_demand_layouts,
            slots=int(slots),
        )
        reason = "add_input_alignment"
    else:
        target_layout, reason = _choose_non_dp_input_layout(
            _base_non_dp_policy(policy),
            edge,
            semantic,
            global_alpha=int(global_alpha),
            global_beta=int(global_beta),
            max_demand_layouts=max_demand_layouts,
            slots=int(slots),
        )
    if int(target_layout.gap) != int(edge.compact.gap):
        return None
    if _same_physical_layout(semantic, target_layout):
        return None
    reason_token = str(reason or "next_consumer_layout")
    return target_layout, f"{policy}_producer_fused_{reason_token}"


def _plan_non_dp_topological(
    dag: NetworkDAG,
    edges: Sequence[EdgeInfo],
    *,
    policy: str,
    slots: int,
) -> PolicyPlan:
    edges_by_source: dict[str, list[EdgeInfo]] = {}
    edges_by_target: dict[str, list[EdgeInfo]] = {}
    for edge in edges:
        edges_by_source.setdefault(str(edge.source), []).append(edge)
        edges_by_target.setdefault(str(edge.target), []).append(edge)
    topo = [str(node) for node in dag.topological_sort()]
    topo_index = {str(node): index for index, node in enumerate(topo)}
    last_consumer_index = {
        str(source): max(topo_index[str(edge.target)] for edge in source_edges)
        for source, source_edges in edges_by_source.items()
    }
    global_alpha = max(int(edge.requirement.top_beta) for edge in edges)
    global_beta = max(int(edge.requirement.bottom_beta) for edge in edges)
    base_policy = _base_non_dp_policy(str(policy))
    max_demand_layouts = _backward_max_demand_layouts(dag, edges, slots=int(slots))
    fuse_local_relayouts = _non_dp_policy_uses_fusion(str(policy))
    force_every_relayout = str(base_policy) == "always"
    live: dict[str, LayoutState] = {}
    live_physical: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    relayout_layouts: list[LayoutState] = []
    node_layouts: list[dict[str, Any]] = []

    for node in topo:
        incoming = tuple(edges_by_target.get(str(node), ()))
        outgoing = tuple(edges_by_source.get(str(node), ()))
        module = dag.nodes[node].get("module")
        incoming_rows: list[dict[str, Any]] = []

        if incoming and _is_join_module(module):
            target_layout = _choose_non_dp_add_layout(
                str(base_policy),
                incoming,
                live,
                global_alpha=int(global_alpha),
                global_beta=int(global_beta),
                max_demand_layouts=max_demand_layouts,
                slots=int(slots),
            )
            for edge in incoming:
                source_layout = live[edge.source]
                source_physical = live_physical.get(str(edge.source), _compact_packing_for_layout(source_layout))
                physical_source_layout = _join_source_layout_for_physical(
                    edge,
                    source_layout,
                    str(source_physical),
                )
                relayout = bool(force_every_relayout) or _join_input_requires_relayout(
                    physical_source_layout,
                    str(source_physical),
                    target_layout,
                )
                if bool(relayout):
                    relayout_layouts.append(target_layout)
                incoming_rows.append(
                    _edge_row(
                        edge,
                        target_layout,
                        relayout=bool(relayout),
                        relayout_reason=f"{policy}_add_input_alignment" if bool(relayout) else "",
                        lt_rotations=_lt_rotations(edge, target_layout),
                        source_layout=physical_source_layout,
                        source_physical_layout=str(source_physical),
                        physical_layout=_compact_packing_for_layout(target_layout),
                    )
                )
        else:
            for edge in incoming:
                source_layout = live[edge.source]
                source_physical = live_physical.get(str(edge.source), PHYSICAL_COMPACT)
                target_layout, reason = _choose_non_dp_input_layout(
                    str(base_policy),
                    edge,
                    source_layout,
                    global_alpha=int(global_alpha),
                    global_beta=int(global_beta),
                    max_demand_layouts=max_demand_layouts,
                    slots=int(slots),
                )
                relayout = str(edge.source) != "x" and (
                    bool(force_every_relayout) or not _same_physical_layout(source_layout, target_layout)
                )
                fused_row = (
                    _non_dp_consumer_fused_row(
                        str(policy),
                        edge,
                        source_layout=source_layout,
                        source_physical=str(source_physical),
                    )
                    if bool(relayout) and bool(fuse_local_relayouts)
                    else None
                )
                if fused_row is not None:
                    incoming_rows.append(fused_row)
                else:
                    if bool(relayout):
                        relayout_layouts.append(target_layout)
                    incoming_rows.append(
                        _edge_row(
                            edge,
                            target_layout,
                            relayout=bool(relayout),
                            relayout_reason=str(reason) if bool(relayout) else "",
                            lt_rotations=_lt_rotations(edge, target_layout),
                            source_layout=source_layout,
                            source_physical_layout=str(source_physical),
                        )
                    )

        rows.extend(incoming_rows)

        for source in list(live):
            if int(last_consumer_index.get(str(source), -1)) <= int(topo_index[str(node)]):
                live.pop(str(source), None)
                live_physical.pop(str(source), None)

        output_layout: LayoutState | None
        output_physical = PHYSICAL_COMPACT
        producer_fused = {"enabled": False, "rotation_count": 0}
        producer_materialized_halo = False
        producer_materialized_halo_reason = ""
        if incoming_rows:
            if _is_join_module(module):
                output_layout = _layout_for_join_output_shape(
                    LayoutState(**dict(incoming_rows[0]["selected_layout"])),
                    outgoing=outgoing,
                    slots=int(slots),
                )
                output_physical = str(incoming_rows[0].get("physical_layout", PHYSICAL_COMPACT))
            elif outgoing:
                candidates = _operator_output_layout_candidates(
                    module,
                    incoming_rows=incoming_rows,
                    outgoing=outgoing,
                    slots=int(slots),
                )
                output_layout = candidates[0] if candidates else None
                fused_output = (
                    _non_dp_producer_fused_output_preference(
                        str(policy),
                        module,
                        incoming_rows=incoming_rows,
                        outgoing=outgoing,
                        live=live,
                        edges_by_target=edges_by_target,
                        global_alpha=int(global_alpha),
                        global_beta=int(global_beta),
                        max_demand_layouts=max_demand_layouts,
                        slots=int(slots),
                    )
                    if bool(fuse_local_relayouts)
                    else None
                )
                if fused_output is not None:
                    output_layout, producer_materialized_halo_reason = fused_output
                if output_layout is not None:
                    producer_fused = _producer_fused_materialization_estimate(
                        module,
                        incoming=incoming,
                        incoming_rows=incoming_rows,
                        outgoing=outgoing,
                        output_layout=output_layout,
                        slots=int(slots),
                    )
                    output_physical = _output_physical_layout(
                        module,
                        incoming_rows=incoming_rows,
                        output_layout=output_layout,
                        producer_fused=producer_fused,
                        slots=int(slots),
                    )
                    logical_halo_materialized = bool(
                        str(output_physical) == PHYSICAL_LOGICAL_HALO
                        and _layout_has_physical_halo(output_layout)
                        and _producer_fused_output_allowed(module)
                    )
                    producer_materialized_halo = bool(producer_fused["enabled"]) or bool(logical_halo_materialized)
                    if not producer_materialized_halo_reason:
                        producer_materialized_halo_reason = (
                            f"{policy}_producer_materialized_halo"
                            if bool(producer_fused["enabled"])
                            else (
                                f"{policy}_logical_halo_materialized_output"
                                if bool(logical_halo_materialized)
                                else ""
                            )
                        )
            else:
                output_layout = None
        else:
            output_layout = _source_initial_layout(
                str(base_policy),
                outgoing,
                global_alpha=int(global_alpha),
                global_beta=int(global_beta),
                max_demand_layouts=max_demand_layouts,
                slots=int(slots),
            )
            output_physical = PHYSICAL_LOGICAL_HALO if output_layout is not None and _layout_has_halo(output_layout) else PHYSICAL_COMPACT

        if output_layout is not None and outgoing:
            live[str(node)] = output_layout
            live_physical[str(node)] = str(output_physical)
            node_layouts.append(
                _node_layout_row(
                    str(node),
                    output_layout,
                    outgoing[0].compact,
                    relayout=False,
                    reason="",
                    producer_materialized_halo=bool(producer_materialized_halo),
                    producer_materialized_halo_reason=str(producer_materialized_halo_reason),
                    producer_fused_rotation_estimate=int(producer_fused["rotation_count"]),
                    physical_layout=str(output_physical),
                    shape=outgoing[0].shape,
                    fhe_shape=outgoing[0].fhe_shape,
                )
            )

    rows.sort(key=lambda row: (topo_index.get(str(row["source"]), 10**9), topo_index.get(str(row["target"]), 10**9)))
    return _finalize_policy(
        policy=str(policy),
        edge_rows=rows,
        relayout_layouts=relayout_layouts,
        node_layouts=node_layouts,
    )


def _plan_fixed_max(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    return _plan_non_dp_topological(dag, edges, policy="fixed_max", slots=int(slots))


def _plan_fixed_max_fused(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    return _plan_non_dp_topological(dag, edges, policy="fixed_max_fused", slots=int(slots))


def _plan_eager(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    return _plan_non_dp_topological(dag, edges, policy="eager", slots=int(slots))


def _plan_eager_fused(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    return _plan_non_dp_topological(dag, edges, policy="eager_fused", slots=int(slots))


def _plan_greedy(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    return _plan_non_dp_topological(dag, edges, policy="greedy", slots=int(slots))


def _plan_greedy_fused(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    return _plan_non_dp_topological(dag, edges, policy="greedy_fused", slots=int(slots))


def _plan_always(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    return _plan_non_dp_topological(dag, edges, policy="always", slots=int(slots))


def _plan_always_fused(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    return _plan_non_dp_topological(dag, edges, policy="always_fused", slots=int(slots))


def _plan_orion_dense(dag: NetworkDAG, edges: Sequence[EdgeInfo], *, slots: int) -> PolicyPlan:
    edges_by_source: dict[str, list[EdgeInfo]] = {}
    edges_by_target: dict[str, list[EdgeInfo]] = {}
    for edge in edges:
        edges_by_source.setdefault(str(edge.source), []).append(edge)
        edges_by_target.setdefault(str(edge.target), []).append(edge)
    topo = [str(node) for node in dag.topological_sort()]
    topo_index = {str(node): index for index, node in enumerate(topo)}
    last_consumer_index = {
        str(source): max(topo_index[str(edge.target)] for edge in source_edges)
        for source, source_edges in edges_by_source.items()
    }
    live: dict[str, LayoutState] = {}
    rows: list[dict[str, Any]] = []
    node_layouts: list[dict[str, Any]] = []

    for node in topo:
        incoming = tuple(edges_by_target.get(str(node), ()))
        outgoing = tuple(edges_by_source.get(str(node), ()))
        module = dag.nodes[node].get("module")
        incoming_rows: list[dict[str, Any]] = []
        for edge in incoming:
            source_layout = live.get(str(edge.source), _no_halo_layout_for_edge(edge, slots=int(slots)))
            target_layout = _no_halo_layout_for_edge(edge, slots=int(slots))
            incoming_rows.append(
                _edge_row(
                    edge,
                    target_layout,
                    relayout=False,
                    relayout_reason="",
                    layout_mode="orion_dense",
                    source_layout=source_layout,
                    physical_layout=PHYSICAL_COMPACT,
                )
            )
        rows.extend(incoming_rows)

        for source in list(live):
            if int(last_consumer_index.get(str(source), -1)) <= int(topo_index[str(node)]):
                live.pop(str(source), None)

        output_layout: LayoutState | None = None
        if incoming_rows and outgoing:
            if _is_join_module(module):
                output_layout = _no_halo_layout_for_edge(outgoing[0], slots=int(slots))
            else:
                semantic = _operator_semantic_output_layout(
                    module,
                    LayoutState(**dict(incoming_rows[0]["selected_layout"])),
                    outgoing[0],
                    slots=int(slots),
                )
                output_layout = _no_halo_layout_from_semantic(
                    semantic,
                    edge=outgoing[0],
                    slots=int(slots),
                )
        elif outgoing:
            output_layout = _no_halo_layout_for_edge(outgoing[0], slots=int(slots))

        if output_layout is not None and outgoing:
            live[str(node)] = output_layout
            node_layouts.append(
                _node_layout_row(
                    str(node),
                    output_layout,
                    outgoing[0].compact,
                    relayout=False,
                    reason="",
                    producer_materialized_halo=False,
                    producer_materialized_halo_reason="",
                    producer_fused_rotation_estimate=0,
                    physical_layout=PHYSICAL_COMPACT,
                    shape=outgoing[0].shape,
                    fhe_shape=outgoing[0].fhe_shape,
                )
            )

    rows.sort(key=lambda row: (topo_index.get(str(row["source"]), 10**9), topo_index.get(str(row["target"]), 10**9)))
    return _finalize_policy(
        policy="orion_dense",
        edge_rows=rows,
        relayout_layouts=[],
        node_layouts=node_layouts,
    )


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
                source_layout=edge.compact,
            )
        )
    return rows, relayout_layouts


@dataclass(frozen=True)
class _FrontierState:
    score: float
    layout_score: float
    live_layouts: tuple[tuple[str, LayoutState], ...]
    live_physical_layouts: tuple[tuple[str, str], ...]
    edge_rows: tuple[dict[str, Any], ...]
    relayout_layouts: tuple[LayoutState, ...]
    node_layouts: tuple[dict[str, Any], ...] = ()

    def live_dict(self) -> dict[str, LayoutState]:
        return {str(node): layout for node, layout in self.live_layouts}

    def physical_dict(self) -> dict[str, str]:
        return {str(node): str(physical) for node, physical in self.live_physical_layouts}


def _frontier_key(
    live: dict[str, LayoutState],
    physical: dict[str, str],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        sorted(
            (str(node), layout.key(), str(physical.get(str(node), PHYSICAL_COMPACT)))
            for node, layout in live.items()
        )
    )


def _frontier_live_items(live: dict[str, LayoutState]) -> tuple[tuple[str, LayoutState], ...]:
    return tuple(sorted(((str(node), layout) for node, layout in live.items()), key=lambda item: item[0]))


def _frontier_physical_items(physical: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(((str(node), str(value)) for node, value in physical.items()), key=lambda item: item[0]))


def _prune_dp_frontier(
    states: dict[tuple[tuple[str, tuple[int, ...], str], ...], _FrontierState],
) -> dict[tuple[tuple[str, tuple[int, ...], str], ...], _FrontierState]:
    if len(states) <= int(DP_FRONTIER_STATE_LIMIT):
        return states
    ranked = sorted(
        states.items(),
        key=lambda item: (
            float(item[1].score),
            float(item[1].layout_score),
            len(item[1].relayout_layouts),
        ),
    )
    return dict(ranked[: int(DP_FRONTIER_STATE_LIMIT)])


def _layout_halo_slot_cost(layout: LayoutState | None) -> float:
    if layout is None:
        return 0.0
    return float(max(0, int(layout.stored_slots) - int(layout.core_slots)))


def _row_layout_halo_slot_cost(row: dict[str, Any]) -> float:
    layout = dict(row.get("selected_layout", {}) or {})
    if not layout:
        return 0.0
    return float(max(0, int(layout.get("stored_slots", 0) or 0) - int(layout.get("core_slots", 0) or 0)))


def _dp_layout_secondary_cost(
    *,
    incoming_rows: Sequence[dict[str, Any]],
    output_layout: LayoutState | None,
    output_relayouts: Sequence[LayoutState],
) -> float:
    return (
        float(sum(_row_layout_halo_slot_cost(row) for row in incoming_rows))
        + _layout_halo_slot_cost(output_layout)
        + float(sum(_layout_halo_slot_cost(layout) for layout in output_relayouts))
    )


def _dedupe_layouts(layouts: Iterable[LayoutState]) -> tuple[LayoutState, ...]:
    deduped: dict[tuple[int, int, int, int], LayoutState] = {}
    for layout in layouts:
        deduped.setdefault(layout.key(), layout)
    return tuple(deduped.values())


def _fill_beta_to_tile_capacity(layout: LayoutState, *, shape: tuple[int, int, int, int], slots: int) -> LayoutState:
    if _compact_height_strip_fits_single_ct(shape=shape, gap=int(layout.gap), slots=int(slots)):
        return layout
    target_tiles = max(1, int(layout.tile_count))
    min_top_beta = int(layout.top_beta)
    min_bottom_beta = int(layout.bottom_beta)
    _n, channels, height, width = shape
    gap = max(1, int(layout.gap))
    phase = max(1, int(gap) * int(gap))
    channel_groups = _ceil_div(int(channels), int(phase))
    row_width_slots = max(1, int(channel_groups) * int(width) * int(gap))
    max_stored_h = max(1, int(target_tiles) * int(slots) // int(row_width_slots))
    max_halo_rows = max(0, int(max_stored_h) - int(height) * int(gap))
    max_alpha_beta = int(max_halo_rows) // int(gap)
    halo_budget = max(int(min_top_beta + min_bottom_beta), int(max_alpha_beta))
    top_beta = max(int(min_top_beta), int(halo_budget // 2))
    bottom_beta = int(halo_budget) - int(top_beta)
    if int(bottom_beta) < int(min_bottom_beta):
        bottom_beta = int(min_bottom_beta)
        top_beta = int(halo_budget) - int(bottom_beta)
    filled = _layout_for_shape(
        shape=shape,
        gap=int(layout.gap),
        top_beta=int(top_beta),
        bottom_beta=int(bottom_beta),
        stride=int(layout.stride),
        slots=int(slots),
    )
    while int(filled.tile_count) > int(target_tiles) and (
        int(top_beta) > int(min_top_beta) or int(bottom_beta) > int(min_bottom_beta)
    ):
        if int(top_beta) > int(bottom_beta) and int(top_beta) > int(min_top_beta):
            top_beta -= 1
        elif int(bottom_beta) > int(min_bottom_beta):
            bottom_beta -= 1
        else:
            top_beta -= 1
        filled = _layout_for_shape(
            shape=shape,
            gap=int(layout.gap),
            top_beta=int(top_beta),
            bottom_beta=int(bottom_beta),
            stride=int(layout.stride),
            slots=int(slots),
        )
    return filled


def _source_layout_candidates(edges: Sequence[EdgeInfo], *, global_alpha: int, global_beta: int, slots: int) -> tuple[LayoutState, ...]:
    if not edges:
        return ()
    edge = edges[0]
    local_need = _max_layout(edges, slots=int(slots))
    future_layouts: list[LayoutState] = []
    for item in edges:
        future_layouts.extend(tuple(item.future_layouts))
    # DP compares actual layouts, so do not add capacity-fill halo that exists
    # only as alignment padding and is not demanded by a consumer.
    del global_alpha, global_beta
    return _dedupe_layouts((edge.compact, local_need, *future_layouts))


def _operator_semantic_output_layout(
    module: Any | None,
    source_layout: LayoutState,
    edge: EdgeInfo,
    *,
    slots: int,
) -> LayoutState:
    if isinstance(module, ConvTranspose2d):
        scale = max(1, int(_pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))[0]))
        top_beta = int(source_layout.top_beta) * int(scale)
        bottom_beta = int(source_layout.bottom_beta) * int(scale)
    elif isinstance(module, AvgPool2d):
        stride_pair = _pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))
        kernel_pair = _pair_tuple(getattr(module, "kernel_size", (1, 1)), (1, 1))
        stride = max(1, int(stride_pair[0]))
        consume = max(0, int(kernel_pair[0]) - int(stride_pair[0]))
        top_beta = _side_after_downsample(source_layout.top_beta, consume=int(consume), stride=int(stride))
        bottom_beta = _side_after_downsample(source_layout.bottom_beta, consume=int(consume), stride=int(stride))
    elif isinstance(module, Conv2d):
        stride = max(1, int(_pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))[0]))
        consume = _conv_halo_consume(module)
        top_beta = _side_after_downsample(source_layout.top_beta, consume=int(consume), stride=int(stride))
        bottom_beta = _side_after_downsample(source_layout.bottom_beta, consume=int(consume), stride=int(stride))
    else:
        top_beta = int(source_layout.top_beta)
        bottom_beta = int(source_layout.bottom_beta)
    return _layout_for_shape(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        top_beta=int(top_beta),
        bottom_beta=int(bottom_beta),
        stride=max(1, int(source_layout.stride)),
        slots=int(slots),
    )


def _producer_fused_output_allowed(module: Any | None) -> bool:
    return isinstance(module, (AvgPool2d, Conv2d, ConvTranspose2d))


def _producer_fused_output_layout_candidates(
    module: Any | None,
    *,
    outgoing: Sequence[EdgeInfo],
    slots: int,
) -> tuple[LayoutState, ...]:
    if not outgoing or not _producer_fused_output_allowed(module):
        return ()
    edge = outgoing[0]
    layouts: list[LayoutState] = []
    local_need = _max_layout(outgoing, slots=int(slots))
    if int(local_need.top_beta) > 0 or int(local_need.bottom_beta) > 0:
        # Producer fusion may materialize consumer-demanded halo, but DP should
        # not inflate that halo merely to fill otherwise unused tile capacity.
        layouts.append(local_need)
    for edge in outgoing:
        layouts.extend(tuple(edge.future_layouts))
        layouts.extend(_future_tconv_input_relayout_candidates(edge, slots=int(slots)))
        if _is_join_op(edge.op_kind):
            layouts.append(
                _layout_for_shape(
                    shape=edge.shape,
                    gap=int(edge.compact.gap),
                    top_beta=max(2, int(edge.requirement.top_beta)),
                    bottom_beta=max(2, int(edge.requirement.bottom_beta)),
                    stride=max(2, int(edge.requirement.stride)),
                    slots=int(slots),
                )
            )
    return _dedupe_layouts(layouts)


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
    layouts: list[LayoutState] = []
    for row in incoming_rows:
        source_layout = LayoutState(**dict(row["selected_layout"]))
        layouts.append(
            _operator_semantic_output_layout(
                module,
                source_layout,
                edge,
                slots=int(slots),
            )
        )
    # ConvTranspose2d may naturally propagate the input halo through stride-k
    # placement, write compact output, or explicitly materialize additional
    # halo beyond the natural output layout.
    layouts.append(edge.compact)
    layouts.extend(_producer_fused_output_layout_candidates(module, outgoing=outgoing, slots=int(slots)))
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
        layouts.append(
            _operator_semantic_output_layout(
                module,
                source_layout,
                edge,
                slots=int(slots),
            )
        )
    if _native_operator_output_layout(module) and any(
        int(dict(row["selected_layout"]).get("top_beta", 0)) > 0
        or int(dict(row["selected_layout"]).get("bottom_beta", 0)) > 0
        for row in incoming_rows
    ):
        layouts.append(edge.compact)
    layouts.extend(_producer_fused_output_layout_candidates(module, outgoing=outgoing, slots=int(slots)))
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
    if _is_join_module(module):
        return _dedupe_layouts(
            _layout_for_join_output_shape(
                LayoutState(**dict(row["selected_layout"])),
                outgoing=outgoing,
                slots=int(slots),
            )
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


def _output_physical_layout(
    module: Any | None,
    *,
    incoming_rows: Sequence[dict[str, Any]],
    output_layout: LayoutState,
    producer_fused: dict[str, Any],
    slots: int,
) -> str:
    if _layout_preserving_output(module) and incoming_rows:
        physicals = {
            str(row.get("physical_layout", PHYSICAL_COMPACT) or PHYSICAL_COMPACT)
            for row in incoming_rows
        }
        if len(physicals) == 1:
            return next(iter(physicals))
    if int(output_layout.top_beta) == 0 and int(output_layout.bottom_beta) == 0:
        return PHYSICAL_COMPACT
    if bool(producer_fused.get("enabled", False)):
        return PHYSICAL_LOGICAL_HALO
    if isinstance(module, Conv2d) and not isinstance(module, ConvTranspose2d):
        if any(str(row.get("physical_layout", "")) == PHYSICAL_NATIVE_SOURCE_STRIPE for row in incoming_rows):
            if not _native_conv_output_target_fits(
                module,
                incoming_rows=incoming_rows,
                output_layout=output_layout,
                slots=int(slots),
            ):
                return PHYSICAL_LOGICAL_HALO
            return PHYSICAL_NATIVE_SOURCE_STRIPE
    return PHYSICAL_LOGICAL_HALO


def _native_conv_output_target_fits(
    module: Any,
    *,
    incoming_rows: Sequence[dict[str, Any]],
    output_layout: LayoutState,
    slots: int,
) -> bool:
    if not incoming_rows:
        return False
    try:
        from orion.experimental.cir.native_halo_conv2d import NativeHaloConv2DSpec, native_halo_conv2d_plan

        input_shape = tuple(int(value) for value in getattr(module, "input_shape", ()))
        output_shape = tuple(int(value) for value in getattr(module, "output_shape", ()))
        if len(input_shape) < 4 or len(output_shape) < 4:
            return False
        kernel = _pair_tuple(getattr(module, "kernel_size", (1, 1)), (1, 1))
        stride = _pair_tuple(getattr(module, "stride", (1, 1)), (1, 1))
        padding = _pair_tuple(getattr(module, "padding", (0, 0)), (0, 0))
        dilation = _pair_tuple(getattr(module, "dilation", (1, 1)), (1, 1))
        groups = max(1, int(getattr(module, "groups", 1) or 1))
        if int(groups) != 1 or int(kernel[0]) != int(kernel[1]) or int(stride[0]) != int(stride[1]):
            return False
        if int(padding[0]) != int(padding[1]) or int(dilation[0]) != int(dilation[1]):
            return False
        input_layout = LayoutState(**dict(incoming_rows[0]["selected_layout"]))
        label = "".join(ch if ch.isalnum() else "_" for ch in str(getattr(module, "name", "") or "conv")).strip("_") or "conv"
        spec = NativeHaloConv2DSpec(
            family_label=(
                f"layout_policy_native_target_fit_{label}_{int(input_shape[1])}x{int(input_shape[2])}x{int(input_shape[3])}"
                f"_to_{int(output_shape[1])}x{int(output_shape[2])}x{int(output_shape[3])}"
                f"_k{int(kernel[0])}s{int(stride[0])}_gap{int(input_layout.gap)}to{int(output_layout.gap)}"
            ),
            c_in=int(input_shape[1]),
            h_in=int(input_shape[2]),
            w_in=int(input_shape[3]),
            c_out=int(output_shape[1]),
            h_out=int(output_shape[2]),
            w_out=int(output_shape[3]),
            gap_in=int(input_layout.gap),
            gap_out=int(output_layout.gap),
            kernel=int(kernel[0]),
            stride=int(stride[0]),
            pad=int(padding[0]),
            dilation=int(dilation[0]),
            groups=int(groups),
            slot_count=int(slots),
            input_top_beta=int(input_layout.top_beta),
            input_bottom_beta=int(input_layout.bottom_beta),
            output_top_beta=int(output_layout.top_beta),
            output_bottom_beta=int(output_layout.bottom_beta),
            input_physical_top_beta=_layout_physical_top_beta(input_layout),
            input_physical_bottom_beta=_layout_physical_bottom_beta(input_layout),
            output_physical_top_beta=_layout_physical_top_beta(output_layout),
            output_physical_bottom_beta=_layout_physical_bottom_beta(output_layout),
        )
        native_halo_conv2d_plan(spec, require_native_target_fit=True)
        return True
    except ValueError as exc:
        message = str(exc)
        if "native halo" in message and "does not fit" in message:
            return False
        return False


def _native_operator_output_layout(module: Any | None) -> bool:
    return isinstance(module, (AvgPool2d, Conv2d, ConvTranspose2d))


def _layout_preserving_output(module: Any | None) -> bool:
    return isinstance(module, (Activation, Chebyshev, Quad, ReLU))


def _direct_output_layout_without_relayout(module: Any | None) -> bool:
    return _native_operator_output_layout(module) or _layout_preserving_output(module)


def _incoming_relayout_candidates(
    edge: EdgeInfo,
    *,
    global_alpha: int,
    global_beta: int,
    slots: int,
) -> tuple[LayoutState, ...]:
    # Refill exactly the consumer requirement; larger capacity-fill layouts are
    # explored by non-DP policies, not by the DP strip/refill comparison.
    del global_alpha, global_beta, slots
    return (edge.requirement,)


def _future_tconv_input_relayout_candidates(edge: EdgeInfo, *, slots: int) -> tuple[LayoutState, ...]:
    if str(edge.op_kind) != "conv_transpose2d":
        return ()
    if _compact_height_strip_fits_single_ct(shape=edge.shape, gap=int(edge.compact.gap), slots=int(slots)):
        return ()
    halo = _layout_for_shape(
        shape=edge.shape,
        gap=int(edge.compact.gap),
        top_beta=max(1, int(edge.requirement.top_beta)),
        bottom_beta=max(1, int(edge.requirement.bottom_beta)),
        stride=max(1, int(edge.requirement.stride)),
        slots=int(slots),
    )
    return (halo,)


def _compact_align_shared_allowed(
    edge: EdgeInfo,
    source_layout: LayoutState,
    source_physical: str = PHYSICAL_COMPACT,
) -> bool:
    if str(edge.op_kind) != "conv2d":
        return False
    if str(edge.source) == "x":
        return False
    if str(source_physical) == PHYSICAL_NATIVE_SOURCE_STRIPE:
        return False
    return int(source_layout.gap) == int(edge.requirement.gap)


def _conv_native_stripe_candidate_allowed(edge: EdgeInfo, source_layout: LayoutState) -> bool:
    if str(edge.op_kind) != "conv2d":
        return False
    if int(edge.requirement.top_beta) == 0 and int(edge.requirement.bottom_beta) == 0:
        return False
    if not source_layout.covers(edge.requirement):
        return False
    return int(source_layout.gap) == int(edge.requirement.gap)


def _compact_halo_shared_allowed(edge: EdgeInfo, source_layout: LayoutState, source_physical: str) -> bool:
    if str(edge.op_kind) != "conv2d":
        return False
    if str(edge.source) == "x":
        return False
    if str(source_physical) == PHYSICAL_NATIVE_SOURCE_STRIPE:
        return False
    if not source_layout.covers(edge.requirement):
        return False
    return int(source_layout.gap) == int(edge.requirement.gap)


def _consumer_fused_extra_rotation(edge: EdgeInfo, fused_layout: LayoutState) -> int:
    """Extra LT rotations paid by fusing boundary exchange into the consumer.

    The row's ``lt_bsgs_rotation_estimate`` stores the actual fused LT cost.
    This helper records only the increment relative to the native-halo
    requirement so diagnostics can explain why a depth-free fused candidate is
    not rotation-free.
    """

    fused = int(_lt_rotations(edge, fused_layout))
    native = int(_lt_rotations(edge, edge.requirement))
    return int(max(0, fused - native))


def _consumer_fused_planner_rotation_cost(edge: EdgeInfo, layout: LayoutState, *, layout_mode: str) -> int:
    del layout_mode
    return int(_lt_rotations(edge, layout))


def _compact_align_shared_penalty(edge: EdgeInfo, source_layout: LayoutState) -> int:
    source_cost = int(_lt_rotations(edge, source_layout))
    requirement_cost = int(_lt_rotations(edge, edge.requirement))
    halo_sides = int(_relayout_halo_side_count(edge.requirement))
    return int(max(0, source_cost - requirement_cost) + halo_sides)


def _physical_for_candidate(layout: LayoutState, *, layout_mode: str, fallback: str | None = None) -> str:
    if fallback is not None:
        return str(fallback)
    if str(layout_mode) == "native_halo_stripe":
        return PHYSICAL_NATIVE_SOURCE_STRIPE
    if int(layout.top_beta) > 0 or int(layout.bottom_beta) > 0:
        return PHYSICAL_LOGICAL_HALO
    return PHYSICAL_COMPACT


def enumerate_execution_candidates(
    edge: EdgeInfo,
    source_layout: LayoutState,
    *,
    source_physical: str = PHYSICAL_COMPACT,
    global_alpha: int,
    global_beta: int,
    slots: int,
) -> tuple[ExecutionCandidate, ...]:
    """Enumerate input-layout execution choices for one non-Add edge.

    This is intentionally separated from cost estimation: the enumerator only
    describes which logical/physical layout transition is being considered.
    ``estimate_candidate_cost`` owns the rotation, relayout, and fusion cost.
    """

    candidates: list[ExecutionCandidate] = []
    if source_layout.covers(edge.requirement):
        selected_layout = _layout_with_stride(
            source_layout,
            max(int(source_layout.stride), int(edge.requirement.stride)),
        )
        if _conv_native_stripe_candidate_allowed(edge, source_layout):
            native_relayout = bool(
                str(edge.source) != "x" and str(source_physical) != PHYSICAL_NATIVE_SOURCE_STRIPE
            )
            candidates.append(
                ExecutionCandidate(
                    edge=edge,
                    source_layout=source_layout,
                    target_layout=selected_layout,
                    source_physical=str(source_physical),
                    target_physical=PHYSICAL_NATIVE_SOURCE_STRIPE,
                    layout_mode="native_halo_stripe",
                    relayout=bool(native_relayout),
                    relayout_reason="dp_native_source_stripe_relayout" if bool(native_relayout) else "",
                )
            )
            if _compact_halo_shared_allowed(edge, source_layout, str(source_physical)):
                candidates.append(
                    ExecutionCandidate(
                        edge=edge,
                        source_layout=source_layout,
                        target_layout=selected_layout,
                        source_physical=str(source_physical),
                        target_physical=PHYSICAL_LOGICAL_HALO,
                        layout_mode="compact_halo_shared",
                        consumer_fused_relayout=True,
                        consumer_fused_rotation_estimate=0,
                    )
                )
        else:
            candidates.append(
                ExecutionCandidate(
                    edge=edge,
                    source_layout=source_layout,
                    target_layout=selected_layout,
                    source_physical=str(source_physical),
                    target_physical=str(source_physical),
                )
            )
    if str(edge.source) == "x":
        return tuple(candidates)
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
        if str(edge.op_kind) == "conv2d" and _compact_align_shared_allowed(
            edge,
            source_layout,
            str(source_physical),
        ):
            continue
        candidates.append(
            ExecutionCandidate(
                edge=edge,
                source_layout=source_layout,
                target_layout=layout,
                source_physical=str(source_physical),
                target_physical=_physical_for_candidate(layout, layout_mode="halo_local"),
                relayout=True,
                relayout_reason="dp_state_consumer_relayout",
            )
        )
    if _compact_align_shared_allowed(edge, source_layout, str(source_physical)):
        selected_layout = _layout_with_stride(
            source_layout,
            max(int(source_layout.stride), int(edge.requirement.stride)),
        )
        candidates.append(
            ExecutionCandidate(
                edge=edge,
                source_layout=source_layout,
                target_layout=selected_layout,
                source_physical=str(source_physical),
                target_physical=(
                    PHYSICAL_LOGICAL_HALO
                    if int(selected_layout.top_beta) > 0 or int(selected_layout.bottom_beta) > 0
                    else PHYSICAL_COMPACT
                ),
                layout_mode="compact_align_shared",
                consumer_fused_relayout=True,
                consumer_fused_rotation_estimate=0,
            )
        )
    return tuple(candidates)


def estimate_candidate_cost(
    candidate: ExecutionCandidate,
    *,
    estimator: str | None = None,
) -> tuple[list[dict[str, Any]], list[LayoutState]]:
    """Turn an execution candidate into planner rows and relayout accounting."""

    target_physical = _physical_for_candidate(
        candidate.target_layout,
        layout_mode=str(candidate.layout_mode),
        fallback=candidate.target_physical,
    )
    planner_rotation_cost: int | None = None
    if bool(candidate.consumer_fused_relayout):
        planner_rotation_cost = _consumer_fused_planner_rotation_cost(
            candidate.edge,
            candidate.target_layout,
            layout_mode=str(candidate.layout_mode),
        )
    row = _edge_row(
        candidate.edge,
        candidate.target_layout,
        relayout=bool(candidate.relayout),
        relayout_reason=str(candidate.relayout_reason) if bool(candidate.relayout) else "",
        layout_mode=str(candidate.layout_mode),
        source_layout=candidate.source_layout,
        source_physical_layout=str(candidate.source_physical),
        physical_layout=str(target_physical),
        consumer_fused_relayout=bool(candidate.consumer_fused_relayout),
        consumer_fused_rotation_estimate=int(candidate.consumer_fused_rotation_estimate),
        planner_rotation_cost=planner_rotation_cost,
        estimator=estimator,
    )
    relayouts = [candidate.target_layout] if bool(candidate.relayout) else []
    return [row], relayouts


def _option_cost_for_refinement(rows: Sequence[dict[str, Any]], relayouts: Sequence[LayoutState]) -> float:
    return float(_policy_linear_cost(rows, relayouts))


def _candidate_needs_template_refinement(cost: float, best_cost: float) -> bool:
    window = max(
        float(LAYOUT_ESTIMATOR_AUTO_ABSOLUTE_WINDOW),
        abs(float(best_cost)) * float(LAYOUT_ESTIMATOR_AUTO_RELATIVE_WINDOW),
    )
    return float(cost) <= float(best_cost) + float(window)


def _estimate_candidate_options(
    candidates: Sequence[ExecutionCandidate],
    *,
    estimator: str | None = None,
) -> tuple[tuple[list[dict[str, Any]], list[LayoutState]], ...]:
    mode = _normalize_layout_estimator(estimator)
    counted = [
        (
            candidate,
            *estimate_candidate_cost(candidate, estimator=LAYOUT_ESTIMATOR_COUNT_ONLY),
        )
        for candidate in candidates
    ]
    if mode == LAYOUT_ESTIMATOR_COUNT_ONLY or not counted:
        return tuple((rows, relayouts) for _candidate, rows, relayouts in counted)

    if mode == LAYOUT_ESTIMATOR_TEMPLATE:
        return tuple(
            estimate_candidate_cost(candidate, estimator=LAYOUT_ESTIMATOR_TEMPLATE)
            for candidate, _rows, _relayouts in counted
        )

    costs = [_option_cost_for_refinement(rows, relayouts) for _candidate, rows, relayouts in counted]
    best = min(costs) if costs else 0.0
    refined: list[tuple[list[dict[str, Any]], list[LayoutState]]] = []
    for (candidate, rows, relayouts), cost in zip(counted, costs, strict=True):
        if _candidate_needs_template_refinement(float(cost), float(best)):
            refined.append(estimate_candidate_cost(candidate, estimator=LAYOUT_ESTIMATOR_TEMPLATE))
        else:
            refined.append((rows, relayouts))
    return tuple(refined)


def _incoming_non_add_options(
    edge: EdgeInfo,
    source_layout: LayoutState,
    *,
    source_physical: str = PHYSICAL_COMPACT,
    global_alpha: int,
    global_beta: int,
    slots: int,
    estimator: str | None = None,
) -> tuple[tuple[list[dict[str, Any]], list[LayoutState]], ...]:
    candidates = enumerate_execution_candidates(
        edge,
        source_layout,
        source_physical=str(source_physical),
        global_alpha=int(global_alpha),
        global_beta=int(global_beta),
        slots=int(slots),
    )
    return _estimate_candidate_options(candidates, estimator=estimator)


def _incoming_add_options(
    incoming: Sequence[EdgeInfo],
    live: dict[str, LayoutState],
    live_physical: dict[str, str],
) -> tuple[tuple[list[dict[str, Any]], list[LayoutState]], ...]:
    source_layouts = [live[edge.source] for edge in incoming]
    candidates: list[LayoutState] = []
    if incoming:
        candidates.append(incoming[0].compact)
    if source_layouts:
        edge = incoming[0]
        candidates.append(
            _layout_for_shape(
                shape=edge.shape,
                gap=int(edge.compact.gap),
                top_beta=max(int(layout.top_beta) for layout in source_layouts),
                bottom_beta=max(int(layout.bottom_beta) for layout in source_layouts),
                stride=max(int(layout.stride) for layout in source_layouts),
                slots=int(edge.slots),
            )
        )
    options: list[tuple[list[dict[str, Any]], list[LayoutState]]] = []
    for target_layout in _dedupe_layouts(candidates):
        rows: list[dict[str, Any]] = []
        relayouts: list[LayoutState] = []
        for edge in incoming:
            source_layout = live[edge.source]
            source_physical = live_physical.get(str(edge.source), _compact_packing_for_layout(source_layout))
            physical_source_layout = _join_source_layout_for_physical(
                edge,
                source_layout,
                str(source_physical),
            )
            relayout = _join_input_requires_relayout(
                physical_source_layout,
                str(source_physical),
                target_layout,
            )
            if bool(relayout):
                relayouts.append(target_layout)
            rows.append(
                _edge_row(
                    edge,
                    target_layout,
                    relayout=bool(relayout),
                    relayout_reason="dp_add_input_alignment" if bool(relayout) else "",
                    lt_rotations=_lt_rotations(edge, target_layout),
                    source_layout=physical_source_layout,
                    source_physical_layout=str(source_physical),
                    physical_layout=_compact_packing_for_layout(target_layout),
                )
            )
        options.append((rows, relayouts))
    return tuple(options)


def _plan_dp(
    dag: NetworkDAG,
    edges: Sequence[EdgeInfo],
    *,
    slots: int,
    estimator: str | None = None,
) -> PolicyPlan:
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
    global_alpha = max(int(edge.requirement.top_beta) for edge in edges)
    global_beta = max(int(edge.requirement.bottom_beta) for edge in edges)

    states: dict[tuple[tuple[str, tuple[int, int, int, int], str], ...], _FrontierState] = {
        (): _FrontierState(
            score=0.0,
            layout_score=0.0,
            live_layouts=(),
            live_physical_layouts=(),
            edge_rows=(),
            relayout_layouts=(),
        )
    }
    for node in topo:
        incoming = tuple(edges_by_target.get(str(node), ()))
        outgoing = tuple(edges_by_source.get(str(node), ()))
        node_index = int(topo_index[str(node)])
        next_states: dict[tuple[tuple[str, tuple[int, int, int, int], str], ...], _FrontierState] = {}
        for state in states.values():
            live = state.live_dict()
            live_physical = state.physical_dict()
            if any(edge.source not in live for edge in incoming):
                continue
            if incoming and _is_join_module(dag.nodes[node].get("module")):
                incoming_options = _incoming_add_options(incoming, live, live_physical)
            else:
                incoming_options_work: tuple[tuple[list[dict[str, Any]], list[LayoutState]], ...] = (([], []),)
                for edge in incoming:
                    edge_options = _incoming_non_add_options(
                        edge,
                        live[edge.source],
                        source_physical=live_physical.get(str(edge.source), PHYSICAL_COMPACT),
                        global_alpha=int(global_alpha),
                        global_beta=int(global_beta),
                        slots=int(slots),
                        estimator=estimator,
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
                next_physical = dict(live_physical)
                for source in list(next_live):
                    if int(last_consumer_index.get(str(source), -1)) <= int(node_index):
                        next_live.pop(str(source), None)
                        next_physical.pop(str(source), None)
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
                        producer_fused = _producer_fused_materialization_estimate(
                            module,
                            incoming=incoming,
                            incoming_rows=incoming_rows,
                            outgoing=outgoing,
                            output_layout=output_layout,
                            slots=int(slots),
                        )
                        output_relayout = (
                            compact is not None
                            and not _same_physical_layout(output_layout, compact)
                            and str(node) != "x"
                            and not _is_join_module(module)
                            and not _direct_output_layout_without_relayout(module)
                        )
                        if bool(output_relayout):
                            output_relayouts.append(output_layout)
                        output_physical = _output_physical_layout(
                            module,
                            incoming_rows=incoming_rows,
                            output_layout=output_layout,
                            producer_fused=producer_fused,
                            slots=int(slots),
                        )
                        candidate_storage = dict(next_physical)
                        candidate_storage[str(node)] = str(output_physical)
                        logical_halo_materialized = bool(
                            str(output_physical) == PHYSICAL_LOGICAL_HALO
                            and _layout_has_physical_halo(output_layout)
                            and _producer_fused_output_allowed(module)
                        )
                        materialized_halo = bool(producer_fused["enabled"]) or bool(logical_halo_materialized)
                        materialized_reason = (
                            "dp_producer_materialized_halo"
                            if bool(producer_fused["enabled"])
                            else ("dp_logical_halo_materialized_output" if bool(logical_halo_materialized) else "")
                        )
                        output_layout_rows.append(
                            _node_layout_row(
                                str(node),
                                output_layout,
                                compact,
                                relayout=bool(output_relayout),
                                reason="dp_producer_materialized_halo" if bool(output_relayout) else "",
                                producer_materialized_halo=bool(materialized_halo),
                                producer_materialized_halo_reason=str(materialized_reason),
                                producer_fused_rotation_estimate=int(producer_fused["rotation_count"]),
                                physical_layout=str(output_physical),
                                shape=outgoing[0].shape if outgoing else None,
                                fhe_shape=outgoing[0].fhe_shape if outgoing else None,
                            )
                        )
                    else:
                        producer_fused = {"enabled": False, "rotation_count": 0}
                        candidate_storage = dict(next_physical)
                    candidate_rows = tuple(list(state.edge_rows) + list(incoming_rows))
                    candidate_relayouts = tuple(
                        list(state.relayout_layouts)
                        + list(incoming_relayouts)
                        + list(output_relayouts)
                    )
                    candidate_score = (
                        float(state.score)
                        + _policy_linear_cost(incoming_rows, incoming_relayouts)
                        + float(int(producer_fused["rotation_count"]))
                        + float(sum(_relayout_linear_cost(layout) for layout in output_relayouts))
                    )
                    candidate_layout_score = float(state.layout_score) + _dp_layout_secondary_cost(
                        incoming_rows=incoming_rows,
                        output_layout=output_layout,
                        output_relayouts=output_relayouts,
                    )
                    key = _frontier_key(candidate_live, candidate_storage)
                    existing = next_states.get(key)
                    if existing is None or (
                        candidate_score,
                        candidate_layout_score,
                        len(candidate_relayouts),
                    ) < (
                        float(existing.score),
                        float(existing.layout_score),
                        len(existing.relayout_layouts),
                    ):
                        next_states[key] = _FrontierState(
                            score=float(candidate_score),
                            layout_score=float(candidate_layout_score),
                            live_layouts=_frontier_live_items(candidate_live),
                            live_physical_layouts=_frontier_physical_items(candidate_storage),
                            edge_rows=candidate_rows,
                            relayout_layouts=candidate_relayouts,
                            node_layouts=tuple(output_layout_rows),
                        )
                        if len(next_states) > int(DP_FRONTIER_STATE_LIMIT) * 4:
                            next_states = _prune_dp_frontier(next_states)
        states = _prune_dp_frontier(next_states)
    if not states:
        raise RuntimeError("layout policy DP found no legal state")
    best_state = min(
        states.values(),
        key=lambda state: (
            float(state.score),
            float(state.layout_score),
            len(state.relayout_layouts),
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


def plan_policy(
    dag: NetworkDAG,
    edges: Sequence[EdgeInfo],
    policy: str,
    *,
    slots: int = DEFAULT_SLOTS,
    estimator: str | None = None,
) -> PolicyPlan:
    normalized = normalize_policy(policy)
    if normalized == "fixed_max":
        return _plan_fixed_max(dag, edges, slots=int(slots))
    if normalized == "fixed_max_fused":
        return _plan_fixed_max_fused(dag, edges, slots=int(slots))
    if normalized == "eager":
        return _plan_eager(dag, edges, slots=int(slots))
    if normalized == "eager_fused":
        return _plan_eager_fused(dag, edges, slots=int(slots))
    if normalized == "greedy":
        return _plan_greedy(dag, edges, slots=int(slots))
    if normalized == "greedy_fused":
        return _plan_greedy_fused(dag, edges, slots=int(slots))
    if normalized == "always":
        return _plan_always(dag, edges, slots=int(slots))
    if normalized == "always_fused":
        return _plan_always_fused(dag, edges, slots=int(slots))
    if normalized == "orion_dense":
        return _plan_orion_dense(dag, edges, slots=int(slots))
    if normalized == "dp":
        return _plan_dp(dag, edges, slots=int(slots), estimator=estimator)
    raise AssertionError(f"unreachable policy {policy!r}")


def build_layout_policy_compile_plan(
    dag: NetworkDAG,
    *,
    policy: str = "dp",
    slots: int = DEFAULT_SLOTS,
    estimator: str | None = None,
) -> dict[str, Any]:
    edges = build_edge_infos(dag, slots=int(slots))
    plan = plan_policy(dag, edges, policy, slots=int(slots), estimator=estimator)
    edge_layouts = [dict(row) for row in plan.edge_layouts]
    node_layouts = [dict(row) for row in plan.node_layouts]
    relayout_edges = [
        {
            "edge": str(row["edge"]),
            "source": str(row["source"]),
            "target": str(row["target"]),
            "reason": str(row.get("relayout_reason", "")),
            "source_layout": dict(row.get("source_layout", {}) or {}),
            "selected_layout": dict(row["selected_layout"]),
            "target_layout": dict(row.get("target_layout", row["selected_layout"]) or {}),
            "rotation_estimate": int(row.get("relayout_rotation_estimate", 0) or 0),
            "mask_mult_estimate": int(row.get("relayout_mask_mult_estimate", 0) or 0),
            "sparse_lt_estimate": int(row.get("relayout_sparse_lt_estimate", 0) or 0),
            "depth_estimate": int(row.get("relayout_depth_estimate", 0) or 0),
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
        sum(int(row.get("depth_estimate", 0) or 0) for row in relayout_edges)
        + sum(int(row.get("depth_estimate", 0)) for row in output_relayout_nodes)
    )
    return {
        "status": "ok",
        "policy": str(plan.policy),
        "policy_label": str(plan.policy_label),
        "metric_source": str(plan.metric_source),
        "slots": int(slots),
        "layout_estimator": _normalize_layout_estimator(estimator),
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
    policies: Sequence[str] = ("fixed_max", "always_fused", "dp"),
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
                int(row["selected_layout"]["top_beta"]),
                int(row["selected_layout"]["bottom_beta"]),
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
    suffixes = {
        "fixed_max": "fixedmax",
        "fixed_max_fused": "fixedmax_fused",
        "orion_dense": "oriondense",
    }
    suffix = suffixes.get(str(normalized), str(normalized))
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
