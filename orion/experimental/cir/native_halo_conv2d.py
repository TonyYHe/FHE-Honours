from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Literal
import time

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.nn.unified_transform import UnifiedTransformGroup

from .r34_orion_same_shape import (
    RING_SLOT_COUNT,
    _add_plaintext_for_add,
    _align_ciphertexts_for_add,
    _bsgs_rotation_sets,
    _coalesce_native_rows,
    _encode_plaintext_for_add,
    _idx_chw_gap,
    _native_best_common_bsgs,
    _rescale_cipher_tensor,
)


def _ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def _pair(value: Any, *, default: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    if value is None:
        return tuple(default)
    if isinstance(value, int):
        return (int(value), int(value))
    values = tuple(int(item) for item in value)
    if len(values) == 1:
        return (int(values[0]), int(values[0]))
    return (int(values[0]), int(values[1]))


def _safe_shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value)
    except Exception:
        return ()


def _phase_count(gap: int) -> int:
    return max(1, int(gap) * int(gap))


def _packed_active_slots(channel_count: int, height: int, width: int, gap: int) -> int:
    groups = _ceil_div(int(channel_count), _phase_count(int(gap)))
    return int(groups) * int(height) * int(gap) * int(width) * int(gap)


def _compact_ct_count(channel_count: int, height: int, width: int, gap: int, slots: int) -> int:
    active_slots = _packed_active_slots(int(channel_count), int(height), int(width), int(gap))
    return max(1, _ceil_div(int(active_slots), int(slots)))


def _spec_has_output_halo(spec: NativeHaloConv2DSpec) -> bool:
    return bool(int(spec.output_alpha) > 0 or int(spec.output_beta) > 0)


def _slot_indices(channel_count: int, height: int, width: int, gap: int) -> torch.Tensor:
    g = max(1, int(gap))
    phases = int(g * g)
    packed_w = int(width) * int(g)
    group_block = int(height) * int(g) * int(packed_w)
    hs = torch.arange(int(height), dtype=torch.int64)[:, None]
    ws = torch.arange(int(width), dtype=torch.int64)[None, :]
    out = torch.empty((int(channel_count), int(height), int(width)), dtype=torch.int64)
    for channel in range(int(channel_count)):
        group = int(channel) // int(phases)
        phase = int(channel) % int(phases)
        phase_h = int(phase) // int(g)
        phase_w = int(phase) % int(g)
        out[int(channel)] = (
            int(group) * int(group_block)
            + (hs * int(g) + int(phase_h)) * int(packed_w)
            + ws * int(g)
            + int(phase_w)
        )
    return out


def _heuristic_channel_tile(channel_count: int, gap: int) -> int:
    """Deterministic native-stripe channel tile.

    Use one natural phase group for gapped layouts.  For gap-1 layouts, use a
    two-channel tile when possible; odd final tiles intentionally leave the
    second lane empty so every tile keeps the same geometry.
    """

    phase = _phase_count(int(gap))
    if int(phase) == 1:
        return 1 if int(channel_count) <= 1 else 2
    return int(phase)


def _source_h_range_for_target(
    *,
    target_h_start: int,
    target_h_end: int,
    input_h_min: int,
    input_h_max: int,
    kernel: int,
    stride: int,
    pad: int,
    dilation: int,
) -> tuple[int, int]:
    source_start = int(target_h_start) * int(stride) - int(pad)
    source_end = (int(target_h_end) - 1) * int(stride) - int(pad) + (int(kernel) - 1) * int(dilation)
    start = max(int(input_h_min), int(source_start))
    end = min(int(input_h_max), int(source_end) + 1)
    if int(end) < int(start):
        end = int(start)
    return int(start), int(end)


def _extend_h_range_to_length(
    *,
    required_start: int,
    required_end: int,
    desired_len: int,
    lower: int,
    upper: int,
) -> tuple[int, int]:
    start = int(required_start)
    end = int(required_end)
    desired = min(int(upper - lower), max(int(end - start), int(desired_len)))
    extra = int(desired - (end - start))
    if int(extra) <= 0:
        return int(start), int(end)
    left = min(int(start - lower), int(extra // 2))
    start -= int(left)
    extra -= int(left)
    right = min(int(upper - end), int(extra))
    end += int(right)
    extra -= int(right)
    if int(extra) > 0:
        left = min(int(start - lower), int(extra))
        start -= int(left)
    return int(start), int(end)


@dataclass(frozen=True)
class NativeHaloConv2DSpec:
    family_label: str
    c_in: int
    h_in: int
    w_in: int
    c_out: int
    h_out: int
    w_out: int
    gap_in: int
    gap_out: int
    kernel: int
    stride: int
    pad: int
    dilation: int = 1
    groups: int = 1
    slot_count: int = RING_SLOT_COUNT
    input_alpha: int = 0
    input_beta: int = 0
    output_alpha: int = 0
    output_beta: int = 0

    @property
    def input_h_min(self) -> int:
        return -max(0, int(self.input_alpha))

    @property
    def input_h_max(self) -> int:
        return int(self.h_in) + max(0, int(self.input_beta))

    @property
    def output_h_min(self) -> int:
        return -max(0, int(self.output_alpha))

    @property
    def output_h_max(self) -> int:
        return int(self.h_out) + max(0, int(self.output_beta))

    @property
    def weight_shape(self) -> tuple[int, int, int, int]:
        return (
            int(self.c_out),
            int(self.c_in) // int(self.groups),
            int(self.kernel),
            int(self.kernel),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_label": str(self.family_label),
            "c_in": int(self.c_in),
            "h_in": int(self.h_in),
            "w_in": int(self.w_in),
            "c_out": int(self.c_out),
            "h_out": int(self.h_out),
            "w_out": int(self.w_out),
            "gap_in": int(self.gap_in),
            "gap_out": int(self.gap_out),
            "kernel": int(self.kernel),
            "stride": int(self.stride),
            "pad": int(self.pad),
            "dilation": int(self.dilation),
            "groups": int(self.groups),
            "slot_count": int(self.slot_count),
            "input_alpha": int(self.input_alpha),
            "input_beta": int(self.input_beta),
            "output_alpha": int(self.output_alpha),
            "output_beta": int(self.output_beta),
        }


@dataclass(frozen=True)
class NativeHaloStripe:
    index: int
    target_h_start: int
    target_h_end: int
    source_h_start: int
    source_h_end: int

    @property
    def source_h(self) -> int:
        return int(self.source_h_end) - int(self.source_h_start)

    @property
    def target_h(self) -> int:
        return int(self.target_h_end) - int(self.target_h_start)

    def to_dict(self) -> dict[str, int]:
        return {
            "index": int(self.index),
            "target_h_start": int(self.target_h_start),
            "target_h_end": int(self.target_h_end),
            "source_h_start": int(self.source_h_start),
            "source_h_end": int(self.source_h_end),
            "stored_source_rows": int(self.source_h),
            "target_rows": int(self.target_h),
            "halo_redundant_rows": int(self.source_h - self.target_h),
        }


@dataclass(frozen=True)
class NativeHaloConv2DPlan:
    spec: NativeHaloConv2DSpec
    source_channel_tile: int
    target_channel_tile: int
    stripes: tuple[NativeHaloStripe, ...]
    program_diagonal_counts: tuple[int, ...]
    program_rotation_counts: tuple[int, ...]
    group_n1s: tuple[int, ...]
    group_shared_rotations: tuple[int, ...]
    group_baby_rotations: tuple[int, ...]
    group_giant_rotations: tuple[int, ...]

    @property
    def source_channel_group_count(self) -> int:
        return _ceil_div(int(self.spec.c_in), int(self.source_channel_tile))

    @property
    def target_channel_group_count(self) -> int:
        return _ceil_div(int(self.spec.c_out), int(self.target_channel_tile))

    @property
    def input_ct_count(self) -> int:
        return int(len(self.stripes) * int(self.source_channel_group_count))

    @property
    def output_ct_count(self) -> int:
        if not _spec_has_output_halo(self.spec):
            return _compact_ct_count(
                int(self.spec.c_out),
                int(self.spec.h_out),
                int(self.spec.w_out),
                int(self.spec.gap_out),
                int(self.spec.slot_count),
            )
        return int(len(self.stripes) * int(self.target_channel_group_count))

    @property
    def submatrix_program_count(self) -> int:
        return int(len(self.program_diagonal_counts))

    @property
    def sharing_group_count(self) -> int:
        return int(len(self.group_shared_rotations))

    @property
    def c_only_rotations(self) -> int:
        return int(sum(int(value) for value in self.program_rotation_counts))

    @property
    def cb_shared_rotations(self) -> int:
        return int(sum(int(value) for value in self.group_shared_rotations))

    @property
    def shared_baby_rotations(self) -> int:
        return int(sum(int(value) for value in self.group_baby_rotations))

    @property
    def shared_giant_rotations(self) -> int:
        return int(sum(int(value) for value in self.group_giant_rotations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_layout": "native_halo_stripe_no_ri",
            "conv_dependency": "native_halo_source_tiles",
            "output_storage_layout": "native_halo_stripe" if _spec_has_output_halo(self.spec) else "tight_compact",
            "spec": self.spec.to_dict(),
            "source_channel_tile": int(self.source_channel_tile),
            "target_channel_tile": int(self.target_channel_tile),
            "source_channel_group_count": int(self.source_channel_group_count),
            "target_channel_group_count": int(self.target_channel_group_count),
            "stripe_count": int(len(self.stripes)),
            "input_ct_count": int(self.input_ct_count),
            "output_ct_count": int(self.output_ct_count),
            "submatrix_program_count": int(self.submatrix_program_count),
            "sharing_group_count": int(self.sharing_group_count),
            "c_only_rotations": int(self.c_only_rotations),
            "cb_shared_rotations": int(self.cb_shared_rotations),
            "shared_baby_rotations": int(self.shared_baby_rotations),
            "shared_giant_rotations": int(self.shared_giant_rotations),
            "program_diagonal_counts": [int(value) for value in self.program_diagonal_counts],
            "program_rotation_counts": [int(value) for value in self.program_rotation_counts],
            "group_n1s": [int(value) for value in self.group_n1s],
            "group_shared_rotations": [int(value) for value in self.group_shared_rotations],
            "group_baby_rotations": [int(value) for value in self.group_baby_rotations],
            "group_giant_rotations": [int(value) for value in self.group_giant_rotations],
            "stripes": [stripe.to_dict() for stripe in self.stripes],
            "notes": [
                "Height stripes come from the halo-stripe oracle source range required by each target stripe.",
                "source_local_h = in_h - source_h_start; target_local_h = out_h - target_h_start.",
                "No real/imag lane packing is used; B sharing is per native source tile.",
            ],
        }


def native_halo_conv2d_spec_from_module(module: Any, *, output_node_id: str) -> NativeHaloConv2DSpec | None:
    weight = getattr(module, "on_weight", None)
    weight_shape = _safe_shape(getattr(weight, "shape", ()))
    input_shape = _safe_shape(getattr(module, "input_shape", ()))
    output_shape = _safe_shape(getattr(module, "output_shape", ()))
    if len(weight_shape) != 4 or len(input_shape) < 4 or len(output_shape) < 4:
        return None
    out_channels, in_per_group, kernel_h, kernel_w = (int(value) for value in weight_shape)
    stride = _pair(getattr(module, "stride", (1, 1)))
    pad = _pair(getattr(module, "padding", (0, 0)), default=(0, 0))
    dilation = _pair(getattr(module, "dilation", (1, 1)))
    groups = int(getattr(module, "groups", 1))
    if int(groups) != 1:
        return None
    if int(kernel_h) != int(kernel_w) or int(stride[0]) != int(stride[1]):
        return None
    if int(pad[0]) != int(pad[1]) or int(dilation[0]) != int(dilation[1]):
        return None
    if int(input_shape[1]) != int(in_per_group) or int(output_shape[1]) != int(out_channels):
        return None
    input_gap = int(getattr(module, "input_gap", 1))
    output_gap = int(getattr(module, "output_gap", input_gap))
    input_layout = dict(getattr(module, "layout_policy_input_layout", {}) or {})
    output_layout = dict(getattr(module, "layout_policy_output_layout", {}) or {})
    params = getattr(getattr(module, "scheme", None), "params", None)
    get_slots = getattr(params, "get_slots", None)
    try:
        slot_count = int(get_slots()) if callable(get_slots) else int(RING_SLOT_COUNT)
    except Exception:
        slot_count = int(RING_SLOT_COUNT)
    raw_label = str(getattr(module, "name", "") or output_node_id or "conv")
    label = "".join(ch if ch.isalnum() else "_" for ch in raw_label).strip("_") or "conv"
    return NativeHaloConv2DSpec(
        family_label=(
            f"native_halo_{label}_{int(input_shape[1])}x{int(input_shape[2])}x{int(input_shape[3])}"
            f"_to_{int(output_shape[1])}x{int(output_shape[2])}x{int(output_shape[3])}"
            f"_k{int(kernel_h)}s{int(stride[0])}_gap{int(input_gap)}to{int(output_gap)}"
        ),
        c_in=int(input_shape[1]),
        h_in=int(input_shape[2]),
        w_in=int(input_shape[3]),
        c_out=int(output_shape[1]),
        h_out=int(output_shape[2]),
        w_out=int(output_shape[3]),
        gap_in=int(input_gap),
        gap_out=int(output_gap),
        kernel=int(kernel_h),
        stride=int(stride[0]),
        pad=int(pad[0]),
        dilation=int(dilation[0]),
        groups=int(groups),
        slot_count=int(slot_count),
        input_alpha=max(0, int(input_layout.get("alpha", 0) or 0)),
        input_beta=max(0, int(input_layout.get("beta", 0) or 0)),
        output_alpha=max(0, int(output_layout.get("alpha", 0) or 0)),
        output_beta=max(0, int(output_layout.get("beta", 0) or 0)),
    )


def _target_h_end_for_source_h(
    spec: NativeHaloConv2DSpec,
    *,
    target_h_start: int,
    source_h: int,
) -> int:
    """Greedily cover as many target rows as one source stripe can support."""

    lo = int(target_h_start) + 1
    hi = int(spec.output_h_max)
    best = int(lo)
    while int(lo) <= int(hi):
        mid = (int(lo) + int(hi)) // 2
        req0, req1 = _source_h_range_for_target(
            target_h_start=int(target_h_start),
            target_h_end=int(mid),
            input_h_min=int(spec.input_h_min),
            input_h_max=int(spec.input_h_max),
            kernel=int(spec.kernel),
            stride=int(spec.stride),
            pad=int(spec.pad),
            dilation=int(spec.dilation),
        )
        if int(req1 - req0) <= int(source_h):
            best = int(mid)
            lo = int(mid) + 1
        else:
            hi = int(mid) - 1
    return int(best)


def _stripes_for_source_h(spec: NativeHaloConv2DSpec, *, source_h: int) -> tuple[NativeHaloStripe, ...]:
    stripes: list[NativeHaloStripe] = []
    target_h = int(spec.output_h_min)
    while int(target_h) < int(spec.output_h_max):
        th0 = int(target_h)
        th1 = _target_h_end_for_source_h(spec, target_h_start=int(th0), source_h=int(source_h))
        req0, req1 = _source_h_range_for_target(
            target_h_start=int(th0),
            target_h_end=int(th1),
            input_h_min=int(spec.input_h_min),
            input_h_max=int(spec.input_h_max),
            kernel=int(spec.kernel),
            stride=int(spec.stride),
            pad=int(spec.pad),
            dilation=int(spec.dilation),
        )
        sh0, sh1 = _extend_h_range_to_length(
            required_start=int(req0),
            required_end=int(req1),
            desired_len=int(source_h),
            lower=int(spec.input_h_min),
            upper=int(spec.input_h_max),
        )
        stripes.append(
            NativeHaloStripe(
                index=int(len(stripes)),
                target_h_start=int(th0),
                target_h_end=int(th1),
                source_h_start=int(sh0),
                source_h_end=int(sh1),
            )
        )
        target_h = int(th1)
    return tuple(stripes)


def _diag_indices_for_task(
    spec: NativeHaloConv2DSpec,
    stripe: NativeHaloStripe,
    *,
    source_channel_count: int,
    target_channel_count: int,
) -> set[int]:
    source_slots = _slot_indices(
        int(source_channel_count),
        int(stripe.source_h),
        int(spec.w_in),
        int(spec.gap_in),
    )
    target_slots = _slot_indices(
        int(target_channel_count),
        int(stripe.target_h),
        int(spec.w_out),
        int(spec.gap_out),
    )
    shifts: set[int] = set()
    out_h_values = torch.arange(int(stripe.target_h_start), int(stripe.target_h_end), dtype=torch.int64)
    target_local_h_values = out_h_values - int(stripe.target_h_start)
    out_w_values = torch.arange(int(spec.w_out), dtype=torch.int64)
    for kh in range(int(spec.kernel)):
        in_h_values = (
            out_h_values * int(spec.stride)
            - int(spec.pad)
            + int(kh) * int(spec.dilation)
        )
        source_local_h_values = in_h_values - int(stripe.source_h_start)
        valid_h = (
            (in_h_values >= int(spec.input_h_min))
            & (in_h_values < int(spec.input_h_max))
            & (source_local_h_values >= 0)
            & (source_local_h_values < int(stripe.source_h))
        )
        if not bool(valid_h.any().item()):
            continue
        source_local_h = source_local_h_values[valid_h]
        target_local_h = target_local_h_values[valid_h]
        for kw in range(int(spec.kernel)):
            in_w_values = (
                out_w_values * int(spec.stride)
                - int(spec.pad)
                + int(kw) * int(spec.dilation)
            )
            valid_w = (in_w_values >= 0) & (in_w_values < int(spec.w_in))
            if not bool(valid_w.any().item()):
                continue
            source_w = in_w_values[valid_w]
            target_w = out_w_values[valid_w]
            source_values = source_slots[
                :,
                source_local_h[:, None],
                source_w[None, :],
            ]
            target_values = target_slots[
                :,
                target_local_h[:, None],
                target_w[None, :],
            ]
            diff = (
                source_values[:, None, :, :]
                - target_values[None, :, :, :]
            ).reshape(-1).remainder(int(spec.slot_count))
            shifts.update(int(value) for value in torch.unique(diff).tolist())
    shifts.discard(0)
    return shifts


_PLAN_CACHE: dict[tuple[Any, ...], NativeHaloConv2DPlan] = {}


def native_halo_conv2d_plan(spec: NativeHaloConv2DSpec) -> NativeHaloConv2DPlan:
    key = tuple(spec.to_dict().items())
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    source_tile = _heuristic_channel_tile(int(spec.c_in), int(spec.gap_in))
    target_tile = _heuristic_channel_tile(int(spec.c_out), int(spec.gap_out))
    source_groups_per_tile = _ceil_div(int(source_tile), _phase_count(int(spec.gap_in)))
    denom = int(source_groups_per_tile) * int(spec.w_in) * _phase_count(int(spec.gap_in))
    input_total_h = int(spec.input_h_max) - int(spec.input_h_min)
    source_h = min(int(input_total_h), max(1, int(spec.slot_count) // max(1, int(denom))))
    if _packed_active_slots(int(source_tile), int(source_h), int(spec.w_in), int(spec.gap_in)) > int(spec.slot_count):
        raise ValueError(f"native halo source tile does not fit {spec.family_label}")
    stripes = _stripes_for_source_h(spec, source_h=int(source_h))
    if any(
        _packed_active_slots(
            int(target_tile),
            int(stripe.target_h),
            int(spec.w_out),
            int(spec.gap_out),
        )
        > int(spec.slot_count)
        for stripe in stripes
    ):
        raise ValueError(f"native halo target tile does not fit {spec.family_label}")
    source_group_count = _ceil_div(int(spec.c_in), int(source_tile))
    target_group_count = _ceil_div(int(spec.c_out), int(target_tile))
    diag_cache: dict[tuple[int, int, int], set[int]] = {}
    program_diags: list[int] = []
    program_rots: list[int] = []
    group_n1s: list[int] = []
    group_rots: list[int] = []
    group_baby: list[int] = []
    group_giant: list[int] = []
    for stripe in stripes:
        for source_group in range(int(source_group_count)):
            source_start = int(source_group) * int(source_tile)
            source_end = min(int(spec.c_in), int(source_start) + int(source_tile))
            entries: list[set[int]] = []
            for target_group in range(int(target_group_count)):
                target_start = int(target_group) * int(target_tile)
                target_end = min(int(spec.c_out), int(target_start) + int(target_tile))
                diag_key = (
                    int(stripe.index),
                    int(source_end - source_start),
                    int(target_end - target_start),
                )
                if diag_key not in diag_cache:
                    diag_cache[diag_key] = _diag_indices_for_task(
                        spec,
                        stripe,
                        source_channel_count=int(source_end - source_start),
                        target_channel_count=int(target_end - target_start),
                    )
                diag_indices = set(diag_cache[diag_key])
                entries.append(diag_indices)
                _n1, rotations, _baby, _giant = _native_best_common_bsgs(
                    (diag_indices,),
                    slots=int(spec.slot_count),
                )
                program_diags.append(int(len(diag_indices)))
                program_rots.append(int(rotations))
            n1, rotations, baby, giant = _native_best_common_bsgs(
                tuple(entries),
                slots=int(spec.slot_count),
            )
            group_n1s.append(int(n1))
            group_rots.append(int(rotations))
            group_baby.append(int(baby))
            group_giant.append(int(giant))

    plan = NativeHaloConv2DPlan(
        spec=spec,
        source_channel_tile=int(source_tile),
        target_channel_tile=int(target_tile),
        stripes=tuple(stripes),
        program_diagonal_counts=tuple(program_diags),
        program_rotation_counts=tuple(program_rots),
        group_n1s=tuple(group_n1s),
        group_shared_rotations=tuple(group_rots),
        group_baby_rotations=tuple(group_baby),
        group_giant_rotations=tuple(group_giant),
    )
    _PLAN_CACHE[key] = plan
    return plan


class NativeHaloRelayoutKernel:
    def __init__(
        self,
        *,
        plan: NativeHaloConv2DPlan,
        direction: Literal["compact_to_native", "native_to_compact"],
        name: str,
        output_shape: torch.Size,
        fhe_output_shape: torch.Size,
    ) -> None:
        if str(direction) not in {"compact_to_native", "native_to_compact"}:
            raise ValueError(f"unknown native halo relayout direction {direction!r}")
        self.plan = plan
        self.spec = plan.spec
        self.direction = str(direction)
        self.name = str(name)
        self.output_shape = torch.Size(output_shape)
        self.fhe_output_shape = torch.Size(fhe_output_shape)
        self.level: int | None = None
        self.bsgs_ratio = 2.0
        self.diagonals: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
        self.transform_ids: dict[tuple[int, int], int] = {}

    def _compact_input_index(self, *, channel: int, h: int, w: int) -> int:
        return _idx_chw_gap(
            int(channel),
            int(h) + max(0, int(self.spec.input_alpha)),
            int(w),
            int(self.spec.h_in) + max(0, int(self.spec.input_alpha)) + max(0, int(self.spec.input_beta)),
            int(self.spec.w_in),
            int(self.spec.gap_in),
        )

    def _compact_output_index(self, *, channel: int, h: int, w: int) -> int:
        return _idx_chw_gap(
            int(channel),
            int(h) + max(0, int(self.spec.output_alpha)),
            int(w),
            int(self.spec.h_out) + max(0, int(self.spec.output_alpha)) + max(0, int(self.spec.output_beta)),
            int(self.spec.w_out),
            int(self.spec.gap_out),
        )

    def _native_source_index(self, *, stripe: NativeHaloStripe, group: int, channel: int, h: int, w: int) -> int:
        block = int(stripe.index) * int(self.plan.source_channel_group_count) + int(group)
        return int(block) * int(self.spec.slot_count) + _idx_chw_gap(
            int(channel),
            int(h),
            int(w),
            int(stripe.source_h),
            int(self.spec.w_in),
            int(self.spec.gap_in),
        )

    def _native_target_index(self, *, stripe: NativeHaloStripe, group: int, channel: int, h: int, w: int) -> int:
        block = int(stripe.index) * int(self.plan.target_channel_group_count) + int(group)
        return int(block) * int(self.spec.slot_count) + _idx_chw_gap(
            int(channel),
            int(h),
            int(w),
            int(stripe.target_h),
            int(self.spec.w_out),
            int(self.spec.gap_out),
        )

    def _iter_compact_to_native(self):
        for stripe in self.plan.stripes:
            for group in range(int(self.plan.source_channel_group_count)):
                channel_start = int(group) * int(self.plan.source_channel_tile)
                channel_end = min(int(self.spec.c_in), int(channel_start) + int(self.plan.source_channel_tile))
                for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
                    for global_h in range(int(stripe.source_h_start), int(stripe.source_h_end)):
                        local_h = int(global_h) - int(stripe.source_h_start)
                        for w_index in range(int(self.spec.w_in)):
                            yield (
                                self._compact_input_index(channel=int(channel), h=int(global_h), w=int(w_index)),
                                self._native_source_index(
                                    stripe=stripe,
                                    group=int(group),
                                    channel=int(local_channel),
                                    h=int(local_h),
                                    w=int(w_index),
                                ),
                            )

    def _iter_native_to_compact(self):
        for stripe in self.plan.stripes:
            for group in range(int(self.plan.target_channel_group_count)):
                channel_start = int(group) * int(self.plan.target_channel_tile)
                channel_end = min(int(self.spec.c_out), int(channel_start) + int(self.plan.target_channel_tile))
                for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
                    for global_h in range(int(stripe.target_h_start), int(stripe.target_h_end)):
                        local_h = int(global_h) - int(stripe.target_h_start)
                        for w_index in range(int(self.spec.w_out)):
                            yield (
                                self._native_target_index(
                                    stripe=stripe,
                                    group=int(group),
                                    channel=int(local_channel),
                                    h=int(local_h),
                                    w=int(w_index),
                                ),
                                self._compact_output_index(channel=int(channel), h=int(global_h), w=int(w_index)),
                            )

    def _iter_mappings(self):
        return self._iter_compact_to_native() if self.direction == "compact_to_native" else self._iter_native_to_compact()

    def compile(self, scheme: Any, *, level: int) -> None:
        self.cleanup(getattr(scheme, "backend", None))
        slots = int(self.spec.slot_count)
        diagonals: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
        for source_index, output_index in self._iter_mappings():
            input_block = int(source_index // int(slots))
            output_block = int(output_index // int(slots))
            source_local = int(source_index % int(slots))
            output_local = int(output_index % int(slots))
            diag_index = int((source_local - output_local) % int(slots))
            block = diagonals.setdefault((int(output_block), int(input_block)), {})
            diag = block.get(int(diag_index))
            if diag is None:
                diag = torch.zeros((int(slots),), dtype=torch.float32)
                block[int(diag_index)] = diag
            diag[int(output_local)] = 1.0
        self.level = int(level)
        self.diagonals = diagonals
        self.transform_ids = {
            (int(row), int(col)): int(transform_id)
            for (row, col), transform_id in scheme.lt_evaluator.generate_transforms(self).items()
        }

    def apply(self, source_ct: Any) -> Any:
        if not self.transform_ids:
            raise RuntimeError(f"native halo relayout kernel {self.name} has not been compiled")
        return source_ct.scheme.lt_evaluator.evaluate_transforms(self, source_ct)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "direction": str(self.direction),
            "name": str(self.name),
            "level": None if self.level is None else int(self.level),
            "rows": int(max((int(row) for row, _col in self.transform_ids), default=-1) + 1),
            "cols": int(max((int(col) for _row, col in self.transform_ids), default=-1) + 1),
            "lt_tasks": int(len(self.transform_ids)),
            "diagonal_count": int(sum(len(block) for block in self.diagonals.values())),
        }

    def cleanup(self, backend: Any | None) -> None:
        if backend is not None:
            delete = getattr(backend, "DeleteLinearTransform", None)
            if callable(delete):
                for transform_id in list(self.transform_ids.values()):
                    try:
                        delete(int(transform_id))
                    except Exception:
                        pass
        self.transform_ids = {}
        self.diagonals = {}


def _build_conv_transform(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    stripe: NativeHaloStripe,
    source_group: int,
    target_group: int,
    level: int,
    scheme: Any,
    group_n1: int,
    compact_target_block: int | None = None,
) -> Any | None:
    slots = int(spec.slot_count)
    compact_output = compact_target_block is not None
    source_start = int(source_group) * int(plan.source_channel_tile)
    source_end = min(int(spec.c_in), int(source_start) + int(plan.source_channel_tile))
    target_start = int(target_group) * int(plan.target_channel_tile)
    target_end = min(int(spec.c_out), int(target_start) + int(plan.target_channel_tile))
    source_count = int(source_end - source_start)
    target_count = int(target_end - target_start)
    source_slots = _slot_indices(int(source_count), int(stripe.source_h), int(spec.w_in), int(spec.gap_in))
    target_slots = (
        None
        if bool(compact_output)
        else _slot_indices(int(target_count), int(stripe.target_h), int(spec.w_out), int(spec.gap_out))
    )
    target_channels = torch.arange(int(target_start), int(target_end), dtype=torch.int64)
    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    for kh in range(int(spec.kernel)):
        for kw in range(int(spec.kernel)):
            coeff = weight[
                int(target_start): int(target_end),
                int(source_start): int(source_end),
                int(kh),
                int(kw),
            ].to(dtype=torch.float32)
            if not bool(torch.any(coeff != 0).item()):
                continue
            coeff_by_source_target = coeff.t().contiguous()
            for out_h in range(int(stripe.target_h_start), int(stripe.target_h_end)):
                in_h = int(out_h) * int(spec.stride) - int(spec.pad) + int(kh) * int(spec.dilation)
                if int(in_h) < int(spec.input_h_min) or int(in_h) >= int(spec.input_h_max):
                    continue
                source_local_h = int(in_h) - int(stripe.source_h_start)
                target_local_h = int(out_h) - int(stripe.target_h_start)
                if int(source_local_h) < 0 or int(source_local_h) >= int(stripe.source_h):
                    continue
                if int(target_local_h) < 0 or int(target_local_h) >= int(stripe.target_h):
                    continue
                for out_w in range(int(spec.w_out)):
                    in_w = int(out_w) * int(spec.stride) - int(spec.pad) + int(kw) * int(spec.dilation)
                    if int(in_w) < 0 or int(in_w) >= int(spec.w_in):
                        continue
                    source_vec = source_slots[:, int(source_local_h), int(in_w)]
                    if bool(compact_output):
                        compact_slots = torch.tensor(
                            [
                                _idx_chw_gap(
                                    int(channel),
                                    int(out_h),
                                    int(out_w),
                                    int(spec.h_out),
                                    int(spec.w_out),
                                    int(spec.gap_out),
                                )
                                for channel in target_channels.tolist()
                            ],
                            dtype=torch.int64,
                        )
                        target_block_mask = (
                            torch.div(compact_slots, int(slots), rounding_mode="floor")
                            == int(compact_target_block)
                        )
                        if not bool(target_block_mask.any().item()):
                            continue
                        target_vec = torch.remainder(compact_slots[target_block_mask], int(slots))
                        coeff_values = coeff_by_source_target[:, target_block_mask].reshape(-1).to(dtype=torch.float32)
                    else:
                        assert target_slots is not None
                        target_vec = target_slots[:, int(target_local_h), int(out_w)]
                        coeff_values = coeff_by_source_target.reshape(-1).to(dtype=torch.float32)
                    diag_index = (source_vec[:, None] - target_vec[None, :]).remainder(int(slots))
                    output_slot = target_vec[None, :].expand_as(diag_index)
                    key_parts.append((diag_index.reshape(-1) * int(slots) + output_slot.reshape(-1)).to(dtype=torch.int64))
                    value_parts.append(coeff_values)
    if not key_parts:
        return None
    keys, values = _coalesce_native_rows(torch.cat(key_parts), torch.cat(value_parts))
    if int(keys.numel()) == 0:
        return None
    diag_indices = torch.div(keys, int(slots), rounding_mode="floor").to(dtype=torch.int64)
    output_slots = torch.remainder(keys, int(slots)).to(dtype=torch.int64)
    diag_tensors: dict[int, torch.Tensor] = {}
    unique, counts = torch.unique_consecutive(diag_indices, return_counts=True)
    start = 0
    for diag_value, count_value in zip(unique.tolist(), counts.tolist()):
        end = int(start + int(count_value))
        diag = torch.zeros((int(slots),), dtype=torch.float32)
        diag.index_add_(0, output_slots[int(start): int(end)], values[int(start): int(end)].to(dtype=torch.float32))
        diag_tensors[int(diag_value)] = diag
        start = int(end)
    target_index = (
        int(compact_target_block)
        if bool(compact_output)
        else int(stripe.index) * int(plan.target_channel_group_count) + int(target_group)
    )
    source_index = int(stripe.index) * int(plan.source_channel_group_count) + int(source_group)
    baby, giant = _bsgs_rotation_sets(set(int(value) for value in diag_tensors), slots=int(slots), n1=int(group_n1))
    return SimpleNamespace(
        name=(
            f"native_halo_{spec.family_label}_s{int(stripe.index)}_src{int(source_group)}"
            f"_tgt{int(target_group)}"
            + ("" if not bool(compact_output) else f"_compact{int(compact_target_block)}")
        ),
        diagonals={(0, 0): {int(index): diag for index, diag in sorted(diag_tensors.items())}},
        level=int(level),
        scheme=scheme,
        fhe_output_shape=torch.Size([1, int(slots)]),
        output_shape=torch.Size([1, int(slots)]),
        target_index=int(target_index),
        input_id=f"native_source_tile_{int(source_index)}",
        selected_n1=int(group_n1),
        baby_shifts=tuple(sorted(int(value) for value in baby)),
        giant_shifts=tuple(sorted(int(value) for value in giant)),
        rotation_group_id=f"native_halo:{spec.family_label}:s{int(stripe.index)}:src{int(source_group)}",
        rotation_cost_owner=bool(int(target_group) == 0 and (not bool(compact_output) or int(compact_target_block) == 0)),
    )


class NativeHaloStripeNoRIConvExecutor:
    kernel_kind = "native_halo_stripe_no_ri_conv2d"
    use_ct_pt_hybrid_packing = False
    native_halo_input_capable = True
    native_halo_output_capable = True

    def __init__(self, *, module: Any, spec: NativeHaloConv2DSpec, output_node_id: str) -> None:
        self.module = module
        self.spec = spec
        self.output_node_id = str(output_node_id)
        self.native_plan = native_halo_conv2d_plan(spec)
        self.slots = int(spec.slot_count)
        self.rows = int(self.native_plan.output_ct_count)
        self.cols = int(self.native_plan.input_ct_count)
        self.output_shape = getattr(module, "output_shape", None)
        self.fhe_output_shape = getattr(module, "fhe_output_shape", None)
        self.groups_by_input_index: dict[int, Any] = {}
        self.target_indices_by_input_index: dict[int, tuple[int, ...]] = {}
        self.input_relayout_kernel: NativeHaloRelayoutKernel | None = None
        self.output_relayout_kernel: NativeHaloRelayoutKernel | None = None
        self.bias_vector: torch.Tensor | None = None
        self.bias_plaintexts: tuple[Any | None, ...] = ()
        self._bias_plaintext_cache: dict[tuple[int, int], Any] = {}
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.compile_count = 0
        self.last_runtime_timing: dict[str, float] = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
            "input_relayout_s": 0.0,
            "output_relayout_s": 0.0,
        }
        self._compile_cache_metadata: dict[str, Any] = {}

    def supports_scheme(self, scheme: Any | None) -> bool:
        return scheme is not None

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def _uses_tight_compact_output(self) -> bool:
        return not _spec_has_output_halo(self.native_plan.spec)

    def runtime_native_fhe_output_shape(self) -> torch.Size:
        if self._uses_tight_compact_output():
            if self.fhe_output_shape is not None:
                return torch.Size(self.fhe_output_shape)
        return torch.Size([int(self.rows), int(self.slots)])

    def _runtime_spec(self) -> NativeHaloConv2DSpec:
        input_layout = dict(getattr(self.module, "layout_policy_input_layout", {}) or {})
        output_layout = dict(getattr(self.module, "layout_policy_output_layout", {}) or {})
        return replace(
            self.spec,
            input_alpha=max(0, int(input_layout.get("alpha", 0) or 0)),
            input_beta=max(0, int(input_layout.get("beta", 0) or 0)),
            output_alpha=max(0, int(output_layout.get("alpha", 0) or 0)),
            output_beta=max(0, int(output_layout.get("beta", 0) or 0)),
        )

    def _refresh_runtime_plan(self) -> bool:
        runtime_spec = self._runtime_spec()
        changed = tuple(runtime_spec.to_dict().items()) != tuple(self.native_plan.spec.to_dict().items())
        if bool(changed):
            self.native_plan = native_halo_conv2d_plan(runtime_spec)
        self.slots = int(self.native_plan.spec.slot_count)
        self.rows = int(self.native_plan.output_ct_count)
        self.cols = int(self.native_plan.input_ct_count)
        self.output_shape = getattr(self.module, "output_shape", self.output_shape)
        self.fhe_output_shape = getattr(self.module, "fhe_output_shape", self.fhe_output_shape)
        return bool(changed)

    def _validate_module(self) -> None:
        weight = getattr(self.module, "on_weight", None)
        if weight is None:
            raise RuntimeError(f"{self.output_node_id} has no fused Orion weight for native halo Conv2d")
        if tuple(int(v) for v in tuple(weight.shape)) != tuple(int(v) for v in self.spec.weight_shape):
            raise RuntimeError(f"{self.output_node_id} weight shape does not match native halo spec")

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        self._compile_cache_metadata = dict(metadata or {})

    def _bias_chunk(self, *, block_index: int) -> torch.Tensor | None:
        if self.bias_vector is None:
            return None
        if self._uses_tight_compact_output():
            return self._compact_bias_chunk(block_index=int(block_index))
        target_group_count = int(self.native_plan.target_channel_group_count)
        stripe_index = int(block_index) // int(target_group_count)
        target_group = int(block_index) % int(target_group_count)
        if int(stripe_index) >= len(self.native_plan.stripes):
            return None
        stripe = self.native_plan.stripes[int(stripe_index)]
        channel_start = int(target_group) * int(self.native_plan.target_channel_tile)
        channel_end = min(int(self.spec.c_out), int(channel_start) + int(self.native_plan.target_channel_tile))
        out = torch.zeros((int(self.slots),), dtype=torch.float32)
        for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
            bias_value = float(self.bias_vector[int(channel)])
            if bias_value == 0.0:
                continue
            for out_h in range(int(stripe.target_h_start), int(stripe.target_h_end)):
                local_h = int(out_h) - int(stripe.target_h_start)
                for out_w in range(int(self.spec.w_out)):
                    slot = _idx_chw_gap(
                        int(local_channel),
                        int(local_h),
                        int(out_w),
                        int(stripe.target_h),
                        int(self.spec.w_out),
                        int(self.spec.gap_out),
                    )
                    out[int(slot)] = float(bias_value)
        return out

    def _compact_bias_chunk(self, *, block_index: int) -> torch.Tensor | None:
        if self.bias_vector is None:
            return None
        spec = self.native_plan.spec
        start = int(block_index) * int(self.slots)
        stop = int(start) + int(self.slots)
        out = torch.zeros((int(self.slots),), dtype=torch.float32)
        for channel in range(int(spec.c_out)):
            bias_value = float(self.bias_vector[int(channel)])
            if bias_value == 0.0:
                continue
            for out_h in range(int(spec.h_out)):
                for out_w in range(int(spec.w_out)):
                    flat_index = _idx_chw_gap(
                        int(channel),
                        int(out_h),
                        int(out_w),
                        int(spec.h_out),
                        int(spec.w_out),
                        int(spec.gap_out),
                    )
                    if int(flat_index) < int(start) or int(flat_index) >= int(stop):
                        continue
                    out[int(flat_index) - int(start)] = float(bias_value)
        return out

    def _compile_bias_plaintexts_at_level(self, scheme: Any, *, level: int) -> tuple[Any | None, ...]:
        if self.bias_vector is None:
            return ()
        scale = int(scheme.params.get_default_scale())
        plaintexts: list[Any | None] = []
        for block_index in range(int(self.rows)):
            chunk = self._bias_chunk(block_index=int(block_index))
            ptxt = None if chunk is None else scheme.encode(chunk, level=int(level), scale=int(scale))
            if ptxt is not None:
                self._bias_plaintext_cache[(int(block_index), int(level))] = ptxt
            plaintexts.append(ptxt)
        return tuple(plaintexts)

    def _add_bias(self, ct: Any, *, block_index: int) -> Any:
        if self.bias_vector is None:
            return ct
        level = int(ct.level())
        ptxt = self._bias_plaintext_cache.get((int(block_index), int(level)))
        if ptxt is None:
            chunk = self._bias_chunk(block_index=int(block_index))
            if chunk is None:
                return ct
            ptxt = _encode_plaintext_for_add(ct, chunk)
            self._bias_plaintext_cache[(int(block_index), int(level))] = ptxt
        return _add_plaintext_for_add(ct, ptxt)

    def compile_cache_metadata(self) -> dict[str, Any]:
        self._refresh_runtime_plan()
        return {
            "kind": type(self).__name__,
            "kernel_kind": self.kernel_kind,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "use_ct_pt_hybrid_packing": False,
            "native_halo_input_capable": True,
            "native_halo_output_capable": True,
            "same_shape_runtime_layout": "native_halo_stripe_no_ri_io",
            "native_internal_runtime_layout": "native_halo_stripe_no_ri",
            "native_halo_conv2d_plan": self.native_plan.to_dict(),
            "conv_lt_raw_submatrix_tasks": int(self.native_plan.submatrix_program_count),
            "conv_lt_effective_submatrix_tasks": int(self.native_plan.sharing_group_count),
            "input_relayout": {},
            "output_relayout": {},
            "relayout_sparse_lt_tasks": 0,
            "native_c_only_rotations": int(self.native_plan.c_only_rotations),
            "native_cb_shared_rotations": int(self.native_plan.cb_shared_rotations),
            "native_shared_baby_rotations": int(self.native_plan.shared_baby_rotations),
            "native_shared_giant_rotations": int(self.native_plan.shared_giant_rotations),
            "groups_by_input_index": [
                {
                    "input_index": int(input_index),
                    "storage_key": str(getattr(group, "_storage_key", "")),
                    "target_indices": [
                        int(value)
                        for value in self.target_indices_by_input_index.get(int(input_index), ())
                    ],
                }
                for input_index, group in sorted(self.groups_by_input_index.items())
            ],
        }

    def compile(self, scheme: Any) -> None:
        changed = self._refresh_runtime_plan()
        if bool(changed) and (
            self.groups_by_input_index or self.input_relayout_kernel is not None or self.output_relayout_kernel is not None
        ):
            self.cleanup(getattr(scheme, "backend", None))
        if self.groups_by_input_index:
            return
        prepare_started = time.time()
        self._validate_module()
        self.last_runtime_timing = {
            "prepare_plans_s": float(time.time() - prepare_started),
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
            "input_relayout_s": 0.0,
            "output_relayout_s": 0.0,
        }
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        weight = getattr(self.module, "on_weight").detach().to(dtype=torch.float32)
        input_level = int(self._level(scheme))
        conv_level = max(0, int(input_level - 1))
        conv_output_level = max(0, int(conv_level - 1))
        self.bias_plaintexts = self._compile_bias_plaintexts_at_level(scheme, level=int(conv_output_level))

        compile_started = time.time()
        group_index = 0
        compact_output = self._uses_tight_compact_output()
        for stripe in self.native_plan.stripes:
            for source_group in range(int(self.native_plan.source_channel_group_count)):
                ordered: list[tuple[int, Any]] = []
                group_n1 = int(self.native_plan.group_n1s[int(group_index)])
                for target_group in range(int(self.native_plan.target_channel_group_count)):
                    target_blocks = range(int(self.rows)) if bool(compact_output) else (None,)
                    for target_block in target_blocks:
                        transform = _build_conv_transform(
                            spec=self.native_plan.spec,
                            plan=self.native_plan,
                            weight=weight,
                            stripe=stripe,
                            source_group=int(source_group),
                            target_group=int(target_group),
                            level=int(conv_level),
                            scheme=scheme,
                            group_n1=int(group_n1),
                            compact_target_block=target_block,
                        )
                        if transform is None:
                            continue
                        ordered.append((int(transform.target_index), transform))
                input_index = int(stripe.index) * int(self.native_plan.source_channel_group_count) + int(source_group)
                if ordered:
                    ordered.sort(key=lambda item: int(item[0]))
                    group = UnifiedTransformGroup([transform for _target_index, transform in ordered])
                    group.compile_unified(scheme.backend)
                    self.groups_by_input_index[int(input_index)] = group
                    self.target_indices_by_input_index[int(input_index)] = tuple(
                        int(target_index) for target_index, _transform in ordered
                    )
                group_index += 1

        self.input_relayout_kernel = None
        self.output_relayout_kernel = None
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.compile(scheme)
        self.last_runtime_timing["input_relayout_s"] = 0.0
        self.last_runtime_timing["output_relayout_s"] = 0.0
        ids = tuple(int(value) for value in getattr(source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            raise RuntimeError(
                f"{self.output_node_id} native halo-stripe Conv2d requires externally relayouted "
                f"native source tiles: expected at least {self.cols} CTs, got {len(ids)}"
            )

        output_blocks: list[Any | None] = [None for _ in range(int(self.rows))]
        evaluate_started = time.time()
        for input_index, group in sorted(self.groups_by_input_index.items()):
            output_ids = group.evaluate_unified(int(ids[int(input_index)]), scheme.backend)
            for target_index, output_id in zip(self.target_indices_by_input_index[int(input_index)], output_ids):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(self.slots)]),
                    torch.Size([1, int(self.slots)]),
                )
                partial = _rescale_cipher_tensor(partial)
                if output_blocks[int(target_index)] is None:
                    output_blocks[int(target_index)] = partial
                else:
                    lhs, rhs = _align_ciphertexts_for_add(output_blocks[int(target_index)], partial)
                    output_blocks[int(target_index)] = lhs + rhs
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)

        postprocess_started = time.time()
        output_ids: list[int] = []
        for block_index, block_ct in enumerate(output_blocks):
            if block_ct is None:
                raise RuntimeError(f"missing native halo output block {block_index} for {self.output_node_id}")
            block_ct = self._add_bias(block_ct, block_index=int(block_index))
            block_ct.set_scale(int(scheme.params.get_default_scale()))
            output_ids.append(int(block_ct.ids[0]))
            block_ct.ids = []
        native_output = CipherTensor(
            scheme,
            output_ids,
            self.output_shape,
            self.runtime_native_fhe_output_shape(),
        )
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        self.last_runtime_io = {
            "runtime_lowering": (
                "native_halo_stripe_no_ri+tight_compact_output"
                if self._uses_tight_compact_output()
                else "native_halo_stripe_no_ri"
            ),
            "provider_executor": type(self).__name__,
            "native_halo_input_capable": True,
            "native_halo_output_capable": True,
            "use_ct_pt_hybrid_packing": False,
            "native_input_ct_count": int(self.cols),
            "native_output_ct_count": int(self.rows),
            "native_output_storage_layout": "tight_compact" if self._uses_tight_compact_output() else "native_halo_stripe",
            "internal_input_relayout": False,
            "internal_output_relayout": False,
        }
        return {self.output_node_id: native_output}

    def cleanup(self, backend: Any | None) -> None:
        if self.input_relayout_kernel is not None:
            self.input_relayout_kernel.cleanup(backend)
        if self.output_relayout_kernel is not None:
            self.output_relayout_kernel.cleanup(backend)
        if backend is not None:
            delete = getattr(backend, "DeleteLinearTransform", None)
            if callable(delete):
                for group in list(self.groups_by_input_index.values()):
                    for transform_id in list(getattr(group, "unified_ids", []) or []):
                        try:
                            delete(int(transform_id))
                        except Exception:
                            pass
        self.groups_by_input_index = {}
        self.target_indices_by_input_index = {}
