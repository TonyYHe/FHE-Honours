from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal
import math
import os
import time

import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.experimental.cir.hybrid_schedule import (
    hybrid_pair_schedule_compatible,
    hybrid_pair_schedule_reject_reason,
    mark_hybrid_schedule_padding_allowed,
    materialize_hybrid_pair_layout_schedules,
    optimize_hybrid_pair_layout,
)
from orion.nn.unified_transform import UnifiedTransformGroup


def _unified_output_fusion_enabled() -> bool:
    return os.environ.get("ORION_UNIFIED_LT_OUTPUT_FUSION", "1").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


try:
    from orion.experimental.cir.runtime_group import (
        _add_plaintext_for_add,
        _align_ciphertexts_for_add,
        _encode_plaintext_for_add,
        _rescale_cipher_tensor,
    )
except ImportError:
    def _rescale_cipher_tensor(ct: Any) -> Any:
        if len(getattr(ct, "ids", ())) != 1:
            raise ValueError("region-first rescale helper expects a single-ciphertext tensor")
        if bool(getattr(ct.scheme.backend, "lt_outputs_are_rescaled", False)):
            return ct
        rescaled_id = ct.evaluator.rescale(int(ct.ids[0]), in_place=False)
        return type(ct)(ct.scheme, [int(rescaled_id)], ct.shape, ct.on_shape)

    def _encode_plaintext_for_add(ct: Any, values: torch.Tensor) -> Any:
        scale = int(ct.scheme.params.get_default_scale())
        if bool(getattr(ct.scheme.backend, "align_addition_scales", False)):
            scale = max(1, int(ct.scale()))
            ct.set_scale(int(scale))
        return ct.scheme.encode(values, ct.level(), scale=scale)

    def _add_plaintext_for_add(ct: Any, ptxt: Any) -> Any:
        if bool(getattr(ct.scheme.backend, "align_addition_scales", False)):
            scale = max(1, int(ct.scale()))
            ct.set_scale(int(scale))
            ptxt.set_scale(int(scale))
        return ct + ptxt

    def _align_ciphertexts_for_add(left: Any, right: Any) -> tuple[Any, Any]:
        if bool(getattr(left.scheme.backend, "align_addition_scales", False)):
            scale = max(1, int(left.scale()))
            left.set_scale(int(scale))
            right.set_scale(int(scale))
        return left, right

from .ir import (
    CanonicalTemplateEntry,
    ConvSchemePlan,
    ExecutionStats,
    FamilyTemplateBank,
    LinearTransformStep,
    LinearTransformTerm,
    PlainCipherTensor,
    PreparedPlaintext,
    TensorRegion,
)
from .r34_section_extractor import build_section_extract_plan, halo_rotation_count


RING_SLOT_COUNT = 32768
R34SameShapePolicy = Literal["inter_group_hybrid", "intra_group_pack2"]


@dataclass(frozen=True)
class R34SameShapeStageSpec:
    family_label: str
    stage: str
    c: int
    h: int
    w: int
    gap: int
    policy: R34SameShapePolicy
    materializer: str
    slot_count: int = RING_SLOT_COUNT

    @property
    def weight_shape(self) -> tuple[int, int, int, int]:
        return (int(self.c), int(self.c), 3, 3)

    @property
    def source_group_count(self) -> int:
        return int(r34_source_group_count(c=int(self.c), gap=int(self.gap)))


R34_STAGE1_SAME_SPEC = R34SameShapeStageSpec(
    family_label="stage1_same",
    stage="stage1",
    c=64,
    h=56,
    w=56,
    gap=4,
    policy="inter_group_hybrid",
    materializer="policy_inter_group_hybrid",
)
R34_STAGE2_SAME_SPEC = R34SameShapeStageSpec(
    family_label="stage2_same",
    stage="stage2",
    c=128,
    h=28,
    w=28,
    gap=8,
    policy="inter_group_hybrid",
    materializer="policy_inter_group_hybrid",
)
R34_STAGE3_SAME_SPEC = R34SameShapeStageSpec(
    family_label="stage3_same",
    stage="stage3",
    c=256,
    h=14,
    w=14,
    gap=16,
    policy="intra_group_pack2",
    materializer="policy_intra_group_pack2",
)
R34_STAGE4_SAME_SPEC = R34SameShapeStageSpec(
    family_label="stage4_same",
    stage="stage4",
    c=512,
    h=7,
    w=7,
    gap=32,
    policy="intra_group_pack2",
    materializer="policy_intra_group_pack2",
)


R34_SAME_SHAPE_STAGE_SPECS = (
    R34_STAGE1_SAME_SPEC,
    R34_STAGE2_SAME_SPEC,
    R34_STAGE3_SAME_SPEC,
    R34_STAGE4_SAME_SPEC,
)
R34_SAME_SHAPE_STAGE_SPEC_BY_LABEL = {str(spec.family_label): spec for spec in R34_SAME_SHAPE_STAGE_SPECS}


def r34_same_shape_spec_for_family_label(family_label: str) -> R34SameShapeStageSpec:
    try:
        return R34_SAME_SHAPE_STAGE_SPEC_BY_LABEL[str(family_label)]
    except KeyError as exc:
        raise KeyError(f"unknown R34 same-shape family label {family_label!r}") from exc


@dataclass(frozen=True)
class R34SameShapeHaloStripe:
    index: int
    target_h_start: int
    target_h_end: int
    source_h_start: int
    source_h_end: int
    local_h: int
    core_h_start: int
    core_h_end: int
    halo_top: int
    halo_bottom: int
    core_shift_rows: int
    active_slots: int
    relayout_rotations: int
    relayout_mask_mults: int
    conv_lt_tasks: int = 1

    def to_dict(self) -> dict[str, int]:
        return {
            "index": int(self.index),
            "target_h_start": int(self.target_h_start),
            "target_h_end": int(self.target_h_end),
            "source_h_start": int(self.source_h_start),
            "source_h_end": int(self.source_h_end),
            "local_h": int(self.local_h),
            "core_h_start": int(self.core_h_start),
            "core_h_end": int(self.core_h_end),
            "halo_top": int(self.halo_top),
            "halo_bottom": int(self.halo_bottom),
            "core_shift_rows": int(self.core_shift_rows),
            "active_slots": int(self.active_slots),
            "relayout_rotations": int(self.relayout_rotations),
            "relayout_mask_mults": int(self.relayout_mask_mults),
            "conv_lt_tasks": int(self.conv_lt_tasks),
        }


@dataclass(frozen=True)
class R34SameShapeHaloRelayoutPlan:
    family_label: str
    policy: R34SameShapePolicy
    c: int
    h: int
    w: int
    gap: int
    slot_count: int
    legacy_flat_block_count: int
    legacy_flat_conv_lt_tasks: int
    legacy_flat_spatial_boundary_tasks: int
    legacy_flat_offdiag_tasks: int
    stripes: tuple[R34SameShapeHaloStripe, ...]
    notes: tuple[str, ...] = ()

    @property
    def raw_conv_lt_tasks(self) -> int:
        return int(sum(int(stripe.conv_lt_tasks) for stripe in self.stripes))

    @property
    def effective_conv_lt_tasks_with_hybrid(self) -> int:
        return int(_ceil_div(int(self.raw_conv_lt_tasks), 2))

    @property
    def effective_conv_lt_tasks_without_hybrid(self) -> int:
        return int(self.raw_conv_lt_tasks)

    @property
    def relayout_rotations(self) -> int:
        return int(sum(int(stripe.relayout_rotations) for stripe in self.stripes))

    @property
    def relayout_mask_mults(self) -> int:
        return int(sum(int(stripe.relayout_mask_mults) for stripe in self.stripes))

    @property
    def stored_slots(self) -> int:
        return int(sum(int(stripe.active_slots) for stripe in self.stripes))

    @property
    def max_active_slots(self) -> int:
        return int(max((int(stripe.active_slots) for stripe in self.stripes), default=0))

    @property
    def conv_source_target_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(index), int(index)) for index, _stripe in enumerate(self.stripes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_label": str(self.family_label),
            "policy": str(self.policy),
            "runtime_layout": "height_stripe_halo_local_hardcoded",
            "conv_dependency": "current_ciphertext_materialized_halo_only",
            "c": int(self.c),
            "h": int(self.h),
            "w": int(self.w),
            "gap": int(self.gap),
            "slot_count": int(self.slot_count),
            "stripe_count": int(len(self.stripes)),
            "raw_conv_lt_tasks": int(self.raw_conv_lt_tasks),
            "effective_conv_lt_tasks_with_hybrid": int(self.effective_conv_lt_tasks_with_hybrid),
            "effective_conv_lt_tasks_without_hybrid": int(self.effective_conv_lt_tasks_without_hybrid),
            "legacy_flat_block_count": int(self.legacy_flat_block_count),
            "legacy_flat_conv_lt_tasks": int(self.legacy_flat_conv_lt_tasks),
            "legacy_flat_spatial_boundary_tasks": int(self.legacy_flat_spatial_boundary_tasks),
            "legacy_flat_offdiag_tasks": int(self.legacy_flat_offdiag_tasks),
            "relayout_rotations": int(self.relayout_rotations),
            "relayout_mask_mults": int(self.relayout_mask_mults),
            "relayout_sparse_lt_tasks": 0,
            "stored_slots": int(self.stored_slots),
            "max_active_slots": int(self.max_active_slots),
            "conv_source_target_pairs": [
                [int(source), int(target)] for source, target in self.conv_source_target_pairs
            ],
            "stripes": [stripe.to_dict() for stripe in self.stripes],
            "notes": [str(note) for note in self.notes],
        }


_R34_SAME_SHAPE_HARDCODED_STRIPE_RANGES: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "stage1_same": (
        (0, 8, 0, 9),
        (8, 15, 7, 16),
        (15, 22, 14, 23),
        (22, 29, 21, 30),
        (29, 36, 28, 37),
        (36, 43, 35, 44),
        (43, 50, 42, 51),
        (50, 56, 49, 56),
    ),
    "stage2_same": (
        (0, 8, 0, 9),
        (8, 15, 7, 16),
        (15, 22, 14, 23),
        (22, 28, 21, 28),
    ),
    "stage3_same": (
        (0, 8, 0, 9),
        (8, 14, 7, 14),
    ),
    "stage4_same": (
        (0, 3, 0, 4),
        (3, 5, 2, 6),
        (5, 7, 4, 7),
    ),
}

_R34_SAME_SHAPE_LEGACY_FLAT_STATS: dict[str, dict[str, int]] = {
    "stage1_same": {
        "flat_block_count": 7,
        "conv_lt_tasks": 47,
        "spatial_boundary_tasks": 36,
        "offdiag_tasks": 40,
    },
    "stage2_same": {
        "flat_block_count": 4,
        "conv_lt_tasks": 14,
        "spatial_boundary_tasks": 9,
        "offdiag_tasks": 10,
    },
    "stage3_same": {
        "flat_block_count": 2,
        "conv_lt_tasks": 4,
        "spatial_boundary_tasks": 2,
        "offdiag_tasks": 2,
    },
    "stage4_same": {
        "flat_block_count": 2,
        "conv_lt_tasks": 4,
        "spatial_boundary_tasks": 2,
        "offdiag_tasks": 2,
    },
}


def r34_same_shape_hardcoded_relayout_plan(
    spec: R34SameShapeStageSpec | None = None,
    *,
    family_label: str | None = None,
) -> R34SameShapeHaloRelayoutPlan:
    if spec is None:
        if family_label is None:
            raise ValueError("R34 same-shape relayout plan requires spec or family_label")
        spec = r34_same_shape_spec_for_family_label(str(family_label))
    ranges = _R34_SAME_SHAPE_HARDCODED_STRIPE_RANGES[str(spec.family_label)]
    legacy = _R34_SAME_SHAPE_LEGACY_FLAT_STATS[str(spec.family_label)]
    stripes: list[R34SameShapeHaloStripe] = []
    for index, (target_h_start, target_h_end, source_h_start, source_h_end) in enumerate(ranges):
        section = build_section_extract_plan(
            family_label=str(spec.family_label),
            c=int(spec.c),
            w=int(spec.w),
            gap=int(spec.gap),
            input_h=int(spec.h),
            kernel=3,
            stride=1,
            pad=1,
            target_h_range=(int(target_h_start), int(target_h_end)),
            source_h_range=(int(source_h_start), int(source_h_end)),
            relocate_core=True,
        )
        has_prev = int(index) > 0
        has_next = int(index + 1) < len(ranges)
        rotations = halo_rotation_count(section, has_prev=bool(has_prev), has_next=bool(has_next))
        if int(section.core_shift_rows) != 0:
            raise ValueError(f"{spec.family_label} hardcoded stripe {index} unexpectedly needs core relayout shift")
        if int(section.active_slots) > int(spec.slot_count):
            raise ValueError(
                f"{spec.family_label} hardcoded stripe {index} uses {section.active_slots} slots > {spec.slot_count}"
            )
        stripes.append(
            R34SameShapeHaloStripe(
                index=int(index),
                target_h_start=int(section.target_h_start),
                target_h_end=int(section.target_h_end),
                source_h_start=int(section.source_h_start),
                source_h_end=int(section.source_h_end),
                local_h=int(section.local_h),
                core_h_start=int(section.core_h_start),
                core_h_end=int(section.core_h_end),
                halo_top=int(section.halo_top),
                halo_bottom=int(section.halo_bottom),
                core_shift_rows=int(section.core_shift_rows),
                active_slots=int(section.active_slots),
                relayout_rotations=int(rotations),
                relayout_mask_mults=int(rotations),
            )
        )
    return R34SameShapeHaloRelayoutPlan(
        family_label=str(spec.family_label),
        policy=spec.policy,
        c=int(spec.c),
        h=int(spec.h),
        w=int(spec.w),
        gap=int(spec.gap),
        slot_count=int(spec.slot_count),
        legacy_flat_block_count=int(legacy["flat_block_count"]),
        legacy_flat_conv_lt_tasks=int(legacy["conv_lt_tasks"]),
        legacy_flat_spatial_boundary_tasks=int(legacy["spatial_boundary_tasks"]),
        legacy_flat_offdiag_tasks=int(legacy["offdiag_tasks"]),
        stripes=tuple(stripe for stripe in stripes),
        notes=(
            "Hardcoded R34 same-shape reduced-halo height stripes; source rows are the minimal stencil range that fits the ring.",
            "The conv LT for each stripe is local to the current ciphertext after top/bottom halo rows have been materialized.",
            "Relayout halo fill is costed separately as one rotation plus one plaintext mask multiply per non-boundary halo side.",
        ),
    )


@dataclass(frozen=True)
class R34NativeAlignedHaloStripe:
    index: int
    target_h_start: int
    target_h_end: int
    source_h_start: int
    source_h_end: int
    local_h: int
    target_local_h_start: int
    target_local_h_end: int
    active_source_slots: int
    active_target_slots: int

    def to_dict(self) -> dict[str, int]:
        return {
            "index": int(self.index),
            "target_h_start": int(self.target_h_start),
            "target_h_end": int(self.target_h_end),
            "source_h_start": int(self.source_h_start),
            "source_h_end": int(self.source_h_end),
            "local_h": int(self.local_h),
            "target_local_h_start": int(self.target_local_h_start),
            "target_local_h_end": int(self.target_local_h_end),
            "active_source_slots": int(self.active_source_slots),
            "active_target_slots": int(self.active_target_slots),
        }


@dataclass(frozen=True)
class R34NativeAlignedHaloPlan:
    family_label: str
    c: int
    h: int
    w: int
    gap: int
    slot_count: int
    channel_tile: int
    source_channel_group_count: int
    target_channel_group_count: int
    stripes: tuple[R34NativeAlignedHaloStripe, ...]
    program_diagonal_counts: tuple[int, ...]
    program_rotation_counts: tuple[int, ...]
    group_n1s: tuple[int, ...]
    group_shared_rotations: tuple[int, ...]
    group_baby_rotations: tuple[int, ...]
    group_giant_rotations: tuple[int, ...]

    @property
    def input_ct_count(self) -> int:
        return int(len(self.stripes) * int(self.source_channel_group_count))

    @property
    def output_ct_count(self) -> int:
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

    @property
    def max_active_source_slots(self) -> int:
        return int(max((int(stripe.active_source_slots) for stripe in self.stripes), default=0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_label": str(self.family_label),
            "runtime_layout": "native_aligned_halo_no_ri",
            "conv_dependency": "native_aligned_halo_source_tiles",
            "c": int(self.c),
            "h": int(self.h),
            "w": int(self.w),
            "gap": int(self.gap),
            "slot_count": int(self.slot_count),
            "channel_tile": int(self.channel_tile),
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
            "max_active_source_slots": int(self.max_active_source_slots),
            "program_diagonal_counts": [int(value) for value in self.program_diagonal_counts],
            "program_rotation_counts": [int(value) for value in self.program_rotation_counts],
            "group_n1s": [int(value) for value in self.group_n1s],
            "group_shared_rotations": [int(value) for value in self.group_shared_rotations],
            "group_baby_rotations": [int(value) for value in self.group_baby_rotations],
            "group_giant_rotations": [int(value) for value in self.group_giant_rotations],
            "stripes": [stripe.to_dict() for stripe in self.stripes],
            "notes": [
                "Channel groups are aligned to one natural gap^2 phase group when possible.",
                "Height stripes use full local halo capacity; active outputs are embedded in source-local coordinates.",
                "No real/imag lane packing is used; B sharing is only same-source UnifiedTransformGroup rotation sharing.",
            ],
        }


def _native_aligned_channel_tile(spec: R34SameShapeStageSpec) -> int:
    phase_count = max(1, int(spec.gap) * int(spec.gap))
    return max(1, min(int(spec.c), int(phase_count)))


def _native_aligned_max_local_h(spec: R34SameShapeStageSpec, *, channel_tile: int) -> int:
    per_row_slots = int(spec.w) * max(1, int(spec.gap) * int(spec.gap))
    if int(channel_tile) > int(spec.gap) * int(spec.gap):
        per_row_slots *= _ceil_div(int(channel_tile), int(spec.gap) * int(spec.gap))
    return max(1, min(int(spec.h), int(_spec_slot_count(spec)) // max(1, int(per_row_slots))))


def _native_aligned_halo_stripes(spec: R34SameShapeStageSpec, *, channel_tile: int) -> tuple[R34NativeAlignedHaloStripe, ...]:
    max_local_h = _native_aligned_max_local_h(spec, channel_tile=int(channel_tile))
    target_chunk_h = max(1, int(max_local_h) - 2)
    stripes: list[R34NativeAlignedHaloStripe] = []
    target_h_start = 0
    while int(target_h_start) < int(spec.h):
        target_h_end = min(int(spec.h), int(target_h_start) + int(target_chunk_h))
        source_h_start = max(0, min(int(target_h_start) - 1, int(spec.h) - int(max_local_h)))
        source_h_end = min(int(spec.h), int(source_h_start) + int(max_local_h))
        if int(source_h_end - source_h_start) < int(max_local_h) and int(source_h_start) > 0:
            source_h_start = max(0, int(source_h_end) - int(max_local_h))
        local_h = int(source_h_end) - int(source_h_start)
        active_source_slots = _packed_active_slots(int(channel_tile), int(local_h), int(spec.w), int(spec.gap))
        if int(active_source_slots) > int(_spec_slot_count(spec)):
            raise ValueError(
                f"{spec.family_label} native aligned stripe uses {active_source_slots} slots > {_spec_slot_count(spec)}"
            )
        active_target_slots = _packed_active_slots(
            int(channel_tile),
            int(target_h_end) - int(target_h_start),
            int(spec.w),
            int(spec.gap),
        )
        stripes.append(
            R34NativeAlignedHaloStripe(
                index=int(len(stripes)),
                target_h_start=int(target_h_start),
                target_h_end=int(target_h_end),
                source_h_start=int(source_h_start),
                source_h_end=int(source_h_end),
                local_h=int(local_h),
                target_local_h_start=int(target_h_start) - int(source_h_start),
                target_local_h_end=int(target_h_end) - int(source_h_start),
                active_source_slots=int(active_source_slots),
                active_target_slots=int(active_target_slots),
            )
        )
        target_h_start = int(target_h_end)
    return tuple(stripes)


def _native_slot_indices(channel_count: int, height: int, width: int, gap: int) -> torch.Tensor:
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


def _native_halo_diag_indices(
    *,
    spec: R34SameShapeStageSpec,
    stripe: R34NativeAlignedHaloStripe,
    source_channel_count: int,
    target_channel_count: int,
) -> set[int]:
    source_slots = _native_slot_indices(
        int(source_channel_count),
        int(stripe.local_h),
        int(spec.w),
        int(spec.gap),
    )
    target_slots = _native_slot_indices(
        int(target_channel_count),
        int(stripe.local_h),
        int(spec.w),
        int(spec.gap),
    )
    diag_indices: set[int] = set()
    for kh in range(3):
        for kw in range(3):
            pieces: list[torch.Tensor] = []
            for out_h in range(int(stripe.target_h_start), int(stripe.target_h_end)):
                in_h = int(out_h) - 1 + int(kh)
                if int(in_h) < 0 or int(in_h) >= int(spec.h):
                    continue
                source_local_h = int(in_h) - int(stripe.source_h_start)
                target_local_h = int(out_h) - int(stripe.source_h_start)
                if (
                    int(source_local_h) < 0
                    or int(source_local_h) >= int(stripe.local_h)
                    or int(target_local_h) < 0
                    or int(target_local_h) >= int(stripe.local_h)
                ):
                    continue
                for out_w in range(int(spec.w)):
                    in_w = int(out_w) - 1 + int(kw)
                    if int(in_w) < 0 or int(in_w) >= int(spec.w):
                        continue
                    diff = (
                        source_slots[:, int(source_local_h), int(in_w)][:, None]
                        - target_slots[:, int(target_local_h), int(out_w)][None, :]
                    ).reshape(-1)
                    pieces.append(diff.remainder(int(_spec_slot_count(spec))))
            if pieces:
                diag_indices.update(int(value) for value in torch.unique(torch.cat(pieces)).tolist())
    diag_indices.discard(0)
    return diag_indices


def _bsgs_rotation_sets(diag_indices: set[int], *, slots: int, n1: int) -> tuple[set[int], set[int]]:
    baby: set[int] = set()
    giant: set[int] = set()
    for value in diag_indices:
        rot = int(value) % int(slots)
        giant.add(int(((rot // int(n1)) * int(n1)) % int(slots)))
        baby.add(int(rot % int(n1)))
    return {int(value) for value in baby if int(value) != 0}, {int(value) for value in giant if int(value) != 0}


def _native_best_common_bsgs(entries: tuple[set[int], ...], *, slots: int) -> tuple[int, int, int, int]:
    best: tuple[int, int, int, int] | None = None
    n1 = 1
    while int(n1) < int(slots):
        shared_baby: set[int] = set()
        giant_total = 0
        for diag_indices in entries:
            baby, giant = _bsgs_rotation_sets(diag_indices, slots=int(slots), n1=int(n1))
            shared_baby.update(baby)
            giant_total += int(len(giant))
        total = int(len(shared_baby) + giant_total)
        candidate = (int(total), int(n1), int(len(shared_baby)), int(giant_total))
        if best is None or int(candidate[0]) < int(best[0]) or (
            int(candidate[0]) == int(best[0]) and int(candidate[1]) > int(best[1])
        ):
            best = candidate
        n1 <<= 1
    if best is None:
        return 1, 0, 0, 0
    return int(best[1]), int(best[0]), int(best[2]), int(best[3])


_R34_NATIVE_ALIGNED_HALO_PLAN_CACHE: dict[str, R34NativeAlignedHaloPlan] = {}


def _relayout_block_rows_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in list((metadata or {}).get("blocks", []))
        if isinstance(row, dict)
    ]


def r34_native_aligned_halo_plan(
    spec: R34SameShapeStageSpec | None = None,
    *,
    family_label: str | None = None,
) -> R34NativeAlignedHaloPlan:
    if spec is None:
        if family_label is None:
            raise ValueError("R34 native aligned halo plan requires spec or family_label")
        spec = r34_same_shape_spec_for_family_label(str(family_label))
    cache_key = str(spec.family_label)
    cached = _R34_NATIVE_ALIGNED_HALO_PLAN_CACHE.get(str(cache_key))
    if cached is not None:
        return cached
    channel_tile = _native_aligned_channel_tile(spec)
    source_group_count = _ceil_div(int(spec.c), int(channel_tile))
    target_group_count = _ceil_div(int(spec.c), int(channel_tile))
    stripes = _native_aligned_halo_stripes(spec, channel_tile=int(channel_tile))
    diag_cache: dict[tuple[int, int, int], set[int]] = {}
    program_diagonal_counts: list[int] = []
    program_rotation_counts: list[int] = []
    group_n1s: list[int] = []
    group_shared_rotations: list[int] = []
    group_baby_rotations: list[int] = []
    group_giant_rotations: list[int] = []
    for stripe in stripes:
        for source_group in range(int(source_group_count)):
            source_start = int(source_group) * int(channel_tile)
            source_end = min(int(spec.c), int(source_start) + int(channel_tile))
            group_entries: list[set[int]] = []
            for target_group in range(int(target_group_count)):
                target_start = int(target_group) * int(channel_tile)
                target_end = min(int(spec.c), int(target_start) + int(channel_tile))
                key = (
                    int(stripe.index),
                    int(source_end) - int(source_start),
                    int(target_end) - int(target_start),
                )
                if key not in diag_cache:
                    diag_cache[key] = _native_halo_diag_indices(
                        spec=spec,
                        stripe=stripe,
                        source_channel_count=int(source_end) - int(source_start),
                        target_channel_count=int(target_end) - int(target_start),
                    )
                diag_indices = set(diag_cache[key])
                group_entries.append(diag_indices)
                _n1, rotations, _baby, _giant = _native_best_common_bsgs((diag_indices,), slots=int(_spec_slot_count(spec)))
                program_diagonal_counts.append(int(len(diag_indices)))
                program_rotation_counts.append(int(rotations))
            n1, rotations, baby, giant = _native_best_common_bsgs(
                tuple(group_entries),
                slots=int(_spec_slot_count(spec)),
            )
            group_n1s.append(int(n1))
            group_shared_rotations.append(int(rotations))
            group_baby_rotations.append(int(baby))
            group_giant_rotations.append(int(giant))
    plan = R34NativeAlignedHaloPlan(
        family_label=str(spec.family_label),
        c=int(spec.c),
        h=int(spec.h),
        w=int(spec.w),
        gap=int(spec.gap),
        slot_count=int(_spec_slot_count(spec)),
        channel_tile=int(channel_tile),
        source_channel_group_count=int(source_group_count),
        target_channel_group_count=int(target_group_count),
        stripes=tuple(stripes),
        program_diagonal_counts=tuple(program_diagonal_counts),
        program_rotation_counts=tuple(program_rotation_counts),
        group_n1s=tuple(group_n1s),
        group_shared_rotations=tuple(group_shared_rotations),
        group_baby_rotations=tuple(group_baby_rotations),
        group_giant_rotations=tuple(group_giant_rotations),
    )
    _R34_NATIVE_ALIGNED_HALO_PLAN_CACHE[str(cache_key)] = plan
    return plan


class R34NativeAlignedRelayoutKernel:
    """Sparse LT adapter between Orion compact CHW-gap blocks and native halo tiles."""

    def __init__(
        self,
        *,
        spec: R34SameShapeStageSpec,
        native_plan: R34NativeAlignedHaloPlan,
        direction: Literal["compact_to_native", "native_to_compact"],
        name: str,
        output_shape: torch.Size,
        fhe_output_shape: torch.Size,
    ) -> None:
        if str(direction) not in {"compact_to_native", "native_to_compact"}:
            raise ValueError(f"unknown R34 native relayout direction {direction!r}")
        self.spec = spec
        self.native_plan = native_plan
        self.direction = str(direction)
        self.name = str(name)
        self.output_shape = torch.Size(output_shape)
        self.fhe_output_shape = torch.Size(fhe_output_shape)
        self.level: int | None = None
        self.bsgs_ratio = 2.0
        self.diagonals: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
        self.transform_ids: dict[tuple[int, int], int] = {}

    def _compiled_diag_indices_by_block(self) -> dict[tuple[int, int], tuple[int, ...]]:
        saved = getattr(self, "_compile_cache_diag_indices_by_block", {}) or {}
        out: dict[tuple[int, int], tuple[int, ...]] = {
            (int(row), int(col)): tuple(sorted(int(idx) for idx in indices))
            for (row, col), indices in dict(saved).items()
        }
        for key, block in self.diagonals.items():
            out.setdefault(
                (int(key[0]), int(key[1])),
                tuple(sorted(int(idx) for idx in block.keys())),
            )
        return out

    def _compiled_slot_count(self, key: tuple[int, int]) -> int:
        saved = getattr(self, "_compile_cache_slot_count_by_block", {}) or {}
        if key in saved:
            return int(saved[key])
        block = self.diagonals.get(key, {})
        for diag in block.values():
            try:
                return int(len(diag))
            except TypeError:
                continue
        return int(_spec_slot_count(self.spec))

    def _compact_index(self, *, channel: int, h: int, w: int) -> int:
        return _idx_chw_gap(
            int(channel),
            int(h),
            int(w),
            int(self.spec.h),
            int(self.spec.w),
            int(self.spec.gap),
        )

    def _native_index(self, *, stripe: R34NativeAlignedHaloStripe, group: int, channel: int, h: int, w: int) -> int:
        block_index = int(stripe.index) * int(self.native_plan.source_channel_group_count) + int(group)
        return int(block_index) * int(_spec_slot_count(self.spec)) + _idx_chw_gap(
            int(channel),
            int(h),
            int(w),
            int(stripe.local_h),
            int(self.spec.w),
            int(self.spec.gap),
        )

    def _iter_compact_to_native_mappings(self):
        for stripe in self.native_plan.stripes:
            for group in range(int(self.native_plan.source_channel_group_count)):
                channel_start = int(group) * int(self.native_plan.channel_tile)
                channel_end = min(int(self.spec.c), int(channel_start) + int(self.native_plan.channel_tile))
                for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
                    for global_h in range(int(stripe.source_h_start), int(stripe.source_h_end)):
                        local_h = int(global_h) - int(stripe.source_h_start)
                        for w_index in range(int(self.spec.w)):
                            yield (
                                self._compact_index(channel=int(channel), h=int(global_h), w=int(w_index)),
                                self._native_index(
                                    stripe=stripe,
                                    group=int(group),
                                    channel=int(local_channel),
                                    h=int(local_h),
                                    w=int(w_index),
                                ),
                            )

    def _iter_native_to_compact_mappings(self):
        for stripe in self.native_plan.stripes:
            for group in range(int(self.native_plan.target_channel_group_count)):
                channel_start = int(group) * int(self.native_plan.channel_tile)
                channel_end = min(int(self.spec.c), int(channel_start) + int(self.native_plan.channel_tile))
                for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
                    for global_h in range(int(stripe.target_h_start), int(stripe.target_h_end)):
                        local_h = int(global_h) - int(stripe.source_h_start)
                        for w_index in range(int(self.spec.w)):
                            yield (
                                self._native_index(
                                    stripe=stripe,
                                    group=int(group),
                                    channel=int(local_channel),
                                    h=int(local_h),
                                    w=int(w_index),
                                ),
                                self._compact_index(channel=int(channel), h=int(global_h), w=int(w_index)),
                            )

    def _iter_mappings(self):
        if self.direction == "compact_to_native":
            return self._iter_compact_to_native_mappings()
        return self._iter_native_to_compact_mappings()

    def compile(self, scheme: Any, *, level: int) -> None:
        self.cleanup(getattr(scheme, "backend", None))
        slots = int(_spec_slot_count(self.spec))
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

    def compile_from_cache_metadata(self, scheme: Any, metadata: dict[str, Any], *, level: int) -> None:
        metadata = dict(metadata or {})
        block_rows = _relayout_block_rows_from_metadata(metadata)
        if not block_rows:
            raise RuntimeError(
                f"Missing cached R34 native relayout block manifest for {self.name!r}; "
                "re-run with io_mode='save' to rebuild the compile cache."
            )
        self.cleanup(getattr(scheme, "backend", None))
        if metadata.get("name"):
            self.name = str(metadata["name"])
        self.level = int(level)
        self.diagonals = {}
        diag_indices_by_block: dict[tuple[int, int], tuple[int, ...]] = {}
        slot_count_by_block: dict[tuple[int, int], int] = {}
        for block in block_rows:
            key = (int(block.get("row", 0)), int(block.get("col", 0)))
            diag_indices = tuple(sorted(int(idx) for idx in block.get("diag_indices", [])))
            if not diag_indices:
                continue
            self.diagonals[key] = {int(idx): [] for idx in diag_indices}
            diag_indices_by_block[key] = diag_indices
            slot_count_by_block[key] = int(block.get("slot_count", _spec_slot_count(self.spec)))
        if not self.diagonals:
            raise RuntimeError(
                f"Cached R34 native relayout manifest for {self.name!r} has no nonempty LT blocks"
            )
        self._compile_cache_diag_indices_by_block = diag_indices_by_block
        self._compile_cache_slot_count_by_block = slot_count_by_block
        self.transform_ids = {
            (int(row), int(col)): int(transform_id)
            for (row, col), transform_id in scheme.lt_evaluator.generate_transforms(self).items()
        }

    def apply(self, source_ct: Any) -> Any:
        if not self.transform_ids:
            raise RuntimeError(f"R34 native relayout kernel {self.name} has not been compiled")
        return source_ct.scheme.lt_evaluator.evaluate_transforms(self, source_ct)

    def to_metadata(self) -> dict[str, Any]:
        diag_indices_by_block = self._compiled_diag_indices_by_block()
        block_keys = sorted(
            set(diag_indices_by_block)
            | {(int(row), int(col)) for row, col in self.transform_ids}
        )
        blocks = [
            {
                "row": int(row),
                "col": int(col),
                "diag_indices": [int(idx) for idx in diag_indices_by_block.get((int(row), int(col)), ())],
                "diag_count": int(len(diag_indices_by_block.get((int(row), int(col)), ()))),
                "slot_count": int(self._compiled_slot_count((int(row), int(col)))),
                "transform_id": (
                    None
                    if (int(row), int(col)) not in self.transform_ids
                    else int(self.transform_ids[(int(row), int(col))])
                ),
            }
            for row, col in block_keys
        ]
        return {
            "cache_schema_version": 1,
            "direction": str(self.direction),
            "name": str(self.name),
            "level": None if self.level is None else int(self.level),
            "rows": int(max((int(row) for row, _col in block_keys), default=-1) + 1),
            "cols": int(max((int(col) for _row, col in block_keys), default=-1) + 1),
            "lt_tasks": int(len(blocks)),
            "diagonal_count": int(sum(len(block["diag_indices"]) for block in blocks)),
            "blocks": blocks,
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


def r34_source_group_count(*, c: int, gap: int) -> int:
    phase_count = max(1, int(gap) * int(gap))
    return int(_ceil_div(int(c), int(phase_count)))


def r34_same_shape_policy_from_source_group_count(source_group_count: int) -> R34SameShapePolicy:
    return "inter_group_hybrid" if int(source_group_count) > 1 else "intra_group_pack2"


def r34_same_shape_policy(*, c: int, gap: int) -> R34SameShapePolicy:
    return r34_same_shape_policy_from_source_group_count(r34_source_group_count(c=int(c), gap=int(gap)))


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _ceil_pow2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (int(value) - 1).bit_length()


def _spec_slot_count(spec: R34SameShapeStageSpec) -> int:
    return max(1, int(getattr(spec, "slot_count", RING_SLOT_COUNT)))


def _packed_active_slots(c: int, h: int, w: int, gap: int) -> int:
    phase_count = max(1, int(gap) * int(gap))
    groups = _ceil_div(int(c), int(phase_count))
    return int(groups) * int(h) * int(gap) * int(w) * int(gap)


def _orion_hybrid_block_height(output_length: int, *, slot_count: int = RING_SLOT_COUNT) -> tuple[int, int]:
    if int(output_length) <= int(slot_count):
        block_height = _ceil_pow2(int(output_length))
        output_rotations = int((int(slot_count) // int(block_height)).bit_length() - 1)
        return int(block_height), int(output_rotations)
    return int(slot_count), 0


def _pack_gap_flat(tensor: torch.Tensor, *, shape: tuple[int, int, int], gap: int) -> torch.Tensor:
    c, h, w = (int(v) for v in shape)
    value = tensor.detach().cpu().to(dtype=torch.float32)
    if tuple(int(v) for v in value.shape) != (int(c), int(h), int(w)):
        raise ValueError(f"expected tensor shape {(int(c), int(h), int(w))}, got {tuple(value.shape)}")
    out = torch.zeros((_packed_active_slots(int(c), int(h), int(w), int(gap)),), dtype=torch.float32)
    for channel in range(int(c)):
        for ih in range(int(h)):
            for iw in range(int(w)):
                out[_idx_chw_gap(int(channel), int(ih), int(iw), int(h), int(w), int(gap))] = value[int(channel), int(ih), int(iw)]
    return out


def _split_flat_into_ring_blocks(
    flat: torch.Tensor,
    *,
    prefix: str,
    slot_count: int = RING_SLOT_COUNT,
) -> dict[str, PlainCipherTensor]:
    inputs: dict[str, PlainCipherTensor] = {}
    block_count = _ceil_div(int(flat.numel()), int(slot_count))
    for block_index in range(int(block_count)):
        start = int(block_index) * int(slot_count)
        end = min(int(flat.numel()), int(start + int(slot_count)))
        block = torch.zeros((int(slot_count),), dtype=flat.dtype)
        block[: int(end - start)] = flat[int(start): int(end)]
        input_id = f"{prefix}_{int(block_index)}"
        inputs[input_id] = PlainCipherTensor(block, label=input_id)
    return inputs


def _idx_chw_gap(c: int, h: int, w: int, H: int, W: int, gap: int) -> int:
    g = max(1, int(gap))
    if int(g) == 1:
        return int(c) * int(H) * int(W) + int(h) * int(W) + int(w)
    s2 = int(g * g)
    group = int(c) // int(s2)
    phase = int(c) % int(s2)
    phase_h = int(phase) // int(g)
    phase_w = int(phase) % int(g)
    Hm = int(H * g)
    Wm = int(W * g)
    return int(group) * int(Hm * Wm) + (int(h) * int(g) + int(phase_h)) * int(Wm) + int(w) * int(g) + int(phase_w)


def _idx_chw_gap_tensor(c: torch.Tensor, h: torch.Tensor, w: torch.Tensor, *, H: int, W: int, gap: int) -> torch.Tensor:
    g = max(1, int(gap))
    c = c.to(dtype=torch.int64)
    h = h.to(dtype=torch.int64)
    w = w.to(dtype=torch.int64)
    if int(g) == 1:
        return c * int(H) * int(W) + h * int(W) + w
    s2 = int(g * g)
    group = torch.div(c, int(s2), rounding_mode="floor")
    phase = torch.remainder(c, int(s2))
    phase_h = torch.div(phase, int(g), rounding_mode="floor")
    phase_w = torch.remainder(phase, int(g))
    Hm = int(H * g)
    Wm = int(W * g)
    return group * int(Hm * Wm) + (h * int(g) + phase_h) * int(Wm) + w * int(g) + phase_w


def _channel_base_indices(*, channels: int, height: int, width: int, gap: int) -> torch.Tensor:
    channel = torch.arange(int(channels), dtype=torch.int64)
    zeros = torch.zeros_like(channel)
    return _idx_chw_gap_tensor(channel, zeros, zeros, H=int(height), W=int(width), gap=int(gap)).to(dtype=torch.int64)


def _same_shape_spatial_parts(*, spec: R34SameShapeStageSpec, kh: int, kw: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    h = int(spec.h)
    w = int(spec.w)
    gap = int(spec.gap)
    oh_all = torch.arange(int(h), dtype=torch.int64)
    ow_all = torch.arange(int(w), dtype=torch.int64)
    grid_oh, grid_ow = torch.meshgrid(oh_all, ow_all, indexing="ij")
    oh = grid_oh.reshape(-1)
    ow = grid_ow.reshape(-1)
    ih = oh - 1 + int(kh)
    iw = ow - 1 + int(kw)
    valid = (ih >= 0) & (ih < int(h)) & (iw >= 0) & (iw < int(w))
    oh = oh[valid]
    ow = ow[valid]
    ih = ih[valid]
    iw = iw[valid]
    packed_w = int(w * gap)
    out_spatial = (oh * int(gap)) * int(packed_w) + ow * int(gap)
    src_spatial = (ih * int(gap)) * int(packed_w) + iw * int(gap)
    if int(out_spatial.numel()) == 0:
        return out_spatial, src_spatial, 0
    offset = int((out_spatial[0] - src_spatial[0]).item())
    return out_spatial.to(dtype=torch.int64), src_spatial.to(dtype=torch.int64), int(offset)


def _same_shape_delta_groups(spec: R34SameShapeStageSpec) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    c = int(spec.c)
    base = _channel_base_indices(channels=int(c), height=int(spec.h), width=int(spec.w), gap=int(spec.gap))
    oc = torch.arange(int(c), dtype=torch.int64)[:, None].expand(int(c), int(c)).reshape(-1)
    ic = torch.arange(int(c), dtype=torch.int64)[None, :].expand(int(c), int(c)).reshape(-1)
    deltas = (base.index_select(0, oc) - base.index_select(0, ic)).to(dtype=torch.int64)
    order = torch.argsort(deltas)
    sorted_deltas = deltas.index_select(0, order)
    sorted_oc = oc.index_select(0, order)
    sorted_ic = ic.index_select(0, order)
    unique, counts = torch.unique_consecutive(sorted_deltas, return_counts=True)
    groups: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    start = 0
    for delta, count in zip(unique.tolist(), counts.tolist()):
        end = int(start + int(count))
        groups[int(delta)] = (sorted_oc[int(start): int(end)], sorted_ic[int(start): int(end)])
        start = int(end)
    return groups


_SAME_SHAPE_GEOMETRY_CACHE: dict[
    tuple[int, int, int, int],
    tuple[
        torch.Tensor,
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        tuple[tuple[int, int, torch.Tensor, torch.Tensor, int], ...],
    ],
] = {}


def _same_shape_geometry(
    spec: R34SameShapeStageSpec,
) -> tuple[
    torch.Tensor,
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    tuple[tuple[int, int, torch.Tensor, torch.Tensor, int], ...],
]:
    key = (int(spec.c), int(spec.h), int(spec.w), int(spec.gap))
    cached = _SAME_SHAPE_GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    base_indices = _channel_base_indices(
        channels=int(spec.c),
        height=int(spec.h),
        width=int(spec.w),
        gap=int(spec.gap),
    )
    delta_groups = _same_shape_delta_groups(spec)
    spatial_parts: list[tuple[int, int, torch.Tensor, torch.Tensor, int]] = []
    for kh in range(3):
        for kw in range(3):
            out_spatial, src_spatial, spatial_offset = _same_shape_spatial_parts(spec=spec, kh=int(kh), kw=int(kw))
            if int(out_spatial.numel()) == 0:
                continue
            spatial_parts.append((int(kh), int(kw), out_spatial, src_spatial, int(spatial_offset)))
    cached = (base_indices, delta_groups, tuple(spatial_parts))
    _SAME_SHAPE_GEOMETRY_CACHE[key] = cached
    return cached


def _bsgs_meta_for_shifts(shifts: set[int]) -> tuple[int, tuple[int, ...], tuple[int, ...], ExecutionStats]:
    ordered = tuple(sorted(int(value) for value in shifts))
    return 0, ordered, (), ExecutionStats(rotations=int(len(ordered)), ct_pt_mults=int(len(ordered)), adds=int(len(ordered)))


_SAME_SHAPE_FAST_SLOT_CACHE: dict[tuple[int, int, int, int, int, int, int, int, int, int], torch.Tensor] = {}


def _same_shape_cached_fast_output_slots(
    *,
    spec: R34SameShapeStageSpec,
    slot_count: int,
    output_offset_start: int,
    input_offset_start: int,
    output_phase_offset: int,
    input_phase_offset: int,
    kh: int,
    kw: int,
    out_spatial: torch.Tensor,
    src_spatial: torch.Tensor,
) -> torch.Tensor:
    key = (
        int(spec.h),
        int(spec.w),
        int(spec.gap),
        int(slot_count),
        int(output_offset_start),
        int(input_offset_start),
        int(output_phase_offset),
        int(input_phase_offset),
        int(kh),
        int(kw),
    )
    cached = _SAME_SHAPE_FAST_SLOT_CACHE.get(key)
    if cached is not None:
        return cached
    output_offset_end = int(output_offset_start + slot_count)
    input_offset_end = int(input_offset_start + slot_count)
    out_local = int(output_phase_offset) + out_spatial
    src_local = int(input_phase_offset) + src_spatial
    valid = (
        (out_local >= int(output_offset_start))
        & (out_local < int(output_offset_end))
        & (src_local >= int(input_offset_start))
        & (src_local < int(input_offset_end))
    )
    if bool(torch.any(valid).item()):
        output_slots = (out_local[valid] - int(output_offset_start)).to(dtype=torch.int64)
    else:
        output_slots = torch.empty((0,), dtype=torch.int64)
    _SAME_SHAPE_FAST_SLOT_CACHE[key] = output_slots
    return output_slots


def _same_shape_single_group_terms(
    *,
    spec: R34SameShapeStageSpec,
    weight: torch.Tensor,
    output_block_index: int,
    input_block_index: int,
    output_start: int,
    output_end: int,
    input_start: int,
    input_end: int,
    spatial_parts: tuple[tuple[int, int, torch.Tensor, torch.Tensor, int], ...],
) -> list[tuple[int, torch.Tensor, torch.Tensor]] | None:
    gap = max(1, int(spec.gap))
    phase_count = int(gap * gap)
    packed_w = int(spec.w * gap)
    group_block = int(spec.h * gap * packed_w)
    slot_count = _spec_slot_count(spec)
    if int(group_block) <= 0 or int(group_block) % int(slot_count) != 0:
        return None
    if int(output_end) <= int(output_start) or int(input_end) <= int(input_start):
        return []
    output_group = int(output_start // int(group_block))
    input_group = int(input_start // int(group_block))
    if (
        int(output_group * phase_count) >= int(spec.c)
        or int(input_group * phase_count) >= int(spec.c)
        or int(output_end - 1) // int(group_block) != int(output_group)
        or int(input_end - 1) // int(group_block) != int(input_group)
    ):
        return None

    output_offset_start = int(output_start - int(output_group) * int(group_block))
    input_offset_start = int(input_start - int(input_group) * int(group_block))
    fast_terms: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for output_phase in range(int(phase_count)):
        output_channel = int(output_group * phase_count + output_phase)
        if int(output_channel) >= int(spec.c):
            continue
        output_phase_offset = int((output_phase // gap) * packed_w + (output_phase % gap))
        for input_phase in range(int(phase_count)):
            input_channel = int(input_group * phase_count + input_phase)
            if int(input_channel) >= int(spec.c):
                continue
            input_phase_offset = int((input_phase // gap) * packed_w + (input_phase % gap))
            for kh, kw, out_spatial, src_spatial, spatial_offset in spatial_parts:
                coeff = float(weight[int(output_channel), int(input_channel), int(kh), int(kw)])
                if coeff == 0.0:
                    continue
                output_slots = _same_shape_cached_fast_output_slots(
                    spec=spec,
                    slot_count=int(slot_count),
                    output_offset_start=int(output_offset_start),
                    input_offset_start=int(input_offset_start),
                    output_phase_offset=int(output_phase_offset),
                    input_phase_offset=int(input_phase_offset),
                    kh=int(kh),
                    kw=int(kw),
                    out_spatial=out_spatial,
                    src_spatial=src_spatial,
                )
                if int(output_slots.numel()) == 0:
                    continue
                values = torch.full((int(output_slots.numel()),), float(coeff), dtype=torch.float32)
                shift = (
                    int(output_phase_offset)
                    - int(input_phase_offset)
                    + int(spatial_offset)
                    - int(output_offset_start)
                    + int(input_offset_start)
                )
                fast_terms.append((int(shift), output_slots, values))
    return fast_terms


def _build_orion_same_shape_block_lt_assets(
    *,
    spec: R34SameShapeStageSpec,
    weight: torch.Tensor,
    family_id: str,
    output_block_index: int,
    input_block_index: int,
) -> tuple[tuple[CanonicalTemplateEntry, ...], tuple[PreparedPlaintext, ...], LinearTransformStep | None]:
    output_length = _packed_active_slots(int(spec.c), int(spec.h), int(spec.w), int(spec.gap))
    slot_count = _spec_slot_count(spec)
    output_start = int(output_block_index) * int(slot_count)
    output_end = min(int(output_length), int(output_start + int(slot_count)))
    input_start = int(input_block_index) * int(slot_count)
    input_end = min(int(output_length), int(input_start + int(slot_count)))
    base_indices, delta_groups, spatial_parts = _same_shape_geometry(spec)
    candidate_shifts: set[int] = set()
    fast_terms = _same_shape_single_group_terms(
        spec=spec,
        weight=weight,
        output_block_index=int(output_block_index),
        input_block_index=int(input_block_index),
        output_start=int(output_start),
        output_end=int(output_end),
        input_start=int(input_start),
        input_end=int(input_end),
        spatial_parts=spatial_parts,
    )
    if fast_terms is not None:
        template_entries: list[CanonicalTemplateEntry] = []
        prepared_plaintexts: list[PreparedPlaintext] = []
        terms: list[LinearTransformTerm] = []
        for shift, output_slots, values in fast_terms:
            term_index = int(len(terms))
            template_id = f"orion_vendored_r34_{spec.stage}_same_t{int(output_block_index)}_c{int(input_block_index)}_template_{int(term_index)}"
            plaintext_id = f"orion_vendored_r34_{spec.stage}_same_t{int(output_block_index)}_c{int(input_block_index)}_pt_{int(term_index)}"
            template_entries.append(
                CanonicalTemplateEntry(
                    template_id=str(template_id),
                    family_id=str(family_id),
                    key=(int(output_block_index), int(shift)),
                    fine_shift=int(shift),
                    indices=output_slots,
                    note="vendored HaloED Orion-layout same-shape template",
                )
            )
            prepared_plaintexts.append(
                PreparedPlaintext(
                    plaintext_id=str(plaintext_id),
                    template_id=str(template_id),
                    level=0,
                    scale=1.0,
                    slot_count=int(slot_count),
                    values=values,
                    note="prepared:orion_same_shape_fast_gap1",
                )
            )
            terms.append(
                LinearTransformTerm(
                    term_id=f"orion_vendored_r34_{spec.stage}_same_t{int(output_block_index)}_c{int(input_block_index)}_term_{int(term_index)}",
                    shift=int(shift),
                    plaintext_id=str(plaintext_id),
                    template_id=str(template_id),
                    lookup_indices=torch.arange(int(output_slots.numel()), dtype=torch.int64),
                    output_slot_indices=output_slots,
                    note="vendored HaloED Orion-layout same-shape term",
                )
            )
        if not terms:
            return tuple(template_entries), tuple(prepared_plaintexts), None
        shift_set = {int(term.shift) for term in terms}
        selected_n1, baby_shifts, giant_shifts, lt_base_cost = _bsgs_meta_for_shifts(shift_set)
        step = LinearTransformStep(
            step_id=f"orion_vendored_r34_{spec.stage}_same_lt_t{int(output_block_index)}_c{int(input_block_index)}",
            input_id=f"orion_source_block_{int(input_block_index)}",
            target_index=int(output_block_index),
            selected_n1=int(selected_n1),
            baby_shifts=tuple(int(v) for v in baby_shifts),
            giant_shifts=tuple(int(v) for v in giant_shifts),
            terms=tuple(terms),
            required_rotations=tuple(int(v) for v in baby_shifts),
            prepared_plaintext_ids=tuple(str(pt.plaintext_id) for pt in prepared_plaintexts),
            expected_cost=ExecutionStats(
                rotations=int(lt_base_cost.rotations),
                ct_pt_mults=int(len(terms)),
                adds=int(len(terms)),
            ),
            representation="real_bsgs",
            note="vendored HaloED Orion-layout same-shape block LT",
        )
        return tuple(template_entries), tuple(prepared_plaintexts), step

    for _kh, _kw, _out_spatial, _src_spatial, spatial_offset in spatial_parts:
        for delta in delta_groups:
            candidate_shifts.add(int(delta) + int(spatial_offset) - int(output_start) + int(input_start))

    template_entries: list[CanonicalTemplateEntry] = []
    prepared_plaintexts: list[PreparedPlaintext] = []
    terms: list[LinearTransformTerm] = []
    for shift in sorted(candidate_shifts):
        output_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        for kh, kw, out_spatial, src_spatial, spatial_offset in spatial_parts:
            needed_delta = int(shift) - int(spatial_offset) + int(output_start) - int(input_start)
            group = delta_groups.get(int(needed_delta))
            if group is None:
                continue
            oc_pairs, ic_pairs = group
            coeff = weight[
                oc_pairs.to(dtype=torch.int64),
                ic_pairs.to(dtype=torch.int64),
                int(kh),
                int(kw),
            ].to(dtype=torch.float32)
            out_global = base_indices.index_select(0, oc_pairs.to(dtype=torch.int64))[:, None] + out_spatial[None, :]
            src_global = base_indices.index_select(0, ic_pairs.to(dtype=torch.int64))[:, None] + src_spatial[None, :]
            valid = (
                (out_global >= int(output_start))
                & (out_global < int(output_end))
                & (src_global >= int(input_start))
                & (src_global < int(input_end))
            )
            if not bool(torch.any(valid).item()):
                continue
            output_parts.append((out_global - int(output_start))[valid].to(dtype=torch.int64))
            value_parts.append(coeff[:, None].expand_as(out_global)[valid].to(dtype=torch.float32))
        if not output_parts:
            continue
        output_slots = torch.cat(output_parts).to(dtype=torch.int64)
        values = torch.cat(value_parts).to(dtype=torch.float32)
        term_index = int(len(terms))
        template_id = f"orion_vendored_r34_{spec.stage}_same_t{int(output_block_index)}_c{int(input_block_index)}_template_{int(term_index)}"
        plaintext_id = f"orion_vendored_r34_{spec.stage}_same_t{int(output_block_index)}_c{int(input_block_index)}_pt_{int(term_index)}"
        template_entries.append(
            CanonicalTemplateEntry(
                template_id=str(template_id),
                family_id=str(family_id),
                key=(int(output_block_index), int(shift)),
                fine_shift=int(shift),
                indices=output_slots,
                note="vendored HaloED Orion-layout same-shape template",
            )
        )
        prepared_plaintexts.append(
            PreparedPlaintext(
                plaintext_id=str(plaintext_id),
                template_id=str(template_id),
                level=0,
                scale=1.0,
                slot_count=int(slot_count),
                values=values,
                note="prepared:orion_same_shape_fast",
            )
        )
        terms.append(
            LinearTransformTerm(
                term_id=f"orion_vendored_r34_{spec.stage}_same_t{int(output_block_index)}_c{int(input_block_index)}_term_{int(term_index)}",
                shift=int(shift),
                plaintext_id=str(plaintext_id),
                template_id=str(template_id),
                lookup_indices=torch.arange(int(output_slots.numel()), dtype=torch.int64),
                output_slot_indices=output_slots,
                note="vendored HaloED Orion-layout same-shape term",
            )
        )
    if not terms:
        return tuple(template_entries), tuple(prepared_plaintexts), None
    shift_set = {int(term.shift) for term in terms}
    selected_n1, baby_shifts, giant_shifts, lt_base_cost = _bsgs_meta_for_shifts(shift_set)
    step = LinearTransformStep(
        step_id=f"orion_vendored_r34_{spec.stage}_same_lt_t{int(output_block_index)}_c{int(input_block_index)}",
        input_id=f"orion_source_block_{int(input_block_index)}",
        target_index=int(output_block_index),
        selected_n1=int(selected_n1),
        baby_shifts=tuple(int(v) for v in baby_shifts),
        giant_shifts=tuple(int(v) for v in giant_shifts),
        terms=tuple(terms),
        required_rotations=tuple(int(v) for v in baby_shifts),
        prepared_plaintext_ids=tuple(str(pt.plaintext_id) for pt in prepared_plaintexts),
        expected_cost=ExecutionStats(
            rotations=int(lt_base_cost.rotations),
            ct_pt_mults=int(len(terms)),
            adds=int(len(terms)),
        ),
        representation="real_bsgs",
        note="vendored HaloED Orion-layout same-shape block LT",
    )
    return tuple(template_entries), tuple(prepared_plaintexts), step


def build_r34_same_shape_orion_plan(
    *,
    spec: R34SameShapeStageSpec,
    weight_override: torch.Tensor | None = None,
    bias_override: torch.Tensor | None = None,
    source_override: torch.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    output_shape: tuple[int, int, int] | None = None,
    input_gap: int | None = None,
    output_gap: int | None = None,
    output_block_indices: tuple[int, ...] | None = None,
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    torch.manual_seed(0)
    expected_policy = r34_same_shape_policy(c=int(spec.c), gap=int(spec.gap))
    if str(spec.policy) != str(expected_policy):
        raise ValueError(
            f"{spec.family_label} policy mismatch: spec says {spec.policy}, derived policy is {expected_policy}"
        )
    expected_weight_shape = tuple(int(v) for v in spec.weight_shape)
    if weight_override is None:
        weight = torch.randn(expected_weight_shape, dtype=torch.float32)
    else:
        weight = weight_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in weight.shape) != expected_weight_shape:
            raise ValueError(
                f"{spec.family_label} fused weight shape mismatch: expected {expected_weight_shape}, got {tuple(weight.shape)}"
            )
    expected_source_shape = (int(spec.c), int(spec.h), int(spec.w))
    if source_override is None:
        x = torch.randn(expected_source_shape, dtype=torch.float32)
    else:
        x = source_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in x.shape) != expected_source_shape:
            raise ValueError(
                f"{spec.family_label} source shape mismatch: expected {expected_source_shape}, got {tuple(x.shape)}"
            )
    if input_shape is not None and tuple(int(v) for v in input_shape) != expected_source_shape:
        raise ValueError(f"{spec.family_label} input_shape mismatch: {input_shape}")
    if output_shape is not None and tuple(int(v) for v in output_shape) != expected_source_shape:
        raise ValueError(f"{spec.family_label} output_shape mismatch: {output_shape}")
    if input_gap is not None and int(input_gap) != int(spec.gap):
        raise ValueError(f"{spec.family_label} input_gap mismatch: {input_gap}")
    if output_gap is not None and int(output_gap) != int(spec.gap):
        raise ValueError(f"{spec.family_label} output_gap mismatch: {output_gap}")
    if bias_override is not None:
        bias = bias_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in bias.shape) != (int(spec.c),):
            raise ValueError(f"{spec.family_label} bias shape mismatch: expected {(int(spec.c),)}, got {tuple(bias.shape)}")

    output_length = _packed_active_slots(int(spec.c), int(spec.h), int(spec.w), int(spec.gap))
    slot_count = _spec_slot_count(spec)
    output_block_count = _ceil_div(int(output_length), int(slot_count))
    if output_block_indices is None:
        selected_output_blocks = tuple(range(int(output_block_count)))
    else:
        selected_output_blocks = tuple(int(index) for index in output_block_indices)
        invalid = [index for index in selected_output_blocks if index < 0 or index >= int(output_block_count)]
        if invalid:
            raise ValueError(f"{spec.family_label} invalid output_block_indices: {invalid}")
    block_height, output_rotations = _orion_hybrid_block_height(int(output_length), slot_count=int(slot_count))
    if int(output_block_count) > 1:
        block_height = int(slot_count)
        output_rotations = 0
    if int(output_rotations) != 0:
        raise ValueError(f"{spec.family_label} unexpected single-block output fold requirement")

    family_id = f"orion_vendored_r34_{spec.stage}_same_family"
    all_templates: list[CanonicalTemplateEntry] = []
    all_prepared: list[PreparedPlaintext] = []
    lt_steps: list[LinearTransformStep] = []
    expected = ExecutionStats()
    for output_block_index in selected_output_blocks:
        for input_block_index in range(int(output_block_count)):
            templates, prepared, step = _build_orion_same_shape_block_lt_assets(
                spec=spec,
                weight=weight,
                family_id=str(family_id),
                output_block_index=int(output_block_index),
                input_block_index=int(input_block_index),
            )
            all_templates.extend(templates)
            all_prepared.extend(prepared)
            if step is None:
                continue
            lt_steps.append(step)
            expected = expected.plus(
                rotations=int(step.expected_cost.rotations),
                ct_pt_mults=int(step.expected_cost.ct_pt_mults),
                adds=int(step.expected_cost.adds),
            )

    output_regions = tuple(
        TensorRegion(c_start=0, c_end=int(spec.c), h_start=0, h_end=int(spec.h), w_start=0, w_end=int(spec.w))
        for _ in selected_output_blocks
    )
    output_active_counts = tuple(
        int(min(int(slot_count), max(0, int(output_length) - int(block_index) * int(slot_count))))
        for block_index in selected_output_blocks
    )
    plan = ConvSchemePlan(
        case_name=f"orion_vendored_r34_{spec.stage}_same_orion_blocks",
        ring_slot_count=int(slot_count),
        output_regions=tuple(output_regions),
        output_active_slot_counts=tuple(output_active_counts),
        family_templates=(
            FamilyTemplateBank(
                family_id=str(family_id),
                family_key=("resnet34_imagenet", str(spec.family_label), str(spec.policy)),
                source_tile_shape=(int(spec.c), int(spec.h), int(spec.w)),
                target_tile_shape=(int(spec.c), int(spec.h), int(spec.w)),
                source_h_range=(0, int(spec.h)),
                target_h_range=(0, int(spec.h)),
                template_entries=tuple(all_templates),
                member_count=int(len(lt_steps)),
                evidence_kind=f"vendored_haloed_{spec.policy}",
                note=f"vectorized same-shape block LT builder compatible with Orion block-chunk source layout; selected by policy={spec.policy}",
            ),
        ),
        prepared_plaintexts=tuple(all_prepared),
        linear_transform_steps=tuple(lt_steps),
        expected_cost=expected,
        evidence_kind=f"vendored_haloed_{spec.policy}",
        notes=(
            f"family_label={spec.family_label}",
            f"policy={spec.policy}",
            f"source_group_count={spec.source_group_count}",
            "same-shape runtime no longer calls pack_conv2d",
            "builder is compatible with Orion's current block-chunk source/output layout",
        ),
    )
    inputs = _split_flat_into_ring_blocks(
        _pack_gap_flat(x, shape=(int(spec.c), int(spec.h), int(spec.w)), gap=int(spec.gap)),
        prefix="orion_source_block",
        slot_count=int(slot_count),
    )
    reference = F.conv2d(x.unsqueeze(0), weight, bias=None, stride=1, padding=1)[0]
    return plan, inputs, reference


def _coalesce_native_rows(keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(keys.numel()) == 0:
        return keys.to(dtype=torch.int64), values.to(dtype=torch.float32)
    unique, inverse = torch.unique(keys.to(dtype=torch.int64), sorted=True, return_inverse=True)
    out = torch.zeros((int(unique.numel()),), dtype=torch.float32)
    out.index_add_(0, inverse.to(dtype=torch.int64), values.to(dtype=torch.float32))
    keep = torch.abs(out) > 0
    return unique[keep], out[keep]


def _pack_native_aligned_halo_tile(
    source: torch.Tensor,
    *,
    spec: R34SameShapeStageSpec,
    stripe: R34NativeAlignedHaloStripe,
    channel_start: int,
    channel_end: int,
) -> torch.Tensor:
    local_c = int(channel_end) - int(channel_start)
    out = torch.zeros((int(_spec_slot_count(spec)),), dtype=torch.float32)
    for local_c_index, channel in enumerate(range(int(channel_start), int(channel_end))):
        for global_h in range(int(stripe.source_h_start), int(stripe.source_h_end)):
            local_h = int(global_h) - int(stripe.source_h_start)
            for w_index in range(int(spec.w)):
                slot = _idx_chw_gap(
                    int(local_c_index),
                    int(local_h),
                    int(w_index),
                    int(stripe.local_h),
                    int(spec.w),
                    int(spec.gap),
                )
                out[int(slot)] = source[int(channel), int(global_h), int(w_index)]
    return out


def _build_native_aligned_halo_lt_assets(
    *,
    spec: R34SameShapeStageSpec,
    native_plan: R34NativeAlignedHaloPlan,
    weight: torch.Tensor,
    family_id: str,
    stripe: R34NativeAlignedHaloStripe,
    source_group_index: int,
    target_group_index: int,
    source_channel_start: int,
    source_channel_end: int,
    target_channel_start: int,
    target_channel_end: int,
    group_index: int,
    group_n1: int,
    group_shared_rotations: int,
    group_baby_rotations: int,
    group_giant_rotations: int,
    rotation_cost_owner: bool,
) -> tuple[tuple[CanonicalTemplateEntry, ...], tuple[PreparedPlaintext, ...], LinearTransformStep | None]:
    slots = int(_spec_slot_count(spec))
    source_channel_count = int(source_channel_end) - int(source_channel_start)
    target_channel_count = int(target_channel_end) - int(target_channel_start)
    source_slots = _native_slot_indices(
        int(source_channel_count),
        int(stripe.local_h),
        int(spec.w),
        int(spec.gap),
    )
    target_slots = _native_slot_indices(
        int(target_channel_count),
        int(stripe.local_h),
        int(spec.w),
        int(spec.gap),
    )
    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    for kh in range(3):
        for kw in range(3):
            coeff = weight[
                int(target_channel_start): int(target_channel_end),
                int(source_channel_start): int(source_channel_end),
                int(kh),
                int(kw),
            ].to(dtype=torch.float32)
            if not bool(torch.any(coeff != 0).item()):
                continue
            coeff_by_source_target = coeff.t().contiguous()
            for out_h in range(int(stripe.target_h_start), int(stripe.target_h_end)):
                in_h = int(out_h) - 1 + int(kh)
                if int(in_h) < 0 or int(in_h) >= int(spec.h):
                    continue
                source_local_h = int(in_h) - int(stripe.source_h_start)
                target_local_h = int(out_h) - int(stripe.source_h_start)
                if (
                    int(source_local_h) < 0
                    or int(source_local_h) >= int(stripe.local_h)
                    or int(target_local_h) < 0
                    or int(target_local_h) >= int(stripe.local_h)
                ):
                    continue
                for out_w in range(int(spec.w)):
                    in_w = int(out_w) - 1 + int(kw)
                    if int(in_w) < 0 or int(in_w) >= int(spec.w):
                        continue
                    source_vec = source_slots[:, int(source_local_h), int(in_w)]
                    target_vec = target_slots[:, int(target_local_h), int(out_w)]
                    diag_index = (source_vec[:, None] - target_vec[None, :]).remainder(int(slots))
                    shift = (-diag_index).remainder(int(slots))
                    output_slot = target_vec[None, :].expand_as(shift)
                    key_parts.append((shift.reshape(-1) * int(slots) + output_slot.reshape(-1)).to(dtype=torch.int64))
                    value_parts.append(coeff_by_source_target.reshape(-1).to(dtype=torch.float32))
    if not key_parts:
        return (), (), None
    keys, values = _coalesce_native_rows(torch.cat(key_parts), torch.cat(value_parts))
    if int(keys.numel()) == 0:
        return (), (), None
    shifts = torch.div(keys, int(slots), rounding_mode="floor").to(dtype=torch.int64)
    output_slots = torch.remainder(keys, int(slots)).to(dtype=torch.int64)
    template_entries: list[CanonicalTemplateEntry] = []
    prepared_plaintexts: list[PreparedPlaintext] = []
    terms: list[LinearTransformTerm] = []
    unique_shifts, counts = torch.unique_consecutive(shifts, return_counts=True)
    start = 0
    target_index = int(stripe.index) * int(native_plan.target_channel_group_count) + int(target_group_index)
    source_index = int(stripe.index) * int(native_plan.source_channel_group_count) + int(source_group_index)
    for term_index, (shift_tensor, count_tensor) in enumerate(zip(unique_shifts.tolist(), counts.tolist())):
        end = int(start) + int(count_tensor)
        shift = int(shift_tensor)
        template_id = (
            f"native_aligned_halo_{spec.stage}_s{int(stripe.index)}_"
            f"src{int(source_group_index)}_tgt{int(target_group_index)}_template_{int(term_index)}"
        )
        plaintext_id = (
            f"native_aligned_halo_{spec.stage}_s{int(stripe.index)}_"
            f"src{int(source_group_index)}_tgt{int(target_group_index)}_pt_{int(term_index)}"
        )
        term_output_slots = output_slots[int(start): int(end)].to(dtype=torch.int64)
        template_entries.append(
            CanonicalTemplateEntry(
                template_id=str(template_id),
                family_id=str(family_id),
                key=(int(target_index), int(source_index), int(shift)),
                fine_shift=int(shift),
                indices=term_output_slots,
                note="native aligned halo no-RI template",
            )
        )
        prepared_plaintexts.append(
            PreparedPlaintext(
                plaintext_id=str(plaintext_id),
                template_id=str(template_id),
                level=0,
                scale=1.0,
                slot_count=int(slots),
                values=values[int(start): int(end)].to(dtype=torch.float32),
                note="native aligned halo no-RI prepared plaintext",
            )
        )
        terms.append(
            LinearTransformTerm(
                term_id=(
                    f"native_aligned_halo_{spec.stage}_s{int(stripe.index)}_"
                    f"src{int(source_group_index)}_tgt{int(target_group_index)}_term_{int(term_index)}"
                ),
                shift=int(shift),
                plaintext_id=str(plaintext_id),
                template_id=str(template_id),
                lookup_indices=torch.arange(int(term_output_slots.numel()), dtype=torch.int64),
                output_slot_indices=term_output_slots,
                note="native aligned halo no-RI term",
            )
        )
        start = int(end)
    diag_indices = {int((-int(term.shift)) % int(slots)) for term in terms}
    baby, giant = _bsgs_rotation_sets(diag_indices, slots=int(slots), n1=int(group_n1))
    expected_rotations = int(group_shared_rotations) if bool(rotation_cost_owner) else 0
    step = LinearTransformStep(
        step_id=(
            f"native_aligned_halo_{spec.stage}_s{int(stripe.index)}_"
            f"src{int(source_group_index)}_tgt{int(target_group_index)}"
        ),
        input_id=f"native_source_tile_{int(source_index)}",
        target_index=int(target_index),
        selected_n1=int(group_n1),
        baby_shifts=tuple(sorted(int(value) for value in baby)),
        giant_shifts=tuple(sorted(int(value) for value in giant)),
        terms=tuple(terms),
        required_rotations=tuple(sorted(set(int(value) for value in baby).union(int(value) for value in giant))),
        prepared_plaintext_ids=tuple(str(pt.plaintext_id) for pt in prepared_plaintexts),
        expected_cost=ExecutionStats(
            rotations=int(expected_rotations),
            ct_pt_mults=int(len(terms)),
            adds=int(len(terms)),
        ),
        representation="native_aligned_halo_no_ri",
        note="native aligned halo no-RI local Toeplitz transform",
        rotation_group_id=f"native_aligned_halo:{spec.family_label}:group_{int(group_index)}",
        rotation_cost_owner=bool(rotation_cost_owner),
        shared_multi_output=False,
    )
    return tuple(template_entries), tuple(prepared_plaintexts), step


def build_r34_native_aligned_halo_no_ri_plan(
    *,
    spec: R34SameShapeStageSpec,
    weight_override: torch.Tensor | None = None,
    bias_override: torch.Tensor | None = None,
    source_override: torch.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    output_shape: tuple[int, int, int] | None = None,
    input_gap: int | None = None,
    output_gap: int | None = None,
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    torch.manual_seed(0)
    expected_weight_shape = tuple(int(v) for v in spec.weight_shape)
    if weight_override is None:
        weight = torch.randn(expected_weight_shape, dtype=torch.float32)
    else:
        weight = weight_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in weight.shape) != expected_weight_shape:
            raise ValueError(
                f"{spec.family_label} fused weight shape mismatch: expected {expected_weight_shape}, got {tuple(weight.shape)}"
            )
    expected_source_shape = (int(spec.c), int(spec.h), int(spec.w))
    if source_override is None:
        x = torch.randn(expected_source_shape, dtype=torch.float32)
    else:
        x = source_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in x.shape) != expected_source_shape:
            raise ValueError(
                f"{spec.family_label} source shape mismatch: expected {expected_source_shape}, got {tuple(x.shape)}"
            )
    if input_shape is not None and tuple(int(v) for v in input_shape) != expected_source_shape:
        raise ValueError(f"{spec.family_label} input_shape mismatch: {input_shape}")
    if output_shape is not None and tuple(int(v) for v in output_shape) != expected_source_shape:
        raise ValueError(f"{spec.family_label} output_shape mismatch: {output_shape}")
    if input_gap is not None and int(input_gap) != int(spec.gap):
        raise ValueError(f"{spec.family_label} input_gap mismatch: {input_gap}")
    if output_gap is not None and int(output_gap) != int(spec.gap):
        raise ValueError(f"{spec.family_label} output_gap mismatch: {output_gap}")
    if bias_override is not None:
        bias = bias_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in bias.shape) != (int(spec.c),):
            raise ValueError(f"{spec.family_label} bias shape mismatch: expected {(int(spec.c),)}, got {tuple(bias.shape)}")

    native_plan = r34_native_aligned_halo_plan(spec)
    family_id = f"native_aligned_halo_{spec.stage}_no_ri_family"
    all_templates: list[CanonicalTemplateEntry] = []
    all_prepared: list[PreparedPlaintext] = []
    lt_steps: list[LinearTransformStep] = []
    expected = ExecutionStats()
    group_index = 0
    for stripe in native_plan.stripes:
        for source_group in range(int(native_plan.source_channel_group_count)):
            source_start = int(source_group) * int(native_plan.channel_tile)
            source_end = min(int(spec.c), int(source_start) + int(native_plan.channel_tile))
            for target_group in range(int(native_plan.target_channel_group_count)):
                target_start = int(target_group) * int(native_plan.channel_tile)
                target_end = min(int(spec.c), int(target_start) + int(native_plan.channel_tile))
                templates, prepared, step = _build_native_aligned_halo_lt_assets(
                    spec=spec,
                    native_plan=native_plan,
                    weight=weight,
                    family_id=str(family_id),
                    stripe=stripe,
                    source_group_index=int(source_group),
                    target_group_index=int(target_group),
                    source_channel_start=int(source_start),
                    source_channel_end=int(source_end),
                    target_channel_start=int(target_start),
                    target_channel_end=int(target_end),
                    group_index=int(group_index),
                    group_n1=int(native_plan.group_n1s[int(group_index)]),
                    group_shared_rotations=int(native_plan.group_shared_rotations[int(group_index)]),
                    group_baby_rotations=int(native_plan.group_baby_rotations[int(group_index)]),
                    group_giant_rotations=int(native_plan.group_giant_rotations[int(group_index)]),
                    rotation_cost_owner=bool(int(target_group) == 0),
                )
                all_templates.extend(templates)
                all_prepared.extend(prepared)
                if step is None:
                    continue
                lt_steps.append(step)
                expected = expected.plus(
                    rotations=int(step.expected_cost.rotations),
                    ct_pt_mults=int(step.expected_cost.ct_pt_mults),
                    adds=int(step.expected_cost.adds),
                )
            group_index += 1

    output_regions: list[TensorRegion] = []
    output_active_counts: list[int] = []
    for stripe in native_plan.stripes:
        for target_group in range(int(native_plan.target_channel_group_count)):
            target_start = int(target_group) * int(native_plan.channel_tile)
            target_end = min(int(spec.c), int(target_start) + int(native_plan.channel_tile))
            output_regions.append(
                TensorRegion(
                    c_start=int(target_start),
                    c_end=int(target_end),
                    h_start=int(stripe.target_h_start),
                    h_end=int(stripe.target_h_end),
                    w_start=0,
                    w_end=int(spec.w),
                )
            )
            output_active_counts.append(int(stripe.active_target_slots))

    inputs: dict[str, PlainCipherTensor] = {}
    for stripe in native_plan.stripes:
        for source_group in range(int(native_plan.source_channel_group_count)):
            source_index = int(stripe.index) * int(native_plan.source_channel_group_count) + int(source_group)
            source_start = int(source_group) * int(native_plan.channel_tile)
            source_end = min(int(spec.c), int(source_start) + int(native_plan.channel_tile))
            input_id = f"native_source_tile_{int(source_index)}"
            inputs[input_id] = PlainCipherTensor(
                _pack_native_aligned_halo_tile(
                    x,
                    spec=spec,
                    stripe=stripe,
                    channel_start=int(source_start),
                    channel_end=int(source_end),
                ),
                label=str(input_id),
            )

    plan = ConvSchemePlan(
        case_name=f"native_aligned_halo_{spec.stage}_same_no_ri",
        ring_slot_count=int(_spec_slot_count(spec)),
        output_regions=tuple(output_regions),
        output_active_slot_counts=tuple(output_active_counts),
        family_templates=(
            FamilyTemplateBank(
                family_id=str(family_id),
                family_key=("resnet34_imagenet", str(spec.family_label), "native_aligned_halo_no_ri"),
                source_tile_shape=(int(native_plan.channel_tile), int(max(stripe.local_h for stripe in native_plan.stripes)), int(spec.w)),
                target_tile_shape=(int(native_plan.channel_tile), int(max(stripe.local_h for stripe in native_plan.stripes)), int(spec.w)),
                source_h_range=(0, 0),
                target_h_range=(0, 0),
                template_entries=tuple(all_templates),
                member_count=int(len(lt_steps)),
                evidence_kind="native_aligned_halo_no_ri",
                note="native aligned halo local Toeplitz transforms with no real/imag packing",
            ),
        ),
        prepared_plaintexts=tuple(all_prepared),
        linear_transform_steps=tuple(lt_steps),
        expected_cost=expected,
        evidence_kind="native_aligned_halo_no_ri",
        notes=(
            f"family_label={spec.family_label}",
            "native aligned halo source/output tile layout",
            "channel_tile=min(C,gap^2); height stripes are generated from ring capacity",
            "real/imag packing disabled",
        ),
    )
    reference = F.conv2d(x.unsqueeze(0), weight, bias=None, stride=1, padding=1)[0]
    return plan, inputs, reference


def build_r34_stage1_same_orion_plan(**kwargs: Any) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    return build_r34_same_shape_orion_plan(spec=R34_STAGE1_SAME_SPEC, **kwargs)


def build_r34_stage2_same_orion_plan(**kwargs: Any) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    return build_r34_same_shape_orion_plan(spec=R34_STAGE2_SAME_SPEC, **kwargs)


def build_r34_stage3_same_orion_plan(**kwargs: Any) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    return build_r34_same_shape_orion_plan(spec=R34_STAGE3_SAME_SPEC, **kwargs)


def build_r34_stage4_same_orion_plan(**kwargs: Any) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    return build_r34_same_shape_orion_plan(spec=R34_STAGE4_SAME_SPEC, **kwargs)


def build_r34_same_shape_inter_group_hybrid_plan(
    *, spec: R34SameShapeStageSpec, **kwargs: Any
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    if str(spec.policy) != "inter_group_hybrid":
        raise ValueError(f"{spec.family_label} is not an inter_group_hybrid family")
    return build_r34_same_shape_orion_plan(spec=spec, **kwargs)


def build_r34_same_shape_intra_group_pack2_plan(
    *, spec: R34SameShapeStageSpec, **kwargs: Any
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    if str(spec.policy) != "intra_group_pack2":
        raise ValueError(f"{spec.family_label} is not an intra_group_pack2 family")
    return build_r34_same_shape_orion_plan(spec=spec, **kwargs)


def _parse_block_index(input_id: str) -> int:
    try:
        return int(str(input_id).rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"cannot parse Orion source block index from {input_id!r}") from exc


def _transform_from_plan_step(
    *,
    plan: ConvSchemePlan,
    step: LinearTransformStep,
    level: int,
    scheme: Any,
    name: str,
    prepared_by_id: dict[str, PreparedPlaintext] | None = None,
    template_by_id: dict[str, CanonicalTemplateEntry] | None = None,
) -> Any:
    prepared = (
        prepared_by_id
        if prepared_by_id is not None
        else {str(plain.plaintext_id): plain for plain in plan.prepared_plaintexts}
    )
    templates = (
        template_by_id
        if template_by_id is not None
        else {str(entry.template_id): entry for family in plan.family_templates for entry in family.template_entries}
    )
    slots = int(plan.ring_slot_count)
    diag_tensors: dict[int, torch.Tensor] = {}
    for term in step.terms:
        template = templates[str(term.template_id)]
        plaintext = prepared[str(term.plaintext_id)]
        mapped_indices = template.indices.to(dtype=torch.int64).index_select(0, term.lookup_indices.to(dtype=torch.int64))
        output_indices = term.output_slot_indices.to(dtype=torch.int64)
        if not bool(torch.equal(mapped_indices, output_indices)):
            raise ValueError(f"term {term.term_id} cannot be encoded as one dense Orion diagonal")
        values = plaintext.values
        value_dtype = torch.complex64 if torch.is_complex(values) else torch.float32
        diag_index = (-int(term.shift)) % int(slots)
        diag = diag_tensors.setdefault(int(diag_index), torch.zeros((int(slots),), dtype=value_dtype))
        diag.index_add_(0, output_indices, values.to(dtype=value_dtype))
    payload: dict[str, Any] = {
        "name": str(name),
        "diagonals": {(0, 0): {int(index): diag for index, diag in sorted(diag_tensors.items())}},
        "level": int(level),
        "scheme": scheme,
        "fhe_output_shape": torch.Size([1, int(slots)]),
        "output_shape": torch.Size([1, int(slots)]),
        "target_index": int(step.target_index),
        "input_id": str(step.input_id),
    }
    if int(step.selected_n1) > 0:
        payload.update(
            {
                "selected_n1": int(step.selected_n1),
                "baby_shifts": tuple(int(value) for value in step.baby_shifts),
                "giant_shifts": tuple(int(value) for value in step.giant_shifts),
                "rotation_group_id": str(step.rotation_group_id),
                "rotation_cost_owner": bool(step.rotation_cost_owner),
            }
        )
    return SimpleNamespace(**payload)


def _cached_transform_shell(*, level: int, scheme: Any) -> Any:
    return SimpleNamespace(diagonals={}, level=int(level), scheme=scheme)


class R34OrionSameShapeRuntimeExecutor:
    def __init__(self, *, module: Any, spec: R34SameShapeStageSpec, output_node_id: str) -> None:
        self.module = module
        self.spec = spec
        self.output_node_id = str(output_node_id)
        self.plan: ConvSchemePlan | None = None
        self.slots = int(_spec_slot_count(spec))
        self.groups_by_input_index: dict[int, Any] = {}
        self.target_indices_by_input_index: dict[int, tuple[int, ...]] = {}
        self.cols = 0
        self.rows = 0
        self.output_shape = getattr(module, "output_shape", None)
        self.fhe_output_shape = getattr(module, "fhe_output_shape", None)
        self.bias_vector: torch.Tensor | None = None
        self.bias_plaintexts: tuple[Any | None, ...] = ()
        self._bias_plaintext_cache: dict[tuple[int, int], Any] = {}
        self.compile_count = 0
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.last_runtime_timing: dict[str, float] = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self._compile_cache_metadata: dict[str, Any] = {}

    def hardcoded_halo_relayout_plan(self) -> R34SameShapeHaloRelayoutPlan:
        return r34_same_shape_hardcoded_relayout_plan(self.spec)

    def _hardcoded_halo_effective_conv_lt_tasks(self, plan: R34SameShapeHaloRelayoutPlan) -> int:
        return int(plan.effective_conv_lt_tasks_without_hybrid)

    def _compiled_generic_conv_lt_task_count(self) -> int:
        chunk_targets = getattr(self, "target_indices_by_input_chunk", None)
        if chunk_targets:
            return int(sum(len(tuple(values)) for values in chunk_targets))
        block_targets = getattr(self, "target_indices_by_input_block", None)
        if block_targets:
            return int(sum(len(tuple(values)) for values in block_targets))
        index_targets = getattr(self, "target_indices_by_input_index", None)
        if isinstance(index_targets, dict) and index_targets:
            return int(sum(len(tuple(values)) for values in index_targets.values()))
        return int(self.rows) * int(self.cols)

    def _hardcoded_halo_relayout_metadata(self) -> dict[str, Any]:
        family_label = str(self.spec.family_label)
        if family_label not in _R34_SAME_SHAPE_HARDCODED_STRIPE_RANGES:
            task_count = int(self._compiled_generic_conv_lt_task_count())
            return {
                "same_shape_runtime_layout": "generic_same_shape_no_hardcoded_halo_relayout",
                "r34_same_shape_halo_relayout_plan": None,
                "conv_lt_raw_submatrix_tasks": int(task_count),
                "conv_lt_effective_submatrix_tasks": int(task_count),
                "legacy_flat_conv_lt_tasks": 0,
                "legacy_flat_offdiag_tasks": 0,
                "relayout_rotation_count": 0,
                "relayout_mask_mult_count": 0,
            }
        plan = self.hardcoded_halo_relayout_plan()
        return {
            "same_shape_runtime_layout": "height_stripe_halo_local_hardcoded",
            "r34_same_shape_halo_relayout_plan": plan.to_dict(),
            "conv_lt_raw_submatrix_tasks": int(plan.raw_conv_lt_tasks),
            "conv_lt_effective_submatrix_tasks": int(self._hardcoded_halo_effective_conv_lt_tasks(plan)),
            "legacy_flat_conv_lt_tasks": int(plan.legacy_flat_conv_lt_tasks),
            "legacy_flat_offdiag_tasks": int(plan.legacy_flat_offdiag_tasks),
            "relayout_rotation_count": int(plan.relayout_rotations),
            "relayout_mask_mult_count": int(plan.relayout_mask_mults),
        }

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def _output_level(self, scheme: Any, *, extra_depth: int = 0) -> int:
        depth = int(self.assigned_depth) if self.assigned_depth is not None else 1
        return max(0, int(self._level(scheme)) - max(0, int(depth)) - max(0, int(extra_depth)))

    def _validate_module(self) -> None:
        weight = getattr(self.module, "on_weight", None)
        if weight is None:
            raise RuntimeError(f"{self.output_node_id} has no fused Orion weight for R34 same-shape runtime")
        if tuple(int(v) for v in tuple(weight.shape)) != tuple(int(v) for v in self.spec.weight_shape):
            raise RuntimeError(f"{self.output_node_id} weight shape does not match {self.spec.family_label}")
        if tuple(int(v) for v in getattr(self.module, "stride", ())) != (1, 1):
            raise RuntimeError(f"{self.output_node_id} stride does not match {self.spec.family_label}")
        if tuple(int(v) for v in getattr(self.module, "padding", ())) != (1, 1):
            raise RuntimeError(f"{self.output_node_id} padding does not match {self.spec.family_label}")
        if tuple(int(v) for v in getattr(self.module, "input_shape", torch.Size())[1:]) != (int(self.spec.c), int(self.spec.h), int(self.spec.w)):
            raise RuntimeError(f"{self.output_node_id} input_shape does not match {self.spec.family_label}")
        if tuple(int(v) for v in getattr(self.module, "output_shape", torch.Size())[1:]) != (int(self.spec.c), int(self.spec.h), int(self.spec.w)):
            raise RuntimeError(f"{self.output_node_id} output_shape does not match {self.spec.family_label}")
        if int(getattr(self.module, "input_gap", -1)) != int(self.spec.gap):
            raise RuntimeError(f"{self.output_node_id} input_gap does not match {self.spec.family_label}")
        if int(getattr(self.module, "output_gap", -1)) != int(self.spec.gap):
            raise RuntimeError(f"{self.output_node_id} output_gap does not match {self.spec.family_label}")

    def _ensure_plan(self) -> ConvSchemePlan:
        if self.plan is None:
            self._validate_module()
            builder = (
                build_r34_same_shape_inter_group_hybrid_plan
                if str(self.spec.policy) == "inter_group_hybrid"
                else build_r34_same_shape_intra_group_pack2_plan
            )
            self.plan, _inputs, _reference = builder(
                spec=self.spec,
                weight_override=getattr(self.module, "on_weight"),
                bias_override=getattr(self.module, "on_bias", None),
                input_shape=(int(self.spec.c), int(self.spec.h), int(self.spec.w)),
                output_shape=(int(self.spec.c), int(self.spec.h), int(self.spec.w)),
                input_gap=int(self.spec.gap),
                output_gap=int(self.spec.gap),
            )
        return self.plan

    def _bias_chunk(self, *, block_index: int) -> torch.Tensor | None:
        if self.bias_vector is None:
            return None
        start = int(block_index) * int(self.slots)
        end = min(int(start + int(self.slots)), int(self.bias_vector.numel()))
        out = torch.zeros((int(self.slots),), dtype=torch.float32)
        if end > start:
            out[: int(end - start)] = self.bias_vector[int(start): int(end)]
        return out

    def _add_bias(self, ct: Any, *, block_index: int) -> Any:
        bias_pt = self.bias_plaintexts[int(block_index)] if int(block_index) < len(self.bias_plaintexts) else None
        if bias_pt is None or int(bias_pt.level()) != int(ct.level()):
            bias_pt = self._bias_plaintext_cache.get((int(block_index), int(ct.level())))
        if bias_pt is None or int(bias_pt.level()) != int(ct.level()):
            chunk = self._bias_chunk(block_index=int(block_index))
            if chunk is None:
                return ct
            bias_pt = _encode_plaintext_for_add(ct, chunk)
            self._bias_plaintext_cache[(int(block_index), int(ct.level()))] = bias_pt
        return _add_plaintext_for_add(ct, bias_pt)

    def _compile_bias_plaintexts(self, scheme: Any, *, extra_depth: int = 0) -> tuple[Any | None, ...]:
        if self.bias_vector is None:
            return ()
        level = self._output_level(scheme, extra_depth=int(extra_depth))
        scale = int(scheme.params.get_default_scale())
        plaintexts: list[Any | None] = []
        for block_index in range(int(self.rows)):
            chunk = self._bias_chunk(block_index=int(block_index))
            ptxt = None if chunk is None else scheme.encode(chunk, level=int(level), scale=int(scale))
            if ptxt is not None:
                self._bias_plaintext_cache[(int(block_index), int(level))] = ptxt
            plaintexts.append(ptxt)
        return tuple(plaintexts)

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        self._compile_cache_metadata = dict(metadata or {})

    def compile_cache_metadata(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "rows": int(self.rows),
            "cols": int(self.cols),
            **self._hardcoded_halo_relayout_metadata(),
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

    def _compile_from_cache_metadata(self, scheme: Any) -> bool:
        metadata = dict(self._compile_cache_metadata or {})
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "load" or not metadata:
            return False
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.rows = int(metadata.get("rows", 0))
        self.cols = int(metadata.get("cols", 0))
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme, extra_depth=0)
        level = int(self._level(scheme))
        compile_started = time.time()
        for group_meta in list(metadata.get("groups_by_input_index", [])):
            input_index = int(group_meta.get("input_index", 0))
            target_indices = tuple(int(value) for value in group_meta.get("target_indices", []))
            group = UnifiedTransformGroup(
                [_cached_transform_shell(level=int(level), scheme=scheme) for _target in target_indices]
            )
            group._storage_key = str(group_meta["storage_key"])
            group.compile_unified(scheme.backend)
            self.groups_by_input_index[int(input_index)] = group
            self.target_indices_by_input_index[int(input_index)] = target_indices
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)
        return True

    def compile(self, scheme: Any) -> None:
        if self.groups_by_input_index:
            return
        if self._compile_from_cache_metadata(scheme):
            return
        prepare_plans_started = time.time()
        plan = self._ensure_plan()
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.last_runtime_timing["prepare_plans_s"] = float(time.time() - prepare_plans_started)
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        self.rows = int(len(plan.output_regions))
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme, extra_depth=0)

        level = self._level(scheme)
        prepare_transforms_started = time.time()
        steps_by_input_index: dict[int, list[Any]] = {}
        for step in plan.linear_transform_steps:
            input_index = int(_parse_block_index(step.input_id))
            steps_by_input_index.setdefault(int(input_index), []).append(step)
        self.cols = int(max(steps_by_input_index.keys(), default=-1) + 1)
        self.last_runtime_timing["prepare_transforms_s"] = float(time.time() - prepare_transforms_started)

        compile_started = time.time()
        prepared_by_id = {str(plain.plaintext_id): plain for plain in plan.prepared_plaintexts}
        template_by_id = {
            str(entry.template_id): entry
            for family in plan.family_templates
            for entry in family.template_entries
        }
        for input_index, steps in sorted(steps_by_input_index.items()):
            ordered = [
                (
                    int(step.target_index),
                    _transform_from_plan_step(
                        plan=plan,
                        step=step,
                        level=int(level),
                        scheme=scheme,
                        name=f"{self.output_node_id}_{step.input_id}_t{int(step.target_index)}",
                        prepared_by_id=prepared_by_id,
                        template_by_id=template_by_id,
                    ),
                )
                for step in sorted(steps, key=lambda value: int(value.target_index))
            ]
            group = UnifiedTransformGroup([transform for _target_index, transform in ordered])
            group.compile_unified(scheme.backend)
            self.groups_by_input_index[int(input_index)] = group
            self.target_indices_by_input_index[int(input_index)] = tuple(int(target_index) for target_index, _transform in ordered)
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.compile(scheme)
        ids = tuple(int(value) for value in getattr(source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            raise RuntimeError(f"{self.output_node_id} requires {self.cols} source ciphertext blocks, got {len(ids)}")

        output_blocks: list[Any | None] = [None for _ in range(int(self.rows))]
        fuse_output_rescale = bool(_unified_output_fusion_enabled())
        evaluate_started = time.time()
        for input_index, group in sorted(self.groups_by_input_index.items()):
            if int(input_index) >= len(ids):
                continue
            output_ids = group.evaluate_unified(int(ids[int(input_index)]), scheme.backend)
            for target_index, output_id in zip(self.target_indices_by_input_index[int(input_index)], output_ids):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(self.slots)]),
                    torch.Size([1, int(self.slots)]),
                )
                if not fuse_output_rescale:
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
                raise RuntimeError(f"missing same-shape output block {block_index} for {self.output_node_id}")
            if fuse_output_rescale:
                block_ct = _rescale_cipher_tensor(block_ct)
            block_ct = self._add_bias(block_ct, block_index=int(block_index))
            block_ct.set_scale(int(scheme.params.get_default_scale()))
            output_ids.append(int(block_ct.ids[0]))
            block_ct.ids = []
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {
            self.output_node_id: CipherTensor(
                scheme,
                output_ids,
                self.output_shape,
                self.fhe_output_shape,
            )
        }


class NativeAlignedHaloNoRIConvExecutor(R34OrionSameShapeRuntimeExecutor):
    """Native halo-stripe same-shape Conv2d executor with no real/imag packing."""

    kernel_kind = "native_aligned_halo_no_ri_conv2d"
    use_ct_pt_hybrid_packing = False
    native_halo_input_capable = True
    native_halo_output_capable = False

    def __init__(self, *, module: Any, spec: R34SameShapeStageSpec, output_node_id: str) -> None:
        super().__init__(module=module, spec=spec, output_node_id=output_node_id)
        self.native_plan = r34_native_aligned_halo_plan(spec)
        self.rows = int(self.native_plan.output_ct_count)
        self.cols = int(self.native_plan.input_ct_count)
        self.input_relayout_kernel: R34NativeAlignedRelayoutKernel | None = None
        self.output_relayout_kernel: R34NativeAlignedRelayoutKernel | None = None

    def runtime_fhe_output_shape(self) -> torch.Size:
        return torch.Size(getattr(self.module, "fhe_output_shape", self.fhe_output_shape))

    def runtime_native_fhe_output_shape(self) -> torch.Size:
        return torch.Size([int(self.rows), int(self.slots)])

    def native_aligned_halo_plan_metadata(self) -> dict[str, Any]:
        return self.native_plan.to_dict()

    def _compact_input_ct_count(self) -> int:
        return int(_ceil_div(int(torch.Size(getattr(self.module, "fhe_input_shape")).numel()), int(self.slots)))

    def _compact_output_ct_count(self) -> int:
        return int(_ceil_div(int(torch.Size(getattr(self.module, "fhe_output_shape")).numel()), int(self.slots)))

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

    def _compile_from_cache_metadata(self, scheme: Any) -> bool:
        metadata = dict(self._compile_cache_metadata or {})
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "load":
            return False
        if not metadata:
            return False
        if "input_relayout" not in metadata or "output_relayout" not in metadata:
            raise RuntimeError(
                f"Cached R34 native same-shape manifest for {self.output_node_id!r} is missing "
                "compact/native relayout metadata; re-run with io_mode='save'."
            )
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
            "input_relayout_s": 0.0,
            "output_relayout_s": 0.0,
        }
        self.rows = int(metadata.get("rows", self.native_plan.output_ct_count))
        self.cols = int(metadata.get("cols", self.native_plan.input_ct_count))
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)

        input_level = int(self._level(scheme))
        conv_level = int(input_level)
        conv_output_level = max(0, int(input_level - 1))
        self.bias_plaintexts = self._compile_bias_plaintexts_at_level(scheme, level=int(conv_output_level))

        compile_started = time.time()
        self.input_relayout_kernel = R34NativeAlignedRelayoutKernel(
            spec=self.spec,
            native_plan=self.native_plan,
            direction="compact_to_native",
            name=f"{self.output_node_id}_compact_to_native_halo",
            output_shape=torch.Size([int(self.native_plan.input_ct_count), int(self.slots)]),
            fhe_output_shape=torch.Size([int(self.native_plan.input_ct_count), int(self.slots)]),
        )
        self.input_relayout_kernel.compile_from_cache_metadata(
            scheme,
            dict(metadata.get("input_relayout") or {}),
            level=int(input_level),
        )

        self.groups_by_input_index = {}
        self.target_indices_by_input_index = {}
        for group_meta in list(metadata.get("groups_by_input_index", [])):
            input_index = int(group_meta.get("input_index", 0))
            target_indices = tuple(int(value) for value in group_meta.get("target_indices", []))
            if not target_indices:
                continue
            group = UnifiedTransformGroup(
                [_cached_transform_shell(level=int(conv_level), scheme=scheme) for _target in target_indices]
            )
            group._storage_key = str(group_meta["storage_key"])
            group.compile_unified(scheme.backend)
            self.groups_by_input_index[int(input_index)] = group
            self.target_indices_by_input_index[int(input_index)] = target_indices

        self.output_relayout_kernel = R34NativeAlignedRelayoutKernel(
            spec=self.spec,
            native_plan=self.native_plan,
            direction="native_to_compact",
            name=f"{self.output_node_id}_native_halo_to_compact",
            output_shape=torch.Size(getattr(self.module, "output_shape")),
            fhe_output_shape=torch.Size(getattr(self.module, "fhe_output_shape")),
        )
        self.output_relayout_kernel.compile_from_cache_metadata(
            scheme,
            dict(metadata.get("output_relayout") or {}),
            level=int(conv_output_level),
        )
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)
        return True

    def _ensure_plan(self) -> ConvSchemePlan:
        if self.plan is None:
            self._validate_module()
            self.plan, _inputs, _reference = build_r34_native_aligned_halo_no_ri_plan(
                spec=self.spec,
                weight_override=getattr(self.module, "on_weight"),
                bias_override=getattr(self.module, "on_bias", None),
                input_shape=(int(self.spec.c), int(self.spec.h), int(self.spec.w)),
                output_shape=(int(self.spec.c), int(self.spec.h), int(self.spec.w)),
                input_gap=int(self.spec.gap),
                output_gap=int(self.spec.gap),
            )
        return self.plan

    def _bias_chunk(self, *, block_index: int) -> torch.Tensor | None:
        if self.bias_vector is None:
            return None
        target_group_count = int(self.native_plan.target_channel_group_count)
        stripe_index = int(block_index) // int(target_group_count)
        target_group = int(block_index) % int(target_group_count)
        if int(stripe_index) >= len(self.native_plan.stripes):
            return None
        stripe = self.native_plan.stripes[int(stripe_index)]
        channel_start = int(target_group) * int(self.native_plan.channel_tile)
        channel_end = min(int(self.spec.c), int(channel_start) + int(self.native_plan.channel_tile))
        out = torch.zeros((int(self.slots),), dtype=torch.float32)
        for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
            bias_value = float(self.bias_vector[int(channel)])
            if bias_value == 0.0:
                continue
            for out_h in range(int(stripe.target_h_start), int(stripe.target_h_end)):
                local_h = int(out_h) - int(stripe.source_h_start)
                if int(local_h) < 0 or int(local_h) >= int(stripe.local_h):
                    continue
                for out_w in range(int(self.spec.w)):
                    slot = _idx_chw_gap(
                        int(local_channel),
                        int(local_h),
                        int(out_w),
                        int(stripe.local_h),
                        int(self.spec.w),
                        int(self.spec.gap),
                    )
                    out[int(slot)] = float(bias_value)
        return out

    def compile_cache_metadata(self) -> dict[str, Any]:
        native_plan = self.native_plan.to_dict()
        input_relayout = (
            self.input_relayout_kernel.to_metadata()
            if self.input_relayout_kernel is not None and self.input_relayout_kernel.transform_ids
            else {
                "direction": "compact_to_native",
                "rows": int(self.native_plan.input_ct_count),
                "cols": int(self._compact_input_ct_count()),
                "lt_tasks": 0,
                "diagonal_count": 0,
            }
        )
        output_relayout = (
            self.output_relayout_kernel.to_metadata()
            if self.output_relayout_kernel is not None and self.output_relayout_kernel.transform_ids
            else {
                "direction": "native_to_compact",
                "rows": int(self._compact_output_ct_count()),
                "cols": int(self.native_plan.output_ct_count),
                "lt_tasks": 0,
                "diagonal_count": 0,
            }
        )
        return {
            "kind": type(self).__name__,
            "kernel_kind": self.kernel_kind,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "use_ct_pt_hybrid_packing": False,
            "native_halo_input_capable": True,
            "native_halo_output_capable": False,
            "same_shape_runtime_layout": "compact_io_native_aligned_halo_no_ri",
            "native_internal_runtime_layout": "native_aligned_halo_no_ri",
            "r34_native_aligned_halo_plan": native_plan,
            "r34_same_shape_halo_relayout_plan": native_plan,
            "conv_lt_raw_submatrix_tasks": int(self.native_plan.submatrix_program_count),
            "conv_lt_effective_submatrix_tasks": int(self.native_plan.sharing_group_count),
            "input_relayout": input_relayout,
            "output_relayout": output_relayout,
            "relayout_sparse_lt_tasks": int(input_relayout.get("lt_tasks", 0)) + int(output_relayout.get("lt_tasks", 0)),
            "native_c_only_rotations": int(self.native_plan.c_only_rotations),
            "native_cb_shared_rotations": int(self.native_plan.cb_shared_rotations),
            "native_shared_baby_rotations": int(self.native_plan.shared_baby_rotations),
            "native_shared_giant_rotations": int(self.native_plan.shared_giant_rotations),
            "legacy_flat_conv_lt_tasks": int(
                _R34_SAME_SHAPE_LEGACY_FLAT_STATS.get(str(self.spec.family_label), {}).get("conv_lt_tasks", 0)
            ),
            "legacy_flat_offdiag_tasks": int(
                _R34_SAME_SHAPE_LEGACY_FLAT_STATS.get(str(self.spec.family_label), {}).get("offdiag_tasks", 0)
            ),
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
        if self.groups_by_input_index and self.input_relayout_kernel is not None and self.output_relayout_kernel is not None:
            return
        if self._compile_from_cache_metadata(scheme):
            return
        prepare_plans_started = time.time()
        plan = self._ensure_plan()
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
            "input_relayout_s": 0.0,
            "output_relayout_s": 0.0,
        }
        self.last_runtime_timing["prepare_plans_s"] = float(time.time() - prepare_plans_started)
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        self.rows = int(len(plan.output_regions))
        self.cols = int(self.native_plan.input_ct_count)

        input_level = int(self._level(scheme))
        conv_level = int(input_level)
        conv_output_level = max(0, int(input_level - 1))
        self.bias_plaintexts = self._compile_bias_plaintexts_at_level(scheme, level=int(conv_output_level))

        compile_started = time.time()
        self.input_relayout_kernel = R34NativeAlignedRelayoutKernel(
            spec=self.spec,
            native_plan=self.native_plan,
            direction="compact_to_native",
            name=f"{self.output_node_id}_compact_to_native_halo",
            output_shape=torch.Size([int(self.native_plan.input_ct_count), int(self.slots)]),
            fhe_output_shape=torch.Size([int(self.native_plan.input_ct_count), int(self.slots)]),
        )
        self.input_relayout_kernel.compile(scheme, level=int(input_level))

        prepared_by_id = {str(plain.plaintext_id): plain for plain in plan.prepared_plaintexts}
        template_by_id = {
            str(entry.template_id): entry
            for family in plan.family_templates
            for entry in family.template_entries
        }
        steps_by_input_index: dict[int, list[Any]] = {}
        for step in plan.linear_transform_steps:
            input_index = int(_parse_block_index(step.input_id))
            steps_by_input_index.setdefault(int(input_index), []).append(step)
        for input_index, steps in sorted(steps_by_input_index.items()):
            ordered = [
                (
                    int(step.target_index),
                    _transform_from_plan_step(
                        plan=plan,
                        step=step,
                        level=int(conv_level),
                        scheme=scheme,
                        name=f"{self.output_node_id}_{step.input_id}_t{int(step.target_index)}",
                        prepared_by_id=prepared_by_id,
                        template_by_id=template_by_id,
                    ),
                )
                for step in sorted(steps, key=lambda value: int(value.target_index))
            ]
            group = UnifiedTransformGroup([transform for _target_index, transform in ordered])
            group.compile_unified(scheme.backend)
            self.groups_by_input_index[int(input_index)] = group
            self.target_indices_by_input_index[int(input_index)] = tuple(
                int(target_index) for target_index, _transform in ordered
            )

        self.output_relayout_kernel = R34NativeAlignedRelayoutKernel(
            spec=self.spec,
            native_plan=self.native_plan,
            direction="native_to_compact",
            name=f"{self.output_node_id}_native_halo_to_compact",
            output_shape=torch.Size(getattr(self.module, "output_shape")),
            fhe_output_shape=torch.Size(getattr(self.module, "fhe_output_shape")),
        )
        self.output_relayout_kernel.compile(scheme, level=int(conv_output_level))
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
            "input_relayout_s": 0.0,
            "output_relayout_s": 0.0,
        }
        self.compile(scheme)
        if self.input_relayout_kernel is None or self.output_relayout_kernel is None:
            raise RuntimeError(f"{self.output_node_id} native aligned halo relayout kernels did not compile")

        relayout_started = time.time()
        native_source_ct = self.input_relayout_kernel.apply(source_ct)
        self.last_runtime_timing["input_relayout_s"] = float(time.time() - relayout_started)

        ids = tuple(int(value) for value in getattr(native_source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            raise RuntimeError(
                f"{self.output_node_id} native aligned halo requires {self.cols} source ciphertext blocks, got {len(ids)}"
            )

        output_blocks: list[Any | None] = [None for _ in range(int(self.rows))]
        fuse_output_rescale = bool(_unified_output_fusion_enabled())
        evaluate_started = time.time()
        for input_index, group in sorted(self.groups_by_input_index.items()):
            if int(input_index) >= len(ids):
                continue
            output_ids = group.evaluate_unified(int(ids[int(input_index)]), scheme.backend)
            for target_index, output_id in zip(self.target_indices_by_input_index[int(input_index)], output_ids):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(self.slots)]),
                    torch.Size([1, int(self.slots)]),
                )
                if not fuse_output_rescale:
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
                raise RuntimeError(f"missing native aligned halo output block {block_index} for {self.output_node_id}")
            if fuse_output_rescale:
                block_ct = _rescale_cipher_tensor(block_ct)
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
        relayout_started = time.time()
        compact_output = self.output_relayout_kernel.apply(native_output)
        compact_output.set_scale(int(scheme.params.get_default_scale()))
        self.last_runtime_timing["output_relayout_s"] = float(time.time() - relayout_started)
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {
            self.output_node_id: compact_output
        }


class R34InterGroupHybridSameShapeRuntimeExecutor(R34OrionSameShapeRuntimeExecutor):
    def __init__(self, *, module: Any, spec: R34SameShapeStageSpec, output_node_id: str) -> None:
        super().__init__(module=module, spec=spec, output_node_id=output_node_id)
        self.groups_by_input_block: list[Any] = []
        self.target_indices_by_input_block: list[tuple[int, ...]] = []
        self.input_block_pairs: list[tuple[int, int | None]] = []
        self.complex_input_block_flags: list[bool] = []
        self.hybrid_group_reject_reasons: list[str] = []
        self.hybrid_pair_count = 0
        self.hybrid_pair_rejected_count = 0
        self.hybrid_pair_reject_reasons: list[str] = []
        self.hybrid_pair_schedule_padded_count = 0
        self.hybrid_pair_schedule_pad_reasons: list[str] = []
        self.hybrid_pair_layout_strategy = ""
        self.hybrid_pair_layout_strict_pair_count = 0
        self.hybrid_pair_layout_covered_output_count = 0
        self.hybrid_pair_layout_reject_reasons: list[str] = []

    def _hardcoded_halo_effective_conv_lt_tasks(self, plan: R34SameShapeHaloRelayoutPlan) -> int:
        return int(plan.effective_conv_lt_tasks_with_hybrid)

    def compile_cache_metadata(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "rows": int(self.rows),
            "cols": int(self.cols),
            **self._hardcoded_halo_relayout_metadata(),
            "hybrid_pair_count": int(self.hybrid_pair_count),
            "hybrid_pair_rejected_count": int(self.hybrid_pair_rejected_count),
            "hybrid_pair_reject_reasons": [str(value) for value in self.hybrid_pair_reject_reasons],
            "hybrid_pair_schedule_padded_count": int(self.hybrid_pair_schedule_padded_count),
            "hybrid_pair_schedule_pad_reasons": [str(value) for value in self.hybrid_pair_schedule_pad_reasons],
            "hybrid_pair_layout_strategy": str(self.hybrid_pair_layout_strategy),
            "hybrid_pair_layout_strict_pair_count": int(self.hybrid_pair_layout_strict_pair_count),
            "hybrid_pair_layout_covered_output_count": int(self.hybrid_pair_layout_covered_output_count),
            "hybrid_pair_layout_reject_reasons": [
                str(value) for value in self.hybrid_pair_layout_reject_reasons
            ],
            "groups_by_input_block": [
                {
                    "storage_key": str(getattr(group, "_storage_key", "")),
                    "target_indices": [
                        int(value)
                        for value in self.target_indices_by_input_block[index]
                    ],
                    "input_block_pair": [
                        int(self.input_block_pairs[index][0]),
                        None if self.input_block_pairs[index][1] is None else int(self.input_block_pairs[index][1]),
                    ],
                    "complex_input_block": bool(self.complex_input_block_flags[index]),
                    "hybrid_pair_reject_reason": (
                        str(self.hybrid_group_reject_reasons[index])
                        if index < len(self.hybrid_group_reject_reasons)
                        else ""
                    ),
                }
                for index, group in enumerate(self.groups_by_input_block)
            ],
        }

    def _compile_from_cache_metadata(self, scheme: Any) -> bool:
        metadata = dict(self._compile_cache_metadata or {})
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "load" or not metadata:
            return False
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.rows = int(metadata.get("rows", 0))
        self.cols = int(metadata.get("cols", 0))
        self.hybrid_pair_count = int(metadata.get("hybrid_pair_count", 0))
        self.hybrid_pair_rejected_count = int(metadata.get("hybrid_pair_rejected_count", 0))
        self.hybrid_pair_reject_reasons = [str(value) for value in metadata.get("hybrid_pair_reject_reasons", [])]
        self.hybrid_pair_schedule_padded_count = int(metadata.get("hybrid_pair_schedule_padded_count", 0))
        self.hybrid_pair_schedule_pad_reasons = [
            str(value) for value in metadata.get("hybrid_pair_schedule_pad_reasons", [])
        ]
        self.hybrid_pair_layout_strategy = str(metadata.get("hybrid_pair_layout_strategy", ""))
        self.hybrid_pair_layout_strict_pair_count = int(metadata.get("hybrid_pair_layout_strict_pair_count", 0))
        self.hybrid_pair_layout_covered_output_count = int(
            metadata.get("hybrid_pair_layout_covered_output_count", 0)
        )
        self.hybrid_pair_layout_reject_reasons = [
            str(value) for value in metadata.get("hybrid_pair_layout_reject_reasons", [])
        ]
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme, extra_depth=0)
        level = int(self._level(scheme))
        compile_started = time.time()
        for group_meta in list(metadata.get("groups_by_input_block", [])):
            target_indices = tuple(int(value) for value in group_meta.get("target_indices", []))
            group = UnifiedTransformGroup(
                [_cached_transform_shell(level=int(level), scheme=scheme) for _target in target_indices]
            )
            group._storage_key = str(group_meta["storage_key"])
            group.compile_unified(scheme.backend)
            pair = list(group_meta.get("input_block_pair", []))
            self.groups_by_input_block.append(group)
            self.target_indices_by_input_block.append(target_indices)
            self.input_block_pairs.append((int(pair[0]), None if len(pair) < 2 or pair[1] is None else int(pair[1])))
            self.complex_input_block_flags.append(bool(group_meta.get("complex_input_block", False)))
            self.hybrid_group_reject_reasons.append(str(group_meta.get("hybrid_pair_reject_reason", "")))
        if "hybrid_pair_count" not in metadata:
            self.hybrid_pair_count = int(sum(1 for value in self.complex_input_block_flags if bool(value)))
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)
        return True

    def compile(self, scheme: Any) -> None:
        if self.groups_by_input_block:
            return
        if self._compile_from_cache_metadata(scheme):
            return
        prepare_plans_started = time.time()
        plan = self._ensure_plan()
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.last_runtime_timing["prepare_plans_s"] = float(time.time() - prepare_plans_started)
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        self.rows = int(len(plan.output_regions))
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme, extra_depth=0)
        self.hybrid_pair_count = 0
        self.hybrid_pair_rejected_count = 0
        self.hybrid_pair_reject_reasons = []
        self.hybrid_pair_schedule_padded_count = 0
        self.hybrid_pair_schedule_pad_reasons = []
        self.hybrid_pair_layout_strategy = ""
        self.hybrid_pair_layout_strict_pair_count = 0
        self.hybrid_pair_layout_covered_output_count = 0
        self.hybrid_pair_layout_reject_reasons = []
        self.hybrid_group_reject_reasons = []

        level = self._level(scheme)
        prepare_transforms_started = time.time()
        steps_by_target_and_input: dict[int, dict[int, Any]] = {}
        max_input_index = -1
        for step in plan.linear_transform_steps:
            input_index = int(_parse_block_index(step.input_id))
            max_input_index = max(int(max_input_index), int(input_index))
            steps_by_target_and_input.setdefault(int(step.target_index), {})[int(input_index)] = step
        self.cols = int(max_input_index + 1)
        self.last_runtime_timing["prepare_transforms_s"] = float(time.time() - prepare_transforms_started)

        compile_started = time.time()
        prepared_by_id = {str(plain.plaintext_id): plain for plain in plan.prepared_plaintexts}
        template_by_id = {
            str(entry.template_id): entry
            for family in plan.family_templates
            for entry in family.template_entries
        }
        schedule_family = f"r34_same_shape:{self.spec.family_label}:{self.spec.policy}"

        def mark_schedule_padding(transform: Any | None) -> Any | None:
            return mark_hybrid_schedule_padding_allowed(transform, family=str(schedule_family))

        def compile_entries(
            entries: list[tuple[int, Any]],
            *,
            pair: tuple[int, int | None],
            is_complex: bool,
            reject_reason: str = "",
        ) -> None:
            if not entries:
                return
            ordered = sorted(entries, key=lambda item: int(item[0]))
            group = UnifiedTransformGroup([transform for _target_index, transform in ordered])
            group.compile_unified(scheme.backend)
            self.groups_by_input_block.append(group)
            self.target_indices_by_input_block.append(tuple(int(target_index) for target_index, _transform in ordered))
            self.input_block_pairs.append((int(pair[0]), None if pair[1] is None else int(pair[1])))
            self.complex_input_block_flags.append(bool(is_complex))
            self.hybrid_group_reject_reasons.append(str(reject_reason))

        transforms_by_input: dict[int, list[Any | None]] = {}
        for input_index in range(int(self.cols)):
            block_transforms: list[Any | None] = []
            for target_index in range(int(self.rows)):
                step = steps_by_target_and_input.get(int(target_index), {}).get(int(input_index))
                block_transforms.append(
                    mark_schedule_padding(
                        None
                        if step is None
                        else _transform_from_plan_step(
                            plan=plan,
                            step=step,
                            level=int(level),
                            scheme=scheme,
                            name=f"{self.output_node_id}_{step.input_id}_t{int(step.target_index)}",
                            prepared_by_id=prepared_by_id,
                            template_by_id=template_by_id,
                        )
                    )
                )
            transforms_by_input[int(input_index)] = block_transforms

        layout_plan = optimize_hybrid_pair_layout(
            transforms_by_input,
            int(self.slots),
            allow_schedule_materialization=True,
        )
        materialization = materialize_hybrid_pair_layout_schedules(
            transforms_by_input,
            layout_plan,
            int(self.slots),
            name_prefix=f"{self.output_node_id}_global_hybrid_layout",
        )
        self.hybrid_pair_layout_strict_pair_count = int(layout_plan.strict_pair_count)
        self.hybrid_pair_layout_covered_output_count = int(layout_plan.covered_output_count)
        self.hybrid_pair_layout_reject_reasons = [
            str(value) for value in layout_plan.rejected_adjacent_pair_reasons
        ]
        self.hybrid_pair_schedule_padded_count += int(materialization.pair_count)
        self.hybrid_pair_schedule_pad_reasons.extend(str(value) for value in materialization.reasons)
        use_strict_layout = int(layout_plan.strict_pair_count) > 0
        if bool(use_strict_layout):
            self.hybrid_pair_layout_strategy = (
                "global_schedule_layout"
                if int(materialization.pair_count) > 0
                else "strict_schedule_dp"
            )
            layout_items = [
                (int(item.left_index), None if item.right_index is None else int(item.right_index), True)
                for item in layout_plan.items
            ]
        else:
            self.hybrid_pair_layout_strategy = "adjacent_strict_reject_fallback"
            layout_items = []
            for left_input_index in range(0, int(self.cols), 2):
                right_input_index = int(left_input_index + 1)
                has_right = int(right_input_index) < int(self.cols)
                layout_items.append(
                    (int(left_input_index), int(right_input_index) if bool(has_right) else None, False)
                )

        for left_input_index, maybe_right_input_index, _layout_pair_planned in layout_items:
            left_transforms = transforms_by_input[int(left_input_index)]
            has_right = maybe_right_input_index is not None
            right_input_index = (
                int(maybe_right_input_index) if maybe_right_input_index is not None else int(left_input_index + 1)
            )
            right_transforms = transforms_by_input.get(int(right_input_index)) if bool(has_right) else None
            if bool(has_right) and right_transforms is not None:
                candidates: list[tuple[int, Any | None, Any | None]] = []
                reject_reasons: list[str] = []
                pair_pad_reasons: list[str] = []
                for target_index in range(int(self.rows)):
                    left_transform = left_transforms[int(target_index)]
                    right_transform = right_transforms[int(target_index)]
                    if left_transform is None and right_transform is None:
                        continue
                    pad_reason = ""
                    if pad_reason:
                        pair_pad_reasons.append(f"target={int(target_index)}:{pad_reason}")
                    candidates.append((int(target_index), left_transform, right_transform))
                    reason = hybrid_pair_schedule_reject_reason(left_transform, right_transform, int(self.slots))
                    if reason:
                        reject_reasons.append(f"target={int(target_index)}:{reason}")
                if not candidates:
                    continue
                if reject_reasons:
                    reason = "; ".join(reject_reasons)
                    self.hybrid_pair_rejected_count += 1
                    self.hybrid_pair_reject_reasons.append(
                        f"input_pair=({int(left_input_index)},{int(right_input_index)}):{reason}"
                    )
                    compile_entries(
                        [
                            (int(target_index), left_transform)
                            for target_index, left_transform, _right_transform in candidates
                            if left_transform is not None
                        ],
                        pair=(int(left_input_index), None),
                        is_complex=False,
                        reject_reason=reason,
                    )
                    compile_entries(
                        [
                            (int(target_index), right_transform)
                            for target_index, _left_transform, right_transform in candidates
                            if right_transform is not None
                        ],
                        pair=(int(right_input_index), None),
                        is_complex=False,
                        reject_reason=reason,
                    )
                else:
                    entries = [
                        (
                            int(target_index),
                            _merge_optional_dense_block_transforms_to_complex(
                                left_transform,
                                right_transform,
                                name=(
                                    f"{self.output_node_id}_hybrid_pair_"
                                    f"{int(left_input_index)}_{int(right_input_index)}_t{int(target_index)}"
                                ),
                                real_lane_output_scale=0.5,
                            ),
                        )
                        for target_index, left_transform, right_transform in candidates
                    ]
                    self.hybrid_pair_count += 1
                    if pair_pad_reasons:
                        self.hybrid_pair_schedule_padded_count += 1
                        self.hybrid_pair_schedule_pad_reasons.append(
                            f"input_pair=({int(left_input_index)},{int(right_input_index)}):"
                            + "; ".join(pair_pad_reasons)
                        )
                    compile_entries(
                        entries,
                        pair=(int(left_input_index), int(right_input_index)),
                        is_complex=True,
                    )
            else:
                entries = [
                    (int(target_index), transform)
                    for target_index, transform in enumerate(left_transforms)
                    if transform is not None
                ]
                compile_entries(
                    entries,
                    pair=(int(left_input_index), None),
                    is_complex=False,
                )
        del transforms_by_input
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.compile(scheme)
        ids = tuple(int(value) for value in getattr(source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            raise RuntimeError(f"{self.output_node_id} requires {self.cols} source ciphertext blocks, got {len(ids)}")

        output_blocks: list[Any | None] = [None for _ in range(int(self.rows))]
        evaluate_started = time.time()
        for block_index, group in enumerate(self.groups_by_input_block):
            left_input_index, right_input_index = self.input_block_pairs[int(block_index)]
            if bool(self.complex_input_block_flags[int(block_index)]):
                if right_input_index is None:
                    raise RuntimeError(f"{self.output_node_id} hybrid input block pair is missing its imaginary lane")
                imag_id = scheme.evaluator.mul_imaginary_unit(int(ids[int(right_input_index)]), +1, False)
                input_id = scheme.evaluator.add_ciphertext(int(ids[int(left_input_index)]), int(imag_id), False)
            else:
                input_id = int(ids[int(left_input_index)])
            output_ids = group.evaluate_unified(int(input_id), scheme.backend)
            for target_index, output_id in zip(self.target_indices_by_input_block[int(block_index)], output_ids):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(self.slots)]),
                    torch.Size([1, int(self.slots)]),
                )
                partial = _rescale_cipher_tensor(partial)
                if bool(self.complex_input_block_flags[int(block_index)]):
                    conj = partial.conjugate(in_place=False)
                    partial, conj = _align_ciphertexts_for_add(partial, conj)
                    partial = partial.add(conj, in_place=True)
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
                raise RuntimeError(f"missing same-shape output block {block_index} for {self.output_node_id}")
            output_blocks[int(block_index)] = None
            block_ct = self._add_bias(block_ct, block_index=int(block_index))
            _maybe_set_default_scale(block_ct)
            output_ids.append(int(block_ct.ids[0]))
            block_ct.ids = []
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {
            self.output_node_id: CipherTensor(
                scheme,
                output_ids,
                self.output_shape,
                self.fhe_output_shape,
            )
        }


class R34IntraGroupPack2SameShapeRuntimeExecutor(R34OrionSameShapeRuntimeExecutor):
    pass


class R34NoHybridSameShapeRuntimeExecutor(R34OrionSameShapeRuntimeExecutor):
    """Memory-bounded real-only same-shape runtime.

    The regular no-hybrid Orion executor groups every output block for one
    source block into a single UnifiedTransformGroup.  That keeps the runtime
    simple but creates very large compile-time matrices for high-resolution
    U22/U256 same-shape convs.  This executor preserves the no real/imag
    contract while compiling small source-target groups, so no-RI ablations
    stay halo-local without materializing the dangerous global LT path.
    """

    def __init__(
        self,
        *,
        module: Any,
        spec: R34SameShapeStageSpec,
        output_node_id: str,
        max_targets_per_group: int | None = None,
    ) -> None:
        super().__init__(module=module, spec=spec, output_node_id=output_node_id)
        self.max_targets_per_group = (
            max(1, int(os.environ.get("ORION_SAME_SHAPE_NO_HYBRID_TARGET_GROUP_SIZE", "1")))
            if max_targets_per_group is None
            else max(1, int(max_targets_per_group))
        )
        self.groups_by_input_chunk: list[Any] = []
        self.target_indices_by_input_chunk: list[tuple[int, ...]] = []
        self.input_index_by_chunk: list[int] = []

    def compile_cache_metadata(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "max_targets_per_group": int(self.max_targets_per_group),
            **self._hardcoded_halo_relayout_metadata(),
            "groups_by_input_chunk": [
                {
                    "input_index": int(self.input_index_by_chunk[index]),
                    "storage_key": str(getattr(group, "_storage_key", "")),
                    "target_indices": [
                        int(value)
                        for value in self.target_indices_by_input_chunk[index]
                    ],
                }
                for index, group in enumerate(self.groups_by_input_chunk)
            ],
        }

    def _compile_from_cache_metadata(self, scheme: Any) -> bool:
        metadata = dict(self._compile_cache_metadata or {})
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "load" or not metadata:
            return False
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.rows = int(metadata.get("rows", 0))
        self.cols = int(metadata.get("cols", 0))
        self.max_targets_per_group = max(1, int(metadata.get("max_targets_per_group", self.max_targets_per_group)))
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme, extra_depth=0)
        level = int(self._level(scheme))
        compile_started = time.time()
        for group_meta in list(metadata.get("groups_by_input_chunk", [])):
            input_index = int(group_meta.get("input_index", 0))
            target_indices = tuple(int(value) for value in group_meta.get("target_indices", []))
            if not target_indices:
                continue
            group = UnifiedTransformGroup(
                [_cached_transform_shell(level=int(level), scheme=scheme) for _target in target_indices]
            )
            group._storage_key = str(group_meta["storage_key"])
            group.compile_unified(scheme.backend)
            self.groups_by_input_chunk.append(group)
            self.target_indices_by_input_chunk.append(target_indices)
            self.input_index_by_chunk.append(int(input_index))
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)
        return True

    def compile(self, scheme: Any) -> None:
        if self.groups_by_input_chunk:
            return
        if self._compile_from_cache_metadata(scheme):
            return
        prepare_plans_started = time.time()
        plan = self._ensure_plan()
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.last_runtime_timing["prepare_plans_s"] = float(time.time() - prepare_plans_started)
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        self.rows = int(len(plan.output_regions))
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme, extra_depth=0)

        level = self._level(scheme)
        prepare_transforms_started = time.time()
        steps_by_input_index: dict[int, list[Any]] = {}
        for step in plan.linear_transform_steps:
            input_index = int(_parse_block_index(step.input_id))
            steps_by_input_index.setdefault(int(input_index), []).append(step)
        self.cols = int(max(steps_by_input_index.keys(), default=-1) + 1)
        self.last_runtime_timing["prepare_transforms_s"] = float(time.time() - prepare_transforms_started)

        compile_started = time.time()
        prepared_by_id = {str(plain.plaintext_id): plain for plain in plan.prepared_plaintexts}
        template_by_id = {
            str(entry.template_id): entry
            for family in plan.family_templates
            for entry in family.template_entries
        }
        for input_index, steps in sorted(steps_by_input_index.items()):
            ordered_steps = sorted(steps, key=lambda value: int(value.target_index))
            for start in range(0, len(ordered_steps), int(self.max_targets_per_group)):
                chunk_steps = ordered_steps[int(start): int(start + int(self.max_targets_per_group))]
                ordered = [
                    (
                        int(step.target_index),
                        _transform_from_plan_step(
                            plan=plan,
                            step=step,
                            level=int(level),
                            scheme=scheme,
                            name=f"{self.output_node_id}_nohybrid_{step.input_id}_t{int(step.target_index)}",
                            prepared_by_id=prepared_by_id,
                            template_by_id=template_by_id,
                        ),
                    )
                    for step in chunk_steps
                ]
                if not ordered:
                    continue
                group = UnifiedTransformGroup([transform for _target_index, transform in ordered])
                group.compile_unified(scheme.backend)
                self.groups_by_input_chunk.append(group)
                self.target_indices_by_input_chunk.append(
                    tuple(int(target_index) for target_index, _transform in ordered)
                )
                self.input_index_by_chunk.append(int(input_index))
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.compile(scheme)
        ids = tuple(int(value) for value in getattr(source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            raise RuntimeError(f"{self.output_node_id} requires {self.cols} source ciphertext blocks, got {len(ids)}")

        output_blocks: list[Any | None] = [None for _ in range(int(self.rows))]
        fuse_output_rescale = bool(_unified_output_fusion_enabled())
        evaluate_started = time.time()
        for chunk_index, group in enumerate(self.groups_by_input_chunk):
            input_index = int(self.input_index_by_chunk[int(chunk_index)])
            if int(input_index) >= len(ids):
                continue
            output_ids = group.evaluate_unified(int(ids[int(input_index)]), scheme.backend)
            for target_index, output_id in zip(
                self.target_indices_by_input_chunk[int(chunk_index)],
                output_ids,
            ):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(self.slots)]),
                    torch.Size([1, int(self.slots)]),
                )
                if not fuse_output_rescale:
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
                raise RuntimeError(f"missing same-shape output block {block_index} for {self.output_node_id}")
            if fuse_output_rescale:
                block_ct = _rescale_cipher_tensor(block_ct)
            block_ct = self._add_bias(block_ct, block_index=int(block_index))
            block_ct.set_scale(int(scheme.params.get_default_scale()))
            output_ids.append(int(block_ct.ids[0]))
            block_ct.ids = []
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {
            self.output_node_id: CipherTensor(
                scheme,
                output_ids,
                self.output_shape,
                self.fhe_output_shape,
            )
        }


class R34Pack2SameShapeRuntimeExecutor:
    def __init__(self, *, module: Any, spec: R34SameShapeStageSpec, output_node_id: str) -> None:
        self.module = module
        self.spec = spec
        self.output_node_id = str(output_node_id)
        self.group: Any | None = None
        self.groups_by_input_block: list[Any] = []
        self.target_indices_by_input_block: list[tuple[int, ...]] = []
        self.input_block_pairs: list[tuple[int, int | None]] = []
        self.complex_input_block_flags: list[bool] = []
        self.hybrid_group_reject_reasons: list[str] = []
        self.hybrid_pair_count = 0
        self.hybrid_pair_rejected_count = 0
        self.hybrid_pair_reject_reasons: list[str] = []
        self.target_indices: tuple[int, ...] = ()
        self.bias_vector: torch.Tensor | None = None
        self.bias_plaintexts: dict[int, Any | None] = {}
        self._bias_plaintext_cache: dict[tuple[int, int], Any] = {}
        self.output_shape = getattr(module, "output_shape", None)
        self.fhe_output_shape = getattr(module, "fhe_output_shape", None)
        self.slots = int(_spec_slot_count(spec))
        self.cols = 2
        self.rows = 0
        self.compile_count = 0
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.last_runtime_timing: dict[str, float] = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self._compile_cache_metadata: dict[str, Any] = {}

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def _output_level(self, scheme: Any, *, extra_depth: int = 0) -> int:
        depth = int(self.assigned_depth) if self.assigned_depth is not None else 1
        return max(0, int(self._level(scheme)) - max(0, int(depth)) - max(0, int(extra_depth)))

    def _validate_module(self) -> None:
        weight = getattr(self.module, "on_weight", None)
        if weight is None:
            raise RuntimeError(f"{self.output_node_id} has no fused Orion weight for R34 pack2 runtime")
        if tuple(int(v) for v in tuple(weight.shape)) != tuple(int(v) for v in self.spec.weight_shape):
            raise RuntimeError(f"{self.output_node_id} weight shape does not match {self.spec.family_label}")
        if tuple(int(v) for v in getattr(self.module, "stride", ())) != (1, 1):
            raise RuntimeError(f"{self.output_node_id} stride does not match {self.spec.family_label}")
        if tuple(int(v) for v in getattr(self.module, "padding", ())) != (1, 1):
            raise RuntimeError(f"{self.output_node_id} padding does not match {self.spec.family_label}")
        if tuple(int(v) for v in getattr(self.module, "input_shape", torch.Size())[1:]) != (int(self.spec.c), int(self.spec.h), int(self.spec.w)):
            raise RuntimeError(f"{self.output_node_id} input_shape does not match {self.spec.family_label}")
        if tuple(int(v) for v in getattr(self.module, "output_shape", torch.Size())[1:]) != (int(self.spec.c), int(self.spec.h), int(self.spec.w)):
            raise RuntimeError(f"{self.output_node_id} output_shape does not match {self.spec.family_label}")
        if int(getattr(self.module, "input_gap", -1)) != int(self.spec.gap):
            raise RuntimeError(f"{self.output_node_id} input_gap does not match {self.spec.family_label}")
        if int(getattr(self.module, "output_gap", -1)) != int(self.spec.gap):
            raise RuntimeError(f"{self.output_node_id} output_gap does not match {self.spec.family_label}")

    def _bias_chunk(self, *, target_index: int) -> torch.Tensor | None:
        if self.bias_vector is None:
            return None
        start = int(target_index) * int(self.slots)
        end = min(int(start + self.slots), int(self.bias_vector.numel()))
        out = torch.zeros((int(self.slots),), dtype=torch.float32)
        if end > start:
            out[: int(end - start)] = self.bias_vector[int(start): int(end)]
        return out

    def _add_bias(self, ct: Any, *, target_index: int) -> Any:
        bias_pt = self.bias_plaintexts.get(int(target_index))
        if bias_pt is None or int(bias_pt.level()) != int(ct.level()):
            bias_pt = self._bias_plaintext_cache.get((int(target_index), int(ct.level())))
        if bias_pt is None or int(bias_pt.level()) != int(ct.level()):
            chunk = self._bias_chunk(target_index=int(target_index))
            if chunk is None:
                return ct
            bias_pt = _encode_plaintext_for_add(ct, chunk)
            self._bias_plaintext_cache[(int(target_index), int(ct.level()))] = bias_pt
        return _add_plaintext_for_add(ct, bias_pt)

    def _compile_bias_plaintexts(self, scheme: Any) -> dict[int, Any | None]:
        if self.bias_vector is None:
            return {}
        level = self._output_level(scheme)
        scale = int(scheme.params.get_default_scale())
        plaintexts: dict[int, Any | None] = {}
        for target_index in self.target_indices:
            chunk = self._bias_chunk(target_index=int(target_index))
            ptxt = None if chunk is None else scheme.encode(chunk, level=int(level), scale=int(scale))
            if ptxt is not None:
                self._bias_plaintext_cache[(int(target_index), int(level))] = ptxt
            plaintexts[int(target_index)] = ptxt
        return plaintexts

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        self._compile_cache_metadata = dict(metadata or {})

    def compile_cache_metadata(self) -> dict[str, Any]:
        relayout_plan = r34_same_shape_hardcoded_relayout_plan(self.spec)
        return {
            "kind": type(self).__name__,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "same_shape_runtime_layout": "height_stripe_halo_local_hardcoded",
            "r34_same_shape_halo_relayout_plan": relayout_plan.to_dict(),
            "conv_lt_raw_submatrix_tasks": int(relayout_plan.raw_conv_lt_tasks),
            "conv_lt_effective_submatrix_tasks": int(relayout_plan.effective_conv_lt_tasks_with_hybrid),
            "legacy_flat_conv_lt_tasks": int(relayout_plan.legacy_flat_conv_lt_tasks),
            "legacy_flat_offdiag_tasks": int(relayout_plan.legacy_flat_offdiag_tasks),
            "relayout_rotation_count": int(relayout_plan.relayout_rotations),
            "relayout_mask_mult_count": int(relayout_plan.relayout_mask_mults),
            "storage_key": "" if self.group is None else str(getattr(self.group, "_storage_key", "")),
            "target_indices": [int(value) for value in self.target_indices],
            "hybrid_pair_count": int(self.hybrid_pair_count),
            "hybrid_pair_rejected_count": int(self.hybrid_pair_rejected_count),
            "hybrid_pair_reject_reasons": [str(value) for value in self.hybrid_pair_reject_reasons],
            "groups_by_input_block": [
                {
                    "storage_key": str(getattr(group, "_storage_key", "")),
                    "target_indices": [
                        int(value)
                        for value in self.target_indices_by_input_block[index]
                    ],
                    "input_block_pair": [
                        int(self.input_block_pairs[index][0]),
                        None if self.input_block_pairs[index][1] is None else int(self.input_block_pairs[index][1]),
                    ],
                    "complex_input_block": bool(self.complex_input_block_flags[index]),
                    "hybrid_pair_reject_reason": (
                        str(self.hybrid_group_reject_reasons[index])
                        if index < len(self.hybrid_group_reject_reasons)
                        else ""
                    ),
                }
                for index, group in enumerate(self.groups_by_input_block)
            ],
        }

    def _compile_from_cache_metadata(self, scheme: Any) -> bool:
        metadata = dict(self._compile_cache_metadata or {})
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "load" or not metadata:
            return False
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.rows = int(metadata.get("rows", 0))
        self.cols = int(metadata.get("cols", 2))
        self.target_indices = tuple(int(value) for value in metadata.get("target_indices", []))
        self.hybrid_pair_count = int(metadata.get("hybrid_pair_count", 0))
        self.hybrid_pair_rejected_count = int(metadata.get("hybrid_pair_rejected_count", 0))
        self.hybrid_pair_reject_reasons = [str(value) for value in metadata.get("hybrid_pair_reject_reasons", [])]
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme)
        level = int(self._level(scheme))
        compile_started = time.time()
        group_rows = list(metadata.get("groups_by_input_block", []))
        if not group_rows:
            group_rows = [
                {
                    "storage_key": str(metadata["storage_key"]),
                    "target_indices": [int(value) for value in self.target_indices],
                    "input_block_pair": [0, 1],
                    "complex_input_block": True,
                    "hybrid_pair_reject_reason": "",
                }
            ]
        self.groups_by_input_block = []
        self.target_indices_by_input_block = []
        self.input_block_pairs = []
        self.complex_input_block_flags = []
        self.hybrid_group_reject_reasons = []
        for group_meta in group_rows:
            target_indices = tuple(int(value) for value in group_meta.get("target_indices", []))
            if not target_indices:
                continue
            group = UnifiedTransformGroup(
                [_cached_transform_shell(level=int(level), scheme=scheme) for _target in target_indices]
            )
            group._storage_key = str(group_meta["storage_key"])
            group.compile_unified(scheme.backend)
            pair = list(group_meta.get("input_block_pair", []))
            self.groups_by_input_block.append(group)
            self.target_indices_by_input_block.append(target_indices)
            self.input_block_pairs.append((int(pair[0]), None if len(pair) < 2 or pair[1] is None else int(pair[1])))
            self.complex_input_block_flags.append(bool(group_meta.get("complex_input_block", False)))
            self.hybrid_group_reject_reasons.append(str(group_meta.get("hybrid_pair_reject_reason", "")))
        self.group = self.groups_by_input_block[0] if self.groups_by_input_block else None
        if "hybrid_pair_count" not in metadata:
            self.hybrid_pair_count = int(sum(1 for value in self.complex_input_block_flags if bool(value)))
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)
        return True

    def compile(self, scheme: Any) -> None:
        if self.group is not None:
            return
        if self._compile_from_cache_metadata(scheme):
            return
        self._validate_module()
        from orion.nn.unified_transform import UnifiedTransformGroup

        prepare_started = time.time()
        assets = build_r34_same_shape_pack2_prototype_assets(
            spec=self.spec,
            weight_override=getattr(self.module, "on_weight"),
            bias_override=getattr(self.module, "on_bias", None),
            input_shape=(int(self.spec.c), int(self.spec.h), int(self.spec.w)),
            output_shape=(int(self.spec.c), int(self.spec.h), int(self.spec.w)),
            input_gap=int(self.spec.gap),
            output_gap=int(self.spec.gap),
            level=self._level(scheme),
            scheme=scheme,
            complex_diag_scale=0.5,
        )
        self.target_indices = tuple(int(v) for v in assets["prototype"]["target_indices"])
        prototype_transforms = list(assets["prototype"]["transforms"])
        reject_reasons = [str(value) for value in assets.get("hybrid_pair_reject_reasons", ())]
        self.hybrid_pair_count = 0
        self.hybrid_pair_rejected_count = 0
        self.hybrid_pair_reject_reasons = []
        self.hybrid_group_reject_reasons = []
        self.groups_by_input_block = []
        self.target_indices_by_input_block = []
        self.input_block_pairs = []
        self.complex_input_block_flags = []
        if prototype_transforms:
            self.hybrid_pair_count = 1
            group_specs = [
                ((0, 1), True, self.target_indices, prototype_transforms, "")
            ]
        else:
            reason = "; ".join(reject_reasons)
            if reject_reasons:
                self.hybrid_pair_rejected_count = 1
                self.hybrid_pair_reject_reasons = [f"input_pair=(0,1):{reason}"]
            baseline_groups = dict(assets["baseline_groups"])
            target_set: set[int] = set()
            group_specs = []
            for input_index in (0, 1):
                payload = baseline_groups.get(int(input_index), {})
                target_indices = tuple(int(value) for value in payload.get("target_indices", ()))
                transforms = list(payload.get("transforms", ()))
                if not transforms:
                    continue
                target_set.update(int(value) for value in target_indices)
                group_specs.append(((int(input_index), None), False, target_indices, transforms, reason))
            self.target_indices = tuple(sorted(target_set))
        self.rows = int(len(self.target_indices))
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": float(time.time() - prepare_started),
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme)
        compile_started = time.time()
        for pair, is_complex, target_indices, transforms, reject_reason in group_specs:
            group = UnifiedTransformGroup(transforms)
            group.compile_unified(scheme.backend)
            self.groups_by_input_block.append(group)
            self.target_indices_by_input_block.append(tuple(int(value) for value in target_indices))
            self.input_block_pairs.append((int(pair[0]), None if pair[1] is None else int(pair[1])))
            self.complex_input_block_flags.append(bool(is_complex))
            self.hybrid_group_reject_reasons.append(str(reject_reason))
        self.group = self.groups_by_input_block[0] if self.groups_by_input_block else None
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.compile(scheme)
        ids = tuple(int(v) for v in getattr(source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            raise RuntimeError(f"{self.output_node_id} pack2 runtime requires {self.cols} source ciphertext blocks, got {len(ids)}")
        evaluate_started = time.time()
        if not self.groups_by_input_block:
            raise RuntimeError(f"{self.output_node_id} pack2 runtime was not compiled")
        output_blocks: dict[int, Any] = {}
        for group_index, group in enumerate(self.groups_by_input_block):
            left_input_index, right_input_index = self.input_block_pairs[int(group_index)]
            if bool(self.complex_input_block_flags[int(group_index)]):
                if right_input_index is None:
                    raise RuntimeError(f"{self.output_node_id} pack2 hybrid group is missing its imaginary lane")
                imag_id = scheme.evaluator.mul_imaginary_unit(int(ids[int(right_input_index)]), +1, False)
                input_id = scheme.evaluator.add_ciphertext(int(ids[int(left_input_index)]), int(imag_id), False)
            else:
                input_id = int(ids[int(left_input_index)])
            output_ids = group.evaluate_unified(int(input_id), scheme.backend)
            for target_index, output_id in zip(self.target_indices_by_input_block[int(group_index)], output_ids):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(self.slots)]),
                    torch.Size([1, int(self.slots)]),
                )
                partial = _rescale_cipher_tensor(partial)
                if bool(self.complex_input_block_flags[int(group_index)]):
                    conj = partial.conjugate(in_place=False)
                    partial, conj = _align_ciphertexts_for_add(partial, conj)
                    partial = partial.add(conj, in_place=True)
                if int(target_index) in output_blocks:
                    lhs, rhs = _align_ciphertexts_for_add(output_blocks[int(target_index)], partial)
                    output_blocks[int(target_index)] = lhs + rhs
                else:
                    output_blocks[int(target_index)] = partial
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)

        postprocess_started = time.time()
        row_outputs: dict[int, int] = {}
        for target_index in self.target_indices:
            if int(target_index) not in output_blocks:
                raise RuntimeError(f"{self.output_node_id} missing pack2 output target {int(target_index)}")
            real = output_blocks.pop(int(target_index))
            real = self._add_bias(real, target_index=int(target_index))
            _maybe_set_default_scale(real)
            row_outputs[int(target_index)] = int(real.ids[0])
            real.ids = []
        ordered_ids = [int(row_outputs[int(index)]) for index in sorted(row_outputs)]
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {
            self.output_node_id: CipherTensor(
                scheme,
                ordered_ids,
                self.output_shape,
                self.fhe_output_shape,
            )
        }


def _maybe_set_default_scale(ct: Any) -> None:
    ct.set_scale(int(ct.scheme.params.get_default_scale()))


def _step_dict_by_target_and_input(plan: ConvSchemePlan) -> dict[int, dict[int, LinearTransformStep]]:
    out: dict[int, dict[int, LinearTransformStep]] = {}
    for step in plan.linear_transform_steps:
        input_index = int(_parse_block_index(step.input_id))
        out.setdefault(int(step.target_index), {})[int(input_index)] = step
    return out


def _merge_dense_block_transforms_to_complex(
    left: Any,
    right: Any,
    *,
    name: str,
    real_lane_output_scale: float = 1.0,
) -> Any:
    return _merge_optional_dense_block_transforms_to_complex(
        left,
        right,
        name=str(name),
        real_lane_output_scale=float(real_lane_output_scale),
    )


def _merge_optional_dense_block_transforms_to_complex(
    left: Any | None,
    right: Any | None,
    *,
    name: str,
    real_lane_output_scale: float = 1.0,
) -> Any:
    if left is None and right is None:
        raise ValueError("at least one dense block transform is required")
    anchor = left if left is not None else right
    slots = int(anchor.fhe_output_shape[-1])
    if not hybrid_pair_schedule_compatible(left, right, int(slots)):
        reason = hybrid_pair_schedule_reject_reason(left, right, int(slots))
        raise ValueError(f"dense-block hybrid merge requires identical schedules: {reason}")
    left_diags = dict(getattr(left, "diagonals", {}).get((0, 0), {})) if left is not None else {}
    right_diags = dict(getattr(right, "diagonals", {}).get((0, 0), {})) if right is not None else {}
    all_keys = sorted({int(key) for key in left_diags.keys()} | {int(key) for key in right_diags.keys()})
    merged: dict[int, torch.Tensor] = {}
    for key in all_keys:
        left_diag = left_diags.get(int(key))
        right_diag = right_diags.get(int(key))
        left_tensor = (
            left_diag.detach().clone().to(dtype=torch.float32)
            if isinstance(left_diag, torch.Tensor)
            else torch.tensor(left_diag, dtype=torch.float32)
        ) if left_diag is not None else torch.zeros((int(slots),), dtype=torch.float32)
        right_tensor = (
            right_diag.detach().clone().to(dtype=torch.float32)
            if isinstance(right_diag, torch.Tensor)
            else torch.tensor(right_diag, dtype=torch.float32)
        ) if right_diag is not None else torch.zeros((int(slots),), dtype=torch.float32)
        scale = float(real_lane_output_scale)
        merged[int(key)] = (
            left_tensor.to(dtype=torch.complex64) * float(scale)
            - 1j * right_tensor.to(dtype=torch.complex64) * float(scale)
        )
    return SimpleNamespace(
        name=str(name),
        diagonals={(0, 0): merged},
        level=int(anchor.level),
        scheme=anchor.scheme,
        fhe_output_shape=anchor.fhe_output_shape,
        output_shape=anchor.output_shape,
        target_index=int(getattr(anchor, "target_index", 0)),
        input_id="orion_complex_source_block_0",
    )


def _build_orion_same_shape_direct_transform(
    *,
    spec: R34SameShapeStageSpec,
    weight: torch.Tensor,
    output_block_index: int,
    input_block_index: int,
    level: int,
    scheme: Any,
    name: str,
) -> Any:
    output_length = _packed_active_slots(int(spec.c), int(spec.h), int(spec.w), int(spec.gap))
    slot_count = _spec_slot_count(spec)
    output_start = int(output_block_index) * int(slot_count)
    output_end = min(int(output_length), int(output_start + int(slot_count)))
    input_start = int(input_block_index) * int(slot_count)
    input_end = min(int(output_length), int(input_start + int(slot_count)))
    base_indices, delta_groups, spatial_parts = _same_shape_geometry(spec)
    candidate_shifts: set[int] = set()
    fast_terms = _same_shape_single_group_terms(
        spec=spec,
        weight=weight,
        output_block_index=int(output_block_index),
        input_block_index=int(input_block_index),
        output_start=int(output_start),
        output_end=int(output_end),
        input_start=int(input_start),
        input_end=int(input_end),
        spatial_parts=spatial_parts,
    )
    if fast_terms is not None:
        diag_tensors: dict[int, torch.Tensor] = {}
        for shift, output_slots, values in fast_terms:
            diag_index = (-int(shift)) % int(slot_count)
            diag = diag_tensors.setdefault(int(diag_index), torch.zeros((int(slot_count),), dtype=torch.float32))
            diag.index_add_(0, output_slots, values)
        if not diag_tensors:
            raise RuntimeError(
                f"{spec.family_label} produced no diagonals for output block {output_block_index} input block {input_block_index}"
            )
        return SimpleNamespace(
            name=str(name),
            diagonals={(0, 0): {int(index): diag for index, diag in sorted(diag_tensors.items())}},
            level=int(level),
            scheme=scheme,
            fhe_output_shape=torch.Size([1, int(slot_count)]),
            output_shape=torch.Size([1, int(slot_count)]),
            target_index=int(output_block_index),
            input_id=f"orion_source_block_{int(input_block_index)}",
        )

    for _kh, _kw, _out_spatial, _src_spatial, spatial_offset in spatial_parts:
        for delta in delta_groups:
            candidate_shifts.add(int(delta) + int(spatial_offset) - int(output_start) + int(input_start))

    diag_tensors: dict[int, torch.Tensor] = {}
    for shift in sorted(candidate_shifts):
        output_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        for kh, kw, out_spatial, src_spatial, spatial_offset in spatial_parts:
            needed_delta = int(shift) - int(spatial_offset) + int(output_start) - int(input_start)
            group = delta_groups.get(int(needed_delta))
            if group is None:
                continue
            oc_pairs, ic_pairs = group
            coeff = weight[
                oc_pairs.to(dtype=torch.int64),
                ic_pairs.to(dtype=torch.int64),
                int(kh),
                int(kw),
            ].to(dtype=torch.float32)
            out_global = base_indices.index_select(0, oc_pairs.to(dtype=torch.int64))[:, None] + out_spatial[None, :]
            src_global = base_indices.index_select(0, ic_pairs.to(dtype=torch.int64))[:, None] + src_spatial[None, :]
            valid = (
                (out_global >= int(output_start))
                & (out_global < int(output_end))
                & (src_global >= int(input_start))
                & (src_global < int(input_end))
            )
            if not bool(torch.any(valid).item()):
                continue
            output_parts.append((out_global - int(output_start))[valid].to(dtype=torch.int64))
            value_parts.append(coeff[:, None].expand_as(out_global)[valid].to(dtype=torch.float32))
        if not output_parts:
            continue
        output_slots = torch.cat(output_parts).to(dtype=torch.int64)
        values = torch.cat(value_parts).to(dtype=torch.float32)
        diag_index = (-int(shift)) % int(slot_count)
        diag = diag_tensors.setdefault(int(diag_index), torch.zeros((int(slot_count),), dtype=torch.float32))
        diag.index_add_(0, output_slots, values)

    if not diag_tensors:
        raise RuntimeError(
            f"{spec.family_label} produced no diagonals for output block {output_block_index} input block {input_block_index}"
        )
    return SimpleNamespace(
        name=str(name),
        diagonals={(0, 0): {int(index): diag for index, diag in sorted(diag_tensors.items())}},
        level=int(level),
        scheme=scheme,
        fhe_output_shape=torch.Size([1, int(slot_count)]),
        output_shape=torch.Size([1, int(slot_count)]),
        target_index=int(output_block_index),
        input_id=f"orion_source_block_{int(input_block_index)}",
    )


def build_r34_same_shape_pack2_prototype_assets(
    *,
    spec: R34SameShapeStageSpec,
    weight_override: torch.Tensor | None = None,
    bias_override: torch.Tensor | None = None,
    source_override: torch.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    output_shape: tuple[int, int, int] | None = None,
    input_gap: int | None = None,
    output_gap: int | None = None,
    level: int | None = None,
    scheme: Any | None = None,
    target_block_indices: tuple[int, ...] | None = None,
    complex_diag_scale: float = 1.0,
) -> dict[str, Any]:
    if scheme is None:
        raise ValueError("build_r34_same_shape_pack2_prototype_assets requires a scheme")
    if level is None:
        level = len(scheme.params.get_logq()) - 1

    torch.manual_seed(0)
    expected_policy = r34_same_shape_policy(c=int(spec.c), gap=int(spec.gap))
    if str(spec.policy) != str(expected_policy):
        raise ValueError(
            f"{spec.family_label} policy mismatch: spec says {spec.policy}, derived policy is {expected_policy}"
        )
    expected_weight_shape = tuple(int(v) for v in spec.weight_shape)
    if weight_override is None:
        weight = torch.randn(expected_weight_shape, dtype=torch.float32)
    else:
        weight = weight_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in weight.shape) != expected_weight_shape:
            raise ValueError(
                f"{spec.family_label} fused weight shape mismatch: expected {expected_weight_shape}, got {tuple(weight.shape)}"
            )
    expected_source_shape = (int(spec.c), int(spec.h), int(spec.w))
    if source_override is None:
        x = torch.randn(expected_source_shape, dtype=torch.float32)
    else:
        x = source_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in x.shape) != expected_source_shape:
            raise ValueError(
                f"{spec.family_label} source shape mismatch: expected {expected_source_shape}, got {tuple(x.shape)}"
            )
    if input_shape is not None and tuple(int(v) for v in input_shape) != expected_source_shape:
        raise ValueError(f"{spec.family_label} input_shape mismatch: {input_shape}")
    if output_shape is not None and tuple(int(v) for v in output_shape) != expected_source_shape:
        raise ValueError(f"{spec.family_label} output_shape mismatch: {output_shape}")
    if input_gap is not None and int(input_gap) != int(spec.gap):
        raise ValueError(f"{spec.family_label} input_gap mismatch: {input_gap}")
    if output_gap is not None and int(output_gap) != int(spec.gap):
        raise ValueError(f"{spec.family_label} output_gap mismatch: {output_gap}")
    if bias_override is not None:
        bias = bias_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in bias.shape) != (int(spec.c),):
            raise ValueError(f"{spec.family_label} bias shape mismatch: expected {(int(spec.c),)}, got {tuple(bias.shape)}")

    output_length = _packed_active_slots(int(spec.c), int(spec.h), int(spec.w), int(spec.gap))
    slot_count = _spec_slot_count(spec)
    output_block_count = _ceil_div(int(output_length), int(slot_count))
    if target_block_indices is None:
        selected_target_blocks = tuple(range(int(output_block_count)))
    else:
        selected_target_blocks = tuple(int(index) for index in target_block_indices)
        invalid = [index for index in selected_target_blocks if index < 0 or index >= int(output_block_count)]
        if invalid:
            raise ValueError(f"{spec.family_label} invalid target_block_indices: {invalid}")

    inputs = _split_flat_into_ring_blocks(
        _pack_gap_flat(x, shape=(int(spec.c), int(spec.h), int(spec.w)), gap=int(spec.gap)),
        prefix="orion_source_block",
        slot_count=int(slot_count),
    )
    reference = F.conv2d(x.unsqueeze(0), weight, bias=None, stride=1, padding=1)[0]

    baseline_groups: dict[int, Any] = {}
    paired_transforms: list[tuple[int, Any, Any]] = []
    reject_reasons: list[str] = []
    for target_index in selected_target_blocks:
        left_transform = _build_orion_same_shape_direct_transform(
            spec=spec,
            weight=weight,
            output_block_index=int(target_index),
            input_block_index=0,
            level=int(level),
            scheme=scheme,
            name=f"{spec.family_label}_baseline_in0_t{int(target_index)}",
        )
        right_transform = _build_orion_same_shape_direct_transform(
            spec=spec,
            weight=weight,
            output_block_index=int(target_index),
            input_block_index=1,
            level=int(level),
            scheme=scheme,
            name=f"{spec.family_label}_baseline_in1_t{int(target_index)}",
        )
        baseline_groups.setdefault(0, []).append((int(target_index), left_transform))
        baseline_groups.setdefault(1, []).append((int(target_index), right_transform))
        paired_transforms.append((int(target_index), left_transform, right_transform))
        reason = hybrid_pair_schedule_reject_reason(left_transform, right_transform, int(slot_count))
        if reason:
            reject_reasons.append(f"target={int(target_index)}:{reason}")

    prototype_transforms: list[Any] = []
    prototype_target_indices: list[int] = []
    if not reject_reasons:
        for target_index, left_transform, right_transform in paired_transforms:
            prototype_transforms.append(
                _merge_dense_block_transforms_to_complex(
                    left_transform,
                    right_transform,
                    name=f"{spec.family_label}_pack2_proto_t{int(target_index)}",
                    real_lane_output_scale=float(complex_diag_scale),
                )
            )
            prototype_target_indices.append(int(target_index))

    block0 = inputs["orion_source_block_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
    block1 = inputs["orion_source_block_1"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
    complex_source = block0.to(dtype=torch.complex64) + 1j * block1.to(dtype=torch.complex64)
    return {
        "plan": None,
        "inputs": inputs,
        "reference": reference,
        "baseline_groups": {
            int(input_index): {
                "target_indices": tuple(int(target_index) for target_index, _transform in transforms),
                "transforms": [transform for _target_index, transform in transforms],
            }
            for input_index, transforms in baseline_groups.items()
        },
        "prototype": {
            "target_indices": tuple(int(value) for value in prototype_target_indices),
            "transforms": list(prototype_transforms),
        },
        "hybrid_pair_reject_reasons": tuple(str(value) for value in reject_reasons),
        "complex_source": complex_source,
        "notes": (
            f"family_label={spec.family_label}",
            "prototype packs source block 0 into the real lane and source block 1 into the imaginary lane",
            "giant-step scratch is not yet materialized; remaining tail slack is reserved for future exploration",
        ),
    }
