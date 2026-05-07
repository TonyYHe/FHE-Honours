from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal
import math
import time

import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.nn.unified_transform import UnifiedTransformGroup

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
    return SimpleNamespace(
        name=str(name),
        diagonals={(0, 0): {int(index): diag for index, diag in sorted(diag_tensors.items())}},
        level=int(level),
        scheme=scheme,
        fhe_output_shape=torch.Size([1, int(slots)]),
        output_shape=torch.Size([1, int(slots)]),
        target_index=int(step.target_index),
        input_id=str(step.input_id),
    )


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


class R34InterGroupHybridSameShapeRuntimeExecutor(R34OrionSameShapeRuntimeExecutor):
    def __init__(self, *, module: Any, spec: R34SameShapeStageSpec, output_node_id: str) -> None:
        super().__init__(module=module, spec=spec, output_node_id=output_node_id)
        self.groups_by_input_block: list[Any] = []
        self.target_indices_by_input_block: list[tuple[int, ...]] = []
        self.input_block_pairs: list[tuple[int, int | None]] = []
        self.complex_input_block_flags: list[bool] = []

    def compile_cache_metadata(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "rows": int(self.rows),
            "cols": int(self.cols),
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
        for left_input_index in range(0, int(self.cols), 2):
            right_input_index = int(left_input_index + 1)
            has_right = int(right_input_index) < int(self.cols)
            entries: list[tuple[int, Any]] = []
            if has_right:
                for target_index in range(int(self.rows)):
                    by_input = steps_by_target_and_input.get(int(target_index), {})
                    left_step = by_input.get(int(left_input_index))
                    right_step = by_input.get(int(right_input_index))
                    left_transform = (
                        None
                        if left_step is None
                        else _transform_from_plan_step(
                            plan=plan,
                            step=left_step,
                            level=int(level),
                            scheme=scheme,
                            name=f"{self.output_node_id}_{left_step.input_id}_t{int(left_step.target_index)}",
                            prepared_by_id=prepared_by_id,
                            template_by_id=template_by_id,
                        )
                    )
                    right_transform = (
                        None
                        if right_step is None
                        else _transform_from_plan_step(
                            plan=plan,
                            step=right_step,
                            level=int(level),
                            scheme=scheme,
                            name=f"{self.output_node_id}_{right_step.input_id}_t{int(right_step.target_index)}",
                            prepared_by_id=prepared_by_id,
                            template_by_id=template_by_id,
                        )
                    )
                    if left_transform is None and right_transform is None:
                        continue
                    entries.append(
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
                    )
                is_complex = True
                pair = (int(left_input_index), int(right_input_index))
            else:
                for target_index in range(int(self.rows)):
                    step = steps_by_target_and_input.get(int(target_index), {}).get(int(left_input_index))
                    transform = (
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
                    if transform is not None:
                        entries.append((int(target_index), transform))
                is_complex = False
                pair = (int(left_input_index), None)
            if entries:
                ordered = sorted(entries, key=lambda item: int(item[0]))
                group = UnifiedTransformGroup([transform for _target_index, transform in ordered])
                group.compile_unified(scheme.backend)
                self.groups_by_input_block.append(group)
                self.target_indices_by_input_block.append(tuple(int(target_index) for target_index, _transform in ordered))
                self.input_block_pairs.append((int(pair[0]), None if pair[1] is None else int(pair[1])))
                self.complex_input_block_flags.append(bool(is_complex))
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
                if output_blocks[int(target_index)] is None:
                    output_blocks[int(target_index)] = partial
                else:
                    lhs, rhs = _align_ciphertexts_for_add(output_blocks[int(target_index)], partial)
                    output_blocks[int(target_index)] = lhs + rhs
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)

        postprocess_started = time.time()
        output_ids: list[int] = []
        needs_real_lane_extract = any(bool(value) for value in self.complex_input_block_flags)
        for block_index, block_ct in enumerate(output_blocks):
            if block_ct is None:
                raise RuntimeError(f"missing same-shape output block {block_index} for {self.output_node_id}")
            output_blocks[int(block_index)] = None
            if bool(needs_real_lane_extract):
                conj = block_ct.conjugate(in_place=False)
                block_ct, conj = _align_ciphertexts_for_add(block_ct, conj)
                block_ct = block_ct.add(conj, in_place=True)
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


class R34Pack2SameShapeRuntimeExecutor:
    def __init__(self, *, module: Any, spec: R34SameShapeStageSpec, output_node_id: str) -> None:
        self.module = module
        self.spec = spec
        self.output_node_id = str(output_node_id)
        self.group: Any | None = None
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
        return {
            "kind": type(self).__name__,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "storage_key": "" if self.group is None else str(getattr(self.group, "_storage_key", "")),
            "target_indices": [int(value) for value in self.target_indices],
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
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme)
        level = int(self._level(scheme))
        compile_started = time.time()
        self.group = UnifiedTransformGroup(
            [_cached_transform_shell(level=int(level), scheme=scheme) for _target in self.target_indices]
        )
        self.group._storage_key = str(metadata["storage_key"])
        self.group.compile_unified(scheme.backend)
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
        self.rows = int(len(self.target_indices))
        transforms = list(assets["prototype"]["transforms"])
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
        self.group = UnifiedTransformGroup(transforms)
        self.group.compile_unified(scheme.backend)
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
        if len(ids) < 2:
            raise RuntimeError(f"{self.output_node_id} pack2 runtime requires 2 source ciphertext blocks, got {len(ids)}")
        imag_id = scheme.evaluator.mul_imaginary_unit(int(ids[1]), +1, False)
        complex_id = scheme.evaluator.add_ciphertext(int(ids[0]), int(imag_id), False)
        complex_source = CipherTensor(
            scheme,
            [int(complex_id)],
            torch.Size([1, int(self.slots)]),
            torch.Size([1, int(self.slots)]),
        )
        evaluate_started = time.time()
        assert self.group is not None
        output_ids = self.group.evaluate_unified(int(complex_source.ids[0]), scheme.backend)
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)

        postprocess_started = time.time()
        row_outputs: dict[int, int] = {}
        for target_index, output_id in zip(self.target_indices, output_ids):
            raw = CipherTensor(
                scheme,
                [int(output_id)],
                torch.Size([1, int(self.slots)]),
                torch.Size([1, int(self.slots)]),
            )
            raw = _rescale_cipher_tensor(raw)
            conj = raw.conjugate(in_place=False)
            raw, conj = _align_ciphertexts_for_add(raw, conj)
            real = raw.add(conj, in_place=True)
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
    prototype_transforms: list[Any] = []
    prototype_target_indices: list[int] = []
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
        "complex_source": complex_source,
        "notes": (
            f"family_label={spec.family_label}",
            "prototype packs source block 0 into the real lane and source block 1 into the imaginary lane",
            "giant-step scratch is not yet materialized; remaining tail slack is reserved for future exploration",
        ),
    }
