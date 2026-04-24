from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import time

import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from .ir import (
    CanonicalTemplateEntry,
    ConvSchemePlan,
    ExecutionStats,
    FamilyTemplateBank,
    LinearTransformStep,
    LinearTransformTerm,
    PlainCipherTensor,
    PreparedPlaintext,
    SharedOutputBank,
    TensorRegion,
)
from .haloed_bridge import transform_from_orion_plan_step
from .runtime_group import transforms_from_conv_scheme_plan
from .r34_geometry import (
    RING_SLOT_COUNT,
    ceil_div,
    extend_h_range_to_length,
    fixed_halo_rows,
    height_stripes_for_partition,
    max_source_h_for_channels,
    hybrid_pair_channel_partitions,
    packed_active_slots,
    source_group_count,
    source_h_range_for_target,
    target_h_from_source_h,
)
from .r34_section_extractor import (
    build_core_section_tensor,
    build_section_extract_plan,
    decode_section_tensor,
    encode_section_tensor,
    extract_section_ciphertext,
)
from .r34_orion_same_shape import R34SameShapeStageSpec, _idx_chw_gap_tensor
from orion.nn.unified_transform import UnifiedTransformGroup


@dataclass(frozen=True)
class R34InterGroupBlockSpec:
    case_name: str
    family_label: str
    c: int
    h: int
    w: int
    gap: int
    kernel: int = 3
    stride: int = 1
    pad: int = 1


def _pack_gap_plain(x: torch.Tensor, *, shape: tuple[int, int, int], gap: int) -> torch.Tensor:
    c, h, w = (int(v) for v in shape)
    if tuple(int(v) for v in x.shape) != (int(c), int(h), int(w)):
        raise ValueError(f"expected source shape {(int(c), int(h), int(w))}, got {tuple(x.shape)}")
    out = torch.zeros((int(RING_SLOT_COUNT),), dtype=torch.float32)
    for channel in range(int(c)):
        for ih in range(int(h)):
            for iw in range(int(w)):
                idx = _idx_chw_gap_tensor(
                    torch.tensor(int(channel)),
                    torch.tensor(int(ih)),
                    torch.tensor(int(iw)),
                    H=int(h),
                    W=int(w),
                    gap=int(gap),
                )
                out[int(idx.item())] = x[int(channel), int(ih), int(iw)].to(dtype=torch.float32)
    return out


def _coalesce_fused_rows(keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(keys.numel()) == 0:
        return keys.to(dtype=torch.int64), values.to(dtype=torch.complex64)
    unique, inverse = torch.unique(keys.to(dtype=torch.int64), sorted=True, return_inverse=True)
    out = torch.zeros((int(unique.numel()),), dtype=torch.complex64)
    out.index_add_(0, inverse.to(dtype=torch.int64), values.to(dtype=torch.complex64))
    keep = torch.abs(out) > 0
    return unique[keep], out[keep]


def _bsgs_meta_for_shift_tensors(shift_tensors: tuple[torch.Tensor, ...]) -> tuple[int, tuple[int, ...], tuple[int, ...], ExecutionStats]:
    all_shifts: set[int] = set()
    for shifts in shift_tensors:
        all_shifts.update(int(value) for value in shifts.to(dtype=torch.int64).tolist())
    ordered = tuple(sorted(all_shifts))
    return 0, ordered, (), ExecutionStats(rotations=int(len(ordered)))


def _rotation_group_id(*, prefix: str, input_id: str) -> str:
    safe = str(input_id).replace(":", "_").replace("/", "_")
    return f"{str(prefix)}:{safe}"


def _lt_term_count(row_shifts: torch.Tensor) -> int:
    if int(row_shifts.numel()) == 0:
        return 0
    return int(torch.unique_consecutive(row_shifts.to(dtype=torch.int64)).numel())


def _source_h_range_for_target(
    *,
    target_h_range: tuple[int, int],
    input_h: int,
    kernel: int,
    stride: int,
    pad: int,
) -> tuple[int, int]:
    target_start, target_end = (int(v) for v in target_h_range)
    source_start = int(target_start) * int(stride) - int(pad)
    source_end = (int(target_end) - 1) * int(stride) - int(pad) + int(kernel) - 1
    return max(0, int(source_start)), min(int(input_h), int(source_end) + 1)


def _extend_h_range_to_length(
    *,
    required: tuple[int, int],
    desired_len: int,
    limit: int,
) -> tuple[int, int]:
    start, end = (int(v) for v in required)
    desired = min(int(limit), max(int(end - start), int(desired_len)))
    extra = int(desired - (end - start))
    if int(extra) <= 0:
        return int(start), int(end)
    left = min(int(start), int(extra // 2))
    start -= int(left)
    extra -= int(left)
    right = min(int(limit - end), int(extra))
    end += int(right)
    extra -= int(right)
    if int(extra) > 0:
        left = min(int(start), int(extra))
        start -= int(left)
    if int(end - start) < int(desired):
        raise ValueError(f"cannot extend source range {required} to length {desired}")
    return int(start), int(end)


def _surface_summary_from_stage_spec(spec: R34SameShapeStageSpec) -> dict[str, int]:
    group_partitions = hybrid_pair_channel_partitions(c=int(spec.c), gap=int(spec.gap))
    bounded_partition = group_partitions[0]
    phase = max(1, int(spec.gap) * int(spec.gap))
    source_groups = int(source_group_count(c=int(spec.c), gap=int(spec.gap)))
    surface_groups = int(max(1, (int(bounded_partition.group_end) - int(bounded_partition.group_start)) // 2))
    groups_per_surface = int(surface_groups)
    bounded_c = int(bounded_partition.c_end - bounded_partition.c_start)
    surface_c = int(max(1, int(bounded_c // 2)))
    bounded_h = int(max(stripe.source_h_end - stripe.source_h_start for stripe in height_stripes_for_partition(
        c=int(surface_c),
        h=int(spec.h),
        w=int(spec.w),
        gap=int(spec.gap),
        kernel=3,
        stride=1,
        pad=1,
        max_slots=int(RING_SLOT_COUNT),
    )))
    return {
        "source_groups": int(source_groups),
        "groups_per_surface": int(groups_per_surface),
        "bounded_c": int(bounded_c),
        "bounded_h": int(bounded_h),
    }


def _build_inter_group_rows_vectorized(
    *,
    spec: R34InterGroupBlockSpec,
    weight: torch.Tensor,
    target_surface_index: int,
    groups_per_surface: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    c = int(spec.c)
    h = int(spec.h)
    w = int(spec.w)
    gap = int(spec.gap)
    phase_count = int(gap * gap)
    if int(c) % int(phase_count) != 0:
        raise ValueError(f"{spec.case_name} requires channel count divisible by gap^2")
    group_count = int(c // phase_count)
    if int(group_count) != int(groups_per_surface) * 2:
        raise ValueError(f"{spec.case_name} inter_group expects exactly two equal channel surfaces")
    local_c = int(groups_per_surface * phase_count)
    left_offset = 0
    right_offset = int(local_c)
    target_offset = int(target_surface_index) * int(local_c)
    chunk_keys: list[torch.Tensor] = []
    chunk_values: list[torch.Tensor] = []
    ic_local = torch.arange(int(local_c), dtype=torch.int64)
    ic_left_global = ic_local + int(left_offset)
    ic_right_global = ic_local + int(right_offset)
    oc_chunk_size = 32 if int(local_c) >= 128 else 64
    for kh in range(int(spec.kernel)):
        for kw in range(int(spec.kernel)):
            oh_all = torch.arange(int(h), dtype=torch.int64)
            ow_all = torch.arange(int(w), dtype=torch.int64)
            grid_oh, grid_ow = torch.meshgrid(oh_all, ow_all, indexing="ij")
            oh_flat = grid_oh.reshape(-1)
            ow_flat = grid_ow.reshape(-1)
            ih_flat = oh_flat * int(spec.stride) - int(spec.pad) + int(kh)
            iw_flat = ow_flat * int(spec.stride) - int(spec.pad) + int(kw)
            valid = (ih_flat >= 0) & (ih_flat < int(h)) & (iw_flat >= 0) & (iw_flat < int(w))
            if not bool(torch.any(valid).item()):
                continue
            oh = oh_flat[valid]
            ow = ow_flat[valid]
            ih = ih_flat[valid]
            iw = iw_flat[valid]
            for oc0 in range(0, int(local_c), int(oc_chunk_size)):
                oc_local = torch.arange(int(oc0), int(min(int(local_c), int(oc0 + oc_chunk_size))), dtype=torch.int64)
                oc_global = oc_local + int(target_offset)
                coeff_left = weight[oc_global[:, None], ic_left_global[None, :], int(kh), int(kw)].to(dtype=torch.float32)
                coeff_right = weight[oc_global[:, None], ic_right_global[None, :], int(kh), int(kw)].to(dtype=torch.float32)
                coeff = coeff_left.to(dtype=torch.complex64) - 1j * coeff_right.to(dtype=torch.complex64)
                out_slot = _idx_chw_gap_tensor(
                    oc_local[:, None],
                    oh[None, :],
                    ow[None, :],
                    H=int(h),
                    W=int(w),
                    gap=int(gap),
                )
                src_slot = _idx_chw_gap_tensor(
                    ic_local[:, None],
                    ih[None, :],
                    iw[None, :],
                    H=int(h),
                    W=int(w),
                    gap=int(gap),
                )
                shift = out_slot[:, None, :] - src_slot[None, :, :]
                output_slot = out_slot[:, None, :].expand_as(shift)
                keys = (shift.reshape(-1) * int(RING_SLOT_COUNT) + output_slot.reshape(-1)).to(dtype=torch.int64)
                vals = coeff[:, :, None].expand_as(shift).reshape(-1).to(dtype=torch.complex64)
                coalesced_keys, coalesced_vals = _coalesce_fused_rows(keys, vals)
                chunk_keys.append(coalesced_keys)
                chunk_values.append(coalesced_vals)
    all_keys, all_values = _coalesce_fused_rows(torch.cat(chunk_keys), torch.cat(chunk_values))
    shifts = torch.div(all_keys, int(RING_SLOT_COUNT), rounding_mode="floor").to(dtype=torch.int64)
    output_slots = torch.remainder(all_keys, int(RING_SLOT_COUNT)).to(dtype=torch.int64)
    return shifts, output_slots, all_values.to(dtype=torch.complex64), int(local_c)


def _build_lt_assets_from_rows(
    *,
    spec: R34InterGroupBlockSpec,
    family_id: str,
    row_shifts: torch.Tensor,
    row_output_slots: torch.Tensor,
    row_values: torch.Tensor,
    selected_n1: int,
    baby_shifts: tuple[int, ...],
    giant_shifts: tuple[int, ...],
    lt_cost: ExecutionStats,
    target_index: int,
    representation: str,
    input_id: str = "complex_source_pair_0",
    step_suffix: str = "",
    rotation_group_id: str = "",
    rotation_cost_owner: bool = True,
    shared_multi_output: bool = False,
    shared_output_banks: tuple[SharedOutputBank, ...] = (),
) -> tuple[tuple[CanonicalTemplateEntry, ...], tuple[PreparedPlaintext, ...], LinearTransformStep]:
    suffix = f"_{step_suffix}" if str(step_suffix) else ""
    template_entries: list[CanonicalTemplateEntry] = []
    prepared_plaintexts: list[PreparedPlaintext] = []
    terms: list[LinearTransformTerm] = []
    term_bank_id = str(shared_output_banks[0].bank_id) if len(shared_output_banks) == 1 else ""
    for index, shift_tensor in enumerate(torch.unique_consecutive(row_shifts).tolist()):
        shift = int(shift_tensor)
        mask = row_shifts == int(shift)
        output_slots = row_output_slots[mask].to(dtype=torch.int64)
        values = row_values[mask].to(dtype=torch.complex64) if str(representation) != "real_bsgs" else row_values[mask].real.to(dtype=torch.float32)
        template_id = f"{spec.case_name}_t{int(target_index)}{suffix}_template_{int(index)}"
        plaintext_id = f"{spec.case_name}_t{int(target_index)}{suffix}_pt_{int(index)}"
        template_entries.append(
            CanonicalTemplateEntry(
                template_id=str(template_id),
                family_id=str(family_id),
                key=(int(target_index), int(shift)),
                fine_shift=int(shift),
                indices=output_slots,
                note="r34 inter-group python prototype template",
            )
        )
        prepared_plaintexts.append(
            PreparedPlaintext(
                plaintext_id=str(plaintext_id),
                template_id=str(template_id),
                level=0,
                scale=1.0,
                slot_count=int(RING_SLOT_COUNT),
                values=values,
                note="r34 inter-group python prototype prepared plaintext",
            )
        )
        terms.append(
            LinearTransformTerm(
                term_id=f"{spec.case_name}_t{int(target_index)}{suffix}_term_{int(index)}",
                shift=int(shift),
                plaintext_id=str(plaintext_id),
                template_id=str(template_id),
                lookup_indices=torch.arange(int(output_slots.numel()), dtype=torch.int64),
                output_slot_indices=output_slots,
                note="r34 inter-group python prototype term",
                bank_id=str(term_bank_id),
            )
        )
    return (
        tuple(template_entries),
        tuple(prepared_plaintexts),
        LinearTransformStep(
            step_id=f"{spec.case_name}_lt_{int(target_index)}{suffix}",
            input_id=str(input_id),
            target_index=int(target_index),
            selected_n1=int(selected_n1),
            baby_shifts=tuple(int(v) for v in baby_shifts),
            giant_shifts=tuple(int(v) for v in giant_shifts),
            terms=tuple(terms),
            required_rotations=tuple(sorted(set(int(v) for v in baby_shifts).union(int(v) for v in giant_shifts))),
            prepared_plaintext_ids=tuple(str(pt.plaintext_id) for pt in prepared_plaintexts),
            expected_cost=lt_cost,
            representation=str(representation),
            note="r34 inter-group python prototype LT",
            rotation_group_id=str(rotation_group_id or _rotation_group_id(prefix=f"{spec.case_name}_lt", input_id=str(input_id))),
            rotation_cost_owner=bool(rotation_cost_owner),
            shared_multi_output=bool(shared_multi_output),
            shared_output_banks=tuple(shared_output_banks),
        ),
    )


def _retarget_step_for_shared_output(
    step: LinearTransformStep,
    *,
    target_index: int,
    active_slot_count: int,
    rotation_group_id: str,
    rotation_cost_owner: bool,
    bank_id: str,
) -> LinearTransformStep:
    terms = tuple(
        replace(
            term,
            term_id=f"{term.term_id}_shared_t{int(target_index)}",
            bank_id=str(bank_id),
        )
        for term in step.terms
    )
    return replace(
        step,
        step_id=f"{step.step_id}_shared_t{int(target_index)}",
        target_index=int(target_index),
        terms=terms,
        expected_cost=ExecutionStats(
            rotations=int(step.expected_cost.rotations) if bool(rotation_cost_owner) else 0,
            ct_pt_mults=int(len(terms)),
            adds=int(len(terms)),
        ),
        rotation_group_id=str(rotation_group_id),
        rotation_cost_owner=bool(rotation_cost_owner),
        shared_multi_output=False,
        shared_output_banks=(
            SharedOutputBank(
                bank_id=str(bank_id),
                target_index=int(target_index),
                fold_lane=int(target_index),
                input_lane_id=int(target_index),
                output_slot_offset=0,
                active_slot_count=int(active_slot_count),
                term_count=int(len(terms)),
                note="full generalized inter-hsplit output surface sharing one input rotation schedule",
            ),
        ),
    )


def _collapse_native_shared_multi_output_steps(
    steps: list[LinearTransformStep],
    *,
    step_prefix: str,
) -> list[LinearTransformStep]:
    grouped: dict[tuple[str, str], list[LinearTransformStep]] = {}
    for step in steps:
        grouped.setdefault((str(step.input_id), str(step.rotation_group_id)), []).append(step)
    collapsed: list[LinearTransformStep] = []
    for (input_id, rotation_group_id), group_steps in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        if len(group_steps) <= 1:
            collapsed.extend(group_steps)
            continue
        owner_steps = [step for step in group_steps if bool(step.rotation_cost_owner)]
        if len(owner_steps) != 1:
            raise ValueError(f"cannot collapse {rotation_group_id}: expected one owner, got {len(owner_steps)}")
        owner = owner_steps[0]
        merged_terms: list[LinearTransformTerm] = []
        merged_plaintext_ids: set[str] = set()
        merged_banks: list[SharedOutputBank] = []
        expected = ExecutionStats()
        for step in group_steps:
            merged_terms.extend(step.terms)
            merged_plaintext_ids.update(str(value) for value in step.prepared_plaintext_ids)
            merged_banks.extend(step.shared_output_banks)
            expected = expected.plus(
                rotations=int(step.expected_cost.rotations),
                conjugations=int(step.expected_cost.conjugations),
                ct_pt_mults=int(step.expected_cost.ct_pt_mults),
                adds=int(step.expected_cost.adds),
            )
        collapsed.append(
            LinearTransformStep(
                step_id=f"{step_prefix}_collapsed_{len(collapsed)}",
                input_id=str(input_id),
                target_index=int(owner.target_index),
                selected_n1=int(owner.selected_n1),
                baby_shifts=tuple(int(v) for v in owner.baby_shifts),
                giant_shifts=tuple(int(v) for v in owner.giant_shifts),
                terms=tuple(merged_terms),
                required_rotations=tuple(sorted(set(int(v) for step in group_steps for v in step.required_rotations))),
                prepared_plaintext_ids=tuple(sorted(str(v) for v in merged_plaintext_ids)),
                expected_cost=expected,
                representation=owner.representation,
                note="collapsed native shared multi-output hsplit LT",
                rotation_group_id=str(rotation_group_id),
                rotation_cost_owner=True,
                shared_multi_output=True,
                shared_output_banks=tuple(merged_banks),
            )
        )
    return collapsed


def build_r34_inter_group_local_plan(
    *,
    spec: R34InterGroupBlockSpec,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    reference = F.conv2d(x.unsqueeze(0), weight, bias=None, stride=1, padding=1)[0]
    phase_count = int(spec.gap * spec.gap)
    group_count = int(spec.c // phase_count)
    group_block = int((spec.h * spec.gap) * (spec.w * spec.gap))
    groups_per_surface = int(RING_SLOT_COUNT // group_block)
    if int(groups_per_surface) <= 0 or int(group_count) != int(groups_per_surface) * 2:
        raise ValueError(f"{spec.case_name} inter_group local plan needs exactly two equal channel surfaces")
    local_c = int(groups_per_surface * phase_count)
    active_slots = packed_active_slots(c=int(local_c), h=int(spec.h), w=int(spec.w), gap=int(spec.gap))
    family_id = f"{spec.case_name}_family"
    all_templates: list[CanonicalTemplateEntry] = []
    all_prepared: list[PreparedPlaintext] = []
    lt_steps: list[LinearTransformStep] = []
    target_rows: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for target_index in (0, 1):
        row_shifts, row_output_slots, row_values, _local_c = _build_inter_group_rows_vectorized(
            spec=spec,
            weight=weight,
            target_surface_index=int(target_index),
            groups_per_surface=int(groups_per_surface),
        )
        target_rows.append((int(target_index), row_shifts, row_output_slots, row_values))
    selected_n1, baby_shifts, giant_shifts, shared_lt_cost = _bsgs_meta_for_shift_tensors(
        tuple(row_shifts for _target_index, row_shifts, _row_output_slots, _row_values in target_rows)
    )
    for row_index, (target_index, row_shifts, row_output_slots, row_values) in enumerate(target_rows):
        term_count = _lt_term_count(row_shifts)
        lt_cost = ExecutionStats(
            rotations=int(shared_lt_cost.rotations) if int(row_index) == 0 else 0,
            ct_pt_mults=int(term_count),
            adds=int(term_count),
        )
        templates, prepared, lt_step = _build_lt_assets_from_rows(
            spec=spec,
            family_id=family_id,
            row_shifts=row_shifts,
            row_output_slots=row_output_slots,
            row_values=row_values,
            selected_n1=int(selected_n1),
            baby_shifts=tuple(baby_shifts),
            giant_shifts=tuple(giant_shifts),
            lt_cost=lt_cost,
            target_index=int(target_index),
            representation="inter_group_complex",
            step_suffix="shared_rot" if int(row_index) == 0 else "shared_rot_reuse",
            rotation_group_id=_rotation_group_id(prefix=f"{spec.case_name}_inter_group", input_id="complex_source_pair_0"),
            rotation_cost_owner=bool(int(row_index) == 0),
        )
        all_templates.extend(templates)
        all_prepared.extend(prepared)
        lt_steps.append(lt_step)
    expected = ExecutionStats(adds=1)
    for lt in lt_steps:
        expected = expected.plus(rotations=int(lt.expected_cost.rotations), ct_pt_mults=int(lt.expected_cost.ct_pt_mults), adds=int(lt.expected_cost.adds))
    expected = expected.plus(conjugations=2, adds=2)
    left = x[: int(local_c)]
    right = x[int(local_c) : int(2 * int(local_c))]
    inputs = {
        "source_0_lane_0": PlainCipherTensor(_pack_gap_plain(left, shape=(int(local_c), int(spec.h), int(spec.w)), gap=int(spec.gap)), label="source_0_lane_0"),
        "source_1_lane_0": PlainCipherTensor(_pack_gap_plain(right, shape=(int(local_c), int(spec.h), int(spec.w)), gap=int(spec.gap)), label="source_1_lane_0"),
    }
    plan = ConvSchemePlan(
        case_name=str(spec.case_name),
        ring_slot_count=int(RING_SLOT_COUNT),
        output_regions=(
            TensorRegion(c_start=0, c_end=int(local_c), h_start=0, h_end=int(spec.h), w_start=0, w_end=int(spec.w)),
            TensorRegion(c_start=int(local_c), c_end=int(spec.c), h_start=0, h_end=int(spec.h), w_start=0, w_end=int(spec.w)),
        ),
        output_active_slot_counts=(int(active_slots), int(active_slots)),
        family_templates=(
            FamilyTemplateBank(
                family_id=str(family_id),
                family_key=(str(spec.case_name), "r34_inter_group_python"),
                source_tile_shape=(int(local_c), int(spec.h), int(spec.w)),
                target_tile_shape=(int(local_c), int(spec.h), int(spec.w)),
                source_h_range=(0, int(spec.h)),
                target_h_range=(0, int(spec.h)),
                template_entries=tuple(all_templates),
                member_count=2,
                evidence_kind="r34_inter_group_python",
            ),
        ),
        prepared_plaintexts=tuple(all_prepared),
        linear_transform_steps=tuple(lt_steps),
        expected_cost=expected,
        evidence_kind="r34_inter_group_python",
        notes=("inter-group two-surface bounded block",),
    )
    return plan, inputs, reference


def build_r34_same_shape_generalized_inter_group_assets(
    *,
    spec: R34SameShapeStageSpec,
    weight_override: torch.Tensor | None = None,
    source_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    if str(spec.policy) != "inter_group_hybrid":
        raise ValueError(f"{spec.family_label} is not an inter-group family")
    bounded = _surface_summary_from_stage_spec(spec)
    bounded_c = int(bounded["bounded_c"])
    if weight_override is None:
        weight = torch.randn((int(spec.c), int(spec.c), 3, 3), dtype=torch.float32)
    else:
        weight = weight_override.detach().to(dtype=torch.float32).clone()
    if source_override is None:
        x = torch.randn((int(spec.c), int(spec.h), int(spec.w)), dtype=torch.float32)
    else:
        x = source_override.detach().to(dtype=torch.float32).clone()
    reference = F.conv2d(x.unsqueeze(0), weight, bias=None, stride=1, padding=1)[0]
    blocks: list[dict[str, Any]] = []
    for partition in hybrid_pair_channel_partitions(c=int(spec.c), gap=int(spec.gap)):
        in_start = int(partition.c_start)
        in_end = int(partition.c_end)
        surface_c = int(max(1, (int(in_end - in_start) // 2)))
        for stripe in height_stripes_for_partition(
            c=int(surface_c),
            h=int(spec.h),
            w=int(spec.w),
            gap=int(spec.gap),
            kernel=3,
            stride=1,
            pad=1,
            max_slots=int(RING_SLOT_COUNT),
        ):
            th0 = int(stripe.target_h_start)
            th1 = int(stripe.target_h_end)
            sh0 = int(stripe.source_h_start)
            sh1 = int(stripe.source_h_end)
            h_src = int(sh1 - sh0)
            output_starts = tuple(range(0, int(spec.c), int(bounded_c)))
            family_templates: list[FamilyTemplateBank] = []
            prepared_plaintexts: list[PreparedPlaintext] = []
            raw_steps: list[LinearTransformStep] = []
            output_regions: list[TensorRegion] = []
            output_active_counts: list[int] = []
            target_index = 0
            shared_rotation_group_id = (
                f"{spec.family_label}_full_shared_inter_hsplit"
                f":i{int(in_start)}_{int(in_end)}:h{int(th0)}_{int(th1)}"
            )
            block_inputs: dict[str, PlainCipherTensor] | None = None
            for out_start in output_starts:
                out_end = int(min(int(spec.c), int(out_start + int(bounded_c))))
                local_spec = R34InterGroupBlockSpec(
                    case_name=f"{spec.family_label}_o{int(out_start)}_{int(out_end)}_i{int(in_start)}_{int(in_end)}_h{int(th0)}_{int(th1)}",
                    family_label=str(spec.family_label),
                    c=int(out_end - out_start),
                    h=int(h_src),
                    w=int(spec.w),
                    gap=int(spec.gap),
                )
                local_plan, local_inputs, _local_reference = build_r34_inter_group_local_plan(
                    spec=local_spec,
                    x=x[int(in_start): int(in_end), int(sh0): int(sh1), :],
                    weight=weight[int(out_start): int(out_end), int(in_start): int(in_end)],
                )
                if block_inputs is None:
                    block_inputs = dict(local_inputs)
                family_templates.extend(local_plan.family_templates)
                prepared_plaintexts.extend(local_plan.prepared_plaintexts)
                for local_target, local_region in enumerate(local_plan.output_regions):
                    region = TensorRegion(
                        c_start=int(out_start + int(local_region.c_start)),
                        c_end=int(out_start + int(local_region.c_end)),
                        h_start=0,
                        h_end=int(h_src),
                        w_start=0,
                        w_end=int(spec.w),
                    )
                    output_regions.append(region)
                    output_active_counts.append(int(local_plan.output_active_slot_counts[int(local_target)]))
                    bank_id = f"{spec.family_label}_o{int(out_start)}_{int(out_end)}_surface{int(local_target)}_target{int(target_index)}"
                    raw_steps.append(
                        _retarget_step_for_shared_output(
                            local_plan.linear_transform_steps[int(local_target)],
                            target_index=int(target_index),
                            active_slot_count=int(local_plan.output_active_slot_counts[int(local_target)]),
                            rotation_group_id=str(shared_rotation_group_id),
                            rotation_cost_owner=bool(int(target_index) == 0),
                            bank_id=str(bank_id),
                        )
                    )
                    target_index += 1
            if block_inputs is None:
                raise RuntimeError("failed to build inter-group block inputs")
            collapsed_steps = _collapse_native_shared_multi_output_steps(
                raw_steps,
                step_prefix=f"{spec.family_label}_collapsed_i{int(in_start)}_{int(in_end)}_h{int(th0)}_{int(th1)}",
            )
            block_plan = ConvSchemePlan(
                case_name=f"{spec.family_label}_full_shared_inter_hsplit_i{int(in_start)}_{int(in_end)}_h{int(th0)}_{int(th1)}",
                ring_slot_count=int(RING_SLOT_COUNT),
                output_regions=tuple(output_regions),
                output_active_slot_counts=tuple(output_active_counts),
                family_templates=tuple(family_templates),
                prepared_plaintexts=tuple(prepared_plaintexts),
                linear_transform_steps=tuple(collapsed_steps),
                expected_cost=ExecutionStats(),
                evidence_kind="r34_generalized_inter_group_python",
                notes=("full generalized inter-hsplit block",),
            )
            block_reference = F.conv2d(
                x[int(in_start): int(in_end), int(sh0): int(sh1), :].unsqueeze(0),
                weight[:, int(in_start): int(in_end)],
                bias=None,
                stride=1,
                padding=1,
            )[0]
            blocks.append(
                {
                    "in_range": (int(in_start), int(in_end)),
                    "target_h_range": (int(th0), int(th1)),
                    "source_h_range": (int(sh0), int(sh1)),
                    "plan": block_plan,
                    "inputs": block_inputs,
                    "reference": block_reference,
                }
            )
    return {
        "family_label": str(spec.family_label),
        "blocks": blocks,
        "reference": reference,
        "bounded": dict(bounded),
        "notes": (
            f"family_label={spec.family_label}",
            "python prototype for generalized inter-group hsplit stage1/stage2 path",
        ),
    }


class R34PythonInterGroupSameShapeRuntimeExecutor:
    def __init__(self, *, module: Any, spec: R34SameShapeStageSpec, output_node_id: str) -> None:
        self.module = module
        self.spec = spec
        self.output_node_id = str(output_node_id)
        self.assets: dict[str, Any] | None = None
        self.block_groups: list[dict[str, Any]] = []
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

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        return str(type(scheme.backend).__name__) == "PythonBackend"

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def _validate_module(self) -> None:
        if tuple(int(v) for v in tuple(getattr(self.module, "on_weight").shape)) != tuple(int(v) for v in self.spec.weight_shape):
            raise RuntimeError(f"{self.output_node_id} weight shape does not match {self.spec.family_label}")
        if tuple(int(v) for v in getattr(self.module, "input_shape", torch.Size())[1:]) != (int(self.spec.c), int(self.spec.h), int(self.spec.w)):
            raise RuntimeError(f"{self.output_node_id} input_shape does not match {self.spec.family_label}")
        if tuple(int(v) for v in getattr(self.module, "output_shape", torch.Size())[1:]) != (int(self.spec.c), int(self.spec.h), int(self.spec.w)):
            raise RuntimeError(f"{self.output_node_id} output_shape does not match {self.spec.family_label}")

    def compile(self, scheme: Any) -> None:
        if self.block_groups:
            return
        if not self.supports_scheme(scheme):
            raise RuntimeError("R34 Python inter-group runtime requires the Python backend")
        self._validate_module()
        prepare_started = time.time()
        self.assets = build_r34_same_shape_generalized_inter_group_assets(
            spec=self.spec,
            weight_override=getattr(self.module, "on_weight"),
        )
        self.last_runtime_timing = {
            "prepare_plans_s": float(time.time() - prepare_started),
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        level = self._level(scheme)
        prepare_transforms_started = time.time()
        groups: list[dict[str, Any]] = []
        for index, block in enumerate(self.assets["blocks"]):
            plan = block["plan"]
            transforms, _bank_ids = transforms_from_conv_scheme_plan(
                plan,
                level=int(level),
                scheme=scheme,
                bank_count=len(plan.output_regions),
            )
            groups.append(
                {
                    "index": int(index),
                    "plan": plan,
                    "group": UnifiedTransformGroup(transforms),
                    "in_range": tuple(int(v) for v in block["in_range"]),
                    "target_h_range": tuple(int(v) for v in block["target_h_range"]),
                    "source_h_range": tuple(int(v) for v in block["source_h_range"]),
                }
            )
        self.last_runtime_timing["prepare_transforms_s"] = float(time.time() - prepare_transforms_started)
        compile_started = time.time()
        for entry in groups:
            entry["group"].compile_unified(scheme.backend)
        self.block_groups = groups
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)

    def _decode_input_clear(self, source_ct: Any) -> torch.Tensor:
        flat = torch.cat(
            [source_ct.scheme.backend._ciphertexts[int(ct_id)].values.detach().clone().to(dtype=torch.float32) for ct_id in source_ct.ids],
            dim=0,
        )
        total = int(torch.Size(getattr(self.module, "fhe_input_shape")).numel())
        on_shape = tuple(int(v) for v in getattr(self.module, "fhe_input_shape"))
        packed = flat[: int(total)].reshape(on_shape)
        clear = packing._demultiplex(
            packed,
            int(self.spec.gap),
            int(self.spec.c),
            int(self.spec.h),
            int(self.spec.w),
        )[0]
        return clear.to(dtype=torch.float32)

    def _encrypt_complex_source(self, left: torch.Tensor, right: torch.Tensor, scheme: Any, level: int) -> CipherTensor:
        return scheme.encrypt(scheme.encode(left.to(dtype=torch.float32), level)) + scheme.encrypt(
            scheme.encode(right.to(dtype=torch.float32), level)
        ).mul_imaginary_unit(+1, in_place=False)

    def _decode_real_flat(self, ct: CipherTensor) -> torch.Tensor:
        decoded = ct.decrypt().decode().detach().cpu()
        if torch.is_complex(decoded):
            decoded = decoded.real
        return decoded.to(dtype=torch.float32).flatten()

    def _encode_output(self, output: torch.Tensor, scheme: Any, level: int) -> CipherTensor:
        packed = packing.multiplex(output.unsqueeze(0), int(self.spec.gap)).squeeze(0)
        target = torch.zeros(tuple(int(v) for v in getattr(self.module, "fhe_output_shape")[1:]), dtype=torch.float32)
        target[: packed.shape[0], : packed.shape[1], : packed.shape[2]] = packed
        flat = target.flatten()
        ids: list[int] = []
        slots = int(scheme.params.get_slots())
        for start in range(0, int(flat.numel()), int(slots)):
            block = flat[int(start) : int(min(int(flat.numel()), int(start + int(slots))))]
            padded = torch.zeros((int(slots),), dtype=torch.float32)
            padded[: int(block.numel())] = block
            ct = scheme.encrypt(scheme.encode(padded, int(level)))
            ids.append(int(ct.ids[0]))
            ct.ids = []
        return CipherTensor(
            scheme,
            ids,
            getattr(self.module, "output_shape"),
            getattr(self.module, "fhe_output_shape"),
        )

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.compile(scheme)
        level = int(source_ct.level()) if hasattr(source_ct, "level") else self._level(scheme)
        clear_input = self._decode_input_clear(source_ct)
        output = torch.zeros((int(self.spec.c), int(self.spec.h), int(self.spec.w)), dtype=torch.float32)

        evaluate_started = time.time()
        partitions: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for entry in self.block_groups:
            partitions.setdefault(tuple(int(v) for v in entry["in_range"]), []).append(entry)
        for in_range, entries in sorted(partitions.items()):
            in_start, in_end = (int(v) for v in in_range)
            entries = sorted(entries, key=lambda item: tuple(int(v) for v in item["target_h_range"]))
            partition_input = clear_input[int(in_start) : int(in_end), :, :]
            lane_c = int((int(in_end) - int(in_start)) // 2)
            left_partition = partition_input[: int(lane_c)]
            right_partition = partition_input[int(lane_c) :]
            prepared: list[dict[str, Any]] = []
            for entry in entries:
                th0, th1 = (int(v) for v in entry["target_h_range"])
                sh0, sh1 = (int(v) for v in entry["source_h_range"])
                section_spec = build_section_extract_plan(
                    family_label=str(self.spec.family_label),
                    c=int(lane_c),
                    w=int(self.spec.w),
                    gap=int(self.spec.gap),
                    input_h=int(self.spec.h),
                    kernel=3,
                    stride=1,
                    pad=1,
                    target_h_range=(int(th0), int(th1)),
                    source_h_range=(int(sh0), int(sh1)),
                    align_source_start_to_stride=False,
                )
                left_core = build_core_section_tensor(
                    left_partition,
                    plan=section_spec,
                )
                right_core = build_core_section_tensor(
                    right_partition,
                    plan=section_spec,
                )
                prepared.append(
                    {
                        "entry": entry,
                        "section_spec": section_spec,
                        "crop_range": (int(section_spec.crop_start), int(section_spec.crop_end)),
                        "left_core_ct": encode_section_tensor(left_core, scheme=scheme, level=int(level), plan=section_spec),
                        "right_core_ct": encode_section_tensor(right_core, scheme=scheme, level=int(level), plan=section_spec),
                    }
                )
            for index, item in enumerate(prepared):
                prev_item = prepared[int(index - 1)] if int(index) > 0 else None
                next_item = prepared[int(index + 1)] if int(index + 1) < len(prepared) else None
                left_filled = extract_section_ciphertext(
                    center_ct=item["left_core_ct"],
                    prev_ct=None if prev_item is None else prev_item["left_core_ct"],
                    next_ct=None if next_item is None else next_item["left_core_ct"],
                    prev_plan=None if prev_item is None else prev_item["section_spec"],
                    next_plan=None if next_item is None else next_item["section_spec"],
                    scheme=scheme,
                    level=int(level),
                    plan=item["section_spec"],
                )
                right_filled = extract_section_ciphertext(
                    center_ct=item["right_core_ct"],
                    prev_ct=None if prev_item is None else prev_item["right_core_ct"],
                    next_ct=None if next_item is None else next_item["right_core_ct"],
                    prev_plan=None if prev_item is None else prev_item["section_spec"],
                    next_plan=None if next_item is None else next_item["section_spec"],
                    scheme=scheme,
                    level=int(level),
                    plan=item["section_spec"],
                )
                ct_complex = left_filled + right_filled.mul_imaginary_unit(+1, in_place=False)
                plan = item["entry"]["plan"]
                group = item["entry"]["group"]
                th0, th1 = (int(v) for v in item["entry"]["target_h_range"])
                crop_start, crop_end = (int(v) for v in item["crop_range"])
                output_ids = group.evaluate_unified(int(ct_complex.ids[0]), scheme.backend)
                for region, output_id in zip(plan.output_regions, output_ids):
                    raw = CipherTensor(
                        scheme,
                        [int(output_id)],
                        torch.Size([1, int(scheme.params.get_slots())]),
                        torch.Size([1, int(scheme.params.get_slots())]),
                    )
                    real = (raw + raw.conjugate(in_place=False)) * 0.5
                    decoded = self._decode_real_flat(real)
                    c0 = int(region.c_start)
                    c1 = int(region.c_end)
                    h_len = int(region.h_end - region.h_start)
                    on_c = max(1, (int(c1 - c0) + int(self.spec.gap * self.spec.gap) - 1) // int(self.spec.gap * self.spec.gap))
                    on_h = int(h_len * self.spec.gap)
                    on_w = int(self.spec.w * self.spec.gap)
                    packed_size = int(on_c * on_h * on_w)
                    demux = packing._demultiplex(
                        decoded[: int(packed_size)].reshape(1, int(on_c), int(on_h), int(on_w)),
                        int(self.spec.gap),
                        int(c1 - c0),
                        int(h_len),
                        int(self.spec.w),
                    )[0]
                    output[int(c0) : int(c1), int(th0) : int(th1), :] += demux[:, int(crop_start) : int(crop_end), :]
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)

        postprocess_started = time.time()
        bias = getattr(self.module, "on_bias", None)
        if bias is not None:
            output = output + bias.detach().to(dtype=torch.float32).view(int(self.spec.c), 1, 1)
        out_ct = self._encode_output(output, scheme, level)
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {self.output_node_id: out_ct}


class R34PythonTransitionFlowRuntimeExecutor:
    def __init__(
        self,
        *,
        conv_module: Any,
        shortcut_module: Any,
        family_label: str,
        kernel_policy: str,
        output_node_ids: tuple[str, str],
    ) -> None:
        self.conv_module = conv_module
        self.shortcut_module = shortcut_module
        self.family_label = str(family_label)
        self.kernel_policy = str(kernel_policy)
        self.output_node_ids = tuple(str(v) for v in output_node_ids)
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
        self._geometry: dict[str, Any] | None = None

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        return str(type(scheme.backend).__name__) == "PythonBackend"

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def _transition_partitions(self) -> tuple[Any, ...]:
        c_in = int(self.conv_module.input_shape[1])
        gap_in = int(self.conv_module.input_gap)
        if str(self.kernel_policy) == "inter_group_hybrid":
            return hybrid_pair_channel_partitions(c=int(c_in), gap=int(gap_in))
        return (
            type("FullPartition", (), {"c_start": 0, "c_end": int(c_in), "group_start": 0, "group_end": int(source_group_count(c=int(c_in), gap=int(gap_in)))})(),
        )

    def compile(self, scheme: Any) -> None:
        if self._geometry is not None:
            return
        if not self.supports_scheme(scheme):
            raise RuntimeError("R34 Python transition runtime requires the Python backend")
        prepare_started = time.time()
        partitions = []
        c_in = int(self.conv_module.input_shape[1])
        h_in = int(self.conv_module.input_shape[2])
        w_in = int(self.conv_module.input_shape[3])
        gap_in = int(self.conv_module.input_gap)
        for partition in self._transition_partitions():
            part_c = int(partition.c_end) - int(partition.c_start)
            if str(self.kernel_policy) == "inter_group_hybrid":
                stripe_c = max(1, int(part_c // 2))
            else:
                stripe_c = int(part_c)
            stripes = height_stripes_for_partition(
                c=int(stripe_c),
                h=int(h_in),
                w=int(w_in),
                gap=int(gap_in),
                kernel=int(self.conv_module.kernel_size[0]),
                stride=int(self.conv_module.stride[0]),
                pad=int(self.conv_module.padding[0]),
                max_slots=int(RING_SLOT_COUNT),
            )
            partitions.append(
                {
                    "c_start": int(partition.c_start),
                    "c_end": int(partition.c_end),
                    "stripes": stripes,
                }
            )
        self._geometry = {"partitions": partitions}
        self.last_runtime_timing = {
            "prepare_plans_s": float(time.time() - prepare_started),
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.compile_count += 1

    def _decode_input_clear(self, source_ct: Any) -> torch.Tensor:
        flat = torch.cat(
            [source_ct.scheme.backend._ciphertexts[int(ct_id)].values.detach().clone().to(dtype=torch.float32) for ct_id in source_ct.ids],
            dim=0,
        )
        total = int(torch.Size(getattr(self.conv_module, "fhe_input_shape")).numel())
        on_shape = tuple(int(v) for v in getattr(self.conv_module, "fhe_input_shape"))
        packed = flat[: int(total)].reshape(on_shape)
        clear = packing._demultiplex(
            packed,
            int(self.conv_module.input_gap),
            int(self.conv_module.input_shape[1]),
            int(self.conv_module.input_shape[2]),
            int(self.conv_module.input_shape[3]),
        )[0]
        return clear.to(dtype=torch.float32)

    def _encode_output(self, output: torch.Tensor, module: Any, scheme: Any, level: int) -> CipherTensor:
        packed = packing.multiplex(output.unsqueeze(0), int(module.output_gap)).squeeze(0)
        target = torch.zeros(tuple(int(v) for v in getattr(module, "fhe_output_shape")[1:]), dtype=torch.float32)
        target[: packed.shape[0], : packed.shape[1], : packed.shape[2]] = packed
        flat = target.flatten()
        ids: list[int] = []
        slots = int(scheme.params.get_slots())
        for start in range(0, int(flat.numel()), int(slots)):
            block = flat[int(start) : int(min(int(flat.numel()), int(start + int(slots))))]
            padded = torch.zeros((int(slots),), dtype=torch.float32)
            padded[: int(block.numel())] = block
            ct = scheme.encrypt(scheme.encode(padded, int(level)))
            ids.append(int(ct.ids[0]))
            ct.ids = []
        return CipherTensor(
            scheme,
            ids,
            getattr(module, "output_shape"),
            getattr(module, "fhe_output_shape"),
        )

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.compile(scheme)
        level = int(source_ct.level()) if hasattr(source_ct, "level") else self._level(scheme)
        clear_input = self._decode_input_clear(source_ct)
        conv_out = torch.zeros(tuple(int(v) for v in self.conv_module.output_shape[1:]), dtype=torch.float32)
        shortcut_out = torch.zeros(tuple(int(v) for v in self.shortcut_module.output_shape[1:]), dtype=torch.float32)

        evaluate_started = time.time()
        assert self._geometry is not None
        for partition in self._geometry["partitions"]:
            c0 = int(partition["c_start"])
            c1 = int(partition["c_end"])
            part_input = clear_input[int(c0) : int(c1)]
            if str(self.kernel_policy) == "inter_group_hybrid":
                lane_c = max(1, int(ceil_div(int(c1 - c0), 2)))
                lane_inputs = [part_input[: int(lane_c)]]
                if int(c1 - c0) > int(lane_c):
                    lane_inputs.append(part_input[int(lane_c) : int(c1 - c0)])
            else:
                lane_inputs = [part_input]
            prepared_lanes: list[list[dict[str, Any]]] = [[] for _ in range(len(lane_inputs))]
            shared_ranges: list[dict[str, Any]] = []
            for stripe in partition["stripes"]:
                th0, th1 = (int(stripe.target_h_start), int(stripe.target_h_end))
                sh0, sh1 = (int(stripe.source_h_start), int(stripe.source_h_end))
                conv_crop_ref = None
                shortcut_crop_ref = None
                for lane_index, lane_input in enumerate(lane_inputs):
                    extract_plan = build_section_extract_plan(
                        family_label=str(self.family_label),
                        c=int(lane_input.shape[0]),
                        w=int(self.conv_module.input_shape[3]),
                        gap=int(self.conv_module.input_gap),
                        input_h=int(self.conv_module.input_shape[2]),
                        kernel=int(self.conv_module.kernel_size[0]),
                        stride=int(self.conv_module.stride[0]),
                        pad=int(self.conv_module.padding[0]),
                        target_h_range=(int(th0), int(th1)),
                        source_h_range=(int(sh0), int(sh1)),
                        align_source_start_to_stride=bool(int(self.conv_module.stride[0]) > 1),
                    )
                    shortcut_plan = build_section_extract_plan(
                        family_label=str(self.family_label),
                        c=int(lane_input.shape[0]),
                        w=int(self.shortcut_module.input_shape[3]),
                        gap=int(self.shortcut_module.input_gap),
                        input_h=int(self.shortcut_module.input_shape[2]),
                        kernel=int(self.shortcut_module.kernel_size[0]),
                        stride=int(self.shortcut_module.stride[0]),
                        pad=int(self.shortcut_module.padding[0]),
                        target_h_range=(int(th0), int(th1)),
                        source_h_range=(int(extract_plan.source_h_start), int(extract_plan.source_h_end)),
                        align_source_start_to_stride=bool(int(self.shortcut_module.stride[0]) > 1),
                    )
                    conv_crop_ref = (
                        int(extract_plan.crop_start),
                        int(extract_plan.crop_end),
                    ) if conv_crop_ref is None else conv_crop_ref
                    shortcut_crop_ref = (
                        int(shortcut_plan.crop_start),
                        int(shortcut_plan.crop_end),
                    ) if shortcut_crop_ref is None else shortcut_crop_ref
                    core_section = build_core_section_tensor(lane_input, plan=extract_plan)
                    prepared_lanes[int(lane_index)].append(
                        {
                            "extract_plan": extract_plan,
                            "core_ct": encode_section_tensor(core_section, scheme=scheme, level=int(level), plan=extract_plan),
                        }
                    )
                shared_ranges.append(
                    {
                        "target_h_range": (int(th0), int(th1)),
                        "conv_crop_range": conv_crop_ref,
                        "shortcut_crop_range": shortcut_crop_ref,
                    }
                )
            for index, shared in enumerate(shared_ranges):
                filled_lanes: list[torch.Tensor] = []
                for lane_items in prepared_lanes:
                    item = lane_items[int(index)]
                    prev_item = lane_items[int(index - 1)] if int(index) > 0 else None
                    next_item = lane_items[int(index + 1)] if int(index + 1) < len(lane_items) else None
                    filled_ct = extract_section_ciphertext(
                        center_ct=item["core_ct"],
                        prev_ct=None if prev_item is None else prev_item["core_ct"],
                        next_ct=None if next_item is None else next_item["core_ct"],
                        prev_plan=None if prev_item is None else prev_item["extract_plan"],
                        next_plan=None if next_item is None else next_item["extract_plan"],
                        scheme=scheme,
                        level=int(level),
                        plan=item["extract_plan"],
                    )
                    filled_lanes.append(decode_section_tensor(filled_ct, scheme=scheme, plan=item["extract_plan"]))
                local_source = torch.cat(filled_lanes, dim=0)
                conv_local = F.conv2d(
                    local_source.unsqueeze(0),
                    self.conv_module.on_weight.detach().to(dtype=torch.float32)[:, int(c0) : int(c1)],
                    bias=None,
                    stride=tuple(int(v) for v in self.conv_module.stride),
                    padding=tuple(int(v) for v in self.conv_module.padding),
                    dilation=tuple(int(v) for v in self.conv_module.dilation),
                    groups=int(self.conv_module.groups),
                )[0]
                shortcut_local = F.conv2d(
                    local_source.unsqueeze(0),
                    self.shortcut_module.on_weight.detach().to(dtype=torch.float32)[:, int(c0) : int(c1)],
                    bias=None,
                    stride=tuple(int(v) for v in self.shortcut_module.stride),
                    padding=tuple(int(v) for v in self.shortcut_module.padding),
                    dilation=tuple(int(v) for v in self.shortcut_module.dilation),
                    groups=int(self.shortcut_module.groups),
                )[0]
                th0, th1 = (int(v) for v in shared["target_h_range"])
                conv_crop_start, conv_crop_end = (int(v) for v in shared["conv_crop_range"])
                shortcut_crop_start, shortcut_crop_end = (int(v) for v in shared["shortcut_crop_range"])
                conv_out[:, int(th0) : int(th1), :] += conv_local[:, int(conv_crop_start) : int(conv_crop_end), :]
                shortcut_out[:, int(th0) : int(th1), :] += shortcut_local[:, int(shortcut_crop_start) : int(shortcut_crop_end), :]
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)

        postprocess_started = time.time()
        conv_bias = getattr(self.conv_module, "on_bias", None)
        shortcut_bias = getattr(self.shortcut_module, "on_bias", None)
        if conv_bias is not None:
            conv_out = conv_out + conv_bias.detach().to(dtype=torch.float32).view(int(conv_out.shape[0]), 1, 1)
        if shortcut_bias is not None:
            shortcut_out = shortcut_out + shortcut_bias.detach().to(dtype=torch.float32).view(int(shortcut_out.shape[0]), 1, 1)
        conv_ct = self._encode_output(conv_out, self.conv_module, scheme, level)
        shortcut_ct = self._encode_output(shortcut_out, self.shortcut_module, scheme, level)
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {
            str(self.output_node_ids[0]): conv_ct,
            str(self.output_node_ids[1]): shortcut_ct,
        }


class R34PythonSingleFlowRuntimeExecutor:
    def __init__(
        self,
        *,
        module: Any,
        family_label: str,
        kernel_policy: str,
        output_node_id: str,
    ) -> None:
        self.module = module
        self.family_label = str(family_label)
        self.kernel_policy = str(kernel_policy)
        self.output_node_id = str(output_node_id)
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
        self._geometry: dict[str, Any] | None = None

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        return str(type(scheme.backend).__name__) == "PythonBackend"

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def _partitions(self) -> tuple[Any, ...]:
        c_in = int(self.module.input_shape[1])
        gap_in = int(self.module.input_gap)
        if str(self.kernel_policy) == "inter_group_hybrid":
            return hybrid_pair_channel_partitions(c=int(c_in), gap=int(gap_in))
        return (
            type(
                "FullPartition",
                (),
                {
                    "c_start": 0,
                    "c_end": int(c_in),
                    "group_start": 0,
                    "group_end": int(source_group_count(c=int(c_in), gap=int(gap_in))),
                },
            )(),
        )

    def compile(self, scheme: Any) -> None:
        if self._geometry is not None:
            return
        if not self.supports_scheme(scheme):
            raise RuntimeError("R34 Python single-flow runtime requires the Python backend")
        prepare_started = time.time()
        partitions = []
        c_in = int(self.module.input_shape[1])
        h_in = int(self.module.input_shape[2])
        h_out = int(self.module.output_shape[2])
        w_in = int(self.module.input_shape[3])
        gap_in = int(self.module.input_gap)
        for partition in self._partitions():
            part_c = int(partition.c_end) - int(partition.c_start)
            if str(self.kernel_policy) == "inter_group_hybrid":
                stripe_c = max(1, int(ceil_div(int(part_c), 2)))
            else:
                stripe_c = int(part_c)
            max_source_h = int(
                max_source_h_for_channels(
                    c=int(stripe_c),
                    w=int(w_in),
                    gap=int(gap_in),
                    max_slots=int(RING_SLOT_COUNT),
                )
            )
            target_tile_h = (
                int(h_out)
                if int(max_source_h) >= int(h_in)
                else int(
                    target_h_from_source_h(
                        source_h=int(max_source_h),
                        kernel=int(self.module.kernel_size[0]),
                        stride=int(self.module.stride[0]),
                    )
                )
            )
            stripes = []
            target_h = 0
            while int(target_h) < int(h_out):
                th0 = int(target_h)
                th1 = min(int(h_out), int(th0 + int(target_tile_h)))
                req0, req1 = source_h_range_for_target(
                    target_h_start=int(th0),
                    target_h_end=int(th1),
                    input_h=int(h_in),
                    kernel=int(self.module.kernel_size[0]),
                    stride=int(self.module.stride[0]),
                    pad=int(self.module.padding[0]),
                )
                sh0, sh1 = extend_h_range_to_length(
                    required_start=int(req0),
                    required_end=int(req1),
                    desired_len=int(max_source_h),
                    limit=int(h_in),
                )
                stripes.append(
                    type(
                        "Stripe",
                        (),
                        {
                            "target_h_start": int(th0),
                            "target_h_end": int(th1),
                            "source_h_start": int(sh0),
                            "source_h_end": int(sh1),
                        },
                    )()
                )
                target_h = int(th1)
            partitions.append(
                {
                    "c_start": int(partition.c_start),
                    "c_end": int(partition.c_end),
                    "stripes": tuple(stripes),
                }
            )
        self._geometry = {"partitions": partitions}
        self.last_runtime_timing = {
            "prepare_plans_s": float(time.time() - prepare_started),
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.compile_count += 1

    def _decode_input_clear(self, source_ct: Any) -> torch.Tensor:
        flat = torch.cat(
            [source_ct.scheme.backend._ciphertexts[int(ct_id)].values.detach().clone().to(dtype=torch.float32) for ct_id in source_ct.ids],
            dim=0,
        )
        total = int(torch.Size(getattr(self.module, "fhe_input_shape")).numel())
        on_shape = tuple(int(v) for v in getattr(self.module, "fhe_input_shape"))
        packed = flat[: int(total)].reshape(on_shape)
        clear = packing._demultiplex(
            packed,
            int(self.module.input_gap),
            int(self.module.input_shape[1]),
            int(self.module.input_shape[2]),
            int(self.module.input_shape[3]),
        )[0]
        return clear.to(dtype=torch.float32)

    def _encode_output(self, output: torch.Tensor, scheme: Any, level: int) -> CipherTensor:
        packed = packing.multiplex(output.unsqueeze(0), int(self.module.output_gap)).squeeze(0)
        target = torch.zeros(tuple(int(v) for v in getattr(self.module, "fhe_output_shape")[1:]), dtype=torch.float32)
        target[: packed.shape[0], : packed.shape[1], : packed.shape[2]] = packed
        flat = target.flatten()
        ids: list[int] = []
        slots = int(scheme.params.get_slots())
        for start in range(0, int(flat.numel()), int(slots)):
            block = flat[int(start) : int(min(int(flat.numel()), int(start + int(slots))))]
            padded = torch.zeros((int(slots),), dtype=torch.float32)
            padded[: int(block.numel())] = block
            ct = scheme.encrypt(scheme.encode(padded, int(level)))
            ids.append(int(ct.ids[0]))
            ct.ids = []
        return CipherTensor(
            scheme,
            ids,
            getattr(self.module, "output_shape"),
            getattr(self.module, "fhe_output_shape"),
        )

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.compile(scheme)
        level = int(source_ct.level()) if hasattr(source_ct, "level") else self._level(scheme)
        clear_input = self._decode_input_clear(source_ct)
        output = torch.zeros(tuple(int(v) for v in self.module.output_shape[1:]), dtype=torch.float32)

        weight = self.module.on_weight.detach().to(dtype=torch.float32)
        groups = int(getattr(self.module, "groups", 1))
        in_channels = int(self.module.input_shape[1])
        out_channels = int(self.module.output_shape[1])

        evaluate_started = time.time()
        assert self._geometry is not None
        for partition in self._geometry["partitions"]:
            c0 = int(partition["c_start"])
            c1 = int(partition["c_end"])
            part_input = clear_input[int(c0) : int(c1)]
            if str(self.kernel_policy) == "inter_group_hybrid":
                lane_c = max(1, int(ceil_div(int(c1 - c0), 2)))
                lane_inputs = [part_input[: int(lane_c)]]
                if int(c1 - c0) > int(lane_c):
                    lane_inputs.append(part_input[int(lane_c) : int(c1 - c0)])
                prepared_lanes: list[list[dict[str, Any]]] = [[] for _ in range(len(lane_inputs))]
                shared_ranges: list[dict[str, Any]] = []
                for stripe in partition["stripes"]:
                    th0, th1 = (int(stripe.target_h_start), int(stripe.target_h_end))
                    sh0, sh1 = (int(stripe.source_h_start), int(stripe.source_h_end))
                    crop_range_ref = None
                    section_specs: list[dict[str, Any]] = []
                    for lane_index, lane_input in enumerate(lane_inputs):
                        section_spec = build_section_extract_plan(
                            family_label=str(self.family_label),
                            c=int(lane_input.shape[0]),
                            w=int(self.module.input_shape[3]),
                            gap=int(self.module.input_gap),
                            input_h=int(self.module.input_shape[2]),
                            kernel=int(self.module.kernel_size[0]),
                            stride=int(self.module.stride[0]),
                            pad=int(self.module.padding[0]),
                            target_h_range=(int(th0), int(th1)),
                            source_h_range=(int(sh0), int(sh1)),
                            align_source_start_to_stride=bool(int(self.module.stride[0]) > 1),
                        )
                        crop_range = (int(section_spec.crop_start), int(section_spec.crop_end))
                        crop_range_ref = crop_range if crop_range_ref is None else crop_range_ref
                        core_section = build_core_section_tensor(
                            lane_input,
                            plan=section_spec,
                        )
                        prepared_lanes[int(lane_index)].append(
                            {
                                "section_spec": section_spec,
                                "core_ct": encode_section_tensor(core_section, scheme=scheme, level=int(level), plan=section_spec),
                            }
                        )
                        section_specs.append({"section_spec": section_spec})
                    shared_ranges.append(
                        {
                            "target_h_range": (int(th0), int(th1)),
                            "crop_range": crop_range_ref,
                            "section_specs": section_specs,
                        }
                    )
                for index, shared in enumerate(shared_ranges):
                    filled_lanes: list[torch.Tensor] = []
                    for lane_index, lane_items in enumerate(prepared_lanes):
                        item = lane_items[int(index)]
                        prev_item = lane_items[int(index - 1)] if int(index) > 0 else None
                        next_item = lane_items[int(index + 1)] if int(index + 1) < len(lane_items) else None
                        filled_ct = extract_section_ciphertext(
                            center_ct=item["core_ct"],
                            prev_ct=None if prev_item is None else prev_item["core_ct"],
                            next_ct=None if next_item is None else next_item["core_ct"],
                            prev_plan=None if prev_item is None else prev_item["section_spec"],
                            next_plan=None if next_item is None else next_item["section_spec"],
                            scheme=scheme,
                            level=int(level),
                            plan=item["section_spec"],
                        )
                        filled_lanes.append(decode_section_tensor(filled_ct, scheme=scheme, plan=item["section_spec"]))
                    local_source = torch.cat(filled_lanes, dim=0)
                    crop_start, crop_end = (int(v) for v in shared["crop_range"])
                    th0, th1 = (int(v) for v in shared["target_h_range"])
                    if int(groups) == 1:
                        local_out = F.conv2d(
                            local_source.unsqueeze(0),
                            weight[:, int(c0) : int(c1)],
                            bias=None,
                            stride=tuple(int(v) for v in self.module.stride),
                            padding=tuple(int(v) for v in self.module.padding),
                            dilation=tuple(int(v) for v in self.module.dilation),
                            groups=1,
                        )[0]
                        output[:, int(th0) : int(th1), :] += local_out[:, int(crop_start) : int(crop_end), :]
                    elif int(groups) == int(in_channels) == int(out_channels):
                        part_weight = weight[int(c0) : int(c1)]
                        local_out = F.conv2d(
                            local_source.unsqueeze(0),
                            part_weight,
                            bias=None,
                            stride=tuple(int(v) for v in self.module.stride),
                            padding=tuple(int(v) for v in self.module.padding),
                            dilation=tuple(int(v) for v in self.module.dilation),
                            groups=int(c1 - c0),
                        )[0]
                        output[int(c0) : int(c1), int(th0) : int(th1), :] += local_out[:, int(crop_start) : int(crop_end), :]
                    else:
                        raise RuntimeError(
                            f"{self.family_label} unsupported grouped geometry: groups={groups}, in_channels={in_channels}, out_channels={out_channels}"
                        )
            else:
                part_out_full = F.conv2d(
                    part_input.unsqueeze(0),
                    weight[:, int(c0) : int(c1)] if int(groups) == 1 else weight[int(c0) : int(c1)],
                    bias=None,
                    stride=tuple(int(v) for v in self.module.stride),
                    padding=tuple(int(v) for v in self.module.padding),
                    dilation=tuple(int(v) for v in self.module.dilation),
                    groups=1 if int(groups) == 1 else int(c1 - c0),
                )[0]
                for stripe in partition["stripes"]:
                    th0 = int(stripe.target_h_start)
                    th1 = int(stripe.target_h_end)
                    if int(groups) == 1:
                        output[:, int(th0) : int(th1), :] += part_out_full[:, int(th0) : int(th1), :]
                    else:
                        output[int(c0) : int(c1), int(th0) : int(th1), :] += part_out_full[:, int(th0) : int(th1), :]
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)

        postprocess_started = time.time()
        bias = getattr(self.module, "on_bias", None)
        if bias is not None:
            output = output + bias.detach().to(dtype=torch.float32).view(int(output.shape[0]), 1, 1)
        out_ct = self._encode_output(output, scheme, level)
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {self.output_node_id: out_ct}
