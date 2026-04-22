from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

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
from .r34_orion_same_shape import R34SameShapeStageSpec, _idx_chw_gap_tensor


RING_SLOT_COUNT = 32768


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


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _packed_active_slots(c: int, h: int, w: int, gap: int) -> int:
    phase = max(1, int(gap) * int(gap))
    groups = _ceil_div(int(c), int(phase))
    return int(groups) * int(h) * int(gap) * int(w) * int(gap)


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
    phase = max(1, int(spec.gap) * int(spec.gap))
    source_groups = _ceil_div(int(spec.c), int(phase))
    max_h = max(1, min(int(spec.h), int(RING_SLOT_COUNT) // max(1, int(spec.w) * int(spec.gap) * int(spec.gap))))
    groups_per_surface = int(RING_SLOT_COUNT // max(1, int(max_h) * int(spec.w) * int(spec.gap) * int(spec.gap)))
    groups_per_surface = max(1, min(int(groups_per_surface), max(1, int(source_groups // 2))))
    bounded_c = int(2 * int(groups_per_surface) * int(phase))
    return {
        "source_groups": int(source_groups),
        "groups_per_surface": int(groups_per_surface),
        "bounded_c": int(bounded_c),
        "bounded_h": int(max_h),
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
    active_slots = _packed_active_slots(int(local_c), int(spec.h), int(spec.w), int(spec.gap))
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
    c_pair = int(bounded["bounded_c"])
    bounded_h = int(bounded["bounded_h"])
    h_tile = int(spec.h) if int(bounded_h) >= int(spec.h) else max(1, int(bounded_h) - 3 + 1)
    if weight_override is None:
        weight = torch.randn((int(spec.c), int(spec.c), 3, 3), dtype=torch.float32)
    else:
        weight = weight_override.detach().to(dtype=torch.float32).clone()
    if source_override is None:
        x = torch.randn((int(spec.c), int(spec.h), int(spec.w)), dtype=torch.float32)
    else:
        x = source_override.detach().to(dtype=torch.float32).clone()
    reference = F.conv2d(x.unsqueeze(0), weight, bias=None, stride=1, padding=1)[0]
    h_stripes: list[tuple[int, int, int, int]] = []
    target_h = 0
    while int(target_h) < int(spec.h):
        th0 = int(target_h)
        th1 = min(int(spec.h), int(th0 + int(h_tile)))
        required_source = _source_h_range_for_target(
            target_h_range=(int(th0), int(th1)),
            input_h=int(spec.h),
            kernel=3,
            stride=1,
            pad=1,
        )
        sh0, sh1 = _extend_h_range_to_length(required=required_source, desired_len=int(bounded_h), limit=int(spec.h))
        h_stripes.append((int(th0), int(th1), int(sh0), int(sh1)))
        target_h = int(th1)

    blocks: list[dict[str, Any]] = []
    for in_start in range(0, int(spec.c), int(c_pair)):
        in_end = int(min(int(spec.c), int(in_start + int(c_pair))))
        for th0, th1, sh0, sh1 in h_stripes:
            h_src = int(sh1 - sh0)
            output_starts = tuple(range(0, int(spec.c), int(c_pair)))
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
                out_end = int(min(int(spec.c), int(out_start + int(c_pair))))
                local_spec = R34InterGroupBlockSpec(
                    case_name=f"{spec.family_label}_o{int(out_start)}_{int(out_end)}_i{int(in_start)}_{int(in_end)}_h{int(th0)}_{int(th1)}",
                    family_label=str(spec.family_label),
                    c=int(c_pair),
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
