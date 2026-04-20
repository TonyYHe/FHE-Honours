from __future__ import annotations

from dataclasses import dataclass

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


RING_SLOT_COUNT = 32768


@dataclass(frozen=True)
class R18SameStageSpec:
    stage: str
    c: int
    h: int
    w: int
    gap: int
    c_pair: int
    rotations_per_block: int

    @property
    def surface_c(self) -> int:
        return int(self.c_pair // 2)

    @property
    def bank_count(self) -> int:
        return int(self.c // self.surface_c)

    @property
    def input_pair_count(self) -> int:
        return int(self.c // self.c_pair)


R18_STAGE1_SPEC = R18SameStageSpec("stage1_same", c=64, h=64, w=64, gap=1, c_pair=16, rotations_per_block=20)
R18_STAGE2_SPEC = R18SameStageSpec("stage2_same", c=128, h=32, w=32, gap=2, c_pair=64, rotations_per_block=42)


def _bank_id(spec: R18SameStageSpec, bank_index: int) -> str:
    return f"r18_{spec.stage}_bank_{int(bank_index)}"


def _slot_index(channel: int, h_index: int, w_index: int, *, h: int = 64, w: int = 64, gap: int = 1) -> int:
    g = max(1, int(gap))
    if int(g) == 1:
        return int(channel) * int(h) * int(w) + int(h_index) * int(w) + int(w_index)
    phase_count = int(g * g)
    group = int(channel) // int(phase_count)
    phase = int(channel) % int(phase_count)
    phase_h = int(phase) // int(g)
    phase_w = int(phase) % int(g)
    packed_w = int(w) * int(g)
    group_block = int(h) * int(g) * int(w) * int(g)
    return int(group) * int(group_block) + (int(h_index) * int(g) + int(phase_h)) * int(packed_w) + (int(w_index) * int(g) + int(phase_w))


def _idx_chw_gap_tensor(channel: torch.Tensor, h_indices: torch.Tensor, w_indices: torch.Tensor, *, h: int, w: int, gap: int) -> torch.Tensor:
    g = max(1, int(gap))
    c = channel.to(dtype=torch.int64)
    if int(g) == 1:
        return c * int(h) * int(w) + h_indices.to(dtype=torch.int64) * int(w) + w_indices.to(dtype=torch.int64)
    phase_count = int(g * g)
    group = torch.div(c, int(phase_count), rounding_mode="floor")
    phase = torch.remainder(c, int(phase_count))
    phase_h = torch.div(phase, int(g), rounding_mode="floor")
    phase_w = torch.remainder(phase, int(g))
    packed_w = int(w) * int(g)
    group_block = int(h) * int(g) * int(w) * int(g)
    return group * int(group_block) + (h_indices.to(dtype=torch.int64) * int(g) + phase_h) * int(packed_w) + (w_indices.to(dtype=torch.int64) * int(g) + phase_w)


def _coalesce_complex_rows(keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(keys.numel()) == 0:
        return keys.to(dtype=torch.int64), values.to(dtype=torch.complex64)
    order = torch.argsort(keys.to(dtype=torch.int64))
    sorted_keys = keys.to(dtype=torch.int64).index_select(0, order)
    sorted_values = values.to(dtype=torch.complex64).index_select(0, order)
    unique, inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)
    out_values = torch.zeros((int(unique.numel()),), dtype=torch.complex64)
    out_values.index_add_(0, inverse.to(dtype=torch.int64), sorted_values)
    return unique.to(dtype=torch.int64), out_values


def _pack_chw(tensor: torch.Tensor, *, channels: int = 8, h: int = 64, w: int = 64, gap: int = 1) -> torch.Tensor:
    out = torch.zeros((RING_SLOT_COUNT,), dtype=torch.float32)
    for c in range(int(channels)):
        for ih in range(int(h)):
            for iw in range(int(w)):
                out[_slot_index(c, ih, iw, h=h, w=w, gap=gap)] = tensor[int(c), int(ih), int(iw)].to(dtype=torch.float32)
    return out


def _valid_spatial(kernel_h: int, kernel_w: int, *, h: int, w: int, pad: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    oh_all = torch.arange(int(h), dtype=torch.int64)
    ow_all = torch.arange(int(w), dtype=torch.int64)
    grid_oh, grid_ow = torch.meshgrid(oh_all, ow_all, indexing="ij")
    oh = grid_oh.reshape(-1)
    ow = grid_ow.reshape(-1)
    ih = oh - int(pad) + int(kernel_h)
    iw = ow - int(pad) + int(kernel_w)
    valid = (ih >= 0) & (ih < int(h)) & (iw >= 0) & (iw < int(w))
    return oh[valid], ow[valid], ih[valid], iw[valid]


def _build_bank_terms(
    *,
    spec: R18SameStageSpec,
    weight: torch.Tensor,
    input_pair_index: int,
    bank_index: int,
    family_id: str,
    case_name: str,
) -> tuple[tuple[CanonicalTemplateEntry, ...], tuple[PreparedPlaintext, ...], tuple[LinearTransformTerm, ...]]:
    surface_c = int(spec.surface_c)
    oc_start = int(bank_index) * int(surface_c)
    input_start = int(input_pair_index) * int(spec.c_pair)
    chunk_keys: list[torch.Tensor] = []
    chunk_values: list[torch.Tensor] = []
    ic_local = torch.arange(int(surface_c), dtype=torch.int64)
    for kh in range(3):
        for kw in range(3):
            oh, ow, ih, iw = _valid_spatial(kh, kw, h=int(spec.h), w=int(spec.w))
            src_slots = _idx_chw_gap_tensor(
                ic_local[:, None],
                ih[None, :],
                iw[None, :],
                h=int(spec.h),
                w=int(spec.w),
                gap=int(spec.gap),
            )
            oc_local = torch.arange(int(surface_c), dtype=torch.int64)
            oc_global = oc_local + int(oc_start)
            out_slots = _idx_chw_gap_tensor(
                oc_local[:, None],
                oh[None, :],
                ow[None, :],
                h=int(spec.h),
                w=int(spec.w),
                gap=int(spec.gap),
            )
            coeff = (
                weight[oc_global[:, None], input_start + ic_local[None, :], kh, kw].to(dtype=torch.complex64)
                - 1j * weight[oc_global[:, None], input_start + surface_c + ic_local[None, :], kh, kw].to(dtype=torch.complex64)
            )
            shifts = out_slots[:, None, :] - src_slots[None, :, :]
            output_slots = out_slots[:, None, :].expand_as(shifts)
            keys = (shifts.reshape(-1) * int(RING_SLOT_COUNT) + output_slots.reshape(-1)).to(dtype=torch.int64)
            vals = coeff[:, :, None].expand_as(shifts).reshape(-1).to(dtype=torch.complex64)
            coalesced_keys, coalesced_vals = _coalesce_complex_rows(keys, vals)
            chunk_keys.append(coalesced_keys)
            chunk_values.append(coalesced_vals)

    templates: list[CanonicalTemplateEntry] = []
    plaintexts: list[PreparedPlaintext] = []
    terms: list[LinearTransformTerm] = []
    all_keys, all_values = _coalesce_complex_rows(torch.cat(chunk_keys), torch.cat(chunk_values))
    shifts = torch.div(all_keys, int(RING_SLOT_COUNT), rounding_mode="floor").to(dtype=torch.int64)
    output_slot_values = torch.remainder(all_keys, int(RING_SLOT_COUNT)).to(dtype=torch.int64)
    for term_index, shift in enumerate(torch.unique_consecutive(shifts).tolist()):
        mask = shifts == int(shift)
        output_slots = output_slot_values[mask].to(dtype=torch.int64)
        values = all_values[mask].to(dtype=torch.complex64)
        order = torch.argsort(output_slots)
        output_slots = output_slots.index_select(0, order)
        values = values.index_select(0, order)
        template_id = f"{case_name}_bank{int(bank_index)}_template_{int(term_index)}"
        plaintext_id = f"{case_name}_bank{int(bank_index)}_pt_{int(term_index)}"
        templates.append(
            CanonicalTemplateEntry(
                template_id=template_id,
                family_id=family_id,
                key=(int(bank_index), int(shift)),
                fine_shift=int(shift),
                indices=output_slots,
                note=f"vendored scripts/cir R18 {spec.stage} inter-group template",
            )
        )
        plaintexts.append(
            PreparedPlaintext(
                plaintext_id=plaintext_id,
                template_id=template_id,
                level=0,
                scale=1.0,
                slot_count=RING_SLOT_COUNT,
                values=values,
                note=f"vendored scripts/cir R18 {spec.stage} inter-group payload",
            )
        )
        terms.append(
            LinearTransformTerm(
                term_id=f"{case_name}_bank{int(bank_index)}_term_{int(term_index)}",
                shift=int(shift),
                plaintext_id=plaintext_id,
                template_id=template_id,
                lookup_indices=torch.arange(int(output_slots.numel()), dtype=torch.int64),
                output_slot_indices=output_slots,
                note=f"vendored scripts/cir R18 {spec.stage} term",
                bank_id=_bank_id(spec, int(bank_index)),
            )
        )
    return tuple(templates), tuple(plaintexts), tuple(terms)


def build_r18_same_stage_shared_block_plan(
    *,
    spec: R18SameStageSpec,
    input_pair_index: int = 0,
    bank_count: int | None = None,
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn((int(spec.c), int(spec.h), int(spec.w)), dtype=torch.float32)
    weight = torch.randn((int(spec.c), int(spec.c), 3, 3), dtype=torch.float32)
    bank_count = int(spec.bank_count if bank_count is None else max(1, min(int(bank_count), int(spec.bank_count))))
    input_pair_index = max(0, min(int(input_pair_index), int(spec.input_pair_count) - 1))
    case_name = f"orion_vendored_r18_{spec.stage}_block{int(input_pair_index)}"
    family_id = f"{case_name}_family"
    all_templates: list[CanonicalTemplateEntry] = []
    all_plaintexts: list[PreparedPlaintext] = []
    all_terms: list[LinearTransformTerm] = []
    banks: list[SharedOutputBank] = []
    regions: list[TensorRegion] = []
    for bank_index in range(int(bank_count)):
        templates, plaintexts, terms = _build_bank_terms(
            spec=spec,
            weight=weight,
            input_pair_index=int(input_pair_index),
            bank_index=int(bank_index),
            family_id=family_id,
            case_name=case_name,
        )
        all_templates.extend(templates)
        all_plaintexts.extend(plaintexts)
        all_terms.extend(terms)
        banks.append(
            SharedOutputBank(
                bank_id=_bank_id(spec, int(bank_index)),
                target_index=int(bank_index),
                fold_lane=int(bank_index),
                input_lane_id=int(bank_index),
                output_slot_offset=0,
                active_slot_count=RING_SLOT_COUNT,
                term_count=int(len(terms)),
                note=f"vendored R18 {spec.stage} output bank",
            )
        )
        regions.append(
            TensorRegion(
                c_start=int(bank_index * int(spec.surface_c)),
                c_end=int(bank_index * int(spec.surface_c) + int(spec.surface_c)),
                h_start=0,
                h_end=int(spec.h),
                w_start=0,
                w_end=int(spec.w),
            )
        )
    step = LinearTransformStep(
        step_id=f"{case_name}_shared_lt",
        input_id="complex_source_pair_0",
        target_index=0,
        selected_n1=0,
        baby_shifts=(),
        giant_shifts=(),
        terms=tuple(all_terms),
        required_rotations=tuple(range(1, int(spec.rotations_per_block) + 1)),
        prepared_plaintext_ids=tuple(plain.plaintext_id for plain in all_plaintexts),
        expected_cost=ExecutionStats(rotations=int(spec.rotations_per_block), ct_pt_mults=int(len(all_terms)), adds=int(len(all_terms))),
        representation="inter_group_complex",
        note="vendored scripts/cir collapsed shared-output LT",
        rotation_group_id=f"{case_name}:complex_source_pair_0",
        rotation_cost_owner=True,
        shared_multi_output=True,
        shared_output_banks=tuple(banks),
    )
    plan = ConvSchemePlan(
        case_name=case_name,
        ring_slot_count=RING_SLOT_COUNT,
        output_regions=tuple(regions),
        output_active_slot_counts=tuple(RING_SLOT_COUNT for _ in regions),
        family_templates=(
            FamilyTemplateBank(
                family_id=family_id,
                family_key=("resnet18_tiny_imagenet", str(spec.stage), f"vendored_inter_group_block{int(input_pair_index)}"),
                source_tile_shape=(int(spec.c_pair), int(spec.h), int(spec.w)),
                target_tile_shape=(int(spec.surface_c), int(spec.h), int(spec.w)),
                source_h_range=(0, int(spec.h)),
                target_h_range=(0, int(spec.h)),
                template_entries=tuple(all_templates),
                member_count=int(bank_count),
                evidence_kind="vendored_scripts_cir_materialized",
            ),
        ),
        prepared_plaintexts=tuple(all_plaintexts),
        linear_transform_steps=(step,),
        expected_cost=ExecutionStats(
            rotations=int(spec.rotations_per_block),
            conjugations=int(bank_count),
            ct_pt_mults=int(len(all_terms)),
            adds=int(len(all_terms) + int(bank_count) + 1),
        ),
        evidence_kind=f"vendored_scripts_cir_r18_{spec.stage}_block",
        notes=(f"R18 {spec.stage} input surface-pair {input_pair_index}, selected output banks",),
    )
    # Set LT cost after all terms are known.
    step = LinearTransformStep(
        **{
            **step.__dict__,
            "expected_cost": ExecutionStats(rotations=int(spec.rotations_per_block), ct_pt_mults=int(len(all_terms)), adds=int(len(all_terms))),
        }
    )
    plan = ConvSchemePlan(
        **{
            **plan.__dict__,
            "linear_transform_steps": (step,),
        }
    )
    input_start = int(input_pair_index) * int(spec.c_pair)
    surface_c = int(spec.surface_c)
    inputs = {
        "source_0_lane_0": PlainCipherTensor(_pack_chw(x[input_start : input_start + surface_c], channels=int(surface_c), h=int(spec.h), w=int(spec.w), gap=int(spec.gap)), label="source_0_lane_0"),
        "source_1_lane_0": PlainCipherTensor(
            _pack_chw(x[input_start + surface_c : input_start + int(spec.c_pair)], channels=int(surface_c), h=int(spec.h), w=int(spec.w), gap=int(spec.gap)),
            label="source_1_lane_0",
        ),
    }
    reference = F.conv2d(x[input_start : input_start + int(spec.c_pair)].unsqueeze(0), weight[: int(bank_count * int(surface_c)), input_start : input_start + int(spec.c_pair)], bias=None, stride=1, padding=1)[0]
    return plan, inputs, reference


def build_r18_stage1_shared_block_plan(*, bank_count: int = 2) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    return build_r18_same_stage_shared_block_plan(spec=R18_STAGE1_SPEC, input_pair_index=0, bank_count=bank_count)


def build_r18_stage2_shared_block_plan(*, input_pair_index: int = 0, bank_count: int | None = None) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    return build_r18_same_stage_shared_block_plan(spec=R18_STAGE2_SPEC, input_pair_index=int(input_pair_index), bank_count=bank_count)
