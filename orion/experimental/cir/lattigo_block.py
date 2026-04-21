from __future__ import annotations

import math
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
R18_STAGE3_SPEC = R18SameStageSpec("stage3_same", c=256, h=16, w=16, gap=4, c_pair=256, rotations_per_block=90)


@dataclass(frozen=True)
class R18CompactStageSpec:
    stage: str
    c: int
    h: int
    w: int
    gap: int
    rotations_per_block: int


R18_STAGE4_SPEC = R18CompactStageSpec("stage4_same", c=512, h=8, w=8, gap=8, rotations_per_block=156)


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


def _phase_mask(*, phases: tuple[int, ...], shape: tuple[int, int, int], gap: int) -> torch.Tensor:
    c, h, w = (int(v) for v in shape)
    wanted = {int(phase) for phase in phases}
    out = torch.zeros((RING_SLOT_COUNT,), dtype=torch.float32)
    for channel in range(int(c)):
        if int(channel) % int(gap * gap) not in wanted:
            continue
        for ih in range(int(h)):
            for iw in range(int(w)):
                out[_slot_index(channel, ih, iw, h=h, w=w, gap=gap)] = 1.0
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
    weight_override: torch.Tensor | None = None,
    bias_override: torch.Tensor | None = None,
    source_override: torch.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    output_shape: tuple[int, int, int] | None = None,
    input_gap: int | None = None,
    output_gap: int | None = None,
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    torch.manual_seed(0)
    expected_shape = (int(spec.c), int(spec.c), 3, 3)
    if weight_override is None:
        weight = torch.randn(expected_shape, dtype=torch.float32)
        weight_source = "deterministic_random"
    else:
        weight = weight_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in weight.shape) != expected_shape:
            raise ValueError(f"{spec.stage} fused weight shape mismatch: expected {expected_shape}, got {tuple(weight.shape)}")
        weight_source = "fused_orion_on_weight"
    if source_override is None:
        x = torch.randn((int(spec.c), int(spec.h), int(spec.w)), dtype=torch.float32)
        source_kind = "deterministic_random"
    else:
        x = source_override.detach().to(dtype=torch.float32).clone()
        expected_source = (int(spec.c), int(spec.h), int(spec.w))
        if tuple(int(v) for v in x.shape) != expected_source:
            raise ValueError(f"{spec.stage} source shape mismatch: expected {expected_source}, got {tuple(x.shape)}")
        source_kind = "provided_source_override"
    if input_shape is not None and tuple(int(v) for v in input_shape) != (int(spec.c), int(spec.h), int(spec.w)):
        raise ValueError(f"{spec.stage} input_shape mismatch: {input_shape}")
    if output_shape is not None and tuple(int(v) for v in output_shape) != (int(spec.c), int(spec.h), int(spec.w)):
        raise ValueError(f"{spec.stage} output_shape mismatch: {output_shape}")
    if input_gap is not None and int(input_gap) != int(spec.gap):
        raise ValueError(f"{spec.stage} input_gap mismatch: {input_gap}")
    if output_gap is not None and int(output_gap) != int(spec.gap):
        raise ValueError(f"{spec.stage} output_gap mismatch: {output_gap}")
    if bias_override is not None:
        bias = bias_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in bias.shape) != (int(spec.c),):
            raise ValueError(f"{spec.stage} bias shape mismatch: expected {(int(spec.c),)}, got {tuple(bias.shape)}")
        bias_source = "accepted_not_folded"
    else:
        bias_source = "none"
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
        notes=(
            f"R18 {spec.stage} input surface-pair {input_pair_index}, selected output banks",
            f"weight_source={weight_source}",
            f"source_kind={source_kind}",
            f"bias_source={bias_source}",
        ),
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


def build_r18_stage1_shared_block_plan(
    *,
    bank_count: int = 2,
    input_pair_index: int = 0,
    weight_override: torch.Tensor | None = None,
    bias_override: torch.Tensor | None = None,
    source_override: torch.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    output_shape: tuple[int, int, int] | None = None,
    input_gap: int | None = None,
    output_gap: int | None = None,
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    return build_r18_same_stage_shared_block_plan(
        spec=R18_STAGE1_SPEC,
        input_pair_index=int(input_pair_index),
        bank_count=bank_count,
        weight_override=weight_override,
        bias_override=bias_override,
        source_override=source_override,
        input_shape=input_shape,
        output_shape=output_shape,
        input_gap=input_gap,
        output_gap=output_gap,
    )


def build_r18_stage2_shared_block_plan(
    *,
    input_pair_index: int = 0,
    bank_count: int | None = None,
    weight_override: torch.Tensor | None = None,
    bias_override: torch.Tensor | None = None,
    source_override: torch.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    output_shape: tuple[int, int, int] | None = None,
    input_gap: int | None = None,
    output_gap: int | None = None,
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    return build_r18_same_stage_shared_block_plan(
        spec=R18_STAGE2_SPEC,
        input_pair_index=int(input_pair_index),
        bank_count=bank_count,
        weight_override=weight_override,
        bias_override=bias_override,
        source_override=source_override,
        input_shape=input_shape,
        output_shape=output_shape,
        input_gap=input_gap,
        output_gap=output_gap,
    )


def build_r18_stage3_shared_block_plan(
    *,
    bank_count: int | None = None,
    weight_override: torch.Tensor | None = None,
    bias_override: torch.Tensor | None = None,
    source_override: torch.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    output_shape: tuple[int, int, int] | None = None,
    input_gap: int | None = None,
    output_gap: int | None = None,
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    return build_r18_same_stage_shared_block_plan(
        spec=R18_STAGE3_SPEC,
        input_pair_index=0,
        bank_count=bank_count,
        weight_override=weight_override,
        bias_override=bias_override,
        source_override=source_override,
        input_shape=input_shape,
        output_shape=output_shape,
        input_gap=input_gap,
        output_gap=output_gap,
    )


@dataclass(frozen=True)
class TconvK2S2Spec:
    """k=2, s=2 transposed convolution spec. Input is gap-interleaved; output has gap//2."""
    stage: str
    c_in: int
    h_in: int
    w_in: int
    c_out: int
    in_gap: int   # input packing gap (e.g. 4 for stage3 output feeding tconv)
    out_gap: int  # output packing gap = in_gap // 2

    @property
    def h_out(self) -> int:
        return int(self.h_in * 2)

    @property
    def w_out(self) -> int:
        return int(self.w_in * 2)

    @property
    def pair_count(self) -> int:
        # 4 phases -> 2 complex pairs per output channel group
        return 2


def _tconv_k2s2_slot_index_in(ic: int, ih: int, iw: int, spec: TconvK2S2Spec) -> int:
    return _slot_index(ic, ih, iw, h=spec.h_in, w=spec.w_in, gap=spec.in_gap)


def _tconv_k2s2_slot_index_out(oc: int, oh: int, ow: int, spec: TconvK2S2Spec) -> int:
    return _slot_index(oc, oh, ow, h=spec.h_out, w=spec.w_out, gap=spec.out_gap)


def _tconv_k2s2_dense_source_group_count(spec: TconvK2S2Spec) -> int:
    return int(math.ceil(int(spec.c_in) / int(spec.in_gap * spec.in_gap)))


def _tconv_k2s2_local_dense_src_slot(group: int, oh: int, ow: int, spec: TconvK2S2Spec) -> int:
    return _slot_index(group, oh, ow, h=spec.h_out, w=spec.w_out, gap=1)


def _tconv_k2s2_local_dense_out_slot(oc_local: int, oh: int, ow: int, spec: TconvK2S2Spec) -> int:
    return _slot_index(oc_local, oh, ow, h=spec.h_out, w=spec.w_out, gap=1)


def _tconv_k2s2_output_phase(oh: int, ow: int) -> int:
    return (int(oh) % 2) * 2 + (int(ow) % 2)


def _tconv_k2s2_phase_expansion_shifts(phase: int, spec: TconvK2S2Spec) -> tuple[int, int]:
    horizontal = 1 if int(phase) in (0, 2) else -1
    vertical = int(spec.w_out) if int(phase) in (0, 1) else -int(spec.w_out)
    return int(horizontal), int(vertical)


def _tconv_k2s2_compact_source_pairs(spec: TconvK2S2Spec) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return (
        ((0, 0), (1, -1)),
        ((2, -int(spec.w_out)), (3, -int(spec.w_out) - 1)),
    )


def _tconv_k2s2_phase_mask(phase: int, spec: TconvK2S2Spec) -> torch.Tensor:
    return _phase_mask(
        phases=(int(phase),),
        shape=(int(spec.c_in), int(spec.h_in), int(spec.w_in)),
        gap=int(spec.in_gap),
    )


def _tconv_k2s2_factorized_candidate_stats(spec: TconvK2S2Spec, *, bank_channels: int) -> dict[str, int | float]:
    bank_width = max(1, int(bank_channels))
    bank_count = int(math.ceil(int(spec.c_out) / int(bank_width)))
    source_group_count = int(_tconv_k2s2_dense_source_group_count(spec))
    expansion_shifts = {
        int(shift)
        for phase in range(int(spec.in_gap * spec.in_gap))
        for shift in _tconv_k2s2_phase_expansion_shifts(int(phase), spec)
    }
    mix_shifts: set[int] = set()
    total_terms = 0
    transform_count = int(spec.in_gap * spec.in_gap) * int(spec.pair_count) * int(bank_count)

    for bank_index in range(int(bank_count)):
        oc_start = int(bank_index) * int(bank_width)
        oc_end = min(int(spec.c_out), oc_start + int(bank_width))
        local_width = max(1, oc_end - oc_start)
        bank_shifts: set[int] = set()
        for src_group in range(int(source_group_count)):
            for oc_local in range(int(local_width)):
                shift = (int(oc_local) - int(src_group)) * int(spec.h_out) * int(spec.w_out)
                bank_shifts.add(int(shift) % int(RING_SLOT_COUNT))
                mix_shifts.add(int(shift) % int(RING_SLOT_COUNT))
        total_terms += int(len(bank_shifts)) * int(spec.in_gap * spec.in_gap) * int(spec.pair_count)

    rotation_count = int(len(expansion_shifts.union(mix_shifts)))
    score = float(rotation_count) * 8.0 + float(total_terms) + float(transform_count) * 8.0
    return {
        "bank_channels": int(bank_width),
        "bank_count": int(bank_count),
        "rotation_count": int(rotation_count),
        "term_count": int(total_terms),
        "transform_count": int(transform_count),
        "score": float(score),
    }


def _choose_tconv_k2s2_factorized_bank_channels(spec: TconvK2S2Spec) -> dict[str, int | float]:
    candidates = [value for value in range(1, int(spec.c_out) + 1) if int(spec.c_out) % int(value) == 0]
    viable = [
        _tconv_k2s2_factorized_candidate_stats(spec, bank_channels=int(value))
        for value in candidates
    ]
    capped = [item for item in viable if int(item["transform_count"]) <= 32]
    pool = capped or viable
    best = min(
        pool,
        key=lambda item: (
            int(item["rotation_count"]),
            float(item["score"]),
            int(item["term_count"]),
            int(item["transform_count"]),
        ),
    )
    return dict(best)


def _build_tconv_factorized_mix_terms(
    *,
    spec: TconvK2S2Spec,
    weight: torch.Tensor,
    input_phase: int,
    pair_index: int,
    bank_index: int,
    oc_start: int,
    bank_channels: int,
    case_name: str,
    family_id: str,
) -> tuple[tuple[CanonicalTemplateEntry, ...], tuple[PreparedPlaintext, ...], tuple[LinearTransformTerm, ...]]:
    """
    Build the factorized mix LT terms for one expanded input phase.

    The input to this LT is already phase-masked and spatially expanded into a
    dense gap-1 layout with `source_group_count` channels. This stage only
    performs channel mixing into a bank-local dense output layout, while the
    two-step spatial expansion is tracked separately.
    """
    phase0 = int(pair_index) * 2
    phase1 = int(pair_index) * 2 + 1

    buckets: dict[int, dict[int, complex]] = {}  # shift -> {out_slot -> complex weight}
    oc_end = min(int(spec.c_out), int(oc_start) + int(bank_channels))
    source_group_count = int(_tconv_k2s2_dense_source_group_count(spec))

    for src_group in range(int(source_group_count)):
        ic = int(src_group) * int(spec.in_gap * spec.in_gap) + int(input_phase)
        if int(ic) >= int(spec.c_in):
            continue
        for oh in range(int(spec.h_out)):
            for ow in range(int(spec.w_out)):
                phase = _tconv_k2s2_output_phase(oh, ow)
                if int(phase) != int(phase0) and int(phase) != int(phase1):
                    continue
                kh = int(oh) % 2
                kw = int(ow) % 2
                src_slot = _tconv_k2s2_local_dense_src_slot(int(src_group), oh, ow, spec)
                for oc in range(int(oc_start), int(oc_end)):
                    w_val = float(weight[ic, oc, kh, kw])
                    if w_val == 0.0:
                        continue
                    out_slot = _tconv_k2s2_local_dense_out_slot(int(oc - oc_start), oh, ow, spec)
                    shift = (int(out_slot) - int(src_slot)) % RING_SLOT_COUNT
                    if int(shift) not in buckets:
                        buckets[int(shift)] = {}
                    cval = complex(w_val) if int(phase) == int(phase0) else complex(0, w_val)
                    buckets[int(shift)][int(out_slot)] = buckets[int(shift)].get(int(out_slot), 0.0) + cval

    templates: list[CanonicalTemplateEntry] = []
    plaintexts: list[PreparedPlaintext] = []
    terms: list[LinearTransformTerm] = []

    for term_index, shift in enumerate(sorted(buckets.keys())):
        slot_map = buckets[int(shift)]
        out_slots = torch.tensor(sorted(slot_map.keys()), dtype=torch.int64)
        values = torch.tensor([slot_map[int(s)] for s in out_slots.tolist()], dtype=torch.complex64)
        template_id = f"{case_name}_phase{int(input_phase)}_bank{int(bank_index)}_pair{int(pair_index)}_template_{int(term_index)}"
        plaintext_id = f"{case_name}_phase{int(input_phase)}_bank{int(bank_index)}_pair{int(pair_index)}_pt_{int(term_index)}"
        templates.append(CanonicalTemplateEntry(
            template_id=template_id,
            family_id=family_id,
            key=(int(input_phase), int(bank_index), int(pair_index), int(shift)),
            fine_shift=int(shift),
            indices=out_slots,
            note=f"tconv k2s2 {spec.stage} factorized mix phase{int(input_phase)} bank{int(bank_index)} pair{int(pair_index)}",
        ))
        plaintexts.append(PreparedPlaintext(
            plaintext_id=plaintext_id,
            template_id=template_id,
            level=0,
            scale=1.0,
            slot_count=RING_SLOT_COUNT,
            values=values,
            note=f"tconv k2s2 {spec.stage} factorized mix payload",
        ))
        terms.append(LinearTransformTerm(
            term_id=f"{case_name}_phase{int(input_phase)}_bank{int(bank_index)}_pair{int(pair_index)}_term_{int(term_index)}",
            shift=int(shift),
            plaintext_id=plaintext_id,
            template_id=template_id,
            lookup_indices=torch.arange(int(out_slots.numel()), dtype=torch.int64),
            output_slot_indices=out_slots,
            note=f"tconv k2s2 {spec.stage} factorized mix term",
            bank_id=f"{case_name}_bank{int(bank_index)}_pair{int(pair_index)}",
        ))
    return tuple(templates), tuple(plaintexts), tuple(terms)


def build_tconv_k2s2_phase_pair_plan(
    spec: TconvK2S2Spec,
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    """
    Build a factorized `expand-then-mix` plan for a k=2, s=2 transposed convolution.

    The input starts in a gap-interleaved packed layout (`gap=spec.in_gap`). We
    first phase-mask the 4 packed input phases and expand each masked surface to
    a dense gap-1 layout using two rotations. We then run bank-local dense
    mixing LT steps from each expanded input phase into output phase-pair banks.
    """
    torch.manual_seed(0)
    x = torch.randn((int(spec.c_in), int(spec.h_in), int(spec.w_in)), dtype=torch.float32)
    weight = torch.randn((int(spec.c_in), int(spec.c_out), 2, 2), dtype=torch.float32)
    banking = _choose_tconv_k2s2_factorized_bank_channels(spec)
    bank_channels = int(banking["bank_channels"])
    bank_count = int(banking["bank_count"])
    input_phase_count = int(spec.in_gap * spec.in_gap)
    compact_pairs = _tconv_k2s2_compact_source_pairs(spec)
    compact_source_rotation_shifts = {
        int(abs(shift))
        for pair in compact_pairs
        for _phase, shift in pair
        if int(shift) != 0
    }
    expansion_shifts = {1, int(spec.w_out)}

    case_name = f"tconv_k2s2_{spec.stage}"
    family_id = f"{case_name}_family"

    all_templates: list[CanonicalTemplateEntry] = []
    all_plaintexts: list[PreparedPlaintext] = []
    lt_steps: list[LinearTransformStep] = []
    regions: list[TensorRegion] = []
    for bank_index in range(int(bank_count)):
        oc_start = int(bank_index) * int(bank_channels)
        oc_end = min(int(spec.c_out), oc_start + int(bank_channels))
        local_bank_width = max(1, oc_end - oc_start)
        for pair_index in range(int(spec.pair_count)):
            regions.append(TensorRegion(
                c_start=int(oc_start),
                c_end=int(oc_end),
                h_start=0,
                h_end=int(spec.h_out),
                w_start=0,
                w_end=int(spec.w_out),
            ))
            target_index = int(bank_index) * int(spec.pair_count) + int(pair_index)
            for input_phase in range(int(input_phase_count)):
                templates, plaintexts, terms = _build_tconv_factorized_mix_terms(
                    spec=spec,
                    weight=weight,
                    input_phase=int(input_phase),
                    pair_index=int(pair_index),
                    bank_index=int(bank_index),
                    oc_start=int(oc_start),
                    bank_channels=int(local_bank_width),
                    case_name=case_name,
                    family_id=family_id,
                )
                all_templates.extend(templates)
                all_plaintexts.extend(plaintexts)
                if not terms:
                    continue
                shift_set = {int(term.shift) for term in terms}
                lt_steps.append(
                    LinearTransformStep(
                        step_id=f"{case_name}_phase{int(input_phase)}_bank{int(bank_index)}_pair{int(pair_index)}",
                        input_id=f"expanded_input_phase_{int(input_phase)}",
                        target_index=int(target_index),
                        selected_n1=0,
                        baby_shifts=(),
                        giant_shifts=(),
                        terms=tuple(terms),
                        required_rotations=tuple(sorted(int(value) for value in shift_set)),
                        prepared_plaintext_ids=tuple(str(plain.plaintext_id) for plain in plaintexts),
                        expected_cost=ExecutionStats(
                            rotations=int(len(shift_set)),
                            ct_pt_mults=int(len(terms)),
                            adds=int(len(terms)),
                        ),
                        representation="factorized_phase_mix_complex",
                        note=f"tconv k2s2 factorized mix for input phase {int(input_phase)}",
                        rotation_group_id=f"{case_name}:expanded_input_phase_{int(input_phase)}",
                        rotation_cost_owner=True,
                    )
                )

    mix_shift_union = {int(term.shift) for step in lt_steps for term in step.terms}
    mix_term_count = sum(int(len(step.terms)) for step in lt_steps)
    source_pair_count = int(len(compact_pairs))
    source_pair_adds = int(source_pair_count)
    expansion_adds = int(source_pair_count) * 2
    expansion_mults = int(input_phase_count)
    total_rotation_count = int(len(compact_source_rotation_shifts.union(expansion_shifts).union(mix_shift_union)))
    plan = ConvSchemePlan(
        case_name=case_name,
        ring_slot_count=RING_SLOT_COUNT,
        output_regions=tuple(regions),
        output_active_slot_counts=tuple(
            int((region.c_end - region.c_start) * int(spec.h_out) * int(spec.w_out))
            for region in regions
        ),
        family_templates=(FamilyTemplateBank(
            family_id=family_id,
            family_key=("tconv_k2s2", str(spec.stage), "factorized_expand_then_mix"),
            source_tile_shape=(int(spec.c_in), int(spec.h_in), int(spec.w_in)),
            target_tile_shape=(int(bank_channels), int(spec.h_out), int(spec.w_out)),
            source_h_range=(0, int(spec.h_in)),
            target_h_range=(0, int(spec.h_out)),
            template_entries=tuple(all_templates),
            member_count=int(len(lt_steps)),
            evidence_kind="tconv_k2s2_factorized_materialized",
        ),),
        prepared_plaintexts=tuple(all_plaintexts),
        linear_transform_steps=tuple(lt_steps),
        expected_cost=ExecutionStats(
            rotations=int(total_rotation_count),
            conjugations=int(source_pair_count),
            ct_pt_mults=int(mix_term_count + expansion_mults),
            adds=int(mix_term_count + source_pair_adds + expansion_adds),
        ),
        evidence_kind="tconv_k2s2_factorized_expand_then_mix",
        notes=(
            f"tconv k2s2 {spec.stage} factorized expand-then-mix",
            f"output_bank_channels={bank_channels}",
            f"output_bank_count={bank_count}",
            f"input_phase_count={input_phase_count}",
            f"source_pair_count={source_pair_count}",
            f"compact_source_rotation_shifts={','.join(str(int(v)) for v in sorted(compact_source_rotation_shifts))}",
            f"expansion_rotation_shifts={','.join(str(int(v)) for v in sorted(expansion_shifts))}",
            f"mix_rotation_count={len(mix_shift_union)}",
            f"selected_rotation_count={total_rotation_count}",
            f"banking_score={float(banking['score']):.2f}",
        ),
    )

    inputs = {
        "source_0_lane_0": PlainCipherTensor(
            _pack_chw(x, channels=int(spec.c_in), h=int(spec.h_in), w=int(spec.w_in), gap=int(spec.in_gap)),
            label="source_0_lane_0",
        ),
    }

    reference = F.conv_transpose2d(
        x.unsqueeze(0),
        weight,
        bias=None,
        stride=2,
        padding=0,
    )[0]

    return plan, inputs, reference


def _build_compact_intra_rows(
    *,
    spec: R18CompactStageSpec,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = int(spec.c)
    h = int(spec.h)
    w = int(spec.w)
    gap = int(spec.gap)
    phase_count = int(gap * gap)
    half = int(phase_count // 2)
    input_group_count = int(c // phase_count)
    pair_left = torch.arange(int(half), dtype=torch.int64)
    groups = torch.arange(int(input_group_count), dtype=torch.int64)
    ic_left = groups[:, None] * int(phase_count) + pair_left[None, :]
    ic_right = ic_left + int(half)
    chunk_keys: list[torch.Tensor] = []
    chunk_values: list[torch.Tensor] = []
    for kh in range(3):
        for kw in range(3):
            oh, ow, ih, iw = _valid_spatial(kh, kw, h=h, w=w)
            for oc0 in range(0, int(c), 32):
                oc = torch.arange(int(oc0), int(min(int(c), int(oc0 + 32))), dtype=torch.int64)
                out_phase = torch.remainder(oc, int(phase_count))
                out_half = torch.div(out_phase, int(gap), rounding_mode="floor") >= int(gap // 2)
                source_phase = torch.where(out_half[:, None], pair_left[None, :] + int(half), pair_left[None, :])
                source_channel = groups[None, :, None] * int(phase_count) + source_phase[:, None, :]
                coeff_left = weight[oc[:, None, None], ic_left[None, :, :], kh, kw].to(dtype=torch.float32)
                coeff_right = weight[oc[:, None, None], ic_right[None, :, :], kh, kw].to(dtype=torch.float32)
                coeff = coeff_left.to(dtype=torch.complex64) - 1j * coeff_right.to(dtype=torch.complex64)
                out_slot = _idx_chw_gap_tensor(oc[:, None], oh[None, :], ow[None, :], h=h, w=w, gap=gap)
                src_slot = _idx_chw_gap_tensor(source_channel.reshape(-1)[:, None], ih[None, :], iw[None, :], h=h, w=w, gap=gap)
                src_slot = src_slot.reshape(int(oc.numel()), int(input_group_count), int(half), -1)
                shift = out_slot[:, None, None, :] - src_slot
                output_slot = out_slot[:, None, None, :].expand_as(shift)
                keys = (shift.reshape(-1) * int(RING_SLOT_COUNT) + output_slot.reshape(-1)).to(dtype=torch.int64)
                vals = coeff[:, :, :, None].expand_as(shift).reshape(-1).to(dtype=torch.complex64)
                coalesced_keys, coalesced_vals = _coalesce_complex_rows(keys, vals)
                chunk_keys.append(coalesced_keys)
                chunk_values.append(coalesced_vals)
    all_keys, all_values = _coalesce_complex_rows(torch.cat(chunk_keys), torch.cat(chunk_values))
    shifts = torch.div(all_keys, int(RING_SLOT_COUNT), rounding_mode="floor").to(dtype=torch.int64)
    output_slots = torch.remainder(all_keys, int(RING_SLOT_COUNT)).to(dtype=torch.int64)
    return shifts, output_slots, all_values.to(dtype=torch.complex64)


def build_r18_stage4_compact_intra_plan(
    *,
    weight_override: torch.Tensor | None = None,
    bias_override: torch.Tensor | None = None,
    source_override: torch.Tensor | None = None,
    input_shape: tuple[int, int, int] | None = None,
    output_shape: tuple[int, int, int] | None = None,
    input_gap: int | None = None,
    output_gap: int | None = None,
) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    spec = R18_STAGE4_SPEC
    torch.manual_seed(0)
    expected_shape = (int(spec.c), int(spec.c), 3, 3)
    if weight_override is None:
        weight = torch.randn(expected_shape, dtype=torch.float32)
        weight_source = "deterministic_random"
    else:
        weight = weight_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in weight.shape) != expected_shape:
            raise ValueError(f"{spec.stage} fused weight shape mismatch: expected {expected_shape}, got {tuple(weight.shape)}")
        weight_source = "fused_orion_on_weight"
    expected_source = (int(spec.c), int(spec.h), int(spec.w))
    if source_override is None:
        x = torch.randn(expected_source, dtype=torch.float32)
        source_kind = "deterministic_random"
    else:
        x = source_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in x.shape) != expected_source:
            raise ValueError(f"{spec.stage} source shape mismatch: expected {expected_source}, got {tuple(x.shape)}")
        source_kind = "provided_source_override"
    if input_shape is not None and tuple(int(v) for v in input_shape) != expected_source:
        raise ValueError(f"{spec.stage} input_shape mismatch: {input_shape}")
    if output_shape is not None and tuple(int(v) for v in output_shape) != expected_source:
        raise ValueError(f"{spec.stage} output_shape mismatch: {output_shape}")
    if input_gap is not None and int(input_gap) != int(spec.gap):
        raise ValueError(f"{spec.stage} input_gap mismatch: {input_gap}")
    if output_gap is not None and int(output_gap) != int(spec.gap):
        raise ValueError(f"{spec.stage} output_gap mismatch: {output_gap}")
    if bias_override is not None:
        bias = bias_override.detach().to(dtype=torch.float32).clone()
        if tuple(int(v) for v in bias.shape) != (int(spec.c),):
            raise ValueError(f"{spec.stage} bias shape mismatch: expected {(int(spec.c),)}, got {tuple(bias.shape)}")
        bias_source = "accepted_not_folded"
    else:
        bias_source = "none"
    shifts, output_slots, values = _build_compact_intra_rows(spec=spec, weight=weight)
    case_name = "orion_vendored_r18_stage4_same_compact_intra"
    family_id = f"{case_name}_family"
    templates: list[CanonicalTemplateEntry] = []
    plaintexts: list[PreparedPlaintext] = []
    terms: list[LinearTransformTerm] = []
    for index, shift in enumerate(torch.unique_consecutive(shifts).tolist()):
        mask = shifts == int(shift)
        term_outputs = output_slots[mask].to(dtype=torch.int64)
        term_values = values[mask].to(dtype=torch.complex64)
        template_id = f"{case_name}_template_{int(index)}"
        plaintext_id = f"{case_name}_pt_{int(index)}"
        templates.append(
            CanonicalTemplateEntry(
                template_id=template_id,
                family_id=family_id,
                key=(int(shift),),
                fine_shift=int(shift),
                indices=term_outputs,
                note="vendored compact-intra template",
            )
        )
        plaintexts.append(
            PreparedPlaintext(
                plaintext_id=plaintext_id,
                template_id=template_id,
                level=0,
                scale=1.0,
                slot_count=RING_SLOT_COUNT,
                values=term_values,
                note="vendored compact-intra payload",
            )
        )
        terms.append(
            LinearTransformTerm(
                term_id=f"{case_name}_term_{int(index)}",
                shift=int(shift),
                plaintext_id=plaintext_id,
                template_id=template_id,
                lookup_indices=torch.arange(int(term_outputs.numel()), dtype=torch.int64),
                output_slot_indices=term_outputs,
                note="vendored compact-intra term",
                bank_id="r18_stage4_compact_bank_0",
            )
        )
    bank = SharedOutputBank(
        bank_id="r18_stage4_compact_bank_0",
        target_index=0,
        fold_lane=0,
        input_lane_id=0,
        output_slot_offset=0,
        active_slot_count=RING_SLOT_COUNT,
        term_count=int(len(terms)),
        note="vendored R18 stage4 compact output bank",
    )
    step = LinearTransformStep(
        step_id=f"{case_name}_lt_0",
        input_id="compact_source_0_lane_0",
        target_index=0,
        selected_n1=0,
        baby_shifts=(),
        giant_shifts=(),
        terms=tuple(terms),
        required_rotations=tuple(range(1, int(spec.rotations_per_block) + 1)),
        prepared_plaintext_ids=tuple(plain.plaintext_id for plain in plaintexts),
        expected_cost=ExecutionStats(rotations=int(spec.rotations_per_block), ct_pt_mults=int(len(terms)), adds=int(len(terms))),
        representation="intra_group_phase_complex",
        note="vendored compact-intra LT",
        rotation_group_id=f"{case_name}:compact_source_0_lane_0",
        rotation_cost_owner=True,
        shared_multi_output=True,
        shared_output_banks=(bank,),
    )
    plan = ConvSchemePlan(
        case_name=case_name,
        ring_slot_count=RING_SLOT_COUNT,
        output_regions=(TensorRegion(c_start=0, c_end=int(spec.c), h_start=0, h_end=int(spec.h), w_start=0, w_end=int(spec.w)),),
        output_active_slot_counts=(RING_SLOT_COUNT,),
        family_templates=(
            FamilyTemplateBank(
                family_id=family_id,
                family_key=("resnet18_tiny_imagenet", "stage4_same", "vendored_compact_intra"),
                source_tile_shape=(int(spec.c), int(spec.h), int(spec.w)),
                target_tile_shape=(int(spec.c), int(spec.h), int(spec.w)),
                source_h_range=(0, int(spec.h)),
                target_h_range=(0, int(spec.h)),
                template_entries=tuple(templates),
                member_count=1,
                evidence_kind="vendored_compact_intra",
            ),
        ),
        prepared_plaintexts=tuple(plaintexts),
        linear_transform_steps=(step,),
        expected_cost=ExecutionStats(rotations=158, conjugations=1, ct_pt_mults=9767, adds=9768),
        evidence_kind="vendored_scripts_cir_r18_stage4_compact_intra",
        notes=(
            "R18 stage4 compact-intra full materializer",
            f"weight_source={weight_source}",
            f"source_kind={source_kind}",
            f"bias_source={bias_source}",
        ),
    )
    packed = _pack_chw(x, channels=int(spec.c), h=int(spec.h), w=int(spec.w), gap=int(spec.gap))
    inputs = {"source_0_lane_0": PlainCipherTensor(packed, label="source_0_lane_0")}
    reference = F.conv2d(x.unsqueeze(0), weight, bias=None, stride=1, padding=1)[0]
    return plan, inputs, reference
