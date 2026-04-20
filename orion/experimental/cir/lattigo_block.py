from __future__ import annotations

from collections import defaultdict

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


def _slot_index(channel: int, h_index: int, w_index: int, *, h: int = 64, w: int = 64) -> int:
    return int(channel) * int(h) * int(w) + int(h_index) * int(w) + int(w_index)


def _pack_chw(tensor: torch.Tensor, *, channels: int = 8, h: int = 64, w: int = 64) -> torch.Tensor:
    out = torch.zeros((RING_SLOT_COUNT,), dtype=torch.float32)
    for c in range(int(channels)):
        for ih in range(int(h)):
            start = _slot_index(c, ih, 0, h=h, w=w)
            out[start : start + int(w)] = tensor[int(c), int(ih), : int(w)].to(dtype=torch.float32)
    return out


def _valid_spatial(kernel_h: int, kernel_w: int, *, h: int = 64, w: int = 64, pad: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    weight: torch.Tensor,
    bank_index: int,
    family_id: str,
    case_name: str,
) -> tuple[tuple[CanonicalTemplateEntry, ...], tuple[PreparedPlaintext, ...], tuple[LinearTransformTerm, ...]]:
    oc_start = int(bank_index) * 8
    buckets: dict[int, list[tuple[int, complex]]] = defaultdict(list)
    for kh in range(3):
        for kw in range(3):
            oh, ow, ih, iw = _valid_spatial(kh, kw)
            for oc_local in range(8):
                oc_global = int(oc_start + oc_local)
                out_base = int(oc_local) * 64 * 64
                out_slots = out_base + oh * 64 + ow
                for ic_local in range(8):
                    src_slots = int(ic_local) * 64 * 64 + ih * 64 + iw
                    shifts = out_slots - src_slots
                    coeff = (
                        weight[oc_global, ic_local, kh, kw].to(dtype=torch.complex64)
                        - 1j * weight[oc_global, ic_local + 8, kh, kw].to(dtype=torch.complex64)
                    )
                    for shift, out_slot in zip(shifts.tolist(), out_slots.tolist()):
                        buckets[int(shift)].append((int(out_slot), complex(coeff)))

    templates: list[CanonicalTemplateEntry] = []
    plaintexts: list[PreparedPlaintext] = []
    terms: list[LinearTransformTerm] = []
    for term_index, shift in enumerate(sorted(buckets)):
        entries = buckets[int(shift)]
        output_slots = torch.tensor([item[0] for item in entries], dtype=torch.int64)
        values = torch.tensor([item[1] for item in entries], dtype=torch.complex64)
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
                note="vendored scripts/cir R18 stage1 inter-group template",
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
                note="vendored scripts/cir R18 stage1 inter-group payload",
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
                note="vendored scripts/cir R18 stage1 term",
                bank_id=f"r18_stage1_bank_{int(bank_index)}",
            )
        )
    return tuple(templates), tuple(plaintexts), tuple(terms)


def build_r18_stage1_shared_block_plan(*, bank_count: int = 2) -> tuple[ConvSchemePlan, dict[str, PlainCipherTensor], torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn((64, 64, 64), dtype=torch.float32)
    weight = torch.randn((64, 64, 3, 3), dtype=torch.float32)
    bank_count = max(1, min(int(bank_count), 8))
    case_name = "orion_vendored_r18_stage1_same_block0"
    family_id = f"{case_name}_family"
    all_templates: list[CanonicalTemplateEntry] = []
    all_plaintexts: list[PreparedPlaintext] = []
    all_terms: list[LinearTransformTerm] = []
    banks: list[SharedOutputBank] = []
    regions: list[TensorRegion] = []
    for bank_index in range(int(bank_count)):
        templates, plaintexts, terms = _build_bank_terms(
            weight=weight,
            bank_index=int(bank_index),
            family_id=family_id,
            case_name=case_name,
        )
        all_templates.extend(templates)
        all_plaintexts.extend(plaintexts)
        all_terms.extend(terms)
        banks.append(
            SharedOutputBank(
                bank_id=f"r18_stage1_bank_{int(bank_index)}",
                target_index=int(bank_index),
                fold_lane=int(bank_index),
                input_lane_id=int(bank_index),
                output_slot_offset=0,
                active_slot_count=RING_SLOT_COUNT,
                term_count=int(len(terms)),
                note="vendored R18 stage1 output bank",
            )
        )
        regions.append(TensorRegion(c_start=int(bank_index * 8), c_end=int(bank_index * 8 + 8), h_start=0, h_end=64, w_start=0, w_end=64))
    step = LinearTransformStep(
        step_id=f"{case_name}_shared_lt",
        input_id="complex_source_pair_0",
        target_index=0,
        selected_n1=0,
        baby_shifts=(),
        giant_shifts=(),
        terms=tuple(all_terms),
        required_rotations=tuple(range(1, 21)),
        prepared_plaintext_ids=tuple(plain.plaintext_id for plain in all_plaintexts),
        expected_cost=ExecutionStats(rotations=20, ct_pt_mults=135 * int(bank_count), adds=135 * int(bank_count)),
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
                family_key=("resnet18_tiny_imagenet", "stage1_same", "vendored_inter_group_block0"),
                source_tile_shape=(16, 64, 64),
                target_tile_shape=(8, 64, 64),
                source_h_range=(0, 64),
                target_h_range=(0, 64),
                template_entries=tuple(all_templates),
                member_count=int(bank_count),
                evidence_kind="vendored_scripts_cir_materialized",
            ),
        ),
        prepared_plaintexts=tuple(all_plaintexts),
        linear_transform_steps=(step,),
        expected_cost=ExecutionStats(rotations=20, conjugations=int(bank_count), ct_pt_mults=135 * int(bank_count), adds=135 * int(bank_count) + int(bank_count) + 1),
        evidence_kind="vendored_scripts_cir_r18_stage1_block",
        notes=("R18 stage1 first input surface-pair, selected output banks",),
    )
    inputs = {
        "source_0_lane_0": PlainCipherTensor(_pack_chw(x[:8]), label="source_0_lane_0"),
        "source_1_lane_0": PlainCipherTensor(_pack_chw(x[8:16]), label="source_1_lane_0"),
    }
    reference = F.conv2d(x[:16].unsqueeze(0), weight[: int(bank_count * 8), :16], bias=None, stride=1, padding=1)[0]
    return plan, inputs, reference

