from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

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


def _stats(value: Any) -> ExecutionStats:
    return ExecutionStats(
        rotations=int(getattr(value, "rotations", 0)),
        conjugations=int(getattr(value, "conjugations", 0)),
        ct_pt_mults=int(getattr(value, "ct_pt_mults", 0)),
        adds=int(getattr(value, "adds", 0)),
    )


def _tensor_region(value: Any) -> TensorRegion:
    return TensorRegion(
        c_start=int(getattr(value, "c_start")),
        c_end=int(getattr(value, "c_end")),
        h_start=int(getattr(value, "h_start")),
        h_end=int(getattr(value, "h_end")),
        w_start=int(getattr(value, "w_start")),
        w_end=int(getattr(value, "w_end")),
    )


def _canonical_template_entry(value: Any) -> CanonicalTemplateEntry:
    return CanonicalTemplateEntry(
        template_id=str(getattr(value, "template_id")),
        family_id=str(getattr(value, "family_id")),
        key=tuple(getattr(value, "key")),
        fine_shift=int(getattr(value, "fine_shift")),
        indices=getattr(value, "indices").detach().clone().to(dtype=torch.int64),
        note=str(getattr(value, "note", "")),
    )


def _family_template_bank(value: Any) -> FamilyTemplateBank:
    return FamilyTemplateBank(
        family_id=str(getattr(value, "family_id")),
        family_key=tuple(getattr(value, "family_key")),
        source_tile_shape=tuple(int(v) for v in getattr(value, "source_tile_shape")),
        target_tile_shape=tuple(int(v) for v in getattr(value, "target_tile_shape")),
        source_h_range=tuple(int(v) for v in getattr(value, "source_h_range")),
        target_h_range=tuple(int(v) for v in getattr(value, "target_h_range")),
        template_entries=tuple(_canonical_template_entry(entry) for entry in getattr(value, "template_entries")),
        member_count=int(getattr(value, "member_count")),
        evidence_kind=str(getattr(value, "evidence_kind", "")),
        note=str(getattr(value, "note", "")),
    )


def _prepared_plaintext(value: Any) -> PreparedPlaintext:
    return PreparedPlaintext(
        plaintext_id=str(getattr(value, "plaintext_id")),
        template_id=str(getattr(value, "template_id")),
        level=int(getattr(value, "level")),
        scale=float(getattr(value, "scale")),
        slot_count=int(getattr(value, "slot_count")),
        values=getattr(value, "values").detach().clone(),
        encoding_status="prepared",
        note=str(getattr(value, "note", "")),
    )


def _shared_output_bank(value: Any) -> SharedOutputBank:
    return SharedOutputBank(
        bank_id=str(getattr(value, "bank_id")),
        target_index=int(getattr(value, "target_index")),
        fold_lane=int(getattr(value, "fold_lane")),
        input_lane_id=int(getattr(value, "input_lane_id")),
        output_slot_offset=int(getattr(value, "output_slot_offset")),
        active_slot_count=int(getattr(value, "active_slot_count")),
        term_count=int(getattr(value, "term_count")),
        note=str(getattr(value, "note", "")),
    )


def _linear_transform_term(value: Any) -> LinearTransformTerm:
    return LinearTransformTerm(
        term_id=str(getattr(value, "term_id")),
        shift=int(getattr(value, "shift")),
        plaintext_id=str(getattr(value, "plaintext_id")),
        template_id=str(getattr(value, "template_id")),
        lookup_indices=getattr(value, "lookup_indices").detach().clone().to(dtype=torch.int64),
        output_slot_indices=getattr(value, "output_slot_indices").detach().clone().to(dtype=torch.int64),
        note=str(getattr(value, "note", "")),
        bank_id=str(getattr(value, "bank_id", "")),
    )


def _linear_transform_step(value: Any) -> LinearTransformStep:
    return LinearTransformStep(
        step_id=str(getattr(value, "step_id")),
        input_id=str(getattr(value, "input_id")),
        target_index=int(getattr(value, "target_index")),
        selected_n1=int(getattr(value, "selected_n1")),
        baby_shifts=tuple(int(v) for v in getattr(value, "baby_shifts")),
        giant_shifts=tuple(int(v) for v in getattr(value, "giant_shifts")),
        terms=tuple(_linear_transform_term(term) for term in getattr(value, "terms")),
        required_rotations=tuple(int(v) for v in getattr(value, "required_rotations")),
        prepared_plaintext_ids=tuple(str(v) for v in getattr(value, "prepared_plaintext_ids")),
        expected_cost=_stats(getattr(value, "expected_cost")),
        representation=str(getattr(value, "representation")),
        note=str(getattr(value, "note", "")),
        rotation_group_id=str(getattr(value, "rotation_group_id", "")),
        rotation_cost_owner=bool(getattr(value, "rotation_cost_owner", True)),
        shared_multi_output=bool(getattr(value, "shared_multi_output", False)),
        shared_output_banks=tuple(_shared_output_bank(bank) for bank in getattr(value, "shared_output_banks")),
    )


def haloed_plan_to_orion(value: Any) -> ConvSchemePlan:
    return ConvSchemePlan(
        case_name=str(getattr(value, "case_name")),
        ring_slot_count=int(getattr(value, "ring_slot_count")),
        output_regions=tuple(_tensor_region(region) for region in getattr(value, "output_regions")),
        output_active_slot_counts=tuple(int(v) for v in getattr(value, "output_active_slot_counts")),
        family_templates=tuple(_family_template_bank(bank) for bank in getattr(value, "family_templates")),
        prepared_plaintexts=tuple(_prepared_plaintext(plain) for plain in getattr(value, "prepared_plaintexts")),
        linear_transform_steps=tuple(_linear_transform_step(step) for step in getattr(value, "linear_transform_steps")),
        expected_cost=_stats(getattr(value, "expected_cost")),
        evidence_kind=str(getattr(value, "evidence_kind", "")),
        notes=tuple(str(v) for v in getattr(value, "notes", ())),
    )


def haloed_inputs_to_orion(value: dict[str, Any]) -> dict[str, PlainCipherTensor]:
    out: dict[str, PlainCipherTensor] = {}
    for key, tensor in value.items():
        raw = tensor.unsafe_raw_tensor_for_debug().detach().clone()
        out[str(key)] = PlainCipherTensor(raw, label=str(key))
    return out


def transform_from_orion_plan_step(
    *,
    plan: ConvSchemePlan,
    step: LinearTransformStep,
    level: int,
    scheme: Any,
    name: str,
) -> Any:
    prepared = {str(plain.plaintext_id): plain for plain in plan.prepared_plaintexts}
    templates = {str(entry.template_id): entry for family in plan.family_templates for entry in family.template_entries}
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
        diagonals={(0, 0): {int(index): diag.tolist() for index, diag in sorted(diag_tensors.items())}},
        level=int(level),
        scheme=scheme,
        fhe_output_shape=torch.Size([1, int(slots)]),
        output_shape=torch.Size([1, int(slots)]),
        target_index=int(step.target_index),
        input_id=str(step.input_id),
    )
