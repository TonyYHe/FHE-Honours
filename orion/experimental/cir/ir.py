from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class ExecutionStats:
    rotations: int = 0
    conjugations: int = 0
    ct_pt_mults: int = 0
    adds: int = 0

    def plus(
        self,
        *,
        rotations: int = 0,
        conjugations: int = 0,
        ct_pt_mults: int = 0,
        adds: int = 0,
    ) -> "ExecutionStats":
        return ExecutionStats(
            rotations=int(self.rotations) + int(rotations),
            conjugations=int(self.conjugations) + int(conjugations),
            ct_pt_mults=int(self.ct_pt_mults) + int(ct_pt_mults),
            adds=int(self.adds) + int(adds),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "rotations": int(self.rotations),
            "conjugations": int(self.conjugations),
            "ct_pt_mults": int(self.ct_pt_mults),
            "adds": int(self.adds),
        }


@dataclass(frozen=True)
class TensorRegion:
    c_start: int
    c_end: int
    h_start: int
    h_end: int
    w_start: int
    w_end: int


@dataclass(frozen=True)
class CanonicalTemplateEntry:
    template_id: str
    family_id: str
    key: tuple[int, ...]
    fine_shift: int
    indices: torch.Tensor
    note: str = ""


@dataclass(frozen=True)
class FamilyTemplateBank:
    family_id: str
    family_key: tuple[Any, ...]
    source_tile_shape: tuple[int, int, int]
    target_tile_shape: tuple[int, int, int]
    source_h_range: tuple[int, int]
    target_h_range: tuple[int, int]
    template_entries: tuple[CanonicalTemplateEntry, ...]
    member_count: int
    evidence_kind: str = ""
    note: str = ""


@dataclass(frozen=True)
class PreparedPlaintext:
    plaintext_id: str
    template_id: str
    level: int
    scale: float
    slot_count: int
    values: torch.Tensor
    encoding_status: Literal["prepared"] = "prepared"
    note: str = ""


@dataclass(frozen=True)
class LinearTransformTerm:
    term_id: str
    shift: int
    plaintext_id: str
    template_id: str
    lookup_indices: torch.Tensor
    output_slot_indices: torch.Tensor
    note: str = ""
    bank_id: str = ""


@dataclass(frozen=True)
class SharedOutputBank:
    bank_id: str
    target_index: int
    fold_lane: int
    input_lane_id: int
    output_slot_offset: int
    active_slot_count: int
    term_count: int
    note: str = ""


@dataclass(frozen=True)
class LinearTransformStep:
    step_id: str
    input_id: str
    target_index: int
    selected_n1: int
    baby_shifts: tuple[int, ...]
    giant_shifts: tuple[int, ...]
    terms: tuple[LinearTransformTerm, ...]
    required_rotations: tuple[int, ...]
    prepared_plaintext_ids: tuple[str, ...]
    expected_cost: ExecutionStats = field(default_factory=ExecutionStats)
    representation: str = "inter_group_complex"
    note: str = ""
    rotation_group_id: str = ""
    rotation_cost_owner: bool = True
    shared_multi_output: bool = False
    shared_output_banks: tuple[SharedOutputBank, ...] = ()


@dataclass(frozen=True)
class ConvSchemePlan:
    case_name: str
    ring_slot_count: int
    output_regions: tuple[TensorRegion, ...]
    output_active_slot_counts: tuple[int, ...]
    family_templates: tuple[FamilyTemplateBank, ...]
    prepared_plaintexts: tuple[PreparedPlaintext, ...]
    linear_transform_steps: tuple[LinearTransformStep, ...]
    expected_cost: ExecutionStats
    evidence_kind: str = ""
    notes: tuple[str, ...] = ()


class PlainCipherTensor:
    def __init__(self, values: torch.Tensor, *, label: str = "") -> None:
        self._values = values.detach().clone()
        self.label = str(label)
        self.slot_count = int(values.numel())

    def unsafe_raw_tensor_for_debug(self) -> torch.Tensor:
        return self._values.detach().clone()

