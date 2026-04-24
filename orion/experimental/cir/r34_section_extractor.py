from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.region_lowering import pack_chw_gap, unpack_chw_gap
from .r34_geometry import source_h_range_for_target


RING_SLOT_COUNT = 32768


@dataclass(frozen=True)
class SectionExtractPlan:
    family_label: str
    c: int
    w: int
    gap: int
    kernel: int
    stride: int
    pad: int
    target_h_start: int
    target_h_end: int
    source_h_start: int
    source_h_end: int
    local_h: int
    core_h_start: int
    core_h_end: int
    halo_top: int
    halo_bottom: int
    core_source_start: int
    core_source_end: int
    crop_start: int
    crop_end: int
    core_shift_rows: int = 0

    @property
    def row_stride_slots(self) -> int:
        return int(self.w) * int(self.gap) * int(self.gap)

    @property
    def active_slots(self) -> int:
        return int(self.c) * int(self.local_h) * int(self.gap) * int(self.w) * int(self.gap)

    @property
    def core_row_count(self) -> int:
        return int(self.core_h_end - self.core_h_start)


def halo_rotation_count(plan: SectionExtractPlan, *, has_prev: bool, has_next: bool) -> int:
    count = 0
    if int(plan.core_shift_rows) != 0:
        count += 1
    if int(plan.halo_top) > 0 and bool(has_prev):
        count += 1
    if int(plan.halo_bottom) > 0 and bool(has_next):
        count += 1
    return int(count)


def build_section_extract_plan(
    *,
    family_label: str,
    c: int,
    w: int,
    gap: int,
    input_h: int,
    kernel: int,
    stride: int,
    pad: int,
    target_h_range: tuple[int, int],
    source_h_range: tuple[int, int],
    align_source_start_to_stride: bool = False,
    relocate_core: bool = False,
) -> SectionExtractPlan:
    th0, th1 = (int(v) for v in target_h_range)
    sh0, sh1 = (int(v) for v in source_h_range)
    req0, req1 = source_h_range_for_target(
        target_h_start=int(th0),
        target_h_end=int(th1),
        input_h=int(input_h),
        kernel=int(kernel),
        stride=int(stride),
        pad=int(pad),
    )
    aligned_sh0 = int(sh0)
    original_len = int(sh1 - sh0)
    if bool(align_source_start_to_stride) and int(stride) > 1:
        remainder = int(sh0) % int(stride)
        floor_aligned = max(0, int(sh0) - int(remainder))
        ceil_aligned = int(sh0) if int(remainder) == 0 else int(sh0) + int(stride - remainder)
        if int(ceil_aligned + original_len) >= int(req1) and int(ceil_aligned) < int(input_h):
            aligned_sh0 = int(ceil_aligned)
        else:
            aligned_sh0 = int(floor_aligned)
    aligned_sh1 = int(sh1)
    if bool(align_source_start_to_stride) and int(stride) > 1:
        aligned_sh1 = min(int(input_h), int(aligned_sh0 + int(original_len)))
        if int(aligned_sh1) < int(req1):
            aligned_sh1 = int(req1)

    core_src_start = max(int(aligned_sh0), int(th0) * int(stride))
    core_src_end = min(int(aligned_sh1), int(th1) * int(stride))
    halo_top = max(0, int(core_src_start) - int(req0))
    halo_bottom = max(0, int(req1) - int(core_src_end))
    local_h = int(aligned_sh1 - aligned_sh0)
    core_h_start = int(core_src_start - aligned_sh0)
    core_h_end = int(core_src_end - aligned_sh0)
    crop_start = int(th0 - (aligned_sh0 // max(1, int(stride))))
    crop_end = int(crop_start + (int(th1) - int(th0)))
    desired_core_start = int(halo_top)
    core_shift_rows = int(desired_core_start - int(core_h_start)) if bool(relocate_core) else 0

    plan = SectionExtractPlan(
        family_label=str(family_label),
        c=int(c),
        w=int(w),
        gap=int(gap),
        kernel=int(kernel),
        stride=int(stride),
        pad=int(pad),
        target_h_start=int(th0),
        target_h_end=int(th1),
        source_h_start=int(aligned_sh0),
        source_h_end=int(aligned_sh1),
        local_h=int(local_h),
        core_h_start=int(core_h_start),
        core_h_end=int(core_h_end),
        halo_top=int(halo_top),
        halo_bottom=int(halo_bottom),
        core_source_start=int(core_src_start),
        core_source_end=int(core_src_end),
        crop_start=int(crop_start),
        crop_end=int(crop_end),
        core_shift_rows=int(core_shift_rows),
    )
    if int(plan.core_h_start) < 0 or int(plan.core_h_end) > int(plan.local_h):
        raise ValueError("section extractor core rows fall outside local section")
    return plan


def build_core_section_tensor(
    source_partition: torch.Tensor,
    *,
    plan: SectionExtractPlan,
) -> torch.Tensor:
    out = torch.zeros((int(plan.c), int(plan.local_h), int(plan.w)), dtype=torch.float32)
    src0 = int(plan.core_source_start)
    src1 = int(plan.core_source_end)
    out[:, int(plan.core_h_start) : int(plan.core_h_end), :] = source_partition[:, int(src0) : int(src1), :].to(dtype=torch.float32)
    return out


def encode_section_tensor(
    section: torch.Tensor,
    *,
    scheme: Any,
    level: int,
    plan: SectionExtractPlan,
) -> CipherTensor:
    flat = pack_chw_gap(
        section.to(dtype=torch.float32),
        shape=(int(plan.c), int(plan.local_h), int(plan.w)),
        gap=int(plan.gap),
        slots=int(RING_SLOT_COUNT),
    )
    return scheme.encrypt(scheme.encode(flat, int(level)))


def decode_section_flat(ct: CipherTensor, *, scheme: Any) -> torch.Tensor:
    pt = ct.decrypt()
    raw = scheme.backend._plaintexts[int(pt.ids[0])].values.detach().clone()
    if torch.is_complex(raw):
        raw = raw.real
    return raw.to(dtype=torch.float32)


def decode_section_tensor(ct: CipherTensor, *, scheme: Any, plan: SectionExtractPlan) -> torch.Tensor:
    flat = decode_section_flat(ct, scheme=scheme)
    return unpack_chw_gap(
        flat,
        shape=(int(plan.c), int(plan.local_h), int(plan.w)),
        gap=int(plan.gap),
    ).to(dtype=torch.float32)


def _encode_mask(mask: torch.Tensor, *, scheme: Any, level: int) -> Any:
    return scheme.encode(mask.to(dtype=torch.float32), level=int(level))


def row_band_mask(*, plan: SectionExtractPlan, row_start: int, row_end: int, slots: int = RING_SLOT_COUNT) -> torch.Tensor:
    mask = torch.zeros((int(slots),), dtype=torch.float32)
    for row in range(int(row_start), int(row_end)):
        start = int(row) * int(plan.row_stride_slots)
        end = int(start + int(plan.row_stride_slots))
        mask[int(start) : int(end)] = 1.0
    return mask


def extract_section_ciphertext(
    *,
    center_ct: CipherTensor,
    scheme: Any,
    level: int,
    plan: SectionExtractPlan,
    prev_ct: CipherTensor | None = None,
    next_ct: CipherTensor | None = None,
    prev_plan: SectionExtractPlan | None = None,
    next_plan: SectionExtractPlan | None = None,
) -> CipherTensor:
    result = center_ct
    prev_plan = prev_plan or plan
    next_plan = next_plan or plan

    if int(plan.core_shift_rows) != 0:
        core_mask = _encode_mask(
            row_band_mask(
                plan=plan,
                row_start=max(0, int(plan.core_h_start + plan.core_shift_rows)),
                row_end=max(0, int(plan.core_h_end + plan.core_shift_rows)),
            ),
            scheme=scheme,
            level=int(level),
        )
        result = result.roll(int(plan.core_shift_rows) * int(plan.row_stride_slots), in_place=False) * core_mask

    if int(plan.halo_top) > 0 and prev_ct is not None:
        target_start = max(0, int(plan.core_h_start + plan.core_shift_rows) - int(plan.halo_top))
        source_start = int(prev_plan.core_h_end) - int(plan.halo_top)
        top_mask = _encode_mask(
            row_band_mask(plan=plan, row_start=int(target_start), row_end=int(target_start + plan.halo_top)),
            scheme=scheme,
            level=int(level),
        )
        shift = int(target_start - source_start) * int(plan.row_stride_slots)
        result = result + prev_ct.roll(int(shift), in_place=False) * top_mask

    if int(plan.halo_bottom) > 0 and next_ct is not None:
        target_start = int(plan.core_h_end + plan.core_shift_rows)
        source_start = int(next_plan.core_h_start)
        bottom_mask = _encode_mask(
            row_band_mask(plan=plan, row_start=int(target_start), row_end=int(target_start + plan.halo_bottom)),
            scheme=scheme,
            level=int(level),
        )
        shift = int(target_start - source_start) * int(plan.row_stride_slots)
        result = result + next_ct.roll(int(shift), in_place=False) * bottom_mask

    return result
