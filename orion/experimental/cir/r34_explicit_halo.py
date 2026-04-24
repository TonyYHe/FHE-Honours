from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.region_lowering import pack_chw_gap, unpack_chw_gap
from .r34_geometry import source_h_range_for_target


RING_SLOT_COUNT = 32768


@dataclass(frozen=True)
class ExplicitHaloSectionSpec:
    family_label: str
    c: int
    w: int
    gap: int
    local_h: int
    core_h_start: int
    core_h_end: int
    halo_top: int
    halo_bottom: int

    @property
    def row_stride_slots(self) -> int:
        return int(self.w) * int(self.gap) * int(self.gap)


R34_STAGE3_SAME_EXPLICIT_HALO = ExplicitHaloSectionSpec(
    family_label="stage3_same",
    c=256,
    w=14,
    gap=16,
    local_h=9,
    core_h_start=1,
    core_h_end=8,
    halo_top=1,
    halo_bottom=1,
)


def _encode_mask(mask: torch.Tensor, *, scheme: Any, level: int) -> Any:
    return scheme.encode(mask.to(dtype=torch.float32), level=int(level))


def explicit_halo_rotation_count(
    spec: ExplicitHaloSectionSpec,
    *,
    has_prev: bool,
    has_next: bool,
) -> int:
    count = 0
    if int(spec.halo_top) > 0 and bool(has_prev):
        count += 1
    if int(spec.halo_bottom) > 0 and bool(has_next):
        count += 1
    return int(count)


def _row_band_mask(*, spec: ExplicitHaloSectionSpec, row_start: int, row_end: int, slots: int = RING_SLOT_COUNT) -> torch.Tensor:
    mask = torch.zeros((int(slots),), dtype=torch.float32)
    for row in range(int(row_start), int(row_end)):
        start = int(row) * int(spec.row_stride_slots)
        end = int(start + int(spec.row_stride_slots))
        mask[int(start) : int(end)] = 1.0
    return mask


def build_stage3_same_core_sections(source: torch.Tensor) -> dict[str, torch.Tensor]:
    spec = R34_STAGE3_SAME_EXPLICIT_HALO
    x = source.detach().cpu().to(dtype=torch.float32)
    expected_shape = (int(spec.c), 14, int(spec.w))
    if tuple(int(v) for v in x.shape) != expected_shape:
        raise ValueError(f"expected stage3 source shape {expected_shape}, got {tuple(x.shape)}")

    left = torch.zeros((int(spec.c), int(spec.local_h), int(spec.w)), dtype=torch.float32)
    right = torch.zeros((int(spec.c), int(spec.local_h), int(spec.w)), dtype=torch.float32)

    left[:, int(spec.core_h_start) : int(spec.core_h_end), :] = x[:, 0:7, :]
    right[:, int(spec.core_h_start) : int(spec.core_h_end), :] = x[:, 7:14, :]
    return {"left_core": left, "right_core": right}


def build_stage3_same_expected_halo_sections(source: torch.Tensor) -> dict[str, torch.Tensor]:
    spec = R34_STAGE3_SAME_EXPLICIT_HALO
    sections = build_stage3_same_core_sections(source)
    x = source.detach().cpu().to(dtype=torch.float32)

    left = sections["left_core"].clone()
    right = sections["right_core"].clone()

    left[:, int(spec.core_h_end), :] = x[:, 7, :]
    right[:, 0, :] = x[:, 6, :]
    return {"left_expected": left, "right_expected": right}


def encrypt_stage3_same_core_sections(*, scheme: Any, level: int, source: torch.Tensor) -> dict[str, CipherTensor]:
    spec = R34_STAGE3_SAME_EXPLICIT_HALO
    sections = build_stage3_same_core_sections(source)
    out: dict[str, CipherTensor] = {}
    for name, section in sections.items():
        flat = pack_chw_gap(
            section,
            shape=(int(spec.c), int(spec.local_h), int(spec.w)),
            gap=int(spec.gap),
            slots=int(RING_SLOT_COUNT),
        )
        out[str(name)] = scheme.encrypt(scheme.encode(flat.to(dtype=torch.float32), int(level)))
    return out


def apply_explicit_halo_fill(
    *,
    center_ct: CipherTensor,
    scheme: Any,
    level: int,
    spec: ExplicitHaloSectionSpec,
    prev_ct: CipherTensor | None = None,
    next_ct: CipherTensor | None = None,
    prev_spec: ExplicitHaloSectionSpec | None = None,
    next_spec: ExplicitHaloSectionSpec | None = None,
) -> CipherTensor:
    result = center_ct
    prev_spec = prev_spec or spec
    next_spec = next_spec or spec
    if int(spec.halo_top) > 0 and prev_ct is not None:
        target_start = int(spec.core_h_start) - int(spec.halo_top)
        source_start = int(prev_spec.core_h_end) - int(spec.halo_top)
        top_mask = _encode_mask(
            _row_band_mask(spec=spec, row_start=int(target_start), row_end=int(spec.core_h_start)),
            scheme=scheme,
            level=int(level),
        )
        shift = int(target_start - source_start) * int(spec.row_stride_slots)
        result = result + prev_ct.roll(int(shift), in_place=False) * top_mask
    if int(spec.halo_bottom) > 0 and next_ct is not None:
        target_start = int(spec.core_h_end)
        source_start = int(next_spec.core_h_start)
        bottom_mask = _encode_mask(
            _row_band_mask(
                spec=spec,
                row_start=int(target_start),
                row_end=int(spec.core_h_end + spec.halo_bottom),
            ),
            scheme=scheme,
            level=int(level),
        )
        shift = int(target_start - source_start) * int(spec.row_stride_slots)
        result = result + next_ct.roll(int(shift), in_place=False) * bottom_mask
    return result


def decode_flat(ct: CipherTensor, *, scheme: Any) -> torch.Tensor:
    pt = ct.decrypt()
    raw = scheme.backend._plaintexts[int(pt.ids[0])].values.detach().clone()
    if torch.is_complex(raw):
        raw = raw.real
    return raw.to(dtype=torch.float32)


def decode_section_tensor(ct: CipherTensor, *, scheme: Any, spec: ExplicitHaloSectionSpec) -> torch.Tensor:
    flat = decode_flat(ct, scheme=scheme)
    return unpack_chw_gap(
        flat,
        shape=(int(spec.c), int(spec.local_h), int(spec.w)),
        gap=int(spec.gap),
    ).to(dtype=torch.float32)


def build_section_spec_from_ranges(
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
) -> tuple[ExplicitHaloSectionSpec, tuple[int, int], tuple[int, int]]:
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
    spec = ExplicitHaloSectionSpec(
        family_label=str(family_label),
        c=int(c),
        w=int(w),
        gap=int(gap),
        local_h=int(aligned_sh1 - aligned_sh0),
        core_h_start=int(core_src_start - aligned_sh0),
        core_h_end=int(core_src_end - aligned_sh0),
        halo_top=int(halo_top),
        halo_bottom=int(halo_bottom),
    )
    if int(spec.core_h_start) < 0 or int(spec.core_h_end) > int(spec.local_h):
        raise ValueError("explicit halo section core rows fall outside local section")
    crop_start = int(th0 - (aligned_sh0 // max(1, int(stride))))
    crop_end = int(crop_start + (int(th1) - int(th0)))
    return spec, (int(core_src_start), int(core_src_end)), (int(crop_start), int(crop_end))


def build_core_only_section_tensor(
    source_partition: torch.Tensor,
    *,
    spec: ExplicitHaloSectionSpec,
    core_source_range: tuple[int, int],
) -> torch.Tensor:
    core_start, core_end = (int(v) for v in core_source_range)
    out = torch.zeros((int(spec.c), int(spec.local_h), int(spec.w)), dtype=torch.float32)
    out[:, int(spec.core_h_start) : int(spec.core_h_end), :] = source_partition[:, int(core_start) : int(core_end), :].to(dtype=torch.float32)
    return out


def encrypt_section_tensor(
    section: torch.Tensor,
    *,
    scheme: Any,
    level: int,
    spec: ExplicitHaloSectionSpec,
) -> CipherTensor:
    flat = pack_chw_gap(
        section.to(dtype=torch.float32),
        shape=(int(spec.c), int(spec.local_h), int(spec.w)),
        gap=int(spec.gap),
        slots=int(RING_SLOT_COUNT),
    )
    return scheme.encrypt(scheme.encode(flat, int(level)))
