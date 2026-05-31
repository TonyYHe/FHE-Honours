"""Raw descriptor-only LT task counter for Orion block tiling vs Halo local tiling.

This script intentionally counts raw independent submatrix-LT programs only. It
does not apply source-side materialization sharing, shared BSGS rotation reuse,
or real/imaginary lane packing.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


RING_SLOT_COUNT = 32768
RAW_COUNTING_CONTRACT = {
    "lt_task_unit": "one independent real-valued submatrix LT program",
    "source_side_materialization_sharing": "disabled",
    "shared_bsgs": "disabled",
    "real_imag_packing": "disabled",
    "rotation_cost_model": "not counted",
}


@dataclass(frozen=True)
class ConvCase:
    name: str
    kind: str
    c_in: int
    h_in: int
    w_in: int
    c_out: int
    h_out: int
    w_out: int
    gap_in: int
    gap_out: int
    kernel: int
    stride: int
    pad: int
    halo_mode: str
    note: str = ""
    raw_input_channels_per_ct: int | None = None
    raw_output_channels_per_ct: int | None = None


@dataclass(frozen=True)
class TconvCase:
    name: str
    kind: str
    c_in: int
    h_in: int
    w_in: int
    c_out: int
    h_out: int
    w_out: int
    gap_in: int
    gap_out: int
    kernel: int = 2
    stride: int = 2
    note: str = ""


@dataclass
class BlockInfo:
    channels: set[int]
    spatial: set[int]


@dataclass
class PairFlags:
    spatial_boundary: bool = False


def ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def phase_count(gap: int) -> int:
    return max(1, int(gap) * int(gap))


def packed_active_slots(c: int, h: int, w: int, gap: int) -> int:
    groups = ceil_div(int(c), phase_count(int(gap)))
    return int(groups) * int(h) * int(gap) * int(w) * int(gap)


def slot_index(channel: int, h_index: int, w_index: int, *, h: int, w: int, gap: int) -> int:
    g = max(1, int(gap))
    if int(g) == 1:
        return int(channel) * int(h) * int(w) + int(h_index) * int(w) + int(w_index)
    phases = int(g * g)
    group = int(channel) // int(phases)
    phase = int(channel) % int(phases)
    phase_h = int(phase) // int(g)
    phase_w = int(phase) % int(g)
    packed_w = int(w) * int(g)
    group_block = int(h) * int(g) * int(packed_w)
    return (
        int(group) * int(group_block)
        + (int(h_index) * int(g) + int(phase_h)) * int(packed_w)
        + int(w_index) * int(g)
        + int(phase_w)
    )


def spatial_id(h_index: int, w_index: int, width: int) -> int:
    return int(h_index) * int(width) + int(w_index)


def block_infos(*, c: int, h: int, w: int, gap: int, slots: int) -> list[BlockInfo]:
    count = ceil_div(packed_active_slots(int(c), int(h), int(w), int(gap)), int(slots))
    infos = [BlockInfo(channels=set(), spatial=set()) for _ in range(int(count))]
    for channel in range(int(c)):
        for ih in range(int(h)):
            for iw in range(int(w)):
                block = slot_index(channel, ih, iw, h=int(h), w=int(w), gap=int(gap)) // int(slots)
                infos[int(block)].channels.add(int(channel))
                infos[int(block)].spatial.add(spatial_id(int(ih), int(iw), int(w)))
    return infos


def source_blocks_by_spatial(*, c: int, h: int, w: int, gap: int, slots: int) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for channel in range(int(c)):
        for ih in range(int(h)):
            for iw in range(int(w)):
                sid = spatial_id(int(ih), int(iw), int(w))
                block = slot_index(channel, ih, iw, h=int(h), w=int(w), gap=int(gap)) // int(slots)
                out.setdefault(int(sid), set()).add(int(block))
    return out


def target_blocks_by_spatial(*, c: int, h: int, w: int, gap: int, slots: int) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for channel in range(int(c)):
        for ih in range(int(h)):
            for iw in range(int(w)):
                sid = spatial_id(int(ih), int(iw), int(w))
                block = slot_index(channel, ih, iw, h=int(h), w=int(w), gap=int(gap)) // int(slots)
                out.setdefault(int(sid), set()).add(int(block))
    return out


def summarize_pairs(
    pairs: dict[tuple[int, int], PairFlags],
    *,
    source_infos: list[BlockInfo],
    target_infos: list[BlockInfo],
    c_in: int,
    c_out: int,
) -> dict[str, int]:
    total = len(pairs)
    spatial_boundary = 0
    channel_pair_factor = 0
    channel_only = 0
    spatial_only = 0
    both = 0
    neither = 0
    for (target_block, source_block), flags in sorted(pairs.items()):
        source_channels = source_infos[int(source_block)].channels
        target_channels = target_infos[int(target_block)].channels
        has_channel_factor = len(source_channels) < int(c_in) or len(target_channels) < int(c_out)
        has_spatial = bool(flags.spatial_boundary)
        spatial_boundary += int(has_spatial)
        channel_pair_factor += int(has_channel_factor)
        channel_only += int(has_channel_factor and not has_spatial)
        spatial_only += int(has_spatial and not has_channel_factor)
        both += int(has_channel_factor and has_spatial)
        neither += int(not has_channel_factor and not has_spatial)
    return {
        "total_lt_tasks": int(total),
        "channel_pair_factor_tasks": int(channel_pair_factor),
        "spatial_boundary_tasks": int(spatial_boundary),
        "channel_only_tasks": int(channel_only),
        "spatial_only_tasks": int(spatial_only),
        "channel_and_spatial_tasks": int(both),
        "neither_channel_nor_spatial_tasks": int(neither),
    }


def _count_orion_gap1_aligned_same_shape_pairs(case: ConvCase, *, slots: int) -> dict[str, object] | None:
    if (
        int(case.gap_in) != 1
        or int(case.gap_out) != 1
        or int(case.stride) != 1
        or int(case.h_in) != int(case.h_out)
        or int(case.w_in) != int(case.w_out)
    ):
        return None
    plane = int(case.h_in) * int(case.w_in)
    if int(plane) < int(slots) or int(plane) % int(slots) != 0:
        return None
    chunks_per_channel = int(plane) // int(slots)
    touched_by_target_chunk: dict[int, set[int]] = {chunk: set() for chunk in range(int(chunks_per_channel))}
    boundary_by_target_chunk: dict[int, set[int]] = {chunk: set() for chunk in range(int(chunks_per_channel))}
    for target_chunk in range(int(chunks_per_channel)):
        sid_start = int(target_chunk) * int(slots)
        sid_end = min(int(plane), int(sid_start + int(slots)))
        for out_sid in range(int(sid_start), int(sid_end)):
            oh = int(out_sid) // int(case.w_out)
            ow = int(out_sid) % int(case.w_out)
            for kh in range(int(case.kernel)):
                ih = int(oh) - int(case.pad) + int(kh)
                if int(ih) < 0 or int(ih) >= int(case.h_in):
                    continue
                for kw in range(int(case.kernel)):
                    iw = int(ow) - int(case.pad) + int(kw)
                    if int(iw) < 0 or int(iw) >= int(case.w_in):
                        continue
                    source_chunk = int((int(ih) * int(case.w_in) + int(iw)) // int(slots))
                    touched_by_target_chunk[int(target_chunk)].add(int(source_chunk))
                    if int(source_chunk) != int(target_chunk):
                        boundary_by_target_chunk[int(target_chunk)].add(int(source_chunk))

    total = int(case.c_in) * int(case.c_out) * sum(len(chunks) for chunks in touched_by_target_chunk.values())
    spatial = int(case.c_in) * int(case.c_out) * sum(len(chunks) for chunks in boundary_by_target_chunk.values())
    has_channel_factor = int(case.c_in) > 1 or int(case.c_out) > 1
    source_block_count = int(case.c_in) * int(chunks_per_channel)
    target_block_count = int(case.c_out) * int(chunks_per_channel)
    pairs: list[list[int]] = []
    for oc in range(int(case.c_out)):
        for target_chunk, source_chunks in sorted(touched_by_target_chunk.items()):
            target_block = int(oc) * int(chunks_per_channel) + int(target_chunk)
            for ic in range(int(case.c_in)):
                for source_chunk in sorted(source_chunks):
                    source_block = int(ic) * int(chunks_per_channel) + int(source_chunk)
                    pairs.append([int(target_block), int(source_block)])
    return {
        "total_lt_tasks": int(total),
        "channel_pair_factor_tasks": int(total if has_channel_factor else 0),
        "spatial_boundary_tasks": int(spatial),
        "channel_only_tasks": int(total - spatial if has_channel_factor else 0),
        "spatial_only_tasks": int(spatial if not has_channel_factor else 0),
        "channel_and_spatial_tasks": int(spatial if has_channel_factor else 0),
        "neither_channel_nor_spatial_tasks": int(0 if has_channel_factor else total - spatial),
        "source_block_count": int(source_block_count),
        "target_block_count": int(target_block_count),
        "source_block_channels": [1 for _ in range(int(source_block_count))],
        "target_block_channels": [1 for _ in range(int(target_block_count))],
        "source_block_spatial_points": [int(slots) for _ in range(int(source_block_count))],
        "target_block_spatial_points": [int(slots) for _ in range(int(target_block_count))],
        "nonempty_pairs": pairs,
        "fast_path": "gap1_aligned_same_shape",
    }


def count_orion_conv_pairs(case: ConvCase, *, slots: int) -> dict[str, object]:
    fast = _count_orion_gap1_aligned_same_shape_pairs(case, slots=int(slots))
    if fast is not None:
        return fast
    target_infos = block_infos(c=case.c_out, h=case.h_out, w=case.w_out, gap=case.gap_out, slots=slots)
    source_infos = block_infos(c=case.c_in, h=case.h_in, w=case.w_in, gap=case.gap_in, slots=slots)
    source_by_spatial = source_blocks_by_spatial(c=case.c_in, h=case.h_in, w=case.w_in, gap=case.gap_in, slots=slots)
    target_by_spatial = target_blocks_by_spatial(c=case.c_out, h=case.h_out, w=case.w_out, gap=case.gap_out, slots=slots)
    pairs: dict[tuple[int, int], PairFlags] = {}
    for oh in range(int(case.h_out)):
        for ow in range(int(case.w_out)):
            out_sid = spatial_id(int(oh), int(ow), int(case.w_out))
            target_blocks = target_by_spatial[int(out_sid)]
            for target_block in target_blocks:
                target_spatial = target_infos[int(target_block)].spatial
                for kh in range(int(case.kernel)):
                    ih = int(oh) * int(case.stride) - int(case.pad) + int(kh)
                    if int(ih) < 0 or int(ih) >= int(case.h_in):
                        continue
                    for kw in range(int(case.kernel)):
                        iw = int(ow) * int(case.stride) - int(case.pad) + int(kw)
                        if int(iw) < 0 or int(iw) >= int(case.w_in):
                            continue
                        sid = spatial_id(int(ih), int(iw), int(case.w_in))
                        is_boundary = int(sid) not in target_spatial
                        for source_block in source_by_spatial[int(sid)]:
                            key = (int(target_block), int(source_block))
                            flags = pairs.setdefault(key, PairFlags())
                            flags.spatial_boundary = bool(flags.spatial_boundary or is_boundary)
    summary = summarize_pairs(
        pairs,
        source_infos=source_infos,
        target_infos=target_infos,
        c_in=int(case.c_in),
        c_out=int(case.c_out),
    )
    return {
        **summary,
        "source_block_count": int(len(source_infos)),
        "target_block_count": int(len(target_infos)),
        "source_block_channels": [len(info.channels) for info in source_infos],
        "target_block_channels": [len(info.channels) for info in target_infos],
        "source_block_spatial_points": [len(info.spatial) for info in source_infos],
        "target_block_spatial_points": [len(info.spatial) for info in target_infos],
        "nonempty_pairs": [[int(t), int(s)] for t, s in sorted(pairs)],
    }


def source_h_range_for_target(
    *,
    target_h_start: int,
    target_h_end: int,
    input_h: int,
    kernel: int,
    stride: int,
    pad: int,
) -> tuple[int, int]:
    source_start = int(target_h_start) * int(stride) - int(pad)
    source_end = (int(target_h_end) - 1) * int(stride) - int(pad) + int(kernel) - 1
    return max(0, int(source_start)), min(int(input_h), int(source_end) + 1)


def extend_h_range_to_length(*, required_start: int, required_end: int, desired_len: int, limit: int) -> tuple[int, int]:
    start = int(required_start)
    end = int(required_end)
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
    return int(start), int(end)


def target_h_from_source_h(*, source_h: int, kernel: int, stride: int) -> int:
    return int(max(1, ((int(source_h) - int(kernel)) // int(stride)) + 1))


def height_stripes(case: ConvCase, *, slots: int) -> list[dict[str, int]]:
    groups = ceil_div(int(case.c_in), phase_count(int(case.gap_in)))
    denom = int(groups) * int(case.w_in) * phase_count(int(case.gap_in))
    max_source_h = max(1, int(slots) // int(denom))
    target_tile_h = (
        int(case.h_out)
        if int(max_source_h) >= int(case.h_in)
        else target_h_from_source_h(source_h=int(max_source_h), kernel=int(case.kernel), stride=int(case.stride))
    )
    stripes: list[dict[str, int]] = []
    target_h = 0
    while int(target_h) < int(case.h_out):
        th0 = int(target_h)
        th1 = min(int(case.h_out), int(th0 + int(target_tile_h)))
        req0, req1 = source_h_range_for_target(
            target_h_start=int(th0),
            target_h_end=int(th1),
            input_h=int(case.h_in),
            kernel=int(case.kernel),
            stride=int(case.stride),
            pad=int(case.pad),
        )
        sh0, sh1 = extend_h_range_to_length(
            required_start=int(req0),
            required_end=int(req1),
            desired_len=int(max_source_h),
            limit=int(case.h_in),
        )
        stripes.append(
            {
                "target_h_start": int(th0),
                "target_h_end": int(th1),
                "source_h_start": int(sh0),
                "source_h_end": int(sh1),
                "required_source_h_start": int(req0),
                "required_source_h_end": int(req1),
                "stored_source_rows": int(sh1 - sh0),
                "target_rows": int(th1 - th0),
                "halo_redundant_rows": int((sh1 - sh0) - (th1 - th0)),
            }
        )
        target_h = int(th1)
    return stripes


def max_channels_per_tile(*, c: int, h: int, w: int, gap: int, slots: int) -> int:
    g = max(1, int(gap))
    denom = int(h) * int(w) * int(g) * int(g)
    if int(denom) <= 0:
        raise ValueError("invalid tile shape")
    groups_fit = int(slots) // int(denom)
    if int(groups_fit) <= 0:
        raise ValueError(f"one packed channel group does not fit: h={h}, w={w}, gap={gap}, slots={slots}")
    return min(int(c), int(groups_fit) * phase_count(int(gap)))


def count_halo_conv_tasks(case: ConvCase, *, slots: int) -> dict[str, object]:
    if str(case.halo_mode) == "channel_surface":
        if case.raw_input_channels_per_ct is None or case.raw_output_channels_per_ct is None:
            raise ValueError(f"{case.name} channel_surface mode requires channel capacities")
        source_blocks = ceil_div(int(case.c_in), int(case.raw_input_channels_per_ct))
        target_blocks = ceil_div(int(case.c_out), int(case.raw_output_channels_per_ct))
        tasks = int(source_blocks * target_blocks)
        return {
            "mode": "raw_channel_surface",
            "total_lt_tasks": int(tasks),
            "channel_pair_factor_tasks": int(tasks),
            "spatial_boundary_tasks": 0,
            "source_block_count": int(source_blocks),
            "target_block_count": int(target_blocks),
            "height_stripe_count": 1,
            "halo_redundant_rows": 0,
            "stripes": [
                {
                    "target_h_start": 0,
                    "target_h_end": int(case.h_out),
                    "source_h_start": 0,
                    "source_h_end": int(case.h_in),
                    "stored_source_rows": int(case.h_in),
                    "target_rows": int(case.h_out),
                    "halo_redundant_rows": 0,
                }
            ],
        }
    if str(case.halo_mode) != "height_stripes":
        raise ValueError(f"unknown halo mode {case.halo_mode!r}")
    stripes = height_stripes(case, slots=int(slots))
    task_count = 0
    stripe_rows: list[dict[str, int]] = []
    for row in stripes:
        source_c_per_tile = max_channels_per_tile(
            c=int(case.c_in),
            h=int(row["stored_source_rows"]),
            w=int(case.w_in),
            gap=int(case.gap_in),
            slots=int(slots),
        )
        target_c_per_tile = max_channels_per_tile(
            c=int(case.c_out),
            h=int(row["target_rows"]),
            w=int(case.w_out),
            gap=int(case.gap_out),
            slots=int(slots),
        )
        source_channel_tiles = ceil_div(int(case.c_in), int(source_c_per_tile))
        target_channel_tiles = ceil_div(int(case.c_out), int(target_c_per_tile))
        stripe_tasks = int(source_channel_tiles * target_channel_tiles)
        task_count += int(stripe_tasks)
        stripe_rows.append(
            dict(row)
            | {
                "source_channels_per_tile": int(source_c_per_tile),
                "target_channels_per_tile": int(target_c_per_tile),
                "source_channel_tiles": int(source_channel_tiles),
                "target_channel_tiles": int(target_channel_tiles),
                "lt_tasks": int(stripe_tasks),
            }
        )
    return {
        "mode": "height_stripes_fixed_slot_capacity",
        "total_lt_tasks": int(task_count),
        "channel_pair_factor_tasks": int(task_count if any(int(row["lt_tasks"]) > 1 for row in stripe_rows) else 0),
        "spatial_boundary_tasks": 0,
        "height_stripe_count": int(len(stripes)),
        "halo_redundant_rows": int(sum(int(row["halo_redundant_rows"]) for row in stripes)),
        "stored_source_rows_total": int(sum(int(row["stored_source_rows"]) for row in stripes)),
        "target_rows_total": int(sum(int(row["target_rows"]) for row in stripes)),
        "stripes": stripe_rows,
    }


def count_tconv_pairs(case: TconvCase, *, slots: int) -> dict[str, object]:
    source_infos = block_infos(c=case.c_in, h=case.h_in, w=case.w_in, gap=case.gap_in, slots=slots)
    target_infos = block_infos(c=case.c_out, h=case.h_out, w=case.w_out, gap=case.gap_out, slots=slots)
    target_by_spatial = target_blocks_by_spatial(
        c=case.c_out,
        h=case.h_out,
        w=case.w_out,
        gap=case.gap_out,
        slots=slots,
    )
    pairs: dict[tuple[int, int], PairFlags] = {}
    for ic in range(int(case.c_in)):
        for ih in range(int(case.h_in)):
            for iw in range(int(case.w_in)):
                source_block = slot_index(
                    int(ic),
                    int(ih),
                    int(iw),
                    h=int(case.h_in),
                    w=int(case.w_in),
                    gap=int(case.gap_in),
                ) // int(slots)
                for kh in range(int(case.kernel)):
                    oh = int(ih) * int(case.stride) + int(kh)
                    for kw in range(int(case.kernel)):
                        ow = int(iw) * int(case.stride) + int(kw)
                        sid = spatial_id(int(oh), int(ow), int(case.w_out))
                        for target_block in target_by_spatial[int(sid)]:
                            pairs.setdefault((int(target_block), int(source_block)), PairFlags(spatial_boundary=False))
    summary = summarize_pairs(
        pairs,
        source_infos=source_infos,
        target_infos=target_infos,
        c_in=int(case.c_in),
        c_out=int(case.c_out),
    )
    raw_input_pairs = int(len(source_infos))
    return {
        **summary,
        "source_block_count": int(len(source_infos)),
        "target_block_count": int(len(target_infos)),
        "source_block_channels": [len(info.channels) for info in source_infos],
        "target_block_channels": [len(info.channels) for info in target_infos],
        "nonempty_pairs": [[int(t), int(s)] for t, s in sorted(pairs)],
        "halo_rows_invalid": True,
        "raw_local_lt_tasks": int(raw_input_pairs * len(target_infos)),
        "raw_note": "k=2,s=2 TConv has no stencil halo row to save; this is a raw local placement count.",
    }


def default_conv_cases() -> tuple[ConvCase, ...]:
    return (
        ConvCase(
            name="synthetic_large_spatial_few_channel",
            kind="conv2d",
            c_in=1,
            h_in=8192,
            w_in=16,
            c_out=1,
            h_out=8192,
            w_out=16,
            gap_in=1,
            gap_out=1,
            kernel=3,
            stride=1,
            pad=1,
            halo_mode="height_stripes",
            note="Large H with one channel: spatial block boundaries dominate.",
        ),
        ConvCase(
            name="r34_imgnet_stage1_same_real_network",
            kind="conv2d",
            c_in=64,
            h_in=56,
            w_in=56,
            c_out=64,
            h_out=56,
            w_out=56,
            gap_in=4,
            gap_out=4,
            kernel=3,
            stride=1,
            pad=1,
            halo_mode="height_stripes",
            note="Real traced ResNet34 ImageNet stage1 same-shape 3x3 conv.",
        ),
        ConvCase(
            name="r34_imgnet_stage2_same_real_network",
            kind="conv2d",
            c_in=128,
            h_in=28,
            w_in=28,
            c_out=128,
            h_out=28,
            w_out=28,
            gap_in=8,
            gap_out=8,
            kernel=3,
            stride=1,
            pad=1,
            halo_mode="height_stripes",
            note="Real traced ResNet34 ImageNet stage2 same-shape 3x3 conv.",
        ),
        ConvCase(
            name="r34_imgnet_stage3_same_real_network",
            kind="conv2d",
            c_in=256,
            h_in=14,
            w_in=14,
            c_out=256,
            h_out=14,
            w_out=14,
            gap_in=16,
            gap_out=16,
            kernel=3,
            stride=1,
            pad=1,
            halo_mode="height_stripes",
            note="Real traced ResNet34 ImageNet stage3 same-shape 3x3 conv.",
        ),
        ConvCase(
            name="u22_256_base32_enc1a_real_network",
            kind="conv2d",
            c_in=3,
            h_in=256,
            w_in=256,
            c_out=32,
            h_out=256,
            w_out=256,
            gap_in=1,
            gap_out=1,
            kernel=3,
            stride=1,
            pad=1,
            halo_mode="height_stripes",
            note="Real traced U-Net 22 Kvasir 256 base32 enc1a 3x3 conv.",
        ),
        ConvCase(
            name="u22_256_base32_enc1b_real_network",
            kind="conv2d",
            c_in=32,
            h_in=256,
            w_in=256,
            c_out=32,
            h_out=256,
            w_out=256,
            gap_in=1,
            gap_out=1,
            kernel=3,
            stride=1,
            pad=1,
            halo_mode="height_stripes",
            note="Real traced U-Net 22 Kvasir 256 base32 enc1b 3x3 conv.",
        ),
        ConvCase(
            name="r18_tiny_stage1_same_current_channel_surface_raw",
            kind="conv2d",
            c_in=64,
            h_in=64,
            w_in=64,
            c_out=64,
            h_out=64,
            w_out=64,
            gap_in=1,
            gap_out=1,
            kernel=3,
            stride=1,
            pad=1,
            halo_mode="channel_surface",
            raw_input_channels_per_ct=8,
            raw_output_channels_per_ct=8,
            note="Current R18 stage1 raw channel-surface path, not a height-halo split.",
        ),
        ConvCase(
            name="r34_imgnet_stage4_same_fixed_capacity_halo",
            kind="conv2d",
            c_in=512,
            h_in=7,
            w_in=7,
            c_out=512,
            h_out=7,
            w_out=7,
            gap_in=32,
            gap_out=32,
            kernel=3,
            stride=1,
            pad=1,
            halo_mode="height_stripes",
            note="Small spatial surface: 2 Orion packed blocks trade into 4 fixed-capacity halo stripes.",
        ),
    )


def default_tconv_cases() -> tuple[TconvCase, ...]:
    return (
        TconvCase(
            name="u22_64_base32_up1_tconv_k2s2",
            kind="conv_transpose2d",
            c_in=64,
            h_in=32,
            w_in=32,
            c_out=32,
            h_out=64,
            w_out=64,
            gap_in=2,
            gap_out=1,
            note="Representative decoder TConv where any benefit is placement/scheduling, not halo rows.",
        ),
    )


def build_payload(cases: Iterable[str], *, slots: int) -> dict[str, object]:
    wanted = {str(case) for case in cases}
    conv_cases = default_conv_cases()
    tconv_cases = default_tconv_cases()
    if wanted:
        conv_cases = tuple(case for case in conv_cases if case.name in wanted)
        tconv_cases = tuple(case for case in tconv_cases if case.name in wanted)
    rows: list[dict[str, object]] = []
    for case in conv_cases:
        orion = count_orion_conv_pairs(case, slots=int(slots))
        halo = count_halo_conv_tasks(case, slots=int(slots))
        orion_effective = int(orion["total_lt_tasks"])
        halo_effective = (
            int(math.ceil(int(halo["total_lt_tasks"]) / 2.0))
            if str(case.halo_mode) == "height_stripes"
            else int(halo["total_lt_tasks"])
        )
        orion["effective_lt_tasks_with_hybrid"] = int(orion_effective)
        halo["effective_lt_tasks_with_hybrid"] = int(halo_effective)
        rows.append(
            {
                "case": asdict(case),
                "counting_contract": dict(RAW_COUNTING_CONTRACT),
                "orion_global_block_descriptor": orion,
                "haloed_local_descriptor": halo,
                "delta_halo_minus_orion_total_lt_tasks": int(halo["total_lt_tasks"]) - int(orion["total_lt_tasks"]),
                "delta_halo_minus_orion_effective_lt_tasks": int(halo_effective) - int(orion_effective),
                "interpretation": interpret_conv(case, orion, halo),
            }
        )
    for case in tconv_cases:
        descriptor = count_tconv_pairs(case, slots=int(slots))
        rows.append(
            {
                "case": asdict(case),
                "counting_contract": dict(RAW_COUNTING_CONTRACT),
                "orion_global_block_descriptor": descriptor,
                "haloed_local_descriptor": {
                    "mode": "raw_tconv_local_placement",
                    "total_lt_tasks": int(descriptor["raw_local_lt_tasks"]),
                    "spatial_boundary_tasks": 0,
                    "halo_rows_invalid": True,
                },
                "delta_halo_minus_orion_total_lt_tasks": int(descriptor["raw_local_lt_tasks"])
                - int(descriptor["total_lt_tasks"]),
                "interpretation": "TConv has no valid halo rows to eliminate; this raw count is a placement descriptor.",
            }
        )
    return {
        "status": "ok",
        "scope": "raw descriptor-only LT task decomposition; no scheme/backend/Lattigo execution",
        "slot_count": int(slots),
        "counting_contract": dict(RAW_COUNTING_CONTRACT),
        "counting_notes": [
            "Orion tasks are nonempty global block-LT source/target block pairs.",
            "Spatial-boundary tasks are block pairs with at least one contributing stencil edge outside the target block spatial footprint.",
            "Channel-pair-factor tasks are block pairs whose source or target block contains only a strict subset of real channels.",
            "The channel and spatial columns can overlap, so mutually exclusive buckets are also emitted.",
            "No source-side materialization sharing, shared BSGS rotation reuse, or real/imaginary lane packing is applied.",
            "effective_lt_tasks_with_hybrid models the planned real/imaginary pairing on halo-local height-stripe tasks; raw counts remain unchanged.",
        ],
        "rows": rows,
    }


def interpret_conv(case: ConvCase, orion: dict[str, object], halo: dict[str, object]) -> str:
    orion_total = int(orion["total_lt_tasks"])
    halo_total = int(halo["total_lt_tasks"])
    spatial = int(orion["spatial_boundary_tasks"])
    channel = int(orion["channel_pair_factor_tasks"])
    if halo_total < orion_total and spatial > 0:
        return "Halo reduces the spatial-boundary component in this descriptor."
    if halo_total == orion_total and channel >= orion_total and spatial == 0:
        return "Task count is channel-pair dominated; halo layout has no spatial-boundary tasks to remove."
    if halo_total == orion_total and spatial > 0:
        return "Fixed slot capacity trades Orion cross-boundary blocks for the same number of halo-local stripes."
    if halo_total > orion_total:
        return "Halo-local stripes exceed Orion block pairs under this fixed-capacity descriptor."
    return f"Descriptor total changes from Orion {orion_total} to Halo {halo_total}."


def print_table(payload: dict[str, object]) -> None:
    rows = list(payload.get("rows", []))
    headers = (
        "case",
        "orion",
        "orion_spatial",
        "orion_channel",
        "halo",
        "delta",
        "read",
    )
    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))
    for row in rows:
        case = dict(row["case"])
        orion = dict(row["orion_global_block_descriptor"])
        halo = dict(row["haloed_local_descriptor"])
        values = (
            str(case["name"]),
            str(orion["total_lt_tasks"]),
            str(orion.get("spatial_boundary_tasks", 0)),
            str(orion.get("channel_pair_factor_tasks", 0)),
            str(halo["total_lt_tasks"]),
            str(row["delta_halo_minus_orion_total_lt_tasks"]),
            str(row["interpretation"]),
        )
        print(" | ".join(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=RING_SLOT_COUNT)
    parser.add_argument("--case", action="append", default=[], help="Restrict to a named default case. May be repeated.")
    parser.add_argument("--out", type=Path, default=Path("/tmp/orion_descriptor_halo_lt_counter.json"))
    parser.add_argument("--table", action="store_true", help="Print a compact markdown table in addition to writing JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args.case, slots=int(args.slots))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if bool(args.table):
        print_table(payload)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
