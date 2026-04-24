from __future__ import annotations

from dataclasses import dataclass


RING_SLOT_COUNT = 32768


@dataclass(frozen=True)
class HaloRows:
    top: int
    bottom: int


@dataclass(frozen=True)
class ChannelPartition:
    group_start: int
    group_end: int
    c_start: int
    c_end: int


@dataclass(frozen=True)
class HeightStripe:
    target_h_start: int
    target_h_end: int
    source_h_start: int
    source_h_end: int
    halo_top: int
    halo_bottom: int


def ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def phase_group_size(gap: int) -> int:
    g = max(1, int(gap))
    return int(g * g)


def source_group_count(*, c: int, gap: int) -> int:
    return int(ceil_div(int(c), int(phase_group_size(int(gap)))))


def packed_active_slots(*, c: int, h: int, w: int, gap: int) -> int:
    groups = int(source_group_count(c=int(c), gap=int(gap)))
    g = max(1, int(gap))
    return int(groups) * int(h) * int(g) * int(w) * int(g)


def fixed_halo_rows(*, kernel: int) -> HaloRows:
    if int(kernel) <= 1:
        return HaloRows(top=0, bottom=0)
    return HaloRows(top=1, bottom=1)


def max_groups_per_ciphertext(*, h: int, w: int, gap: int, max_slots: int = RING_SLOT_COUNT) -> int:
    group_block = int(h) * int(w) * int(phase_group_size(int(gap)))
    if int(group_block) <= 0:
        raise ValueError("invalid group block geometry")
    return int(max(0, int(max_slots) // int(group_block)))


def max_source_h_for_channels(*, c: int, w: int, gap: int, max_slots: int = RING_SLOT_COUNT) -> int:
    groups = int(source_group_count(c=int(c), gap=int(gap)))
    denom = int(groups) * int(w) * int(phase_group_size(int(gap)))
    if int(denom) <= 0:
        raise ValueError("invalid source height denominator")
    return int(max(1, int(max_slots) // int(denom)))


def channel_partitions(*, c: int, h: int, w: int, gap: int, max_slots: int = RING_SLOT_COUNT) -> tuple[ChannelPartition, ...]:
    total_groups = int(source_group_count(c=int(c), gap=int(gap)))
    group_size = int(phase_group_size(int(gap)))
    groups_per_ct = int(max_groups_per_ciphertext(h=int(h), w=int(w), gap=int(gap), max_slots=int(max_slots)))
    groups_per_partition = max(1, min(int(total_groups), int(groups_per_ct or 1)))
    partitions: list[ChannelPartition] = []
    group_start = 0
    while int(group_start) < int(total_groups):
        group_end = min(int(total_groups), int(group_start + int(groups_per_partition)))
        c_start = int(group_start) * int(group_size)
        c_end = min(int(c), int(group_end) * int(group_size))
        partitions.append(
            ChannelPartition(
                group_start=int(group_start),
                group_end=int(group_end),
                c_start=int(c_start),
                c_end=int(c_end),
            )
        )
        group_start = int(group_end)
    return tuple(partitions)


def hybrid_pair_channel_partitions(*, c: int, gap: int) -> tuple[ChannelPartition, ...]:
    total_groups = int(source_group_count(c=int(c), gap=int(gap)))
    group_size = int(phase_group_size(int(gap)))
    partitions: list[ChannelPartition] = []
    group_start = 0
    while int(group_start) < int(total_groups):
        group_end = min(int(total_groups), int(group_start + 2))
        partitions.append(
            ChannelPartition(
                group_start=int(group_start),
                group_end=int(group_end),
                c_start=int(group_start) * int(group_size),
                c_end=min(int(c), int(group_end) * int(group_size)),
            )
        )
        group_start = int(group_end)
    return tuple(partitions)


def target_h_from_source_h(*, source_h: int, kernel: int, stride: int) -> int:
    return int(max(1, ((int(source_h) - int(kernel)) // int(stride)) + 1))


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


def extend_h_range_to_length(
    *,
    required_start: int,
    required_end: int,
    desired_len: int,
    limit: int,
) -> tuple[int, int]:
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
    if int(end - start) < int(desired):
        raise ValueError("cannot extend source range to requested length")
    return int(start), int(end)


def height_stripes_for_partition(
    *,
    c: int,
    h: int,
    w: int,
    gap: int,
    kernel: int,
    stride: int,
    pad: int,
    max_slots: int = RING_SLOT_COUNT,
) -> tuple[HeightStripe, ...]:
    halo = fixed_halo_rows(kernel=int(kernel))
    max_source_h = int(max_source_h_for_channels(c=int(c), w=int(w), gap=int(gap), max_slots=int(max_slots)))
    target_tile_h = int(h) if int(max_source_h) >= int(h) else int(target_h_from_source_h(source_h=int(max_source_h), kernel=int(kernel), stride=int(stride)))
    stripes: list[HeightStripe] = []
    target_h = 0
    while int(target_h) < int(h):
        th0 = int(target_h)
        th1 = min(int(h), int(th0 + int(target_tile_h)))
        req0, req1 = source_h_range_for_target(
            target_h_start=int(th0),
            target_h_end=int(th1),
            input_h=int(h),
            kernel=int(kernel),
            stride=int(stride),
            pad=int(pad),
        )
        sh0, sh1 = extend_h_range_to_length(
            required_start=int(req0),
            required_end=int(req1),
            desired_len=int(max_source_h),
            limit=int(h),
        )
        stripes.append(
            HeightStripe(
                target_h_start=int(th0),
                target_h_end=int(th1),
                source_h_start=int(sh0),
                source_h_end=int(sh1),
                halo_top=int(halo.top),
                halo_bottom=int(halo.bottom),
            )
        )
        target_h = int(th1)
    return tuple(stripes)
