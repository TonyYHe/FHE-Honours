"""Region-first shared linear-transform planning primitives.

This module is intentionally backend-agnostic. It defines the compiler-side
contracts that sit above Orion's existing diagonal LT evaluator:

1. RegionPlanner discovers multiple useful output banks from one source.
2. PackingPlanner creates CH/Halo source tiles and output-bank layouts.
3. SharedLTGroup describes the final shared-rotation LT primitive.

The backend implementation can consume SharedLTGroup by evaluating the union of
rotations for one source ciphertext and applying per-bank plaintext masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


BankKind = Literal["regular", "real_imag_pair"]
SourcePackingKind = Literal["none", "input_replication"]
BoundaryAction = Literal["keep_layout", "insert_extract", "validate_relu_safe"]


def _ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def packed_active_slots(c: int, h: int, w: int, gap: int) -> int:
    """Return Orion/Halo gap-packed active slots for a CHW tile."""

    phase = max(1, int(gap) * int(gap))
    groups = _ceil_div(int(c), int(phase))
    return int(groups * (int(h) * int(gap)) * (int(w) * int(gap)))


@dataclass(frozen=True)
class RegionNode:
    node_id: str
    op_kind: str
    input_id: str
    output_id: str
    family: str = ""


@dataclass(frozen=True)
class RegionCandidate:
    region_id: str
    source_input_id: str
    output_node_ids: tuple[str, ...]
    useful_output_banks: int
    reason: str = ""


class RegionPlanner:
    """Find same-source output regions without deciding packing/layout."""

    @staticmethod
    def discover_same_source_regions(nodes: Sequence[RegionNode], *, min_outputs: int = 2) -> tuple[RegionCandidate, ...]:
        by_source: dict[str, list[RegionNode]] = {}
        for node in nodes:
            if str(node.op_kind) != "linear":
                continue
            by_source.setdefault(str(node.input_id), []).append(node)

        regions: list[RegionCandidate] = []
        for source_id, group in sorted(by_source.items()):
            if len(group) < int(min_outputs):
                continue
            regions.append(
                RegionCandidate(
                    region_id=f"region_{source_id}",
                    source_input_id=str(source_id),
                    output_node_ids=tuple(str(node.node_id) for node in group),
                    useful_output_banks=int(len(group)),
                    reason="same source feeds multiple linear outputs",
                )
            )
        return tuple(regions)


@dataclass(frozen=True)
class SourceTile:
    tile_id: str
    c_start: int
    c_end: int
    h_start: int
    h_end: int
    w: int
    gap: int
    halo_top: int = 0
    halo_bottom: int = 0

    @property
    def c(self) -> int:
        return int(self.c_end) - int(self.c_start)

    @property
    def h(self) -> int:
        return int(self.h_end) - int(self.h_start)

    @property
    def active_slots(self) -> int:
        return packed_active_slots(self.c, self.h, int(self.w), int(self.gap))


@dataclass(frozen=True)
class TargetTile:
    tile_id: str
    c_start: int
    c_end: int
    h_start: int
    h_end: int
    w: int
    gap: int

    @property
    def c(self) -> int:
        return int(self.c_end) - int(self.c_start)

    @property
    def h(self) -> int:
        return int(self.h_end) - int(self.h_start)

    @property
    def active_slots(self) -> int:
        return packed_active_slots(self.c, self.h, int(self.w), int(self.gap))


@dataclass(frozen=True)
class SourcePacking:
    kind: SourcePackingKind = "none"
    replication: int = 1
    rotations: int = 0


@dataclass(frozen=True)
class OutputBank:
    bank_id: str
    target_tile_id: str
    kind: BankKind = "regular"
    real_output_id: str = ""
    imag_output_id: str = ""


@dataclass(frozen=True)
class LoweredRegionPlan:
    region_id: str
    source_tiles: tuple[SourceTile, ...]
    target_tiles: tuple[TargetTile, ...]
    output_banks: tuple[OutputBank, ...]
    source_packing: SourcePacking
    boundary_actions: tuple[BoundaryAction, ...]
    relu_safe_boundary: bool


class PackingPlanner:
    """Build CH/Halo source tiles and output bank layouts."""

    @staticmethod
    def build_ch_halo_source_tiles(
        *,
        c: int,
        h: int,
        w: int,
        gap: int,
        kernel: int,
        stride: int,
        pad: int,
        max_slots: int,
    ) -> tuple[SourceTile, ...]:
        # Prefer full-width CH tiles. First find max H for one channel group,
        # then max C for that H. This keeps downsample halo handling explicit.
        max_h = 1
        for candidate_h in range(1, int(h) + 1):
            if packed_active_slots(1, candidate_h, int(w), int(gap)) > int(max_slots):
                break
            max_h = int(candidate_h)

        tiles: list[SourceTile] = []
        h0 = 0
        while h0 < int(h):
            h1 = min(int(h), int(h0 + max_h))
            c_tile = 1
            for candidate_c in range(1, int(c) + 1):
                if packed_active_slots(candidate_c, int(h1 - h0), int(w), int(gap)) > int(max_slots):
                    break
                c_tile = int(candidate_c)
            for c0 in range(0, int(c), int(c_tile)):
                c1 = min(int(c), int(c0 + c_tile))
                tiles.append(
                    SourceTile(
                        tile_id=f"source_c{int(c0)}_{int(c1)}_h{int(h0)}_{int(h1)}",
                        c_start=int(c0),
                        c_end=int(c1),
                        h_start=int(h0),
                        h_end=int(h1),
                        w=int(w),
                        gap=int(gap),
                        halo_top=min(int(kernel) - 1, int(h0)),
                        halo_bottom=min(int(kernel) - 1, max(0, int(h) - int(h1))),
                    )
                )
            h0 = int(h1)

        if any(tile.active_slots > int(max_slots) for tile in tiles):
            raise ValueError("source tile exceeds max_slots")
        return tuple(tiles)

    @staticmethod
    def build_target_tiles(*, c: int, h: int, w: int, gap: int, max_slots: int) -> tuple[TargetTile, ...]:
        h_tile = 1
        for candidate_h in range(1, int(h) + 1):
            if packed_active_slots(int(c), int(candidate_h), int(w), int(gap)) > int(max_slots):
                break
            h_tile = int(candidate_h)
        tiles = []
        for h0 in range(0, int(h), int(h_tile)):
            h1 = min(int(h), int(h0 + h_tile))
            tiles.append(
                TargetTile(
                    tile_id=f"target_h{int(h0)}_{int(h1)}",
                    c_start=0,
                    c_end=int(c),
                    h_start=int(h0),
                    h_end=int(h1),
                    w=int(w),
                    gap=int(gap),
                )
            )
        if any(tile.active_slots > int(max_slots) for tile in tiles):
            raise ValueError("target tile exceeds max_slots")
        return tuple(tiles)

    @staticmethod
    def lower_transition_region(
        *,
        region: RegionCandidate,
        c_in: int,
        h_in: int,
        w_in: int,
        input_gap: int,
        c_out: int,
        h_out: int,
        w_out: int,
        output_gap: int,
        kernel: int,
        stride: int,
        pad: int,
        max_slots: int,
        use_real_imag_hybrid: bool = False,
        input_replication: int = 1,
    ) -> LoweredRegionPlan:
        source_tiles = PackingPlanner.build_ch_halo_source_tiles(
            c=int(c_in),
            h=int(h_in),
            w=int(w_in),
            gap=int(input_gap),
            kernel=int(kernel),
            stride=int(stride),
            pad=int(pad),
            max_slots=int(max_slots),
        )
        target_tiles = PackingPlanner.build_target_tiles(
            c=int(c_out),
            h=int(h_out),
            w=int(w_out),
            gap=int(output_gap),
            max_slots=int(max_slots),
        )
        banks: list[OutputBank] = []
        for target_tile in target_tiles:
            if bool(use_real_imag_hybrid):
                banks.append(
                    OutputBank(
                        bank_id=f"{target_tile.tile_id}_real_imag",
                        target_tile_id=str(target_tile.tile_id),
                        kind="real_imag_pair",
                        real_output_id=str(region.output_node_ids[0]),
                        imag_output_id=str(region.output_node_ids[1] if len(region.output_node_ids) > 1 else ""),
                    )
                )
            else:
                for node_id in region.output_node_ids:
                    banks.append(
                        OutputBank(
                            bank_id=f"{target_tile.tile_id}_{node_id}",
                            target_tile_id=str(target_tile.tile_id),
                            kind="regular",
                            real_output_id=str(node_id),
                        )
                    )
        rep = max(1, int(input_replication))
        packing = SourcePacking(
            kind="input_replication" if rep > 1 else "none",
            replication=int(rep),
            rotations=max(0, int(rep).bit_length() - 1),
        )
        return LoweredRegionPlan(
            region_id=str(region.region_id),
            source_tiles=source_tiles,
            target_tiles=target_tiles,
            output_banks=tuple(banks),
            source_packing=packing,
            boundary_actions=("insert_extract", "validate_relu_safe") if bool(use_real_imag_hybrid) else ("keep_layout", "validate_relu_safe"),
            relu_safe_boundary=True,
        )


@dataclass(frozen=True)
class SharedLTTransformSpec:
    transform_id: str
    source_tile_id: str
    output_bank_id: str
    shifts: tuple[int, ...]
    ct_pt_mults: int


@dataclass(frozen=True)
class SharedLTGroup:
    group_id: str
    source_tile_id: str
    transforms: tuple[SharedLTTransformSpec, ...]

    @property
    def union_rotations(self) -> tuple[int, ...]:
        return tuple(sorted({int(shift) for transform in self.transforms for shift in transform.shifts if int(shift) != 0}))

    @property
    def rotation_count(self) -> int:
        return int(len(self.union_rotations))

    @property
    def separate_rotation_count(self) -> int:
        return int(sum(len({int(shift) for shift in transform.shifts if int(shift) != 0}) for transform in self.transforms))

    @property
    def rotation_savings(self) -> int:
        return int(self.separate_rotation_count - self.rotation_count)

    @property
    def ct_pt_mults(self) -> int:
        return int(sum(int(transform.ct_pt_mults) for transform in self.transforms))
