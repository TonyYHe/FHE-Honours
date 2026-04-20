from __future__ import annotations

from orion.core.shared_lt import (
    PackingPlanner,
    RegionNode,
    RegionPlanner,
    SharedLTGroup,
    SharedLTTransformSpec,
    packed_active_slots,
)


def test_region_planner_discovers_same_source_outputs() -> None:
    nodes = (
        RegionNode("conv1", "linear", "stage_input", "conv1_out"),
        RegionNode("downsample", "linear", "stage_input", "downsample_out"),
        RegionNode("other", "linear", "other_input", "other_out"),
    )

    regions = RegionPlanner.discover_same_source_regions(nodes)

    assert len(regions) == 1
    assert regions[0].source_input_id == "stage_input"
    assert regions[0].useful_output_banks == 2
    assert regions[0].output_node_ids == ("conv1", "downsample")


def test_packing_planner_builds_ch_halo_tiles_with_slot_bounds() -> None:
    region = RegionPlanner.discover_same_source_regions(
        (
            RegionNode("conv1", "linear", "stage_input", "conv1_out"),
            RegionNode("downsample", "linear", "stage_input", "downsample_out"),
        )
    )[0]

    plan = PackingPlanner.lower_transition_region(
        region=region,
        c_in=64,
        h_in=56,
        w_in=56,
        input_gap=1,
        c_out=128,
        h_out=28,
        w_out=28,
        output_gap=8,
        kernel=3,
        stride=2,
        pad=1,
        max_slots=32768,
    )

    assert plan.source_tiles
    assert plan.target_tiles
    assert all(tile.active_slots <= 32768 for tile in plan.source_tiles)
    assert all(tile.active_slots <= 32768 for tile in plan.target_tiles)
    assert len({(tile.c_start, tile.c_end) for tile in plan.source_tiles}) > 1
    assert len(plan.output_banks) == len(plan.target_tiles) * 2
    assert plan.relu_safe_boundary is True


def test_real_imag_hybrid_uses_complex_pair_banks_and_extract_boundary() -> None:
    region = RegionPlanner.discover_same_source_regions(
        (
            RegionNode("conv1", "linear", "stage_input", "conv1_out"),
            RegionNode("downsample", "linear", "stage_input", "downsample_out"),
        )
    )[0]

    plan = PackingPlanner.lower_transition_region(
        region=region,
        c_in=128,
        h_in=28,
        w_in=28,
        input_gap=8,
        c_out=256,
        h_out=14,
        w_out=14,
        output_gap=16,
        kernel=3,
        stride=2,
        pad=1,
        max_slots=32768,
        use_real_imag_hybrid=True,
    )

    assert plan.output_banks
    assert all(bank.kind == "real_imag_pair" for bank in plan.output_banks)
    assert all(bank.real_output_id == "conv1" for bank in plan.output_banks)
    assert all(bank.imag_output_id == "downsample" for bank in plan.output_banks)
    assert "insert_extract" in plan.boundary_actions
    assert plan.relu_safe_boundary is True


def test_input_replication_is_source_packing_not_shared_lt_replacement() -> None:
    region = RegionPlanner.discover_same_source_regions(
        (
            RegionNode("a", "linear", "small_input", "a_out"),
            RegionNode("b", "linear", "small_input", "b_out"),
            RegionNode("c", "linear", "small_input", "c_out"),
            RegionNode("d", "linear", "small_input", "d_out"),
        )
    )[0]

    plan = PackingPlanner.lower_transition_region(
        region=region,
        c_in=16,
        h_in=8,
        w_in=8,
        input_gap=1,
        c_out=16,
        h_out=8,
        w_out=8,
        output_gap=1,
        kernel=3,
        stride=1,
        pad=1,
        max_slots=32768,
        input_replication=4,
    )

    assert plan.source_packing.kind == "input_replication"
    assert plan.source_packing.replication == 4
    assert plan.source_packing.rotations == 2


def test_shared_lt_group_counts_union_rotations() -> None:
    group = SharedLTGroup(
        group_id="source0_shared",
        source_tile_id="source0",
        transforms=(
            SharedLTTransformSpec("conv1", "source0", "bank0", (0, 1, 2, 8), 4),
            SharedLTTransformSpec("downsample", "source0", "bank1", (0, 2, 8, 16), 4),
        ),
    )

    assert group.union_rotations == (1, 2, 8, 16)
    assert group.rotation_count == 4
    assert group.separate_rotation_count == 6
    assert group.rotation_savings == 2
    assert group.ct_pt_mults == 8


def test_packed_active_slots_matches_gap_group_formula() -> None:
    assert packed_active_slots(64, 56, 56, 1) == 64 * 56 * 56
    assert packed_active_slots(128, 28, 28, 8) == 2 * (28 * 8) * (28 * 8)
