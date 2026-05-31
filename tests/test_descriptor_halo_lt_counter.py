from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_counter_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "descriptor_halo_lt_counter.py"
    spec = importlib.util.spec_from_file_location("descriptor_halo_lt_counter", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def _load_halo_bsgs_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "analyze_halo_bsgs_combined_descriptor.py"
    spec = importlib.util.spec_from_file_location("analyze_halo_bsgs_combined_descriptor", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def _row_by_name(payload: dict, name: str) -> dict:
    for row in payload["rows"]:
        if row["case"]["name"] == name:
            return row
    raise AssertionError(f"missing descriptor row {name}")


def test_descriptor_counter_separates_spatial_and_channel_factors() -> None:
    counter = _load_counter_module()
    payload = counter.build_payload(
        (
            "synthetic_large_spatial_few_channel",
            "r18_tiny_stage1_same_current_channel_surface_raw",
            "r34_imgnet_stage4_same_fixed_capacity_halo",
        ),
        slots=counter.RING_SLOT_COUNT,
    )

    assert payload["counting_contract"]["shared_bsgs"] == "disabled"
    assert payload["counting_contract"]["real_imag_packing"] == "disabled"

    large = _row_by_name(payload, "synthetic_large_spatial_few_channel")
    assert large["orion_global_block_descriptor"]["total_lt_tasks"] == 10
    assert large["orion_global_block_descriptor"]["spatial_boundary_tasks"] == 6
    assert large["haloed_local_descriptor"]["total_lt_tasks"] == 5

    r18 = _row_by_name(payload, "r18_tiny_stage1_same_current_channel_surface_raw")
    assert r18["orion_global_block_descriptor"]["total_lt_tasks"] == 64
    assert r18["orion_global_block_descriptor"]["spatial_boundary_tasks"] == 0
    assert r18["orion_global_block_descriptor"]["channel_pair_factor_tasks"] == 64
    assert r18["haloed_local_descriptor"]["total_lt_tasks"] == 64
    assert r18["counting_contract"]["source_side_materialization_sharing"] == "disabled"

    r34 = _row_by_name(payload, "r34_imgnet_stage4_same_fixed_capacity_halo")
    assert r34["orion_global_block_descriptor"]["total_lt_tasks"] == 4
    assert r34["orion_global_block_descriptor"]["spatial_boundary_tasks"] == 2
    assert r34["haloed_local_descriptor"]["total_lt_tasks"] == 4
    assert r34["haloed_local_descriptor"]["effective_lt_tasks_with_hybrid"] == 2
    assert r34["delta_halo_minus_orion_effective_lt_tasks"] < 0
    assert r34["haloed_local_descriptor"]["height_stripe_count"] == 4


def test_descriptor_counter_finds_real_network_halo_positive_cases() -> None:
    counter = _load_counter_module()
    payload = counter.build_payload(
        (
            "r34_imgnet_stage1_same_real_network",
            "r34_imgnet_stage2_same_real_network",
            "r34_imgnet_stage3_same_real_network",
            "u22_256_base32_enc1a_real_network",
            "u22_256_base32_enc1b_real_network",
        ),
        slots=counter.RING_SLOT_COUNT,
    )

    expectations = {
        "r34_imgnet_stage1_same_real_network": (47, 36, 8),
        "r34_imgnet_stage2_same_real_network": (14, 9, 4),
        "r34_imgnet_stage3_same_real_network": (4, 2, 2),
        "u22_256_base32_enc1a_real_network": (384, 192, 70),
        "u22_256_base32_enc1b_real_network": (4096, 2048, 128),
    }
    for name, (orion_total, orion_spatial, halo_total) in expectations.items():
        row = _row_by_name(payload, name)
        assert row["orion_global_block_descriptor"]["total_lt_tasks"] == orion_total
        assert row["orion_global_block_descriptor"]["spatial_boundary_tasks"] == orion_spatial
        assert row["haloed_local_descriptor"]["total_lt_tasks"] == halo_total
        assert row["haloed_local_descriptor"]["effective_lt_tasks_with_hybrid"] <= halo_total
        assert row["delta_halo_minus_orion_effective_lt_tasks"] < 0
        assert row["delta_halo_minus_orion_total_lt_tasks"] < 0


def test_descriptor_counter_marks_tconv_as_non_halo_scheduling_case() -> None:
    counter = _load_counter_module()
    payload = counter.build_payload(("u22_64_base32_up1_tconv_k2s2",), slots=counter.RING_SLOT_COUNT)
    row = _row_by_name(payload, "u22_64_base32_up1_tconv_k2s2")

    assert row["orion_global_block_descriptor"]["total_lt_tasks"] == 8
    assert row["orion_global_block_descriptor"]["spatial_boundary_tasks"] == 0
    assert row["haloed_local_descriptor"]["halo_rows_invalid"] is True
    assert row["haloed_local_descriptor"]["total_lt_tasks"] == 8
    assert row["counting_contract"]["real_imag_packing"] == "disabled"


def test_r34_stage1_native_halo_stripe_oracle_separates_legacy_cap_ablation() -> None:
    halo_bsgs = _load_halo_bsgs_module()
    layout_counter = halo_bsgs.layout_counter
    from orion.experimental.cir.r34_orion_same_shape import r34_native_aligned_halo_plan

    case = {
        str(case.name): case
        for case in layout_counter.default_conv_cases()
    }["r34_imgnet_stage1_same_real_network"]

    descriptor = layout_counter.count_halo_conv_tasks(case, slots=layout_counter.RING_SLOT_COUNT)
    assert descriptor["height_stripe_count"] == 8
    assert descriptor["total_lt_tasks"] == 8
    assert [int(stripe["target_rows"]) for stripe in descriptor["stripes"]] == [7] * 8
    assert [int(stripe["stored_source_rows"]) for stripe in descriptor["stripes"]] == [9] * 8
    assert [
        [int(stripe["target_h_start"]), int(stripe["target_h_end"])]
        for stripe in descriptor["stripes"]
    ] == [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 42], [42, 49], [49, 56]]

    # This is not the native stripe CHW cut. It is the legacy active-block
    # max-channel exposure ablation used to isolate the old packing constraint.
    legacy_cap32 = halo_bsgs._halo_embedded_bsgs(
        case,
        slots=layout_counter.RING_SLOT_COUNT,
        source_channel_cap=32,
        target_channel_cap=32,
    )
    assert legacy_cap32["task_count"] == 32
    assert legacy_cap32["individual_rotations"] == 3008
    assert legacy_cap32["shared_rotations"] == 2544
    assert sorted(
        int(count)
        for group in legacy_cap32["groups"]
        for count in group["diag_counts"]
    ) == [674] * 32

    # The native provider cut is aligned to one gap^2 phase group per channel
    # tile and then uses the largest local-H that fits the ring. For R34 s1
    # that is 16ch x H36, producing 32 local Toeplitz programs and 8 B-sharing
    # groups.
    native = r34_native_aligned_halo_plan(family_label="stage1_same")
    assert native.channel_tile == 16
    assert [(s.target_h_start, s.target_h_end, s.source_h_start, s.source_h_end) for s in native.stripes] == [
        (0, 34, 0, 36),
        (34, 56, 20, 56),
    ]
    assert native.submatrix_program_count == 32
    assert native.sharing_group_count == 8
    assert native.c_only_rotations == 1152
    assert native.cb_shared_rotations == 904
    assert native.shared_baby_rotations == 232
    assert native.shared_giant_rotations == 672
    assert sorted(set(int(count) for count in native.program_diagonal_counts)) == [224]
