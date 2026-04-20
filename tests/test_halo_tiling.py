from __future__ import annotations

from orion.core.region_lowering import build_halo_tiling_proof


def test_explicit_source_tiles_contain_halo_rows_when_needed() -> None:
    proof = build_halo_tiling_proof()
    source = proof["with_halo"]["source_tile"]
    target = proof["target_tile"]

    assert source["h_start"] < target["h_start"]
    assert source["h_end"] > target["h_end"]
    assert source["halo_top"] > 0
    assert source["halo_bottom"] > 0


def test_halo_output_matches_torch_conv_and_no_halo_control_fails() -> None:
    proof = build_halo_tiling_proof()

    assert proof["status"] == "ok"
    assert proof["with_halo"]["max_abs"] <= 1.0e-5
    assert proof["without_halo"]["max_abs"] > proof["with_halo"]["max_abs"]
    assert proof["without_halo"]["max_abs"] > 1.0e-3


def test_halo_tiling_stays_within_32768_slot_bound() -> None:
    proof = build_halo_tiling_proof()

    assert proof["all_source_tiles_within_slot_bound"] is True
    assert proof["with_halo"]["source_tile"]["active_slots"] <= 32768
    assert proof["without_halo"]["source_tile"]["active_slots"] <= 32768
