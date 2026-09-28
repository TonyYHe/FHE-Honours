from __future__ import annotations

import numpy as np

from orion.experimental.wpc_cips_baseline import (
    CIPSConvCase,
    CIPSMultiGroupConvCase,
    LAYOUT_CHANNEL_FIRST,
    LAYOUT_CIPS,
    build_cyclic_diagonals,
    pack_tensor,
    rotation_padded_reference,
    rotation_padded_source_coordinates,
    run_clear_cips_group_comparison,
    run_clear_layout_comparison,
    zero_padded_reference,
)


def _paper_toy() -> tuple[CIPSConvCase, np.ndarray, np.ndarray]:
    case = CIPSConvCase(
        slots=8,
        input_channels=2,
        output_channels=2,
        height=2,
        width=2,
        kernel_height=2,
        kernel_width=1,
        pad_height_before=0,
        pad_width_before=0,
    )
    tensor = np.arange(1, 9, dtype=np.float64).reshape(2, 2, 2)
    weights = np.arange(1, 9, dtype=np.float64).reshape(2, 2, 2, 1)
    return case, tensor, weights


def test_cips_packing_matches_channel_innermost_paper_order() -> None:
    case, tensor, _weights = _paper_toy()

    packed = pack_tensor(tensor, case, LAYOUT_CIPS)

    assert packed.tolist() == [
        tensor[0, 0, 0],
        tensor[1, 0, 0],
        tensor[0, 0, 1],
        tensor[1, 0, 1],
        tensor[0, 1, 0],
        tensor[1, 1, 0],
        tensor[0, 1, 1],
        tensor[1, 1, 1],
    ]


def test_matched_layouts_equal_rotation_padding_reference() -> None:
    case, tensor, weights = _paper_toy()

    result = run_clear_layout_comparison(
        tensor,
        weights,
        case,
        full_encoded_bytes_per_diagonal=1024,
    )

    assert result["valid"] is True
    assert result["layout_output_max_abs_delta"] == 0.0
    assert result["layouts"][LAYOUT_CIPS]["correct"] is True
    assert result["layouts"][LAYOUT_CHANNEL_FIRST]["correct"] is True


def test_cips_messages_are_periodic_but_channel_first_control_is_not() -> None:
    case, tensor, weights = _paper_toy()
    result = run_clear_layout_comparison(
        tensor,
        weights,
        case,
        full_encoded_bytes_per_diagonal=1024,
    )

    cips = result["layouts"][LAYOUT_CIPS]
    channel_first = result["layouts"][LAYOUT_CHANNEL_FIRST]
    assert cips["diagonal_count"] == 6
    assert cips["periodicity"]["periodic_count"] == 6
    assert set(cips["minimal_slot_period_by_rotation"].values()) == {2}
    assert cips["periodicity"]["partial_storage_compression_ratio"] == 4.0
    assert cips["message_period_reconstruction_exact"] is True

    assert channel_first["diagonal_count"] == 4
    assert channel_first["periodicity"]["periodic_count"] == 0
    assert channel_first["periodicity"]["partial_storage_compression_ratio"] == 1.0


def test_rotation_padding_is_not_silently_treated_as_zero_padding() -> None:
    case, tensor, weights = _paper_toy()

    rotation_padded = rotation_padded_reference(tensor, weights, case)
    zero_padded = zero_padded_reference(tensor, weights, case)

    assert not np.array_equal(rotation_padded, zero_padded)
    assert float(np.max(np.abs(rotation_padded - zero_padded))) > 0.0


def test_width_padding_follows_algorithm2_flattened_rotation() -> None:
    case = CIPSConvCase(
        slots=16,
        input_channels=1,
        output_channels=1,
        height=2,
        width=2,
        kernel_height=1,
        kernel_width=3,
        pad_height_before=0,
        pad_width_before=1,
    )
    tensor = np.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    weights = np.zeros((1, 1, 1, 3), dtype=np.float64)
    weights[0, 0, 0, 2] = 1.0

    assert rotation_padded_source_coordinates(case, 0, 1, 0, 2) == (1, 0)
    assert rotation_padded_source_coordinates(case, 1, 1, 0, 2) == (0, 0)
    output = rotation_padded_reference(tensor, weights, case)
    assert output.tolist() == [[[2.0, 3.0], [4.0, 1.0]]]

    comparison = run_clear_layout_comparison(
        tensor,
        weights,
        case,
        full_encoded_bytes_per_diagonal=1024,
    )
    assert comparison["valid"] is True
    assert comparison["flattened_rotation_differs_from_independent_axis_wrap"] is True


def test_larger_random_case_preserves_cips_periodicity_and_correctness() -> None:
    case = CIPSConvCase(
        slots=512,
        input_channels=4,
        output_channels=4,
        height=8,
        width=8,
        kernel_height=3,
        kernel_width=3,
        pad_height_before=1,
        pad_width_before=1,
    )
    rng = np.random.default_rng(17)
    tensor = rng.normal(size=(4, 8, 8))
    weights = rng.normal(size=(4, 4, 3, 3))

    result = run_clear_layout_comparison(
        tensor,
        weights,
        case,
        full_encoded_bytes_per_diagonal=4096,
    )
    cips = result["layouts"][LAYOUT_CIPS]

    assert result["valid"] is True
    assert cips["periodicity"]["periodic_count"] == cips["diagonal_count"]
    assert cips["periodicity"]["byte_coverage_pct"] == 100.0
    assert set(cips["minimal_slot_period_by_rotation"].values()) == {8}
    assert cips["periodicity"]["partial_storage_compression_ratio"] == 64.0
    assert cips["message_period_reconstruction_exact"] is True


def test_build_rejects_weight_shape_mismatch() -> None:
    case, _tensor, weights = _paper_toy()
    bad = weights[:, :, :, 0]

    try:
        build_cyclic_diagonals(bad, case, LAYOUT_CIPS)
    except ValueError as exc:
        assert "weight tensor shape" in str(exc)
    else:
        raise AssertionError("shape mismatch was not rejected")


def test_multigroup_cips_matches_global_reference_and_preserves_periodicity() -> None:
    case = CIPSMultiGroupConvCase(
        slots=32,
        input_channels=12,
        output_channels=10,
        height=2,
        width=2,
        kernel_height=3,
        kernel_width=3,
        pad_height_before=1,
        pad_width_before=1,
    )
    rng = np.random.default_rng(20260928)
    tensor = rng.normal(size=(12, 2, 2))
    weights = rng.normal(size=(10, 12, 3, 3))

    result = run_clear_cips_group_comparison(
        tensor,
        weights,
        case,
        full_encoded_bytes_per_diagonal=4096,
    )

    assert case.channel_capacity == 8
    assert case.input_group_ranges == ((0, 8), (8, 12))
    assert case.output_group_ranges == ((0, 8), (8, 10))
    assert result["transform_count"] == 4
    assert result["ciphertext_accumulation_add_count"] == 2
    assert result["correct"] is True
    assert result["max_abs_error_vs_reference"] < 1e-10
    assert result["all_transforms_proper_periodic"] is True
    assert result["all_message_period_roundtrips_exact"] is True
    assert set(result["transforms"]) == {
        "out0_in0",
        "out0_in1",
        "out1_in0",
        "out1_in1",
    }
    for row in result["transforms"].values():
        assert set(row["minimal_slot_period_by_rotation"].values()) == {8}
