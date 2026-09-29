from __future__ import annotations

import numpy as np
import pytest

from orion.experimental.wpc_cips_baseline import (
    LAYOUT_CIPS,
    evaluate_cyclic_diagonals,
    pack_tensor,
)
from orion.experimental.wpc_cips_downsample import (
    CIPSStride2ConvCase,
    WPCCIPSDownsampleReshapePlan,
    WPCCIPSStride2Conv2dPlan,
    build_downsample_reshape_transforms,
    build_stride2_cips_group_transforms,
    pack_compact_output,
    pack_sparse_output,
    rotation_padded_stride2_reference,
    unpack_compact_output,
    unpack_sparse_output,
)
from orion.experimental.wpc_periodicity import analyze_slot_periodicity
from orion.nn import Conv2d


def _case() -> CIPSStride2ConvCase:
    return CIPSStride2ConvCase(
        slots=512,
        input_channels=12,
        output_channels=12,
        input_height=8,
        input_width=8,
        kernel_height=3,
        kernel_width=3,
        pad_height_before=1,
        pad_width_before=1,
    )


def _evaluate_grouped_stride2(
    tensor: np.ndarray,
    weights: np.ndarray,
    case: CIPSStride2ConvCase,
) -> np.ndarray:
    transforms = build_stride2_cips_group_transforms(weights, case)
    input_messages = []
    for input_group, (start, end) in enumerate(case.input_group_ranges):
        input_messages.append(
            pack_tensor(
                tensor[start:end],
                case.local_case(0, input_group),
                LAYOUT_CIPS,
            )
        )
    output_messages = []
    for output_group in range(case.output_group_count):
        accumulated = np.zeros((case.slots,), dtype=np.float64)
        for input_group in range(case.input_group_count):
            key = f"out{output_group}_in{input_group}"
            accumulated += evaluate_cyclic_diagonals(
                input_messages[input_group], transforms[key]["diagonals"]
            )
        output_messages.append(accumulated)
    return unpack_sparse_output(np.stack(output_messages), case)


def _evaluate_reshape(
    sparse_messages: np.ndarray, case: CIPSStride2ConvCase
) -> np.ndarray:
    transforms = build_downsample_reshape_transforms(case)
    outputs = []
    for output_group in range(case.compact_output_group_count):
        accumulated = np.zeros((case.slots,), dtype=np.float64)
        for input_group in range(case.output_group_count):
            key = f"out{output_group}_in{input_group}"
            if key in transforms:
                accumulated += evaluate_cyclic_diagonals(
                    sparse_messages[input_group], transforms[key]["diagonals"]
                )
        outputs.append(accumulated)
    return np.stack(outputs)


def test_stride2_diagonals_match_independent_rotation_padding_reference() -> None:
    rng = np.random.default_rng(20260929)
    case = _case()
    tensor = rng.normal(size=(12, 8, 8))
    weights = rng.normal(size=(12, 12, 3, 3))

    reference = rotation_padded_stride2_reference(tensor, weights, case)
    evaluated = _evaluate_grouped_stride2(tensor, weights, case)

    assert evaluated.shape == (12, 4, 4)
    assert np.allclose(evaluated, reference, rtol=0.0, atol=1e-10)


def test_stride2_weight_diagonals_have_expected_short_period() -> None:
    case = _case()
    weights = np.ones((12, 12, 3, 3), dtype=np.float64)
    transforms = build_stride2_cips_group_transforms(weights, case)
    periods = {
        analyze_slot_periodicity(diagonal, payload_format="real").minimal_period
        for row in transforms.values()
        for diagonal in row["diagonals"].values()
    }

    assert len(transforms) == 4
    assert periods == {128}
    assert case.slots // next(iter(periods)) == 4


def test_sparse_reshape_restores_compact_cips_and_merges_groups() -> None:
    rng = np.random.default_rng(7)
    case = _case()
    logical = rng.normal(size=(12, 4, 4))
    sparse = pack_sparse_output(logical, case)
    compact = _evaluate_reshape(sparse, case)

    assert sparse.shape == (2, 512)
    assert compact.shape == (1, 512)
    assert np.array_equal(unpack_sparse_output(sparse, case), logical)
    assert np.array_equal(compact, pack_compact_output(logical, case))
    assert np.array_equal(unpack_compact_output(compact, case), logical)


def test_stride2_plan_signatures_chain_through_one_level_reshape() -> None:
    stride_layer = Conv2d(12, 12, 3, stride=2, padding=1, level=3)
    next_layer = Conv2d(12, 12, 3, stride=1, padding=1, level=1)
    stride_layer.name = "downsample"
    stride_layer.init_orion_params()
    next_layer.init_orion_params()

    source = WPCCIPSStride2Conv2dPlan(
        stride_layer, input_shape=(1, 12, 8, 8), slots=512
    )
    source.level = 3
    source.output_level = 2
    reshape = WPCCIPSDownsampleReshapePlan(source, level=2)
    consumer = __import__(
        "orion.experimental.wpc_cips_layer", fromlist=["WPCCIPSConv2dPlan"]
    ).WPCCIPSConv2dPlan(next_layer, input_shape=(1, 12, 4, 4), slots=512)
    consumer.level = 1

    assert source.case.output_group_count == 2
    assert source.case.compact_output_group_count == 1
    assert reshape.output_packing_signature == consumer.input_packing_signature
    assert reshape.output_level == consumer.level


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"stride": 1}, "stride 2x2"),
        ({"stride": 2, "dilation": 2, "padding": 2}, "dilation one"),
        ({"stride": 2, "kernel_size": 2, "padding": 1}, "odd kernels"),
    ],
)
def test_stride2_plan_rejects_unsupported_geometry(kwargs, message) -> None:
    options = {
        "in_channels": 12,
        "out_channels": 12,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "bias": False,
        "level": 3,
        **kwargs,
    }
    layer = Conv2d(**options)
    with pytest.raises(ValueError, match=message):
        WPCCIPSStride2Conv2dPlan(
            layer, input_shape=(1, 12, 8, 8), slots=512
        )
