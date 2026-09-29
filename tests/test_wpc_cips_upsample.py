from __future__ import annotations

import numpy as np
import pytest
import torch

from orion.experimental.wpc_cips_baseline import (
    LAYOUT_CIPS,
    evaluate_cyclic_diagonals,
    pack_tensor,
)
from orion.experimental.wpc_cips_upsample import (
    CIPSTranspose2ConvCase,
    WPCCIPSConvTranspose2dPlan,
    build_transpose2_cips_group_transforms,
    conv_transpose2d_reference,
    pack_upsample_output,
    unpack_upsample_output,
)
from orion.experimental.wpc_periodicity import analyze_slot_periodicity
from orion.nn import Conv2d, ConvTranspose2d


def _case() -> CIPSTranspose2ConvCase:
    return CIPSTranspose2ConvCase(
        slots=512,
        input_channels=12,
        output_channels=12,
        input_height=4,
        input_width=4,
    )


def _evaluate_grouped_transpose(
    tensor: np.ndarray,
    weights: np.ndarray,
    case: CIPSTranspose2ConvCase,
) -> np.ndarray:
    transforms = build_transpose2_cips_group_transforms(weights, case)
    input_messages = [
        pack_tensor(
            tensor[start:end],
            case.local_case(0, input_group),
            LAYOUT_CIPS,
        )
        for input_group, (start, end) in enumerate(case.input_group_ranges)
    ]
    output_messages = []
    for output_group in range(case.output_group_count):
        accumulated = np.zeros((case.slots,), dtype=np.float64)
        for input_group in range(case.input_group_count):
            key = f"out{output_group}_in{input_group}"
            accumulated += evaluate_cyclic_diagonals(
                input_messages[input_group], transforms[key]["diagonals"]
            )
        output_messages.append(accumulated)
    return unpack_upsample_output(np.stack(output_messages), case)


def test_transpose2_diagonals_match_independent_and_torch_references() -> None:
    rng = np.random.default_rng(20260929)
    case = _case()
    tensor = rng.normal(size=(12, 4, 4))
    weights = rng.normal(size=(12, 12, 2, 2))

    independent = conv_transpose2d_reference(tensor, weights, case)
    evaluated = _evaluate_grouped_transpose(tensor, weights, case)
    torch_reference = torch.nn.functional.conv_transpose2d(
        torch.tensor(tensor[None, ...], dtype=torch.float64),
        torch.tensor(weights, dtype=torch.float64),
        stride=2,
    )[0].numpy()

    assert evaluated.shape == (12, 8, 8)
    assert np.allclose(independent, torch_reference, rtol=0.0, atol=1e-10)
    assert np.allclose(evaluated, independent, rtol=0.0, atol=1e-10)


def test_transpose2_weight_diagonals_are_periodic_and_expand_groups() -> None:
    case = _case()
    weights = np.ones((12, 12, 2, 2), dtype=np.float64)
    transforms = build_transpose2_cips_group_transforms(weights, case)
    periods = {
        analyze_slot_periodicity(diagonal, payload_format="real").minimal_period
        for row in transforms.values()
        for diagonal in row["diagonals"].values()
    }

    assert case.input_group_count == 1
    assert case.output_group_count == 2
    assert len(transforms) == 2
    assert periods == {128}
    assert case.slots // next(iter(periods)) == 4


def test_transpose2_output_packing_roundtrips_exactly() -> None:
    rng = np.random.default_rng(7)
    case = _case()
    logical = rng.normal(size=(12, 8, 8))
    packed = pack_upsample_output(logical, case)

    assert packed.shape == (2, 512)
    assert np.array_equal(unpack_upsample_output(packed, case), logical)


def test_actual_orion_transpose_plan_signature_chains_to_conv_consumer() -> None:
    upsample = ConvTranspose2d(12, 12, 2, stride=2, level=2)
    consumer = Conv2d(12, 8, 3, stride=1, padding=1, level=1)
    upsample.name = "decoder_up"
    upsample.init_orion_params()
    consumer.init_orion_params()

    plan = WPCCIPSConvTranspose2dPlan(
        upsample,
        input_shape=(1, 12, 4, 4),
        slots=512,
    )
    plan.level = 2
    plan.output_level = 1
    consumer_plan = __import__(
        "orion.experimental.wpc_cips_layer", fromlist=["WPCCIPSConv2dPlan"]
    ).WPCCIPSConv2dPlan(consumer, input_shape=(1, 12, 8, 8), slots=512)
    consumer_plan.level = 1

    assert plan.output_shape == (1, 12, 8, 8)
    assert plan.input_packing_signature[5] == ((0, 12),)
    assert plan.output_packing_signature[5] == ((0, 8), (8, 12))
    assert plan.output_packing_signature == consumer_plan.input_packing_signature
    assert plan.output_level == consumer_plan.level


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"kernel_size": 3, "stride": 2}, "kernel size 2x2"),
        ({"kernel_size": 2, "stride": 1}, "stride 2x2"),
        ({"kernel_size": 2, "stride": 2, "padding": 1}, "zero padding"),
        (
            {"kernel_size": 2, "stride": 2, "output_padding": 1},
            "zero output padding",
        ),
        ({"kernel_size": 2, "stride": 2, "dilation": 2}, "dilation one"),
    ],
)
def test_transpose2_plan_rejects_unsupported_geometry(kwargs, message) -> None:
    options = {
        "in_channels": 12,
        "out_channels": 12,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "output_padding": 0,
        "dilation": 1,
        "bias": False,
        "level": 2,
        **kwargs,
    }
    layer = ConvTranspose2d(**options)
    with pytest.raises(ValueError, match=message):
        WPCCIPSConvTranspose2dPlan(
            layer,
            input_shape=(1, 12, 4, 4),
            slots=512,
        )
