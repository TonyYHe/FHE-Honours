from __future__ import annotations

import numpy as np
import pytest

from orion.experimental.wpc_cips_layer import WPCCIPSConv2dPlan
from orion.nn import Conv2d


def test_actual_orion_conv2d_builds_chainable_multigroup_plan() -> None:
    first = Conv2d(12, 12, 3, padding=1, bias=True, level=2)
    second = Conv2d(12, 10, 3, padding=1, bias=True, level=1)
    first.name = "block_conv1"
    second.name = "block_conv2"
    first.init_orion_params()
    second.init_orion_params()

    first_plan = WPCCIPSConv2dPlan(
        first,
        input_shape=(1, 12, 8, 8),
        slots=512,
    )
    second_plan = WPCCIPSConv2dPlan(
        second,
        input_shape=(1, 12, 8, 8),
        slots=512,
    )

    assert first_plan.case.input_group_ranges == ((0, 8), (8, 12))
    assert first_plan.case.output_group_ranges == ((0, 8), (8, 12))
    assert first_plan.case.transform_count == 4
    assert second_plan.case.output_group_ranges == ((0, 8), (8, 10))
    assert first_plan.output_packing_signature == second_plan.input_packing_signature

    tensor = np.arange(12 * 8 * 8, dtype=np.float64).reshape(1, 12, 8, 8) / 100
    reference = first_plan.clear_reference(tensor)
    assert reference.shape == (12, 8, 8)
    assert np.isfinite(reference).all()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"stride": 2, "padding": 1}, "stride one"),
        ({"dilation": 2, "padding": 2}, "dilation one"),
        ({"kernel_size": 2, "padding": 1}, "odd kernels"),
    ],
)
def test_wpc_conv2d_plan_rejects_unsupported_geometry(kwargs, message) -> None:
    options = {
        "in_channels": 12,
        "out_channels": 12,
        "kernel_size": 3,
        "padding": 1,
        "bias": False,
        "level": 2,
        **kwargs,
    }
    layer = Conv2d(**options)
    with pytest.raises(ValueError, match=message):
        WPCCIPSConv2dPlan(layer, input_shape=(1, 12, 8, 8), slots=512)


def test_conv2d_forward_dispatches_only_when_wpc_plan_is_installed() -> None:
    layer = Conv2d(1, 1, 1, bias=False)

    class _FakePlan:
        def __init__(self) -> None:
            self.seen = None

        def evaluate(self, value, *, compressed=True):
            self.seen = (value, compressed)
            return "wpc-output"

    fake = _FakePlan()
    layer._wpc_cips_plan = fake
    layer.he()
    marker = object()
    assert layer(marker) == "wpc-output"
    assert fake.seen == (marker, True)
    layer._wpc_cips_plan = None


def test_conv2d_forward_dispatches_full_storage_without_compression() -> None:
    layer = Conv2d(1, 1, 1, bias=False)

    class _FakePlan:
        storage_mode = "full"

        def __init__(self) -> None:
            self.compressed = None

        def evaluate(self, value, *, compressed=True):
            del value
            self.compressed = compressed
            return "full-output"

    fake = _FakePlan()
    layer._wpc_cips_plan = fake
    layer.he()
    assert layer(object()) == "full-output"
    assert fake.compressed is False
    layer._wpc_cips_plan = None


def test_wpc_plan_validates_storage_mode_before_backend_use() -> None:
    layer = Conv2d(1, 1, 1, bias=False)
    layer.init_orion_params()
    plan = WPCCIPSConv2dPlan(layer, input_shape=(1, 1, 8, 8), slots=512)

    with pytest.raises(ValueError, match="storage_mode"):
        plan.compile(object(), storage_mode="unknown")
    with pytest.raises(ValueError, match="requires compressed storage"):
        plan.compile(object(), storage_mode="full", verify_exact_qp=True)


def test_wpc_plan_rejects_coexistence_with_ordinary_transform_resources() -> None:
    layer = Conv2d(1, 1, 1, bias=False)
    layer.transform_ids[(0, 0)] = 123
    layer.scheme = object()

    with pytest.raises(RuntimeError, match="before compiling the ordinary Orion plan"):
        layer.install_wpc_cips_plan((1, 1, 8, 8))
