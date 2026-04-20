from __future__ import annotations

from types import SimpleNamespace

import torch

from orion.experimental.cir.runtime_group import RegionFirstRuntimeGroup
from orion.nn.linear import Conv2d


def test_selected_conv_forward_uses_executable_region_runtime() -> None:
    group = RegionFirstRuntimeGroup(
        region_id="r",
        network="R18",
        stage="stage1",
        module_prefix="layers.0",
        conv_nodes=("conv_a",),
        strategy="test",
        materializer="test",
        depth=2,
        boundary_actions=(),
        expected_stats={},
        executable=True,
        fallback_reason="",
        executor=lambda source: {"conv_a": "region-output"},
    )
    conv = Conv2d(1, 1, 3, padding=1, bias=False)
    conv.he_mode = True
    conv.region_runtime = group
    conv.region_output_id = "conv_a"

    assert conv(SimpleNamespace(ids=[1])) == "region-output"
    assert group.execute_count == 1


def test_unselected_conv_forward_still_uses_cleartext_dense_path() -> None:
    conv = Conv2d(1, 1, 3, padding=1, bias=False)
    conv.eval()
    x = torch.randn((1, 1, 4, 4), dtype=torch.float32)

    out = conv(x)

    assert tuple(out.shape) == (1, 1, 4, 4)
