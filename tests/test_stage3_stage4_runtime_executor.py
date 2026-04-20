from __future__ import annotations

from pathlib import Path

import pytest
import torch

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.orion import scheme
from orion.core.region_cir_replay import _compact_stage4_source_from_regular
from orion.experimental.cir.lattigo_block import (
    build_r18_stage3_shared_block_plan,
    build_r18_stage4_compact_intra_plan,
)
from orion.experimental.cir.runtime_group import (
    RegionFirstRuntimeGroup,
    Stage3RuntimeExecutor,
    Stage4RuntimeExecutor,
    transforms_from_conv_scheme_plan,
)
from orion.nn.linear import Conv2d


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def _encrypt_complex_source(left: torch.Tensor, right: torch.Tensor, level: int) -> CipherTensor:
    ct_left = scheme.encrypt(scheme.encode(left.to(dtype=torch.float32), level))
    ct_right = scheme.encrypt(scheme.encode(right.to(dtype=torch.float32), level))
    return ct_left + ct_right.mul_imaginary_unit(+1, in_place=False)


def test_stage3_runtime_executor_builds_transforms() -> None:
    plan, _inputs, _reference = build_r18_stage3_shared_block_plan(bank_count=None)

    transforms, bank_ids = transforms_from_conv_scheme_plan(plan, level=3, scheme=scheme, bank_count=2)

    assert len(transforms) == 2
    assert len(bank_ids) == 2
    assert all(transform.diagonals[(0, 0)] for transform in transforms)


def test_stage4_runtime_executor_rejects_non_compact_source() -> None:
    executor = Stage4RuntimeExecutor(
        output_node_ids=("conv_a",),
        plan_builder=build_r18_stage4_compact_intra_plan,
    )

    with pytest.raises(RuntimeError, match="compact-intra source layout"):
        executor(type("OpaqueSource", (), {"ids": [11]})())


def test_stage3_runtime_executor_lattigo_outputs_through_conv_proxy(monkeypatch) -> None:
    _require_lattigo()

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("stage3 runtime executor must not call pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
    plan, inputs, _reference = build_r18_stage3_shared_block_plan(bank_count=2)
    executor = Stage3RuntimeExecutor(plan=plan, output_node_ids=("conv_a", "conv_b"))
    group = RegionFirstRuntimeGroup(
        region_id="r18_tiny_stage3",
        network="R18",
        stage="stage3",
        module_prefix="layers.2",
        conv_nodes=("conv_a", "conv_b"),
        strategy="inter_group_shared_lt",
        materializer="build_r18_stage3_shared_block_plan",
        depth=2,
        boundary_actions=("insert_extract_before_relu_or_add",),
        expected_stats={},
        executable=True,
        fallback_reason="",
        executor=executor,
    )
    conv_a = Conv2d(1, 1, 3, padding=1, bias=False)
    conv_b = Conv2d(1, 1, 3, padding=1, bias=False)
    for conv, output_id in ((conv_a, "conv_a"), (conv_b, "conv_b")):
        conv.he_mode = True
        conv.region_runtime = group
        conv.region_output_id = output_id

    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        level = len(scheme.params.get_logq()) - 1
        source = _encrypt_complex_source(
            inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug(),
            inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug(),
            level,
        )

        out_a = conv_a(source)
        out_b = conv_b(source)

        assert isinstance(out_a, CipherTensor)
        assert isinstance(out_b, CipherTensor)
        assert group.execute_count == 1
        assert executor.compile_count == 1
    finally:
        scheme.delete_scheme()


def test_stage4_runtime_executor_lattigo_outputs_through_conv_proxy(monkeypatch) -> None:
    _require_lattigo()

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("stage4 runtime executor must not call pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
    plan, inputs, _reference = build_r18_stage4_compact_intra_plan()
    executor = Stage4RuntimeExecutor(plan=plan, output_node_ids=("conv_a",))
    group = RegionFirstRuntimeGroup(
        region_id="r18_tiny_stage4",
        network="R18",
        stage="stage4",
        module_prefix="layers.3",
        conv_nodes=("conv_a",),
        strategy="compact_intra_group_phase",
        materializer="build_r18_stage4_compact_intra_plan",
        depth=3,
        boundary_actions=("insert_extract_before_relu_or_add",),
        expected_stats={},
        executable=True,
        fallback_reason="",
        executor=executor,
    )
    conv = Conv2d(1, 1, 3, padding=1, bias=False)
    conv.he_mode = True
    conv.region_runtime = group
    conv.region_output_id = "conv_a"

    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        level = len(scheme.params.get_logq()) - 1
        compact = _compact_stage4_source_from_regular(inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32))
        source = _encrypt_complex_source(compact.real, compact.imag, level)
        source.region_first_compact_source = True

        out = conv(source)

        assert isinstance(out, CipherTensor)
        assert group.execute_count == 1
        assert executor.compile_count == 1
    finally:
        scheme.delete_scheme()
