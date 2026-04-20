from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.orion import scheme
from orion.experimental.cir.lattigo_block import build_r18_stage2_shared_block_plan
from orion.experimental.cir.runtime_group import (
    RegionFirstRuntimeGroup,
    Stage2RuntimeExecutor,
    transforms_from_conv_scheme_plan,
)
from orion.nn.linear import Conv2d


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def _encrypt_complex_source(inputs: dict[str, object], level: int) -> CipherTensor:
    left = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
    right = inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
    ct_left = scheme.encrypt(scheme.encode(left, level))
    ct_right = scheme.encrypt(scheme.encode(right, level))
    return ct_left + ct_right.mul_imaginary_unit(+1, in_place=False)


def test_stage2_runtime_executor_builds_two_block_transforms() -> None:
    plans = tuple(
        build_r18_stage2_shared_block_plan(input_pair_index=input_pair_index, bank_count=None)[0]
        for input_pair_index in (0, 1)
    )

    for plan in plans:
        transforms, bank_ids = transforms_from_conv_scheme_plan(plan, level=3, scheme=scheme, bank_count=4)
        assert len(transforms) == 4
        assert len(bank_ids) == 4
        assert all(transform.diagonals[(0, 0)] for transform in transforms)


def test_stage2_runtime_executor_rejects_single_opaque_source() -> None:
    plans = tuple(
        build_r18_stage2_shared_block_plan(input_pair_index=input_pair_index, bank_count=2)[0]
        for input_pair_index in (0, 1)
    )
    executor = Stage2RuntimeExecutor(plans=plans, output_node_ids=("conv_a", "conv_b"))

    with pytest.raises(RuntimeError, match="two input surface-pair blocks"):
        executor._source_ids_for_blocks(SimpleNamespace(ids=[11]))


def test_stage2_runtime_executor_lattigo_outputs_through_conv_proxy(monkeypatch) -> None:
    _require_lattigo()

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("stage2 runtime executor must not call pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
    plan0, inputs0, _reference0 = build_r18_stage2_shared_block_plan(input_pair_index=0, bank_count=2)
    plan1, inputs1, _reference1 = build_r18_stage2_shared_block_plan(input_pair_index=1, bank_count=2)
    executor = Stage2RuntimeExecutor(plans=(plan0, plan1), output_node_ids=("conv_a", "conv_b"))
    group = RegionFirstRuntimeGroup(
        region_id="r18_tiny_stage2",
        network="R18",
        stage="stage2",
        module_prefix="layers.1",
        conv_nodes=("conv_a", "conv_b"),
        strategy="inter_group_shared_lt",
        materializer="build_r18_stage2_shared_block_plan",
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
        block0_source = _encrypt_complex_source(inputs0, level)
        block1_source = _encrypt_complex_source(inputs1, level)
        source = SimpleNamespace(
            scheme=scheme,
            ids=[int(block0_source.ids[0]), int(block1_source.ids[0])],
            region_first_block_sources=(block0_source, block1_source),
        )

        out_a = conv_a(source)
        out_b = conv_b(source)

        assert isinstance(out_a, CipherTensor)
        assert isinstance(out_b, CipherTensor)
        assert group.execute_count == 1
        assert executor.compile_count == 1
        assert executor.block_evaluate_count == 2
    finally:
        scheme.delete_scheme()
