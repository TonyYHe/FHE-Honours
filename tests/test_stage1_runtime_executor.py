from __future__ import annotations

from pathlib import Path

import pytest
import torch

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.orion import scheme
from orion.experimental.cir.lattigo_block import build_r18_stage1_shared_block_plan
from orion.experimental.cir.runtime_group import Stage1RuntimeExecutor, transforms_from_conv_scheme_plan


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def test_stage1_runtime_executor_builds_unified_transforms() -> None:
    plan, _inputs, _reference = build_r18_stage1_shared_block_plan(bank_count=8)

    transforms, bank_ids = transforms_from_conv_scheme_plan(plan, level=3, scheme=scheme, bank_count=8)

    assert len(transforms) == 8
    assert len(bank_ids) == 8
    assert all(transform.diagonals[(0, 0)] for transform in transforms)


def test_stage1_runtime_executor_lattigo_outputs_through_conv_proxy(monkeypatch) -> None:
    _require_lattigo()

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("stage1 runtime executor must not call pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
    plan, inputs, _reference = build_r18_stage1_shared_block_plan(bank_count=2)
    executor = Stage1RuntimeExecutor(plan=plan, output_node_ids=("conv_a", "conv_b"))

    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        level = len(scheme.params.get_logq()) - 1
        left = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        right = inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        ct_left = scheme.encrypt(scheme.encode(left, level))
        ct_right = scheme.encrypt(scheme.encode(right, level))
        source = ct_left + ct_right.mul_imaginary_unit(+1, in_place=False)
        outputs = executor(source)
        assert set(outputs) == {"conv_a", "conv_b"}
        assert all(isinstance(value, CipherTensor) for value in outputs.values())
        assert executor.compile_count == 1
        outputs_again = executor(source)
        assert set(outputs_again) == {"conv_a", "conv_b"}
        assert executor.compile_count == 1
    finally:
        scheme.delete_scheme()
