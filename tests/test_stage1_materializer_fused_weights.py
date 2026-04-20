from __future__ import annotations

import pytest
import torch

from orion.experimental.cir.lattigo_block import build_r18_stage1_shared_block_plan


def test_stage1_materializer_accepts_fused_weight_override() -> None:
    weight = torch.randn((64, 64, 3, 3), dtype=torch.float32)
    bias = torch.randn((64,), dtype=torch.float32)

    plan, _inputs, _reference = build_r18_stage1_shared_block_plan(
        bank_count=8,
        weight_override=weight,
        bias_override=bias,
        input_shape=(64, 64, 64),
        output_shape=(64, 64, 64),
        input_gap=1,
        output_gap=1,
    )

    assert plan.expected_cost.to_dict() == {
        "rotations": 20,
        "conjugations": 8,
        "ct_pt_mults": 1080,
        "adds": 1089,
    }
    assert any("weight_source=fused_orion_on_weight" in note for note in plan.notes)
    assert any("bias_source=accepted_not_folded" in note for note in plan.notes)


def test_stage1_materializer_rejects_bad_fused_weight_shape() -> None:
    with pytest.raises(ValueError, match="fused weight shape mismatch"):
        build_r18_stage1_shared_block_plan(weight_override=torch.randn((32, 64, 3, 3)))
