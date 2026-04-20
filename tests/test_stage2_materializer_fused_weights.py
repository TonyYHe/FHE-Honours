from __future__ import annotations

import pytest
import torch

from orion.experimental.cir.lattigo_block import build_r18_stage2_shared_block_plan


def test_stage2_materializer_accepts_fused_weight_override() -> None:
    weight = torch.randn((128, 128, 3, 3), dtype=torch.float32)
    bias = torch.randn((128,), dtype=torch.float32)
    total = {key: 0 for key in ("rotations", "conjugations", "ct_pt_mults", "adds")}

    for input_pair_index in (0, 1):
        plan, _inputs, _reference = build_r18_stage2_shared_block_plan(
            input_pair_index=int(input_pair_index),
            bank_count=None,
            weight_override=weight,
            bias_override=bias,
            input_shape=(128, 32, 32),
            output_shape=(128, 32, 32),
            input_gap=2,
            output_gap=2,
        )
        for key, value in plan.expected_cost.to_dict().items():
            total[key] += int(value)
        assert any("weight_source=fused_orion_on_weight" in note for note in plan.notes)
        assert any("bias_source=accepted_not_folded" in note for note in plan.notes)

    assert total == {
        "rotations": 84,
        "conjugations": 8,
        "ct_pt_mults": 5880,
        "adds": 5890,
    }


def test_stage2_materializer_rejects_bad_fused_weight_shape() -> None:
    with pytest.raises(ValueError, match="fused weight shape mismatch"):
        build_r18_stage2_shared_block_plan(weight_override=torch.randn((64, 128, 3, 3)))
