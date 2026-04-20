from __future__ import annotations

import pytest
import torch

from orion.experimental.cir.lattigo_block import (
    build_r18_stage3_shared_block_plan,
    build_r18_stage4_compact_intra_plan,
)


def test_stage3_materializer_accepts_fused_weight_override() -> None:
    weight = torch.randn((256, 256, 3, 3), dtype=torch.float32)
    bias = torch.randn((256,), dtype=torch.float32)

    plan, _inputs, _reference = build_r18_stage3_shared_block_plan(
        bank_count=None,
        weight_override=weight,
        bias_override=bias,
        input_shape=(256, 16, 16),
        output_shape=(256, 16, 16),
        input_gap=4,
        output_gap=4,
    )

    assert plan.expected_cost.to_dict() == {
        "rotations": 90,
        "conjugations": 2,
        "ct_pt_mults": 6750,
        "adds": 6753,
    }
    assert any("weight_source=fused_orion_on_weight" in note for note in plan.notes)
    assert any("bias_source=accepted_not_folded" in note for note in plan.notes)


def test_stage4_materializer_accepts_fused_weight_override() -> None:
    weight = torch.randn((512, 512, 3, 3), dtype=torch.float32)
    bias = torch.randn((512,), dtype=torch.float32)

    plan, _inputs, _reference = build_r18_stage4_compact_intra_plan(
        weight_override=weight,
        bias_override=bias,
        input_shape=(512, 8, 8),
        output_shape=(512, 8, 8),
        input_gap=8,
        output_gap=8,
    )

    assert plan.expected_cost.to_dict() == {
        "rotations": 158,
        "conjugations": 1,
        "ct_pt_mults": 9767,
        "adds": 9768,
    }
    assert any("weight_source=fused_orion_on_weight" in note for note in plan.notes)
    assert any("bias_source=accepted_not_folded" in note for note in plan.notes)


def test_stage3_and_stage4_materializers_reject_bad_fused_weight_shape() -> None:
    with pytest.raises(ValueError, match="fused weight shape mismatch"):
        build_r18_stage3_shared_block_plan(weight_override=torch.randn((128, 256, 3, 3)))
    with pytest.raises(ValueError, match="fused weight shape mismatch"):
        build_r18_stage4_compact_intra_plan(weight_override=torch.randn((256, 512, 3, 3)))
