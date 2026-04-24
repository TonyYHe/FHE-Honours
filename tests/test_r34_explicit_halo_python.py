from __future__ import annotations

import torch

from orion.core.region_lowering import pack_chw_gap
from orion.core.orion import scheme
from orion.experimental.cir.r34_section_extractor import (
    SectionExtractPlan,
    build_core_section_tensor,
    build_section_extract_plan,
    decode_section_flat,
    encode_section_tensor,
    extract_section_ciphertext,
    halo_rotation_count,
)
from orion.nn.module import Module


def _init_python_scheme() -> None:
    config = {
        "ckks_params": {
            "LogN": 16,
            "LogQ": [45, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "python",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    Module.set_scheme(scheme)
    Module.set_margin(scheme.params.get_margin())


def test_stage3_same_explicit_halo_fill_matches_expected_sections() -> None:
    _init_python_scheme()
    try:
        torch.manual_seed(0)
        x = torch.randn((256, 14, 14), dtype=torch.float32)
        level = len(scheme.params.get_logq()) - 1

        left_plan = build_section_extract_plan(
            family_label="stage3_same",
            c=256,
            w=14,
            gap=16,
            input_h=14,
            kernel=3,
            stride=1,
            pad=1,
            target_h_range=(0, 7),
            source_h_range=(0, 9),
        )
        right_plan = build_section_extract_plan(
            family_label="stage3_same",
            c=256,
            w=14,
            gap=16,
            input_h=14,
            kernel=3,
            stride=1,
            pad=1,
            target_h_range=(7, 14),
            source_h_range=(5, 14),
        )

        left_source = x[:, 0:14, :]
        right_source = x[:, 0:14, :]
        left_core = build_core_section_tensor(left_source, plan=left_plan)
        right_core = build_core_section_tensor(right_source, plan=right_plan)
        left_core_ct = encode_section_tensor(left_core, scheme=scheme, level=int(level), plan=left_plan)
        right_core_ct = encode_section_tensor(right_core, scheme=scheme, level=int(level), plan=right_plan)

        left_filled = extract_section_ciphertext(
            center_ct=left_core_ct,
            next_ct=right_core_ct,
            prev_ct=None,
            scheme=scheme,
            level=int(level),
            plan=left_plan,
            next_plan=right_plan,
        )
        right_filled = extract_section_ciphertext(
            center_ct=right_core_ct,
            next_ct=None,
            prev_ct=left_core_ct,
            scheme=scheme,
            level=int(level),
            plan=right_plan,
            prev_plan=left_plan,
        )

        left_expected_tensor = torch.zeros((256, int(left_plan.local_h), 14), dtype=torch.float32)
        left_expected_tensor[:, 0:7, :] = x[:, 0:7, :]
        left_expected_tensor[:, 7:8, :] = x[:, 7:8, :]
        right_expected_tensor = torch.zeros((256, int(right_plan.local_h), 14), dtype=torch.float32)
        right_expected_tensor[:, int(right_plan.core_h_start) : int(right_plan.core_h_end), :] = x[:, 7:14, :]
        right_expected_tensor[:, int(right_plan.core_h_start - right_plan.halo_top) : int(right_plan.core_h_start), :] = x[:, 6:7, :]

        left_expected = pack_chw_gap(
            left_expected_tensor,
            shape=(256, int(left_plan.local_h), 14),
            gap=16,
            slots=32768,
        ).to(dtype=torch.float32)
        right_expected = pack_chw_gap(
            right_expected_tensor,
            shape=(256, int(right_plan.local_h), 14),
            gap=16,
            slots=32768,
        ).to(dtype=torch.float32)

        left_actual = decode_section_flat(left_filled, scheme=scheme)
        right_actual = decode_section_flat(right_filled, scheme=scheme)
        left_without = decode_section_flat(left_core_ct, scheme=scheme)
        right_without = decode_section_flat(right_core_ct, scheme=scheme)

        left_max = float((left_actual - left_expected).abs().max().item())
        right_max = float((right_actual - right_expected).abs().max().item())
        left_without_max = float((left_without - left_expected).abs().max().item())
        right_without_max = float((right_without - right_expected).abs().max().item())

        assert left_max <= 1.0e-5
        assert right_max <= 1.0e-5
        assert left_without_max > left_max
        assert right_without_max > right_max
        assert left_without_max > 1.0e-3
        assert right_without_max > 1.0e-3
    finally:
        scheme.delete_scheme()


def test_stage3_same_explicit_halo_fill_uses_expected_row_stride() -> None:
    plan = build_section_extract_plan(
        family_label="stage3_same",
        c=256,
        w=14,
        gap=16,
        input_h=14,
        kernel=3,
        stride=1,
        pad=1,
        target_h_range=(0, 7),
        source_h_range=(0, 9),
    )
    assert plan.row_stride_slots == int(plan.w) * int(plan.gap) * int(plan.gap)
    assert plan.row_stride_slots == 3584


def test_explicit_halo_fill_uses_two_rotations_for_interior_section() -> None:
    stage3 = build_section_extract_plan(
        family_label="stage3_same",
        c=256,
        w=14,
        gap=16,
        input_h=14,
        kernel=3,
        stride=1,
        pad=1,
        target_h_range=(7, 14),
        source_h_range=(5, 14),
    )
    assert halo_rotation_count(stage3, has_prev=True, has_next=True) == 1
    assert halo_rotation_count(stage3, has_prev=False, has_next=True) == 0
    assert halo_rotation_count(stage3, has_prev=True, has_next=False) == 1

    interior = SectionExtractPlan(
        family_label="interior",
        c=256,
        w=14,
        gap=16,
        kernel=3,
        stride=1,
        pad=1,
        target_h_start=4,
        target_h_end=10,
        source_h_start=2,
        source_h_end=12,
        local_h=10,
        core_h_start=2,
        core_h_end=8,
        halo_top=1,
        halo_bottom=1,
        core_source_start=4,
        core_source_end=10,
        crop_start=2,
        crop_end=8,
        core_shift_rows=0,
    )
    assert halo_rotation_count(interior, has_prev=True, has_next=True) == 2

    stem_like = SectionExtractPlan(
        family_label="stem_conv",
        c=2,
        w=224,
        gap=1,
        kernel=7,
        stride=2,
        pad=3,
        target_h_start=70,
        target_h_end=140,
        source_h_start=78,
        source_h_end=224,
        local_h=146,
        core_h_start=3,
        core_h_end=143,
        halo_top=3,
        halo_bottom=2,
        core_source_start=81,
        core_source_end=221,
        crop_start=31,
        crop_end=101,
        core_shift_rows=-3,
    )
    assert halo_rotation_count(stem_like, has_prev=True, has_next=True) == 3
