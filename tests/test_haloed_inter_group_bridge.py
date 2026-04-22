from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.core.packing import _demultiplex
from orion.experimental.cir.haloed_bridge import (
    haloed_inputs_to_orion,
    haloed_plan_to_orion,
    transform_from_orion_plan_step,
)
from orion.nn.module import Module
from orion.nn.unified_transform import UnifiedTransformGroup


HALOED_ROOT = Path("/home/anakano/CLionProjects/HaloED")


def _require_haloed() -> None:
    if not HALOED_ROOT.exists():
        pytest.skip("local HaloED checkout is not available")


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


def test_haloed_direct_inter_group_plan_bridges_into_orion_python_backend() -> None:
    _require_haloed()
    sys.path.insert(0, str(HALOED_ROOT))
    sys.path.insert(0, str(HALOED_ROOT / "src"))
    from scripts.cir.model_axis_actual_kernel_counts import ActualKernelSpec, build_inter_group_plan

    spec = ActualKernelSpec(
        case_name="inter_group_shared_unit",
        model="unit",
        family="inter_group_shared",
        c=32,
        h=32,
        w=64,
        gap=4,
        kernel=1,
        stride=1,
        pad=0,
    )

    _init_python_scheme()
    try:
        haloed_plan, haloed_inputs, reference = build_inter_group_plan(spec)
        plan = haloed_plan_to_orion(haloed_plan)
        inputs = haloed_inputs_to_orion(haloed_inputs)
        level = len(scheme.params.get_logq()) - 1

        transforms = [
            transform_from_orion_plan_step(
                plan=plan,
                step=step,
                level=int(level),
                scheme=scheme,
                name=f"haloed_bridge_t{int(index)}",
            )
            for index, step in enumerate(plan.linear_transform_steps)
        ]
        group = UnifiedTransformGroup(transforms)
        group.compile_unified(scheme.backend)

        left = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        right = inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        ct = scheme.encrypt(scheme.encode(left, level)) + scheme.encrypt(
            scheme.encode(right, level)
        ).mul_imaginary_unit(+1, in_place=False)

        output_ids = group.evaluate_unified(int(ct.ids[0]), scheme.backend)
        local_c = 16
        recovered = torch.zeros_like(reference)
        on_h = int(spec.h * spec.gap)
        on_w = int(spec.w * spec.gap)
        for surface_index, output_id in enumerate(output_ids):
            raw = CipherTensor(
                scheme,
                [int(output_id)],
                torch.Size([1, int(scheme.params.get_slots())]),
                torch.Size([1, int(scheme.params.get_slots())]),
            )
            real = (raw + raw.conjugate(in_place=False)) * 0.5
            decoded = real.decrypt().decode().detach().cpu().reshape(1, 1, on_h, on_w)
            if torch.is_complex(decoded):
                decoded = decoded.real
            demux = _demultiplex(decoded, int(spec.gap), int(local_c), int(spec.h), int(spec.w))[0]
            recovered[int(surface_index * local_c) : int((surface_index + 1) * local_c)] = demux

        max_abs = float((recovered - reference).abs().max().item())
        mae = float((recovered - reference).abs().mean().item())
        assert max_abs <= 1.0e-4
        assert mae <= 1.0e-5
    finally:
        scheme.delete_scheme()
