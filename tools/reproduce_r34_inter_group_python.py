from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.core.packing import _demultiplex
from orion.experimental.cir.r34_inter_group_python import (
    build_r34_same_shape_generalized_inter_group_assets,
)
from orion.experimental.cir.r34_orion_same_shape import (
    R34_STAGE1_SAME_SPEC,
    R34_STAGE2_SAME_SPEC,
)
from orion.experimental.cir.runtime_group import transforms_from_conv_scheme_plan
from orion.nn.module import Module
from orion.nn.unified_transform import UnifiedTransformGroup


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


def _encrypt_complex_source(inputs: dict[str, Any], level: int) -> CipherTensor:
    left = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
    right = inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
    return scheme.encrypt(scheme.encode(left, level)) + scheme.encrypt(
        scheme.encode(right, level)
    ).mul_imaginary_unit(+1, in_place=False)


def _decode_real_tensor(ct: CipherTensor) -> torch.Tensor:
    decoded = ct.decrypt().decode().detach().cpu()
    if torch.is_complex(decoded):
        decoded = decoded.real
    return decoded.to(dtype=torch.float32).flatten()


def _run_family(spec) -> dict[str, Any]:
    assets = build_r34_same_shape_generalized_inter_group_assets(spec=spec)
    level = len(scheme.params.get_logq()) - 1
    reference = assets["reference"].detach().cpu().to(dtype=torch.float32)
    decoded = torch.zeros_like(reference)
    block_rows: list[dict[str, Any]] = []
    for block in assets["blocks"]:
        plan = block["plan"]
        block_reference = block["reference"].detach().cpu().to(dtype=torch.float32)
        transforms, bank_ids = transforms_from_conv_scheme_plan(
            plan,
            level=int(level),
            scheme=scheme,
            bank_count=len(plan.output_regions),
        )
        group = UnifiedTransformGroup(transforms)
        group.compile_unified(scheme.backend)
        ct_source = _encrypt_complex_source(block["inputs"], level)
        output_ids = group.evaluate_unified(int(ct_source.ids[0]), scheme.backend)
        th0, th1 = block["target_h_range"]
        sh0, sh1 = block["source_h_range"]
        crop_start = int(th0 - sh0)
        crop_end = int(th1 - sh0)
        block_decoded = torch.zeros_like(block_reference)
        for region, output_id in zip(plan.output_regions, output_ids):
            raw = CipherTensor(
                scheme,
                [int(output_id)],
                torch.Size([1, int(scheme.params.get_slots())]),
                torch.Size([1, int(scheme.params.get_slots())]),
            )
            real = (raw + raw.conjugate(in_place=False)) * 0.5
            decoded_block = _decode_real_tensor(real)
            c0 = int(region.c_start)
            c1 = int(region.c_end)
            h_len = int(region.h_end - region.h_start)
            on_c = max(1, (int(c1 - c0) + int(spec.gap * spec.gap) - 1) // int(spec.gap * spec.gap))
            on_h = int(h_len * spec.gap)
            on_w = int(spec.w * spec.gap)
            packed_size = int(on_c * on_h * on_w)
            reshaped = _demultiplex(
                decoded_block[: int(packed_size)].reshape(1, int(on_c), int(on_h), int(on_w)),
                int(spec.gap),
                int(c1 - c0),
                int(h_len),
                int(spec.w),
            )[0]
            decoded[int(c0): int(c1), int(th0): int(th1), :] += reshaped[:, int(crop_start): int(crop_end), :]
            block_decoded[int(c0): int(c1)] += reshaped
        block_target = block_reference[:, int(crop_start): int(crop_end), :]
        diff = block_decoded[:, int(crop_start): int(crop_end), :] - block_target
        block_rows.append(
            {
                "in_range": list(block["in_range"]),
                "target_h_range": [int(th0), int(th1)],
                "mae": float(diff.abs().mean().item()),
                "max_abs": float(diff.abs().max().item()),
                "bank_count": int(len(bank_ids)),
            }
        )
    diff = decoded - reference
    return {
        "family": str(spec.family_label),
        "bounded": dict(assets["bounded"]),
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "block_rows": block_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Python-backend generalized inter-group prototype for R34 stage1/2.")
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    args = parser.parse_args()
    spec = R34_STAGE1_SAME_SPEC if str(args.stage) == "stage1" else R34_STAGE2_SAME_SPEC
    _init_python_scheme()
    try:
        payload = _run_family(spec)
        print(json.dumps(payload, indent=2))
    finally:
        scheme.delete_scheme()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
