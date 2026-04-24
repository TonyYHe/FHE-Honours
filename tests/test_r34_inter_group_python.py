from __future__ import annotations

import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.core import packing
from orion.core.packing import _demultiplex
from orion.experimental.cir.r34_inter_group_python import (
    R34PythonInterGroupSameShapeRuntimeExecutor,
    build_r34_same_shape_generalized_inter_group_assets,
)
from orion.experimental.cir.r34_orion_same_shape import R34SameShapeStageSpec, R34_STAGE2_SAME_SPEC
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


def _encrypt_complex_source(inputs: dict[str, object], level: int) -> CipherTensor:
    left = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
    right = inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
    return scheme.encrypt(scheme.encode(left, level)) + scheme.encrypt(
        scheme.encode(right, level)
    ).mul_imaginary_unit(+1, in_place=False)


def _decode_real_flat(ct: CipherTensor) -> torch.Tensor:
    decoded = ct.decrypt().decode().detach().cpu()
    if torch.is_complex(decoded):
        decoded = decoded.real
    return decoded.to(dtype=torch.float32).flatten()


def _encrypt_orion_packed_source(x: torch.Tensor, *, spec: R34SameShapeStageSpec, level: int) -> CipherTensor:
    packed = packing.multiplex(x.unsqueeze(0), int(spec.gap)).squeeze(0)
    target = torch.zeros((int((int(spec.c) + int(spec.gap * spec.gap) - 1) // int(spec.gap * spec.gap)), int(spec.h * spec.gap), int(spec.w * spec.gap)), dtype=torch.float32)
    target[: packed.shape[0], : packed.shape[1], : packed.shape[2]] = packed
    flat = target.flatten()
    ids: list[int] = []
    slots = int(scheme.params.get_slots())
    for start in range(0, int(flat.numel()), int(slots)):
        block = flat[int(start) : int(min(int(flat.numel()), int(start + int(slots))))]
        padded = torch.zeros((int(slots),), dtype=torch.float32)
        padded[: int(block.numel())] = block
        ct = scheme.encrypt(scheme.encode(padded, level))
        ids.append(int(ct.ids[0]))
        ct.ids = []
    return CipherTensor(
        scheme,
        ids,
        torch.Size([1, int(spec.c), int(spec.h), int(spec.w)]),
        torch.Size([1, int(target.shape[0]), int(target.shape[1]), int(target.shape[2])]),
    )


def test_small_generalized_inter_group_python_assets_recover_reference() -> None:
    spec = R34SameShapeStageSpec(
        family_label="stage2_same_small",
        stage="stage2_small",
        c=128,
        h=28,
        w=28,
        gap=8,
        policy="inter_group_hybrid",
        materializer="policy_inter_group_hybrid",
    )

    _init_python_scheme()
    try:
        assets = build_r34_same_shape_generalized_inter_group_assets(spec=spec)
        level = len(scheme.params.get_logq()) - 1
        reference = assets["reference"].detach().cpu().to(dtype=torch.float32)
        decoded = torch.zeros_like(reference)

        for block in assets["blocks"]:
            plan = block["plan"]
            transforms, _bank_ids = transforms_from_conv_scheme_plan(
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
            for region, output_id in zip(plan.output_regions, output_ids):
                raw = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(scheme.params.get_slots())]),
                    torch.Size([1, int(scheme.params.get_slots())]),
                )
                real = (raw + raw.conjugate(in_place=False)) * 0.5
                decoded_block = _decode_real_flat(real)
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

        max_abs = float((decoded - reference).abs().max().item())
        mae = float((decoded - reference).abs().mean().item())
        assert max_abs <= 2.5e-4
        assert mae <= 1.5e-5
    finally:
        scheme.delete_scheme()


def test_stage2_inter_group_python_runtime_matches_conv_reference() -> None:
    spec = R34_STAGE2_SAME_SPEC
    _init_python_scheme()
    try:
        torch.manual_seed(0)
        x = torch.randn((int(spec.c), int(spec.h), int(spec.w)), dtype=torch.float32)
        weight = torch.randn((int(spec.c), int(spec.c), 3, 3), dtype=torch.float32)
        bias = torch.randn((int(spec.c),), dtype=torch.float32)
        level = len(scheme.params.get_logq()) - 1

        module = type("FakeR34Stage2", (), {})()
        module.on_weight = weight
        module.on_bias = bias
        module.input_shape = torch.Size([1, int(spec.c), int(spec.h), int(spec.w)])
        module.output_shape = torch.Size([1, int(spec.c), int(spec.h), int(spec.w)])
        module.fhe_input_shape = torch.Size([1, int((int(spec.c) + int(spec.gap * spec.gap) - 1) // int(spec.gap * spec.gap)), int(spec.h * spec.gap), int(spec.w * spec.gap)])
        module.fhe_output_shape = torch.Size([1, int((int(spec.c) + int(spec.gap * spec.gap) - 1) // int(spec.gap * spec.gap)), int(spec.h * spec.gap), int(spec.w * spec.gap)])
        module.input_gap = int(spec.gap)
        module.output_gap = int(spec.gap)
        module.stride = (1, 1)
        module.padding = (1, 1)

        executor = R34PythonInterGroupSameShapeRuntimeExecutor(
            module=module,
            spec=spec,
            output_node_id="stage2_same_out",
        )
        source = _encrypt_orion_packed_source(x, spec=spec, level=int(level))
        out = executor(source)["stage2_same_out"]
        decoded = _decode_real_flat(out)
        packed_size = int(module.fhe_output_shape[1] * module.fhe_output_shape[2] * module.fhe_output_shape[3])
        demux = _demultiplex(
            decoded[: int(packed_size)].reshape(1, int(module.fhe_output_shape[1]), int(module.fhe_output_shape[2]), int(module.fhe_output_shape[3])),
            int(spec.gap),
            int(spec.c),
            int(spec.h),
            int(spec.w),
        )[0]
        reference = F.conv2d(x.unsqueeze(0), weight, bias=bias, stride=1, padding=1)[0]
        max_abs = float((demux - reference).abs().max().item())
        mae = float((demux - reference).abs().mean().item())
        assert max_abs <= 2.5e-4
        assert mae <= 1.5e-5
    finally:
        scheme.delete_scheme()
