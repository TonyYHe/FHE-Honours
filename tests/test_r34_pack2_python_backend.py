from __future__ import annotations

from typing import Any

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.nn.module import Module
from orion.nn.unified_transform import UnifiedTransformGroup
from orion.experimental.cir.r34_orion_same_shape import (
    R34_STAGE3_SAME_SPEC,
    R34_STAGE4_SAME_SPEC,
    build_r34_same_shape_pack2_prototype_assets,
)


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


def _encrypt_real_source(block: torch.Tensor, level: int) -> CipherTensor:
    return scheme.encrypt(scheme.encode(block.to(dtype=torch.float32), level))


def _encrypt_complex_source(source: torch.Tensor, level: int) -> CipherTensor:
    return scheme.encrypt(scheme.encode(source.real.to(dtype=torch.float32), level)) + scheme.encrypt(
        scheme.encode(source.imag.to(dtype=torch.float32), level)
    ).mul_imaginary_unit(+1, in_place=False)


def _decode_complex_tensor(ct: CipherTensor) -> torch.Tensor:
    pt = ct.decrypt()
    raw = scheme.backend.DecodeComplex(pt.ids[0])
    slots = int(ct.slots())
    return torch.tensor([complex(raw[2 * i], raw[2 * i + 1]) for i in range(slots)], dtype=torch.complex64)


def _run_baseline_outputs(assets: dict[str, Any]) -> dict[int, CipherTensor]:
    level = len(scheme.params.get_logq()) - 1
    outputs: dict[int, CipherTensor] = {}
    for input_index, payload in sorted(dict(assets["baseline_groups"]).items()):
        group = UnifiedTransformGroup(list(payload["transforms"]))
        group.compile_unified(scheme.backend)
        source_block = assets["inputs"][f"orion_source_block_{int(input_index)}"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        ct_source = _encrypt_real_source(source_block, level)
        output_ids = group.evaluate_unified(int(ct_source.ids[0]), scheme.backend)
        for target_index, output_id in zip(tuple(payload["target_indices"]), output_ids):
            partial = CipherTensor(
                scheme,
                [int(output_id)],
                torch.Size([1, int(scheme.params.get_slots())]),
                torch.Size([1, int(scheme.params.get_slots())]),
            )
            if int(target_index) in outputs:
                outputs[int(target_index)] = outputs[int(target_index)] + partial
            else:
                outputs[int(target_index)] = partial
    return outputs


def _run_prototype_outputs(assets: dict[str, Any]) -> dict[int, CipherTensor]:
    level = len(scheme.params.get_logq()) - 1
    group = UnifiedTransformGroup(list(assets["prototype"]["transforms"]))
    group.compile_unified(scheme.backend)
    ct_source = _encrypt_complex_source(assets["complex_source"], level)
    output_ids = group.evaluate_unified(int(ct_source.ids[0]), scheme.backend)
    outputs: dict[int, CipherTensor] = {}
    for target_index, output_id in zip(tuple(assets["prototype"]["target_indices"]), output_ids):
        raw = CipherTensor(
            scheme,
            [int(output_id)],
            torch.Size([1, int(scheme.params.get_slots())]),
            torch.Size([1, int(scheme.params.get_slots())]),
        )
        outputs[int(target_index)] = (raw + raw.conjugate(in_place=False)) * 0.5
    return outputs


def _assert_proto_matches_current(spec: Any) -> None:
    _init_python_scheme()
    try:
        assets = build_r34_same_shape_pack2_prototype_assets(
            spec=spec,
            input_shape=(int(spec.c), int(spec.h), int(spec.w)),
            output_shape=(int(spec.c), int(spec.h), int(spec.w)),
            input_gap=int(spec.gap),
            output_gap=int(spec.gap),
            level=len(scheme.params.get_logq()) - 1,
            scheme=scheme,
        )
        baseline = _run_baseline_outputs(assets)
        prototype = _run_prototype_outputs(assets)
        assert tuple(sorted(baseline.keys())) == tuple(sorted(prototype.keys()))
        for target_index in sorted(baseline.keys()):
            left = _decode_complex_tensor(baseline[target_index])
            right = _decode_complex_tensor(prototype[target_index])
            assert float((left - right).abs().max()) <= 1.0e-5
    finally:
        scheme.delete_scheme()


def test_r34_stage3_pack2_prototype_matches_current_python_backend() -> None:
    _assert_proto_matches_current(R34_STAGE3_SAME_SPEC)


def test_r34_stage4_pack2_prototype_matches_current_python_backend() -> None:
    _assert_proto_matches_current(R34_STAGE4_SAME_SPEC)
