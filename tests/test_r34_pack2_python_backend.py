from __future__ import annotations

from typing import Any

import torch
import gc

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
    return torch.tensor([complex(raw[2 * i], raw[2 * i + 1]) for i in range(slots)], dtype=torch.complex128)


def _run_baseline_outputs(assets: dict[str, Any]) -> dict[int, CipherTensor]:
    level = len(scheme.params.get_logq()) - 1
    outputs: dict[int, CipherTensor] = {}
    for input_index, payload in sorted(dict(assets["baseline_groups"]).items()):
        source_block = assets["inputs"][f"orion_source_block_{int(input_index)}"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        ct_source = _encrypt_real_source(source_block, level)
        for target_index, transform in zip(tuple(payload["target_indices"]), list(payload["transforms"])):
            group = UnifiedTransformGroup([transform])
            group.compile_unified(scheme.backend)
            output_id = group.evaluate_unified(int(ct_source.ids[0]), scheme.backend)[0]
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
            group.cleanup(scheme.backend)
            del group
            gc.collect()
    return outputs


def _run_prototype_outputs(assets: dict[str, Any]) -> dict[int, CipherTensor]:
    level = len(scheme.params.get_logq()) - 1
    ct_source = _encrypt_complex_source(assets["complex_source"], level)
    outputs: dict[int, CipherTensor] = {}
    for target_index, transform in zip(
        tuple(assets["prototype"]["target_indices"]),
        list(assets["prototype"]["transforms"]),
    ):
        group = UnifiedTransformGroup([transform])
        group.compile_unified(scheme.backend)
        output_id = group.evaluate_unified(int(ct_source.ids[0]), scheme.backend)[0]
        raw = CipherTensor(
            scheme,
            [int(output_id)],
            torch.Size([1, int(scheme.params.get_slots())]),
            torch.Size([1, int(scheme.params.get_slots())]),
        )
        outputs[int(target_index)] = (raw + raw.conjugate(in_place=False)) * 0.5
        group.cleanup(scheme.backend)
        del group
        gc.collect()
    return outputs


def _assert_proto_matches_current(spec: Any, *, target_block_index: int) -> None:
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
            target_block_indices=(int(target_block_index),),
        )
        baseline = _run_baseline_outputs(assets)
        prototype = _run_prototype_outputs(assets)
        assert tuple(sorted(baseline.keys())) == tuple(sorted(prototype.keys()))
        for target_index in sorted(baseline.keys()):
            left = _decode_complex_tensor(baseline[target_index])
            right = _decode_complex_tensor(prototype[target_index])
            assert float((left - right).abs().max()) <= 1.0e-5
        del assets
        del baseline
        del prototype
        gc.collect()
    finally:
        scheme.delete_scheme()


def test_r34_stage3_pack2_prototype_matches_current_python_backend_block0() -> None:
    _assert_proto_matches_current(R34_STAGE3_SAME_SPEC, target_block_index=0)


def test_r34_stage3_pack2_prototype_matches_current_python_backend_block1() -> None:
    _assert_proto_matches_current(R34_STAGE3_SAME_SPEC, target_block_index=1)


def test_r34_stage4_pack2_prototype_matches_current_python_backend_block0() -> None:
    _assert_proto_matches_current(R34_STAGE4_SAME_SPEC, target_block_index=0)


def test_r34_stage4_pack2_prototype_matches_current_python_backend_block1() -> None:
    _assert_proto_matches_current(R34_STAGE4_SAME_SPEC, target_block_index=1)
