from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.core.region_cir_replay import _compiled_rotation_key_count
from orion.experimental.cir.r34_orion_same_shape import (
    R34_STAGE3_SAME_SPEC,
    R34_STAGE4_SAME_SPEC,
    build_r34_same_shape_pack2_prototype_assets,
)
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


def _run_baseline(assets: dict[str, Any]) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
    level = len(scheme.params.get_logq()) - 1
    outputs: dict[int, CipherTensor] = {}
    total_compile_s = 0.0
    total_eval_s = 0.0
    total_rotations = 0
    for input_index, payload in sorted(dict(assets["baseline_groups"]).items()):
        print(f"baseline: compile input_group={int(input_index)}", flush=True)
        group = UnifiedTransformGroup(list(payload["transforms"]))
        started = time.time()
        group.compile_unified(scheme.backend)
        total_compile_s += float(time.time() - started)
        total_rotations += int(_compiled_rotation_key_count(group, scheme.backend))
        source_block = assets["inputs"][f"orion_source_block_{int(input_index)}"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        ct_source = _encrypt_real_source(source_block, level)
        print(f"baseline: evaluate input_group={int(input_index)}", flush=True)
        started = time.time()
        output_ids = group.evaluate_unified(int(ct_source.ids[0]), scheme.backend)
        total_eval_s += float(time.time() - started)
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
    decoded = {int(target): _decode_complex_tensor(ct) for target, ct in outputs.items()}
    return {
        "compile_unified_s": float(total_compile_s),
        "evaluate_unified_s": float(total_eval_s),
        "rotation_keys": int(total_rotations),
        "target_count": int(len(outputs)),
    }, decoded


def _run_prototype(assets: dict[str, Any]) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
    level = len(scheme.params.get_logq()) - 1
    group = UnifiedTransformGroup(list(assets["prototype"]["transforms"]))
    print("prototype: compile unified group", flush=True)
    started = time.time()
    group.compile_unified(scheme.backend)
    compile_s = float(time.time() - started)
    rotation_keys = int(_compiled_rotation_key_count(group, scheme.backend))
    ct_source = _encrypt_complex_source(assets["complex_source"], level)
    print("prototype: evaluate unified group", flush=True)
    started = time.time()
    output_ids = group.evaluate_unified(int(ct_source.ids[0]), scheme.backend)
    eval_s = float(time.time() - started)
    outputs: dict[int, torch.Tensor] = {}
    for target_index, output_id in zip(tuple(assets["prototype"]["target_indices"]), output_ids):
        raw = CipherTensor(
            scheme,
            [int(output_id)],
            torch.Size([1, int(scheme.params.get_slots())]),
            torch.Size([1, int(scheme.params.get_slots())]),
        )
        outputs[int(target_index)] = _decode_complex_tensor((raw + raw.conjugate(in_place=False)) * 0.5)
    return {
        "compile_unified_s": float(compile_s),
        "evaluate_unified_s": float(eval_s),
        "rotation_keys": int(rotation_keys),
        "target_count": int(len(outputs)),
    }, outputs


def _spec_from_name(name: str):
    normalized = str(name).lower()
    if normalized == "stage3":
        return R34_STAGE3_SAME_SPEC
    if normalized == "stage4":
        return R34_STAGE4_SAME_SPEC
    raise ValueError(f"unsupported stage {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare current R34 same-shape runtime against block-pack2 prototype on the Python backend.")
    parser.add_argument("--stage", choices=("stage3", "stage4"), required=True)
    args = parser.parse_args()

    spec = _spec_from_name(args.stage)
    _init_python_scheme()
    try:
        print(f"building assets for {spec.family_label}", flush=True)
        assets = build_r34_same_shape_pack2_prototype_assets(
            spec=spec,
            input_shape=(int(spec.c), int(spec.h), int(spec.w)),
            output_shape=(int(spec.c), int(spec.h), int(spec.w)),
            input_gap=int(spec.gap),
            output_gap=int(spec.gap),
            level=len(scheme.params.get_logq()) - 1,
            scheme=scheme,
        )
        print("assets built", flush=True)
        baseline_stats, baseline_outputs = _run_baseline(assets)
        prototype_stats, prototype_outputs = _run_prototype(assets)
        correctness = {
            str(target): float((baseline_outputs[int(target)] - prototype_outputs[int(target)]).abs().max())
            for target in sorted(baseline_outputs.keys())
        }
        payload = {
            "status": "ok",
            "family": str(spec.family_label),
            "baseline": baseline_stats,
            "prototype": prototype_stats,
            "rotation_diff": int(prototype_stats["rotation_keys"]) - int(baseline_stats["rotation_keys"]),
            "rotation_speedup": float(baseline_stats["rotation_keys"]) / float(prototype_stats["rotation_keys"]),
            "compile_speedup": None
            if float(prototype_stats["compile_unified_s"]) == 0.0
            else float(baseline_stats["compile_unified_s"]) / float(prototype_stats["compile_unified_s"]),
            "evaluate_speedup": None
            if float(prototype_stats["evaluate_unified_s"]) == 0.0
            else float(baseline_stats["evaluate_unified_s"]) / float(prototype_stats["evaluate_unified_s"]),
            "max_abs_error_by_target": correctness,
            "notes": list(assets["notes"]),
        }
        print(json.dumps(payload, indent=2))
    finally:
        scheme.delete_scheme()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
