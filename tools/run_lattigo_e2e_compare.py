from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.orion import scheme
from orion.core import packing
from orion.backend.python.tensors import CipherTensor
from orion.backend.python.memory_lifecycle import trim_runtime_memory
from orion.backend.python.compile_policy import auto_worker_count, policy_audit
from orion.models.resnet import ResNet18, ResNet20, ResNet34
from orion.models.unet import UNet22, UNet22Encoder
from orion.models.vgg import VGG
from orion.nn.linear import LinearTransform
from orion.nn.module import Module
from orion.nn.operations import Bootstrap

try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:
    def _dynamo_disable(fn):
        return fn


DEFAULT_OUT = Path("/tmp/orion_lattigo_e2e_compare.json")


def _layout_policy_input_layout_row() -> dict[str, Any] | None:
    audit = dict(getattr(scheme, "region_first_attach_audit", {}) or {})
    graph = dict(audit.get("graph_audit", {}) or {})
    for row in graph.get("layout_policy_node_layouts", []) or []:
        if str(dict(row).get("node", "")) != "x":
            continue
        layout = dict(dict(row).get("selected_layout", {}) or {})
        if int(layout.get("alpha", 0)) <= 0 and int(layout.get("beta", 0)) <= 0:
            return None
        return dict(row)
    return None


def _layout_policy_plaintext_halo_input(x: torch.Tensor, row: dict[str, Any]) -> torch.Tensor:
    layout = dict(row.get("selected_layout", {}) or {})
    gap = max(1, int(layout.get("gap", 1)))
    alpha = max(0, int(layout.get("alpha", 0)))
    beta = max(0, int(layout.get("beta", 0)))
    compact = packing.multiplex(x, int(gap)).detach().cpu().to(dtype=torch.float32)
    if compact.dim() != 4:
        raise ValueError(f"layout-policy input halo expects NCHW input, got {tuple(compact.shape)}")
    top_rows = int(alpha * gap)
    bottom_rows = int(beta * gap)
    if top_rows <= 0 and bottom_rows <= 0:
        return compact
    halo = torch.zeros(
        (
            int(compact.shape[0]),
            int(compact.shape[1]),
            int(compact.shape[2]) + int(top_rows) + int(bottom_rows),
            int(compact.shape[3]),
        ),
        dtype=torch.float32,
    )
    halo[:, :, int(top_rows) : int(top_rows) + int(compact.shape[2]), :] = compact
    if top_rows > 0:
        for h in range(int(top_rows)):
            halo[:, :, int(h), :] = compact[:, :, 0, :]
    if bottom_rows > 0:
        start = int(top_rows + compact.shape[2])
        for h in range(int(bottom_rows)):
            halo[:, :, int(start + h), :] = compact[:, :, int(compact.shape[2]) - 1, :]
    return halo


def _walk_executor_objects(root: Any):
    stack = [root]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("base_executor", "delegate", "executor"):
            child = getattr(current, attr, None)
            if child is not None:
                stack.append(child)


_RUNTIME_FAIRNESS_NUMERIC_KEYS = (
    "resident_compute_s",
    "serving_hot_s",
    "artifact_read_s",
    "artifact_load_s",
    "artifact_unload_s",
    "trim_s",
    "read_bundle_s",
    "load_keys_s",
    "load_plaintexts_s",
    "eval_s",
    "eval_total_s",
    "unload_s",
)


def _append_runtime_group(groups: list[Any], seen: set[int], group: Any) -> None:
    if group is None or id(group) in seen:
        return
    seen.add(id(group))
    groups.append(group)


def _executor_unified_groups(executor: Any) -> list[Any]:
    groups: list[Any] = []
    seen: set[int] = set()
    for candidate in _walk_executor_objects(executor):
        _append_runtime_group(groups, seen, getattr(candidate, "group", None))
        for attr in ("groups", "groups_by_input_block", "groups_by_input_chunk", "groups_by_pair"):
            value = getattr(candidate, attr, None)
            if isinstance(value, dict):
                for _key, group in sorted(value.items()):
                    _append_runtime_group(groups, seen, group)
            else:
                for group in list(value or []):
                    _append_runtime_group(groups, seen, group)
        groups_by_input_index = getattr(candidate, "groups_by_input_index", None)
        if isinstance(groups_by_input_index, dict):
            for _key, group in sorted(groups_by_input_index.items()):
                _append_runtime_group(groups, seen, group)
    return groups


def _runtime_fairness_mode_from_env() -> str:
    raw = os.environ.get("ORION_LATTIGO_STREAMING_LT", "")
    if str(raw).strip().lower() in {"1", "true", "yes", "on", "force", "always"}:
        return "streaming_eval_encode"
    return "unknown"


def _aggregate_runtime_fairness(timings: list[dict[str, Any]], *, serving_hot_s: float) -> dict[str, Any]:
    payload: dict[str, Any] = {key: 0.0 for key in _RUNTIME_FAIRNESS_NUMERIC_KEYS}
    modes: list[str] = []
    resident_available = True
    for timing in timings:
        mode = str(timing.get("runtime_fairness_mode", "unknown") or "unknown")
        modes.append(mode)
        if mode == "streaming_eval_encode":
            resident_available = False
        for key in _RUNTIME_FAIRNESS_NUMERIC_KEYS:
            value = timing.get(key)
            if value is None:
                if key == "resident_compute_s":
                    resident_available = False
                continue
            try:
                payload[key] = float(payload.get(key, 0.0)) + float(value)
            except (TypeError, ValueError):
                if key == "resident_compute_s":
                    resident_available = False
    if not timings:
        mode = _runtime_fairness_mode_from_env()
        resident_available = False
    elif any(mode == "streaming_eval_encode" for mode in modes):
        mode = "streaming_eval_encode"
    elif any(mode == "memory_bounded_load_eval" for mode in modes):
        mode = "memory_bounded_load_eval"
    elif modes and all(mode == "resident_compute" for mode in modes):
        mode = "resident_compute"
    else:
        mode = "unknown"
        resident_available = False
    payload["serving_hot_s"] = float(serving_hot_s)
    if not resident_available:
        payload["resident_compute_s"] = None
    payload["runtime_fairness_mode"] = str(mode)
    payload["source_count"] = int(len(timings))
    return payload


def _collect_runtime_fairness(net: torch.nn.Module, *, serving_hot_s: float) -> dict[str, Any]:
    timings: list[dict[str, Any]] = []
    for _module_name, module in net.named_modules():
        executor = getattr(getattr(module, "region_runtime", None), "executor", None)
        for group in _executor_unified_groups(executor):
            timing = getattr(group, "last_runtime_timing", None)
            if isinstance(timing, dict):
                timings.append(dict(timing))
    evaluator_timing = getattr(getattr(scheme, "lt_evaluator", None), "last_runtime_timing", None)
    if isinstance(evaluator_timing, dict) and not timings:
        timings.append(dict(evaluator_timing))
    return _aggregate_runtime_fairness(timings, serving_hot_s=float(serving_hot_s))


def _model_input_native_halo_plan(net: torch.nn.Module) -> dict[str, Any] | None:
    for module_name, module in net.named_modules():
        runtime = getattr(module, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        if runtime is None or executor is None:
            continue
        if not bool(getattr(executor, "native_halo_input", False)):
            continue
        if tuple(getattr(executor, "relayout_rows", ()) or ()):
            continue
        native_rows = tuple(dict(row) for row in (getattr(executor, "native_input_rows", ()) or ()))
        if not any(str(row.get("source", "")) == "x" for row in native_rows):
            continue
        plan_candidates: list[tuple[Any, Any]] = []
        for candidate in _walk_executor_objects(executor):
            plan = getattr(candidate, "native_plan", None)
            if plan is not None and hasattr(plan, "input_ct_count") and hasattr(plan, "stripes"):
                plan_candidates.append((candidate, plan))
        if plan_candidates:
            candidate, plan = next(
                (
                    (candidate, plan)
                    for candidate, plan in plan_candidates
                    if type(candidate).__name__ == "NativeHaloStripeNoRIConvExecutor"
                ),
                plan_candidates[-1],
            )
            return {
                "node": str(getattr(runtime, "module_prefix", "") or getattr(module, "region_output_id", "") or module_name),
                "module_path": str(module_name),
                "plan": plan,
                "native_rows": [dict(row) for row in native_rows],
                "executor": type(executor).__name__,
                "native_executor": type(candidate).__name__,
            }
    return None


def _encrypt_native_halo_model_input(x: torch.Tensor, input_level: int, native_input: dict[str, Any]) -> CipherTensor:
    from orion.experimental.cir.native_halo_conv2d import native_halo_source_plaintext_blocks_from_nchw

    plan = native_input["plan"]
    blocks = native_halo_source_plaintext_blocks_from_nchw(x, plan)
    ids: list[int] = []
    for block in blocks:
        ct = scheme.encrypt(scheme.encode(block, int(input_level)))
        ids.append(int(ct.ids[0]))
        ct.ids = []
    slots = int(getattr(plan.spec, "slot_count", scheme.params.get_slots()))
    return CipherTensor(
        scheme,
        ids,
        torch.Size(tuple(int(value) for value in x.shape)),
        torch.Size([int(len(ids)), int(slots)]),
    )


def _encrypt_model_input(
    x: torch.Tensor,
    input_level: int,
    *,
    net: torch.nn.Module | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    if net is not None:
        native_input = _model_input_native_halo_plan(net)
        if native_input is not None:
            ct = _encrypt_native_halo_model_input(x, int(input_level), native_input)
            if payload is not None:
                plan = native_input["plan"]
                payload["model_input_encoding"] = {
                    "kind": "native_halo_plaintext_source_tiles",
                    "node": str(native_input["node"]),
                    "module_path": str(native_input["module_path"]),
                    "executor": str(native_input["executor"]),
                    "native_executor": str(native_input["native_executor"]),
                    "input_ct_count": int(len(ct.ids)),
                    "native_plan_input_ct_count": int(plan.input_ct_count),
                    "stripe_count": int(len(plan.stripes)),
                    "source_channel_group_count": int(plan.source_channel_group_count),
                    "source_channel_tile": int(plan.source_channel_tile),
                    "slot_count": int(plan.spec.slot_count),
                }
            return ct
    row = _layout_policy_input_layout_row()
    if row is None:
        return scheme.encrypt(scheme.encode(x, int(input_level)))
    halo = _layout_policy_plaintext_halo_input(x, row)
    ct = scheme.encrypt(scheme.encode(halo, int(input_level)))
    if payload is not None:
        payload["model_input_encoding"] = {
            "kind": "flat_halo_plaintext",
            "input_ct_count": int(len(getattr(ct, "ids", ()) or ())),
            "layout": dict(row.get("selected_layout", {}) or {}),
        }
    return ct


def _r20_config(provider_mode: str, *, backend: str = "lattigo") -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": 16,
            "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            "LogP": [61, 61, 61],
            "LogScale": 40,
            "H": 192,
            "RingType": "Standard",
        },
        "boot_params": {"LogP": [61, 61, 61, 61, 61, 61, 61, 61]},
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": str(backend),
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
            "experimental_region_first": str(provider_mode),
        },
    }


def _r18_config(provider_mode: str, *, backend: str = "lattigo") -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": 16,
            "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            "LogP": [61, 61, 61],
            "LogScale": 40,
            "H": 192,
            "RingType": "Standard",
        },
        "boot_params": {"LogP": [61, 61, 61, 61, 61, 61, 61, 61]},
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": str(backend),
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
            "experimental_region_first": str(provider_mode),
        },
    }


def _r34_config(provider_mode: str, *, backend: str = "lattigo") -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": 16,
            "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            "LogP": [61, 61, 61],
            "LogScale": 40,
            "H": 192,
            "RingType": "Standard",
        },
        "boot_params": {"LogP": [61, 61, 61, 61, 61, 61, 61, 61]},
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": str(backend),
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
            "experimental_region_first": str(provider_mode),
        },
    }


def _u22_config(*, logn: int, provider_mode: str, backend: str = "lattigo") -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [45, 30, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": str(backend),
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
            "experimental_region_first": str(provider_mode),
        },
    }


def _activation_or_relu(activation: str | None) -> str:
    return str(activation or "relu").lower()


def _stem_relu_for_activation(activation: str | None) -> bool:
    return _activation_or_relu(activation) == "relu"


def _build_r18_tiny(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return ResNet18(
        dataset="tiny",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
        stem_relu=_stem_relu_for_activation(activation),
    )


def _build_r20_cifar10(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return ResNet20(dataset="cifar10")


def _build_r18_imgnet(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return ResNet18(
        dataset="imagenet",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
        stem_relu=_stem_relu_for_activation(activation),
    )


def _build_r34_imgnet(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return ResNet34(
        dataset="imagenet",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
        stem_relu=_stem_relu_for_activation(activation),
    )


def _build_vgg16_imgnet(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return VGG(
        "VGG16",
        dataset="imagenet",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
    )


def _build_u22_64_base32(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22(
        dataset="montgomery_lung_64",
        base_dim=32,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_64_base8(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22(
        dataset="montgomery_lung_64",
        base_dim=8,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_256_base32(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22(
        dataset="kvasir_polyp_256",
        base_dim=32,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_256_base8(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22(
        dataset="kvasir_polyp_256",
        base_dim=8,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_256_base8_encoder(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22Encoder(
        dataset="kvasir_polyp_256",
        base_dim=8,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_256_base64_encoder(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22Encoder(
        dataset="kvasir_polyp_256",
        base_dim=64,
        activation=activation,
        silu_degree=int(silu_degree),
    )


NETWORKS: dict[str, dict[str, Any]] = {
    "resnet20_cifar10": {
        "label": "ResNet20 CIFAR10",
        "model": "ResNet20",
        "dataset": "cifar10",
        "input_shape": (1, 3, 32, 32),
        "provider_mode": "",
        "config": _r20_config,
        "builder": _build_r20_cifar10,
    },
    "r18_tiny": {
        "label": "R18 Tiny",
        "model": "ResNet18",
        "dataset": "tiny",
        "input_shape": (1, 3, 64, 64),
        "provider_mode": "r18_tiny_e2e",
        "config": _r18_config,
        "builder": _build_r18_tiny,
    },
    "vgg16_imgnet": {
        "label": "VGG16 ImageNet",
        "model": "VGG16",
        "dataset": "imagenet",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "vgg_imgnet_layout_dp",
        "config": _r34_config,
        "builder": _build_vgg16_imgnet,
    },
    "r18_imgnet": {
        "label": "R18 ImageNet",
        "model": "ResNet18",
        "dataset": "imagenet",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "r18_imgnet_layout_dp",
        "config": _r34_config,
        "builder": _build_r18_imgnet,
    },
    "r34_imgnet": {
        "label": "R34 ImageNet",
        "model": "ResNet34",
        "dataset": "imagenet",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "r34_imgnet_phase1",
        "config": _r34_config,
        "builder": _build_r34_imgnet,
    },
    "u22_64_base32": {
        "label": "U22 64 base32",
        "model": "UNet22",
        "dataset": "montgomery_lung_64",
        "input_shape": (1, 1, 64, 64),
        "provider_mode": "u22_64_base32",
        "config": lambda provider_mode, *, backend="lattigo": _u22_config(
            logn=16,
            provider_mode=str(provider_mode),
            backend=str(backend),
        ),
        "builder": _build_u22_64_base32,
    },
    "u22_64_base8": {
        "label": "U22 64 base8",
        "model": "UNet22",
        "dataset": "montgomery_lung_64",
        "input_shape": (1, 1, 64, 64),
        "provider_mode": "u22_64_base8",
        "config": lambda provider_mode, *, backend="lattigo": _u22_config(
            logn=16,
            provider_mode=str(provider_mode),
            backend=str(backend),
        ),
        "builder": _build_u22_64_base8,
    },
    "u22_256_base32": {
        "label": "U22 256 base32",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 256, 256),
        "provider_mode": "u22_256_base32",
        "config": lambda provider_mode, *, backend="lattigo": _u22_config(
            logn=16,
            provider_mode=str(provider_mode),
            backend=str(backend),
        ),
        "builder": _build_u22_256_base32,
    },
    "u22_224_base32": {
        "label": "U22 224 base32",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "u22_256_base32",
        "config": lambda provider_mode, *, backend="lattigo": _u22_config(
            logn=16,
            provider_mode=str(provider_mode),
            backend=str(backend),
        ),
        "builder": _build_u22_256_base32,
    },
    "u22_192_base32": {
        "label": "U22 192 base32",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 192, 192),
        "provider_mode": "u22_256_base32",
        "config": lambda provider_mode, *, backend="lattigo": _u22_config(
            logn=16,
            provider_mode=str(provider_mode),
            backend=str(backend),
        ),
        "builder": _build_u22_256_base32,
    },
    "u22_256_base8": {
        "label": "U22 256 base8",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 256, 256),
        "provider_mode": "u22_256_base8",
        "config": lambda provider_mode, *, backend="lattigo": _u22_config(
            logn=16,
            provider_mode=str(provider_mode),
            backend=str(backend),
        ),
        "builder": _build_u22_256_base8,
    },
    "u22_256_base8_encoder": {
        "label": "U22 256 base8 encoder",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 256, 256),
        "provider_mode": "u22_256_base8",
        "config": lambda provider_mode, *, backend="lattigo": _u22_config(
            logn=16,
            provider_mode=str(provider_mode),
            backend=str(backend),
        ),
        "builder": _build_u22_256_base8_encoder,
        "scope": "encoder",
    },
    "u22_192_base64_encoder": {
        "label": "U22 192 base64 encoder",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 192, 192),
        "provider_mode": "u22_256_base32",
        "config": lambda provider_mode, *, backend="lattigo": _u22_config(
            logn=16,
            provider_mode=str(provider_mode),
            backend=str(backend),
        ),
        "builder": _build_u22_256_base64_encoder,
        "scope": "encoder",
        "base_dim": 64,
    },
    "u22_224_base64_encoder": {
        "label": "U22 224 base64 encoder",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "u22_256_base32",
        "config": lambda provider_mode, *, backend="lattigo": _u22_config(
            logn=16,
            provider_mode=str(provider_mode),
            backend=str(backend),
        ),
        "builder": _build_u22_256_base64_encoder,
        "scope": "encoder",
        "base_dim": 64,
    },
    "u22_256_base64_encoder": {
        "label": "U22 256 base64 encoder",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 256, 256),
        "provider_mode": "u22_256_base32",
        "config": lambda provider_mode, *, backend="lattigo": _u22_config(
            logn=16,
            provider_mode=str(provider_mode),
            backend=str(backend),
        ),
        "builder": _build_u22_256_base64_encoder,
        "scope": "encoder",
        "base_dim": 64,
    },
}


def _write(payload: dict[str, Any], out_path: Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _apply_io_config(
    config: dict[str, Any],
    *,
    backend: str | None = None,
    io_mode: str = "none",
    io_dir: Path | None = None,
    diags_path: Path | None = None,
    keys_path: Path | None = None,
    logn_override: int | None = None,
) -> dict[str, Any]:
    config = dict(config)
    config["ckks_params"] = dict(config.get("ckks_params", {}))
    config["orion"] = dict(config.get("orion", {}))
    if backend is not None:
        config["orion"]["backend"] = str(backend)
    if logn_override is not None:
        config["ckks_params"]["LogN"] = int(logn_override)
    config["orion"]["io_mode"] = str(io_mode)
    if io_dir is not None:
        diags_path = Path(io_dir) / "diagonals.h5" if diags_path is None else Path(diags_path)
        keys_path = Path(io_dir) / "keys.h5" if keys_path is None else Path(keys_path)
    if diags_path is not None:
        Path(diags_path).parent.mkdir(parents=True, exist_ok=True)
        config["orion"]["diags_path"] = str(Path(diags_path))
    if keys_path is not None:
        Path(keys_path).parent.mkdir(parents=True, exist_ok=True)
        config["orion"]["keys_path"] = str(Path(keys_path))
    return config


def _apply_ckks_preset(config: dict[str, Any], preset: str | None) -> dict[str, Any]:
    normalized = str(preset or "network-default").strip().lower()
    if normalized in {"", "network-default", "default"}:
        return config
    if normalized != "resnet":
        raise ValueError(f"Unsupported CKKS preset {preset!r}")
    config = dict(config)
    config["ckks_params"] = {
        "LogN": 16,
        "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
        "LogP": [61, 61, 61],
        "LogScale": 40,
        "H": 192,
        "RingType": "Standard",
    }
    config["boot_params"] = {"LogP": [61, 61, 61, 61, 61, 61, 61, 61]}
    return config


def _configure_cheddar_runtime_defaults() -> dict[str, str]:
    defaults = {
        "ORION_CHEDDAR_LT_STREAMING": "auto",
        "ORION_UNIFIED_LT_ROTKEY_RESIDENCY": "1",
        "ORION_UNIFIED_LT_PLAINTEXT_RESIDENCY": "1",
        "ORION_CHEDDAR_SHARED_CACHE_PLAN_PERSIST": "1",
        "ORION_CHEDDAR_GPU_PREFETCH": "0",
        "ORION_LATTIGO_BOOTSTRAP_MANY": "0",
    }
    applied: dict[str, str] = {}
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
        applied[name] = str(os.environ.get(name, ""))
    return applied


def _lattigo_compile_worker_default() -> str:
    cpu_count = max(1, int(os.cpu_count() or 1))
    return str(
        auto_worker_count(
            cpu_count,
            (
                "ORION_LT_COMPILE_WORKERS",
                "ORION_UNIFIED_COMPILE_WORKERS",
                "ORION_LATTIGO_COMPILE_WORKERS",
                "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS",
            ),
            default_workers=4,
            estimated_per_worker_bytes=24 * 1024**3,
            cpu_count=cpu_count,
        )
    )


def _lattigo_pack_worker_default() -> str:
    cpu_count = max(1, int(os.cpu_count() or 1))
    return str(
        auto_worker_count(
            cpu_count,
            ("ORION_PACK_CONV_WORKERS",),
            default_workers=8,
            estimated_per_worker_bytes=8 * 1024**3,
            cpu_count=cpu_count,
        )
    )


def _configure_lattigo_runtime_defaults() -> dict[str, str]:
    workers = _lattigo_compile_worker_default()
    pack_workers = _lattigo_pack_worker_default()
    defaults = {
        "ORION_LATTIGO_BOOTSTRAP_MANY": "0",
        "ORION_PACK_CONV_WORKERS": pack_workers,
        "ORION_LT_COMPILE_WORKERS": workers,
        "ORION_UNIFIED_COMPILE_WORKERS": workers,
        "ORION_UNIFIED_STREAM_COMPILE_BATCH_TRANSFORMS": workers,
        "ORION_UNIFIED_COMPILE_BATCH_TRANSFORMS": workers,
        "ORION_UNIFIED_LOAD_BATCH_TRANSFORMS": workers,
        "ORION_UNIFIED_CACHED_LOAD_BATCH_TRANSFORMS": workers,
        "ORION_LATTIGO_COMPILE_WORKERS": workers,
        "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS": workers,
        "ORION_LATTIGO_BOOTSTRAP_WORKERS": workers,
        "ORION_UNIFIED_STREAM_COMPILE_BATCH_GB": "2",
        "ORION_UNIFIED_LT_FORCE_COMPILE_TRIM_EACH_TRANSFORM": "1",
        "ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE": "1",
    }
    applied: dict[str, str] = {}
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
        applied[name] = str(os.environ.get(name, ""))
    applied["ORION_COMPILE_PARALLEL_POLICY"] = os.environ.get("ORION_COMPILE_PARALLEL_POLICY", "auto")
    applied["compile_parallel_policy_audit"] = json.dumps(policy_audit(), sort_keys=True)
    return applied


def _timed(payload: dict[str, Any], out_path: Path, step: str, fn: Callable[[], Any]) -> Any:
    payload["step"] = str(step)
    _write(payload, out_path)
    started = time.perf_counter()
    value = fn()
    payload.setdefault("timing_s", {})[str(step)] = float(time.perf_counter() - started)
    _write(payload, out_path)
    return value


def _align(reference: torch.Tensor, actual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left = reference.detach().cpu().to(dtype=torch.float32)
    right = actual.detach().cpu()
    if torch.is_complex(right):
        right = right.real
    right = right.to(dtype=torch.float32)
    if tuple(left.shape) != tuple(right.shape) and int(left.numel()) == int(right.numel()):
        right = right.reshape(tuple(left.shape))
    return left, right


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    left, right = _align(reference, actual)
    diff = right - left
    return {
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.pow(2))).item()),
    }


def _tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu()
    if torch.is_complex(value):
        value = value.real
    value = value.to(dtype=torch.float32)
    return {
        "shape": [int(v) for v in tuple(value.shape)],
        "checksum": float(value.sum().item()),
        "l2": float(torch.linalg.vector_norm(value).item()),
        "values": [float(v) for v in value.reshape(-1).tolist()],
    }


def _json_shape(value: Any) -> list[int]:
    try:
        return [int(v) for v in tuple(value)]
    except Exception:
        return []


def _cipher_ids_state(value: Any) -> dict[str, Any]:
    ids = [int(v) for v in list(getattr(value, "ids", []) or [])]
    backend = getattr(value, "backend", None)
    if backend is None and getattr(value, "scheme", None) is not None:
        backend = getattr(value.scheme, "backend", None)
    levels: list[int | None] = []
    scales: list[int | None] = []
    scale_log2: list[float | None] = []
    slots: list[int | None] = []
    for cid in ids:
        try:
            levels.append(int(backend.GetCiphertextLevel(int(cid)))) if backend is not None else levels.append(None)
        except Exception:
            levels.append(None)
        try:
            scales.append(int(backend.GetCiphertextScale(int(cid)))) if backend is not None else scales.append(None)
        except Exception:
            scales.append(None)
        try:
            scale_log2.append(float(backend.GetCiphertextScaleLog2(int(cid)))) if backend is not None else scale_log2.append(None)
        except Exception:
            scale_log2.append(None)
        try:
            slots.append(int(backend.GetCiphertextSlots(int(cid)))) if backend is not None else slots.append(None)
        except Exception:
            slots.append(None)
    return {
        "kind": type(value).__name__,
        "id_count": int(len(ids)),
        "shape": _json_shape(getattr(value, "shape", ())),
        "on_shape": _json_shape(getattr(value, "on_shape", ())),
        "levels": levels,
        "scales": scales,
        "scale_log2": scale_log2,
        "slots": slots,
    }


def _value_profile(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if hasattr(value, "ids") and hasattr(value, "shape"):
        return _cipher_ids_state(value)
    if isinstance(value, torch.Tensor):
        return {
            "kind": "Tensor",
            "shape": [int(v) for v in tuple(value.shape)],
            "dtype": str(value.dtype),
            "is_complex": bool(torch.is_complex(value)),
        }
    if isinstance(value, (int, float, bool, str)) or value is None:
        return {"kind": type(value).__name__, "value": value}
    if depth >= 2:
        return {"kind": type(value).__name__}
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "length": int(len(value)),
            "items": [_value_profile(v, depth=int(depth) + 1) for v in list(value)[:4]],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "length": int(len(value)),
            "items": {
                str(k): _value_profile(v, depth=int(depth) + 1)
                for k, v in list(value.items())[:4]
            },
        }
    return {"kind": type(value).__name__}


def _module_category(module: Module) -> str:
    cls_name = type(module).__name__
    if isinstance(module, Bootstrap):
        return "bootstrap"
    if "Pool" in cls_name:
        return "pool"
    if isinstance(module, LinearTransform):
        if cls_name == "ConvTranspose2d":
            return "conv_transpose2d"
        if cls_name == "Conv2d":
            return "conv2d"
        return "linear_transform"
    if cls_name in {"Add"}:
        return "add"
    if cls_name in {"Mult"}:
        return "multiply"
    if cls_name in {"Flatten"}:
        return "reshape"
    if cls_name in {
        "Activation",
        "Chebyshev",
        "ELU",
        "GELU",
        "Hardshrink",
        "Mish",
        "Quad",
        "ReLU",
        "SELU",
        "SiLU",
        "Sigmoid",
        "Softplus",
        "_Sign",
    }:
        return "activation"
    return "other"


def _install_he_module_profiler(
    net: torch.nn.Module,
    *,
    memory_trace_path: Path | None = None,
) -> tuple[Callable[[], dict[str, Any]], Callable[[], None]]:
    rows_by_id: dict[int, dict[str, Any]] = {}
    stacks: dict[int, list[float]] = {}
    handles: list[Any] = []
    trace_counter = 0

    def append_trace(event: dict[str, Any]) -> None:
        nonlocal trace_counter
        if memory_trace_path is None:
            return
        trace_counter += 1
        event = dict(event)
        event["event_index"] = int(trace_counter)
        event["elapsed_since_trace_start_s"] = float(time.perf_counter() - trace_start)
        event["device_memory"] = _device_memory_snapshot()
        event["live_ciphertexts"] = _live_ciphertext_snapshot()
        event["host_maxrss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        Path(memory_trace_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(memory_trace_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def should_profile(name: str, module: torch.nn.Module) -> bool:
        if not name or not isinstance(module, Module):
            return False
        return True

    def children_count(module: torch.nn.Module) -> int:
        return int(sum(1 for _child in module.children()))

    def has_direct_bootstrapper(module: torch.nn.Module) -> bool:
        return isinstance(getattr(module, "bootstrapper", None), Bootstrap)

    trace_start = time.perf_counter()
    if memory_trace_path is not None:
        Path(memory_trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(memory_trace_path).write_text("", encoding="utf-8")
        append_trace({"phase": "forward", "hook": "trace_start"})

    def make_pre(row: dict[str, Any]):
        @_dynamo_disable
        def pre_hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if not bool(getattr(module, "he_mode", False)):
                return
            key = int(id(module))
            stacks.setdefault(key, []).append(float(time.perf_counter()))
            append_trace(
                {
                    "phase": "forward",
                    "hook": "pre",
                    "module_path": str(row["module_path"]),
                    "class": str(row["class"]),
                    "category": str(row["category"]),
                    "is_leaf": bool(row["is_leaf"]),
                    "region_runtime": bool(row["region_runtime"]),
                    "region_strategy": str(row["region_strategy"]),
                }
            )

        return pre_hook

    def make_post(row: dict[str, Any]):
        @_dynamo_disable
        def post_hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not bool(getattr(module, "he_mode", False)):
                return
            key = int(id(module))
            stack = stacks.get(key) or []
            if not stack:
                return
            elapsed = float(time.perf_counter() - stack.pop())
            row["call_count"] = int(row.get("call_count", 0)) + 1
            row["elapsed_s"] = float(row.get("elapsed_s", 0.0)) + elapsed
            row["max_call_s"] = max(float(row.get("max_call_s", 0.0)), elapsed)
            row["last_call_s"] = elapsed
            append_trace(
                {
                    "phase": "forward",
                    "hook": "post",
                    "module_path": str(row["module_path"]),
                    "class": str(row["class"]),
                    "category": str(row["category"]),
                    "is_leaf": bool(row["is_leaf"]),
                    "region_runtime": bool(row["region_runtime"]),
                    "region_strategy": str(row["region_strategy"]),
                    "call_count": int(row.get("call_count", 0)),
                    "last_call_s": float(elapsed),
                }
            )

        return post_hook

    for name, module in net.named_modules():
        if not should_profile(str(name), module):
            continue
        child_count = children_count(module)
        is_bootstrapper_child = str(name).endswith(".bootstrapper")
        direct_bootstrapper = has_direct_bootstrapper(module)
        row = {
            "module_path": str(name),
            "class": type(module).__name__,
            "category": _module_category(module),
            "level": None if getattr(module, "level", None) is None else int(getattr(module, "level")),
            "depth": None if getattr(module, "depth", None) is None else int(getattr(module, "depth")),
            "children_count": int(child_count),
            "is_leaf": bool(child_count == 0),
            "is_bootstrapper_child": bool(is_bootstrapper_child),
            "has_bootstrapper_child": bool(direct_bootstrapper),
            "bootstrapper_path": f"{name}.bootstrapper" if direct_bootstrapper else "",
            "region_runtime": bool(getattr(module, "region_runtime", None) is not None),
            "region_strategy": str(getattr(getattr(module, "region_runtime", None), "strategy", "")),
            "call_count": 0,
            "elapsed_s": 0.0,
            "max_call_s": 0.0,
            "last_call_s": 0.0,
        }
        rows_by_id[int(id(module))] = row
        handles.append(module.register_forward_pre_hook(make_pre(row)))
        handles.append(module.register_forward_hook(make_post(row)))

    def snapshot() -> dict[str, Any]:
        rows = sorted(rows_by_id.values(), key=lambda item: str(item["module_path"]))
        active_rows = [row for row in rows if int(row.get("call_count", 0)) > 0]
        rows_by_path = {str(row["module_path"]): row for row in active_rows}

        def totals_for(selected_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
            totals: dict[str, dict[str, float | int]] = {}
            for row in selected_rows:
                cat = str(row["category"])
                entry = totals.setdefault(cat, {"elapsed_s": 0.0, "call_count": 0, "module_count": 0})
                entry["elapsed_s"] = float(entry["elapsed_s"]) + float(row.get("elapsed_s", 0.0))
                entry["call_count"] = int(entry["call_count"]) + int(row.get("call_count", 0))
                entry["module_count"] = int(entry["module_count"]) + 1
            return totals

        leaf_rows = [row for row in active_rows if bool(row.get("is_leaf", False))]
        primary_rows = [row for row in active_rows if not bool(row.get("is_bootstrapper_child", False))]
        adjusted_rows: list[dict[str, Any]] = []
        adjusted_totals_by_category: dict[str, dict[str, float | int]] = {}
        for row in primary_rows:
            bootstrap_row = rows_by_path.get(str(row.get("bootstrapper_path", "")))
            bootstrap_elapsed = float(bootstrap_row.get("elapsed_s", 0.0)) if bootstrap_row is not None else 0.0
            core_elapsed = max(0.0, float(row.get("elapsed_s", 0.0)) - bootstrap_elapsed)
            adjusted = {
                "module_path": str(row["module_path"]),
                "class": str(row["class"]),
                "category": str(row["category"]),
                "elapsed_s": float(row.get("elapsed_s", 0.0)),
                "bootstrap_child_s": float(bootstrap_elapsed),
                "core_excluding_bootstrap_s": float(core_elapsed),
                "call_count": int(row.get("call_count", 0)),
                "level": row.get("level"),
                "depth": row.get("depth"),
                "region_runtime": bool(row.get("region_runtime", False)),
                "region_strategy": str(row.get("region_strategy", "")),
            }
            adjusted_rows.append(adjusted)
            cat = str(row["category"])
            entry = adjusted_totals_by_category.setdefault(cat, {"elapsed_s": 0.0, "call_count": 0, "module_count": 0})
            entry["elapsed_s"] = float(entry["elapsed_s"]) + float(core_elapsed)
            entry["call_count"] = int(entry["call_count"]) + int(row.get("call_count", 0))
            entry["module_count"] = int(entry["module_count"]) + 1

        top_by_elapsed = sorted(active_rows, key=lambda item: float(item.get("elapsed_s", 0.0)), reverse=True)[:30]
        top_adjusted_by_core = sorted(
            adjusted_rows,
            key=lambda item: float(item.get("core_excluding_bootstrap_s", 0.0)),
            reverse=True,
        )[:30]
        top_bootstrap_by_parent = sorted(
            [row for row in adjusted_rows if float(row.get("bootstrap_child_s", 0.0)) > 0.0],
            key=lambda item: float(item.get("bootstrap_child_s", 0.0)),
            reverse=True,
        )[:30]
        return {
            "enabled": True,
            "profiled_module_count": int(len(rows)),
            "active_module_count": int(len(active_rows)),
            "notes": [
                "Hooks record wall-time only; ciphertext level/scale are intentionally not queried inside hooks.",
                "primary_inclusive totals include child bootstrap hooks for modules that own a bootstrapper.",
                "primary_adjusted subtracts the direct .bootstrapper child from its parent to estimate module core time.",
                "leaf_totals_by_category is additive but excludes non-leaf parents that own bootstrappers.",
            ],
            "totals_by_category": totals_for(leaf_rows),
            "leaf_totals_by_category": totals_for(leaf_rows),
            "primary_inclusive_totals_by_category": totals_for(primary_rows),
            "primary_adjusted_totals_by_category": adjusted_totals_by_category,
            "top_by_elapsed": [
                {
                    "module_path": str(row["module_path"]),
                    "class": str(row["class"]),
                    "category": str(row["category"]),
                    "elapsed_s": float(row.get("elapsed_s", 0.0)),
                    "children_count": int(row.get("children_count", 0)),
                    "is_leaf": bool(row.get("is_leaf", False)),
                    "has_bootstrapper_child": bool(row.get("has_bootstrapper_child", False)),
                    "bootstrapper_path": str(row.get("bootstrapper_path", "")),
                    "call_count": int(row.get("call_count", 0)),
                    "max_call_s": float(row.get("max_call_s", 0.0)),
                    "level": row.get("level"),
                    "depth": row.get("depth"),
                    "region_runtime": bool(row.get("region_runtime", False)),
                    "region_strategy": str(row.get("region_strategy", "")),
                }
                for row in top_by_elapsed
            ],
            "top_adjusted_by_core": top_adjusted_by_core,
            "top_bootstrap_by_parent": top_bootstrap_by_parent,
            "rows": rows,
            "primary_adjusted_rows": sorted(adjusted_rows, key=lambda item: str(item["module_path"])),
        }

    def remove() -> None:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    return snapshot, remove


def _collect_region_audit(net: torch.nn.Module) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, module in net.named_modules():
        runtime = getattr(module, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        if runtime is None:
            continue
        row = {
            "node": str(getattr(module, "region_output_id", name)),
            "module_path": str(name),
            "stage": str(getattr(runtime, "stage", "")),
            "executable": bool(getattr(runtime, "executable", False)),
            "compile_count": int(getattr(executor, "compile_count", 0)) if executor is not None else 0,
            "execute_count": int(getattr(runtime, "execute_count", 0)),
            "strategy": str(getattr(runtime, "strategy", "")),
            "fallback_reason": str(getattr(runtime, "fallback_reason", "")),
        }
        if executor is not None:
            row["last_runtime_timing"] = dict(getattr(executor, "last_runtime_timing", {}) or {})
            row["last_runtime_counts"] = dict(getattr(executor, "last_runtime_counts", {}) or {})
            row["last_runtime_io"] = dict(getattr(executor, "last_runtime_io", {}) or {})
        rows.append(row)
    return {
        "selected_region_count": int(len(rows)),
        "executable_region_count": int(sum(1 for row in rows if bool(row["executable"]))),
        "rows": rows,
    }


def _backend_u64_array(callable_obj: Callable[..., Any], *args: Any) -> list[int]:
    values = callable_obj(*args)
    if isinstance(values, int):
        return [int(values)]
    try:
        return [int(value) for value in list(values)]
    except TypeError:
        return []


def _live_ciphertext_snapshot() -> dict[str, Any]:
    backend = getattr(scheme, "backend", None)
    get_live = getattr(backend, "GetLiveCiphertexts", None)
    if not callable(get_live):
        return {"available": False}
    raw_values = get_live()
    if isinstance(raw_values, int):
        return {
            "available": True,
            "count": int(raw_values),
            "ids_sample": [],
            "max_id": None,
        }
    try:
        values = [int(value) for value in list(raw_values)]
    except TypeError:
        return {
            "available": False,
            "error": f"non-iterable GetLiveCiphertexts result: {type(raw_values).__name__}",
            "ids_sample": [],
            "max_id": None,
        }
    return {
        "available": True,
        "count": int(len(values)),
        "ids_sample": [int(value) for value in values[:8]],
        "ids_tail": [int(value) for value in values[-8:]],
        "max_id": int(max(values)) if values else None,
    }


def _device_memory_snapshot() -> dict[str, Any]:
    backend = getattr(scheme, "backend", None)
    get_info = getattr(backend, "GetDeviceMemoryInfo", None)
    if not callable(get_info):
        return {"available": False}
    values = _backend_u64_array(get_info)
    free_bytes = int(values[0]) if len(values) >= 1 else 0
    total_bytes = int(values[1]) if len(values) >= 2 else 0
    return {
        "available": True,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "used_bytes": int(max(0, total_bytes - free_bytes)) if total_bytes else 0,
    }


def _linear_transform_device_estimate(transform_id: int) -> dict[str, Any]:
    backend = getattr(scheme, "backend", None)
    estimate = getattr(backend, "EstimateLinearTransformDeviceBytes", None)
    uses_streaming = getattr(backend, "LinearTransformUsesStreaming", None)
    values = _backend_u64_array(estimate, int(transform_id)) if callable(estimate) else []
    return {
        "transform_id": int(transform_id),
        "estimate_device_bytes": int(values[0]) if values else None,
        "uses_streaming": (
            bool(int(uses_streaming(int(transform_id)))) if callable(uses_streaming) else None
        ),
    }


def _dense_cols(module: torch.nn.Module) -> int:
    keys = list(getattr(module, "transform_ids", {}).keys())
    if not keys:
        return 0
    return max(int(col) for _row, col in keys) + 1


def _linear_transform_rotation_stats(module: torch.nn.Module) -> dict[str, Any]:
    transform_ids = dict(getattr(module, "transform_ids", {}) or {})
    per_transform: list[dict[str, Any]] = []
    unique_keys: set[int] = set()
    transform_rotation_total = 0
    device_estimates: list[dict[str, Any]] = []
    for (row, col), transform_id in sorted(transform_ids.items()):
        keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
        nonzero_keys = sorted(int(key) for key in keys if int(key) != 0)
        unique_keys.update(nonzero_keys)
        transform_rotation_total += int(len(nonzero_keys))
        estimate = _linear_transform_device_estimate(int(transform_id))
        device_estimates.append(estimate)
        per_transform.append(
            {
                "row": int(row),
                "col": int(col),
                "transform_id": int(transform_id),
                "rotation_key_count": int(len(nonzero_keys)),
                "estimate_device_bytes": estimate.get("estimate_device_bytes"),
                "uses_streaming": estimate.get("uses_streaming"),
            }
        )
    cols = _dense_cols(module)
    rows = int(len(transform_ids) // max(1, int(cols)))
    output_rotations = int(getattr(module, "output_rotations", 0))
    output_rotation_evals = int(rows * output_rotations)
    estimate_values = [
        int(item["estimate_device_bytes"])
        for item in device_estimates
        if item.get("estimate_device_bytes") is not None
    ]
    return {
        "source": "compiled_backend_transform_rotation_keys",
        "transform_count": int(len(transform_ids)),
        "rows": int(rows),
        "cols": int(cols),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": int(output_rotations),
        "output_rotation_eval_count": int(output_rotation_evals),
        "rotation_eval_count_estimate": int(transform_rotation_total + output_rotation_evals),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "estimate_device_bytes_total": int(sum(estimate_values)) if estimate_values else None,
        "estimate_device_bytes_max": int(max(estimate_values)) if estimate_values else None,
        "streaming_transform_count": int(
            sum(1 for item in device_estimates if item.get("uses_streaming") is True)
        ),
        "per_transform": per_transform,
    }


def _unified_group_rotation_stats(groups: list[Any]) -> dict[str, Any]:
    per_group: list[dict[str, Any]] = []
    unique_keys: set[int] = set()
    transform_rotation_total = 0
    shared_rotation_total = 0
    transform_count = 0
    estimate_values: list[int] = []
    streaming_transform_count = 0
    for group_index, group in enumerate(groups):
        ids = [int(value) for value in (getattr(group, "unified_ids", None) or [])]
        group_keys: set[int] = set()
        per_transform: list[dict[str, Any]] = []
        for transform_index, transform_id in enumerate(ids):
            keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
            nonzero_keys = sorted(int(key) for key in keys if int(key) != 0)
            estimate = _linear_transform_device_estimate(int(transform_id))
            if estimate.get("estimate_device_bytes") is not None:
                estimate_values.append(int(estimate["estimate_device_bytes"]))
            if estimate.get("uses_streaming") is True:
                streaming_transform_count += 1
            group_keys.update(nonzero_keys)
            unique_keys.update(nonzero_keys)
            transform_rotation_total += int(len(nonzero_keys))
            transform_count += 1
            per_transform.append(
                {
                    "transform_index": int(transform_index),
                    "transform_id": int(transform_id),
                    "rotation_key_count": int(len(nonzero_keys)),
                    "estimate_device_bytes": estimate.get("estimate_device_bytes"),
                    "uses_streaming": estimate.get("uses_streaming"),
                }
            )
        shared_rotation_total += int(len(group_keys))
        per_group.append(
            {
                "group_index": int(group_index),
                "transform_count": int(len(ids)),
                "rotation_key_count_total": int(sum(int(item["rotation_key_count"]) for item in per_transform)),
                "shared_rotation_eval_count": int(len(group_keys)),
                "unique_rotation_key_count": int(len(group_keys)),
                "per_transform": per_transform,
            }
        )
    return {
        "source": "compiled_backend_unified_transform_rotation_keys",
        "group_count": int(len(groups)),
        "transform_count": int(transform_count),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "shared_rotation_eval_count_total": int(shared_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": 0,
        "output_rotation_eval_count": 0,
        "rotation_eval_count_estimate": int(shared_rotation_total),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "estimate_device_bytes_total": int(sum(estimate_values)) if estimate_values else None,
        "estimate_device_bytes_max": int(max(estimate_values)) if estimate_values else None,
        "streaming_transform_count": int(streaming_transform_count),
        "per_group": per_group,
    }


def _provider_rotation_stats(executor: Any) -> dict[str, Any]:
    if getattr(executor, "group", None) is not None:
        return _unified_group_rotation_stats([executor.group])
    groups = list(getattr(executor, "groups", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups = list(getattr(executor, "groups_by_input_block", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups = list(getattr(executor, "groups_by_pair", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups = list(getattr(executor, "groups_by_input", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups = list(getattr(executor, "groups_by_input_chunk", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups_by_input_index = getattr(executor, "groups_by_input_index", None) or {}
    if groups_by_input_index:
        return _unified_group_rotation_stats([group for _input_index, group in sorted(groups_by_input_index.items())])
    transform_ids = dict(getattr(executor, "transform_ids", {}) or {})
    if not transform_ids:
        return {
            "source": "no_backend_unified_transform_ids",
            "group_count": 0,
            "transform_count": 0,
            "transform_rotation_key_count_total": 0,
            "unique_rotation_key_count": 0,
            "output_rotations_per_output_ct": 0,
            "output_rotation_eval_count": 0,
            "rotation_eval_count_estimate": 0,
            "unique_rotation_keys": [],
        }

    unique_keys: set[int] = set()
    per_transform: list[dict[str, Any]] = []
    transform_rotation_total = 0
    cols = max(int(col) for _row, col in transform_ids) + 1
    rows = max(int(row) for row, _col in transform_ids) + 1
    estimate_values: list[int] = []
    streaming_transform_count = 0
    for (row, col), transform_id in sorted(transform_ids.items()):
        keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
        nonzero_keys = sorted(int(key) for key in keys if int(key) != 0)
        estimate = _linear_transform_device_estimate(int(transform_id))
        if estimate.get("estimate_device_bytes") is not None:
            estimate_values.append(int(estimate["estimate_device_bytes"]))
        if estimate.get("uses_streaming") is True:
            streaming_transform_count += 1
        unique_keys.update(nonzero_keys)
        transform_rotation_total += int(len(nonzero_keys))
        per_transform.append(
            {
                "row": int(row),
                "col": int(col),
                "transform_id": int(transform_id),
                "rotation_key_count": int(len(nonzero_keys)),
                "estimate_device_bytes": estimate.get("estimate_device_bytes"),
                "uses_streaming": estimate.get("uses_streaming"),
            }
        )
    output_rotations = int(getattr(executor, "output_rotations", 0))
    output_rotation_evals = int(rows * output_rotations)
    return {
        "source": "compiled_backend_executor_transform_rotation_keys",
        "transform_count": int(len(transform_ids)),
        "rows": int(rows),
        "cols": int(cols),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": int(output_rotations),
        "output_rotation_eval_count": int(output_rotation_evals),
        "rotation_eval_count_estimate": int(transform_rotation_total + output_rotation_evals),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "estimate_device_bytes_total": int(sum(estimate_values)) if estimate_values else None,
        "estimate_device_bytes_max": int(max(estimate_values)) if estimate_values else None,
        "streaming_transform_count": int(streaming_transform_count),
        "per_transform": per_transform,
    }


def _collect_rotation_report(net: torch.nn.Module, *, mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_runtime_ids: set[int] = set()
    for name, module in net.named_modules():
        if not isinstance(module, LinearTransform):
            continue
        runtime = getattr(module, "region_runtime", None)
        runtime_supported = bool(runtime is not None and getattr(runtime, "supports_scheme", lambda _scheme: True)(scheme))
        if (
            runtime is not None
            and bool(getattr(runtime, "executable", False))
            and bool(runtime_supported)
            and getattr(runtime, "executor", None) is not None
        ):
            runtime_id = id(runtime)
            if runtime_id not in seen_runtime_ids:
                seen_runtime_ids.add(runtime_id)
                rows.append(
                    {
                        "node": str(getattr(module, "region_output_id", name)),
                        "module_path": str(name),
                        "kind": "provider_region",
                        "stage": str(getattr(runtime, "stage", "")),
                        "nodes": list(getattr(runtime, "conv_nodes", (str(name),))),
                        "stats": _provider_rotation_stats(runtime.executor),
                    }
                )
            continue
        if getattr(module, "transform_ids", None):
            rows.append(
                {
                    "node": str(name),
                    "module_path": str(name),
                    "kind": type(module).__name__,
                    "stage": "",
                    "nodes": [str(name)],
                    "stats": _linear_transform_rotation_stats(module),
                }
            )

    total_rotation_estimate = int(
        sum(int(row.get("stats", {}).get("rotation_eval_count_estimate", 0)) for row in rows)
    )
    total_transform_rotation_keys = int(
        sum(int(row.get("stats", {}).get("transform_rotation_key_count_total", 0)) for row in rows)
    )
    total_shared_rotation_evals = int(
        sum(int(row.get("stats", {}).get("shared_rotation_eval_count_total", 0)) for row in rows)
    )
    total_output_rotation_evals = int(
        sum(int(row.get("stats", {}).get("output_rotation_eval_count", 0)) for row in rows)
    )
    return {
        "mode": str(mode),
        "source": "compiled_backend_rotation_keys_plus_output_rotation_estimate",
        "row_count": int(len(rows)),
        "total_rotation_eval_count_estimate": int(total_rotation_estimate),
        "total_transform_rotation_key_count": int(total_transform_rotation_keys),
        "total_shared_rotation_eval_count": int(total_shared_rotation_evals),
        "total_output_rotation_eval_count": int(total_output_rotation_evals),
        "rows": rows,
    }


def _collect_bootstrap_report(net: torch.nn.Module) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, module in net.named_modules():
        if isinstance(module, Bootstrap):
            rows.append(
                {
                    "name": str(name),
                    "input_level": int(getattr(module, "input_level", -1)),
                    "bootstrap_slots": int(getattr(module, "bootstrap_slots", 0) or 0),
                    "prescale": float(getattr(module, "prescale", 0.0)),
                    "postscale": float(getattr(module, "postscale", 0.0)),
                }
            )
    by_slots: dict[str, int] = {}
    for row in rows:
        slots = str(int(row["bootstrap_slots"]))
        by_slots[slots] = int(by_slots.get(slots, 0)) + 1
    return {"count": int(len(rows)), "by_slots": by_slots, "rows": rows}


def _name_bootstraps(net: torch.nn.Module) -> None:
    for name, module in net.named_modules():
        if isinstance(module, Bootstrap):
            module.bootstrap_debug_name = str(name)


def _attempt_timed(
    payload: dict[str, Any],
    out_path: Path,
    attempt: dict[str, Any],
    attempt_index: int,
    step: str,
    fn: Callable[[], Any],
    *,
    record_primary_timing: bool,
) -> Any:
    payload["phase"] = "forward"
    payload["active_forward_attempt"] = int(attempt_index)
    payload["step"] = f"forward_{int(attempt_index)}_{step}"
    attempt["step"] = str(step)
    _write(payload, out_path)
    started = time.perf_counter()
    value = fn()
    elapsed = float(time.perf_counter() - started)
    attempt.setdefault("timing_s", {})[str(step)] = elapsed
    payload.setdefault("timing_s", {})[f"forward_{int(attempt_index)}_{step}"] = elapsed
    if bool(record_primary_timing):
        payload.setdefault("timing_s", {})[str(step)] = elapsed
    _write(payload, out_path)
    return value


def _run_forward_attempt(
    *,
    payload: dict[str, Any],
    out_path: Path,
    net: torch.nn.Module,
    x0: torch.Tensor,
    clear: torch.Tensor,
    input_level: int,
    mode: str,
    attempt_index: int,
    attempt_kind: str,
    profile_modules: bool,
    trace_forward_memory: bool,
    record_primary: bool,
) -> dict[str, Any]:
    memory_trace_path = (
        out_path.with_name(f"{out_path.stem}.forward{int(attempt_index)}.memory_trace.jsonl")
        if bool(trace_forward_memory)
        else None
    )
    attempt: dict[str, Any] = {
        "attempt_index": int(attempt_index),
        "kind": str(attempt_kind),
        "status": "started",
        "step": "init",
        "device_memory_before_encrypt": _device_memory_snapshot(),
        "live_ciphertexts_before_encrypt": _live_ciphertext_snapshot(),
    }
    if memory_trace_path is not None:
        attempt["memory_trace_path"] = str(memory_trace_path)
    payload.setdefault("forward_attempts", []).append(attempt)
    _write(payload, out_path)

    profile_snapshot = None
    remove_profile = None
    if bool(profile_modules) or memory_trace_path is not None:
        profile_snapshot, remove_profile = _install_he_module_profiler(
            net,
            memory_trace_path=memory_trace_path,
        )
    x0_ct = None
    out_ct = None
    try:
        x0_ct = _attempt_timed(
            payload,
            out_path,
            attempt,
            int(attempt_index),
            "encrypt",
            lambda: _encrypt_model_input(x0, int(input_level), net=net, payload=payload),
            record_primary_timing=bool(record_primary),
        )
        attempt["model_input_encoding"] = dict(payload.get("model_input_encoding", {}) or {})
        attempt["input_ciphertext_count"] = int(len(getattr(x0_ct, "ids", ()) or ()))
        attempt["device_memory_after_encrypt"] = _device_memory_snapshot()
        attempt["live_ciphertexts_after_encrypt"] = _live_ciphertext_snapshot()
        _write(payload, out_path)
        try:
            out_ct = _attempt_timed(
                payload,
                out_path,
                attempt,
                int(attempt_index),
                "he_forward",
                lambda: net(x0_ct),
                record_primary_timing=bool(record_primary),
            )
        finally:
            if profile_snapshot is not None:
                snapshot = profile_snapshot()
                if bool(profile_modules):
                    attempt["module_profile_after_forward"] = snapshot
                    if bool(record_primary):
                        payload["module_profile_after_forward"] = attempt["module_profile_after_forward"]
                _write(payload, out_path)
            if remove_profile is not None:
                remove_profile()
        attempt["device_memory_after_he_forward"] = _device_memory_snapshot()
        attempt["live_ciphertexts_after_he_forward"] = _live_ciphertext_snapshot()
        runtime_fairness = _collect_runtime_fairness(
            net,
            serving_hot_s=float(attempt.get("timing_s", {}).get("he_forward", 0.0)),
        )
        attempt["runtime_fairness_timing"] = dict(runtime_fairness)
        attempt["resident_compute_s"] = runtime_fairness.get("resident_compute_s")
        attempt["serving_hot_s"] = runtime_fairness.get("serving_hot_s")
        attempt["artifact_read_s"] = runtime_fairness.get("artifact_read_s")
        attempt["artifact_load_s"] = runtime_fairness.get("artifact_load_s")
        attempt["artifact_unload_s"] = runtime_fairness.get("artifact_unload_s")
        attempt["trim_s"] = runtime_fairness.get("trim_s")
        attempt["runtime_fairness_mode"] = str(runtime_fairness.get("runtime_fairness_mode", "unknown"))
        if bool(record_primary):
            payload["runtime_fairness_timing_after_forward"] = dict(runtime_fairness)
            payload["resident_compute_s"] = runtime_fairness.get("resident_compute_s")
            payload["serving_hot_s"] = runtime_fairness.get("serving_hot_s")
            payload["artifact_read_s"] = runtime_fairness.get("artifact_read_s")
            payload["artifact_load_s"] = runtime_fairness.get("artifact_load_s")
            payload["artifact_unload_s"] = runtime_fairness.get("artifact_unload_s")
            payload["trim_s"] = runtime_fairness.get("trim_s")
            payload["runtime_fairness_mode"] = str(runtime_fairness.get("runtime_fairness_mode", "unknown"))
        _write(payload, out_path)
        decoded = _attempt_timed(
            payload,
            out_path,
            attempt,
            int(attempt_index),
            "decrypt_decode",
            lambda: out_ct.decrypt().decode(),
            record_primary_timing=bool(record_primary),
        )
        decoded = decoded.detach().cpu()
        if torch.is_complex(decoded):
            decoded = decoded.real
        decoded = decoded.to(dtype=torch.float32)
        attempt["input_ciphertext_count"] = int(len(x0_ct.ids))
        attempt["output_ciphertext_count"] = int(len(out_ct.ids))
        attempt["decoded"] = _tensor_payload(decoded)
        attempt["mae_vs_clear"] = _metrics(clear, decoded)
        attempt["region_audit_after_forward"] = _collect_region_audit(net)
        attempt["bootstrap_report_after_forward"] = _collect_bootstrap_report(net)
        attempt["rotation_report_after_forward"] = _collect_rotation_report(net, mode=mode)
        attempt["device_memory_after_decrypt_decode"] = _device_memory_snapshot()
        attempt["live_ciphertexts_after_decrypt_decode"] = _live_ciphertext_snapshot()
        attempt["status"] = "ok"
        attempt["step"] = "done"
        if bool(record_primary):
            payload["input_ciphertext_count"] = int(attempt["input_ciphertext_count"])
            payload["output_ciphertext_count"] = int(attempt["output_ciphertext_count"])
            payload["decoded"] = attempt["decoded"]
            payload["mae_vs_clear"] = attempt["mae_vs_clear"]
            payload["region_audit_after_forward"] = attempt["region_audit_after_forward"]
            payload["bootstrap_report_after_forward"] = attempt["bootstrap_report_after_forward"]
            payload["rotation_report_after_forward"] = attempt["rotation_report_after_forward"]
            payload["device_memory_after_he_forward"] = attempt["device_memory_after_he_forward"]
            payload["device_memory_after_decrypt_decode"] = attempt["device_memory_after_decrypt_decode"]
            payload["live_ciphertexts_after_he_forward"] = attempt["live_ciphertexts_after_he_forward"]
            payload["live_ciphertexts_after_decrypt_decode"] = attempt["live_ciphertexts_after_decrypt_decode"]
        _write(payload, out_path)
        return attempt
    except BaseException as exc:
        attempt["status"] = "failed"
        attempt["error_type"] = type(exc).__name__
        attempt["error"] = str(exc)
        attempt["traceback"] = traceback.format_exc(limit=120)
        _write(payload, out_path)
        raise
    finally:
        if remove_profile is not None:
            try:
                remove_profile()
            except Exception:
                pass
        for tensor in (out_ct, x0_ct):
            release = getattr(tensor, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass
        backend = getattr(scheme, "backend", None)
        if backend is not None:
            trim_runtime_memory(backend, reason=f"after_forward_attempt:{int(attempt_index)}")
        gc.collect()


def _run_one(
    *,
    network: str,
    backend: str,
    mode: str,
    out_path: Path,
    seed: int,
    compile_only: bool = False,
    forward_runs: int = 1,
    warmup_runs: int = 0,
    profile_modules: bool = False,
    trace_forward_memory: bool = False,
    provider_mode_override: str | None = None,
    provider_no_hybrid: bool = False,
    io_mode: str = "none",
    io_dir: Path | None = None,
    diags_path: Path | None = None,
    keys_path: Path | None = None,
    logn_override: int | None = None,
    activation: str | None = None,
    silu_degree: int = 31,
    ckks_preset: str | None = None,
) -> dict[str, Any]:
    env_defaults = _configure_cheddar_runtime_defaults() if str(backend) == "cheddar" else {}
    lattigo_env_defaults = _configure_lattigo_runtime_defaults() if str(backend) == "lattigo" else {}
    if str(backend) != "cheddar":
        os.environ.setdefault("ORION_LATTIGO_BOOTSTRAP_MANY", "0")
    if str(backend) == "cheddar" and str(io_mode).lower() in {"save", "load"}:
        os.environ.setdefault("ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE", "1")
        env_defaults["ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE"] = str(
            os.environ.get("ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE", "")
        )
    spec = NETWORKS[str(network)]
    provider_mode = str(provider_mode_override or spec["provider_mode"]) if str(mode) == "provider" else ""
    if str(mode) == "provider" and bool(provider_no_hybrid):
        provider_mode = (
            str(provider_mode)
            if "nohybrid" in str(provider_mode).lower()
            else f"{provider_mode}_nohybrid"
        )
    base_config = _apply_ckks_preset(
        spec["config"](provider_mode, backend=str(backend)),
        ckks_preset,
    )
    config = _apply_io_config(
        base_config,
        backend=str(backend),
        io_mode=str(io_mode),
        io_dir=io_dir,
        diags_path=diags_path,
        keys_path=keys_path,
        logn_override=logn_override,
    )
    payload: dict[str, Any] = {
        "status": "started",
        "step": "init",
        "network": str(network),
        "backend": str(backend),
        "label": str(spec["label"]),
        "model": str(spec["model"]),
        "dataset": str(spec["dataset"]),
        "network_scope": str(spec.get("scope", "full")),
        "mode": str(mode),
        "provider_mode": str(provider_mode),
        "provider_no_hybrid": bool(provider_no_hybrid),
        "io_mode": str(io_mode),
        "io_dir": None if io_dir is None else str(Path(io_dir)),
        "diags_path": str(config.get("orion", {}).get("diags_path", "")),
        "keys_path": str(config.get("orion", {}).get("keys_path", "")),
        "logn_override": None if logn_override is None else int(logn_override),
        "ckks_preset": str(ckks_preset or "network-default"),
        "activation": {
            "kind": str(activation) if activation is not None else None,
            "silu_degree": int(silu_degree) if str(activation or "").lower() == "silu" else None,
        },
        "seed": int(seed),
        "input_shape": [int(v) for v in tuple(spec["input_shape"])],
        "compile_only": bool(compile_only),
        "forward_runs": int(forward_runs),
        "warmup_runs": int(warmup_runs),
        "profile_modules": bool(profile_modules),
        "trace_forward_memory": bool(trace_forward_memory),
        "bootstrap_many_enabled": os.environ.get("ORION_LATTIGO_BOOTSTRAP_MANY", "0") != "0",
        "cheddar_runtime_env": env_defaults,
        "lattigo_runtime_env": lattigo_env_defaults,
        "config": config,
    }
    _write(payload, out_path)
    try:
        if bool(compile_only) and str(io_mode) == "save" and str(backend) == "cheddar":
            os.environ.setdefault("ORION_UNIFIED_LT_RELEASE_INDEX_ONLY_RAW_MATRICES_AFTER_SAVE", "1")
            os.environ.setdefault("ORION_COMPILE_SKIP_BOOTSTRAPPER_GENERATION", "1")
            os.environ.setdefault("ORION_UNIFIED_LT_PREPARE_SHARED_CACHE_PLAN", "0")
        payload["phase"] = "compile_load"
        _write(payload, out_path)
        torch.manual_seed(int(seed))
        net = spec["builder"](activation=activation, silu_degree=int(silu_degree))
        net.eval()
        x0 = torch.randn(tuple(int(v) for v in spec["input_shape"]), dtype=torch.float32)
        with torch.no_grad():
            clear = _timed(payload, out_path, "clear_forward", lambda: net(x0))
        payload["clear"] = _tensor_payload(clear)
        _write(payload, out_path)

        _timed(payload, out_path, "init_scheme", lambda: scheme.init_scheme(config))
        payload["device_memory_after_init_scheme"] = _device_memory_snapshot()
        _write(payload, out_path)
        _timed(payload, out_path, "fit", lambda: scheme.fit(net, x0))
        payload["device_memory_after_fit"] = _device_memory_snapshot()
        _write(payload, out_path)
        input_level = _timed(payload, out_path, "compile", lambda: scheme.compile(net))
        _name_bootstraps(net)
        payload["input_level"] = int(input_level)
        payload["attach_audit"] = getattr(scheme, "region_first_attach_audit", {})
        payload["region_audit_after_compile"] = _collect_region_audit(net)
        payload["bootstrap_report_after_compile"] = _collect_bootstrap_report(net)
        payload["rotation_report_after_compile"] = _collect_rotation_report(net, mode=mode)
        get_compile_load_profile = getattr(getattr(scheme, "lt_evaluator", None), "get_compile_load_profile", None)
        if callable(get_compile_load_profile):
            payload["compile_load_profile_after_compile"] = get_compile_load_profile()
        payload["device_memory_after_compile"] = _device_memory_snapshot()
        payload["compile_load_done"] = True
        payload["phase"] = "compile_load_done"
        _write(payload, out_path)

        if bool(compile_only):
            payload["status"] = "ok_compile_only"
            payload["step"] = "done_compile_only"
            _write(payload, out_path)
            return payload

        net.he()
        payload["forward_attempts"] = []
        _write(payload, out_path)
        warmups = max(0, int(warmup_runs))
        runs = max(1, int(forward_runs))
        total_attempts = warmups + runs
        first_measured_index = warmups
        for attempt_index in range(total_attempts):
            attempt_kind = "warmup" if int(attempt_index) < warmups else "measured"
            _run_forward_attempt(
                payload=payload,
                out_path=out_path,
                net=net,
                x0=x0,
                clear=clear,
                input_level=int(input_level),
                mode=str(mode),
                attempt_index=int(attempt_index),
                attempt_kind=str(attempt_kind),
                profile_modules=bool(profile_modules),
                trace_forward_memory=bool(trace_forward_memory),
                record_primary=bool(int(attempt_index) == int(first_measured_index)),
            )
        ok_attempts = [
            attempt
            for attempt in payload.get("forward_attempts", [])
            if str(attempt.get("status")) == "ok"
        ]
        payload["forward_ok_count"] = int(len(ok_attempts))
        measured_attempts = [
            attempt for attempt in ok_attempts if str(attempt.get("kind")) == "measured"
        ]
        warmup_attempts = [
            attempt for attempt in ok_attempts if str(attempt.get("kind")) == "warmup"
        ]
        payload["warmup_ok_count"] = int(len(warmup_attempts))
        payload["measured_forward_ok_count"] = int(len(measured_attempts))
        if measured_attempts:
            payload["forward_mean_timing_s"] = {
                key: float(
                    sum(float(attempt.get("timing_s", {}).get(key, 0.0)) for attempt in measured_attempts)
                    / max(1, len(measured_attempts))
                )
                for key in ("encrypt", "he_forward", "decrypt_decode")
            }
            payload["measured_forward_mean_timing_s"] = dict(payload["forward_mean_timing_s"])
            runtime_fairness = _mean_runtime_fairness(measured_attempts)
            payload["measured_runtime_fairness_timing"] = dict(runtime_fairness)
            payload["resident_compute_s"] = runtime_fairness.get("resident_compute_s")
            payload["serving_hot_s"] = runtime_fairness.get("serving_hot_s")
            payload["artifact_read_s"] = runtime_fairness.get("artifact_read_s")
            payload["artifact_load_s"] = runtime_fairness.get("artifact_load_s")
            payload["artifact_unload_s"] = runtime_fairness.get("artifact_unload_s")
            payload["trim_s"] = runtime_fairness.get("trim_s")
            payload["runtime_fairness_mode"] = str(runtime_fairness.get("runtime_fairness_mode", "unknown"))
        payload["status"] = "ok"
        payload["step"] = "done"
        payload["phase"] = "done"
        _write(payload, out_path)
        return payload
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc(limit=120)
        _write(payload, out_path)
        raise
    finally:
        try:
            scheme.delete_scheme()
        except Exception:
            pass


def _artifact_runtime(payload: dict[str, Any]) -> float | None:
    timing = (
        payload.get("measured_forward_mean_timing_s")
        or payload.get("forward_mean_timing_s")
        or payload.get("timing_s", {})
    )
    values = [
        timing.get("encrypt"),
        timing.get("he_forward"),
        timing.get("decrypt_decode"),
    ]
    if any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values))


def _mean_runtime_fairness(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempts:
        return _aggregate_runtime_fairness([], serving_hot_s=0.0)
    timings = [dict(attempt.get("runtime_fairness_timing", {}) or {}) for attempt in attempts]
    result: dict[str, Any] = {}
    for key in _RUNTIME_FAIRNESS_NUMERIC_KEYS:
        values = [timing.get(key) for timing in timings if timing.get(key) is not None]
        result[key] = (
            float(sum(float(value) for value in values) / max(1, len(values)))
            if values
            else None
        )
    modes = [str(timing.get("runtime_fairness_mode", "unknown") or "unknown") for timing in timings]
    if any(mode == "streaming_eval_encode" for mode in modes):
        mode = "streaming_eval_encode"
    elif any(mode == "memory_bounded_load_eval" for mode in modes):
        mode = "memory_bounded_load_eval"
    elif modes and all(mode == "resident_compute" for mode in modes):
        mode = "resident_compute"
    else:
        mode = "unknown"
    result["runtime_fairness_mode"] = str(mode)
    result["source_count"] = int(sum(int(timing.get("source_count", 0) or 0) for timing in timings))
    return result


def _he_forward_runtime(payload: dict[str, Any]) -> float:
    timing = (
        payload.get("measured_forward_mean_timing_s")
        or payload.get("forward_mean_timing_s")
        or payload.get("timing_s", {})
    )
    return float(timing.get("he_forward", math.nan))


def _runtime_fairness_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        timing = (
            payload.get("measured_runtime_fairness_timing")
            or payload.get("runtime_fairness_timing_after_forward")
            or payload.get("runtime_fairness_timing")
            or {}
        )
        if isinstance(timing, dict):
            value = timing.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _runtime_fairness_mode(payload: dict[str, Any]) -> str:
    mode = payload.get("runtime_fairness_mode")
    if mode is None:
        timing = (
            payload.get("measured_runtime_fairness_timing")
            or payload.get("runtime_fairness_timing_after_forward")
            or payload.get("runtime_fairness_timing")
            or {}
        )
        if isinstance(timing, dict):
            mode = timing.get("runtime_fairness_mode")
    return str(mode or "unknown")


def _ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(float(numerator) / float(denominator))


def _rotation_total(payload: dict[str, Any]) -> int | None:
    report = payload.get("rotation_report_after_forward") or payload.get("rotation_report_after_compile") or {}
    value = report.get("total_rotation_eval_count_estimate")
    return None if value is None else int(value)


def _bootstrap_count(payload: dict[str, Any]) -> int | None:
    report = payload.get("bootstrap_report_after_forward") or payload.get("bootstrap_report_after_compile") or {}
    value = report.get("count")
    return None if value is None else int(value)


def _profile_category_seconds(payload: dict[str, Any]) -> dict[str, float]:
    profile = payload.get("module_profile_after_forward") or {}
    totals = profile.get("leaf_totals_by_category") or profile.get("totals_by_category") or {}
    return {
        str(category): float(values.get("elapsed_s", 0.0))
        for category, values in totals.items()
        if isinstance(values, dict)
    }


def _summarize(*, dense_path: Path, provider_path: Path, out_path: Path) -> dict[str, Any]:
    dense = json.loads(Path(dense_path).read_text(encoding="utf-8"))
    provider = json.loads(Path(provider_path).read_text(encoding="utf-8"))
    dense_runtime_fairness = {
        "resident_compute_s": _runtime_fairness_value(dense, "resident_compute_s"),
        "serving_hot_s": _runtime_fairness_value(dense, "serving_hot_s"),
        "artifact_read_s": _runtime_fairness_value(dense, "artifact_read_s"),
        "artifact_load_s": _runtime_fairness_value(dense, "artifact_load_s"),
        "artifact_unload_s": _runtime_fairness_value(dense, "artifact_unload_s"),
        "trim_s": _runtime_fairness_value(dense, "trim_s"),
        "runtime_fairness_mode": _runtime_fairness_mode(dense),
    }
    provider_runtime_fairness = {
        "resident_compute_s": _runtime_fairness_value(provider, "resident_compute_s"),
        "serving_hot_s": _runtime_fairness_value(provider, "serving_hot_s"),
        "artifact_read_s": _runtime_fairness_value(provider, "artifact_read_s"),
        "artifact_load_s": _runtime_fairness_value(provider, "artifact_load_s"),
        "artifact_unload_s": _runtime_fairness_value(provider, "artifact_unload_s"),
        "trim_s": _runtime_fairness_value(provider, "trim_s"),
        "runtime_fairness_mode": _runtime_fairness_mode(provider),
    }
    payload: dict[str, Any] = {
        "status": "ok" if dense.get("status") == "ok" and provider.get("status") == "ok" else "partial",
        "network": provider.get("network", dense.get("network")),
        "backend": provider.get("backend", dense.get("backend")),
        "label": provider.get("label", dense.get("label")),
        "activation": provider.get("activation", dense.get("activation")),
        "dense_path": str(Path(dense_path)),
        "provider_path": str(Path(provider_path)),
        "dense": {
            "status": dense.get("status"),
            "timing_s": dense.get("timing_s", {}),
            "measured_forward_mean_timing_s": dense.get("measured_forward_mean_timing_s", {}),
            "runtime_s": _artifact_runtime(dense),
            "mae_vs_clear": dense.get("mae_vs_clear"),
            "input_level": dense.get("input_level"),
            "input_ciphertext_count": dense.get("input_ciphertext_count"),
            "output_ciphertext_count": dense.get("output_ciphertext_count"),
            "rotation_eval_count_estimate": _rotation_total(dense),
            "bootstrap_count": _bootstrap_count(dense),
            "profile_category_s": _profile_category_seconds(dense),
            "runtime_fairness_timing": dense.get("measured_runtime_fairness_timing")
            or dense.get("runtime_fairness_timing_after_forward")
            or dense.get("runtime_fairness_timing")
            or {},
            **dense_runtime_fairness,
        },
        "provider": {
            "status": provider.get("status"),
            "timing_s": provider.get("timing_s", {}),
            "measured_forward_mean_timing_s": provider.get("measured_forward_mean_timing_s", {}),
            "runtime_s": _artifact_runtime(provider),
            "mae_vs_clear": provider.get("mae_vs_clear"),
            "input_level": provider.get("input_level"),
            "input_ciphertext_count": provider.get("input_ciphertext_count"),
            "output_ciphertext_count": provider.get("output_ciphertext_count"),
            "rotation_eval_count_estimate": _rotation_total(provider),
            "bootstrap_count": _bootstrap_count(provider),
            "profile_category_s": _profile_category_seconds(provider),
            "attach_audit": provider.get("attach_audit", {}),
            "runtime_fairness_timing": provider.get("measured_runtime_fairness_timing")
            or provider.get("runtime_fairness_timing_after_forward")
            or provider.get("runtime_fairness_timing")
            or {},
            **provider_runtime_fairness,
        },
    }
    if dense.get("status") == "ok" and provider.get("status") == "ok":
        dense_values = torch.tensor(dense["decoded"]["values"], dtype=torch.float32)
        provider_values = torch.tensor(provider["decoded"]["values"], dtype=torch.float32)
        clear_dense = torch.tensor(dense["clear"]["values"], dtype=torch.float32)
        clear_provider = torch.tensor(provider["clear"]["values"], dtype=torch.float32)
        payload["clear_consistency"] = _metrics(clear_dense, clear_provider)
        payload["provider_vs_dense_decoded"] = _metrics(dense_values, provider_values)
        dense_he = _he_forward_runtime(dense)
        provider_he = _he_forward_runtime(provider)
        dense_compile = float(dense.get("timing_s", {}).get("compile", math.nan))
        provider_compile = float(provider.get("timing_s", {}).get("compile", math.nan))
        dense_runtime = _artifact_runtime(dense)
        provider_runtime = _artifact_runtime(provider)
        dense_resident = dense_runtime_fairness["resident_compute_s"]
        provider_resident = provider_runtime_fairness["resident_compute_s"]
        dense_serving = dense_runtime_fairness["serving_hot_s"]
        provider_serving = provider_runtime_fairness["serving_hot_s"]
        payload["ratios"] = {
            "he_forward_dense_over_provider": (
                float(dense_he / provider_he) if provider_he and math.isfinite(provider_he) else None
            ),
            "runtime_dense_over_provider": _ratio_or_none(dense_resident, provider_resident),
            "resident_compute_dense_over_provider": _ratio_or_none(dense_resident, provider_resident),
            "serving_hot_dense_over_provider": _ratio_or_none(dense_serving, provider_serving),
            "artifact_runtime_dense_over_provider": _ratio_or_none(dense_runtime, provider_runtime),
            "runtime_speedup_metric": "resident_compute_s",
            "compile_dense_over_provider": (
                float(dense_compile / provider_compile)
                if provider_compile and math.isfinite(provider_compile)
                else None
            ),
            "rotation_dense_over_provider": (
                float(_rotation_total(dense) / _rotation_total(provider))
                if _rotation_total(provider) not in (None, 0) and _rotation_total(dense) is not None
                else None
            ),
        }
    _write(payload, out_path)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or summarize Orion E2E dense/provider comparisons.")
    parser.add_argument("--mode", choices=("dense", "provider", "summarize"), required=True)
    parser.add_argument("--backend", choices=("lattigo", "cheddar"), default="lattigo")
    parser.add_argument("--network", choices=tuple(sorted(NETWORKS)), default="r34_imgnet")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dense-path", type=Path, default=Path("/tmp/orion_e2e_dense.json"))
    parser.add_argument("--provider-path", type=Path, default=Path("/tmp/orion_e2e_provider.json"))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--forward-runs",
        type=int,
        default=1,
        help="Run this many HE forward attempts after a single compile/load phase.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Run this many unmeasured HE forward attempts first, keeping boot/rotation keys resident.",
    )
    parser.add_argument("--profile-modules", action="store_true")
    parser.add_argument(
        "--trace-forward-memory",
        action="store_true",
        help="Write per-module forward memory/live-ciphertext events to a JSONL sidecar.",
    )
    parser.add_argument("--provider-mode", type=str, default=None)
    parser.add_argument(
        "--provider-no-hybrid",
        action="store_true",
        help="Keep provider mode enabled but disable provider real/imag hybrid packing before compile.",
    )
    parser.add_argument("--io-mode", choices=("none", "save", "load"), default="none")
    parser.add_argument("--io-dir", type=Path, default=None)
    parser.add_argument("--diags-path", type=Path, default=None)
    parser.add_argument("--keys-path", type=Path, default=None)
    parser.add_argument("--logn-override", type=int, default=None)
    parser.add_argument("--activation", choices=("none", "relu", "silu"), default=None)
    parser.add_argument("--silu-degree", type=int, default=31)
    parser.add_argument(
        "--ckks-preset",
        choices=("network-default", "resnet"),
        default="network-default",
        help="Override the network's default CKKS/bootstrapping parameters.",
    )
    args = parser.parse_args()

    activation = None if args.activation in (None, "none") else str(args.activation)

    if str(args.mode) == "summarize":
        _summarize(dense_path=Path(args.dense_path), provider_path=Path(args.provider_path), out_path=Path(args.out))
        return 0

    _run_one(
        network=str(args.network),
        backend=str(args.backend),
        mode=str(args.mode),
        out_path=Path(args.out),
        seed=int(args.seed),
        compile_only=bool(args.compile_only),
        forward_runs=int(args.forward_runs),
        warmup_runs=int(args.warmup_runs),
        profile_modules=bool(args.profile_modules),
        trace_forward_memory=bool(args.trace_forward_memory),
        provider_mode_override=args.provider_mode,
        provider_no_hybrid=bool(args.provider_no_hybrid),
        io_mode=str(args.io_mode),
        io_dir=args.io_dir,
        diags_path=args.diags_path,
        keys_path=args.keys_path,
        logn_override=args.logn_override,
        activation=activation,
        silu_degree=int(args.silu_degree),
        ckks_preset=str(args.ckks_preset),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
