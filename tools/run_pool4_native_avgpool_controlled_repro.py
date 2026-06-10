#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import orion.nn as on  # noqa: E402
from orion.backend.python.tensors import CipherTensor  # noqa: E402
from orion.core.orion import scheme  # noqa: E402
from tools.medseg_cheb7_orion_adapter import build_orion_cheb7_model_from_checkpoint  # noqa: E402
from tools.run_lattigo_e2e_compare import (  # noqa: E402
    _collect_bootstrap_report,
    _layer_mae_decode_cipher_output,
    _layer_mae_metric,
    _layer_mae_tensor,
    _layer_mae_tensor_summary,
    _name_bootstraps,
)
from tools.run_medseg_cheb7_enc4b_pool4_handoff_probe import (  # noqa: E402
    _apply_backend_env_compat,
    _backend_name,
    _native_halo_source_plan,
    _native_source_blocks_and_active_mask,
    _runtime_profile,
    _serialise_native_stripes,
    _tensor_metric,
    _write,
)
from tools.verify_medseg_cheb7_orion_clear_adapter import (  # noqa: E402
    DEFAULT_CHECKPOINTS,
    DEFAULT_REPLACEMENTS,
    _backend_config,
    _load_val_sample,
)


DEFAULT_OUT_ROOT = REPO_ROOT / ".tmp" / "results" / "haloed_paper_eval" / "pool4_native_avgpool_controlled_repro"
DEFAULT_DATA_ROOT = REPO_ROOT / "artifacts" / "haloed_paper_eval" / "fixtures" / "fhelipe_medseg_accuracy10"
U22_POOL4_SOURCE_SIGNATURE = (
    (0, 14, 0, 64),
    (0, 14, 64, 64),
    (0, 14, 128, 64),
    (0, 14, 192, 64),
    (14, 26, 0, 64),
    (14, 26, 64, 64),
    (14, 26, 128, 64),
    (14, 26, 192, 64),
    (26, 32, 0, 128),
    (26, 32, 128, 128),
)
U22_POOL4_TARGET_SIGNATURE_FULL = (
    (0, 8, 0, 256),
    (4, 12, 0, 256),
    (8, 16, 0, 256),
)
U22_POOL4_TARGET_SIGNATURE_TERMINAL = (
    (0, 8, 0, 256),
    (7, 15, 0, 256),
    (14, 16, 0, 256),
)


class Pool4Only(on.Module):
    def __init__(self, pool4: torch.nn.Module) -> None:
        super().__init__()
        self.pool4 = pool4

    def forward(self, x):
        return self.pool4(x)


def _u22_pool4_target_signature(mode: str) -> tuple[tuple[int, int, int, int], ...]:
    normalized = str(mode or "full").strip().lower().replace("-", "_")
    if normalized in {"full", "full_graph", "overlap", "overlapped"}:
        return U22_POOL4_TARGET_SIGNATURE_FULL
    if normalized in {"terminal", "encoder", "encoder_pool4", "good"}:
        return U22_POOL4_TARGET_SIGNATURE_TERMINAL
    raise ValueError(f"unknown U22 pool4 target signature mode: {mode!r}")


def _force_u22_pool4_signature(pool4: Any, *, target_mode: str = "full") -> None:
    pool4.layout_policy_input_physical_layout = "native_source_stripe"
    pool4.layout_policy_output_materialization = "native_halo_stripe"
    pool4.native_halo_output_storage_layout = "native_source_stripe"
    pool4.layout_policy_native_input_source_signature = [
        [int(value) for value in item] for item in U22_POOL4_SOURCE_SIGNATURE
    ]
    pool4.layout_policy_native_output_target_signature = [
        [int(value) for value in item] for item in _u22_pool4_target_signature(str(target_mode))
    ]
    pool4.layout_policy_native_halo_channel_fold_mode = "per_stripe"


def _json_shape(value: Any) -> list[int]:
    try:
        return [int(v) for v in tuple(value)]
    except Exception:
        return []


def _pool4_plan_payload(plan: Any, active_mask: torch.Tensor) -> dict[str, Any]:
    return {
        "input_ct_count": int(getattr(plan, "input_ct_count", 0) or 0),
        "output_ct_count": int(getattr(plan, "output_ct_count", 0) or 0),
        "output_storage_layout": str((plan.to_dict()).get("output_storage_layout", "")),
        "target_internal_halo_overlap": int(getattr(plan, "target_internal_halo_overlap", 0) or 0),
        "independent_source_target_stripes": bool(getattr(plan, "independent_source_target_stripes", False)),
        "source_storage_signature": [
            [int(value) for value in item]
            for item in tuple(getattr(plan, "source_storage_signature", ()) or ())
        ],
        "target_storage_signature": [
            [int(value) for value in item]
            for item in tuple(getattr(plan, "target_storage_signature", ()) or ())
        ],
        "source_stripes": _serialise_native_stripes(getattr(plan, "source_stripes", ()) or ()),
        "target_stripes": _serialise_native_stripes(getattr(plan, "target_stripes", ()) or ()),
        "effective_source_stripes": _serialise_native_stripes(getattr(plan, "effective_source_stripes", ()) or ()),
        "effective_target_stripes": _serialise_native_stripes(getattr(plan, "effective_target_stripes", ()) or ()),
        "active_slot_count": int(active_mask.sum().item()),
        "inactive_slot_count": int((~active_mask).sum().item()),
    }


def _apply_runner_env(args: argparse.Namespace) -> dict[str, str]:
    env = _apply_backend_env_compat(args)
    workers = max(1, int(args.workers))
    extra = {
        "PYTHONUNBUFFERED": "1",
        "MALLOC_ARENA_MAX": "2",
        "GOMAXPROCS": str(workers),
        "ORION_COMPILE_PARALLEL_POLICY": "manual",
        "ORION_SINGLE_SLOT_LAYER_CACHE": "1",
        "ORION_SINGLE_SLOT_ENCODE_WORKERS": str(workers),
        "ORION_PACK_CONV_WORKERS": str(workers),
        "ORION_DIRECT_PACK_WORKERS": str(workers),
        "ORION_LT_COMPILE_WORKERS": str(workers),
        "ORION_UNIFIED_COMPILE_WORKERS": str(workers),
        "ORION_LATTIGO_COMPILE_WORKERS": str(workers),
        "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS": str(workers),
        "ORION_UNIFIED_LT_OUTPUT_FUSION": "1",
        "ORION_LAYOUT_POLICY_RELAYOUT_KERNEL": "0",
        "ORION_LAYOUT_POLICY_PROVIDER_NATIVE_HALO": "1",
        "ORION_UNIFIED_LT_INDIVIDUAL_EVAL": "1",
        "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS": "0",
        "ORION_CONCAT_FUSION": "0",
        "ORION_BOOTSTRAP_LAYOUT_REFINEMENT": "0",
        "ORION_COMPILE_SKIP_BOOTSTRAPPER_GENERATION": "0"
        if (bool(args.bootstrap_ab) or bool(getattr(args, "actual_provider_ab", False)))
        else "1",
        "ORION_LATTIGO_BOOTSTRAP_MANY": str(args.bootstrap_many),
    }
    if str(getattr(args, "stop_after_layer_compile", "") or "").strip():
        extra["ORION_STOP_AFTER_LAYER_COMPILE"] = str(args.stop_after_layer_compile).strip()
    if bool(getattr(args, "force_u22_pool4_signature", False)) or str(
        getattr(args, "pool4_target_signature", "default") or "default"
    ) != "default":
        extra["ORION_RESPECT_FORCED_NATIVE_SIGNATURE_ATTRS"] = "1"
    if bool(getattr(args, "lattigo_no_bsgs", False)):
        extra["ORION_LATTIGO_UNIFIED_NO_BSGS"] = "1"
    if bool(getattr(args, "disable_sources_target_sum_fusion", False)):
        extra["ORION_DISABLE_SOURCES_TARGET_SUM_FUSION"] = "1"
    if bool(getattr(args, "disable_cpp_diag_builder", False)):
        extra.update(
            {
                "ORION_CPP_DIAG_BUILDER": "0",
                "ORION_CPP_DIAG_BUILDER_PROVIDER": "0",
                "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE": "0",
                "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE_SINGLE_SLOT_METADATA": "0",
            }
        )
    for key, value in extra.items():
        os.environ[str(key)] = str(value)
        env[str(key)] = str(value)
    return {key: str(env[key]) for key in sorted(env)}


def _source_signature_for_group(plan: Any, index: int) -> tuple[int, int, int, int]:
    signature = tuple(tuple(int(v) for v in raw) for raw in getattr(plan, "source_storage_signature", ()) or ())
    if int(index) < 0 or int(index) >= len(signature):
        raise IndexError(f"group index {index} outside source signature length {len(signature)}")
    return signature[int(index)]


def _pattern_input(
    *,
    plan: Any,
    active_groups: set[int] | None,
    scale: float,
) -> torch.Tensor:
    spec = plan.spec
    source = torch.zeros((1, int(spec.c_in), int(spec.h_in), int(spec.w_in)), dtype=torch.float32)
    signature = tuple(tuple(int(v) for v in raw) for raw in getattr(plan, "source_storage_signature", ()) or ())
    groups = set(range(len(signature))) if active_groups is None else {int(v) for v in active_groups}
    for group_index in sorted(groups):
        h_start, h_end, c_start, c_count = _source_signature_for_group(plan, int(group_index))
        c_end = int(c_start) + int(c_count)
        h = torch.arange(int(h_start), int(h_end), dtype=torch.float32).view(1, -1, 1)
        w = torch.arange(int(spec.w_in), dtype=torch.float32).view(1, 1, -1)
        c = torch.arange(int(c_start), int(c_end), dtype=torch.float32).view(-1, 1, 1)
        values = (
            float(scale)
            * (
                float(group_index + 1)
                + 0.01 * h
                + 0.0001 * w
                + 0.000001 * c
            )
        )
        source[0, int(c_start):int(c_end), int(h_start):int(h_end), :] = values
    return source


def _encrypt_blocks(
    blocks: torch.Tensor,
    *,
    level: int,
    shape: torch.Size,
    inactive_mask: torch.Tensor,
    inactive_value: float,
    inactive_poison_groups: set[int] | None = None,
) -> CipherTensor:
    materialized = blocks.detach().cpu().to(dtype=torch.float32).clone()
    if float(inactive_value) != 0.0:
        poison_mask = inactive_mask.detach().cpu().to(dtype=torch.bool)
        if inactive_poison_groups is not None:
            group_mask = torch.zeros_like(poison_mask, dtype=torch.bool)
            for group_index in sorted(int(value) for value in inactive_poison_groups):
                if int(group_index) < 0 or int(group_index) >= int(group_mask.shape[0]):
                    raise IndexError(
                        f"inactive poison group {int(group_index)} outside source block count {int(group_mask.shape[0])}"
                    )
                group_mask[int(group_index), :] = True
            poison_mask = poison_mask & group_mask
        materialized[poison_mask] = float(inactive_value)
    ids: list[int] = []
    for block in materialized:
        ptxt = scheme.encode(block, int(level))
        ct = scheme.encrypt(ptxt)
        ids.append(int(ct.ids[0]))
        ct.ids = []
        ptxt.release()
    return CipherTensor(
        scheme,
        ids,
        torch.Size(tuple(int(v) for v in shape)),
        torch.Size([int(len(ids)), int(materialized.shape[-1])]),
    )


def _cipher_meta(value: Any) -> dict[str, Any]:
    ids = [int(v) for v in tuple(getattr(value, "ids", ()) or ())]
    meta: dict[str, Any] = {
        "ct_count": int(len(ids)),
        "shape": _json_shape(getattr(value, "shape", ())),
        "on_shape": _json_shape(getattr(value, "on_shape", ())),
    }
    if ids:
        for name, fn in (
            ("level", lambda: int(value.level())),
            ("scale", lambda: int(value.scale())),
            ("scale_log2", lambda: float(value.scale_log2())),
            ("slots", lambda: int(value.slots())),
        ):
            try:
                meta[name] = fn()
            except Exception:
                pass
    return meta


def _raw_blocks_vs_expected(value: CipherTensor, expected_blocks: torch.Tensor, active_mask: torch.Tensor) -> dict[str, Any]:
    plain = value.decrypt()
    try:
        raw = plain.decode().detach().cpu().to(dtype=torch.float32)
    finally:
        release = getattr(plain, "release", None)
        if callable(release):
            release()
    raw_blocks = raw.reshape(tuple(expected_blocks.shape))
    return {
        "all_slots": _tensor_metric(expected_blocks, raw_blocks),
        "active_slots": _tensor_metric(expected_blocks, raw_blocks, active_mask),
        "inactive_slots": _tensor_metric(expected_blocks, raw_blocks, ~active_mask),
    }


def _bootstrapper_payload(bootstrapper: Any | None) -> dict[str, Any]:
    if bootstrapper is None:
        return {}
    payload: dict[str, Any] = {
        "type": f"{type(bootstrapper).__module__}.{type(bootstrapper).__qualname__}",
        "preprocess_fused": bool(getattr(bootstrapper, "preprocess_fused", False)),
        "preprocess_fusion_kind": str(getattr(bootstrapper, "preprocess_fusion_kind", "") or ""),
    }
    for name in ("input_level", "bootstrap_slots", "prescale", "postscale", "constant", "low", "high"):
        if hasattr(bootstrapper, name):
            try:
                value = getattr(bootstrapper, name)
                payload[name] = float(value) if isinstance(value, float) else int(value)
            except Exception:
                payload[name] = str(getattr(bootstrapper, name))
    active_mask = getattr(bootstrapper, "_bootstrap_prescale_active_mask", None)
    if active_mask is not None:
        mask = active_mask.detach().cpu().to(dtype=torch.bool)
        payload["active_mask"] = {
            "shape": [int(v) for v in tuple(mask.shape)],
            "active": int(mask.sum().item()),
            "inactive": int((~mask).sum().item()),
        }
    return payload


def _apply_pre_pool4_bootstrap(
    cipher: CipherTensor,
    *,
    bootstrapper: Any,
    mode: str,
) -> tuple[CipherTensor, dict[str, Any]]:
    normalized = str(mode or "unfused").strip().lower().replace("-", "_")
    if normalized not in {"unfused", "fused_plain_prescale"}:
        raise ValueError(f"unknown bootstrap AB mode: {mode!r}")
    before = _cipher_meta(cipher)
    original_preprocess_fused = bool(getattr(bootstrapper, "preprocess_fused", False))
    ptxt = None
    transformed = cipher
    try:
        if normalized == "fused_plain_prescale":
            ptxt = bootstrapper._get_prescale_ptxt(transformed.level())
            transformed = transformed * ptxt
            setattr(bootstrapper, "preprocess_fused", True)
        else:
            setattr(bootstrapper, "preprocess_fused", False)
        out = bootstrapper(transformed)
    finally:
        setattr(bootstrapper, "preprocess_fused", original_preprocess_fused)
        if transformed is not cipher:
            release = getattr(transformed, "release", None)
            if callable(release):
                release()
    return out, {
        "mode": normalized,
        "before": before,
        "after": _cipher_meta(out),
        "bootstrapper": _bootstrapper_payload(bootstrapper),
        "runtime_profile": list(getattr(bootstrapper, "_bootstrap_runtime_profile", []) or []),
    }


def _target_stripe_metrics(reference: torch.Tensor, actual: torch.Tensor, plan: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ref = reference.detach().cpu().to(dtype=torch.float32)
    act = actual.detach().cpu().to(dtype=torch.float32)
    for index, raw in enumerate(tuple(getattr(plan, "target_storage_signature", ()) or ())):
        h_start, h_end, c_start, c_count = (int(v) for v in raw)
        c_end = int(c_start) + int(c_count)
        r = ref[:, int(c_start):int(c_end), int(h_start):int(h_end), :]
        a = act[:, int(c_start):int(c_end), int(h_start):int(h_end), :]
        metric = _layer_mae_metric(r, a)
        rows.append(
            {
                "target_group": int(index),
                "signature": [int(h_start), int(h_end), int(c_start), int(c_count)],
                "mae": metric.get("mae"),
                "max_abs": metric.get("max_abs"),
                "rmse": metric.get("rmse"),
            }
        )
    return rows


def _run_case(
    *,
    model: torch.nn.Module,
    plan: Any,
    active_mask: torch.Tensor,
    level: int,
    name: str,
    active_groups: set[int] | None,
    inactive_value: float,
    inactive_poison_groups: set[int] | None,
    pattern_scale: float,
    bootstrapper: Any | None = None,
    bootstrap_mode: str = "",
) -> dict[str, Any]:
    source = _pattern_input(plan=plan, active_groups=active_groups, scale=float(pattern_scale))
    expected_output = F.avg_pool2d(source, kernel_size=2, stride=2)
    expected_blocks, expected_active_mask = _native_source_blocks_and_active_mask(source, plan)
    if tuple(expected_active_mask.shape) != tuple(active_mask.shape):
        raise RuntimeError("active mask shape changed across controlled cases")
    cipher = _encrypt_blocks(
        expected_blocks,
        level=int(level),
        shape=torch.Size(tuple(int(v) for v in source.shape)),
        inactive_mask=~expected_active_mask,
        inactive_value=float(inactive_value),
        inactive_poison_groups=inactive_poison_groups,
    )
    pre_pool4_bootstrap: dict[str, Any] = {"enabled": False}
    pool_input = cipher
    if bootstrapper is not None:
        pre_pool4_bootstrap = {"enabled": True, "requested_mode": str(bootstrap_mode)}
        booted, boot_payload = _apply_pre_pool4_bootstrap(
            cipher,
            bootstrapper=bootstrapper,
            mode=str(bootstrap_mode or "unfused"),
        )
        pre_pool4_bootstrap.update(boot_payload)
        pre_pool4_bootstrap["raw_vs_expected_after_bootstrap"] = _raw_blocks_vs_expected(
            booted,
            expected_blocks,
            expected_active_mask,
        )
        pool_input = booted
    started = time.perf_counter()
    out = model.pool4(pool_input)
    pool_s = float(time.perf_counter() - started)
    decode_info: dict[str, Any] = {}
    decoded = _layer_mae_decode_cipher_output(model.pool4, out, module_name="pool4", decode_info=decode_info)
    decoded = _layer_mae_tensor(decoded)
    result = {
        "name": str(name),
        "active_groups": None if active_groups is None else sorted(int(v) for v in active_groups),
        "inactive_poison_groups": (
            None if inactive_poison_groups is None else sorted(int(v) for v in inactive_poison_groups)
        ),
        "inactive_value": float(inactive_value),
        "pattern_scale": float(pattern_scale),
        "source_summary": _layer_mae_tensor_summary(source),
        "expected_output_summary": _layer_mae_tensor_summary(expected_output),
        "cipher": {
            **_cipher_meta(cipher),
        },
        "pool_input_cipher": _cipher_meta(pool_input),
        "pre_pool4_bootstrap": dict(pre_pool4_bootstrap),
        "pool4_s": float(pool_s),
        "decode_info": dict(decode_info),
        "metrics_vs_clear": _layer_mae_metric(expected_output, decoded),
        "target_stripe_metrics": _target_stripe_metrics(expected_output, decoded, plan),
        "input_blocks_vs_expected_active": _tensor_metric(expected_blocks, expected_blocks, expected_active_mask),
    }
    for value in (out, pool_input, cipher):
        release = getattr(value, "release", None)
        if callable(release) and value is not cipher:
            release()
    release = getattr(cipher, "release", None)
    if callable(release):
        release()
    return result


def _module_runtime_payload(module: Any) -> dict[str, Any]:
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None) if runtime is not None else None
    payload: dict[str, Any] = {
        "module": type(module).__name__,
        "runtime": type(runtime).__name__ if runtime is not None else "",
        "executor": type(executor).__name__ if executor is not None else "",
    }
    for attr in ("last_runtime_io", "last_runtime_timing", "last_runtime_counts"):
        value = getattr(executor, attr, None) if executor is not None else getattr(module, attr, None)
        if value is not None:
            payload[attr] = value
    return payload


def _conv2d_reference(module: Any, x: torch.Tensor) -> torch.Tensor:
    return F.conv2d(
        x,
        module.weight.detach().cpu(),
        None if getattr(module, "bias", None) is None else module.bias.detach().cpu(),
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=int(module.groups),
    )


def _clear_module_call(module: Any, x: torch.Tensor) -> torch.Tensor:
    old_he_mode = getattr(module, "he_mode", None)
    try:
        if old_he_mode is not None:
            module.he_mode = False
        return module(x)
    finally:
        if old_he_mode is not None:
            module.he_mode = bool(old_he_mode)


def _run_actual_provider_window_case(
    *,
    model: torch.nn.Module,
    active_groups: set[int] | None,
    inactive_value: float,
    inactive_poison_groups: set[int] | None,
    pattern_scale: float,
    name: str,
) -> dict[str, Any]:
    enc4b = getattr(model, "enc4b", None)
    enc4b_act = getattr(model, "enc4b_act", None)
    pool4 = getattr(model, "pool4", None)
    if enc4b is None or enc4b_act is None or pool4 is None:
        raise RuntimeError("actual provider AB requires enc4b, enc4b_act, and pool4 modules")
    plan = _native_halo_source_plan(enc4b)
    if plan is None:
        raise RuntimeError("enc4b has no native halo source plan")

    source = _pattern_input(plan=plan, active_groups=active_groups, scale=float(pattern_scale))
    expected_conv = _conv2d_reference(enc4b, source)
    with torch.no_grad():
        expected_act = _clear_module_call(enc4b_act, expected_conv.detach().clone())
        expected_pool4 = F.avg_pool2d(expected_act, kernel_size=2, stride=2)
    expected_blocks, expected_active_mask = _native_source_blocks_and_active_mask(source, plan)
    level = int(getattr(enc4b, "level", 0) or 0)
    cipher = _encrypt_blocks(
        expected_blocks,
        level=int(level),
        shape=torch.Size(tuple(int(v) for v in source.shape)),
        inactive_mask=~expected_active_mask,
        inactive_value=float(inactive_value),
        inactive_poison_groups=inactive_poison_groups,
    )
    result: dict[str, Any] = {
        "name": str(name),
        "active_groups": None if active_groups is None else sorted(int(v) for v in active_groups),
        "inactive_poison_groups": (
            None if inactive_poison_groups is None else sorted(int(v) for v in inactive_poison_groups)
        ),
        "inactive_value": float(inactive_value),
        "pattern_scale": float(pattern_scale),
        "enc4b_level": int(level),
        "enc4b_bootstrapper": _bootstrapper_payload(getattr(enc4b, "bootstrapper", None)),
        "enc4b_native_plan": _pool4_plan_payload(plan, expected_active_mask),
        "source_summary": _layer_mae_tensor_summary(source),
        "expected_conv_summary": _layer_mae_tensor_summary(expected_conv),
        "expected_act_summary": _layer_mae_tensor_summary(expected_act),
        "expected_pool4_summary": _layer_mae_tensor_summary(expected_pool4),
        "cipher": _cipher_meta(cipher),
    }
    enc4b_out = None
    act_out = None
    pool_out = None
    try:
        enc4b_started = time.perf_counter()
        enc4b_out = enc4b(cipher)
        result["enc4b_s"] = float(time.perf_counter() - enc4b_started)
        enc4b_decode_info: dict[str, Any] = {}
        decoded_enc4b = _layer_mae_tensor(
            _layer_mae_decode_cipher_output(enc4b, enc4b_out, module_name="enc4b", decode_info=enc4b_decode_info)
        )
        result["enc4b_output_cipher"] = _cipher_meta(enc4b_out)
        result["enc4b_decode_info"] = dict(enc4b_decode_info)
        result["enc4b_metrics_vs_clear"] = _layer_mae_metric(expected_conv, decoded_enc4b)
        result["enc4b_runtime"] = _module_runtime_payload(enc4b)

        act_started = time.perf_counter()
        act_out = enc4b_act(enc4b_out)
        result["enc4b_act_s"] = float(time.perf_counter() - act_started)
        act_decode_info: dict[str, Any] = {}
        decoded_act = _layer_mae_tensor(
            _layer_mae_decode_cipher_output(
                enc4b_act,
                act_out,
                module_name="enc4b_act",
                decode_info=act_decode_info,
            )
        )
        result["enc4b_act_output_cipher"] = _cipher_meta(act_out)
        result["enc4b_act_decode_info"] = dict(act_decode_info)
        result["enc4b_act_metrics_vs_clear"] = _layer_mae_metric(expected_act, decoded_act)

        pool_started = time.perf_counter()
        pool_out = pool4(act_out)
        result["pool4_s"] = float(time.perf_counter() - pool_started)
        pool_decode_info: dict[str, Any] = {}
        decoded_pool = _layer_mae_tensor(
            _layer_mae_decode_cipher_output(pool4, pool_out, module_name="pool4", decode_info=pool_decode_info)
        )
        result["pool4_output_cipher"] = _cipher_meta(pool_out)
        result["pool4_decode_info"] = dict(pool_decode_info)
        result["pool4_metrics_vs_clear"] = _layer_mae_metric(expected_pool4, decoded_pool)
        result["pool4_runtime"] = _module_runtime_payload(pool4)
    finally:
        for value in (pool_out, act_out, enc4b_out, cipher):
            release = getattr(value, "release", None)
            if callable(release):
                release()
    return result


def _case_specs(args: argparse.Namespace, plan: Any) -> list[dict[str, Any]]:
    group_count = int(getattr(plan, "input_ct_count", 0) or 0)
    cases: list[dict[str, Any]] = [
        {
            "name": "all_active_inactive_zero",
            "active_groups": None,
            "inactive_value": 0.0,
        },
        {
            "name": "all_active_inactive_poison",
            "active_groups": None,
            "inactive_value": float(args.inactive_poison),
        },
    ]
    if bool(args.group_sweep):
        for group_index in range(int(group_count)):
            cases.append(
                {
                    "name": f"group_{group_index:02d}_inactive_zero",
                    "active_groups": {int(group_index)},
                    "inactive_value": 0.0,
                }
            )
            if bool(args.group_poison):
                cases.append(
                    {
                        "name": f"group_{group_index:02d}_inactive_poison",
                        "active_groups": {int(group_index)},
                        "inactive_value": float(args.inactive_poison),
                    }
                )
    if bool(args.group_drop_sweep):
        all_groups = set(range(int(group_count)))
        for group_index in range(int(group_count)):
            cases.append(
                {
                    "name": f"drop_group_{group_index:02d}_inactive_zero",
                    "active_groups": all_groups - {int(group_index)},
                    "inactive_value": 0.0,
                }
            )
            if bool(args.group_poison):
                cases.append(
                    {
                        "name": f"drop_group_{group_index:02d}_inactive_poison",
                        "active_groups": all_groups - {int(group_index)},
                        "inactive_value": float(args.inactive_poison),
                    }
                )
    if bool(args.inactive_poison_sweep):
        for group_index in range(int(group_count)):
            cases.append(
                {
                    "name": f"poison_inactive_group_{group_index:02d}",
                    "active_groups": None,
                    "inactive_value": float(args.inactive_poison),
                    "inactive_poison_groups": {int(group_index)},
                }
            )
    return cases


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = out_dir / out_path
    checkpoint = Path(args.checkpoint or DEFAULT_CHECKPOINTS["covid19"])
    replacements = Path(args.replacements or DEFAULT_REPLACEMENTS["covid19"])
    payload: dict[str, Any] = {
        "status": "started",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "out": str(out_path),
        "backend_kind": str(args.backend_kind),
        "provider_mode": str(args.provider_mode),
        "compile_scope": str(args.compile_scope),
        "checkpoint": str(checkpoint),
        "replacements": str(replacements),
        "inactive_poison": float(args.inactive_poison),
        "pattern_scale": float(args.pattern_scale),
        "force_u22_pool4_signature": bool(args.force_u22_pool4_signature),
        "pool4_target_signature": str(args.pool4_target_signature),
        "bootstrap_ab": bool(args.bootstrap_ab),
        "bootstrap_ab_modes": [str(v) for v in tuple(args.bootstrap_ab_mode or ())],
        "actual_provider_ab": bool(args.actual_provider_ab),
        "fit_source": str(args.fit_source),
    }
    _write(out_path, payload)
    try:
        if str(args.compile_scope) == "pool4_only":
            pool4 = on.AvgPool2d(kernel_size=2, stride=2)
            if bool(args.force_u22_pool4_signature) or str(args.pool4_target_signature) != "default":
                target_mode = "full" if str(args.pool4_target_signature) == "default" else str(args.pool4_target_signature)
                _force_u22_pool4_signature(pool4, target_mode=target_mode)
            model = Pool4Only(pool4)
            dummy = torch.zeros((1, 256, 32, 32), dtype=torch.float32)
            model_meta = {"pool4_only": True}
        else:
            full_model, model_meta = build_orion_cheb7_model_from_checkpoint(
                checkpoint,
                replacements_path=replacements,
                device="cpu",
            )
            if bool(args.force_u22_pool4_signature) or str(args.pool4_target_signature) != "default":
                target_mode = "full" if str(args.pool4_target_signature) == "default" else str(args.pool4_target_signature)
                _force_u22_pool4_signature(full_model.pool4, target_mode=target_mode)
            model = full_model
            if str(args.fit_source) == "sample":
                dummy, _mask, sample_meta = _load_val_sample(
                    dataset="covid19",
                    data_root=Path(args.data_root),
                    image_size=int(args.image_size),
                    val_index=int(args.val_index),
                    seed=int(args.seed),
                )
                payload["fit_sample"] = dict(sample_meta)
            else:
                dummy = torch.zeros((1, 1, int(args.image_size), int(args.image_size)), dtype=torch.float32)
        model.eval()

        env = _apply_runner_env(args)
        config = _backend_config(provider_mode=str(args.provider_mode), mode="provider")
        config["orion"]["backend"] = _backend_name(str(args.backend_kind))
        payload.update({"status": "init_scheme", "env": dict(env), "config": config, "model_meta": dict(model_meta)})
        _write(out_path, payload)

        scheme.init_scheme(config)
        payload["status"] = "fit"
        _write(out_path, payload)
        scheme.fit(model, dummy)
        payload["status"] = "compile"
        _write(out_path, payload)
        partial_compile: dict[str, Any] = {}
        try:
            input_level = scheme.compile(model)
        except RuntimeError as exc:
            stop_after = str(getattr(args, "stop_after_layer_compile", "") or "").strip()
            sentinel = f"ORION_STOP_AFTER_LAYER_COMPILE:{stop_after}"
            if not stop_after or str(exc) != sentinel:
                raise
            partial_input_level = getattr(scheme, "partial_compile_input_level", None)
            if partial_input_level is None:
                raise RuntimeError(f"{sentinel} did not publish partial_compile_input_level") from exc
            input_level = int(partial_input_level)
            partial_compile = {
                "stop_layer": str(getattr(scheme, "partial_compile_stop_layer", stop_after)),
                "compiled_linear_layers": [
                    str(getattr(layer, "name", "")) for layer in tuple(getattr(scheme, "partial_compiled_linear_layers", ()) or ())
                ],
            }
        _name_bootstraps(model)
        for module in model.modules():
            if hasattr(module, "he_mode"):
                module.he_mode = True
        plan = _native_halo_source_plan(model.pool4)
        if plan is None:
            raise RuntimeError("pool4 has no native halo source plan")
        probe_source = _pattern_input(plan=plan, active_groups=None, scale=float(args.pattern_scale))
        _, active_mask = _native_source_blocks_and_active_mask(probe_source, plan)
        pool4_level = int(getattr(model.pool4, "level", input_level) or input_level)
        payload.update(
            {
                "status": "compiled",
                "input_level": int(input_level),
                "pool4_level": int(pool4_level),
                "pool4_native_plan": _pool4_plan_payload(plan, active_mask),
                "pool4_module_layout": {
                    "layout_policy_input_physical_layout": str(
                        getattr(model.pool4, "layout_policy_input_physical_layout", "") or ""
                    ),
                    "layout_policy_output_materialization": str(
                        getattr(model.pool4, "layout_policy_output_materialization", "") or ""
                    ),
                    "native_halo_output_storage_layout": str(
                        getattr(model.pool4, "native_halo_output_storage_layout", "") or ""
                    ),
                },
                "bootstrap_report_after_compile": _collect_bootstrap_report(model),
                "enc4b_bootstrapper": _bootstrapper_payload(
                    getattr(getattr(model, "enc4b", None), "bootstrapper", None)
                ),
                "partial_compile": dict(partial_compile),
                "cases": [],
            }
        )
        _write(out_path, payload)
        if bool(args.compile_only):
            payload["status"] = "ok_compile_only"
            payload["elapsed_s"] = float(time.perf_counter() - started)
            _write(out_path, payload)
            return payload

        bootstrapper = getattr(getattr(model, "enc4b", None), "bootstrapper", None)
        if bool(args.bootstrap_ab):
            if bootstrapper is None:
                raise RuntimeError("--bootstrap-ab requires a compiled model with enc4b.bootstrapper")
            try:
                scheme.bootstrapper.generate_bootstrapper(int(getattr(bootstrapper, "bootstrap_slots", 32768) or 32768))
            except Exception:
                # The compile path may already have generated it.
                pass
        for spec in _case_specs(args, plan):
            payload["status"] = f"running_case:{spec['name']}"
            _write(out_path, payload)
            direct_case = _run_case(
                model=model,
                plan=plan,
                active_mask=active_mask,
                level=int(pool4_level),
                name=str(spec["name"]),
                active_groups=spec["active_groups"],
                inactive_value=float(spec["inactive_value"]),
                inactive_poison_groups=spec.get("inactive_poison_groups"),
                pattern_scale=float(args.pattern_scale),
            )
            payload["cases"].append(direct_case)
            _write(out_path, payload)
            if bool(args.bootstrap_ab):
                for mode in tuple(args.bootstrap_ab_mode or ("unfused",)):
                    payload["status"] = f"running_case:{spec['name']}:bootstrap_{mode}"
                    _write(out_path, payload)
                    boot_case = _run_case(
                        model=model,
                        plan=plan,
                        active_mask=active_mask,
                        level=int(getattr(bootstrapper, "input_level", pool4_level) or pool4_level),
                        name=f"{spec['name']}__bootstrap_{mode}",
                        active_groups=spec["active_groups"],
                        inactive_value=float(spec["inactive_value"]),
                        inactive_poison_groups=spec.get("inactive_poison_groups"),
                        pattern_scale=float(args.pattern_scale),
                        bootstrapper=bootstrapper,
                        bootstrap_mode=str(mode),
                    )
                    payload["cases"].append(boot_case)
            if bool(args.actual_provider_ab):
                payload["status"] = f"running_case:{spec['name']}:actual_provider_window"
                _write(out_path, payload)
                actual_case = _run_actual_provider_window_case(
                    model=model,
                    active_groups=spec["active_groups"],
                    inactive_value=float(spec["inactive_value"]),
                    inactive_poison_groups=spec.get("inactive_poison_groups"),
                    pattern_scale=float(args.pattern_scale),
                    name=f"{spec['name']}__actual_provider_window",
                )
                payload["cases"].append(actual_case)
            _write(out_path, payload)
        payload["status"] = "ok"
        payload["pool4_runtime_profile"] = _runtime_profile(model.pool4)
        payload["elapsed_s"] = float(time.perf_counter() - started)
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        payload["elapsed_s"] = float(time.perf_counter() - started)
    finally:
        try:
            scheme.delete_scheme()
        except Exception:
            pass
        gc.collect()
        _write(out_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled native-source pool4 repro.")
    parser.add_argument("--backend-kind", choices=("clear", "ckks"), default="ckks")
    parser.add_argument("--compile-scope", choices=("full", "pool4_only"), default="full")
    parser.add_argument("--provider-mode", default="u22_256_base32_layout_dp_no_share_fold")
    parser.add_argument("--bootstrap-many", default="1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--inactive-poison", type=float, default=4.720082944e10)
    parser.add_argument("--pattern-scale", type=float, default=0.01)
    parser.add_argument("--group-sweep", action="store_true")
    parser.add_argument("--group-drop-sweep", action="store_true")
    parser.add_argument("--inactive-poison-sweep", action="store_true")
    parser.add_argument("--group-poison", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--stop-after-layer-compile", default="")
    parser.add_argument("--force-u22-pool4-signature", action="store_true")
    parser.add_argument(
        "--pool4-target-signature",
        choices=("default", "full", "terminal"),
        default="default",
        help="Force the U22 pool4 source signature and choose the target signature.",
    )
    parser.add_argument(
        "--bootstrap-ab",
        action="store_true",
        help="For each controlled source case, also run pool4 after an enc4b bootstrapper transform.",
    )
    parser.add_argument(
        "--bootstrap-ab-mode",
        action="append",
        choices=("unfused", "fused_plain_prescale"),
        default=[],
        help="Bootstrap transform variants to run when --bootstrap-ab is set. May be repeated.",
    )
    parser.add_argument(
        "--actual-provider-ab",
        action="store_true",
        help="For each controlled case, run enc4b provider + enc4b_act + pool4 on controlled enc4b input.",
    )
    parser.add_argument("--single-slot-layer-cache", action="store_true")
    parser.add_argument("--disable-cpp-diag-builder", action="store_true")
    parser.add_argument("--lattigo-no-bsgs", action="store_true")
    parser.add_argument("--disable-sources-target-sum-fusion", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--replacements", type=Path, default=None)
    parser.add_argument("--fit-source", choices=("dummy", "sample"), default="dummy")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-index", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("pool4_native_avgpool_controlled_repro.json"))
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    summary = {
        "status": payload.get("status"),
        "out": payload.get("out"),
        "input_level": payload.get("input_level"),
        "pool4_level": payload.get("pool4_level"),
        "pool4_native_plan": payload.get("pool4_native_plan"),
        "cases": [
            {
                "name": case.get("name"),
                "mae": (case.get("metrics_vs_clear") or {}).get("mae"),
                "max_abs": (case.get("metrics_vs_clear") or {}).get("max_abs"),
            }
            for case in payload.get("cases", [])
        ],
        "error": payload.get("error"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(payload.get("status")) in {"ok", "ok_compile_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
