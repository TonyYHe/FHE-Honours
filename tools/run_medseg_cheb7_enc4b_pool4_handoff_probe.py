#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import orion.nn as on  # noqa: E402
from orion.backend.python.tensors import CipherTensor  # noqa: E402
from orion.core.orion import scheme  # noqa: E402
from tools.medseg_cheb7_orion_adapter import build_orion_cheb7_model_from_checkpoint  # noqa: E402
from tools.run_lattigo_e2e_compare import (  # noqa: E402
    _collect_bootstrap_report,
    _encrypt_model_input,
    _layer_mae_decode_cipher_output,
    _layer_mae_metric,
    _layer_mae_tensor,
    _layer_mae_tensor_summary,
    _name_bootstraps,
)
from tools.verify_medseg_cheb7_orion_clear_adapter import (  # noqa: E402
    DEFAULT_CHECKPOINTS,
    DEFAULT_REPLACEMENTS,
    _apply_backend_env,
    _backend_config,
    _load_val_sample,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "artifacts" / "haloed_paper_eval" / "fixtures" / "fhelipe_medseg_accuracy10"
DEFAULT_OUT_ROOT = REPO_ROOT / ".tmp" / "results" / "haloed_paper_eval" / "enc4b_pool4_handoff_probe"


class EncoderPool4Probe(on.Module):
    """Encoder-only wrapper that preserves the enc4b_act -> pool4 handoff."""

    def __init__(self, full_model: torch.nn.Module) -> None:
        super().__init__()
        self.enc1a = full_model.enc1a
        self.enc1a_act = full_model.enc1a_act
        self.enc1b = full_model.enc1b
        self.enc1b_act = full_model.enc1b_act
        self.pool1 = full_model.pool1
        self.enc2a = full_model.enc2a
        self.enc2a_act = full_model.enc2a_act
        self.enc2b = full_model.enc2b
        self.enc2b_act = full_model.enc2b_act
        self.pool2 = full_model.pool2
        self.enc3a = full_model.enc3a
        self.enc3a_act = full_model.enc3a_act
        self.enc3b = full_model.enc3b
        self.enc3b_act = full_model.enc3b_act
        self.pool3 = full_model.pool3
        self.enc4a = full_model.enc4a
        self.enc4a_act = full_model.enc4a_act
        self.enc4b = full_model.enc4b
        self.enc4b_act = full_model.enc4b_act
        self.pool4 = full_model.pool4

    def forward(self, x):
        skip4 = _prefix_to_enc4b_act(self, x)
        return self.pool4(skip4)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_shape(value: Any) -> list[int]:
    try:
        return [int(v) for v in tuple(value)]
    except Exception:
        return []


def _backend_name(kind: str) -> str:
    return "lattigo" if str(kind) == "ckks" else "clear_lattigo"


def _apply_backend_env_compat(args: argparse.Namespace) -> dict[str, str]:
    try:
        env = _apply_backend_env(
            backend_kind=str(args.backend_kind),
            mode="provider",
            single_slot_layer_cache=bool(args.single_slot_layer_cache),
            bootstrap_many=str(args.bootstrap_many),
        )
    except TypeError as exc:
        if "bootstrap_many" not in str(exc):
            raise
        env = _apply_backend_env(
            backend_kind=str(args.backend_kind),
            mode="provider",
            single_slot_layer_cache=bool(args.single_slot_layer_cache),
        )
        os.environ["ORION_LATTIGO_BOOTSTRAP_MANY"] = str(args.bootstrap_many)
        env["ORION_LATTIGO_BOOTSTRAP_MANY"] = str(args.bootstrap_many)
    if bool(args.disable_cpp_diag_builder):
        for key in (
            "ORION_CPP_DIAG_BUILDER",
            "ORION_CPP_DIAG_BUILDER_PROVIDER",
            "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE",
            "ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_SOURCE",
            "ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT",
            "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE_SINGLE_SLOT_METADATA",
            "ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT_SINGLE_SLOT_METADATA",
        ):
            os.environ[str(key)] = "0"
            env[str(key)] = "0"
    return {key: str(os.environ.get(key, value)) for key, value in sorted(env.items())}


def _capture_clear_io(model: torch.nn.Module, image: torch.Tensor, target: str) -> tuple[torch.Tensor, torch.Tensor]:
    modules = dict(model.named_modules())
    if str(target) not in modules:
        raise ValueError(f"unknown target module: {target}")
    module = modules[str(target)]
    holder: dict[str, torch.Tensor] = {}

    def pre_hook(_module: torch.nn.Module, values: tuple[Any, ...]) -> None:
        if values and torch.is_tensor(values[0]):
            holder["input"] = values[0].detach().cpu().to(dtype=torch.float32)

    def post_hook(_module: torch.nn.Module, _values: tuple[Any, ...], output: Any) -> None:
        if torch.is_tensor(output):
            holder["output"] = output.detach().cpu().to(dtype=torch.float32)

    h1 = module.register_forward_pre_hook(pre_hook)
    h2 = module.register_forward_hook(post_hook)
    try:
        with torch.no_grad():
            model(image)
    finally:
        h1.remove()
        h2.remove()
    if "input" not in holder or "output" not in holder:
        raise RuntimeError(f"failed to capture clear IO for {target}")
    return holder["input"], holder["output"]


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


def _native_halo_source_plan(module: Any) -> Any | None:
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None)
    if executor is None:
        return None
    for candidate in _walk_executor_objects(executor):
        plan = getattr(candidate, "native_plan", None)
        if plan is not None and hasattr(plan, "input_ct_count") and hasattr(plan, "stripes"):
            return plan
    return None


def _executor_unified_groups(executor: Any) -> list[Any]:
    groups: list[Any] = []
    seen: set[int] = set()

    def append(group: Any) -> None:
        if group is None or id(group) in seen:
            return
        seen.add(id(group))
        groups.append(group)

    for candidate in _walk_executor_objects(executor):
        append(getattr(candidate, "group", None))
        for attr in (
            "groups",
            "groups_by_input_block",
            "groups_by_input_chunk",
            "groups_by_pair",
            "groups_by_input_index",
        ):
            value = getattr(candidate, attr, None)
            if isinstance(value, dict):
                for _key, group in sorted(value.items()):
                    append(group)
            else:
                for group in list(value or []):
                    append(group)
    return groups


def _sum_numeric_dicts(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                out[str(key)] = float(out.get(str(key), 0.0) + numeric)
    return out


def _runtime_profile(module: Any) -> dict[str, Any]:
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None)
    executor_timing = dict(getattr(executor, "last_runtime_timing", {}) or {})
    executor_io = dict(getattr(executor, "last_runtime_io", {}) or {})
    groups = _executor_unified_groups(executor)
    group_timings = [dict(getattr(group, "last_runtime_timing", {}) or {}) for group in groups]
    group_compile_profiles = [dict(getattr(group, "last_compile_profile", {}) or {}) for group in groups]
    metadata: dict[str, Any] = {}
    get_metadata = getattr(executor, "compile_cache_metadata", None)
    if callable(get_metadata):
        try:
            metadata = dict(get_metadata())
        except Exception as exc:
            metadata = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "executor_type": "" if executor is None else f"{type(executor).__module__}.{type(executor).__qualname__}",
        "executor_timing": executor_timing,
        "executor_io": executor_io,
        "unified_group_count": int(len(groups)),
        "unified_group_timing_sum": _sum_numeric_dicts(group_timings),
        "unified_group_compile_profile_sum": _sum_numeric_dicts(group_compile_profiles),
        "unified_group_compile_profiles": group_compile_profiles,
        "compile_cache_metadata": metadata,
    }


def _native_source_blocks_and_active_mask(x: torch.Tensor, plan: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from orion.experimental.cir.native_halo_conv2d import _idx_chw_gap

    spec = plan.spec
    values = x.detach().cpu().to(dtype=torch.float32)
    if values.dim() == 3:
        values = values.unsqueeze(0)
    if values.dim() != 4 or int(values.shape[0]) != 1:
        raise ValueError(f"native source block materialization expects NCHW batch-1 input, got {tuple(values.shape)}")
    src = values[0]
    slots = int(spec.slot_count)
    blocks: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for stripe in plan.effective_source_stripes:
        source_tile = int(plan.source_tile_for_stripe(stripe))
        for group in range(int(plan.source_group_count_for_stripe(stripe))):
            block = torch.zeros((int(slots),), dtype=torch.float32)
            mask = torch.zeros((int(slots),), dtype=torch.bool)
            channel_start = int(group) * int(source_tile)
            channel_end = min(int(spec.c_in), int(channel_start) + int(source_tile))
            for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
                for local_h in range(int(stripe.source_h)):
                    global_h = int(stripe.source_h_start) + int(local_h)
                    if int(global_h) < 0 or int(global_h) >= int(spec.h_in):
                        continue
                    for w_index in range(int(spec.w_in)):
                        slot = _idx_chw_gap(
                            int(local_channel),
                            int(local_h),
                            int(w_index),
                            int(stripe.source_h),
                            int(spec.w_in),
                            int(spec.gap_in),
                        )
                        block[int(slot)] = src[int(channel), int(global_h), int(w_index)]
                        mask[int(slot)] = True
            blocks.append(block)
            masks.append(mask)
    if len(blocks) != int(plan.input_ct_count):
        raise RuntimeError(f"expected {int(plan.input_ct_count)} source blocks, got {len(blocks)}")
    return torch.stack(blocks), torch.stack(masks)


def _serialise_native_stripes(stripes: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stripe in tuple(stripes or ()):
        to_dict = getattr(stripe, "to_dict", None)
        if callable(to_dict):
            try:
                rows.append({str(key): value for key, value in dict(to_dict()).items()})
                continue
            except Exception:
                pass
        row: dict[str, Any] = {}
        for key in (
            "index",
            "target_h_start",
            "target_h_end",
            "source_h_start",
            "source_h_end",
            "source_channel_tile",
            "target_channel_tile",
            "source_owner_h_start",
            "source_owner_h_end",
        ):
            if hasattr(stripe, key):
                try:
                    row[str(key)] = int(getattr(stripe, key))
                except Exception:
                    row[str(key)] = str(getattr(stripe, key))
        if row:
            rows.append(row)
    return rows


def _tensor_metric(reference: torch.Tensor, actual: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    ref = reference.detach().cpu().to(dtype=torch.float32)
    act = actual.detach().cpu().to(dtype=torch.float32)
    if mask is not None:
        m = mask.detach().cpu().to(dtype=torch.bool)
        ref = ref[m]
        act = act[m]
    delta = act - ref
    return {
        "count": int(delta.numel()),
        "mae": float(delta.abs().mean().item()) if int(delta.numel()) else 0.0,
        "max_abs": float(delta.abs().max().item()) if int(delta.numel()) else 0.0,
        "rmse": float(torch.sqrt(torch.mean(delta * delta)).item()) if int(delta.numel()) else 0.0,
        "reference_summary": _layer_mae_tensor_summary(ref) if int(ref.numel()) else {},
        "actual_summary": _layer_mae_tensor_summary(act) if int(act.numel()) else {},
    }


def _masked_value_summary(values: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    x = values.detach().cpu().to(dtype=torch.float32)
    if mask is not None:
        m = mask.detach().cpu().to(dtype=torch.bool)
        x = x[m]
    if int(x.numel()) <= 0:
        return {"count": 0}
    flat = x.reshape(-1)
    abs_flat = flat.abs()
    quantiles = torch.quantile(abs_flat, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float32))
    return {
        "count": int(flat.numel()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "rms": float(torch.sqrt(torch.mean(flat * flat)).item()),
        "abs_mean": float(abs_flat.mean().item()),
        "abs_max": float(abs_flat.max().item()),
        "abs_p50": float(quantiles[0].item()),
        "abs_p90": float(quantiles[1].item()),
        "abs_p95": float(quantiles[2].item()),
        "abs_p99": float(quantiles[3].item()),
    }


def _physical_slot_stats(
    expected_blocks: torch.Tensor,
    raw_blocks: torch.Tensor,
    active_mask: torch.Tensor,
    plan: Any,
) -> dict[str, Any]:
    active = active_mask.detach().cpu().to(dtype=torch.bool)
    inactive = ~active
    rows: list[dict[str, Any]] = []
    signatures = tuple(getattr(plan, "source_storage_signature", ()) or ())
    for index in range(int(raw_blocks.shape[0])):
        group_active = active[int(index)]
        group_inactive = inactive[int(index)]
        signature = []
        if int(index) < len(signatures):
            signature = [int(value) for value in tuple(signatures[int(index)])]
        rows.append(
            {
                "group": int(index),
                "signature": signature,
                "active": _masked_value_summary(raw_blocks[int(index)], group_active),
                "inactive": _masked_value_summary(raw_blocks[int(index)], group_inactive),
                "inactive_vs_zero": _tensor_metric(
                    torch.zeros_like(expected_blocks[int(index)]),
                    raw_blocks[int(index)],
                    group_inactive,
                ),
            }
        )
    active_abs_max = float(raw_blocks[active].abs().max().item()) if bool(active.any().item()) else 0.0
    inactive_abs_max = float(raw_blocks[inactive].abs().max().item()) if bool(inactive.any().item()) else 0.0
    return {
        "active": _masked_value_summary(raw_blocks, active),
        "inactive": _masked_value_summary(raw_blocks, inactive),
        "inactive_to_active_abs_max_ratio": (
            float(inactive_abs_max / active_abs_max) if float(active_abs_max) > 0.0 else None
        ),
        "groups": rows,
    }


def _cipher_meta(value: Any) -> dict[str, Any]:
    ids = [int(v) for v in tuple(getattr(value, "ids", ()) or ())]
    meta: dict[str, Any] = {
        "ct_count": int(len(ids)),
        "shape": _json_shape(getattr(value, "shape", ())),
        "on_shape": _json_shape(getattr(value, "on_shape", ())),
    }
    if ids:
        try:
            meta["level"] = int(value.level())
        except Exception:
            pass
        try:
            meta["scale"] = int(value.scale())
        except Exception:
            pass
        try:
            meta["log_scale"] = float(value.log_scale())
        except Exception:
            pass
    return meta


def _zero_inactive_native_slots_no_rescale(value: CipherTensor, active_mask: torch.Tensor) -> tuple[CipherTensor, dict[str, Any]]:
    ids = [int(v) for v in tuple(getattr(value, "ids", ()) or ())]
    if len(ids) != int(active_mask.shape[0]):
        raise RuntimeError(
            "inactive-slot mask block count mismatch: "
            f"cipher has {len(ids)} ids, mask has {int(active_mask.shape[0])}"
        )
    backend = value.backend
    mul_plain_new = getattr(backend, "MulPlaintextNew", None)
    if not callable(mul_plain_new):
        raise RuntimeError("backend does not expose MulPlaintextNew for no-rescale slot masking")
    level = int(value.level())
    before = _cipher_meta(value)
    out_ids: list[int] = []
    plaintext_ids: list[int] = []
    for block_index, ct_id in enumerate(ids):
        mask = active_mask[int(block_index)].detach().cpu().to(dtype=torch.float32)
        ptxt = scheme.encode(mask, int(level), scale=1)
        try:
            pt_id = int(ptxt.ids[0])
            plaintext_ids.append(int(pt_id))
            out_ids.append(int(mul_plain_new(int(ct_id), int(pt_id))))
        finally:
            ptxt.release()
    masked = CipherTensor(
        value.scheme,
        out_ids,
        value.shape,
        value.on_shape,
    )
    after = _cipher_meta(masked)
    return masked, {
        "kind": "native_slot_plaintext_mask_no_rescale",
        "mask_scale": 1,
        "plaintext_count": int(len(plaintext_ids)),
        "before": before,
        "after": after,
        "level_delta": (
            int(after["level"]) - int(before["level"])
            if "level" in before and "level" in after
            else None
        ),
        "log_scale_delta": (
            float(after["log_scale"]) - float(before["log_scale"])
            if "log_scale" in before and "log_scale" in after
            else None
        ),
    }


def _prefix_to_enc4b_act(model: torch.nn.Module, source: Any) -> Any:
    x = model.enc1a_act(model.enc1a(source))
    skip1 = model.enc1b_act(model.enc1b(x))
    x = model.enc2a_act(model.enc2a(model.pool1(skip1)))
    skip2 = model.enc2b_act(model.enc2b(x))
    x = model.enc3a_act(model.enc3a(model.pool2(skip2)))
    skip3 = model.enc3b_act(model.enc3b(x))
    x = model.enc4a_act(model.enc4a(model.pool3(skip3)))
    return model.enc4b_act(model.enc4b(x))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = out_dir / out_path
    dataset = str(args.dataset)
    checkpoint = Path(args.checkpoint or DEFAULT_CHECKPOINTS[dataset])
    replacements = Path(args.replacements or DEFAULT_REPLACEMENTS[dataset])
    payload: dict[str, Any] = {
        "status": "started",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "out": str(out_path),
        "backend_kind": str(args.backend_kind),
        "backend": _backend_name(str(args.backend_kind)),
        "compile_scope": str(args.compile_scope),
        "dataset": dataset,
        "checkpoint": str(checkpoint),
        "replacements": str(replacements),
    }
    _write(out_path, payload)
    try:
        image, _mask, sample_meta = _load_val_sample(
            dataset=dataset,
            data_root=Path(args.data_root),
            image_size=int(args.image_size),
            val_index=int(args.val_index),
            seed=int(args.seed),
        )
        model, model_meta = build_orion_cheb7_model_from_checkpoint(
            checkpoint,
            replacements_path=replacements,
            device="cpu",
        )
        model.eval()
        clear_pool_input, clear_pool_output = _capture_clear_io(model, image, "pool4")
        if str(args.compile_scope) == "encoder_pool4":
            compile_model = EncoderPool4Probe(model)
        else:
            compile_model = model
        compile_model.eval()
        payload.update(
            {
                "sample": dict(sample_meta),
                "model_meta": dict(model_meta),
                "clear_pool4_input": _layer_mae_tensor_summary(clear_pool_input),
                "clear_pool4_output": _layer_mae_tensor_summary(clear_pool_output),
                "status": "init_scheme",
            }
        )
        _write(out_path, payload)

        env = _apply_backend_env_compat(args)
        config = _backend_config(provider_mode=str(args.provider_mode), mode="provider")
        config["orion"]["backend"] = _backend_name(str(args.backend_kind))
        payload["env"] = dict(env)
        payload["config"] = config
        _write(out_path, payload)
        scheme.init_scheme(config)
        payload["status"] = "fit"
        _write(out_path, payload)
        scheme.fit(compile_model, image)
        payload["status"] = "compile"
        _write(out_path, payload)
        input_level = scheme.compile(compile_model)
        _name_bootstraps(compile_model)
        for module in compile_model.modules():
            if hasattr(module, "he_mode"):
                module.he_mode = True
        pool4_plan = _native_halo_source_plan(compile_model.pool4)
        if pool4_plan is None:
            raise RuntimeError("pool4 has no native halo source plan")
        expected_blocks, active_mask = _native_source_blocks_and_active_mask(clear_pool_input, pool4_plan)
        payload["status"] = "encrypt_input"
        payload["input_level"] = int(input_level)
        payload["pool4_native_plan"] = {
            "input_ct_count": int(getattr(pool4_plan, "input_ct_count", 0) or 0),
            "output_ct_count": int(getattr(pool4_plan, "output_ct_count", 0) or 0),
            "output_storage_layout": str((pool4_plan.to_dict()).get("output_storage_layout", "")),
            "target_internal_halo_overlap": int(getattr(pool4_plan, "target_internal_halo_overlap", 0) or 0),
            "independent_source_target_stripes": bool(
                getattr(pool4_plan, "independent_source_target_stripes", False)
            ),
            "source_storage_signature": [
                [int(value) for value in item]
                for item in tuple(getattr(pool4_plan, "source_storage_signature", ()) or ())
            ],
            "target_storage_signature": [
                [int(value) for value in item]
                for item in tuple(getattr(pool4_plan, "target_storage_signature", ()) or ())
            ],
            "stripes": _serialise_native_stripes(getattr(pool4_plan, "stripes", ()) or ()),
            "source_stripes": _serialise_native_stripes(getattr(pool4_plan, "source_stripes", ()) or ()),
            "target_stripes": _serialise_native_stripes(getattr(pool4_plan, "target_stripes", ()) or ()),
            "effective_source_stripes": _serialise_native_stripes(
                getattr(pool4_plan, "effective_source_stripes", ()) or ()
            ),
            "effective_target_stripes": _serialise_native_stripes(
                getattr(pool4_plan, "effective_target_stripes", ()) or ()
            ),
            "active_slot_count": int(active_mask.sum().item()),
            "inactive_slot_count": int((~active_mask).sum().item()),
        }
        payload["pool4_module_layout"] = {
            "layout_policy_output_materialization": str(
                getattr(compile_model.pool4, "layout_policy_output_materialization", "") or ""
            ),
            "layout_policy_input_physical_layout": str(
                getattr(compile_model.pool4, "layout_policy_input_physical_layout", "") or ""
            ),
            "native_halo_output_storage_layout": str(
                getattr(compile_model.pool4, "native_halo_output_storage_layout", "") or ""
            ),
            "layout_policy_native_output_target_signature": [
                [int(value) for value in item]
                for item in tuple(
                    getattr(compile_model.pool4, "layout_policy_native_output_target_signature", ()) or ()
                )
            ],
        }
        payload["bootstrap_report_after_compile"] = _collect_bootstrap_report(compile_model)
        if bool(args.compile_only):
            payload["status"] = "ok_compile_only"
            payload["elapsed_s"] = float(time.perf_counter() - started)
            _write(out_path, payload)
            return payload
        _write(out_path, payload)
        model_input_payload: dict[str, Any] = {}
        source = _encrypt_model_input(image, int(input_level), net=compile_model, payload=model_input_payload)
        payload["model_input_encoding"] = dict(model_input_payload.get("model_input_encoding", {}))
        payload["status"] = "prefix_forward"
        _write(out_path, payload)
        prefix_started = time.perf_counter()
        skip4 = _prefix_to_enc4b_act(compile_model, source)
        prefix_s = float(time.perf_counter() - prefix_started)
        payload["prefix_s"] = float(prefix_s)
        payload["skip4_cipher"] = {
            "shape": _json_shape(getattr(skip4, "shape", ())),
            "on_shape": _json_shape(getattr(skip4, "on_shape", ())),
            "ct_count": int(len(getattr(skip4, "ids", ()) or ())),
            "level": int(skip4.level()),
        }
        payload["status"] = "decode_skip4_raw"
        _write(out_path, payload)
        plain = skip4.decrypt()
        try:
            raw = plain.decode().detach().cpu().to(dtype=torch.float32)
        finally:
            release = getattr(plain, "release", None)
            if callable(release):
                release()
        raw_blocks = raw.reshape(tuple(expected_blocks.shape))
        payload["skip4_raw_vs_expected"] = {
            "all_slots": _tensor_metric(expected_blocks, raw_blocks),
            "active_slots": _tensor_metric(expected_blocks, raw_blocks, active_mask),
            "inactive_slots": _tensor_metric(expected_blocks, raw_blocks, ~active_mask),
        }
        payload["skip4_physical_slot_stats"] = _physical_slot_stats(
            expected_blocks,
            raw_blocks,
            active_mask,
            pool4_plan,
        )
        if bool(args.zero_skip4_inactive_slots):
            payload["status"] = "zero_skip4_inactive_slots"
            _write(out_path, payload)
            masked_skip4, mask_payload = _zero_inactive_native_slots_no_rescale(skip4, active_mask)
            payload["skip4_inactive_zero_mask"] = dict(mask_payload)
            if masked_skip4 is not skip4:
                release = getattr(skip4, "release", None)
                if callable(release):
                    release()
                skip4 = masked_skip4
            payload["skip4_cipher_after_inactive_zero"] = _cipher_meta(skip4)
        payload["status"] = "pool4_forward_from_skip4"
        _write(out_path, payload)
        pool_started = time.perf_counter()
        pool_out = compile_model.pool4(skip4)
        pool_s = float(time.perf_counter() - pool_started)
        decode_info: dict[str, Any] = {}
        decoded = _layer_mae_decode_cipher_output(
            compile_model.pool4,
            pool_out,
            module_name="pool4",
            decode_info=decode_info,
        )
        decoded = _layer_mae_tensor(decoded)
        reference = clear_pool_output.detach().cpu().to(dtype=torch.float32)
        payload["status"] = "ok"
        payload["pool4_s"] = float(pool_s)
        payload["pool4_decode_info"] = dict(decode_info)
        payload["pool4_metrics_vs_clear"] = _layer_mae_metric(reference, decoded)
        payload["pool4_runtime_profile"] = _runtime_profile(compile_model.pool4)
        payload["elapsed_s"] = float(time.perf_counter() - started)
        for value in (pool_out, skip4, source):
            release = getattr(value, "release", None)
            if callable(release):
                release()
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
    parser = argparse.ArgumentParser(description="Probe the exact enc4b_act -> pool4 native stripe handoff.")
    parser.add_argument("--dataset", choices=sorted(DEFAULT_CHECKPOINTS), default="covid19")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-index", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend-kind", choices=("clear", "ckks"), default="ckks")
    parser.add_argument("--compile-scope", choices=("full", "encoder_pool4"), default="full")
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Alias for --compile-scope encoder_pool4.",
    )
    parser.add_argument("--provider-mode", default="u22_256_base32_layout_dp_no_share_fold")
    parser.add_argument("--bootstrap-many", default="1")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--single-slot-layer-cache", action="store_true")
    parser.add_argument("--disable-cpp-diag-builder", action="store_true")
    parser.add_argument(
        "--zero-skip4-inactive-slots",
        action="store_true",
        help="Mask enc4b_act physical inactive native slots with a scale-1 plaintext before pool4.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--replacements", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("enc4b_pool4_handoff_probe.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.encoder_only):
        args.compile_scope = "encoder_pool4"
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "skip4_raw_active": (payload.get("skip4_raw_vs_expected") or {}).get("active_slots"),
                "pool4_native_plan": payload.get("pool4_native_plan"),
                "pool4_module_layout": payload.get("pool4_module_layout"),
                "pool4_metrics_vs_clear": payload.get("pool4_metrics_vs_clear"),
                "out": payload.get("out"),
                "error": payload.get("error"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if str(payload.get("status")) in {"ok", "ok_compile_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
