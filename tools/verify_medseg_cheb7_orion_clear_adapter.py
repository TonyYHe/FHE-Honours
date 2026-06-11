#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.orion import scheme

from tools.medseg_cheb7_orion_adapter import (
    ACTIVATION_NAMES,
    build_orion_cheb7_model_from_checkpoint,
    load_degree_replacements,
)
from tools.train_fhelipe_medseg_staged_unet import ScaledChebyshevSiLU, make_stage_model
from tools.train_fhelipe_medseg_unet22 import FhelipeSegmentationDataset, SPECS


DEFAULT_CHECKPOINTS = {
    "covid19": REPO_ROOT
    / "checkpoints"
    / "fhelipe_medseg_staged_covid19_256_scaled_silu_freeze15_cheb7_20260603"
    / "covid19_unet22_plus_output_base32_256_scaled_silu_avgpool_degree_7_rawgain_tight_g045_finetune_best.pt",
    "nusetmsb": REPO_ROOT
    / "checkpoints"
    / "fhelipe_medseg_staged_nusetmsb_384_scaled_silu_freeze15_cheb7_20260603"
    / "nusetmsb_unet22_plus_output_base32_384_scaled_silu_avgpool_degree_7_dec4a1536_rawgain_g045_finetune_best.pt",
}

DEFAULT_REPLACEMENTS = {
    "covid19": REPO_ROOT
    / "artifacts"
    / "haloed_paper_eval"
    / "fixtures"
    / "fhelipe_medseg_covid19_unet22_plus_output_256_base32_scaled_silu_freeze15_cheb7_finetune20_20260603.json",
    "nusetmsb": REPO_ROOT
    / "artifacts"
    / "haloed_paper_eval"
    / "fixtures"
    / "fhelipe_medseg_nusetmsb_unet22_plus_output_384_base32_scaled_silu_freeze15_cheb7_finetune20_20260603.json",
}

DEFAULT_IMAGE_SIZE = {"covid19": 256, "nusetmsb": 384}
DEFAULT_VAL_INDEX = {"covid19": 1937, "nusetmsb": 117}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Size):
        return [int(v) for v in value]
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _resolve_default(mapping: dict[str, Path], dataset: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(mapping[str(dataset)])


def _load_val_sample(
    *,
    dataset: str,
    data_root: Path,
    image_size: int,
    val_index: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    spec = SPECS[str(dataset)]
    npz_path = Path(data_root) / spec.filename
    ds = FhelipeSegmentationDataset(
        npz_path,
        image_key=spec.val_image_key,
        label_key=spec.val_label_key,
        image_size=int(image_size),
        limit=0,
        seed=int(seed),
    )
    if len(ds) <= 0:
        raise ValueError(f"empty validation dataset: {npz_path}")
    resolved_index = int(val_index) % int(len(ds))
    image, mask = ds[resolved_index]
    meta = {
        "npz_path": str(npz_path),
        "val_image_key": str(spec.val_image_key),
        "val_label_key": str(spec.val_label_key),
        "requested_val_index": int(val_index),
        "resolved_val_index": int(resolved_index),
        "validation_count": int(len(ds)),
        "image_size": int(image_size),
        "image_shape": [int(v) for v in image.shape],
        "mask_shape": [int(v) for v in mask.shape],
    }
    return image.unsqueeze(0).contiguous(), mask.unsqueeze(0).contiguous(), meta


def _state_dict(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state_dict is not a dict: {checkpoint_path}")
    return state, dict(checkpoint.get("model", {}) or {})


def _activation_postscale(name: str, state: dict[str, torch.Tensor], replacements: dict[str, dict[str, Any]]) -> float:
    log_key = f"{name}.log_postscale"
    tensor_key = f"{name}.postscale_tensor"
    if log_key in state:
        return float(torch.exp(state[log_key].detach().cpu().to(dtype=torch.float32)).item())
    if tensor_key in state:
        return float(state[tensor_key].detach().cpu().to(dtype=torch.float32).item())
    row = dict(replacements.get(name, {}) or {})
    return float(row.get("postscale", row.get("domain_postscale", 1.0)))


def _activation_prescale(
    name: str,
    state: dict[str, torch.Tensor],
    replacements: dict[str, dict[str, Any]],
    postscale: float,
) -> float:
    log_key = f"{name}.log_prescale"
    tensor_key = f"{name}.prescale_tensor"
    if log_key in state:
        return float(torch.exp(state[log_key].detach().cpu().to(dtype=torch.float32)).item())
    if tensor_key in state:
        return float(state[tensor_key].detach().cpu().to(dtype=torch.float32).item())
    row = dict(replacements.get(name, {}) or {})
    return float(row.get("prescale", 1.0 / postscale if postscale != 0.0 else 1.0))


def _replace_reference_activations(
    model: torch.nn.Module,
    *,
    state: dict[str, torch.Tensor],
    replacements: dict[str, dict[str, Any]],
    default_reference_activation: str,
) -> dict[str, dict[str, float]]:
    installed: dict[str, dict[str, float]] = {}
    for name in ACTIVATION_NAMES:
        coeff_key = f"{name}.coeffs"
        if coeff_key not in state:
            continue
        row = dict(replacements.get(name, {}) or {})
        coeffs = state[coeff_key].detach().cpu().flatten().to(dtype=torch.float32).tolist()
        degree = int(len(coeffs) - 1)
        postscale = _activation_postscale(name, state, replacements)
        log_key = f"{name}.log_postscale"
        prescale = _activation_prescale(name, state, replacements, postscale)
        prescale_log_key = f"{name}.log_prescale"
        prescale_tensor_key = f"{name}.prescale_tensor"
        reference_key = f"{name}.reference_postscale_tensor"
        reference_activation = str(row.get("reference_activation", default_reference_activation))
        reference_postscale = (
            float(state[reference_key].detach().cpu().to(dtype=torch.float32).item())
            if reference_key in state
            else float(row.get("reference_postscale", row.get("target_postscale", postscale)))
        )
        setattr(
            model,
            name,
            ScaledChebyshevSiLU(
                coeffs,
                postscale=float(postscale),
                prescale=float(prescale)
                if (prescale_log_key in state or prescale_tensor_key in state or "prescale" in row)
                else None,
                trainable_coeffs=False,
                trainable_scale=log_key in state,
                trainable_prescale=prescale_log_key in state,
                blend_alpha=1.0,
                reference_activation=reference_activation,
                reference_postscale=reference_postscale,
            ),
        )
        installed[name] = {
            "degree": float(degree),
            "postscale": float(postscale),
            "prescale": float(prescale),
            "domain_postscale": float(1.0 / prescale if prescale != 0.0 else float("inf")),
            "reference_postscale": float(reference_postscale),
        }
    return installed


def _build_training_reference_model(
    checkpoint_path: Path,
    *,
    replacements_path: Path,
    device: torch.device | str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    state, model_cfg = _state_dict(checkpoint_path)
    replacements = load_degree_replacements(replacements_path)
    checkpoint_activation = str(model_cfg.get("activation", "plain-silu")).replace("_", "-")
    reference_activation = "scaled-silu" if checkpoint_activation == "scaled-silu" else "plain-silu"
    model = make_stage_model(
        architecture=str(model_cfg.get("architecture", "unet22-plus-output")),
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_dim=int(model_cfg.get("base_dim", 32)),
        activation=checkpoint_activation,
        pool="avg",
        scale_margin=1.0,
        silu_degree=7,
    )
    installed = _replace_reference_activations(
        model,
        state=state,
        replacements=replacements,
        default_reference_activation=reference_activation,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    ignored_missing = set()
    for name in installed:
        ignored_missing.update(
            {
                f"{name}.postscale_tensor",
                f"{name}.prescale_tensor",
                f"{name}.blend_alpha",
                f"{name}.reference_postscale_tensor",
            }
        )
    remaining_missing = [key for key in missing if key not in ignored_missing]
    if unexpected or remaining_missing:
        raise RuntimeError(
            "checkpoint did not match training-side Cheb7 reference: "
            f"missing={remaining_missing[:8]} unexpected={list(unexpected)[:8]}"
        )
    model.to(device)
    model.eval()
    return model, {
        "checkpoint": str(checkpoint_path),
        "replacements_path": str(replacements_path),
        "installed_activations": installed,
    }


def _prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.detach().cpu().to(dtype=torch.float32))


def _mask_from_prob(prob: torch.Tensor, threshold: float) -> torch.Tensor:
    return (prob >= float(threshold)).to(dtype=torch.float32)


def _diff_metrics(reference: torch.Tensor, candidate: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    ref = reference.detach().cpu().to(dtype=torch.float32)
    cand = candidate.detach().cpu().to(dtype=torch.float32)
    shape_match = tuple(ref.shape) == tuple(cand.shape)
    if not shape_match and ref.numel() == cand.numel():
        cand = cand.reshape(ref.shape)
        shape_match = True
    delta = ref - cand
    return {
        "shape_match": bool(shape_match),
        "reference_shape": [int(v) for v in ref.shape],
        "candidate_shape": [int(v) for v in cand.shape],
        "mae": float(delta.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(delta * delta)).item()),
        "max_abs": float(delta.abs().max().item()),
        "allclose": bool(torch.allclose(ref, cand, atol=float(atol), rtol=float(rtol))),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _mask_diff_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().cpu().to(dtype=torch.float32)
    cand = candidate.detach().cpu().to(dtype=torch.float32)
    if ref.shape != cand.shape and ref.numel() == cand.numel():
        cand = cand.reshape(ref.shape)
    diff = (ref != cand).to(dtype=torch.float32)
    return {
        "shape_match": bool(tuple(ref.shape) == tuple(cand.shape)),
        "changed_pixels": int(diff.sum().item()),
        "pixel_count": int(diff.numel()),
        "changed_fraction": float(diff.mean().item()),
    }


def _segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, *, threshold: float, eps: float = 1.0e-6) -> dict[str, float]:
    pred = _mask_from_prob(_prob(logits), float(threshold)).to(dtype=torch.float32)
    tgt = target.detach().cpu().to(dtype=torch.float32)
    if pred.shape != tgt.shape and pred.numel() == tgt.numel():
        pred = pred.reshape(tgt.shape)
    dims = tuple(range(1, pred.dim()))
    intersection = (pred * tgt).sum(dim=dims)
    pred_area = pred.sum(dim=dims)
    target_area = tgt.sum(dim=dims)
    union = pred_area + target_area - intersection
    dice = ((2.0 * intersection + eps) / (pred_area + target_area + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "pred_area": float(pred_area.mean().item()),
        "target_area": float(target_area.mean().item()),
    }


def _segmentation_acceptance_metrics(payload: dict[str, Any], *, dice_atol: float | None) -> dict[str, Any]:
    outputs = dict(payload.get("outputs") or {})
    adapter = dict((outputs.get("adapter") or {}).get("segmentation") or {})
    backend = dict((outputs.get("backend") or {}).get("segmentation") or {})
    adapter_dice = adapter.get("dice")
    backend_dice = backend.get("dice")
    metrics: dict[str, Any] = {
        "adapter_dice": adapter_dice,
        "backend_dice": backend_dice,
        "dice_atol": dice_atol,
        "dice_close": False,
    }
    if adapter_dice is None or backend_dice is None or dice_atol is None:
        return metrics
    delta = abs(float(adapter_dice) - float(backend_dice))
    metrics["dice_abs_delta"] = float(delta)
    metrics["dice_close"] = bool(delta <= float(dice_atol))
    return metrics


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().cpu().to(dtype=torch.float32)
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def _save_arrays(
    path: Path,
    *,
    image: torch.Tensor,
    target: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    threshold: float,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "image": image.detach().cpu().numpy().astype(np.float32),
        "target": target.detach().cpu().numpy().astype(np.float32),
    }
    for name, logits in outputs.items():
        logits_cpu = logits.detach().cpu().to(dtype=torch.float32)
        prob_cpu = _prob(logits_cpu)
        mask_cpu = _mask_from_prob(prob_cpu, float(threshold))
        arrays[f"{name}_logits"] = logits_cpu.numpy().astype(np.float32)
        arrays[f"{name}_prob"] = prob_cpu.numpy().astype(np.float32)
        arrays[f"{name}_mask"] = mask_cpu.numpy().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _provider_mode(policy: str) -> str:
    from tools.run_u22_dim32_dense_provider_e2e_matrix import _provider_mode as resolve

    return resolve(str(policy))


def _apply_backend_env(
    *,
    backend_kind: str,
    mode: str,
    single_slot_layer_cache: bool,
    bootstrap_many: str | None = None,
) -> dict[str, str]:
    from tools.run_u22_dim32_dense_provider_e2e_matrix import (
        _apply_env_defaults,
        _apply_layer_cache_override,
        _apply_mode_env,
    )

    inherited_bootstrap_many = os.environ.get("ORION_LATTIGO_BOOTSTRAP_MANY")
    env = _apply_env_defaults(os.environ, backend="clear" if str(backend_kind) == "clear" else "ckks")
    env = _apply_mode_env(env, str(mode))
    env = _apply_layer_cache_override(env, single_slot_layer_cache=bool(single_slot_layer_cache))
    requested_bootstrap_many = bootstrap_many if bootstrap_many is not None else inherited_bootstrap_many
    if requested_bootstrap_many is not None:
        env["ORION_LATTIGO_BOOTSTRAP_MANY"] = str(requested_bootstrap_many)
    os.environ.update(env)
    return {key: str(os.environ.get(key, "")) for key in sorted(env)}


def _backend_config(*, provider_mode: str, mode: str) -> dict[str, Any]:
    from tools.run_lattigo_e2e_compare import _r18_config

    return _r18_config(str(provider_mode) if str(mode) == "provider" else "", backend="lattigo")


def _parse_layer_mae_targets(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def _run_clear_or_ckks_backend(
    *,
    checkpoint_path: Path,
    replacements_path: Path,
    image: torch.Tensor,
    expected_logits: torch.Tensor,
    args: argparse.Namespace,
    payload_path: Path,
    payload: dict[str, Any],
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    from tools.run_lattigo_e2e_compare import (
        _collect_bootstrap_report,
        _collect_layer_mae_polynomial_clear_outputs,
        _encrypt_model_input,
        _install_layer_mae_clear_capture,
        _install_layer_mae_he_capture,
        _layer_mae_adjust_clear_outputs_after_compile,
        _layer_mae_decode_model_output,
        _layer_mae_summary,
        _layer_mae_target_names,
        _name_bootstraps,
    )

    backend_started = time.perf_counter()
    backend_model, backend_meta = build_orion_cheb7_model_from_checkpoint(
        checkpoint_path,
        replacements_path=replacements_path,
        device="cpu",
    )
    provider_mode = str(args.provider_mode or (_provider_mode(str(args.policy)) if str(args.mode) == "provider" else ""))
    env_snapshot = _apply_backend_env(
        backend_kind=str(args.backend_kind),
        mode=str(args.mode),
        single_slot_layer_cache=bool(args.single_slot_layer_cache),
        bootstrap_many=args.backend_bootstrap_many,
    )
    config = _backend_config(provider_mode=provider_mode, mode=str(args.mode))
    payload["backend"] = {
        "status": "started",
        "kind": str(args.backend_kind),
        "mode": str(args.mode),
        "policy": str(args.policy),
        "provider_mode": str(provider_mode),
        "single_slot_layer_cache": bool(args.single_slot_layer_cache),
        "env": env_snapshot,
        "config": config,
        "model_meta": backend_meta,
    }
    _write_json(payload_path, payload)
    layer_mae_enabled = bool(args.backend_layer_mae)
    layer_mae_names: set[str] = set()
    layer_mae_clear_outputs: dict[str, torch.Tensor] | None = None
    layer_mae_reference_transforms: dict[str, dict[str, Any]] = {}
    remove_layer_mae_clear = None
    try:
        if layer_mae_enabled:
            layer_mae_names = set(_layer_mae_target_names(backend_model))
            requested_layer_mae_targets = _parse_layer_mae_targets(args.backend_layer_mae_targets)
            if requested_layer_mae_targets:
                layer_mae_names &= requested_layer_mae_targets
            layer_mae_clear_outputs, remove_layer_mae_clear = _install_layer_mae_clear_capture(
                backend_model,
                layer_mae_names,
            )
            payload["backend"]["layer_mae"] = {
                "enabled": True,
                "requested_targets": sorted(requested_layer_mae_targets),
                "requested_targets_missing_initial": sorted(requested_layer_mae_targets - layer_mae_names),
                "target_modules_initial": sorted(layer_mae_names),
            }
            _write_json(payload_path, payload)
            with torch.no_grad():
                _ = backend_model(image)
            payload["backend"]["layer_mae"]["clear_output_count"] = int(len(layer_mae_clear_outputs or {}))
            _write_json(payload_path, payload)
        if remove_layer_mae_clear is not None:
            remove_layer_mae_clear()
            remove_layer_mae_clear = None
        scheme.init_scheme(config)
        payload["backend"]["status"] = "fitting"
        payload["backend"]["timing_s"] = {"total_so_far": float(time.perf_counter() - backend_started)}
        _write_json(payload_path, payload)
        scheme.fit(backend_model, image)
        payload["backend"]["status"] = "compiling"
        payload["backend"]["timing_s"] = {"total_so_far": float(time.perf_counter() - backend_started)}
        _write_json(payload_path, payload)
        input_level = scheme.compile(backend_model)
        _name_bootstraps(backend_model)
        payload["backend"]["status"] = "compiled"
        payload["backend"]["input_level"] = int(input_level)
        payload["backend"]["bootstrap_report_after_compile"] = _collect_bootstrap_report(backend_model)
        payload["backend"]["timing_s"] = {"total_so_far": float(time.perf_counter() - backend_started)}
        if layer_mae_enabled:
            adjusted_outputs, reference_transforms, reference_transform_diagnostics = (
                _layer_mae_adjust_clear_outputs_after_compile(
                    backend_model,
                    layer_mae_clear_outputs,
                )
            )
            post_compile_layer_mae_names = set(_layer_mae_target_names(backend_model))
            requested_layer_mae_targets = _parse_layer_mae_targets(args.backend_layer_mae_targets)
            if requested_layer_mae_targets:
                post_compile_layer_mae_names &= requested_layer_mae_targets
            polynomial_outputs, polynomial_reference_diagnostics = _collect_layer_mae_polynomial_clear_outputs(
                backend_model,
                image,
                post_compile_layer_mae_names,
            )
            if str(polynomial_reference_diagnostics.get("status", "")) == "ok":
                adjusted_outputs.update(polynomial_outputs)
            layer_mae_clear_outputs = {
                str(name): value
                for name, value in dict(adjusted_outputs or {}).items()
                if str(name) in post_compile_layer_mae_names
            }
            layer_mae_reference_transforms = dict(reference_transforms)
            payload["backend"]["layer_mae"].update(
                {
                    "target_modules_after_compile": sorted(post_compile_layer_mae_names),
                    "requested_targets_missing_after_compile": sorted(
                        requested_layer_mae_targets - post_compile_layer_mae_names
                    ),
                    "target_modules_removed_after_compile": sorted(
                        str(name) for name in layer_mae_names - post_compile_layer_mae_names
                    ),
                    "reference_transform_count": int(len(layer_mae_reference_transforms)),
                    "reference_transforms": dict(layer_mae_reference_transforms),
                    "reference_transform_diagnostics": dict(reference_transform_diagnostics),
                    "polynomial_reference_diagnostics": dict(polynomial_reference_diagnostics),
                    "clear_output_count_after_reference_alignment": int(len(layer_mae_clear_outputs or {})),
                }
            )
        _write_json(payload_path, payload)
        if bool(args.compile_only):
            payload["backend"]["status"] = "ok_compile_only"
            payload["backend"]["timing_s"] = {"total": float(time.perf_counter() - backend_started)}
            return None, payload["backend"]
        backend_model.he()
        payload["backend"]["status"] = "encoding"
        _write_json(payload_path, payload)
        x_ct = _encrypt_model_input(image, int(input_level), net=backend_model, payload=payload["backend"])
        payload["backend"]["input_ciphertext_count"] = int(len(getattr(x_ct, "ids", ()) or ()))
        payload["backend"]["status"] = "forwarding"
        _write_json(payload_path, payload)
        layer_mae_rows = None
        remove_layer_mae = None
        if layer_mae_enabled:
            layer_mae_path = payload_path.with_name(f"{payload_path.stem}.forward0.layer_mae.jsonl")
            layer_mae_rows, remove_layer_mae = _install_layer_mae_he_capture(
                backend_model,
                clear_outputs=dict(layer_mae_clear_outputs or {}),
                reference_transforms=dict(layer_mae_reference_transforms or {}),
                names=set((layer_mae_clear_outputs or {}).keys()),
                jsonl_path=layer_mae_path,
            )
            payload["backend"]["layer_mae"]["jsonl_path"] = str(layer_mae_path)
            payload["backend"]["layer_mae"]["timing_note"] = (
                "layer MAE decrypts module outputs inside HE forward; this run is for diagnostics, not speed."
            )
            _write_json(payload_path, payload)
        forward_started = time.perf_counter()
        try:
            out_ct = backend_model(x_ct)
        finally:
            if remove_layer_mae is not None:
                remove_layer_mae()
                remove_layer_mae = None
        forward_s = float(time.perf_counter() - forward_started)
        payload["backend"]["bootstrap_report_after_forward"] = _collect_bootstrap_report(backend_model)
        if layer_mae_rows is not None:
            layer_mae_summary = _layer_mae_summary(layer_mae_rows, expected_names=set((layer_mae_clear_outputs or {}).keys()))
            payload["backend"]["layer_mae"].update(
                {
                    "summary": layer_mae_summary,
                    "rows": list(layer_mae_rows),
                    "overall_ok": bool(layer_mae_summary.get("overall_ok", False)),
                }
            )
        payload["backend"]["status"] = "decoding"
        payload["backend"]["timing_s"] = {
            "total_so_far": float(time.perf_counter() - backend_started),
            "he_forward": float(forward_s),
        }
        _write_json(payload_path, payload)
        decoded, decode_info = _layer_mae_decode_model_output(backend_model, out_ct)
        payload["backend"]["output_decode"] = decode_info
        _write_json(payload_path, payload)
        decoded = decoded.detach().cpu()
        if torch.is_complex(decoded):
            decoded = decoded.real
        decoded = decoded.to(dtype=torch.float32)
        if decoded.shape != expected_logits.shape and decoded.numel() == expected_logits.numel():
            decoded = decoded.reshape(expected_logits.shape)
        output_ciphertext_count = int(len(getattr(out_ct, "ids", ()) or ()))
        release = getattr(out_ct, "release", None)
        if callable(release):
            release()
        release = getattr(x_ct, "release", None)
        if callable(release):
            release()
        payload["backend"]["status"] = "ok"
        payload["backend"]["output_ciphertext_count"] = int(output_ciphertext_count)
        payload["backend"]["timing_s"] = {
            "total": float(time.perf_counter() - backend_started),
            "he_forward": float(forward_s),
        }
        return decoded, payload["backend"]
    finally:
        if remove_layer_mae_clear is not None:
            remove_layer_mae_clear()
        scheme.delete_scheme()


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = str(args.dataset)
    image_size = int(args.image_size or DEFAULT_IMAGE_SIZE[dataset])
    val_index = int(args.val_index if args.val_index is not None else DEFAULT_VAL_INDEX[dataset])
    checkpoint_path = _resolve_default(DEFAULT_CHECKPOINTS, dataset, args.checkpoint)
    replacements_path = _resolve_default(DEFAULT_REPLACEMENTS, dataset, args.replacements)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / f"{dataset}_{image_size}_val{val_index}_cheb7_orion_adapter_verify.json"
    arrays_path = out_dir / f"{dataset}_{image_size}_val{val_index}_cheb7_orion_adapter_verify.npz"

    payload: dict[str, Any] = {
        "status": "started",
        "dataset": dataset,
        "image_size": int(image_size),
        "val_index": int(val_index),
        "threshold": float(args.threshold),
        "checkpoint": str(checkpoint_path),
        "replacements": str(replacements_path),
        "out_dir": str(out_dir),
        "arrays_path": str(arrays_path),
    }
    _write_json(payload_path, payload)

    image, target, sample_meta = _load_val_sample(
        dataset=dataset,
        data_root=Path(args.data_root),
        image_size=int(image_size),
        val_index=int(val_index),
        seed=int(args.seed),
    )
    reference_model, reference_meta = _build_training_reference_model(
        checkpoint_path,
        replacements_path=replacements_path,
        device="cpu",
    )
    adapter_model, adapter_meta = build_orion_cheb7_model_from_checkpoint(
        checkpoint_path,
        replacements_path=replacements_path,
        device="cpu",
    )
    payload.update(
        {
            "sample": sample_meta,
            "reference_model": reference_meta,
            "adapter_model": adapter_meta,
            "status": "running_pytorch_alignment",
        }
    )
    _write_json(payload_path, payload)
    with torch.no_grad():
        reference_logits = reference_model(image)
        adapter_logits = adapter_model(image)
    reference_prob = _prob(reference_logits)
    adapter_prob = _prob(adapter_logits)
    reference_mask = _mask_from_prob(reference_prob, float(args.threshold))
    adapter_mask = _mask_from_prob(adapter_prob, float(args.threshold))

    outputs: dict[str, torch.Tensor] = {
        "reference": reference_logits,
        "adapter": adapter_logits,
    }
    payload["outputs"] = {
        "reference": {
            "logits_shape": [int(v) for v in reference_logits.shape],
            "logits_stats": _tensor_stats(reference_logits),
            "prob_stats": _tensor_stats(reference_prob),
            "segmentation": _segmentation_metrics(reference_logits, target, threshold=float(args.threshold)),
        },
        "adapter": {
            "logits_shape": [int(v) for v in adapter_logits.shape],
            "logits_stats": _tensor_stats(adapter_logits),
            "prob_stats": _tensor_stats(adapter_prob),
            "segmentation": _segmentation_metrics(adapter_logits, target, threshold=float(args.threshold)),
        },
    }
    payload["comparisons"] = {
        "reference_vs_adapter": {
            "logits": _diff_metrics(reference_logits, adapter_logits, atol=float(args.atol), rtol=float(args.rtol)),
            "prob": _diff_metrics(reference_prob, adapter_prob, atol=float(args.atol), rtol=float(args.rtol)),
            "mask": _mask_diff_metrics(reference_mask, adapter_mask),
        }
    }

    if bool(args.run_backend):
        payload["status"] = "running_backend"
        _write_json(payload_path, payload)
        backend_logits, backend_meta = _run_clear_or_ckks_backend(
            checkpoint_path=checkpoint_path,
            replacements_path=replacements_path,
            image=image,
            expected_logits=adapter_logits,
            args=args,
            payload_path=payload_path,
            payload=payload,
        )
        if backend_logits is not None:
            backend_prob = _prob(backend_logits)
            backend_mask = _mask_from_prob(backend_prob, float(args.threshold))
            outputs["backend"] = backend_logits
            payload["outputs"]["backend"] = {
                "logits_shape": [int(v) for v in backend_logits.shape],
                "logits_stats": _tensor_stats(backend_logits),
                "prob_stats": _tensor_stats(backend_prob),
                "segmentation": _segmentation_metrics(backend_logits, target, threshold=float(args.threshold)),
            }
            payload["comparisons"]["adapter_vs_backend"] = {
                "logits": _diff_metrics(adapter_logits, backend_logits, atol=float(args.backend_atol), rtol=float(args.backend_rtol)),
                "prob": _diff_metrics(adapter_prob, backend_prob, atol=float(args.backend_atol), rtol=float(args.backend_rtol)),
                "mask": _mask_diff_metrics(adapter_mask, backend_mask),
            }
        payload["backend"] = backend_meta

    _save_arrays(arrays_path, image=image, target=target, outputs=outputs, threshold=float(args.threshold))
    alignment_ok = bool(payload["comparisons"]["reference_vs_adapter"]["logits"]["allclose"]) and bool(
        payload["comparisons"]["reference_vs_adapter"]["prob"]["allclose"]
    )
    if bool(args.run_backend):
        if bool(args.compile_only):
            payload["status"] = "ok_compile_only" if alignment_ok else "mismatch"
        else:
            backend_allclose = bool(payload["comparisons"]["adapter_vs_backend"]["logits"]["allclose"]) and bool(
                payload["comparisons"]["adapter_vs_backend"]["prob"]["allclose"]
            )
            segmentation_acceptance = _segmentation_acceptance_metrics(
                payload,
                dice_atol=args.backend_dice_atol,
            )
            payload["comparisons"]["adapter_vs_backend"]["segmentation"] = segmentation_acceptance
            backend_dice_close = bool(segmentation_acceptance.get("dice_close", False))
            if alignment_ok and backend_allclose:
                payload["status"] = "ok"
            elif alignment_ok and backend_dice_close:
                payload["status"] = "ok_backend_dice_close"
            else:
                payload["status"] = "mismatch"
    else:
        payload["status"] = "ok_skip_backend" if alignment_ok else "mismatch"
    _write_json(payload_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify fixed-index medseg Cheb7 checkpoint alignment between training PyTorch, Orion adapter, and optional Orion backend."
    )
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--val-index", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--replacements", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "fhelipe_medseg")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / ".tmp" / "results" / "medseg_cheb7_orion_adapter_verify")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--run-backend", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--backend-kind", choices=("clear", "ckks"), default="clear")
    parser.add_argument("--backend-atol", type=float, default=1.0e-5)
    parser.add_argument("--backend-rtol", type=float, default=1.0e-4)
    parser.add_argument("--backend-dice-atol", type=float, default=None)
    parser.add_argument("--backend-bootstrap-many", choices=("0", "1"), default=None)
    parser.add_argument("--backend-layer-mae", action="store_true")
    parser.add_argument(
        "--backend-layer-mae-targets",
        default="",
        help="Comma-separated module names to capture for backend layer MAE; default captures every supported module.",
    )
    parser.add_argument("--mode", choices=("provider", "dense"), default="provider")
    parser.add_argument("--policy", default="dp_no_share_fold")
    parser.add_argument("--provider-mode", default=None)
    parser.add_argument("--single-slot-layer-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0 if str(payload.get("status", "")).startswith("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
