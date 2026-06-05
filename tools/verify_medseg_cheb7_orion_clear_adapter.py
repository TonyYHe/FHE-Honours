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
    / "covid19_unet22_plus_output_base32_256_scaled_silu_avgpool_degree_7_finetune_best.pt",
    "nusetmsb": REPO_ROOT
    / "checkpoints"
    / "fhelipe_medseg_staged_nusetmsb_384_scaled_silu_freeze15_cheb7_20260603"
    / "nusetmsb_unet22_plus_output_base32_384_scaled_silu_avgpool_degree_7_finetune_best.pt",
}

DEFAULT_REPLACEMENTS = {
    "covid19": REPO_ROOT
    / ".tmp"
    / "results"
    / "fhelipe_medseg_covid19_unet22_plus_output_256_base32_scaled_silu_freeze15_cheb7_finetune20_20260603.json",
    "nusetmsb": REPO_ROOT
    / ".tmp"
    / "results"
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
                trainable_coeffs=False,
                trainable_scale=log_key in state,
                blend_alpha=1.0,
                reference_activation=reference_activation,
                reference_postscale=reference_postscale,
            ),
        )
        installed[name] = {
            "degree": float(degree),
            "postscale": float(postscale),
            "prescale": float(1.0 / postscale if postscale != 0.0 else 1.0),
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


def _apply_backend_env(*, backend_kind: str, mode: str, single_slot_layer_cache: bool) -> dict[str, str]:
    from tools.run_u22_dim32_dense_provider_e2e_matrix import (
        _apply_env_defaults,
        _apply_layer_cache_override,
        _apply_mode_env,
    )

    env = _apply_env_defaults(os.environ, backend="clear" if str(backend_kind) == "clear" else "ckks")
    env = _apply_mode_env(env, str(mode))
    env = _apply_layer_cache_override(env, single_slot_layer_cache=bool(single_slot_layer_cache))
    os.environ.update(env)
    return {key: str(os.environ.get(key, "")) for key in sorted(env)}


def _backend_config(*, provider_mode: str, mode: str) -> dict[str, Any]:
    from tools.run_lattigo_e2e_compare import _r18_config

    return _r18_config(str(provider_mode) if str(mode) == "provider" else "", backend="lattigo")


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
    from tools.run_lattigo_e2e_compare import _encrypt_model_input, _layer_mae_decode_model_output

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
    try:
        scheme.init_scheme(config)
        payload["backend"]["status"] = "fitting"
        payload["backend"]["timing_s"] = {"total_so_far": float(time.perf_counter() - backend_started)}
        _write_json(payload_path, payload)
        scheme.fit(backend_model, image)
        payload["backend"]["status"] = "compiling"
        payload["backend"]["timing_s"] = {"total_so_far": float(time.perf_counter() - backend_started)}
        _write_json(payload_path, payload)
        input_level = scheme.compile(backend_model)
        payload["backend"]["status"] = "compiled"
        payload["backend"]["input_level"] = int(input_level)
        payload["backend"]["timing_s"] = {"total_so_far": float(time.perf_counter() - backend_started)}
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
        forward_started = time.perf_counter()
        out_ct = backend_model(x_ct)
        forward_s = float(time.perf_counter() - forward_started)
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
            backend_ok = bool(payload["comparisons"]["adapter_vs_backend"]["logits"]["allclose"]) and bool(
                payload["comparisons"]["adapter_vs_backend"]["prob"]["allclose"]
            )
            payload["status"] = "ok" if alignment_ok and backend_ok else "mismatch"
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
