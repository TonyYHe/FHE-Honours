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
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.orion import scheme
from orion.models.unet import get_unet22_medical_model
from scripts.train_unet22_medical_seg import DATASETS, build_dataset, normalize_dataset_key, split_indices


U22_E2E_LOGQ = [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
U22_E2E_LOGP = [61, 61, 61]
U22_E2E_BOOT_LOGP = [61, 61, 61, 61, 61, 61, 61, 61]


def _config(
    *,
    dataset: str,
    backend: str,
    provider_mode: str,
    io_mode: str,
    io_dir: Path | None,
) -> dict[str, Any]:
    logn = 16
    config: dict[str, Any] = {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": list(U22_E2E_LOGQ),
            "LogP": list(U22_E2E_LOGP),
            "LogScale": 40,
            "H": 192,
            "RingType": "standard",
        },
        "boot_params": {
            "LogP": list(U22_E2E_BOOT_LOGP),
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": str(backend),
            "fuse_modules": True,
            "debug": False,
            "io_mode": str(io_mode),
            "experimental_region_first": str(provider_mode),
        },
    }
    if io_dir is not None:
        config["orion"]["diags_path"] = str(Path(io_dir) / "diagonals.h5")
        config["orion"]["keys_path"] = str(Path(io_dir) / "keys.h5")
    return config


def _provider_pressure_payload() -> dict[str, Any]:
    registry = getattr(scheme, "region_first_registry", None)
    if registry is None:
        return {"summary": {}, "regions": []}
    from orion.experimental.u22_phase1 import collect_layout_policy_provider_pressure

    return collect_layout_policy_provider_pressure(
        registry,
        backend=getattr(scheme, "backend", None),
        slots=int(scheme.params.get_slots()),
    )


def _default_checkpoint(dataset: str, *, base_dim: int) -> Path:
    return REPO_ROOT / "checkpoints" / "medical_unet22" / f"{dataset}_unet22_base{int(base_dim)}_best.pt"


def _as_image(tensor: torch.Tensor) -> Image.Image:
    values = tensor.detach().cpu().to(dtype=torch.float32)
    if values.ndim == 4:
        values = values[0]
    if values.ndim == 3 and int(values.shape[0]) == 1:
        array = values[0].clamp(0.0, 1.0).numpy()
        return Image.fromarray((array * 255.0).astype(np.uint8), mode="L").convert("RGB")
    if values.ndim == 3 and int(values.shape[0]) == 3:
        array = values.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")
    if values.ndim == 2:
        array = values.clamp(0.0, 1.0).numpy()
        return Image.fromarray((array * 255.0).astype(np.uint8), mode="L").convert("RGB")
    raise ValueError(f"cannot render tensor with shape {tuple(values.shape)}")


def _mask_image(logits_or_mask: torch.Tensor, *, from_logits: bool) -> Image.Image:
    values = logits_or_mask.detach().cpu().to(dtype=torch.float32)
    if bool(from_logits):
        values = torch.sigmoid(values)
    if values.ndim == 4:
        values = values[0]
    if values.ndim == 3:
        values = values[0]
    values = values.clamp(0.0, 1.0)
    return Image.fromarray((values.numpy() * 255.0).astype(np.uint8), mode="L").convert("RGB")


def _save_panel(path: Path, images: list[tuple[str, Image.Image]]) -> None:
    if not images:
        return
    width, height = images[0][1].size
    title_h = 28
    canvas = Image.new("RGB", (width * len(images), height + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(images):
        x = int(index * width)
        canvas.paste(image.resize((width, height)), (x, title_h))
        draw.text((x + 6, 7), str(title), fill=(0, 0, 0))
    canvas.save(path)


def _metrics(reference: torch.Tensor, predicted: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().cpu().to(dtype=torch.float32)
    pred = predicted.detach().cpu().to(dtype=torch.float32)
    return {
        "mae": float((ref - pred).abs().mean().item()),
        "max_abs": float((ref - pred).abs().max().item()),
    }


def _segmentation_metrics(logits_or_probs: torch.Tensor, mask: torch.Tensor, *, from_logits: bool) -> dict[str, float]:
    values = logits_or_probs.detach().cpu().to(dtype=torch.float32)
    target = mask.detach().cpu().to(dtype=torch.float32)
    if bool(from_logits):
        values = torch.sigmoid(values)
    pred = (values >= 0.5).to(dtype=torch.float32)
    if pred.shape != target.shape:
        pred = pred.reshape(target.shape)
    dims = tuple(range(1, pred.dim()))
    intersection = (pred * target).sum(dim=dims)
    pred_area = pred.sum(dim=dims)
    target_area = target.sum(dim=dims)
    union = pred_area + target_area - intersection
    eps = 1.0e-6
    dice = ((2.0 * intersection + eps) / (pred_area + target_area + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "pred_area": float(pred_area.mean().item()),
        "target_area": float(target_area.mean().item()),
    }


def _reset_he_operation_counters() -> bool:
    backend = getattr(scheme, "backend", None)
    reset = getattr(backend, "ResetOperationCounters", None)
    if not callable(reset):
        return False
    reset()
    return True


def _he_operation_counters() -> dict[str, int] | None:
    backend = getattr(scheme, "backend", None)
    get_counters = getattr(backend, "GetOperationCounters", None)
    if not callable(get_counters):
        return None
    values = [int(value) for value in list(get_counters())]
    if len(values) < 4:
        return None
    return {
        "rotation": int(values[0]),
        "lintrans_rotation": int(values[1]),
        "direct_rotation": int(values[2]),
        "conjugation": int(values[3]),
    }


def _load_sample(*, dataset: str, data_root: Path, sample_index: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    cfg = DATASETS[str(dataset)]
    ds = build_dataset(cfg, data_root, download=False, augment=None)
    _train_idx, val_idx = split_indices(len(ds), val_fraction=0.2, seed=int(seed))
    indices = val_idx or list(range(len(ds)))
    actual_index = int(indices[int(sample_index) % len(indices)])
    image, mask = ds[actual_index]
    return image.unsqueeze(0), mask.unsqueeze(0), actual_index


def _load_samples(
    *,
    dataset: str,
    data_root: Path,
    sample_index: int,
    sample_count: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    cfg = DATASETS[str(dataset)]
    ds = build_dataset(cfg, data_root, download=False, augment=None)
    _train_idx, val_idx = split_indices(len(ds), val_fraction=0.2, seed=int(seed))
    indices = val_idx or list(range(len(ds)))
    samples: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for offset in range(max(1, int(sample_count))):
        actual_index = int(indices[(int(sample_index) + int(offset)) % len(indices)])
        image, mask = ds[actual_index]
        samples.append((image.unsqueeze(0), mask.unsqueeze(0), actual_index))
    return samples


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = normalize_dataset_key(str(args.dataset))
    base_suffix = f"base{int(args.base_dim)}"
    provider_mode = str(
        args.provider_mode
        or (
            f"u22_64_{base_suffix}"
            if dataset == "montgomery_lung_64"
            else f"u22_256_{base_suffix}"
        )
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint is not None else _default_checkpoint(dataset, base_dim=int(args.base_dim))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / f"{dataset}_samples{int(args.sample_index)}_{int(args.sample_count)}_{str(args.mode)}_fhe_figure.json"

    payload: dict[str, Any] = {
        "status": "started",
        "dataset": str(dataset),
        "checkpoint": str(checkpoint),
        "provider_mode": str(provider_mode),
        "backend": str(args.backend),
        "mode": str(args.mode),
        "io_mode": str(args.io_mode),
        "io_dir": None if args.io_dir is None else str(Path(args.io_dir)),
        "sample_index": int(args.sample_index),
        "sample_count": int(args.sample_count),
        "bootstrap_margin_override": None if args.bootstrap_margin is None else float(args.bootstrap_margin),
        "out_dir": str(out_dir),
    }
    payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not checkpoint.exists():
        payload["status"] = "missing_checkpoint"
        payload["error"] = f"checkpoint not found: {checkpoint}"
        payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    samples = _load_samples(
        dataset=str(dataset),
        data_root=Path(args.data_root),
        sample_index=int(args.sample_index),
        sample_count=int(args.sample_count),
        seed=int(args.seed),
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = get_unet22_medical_model(dataset, base_dim=int(args.base_dim))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    clear_rows: list[dict[str, Any]] = []
    rendered_samples: list[dict[str, Any]] = []
    for image, mask, actual_index in samples:
        with torch.no_grad():
            clear = model(image)
        pred_img = _mask_image(clear, from_logits=True)
        mask_img = _mask_image(mask, from_logits=False)
        input_img = _as_image(image)
        input_path = out_dir / f"{dataset}_sample{actual_index}_input.png"
        mask_path = out_dir / f"{dataset}_sample{actual_index}_reference_mask.png"
        pred_path = out_dir / f"{dataset}_sample{actual_index}_pytorch_pred.png"
        input_img.save(input_path)
        mask_img.save(mask_path)
        pred_img.save(pred_path)
        panel_path = out_dir / f"{dataset}_sample{actual_index}_ref_vs_pytorch.png"
        _save_panel(panel_path, [("Input", input_img), ("Reference", mask_img), ("PyTorch", pred_img)])
        row = {
            "actual_dataset_index": int(actual_index),
            "pytorch_logits_shape": [int(v) for v in clear.shape],
            "pytorch_vs_reference": _segmentation_metrics(clear, mask, from_logits=True),
            "artifacts": {
                "input": str(input_path),
                "reference_mask": str(mask_path),
                "pytorch_pred": str(pred_path),
                "panel": str(panel_path),
            },
        }
        clear_rows.append(row)
        rendered_samples.append(
            {
                "image": image,
                "mask": mask,
                "clear": clear.detach().cpu(),
                "input_img": input_img,
                "mask_img": mask_img,
                "pred_img": pred_img,
                "row": row,
            }
        )

    payload.update(
        {
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_best_val_dice": ckpt.get("best_val_dice"),
            "samples": clear_rows,
        }
    )
    payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if bool(args.skip_fhe):
        payload["status"] = "ok_skip_fhe"
        payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    if args.io_dir is not None:
        Path(args.io_dir).mkdir(parents=True, exist_ok=True)

    config = _config(
        dataset=str(dataset),
        backend=str(args.backend),
        provider_mode=str(provider_mode) if args.mode == "provider" else "",
        io_mode=str(args.io_mode),
        io_dir=args.io_dir,
    )
    payload["config"] = config
    started = time.perf_counter()
    previous_bootstrap_margin = os.environ.get("ORION_BOOTSTRAP_MARGIN_OVERRIDE")
    if args.bootstrap_margin is not None:
        os.environ["ORION_BOOTSTRAP_MARGIN_OVERRIDE"] = str(float(args.bootstrap_margin))
    try:
        scheme.init_scheme(config)
        try:
            fit_image = rendered_samples[0]["image"]
            payload.update(
                {
                    "status": "fitting_ranges",
                    "timing_s": {"total_so_far": float(time.perf_counter() - started)},
                }
            )
            payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            scheme.fit(model, fit_image)
            payload.update(
                {
                    "status": "compiling_network",
                    "timing_s": {"total_so_far": float(time.perf_counter() - started)},
                }
            )
            payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            input_level = scheme.compile(model)
            payload.update(
                {
                    "status": "compiled_network",
                    "input_level": int(input_level),
                    "timing_s": {"total_so_far": float(time.perf_counter() - started)},
                }
            )
            payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            if bool(args.compile_only):
                payload.update(
                    {
                        "status": "ok_compile_only",
                        "provider_pressure": _provider_pressure_payload(),
                        "timing_s": {"total": float(time.perf_counter() - started)},
                    }
                )
                payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                return payload
            payload.update(
                {
                    "status": "materializing_he_model",
                    "timing_s": {"total_so_far": float(time.perf_counter() - started)},
                }
            )
            payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            model.he()
            payload.update(
                {
                    "status": "he_model_ready",
                    "timing_s": {"total_so_far": float(time.perf_counter() - started)},
                }
            )
            payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            total_forward_s = 0.0
            for sample in rendered_samples:
                image = sample["image"]
                mask = sample["mask"]
                clear = sample["clear"]
                actual_index = int(sample["row"]["actual_dataset_index"])
                payload.update(
                    {
                        "status": "encrypting_sample",
                        "current_sample_actual_index": int(actual_index),
                        "timing_s": {
                            "total_so_far": float(time.perf_counter() - started),
                            "he_forward_total_so_far": float(total_forward_s),
                        },
                    }
                )
                payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                x_ct = scheme.encrypt(scheme.encode(image, int(input_level)))
                payload.update(
                    {
                        "status": "forwarding_fhe_sample",
                        "current_sample_actual_index": int(actual_index),
                        "timing_s": {
                            "total_so_far": float(time.perf_counter() - started),
                            "he_forward_total_so_far": float(total_forward_s),
                        },
                    }
                )
                payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                forward_started = time.perf_counter()
                counters_enabled = _reset_he_operation_counters()
                out_ct = model(x_ct)
                forward_s = float(time.perf_counter() - forward_started)
                operation_counts = _he_operation_counters() if bool(counters_enabled) else None
                total_forward_s += float(forward_s)
                payload.update(
                    {
                        "status": "decrypting_sample",
                        "current_sample_actual_index": int(actual_index),
                        "timing_s": {
                            "total_so_far": float(time.perf_counter() - started),
                            "he_forward_total_so_far": float(total_forward_s),
                        },
                    }
                )
                payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                decoded = out_ct.decrypt().decode().detach().cpu()
                if torch.is_complex(decoded):
                    decoded = decoded.real
                decoded = decoded.to(dtype=torch.float32)
                fhe_img = _mask_image(decoded, from_logits=True)
                fhe_path = out_dir / f"{dataset}_sample{actual_index}_fhe_pred.png"
                fhe_img.save(fhe_path)
                panel_path = out_dir / f"{dataset}_sample{actual_index}_ref_vs_pytorch_vs_fhe.png"
                _save_panel(
                    panel_path,
                    [
                        ("Input", sample["input_img"]),
                        ("Reference", sample["mask_img"]),
                        ("PyTorch", sample["pred_img"]),
                        ("FHE", fhe_img),
                    ],
                )
                sample["row"]["fhe_decoded_shape"] = [int(v) for v in decoded.shape]
                sample["row"]["fhe_vs_pytorch_logits"] = _metrics(clear, decoded)
                sample["row"]["fhe_vs_reference"] = _segmentation_metrics(decoded, mask, from_logits=True)
                sample["row"]["timing_s"] = {"he_forward": float(forward_s)}
                if operation_counts is not None:
                    sample["row"]["he_operation_counts"] = operation_counts
                sample["row"]["artifacts"]["fhe_pred"] = str(fhe_path)
                sample["row"]["artifacts"]["panel"] = str(panel_path)
                for tensor in (out_ct, x_ct):
                    release = getattr(tensor, "release", None)
                    if callable(release):
                        release()
                payload.update(
                    {
                        "status": "running_fhe",
                        "input_level": int(input_level),
                        "timing_s": {
                            "total_so_far": float(time.perf_counter() - started),
                            "he_forward_total_so_far": float(total_forward_s),
                        },
                        "samples": [dict(item["row"]) for item in rendered_samples],
                    }
                )
                payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "event": "sample_done",
                            "dataset": str(dataset),
                            "actual_dataset_index": int(actual_index),
                            "he_forward_s": float(forward_s),
                            "fhe_vs_reference": sample["row"]["fhe_vs_reference"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            payload.update(
                {
                    "status": "ok",
                    "input_level": int(input_level),
                    "timing_s": {
                        "total": float(time.perf_counter() - started),
                        "he_forward_total": float(total_forward_s),
                    },
                    "provider_pressure": _provider_pressure_payload(),
                    "samples": [dict(sample["row"]) for sample in rendered_samples],
                }
            )
        finally:
            scheme.delete_scheme()
    finally:
        if args.bootstrap_margin is not None:
            if previous_bootstrap_margin is None:
                os.environ.pop("ORION_BOOTSTRAP_MARGIN_OVERRIDE", None)
            else:
                os.environ["ORION_BOOTSTRAP_MARGIN_OVERRIDE"] = previous_bootstrap_margin
    payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real UNet22 medical sample inference and save ref/PyTorch/FHE figure panels.")
    parser.add_argument("--dataset", default="montgomery_lung_64")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "medical")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / ".tmp" / "results" / "medical_fhe_figures_20260504")
    parser.add_argument("--base-dim", type=int, default=32)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("lattigo", "cheddar", "python"), default="lattigo")
    parser.add_argument("--mode", choices=("provider", "dense"), default="provider")
    parser.add_argument("--provider-mode", default=None)
    parser.add_argument("--io-mode", choices=("none", "save", "load"), default="none")
    parser.add_argument("--io-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-margin", type=float, default=None)
    parser.add_argument("--skip-fhe", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2))
    return 0 if str(payload.get("status", "")).startswith("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
