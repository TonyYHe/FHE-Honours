#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import orion.nn as on
from orion.models.ternaus import TernausVGGUNet
from orion.models.unet import UNet22


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    image_key: str
    label_key: str
    val_image_key: str
    val_label_key: str
    source_size: int


SPECS = {
    "covid19": DatasetSpec(
        name="covid19",
        filename="covid19radio_512.npz",
        image_key="train_images",
        label_key="train_label",
        val_image_key="val_images",
        val_label_key="val_label",
        source_size=512,
    ),
    "nusetmsb": DatasetSpec(
        name="nusetmsb",
        filename="nuset_512.npz",
        image_key="train_images",
        label_key="train_label",
        val_image_key="val_images",
        val_label_key="val_label",
        source_size=512,
    ),
}


class FhelipeSegmentationDataset(Dataset):
    def __init__(
        self,
        npz_path: Path,
        *,
        image_key: str,
        label_key: str,
        image_size: int,
        limit: int = 0,
        seed: int = 0,
    ) -> None:
        data = np.load(npz_path)
        images = data[image_key]
        labels = data[label_key]
        if images.shape != labels.shape:
            raise ValueError(f"image/label shape mismatch: {images.shape} vs {labels.shape}")
        if 0 < int(limit) < int(images.shape[0]):
            rng = np.random.default_rng(int(seed))
            indices = np.sort(rng.choice(int(images.shape[0]), size=int(limit), replace=False))
            images = images[indices]
            labels = labels[indices]
        self.images = images
        self.labels = labels
        self.image_size = int(image_size)
        self.image_scale = 1.0 if float(np.max(images)) <= 1.0 else 255.0
        self.label_threshold = 0.5 if float(np.max(labels)) <= 1.0 else 127.0

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(np.asarray(self.images[index], dtype=np.float32)).unsqueeze(0)
        label = torch.from_numpy(np.asarray(self.labels[index], dtype=np.float32)).unsqueeze(0)
        image = image * 2.0 - 1.0 if self.image_scale <= 1.0 else image / 127.5 - 1.0
        label = (label > self.label_threshold).to(dtype=torch.float32)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            label = F.interpolate(
                label.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0)
        return image.contiguous(), label.contiguous()


class TorchChebyshevSiLU(nn.Module):
    def __init__(self, coeffs: Iterable[float], *, low: float, high: float) -> None:
        super().__init__()
        coeff_tensor = torch.tensor(list(coeffs), dtype=torch.float32)
        self.register_buffer("coeffs", coeff_tensor)
        self.low = float(low)
        self.high = float(high)
        if self.low < -1.0 or self.high > 1.0:
            self.prescale = 2.0 / (self.high - self.low)
            self.constant = -self.prescale * (self.low + self.high) / 2.0
        else:
            self.prescale = 1.0
            self.constant = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        if self.prescale != 1.0:
            z = z * self.prescale
        if self.constant != 0.0:
            z = z + self.constant
        coeffs = self.coeffs.to(device=z.device, dtype=z.dtype)
        if coeffs.numel() == 0:
            return torch.zeros_like(z)
        out = coeffs[0].expand_as(z)
        if coeffs.numel() == 1:
            return out
        t0 = torch.ones_like(z)
        t1 = z
        out = out + coeffs[1] * t1
        for i in range(2, int(coeffs.numel())):
            t2 = 2.0 * z * t1 - t0
            out = out + coeffs[i] * t2
            t0, t1 = t1, t2
        return out


class ScaledExactSiLU(nn.Module):
    def __init__(self, postscale: float) -> None:
        super().__init__()
        self.postscale = float(postscale)
        self.prescale = 1.0 / self.postscale if self.postscale != 0.0 else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.postscale * F.silu(x * self.prescale)


class TrackingScaledExactSiLU(nn.Module):
    def __init__(self, *, margin: float = 2.0, initial_postscale: float = 1.0) -> None:
        super().__init__()
        self.margin = float(margin)
        self.track_scale_updates = True
        self.register_buffer("input_min", torch.zeros(()))
        self.register_buffer("input_max", torch.zeros(()))
        self.register_buffer("postscale", torch.tensor(float(initial_postscale), dtype=torch.float32))

    @property
    def prescale(self) -> float:
        postscale = float(self.postscale.detach().cpu().item())
        return 1.0 / postscale if postscale != 0.0 else 1.0

    def _update_scale(self, x: torch.Tensor) -> None:
        x_detached = x.detach()
        current_min = x_detached.min()
        current_max = x_detached.max()
        if bool(torch.isfinite(current_min)) and bool(torch.isfinite(current_max)):
            self.input_min.copy_(torch.minimum(self.input_min.to(current_min.device), current_min).to(self.input_min.device))
            self.input_max.copy_(torch.maximum(self.input_max.to(current_max.device), current_max).to(self.input_max.device))
            absmax = torch.maximum(self.input_min.abs(), self.input_max.abs()) * float(self.margin)
            postscale = torch.ceil(absmax).clamp_min(1.0)
            self.postscale.copy_(postscale.to(self.postscale.device, dtype=self.postscale.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.track_scale_updates:
            self._update_scale(x)
        postscale = self.postscale.to(device=x.device, dtype=x.dtype)
        return postscale * F.silu(x / postscale)


class ScaledChebyshevSiLU(nn.Module):
    def __init__(self, coeffs: Iterable[float], *, postscale: float) -> None:
        super().__init__()
        coeff_tensor = torch.tensor(list(coeffs), dtype=torch.float32)
        self.register_buffer("coeffs", coeff_tensor)
        self.postscale = float(postscale)
        self.prescale = 1.0 / self.postscale if self.postscale != 0.0 else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x * self.prescale
        coeffs = self.coeffs.to(device=z.device, dtype=z.dtype)
        if coeffs.numel() == 0:
            return torch.zeros_like(z)
        out = coeffs[0].expand_as(z)
        if coeffs.numel() == 1:
            return self.postscale * out
        t0 = torch.ones_like(z)
        t1 = z
        out = out + coeffs[1] * t1
        for i in range(2, int(coeffs.numel())):
            t2 = 2.0 * z * t1 - t0
            out = out + coeffs[i] * t2
            t0, t1 = t1, t2
        return self.postscale * out


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss_with_logits(logits, targets)


@torch.no_grad()
def batch_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0e-6) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).to(targets.dtype)
    dims = tuple(range(1, pred.dim()))
    intersection = (pred * targets).sum(dim=dims)
    pred_area = pred.sum(dim=dims)
    target_area = targets.sum(dim=dims)
    union = pred_area + target_area - intersection
    dice = ((2.0 * intersection + eps) / (pred_area + target_area + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return {"dice": float(dice.item()), "iou": float(iou.item())}


def make_model(*, architecture: str, base_dim: int, silu_degree: int) -> nn.Module:
    normalized = str(architecture).strip().lower().replace("_", "-")
    if normalized == "unet22":
        return UNet22(
            dataset="kvasir_polyp_256",
            in_channels=1,
            out_channels=1,
            base_dim=int(base_dim),
            activation="silu",
            silu_degree=int(silu_degree),
        )
    if normalized in {"ternaus-vgg-unet", "vgg-unet", "ternaus"}:
        return TernausVGGUNet(
            in_channels=1,
            out_channels=1,
            base_dim=int(base_dim),
            activation="silu",
            silu_degree=int(silu_degree),
        )
    raise ValueError(f"unsupported architecture {architecture!r}")


def replace_train_silu_modules_with_tracking_scale(model: nn.Module, *, margin: float) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, on.SiLU):
                setattr(parent, child_name, TrackingScaledExactSiLU(margin=float(margin)))
                metadata[full_name] = {"margin": float(margin), "initial_postscale": 1.0}
            else:
                visit(child, full_name)

    visit(model)
    return metadata


def set_tracking_scaled_silu_updates(model: nn.Module, *, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, TrackingScaledExactSiLU):
            module.track_scale_updates = bool(enabled)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_items = 0
    label = "train" if training else "valid"
    for images, masks in tqdm(loader, desc=label, leave=False):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = segmentation_loss(logits, masks)
        if training:
            loss.backward()
            if float(grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
        metrics = batch_metrics(logits.detach(), masks)
        batch = int(images.shape[0])
        total_loss += float(loss.detach().item()) * batch
        total_dice += float(metrics["dice"]) * batch
        total_iou += float(metrics["iou"]) * batch
        total_items += batch
    return {
        "loss": total_loss / max(1, total_items),
        "dice": total_dice / max(1, total_items),
        "iou": total_iou / max(1, total_items),
    }


@torch.no_grad()
def evaluate_with_reference(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    reference: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    if reference is not None:
        reference.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_logits_mae = 0.0
    total_prob_mae = 0.0
    total_flip_rate = 0.0
    total_items = 0
    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        metrics = batch_metrics(logits, masks)
        batch = int(images.shape[0])
        total_loss += float(loss.item()) * batch
        total_dice += float(metrics["dice"]) * batch
        total_iou += float(metrics["iou"]) * batch
        if reference is not None:
            ref_logits = reference(images)
            total_logits_mae += float((logits - ref_logits).abs().mean().item()) * batch
            probs = torch.sigmoid(logits)
            ref_probs = torch.sigmoid(ref_logits)
            total_prob_mae += float((probs - ref_probs).abs().mean().item()) * batch
            flips = ((probs >= 0.5) != (ref_probs >= 0.5)).to(dtype=torch.float32)
            total_flip_rate += float(flips.mean().item()) * batch
        total_items += batch
    out = {
        "loss": total_loss / max(1, total_items),
        "dice": total_dice / max(1, total_items),
        "iou": total_iou / max(1, total_items),
    }
    if reference is not None:
        out.update(
            {
                "logits_mae_vs_exact": total_logits_mae / max(1, total_items),
                "prob_mae_vs_exact": total_prob_mae / max(1, total_items),
                "pred_flip_rate_vs_exact": total_flip_rate / max(1, total_items),
            }
        )
    return out


def collect_silu_ranges(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, on.SiLU):
            stats[name] = {"min": math.inf, "max": -math.inf}

            def make_hook(module_name: str):
                def hook(_module, inputs):
                    x = inputs[0].detach()
                    stats[module_name]["min"] = min(stats[module_name]["min"], float(x.min().item()))
                    stats[module_name]["max"] = max(stats[module_name]["max"], float(x.max().item()))

                return hook

            hooks.append(module.register_forward_pre_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _masks) in enumerate(tqdm(loader, desc="calibrate", leave=False)):
            if int(max_batches) > 0 and batch_idx >= int(max_batches):
                break
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            model(images)
    for hook in hooks:
        hook.remove()
    return stats


def fit_chebyshev_silu(low: float, high: float, *, degree: int, margin: float = 2.0) -> tuple[list[float], float, float]:
    low = float(low)
    high = float(high)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"non-finite activation range low={low} high={high}")
    if abs(high - low) < 1.0e-7:
        low -= 1.0e-3
        high += 1.0e-3
    center = (low + high) / 2.0
    half_range = (high - low) / 2.0
    fit_low = center - float(margin) * half_range
    fit_high = center + float(margin) * half_range
    nodes = np.polynomial.chebyshev.chebpts1(int(degree) + 1)
    if fit_low < -1.0 or fit_high > 1.0:
        evals = (nodes + 1.0) * (fit_high - fit_low) / 2.0 + fit_low
    else:
        evals = nodes
    values = torch.nn.functional.silu(torch.tensor(evals, dtype=torch.float64)).numpy()
    coeffs = np.polynomial.chebyshev.chebfit(nodes, values, int(degree))
    return [float(v) for v in coeffs.tolist()], float(fit_low), float(fit_high)


def fit_unit_chebyshev_silu(*, degree: int) -> list[float]:
    nodes = np.polynomial.chebyshev.chebpts1(int(degree) + 1)
    values = torch.nn.functional.silu(torch.tensor(nodes, dtype=torch.float64)).numpy()
    coeffs = np.polynomial.chebyshev.chebfit(nodes, values, int(degree))
    return [float(v) for v in coeffs.tolist()]


def _orion_relu_like_scale(low: float, high: float, *, margin: float) -> float:
    absmax = max(abs(float(low)), abs(float(high))) * float(margin)
    if absmax <= 1.0:
        return 1.0
    return float(math.ceil(absmax))


def replace_silu_modules(
    model: nn.Module,
    ranges: dict[str, dict[str, float]],
    *,
    degree: int,
    margin: float,
) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, on.SiLU):
                if full_name not in ranges:
                    raise KeyError(f"missing calibrated range for {full_name}")
                coeffs, low, high = fit_chebyshev_silu(
                    ranges[full_name]["min"],
                    ranges[full_name]["max"],
                    degree=int(degree),
                    margin=float(margin),
                )
                setattr(parent, child_name, TorchChebyshevSiLU(coeffs, low=low, high=high))
                metadata[full_name] = {
                    "observed_min": float(ranges[full_name]["min"]),
                    "observed_max": float(ranges[full_name]["max"]),
                    "fit_low": float(low),
                    "fit_high": float(high),
                    "degree": int(degree),
                }
            else:
                visit(child, full_name)

    visit(model)
    return metadata


def replace_silu_modules_orion_scaled(
    model: nn.Module,
    ranges: dict[str, dict[str, float]],
    *,
    degree: int | None,
    margin: float,
) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}
    coeffs = None if degree is None else fit_unit_chebyshev_silu(degree=int(degree))

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, on.SiLU):
                if full_name not in ranges:
                    raise KeyError(f"missing calibrated range for {full_name}")
                observed_min = float(ranges[full_name]["min"])
                observed_max = float(ranges[full_name]["max"])
                postscale = _orion_relu_like_scale(observed_min, observed_max, margin=float(margin))
                if coeffs is None:
                    setattr(parent, child_name, ScaledExactSiLU(postscale=postscale))
                else:
                    setattr(parent, child_name, ScaledChebyshevSiLU(coeffs, postscale=postscale))
                metadata[full_name] = {
                    "observed_min": observed_min,
                    "observed_max": observed_max,
                    "postscale": float(postscale),
                    "prescale": float(1.0 / postscale),
                    "degree": None if degree is None else int(degree),
                }
            else:
                visit(child, full_name)

    visit(model)
    return metadata


def replace_tracking_scaled_silu_with_poly(
    model: nn.Module,
    *,
    degree: int,
) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}
    coeffs = fit_unit_chebyshev_silu(degree=int(degree))

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, TrackingScaledExactSiLU):
                postscale = float(child.postscale.detach().cpu().item())
                setattr(parent, child_name, ScaledChebyshevSiLU(coeffs, postscale=postscale))
                metadata[full_name] = {
                    "observed_min": float(child.input_min.detach().cpu().item()),
                    "observed_max": float(child.input_max.detach().cpu().item()),
                    "postscale": float(postscale),
                    "prescale": float(1.0 / postscale if postscale != 0.0 else 1.0),
                    "degree": int(degree),
                }
            else:
                visit(child, full_name)

    visit(model)
    return metadata


def collect_tracking_scaled_silu_meta(model: nn.Module) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}
    for name, module in model.named_modules():
        if isinstance(module, TrackingScaledExactSiLU):
            postscale = float(module.postscale.detach().cpu().item())
            metadata[name] = {
                "observed_min": float(module.input_min.detach().cpu().item()),
                "observed_max": float(module.input_max.detach().cpu().item()),
                "postscale": float(postscale),
                "prescale": float(1.0 / postscale if postscale != 0.0 else 1.0),
                "margin": float(module.margin),
            }
    return metadata


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Orion medical segmentation models on FHELIPE NPZ datasets.")
    parser.add_argument("--dataset", choices=sorted(SPECS), default="covid19")
    parser.add_argument(
        "--architecture",
        choices=("unet22", "ternaus-vgg-unet"),
        default="unet22",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "fhelipe_medseg")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints" / "fhelipe_medseg")
    parser.add_argument("--result", type=Path, default=ROOT / ".tmp" / "results" / "fhelipe_medseg_silu_degree_probe.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--base-dim", type=int, default=32)
    parser.add_argument("--train-silu-degree", type=int, default=31)
    parser.add_argument("--train-activation", choices=("exact-silu", "scaled-silu"), default="scaled-silu")
    parser.add_argument(
        "--freeze-scale-after-epoch",
        type=int,
        default=0,
        help="For scaled-SiLU training, stop updating per-activation scales after this epoch; 0 keeps tracking.",
    )
    parser.add_argument("--degrees", type=int, nargs="*", default=[7, 15])
    parser.add_argument("--cheb-margin", type=float, default=2.0)
    parser.add_argument("--scale-mode", choices=("orion-relu", "range-chebyshev"), default="orion-relu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-only", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    spec = SPECS[str(args.dataset)]
    npz_path = Path(args.data_root) / spec.filename
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")
    device = torch.device(args.device)
    train_set = FhelipeSegmentationDataset(
        npz_path,
        image_key=spec.image_key,
        label_key=spec.label_key,
        image_size=int(args.image_size),
        limit=int(args.train_limit),
        seed=int(args.seed),
    )
    val_set = FhelipeSegmentationDataset(
        npz_path,
        image_key=spec.val_image_key,
        label_key=spec.val_label_key,
        image_size=int(args.image_size),
        limit=int(args.val_limit),
        seed=int(args.seed) + 1,
    )
    train_loader = make_loader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    val_loader = make_loader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    architecture_tag = str(args.architecture).replace("-", "_")
    model = make_model(
        architecture=str(args.architecture),
        base_dim=int(args.base_dim),
        silu_degree=int(args.train_silu_degree),
    ).to(device)
    train_activation_meta = None
    if str(args.train_activation) == "scaled-silu":
        train_activation_meta = replace_train_silu_modules_with_tracking_scale(model, margin=float(args.cheb_margin))
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val_dice = -1.0
    epoch_metrics = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    activation_tag = str(args.train_activation).replace("-", "_")
    freeze_tag = ""
    if str(args.train_activation) == "scaled-silu":
        freeze_epoch = int(args.freeze_scale_after_epoch)
        freeze_tag = f"_freeze{freeze_epoch}" if freeze_epoch > 0 else "_tracking"
    ckpt_path = args.out_dir / f"{spec.name}_{architecture_tag}_base{int(args.base_dim)}_{activation_tag}{freeze_tag}_best.pt"
    started = time.time()
    if args.eval_only is None:
        for epoch in range(1, int(args.epochs) + 1):
            if str(args.train_activation) == "scaled-silu" and int(args.freeze_scale_after_epoch) > 0:
                set_tracking_scaled_silu_updates(
                    model,
                    enabled=int(epoch) <= int(args.freeze_scale_after_epoch),
                )
            train_metrics = run_epoch(
                model,
                train_loader,
                device=device,
                optimizer=optimizer,
                grad_clip_norm=float(args.grad_clip_norm),
            )
            val_metrics = run_epoch(model, val_loader, device=device, optimizer=None)
            row = {"epoch": int(epoch), "train": train_metrics, "valid": val_metrics}
            epoch_metrics.append(row)
            print("epoch", int(epoch), json.dumps(row, sort_keys=True), flush=True)
            if val_metrics["dice"] >= best_val_dice:
                best_val_dice = float(val_metrics["dice"])
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model": {
                            "architecture": str(args.architecture),
                            "dataset": spec.name,
                            "base_dim": int(args.base_dim),
                            "activation": "silu_exact_train",
                            "train_activation": str(args.train_activation),
                            "freeze_scale_after_epoch": int(args.freeze_scale_after_epoch),
                            "grad_clip_norm": float(args.grad_clip_norm),
                            "train_silu_degree_metadata": int(args.train_silu_degree),
                            "image_size": int(args.image_size),
                            "in_channels": 1,
                            "out_channels": 1,
                        },
                        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                        "epoch": int(epoch),
                        "valid": val_metrics,
                    },
                    ckpt_path,
                )
    else:
        ckpt_path = Path(args.eval_only)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    exact_metrics = evaluate_with_reference(model, val_loader, device=device, reference=None)
    ranges = {}
    if str(args.train_activation) == "exact-silu":
        ranges = collect_silu_ranges(
            model,
            train_loader,
            device=device,
            max_batches=int(args.calib_batches),
        )
    degree_results = {}
    scaled_exact_metrics = None
    scaled_exact_meta = None
    trained_scaled_meta = None
    if str(args.train_activation) == "scaled-silu":
        trained_scaled_meta = collect_tracking_scaled_silu_meta(model)
        for degree in [int(v) for v in args.degrees]:
            poly_model = copy.deepcopy(model).to(device)
            replace_meta = replace_tracking_scaled_silu_with_poly(poly_model, degree=int(degree))
            poly_metrics = evaluate_with_reference(poly_model, val_loader, device=device, reference=model)
            poly_metrics["delta_dice_vs_scaled_exact_train"] = float(poly_metrics["dice"] - exact_metrics["dice"])
            poly_metrics["delta_iou_vs_scaled_exact_train"] = float(poly_metrics["iou"] - exact_metrics["iou"])
            degree_results[str(degree)] = {
                "metrics": poly_metrics,
                "silu_replacements": replace_meta,
            }
            print("degree", degree, json.dumps(poly_metrics, sort_keys=True), flush=True)
    elif str(args.scale_mode) == "orion-relu":
        scaled_exact_model = copy.deepcopy(model).to(device)
        scaled_exact_meta = replace_silu_modules_orion_scaled(
            scaled_exact_model,
            ranges,
            degree=None,
            margin=float(args.cheb_margin),
        )
        scaled_exact_metrics = evaluate_with_reference(scaled_exact_model, val_loader, device=device, reference=model)
        scaled_exact_metrics["delta_dice_vs_exact_silu"] = float(scaled_exact_metrics["dice"] - exact_metrics["dice"])
        scaled_exact_metrics["delta_iou_vs_exact_silu"] = float(scaled_exact_metrics["iou"] - exact_metrics["iou"])
        print("scaled_exact", json.dumps(scaled_exact_metrics, sort_keys=True), flush=True)
        for degree in [int(v) for v in args.degrees]:
            poly_model = copy.deepcopy(model).to(device)
            replace_meta = replace_silu_modules_orion_scaled(
                poly_model,
                ranges,
                degree=int(degree),
                margin=float(args.cheb_margin),
            )
            poly_metrics = evaluate_with_reference(poly_model, val_loader, device=device, reference=scaled_exact_model)
            poly_metrics["delta_dice_vs_scaled_exact"] = float(poly_metrics["dice"] - scaled_exact_metrics["dice"])
            poly_metrics["delta_iou_vs_scaled_exact"] = float(poly_metrics["iou"] - scaled_exact_metrics["iou"])
            poly_metrics["delta_dice_vs_exact_silu"] = float(poly_metrics["dice"] - exact_metrics["dice"])
            poly_metrics["delta_iou_vs_exact_silu"] = float(poly_metrics["iou"] - exact_metrics["iou"])
            degree_results[str(degree)] = {
                "metrics": poly_metrics,
                "silu_replacements": replace_meta,
            }
            print("degree", degree, json.dumps(poly_metrics, sort_keys=True), flush=True)
    else:
        for degree in [int(v) for v in args.degrees]:
            poly_model = copy.deepcopy(model).to(device)
            replace_meta = replace_silu_modules(
                poly_model,
                ranges,
                degree=int(degree),
                margin=float(args.cheb_margin),
            )
            poly_metrics = evaluate_with_reference(poly_model, val_loader, device=device, reference=model)
            poly_metrics["delta_dice_vs_exact"] = float(poly_metrics["dice"] - exact_metrics["dice"])
            poly_metrics["delta_iou_vs_exact"] = float(poly_metrics["iou"] - exact_metrics["iou"])
            degree_results[str(degree)] = {
                "metrics": poly_metrics,
                "silu_replacements": replace_meta,
            }
            print("degree", degree, json.dumps(poly_metrics, sort_keys=True), flush=True)

    result = {
        "dataset": asdict(spec),
        "npz_path": str(npz_path),
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "config": {
            "base_dim": int(args.base_dim),
            "architecture": str(args.architecture),
            "image_size": int(args.image_size),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "train_limit": int(args.train_limit),
            "val_limit": int(args.val_limit),
            "calib_batches": int(args.calib_batches),
            "cheb_margin": float(args.cheb_margin),
            "scale_mode": str(args.scale_mode),
            "train_activation": str(args.train_activation),
            "freeze_scale_after_epoch": int(args.freeze_scale_after_epoch),
            "grad_clip_norm": float(args.grad_clip_norm),
            "lr": float(args.lr),
        },
        "epoch_metrics": epoch_metrics,
        "exact_eval": exact_metrics,
        "train_activation_replacements": train_activation_meta,
        "trained_scaled_replacements": trained_scaled_meta,
        "scaled_exact_eval": scaled_exact_metrics,
        "scaled_exact_replacements": scaled_exact_meta,
        "degree_eval": degree_results,
        "wall_s": float(time.time() - started),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "wrote", "result": str(args.result), "checkpoint": str(ckpt_path)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
