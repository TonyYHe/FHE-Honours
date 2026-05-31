#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_fhelipe_medseg_unet22 import (
    FhelipeSegmentationDataset,
    SPECS,
    batch_metrics,
    make_loader,
    segmentation_loss,
    set_seed,
)


class TrackingScaledSiLU(nn.Module):
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
            self.postscale.copy_(torch.ceil(absmax).clamp_min(1.0).to(self.postscale.device, dtype=self.postscale.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.track_scale_updates:
            self._update_scale(x)
        postscale = self.postscale.to(device=x.device, dtype=x.dtype)
        return postscale * F.silu(x / postscale)


class ScaledChebyshevSiLU(nn.Module):
    def __init__(self, coeffs: Iterable[float], *, postscale: float, trainable_coeffs: bool = False) -> None:
        super().__init__()
        coeff_tensor = torch.tensor(list(coeffs), dtype=torch.float32)
        if bool(trainable_coeffs):
            self.coeffs = nn.Parameter(coeff_tensor)
        else:
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
        for index in range(2, int(coeffs.numel())):
            t2 = 2.0 * z * t1 - t0
            out = out + coeffs[index] * t2
            t0, t1 = t1, t2
        return self.postscale * out


def fit_unit_chebyshev_silu(*, degree: int) -> list[float]:
    nodes = np.polynomial.chebyshev.chebpts1(int(degree) + 1)
    values = F.silu(torch.tensor(nodes, dtype=torch.float64)).numpy()
    poly = np.polynomial.Chebyshev.fit(nodes, values, int(degree))
    return [float(v) for v in poly.coef.tolist()]


def fit_scaled_domain_chebyshev_silu(*, degree: int, postscale: float) -> list[float]:
    scale = max(1.0, float(postscale))
    nodes = np.polynomial.chebyshev.chebpts1(int(degree) + 1)
    values = (F.silu(torch.tensor(nodes, dtype=torch.float64) * scale) / scale).numpy()
    poly = np.polynomial.Chebyshev.fit(nodes, values, int(degree))
    return [float(v) for v in poly.coef.tolist()]


def make_activation(kind: str, *, scale_margin: float) -> nn.Module:
    normalized = str(kind).strip().lower()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized in {"plain-silu", "plain_silu", "unscaled-silu", "unscaled_silu"}:
        return nn.SiLU(inplace=True)
    if normalized in {"scaled-silu", "scaled_silu", "silu"}:
        return TrackingScaledSiLU(margin=float(scale_margin))
    raise ValueError(f"unsupported activation {kind!r}")


def make_pool(kind: str) -> nn.Module:
    normalized = str(kind).strip().lower()
    if normalized == "max":
        return nn.MaxPool2d(kernel_size=2, stride=2)
    if normalized == "avg":
        return nn.AvgPool2d(kernel_size=2, stride=2)
    raise ValueError(f"unsupported pool {kind!r}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, activation: str, scale_margin: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(out_channels))
        self.act1 = make_activation(activation, scale_margin=scale_margin)
        self.conv2 = nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(out_channels))
        self.act2 = make_activation(activation, scale_margin=scale_margin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, activation: str, pool: str, scale_margin: float) -> None:
        super().__init__()
        self.pool = make_pool(pool)
        self.block = ConvBlock(in_channels, out_channels, activation=activation, scale_margin=scale_margin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, activation: str, scale_margin: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_channels), int(out_channels), kernel_size=2, stride=2, bias=False)
        self.up_bn = nn.BatchNorm2d(int(out_channels))
        self.up_act = make_activation(activation, scale_margin=scale_margin)
        self.block = ConvBlock(2 * int(out_channels), int(out_channels), activation=activation, scale_margin=scale_margin)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_act(self.up_bn(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class ConcatUNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        base_dim: int = 32,
        activation: str = "relu",
        pool: str = "max",
        scale_margin: float = 2.0,
    ) -> None:
        super().__init__()
        base = int(base_dim)
        self.config = {
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "base_dim": base,
            "activation": str(activation),
            "pool": str(pool),
            "skip": "concat",
        }
        self.inc = ConvBlock(int(in_channels), base, activation=activation, scale_margin=scale_margin)
        self.down1 = DownBlock(base, 2 * base, activation=activation, pool=pool, scale_margin=scale_margin)
        self.down2 = DownBlock(2 * base, 4 * base, activation=activation, pool=pool, scale_margin=scale_margin)
        self.down3 = DownBlock(4 * base, 8 * base, activation=activation, pool=pool, scale_margin=scale_margin)
        self.down4 = DownBlock(8 * base, 16 * base, activation=activation, pool=pool, scale_margin=scale_margin)
        self.up1 = UpBlock(16 * base, 8 * base, activation=activation, scale_margin=scale_margin)
        self.up2 = UpBlock(8 * base, 4 * base, activation=activation, scale_margin=scale_margin)
        self.up3 = UpBlock(4 * base, 2 * base, activation=activation, scale_margin=scale_margin)
        self.up4 = UpBlock(2 * base, base, activation=activation, scale_margin=scale_margin)
        self.outc = nn.Conv2d(base, int(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def init_model(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def set_scaled_silu_updates(model: nn.Module, *, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, TrackingScaledSiLU):
            module.track_scale_updates = bool(enabled)


def collect_scaled_silu_meta(model: nn.Module) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}
    for name, module in model.named_modules():
        if isinstance(module, TrackingScaledSiLU):
            postscale = float(module.postscale.detach().cpu().item())
            metadata[name] = {
                "observed_min": float(module.input_min.detach().cpu().item()),
                "observed_max": float(module.input_max.detach().cpu().item()),
                "postscale": postscale,
                "prescale": float(1.0 / postscale if postscale != 0.0 else 1.0),
                "margin": float(module.margin),
            }
    return metadata


def replace_scaled_silu_with_poly(
    model: nn.Module,
    *,
    degree: int,
    trainable_coeffs: bool = False,
) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}
    coeffs = fit_unit_chebyshev_silu(degree=int(degree))

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, TrackingScaledSiLU):
                postscale = float(child.postscale.detach().cpu().item())
                setattr(parent, child_name, ScaledChebyshevSiLU(coeffs, postscale=postscale, trainable_coeffs=bool(trainable_coeffs)))
                metadata[full_name] = {
                    "observed_min": float(child.input_min.detach().cpu().item()),
                    "observed_max": float(child.input_max.detach().cpu().item()),
                    "postscale": postscale,
                    "prescale": float(1.0 / postscale if postscale != 0.0 else 1.0),
                    "degree": int(degree),
                    "trainable_coeffs": bool(trainable_coeffs),
                }
            else:
                visit(child, full_name)

    visit(model)
    return metadata


@torch.no_grad()
def collect_plain_silu_input_ranges(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            x = inputs[0].detach()
            current_min = x.min()
            current_max = x.max()
            if not (bool(torch.isfinite(current_min)) and bool(torch.isfinite(current_max))):
                return
            row = ranges.setdefault(name, {"observed_min": math.inf, "observed_max": -math.inf})
            row["observed_min"] = min(float(row["observed_min"]), float(current_min.cpu().item()))
            row["observed_max"] = max(float(row["observed_max"]), float(current_max.cpu().item()))

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.SiLU):
            handles.append(module.register_forward_pre_hook(make_hook(name)))
    model.eval()
    for images, _masks in tqdm(loader, desc="calibrate", leave=False):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        _ = model(images)
    for handle in handles:
        handle.remove()
    return ranges


def replace_plain_silu_with_range_poly(
    model: nn.Module,
    *,
    ranges: dict[str, dict[str, float]],
    degree: int,
    scale_margin: float,
    trainable_coeffs: bool = False,
) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.SiLU):
                row = ranges.get(full_name, {})
                observed_min = float(row.get("observed_min", 0.0))
                observed_max = float(row.get("observed_max", 0.0))
                absmax = max(abs(observed_min), abs(observed_max)) * float(scale_margin)
                postscale = float(max(1.0, math.ceil(absmax)))
                coeffs = fit_scaled_domain_chebyshev_silu(degree=int(degree), postscale=postscale)
                setattr(parent, child_name, ScaledChebyshevSiLU(coeffs, postscale=postscale, trainable_coeffs=bool(trainable_coeffs)))
                metadata[full_name] = {
                    "observed_min": observed_min,
                    "observed_max": observed_max,
                    "postscale": postscale,
                    "prescale": float(1.0 / postscale if postscale != 0.0 else 1.0),
                    "degree": int(degree),
                    "fit": "silu_x_on_scaled_domain",
                    "trainable_coeffs": bool(trainable_coeffs),
                }
            else:
                visit(child, full_name)

    visit(model)
    return metadata


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
        total_loss += float(loss.detach().item()) * batch
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
                "logits_mae_vs_reference": total_logits_mae / max(1, total_items),
                "prob_mae_vs_reference": total_prob_mae / max(1, total_items),
                "pred_flip_rate_vs_reference": total_flip_rate / max(1, total_items),
            }
        )
    return out


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    checkpoint_path: Path,
    stage_name: str,
    freeze_scale_after_epoch: int = 0,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_metrics: dict[str, float] | None = None
    best_dice = -math.inf
    history: list[dict[str, object]] = []
    for epoch in range(1, int(epochs) + 1):
        if int(freeze_scale_after_epoch) > 0:
            set_scaled_silu_updates(model, enabled=int(epoch) <= int(freeze_scale_after_epoch))
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=float(grad_clip_norm),
        )
        val_metrics = run_epoch(model, val_loader, device=device, optimizer=None)
        row = {
            "stage": str(stage_name),
            "epoch": int(epoch),
            "train": train_metrics,
            "valid": val_metrics,
        }
        history.append(row)
        print("epoch", json.dumps(row, sort_keys=True), flush=True)
        if float(val_metrics["dice"]) >= best_dice:
            best_dice = float(val_metrics["dice"])
            best_metrics = val_metrics
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "stage": str(stage_name),
                    "epoch": int(epoch),
                    "valid": val_metrics,
                    "model": getattr(model, "config", {}),
                },
                checkpoint_path,
            )
    if best_metrics is None:
        best_metrics = evaluate_with_reference(model, val_loader, device=device)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "stage": str(stage_name),
                "epoch": 0,
                "valid": best_metrics,
                "model": getattr(model, "config", {}),
            },
            checkpoint_path,
        )
    return best_metrics, history


def load_conv_compatible_weights(dst: nn.Module, src_state: dict[str, torch.Tensor]) -> list[str]:
    dst_state = dst.state_dict()
    compatible = {
        key: value
        for key, value in src_state.items()
        if key in dst_state and tuple(dst_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = dst.load_state_dict(compatible, strict=False)
    return [f"missing:{key}" for key in missing] + [f"unexpected:{key}" for key in unexpected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain staged concat-skip U-Net baselines on FHELIPE medical NPZ datasets.")
    parser.add_argument("--dataset", choices=sorted(SPECS), default="covid19")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "fhelipe_medseg")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints" / "fhelipe_medseg_staged")
    parser.add_argument("--result", type=Path, default=ROOT / ".tmp" / "results" / "fhelipe_medseg_staged_unet.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--base-dim", type=int, default=32)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage1-pool", choices=("max", "avg"), default="max")
    parser.add_argument("--stage1-lr", type=float, default=3.0e-4)
    parser.add_argument("--stage2-lr", type=float, default=1.0e-4)
    parser.add_argument(
        "--stage2-init",
        choices=("stage1", "scratch"),
        default="stage1",
        help="Initialize scaled-SiLU + AvgPool from stage1 compatible weights or train it from scratch.",
    )
    parser.add_argument(
        "--stage2-activation",
        choices=("scaled-silu", "plain-silu"),
        default="scaled-silu",
        help="Activation used by the AvgPool stage2 model.",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scale-margin", type=float, default=2.0)
    parser.add_argument("--poly-degree", type=int, default=7)
    parser.add_argument("--poly-finetune-epochs", type=int, default=0)
    parser.add_argument("--poly-lr", type=float, default=1.0e-5)
    parser.add_argument(
        "--train-poly-coeffs",
        action="store_true",
        help="Make polynomial coefficients trainable during optional polynomial fine-tuning.",
    )
    parser.add_argument(
        "--eval-poly-only",
        action="store_true",
        help="Load existing stage checkpoints and only evaluate the requested polynomial degree.",
    )
    parser.add_argument(
        "--skip-poly",
        action="store_true",
        help="Only train/evaluate the ReLU and SiLU stages; skip Chebyshev polynomial replacement.",
    )
    parser.add_argument(
        "--freeze-scale-after-epoch",
        type=int,
        default=0,
        help="For stage2 scaled-SiLU training, stop updating per-activation scales after this epoch; 0 keeps tracking.",
    )
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

    stage1_pool = str(args.stage1_pool)
    tag = f"{spec.name}_concat_unet_base{int(args.base_dim)}_{int(args.image_size)}"
    stage1_name = f"relu_{stage1_pool}pool"
    stage2_init = str(args.stage2_init)
    stage2_activation = str(args.stage2_activation)
    stage2_prefix = "scaled_silu" if stage2_activation == "scaled-silu" else "silu"
    stage2_suffix = "finetune" if stage2_init == "stage1" else "retrain"
    stage2_name = f"{stage2_prefix}_avgpool_{stage2_suffix}"
    poly_stage_name = f"{stage2_prefix}_avgpool_degree_{int(args.poly_degree)}"
    stage1_path = Path(args.out_dir) / f"{tag}_{stage1_name}_best.pt"
    stage2_path = Path(args.out_dir) / f"{tag}_{stage2_name}_best.pt"
    started = time.time()

    stage1_model = ConcatUNet(
        in_channels=1,
        out_channels=1,
        base_dim=int(args.base_dim),
        activation="relu",
        pool=stage1_pool,
        scale_margin=float(args.scale_margin),
    ).to(device)
    if bool(args.eval_poly_only):
        if not stage1_path.exists():
            raise FileNotFoundError(f"missing stage1 checkpoint for eval-only mode: {stage1_path}")
        stage1_checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=False)
        stage1_model.load_state_dict(stage1_checkpoint["state_dict"])
        stage1_model.to(device)
        stage1_metrics = evaluate_with_reference(stage1_model, val_loader, device=device)
        stage1_history: list[dict[str, object]] = []
    else:
        init_model(stage1_model)
        stage1_metrics, stage1_history = train_stage(
            stage1_model,
            train_loader,
            val_loader,
            device=device,
            epochs=int(args.stage1_epochs),
            lr=float(args.stage1_lr),
            weight_decay=float(args.weight_decay),
            grad_clip_norm=float(args.grad_clip_norm),
            checkpoint_path=stage1_path,
            stage_name=stage1_name,
        )
        stage1_checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=False)
        stage1_model.load_state_dict(stage1_checkpoint["state_dict"])
        stage1_model.to(device)
        stage1_metrics = evaluate_with_reference(stage1_model, val_loader, device=device)

    stage2_model = ConcatUNet(
        in_channels=1,
        out_channels=1,
        base_dim=int(args.base_dim),
        activation=stage2_activation,
        pool="avg",
        scale_margin=float(args.scale_margin),
    ).to(device)
    if bool(args.eval_poly_only):
        if not stage2_path.exists():
            raise FileNotFoundError(f"missing stage2 checkpoint for eval-only mode: {stage2_path}")
        stage2_checkpoint = torch.load(stage2_path, map_location="cpu", weights_only=False)
        stage2_model.load_state_dict(stage2_checkpoint["state_dict"])
        stage2_model.to(device)
        stage2_history: list[dict[str, object]] = []
        incompatible = [
            "eval_poly_only",
            f"stage1_checkpoint:{stage1_path}",
            f"stage2_checkpoint:{stage2_path}",
        ]
    else:
        if stage2_init == "stage1":
            incompatible = load_conv_compatible_weights(stage2_model, stage1_checkpoint["state_dict"])
        else:
            init_model(stage2_model)
            incompatible = ["stage2_init:scratch"]
        stage2_metrics, stage2_history = train_stage(
            stage2_model,
            train_loader,
            val_loader,
            device=device,
            epochs=int(args.stage2_epochs),
            lr=float(args.stage2_lr),
            weight_decay=float(args.weight_decay),
            grad_clip_norm=float(args.grad_clip_norm),
            checkpoint_path=stage2_path,
            stage_name=stage2_name,
            freeze_scale_after_epoch=int(args.freeze_scale_after_epoch),
        )
        stage2_checkpoint = torch.load(stage2_path, map_location="cpu", weights_only=False)
        stage2_model.load_state_dict(stage2_checkpoint["state_dict"])
        stage2_model.to(device)
    stage2_metrics = evaluate_with_reference(stage2_model, val_loader, device=device, reference=stage1_model)

    skip_poly = bool(args.skip_poly)
    plain_silu_ranges: dict[str, dict[str, float]] = {}
    poly_meta: dict[str, dict[str, float]] = {}
    poly_metrics: dict[str, float] | None = None
    poly_history: list[dict[str, object]] = []
    poly_finetune_metrics: dict[str, float] | None = None
    poly_finetune_name = f"{poly_stage_name}_finetune"
    poly_finetune_path = Path(args.out_dir) / f"{tag}_{poly_finetune_name}_best.pt"
    if not skip_poly:
        poly_model = copy.deepcopy(stage2_model).to(device)
        if stage2_activation == "plain-silu":
            plain_silu_ranges = collect_plain_silu_input_ranges(stage2_model, train_loader, device=device)
            poly_meta = replace_plain_silu_with_range_poly(
                poly_model,
                ranges=plain_silu_ranges,
                degree=int(args.poly_degree),
                scale_margin=float(args.scale_margin),
                trainable_coeffs=bool(args.train_poly_coeffs),
            )
        else:
            poly_meta = replace_scaled_silu_with_poly(
                poly_model,
                degree=int(args.poly_degree),
                trainable_coeffs=bool(args.train_poly_coeffs),
            )
        poly_metrics = evaluate_with_reference(poly_model, val_loader, device=device, reference=stage2_model)
        if int(args.poly_finetune_epochs) > 0:
            poly_finetune_metrics, poly_history = train_stage(
                poly_model,
                train_loader,
                val_loader,
                device=device,
                epochs=int(args.poly_finetune_epochs),
                lr=float(args.poly_lr),
                weight_decay=float(args.weight_decay),
                grad_clip_norm=float(args.grad_clip_norm),
                checkpoint_path=poly_finetune_path,
                stage_name=poly_finetune_name,
            )
            poly_checkpoint = torch.load(poly_finetune_path, map_location="cpu", weights_only=False)
            poly_model.load_state_dict(poly_checkpoint["state_dict"])
            poly_model.to(device)
            poly_finetune_metrics = evaluate_with_reference(poly_model, val_loader, device=device, reference=stage2_model)

    drops = {
        f"{stage1_name}_to_{stage2_name}": {
            "dice_drop": float(stage1_metrics["dice"] - stage2_metrics["dice"]),
            "iou_drop": float(stage1_metrics["iou"] - stage2_metrics["iou"]),
        }
    }
    if poly_metrics is not None:
        drops.update(
            {
                f"{stage2_name}_to_degree_{int(args.poly_degree)}": {
                    "dice_drop": float(stage2_metrics["dice"] - poly_metrics["dice"]),
                    "iou_drop": float(stage2_metrics["iou"] - poly_metrics["iou"]),
                },
                f"{stage1_name}_to_degree_{int(args.poly_degree)}": {
                    "dice_drop": float(stage1_metrics["dice"] - poly_metrics["dice"]),
                    "iou_drop": float(stage1_metrics["iou"] - poly_metrics["iou"]),
                },
            }
        )
    if poly_finetune_metrics is not None:
        drops.update(
            {
                f"{stage2_name}_to_{poly_finetune_name}": {
                    "dice_drop": float(stage2_metrics["dice"] - poly_finetune_metrics["dice"]),
                    "iou_drop": float(stage2_metrics["iou"] - poly_finetune_metrics["iou"]),
                },
                f"{poly_stage_name}_to_{poly_finetune_name}": {
                    "dice_drop": float(poly_metrics["dice"] - poly_finetune_metrics["dice"]),
                    "iou_drop": float(poly_metrics["iou"] - poly_finetune_metrics["iou"]),
                },
                f"{stage1_name}_to_{poly_finetune_name}": {
                    "dice_drop": float(stage1_metrics["dice"] - poly_finetune_metrics["dice"]),
                    "iou_drop": float(stage1_metrics["iou"] - poly_finetune_metrics["iou"]),
                },
            }
        )
    stages = {
        stage1_name: stage1_metrics,
        stage2_name: stage2_metrics,
    }
    if poly_metrics is not None:
        stages[poly_stage_name] = poly_metrics
    if poly_finetune_metrics is not None:
        stages[poly_finetune_name] = poly_finetune_metrics
    result = {
        "dataset": asdict(spec),
        "npz_path": str(npz_path),
        "device": str(device),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "config": {
            "base_dim": int(args.base_dim),
            "image_size": int(args.image_size),
            "batch_size": int(args.batch_size),
            "train_limit": int(args.train_limit),
            "val_limit": int(args.val_limit),
            "stage1_epochs": int(args.stage1_epochs),
            "stage2_epochs": int(args.stage2_epochs),
            "stage1_lr": float(args.stage1_lr),
            "stage2_lr": float(args.stage2_lr),
            "stage2_init": stage2_init,
            "stage2_activation": stage2_activation,
            "stage1_pool": stage1_pool,
            "weight_decay": float(args.weight_decay),
            "grad_clip_norm": float(args.grad_clip_norm),
            "scale_margin": float(args.scale_margin),
            "poly_degree": int(args.poly_degree),
            "poly_finetune_epochs": int(args.poly_finetune_epochs),
            "poly_lr": float(args.poly_lr),
            "train_poly_coeffs": bool(args.train_poly_coeffs),
            "eval_poly_only": bool(args.eval_poly_only),
            "skip_poly": skip_poly,
            "freeze_scale_after_epoch": int(args.freeze_scale_after_epoch),
            "skip": "concat",
        },
        "checkpoints": {
            stage1_name: str(stage1_path),
            stage2_name: str(stage2_path),
            poly_finetune_name: str(poly_finetune_path) if poly_finetune_metrics is not None else "",
        },
        "stages": stages,
        "drops": drops,
        "stage1_history": stage1_history,
        "stage2_history": stage2_history,
        "poly_history": poly_history,
        "scaled_silu_replacements": collect_scaled_silu_meta(stage2_model),
        "plain_silu_fit_ranges": plain_silu_ranges,
        "degree_replacements": poly_meta,
        "load_notes": incompatible,
        "wall_s": float(time.time() - started),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "wrote", "result": str(args.result)}, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "stages": result["stages"],
                "drops": result["drops"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
