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

import orion.nn as on
from orion.models.unet import UNet22PlusOutput
from orion.nn.linear import Conv2d as OrionConv2d
from orion.nn.linear import ConvTranspose2d as OrionConvTranspose2d
from orion.nn.pooling import AvgPool2d as OrionAvgPool2d
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
    def __init__(
        self,
        coeffs: Iterable[float],
        *,
        postscale: float,
        trainable_coeffs: bool = False,
        trainable_scale: bool = False,
        blend_alpha: float = 1.0,
        reference_activation: str = "plain-silu",
        reference_postscale: float | None = None,
    ) -> None:
        super().__init__()
        coeff_tensor = torch.tensor(list(coeffs), dtype=torch.float32)
        if bool(trainable_coeffs):
            self.coeffs = nn.Parameter(coeff_tensor)
        else:
            self.register_buffer("coeffs", coeff_tensor)
        self.postscale = float(postscale)
        self.prescale = 1.0 / self.postscale if self.postscale != 0.0 else 1.0
        if bool(trainable_scale):
            self.log_postscale = nn.Parameter(torch.log(torch.tensor(float(max(self.postscale, 1.0)), dtype=torch.float32)))
        else:
            self.register_buffer("postscale_tensor", torch.tensor(float(self.postscale), dtype=torch.float32))
        self.register_buffer("blend_alpha", torch.tensor(float(blend_alpha), dtype=torch.float32))
        self.reference_activation = str(reference_activation)
        reference_postscale_value = float(self.postscale if reference_postscale is None else reference_postscale)
        self.register_buffer("reference_postscale_tensor", torch.tensor(reference_postscale_value, dtype=torch.float32))

    def _reference(self, x: torch.Tensor, postscale: torch.Tensor) -> torch.Tensor:
        if self.reference_activation in {"scaled-silu", "scaled_silu"}:
            reference_postscale = self.reference_postscale_tensor.to(device=x.device, dtype=x.dtype)
            return reference_postscale * F.silu(x / reference_postscale)
        return F.silu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "log_postscale"):
            postscale = torch.exp(self.log_postscale).to(device=x.device, dtype=x.dtype).clamp_min(1.0e-6)
        else:
            postscale = self.postscale_tensor.to(device=x.device, dtype=x.dtype)
        z = x / postscale
        coeffs = self.coeffs.to(device=z.device, dtype=z.dtype)
        if coeffs.numel() == 0:
            cheb = torch.zeros_like(z)
            alpha = self.blend_alpha.to(device=x.device, dtype=x.dtype)
            return (1.0 - alpha) * self._reference(x, postscale) + alpha * (postscale * cheb)
        out = coeffs[0].expand_as(z)
        if coeffs.numel() == 1:
            cheb = postscale * out
            alpha = self.blend_alpha.to(device=x.device, dtype=x.dtype)
            return (1.0 - alpha) * self._reference(x, postscale) + alpha * cheb
        t0 = torch.ones_like(z)
        t1 = z
        out = out + coeffs[1] * t1
        for index in range(2, int(coeffs.numel())):
            t2 = 2.0 * z * t1 - t0
            out = out + coeffs[index] * t2
            t0, t1 = t1, t2
        cheb = postscale * out
        alpha = self.blend_alpha.to(device=x.device, dtype=x.dtype)
        return (1.0 - alpha) * self._reference(x, postscale) + alpha * cheb


def fit_unit_chebyshev_silu(*, degree: int) -> list[float]:
    nodes = np.polynomial.chebyshev.chebpts1(int(degree) + 1)
    values = F.silu(torch.tensor(nodes, dtype=torch.float64)).numpy()
    coeffs = np.polynomial.chebyshev.chebfit(nodes, values, int(degree))
    return [float(v) for v in coeffs.tolist()]


def fit_scaled_domain_chebyshev_silu(*, degree: int, postscale: float) -> list[float]:
    scale = max(1.0, float(postscale))
    nodes = np.polynomial.chebyshev.chebpts1(int(degree) + 1)
    values = (F.silu(torch.tensor(nodes, dtype=torch.float64) * scale) / scale).numpy()
    coeffs = np.polynomial.chebyshev.chebfit(nodes, values, int(degree))
    return [float(v) for v in coeffs.tolist()]


def fit_scaled_target_domain_chebyshev_silu(*, degree: int, domain_postscale: float, target_postscale: float) -> list[float]:
    domain_scale = max(1.0, float(domain_postscale))
    target_scale = max(1.0e-6, float(target_postscale))
    nodes = np.polynomial.chebyshev.chebpts1(int(degree) + 1)
    inputs = torch.tensor(nodes, dtype=torch.float64) * domain_scale
    values = (target_scale * F.silu(inputs / target_scale) / domain_scale).numpy()
    coeffs = np.polynomial.chebyshev.chebfit(nodes, values, int(degree))
    return [float(v) for v in coeffs.tolist()]


def parse_poly_degree_overrides(value: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    text = str(value or "").strip()
    if not text:
        return overrides
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid --poly-degree-overrides entry {item!r}; expected module=degree")
        name, degree_text = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"invalid --poly-degree-overrides entry {item!r}; empty module name")
        degree = int(degree_text.strip())
        if degree < 0:
            raise ValueError(f"invalid polynomial degree for {name!r}: {degree}")
        overrides[name] = degree
    return overrides


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


def replace_silu_with_tracking_scaled(model: nn.Module, *, scale_margin: float) -> None:
    for child_name, child in list(model.named_children()):
        if isinstance(child, (nn.SiLU, on.SiLU)):
            setattr(model, child_name, TrackingScaledSiLU(margin=float(scale_margin)))
        else:
            replace_silu_with_tracking_scaled(child, scale_margin=float(scale_margin))


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


def make_stage_model(
    *,
    architecture: str,
    in_channels: int,
    out_channels: int,
    base_dim: int,
    activation: str,
    pool: str,
    scale_margin: float,
    silu_degree: int,
) -> nn.Module:
    normalized = str(architecture).strip().lower().replace("_", "-")
    if normalized == "concat-unet":
        return ConcatUNet(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            base_dim=int(base_dim),
            activation=str(activation),
            pool=str(pool),
            scale_margin=float(scale_margin),
        )
    if normalized in {"unet22-plus-output", "u22-plus-output", "u23"}:
        orion_activation = "silu" if str(activation) in {"plain-silu", "plain_silu", "scaled-silu", "scaled_silu", "silu"} else str(activation)
        model = UNet22PlusOutput(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            base_channels=int(base_dim),
            activation=orion_activation,
            silu_degree=int(silu_degree),
        )
        if str(activation) in {"scaled-silu", "scaled_silu"}:
            replace_silu_with_tracking_scaled(model, scale_margin=float(scale_margin))
        model.config = {
            "architecture": "unet22-plus-output",
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "base_dim": int(base_dim),
            "activation": str(activation),
            "silu_degree": int(silu_degree),
            "pool": "avg",
            "skip": "concat",
            "output_head": "1x1",
        }
        return model
    raise ValueError(f"unsupported architecture {architecture!r}")


def init_model(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (OrionConv2d, OrionConvTranspose2d)) and not isinstance(module, OrionAvgPool2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def set_scaled_silu_updates(model: nn.Module, *, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, TrackingScaledSiLU):
            module.track_scale_updates = bool(enabled)
        if isinstance(module, ScaledChebyshevSiLU) and hasattr(module, "log_postscale"):
            module.log_postscale.requires_grad_(bool(enabled))


def cheb_scale_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, ScaledChebyshevSiLU) and hasattr(module, "log_postscale"):
            params.append(module.log_postscale)
    return params


def set_cheb_scale_only_updates(model: nn.Module, *, enabled: bool) -> dict[int, bool]:
    previous = {id(param): bool(param.requires_grad) for param in model.parameters()}
    scale_ids = {id(param) for param in cheb_scale_parameters(model)}
    for param in model.parameters():
        param.requires_grad_(bool(enabled) and id(param) in scale_ids)
    return previous


def restore_requires_grad(model: nn.Module, previous: dict[int, bool]) -> None:
    for param in model.parameters():
        param.requires_grad_(bool(previous.get(id(param), param.requires_grad)))


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
    degree_overrides: dict[str, int] | None = None,
    trainable_coeffs: bool = False,
    trainable_scale: bool = False,
) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}
    overrides = dict(degree_overrides or {})
    used_overrides: set[str] = set()
    coeff_cache: dict[int, list[float]] = {}

    def coeffs_for(module_name: str) -> tuple[int, list[float]]:
        module_degree = int(overrides.get(module_name, int(degree)))
        used_overrides.add(module_name) if module_name in overrides else None
        if module_degree not in coeff_cache:
            coeff_cache[module_degree] = fit_unit_chebyshev_silu(degree=module_degree)
        return module_degree, coeff_cache[module_degree]

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, TrackingScaledSiLU):
                postscale = float(child.postscale.detach().cpu().item())
                module_degree, coeffs = coeffs_for(full_name)
                setattr(
                    parent,
                    child_name,
                    ScaledChebyshevSiLU(
                        coeffs,
                        postscale=postscale,
                        trainable_coeffs=bool(trainable_coeffs),
                        trainable_scale=bool(trainable_scale),
                        reference_activation="scaled-silu",
                    ),
                )
                metadata[full_name] = {
                    "observed_min": float(child.input_min.detach().cpu().item()),
                    "observed_max": float(child.input_max.detach().cpu().item()),
                    "postscale": postscale,
                    "prescale": float(1.0 / postscale if postscale != 0.0 else 1.0),
                    "degree": int(module_degree),
                    "trainable_coeffs": bool(trainable_coeffs),
                    "trainable_scale": bool(trainable_scale),
                    "reference_activation": "scaled-silu",
                }
            else:
                visit(child, full_name)

    visit(model)
    unused = sorted(set(overrides) - used_overrides)
    if unused:
        raise ValueError(f"--poly-degree-overrides did not match activation modules: {', '.join(unused)}")
    return metadata


def replace_scaled_silu_with_range_poly(
    model: nn.Module,
    *,
    ranges: dict[str, dict[str, float]],
    degree: int,
    degree_overrides: dict[str, int] | None = None,
    scale_margin: float,
    trainable_coeffs: bool = False,
    trainable_scale: bool = False,
) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}
    overrides = dict(degree_overrides or {})
    used_overrides: set[str] = set()
    coeff_cache: dict[tuple[int, float, float], list[float]] = {}

    def coeffs_for(module_name: str, domain_postscale: float, target_postscale: float) -> tuple[int, list[float]]:
        module_degree = int(overrides.get(module_name, int(degree)))
        used_overrides.add(module_name) if module_name in overrides else None
        key = (module_degree, float(domain_postscale), float(target_postscale))
        if key not in coeff_cache:
            coeff_cache[key] = fit_scaled_target_domain_chebyshev_silu(
                degree=module_degree,
                domain_postscale=domain_postscale,
                target_postscale=target_postscale,
            )
        return module_degree, coeff_cache[key]

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, TrackingScaledSiLU):
                target_postscale = float(child.postscale.detach().cpu().item())
                row = ranges.get(full_name, {})
                observed_min = float(row.get("observed_min", child.input_min.detach().cpu().item()))
                observed_max = float(row.get("observed_max", child.input_max.detach().cpu().item()))
                absmax = max(abs(observed_min), abs(observed_max)) * float(scale_margin)
                domain_postscale = float(max(1.0, math.ceil(absmax)))
                module_degree, coeffs = coeffs_for(full_name, domain_postscale, target_postscale)
                setattr(
                    parent,
                    child_name,
                    ScaledChebyshevSiLU(
                        coeffs,
                        postscale=domain_postscale,
                        trainable_coeffs=bool(trainable_coeffs),
                        trainable_scale=bool(trainable_scale),
                        reference_activation="scaled-silu",
                        reference_postscale=target_postscale,
                    ),
                )
                metadata[full_name] = {
                    "observed_min": observed_min,
                    "observed_max": observed_max,
                    "target_postscale": target_postscale,
                    "domain_postscale": domain_postscale,
                    "domain_prescale": float(1.0 / domain_postscale if domain_postscale != 0.0 else 1.0),
                    "degree": int(module_degree),
                    "fit": "scaled_silu_target_on_range_domain",
                    "trainable_coeffs": bool(trainable_coeffs),
                    "trainable_scale": bool(trainable_scale),
                    "reference_activation": "scaled-silu",
                }
            else:
                visit(child, full_name)

    visit(model)
    unused = sorted(set(overrides) - used_overrides)
    if unused:
        raise ValueError(f"--poly-degree-overrides did not match activation modules: {', '.join(unused)}")
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
        if isinstance(module, (nn.SiLU, on.SiLU)):
            handles.append(module.register_forward_pre_hook(make_hook(name)))
    model.eval()
    for images, _masks in tqdm(loader, desc="calibrate", leave=False):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        _ = model(images)
    for handle in handles:
        handle.remove()
    return ranges


@torch.no_grad()
def collect_tracking_scaled_silu_input_ranges(
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
        if isinstance(module, TrackingScaledSiLU):
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
    degree_overrides: dict[str, int] | None = None,
    scale_margin: float,
    trainable_coeffs: bool = False,
    trainable_scale: bool = False,
) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}
    overrides = dict(degree_overrides or {})
    used_overrides: set[str] = set()
    coeff_cache: dict[tuple[int, float], list[float]] = {}

    def coeffs_for(module_name: str, postscale: float) -> tuple[int, list[float]]:
        module_degree = int(overrides.get(module_name, int(degree)))
        used_overrides.add(module_name) if module_name in overrides else None
        key = (module_degree, float(postscale))
        if key not in coeff_cache:
            coeff_cache[key] = fit_scaled_domain_chebyshev_silu(degree=module_degree, postscale=postscale)
        return module_degree, coeff_cache[key]

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (nn.SiLU, on.SiLU)):
                row = ranges.get(full_name, {})
                observed_min = float(row.get("observed_min", 0.0))
                observed_max = float(row.get("observed_max", 0.0))
                absmax = max(abs(observed_min), abs(observed_max)) * float(scale_margin)
                postscale = float(max(1.0, math.ceil(absmax)))
                module_degree, coeffs = coeffs_for(full_name, postscale)
                setattr(
                    parent,
                    child_name,
                    ScaledChebyshevSiLU(
                        coeffs,
                        postscale=postscale,
                        trainable_coeffs=bool(trainable_coeffs),
                        trainable_scale=bool(trainable_scale),
                        reference_activation="plain-silu",
                    ),
                )
                metadata[full_name] = {
                    "observed_min": observed_min,
                    "observed_max": observed_max,
                    "postscale": postscale,
                    "prescale": float(1.0 / postscale if postscale != 0.0 else 1.0),
                    "degree": int(module_degree),
                    "fit": "silu_x_on_scaled_domain",
                    "trainable_coeffs": bool(trainable_coeffs),
                    "trainable_scale": bool(trainable_scale),
                    "reference_activation": "plain-silu",
                }
            else:
                visit(child, full_name)

    visit(model)
    unused = sorted(set(overrides) - used_overrides)
    if unused:
        raise ValueError(f"--poly-degree-overrides did not match activation modules: {', '.join(unused)}")
    return metadata


def set_poly_blend_alpha(model: nn.Module, *, alpha: float) -> None:
    for module in model.modules():
        if isinstance(module, ScaledChebyshevSiLU):
            module.blend_alpha.copy_(torch.tensor(float(alpha), device=module.blend_alpha.device, dtype=module.blend_alpha.dtype))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = 0.0,
    teacher: nn.Module | None = None,
    distill_weight: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_items = 0
    label = "train" if training else "valid"
    if teacher is not None:
        teacher.eval()
    for images, masks in tqdm(loader, desc=label, leave=False):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            if training and teacher is not None and float(distill_weight) > 0.0:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = loss + float(distill_weight) * F.mse_loss(logits, teacher_logits)
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
    initial_metrics: dict[str, float] | None = None,
    teacher_model: nn.Module | None = None,
    distill_weight: float = 0.0,
    homotopy_epochs: int = 0,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_metrics: dict[str, float] | None = None
    best_dice = -math.inf
    history: list[dict[str, object]] = []
    if initial_metrics is not None and int(homotopy_epochs) <= 0:
        initial_values = [initial_metrics["dice"], initial_metrics["iou"], initial_metrics["loss"]]
        if all(math.isfinite(float(value)) for value in initial_values):
            best_metrics = dict(initial_metrics)
            best_dice = float(initial_metrics["dice"])
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
    for epoch in range(1, int(epochs) + 1):
        blend_alpha = 1.0
        if int(homotopy_epochs) > 0:
            blend_alpha = min(1.0, max(0.0, float(epoch - 1) / float(max(1, int(homotopy_epochs)))))
            set_poly_blend_alpha(model, alpha=blend_alpha)
        if int(freeze_scale_after_epoch) > 0:
            set_scaled_silu_updates(model, enabled=int(epoch) <= int(freeze_scale_after_epoch))
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=float(grad_clip_norm),
            teacher=teacher_model,
            distill_weight=float(distill_weight),
        )
        val_metrics = run_epoch(model, val_loader, device=device, optimizer=None)
        row = {
            "stage": str(stage_name),
            "epoch": int(epoch),
            "blend_alpha": float(blend_alpha),
            "train": train_metrics,
            "valid": val_metrics,
        }
        history.append(row)
        print("epoch", json.dumps(row, sort_keys=True), flush=True)
        metric_values = [
            train_metrics["dice"],
            train_metrics["iou"],
            train_metrics["loss"],
            val_metrics["dice"],
            val_metrics["iou"],
            val_metrics["loss"],
        ]
        metrics_are_finite = all(math.isfinite(float(value)) for value in metric_values)
        save_allowed = int(homotopy_epochs) <= 0 or float(blend_alpha) >= 1.0
        if metrics_are_finite and save_allowed and float(val_metrics["dice"]) >= best_dice:
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
        if not metrics_are_finite:
            print(
                "event",
                json.dumps(
                    {
                        "event": "early_stop_nonfinite_metrics",
                        "stage": str(stage_name),
                        "epoch": int(epoch),
                        "best_dice": float(best_dice),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break
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
    parser.add_argument(
        "--architecture",
        choices=("concat-unet", "unet22-plus-output"),
        default="concat-unet",
        help="Model family to train. unet22-plus-output matches Orion's U22 body plus explicit 1x1 output head.",
    )
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
    parser.add_argument("--silu-degree", type=int, default=31)
    parser.add_argument("--poly-degree", type=int, default=7)
    parser.add_argument(
        "--poly-degree-overrides",
        default="",
        help="Comma-separated activation degree overrides, e.g. dec1a_act=15,enc3b_act=15.",
    )
    parser.add_argument("--poly-finetune-epochs", type=int, default=0)
    parser.add_argument("--poly-lr", type=float, default=1.0e-5)
    parser.add_argument(
        "--train-poly-coeffs",
        action="store_true",
        help="Make polynomial coefficients trainable during optional polynomial fine-tuning.",
    )
    parser.add_argument(
        "--train-poly-scale",
        action="store_true",
        help="Make per-activation Chebyshev postscale trainable during optional polynomial fine-tuning.",
    )
    parser.add_argument(
        "--range-fit-scaled-poly",
        action="store_true",
        help="For scaled-SiLU replacement, fit Chebyshev on observed activation ranges while preserving trained target scales.",
    )
    parser.add_argument(
        "--poly-distill-weight",
        type=float,
        default=0.0,
        help="Optional MSE distillation weight against the exact SiLU stage during polynomial fine-tuning.",
    )
    parser.add_argument(
        "--poly-homotopy-epochs",
        type=int,
        default=0,
        help="Ramp polynomial modules from exact SiLU to pure Chebyshev over this many fine-tuning epochs.",
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

    architecture = str(args.architecture)
    architecture_tag = str(architecture).replace("-", "_")
    stage1_pool = "avg" if architecture == "unet22-plus-output" else str(args.stage1_pool)
    tag = f"{spec.name}_{architecture_tag}_base{int(args.base_dim)}_{int(args.image_size)}"
    stage1_name = f"relu_{stage1_pool}pool"
    stage2_init = str(args.stage2_init)
    stage2_activation = str(args.stage2_activation)
    stage2_prefix = "scaled_silu" if stage2_activation == "scaled-silu" else "silu"
    stage2_suffix = "finetune" if stage2_init == "stage1" else "retrain"
    stage2_name = f"{stage2_prefix}_avgpool_{stage2_suffix}"
    poly_degree_overrides = parse_poly_degree_overrides(str(args.poly_degree_overrides))
    poly_stage_label = f"degree_{int(args.poly_degree)}"
    if poly_degree_overrides:
        poly_stage_label += "_mixed"
    poly_stage_name = f"{stage2_prefix}_avgpool_{poly_stage_label}"
    stage1_path = Path(args.out_dir) / f"{tag}_{stage1_name}_best.pt"
    stage2_path = Path(args.out_dir) / f"{tag}_{stage2_name}_best.pt"
    started = time.time()

    stage1_model = make_stage_model(
        architecture=architecture,
        in_channels=1,
        out_channels=1,
        base_dim=int(args.base_dim),
        activation="relu",
        pool=stage1_pool,
        scale_margin=float(args.scale_margin),
        silu_degree=int(args.silu_degree),
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

    stage2_model = make_stage_model(
        architecture=architecture,
        in_channels=1,
        out_channels=1,
        base_dim=int(args.base_dim),
        activation=stage2_activation,
        pool="avg",
        scale_margin=float(args.scale_margin),
        silu_degree=int(args.silu_degree),
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
    scaled_silu_ranges: dict[str, dict[str, float]] = {}
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
                degree_overrides=poly_degree_overrides,
                scale_margin=float(args.scale_margin),
                trainable_coeffs=bool(args.train_poly_coeffs),
                trainable_scale=bool(args.train_poly_scale),
            )
        else:
            if bool(args.range_fit_scaled_poly):
                scaled_silu_ranges = collect_tracking_scaled_silu_input_ranges(stage2_model, train_loader, device=device)
                poly_meta = replace_scaled_silu_with_range_poly(
                    poly_model,
                    ranges=scaled_silu_ranges,
                    degree=int(args.poly_degree),
                    degree_overrides=poly_degree_overrides,
                    scale_margin=float(args.scale_margin),
                    trainable_coeffs=bool(args.train_poly_coeffs),
                    trainable_scale=bool(args.train_poly_scale),
                )
            else:
                poly_meta = replace_scaled_silu_with_poly(
                    poly_model,
                    degree=int(args.poly_degree),
                    degree_overrides=poly_degree_overrides,
                    trainable_coeffs=bool(args.train_poly_coeffs),
                    trainable_scale=bool(args.train_poly_scale),
                )
        poly_metrics = evaluate_with_reference(poly_model, val_loader, device=device, reference=stage2_model)
        if int(args.poly_finetune_epochs) > 0:
            if int(args.poly_homotopy_epochs) > 0:
                set_poly_blend_alpha(poly_model, alpha=0.0)
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
                initial_metrics=poly_metrics,
                teacher_model=stage2_model if float(args.poly_distill_weight) > 0.0 else None,
                distill_weight=float(args.poly_distill_weight),
                homotopy_epochs=int(args.poly_homotopy_epochs),
                freeze_scale_after_epoch=int(args.freeze_scale_after_epoch),
            )
            poly_checkpoint = torch.load(poly_finetune_path, map_location="cpu", weights_only=False)
            poly_model.load_state_dict(poly_checkpoint["state_dict"])
            poly_model.to(device)
            set_poly_blend_alpha(poly_model, alpha=1.0)
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
                f"{stage2_name}_to_{poly_stage_label}": {
                    "dice_drop": float(stage2_metrics["dice"] - poly_metrics["dice"]),
                    "iou_drop": float(stage2_metrics["iou"] - poly_metrics["iou"]),
                },
                f"{stage1_name}_to_{poly_stage_label}": {
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
            "architecture": architecture,
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
            "silu_degree": int(args.silu_degree),
            "poly_degree": int(args.poly_degree),
            "poly_degree_overrides": poly_degree_overrides,
            "poly_finetune_epochs": int(args.poly_finetune_epochs),
            "poly_lr": float(args.poly_lr),
            "train_poly_coeffs": bool(args.train_poly_coeffs),
            "train_poly_scale": bool(args.train_poly_scale),
            "range_fit_scaled_poly": bool(args.range_fit_scaled_poly),
            "poly_distill_weight": float(args.poly_distill_weight),
            "poly_homotopy_epochs": int(args.poly_homotopy_epochs),
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
        "scaled_silu_fit_ranges": scaled_silu_ranges,
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
