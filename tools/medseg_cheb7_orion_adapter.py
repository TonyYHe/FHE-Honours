#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import orion.nn as on
from orion.models.unet import UNet22PlusOutput
from orion.nn.activation import _bootstrap_prescale_fusion
from orion.nn.module import timer


ACTIVATION_NAMES = (
    "enc1a_act",
    "enc1b_act",
    "enc2a_act",
    "enc2b_act",
    "enc3a_act",
    "enc3b_act",
    "enc4a_act",
    "enc4b_act",
    "bottlenecka_act",
    "bottleneckb_act",
    "dec4a_act",
    "dec4b_act",
    "dec3a_act",
    "dec3b_act",
    "dec2a_act",
    "dec2b_act",
    "dec1a_act",
    "dec1b_act",
)


class CheckpointScaledChebyshevSiLU(on.Chebyshev):
    """Orion-compatible scaled-domain Chebyshev SiLU loaded from medseg checkpoints."""

    def __init__(self, *, degree: int = 7, postscale: float = 1.0) -> None:
        on.Module.__init__(self)
        self.degree = int(degree)
        self.fn = F.silu
        self.within_composite = False
        self.register_buffer("coeffs", torch.zeros(int(self.degree) + 1, dtype=torch.float32))
        self.register_buffer("postscale_tensor", torch.tensor(float(postscale), dtype=torch.float32))
        self.output_scale = None
        self.constant = 0.0
        self.set_depth()

    @property
    def postscale(self) -> float:
        return float(self.postscale_tensor.detach().cpu().item())

    @property
    def prescale(self) -> float:
        postscale = self.postscale
        return 1.0 / postscale if postscale != 0.0 else 1.0

    def set_depth(self) -> None:
        self.depth = int(math.ceil(math.log2(int(self.degree) + 1)))
        if self.prescale != 1.0:
            self.depth += 1

    def set_output_scale(self, output_scale) -> None:
        self.output_scale = output_scale

    def fit(self) -> None:
        self.set_depth()

    def set_coeffs(self, coeffs) -> None:
        values = torch.tensor(list(coeffs), dtype=self.coeffs.dtype, device=self.coeffs.device)
        if int(values.numel()) != int(self.coeffs.numel()):
            raise ValueError(
                f"cannot set {int(values.numel())} coefficients on degree-{int(self.degree)} "
                f"Chebyshev SiLU adapter"
            )
        self.coeffs.copy_(values)

    def _effective_coeffs(self) -> list[float]:
        postscale = self.postscale
        coeffs = [float(value) * float(postscale) for value in self.coeffs.detach().cpu().flatten().tolist()]
        output_scale_fusion = getattr(self, "_bootstrap_output_scale_fusion", None)
        if output_scale_fusion is not None and coeffs:
            coeffs = [float(value) * float(output_scale_fusion) for value in coeffs]
        fusion = _bootstrap_prescale_fusion(self)
        if fusion is not None and coeffs:
            coeffs = [float(value) * float(fusion["scale"]) for value in coeffs]
            coeffs[0] = float(coeffs[0]) + float(fusion["bias"])
        return coeffs

    def compile(self) -> None:
        self.set_depth()
        self.poly = self.scheme.poly_evaluator.generate_chebyshev(self._effective_coeffs())

    @staticmethod
    def _chebyshev_eval(z: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        if int(coeffs.numel()) == 0:
            return torch.zeros_like(z)
        out = coeffs[0].expand_as(z)
        if int(coeffs.numel()) == 1:
            return out
        t0 = torch.ones_like(z)
        t1 = z
        out = out + coeffs[1] * t1
        for index in range(2, int(coeffs.numel())):
            t2 = 2.0 * z * t1 - t0
            out = out + coeffs[index] * t2
            t0, t1 = t1, t2
        return out

    @timer
    def forward(self, x):
        prescale = self.prescale
        if self.he_mode:
            if not self.fused and prescale != 1.0:
                x *= prescale
            return self.scheme.poly_evaluator.evaluate_polynomial(x, self.poly, self.output_scale)

        postscale = self.postscale_tensor.to(device=x.device, dtype=x.dtype)
        z = x / postscale
        coeffs = self.coeffs.to(device=x.device, dtype=x.dtype)
        return postscale * self._chebyshev_eval(z, coeffs)


def load_degree_replacements(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("degree_replacements", {})
    if not isinstance(rows, dict):
        raise ValueError(f"{path} does not contain a degree_replacements object")
    return {str(key): dict(value) for key, value in rows.items() if isinstance(value, dict)}


def replace_orion_silu_with_checkpoint_cheb(
    model: torch.nn.Module,
    *,
    checkpoint_state: dict[str, torch.Tensor],
    replacements: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    installed: dict[str, dict[str, float]] = {}
    for name in ACTIVATION_NAMES:
        key = f"{name}.coeffs"
        if key not in checkpoint_state:
            continue
        coeffs = checkpoint_state[key]
        degree = int(coeffs.numel()) - 1
        row = dict(replacements.get(name, {}) or {})
        log_key = f"{name}.log_postscale"
        tensor_key = f"{name}.postscale_tensor"
        if log_key in checkpoint_state:
            postscale = float(torch.exp(checkpoint_state[log_key].detach().cpu().to(dtype=torch.float32)).item())
        elif tensor_key in checkpoint_state:
            postscale = float(checkpoint_state[tensor_key].detach().cpu().to(dtype=torch.float32).item())
        else:
            postscale = float(row.get("postscale", row.get("domain_postscale", 1.0)))
        setattr(model, name, CheckpointScaledChebyshevSiLU(degree=degree, postscale=postscale))
        installed[name] = {
            "degree": float(degree),
            "postscale": float(postscale),
            "prescale": float(1.0 / postscale if postscale != 0.0 else 1.0),
        }
    return installed


def build_orion_cheb7_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    replacements_path: Path | None = None,
    device: torch.device | str = "cpu",
) -> tuple[UNet22PlusOutput, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model_cfg = dict(checkpoint.get("model", {}) or {})
    replacements = load_degree_replacements(replacements_path)
    model = UNet22PlusOutput(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_dim", 32)),
        activation="silu",
        silu_degree=7,
    )
    installed = replace_orion_silu_with_checkpoint_cheb(
        model,
        checkpoint_state=state,
        replacements=replacements,
    )
    adapter_state = dict(state)
    for name in installed:
        log_key = f"{name}.log_postscale"
        tensor_key = f"{name}.postscale_tensor"
        if log_key in adapter_state and tensor_key not in adapter_state:
            adapter_state[tensor_key] = torch.exp(adapter_state[log_key].detach().cpu().to(dtype=torch.float32))
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    ignored_missing = {f"{name}.postscale_tensor" for name in installed}
    ignored_unexpected = {
        f"{name}.{suffix}"
        for name in installed
        for suffix in ("log_postscale", "blend_alpha", "reference_postscale_tensor")
    }
    remaining_missing = [key for key in missing if key not in ignored_missing]
    remaining_unexpected = [key for key in unexpected if key not in ignored_unexpected]
    if remaining_unexpected or remaining_missing:
        raise RuntimeError(
            "checkpoint did not match Orion Cheb7 adapter: "
            f"missing={remaining_missing[:8]} unexpected={remaining_unexpected[:8]}"
        )
    model.to(device)
    model.eval()
    return model, {
        "checkpoint": str(checkpoint_path),
        "replacements_path": None if replacements_path is None else str(replacements_path),
        "installed_activations": installed,
    }
