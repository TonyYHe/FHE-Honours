"""Checkpoint-trained activation support for grouped WPC CIPS ciphertexts.

This module keeps the experimental CIPS packing contract around Orion's
Chebyshev evaluator and bootstrap.  The coefficients and the independent
pre/post scales are loaded from the trained medseg checkpoints; they are not
refitted by the experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import torch

from orion.backend.python.tensors import CipherTensor
from orion.experimental.wpc_cips_activation import (
    _normalise_shape,
    _validate_signature,
)
from orion.nn.activation import _bootstrap_prescale_fusion
from orion.nn.module import Module, timer
from orion.nn.operations import Bootstrap


@dataclass(frozen=True)
class CheckpointChebyshevSpec:
    """The exact learned parameters of one scaled Chebyshev activation."""

    name: str
    coefficients: tuple[float, ...]
    postscale: float
    prescale: float
    blend_alpha: float

    @property
    def degree(self) -> int:
        return int(len(self.coefficients) - 1)

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
        name: str,
    ) -> "CheckpointChebyshevSpec":
        prefix = str(name).rstrip(".")
        coeff_key = f"{prefix}.coeffs"
        if coeff_key not in state:
            raise KeyError(f"checkpoint is missing {coeff_key}")
        coefficients = tuple(
            float(value)
            for value in state[coeff_key].detach().cpu().flatten().tolist()
        )
        if len(coefficients) < 2:
            raise ValueError("checkpoint Chebyshev activation needs degree at least one")

        def positive_scale(log_key: str, tensor_key: str, fallback: float) -> float:
            if log_key in state:
                value = float(
                    torch.exp(state[log_key].detach().cpu().to(torch.float64)).item()
                )
            elif tensor_key in state:
                value = float(state[tensor_key].detach().cpu().to(torch.float64).item())
            else:
                value = float(fallback)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"checkpoint activation scale {log_key} is invalid")
            return value

        postscale = positive_scale(
            f"{prefix}.log_postscale",
            f"{prefix}.postscale_tensor",
            1.0,
        )
        prescale = positive_scale(
            f"{prefix}.log_prescale",
            f"{prefix}.prescale_tensor",
            1.0 / postscale,
        )
        alpha_tensor = state.get(f"{prefix}.blend_alpha")
        blend_alpha = (
            float(alpha_tensor.detach().cpu().item())
            if alpha_tensor is not None
            else 1.0
        )
        if not math.isclose(blend_alpha, 1.0, rel_tol=0.0, abs_tol=1.0e-7):
            raise ValueError(
                "the FHE checkpoint path requires blend_alpha=1; "
                "a plaintext SiLU blend cannot be evaluated homomorphically"
            )
        return cls(
            name=prefix,
            coefficients=coefficients,
            postscale=float(postscale),
            prescale=float(prescale),
            blend_alpha=float(blend_alpha),
        )

    def evaluate(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate the checkpoint formula with its Chebyshev recurrence."""

        z = value * float(self.prescale)
        coefficients = torch.tensor(
            self.coefficients,
            dtype=value.dtype,
            device=value.device,
        )
        t0 = torch.ones_like(z)
        t1 = z
        result = coefficients[0] * t0 + coefficients[1] * t1
        for index in range(2, len(self.coefficients)):
            t2 = 2.0 * z * t1 - t0
            result = result + coefficients[index] * t2
            t0, t1 = t1, t2
        return float(self.postscale) * result


class CheckpointScaledChebyshevSiLU(Module):
    """Orion polynomial evaluator using exact checkpoint coefficients."""

    def __init__(self, spec: CheckpointChebyshevSpec) -> None:
        super().__init__()
        self.spec = spec
        self.degree = int(spec.degree)
        self.register_buffer(
            "coeffs",
            torch.tensor(spec.coefficients, dtype=torch.float32),
        )
        self.register_buffer(
            "postscale_tensor",
            torch.tensor(float(spec.postscale), dtype=torch.float32),
        )
        self.register_buffer(
            "prescale_tensor",
            torch.tensor(float(spec.prescale), dtype=torch.float32),
        )
        self.output_scale = None
        self.constant = 0.0
        self.set_depth_from_parameters()

    @property
    def postscale(self) -> float:
        return float(self.postscale_tensor.detach().cpu().item())

    @property
    def prescale(self) -> float:
        return float(self.prescale_tensor.detach().cpu().item())

    def set_depth_from_parameters(self) -> None:
        self.depth = int(math.ceil(math.log2(int(self.degree) + 1)))
        if self.prescale != 1.0:
            self.depth += 1

    def set_output_scale(self, output_scale: Any) -> None:
        self.output_scale = output_scale

    def _effective_coefficients(self) -> list[float]:
        coefficients = [
            float(value) * float(self.postscale)
            for value in self.coeffs.detach().cpu().flatten().tolist()
        ]
        output_scale_fusion = getattr(self, "_bootstrap_output_scale_fusion", None)
        if output_scale_fusion is not None:
            coefficients = [
                value * float(output_scale_fusion) for value in coefficients
            ]
        fusion = _bootstrap_prescale_fusion(self)
        if fusion is not None:
            coefficients = [
                value * float(fusion["scale"]) for value in coefficients
            ]
            coefficients[0] += float(fusion["bias"])
        return coefficients

    def compile(self) -> None:
        self.set_depth_from_parameters()
        self.poly = self.scheme.poly_evaluator.generate_chebyshev(
            self._effective_coefficients()
        )

    @timer
    def forward(self, value: Any):
        if self.he_mode:
            scaled = value
            if not self.fused and self.prescale != 1.0:
                # Do not consume the caller's ciphertext in place.  Keeping
                # the pre-activation value intact is important for skip/fanout
                # safety and for intermediate correctness diagnostics.
                scaled = value * self.prescale
            try:
                return self.scheme.poly_evaluator.evaluate_polynomial(
                    scaled,
                    self.poly,
                    self.output_scale,
                )
            finally:
                if scaled is not value:
                    scaled.release()
        return self.spec.evaluate(value)


class WPCCIPSTrainedActivationBootstrap(Module):
    """Checkpoint Chebyshev activation followed by a real CIPS bootstrap."""

    def __init__(
        self,
        *,
        logical_shape: Any,
        packing_signature: Any,
        input_level: int,
        output_level: int,
        activation_spec: CheckpointChebyshevSpec,
        bootstrap_bound: float,
    ) -> None:
        super().__init__()
        self.logical_shape = _normalise_shape(logical_shape)
        self.packing_signature = _validate_signature(
            packing_signature,
            logical_shape=self.logical_shape,
        )
        self.input_level = int(input_level)
        self.output_level = int(output_level)
        self.activation_spec = activation_spec
        self.activation = CheckpointScaledChebyshevSiLU(activation_spec)
        self.activation_output_level = int(
            self.input_level - int(self.activation.depth)
        )
        self.bootstrap_bound = float(bootstrap_bound)
        if self.activation_output_level < 1:
            raise ValueError(
                "trained activation requires enough levels for its polynomial "
                "and one bootstrap preprocessing level"
            )
        if self.output_level <= self.activation_output_level:
            raise ValueError("bootstrap must restore the activation to a higher level")
        if not math.isfinite(self.bootstrap_bound) or self.bootstrap_bound <= 0.0:
            raise ValueError("bootstrap_bound must be a positive finite number")

        bound = torch.tensor(self.bootstrap_bound, dtype=torch.float64)
        self.bootstrap = Bootstrap(
            input_min=-bound,
            input_max=bound,
            input_level=int(self.activation_output_level),
        )
        self.bootstrap.fhe_input_shape = torch.Size(
            [self.group_count, self.slots]
        )
        self.bootstrap._bootstrap_prescale_active_mask = self._active_slot_mask()
        self.bootstrap.bootstrap_debug_name = (
            f"wpc_cips_checkpoint_{activation_spec.name}"
        )
        self.compiled = False
        self.cleaned = False
        self.compile_summary: dict[str, Any] = {}
        self.last_evaluation: dict[str, Any] = {}
        self.set_depth(0)

    @property
    def slots(self) -> int:
        return int(self.packing_signature[1])

    @property
    def group_count(self) -> int:
        return int(len(self.packing_signature[5]))

    def _active_slot_mask(self) -> torch.Tensor:
        _, _, height, width = self.logical_shape
        capacity = int(self.slots // (height * width))
        masks: list[torch.Tensor] = []
        for start, end in self.packing_signature[5]:
            mask = torch.zeros((height * width, capacity), dtype=torch.bool)
            mask[:, : int(end - start)] = True
            masks.append(mask.reshape(-1))
        return torch.cat(masks)

    def _levels(self, value: CipherTensor) -> list[int]:
        return [
            int(value.backend.GetCiphertextLevel(int(identifier)))
            for identifier in value.ids
        ]

    def _validate_value(self, value: CipherTensor, level: int) -> None:
        if not isinstance(value, CipherTensor):
            raise TypeError("trained activation bridge expects a CipherTensor")
        if getattr(value, "_wpc_cips_packing_signature", None) != self.packing_signature:
            raise ValueError("ciphertext does not have the required WPC CIPS packing")
        if len(value.ids) != self.group_count:
            raise ValueError("ciphertext group count does not match the CIPS contract")
        levels = self._levels(value)
        if any(actual != int(level) for actual in levels):
            raise ValueError(
                f"ciphertext levels {levels} do not match required level {level}"
            )

    def compile(self, scheme: Any | None = None) -> dict[str, Any]:
        if self.compiled:
            raise RuntimeError("trained activation bridge is already compiled")
        if self.cleaned:
            raise RuntimeError("a cleaned trained activation bridge cannot be reused")
        scheme = scheme if scheme is not None else self.scheme
        if scheme is None:
            raise RuntimeError("set the Orion scheme before compiling the bridge")
        if int(scheme.params.get_slots()) != self.slots:
            raise ValueError("bridge slot count does not match the scheme")
        if int(scheme.params.get_max_level()) != self.output_level:
            raise ValueError("bridge output level must equal the scheme maximum level")

        Module.set_scheme(scheme)
        Module.set_margin(float(scheme.params.get_margin()))
        self.activation.compile()
        expected_poly_depth = int(
            math.ceil(math.log2(int(self.activation_spec.degree) + 1))
        )
        # Lattigo's Chebyshev evaluator uses the documented optimal depth
        # ceil(log2(degree+1)); unlike the clear backend it does not expose a
        # GetPolyDepth binding.  The runtime level assertion after evaluation
        # is the authoritative correctness check for this schedule.
        backend_poly_depth = expected_poly_depth
        expected_activation_depth = int(
            backend_poly_depth
            + (1 if not math.isclose(self.activation_spec.prescale, 1.0) else 0)
        )
        self.activation_output_level = int(self.input_level - expected_activation_depth)
        if self.activation_output_level < 1:
            raise ValueError("compiled activation leaves no bootstrap preprocessing level")
        self.bootstrap.input_level = int(self.activation_output_level)
        self.bootstrap.fit()
        if float(self.bootstrap.constant) != 0.0:
            raise RuntimeError("symmetric bootstrap bounds must have zero affine shift")
        self.bootstrap.compile()
        scheme.bootstrapper.generate_bootstrapper(int(self.bootstrap.bootstrap_slots))
        self.he()
        self.compiled = True
        self.compile_summary = {
            "activation_name": self.activation_spec.name,
            "activation_degree": int(self.activation_spec.degree),
            "activation_coefficients": list(self.activation_spec.coefficients),
            "activation_postscale": float(self.activation_spec.postscale),
            "activation_prescale": float(self.activation_spec.prescale),
            "backend_polynomial_depth": int(backend_poly_depth),
            "activation_total_depth": int(expected_activation_depth),
            "input_level": int(self.input_level),
            "activation_output_level": int(self.activation_output_level),
            "bootstrap_preprocess_output_level": int(
                self.activation_output_level - 1
            ),
            "bootstrap_output_level": int(self.output_level),
            "bootstrap_slots": int(self.bootstrap.bootstrap_slots),
            "ciphertext_group_count": int(self.group_count),
            "bootstrap_bound": float(self.bootstrap_bound),
            "active_slot_count": int(
                self.bootstrap._bootstrap_prescale_active_mask.sum().item()
            ),
            "packed_slot_count": int(
                self.bootstrap._bootstrap_prescale_active_mask.numel()
            ),
        }
        return dict(self.compile_summary)

    def forward(self, value: Any):
        if not self.he_mode:
            return self.activation(value)
        if not self.compiled:
            raise RuntimeError("compile the trained activation bridge before use")
        self._validate_value(value, self.input_level)
        started = time.perf_counter()
        activation_started = time.perf_counter()
        activated = self.activation(value)
        activation_s = float(time.perf_counter() - activation_started)
        activated._wpc_cips_packing_signature = self.packing_signature
        try:
            self._validate_value(activated, self.activation_output_level)
        except Exception:
            activated.release()
            raise

        bootstrap_started = time.perf_counter()
        try:
            refreshed = self.bootstrap(activated)
        finally:
            activated.release()
        bootstrap_s = float(time.perf_counter() - bootstrap_started)
        refreshed._wpc_cips_packing_signature = self.packing_signature
        try:
            self._validate_value(refreshed, self.output_level)
        except Exception:
            refreshed.release()
            raise
        records = list(self.bootstrap._bootstrap_runtime_profile)
        self.last_evaluation = {
            "activation_name": self.activation_spec.name,
            "input_level": int(self.input_level),
            "activation_output_level": int(self.activation_output_level),
            "bootstrap_output_level": int(self.output_level),
            "ciphertext_group_count": int(self.group_count),
            "activation_s": float(activation_s),
            "bootstrap_s": float(bootstrap_s),
            "total_s": float(time.perf_counter() - started),
            "packing_signature_preserved": True,
            "clear_repack_or_encode_count": 0,
            "bootstrap_runtime_records": [dict(row) for row in records[-self.group_count :]],
        }
        return refreshed

    def clear_runtime_profile(self) -> None:
        self.bootstrap._bootstrap_runtime_profile = []
        self.bootstrap._bootstrap_runtime_call_index = 0

    def cleanup(self) -> None:
        if self.cleaned:
            return
        plaintexts: dict[int, Any] = {}
        for cache_name in ("_prescale_ptxt_cache", "_prescale_part_ptxt_cache"):
            for plaintext in getattr(self.bootstrap, cache_name, {}).values():
                plaintexts[id(plaintext)] = plaintext
        for plaintext in plaintexts.values():
            release = getattr(plaintext, "release", None)
            if callable(release):
                release()
        self.bootstrap._prescale_ptxt_cache = {}
        self.bootstrap._prescale_part_ptxt_cache = {}
        self.bootstrap.prescale_ptxt = None
        self.compiled = False
        self.cleaned = True


__all__ = [
    "CheckpointChebyshevSpec",
    "CheckpointScaledChebyshevSiLU",
    "WPCCIPSTrainedActivationBootstrap",
]
