"""Activation and bootstrap bridge for grouped WPC CIPS ciphertexts.

The ordinary Orion activation and bootstrap operators are layout agnostic, but
their output ``CipherTensor`` objects do not carry the experimental CIPS
packing contract.  This module provides an opt-in bridge that validates and
preserves that contract while executing real Orion ``Quad`` and ``Bootstrap``
modules.  It is intended to sit between two compatible
``WPCCIPSConv2dPlan`` instances.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from orion.backend.python.tensors import CipherTensor
from orion.nn.activation import Quad
from orion.nn.module import Module
from orion.nn.operations import Bootstrap


def _normalise_shape(shape: Any) -> tuple[int, int, int, int]:
    values = tuple(int(value) for value in shape)
    if len(values) != 4 or values[0] != 1:
        raise ValueError("WPC CIPS activation bridge requires shape [1,C,H,W]")
    if any(value <= 0 for value in values):
        raise ValueError("WPC CIPS activation bridge shape must be positive")
    return values


def _validate_signature(
    signature: Any,
    *,
    logical_shape: tuple[int, int, int, int],
) -> tuple[Any, ...]:
    if not isinstance(signature, tuple) or len(signature) != 6:
        raise ValueError("invalid WPC CIPS packing signature")
    batch, channels, height, width = logical_shape
    del batch
    if (
        signature[0] != "wpc_cips"
        or int(signature[2]) != int(channels)
        or int(signature[3]) != int(height)
        or int(signature[4]) != int(width)
    ):
        raise ValueError("packing signature does not match the logical shape")
    slots = int(signature[1])
    ranges = tuple(tuple(int(value) for value in row) for row in signature[5])
    if slots <= 0 or slots & (slots - 1):
        raise ValueError("packing signature slot count must be a power of two")
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != channels:
        raise ValueError("packing signature channel groups are incomplete")
    for index, (start, end) in enumerate(ranges):
        if start < 0 or end <= start or end > channels:
            raise ValueError("packing signature contains an invalid channel group")
        if index and start != ranges[index - 1][1]:
            raise ValueError("packing signature channel groups must be contiguous")
        if (end - start) * height * width > slots:
            raise ValueError("packing signature channel group exceeds its slot count")
    return (
        "wpc_cips",
        slots,
        channels,
        height,
        width,
        ranges,
    )


class WPCCIPSActivationBootstrap(Module):
    """Real quadratic activation plus bootstrap for grouped CIPS ciphertexts.

    The quadratic activation consumes one ciphertext level.  Bootstrap
    preprocessing consumes the following level, and the backend bootstrap
    refreshes every ciphertext group to the scheme maximum level.  Symmetric
    bootstrap bounds keep unused slots at zero, which is required by the CIPS
    channel-group packing contract.
    """

    def __init__(
        self,
        *,
        logical_shape: Any,
        packing_signature: Any,
        input_level: int,
        output_level: int,
        bootstrap_bound: float,
    ) -> None:
        super().__init__()
        self.logical_shape = _normalise_shape(logical_shape)
        self.packing_signature = _validate_signature(
            packing_signature,
            logical_shape=self.logical_shape,
        )
        self.input_level = int(input_level)
        self.activation_output_level = int(self.input_level - 1)
        self.bootstrap_input_level = int(self.activation_output_level)
        self.output_level = int(output_level)
        self.bootstrap_bound = float(bootstrap_bound)
        if self.input_level < 2:
            raise ValueError(
                "activation/bootstrap bridge requires two levels before refresh"
            )
        if not math.isfinite(self.bootstrap_bound) or self.bootstrap_bound <= 0:
            raise ValueError("bootstrap_bound must be a positive finite number")

        self.activation = Quad()
        # The real activation output is nonnegative, but symmetric bounds are
        # intentional: a zero affine shift prevents unused CIPS slots from
        # becoming nonzero during bootstrap postprocessing.
        bound = torch.tensor(self.bootstrap_bound, dtype=torch.float64)
        self.bootstrap = Bootstrap(
            input_min=-bound,
            input_max=bound,
            input_level=int(self.bootstrap_input_level),
        )
        self.bootstrap.fhe_input_shape = torch.Size(self.logical_shape)
        self.bootstrap._bootstrap_prescale_active_mask = self._active_slot_mask()
        self.bootstrap.bootstrap_debug_name = "wpc_cips_activation_bootstrap"
        self.compiled = False
        self.cleaned = False
        self.last_evaluation: dict[str, Any] = {}
        self.compile_summary: dict[str, Any] = {}
        self.set_depth(0)

    @classmethod
    def between_plans(
        cls,
        producer_plan: Any,
        consumer_plan: Any,
        *,
        bootstrap_bound: float,
    ) -> "WPCCIPSActivationBootstrap":
        producer_signature = producer_plan.output_packing_signature
        consumer_signature = consumer_plan.input_packing_signature
        if producer_signature != consumer_signature:
            raise ValueError("producer and consumer CIPS packing signatures differ")
        if tuple(producer_plan.output_shape) != tuple(consumer_plan.input_shape):
            raise ValueError("producer and consumer logical shapes differ")
        return cls(
            logical_shape=producer_plan.output_shape,
            packing_signature=producer_signature,
            input_level=int(producer_plan.output_level),
            output_level=int(consumer_plan.level),
            bootstrap_bound=float(bootstrap_bound),
        )

    @property
    def slots(self) -> int:
        return int(self.packing_signature[1])

    @property
    def group_count(self) -> int:
        return int(len(self.packing_signature[5]))

    def _active_slot_mask(self) -> torch.Tensor:
        """Return the exact interleaved CIPS mask for every ciphertext group."""

        _, _, height, width = self.logical_shape
        ranges = self.packing_signature[5]
        channel_capacity = int(self.slots // (height * width))
        masks: list[torch.Tensor] = []
        for start, end in ranges:
            active_channels = int(end - start)
            spatial_mask = torch.zeros(
                (int(height * width), int(channel_capacity)),
                dtype=torch.bool,
            )
            spatial_mask[:, :active_channels] = True
            masks.append(spatial_mask.reshape(-1))
        return torch.cat(masks)

    def compile(self, scheme: Any | None = None) -> dict[str, Any]:
        if self.compiled:
            raise RuntimeError("WPC CIPS activation/bootstrap bridge is already compiled")
        if self.cleaned:
            raise RuntimeError("a cleaned activation/bootstrap bridge cannot be reused")
        scheme = scheme if scheme is not None else self.scheme
        if scheme is None:
            raise RuntimeError("set the Orion scheme before compiling the bridge")
        if int(scheme.params.get_slots()) != int(self.slots):
            raise ValueError("bridge slot count does not match the scheme")
        if int(scheme.params.get_max_level()) != int(self.output_level):
            raise ValueError(
                "consumer level must equal the backend bootstrap output level"
            )

        # Module.scheme and Module.margin are normally set by Orion's network
        # compiler.  This experimental plan is compiled explicitly, so set the
        # same shared module context here.
        Module.set_scheme(scheme)
        Module.set_margin(float(scheme.params.get_margin()))
        self.bootstrap.fit()
        if float(self.bootstrap.constant) != 0.0:
            raise RuntimeError(
                "symmetric bootstrap bounds must produce a zero affine shift"
            )
        self.bootstrap.compile()
        scheme.bootstrapper.generate_bootstrapper(int(self.slots))
        self.he()
        self.compiled = True
        self.cleaned = False
        self.compile_summary = {
            "activation": "orion.nn.Quad",
            "activation_depth": 1,
            "input_level": int(self.input_level),
            "activation_output_level": int(self.activation_output_level),
            "bootstrap_preprocess_output_level": int(
                self.bootstrap_input_level - 1
            ),
            "bootstrap_output_level": int(self.output_level),
            "bootstrap_slots": int(self.bootstrap.bootstrap_slots),
            "ciphertext_group_count": int(self.group_count),
            "bootstrap_bound": float(self.bootstrap_bound),
            "bootstrap_prescale": float(self.bootstrap.prescale),
            "bootstrap_postscale": float(self.bootstrap.postscale),
            "bootstrap_constant": float(self.bootstrap.constant),
            "active_slot_count": int(
                self.bootstrap._bootstrap_prescale_active_mask.sum().item()
            ),
            "packed_slot_count": int(
                self.bootstrap._bootstrap_prescale_active_mask.numel()
            ),
            "inactive_slot_count": int(
                self.bootstrap._bootstrap_prescale_active_mask.numel()
                - self.bootstrap._bootstrap_prescale_active_mask.sum().item()
            ),
            "packing_signature": list(self.packing_signature),
        }
        return dict(self.compile_summary)

    def _cipher_levels(self, value: CipherTensor) -> list[int]:
        return [
            int(value.backend.GetCiphertextLevel(int(ciphertext_id)))
            for ciphertext_id in value.ids
        ]

    def _validate_value(self, value: CipherTensor, *, level: int) -> None:
        if not isinstance(value, CipherTensor):
            raise TypeError("WPC CIPS activation bridge expects a CipherTensor")
        if getattr(value, "_wpc_cips_packing_signature", None) != self.packing_signature:
            raise ValueError("ciphertext does not have the required WPC CIPS packing")
        if len(value.ids) != int(self.group_count):
            raise ValueError(
                f"expected {self.group_count} ciphertext groups, got {len(value.ids)}"
            )
        levels = self._cipher_levels(value)
        if any(int(actual) != int(level) for actual in levels):
            raise ValueError(
                f"ciphertext levels {levels} do not match required level {level}"
            )

    def forward(self, value: Any):
        if not self.he_mode:
            return self.bootstrap(self.activation(value))
        if not self.compiled:
            raise RuntimeError("compile the WPC CIPS activation bridge before use")
        self._validate_value(value, level=int(self.input_level))

        started = time.perf_counter()
        activation_started = time.perf_counter()
        activated = self.activation(value)
        activation_s = float(time.perf_counter() - activation_started)
        activated._wpc_cips_packing_signature = self.packing_signature
        try:
            self._validate_value(activated, level=int(self.activation_output_level))
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
            self._validate_value(refreshed, level=int(self.output_level))
        except Exception:
            refreshed.release()
            raise

        runtime_records = list(self.bootstrap._bootstrap_runtime_profile)
        latest = dict(runtime_records[-1]) if runtime_records else {}
        self.last_evaluation = {
            "input_level": int(self.input_level),
            "activation_output_level": int(self.activation_output_level),
            "bootstrap_output_level": int(self.output_level),
            "ciphertext_group_count": int(self.group_count),
            "activation_s": float(activation_s),
            "bootstrap_s": float(bootstrap_s),
            "total_s": float(time.perf_counter() - started),
            "packing_signature_preserved": bool(
                getattr(refreshed, "_wpc_cips_packing_signature", None)
                == self.packing_signature
            ),
            "bootstrap_runtime_record": latest,
        }
        return refreshed

    def clear_runtime_profile(self) -> None:
        self.bootstrap._bootstrap_runtime_profile = []
        self.bootstrap._bootstrap_runtime_call_index = 0

    def cleanup(self) -> None:
        if self.cleaned:
            return
        plaintexts: dict[int, Any] = {}
        for plaintext in list(
            getattr(self.bootstrap, "_prescale_ptxt_cache", {}).values()
        ):
            plaintexts[id(plaintext)] = plaintext
        for plaintext in list(
            getattr(self.bootstrap, "_prescale_part_ptxt_cache", {}).values()
        ):
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


__all__ = ["WPCCIPSActivationBootstrap"]
