"""Composition helpers for an encrypted WPC CIPS U-Net block."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from orion.backend.python.tensors import CipherTensor
from orion.experimental.wpc_cips_activation import (
    _normalise_shape,
    _validate_signature,
)
from orion.experimental.wpc_cips_branches import unpack_cips_groups
from orion.nn.module import Module
from orion.nn.operations import Bootstrap


class WPCCIPSBootstrapRefresh(Module):
    """Bootstrap grouped CIPS ciphertexts without changing logical values.

    A U-Net skip tensor is retained while the main branch consumes several
    levels in downsampling and bottleneck work.  Refreshing the main branch
    before transposed convolution restores it to the scheme maximum level, so
    the upsampled result and the saved skip reach concatenation at the same
    level.  Symmetric bounds keep the affine bootstrap shift at zero and an
    exact interleaved active-slot mask preserves unused CIPS slots as zero.
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
        self.input_shape = _normalise_shape(logical_shape)
        self.output_shape = self.input_shape
        self.input_packing_signature = _validate_signature(
            packing_signature,
            logical_shape=self.input_shape,
        )
        self.output_packing_signature = self.input_packing_signature
        self.input_level = int(input_level)
        self.output_level = int(output_level)
        self.bootstrap_bound = float(bootstrap_bound)
        if self.input_level < 1:
            raise ValueError("CIPS bootstrap refresh requires one preprocessing level")
        if self.output_level <= self.input_level:
            raise ValueError("CIPS bootstrap refresh must restore a higher level")
        if not math.isfinite(self.bootstrap_bound) or self.bootstrap_bound <= 0:
            raise ValueError("bootstrap_bound must be a positive finite number")

        bound = torch.tensor(self.bootstrap_bound, dtype=torch.float64)
        self.bootstrap = Bootstrap(
            input_min=-bound,
            input_max=bound,
            input_level=int(self.input_level),
        )
        # Bootstrap must cover the physical CIPS messages, not merely the
        # smaller logical tensor. Active channels are interleaved throughout
        # all ``slots`` positions, including positions beyond the next power
        # of two of C*H*W for a partial channel group.
        self.bootstrap.fhe_input_shape = torch.Size(
            [len(self.input_packing_signature[5]), int(self.input_packing_signature[1])]
        )
        self.bootstrap._bootstrap_prescale_active_mask = self._active_slot_mask()
        self.bootstrap.bootstrap_debug_name = "wpc_cips_unet_level_refresh"
        self.compiled = False
        self.cleaned = False
        self.last_evaluation: dict[str, Any] = {}
        self.compile_summary: dict[str, Any] = {}
        self.set_depth(0)

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    @classmethod
    def between_plans(
        cls,
        producer_plan: Any,
        consumer_plan: Any,
        *,
        bootstrap_bound: float,
    ) -> "WPCCIPSBootstrapRefresh":
        if producer_plan.output_packing_signature != consumer_plan.input_packing_signature:
            raise ValueError("producer and consumer CIPS packing signatures differ")
        if tuple(producer_plan.output_shape) != tuple(consumer_plan.input_shape):
            raise ValueError("producer and consumer logical shapes differ")
        return cls(
            logical_shape=producer_plan.output_shape,
            packing_signature=producer_plan.output_packing_signature,
            input_level=int(producer_plan.output_level),
            output_level=int(consumer_plan.level),
            bootstrap_bound=float(bootstrap_bound),
        )

    @property
    def slots(self) -> int:
        return int(self.input_packing_signature[1])

    @property
    def group_count(self) -> int:
        return int(len(self.input_packing_signature[5]))

    def _active_slot_mask(self) -> torch.Tensor:
        _, _, height, width = self.input_shape
        capacity = int(self.slots // (height * width))
        masks: list[torch.Tensor] = []
        for start, end in self.input_packing_signature[5]:
            active_channels = int(end - start)
            spatial = torch.zeros(
                (int(height * width), int(capacity)),
                dtype=torch.bool,
            )
            spatial[:, :active_channels] = True
            masks.append(spatial.reshape(-1))
        return torch.cat(masks)

    def compile(self, scheme: Any | None = None) -> dict[str, Any]:
        if self.compiled:
            raise RuntimeError("CIPS bootstrap refresh is already compiled")
        if self.cleaned:
            raise RuntimeError("a cleaned CIPS bootstrap refresh cannot be reused")
        scheme = scheme if scheme is not None else self.scheme
        if scheme is None:
            raise RuntimeError("set the Orion scheme before compiling the refresh")
        if int(scheme.params.get_slots()) != int(self.slots):
            raise ValueError("refresh slot count does not match the scheme")
        if int(scheme.params.get_max_level()) != int(self.output_level):
            raise ValueError("refresh output level must equal the scheme maximum level")

        Module.set_scheme(scheme)
        Module.set_margin(float(scheme.params.get_margin()))
        self.bootstrap.fit()
        if float(self.bootstrap.constant) != 0.0:
            raise RuntimeError("symmetric bootstrap bounds must have zero affine shift")
        self.bootstrap.compile()
        scheme.bootstrapper.generate_bootstrapper(
            int(self.bootstrap.bootstrap_slots)
        )
        self.he()
        self.compiled = True
        self.cleaned = False
        mask = self.bootstrap._bootstrap_prescale_active_mask
        self.compile_summary = {
            "operation": "identity_bootstrap_level_refresh",
            "input_level": int(self.input_level),
            "bootstrap_preprocess_output_level": int(self.input_level - 1),
            "output_level": int(self.output_level),
            "bootstrap_slots": int(self.bootstrap.bootstrap_slots),
            "ciphertext_group_count": int(self.group_count),
            "bootstrap_bound": float(self.bootstrap_bound),
            "bootstrap_prescale": float(self.bootstrap.prescale),
            "bootstrap_postscale": float(self.bootstrap.postscale),
            "bootstrap_constant": float(self.bootstrap.constant),
            "active_slot_count": int(mask.sum().item()),
            "packed_slot_count": int(mask.numel()),
            "inactive_slot_count": int(mask.numel() - mask.sum().item()),
            "packing_signature": list(self.input_packing_signature),
        }
        return dict(self.compile_summary)

    def _cipher_levels(self, value: CipherTensor) -> list[int]:
        return [
            int(value.backend.GetCiphertextLevel(int(ciphertext_id)))
            for ciphertext_id in value.ids
        ]

    def _validate_value(self, value: CipherTensor, *, level: int) -> None:
        if not isinstance(value, CipherTensor):
            raise TypeError("CIPS bootstrap refresh expects a CipherTensor")
        if value.scheme is not self.scheme:
            raise ValueError("CIPS bootstrap refresh received a different scheme")
        if (
            getattr(value, "_wpc_cips_packing_signature", None)
            != self.input_packing_signature
        ):
            raise ValueError("ciphertext does not have the required CIPS packing")
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
            return value
        if not self.compiled:
            raise RuntimeError("compile the CIPS bootstrap refresh before use")
        self._validate_value(value, level=int(self.input_level))
        started = time.perf_counter()
        refreshed = self.bootstrap(value)
        refreshed._wpc_cips_packing_signature = self.output_packing_signature
        try:
            self._validate_value(refreshed, level=int(self.output_level))
        except Exception:
            refreshed.release()
            raise
        runtime_records = list(self.bootstrap._bootstrap_runtime_profile)
        self.last_evaluation = {
            "input_level": int(self.input_level),
            "bootstrap_preprocess_output_level": int(self.input_level - 1),
            "output_level": int(self.output_level),
            "ciphertext_group_count": int(self.group_count),
            "total_s": float(time.perf_counter() - started),
            "packing_signature_preserved": bool(
                getattr(refreshed, "_wpc_cips_packing_signature", None)
                == self.output_packing_signature
            ),
            "clear_repack_or_encode_count": 0,
            "bootstrap_runtime_record": (
                dict(runtime_records[-1]) if runtime_records else {}
            ),
        }
        return refreshed

    def decrypt_unpack(self, value: CipherTensor) -> np.ndarray:
        if self.scheme is None:
            raise RuntimeError("CIPS bootstrap refresh has no scheme")
        self._validate_value(value, level=int(self.output_level))
        decoded = np.asarray(self.scheme.decode(self.scheme.decrypt(value)), dtype=np.float64)
        return unpack_cips_groups(decoded, self.output_packing_signature)

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


__all__ = ["WPCCIPSBootstrapRefresh"]
