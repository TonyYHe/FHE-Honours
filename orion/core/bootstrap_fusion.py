from __future__ import annotations

import math
import os
from typing import Any

import torch

_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}


def bootstrap_prescale_fusion_disabled() -> bool:
    value = os.environ.get("ORION_DISABLE_BOOTSTRAP_PRESCALE_FUSION", "0")
    return value.strip().lower() not in _FALSE_ENV_VALUES


def runtime_fhe_output_shape(module: Any) -> torch.Size | Any:
    """Return the materialized FHE output shape seen by a bootstrap hook."""

    materialization = str(getattr(module, "layout_policy_output_materialization", "") or "")
    if materialization in {"native_halo_stripe", "native_stripe", "channel_aligned_native_stripe"}:
        target_signature = tuple(getattr(module, "layout_policy_native_output_target_signature", ()) or ())
        scheme = getattr(module, "scheme", None)
        params = getattr(scheme, "params", None)
        get_slots = getattr(params, "get_slots", None)
        if target_signature and callable(get_slots):
            return torch.Size((int(len(target_signature)), int(get_slots())))

    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None) if runtime is not None else None
    for candidate in (
        executor,
        getattr(module, "layout_policy_add_runtime", None),
        getattr(module, "layout_policy_concat_runtime", None),
    ):
        get_shape = getattr(candidate, "runtime_fhe_output_shape", None)
        if callable(get_shape):
            shape = get_shape()
            if shape is not None:
                return shape
    return getattr(module, "fhe_output_shape", None)


def bootstrap_slots_for_shape(shape: Any, *, max_slots: int) -> int:
    elements = int(torch.Size(shape).numel())
    if int(elements) <= 0:
        return 0
    curr_slots = 2 ** math.ceil(math.log2(int(elements)))
    return int(min(int(max_slots), int(curr_slots)))


def module_bootstrap_slots(module: Any) -> int:
    scheme = getattr(module, "scheme", None)
    params = getattr(scheme, "params", None)
    get_slots = getattr(params, "get_slots", None)
    if not callable(get_slots):
        return 0
    shape = runtime_fhe_output_shape(module)
    if shape is None:
        return 0
    return bootstrap_slots_for_shape(shape, max_slots=int(get_slots()))


def module_bootstrap_ct_count(module: Any) -> int:
    scheme = getattr(module, "scheme", None)
    params = getattr(scheme, "params", None)
    get_slots = getattr(params, "get_slots", None)
    if not callable(get_slots):
        return 0
    shape = runtime_fhe_output_shape(module)
    if shape is None:
        return 0
    elements = int(torch.Size(shape).numel())
    if int(elements) <= 0:
        return 0
    return int(math.ceil(int(elements) / float(int(get_slots()))))


def module_uses_full_bootstrap_slots(module: Any) -> bool:
    scheme = getattr(module, "scheme", None)
    params = getattr(scheme, "params", None)
    get_slots = getattr(params, "get_slots", None)
    if not callable(get_slots):
        return False
    return int(module_bootstrap_slots(module)) == int(get_slots())


def _idx_chw_gap(channel: int, h: int, w: int, height: int, width: int, gap: int) -> int:
    g = max(1, int(gap))
    phases = int(g * g)
    packed_w = int(width) * int(g)
    group_block = int(height) * int(g) * int(packed_w)
    group = int(channel) // int(phases)
    phase = int(channel) % int(phases)
    phase_h = int(phase) // int(g)
    phase_w = int(phase) % int(g)
    return int(
        int(group) * int(group_block)
        + (int(h) * int(g) + int(phase_h)) * int(packed_w)
        + int(w) * int(g)
        + int(phase_w)
    )


def native_bootstrap_active_mask(module: Any) -> torch.Tensor | None:
    """Return per-CT active physical slots for native-stripe module outputs.

    Native stripe outputs can use the full CKKS slot count while still leaving
    physical row/channel slack inactive inside each ciphertext.  Bootstrap
    preprocess masks must therefore be slot-accurate instead of treating every
    slot in a full-slot ciphertext as semantically active.
    """

    raw_signature = tuple(getattr(module, "layout_policy_native_output_target_signature", ()) or ())
    if not raw_signature:
        return None

    scheme = getattr(module, "scheme", None)
    params = getattr(scheme, "params", None)
    get_slots = getattr(params, "get_slots", None)
    if not callable(get_slots):
        return None
    slots = int(get_slots())
    if int(slots) <= 0:
        return None

    output_shape = tuple(int(value) for value in tuple(getattr(module, "output_shape", ()) or ()))
    if len(output_shape) < 4:
        return None
    width = int(output_shape[3])
    if int(width) <= 0:
        return None
    gap = max(1, int(getattr(module, "output_gap", getattr(module, "input_gap", 1)) or 1))
    signature = tuple(tuple(int(value) for value in tuple(raw)) for raw in raw_signature)
    cache_key = (signature, tuple(int(value) for value in output_shape), int(gap), int(slots))
    cached_key = getattr(module, "_bootstrap_prescale_active_mask_cache_key", None)
    cached_mask = getattr(module, "_bootstrap_prescale_active_mask_cache", None)
    if cached_key == cache_key and cached_mask is not None:
        return cached_mask

    masks: list[torch.Tensor] = []
    for raw in signature:
        if len(tuple(raw)) != 4:
            return None
        h_start, h_end, _channel_start, channel_count = (int(value) for value in tuple(raw))
        height = max(0, int(h_end) - int(h_start))
        channels = max(0, int(channel_count))
        mask = torch.zeros((int(slots),), dtype=torch.bool)
        for local_channel in range(int(channels)):
            for local_h in range(int(height)):
                for w_index in range(int(width)):
                    slot = _idx_chw_gap(
                        int(local_channel),
                        int(local_h),
                        int(w_index),
                        int(height),
                        int(width),
                        int(gap),
                    )
                    if int(slot) >= int(slots):
                        return None
                    mask[int(slot)] = True
        masks.append(mask)
    if not masks:
        return None
    result = torch.stack(masks)
    setattr(module, "_bootstrap_prescale_active_mask_cache_key", cache_key)
    setattr(module, "_bootstrap_prescale_active_mask_cache", result)
    return result


def native_bootstrap_has_inactive_slots(module: Any) -> bool:
    mask = native_bootstrap_active_mask(module)
    if mask is None or int(mask.numel()) <= 0:
        return False
    cache_key = getattr(module, "_bootstrap_prescale_active_mask_cache_key", None)
    cached_key = getattr(module, "_bootstrap_prescale_has_inactive_slots_cache_key", None)
    cached_value = getattr(module, "_bootstrap_prescale_has_inactive_slots_cache", None)
    if cached_key == cache_key and cached_value is not None:
        return bool(cached_value)
    value = bool(not bool(mask.all().item()))
    setattr(module, "_bootstrap_prescale_has_inactive_slots_cache_key", cache_key)
    setattr(module, "_bootstrap_prescale_has_inactive_slots_cache", bool(value))
    return bool(value)


def _scalar_polynomial_fusion_target(module: Any) -> bool:
    return bool(hasattr(module, "coeffs") and not hasattr(module, "on_weight"))


def _activation_fusion_capable(module: Any) -> bool:
    class_name = type(module).__name__
    return bool(
        class_name in {"Chebyshev", "ELU", "GELU", "SiLU", "Sigmoid", "SELU", "Softplus", "Mish"}
        and hasattr(module, "coeffs")
    )


def _relu_parent(module: Any) -> Any | None:
    parent_ref = getattr(module, "_relu_parent_ref", None)
    if not callable(parent_ref):
        return None
    return parent_ref()


def _relu_output_fusion_capable(module: Any) -> bool:
    if type(module).__name__ != "Mult":
        return False
    if str(getattr(module, "_relu_role", "")) != "output":
        return False
    parent = _relu_parent(module)
    sign = getattr(parent, "sign", None)
    return bool(type(parent).__name__ == "ReLU" and getattr(sign, "acts", None))


def _runtime_fusion_capable(module: Any) -> bool:
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None) if runtime is not None else None
    capable = getattr(executor, "bootstrap_prescale_fusion_capable", None)
    if callable(capable):
        return bool(capable())
    return bool(getattr(executor, "bootstrap_prescale_fusion_capable", False))


def _add_fusion_capable(module: Any) -> bool:
    runtime = getattr(module, "layout_policy_add_runtime", None)
    capable = getattr(runtime, "bootstrap_prescale_fusion_capable", None)
    if callable(capable):
        return bool(capable())
    return bool(getattr(runtime, "bootstrap_prescale_fusion_capable", False))


def bootstrap_prescale_fusion_supported(module: Any) -> bool:
    """Whether this module can absorb Bootstrap.forward's pre-affine multiply.

    The current implementation intentionally requires full-slot bootstrapping.
    Sparse bootstrap still needs Bootstrap.forward's mask vector to zero unused
    slots, while the full-slot U22 cases can fold the affine into producers.
    """

    if bootstrap_prescale_fusion_disabled():
        return False
    if module is None or not module_uses_full_bootstrap_slots(module):
        return False
    if native_bootstrap_has_inactive_slots(module) and (
        _scalar_polynomial_fusion_target(module) or _relu_output_fusion_capable(module)
    ):
        return False
    return bool(_runtime_fusion_capable(module) or _add_fusion_capable(module))


def bootstrap_prescale_affine(bootstrapper: Any) -> dict[str, float]:
    scale = float(getattr(bootstrapper, "prescale", 1.0))
    constant = float(getattr(bootstrapper, "constant", 0.0))
    return {
        "scale": float(scale),
        "bias": float(scale * constant),
    }


def install_bootstrap_prescale_fusion(module: Any, bootstrapper: Any) -> bool:
    if not bootstrap_prescale_fusion_supported(module):
        return False
    affine = bootstrap_prescale_affine(bootstrapper)
    if _relu_output_fusion_capable(module):
        parent = _relu_parent(module)
        last_sign_act = getattr(parent.sign, "acts")[-1]
        setattr(last_sign_act, "_bootstrap_output_scale_fusion", float(affine["scale"]))
        setattr(module, "_bootstrap_output_bias_fusion", float(affine["bias"]))
    else:
        setattr(module, "_bootstrap_prescale_fusion", dict(affine))
        for candidate in (
            getattr(getattr(module, "region_runtime", None), "executor", None),
            getattr(module, "layout_policy_add_runtime", None),
            getattr(module, "layout_policy_concat_runtime", None),
        ):
            if candidate is not None:
                setattr(candidate, "_bootstrap_prescale_fusion", dict(affine))
    setattr(bootstrapper, "preprocess_fused", True)
    setattr(bootstrapper, "preprocess_fusion_kind", "producer_affine")
    return True
