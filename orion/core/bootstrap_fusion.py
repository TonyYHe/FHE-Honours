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
    return bool(
        _activation_fusion_capable(module)
        or _relu_output_fusion_capable(module)
        or _runtime_fusion_capable(module)
        or _add_fusion_capable(module)
    )


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
