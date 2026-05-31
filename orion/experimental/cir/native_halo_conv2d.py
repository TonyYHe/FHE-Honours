from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable, Literal
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time

import numpy as np
import torch

from orion.backend.python.tensors import CipherTensor
from orion.nn.unified_transform import UnifiedTransformGroup

from .r34_orion_same_shape import (
    RING_SLOT_COUNT,
    _add_plaintext_for_add,
    _align_ciphertexts_for_add,
    _bsgs_rotation_sets,
    _coalesce_native_rows,
    _encode_plaintext_for_add,
    _idx_chw_gap,
    _native_best_common_bsgs,
    _rescale_cipher_tensor,
    _unified_output_fusion_enabled,
)

_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}
_CPP_DIAG_BUILDER_FALLBACK = object()
_NATIVE_BSGS_CACHE: dict[tuple[int, tuple[tuple[int, ...], ...]], tuple[int, int, int, int]] = {}
_NATIVE_BSGS_ROTATION_SETS_CACHE: dict[tuple[int, int, tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]] = {}
_NATIVE_DIAG_INDICES_CACHE: dict[tuple[Any, ...], tuple[int, ...]] = {}
_COMPACT_OUTPUT_DIAG_SET_CACHE: dict[tuple[Any, ...], tuple[tuple[int, tuple[int, ...]], ...]] = {}
_PER_STRIPE_FOLD_STRIPES_CACHE: dict[tuple[Any, ...], tuple["NativeHaloStripe", ...]] = {}
_NATIVE_BSGS_CACHE_MAX_ENTRIES = 20000
_NATIVE_DIAG_CACHE_MAX_ENTRIES = 50000
_MISSING_ATTR = object()


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(str(name))
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in _FALSE_ENV_VALUES


def _provider_diag_builder_enabled() -> bool:
    return bool(_env_enabled("ORION_CPP_DIAG_BUILDER") and _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER", True))


def _provider_diag_builder_strict() -> bool:
    return bool(_env_enabled("ORION_CPP_DIAG_BUILDER_STRICT"))


def _provider_diag_builder_shadow() -> bool:
    return bool(_env_enabled("ORION_CPP_DIAG_BUILDER_SHADOW"))


def _provider_diag_build_workers(item_count: int) -> int:
    if int(item_count) <= 1:
        return 1
    raw = (
        os.environ.get("ORION_PROVIDER_DIAG_BUILD_WORKERS")
        or os.environ.get("ORION_CPP_DIAG_BUILDER_PROVIDER_WORKERS")
        or ""
    )
    if not raw:
        return 1
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        requested = 1
    if int(requested) <= 1:
        return 1
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(int(item_count), int(cpu), int(requested)))


def _record_provider_diag_builder_metadata(target: Any, metadata: dict[str, Any] | None) -> None:
    if not isinstance(metadata, dict) or not metadata:
        return
    timing = getattr(target, "last_runtime_timing", None)
    if not isinstance(timing, dict):
        return
    for key, value in metadata.items():
        if key in {
            "diag_builder_build_s",
            "diag_builder_wall_s",
            "diag_builder_shadow_s",
            "diag_builder_payload_count",
            "diag_builder_fallback_count",
        }:
            timing[str(key)] = float(timing.get(str(key), 0.0) or 0.0) + float(value or 0.0)
        else:
            timing[str(key)] = value


_GROUP_COMPILE_PROFILE_NUMERIC_KEYS = (
    "detect_complex_s",
    "flatten_s",
    "backend_pointer_pack_s",
    "backend_generate_s",
    "backend_generate_calls",
    "record_keys_s",
    "rotation_key_compile_s",
    "rotation_key_required_count",
    "rotation_key_generated_count",
    "rotation_key_cached_count",
    "save_unload_s",
    "compile_gc_s",
    "compile_gc_count",
    "prepare_shared_cache_s",
    "total_s",
    "transform_count",
    "stream_batch_limit",
    "stream_batch_count",
    "diag_index_count",
    "diag_data_count",
    "payload_bytes",
)


def _record_group_compile_profile(target: Any, group: Any) -> None:
    profile = getattr(group, "last_compile_profile", None)
    if not isinstance(profile, dict) or not profile:
        return
    timing = getattr(target, "last_runtime_timing", None)
    if not isinstance(timing, dict):
        return
    timing["group_compile_profile_count"] = float(
        timing.get("group_compile_profile_count", 0.0) or 0.0
    ) + 1.0
    for key in _GROUP_COMPILE_PROFILE_NUMERIC_KEYS:
        value = profile.get(str(key), 0.0)
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            timing[f"group_compile_{key}"] = float(
                timing.get(f"group_compile_{key}", 0.0) or 0.0
            ) + float(value)
    mode = profile.get("mode")
    if mode is not None:
        modes = timing.setdefault("group_compile_modes", {})
        if isinstance(modes, dict):
            modes[str(mode)] = int(modes.get(str(mode), 0) or 0) + 1


def _canonical_cache_key(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((str(key), _canonical_cache_key(val)) for key, val in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_cache_key(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _diag_set_key(diag_indices: set[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in diag_indices))


def _native_diag_cache_guard() -> None:
    if len(_NATIVE_DIAG_INDICES_CACHE) + len(_COMPACT_OUTPUT_DIAG_SET_CACHE) > int(_NATIVE_DIAG_CACHE_MAX_ENTRIES):
        _NATIVE_DIAG_INDICES_CACHE.clear()
        _COMPACT_OUTPUT_DIAG_SET_CACHE.clear()


def _native_spec_structural_cache_key(spec: "NativeHaloConv2DSpec") -> tuple[tuple[str, Any], ...]:
    payload = dict(spec.to_dict())
    payload.pop("family_label", None)
    return _canonical_cache_key(payload)


def _native_stripe_cache_key(stripe: "NativeHaloStripe") -> tuple[tuple[str, Any], ...]:
    return _canonical_cache_key(stripe.to_dict())


def _native_plan_structural_cache_key(plan: "NativeHaloConv2DPlan") -> tuple[tuple[str, Any], ...]:
    payload = dict(plan.to_dict())
    spec_payload = dict(payload.get("spec", {}) or {})
    spec_payload.pop("family_label", None)
    payload["spec"] = spec_payload
    return _canonical_cache_key(payload)


def _cached_native_best_common_bsgs(entries: tuple[set[int], ...], *, slots: int) -> tuple[int, int, int, int]:
    key = (int(slots), tuple(_diag_set_key(entry) for entry in entries))
    cached = _NATIVE_BSGS_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_NATIVE_BSGS_CACHE) > int(_NATIVE_BSGS_CACHE_MAX_ENTRIES):
        _NATIVE_BSGS_CACHE.clear()
        _NATIVE_BSGS_ROTATION_SETS_CACHE.clear()
    result = _native_best_common_bsgs(entries, slots=int(slots))
    _NATIVE_BSGS_CACHE[key] = result
    return result


def _cached_bsgs_rotation_sets(diag_indices: set[int], *, slots: int, n1: int) -> tuple[set[int], set[int]]:
    diag_key = _diag_set_key(diag_indices)
    key = (int(slots), int(n1), diag_key)
    cached = _NATIVE_BSGS_ROTATION_SETS_CACHE.get(key)
    if cached is not None:
        baby, giant = cached
        return set(int(value) for value in baby), set(int(value) for value in giant)
    if len(_NATIVE_BSGS_ROTATION_SETS_CACHE) > int(_NATIVE_BSGS_CACHE_MAX_ENTRIES):
        _NATIVE_BSGS_ROTATION_SETS_CACHE.clear()
    baby_set, giant_set = _bsgs_rotation_sets(set(int(value) for value in diag_indices), slots=int(slots), n1=int(n1))
    _NATIVE_BSGS_ROTATION_SETS_CACHE[key] = (
        tuple(sorted(int(value) for value in baby_set)),
        tuple(sorted(int(value) for value in giant_set)),
    )
    return baby_set, giant_set


def _ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def _single_slot_layer_cache_enabled_for_scheme(scheme: Any) -> bool:
    evaluator = getattr(scheme, "lt_evaluator", None)
    enabled = getattr(evaluator, "single_slot_layer_cache_enabled", None)
    return callable(enabled) and bool(enabled())


def _pair(value: Any, *, default: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    if value is None:
        return tuple(default)
    if isinstance(value, int):
        return (int(value), int(value))
    values = tuple(int(item) for item in value)
    if len(values) == 1:
        return (int(values[0]), int(values[0]))
    return (int(values[0]), int(values[1]))


def _safe_shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value)
    except Exception:
        return ()


def _phase_count(gap: int) -> int:
    return max(1, int(gap) * int(gap))


def _packed_active_slots(channel_count: int, height: int, width: int, gap: int) -> int:
    groups = _ceil_div(int(channel_count), _phase_count(int(gap)))
    return int(groups) * int(height) * int(gap) * int(width) * int(gap)


def _compact_ct_count(channel_count: int, height: int, width: int, gap: int, slots: int) -> int:
    active_slots = _packed_active_slots(int(channel_count), int(height), int(width), int(gap))
    return max(1, _ceil_div(int(active_slots), int(slots)))


def _compact_output_fhe_shape_for_spec(spec: "NativeHaloConv2DSpec") -> torch.Size:
    gap = max(1, int(spec.gap_out))
    on_channels = _ceil_div(int(spec.c_out), _phase_count(int(gap)))
    return torch.Size(
        (
            1,
            int(on_channels),
            int(_spec_physical_output_h(spec)) * int(gap),
            int(spec.w_out) * int(gap),
        )
    )


def _layout_top_beta(layout: dict[str, Any], *, default: int = 0) -> int:
    return max(0, int(layout.get("top_beta", layout.get("alpha", int(default))) or 0))


def _layout_bottom_beta(layout: dict[str, Any], *, default: int = 0) -> int:
    return max(0, int(layout.get("bottom_beta", layout.get("beta", int(default))) or 0))


def _layout_physical_top_beta(layout: dict[str, Any], *, default: int = 0) -> int:
    return max(
        0,
        int(
            layout.get(
                "physical_top_beta",
                layout.get("top_beta", layout.get("alpha", int(default))),
            )
            or 0
        ),
    )


def _layout_physical_bottom_beta(layout: dict[str, Any], *, default: int = 0) -> int:
    return max(
        0,
        int(
            layout.get(
                "physical_bottom_beta",
                layout.get("bottom_beta", layout.get("beta", int(default))),
            )
            or 0
        ),
    )


def _spec_has_semantic_output_halo(spec: NativeHaloConv2DSpec) -> bool:
    return bool(int(spec.output_top_beta) > 0 or int(spec.output_bottom_beta) > 0)


def _spec_has_physical_output_halo(spec: NativeHaloConv2DSpec) -> bool:
    return bool(int(spec.output_physical_top_beta) > 0 or int(spec.output_physical_bottom_beta) > 0)


def _slot_indices(channel_count: int, height: int, width: int, gap: int) -> torch.Tensor:
    g = max(1, int(gap))
    phases = int(g * g)
    packed_w = int(width) * int(g)
    group_block = int(height) * int(g) * int(packed_w)
    hs = torch.arange(int(height), dtype=torch.int64)[:, None]
    ws = torch.arange(int(width), dtype=torch.int64)[None, :]
    out = torch.empty((int(channel_count), int(height), int(width)), dtype=torch.int64)
    for channel in range(int(channel_count)):
        group = int(channel) // int(phases)
        phase = int(channel) % int(phases)
        phase_h = int(phase) // int(g)
        phase_w = int(phase) % int(g)
        out[int(channel)] = (
            int(group) * int(group_block)
            + (hs * int(g) + int(phase_h)) * int(packed_w)
            + ws * int(g)
            + int(phase_w)
        )
    return out


def _idx_chw_gap_channels(
    channels: torch.Tensor,
    *,
    h: int,
    w: int,
    height: int,
    width: int,
    gap: int,
) -> torch.Tensor:
    g = max(1, int(gap))
    phases = int(g * g)
    packed_w = int(width) * int(g)
    group_block = int(height) * int(g) * int(packed_w)
    values = channels.to(dtype=torch.int64)
    group = torch.div(values, int(phases), rounding_mode="floor")
    phase = torch.remainder(values, int(phases))
    phase_h = torch.div(phase, int(g), rounding_mode="floor")
    phase_w = torch.remainder(phase, int(g))
    return (
        group * int(group_block)
        + (int(h) * int(g) + phase_h) * int(packed_w)
        + int(w) * int(g)
        + phase_w
    ).to(dtype=torch.int64)


def _idx_chw_gap_channel_positions(
    channels: torch.Tensor,
    *,
    h: torch.Tensor,
    w: torch.Tensor,
    height: int,
    width: int,
    gap: int,
) -> torch.Tensor:
    g = max(1, int(gap))
    phases = int(g * g)
    packed_w = int(width) * int(g)
    group_block = int(height) * int(g) * int(packed_w)
    values = channels.to(dtype=torch.int64)[:, None]
    h_values = h.to(dtype=torch.int64)[None, :]
    w_values = w.to(dtype=torch.int64)[None, :]
    group = torch.div(values, int(phases), rounding_mode="floor")
    phase = torch.remainder(values, int(phases))
    phase_h = torch.div(phase, int(g), rounding_mode="floor")
    phase_w = torch.remainder(phase, int(g))
    return (
        group * int(group_block)
        + (h_values * int(g) + phase_h) * int(packed_w)
        + w_values * int(g)
        + phase_w
    ).to(dtype=torch.int64)


def _channel_base_offsets_chw_gap(
    channel_start: int,
    channel_count: int,
    *,
    height: int,
    width: int,
    gap: int,
) -> np.ndarray:
    count = max(0, int(channel_count))
    if int(count) <= 0:
        return np.empty((0,), dtype=np.int64)
    g = max(1, int(gap))
    phases = int(g * g)
    packed_w = int(width) * int(g)
    group_block = int(height) * int(g) * int(packed_w)
    channels = np.arange(int(channel_start), int(channel_start) + int(count), dtype=np.int64)
    groups = channels // int(phases)
    phase = channels % int(phases)
    phase_h = phase // int(g)
    phase_w = phase % int(g)
    return (
        groups * int(group_block)
        + phase_h * int(packed_w)
        + phase_w
    ).astype(np.int64, copy=False)


def _mod_sum_values(left: set[int] | np.ndarray, right: set[int] | np.ndarray, *, slots: int) -> set[int]:
    slots = max(1, int(slots))
    if isinstance(left, np.ndarray):
        left_values = left.astype(np.int64, copy=False).reshape(-1)
    else:
        left_values = np.asarray(tuple(int(value) for value in left), dtype=np.int64)
    if isinstance(right, np.ndarray):
        right_values = right.astype(np.int64, copy=False).reshape(-1)
    else:
        right_values = np.asarray(tuple(int(value) for value in right), dtype=np.int64)
    if int(left_values.size) == 0 or int(right_values.size) == 0:
        return set()
    left_values = np.unique(np.remainder(left_values, int(slots)))
    right_values = np.unique(np.remainder(right_values, int(slots)))
    if int(left_values.size) > int(right_values.size):
        left_values, right_values = right_values, left_values
    mask = np.zeros((int(slots),), dtype=np.bool_)
    chunk = max(1, min(4096, int(left_values.size)))
    for start in range(0, int(left_values.size), int(chunk)):
        values = np.remainder(right_values[None, :] + left_values[start : start + int(chunk), None], int(slots))
        mask[values.reshape(-1)] = True
    return set(int(value) for value in np.flatnonzero(mask).tolist())


def _mod_difference_values(left: np.ndarray, right: np.ndarray, *, slots: int) -> set[int]:
    slots = max(1, int(slots))
    if int(left.size) == 0 or int(right.size) == 0:
        return set()
    left_values = np.unique(np.remainder(left.astype(np.int64, copy=False).reshape(-1), int(slots)))
    right_values = np.unique(np.remainder(right.astype(np.int64, copy=False).reshape(-1), int(slots)))
    mask = np.zeros((int(slots),), dtype=np.bool_)
    chunk = max(1, min(4096, int(left_values.size)))
    for start in range(0, int(left_values.size), int(chunk)):
        values = np.remainder(left_values[start : start + int(chunk), None] - right_values[None, :], int(slots))
        mask[values.reshape(-1)] = True
    return set(int(value) for value in np.flatnonzero(mask).tolist())


def _native_diag_indices_closed_form(
    spec: "NativeHaloConv2DSpec",
    stripe: "NativeHaloStripe",
    *,
    source_channel_count: int,
    target_channel_count: int,
) -> set[int]:
    slots = int(spec.slot_count)
    source_offsets = _channel_base_offsets_chw_gap(
        0,
        int(source_channel_count),
        height=int(stripe.source_h),
        width=int(spec.w_in),
        gap=int(spec.gap_in),
    )
    target_offsets = _channel_base_offsets_chw_gap(
        0,
        int(target_channel_count),
        height=int(stripe.target_h),
        width=int(spec.w_out),
        gap=int(spec.gap_out),
    )
    channel_shifts = _mod_difference_values(source_offsets, target_offsets, slots=int(slots))
    if not channel_shifts:
        return set()

    source_packed_w = int(spec.w_in) * max(1, int(spec.gap_in))
    target_packed_w = int(spec.w_out) * max(1, int(spec.gap_out))
    spatial_shifts: set[int] = set()
    out_h_values = range(int(stripe.target_h_start), int(stripe.target_h_end))
    out_w_values = range(int(spec.w_out))
    for kh in range(int(spec.kernel)):
        h_shifts: set[int] = set()
        for out_h in out_h_values:
            in_h = int(out_h) * int(spec.stride) - int(spec.pad) + int(kh) * int(spec.dilation)
            source_local_h = int(in_h) - int(stripe.source_h_start)
            if (
                int(in_h) < int(spec.input_h_min)
                or int(in_h) >= int(spec.input_h_max)
                or int(source_local_h) < 0
                or int(source_local_h) >= int(stripe.source_h)
            ):
                continue
            target_local_h = int(out_h) - int(stripe.target_h_start)
            h_shifts.add(
                int(
                    int(source_local_h) * max(1, int(spec.gap_in)) * int(source_packed_w)
                    - int(target_local_h) * max(1, int(spec.gap_out)) * int(target_packed_w)
                )
            )
        if not h_shifts:
            continue
        for kw in range(int(spec.kernel)):
            w_shifts: set[int] = set()
            for out_w in out_w_values:
                in_w = int(out_w) * int(spec.stride) - int(spec.pad) + int(kw) * int(spec.dilation)
                if int(in_w) < 0 or int(in_w) >= int(spec.w_in):
                    continue
                w_shifts.add(int(int(in_w) * max(1, int(spec.gap_in)) - int(out_w) * max(1, int(spec.gap_out))))
            if not w_shifts:
                continue
            spatial_shifts.update(_mod_sum_values(h_shifts, w_shifts, slots=int(slots)))
    shifts = _mod_sum_values(channel_shifts, spatial_shifts, slots=int(slots))
    shifts.discard(0)
    return shifts


def _native_halo_build_pair_chunk_limit() -> int:
    raw = __import__("os").environ.get("ORION_NATIVE_HALO_BUILD_PAIR_CHUNK")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 8_000_000


def _materialized_output_source_h(output_h: torch.Tensor | int, *, h_out: int, output_top_beta: int, output_bottom_beta: int):
    if isinstance(output_h, torch.Tensor):
        values = output_h.clone().to(dtype=torch.int64)
        top = values < 0
        bottom = values >= int(h_out)
        values[top] = values[top] + int(output_top_beta)
        values[bottom] = values[bottom] - int(output_bottom_beta)
        return values.clamp(0, max(0, int(h_out) - 1))
    value = int(output_h)
    if int(value) < 0:
        value += int(output_top_beta)
    elif int(value) >= int(h_out):
        value -= int(output_bottom_beta)
    return min(max(int(value), 0), max(0, int(h_out) - 1))


def _spec_physical_output_h(spec: NativeHaloConv2DSpec) -> int:
    return int(spec.h_out) + max(0, int(spec.output_physical_top_beta or 0)) + max(
        0,
        int(spec.output_physical_bottom_beta or 0),
    )


def _spec_physical_output_h_positions(spec: NativeHaloConv2DSpec, output_h: torch.Tensor | int):
    physical_top = max(0, int(spec.output_physical_top_beta or 0))
    physical_bottom = max(0, int(spec.output_physical_bottom_beta or 0))
    if int(physical_top) >= int(spec.output_top_beta) and int(physical_bottom) >= int(spec.output_bottom_beta):
        return output_h + int(physical_top) if isinstance(output_h, torch.Tensor) else int(output_h) + int(physical_top)
    return output_h + int(physical_top) if isinstance(output_h, torch.Tensor) else int(output_h) + int(physical_top)


def _spec_physical_output_h_valid(spec: NativeHaloConv2DSpec, output_h: torch.Tensor | int):
    physical_top = max(0, int(spec.output_physical_top_beta or 0))
    physical_bottom = max(0, int(spec.output_physical_bottom_beta or 0))
    if isinstance(output_h, torch.Tensor):
        return (output_h >= -int(physical_top)) & (output_h < int(spec.h_out) + int(physical_bottom))
    value = int(output_h)
    return bool(value >= -int(physical_top) and value < int(spec.h_out) + int(physical_bottom))


def _heuristic_channel_tile(channel_count: int, gap: int) -> int:
    """Deterministic native-stripe channel tile.

    Use one natural phase group for gapped layouts.  For gap-1 layouts, keep
    each channel in its own native stripe tile.
    """

    phase = _phase_count(int(gap))
    if int(phase) == 1:
        return 1
    return int(phase)


def _source_h_range_for_target(
    *,
    target_h_start: int,
    target_h_end: int,
    input_h_min: int,
    input_h_max: int,
    kernel: int,
    stride: int,
    pad: int,
    dilation: int,
) -> tuple[int, int]:
    source_start = int(target_h_start) * int(stride) - int(pad)
    source_end = (int(target_h_end) - 1) * int(stride) - int(pad) + (int(kernel) - 1) * int(dilation)
    start = max(int(input_h_min), int(source_start))
    end = min(int(input_h_max), int(source_end) + 1)
    if int(end) < int(start):
        end = int(start)
    return int(start), int(end)


def _extend_h_range_to_length(
    *,
    required_start: int,
    required_end: int,
    desired_len: int,
    lower: int,
    upper: int,
) -> tuple[int, int]:
    start = int(required_start)
    end = int(required_end)
    desired = min(int(upper - lower), max(int(end - start), int(desired_len)))
    extra = int(desired - (end - start))
    if int(extra) <= 0:
        return int(start), int(end)
    left = min(int(start - lower), int(extra // 2))
    start -= int(left)
    extra -= int(left)
    right = min(int(upper - end), int(extra))
    end += int(right)
    extra -= int(right)
    if int(extra) > 0:
        left = min(int(start - lower), int(extra))
        start -= int(left)
    return int(start), int(end)


def _source_h_range_with_physical_input_halo(
    spec: NativeHaloConv2DSpec,
    *,
    required_start: int,
    required_end: int,
) -> tuple[int, int]:
    start = int(required_start)
    end = int(required_end)
    if int(start) <= 0:
        start = min(int(start), -max(0, int(spec.input_physical_top_beta or 0)))
    if int(end) >= int(spec.h_in):
        end = max(int(end), int(spec.h_in) + max(0, int(spec.input_physical_bottom_beta or 0)))
    return max(int(spec.input_h_min), int(start)), min(int(spec.input_h_max), int(end))


@dataclass(frozen=True)
class NativeHaloConv2DSpec:
    family_label: str
    c_in: int
    h_in: int
    w_in: int
    c_out: int
    h_out: int
    w_out: int
    gap_in: int
    gap_out: int
    kernel: int
    stride: int
    pad: int
    dilation: int = 1
    groups: int = 1
    slot_count: int = RING_SLOT_COUNT
    input_top_beta: int = 0
    input_bottom_beta: int = 0
    output_top_beta: int = 0
    output_bottom_beta: int = 0
    input_physical_top_beta: int | None = None
    input_physical_bottom_beta: int | None = None
    output_physical_top_beta: int | None = None
    output_physical_bottom_beta: int | None = None

    def __post_init__(self) -> None:
        if self.input_physical_top_beta is None:
            object.__setattr__(self, "input_physical_top_beta", int(self.input_top_beta))
        if self.input_physical_bottom_beta is None:
            object.__setattr__(self, "input_physical_bottom_beta", int(self.input_bottom_beta))
        if self.output_physical_top_beta is None:
            object.__setattr__(self, "output_physical_top_beta", int(self.output_top_beta))
        if self.output_physical_bottom_beta is None:
            object.__setattr__(self, "output_physical_bottom_beta", int(self.output_bottom_beta))

    @property
    def input_h_min(self) -> int:
        return -max(0, int(self.input_top_beta))

    @property
    def input_h_max(self) -> int:
        return int(self.h_in) + max(0, int(self.input_bottom_beta))

    @property
    def output_h_min(self) -> int:
        return -max(0, int(self.output_top_beta))

    @property
    def output_h_max(self) -> int:
        return int(self.h_out) + max(0, int(self.output_bottom_beta))

    @property
    def weight_shape(self) -> tuple[int, int, int, int]:
        return (
            int(self.c_out),
            int(self.c_in) // int(self.groups),
            int(self.kernel),
            int(self.kernel),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_label": str(self.family_label),
            "c_in": int(self.c_in),
            "h_in": int(self.h_in),
            "w_in": int(self.w_in),
            "c_out": int(self.c_out),
            "h_out": int(self.h_out),
            "w_out": int(self.w_out),
            "gap_in": int(self.gap_in),
            "gap_out": int(self.gap_out),
            "kernel": int(self.kernel),
            "stride": int(self.stride),
            "pad": int(self.pad),
            "dilation": int(self.dilation),
            "groups": int(self.groups),
            "slot_count": int(self.slot_count),
            "input_top_beta": int(self.input_top_beta),
            "input_bottom_beta": int(self.input_bottom_beta),
            "output_top_beta": int(self.output_top_beta),
            "output_bottom_beta": int(self.output_bottom_beta),
            "input_physical_top_beta": int(self.input_physical_top_beta or 0),
            "input_physical_bottom_beta": int(self.input_physical_bottom_beta or 0),
            "output_physical_top_beta": int(self.output_physical_top_beta or 0),
            "output_physical_bottom_beta": int(self.output_physical_bottom_beta or 0),
        }


@dataclass(frozen=True)
class NativeHaloStripe:
    index: int
    target_h_start: int
    target_h_end: int
    source_h_start: int
    source_h_end: int
    source_channel_tile: int = 0
    target_channel_tile: int = 0

    @property
    def source_h(self) -> int:
        return int(self.source_h_end) - int(self.source_h_start)

    @property
    def target_h(self) -> int:
        return int(self.target_h_end) - int(self.target_h_start)

    def to_dict(self) -> dict[str, int]:
        return {
            "index": int(self.index),
            "target_h_start": int(self.target_h_start),
            "target_h_end": int(self.target_h_end),
            "source_h_start": int(self.source_h_start),
            "source_h_end": int(self.source_h_end),
            "stored_source_rows": int(self.source_h),
            "target_rows": int(self.target_h),
            "halo_redundant_rows": int(self.source_h - self.target_h),
            "source_channel_tile": int(self.source_channel_tile or 0),
            "target_channel_tile": int(self.target_channel_tile or 0),
        }


@dataclass(frozen=True)
class NativeHaloConv2DPlan:
    spec: NativeHaloConv2DSpec
    source_channel_tile: int
    target_channel_tile: int
    stripes: tuple[NativeHaloStripe, ...]
    program_diagonal_counts: tuple[int, ...]
    program_rotation_counts: tuple[int, ...]
    group_n1s: tuple[int, ...]
    group_shared_rotations: tuple[int, ...]
    group_baby_rotations: tuple[int, ...]
    group_giant_rotations: tuple[int, ...]
    channel_fold_mode: str = "heuristic"

    @property
    def source_channel_group_count(self) -> int:
        return max(int(value) for value in self.source_channel_group_counts)

    @property
    def target_channel_group_count(self) -> int:
        return max(int(value) for value in self.target_channel_group_counts)

    def source_tile_for_stripe(self, stripe: NativeHaloStripe) -> int:
        return int(stripe.source_channel_tile or self.source_channel_tile)

    def target_tile_for_stripe(self, stripe: NativeHaloStripe) -> int:
        return int(stripe.target_channel_tile or self.target_channel_tile)

    def source_group_count_for_stripe(self, stripe: NativeHaloStripe) -> int:
        return _ceil_div(int(self.spec.c_in), int(self.source_tile_for_stripe(stripe)))

    def target_group_count_for_stripe(self, stripe: NativeHaloStripe) -> int:
        return _ceil_div(int(self.spec.c_out), int(self.target_tile_for_stripe(stripe)))

    @property
    def source_channel_group_counts(self) -> tuple[int, ...]:
        return tuple(int(self.source_group_count_for_stripe(stripe)) for stripe in self.stripes)

    @property
    def target_channel_group_counts(self) -> tuple[int, ...]:
        return tuple(int(self.target_group_count_for_stripe(stripe)) for stripe in self.stripes)

    @property
    def source_stripe_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        total = 0
        for count in self.source_channel_group_counts:
            offsets.append(int(total))
            total += int(count)
        return tuple(offsets)

    @property
    def target_stripe_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        total = 0
        for count in self.target_channel_group_counts:
            offsets.append(int(total))
            total += int(count)
        return tuple(offsets)

    def source_block_index(self, stripe: NativeHaloStripe, group: int) -> int:
        return int(self.source_stripe_offsets[int(stripe.index)] + int(group))

    def target_block_index(self, stripe: NativeHaloStripe, group: int) -> int:
        return int(self.target_stripe_offsets[int(stripe.index)] + int(group))

    def target_stripe_and_group_for_block(self, block_index: int) -> tuple[NativeHaloStripe, int] | None:
        block_index = int(block_index)
        offsets = self.target_stripe_offsets
        counts = self.target_channel_group_counts
        for stripe, offset, count in zip(self.stripes, offsets, counts, strict=True):
            if int(offset) <= int(block_index) < int(offset) + int(count):
                return stripe, int(block_index - int(offset))
        return None

    @property
    def input_ct_count(self) -> int:
        return int(sum(int(value) for value in self.source_channel_group_counts))

    @property
    def output_ct_count(self) -> int:
        if not _spec_has_physical_output_halo(self.spec):
            return _compact_ct_count(
                int(self.spec.c_out),
                _spec_physical_output_h(self.spec),
                int(self.spec.w_out),
                int(self.spec.gap_out),
                int(self.spec.slot_count),
            )
        return int(sum(int(value) for value in self.target_channel_group_counts))

    @property
    def submatrix_program_count(self) -> int:
        return int(len(self.program_diagonal_counts))

    @property
    def sharing_group_count(self) -> int:
        return int(len(self.group_shared_rotations))

    @property
    def c_only_rotations(self) -> int:
        return int(sum(int(value) for value in self.program_rotation_counts))

    @property
    def cb_shared_rotations(self) -> int:
        return int(sum(int(value) for value in self.group_shared_rotations))

    @property
    def shared_baby_rotations(self) -> int:
        return int(sum(int(value) for value in self.group_baby_rotations))

    @property
    def shared_giant_rotations(self) -> int:
        return int(sum(int(value) for value in self.group_giant_rotations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_layout": "native_halo_stripe_no_ri",
            "conv_dependency": "native_halo_source_tiles",
            "channel_fold_mode": str(self.channel_fold_mode),
            "output_storage_layout": (
                "native_halo_stripe"
                if _spec_has_physical_output_halo(self.spec)
                else ("logical_halo_compact" if _spec_has_semantic_output_halo(self.spec) else "tight_compact")
            ),
            "spec": self.spec.to_dict(),
            "source_channel_tile": int(self.source_channel_tile),
            "target_channel_tile": int(self.target_channel_tile),
            "source_channel_group_count": int(self.source_channel_group_count),
            "target_channel_group_count": int(self.target_channel_group_count),
            "source_channel_group_counts": [int(value) for value in self.source_channel_group_counts],
            "target_channel_group_counts": [int(value) for value in self.target_channel_group_counts],
            "source_stripe_offsets": [int(value) for value in self.source_stripe_offsets],
            "target_stripe_offsets": [int(value) for value in self.target_stripe_offsets],
            "stripe_count": int(len(self.stripes)),
            "input_ct_count": int(self.input_ct_count),
            "output_ct_count": int(self.output_ct_count),
            "submatrix_program_count": int(self.submatrix_program_count),
            "sharing_group_count": int(self.sharing_group_count),
            "c_only_rotations": int(self.c_only_rotations),
            "cb_shared_rotations": int(self.cb_shared_rotations),
            "shared_baby_rotations": int(self.shared_baby_rotations),
            "shared_giant_rotations": int(self.shared_giant_rotations),
            "program_diagonal_counts": [int(value) for value in self.program_diagonal_counts],
            "program_rotation_counts": [int(value) for value in self.program_rotation_counts],
            "group_n1s": [int(value) for value in self.group_n1s],
            "group_shared_rotations": [int(value) for value in self.group_shared_rotations],
            "group_baby_rotations": [int(value) for value in self.group_baby_rotations],
            "group_giant_rotations": [int(value) for value in self.group_giant_rotations],
            "stripes": [stripe.to_dict() for stripe in self.stripes],
            "notes": [
                "Height stripes come from the halo-stripe oracle source range required by each target stripe.",
                "source_local_h = in_h - source_h_start; target_local_h = out_h - target_h_start.",
                "No real/imag lane packing is used; B sharing is per native source tile.",
            ],
        }


@dataclass(frozen=True)
class NativeHaloCompactOutputRotationStats:
    c_only_rotations: int
    cb_shared_rotations: int
    shared_baby_rotations: int
    shared_giant_rotations: int
    transform_count: int
    sharing_group_count: int
    output_ct_count: int
    diagonal_counts: tuple[int, ...]
    rotation_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "c_only_rotations": int(self.c_only_rotations),
            "cb_shared_rotations": int(self.cb_shared_rotations),
            "shared_baby_rotations": int(self.shared_baby_rotations),
            "shared_giant_rotations": int(self.shared_giant_rotations),
            "transform_count": int(self.transform_count),
            "sharing_group_count": int(self.sharing_group_count),
            "output_ct_count": int(self.output_ct_count),
            "diagonal_counts": [int(value) for value in self.diagonal_counts],
            "rotation_counts": [int(value) for value in self.rotation_counts],
        }


_COMPACT_OUTPUT_ROTATION_STATS_CACHE: dict[tuple[Any, Any], NativeHaloCompactOutputRotationStats] = {}


def _compact_output_diag_sets_for_task(
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    stripe: NativeHaloStripe,
    *,
    source_group: int,
    target_group: int,
) -> dict[int, set[int]]:
    source_tile = int(plan.source_tile_for_stripe(stripe))
    target_tile = int(plan.target_tile_for_stripe(stripe))
    source_start = int(source_group) * int(source_tile)
    source_end = min(int(spec.c_in), int(source_start) + int(source_tile))
    target_start = int(target_group) * int(target_tile)
    target_end = min(int(spec.c_out), int(target_start) + int(target_tile))
    key = (
        "compact_output_diag_sets",
        _native_spec_structural_cache_key(spec),
        _native_plan_structural_cache_key(plan),
        _native_stripe_cache_key(stripe),
        int(source_end - source_start),
        int(target_start),
        int(target_end - target_start),
    )
    cached = _COMPACT_OUTPUT_DIAG_SET_CACHE.get(key)
    if cached is not None:
        return {int(block): set(int(value) for value in values) for block, values in cached}
    _native_diag_cache_guard()
    diag_sets_by_block = _compact_output_diag_sets_for_task_closed_form(
        spec,
        plan,
        stripe,
        source_group=int(source_group),
        target_group=int(target_group),
    )
    _COMPACT_OUTPUT_DIAG_SET_CACHE[key] = tuple(
        (int(block), tuple(sorted(int(value) for value in diag_set)))
        for block, diag_set in sorted(diag_sets_by_block.items())
    )
    return {int(block): set(int(value) for value in values) for block, values in _COMPACT_OUTPUT_DIAG_SET_CACHE[key]}


def _compact_output_diag_sets_for_task_closed_form(
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    stripe: NativeHaloStripe,
    *,
    source_group: int,
    target_group: int,
) -> dict[int, set[int]]:
    slots = int(spec.slot_count)
    source_tile = int(plan.source_tile_for_stripe(stripe))
    target_tile = int(plan.target_tile_for_stripe(stripe))
    source_start = int(source_group) * int(source_tile)
    source_end = min(int(spec.c_in), int(source_start) + int(source_tile))
    target_start = int(target_group) * int(target_tile)
    target_end = min(int(spec.c_out), int(target_start) + int(target_tile))
    source_count = int(source_end - source_start)
    target_count = int(target_end - target_start)
    if int(source_count) <= 0 or int(target_count) <= 0:
        return {}

    compact_output_h = _spec_physical_output_h(spec)
    source_offsets = _channel_base_offsets_chw_gap(
        0,
        int(source_count),
        height=int(stripe.source_h),
        width=int(spec.w_in),
        gap=int(spec.gap_in),
    )
    target_packed_w = int(spec.w_out) * max(1, int(spec.gap_out))
    target_group_block = int(compact_output_h) * max(1, int(spec.gap_out)) * int(target_packed_w)
    target_channel_bases = _channel_base_offsets_chw_gap(
        int(target_start),
        int(target_count),
        height=int(compact_output_h),
        width=int(spec.w_out),
        gap=int(spec.gap_out),
    )

    gap_in = max(1, int(spec.gap_in))
    gap_out = max(1, int(spec.gap_out))
    source_packed_w = int(spec.w_in) * int(gap_in)
    h_in_min = int(spec.input_h_min)
    h_in_max = int(spec.input_h_max)
    source_h_start = int(stripe.source_h_start)
    stripe_source_h = int(stripe.source_h)
    stride = int(spec.stride)
    pad = int(spec.pad)
    dilation = int(spec.dilation)
    diag_masks_by_block: dict[int, np.ndarray] = {}
    out_h_values = range(int(stripe.target_h_start), int(stripe.target_h_end))
    out_w_values = range(int(spec.w_out))
    for kh in range(int(spec.kernel)):
        h_pairs: list[tuple[int, int]] = []
        for out_h in out_h_values:
            op_out_h = int(_materialized_output_source_h(
                int(out_h),
                h_out=int(spec.h_out),
                output_top_beta=int(spec.output_top_beta),
                output_bottom_beta=int(spec.output_bottom_beta),
            ))
            in_h = int(op_out_h) * int(stride) - int(pad) + int(kh) * int(dilation)
            source_local_h = int(in_h) - int(source_h_start)
            if (
                int(in_h) < int(h_in_min)
                or int(in_h) >= int(h_in_max)
                or int(source_local_h) < 0
                or int(source_local_h) >= int(stripe_source_h)
                or not bool(_spec_physical_output_h_valid(spec, int(out_h)))
            ):
                continue
            target_physical_h = int(_spec_physical_output_h_positions(spec, int(out_h)))
            h_pairs.append((int(source_local_h), int(target_physical_h)))
        if not h_pairs:
            continue
        h_pairs_array = np.asarray(h_pairs, dtype=np.int64)
        source_h_base_values = h_pairs_array[:, 0] * int(gap_in) * int(source_packed_w)
        target_h_base_values = h_pairs_array[:, 1] * int(gap_out) * int(target_packed_w)
        for kw in range(int(spec.kernel)):
            w_pairs: list[tuple[int, int]] = []
            for out_w in out_w_values:
                in_w = int(out_w) * int(stride) - int(pad) + int(kw) * int(dilation)
                if int(in_w) < 0 or int(in_w) >= int(spec.w_in):
                    continue
                w_pairs.append((int(in_w), int(out_w)))
            if not w_pairs:
                continue
            w_pairs_array = np.asarray(w_pairs, dtype=np.int64)
            source_w_values = w_pairs_array[:, 0] * int(gap_in)
            target_w_values = w_pairs_array[:, 1] * int(gap_out)
            source_spatial_values = (source_h_base_values[:, None] + source_w_values[None, :]).reshape(-1)
            target_spatial_values = (target_h_base_values[:, None] + target_w_values[None, :]).reshape(-1)
            if int(source_spatial_values.size) == 0 or int(target_spatial_values.size) == 0:
                continue
            target_abs_grid = target_channel_bases[:, None] + target_spatial_values[None, :]
            target_blocks_grid = target_abs_grid // int(slots)
            target_slots_grid = target_abs_grid % int(slots)
            pair_index_grid = np.broadcast_to(
                np.arange(int(target_spatial_values.size), dtype=np.int64)[None, :],
                target_blocks_grid.shape,
            )
            for target_block in np.unique(target_blocks_grid).tolist():
                block = int(target_block)
                block_mask = target_blocks_grid == int(block)
                if not bool(block_mask.any()):
                    continue
                diag_mask = diag_masks_by_block.get(int(block))
                if diag_mask is None:
                    diag_mask = np.zeros((int(slots),), dtype=np.bool_)
                    diag_masks_by_block[int(block)] = diag_mask
                block_slots = target_slots_grid[block_mask].astype(np.int64, copy=False).reshape(-1)
                block_pair_indices = pair_index_grid[block_mask].astype(np.int64, copy=False).reshape(-1)
                event_count = int(block_slots.size)
                chunk = max(1, min(8192, int(event_count)))
                for start in range(0, int(event_count), int(chunk)):
                    end = min(int(event_count), int(start) + int(chunk))
                    source_for_events = source_spatial_values[block_pair_indices[start:end]]
                    values = np.remainder(
                        source_offsets[:, None]
                        + source_for_events[None, :]
                        - block_slots[start:end][None, :],
                        int(slots),
                    ).reshape(-1)
                    diag_mask[values] = True
    for mask in diag_masks_by_block.values():
        mask[0] = False
    return {
        int(block): set(int(value) for value in np.flatnonzero(mask).tolist())
        for block, mask in sorted(diag_masks_by_block.items())
        if bool(mask.any())
    }


def _compact_output_diag_sets_for_task_torch_oracle(
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    stripe: NativeHaloStripe,
    *,
    source_group: int,
    target_group: int,
) -> dict[int, set[int]]:
    slots = int(spec.slot_count)
    source_tile = int(plan.source_tile_for_stripe(stripe))
    target_tile = int(plan.target_tile_for_stripe(stripe))
    source_start = int(source_group) * int(source_tile)
    source_end = min(int(spec.c_in), int(source_start) + int(source_tile))
    target_start = int(target_group) * int(target_tile)
    target_end = min(int(spec.c_out), int(target_start) + int(target_tile))
    source_count = int(source_end - source_start)
    target_count = int(target_end - target_start)
    if int(source_count) <= 0 or int(target_count) <= 0:
        return {}

    compact_output_h = _spec_physical_output_h(spec)
    source_channels = torch.arange(int(source_count), dtype=torch.int64)
    target_channels = torch.arange(int(target_start), int(target_end), dtype=torch.int64)
    out_h_values = torch.arange(
        int(stripe.target_h_start),
        int(stripe.target_h_end),
        dtype=torch.int64,
    )
    out_w_values = torch.arange(int(spec.w_out), dtype=torch.int64)
    diag_sets_by_block: dict[int, set[int]] = {}
    for kh in range(int(spec.kernel)):
        op_out_h_values = _materialized_output_source_h(
            out_h_values,
            h_out=int(spec.h_out),
            output_top_beta=int(spec.output_top_beta),
            output_bottom_beta=int(spec.output_bottom_beta),
        )
        in_h_values = (
            op_out_h_values * int(spec.stride)
            - int(spec.pad)
            + int(kh) * int(spec.dilation)
        )
        source_local_h_values = in_h_values - int(stripe.source_h_start)
        valid_h = (
            (in_h_values >= int(spec.input_h_min))
            & (in_h_values < int(spec.input_h_max))
            & (source_local_h_values >= 0)
            & (source_local_h_values < int(stripe.source_h))
        )
        if not bool(valid_h.any().item()):
            continue
        valid_source_h = source_local_h_values[valid_h]
        valid_out_h = out_h_values[valid_h]
        for kw in range(int(spec.kernel)):
            in_w_values = (
                out_w_values * int(spec.stride)
                - int(spec.pad)
                + int(kw) * int(spec.dilation)
            )
            valid_w = (in_w_values >= 0) & (in_w_values < int(spec.w_in))
            if not bool(valid_w.any().item()):
                continue
            valid_source_w = in_w_values[valid_w]
            valid_out_w = out_w_values[valid_w]
            source_h_grid, source_w_grid = torch.meshgrid(valid_source_h, valid_source_w, indexing="ij")
            out_h_grid, out_w_grid = torch.meshgrid(valid_out_h, valid_out_w, indexing="ij")
            source_h_flat = source_h_grid.reshape(-1)
            source_w_flat = source_w_grid.reshape(-1)
            out_h_flat = out_h_grid.reshape(-1)
            out_w_flat = out_w_grid.reshape(-1)
            if int(source_h_flat.numel()) == 0:
                continue
            source_vec = _idx_chw_gap_channel_positions(
                source_channels,
                h=source_h_flat,
                w=source_w_flat,
                height=int(stripe.source_h),
                width=int(spec.w_in),
                gap=int(spec.gap_in),
            )
            target_h_valid = _spec_physical_output_h_valid(spec, out_h_flat)
            compact_slots = _idx_chw_gap_channel_positions(
                target_channels,
                h=_spec_physical_output_h_positions(spec, out_h_flat),
                w=out_w_flat,
                height=int(compact_output_h),
                width=int(spec.w_out),
                gap=int(spec.gap_out),
            )
            target_blocks = torch.div(compact_slots, int(slots), rounding_mode="floor")
            target_slots = torch.remainder(compact_slots, int(slots))
            diag_index = (source_vec[:, None, :] - target_slots[None, :, :]).remainder(int(slots))
            valid = target_h_valid[None, None, :].expand_as(diag_index)
            if not bool(valid.any().item()):
                continue
            for target_block in torch.unique(target_blocks).tolist():
                block = int(target_block)
                block_mask = target_blocks == int(block)
                mask = block_mask[None, :, :] & valid
                if not bool(mask.any().item()):
                    continue
                diag_sets_by_block.setdefault(int(block), set()).update(
                    int(value)
                    for value in torch.unique(diag_index[mask]).tolist()
                    if int(value) != 0
                )
    return diag_sets_by_block


def native_halo_conv2d_compact_output_rotation_stats(
    plan: NativeHaloConv2DPlan,
) -> NativeHaloCompactOutputRotationStats:
    spec = plan.spec
    key = (
        _native_spec_structural_cache_key(spec),
        _native_plan_structural_cache_key(plan),
    )
    cached = _COMPACT_OUTPUT_ROTATION_STATS_CACHE.get(key)
    if cached is not None:
        return cached

    transform_count = 0
    c_only_rotations = 0
    cb_shared_rotations = 0
    shared_baby_rotations = 0
    shared_giant_rotations = 0
    diagonal_counts: list[int] = []
    rotation_counts: list[int] = []
    diag_sets_cache: dict[tuple[int, int, int], dict[int, set[int]]] = {}
    output_ct_count = _compact_ct_count(
        int(spec.c_out),
        _spec_physical_output_h(spec),
        int(spec.w_out),
        int(spec.gap_out),
        int(spec.slot_count),
    )
    group_index = 0
    for stripe in plan.stripes:
        for source_group in range(int(plan.source_group_count_for_stripe(stripe))):
            group_sets: list[set[int]] = []
            group_n1 = int(plan.group_n1s[int(group_index)]) if int(group_index) < len(plan.group_n1s) else 1
            for target_group in range(int(plan.target_group_count_for_stripe(stripe))):
                source_tile = int(plan.source_tile_for_stripe(stripe))
                source_start = int(source_group) * int(source_tile)
                source_end = min(int(spec.c_in), int(source_start) + int(source_tile))
                diag_key = (
                    int(stripe.index),
                    int(source_end - source_start),
                    int(target_group),
                )
                cached_by_block = diag_sets_cache.get(diag_key)
                if cached_by_block is None:
                    cached_by_block = _compact_output_diag_sets_for_task(
                        spec,
                        plan,
                        stripe,
                        source_group=int(source_group),
                        target_group=int(target_group),
                    )
                    diag_sets_cache[diag_key] = {
                        int(block): set(int(value) for value in diag_set)
                        for block, diag_set in cached_by_block.items()
                    }
                by_block = cached_by_block
                for _target_block, diag_set in sorted(by_block.items()):
                    if not diag_set:
                        continue
                    baby, giant = _cached_bsgs_rotation_sets(
                        set(int(value) for value in diag_set),
                        slots=int(spec.slot_count),
                        n1=int(group_n1),
                    )
                    rotations = int(len(baby) + len(giant))
                    transform_count += 1
                    c_only_rotations += int(rotations)
                    diagonal_counts.append(int(len(diag_set)))
                    rotation_counts.append(int(rotations))
                    group_sets.append(set(int(value) for value in diag_set))
            if group_sets:
                _n1, group_rotations, baby_count, giant_count = _cached_native_best_common_bsgs(
                    tuple(group_sets),
                    slots=int(spec.slot_count),
                )
                cb_shared_rotations += int(group_rotations)
                shared_baby_rotations += int(baby_count)
                shared_giant_rotations += int(giant_count)
            group_index += 1

    stats = NativeHaloCompactOutputRotationStats(
        c_only_rotations=int(c_only_rotations),
        cb_shared_rotations=int(cb_shared_rotations),
        shared_baby_rotations=int(shared_baby_rotations),
        shared_giant_rotations=int(shared_giant_rotations),
        transform_count=int(transform_count),
        sharing_group_count=int(group_index),
        output_ct_count=int(output_ct_count),
        diagonal_counts=tuple(int(value) for value in diagonal_counts),
        rotation_counts=tuple(int(value) for value in rotation_counts),
    )
    _COMPACT_OUTPUT_ROTATION_STATS_CACHE[key] = stats
    return stats


class _BlockDiagonalCache:
    def __init__(self, build_all: Callable[[], dict[tuple[int, int], dict[int, Any]]]) -> None:
        self._build_all = build_all
        self._lock = threading.Lock()
        self._blocks: dict[tuple[int, int], dict[int, Any]] | None = None

    def get(self, row: int, col: int) -> dict[int, Any]:
        blocks = self._blocks
        if blocks is None:
            with self._lock:
                blocks = self._blocks
                if blocks is None:
                    blocks = self._build_all()
                    self._blocks = blocks
        return dict(blocks.get((int(row), int(col)), {}) or {})

    def get_required(self, row: int, col: int, *, context: str = "") -> dict[int, Any]:
        block = self.get(int(row), int(col))
        if block:
            return block
        detail = f" for {context}" if context else ""
        raise RuntimeError(
            f"Native halo Conv2d single-slot diagonal cache missing block "
            f"(row={int(row)}, col={int(col)}){detail}"
        )

    def release(self) -> None:
        with self._lock:
            self._blocks = None


def _diag_tensors_to_payload(diag_tensors: dict[int, Any], *, slots: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(sorted(int(index) for index in dict(diag_tensors or {}).keys()), dtype=np.int32)
    chunks: list[np.ndarray] = []
    for index in indices.tolist():
        value = dict(diag_tensors)[int(index)]
        if isinstance(value, torch.Tensor):
            chunks.append(value.detach().cpu().to(dtype=torch.float32).numpy().reshape(-1))
        else:
            chunks.append(np.asarray(value, dtype=np.float32).reshape(-1))
    data = (
        np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
        if chunks
        else np.zeros((0,), dtype=np.float32)
    )
    expected = int(indices.size) * int(slots)
    if int(data.size) != int(expected):
        raise RuntimeError(f"provider diag payload length {int(data.size)} != expected {int(expected)}")
    return indices, data


def _payload_to_diag_tensors(diag_indices: np.ndarray, diag_data: np.ndarray, *, slots: int) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    indices = np.asarray(diag_indices, dtype=np.int32).reshape(-1)
    data = np.asarray(diag_data, dtype=np.float32).reshape(-1)
    for offset, index in enumerate(indices.tolist()):
        start = int(offset) * int(slots)
        end = int(start) + int(slots)
        out[int(index)] = torch.as_tensor(np.ascontiguousarray(data[start:end], dtype=np.float32))
    return out


def _transform_payload_block(transform: Any, *, slots: int, context: str) -> dict[int, torch.Tensor]:
    block = dict(getattr(transform, "diagonals", {}).get((0, 0), {}) or {})
    materialized: dict[int, torch.Tensor] = {}
    for diag_idx, diag_values in block.items():
        if isinstance(diag_values, torch.Tensor):
            values = diag_values.detach().cpu().to(dtype=torch.float32).reshape(-1)
        else:
            values = torch.as_tensor(np.asarray(diag_values, dtype=np.float32)).reshape(-1)
        if int(values.numel()) == int(slots):
            materialized[int(diag_idx)] = values
            continue
        if int(values.numel()) == 0:
            continue
        raise RuntimeError(
            f"{context} rebuilt diagonal {int(diag_idx)} has {int(values.numel())} values; "
            f"expected {int(slots)}"
        )
    if materialized:
        return materialized

    preflattened = getattr(transform, "_preflattened_diag_payload", None)
    if preflattened is None:
        return {}
    diag_indices, diag_data, _level = preflattened
    indices = np.asarray(diag_indices, dtype=np.int32).reshape(-1)
    data = np.asarray(diag_data, dtype=np.float32).reshape(-1)
    expected = int(indices.size) * int(slots)
    if int(data.size) != int(expected):
        raise RuntimeError(
            f"{context} rebuilt preflattened payload has {int(data.size)} values for "
            f"{int(indices.size)} diagonals; expected {int(expected)}"
        )
    return _payload_to_diag_tensors(indices, data, slots=int(slots))


def _payload_to_diag_dict(
    diag_indices: np.ndarray,
    diag_data: np.ndarray,
    *,
    slots: int,
    materialize_tensors: bool,
) -> dict[int, torch.Tensor]:
    if bool(materialize_tensors):
        return _payload_to_diag_tensors(diag_indices, diag_data, slots=int(slots))
    indices = np.asarray(diag_indices, dtype=np.int32).reshape(-1)
    data = np.asarray(diag_data, dtype=np.float32).reshape(-1)
    if int(indices.size) <= 0:
        return {}
    expected = int(indices.size) * int(slots)
    if int(data.size) != int(expected):
        raise RuntimeError(f"provider diag payload length {int(data.size)} != expected {int(expected)}")
    return {int(index): torch.empty((0,), dtype=torch.float32) for index in indices.tolist()}


def _provider_payload_transform(
    *,
    name: str,
    diag_indices: np.ndarray,
    diag_data: np.ndarray,
    metadata: dict[str, Any],
    slots: int,
    level: int,
    scheme: Any,
    target_index: int,
    input_id: str,
    group_n1: int,
    rotation_group_id: str,
    rotation_cost_owner: bool,
    materialize_tensors: bool = True,
) -> Any:
    diag_tensors = _payload_to_diag_dict(
        diag_indices,
        diag_data,
        slots=int(slots),
        materialize_tensors=bool(materialize_tensors),
    )
    if not diag_tensors:
        return None
    diag_set = set(int(value) for value in diag_tensors)
    flat_diag_indices = np.ascontiguousarray(np.asarray(diag_indices, dtype=np.int32).reshape(-1), dtype=np.int32)
    flat_diag_data = np.ascontiguousarray(np.asarray(diag_data, dtype=np.float32).reshape(-1), dtype=np.float32)
    baby, giant = _bsgs_rotation_sets(set(diag_set), slots=int(slots), n1=int(group_n1))
    payload = SimpleNamespace(
        name=str(name),
        diagonals={(0, 0): {int(index): diag for index, diag in sorted(diag_tensors.items())}},
        level=int(level),
        scheme=scheme,
        fhe_output_shape=torch.Size([1, int(slots)]),
        output_shape=torch.Size([1, int(slots)]),
        target_index=int(target_index),
        input_id=str(input_id),
        selected_n1=int(group_n1),
        baby_shifts=tuple(sorted(int(value) for value in baby)),
        giant_shifts=tuple(sorted(int(value) for value in giant)),
        rotation_group_id=str(rotation_group_id),
        rotation_cost_owner=bool(rotation_cost_owner),
    )
    setattr(payload, "_diag_builder_metadata", dict(metadata))
    if not bool(materialize_tensors):
        setattr(payload, "_preflattened_diag_payload", (flat_diag_indices, flat_diag_data, int(level)))
        setattr(payload, "_single_slot_diag_indices_by_block", {(0, 0): tuple(int(v) for v in flat_diag_indices)})
    return payload


def _single_slot_diag_indices_from_transform(transform: Any) -> tuple[int, ...]:
    indices_by_block = getattr(transform, "_single_slot_diag_indices_by_block", None)
    values: set[int] = set()
    if indices_by_block is not None:
        for diag_indices in dict(indices_by_block or {}).values():
            values.update(int(value) for value in diag_indices)
    if not values:
        for _block_key, block in dict(getattr(transform, "diagonals", {}) or {}).items():
            values.update(int(value) for value in dict(block or {}).keys())
    return tuple(sorted(int(value) for value in values))


def _make_provider_native_source_transform_from_payload(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    stripe: NativeHaloStripe,
    source_group: int,
    target_group: int,
    compact_target_block: int | None,
    diag_indices: np.ndarray,
    diag_data: np.ndarray,
    metadata: dict[str, Any],
    level: int,
    scheme: Any,
    group_n1: int,
) -> Any | None:
    metadata = dict(metadata)
    compact_output = compact_target_block is not None
    target_index = (
        int(compact_target_block)
        if compact_target_block is not None
        else int(plan.target_block_index(stripe, int(target_group)))
    )
    source_index = int(plan.source_block_index(stripe, int(source_group)))
    name = (
        f"native_halo_{spec.family_label}_s{int(stripe.index)}_src{int(source_group)}"
        f"_tgt{int(target_group)}"
        + ("" if compact_target_block is None else f"_compact{int(compact_target_block)}")
    )
    return _provider_payload_transform(
        name=name,
        diag_indices=diag_indices,
        diag_data=diag_data,
        metadata=metadata,
        slots=int(spec.slot_count),
        level=int(level),
        scheme=scheme,
        target_index=int(target_index),
        input_id=f"native_source_tile_{int(source_index)}",
        group_n1=int(group_n1),
        rotation_group_id=f"native_halo:{spec.family_label}:s{int(stripe.index)}:src{int(source_group)}",
        rotation_cost_owner=bool(
            int(target_group) == 0
            and (not bool(compact_output) or int(compact_target_block) == 0)
        ),
        materialize_tensors=not bool(str(metadata.get("diag_builder_source", "")).startswith("cpp")),
    )


def _cpp_provider_native_source_transform(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    weight_np: np.ndarray | None = None,
    stripe: NativeHaloStripe,
    source_group: int,
    target_group: int,
    level: int,
    scheme: Any,
    group_n1: int,
    compact_target_block: int | None = None,
    path: str = "native_source",
    env_gate: str = "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE",
) -> Any | None | object:
    if not _provider_diag_builder_enabled():
        return _CPP_DIAG_BUILDER_FALLBACK
    if not _env_enabled(str(env_gate)):
        if _provider_diag_builder_strict():
            raise RuntimeError(f"provider {path} subpath unsupported by C++ diag builder env gate")
        return _CPP_DIAG_BUILDER_FALLBACK
    if _provider_diag_builder_shadow():
        return _CPP_DIAG_BUILDER_FALLBACK

    started = time.time()
    try:
        from orion.backend.diag_builder import bindings as diag_builder

        built = diag_builder.build_provider_native_source_conv2d_payload(
            spec=spec,
            plan=plan,
            weight=weight,
            weight_np=weight_np,
            stripe=stripe,
            source_group=int(source_group),
            target_group=int(target_group),
            compact_target_block=compact_target_block,
        )
    except Exception:
        if _provider_diag_builder_strict():
            raise
        return _CPP_DIAG_BUILDER_FALLBACK

    if built is None:
        return None
    diag_indices, diag_data, metadata = built
    metadata = dict(metadata)
    metadata["diag_builder_kind"] = f"provider_native_halo_conv2d:{path}"
    metadata["diag_builder_source"] = "cpp"
    metadata["diag_builder_build_s"] = float(metadata.get("diag_builder_build_s", 0.0) or 0.0)
    metadata["diag_builder_shadow_s"] = 0.0
    metadata["diag_builder_fallback_count"] = 0.0
    metadata["diag_builder_fallback_reason"] = ""
    metadata.setdefault("diag_builder_payload_count", 1.0)
    transform = _make_provider_native_source_transform_from_payload(
        spec=spec,
        plan=plan,
        stripe=stripe,
        source_group=int(source_group),
        target_group=int(target_group),
        compact_target_block=compact_target_block,
        diag_indices=diag_indices,
        diag_data=diag_data,
        metadata=metadata,
        level=int(level),
        scheme=scheme,
        group_n1=int(group_n1),
    )
    if transform is None and _provider_diag_builder_strict():
        raise RuntimeError("provider C++ diag builder produced empty payload for non-empty native-source transform")
    if transform is not None:
        elapsed = float(time.time() - started)
        transform._diag_builder_metadata["diag_builder_build_s"] = float(
            transform._diag_builder_metadata.get("diag_builder_build_s", elapsed) or elapsed
        )
        transform._diag_builder_metadata["diag_builder_wall_s"] = float(elapsed)
    return transform


def _build_conv_transform_batch(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    weight_np: np.ndarray | None,
    stripe: NativeHaloStripe,
    source_group: int,
    target_groups: list[int],
    level: int,
    scheme: Any,
    group_n1: int,
) -> list[tuple[int, Any]]:
    if not target_groups:
        return []
    workers = _provider_diag_build_workers(len(target_groups))
    single_slot_recipe = bool(_single_slot_layer_cache_enabled_for_scheme(scheme))
    if (
        bool(single_slot_recipe)
        or int(workers) <= 1
        or not _provider_diag_builder_enabled()
        or not _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE")
        or _provider_diag_builder_shadow()
    ):
        return [
            (int(transform.target_index), transform)
            for target_group in target_groups
            for transform in [
                _build_conv_transform(
                    spec=spec,
                    plan=plan,
                    weight=weight,
                    weight_np=weight_np,
                    stripe=stripe,
                    source_group=int(source_group),
                    target_group=int(target_group),
                    level=int(level),
                    scheme=scheme,
                    group_n1=int(group_n1),
                    compact_target_block=None,
                )
            ]
            if transform is not None
        ]

    try:
        from orion.backend.diag_builder import bindings as diag_builder

        if diag_builder.load_library() is None:
            raise RuntimeError(diag_builder.load_error() or "provider native-source diag builder unavailable")
    except Exception:
        if _provider_diag_builder_strict():
            raise
        return [
            (int(transform.target_index), transform)
            for target_group in target_groups
            for transform in [
                _build_conv_transform(
                    spec=spec,
                    plan=plan,
                    weight=weight,
                    weight_np=weight_np,
                    stripe=stripe,
                    source_group=int(source_group),
                    target_group=int(target_group),
                    level=int(level),
                    scheme=scheme,
                    group_n1=int(group_n1),
                    compact_target_block=None,
                )
            ]
            if transform is not None
        ]

    def build_one(target_group: int) -> tuple[int, tuple[np.ndarray, np.ndarray, dict[str, Any]] | None]:
        built = diag_builder.build_provider_native_source_conv2d_payload(
            spec=spec,
            plan=plan,
            weight=weight,
            weight_np=weight_np,
            stripe=stripe,
            source_group=int(source_group),
            target_group=int(target_group),
            compact_target_block=None,
        )
        return int(target_group), built

    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="orion-provider-diag-build") as executor:
            built_payloads = list(executor.map(build_one, [int(value) for value in target_groups]))
    except Exception:
        if _provider_diag_builder_strict():
            raise
        return [
            (int(transform.target_index), transform)
            for target_group in target_groups
            for transform in [
                _build_conv_transform(
                    spec=spec,
                    plan=plan,
                    weight=weight,
                    weight_np=weight_np,
                    stripe=stripe,
                    source_group=int(source_group),
                    target_group=int(target_group),
                    level=int(level),
                    scheme=scheme,
                    group_n1=int(group_n1),
                    compact_target_block=None,
                )
            ]
            if transform is not None
        ]
    ordered: list[tuple[int, Any]] = []
    for target_group, built in built_payloads:
        if built is None:
            continue
        diag_indices, diag_data, metadata = built
        metadata = dict(metadata)
        metadata["diag_builder_kind"] = "provider_native_halo_conv2d:native_source"
        metadata["diag_builder_source"] = "cpp"
        metadata["diag_builder_build_s"] = float(metadata.get("diag_builder_build_s", 0.0) or 0.0)
        metadata["diag_builder_wall_s"] = 0.0
        metadata["diag_builder_shadow_s"] = 0.0
        metadata["diag_builder_fallback_count"] = 0.0
        metadata["diag_builder_fallback_reason"] = ""
        metadata.setdefault("diag_builder_payload_count", 1.0)
        transform = _make_provider_native_source_transform_from_payload(
            spec=spec,
            plan=plan,
            stripe=stripe,
            source_group=int(source_group),
            target_group=int(target_group),
            compact_target_block=None,
            diag_indices=diag_indices,
            diag_data=diag_data,
            metadata=metadata,
            level=int(level),
            scheme=scheme,
            group_n1=int(group_n1),
        )
        if transform is None:
            if _provider_diag_builder_strict():
                raise RuntimeError("provider C++ diag builder produced empty payload for non-empty native-source transform")
            continue
        ordered.append((int(transform.target_index), transform))
    batch_wall_s = float(time.time() - started)
    if ordered:
        wall_share = float(batch_wall_s) / max(1.0, float(len(ordered)))
        for _target_index, transform in ordered:
            transform._diag_builder_metadata["diag_builder_wall_s"] = float(wall_share)
    return ordered


def _shadow_provider_payload(
    *,
    transform: Any,
    slots: int,
    build_cpp_payload: Callable[[], tuple[np.ndarray, np.ndarray] | None],
    path: str,
) -> Any | None:
    if transform is None:
        return None
    if not _provider_diag_builder_enabled():
        return transform
    started = time.time()
    try:
        cpp_payload = build_cpp_payload()
    except Exception as exc:
        if _provider_diag_builder_strict():
            raise
        setattr(
            transform,
            "_diag_builder_metadata",
            {
                "diag_builder_kind": f"provider_native_halo_conv2d:{path}",
                "diag_builder_source": "python_fallback",
                "diag_builder_build_s": float(time.time() - started),
                "diag_builder_payload_count": 0.0,
                "diag_builder_fallback_count": 1.0,
                "diag_builder_fallback_reason": str(exc),
            },
        )
        return transform
    build_s = float(time.time() - started)
    if cpp_payload is None:
        if _provider_diag_builder_strict():
            raise RuntimeError("provider subpath unsupported by C++ diag builder")
        setattr(
            transform,
            "_diag_builder_metadata",
            {
                "diag_builder_kind": f"provider_native_halo_conv2d:{path}",
                "diag_builder_source": "python_fallback",
                "diag_builder_build_s": float(build_s),
                "diag_builder_payload_count": 0.0,
                "diag_builder_fallback_count": 1.0,
                "diag_builder_fallback_reason": "provider subpath unsupported by C++ diag builder",
            },
        )
        return transform
    diag_indices, diag_data = cpp_payload
    py_block = dict(getattr(transform, "diagonals", {}).get((0, 0), {}) or {})
    py_indices, py_data = _diag_tensors_to_payload(py_block, slots=int(slots))
    ok = bool(
        np.array_equal(np.asarray(diag_indices, dtype=np.int32), py_indices)
        and np.allclose(np.asarray(diag_data, dtype=np.float32), py_data, atol=1.0e-6, rtol=1.0e-6)
    )
    metadata = {
        "diag_builder_kind": f"provider_native_halo_conv2d:{path}",
        "diag_builder_source": "cpp_shadow",
        "diag_builder_build_s": float(build_s),
        "diag_builder_shadow_s": 0.0,
        "diag_builder_payload_count": 1.0,
        "diag_builder_fallback_count": 0.0 if ok else 1.0,
        "diag_builder_shadow_ok": bool(ok),
        "diag_builder_fallback_reason": "" if ok else "provider C++ payload mismatch",
    }
    setattr(transform, "_diag_builder_metadata", metadata)
    if not ok and _provider_diag_builder_strict():
        raise RuntimeError(str(metadata["diag_builder_fallback_reason"]))
    return transform


def _cpp_provider_compact_source_transform(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    stripe: NativeHaloStripe,
    source_block: int,
    target_group: int,
    level: int,
    scheme: Any,
    source_layout: dict[str, Any],
    group_n1: int,
    compact_target_block: int | None = None,
) -> Any | None | object:
    if not _provider_diag_builder_enabled():
        return _CPP_DIAG_BUILDER_FALLBACK
    if not _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_SOURCE"):
        if _provider_diag_builder_strict():
            raise RuntimeError("provider compact_source subpath unsupported by C++ diag builder env gate")
        return _CPP_DIAG_BUILDER_FALLBACK
    if _provider_diag_builder_shadow():
        return _CPP_DIAG_BUILDER_FALLBACK

    started = time.time()
    try:
        from orion.backend.diag_builder import bindings as diag_builder

        built = diag_builder.build_provider_compact_source_conv2d_payload(
            spec=spec,
            plan=plan,
            weight=weight,
            stripe=stripe,
            source_block=int(source_block),
            target_group=int(target_group),
            source_layout=dict(source_layout),
            compact_target_block=compact_target_block,
        )
    except Exception:
        if _provider_diag_builder_strict():
            raise
        return _CPP_DIAG_BUILDER_FALLBACK
    if built is None:
        return None
    diag_indices, diag_data, metadata = built
    metadata = dict(metadata)
    metadata["diag_builder_kind"] = "provider_native_halo_conv2d:compact_source"
    metadata["diag_builder_source"] = "cpp"
    metadata["diag_builder_build_s"] = float(metadata.get("diag_builder_build_s", 0.0) or 0.0)
    metadata["diag_builder_shadow_s"] = 0.0
    metadata["diag_builder_fallback_count"] = 0.0
    metadata["diag_builder_fallback_reason"] = ""
    metadata.setdefault("diag_builder_payload_count", 1.0)
    compact_output = compact_target_block is not None
    target_index = (
        int(compact_target_block)
        if bool(compact_output)
        else int(plan.target_block_index(stripe, int(target_group)))
    )
    name = (
        f"native_halo_{spec.family_label}_compactsrc{int(source_block)}"
        f"_s{int(stripe.index)}_tgt{int(target_group)}"
        + ("" if not bool(compact_output) else f"_compact{int(compact_target_block)}")
    )
    transform = _provider_payload_transform(
        name=name,
        diag_indices=diag_indices,
        diag_data=diag_data,
        metadata=metadata,
        slots=int(spec.slot_count),
        level=int(level),
        scheme=scheme,
        target_index=int(target_index),
        input_id=f"compact_source_block_{int(source_block)}",
        group_n1=int(group_n1),
        rotation_group_id=f"native_halo:{spec.family_label}:compact_src{int(source_block)}",
        rotation_cost_owner=bool(int(target_group) == 0 and (not bool(compact_output) or int(compact_target_block) == 0)),
    )
    if transform is None and _provider_diag_builder_strict():
        raise RuntimeError("provider C++ diag builder produced empty payload for non-empty compact-source transform")
    if transform is not None:
        elapsed = float(time.time() - started)
        transform._diag_builder_metadata["diag_builder_build_s"] = float(
            transform._diag_builder_metadata.get("diag_builder_build_s", elapsed) or elapsed
        )
    return transform


def _build_compact_source_conv_payload_shadow(
    transform: Any | None,
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    stripe: NativeHaloStripe,
    source_block: int,
    target_group: int,
    source_layout: dict[str, Any],
    compact_target_block: int | None,
) -> Any | None:
    if transform is None:
        return None
    slots = int(transform.fhe_output_shape[-1])

    def build_cpp_payload():
        if not _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_SOURCE"):
            return None
        from orion.backend.diag_builder import bindings as diag_builder

        built = diag_builder.build_provider_compact_source_conv2d_payload(
            spec=spec,
            plan=plan,
            weight=weight,
            stripe=stripe,
            source_block=int(source_block),
            target_group=int(target_group),
            source_layout=dict(source_layout),
            compact_target_block=compact_target_block,
        )
        if built is None:
            return None
        diag_indices, diag_data, _metadata = built
        return diag_indices, diag_data

    return _shadow_provider_payload(
        transform=transform,
        slots=int(slots),
        build_cpp_payload=build_cpp_payload,
        path="compact_source",
    )


def _normalise_compact_source_concat_index_result(
    value: dict[int, list[tuple[int, Any]]],
) -> dict[int, list[tuple[int, tuple[int, ...]]]]:
    return {
        int(source_block): [
            (int(target_block), tuple(sorted(int(index) for index in tuple(indices))))
            for target_block, indices in sorted(list(items), key=lambda item: int(item[0]))
            if tuple(indices)
        ]
        for source_block, items in sorted(dict(value).items(), key=lambda item: int(item[0]))
        if items
    }


def _compact_source_concat_diag_sets_from_index_result(
    value: dict[int, list[tuple[int, Any]]],
) -> dict[tuple[int, int], set[int]]:
    return {
        (int(source_block), int(target_block)): set(int(index) for index in tuple(indices))
        for source_block, items in _normalise_compact_source_concat_index_result(value).items()
        for target_block, indices in items
        if tuple(indices)
    }


def _build_compact_source_concat_indices_cpp(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    source_layout: dict[str, Any],
    source_ct_count: int,
    target_ct_count: int,
    output_materialization: str,
) -> tuple[dict[int, list[tuple[int, tuple[int, ...]]]], dict[str, Any]] | object:
    if not _provider_diag_builder_enabled():
        return _CPP_DIAG_BUILDER_FALLBACK
    if not _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_SOURCE"):
        if _provider_diag_builder_strict():
            raise RuntimeError("provider compact_source concat index subpath unsupported by C++ diag builder env gate")
        return _CPP_DIAG_BUILDER_FALLBACK
    try:
        from orion.backend.diag_builder import bindings as diag_builder

        built = diag_builder.build_provider_compact_source_concat_conv2d_indices(
            spec=spec,
            plan=plan,
            weight=weight,
            source_layout=dict(source_layout),
            source_ct_count=int(source_ct_count),
            target_ct_count=int(target_ct_count),
            output_materialization=str(output_materialization or ""),
        )
    except Exception:
        if _provider_diag_builder_strict():
            raise
        return _CPP_DIAG_BUILDER_FALLBACK
    if built is None:
        return {}, {
            "diag_builder_kind": "provider_native_halo_conv2d:compact_source_concat_index_only",
            "diag_builder_source": "cpp",
            "diag_builder_build_s": 0.0,
            "diag_builder_payload_count": 0.0,
            "diag_builder_fallback_count": 0.0,
            "diag_builder_fallback_reason": "",
        }
    result, metadata = built
    metadata = dict(metadata)
    metadata["diag_builder_kind"] = "provider_native_halo_conv2d:compact_source_concat_index_only"
    metadata["diag_builder_source"] = "cpp"
    metadata["diag_builder_build_s"] = float(metadata.get("diag_builder_build_s", 0.0) or 0.0)
    metadata["diag_builder_shadow_s"] = 0.0
    metadata["diag_builder_fallback_count"] = 0.0
    metadata["diag_builder_fallback_reason"] = ""
    metadata.setdefault("diag_builder_payload_count", float(sum(len(items) for items in dict(result).values())))
    return _normalise_compact_source_concat_index_result(result), metadata


def _collect_compact_source_conv_diag_sets(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    stripe: NativeHaloStripe,
    source_block: int,
    target_group: int,
    source_layout: dict[str, Any],
    compact_output: bool,
    compact_target_block: int | None = None,
) -> dict[int, set[int]]:
    slots = int(spec.slot_count)
    target_tile = int(plan.target_tile_for_stripe(stripe))
    target_start = int(target_group) * int(target_tile)
    target_end = min(int(spec.c_out), int(target_start) + int(target_tile))
    target_count = int(target_end - target_start)
    if int(target_count) <= 0:
        return {}
    target_index_key = int(plan.target_block_index(stripe, int(target_group)))
    source_channels = torch.arange(int(spec.c_in), dtype=torch.int64)
    target_channels = (
        torch.arange(int(target_start), int(target_end), dtype=torch.int64)
        if bool(compact_output)
        else torch.arange(int(target_count), dtype=torch.int64)
    )
    source_top_beta = max(0, int(_layout_physical_top_beta(source_layout, default=spec.input_physical_top_beta or 0) or 0))
    source_bottom_beta = max(
        0,
        int(_layout_physical_bottom_beta(source_layout, default=spec.input_physical_bottom_beta or 0) or 0),
    )
    source_gap = max(1, int(source_layout.get("gap", spec.gap_in) or 1))
    source_height = int(spec.h_in) + int(source_top_beta) + int(source_bottom_beta)
    compact_output_h = _spec_physical_output_h(spec)
    out_h_values = torch.arange(
        int(stripe.target_h_start),
        int(stripe.target_h_end),
        dtype=torch.int64,
    )
    out_w_values = torch.arange(int(spec.w_out), dtype=torch.int64)
    max_pair_count = int(_native_halo_build_pair_chunk_limit())
    diag_sets: dict[int, set[int]] = {}

    for kh in range(int(spec.kernel)):
        for kw in range(int(spec.kernel)):
            coeff = weight[int(target_start): int(target_end), :, int(kh), int(kw)].to(dtype=torch.float32)
            if not bool(torch.any(coeff != 0).item()):
                continue
            coeff_by_source_target = coeff.t().contiguous()
            coeff_nonzero = torch.abs(coeff_by_source_target) > 0
            if not bool(coeff_nonzero.any().item()):
                continue
            op_out_h_values = (
                _materialized_output_source_h(
                    out_h_values,
                    h_out=int(spec.h_out),
                    output_top_beta=int(spec.output_top_beta),
                    output_bottom_beta=int(spec.output_bottom_beta),
                )
                if bool(compact_output)
                else out_h_values
            )
            in_h_values = (
                op_out_h_values * int(spec.stride)
                - int(spec.pad)
                + int(kh) * int(spec.dilation)
            )
            valid_h = (in_h_values >= 0) & (in_h_values < int(spec.h_in))
            if not bool(valid_h.any().item()):
                continue
            in_w_values = (
                out_w_values * int(spec.stride)
                - int(spec.pad)
                + int(kw) * int(spec.dilation)
            )
            valid_w = (in_w_values >= 0) & (in_w_values < int(spec.w_in))
            if not bool(valid_w.any().item()):
                continue

            valid_out_h = out_h_values[valid_h]
            valid_source_h = int(source_top_beta) + in_h_values[valid_h]
            valid_out_w = out_w_values[valid_w]
            valid_source_w = in_w_values[valid_w]
            grid_h, grid_w = torch.meshgrid(valid_source_h, valid_source_w, indexing="ij")
            out_grid_h, out_grid_w = torch.meshgrid(valid_out_h, valid_out_w, indexing="ij")
            source_h_flat = grid_h.reshape(-1)
            source_w_flat = grid_w.reshape(-1)
            out_h_flat = out_grid_h.reshape(-1)
            out_w_flat = out_grid_w.reshape(-1)
            if int(source_h_flat.numel()) == 0:
                continue

            channel_pair_count = max(1, int(source_channels.numel()) * int(target_count))
            position_chunk = max(1, int(max_pair_count // channel_pair_count))
            for start in range(0, int(source_h_flat.numel()), int(position_chunk)):
                end = min(int(source_h_flat.numel()), int(start) + int(position_chunk))
                source_index = _idx_chw_gap_channel_positions(
                    source_channels,
                    h=source_h_flat[int(start): int(end)],
                    w=source_w_flat[int(start): int(end)],
                    height=int(source_height),
                    width=int(spec.w_in),
                    gap=int(source_gap),
                )
                source_block_mask = (
                    torch.div(source_index, int(slots), rounding_mode="floor")
                    == int(source_block)
                )
                if not bool(source_block_mask.any().item()):
                    continue
                source_vec = torch.remainder(source_index, int(slots))

                if bool(compact_output):
                    target_h_slice = out_h_flat[int(start): int(end)]
                    target_h_valid = _spec_physical_output_h_valid(spec, target_h_slice)
                    target_index = _idx_chw_gap_channel_positions(
                        target_channels,
                        h=_spec_physical_output_h_positions(spec, target_h_slice),
                        w=out_w_flat[int(start): int(end)],
                        height=int(compact_output_h),
                        width=int(spec.w_out),
                        gap=int(spec.gap_out),
                    )
                    target_blocks = torch.div(target_index, int(slots), rounding_mode="floor")
                    target_vec = torch.remainder(target_index, int(slots))
                    block_values = (
                        [int(compact_target_block)]
                        if compact_target_block is not None
                        else [int(value) for value in torch.unique(target_blocks).tolist()]
                    )
                else:
                    target_local_h = out_h_flat[int(start): int(end)] - int(stripe.target_h_start)
                    target_h_valid = torch.ones_like(target_local_h, dtype=torch.bool)
                    target_vec = _idx_chw_gap_channel_positions(
                        torch.arange(int(target_count), dtype=torch.int64),
                        h=target_local_h,
                        w=out_w_flat[int(start): int(end)],
                        height=int(stripe.target_h),
                        width=int(spec.w_out),
                        gap=int(spec.gap_out),
                    )
                    target_blocks = torch.full_like(target_vec, int(target_index_key), dtype=torch.int64)
                    block_values = [int(target_index_key)]

                diag_index = (source_vec[:, None, :] - target_vec[None, :, :]).remainder(int(slots))
                for block in block_values:
                    block_mask = target_blocks == int(block)
                    if not bool(block_mask.any().item()):
                        continue
                    pair_mask = (
                        source_block_mask[:, None, :]
                        & block_mask[None, :, :]
                        & target_h_valid[None, None, :]
                        & coeff_nonzero[:, :, None]
                    )
                    if not bool(pair_mask.any().item()):
                        continue
                    diag_sets.setdefault(int(block), set()).update(
                        int(value) for value in torch.unique(diag_index[pair_mask]).tolist()
                    )
    return {int(block): values for block, values in diag_sets.items() if values}


def _make_compact_source_conv_single_slot_transform(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    stripe: NativeHaloStripe,
    source_block: int,
    target_group: int,
    level: int,
    scheme: Any,
    source_layout: dict[str, Any],
    group_n1: int,
    target_index: int,
    diag_set: set[int],
    compact_target_block: int | None = None,
    diagonal_cache: _BlockDiagonalCache | None = None,
    release_diagonal_cache_on_materialize: bool = True,
) -> Any:
    slots = int(spec.slot_count)
    baby, giant = _bsgs_rotation_sets(set(int(value) for value in diag_set), slots=int(slots), n1=int(group_n1))
    fallback_diag = int(min(diag_set)) if diag_set else 0

    def build_diagonals(
        *,
        spec=spec,
        plan=plan,
        weight=weight,
        stripe=stripe,
        source_block=int(source_block),
        target_group=int(target_group),
        level=int(level),
        scheme=scheme,
        source_layout=dict(source_layout),
        group_n1=int(group_n1),
        compact_target_block=compact_target_block,
        fallback_diag=int(fallback_diag),
        slots=int(slots),
        diagonal_cache=diagonal_cache,
        target_index=int(target_index),
    ):
        if diagonal_cache is not None:
            block = diagonal_cache.get_required(
                int(target_index),
                int(source_block),
                context=(
                    f"{spec.family_label} compact-source source_block={int(source_block)} "
                    f"target_index={int(target_index)}"
                ),
            )
            return {(0, 0): block}
        rebuilt = _build_compact_source_conv_transform(
            spec=spec,
            plan=plan,
            weight=weight,
            stripe=stripe,
            source_block=int(source_block),
            target_group=int(target_group),
            level=int(level),
            scheme=scheme,
            source_layout=dict(source_layout),
            group_n1=int(group_n1),
            compact_target_block=compact_target_block,
            force_payload=True,
        )
        if rebuilt is None:
            if diag_set:
                raise RuntimeError(
                    f"{spec.family_label} compact-source single-slot payload rebuild returned no transform "
                    f"for source_block={int(source_block)} target_index={int(target_index)}"
                )
            return {(0, 0): {int(fallback_diag): torch.zeros((int(slots),), dtype=torch.float32)}}
        block = _transform_payload_block(
            rebuilt,
            slots=int(slots),
            context=(
                f"{spec.family_label} compact-source single-slot payload rebuild "
                f"source_block={int(source_block)} target_index={int(target_index)}"
            ),
        )
        if not block:
            if diag_set:
                raise RuntimeError(
                    f"{spec.family_label} compact-source single-slot payload rebuild returned empty diagonals "
                    f"for source_block={int(source_block)} target_index={int(target_index)}"
                )
            return {(0, 0): {int(fallback_diag): torch.zeros((int(slots),), dtype=torch.float32)}}
        return {(0, 0): block}

    compact_output = compact_target_block is not None
    return SimpleNamespace(
        name=(
            f"native_halo_{spec.family_label}_compactsrc{int(source_block)}"
            f"_s{int(stripe.index)}_tgt{int(target_group)}"
            + ("" if not bool(compact_output) else f"_compact{int(compact_target_block)}")
        ),
        diagonals={},
        _single_slot_diag_indices_by_block={(0, 0): tuple(sorted(int(value) for value in diag_set))},
        _single_slot_build_diagonals=build_diagonals,
        level=int(level),
        scheme=scheme,
        fhe_output_shape=torch.Size([1, int(slots)]),
        output_shape=torch.Size([1, int(slots)]),
        target_index=int(target_index),
        input_id=f"compact_source_block_{int(source_block)}",
        selected_n1=int(group_n1),
        baby_shifts=tuple(sorted(int(value) for value in baby)),
        giant_shifts=tuple(sorted(int(value) for value in giant)),
        rotation_group_id=f"native_halo:{spec.family_label}:compact_src{int(source_block)}",
        rotation_cost_owner=bool(int(target_group) == 0 and (not bool(compact_output) or int(compact_target_block) == 0)),
        _single_slot_diagonal_cache=diagonal_cache,
        _single_slot_release_diagonal_cache=(
            diagonal_cache.release
            if diagonal_cache is not None and bool(release_diagonal_cache_on_materialize)
            else None
        ),
        _concat_branch_diagonal_cache=diagonal_cache,
    )


def _build_compact_source_concat_transforms_single_slot(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    level: int,
    scheme: Any,
    source_layout: dict[str, Any],
    source_ct_count: int,
    target_ct_count: int,
    group_n1: int,
    build_diagonals_by_block: Callable[[], dict[tuple[int, int], dict[int, Any]]] | None = None,
    output_materialization: str = "",
    index_only: bool = False,
) -> dict[int, list[tuple[int, Any]]]:
    cpp_started = time.perf_counter()
    cpp_built = _build_compact_source_concat_indices_cpp(
        spec=spec,
        plan=plan,
        weight=weight,
        source_layout=dict(source_layout),
        source_ct_count=int(source_ct_count),
        target_ct_count=int(target_ct_count),
        output_materialization=str(output_materialization or ""),
    )
    if bool(index_only):
        if cpp_built is not _CPP_DIAG_BUILDER_FALLBACK and not _provider_diag_builder_shadow():
            result, metadata = cpp_built
            metadata = dict(metadata)
            metadata["diag_builder_wall_s"] = float(time.perf_counter() - cpp_started)
            return dict(_normalise_compact_source_concat_index_result(result))

    slots = int(spec.slot_count)
    source_channels = torch.arange(int(spec.c_in), dtype=torch.int64)
    source_top_beta = max(0, int(_layout_physical_top_beta(source_layout, default=spec.input_physical_top_beta or 0) or 0))
    source_bottom_beta = max(
        0,
        int(_layout_physical_bottom_beta(source_layout, default=spec.input_physical_bottom_beta or 0) or 0),
    )
    source_gap = max(1, int(source_layout.get("gap", spec.gap_in) or 1))
    source_height = int(spec.h_in) + int(source_top_beta) + int(source_bottom_beta)
    compact_output_h = _spec_physical_output_h(spec)
    out_w_values = torch.arange(int(spec.w_out), dtype=torch.int64)
    max_pair_count = int(_native_halo_build_pair_chunk_limit())
    precomputed_index_diag_sets: dict[tuple[int, int], set[int]] | None = None
    if cpp_built is not _CPP_DIAG_BUILDER_FALLBACK and not _provider_diag_builder_shadow():
        cpp_result, _metadata = cpp_built
        precomputed_index_diag_sets = _compact_source_concat_diag_sets_from_index_result(cpp_result)
    diag_sets: dict[tuple[int, int], set[int]] = {} if precomputed_index_diag_sets is None else dict(precomputed_index_diag_sets)
    fuse_output_relayout = str(output_materialization or "") == "fused_relayout"

    if precomputed_index_diag_sets is None:
        for stripe in plan.stripes:
            out_h_values = torch.arange(
                int(stripe.target_h_start),
                int(stripe.target_h_end),
                dtype=torch.int64,
            )
            op_out_h_values = (
                _materialized_output_source_h(
                    out_h_values,
                    h_out=int(spec.h_out),
                    output_top_beta=int(spec.output_top_beta),
                    output_bottom_beta=int(spec.output_bottom_beta),
                )
                if bool(fuse_output_relayout)
                else out_h_values
            )
            for target_group in range(int(plan.target_group_count_for_stripe(stripe))):
                target_tile = int(plan.target_tile_for_stripe(stripe))
                target_start = int(target_group) * int(target_tile)
                target_end = min(int(spec.c_out), int(target_start) + int(target_tile))
                target_count = int(target_end - target_start)
                if int(target_count) <= 0:
                    continue
                target_channels = torch.arange(int(target_start), int(target_end), dtype=torch.int64)
                for kh in range(int(spec.kernel)):
                    for kw in range(int(spec.kernel)):
                        coeff = weight[int(target_start): int(target_end), :, int(kh), int(kw)].to(dtype=torch.float32)
                        if not bool(torch.any(coeff != 0).item()):
                            continue
                        coeff_by_source_target = coeff.t().contiguous()
                        coeff_nonzero = torch.abs(coeff_by_source_target) > 0
                        if not bool(coeff_nonzero.any().item()):
                            continue
                        in_h_values = (
                            op_out_h_values * int(spec.stride)
                            - int(spec.pad)
                            + int(kh) * int(spec.dilation)
                        )
                        valid_h = (in_h_values >= 0) & (in_h_values < int(spec.h_in))
                        if not bool(valid_h.any().item()):
                            continue
                        in_w_values = (
                            out_w_values * int(spec.stride)
                            - int(spec.pad)
                            + int(kw) * int(spec.dilation)
                        )
                        valid_w = (in_w_values >= 0) & (in_w_values < int(spec.w_in))
                        if not bool(valid_w.any().item()):
                            continue

                        valid_out_h = out_h_values[valid_h]
                        valid_source_h = int(source_top_beta) + in_h_values[valid_h]
                        valid_out_w = out_w_values[valid_w]
                        valid_source_w = in_w_values[valid_w]
                        grid_h, grid_w = torch.meshgrid(valid_source_h, valid_source_w, indexing="ij")
                        out_grid_h, out_grid_w = torch.meshgrid(valid_out_h, valid_out_w, indexing="ij")
                        source_h_flat = grid_h.reshape(-1)
                        source_w_flat = grid_w.reshape(-1)
                        out_h_flat = out_grid_h.reshape(-1)
                        out_w_flat = out_grid_w.reshape(-1)
                        if int(source_h_flat.numel()) == 0:
                            continue

                        channel_pair_count = max(1, int(source_channels.numel()) * int(target_count))
                        position_chunk = max(1, int(max_pair_count // channel_pair_count))
                        for start in range(0, int(source_h_flat.numel()), int(position_chunk)):
                            end = min(int(source_h_flat.numel()), int(start) + int(position_chunk))
                            source_index = _idx_chw_gap_channel_positions(
                                source_channels,
                                h=source_h_flat[int(start): int(end)],
                                w=source_w_flat[int(start): int(end)],
                                height=int(source_height),
                                width=int(spec.w_in),
                                gap=int(source_gap),
                            )
                            source_blocks = torch.div(source_index, int(slots), rounding_mode="floor")
                            source_valid = (
                                (source_blocks >= 0)
                                & (source_blocks < int(source_ct_count))
                            )
                            if not bool(source_valid.any().item()):
                                continue
                            source_vec = torch.remainder(source_index, int(slots))

                            target_h_slice = out_h_flat[int(start): int(end)]
                            target_h_valid = _spec_physical_output_h_valid(spec, target_h_slice)
                            target_index = _idx_chw_gap_channel_positions(
                                target_channels,
                                h=_spec_physical_output_h_positions(spec, target_h_slice),
                                w=out_w_flat[int(start): int(end)],
                                height=int(compact_output_h),
                                width=int(spec.w_out),
                                gap=int(spec.gap_out),
                            )
                            target_blocks = torch.div(target_index, int(slots), rounding_mode="floor")
                            target_valid = (
                                (target_blocks >= 0)
                                & (target_blocks < int(target_ct_count))
                                & target_h_valid
                            )
                            if not bool(target_valid.any().item()):
                                continue
                            target_vec = torch.remainder(target_index, int(slots))
                            diag_index = (source_vec[:, None, :] - target_vec[None, :, :]).remainder(int(slots))
                            pair_mask = (
                                source_valid[:, None, :]
                                & target_valid[None, :, :]
                                & coeff_nonzero[:, :, None]
                            )
                            if not bool(pair_mask.any().item()):
                                continue
                            combined = (
                                (
                                    source_blocks[:, None, :] * int(target_ct_count)
                                    + target_blocks[None, :, :]
                                )
                                * int(slots)
                                + diag_index
                            )
                            for value in torch.unique(combined[pair_mask]).tolist():
                                encoded = int(value)
                                diag = int(encoded % int(slots))
                                pair = int(encoded // int(slots))
                                target_block = int(pair % int(target_ct_count))
                                source_block = int(pair // int(target_ct_count))
                                diag_sets.setdefault((int(source_block), int(target_block)), set()).add(int(diag))

    if bool(index_only):
        python_result = {
            int(source_block): [
                (int(target_block), tuple(sorted(int(value) for value in diag_set)))
                for target_block, diag_set in sorted(target_items, key=lambda item: int(item[0]))
                if diag_set
            ]
            for source_block, target_items in sorted(
                (
                    (
                        int(source_block),
                        [
                            (int(target_block), set(int(value) for value in diag_set))
                            for (candidate_source, target_block), diag_set in sorted(diag_sets.items())
                            if int(candidate_source) == int(source_block)
                        ],
                    )
                    for source_block in sorted({int(source_block) for source_block, _target_block in diag_sets})
                ),
                key=lambda item: int(item[0]),
            )
        }
        if cpp_built is not _CPP_DIAG_BUILDER_FALLBACK and _provider_diag_builder_shadow():
            cpp_result, _metadata = cpp_built
            normal_cpp = _normalise_compact_source_concat_index_result(cpp_result)
            normal_python = _normalise_compact_source_concat_index_result(python_result)
            if normal_cpp != normal_python:
                if _provider_diag_builder_strict():
                    raise RuntimeError("provider compact-source concat C++ index mismatch")
                return python_result
        return python_result

    ordered: dict[int, list[tuple[int, Any]]] = {}
    representative_stripe = plan.stripes[0]
    diagonal_cache = (
        _BlockDiagonalCache(build_diagonals_by_block)
        if callable(build_diagonals_by_block)
        else None
    )
    for (source_block, target_block), diag_set in sorted(diag_sets.items()):
        if not diag_set:
            continue
        transform = _make_compact_source_conv_single_slot_transform(
            spec=spec,
            plan=plan,
            weight=weight,
            stripe=representative_stripe,
            source_block=int(source_block),
            target_group=0,
            level=int(level),
            scheme=scheme,
            source_layout=dict(source_layout),
            group_n1=int(group_n1),
            target_index=int(target_block),
            diag_set=set(int(value) for value in diag_set),
            compact_target_block=int(target_block),
            diagonal_cache=diagonal_cache,
            release_diagonal_cache_on_materialize=False,
        )
        ordered.setdefault(int(source_block), []).append((int(target_block), transform))
    return ordered


def native_halo_conv2d_spec_from_module(module: Any, *, output_node_id: str) -> NativeHaloConv2DSpec | None:
    weight = getattr(module, "on_weight", None)
    weight_shape = _safe_shape(getattr(weight, "shape", ()))
    input_shape = _safe_shape(getattr(module, "input_shape", ()))
    output_shape = _safe_shape(getattr(module, "output_shape", ()))
    if len(weight_shape) != 4 or len(input_shape) < 4 or len(output_shape) < 4:
        return None
    out_channels, in_per_group, kernel_h, kernel_w = (int(value) for value in weight_shape)
    stride = _pair(getattr(module, "stride", (1, 1)))
    pad = _pair(getattr(module, "padding", (0, 0)), default=(0, 0))
    dilation = _pair(getattr(module, "dilation", (1, 1)))
    groups = int(getattr(module, "groups", 1))
    if int(groups) != 1:
        return None
    if int(kernel_h) != int(kernel_w) or int(stride[0]) != int(stride[1]):
        return None
    if int(pad[0]) != int(pad[1]) or int(dilation[0]) != int(dilation[1]):
        return None
    if int(input_shape[1]) != int(in_per_group) or int(output_shape[1]) != int(out_channels):
        return None
    input_gap = int(getattr(module, "input_gap", 1))
    output_gap = int(getattr(module, "output_gap", input_gap))
    input_layout = dict(getattr(module, "layout_policy_input_layout", {}) or {})
    output_layout = dict(getattr(module, "layout_policy_output_layout", {}) or {})
    explicit_slot_count = int(getattr(module, "layout_policy_slot_count", 0) or 0)
    if int(explicit_slot_count) > 0:
        slot_count = int(explicit_slot_count)
    else:
        params = getattr(getattr(module, "scheme", None), "params", None)
        get_slots = getattr(params, "get_slots", None)
        try:
            slot_count = int(get_slots()) if callable(get_slots) else int(RING_SLOT_COUNT)
        except Exception:
            slot_count = int(RING_SLOT_COUNT)
    raw_label = str(getattr(module, "name", "") or output_node_id or "conv")
    label = "".join(ch if ch.isalnum() else "_" for ch in raw_label).strip("_") or "conv"
    return NativeHaloConv2DSpec(
        family_label=(
            f"native_halo_{label}_{int(input_shape[1])}x{int(input_shape[2])}x{int(input_shape[3])}"
            f"_to_{int(output_shape[1])}x{int(output_shape[2])}x{int(output_shape[3])}"
            f"_k{int(kernel_h)}s{int(stride[0])}_gap{int(input_gap)}to{int(output_gap)}"
        ),
        c_in=int(input_shape[1]),
        h_in=int(input_shape[2]),
        w_in=int(input_shape[3]),
        c_out=int(output_shape[1]),
        h_out=int(output_shape[2]),
        w_out=int(output_shape[3]),
        gap_in=int(input_gap),
        gap_out=int(output_gap),
        kernel=int(kernel_h),
        stride=int(stride[0]),
        pad=int(pad[0]),
        dilation=int(dilation[0]),
        groups=int(groups),
        slot_count=int(slot_count),
        input_top_beta=_layout_top_beta(input_layout),
        input_bottom_beta=_layout_bottom_beta(input_layout),
        output_top_beta=_layout_top_beta(output_layout),
        output_bottom_beta=_layout_bottom_beta(output_layout),
        input_physical_top_beta=_layout_physical_top_beta(input_layout),
        input_physical_bottom_beta=_layout_physical_bottom_beta(input_layout),
        output_physical_top_beta=_layout_physical_top_beta(output_layout),
        output_physical_bottom_beta=_layout_physical_bottom_beta(output_layout),
    )


def _target_h_end_for_source_h(
    spec: NativeHaloConv2DSpec,
    *,
    target_h_start: int,
    source_h: int,
) -> int:
    """Greedily cover as many target rows as one source stripe can support."""

    lo = int(target_h_start) + 1
    hi = int(spec.output_h_max)
    best = int(lo)
    while int(lo) <= int(hi):
        mid = (int(lo) + int(hi)) // 2
        req0, req1 = _source_h_range_for_target(
            target_h_start=int(target_h_start),
            target_h_end=int(mid),
            input_h_min=int(spec.input_h_min),
            input_h_max=int(spec.input_h_max),
            kernel=int(spec.kernel),
            stride=int(spec.stride),
            pad=int(spec.pad),
            dilation=int(spec.dilation),
        )
        req0, req1 = _source_h_range_with_physical_input_halo(
            spec,
            required_start=int(req0),
            required_end=int(req1),
        )
        if int(req1 - req0) <= int(source_h):
            best = int(mid)
            lo = int(mid) + 1
        else:
            hi = int(mid) - 1
    return int(best)


def _stripes_for_source_h(spec: NativeHaloConv2DSpec, *, source_h: int) -> tuple[NativeHaloStripe, ...]:
    stripes: list[NativeHaloStripe] = []
    target_h = int(spec.output_h_min)
    while int(target_h) < int(spec.output_h_max):
        th0 = int(target_h)
        th1 = _target_h_end_for_source_h(spec, target_h_start=int(th0), source_h=int(source_h))
        req0, req1 = _source_h_range_for_target(
            target_h_start=int(th0),
            target_h_end=int(th1),
            input_h_min=int(spec.input_h_min),
            input_h_max=int(spec.input_h_max),
            kernel=int(spec.kernel),
            stride=int(spec.stride),
            pad=int(spec.pad),
            dilation=int(spec.dilation),
        )
        req0, req1 = _source_h_range_with_physical_input_halo(
            spec,
            required_start=int(req0),
            required_end=int(req1),
        )
        sh0, sh1 = _extend_h_range_to_length(
            required_start=int(req0),
            required_end=int(req1),
            desired_len=int(source_h),
            lower=int(spec.input_h_min),
            upper=int(spec.input_h_max),
        )
        stripes.append(
            NativeHaloStripe(
                index=int(len(stripes)),
                target_h_start=int(th0),
                target_h_end=int(th1),
                source_h_start=int(sh0),
                source_h_end=int(sh1),
            )
        )
        target_h = int(th1)
    return tuple(stripes)


def _diag_indices_for_task(
    spec: NativeHaloConv2DSpec,
    stripe: NativeHaloStripe,
    *,
    source_channel_count: int,
    target_channel_count: int,
) -> set[int]:
    key = (
        "diag_indices",
        _native_spec_structural_cache_key(spec),
        _native_stripe_cache_key(stripe),
        int(source_channel_count),
        int(target_channel_count),
    )
    cached = _NATIVE_DIAG_INDICES_CACHE.get(key)
    if cached is not None:
        return set(int(value) for value in cached)
    _native_diag_cache_guard()
    shifts = _native_diag_indices_closed_form(
        spec,
        stripe,
        source_channel_count=int(source_channel_count),
        target_channel_count=int(target_channel_count),
    )
    _NATIVE_DIAG_INDICES_CACHE[key] = tuple(sorted(int(value) for value in shifts))
    return set(int(value) for value in shifts)


def _diag_indices_for_task_torch_oracle(
    spec: NativeHaloConv2DSpec,
    stripe: NativeHaloStripe,
    *,
    source_channel_count: int,
    target_channel_count: int,
) -> set[int]:
    source_slots = _slot_indices(
        int(source_channel_count),
        int(stripe.source_h),
        int(spec.w_in),
        int(spec.gap_in),
    )
    target_slots = _slot_indices(
        int(target_channel_count),
        int(stripe.target_h),
        int(spec.w_out),
        int(spec.gap_out),
    )
    shifts: set[int] = set()
    out_h_values = torch.arange(int(stripe.target_h_start), int(stripe.target_h_end), dtype=torch.int64)
    target_local_h_values = out_h_values - int(stripe.target_h_start)
    out_w_values = torch.arange(int(spec.w_out), dtype=torch.int64)
    for kh in range(int(spec.kernel)):
        in_h_values = (
            out_h_values * int(spec.stride)
            - int(spec.pad)
            + int(kh) * int(spec.dilation)
        )
        source_local_h_values = in_h_values - int(stripe.source_h_start)
        valid_h = (
            (in_h_values >= int(spec.input_h_min))
            & (in_h_values < int(spec.input_h_max))
            & (source_local_h_values >= 0)
            & (source_local_h_values < int(stripe.source_h))
        )
        if not bool(valid_h.any().item()):
            continue
        source_local_h = source_local_h_values[valid_h]
        target_local_h = target_local_h_values[valid_h]
        for kw in range(int(spec.kernel)):
            in_w_values = (
                out_w_values * int(spec.stride)
                - int(spec.pad)
                + int(kw) * int(spec.dilation)
            )
            valid_w = (in_w_values >= 0) & (in_w_values < int(spec.w_in))
            if not bool(valid_w.any().item()):
                continue
            source_w = in_w_values[valid_w]
            target_w = out_w_values[valid_w]
            source_values = source_slots[
                :,
                source_local_h[:, None],
                source_w[None, :],
            ]
            target_values = target_slots[
                :,
                target_local_h[:, None],
                target_w[None, :],
            ]
            diff = (
                source_values[:, None, :, :]
                - target_values[None, :, :, :]
            ).reshape(-1).remainder(int(spec.slot_count))
            shifts.update(int(value) for value in torch.unique(diff).tolist())
    shifts.discard(0)
    return shifts


_PLAN_CACHE: dict[tuple[Any, ...], NativeHaloConv2DPlan] = {}

_CACHE_PLAN_GUARD_KEYS = (
    "spec",
    "input_ct_count",
    "output_ct_count",
    "source_channel_group_count",
    "target_channel_group_count",
    "source_channel_group_counts",
    "target_channel_group_counts",
    "source_stripe_offsets",
    "target_stripe_offsets",
    "stripes",
)


def _cache_plan_guard(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: plan.get(key) for key in _CACHE_PLAN_GUARD_KEYS}


def _normalise_channel_fold_mode(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"", "heuristic", "native", "uniform"}:
        return "heuristic"
    if text in {"per_stripe", "perstripe", "variable", "variable_per_stripe"}:
        return "per_stripe"
    raise ValueError(f"unsupported native halo channel fold mode: {value!r}")


def _channel_tile_candidates(channel_count: int, gap: int) -> tuple[int, ...]:
    phase = _phase_count(int(gap))
    max_fold = max(1, _ceil_div(int(channel_count), int(phase)))
    folds = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, int(max_fold)]
    tiles = {
        min(int(channel_count), int(phase) * int(fold))
        for fold in folds
        if 1 <= int(fold) <= int(max_fold)
    }
    return tuple(sorted(int(tile) for tile in tiles if int(tile) > 0))


def _source_h_capacity_for_tile(spec: NativeHaloConv2DSpec, source_tile: int) -> int:
    source_groups_per_tile = _ceil_div(int(source_tile), _phase_count(int(spec.gap_in)))
    denom = int(source_groups_per_tile) * int(spec.w_in) * _phase_count(int(spec.gap_in))
    input_total_h = int(spec.input_h_max) - int(spec.input_h_min)
    return min(int(input_total_h), max(1, int(spec.slot_count) // max(1, int(denom))))


def _with_stripe_channel_tiles(
    stripe: NativeHaloStripe,
    *,
    source_tile: int,
    target_tile: int,
    source_h_start: int | None = None,
    source_h_end: int | None = None,
) -> NativeHaloStripe:
    return NativeHaloStripe(
        index=int(stripe.index),
        target_h_start=int(stripe.target_h_start),
        target_h_end=int(stripe.target_h_end),
        source_h_start=int(stripe.source_h_start if source_h_start is None else source_h_start),
        source_h_end=int(stripe.source_h_end if source_h_end is None else source_h_end),
        source_channel_tile=int(source_tile),
        target_channel_tile=int(target_tile),
    )


def _per_stripe_fold_stripes(
    spec: NativeHaloConv2DSpec,
    *,
    base_source_tile: int,
    base_target_tile: int,
) -> tuple[NativeHaloStripe, ...]:
    cache_key = (
        _native_spec_structural_cache_key(spec),
        int(base_source_tile),
        int(base_target_tile),
    )
    cached = _PER_STRIPE_FOLD_STRIPES_CACHE.get(cache_key)
    if cached is not None:
        return tuple(cached)
    if len(_PER_STRIPE_FOLD_STRIPES_CACHE) > 20000:
        _PER_STRIPE_FOLD_STRIPES_CACHE.clear()
    base_source_h = _source_h_capacity_for_tile(spec, int(base_source_tile))
    target_stripes = _stripes_for_source_h(spec, source_h=int(base_source_h))
    source_tiles = _channel_tile_candidates(int(spec.c_in), int(spec.gap_in))
    target_tiles = _channel_tile_candidates(int(spec.c_out), int(spec.gap_out))
    diag_cache: dict[tuple[int, int, int, int, int], set[int]] = {}
    bsgs_cache: dict[tuple[int, ...], int] = {}
    selected: list[NativeHaloStripe] = []
    for target_stripe in target_stripes:
        req0, req1 = _source_h_range_for_target(
            target_h_start=int(target_stripe.target_h_start),
            target_h_end=int(target_stripe.target_h_end),
            input_h_min=int(spec.input_h_min),
            input_h_max=int(spec.input_h_max),
            kernel=int(spec.kernel),
            stride=int(spec.stride),
            pad=int(spec.pad),
            dilation=int(spec.dilation),
        )
        req0, req1 = _source_h_range_with_physical_input_halo(
            spec,
            required_start=int(req0),
            required_end=int(req1),
        )
        req_h = int(req1 - req0)
        candidate_specs: list[tuple[tuple[int, int, int, int], int, int, NativeHaloStripe]] = []
        best: tuple[tuple[int, int, int, int], NativeHaloStripe] | None = None
        for source_tile in source_tiles:
            source_h_capacity = _source_h_capacity_for_tile(spec, int(source_tile))
            if int(req_h) <= 0 or int(req_h) > int(source_h_capacity):
                continue
            for target_tile in target_tiles:
                candidate = _with_stripe_channel_tiles(
                    target_stripe,
                    source_tile=int(source_tile),
                    target_tile=int(target_tile),
                    source_h_start=int(req0),
                    source_h_end=int(req1),
                )
                if _packed_active_slots(int(source_tile), int(candidate.source_h), int(spec.w_in), int(spec.gap_in)) > int(spec.slot_count):
                    continue
                if _packed_active_slots(int(target_tile), int(candidate.target_h), int(spec.w_out), int(spec.gap_out)) > int(spec.slot_count):
                    continue
                source_group_count = _ceil_div(int(spec.c_in), int(source_tile))
                target_group_count = _ceil_div(int(spec.c_out), int(target_tile))
                programs = int(source_group_count) * int(target_group_count)
                candidate_specs.append(
                    (
                        (
                            int(programs),
                            int(source_group_count + target_group_count),
                            -int(source_tile),
                            -int(target_tile),
                        ),
                        int(source_group_count),
                        int(target_group_count),
                        candidate,
                    )
                )
        for _approx_score, source_group_count, target_group_count, candidate in sorted(candidate_specs, key=lambda item: item[0])[:32]:
            source_tile = int(candidate.source_channel_tile)
            target_tile = int(candidate.target_channel_tile)
            rotations = 0
            programs = 0
            source_counts = Counter(
                min(int(spec.c_in), int(source_group + 1) * int(source_tile)) - int(source_group) * int(source_tile)
                for source_group in range(int(source_group_count))
            )
            target_counts = Counter(
                min(int(spec.c_out), int(target_group + 1) * int(target_tile)) - int(target_group) * int(target_tile)
                for target_group in range(int(target_group_count))
            )
            for source_count, source_multiplicity in sorted(source_counts.items()):
                for target_count, target_multiplicity in sorted(target_counts.items()):
                    key = (
                        int(candidate.target_h),
                        int(candidate.source_h),
                        int(candidate.source_h_start),
                        int(source_count),
                        int(target_count),
                    )
                    if key not in diag_cache:
                        diag_cache[key] = _diag_indices_for_task(
                            spec,
                            candidate,
                            source_channel_count=int(source_count),
                            target_channel_count=int(target_count),
                        )
                    diag_indices = diag_cache[key]
                    bkey = tuple(sorted(int(value) for value in diag_indices))
                    if bkey not in bsgs_cache:
                        _n1, cost, _baby, _giant = _cached_native_best_common_bsgs((diag_indices,), slots=int(spec.slot_count))
                        bsgs_cache[bkey] = int(cost)
                    multiplicity = int(source_multiplicity) * int(target_multiplicity)
                    rotations += int(multiplicity) * int(bsgs_cache[bkey])
                    programs += int(multiplicity)
            score = (
                int(rotations),
                int(source_group_count + target_group_count),
                int(programs),
                -int(source_tile),
            )
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is None:
            selected.append(
                _with_stripe_channel_tiles(
                    target_stripe,
                    source_tile=int(base_source_tile),
                    target_tile=int(base_target_tile),
                )
            )
        else:
            selected.append(best[1])
    result = tuple(selected)
    _PER_STRIPE_FOLD_STRIPES_CACHE[cache_key] = result
    return result


def native_halo_conv2d_plan(
    spec: NativeHaloConv2DSpec,
    *,
    require_native_target_fit: bool = True,
    channel_fold_mode: str | None = None,
) -> NativeHaloConv2DPlan:
    fold_mode = _normalise_channel_fold_mode(channel_fold_mode)
    key = (_native_spec_structural_cache_key(spec), bool(require_native_target_fit), str(fold_mode))
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    source_tile = _heuristic_channel_tile(int(spec.c_in), int(spec.gap_in))
    target_tile = _heuristic_channel_tile(int(spec.c_out), int(spec.gap_out))
    source_h = _source_h_capacity_for_tile(spec, int(source_tile))
    if _packed_active_slots(int(source_tile), int(source_h), int(spec.w_in), int(spec.gap_in)) > int(spec.slot_count):
        raise ValueError(f"native halo source tile does not fit {spec.family_label}")
    if str(fold_mode) == "per_stripe":
        stripes = _per_stripe_fold_stripes(
            spec,
            base_source_tile=int(source_tile),
            base_target_tile=int(target_tile),
        )
    else:
        stripes = tuple(
            _with_stripe_channel_tiles(
                stripe,
                source_tile=int(source_tile),
                target_tile=int(target_tile),
            )
            for stripe in _stripes_for_source_h(spec, source_h=int(source_h))
        )
    if bool(require_native_target_fit) and any(
        _packed_active_slots(
            int(target_tile if not int(stripe.target_channel_tile or 0) else stripe.target_channel_tile),
            int(stripe.target_h),
            int(spec.w_out),
            int(spec.gap_out),
        )
        > int(spec.slot_count)
        for stripe in stripes
    ):
        raise ValueError(f"native halo target tile does not fit {spec.family_label}")
    diag_cache: dict[tuple[int, int, int], set[int]] = {}
    program_bsgs_cache: dict[tuple[int, ...], tuple[int, int, int, int]] = {}
    group_bsgs_cache: dict[tuple[tuple[int, ...], ...], tuple[int, int, int, int]] = {}
    program_diags: list[int] = []
    program_rots: list[int] = []
    group_n1s: list[int] = []
    group_rots: list[int] = []
    group_baby: list[int] = []
    group_giant: list[int] = []
    for stripe in stripes:
        stripe_source_tile = int(stripe.source_channel_tile or source_tile)
        stripe_target_tile = int(stripe.target_channel_tile or target_tile)
        source_group_count = _ceil_div(int(spec.c_in), int(stripe_source_tile))
        target_group_count = _ceil_div(int(spec.c_out), int(stripe_target_tile))
        for source_group in range(int(source_group_count)):
            source_start = int(source_group) * int(stripe_source_tile)
            source_end = min(int(spec.c_in), int(source_start) + int(stripe_source_tile))
            entries: list[set[int]] = []
            entry_keys: list[tuple[int, ...]] = []
            for target_group in range(int(target_group_count)):
                target_start = int(target_group) * int(stripe_target_tile)
                target_end = min(int(spec.c_out), int(target_start) + int(stripe_target_tile))
                diag_key = (
                    int(stripe.index),
                    int(source_end - source_start),
                    int(target_end - target_start),
                )
                if diag_key not in diag_cache:
                    diag_cache[diag_key] = _diag_indices_for_task(
                        spec,
                        stripe,
                        source_channel_count=int(source_end - source_start),
                        target_channel_count=int(target_end - target_start),
                    )
                diag_indices = set(diag_cache[diag_key])
                entries.append(diag_indices)
                bsgs_key = tuple(sorted(int(value) for value in diag_indices))
                entry_keys.append(bsgs_key)
                if bsgs_key not in program_bsgs_cache:
                    program_bsgs_cache[bsgs_key] = _cached_native_best_common_bsgs(
                        (diag_indices,),
                        slots=int(spec.slot_count),
                    )
                _n1, rotations, _baby, _giant = program_bsgs_cache[bsgs_key]
                program_diags.append(int(len(diag_indices)))
                program_rots.append(int(rotations))
            group_key = tuple(entry_keys)
            if group_key not in group_bsgs_cache:
                group_bsgs_cache[group_key] = _cached_native_best_common_bsgs(
                    tuple(entries),
                    slots=int(spec.slot_count),
                )
            n1, rotations, baby, giant = group_bsgs_cache[group_key]
            group_n1s.append(int(n1))
            group_rots.append(int(rotations))
            group_baby.append(int(baby))
            group_giant.append(int(giant))

    plan = NativeHaloConv2DPlan(
        spec=spec,
        source_channel_tile=int(source_tile),
        target_channel_tile=int(target_tile),
        stripes=tuple(stripes),
        program_diagonal_counts=tuple(program_diags),
        program_rotation_counts=tuple(program_rots),
        group_n1s=tuple(group_n1s),
        group_shared_rotations=tuple(group_rots),
        group_baby_rotations=tuple(group_baby),
        group_giant_rotations=tuple(group_giant),
        channel_fold_mode=str(fold_mode),
    )
    _PLAN_CACHE[key] = plan
    return plan


def native_halo_source_plaintext_blocks_from_nchw(
    x: torch.Tensor,
    plan: NativeHaloConv2DPlan,
) -> list[torch.Tensor]:
    """Materialize a clear NCHW tensor into native halo source stripe blocks.

    The native Conv2d executor consumes one ciphertext per
    (height-stripe, source-channel-group).  Encrypted relayout can build that
    representation from a compact ciphertext, but the model input should be
    encoded this way directly so the first convolution does not spend a level
    and rotations merely to reshape plaintext data.
    """

    spec = plan.spec
    values = x.detach().cpu().to(dtype=torch.float32)
    if values.dim() == 3:
        values = values.unsqueeze(0)
    if values.dim() != 4:
        raise ValueError(f"native halo plaintext input expects NCHW, got {tuple(values.shape)}")
    if int(values.shape[0]) != 1:
        raise ValueError("native halo plaintext input currently supports batch size 1")
    if (
        int(values.shape[1]) < int(spec.c_in)
        or int(values.shape[2]) != int(spec.h_in)
        or int(values.shape[3]) != int(spec.w_in)
    ):
        raise ValueError(
            "native halo plaintext input shape does not match plan: "
            f"got {tuple(int(v) for v in values.shape)}, expected "
            f"(1, >= {int(spec.c_in)}, {int(spec.h_in)}, {int(spec.w_in)})"
        )

    src = values[0]
    slots = int(spec.slot_count)
    blocks: list[torch.Tensor] = []
    for stripe in plan.stripes:
        source_tile = int(plan.source_tile_for_stripe(stripe))
        for group in range(int(plan.source_group_count_for_stripe(stripe))):
            block = torch.zeros((int(slots),), dtype=torch.float32)
            channel_start = int(group) * int(source_tile)
            channel_end = min(int(spec.c_in), int(channel_start) + int(source_tile))
            for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
                for local_h in range(int(stripe.source_h)):
                    global_h = int(stripe.source_h_start) + int(local_h)
                    if int(global_h) < 0 or int(global_h) >= int(spec.h_in):
                        continue
                    for w_index in range(int(spec.w_in)):
                        slot = _idx_chw_gap(
                            int(local_channel),
                            int(local_h),
                            int(w_index),
                            int(stripe.source_h),
                            int(spec.w_in),
                            int(spec.gap_in),
                        )
                        block[int(slot)] = src[int(channel), int(global_h), int(w_index)]
            blocks.append(block)
    if len(blocks) != int(plan.input_ct_count):
        raise RuntimeError(
            f"native halo plaintext materialization produced {len(blocks)} blocks, "
            f"expected {int(plan.input_ct_count)}"
        )
    return blocks


class NativeHaloRelayoutKernel:
    def __init__(
        self,
        *,
        plan: NativeHaloConv2DPlan,
        direction: Literal["compact_to_native", "native_to_compact"],
        name: str,
        output_shape: torch.Size,
        fhe_output_shape: torch.Size,
        source_layout: dict[str, Any] | None = None,
    ) -> None:
        if str(direction) not in {"compact_to_native", "native_to_compact"}:
            raise ValueError(f"unknown native halo relayout direction {direction!r}")
        self.plan = plan
        self.spec = plan.spec
        self.direction = str(direction)
        self.name = str(name)
        self.output_shape = torch.Size(output_shape)
        self.fhe_output_shape = torch.Size(fhe_output_shape)
        self.source_layout = dict(source_layout or {})
        self.level: int | None = None
        self.bsgs_ratio = 2.0
        self.diagonals: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
        self.transform_ids: dict[tuple[int, int], int] = {}

    def _compact_input_index(self, *, channel: int, h: int, w: int) -> int | None:
        source_top_beta = _layout_physical_top_beta(
            self.source_layout,
            default=int(self.spec.input_physical_top_beta or 0),
        )
        source_bottom_beta = _layout_physical_bottom_beta(
            self.source_layout,
            default=int(self.spec.input_physical_bottom_beta or 0),
        )
        if int(h) < 0:
            if int(h) < -int(source_top_beta):
                return None
            source_h = int(h) + int(source_top_beta)
        elif int(h) >= int(self.spec.h_in):
            if int(h) >= int(self.spec.h_in) + int(source_bottom_beta):
                return None
            source_h = int(source_top_beta) + int(h)
        else:
            source_h = int(source_top_beta) + int(h)
        source_height = int(self.spec.h_in) + int(source_top_beta) + int(source_bottom_beta)
        if int(source_h) < 0 or int(source_h) >= int(source_height):
            return None
        return _idx_chw_gap(
            int(channel),
            int(source_h),
            int(w),
            int(self.spec.h_in) + int(source_top_beta) + int(source_bottom_beta),
            int(self.spec.w_in),
            int(self.spec.gap_in),
        )

    def _compact_output_index(self, *, channel: int, h: int, w: int) -> int:
        if not bool(_spec_physical_output_h_valid(self.spec, int(h))):
            return -1
        return _idx_chw_gap(
            int(channel),
            int(_spec_physical_output_h_positions(self.spec, int(h))),
            int(w),
            _spec_physical_output_h(self.spec),
            int(self.spec.w_out),
            int(self.spec.gap_out),
        )

    def _native_source_index(self, *, stripe: NativeHaloStripe, group: int, channel: int, h: int, w: int) -> int:
        block = int(self.plan.source_block_index(stripe, int(group)))
        return int(block) * int(self.spec.slot_count) + _idx_chw_gap(
            int(channel),
            int(h),
            int(w),
            int(stripe.source_h),
            int(self.spec.w_in),
            int(self.spec.gap_in),
        )

    def _native_target_index(self, *, stripe: NativeHaloStripe, group: int, channel: int, h: int, w: int) -> int:
        block = int(self.plan.target_block_index(stripe, int(group)))
        return int(block) * int(self.spec.slot_count) + _idx_chw_gap(
            int(channel),
            int(h),
            int(w),
            int(stripe.target_h),
            int(self.spec.w_out),
            int(self.spec.gap_out),
        )

    def _iter_compact_to_native(self):
        for stripe in self.plan.stripes:
            source_tile = int(self.plan.source_tile_for_stripe(stripe))
            for group in range(int(self.plan.source_group_count_for_stripe(stripe))):
                channel_start = int(group) * int(source_tile)
                channel_end = min(int(self.spec.c_in), int(channel_start) + int(source_tile))
                for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
                    for global_h in range(int(stripe.source_h_start), int(stripe.source_h_end)):
                        local_h = int(global_h) - int(stripe.source_h_start)
                        for w_index in range(int(self.spec.w_in)):
                            source_index = self._compact_input_index(
                                channel=int(channel),
                                h=int(global_h),
                                w=int(w_index),
                            )
                            if source_index is None:
                                continue
                            yield (
                                int(source_index),
                                self._native_source_index(
                                    stripe=stripe,
                                    group=int(group),
                                    channel=int(local_channel),
                                    h=int(local_h),
                                    w=int(w_index),
                                ),
                            )

    def _iter_native_to_compact(self):
        for stripe in self.plan.stripes:
            target_tile = int(self.plan.target_tile_for_stripe(stripe))
            for group in range(int(self.plan.target_group_count_for_stripe(stripe))):
                channel_start = int(group) * int(target_tile)
                channel_end = min(int(self.spec.c_out), int(channel_start) + int(target_tile))
                for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
                    for global_h in range(int(stripe.target_h_start), int(stripe.target_h_end)):
                        local_h = int(global_h) - int(stripe.target_h_start)
                        for w_index in range(int(self.spec.w_out)):
                            compact_output_index = self._compact_output_index(
                                channel=int(channel),
                                h=int(global_h),
                                w=int(w_index),
                            )
                            if int(compact_output_index) < 0:
                                continue
                            yield (
                                self._native_target_index(
                                    stripe=stripe,
                                    group=int(group),
                                    channel=int(local_channel),
                                    h=int(local_h),
                                    w=int(w_index),
                                ),
                                int(compact_output_index),
                            )

    def _iter_mappings(self):
        return self._iter_compact_to_native() if self.direction == "compact_to_native" else self._iter_native_to_compact()

    def _diag_indices_by_block(self, slots: int) -> dict[tuple[int, int], tuple[int, ...]]:
        indices: dict[tuple[int, int], set[int]] = {}
        for source_index, output_index in self._iter_mappings():
            input_block = int(source_index // int(slots))
            output_block = int(output_index // int(slots))
            source_local = int(source_index % int(slots))
            output_local = int(output_index % int(slots))
            diag_index = int((source_local - output_local) % int(slots))
            indices.setdefault((int(output_block), int(input_block)), set()).add(int(diag_index))
        return {
            block: tuple(sorted(values)) if values else (0,)
            for block, values in sorted(indices.items())
        }

    def _build_diagonals(self, slots: int) -> dict[tuple[int, int], dict[int, torch.Tensor]]:
        diagonals: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
        for source_index, output_index in self._iter_mappings():
            input_block = int(source_index // int(slots))
            output_block = int(output_index // int(slots))
            source_local = int(source_index % int(slots))
            output_local = int(output_index % int(slots))
            diag_index = int((source_local - output_local) % int(slots))
            block = diagonals.setdefault((int(output_block), int(input_block)), {})
            diag = block.get(int(diag_index))
            if diag is None:
                diag = torch.zeros((int(slots),), dtype=torch.float32)
                block[int(diag_index)] = diag
            diag[int(output_local)] = 1.0
        return diagonals

    def compile(self, scheme: Any, *, level: int) -> None:
        self.cleanup(getattr(scheme, "backend", None))
        slots = int(self.spec.slot_count)
        self.level = int(level)
        if scheme.lt_evaluator.single_slot_layer_cache_enabled():
            self._dense_layer_cache_diag_indices_by_block = self._diag_indices_by_block(int(slots))
            self._dense_layer_cache_build_diagonals = lambda self=self, slots=int(slots): self._build_diagonals(int(slots))
            self.diagonals = {}
        else:
            self.diagonals = self._build_diagonals(int(slots))
        self.transform_ids = {
            (int(row), int(col)): int(transform_id)
            for (row, col), transform_id in scheme.lt_evaluator.generate_transforms(self).items()
        }

    def apply(self, source_ct: Any) -> Any:
        if not self.transform_ids and not getattr(self, "_dense_layer_cache_deferred", False):
            raise RuntimeError(f"native halo relayout kernel {self.name} has not been compiled")
        return source_ct.scheme.lt_evaluator.evaluate_transforms(self, source_ct)

    def _diag_indices_for_metadata(self) -> dict[tuple[int, int], tuple[int, ...]]:
        indices = getattr(self, "_dense_layer_cache_diag_indices_by_block", None)
        if indices:
            return {
                (int(row), int(col)): tuple(int(value) for value in values)
                for (row, col), values in dict(indices).items()
            }
        return {
            (int(row), int(col)): tuple(int(value) for value in dict(block or {}).keys())
            for (row, col), block in dict(self.diagonals or {}).items()
        }

    def operation_estimate(self) -> dict[str, int | str]:
        diag_indices = self._diag_indices_for_metadata()
        return {
            "kind": "native_halo_physical_relayout_lt",
            "rotation_count": int(sum(len(block) for block in diag_indices.values())),
            "mask_mult_count": 0,
            "sparse_lt_count": int(len(diag_indices) or len(self.transform_ids) or 1),
        }

    def to_metadata(self) -> dict[str, Any]:
        diag_indices = self._diag_indices_for_metadata()
        block_keys = set(diag_indices) | set(dict(self.transform_ids or {}).keys())
        return {
            "direction": str(self.direction),
            "name": str(self.name),
            "level": None if self.level is None else int(self.level),
            "source_layout": dict(self.source_layout),
            "rows": int(max((int(row) for row, _col in block_keys), default=-1) + 1),
            "cols": int(max((int(col) for _row, col in block_keys), default=-1) + 1),
            "lt_tasks": int(len(block_keys)),
            "diagonal_count": int(sum(len(block) for block in diag_indices.values())),
        }

    def cleanup(self, backend: Any | None) -> None:
        if backend is not None:
            delete = getattr(backend, "DeleteLinearTransform", None)
            if callable(delete):
                for transform_id in list(self.transform_ids.values()):
                    try:
                        delete(int(transform_id))
                    except Exception:
                        pass
        self.transform_ids = {}
        self.diagonals = {}
        self._dense_layer_cache_diag_indices_by_block = {}
        self._dense_layer_cache_build_diagonals = None


def _build_conv_transform(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    weight_np: np.ndarray | None = None,
    stripe: NativeHaloStripe,
    source_group: int,
    target_group: int,
    level: int,
    scheme: Any,
    group_n1: int,
    compact_target_block: int | None = None,
    diagonal_cache: _BlockDiagonalCache | None = None,
    force_payload: bool = False,
) -> Any | None:
    slots = int(spec.slot_count)
    compact_output = compact_target_block is not None
    single_slot_recipe = bool(_single_slot_layer_cache_enabled_for_scheme(scheme) and not bool(force_payload))
    target_index = (
        int(compact_target_block)
        if bool(compact_output)
        else int(plan.target_block_index(stripe, int(target_group)))
    )
    source_index = int(plan.source_block_index(stripe, int(source_group)))

    def build_all_native_source_blocks(
        *,
        spec=spec,
        plan=plan,
        weight=weight,
        weight_np=weight_np,
        stripe=stripe,
        source_group=int(source_group),
        target_group=int(target_group),
        level=int(level),
        scheme=scheme,
        group_n1=int(group_n1),
        compact_target_block=compact_target_block,
        target_index=int(target_index),
        source_index=int(source_index),
    ) -> dict[tuple[int, int], dict[int, Any]]:
        rebuilt = _build_conv_transform(
            spec=spec,
            plan=plan,
            weight=weight,
            weight_np=weight_np,
            stripe=stripe,
            source_group=int(source_group),
            target_group=int(target_group),
            level=int(level),
            scheme=scheme,
            group_n1=int(group_n1),
            compact_target_block=compact_target_block,
            force_payload=True,
        )
        if rebuilt is None:
            return {}
        block = _transform_payload_block(
            rebuilt,
            slots=int(slots),
            context=(
                f"{spec.family_label} native-source single-slot payload rebuild "
                f"source_index={int(source_index)} target_index={int(target_index)}"
            ),
        )
        return {(int(target_index), int(source_index)): block} if block else {}

    single_slot_cpp_metadata_enabled = (
        _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT_SINGLE_SLOT_METADATA")
        if bool(compact_output)
        else _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE_SINGLE_SLOT_METADATA")
    )
    cpp_transform = _CPP_DIAG_BUILDER_FALLBACK
    if not bool(single_slot_recipe) or bool(single_slot_cpp_metadata_enabled):
        cpp_transform = _cpp_provider_native_source_transform(
            spec=spec,
            plan=plan,
            weight=weight,
            weight_np=weight_np,
            stripe=stripe,
            source_group=int(source_group),
            target_group=int(target_group),
            level=int(level),
            scheme=scheme,
            group_n1=int(group_n1),
            compact_target_block=compact_target_block,
            path="native_source_compact_output" if bool(compact_output) else "native_source",
            env_gate=(
                "ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT"
                if bool(compact_output)
                else "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE"
            ),
        )
    if cpp_transform is not _CPP_DIAG_BUILDER_FALLBACK:
        if cpp_transform is None:
            return None
        if bool(single_slot_recipe):
            diag_indices = _single_slot_diag_indices_from_transform(cpp_transform)
            if not diag_indices:
                return None
            cache = diagonal_cache or _BlockDiagonalCache(build_all_native_source_blocks)

            def build_diagonals(
                *,
                cache=cache,
                target_index=int(target_index),
                source_index=int(source_index),
            ):
                block = cache.get_required(
                    int(target_index),
                    int(source_index),
                    context=(
                        f"{spec.family_label} native-source source_index={int(source_index)} "
                        f"target_index={int(target_index)}"
                    ),
                )
                return {(0, 0): block}

            baby, giant = _bsgs_rotation_sets(set(int(value) for value in diag_indices), slots=int(slots), n1=int(group_n1))
            metadata = dict(getattr(cpp_transform, "_diag_builder_metadata", {}) or {})
            metadata["diag_builder_single_slot_metadata_only"] = True
            transform = SimpleNamespace(
                name=(
                    f"native_halo_{spec.family_label}_s{int(stripe.index)}_src{int(source_group)}"
                    f"_tgt{int(target_group)}"
                    + ("" if not bool(compact_output) else f"_compact{int(compact_target_block)}")
                ),
                diagonals={},
                _single_slot_diag_indices_by_block={(0, 0): tuple(int(value) for value in diag_indices)},
                _single_slot_build_diagonals=build_diagonals,
                level=int(level),
                scheme=scheme,
                fhe_output_shape=torch.Size([1, int(slots)]),
                output_shape=torch.Size([1, int(slots)]),
                target_index=int(target_index),
                input_id=f"native_source_tile_{int(source_index)}",
                selected_n1=int(group_n1),
                baby_shifts=tuple(sorted(int(value) for value in baby)),
                giant_shifts=tuple(sorted(int(value) for value in giant)),
                rotation_group_id=f"native_halo:{spec.family_label}:s{int(stripe.index)}:src{int(source_group)}",
                rotation_cost_owner=bool(
                    int(target_group) == 0
                    and (not bool(compact_output) or int(compact_target_block) == 0)
                ),
                _single_slot_diagonal_cache=cache,
                _single_slot_release_diagonal_cache=cache.release,
            )
            setattr(transform, "_diag_builder_metadata", metadata)
            del cpp_transform
            return transform
        return cpp_transform
    source_tile = int(plan.source_tile_for_stripe(stripe))
    target_tile = int(plan.target_tile_for_stripe(stripe))
    source_start = int(source_group) * int(source_tile)
    source_end = min(int(spec.c_in), int(source_start) + int(source_tile))
    target_start = int(target_group) * int(target_tile)
    target_end = min(int(spec.c_out), int(target_start) + int(target_tile))
    source_count = int(source_end - source_start)
    target_count = int(target_end - target_start)
    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    source_channels = torch.arange(int(source_count), dtype=torch.int64)
    target_channels = (
        torch.arange(int(target_start), int(target_end), dtype=torch.int64)
        if bool(compact_output)
        else torch.arange(int(target_count), dtype=torch.int64)
    )
    out_h_values = torch.arange(
        int(stripe.target_h_start),
        int(stripe.target_h_end),
        dtype=torch.int64,
    )
    out_w_values = torch.arange(int(spec.w_out), dtype=torch.int64)
    max_pair_count = int(_native_halo_build_pair_chunk_limit())
    for kh in range(int(spec.kernel)):
        for kw in range(int(spec.kernel)):
            coeff = weight[
                int(target_start): int(target_end),
                int(source_start): int(source_end),
                int(kh),
                int(kw),
            ].to(dtype=torch.float32)
            if not bool(torch.any(coeff != 0).item()):
                continue
            coeff_by_source_target = coeff.t().contiguous()
            coeff_nonzero = torch.abs(coeff_by_source_target) > 0
            if not bool(coeff_nonzero.any().item()):
                continue

            op_out_h_values = (
                _materialized_output_source_h(
                    out_h_values,
                    h_out=int(spec.h_out),
                    output_top_beta=int(spec.output_top_beta),
                    output_bottom_beta=int(spec.output_bottom_beta),
                )
                if bool(compact_output)
                else out_h_values
            )
            in_h_values = (
                op_out_h_values * int(spec.stride)
                - int(spec.pad)
                + int(kh) * int(spec.dilation)
            )
            source_local_h_values = in_h_values - int(stripe.source_h_start)
            target_local_h_values = out_h_values - int(stripe.target_h_start)
            valid_h = (
                (in_h_values >= int(spec.input_h_min))
                & (in_h_values < int(spec.input_h_max))
                & (source_local_h_values >= 0)
                & (source_local_h_values < int(stripe.source_h))
                & (target_local_h_values >= 0)
                & (target_local_h_values < int(stripe.target_h))
            )
            if not bool(valid_h.any().item()):
                continue
            in_w_values = (
                out_w_values * int(spec.stride)
                - int(spec.pad)
                + int(kw) * int(spec.dilation)
            )
            valid_w = (in_w_values >= 0) & (in_w_values < int(spec.w_in))
            if not bool(valid_w.any().item()):
                continue

            valid_source_h = source_local_h_values[valid_h]
            valid_target_h = target_local_h_values[valid_h]
            valid_out_h = out_h_values[valid_h]
            valid_source_w = in_w_values[valid_w]
            valid_out_w = out_w_values[valid_w]
            source_h_grid, source_w_grid = torch.meshgrid(valid_source_h, valid_source_w, indexing="ij")
            target_h_grid, target_w_grid = torch.meshgrid(valid_target_h, valid_out_w, indexing="ij")
            out_h_grid, out_w_grid = torch.meshgrid(valid_out_h, valid_out_w, indexing="ij")
            source_h_flat = source_h_grid.reshape(-1)
            source_w_flat = source_w_grid.reshape(-1)
            target_h_flat = target_h_grid.reshape(-1)
            target_w_flat = target_w_grid.reshape(-1)
            out_h_flat = out_h_grid.reshape(-1)
            out_w_flat = out_w_grid.reshape(-1)
            if int(source_h_flat.numel()) == 0:
                continue

            channel_pair_count = max(1, int(source_count) * int(target_count))
            position_chunk = max(1, int(max_pair_count // channel_pair_count))
            for start in range(0, int(source_h_flat.numel()), int(position_chunk)):
                end = min(int(source_h_flat.numel()), int(start) + int(position_chunk))
                source_vec = _idx_chw_gap_channel_positions(
                    source_channels,
                    h=source_h_flat[int(start): int(end)],
                    w=source_w_flat[int(start): int(end)],
                    height=int(stripe.source_h),
                    width=int(spec.w_in),
                    gap=int(spec.gap_in),
                )
                if bool(compact_output):
                    compact_output_h = _spec_physical_output_h(spec)
                    target_h_slice = out_h_flat[int(start): int(end)]
                    target_h_valid = _spec_physical_output_h_valid(spec, target_h_slice)
                    target_index = _idx_chw_gap_channel_positions(
                        target_channels,
                        h=_spec_physical_output_h_positions(spec, target_h_slice),
                        w=out_w_flat[int(start): int(end)],
                        height=int(compact_output_h),
                        width=int(spec.w_out),
                        gap=int(spec.gap_out),
                    )
                    target_block_mask = (
                        torch.div(target_index, int(slots), rounding_mode="floor")
                        == int(compact_target_block)
                    ) & target_h_valid
                    if not bool(target_block_mask.any().item()):
                        continue
                    target_vec = torch.remainder(target_index, int(slots))
                else:
                    target_vec = _idx_chw_gap_channel_positions(
                        target_channels,
                        h=target_h_flat[int(start): int(end)],
                        w=target_w_flat[int(start): int(end)],
                        height=int(stripe.target_h),
                        width=int(spec.w_out),
                        gap=int(spec.gap_out),
                    )
                    target_block_mask = torch.ones_like(target_vec, dtype=torch.bool)

                pair_mask = coeff_nonzero[:, :, None] & target_block_mask[None, :, :]
                if not bool(pair_mask.any().item()):
                    continue
                diag_index = (source_vec[:, None, :] - target_vec[None, :, :]).remainder(int(slots))
                output_slot = target_vec[None, :, :].expand_as(diag_index)
                flat_keys = (diag_index * int(slots) + output_slot).to(dtype=torch.int64)
                flat_values = coeff_by_source_target[:, :, None].expand_as(diag_index).to(dtype=torch.float32)
                key_parts.append(flat_keys[pair_mask])
                value_parts.append(flat_values[pair_mask])
    if not key_parts:
        return None
    keys, values = _coalesce_native_rows(torch.cat(key_parts), torch.cat(value_parts))
    if int(keys.numel()) == 0:
        return None
    diag_indices = torch.div(keys, int(slots), rounding_mode="floor").to(dtype=torch.int64)
    output_slots = torch.remainder(keys, int(slots)).to(dtype=torch.int64)
    diag_set = set(int(value) for value in torch.unique_consecutive(diag_indices).tolist())
    if not diag_set:
        return None
    baby, giant = _bsgs_rotation_sets(set(int(value) for value in diag_set), slots=int(slots), n1=int(group_n1))
    if bool(single_slot_recipe):
        cache = diagonal_cache
        if cache is None:
            cache = _BlockDiagonalCache(build_all_native_source_blocks)

        def build_diagonals(
            *,
            cache=cache,
            target_index=int(target_index),
            source_index=int(source_index),
        ):
            block = cache.get_required(
                int(target_index),
                int(source_index),
                context=(
                    f"{spec.family_label} native-source source_index={int(source_index)} "
                    f"target_index={int(target_index)}"
                ),
            )
            return {(0, 0): block}

        return SimpleNamespace(
            name=(
                f"native_halo_{spec.family_label}_s{int(stripe.index)}_src{int(source_group)}"
                f"_tgt{int(target_group)}"
                + ("" if not bool(compact_output) else f"_compact{int(compact_target_block)}")
            ),
            diagonals={},
            _single_slot_diag_indices_by_block={(0, 0): tuple(sorted(int(value) for value in diag_set))},
            _single_slot_build_diagonals=build_diagonals,
            level=int(level),
            scheme=scheme,
            fhe_output_shape=torch.Size([1, int(slots)]),
            output_shape=torch.Size([1, int(slots)]),
            target_index=int(target_index),
            input_id=f"native_source_tile_{int(source_index)}",
            selected_n1=int(group_n1),
            baby_shifts=tuple(sorted(int(value) for value in baby)),
            giant_shifts=tuple(sorted(int(value) for value in giant)),
            rotation_group_id=f"native_halo:{spec.family_label}:s{int(stripe.index)}:src{int(source_group)}",
            rotation_cost_owner=bool(int(target_group) == 0 and (not bool(compact_output) or int(compact_target_block) == 0)),
            _single_slot_diagonal_cache=cache,
            _single_slot_release_diagonal_cache=cache.release,
        )

    diag_tensors: dict[int, torch.Tensor] = {}
    unique, counts = torch.unique_consecutive(diag_indices, return_counts=True)
    start = 0
    for diag_value, count_value in zip(unique.tolist(), counts.tolist()):
        end = int(start + int(count_value))
        diag = torch.zeros((int(slots),), dtype=torch.float32)
        diag.index_add_(0, output_slots[int(start): int(end)], values[int(start): int(end)].to(dtype=torch.float32))
        diag_tensors[int(diag_value)] = diag
        start = int(end)
    transform = SimpleNamespace(
        name=(
            f"native_halo_{spec.family_label}_s{int(stripe.index)}_src{int(source_group)}"
            f"_tgt{int(target_group)}"
            + ("" if not bool(compact_output) else f"_compact{int(compact_target_block)}")
        ),
        diagonals={(0, 0): {int(index): diag for index, diag in sorted(diag_tensors.items())}},
        level=int(level),
        scheme=scheme,
        fhe_output_shape=torch.Size([1, int(slots)]),
        output_shape=torch.Size([1, int(slots)]),
        target_index=int(target_index),
        input_id=f"native_source_tile_{int(source_index)}",
        selected_n1=int(group_n1),
        baby_shifts=tuple(sorted(int(value) for value in baby)),
        giant_shifts=tuple(sorted(int(value) for value in giant)),
        rotation_group_id=f"native_halo:{spec.family_label}:s{int(stripe.index)}:src{int(source_group)}",
        rotation_cost_owner=bool(int(target_group) == 0 and (not bool(compact_output) or int(compact_target_block) == 0)),
    )
    return _build_conv_transform_payload_shadow(
        transform,
        spec=spec,
        plan=plan,
        weight=weight,
        stripe=stripe,
        source_group=int(source_group),
        target_group=int(target_group),
        compact_target_block=compact_target_block,
    )


def _build_conv_transform_payload_shadow(
    transform: Any | None,
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    stripe: NativeHaloStripe,
    source_group: int,
    target_group: int,
    compact_target_block: int | None,
) -> Any | None:
    if transform is None:
        return None
    slots = int(transform.fhe_output_shape[-1])

    def build_cpp_payload(transform=transform, slots=int(slots)):
        if not _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE"):
            return None
        from orion.backend.diag_builder import bindings as diag_builder

        built = diag_builder.build_provider_native_source_conv2d_payload(
            spec=spec,
            plan=plan,
            weight=weight,
            stripe=stripe,
            source_group=int(source_group),
            target_group=int(target_group),
            compact_target_block=compact_target_block,
        )
        if built is None:
            return None
        diag_indices, diag_data, _metadata = built
        return diag_indices, diag_data

    return _shadow_provider_payload(
        transform=transform,
        slots=int(slots),
        build_cpp_payload=build_cpp_payload,
        path="native_source",
    )


def _build_conv_transforms_for_compact_output(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    stripe: NativeHaloStripe,
    source_group: int,
    target_group: int,
    level: int,
    scheme: Any,
    group_n1: int,
    force_payload: bool = False,
) -> list[tuple[int, Any]]:
    slots = int(spec.slot_count)
    source_tile = int(plan.source_tile_for_stripe(stripe))
    target_tile = int(plan.target_tile_for_stripe(stripe))
    source_start = int(source_group) * int(source_tile)
    source_end = min(int(spec.c_in), int(source_start) + int(source_tile))
    target_start = int(target_group) * int(target_tile)
    target_end = min(int(spec.c_out), int(target_start) + int(target_tile))
    source_count = int(source_end - source_start)
    target_count = int(target_end - target_start)
    if source_count <= 0 or target_count <= 0:
        return []
    single_slot_recipe = bool(_single_slot_layer_cache_enabled_for_scheme(scheme) and not bool(force_payload))
    source_index = int(plan.source_block_index(stripe, int(source_group)))

    def build_all_compact_output_blocks(
        *,
        spec=spec,
        plan=plan,
        weight=weight,
        stripe=stripe,
        source_group=int(source_group),
        target_group=int(target_group),
        level=int(level),
        scheme=scheme,
        group_n1=int(group_n1),
        source_index=int(source_index),
    ) -> dict[tuple[int, int], dict[int, Any]]:
        rebuilt = _build_conv_transforms_for_compact_output(
            spec=spec,
            plan=plan,
            weight=weight,
            stripe=stripe,
            source_group=int(source_group),
            target_group=int(target_group),
            level=int(level),
            scheme=scheme,
            group_n1=int(group_n1),
            force_payload=True,
        )
        blocks: dict[tuple[int, int], dict[int, Any]] = {}
        for rebuilt_target, rebuilt_transform in rebuilt:
            block = _transform_payload_block(
                rebuilt_transform,
                slots=int(slots),
                context=(
                    f"{spec.family_label} native-source compact-output single-slot payload rebuild "
                    f"source_index={int(source_index)} target_block={int(rebuilt_target)}"
                ),
            )
            if block:
                blocks[(int(rebuilt_target), int(source_index))] = block
        return blocks

    if (
        _provider_diag_builder_enabled()
        and _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT")
        and not _provider_diag_builder_shadow()
        and (
            not bool(single_slot_recipe)
            or _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT_SINGLE_SLOT_METADATA")
        )
    ):
        transforms: list[tuple[int, Any]] = []
        diagonal_cache = _BlockDiagonalCache(build_all_compact_output_blocks) if bool(single_slot_recipe) else None
        for target_block in range(int(_compact_ct_count(int(spec.c_out), _spec_physical_output_h(spec), int(spec.w_out), int(spec.gap_out), int(slots)))):
            built = _cpp_provider_native_source_transform(
                spec=spec,
                plan=plan,
                weight=weight,
                stripe=stripe,
                source_group=int(source_group),
                target_group=int(target_group),
                level=int(level),
                scheme=scheme,
                group_n1=int(group_n1),
                compact_target_block=int(target_block),
                path="native_source_compact_output",
                env_gate="ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT",
            )
            if built is _CPP_DIAG_BUILDER_FALLBACK:
                break
            if built is not None:
                if bool(single_slot_recipe):
                    diag_indices_by_block = dict(getattr(built, "_single_slot_diag_indices_by_block", {}) or {})
                    diag_indices = tuple(
                        sorted(
                            set(
                                int(value)
                                for value in diag_indices_by_block.get((0, 0), ())
                            )
                        )
                    )
                    if not diag_indices:
                        diag_indices = tuple(
                            sorted(
                                int(value)
                                for value in dict(
                                    (dict(getattr(built, "diagonals", {}) or {}).get((0, 0), {}) or {})
                                ).keys()
                            )
                        )
                    if not diag_indices:
                        continue
                    baby, giant = _bsgs_rotation_sets(set(int(value) for value in diag_indices), slots=int(slots), n1=int(group_n1))

                    def build_diagonals(
                        *,
                        target_block=int(target_block),
                        source_index=int(source_index),
                        diagonal_cache=diagonal_cache,
                    ):
                        if diagonal_cache is None:
                            return {}
                        block = diagonal_cache.get_required(
                            int(target_block),
                            int(source_index),
                            context=(
                                f"{spec.family_label} native-source compact-output "
                                f"source_index={int(source_index)} target_block={int(target_block)}"
                            ),
                        )
                        return {(0, 0): block}

                    transform = SimpleNamespace(
                        name=(
                            f"native_halo_{spec.family_label}_s{int(stripe.index)}_src{int(source_group)}"
                            f"_tgt{int(target_group)}_compact{int(target_block)}"
                        ),
                        diagonals={},
                        _single_slot_diag_indices_by_block={(0, 0): diag_indices},
                        _single_slot_build_diagonals=build_diagonals,
                        level=int(level),
                        scheme=scheme,
                        fhe_output_shape=torch.Size([1, int(slots)]),
                        output_shape=torch.Size([1, int(slots)]),
                        target_index=int(target_block),
                        input_id=f"native_source_tile_{int(source_index)}",
                        selected_n1=int(group_n1),
                        baby_shifts=tuple(sorted(int(value) for value in baby)),
                        giant_shifts=tuple(sorted(int(value) for value in giant)),
                        rotation_group_id=f"native_halo:{spec.family_label}:s{int(stripe.index)}:src{int(source_group)}",
                        rotation_cost_owner=bool(int(target_group) == 0 and int(target_block) == 0),
                        _single_slot_diagonal_cache=diagonal_cache,
                        _single_slot_release_diagonal_cache=(
                            diagonal_cache.release if diagonal_cache is not None else None
                        ),
                    )
                    metadata = dict(getattr(built, "_diag_builder_metadata", {}) or {})
                    metadata["diag_builder_single_slot_metadata_only"] = True
                    setattr(transform, "_diag_builder_metadata", metadata)
                    transforms.append((int(target_block), transform))
                    del built
                    continue
                transforms.append((int(target_block), built))
        else:
            return transforms

    target_channels = torch.arange(int(target_start), int(target_end), dtype=torch.int64)
    compact_output_h = _spec_physical_output_h(spec)
    key_parts_by_block: dict[int, list[torch.Tensor]] = {}
    value_parts_by_block: dict[int, list[torch.Tensor]] = {}
    source_channels = torch.arange(int(source_count), dtype=torch.int64)
    out_h_values = torch.arange(
        int(stripe.target_h_start),
        int(stripe.target_h_end),
        dtype=torch.int64,
    )
    out_w_values = torch.arange(int(spec.w_out), dtype=torch.int64)
    max_pair_count = int(_native_halo_build_pair_chunk_limit())

    for kh in range(int(spec.kernel)):
        for kw in range(int(spec.kernel)):
            coeff = weight[
                int(target_start): int(target_end),
                int(source_start): int(source_end),
                int(kh),
                int(kw),
            ].to(dtype=torch.float32)
            if not bool(torch.any(coeff != 0).item()):
                continue
            coeff_by_source_target = coeff.t().contiguous()
            coeff_nonzero = torch.abs(coeff_by_source_target) > 0
            if not bool(coeff_nonzero.any().item()):
                continue

            op_out_h_values = _materialized_output_source_h(
                out_h_values,
                h_out=int(spec.h_out),
                output_top_beta=int(spec.output_top_beta),
                output_bottom_beta=int(spec.output_bottom_beta),
            )
            in_h_values = (
                op_out_h_values * int(spec.stride)
                - int(spec.pad)
                + int(kh) * int(spec.dilation)
            )
            source_local_h_values = in_h_values - int(stripe.source_h_start)
            valid_h = (
                (in_h_values >= int(spec.input_h_min))
                & (in_h_values < int(spec.input_h_max))
                & (source_local_h_values >= 0)
                & (source_local_h_values < int(stripe.source_h))
            )
            if not bool(valid_h.any().item()):
                continue
            in_w_values = (
                out_w_values * int(spec.stride)
                - int(spec.pad)
                + int(kw) * int(spec.dilation)
            )
            valid_w = (in_w_values >= 0) & (in_w_values < int(spec.w_in))
            if not bool(valid_w.any().item()):
                continue

            valid_source_h = source_local_h_values[valid_h]
            valid_out_h = out_h_values[valid_h]
            valid_source_w = in_w_values[valid_w]
            valid_out_w = out_w_values[valid_w]
            source_h_grid, source_w_grid = torch.meshgrid(valid_source_h, valid_source_w, indexing="ij")
            out_h_grid, out_w_grid = torch.meshgrid(valid_out_h, valid_out_w, indexing="ij")
            source_h_flat = source_h_grid.reshape(-1)
            source_w_flat = source_w_grid.reshape(-1)
            out_h_flat = out_h_grid.reshape(-1)
            out_w_flat = out_w_grid.reshape(-1)
            if int(source_h_flat.numel()) == 0:
                continue

            channel_pair_count = max(1, int(source_count) * int(target_count))
            position_chunk = max(1, int(max_pair_count // channel_pair_count))
            for start in range(0, int(source_h_flat.numel()), int(position_chunk)):
                end = min(int(source_h_flat.numel()), int(start) + int(position_chunk))
                source_vec = _idx_chw_gap_channel_positions(
                    source_channels,
                    h=source_h_flat[int(start): int(end)],
                    w=source_w_flat[int(start): int(end)],
                    height=int(stripe.source_h),
                    width=int(spec.w_in),
                    gap=int(spec.gap_in),
                )
                target_h_slice = out_h_flat[int(start): int(end)]
                target_h_valid = _spec_physical_output_h_valid(spec, target_h_slice)
                compact_slots = _idx_chw_gap_channel_positions(
                    target_channels,
                    h=_spec_physical_output_h_positions(spec, target_h_slice),
                    w=out_w_flat[int(start): int(end)],
                    height=int(compact_output_h),
                    width=int(spec.w_out),
                    gap=int(spec.gap_out),
                )
                target_blocks = torch.div(compact_slots, int(slots), rounding_mode="floor")
                target_slots = torch.remainder(compact_slots, int(slots))
                diag_index = (source_vec[:, None, :] - target_slots[None, :, :]).remainder(int(slots))
                output_slot = target_slots[None, :, :].expand_as(diag_index)
                flat_keys = (diag_index * int(slots) + output_slot).to(dtype=torch.int64)
                flat_values = coeff_by_source_target[:, :, None].expand_as(diag_index).to(dtype=torch.float32)
                for target_block in torch.unique(target_blocks).tolist():
                    block = int(target_block)
                    block_mask = target_blocks == int(block)
                    pair_mask = coeff_nonzero[:, :, None] & block_mask[None, :, :] & target_h_valid[None, None, :]
                    if not bool(pair_mask.any().item()):
                        continue
                    key_parts_by_block.setdefault(int(block), []).append(flat_keys[pair_mask])
                    value_parts_by_block.setdefault(int(block), []).append(flat_values[pair_mask])

    transforms: list[tuple[int, Any]] = []
    diagonal_cache: _BlockDiagonalCache | None = None
    if bool(single_slot_recipe):
        diagonal_cache = _BlockDiagonalCache(build_all_compact_output_blocks)

    for target_block in sorted(key_parts_by_block):
        key_parts = key_parts_by_block[int(target_block)]
        value_parts = value_parts_by_block.get(int(target_block), [])
        if not key_parts or not value_parts:
            continue
        keys, values = _coalesce_native_rows(torch.cat(key_parts), torch.cat(value_parts))
        if int(keys.numel()) == 0:
            continue
        diag_indices = torch.div(keys, int(slots), rounding_mode="floor").to(dtype=torch.int64)
        output_slots = torch.remainder(keys, int(slots)).to(dtype=torch.int64)
        diag_set = set(int(value) for value in torch.unique_consecutive(diag_indices).tolist())
        if not diag_set:
            continue
        baby, giant = _bsgs_rotation_sets(set(int(value) for value in diag_set), slots=int(slots), n1=int(group_n1))
        if bool(single_slot_recipe):
            def build_diagonals(
                *,
                target_block=int(target_block),
                source_index=int(source_index),
                diagonal_cache=diagonal_cache,
            ):
                if diagonal_cache is None:
                    return {}
                block = diagonal_cache.get_required(
                    int(target_block),
                    int(source_index),
                    context=(
                        f"{spec.family_label} native-source compact-output "
                        f"source_index={int(source_index)} target_block={int(target_block)}"
                    ),
                )
                return {(0, 0): block}

            transform = SimpleNamespace(
                name=(
                    f"native_halo_{spec.family_label}_s{int(stripe.index)}_src{int(source_group)}"
                    f"_tgt{int(target_group)}_compact{int(target_block)}"
                ),
                diagonals={},
                _single_slot_diag_indices_by_block={(0, 0): tuple(sorted(int(value) for value in diag_set))},
                _single_slot_build_diagonals=build_diagonals,
                level=int(level),
                scheme=scheme,
                fhe_output_shape=torch.Size([1, int(slots)]),
                output_shape=torch.Size([1, int(slots)]),
                target_index=int(target_block),
                input_id=f"native_source_tile_{int(source_index)}",
                selected_n1=int(group_n1),
                baby_shifts=tuple(sorted(int(value) for value in baby)),
                giant_shifts=tuple(sorted(int(value) for value in giant)),
                rotation_group_id=f"native_halo:{spec.family_label}:s{int(stripe.index)}:src{int(source_group)}",
                rotation_cost_owner=bool(int(target_group) == 0 and int(target_block) == 0),
                _single_slot_diagonal_cache=diagonal_cache,
                _single_slot_release_diagonal_cache=(
                    diagonal_cache.release if diagonal_cache is not None else None
                ),
            )
            transforms.append((int(target_block), transform))
            continue

        diag_tensors: dict[int, torch.Tensor] = {}
        unique, counts = torch.unique_consecutive(diag_indices, return_counts=True)
        start = 0
        for diag_value, count_value in zip(unique.tolist(), counts.tolist()):
            end = int(start + int(count_value))
            diag = torch.zeros((int(slots),), dtype=torch.float32)
            diag.index_add_(0, output_slots[int(start): int(end)], values[int(start): int(end)].to(dtype=torch.float32))
            diag_tensors[int(diag_value)] = diag
            start = int(end)
        transform = SimpleNamespace(
            name=(
                f"native_halo_{spec.family_label}_s{int(stripe.index)}_src{int(source_group)}"
                f"_tgt{int(target_group)}_compact{int(target_block)}"
            ),
            diagonals={(0, 0): {int(index): diag for index, diag in sorted(diag_tensors.items())}},
            level=int(level),
            scheme=scheme,
            fhe_output_shape=torch.Size([1, int(slots)]),
            output_shape=torch.Size([1, int(slots)]),
            target_index=int(target_block),
            input_id=f"native_source_tile_{int(source_index)}",
            selected_n1=int(group_n1),
            baby_shifts=tuple(sorted(int(value) for value in baby)),
            giant_shifts=tuple(sorted(int(value) for value in giant)),
            rotation_group_id=f"native_halo:{spec.family_label}:s{int(stripe.index)}:src{int(source_group)}",
            rotation_cost_owner=bool(int(target_group) == 0 and int(target_block) == 0),
        )
        transforms.append(
            (
                int(target_block),
                _build_conv_compact_output_payload_shadow(
                    transform,
                    spec=spec,
                    plan=plan,
                    weight=weight,
                    stripe=stripe,
                    source_group=int(source_group),
                    target_group=int(target_group),
                ),
            )
        )
    return transforms


def _build_conv_compact_output_payload_shadow(
    transform: Any | None,
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    stripe: NativeHaloStripe,
    source_group: int,
    target_group: int,
) -> Any | None:
    if transform is None:
        return None
    slots = int(transform.fhe_output_shape[-1])

    def build_cpp_payload(transform=transform, slots=int(slots)):
        if not _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT"):
            return None
        from orion.backend.diag_builder import bindings as diag_builder

        built = diag_builder.build_provider_native_source_conv2d_payload(
            spec=spec,
            plan=plan,
            weight=weight,
            stripe=stripe,
            source_group=int(source_group),
            target_group=int(target_group),
            compact_target_block=int(transform.target_index),
        )
        if built is None:
            return None
        diag_indices, diag_data, _metadata = built
        return diag_indices, diag_data

    return _shadow_provider_payload(
        transform=transform,
        slots=int(slots),
        build_cpp_payload=build_cpp_payload,
        path="native_source_compact_output",
    )


def _compact_source_index(
    spec: NativeHaloConv2DSpec,
    source_layout: dict[str, Any],
    *,
    channel: int,
    h: int,
    w: int,
) -> int | None:
    source_top_beta = max(
        0,
        int(_layout_physical_top_beta(source_layout, default=spec.input_physical_top_beta or 0) or 0),
    )
    source_bottom_beta = max(
        0,
        int(_layout_physical_bottom_beta(source_layout, default=spec.input_physical_bottom_beta or 0) or 0),
    )
    source_gap = max(1, int(source_layout.get("gap", spec.gap_in) or 1))
    if int(h) < 0:
        if int(h) < -int(source_top_beta):
            return None
        source_h = int(h) + int(source_top_beta)
    elif int(h) >= int(spec.h_in):
        if int(h) >= int(spec.h_in) + int(source_bottom_beta):
            return None
        source_h = int(source_top_beta) + int(h)
    else:
        source_h = int(source_top_beta) + int(h)
    source_height = int(spec.h_in) + int(source_top_beta) + int(source_bottom_beta)
    if int(source_h) < 0 or int(source_h) >= int(source_height):
        return None
    return _idx_chw_gap(
        int(channel),
        int(source_h),
        int(w),
        int(source_height),
        int(spec.w_in),
        int(source_gap),
    )


def _build_compact_source_conv_transform(
    *,
    spec: NativeHaloConv2DSpec,
    plan: NativeHaloConv2DPlan,
    weight: torch.Tensor,
    stripe: NativeHaloStripe,
    source_block: int,
    target_group: int,
    level: int,
    scheme: Any,
    source_layout: dict[str, Any],
    group_n1: int,
    compact_target_block: int | None = None,
    diagonal_cache: _BlockDiagonalCache | None = None,
    force_payload: bool = False,
) -> Any | None:
    slots = int(spec.slot_count)
    compact_output = compact_target_block is not None
    single_slot_recipe = bool(_single_slot_layer_cache_enabled_for_scheme(scheme) and not bool(force_payload))
    if not bool(single_slot_recipe):
        cpp_transform = _cpp_provider_compact_source_transform(
            spec=spec,
            plan=plan,
            weight=weight,
            stripe=stripe,
            source_block=int(source_block),
            target_group=int(target_group),
            level=int(level),
            scheme=scheme,
            source_layout=dict(source_layout),
            group_n1=int(group_n1),
            compact_target_block=compact_target_block,
        )
        if cpp_transform is not _CPP_DIAG_BUILDER_FALLBACK:
            return cpp_transform
    if bool(single_slot_recipe):
        diag_sets_by_target = _collect_compact_source_conv_diag_sets(
            spec=spec,
            plan=plan,
            weight=weight,
            stripe=stripe,
            source_block=int(source_block),
            target_group=int(target_group),
            source_layout=dict(source_layout),
            compact_output=bool(compact_output),
            compact_target_block=compact_target_block,
        )
        target_index = (
            int(compact_target_block)
            if bool(compact_output)
            else int(plan.target_block_index(stripe, int(target_group)))
        )
        diag_set = diag_sets_by_target.get(int(target_index), set())
        if not diag_set:
            return None
        cache = diagonal_cache
        if cache is None:
            def build_all_blocks(
                *,
                spec=spec,
                plan=plan,
                weight=weight,
                stripe=stripe,
                source_block=int(source_block),
                target_group=int(target_group),
                level=int(level),
                scheme=scheme,
                source_layout=dict(source_layout),
                group_n1=int(group_n1),
                compact_target_block=compact_target_block,
                target_index=int(target_index),
            ) -> dict[tuple[int, int], dict[int, Any]]:
                rebuilt = _build_compact_source_conv_transform(
                    spec=spec,
                    plan=plan,
                    weight=weight,
                    stripe=stripe,
                    source_block=int(source_block),
                    target_group=int(target_group),
                    level=int(level),
                    scheme=scheme,
                    source_layout=dict(source_layout),
                    group_n1=int(group_n1),
                    compact_target_block=compact_target_block,
                    force_payload=True,
                )
                if rebuilt is None:
                    raise RuntimeError(
                        f"{spec.family_label} compact-source single-slot payload rebuild returned no transform "
                        f"for source_block={int(source_block)} target_index={int(target_index)}"
                    )
                block = _transform_payload_block(
                    rebuilt,
                    slots=int(spec.slot_count),
                    context=(
                        f"{spec.family_label} compact-source single-slot payload rebuild "
                        f"source_block={int(source_block)} target_index={int(target_index)}"
                    ),
                )
                if not block:
                    raise RuntimeError(
                        f"{spec.family_label} compact-source single-slot payload rebuild returned empty diagonals "
                        f"for source_block={int(source_block)} target_index={int(target_index)}"
                    )
                return {(int(target_index), int(source_block)): block}

            cache = _BlockDiagonalCache(build_all_blocks)
        return _make_compact_source_conv_single_slot_transform(
            spec=spec,
            plan=plan,
            weight=weight,
            stripe=stripe,
            source_block=int(source_block),
            target_group=int(target_group),
            level=int(level),
            scheme=scheme,
            source_layout=dict(source_layout),
            group_n1=int(group_n1),
            target_index=int(target_index),
            diag_set=set(int(value) for value in diag_set),
            compact_target_block=compact_target_block,
            diagonal_cache=cache,
        )
    target_tile = int(plan.target_tile_for_stripe(stripe))
    target_start = int(target_group) * int(target_tile)
    target_end = min(int(spec.c_out), int(target_start) + int(target_tile))
    target_count = int(target_end - target_start)
    target_slots = (
        None
        if bool(compact_output)
        else _slot_indices(int(target_count), int(stripe.target_h), int(spec.w_out), int(spec.gap_out))
    )
    source_channels = torch.arange(int(spec.c_in), dtype=torch.int64)
    target_channels = torch.arange(int(target_start), int(target_end), dtype=torch.int64)
    source_top_beta = max(
        0,
        int(_layout_physical_top_beta(source_layout, default=spec.input_physical_top_beta or 0) or 0),
    )
    source_bottom_beta = max(
        0,
        int(_layout_physical_bottom_beta(source_layout, default=spec.input_physical_bottom_beta or 0) or 0),
    )
    source_gap = max(1, int(source_layout.get("gap", spec.gap_in) or 1))
    source_height = int(spec.h_in) + int(source_top_beta) + int(source_bottom_beta)
    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    out_h_values = torch.arange(
        int(stripe.target_h_start),
        int(stripe.target_h_end),
        dtype=torch.int64,
    )
    out_w_values = torch.arange(int(spec.w_out), dtype=torch.int64)
    max_pair_count = int(_native_halo_build_pair_chunk_limit())
    for kh in range(int(spec.kernel)):
        for kw in range(int(spec.kernel)):
            coeff = weight[int(target_start): int(target_end), :, int(kh), int(kw)].to(dtype=torch.float32)
            if not bool(torch.any(coeff != 0).item()):
                continue
            op_out_h_values = (
                _materialized_output_source_h(
                    out_h_values,
                    h_out=int(spec.h_out),
                    output_top_beta=int(spec.output_top_beta),
                    output_bottom_beta=int(spec.output_bottom_beta),
                )
                if bool(compact_output)
                else out_h_values
            )
            in_h_values = (
                op_out_h_values * int(spec.stride)
                - int(spec.pad)
                + int(kh) * int(spec.dilation)
            )
            valid_h = (
                (in_h_values >= 0)
                & (in_h_values < int(spec.h_in))
            )
            if not bool(valid_h.any().item()):
                continue
            in_w_values = (
                out_w_values * int(spec.stride)
                - int(spec.pad)
                + int(kw) * int(spec.dilation)
            )
            valid_w = (in_w_values >= 0) & (in_w_values < int(spec.w_in))
            if not bool(valid_w.any().item()):
                continue

            valid_out_h = out_h_values[valid_h]
            valid_source_h = int(source_top_beta) + in_h_values[valid_h]
            valid_out_w = out_w_values[valid_w]
            valid_source_w = in_w_values[valid_w]
            grid_h, grid_w = torch.meshgrid(valid_source_h, valid_source_w, indexing="ij")
            out_grid_h, out_grid_w = torch.meshgrid(valid_out_h, valid_out_w, indexing="ij")
            source_h_flat = grid_h.reshape(-1)
            source_w_flat = grid_w.reshape(-1)
            out_h_flat = out_grid_h.reshape(-1)
            out_w_flat = out_grid_w.reshape(-1)
            if int(source_h_flat.numel()) == 0:
                continue

            coeff_by_source_target = coeff.t().contiguous()
            coeff_nonzero = torch.abs(coeff_by_source_target) > 0
            if not bool(coeff_nonzero.any().item()):
                continue
            channel_pair_count = max(1, int(source_channels.numel()) * int(target_count))
            position_chunk = max(1, int(max_pair_count // channel_pair_count))

            for start in range(0, int(source_h_flat.numel()), int(position_chunk)):
                end = min(int(source_h_flat.numel()), int(start) + int(position_chunk))
                source_index = _idx_chw_gap_channel_positions(
                    source_channels,
                    h=source_h_flat[int(start): int(end)],
                    w=source_w_flat[int(start): int(end)],
                    height=int(source_height),
                    width=int(spec.w_in),
                    gap=int(source_gap),
                )
                source_block_mask = (
                    torch.div(source_index, int(slots), rounding_mode="floor")
                    == int(source_block)
                )
                if not bool(source_block_mask.any().item()):
                    continue
                source_vec = torch.remainder(source_index, int(slots))

                if bool(compact_output):
                    compact_output_h = _spec_physical_output_h(spec)
                    target_h_slice = out_h_flat[int(start): int(end)]
                    target_h_valid = _spec_physical_output_h_valid(spec, target_h_slice)
                    target_index = _idx_chw_gap_channel_positions(
                        target_channels,
                        h=_spec_physical_output_h_positions(spec, target_h_slice),
                        w=out_w_flat[int(start): int(end)],
                        height=int(compact_output_h),
                        width=int(spec.w_out),
                        gap=int(spec.gap_out),
                    )
                    target_block_mask = (
                        torch.div(target_index, int(slots), rounding_mode="floor")
                        == int(compact_target_block)
                    ) & target_h_valid
                    target_vec = torch.remainder(target_index, int(slots))
                else:
                    assert target_slots is not None
                    target_local_h = out_h_flat[int(start): int(end)] - int(stripe.target_h_start)
                    target_vec = _idx_chw_gap_channel_positions(
                        torch.arange(int(target_count), dtype=torch.int64),
                        h=target_local_h,
                        w=out_w_flat[int(start): int(end)],
                        height=int(stripe.target_h),
                        width=int(spec.w_out),
                        gap=int(spec.gap_out),
                    )
                    target_block_mask = torch.ones_like(target_vec, dtype=torch.bool)

                pair_mask = (
                    source_block_mask[:, None, :]
                    & target_block_mask[None, :, :]
                    & coeff_nonzero[:, :, None]
                )
                if not bool(pair_mask.any().item()):
                    continue
                diag_index = (source_vec[:, None, :] - target_vec[None, :, :]).remainder(int(slots))
                output_slot = target_vec[None, :, :].expand_as(diag_index)
                flat_keys = (diag_index * int(slots) + output_slot).to(dtype=torch.int64)
                flat_values = coeff_by_source_target[:, :, None].expand_as(diag_index).to(dtype=torch.float32)
                key_parts.append(flat_keys[pair_mask])
                value_parts.append(flat_values[pair_mask])
    if not key_parts:
        return None
    keys, values = _coalesce_native_rows(torch.cat(key_parts), torch.cat(value_parts))
    if int(keys.numel()) == 0:
        return None
    diag_indices = torch.div(keys, int(slots), rounding_mode="floor").to(dtype=torch.int64)
    output_slots = torch.remainder(keys, int(slots)).to(dtype=torch.int64)
    target_index = (
        int(compact_target_block)
        if bool(compact_output)
        else int(plan.target_block_index(stripe, int(target_group)))
    )
    diag_set = set(int(value) for value in torch.unique_consecutive(diag_indices).tolist())
    if not diag_set:
        return None
    baby, giant = _bsgs_rotation_sets(set(int(value) for value in diag_set), slots=int(slots), n1=int(group_n1))
    diag_tensors: dict[int, torch.Tensor] = {}
    unique, counts = torch.unique_consecutive(diag_indices, return_counts=True)
    start = 0
    for diag_value, count_value in zip(unique.tolist(), counts.tolist()):
        end = int(start + int(count_value))
        diag = torch.zeros((int(slots),), dtype=torch.float32)
        diag.index_add_(0, output_slots[int(start): int(end)], values[int(start): int(end)].to(dtype=torch.float32))
        diag_tensors[int(diag_value)] = diag
        start = int(end)
    transform = SimpleNamespace(
        name=(
            f"native_halo_{spec.family_label}_compactsrc{int(source_block)}"
            f"_s{int(stripe.index)}_tgt{int(target_group)}"
            + ("" if not bool(compact_output) else f"_compact{int(compact_target_block)}")
        ),
        diagonals={(0, 0): {int(index): diag for index, diag in sorted(diag_tensors.items())}},
        level=int(level),
        scheme=scheme,
        fhe_output_shape=torch.Size([1, int(slots)]),
        output_shape=torch.Size([1, int(slots)]),
        target_index=int(target_index),
        input_id=f"compact_source_block_{int(source_block)}",
        selected_n1=int(group_n1),
        baby_shifts=tuple(sorted(int(value) for value in baby)),
        giant_shifts=tuple(sorted(int(value) for value in giant)),
        rotation_group_id=f"native_halo:{spec.family_label}:compact_src{int(source_block)}",
        rotation_cost_owner=bool(int(target_group) == 0 and (not bool(compact_output) or int(compact_target_block) == 0)),
    )
    return _build_compact_source_conv_payload_shadow(
        transform,
        spec=spec,
        plan=plan,
        weight=weight,
        stripe=stripe,
        source_block=int(source_block),
        target_group=int(target_group),
        source_layout=dict(source_layout),
        compact_target_block=compact_target_block,
    )


def _retune_transform_group_bsgs(transforms: list[Any], *, slots: int) -> None:
    def diag_set_for(transform: Any) -> set[int]:
        indices_by_block = getattr(transform, "_single_slot_diag_indices_by_block", None)
        if indices_by_block is not None:
            values: set[int] = set()
            for diag_indices in dict(indices_by_block).values():
                values.update(int(value) for value in diag_indices)
            return values
        return set(int(value) for value in dict(transform.diagonals.get((0, 0), {})).keys())

    diag_sets = tuple(
        diag_set_for(transform)
        for transform in transforms
    )
    if not diag_sets:
        return
    n1, _rotations, _baby_count, _giant_count = _native_best_common_bsgs(diag_sets, slots=int(slots))
    for transform in transforms:
        diag_set = diag_set_for(transform)
        baby, giant = _bsgs_rotation_sets(diag_set, slots=int(slots), n1=int(n1))
        transform.selected_n1 = int(n1)
        transform.baby_shifts = tuple(sorted(int(value) for value in baby))
        transform.giant_shifts = tuple(sorted(int(value) for value in giant))


def _cached_transform_shell(*, level: int, scheme: Any) -> Any:
    return SimpleNamespace(diagonals={}, level=int(level), scheme=scheme)


@dataclass
class _RuntimeUnifiedTransformGroup:
    input_index: int
    group: Any
    target_indices: tuple[int, ...]


class NativeHaloStripeNoRIConvExecutor:
    kernel_kind = "native_halo_stripe_no_ri_conv2d"
    use_ct_pt_hybrid_packing = False
    native_halo_input_capable = True
    native_halo_output_capable = True

    def __init__(self, *, module: Any, spec: NativeHaloConv2DSpec, output_node_id: str) -> None:
        self.module = module
        self.spec = spec
        self.output_node_id = str(output_node_id)
        self._native_plan_require_target_fit = not self._uses_tight_compact_output_for_spec(spec)
        fold_mode = self._channel_fold_mode()
        self.native_plan = native_halo_conv2d_plan(
            spec,
            require_native_target_fit=bool(self._native_plan_require_target_fit),
            channel_fold_mode=str(fold_mode),
        )
        self.slots = int(spec.slot_count)
        self.rows = int(self._runtime_output_ct_count())
        self.cols = int(self.native_plan.input_ct_count)
        self.output_shape = getattr(module, "output_shape", None)
        self.fhe_output_shape = getattr(module, "fhe_output_shape", None)
        self.runtime_groups: list[_RuntimeUnifiedTransformGroup] = []
        self.groups_by_input_index: dict[int, Any] = {}
        self.target_indices_by_input_index: dict[int, tuple[int, ...]] = {}
        self._compiled_lt_grouping_mode = "shared"
        self._deferred_single_slot_cache_releases: dict[int, Callable[[], None]] = {}
        self.input_relayout_kernel: NativeHaloRelayoutKernel | None = None
        self.output_relayout_kernel: NativeHaloRelayoutKernel | None = None
        self.bias_vector: torch.Tensor | None = None
        self.bias_plaintexts: tuple[Any | None, ...] = ()
        self._bias_plaintext_cache: dict[tuple[int, int], Any] = {}
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.compile_count = 0
        self.last_runtime_timing: dict[str, float] = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "build_transform_s": 0.0,
            "retune_bsgs_s": 0.0,
            "group_compile_s": 0.0,
            "diag_builder_build_s": 0.0,
            "diag_builder_wall_s": 0.0,
            "diag_builder_shadow_s": 0.0,
            "diag_builder_payload_count": 0.0,
            "diag_builder_fallback_count": 0.0,
            "built_transform_count": 0.0,
            "compiled_group_count": 0.0,
            "evaluate_unified_s": 0.0,
            "group_eval_s": 0.0,
            "partial_wrap_s": 0.0,
            "partial_rescale_s": 0.0,
            "partial_accumulate_s": 0.0,
            "postprocess_s": 0.0,
            "input_relayout_s": 0.0,
            "output_relayout_s": 0.0,
        }
        self.last_runtime_counts: dict[str, int] = {}
        self._compile_cache_metadata: dict[str, Any] = {}

    def supports_scheme(self, scheme: Any | None) -> bool:
        return scheme is not None

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def _uses_tight_compact_output_for_spec(self, spec: NativeHaloConv2DSpec) -> bool:
        materialization = str(getattr(self.module, "layout_policy_output_materialization", "") or "")
        if materialization in {"native_halo_stripe", "native_stripe", "channel_aligned_native_stripe"}:
            return False
        return bool(str(materialization) == "fused_relayout" or not _spec_has_physical_output_halo(spec))

    def _uses_tight_compact_output(self) -> bool:
        return self._uses_tight_compact_output_for_spec(self.native_plan.spec)

    def _compact_output_storage_layout(self) -> str:
        if _spec_has_semantic_output_halo(self.native_plan.spec):
            return "logical_halo_compact"
        return "tight_compact"

    def _input_physical_layout(self) -> str:
        value = str(getattr(self.module, "layout_policy_input_physical_layout", "") or "")
        return value if value else "native_source_stripe"

    def _uses_compact_source_input(self) -> bool:
        return self._input_physical_layout() in {"packed_compact", "logical_halo_compact"}

    def _compact_source_layout(self) -> dict[str, int]:
        layout = dict(getattr(self.module, "layout_policy_input_layout", {}) or {})
        return {
            "top_beta": _layout_top_beta(layout),
            "bottom_beta": _layout_bottom_beta(layout),
            "physical_top_beta": _layout_physical_top_beta(layout),
            "physical_bottom_beta": _layout_physical_bottom_beta(layout),
            "gap": max(1, int(layout.get("gap", self.native_plan.spec.gap_in) or 1)),
        }

    def _compact_source_ct_count(self) -> int:
        spec = self.native_plan.spec
        layout = self._compact_source_layout()
        height = int(spec.h_in) + int(layout["physical_top_beta"]) + int(layout["physical_bottom_beta"])
        return _compact_ct_count(
            int(spec.c_in),
            int(height),
            int(spec.w_in),
            int(layout["gap"]),
            int(spec.slot_count),
        )

    def _compact_output_ct_count(self) -> int:
        spec = self.native_plan.spec
        height = _spec_physical_output_h(spec)
        return _compact_ct_count(
            int(spec.c_out),
            int(height),
            int(spec.w_out),
            int(spec.gap_out),
            int(spec.slot_count),
        )

    def _native_stripe_output_ct_count(self) -> int:
        return int(sum(int(value) for value in self.native_plan.target_channel_group_counts))

    def _runtime_output_ct_count(self) -> int:
        return int(
            self._compact_output_ct_count()
            if self._uses_tight_compact_output()
            else self._native_stripe_output_ct_count()
        )

    def runtime_native_fhe_output_shape(self) -> torch.Size:
        if self._uses_tight_compact_output():
            return _compact_output_fhe_shape_for_spec(self.native_plan.spec)
        return torch.Size([int(self.rows), int(self.slots)])

    def _runtime_spec(self) -> NativeHaloConv2DSpec:
        input_layout = dict(getattr(self.module, "layout_policy_input_layout", {}) or {})
        output_layout = dict(getattr(self.module, "layout_policy_output_layout", {}) or {})
        return replace(
            self.spec,
            input_top_beta=_layout_top_beta(input_layout),
            input_bottom_beta=_layout_bottom_beta(input_layout),
            output_top_beta=_layout_top_beta(output_layout),
            output_bottom_beta=_layout_bottom_beta(output_layout),
            input_physical_top_beta=_layout_physical_top_beta(input_layout),
            input_physical_bottom_beta=_layout_physical_bottom_beta(input_layout),
            output_physical_top_beta=_layout_physical_top_beta(output_layout),
            output_physical_bottom_beta=_layout_physical_bottom_beta(output_layout),
        )

    def _refresh_runtime_plan(self) -> bool:
        runtime_spec = self._runtime_spec()
        require_native_target_fit = not self._uses_tight_compact_output_for_spec(runtime_spec)
        fold_mode = self._channel_fold_mode()
        changed = (
            tuple(runtime_spec.to_dict().items()) != tuple(self.native_plan.spec.to_dict().items())
            or bool(require_native_target_fit) != bool(self._native_plan_require_target_fit)
            or str(fold_mode) != str(self.native_plan.channel_fold_mode)
        )
        if bool(changed):
            self.native_plan = native_halo_conv2d_plan(
                runtime_spec,
                require_native_target_fit=bool(require_native_target_fit),
                channel_fold_mode=str(fold_mode),
            )
            self._native_plan_require_target_fit = bool(require_native_target_fit)
        self.slots = int(self.native_plan.spec.slot_count)
        self.rows = int(self._runtime_output_ct_count())
        self.cols = int(self._compact_source_ct_count() if self._uses_compact_source_input() else self.native_plan.input_ct_count)
        self.output_shape = getattr(self.module, "output_shape", self.output_shape)
        self.fhe_output_shape = getattr(self.module, "fhe_output_shape", self.fhe_output_shape)
        return bool(changed)

    def _validate_module(self) -> None:
        weight = getattr(self.module, "on_weight", None)
        if weight is None:
            raise RuntimeError(f"{self.output_node_id} has no fused Orion weight for native halo Conv2d")
        if tuple(int(v) for v in tuple(weight.shape)) != tuple(int(v) for v in self.spec.weight_shape):
            raise RuntimeError(f"{self.output_node_id} weight shape does not match native halo spec")

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        self._compile_cache_metadata = dict(metadata or {})

    @staticmethod
    def _normalise_lt_grouping_mode(value: Any) -> Literal["shared", "individual"]:
        text = str(value or "").strip().lower().replace("-", "_")
        if text in {"", "shared", "grouped", "provider_shared"}:
            return "shared"
        if text in {
            "individual",
            "individual_lt",
            "per_lt",
            "per_linear_transform",
            "disable_shared_rotation",
            "no_shared_rotation",
            "no_share",
        }:
            return "individual"
        raise ValueError(f"unsupported provider LT grouping mode: {value!r}")

    def _lt_grouping_mode(self) -> Literal["shared", "individual"]:
        for attr in ("layout_policy_provider_lt_grouping_mode", "layout_policy_lt_grouping_mode"):
            raw = getattr(self.module, attr, None)
            if raw is not None and str(raw).strip() != "":
                return self._normalise_lt_grouping_mode(raw)
        if bool(getattr(self.module, "layout_policy_provider_disable_shared_rotation", False)):
            return "individual"
        return "shared"

    def _channel_fold_mode(self) -> str:
        return _normalise_channel_fold_mode(
            getattr(self.module, "layout_policy_native_halo_channel_fold_mode", "")
        )

    def _reset_runtime_groups(self) -> None:
        self.runtime_groups = []
        self.groups_by_input_index = {}
        self.target_indices_by_input_index = {}
        self._deferred_single_slot_cache_releases = {}

    def _defer_individual_single_slot_cache_release(self, transform: Any) -> None:
        release_cache = getattr(transform, "_single_slot_release_diagonal_cache", None)
        if not callable(release_cache):
            return
        cache = getattr(transform, "_single_slot_diagonal_cache", None)
        key_obj = cache if cache is not None else release_cache
        self._deferred_single_slot_cache_releases[int(id(key_obj))] = release_cache
        setattr(transform, "_single_slot_release_diagonal_cache", None)

    def _release_deferred_single_slot_diagonal_caches(self) -> None:
        for release_cache in list(self._deferred_single_slot_cache_releases.values()):
            try:
                release_cache()
            except Exception:
                pass

    def _add_runtime_group(
        self,
        *,
        input_index: int,
        group: Any,
        target_indices: tuple[int, ...] | list[int],
    ) -> None:
        targets = tuple(int(value) for value in target_indices)
        if not targets:
            return
        input_index = int(input_index)
        self.runtime_groups.append(
            _RuntimeUnifiedTransformGroup(
                input_index=int(input_index),
                group=group,
                target_indices=targets,
            )
        )
        if str(self._compiled_lt_grouping_mode) == "shared":
            self.groups_by_input_index[int(input_index)] = group
            self.target_indices_by_input_index[int(input_index)] = targets

    def _validate_runtime_group_coverage(self, *, context: str) -> None:
        expected_inputs = int(self.cols)
        expected_targets = int(self.rows)
        if expected_inputs < 0 or expected_targets < 0:
            raise RuntimeError(
                f"Native halo Conv2d {self.output_node_id!r} has invalid runtime dimensions "
                f"cols={expected_inputs} rows={expected_targets} while validating {context}."
            )
        covered_targets: set[int] = set()
        for runtime_group in self.runtime_groups:
            input_index = int(runtime_group.input_index)
            if input_index < 0 or input_index >= int(expected_inputs):
                raise RuntimeError(
                    f"Native halo Conv2d {self.output_node_id!r} {context} has input_index={input_index} "
                    f"outside [0, {int(expected_inputs)})."
                )
            target_indices = tuple(int(value) for value in runtime_group.target_indices)
            if not target_indices:
                raise RuntimeError(
                    f"Native halo Conv2d {self.output_node_id!r} {context} has an empty target set "
                    f"for input_index={input_index}."
                )
            for target_index in target_indices:
                if target_index < 0 or target_index >= int(expected_targets):
                    raise RuntimeError(
                        f"Native halo Conv2d {self.output_node_id!r} {context} has target_index={target_index} "
                        f"outside [0, {int(expected_targets)})."
                    )
                covered_targets.add(int(target_index))
        required_targets = set(range(int(expected_targets)))
        missing = sorted(required_targets.difference(covered_targets))
        if missing:
            preview = ", ".join(str(value) for value in missing[:8])
            suffix = "" if len(missing) <= 8 else f", ... (+{len(missing) - 8})"
            raise RuntimeError(
                f"Native halo Conv2d {self.output_node_id!r} {context} does not cover output target(s) "
                f"{preview}{suffix}; re-save the provider compile cache for this shape/layout."
            )

    def _runtime_group_metadata_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "input_index": int(runtime_group.input_index),
                "storage_key": str(getattr(runtime_group.group, "_storage_key", "")),
                "target_indices": [int(value) for value in runtime_group.target_indices],
            }
            for runtime_group in self.runtime_groups
        ]

    def _compile_ordered_runtime_groups(
        self,
        scheme: Any,
        *,
        input_index: int,
        ordered: list[tuple[int, Any]],
        retune_shared_group: bool,
    ) -> None:
        ordered = sorted(
            [(int(target_index), transform) for target_index, transform in ordered],
            key=lambda item: int(item[0]),
        )
        if not ordered:
            return
        mode = self._compiled_lt_grouping_mode
        if str(mode) == "individual":
            for target_index, transform in ordered:
                retune_started = time.time()
                _retune_transform_group_bsgs([transform], slots=int(self.slots))
                self.last_runtime_timing["retune_bsgs_s"] = float(
                    self.last_runtime_timing.get("retune_bsgs_s", 0.0)
                ) + float(time.time() - retune_started)
                self._defer_individual_single_slot_cache_release(transform)
                group = UnifiedTransformGroup([transform])
                group_compile_started = time.time()
                group.compile_unified(scheme.backend)
                _record_group_compile_profile(self, group)
                _record_provider_diag_builder_metadata(self, getattr(transform, "_diag_builder_metadata", None))
                self.last_runtime_timing["group_compile_s"] = float(
                    self.last_runtime_timing.get("group_compile_s", 0.0)
                ) + float(time.time() - group_compile_started)
                self.last_runtime_timing["compiled_group_count"] = float(
                    self.last_runtime_timing.get("compiled_group_count", 0.0)
                ) + 1.0
                self._add_runtime_group(
                    input_index=int(input_index),
                    group=group,
                    target_indices=(int(target_index),),
                )
            return

        transforms = [transform for _target_index, transform in ordered]
        if bool(retune_shared_group):
            retune_started = time.time()
            _retune_transform_group_bsgs(transforms, slots=int(self.slots))
            self.last_runtime_timing["retune_bsgs_s"] = float(
                self.last_runtime_timing.get("retune_bsgs_s", 0.0)
            ) + float(time.time() - retune_started)
        group = UnifiedTransformGroup(transforms)
        group_compile_started = time.time()
        group.compile_unified(scheme.backend)
        _record_group_compile_profile(self, group)
        for transform in transforms:
            _record_provider_diag_builder_metadata(self, getattr(transform, "_diag_builder_metadata", None))
        self.last_runtime_timing["group_compile_s"] = float(
            self.last_runtime_timing.get("group_compile_s", 0.0)
        ) + float(time.time() - group_compile_started)
        self.last_runtime_timing["compiled_group_count"] = float(
            self.last_runtime_timing.get("compiled_group_count", 0.0)
        ) + 1.0
        self._add_runtime_group(
            input_index=int(input_index),
            group=group,
            target_indices=tuple(int(target_index) for target_index, _transform in ordered),
        )

    def _empty_compile_timing(self) -> dict[str, float]:
        return {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "build_transform_s": 0.0,
            "retune_bsgs_s": 0.0,
            "group_compile_s": 0.0,
            "diag_builder_build_s": 0.0,
            "diag_builder_wall_s": 0.0,
            "diag_builder_shadow_s": 0.0,
            "diag_builder_payload_count": 0.0,
            "diag_builder_fallback_count": 0.0,
            "built_transform_count": 0.0,
            "compiled_group_count": 0.0,
            "evaluate_unified_s": 0.0,
            "group_eval_s": 0.0,
            "partial_wrap_s": 0.0,
            "partial_rescale_s": 0.0,
            "partial_accumulate_s": 0.0,
            "postprocess_s": 0.0,
            "input_relayout_s": 0.0,
            "output_relayout_s": 0.0,
        }

    def _bias_chunk(self, *, block_index: int) -> torch.Tensor | None:
        if self.bias_vector is None:
            return None
        if self._uses_tight_compact_output():
            return self._compact_bias_chunk(block_index=int(block_index))
        stripe_group = self.native_plan.target_stripe_and_group_for_block(int(block_index))
        if stripe_group is None:
            return None
        stripe, target_group = stripe_group
        target_tile = int(self.native_plan.target_tile_for_stripe(stripe))
        channel_start = int(target_group) * int(target_tile)
        channel_end = min(int(self.spec.c_out), int(channel_start) + int(target_tile))
        out = torch.zeros((int(self.slots),), dtype=torch.float32)
        if int(channel_end) <= int(channel_start):
            return out
        active_target_h = int(stripe.target_h_end) - int(stripe.target_h_start)
        if int(active_target_h) <= 0:
            return out
        local_channels = torch.arange(int(channel_end - channel_start), dtype=torch.int64)
        out_h = torch.arange(int(active_target_h), dtype=torch.int64).repeat_interleave(int(self.spec.w_out))
        out_w = torch.arange(int(self.spec.w_out), dtype=torch.int64).repeat(int(active_target_h))
        flat_index = _idx_chw_gap_channel_positions(
            local_channels,
            h=out_h,
            w=out_w,
            height=int(stripe.target_h),
            width=int(self.spec.w_out),
            gap=int(self.spec.gap_out),
        )
        values = (
            self.bias_vector.detach().to(dtype=torch.float32)[int(channel_start): int(channel_end), None]
            .expand_as(flat_index)
            .reshape(-1)
        )
        out.index_copy_(0, flat_index.reshape(-1).to(dtype=torch.int64), values)
        return out

    def _compact_bias_chunk(self, *, block_index: int) -> torch.Tensor | None:
        if self.bias_vector is None:
            return None
        spec = self.native_plan.spec
        start = int(block_index) * int(self.slots)
        stop = int(start) + int(self.slots)
        out = torch.zeros((int(self.slots),), dtype=torch.float32)
        compact_output_h = _spec_physical_output_h(spec)
        out_h = torch.arange(int(compact_output_h), dtype=torch.int64).repeat_interleave(int(spec.w_out))
        out_w = torch.arange(int(spec.w_out), dtype=torch.int64).repeat(int(compact_output_h))
        channels = torch.arange(int(spec.c_out), dtype=torch.int64)
        flat_index = _idx_chw_gap_channel_positions(
            channels,
            h=out_h,
            w=out_w,
            height=int(compact_output_h),
            width=int(spec.w_out),
            gap=int(spec.gap_out),
        )
        mask = (flat_index >= int(start)) & (flat_index < int(stop))
        if bool(mask.any().item()):
            positions = (flat_index[mask] - int(start)).to(dtype=torch.int64)
            values = self.bias_vector.detach().to(dtype=torch.float32)[:, None].expand_as(flat_index)[mask]
            out.index_copy_(0, positions, values)
        return out

    def _compile_bias_plaintexts_at_level(self, scheme: Any, *, level: int) -> tuple[Any | None, ...]:
        if self.bias_vector is None:
            return ()
        scale = int(scheme.params.get_default_scale())
        plaintexts: list[Any | None] = []
        for block_index in range(int(self.rows)):
            chunk = self._bias_chunk(block_index=int(block_index))
            ptxt = None if chunk is None else scheme.encode(chunk, level=int(level), scale=int(scale))
            if ptxt is not None:
                self._bias_plaintext_cache[(int(block_index), int(level))] = ptxt
            plaintexts.append(ptxt)
        return tuple(plaintexts)

    def _compile_from_cache_metadata(self, scheme: Any) -> bool:
        metadata = dict(self._compile_cache_metadata or {})
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "load" or not metadata:
            return False

        requested_mode = self._lt_grouping_mode()
        stored_mode = self._normalise_lt_grouping_mode(metadata.get("lt_grouping_mode", "shared"))
        if str(stored_mode) != str(requested_mode):
            raise RuntimeError(
                f"Cached native halo Conv2d manifest for {self.output_node_id!r} was compiled with "
                f"lt_grouping_mode={stored_mode!r}, but this run requested {requested_mode!r}; "
                "re-save the compile cache for the requested provider LT grouping mode."
            )
        stored_plan = dict(metadata.get("native_halo_conv2d_plan", {}) or {})
        if not stored_plan:
            raise RuntimeError(
                f"Cached native halo Conv2d manifest for {self.output_node_id!r} is missing "
                "native_halo_conv2d_plan; re-run with io_mode='save'."
            )
        stored_fold_mode = _normalise_channel_fold_mode(
            stored_plan.get("channel_fold_mode", metadata.get("native_halo_channel_fold_mode", "heuristic"))
        )
        requested_fold_mode = self._channel_fold_mode()
        if str(stored_fold_mode) != str(requested_fold_mode):
            raise RuntimeError(
                f"Cached native halo Conv2d manifest for {self.output_node_id!r} was compiled with "
                f"channel_fold_mode={stored_fold_mode!r}, but this run requested {requested_fold_mode!r}; "
                "re-save the compile cache for the requested native halo channel fold mode."
            )
        self._refresh_runtime_plan()
        current_plan = self.native_plan.to_dict()
        if _cache_plan_guard(stored_plan) != _cache_plan_guard(current_plan):
            raise RuntimeError(
                f"Cached native halo Conv2d manifest for {self.output_node_id!r} does not match the "
                "current native halo plan structure; re-save the compile cache for this shape/layout."
            )

        group_rows = list(metadata.get("runtime_groups") or metadata.get("groups_by_input_index", []))
        if not group_rows:
            raise RuntimeError(
                f"Cached native halo Conv2d manifest for {self.output_node_id!r} is missing "
                "runtime_groups; re-run with io_mode='save'."
            )

        self._compiled_lt_grouping_mode = str(stored_mode)
        self.rows = int(self._runtime_output_ct_count())
        self.cols = int(metadata.get("cols", self.cols))
        module_bias = getattr(self.module, "on_bias", None)
        self.bias_vector = None if module_bias is None else module_bias.detach().to(dtype=torch.float32)
        input_level = int(self._level(scheme))
        conv_level = int(input_level)
        conv_output_level = max(0, int(input_level - 1))
        self.bias_plaintexts = self._compile_bias_plaintexts_at_level(scheme, level=int(conv_output_level))
        self._reset_runtime_groups()
        self.input_relayout_kernel = None
        self.output_relayout_kernel = None
        self.last_runtime_timing = self._empty_compile_timing()
        self.last_runtime_counts = {}

        compile_started = time.time()
        for group_meta in group_rows:
            input_index = int(group_meta.get("input_index", 0))
            target_indices = tuple(int(value) for value in group_meta.get("target_indices", []))
            if not target_indices:
                continue
            storage_key = str(group_meta.get("storage_key", ""))
            if not storage_key:
                raise RuntimeError(
                    f"Cached native halo Conv2d manifest for {self.output_node_id!r} has an empty storage key "
                    f"for input_index={input_index}."
                )
            group = UnifiedTransformGroup(
                [_cached_transform_shell(level=int(conv_level), scheme=scheme) for _target in target_indices]
            )
            group._storage_key = storage_key
            group.compile_unified(scheme.backend)
            self._add_runtime_group(
                input_index=int(input_index),
                group=group,
                target_indices=target_indices,
            )
        if not self.runtime_groups:
            raise RuntimeError(
                f"Cached native halo Conv2d manifest for {self.output_node_id!r} did not contain any non-empty "
                "transform groups; re-run with io_mode='save'."
            )
        self._validate_runtime_group_coverage(context="cached compile metadata")
        self.compile_count += 1
        elapsed = float(time.time() - compile_started)
        self.last_runtime_timing["compile_unified_s"] = elapsed
        self.last_runtime_timing["group_compile_s"] = elapsed
        self.last_runtime_timing["compiled_group_count"] = float(len(self.runtime_groups))
        return True

    def _compact_physical_output_module_attrs(self) -> dict[str, Any]:
        spec = self.native_plan.spec
        gap = max(1, int(spec.gap_out))
        physical_top = max(0, int(spec.output_physical_top_beta or 0))
        physical_bottom = max(0, int(spec.output_physical_bottom_beta or 0))
        existing = dict(getattr(self.module, "layout_policy_output_layout", {}) or {})
        layout = {
            **existing,
            "top_beta": int(physical_top),
            "bottom_beta": int(physical_bottom),
            "physical_top_beta": int(physical_top),
            "physical_bottom_beta": int(physical_bottom),
            "gap": int(gap),
            "boundary_pruned": False,
        }
        layout.pop("alpha", None)
        layout.pop("beta", None)
        return {
            "fhe_output_shape": _compact_output_fhe_shape_for_spec(spec),
            "layout_policy_output_layout": layout,
            "layout_policy_output_row_offset": int(physical_top * gap),
            "layout_policy_output_materialization": (
                "fused_relayout"
                if (
                    str(getattr(self.module, "layout_policy_output_materialization", "") or "") == "fused_relayout"
                    and int(physical_top + physical_bottom) > 0
                )
                else ""
            ),
        }

    def _with_compact_physical_output_module_attrs(self, callback: Callable[[], Any]) -> Any:
        if not self._uses_tight_compact_output():
            return callback()
        attrs = self._compact_physical_output_module_attrs()
        saved = {name: getattr(self.module, name, _MISSING_ATTR) for name in attrs}
        try:
            for name, value in attrs.items():
                setattr(self.module, name, value)
            return callback()
        finally:
            for name, value in saved.items():
                if value is _MISSING_ATTR:
                    try:
                        delattr(self.module, name)
                    except AttributeError:
                        pass
                else:
                    setattr(self.module, name, value)

    def _compile_compact_source_layout_diagonals(
        self,
        scheme: Any,
        *,
        level: int,
    ) -> None:
        from orion.core import packing

        build_started = time.time()
        weight = getattr(self.module, "on_weight").detach()
        if int(getattr(self.module, "groups", 1) or 1) > 1:
            weight = packing.resolve_grouped_conv(self.module)
        single_slot = bool(_single_slot_layer_cache_enabled_for_scheme(scheme))
        if bool(single_slot):
            diagonals, output_rotations = self._with_compact_physical_output_module_attrs(
                lambda: packing.pack_conv2d_diagonal_indices(
                    self.module,
                    False,
                    allow_hybrid=False,
                )
            )
        else:
            diagonals, output_rotations = self._with_compact_physical_output_module_attrs(
                lambda: packing.direct_diagonalize_conv2d(
                    self.module,
                    weight,
                    int(self.slots),
                    str(scheme.params.get_embedding_method()),
                    False,
                    allow_hybrid=False,
                )
            )
            diagonals = packing.prune_zero_diagonal_blocks(diagonals, preserve_empty_rows=True)
        self.last_runtime_timing["build_transform_s"] = float(
            self.last_runtime_timing.get("build_transform_s", 0.0)
        ) + float(time.time() - build_started)
        if int(output_rotations) != 0:
            raise RuntimeError("compact-source provider diagonal generator does not support hybrid output rotations")

        shared_payload_cache: _BlockDiagonalCache | None = None
        if bool(single_slot) and _env_enabled("ORION_SINGLE_SLOT_COMPACT_SOURCE_SHARED_PAYLOAD_CACHE"):
            def build_all_compact_source_blocks() -> dict[tuple[int, int], dict[int, Any]]:
                packed, runtime_output_rotations = self._with_compact_physical_output_module_attrs(
                    lambda: packing.direct_diagonalize_conv2d(
                        self.module,
                        weight,
                        int(self.slots),
                        str(scheme.params.get_embedding_method()),
                        False,
                        allow_hybrid=False,
                    )
                )
                if int(runtime_output_rotations) != 0:
                    raise RuntimeError(
                        "compact-source provider diagonal generator does not support hybrid output rotations"
                    )
                blocks: dict[tuple[int, int], dict[int, Any]] = {}
                for (target_block, source_block), block in dict(packed).items():
                    block_map = dict(block or {})
                    if block_map:
                        blocks[(int(target_block), int(source_block))] = block_map
                return blocks

            shared_payload_cache = _BlockDiagonalCache(build_all_compact_source_blocks)

        diagonal_cache_by_source: dict[int, _BlockDiagonalCache] = {}

        def source_diagonal_cache(source_index: int) -> _BlockDiagonalCache:
            source_index = int(source_index)
            if shared_payload_cache is not None:
                return shared_payload_cache
            cache = diagonal_cache_by_source.get(int(source_index))
            if cache is not None:
                return cache

            def build_all_for_source(
                *,
                source_index=source_index,
            ) -> dict[tuple[int, int], dict[int, Any]]:
                requested_blocks = tuple(
                    (int(target_index), int(source_index))
                    for target_index in range(int(self.rows))
                )
                packed = self._with_compact_physical_output_module_attrs(
                    lambda: packing.pack_conv2d_blocks(
                        self.module,
                        False,
                        requested_blocks,
                        allow_hybrid=False,
                    )
                )
                blocks: dict[tuple[int, int], dict[int, Any]] = {}
                for target_block, source_block in requested_blocks:
                    block = dict(packed).get((int(target_block), int(source_block)), {})
                    block_map = dict(block or {})
                    if not block_map:
                        block_map = {0: torch.zeros((int(self.slots),), dtype=torch.float32)}
                    if block_map:
                        blocks[(int(target_block), int(source_block))] = block_map
                return blocks

            cache = _BlockDiagonalCache(build_all_for_source)
            diagonal_cache_by_source[int(source_index)] = cache
            return cache

        def build_block_diagonals(target_index: int, source_index: int):
            cache = source_diagonal_cache(int(source_index))
            block = cache.get_required(
                int(target_index),
                int(source_index),
                context=(
                    f"{self.native_plan.spec.family_label} compact-dense "
                    f"source_index={int(source_index)} target_index={int(target_index)}"
                ),
            )
            return {(0, 0): block}

        ordered_by_source: dict[int, list[tuple[int, Any]]] = {}
        for (target_index, source_index), diag_map in sorted(dict(diagonals).items()):
            if int(source_index) < 0 or int(source_index) >= int(self.cols):
                continue
            if int(target_index) < 0 or int(target_index) >= int(self.rows):
                continue
            diag_indices = tuple(
                sorted(int(index) for index in (diag_map if bool(single_slot) else dict(diag_map or {}).keys()))
            )
            if not diag_indices:
                continue
            baby, giant = _bsgs_rotation_sets(
                set(int(value) for value in diag_indices),
                slots=int(self.slots),
                n1=1,
            )
            transform = SimpleNamespace(
                name=(
                    f"native_halo_{self.native_plan.spec.family_label}"
                    f"_compact_dense_src{int(source_index)}_tgt{int(target_index)}"
                ),
                diagonals=(
                    {}
                    if bool(single_slot)
                    else {(0, 0): {int(index): value for index, value in sorted(dict(diag_map or {}).items())}}
                ),
                _single_slot_diag_indices_by_block=({(0, 0): tuple(diag_indices)} if bool(single_slot) else None),
                _single_slot_build_diagonals=(
                    (
                        lambda target_index=int(target_index), source_index=int(source_index): build_block_diagonals(
                            int(target_index),
                            int(source_index),
                        )
                    )
                    if bool(single_slot)
                    else None
                ),
                level=int(level),
                scheme=scheme,
                fhe_output_shape=torch.Size([1, int(self.slots)]),
                output_shape=torch.Size([1, int(self.slots)]),
                target_index=int(target_index),
                input_id=f"compact_source_block_{int(source_index)}",
                selected_n1=1,
                baby_shifts=tuple(sorted(int(value) for value in baby)),
                giant_shifts=tuple(sorted(int(value) for value in giant)),
                rotation_group_id=f"native_halo:{self.native_plan.spec.family_label}:compact_dense_src{int(source_index)}",
                rotation_cost_owner=bool(int(target_index) == 0),
                _single_slot_diagonal_cache=(
                    source_diagonal_cache(int(source_index)) if bool(single_slot) else None
                ),
                _single_slot_release_diagonal_cache=(
                    source_diagonal_cache(int(source_index)).release if bool(single_slot) else None
                ),
            )
            ordered_by_source.setdefault(int(source_index), []).append((int(target_index), transform))
            self.last_runtime_timing["built_transform_count"] = float(
                self.last_runtime_timing.get("built_transform_count", 0.0)
            ) + 1.0

        if not ordered_by_source:
            raise RuntimeError("compact-source provider diagonal generator produced no transforms")

        for source_index, ordered in sorted(ordered_by_source.items()):
            self._compile_ordered_runtime_groups(
                scheme,
                input_index=int(source_index),
                ordered=ordered,
                retune_shared_group=True,
            )

    def _add_bias(self, ct: Any, *, block_index: int) -> Any:
        if self.bias_vector is None:
            return ct
        level = int(ct.level())
        ptxt = self._bias_plaintext_cache.get((int(block_index), int(level)))
        if ptxt is None:
            chunk = self._bias_chunk(block_index=int(block_index))
            if chunk is None:
                return ct
            ptxt = _encode_plaintext_for_add(ct, chunk)
            self._bias_plaintext_cache[(int(block_index), int(level))] = ptxt
        return _add_plaintext_for_add(ct, ptxt)

    def compile_cache_metadata(self) -> dict[str, Any]:
        self._refresh_runtime_plan()
        lt_grouping_mode = (
            str(self._compiled_lt_grouping_mode)
            if self.runtime_groups
            else str(self._lt_grouping_mode())
        )
        runtime_group_rows = self._runtime_group_metadata_rows()
        return {
            "kind": type(self).__name__,
            "kernel_kind": self.kernel_kind,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "use_ct_pt_hybrid_packing": False,
            "native_halo_input_capable": True,
            "native_halo_output_capable": True,
            "same_shape_runtime_layout": "native_halo_stripe_no_ri_io",
            "native_internal_runtime_layout": "native_halo_stripe_no_ri",
            "native_halo_conv2d_plan": self.native_plan.to_dict(),
            "native_halo_channel_fold_mode": str(self.native_plan.channel_fold_mode),
            "input_physical_layout": self._input_physical_layout(),
            "runtime_input_ct_count": int(self.cols),
            "runtime_output_ct_count": int(self.rows),
            "compact_source_layout": dict(self._compact_source_layout()) if self._uses_compact_source_input() else {},
            "conv_lt_raw_submatrix_tasks": int(self.native_plan.submatrix_program_count),
            "conv_lt_effective_submatrix_tasks": int(self.native_plan.sharing_group_count),
            "input_relayout": {},
            "output_relayout": {},
            "runtime_output_storage_layout": self._compact_output_storage_layout()
            if self._uses_tight_compact_output()
            else "native_halo_stripe",
            "relayout_sparse_lt_tasks": 0,
            "native_c_only_rotations": int(self.native_plan.c_only_rotations),
            "native_cb_shared_rotations": int(self.native_plan.cb_shared_rotations),
            "native_shared_baby_rotations": int(self.native_plan.shared_baby_rotations),
            "native_shared_giant_rotations": int(self.native_plan.shared_giant_rotations),
            "lt_grouping_mode": str(lt_grouping_mode),
            "provider_lt_grouping_mode": str(lt_grouping_mode),
            "provider_disable_shared_rotation": bool(str(lt_grouping_mode) == "individual"),
            "runtime_group_count": int(len(self.runtime_groups)),
            "runtime_transform_count": int(sum(len(item.target_indices) for item in self.runtime_groups)),
            "runtime_groups": runtime_group_rows,
            "groups_by_input_index": runtime_group_rows if str(lt_grouping_mode) == "shared" else [],
        }

    def compile(self, scheme: Any) -> None:
        changed = self._refresh_runtime_plan()
        if bool(changed) and (
            self.runtime_groups or self.input_relayout_kernel is not None or self.output_relayout_kernel is not None
        ):
            self.cleanup(getattr(scheme, "backend", None))
        requested_mode = self._lt_grouping_mode()
        if self.runtime_groups and str(requested_mode) != str(self._compiled_lt_grouping_mode):
            self.cleanup(getattr(scheme, "backend", None))
        if self.runtime_groups:
            return
        self._compiled_lt_grouping_mode = str(requested_mode)
        if self._compile_from_cache_metadata(scheme):
            return
        prepare_started = time.time()
        self._validate_module()
        self.last_runtime_timing = self._empty_compile_timing()
        self.last_runtime_timing["prepare_plans_s"] = float(time.time() - prepare_started)
        self.last_runtime_counts = {}
        module_bias = getattr(self.module, "on_bias", None)
        self.bias_vector = None if module_bias is None else module_bias.detach().to(dtype=torch.float32)
        weight = getattr(self.module, "on_weight").detach().to(dtype=torch.float32)
        input_level = int(self._level(scheme))
        conv_level = int(input_level)
        conv_output_level = max(0, int(input_level - 1))
        self.bias_plaintexts = self._compile_bias_plaintexts_at_level(scheme, level=int(conv_output_level))

        compile_started = time.time()
        group_index = 0
        compact_output = self._uses_tight_compact_output()
        compact_source = self._uses_compact_source_input()
        compact_source_layout = self._compact_source_layout()
        provider_weight_np = (
            np.ascontiguousarray(weight.detach().cpu().to(dtype=torch.float32).numpy().reshape(-1), dtype=np.float32)
            if (
                _provider_diag_builder_enabled()
                and _env_enabled("ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE")
                and not _provider_diag_builder_shadow()
                and not bool(compact_source)
            )
            else None
        )
        if bool(compact_source):
            if bool(compact_output):
                self._compile_compact_source_layout_diagonals(scheme, level=int(conv_level))
                self.last_runtime_timing["prepare_transforms_s"] = float(time.time() - compile_started)
                self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)
                self.input_relayout_kernel = None
                self.output_relayout_kernel = None
                self.compile_count += 1
                return
            for source_block in range(int(self.cols)):
                ordered: list[tuple[int, Any]] = []
                for stripe in self.native_plan.stripes:
                    for target_group in range(int(self.native_plan.target_group_count_for_stripe(stripe))):
                        target_blocks = range(int(self.rows)) if bool(compact_output) else (None,)
                        for target_block in target_blocks:
                            build_started = time.time()
                            transform = _build_compact_source_conv_transform(
                                spec=self.native_plan.spec,
                                plan=self.native_plan,
                                weight=weight,
                                stripe=stripe,
                                source_block=int(source_block),
                                target_group=int(target_group),
                                level=int(conv_level),
                                scheme=scheme,
                                source_layout=compact_source_layout,
                                group_n1=1,
                                compact_target_block=target_block,
                            )
                            self.last_runtime_timing["build_transform_s"] = float(
                                self.last_runtime_timing.get("build_transform_s", 0.0)
                            ) + float(time.time() - build_started)
                            if transform is None:
                                continue
                            self.last_runtime_timing["built_transform_count"] = float(
                                self.last_runtime_timing.get("built_transform_count", 0.0)
                            ) + 1.0
                            ordered.append((int(transform.target_index), transform))
                if ordered:
                    self._compile_ordered_runtime_groups(
                        scheme,
                        input_index=int(source_block),
                        ordered=ordered,
                        retune_shared_group=True,
                    )
        else:
            for stripe in self.native_plan.stripes:
                for source_group in range(int(self.native_plan.source_group_count_for_stripe(stripe))):
                    ordered: list[tuple[int, Any]] = []
                    group_n1 = int(self.native_plan.group_n1s[int(group_index)])
                    target_group_count = int(self.native_plan.target_group_count_for_stripe(stripe))
                    if not bool(compact_output):
                        build_started = time.time()
                        ordered = _build_conv_transform_batch(
                            spec=self.native_plan.spec,
                            plan=self.native_plan,
                            weight=weight,
                            weight_np=provider_weight_np,
                            stripe=stripe,
                            source_group=int(source_group),
                            target_groups=[int(value) for value in range(int(target_group_count))],
                            level=int(conv_level),
                            scheme=scheme,
                            group_n1=int(group_n1),
                        )
                        self.last_runtime_timing["build_transform_s"] = float(
                            self.last_runtime_timing.get("build_transform_s", 0.0)
                        ) + float(time.time() - build_started)
                        self.last_runtime_timing["built_transform_count"] = float(
                            self.last_runtime_timing.get("built_transform_count", 0.0)
                        ) + float(len(ordered))
                    else:
                        for target_group in range(int(target_group_count)):
                            build_started = time.time()
                            transforms = _build_conv_transforms_for_compact_output(
                                spec=self.native_plan.spec,
                                plan=self.native_plan,
                                weight=weight,
                                stripe=stripe,
                                source_group=int(source_group),
                                target_group=int(target_group),
                                level=int(conv_level),
                                scheme=scheme,
                                group_n1=int(group_n1),
                            )
                            self.last_runtime_timing["build_transform_s"] = float(
                                self.last_runtime_timing.get("build_transform_s", 0.0)
                            ) + float(time.time() - build_started)
                            for target_index, transform in transforms:
                                self.last_runtime_timing["built_transform_count"] = float(
                                    self.last_runtime_timing.get("built_transform_count", 0.0)
                                ) + 1.0
                                ordered.append((int(target_index), transform))
                    input_index = int(self.native_plan.source_block_index(stripe, int(source_group)))
                    if ordered:
                        self._compile_ordered_runtime_groups(
                            scheme,
                            input_index=int(input_index),
                            ordered=ordered,
                            retune_shared_group=False,
                        )
                    group_index += 1

        self.input_relayout_kernel = None
        self.output_relayout_kernel = None
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.compile(scheme)
        self.last_runtime_timing["input_relayout_s"] = 0.0
        self.last_runtime_timing["output_relayout_s"] = 0.0
        self.last_runtime_timing["evaluate_unified_s"] = 0.0
        self.last_runtime_timing["group_eval_s"] = 0.0
        self.last_runtime_timing["partial_wrap_s"] = 0.0
        self.last_runtime_timing["partial_rescale_s"] = 0.0
        self.last_runtime_timing["partial_accumulate_s"] = 0.0
        for key in (
            "layer_cache_encode_s",
            "layer_cache_key_prepare_s",
            "layer_cache_evict_s",
            "layer_cache_turnover_s",
            "provider_layer_cache_batch_count",
            "provider_layer_cache_batch_materialize_s",
            "provider_layer_cache_encode_s",
            "provider_layer_cache_key_prepare_s",
            "provider_layer_cache_group_estimated_bytes_max",
        ):
            self.last_runtime_timing[str(key)] = 0.0
        ids = tuple(int(value) for value in getattr(source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            source_kind = "compact source blocks" if self._uses_compact_source_input() else "native source tiles"
            raise RuntimeError(
                f"{self.output_node_id} native halo-stripe Conv2d requires {source_kind}: "
                f"expected at least {self.cols} CTs, got {len(ids)}"
            )

        output_blocks: list[Any | None] = [None for _ in range(int(self.rows))]
        evaluated_group_count = 0
        partial_count = 0
        rescale_count = 0
        accumulate_count = 0
        fuse_output_rescale = bool(_unified_output_fusion_enabled()) and str(self._compiled_lt_grouping_mode) != "individual"
        sorted_groups = [
            runtime_group
            for _index, runtime_group in sorted(
                enumerate(self.runtime_groups),
                key=lambda item: (int(item[1].input_index), int(item[0])),
            )
        ]
        target_sum_output_ids: list[int] | None = None
        evaluate_started = time.time()
        if fuse_output_rescale and sorted_groups:
            group_started = time.time()
            target_sum_output_ids = UnifiedTransformGroup.evaluate_sources_with_target_sum(
                [runtime_group.group for runtime_group in sorted_groups],
                [int(ids[int(runtime_group.input_index)]) for runtime_group in sorted_groups],
                [runtime_group.target_indices for runtime_group in sorted_groups],
                int(self.rows),
                scheme.backend,
            )
            if target_sum_output_ids is not None:
                self.last_runtime_timing["group_eval_s"] += float(time.time() - group_started)
        if target_sum_output_ids is not None:
            if len(target_sum_output_ids) != int(self.rows):
                raise RuntimeError(
                    f"{self.output_node_id} target-sum reduction returned {len(target_sum_output_ids)} "
                    f"outputs for {self.rows} targets"
                )
            evaluated_group_count = int(len(sorted_groups))
            partial_count = int(sum(len(runtime_group.target_indices) for runtime_group in sorted_groups))
            accumulate_count = max(0, int(partial_count) - int(len(target_sum_output_ids)))
            for target_index, output_id in enumerate(target_sum_output_ids):
                wrap_started = time.time()
                output_blocks[int(target_index)] = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(self.slots)]),
                    torch.Size([1, int(self.slots)]),
                )
                self.last_runtime_timing["partial_wrap_s"] += float(time.time() - wrap_started)
        else:
            lt_evaluator = getattr(scheme, "lt_evaluator", None)
            provider_auto_group = bool(
                str(self._compiled_lt_grouping_mode) == "individual"
                and lt_evaluator is not None
                and callable(getattr(lt_evaluator, "dense_layer_cache_auto_group_enabled", None))
                and lt_evaluator.dense_layer_cache_auto_group_enabled()
            )
            provider_group_budget_bytes = (
                int(lt_evaluator._dense_layer_cache_auto_budget_bytes())
                if bool(provider_auto_group) and callable(getattr(lt_evaluator, "_dense_layer_cache_auto_budget_bytes", None))
                else 0
            )

            def runtime_group_can_pre_materialize(runtime_group) -> bool:
                if not bool(provider_auto_group):
                    return False
                group = runtime_group.group
                return bool(getattr(group, "_single_slot_layer_cache", False)) and bool(
                    getattr(group, "_single_slot_recipes", None)
                )

            def runtime_group_estimated_bytes(runtime_group) -> int:
                if not bool(provider_auto_group):
                    return 0
                group = runtime_group.group
                recipes = getattr(group, "_single_slot_recipes", None)
                if not bool(runtime_group_can_pre_materialize(runtime_group)):
                    return int(provider_group_budget_bytes) + 1
                total = 0
                for diag_idxs, level, _transform in recipes:
                    raw_diag_count = int(len(tuple(int(value) for value in diag_idxs)))
                    total += int(
                        lt_evaluator._dense_layer_cache_encoded_plaintext_estimate_bytes(
                            raw_diag_count=int(raw_diag_count),
                            level=int(level),
                        )
                        + lt_evaluator._dense_layer_cache_payload_estimate_bytes(
                            raw_diag_count=int(raw_diag_count)
                        )
                    )
                return int(total)

            def provider_runtime_batches() -> list[list[Any]]:
                if not bool(provider_auto_group):
                    return [[runtime_group] for runtime_group in sorted_groups]
                batches: list[list[Any]] = []
                current: list[Any] = []
                current_bytes = 0
                for runtime_group in sorted_groups:
                    group_bytes = int(runtime_group_estimated_bytes(runtime_group))
                    if current and int(current_bytes + group_bytes) > int(provider_group_budget_bytes):
                        batches.append(list(current))
                        current = []
                        current_bytes = 0
                    current.append(runtime_group)
                    current_bytes += int(group_bytes)
                if current:
                    batches.append(list(current))
                return batches

            def materialize_provider_batch(batch: list[Any]) -> None:
                if not bool(provider_auto_group):
                    return
                materializable_batch = [
                    runtime_group for runtime_group in batch if bool(runtime_group_can_pre_materialize(runtime_group))
                ]
                if not materializable_batch:
                    return
                batch_started = time.time()
                encoded = 0.0
                key_prepare = 0.0
                estimated = sum(int(runtime_group_estimated_bytes(runtime_group)) for runtime_group in materializable_batch)
                bypass_auto_memory_cap = bool(
                    len(batch) == 1
                    and len(materializable_batch) == 1
                    and int(estimated) > int(provider_group_budget_bytes)
                )
                if (
                    not bool(bypass_auto_memory_cap)
                    and callable(getattr(lt_evaluator, "_dense_layer_cache_check_auto_memory_cap", None))
                ):
                    lt_evaluator._dense_layer_cache_check_auto_memory_cap(
                        reason=f"provider_single_slot_materialize:{self.output_node_id}",
                        estimated_bytes=int(estimated),
                    )
                for runtime_group in materializable_batch:
                    group = runtime_group.group
                    if (
                        bool(getattr(group, "_single_slot_layer_cache", False))
                        and getattr(group, "unified_ids", None) is None
                    ):
                        timing = group._materialize_single_slot_for_eval(scheme.backend)
                        setattr(group, "_single_slot_prematerialized_timing", dict(timing))
                        encoded += float(timing.get("layer_cache_encode_s", 0.0) or 0.0)
                        key_prepare += float(timing.get("layer_cache_key_prepare_s", 0.0) or 0.0)
                self.last_runtime_timing["provider_layer_cache_batch_count"] = float(
                    self.last_runtime_timing.get("provider_layer_cache_batch_count", 0.0)
                ) + 1.0
                self.last_runtime_timing["provider_layer_cache_batch_materialize_s"] = float(
                    self.last_runtime_timing.get("provider_layer_cache_batch_materialize_s", 0.0)
                ) + float(time.time() - batch_started)
                self.last_runtime_timing["provider_layer_cache_encode_s"] = float(
                    self.last_runtime_timing.get("provider_layer_cache_encode_s", 0.0)
                ) + float(encoded)
                self.last_runtime_timing["provider_layer_cache_key_prepare_s"] = float(
                    self.last_runtime_timing.get("provider_layer_cache_key_prepare_s", 0.0)
                ) + float(key_prepare)
                self.last_runtime_timing["layer_cache_encode_s"] = float(
                    self.last_runtime_timing.get("layer_cache_encode_s", 0.0)
                ) + float(encoded)
                self.last_runtime_timing["layer_cache_key_prepare_s"] = float(
                    self.last_runtime_timing.get("layer_cache_key_prepare_s", 0.0)
                ) + float(key_prepare)
                self.last_runtime_timing["layer_cache_turnover_s"] = float(
                    self.last_runtime_timing.get("layer_cache_encode_s", 0.0)
                    + self.last_runtime_timing.get("layer_cache_key_prepare_s", 0.0)
                    + self.last_runtime_timing.get("layer_cache_evict_s", 0.0)
                )
                self.last_runtime_timing["provider_layer_cache_group_estimated_bytes_max"] = float(
                    max(
                        float(self.last_runtime_timing.get("provider_layer_cache_group_estimated_bytes_max", 0.0) or 0.0),
                        float(estimated),
                    )
                )

            def cleanup_materialized_provider_batch(batch: list[Any]) -> None:
                if not bool(provider_auto_group):
                    return
                for runtime_group in batch:
                    group = runtime_group.group
                    if (
                        bool(getattr(group, "_single_slot_layer_cache", False))
                        and getattr(group, "unified_ids", None) is not None
                    ):
                        try:
                            group._evict_single_slot_after_eval(scheme.backend)
                        finally:
                            setattr(group, "_single_slot_prematerialized_timing", None)

            def evaluate_runtime_group(runtime_group) -> None:
                nonlocal evaluated_group_count, partial_count, rescale_count, accumulate_count
                group_started = time.time()
                output_ids = runtime_group.group.evaluate_unified(int(ids[int(runtime_group.input_index)]), scheme.backend)
                self.last_runtime_timing["group_eval_s"] += float(time.time() - group_started)
                evaluated_group_count += 1
                for target_index, output_id in zip(runtime_group.target_indices, output_ids):
                    wrap_started = time.time()
                    partial = CipherTensor(
                        scheme,
                        [int(output_id)],
                        torch.Size([1, int(self.slots)]),
                        torch.Size([1, int(self.slots)]),
                    )
                    self.last_runtime_timing["partial_wrap_s"] += float(time.time() - wrap_started)
                    partial_count += 1
                    if not fuse_output_rescale:
                        rescale_started = time.time()
                        partial = _rescale_cipher_tensor(partial)
                        self.last_runtime_timing["partial_rescale_s"] += float(time.time() - rescale_started)
                        rescale_count += 1
                    if output_blocks[int(target_index)] is None:
                        output_blocks[int(target_index)] = partial
                    else:
                        accumulate_started = time.time()
                        lhs, rhs = _align_ciphertexts_for_add(output_blocks[int(target_index)], partial)
                        output_blocks[int(target_index)] = lhs + rhs
                        self.last_runtime_timing["partial_accumulate_s"] += float(time.time() - accumulate_started)
                        accumulate_count += 1
            for batch in provider_runtime_batches():
                try:
                    materialize_provider_batch(batch)
                    for runtime_group in batch:
                        evaluate_runtime_group(runtime_group)
                finally:
                    cleanup_materialized_provider_batch(batch)
        if fuse_output_rescale:
            for block_index, block_ct in enumerate(output_blocks):
                if block_ct is None:
                    continue
                rescale_started = time.time()
                output_blocks[int(block_index)] = _rescale_cipher_tensor(block_ct)
                self.last_runtime_timing["partial_rescale_s"] += float(time.time() - rescale_started)
                rescale_count += 1
        if str(self._compiled_lt_grouping_mode) == "individual":
            self._release_deferred_single_slot_diagonal_caches()
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)
        self.last_runtime_counts = {
            "group_count": int(evaluated_group_count),
            "runtime_group_count": int(len(sorted_groups)),
            "partial_count": int(partial_count),
            "partial_rescale_count": int(rescale_count),
            "partial_accumulate_count": int(accumulate_count),
            "target_count": int(self.rows),
        }

        postprocess_started = time.time()
        output_ids: list[int] = []
        for block_index, block_ct in enumerate(output_blocks):
            if block_ct is None:
                raise RuntimeError(f"missing native halo output block {block_index} for {self.output_node_id}")
            block_ct = self._add_bias(block_ct, block_index=int(block_index))
            block_ct.set_scale(int(scheme.params.get_default_scale()))
            output_ids.append(int(block_ct.ids[0]))
            block_ct.ids = []
        native_output = CipherTensor(
            scheme,
            output_ids,
            self.output_shape,
            self.runtime_native_fhe_output_shape(),
        )
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        compact_output_rotation_stats = (
            native_halo_conv2d_compact_output_rotation_stats(self.native_plan)
            if self._uses_tight_compact_output()
            else None
        )
        self.last_runtime_io = {
            "runtime_lowering": (
                f"native_halo_stripe_no_ri+{self._compact_output_storage_layout()}_output"
                if self._uses_tight_compact_output()
                else "native_halo_stripe_no_ri"
            ),
            "provider_executor": type(self).__name__,
            "native_halo_input_capable": True,
            "native_halo_output_capable": True,
            "use_ct_pt_hybrid_packing": False,
            "input_physical_layout": self._input_physical_layout(),
            "runtime_input_ct_count": int(self.cols),
            "native_input_ct_count": int(self.native_plan.input_ct_count),
            "native_output_ct_count": int(self.rows),
            "native_output_storage_layout": self._compact_output_storage_layout()
            if self._uses_tight_compact_output()
            else "native_halo_stripe",
            "provider_lt_grouping_mode": str(self._compiled_lt_grouping_mode),
            "provider_disable_shared_rotation": bool(str(self._compiled_lt_grouping_mode) == "individual"),
            "runtime_group_count": int(len(self.runtime_groups)),
            "internal_input_relayout": False,
            "internal_output_relayout": False,
            "native_c_only_rotations": int(
                compact_output_rotation_stats.c_only_rotations
                if compact_output_rotation_stats is not None
                else self.native_plan.c_only_rotations
            ),
            "native_cb_shared_rotations": int(
                compact_output_rotation_stats.cb_shared_rotations
                if compact_output_rotation_stats is not None
                else self.native_plan.cb_shared_rotations
            ),
            "native_shared_baby_rotations": int(
                compact_output_rotation_stats.shared_baby_rotations
                if compact_output_rotation_stats is not None
                else self.native_plan.shared_baby_rotations
            ),
            "native_shared_giant_rotations": int(
                compact_output_rotation_stats.shared_giant_rotations
                if compact_output_rotation_stats is not None
                else self.native_plan.shared_giant_rotations
            ),
            "native_halo_channel_fold_mode": str(self.native_plan.channel_fold_mode),
            "native_plan_c_only_rotations": int(self.native_plan.c_only_rotations),
            "native_plan_cb_shared_rotations": int(self.native_plan.cb_shared_rotations),
        }
        return {self.output_node_id: native_output}

    def cleanup(self, backend: Any | None) -> None:
        if self.input_relayout_kernel is not None:
            self.input_relayout_kernel.cleanup(backend)
        if self.output_relayout_kernel is not None:
            self.output_relayout_kernel.cleanup(backend)
        self._release_deferred_single_slot_diagonal_caches()
        if backend is not None:
            delete = getattr(backend, "DeleteLinearTransform", None)
            if callable(delete):
                groups: list[Any] = []
                seen: set[int] = set()
                for runtime_group in list(self.runtime_groups):
                    marker = int(id(runtime_group.group))
                    if marker not in seen:
                        seen.add(marker)
                        groups.append(runtime_group.group)
                for group in list(self.groups_by_input_index.values()):
                    marker = int(id(group))
                    if marker in seen:
                        continue
                    seen.add(marker)
                    groups.append(group)
                for group in groups:
                    for transform_id in list(getattr(group, "unified_ids", []) or []):
                        try:
                            delete(int(transform_id))
                        except Exception:
                            pass
        self._reset_runtime_groups()
        self._compiled_lt_grouping_mode = "shared"
