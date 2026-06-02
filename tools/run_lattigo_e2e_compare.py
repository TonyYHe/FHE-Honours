from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.orion import scheme
from orion.core import packing
from orion.core.bsgs_rotation_stats import (
    identity_galois_element,
    unified_bsgs_rotation_stats,
)
from orion.backend.python.tensors import CipherTensor
from orion.backend.python.compile_policy import auto_worker_count, policy_audit
from orion.models.resnet import ResNet18, ResNet20, ResNet32, ResNet34, ResNet50
from orion.models.ternaus import TernausVGGUNet
from orion.models.unet import UNet22, UNet22Encoder
from orion.models.vgg import VGG
from orion.nn.activation import Chebyshev
from orion.nn.linear import LinearTransform
from orion.nn.module import Module
from orion.nn.operations import Bootstrap

try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:
    def _dynamo_disable(fn):
        return fn


DEFAULT_OUT = Path("/tmp/orion_lattigo_e2e_compare.json")


def _layout_top_beta(layout: dict[str, Any]) -> int:
    return max(0, int(layout.get("top_beta", layout.get("alpha", 0)) or 0))


def _layout_bottom_beta(layout: dict[str, Any]) -> int:
    return max(0, int(layout.get("bottom_beta", layout.get("beta", 0)) or 0))


def _layout_physical_top_beta(layout: dict[str, Any]) -> int:
    return max(
        0,
        int(layout.get("physical_top_beta", layout.get("top_beta", layout.get("alpha", 0))) or 0),
    )


def _layout_physical_bottom_beta(layout: dict[str, Any]) -> int:
    return max(
        0,
        int(layout.get("physical_bottom_beta", layout.get("bottom_beta", layout.get("beta", 0))) or 0),
    )


def _explicit_layout_physical_top_beta(layout: dict[str, Any]) -> int:
    if "physical_top_beta" not in layout:
        return 0
    return max(0, int(layout.get("physical_top_beta", 0) or 0))


def _explicit_layout_physical_bottom_beta(layout: dict[str, Any]) -> int:
    if "physical_bottom_beta" not in layout:
        return 0
    return max(0, int(layout.get("physical_bottom_beta", 0) or 0))


def _layout_policy_input_layout_row() -> dict[str, Any] | None:
    audit = dict(getattr(scheme, "region_first_attach_audit", {}) or {})
    graph = dict(audit.get("graph_audit", {}) or {})
    for row in graph.get("layout_policy_node_layouts", []) or []:
        if str(dict(row).get("node", "")) != "x":
            continue
        layout = dict(dict(row).get("selected_layout", {}) or {})
        gap = max(1, int(layout.get("gap", 1) or 1))
        if (
            _layout_physical_top_beta(layout) <= 0
            and _layout_physical_bottom_beta(layout) <= 0
            and int(gap) == 1
        ):
            return None
        return dict(row)
    return None


def _layout_policy_plaintext_halo_input(x: torch.Tensor, row: dict[str, Any]) -> torch.Tensor:
    layout = dict(row.get("selected_layout", {}) or {})
    gap = max(1, int(layout.get("gap", 1)))
    top_beta = _layout_physical_top_beta(layout)
    bottom_beta = _layout_physical_bottom_beta(layout)
    compact = packing.multiplex(x, int(gap)).detach().cpu().to(dtype=torch.float32)
    if compact.dim() != 4:
        raise ValueError(f"layout-policy input halo expects NCHW input, got {tuple(compact.shape)}")
    top_rows = int(top_beta * gap)
    bottom_rows = int(bottom_beta * gap)
    if top_rows <= 0 and bottom_rows <= 0:
        return compact
    halo = torch.zeros(
        (
            int(compact.shape[0]),
            int(compact.shape[1]),
            int(compact.shape[2]) + int(top_rows) + int(bottom_rows),
            int(compact.shape[3]),
        ),
        dtype=torch.float32,
    )
    halo[:, :, int(top_rows) : int(top_rows) + int(compact.shape[2]), :] = compact
    if top_rows > 0:
        for h in range(int(top_rows)):
            halo[:, :, int(h), :] = compact[:, :, 0, :]
    if bottom_rows > 0:
        start = int(top_rows + compact.shape[2])
        for h in range(int(bottom_rows)):
            halo[:, :, int(start + h), :] = compact[:, :, int(compact.shape[2]) - 1, :]
    return halo


def _walk_executor_objects(root: Any):
    stack = [root]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("base_executor", "_delegate", "delegate", "executor"):
            child = None
            current_dict = getattr(current, "__dict__", {})
            if isinstance(current_dict, dict) and attr in current_dict:
                child = current_dict.get(attr)
            if child is None:
                child = getattr(current, attr, None)
            if child is not None:
                stack.append(child)


_RUNTIME_FAIRNESS_NUMERIC_KEYS = (
    "resident_compute_s",
    "serving_hot_s",
    "artifact_read_s",
    "artifact_load_s",
    "artifact_unload_s",
    "layer_cache_encode_s",
    "layer_cache_key_prepare_s",
    "layer_cache_evict_s",
    "layer_cache_turnover_s",
    "trim_s",
    "read_bundle_s",
    "load_keys_s",
    "load_plaintexts_s",
    "eval_s",
    "eval_total_s",
    "unload_s",
    "cpp_plan_s",
    "cpp_level_adjust_s",
    "cpp_baby_step_s",
    "cpp_giant_step_s",
    "stream_build_map_s",
    "stream_encode_hoist_s",
    "stream_load_payload_s",
    "stream_eval_s",
    "stream_accumulate_s",
    "cpp_push_s",
    "cpp_trim_s",
    "linear_wrapper_accumulate_s",
    "linear_wrapper_rescale_s",
    "linear_wrapper_bias_s",
    "linear_wrapper_output_rotation_s",
    "linear_wrapper_postprocess_s",
    "executor_wrap_s",
    "executor_postprocess_s",
    "executor_rescale_s",
    "executor_accumulate_s",
    "executor_overhead_s",
)


def _append_runtime_group(groups: list[Any], seen: set[int], group: Any) -> None:
    if group is None or id(group) in seen:
        return
    seen.add(id(group))
    groups.append(group)


def _executor_unified_groups(executor: Any) -> list[Any]:
    groups: list[Any] = []
    seen: set[int] = set()

    def emit_value(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for _key, child in sorted(value.items()):
                emit_value(child)
            return
        if isinstance(value, (list, tuple, set)):
            for child in list(value):
                emit_value(child)
            return
        nested_group = getattr(value, "group", None)
        if nested_group is not None and not hasattr(value, "last_runtime_timing"):
            emit_value(nested_group)
            return
        _append_runtime_group(groups, seen, value)

    for candidate in _walk_executor_objects(executor):
        for attr in (
            "group",
            "groups",
            "groups_by_input_block",
            "groups_by_input_chunk",
            "groups_by_pair",
            "groups_by_input_index",
            "groups_by_source",
            "runtime_groups",
            "_concat_unified_groups_by_input",
        ):
            emit_value(getattr(candidate, attr, None))
    return groups


def _runtime_fairness_mode_from_env() -> str:
    raw_single_slot = os.environ.get("ORION_SINGLE_SLOT_LAYER_CACHE", "")
    if str(raw_single_slot).strip().lower() in {"1", "true", "yes", "on"}:
        return "single_slot_layer_cache"
    raw = os.environ.get("ORION_LATTIGO_STREAMING_LT", "")
    raw_legacy = os.environ.get("ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT", "")
    if (
        str(raw_legacy).strip().lower() in {"1", "true", "yes", "on"}
        and str(raw).strip().lower() in {"1", "true", "yes", "on", "force", "always"}
    ):
        return "streaming_eval_encode"
    return "unknown"


def _aggregate_runtime_fairness(timings: list[dict[str, Any]], *, serving_hot_s: float) -> dict[str, Any]:
    payload: dict[str, Any] = {key: 0.0 for key in _RUNTIME_FAIRNESS_NUMERIC_KEYS}
    modes: list[str] = []
    resident_available = True
    for timing in timings:
        mode = str(timing.get("runtime_fairness_mode", "unknown") or "unknown")
        modes.append(mode)
        if mode == "streaming_eval_encode":
            resident_available = False
        for key in _RUNTIME_FAIRNESS_NUMERIC_KEYS:
            value = timing.get(key)
            if value is None:
                if key == "resident_compute_s":
                    resident_available = False
                continue
            try:
                payload[key] = float(payload.get(key, 0.0)) + float(value)
            except (TypeError, ValueError):
                if key == "resident_compute_s":
                    resident_available = False
    if not timings:
        mode = _runtime_fairness_mode_from_env()
        resident_available = False
    elif any(mode == "streaming_eval_encode" for mode in modes):
        mode = "streaming_eval_encode"
    elif any(mode == "single_slot_layer_cache" for mode in modes):
        mode = "single_slot_layer_cache"
    elif any(mode == "memory_bounded_load_eval" for mode in modes):
        mode = "memory_bounded_load_eval"
    elif modes and all(mode == "resident_compute" for mode in modes):
        mode = "resident_compute"
    else:
        mode = "unknown"
        resident_available = False
    payload["serving_hot_s"] = float(serving_hot_s)
    if not resident_available:
        payload["resident_compute_s"] = None
    payload["runtime_fairness_mode"] = str(mode)
    payload["source_count"] = int(len(timings))
    return payload


def _executor_runtime_overhead_components(timing: dict[str, Any]) -> dict[str, float]:
    executor_wrap = _timing_float(timing, "partial_wrap_s")
    executor_rescale = float(
        _timing_float(timing, "partial_rescale_s")
        + _timing_float(timing, "relayout_s")
    )
    executor_accumulate = float(
        _timing_float(timing, "accumulate_s")
        + _timing_float(timing, "partial_accumulate_s")
    )
    executor_postprocess_children = float(
        _timing_float(timing, "bias_s")
        + _timing_float(timing, "output_fold_s")
        + _timing_float(timing, "branch_extract_s")
        + _timing_float(timing, "output_relayout_s")
    )
    executor_prepostprocess_children = float(
        _timing_float(timing, "projection_s")
        + _timing_float(timing, "input_pack_s")
        + _timing_float(timing, "real_extract_s")
        + _timing_float(timing, "input_relayout_s")
    )
    postprocess_s = float(_timing_float(timing, "postprocess_s"))
    covered_rescale = (
        float(executor_rescale)
        if executor_rescale > 0.0
        and executor_postprocess_children + executor_rescale <= postprocess_s + 1e-9
        else 0.0
    )
    covered_accumulate = (
        float(executor_accumulate)
        if executor_accumulate > 0.0
        and executor_postprocess_children + covered_rescale + executor_accumulate <= postprocess_s + 1e-9
        else 0.0
    )
    postprocess_exclusive = max(
        float(executor_postprocess_children),
        float(postprocess_s - covered_rescale - covered_accumulate),
    )
    executor_postprocess = float(executor_prepostprocess_children + postprocess_exclusive)
    return {
        "executor_wrap_s": float(executor_wrap),
        "executor_postprocess_s": float(executor_postprocess),
        "executor_rescale_s": float(executor_rescale),
        "executor_accumulate_s": float(executor_accumulate),
        "executor_overhead_s": float(
            executor_wrap + executor_postprocess + executor_rescale + executor_accumulate
        ),
    }


def _executor_runtime_core_signature(
    *,
    module_name: str,
    node: str,
    timing: dict[str, Any],
) -> tuple[Any, ...]:
    timing_part = tuple(
        (key, round(_timing_float(timing, key), 12))
        for key in (
            "group_eval_s",
            "evaluate_unified_s",
            "postprocess_s",
            "bias_s",
            "output_fold_s",
            "projection_s",
            "input_pack_s",
            "real_extract_s",
            "branch_extract_s",
            "input_relayout_s",
            "output_relayout_s",
        )
    )
    return (str(module_name), str(node), timing_part)


def _merge_numeric_max(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in dict(incoming).items():
        if value is None:
            continue
        try:
            incoming_float = float(value)
        except (TypeError, ValueError):
            if key not in merged:
                merged[key] = value
            continue
        try:
            existing_float = float(merged.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            existing_float = 0.0
        merged[key] = max(float(existing_float), float(incoming_float))
    return merged


def _collect_runtime_fairness(net: torch.nn.Module, *, serving_hot_s: float) -> dict[str, Any]:
    timings: list[dict[str, Any]] = []
    executor_timings_by_signature: dict[tuple[Any, ...], dict[str, Any]] = {}
    for _module_name, module in net.named_modules():
        layer_timing = getattr(module, "_last_runtime_timing", None)
        if isinstance(layer_timing, dict):
            timings.append(dict(layer_timing))
        for group in _executor_unified_groups(module):
            timing = getattr(group, "last_runtime_timing", None)
            if isinstance(timing, dict):
                timings.append(dict(timing))
        executor = getattr(getattr(module, "region_runtime", None), "executor", None)
        for candidate in _walk_executor_objects(executor):
            executor_timing = getattr(candidate, "last_runtime_timing", None)
            if isinstance(executor_timing, dict):
                node = str(getattr(module, "region_output_id", _module_name))
                signature = _executor_runtime_core_signature(
                    module_name=str(_module_name),
                    node=node,
                    timing=executor_timing,
                )
                enriched = dict(executor_timing)
                enriched.update(_executor_runtime_overhead_components(executor_timing))
                if signature in executor_timings_by_signature:
                    executor_timings_by_signature[signature] = _merge_numeric_max(
                        executor_timings_by_signature[signature],
                        enriched,
                    )
                else:
                    executor_timings_by_signature[signature] = enriched
        for group in _executor_unified_groups(executor):
            timing = getattr(group, "last_runtime_timing", None)
            if isinstance(timing, dict):
                timings.append(dict(timing))
    timings.extend(dict(timing) for timing in executor_timings_by_signature.values())
    evaluator_timing = getattr(getattr(scheme, "lt_evaluator", None), "last_runtime_timing", None)
    if isinstance(evaluator_timing, dict) and not timings:
        timings.append(dict(evaluator_timing))
    return _aggregate_runtime_fairness(timings, serving_hot_s=float(serving_hot_s))


_LT_PROFILE_COUNTER_NAMES = (
    "diag_terms",
    "q_mul",
    "qp_mul",
    "final_moddown",
    "transform_count",
    "baby_rotation_count",
    "giant_rotation_count",
    "inner_reduce_count",
    "outer_reduce_count",
)


def _set_lattigo_lt_profile_enabled(enabled: bool) -> bool:
    backend = getattr(scheme, "backend", None)
    enable = getattr(backend, "EnableLinearTransformEvaluationProfile", None)
    reset = getattr(backend, "ResetLinearTransformEvaluationProfile", None)
    if not callable(enable) or not callable(reset):
        return False
    if bool(enabled):
        reset()
        enable(1)
    else:
        enable(0)
    return True


def _collect_lattigo_lt_profile() -> dict[str, int]:
    backend = getattr(scheme, "backend", None)
    getter = getattr(backend, "GetLinearTransformEvaluationProfileCounters", None)
    if not callable(getter):
        return {}
    values = [int(value) for value in getter()]
    return {
        str(name): int(values[index]) if index < len(values) else 0
        for index, name in enumerate(_LT_PROFILE_COUNTER_NAMES)
    }


_BOOTSTRAP_PROFILE_COUNTER_NAMES = (
    "seq",
    "kind",
    "slots",
    "input_level",
    "output_level",
    "input_slots",
    "output_slots",
    "input_log_cols",
    "output_log_cols",
    "total_ns",
    "retrieve_ns",
    "copy_ns",
    "evaluator_bootstrap_ns",
    "postscale_ns",
    "push_ns",
    "input_degree",
    "output_degree",
)


def _set_lattigo_bootstrap_profile_enabled(enabled: bool) -> bool:
    backend = getattr(scheme, "backend", None)
    enable = getattr(backend, "EnableBootstrapProfile", None)
    reset = getattr(backend, "ResetBootstrapProfile", None)
    if not callable(enable) or not callable(reset):
        return False
    if bool(enabled):
        reset()
        enable(1)
    else:
        enable(0)
    return True


def _collect_lattigo_bootstrap_profile() -> dict[str, Any]:
    backend = getattr(scheme, "backend", None)
    getter = getattr(backend, "GetBootstrapProfileCounters", None)
    if not callable(getter):
        return {"available": False, "rows": [], "totals": {}}
    values = [int(value) for value in getter()]
    width = int(len(_BOOTSTRAP_PROFILE_COUNTER_NAMES))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(values), width):
        chunk = values[start : start + width]
        if len(chunk) != width:
            continue
        row = {
            str(name): int(chunk[index])
            for index, name in enumerate(_BOOTSTRAP_PROFILE_COUNTER_NAMES)
        }
        for key in (
            "total_ns",
            "retrieve_ns",
            "copy_ns",
            "evaluator_bootstrap_ns",
            "postscale_ns",
            "push_ns",
        ):
            row[key.replace("_ns", "_s")] = float(row[key]) / 1e9
        rows.append(row)

    totals = {
        "row_count": int(len(rows)),
        "total_s": float(sum(float(row.get("total_s", 0.0)) for row in rows)),
        "retrieve_s": float(sum(float(row.get("retrieve_s", 0.0)) for row in rows)),
        "copy_s": float(sum(float(row.get("copy_s", 0.0)) for row in rows)),
        "evaluator_bootstrap_s": float(
            sum(float(row.get("evaluator_bootstrap_s", 0.0)) for row in rows)
        ),
        "postscale_s": float(sum(float(row.get("postscale_s", 0.0)) for row in rows)),
        "push_s": float(sum(float(row.get("push_s", 0.0)) for row in rows)),
    }
    by_slots: dict[str, dict[str, float | int]] = {}
    for row in rows:
        key = str(int(row.get("slots", 0)))
        entry = by_slots.setdefault(
            key,
            {
                "row_count": 0,
                "total_s": 0.0,
                "evaluator_bootstrap_s": 0.0,
                "postscale_s": 0.0,
            },
        )
        entry["row_count"] = int(entry["row_count"]) + 1
        entry["total_s"] = float(entry["total_s"]) + float(row.get("total_s", 0.0))
        entry["evaluator_bootstrap_s"] = float(entry["evaluator_bootstrap_s"]) + float(
            row.get("evaluator_bootstrap_s", 0.0)
        )
        entry["postscale_s"] = float(entry["postscale_s"]) + float(row.get("postscale_s", 0.0))
    return {
        "available": True,
        "row_width": int(width),
        "field_names": list(_BOOTSTRAP_PROFILE_COUNTER_NAMES),
        "totals": totals,
        "by_slots": by_slots,
        "rows": rows,
    }


def _collect_provider_group_counts(net: torch.nn.Module) -> dict[str, Any]:
    module_rows: list[dict[str, Any]] = []
    totals = {
        "executor_count": 0,
        "source_group_count": 0,
        "target_count": 0,
        "partial_count": 0,
        "same_target_partial_count": 0,
        "same_target_accumulate_count": 0,
        "same_source_same_target_merge_count": 0,
        "runtime_partial_count": 0,
        "runtime_partial_accumulate_count": 0,
        "runtime_partial_rescale_count": 0,
    }
    seen: set[int] = set()
    seen_group_sets: set[tuple[int, ...]] = set()
    for module_name, module in net.named_modules():
        runtime = getattr(module, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        if executor is None:
            continue
        for candidate in _walk_executor_objects(executor):
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            groups_by_input = getattr(candidate, "groups_by_input_index", None)
            target_indices_by_input = getattr(candidate, "target_indices_by_input_index", None)
            if not isinstance(groups_by_input, dict) or not groups_by_input:
                continue
            group_set_key = tuple(
                id(group)
                for _input_index, group in sorted(groups_by_input.items())
            )
            if group_set_key in seen_group_sets:
                continue
            seen_group_sets.add(group_set_key)
            if not isinstance(target_indices_by_input, dict):
                target_indices_by_input = {}
            target_hist: dict[int, int] = {}
            partial_count = 0
            same_source_merge = 0
            for input_index, _group in sorted(groups_by_input.items()):
                per_source_targets = [
                    int(value)
                    for value in tuple(target_indices_by_input.get(int(input_index), ()) or ())
                ]
                partial_count += int(len(per_source_targets))
                per_source_hist: dict[int, int] = {}
                for target in per_source_targets:
                    per_source_hist[int(target)] = int(per_source_hist.get(int(target), 0)) + 1
                    target_hist[int(target)] = int(target_hist.get(int(target), 0)) + 1
                same_source_merge += sum(
                    max(0, int(count) - 1)
                    for count in per_source_hist.values()
                )
            same_target_partial = sum(
                int(count)
                for count in target_hist.values()
                if int(count) > 1
            )
            same_target_accumulate = sum(
                max(0, int(count) - 1)
                for count in target_hist.values()
            )
            runtime_counts = dict(getattr(candidate, "last_runtime_counts", {}) or {})
            row = {
                "module_path": str(module_name),
                "node": str(getattr(module, "region_output_id", module_name)),
                "executor": type(candidate).__name__,
                "source_group_count": int(len(groups_by_input)),
                "target_count": int(len(target_hist)),
                "partial_count": int(partial_count),
                "same_target_partial_count": int(same_target_partial),
                "same_target_accumulate_count": int(same_target_accumulate),
                "same_source_same_target_merge_count": int(same_source_merge),
                "runtime_counts": runtime_counts,
            }
            module_rows.append(row)
            totals["executor_count"] += 1
            totals["source_group_count"] += int(row["source_group_count"])
            totals["target_count"] += int(row["target_count"])
            totals["partial_count"] += int(row["partial_count"])
            totals["same_target_partial_count"] += int(row["same_target_partial_count"])
            totals["same_target_accumulate_count"] += int(row["same_target_accumulate_count"])
            totals["same_source_same_target_merge_count"] += int(row["same_source_same_target_merge_count"])
            totals["runtime_partial_count"] += int(runtime_counts.get("partial_count", 0) or 0)
            totals["runtime_partial_accumulate_count"] += int(runtime_counts.get("partial_accumulate_count", 0) or 0)
            totals["runtime_partial_rescale_count"] += int(runtime_counts.get("partial_rescale_count", 0) or 0)
    return {
        "totals": totals,
        "rows": module_rows,
    }


def _model_input_native_halo_plan(net: torch.nn.Module) -> dict[str, Any] | None:
    for module_name, module in net.named_modules():
        runtime = getattr(module, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        if runtime is None or executor is None:
            continue
        if not bool(getattr(executor, "native_halo_input", False)):
            continue
        if tuple(getattr(executor, "relayout_rows", ()) or ()):
            continue
        native_rows = tuple(dict(row) for row in (getattr(executor, "native_input_rows", ()) or ()))
        if not any(str(row.get("source", "")) == "x" for row in native_rows):
            continue
        plan_candidates: list[tuple[Any, Any]] = []
        for candidate in _walk_executor_objects(executor):
            plan = getattr(candidate, "native_plan", None)
            if plan is not None and hasattr(plan, "input_ct_count") and hasattr(plan, "stripes"):
                plan_candidates.append((candidate, plan))
        if plan_candidates:
            candidate, plan = next(
                (
                    (candidate, plan)
                    for candidate, plan in plan_candidates
                    if type(candidate).__name__ == "NativeHaloStripeNoRIConvExecutor"
                ),
                plan_candidates[-1],
            )
            return {
                "node": str(getattr(runtime, "module_prefix", "") or getattr(module, "region_output_id", "") or module_name),
                "module_path": str(module_name),
                "plan": plan,
                "native_rows": [dict(row) for row in native_rows],
                "executor": type(executor).__name__,
                "native_executor": type(candidate).__name__,
            }
    return None


def _encrypt_native_halo_model_input(x: torch.Tensor, input_level: int, native_input: dict[str, Any]) -> CipherTensor:
    from orion.experimental.cir.native_halo_conv2d import native_halo_source_plaintext_blocks_from_nchw

    plan = native_input["plan"]
    blocks = native_halo_source_plaintext_blocks_from_nchw(x, plan)
    ids: list[int] = []
    for block in blocks:
        ct = scheme.encrypt(scheme.encode(block, int(input_level)))
        ids.append(int(ct.ids[0]))
        ct.ids = []
    slots = int(getattr(plan.spec, "slot_count", scheme.params.get_slots()))
    return CipherTensor(
        scheme,
        ids,
        torch.Size(tuple(int(value) for value in x.shape)),
        torch.Size([int(len(ids)), int(slots)]),
    )


def _encrypt_model_input(
    x: torch.Tensor,
    input_level: int,
    *,
    net: torch.nn.Module | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    if net is not None:
        native_input = _model_input_native_halo_plan(net)
        if native_input is not None:
            ct = _encrypt_native_halo_model_input(x, int(input_level), native_input)
            if payload is not None:
                plan = native_input["plan"]
                payload["model_input_encoding"] = {
                    "kind": "native_halo_plaintext_source_tiles",
                    "node": str(native_input["node"]),
                    "module_path": str(native_input["module_path"]),
                    "executor": str(native_input["executor"]),
                    "native_executor": str(native_input["native_executor"]),
                    "input_ct_count": int(len(ct.ids)),
                    "native_plan_input_ct_count": int(plan.input_ct_count),
                    "stripe_count": int(len(plan.stripes)),
                    "source_channel_group_count": int(plan.source_channel_group_count),
                    "source_channel_tile": int(plan.source_channel_tile),
                    "slot_count": int(plan.spec.slot_count),
                }
            return ct
    row = _layout_policy_input_layout_row()
    if row is None:
        return scheme.encrypt(scheme.encode(x, int(input_level)))
    halo = _layout_policy_plaintext_halo_input(x, row)
    ct = scheme.encrypt(scheme.encode(halo, int(input_level)))
    if payload is not None:
        payload["model_input_encoding"] = {
            "kind": "flat_halo_plaintext",
            "input_ct_count": int(len(getattr(ct, "ids", ()) or ())),
            "layout": dict(row.get("selected_layout", {}) or {}),
            "selected_layout": dict(row.get("selected_layout", {}) or {}),
            "materialized_layout": {
                "top_beta": _layout_physical_top_beta(dict(row.get("selected_layout", {}) or {})),
                "bottom_beta": _layout_physical_bottom_beta(dict(row.get("selected_layout", {}) or {})),
                "gap": max(1, int(dict(row.get("selected_layout", {}) or {}).get("gap", 1) or 1)),
                "source": "physical",
            },
        }
    return ct


def _r18_config(provider_mode: str, *, backend: str = "lattigo") -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": 16,
            "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            "LogP": [61, 61, 61],
            "LogScale": 40,
            "H": 192,
            "RingType": "Standard",
        },
        "boot_params": {"LogP": [61, 61, 61, 61, 61, 61, 61, 61]},
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": str(backend),
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
            "experimental_region_first": str(provider_mode),
        },
    }


def _activation_or_relu(activation: str | None) -> str:
    return str(activation or "relu").lower()


def _stem_relu_for_activation(activation: str | None) -> bool:
    return _activation_or_relu(activation) == "relu"


def _build_r18_tiny(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return ResNet18(
        dataset="tiny",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
        stem_relu=_stem_relu_for_activation(activation),
    )


def _build_r20_cifar10(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return ResNet20(dataset="cifar10")


def _build_r18_imgnet(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return ResNet18(
        dataset="imagenet",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
        stem_relu=_stem_relu_for_activation(activation),
    )


def _build_r34_imgnet(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return ResNet34(
        dataset="imagenet",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
        stem_relu=_stem_relu_for_activation(activation),
    )


def _build_r32_imgnet(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return ResNet32(
        dataset="imagenet",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
        stem_relu=_stem_relu_for_activation(activation),
    )


def _build_r50_imgnet(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return ResNet50(
        dataset="imagenet",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
        stem_relu=_stem_relu_for_activation(activation),
    )


def _build_vgg_imgnet_base16(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return VGG(
        "VGG16",
        dataset="imagenet",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
        base_dim=16,
    )


def _build_vgg_imgnet_base64(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return VGG(
        "VGG16",
        dataset="imagenet",
        activation=_activation_or_relu(activation),
        silu_degree=int(silu_degree),
        base_dim=64,
    )


def _build_u22_64_base32(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22(
        dataset="montgomery_lung_64",
        base_dim=32,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_64_base8(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22(
        dataset="montgomery_lung_64",
        base_dim=8,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_256_base32(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22(
        dataset="kvasir_polyp_256",
        base_dim=32,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_256_base8(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22(
        dataset="kvasir_polyp_256",
        base_dim=8,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_256_base8_encoder(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22Encoder(
        dataset="kvasir_polyp_256",
        base_dim=8,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_192_base8_encoder(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22Encoder(
        dataset="kvasir_polyp_256",
        base_dim=8,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_224_base8_encoder(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22Encoder(
        dataset="kvasir_polyp_256",
        base_dim=8,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_u22_256_base64_encoder(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22Encoder(
        dataset="kvasir_polyp_256",
        base_dim=64,
        activation=activation,
        silu_degree=int(silu_degree),
    )


def _build_ternaus_vgg_unet_192_base64(
    *,
    activation: str | None = None,
    silu_degree: int = 31,
) -> torch.nn.Module:
    return TernausVGGUNet(
        in_channels=1,
        out_channels=1,
        base_dim=64,
        activation=activation,
        silu_degree=int(silu_degree),
    )


NETWORKS: dict[str, dict[str, Any]] = {
    "resnet20_cifar10": {
        "label": "ResNet20 CIFAR10",
        "model": "ResNet20",
        "dataset": "cifar10",
        "input_shape": (1, 3, 32, 32),
        "provider_mode": "",
        "config": _r18_config,
        "builder": _build_r20_cifar10,
    },
    "r18_tiny": {
        "label": "R18 Tiny",
        "model": "ResNet18",
        "dataset": "tiny",
        "input_shape": (1, 3, 64, 64),
        "provider_mode": "r18_tiny_e2e",
        "config": _r18_config,
        "builder": _build_r18_tiny,
    },
    "vgg16_imgnet": {
        "label": "VGG16/base16 ImageNet 224",
        "model": "VGG16-base16",
        "dataset": "imagenet",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "vgg_imgnet_layout_dp",
        "config": _r18_config,
        "builder": _build_vgg_imgnet_base16,
    },
    "vgg64_imgnet": {
        "label": "VGG16/base64 ImageNet 224",
        "model": "VGG16-base64",
        "dataset": "imagenet",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "vgg_imgnet_layout_dp",
        "config": _r18_config,
        "builder": _build_vgg_imgnet_base64,
    },
    "r18_imgnet": {
        "label": "R18 ImageNet",
        "model": "ResNet18",
        "dataset": "imagenet",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "r18_imgnet_layout_dp",
        "config": _r18_config,
        "builder": _build_r18_imgnet,
    },
    "r34_imgnet": {
        "label": "R34 ImageNet",
        "model": "ResNet34",
        "dataset": "imagenet",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "r34_imgnet_phase1",
        "config": _r18_config,
        "builder": _build_r34_imgnet,
    },
    "r32_imgnet": {
        "label": "R32 ImageNet 224",
        "model": "ResNet32",
        "dataset": "imagenet",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "generic_layout_dp",
        "config": _r18_config,
        "builder": _build_r32_imgnet,
    },
    "r50_imgnet": {
        "label": "R50 ImageNet 224",
        "model": "ResNet50",
        "dataset": "imagenet",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "generic_layout_dp",
        "config": _r18_config,
        "builder": _build_r50_imgnet,
    },
    "u22_64_base32": {
        "label": "U22 64 base32",
        "model": "UNet22",
        "dataset": "montgomery_lung_64",
        "input_shape": (1, 1, 64, 64),
        "provider_mode": "u22_64_base32",
        "config": _r18_config,
        "builder": _build_u22_64_base32,
    },
    "u22_64_base8": {
        "label": "U22 64 base8",
        "model": "UNet22",
        "dataset": "montgomery_lung_64",
        "input_shape": (1, 1, 64, 64),
        "provider_mode": "u22_64_base8",
        "config": _r18_config,
        "builder": _build_u22_64_base8,
    },
    "u22_256_base32": {
        "label": "U22 256 base32",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 256, 256),
        "provider_mode": "u22_256_base32",
        "config": _r18_config,
        "builder": _build_u22_256_base32,
    },
    "u22_224_base32": {
        "label": "U22 224 base32",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "u22_256_base32",
        "config": _r18_config,
        "builder": _build_u22_256_base32,
    },
    "u22_192_base32": {
        "label": "U22 192 base32",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 192, 192),
        "provider_mode": "u22_256_base32",
        "config": _r18_config,
        "builder": _build_u22_256_base32,
    },
    "u22_256_base8": {
        "label": "U22 256 base8",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 256, 256),
        "provider_mode": "u22_256_base8",
        "config": _r18_config,
        "builder": _build_u22_256_base8,
    },
    "u22_256_base8_encoder": {
        "label": "U22 256 base8 encoder",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 256, 256),
        "provider_mode": "u22_256_base8",
        "config": _r18_config,
        "builder": _build_u22_256_base8_encoder,
        "scope": "encoder",
    },
    "u22_192_base8_encoder": {
        "label": "U22 192 base8 encoder",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 192, 192),
        "provider_mode": "u22_256_base8",
        "config": _r18_config,
        "builder": _build_u22_192_base8_encoder,
        "scope": "encoder",
    },
    "u22_224_base8_encoder": {
        "label": "U22 224 base8 encoder",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "u22_256_base8",
        "config": _r18_config,
        "builder": _build_u22_224_base8_encoder,
        "scope": "encoder",
    },
    "u22_192_base64_encoder": {
        "label": "U22 192 base64 encoder",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 192, 192),
        "provider_mode": "u22_256_base32",
        "config": _r18_config,
        "builder": _build_u22_256_base64_encoder,
        "scope": "encoder",
        "base_dim": 64,
    },
    "u22_224_base64_encoder": {
        "label": "U22 224 base64 encoder",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 224, 224),
        "provider_mode": "u22_256_base32",
        "config": _r18_config,
        "builder": _build_u22_256_base64_encoder,
        "scope": "encoder",
        "base_dim": 64,
    },
    "u22_256_base64_encoder": {
        "label": "U22 256 base64 encoder",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "input_shape": (1, 3, 256, 256),
        "provider_mode": "u22_256_base32",
        "config": _r18_config,
        "builder": _build_u22_256_base64_encoder,
        "scope": "encoder",
        "base_dim": 64,
    },
    "ternaus_vgg_unet_192_base64": {
        "label": "Ternaus-style VGG-UNet 192 base64",
        "model": "TernausVGGUNet",
        "dataset": "medical_seg_192",
        "input_shape": (1, 1, 192, 192),
        "provider_mode": "generic_layout_dp",
        "config": _r18_config,
        "builder": _build_ternaus_vgg_unet_192_base64,
        "base_dim": 64,
    },
}


def _write(payload: dict[str, Any], out_path: Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _apply_io_config(
    config: dict[str, Any],
    *,
    backend: str | None = None,
    io_mode: str = "none",
    io_dir: Path | None = None,
    diags_path: Path | None = None,
    keys_path: Path | None = None,
    logn_override: int | None = None,
) -> dict[str, Any]:
    config = dict(config)
    config["ckks_params"] = dict(config.get("ckks_params", {}))
    config["orion"] = dict(config.get("orion", {}))
    if backend is not None:
        config["orion"]["backend"] = str(backend)
    if logn_override is not None:
        config["ckks_params"]["LogN"] = int(logn_override)
    config["orion"]["io_mode"] = str(io_mode)
    if io_dir is not None:
        diags_path = Path(io_dir) / "diagonals.h5" if diags_path is None else Path(diags_path)
        keys_path = Path(io_dir) / "keys.h5" if keys_path is None else Path(keys_path)
    if diags_path is not None:
        Path(diags_path).parent.mkdir(parents=True, exist_ok=True)
        config["orion"]["diags_path"] = str(Path(diags_path))
    if keys_path is not None:
        Path(keys_path).parent.mkdir(parents=True, exist_ok=True)
        config["orion"]["keys_path"] = str(Path(keys_path))
    return config


def _apply_ckks_preset(config: dict[str, Any], preset: str | None) -> dict[str, Any]:
    normalized = str(preset or "network-default").strip().lower()
    if normalized in {"", "network-default", "default"}:
        return config
    if normalized != "resnet":
        raise ValueError(f"Unsupported CKKS preset {preset!r}")
    config = dict(config)
    config["ckks_params"] = {
        "LogN": 16,
        "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
        "LogP": [61, 61, 61],
        "LogScale": 40,
        "H": 192,
        "RingType": "Standard",
    }
    config["boot_params"] = {"LogP": [61, 61, 61, 61, 61, 61, 61, 61]}
    return config


def _configure_cheddar_runtime_defaults() -> dict[str, str]:
    defaults = {
        "ORION_CHEDDAR_LT_STREAMING": "auto",
        "ORION_UNIFIED_LT_ROTKEY_RESIDENCY": "1",
        "ORION_UNIFIED_LT_PLAINTEXT_RESIDENCY": "1",
        "ORION_CHEDDAR_SHARED_CACHE_PLAN_PERSIST": "1",
        "ORION_CHEDDAR_GPU_PREFETCH": "0",
        "ORION_LATTIGO_BOOTSTRAP_MANY": "0",
    }
    applied: dict[str, str] = {}
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
        applied[name] = str(os.environ.get(name, ""))
    return applied


def _lattigo_compile_worker_default() -> str:
    cpu_count = max(1, int(os.cpu_count() or 1))
    return str(
        auto_worker_count(
            cpu_count,
            (
                "ORION_LT_COMPILE_WORKERS",
                "ORION_UNIFIED_COMPILE_WORKERS",
                "ORION_LATTIGO_COMPILE_WORKERS",
            ),
            default_workers=4,
            estimated_per_worker_bytes=24 * 1024**3,
            cpu_count=cpu_count,
        )
    )


def _lattigo_diagonal_encode_worker_default() -> str:
    cpu_count = max(1, int(os.cpu_count() or 1))
    return str(
        auto_worker_count(
            cpu_count,
            ("ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS",),
            default_workers=cpu_count,
            estimated_per_worker_bytes=4 * 1024**3,
            cpu_count=cpu_count,
        )
    )


def _lattigo_pack_worker_default() -> str:
    cpu_count = max(1, int(os.cpu_count() or 1))
    return str(
        auto_worker_count(
            cpu_count,
            ("ORION_PACK_CONV_WORKERS",),
            default_workers=8,
            estimated_per_worker_bytes=8 * 1024**3,
            cpu_count=cpu_count,
        )
    )


def _configure_lattigo_runtime_defaults() -> dict[str, str]:
    workers = _lattigo_compile_worker_default()
    diagonal_encode_workers = _lattigo_diagonal_encode_worker_default()
    pack_workers = _lattigo_pack_worker_default()
    defaults = {
        "ORION_LATTIGO_BOOTSTRAP_MANY": "0",
        "ORION_PACK_CONV_WORKERS": pack_workers,
        "ORION_LT_COMPILE_WORKERS": workers,
        "ORION_UNIFIED_COMPILE_WORKERS": workers,
        "ORION_UNIFIED_STREAM_COMPILE_BATCH_TRANSFORMS": workers,
        "ORION_UNIFIED_COMPILE_BATCH_TRANSFORMS": workers,
        "ORION_UNIFIED_LOAD_BATCH_TRANSFORMS": workers,
        "ORION_UNIFIED_CACHED_LOAD_BATCH_TRANSFORMS": workers,
        "ORION_LATTIGO_COMPILE_WORKERS": workers,
        "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS": diagonal_encode_workers,
        "ORION_SINGLE_SLOT_ENCODE_WORKERS": diagonal_encode_workers,
        "ORION_LATTIGO_BOOTSTRAP_WORKERS": workers,
        "ORION_UNIFIED_STREAM_COMPILE_BATCH_GB": "2",
        "ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE": "1",
    }
    applied: dict[str, str] = {}
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
        applied[name] = str(os.environ.get(name, ""))
    applied["ORION_COMPILE_PARALLEL_POLICY"] = os.environ.get("ORION_COMPILE_PARALLEL_POLICY", "auto")
    applied["compile_parallel_policy_audit"] = json.dumps(policy_audit(), sort_keys=True)
    return applied


def _timed(payload: dict[str, Any], out_path: Path, step: str, fn: Callable[[], Any]) -> Any:
    payload["step"] = str(step)
    _write(payload, out_path)
    started = time.perf_counter()
    value = fn()
    payload.setdefault("timing_s", {})[str(step)] = float(time.perf_counter() - started)
    _write(payload, out_path)
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(str(name))
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return float(value) if math.isfinite(float(value)) else float(default)


def _saved_io_prewarm_mode() -> str:
    return str(os.environ.get("ORION_SAVED_IO_PREWARM_MODE", "") or "").strip().lower()


def _saved_io_prewarm_enabled() -> bool:
    return _saved_io_prewarm_mode() not in {"", "0", "false", "no", "off"}


def _saved_io_prewarm_max_units() -> int | None:
    raw = os.environ.get("ORION_SAVED_IO_PREWARM_MAX_UNITS")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _prewarm_saved_io(scheme) -> dict[str, Any]:
    mode = _saved_io_prewarm_mode() or "raw"
    evaluator = getattr(scheme, "lt_evaluator", None)
    prewarm = getattr(evaluator, "prewarm_saved_io", None)
    if not callable(prewarm):
        return {
            "enabled": False,
            "mode": str(mode),
            "reason": "lt_evaluator_missing_prewarm_saved_io",
        }
    profile = prewarm(mode=str(mode), max_units=_saved_io_prewarm_max_units())
    return dict(profile or {})


def _align(reference: torch.Tensor, actual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left = reference.detach().cpu().to(dtype=torch.float32)
    right = actual.detach().cpu()
    if torch.is_complex(right):
        right = right.real
    right = right.to(dtype=torch.float32)
    if tuple(left.shape) != tuple(right.shape) and int(left.numel()) == int(right.numel()):
        right = right.reshape(tuple(left.shape))
    return left, right


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    return _layer_mae_metric(reference, actual)


def _layer_mae_tensor(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().cpu()
    if torch.is_complex(result):
        result = result.real
    return result.to(dtype=torch.float32)


def _layer_mae_tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    result = _layer_mae_tensor(value)
    if int(result.numel()) <= 0:
        return {
            "shape": [int(v) for v in tuple(result.shape)],
            "numel": 0,
            "checksum": 0.0,
            "l2": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "shape": [int(v) for v in tuple(result.shape)],
        "numel": int(result.numel()),
        "checksum": float(result.sum().item()),
        "l2": float(torch.linalg.vector_norm(result).item()),
        "min": float(result.min().item()),
        "max": float(result.max().item()),
    }


def _layer_mae_metric(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    left = _layer_mae_tensor(reference)
    right = _layer_mae_tensor(actual)
    original_right_shape = tuple(right.shape)
    shape_match = tuple(left.shape) == tuple(right.shape)
    numel_match = int(left.numel()) == int(right.numel())
    reshaped_for_diagnostic = False
    if not shape_match and bool(numel_match):
        right = right.reshape(tuple(left.shape))
        reshaped_for_diagnostic = True
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    compare_count = min(int(left_flat.numel()), int(right_flat.numel()))
    if compare_count <= 0:
        return {
            "mae": None,
            "max_abs": None,
            "rmse": None,
            "checksum_delta": None,
            "shape_match": bool(shape_match),
            "numel_match": bool(numel_match),
            "reshaped_for_diagnostic": bool(reshaped_for_diagnostic),
            "compare_count": 0,
            "reference_shape": [int(v) for v in tuple(left.shape)],
            "actual_shape": [int(v) for v in tuple(original_right_shape)],
            "compare_actual_shape": [int(v) for v in tuple(right.shape)],
        }
    diff = right_flat[:compare_count] - left_flat[:compare_count]
    return {
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.pow(2))).item()),
        "checksum_delta": float(diff.sum().item()),
        "shape_match": bool(shape_match),
        "numel_match": bool(numel_match),
        "reshaped_for_diagnostic": bool(reshaped_for_diagnostic),
        "compare_count": int(compare_count),
        "reference_shape": [int(v) for v in tuple(left.shape)],
        "actual_shape": [int(v) for v in tuple(original_right_shape)],
        "compare_actual_shape": [int(v) for v in tuple(right.shape)],
    }


def _layer_mae_infer_gap(*, logical_shape: Any, on_shape: Any) -> int:
    logical = tuple(int(v) for v in tuple(logical_shape or ()))
    physical = tuple(int(v) for v in tuple(on_shape or ()))
    if len(logical) != 4 or len(physical) != 4:
        return 1
    channels = max(1, int(logical[1]))
    width = max(1, int(logical[3]))
    physical_channels = max(1, int(physical[1]))
    physical_width = max(1, int(physical[3]))
    candidates = []
    for gap in (1, 2, 4, 8, 16):
        if int(width * gap) != int(physical_width):
            continue
        if int(math.ceil(float(channels) / float(gap * gap))) > int(physical_channels):
            continue
        candidates.append(int(gap))
    return int(max(candidates) if candidates else 1)


def _layer_mae_demultiplex(
    decoded: torch.Tensor,
    *,
    logical_shape: Any,
    on_shape: Any,
    gap: int | None,
    output_layout: dict[str, Any] | None = None,
) -> torch.Tensor:
    logical = tuple(int(v) for v in tuple(logical_shape or ()))
    physical = tuple(int(v) for v in tuple(on_shape or tuple(decoded.shape)))
    result = _layer_mae_tensor(decoded)
    if physical:
        physical_numel = int(torch.Size(physical).numel())
        result_numel = int(result.numel())
        if int(physical_numel) == int(result_numel):
            result = result.reshape(physical)
        elif len(physical) == 4 and 0 < int(physical_numel) < int(result_numel):
            result = result.reshape(-1)[: int(physical_numel)].reshape(physical)
    if len(logical) == 4 and len(tuple(result.shape)) == 4:
        actual_gap = int(gap) if gap is not None and int(gap) > 0 else _layer_mae_infer_gap(
            logical_shape=logical,
            on_shape=tuple(result.shape),
        )
        layout = dict(output_layout or {})
        top_rows = int(_explicit_layout_physical_top_beta(layout) * int(actual_gap))
        bottom_rows = int(_explicit_layout_physical_bottom_beta(layout) * int(actual_gap))
        if (top_rows > 0 or bottom_rows > 0) and int(result.shape[2]) > int(top_rows + bottom_rows):
            end = int(result.shape[2]) - int(bottom_rows) if bottom_rows > 0 else int(result.shape[2])
            result = result[:, :, int(top_rows) : int(end), :]
        return packing._demultiplex(
            result,
            int(actual_gap),
            int(logical[1]),
            int(logical[2]),
            int(logical[3]),
        )
    if logical and int(torch.Size(logical).numel()) == int(result.numel()):
        return result.reshape(logical)
    return result


def _layer_mae_native_halo_executor(module: torch.nn.Module) -> Any | None:
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None) if runtime is not None else None
    if executor is None:
        return None
    base = getattr(executor, "base_executor", executor)
    if not bool(getattr(base, "native_halo_output_capable", False)):
        return None
    if getattr(base, "native_plan", None) is None:
        return None
    return base


def _layer_mae_native_halo_expected_ct_count(executor: Any) -> int:
    if _layer_mae_native_halo_uses_compact_physical_targets(executor):
        plan = executor.native_plan
        return int(getattr(plan, "output_ct_count", 0) or 0)
    return int(_layer_mae_native_halo_native_target_ct_count(executor))


def _layer_mae_native_halo_native_target_ct_count(executor: Any) -> int:
    plan = executor.native_plan
    max_block_index = -1
    for stripe in plan.stripes:
        for target_group in range(int(plan.target_group_count_for_stripe(stripe))):
            max_block_index = max(
                int(max_block_index),
                int(plan.target_block_index(stripe, int(target_group))),
            )
    return int(max_block_index + 1)


def _layer_mae_native_halo_uses_compact_physical_targets(executor: Any) -> bool:
    module = getattr(executor, "module", None)
    storage = str(getattr(module, "native_halo_output_storage_layout", "") or "")
    if storage in {"tight_compact", "packed_compact", "logical_halo_compact"}:
        return True
    materialization = str(getattr(module, "layout_policy_output_materialization", "") or "")
    return bool(materialization in {"bootstrap_compact", "fused_relayout"})


def _layer_mae_output_relayout_present(module: Any, executor: Any | None = None) -> bool:
    runtime = getattr(module, "region_runtime", None)
    wrapper = getattr(runtime, "executor", None) if runtime is not None else None
    for owner in (wrapper, executor):
        if owner is None:
            continue
        if getattr(owner, "output_relayout_kernel", None) is not None:
            return True
        if tuple(getattr(owner, "output_relayout_rows", ()) or ()):
            return True
        if tuple(getattr(owner, "output_relayout_kernels", ()) or ()):
            return True
    return False


def _layer_mae_should_decode_native_halo_output(executor: Any | None, value: CipherTensor) -> bool:
    if executor is None:
        return False
    module = getattr(executor, "module", None)
    materialization = str(getattr(module, "layout_policy_output_materialization", "") or "")
    if materialization and materialization not in {"native_halo_stripe", "native_stripe", "channel_aligned_native_stripe"}:
        return False
    if _layer_mae_output_relayout_present(module, executor):
        return False
    return int(len(getattr(value, "ids", ()) or ())) == int(_layer_mae_native_halo_expected_ct_count(executor))


def _layer_mae_native_layout_preserving_no_decoder(module: torch.nn.Module) -> bool:
    if _module_category(module) != "activation":
        return False
    materialization = str(getattr(module, "layout_policy_output_materialization", "") or "")
    if materialization not in {"native_halo_stripe", "native_stripe", "channel_aligned_native_stripe"}:
        return False
    return _layer_mae_native_halo_executor(module) is None


def _layer_mae_decode_native_halo_output(decoded: torch.Tensor, executor: Any) -> torch.Tensor:
    plan = executor.native_plan
    spec = plan.spec
    values = _layer_mae_tensor(decoded).reshape(-1)
    expected_values = int(_layer_mae_native_halo_native_target_ct_count(executor)) * int(spec.slot_count)
    if int(values.numel()) < int(expected_values):
        raise RuntimeError(
            "native halo stripe decode expected at least "
            f"{int(expected_values)} slots, got {int(values.numel())}"
        )
    result = torch.zeros(
        (1, int(spec.c_out), int(spec.h_out), int(spec.w_out)),
        dtype=torch.float32,
    )
    gap = max(1, int(spec.gap_out))
    phases = int(gap * gap)
    for stripe in plan.stripes:
        target_tile = int(plan.target_tile_for_stripe(stripe))
        for target_group in range(int(plan.target_group_count_for_stripe(stripe))):
            block_index = int(plan.target_block_index(stripe, int(target_group)))
            block_base = int(block_index) * int(spec.slot_count)
            channel_start = int(target_group) * int(target_tile)
            channel_end = min(int(spec.c_out), int(channel_start) + int(target_tile))
            packed_w = int(spec.w_out) * int(gap)
            group_block = int(stripe.target_h) * int(gap) * int(packed_w)
            for local_channel, channel in enumerate(range(int(channel_start), int(channel_end))):
                group = int(local_channel) // int(phases)
                phase = int(local_channel) % int(phases)
                phase_h = int(phase) // int(gap)
                phase_w = int(phase) % int(gap)
                for global_h in range(int(stripe.target_h_start), int(stripe.target_h_end)):
                    if int(global_h) < 0 or int(global_h) >= int(spec.h_out):
                        continue
                    local_h = int(global_h) - int(stripe.target_h_start)
                    for w_index in range(int(spec.w_out)):
                        slot = (
                            int(block_base)
                            + int(group) * int(group_block)
                            + (int(local_h) * int(gap) + int(phase_h)) * int(packed_w)
                            + int(w_index) * int(gap)
                            + int(phase_w)
                        )
                        if int(slot) >= int(values.numel()):
                            raise RuntimeError(
                                "native halo stripe decode target slot out of range: "
                                f"slot={int(slot)} values={int(values.numel())}"
                            )
                        result[0, int(channel), int(global_h), int(w_index)] = values[int(slot)]
    return result


def _layer_mae_decode_native_halo_compact_physical_output(decoded: torch.Tensor, executor: Any) -> torch.Tensor:
    from orion.experimental.cir.native_halo_conv2d import _idx_chw_gap_channel_positions

    plan = executor.native_plan
    spec = plan.spec
    values = _layer_mae_tensor(decoded).reshape(-1)
    expected_values = int(_layer_mae_native_halo_expected_ct_count(executor)) * int(spec.slot_count)
    if int(values.numel()) < int(expected_values):
        raise RuntimeError(
            "native halo compact-physical decode expected at least "
            f"{int(expected_values)} slots, got {int(values.numel())}"
        )
    physical_top = max(0, int(getattr(spec, "output_physical_top_beta", 0) or 0))
    physical_bottom = max(0, int(getattr(spec, "output_physical_bottom_beta", 0) or 0))
    compact_output_h = int(spec.h_out) + int(physical_top) + int(physical_bottom)
    channels = torch.arange(int(spec.c_out), dtype=torch.int64)
    logical_h = torch.arange(int(spec.h_out), dtype=torch.int64).repeat_interleave(int(spec.w_out))
    output_h = logical_h + int(physical_top)
    output_w = torch.arange(int(spec.w_out), dtype=torch.int64).repeat(int(spec.h_out))
    flat_index = _idx_chw_gap_channel_positions(
        channels,
        h=output_h,
        w=output_w,
        height=int(compact_output_h),
        width=int(spec.w_out),
        gap=max(1, int(spec.gap_out)),
    )
    if bool((flat_index >= int(expected_values)).any().item()):
        raise RuntimeError(
            "native halo compact-physical decode target slot out of range: "
            f"max_slot={int(flat_index.max().item())} values={int(expected_values)}"
        )
    result = values[flat_index.reshape(-1).to(dtype=torch.int64)]
    return result.reshape(1, int(spec.c_out), int(spec.h_out), int(spec.w_out)).to(dtype=torch.float32)


def _layer_mae_decode_plain_raw(plain: Any) -> torch.Tensor:
    values: list[float] = []
    backend = getattr(plain, "backend", None)
    decode = getattr(backend, "Decode", None)
    if not callable(decode):
        return _layer_mae_tensor(plain.decode()).reshape(-1)
    for plaintext_id in getattr(plain, "ids", ()) or ():
        values.extend(decode(int(plaintext_id)))
    return torch.tensor(values, dtype=torch.float32)


def _layer_mae_shape_numel(shape: Any) -> int | None:
    if shape is None:
        return None
    try:
        return int(torch.Size(tuple(int(v) for v in tuple(shape))).numel())
    except Exception:
        return None


def _layer_mae_physical_fhe_shape_from_module(module: torch.nn.Module) -> tuple[int, int, int, int] | None:
    logical = tuple(int(v) for v in tuple(getattr(module, "output_shape", ()) or ()))
    if len(logical) != 4:
        return None
    gap = max(1, int(getattr(module, "output_gap", 1) or 1))
    layout = dict(getattr(module, "layout_policy_output_layout", {}) or {})
    physical_top = _explicit_layout_physical_top_beta(layout)
    physical_bottom = _explicit_layout_physical_bottom_beta(layout)
    n, channels, height, width = logical
    physical_channels = int(math.ceil(float(int(channels)) / float(int(gap) * int(gap))))
    physical_height = int(height) * int(gap) + int(physical_top + physical_bottom) * int(gap)
    physical_width = int(width) * int(gap)
    if int(physical_channels) <= 0 or int(physical_height) <= 0 or int(physical_width) <= 0:
        return None
    return (int(n), int(physical_channels), int(physical_height), int(physical_width))


def _layer_mae_select_decode_on_shape(
    module: torch.nn.Module,
    value: CipherTensor,
    decoded: Any,
    *,
    plain_on_shape: Any = None,
    decode_info: dict[str, Any] | None = None,
) -> Any:
    decoded_numel = int(_layer_mae_tensor(decoded).numel())
    module_fhe_shape = getattr(module, "fhe_output_shape", None)
    module_fhe_numel = _layer_mae_shape_numel(module_fhe_shape)
    decoded_shape = tuple(getattr(decoded, "shape", ()) or ())
    cipher_on_shape = getattr(value, "on_shape", None)
    plain_shape = None if plain_on_shape is None else tuple(plain_on_shape or ())
    slot_shapes = (cipher_on_shape, plain_shape, decoded_shape)
    has_ct_slot_shape = any(
        isinstance(shape, (list, tuple))
        and len(tuple(shape)) == 2
        for shape in slot_shapes
        if shape is not None
    )
    synthesized_fhe_shape = _layer_mae_physical_fhe_shape_from_module(module)
    synthesized_fhe_numel = _layer_mae_shape_numel(synthesized_fhe_shape)
    if (
        bool(has_ct_slot_shape)
        and module_fhe_shape is not None
        and len(tuple(module_fhe_shape)) == 4
        and module_fhe_numel is not None
        and 0 < int(module_fhe_numel) <= int(decoded_numel)
    ):
        if decode_info is not None:
            decode_info["decode_on_shape_source"] = (
                "module_fhe_output_shape"
                if int(module_fhe_numel) == int(decoded_numel)
                else "module_fhe_output_shape_slot_prefix"
            )
        return module_fhe_shape
    if (
        bool(has_ct_slot_shape)
        and synthesized_fhe_shape is not None
        and synthesized_fhe_numel is not None
        and 0 < int(synthesized_fhe_numel) <= int(decoded_numel)
    ):
        if decode_info is not None:
            decode_info["decode_on_shape_source"] = (
                "synthesized_physical_fhe_shape"
                if int(synthesized_fhe_numel) == int(decoded_numel)
                else "synthesized_physical_fhe_shape_slot_prefix"
            )
        return synthesized_fhe_shape
    exact_candidates = (
        ("cipher_on_shape", cipher_on_shape),
        ("plain_on_shape", plain_on_shape),
        ("decoded_shape", decoded_shape),
    )
    for source, shape in exact_candidates:
        if shape is None:
            continue
        if _layer_mae_shape_numel(shape) == int(decoded_numel):
            if decode_info is not None:
                decode_info["decode_on_shape_source"] = str(source)
            return shape
    if (
        module_fhe_shape is not None
        and len(tuple(module_fhe_shape)) == 4
        and module_fhe_numel is not None
        and 0 < int(module_fhe_numel) <= int(decoded_numel)
    ):
        if decode_info is not None:
            decode_info["decode_on_shape_source"] = (
                "module_fhe_output_shape"
                if int(module_fhe_numel) == int(decoded_numel)
                else "module_fhe_output_shape_slot_prefix"
            )
        return module_fhe_shape
    candidates = (("module_fhe_output_shape", module_fhe_shape),)
    fallback = getattr(module, "fhe_output_shape", getattr(value, "on_shape", tuple(getattr(decoded, "shape", ()))))
    for source, shape in candidates:
        if shape is None:
            continue
        if _layer_mae_shape_numel(shape) == int(decoded_numel):
            if decode_info is not None:
                decode_info["decode_on_shape_source"] = str(source)
            return shape
    if decode_info is not None:
        decode_info["decode_on_shape_source"] = "module_fhe_output_shape_fallback"
    return fallback


def _layer_mae_try_decode_stale_compact_raw(
    module: torch.nn.Module,
    value: CipherTensor,
    plain: Any,
) -> torch.Tensor | None:
    backend = getattr(plain, "backend", None)
    if not callable(getattr(backend, "Decode", None)):
        return None
    logical_shape = tuple(int(v) for v in tuple(getattr(module, "output_shape", getattr(value, "shape", ())) or ()))
    if len(logical_shape) != 4:
        return None
    if int(getattr(module, "output_gap", 1) or 1) != 1:
        return None
    if _layer_mae_output_relayout_present(module, None):
        return None
    logical_numel = int(torch.Size(logical_shape).numel())
    raw = _layer_mae_decode_plain_raw(plain).reshape(-1)
    if int(raw.numel()) != int(logical_numel):
        return None
    stale_numels = [
        _layer_mae_shape_numel(getattr(plain, "on_shape", None)),
        _layer_mae_shape_numel(getattr(value, "on_shape", None)),
        _layer_mae_shape_numel(getattr(module, "fhe_output_shape", None)),
    ]
    if not any(numel is not None and int(numel) != int(raw.numel()) for numel in stale_numels):
        return None
    return raw.reshape(logical_shape)


def _layer_mae_decode_cipher_output(
    module: torch.nn.Module,
    value: CipherTensor,
    *,
    decode_info: dict[str, Any] | None = None,
    native_executor_override: Any | None = None,
) -> torch.Tensor:
    plain = value.decrypt()
    plain_on_shape = None
    try:
        plain_on_shape = getattr(plain, "on_shape", None)
        if decode_info is not None:
            decode_info["plain_on_shape"] = _json_shape(plain_on_shape)
            decode_info["plaintext_count"] = int(len(getattr(plain, "ids", ()) or ()))
        native_executor = native_executor_override or _layer_mae_native_halo_executor(module)
        if _layer_mae_should_decode_native_halo_output(native_executor, value):
            raw_decoded = _layer_mae_decode_plain_raw(plain)
            if decode_info is not None:
                decode_info["decode_kind"] = (
                    "native_halo_compact_physical"
                    if bool(_layer_mae_native_halo_uses_compact_physical_targets(native_executor))
                    else "native_halo_stripe"
                )
                decode_info["raw_decoded_slot_count"] = int(raw_decoded.numel())
                decode_info["native_halo_expected_ct_count"] = int(_layer_mae_native_halo_expected_ct_count(native_executor))
            if bool(_layer_mae_native_halo_uses_compact_physical_targets(native_executor)):
                return _layer_mae_decode_native_halo_compact_physical_output(raw_decoded, native_executor)
            return _layer_mae_decode_native_halo_output(raw_decoded, native_executor)
        try:
            decoded = plain.decode()
            if decode_info is not None:
                decode_info["decoded_shape"] = _json_shape(getattr(decoded, "shape", ()))
                decode_info["decoded_numel"] = int(_layer_mae_tensor(decoded).numel())
        except RuntimeError:
            stale_compact = _layer_mae_try_decode_stale_compact_raw(module, value, plain)
            if stale_compact is None:
                raise
            if decode_info is not None:
                decode_info["decode_kind"] = "stale_compact_raw"
                decode_info["decoded_shape"] = _json_shape(getattr(stale_compact, "shape", ()))
                decode_info["decoded_numel"] = int(_layer_mae_tensor(stale_compact).numel())
                decode_info["raw_decoded_slot_count"] = int(_layer_mae_tensor(stale_compact).numel())
            return stale_compact
    finally:
        release = getattr(plain, "release", None)
        if callable(release):
            release()
    logical_shape = getattr(module, "output_shape", getattr(value, "shape", tuple(decoded.shape)))
    on_shape = _layer_mae_select_decode_on_shape(
        module,
        value,
        decoded,
        plain_on_shape=plain_on_shape,
        decode_info=decode_info,
    )
    gap = getattr(module, "output_gap", None)
    return _layer_mae_demultiplex(
        decoded,
        logical_shape=logical_shape,
        on_shape=on_shape,
        gap=gap,
        output_layout=dict(getattr(module, "layout_policy_output_layout", {}) or {}),
    )


def _layer_mae_decode_cipher_tensor(value: CipherTensor) -> torch.Tensor:
    plain = value.decrypt()
    try:
        decoded = plain.decode()
    finally:
        release = getattr(plain, "release", None)
        if callable(release):
            release()
    return _layer_mae_demultiplex(
        decoded,
        logical_shape=getattr(value, "shape", tuple(decoded.shape)),
        on_shape=getattr(value, "on_shape", tuple(decoded.shape)),
        gap=None,
    )


def _layer_mae_decode_concat_output(module: torch.nn.Module, value: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if type(value).__name__ != "ConcatCipherTensor" or not hasattr(value, "parts"):
        raise TypeError(f"expected ConcatCipherTensor, got {type(value).__name__}")
    parts = tuple(getattr(value, "parts", ()) or ())
    decoded_parts = []
    part_summaries = []
    for index, part in enumerate(parts):
        if not isinstance(part, CipherTensor):
            raise TypeError(f"concat part {int(index)} is {type(part).__name__}, expected CipherTensor")
        decoded = _layer_mae_decode_cipher_tensor(part)
        decoded_parts.append(decoded)
        part_summaries.append(
            {
                "index": int(index),
                "shape": [int(v) for v in tuple(getattr(part, "shape", ()))],
                "on_shape": [int(v) for v in tuple(getattr(part, "on_shape", ()))],
                "decoded": _layer_mae_tensor_summary(decoded),
            }
        )
    dim = int(getattr(module, "dim", 1))
    result = torch.cat(tuple(decoded_parts), dim=int(dim))
    return result, {
        "part_count": int(len(parts)),
        "parts": part_summaries,
        "decode_kind": "concat_parts_no_materialize",
    }


def _layer_mae_final_module(net: torch.nn.Module) -> tuple[str, torch.nn.Module] | None:
    candidates = [(name, module) for name, module in net.named_modules() if name and isinstance(module, Module)]
    if not candidates:
        return None
    return candidates[-1]


def _layer_mae_decode_model_output(net: torch.nn.Module, output: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    final = _layer_mae_final_module(net)
    if isinstance(output, CipherTensor) and final is not None:
        module_name, module = final
        return _layer_mae_decode_cipher_output(module, output), {
            "decode_kind": "module_aware_cipher_tensor",
            "module_path": str(module_name),
            "class": type(module).__name__,
        }
    if isinstance(output, CipherTensor):
        plain = output.decrypt()
        try:
            decoded = plain.decode()
        finally:
            release = getattr(plain, "release", None)
            if callable(release):
                release()
        return _layer_mae_tensor(decoded), {"decode_kind": "generic_cipher_tensor"}
    if type(output).__name__ == "ConcatCipherTensor" and hasattr(output, "parts"):
        raise RuntimeError("final model output is a lazy ConcatCipherTensor; direct final MAE is not supported")
    if isinstance(output, torch.Tensor):
        return _layer_mae_tensor(output), {"decode_kind": "torch_tensor"}
    raise TypeError(f"unsupported model output type for final decode: {type(output).__name__}")


def _layer_mae_target_names(net: torch.nn.Module) -> list[str]:
    names: list[str] = []
    for name, module in net.named_modules():
        if not name or not isinstance(module, Module):
            continue
        if isinstance(module, Bootstrap):
            continue
        if bool(getattr(module, "fused", False)) and not isinstance(module, Chebyshev):
            continue
        names.append(str(name))
    return names


def _install_layer_mae_clear_capture(
    net: torch.nn.Module,
    names: set[str],
) -> tuple[dict[str, torch.Tensor], Callable[[], None]]:
    outputs: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(module_name: str):
        @_dynamo_disable
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                outputs[str(module_name)] = _layer_mae_tensor(output)

        return hook

    for name, module in net.named_modules():
        if str(name) not in names:
            continue
        handles.append(module.register_forward_hook(make_hook(str(name))))

    def remove() -> None:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    return outputs, remove


def _layer_mae_module_by_name(net: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return {str(name): module for name, module in net.named_modules() if str(name)}


def _layer_mae_fused_reference_transforms(
    net: torch.nn.Module,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "source": "post_compile_network_dag",
        "status": "started",
        "error": "",
    }
    traced = getattr(getattr(net, "scheme", None), "traced", None)
    if traced is None:
        diagnostics["status"] = "no_traced_graph"
        return {}, diagnostics
    try:
        from orion.core.network_dag import NetworkDAG

        dag = NetworkDAG(traced)
        dag.build_dag()
    except Exception as exc:
        diagnostics["status"] = "dag_build_failed"
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return {}, diagnostics

    transforms: dict[str, dict[str, Any]] = {}
    module_paths = _layer_mae_module_by_name(net)
    module_name_by_id = {id(module): str(name) for name, module in module_paths.items()}
    for node in list(dag.nodes):
        module = dag.nodes[node].get("module")
        if not isinstance(module, Chebyshev) or not bool(getattr(module, "fused", False)):
            continue
        prescale = float(getattr(module, "prescale", 1.0))
        constant = float(getattr(module, "constant", 0.0))
        if prescale == 1.0 and constant == 0.0:
            continue
        for predecessor in dag.predecessors(node):
            predecessor_module = dag.nodes[predecessor].get("module")
            producer = predecessor_module
            producer_node = predecessor
            if not isinstance(producer, LinearTransform) and bool(getattr(predecessor_module, "fused", False)):
                linear_predecessors = [
                    parent
                    for parent in dag.predecessors(predecessor)
                    if isinstance(dag.nodes[parent].get("module"), LinearTransform)
                ]
                if len(linear_predecessors) == 1:
                    producer_node = linear_predecessors[0]
                    producer = dag.nodes[producer_node].get("module")
            if not isinstance(producer, LinearTransform):
                continue
            predecessor_name = str(module_name_by_id.get(id(producer), str(producer_node)))
            consumer_name = str(module_name_by_id.get(id(module), str(node)))
            transforms[str(predecessor_name)] = {
                "kind": "producer_fused_chebyshev_prescale",
                "consumer": str(consumer_name),
                "consumer_class": type(module).__name__,
                "producer_dag_node": str(producer_node),
                "consumer_dag_node": str(node),
                "via_fused_predecessor_dag_node": "" if str(producer_node) == str(predecessor) else str(predecessor),
                "via_fused_predecessor_class": (
                    "" if str(producer_node) == str(predecessor) else type(predecessor_module).__name__
                ),
                "prescale": float(prescale),
                "constant": float(constant),
            }
    diagnostics["status"] = "ok"
    diagnostics["transform_count"] = int(len(transforms))
    return transforms, diagnostics


def _layer_mae_adjust_clear_outputs_after_compile(
    net: torch.nn.Module,
    clear_outputs: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]], dict[str, Any]]:
    outputs = dict(clear_outputs or {})
    transforms, diagnostics = _layer_mae_fused_reference_transforms(net)
    applied: dict[str, dict[str, Any]] = {}
    for module_name, transform in sorted(transforms.items()):
        reference = outputs.get(str(module_name))
        if reference is None:
            continue
        prescale = float(transform.get("prescale", 1.0))
        constant = float(transform.get("constant", 0.0))
        outputs[str(module_name)] = _layer_mae_tensor(reference) * float(prescale) + float(constant)
        applied[str(module_name)] = dict(transform)
    diagnostics["applied_count"] = int(len(applied))
    diagnostics["unapplied_modules"] = sorted(str(name) for name in set(transforms) - set(applied))
    return outputs, applied, diagnostics


def _install_layer_mae_polynomial_clear_capture(
    net: torch.nn.Module,
    names: set[str],
) -> tuple[dict[str, torch.Tensor], Callable[[], None]]:
    outputs: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(module_name: str):
        @_dynamo_disable
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                outputs[str(module_name)] = _layer_mae_tensor(output)

        return hook

    for name, module in net.named_modules():
        if str(name) in names:
            handles.append(module.register_forward_hook(make_hook(str(name))))

    def remove() -> None:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    return outputs, remove


def _chebyshev_eval_tensor(value: torch.Tensor, coeffs: Any) -> torch.Tensor:
    coeff_list = [float(coeff) for coeff in (coeffs or [])]
    x = _layer_mae_tensor(value)
    if not coeff_list:
        return torch.zeros_like(x)
    t0 = torch.ones_like(x)
    result = t0 * float(coeff_list[0])
    if len(coeff_list) == 1:
        return result
    t1 = x
    result = result + t1 * float(coeff_list[1])
    for coeff in coeff_list[2:]:
        tk = 2.0 * x * t1 - t0
        result = result + tk * float(coeff)
        t0, t1 = t1, tk
    return result


def _activation_polynomial_reference(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor | None:
    if isinstance(module, Chebyshev):
        x = _layer_mae_tensor(value)
        if not bool(getattr(module, "fused", False)):
            prescale = float(getattr(module, "prescale", 1.0))
            constant = float(getattr(module, "constant", 0.0))
            if prescale != 1.0:
                x = x * float(prescale)
            if constant != 0.0:
                x = x + float(constant)
        return _chebyshev_eval_tensor(x, getattr(module, "coeffs", None))
    return None


def _collect_layer_mae_polynomial_clear_outputs(
    net: torch.nn.Module,
    x0: torch.Tensor,
    names: set[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "source": "post_compile_clear_forward_with_fitted_polynomials",
        "status": "started",
        "target_count": int(len(names)),
        "error": "",
    }
    outputs, remove = _install_layer_mae_polynomial_clear_capture(net, names)
    old_he_modes: list[tuple[Module, bool]] = []
    old_linear_params: list[tuple[LinearTransform, torch.Tensor, torch.Tensor | None]] = []
    old_chebyshev_forwards: list[tuple[Chebyshev, Any]] = []
    skipped_linear_substitutions: list[str] = []
    try:
        for module_name, module in net.named_modules():
            if isinstance(module, Module):
                old_he_modes.append((module, bool(getattr(module, "he_mode", False))))
                module.he_mode = False
            if isinstance(module, LinearTransform) and hasattr(module, "on_weight"):
                on_weight = _layer_mae_tensor(getattr(module, "on_weight"))
                if tuple(on_weight.shape) != tuple(module.weight.data.shape):
                    skipped_linear_substitutions.append(str(module_name))
                    continue
                old_weight = module.weight.data.detach().clone()
                old_bias = (
                    module.bias.data.detach().clone()
                    if hasattr(module, "bias") and module.bias is not None
                    else None
                )
                module.weight.data.copy_(on_weight.to(module.weight.device))
                if hasattr(module, "bias") and module.bias is not None and getattr(module, "on_bias", None) is not None:
                    on_bias = _layer_mae_tensor(getattr(module, "on_bias"))
                    if tuple(on_bias.shape) == tuple(module.bias.data.shape):
                        module.bias.data.copy_(on_bias.to(module.bias.device))
                old_linear_params.append((module, old_weight, old_bias))
            if isinstance(module, Chebyshev):
                old_chebyshev_forwards.append((module, module.forward))

                def _poly_forward(self, x):
                    reference = _activation_polynomial_reference(self, x)
                    if reference is None:
                        return x
                    return reference.to(device=x.device, dtype=x.dtype)

                module.forward = _poly_forward.__get__(module, type(module))
        with torch.no_grad():
            net(x0)
    except Exception as exc:
        diagnostics["status"] = "failed"
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        diagnostics["traceback"] = traceback.format_exc(limit=20)
    finally:
        remove()
        for module, old_he_mode in old_he_modes:
            module.he_mode = bool(old_he_mode)
        for module, old_forward in old_chebyshev_forwards:
            module.forward = old_forward
        for module, old_weight, old_bias in old_linear_params:
            module.weight.data.copy_(old_weight.to(module.weight.device))
            if old_bias is not None and hasattr(module, "bias") and module.bias is not None:
                module.bias.data.copy_(old_bias.to(module.bias.device))

    updated = dict(outputs)
    diagnostics["captured_count"] = int(len(outputs))
    diagnostics["linear_parameter_substitution_count"] = int(len(old_linear_params))
    diagnostics["skipped_linear_parameter_substitutions"] = sorted(skipped_linear_substitutions)
    diagnostics["polynomial_reference_count"] = int(len(old_chebyshev_forwards))
    diagnostics["polynomial_reference_modules"] = sorted(
        str(name)
        for name, module in _layer_mae_module_by_name(net).items()
        if isinstance(module, Chebyshev)
    )
    if str(diagnostics.get("status")) == "started":
        diagnostics["status"] = "ok"
    return updated, diagnostics


def _install_layer_mae_he_capture(
    net: torch.nn.Module,
    *,
    clear_outputs: dict[str, torch.Tensor],
    reference_transforms: dict[str, dict[str, Any]] | None = None,
    names: set[str],
    jsonl_path: Path,
) -> tuple[list[dict[str, Any]], Callable[[], None]]:
    rows: list[dict[str, Any]] = []
    handles: list[Any] = []
    if jsonl_path.exists():
        jsonl_path.unlink()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    module_by_name = _layer_mae_module_by_name(net)
    reference_transforms = dict(reference_transforms or {})

    def native_executor_for_layer_mae(module_name: str, module: torch.nn.Module) -> tuple[Any | None, str]:
        native_executor = _layer_mae_native_halo_executor(module)
        if native_executor is not None:
            return native_executor, str(module_name)
        if _module_category(module) != "activation":
            return None, ""
        if not str(module_name).endswith("_act"):
            return None, ""
        predecessor_name = str(module_name)[: -len("_act")]
        predecessor = module_by_name.get(predecessor_name)
        if predecessor is None:
            return None, ""
        native_executor = _layer_mae_native_halo_executor(predecessor)
        if native_executor is None:
            return None, ""
        return native_executor, str(predecessor_name)

    def append_row(row: dict[str, Any]) -> None:
        rows.append(row)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        metrics = row.get("metrics_vs_clear")
        if isinstance(metrics, dict):
            print(
                "[layer-mae] "
                f"node={row.get('module_path')} "
                f"type={row.get('class')} "
                f"status={row.get('status')} "
                f"mae={metrics.get('mae')} "
                f"max_abs={metrics.get('max_abs')} "
                f"rmse={metrics.get('rmse')} "
                f"shape_match={metrics.get('shape_match')}",
                flush=True,
            )
        else:
            print(
                "[layer-mae] "
                f"node={row.get('module_path')} "
                f"type={row.get('class')} "
                f"status={row.get('status')} "
                f"reason={row.get('skip_reason', '')} "
                f"error={row.get('error', '')}",
                flush=True,
            )

    def make_hook(module_name: str):
        @_dynamo_disable
        def hook(module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            started = time.perf_counter()
            runtime = getattr(module, "region_runtime", None)
            row: dict[str, Any] = {
                "module_path": str(module_name),
                "class": type(module).__name__,
                "category": _module_category(module),
                "status": "started",
                "level": None if getattr(module, "level", None) is None else int(getattr(module, "level")),
                "depth": None if getattr(module, "depth", None) is None else int(getattr(module, "depth")),
                "output_shape": _json_shape(getattr(module, "output_shape", ())),
                "fhe_output_shape": _json_shape(getattr(module, "fhe_output_shape", ())),
                "output_gap": None if getattr(module, "output_gap", None) is None else int(getattr(module, "output_gap")),
                "layout_policy_output_layout": dict(getattr(module, "layout_policy_output_layout", {}) or {}),
                "layout_policy_output_materialization": str(
                    getattr(module, "layout_policy_output_materialization", "") or ""
                ),
                "layout_policy_output_row_offset": (
                    None
                    if getattr(module, "layout_policy_output_row_offset", None) is None
                    else int(getattr(module, "layout_policy_output_row_offset"))
                ),
                "has_bootstrapper_child": bool(isinstance(getattr(module, "bootstrapper", None), Bootstrap)),
                "bootstrapper_path": (
                    f"{module_name}.bootstrapper" if isinstance(getattr(module, "bootstrapper", None), Bootstrap) else ""
                ),
                "region_runtime": bool(runtime is not None),
                "region_stage": str(getattr(runtime, "stage", "")) if runtime is not None else "",
                "region_strategy": str(getattr(runtime, "strategy", "")) if runtime is not None else "",
                "region_executable": bool(getattr(runtime, "executable", False)) if runtime is not None else False,
            }
            try:
                decode_info: dict[str, Any] = {}
                if type(output).__name__ == "ConcatCipherTensor" and hasattr(output, "parts"):
                    row["status"] = "skipped"
                    row["skip_reason"] = "lazy_concat_not_materialized"
                    row["decode_kind"] = "lazy_concat_not_materialized"
                    row["ciphertext_count"] = int(
                        sum(len(getattr(part, "ids", ()) or ()) for part in tuple(getattr(output, "parts", ()) or ()))
                    )
                    row["decode_s"] = float(time.perf_counter() - started)
                    append_row(row)
                    return
                elif isinstance(output, CipherTensor):
                    native_executor, native_executor_owner = native_executor_for_layer_mae(str(module_name), module)
                    native_decode = _layer_mae_should_decode_native_halo_output(native_executor, output)
                    decoded = _layer_mae_decode_cipher_output(
                        module,
                        output,
                        decode_info=decode_info,
                        native_executor_override=native_executor,
                    )
                    row["ciphertext_count"] = int(len(getattr(output, "ids", ()) or ()))
                    row["cipher_shape"] = _json_shape(getattr(output, "shape", ()))
                    row["cipher_on_shape"] = _json_shape(getattr(output, "on_shape", ()))
                    if native_executor_owner:
                        decode_info.setdefault("native_halo_executor_owner", str(native_executor_owner))
                    decode_info.setdefault("decode_kind", "native_halo_stripe" if bool(native_decode) else "cipher_tensor")
                elif isinstance(output, torch.Tensor):
                    decoded = _layer_mae_tensor(output)
                    row["ciphertext_count"] = 0
                    decode_info = {"decode_kind": "torch_tensor"}
                else:
                    row["status"] = "skipped"
                    row["skip_reason"] = f"unsupported_output_type:{type(output).__name__}"
                    row["decode_s"] = float(time.perf_counter() - started)
                    append_row(row)
                    return
                row.update(decode_info)
                row["actual"] = _layer_mae_tensor_summary(decoded)
                reference = clear_outputs.get(str(module_name))
                reference_transform = dict(reference_transforms.get(str(module_name), {}) or {})
                if reference_transform:
                    row["reference_transform"] = reference_transform
                    if reference is not None:
                        row["producer_fused_reference"] = _layer_mae_tensor_summary(reference)
                        row["producer_fused_metrics_vs_clear"] = _layer_mae_metric(reference, decoded)
                    consumer = str(reference_transform.get("consumer", ""))
                    row["status"] = "skipped"
                    row["skip_reason"] = "producer_fused_reference_covered_by_consumer"
                    if consumer in clear_outputs:
                        row["reference"] = _layer_mae_tensor_summary(clear_outputs[str(consumer)])
                    row["decode_s"] = float(time.perf_counter() - started)
                    append_row(row)
                    return
                if reference is None:
                    if reference_transform and str(reference_transform.get("consumer", "")) in clear_outputs:
                        consumer = str(reference_transform.get("consumer", ""))
                        row["status"] = "skipped"
                        row["skip_reason"] = "producer_fused_reference_covered_by_consumer"
                        row["reference"] = _layer_mae_tensor_summary(clear_outputs[str(consumer)])
                    elif bool(getattr(module_by_name.get(str(module_name)), "fused", False)):
                        row["status"] = "skipped"
                        row["skip_reason"] = "fused_module_no_clear_reference"
                    else:
                        row["status"] = "missing_clear_reference"
                        row["skip_reason"] = "missing_clear_reference"
                else:
                    row["reference"] = _layer_mae_tensor_summary(reference)
                    row["metrics_vs_clear"] = _layer_mae_metric(reference, decoded)
                    metrics = dict(row["metrics_vs_clear"])
                    if (
                        not bool(metrics.get("shape_match", False))
                        and isinstance(output, CipherTensor)
                        and _layer_mae_native_layout_preserving_no_decoder(module)
                    ):
                        row["status"] = "skipped"
                        row["skip_reason"] = "native_halo_layout_preserving_no_decoder"
                    elif not bool(metrics.get("shape_match", False)):
                        row["status"] = "shape_mismatch"
                        row["skip_reason"] = "shape_mismatch"
                    elif not bool(metrics.get("numel_match", False)):
                        row["status"] = "shape_mismatch"
                        row["skip_reason"] = "numel_mismatch"
                    else:
                        row["status"] = "ok"
            except BaseException as exc:
                row["status"] = "failed"
                row["error_type"] = type(exc).__name__
                row["error"] = str(exc)
                row["traceback"] = traceback.format_exc(limit=20)
            row["decode_s"] = float(time.perf_counter() - started)
            append_row(row)

        return hook

    for name, module in net.named_modules():
        if str(name) not in names:
            continue
        handles.append(module.register_forward_hook(make_hook(str(name))))

    def remove() -> None:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    return rows, remove


def _layer_mae_summary(rows: list[dict[str, Any]], *, expected_names: set[str] | None = None) -> dict[str, Any]:
    ok_rows = [row for row in rows if str(row.get("status")) == "ok" and isinstance(row.get("metrics_vs_clear"), dict)]
    skipped_rows = [row for row in rows if str(row.get("status")) != "ok"]
    allowed_skip_reasons = {
        "fused_module_no_clear_reference",
        "lazy_concat_not_materialized",
        "native_halo_layout_preserving_no_decoder",
        "producer_fused_reference_covered_by_consumer",
    }
    unexpected_rows = [
        row
        for row in skipped_rows
        if str(row.get("skip_reason", "")) not in allowed_skip_reasons
    ]
    observed = {str(row.get("module_path", "")) for row in rows if str(row.get("module_path", ""))}
    covered_by_consumer = {
        str(row.get("module_path", ""))
        for row in rows
        if str(row.get("skip_reason", "")) == "producer_fused_reference_covered_by_consumer"
    }
    missing_he_modules = sorted(str(name) for name in set(expected_names or set()) - observed - covered_by_consumer)
    reference_transform_rows = [
        {
            "module_path": str(row.get("module_path", "")),
            **dict(row.get("reference_transform", {}) or {}),
        }
        for row in rows
        if isinstance(row.get("reference_transform"), dict) and bool(row.get("reference_transform"))
    ]
    values = [
        float(row["metrics_vs_clear"]["mae"])
        for row in ok_rows
        if row.get("metrics_vs_clear", {}).get("mae") is not None
    ]
    max_values = [
        float(row["metrics_vs_clear"]["max_abs"])
        for row in ok_rows
        if row.get("metrics_vs_clear", {}).get("max_abs") is not None
    ]
    worst_by_mae = sorted(
        ok_rows,
        key=lambda row: float(row.get("metrics_vs_clear", {}).get("mae") or -1.0),
        reverse=True,
    )[:20]
    first_nonfinite = next(
        (
            row
            for row in ok_rows
            if not math.isfinite(float(row.get("metrics_vs_clear", {}).get("mae") or 0.0))
        ),
        None,
    )
    mae_threshold = _env_float("ORION_LAYER_MAE_MAX_MAE", 1e-3)
    max_abs_threshold = _env_float("ORION_LAYER_MAE_MAX_ABS", 1e-2)
    failing_threshold_rows = [
        row
        for row in ok_rows
        if (
            row.get("metrics_vs_clear", {}).get("mae") is None
            or row.get("metrics_vs_clear", {}).get("max_abs") is None
            or float(row.get("metrics_vs_clear", {}).get("mae") or 0.0) > float(mae_threshold)
            or float(row.get("metrics_vs_clear", {}).get("max_abs") or 0.0) > float(max_abs_threshold)
        )
    ]
    return {
        "row_count": int(len(rows)),
        "ok_count": int(len(ok_rows)),
        "skipped_count": int(len(skipped_rows)),
        "unexpected_count": int(len(unexpected_rows)),
        "missing_he_count": int(len(missing_he_modules)),
        "missing_he_modules": missing_he_modules,
        "reference_transform_count": int(len(reference_transform_rows)),
        "reference_transforms": reference_transform_rows,
        "overall_ok": bool(
            not unexpected_rows
            and not missing_he_modules
            and first_nonfinite is None
            and not failing_threshold_rows
        ),
        "max_mae": max(values) if values else None,
        "max_abs": max(max_values) if max_values else None,
        "mae_threshold": float(mae_threshold),
        "max_abs_threshold": float(max_abs_threshold),
        "threshold_failure_count": int(len(failing_threshold_rows)),
        "total_decode_s": float(sum(float(row.get("decode_s", 0.0) or 0.0) for row in rows)),
        "first_nonfinite": None if first_nonfinite is None else str(first_nonfinite.get("module_path", "")),
        "skipped_rows": [
            {
                "module_path": str(row.get("module_path", "")),
                "class": str(row.get("class", "")),
                "status": str(row.get("status", "")),
                "skip_reason": str(row.get("skip_reason", "")),
                "error": str(row.get("error", "")),
            }
            for row in skipped_rows
        ],
        "unexpected_rows": [
            {
                "module_path": str(row.get("module_path", "")),
                "class": str(row.get("class", "")),
                "status": str(row.get("status", "")),
                "skip_reason": str(row.get("skip_reason", "")),
                "error": str(row.get("error", "")),
            }
            for row in unexpected_rows
        ],
        "threshold_failures": [
            {
                "module_path": str(row.get("module_path", "")),
                "class": str(row.get("class", "")),
                "mae": row.get("metrics_vs_clear", {}).get("mae"),
                "max_abs": row.get("metrics_vs_clear", {}).get("max_abs"),
                "rmse": row.get("metrics_vs_clear", {}).get("rmse"),
                "shape_match": row.get("metrics_vs_clear", {}).get("shape_match"),
            }
            for row in sorted(
                failing_threshold_rows,
                key=lambda row: float(row.get("metrics_vs_clear", {}).get("mae") or -1.0),
                reverse=True,
            )[:20]
        ],
        "worst_by_mae": [
            {
                "module_path": str(row.get("module_path", "")),
                "class": str(row.get("class", "")),
                "mae": row.get("metrics_vs_clear", {}).get("mae"),
                "max_abs": row.get("metrics_vs_clear", {}).get("max_abs"),
                "rmse": row.get("metrics_vs_clear", {}).get("rmse"),
                "shape_match": row.get("metrics_vs_clear", {}).get("shape_match"),
            }
            for row in worst_by_mae
        ],
    }


def _tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu()
    if torch.is_complex(value):
        value = value.real
    value = value.to(dtype=torch.float32)
    return {
        "shape": [int(v) for v in tuple(value.shape)],
        "checksum": float(value.sum().item()),
        "l2": float(torch.linalg.vector_norm(value).item()),
        "values": [float(v) for v in value.reshape(-1).tolist()],
    }


def _json_shape(value: Any) -> list[int]:
    try:
        return [int(v) for v in tuple(value)]
    except Exception:
        return []


def _cipher_ids_state(value: Any) -> dict[str, Any]:
    ids = [int(v) for v in list(getattr(value, "ids", []) or [])]
    backend = getattr(value, "backend", None)
    if backend is None and getattr(value, "scheme", None) is not None:
        backend = getattr(value.scheme, "backend", None)
    levels: list[int | None] = []
    scales: list[int | None] = []
    scale_log2: list[float | None] = []
    slots: list[int | None] = []
    for cid in ids:
        try:
            levels.append(int(backend.GetCiphertextLevel(int(cid)))) if backend is not None else levels.append(None)
        except Exception:
            levels.append(None)
        try:
            scales.append(int(backend.GetCiphertextScale(int(cid)))) if backend is not None else scales.append(None)
        except Exception:
            scales.append(None)
        try:
            scale_log2.append(float(backend.GetCiphertextScaleLog2(int(cid)))) if backend is not None else scale_log2.append(None)
        except Exception:
            scale_log2.append(None)
        try:
            slots.append(int(backend.GetCiphertextSlots(int(cid)))) if backend is not None else slots.append(None)
        except Exception:
            slots.append(None)
    return {
        "kind": type(value).__name__,
        "id_count": int(len(ids)),
        "shape": _json_shape(getattr(value, "shape", ())),
        "on_shape": _json_shape(getattr(value, "on_shape", ())),
        "levels": levels,
        "scales": scales,
        "scale_log2": scale_log2,
        "slots": slots,
    }


def _value_profile(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if hasattr(value, "ids") and hasattr(value, "shape"):
        return _cipher_ids_state(value)
    if isinstance(value, torch.Tensor):
        return {
            "kind": "Tensor",
            "shape": [int(v) for v in tuple(value.shape)],
            "dtype": str(value.dtype),
            "is_complex": bool(torch.is_complex(value)),
        }
    if isinstance(value, (int, float, bool, str)) or value is None:
        return {"kind": type(value).__name__, "value": value}
    if depth >= 2:
        return {"kind": type(value).__name__}
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "length": int(len(value)),
            "items": [_value_profile(v, depth=int(depth) + 1) for v in list(value)[:4]],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "length": int(len(value)),
            "items": {
                str(k): _value_profile(v, depth=int(depth) + 1)
                for k, v in list(value.items())[:4]
            },
        }
    return {"kind": type(value).__name__}


def _module_category(module: Module) -> str:
    cls_name = type(module).__name__
    if isinstance(module, Bootstrap):
        return "bootstrap"
    if "Pool" in cls_name:
        return "pool"
    if isinstance(module, LinearTransform):
        if cls_name == "ConvTranspose2d":
            return "conv_transpose2d"
        if cls_name == "Conv2d":
            return "conv2d"
        return "linear_transform"
    if cls_name in {"Add"}:
        return "add"
    if cls_name in {"Concat"}:
        return "concat"
    if cls_name in {"Mult"}:
        return "multiply"
    if cls_name in {"Flatten"}:
        return "reshape"
    if cls_name in {
        "Activation",
        "Chebyshev",
        "ELU",
        "GELU",
        "Hardshrink",
        "Mish",
        "Quad",
        "ReLU",
        "SELU",
        "SiLU",
        "Sigmoid",
        "Softplus",
        "_Sign",
    }:
        return "activation"
    return "other"


def _timing_float(timing: dict[str, Any], key: str) -> float:
    try:
        return float(timing.get(str(key), 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _install_activation_breakdown_profiler(
    net: torch.nn.Module,
) -> tuple[Callable[[], dict[str, Any]], Callable[[], None]]:
    rows_by_id: dict[int, dict[str, Any]] = {}
    stacks: dict[int, list[float]] = {}
    handles: list[Any] = []

    def make_pre(row: dict[str, Any]):
        @_dynamo_disable
        def pre_hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if not bool(getattr(module, "he_mode", False)):
                return
            stacks.setdefault(int(id(module)), []).append(float(time.perf_counter()))

        return pre_hook

    def make_post(row: dict[str, Any]):
        @_dynamo_disable
        def post_hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not bool(getattr(module, "he_mode", False)):
                return
            stack = stacks.get(int(id(module))) or []
            if not stack:
                return
            elapsed = float(time.perf_counter() - stack.pop())
            row["call_count"] = int(row.get("call_count", 0)) + 1
            row["elapsed_s"] = float(row.get("elapsed_s", 0.0)) + elapsed
            row["max_call_s"] = max(float(row.get("max_call_s", 0.0)), elapsed)
            row["last_call_s"] = float(elapsed)

        return post_hook

    for name, module in net.named_modules():
        if not isinstance(module, Module):
            continue
        if _module_category(module) != "activation":
            continue
        bootstrapper = getattr(module, "bootstrapper", None)
        row = {
            "module_path": str(name),
            "class": type(module).__name__,
            "category": "activation",
            "level": None if getattr(module, "level", None) is None else int(getattr(module, "level")),
            "depth": None if getattr(module, "depth", None) is None else int(getattr(module, "depth")),
            "bootstrapper_path": (
                f"{name}.bootstrapper" if isinstance(bootstrapper, Bootstrap) else ""
            ),
            "call_count": 0,
            "elapsed_s": 0.0,
            "max_call_s": 0.0,
            "last_call_s": 0.0,
        }
        rows_by_id[int(id(module))] = row
        handles.append(module.register_forward_pre_hook(make_pre(row)))
        handles.append(module.register_forward_hook(make_post(row)))

    def snapshot() -> dict[str, Any]:
        rows = sorted(rows_by_id.values(), key=lambda item: str(item["module_path"]))
        active_rows = [dict(row) for row in rows if int(row.get("call_count", 0) or 0) > 0]
        return {
            "enabled": True,
            "profiled_module_count": int(len(rows)),
            "active_module_count": int(len(active_rows)),
            "inclusive_elapsed_s": float(sum(float(row.get("elapsed_s", 0.0)) for row in active_rows)),
            "rows": active_rows,
        }

    def remove() -> None:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    return snapshot, remove


def _install_he_module_profiler(
    net: torch.nn.Module,
    *,
    memory_trace_path: Path | None = None,
) -> tuple[Callable[[], dict[str, Any]], Callable[[], None]]:
    rows_by_id: dict[int, dict[str, Any]] = {}
    stacks: dict[int, list[float]] = {}
    handles: list[Any] = []
    trace_counter = 0

    def append_trace(event: dict[str, Any]) -> None:
        nonlocal trace_counter
        if memory_trace_path is None:
            return
        trace_counter += 1
        event = dict(event)
        event["event_index"] = int(trace_counter)
        event["elapsed_since_trace_start_s"] = float(time.perf_counter() - trace_start)
        event["device_memory"] = _device_memory_snapshot()
        event["live_ciphertexts"] = _live_ciphertext_snapshot()
        event["host_maxrss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        Path(memory_trace_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(memory_trace_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def should_profile(name: str, module: torch.nn.Module) -> bool:
        if not name or not isinstance(module, Module):
            return False
        return True

    def children_count(module: torch.nn.Module) -> int:
        return int(sum(1 for _child in module.children()))

    def has_direct_bootstrapper(module: torch.nn.Module) -> bool:
        return isinstance(getattr(module, "bootstrapper", None), Bootstrap)

    trace_start = time.perf_counter()
    if memory_trace_path is not None:
        Path(memory_trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(memory_trace_path).write_text("", encoding="utf-8")
        append_trace({"phase": "forward", "hook": "trace_start"})

    def make_pre(row: dict[str, Any]):
        @_dynamo_disable
        def pre_hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if not bool(getattr(module, "he_mode", False)):
                return
            key = int(id(module))
            stacks.setdefault(key, []).append(float(time.perf_counter()))
            append_trace(
                {
                    "phase": "forward",
                    "hook": "pre",
                    "module_path": str(row["module_path"]),
                    "class": str(row["class"]),
                    "category": str(row["category"]),
                    "is_leaf": bool(row["is_leaf"]),
                    "region_runtime": bool(row["region_runtime"]),
                    "region_strategy": str(row["region_strategy"]),
                }
            )

        return pre_hook

    def make_post(row: dict[str, Any]):
        @_dynamo_disable
        def post_hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not bool(getattr(module, "he_mode", False)):
                return
            key = int(id(module))
            stack = stacks.get(key) or []
            if not stack:
                return
            elapsed = float(time.perf_counter() - stack.pop())
            row["call_count"] = int(row.get("call_count", 0)) + 1
            row["elapsed_s"] = float(row.get("elapsed_s", 0.0)) + elapsed
            row["max_call_s"] = max(float(row.get("max_call_s", 0.0)), elapsed)
            row["last_call_s"] = elapsed
            append_trace(
                {
                    "phase": "forward",
                    "hook": "post",
                    "module_path": str(row["module_path"]),
                    "class": str(row["class"]),
                    "category": str(row["category"]),
                    "is_leaf": bool(row["is_leaf"]),
                    "region_runtime": bool(row["region_runtime"]),
                    "region_strategy": str(row["region_strategy"]),
                    "call_count": int(row.get("call_count", 0)),
                    "last_call_s": float(elapsed),
                }
            )

        return post_hook

    for name, module in net.named_modules():
        if not should_profile(str(name), module):
            continue
        child_count = children_count(module)
        is_bootstrapper_child = str(name).endswith(".bootstrapper")
        direct_bootstrapper = has_direct_bootstrapper(module)
        row = {
            "module_path": str(name),
            "class": type(module).__name__,
            "category": _module_category(module),
            "level": None if getattr(module, "level", None) is None else int(getattr(module, "level")),
            "depth": None if getattr(module, "depth", None) is None else int(getattr(module, "depth")),
            "children_count": int(child_count),
            "is_leaf": bool(child_count == 0),
            "is_bootstrapper_child": bool(is_bootstrapper_child),
            "has_bootstrapper_child": bool(direct_bootstrapper),
            "bootstrapper_path": f"{name}.bootstrapper" if direct_bootstrapper else "",
            "region_runtime": bool(getattr(module, "region_runtime", None) is not None),
            "region_strategy": str(getattr(getattr(module, "region_runtime", None), "strategy", "")),
            "call_count": 0,
            "elapsed_s": 0.0,
            "max_call_s": 0.0,
            "last_call_s": 0.0,
        }
        rows_by_id[int(id(module))] = row
        handles.append(module.register_forward_pre_hook(make_pre(row)))
        handles.append(module.register_forward_hook(make_post(row)))

    def snapshot() -> dict[str, Any]:
        rows = sorted(rows_by_id.values(), key=lambda item: str(item["module_path"]))
        active_rows = [row for row in rows if int(row.get("call_count", 0)) > 0]
        rows_by_path = {str(row["module_path"]): row for row in active_rows}

        def totals_for(selected_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
            totals: dict[str, dict[str, float | int]] = {}
            for row in selected_rows:
                cat = str(row["category"])
                entry = totals.setdefault(cat, {"elapsed_s": 0.0, "call_count": 0, "module_count": 0})
                entry["elapsed_s"] = float(entry["elapsed_s"]) + float(row.get("elapsed_s", 0.0))
                entry["call_count"] = int(entry["call_count"]) + int(row.get("call_count", 0))
                entry["module_count"] = int(entry["module_count"]) + 1
            return totals

        leaf_rows = [row for row in active_rows if bool(row.get("is_leaf", False))]
        primary_rows = [row for row in active_rows if not bool(row.get("is_bootstrapper_child", False))]
        adjusted_rows: list[dict[str, Any]] = []
        adjusted_totals_by_category: dict[str, dict[str, float | int]] = {}
        for row in primary_rows:
            bootstrap_row = rows_by_path.get(str(row.get("bootstrapper_path", "")))
            bootstrap_elapsed = float(bootstrap_row.get("elapsed_s", 0.0)) if bootstrap_row is not None else 0.0
            core_elapsed = max(0.0, float(row.get("elapsed_s", 0.0)) - bootstrap_elapsed)
            adjusted = {
                "module_path": str(row["module_path"]),
                "class": str(row["class"]),
                "category": str(row["category"]),
                "elapsed_s": float(row.get("elapsed_s", 0.0)),
                "bootstrap_child_s": float(bootstrap_elapsed),
                "core_excluding_bootstrap_s": float(core_elapsed),
                "call_count": int(row.get("call_count", 0)),
                "level": row.get("level"),
                "depth": row.get("depth"),
                "region_runtime": bool(row.get("region_runtime", False)),
                "region_strategy": str(row.get("region_strategy", "")),
            }
            adjusted_rows.append(adjusted)
            cat = str(row["category"])
            entry = adjusted_totals_by_category.setdefault(cat, {"elapsed_s": 0.0, "call_count": 0, "module_count": 0})
            entry["elapsed_s"] = float(entry["elapsed_s"]) + float(core_elapsed)
            entry["call_count"] = int(entry["call_count"]) + int(row.get("call_count", 0))
            entry["module_count"] = int(entry["module_count"]) + 1

        top_by_elapsed = sorted(active_rows, key=lambda item: float(item.get("elapsed_s", 0.0)), reverse=True)[:30]
        top_adjusted_by_core = sorted(
            adjusted_rows,
            key=lambda item: float(item.get("core_excluding_bootstrap_s", 0.0)),
            reverse=True,
        )[:30]
        top_bootstrap_by_parent = sorted(
            [row for row in adjusted_rows if float(row.get("bootstrap_child_s", 0.0)) > 0.0],
            key=lambda item: float(item.get("bootstrap_child_s", 0.0)),
            reverse=True,
        )[:30]
        return {
            "enabled": True,
            "profiled_module_count": int(len(rows)),
            "active_module_count": int(len(active_rows)),
            "notes": [
                "Hooks record wall-time only; ciphertext level/scale are intentionally not queried inside hooks.",
                "primary_inclusive totals include child bootstrap hooks for modules that own a bootstrapper.",
                "primary_adjusted subtracts the direct .bootstrapper child from its parent to estimate module core time.",
                "leaf_totals_by_category is additive but excludes non-leaf parents that own bootstrappers.",
            ],
            "totals_by_category": totals_for(leaf_rows),
            "leaf_totals_by_category": totals_for(leaf_rows),
            "primary_inclusive_totals_by_category": totals_for(primary_rows),
            "primary_adjusted_totals_by_category": adjusted_totals_by_category,
            "top_by_elapsed": [
                {
                    "module_path": str(row["module_path"]),
                    "class": str(row["class"]),
                    "category": str(row["category"]),
                    "elapsed_s": float(row.get("elapsed_s", 0.0)),
                    "children_count": int(row.get("children_count", 0)),
                    "is_leaf": bool(row.get("is_leaf", False)),
                    "has_bootstrapper_child": bool(row.get("has_bootstrapper_child", False)),
                    "bootstrapper_path": str(row.get("bootstrapper_path", "")),
                    "call_count": int(row.get("call_count", 0)),
                    "max_call_s": float(row.get("max_call_s", 0.0)),
                    "level": row.get("level"),
                    "depth": row.get("depth"),
                    "region_runtime": bool(row.get("region_runtime", False)),
                    "region_strategy": str(row.get("region_strategy", "")),
                }
                for row in top_by_elapsed
            ],
            "top_adjusted_by_core": top_adjusted_by_core,
            "top_bootstrap_by_parent": top_bootstrap_by_parent,
            "rows": rows,
            "primary_adjusted_rows": sorted(adjusted_rows, key=lambda item: str(item["module_path"])),
        }

    def remove() -> None:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    return snapshot, remove


def _collect_region_audit(net: torch.nn.Module) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, module in net.named_modules():
        runtime = getattr(module, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        if runtime is None:
            continue
        row = {
            "node": str(getattr(module, "region_output_id", name)),
            "module_path": str(name),
            "stage": str(getattr(runtime, "stage", "")),
            "executable": bool(getattr(runtime, "executable", False)),
            "compile_count": int(getattr(executor, "compile_count", 0)) if executor is not None else 0,
            "execute_count": int(getattr(runtime, "execute_count", 0)),
            "strategy": str(getattr(runtime, "strategy", "")),
            "fallback_reason": str(getattr(runtime, "fallback_reason", "")),
        }
        if executor is not None:
            row["last_runtime_timing"] = dict(getattr(executor, "last_runtime_timing", {}) or {})
            row["last_runtime_counts"] = dict(getattr(executor, "last_runtime_counts", {}) or {})
            row["last_runtime_io"] = dict(getattr(executor, "last_runtime_io", {}) or {})
        rows.append(row)
    return {
        "selected_region_count": int(len(rows)),
        "executable_region_count": int(sum(1 for row in rows if bool(row["executable"]))),
        "rows": rows,
    }


def _iter_runtime_executors(net: torch.nn.Module):
    seen: set[int] = set()
    for module_name, module in net.named_modules():
        roots = []
        runtime = getattr(module, "region_runtime", None)
        executor = getattr(runtime, "executor", None) if runtime is not None else None
        if executor is not None:
            roots.append(executor)
        for attr in ("layout_policy_add_runtime", "layout_policy_concat_runtime"):
            candidate = getattr(module, attr, None)
            if candidate is not None:
                roots.append(candidate)
        for root in roots:
            for candidate in _walk_executor_objects(root):
                if id(candidate) in seen:
                    continue
                seen.add(id(candidate))
                yield str(module_name), module, candidate


def _iter_unified_groups(executor: Any):
    seen: set[int] = set()

    def emit(value: Any):
        if value is None:
            return
        if isinstance(value, dict):
            iterable = list(value.values())
        elif isinstance(value, (list, tuple, set)):
            iterable = list(value)
        else:
            iterable = [value]
        for item in iterable:
            if isinstance(item, (dict, list, tuple, set)):
                yield from emit(item)
                continue
            nested_group = getattr(item, "group", None)
            if nested_group is not None and not hasattr(item, "last_runtime_timing"):
                yield from emit(nested_group)
                continue
            if item is None or not hasattr(item, "last_runtime_timing"):
                continue
            if id(item) in seen:
                continue
            seen.add(id(item))
            yield item

    for attr in (
        "group",
        "groups",
        "groups_by_input_block",
        "groups_by_pair",
        "groups_by_input",
        "groups_by_input_chunk",
        "groups_by_input_index",
        "groups_by_source",
        "runtime_groups",
        "_concat_unified_groups_by_input",
    ):
        yield from emit(getattr(executor, attr, None))


def _collect_bootstrap_runtime_breakdown(bootstrap_report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals = {
        "bootstrap_s": 0.0,
        "backend_bootstrap_s": 0.0,
        "preprocess_s": 0.0,
        "postprocess_s": 0.0,
        "call_count": 0,
    }
    by_name: dict[str, float] = {}
    for row in bootstrap_report.get("rows", []) or []:
        row = dict(row)
        name = str(row.get("name", ""))
        profiles = list(row.get("runtime_profile", []) or [])
        row_total = 0.0
        row_backend = 0.0
        row_pre = 0.0
        row_post = 0.0
        for profile in profiles:
            timing = dict(dict(profile).get("timing_s", {}) or {})
            total = _timing_float(timing, "forward_total_inner")
            backend = _timing_float(timing, "backend_bootstrap_call")
            pre = _timing_float(timing, "preprocess_total")
            post = _timing_float(timing, "postprocess_total")
            if total <= 0.0:
                total = float(pre + backend + post)
            row_total += float(total)
            row_backend += float(backend)
            row_pre += float(pre)
            row_post += float(post)
        by_name[str(name)] = float(by_name.get(str(name), 0.0) + row_total)
        rows.append(
            {
                "name": str(name),
                "bootstrap_slots": int(row.get("bootstrap_slots", 0) or 0),
                "runtime_call_count": int(len(profiles)),
                "bootstrap_s": float(row_total),
                "backend_bootstrap_s": float(row_backend),
                "preprocess_s": float(row_pre),
                "postprocess_s": float(row_post),
            }
        )
        totals["bootstrap_s"] += float(row_total)
        totals["backend_bootstrap_s"] += float(row_backend)
        totals["preprocess_s"] += float(row_pre)
        totals["postprocess_s"] += float(row_post)
        totals["call_count"] += int(len(profiles))
    return {
        "totals": totals,
        "by_name": by_name,
        "rows": rows,
    }


def _collect_activation_runtime_breakdown(
    activation_profile: dict[str, Any] | None,
    bootstrap_breakdown: dict[str, Any],
) -> dict[str, Any]:
    profile = dict(activation_profile or {})
    boot_by_name = dict(bootstrap_breakdown.get("by_name", {}) or {})
    rows: list[dict[str, Any]] = []
    inclusive_s = 0.0
    bootstrap_child_s = 0.0
    activation_s = 0.0
    call_count = 0
    for row in profile.get("rows", []) or []:
        row = dict(row)
        elapsed = float(row.get("elapsed_s", 0.0) or 0.0)
        child_path = str(row.get("bootstrapper_path", ""))
        child_bootstrap = float(boot_by_name.get(child_path, 0.0) or 0.0)
        core = max(0.0, float(elapsed - child_bootstrap))
        row["bootstrap_child_s"] = float(child_bootstrap)
        row["activation_excluding_bootstrap_s"] = float(core)
        rows.append(row)
        inclusive_s += float(elapsed)
        bootstrap_child_s += float(child_bootstrap)
        activation_s += float(core)
        call_count += int(row.get("call_count", 0) or 0)
    rows.sort(key=lambda item: float(item.get("activation_excluding_bootstrap_s", 0.0)), reverse=True)
    return {
        "enabled": bool(profile.get("enabled", False)),
        "profiled_module_count": int(profile.get("profiled_module_count", 0) or 0),
        "active_module_count": int(profile.get("active_module_count", 0) or 0),
        "totals": {
            "activation_s": float(activation_s),
            "activation_inclusive_s": float(inclusive_s),
            "bootstrap_child_s": float(bootstrap_child_s),
            "call_count": int(call_count),
        },
        "top_rows": rows[:40],
        "rows": sorted(rows, key=lambda item: str(item.get("module_path", ""))),
    }


def _compile_profile_float(profile: dict[str, Any], key: str) -> float:
    return _timing_float(profile, key)


def _collect_compile_operator_breakdown(
    net: torch.nn.Module,
    *,
    compile_s: float | None = None,
    compile_load_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    executor_rows: list[dict[str, Any]] = []
    seen_groups: set[int] = set()
    totals: dict[str, Any] = {
        "compile_s": None if compile_s is None else float(compile_s),
        "group_count": 0,
        "transform_count": 0,
        "payload_bytes": 0,
        "group_total_s": 0.0,
        "group_flatten_s": 0.0,
        "group_backend_generate_s": 0.0,
        "group_rotation_key_compile_s": 0.0,
        "group_record_keys_s": 0.0,
        "group_save_unload_s": 0.0,
        "group_compile_gc_s": 0.0,
        "executor_prepare_s": 0.0,
        "executor_compile_unified_s": 0.0,
        "executor_build_transform_s": 0.0,
        "executor_group_compile_s": 0.0,
    }
    for module_name, module, executor in _iter_runtime_executors(net):
        executor_timing = dict(getattr(executor, "last_runtime_timing", {}) or {})
        if executor_timing:
            row = {
                "module_path": str(module_name),
                "node": str(getattr(module, "region_output_id", module_name)),
                "executor": type(executor).__name__,
                "timing": executor_timing,
            }
            executor_rows.append(row)
            totals["executor_prepare_s"] += float(
                _timing_float(executor_timing, "prepare_plans_s")
                + _timing_float(executor_timing, "prepare_transforms_s")
            )
            totals["executor_compile_unified_s"] += _timing_float(executor_timing, "compile_unified_s")
            totals["executor_build_transform_s"] += _timing_float(executor_timing, "build_transform_s")
            totals["executor_group_compile_s"] += _timing_float(executor_timing, "group_compile_s")
        for group in _iter_unified_groups(executor):
            if id(group) in seen_groups:
                continue
            seen_groups.add(id(group))
            profile = dict(getattr(group, "last_compile_profile", {}) or {})
            if not profile:
                continue
            row = {
                "module_path": str(module_name),
                "node": str(getattr(module, "region_output_id", module_name)),
                "executor": type(executor).__name__,
                "storage_key": str(getattr(group, "_storage_key", "")),
                "profile": profile,
            }
            group_rows.append(row)
            totals["group_count"] += 1
            totals["transform_count"] += int(profile.get("transform_count", 0) or 0)
            totals["payload_bytes"] += int(profile.get("payload_bytes", 0) or 0)
            for key, total_key in (
                ("total_s", "group_total_s"),
                ("flatten_s", "group_flatten_s"),
                ("backend_generate_s", "group_backend_generate_s"),
                ("rotation_key_compile_s", "group_rotation_key_compile_s"),
                ("record_keys_s", "group_record_keys_s"),
                ("save_unload_s", "group_save_unload_s"),
                ("compile_gc_s", "group_compile_gc_s"),
            ):
                totals[total_key] += _compile_profile_float(profile, key)
    return {
        "totals": totals,
        "compile_load_profile": dict(compile_load_profile or {}),
        "group_rows": group_rows,
        "executor_rows": executor_rows,
    }


def _group_mvm_kernel_s(timing: dict[str, Any]) -> float:
    if str(timing.get("runtime_fairness_mode", "") or "") == "linear_wrapper_only":
        return 0.0
    kernel = (
        _timing_float(timing, "stream_eval_s")
        + _timing_float(timing, "stream_accumulate_s")
        + _timing_float(timing, "cpp_baby_step_s")
        + _timing_float(timing, "cpp_giant_step_s")
    )
    if kernel > 0.0:
        return float(kernel)
    return float(
        _timing_float(timing, "eval_s")
        or _timing_float(timing, "eval_total_s")
        or _timing_float(timing, "serving_hot_s")
    )


def _group_load_encode_s(timing: dict[str, Any]) -> float:
    return float(
        _timing_float(timing, "read_bundle_s")
        + _timing_float(timing, "load_keys_s")
        + _timing_float(timing, "load_plaintexts_s")
        + _timing_float(timing, "stream_build_map_s")
        + _timing_float(timing, "stream_encode_hoist_s")
        + _timing_float(timing, "stream_load_payload_s")
    )


def _executor_group_eval_wall_s(timing: dict[str, Any]) -> float:
    # Native provider group_eval_s wraps UnifiedTransformGroup.evaluate_unified().
    # In single-slot mode that call also includes materialize/encode/evict wall
    # time, so it is diagnostic wall time only.  MVM is collected from the
    # groups' own eval_s/stream_* timing via _iter_unified_groups().
    return float(_timing_float(timing, "group_eval_s"))


def _executor_timing_signature(
    *,
    module_name: str,
    node: str,
    timing: dict[str, Any],
    counts: dict[str, Any],
) -> tuple[Any, ...]:
    del counts
    return _executor_runtime_core_signature(
        module_name=str(module_name),
        node=str(node),
        timing=timing,
    )


def _collect_mvm_runtime_breakdown(net: torch.nn.Module) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    executor_timing_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    seen_groups: set[int] = set()
    totals = {
        "group_count": 0,
        "executor_count": 0,
        "mvm_kernel_s": 0.0,
        "mvm_eval_total_s": 0.0,
        "executor_group_eval_wall_s": 0.0,
        "executor_evaluate_unified_s": 0.0,
        "lt_runtime_load_encode_s": 0.0,
        "lt_layer_cache_turnover_s": 0.0,
        "lt_layer_cache_encode_s": 0.0,
        "lt_layer_cache_key_prepare_s": 0.0,
        "lt_layer_cache_evict_s": 0.0,
        "lt_runtime_read_bundle_s": 0.0,
        "lt_runtime_load_keys_s": 0.0,
        "lt_runtime_load_plaintexts_s": 0.0,
        "lt_runtime_stream_build_map_s": 0.0,
        "lt_runtime_stream_encode_hoist_s": 0.0,
        "lt_runtime_stream_load_payload_s": 0.0,
        "lt_runtime_stream_eval_s": 0.0,
        "lt_runtime_stream_accumulate_s": 0.0,
        "lt_runtime_trim_unload_s": 0.0,
        "linear_wrapper_accumulate_s": 0.0,
        "linear_wrapper_rescale_s": 0.0,
        "linear_wrapper_bias_s": 0.0,
        "linear_wrapper_output_rotation_s": 0.0,
        "linear_wrapper_postprocess_s": 0.0,
        "executor_wrap_s": 0.0,
        "executor_postprocess_s": 0.0,
        "executor_rescale_s": 0.0,
        "executor_accumulate_s": 0.0,
    }

    def add_timing_row(
        *,
        module_name: str,
        module: torch.nn.Module,
        executor_label: str,
        storage_key: str,
        transform_count: int,
        timing: dict[str, Any],
    ) -> None:
        serving = _timing_float(timing, "serving_hot_s")
        wrapper_only = str(timing.get("runtime_fairness_mode", "") or "") == "linear_wrapper_only"
        eval_total = (
            _timing_float(timing, "eval_total_s")
            or _timing_float(timing, "eval_s")
            or (0.0 if wrapper_only else serving)
        )
        mvm_kernel = _group_mvm_kernel_s(timing)
        load_encode = _group_load_encode_s(timing)
        trim_unload = float(
            _timing_float(timing, "unload_s")
            + _timing_float(timing, "trim_s")
            + _timing_float(timing, "cpp_trim_s")
        )
        layer_cache_turnover = float(
            _timing_float(timing, "layer_cache_turnover_s")
            or (
                _timing_float(timing, "layer_cache_encode_s")
                + _timing_float(timing, "layer_cache_key_prepare_s")
                + _timing_float(timing, "layer_cache_evict_s")
            )
        )
        linear_wrapper_accumulate = _timing_float(timing, "linear_wrapper_accumulate_s")
        linear_wrapper_rescale = _timing_float(timing, "linear_wrapper_rescale_s")
        linear_wrapper_bias = _timing_float(timing, "linear_wrapper_bias_s")
        linear_wrapper_output_rotation = _timing_float(timing, "linear_wrapper_output_rotation_s")
        linear_wrapper_postprocess = float(
            _timing_float(timing, "linear_wrapper_postprocess_s")
            or (
                linear_wrapper_accumulate
                + linear_wrapper_rescale
                + linear_wrapper_bias
                + linear_wrapper_output_rotation
            )
        )
        if (
            serving <= 0.0
            and eval_total <= 0.0
            and load_encode <= 0.0
            and trim_unload <= 0.0
            and layer_cache_turnover <= 0.0
            and linear_wrapper_postprocess <= 0.0
        ):
            return
        row = {
            "module_path": str(module_name),
            "node": str(getattr(module, "region_output_id", module_name)),
            "executor": str(executor_label),
            "storage_key": str(storage_key),
            "transform_count": int(transform_count),
            "runtime_fairness_mode": str(timing.get("runtime_fairness_mode", "unknown")),
            "mvm_kernel_s": float(mvm_kernel),
            "mvm_eval_total_s": float(eval_total),
            "lt_runtime_load_encode_s": float(load_encode),
            "lt_layer_cache_turnover_s": float(layer_cache_turnover),
            "lt_layer_cache_encode_s": _timing_float(timing, "layer_cache_encode_s"),
            "lt_layer_cache_key_prepare_s": _timing_float(timing, "layer_cache_key_prepare_s"),
            "lt_layer_cache_evict_s": _timing_float(timing, "layer_cache_evict_s"),
            "lt_runtime_trim_unload_s": float(trim_unload),
            "linear_wrapper_accumulate_s": float(linear_wrapper_accumulate),
            "linear_wrapper_rescale_s": float(linear_wrapper_rescale),
            "linear_wrapper_bias_s": float(linear_wrapper_bias),
            "linear_wrapper_output_rotation_s": float(linear_wrapper_output_rotation),
            "linear_wrapper_postprocess_s": float(linear_wrapper_postprocess),
            "timing": dict(timing),
        }
        group_rows.append(row)
        totals["group_count"] += 1
        totals["mvm_kernel_s"] += float(mvm_kernel)
        totals["mvm_eval_total_s"] += float(eval_total)
        totals["lt_runtime_load_encode_s"] += float(load_encode)
        totals["lt_layer_cache_turnover_s"] += float(layer_cache_turnover)
        totals["lt_layer_cache_encode_s"] += _timing_float(timing, "layer_cache_encode_s")
        totals["lt_layer_cache_key_prepare_s"] += _timing_float(timing, "layer_cache_key_prepare_s")
        totals["lt_layer_cache_evict_s"] += _timing_float(timing, "layer_cache_evict_s")
        totals["lt_runtime_read_bundle_s"] += _timing_float(timing, "read_bundle_s")
        totals["lt_runtime_load_keys_s"] += _timing_float(timing, "load_keys_s")
        totals["lt_runtime_load_plaintexts_s"] += _timing_float(timing, "load_plaintexts_s")
        totals["lt_runtime_stream_build_map_s"] += _timing_float(timing, "stream_build_map_s")
        totals["lt_runtime_stream_encode_hoist_s"] += _timing_float(timing, "stream_encode_hoist_s")
        totals["lt_runtime_stream_load_payload_s"] += _timing_float(timing, "stream_load_payload_s")
        totals["lt_runtime_stream_eval_s"] += _timing_float(timing, "stream_eval_s")
        totals["lt_runtime_stream_accumulate_s"] += _timing_float(timing, "stream_accumulate_s")
        totals["lt_runtime_trim_unload_s"] += float(trim_unload)
        totals["linear_wrapper_accumulate_s"] += float(linear_wrapper_accumulate)
        totals["linear_wrapper_rescale_s"] += float(linear_wrapper_rescale)
        totals["linear_wrapper_bias_s"] += float(linear_wrapper_bias)
        totals["linear_wrapper_output_rotation_s"] += float(linear_wrapper_output_rotation)
        totals["linear_wrapper_postprocess_s"] += float(linear_wrapper_postprocess)

    for module_name, module in net.named_modules():
        layer_timing = dict(getattr(module, "_last_runtime_timing", {}) or {})
        if layer_timing:
            add_timing_row(
                module_name=str(module_name),
                module=module,
                executor_label=f"{type(module).__name__}._last_runtime_timing",
                storage_key="",
                transform_count=0,
                timing=layer_timing,
            )
        for proxy_index, proxy in enumerate(list(getattr(module, "_concat_transform_sources_by_input", []) or [])):
            proxy_timing = dict(getattr(proxy, "_last_runtime_timing", {}) or {})
            if not proxy_timing:
                continue
            add_timing_row(
                module_name=str(module_name),
                module=module,
                executor_label=f"{type(module).__name__}.concat_source[{int(proxy_index)}]",
                storage_key=str(getattr(proxy, "name", "")),
                transform_count=int(len(getattr(proxy, "transform_ids", {}) or {})),
                timing=proxy_timing,
            )
        for group in _iter_unified_groups(module):
            if id(group) in seen_groups:
                continue
            seen_groups.add(id(group))
            timing = dict(getattr(group, "last_runtime_timing", {}) or {})
            if not timing:
                continue
            add_timing_row(
                module_name=str(module_name),
                module=module,
                executor_label=f"{type(module).__name__}.module_group",
                storage_key=str(getattr(group, "_storage_key", "")),
                transform_count=int(len(getattr(group, "unified_ids", []) or [])),
                timing=timing,
            )

    for module_name, module, executor in _iter_runtime_executors(net):
        executor_timing = dict(getattr(executor, "last_runtime_timing", {}) or {})
        if executor_timing:
            node = str(getattr(module, "region_output_id", module_name))
            counts = dict(getattr(executor, "last_runtime_counts", {}) or {})
            signature = _executor_timing_signature(
                module_name=str(module_name),
                node=node,
                timing=executor_timing,
                counts=counts,
            )
            executor_group_eval_wall = _executor_group_eval_wall_s(executor_timing)
            executor_evaluate_unified = _timing_float(executor_timing, "evaluate_unified_s")
            executor_components = _executor_runtime_overhead_components(executor_timing)
            row_update = {
                "module_path": str(module_name),
                "node": node,
                "executor": type(executor).__name__,
                "executor_group_eval_wall_s": float(executor_group_eval_wall),
                "executor_evaluate_unified_s": float(executor_evaluate_unified),
                "executor_wrap_s": float(executor_components["executor_wrap_s"]),
                "executor_postprocess_s": float(executor_components["executor_postprocess_s"]),
                "executor_rescale_s": float(executor_components["executor_rescale_s"]),
                "executor_accumulate_s": float(executor_components["executor_accumulate_s"]),
                "executor_overhead_s": float(executor_components["executor_overhead_s"]),
                "timing": executor_timing,
                "counts": counts,
            }
            if signature in executor_timing_rows:
                existing = executor_timing_rows[signature]
                merged_timing = _merge_numeric_max(dict(existing.get("timing", {}) or {}), executor_timing)
                merged_counts = _merge_numeric_max(dict(existing.get("counts", {}) or {}), counts)
                merged_components = _executor_runtime_overhead_components(merged_timing)
                existing.update(
                    {
                        "executor": f"{existing.get('executor', type(executor).__name__)}+{type(executor).__name__}",
                        "executor_group_eval_wall_s": max(
                            float(existing.get("executor_group_eval_wall_s", 0.0) or 0.0),
                            float(_executor_group_eval_wall_s(merged_timing)),
                        ),
                        "executor_evaluate_unified_s": max(
                            float(existing.get("executor_evaluate_unified_s", 0.0) or 0.0),
                            float(_timing_float(merged_timing, "evaluate_unified_s")),
                        ),
                        "executor_wrap_s": float(merged_components["executor_wrap_s"]),
                        "executor_postprocess_s": float(merged_components["executor_postprocess_s"]),
                        "executor_rescale_s": float(merged_components["executor_rescale_s"]),
                        "executor_accumulate_s": float(merged_components["executor_accumulate_s"]),
                        "executor_overhead_s": float(merged_components["executor_overhead_s"]),
                        "timing": merged_timing,
                        "counts": merged_counts,
                    }
                )
            else:
                executor_timing_rows[signature] = row_update
        for group in _iter_unified_groups(executor):
            if id(group) in seen_groups:
                continue
            seen_groups.add(id(group))
            timing = dict(getattr(group, "last_runtime_timing", {}) or {})
            if not timing:
                continue
            add_timing_row(
                module_name=str(module_name),
                module=module,
                executor_label=type(executor).__name__,
                storage_key=str(getattr(group, "_storage_key", "")),
                transform_count=int(len(getattr(group, "unified_ids", []) or [])),
                timing=timing,
            )
    group_rows.sort(key=lambda item: float(item.get("mvm_eval_total_s", 0.0)), reverse=True)
    executor_rows = list(executor_timing_rows.values())
    executor_rows.sort(key=lambda item: float(item.get("executor_overhead_s", 0.0)), reverse=True)
    totals["executor_count"] = int(len(executor_rows))
    for row in executor_rows:
        totals["executor_group_eval_wall_s"] += float(row.get("executor_group_eval_wall_s", 0.0) or 0.0)
        totals["executor_evaluate_unified_s"] += float(row.get("executor_evaluate_unified_s", 0.0) or 0.0)
        totals["executor_wrap_s"] += float(row.get("executor_wrap_s", 0.0) or 0.0)
        totals["executor_postprocess_s"] += float(row.get("executor_postprocess_s", 0.0) or 0.0)
        totals["executor_rescale_s"] += float(row.get("executor_rescale_s", 0.0) or 0.0)
        totals["executor_accumulate_s"] += float(row.get("executor_accumulate_s", 0.0) or 0.0)
    return {
        "totals": totals,
        "top_groups": group_rows[:60],
        "group_rows": group_rows,
        "executor_rows": executor_rows,
    }


def _collect_he_module_wall_breakdown(
    module_profile: dict[str, Any] | None,
    *,
    activation: dict[str, Any],
    bootstrap: dict[str, Any],
    mvm: dict[str, Any],
) -> dict[str, Any]:
    profile = dict(module_profile or {})
    profile_rows = list(profile.get("primary_adjusted_rows", []) or [])
    if not profile_rows:
        profile_rows = [
            row
            for row in list(profile.get("rows", []) or [])
            if not bool(row.get("is_bootstrapper_child", False))
        ]
    known_by_module: dict[str, dict[str, float]] = {}

    def known_entry(module_path: str) -> dict[str, float]:
        return known_by_module.setdefault(
            str(module_path),
            {
                "mvm_kernel_s": 0.0,
                "activation_s": 0.0,
                "bootstrap_s": 0.0,
                "layer_cache_turnover_s": 0.0,
                "executor_overhead_s": 0.0,
                "linear_wrapper_postprocess_s": 0.0,
                "runtime_load_trim_s": 0.0,
            },
        )

    for row in list(mvm.get("group_rows", []) or []):
        if not isinstance(row, dict):
            continue
        entry = known_entry(str(row.get("module_path", "")))
        entry["mvm_kernel_s"] += float(row.get("mvm_kernel_s", 0.0) or 0.0)
        entry["layer_cache_turnover_s"] += float(row.get("lt_layer_cache_turnover_s", 0.0) or 0.0)
        entry["linear_wrapper_postprocess_s"] += float(row.get("linear_wrapper_postprocess_s", 0.0) or 0.0)
        entry["runtime_load_trim_s"] += float(row.get("lt_runtime_load_encode_s", 0.0) or 0.0) + float(
            row.get("lt_runtime_trim_unload_s", 0.0) or 0.0
        )
    for row in list(mvm.get("executor_rows", []) or []):
        if not isinstance(row, dict):
            continue
        entry = known_entry(str(row.get("module_path", "")))
        entry["mvm_kernel_s"] += float(row.get("mvm_kernel_s", 0.0) or 0.0)
        entry["executor_overhead_s"] += float(row.get("executor_overhead_s", 0.0) or 0.0)
    for row in list(activation.get("rows", []) or []):
        if not isinstance(row, dict):
            continue
        entry = known_entry(str(row.get("module_path", "")))
        entry["activation_s"] += float(row.get("activation_excluding_bootstrap_s", 0.0) or 0.0)
    for row in list(bootstrap.get("rows", []) or []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        module_path = name[: -len(".bootstrapper")] if name.endswith(".bootstrapper") else name
        entry = known_entry(module_path)
        entry["bootstrap_s"] += float(row.get("bootstrap_s", 0.0) or 0.0)

    rows: list[dict[str, Any]] = []
    totals_by_category: dict[str, dict[str, Any]] = {}
    totals = {
        "module_wall_s": 0.0,
        "mvm_kernel_s": 0.0,
        "activation_s": 0.0,
        "bootstrap_s": 0.0,
        "layer_cache_turnover_s": 0.0,
        "executor_overhead_s": 0.0,
        "linear_wrapper_postprocess_s": 0.0,
        "runtime_load_trim_s": 0.0,
        "residual_after_known_subtimers_s": 0.0,
        "call_count": 0,
        "module_count": 0,
    }
    for profile_row in profile_rows:
        if not isinstance(profile_row, dict):
            continue
        module_path = str(profile_row.get("module_path", ""))
        if not module_path:
            continue
        module_wall = float(profile_row.get("elapsed_s", 0.0) or 0.0)
        if module_wall <= 0.0:
            continue
        known = known_by_module.get(module_path, {})
        mvm_kernel = float(known.get("mvm_kernel_s", 0.0) or 0.0)
        activation_s = float(known.get("activation_s", 0.0) or 0.0)
        bootstrap_s = float(known.get("bootstrap_s", 0.0) or 0.0)
        layer_cache_turnover = float(known.get("layer_cache_turnover_s", 0.0) or 0.0)
        executor_overhead = float(known.get("executor_overhead_s", 0.0) or 0.0)
        linear_wrapper_postprocess = float(known.get("linear_wrapper_postprocess_s", 0.0) or 0.0)
        runtime_load_trim = float(known.get("runtime_load_trim_s", 0.0) or 0.0)
        known = float(
            mvm_kernel
            + activation_s
            + bootstrap_s
            + layer_cache_turnover
            + executor_overhead
            + linear_wrapper_postprocess
            + runtime_load_trim
        )
        residual = float(module_wall - known)
        category = str(profile_row.get("category", "other"))
        call_count = int(profile_row.get("call_count", 0) or 0)
        row = {
            "module_path": str(module_path),
            "class": str(profile_row.get("class", "")),
            "category": str(category),
            "call_count": int(call_count),
            "module_wall_s": float(module_wall),
            "mvm_kernel_s": float(mvm_kernel),
            "activation_s": float(activation_s),
            "bootstrap_s": float(bootstrap_s),
            "layer_cache_turnover_s": float(layer_cache_turnover),
            "executor_overhead_s": float(executor_overhead),
            "linear_wrapper_postprocess_s": float(linear_wrapper_postprocess),
            "runtime_load_trim_s": float(runtime_load_trim),
            "known_subtimers_s": float(known),
            "residual_after_known_subtimers_s": float(residual),
        }
        rows.append(row)
        totals["module_wall_s"] += float(module_wall)
        totals["mvm_kernel_s"] += float(mvm_kernel)
        totals["activation_s"] += float(activation_s)
        totals["bootstrap_s"] += float(bootstrap_s)
        totals["layer_cache_turnover_s"] += float(layer_cache_turnover)
        totals["executor_overhead_s"] += float(executor_overhead)
        totals["linear_wrapper_postprocess_s"] += float(linear_wrapper_postprocess)
        totals["runtime_load_trim_s"] += float(runtime_load_trim)
        totals["residual_after_known_subtimers_s"] += float(residual)
        totals["call_count"] += int(call_count)
        totals["module_count"] += 1
        category_totals = totals_by_category.setdefault(
            str(category),
            {
                "module_wall_s": 0.0,
                "mvm_kernel_s": 0.0,
                "activation_s": 0.0,
                "bootstrap_s": 0.0,
                "layer_cache_turnover_s": 0.0,
                "executor_overhead_s": 0.0,
                "linear_wrapper_postprocess_s": 0.0,
                "runtime_load_trim_s": 0.0,
                "residual_after_known_subtimers_s": 0.0,
                "call_count": 0,
                "module_count": 0,
            },
        )
        for key in (
            "module_wall_s",
            "mvm_kernel_s",
            "activation_s",
            "bootstrap_s",
            "layer_cache_turnover_s",
            "executor_overhead_s",
            "linear_wrapper_postprocess_s",
            "runtime_load_trim_s",
            "residual_after_known_subtimers_s",
        ):
            category_totals[key] = float(category_totals[key]) + float(row[key])
        category_totals["call_count"] = int(category_totals["call_count"]) + int(call_count)
        category_totals["module_count"] = int(category_totals["module_count"]) + 1
    rows.sort(key=lambda item: float(item.get("residual_after_known_subtimers_s", 0.0)), reverse=True)
    return {
        "enabled": bool(profile.get("enabled", False)),
        "notes": [
            "This hook times each Orion Module during HE forward and subtracts known MVM, activation, bootstrap, layer-cache, executor, and runtime load/trim subtimers.",
            "Residual by module identifies where remaining he_forward wall time lives; it is diagnostic wall-time, not an additive compute comparison metric.",
        ],
        "totals": totals,
        "totals_by_category": totals_by_category,
        "top_residual_rows": rows[:40],
        "rows": sorted(rows, key=lambda item: str(item["module_path"])),
    }


def _collect_forward_operator_breakdown(
    net: torch.nn.Module,
    *,
    he_forward_s: float | None,
    activation_profile: dict[str, Any] | None,
    module_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bootstrap_report = _collect_bootstrap_report(net)
    bootstrap = _collect_bootstrap_runtime_breakdown(bootstrap_report)
    activation = _collect_activation_runtime_breakdown(activation_profile, bootstrap)
    mvm = _collect_mvm_runtime_breakdown(net)
    module_wall = _collect_he_module_wall_breakdown(
        module_profile,
        activation=activation,
        bootstrap=bootstrap,
        mvm=mvm,
    )
    he_forward_value = None if he_forward_s is None else float(he_forward_s)
    totals = {
        "he_forward_s": he_forward_value,
        "mvm_kernel_s": float(mvm["totals"].get("mvm_kernel_s", 0.0)),
        "mvm_eval_total_s": float(mvm["totals"].get("mvm_eval_total_s", 0.0)),
        "activation_s": float(activation["totals"].get("activation_s", 0.0)),
        "activation_inclusive_s": float(activation["totals"].get("activation_inclusive_s", 0.0)),
        "bootstrap_s": float(bootstrap["totals"].get("bootstrap_s", 0.0)),
        "bootstrap_backend_s": float(bootstrap["totals"].get("backend_bootstrap_s", 0.0)),
        "lt_runtime_load_encode_s": float(mvm["totals"].get("lt_runtime_load_encode_s", 0.0)),
        "lt_layer_cache_turnover_s": float(mvm["totals"].get("lt_layer_cache_turnover_s", 0.0)),
        "lt_layer_cache_encode_s": float(mvm["totals"].get("lt_layer_cache_encode_s", 0.0)),
        "lt_layer_cache_key_prepare_s": float(mvm["totals"].get("lt_layer_cache_key_prepare_s", 0.0)),
        "lt_layer_cache_evict_s": float(mvm["totals"].get("lt_layer_cache_evict_s", 0.0)),
        "lt_runtime_stream_build_map_s": float(mvm["totals"].get("lt_runtime_stream_build_map_s", 0.0)),
        "lt_runtime_stream_encode_hoist_s": float(mvm["totals"].get("lt_runtime_stream_encode_hoist_s", 0.0)),
        "lt_runtime_stream_load_payload_s": float(mvm["totals"].get("lt_runtime_stream_load_payload_s", 0.0)),
        "lt_runtime_stream_eval_s": float(mvm["totals"].get("lt_runtime_stream_eval_s", 0.0)),
        "lt_runtime_stream_accumulate_s": float(mvm["totals"].get("lt_runtime_stream_accumulate_s", 0.0)),
        "lt_runtime_trim_unload_s": float(mvm["totals"].get("lt_runtime_trim_unload_s", 0.0)),
        "linear_wrapper_accumulate_s": float(mvm["totals"].get("linear_wrapper_accumulate_s", 0.0)),
        "linear_wrapper_rescale_s": float(mvm["totals"].get("linear_wrapper_rescale_s", 0.0)),
        "linear_wrapper_bias_s": float(mvm["totals"].get("linear_wrapper_bias_s", 0.0)),
        "linear_wrapper_output_rotation_s": float(mvm["totals"].get("linear_wrapper_output_rotation_s", 0.0)),
        "linear_wrapper_postprocess_s": float(mvm["totals"].get("linear_wrapper_postprocess_s", 0.0)),
        "executor_wrap_s": float(mvm["totals"].get("executor_wrap_s", 0.0)),
        "executor_postprocess_s": float(mvm["totals"].get("executor_postprocess_s", 0.0)),
        "executor_rescale_s": float(mvm["totals"].get("executor_rescale_s", 0.0)),
        "executor_accumulate_s": float(mvm["totals"].get("executor_accumulate_s", 0.0)),
        "executor_group_eval_wall_s": float(mvm["totals"].get("executor_group_eval_wall_s", 0.0)),
        "executor_evaluate_unified_s": float(mvm["totals"].get("executor_evaluate_unified_s", 0.0)),
    }
    compute_accounted = float(
        totals["mvm_kernel_s"]
        + totals["activation_s"]
        + totals["bootstrap_s"]
    )
    runtime_load_trim = float(
        totals["lt_runtime_load_encode_s"] + totals["lt_runtime_trim_unload_s"]
    )
    executor_overhead = float(
        totals["executor_wrap_s"]
        + totals["executor_postprocess_s"]
        + totals["executor_rescale_s"]
        + totals["executor_accumulate_s"]
    )
    layer_cache_turnover = float(totals["lt_layer_cache_turnover_s"])
    linear_wrapper_postprocess = float(totals["linear_wrapper_postprocess_s"])
    wall_aux_accounted = float(
        layer_cache_turnover
        + runtime_load_trim
        + linear_wrapper_postprocess
        + executor_overhead
    )
    wall_accounted = float(compute_accounted + wall_aux_accounted)
    totals["compute_mvm_activation_bootstrap_s"] = float(compute_accounted)
    totals["compute_accounted_s"] = float(compute_accounted)
    totals["he_forward_minus_compute_s"] = (
        None if he_forward_value is None else float(he_forward_value - compute_accounted)
    )
    totals["wall_layer_cache_turnover_s"] = float(layer_cache_turnover)
    totals["wall_runtime_load_trim_s"] = float(runtime_load_trim)
    totals["wall_executor_overhead_s"] = float(executor_overhead)
    totals["wall_linear_wrapper_postprocess_s"] = float(linear_wrapper_postprocess)
    totals["wall_aux_accounted_s"] = float(wall_aux_accounted)
    totals["wall_accounted_s"] = float(wall_accounted)
    wall_residual = None if he_forward_value is None else float(he_forward_value - wall_accounted)
    totals["wall_unattributed_he_forward_s"] = wall_residual
    totals["unattributed_he_forward_s"] = wall_residual
    legacy_accounted = float(
        totals["mvm_kernel_s"]
        + totals["activation_s"]
        + totals["bootstrap_s"]
        + totals["lt_runtime_load_encode_s"]
        + totals["lt_runtime_trim_unload_s"]
        + totals["linear_wrapper_postprocess_s"]
        + executor_overhead
    )
    totals["legacy_accounted_without_layer_cache_s"] = float(legacy_accounted)
    totals["legacy_unattributed_without_layer_cache_s"] = (
        None if he_forward_value is None else float(he_forward_value - legacy_accounted)
    )
    return {
        "notes": [
            "MVM is collected from UnifiedTransformGroup runtime timing and is streaming-safe, including provider runtime groups.",
            "mvm_kernel_s uses stream_eval+stream_accumulate or baby+giant-step counters, with eval_s as fallback.",
            "Provider executor group_eval_s/evaluate_unified_s are inclusive wall diagnostics only; they are not used as MVM.",
            "linear_wrapper_postprocess_s is exclusive non-MVM LinearTransform wrapper work: dense/common row accumulation/rescale/bias/output rotation.",
            "wall_executor_overhead_s is exclusive provider executor wrapper work: wrap/postprocess/rescale/accumulate outside MVM.",
            "lt_runtime_load_encode_s includes runtime artifact read/key/plaintext load plus legacy chunk-stream build-map/encode/load-payload time.",
            "Single-slot layer-cache turnover is reported separately as lt_layer_cache_turnover_s.",
            "compute_mvm_activation_bootstrap_s is the strict compute-comparison sum: MVM kernel + activation excluding bootstrap + bootstrap.",
            "wall_aux_accounted_s adds layer-cache turnover, runtime load/trim, linear wrapper work, and provider executor overhead that occur inside he_forward_s but are not MVM+activation+bootstrap compute.",
            "unattributed_he_forward_s and wall_unattributed_he_forward_s are the residual after compute plus wall auxiliary accounting.",
            "Activation is a targeted activation-only wall-time hook; direct bootstrap child time is subtracted.",
            "Bootstrap uses Bootstrap._bootstrap_runtime_profile and does not require broad module profiling.",
        ],
        "totals": totals,
        "mvm": mvm,
        "activation": activation,
        "bootstrap": bootstrap,
        "module_wall": module_wall,
    }


def _backend_u64_array(callable_obj: Callable[..., Any], *args: Any) -> list[int]:
    values = callable_obj(*args)
    if isinstance(values, int):
        return [int(values)]
    try:
        return [int(value) for value in list(values)]
    except TypeError:
        return []


def _live_ciphertext_snapshot() -> dict[str, Any]:
    backend = getattr(scheme, "backend", None)
    get_live = getattr(backend, "GetLiveCiphertexts", None)
    if not callable(get_live):
        return {"available": False}
    raw_values = get_live()
    if isinstance(raw_values, int):
        return {
            "available": True,
            "count": int(raw_values),
            "ids_sample": [],
            "max_id": None,
        }
    try:
        values = [int(value) for value in list(raw_values)]
    except TypeError:
        return {
            "available": False,
            "error": f"non-iterable GetLiveCiphertexts result: {type(raw_values).__name__}",
            "ids_sample": [],
            "max_id": None,
        }
    return {
        "available": True,
        "count": int(len(values)),
        "ids_sample": [int(value) for value in values[:8]],
        "ids_tail": [int(value) for value in values[-8:]],
        "max_id": int(max(values)) if values else None,
    }


def _device_memory_snapshot() -> dict[str, Any]:
    backend = getattr(scheme, "backend", None)
    get_info = getattr(backend, "GetDeviceMemoryInfo", None)
    if not callable(get_info):
        return {"available": False}
    values = _backend_u64_array(get_info)
    free_bytes = int(values[0]) if len(values) >= 1 else 0
    total_bytes = int(values[1]) if len(values) >= 2 else 0
    return {
        "available": True,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "used_bytes": int(max(0, total_bytes - free_bytes)) if total_bytes else 0,
    }


def _linear_transform_device_estimate(transform_id: int) -> dict[str, Any]:
    backend = getattr(scheme, "backend", None)
    estimate = getattr(backend, "EstimateLinearTransformDeviceBytes", None)
    uses_streaming = getattr(backend, "LinearTransformUsesStreaming", None)
    values = _backend_u64_array(estimate, int(transform_id)) if callable(estimate) else []
    return {
        "transform_id": int(transform_id),
        "estimate_device_bytes": int(values[0]) if values else None,
        "uses_streaming": (
            bool(int(uses_streaming(int(transform_id)))) if callable(uses_streaming) else None
        ),
    }


def _nonidentity_rotation_keys(keys: list[int], *, slots: int | None = None) -> list[int]:
    if slots is None:
        try:
            slots = int(scheme.params.get_slots())
        except Exception:
            slots = 32768
    identity = identity_galois_element(slots=int(slots))
    return sorted(int(key) for key in keys if int(key) != int(identity))


def _backend_transform_rotation_eval_count(transform_id: int, *, fallback_keys: list[int] | None = None) -> int:
    backend = getattr(scheme, "backend", None)
    getter = getattr(backend, "GetLinearTransformRotationEvalCount", None)
    if callable(getter):
        return int(getter(int(transform_id)))
    keys = list(fallback_keys or getattr(backend, "GetLinearTransformRotationKeys")(int(transform_id)))
    return int(len(_nonidentity_rotation_keys([int(key) for key in keys])))


def _dense_cols(module: torch.nn.Module) -> int:
    keys = list(getattr(module, "transform_ids", {}).keys())
    if not keys:
        return 0
    return max(int(col) for _row, col in keys) + 1


def _linear_transform_rotation_stats(module: torch.nn.Module) -> dict[str, Any]:
    transform_ids = dict(getattr(module, "transform_ids", {}) or {})
    if not transform_ids:
        cached = getattr(module, "_dense_layer_cache_rotation_stats", None)
        if isinstance(cached, dict):
            return dict(cached)
    per_transform: list[dict[str, Any]] = []
    unique_keys: set[int] = set()
    transform_rotation_total = 0
    device_estimates: list[dict[str, Any]] = []
    for (row, col), transform_id in sorted(transform_ids.items()):
        keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
        nonidentity_keys = _nonidentity_rotation_keys(keys)
        rotation_eval_count = _backend_transform_rotation_eval_count(int(transform_id), fallback_keys=keys)
        unique_keys.update(nonidentity_keys)
        transform_rotation_total += int(rotation_eval_count)
        estimate = _linear_transform_device_estimate(int(transform_id))
        device_estimates.append(estimate)
        per_transform.append(
            {
                "row": int(row),
                "col": int(col),
                "transform_id": int(transform_id),
                "rotation_eval_count": int(rotation_eval_count),
                "rotation_key_count": int(len(nonidentity_keys)),
                "rotation_key_request_count": int(len(keys)),
                "estimate_device_bytes": estimate.get("estimate_device_bytes"),
                "uses_streaming": estimate.get("uses_streaming"),
            }
        )
    cols = _dense_cols(module)
    rows = int(len(transform_ids) // max(1, int(cols)))
    output_rotations = int(getattr(module, "output_rotations", 0))
    output_rotation_evals = int(rows * output_rotations)
    estimate_values = [
        int(item["estimate_device_bytes"])
        for item in device_estimates
        if item.get("estimate_device_bytes") is not None
    ]
    return {
        "source": "compiled_backend_transform_rotation_keys",
        "transform_count": int(len(transform_ids)),
        "rows": int(rows),
        "cols": int(cols),
        "transform_rotation_eval_count_total": int(transform_rotation_total),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": int(output_rotations),
        "output_rotation_eval_count": int(output_rotation_evals),
        "rotation_eval_count_estimate": int(transform_rotation_total + output_rotation_evals),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "estimate_device_bytes_total": int(sum(estimate_values)) if estimate_values else None,
        "estimate_device_bytes_max": int(max(estimate_values)) if estimate_values else None,
        "streaming_transform_count": int(
            sum(1 for item in device_estimates if item.get("uses_streaming") is True)
        ),
        "per_transform": per_transform,
    }


def _unified_group_rotation_stats(groups: list[Any]) -> dict[str, Any]:
    individual_eval = str(os.environ.get("ORION_UNIFIED_LT_INDIVIDUAL_EVAL", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    per_group: list[dict[str, Any]] = []
    unique_keys: set[int] = set()
    transform_rotation_total = 0
    shared_rotation_total = 0
    transform_count = 0
    estimate_values: list[int] = []
    streaming_transform_count = 0
    for group_index, group in enumerate(groups):
        ids = [int(value) for value in (getattr(group, "unified_ids", None) or [])]
        group_keys: set[int] = set()
        per_transform: list[dict[str, Any]] = []
        cached = getattr(group, "_single_slot_rotation_stats", None)
        if not ids and isinstance(cached, dict):
            cached_keys = {int(key) for key in cached.get("unique_rotation_keys", [])}
            group_keys.update(cached_keys)
            unique_keys.update(cached_keys)
            group_transform_total = int(
                cached.get(
                    "transform_rotation_eval_count_total",
                    cached.get("transform_rotation_key_count_total", 0),
                )
                or 0
            )
            group_shared_total = int(
                cached.get("shared_rotation_eval_count", cached.get("shared_rotation_eval_count_total", 0)) or 0
            )
            transform_rotation_total += int(group_transform_total)
            shared_rotation_total += int(group_shared_total)
            group_transform_count = int(cached.get("transform_count", 0) or 0)
            transform_count += int(group_transform_count)
            per_group.append(
                {
                    "group_index": int(group_index),
                    "transform_count": int(group_transform_count),
                    "rotation_key_count_total": int(group_transform_total),
                    "shared_rotation_eval_count": int(group_shared_total),
                    "unique_rotation_key_count": int(len(cached_keys)),
                    "per_transform": list(cached.get("per_transform", []) or []),
                    "source": str(cached.get("source", "planned_single_slot_unified_rotation_keys")),
                }
            )
            continue
        diag_indices_by_transform = getattr(group, "_diag_indices_by_transform", None)
        if isinstance(diag_indices_by_transform, dict) and ids:
            diag_sets = [
                tuple(int(value) for value in diag_indices_by_transform.get(int(transform_id), ()))
                for transform_id in ids
            ]
            try:
                slots = int(scheme.params.get_slots())
            except Exception:
                slots = 32768
            group_stats = unified_bsgs_rotation_stats(
                diag_sets,
                slots=int(slots),
                individual_eval=bool(individual_eval),
            )
            group_transform_total = int(group_stats["transform_rotation_eval_count_total"])
            group_shared_total = int(group_stats["shared_rotation_eval_count_total"])
            transform_rotation_total += int(group_transform_total)
            shared_rotation_total += int(group_shared_total)
            transform_count += int(len(ids))
            unique_keys.update(int(key) for key in group_stats["unique_rotation_keys"])
            for transform_index, transform_id in enumerate(ids):
                estimate = _linear_transform_device_estimate(int(transform_id))
                if estimate.get("estimate_device_bytes") is not None:
                    estimate_values.append(int(estimate["estimate_device_bytes"]))
                if estimate.get("uses_streaming") is True:
                    streaming_transform_count += 1
                stat = dict(group_stats["per_transform"][transform_index])
                per_transform.append(
                    {
                        "transform_index": int(transform_index),
                        "transform_id": int(transform_id),
                        "rotation_eval_count": int(stat.get("rotation_eval_count", 0) or 0),
                        "rotation_key_count": int(stat.get("rotation_key_count", 0) or 0),
                        "rotation_key_request_count": int(stat.get("rotation_key_request_count", 0) or 0),
                        "estimate_device_bytes": estimate.get("estimate_device_bytes"),
                        "uses_streaming": estimate.get("uses_streaming"),
                    }
                )
            per_group.append(
                {
                    "group_index": int(group_index),
                    "transform_count": int(len(ids)),
                    "rotation_key_count_total": int(group_transform_total),
                    "transform_rotation_eval_count_total": int(group_transform_total),
                    "shared_rotation_eval_count": int(group_shared_total),
                    "unique_rotation_key_count": int(group_stats["unique_rotation_key_count"]),
                    "per_transform": per_transform,
                    "source": "compiled_group_diag_indices_bsgs_eval_rotations",
                    "n1": int(group_stats["n1"]),
                }
            )
            continue
        for transform_index, transform_id in enumerate(ids):
            keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
            nonidentity_keys = _nonidentity_rotation_keys(keys)
            rotation_eval_count = _backend_transform_rotation_eval_count(int(transform_id), fallback_keys=keys)
            estimate = _linear_transform_device_estimate(int(transform_id))
            if estimate.get("estimate_device_bytes") is not None:
                estimate_values.append(int(estimate["estimate_device_bytes"]))
            if estimate.get("uses_streaming") is True:
                streaming_transform_count += 1
            group_keys.update(nonidentity_keys)
            unique_keys.update(nonidentity_keys)
            transform_rotation_total += int(rotation_eval_count)
            transform_count += 1
            per_transform.append(
                {
                    "transform_index": int(transform_index),
                    "transform_id": int(transform_id),
                    "rotation_eval_count": int(rotation_eval_count),
                    "rotation_key_count": int(len(nonidentity_keys)),
                    "rotation_key_request_count": int(len(keys)),
                    "estimate_device_bytes": estimate.get("estimate_device_bytes"),
                    "uses_streaming": estimate.get("uses_streaming"),
                }
            )
        shared_rotation_total += int(len(group_keys))
        per_group.append(
            {
                "group_index": int(group_index),
                "transform_count": int(len(ids)),
                "rotation_key_count_total": int(sum(int(item["rotation_key_count"]) for item in per_transform)),
                "shared_rotation_eval_count": int(len(group_keys)),
                "unique_rotation_key_count": int(len(group_keys)),
                "per_transform": per_transform,
            }
        )
    return {
        "source": "compiled_backend_unified_transform_rotation_keys",
        "group_count": int(len(groups)),
        "group_object_ids": [int(id(group)) for group in groups],
        "transform_count": int(transform_count),
        "transform_rotation_eval_count_total": int(transform_rotation_total),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "shared_rotation_eval_count_total": int(shared_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": 0,
        "output_rotation_eval_count": 0,
        "rotation_eval_count_estimate": int(transform_rotation_total if individual_eval else shared_rotation_total),
        "rotation_eval_count_mode": "independent_transform_bsgs" if individual_eval else "shared_group_bsgs",
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "estimate_device_bytes_total": int(sum(estimate_values)) if estimate_values else None,
        "estimate_device_bytes_max": int(max(estimate_values)) if estimate_values else None,
        "streaming_transform_count": int(streaming_transform_count),
        "per_group": per_group,
    }


def _rotation_report_stats_key(stats: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    group_ids = tuple(int(value) for value in list(stats.get("group_object_ids", []) or []))
    if group_ids:
        return ("groups", tuple(sorted(group_ids)))
    transform_ids: list[int] = []
    for item in list(stats.get("per_transform", []) or []):
        if isinstance(item, dict) and item.get("transform_id") is not None:
            transform_ids.append(int(item["transform_id"]))
    for group in list(stats.get("per_group", []) or []):
        if not isinstance(group, dict):
            continue
        for item in list(group.get("per_transform", []) or []):
            if isinstance(item, dict) and item.get("transform_id") is not None:
                transform_ids.append(int(item["transform_id"]))
    if transform_ids:
        return ("transforms", tuple(sorted(set(transform_ids))))
    return ("stats", (int(id(stats)),))


def _provider_rotation_stats(executor: Any) -> dict[str, Any]:
    groups = _executor_unified_groups(executor)
    if groups:
        return _unified_group_rotation_stats(groups)
    if getattr(executor, "group", None) is not None:
        return _unified_group_rotation_stats([executor.group])
    groups = list(getattr(executor, "groups", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups = list(getattr(executor, "groups_by_input_block", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups = list(getattr(executor, "groups_by_pair", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups = list(getattr(executor, "groups_by_input", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups = list(getattr(executor, "groups_by_input_chunk", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups_by_input_index = getattr(executor, "groups_by_input_index", None) or {}
    if groups_by_input_index:
        return _unified_group_rotation_stats([group for _input_index, group in sorted(groups_by_input_index.items())])
    runtime_groups = list(getattr(executor, "runtime_groups", []) or [])
    if runtime_groups:
        groups = []
        for runtime_group in runtime_groups:
            group = getattr(runtime_group, "group", runtime_group)
            if group is not None:
                groups.append(group)
        if groups:
            return _unified_group_rotation_stats(groups)
    runtime_io = dict(getattr(executor, "last_runtime_io", {}) or {})
    if runtime_io.get("rotation_eval_count_estimate") is not None:
        group_mode = str(runtime_io.get("rotation_eval_count_mode", "") or "")
        return {
            "source": str(runtime_io.get("source", "runtime_io_unified_rotation_snapshot")),
            "group_count": int(runtime_io.get("runtime_group_count") or 0),
            "transform_count": int(runtime_io.get("runtime_transform_count") or 0),
            "transform_rotation_key_count_total": int(runtime_io.get("transform_rotation_key_count_total") or 0),
            "shared_rotation_eval_count_total": int(runtime_io.get("shared_rotation_eval_count_total") or 0),
            "unique_rotation_key_count": int(runtime_io.get("unique_rotation_key_count") or 0),
            "output_rotations_per_output_ct": int(runtime_io.get("output_rotations") or 0),
            "output_rotation_eval_count": int(runtime_io.get("output_rotation_eval_count") or 0),
            "rotation_eval_count_estimate": int(runtime_io.get("rotation_eval_count_estimate") or 0),
            "rotation_eval_count_mode": str(group_mode),
            "unique_rotation_keys": [
                int(key)
                for key in list(runtime_io.get("unique_rotation_keys", []) or [])
            ],
            "per_group": list(runtime_io.get("runtime_rotation_groups", []) or []),
        }
    if runtime_io.get("native_c_only_rotations") is not None:
        group_mode = str(
            runtime_io.get("provider_lt_grouping_mode")
            or runtime_io.get("lt_grouping_mode")
            or ""
        ).strip().lower().replace("-", "_")
        individual = group_mode in {
            "individual",
            "individual_lt",
            "per_lt",
            "per_linear_transform",
            "no_share",
            "no_shared_rotation",
            "disable_shared_rotation",
        } or bool(runtime_io.get("provider_disable_shared_rotation", False))
        transform_total = int(runtime_io.get("native_c_only_rotations") or 0)
        shared_total = int(runtime_io.get("native_cb_shared_rotations") or 0)
        transform_count = int(
            runtime_io.get("runtime_transform_count")
            or getattr(executor, "last_runtime_counts", {}).get("partial_count", 0)
            or getattr(executor, "last_runtime_timing", {}).get("built_transform_count", 0)
            or runtime_io.get("runtime_group_count", 0)
            or 0
        )
        return {
            "source": "runtime_io_native_halo_rotation_estimate",
            "group_count": int(runtime_io.get("runtime_group_count") or 0),
            "transform_count": int(transform_count),
            "transform_rotation_key_count_total": int(transform_total),
            "shared_rotation_eval_count_total": int(shared_total),
            "unique_rotation_key_count": 0,
            "output_rotations_per_output_ct": 0,
            "output_rotation_eval_count": 0,
            "rotation_eval_count_estimate": int(transform_total if individual else shared_total),
            "rotation_eval_count_mode": "independent_transform_bsgs" if individual else "shared_group_bsgs",
            "unique_rotation_keys": [],
            "native_plan_c_only_rotation_estimate": (
                None
                if runtime_io.get("native_plan_c_only_rotations") is None
                else int(runtime_io.get("native_plan_c_only_rotations") or 0)
            ),
            "native_plan_cb_shared_rotation_estimate": (
                None
                if runtime_io.get("native_plan_cb_shared_rotations") is None
                else int(runtime_io.get("native_plan_cb_shared_rotations") or 0)
            ),
            "native_halo_channel_fold_mode": str(runtime_io.get("native_halo_channel_fold_mode", "")),
            "native_output_storage_layout": str(runtime_io.get("native_output_storage_layout", "")),
        }
    transform_ids = dict(getattr(executor, "transform_ids", {}) or {})
    if not transform_ids:
        return {
            "source": "no_backend_unified_transform_ids",
            "group_count": 0,
            "transform_count": 0,
            "transform_rotation_key_count_total": 0,
            "unique_rotation_key_count": 0,
            "output_rotations_per_output_ct": 0,
            "output_rotation_eval_count": 0,
            "rotation_eval_count_estimate": 0,
            "unique_rotation_keys": [],
        }

    unique_keys: set[int] = set()
    per_transform: list[dict[str, Any]] = []
    transform_rotation_total = 0
    cols = max(int(col) for _row, col in transform_ids) + 1
    rows = max(int(row) for row, _col in transform_ids) + 1
    estimate_values: list[int] = []
    streaming_transform_count = 0
    for (row, col), transform_id in sorted(transform_ids.items()):
        keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
        nonidentity_keys = _nonidentity_rotation_keys(keys)
        rotation_eval_count = _backend_transform_rotation_eval_count(int(transform_id), fallback_keys=keys)
        estimate = _linear_transform_device_estimate(int(transform_id))
        if estimate.get("estimate_device_bytes") is not None:
            estimate_values.append(int(estimate["estimate_device_bytes"]))
        if estimate.get("uses_streaming") is True:
            streaming_transform_count += 1
        unique_keys.update(nonidentity_keys)
        transform_rotation_total += int(rotation_eval_count)
        per_transform.append(
            {
                "row": int(row),
                "col": int(col),
                "transform_id": int(transform_id),
                "rotation_eval_count": int(rotation_eval_count),
                "rotation_key_count": int(len(nonidentity_keys)),
                "rotation_key_request_count": int(len(keys)),
                "estimate_device_bytes": estimate.get("estimate_device_bytes"),
                "uses_streaming": estimate.get("uses_streaming"),
            }
        )
    output_rotations = int(getattr(executor, "output_rotations", 0))
    output_rotation_evals = int(rows * output_rotations)
    return {
        "source": "compiled_backend_executor_transform_rotation_keys",
        "transform_count": int(len(transform_ids)),
        "rows": int(rows),
        "cols": int(cols),
        "transform_rotation_eval_count_total": int(transform_rotation_total),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": int(output_rotations),
        "output_rotation_eval_count": int(output_rotation_evals),
        "rotation_eval_count_estimate": int(transform_rotation_total + output_rotation_evals),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "estimate_device_bytes_total": int(sum(estimate_values)) if estimate_values else None,
        "estimate_device_bytes_max": int(max(estimate_values)) if estimate_values else None,
        "streaming_transform_count": int(streaming_transform_count),
        "per_transform": per_transform,
    }


def _collect_rotation_report(net: torch.nn.Module, *, mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_runtime_ids: set[int] = set()
    for name, module in net.named_modules():
        if not isinstance(module, LinearTransform):
            continue
        module_unified_groups = _executor_unified_groups(module)
        if module_unified_groups:
            rows.append(
                {
                    "node": str(name),
                    "module_path": str(name),
                    "kind": f"{type(module).__name__}_module_unified",
                    "stage": "module_unified",
                    "nodes": [str(name)],
                    "stats": _unified_group_rotation_stats(module_unified_groups),
                }
            )
        runtime = getattr(module, "region_runtime", None)
        runtime_supported = bool(runtime is not None and getattr(runtime, "supports_scheme", lambda _scheme: True)(scheme))
        if (
            runtime is not None
            and bool(getattr(runtime, "executable", False))
            and bool(runtime_supported)
            and getattr(runtime, "executor", None) is not None
        ):
            runtime_id = id(runtime)
            if runtime_id not in seen_runtime_ids:
                seen_runtime_ids.add(runtime_id)
                rows.append(
                    {
                        "node": str(getattr(module, "region_output_id", name)),
                        "module_path": str(name),
                        "kind": "provider_region",
                        "stage": str(getattr(runtime, "stage", "")),
                        "nodes": list(getattr(runtime, "conv_nodes", (str(name),))),
                        "stats": _provider_rotation_stats(runtime.executor),
                    }
                )
            continue
        if getattr(module, "transform_ids", None) or getattr(module, "_dense_layer_cache_rotation_stats", None):
            rows.append(
                {
                    "node": str(name),
                    "module_path": str(name),
                    "kind": type(module).__name__,
                    "stage": "",
                    "nodes": [str(name)],
                    "stats": _linear_transform_rotation_stats(module),
                }
            )

    total_rows: list[dict[str, Any]] = []
    seen_stats: set[tuple[str, tuple[int, ...]]] = set()
    for row in rows:
        stats = dict(row.get("stats", {}) or {})
        key = _rotation_report_stats_key(stats)
        row["stats_identity"] = {"kind": key[0], "ids": [int(value) for value in key[1]]}
        duplicate = key in seen_stats
        row["duplicate_rotation_stats"] = bool(duplicate)
        if not duplicate:
            seen_stats.add(key)
            total_rows.append(row)

    total_rotation_estimate = int(
        sum(int(row.get("stats", {}).get("rotation_eval_count_estimate", 0)) for row in total_rows)
    )
    total_transform_rotation_keys = int(
        sum(int(row.get("stats", {}).get("transform_rotation_key_count_total", 0)) for row in total_rows)
    )
    total_shared_rotation_evals = int(
        sum(int(row.get("stats", {}).get("shared_rotation_eval_count_total", 0)) for row in total_rows)
    )
    total_output_rotation_evals = int(
        sum(int(row.get("stats", {}).get("output_rotation_eval_count", 0)) for row in total_rows)
    )
    return {
        "mode": str(mode),
        "source": "compiled_backend_rotation_keys_plus_output_rotation_estimate",
        "row_count": int(len(rows)),
        "unique_rotation_stats_row_count": int(len(total_rows)),
        "duplicate_rotation_stats_row_count": int(len(rows) - len(total_rows)),
        "total_rotation_eval_count_estimate": int(total_rotation_estimate),
        "total_transform_rotation_key_count": int(total_transform_rotation_keys),
        "total_shared_rotation_eval_count": int(total_shared_rotation_evals),
        "total_output_rotation_eval_count": int(total_output_rotation_evals),
        "rows": rows,
    }


def _collect_bootstrap_report(net: torch.nn.Module) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, module in net.named_modules():
        if isinstance(module, Bootstrap):
            runtime_profile = list(getattr(module, "_bootstrap_runtime_profile", []) or [])
            rows.append(
                {
                    "name": str(name),
                    "input_level": int(getattr(module, "input_level", -1)),
                    "bootstrap_slots": int(getattr(module, "bootstrap_slots", 0) or 0),
                    "prescale": float(getattr(module, "prescale", 0.0)),
                    "postscale": float(getattr(module, "postscale", 0.0)),
                    "runtime_profile": runtime_profile,
                    "runtime_call_count": int(len(runtime_profile)),
                }
            )
    by_slots: dict[str, int] = {}
    for row in rows:
        slots = str(int(row["bootstrap_slots"]))
        by_slots[slots] = int(by_slots.get(slots, 0)) + 1
    return {"count": int(len(rows)), "by_slots": by_slots, "rows": rows}


def _name_bootstraps(net: torch.nn.Module) -> None:
    for name, module in net.named_modules():
        if isinstance(module, Bootstrap):
            module.bootstrap_debug_name = str(name)


def _attempt_timed(
    payload: dict[str, Any],
    out_path: Path,
    attempt: dict[str, Any],
    attempt_index: int,
    step: str,
    fn: Callable[[], Any],
    *,
    record_primary_timing: bool,
) -> Any:
    payload["phase"] = "forward"
    payload["active_forward_attempt"] = int(attempt_index)
    payload["step"] = f"forward_{int(attempt_index)}_{step}"
    attempt["step"] = str(step)
    _write(payload, out_path)
    started = time.perf_counter()
    value = fn()
    elapsed = float(time.perf_counter() - started)
    attempt.setdefault("timing_s", {})[str(step)] = elapsed
    payload.setdefault("timing_s", {})[f"forward_{int(attempt_index)}_{step}"] = elapsed
    if bool(record_primary_timing):
        payload.setdefault("timing_s", {})[str(step)] = elapsed
    _write(payload, out_path)
    return value


def _run_forward_attempt(
    *,
    payload: dict[str, Any],
    out_path: Path,
    net: torch.nn.Module,
    x0: torch.Tensor,
    clear: torch.Tensor,
    input_level: int,
    mode: str,
    attempt_index: int,
    attempt_kind: str,
    profile_modules: bool,
    profile_lt: bool,
    trace_forward_memory: bool,
    operator_breakdown: bool,
    layer_mae: bool,
    layer_mae_clear_outputs: dict[str, torch.Tensor] | None,
    layer_mae_reference_transforms: dict[str, dict[str, Any]] | None,
    record_primary: bool,
) -> dict[str, Any]:
    memory_trace_path = (
        out_path.with_name(f"{out_path.stem}.forward{int(attempt_index)}.memory_trace.jsonl")
        if bool(trace_forward_memory)
        else None
    )
    progress_path = out_path.with_name(f"{out_path.stem}.forward{int(attempt_index)}.progress.jsonl")
    progress_state_path = out_path.with_name(f"{out_path.stem}.forward{int(attempt_index)}.progress_state.json")
    attempt: dict[str, Any] = {
        "attempt_index": int(attempt_index),
        "kind": str(attempt_kind),
        "status": "started",
        "step": "init",
        "progress_path": str(progress_path),
        "progress_state_path": str(progress_state_path),
        "device_memory_before_encrypt": _device_memory_snapshot(),
        "live_ciphertexts_before_encrypt": _live_ciphertext_snapshot(),
    }
    if memory_trace_path is not None:
        attempt["memory_trace_path"] = str(memory_trace_path)
    payload.setdefault("forward_attempts", []).append(attempt)
    _write(payload, out_path)

    profile_snapshot = None
    remove_profile = None
    activation_breakdown_snapshot = None
    remove_activation_breakdown = None
    layer_mae_rows: list[dict[str, Any]] | None = None
    remove_layer_mae = None
    layer_mae_overall_ok: bool | None = None
    if bool(profile_modules) or bool(operator_breakdown) or memory_trace_path is not None:
        profile_snapshot, remove_profile = _install_he_module_profiler(
            net,
            memory_trace_path=memory_trace_path,
        )
    if bool(operator_breakdown):
        activation_breakdown_snapshot, remove_activation_breakdown = (
            _install_activation_breakdown_profiler(net)
        )
    if bool(layer_mae):
        layer_mae_path = out_path.with_name(f"{out_path.stem}.forward{int(attempt_index)}.layer_mae.jsonl")
        attempt["layer_mae_path"] = str(layer_mae_path)
        expected_layer_mae_names = set((layer_mae_clear_outputs or {}).keys())
        layer_mae_rows, remove_layer_mae = _install_layer_mae_he_capture(
            net,
            clear_outputs=dict(layer_mae_clear_outputs or {}),
            reference_transforms=dict(layer_mae_reference_transforms or {}),
            names=expected_layer_mae_names,
            jsonl_path=layer_mae_path,
        )
        attempt["layer_mae_timing_note"] = (
            "layer MAE decodes module outputs inside the timed HE forward; use this run for correctness, "
            "not speed comparisons"
        )
    x0_ct = None
    out_ct = None
    progress_env_keys = ("ORION_PROGRESS_JSONL", "ORION_PROGRESS_STATE_JSON", "ORION_PROGRESS_CONTEXT")
    old_progress_env = {key: os.environ.get(key) for key in progress_env_keys}
    os.environ["ORION_PROGRESS_JSONL"] = str(progress_path)
    os.environ["ORION_PROGRESS_STATE_JSON"] = str(progress_state_path)
    os.environ["ORION_PROGRESS_CONTEXT"] = json.dumps(
        {
            "network": str(payload.get("network", "")),
            "mode": str(mode),
            "attempt_index": int(attempt_index),
            "attempt_kind": str(attempt_kind),
            "result_path": str(out_path),
        },
        sort_keys=True,
    )
    try:
        x0_ct = _attempt_timed(
            payload,
            out_path,
            attempt,
            int(attempt_index),
            "encrypt",
            lambda: _encrypt_model_input(x0, int(input_level), net=net, payload=payload),
            record_primary_timing=bool(record_primary),
        )
        attempt["model_input_encoding"] = dict(payload.get("model_input_encoding", {}) or {})
        attempt["input_ciphertext_count"] = int(len(getattr(x0_ct, "ids", ()) or ()))
        attempt["device_memory_after_encrypt"] = _device_memory_snapshot()
        attempt["live_ciphertexts_after_encrypt"] = _live_ciphertext_snapshot()
        _write(payload, out_path)
        lt_profile_enabled = _set_lattigo_lt_profile_enabled(True) if bool(profile_lt) else False
        bootstrap_profile_enabled = (
            _set_lattigo_bootstrap_profile_enabled(True) if bool(profile_modules) else False
        )
        try:
            out_ct = _attempt_timed(
                payload,
                out_path,
                attempt,
                int(attempt_index),
                "he_forward",
                lambda: net(x0_ct),
                record_primary_timing=bool(record_primary),
            )
        finally:
            if bool(lt_profile_enabled):
                _set_lattigo_lt_profile_enabled(False)
                attempt["lattigo_lt_profile_after_he_forward"] = _collect_lattigo_lt_profile()
                if bool(record_primary):
                    payload["lattigo_lt_profile_after_forward"] = dict(
                        attempt["lattigo_lt_profile_after_he_forward"]
                    )
                _write(payload, out_path)
            if bool(bootstrap_profile_enabled):
                _set_lattigo_bootstrap_profile_enabled(False)
                attempt["lattigo_bootstrap_profile_after_he_forward"] = (
                    _collect_lattigo_bootstrap_profile()
                )
                if bool(record_primary):
                    payload["lattigo_bootstrap_profile_after_forward"] = dict(
                        attempt["lattigo_bootstrap_profile_after_he_forward"]
                    )
                _write(payload, out_path)
            if profile_snapshot is not None:
                snapshot = profile_snapshot()
                if bool(profile_modules):
                    attempt["module_profile_after_forward"] = snapshot
                    if bool(record_primary):
                        payload["module_profile_after_forward"] = attempt["module_profile_after_forward"]
                _write(payload, out_path)
            if remove_profile is not None:
                remove_profile()
            if bool(operator_breakdown):
                activation_profile = (
                    activation_breakdown_snapshot()
                    if activation_breakdown_snapshot is not None
                    else None
                )
                try:
                    attempt["operator_breakdown_after_forward"] = _collect_forward_operator_breakdown(
                        net,
                        he_forward_s=float(attempt.get("timing_s", {}).get("he_forward", 0.0)),
                        activation_profile=activation_profile,
                        module_profile=snapshot if profile_snapshot is not None else None,
                    )
                    if bool(record_primary):
                        payload["operator_breakdown_after_forward"] = attempt[
                            "operator_breakdown_after_forward"
                        ]
                except Exception as exc:
                    attempt["operator_breakdown_error"] = f"{type(exc).__name__}: {exc}"
                    if bool(record_primary):
                        payload["operator_breakdown_error"] = attempt["operator_breakdown_error"]
                _write(payload, out_path)
            if remove_activation_breakdown is not None:
                remove_activation_breakdown()
            if remove_layer_mae is not None:
                remove_layer_mae()
                remove_layer_mae = None
            if layer_mae_rows is not None:
                layer_mae_summary = _layer_mae_summary(layer_mae_rows, expected_names=expected_layer_mae_names)
                layer_mae_overall_ok = bool(layer_mae_summary.get("overall_ok", False))
                attempt["layer_mae_after_forward"] = {
                    "enabled": True,
                    "jsonl_path": str(attempt.get("layer_mae_path", "")),
                    "timing_note": str(attempt.get("layer_mae_timing_note", "")),
                    "summary": layer_mae_summary,
                    "rows": list(layer_mae_rows),
                }
                attempt["layer_mae_overall_ok"] = bool(layer_mae_overall_ok)
                if bool(record_primary):
                    payload["layer_mae_after_forward"] = attempt["layer_mae_after_forward"]
                    payload["layer_mae_overall_ok"] = bool(layer_mae_overall_ok)
                    payload["layer_mae_timing_note"] = str(attempt.get("layer_mae_timing_note", ""))
                _write(payload, out_path)
        attempt["device_memory_after_he_forward"] = _device_memory_snapshot()
        attempt["live_ciphertexts_after_he_forward"] = _live_ciphertext_snapshot()
        runtime_fairness = _collect_runtime_fairness(
            net,
            serving_hot_s=float(attempt.get("timing_s", {}).get("he_forward", 0.0)),
        )
        provider_group_counts = _collect_provider_group_counts(net)
        attempt["runtime_fairness_timing"] = dict(runtime_fairness)
        attempt["provider_group_counts_after_forward"] = dict(provider_group_counts)
        attempt["resident_compute_s"] = runtime_fairness.get("resident_compute_s")
        attempt["serving_hot_s"] = runtime_fairness.get("serving_hot_s")
        attempt["artifact_read_s"] = runtime_fairness.get("artifact_read_s")
        attempt["artifact_load_s"] = runtime_fairness.get("artifact_load_s")
        attempt["artifact_unload_s"] = runtime_fairness.get("artifact_unload_s")
        attempt["layer_cache_turnover_s"] = runtime_fairness.get("layer_cache_turnover_s")
        attempt["layer_cache_encode_s"] = runtime_fairness.get("layer_cache_encode_s")
        attempt["layer_cache_key_prepare_s"] = runtime_fairness.get("layer_cache_key_prepare_s")
        attempt["layer_cache_evict_s"] = runtime_fairness.get("layer_cache_evict_s")
        attempt["trim_s"] = runtime_fairness.get("trim_s")
        attempt["runtime_fairness_mode"] = str(runtime_fairness.get("runtime_fairness_mode", "unknown"))
        if bool(record_primary):
            payload["runtime_fairness_timing_after_forward"] = dict(runtime_fairness)
            payload["provider_group_counts_after_forward"] = dict(provider_group_counts)
            payload["resident_compute_s"] = runtime_fairness.get("resident_compute_s")
            payload["serving_hot_s"] = runtime_fairness.get("serving_hot_s")
            payload["artifact_read_s"] = runtime_fairness.get("artifact_read_s")
            payload["artifact_load_s"] = runtime_fairness.get("artifact_load_s")
            payload["artifact_unload_s"] = runtime_fairness.get("artifact_unload_s")
            payload["layer_cache_turnover_s"] = runtime_fairness.get("layer_cache_turnover_s")
            payload["layer_cache_encode_s"] = runtime_fairness.get("layer_cache_encode_s")
            payload["layer_cache_key_prepare_s"] = runtime_fairness.get("layer_cache_key_prepare_s")
            payload["layer_cache_evict_s"] = runtime_fairness.get("layer_cache_evict_s")
            payload["trim_s"] = runtime_fairness.get("trim_s")
            payload["runtime_fairness_mode"] = str(runtime_fairness.get("runtime_fairness_mode", "unknown"))
        _write(payload, out_path)
        decoded, decode_info = _attempt_timed(
            payload,
            out_path,
            attempt,
            int(attempt_index),
            "decrypt_decode",
            lambda: _layer_mae_decode_model_output(net, out_ct),
            record_primary_timing=bool(record_primary),
        )
        decoded = _layer_mae_tensor(decoded)
        attempt["input_ciphertext_count"] = int(len(x0_ct.ids))
        attempt["output_ciphertext_count"] = int(len(out_ct.ids))
        attempt["final_decode_info"] = dict(decode_info)
        attempt["decoded"] = _tensor_payload(decoded)
        attempt["mae_vs_clear"] = _metrics(clear, decoded)
        if not bool(attempt["mae_vs_clear"].get("shape_match", False)):
            attempt["final_decode_info"]["status"] = "shape_mismatch"
            attempt["final_decode_info"]["skip_reason"] = "shape_mismatch"
        attempt["region_audit_after_forward"] = _collect_region_audit(net)
        attempt["bootstrap_report_after_forward"] = _collect_bootstrap_report(net)
        attempt["rotation_report_after_forward"] = _collect_rotation_report(net, mode=mode)
        attempt["device_memory_after_decrypt_decode"] = _device_memory_snapshot()
        attempt["live_ciphertexts_after_decrypt_decode"] = _live_ciphertext_snapshot()
        final_shape_ok = bool(attempt["mae_vs_clear"].get("shape_match", False))
        layer_mae_ok = True if layer_mae_overall_ok is None else bool(layer_mae_overall_ok)
        if not bool(final_shape_ok):
            attempt["status"] = "final_shape_mismatch"
        elif not bool(layer_mae_ok):
            attempt["status"] = "layer_mae_failed"
        else:
            attempt["status"] = "ok"
        attempt["step"] = "done"
        if bool(record_primary):
            payload["input_ciphertext_count"] = int(attempt["input_ciphertext_count"])
            payload["output_ciphertext_count"] = int(attempt["output_ciphertext_count"])
            payload["final_decode_info"] = dict(attempt["final_decode_info"])
            payload["decoded"] = attempt["decoded"]
            payload["mae_vs_clear"] = attempt["mae_vs_clear"]
            payload["region_audit_after_forward"] = attempt["region_audit_after_forward"]
            payload["bootstrap_report_after_forward"] = attempt["bootstrap_report_after_forward"]
            payload["rotation_report_after_forward"] = attempt["rotation_report_after_forward"]
            payload["device_memory_after_he_forward"] = attempt["device_memory_after_he_forward"]
            payload["device_memory_after_decrypt_decode"] = attempt["device_memory_after_decrypt_decode"]
            payload["live_ciphertexts_after_he_forward"] = attempt["live_ciphertexts_after_he_forward"]
            payload["live_ciphertexts_after_decrypt_decode"] = attempt["live_ciphertexts_after_decrypt_decode"]
        _write(payload, out_path)
        return attempt
    except BaseException as exc:
        attempt["status"] = "failed"
        attempt["error_type"] = type(exc).__name__
        attempt["error"] = str(exc)
        attempt["traceback"] = traceback.format_exc(limit=120)
        _write(payload, out_path)
        raise
    finally:
        if remove_profile is not None:
            try:
                remove_profile()
            except Exception:
                pass
        if remove_activation_breakdown is not None:
            try:
                remove_activation_breakdown()
            except Exception:
                pass
        if remove_layer_mae is not None:
            try:
                remove_layer_mae()
            except Exception:
                pass
        for tensor in (out_ct, x0_ct):
            release = getattr(tensor, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass
        for key, value in old_progress_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        gc.collect()


def _run_one(
    *,
    network: str,
    backend: str,
    mode: str,
    out_path: Path,
    seed: int,
    compile_only: bool = False,
    forward_runs: int = 1,
    warmup_runs: int = 0,
    profile_modules: bool = False,
    profile_lt: bool = False,
    trace_forward_memory: bool = False,
    operator_breakdown: bool = False,
    layer_mae: bool = False,
    provider_mode_override: str | None = None,
    io_mode: str = "none",
    io_dir: Path | None = None,
    diags_path: Path | None = None,
    keys_path: Path | None = None,
    logn_override: int | None = None,
    activation: str | None = None,
    silu_degree: int = 31,
    ckks_preset: str | None = None,
) -> dict[str, Any]:
    env_defaults = _configure_cheddar_runtime_defaults() if str(backend) == "cheddar" else {}
    lattigo_env_defaults = _configure_lattigo_runtime_defaults() if str(backend) == "lattigo" else {}
    if str(backend) != "cheddar":
        os.environ.setdefault("ORION_LATTIGO_BOOTSTRAP_MANY", "0")
    if str(backend) == "cheddar" and str(io_mode).lower() in {"save", "load"}:
        os.environ.setdefault("ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE", "1")
        env_defaults["ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE"] = str(
            os.environ.get("ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE", "")
        )
    spec = NETWORKS[str(network)]
    provider_mode = str(provider_mode_override or spec["provider_mode"]) if str(mode) == "provider" else ""
    base_config = _apply_ckks_preset(
        spec["config"](provider_mode, backend=str(backend)),
        ckks_preset,
    )
    config = _apply_io_config(
        base_config,
        backend=str(backend),
        io_mode=str(io_mode),
        io_dir=io_dir,
        diags_path=diags_path,
        keys_path=keys_path,
        logn_override=logn_override,
    )
    payload: dict[str, Any] = {
        "status": "started",
        "step": "init",
        "network": str(network),
        "backend": str(backend),
        "label": str(spec["label"]),
        "model": str(spec["model"]),
        "dataset": str(spec["dataset"]),
        "network_scope": str(spec.get("scope", "full")),
        "mode": str(mode),
        "provider_mode": str(provider_mode),
        "io_mode": str(io_mode),
        "io_dir": None if io_dir is None else str(Path(io_dir)),
        "diags_path": str(config.get("orion", {}).get("diags_path", "")),
        "keys_path": str(config.get("orion", {}).get("keys_path", "")),
        "logn_override": None if logn_override is None else int(logn_override),
        "ckks_preset": str(ckks_preset or "network-default"),
        "activation": {
            "kind": str(activation) if activation is not None else None,
            "silu_degree": int(silu_degree) if str(activation or "").lower() == "silu" else None,
        },
        "seed": int(seed),
        "input_shape": [int(v) for v in tuple(spec["input_shape"])],
        "compile_only": bool(compile_only),
        "forward_runs": int(forward_runs),
        "warmup_runs": int(warmup_runs),
        "profile_modules": bool(profile_modules),
        "profile_lt": bool(profile_lt),
        "trace_forward_memory": bool(trace_forward_memory),
        "operator_breakdown": bool(operator_breakdown),
        "layer_mae": bool(layer_mae),
        "bootstrap_many_enabled": os.environ.get("ORION_LATTIGO_BOOTSTRAP_MANY", "0") != "0",
        "saved_io_prewarm_mode": _saved_io_prewarm_mode(),
        "saved_io_prewarm_max_units": _saved_io_prewarm_max_units(),
        "cheddar_runtime_env": env_defaults,
        "lattigo_runtime_env": lattigo_env_defaults,
        "config": config,
    }
    _write(payload, out_path)
    try:
        if bool(compile_only) and str(io_mode) == "save" and str(backend) == "cheddar":
            os.environ.setdefault("ORION_UNIFIED_LT_RELEASE_INDEX_ONLY_RAW_MATRICES_AFTER_SAVE", "1")
            os.environ.setdefault("ORION_COMPILE_SKIP_BOOTSTRAPPER_GENERATION", "1")
            os.environ.setdefault("ORION_UNIFIED_LT_PREPARE_SHARED_CACHE_PLAN", "0")
        payload["phase"] = "compile_load"
        _write(payload, out_path)
        torch.manual_seed(int(seed))
        net = spec["builder"](activation=activation, silu_degree=int(silu_degree))
        net.eval()
        x0 = torch.randn(tuple(int(v) for v in spec["input_shape"]), dtype=torch.float32)
        layer_mae_clear_outputs: dict[str, torch.Tensor] | None = None
        layer_mae_reference_transforms: dict[str, dict[str, Any]] = {}
        remove_layer_mae_clear = None
        if bool(layer_mae):
            layer_mae_names = set(_layer_mae_target_names(net))
            layer_mae_clear_outputs, remove_layer_mae_clear = _install_layer_mae_clear_capture(
                net,
                layer_mae_names,
            )
            payload["layer_mae_target_modules_initial"] = sorted(layer_mae_names)
            payload["layer_mae_target_modules"] = sorted(layer_mae_names)
        with torch.no_grad():
            try:
                clear = _timed(payload, out_path, "clear_forward", lambda: net(x0))
            finally:
                if remove_layer_mae_clear is not None:
                    remove_layer_mae_clear()
        payload["clear"] = _tensor_payload(clear)
        if bool(layer_mae):
            payload["layer_mae_clear_output_count"] = int(len(layer_mae_clear_outputs or {}))
        _write(payload, out_path)

        _timed(payload, out_path, "init_scheme", lambda: scheme.init_scheme(config))
        payload["device_memory_after_init_scheme"] = _device_memory_snapshot()
        _write(payload, out_path)
        _timed(payload, out_path, "fit", lambda: scheme.fit(net, x0))
        payload["device_memory_after_fit"] = _device_memory_snapshot()
        _write(payload, out_path)
        input_level = _timed(payload, out_path, "compile", lambda: scheme.compile(net))
        _name_bootstraps(net)
        payload["input_level"] = int(input_level)
        payload["attach_audit"] = getattr(scheme, "region_first_attach_audit", {})
        payload["region_audit_after_compile"] = _collect_region_audit(net)
        payload["bootstrap_report_after_compile"] = _collect_bootstrap_report(net)
        payload["rotation_report_after_compile"] = _collect_rotation_report(net, mode=mode)
        get_compile_load_profile = getattr(getattr(scheme, "lt_evaluator", None), "get_compile_load_profile", None)
        if callable(get_compile_load_profile):
            payload["compile_load_profile_after_compile"] = get_compile_load_profile()
        if bool(layer_mae):
            adjusted_outputs, reference_transforms, reference_transform_diagnostics = _layer_mae_adjust_clear_outputs_after_compile(
                net,
                layer_mae_clear_outputs,
            )
            post_compile_layer_mae_names = set(_layer_mae_target_names(net))
            polynomial_outputs, polynomial_reference_diagnostics = _collect_layer_mae_polynomial_clear_outputs(
                net,
                x0,
                post_compile_layer_mae_names,
            )
            if str(polynomial_reference_diagnostics.get("status", "")) == "ok":
                adjusted_outputs.update(polynomial_outputs)
            layer_mae_clear_outputs = {
                str(name): value
                for name, value in dict(adjusted_outputs or {}).items()
                if str(name) in post_compile_layer_mae_names
            }
            layer_mae_reference_transforms = dict(reference_transforms)
            payload["layer_mae_target_modules_after_compile"] = sorted(post_compile_layer_mae_names)
            payload["layer_mae_target_modules"] = sorted(post_compile_layer_mae_names)
            payload["layer_mae_target_modules_removed_after_compile"] = sorted(
                str(name) for name in set(layer_mae_names) - post_compile_layer_mae_names
            )
            payload["layer_mae_reference_transform_count"] = int(len(layer_mae_reference_transforms))
            payload["layer_mae_reference_transforms"] = dict(layer_mae_reference_transforms)
            payload["layer_mae_reference_transform_diagnostics"] = dict(reference_transform_diagnostics)
            payload["layer_mae_polynomial_reference_diagnostics"] = dict(polynomial_reference_diagnostics)
            payload["layer_mae_clear_output_count_after_reference_alignment"] = int(
                len(layer_mae_clear_outputs or {})
            )
        try:
            payload["operator_breakdown_after_compile"] = _collect_compile_operator_breakdown(
                net,
                compile_s=payload.get("timing_s", {}).get("compile"),
                compile_load_profile=payload.get("compile_load_profile_after_compile", {}),
            )
        except Exception as exc:
            payload["operator_breakdown_after_compile_error"] = f"{type(exc).__name__}: {exc}"
        if _saved_io_prewarm_enabled():
            payload["phase"] = "saved_io_prewarm"
            _write(payload, out_path)
            prewarm_profile = _timed(
                payload,
                out_path,
                "saved_io_prewarm",
                lambda: _prewarm_saved_io(scheme),
            )
            payload["saved_io_prewarm_profile_after_compile"] = dict(prewarm_profile)
        payload["device_memory_after_compile"] = _device_memory_snapshot()
        payload["compile_load_done"] = True
        payload["phase"] = "compile_load_done"
        _write(payload, out_path)

        if bool(compile_only):
            payload["status"] = "ok_compile_only"
            payload["step"] = "done_compile_only"
            _write(payload, out_path)
            return payload

        net.he()
        payload["forward_attempts"] = []
        _write(payload, out_path)
        warmups = max(0, int(warmup_runs))
        runs = max(1, int(forward_runs))
        total_attempts = warmups + runs
        first_measured_index = warmups
        for attempt_index in range(total_attempts):
            attempt_kind = "warmup" if int(attempt_index) < warmups else "measured"
            _run_forward_attempt(
                payload=payload,
                out_path=out_path,
                net=net,
                x0=x0,
                clear=clear,
                input_level=int(input_level),
                mode=str(mode),
                attempt_index=int(attempt_index),
                attempt_kind=str(attempt_kind),
                profile_modules=bool(profile_modules),
                profile_lt=bool(profile_lt),
                trace_forward_memory=bool(trace_forward_memory),
                operator_breakdown=bool(operator_breakdown),
                layer_mae=bool(layer_mae),
                layer_mae_clear_outputs=layer_mae_clear_outputs,
                layer_mae_reference_transforms=layer_mae_reference_transforms,
                record_primary=bool(int(attempt_index) == int(first_measured_index)),
            )
        ok_attempts = [
            attempt
            for attempt in payload.get("forward_attempts", [])
            if str(attempt.get("status")) == "ok"
        ]
        payload["forward_ok_count"] = int(len(ok_attempts))
        measured_attempts = [
            attempt for attempt in ok_attempts if str(attempt.get("kind")) == "measured"
        ]
        warmup_attempts = [
            attempt for attempt in ok_attempts if str(attempt.get("kind")) == "warmup"
        ]
        payload["warmup_ok_count"] = int(len(warmup_attempts))
        payload["measured_forward_ok_count"] = int(len(measured_attempts))
        failed_attempts = [
            attempt
            for attempt in payload.get("forward_attempts", [])
            if str(attempt.get("status")) != "ok"
        ]
        payload["forward_failed_count"] = int(len(failed_attempts))
        if failed_attempts:
            payload["forward_failed_attempts"] = [
                {
                    "attempt_index": int(attempt.get("attempt_index", -1)),
                    "kind": str(attempt.get("kind", "")),
                    "status": str(attempt.get("status", "")),
                    "error": str(attempt.get("error", "")),
                    "mae_vs_clear": attempt.get("mae_vs_clear"),
                    "final_decode_info": attempt.get("final_decode_info"),
                }
                for attempt in failed_attempts
            ]
        if measured_attempts:
            payload["forward_mean_timing_s"] = {
                key: float(
                    sum(float(attempt.get("timing_s", {}).get(key, 0.0)) for attempt in measured_attempts)
                    / max(1, len(measured_attempts))
                )
                for key in ("encrypt", "he_forward", "decrypt_decode")
            }
            payload["measured_forward_mean_timing_s"] = dict(payload["forward_mean_timing_s"])
            runtime_fairness = _mean_runtime_fairness(measured_attempts)
            payload["measured_runtime_fairness_timing"] = dict(runtime_fairness)
            payload["resident_compute_s"] = runtime_fairness.get("resident_compute_s")
            payload["serving_hot_s"] = runtime_fairness.get("serving_hot_s")
            payload["artifact_read_s"] = runtime_fairness.get("artifact_read_s")
            payload["artifact_load_s"] = runtime_fairness.get("artifact_load_s")
            payload["artifact_unload_s"] = runtime_fairness.get("artifact_unload_s")
            payload["layer_cache_turnover_s"] = runtime_fairness.get("layer_cache_turnover_s")
            payload["layer_cache_encode_s"] = runtime_fairness.get("layer_cache_encode_s")
            payload["layer_cache_key_prepare_s"] = runtime_fairness.get("layer_cache_key_prepare_s")
            payload["layer_cache_evict_s"] = runtime_fairness.get("layer_cache_evict_s")
            payload["trim_s"] = runtime_fairness.get("trim_s")
            payload["runtime_fairness_mode"] = str(runtime_fairness.get("runtime_fairness_mode", "unknown"))
        payload["status"] = "ok" if payload.get("measured_forward_ok_count", 0) > 0 else "failed_forward"
        payload["step"] = "done"
        payload["phase"] = "done"
        _write(payload, out_path)
        return payload
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc(limit=120)
        _write(payload, out_path)
        raise
    finally:
        try:
            scheme.delete_scheme()
        except Exception:
            pass


def _artifact_runtime(payload: dict[str, Any]) -> float | None:
    timing = (
        payload.get("measured_forward_mean_timing_s")
        or payload.get("forward_mean_timing_s")
        or payload.get("timing_s", {})
    )
    values = [
        timing.get("encrypt"),
        timing.get("he_forward"),
        timing.get("decrypt_decode"),
    ]
    if any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values))


def _mean_runtime_fairness(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempts:
        return _aggregate_runtime_fairness([], serving_hot_s=0.0)
    timings = [dict(attempt.get("runtime_fairness_timing", {}) or {}) for attempt in attempts]
    result: dict[str, Any] = {}
    for key in _RUNTIME_FAIRNESS_NUMERIC_KEYS:
        values = [timing.get(key) for timing in timings if timing.get(key) is not None]
        result[key] = (
            float(sum(float(value) for value in values) / max(1, len(values)))
            if values
            else None
        )
    modes = [str(timing.get("runtime_fairness_mode", "unknown") or "unknown") for timing in timings]
    if any(mode == "streaming_eval_encode" for mode in modes):
        mode = "streaming_eval_encode"
    elif any(mode == "single_slot_layer_cache" for mode in modes):
        mode = "single_slot_layer_cache"
    elif any(mode == "memory_bounded_load_eval" for mode in modes):
        mode = "memory_bounded_load_eval"
    elif modes and all(mode == "resident_compute" for mode in modes):
        mode = "resident_compute"
    else:
        mode = "unknown"
    result["runtime_fairness_mode"] = str(mode)
    result["source_count"] = int(sum(int(timing.get("source_count", 0) or 0) for timing in timings))
    return result


def _he_forward_runtime(payload: dict[str, Any]) -> float:
    timing = (
        payload.get("measured_forward_mean_timing_s")
        or payload.get("forward_mean_timing_s")
        or payload.get("timing_s", {})
    )
    return float(timing.get("he_forward", math.nan))


def _runtime_fairness_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        timing = (
            payload.get("measured_runtime_fairness_timing")
            or payload.get("runtime_fairness_timing_after_forward")
            or payload.get("runtime_fairness_timing")
            or {}
        )
        if isinstance(timing, dict):
            value = timing.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _runtime_fairness_mode(payload: dict[str, Any]) -> str:
    mode = payload.get("runtime_fairness_mode")
    if mode is None:
        timing = (
            payload.get("measured_runtime_fairness_timing")
            or payload.get("runtime_fairness_timing_after_forward")
            or payload.get("runtime_fairness_timing")
            or {}
        )
        if isinstance(timing, dict):
            mode = timing.get("runtime_fairness_mode")
    return str(mode or "unknown")


def _ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(float(numerator) / float(denominator))


def _rotation_total(payload: dict[str, Any]) -> int | None:
    report = payload.get("rotation_report_after_forward") or payload.get("rotation_report_after_compile") or {}
    value = report.get("total_rotation_eval_count_estimate")
    return None if value is None else int(value)


def _bootstrap_count(payload: dict[str, Any]) -> int | None:
    report = payload.get("bootstrap_report_after_forward") or payload.get("bootstrap_report_after_compile") or {}
    value = report.get("count")
    return None if value is None else int(value)


def _profile_category_seconds(payload: dict[str, Any]) -> dict[str, float]:
    profile = payload.get("module_profile_after_forward") or {}
    totals = profile.get("leaf_totals_by_category") or profile.get("totals_by_category") or {}
    return {
        str(category): float(values.get("elapsed_s", 0.0))
        for category, values in totals.items()
        if isinstance(values, dict)
    }


def _summarize(*, dense_path: Path, provider_path: Path, out_path: Path) -> dict[str, Any]:
    dense = json.loads(Path(dense_path).read_text(encoding="utf-8"))
    provider = json.loads(Path(provider_path).read_text(encoding="utf-8"))
    dense_runtime_fairness = {
        "resident_compute_s": _runtime_fairness_value(dense, "resident_compute_s"),
        "serving_hot_s": _runtime_fairness_value(dense, "serving_hot_s"),
        "artifact_read_s": _runtime_fairness_value(dense, "artifact_read_s"),
        "artifact_load_s": _runtime_fairness_value(dense, "artifact_load_s"),
        "artifact_unload_s": _runtime_fairness_value(dense, "artifact_unload_s"),
        "layer_cache_turnover_s": _runtime_fairness_value(dense, "layer_cache_turnover_s"),
        "layer_cache_encode_s": _runtime_fairness_value(dense, "layer_cache_encode_s"),
        "layer_cache_key_prepare_s": _runtime_fairness_value(dense, "layer_cache_key_prepare_s"),
        "layer_cache_evict_s": _runtime_fairness_value(dense, "layer_cache_evict_s"),
        "trim_s": _runtime_fairness_value(dense, "trim_s"),
        "runtime_fairness_mode": _runtime_fairness_mode(dense),
    }
    provider_runtime_fairness = {
        "resident_compute_s": _runtime_fairness_value(provider, "resident_compute_s"),
        "serving_hot_s": _runtime_fairness_value(provider, "serving_hot_s"),
        "artifact_read_s": _runtime_fairness_value(provider, "artifact_read_s"),
        "artifact_load_s": _runtime_fairness_value(provider, "artifact_load_s"),
        "artifact_unload_s": _runtime_fairness_value(provider, "artifact_unload_s"),
        "layer_cache_turnover_s": _runtime_fairness_value(provider, "layer_cache_turnover_s"),
        "layer_cache_encode_s": _runtime_fairness_value(provider, "layer_cache_encode_s"),
        "layer_cache_key_prepare_s": _runtime_fairness_value(provider, "layer_cache_key_prepare_s"),
        "layer_cache_evict_s": _runtime_fairness_value(provider, "layer_cache_evict_s"),
        "trim_s": _runtime_fairness_value(provider, "trim_s"),
        "runtime_fairness_mode": _runtime_fairness_mode(provider),
    }
    payload: dict[str, Any] = {
        "status": "ok" if dense.get("status") == "ok" and provider.get("status") == "ok" else "partial",
        "network": provider.get("network", dense.get("network")),
        "backend": provider.get("backend", dense.get("backend")),
        "label": provider.get("label", dense.get("label")),
        "activation": provider.get("activation", dense.get("activation")),
        "dense_path": str(Path(dense_path)),
        "provider_path": str(Path(provider_path)),
        "dense": {
            "status": dense.get("status"),
            "timing_s": dense.get("timing_s", {}),
            "measured_forward_mean_timing_s": dense.get("measured_forward_mean_timing_s", {}),
            "runtime_s": _artifact_runtime(dense),
            "mae_vs_clear": dense.get("mae_vs_clear"),
            "input_level": dense.get("input_level"),
            "input_ciphertext_count": dense.get("input_ciphertext_count"),
            "output_ciphertext_count": dense.get("output_ciphertext_count"),
            "rotation_eval_count_estimate": _rotation_total(dense),
            "bootstrap_count": _bootstrap_count(dense),
            "profile_category_s": _profile_category_seconds(dense),
            "runtime_fairness_timing": dense.get("measured_runtime_fairness_timing")
            or dense.get("runtime_fairness_timing_after_forward")
            or dense.get("runtime_fairness_timing")
            or {},
            **dense_runtime_fairness,
        },
        "provider": {
            "status": provider.get("status"),
            "timing_s": provider.get("timing_s", {}),
            "measured_forward_mean_timing_s": provider.get("measured_forward_mean_timing_s", {}),
            "runtime_s": _artifact_runtime(provider),
            "mae_vs_clear": provider.get("mae_vs_clear"),
            "input_level": provider.get("input_level"),
            "input_ciphertext_count": provider.get("input_ciphertext_count"),
            "output_ciphertext_count": provider.get("output_ciphertext_count"),
            "rotation_eval_count_estimate": _rotation_total(provider),
            "bootstrap_count": _bootstrap_count(provider),
            "profile_category_s": _profile_category_seconds(provider),
            "attach_audit": provider.get("attach_audit", {}),
            "runtime_fairness_timing": provider.get("measured_runtime_fairness_timing")
            or provider.get("runtime_fairness_timing_after_forward")
            or provider.get("runtime_fairness_timing")
            or {},
            **provider_runtime_fairness,
        },
    }
    if dense.get("status") == "ok" and provider.get("status") == "ok":
        dense_values = torch.tensor(dense["decoded"]["values"], dtype=torch.float32)
        provider_values = torch.tensor(provider["decoded"]["values"], dtype=torch.float32)
        clear_dense = torch.tensor(dense["clear"]["values"], dtype=torch.float32)
        clear_provider = torch.tensor(provider["clear"]["values"], dtype=torch.float32)
        payload["clear_consistency"] = _metrics(clear_dense, clear_provider)
        payload["provider_vs_dense_decoded"] = _metrics(dense_values, provider_values)
        dense_he = _he_forward_runtime(dense)
        provider_he = _he_forward_runtime(provider)
        dense_compile = float(dense.get("timing_s", {}).get("compile", math.nan))
        provider_compile = float(provider.get("timing_s", {}).get("compile", math.nan))
        dense_runtime = _artifact_runtime(dense)
        provider_runtime = _artifact_runtime(provider)
        dense_resident = dense_runtime_fairness["resident_compute_s"]
        provider_resident = provider_runtime_fairness["resident_compute_s"]
        dense_serving = dense_runtime_fairness["serving_hot_s"]
        provider_serving = provider_runtime_fairness["serving_hot_s"]
        payload["ratios"] = {
            "he_forward_dense_over_provider": (
                float(dense_he / provider_he) if provider_he and math.isfinite(provider_he) else None
            ),
            "runtime_dense_over_provider": _ratio_or_none(dense_resident, provider_resident),
            "resident_compute_dense_over_provider": _ratio_or_none(dense_resident, provider_resident),
            "serving_hot_dense_over_provider": _ratio_or_none(dense_serving, provider_serving),
            "artifact_runtime_dense_over_provider": _ratio_or_none(dense_runtime, provider_runtime),
            "runtime_speedup_metric": "resident_compute_s",
            "compile_dense_over_provider": (
                float(dense_compile / provider_compile)
                if provider_compile and math.isfinite(provider_compile)
                else None
            ),
            "rotation_dense_over_provider": (
                float(_rotation_total(dense) / _rotation_total(provider))
                if _rotation_total(provider) not in (None, 0) and _rotation_total(dense) is not None
                else None
            ),
        }
    _write(payload, out_path)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or summarize Orion E2E dense/provider comparisons.")
    parser.add_argument("--mode", choices=("dense", "provider", "summarize"), required=True)
    parser.add_argument("--backend", choices=("lattigo", "cheddar"), default="lattigo")
    parser.add_argument("--network", choices=tuple(sorted(NETWORKS)), default="r34_imgnet")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dense-path", type=Path, default=Path("/tmp/orion_e2e_dense.json"))
    parser.add_argument("--provider-path", type=Path, default=Path("/tmp/orion_e2e_provider.json"))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--forward-runs",
        type=int,
        default=1,
        help="Run this many HE forward attempts after a single compile/load phase.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Run this many unmeasured HE forward attempts first, keeping boot/rotation keys resident.",
    )
    parser.add_argument("--profile-modules", action="store_true")
    parser.add_argument(
        "--profile-lt",
        action="store_true",
        help="Collect Lattigo linear-transform evaluator counters during HE forward.",
    )
    parser.add_argument(
        "--trace-forward-memory",
        action="store_true",
        help="Write per-module forward memory/live-ciphertext events to a JSONL sidecar.",
    )
    parser.add_argument(
        "--operator-breakdown",
        action="store_true",
        help="Collect streaming-safe MVM/activation/bootstrap/load breakdowns.",
    )
    parser.add_argument(
        "--layer-mae",
        action="store_true",
        help="Decode each instrumented module output during HE forward and compare MAE to clear reference.",
    )
    parser.add_argument("--provider-mode", type=str, default=None)
    parser.add_argument("--io-mode", choices=("none", "save", "load"), default="none")
    parser.add_argument("--io-dir", type=Path, default=None)
    parser.add_argument("--diags-path", type=Path, default=None)
    parser.add_argument("--keys-path", type=Path, default=None)
    parser.add_argument("--logn-override", type=int, default=None)
    parser.add_argument("--activation", choices=("none", "relu", "silu"), default=None)
    parser.add_argument("--silu-degree", type=int, default=31)
    parser.add_argument(
        "--ckks-preset",
        choices=("network-default", "resnet"),
        default="network-default",
        help="Override the network's default CKKS/bootstrapping parameters.",
    )
    args = parser.parse_args()

    activation = None if args.activation in (None, "none") else str(args.activation)

    if str(args.mode) == "summarize":
        _summarize(dense_path=Path(args.dense_path), provider_path=Path(args.provider_path), out_path=Path(args.out))
        return 0

    _run_one(
        network=str(args.network),
        backend=str(args.backend),
        mode=str(args.mode),
        out_path=Path(args.out),
        seed=int(args.seed),
        compile_only=bool(args.compile_only),
        forward_runs=int(args.forward_runs),
        warmup_runs=int(args.warmup_runs),
        profile_modules=bool(args.profile_modules),
        profile_lt=bool(args.profile_lt),
        trace_forward_memory=bool(args.trace_forward_memory),
        operator_breakdown=bool(args.operator_breakdown),
        layer_mae=bool(args.layer_mae),
        provider_mode_override=args.provider_mode,
        io_mode=str(args.io_mode),
        io_dir=args.io_dir,
        diags_path=args.diags_path,
        keys_path=args.keys_path,
        logn_override=args.logn_override,
        activation=activation,
        silu_degree=int(args.silu_degree),
        ckks_preset=str(args.ckks_preset),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
