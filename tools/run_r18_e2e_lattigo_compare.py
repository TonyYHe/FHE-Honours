from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.orion import scheme
from orion.experimental.cir.region_first_data import (
    R18_TINY_DENSE_FULL_STATS,
    R18_TINY_REGION_FIRST_FULL_STATS,
)
from orion.models.resnet import ResNet18
from orion.nn.activation import Activation, Chebyshev, Quad, ReLU
from orion.nn.linear import LinearTransform
from orion.nn.normalization import BatchNormNd
from orion.nn.operations import Add, Bootstrap, Mult
from orion.nn.reshape import Flatten

try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:
    def _dynamo_disable(fn):
        return fn


DEFAULT_CONFIG = Path("configs/resnet.yml")
DEFAULT_OUT = Path("/tmp/orion_r18_e2e_lattigo_compare.json")


class HeForwardBreakdownCollector:
    def __init__(self, net: torch.nn.Module) -> None:
        self.net = net
        self._module_names = {
            module: str(name) if str(name) else type(module).__name__
            for name, module in net.named_modules()
        }
        self._handles: list[Any] = []
        self._stack: list[dict[str, Any]] = []
        self._stats_by_category: dict[str, dict[str, float | int]] = {}
        self._stats_by_kind: dict[str, dict[str, float | int]] = {}
        self._stats_by_module: dict[str, dict[str, Any]] = {}

    def _categorize(self, module: torch.nn.Module) -> tuple[str, str] | None:
        if isinstance(module, Bootstrap):
            return ("bootstrap", "bootstrap")
        if isinstance(module, LinearTransform):
            return ("linear", type(module).__name__.lower())
        if isinstance(module, Mult):
            return ("activation", "mult")
        if isinstance(module, Quad):
            return ("activation", "quad")
        if isinstance(module, ReLU):
            return ("activation", "relu")
        if isinstance(module, Activation):
            return ("activation", "activation")
        if isinstance(module, Chebyshev):
            return ("activation", "chebyshev")
        if isinstance(module, Add):
            return ("elementwise", "add")
        if isinstance(module, BatchNormNd):
            return ("normalization", type(module).__name__.lower())
        if isinstance(module, Flatten):
            return ("reshape", "flatten")
        return None

    @_dynamo_disable
    def _pre_hook(self, module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
        if not bool(getattr(module, "he_mode", False)):
            return
        categorized = self._categorize(module)
        if categorized is None:
            return
        category, kind = categorized
        input_ciphertext_count = 0
        if _inputs:
            ids = getattr(_inputs[0], "ids", None)
            if ids is not None:
                try:
                    input_ciphertext_count = int(len(ids))
                except Exception:
                    input_ciphertext_count = 0
        self._stack.append(
            {
                "module_id": id(module),
                "name": self._module_names.get(module, type(module).__name__),
                "category": str(category),
                "kind": str(kind),
                "started_at": time.perf_counter(),
                "child_s": 0.0,
                "input_ciphertext_count": int(input_ciphertext_count),
            }
        )

    @_dynamo_disable
    def _post_hook(self, module: torch.nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        if not bool(getattr(module, "he_mode", False)):
            return
        frame = None
        for index in range(len(self._stack) - 1, -1, -1):
            if int(self._stack[index]["module_id"]) == id(module):
                frame = self._stack.pop(index)
                break
        if frame is None:
            return

        elapsed = float(time.perf_counter() - float(frame["started_at"]))
        exclusive_elapsed = float(max(0.0, elapsed - float(frame.get("child_s", 0.0))))
        if self._stack:
            self._stack[-1]["child_s"] = float(self._stack[-1].get("child_s", 0.0)) + elapsed

        category = str(frame["category"])
        kind = str(frame["kind"])
        name = str(frame["name"])
        input_ciphertext_count = int(frame.get("input_ciphertext_count", 0))
        bootstrap_op_count = int(input_ciphertext_count if category == "bootstrap" else 0)

        category_stats = self._stats_by_category.setdefault(
            str(category),
            {
                "exclusive_total_s": 0.0,
                "inclusive_total_s": 0.0,
                "count": 0,
                "input_ciphertext_count_total": 0,
                "bootstrap_op_count_total": 0,
            },
        )
        category_stats["exclusive_total_s"] = float(category_stats["exclusive_total_s"]) + exclusive_elapsed
        category_stats["inclusive_total_s"] = float(category_stats["inclusive_total_s"]) + elapsed
        category_stats["count"] = int(category_stats["count"]) + 1
        category_stats["input_ciphertext_count_total"] = int(category_stats["input_ciphertext_count_total"]) + input_ciphertext_count
        category_stats["bootstrap_op_count_total"] = int(category_stats["bootstrap_op_count_total"]) + bootstrap_op_count

        kind_stats = self._stats_by_kind.setdefault(
            str(kind),
            {
                "exclusive_total_s": 0.0,
                "inclusive_total_s": 0.0,
                "count": 0,
                "category": str(category),
                "input_ciphertext_count_total": 0,
                "bootstrap_op_count_total": 0,
            },
        )
        kind_stats["exclusive_total_s"] = float(kind_stats["exclusive_total_s"]) + exclusive_elapsed
        kind_stats["inclusive_total_s"] = float(kind_stats["inclusive_total_s"]) + elapsed
        kind_stats["count"] = int(kind_stats["count"]) + 1
        kind_stats["input_ciphertext_count_total"] = int(kind_stats["input_ciphertext_count_total"]) + input_ciphertext_count
        kind_stats["bootstrap_op_count_total"] = int(kind_stats["bootstrap_op_count_total"]) + bootstrap_op_count

        module_stats = self._stats_by_module.setdefault(
            str(name),
            {
                "exclusive_total_s": 0.0,
                "inclusive_total_s": 0.0,
                "count": 0,
                "category": str(category),
                "kind": str(kind),
                "input_ciphertext_count_total": 0,
                "bootstrap_op_count_total": 0,
            },
        )
        module_stats["exclusive_total_s"] = float(module_stats["exclusive_total_s"]) + exclusive_elapsed
        module_stats["inclusive_total_s"] = float(module_stats["inclusive_total_s"]) + elapsed
        module_stats["count"] = int(module_stats["count"]) + 1
        module_stats["input_ciphertext_count_total"] = int(module_stats["input_ciphertext_count_total"]) + input_ciphertext_count
        module_stats["bootstrap_op_count_total"] = int(module_stats["bootstrap_op_count_total"]) + bootstrap_op_count

    def install(self) -> None:
        for module in self.net.modules():
            if self._categorize(module) is None:
                continue
            self._handles.append(module.register_forward_pre_hook(self._pre_hook))
            self._handles.append(module.register_forward_hook(self._post_hook))

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []
        self._stack = []

    def summary(self, total_he_forward_s: float) -> dict[str, Any]:
        total_he_forward_s = float(total_he_forward_s)
        attributed_exclusive_total_s = float(
            sum(float(stats.get("exclusive_total_s", 0.0)) for stats in self._stats_by_category.values())
        )
        attributed_inclusive_total_s = float(
            sum(float(stats.get("inclusive_total_s", 0.0)) for stats in self._stats_by_category.values())
        )
        categories: dict[str, Any] = {}
        for category, stats in sorted(self._stats_by_category.items()):
            module_rows = [
                {
                    "name": str(name),
                    "kind": str(values.get("kind", "")),
                    "exclusive_total_s": float(values.get("exclusive_total_s", 0.0)),
                    "inclusive_total_s": float(values.get("inclusive_total_s", 0.0)),
                    "count": int(values.get("count", 0)),
                    "input_ciphertext_count_total": int(values.get("input_ciphertext_count_total", 0)),
                    "bootstrap_op_count_total": int(values.get("bootstrap_op_count_total", 0)),
                }
                for name, values in self._stats_by_module.items()
                if str(values.get("category", "")) == str(category)
            ]
            module_rows.sort(key=lambda row: float(row["exclusive_total_s"]), reverse=True)
            kind_rows = {
                str(kind): {
                    "exclusive_total_s": float(values.get("exclusive_total_s", 0.0)),
                    "inclusive_total_s": float(values.get("inclusive_total_s", 0.0)),
                    "count": int(values.get("count", 0)),
                    "input_ciphertext_count_total": int(values.get("input_ciphertext_count_total", 0)),
                    "bootstrap_op_count_total": int(values.get("bootstrap_op_count_total", 0)),
                }
                for kind, values in self._stats_by_kind.items()
                if str(values.get("category", "")) == str(category)
            }
            categories[str(category)] = {
                "exclusive_total_s": float(stats.get("exclusive_total_s", 0.0)),
                "inclusive_total_s": float(stats.get("inclusive_total_s", 0.0)),
                "count": int(stats.get("count", 0)),
                "input_ciphertext_count_total": int(stats.get("input_ciphertext_count_total", 0)),
                "bootstrap_op_count_total": int(stats.get("bootstrap_op_count_total", 0)),
                "exclusive_share_of_he_forward": (
                    float(stats.get("exclusive_total_s", 0.0)) / total_he_forward_s
                    if total_he_forward_s > 0.0
                    else None
                ),
                "by_kind": kind_rows,
                "top_modules": module_rows[:20],
            }
        return {
            "total_he_forward_s": float(total_he_forward_s),
            "attributed_exclusive_total_s": float(attributed_exclusive_total_s),
            "attributed_inclusive_total_s": float(attributed_inclusive_total_s),
            "unattributed_total_s": float(max(0.0, total_he_forward_s - attributed_exclusive_total_s)),
            "categories": categories,
            "per_node_top_exclusive": sorted(
                [
                    {
                        "name": str(name),
                        "category": str(values.get("category", "")),
                        "kind": str(values.get("kind", "")),
                        "exclusive_total_s": float(values.get("exclusive_total_s", 0.0)),
                        "inclusive_total_s": float(values.get("inclusive_total_s", 0.0)),
                        "count": int(values.get("count", 0)),
                        "input_ciphertext_count_total": int(values.get("input_ciphertext_count_total", 0)),
                        "bootstrap_op_count_total": int(values.get("bootstrap_op_count_total", 0)),
                    }
                    for name, values in self._stats_by_module.items()
                ],
                key=lambda row: float(row["exclusive_total_s"]),
                reverse=True,
            )[:80],
        }


def _load_config(
    config_path: Path,
    *,
    mode: str,
    io_mode: str = "none",
    io_dir: Path | None = None,
    diags_path: Path | None = None,
    keys_path: Path | None = None,
) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = dict(config)
    config["ckks_params"] = dict(config.get("ckks_params", {}))
    config["boot_params"] = dict(config.get("boot_params", {}))
    config["orion"] = dict(config.get("orion", {}))
    config["orion"]["backend"] = "lattigo"
    config["orion"]["io_mode"] = str(io_mode)
    config["orion"]["debug"] = False
    config["orion"]["experimental_region_first"] = "r18_tiny_e2e" if str(mode) == "provider" else ""
    if io_dir is not None:
        io_dir = Path(io_dir)
        io_dir.mkdir(parents=True, exist_ok=True)
        diags_path = io_dir / "diagonals.h5" if diags_path is None else diags_path
        keys_path = io_dir / "keys.h5" if keys_path is None else keys_path
    if diags_path is not None:
        Path(diags_path).parent.mkdir(parents=True, exist_ok=True)
        config["orion"]["diags_path"] = str(Path(diags_path))
    if keys_path is not None:
        Path(keys_path).parent.mkdir(parents=True, exist_ok=True)
        config["orion"]["keys_path"] = str(Path(keys_path))
    return config


def _timed(payload: dict[str, Any], out_path: Path, step: str, fn):
    payload["step"] = str(step)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    started = time.time()
    value = fn()
    payload.setdefault("timing_s", {})[str(step)] = float(time.time() - started)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return value


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    left = reference.detach().cpu().to(dtype=torch.float32)
    right = actual.detach().cpu().to(dtype=torch.float32)
    diff = right - left
    return {
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.pow(2))).item()),
    }


def _parse_eval_budget_bytes_list(raw: str | None) -> tuple[int | None, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    values: list[int | None] = []
    for token in str(raw).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"auto", "default", "none", "unset"}:
            values.append(None)
            continue
        value = int(token, 0)
        if value <= 0:
            raise ValueError("--eval-budget-bytes-list entries must be positive integers or 'auto'")
        values.append(int(value))
    return tuple(values) if values else None


def _executor_unified_groups(executor: Any) -> list[Any]:
    groups: list[Any] = []
    seen: set[int] = set()

    def add_group(value: Any) -> None:
        if value is None or id(value) in seen:
            return
        if not hasattr(value, "memory_trace"):
            return
        seen.add(id(value))
        groups.append(value)

    add_group(getattr(executor, "group", None))
    for value in list(getattr(executor, "groups", []) or []):
        add_group(value)
    for value in list(getattr(executor, "groups_by_input_block", []) or []):
        add_group(value)
    groups_by_input_index = getattr(executor, "groups_by_input_index", None) or {}
    for _key, value in sorted(groups_by_input_index.items()):
        add_group(value)
    return groups


def _memory_trace_summary(executor: Any) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    timing_keys = (
        "read_bundle_s",
        "load_keys_s",
        "load_plaintexts_s",
        "eval_s",
        "eval_total_s",
        "unload_s",
        "trim_s",
    )
    for group_index, group in enumerate(_executor_unified_groups(executor)):
        events = list(getattr(group, "memory_trace", []) or [])
        eval_group_events = [event for event in events if str(event.get("event")) == "before_eval_group_memory_bounded"]
        after_group_events = [event for event in events if str(event.get("event")) == "after_eval_group_memory_bounded"]
        chunk_events = [event for event in events if str(event.get("event")) == "after_eval_chunk_unload"]
        timing_totals = {key: 0.0 for key in timing_keys}
        for event in after_group_events:
            timing = event.get("timing", {}) or {}
            for key in timing_keys:
                timing_totals[key] += float(timing.get(key, 0.0))
        chunk_counts = [int(event.get("chunk_count", 0)) for event in eval_group_events]
        budgets = [int(event.get("eval_budget_bytes", 0)) for event in eval_group_events]
        summaries.append(
            {
                "group_index": int(group_index),
                "event_count": int(len(events)),
                "eval_group_count": int(len(eval_group_events)),
                "chunk_counts": chunk_counts[-8:],
                "eval_budget_bytes": budgets[-8:],
                "timing_totals": timing_totals,
                "last_chunks": [
                    {
                        "chunk_index": int(event.get("chunk_index", -1)),
                        "chunk_transform_count": int(event.get("chunk_transform_count", 0)),
                        "timing": dict(event.get("timing", {}) or {}),
                        "linear_transform_device_bytes": dict(event.get("linear_transform_device_bytes", {}) or {}),
                    }
                    for event in chunk_events[-8:]
                ],
            }
        )
    return {"group_count": int(len(summaries)), "groups": summaries}


def _collect_region_audit(net: torch.nn.Module) -> dict[str, Any]:
    rows = []
    for name, module in net.named_modules():
        runtime = getattr(module, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        if runtime is None or executor is None:
            continue
        rows.append(
            {
                "node": str(getattr(module, "region_output_id", name)),
                "compile_count": int(getattr(executor, "compile_count", 0)),
                "execute_count": int(getattr(runtime, "execute_count", 0)),
                "last_runtime_timing": dict(getattr(executor, "last_runtime_timing", {})),
                "memory_trace_summary": _memory_trace_summary(executor),
                "lazy_region_compile": bool(getattr(module, "region_first_probe_lazy_region_compile", False)),
            }
        )
    return {
        "selected_region_count": int(len(rows)),
        "rows": rows,
        "all_precompiled": bool(rows) and all(int(row["compile_count"]) >= 1 for row in rows),
    }


def _dense_cols(module: torch.nn.Module) -> int:
    keys = list(getattr(module, "transform_ids", {}).keys())
    if not keys:
        return 0
    return max(int(col) for _row, col in keys) + 1


def _linear_transform_rotation_stats(module: torch.nn.Module) -> dict[str, Any]:
    transform_ids = dict(getattr(module, "transform_ids", {}))
    per_transform: list[dict[str, Any]] = []
    unique_keys: set[int] = set()
    transform_rotation_total = 0
    for (row, col), transform_id in sorted(transform_ids.items()):
        keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
        nonzero_keys = sorted(int(key) for key in keys if int(key) != 0)
        unique_keys.update(nonzero_keys)
        transform_rotation_total += int(len(nonzero_keys))
        per_transform.append(
            {
                "row": int(row),
                "col": int(col),
                "transform_id": int(transform_id),
                "rotation_key_count": int(len(nonzero_keys)),
            }
        )
    cols = _dense_cols(module)
    rows = int(len(transform_ids) // max(1, int(cols)))
    output_rotations = int(getattr(module, "output_rotations", 0))
    output_rotation_evals = int(rows * output_rotations)
    return {
        "source": "lattigo_transform_rotation_keys",
        "transform_count": int(len(transform_ids)),
        "rows": int(rows),
        "cols": int(cols),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": int(output_rotations),
        "output_rotation_eval_count": int(output_rotation_evals),
        "rotation_eval_count_estimate": int(transform_rotation_total + output_rotation_evals),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "per_transform": per_transform,
    }


def _unified_group_rotation_stats(groups: list[Any]) -> dict[str, Any]:
    per_group: list[dict[str, Any]] = []
    unique_keys: set[int] = set()
    transform_rotation_total = 0
    shared_rotation_total = 0
    transform_count = 0
    for group_index, group in enumerate(groups):
        ids = [int(value) for value in (getattr(group, "unified_ids", None) or [])]
        group_keys: set[int] = set()
        per_transform: list[dict[str, Any]] = []
        for transform_index, transform_id in enumerate(ids):
            keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
            nonzero_keys = sorted(int(key) for key in keys if int(key) != 0)
            group_keys.update(nonzero_keys)
            unique_keys.update(nonzero_keys)
            transform_rotation_total += int(len(nonzero_keys))
            transform_count += 1
            per_transform.append(
                {
                    "transform_index": int(transform_index),
                    "transform_id": int(transform_id),
                    "rotation_key_count": int(len(nonzero_keys)),
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
        "source": "lattigo_unified_transform_rotation_keys",
        "group_count": int(len(groups)),
        "transform_count": int(transform_count),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "shared_rotation_eval_count_total": int(shared_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": 0,
        "output_rotation_eval_count": 0,
        "rotation_eval_count_estimate": int(shared_rotation_total),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "per_group": per_group,
    }


def _provider_rotation_stats(executor: Any) -> dict[str, Any]:
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
    groups_by_input_index = getattr(executor, "groups_by_input_index", None) or {}
    if groups_by_input_index:
        return _unified_group_rotation_stats([group for _input_index, group in sorted(groups_by_input_index.items())])
    transform_ids = dict(getattr(executor, "transform_ids", {}) or {})
    if transform_ids:
        unique_keys: set[int] = set()
        per_transform: list[dict[str, Any]] = []
        transform_rotation_total = 0
        cols = 0 if not transform_ids else max(int(col) for _row, col in transform_ids) + 1
        rows = 0 if not transform_ids else max(int(row) for row, _col in transform_ids) + 1
        for (row, col), transform_id in sorted(transform_ids.items()):
            keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
            nonzero_keys = sorted(int(key) for key in keys if int(key) != 0)
            unique_keys.update(nonzero_keys)
            transform_rotation_total += int(len(nonzero_keys))
            per_transform.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "transform_id": int(transform_id),
                    "rotation_key_count": int(len(nonzero_keys)),
                }
            )
        output_rotations = int(getattr(executor, "output_rotations", 0))
        output_rotation_evals = int(rows * output_rotations)
        return {
            "source": "lattigo_executor_transform_rotation_keys",
            "transform_count": int(len(transform_ids)),
            "rows": int(rows),
            "cols": int(cols),
            "transform_rotation_key_count_total": int(transform_rotation_total),
            "unique_rotation_key_count": int(len(unique_keys)),
            "output_rotations_per_output_ct": int(output_rotations),
            "output_rotation_eval_count": int(output_rotation_evals),
            "rotation_eval_count_estimate": int(transform_rotation_total + output_rotation_evals),
            "unique_rotation_keys": sorted(int(key) for key in unique_keys),
            "per_transform": per_transform,
        }
    return {
        "source": "no_lattigo_unified_transform_ids",
        "group_count": 0,
        "transform_count": 0,
        "transform_rotation_key_count_total": 0,
        "unique_rotation_key_count": 0,
        "output_rotations_per_output_ct": 0,
        "output_rotation_eval_count": 0,
        "rotation_eval_count_estimate": 0,
        "unique_rotation_keys": [],
    }


def _collect_rotation_report(net: torch.nn.Module, *, mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_runtime_ids: set[int] = set()
    for name, module in net.named_modules():
        if not isinstance(module, LinearTransform):
            continue
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
                stats = _provider_rotation_stats(runtime.executor)
                rows.append(
                    {
                        "node": str(getattr(module, "region_output_id", name)),
                        "module_path": str(name),
                        "kind": "provider_region",
                        "stage": str(getattr(runtime, "stage", "")),
                        "nodes": list(getattr(runtime, "conv_nodes", (str(name),))),
                        "stats": stats,
                    }
                )
            continue
        if getattr(module, "transform_ids", None):
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
    total_rotation_estimate = int(
        sum(int(row.get("stats", {}).get("rotation_eval_count_estimate", 0)) for row in rows)
    )
    return {
        "mode": str(mode),
        "source": "compiled_lattigo_rotation_keys_plus_output_rotation_estimate",
        "row_count": int(len(rows)),
        "total_rotation_eval_count_estimate": int(total_rotation_estimate),
        "static_reference_stats": (
            dict(R18_TINY_REGION_FIRST_FULL_STATS)
            if str(mode) == "provider"
            else dict(R18_TINY_DENSE_FULL_STATS)
        ),
        "rows": rows,
    }


def _collect_bootstrap_report(net: torch.nn.Module) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, module in net.named_modules():
        if isinstance(module, Bootstrap):
            rows.append(
                {
                    "name": str(name),
                    "input_level": int(getattr(module, "input_level", -1)),
                    "bootstrap_slots": int(getattr(module, "bootstrap_slots", 0) or 0),
                    "prescale": float(getattr(module, "prescale", 0.0)),
                    "postscale": float(getattr(module, "postscale", 0.0)),
                }
            )
    return {"count": int(len(rows)), "rows": rows}


def _run_one(
    *,
    mode: str,
    out_path: Path,
    config_path: Path,
    seed: int,
    profile_he_breakdown: bool = False,
    io_mode: str = "none",
    io_dir: Path | None = None,
    diags_path: Path | None = None,
    keys_path: Path | None = None,
    activation: str = "relu",
    silu_degree: int = 127,
    stem_relu: bool = True,
    compile_only: bool = False,
    eval_budget_bytes_list: tuple[int | None, ...] | None = None,
) -> dict[str, Any]:
    mode = str(mode)
    config = _load_config(
        Path(config_path),
        mode=mode,
        io_mode=str(io_mode),
        io_dir=io_dir,
        diags_path=diags_path,
        keys_path=keys_path,
    )
    payload: dict[str, Any] = {
        "status": "started",
        "step": "init",
        "network": "R18",
        "dataset": "tiny",
        "mode": mode,
        "config_path": str(Path(config_path)),
        "config": config,
        "seed": int(seed),
        "activation": {
            "kind": str(activation),
            "silu_degree": int(silu_degree) if str(activation).lower() == "silu" else None,
            "stem_relu": bool(stem_relu),
        },
        "compile_only": bool(compile_only),
        "eval_budget_bytes_list": [
            None if value is None else int(value)
            for value in (eval_budget_bytes_list or ())
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    he_breakdown_collector: HeForwardBreakdownCollector | None = None

    try:
        torch.manual_seed(int(seed))
        net = ResNet18(
            dataset="tiny",
            activation=str(activation),
            silu_degree=int(silu_degree),
            stem_relu=bool(stem_relu),
        )
        net.eval()
        x0 = torch.randn((1, 3, 64, 64), dtype=torch.float32)
        clear = _timed(payload, out_path, "clear_forward", lambda: net(x0))
        payload["clear"] = {
            "shape": list(clear.shape),
            "checksum": float(clear.detach().sum().item()),
            "l2": float(torch.linalg.vector_norm(clear.detach()).item()),
            "values": [float(v) for v in clear.detach().cpu().reshape(-1).tolist()],
        }
        Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        _timed(payload, out_path, "init_scheme", lambda: scheme.init_scheme(config))
        _timed(payload, out_path, "fit", lambda: scheme.fit(net, x0))
        input_level = _timed(payload, out_path, "compile", lambda: scheme.compile(net))
        payload["input_level"] = int(input_level)
        payload["attach_audit"] = getattr(scheme, "region_first_attach_audit", {})
        payload["region_audit_after_compile"] = _collect_region_audit(net)
        payload["bootstrap_report_after_compile"] = _collect_bootstrap_report(net)
        payload["rotation_report_after_compile"] = _collect_rotation_report(net, mode=mode)
        Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        if bool(compile_only):
            payload["status"] = "ok_compile_only"
            payload["step"] = "done_compile_only"
            Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            return payload

        x0_ct = _timed(payload, out_path, "encrypt", lambda: scheme.encrypt(scheme.encode(x0, int(input_level))))
        net.he()
        budget_trials = list(eval_budget_bytes_list or (None,))
        original_eval_budget = getattr(scheme.backend, "unified_transform_eval_budget_bytes", None)
        original_had_eval_budget = hasattr(scheme.backend, "unified_transform_eval_budget_bytes")
        trial_rows: list[dict[str, Any]] = []
        last_out_ct = None
        last_decoded = None
        for trial_index, eval_budget in enumerate(budget_trials):
            if eval_budget is None:
                if original_had_eval_budget:
                    scheme.backend.unified_transform_eval_budget_bytes = original_eval_budget
            else:
                scheme.backend.unified_transform_eval_budget_bytes = int(eval_budget)
            payload["step"] = f"he_forward_trial_{trial_index}"
            Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            trial_collector = HeForwardBreakdownCollector(net) if bool(profile_he_breakdown) else None
            if trial_collector is not None:
                trial_collector.install()
                he_breakdown_collector = trial_collector
            started = time.time()
            out_ct = net(x0_ct)
            he_forward_s = float(time.time() - started)
            if trial_collector is not None:
                trial_collector.remove()
                he_breakdown_collector = None
            decode_started = time.time()
            decoded = out_ct.decrypt().decode().detach().cpu().to(dtype=torch.float32)
            decrypt_decode_s = float(time.time() - decode_started)
            trial: dict[str, Any] = {
                "index": int(trial_index),
                "eval_budget_bytes": None if eval_budget is None else int(eval_budget),
                "he_forward_s": float(he_forward_s),
                "decrypt_decode_s": float(decrypt_decode_s),
                "output_ciphertext_count": int(len(out_ct.ids)),
                "mae_vs_clear": _metrics(clear, decoded),
            }
            if trial_collector is not None:
                trial["he_forward_breakdown"] = trial_collector.summary(float(he_forward_s))
            trial_rows.append(trial)
            payload["he_forward_trials"] = list(trial_rows)
            payload.setdefault("timing_s", {})["he_forward"] = float(he_forward_s)
            payload["timing_s"]["decrypt_decode"] = float(decrypt_decode_s)
            payload["input_ciphertext_count"] = int(len(x0_ct.ids))
            payload["output_ciphertext_count"] = int(len(out_ct.ids))
            payload["mae_vs_clear"] = trial["mae_vs_clear"]
            if last_out_ct is not None:
                del last_out_ct
            last_out_ct = out_ct
            last_decoded = decoded
            Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            if trial_index + 1 < len(budget_trials):
                del out_ct, decoded
                last_out_ct = None
                last_decoded = None
                gc.collect()
        if last_out_ct is None or last_decoded is None:
            raise RuntimeError("HE forward did not produce an output")
        out_ct = last_out_ct
        decoded = last_decoded
        if trial_rows:
            best_trial = min(trial_rows, key=lambda row: float(row.get("he_forward_s", math.inf)))
            payload["best_he_forward_trial"] = dict(best_trial)
            if "he_forward_breakdown" in best_trial:
                payload["he_forward_breakdown"] = best_trial["he_forward_breakdown"]
        payload["decoded"] = {
            "shape": list(decoded.shape),
            "checksum": float(decoded.sum().item()),
            "l2": float(torch.linalg.vector_norm(decoded).item()),
            "values": [float(v) for v in decoded.reshape(-1).tolist()],
        }
        payload["region_audit_after_forward"] = _collect_region_audit(net)
        payload["bootstrap_report_after_forward"] = _collect_bootstrap_report(net)
        payload["rotation_report_after_forward"] = _collect_rotation_report(net, mode=mode)
        payload["status"] = "ok"
        payload["step"] = "done"
        Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc(limit=120)
        Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if he_breakdown_collector is not None:
            he_breakdown_collector.remove()
        try:
            scheme.delete_scheme()
        except Exception:
            pass


def _he_forward_runtime_s(payload: dict[str, Any]) -> tuple[float, str]:
    best_trial = payload.get("best_he_forward_trial")
    if isinstance(best_trial, dict) and best_trial.get("he_forward_s") is not None:
        return float(best_trial["he_forward_s"]), "best_he_forward_trial"
    return float(payload.get("timing_s", {}).get("he_forward", math.nan)), "timing_s.he_forward"


def _summarize(*, dense_path: Path, provider_path: Path, out_path: Path) -> dict[str, Any]:
    dense = json.loads(Path(dense_path).read_text())
    provider = json.loads(Path(provider_path).read_text())
    summary: dict[str, Any] = {
        "status": "ok" if dense.get("status") == "ok" and provider.get("status") == "ok" else "partial",
        "network": "R18",
        "dense_path": str(Path(dense_path)),
        "provider_path": str(Path(provider_path)),
        "dense": {
            "status": dense.get("status"),
            "timing_s": dense.get("timing_s", {}),
            "mae_vs_clear": dense.get("mae_vs_clear"),
            "input_level": dense.get("input_level"),
            "output_ciphertext_count": dense.get("output_ciphertext_count"),
            "he_forward_breakdown": dense.get("he_forward_breakdown"),
        },
        "provider": {
            "status": provider.get("status"),
            "timing_s": provider.get("timing_s", {}),
            "mae_vs_clear": provider.get("mae_vs_clear"),
            "input_level": provider.get("input_level"),
            "output_ciphertext_count": provider.get("output_ciphertext_count"),
            "attach_audit": provider.get("attach_audit", {}),
            "he_forward_breakdown": provider.get("he_forward_breakdown"),
        },
    }
    if dense.get("status") == "ok" and provider.get("status") == "ok":
        dense_values = torch.tensor(dense["decoded"]["values"], dtype=torch.float32)
        provider_values = torch.tensor(provider["decoded"]["values"], dtype=torch.float32)
        clear_dense = torch.tensor(dense["clear"]["values"], dtype=torch.float32)
        clear_provider = torch.tensor(provider["clear"]["values"], dtype=torch.float32)
        summary["clear_consistency"] = _metrics(clear_dense, clear_provider)
        summary["provider_vs_dense_decoded"] = _metrics(dense_values, provider_values)
        dense_runtime, dense_runtime_source = _he_forward_runtime_s(dense)
        provider_runtime, provider_runtime_source = _he_forward_runtime_s(provider)
        dense_compile = float(dense.get("timing_s", {}).get("compile", math.nan))
        provider_compile = float(provider.get("timing_s", {}).get("compile", math.nan))
        summary["ratios"] = {
            "he_forward_dense_over_provider": float(dense_runtime / provider_runtime) if provider_runtime and math.isfinite(provider_runtime) else None,
            "compile_dense_over_provider": float(dense_compile / provider_compile) if provider_compile and math.isfinite(provider_compile) else None,
        }
        summary["he_forward_runtime_source"] = {
            "dense": str(dense_runtime_source),
            "provider": str(provider_runtime_source),
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or summarize R18 Lattigo E2E dense/provider comparison.")
    parser.add_argument("--mode", choices=("dense", "provider", "summarize"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dense-path", type=Path, default=Path("/tmp/orion_r18_e2e_dense.json"))
    parser.add_argument("--provider-path", type=Path, default=Path("/tmp/orion_r18_e2e_provider.json"))
    parser.add_argument("--profile-he-breakdown", action="store_true")
    parser.add_argument("--io-mode", choices=("none", "save", "load"), default="none")
    parser.add_argument("--io-dir", type=Path, default=None)
    parser.add_argument("--diags-path", type=Path, default=None)
    parser.add_argument("--keys-path", type=Path, default=None)
    parser.add_argument("--activation", choices=("relu", "silu"), default="relu")
    parser.add_argument("--silu-degree", type=int, default=127)
    parser.add_argument("--stem-relu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--eval-budget-bytes-list",
        default=None,
        help="Comma-separated eval budgets to run after one compile/encrypt; use 'auto' for backend default.",
    )
    args = parser.parse_args()
    if str(args.mode) == "summarize":
        _summarize(dense_path=Path(args.dense_path), provider_path=Path(args.provider_path), out_path=Path(args.out))
        return 0
    _run_one(
        mode=str(args.mode),
        out_path=Path(args.out),
        config_path=Path(args.config),
        seed=int(args.seed),
        profile_he_breakdown=bool(args.profile_he_breakdown),
        io_mode=str(args.io_mode),
        io_dir=args.io_dir,
        diags_path=args.diags_path,
        keys_path=args.keys_path,
        activation=str(args.activation),
        silu_degree=int(args.silu_degree),
        stem_relu=bool(args.stem_relu),
        compile_only=bool(args.compile_only),
        eval_budget_bytes_list=_parse_eval_budget_bytes_list(args.eval_budget_bytes_list),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
