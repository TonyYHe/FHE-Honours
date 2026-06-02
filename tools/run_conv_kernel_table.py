#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import resource
import shlex
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
from orion.experimental.cir.r34_orion_same_shape import _align_ciphertexts_for_add, _rescale_cipher_tensor
from orion.experimental.cir.runtime_group import RegionFirstRuntimeGroup
from orion.nn.linear import Conv2d
from orion.nn.module import Module
from tools import benchmark_node_specific_lattigo_provider_vs_dense as bench


DOC_MARKER = "CONV_KERNEL_TABLE"
DEFAULT_RUN_ROOT_BASE = REPO_ROOT / ".tmp" / "results"
LATEST_POINTER = REPO_ROOT / ".tmp" / "latest_conv_kernel_table.txt"
DEFAULT_DOC = REPO_ROOT / "docs" / "u22_orion_streaming_haloed_mainline.md"
ROW_DIR_NAME = "rows"

DEFAULT_HW = ("192x192", "224x224", "384x288", "384x384")
DEFAULT_CHANNELS = (32, 64, 128, 256)
DEFAULT_VARIANTS = ("orion", "provider_halo1_no_share", "provider_halo1_individual_lt")
DEFAULT_KERNEL_CASES = ("conv32", "conv64", "conv128", "conv256", "dec1b", "dec2b", "dec3b", "dec4b", "bottleneckb")
DEFAULT_INPUT_LEVEL = 2
DEFAULT_CLIP_PROVIDER_BOUNDARY_HALO = True
CONV_KERNEL_DEPTH = 1
CKKS_PROFILE_ID = "resnet_e2e_logn16_logscale40_h192"
E2E_LOGQ = (55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40)
E2E_LOGP = (61, 61, 61)
E2E_BOOT_LOGP = (61, 61, 61, 61, 61, 61, 61, 61)
VARIANT_CHOICES = (
    "orion",
    "provider_halo1",
    "provider_halo2",
    "provider_halo1_no_share",
    "provider_halo1_individual_lt",
    "provider_halo2_individual_lt",
)
PROVIDER_OUTPUT_LAYOUTS = ("tight_compact", "native_stripe")
VARIANT_LABELS = {
    "orion": "dense",
    "provider_halo1": "provider beta=1 shared",
    "provider_halo2": "provider beta=2 shared",
    "provider_halo1_no_share": "no-sharing",
    "provider_halo1_individual_lt": "no-sharing stripe",
    "provider_halo2_individual_lt": "provider beta=2 no-share stripe",
}


@dataclass(frozen=True)
class KernelCase:
    case_id: str
    label: str
    channels: int
    stage_gap: int
    note: str = ""


KERNEL_CASES: dict[str, KernelCase] = {
    "conv32": KernelCase(
        case_id="conv32",
        label="Conv 32,32",
        channels=32,
        stage_gap=1,
        note="existing stage-packed kernel",
    ),
    "conv64": KernelCase(
        case_id="conv64",
        label="Conv 64,64",
        channels=64,
        stage_gap=2,
        note="existing stage-packed kernel",
    ),
    "conv128": KernelCase(
        case_id="conv128",
        label="Conv 128,128",
        channels=128,
        stage_gap=4,
        note="existing stage-packed kernel",
    ),
    "conv256": KernelCase(
        case_id="conv256",
        label="Conv 256,256",
        channels=256,
        stage_gap=8,
        note="existing stage-packed kernel",
    ),
    "dec1b": KernelCase(
        case_id="dec1b",
        label="dec1b Conv 32,32",
        channels=32,
        stage_gap=1,
        note="U22 base_dim=32 decoder same-in/out conv",
    ),
    "dec2b": KernelCase(
        case_id="dec2b",
        label="dec2b Conv 64,64",
        channels=64,
        stage_gap=2,
        note="U22 base_dim=32 decoder same-in/out conv",
    ),
    "dec3b": KernelCase(
        case_id="dec3b",
        label="dec3b Conv 128,128",
        channels=128,
        stage_gap=4,
        note="U22 base_dim=32 decoder same-in/out conv",
    ),
    "dec4b": KernelCase(
        case_id="dec4b",
        label="dec4b Conv 256,256",
        channels=256,
        stage_gap=8,
        note="U22 base_dim=32 decoder same-in/out conv",
    ),
    "bottleneckb": KernelCase(
        case_id="bottleneckb",
        label="bottleneckb Conv 512,512",
        channels=512,
        stage_gap=16,
        note="U22 base_dim=32 bottleneck same-in/out conv",
    ),
}


@dataclass(frozen=True)
class ConvKernelRow:
    channels: int
    height: int
    width: int
    variant: str
    input_level: int = DEFAULT_INPUT_LEVEL
    case_id: str = ""
    stage_gap_override: int | None = None
    kernel_label: str = ""

    @property
    def hw(self) -> str:
        return f"{int(self.height)}x{int(self.width)}"

    @property
    def stage_gap(self) -> int:
        if self.stage_gap_override is not None:
            return int(self.stage_gap_override)
        mapping = {32: 1, 64: 2, 128: 4, 256: 8}
        try:
            return int(mapping[int(self.channels)])
        except KeyError as exc:
            raise ValueError(
                f"unsupported U-Net stage channel count: {self.channels}; "
                "pass an explicit --kernel-cases entry with a stage gap"
            ) from exc

    @property
    def channel_group_size(self) -> int:
        return int(self.stage_gap) * int(self.stage_gap)

    @property
    def logical_height(self) -> int:
        if int(self.height) % int(self.stage_gap) != 0:
            raise ValueError(f"HW height {self.height} is not divisible by stage gap {self.stage_gap}")
        return int(self.height) // int(self.stage_gap)

    @property
    def logical_width(self) -> int:
        if int(self.width) % int(self.stage_gap) != 0:
            raise ValueError(f"HW width {self.width} is not divisible by stage gap {self.stage_gap}")
        return int(self.width) // int(self.stage_gap)

    @property
    def logical_hw(self) -> str:
        return f"{int(self.logical_height)}x{int(self.logical_width)}"

    @property
    def fhe_channels(self) -> int:
        if int(self.channels) % int(self.channel_group_size) != 0:
            raise ValueError(
                f"channels {self.channels} are not divisible by channel group size {self.channel_group_size}"
            )
        return int(self.channels) // int(self.channel_group_size)

    @property
    def logical_chw(self) -> str:
        return f"{int(self.channels)}x{int(self.logical_height)}x{int(self.logical_width)}"

    @property
    def packed_chw(self) -> str:
        return f"{int(self.fhe_channels)}x{int(self.height)}x{int(self.width)}"

    @property
    def path(self) -> str:
        return "dense" if str(self.variant) == "orion" else "provider"

    @property
    def halo(self) -> int | None:
        if str(self.variant) in {"provider_halo1", "provider_halo1_no_share", "provider_halo1_individual_lt"}:
            return 1
        if str(self.variant) in {"provider_halo2", "provider_halo2_individual_lt"}:
            return 2
        return None

    @property
    def provider_lt_grouping_mode(self) -> str:
        if str(self.variant) == "provider_halo1_no_share":
            return "individual"
        return "individual" if str(self.variant).endswith("_individual_lt") else "shared"

    @property
    def provider_disable_shared_rotation(self) -> bool:
        return bool(self.provider_lt_grouping_mode == "individual")

    @property
    def expected_output_level(self) -> int:
        return int(self.input_level) - int(CONV_KERNEL_DEPTH)

    @property
    def native_halo_channel_fold_mode(self) -> str:
        if self.path != "provider":
            return ""
        if str(self.variant) == "provider_halo1_no_share":
            return "heuristic"
        return "per_stripe" if str(self.variant).endswith("_individual_lt") else "heuristic"

    @property
    def row_id(self) -> str:
        prefix = str(self.case_id or f"conv{int(self.channels)}")
        return f"{prefix}_{self.hw}_{self.variant}".replace("x", "x")

    @property
    def kernel_name(self) -> str:
        return str(self.kernel_label or f"Conv {int(self.channels)},{int(self.channels)}")


@contextlib.contextmanager
def _capture_stdout() -> Iterator[io.StringIO]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_root() -> Path:
    return DEFAULT_RUN_ROOT_BASE / f"conv_kernel_table_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"


def _parse_hw(value: str) -> tuple[int, int]:
    text = str(value).strip().lower().replace("*", "x")
    if "x" not in text:
        raise argparse.ArgumentTypeError(f"HW must look like HxW, got {value!r}")
    left, right = text.split("x", 1)
    try:
        height = int(left)
        width = int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"HW must look like HxW, got {value!r}") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(f"HW must be positive, got {value!r}")
    return int(height), int(width)


def _row_path(run_root: Path, row: ConvKernelRow) -> Path:
    return Path(run_root) / ROW_DIR_NAME / f"{row.row_id}.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - surface corrupt files in the table.
        return {"status": "bad_json", "error": f"{type(exc).__name__}: {exc}"}


def _iter_reuse_jsons(sources: list[Path]) -> Iterator[Path]:
    for source in sources:
        path = Path(source)
        if path.is_file() and path.suffix == ".json":
            yield path
        elif path.is_dir():
            for item in sorted(path.rglob("*.json")):
                if "fontlist" in item.name:
                    continue
                yield item


def _result_path_kind(result: dict[str, Any], fallback: str = "") -> str:
    path = str(result.get("path") or fallback or "")
    if path == "orion":
        return "dense"
    return path


def _result_input_chw(result: dict[str, Any]) -> tuple[int, int, int] | None:
    module = result.get("module") if isinstance(result.get("module"), dict) else {}
    shape = module.get("input_shape") or result.get("module_input_shape") or result.get("input_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) < 4:
        return None
    try:
        return int(shape[1]), int(shape[2]), int(shape[3])
    except (TypeError, ValueError):
        return None


def _result_input_chw_gap(result: dict[str, Any]) -> tuple[int, int, int, int]:
    module = result.get("module") if isinstance(result.get("module"), dict) else {}
    chw = _result_input_chw(result)
    if chw is None:
        raise ValueError("missing input CHW")
    raw_gap = module.get("input_gap")
    if raw_gap is None:
        raw_gap = result.get("input_gap")
    try:
        gap = int(raw_gap) if raw_gap is not None else 0
    except (TypeError, ValueError):
        gap = 0
    channels, height, width = chw
    if gap <= 0:
        fhe_shape = module.get("fhe_input_shape") or result.get("fhe_input_shape")
        if isinstance(fhe_shape, (list, tuple)) and len(fhe_shape) >= 4:
            try:
                fhe_height = int(fhe_shape[2])
                fhe_width = int(fhe_shape[3])
                height_ratio = int(fhe_height // int(height)) if int(height) > 0 and fhe_height % int(height) == 0 else 0
                width_ratio = int(fhe_width // int(width)) if int(width) > 0 and fhe_width % int(width) == 0 else 0
                if height_ratio > 0 and height_ratio == width_ratio:
                    gap = int(height_ratio)
            except (TypeError, ValueError, ZeroDivisionError):
                gap = 0
    if gap <= 0:
        gap = 1
    return int(channels), int(height), int(width), max(1, int(gap))


def _explicit_halo(result: dict[str, Any]) -> tuple[int | None, int | None]:
    top = result.get("input_halo_top")
    bottom = result.get("input_halo_bottom")
    if top is None or bottom is None:
        return None, None
    try:
        return int(top), int(bottom)
    except (TypeError, ValueError):
        return None, None


def _explicit_output_halo(result: dict[str, Any]) -> tuple[int | None, int | None]:
    top = result.get("output_halo_top")
    bottom = result.get("output_halo_bottom")
    if top is None or bottom is None:
        metadata = result.get("provider_metadata") if isinstance(result.get("provider_metadata"), dict) else {}
        native_plan = metadata.get("native_halo_conv2d_plan") if isinstance(metadata.get("native_halo_conv2d_plan"), dict) else {}
        spec = native_plan.get("spec") if isinstance(native_plan.get("spec"), dict) else {}
        top = spec.get("output_top_beta")
        bottom = spec.get("output_bottom_beta")
    if top is None or bottom is None:
        return None, None
    try:
        return int(top), int(bottom)
    except (TypeError, ValueError):
        return None, None


def _is_streaming_result(result: dict[str, Any]) -> bool:
    env = result.get("env") if isinstance(result.get("env"), dict) else {}
    raw = str(env.get("ORION_LATTIGO_STREAMING_LT", "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on", "force", "forced"}:
        return True
    mode = str(result.get("runtime_mode") or result.get("runtime_fairness_mode") or "")
    return "stream" in mode.lower()


def _candidate_results_from_payload(payload: dict[str, Any], source_path: Path) -> Iterator[tuple[dict[str, Any], str, str]]:
    if payload.get("status") == "ok" and (payload.get("row_id") or payload.get("module")):
        yield payload, _result_path_kind(payload), str(source_path)

    for detail in payload.get("details", []) if isinstance(payload.get("details"), list) else []:
        if not isinstance(detail, dict):
            continue
        case = str(detail.get("case", ""))
        for path_name in ("dense", "provider"):
            result = detail.get(path_name)
            if isinstance(result, dict):
                yield result, path_name, f"{source_path}:{case}/{path_name}"

    for network in payload.get("networks", []) if isinstance(payload.get("networks"), list) else []:
        if not isinstance(network, dict):
            continue
        network_name = str(network.get("network", ""))
        for case in network.get("cases", []) if isinstance(network.get("cases"), list) else []:
            if not isinstance(case, dict):
                continue
            node = str(case.get("node") or case.get("case") or "")
            paths = case.get("paths") if isinstance(case.get("paths"), dict) else {}
            for path_name, wrapper in paths.items():
                if not isinstance(wrapper, dict):
                    continue
                result = wrapper.get("result") if isinstance(wrapper.get("result"), dict) else wrapper
                if isinstance(result, dict):
                    yield result, str(path_name), f"{source_path}:{network_name}/{node}/{path_name}"


def _normalise_provider_lt_grouping_mode(value: Any) -> str:
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
    }:
        return "individual"
    return text


def _normalise_native_halo_channel_fold_mode(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"", "heuristic", "default", "auto"}:
        return "heuristic"
    if text in {"per_stripe", "perstripe", "variable", "variable_per_stripe"}:
        return "per_stripe"
    return text


def _provider_channel_fold_mode(result: dict[str, Any]) -> str:
    metadata = result.get("provider_metadata") if isinstance(result.get("provider_metadata"), dict) else {}
    native_plan = metadata.get("native_halo_conv2d_plan") if isinstance(metadata.get("native_halo_conv2d_plan"), dict) else {}
    top_level_native_plan = (
        result.get("native_halo_conv2d_plan") if isinstance(result.get("native_halo_conv2d_plan"), dict) else {}
    )
    return _normalise_native_halo_channel_fold_mode(
        result.get("native_halo_channel_fold_mode")
        or result.get("provider_native_halo_channel_fold_mode")
        or metadata.get("native_halo_channel_fold_mode")
        or native_plan.get("channel_fold_mode")
        or top_level_native_plan.get("channel_fold_mode")
    )


def _normalise_reused_result(row: ConvKernelRow, result: dict[str, Any], source_label: str) -> dict[str, Any]:
    timing = dict(result.get("runtime_fairness_timing") or {})
    rotation_count = (
        result.get("runtime_rotation_eval_count")
        if result.get("runtime_rotation_eval_count") is not None
        else result.get("rotation_eval_count", result.get("post_bsgs_rotation_eval_count", 0))
    )
    input_cts = result.get("input_cts", result.get("source_ciphertext_count", result.get("runtime_source_cts", 0)))
    output_cts = result.get("output_cts", result.get("output_ciphertext_count", result.get("runtime_output_cts", 0)))
    input_halo_top, input_halo_bottom = _explicit_halo(result)
    if input_halo_top is None:
        input_halo_top = row.halo
    if input_halo_bottom is None:
        input_halo_bottom = row.halo
    output_halo_top, output_halo_bottom = _explicit_output_halo(result)
    if row.path == "provider" and (output_halo_top is None or output_halo_bottom is None):
        output_halo_top, output_halo_bottom = _expected_provider_output_halo(
            row,
            output_layout=_provider_result_output_layout(result),
        )
    metadata = result.get("provider_metadata") if isinstance(result.get("provider_metadata"), dict) else {}
    provider_lt_grouping_mode = ""
    if row.path == "provider":
        provider_lt_grouping_mode = _normalise_provider_lt_grouping_mode(
            result.get("provider_lt_grouping_mode")
            or metadata.get("provider_lt_grouping_mode")
            or metadata.get("lt_grouping_mode")
            or row.provider_lt_grouping_mode
        )
    return {
        "status": "ok",
        "created_at_utc": _now_utc(),
        "reused_from": str(source_label),
        "backend": str(result.get("backend", "lattigo")),
        "row_id": row.row_id,
        "case_id": str(row.case_id or ""),
        "hw": row.hw,
        "logical_hw": row.logical_hw,
        "channels": int(row.channels),
        "conv": f"{int(row.channels)},{int(row.channels)}",
        "kernel_label": row.kernel_name,
        "input_level": int(row.input_level),
        "expected_output_level": int(row.expected_output_level),
        "actual_output_level": result.get("actual_output_level"),
        "ckks_profile": CKKS_PROFILE_ID,
        "input_gap": int(row.stage_gap),
        "output_gap": int(row.stage_gap),
        "channel_group_size": int(row.channel_group_size),
        "logical_chw": row.logical_chw,
        "packed_chw": row.packed_chw,
        "variant": str(row.variant),
        "variant_label": VARIANT_LABELS[str(row.variant)],
        "path": row.path,
        "provider_requested_input_halo": row.halo,
        "provider_lt_grouping_mode": provider_lt_grouping_mode,
        "provider_disable_shared_rotation": bool(provider_lt_grouping_mode == "individual"),
        "native_halo_channel_fold_mode": _provider_channel_fold_mode(result) if row.path == "provider" else "",
        "input_halo_top": input_halo_top,
        "input_halo_bottom": input_halo_bottom,
        "output_halo_top": output_halo_top if row.path == "provider" else None,
        "output_halo_bottom": output_halo_bottom if row.path == "provider" else None,
        "provider_output_storage_layout": _provider_result_output_layout(result) if row.path == "provider" else "",
        "kernel": "3x3/pad1/stride1",
        "compile_s": float(result.get("compile_s") or 0.0),
        "generate_diagonals_s": float(result.get("generate_diagonals_s") or 0.0),
        "compile_backend_s": float(result.get("compile_backend_s") or 0.0),
        "compile_load_profile": dict(result.get("compile_load_profile") or {}),
        "diag_builder_kind": str(result.get("diag_builder_kind") or ""),
        "diag_builder_source": str(result.get("diag_builder_source") or ""),
        "diag_builder_build_s": float(result.get("diag_builder_build_s") or 0.0),
        "diag_builder_shadow_s": float(result.get("diag_builder_shadow_s") or 0.0),
        "diag_builder_payload_count": int(float(result.get("diag_builder_payload_count") or 0.0)),
        "diag_builder_fallback_reason": str(result.get("diag_builder_fallback_reason") or ""),
        "run_count": int(result.get("run_count") or len(result.get("hot_run_s", []) or []) or 1),
        "hot_run_s": [float(value) for value in (result.get("hot_run_s") or [])],
        "hot_run_mean_s": float(result.get("hot_run_mean_s") or result.get("serving_hot_s") or 0.0),
        "runtime_fairness_timing": timing,
        "lt_accumulate_s": float(_lt_accumulate_s(timing) or result.get("resident_compute_s") or result.get("hot_run_mean_s") or 0.0),
        "runtime_mode": str(result.get("runtime_fairness_mode") or timing.get("runtime_fairness_mode") or ""),
        "rotation_eval_count": int(rotation_count or 0),
        "runtime_operation_counts": dict(result.get("runtime_operation_counts") or {}),
        "input_cts": int(input_cts or 0),
        "output_cts": int(output_cts or 0),
        "module": dict(result.get("module") or {}),
        "rotation_stats": dict(result.get("rotation_stats") or {}),
        "maxrss_bytes": None if result.get("maxrss_bytes") is None else int(result.get("maxrss_bytes") or 0),
        "env": dict(result.get("env") or {}),
    }


def _provider_result_output_layout(result: dict[str, Any]) -> str:
    metadata = result.get("provider_metadata") if isinstance(result.get("provider_metadata"), dict) else {}
    return str(
        result.get("provider_output_storage_layout")
        or metadata.get("runtime_output_storage_layout")
        or result.get("runtime_output_storage_layout")
        or ""
    )


def _provider_output_halo_mismatch(
    row: ConvKernelRow,
    result: dict[str, Any],
    *,
    provider_output_layout: str,
) -> tuple[int, int, int, int] | None:
    if row.path != "provider":
        return None
    expected_top, expected_bottom = _expected_provider_output_halo(row, output_layout=str(provider_output_layout))
    actual_top, actual_bottom = _explicit_output_halo(result)
    if actual_top is None or actual_bottom is None:
        actual_top = actual_bottom = 0
    expected = (int(expected_top or 0), int(expected_bottom or 0))
    actual = (int(actual_top), int(actual_bottom))
    if actual != expected:
        return int(actual[0]), int(actual[1]), int(expected[0]), int(expected[1])
    return None


def reuse_existing_rows(
    run_root: Path,
    rows: list[ConvKernelRow],
    sources: list[Path],
    *,
    force: bool = False,
    clip_provider_boundary_halo: bool = False,
    provider_output_layout: str = "tight_compact",
) -> int:
    if not sources:
        return 0
    row_by_key: dict[tuple[str, int, int, int, int, int, int | None, str, str, str], ConvKernelRow] = {}
    for row in rows:
        halo_key = row.halo
        if row.path == "provider" and bool(clip_provider_boundary_halo):
            halo_key = 0
        grouping_key = row.provider_lt_grouping_mode if row.path == "provider" else ""
        fold_key = row.native_halo_channel_fold_mode if row.path == "provider" else ""
        row_by_key[
            (
                row.path,
                row.channels,
                row.logical_height,
                row.logical_width,
                row.stage_gap,
                row.input_level,
                halo_key,
                grouping_key,
                fold_key,
                CKKS_PROFILE_ID,
            )
        ] = row
    reused = 0
    for json_path in _iter_reuse_jsons([Path(value) for value in sources]):
        payload = _read_json(json_path)
        if not isinstance(payload, dict):
            continue
        for result, path_name, source_label in _candidate_results_from_payload(payload, json_path):
            if result.get("status") != "ok" or _is_streaming_result(result):
                continue
            try:
                channels, height, width, gap = _result_input_chw_gap(result)
            except ValueError:
                continue
            path_kind = _result_path_kind(result, fallback=path_name)
            try:
                input_level = int(result.get("input_level"))
            except (TypeError, ValueError):
                continue
            ckks_profile = str(result.get("ckks_profile") or result.get("ckks_profile_id") or "")
            if ckks_profile != CKKS_PROFILE_ID:
                continue
            if path_kind == "dense":
                key = (
                    "dense",
                    int(channels),
                    int(height),
                    int(width),
                    int(gap),
                    int(input_level),
                    None,
                    "",
                    "",
                    ckks_profile,
                )
            elif path_kind == "provider":
                top, bottom = _explicit_halo(result)
                # Older provider JSONs did not record the halo setting; those
                # runs used the default provider halo, which is top/bottom 1/1.
                if top is None and bottom is None and not bool(clip_provider_boundary_halo):
                    top = bottom = 1
                if top is None or bottom is None or top != bottom:
                    continue
                expected_layout = "native_halo_stripe" if str(provider_output_layout) == "native_stripe" else "tight_compact"
                if _provider_result_output_layout(result) != expected_layout:
                    continue
                row_probe = row_by_key.get(
                    (
                        "provider",
                        int(channels),
                        int(height),
                        int(width),
                        int(gap),
                        int(input_level),
                        int(top),
                        _normalise_provider_lt_grouping_mode(
                            result.get("provider_lt_grouping_mode")
                            or (
                                result.get("provider_metadata", {}).get("provider_lt_grouping_mode")
                                if isinstance(result.get("provider_metadata"), dict)
                                else ""
                            )
                            or (
                                result.get("provider_metadata", {}).get("lt_grouping_mode")
                                if isinstance(result.get("provider_metadata"), dict)
                                else ""
                            )
                            or "shared"
                        ),
                        _provider_channel_fold_mode(result),
                        ckks_profile,
                    )
                )
                if row_probe is None:
                    continue
                output_mismatch = _provider_output_halo_mismatch(
                    row_probe,
                    result,
                    provider_output_layout=str(provider_output_layout),
                )
                if output_mismatch is not None:
                    continue
                metadata = result.get("provider_metadata") if isinstance(result.get("provider_metadata"), dict) else {}
                grouping_mode = _normalise_provider_lt_grouping_mode(
                    result.get("provider_lt_grouping_mode")
                    or metadata.get("provider_lt_grouping_mode")
                    or metadata.get("lt_grouping_mode")
                    or "shared"
                )
                fold_mode = _provider_channel_fold_mode(result)
                key = (
                    "provider",
                    int(channels),
                    int(height),
                    int(width),
                    int(gap),
                    int(input_level),
                    int(top),
                    grouping_mode,
                    fold_mode,
                    ckks_profile,
                )
            else:
                continue
            row = row_by_key.get(key)
            if row is None:
                continue
            target = _row_path(run_root, row)
            if not force:
                existing = _read_json(target)
                if isinstance(existing, dict) and existing.get("status") == "ok":
                    continue
            _write_json(target, _normalise_reused_result(row, result, source_label))
            reused += 1
    return int(reused)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt_float(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return ""


def _gib_from_bytes(value: Any) -> str:
    try:
        return f"{float(value) / (1024 ** 3):.1f}"
    except (TypeError, ValueError):
        return ""


def _markdown_escape(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _replace_block(text: str, marker: str, body: str) -> str:
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    left, sep, rest = text.partition(start)
    if not sep:
        raise ValueError(f"missing marker {start}")
    _old, sep, right = rest.partition(end)
    if not sep:
        raise ValueError(f"missing marker {end}")
    return f"{left}{start}\n{body}\n{end}{right}"


def _config(*, backend: str) -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": 16,
            "LogQ": list(E2E_LOGQ),
            "LogP": list(E2E_LOGP),
            "LogScale": 40,
            "H": 192,
            "RingType": "standard",
        },
        "boot_params": {
            "LogP": list(E2E_BOOT_LOGP),
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": str(backend),
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }


def _init_scheme(*, backend: str) -> None:
    os.environ["ORION_SINGLE_SLOT_LAYER_CACHE"] = "0"
    scheme.init_scheme(_config(backend=str(backend)))
    Module.set_scheme(scheme)
    Module.set_margin(scheme.params.get_margin())


def _cleanup_scheme() -> None:
    try:
        scheme.delete_scheme()
    except Exception:
        pass


def _make_conv(row: ConvKernelRow, *, seed: int) -> Conv2d:
    torch.manual_seed(int(seed))
    conv = Conv2d(
        int(row.channels),
        int(row.channels),
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    conv.eval()
    conv.name = row.row_id
    conv.region_output_id = str(conv.name)
    x = torch.randn((1, int(row.channels), int(row.logical_height), int(row.logical_width)), dtype=torch.float32)
    y = conv(x)
    conv.init_orion_params()
    conv.input_shape = torch.Size(x.shape)
    conv.output_shape = torch.Size(y.shape)
    conv.input_gap = int(row.stage_gap)
    conv.output_gap = int(row.stage_gap)
    conv.fhe_input_shape = torch.Size((1, int(row.fhe_channels), int(row.height), int(row.width)))
    conv.fhe_output_shape = torch.Size((1, int(row.fhe_channels), int(row.height), int(row.width)))
    if row.halo is not None:
        conv.layout_policy_input_layout = {"top_beta": int(row.halo), "bottom_beta": int(row.halo)}
    conv.set_level(int(row.input_level))
    return conv


def _effective_provider_input_halo(
    row: ConvKernelRow,
    *,
    clip_boundary_halo: bool,
) -> tuple[int | None, int | None]:
    if row.halo is None:
        return None, None
    if bool(clip_boundary_halo):
        return 0, 0
    return int(row.halo), int(row.halo)


def _expected_provider_output_halo(row: ConvKernelRow, *, output_layout: str) -> tuple[int | None, int | None]:
    if row.path != "provider":
        return None, None
    layout = str(output_layout or "tight_compact")
    if layout in {"native_stripe", "native_halo_stripe"}:
        halo = max(0, int(row.halo or 0) - 1)
        return int(halo), int(halo)
    if layout == "tight_compact":
        return 0, 0
    return None, None


def _apply_provider_input_halo(
    conv: Conv2d,
    row: ConvKernelRow,
    *,
    clip_boundary_halo: bool,
) -> tuple[int | None, int | None]:
    top, bottom = _effective_provider_input_halo(row, clip_boundary_halo=bool(clip_boundary_halo))
    if top is not None and bottom is not None:
        conv.layout_policy_input_layout = {"top_beta": int(top), "bottom_beta": int(bottom)}
    return top, bottom


def _apply_provider_output_layout(conv: Conv2d, row: ConvKernelRow, *, output_layout: str) -> str:
    if row.path != "provider":
        return ""
    layout = str(output_layout or "tight_compact")
    if layout == "native_stripe":
        output_top, output_bottom = _expected_provider_output_halo(row, output_layout=layout)
        conv.layout_policy_output_materialization = "native_halo_stripe"
        conv.layout_policy_output_layout = {"top_beta": int(output_top or 0), "bottom_beta": int(output_bottom or 0)}
        return "native_halo_stripe"
    if layout == "tight_compact":
        conv.layout_policy_output_materialization = "fused_relayout"
        conv.layout_policy_output_layout = {"top_beta": 0, "bottom_beta": 0}
        return "tight_compact"
    raise ValueError(f"unsupported provider output layout: {output_layout!r}")


def _attach_provider_runtime(conv: Conv2d, row: ConvKernelRow) -> None:
    conv.layout_policy_provider_lt_grouping_mode = row.provider_lt_grouping_mode
    conv.layout_policy_provider_disable_shared_rotation = bool(row.provider_disable_shared_rotation)
    conv.layout_policy_native_halo_channel_fold_mode = row.native_halo_channel_fold_mode
    executor = HaloLocalConvRuntimeExecutor(
        module=conv,
        output_node_id=str(conv.region_output_id),
    )
    runtime = RegionFirstRuntimeGroup(
        region_id=f"conv_kernel_{row.row_id}",
        network="conv_kernel_table",
        stage="synthetic",
        module_prefix=str(conv.name),
        conv_nodes=(str(conv.region_output_id),),
        strategy="native_halo_stripe_no_ri_conv2d",
        materializer="native_halo_stripe_no_ri",
        depth=1,
        boundary_actions=("native_halo_stripe_no_ri_conv2d",),
        expected_stats={},
        executable=True,
        executor=executor,
    )
    conv.region_runtime = runtime
    conv.region_first_skip_dense_pack = True


def _ct_count_from_shape(shape: Any, *, slots: int) -> int:
    return max(1, int((int(torch.Size(shape).numel()) + int(slots) - 1) // int(slots)))


def _dense_rows_cols(module: Any) -> tuple[int, int]:
    keys = list(dict(getattr(module, "transform_ids", {}) or {}).keys())
    if not keys:
        return 0, 0
    rows = max(int(row) for row, _col in keys) + 1
    cols = max(int(col) for _row, col in keys) + 1
    return int(rows), int(cols)


def _make_cipher_source(
    *,
    count: int,
    seed: int,
    shape: torch.Size,
    fhe_shape: torch.Size,
    level: int,
) -> CipherTensor:
    ids: list[int] = []
    gen = torch.Generator().manual_seed(int(seed))
    slots = int(scheme.params.get_slots())
    for _index in range(int(count)):
        packed = torch.randn((int(slots),), generator=gen, dtype=torch.float32) * 0.01
        ct = scheme.encrypt(scheme.encode(packed, int(level)))
        ids.append(int(ct.ids[0]))
        ct.ids = []
    return CipherTensor(scheme, ids, torch.Size(shape), torch.Size(fhe_shape))


def _make_compact_source(module: Any, *, count: int, seed: int) -> CipherTensor:
    return _make_cipher_source(
        count=int(count),
        seed=int(seed),
        shape=torch.Size(getattr(module, "input_shape")),
        fhe_shape=torch.Size(getattr(module, "fhe_input_shape")),
        level=int(module.level),
    )


def _native_provider_delegate(executor: Any) -> Any:
    return getattr(executor, "delegate", executor)


def _runtime_operation_counts() -> dict[str, Any]:
    counts = bench._runtime_operation_counters()
    return {} if counts is None else dict(counts)


def _unified_group_runtime_fairness(executor: Any, *, serving_hot_s: float) -> dict[str, Any]:
    timings: list[dict[str, Any]] = []
    for group in bench._executor_unified_groups(executor):
        timing = getattr(group, "last_runtime_timing", None)
        if isinstance(timing, dict):
            timings.append(dict(timing))
    return bench._aggregate_runtime_fairness_timings(timings, serving_hot_s=float(serving_hot_s))


def _run_dense_forward(module: Any, *, source_count: int, seed: int) -> dict[str, Any]:
    module.he_mode = True
    source = _make_compact_source(module, count=int(source_count), seed=int(seed))
    counters_enabled = bench._reset_runtime_operation_counters()
    started = time.perf_counter()
    out = module(source)
    bench._synchronize_backend()
    elapsed = float(time.perf_counter() - started)
    fairness = bench._runtime_fairness_for_module(module, serving_hot_s=float(elapsed))
    counts = _runtime_operation_counts() if bool(counters_enabled) else {}
    output_count = int(len(getattr(out, "ids", []) or []))
    output_level = None
    if int(output_count) > 0 and callable(getattr(out, "level", None)):
        output_level = int(out.level())
    del out
    del source
    return {
        "elapsed_s": float(elapsed),
        "runtime_fairness_timing": dict(fairness),
        "operation_counts": dict(counts),
        "output_cts": int(output_count),
        "actual_output_level": output_level,
    }


def _run_native_provider_forward(executor: Any, *, source_count: int, seed: int) -> dict[str, Any]:
    delegate = _native_provider_delegate(executor)
    native_shape = torch.Size(delegate.runtime_native_fhe_output_shape())
    input_level = int(getattr(delegate, "assigned_level", DEFAULT_INPUT_LEVEL))
    source = _make_cipher_source(
        count=int(source_count),
        seed=int(seed),
        shape=torch.Size([int(source_count), int(scheme.params.get_slots())]),
        fhe_shape=torch.Size([int(source_count), int(scheme.params.get_slots())]),
        level=int(input_level),
    )
    counters_enabled = bench._reset_runtime_operation_counters()
    started = time.perf_counter()
    ids = tuple(int(value) for value in getattr(source, "ids", ()))
    if len(ids) < int(source_count):
        raise RuntimeError(f"native provider requires {source_count} source CTs, got {len(ids)}")

    output_blocks: list[Any | None] = [None for _ in range(int(delegate.rows))]
    evaluate_started = time.perf_counter()
    try:
        runtime_groups = [
            (int(item.input_index), item.group, tuple(int(value) for value in item.target_indices))
            for _index, item in sorted(
                enumerate(list(getattr(delegate, "runtime_groups", []) or [])),
                key=lambda pair: (int(pair[1].input_index), int(pair[0])),
            )
        ]
        if not runtime_groups:
            runtime_groups = [
                (
                    int(input_index),
                    group,
                    tuple(int(value) for value in delegate.target_indices_by_input_index[int(input_index)]),
                )
                for input_index, group in sorted(dict(delegate.groups_by_input_index).items())
            ]
        for input_index, group, target_indices in runtime_groups:
            output_ids = group.evaluate_unified(int(ids[int(input_index)]), scheme.backend)
            for target_index, output_id in zip(target_indices, output_ids):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(delegate.slots)]),
                    torch.Size([1, int(delegate.slots)]),
                )
                partial = _rescale_cipher_tensor(partial)
                if output_blocks[int(target_index)] is None:
                    output_blocks[int(target_index)] = partial
                else:
                    lhs, rhs = _align_ciphertexts_for_add(output_blocks[int(target_index)], partial)
                    output_blocks[int(target_index)] = lhs + rhs
    finally:
        release_deferred = getattr(delegate, "_release_deferred_single_slot_diagonal_caches", None)
        if callable(release_deferred):
            release_deferred()
    delegate.last_runtime_timing["evaluate_unified_s"] = float(time.perf_counter() - evaluate_started)

    postprocess_started = time.perf_counter()
    output_ids: list[int] = []
    for block_index, block_ct in enumerate(output_blocks):
        if block_ct is None:
            raise RuntimeError(f"missing native halo output block {block_index}")
        block_ct = delegate._add_bias(block_ct, block_index=int(block_index))
        block_ct.set_scale(int(scheme.params.get_default_scale()))
        output_ids.append(int(block_ct.ids[0]))
        block_ct.ids = []
    native_output = CipherTensor(scheme, output_ids, native_shape, native_shape)
    delegate.last_runtime_timing["postprocess_s"] = float(time.perf_counter() - postprocess_started)
    delegate.last_runtime_timing["input_relayout_s"] = 0.0
    delegate.last_runtime_timing["output_relayout_s"] = 0.0
    bench._synchronize_backend()
    elapsed = float(time.perf_counter() - started)
    fairness = _unified_group_runtime_fairness(executor, serving_hot_s=float(elapsed))
    counts = _runtime_operation_counts() if bool(counters_enabled) else {}
    output_count = int(len(getattr(native_output, "ids", []) or []))
    output_level = None
    if int(output_count) > 0 and callable(getattr(native_output, "level", None)):
        output_level = int(native_output.level())
    del native_output
    del source
    return {
        "elapsed_s": float(elapsed),
        "runtime_fairness_timing": dict(fairness),
        "operation_counts": dict(counts),
        "output_cts": int(output_count),
        "actual_output_level": output_level,
        "executor_last_runtime_timing": dict(getattr(delegate, "last_runtime_timing", {}) or {}),
    }


def _metadata_for_native_executor(executor: Any) -> dict[str, Any]:
    get_metadata = getattr(executor, "compile_cache_metadata", None)
    return dict(get_metadata()) if callable(get_metadata) else {}


def _stream_encode_s(timing: dict[str, Any]) -> float:
    return float(timing.get("stream_build_map_s") or 0.0) + float(timing.get("stream_encode_hoist_s") or 0.0) + float(
        timing.get("stream_load_payload_s") or 0.0
    )


def _lt_accumulate_s(timing: dict[str, Any]) -> float:
    total = (
        float(timing.get("stream_eval_s") or 0.0)
        + float(timing.get("stream_accumulate_s") or 0.0)
        + float(timing.get("cpp_baby_step_s") or 0.0)
        + float(timing.get("cpp_giant_step_s") or 0.0)
    )
    if total <= 0.0:
        total = float(timing.get("eval_s") or timing.get("eval_total_s") or 0.0)
    return float(total)


def _maxrss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _run_row(
    row: ConvKernelRow,
    *,
    backend: str,
    repeats: int,
    seed: int,
    clip_provider_boundary_halo: bool = False,
    provider_output_layout: str = "tight_compact",
) -> dict[str, Any]:
    started = time.perf_counter()
    if str(backend) != "python":
        bench._require_backend(str(backend))
    _init_scheme(backend=str(backend))
    compile_stdout = ""
    try:
        conv = _make_conv(row, seed=int(seed))
        input_halo_top, input_halo_bottom = _apply_provider_input_halo(
            conv,
            row,
            clip_boundary_halo=bool(clip_provider_boundary_halo and row.path == "provider"),
        )
        applied_provider_output_layout = _apply_provider_output_layout(
            conv,
            row,
            output_layout=str(provider_output_layout),
        )
        output_halo_top, output_halo_bottom = _expected_provider_output_halo(
            row,
            output_layout=str(provider_output_layout),
        )
        if row.path == "provider":
            _attach_provider_runtime(conv, row)
        with _capture_stdout() as buffer:
            generate_started = time.perf_counter()
            conv.generate_diagonals(last=False)
            generate_diagonals_s = float(time.perf_counter() - generate_started)
            compile_started = time.perf_counter()
            conv.compile()
            compile_backend_s = float(time.perf_counter() - compile_started)
            compile_stdout = buffer.getvalue()
        compile_load_profile = dict(getattr(scheme.lt_evaluator, "get_compile_load_profile", lambda: {})() or {})

        slots = int(scheme.params.get_slots())
        compact_input_cts = _ct_count_from_shape(conv.fhe_input_shape, slots=int(slots))
        compact_output_cts = _ct_count_from_shape(conv.fhe_output_shape, slots=int(slots))
        if row.path == "dense":
            dense_rows, dense_cols = _dense_rows_cols(conv)
            rotation_stats = bench._linear_transform_rotation_stats(conv)
            source_count = int(dense_cols)
            input_cts = int(dense_cols)
            output_cts = int(dense_rows)
            provider_metadata: dict[str, Any] = {}
            provider_diag_builder_metadata: dict[str, Any] = {}
            provider_lt_grouping_mode = ""
        else:
            executor = getattr(conv.region_runtime, "executor")
            provider_metadata = _metadata_for_native_executor(executor)
            provider_diag_builder_metadata = {
                key: value
                for key, value in dict(getattr(executor, "last_runtime_timing", {}) or {}).items()
                if str(key).startswith("diag_builder_")
            }
            provider_lt_grouping_mode = _normalise_provider_lt_grouping_mode(
                provider_metadata.get("provider_lt_grouping_mode")
                or provider_metadata.get("lt_grouping_mode")
                or row.provider_lt_grouping_mode
            )
            native_plan = dict(
                provider_metadata.get("native_halo_conv2d_plan", {})
                or provider_metadata.get("r34_native_aligned_halo_plan", {})
                or {}
            )
            rotation_stats = bench._provider_rotation_stats(conv, path_kind="provider")
            source_count = int(native_plan.get("input_ct_count", getattr(executor, "cols", 0)) or 0)
            input_cts = int(source_count)
            output_cts = int(
                provider_metadata.get("runtime_output_ct_count")
                or native_plan.get("output_ct_count", getattr(executor, "rows", 0))
                or 0
            )
            provider_output_storage_layout = str(
                provider_metadata.get("runtime_output_storage_layout")
                or native_plan.get("output_storage_layout")
                or ""
            )
        diag_builder_metadata = dict(getattr(conv, "_diag_builder_metadata", {}) or {})
        if row.path == "provider":
            diag_builder_metadata.update(provider_diag_builder_metadata)

        runs: list[dict[str, Any]] = []
        for index in range(int(repeats)):
            run_seed = int(seed) + 50_000 + int(index) * 1009
            if row.path == "dense":
                run_payload = _run_dense_forward(conv, source_count=int(source_count), seed=int(run_seed))
            else:
                run_payload = _run_native_provider_forward(
                    getattr(conv.region_runtime, "executor"),
                    source_count=int(source_count),
                    seed=int(run_seed),
                )
            runs.append(dict(run_payload))

        first_counts = dict((runs[0].get("operation_counts") if runs else {}) or {})
        timing = dict((runs[0].get("runtime_fairness_timing") if runs else {}) or {})
        rotation_runtime = first_counts.get("rotation")
        actual_output_level = None if not runs else runs[0].get("actual_output_level")
        if actual_output_level is None:
            raise RuntimeError("worker did not record actual output ciphertext level")
        if int(actual_output_level) != int(row.expected_output_level):
            raise RuntimeError(
                f"actual output level {int(actual_output_level)} != expected {int(row.expected_output_level)} "
                f"for input level {int(row.input_level)}"
            )
        result = {
            "status": "ok",
            "created_at_utc": _now_utc(),
            "backend": str(backend),
            "row_id": row.row_id,
            "case_id": str(row.case_id or ""),
            "hw": row.hw,
            "logical_hw": row.logical_hw,
            "channels": int(row.channels),
            "conv": f"{int(row.channels)},{int(row.channels)}",
            "kernel_label": row.kernel_name,
            "input_level": int(row.input_level),
            "expected_output_level": int(row.expected_output_level),
            "actual_output_level": int(actual_output_level),
            "ckks_profile": CKKS_PROFILE_ID,
            "ckks_params": {
                "LogN": 16,
                "LogQ": list(E2E_LOGQ),
                "LogP": list(E2E_LOGP),
                "LogScale": 40,
                "H": 192,
            },
            "input_gap": int(row.stage_gap),
            "output_gap": int(row.stage_gap),
            "channel_group_size": int(row.channel_group_size),
            "logical_chw": row.logical_chw,
            "packed_chw": row.packed_chw,
            "variant": str(row.variant),
            "variant_label": VARIANT_LABELS[str(row.variant)],
            "path": row.path,
            "provider_requested_input_halo": row.halo,
            "provider_lt_grouping_mode": provider_lt_grouping_mode,
            "provider_disable_shared_rotation": bool(provider_lt_grouping_mode == "individual"),
            "native_halo_channel_fold_mode": (
                str(provider_metadata.get("native_halo_channel_fold_mode") or row.native_halo_channel_fold_mode)
                if row.path == "provider"
                else ""
            ),
            "provider_boundary_halo_clipped": bool(clip_provider_boundary_halo and row.path == "provider"),
            "provider_output_layout": applied_provider_output_layout,
            "input_halo_top": input_halo_top,
            "input_halo_bottom": input_halo_bottom,
            "output_halo_top": output_halo_top if row.path == "provider" else None,
            "output_halo_bottom": output_halo_bottom if row.path == "provider" else None,
            "kernel": "3x3/pad1/stride1",
            "compile_s": float(generate_diagonals_s + compile_backend_s),
            "generate_diagonals_s": float(generate_diagonals_s),
            "compile_backend_s": float(compile_backend_s),
            "compile_load_profile": dict(compile_load_profile),
            "diag_builder_kind": str(
                diag_builder_metadata.get("diag_builder_kind")
                or compile_load_profile.get("diag_builder_kind")
                or ""
            ),
            "diag_builder_source": str(
                diag_builder_metadata.get("diag_builder_source")
                or ""
            ),
            "diag_builder_build_s": float(
                diag_builder_metadata.get("diag_builder_build_s")
                or compile_load_profile.get("diag_builder_build_s")
                or 0.0
            ),
            "diag_builder_shadow_s": float(
                diag_builder_metadata.get("diag_builder_shadow_s")
                or compile_load_profile.get("diag_builder_shadow_s")
                or 0.0
            ),
            "diag_builder_payload_count": int(
                float(diag_builder_metadata.get("diag_builder_payload_count") or 0.0)
            ),
            "diag_builder_fallback_reason": str(
                diag_builder_metadata.get("diag_builder_fallback_reason")
                or ""
            ),
            "run_count": int(repeats),
            "hot_run_s": [float(item.get("elapsed_s") or 0.0) for item in runs],
            "hot_run_mean_s": (
                sum(float(item.get("elapsed_s") or 0.0) for item in runs) / max(1, int(len(runs)))
            ),
            "runtime_fairness_timing": dict(timing),
            "stream_encode_s": float(_stream_encode_s(timing)),
            "lt_accumulate_s": float(_lt_accumulate_s(timing)),
            "runtime_mode": str(timing.get("runtime_fairness_mode", "")),
            "rotation_eval_count": int(rotation_runtime if rotation_runtime is not None else rotation_stats.get("rotation_eval_count", 0)),
            "planned_rotation_eval_count": int(rotation_stats.get("rotation_eval_count", 0) or 0),
            "runtime_operation_counts": {
                "available": bool(first_counts),
                "first_run": dict(first_counts),
            },
            "input_cts": int(input_cts),
            "output_cts": int(output_cts),
            "compact_input_cts": int(compact_input_cts),
            "compact_output_cts": int(compact_output_cts),
            "module": {
                "input_shape": [int(value) for value in tuple(conv.input_shape)],
                "output_shape": [int(value) for value in tuple(conv.output_shape)],
                "input_gap": int(getattr(conv, "input_gap", 1)),
                "output_gap": int(getattr(conv, "output_gap", 1)),
                "fhe_input_shape": [int(value) for value in tuple(conv.fhe_input_shape)],
                "fhe_output_shape": [int(value) for value in tuple(conv.fhe_output_shape)],
            },
            "provider_metadata": provider_metadata,
            "provider_output_storage_layout": provider_output_storage_layout if row.path == "provider" else "",
            "rotation_stats": rotation_stats,
            "compile_stdout_tail": str(compile_stdout[-12000:]),
            "maxrss_bytes": int(_maxrss_bytes()),
            "elapsed_s": float(time.perf_counter() - started),
            "env": _env_snapshot(),
        }
        return result
    finally:
        _cleanup_scheme()


def _env_snapshot() -> dict[str, str]:
    keys = [
        "GOMAXPROCS",
        "MALLOC_ARENA_MAX",
        "ORION_COMPILE_PARALLEL_POLICY",
        "ORION_SINGLE_SLOT_ENCODE_WORKERS",
        "ORION_LATTIGO_STREAMING_LT",
        "ORION_UNIFIED_STREAM_COMPILE_IO_NONE",
        "ORION_LATTIGO_MEMORY_BOUNDED_COMPILE",
        "ORION_LATTIGO_MEMORY_BOUNDED_EVAL",
        "ORION_SINGLE_SLOT_LAYER_CACHE",
        "ORION_PACK_CONV_WORKERS",
        "ORION_DIRECT_PACK_WORKERS",
        "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS",
        "ORION_LT_COMPILE_WORKERS",
        "ORION_UNIFIED_COMPILE_WORKERS",
        "ORION_LATTIGO_COMPILE_WORKERS",
        "ORION_DENSE_LT_COMPILE_BATCH_TRANSFORMS",
        "ORION_UNIFIED_LT_INDIVIDUAL_EVAL",
        "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS",
        "ORION_LATTIGO_UNIFIED_NO_BSGS",
        "ORION_CPP_DIAG_BUILDER",
        "ORION_CPP_DIAG_BUILDER_DENSE",
        "ORION_CPP_DIAG_BUILDER_PROVIDER",
        "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE",
        "ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_SOURCE",
        "ORION_CPP_DIAG_BUILDER_STRICT",
        "ORION_CPP_DIAG_BUILDER_SHADOW",
        "ORION_DIAG_BUILDER_LIB",
    ]
    return {key: str(os.environ.get(key, "")) for key in keys}


def _apply_env_defaults(env: dict[str, str]) -> dict[str, str]:
    updated = dict(env)
    cpu_count = max(1, int(os.cpu_count() or 1))
    updated.setdefault("PYTHONUNBUFFERED", "1")
    updated.setdefault("MALLOC_ARENA_MAX", "2")
    updated["GOMAXPROCS"] = "1"
    updated["ORION_COMPILE_PARALLEL_POLICY"] = "manual"
    updated["ORION_LATTIGO_STREAMING_LT"] = "0"
    updated["ORION_UNIFIED_STREAM_COMPILE_IO_NONE"] = "0"
    updated["ORION_LATTIGO_MEMORY_BOUNDED_COMPILE"] = "0"
    updated["ORION_LATTIGO_MEMORY_BOUNDED_EVAL"] = "0"
    updated["ORION_SINGLE_SLOT_LAYER_CACHE"] = "0"
    updated.setdefault("ORION_SINGLE_SLOT_ENCODE_WORKERS", str(cpu_count))
    updated.setdefault("ORION_PACK_CONV_WORKERS", str(cpu_count))
    updated.setdefault("ORION_DIRECT_PACK_WORKERS", str(cpu_count))
    updated.setdefault("ORION_LT_COMPILE_WORKERS", str(cpu_count))
    updated.setdefault("ORION_UNIFIED_COMPILE_WORKERS", str(cpu_count))
    updated.setdefault("ORION_LATTIGO_COMPILE_WORKERS", str(cpu_count))
    updated.setdefault("ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS", str(cpu_count))
    updated.setdefault("ORION_CONCAT_FUSION", "0")
    updated["ORION_UNIFIED_LT_INDIVIDUAL_EVAL"] = "1"
    updated["ORION_UNIFIED_LT_SHARED_ROTATION_KEYS"] = "0"
    updated["ORION_LATTIGO_UNIFIED_NO_BSGS"] = "0"
    return updated


def _rows_from_args(args: argparse.Namespace) -> list[ConvKernelRow]:
    hw_values = [_parse_hw(str(value)) for value in args.hw]
    rows: list[ConvKernelRow] = []
    explicit_cases = [str(value) for value in getattr(args, "kernel_cases", []) or []]
    if explicit_cases:
        case_specs: list[KernelCase] = []
        for case_id in explicit_cases:
            if case_id not in KERNEL_CASES:
                choices = ", ".join(sorted(KERNEL_CASES))
                raise ValueError(f"unknown kernel case {case_id!r}; choose from: {choices}")
            case_specs.append(KERNEL_CASES[str(case_id)])
    else:
        channel_stage_gaps = {32: 1, 64: 2, 128: 4, 256: 8}
        case_specs = [
            KernelCase(
                case_id=f"conv{int(channels)}",
                label=f"Conv {int(channels)},{int(channels)}",
                channels=int(channels),
                stage_gap=channel_stage_gaps[int(channels)],
            )
            for channels in [int(value) for value in args.channels]
        ]
    for case in case_specs:
        for height, width in hw_values:
            for variant in [str(value) for value in args.variants]:
                rows.append(
                    ConvKernelRow(
                        channels=int(case.channels),
                        height=int(height),
                        width=int(width),
                        variant=str(variant),
                        input_level=int(args.input_level),
                        case_id=str(case.case_id),
                        stage_gap_override=int(case.stage_gap),
                        kernel_label=str(case.label),
                    )
                )
    return rows


def _table_payload(
    run_root: Path,
    rows: list[ConvKernelRow],
    *,
    provider_output_layout: str = "tight_compact",
) -> list[list[str]]:
    table_rows: list[list[str]] = []
    for row in rows:
        result_path = _row_path(Path(run_root), row)
        payload = _read_json(result_path)
        status = "pending" if payload is None else str(payload.get("status", "unknown"))
        note = ""
        output_mismatch = (
            _provider_output_halo_mismatch(row, payload, provider_output_layout=str(provider_output_layout))
            if isinstance(payload, dict)
            else None
        )
        if output_mismatch is not None:
            actual_top, actual_bottom, expected_top, expected_bottom = output_mismatch
            status = "stale"
            note = f"output halo {actual_top}/{actual_bottom}; expected {expected_top}/{expected_bottom}"
        if isinstance(payload, dict) and status != "ok":
            note = note or str(payload.get("failure_kind") or payload.get("error") or payload.get("message") or "")[:120]
        elif isinstance(payload, dict) and payload.get("reused_from"):
            source = str(payload.get("reused_from"))
            source_file, _sep, source_detail = source.partition(":")
            note = f"reused: {Path(source_file).name}"
            if source_detail:
                note = f"{note}:{source_detail}"
            note = note[:120]
        halo_cell = ""
        if row.halo is not None:
            top = (payload or {}).get("input_halo_top") if isinstance(payload, dict) else row.halo
            bottom = (payload or {}).get("input_halo_bottom") if isinstance(payload, dict) else row.halo
            requested = (payload or {}).get("provider_requested_input_halo") if isinstance(payload, dict) else row.halo
            clipped = bool((payload or {}).get("provider_boundary_halo_clipped")) if isinstance(payload, dict) else False
            if clipped:
                halo_cell = f"{int(top)}/{int(bottom)} (beta={int(requested)})"
            elif top is not None and bottom is not None:
                halo_cell = f"{int(top)}/{int(bottom)}"
            else:
                halo_cell = f"{int(row.halo)}/{int(row.halo)}"
        output_layout = _provider_result_output_layout(payload) if isinstance(payload, dict) else ""
        channel_fold = ""
        if row.path == "provider":
            channel_fold = (
                str((payload or {}).get("native_halo_channel_fold_mode") or row.native_halo_channel_fold_mode)
                if isinstance(payload, dict)
                else row.native_halo_channel_fold_mode
            )
        lt_grouping = ""
        if row.path == "provider":
            lt_grouping = (
                str((payload or {}).get("provider_lt_grouping_mode") or row.provider_lt_grouping_mode)
                if isinstance(payload, dict)
                else row.provider_lt_grouping_mode
            )
        table_rows.append(
            [
                row.hw,
                row.kernel_name,
                row.logical_chw,
                str(int(row.stage_gap)),
                str(int(row.channel_group_size)),
                row.packed_chw,
                VARIANT_LABELS[str(row.variant)],
                status,
                str(int(row.input_level)),
                str(int(row.expected_output_level)),
                _fmt_int((payload or {}).get("actual_output_level") if isinstance(payload, dict) else None),
                halo_cell,
                output_layout,
                channel_fold,
                lt_grouping,
                _fmt_int((payload or {}).get("rotation_eval_count") if isinstance(payload, dict) else None),
                _fmt_float((payload or {}).get("lt_accumulate_s") if isinstance(payload, dict) else None),
                _fmt_float((payload or {}).get("hot_run_mean_s") if isinstance(payload, dict) else None),
                _fmt_float((payload or {}).get("compile_s") if isinstance(payload, dict) else None),
                _fmt_float((payload or {}).get("diag_builder_build_s") if isinstance(payload, dict) else None),
                _fmt_float((payload or {}).get("diag_builder_shadow_s") if isinstance(payload, dict) else None),
                _fmt_int((payload or {}).get("input_cts") if isinstance(payload, dict) else None),
                _fmt_int((payload or {}).get("output_cts") if isinstance(payload, dict) else None),
                _gib_from_bytes((payload or {}).get("maxrss_bytes") if isinstance(payload, dict) else None),
                str((payload or {}).get("runtime_mode", "") if isinstance(payload, dict) else ""),
                str(result_path),
                note,
            ]
        )
    return table_rows


def _markdown_table(rows: list[list[str]]) -> str:
    headers = [
        "HW",
        "kernel",
        "logical input",
        "multiplex",
        "channels/group",
        "packed FHE input",
        "path / beta",
        "status",
        "input level",
        "expected output level",
        "actual output level",
        "input halo T/B",
        "output layout",
        "channel fold",
        "LT grouping",
        "rotations",
        "LT+accumulate s",
        "hot run s",
        "compile s",
        "diag build s",
        "diag shadow s",
        "input ct",
        "output ct",
        "peak RSS GiB",
        "runtime mode",
        "result file",
        "note",
    ]
    aligns = [
        "---",
        "---",
        "---",
        "---",
        "---",
        "---",
        "---",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---",
        "---",
        "---",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---",
        "---",
        "---",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def update_doc(
    doc_path: Path,
    run_root: Path,
    rows: list[ConvKernelRow],
    *,
    provider_output_layout: str = "tight_compact",
) -> None:
    table = _markdown_table(_table_payload(Path(run_root), rows, provider_output_layout=str(provider_output_layout)))
    text = Path(doc_path).read_text(encoding="utf-8")
    Path(doc_path).write_text(_replace_block(text, DOC_MARKER, table), encoding="utf-8")


def _write_summary_csv(
    run_root: Path,
    rows: list[ConvKernelRow],
    *,
    provider_output_layout: str = "tight_compact",
) -> None:
    csv_path = Path(run_root) / "conv_kernel_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "HW",
        "kernel",
        "logical_input",
        "multiplex",
        "channels_per_group",
        "packed_fhe_input",
        "path_beta",
        "status",
        "input_level",
        "expected_output_level",
        "actual_output_level",
        "input_halo_tb",
        "output_layout",
        "channel_fold",
        "lt_grouping",
        "rotations",
        "lt_accumulate_s",
        "hot_run_s",
        "compile_s",
        "diag_build_s",
        "diag_shadow_s",
        "input_ct",
        "output_ct",
        "peak_rss_gib",
        "runtime_mode",
        "result_file",
        "note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(_table_payload(Path(run_root), rows, provider_output_layout=str(provider_output_layout)))


def _process_rss_bytes(pid: int) -> int | None:
    try:
        with Path(f"/proc/{int(pid)}/status").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def _write_running_placeholder(path: Path, row: ConvKernelRow, args: argparse.Namespace) -> None:
    input_halo_top, input_halo_bottom = _effective_provider_input_halo(
        row,
        clip_boundary_halo=bool(getattr(args, "clip_provider_boundary_halo", False) and row.path == "provider"),
    )
    provider_output_layout = ""
    output_halo_top, output_halo_bottom = None, None
    if row.path == "provider":
        provider_output_layout = "native_halo_stripe" if str(args.provider_output_layout) == "native_stripe" else "tight_compact"
        output_halo_top, output_halo_bottom = _expected_provider_output_halo(
            row,
            output_layout=str(args.provider_output_layout),
        )
    _write_json(
        path,
        {
            "status": "running",
            "created_at_utc": _now_utc(),
            "row_id": row.row_id,
            "case_id": str(row.case_id or ""),
            "hw": row.hw,
            "logical_hw": row.logical_hw,
            "channels": int(row.channels),
            "conv": f"{int(row.channels)},{int(row.channels)}",
            "kernel_label": row.kernel_name,
            "input_level": int(row.input_level),
            "expected_output_level": int(row.expected_output_level),
            "ckks_profile": CKKS_PROFILE_ID,
            "input_gap": int(row.stage_gap),
            "output_gap": int(row.stage_gap),
            "channel_group_size": int(row.channel_group_size),
            "logical_chw": row.logical_chw,
            "packed_chw": row.packed_chw,
            "variant": str(row.variant),
            "variant_label": VARIANT_LABELS[str(row.variant)],
            "path": row.path,
            "provider_requested_input_halo": row.halo,
            "provider_lt_grouping_mode": row.provider_lt_grouping_mode if row.path == "provider" else "",
            "provider_disable_shared_rotation": bool(row.provider_disable_shared_rotation and row.path == "provider"),
            "native_halo_channel_fold_mode": row.native_halo_channel_fold_mode if row.path == "provider" else "",
            "provider_boundary_halo_clipped": bool(
                getattr(args, "clip_provider_boundary_halo", False) and row.path == "provider"
            ),
            "provider_output_layout": provider_output_layout,
            "provider_output_storage_layout": provider_output_layout,
            "input_halo_top": input_halo_top,
            "input_halo_bottom": input_halo_bottom,
            "output_halo_top": output_halo_top,
            "output_halo_bottom": output_halo_bottom,
            "ckks_params": {
                "LogN": 16,
                "LogQ": list(E2E_LOGQ),
                "LogP": list(E2E_LOGP),
                "LogScale": 40,
                "H": 192,
            },
            "kernel": "3x3/pad1/stride1",
            "run_root": str(args.run_root),
            "env": _env_snapshot(),
        },
    )


def _mark_worker_failure(path: Path, row: ConvKernelRow, *, failure_kind: str, message: str, return_code: int | None) -> None:
    payload = _read_json(path) or {}
    input_halo_top = payload.get("input_halo_top", row.halo)
    input_halo_bottom = payload.get("input_halo_bottom", row.halo)
    provider_output_layout = payload.get("provider_output_layout", "")
    provider_output_storage_layout = payload.get("provider_output_storage_layout", provider_output_layout)
    output_halo_top = payload.get("output_halo_top")
    output_halo_bottom = payload.get("output_halo_bottom")
    if row.path == "provider" and (output_halo_top is None or output_halo_bottom is None):
        output_halo_top, output_halo_bottom = _expected_provider_output_halo(
            row,
            output_layout=str(provider_output_layout or provider_output_storage_layout),
        )
    payload.update(
        {
            "status": "error",
            "finished_at_utc": _now_utc(),
            "row_id": row.row_id,
            "case_id": str(row.case_id or ""),
            "hw": row.hw,
            "logical_hw": row.logical_hw,
            "channels": int(row.channels),
            "conv": f"{int(row.channels)},{int(row.channels)}",
            "kernel_label": row.kernel_name,
            "input_level": int(row.input_level),
            "expected_output_level": int(row.expected_output_level),
            "ckks_profile": CKKS_PROFILE_ID,
            "input_gap": int(row.stage_gap),
            "output_gap": int(row.stage_gap),
            "channel_group_size": int(row.channel_group_size),
            "logical_chw": row.logical_chw,
            "packed_chw": row.packed_chw,
            "variant": str(row.variant),
            "variant_label": VARIANT_LABELS[str(row.variant)],
            "path": row.path,
            "provider_requested_input_halo": row.halo,
            "provider_lt_grouping_mode": row.provider_lt_grouping_mode if row.path == "provider" else "",
            "provider_disable_shared_rotation": bool(row.provider_disable_shared_rotation and row.path == "provider"),
            "native_halo_channel_fold_mode": row.native_halo_channel_fold_mode if row.path == "provider" else "",
            "provider_output_layout": provider_output_layout,
            "provider_output_storage_layout": provider_output_storage_layout,
            "input_halo_top": input_halo_top,
            "input_halo_bottom": input_halo_bottom,
            "output_halo_top": output_halo_top if row.path == "provider" else None,
            "output_halo_bottom": output_halo_bottom if row.path == "provider" else None,
            "ckks_params": {
                "LogN": 16,
                "LogQ": list(E2E_LOGQ),
                "LogP": list(E2E_LOGP),
                "LogScale": 40,
                "H": 192,
            },
            "failure_kind": str(failure_kind),
            "message": str(message),
            "return_code": None if return_code is None else int(return_code),
        }
    )
    _write_json(path, payload)


def run_all(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    rows = _rows_from_args(args)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / ROW_DIR_NAME).mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.parent.mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.write_text(str(run_root) + "\n", encoding="utf-8")

    env = _apply_env_defaults(os.environ)
    manifest = {
        "status": "running",
        "created_at_utc": _now_utc(),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "run_root": str(run_root),
        "doc": str(args.doc),
        "backend": str(args.backend),
        "ckks_profile": CKKS_PROFILE_ID,
        "ckks_params": {
            "LogN": 16,
            "LogQ": list(E2E_LOGQ),
            "LogP": list(E2E_LOGP),
            "LogScale": 40,
            "H": 192,
            "RingType": "standard",
        },
        "input_level": int(args.input_level),
        "expected_output_level": int(args.input_level) - int(CONV_KERNEL_DEPTH),
        "channels": [int(value) for value in args.channels],
        "kernel_cases": [str(value) for value in getattr(args, "kernel_cases", []) or []],
        "hw": [str(value) for value in args.hw],
        "variants": [str(value) for value in args.variants],
        "repeats": int(args.repeats),
        "rss_cap_gib": float(args.max_worker_rss_gb),
        "clip_provider_boundary_halo": bool(args.clip_provider_boundary_halo),
        "provider_output_layout": str(args.provider_output_layout),
        "measurement": {
            "io_mode": "none",
            "lt_accumulate_s": "resident eval_s / eval_total_s, excluding artifact I/O",
            "level_policy": "input level 2 is the main kernel benchmark level; Conv2d depth 1 yields output level 1",
            "runtime_parallelism": "GOMAXPROCS=1 during kernel evaluation; compile and diagonal encode worker pools may use host CPU count",
            "row_scheduling": "parent runner executes one worker process at a time; no concurrent LT rows",
            "orion": "dense Conv2d path with default resident Lattigo LT",
            "variant_scope": "current AMD kernel table uses dense, beta=1 no-sharing, and beta=1 no-sharing stripe only; beta=2 is intentionally excluded",
            "provider_halo1": "legacy native halo provider beta=1 with shared rotation grouping and heuristic channel fold",
            "provider_halo2": "legacy native halo provider beta=2 with shared rotation grouping and heuristic channel fold",
            "provider_halo1_no_share": (
                "native halo provider beta=1; no shared rotations; heuristic native halo channel fold; BSGS preserved within each individual LT"
            ),
            "provider_halo1_individual_lt": (
                "native halo provider beta=1; no shared rotations; per-stripe native halo channel fold; BSGS preserved within each individual LT"
            ),
            "provider_halo2_individual_lt": (
                "native halo provider beta=2; no shared rotations; per-stripe native halo channel fold; BSGS preserved within each individual LT"
            ),
            "u_net_stage_packing": {
                "Conv 32,32": "logical 32xHorigxWorig, multiplex/input_gap/output_gap=1, packed FHE 32xHorigxWorig",
                "Conv 64,64": "logical 64x(Horig/2)x(Worig/2), multiplex/input_gap/output_gap=2, 4 channels/group, packed FHE 16xHorigxWorig",
                "Conv 128,128": "logical 128x(Horig/4)x(Worig/4), multiplex/input_gap/output_gap=4, 16 channels/group, packed FHE 8xHorigxWorig",
                "Conv 256,256": "logical 256x(Horig/8)x(Worig/8), multiplex/input_gap/output_gap=8, 64 channels/group, packed FHE 4xHorigxWorig",
                "dec1b Conv 32,32": "logical 32xHorigxWorig, multiplex/input_gap/output_gap=1, packed FHE 32xHorigxWorig",
                "dec2b Conv 64,64": "logical 64x(Horig/2)x(Worig/2), multiplex/input_gap/output_gap=2, 4 channels/group, packed FHE 16xHorigxWorig",
                "dec3b Conv 128,128": "logical 128x(Horig/4)x(Worig/4), multiplex/input_gap/output_gap=4, 16 channels/group, packed FHE 8xHorigxWorig",
                "dec4b Conv 256,256": "logical 256x(Horig/8)x(Worig/8), multiplex/input_gap/output_gap=8, 64 channels/group, packed FHE 4xHorigxWorig",
                "bottleneckb Conv 512,512": "logical 512x(Horig/16)x(Worig/16), multiplex/input_gap/output_gap=16, 256 channels/group, packed FHE 2xHorigxWorig",
            },
        },
        "env": {key: str(env.get(key, "")) for key in sorted(_env_snapshot())},
    }
    _write_json(run_root / "manifest.json", manifest)
    reused_count = reuse_existing_rows(
        run_root,
        rows,
        [Path(value) for value in getattr(args, "reuse_from", [])],
        force=bool(args.force),
        clip_provider_boundary_halo=bool(args.clip_provider_boundary_halo),
        provider_output_layout=str(args.provider_output_layout),
    )
    if reused_count:
        manifest["reused_rows"] = int(reused_count)
        _write_json(run_root / "manifest.json", manifest)
    update_doc(Path(args.doc), run_root, rows, provider_output_layout=str(args.provider_output_layout))
    _write_summary_csv(run_root, rows, provider_output_layout=str(args.provider_output_layout))
    if bool(getattr(args, "prepare_only", False)):
        manifest["status"] = "prepared"
        manifest["prepared_at_utc"] = _now_utc()
        _write_json(run_root / "manifest.json", manifest)
        return 0

    active: subprocess.Popen[str] | None = None

    def _terminate(_signum: int, _frame: Any) -> None:
        if active is not None and active.poll() is None:
            active.terminate()
        raise KeyboardInterrupt

    old_int = signal.signal(signal.SIGINT, _terminate)
    old_term = signal.signal(signal.SIGTERM, _terminate)
    try:
        for row in rows:
            result_path = _row_path(run_root, row)
            if not bool(args.force):
                payload = _read_json(result_path)
                if isinstance(payload, dict) and payload.get("status") == "ok":
                    if row.path == "provider":
                        expected_output_top, expected_output_bottom = _expected_provider_output_halo(
                            row,
                            output_layout=str(args.provider_output_layout),
                        )
                        output_top, output_bottom = _explicit_output_halo(payload)
                        if output_top is None or output_bottom is None:
                            output_top = output_bottom = 0
                        if (int(output_top), int(output_bottom)) != (
                            int(expected_output_top or 0),
                            int(expected_output_bottom or 0),
                        ):
                            print(
                                f"[{datetime.now().isoformat(timespec='seconds')}] stale output halo; rerun {row.row_id}",
                                flush=True,
                            )
                        else:
                            print(f"[{datetime.now().isoformat(timespec='seconds')}] skip {row.row_id}", flush=True)
                            continue
                    else:
                        print(f"[{datetime.now().isoformat(timespec='seconds')}] skip {row.row_id}", flush=True)
                        continue
            _write_running_placeholder(result_path, row, args)
            update_doc(Path(args.doc), run_root, rows, provider_output_layout=str(args.provider_output_layout))
            _write_summary_csv(run_root, rows, provider_output_layout=str(args.provider_output_layout))
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-one",
                "--backend",
                str(args.backend),
                "--channels",
                str(row.channels),
                "--kernel-cases",
                str(row.case_id or f"conv{int(row.channels)}"),
                "--hw",
                row.hw,
                "--variants",
                str(row.variant),
                "--repeats",
                str(args.repeats),
                "--seed",
                str(args.seed),
                "--out",
                str(result_path),
                "--input-level",
                str(args.input_level),
                "--provider-output-layout",
                str(args.provider_output_layout),
            ]
            if bool(args.clip_provider_boundary_halo):
                command.append("--clip-provider-boundary-halo")
            else:
                command.append("--keep-provider-boundary-halo")
            log_path = result_path.with_suffix(".log")
            print(f"[{datetime.now().isoformat(timespec='seconds')}] start {row.row_id}", flush=True)
            print(" ".join(shlex.quote(part) for part in command), flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                active = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                rss_cap = int(float(args.max_worker_rss_gb) * (1024**3)) if float(args.max_worker_rss_gb) > 0 else 0
                killed_for_rss = False
                peak_rss = 0
                while active.poll() is None:
                    rss = _process_rss_bytes(int(active.pid))
                    if rss is not None:
                        peak_rss = max(int(peak_rss), int(rss))
                    if rss_cap > 0 and rss is not None and int(rss) > int(rss_cap):
                        killed_for_rss = True
                        active.terminate()
                        time.sleep(5)
                        if active.poll() is None:
                            active.kill()
                        break
                    time.sleep(float(args.poll_interval_s))
                return_code = active.wait()
            active = None
            if killed_for_rss:
                _mark_worker_failure(
                    result_path,
                    row,
                    failure_kind="rss_cap_exceeded",
                    message=f"worker RSS exceeded {float(args.max_worker_rss_gb):.1f} GiB; peak observed {_gib_from_bytes(peak_rss)} GiB",
                    return_code=return_code,
                )
            elif int(return_code) != 0:
                tail = ""
                try:
                    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
                except OSError:
                    pass
                _mark_worker_failure(
                    result_path,
                    row,
                    failure_kind="worker_failed",
                    message=tail[-2000:],
                    return_code=return_code,
                )
            update_doc(Path(args.doc), run_root, rows, provider_output_layout=str(args.provider_output_layout))
            _write_summary_csv(run_root, rows, provider_output_layout=str(args.provider_output_layout))
            if int(return_code) != 0 and bool(args.fail_fast):
                manifest["status"] = "error"
                manifest["finished_at_utc"] = _now_utc()
                _write_json(run_root / "manifest.json", manifest)
                return int(return_code) if int(return_code) != 0 else 1
        manifest["status"] = "finished"
        manifest["finished_at_utc"] = _now_utc()
        _write_json(run_root / "manifest.json", manifest)
        update_doc(Path(args.doc), run_root, rows, provider_output_layout=str(args.provider_output_layout))
        _write_summary_csv(run_root, rows, provider_output_layout=str(args.provider_output_layout))
        return 0
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def run_one(args: argparse.Namespace) -> int:
    row = _rows_from_args(args)[0]
    try:
        payload = _run_row(
            row,
            backend=str(args.backend),
            repeats=int(args.repeats),
            seed=int(args.seed),
            clip_provider_boundary_halo=bool(args.clip_provider_boundary_halo),
            provider_output_layout=str(args.provider_output_layout),
        )
    except Exception as exc:  # noqa: BLE001
        input_halo_top, input_halo_bottom = _effective_provider_input_halo(
            row,
            clip_boundary_halo=bool(args.clip_provider_boundary_halo and row.path == "provider"),
        )
        provider_output_layout = ""
        if row.path == "provider":
            provider_output_layout = (
                "native_halo_stripe" if str(args.provider_output_layout) == "native_stripe" else "tight_compact"
            )
        payload = {
            "status": "error",
            "created_at_utc": _now_utc(),
            "backend": str(args.backend),
            "row_id": row.row_id,
            "case_id": str(row.case_id or ""),
            "hw": row.hw,
            "logical_hw": row.logical_hw,
            "channels": int(row.channels),
            "conv": f"{int(row.channels)},{int(row.channels)}",
            "kernel_label": row.kernel_name,
            "input_level": int(row.input_level),
            "expected_output_level": int(row.expected_output_level),
            "ckks_profile": CKKS_PROFILE_ID,
            "input_gap": int(row.stage_gap),
            "output_gap": int(row.stage_gap),
            "channel_group_size": int(row.channel_group_size),
            "logical_chw": row.logical_chw,
            "packed_chw": row.packed_chw,
            "variant": str(row.variant),
            "variant_label": VARIANT_LABELS[str(row.variant)],
            "path": row.path,
            "provider_requested_input_halo": row.halo,
            "provider_lt_grouping_mode": row.provider_lt_grouping_mode if row.path == "provider" else "",
            "provider_disable_shared_rotation": bool(row.provider_disable_shared_rotation and row.path == "provider"),
            "native_halo_channel_fold_mode": row.native_halo_channel_fold_mode if row.path == "provider" else "",
            "provider_boundary_halo_clipped": bool(args.clip_provider_boundary_halo and row.path == "provider"),
            "provider_output_layout": provider_output_layout,
            "provider_output_storage_layout": provider_output_layout,
            "input_halo_top": input_halo_top,
            "input_halo_bottom": input_halo_bottom,
            "ckks_params": {
                "LogN": 16,
                "LogQ": list(E2E_LOGQ),
                "LogP": list(E2E_LOGP),
                "LogScale": 40,
                "H": 192,
            },
            "kernel": "3x3/pad1/stride1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "maxrss_bytes": int(_maxrss_bytes()),
            "env": _env_snapshot(),
        }
    _write_json(Path(args.out), payload)
    return 0 if payload.get("status") == "ok" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Conv C,C kernel table for Orion dense and native halo provider.")
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--backend", choices=("lattigo", "python"), default="lattigo")
    parser.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument(
        "--kernel-cases",
        nargs="+",
        choices=tuple(KERNEL_CASES),
        default=list(DEFAULT_KERNEL_CASES),
        help=(
            "Explicit kernel cases to run. Overrides --channels and allows stage-packed U22 cases such as "
            "dec1b/dec2b/dec3b/dec4b/bottleneckb."
        ),
    )
    parser.add_argument("--hw", nargs="+", default=list(DEFAULT_HW))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANT_CHOICES), default=list(DEFAULT_VARIANTS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--input-level",
        type=int,
        default=DEFAULT_INPUT_LEVEL,
        help="Ciphertext input level for each depth-1 Conv kernel; default 2 yields output level 1.",
    )
    parser.add_argument("--max-worker-rss-gb", type=float, default=850.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--reuse-from",
        type=Path,
        nargs="*",
        default=[],
        help="Existing JSON files or result directories to import exact-shape non-stream rows from before running missing rows.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--clip-provider-boundary-halo",
        dest="clip_provider_boundary_halo",
        action="store_true",
        default=DEFAULT_CLIP_PROVIDER_BOUNDARY_HALO,
        help=(
            "For provider rows, keep the requested beta label but omit global top/bottom input halo at image "
            "boundaries. This is the default operator-level convention; internal stripe halo is still preserved."
        ),
    )
    parser.add_argument(
        "--keep-provider-boundary-halo",
        dest="clip_provider_boundary_halo",
        action="store_false",
        help="Legacy/debug mode: materialize global top/bottom provider input halo at image boundaries.",
    )
    parser.add_argument(
        "--provider-output-layout",
        choices=PROVIDER_OUTPUT_LAYOUTS,
        default="native_stripe",
        help="Provider output storage layout for kernel rows; native_stripe is the active no-sharing stripe mode.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare/reuse rows and update the doc without launching workers.")
    parser.add_argument("--update-doc-only", action="store_true")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=Path("/tmp/conv_kernel_row.json"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if any(
        str(variant).endswith("_individual_lt") or str(variant) == "provider_halo1_no_share"
        for variant in args.variants
    ) and str(args.provider_output_layout) != "native_stripe":
        parser.error("provider no-sharing variants require --provider-output-layout native_stripe")
    if int(args.input_level) < int(CONV_KERNEL_DEPTH):
        parser.error(f"--input-level must be at least {CONV_KERNEL_DEPTH} for a depth-{CONV_KERNEL_DEPTH} Conv kernel")
    if int(args.input_level) >= len(E2E_LOGQ):
        parser.error(f"--input-level must be < {len(E2E_LOGQ)} for {CKKS_PROFILE_ID}")

    if bool(args.update_doc_only):
        update_doc(Path(args.doc), Path(args.run_root), _rows_from_args(args), provider_output_layout=str(args.provider_output_layout))
        _write_summary_csv(Path(args.run_root), _rows_from_args(args), provider_output_layout=str(args.provider_output_layout))
        return 0
    if bool(args.run_one):
        os.environ.update(_apply_env_defaults(os.environ))
        return run_one(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
