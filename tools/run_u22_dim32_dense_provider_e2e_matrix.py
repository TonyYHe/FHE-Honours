#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.models.unet import UNet22PlusOutput


DOC_MARKER = "U22_BASE32_SILU7_STREAMING_PROVIDER_E2E_TABLE"
SUMMARY_DOC_MARKER = "U22_BASE32_SILU7_NETWORK_SUMMARY_TABLE"
DEFAULT_RUN_ROOT_BASE = REPO_ROOT / ".tmp" / "results"
LATEST_POINTER = REPO_ROOT / ".tmp" / "latest_u22_dim32_dense_provider_e2e_matrix.txt"

CASES: dict[str, dict[str, Any]] = {
    "192x192": {
        "dataset": "IBSR BRAIN 2D",
        "input_shape": (1, 1, 192, 192),
        "out_channels": 4,
    },
    "224x224": {
        "dataset": "HanCo Hand",
        "input_shape": (1, 3, 224, 224),
        "out_channels": 1,
    },
    "256x256": {
        "dataset": "COVID-19 lung",
        "input_shape": (1, 1, 256, 256),
        "out_channels": 1,
    },
    "384x384": {
        "dataset": "NuSegMSBench",
        "input_shape": (1, 1, 384, 384),
        "out_channels": 1,
    },
}

LINEAR_LAYERS = [
    "enc1a",
    "enc1b",
    "enc2a",
    "enc2b",
    "enc3a",
    "enc3b",
    "enc4a",
    "enc4b",
    "bottlenecka",
    "bottleneckb",
    "up4",
    "dec4a",
    "dec4b",
    "up3",
    "dec3a",
    "dec3b",
    "up2",
    "dec2a",
    "dec2b",
    "up1",
    "dec1a",
    "dec1b",
    "output",
]

BOOTSTRAP_OWNER_HINTS = {
    "enc1a_act": "enc1a",
    "enc1b_act": "enc1b",
    "pool1": "enc1b",
    "enc2a_act": "enc2a",
    "enc2b_act": "enc2b",
    "pool2": "enc2b",
    "enc3a_act": "enc3a",
    "enc3b_act": "enc3b",
    "pool3": "enc3b",
    "enc4a_act": "enc4a",
    "enc4b_act": "enc4b",
    "pool4": "enc4b",
    "bottlenecka_act": "bottlenecka",
    "bottleneckb_act": "bottleneckb",
    "up4": "up4",
    "cat4": "up4",
    "dec4a_act": "dec4a",
    "dec4b_act": "dec4b",
    "up3": "up3",
    "cat3": "up3",
    "dec3a_act": "dec3a",
    "dec3b_act": "dec3b",
    "up2": "up2",
    "cat2": "up2",
    "dec2a_act": "dec2a",
    "dec2b_act": "dec2b",
    "up1": "up1",
    "cat1": "up1",
    "dec1a_act": "dec1a",
    "dec1b_act": "dec1b",
}

PROVIDER_MODES = {
    "dp": "u22_256_base32_layout_dp",
    "dp_no_share_fold": "u22_256_base32_layout_dp_no_share_fold",
    "dp_noshare_fold": "u22_256_base32_layout_dp_no_share_fold",
    "noshare_fold": "u22_256_base32_layout_dp_no_share_fold",
    "greedy": "u22_256_base32_layout_greedy",
    "always": "u22_256_base32_layout_always",
    "always_fused": "u22_256_base32_layout_always_fused",
    "always_no_share": "u22_256_base32_layout_always_no_share",
    "always_noshare": "u22_256_base32_layout_always_no_share",
    "always_relayout_no_share": "u22_256_base32_layout_always_no_share",
    "always_relayout_noshare": "u22_256_base32_layout_always_no_share",
    "always_no_share_fused": "u22_256_base32_layout_always_no_share",
    "always_noshare_fused": "u22_256_base32_layout_always_no_share",
    "always_relayout_no_share_fused": "u22_256_base32_layout_always_no_share",
    "always_relayout_noshare_fused": "u22_256_base32_layout_always_no_share",
    "always_no_share_unfused": "u22_256_base32_layout_always_no_share_unfused",
    "always_noshare_unfused": "u22_256_base32_layout_always_no_share_unfused",
    "fixed_max": "u22_256_base32_layout_fixedmax_no_share",
    "fixed_max_no_share": "u22_256_base32_layout_fixedmax_no_share",
    "fixedmax_no_share": "u22_256_base32_layout_fixedmax_no_share",
    "fixed_noshare": "u22_256_base32_layout_fixedmax_no_share",
    "fixedmax_noshare": "u22_256_base32_layout_fixedmax_no_share",
    "fixed_max_no_share_fused": "u22_256_base32_layout_fixedmax_no_share",
    "fixedmax_no_share_fused": "u22_256_base32_layout_fixedmax_no_share",
    "fixed_noshare_fused": "u22_256_base32_layout_fixedmax_no_share",
    "fixedmax_noshare_fused": "u22_256_base32_layout_fixedmax_no_share",
    "fixed_max_no_share_unfused": "u22_256_base32_layout_fixedmax_no_share_unfused",
    "fixedmax_no_share_unfused": "u22_256_base32_layout_fixedmax_no_share_unfused",
}

CPU_COUNT = max(1, int(os.cpu_count() or 1))

ENV_DEFAULTS: dict[str, str] = {
    "PYTHONUNBUFFERED": "1",
    "MALLOC_ARENA_MAX": "2",
    "GOMAXPROCS": "1",
    "ORION_COMPILE_PARALLEL_POLICY": "manual",
    "ORION_SINGLE_SLOT_LAYER_CACHE": "1",
    "ORION_SINGLE_SLOT_ENCODE_WORKERS": str(CPU_COUNT),
    "ORION_LATTIGO_STREAMING_LT": "0",
    "ORION_UNIFIED_STREAM_COMPILE_IO_NONE": "0",
    "ORION_LATTIGO_MEMORY_BOUNDED_COMPILE": "0",
    "ORION_LATTIGO_MEMORY_BOUNDED_EVAL": "0",
    "ORION_LATTIGO_BOOTSTRAP_MANY": "0",
    "ORION_LATTIGO_CLEAR_BACKEND": "0",
    "ORION_CPP_DIAG_BUILDER": "1",
    "ORION_CPP_DIAG_BUILDER_DENSE": "1",
    "ORION_CPP_DIAG_BUILDER_PROVIDER": "1",
    "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE": "1",
    "ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT": "1",
    "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE_SINGLE_SLOT_METADATA": "1",
    "ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT_SINGLE_SLOT_METADATA": "1",
    "ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_SOURCE": "1",
    "ORION_PACK_CONV_WORKERS": str(CPU_COUNT),
    "ORION_DIRECT_PACK_WORKERS": str(CPU_COUNT),
    "ORION_LT_COMPILE_WORKERS": str(CPU_COUNT),
    "ORION_UNIFIED_COMPILE_WORKERS": str(CPU_COUNT),
    "ORION_LATTIGO_COMPILE_WORKERS": str(CPU_COUNT),
    "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS": str(CPU_COUNT),
    "ORION_UNIFIED_LT_OUTPUT_FUSION": "1",
    "ORION_LAYOUT_POLICY_RELAYOUT_KERNEL": "1",
    "ORION_LAYOUT_POLICY_PROVIDER_NATIVE_HALO": "1",
    "ORION_UNIFIED_LT_INDIVIDUAL_EVAL": "1",
    "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS": "0",
    "ORION_LATTIGO_UNIFIED_NO_BSGS": "0",
    "ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE": "1",
    "ORION_REGION_FIRST_CLEANUP_AFTER_OUTPUTS": "1",
    "ORION_CONCAT_FUSION": "auto",
    "ORION_DISABLE_BOOTSTRAP_PRESCALE_FUSION": "0",
}

REQUIRED_MAINLINE_ENV: dict[str, str] = {
    "GOMAXPROCS": "1",
    "ORION_SINGLE_SLOT_LAYER_CACHE": "1",
    "ORION_LATTIGO_STREAMING_LT": "0",
    "ORION_LATTIGO_BOOTSTRAP_MANY": "0",
    "ORION_LATTIGO_CLEAR_BACKEND": "0",
    "ORION_CPP_DIAG_BUILDER": "1",
    "ORION_CPP_DIAG_BUILDER_DENSE": "1",
    "ORION_CPP_DIAG_BUILDER_PROVIDER": "1",
    "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE": "1",
    "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE_SINGLE_SLOT_METADATA": "1",
    "ORION_UNIFIED_LT_INDIVIDUAL_EVAL": "1",
    "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS": "0",
    "ORION_LATTIGO_UNIFIED_NO_BSGS": "0",
}

DENSE_MODE_ENV: dict[str, str] = {
    "ORION_DENSE_LAYER_CACHE_GRANULARITY": "group",
    "ORION_DENSE_LAYER_CACHE_GROUP_TRANSFORMS": "auto",
    "ORION_CONCAT_FUSION": "0",
    "ORION_UNIFIED_LT_OUTPUT_FUSION": "0",
    "ORION_DISABLE_BOOTSTRAP_PRESCALE_FUSION": "1",
}

PROVIDER_MODE_ENV: dict[str, str] = {
    "ORION_DENSE_LAYER_CACHE_GRANULARITY": "layer",
    "ORION_DENSE_LAYER_CACHE_GROUP_TRANSFORMS": "auto",
    "ORION_CONCAT_FUSION": "auto",
    "ORION_UNIFIED_LT_OUTPUT_FUSION": "1",
    "ORION_DISABLE_BOOTSTRAP_PRESCALE_FUSION": "0",
}

ENV_TUNING_KEYS = (
    "GOMAXPROCS",
    "ORION_COMPILE_MEMORY_RESERVE_GB",
    "ORION_SINGLE_SLOT_LAYER_CACHE",
    "ORION_DENSE_LAYER_CACHE_GRANULARITY",
    "ORION_DENSE_LAYER_CACHE_GROUP_TRANSFORMS",
    "ORION_DENSE_LAYER_CACHE_GROUP_BUDGET_GB",
    "ORION_SINGLE_SLOT_ENCODE_WORKERS",
    "ORION_LT_COMPILE_WORKERS",
    "ORION_UNIFIED_COMPILE_WORKERS",
    "ORION_LATTIGO_COMPILE_WORKERS",
    "ORION_UNIFIED_STREAM_COMPILE_BATCH_TRANSFORMS",
    "ORION_UNIFIED_COMPILE_BATCH_TRANSFORMS",
    "ORION_UNIFIED_LOAD_BATCH_TRANSFORMS",
    "ORION_UNIFIED_CACHED_LOAD_BATCH_TRANSFORMS",
    "ORION_PACK_CONV_WORKERS",
    "ORION_DIRECT_PACK_WORKERS",
    "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS",
    "ORION_LATTIGO_BOOTSTRAP_WORKERS",
    "ORION_CONCAT_FUSION",
    "ORION_UNIFIED_LT_OUTPUT_FUSION",
    "ORION_DISABLE_BOOTSTRAP_PRESCALE_FUSION",
    "ORION_LATTIGO_CLEAR_BACKEND",
    "ORION_CPP_DIAG_BUILDER",
    "ORION_CPP_DIAG_BUILDER_DENSE",
    "ORION_CPP_DIAG_BUILDER_PROVIDER",
    "ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE",
    "ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_OUTPUT",
    "ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_SOURCE",
    "ORION_CPP_DIAG_BUILDER_SHADOW",
    "ORION_CPP_DIAG_BUILDER_STRICT",
)

ENV_SNAPSHOT_KEYS = tuple(sorted(set(ENV_DEFAULTS) | set(ENV_TUNING_KEYS)))


def _safe_case(case: str) -> str:
    return str(case).replace("x", "_")


def _network_name(case: str) -> str:
    return f"u23_dim32_{_safe_case(case)}_full"


def _backend_clear_value(backend: str) -> str:
    value = str(backend).strip().lower()
    if value == "clear":
        return "1"
    if value == "ckks":
        return "0"
    raise ValueError(f"unknown backend {backend!r}; expected 'ckks' or 'clear'")


def _apply_env_defaults(env: dict[str, str], *, backend: str = "ckks") -> dict[str, str]:
    updated = dict(env)
    for key, value in ENV_DEFAULTS.items():
        updated.setdefault(key, value)
    for key, value in REQUIRED_MAINLINE_ENV.items():
        updated[key] = value
    updated["ORION_LATTIGO_CLEAR_BACKEND"] = _backend_clear_value(str(backend))
    return updated


def _apply_mode_env(env: dict[str, str], mode: str) -> dict[str, str]:
    updated = dict(env)
    if str(mode) == "provider":
        updated.update(PROVIDER_MODE_ENV)
    else:
        updated.update(DENSE_MODE_ENV)
    return updated


def _mode_layer_mae_enabled(args: argparse.Namespace, mode: str) -> bool:
    if bool(getattr(args, "layer_mae", False)):
        return True
    if str(mode) == "provider":
        return bool(getattr(args, "provider_layer_mae", False))
    return bool(getattr(args, "dense_layer_mae", False))


def _provider_mode(policy: str) -> str:
    key = str(policy).strip().lower().replace("-", "_")
    if key == "fixedmax":
        key = "fixed_max"
    if key not in PROVIDER_MODES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {sorted(PROVIDER_MODES)}")
    return PROVIDER_MODES[key]


def _register_networks() -> Any:
    from tools import run_lattigo_e2e_compare as base

    for case, spec in CASES.items():
        input_shape = tuple(int(v) for v in spec["input_shape"])
        out_channels = int(spec["out_channels"])

        def _builder(
            *,
            activation: str | None = None,
            silu_degree: int = 31,
            _input_shape: tuple[int, int, int, int] = input_shape,
            _out_channels: int = out_channels,
        ):
            return UNet22PlusOutput(
                in_channels=int(_input_shape[1]),
                out_channels=int(_out_channels),
                base_channels=32,
                activation=str(activation or "silu"),
                silu_degree=int(silu_degree),
            )

        base.NETWORKS[_network_name(case)] = {
            "label": f"U22+output dim32 {case}",
            "model": "UNet22PlusOutput",
            "dataset": str(spec["dataset"]),
            "input_shape": input_shape,
            "provider_mode": "u22_256_base32_layout_dp",
            "config": base._r18_config,
            "builder": _builder,
            "scope": "full",
            "base_dim": 32,
            "out_channels": out_channels,
        }
    return base


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {"status": "bad_json", "error": f"{type(exc).__name__}: {exc}"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _status_ok(path: Path) -> bool:
    payload = _read_json(path)
    return isinstance(payload, dict) and payload.get("status") == "ok"


def _resource_maxrss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _case_paths(run_root: Path, case: str, mode: str) -> tuple[Path, Path]:
    case_dir = Path(run_root) / _safe_case(case)
    return case_dir / f"{mode}_e2e.json", case_dir / f"{mode}_e2e.log"


def _annotate_result(
    out_path: Path,
    *,
    case: str,
    mode: str,
    backend: str,
    policy: str,
    provider_mode: str,
    run_root: Path,
    started_at: float,
    env_snapshot: dict[str, str],
) -> None:
    payload = _read_json(out_path) or {}
    spec = CASES[str(case)]
    payload["runner"] = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "run_root": str(run_root),
        "case": str(case),
        "mode": str(mode),
        "backend": str(backend),
        "policy": str(policy),
        "provider_mode": str(provider_mode),
        "local_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": float(time.perf_counter() - started_at),
        "maxrss_bytes": int(_resource_maxrss_bytes()),
        "env": {key: str(env_snapshot.get(key, "")) for key in ENV_SNAPSHOT_KEYS},
    }
    payload["model_variant"] = {
        "name": "UNet22PlusOutput",
        "linear_layers": 23,
        "base_dim": 32,
        "input_shape": [int(v) for v in tuple(spec["input_shape"])],
        "out_channels": int(spec["out_channels"]),
        "dataset": str(spec["dataset"]),
        "activation": "SiLU7",
        "output_head": "1x1/pad0",
    }
    _write_json(out_path, payload)


def run_one(args: argparse.Namespace) -> int:
    mode = str(args.mode)
    os.environ.update(_apply_mode_env(_apply_env_defaults(os.environ, backend=str(args.backend)), mode))
    env_snapshot = dict(os.environ)
    base = _register_networks()
    out_path = Path(args.out)
    started_at = time.perf_counter()
    provider_mode = _provider_mode(str(args.policy)) if mode == "provider" else ""
    try:
        base._run_one(
            network=_network_name(str(args.case)),
            backend="lattigo",
            mode=mode,
            out_path=out_path,
            seed=int(args.seed),
            compile_only=False,
            forward_runs=int(args.forward_runs),
            warmup_runs=int(args.warmup_runs),
            profile_modules=False,
            profile_lt=bool(args.profile_lt),
            trace_forward_memory=bool(args.trace_forward_memory),
            operator_breakdown=bool(args.operator_breakdown),
            layer_mae=_mode_layer_mae_enabled(args, mode),
            provider_mode_override=provider_mode if mode == "provider" else None,
            io_mode="none",
            io_dir=None,
            diags_path=None,
            keys_path=None,
            logn_override=None,
            activation="silu",
            silu_degree=7,
            ckks_preset="resnet",
        )
        return 0
    finally:
        _annotate_result(
            out_path,
            case=str(args.case),
            mode=mode,
            backend=str(args.backend),
            policy=str(args.policy),
            provider_mode=provider_mode,
            run_root=Path(args.run_root),
            started_at=started_at,
            env_snapshot=env_snapshot,
        )


def _metric(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(number)


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


def _fmt_bool(value: Any) -> str:
    if value is None or value == "":
        return ""
    return "yes" if bool(value) else "no"


def _gib(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value) / (1024 ** 3):.1f}"
    except (TypeError, ValueError):
        return ""


def _markdown_escape(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _forward_timing(payload: dict[str, Any], key: str) -> float | None:
    timing = (
        payload.get("measured_forward_mean_timing_s")
        or payload.get("forward_mean_timing_s")
        or payload.get("timing_s")
        or {}
    )
    if not isinstance(timing, dict) or timing.get(key) is None:
        return None
    return _as_float(timing.get(key))


def _rotation_count(payload: dict[str, Any]) -> int | None:
    report = payload.get("rotation_report_after_forward") or payload.get("rotation_report_after_compile") or {}
    value = report.get("total_rotation_eval_count_estimate") if isinstance(report, dict) else None
    return None if value is None else int(value)


def _bootstrap_count(payload: dict[str, Any]) -> int | None:
    report = payload.get("bootstrap_report_after_forward") or payload.get("bootstrap_report_after_compile") or {}
    value = report.get("count") if isinstance(report, dict) else None
    return None if value is None else int(value)


def _first_error(result_path: Path, payload: dict[str, Any] | None) -> str:
    if isinstance(payload, dict):
        error = payload.get("error") or payload.get("error_type")
        if error:
            return str(error).replace("\n", " ")[:160]
    log_path = result_path.with_suffix(".log")
    if log_path.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        return " ".join(tail)[-180:]
    return ""


def _layer_for_row(row: dict[str, Any]) -> str | None:
    candidates = [
        str(row.get("module_path", "")),
        str(row.get("node", "")),
        str(row.get("storage_key", "")),
        str(row.get("executor", "")),
    ]
    for layer in LINEAR_LAYERS:
        prefixes = (layer, f"{layer}.", f"{layer}_")
        if any(candidate == layer or candidate.startswith(prefixes) for candidate in candidates):
            return layer
    for candidate in candidates:
        for token in candidate.replace("/", ".").replace(":", ".").split("."):
            if token in LINEAR_LAYERS:
                return token
    return None


def _empty_layer_stats() -> dict[str, Any]:
    return {
        "row_count": 0,
        "transform_count": 0,
        "legacy_load_encode_s": 0.0,
        "layer_cache_turnover_s": 0.0,
        "layer_cache_encode_s": 0.0,
        "layer_cache_key_prepare_s": 0.0,
        "layer_cache_evict_s": 0.0,
        "lt_accumulate_s": 0.0,
        "eval_total_s": 0.0,
        "stream_build_map_s": 0.0,
        "stream_encode_hoist_s": 0.0,
        "stream_load_payload_s": 0.0,
        "stream_eval_s": 0.0,
        "stream_accumulate_s": 0.0,
        "cpp_baby_giant_s": 0.0,
    }


def _empty_boot_stats() -> dict[str, Any]:
    return {
        "module_count": 0,
        "runtime_call_count": 0,
        "bootstrap_s": 0.0,
        "backend_bootstrap_s": 0.0,
        "nodes": [],
    }


def _bootstrap_anchor_name(name: str) -> str:
    value = str(name)
    if value.endswith(".bootstrapper"):
        value = value[: -len(".bootstrapper")]
    return value


def _bootstrap_owner(name: str) -> str | None:
    anchor = _bootstrap_anchor_name(str(name))
    token = anchor.split(".")[-1]
    if token in LINEAR_LAYERS:
        return token
    if token in BOOTSTRAP_OWNER_HINTS:
        return BOOTSTRAP_OWNER_HINTS[token]
    if token.endswith("_act") and token[:-4] in LINEAR_LAYERS:
        return token[:-4]
    return None


def _bootstrap_stats(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stats = {layer: _empty_boot_stats() for layer in LINEAR_LAYERS}
    extra: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    report = payload.get("bootstrap_report_after_forward") or payload.get("bootstrap_report_after_compile") or {}
    for row in report.get("rows", []) or [] if isinstance(report, dict) else []:
        row = dict(row)
        name = str(row.get("name", ""))
        if not name:
            continue
        records[name] = {
            "name": name,
            "module_count": 1,
            "runtime_call_count": int(row.get("runtime_call_count", 0) or 0),
            "bootstrap_s": 0.0,
            "backend_bootstrap_s": 0.0,
        }
    breakdown_rows = _metric(payload, ("operator_breakdown_after_forward", "bootstrap", "rows")) or []
    if isinstance(breakdown_rows, list):
        for row in breakdown_rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", ""))
            if not name:
                continue
            record = records.setdefault(
                name,
                {
                    "name": name,
                    "module_count": 1,
                    "runtime_call_count": 0,
                    "bootstrap_s": 0.0,
                    "backend_bootstrap_s": 0.0,
                },
            )
            record["runtime_call_count"] = int(row.get("runtime_call_count", record.get("runtime_call_count", 0)) or 0)
            record["bootstrap_s"] = _as_float(row.get("bootstrap_s"))
            record["backend_bootstrap_s"] = _as_float(row.get("backend_bootstrap_s"))
    for record in records.values():
        owner = _bootstrap_owner(str(record.get("name", "")))
        target = stats.get(owner) if owner is not None else None
        if target is None:
            extra.append(dict(record))
            continue
        target["module_count"] += int(record.get("module_count", 0) or 0)
        target["runtime_call_count"] += int(record.get("runtime_call_count", 0) or 0)
        target["bootstrap_s"] += _as_float(record.get("bootstrap_s"))
        target["backend_bootstrap_s"] += _as_float(record.get("backend_bootstrap_s"))
        target["nodes"].append(_bootstrap_anchor_name(str(record.get("name", ""))))
    return stats, extra


def _layer_stats(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats = {layer: _empty_layer_stats() for layer in LINEAR_LAYERS}
    rows = _metric(payload, ("operator_breakdown_after_forward", "mvm", "group_rows")) or []
    if not isinstance(rows, list):
        return stats

    def add_row(row: dict[str, Any]) -> None:
        layer = _layer_for_row(row)
        if layer is None:
            return
        timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
        stream_build = _as_float(timing.get("stream_build_map_s"))
        stream_encode = _as_float(timing.get("stream_encode_hoist_s"))
        stream_payload = _as_float(timing.get("stream_load_payload_s"))
        stream_eval = _as_float(timing.get("stream_eval_s"))
        stream_accum = _as_float(timing.get("stream_accumulate_s"))
        baby_giant = _as_float(timing.get("cpp_baby_step_s")) + _as_float(timing.get("cpp_giant_step_s"))
        layer_cache_encode = (
            _as_float(row.get("lt_layer_cache_encode_s"))
            or _as_float(timing.get("layer_cache_encode_s"))
            or _as_float(timing.get("provider_layer_cache_encode_s"))
        )
        layer_cache_key_prepare = (
            _as_float(row.get("lt_layer_cache_key_prepare_s"))
            or _as_float(timing.get("layer_cache_key_prepare_s"))
            or _as_float(timing.get("provider_layer_cache_key_prepare_s"))
        )
        layer_cache_evict = _as_float(row.get("lt_layer_cache_evict_s")) or _as_float(timing.get("layer_cache_evict_s"))
        layer_cache_turnover = _as_float(row.get("lt_layer_cache_turnover_s")) or (
            layer_cache_encode + layer_cache_key_prepare + layer_cache_evict
        )
        lt_accum = stream_eval + stream_accum + baby_giant
        if lt_accum <= 0.0:
            lt_accum = _as_float(row.get("mvm_kernel_s"))
        entry = stats[layer]
        entry["row_count"] += 1
        entry["transform_count"] += int(row.get("transform_count") or 0)
        entry["stream_build_map_s"] += stream_build
        entry["stream_encode_hoist_s"] += stream_encode
        entry["stream_load_payload_s"] += stream_payload
        entry["stream_eval_s"] += stream_eval
        entry["stream_accumulate_s"] += stream_accum
        entry["cpp_baby_giant_s"] += baby_giant
        entry["legacy_load_encode_s"] += stream_build + stream_encode + stream_payload
        entry["layer_cache_turnover_s"] += layer_cache_turnover
        entry["layer_cache_encode_s"] += layer_cache_encode
        entry["layer_cache_key_prepare_s"] += layer_cache_key_prepare
        entry["layer_cache_evict_s"] += layer_cache_evict
        entry["lt_accumulate_s"] += lt_accum
        entry["eval_total_s"] += _as_float(row.get("mvm_eval_total_s"))

    for row in rows:
        if not isinstance(row, dict):
            continue
        add_row(row)
    return stats


def _breakdown_totals(payload: dict[str, Any]) -> dict[str, Any]:
    return _metric(payload, ("operator_breakdown_after_forward", "totals")) or {}


def _summary_total(totals: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in totals and totals.get(key) not in (None, ""):
            return _as_float(totals.get(key))
    return None


def _summary_unattributed(
    totals: dict[str, Any],
    he_forward_s: float | None,
    parts: list[float | None],
) -> float | None:
    direct = _summary_total(totals, "wall_unattributed_he_forward_s", "unattributed_he_forward_s")
    if direct is not None:
        return direct
    if he_forward_s is None or any(value is None for value in parts):
        return None
    return float(he_forward_s) - sum(float(value or 0.0) for value in parts)


def _summary_sum(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return float(sum(float(value or 0.0) for value in values))


def _cell_float(cell: str) -> float | None:
    value = str(cell or "").strip().replace(",", "")
    if not value:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def _summary_layer_cache_turnover(totals: dict[str, Any]) -> float | None:
    direct = _summary_total(totals, "wall_layer_cache_turnover_s", "lt_layer_cache_turnover_s")
    if direct is not None:
        return direct
    parts = [
        _summary_total(totals, "lt_layer_cache_encode_s"),
        _summary_total(totals, "lt_layer_cache_key_prepare_s"),
        _summary_total(totals, "lt_layer_cache_evict_s"),
    ]
    return _summary_sum(*parts)


def _summary_runtime_load_trim(totals: dict[str, Any]) -> float | None:
    direct = _summary_total(totals, "wall_runtime_load_trim_s")
    if direct is not None:
        return direct
    load_encode = _summary_total(totals, "lt_runtime_load_encode_s")
    trim_unload = _summary_total(totals, "lt_runtime_trim_unload_s")
    return _summary_sum(load_encode, trim_unload)


def _summary_executor_overhead(totals: dict[str, Any]) -> float | None:
    direct = _summary_total(totals, "wall_executor_overhead_s")
    if direct is not None:
        return direct
    return _summary_sum(
        _summary_total(totals, "executor_wrap_s") or 0.0,
        _summary_total(totals, "executor_postprocess_s"),
        _summary_total(totals, "executor_rescale_s"),
        _summary_total(totals, "executor_accumulate_s"),
    )


def _summary_linear_wrapper(totals: dict[str, Any]) -> float | None:
    direct = _summary_total(totals, "wall_linear_wrapper_postprocess_s")
    if direct is not None:
        return direct
    return _summary_total(totals, "linear_wrapper_postprocess_s")


def _summary_wall_residual(
    totals: dict[str, Any],
    he_forward_s: float | None,
    parts: list[float | None],
) -> float | None:
    direct = _summary_total(totals, "wall_unattributed_he_forward_s")
    if direct is not None:
        return direct
    if he_forward_s is None or any(value is None for value in parts):
        return None
    return float(he_forward_s) - sum(float(value or 0.0) for value in parts)


def _runtime_mode(payload: dict[str, Any]) -> str:
    return str(
        payload.get("runtime_fairness_mode")
        or _metric(payload, ("measured_runtime_fairness_timing", "runtime_fairness_mode"))
        or _metric(payload, ("runtime_fairness_timing_after_forward", "runtime_fairness_mode"))
        or ""
    )


def _case_table_rows(run_root: Path, case: str, mode: str) -> list[list[str]]:
    result_path, _log_path = _case_paths(run_root, case, mode)
    payload = _read_json(result_path)
    has_payload = isinstance(payload, dict)
    status = "pending" if payload is None else str(payload.get("status", "unknown"))
    spec = CASES[str(case)]
    timing = payload.get("timing_s", {}) if isinstance(payload, dict) else {}
    totals = _breakdown_totals(payload or {})
    encrypt_s = _forward_timing(payload or {}, "encrypt")
    he_forward_s = _forward_timing(payload or {}, "he_forward")
    decode_s = _forward_timing(payload or {}, "decrypt_decode")
    hot_s = None
    if encrypt_s is not None and he_forward_s is not None and decode_s is not None:
        hot_s = float(encrypt_s + he_forward_s + decode_s)
    total_legacy_load_encode = None
    total_layer_cache_turnover = None
    total_layer_cache_encode = None
    total_layer_cache_key_prepare = None
    total_layer_cache_evict = None
    total_lt_accumulate = None
    if has_payload:
        total_legacy_load_encode = _as_float(totals.get("lt_runtime_load_encode_s")) or (
            _as_float(totals.get("lt_runtime_stream_build_map_s"))
            + _as_float(totals.get("lt_runtime_stream_encode_hoist_s"))
            + _as_float(totals.get("lt_runtime_stream_load_payload_s"))
        )
        total_layer_cache_encode = _as_float(totals.get("lt_layer_cache_encode_s"))
        total_layer_cache_key_prepare = _as_float(totals.get("lt_layer_cache_key_prepare_s"))
        total_layer_cache_evict = _as_float(totals.get("lt_layer_cache_evict_s"))
        total_layer_cache_turnover = _as_float(totals.get("lt_layer_cache_turnover_s")) or (
            total_layer_cache_encode + total_layer_cache_key_prepare + total_layer_cache_evict
        )
        total_lt_accumulate = _as_float(totals.get("mvm_kernel_s"))
    total_row = [
        str(case),
        str(spec["dataset"]),
        f"{int(spec['input_shape'][1])}->{int(spec['out_channels'])}",
        str(mode),
        status,
        "TOTAL",
        _fmt_int(_metric(payload or {}, ("operator_breakdown_after_forward", "mvm", "totals", "group_count"))),
        _fmt_float(total_layer_cache_turnover),
        _fmt_float(total_layer_cache_encode),
        _fmt_float(total_layer_cache_key_prepare),
        _fmt_float(total_layer_cache_evict),
        _fmt_float(total_lt_accumulate),
        _fmt_float(totals.get("mvm_eval_total_s")),
        _fmt_float(total_legacy_load_encode),
        _fmt_float(totals.get("lt_runtime_stream_build_map_s")),
        _fmt_float(totals.get("lt_runtime_stream_encode_hoist_s")),
        _fmt_float(totals.get("lt_runtime_stream_load_payload_s")),
        _fmt_float(totals.get("lt_runtime_stream_eval_s")),
        _fmt_float(totals.get("lt_runtime_stream_accumulate_s")),
        "",
        _fmt_int(_bootstrap_count(payload or {})),
        _fmt_float(totals.get("bootstrap_s")),
        "",
        _fmt_float(timing.get("compile") if isinstance(timing, dict) else None),
        _fmt_float(he_forward_s),
        _fmt_float(hot_s),
        _fmt_int(_rotation_count(payload or {})),
        _fmt_int(_bootstrap_count(payload or {})),
        _runtime_mode(payload or {}),
        _gib(_metric(payload or {}, ("runner", "maxrss_bytes"))),
        str(result_path),
        _first_error(result_path, payload),
    ]
    rows = [total_row]
    stats = _layer_stats(payload or {})
    boot_stats, extra_boot = _bootstrap_stats(payload or {}) if has_payload else (
        {layer: _empty_boot_stats() for layer in LINEAR_LAYERS},
        [],
    )
    for layer in LINEAR_LAYERS:
        item = stats[layer]
        boot_item = boot_stats[layer]
        has_row = bool(item["row_count"])
        rows.append(
            [
                str(case),
                str(spec["dataset"]),
                f"{int(spec['input_shape'][1])}->{int(spec['out_channels'])}",
                str(mode),
                status if has_row else ("pending" if payload is None else "no-row"),
                layer,
                _fmt_int(item["transform_count"] or item["row_count"]) if has_row else "",
                _fmt_float(item["layer_cache_turnover_s"]) if has_row else "",
                _fmt_float(item["layer_cache_encode_s"]) if has_row else "",
                _fmt_float(item["layer_cache_key_prepare_s"]) if has_row else "",
                _fmt_float(item["layer_cache_evict_s"]) if has_row else "",
                _fmt_float(item["lt_accumulate_s"]) if has_row else "",
                _fmt_float(item["eval_total_s"]) if has_row else "",
                _fmt_float(item["legacy_load_encode_s"]) if has_row else "",
                _fmt_float(item["stream_build_map_s"]) if has_row else "",
                _fmt_float(item["stream_encode_hoist_s"]) if has_row else "",
                _fmt_float(item["stream_load_payload_s"]) if has_row else "",
                _fmt_float(item["stream_eval_s"]) if has_row else "",
                _fmt_float(item["stream_accumulate_s"]) if has_row else "",
                _fmt_float(item["cpp_baby_giant_s"]) if has_row else "",
                _fmt_int(boot_item["module_count"]) if boot_item["module_count"] else "",
                _fmt_float(boot_item["bootstrap_s"]) if boot_item["module_count"] else "",
                ",".join(str(node) for node in boot_item["nodes"][:6]),
                "",
                "",
                "",
                "",
                "",
                _runtime_mode(payload or {}),
                "",
                str(result_path),
                "",
            ]
        )
    for record in extra_boot:
        rows.append(
            [
                str(case),
                str(spec["dataset"]),
                f"{int(spec['input_shape'][1])}->{int(spec['out_channels'])}",
                str(mode),
                status,
                f"boot-only:{_bootstrap_anchor_name(str(record.get('name', '')))}",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                _fmt_int(record.get("module_count")),
                _fmt_float(record.get("bootstrap_s")),
                _bootstrap_anchor_name(str(record.get("name", ""))),
                "",
                "",
                "",
                "",
                "",
                _runtime_mode(payload or {}),
                "",
                str(result_path),
                "bootstrap attached to a non-linear/non-U22 layer node",
            ]
        )
    return rows


def _markdown_table(rows: list[list[str]]) -> str:
    headers = [
        "input",
        "dataset",
        "I/O ch",
        "path",
        "status",
        "layer",
        "groups/transforms",
        "layer cache turnover s",
        "layer cache diag+encode s",
        "layer cache key prep s",
        "layer cache evict s",
        "LT+accum s",
        "eval total s",
        "legacy load/encode s",
        "stream build s",
        "encode hoist s",
        "stream load s",
        "LT eval s",
        "LT accum s",
        "baby+giant s",
        "boot after count",
        "boot after s",
        "boot after nodes",
        "compile s",
        "HE forward s",
        "hot E2E s",
        "rotations",
        "boots",
        "runtime mode",
        "peak RSS GiB",
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
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
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
        "---:",
        "---",
        "---",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def _case_summary(run_root: Path, case: str, mode: str) -> dict[str, Any]:
    result_path, _log_path = _case_paths(run_root, case, mode)
    raw_payload = _read_json(result_path)
    has_payload = isinstance(raw_payload, dict)
    payload = raw_payload if has_payload else {}
    timing = payload.get("timing_s", {}) if isinstance(payload.get("timing_s"), dict) else {}
    encrypt_s = _forward_timing(payload, "encrypt")
    he_forward_s = _forward_timing(payload, "he_forward")
    decode_s = _forward_timing(payload, "decrypt_decode")
    hot_s = None
    if encrypt_s is not None and he_forward_s is not None and decode_s is not None:
        hot_s = float(encrypt_s + he_forward_s + decode_s)
    totals = _breakdown_totals(payload)
    mae = payload.get("mae_vs_clear") if isinstance(payload.get("mae_vs_clear"), dict) else {}
    layer_mae = _metric(payload, ("layer_mae_after_forward", "summary"))
    layer_mae = layer_mae if isinstance(layer_mae, dict) else {}
    mvm_lt_s = _summary_total(totals, "mvm_kernel_s")
    activation_s = _summary_total(totals, "activation_excluding_bootstrap_s", "activation_s")
    bootstrap_s = _summary_total(totals, "bootstrap_s")
    compute_s = _summary_total(totals, "compute_mvm_activation_bootstrap_s", "compute_accounted_s")
    if compute_s is None:
        compute_s = _summary_sum(mvm_lt_s, activation_s, bootstrap_s)
    diag_encode_s = _summary_total(totals, "lt_layer_cache_encode_s")
    layer_cache_turnover_s = _summary_layer_cache_turnover(totals)
    executor_overhead_s = _summary_executor_overhead(totals)
    linear_wrapper_s = _summary_linear_wrapper(totals)
    runtime_load_trim_s = _summary_runtime_load_trim(totals)
    wall_residual_s = _summary_wall_residual(
        totals,
        he_forward_s,
        [
            compute_s,
            layer_cache_turnover_s,
            linear_wrapper_s,
            executor_overhead_s,
            runtime_load_trim_s,
        ],
    )
    return {
        "status": str(payload.get("status", "pending" if not has_payload else "unknown")),
        "compile_s": timing.get("compile") if isinstance(timing, dict) else None,
        "he_forward_s": he_forward_s,
        "hot_e2e_s": hot_s,
        "mvm_lt_s": mvm_lt_s,
        "activation_excl_boot_s": activation_s,
        "bootstrap_s": bootstrap_s,
        "compute_mvm_activation_bootstrap_s": compute_s,
        "diag_encode_s": diag_encode_s,
        "layer_cache_turnover_s": layer_cache_turnover_s,
        "linear_wrapper_s": linear_wrapper_s,
        "executor_overhead_s": executor_overhead_s,
        "runtime_load_trim_s": runtime_load_trim_s,
        "wall_residual_s": wall_residual_s,
        "unattributed_s": _summary_unattributed(
            totals,
            he_forward_s,
            [mvm_lt_s, activation_s, bootstrap_s, diag_encode_s],
        ),
        "rotations": _rotation_count(payload) if has_payload else None,
        "boots": _bootstrap_count(payload) if has_payload else None,
        "e2e_mae": mae.get("mae"),
        "e2e_max_abs": mae.get("max_abs"),
        "layer_mae_overall_ok": (
            layer_mae.get("overall_ok")
            if "overall_ok" in layer_mae
            else payload.get("layer_mae_overall_ok")
        ),
        "layer_mae_max_mae": layer_mae.get("max_mae"),
        "layer_mae_max_abs": layer_mae.get("max_abs"),
        "peak_rss_gib": _metric(payload, ("runner", "maxrss_bytes")),
        "runtime_mode": _runtime_mode(payload),
        "result_path": str(result_path),
        "note": _first_error(result_path, payload),
    }


def _ratio(numerator: Any, denominator: Any) -> float | None:
    left = _as_float(numerator)
    right = _as_float(denominator)
    if right <= 0.0:
        return None
    return float(left / right)


def _network_summary_table(run_root: Path, cases: list[str]) -> str:
    headers = [
        "input",
        "dataset",
        "I/O ch",
        "dense status",
        "Halo status",
        "dense HE forward s",
        "Halo HE forward s",
        "dense/Halo HE",
        "dense hot E2E s",
        "Halo hot E2E s",
        "dense MVM/LT s (incl pool)",
        "Halo MVM/LT s (incl pool)",
        "dense activation excl boot s",
        "Halo activation excl boot s",
        "dense bootstrap s",
        "Halo bootstrap s",
        "dense MVM+act+boot s",
        "Halo MVM+act+boot s",
        "dense diag+encode s",
        "Halo diag+encode s",
        "dense layer-cache turnover s",
        "Halo layer-cache turnover s",
        "dense linear wrapper s",
        "Halo linear wrapper s",
        "dense executor overhead s",
        "Halo executor overhead s",
        "dense load/trim s",
        "Halo load/trim s",
        "dense wall residual s",
        "Halo wall residual s",
        "dense rotations",
        "Halo rotations",
        "dense boots",
        "Halo boots",
        "dense e2e MAE",
        "Halo e2e MAE",
        "dense e2e max abs",
        "Halo e2e max abs",
        "Halo layer-MAE ok",
        "Halo layer max MAE",
        "Halo layer max abs",
        "dense RSS GiB",
        "Halo RSS GiB",
        "runtime mode",
        "result files",
        "note",
    ]
    aligns = ["---", "---", "---", "---", "---"] + ["---:"] * (len(headers) - 8) + ["---", "---", "---"]
    rows: list[list[str]] = []
    for case in cases:
        spec = CASES[str(case)]
        dense = _case_summary(run_root, str(case), "dense")
        provider = _case_summary(run_root, str(case), "provider")
        notes = [str(value) for value in (dense.get("note"), provider.get("note")) if str(value or "")]
        rows.append(
            [
                str(case),
                str(spec["dataset"]),
                f"{int(spec['input_shape'][1])}->{int(spec['out_channels'])}",
                str(dense["status"]),
                str(provider["status"]),
                _fmt_float(dense["he_forward_s"]),
                _fmt_float(provider["he_forward_s"]),
                _fmt_float(_ratio(dense["he_forward_s"], provider["he_forward_s"])),
                _fmt_float(dense["hot_e2e_s"]),
                _fmt_float(provider["hot_e2e_s"]),
                _fmt_float(dense["mvm_lt_s"]),
                _fmt_float(provider["mvm_lt_s"]),
                _fmt_float(dense["activation_excl_boot_s"]),
                _fmt_float(provider["activation_excl_boot_s"]),
                _fmt_float(dense["bootstrap_s"]),
                _fmt_float(provider["bootstrap_s"]),
                _fmt_float(dense["compute_mvm_activation_bootstrap_s"]),
                _fmt_float(provider["compute_mvm_activation_bootstrap_s"]),
                _fmt_float(dense["diag_encode_s"]),
                _fmt_float(provider["diag_encode_s"]),
                _fmt_float(dense["layer_cache_turnover_s"]),
                _fmt_float(provider["layer_cache_turnover_s"]),
                _fmt_float(dense["linear_wrapper_s"]),
                _fmt_float(provider["linear_wrapper_s"]),
                _fmt_float(dense["executor_overhead_s"]),
                _fmt_float(provider["executor_overhead_s"]),
                _fmt_float(dense["runtime_load_trim_s"]),
                _fmt_float(provider["runtime_load_trim_s"]),
                _fmt_float(dense["wall_residual_s"]),
                _fmt_float(provider["wall_residual_s"]),
                _fmt_int(dense["rotations"]),
                _fmt_int(provider["rotations"]),
                _fmt_int(dense["boots"]),
                _fmt_int(provider["boots"]),
                _fmt_float(dense["e2e_mae"], digits=6),
                _fmt_float(provider["e2e_mae"], digits=6),
                _fmt_float(dense["e2e_max_abs"], digits=6),
                _fmt_float(provider["e2e_max_abs"], digits=6),
                _fmt_bool(provider["layer_mae_overall_ok"]),
                _fmt_float(provider["layer_mae_max_mae"], digits=6),
                _fmt_float(provider["layer_mae_max_abs"], digits=6),
                _gib(dense["peak_rss_gib"]),
                _gib(provider["peak_rss_gib"]),
                str(provider["runtime_mode"] or dense["runtime_mode"]),
                f"dense:{dense['result_path']}; provider:{provider['result_path']}",
                "; ".join(notes),
            ]
        )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def _network_summary_table_from_rows(rows: list[list[str]]) -> str:
    headers = [
        "input",
        "dataset",
        "I/O ch",
        "dense status",
        "Halo status",
        "dense HE forward s",
        "Halo HE forward s",
        "dense/Halo HE",
        "dense hot E2E s",
        "Halo hot E2E s",
        "dense MVM/LT s (incl pool)",
        "Halo MVM/LT s (incl pool)",
        "dense activation excl boot s",
        "Halo activation excl boot s",
        "dense bootstrap s",
        "Halo bootstrap s",
        "dense MVM+act+boot s",
        "Halo MVM+act+boot s",
        "dense diag+encode s",
        "Halo diag+encode s",
        "dense layer-cache turnover s",
        "Halo layer-cache turnover s",
        "dense linear wrapper s",
        "Halo linear wrapper s",
        "dense executor overhead s",
        "Halo executor overhead s",
        "dense load/trim s",
        "Halo load/trim s",
        "dense wall residual s",
        "Halo wall residual s",
        "dense rotations",
        "Halo rotations",
        "dense boots",
        "Halo boots",
        "dense e2e MAE",
        "Halo e2e MAE",
        "dense e2e max abs",
        "Halo e2e max abs",
        "Halo layer-MAE ok",
        "Halo layer max MAE",
        "Halo layer max abs",
        "dense RSS GiB",
        "Halo RSS GiB",
        "runtime mode",
        "result files",
        "note",
    ]
    aligns = ["---", "---", "---", "---", "---"] + ["---:"] * (len(headers) - 8) + ["---", "---", "---"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(cell) for cell in _pad_row(row, len(headers))) + " |")
    return "\n".join(lines)


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


def _block_body(text: str, marker: str) -> str:
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    _left, sep, rest = text.partition(start)
    if not sep:
        return ""
    body, sep, _right = rest.partition(end)
    if not sep:
        return ""
    return body.strip("\n")


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip().replace("\\|", "|") for cell in stripped[1:-1].split(" | ")]


def _existing_markdown_rows(text: str, marker: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in _block_body(text, marker).splitlines():
        cells = _split_markdown_row(line)
        if not cells:
            continue
        if cells[0] == "input" or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def _pad_row(row: list[str], width: int) -> list[str]:
    padded = list(row[:width])
    padded.extend([""] * max(0, int(width) - len(padded)))
    return padded


def _layer_order(layer: str) -> int:
    if str(layer) == "TOTAL":
        return 0
    if str(layer).startswith("boot-only:"):
        return 1000
    try:
        return 1 + LINEAR_LAYERS.index(str(layer))
    except ValueError:
        return 900


def _merge_case_rows(existing: list[list[str]], new_rows: list[list[str]], requested: set[tuple[str, str]]) -> list[list[str]]:
    width = 32
    keyed: dict[tuple[str, str, str], list[str]] = {}
    for row in existing:
        row = _pad_row(row, width)
        if (row[0], row[3]) in requested:
            continue
        keyed[(row[0], row[3], row[5])] = row
    for row in new_rows:
        row = _pad_row(row, width)
        keyed[(row[0], row[3], row[5])] = row
    case_order = {case: index for index, case in enumerate(CASES)}
    mode_order = {"dense": 0, "provider": 1}
    return [
        keyed[key]
        for key in sorted(
            keyed,
            key=lambda item: (
                case_order.get(item[0], 100),
                mode_order.get(item[1], 10),
                _layer_order(item[2]),
                item[2],
            ),
        )
    ]


def _result_file_parts(cell: str) -> dict[str, str]:
    parts = {"dense": "", "provider": ""}
    for piece in str(cell or "").split(";"):
        piece = piece.strip()
        if piece.startswith("dense:"):
            parts["dense"] = piece[len("dense:") :]
        elif piece.startswith("provider:"):
            parts["provider"] = piece[len("provider:") :]
    return parts


def _join_result_file_parts(parts: dict[str, str]) -> str:
    return f"dense:{parts.get('dense', '')}; provider:{parts.get('provider', '')}"


def _upgrade_summary_row(row: list[str]) -> list[str] | None:
    if len(row) == 46:
        return _pad_row(row, 46)
    legacy: list[str] | None = None
    if len(row) == 39:
        legacy = _pad_row(row, 39)
    elif len(row) == 37:
        legacy = _pad_row(row[:22] + ["", ""] + row[22:], 39)
    elif len(row) == 29:
        dense_compute = _fmt_float(
            _summary_sum(
                _cell_float(row[10]),
                _cell_float(row[12]),
                _cell_float(row[14]),
            )
        )
        provider_compute = _fmt_float(
            _summary_sum(
                _cell_float(row[11]),
                _cell_float(row[13]),
                _cell_float(row[15]),
            )
        )
        legacy = [
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
            row[10],
            row[11],
            row[12],
            row[13],
            row[14],
            row[15],
            dense_compute,
            provider_compute,
            row[16],
            row[17],
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            row[20],
            row[21],
            row[22],
            row[23],
            row[24],
            row[25],
            row[26],
            row[27],
            row[28],
        ]
    elif len(row) == 19:
        legacy = [
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            row[10],
            row[11],
            row[12],
            row[13],
            row[14],
            row[15],
            row[16],
            row[17],
            row[18],
        ]
    if legacy is not None:
        return _pad_row(legacy[:34] + [""] * 7 + legacy[34:], 46)
    return None


def _merge_summary_rows(existing: list[list[str]], new_rows: list[list[str]], modes: list[str]) -> list[list[str]]:
    width = 46
    by_case = {}
    for row in existing:
        upgraded = _upgrade_summary_row(list(row))
        if upgraded is not None:
            by_case[str(upgraded[0])] = upgraded
    dense_cols = (3, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 41)
    provider_cols = (4, 6, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 38, 39, 40, 42)
    for new_row in new_rows:
        new_row = _pad_row(new_row, width)
        case = str(new_row[0])
        old_row = by_case.get(case)
        if old_row is None or set(modes) >= {"dense", "provider"}:
            by_case[case] = new_row
            continue
        merged = list(old_row)
        for index in (0, 1, 2):
            merged[index] = new_row[index]
        if "dense" in modes:
            for index in dense_cols:
                merged[index] = new_row[index]
        if "provider" in modes:
            for index in provider_cols:
                merged[index] = new_row[index]
        ratio = _ratio(merged[5].replace(",", ""), merged[6].replace(",", ""))
        merged[7] = _fmt_float(ratio)
        if new_row[43]:
            merged[43] = new_row[43]
        files = _result_file_parts(merged[44])
        new_files = _result_file_parts(new_row[44])
        for mode in modes:
            if new_files.get(mode):
                files[mode] = new_files[mode]
        merged[44] = _join_result_file_parts(files)
        if new_row[45]:
            merged[45] = new_row[45]
        by_case[case] = merged
    case_order = {case: index for index, case in enumerate(CASES)}
    return [by_case[key] for key in sorted(by_case, key=lambda case: case_order.get(case, 100))]


def update_doc(doc_path: Path, run_root: Path, cases: list[str], modes: list[str]) -> dict[str, Any]:
    rows: list[list[str]] = []
    for case in cases:
        for mode in modes:
            rows.extend(_case_table_rows(Path(run_root), str(case), str(mode)))
    text = Path(doc_path).read_text(encoding="utf-8")
    requested = {(str(case), str(mode)) for case in cases for mode in modes}
    existing_rows = _existing_markdown_rows(text, DOC_MARKER)
    if existing_rows:
        rows = _merge_case_rows(existing_rows, rows, requested)
    table = _markdown_table(rows)
    summary_rows = [
        _split_markdown_row(line)
        for line in _network_summary_table(Path(run_root), cases).splitlines()
        if _split_markdown_row(line)
    ]
    summary_data = [
        row
        for row in summary_rows
        if row[0] != "input" and not all(set(cell) <= {"-", ":"} for cell in row)
    ]
    existing_summary = _existing_markdown_rows(text, SUMMARY_DOC_MARKER)
    if existing_summary:
        summary_data = _merge_summary_rows(existing_summary, summary_data, [str(mode) for mode in modes])
        summary_table = _network_summary_table_from_rows(summary_data)
    else:
        summary_table = _network_summary_table(Path(run_root), cases)
    text = _replace_block(text, DOC_MARKER, table)
    text = _replace_block(text, SUMMARY_DOC_MARKER, summary_table)
    Path(doc_path).write_text(text, encoding="utf-8")
    summary = {
        "run_root": str(run_root),
        "doc": str(doc_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": list(cases),
        "modes": list(modes),
    }
    _write_json(Path(run_root) / "summary.json", summary)
    return summary


def run_all(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "tmp").mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.parent.mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.write_text(str(run_root) + "\n", encoding="utf-8")

    cases = [str(case) for case in args.cases]
    modes = [str(mode) for mode in args.modes]
    env = _apply_env_defaults(os.environ, backend=str(args.backend))
    env["TMPDIR"] = str(run_root / "tmp")
    env.setdefault("XDG_CACHE_HOME", str(run_root / "xdg-cache"))
    mode_env = {
        str(mode): {
            key: str(_apply_mode_env(dict(env), str(mode)).get(key, ""))
            for key in ENV_SNAPSHOT_KEYS
        }
        for mode in modes
    }

    manifest = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "run_root": str(run_root),
        "doc": str(args.doc),
        "cases": cases,
        "modes": modes,
        "backend": str(args.backend),
        "policy": str(args.policy),
        "provider_mode": _provider_mode(str(args.policy)),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "env": {key: str(env.get(key, "")) for key in ENV_SNAPSHOT_KEYS},
        "mode_env": mode_env,
        "model_variant": "UNet22PlusOutput dim32, 22 body layers plus explicit output layer",
        "measurement": {
            "per_layer_source": "operator_breakdown_after_forward.mvm.group_rows",
            "stream_encode_s": "stream_build_map_s + stream_encode_hoist_s + stream_load_payload_s",
            "lt_accumulate_s": "stream_eval_s + stream_accumulate_s + cpp_baby_step_s + cpp_giant_step_s, with mvm_kernel_s fallback",
            "dense_lt": "independent Orion LT; dense mode disables concat fusion, unified output fusion, and bootstrap prescale fusion",
            "provider_lt": "HaloED/provider mode keeps concat fusion on auto and unified output/bootstrap prescale fusion enabled",
            "bootstrap_many": "disabled via ORION_LATTIGO_BOOTSTRAP_MANY=0",
        },
    }
    _write_json(run_root / "manifest.json", manifest)
    update_doc(Path(args.doc), run_root, cases, modes)

    active: subprocess.Popen[str] | None = None

    def _terminate(_signum: int, _frame: Any) -> None:
        if active is not None and active.poll() is None:
            active.terminate()
        raise KeyboardInterrupt

    old_int = signal.signal(signal.SIGINT, _terminate)
    old_term = signal.signal(signal.SIGTERM, _terminate)
    try:
        for case in cases:
            for mode in modes:
                env = _apply_mode_env(_apply_env_defaults(os.environ, backend=str(args.backend)), mode)
                env["TMPDIR"] = str(run_root / "tmp")
                env.setdefault("XDG_CACHE_HOME", str(run_root / "xdg-cache"))
                out_path, log_path = _case_paths(run_root, case, mode)
                if _status_ok(out_path) and not bool(args.force):
                    print(f"[{datetime.now().isoformat(timespec='seconds')}] skip {case} {mode}: {out_path}", flush=True)
                    continue
                provider_mode = _provider_mode(str(args.policy)) if mode == "provider" else ""
                spec = CASES[str(case)]
                _write_json(
                    out_path,
                    {
                        "status": "running",
                        "network": _network_name(case),
                        "mode": mode,
                        "provider_mode": provider_mode,
                        "layout_policy": str(args.policy) if mode == "provider" else "",
                        "activation": {"kind": "silu", "silu_degree": 7},
                        "input_shape": [int(v) for v in tuple(spec["input_shape"])],
                        "out_channels": int(spec["out_channels"]),
                        "runner": {
                            "script": str(Path(__file__).relative_to(REPO_ROOT)),
                            "run_root": str(run_root),
                            "case": str(case),
                            "mode": str(mode),
                            "backend": str(args.backend),
                            "policy": str(args.policy),
                            "provider_mode": provider_mode,
                            "local_time": datetime.now().isoformat(timespec="seconds"),
                            "env": {key: str(env.get(key, "")) for key in ENV_SNAPSHOT_KEYS},
                        },
                    },
                )
                update_doc(Path(args.doc), run_root, cases, modes)
                command = [
                    "/usr/bin/time",
                    "-v",
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--run-one",
                    "--case",
                    str(case),
                    "--mode",
                    str(mode),
                    "--backend",
                    str(args.backend),
                    "--policy",
                    str(args.policy),
                    "--run-root",
                    str(run_root),
                    "--out",
                    str(out_path),
                    "--seed",
                    str(args.seed),
                    "--forward-runs",
                    str(args.forward_runs),
                    "--warmup-runs",
                    str(args.warmup_runs),
                ]
                if bool(args.profile_lt):
                    command.append("--profile-lt")
                if bool(args.trace_forward_memory):
                    command.append("--trace-forward-memory")
                if bool(args.operator_breakdown):
                    command.append("--operator-breakdown")
                if bool(args.layer_mae):
                    command.append("--layer-mae")
                if bool(args.dense_layer_mae):
                    command.append("--dense-layer-mae")
                if bool(args.provider_layer_mae):
                    command.append("--provider-layer-mae")
                print(f"[{datetime.now().isoformat(timespec='seconds')}] start {case} {mode}", flush=True)
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
                    return_code = active.wait()
                active = None
                print(f"[{datetime.now().isoformat(timespec='seconds')}] done {case} {mode} rc={return_code}", flush=True)
                update_doc(Path(args.doc), run_root, cases, modes)
                if int(return_code) != 0 and not bool(args.keep_going):
                    return int(return_code)
        update_doc(Path(args.doc), run_root, cases, modes)
        return 0
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def _default_run_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUN_ROOT_BASE / f"u22_dim32_dense_provider_e2e_matrix_{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run U22+output dim32 dense/provider full-network E2E matrix with per-layer LT timing."
    )
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--doc", type=Path, default=REPO_ROOT / "docs" / "u22_orion_streaming_haloed_mainline.md")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--modes", nargs="+", choices=("dense", "provider"), default=["dense", "provider"])
    parser.add_argument("--backend", choices=("ckks", "clear"), default="ckks")
    parser.add_argument("--policy", choices=tuple(PROVIDER_MODES), default="dp_no_share_fold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--profile-lt", action="store_true")
    parser.add_argument("--trace-forward-memory", action="store_true")
    parser.add_argument("--operator-breakdown", dest="operator_breakdown", action="store_true")
    parser.add_argument("--no-operator-breakdown", dest="operator_breakdown", action="store_false")
    parser.set_defaults(operator_breakdown=True)
    parser.add_argument("--layer-mae", action="store_true", help="enable layer-MAE checks for all modes")
    parser.add_argument("--dense-layer-mae", action="store_true", help="enable layer-MAE checks for dense mode")
    parser.add_argument("--provider-layer-mae", action="store_true", help="enable layer-MAE checks for provider mode")
    parser.add_argument("--update-doc-only", action="store_true")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=tuple(CASES), default="192x192", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("dense", "provider"), default="dense", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=Path("/tmp/u22_dim32_matrix_case.json"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if bool(args.update_doc_only):
        update_doc(Path(args.doc), Path(args.run_root), [str(case) for case in args.cases], [str(mode) for mode in args.modes])
        return 0
    if bool(args.run_one):
        return run_one(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
