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

from tools.generate_unet22_compile_plan_csv import UNet22PlusOutput


DOC_MARKER = "U22_BASE32_SILU7_STREAMING_PROVIDER_E2E_TABLE"
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
    "384x288": {
        "dataset": "CVC-ClinicDB",
        "input_shape": (1, 3, 384, 288),
        "out_channels": 1,
    },
    "384x384": {
        "dataset": "Satellite cloud",
        "input_shape": (1, 4, 384, 384),
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
    "greedy": "u22_256_base32_layout_greedy",
    "always": "u22_256_base32_layout_always",
    "always_fused": "u22_256_base32_layout_always_fused",
    "fixed_max": "u22_256_base32_layout_fixedmax",
}

ENV_DEFAULTS: dict[str, str] = {
    "PYTHONUNBUFFERED": "1",
    "MALLOC_ARENA_MAX": "2",
    "ORION_COMPILE_PARALLEL_POLICY": "auto",
    "ORION_LATTIGO_STREAMING_LT": "force",
    "ORION_UNIFIED_STREAM_COMPILE_IO_NONE": "1",
    "ORION_LATTIGO_MEMORY_BOUNDED_COMPILE": "1",
    "ORION_LATTIGO_MEMORY_BOUNDED_EVAL": "1",
    "ORION_LATTIGO_BOOTSTRAP_MANY": "0",
    "ORION_UNIFIED_LT_OUTPUT_FUSION": "1",
    "ORION_LAYOUT_POLICY_RELAYOUT_KERNEL": "1",
    "ORION_LAYOUT_POLICY_PROVIDER_NATIVE_HALO": "1",
    "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS": "1",
    "ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE": "1",
    "ORION_UNIFIED_LT_FORCE_COMPILE_TRIM_EACH_TRANSFORM": "0",
    "ORION_REGION_FIRST_CLEANUP_AFTER_OUTPUTS": "1",
}

ENV_TUNING_KEYS = (
    "GOMAXPROCS",
    "ORION_COMPILE_MEMORY_RESERVE_GB",
    "ORION_UNIFIED_STREAM_COMPILE_BATCH_GB",
    "ORION_LT_COMPILE_WORKERS",
    "ORION_UNIFIED_COMPILE_WORKERS",
    "ORION_LATTIGO_COMPILE_WORKERS",
    "ORION_UNIFIED_STREAM_COMPILE_BATCH_TRANSFORMS",
    "ORION_UNIFIED_COMPILE_BATCH_TRANSFORMS",
    "ORION_UNIFIED_LOAD_BATCH_TRANSFORMS",
    "ORION_UNIFIED_CACHED_LOAD_BATCH_TRANSFORMS",
    "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS",
    "ORION_PACK_CONV_WORKERS",
    "ORION_LATTIGO_BOOTSTRAP_WORKERS",
    "ORION_DENSE_LT_SHARED_CACHE",
    "ORION_DENSE_LT_HOST_PAYLOAD_CACHE",
    "ORION_CONCAT_FUSION",
)

ENV_SNAPSHOT_KEYS = tuple(sorted(set(ENV_DEFAULTS) | set(ENV_TUNING_KEYS)))


def _safe_case(case: str) -> str:
    return str(case).replace("x", "_")


def _network_name(case: str) -> str:
    return f"u23_dim32_{_safe_case(case)}_full"


def _apply_env_defaults(env: dict[str, str]) -> dict[str, str]:
    updated = dict(env)
    for key, value in ENV_DEFAULTS.items():
        updated.setdefault(key, value)
    return updated


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
    os.environ.update(_apply_env_defaults(os.environ))
    mode = str(args.mode)
    if mode == "dense":
        os.environ["ORION_DENSE_LT_SHARED_CACHE"] = "0"
        os.environ["ORION_DENSE_LT_HOST_PAYLOAD_CACHE"] = "0"
        os.environ["ORION_CONCAT_FUSION"] = "0"
    else:
        os.environ.setdefault("ORION_CONCAT_FUSION", "1")
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
        "stream_encode_s": 0.0,
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
    for row in rows:
        if not isinstance(row, dict):
            continue
        layer = _layer_for_row(row)
        if layer is None:
            continue
        timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
        stream_build = _as_float(timing.get("stream_build_map_s"))
        stream_encode = _as_float(timing.get("stream_encode_hoist_s"))
        stream_payload = _as_float(timing.get("stream_load_payload_s"))
        stream_eval = _as_float(timing.get("stream_eval_s"))
        stream_accum = _as_float(timing.get("stream_accumulate_s"))
        baby_giant = _as_float(timing.get("cpp_baby_step_s")) + _as_float(timing.get("cpp_giant_step_s"))
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
        entry["stream_encode_s"] += stream_build + stream_encode + stream_payload
        entry["lt_accumulate_s"] += lt_accum
        entry["eval_total_s"] += _as_float(row.get("mvm_eval_total_s"))
    return stats


def _breakdown_totals(payload: dict[str, Any]) -> dict[str, Any]:
    return _metric(payload, ("operator_breakdown_after_forward", "totals")) or {}


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
    total_stream_encode = None
    total_lt_accumulate = None
    if has_payload:
        total_stream_encode = (
            _as_float(totals.get("lt_runtime_stream_build_map_s"))
            + _as_float(totals.get("lt_runtime_stream_encode_hoist_s"))
            + _as_float(totals.get("lt_runtime_stream_load_payload_s"))
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
        _fmt_float(total_stream_encode),
        _fmt_float(total_lt_accumulate),
        _fmt_float(totals.get("mvm_eval_total_s")),
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
                _fmt_float(item["stream_encode_s"]) if has_row else "",
                _fmt_float(item["lt_accumulate_s"]) if has_row else "",
                _fmt_float(item["eval_total_s"]) if has_row else "",
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
        "stream+encode s",
        "LT+accum s",
        "eval total s",
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
        "---",
        "---:",
        "---",
        "---",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(cell) for cell in row) + " |")
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


def update_doc(doc_path: Path, run_root: Path, cases: list[str], modes: list[str]) -> dict[str, Any]:
    rows: list[list[str]] = []
    for case in cases:
        for mode in modes:
            rows.extend(_case_table_rows(Path(run_root), str(case), str(mode)))
    table = _markdown_table(rows)
    text = Path(doc_path).read_text(encoding="utf-8")
    Path(doc_path).write_text(_replace_block(text, DOC_MARKER, table), encoding="utf-8")
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
    env = _apply_env_defaults(os.environ)
    env["TMPDIR"] = str(run_root / "tmp")
    env.setdefault("XDG_CACHE_HOME", str(run_root / "xdg-cache"))

    manifest = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "run_root": str(run_root),
        "doc": str(args.doc),
        "cases": cases,
        "modes": modes,
        "policy": str(args.policy),
        "provider_mode": _provider_mode(str(args.policy)),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "env": {key: str(env.get(key, "")) for key in ENV_SNAPSHOT_KEYS},
        "model_variant": "UNet22PlusOutput dim32, 22 body layers plus explicit output layer",
        "measurement": {
            "per_layer_source": "operator_breakdown_after_forward.mvm.group_rows",
            "stream_encode_s": "stream_build_map_s + stream_encode_hoist_s + stream_load_payload_s",
            "lt_accumulate_s": "stream_eval_s + stream_accumulate_s + cpp_baby_step_s + cpp_giant_step_s, with mvm_kernel_s fallback",
            "dense_unified_bsgs": "disabled via ORION_DENSE_LT_SHARED_CACHE=0 and ORION_CONCAT_FUSION=0 for dense runs",
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
    parser.add_argument("--policy", choices=tuple(PROVIDER_MODES), default="dp")
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
