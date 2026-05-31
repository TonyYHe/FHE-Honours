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

from orion.models.unet import UNet22Encoder


DOC_MARKER = "U22_BASE32_ENCODER4_NOSHARE_E2E_TABLE"
SUMMARY_DOC_MARKER = "U22_BASE32_ENCODER4_NOSHARE_SUMMARY_TABLE"
DEFAULT_RUN_ROOT_BASE = REPO_ROOT / ".tmp" / "results"
LATEST_POINTER = REPO_ROOT / ".tmp" / "latest_u22_dim32_encoder4_noshare_e2e.txt"

CASES: dict[str, dict[str, Any]] = {
    "192x192": {"dataset": "IBSR BRAIN 2D", "input_shape": (1, 1, 192, 192), "out_channels": 256},
    "224x224": {"dataset": "HanCo Hand", "input_shape": (1, 3, 224, 224), "out_channels": 256},
    "384x288": {"dataset": "CVC-ClinicDB", "input_shape": (1, 3, 384, 288), "out_channels": 256},
    "384x384": {"dataset": "Satellite cloud", "input_shape": (1, 4, 384, 384), "out_channels": 256},
}

LINEAR_LAYERS = ["enc1a", "enc1b", "enc2a", "enc2b", "enc3a", "enc3b", "enc4a", "enc4b"]
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
}

PROVIDER_MODES = {
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
    "always": "u22_256_base32_layout_always",
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
    "dp": "u22_256_base32_layout_dp",
    "dp_no_share_fold": "u22_256_base32_layout_dp_no_share_fold",
    "dp_noshare_fold": "u22_256_base32_layout_dp_no_share_fold",
    "noshare_fold": "u22_256_base32_layout_dp_no_share_fold",
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
    "ORION_PACK_CONV_WORKERS": str(CPU_COUNT),
    "ORION_DIRECT_PACK_WORKERS": str(CPU_COUNT),
    "ORION_LT_COMPILE_WORKERS": str(CPU_COUNT),
    "ORION_UNIFIED_COMPILE_WORKERS": str(CPU_COUNT),
    "ORION_LATTIGO_COMPILE_WORKERS": str(CPU_COUNT),
    "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS": str(CPU_COUNT),
    "ORION_LAYOUT_POLICY_RELAYOUT_KERNEL": "1",
    "ORION_LAYOUT_POLICY_PROVIDER_NATIVE_HALO": "1",
    "ORION_REGION_FIRST_CLEANUP_AFTER_OUTPUTS": "1",
    "ORION_CONCAT_FUSION": "1",
    "ORION_UNIFIED_LT_INDIVIDUAL_EVAL": "1",
    "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS": "0",
    "ORION_LATTIGO_UNIFIED_NO_BSGS": "0",
}

ENV_SNAPSHOT_KEYS = tuple(sorted(ENV_DEFAULTS))


def _safe_case(case: str) -> str:
    return str(case).replace("x", "_")


def _network_name(case: str) -> str:
    return f"u22_dim32_encoder4_{_safe_case(case)}"


def _apply_env_defaults(env: dict[str, str]) -> dict[str, str]:
    updated = dict(env)
    for key, value in ENV_DEFAULTS.items():
        updated.setdefault(key, value)
    updated["GOMAXPROCS"] = "1"
    updated["ORION_SINGLE_SLOT_LAYER_CACHE"] = "1"
    updated["ORION_LATTIGO_BOOTSTRAP_MANY"] = "0"
    updated["ORION_UNIFIED_LT_INDIVIDUAL_EVAL"] = "1"
    updated["ORION_UNIFIED_LT_SHARED_ROTATION_KEYS"] = "0"
    updated["ORION_LATTIGO_UNIFIED_NO_BSGS"] = "0"
    updated["ORION_CONCAT_FUSION"] = "1"
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

        def _builder(
            *,
            activation: str | None = None,
            silu_degree: int = 31,
            _input_shape: tuple[int, int, int, int] = input_shape,
        ):
            return UNet22Encoder(
                dataset="kvasir_polyp_256",
                in_channels=int(_input_shape[1]),
                base_channels=32,
                activation=str(activation or "silu"),
                silu_degree=int(silu_degree),
            )

        base.NETWORKS[_network_name(case)] = {
            "label": f"U22 encoder4 dim32 {case}",
            "model": "UNet22Encoder",
            "dataset": str(spec["dataset"]),
            "input_shape": input_shape,
            "provider_mode": PROVIDER_MODES["fixed_max"],
            "config": base._r18_config,
            "builder": _builder,
            "scope": "encoder4",
            "base_dim": 32,
            "out_channels": int(spec["out_channels"]),
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
    return case_dir / f"{mode}_encoder4_e2e.json", case_dir / f"{mode}_encoder4_e2e.log"


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
        "name": "UNet22Encoder",
        "linear_layers": len(LINEAR_LAYERS),
        "layers": list(LINEAR_LAYERS),
        "base_dim": 32,
        "input_shape": [int(v) for v in tuple(spec["input_shape"])],
        "out_channels": int(spec["out_channels"]),
        "dataset": str(spec["dataset"]),
        "activation": "SiLU7",
        "scope": "encoder4_only_no_bottleneck",
        "sharing": "dense independent LT; provider ORION_UNIFIED_LT_INDIVIDUAL_EVAL=1",
    }
    _write_json(out_path, payload)


def run_one(args: argparse.Namespace) -> int:
    os.environ.update(_apply_env_defaults(os.environ))
    env_snapshot = dict(os.environ)
    base = _register_networks()
    mode = str(args.mode)
    out_path = Path(args.out)
    provider_mode = _provider_mode(str(args.policy)) if mode == "provider" else ""
    started_at = time.perf_counter()
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
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return float(number) if math.isfinite(number) else 0.0


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
    if not isinstance(report, dict):
        return None
    value = report.get("total_rotation_eval_count_estimate")
    if value is None:
        rows = report.get("rows") or []
        if isinstance(rows, list):
            value = sum(
                int((row.get("stats") or {}).get("rotation_eval_count_estimate", 0) or 0)
                for row in rows
                if isinstance(row, dict)
            )
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
        "compile_group_count": 0,
        "diagonals": 0,
        "rotation_estimate": 0,
        "layer_cache_encode_s": 0.0,
        "layer_cache_key_prepare_s": 0.0,
        "layer_cache_evict_s": 0.0,
        "layer_cache_turnover_s": 0.0,
        "lt_accumulate_s": 0.0,
        "eval_total_s": 0.0,
    }


def _compile_stats(payload: dict[str, Any], stats: dict[str, dict[str, Any]]) -> None:
    rows = _metric(payload, ("operator_breakdown_after_compile", "group_rows")) or []
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        layer = _layer_for_row(row)
        if layer is None:
            continue
        profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
        entry = stats[layer]
        entry["compile_group_count"] += 1
        entry["diagonals"] += int(profile.get("diag_index_count") or profile.get("diag_data_count") or 0)
        entry["transform_count"] += int(profile.get("transform_count") or 0)


def _rotation_stats(payload: dict[str, Any], stats: dict[str, dict[str, Any]]) -> None:
    report = payload.get("rotation_report_after_forward") or payload.get("rotation_report_after_compile") or {}
    rows = report.get("rows") if isinstance(report, dict) else []
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        layer = _layer_for_row(row)
        if layer is None:
            nodes = [str(node) for node in row.get("nodes", []) or []]
            layer = next((node for node in nodes if node in LINEAR_LAYERS), None)
        if layer is None:
            continue
        row_stats = row.get("stats") if isinstance(row.get("stats"), dict) else {}
        stats[layer]["rotation_estimate"] += int(row_stats.get("rotation_eval_count_estimate", 0) or 0)
        if not stats[layer]["transform_count"] and row_stats.get("transform_count") is not None:
            stats[layer]["transform_count"] = int(row_stats.get("transform_count") or 0)


def _dense_log_diagonal_stats(result_path: Path, stats: dict[str, dict[str, Any]]) -> None:
    """Recover dense single-slot diagonal counts from the compile log.

    In single-slot mode the dense backend does not keep compile-time group rows in
    the JSON, but the compile log still emits the packed matrix diagonal count in
    the same order as the "Packing <layer>:" lines.
    """
    if sum(int(value["diagonals"]) for value in stats.values()) > 0:
        return
    log_path = result_path.with_suffix(".log")
    if not log_path.exists():
        return
    packing_order: list[str] = []
    diagonal_counts: list[int] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("Packing ") and stripped.endswith(":"):
            packing_order.append(stripped[len("Packing ") : -1])
        if "# diagonals =" in stripped:
            try:
                diagonal_counts.append(int(stripped.rsplit("=", 1)[1].strip().replace(",", "")))
            except ValueError:
                continue
    for layer, count in zip(packing_order, diagonal_counts, strict=False):
        if layer in stats:
            stats[layer]["diagonals"] = int(count)


def _forward_stats(payload: dict[str, Any], stats: dict[str, dict[str, Any]]) -> None:
    rows = _metric(payload, ("operator_breakdown_after_forward", "mvm", "group_rows")) or []
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        layer = _layer_for_row(row)
        if layer is None:
            continue
        timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
        layer_cache_encode = _as_float(row.get("lt_layer_cache_encode_s")) or _as_float(timing.get("layer_cache_encode_s"))
        layer_cache_key_prepare = _as_float(row.get("lt_layer_cache_key_prepare_s")) or _as_float(
            timing.get("layer_cache_key_prepare_s")
        )
        layer_cache_evict = _as_float(row.get("lt_layer_cache_evict_s")) or _as_float(timing.get("layer_cache_evict_s"))
        layer_cache_turnover = _as_float(row.get("lt_layer_cache_turnover_s")) or (
            layer_cache_encode + layer_cache_key_prepare + layer_cache_evict
        )
        baby_giant = _as_float(timing.get("cpp_baby_step_s")) + _as_float(timing.get("cpp_giant_step_s"))
        lt_accum = _as_float(timing.get("stream_eval_s")) + _as_float(timing.get("stream_accumulate_s")) + baby_giant
        if lt_accum <= 0.0:
            lt_accum = _as_float(row.get("mvm_kernel_s"))
        entry = stats[layer]
        entry["row_count"] += 1
        if row.get("transform_count"):
            entry["transform_count"] += int(row.get("transform_count") or 0)
        entry["layer_cache_encode_s"] += layer_cache_encode
        entry["layer_cache_key_prepare_s"] += layer_cache_key_prepare
        entry["layer_cache_evict_s"] += layer_cache_evict
        entry["layer_cache_turnover_s"] += layer_cache_turnover
        entry["lt_accumulate_s"] += lt_accum
        entry["eval_total_s"] += _as_float(row.get("mvm_eval_total_s"))


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


def _bootstrap_stats(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats = {layer: {"count": 0, "s": 0.0, "nodes": []} for layer in LINEAR_LAYERS}
    records: dict[str, dict[str, Any]] = {}
    report = payload.get("bootstrap_report_after_forward") or payload.get("bootstrap_report_after_compile") or {}
    for row in report.get("rows", []) or [] if isinstance(report, dict) else []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        records[name] = {"name": name, "count": int(row.get("runtime_call_count", 0) or 0), "s": 0.0}
    breakdown_rows = _metric(payload, ("operator_breakdown_after_forward", "bootstrap", "rows")) or []
    if isinstance(breakdown_rows, list):
        for row in breakdown_rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", ""))
            record = records.setdefault(name, {"name": name, "count": 0, "s": 0.0})
            record["count"] = int(row.get("runtime_call_count", record.get("count", 0)) or 0)
            record["s"] = _as_float(row.get("bootstrap_s"))
    for record in records.values():
        owner = _bootstrap_owner(str(record.get("name", "")))
        if owner not in stats:
            continue
        stats[owner]["count"] += int(record.get("count", 0) or 0)
        stats[owner]["s"] += _as_float(record.get("s"))
        stats[owner]["nodes"].append(_bootstrap_anchor_name(str(record.get("name", ""))))
    return stats


def _layer_stats(payload: dict[str, Any], result_path: Path | None = None) -> dict[str, dict[str, Any]]:
    stats = {layer: _empty_layer_stats() for layer in LINEAR_LAYERS}
    _compile_stats(payload, stats)
    _rotation_stats(payload, stats)
    _forward_stats(payload, stats)
    if result_path is not None and str(payload.get("mode", "")) == "dense":
        _dense_log_diagonal_stats(result_path, stats)
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
    direct = _summary_total(totals, "unattributed_he_forward_s")
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
    he_forward_s = _forward_timing(payload or {}, "he_forward")
    boot_stats = _bootstrap_stats(payload or {}) if has_payload else {layer: {"count": 0, "s": 0.0, "nodes": []} for layer in LINEAR_LAYERS}
    layer_stats = (
        _layer_stats(payload or {}, result_path)
        if has_payload
        else {layer: _empty_layer_stats() for layer in LINEAR_LAYERS}
    )
    total_boot_s = sum(_as_float(value["s"]) for value in boot_stats.values())
    total_lt_s = sum(_as_float(value["lt_accumulate_s"]) for value in layer_stats.values())
    total_row = [
        str(case),
        str(spec["dataset"]),
        f"{int(spec['input_shape'][1])}->{int(spec['out_channels'])}",
        str(mode),
        status,
        "TOTAL",
        _fmt_int(sum(int(value["compile_group_count"]) for value in layer_stats.values())),
        _fmt_int(sum(int(value["transform_count"]) for value in layer_stats.values())),
        _fmt_int(sum(int(value["diagonals"]) for value in layer_stats.values())),
        _fmt_int(_rotation_count(payload or {})),
        _fmt_float(sum(_as_float(value["layer_cache_encode_s"]) for value in layer_stats.values())),
        _fmt_float(sum(_as_float(value["layer_cache_key_prepare_s"]) for value in layer_stats.values())),
        _fmt_float(sum(_as_float(value["layer_cache_evict_s"]) for value in layer_stats.values())),
        _fmt_float(sum(_as_float(value["layer_cache_turnover_s"]) for value in layer_stats.values())),
        _fmt_float(total_lt_s),
        _fmt_float(sum(_as_float(value["eval_total_s"]) for value in layer_stats.values())),
        _fmt_int(_bootstrap_count(payload or {})),
        _fmt_float(total_boot_s),
        "",
        _fmt_float(timing.get("compile") if isinstance(timing, dict) else None),
        _fmt_float(he_forward_s),
        _runtime_mode(payload or {}),
        _gib(_metric(payload or {}, ("runner", "maxrss_bytes"))),
        str(result_path),
        _first_error(result_path, payload),
    ]
    rows = [total_row]
    for layer in LINEAR_LAYERS:
        item = layer_stats[layer]
        boot_item = boot_stats[layer]
        has_row = bool(item["row_count"] or item["compile_group_count"] or item["rotation_estimate"])
        rows.append(
            [
                str(case),
                str(spec["dataset"]),
                f"{int(spec['input_shape'][1])}->{int(spec['out_channels'])}",
                str(mode),
                status if has_row else ("pending" if payload is None else "no-row"),
                layer,
                _fmt_int(item["compile_group_count"]) if has_row else "",
                _fmt_int(item["transform_count"]) if has_row else "",
                _fmt_int(item["diagonals"]) if has_row else "",
                _fmt_int(item["rotation_estimate"]) if has_row else "",
                _fmt_float(item["layer_cache_encode_s"]) if has_row else "",
                _fmt_float(item["layer_cache_key_prepare_s"]) if has_row else "",
                _fmt_float(item["layer_cache_evict_s"]) if has_row else "",
                _fmt_float(item["layer_cache_turnover_s"]) if has_row else "",
                _fmt_float(item["lt_accumulate_s"]) if has_row else "",
                _fmt_float(item["eval_total_s"]) if has_row else "",
                _fmt_int(boot_item["count"]) if boot_item["count"] else "",
                _fmt_float(boot_item["s"]) if boot_item["count"] else "",
                ",".join(str(node) for node in boot_item["nodes"][:6]),
                "",
                "",
                _runtime_mode(payload or {}),
                "",
                str(result_path),
                "",
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
        "groups",
        "transforms",
        "diagonals",
        "rotations",
        "diag+encode s",
        "key prep s",
        "evict s",
        "turnover s",
        "LT+accum s",
        "eval total s",
        "boot count",
        "boot s",
        "boot nodes",
        "compile s",
        "HE forward s",
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
        "---",
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
    boot_stats = _bootstrap_stats(payload)
    layer_stats = (
        _layer_stats(payload, result_path)
        if has_payload
        else {layer: _empty_layer_stats() for layer in LINEAR_LAYERS}
    )
    totals = _breakdown_totals(payload)
    he_forward_s = _forward_timing(payload, "he_forward")
    mvm_lt_s = _summary_total(totals, "mvm_kernel_s")
    activation_s = _summary_total(totals, "activation_excluding_bootstrap_s", "activation_s")
    bootstrap_s = _summary_total(totals, "bootstrap_s")
    diag_encode_s = _summary_total(totals, "lt_layer_cache_encode_s")
    if bootstrap_s is None and payload:
        bootstrap_s = sum(_as_float(value["s"]) for value in boot_stats.values())
    if mvm_lt_s is None and payload:
        mvm_lt_s = sum(_as_float(value["lt_accumulate_s"]) for value in layer_stats.values())
    return {
        "status": str(payload.get("status", "pending" if not has_payload else "unknown")),
        "compile_s": timing.get("compile") if isinstance(timing, dict) else None,
        "he_forward_s": he_forward_s,
        "mvm_lt_s": mvm_lt_s,
        "activation_excl_boot_s": activation_s,
        "bootstrap_s": bootstrap_s,
        "diag_encode_s": diag_encode_s,
        "unattributed_s": _summary_unattributed(
            totals,
            he_forward_s,
            [mvm_lt_s, activation_s, bootstrap_s, diag_encode_s],
        ),
        "rotations": _rotation_count(payload) if has_payload else None,
        "boots": _bootstrap_count(payload) if has_payload else None,
        "diagonals": sum(int(value["diagonals"]) for value in layer_stats.values()) if has_payload else None,
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
        "dense status",
        "Halo status",
        "dense HE forward s",
        "Halo HE forward s",
        "dense/Halo HE",
        "dense MVM/LT s (incl pool)",
        "Halo MVM/LT s (incl pool)",
        "dense activation excl boot s",
        "Halo activation excl boot s",
        "dense bootstrap s",
        "Halo bootstrap s",
        "dense diag+encode s",
        "Halo diag+encode s",
        "dense unattributed s",
        "Halo unattributed s",
        "dense rotations",
        "Halo rotations",
        "dense diagonals",
        "Halo diagonals",
        "dense boots",
        "Halo boots",
        "dense RSS GiB",
        "Halo RSS GiB",
        "result files",
        "note",
    ]
    aligns = ["---", "---", "---", "---"] + ["---:"] * (len(headers) - 6) + ["---", "---"]
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
                str(dense["status"]),
                str(provider["status"]),
                _fmt_float(dense["he_forward_s"]),
                _fmt_float(provider["he_forward_s"]),
                _fmt_float(_ratio(dense["he_forward_s"], provider["he_forward_s"])),
                _fmt_float(dense["mvm_lt_s"]),
                _fmt_float(provider["mvm_lt_s"]),
                _fmt_float(dense["activation_excl_boot_s"]),
                _fmt_float(provider["activation_excl_boot_s"]),
                _fmt_float(dense["bootstrap_s"]),
                _fmt_float(provider["bootstrap_s"]),
                _fmt_float(dense["diag_encode_s"]),
                _fmt_float(provider["diag_encode_s"]),
                _fmt_float(dense["unattributed_s"]),
                _fmt_float(provider["unattributed_s"]),
                _fmt_int(dense["rotations"]),
                _fmt_int(provider["rotations"]),
                _fmt_int(dense["diagonals"]),
                _fmt_int(provider["diagonals"]),
                _fmt_int(dense["boots"]),
                _fmt_int(provider["boots"]),
                _gib(dense["peak_rss_gib"]),
                _gib(provider["peak_rss_gib"]),
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
        "dense status",
        "Halo status",
        "dense HE forward s",
        "Halo HE forward s",
        "dense/Halo HE",
        "dense MVM/LT s (incl pool)",
        "Halo MVM/LT s (incl pool)",
        "dense activation excl boot s",
        "Halo activation excl boot s",
        "dense bootstrap s",
        "Halo bootstrap s",
        "dense diag+encode s",
        "Halo diag+encode s",
        "dense unattributed s",
        "Halo unattributed s",
        "dense rotations",
        "Halo rotations",
        "dense diagonals",
        "Halo diagonals",
        "dense boots",
        "Halo boots",
        "dense RSS GiB",
        "Halo RSS GiB",
        "result files",
        "note",
    ]
    aligns = ["---", "---", "---", "---"] + ["---:"] * (len(headers) - 6) + ["---", "---"]
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


def _ensure_doc_section(text: str) -> str:
    if f"<!-- {SUMMARY_DOC_MARKER}_START -->" in text and f"<!-- {DOC_MARKER}_START -->" in text:
        return text
    section = f"""

## Step 1c: Dim32 Encoder4 No-Sharing E2E

Main E2E comparison for the paper table: U22 encoder Conv2d stages only (`enc1a..enc4b`), no bottleneck and no decoder. Dense uses independent Orion LTs. HaloED provider uses encoder Conv2d provider lowering with native/halo output materialization and `ORION_UNIFIED_LT_INDIVIDUAL_EVAL=1`, so each LT preserves its own BSGS and there is no cross-LT shared-cache evaluation. Single-slot layer cache is used only for memory feasibility; `bootstrap_many` is disabled.

<!-- {SUMMARY_DOC_MARKER}_START -->
pending
<!-- {SUMMARY_DOC_MARKER}_END -->

<!-- {DOC_MARKER}_START -->
pending
<!-- {DOC_MARKER}_END -->
"""
    return text.rstrip() + section + "\n"


def _block_body(text: str, marker: str) -> str:
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    _left, sep, rest = text.partition(start)
    if not sep:
        return ""
    body, sep, _right = rest.partition(end)
    if not sep:
        return ""
    return body.strip()


def _split_markdown_row(line: str) -> list[str]:
    stripped = str(line).strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip().replace("\\|", "|") for cell in stripped.strip("|").split("|")]


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
    try:
        return 1 + LINEAR_LAYERS.index(str(layer))
    except ValueError:
        return 900


def _merge_case_rows(existing: list[list[str]], new_rows: list[list[str]], requested: set[tuple[str, str]]) -> list[list[str]]:
    width = 25
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
    if len(row) == 27:
        return _pad_row(row, 27)
    if len(row) == 21:
        return [
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            "",
            "",
            row[15],
            row[16],
            "",
            "",
            "",
            "",
            row[9],
            row[10],
            row[11],
            row[12],
            row[13],
            row[14],
            row[17],
            row[18],
            row[19],
            row[20],
        ]
    return None


def _merge_summary_rows(existing: list[list[str]], new_rows: list[list[str]], modes: list[str]) -> list[list[str]]:
    width = 27
    by_case = {}
    for row in existing:
        upgraded = _upgrade_summary_row(list(row))
        if upgraded is not None:
            by_case[str(upgraded[0])] = upgraded
    dense_cols = (2, 4, 7, 9, 11, 13, 15, 17, 19, 21, 23)
    provider_cols = (3, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24)
    for new_row in new_rows:
        new_row = _pad_row(new_row, width)
        case = str(new_row[0])
        old_row = by_case.get(case)
        if old_row is None or set(modes) >= {"dense", "provider"}:
            by_case[case] = new_row
            continue
        merged = list(old_row)
        for index in (0, 1):
            merged[index] = new_row[index]
        if "dense" in modes:
            for index in dense_cols:
                merged[index] = new_row[index]
        if "provider" in modes:
            for index in provider_cols:
                merged[index] = new_row[index]
        merged[6] = _fmt_float(_ratio(merged[4].replace(",", ""), merged[5].replace(",", "")))
        files = _result_file_parts(merged[25])
        new_files = _result_file_parts(new_row[25])
        for mode in modes:
            if new_files.get(mode):
                files[mode] = new_files[mode]
        merged[25] = _join_result_file_parts(files)
        if new_row[26]:
            merged[26] = new_row[26]
        by_case[case] = merged
    case_order = {case: index for index, case in enumerate(CASES)}
    return [by_case[key] for key in sorted(by_case, key=lambda case: case_order.get(case, 100))]


def update_doc(doc_path: Path, run_root: Path, cases: list[str], modes: list[str]) -> dict[str, Any]:
    rows: list[list[str]] = []
    for case in cases:
        for mode in modes:
            rows.extend(_case_table_rows(Path(run_root), str(case), str(mode)))
    text = Path(doc_path).read_text(encoding="utf-8")
    text = _ensure_doc_section(text)
    requested = {(str(case), str(mode)) for case in cases for mode in modes}
    existing_rows = _existing_markdown_rows(text, DOC_MARKER)
    if existing_rows:
        rows = _merge_case_rows(existing_rows, rows, requested)
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
    text = _replace_block(text, DOC_MARKER, _markdown_table(rows))
    text = _replace_block(text, SUMMARY_DOC_MARKER, _network_summary_table_from_rows(summary_data))
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
        "model_variant": "UNet22Encoder dim32, enc1a..enc4b only, no bottleneck",
        "measurement": {
            "per_layer_source": "operator_breakdown_after_forward.mvm.group_rows + rotation_report_after_forward.rows",
            "sharing": "dense independent LT; provider individual LT evaluation with per-LT BSGS preserved",
            "bootstrap_many": "disabled via ORION_LATTIGO_BOOTSTRAP_MANY=0",
            "primary_total_time": "HE forward time; layer compute and boot time are also reported separately",
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
    return DEFAULT_RUN_ROOT_BASE / f"u22_dim32_encoder4_noshare_e2e_{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run U22 dim32 encoder-only no-sharing dense/provider E2E matrix.")
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--doc", type=Path, default=REPO_ROOT / "docs" / "u22_orion_streaming_haloed_mainline.md")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--modes", nargs="+", choices=("dense", "provider"), default=["dense", "provider"])
    parser.add_argument("--policy", choices=tuple(PROVIDER_MODES), default="fixed_max")
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
    parser.add_argument("--out", type=Path, default=Path("/tmp/u22_encoder4_case.json"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if bool(args.update_doc_only):
        update_doc(Path(args.doc), Path(args.run_root), [str(case) for case in args.cases], [str(mode) for mode in args.modes])
        return 0
    if bool(args.run_one):
        return run_one(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
