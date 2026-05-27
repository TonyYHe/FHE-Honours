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

DOC_MARKER = "U22_BASE32_SILU7_STREAMING_PROVIDER_E2E_TABLE"
DEFAULT_RUN_ROOT_BASE = Path("/run/media/anakano/7TB/haloed-cache")
LATEST_POINTER = REPO_ROOT / ".tmp" / "latest_u22_base32_silu7_streaming_provider_e2e.txt"

SIZES: dict[str, tuple[int, int, int, int]] = {
    "192x192": (1, 3, 192, 192),
    "224x224": (1, 3, 224, 224),
    "384x288": (1, 3, 384, 288),
    "384x384": (1, 3, 384, 384),
}

POLICY_PROVIDER_SUFFIX: dict[str, str] = {
    "fixed_max": "fixedmax",
    "always": "always",
    "always_fused": "always_fused",
    "greedy": "greedy",
    "dp": "dp",
}

ENV_DEFAULTS: dict[str, str] = {
    "PYTHONUNBUFFERED": "1",
    "MALLOC_ARENA_MAX": "2",
    "ORION_COMPILE_PARALLEL_POLICY": "manual",
    "ORION_UNIFIED_SINGLE_SLOT_LAYER_CACHE": "1",
    "ORION_SINGLE_SLOT_ENCODE_WORKERS": "16",
    "ORION_LATTIGO_STREAMING_LT": "0",
    "ORION_UNIFIED_STREAM_COMPILE_IO_NONE": "0",
    "ORION_LATTIGO_MEMORY_BOUNDED_COMPILE": "0",
    "ORION_LATTIGO_MEMORY_BOUNDED_EVAL": "0",
    "ORION_LATTIGO_BOOTSTRAP_MANY": "1",
    "ORION_UNIFIED_LT_OUTPUT_FUSION": "1",
    "ORION_LAYOUT_POLICY_RELAYOUT_KERNEL": "1",
    "ORION_LAYOUT_POLICY_PROVIDER_NATIVE_HALO": "1",
    "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS": "1",
    "ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE": "1",
    "ORION_REGION_FIRST_CLEANUP_AFTER_OUTPUTS": "1",
}


def _safe_size(size: str) -> str:
    return str(size).replace("x", "_")


def _network_name(size: str) -> str:
    return f"u22_{_safe_size(size)}_base32_full"


def _provider_mode(policy: str) -> str:
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized == "fixedmax":
        normalized = "fixed_max"
    if normalized not in POLICY_PROVIDER_SUFFIX:
        raise ValueError(f"unknown layout policy {policy!r}; expected one of {sorted(POLICY_PROVIDER_SUFFIX)}")
    return f"u22_256_base32_layout_{POLICY_PROVIDER_SUFFIX[normalized]}"


def _register_networks() -> Any:
    from orion.models.unet import UNet22
    from tools import run_lattigo_e2e_compare as base

    def _build_full(*, activation: str | None = None, silu_degree: int = 31):
        return UNet22(
            dataset="kvasir_polyp_256",
            base_dim=32,
            activation=activation,
            silu_degree=int(silu_degree),
        )

    for size, shape in SIZES.items():
        base.NETWORKS[_network_name(size)] = {
            "label": f"U22 {size} base32 full",
            "model": "UNet22",
            "dataset": "kvasir_polyp_256",
            "input_shape": tuple(int(v) for v in shape),
            "provider_mode": "u22_256_base32",
            "config": base._r18_config,
            "builder": _build_full,
            "scope": "full",
            "base_dim": 32,
        }
    return base


def _apply_env_defaults(env: dict[str, str]) -> dict[str, str]:
    updated = dict(env)
    for key, value in ENV_DEFAULTS.items():
        updated.setdefault(key, value)
    return updated


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
    # Linux returns ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _annotate_result(
    out_path: Path,
    *,
    size: str,
    policy: str,
    provider_mode: str,
    run_root: Path,
    started_at: float,
    env_snapshot: dict[str, str],
) -> None:
    payload = _read_json(out_path) or {}
    payload["runner"] = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "run_root": str(run_root),
        "size": str(size),
        "policy": str(policy),
        "provider_mode": str(provider_mode),
        "local_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": float(time.perf_counter() - started_at),
        "maxrss_bytes": int(_resource_maxrss_bytes()),
        "env": {key: str(env_snapshot.get(key, "")) for key in sorted(ENV_DEFAULTS)},
    }
    _write_json(out_path, payload)


def run_one(args: argparse.Namespace) -> int:
    os.environ.update(_apply_env_defaults(os.environ))
    env_snapshot = dict(os.environ)
    base = _register_networks()
    started_at = time.perf_counter()
    out_path = Path(args.out)
    provider_mode = _provider_mode(str(args.policy))
    try:
        base._run_one(
            network=_network_name(str(args.size)),
            backend="lattigo",
            mode="provider",
            out_path=out_path,
            seed=int(args.seed),
            compile_only=False,
            forward_runs=int(args.forward_runs),
            warmup_runs=int(args.warmup_runs),
            profile_modules=bool(args.profile_modules),
            profile_lt=bool(args.profile_lt),
            trace_forward_memory=bool(args.trace_forward_memory),
            operator_breakdown=bool(args.operator_breakdown),
            provider_mode_override=provider_mode,
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
            size=str(args.size),
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


def _runtime_mode(payload: dict[str, Any]) -> str:
    return str(
        payload.get("runtime_fairness_mode")
        or _metric(payload, ("measured_runtime_fairness_timing", "runtime_fairness_mode"))
        or _metric(payload, ("runtime_fairness_timing_after_forward", "runtime_fairness_mode"))
        or ""
    )


def _forward_timing(payload: dict[str, Any], key: str) -> float | None:
    timing = (
        payload.get("measured_forward_mean_timing_s")
        or payload.get("forward_mean_timing_s")
        or payload.get("timing_s")
        or {}
    )
    if not isinstance(timing, dict):
        return None
    value = timing.get(key)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _breakdown_timing(payload: dict[str, Any], key: str) -> float | None:
    breakdown = payload.get("operator_breakdown_after_forward") or {}
    totals = breakdown.get("totals", {}) if isinstance(breakdown, dict) else {}
    if not isinstance(totals, dict):
        return None
    value = totals.get(str(key))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rotation_count(payload: dict[str, Any]) -> int | None:
    report = payload.get("rotation_report_after_forward") or payload.get("rotation_report_after_compile") or {}
    value = report.get("total_rotation_eval_count_estimate") if isinstance(report, dict) else None
    return None if value is None else int(value)


def _bootstrap_count(payload: dict[str, Any]) -> int | None:
    report = payload.get("bootstrap_report_after_forward") or payload.get("bootstrap_report_after_compile") or {}
    value = report.get("count") if isinstance(report, dict) else None
    return None if value is None else int(value)


def _layout_policy(payload: dict[str, Any]) -> str:
    attach = payload.get("attach_audit") or {}
    graph = attach.get("graph_audit") if isinstance(attach, dict) else {}
    if isinstance(graph, dict):
        return str(graph.get("layout_policy") or graph.get("layout_policy_runtime") or "")
    return ""


def _first_error(path: Path, payload: dict[str, Any] | None) -> str:
    if isinstance(payload, dict):
        error = payload.get("error") or payload.get("error_type")
        if error:
            return str(error).replace("\n", " ")[:120]
    log_path = path.with_suffix(".log")
    if log_path.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        return " ".join(tail)[-160:]
    return ""


def _format_float(value: Any, digits: int = 1) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.{digits}f}"


def _format_int(value: Any) -> str:
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


def _case_row(run_root: Path, size: str) -> tuple[list[str], dict[str, Any]]:
    result_path = run_root / _safe_size(size) / "provider_e2e.json"
    payload = _read_json(result_path)
    status = "pending" if payload is None else str(payload.get("status", "unknown"))
    timing = payload.get("timing_s", {}) if isinstance(payload, dict) else {}
    runner = payload.get("runner", {}) if isinstance(payload, dict) else {}
    encrypt_s = _forward_timing(payload or {}, "encrypt")
    he_forward_s = _forward_timing(payload or {}, "he_forward")
    decode_s = _forward_timing(payload or {}, "decrypt_decode")
    hot_s = None
    if encrypt_s is not None and he_forward_s is not None and decode_s is not None:
        hot_s = float(encrypt_s + he_forward_s + decode_s)
    mvm_s = _breakdown_timing(payload or {}, "mvm_kernel_s")
    act_s = _breakdown_timing(payload or {}, "activation_s")
    boot_s = _breakdown_timing(payload or {}, "bootstrap_s")
    load_encode_s = _breakdown_timing(payload or {}, "lt_runtime_load_encode_s")
    layer_turnover_s = _breakdown_timing(payload or {}, "lt_layer_cache_turnover_s")
    layer_encode_s = _breakdown_timing(payload or {}, "lt_layer_cache_encode_s")
    layer_key_prepare_s = _breakdown_timing(payload or {}, "lt_layer_cache_key_prepare_s")
    layer_evict_s = _breakdown_timing(payload or {}, "lt_layer_cache_evict_s")
    unattributed_s = _breakdown_timing(payload or {}, "unattributed_he_forward_s")
    row = [
        str(size),
        status,
        _layout_policy(payload or {}),
        "yes" if (payload or {}).get("bootstrap_many_enabled") is True else "",
        _format_float(timing.get("compile") if isinstance(timing, dict) else None),
        _format_float(encrypt_s),
        _format_float(he_forward_s),
        _format_float(decode_s),
        _format_float(hot_s),
        _format_float(mvm_s),
        _format_float(act_s),
        _format_float(boot_s),
        _format_float(load_encode_s),
        _format_float(layer_turnover_s),
        _format_float(unattributed_s),
        _runtime_mode(payload or {}),
        _format_int(_rotation_count(payload or {})),
        _format_int(_bootstrap_count(payload or {})),
        _format_int((payload or {}).get("input_ciphertext_count")),
        _format_int((payload or {}).get("output_ciphertext_count")),
        _format_float(_metric(payload or {}, ("mae_vs_clear", "mae")), digits=3),
        _gib(runner.get("maxrss_bytes") if isinstance(runner, dict) else None),
        "" if payload is None else str(result_path),
        _first_error(result_path, payload),
    ]
    summary = {
        "size": str(size),
        "status": status,
        "result_path": str(result_path),
        "compile_s": timing.get("compile") if isinstance(timing, dict) else None,
        "encrypt_s": encrypt_s,
        "he_forward_s": he_forward_s,
        "decrypt_decode_s": decode_s,
        "hot_s": hot_s,
        "mvm_kernel_s": mvm_s,
        "activation_s": act_s,
        "bootstrap_s": boot_s,
        "lt_runtime_load_encode_s": load_encode_s,
        "lt_layer_cache_turnover_s": layer_turnover_s,
        "lt_layer_cache_encode_s": layer_encode_s,
        "lt_layer_cache_key_prepare_s": layer_key_prepare_s,
        "lt_layer_cache_evict_s": layer_evict_s,
        "unattributed_he_forward_s": unattributed_s,
        "runtime_mode": _runtime_mode(payload or {}),
        "rotation_count": _rotation_count(payload or {}),
        "bootstrap_count": _bootstrap_count(payload or {}),
        "input_ciphertext_count": (payload or {}).get("input_ciphertext_count"),
        "output_ciphertext_count": (payload or {}).get("output_ciphertext_count"),
        "mae": _metric(payload or {}, ("mae_vs_clear", "mae")),
        "maxrss_bytes": runner.get("maxrss_bytes") if isinstance(runner, dict) else None,
        "error": _first_error(result_path, payload),
    }
    return row, summary


def _markdown_table(rows: list[list[str]]) -> str:
    headers = [
        "input",
        "status",
        "planner",
        "bootmany",
        "compile s",
        "encrypt s",
        "HE forward s",
        "decode s",
        "hot E2E s",
        "MVM kernel s",
        "ACT s",
        "boot s",
        "LT load/enc s",
        "layer turnover s",
        "unattrib HE s",
        "runtime mode",
        "rotations",
        "boot",
        "input ct",
        "output ct",
        "MAE",
        "peak RSS GiB",
        "result file",
        "note",
    ]
    aligns = [
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
        "---",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---",
        "---",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
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


def update_doc(doc_path: Path, run_root: Path, sizes: list[str]) -> dict[str, Any]:
    rows: list[list[str]] = []
    summaries: list[dict[str, Any]] = []
    for size in sizes:
        row, summary = _case_row(run_root, str(size))
        rows.append(row)
        summaries.append(summary)
    table = _markdown_table(rows)
    text = doc_path.read_text(encoding="utf-8")
    doc_path.write_text(_replace_block(text, DOC_MARKER, table), encoding="utf-8")
    summary_payload = {
        "run_root": str(run_root),
        "doc": str(doc_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "sizes": summaries,
    }
    _write_json(run_root / "summary.json", summary_payload)
    return summary_payload


def _case_paths(run_root: Path, size: str) -> tuple[Path, Path]:
    case_dir = run_root / _safe_size(size)
    return case_dir / "provider_e2e.json", case_dir / "provider_e2e.log"


def run_all(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "tmp").mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.parent.mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.write_text(str(run_root) + "\n", encoding="utf-8")

    sizes = [str(size) for size in args.sizes]
    bad_sizes = sorted(set(sizes) - set(SIZES))
    if bad_sizes:
        raise SystemExit(f"unknown sizes: {', '.join(bad_sizes)}")

    env = _apply_env_defaults(os.environ)
    env["TMPDIR"] = str(run_root / "tmp")
    env.setdefault("XDG_CACHE_HOME", str(run_root / "xdg-cache"))

    manifest = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "run_root": str(run_root),
        "doc": str(args.doc),
        "sizes": sizes,
        "policy": str(args.policy),
        "provider_mode": _provider_mode(str(args.policy)),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "env": {key: str(env.get(key, "")) for key in sorted(ENV_DEFAULTS)},
    }
    _write_json(run_root / "manifest.json", manifest)
    update_doc(Path(args.doc), run_root, sizes)

    active: subprocess.Popen[str] | None = None

    def _terminate(_signum: int, _frame: Any) -> None:
        if active is not None and active.poll() is None:
            active.terminate()
        raise KeyboardInterrupt

    old_int = signal.signal(signal.SIGINT, _terminate)
    old_term = signal.signal(signal.SIGTERM, _terminate)
    try:
        for size in sizes:
            provider_mode = _provider_mode(str(args.policy))
            out_path, log_path = _case_paths(run_root, size)
            if _status_ok(out_path) and not bool(args.force):
                print(f"[{datetime.now().isoformat(timespec='seconds')}] skip {size}: {out_path}", flush=True)
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(
                out_path,
                {
                    "status": "running",
                    "network": _network_name(size),
                    "mode": "provider",
                    "provider_mode": provider_mode,
                    "layout_policy": str(args.policy),
                    "activation": {"kind": "silu", "silu_degree": 7},
                    "input_shape": list(SIZES[size]),
                    "runner": {
                        "script": str(Path(__file__).relative_to(REPO_ROOT)),
                        "run_root": str(run_root),
                        "size": str(size),
                        "policy": str(args.policy),
                        "provider_mode": provider_mode,
                        "local_time": datetime.now().isoformat(timespec="seconds"),
                        "env": {key: str(env.get(key, "")) for key in sorted(ENV_DEFAULTS)},
                    },
                },
            )
            update_doc(Path(args.doc), run_root, sizes)
            command = [
                "/usr/bin/time",
                "-v",
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-one",
                "--size",
                str(size),
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
            if bool(args.profile_modules):
                command.append("--profile-modules")
            if bool(args.profile_lt):
                command.append("--profile-lt")
            if bool(args.trace_forward_memory):
                command.append("--trace-forward-memory")
            if bool(args.operator_breakdown):
                command.append("--operator-breakdown")
            print(f"[{datetime.now().isoformat(timespec='seconds')}] start {size}", flush=True)
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
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] done {size} rc={return_code}",
                flush=True,
            )
            update_doc(Path(args.doc), run_root, sizes)
            if int(return_code) != 0 and not bool(args.keep_going):
                return int(return_code)
        update_doc(Path(args.doc), run_root, sizes)
        return 0
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def _default_run_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUN_ROOT_BASE / f"u22_base32_silu7_streaming_provider_e2e_{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local U22 base32 SiLU7 full-network provider E2E with single-slot layer-cache LT."
    )
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--doc", type=Path, default=REPO_ROOT / "docs" / "u22_orion_streaming_haloed_mainline.md")
    parser.add_argument("--sizes", nargs="+", choices=tuple(SIZES), default=list(SIZES))
    parser.add_argument("--policy", choices=tuple(POLICY_PROVIDER_SUFFIX), default="dp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--profile-modules", action="store_true")
    parser.add_argument("--profile-lt", action="store_true")
    parser.add_argument("--trace-forward-memory", action="store_true")
    parser.add_argument("--operator-breakdown", dest="operator_breakdown", action="store_true")
    parser.add_argument("--no-operator-breakdown", dest="operator_breakdown", action="store_false")
    parser.set_defaults(operator_breakdown=True)
    parser.add_argument("--update-doc-only", action="store_true")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--size", choices=tuple(SIZES), default="192x192", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=Path("/tmp/u22_provider_e2e.json"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if bool(args.update_doc_only):
        update_doc(Path(args.doc), Path(args.run_root), [str(size) for size in args.sizes])
        return 0
    if bool(args.run_one):
        return run_one(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
