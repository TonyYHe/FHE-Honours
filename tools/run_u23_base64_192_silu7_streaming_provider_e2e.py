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

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.generate_unet22_compile_plan_csv import UNet22PlusOutput


DOC_MARKER = "U23_BASE64_192_SILU7_STREAMING_PROVIDER_E2E_TABLE"
NETWORK_NAME = "u23_192_base64_ibsr_full"
SIZE = "192x192"
INPUT_SHAPE = (1, 1, 192, 192)
OUT_CHANNELS = 4
BASE_DIM = 64
PROVIDER_MODE = "u22_256_base32"
DEFAULT_RUN_ROOT_BASE = REPO_ROOT / ".tmp" / "results"
LATEST_POINTER = REPO_ROOT / ".tmp" / "latest_u23_base64_192_silu7_streaming_provider_e2e.txt"

ENV_DEFAULTS: dict[str, str] = {
    "PYTHONUNBUFFERED": "1",
    "MALLOC_ARENA_MAX": "2",
    "ORION_COMPILE_PARALLEL_POLICY": "auto",
    "ORION_LATTIGO_STREAMING_LT": "force",
    "ORION_UNIFIED_STREAM_COMPILE_IO_NONE": "1",
    "ORION_LATTIGO_MEMORY_BOUNDED_COMPILE": "1",
    "ORION_LATTIGO_MEMORY_BOUNDED_EVAL": "1",
    "ORION_LATTIGO_BOOTSTRAP_MANY": "1",
    "ORION_UNIFIED_LT_OUTPUT_FUSION": "1",
    "ORION_LAYOUT_POLICY_RELAYOUT_KERNEL": "1",
    "ORION_LAYOUT_POLICY_PROVIDER_NATIVE_HALO": "1",
    "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS": "1",
    "ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE": "1",
    "ORION_UNIFIED_LT_FORCE_COMPILE_TRIM_EACH_TRANSFORM": "0",
    "ORION_UNIFIED_STREAM_COMPILE_BATCH_GB": "4",
    "ORION_REGION_FIRST_CLEANUP_AFTER_OUTPUTS": "1",
}

ENV_TUNING_KEYS: tuple[str, ...] = (
    "GOMAXPROCS",
    "ORION_COMPILE_MEMORY_RESERVE_GB",
    "ORION_LATTIGO_STREAMING_LT_MEMORY_FRACTION",
    "ORION_LATTIGO_STREAMING_LT_MEMORY_OVERHEAD",
    "ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS_MIN",
    "ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS_MAX",
    "ORION_LATTIGO_STREAMING_LT_SHARED_TRANSFORMS_MAX",
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
)

ENV_SNAPSHOT_KEYS: tuple[str, ...] = tuple(sorted(set(ENV_DEFAULTS) | set(ENV_TUNING_KEYS)))


def _build_u23(*, activation: str | None = None, silu_degree: int = 31) -> torch.nn.Module:
    return UNet22PlusOutput(
        in_channels=int(INPUT_SHAPE[1]),
        out_channels=int(OUT_CHANNELS),
        base_channels=int(BASE_DIM),
        activation=str(activation or "silu"),
        silu_degree=int(silu_degree),
    )


def _register_network() -> Any:
    from tools import run_lattigo_e2e_compare as base

    base.NETWORKS[NETWORK_NAME] = {
        "label": "U23 192x192 base64 IBSR full",
        "model": "UNet22PlusOutput",
        "dataset": "IBSR_BRAIN_2D",
        "input_shape": tuple(int(v) for v in INPUT_SHAPE),
        "provider_mode": PROVIDER_MODE,
        "config": base._r18_config,
        "builder": _build_u23,
        "scope": "full",
        "base_dim": int(BASE_DIM),
        "out_channels": int(OUT_CHANNELS),
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
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _annotate_result(
    out_path: Path,
    *,
    run_root: Path,
    started_at: float,
    env_snapshot: dict[str, str],
) -> None:
    payload = _read_json(out_path) or {}
    payload["runner"] = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "run_root": str(run_root),
        "size": SIZE,
        "local_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": float(time.perf_counter() - started_at),
        "maxrss_bytes": int(_resource_maxrss_bytes()),
        "env": {key: str(env_snapshot.get(key, "")) for key in ENV_SNAPSHOT_KEYS},
    }
    payload["model_variant"] = {
        "name": "UNet22PlusOutput",
        "linear_layers": 23,
        "body": "UNet22 with dec1b 64->64 3x3/pad1",
        "output": f"1x1/pad0 {BASE_DIM}->{OUT_CHANNELS}",
        "input_shape": list(INPUT_SHAPE),
        "dataset_label": "IBSR BRAIN 2D",
    }
    _write_json(out_path, payload)


def run_one(args: argparse.Namespace) -> int:
    os.environ.update(_apply_env_defaults(os.environ))
    env_snapshot = dict(os.environ)
    base = _register_network()
    started_at = time.perf_counter()
    out_path = Path(args.out)
    try:
        base._run_one(
            network=NETWORK_NAME,
            backend="lattigo",
            mode="provider",
            out_path=out_path,
            seed=int(args.seed),
            compile_only=False,
            forward_runs=int(args.forward_runs),
            warmup_runs=int(args.warmup_runs),
            profile_modules=False,
            profile_lt=bool(args.profile_lt),
            trace_forward_memory=bool(args.trace_forward_memory),
            operator_breakdown=bool(args.operator_breakdown),
            provider_mode_override=PROVIDER_MODE,
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
    totals = _metric(payload, ("operator_breakdown_after_forward", "totals"))
    if not isinstance(totals, dict):
        return None
    value = totals.get(str(key))
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _compile_breakdown(payload: dict[str, Any], key: str) -> float | int | None:
    totals = _metric(payload, ("operator_breakdown_after_compile", "totals"))
    if not isinstance(totals, dict):
        return None
    value = totals.get(str(key))
    if value is None:
        return None
    try:
        if str(key).endswith("_count") or str(key) in {"payload_bytes"}:
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return None


def _runtime_mode(payload: dict[str, Any]) -> str:
    return str(
        payload.get("runtime_fairness_mode")
        or _metric(payload, ("measured_runtime_fairness_timing", "runtime_fairness_mode"))
        or _metric(payload, ("runtime_fairness_timing_after_forward", "runtime_fairness_mode"))
        or ""
    )


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


def _case_row(run_root: Path) -> tuple[list[str], dict[str, Any]]:
    result_path = run_root / "provider_e2e.json"
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
    unattributed_s = _breakdown_timing(payload or {}, "unattributed_he_forward_s")
    compile_s = timing.get("compile") if isinstance(timing, dict) else None
    row = [
        SIZE,
        "IBSR BRAIN 2D",
        "1 -> 4",
        status,
        _layout_policy(payload or {}),
        "yes" if (payload or {}).get("bootstrap_many_enabled") is True else "",
        "yes" if (payload or {}).get("operator_breakdown") is True else "",
        "no" if (payload or {}).get("profile_modules") is False else str((payload or {}).get("profile_modules", "")),
        _format_float(compile_s),
        _format_float(_compile_breakdown(payload or {}, "group_total_s")),
        _format_int(_compile_breakdown(payload or {}, "transform_count")),
        _gib(_compile_breakdown(payload or {}, "payload_bytes")),
        _format_float(encrypt_s),
        _format_float(he_forward_s),
        _format_float(decode_s),
        _format_float(hot_s),
        _format_float(mvm_s),
        _format_float(act_s),
        _format_float(boot_s),
        _format_float(load_encode_s),
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
        "size": SIZE,
        "status": status,
        "result_path": str(result_path),
        "compile_s": compile_s,
        "compile_group_total_s": _compile_breakdown(payload or {}, "group_total_s"),
        "compile_transform_count": _compile_breakdown(payload or {}, "transform_count"),
        "compile_payload_bytes": _compile_breakdown(payload or {}, "payload_bytes"),
        "encrypt_s": encrypt_s,
        "he_forward_s": he_forward_s,
        "decrypt_decode_s": decode_s,
        "hot_s": hot_s,
        "mvm_kernel_s": mvm_s,
        "activation_s": act_s,
        "bootstrap_s": boot_s,
        "lt_runtime_load_encode_s": load_encode_s,
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
        "dataset",
        "I/O ch",
        "status",
        "planner",
        "bootmany",
        "op breakdown",
        "broad modules",
        "compile/load s",
        "compile LT group s",
        "compile transforms",
        "compile payload GiB",
        "encrypt s",
        "HE forward s",
        "decode s",
        "hot E2E s",
        "MVM kernel s",
        "ACT s",
        "boot s",
        "LT load/enc s",
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


def update_doc(doc_path: Path, run_root: Path) -> dict[str, Any]:
    row, summary = _case_row(run_root)
    table = _markdown_table([row])
    text = doc_path.read_text(encoding="utf-8")
    doc_path.write_text(_replace_block(text, DOC_MARKER, table), encoding="utf-8")
    summary_payload = {
        "run_root": str(run_root),
        "doc": str(doc_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "case": summary,
    }
    _write_json(run_root / "summary.json", summary_payload)
    return summary_payload


def run_all(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "tmp").mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.parent.mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.write_text(str(run_root) + "\n", encoding="utf-8")

    env = _apply_env_defaults(os.environ)
    env["TMPDIR"] = str(run_root / "tmp")
    env.setdefault("XDG_CACHE_HOME", str(run_root / "xdg-cache"))

    manifest = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "run_root": str(run_root),
        "doc": str(args.doc),
        "network": NETWORK_NAME,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "env": {key: str(env.get(key, "")) for key in ENV_SNAPSHOT_KEYS},
        "profile_modules": False,
        "operator_breakdown": bool(args.operator_breakdown),
        "model_variant": {
            "linear_layers": 23,
            "input_shape": list(INPUT_SHAPE),
            "base_dim": int(BASE_DIM),
            "out_channels": int(OUT_CHANNELS),
            "activation": "SiLU7",
            "output_head": "1x1/pad0",
        },
    }
    _write_json(run_root / "manifest.json", manifest)
    update_doc(Path(args.doc), run_root)

    out_path = run_root / "provider_e2e.json"
    log_path = run_root / "provider_e2e.log"
    if _status_ok(out_path) and not bool(args.force):
        print(f"[{datetime.now().isoformat(timespec='seconds')}] skip {SIZE}: {out_path}", flush=True)
        update_doc(Path(args.doc), run_root)
        return 0

    _write_json(
        out_path,
        {
            "status": "running",
            "network": NETWORK_NAME,
            "mode": "provider",
            "provider_mode": PROVIDER_MODE,
            "activation": {"kind": "silu", "silu_degree": 7},
            "input_shape": list(INPUT_SHAPE),
            "runner": {
                "script": str(Path(__file__).relative_to(REPO_ROOT)),
                "run_root": str(run_root),
                "size": SIZE,
                "local_time": datetime.now().isoformat(timespec="seconds"),
                "env": {key: str(env.get(key, "")) for key in ENV_SNAPSHOT_KEYS},
            },
        },
    )
    update_doc(Path(args.doc), run_root)

    command = [
        "/usr/bin/time",
        "-v",
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one",
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

    active: subprocess.Popen[str] | None = None

    def _terminate(_signum: int, _frame: Any) -> None:
        if active is not None and active.poll() is None:
            active.terminate()
        raise KeyboardInterrupt

    old_int = signal.signal(signal.SIGINT, _terminate)
    old_term = signal.signal(signal.SIGTERM, _terminate)
    try:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] start {SIZE}", flush=True)
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
        print(f"[{datetime.now().isoformat(timespec='seconds')}] done {SIZE} rc={return_code}", flush=True)
        update_doc(Path(args.doc), run_root)
        return int(return_code)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def _default_run_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUN_ROOT_BASE / f"u23_base64_192_silu7_streaming_provider_e2e_{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run U23 192x192 base64 SiLU7 provider E2E with streaming LT."
    )
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--doc", type=Path, default=REPO_ROOT / "docs" / "u22_orion_streaming_haloed_mainline.md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--profile-lt", action="store_true")
    parser.add_argument("--trace-forward-memory", action="store_true")
    parser.add_argument("--operator-breakdown", dest="operator_breakdown", action="store_true")
    parser.add_argument("--no-operator-breakdown", dest="operator_breakdown", action="store_false")
    parser.set_defaults(operator_breakdown=True)
    parser.add_argument("--update-doc-only", action="store_true")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=Path("/tmp/u23_provider_e2e.json"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if bool(args.update_doc_only):
        update_doc(Path(args.doc), Path(args.run_root))
        return 0
    if bool(args.run_one):
        return run_one(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
