#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "haloed_paper_eval"
DEFAULT_RESULT_ROOT = REPO_ROOT / ".tmp" / "results" / "haloed_paper_eval"

CPP_LIBRARY_BUILDERS: tuple[dict[str, str], ...] = (
    {"name": "diag_builder", "script": "tools/build_diag_builder.py"},
    {"name": "clear_lattigo", "script": "tools/build_clear_lattigo.py"},
    {"name": "cheddar", "script": "tools/build_cheddar_backend.py", "required_env": "CHEDDAR_ROOT"},
)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--check-existing", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")


def resolve_run_root(args: argparse.Namespace, artifact_name: str) -> Path:
    if getattr(args, "check_existing", None) is not None:
        return Path(args.check_existing).resolve()
    if getattr(args, "run_root", None) is not None:
        return Path(args.run_root).resolve()
    return (DEFAULT_RESULT_ROOT / f"{artifact_name}_{timestamp()}").resolve()


def ensure_layout(run_root: Path) -> dict[str, Path]:
    dirs = {
        "root": Path(run_root),
        "raw": Path(run_root) / "raw",
        "paper": Path(run_root) / "paper",
        "logs": Path(run_root) / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def command_text(command: Iterable[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command: list[str | Path], *, log_path: Path, dry_run: bool = False) -> int:
    print(command_text(command), flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return int(proc.wait())


def rebuild_local_cpp_libraries(*, log_dir: Path, dry_run: bool = False) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for builder in CPP_LIBRARY_BUILDERS:
        name = str(builder["name"])
        required_env = str(builder.get("required_env", ""))
        if required_env and not os.environ.get(required_env, "").strip():
            records.append(
                {
                    "name": name,
                    "status": "skipped",
                    "reason": f"{required_env} is not set",
                    "log": "",
                }
            )
            continue
        command: list[str | Path] = [sys.executable, REPO_ROOT / str(builder["script"])]
        log_path = Path(log_dir) / f"build_cpp_{name}.log"
        rc = run_command(command, log_path=log_path, dry_run=bool(dry_run))
        status = "dry_run" if bool(dry_run) else "ok" if int(rc) == 0 else "failed"
        record = {
            "name": name,
            "status": status,
            "command": command_text(command),
            "log": str(log_path),
        }
        records.append(record)
        if int(rc) != 0:
            raise RuntimeError(f"C++ library rebuild failed for {name}; see {log_path}")
    return records


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(str(key))
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_number(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    if not text:
        return float(default)
    return float(text)


def clean_int(value: Any, default: int = 0) -> int:
    return int(round(clean_number(value, float(default))))


def tex_int(value: Any) -> str:
    return f"{clean_int(value):,}".replace(",", "{,}")


def tex_time(value: Any) -> str:
    return tex_int(round(clean_number(value)))


def tex_speedup(value: Any) -> str:
    return f"{clean_number(value):.2f}$\\times$"


def git_info() -> dict[str, Any]:
    def _run(*parts: str) -> str:
        try:
            return subprocess.check_output(parts, cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "dirty": bool(_run("git", "status", "--short")),
    }


def write_manifest(
    run_root: Path,
    *,
    artifact: str,
    source_script: str,
    command: list[str | Path] | None,
    outputs: dict[str, str | list[str]],
    measurement: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "artifact": artifact,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "repo_root": str(REPO_ROOT),
        "source_script": source_script,
        "command": command_text(command or []),
        "measurement": measurement,
        "outputs": outputs,
        "git": git_info(),
        "env": {
            key: os.environ.get(key, "")
            for key in (
                "ORION_SINGLE_SLOT_LAYER_CACHE",
                "ORION_LATTIGO_BOOTSTRAP_MANY",
                "ORION_UNIFIED_LT_INDIVIDUAL_EVAL",
                "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS",
                "ORION_LATTIGO_UNIFIED_NO_BSGS",
                "ORION_CONCAT_FUSION",
                "ORION_PROVIDER_MVM_MASKED_MATERIALIZATION",
                "ORION_BOOTSTRAP_LAYOUT_REFINEMENT",
                "ORION_DISABLE_BOOTSTRAP_PRESCALE_FUSION",
                "ORION_STRICT_INACTIVE_SLOT_ZERO",
                "ORION_NATIVE_MVM_ZERO_INACTIVE_SLOTS",
                "ORION_LATTIGO_CLEAR_BACKEND",
                "ORION_SINGLE_SLOT_ENCODE_WORKERS",
                "ORION_PACK_CONV_WORKERS",
                "ORION_DIRECT_PACK_WORKERS",
                "ORION_LT_COMPILE_WORKERS",
                "ORION_UNIFIED_COMPILE_WORKERS",
                "ORION_LATTIGO_COMPILE_WORKERS",
                "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS",
                "ORION_LATTIGO_BOOTSTRAP_WORKERS",
            )
        },
    }
    if extra:
        payload.update(extra)
    write_json(Path(run_root) / "manifest.json", payload)


def copy_doc_snapshot(target: Path) -> Path:
    source = REPO_ROOT / "docs" / "u22_orion_streaming_haloed_mainline.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def find_first(root: Path, names: Iterable[str]) -> Path | None:
    wanted = set(names)
    for path in Path(root).rglob("*"):
        if path.name in wanted:
            return path
    return None


def maybe_existing_artifact_root(root: Path, artifact_prefix: str) -> Path:
    root = Path(root).resolve()
    if root.name.startswith(artifact_prefix):
        return root
    matches = sorted(path for path in root.glob(f"{artifact_prefix}_*") if path.is_dir())
    if not matches:
        return root
    return matches[-1]


def print_outputs(outputs: dict[str, str | list[str]]) -> None:
    for key, value in outputs.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  {item}")
        else:
            print(f"{key}: {value}")
