#!/usr/bin/env python3
"""Compare full and compressed CIPS transforms in fresh worker processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.experimental.wpc_cips_benchmark import compare_worker_payloads


DEFAULT_OUT_DIR = (
    REPO_ROOT / ".tmp/results/honours/15_wpc_isolated_resource_benchmark"
)
WORKER = REPO_ROOT / "tools/run_wpc_cips_isolated_worker.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--forward-runs", type=int, default=10)
    parser.add_argument("--rss-sample-ms", type=float, default=5.0)
    parser.add_argument("--atol", type=float, default=2e-6)
    return parser


def _portable_path(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _portable_command(command: list[str]) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(command):
        if index == 0:
            values.append("python")
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            portable = _portable_path(candidate)
            values.append(portable)
        else:
            values.append(value)
    return values


def _read_phase(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["phase"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError):
        return "process_start"


def _read_process_memory(pid: int) -> tuple[dict[str, int] | None, str]:
    path = Path(f"/proc/{int(pid)}/status")
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        rows = []
    values: dict[str, int] = {}
    for line in rows:
        if line.startswith("VmRSS:"):
            values["rss_bytes"] = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            values["hwm_bytes"] = int(line.split()[1]) * 1024
    if "rss_bytes" in values:
        return values, "linux_proc_pid_status"
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(int(pid))],
                text=True,
            ).strip()
            if output:
                return {"rss_bytes": int(output) * 1024}, "macos_ps_rss"
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return None, "unavailable"


def _sample_worker(
    *,
    mode: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    result_path = out_dir / f"{mode}.worker.json"
    log_path = out_dir / f"{mode}.worker.log"
    phase_path = out_dir / f"{mode}.phase.json"
    phase_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(WORKER),
        "--mode",
        str(mode),
        "--out",
        str(result_path),
        "--phase-file",
        str(phase_path),
        "--seed",
        str(int(args.seed)),
        "--logn",
        str(int(args.logn)),
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--channels",
        str(int(args.channels)),
        "--bsgs-ratio",
        str(float(args.bsgs_ratio)),
        "--warmup-runs",
        str(int(args.warmup_runs)),
        "--forward-runs",
        str(int(args.forward_runs)),
        "--atol",
        str(float(args.atol)),
    ]
    peak_rss: dict[str, int] = {}
    peak_hwm: dict[str, int] = {}
    sample_count: dict[str, int] = {}
    total_samples = 0
    sampling_source = "unavailable"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
        )
        while process.poll() is None:
            phase = _read_phase(phase_path)
            sample, source = _read_process_memory(process.pid)
            if sample is not None:
                sampling_source = source
                total_samples += 1
                sample_count[phase] = int(sample_count.get(phase, 0) + 1)
                peak_rss[phase] = max(
                    int(peak_rss.get(phase, 0)), int(sample["rss_bytes"])
                )
                if "hwm_bytes" in sample:
                    peak_hwm[phase] = max(
                        int(peak_hwm.get(phase, 0)), int(sample["hwm_bytes"])
                    )
            time.sleep(float(args.rss_sample_ms) / 1000.0)
        return_code = int(process.wait())

    if return_code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"{mode} worker exited with {return_code}\n" + "\n".join(tail)
        )
    if not result_path.is_file():
        raise RuntimeError(f"{mode} worker did not produce {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    sampling = {
        "source": sampling_source,
        "interval_ms": float(args.rss_sample_ms),
        "total_sample_count": int(total_samples),
        "sample_count_by_phase": sample_count,
        "peak_rss_by_phase": peak_rss,
        "peak_hwm_by_phase": peak_hwm,
    }
    return payload, sampling, command


def _without_output_values(payload: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(payload))
    clone.get("correctness", {}).pop("output_values", None)
    return clone


def _mib(value: int | float | None) -> str:
    return "n/a" if value is None else f"{float(value) / (1024**2):.2f}"


def _ms(value: int | float | None) -> str:
    return "n/a" if value is None else f"{1000.0 * float(value):.3f}"


def _render_report(payload: dict[str, Any]) -> str:
    comparison = payload["comparison"]
    latency = comparison["latency"]
    memory = comparison["memory"]
    correctness = comparison["correctness"]
    full = payload["workers"]["full"]
    compressed = payload["workers"]["compressed"]
    return f"""# Isolated WPC CIPS resource benchmark

## Result

Status: **{payload['status']}**. The ordinary full-Q/P and WPC-compressed paths
ran in distinct fresh processes with identical parameters, weights, inputs,
warmups, and measured-forward counts.

| Metric | Full Q/P | WPC compressed |
|---|---:|---:|
| Compile time (s) | {full['compile_s']:.6f} | {compressed['compile_s']:.6f} |
| Median two-layer forward (ms) | {_ms(latency['full_forward_s']['median'])} | {_ms(latency['compressed_forward_s']['median'])} |
| p95 two-layer forward (ms) | {_ms(latency['full_forward_s']['p95'])} | {_ms(latency['compressed_forward_s']['p95'])} |
| Logical resident plaintext storage (MiB) | {_mib(memory['logical_full_storage_bytes'])} | {_mib(memory['logical_compressed_storage_bytes'])} |
| Pre-online RSS (MiB) | {_mib(memory['full_pre_online_rss_bytes'])} | {_mib(memory['compressed_pre_online_rss_bytes'])} |
| Measured-phase peak RSS (MiB) | {_mib(memory['full_online_peak_rss_bytes'])} | {_mib(memory['compressed_online_peak_rss_bytes'])} |
| Go heap-in-use after compile GC (MiB) | {_mib(memory['full_go_heap_inuse_after_compile_gc_bytes'])} | {_mib(memory['compressed_go_heap_inuse_after_compile_gc_bytes'])} |
| Maximum error versus clear | {correctness['full_max_abs_error_vs_clear']:.3e} | {correctness['compressed_max_abs_error_vs_clear']:.3e} |

The logical plaintext-storage ratio is
**{memory['logical_storage_compression_ratio']:.3f}x**, and the compressed/full
median-forward ratio is
**{latency['compressed_over_full_median_ratio']:.3f}x**. Median online
decompression takes
**{_ms(latency['compressed_decompression_s']['median'])} ms**, or
**{latency['compressed_decompression_pct_of_forward']['median']:.3f}%** of the
compressed forward.

The largest difference between independently encrypted full and compressed
outputs is `{correctness['max_abs_delta_between_isolated_outputs']:.3e}`.
Both paths perform the same recorded homomorphic operations and no online
weight Encode.

## Measurement boundary

- Forward timing includes two chained `Conv2d` layers.
- Compilation, input encryption, output decryption, and cleanup are excluded.
- An explicit Go garbage collection and unused-page release occurs between
  phases, outside timed forwards, to remove dead compilation allocations.
- RSS is sampled externally by the parent process; Go heap and logical Q/P
  storage are reported separately.
- RSS includes the Python runtime, Torch, keys, ciphertexts, and allocator
  pages. The logical storage ratio therefore must not be presented as an RSS
  ratio.
- This is a deterministic two-layer microbenchmark, not a trained-model
  latency or accuracy result.

All acceptance gates passed: **{comparison['acceptance']['valid']}**.
"""


def main() -> int:
    args = _parser().parse_args()
    if float(args.rss_sample_ms) <= 0:
        raise SystemExit("rss-sample-ms must be positive")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    workers: dict[str, dict[str, Any]] = {}
    sampling: dict[str, dict[str, Any]] = {}
    commands: dict[str, list[str]] = {}
    started = time.perf_counter()
    for mode in ("full", "compressed"):
        worker, rss, command = _sample_worker(
            mode=mode,
            out_dir=out_dir,
            args=args,
        )
        workers[mode] = worker
        sampling[mode] = rss
        commands[mode] = _portable_command(command)

    comparison = compare_worker_payloads(
        workers["full"],
        workers["compressed"],
        rss_samples=sampling,
        atol=float(args.atol),
    )
    valid = bool(comparison["acceptance"]["valid"])
    payload = {
        "schema_version": 1,
        "profile": "wpc_cips_isolated_full_vs_compressed_resource_benchmark",
        "status": "ok" if valid else "invalid",
        "timing_policy": (
            "full and compressed paths run in distinct fresh processes; warmups "
            "excluded; repeated forward wall times include two Conv2d layers and "
            "exclude compile, input encryption, decryption, and cleanup"
        ),
        "elapsed_s": float(time.perf_counter() - started),
        "worker_order": ["full", "compressed"],
        "commands": commands,
        "artifacts": {
            mode: {
                "result": _portable_path(out_dir / f"{mode}.worker.json"),
                "log": _portable_path(out_dir / f"{mode}.worker.log"),
                "phase": _portable_path(out_dir / f"{mode}.phase.json"),
            }
            for mode in ("full", "compressed")
        },
        "workers": {
            mode: _without_output_values(worker)
            for mode, worker in workers.items()
        },
        "comparison": comparison,
        "acceptance": comparison["acceptance"],
    }
    result_path = out_dir / "comparison.json"
    report_path = out_dir / "comparison.md"
    payload["artifacts"]["comparison"] = _portable_path(result_path)
    payload["artifacts"]["report"] = _portable_path(report_path)
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "result": str(result_path),
        "latency": comparison["latency"],
        "memory": {
            key: value
            for key, value in comparison["memory"].items()
            if key not in ("rss_sampling", "interpretation")
        },
        "correctness": comparison["correctness"],
        "operations": comparison["operations"],
        "acceptance": comparison["acceptance"],
    }, indent=2, sort_keys=True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
