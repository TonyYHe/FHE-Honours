#!/usr/bin/env python3
"""Benchmark the trained WPC decoder with isolated full/compressed workers."""

from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path
import subprocess
import signal
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.experimental.wpc_cips_trained_benchmark import (
    compare_trained_decoder_workers,
)
from tools.run_wpc_cips_isolated_benchmark import (
    _portable_command,
    _portable_path,
    _read_phase,
    _read_process_memory,
)
from tools.run_wpc_cips_trained_decoder import DEFAULT_CHECKPOINT
from orion.experimental.wpc_decoder_geometry import decoder_geometry, resolve_decoder_config


DEFAULT_OUT_DIR = (
    REPO_ROOT
    / ".tmp/results/honours/25_wpc_finetuned_decoder_isolated_benchmark"
)
WORKER = REPO_ROOT / "tools/run_wpc_cips_trained_isolated_worker.py"
WORKER_EXIT_CONFIRMATION_S = 0.1
WORKER_EXITING_REAP_S = 5.0
LINUX_PF_EXITING = 0x00000004


def _probe_linux_worker_shutdown(
    pid: int, *, proc_root: Path | None = None,
) -> tuple[dict[str, int] | None, dict[str, Any] | None]:
    """Recover shared RSS or prove all observed threads have entered kernel exit.

    A dead leader alone does not prove a multithreaded worker has exited. Linux
    can clear task->mm before the task becomes reapable (kernel/exit.c), while
    PF_EXITING is exposed in field 9 of each thread's stat (fs/proc/array.c).
    Explicit proc_root supports deterministic fixtures without a Linux host.
    """
    if proc_root is None:
        if sys.platform != "linux":
            return None, None
        proc_root = Path("/proc")
    task_root = proc_root / str(int(pid)) / "task"
    probe: dict[str, Any] = {"source": "linux_proc_pid_task_stat", "threads": [],
                             "errors": [], "all_threads_exiting": False}
    try:
        initial = {int(path.name) for path in task_root.iterdir() if path.name.isdecimal()}
    except OSError as error:
        probe["errors"].append(type(error).__name__)
        return None, probe
    disappeared: set[int] = set()
    samples = []
    for tid in sorted(initial):
        directory = task_root / str(tid)
        try:
            text = (directory / "stat").read_text(encoding="utf-8")
            # comm can contain spaces and parentheses; never split the whole line.
            fields = text.rsplit(")", 1)[1].split()
            if int(text.split(" ", 1)[0]) != tid or fields[0] not in set("RSDTtZXxKWPI"):
                raise ValueError("invalid thread identity/state")
            state, flags = fields[0], int(fields[6])
            if not 0 <= flags <= 0xFFFFFFFF:
                raise ValueError("invalid thread flags")
        except FileNotFoundError:
            disappeared.add(tid)
            continue
        except (OSError, ValueError, IndexError) as error:
            probe["errors"].append(f"tid={tid} stat: {type(error).__name__}")
            continue
        probe["threads"].append({"tid": tid, "state": state, "flags": flags,
            "exiting": bool(flags & LINUX_PF_EXITING) or state in ("Z", "X", "x")})
        try:
            values = {}
            for line in (directory / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    name, value, unit = line.split()
                    if unit != "kB" or int(value) < 0:
                        raise ValueError("invalid memory observation")
                    values["rss_bytes" if name == "VmRSS:" else "hwm_bytes"] = int(value) * 1024
            if "rss_bytes" in values:
                samples.append(values)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as error:
            probe["errors"].append(f"tid={tid} status: {type(error).__name__}")
    try:
        remaining = {int(path.name) for path in task_root.iterdir() if path.name.isdecimal()}
    except FileNotFoundError:
        remaining = set()
    except OSError as error:
        probe["errors"].append(type(error).__name__)
        remaining = initial
    probe["new_thread_ids"] = sorted(remaining - initial)
    probe["unreadable_remaining_thread_ids"] = sorted(disappeared & remaining)
    probe["all_threads_exiting"] = bool(probe["threads"]) and not (
        probe["errors"] or probe["new_thread_ids"] or probe["unreadable_remaining_thread_ids"]
    ) and all(row["exiting"] for row in probe["threads"])
    if not samples:
        return None, probe
    # Thread RSS is for the shared address space: take a maximum, never a sum.
    sample = {"rss_bytes": max(row["rss_bytes"] for row in samples)}
    hwm = [row["hwm_bytes"] for row in samples if "hwm_bytes" in row]
    if hwm:
        sample["hwm_bytes"] = max(hwm)
    return sample, probe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--logn", type=int, default=None)
    parser.add_argument("--ckks-config", type=Path)
    parser.add_argument("--verify-exact-qp", action="store_true")
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--feature-std", type=float, default=0.02)
    parser.add_argument("--bound-headroom", type=float, default=1.25)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--forward-runs", type=int, default=10)
    parser.add_argument("--rss-sample-ms", type=float, default=5.0)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--max-worker-rss-mib", type=float, default=0, help="sampled RSS guard; 0 disables (not an OS allocation limit)")
    parser.add_argument("--worker-timeout-s", type=float, default=0, help="wall-clock guard per worker; 0 disables")
    return parser


def _stop_worker(process: subprocess.Popen) -> None:
    """Stop only the new session created for this worker, including descendants."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _sample_worker(
    *,
    mode: str,
    checkpoint: Path,
    out_dir: Path,
    args: argparse.Namespace,
    worker_script: Path | None = None,
    worker_args: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    result_path = out_dir / f"{mode}.worker.json"
    log_path = out_dir / f"{mode}.worker.log"
    phase_path = out_dir / f"{mode}.phase.json"
    phase_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(WORKER if worker_script is None else worker_script),
        "--mode",
        str(mode),
        "--checkpoint",
        str(checkpoint),
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
        "--bsgs-ratio",
        str(float(args.bsgs_ratio)),
        "--feature-std",
        str(float(args.feature_std)),
        "--bound-headroom",
        str(float(args.bound_headroom)),
        "--warmup-runs",
        str(int(args.warmup_runs)),
        "--forward-runs",
        str(int(args.forward_runs)),
        "--atol",
        str(float(args.atol)),
    ]
    if getattr(args, "ckks_config", None) is not None:
        command += ["--ckks-config", str(args.ckks_config.expanduser().resolve())]
    if getattr(args, "verify_exact_qp", False):
        command += ["--verify-exact-qp"]
    command += list(worker_args or ())
    rss_limit = float(getattr(args, "max_worker_rss_mib", 0))
    timeout = float(getattr(args, "worker_timeout_s", 0))
    if any(not math.isfinite(value) or value < 0 for value in (rss_limit, timeout)):
        raise ValueError("worker resource guards must be finite and nonnegative")
    if rss_limit and _read_process_memory(os.getpid())[0] is None:
        raise RuntimeError("RSS watchdog unavailable on this host; refusing to launch an unguarded worker")
    peak_rss: dict[str, int] = {}
    peak_hwm: dict[str, int] = {}
    sample_count: dict[str, int] = {}
    total_samples = 0
    unavailable_samples = 0
    shutdown_probe_count = 0
    last_shutdown_probe = None
    sample_count_by_source: dict[str, int] = {}
    exit_confirmed_after_unavailable_sample = False
    sampling_source = "unavailable"
    started = time.monotonic()
    termination_reason = None
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                phase = _read_phase(phase_path)
                sample, source = _read_process_memory(process.pid)
                shutdown_probe = None
                if sample is None and rss_limit:
                    sample, shutdown_probe = _probe_linux_worker_shutdown(process.pid)
                    if shutdown_probe is not None:
                        shutdown_probe_count += 1
                        last_shutdown_probe = dict(shutdown_probe, phase=phase)
                    if sample is not None:
                        source = "linux_proc_pid_task_status"
                if sample is not None:
                    sampling_source = source
                    total_samples += 1
                    sample_count_by_source[source] = sample_count_by_source.get(source, 0) + 1
                    sample_count[phase] = int(sample_count.get(phase, 0) + 1)
                    peak_rss[phase] = max(
                        int(peak_rss.get(phase, 0)), int(sample["rss_bytes"]),
                    )
                    if "hwm_bytes" in sample:
                        peak_hwm[phase] = max(int(peak_hwm.get(phase, 0)), int(sample["hwm_bytes"]))
                    if rss_limit and sample["rss_bytes"] > rss_limit * 2**20:
                        termination_reason = f"sampled RSS exceeded {rss_limit:g} MiB in phase {phase}"
                else:
                    unavailable_samples += 1
                    if rss_limit:
                        if process.poll() is not None:
                            exit_confirmed_after_unavailable_sample = True
                            break
                        exiting = bool(shutdown_probe and shutdown_probe["all_threads_exiting"])
                        exit_wait = WORKER_EXITING_REAP_S if exiting else WORKER_EXIT_CONFIRMATION_S
                        if timeout:
                            remaining = started + timeout - time.monotonic()
                            exit_wait = min(exit_wait, max(0.0, remaining))
                        try:
                            process.wait(timeout=exit_wait)
                        except subprocess.TimeoutExpired:
                            if timeout and time.monotonic() - started >= timeout:
                                termination_reason = f"worker exceeded {timeout:g} seconds in phase {phase}"
                            elif exiting:
                                termination_reason = "worker did not finish kernel-confirmed shutdown within the reap timeout"
                            else:
                                termination_reason = "RSS watchdog lost access to the live worker; refusing an unguarded run"
                        else:
                            exit_confirmed_after_unavailable_sample = True
                            break
                if timeout and time.monotonic() - started > timeout:
                    termination_reason = f"worker exceeded {timeout:g} seconds in phase {phase}"
                if termination_reason:
                    _stop_worker(process)
                    break
                time.sleep(float(args.rss_sample_ms) / 1000.0)
        finally:
            _stop_worker(process)
        return_code = int(process.wait())

    sampling = {
        "source": sampling_source, "interval_ms": float(args.rss_sample_ms),
        "total_sample_count": int(total_samples), "sample_count_by_phase": sample_count,
        "unavailable_sample_count": int(unavailable_samples),
        "exit_confirmed_after_unavailable_sample": exit_confirmed_after_unavailable_sample,
        "sample_count_by_source": sample_count_by_source,
        "shutdown_probe_count": shutdown_probe_count, "last_shutdown_probe": last_shutdown_probe,
        "peak_rss_by_phase": peak_rss, "peak_hwm_by_phase": peak_hwm,
        "return_code": return_code, "termination_reason": termination_reason,
        "elapsed_s": time.monotonic() - started,
        "guards": {"max_worker_rss_mib": rss_limit, "worker_timeout_s": timeout,
                   "exit_confirmation_timeout_s": WORKER_EXIT_CONFIRMATION_S,
                   "kernel_exiting_reap_timeout_s": WORKER_EXITING_REAP_S,
                   "policy": "sampled RSS watchdog, not an OS-enforced allocation ceiling"},
    }
    # Preserve failure evidence even when no worker JSON was produced.
    (out_dir / f"{mode}.rss.json").write_text(json.dumps(sampling, indent=2, allow_nan=False) + "\n")
    if return_code != 0 or termination_reason:
        tail = log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-100:]
        raise RuntimeError(
            f"{mode} worker exited with {return_code}; {termination_reason or 'see worker log'}\n" + "\n".join(tail)
        )
    if not result_path.is_file():
        raise RuntimeError(f"{mode} worker did not produce {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return payload, sampling, command


def _without_output_values(payload: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(payload))
    clone.get("correctness", {}).pop("output_values", None)
    clone.get("correctness", {}).pop("independent_clear_output_values", None)
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
    return f"""# Isolated fine-tuned WPC decoder benchmark

## Result

Status: **{payload['status']}**. Full-Q/P and compressed-Q/P workers used the
same checkpoint, graph, synthetic features, CKKS parameters, warmups, and
measured-forward count in distinct fresh processes.

| Metric | Full Q/P | WPC compressed |
|---|---:|---:|
| Median forward (ms) | {_ms(latency['full_forward_s']['median'])} | {_ms(latency['compressed_forward_s']['median'])} |
| Mean forward (ms) | {_ms(latency['full_forward_s']['mean'])} | {_ms(latency['compressed_forward_s']['mean'])} |
| p95 forward (ms) | {_ms(latency['full_forward_s']['p95'])} | {_ms(latency['compressed_forward_s']['p95'])} |
| Logical resident plaintexts (MiB) | {_mib(memory['logical_full_resident_bytes'])} | {_mib(memory['logical_compressed_resident_bytes'])} |
| Pre-online RSS (MiB) | {_mib(memory['full_pre_online_rss_bytes'])} | {_mib(memory['compressed_pre_online_rss_bytes'])} |
| Measured peak RSS (MiB) | {_mib(memory['full_online_peak_rss_bytes'])} | {_mib(memory['compressed_online_peak_rss_bytes'])} |
| Maximum error vs clear | {correctness['full_max_abs_error_vs_clear']:.6g} | {correctness['compressed_max_abs_error_vs_clear']:.6g} |

## Derived results

- Compressed/full median latency ratio: `{latency['compressed_over_full_median_ratio']:.6g}`.
- Median online decompression: `{_ms(latency['compressed_decompression_s']['median'])} ms`
  (`{latency['compressed_decompression_pct_of_forward']['median']:.6g}%` of
  compressed forward wall time).
- Logical resident-storage compression: `{memory['logical_storage_compression_ratio']:.6g}x`.
- Maximum isolated output delta: `{correctness['max_abs_delta_between_isolated_outputs']:.6g}`.
- Per-forward operation counters match: `{comparison['operations']['match']}`.
- Online Python Encode calls: full
  `{full['measurements']['online_python_encode_call_count']}`, compressed
  `{compressed['measurements']['online_python_encode_call_count']}`.

## Scope

The measured graph is `up1 + skip1 -> cat1 -> dec1a -> trained Cheb7 -> real
bootstrap -> dec1b` with synthetic internal features. It is a trained decoder-
stage resource result, not whole-network U-Net latency or dataset accuracy.
RSS includes runtimes, keys, ciphertexts, bootstrap state, and allocator pages;
logical plaintext storage is reported separately.

All acceptance gates passed: **{comparison['acceptance']['valid']}**.
"""


def main() -> int:
    args = _parser().parse_args()
    config, _source = resolve_decoder_config(args)
    decoder_geometry(args.logn, args.height, args.width, len(config["ckks_params"]["LogQ"]) - 1)
    if float(args.rss_sample_ms) <= 0.0:
        raise SystemExit("rss-sample-ms must be positive")
    if int(args.warmup_runs) < 0 or int(args.forward_runs) <= 0:
        raise SystemExit("warmup-runs must be nonnegative and forward-runs positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    workers: dict[str, dict[str, Any]] = {}
    sampling: dict[str, dict[str, Any]] = {}
    commands: dict[str, list[str]] = {}
    started = time.perf_counter()
    for mode in ("full", "compressed"):
        worker, rss, command = _sample_worker(
            mode=mode,
            checkpoint=checkpoint,
            out_dir=out_dir,
            args=args,
        )
        workers[mode] = worker
        sampling[mode] = rss
        commands[mode] = _portable_command(command)

    comparison = compare_trained_decoder_workers(
        workers["full"],
        workers["compressed"],
        rss_samples=sampling,
        atol=float(args.atol),
    )
    valid = bool(comparison["acceptance"]["valid"])
    payload = {
        "schema_version": 1,
        "profile": "wpc_cips_finetuned_decoder_isolated_resource_benchmark",
        "status": "ok" if valid else "invalid",
        "timing_policy": (
            "full and compressed checkpoint-decoder paths run sequentially in "
            "distinct fresh processes; warmups excluded; repeated forward wall "
            "times include the trained Cheb7 and real bootstrap but exclude "
            "compilation, encryption, decryption, and cleanup"
        ),
        "elapsed_s": float(time.perf_counter() - started),
        "worker_order": ["full", "compressed"],
        "checkpoint": workers["full"]["checkpoint"],
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
    print(
        json.dumps(
            {
                "status": payload["status"],
                "result": str(result_path),
                "report": str(report_path),
                "latency": comparison["latency"],
                "memory": {
                    key: value
                    for key, value in comparison["memory"].items()
                    if key not in ("rss_sampling", "interpretation")
                },
                "correctness": comparison["correctness"],
                "operations": comparison["operations"],
                "acceptance": comparison["acceptance"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
