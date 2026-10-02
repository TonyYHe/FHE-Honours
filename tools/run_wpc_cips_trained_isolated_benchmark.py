#!/usr/bin/env python3
"""Benchmark the trained WPC decoder with isolated full/compressed workers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
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


DEFAULT_OUT_DIR = (
    REPO_ROOT
    / ".tmp/results/honours/25_wpc_finetuned_decoder_isolated_benchmark"
)
WORKER = REPO_ROOT / "tools/run_wpc_cips_trained_isolated_worker.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--feature-std", type=float, default=0.02)
    parser.add_argument("--bound-headroom", type=float, default=1.25)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--forward-runs", type=int, default=10)
    parser.add_argument("--rss-sample-ms", type=float, default=5.0)
    parser.add_argument("--atol", type=float, default=2e-3)
    return parser


def _sample_worker(
    *,
    mode: str,
    checkpoint: Path,
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
                    int(peak_rss.get(phase, 0)),
                    int(sample["rss_bytes"]),
                )
                if "hwm_bytes" in sample:
                    peak_hwm[phase] = max(
                        int(peak_hwm.get(phase, 0)),
                        int(sample["hwm_bytes"]),
                    )
            time.sleep(float(args.rss_sample_ms) / 1000.0)
        return_code = int(process.wait())

    if return_code != 0:
        tail = log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-100:]
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
