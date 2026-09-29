#!/usr/bin/env python3
"""Run one isolated full or compressed WPC CIPS benchmark worker."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import orion
from orion.experimental.wpc_cips_benchmark import summarize_samples
from orion.experimental.wpc_cips_layer import (
    WPC_GLOBAL_STATS_FIELDS,
    WPC_STORAGE_COMPRESSED,
    WPC_STORAGE_FULL,
)
from orion.nn import Conv2d


GO_MEMORY_FIELDS = (
    "alloc_bytes",
    "total_alloc_bytes",
    "sys_bytes",
    "heap_alloc_bytes",
    "heap_sys_bytes",
    "heap_idle_bytes",
    "heap_released_bytes",
    "heap_inuse_bytes",
    "stack_inuse_bytes",
    "mspan_inuse_bytes",
    "mcache_inuse_bytes",
    "num_gc",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "compressed"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--forward-runs", type=int, default=10)
    parser.add_argument("--atol", type=float, default=2e-6)
    return parser


def _write_phase(path: Path, phase: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps({"phase": str(phase), "monotonic_s": time.monotonic()}),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _current_rss_bytes() -> tuple[int, str]:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024, "proc_status_vmrss"
    if sys.platform == "darwin":
        try:
            value = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                text=True,
            ).strip()
            if value:
                return int(value) * 1024, "ps_rss"
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value, "ru_maxrss_fallback"
    return value * 1024, "ru_maxrss_fallback"


def _maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _go_memory(backend: Any) -> dict[str, int]:
    values = [int(value) for value in backend.GetRuntimeMemoryStats()]
    if len(values) != len(GO_MEMORY_FIELDS):
        raise RuntimeError(
            f"unexpected Go memory-statistics length {len(values)}"
        )
    return {name: value for name, value in zip(GO_MEMORY_FIELDS, values)}


def _memory_snapshot(backend: Any) -> dict[str, Any]:
    current, source = _current_rss_bytes()
    return {
        "current_rss_bytes": int(current),
        "current_rss_source": source,
        "process_maxrss_bytes": int(_maxrss_bytes()),
        "go": _go_memory(backend),
    }


def _collect_runtime_memory(backend: Any) -> None:
    collect = getattr(backend, "CollectRuntimeMemory", None)
    if not callable(collect):
        raise RuntimeError("backend does not expose CollectRuntimeMemory")
    collect(1)


def _config(logn: int) -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [55, 45, 45],
            "LogP": [60],
            "LogScale": 45,
            "H": 192,
            "RingType": "Standard",
        },
        "orion": {
            "backend": "lattigo",
            "embedding_method": "square",
            "io_mode": "none",
            "debug": False,
        },
    }


def _make_layer(
    *,
    name: str,
    channels: int,
    level: int,
    bsgs_ratio: float,
    weight: np.ndarray,
    bias: np.ndarray,
) -> Conv2d:
    layer = Conv2d(
        int(channels),
        int(channels),
        3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        bsgs_ratio=float(bsgs_ratio),
        level=int(level),
    )
    layer.name = str(name)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    layer.init_orion_params()
    return layer


def _operation_counters(backend: Any) -> dict[str, int]:
    values = [int(value) for value in backend.GetOperationCounters()]
    if len(values) != 4:
        raise RuntimeError("unexpected operation-counter length")
    return dict(
        zip(
            (
                "rotation_total",
                "linear_transform_rotation",
                "direct_rotation",
                "conjugation",
            ),
            values,
        )
    )


def _per_forward(counters: dict[str, int], runs: int) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for name, value in counters.items():
        result[name] = (
            int(value // runs)
            if int(value) % int(runs) == 0
            else float(value / runs)
        )
    return result


def _global_stats(backend: Any) -> dict[str, Any]:
    values = [int(value) for value in backend.GetWPCCompressedGlobalStats()]
    if len(values) != len(WPC_GLOBAL_STATS_FIELDS):
        raise RuntimeError("unexpected WPC global-statistics length")
    result = {
        name: int(value) for name, value in zip(WPC_GLOBAL_STATS_FIELDS, values)
    }
    full_bytes = int(result["aggregate_full_payload_bytes"])
    stored_bytes = int(result["aggregate_stored_payload_plus_metadata_bytes"])
    result["aggregate_storage_compression_ratio_including_metadata"] = (
        float(full_bytes / stored_bytes) if stored_bytes else None
    )
    return result


def _aggregate_storage(plans: list[Any]) -> dict[str, Any]:
    rows = {plan.layer_name: plan.storage_summary() for plan in plans}
    fields = (
        "full_weight_qp_payload_bytes",
        "resident_weight_qp_payload_bytes",
        "weight_metadata_bytes",
        "uncompressed_bias_q_payload_bytes",
        "full_weight_plus_bias_payload_bytes",
        "stored_weight_plus_metadata_plus_bias_bytes",
    )
    result: dict[str, Any] = {
        name: int(sum(int(row[name]) for row in rows.values()))
        for name in fields
    }
    result["storage_mode"] = plans[0].storage_mode
    result["transform_count"] = int(sum(row["transform_count"] for row in rows.values()))
    result["full_to_stored_ratio"] = float(
        result["full_weight_plus_bias_payload_bytes"]
        / result["stored_weight_plus_metadata_plus_bias_bytes"]
    )
    result["by_layer"] = rows
    return result


def _run_pipeline(layers: list[Conv2d], input_ciphertext: Any):
    first = layers[0](input_ciphertext)
    try:
        return layers[1](first)
    finally:
        first.release()


def main() -> int:
    args = _parser().parse_args()
    if int(args.warmup_runs) < 0 or int(args.forward_runs) <= 0:
        raise SystemExit("warmup-runs must be nonnegative and forward-runs positive")
    slots = 1 << (int(args.logn) - 1)
    channel_capacity = int(slots // (int(args.height) * int(args.width)))
    if int(args.channels) <= channel_capacity:
        raise SystemExit(
            f"channels must exceed one-ciphertext capacity {channel_capacity}"
        )

    _write_phase(args.phase_file, "startup")
    rng = np.random.default_rng(int(args.seed))
    input_values = rng.normal(
        0.0,
        0.4,
        size=(1, int(args.channels), int(args.height), int(args.width)),
    ).astype(np.float64)
    weights = [
        rng.normal(
            0.0,
            0.08,
            size=(int(args.channels), int(args.channels), 3, 3),
        ).astype(np.float64)
        for _ in range(2)
    ]
    biases = [
        rng.normal(0.0, 0.03, size=(int(args.channels),)).astype(np.float64)
        for _ in range(2)
    ]

    scheme = None
    layers: list[Conv2d] = []
    plans: list[Any] = []
    input_ciphertext = None
    final_output = None
    payload: dict[str, Any] | None = None
    try:
        _write_phase(args.phase_file, "scheme_init")
        scheme_started = time.perf_counter()
        scheme = orion.init_scheme(_config(int(args.logn)))
        scheme_init_s = float(time.perf_counter() - scheme_started)
        Conv2d.set_scheme(scheme)
        memory: dict[str, Any] = {
            "after_scheme_init": _memory_snapshot(scheme.backend)
        }
        layers = [
            _make_layer(
                name="wpc_block_conv1",
                channels=int(args.channels),
                level=2,
                bsgs_ratio=float(args.bsgs_ratio),
                weight=weights[0],
                bias=biases[0],
            ),
            _make_layer(
                name="wpc_block_conv2",
                channels=int(args.channels),
                level=1,
                bsgs_ratio=float(args.bsgs_ratio),
                weight=weights[1],
                bias=biases[1],
            ),
        ]

        _write_phase(args.phase_file, "compile")
        compile_started = time.perf_counter()
        for layer in layers:
            plans.append(
                layer.install_wpc_cips_plan(
                    (1, int(args.channels), int(args.height), int(args.width)),
                    storage_mode=str(args.mode),
                    include_full_control=False,
                    verify_exact_qp=False,
                )
            )
            layer.he()
        compile_s = float(time.perf_counter() - compile_started)
        _write_phase(args.phase_file, "post_compile_raw")
        memory["post_compile_raw"] = _memory_snapshot(scheme.backend)
        _collect_runtime_memory(scheme.backend)
        _write_phase(args.phase_file, "post_compile_gc")
        memory["post_compile_gc"] = _memory_snapshot(scheme.backend)

        reference_first = plans[0].clear_reference(input_values)
        reference_second = plans[1].clear_reference(reference_first[None, ...])
        if plans[0].output_packing_signature != plans[1].input_packing_signature:
            raise RuntimeError("layer CIPS signatures do not chain")

        _write_phase(args.phase_file, "input_prepare")
        input_ciphertext = plans[0].encrypt_input(input_values)
        _collect_runtime_memory(scheme.backend)
        memory["after_input_gc"] = _memory_snapshot(scheme.backend)

        online_encode_call_count = 0
        original_encode = scheme.encoder.encode

        def counted_online_encode(*encode_args, **encode_kwargs):
            nonlocal online_encode_call_count
            online_encode_call_count += 1
            return original_encode(*encode_args, **encode_kwargs)

        scheme.encoder.encode = counted_online_encode
        try:
            _write_phase(args.phase_file, "warmup")
            for _ in range(int(args.warmup_runs)):
                warmup_output = _run_pipeline(layers, input_ciphertext)
                warmup_output.release()
            _collect_runtime_memory(scheme.backend)
            _write_phase(args.phase_file, "pre_measured_gc")
            memory["pre_measured_gc"] = _memory_snapshot(scheme.backend)

            scheme.backend.ResetOperationCounters()
            if args.mode == WPC_STORAGE_COMPRESSED:
                scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
            forward_wall_s: list[float] = []
            decompression_s: list[float] = []
            transform_evaluate_s: list[float] = []
            _write_phase(args.phase_file, "measured")
            for _ in range(int(args.forward_runs)):
                started = time.perf_counter()
                output = _run_pipeline(layers, input_ciphertext)
                forward_wall_s.append(float(time.perf_counter() - started))
                if final_output is not None:
                    final_output.release()
                final_output = output
                if args.mode == WPC_STORAGE_COMPRESSED:
                    runtime_rows = [
                        row
                        for plan in plans
                        for row in plan.compressed_stats().values()
                    ]
                    decompression_s.append(
                        float(sum(row["last_decompress_s"] for row in runtime_rows))
                    )
                    transform_evaluate_s.append(
                        float(sum(row["last_evaluate_s"] for row in runtime_rows))
                    )
                else:
                    decompression_s.append(0.0)
                    transform_evaluate_s.append(0.0)
            counters_total = _operation_counters(scheme.backend)
            _write_phase(args.phase_file, "post_measured")
            memory["after_measured"] = _memory_snapshot(scheme.backend)

            _write_phase(args.phase_file, "correctness")
            decoded = plans[1].decrypt_unpack(final_output)
        finally:
            scheme.encoder.encode = original_encode

        error = np.abs(decoded - reference_second)
        storage = _aggregate_storage(plans)
        compressed_global_stats = _global_stats(scheme.backend)
        compressed_rows = {
            plan.layer_name: plan.compressed_stats() for plan in plans
        }
        worker_acceptance = {
            "correct_vs_clear": bool(
                np.allclose(
                    decoded,
                    reference_second,
                    rtol=0.0,
                    atol=float(args.atol),
                )
            ),
            "expected_transform_count": storage["transform_count"] == 8,
            "zero_online_python_encode_calls": online_encode_call_count == 0,
            "storage_mode_matches_request": storage["storage_mode"] == args.mode,
            "measured_forward_count_matches_request": len(forward_wall_s)
            == int(args.forward_runs),
            "compressed_registry_matches_mode": bool(
                (
                    args.mode == WPC_STORAGE_COMPRESSED
                    and compressed_global_stats["registered_transform_count"] == 8
                )
                or (
                    args.mode == WPC_STORAGE_FULL
                    and compressed_global_stats["registered_transform_count"] == 0
                )
            ),
            "compressed_materialization_released": bool(
                args.mode == WPC_STORAGE_FULL
                or (
                    compressed_global_stats[
                        "current_materialized_full_payload_bytes"
                    ]
                    == 0
                    and compressed_global_stats[
                        "current_materialized_transform_count"
                    ]
                    == 0
                )
            ),
        }
        worker_acceptance["valid"] = all(worker_acceptance.values())
        payload = {
            "schema_version": 1,
            "profile": "wpc_cips_isolated_worker",
            "status": "ok" if worker_acceptance["valid"] else "invalid",
            "mode": str(args.mode),
            "seed": int(args.seed),
            "process": {
                "pid": int(os.getpid()),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "experiment": {
                "logn": int(args.logn),
                "slots": int(slots),
                "height": int(args.height),
                "width": int(args.width),
                "channels": int(args.channels),
                "bsgs_ratio": float(args.bsgs_ratio),
                "warmup_runs": int(args.warmup_runs),
                "forward_runs": int(args.forward_runs),
                "atol": float(args.atol),
                "levels": [2, 1],
                "kernel": [3, 3],
                "padding": "rotation_padding",
            },
            "timing_policy": (
                "fresh_process_with_warmup_and_repetitions; compilation, input "
                "preparation, decoding, and cleanup excluded from forward latency"
            ),
            "scheme_init_s": scheme_init_s,
            "compile_s": compile_s,
            "storage": storage,
            "measurements": {
                "forward_wall_s": forward_wall_s,
                "forward_wall_summary_s": summarize_samples(forward_wall_s),
                "decompression_s": decompression_s,
                "decompression_summary_s": summarize_samples(decompression_s),
                "transform_evaluate_s": transform_evaluate_s,
                "transform_evaluate_summary_s": summarize_samples(
                    transform_evaluate_s
                ),
                "operation_counters_total": counters_total,
                "operation_counters_per_forward": _per_forward(
                    counters_total, int(args.forward_runs)
                ),
                "online_python_encode_call_count": int(
                    online_encode_call_count
                ),
            },
            "correctness": {
                "correct": worker_acceptance["correct_vs_clear"],
                "max_abs_error": float(np.max(error)),
                "mean_abs_error": float(np.mean(error)),
                "output_shape": list(decoded.shape),
                "output_values": decoded.reshape(-1).tolist(),
            },
            "memory": memory,
            "backend": {
                "compressed_global_stats": compressed_global_stats,
                "compressed_transform_stats": compressed_rows,
                "weight_plaintext_offline_encode_calls": (
                    int(compressed_global_stats[
                        "total_weight_plaintext_offline_encode_calls"
                    ])
                    if args.mode == WPC_STORAGE_COMPRESSED
                    else int(storage["transform_count"])
                ),
                "weight_plaintext_offline_encode_accounting": (
                    "backend_counter"
                    if args.mode == WPC_STORAGE_COMPRESSED
                    else "one_GenerateLinearTransform_call_per_group_transform"
                ),
                "weight_plaintext_online_encode_calls": (
                    int(compressed_global_stats[
                        "total_weight_plaintext_online_encode_calls"
                    ])
                    if args.mode == WPC_STORAGE_COMPRESSED
                    else 0
                ),
            },
            "acceptance": worker_acceptance,
        }
    finally:
        if final_output is not None:
            final_output.release()
        if input_ciphertext is not None:
            input_ciphertext.release()
        for layer in layers:
            layer.remove_wpc_cips_plan()
        if scheme is not None:
            scheme.delete_scheme()

    if payload is None:
        raise RuntimeError("worker did not produce a result")
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_phase(args.phase_file, "complete")
    print(json.dumps({
        "mode": payload["mode"],
        "status": payload["status"],
        "compile_s": payload["compile_s"],
        "forward_wall_summary_s": payload["measurements"][
            "forward_wall_summary_s"
        ],
        "storage": payload["storage"],
        "correctness": {
            key: value
            for key, value in payload["correctness"].items()
            if key != "output_values"
        },
        "acceptance": payload["acceptance"],
    }, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
