from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.orion import scheme
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import U22CompileRegistry
from orion.models.unet import UNet22
from orion.nn.module import Module

from tools.validate_u22_unique_nodes_lattigo_vs_orion import _cleanup_scheme, _encode_input


DEFAULT_OUT = Path("/tmp/orion_u22_provider_vs_dense_bench.json")
DECODER_TCONV_NODES = ("up4", "up3", "up2", "up1")
DATASET_IMAGE_SIZES = {"tiny": 64, "imagenet": 256}


def _init_scheme(*, logn: int) -> None:
    config = {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [45, 30, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    Module.set_scheme(scheme)
    Module.set_margin(scheme.params.get_margin())


def _prepared_dag(*, dataset: str, base_channels: int, provider: bool) -> tuple[NetworkDAG, dict[str, Any]]:
    torch.manual_seed(0)
    image_size = int(DATASET_IMAGE_SIZES[str(dataset)])
    traced = OrionTracer().trace_model(UNet22(dataset=str(dataset), base_channels=int(base_channels)))
    StatsTracker(traced).propagate(torch.randn((1, 3, int(image_size), int(image_size)), dtype=torch.float32))
    dag = NetworkDAG(traced)
    dag.build_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()
    attach_audit: dict[str, Any] = {}
    if bool(provider):
        registry = U22CompileRegistry.for_dag(dag)
        attach_audit = registry.attach_to_dag(dag)
    _set_compile_level(dag)
    return dag, attach_audit


def _set_compile_level(dag: NetworkDAG) -> None:
    level = len(scheme.params.get_logq()) - 1
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None and hasattr(module, "set_level"):
            module.set_level(level)


def _make_sources(module, *, repeats: int, seed: int) -> list[Any]:
    gen = torch.Generator().manual_seed(int(seed))
    sources: list[Any] = []
    for _ in range(int(repeats)):
        x = torch.randn(tuple(int(v) for v in module.input_shape[1:]), generator=gen, dtype=torch.float32)
        sources.append(_encode_input(module, x))
    return sources


def _bench_case(
    *,
    dataset: str,
    node_name: str,
    base_channels: int,
    logn: int,
    repeats: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "running",
        "dataset": str(dataset),
        "node": str(node_name),
        "base_channels": int(base_channels),
        "logn": int(logn),
        "slots": None,
        "paths": {},
    }
    _init_scheme(logn=int(logn))
    try:
        payload["slots"] = int(scheme.params.get_slots())
        dense_dag, _ = _prepared_dag(dataset=str(dataset), base_channels=int(base_channels), provider=False)
        provider_dag, attach = _prepared_dag(dataset=str(dataset), base_channels=int(base_channels), provider=True)

        dense = dense_dag.nodes[str(node_name)]["module"]
        provider = provider_dag.nodes[str(node_name)]["module"]
        runtime = getattr(provider, "region_runtime", None)
        provider_supported = bool(runtime is not None and runtime.supports_scheme(scheme))
        payload["provider_supports_scheme"] = bool(provider_supported)
        payload["module_shapes"] = {
            "input_shape": [int(v) for v in dense.input_shape],
            "output_shape": [int(v) for v in dense.output_shape],
            "fhe_input_shape": [int(v) for v in dense.fhe_input_shape],
            "fhe_output_shape": [int(v) for v in dense.fhe_output_shape],
            "fhe_input_numel": int(torch.Size(dense.fhe_input_shape).numel()),
            "fhe_output_numel": int(torch.Size(dense.fhe_output_shape).numel()),
        }

        t0 = time.perf_counter()
        dense.generate_diagonals(last=False)
        dense_generate_s = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        dense.compile()
        dense_compile_backend_s = float(time.perf_counter() - t0)
        dense.he_mode = True
        dense_warm_source = _make_sources(dense, repeats=1, seed=11)[0]
        t0 = time.perf_counter()
        dense_warm_out = dense(dense_warm_source)
        dense_warmup_s = float(time.perf_counter() - t0)
        dense_run_times: list[float] = []
        for source in _make_sources(dense, repeats=int(repeats), seed=101):
            t0 = time.perf_counter()
            out = dense(source)
            dense_run_times.append(float(time.perf_counter() - t0))
            del out
            del source
        payload["paths"]["dense"] = {
            "status": "ok",
            "compile_s": float(dense_generate_s + dense_compile_backend_s),
            "generate_diagonals_s": float(dense_generate_s),
            "compile_backend_s": float(dense_compile_backend_s),
            "warmup_s": float(dense_warmup_s),
            "hot_run_s": [float(value) for value in dense_run_times],
            "hot_run_median_s": float(statistics.median(dense_run_times)),
            "hot_run_mean_s": float(statistics.fmean(dense_run_times)),
            "source_ciphertext_count": int(len(dense_warm_source.ids)),
            "output_ciphertext_count": int(len(dense_warm_out.ids)),
        }
        del dense_warm_out
        del dense_warm_source

        if not bool(provider_supported):
            payload["paths"]["provider"] = {
                "status": "unsupported",
                "attach_audit": dict(attach),
                "strategy": "" if runtime is None else str(getattr(runtime, "strategy", "")),
            }
            payload["status"] = "provider_unsupported"
            return payload

        t0 = time.perf_counter()
        provider.generate_diagonals(last=False)
        provider_generate_s = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        provider.compile()
        provider_compile_backend_s = float(time.perf_counter() - t0)
        provider.he_mode = True
        provider_warm_source = _make_sources(provider, repeats=1, seed=22)[0]
        t0 = time.perf_counter()
        provider_warm_out = provider(provider_warm_source)
        provider_warmup_s = float(time.perf_counter() - t0)
        provider_run_times: list[float] = []
        for source in _make_sources(provider, repeats=int(repeats), seed=202):
            t0 = time.perf_counter()
            out = provider(source)
            provider_run_times.append(float(time.perf_counter() - t0))
            del out
            del source
        payload["paths"]["provider"] = {
            "status": "ok",
            "compile_s": float(provider_generate_s + provider_compile_backend_s),
            "generate_diagonals_s": float(provider_generate_s),
            "compile_backend_s": float(provider_compile_backend_s),
            "warmup_s": float(provider_warmup_s),
            "hot_run_s": [float(value) for value in provider_run_times],
            "hot_run_median_s": float(statistics.median(provider_run_times)),
            "hot_run_mean_s": float(statistics.fmean(provider_run_times)),
            "source_ciphertext_count": int(len(provider_warm_source.ids)),
            "output_ciphertext_count": int(len(provider_warm_out.ids)),
            "attach_audit": dict(attach),
            "strategy": "" if runtime is None else str(getattr(runtime, "strategy", "")),
        }
        del provider_warm_out
        del provider_warm_source

        payload["delta"] = {
            "compile_s": float(payload["paths"]["provider"]["compile_s"] - payload["paths"]["dense"]["compile_s"]),
            "hot_run_median_s": float(payload["paths"]["provider"]["hot_run_median_s"] - payload["paths"]["dense"]["hot_run_median_s"]),
            "hot_run_mean_s": float(payload["paths"]["provider"]["hot_run_mean_s"] - payload["paths"]["dense"]["hot_run_mean_s"]),
        }
        payload["speedup"] = {
            "compile_ratio_dense_over_provider": (
                None
                if float(payload["paths"]["provider"]["compile_s"]) == 0.0
                else float(payload["paths"]["dense"]["compile_s"] / payload["paths"]["provider"]["compile_s"])
            ),
            "hot_run_ratio_dense_over_provider": (
                None
                if float(payload["paths"]["provider"]["hot_run_median_s"]) == 0.0
                else float(payload["paths"]["dense"]["hot_run_median_s"] / payload["paths"]["provider"]["hot_run_median_s"])
            ),
        }
        payload["status"] = "ok"
        return payload
    finally:
        _cleanup_scheme()
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark U22 per-node provider vs Orion dense with explicit base_channels/logn.")
    parser.add_argument("--dataset", required=True, choices=("tiny", "imagenet"))
    parser.add_argument("--nodes", nargs="*", default=list(DECODER_TCONV_NODES))
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--logn", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    payload = {
        "status": "running",
        "scope": "U22 per-node provider vs Orion dense benchmark",
        "dataset": str(args.dataset),
        "base_channels": int(args.base_channels),
        "logn": int(args.logn),
        "repeats": int(args.repeats),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for node_name in args.nodes:
        row = _bench_case(
            dataset=str(args.dataset),
            node_name=str(node_name),
            base_channels=int(args.base_channels),
            logn=int(args.logn),
            repeats=int(args.repeats),
        )
        rows.append(row)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"phase": "case_done", "dataset": str(args.dataset), "node": str(node_name), "status": row["status"]}), flush=True)

    payload["status"] = "ok"
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
