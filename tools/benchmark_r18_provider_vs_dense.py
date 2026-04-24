from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental.cir.runtime_group import RegionFirstCompileRegistry
from orion.models.resnet import ResNet18
from orion.nn.module import Module


DEFAULT_OUT = Path("/tmp/orion_r18_provider_vs_dense_bench.json")
DEFAULT_CASES = ("stage1_same", "stage2_same", "stage3_same", "stage4_same")


CASE_SPECS = {
    "stage1_same": {"node": "layers_0_0_conv1", "seed": 101},
    "stage2_same": {"node": "layers_1_1_conv1", "seed": 202},
    "stage3_same": {"node": "layers_2_1_conv1", "seed": 303},
    "stage4_same": {"node": "layers_3_0_conv2", "seed": 404},
}


def _build_config() -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": 16,
            "LogQ": [45, 30, 30, 45],
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


def _emit_phase(phase: str, **data: Any) -> None:
    print(json.dumps({"phase": str(phase), **data}), flush=True)


def _prepare_r18_dag() -> NetworkDAG:
    torch.manual_seed(0)
    net = ResNet18(dataset="tiny")
    net.eval()
    traced = OrionTracer().trace_model(net)
    StatsTracker(traced).propagate(torch.randn((1, 3, 64, 64), dtype=torch.float32))
    dag = NetworkDAG(traced)
    dag.build_dag()
    for module in net.modules():
        if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
            module.init_orion_params()
    return dag


def _set_compile_level(dag: NetworkDAG) -> None:
    level = len(scheme.params.get_logq()) - 1
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None and hasattr(module, "set_level"):
            module.set_level(level)


def _prebuilt_sources(module, *, count: int, repeats: int, seed: int) -> list[CipherTensor]:
    sources: list[CipherTensor] = []
    level = len(scheme.params.get_logq()) - 1
    gen = torch.Generator().manual_seed(int(seed))
    for _ in range(int(repeats)):
        ids: list[int] = []
        for _source_index in range(int(count)):
            packed = torch.randn((scheme.params.get_slots(),), generator=gen, dtype=torch.float32) * 0.01
            ct = scheme.encrypt(scheme.encode(packed, level))
            ids.append(int(ct.ids[0]))
            ct.ids = []
        sources.append(CipherTensor(scheme, ids, module.input_shape, module.fhe_input_shape))
    return sources


def _dense_cols(module) -> int:
    keys = list(getattr(module, "transform_ids", {}).keys())
    if not keys:
        return 0
    return max(int(col) for _row, col in keys) + 1


def _bench_dense_single(node_name: str, *, repeats: int, seed: int) -> dict[str, Any]:
    dag = _prepare_r18_dag()
    module = dag.nodes[str(node_name)]["module"]
    _set_compile_level(dag)
    generate_started = time.perf_counter()
    module.generate_diagonals(last=False)
    generate_diagonals_s = float(time.perf_counter() - generate_started)
    _emit_phase("dense_single_generate_done", node=str(node_name), seconds=float(generate_diagonals_s))
    compile_started = time.perf_counter()
    module.compile()
    compile_backend_s = float(time.perf_counter() - compile_started)
    _emit_phase("dense_single_compile_done", node=str(node_name), seconds=float(compile_backend_s))
    compile_s = float(generate_diagonals_s + compile_backend_s)
    module.he_mode = True
    cols = _dense_cols(module)
    warm_source = _prebuilt_sources(module, count=int(cols), repeats=1, seed=int(seed))[0]
    warm_started = time.perf_counter()
    _ = module(warm_source)
    warmup_s = float(time.perf_counter() - warm_started)
    _emit_phase("dense_single_warmup_done", node=str(node_name), seconds=float(warmup_s))
    run_sources = _prebuilt_sources(module, count=int(cols), repeats=int(repeats), seed=int(seed + 1))
    run_times: list[float] = []
    for source in run_sources:
        started = time.perf_counter()
        out = module(source)
        run_times.append(float(time.perf_counter() - started))
        del out
    return {
        "path": "dense",
        "node": str(node_name),
        "compile_s": float(compile_s),
        "generate_diagonals_s": float(generate_diagonals_s),
        "compile_backend_s": float(compile_backend_s),
        "warmup_s": float(warmup_s),
        "source_ciphertext_count": int(cols),
        "output_ciphertext_count": int(len(getattr(module, "transform_ids", {})) // max(1, cols)),
        "hot_run_s": run_times,
        "hot_run_median_s": float(statistics.median(run_times)),
    }


def _bench_provider_single(node_name: str, *, repeats: int, seed: int) -> dict[str, Any]:
    dag = _prepare_r18_dag()
    registry = RegionFirstCompileRegistry.for_r18_tiny(dag)
    attach = registry.attach_to_dag(dag)
    module = dag.nodes[str(node_name)]["module"]
    _set_compile_level(dag)
    generate_started = time.perf_counter()
    module.generate_diagonals(last=False)
    generate_diagonals_s = float(time.perf_counter() - generate_started)
    _emit_phase("provider_single_generate_done", node=str(node_name), seconds=float(generate_diagonals_s))
    compile_started = time.perf_counter()
    module.compile()
    compile_backend_s = float(time.perf_counter() - compile_started)
    _emit_phase("provider_single_compile_done", node=str(node_name), seconds=float(compile_backend_s))
    compile_s = float(generate_diagonals_s + compile_backend_s)
    module.he_mode = True
    runtime = module.region_runtime
    executor = runtime.executor
    cols = int(getattr(executor, "source_pair_count", 0) * 2) if hasattr(executor, "source_pair_count") else 1
    if hasattr(executor, "groups") and executor.groups:
        cols = len(executor.groups)
    if cols <= 0:
        cols = 1
    warm_source = _prebuilt_sources(module, count=int(cols), repeats=1, seed=int(seed))[0]
    runtime = module.region_runtime
    if str(getattr(runtime, "stage", "")) == "stage4":
        warm_source.region_first_compact_source = True
    warm_started = time.perf_counter()
    warm_out = module(warm_source)
    warmup_s = float(time.perf_counter() - warm_started)
    _emit_phase("provider_single_warmup_done", node=str(node_name), seconds=float(warmup_s))
    run_sources = _prebuilt_sources(module, count=int(cols), repeats=int(repeats), seed=int(seed + 1))
    run_times: list[float] = []
    executor_timings: list[dict[str, float]] = []
    for source in run_sources:
        if str(getattr(runtime, "stage", "")) == "stage4":
            source.region_first_compact_source = True
        started = time.perf_counter()
        out = module(source)
        run_times.append(float(time.perf_counter() - started))
        executor_timings.append(dict(getattr(executor, "last_runtime_timing", {})))
        del out
    return {
        "path": "provider",
        "node": str(node_name),
        "compile_s": float(compile_s),
        "generate_diagonals_s": float(generate_diagonals_s),
        "compile_backend_s": float(compile_backend_s),
        "warmup_s": float(warmup_s),
        "attach_audit": attach,
        "source_ciphertext_count": int(cols),
        "output_ciphertext_count": int(len(warm_out.ids)),
        "hot_run_s": run_times,
        "hot_run_median_s": float(statistics.median(run_times)),
        "executor_last_runtime_timing": dict(getattr(executor, "last_runtime_timing", {})),
        "executor_hot_runtime_timings": executor_timings,
    }


def _bench_case_path(case_name: str, *, path_kind: str, repeats: int) -> dict[str, Any]:
    spec = dict(CASE_SPECS[str(case_name)])
    config = _build_config()

    def _run_once(fn):
        scheme.init_scheme(config)
        try:
            Module.set_scheme(scheme)
            Module.set_margin(scheme.params.get_margin())
            return fn()
        finally:
            try:
                scheme.delete_scheme()
            except Exception:
                pass
            gc.collect()

    if str(path_kind) == "dense":
        return _run_once(lambda: _bench_dense_single(str(spec["node"]), repeats=int(repeats), seed=int(spec["seed"])))
    return _run_once(lambda: _bench_provider_single(str(spec["node"]), repeats=int(repeats), seed=int(spec["seed"])))


def _worker_main(case_name: str, *, path_kind: str, repeats: int) -> int:
    payload = _bench_case_path(str(case_name), path_kind=str(path_kind), repeats=int(repeats))
    print(json.dumps(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark R18 provider kernels against Orion dense conv on representative stages.")
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout-s", type=int, default=1000)
    parser.add_argument("--worker-case", default="")
    parser.add_argument("--worker-path", choices=("dense", "provider"), default="")
    args = parser.parse_args()

    if str(args.worker_case):
        return _worker_main(str(args.worker_case), path_kind=str(args.worker_path), repeats=int(args.repeats))

    payload = {
        "status": "started",
        "scope": "R18 representative conv compile/runtime comparison: provider bridge vs Orion dense",
        "config": _build_config(),
        "cases": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for case_name in args.cases:
        case_payload: dict[str, Any] = {"case": str(case_name), "status": "running", "paths": {}}
        payload["cases"].append(case_payload)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        for path_kind in ("dense", "provider"):
            started = time.perf_counter()
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-case",
                str(case_name),
                "--worker-path",
                str(path_kind),
                "--repeats",
                str(int(args.repeats)),
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=int(args.timeout_s),
                    check=False,
                )
                elapsed = float(time.perf_counter() - started)
                if completed.returncode != 0:
                    case_payload["paths"][str(path_kind)] = {
                        "status": "failed",
                        "elapsed_s": float(elapsed),
                        "returncode": int(completed.returncode),
                        "stdout": str(completed.stdout),
                        "stderr": str(completed.stderr),
                    }
                else:
                    lines = [line for line in str(completed.stdout).splitlines() if line.strip()]
                    payload_line = lines[-1] if lines else "{}"
                    case_payload["paths"][str(path_kind)] = {
                        "status": "ok",
                        "elapsed_s": float(elapsed),
                        "result": json.loads(payload_line),
                        "stdout": str(completed.stdout),
                    }
            except subprocess.TimeoutExpired as exc:
                elapsed = float(time.perf_counter() - started)
                case_payload["paths"][str(path_kind)] = {
                    "status": "timeout",
                    "elapsed_s": float(elapsed),
                    "timeout_s": int(args.timeout_s),
                    "stdout": "" if exc.stdout is None else str(exc.stdout),
                    "stderr": "" if exc.stderr is None else str(exc.stderr),
                }
            Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        dense_entry = case_payload["paths"].get("dense", {})
        provider_entry = case_payload["paths"].get("provider", {})
        if dense_entry.get("status") == "ok" and provider_entry.get("status") == "ok":
            dense = dict(dense_entry["result"])
            provider = dict(provider_entry["result"])
            case_payload.update(
                {
                    "status": "ok",
                    "dense": dense,
                    "provider": provider,
                    "delta": {
                        "compile_s": float(provider["compile_s"] - dense["compile_s"]),
                        "hot_run_median_s": float(provider["hot_run_median_s"] - dense["hot_run_median_s"]),
                    },
                    "speedup": {
                        "compile_ratio_dense_over_provider": None if float(provider["compile_s"]) == 0.0 else float(dense["compile_s"] / provider["compile_s"]),
                        "hot_run_ratio_dense_over_provider": None
                        if float(provider["hot_run_median_s"]) == 0.0
                        else float(dense["hot_run_median_s"] / provider["hot_run_median_s"]),
                    },
                }
            )
        else:
            case_payload["status"] = "partial"
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    payload["status"] = "ok"
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
