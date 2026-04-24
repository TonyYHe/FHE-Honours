from __future__ import annotations

import argparse
import gc
import importlib.util
import json
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
from orion.nn.module import Module


DEFAULT_OUT = Path("/tmp/orion_provider_dense_mae_validation.json")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R34_BENCH = _load_module(REPO_ROOT / "tools" / "benchmark_r34_provider_vs_dense.py", "bench_r34")
R18_BENCH = _load_module(REPO_ROOT / "tools" / "benchmark_r18_provider_vs_dense.py", "bench_r18")


def _result_metrics(dense: torch.Tensor, provider: torch.Tensor) -> dict[str, float]:
    dense_f = dense.detach().to(dtype=torch.float32)
    provider_f = provider.detach().to(dtype=torch.float32)
    diff = provider_f - dense_f
    return {
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.pow(2))).item()),
    }


def _decode(ctxt: CipherTensor) -> torch.Tensor:
    return ctxt.decrypt().decode().detach().cpu().to(dtype=torch.float32)


def _make_source(
    *,
    module: Any,
    count: int,
    seed: int,
) -> CipherTensor:
    level = len(scheme.params.get_logq()) - 1
    gen = torch.Generator().manual_seed(int(seed))
    ids: list[int] = []
    for _ in range(int(count)):
        packed = torch.randn((scheme.params.get_slots(),), generator=gen, dtype=torch.float32) * 0.01
        ct = scheme.encrypt(scheme.encode(packed, level))
        ids.append(int(ct.ids[0]))
        ct.ids = []
    return CipherTensor(scheme, ids, module.input_shape, module.fhe_input_shape)


def _init_scheme(config: dict[str, Any]) -> None:
    scheme.init_scheme(config)
    Module.set_scheme(scheme)
    Module.set_margin(scheme.params.get_margin())


def _cleanup_scheme() -> None:
    try:
        scheme.delete_scheme()
    except Exception:
        pass
    gc.collect()


def _compile_dense_single(helper: Any, node_name: str):
    dag = helper._prepare_fused_r34_dag() if hasattr(helper, "_prepare_fused_r34_dag") else helper._prepare_r18_dag()
    module = dag.nodes[str(node_name)]["module"]
    helper._set_compile_level(dag)
    module.generate_diagonals(last=False)
    module.compile()
    module.he_mode = True
    return dag, module


def _compile_provider_single(helper: Any, node_name: str):
    dag = helper._prepare_fused_r34_dag() if hasattr(helper, "_prepare_fused_r34_dag") else helper._prepare_r18_dag()
    if hasattr(helper, "R34CompileRegistry"):
        registry = helper.R34CompileRegistry.for_r34_imgnet_phase1(dag)
    else:
        registry = helper.RegionFirstCompileRegistry.for_r18_tiny(dag)
    registry.attach_to_dag(dag)
    module = dag.nodes[str(node_name)]["module"]
    helper._set_compile_level(dag)
    module.generate_diagonals(last=False)
    module.compile()
    module.he_mode = True
    return dag, module


def _source_count_dense(module: Any) -> int:
    keys = list(getattr(module, "transform_ids", {}).keys())
    if not keys:
        return 0
    return max(int(col) for _row, col in keys) + 1


def _source_count_provider(module: Any) -> int:
    runtime = module.region_runtime
    executor = runtime.executor
    if hasattr(executor, "cols"):
        return int(executor.cols)
    if hasattr(executor, "source_pair_count"):
        return int(executor.source_pair_count) * 2
    if hasattr(executor, "groups") and getattr(executor, "groups"):
        return len(executor.groups)
    return 1


def _mark_special_source(module: Any, source: CipherTensor) -> None:
    runtime = getattr(module, "region_runtime", None)
    if runtime is not None and str(getattr(runtime, "stage", "")) == "stage4":
        source.region_first_compact_source = True


def _validate_r34_single(case_name: str) -> dict[str, Any]:
    spec = dict(R34_BENCH.CASE_SPECS[str(case_name)])
    config = R34_BENCH._build_config()
    _init_scheme(config)
    try:
        _dense_dag, dense_module = _compile_dense_single(R34_BENCH, str(spec["node"]))
        _provider_dag, provider_module = _compile_provider_single(R34_BENCH, str(spec["node"]))
        count = max(int(_source_count_dense(dense_module)), int(_source_count_provider(provider_module)))
        source = _make_source(module=dense_module, count=int(count), seed=int(spec["seed"]))
        dense_out = dense_module(source)
        provider_out = provider_module(source)
        dense_decoded = _decode(dense_out)
        provider_decoded = _decode(provider_out)
        return {
            "network": "R34",
            "case": str(case_name),
            "kind": "single",
            **_result_metrics(dense_decoded, provider_decoded),
        }
    finally:
        _cleanup_scheme()


def _validate_r34_pair(case_name: str) -> dict[str, Any]:
    spec = dict(R34_BENCH.CASE_SPECS[str(case_name)])
    config = R34_BENCH._build_config()
    _init_scheme(config)
    try:
        dense_dag = R34_BENCH._prepare_fused_r34_dag()
        left_dense = dense_dag.nodes[str(spec["nodes"][0])]["module"]
        right_dense = dense_dag.nodes[str(spec["nodes"][1])]["module"]
        R34_BENCH._set_compile_level(dense_dag)
        for module in (left_dense, right_dense):
            module.generate_diagonals(last=False)
            module.compile()
            module.he_mode = True

        provider_dag = R34_BENCH._prepare_fused_r34_dag()
        registry = R34_BENCH.R34CompileRegistry.for_r34_imgnet_phase1(provider_dag)
        registry.attach_to_dag(provider_dag)
        left_provider = provider_dag.nodes[str(spec["nodes"][0])]["module"]
        right_provider = provider_dag.nodes[str(spec["nodes"][1])]["module"]
        R34_BENCH._set_compile_level(provider_dag)
        left_provider.generate_diagonals(last=False)
        right_provider.generate_diagonals(last=False)
        left_provider.compile()
        right_provider.compile()
        left_provider.he_mode = True
        right_provider.he_mode = True

        count = max(
            int(_source_count_dense(left_dense)),
            int(_source_count_provider(left_provider)),
        )
        source = _make_source(module=left_dense, count=int(count), seed=int(spec["seed"]))
        dense_left = _decode(left_dense(source))
        dense_right = _decode(right_dense(source))
        provider_left = _decode(left_provider(source))
        provider_right = _decode(right_provider(source))
        left_metrics = _result_metrics(dense_left, provider_left)
        right_metrics = _result_metrics(dense_right, provider_right)
        return {
            "network": "R34",
            "case": str(case_name),
            "kind": "pair",
            "left": left_metrics,
            "right": right_metrics,
            "mae": float((left_metrics["mae"] + right_metrics["mae"]) / 2.0),
            "max_abs": float(max(left_metrics["max_abs"], right_metrics["max_abs"])),
            "rmse": float((left_metrics["rmse"] + right_metrics["rmse"]) / 2.0),
        }
    finally:
        _cleanup_scheme()


def _validate_r18_single(case_name: str) -> dict[str, Any]:
    spec = dict(R18_BENCH.CASE_SPECS[str(case_name)])
    config = R18_BENCH._build_config()
    _init_scheme(config)
    try:
        _dense_dag, dense_module = _compile_dense_single(R18_BENCH, str(spec["node"]))
        _provider_dag, provider_module = _compile_provider_single(R18_BENCH, str(spec["node"]))
        count = max(int(_source_count_dense(dense_module)), int(_source_count_provider(provider_module)))
        source = _make_source(module=dense_module, count=int(count), seed=int(spec["seed"]))
        _mark_special_source(provider_module, source)
        dense_out = dense_module(source)
        provider_out = provider_module(source)
        dense_decoded = _decode(dense_out)
        provider_decoded = _decode(provider_out)
        return {
            "network": "R18",
            "case": str(case_name),
            "kind": "single",
            **_result_metrics(dense_decoded, provider_decoded),
        }
    finally:
        _cleanup_scheme()


def _validate_case(network: str, case_name: str) -> dict[str, Any]:
    if str(network) == "R34":
        kind = str(R34_BENCH.CASE_SPECS[str(case_name)]["kind"])
        return _validate_r34_pair(str(case_name)) if kind == "pair" else _validate_r34_single(str(case_name))
    return _validate_r18_single(str(case_name))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate decrypted MAE between provider and Orion dense outputs.")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[
            "R34:stage1_same",
            "R34:stage2_same",
            "R34:stage3_same",
            "R34:stage3_transition",
            "R18:stage1_same",
            "R18:stage2_same",
            "R18:stage3_same",
            "R18:stage4_same",
        ],
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    payload = {
        "status": "running",
        "scope": "provider_vs_dense_mae_validation",
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for entry in args.cases:
        network, case_name = str(entry).split(":", 1)
        row = _validate_case(str(network), str(case_name))
        rows.append(row)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"phase": "case_done", "network": network, "case": case_name, **row}), flush=True)
    payload["status"] = "ok"
    payload["timing_s"] = {"total_s": float(time.perf_counter() - started)}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
