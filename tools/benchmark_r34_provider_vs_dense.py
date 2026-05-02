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
from orion.core.fuser import Fuser
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import R34CompileRegistry
from orion.models.resnet import ResNet34
from orion.nn.module import Module


DEFAULT_OUT = Path("/tmp/orion_r34_provider_vs_dense_bench.json")
DEFAULT_CASES = (
    "stem_conv",
    "stem_pool",
    "stage1_same",
    "stage2_transition",
    "stage2_same",
    "stage3_transition",
    "stage3_same",
    "stage4_transition",
    "stage4_same",
    "global_avgpool_exit",
)


CASE_SPECS = {
    "stem_conv": {
        "kind": "single",
        "node": "conv1",
        "seed": 11,
    },
    "stem_pool": {
        "kind": "single",
        "node": "pool",
        "seed": 22,
    },
    "stage1_same": {
        "kind": "single",
        "node": "layers_0_0_conv1",
        "seed": 101,
    },
    "stage2_transition": {
        "kind": "pair",
        "nodes": ("layers_1_0_conv1", "layers_1_0_shortcut_0"),
        "seed": 151,
    },
    "stage2_same": {
        "kind": "single",
        "node": "layers_1_1_conv1",
        "seed": 202,
    },
    "stage3_same": {
        "kind": "single",
        "node": "layers_2_1_conv1",
        "seed": 303,
    },
    "stage3_transition": {
        "kind": "pair",
        "nodes": ("layers_2_0_conv1", "layers_2_0_shortcut_0"),
        "seed": 404,
    },
    "stage4_transition": {
        "kind": "pair",
        "nodes": ("layers_3_0_conv1", "layers_3_0_shortcut_0"),
        "seed": 505,
    },
    "stage4_same": {
        "kind": "single",
        "node": "layers_3_1_conv1",
        "seed": 606,
    },
    "global_avgpool_exit": {
        "kind": "single",
        "node": "avgpool",
        "seed": 707,
    },
}


def _emit_phase(phase: str, **data: Any) -> None:
    print(json.dumps({"phase": str(phase), **data}), flush=True)


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


def _prepare_fused_r34_dag() -> NetworkDAG:
    torch.manual_seed(0)
    net = ResNet34(dataset="imagenet")
    net.eval()
    traced = OrionTracer().trace_model(net)
    StatsTracker(traced).propagate(torch.randn((1, 3, 224, 224), dtype=torch.float32))
    dag = NetworkDAG(traced)
    dag.build_dag()
    for module in net.modules():
        if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
            module.init_orion_params()
    for module in net.modules():
        if hasattr(module, "update_params") and callable(module.update_params):
            module.update_params()
    fuser = Fuser(dag)
    fuser.fuse_modules()
    dag.remove_fused_batchnorms()
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


def _linear_transform_rotation_stats(module) -> dict[str, Any]:
    transform_ids = dict(getattr(module, "transform_ids", {}))
    per_transform: list[dict[str, Any]] = []
    unique_keys: set[int] = set()
    transform_rotation_total = 0
    for (row, col), transform_id in sorted(transform_ids.items()):
        keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
        nonzero_keys = sorted(int(key) for key in keys if int(key) != 0)
        unique_keys.update(nonzero_keys)
        transform_rotation_total += int(len(nonzero_keys))
        per_transform.append(
            {
                "row": int(row),
                "col": int(col),
                "transform_id": int(transform_id),
                "rotation_keys": nonzero_keys,
                "rotation_key_count": int(len(nonzero_keys)),
            }
        )
    cols = _dense_cols(module)
    rows = int(len(transform_ids) // max(1, int(cols)))
    output_rotations = int(getattr(module, "output_rotations", 0))
    output_rotation_evals = int(rows * output_rotations)
    return {
        "source": "lattigo_transform_rotation_keys",
        "transform_count": int(len(transform_ids)),
        "rows": int(rows),
        "cols": int(cols),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": int(output_rotations),
        "output_rotation_eval_count": int(output_rotation_evals),
        "rotation_eval_count_estimate": int(transform_rotation_total + output_rotation_evals),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "per_transform": per_transform,
    }


def _unified_group_rotation_stats(groups: list[Any]) -> dict[str, Any]:
    per_group: list[dict[str, Any]] = []
    unique_keys: set[int] = set()
    transform_rotation_total = 0
    shared_rotation_total = 0
    transform_count = 0
    for group_index, group in enumerate(groups):
        ids = [int(value) for value in (getattr(group, "unified_ids", None) or [])]
        group_keys: set[int] = set()
        per_transform: list[dict[str, Any]] = []
        for transform_index, transform_id in enumerate(ids):
            keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
            nonzero_keys = sorted(int(key) for key in keys if int(key) != 0)
            group_keys.update(nonzero_keys)
            unique_keys.update(nonzero_keys)
            transform_rotation_total += int(len(nonzero_keys))
            transform_count += 1
            per_transform.append(
                {
                    "transform_index": int(transform_index),
                    "transform_id": int(transform_id),
                    "rotation_keys": nonzero_keys,
                    "rotation_key_count": int(len(nonzero_keys)),
                }
            )
        shared_rotation_total += int(len(group_keys))
        per_group.append(
            {
                "group_index": int(group_index),
                "transform_count": int(len(ids)),
                "rotation_key_count_total": int(sum(int(item["rotation_key_count"]) for item in per_transform)),
                "shared_rotation_eval_count": int(len(group_keys)),
                "unique_rotation_key_count": int(len(group_keys)),
                "unique_rotation_keys": sorted(int(key) for key in group_keys),
                "per_transform": per_transform,
            }
        )
    return {
        "source": "lattigo_unified_transform_rotation_keys",
        "group_count": int(len(groups)),
        "transform_count": int(transform_count),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "shared_rotation_eval_count_total": int(shared_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": 0,
        "output_rotation_eval_count": 0,
        "rotation_eval_count_estimate": int(shared_rotation_total),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "per_group": per_group,
    }


def _provider_rotation_stats(executor: Any) -> dict[str, Any]:
    if getattr(executor, "group", None) is not None:
        return _unified_group_rotation_stats([executor.group])
    groups = list(getattr(executor, "groups_by_input_block", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups = list(getattr(executor, "groups_by_input", []) or [])
    if groups:
        return _unified_group_rotation_stats(groups)
    groups_by_input_index = getattr(executor, "groups_by_input_index", None) or {}
    if groups_by_input_index:
        return _unified_group_rotation_stats([group for _input_index, group in sorted(groups_by_input_index.items())])
    transform_ids = dict(getattr(executor, "transform_ids", {}) or {})
    if transform_ids:
        unique_keys: set[int] = set()
        per_transform: list[dict[str, Any]] = []
        transform_rotation_total = 0
        cols = 0 if not transform_ids else max(int(col) for _row, col in transform_ids) + 1
        rows = 0 if not transform_ids else max(int(row) for row, _col in transform_ids) + 1
        for (row, col), transform_id in sorted(transform_ids.items()):
            keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
            nonzero_keys = sorted(int(key) for key in keys if int(key) != 0)
            unique_keys.update(nonzero_keys)
            transform_rotation_total += int(len(nonzero_keys))
            per_transform.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "transform_id": int(transform_id),
                    "rotation_keys": nonzero_keys,
                    "rotation_key_count": int(len(nonzero_keys)),
                }
            )
        output_rotations = int(getattr(executor, "output_rotations", 0))
        output_rotation_evals = int(rows * output_rotations)
        return {
            "source": "lattigo_executor_transform_rotation_keys",
            "transform_count": int(len(transform_ids)),
            "rows": int(rows),
            "cols": int(cols),
            "transform_rotation_key_count_total": int(transform_rotation_total),
            "unique_rotation_key_count": int(len(unique_keys)),
            "output_rotations_per_output_ct": int(output_rotations),
            "output_rotation_eval_count": int(output_rotation_evals),
            "rotation_eval_count_estimate": int(transform_rotation_total + output_rotation_evals),
            "unique_rotation_keys": sorted(int(key) for key in unique_keys),
            "per_transform": per_transform,
        }
    return {
        "source": "no_lattigo_unified_transform_ids",
        "group_count": 0,
        "transform_count": 0,
        "transform_rotation_key_count_total": 0,
        "unique_rotation_key_count": 0,
        "output_rotations_per_output_ct": 0,
        "output_rotation_eval_count": 0,
        "rotation_eval_count_estimate": 0,
        "unique_rotation_keys": [],
    }


def _bench_dense_single(node_name: str, *, repeats: int, seed: int) -> dict[str, Any]:
    dag = _prepare_fused_r34_dag()
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
    if int(repeats) <= 0:
        return {
            "path": "dense",
            "node": str(node_name),
            "compile_s": float(compile_s),
            "generate_diagonals_s": float(generate_diagonals_s),
            "compile_backend_s": float(compile_backend_s),
            "warmup_s": 0.0,
            "rotation_stats": _linear_transform_rotation_stats(module),
            "source_ciphertext_count": int(cols),
            "output_ciphertext_count": int(len(getattr(module, "transform_ids", {})) // max(1, cols)),
            "hot_run_s": [],
            "hot_run_median_s": None,
            "compile_only": True,
        }
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
        "rotation_stats": _linear_transform_rotation_stats(module),
        "source_ciphertext_count": int(cols),
        "output_ciphertext_count": int(len(getattr(module, "transform_ids", {})) // max(1, cols)),
        "hot_run_s": run_times,
        "hot_run_median_s": float(statistics.median(run_times)),
    }


def _bench_provider_single(node_name: str, *, repeats: int, seed: int) -> dict[str, Any]:
    dag = _prepare_fused_r34_dag()
    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
    attach = registry.attach_to_dag(dag)
    module = dag.nodes[str(node_name)]["module"]
    runtime = module.region_runtime
    runtime_supported = bool(getattr(runtime, "supports_scheme", lambda _scheme: True)(scheme))
    if not bool(runtime_supported):
        return {
            "status": "unsupported",
            "path": "provider",
            "node": str(node_name),
            "compile_s": None,
            "generate_diagonals_s": None,
            "compile_backend_s": None,
            "warmup_s": None,
            "attach_audit": attach,
            "runtime_supported": False,
            "executor": type(getattr(runtime, "executor", None)).__name__,
            "reason": "provider executor does not support the active Lattigo scheme",
        }
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
    executor = runtime.executor
    if int(repeats) <= 0:
        return {
            "status": "ok",
            "path": "provider",
            "node": str(node_name),
            "compile_s": float(compile_s),
            "generate_diagonals_s": float(generate_diagonals_s),
            "compile_backend_s": float(compile_backend_s),
            "warmup_s": 0.0,
            "attach_audit": attach,
            "runtime_supported": True,
            "executor": type(executor).__name__,
            "rotation_stats": _provider_rotation_stats(executor),
            "source_ciphertext_count": int(executor.cols),
            "output_ciphertext_count": int(executor.rows),
            "hot_run_s": [],
            "hot_run_median_s": None,
            "executor_last_runtime_timing": dict(executor.last_runtime_timing),
            "executor_hot_runtime_timings": [],
            "compile_only": True,
        }
    warm_source = _prebuilt_sources(module, count=int(executor.cols), repeats=1, seed=int(seed))[0]
    warm_started = time.perf_counter()
    _ = module(warm_source)
    warmup_s = float(time.perf_counter() - warm_started)
    _emit_phase("provider_single_warmup_done", node=str(node_name), seconds=float(warmup_s))
    run_sources = _prebuilt_sources(module, count=int(executor.cols), repeats=int(repeats), seed=int(seed + 1))
    run_times: list[float] = []
    executor_timings: list[dict[str, float]] = []
    for source in run_sources:
        started = time.perf_counter()
        out = module(source)
        run_times.append(float(time.perf_counter() - started))
        executor_timings.append(dict(executor.last_runtime_timing))
        del out
    return {
        "status": "ok",
        "path": "provider",
        "node": str(node_name),
        "compile_s": float(compile_s),
        "generate_diagonals_s": float(generate_diagonals_s),
        "compile_backend_s": float(compile_backend_s),
        "warmup_s": float(warmup_s),
        "attach_audit": attach,
        "runtime_supported": True,
        "executor": type(executor).__name__,
        "rotation_stats": _provider_rotation_stats(executor),
        "source_ciphertext_count": int(executor.cols),
        "output_ciphertext_count": int(executor.rows),
        "hot_run_s": run_times,
        "hot_run_median_s": float(statistics.median(run_times)),
        "executor_last_runtime_timing": dict(executor.last_runtime_timing),
        "executor_hot_runtime_timings": executor_timings,
    }


def _bench_dense_pair(node_names: tuple[str, str], *, repeats: int, seed: int) -> dict[str, Any]:
    dag = _prepare_fused_r34_dag()
    left = dag.nodes[str(node_names[0])]["module"]
    right = dag.nodes[str(node_names[1])]["module"]
    _set_compile_level(dag)
    generate_started = time.perf_counter()
    for module in (left, right):
        module.generate_diagonals(last=False)
    generate_diagonals_s = float(time.perf_counter() - generate_started)
    _emit_phase("dense_pair_generate_done", nodes=list(node_names), seconds=float(generate_diagonals_s))
    compile_started = time.perf_counter()
    for module in (left, right):
        module.compile()
        module.he_mode = True
    compile_backend_s = float(time.perf_counter() - compile_started)
    _emit_phase("dense_pair_compile_done", nodes=list(node_names), seconds=float(compile_backend_s))
    compile_s = float(generate_diagonals_s + compile_backend_s)
    cols = _dense_cols(left)
    if int(repeats) <= 0:
        left_rotation_stats = _linear_transform_rotation_stats(left)
        right_rotation_stats = _linear_transform_rotation_stats(right)
        return {
            "path": "dense_pair",
            "nodes": list(node_names),
            "compile_s": float(compile_s),
            "generate_diagonals_s": float(generate_diagonals_s),
            "compile_backend_s": float(compile_backend_s),
            "warmup_s": 0.0,
            "rotation_stats": {
                "source": "lattigo_transform_rotation_keys_pair_sum",
                "left": left_rotation_stats,
                "right": right_rotation_stats,
                "rotation_eval_count_estimate": int(
                    left_rotation_stats["rotation_eval_count_estimate"]
                    + right_rotation_stats["rotation_eval_count_estimate"]
                ),
            },
            "source_ciphertext_count": int(cols),
            "left_output_ciphertext_count": int(len(left.transform_ids) // max(1, cols)),
            "right_output_ciphertext_count": int(len(right.transform_ids) // max(1, cols)),
            "hot_run_s": [],
            "hot_run_median_s": None,
            "compile_only": True,
        }
    warm_source = _prebuilt_sources(left, count=int(cols), repeats=1, seed=int(seed))[0]
    warm_started = time.perf_counter()
    _ = left(warm_source)
    _ = right(warm_source)
    warmup_s = float(time.perf_counter() - warm_started)
    _emit_phase("dense_pair_warmup_done", nodes=list(node_names), seconds=float(warmup_s))
    run_sources = _prebuilt_sources(left, count=int(cols), repeats=int(repeats), seed=int(seed + 1))
    run_times: list[float] = []
    for source in run_sources:
        started = time.perf_counter()
        out_left = left(source)
        out_right = right(source)
        run_times.append(float(time.perf_counter() - started))
        del out_left, out_right
    left_rotation_stats = _linear_transform_rotation_stats(left)
    right_rotation_stats = _linear_transform_rotation_stats(right)
    return {
        "path": "dense_pair",
        "nodes": list(node_names),
        "compile_s": float(compile_s),
        "generate_diagonals_s": float(generate_diagonals_s),
        "compile_backend_s": float(compile_backend_s),
        "warmup_s": float(warmup_s),
        "rotation_stats": {
            "source": "lattigo_transform_rotation_keys_pair_sum",
            "left": left_rotation_stats,
            "right": right_rotation_stats,
            "rotation_eval_count_estimate": int(
                left_rotation_stats["rotation_eval_count_estimate"]
                + right_rotation_stats["rotation_eval_count_estimate"]
            ),
        },
        "source_ciphertext_count": int(cols),
        "left_output_ciphertext_count": int(len(left.transform_ids) // max(1, cols)),
        "right_output_ciphertext_count": int(len(right.transform_ids) // max(1, cols)),
        "hot_run_s": run_times,
        "hot_run_median_s": float(statistics.median(run_times)),
    }


def _bench_provider_pair(node_names: tuple[str, str], *, repeats: int, seed: int) -> dict[str, Any]:
    dag = _prepare_fused_r34_dag()
    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
    attach = registry.attach_to_dag(dag)
    left = dag.nodes[str(node_names[0])]["module"]
    right = dag.nodes[str(node_names[1])]["module"]
    runtime = left.region_runtime
    executor = runtime.executor
    runtime_supported = bool(getattr(runtime, "supports_scheme", lambda _scheme: True)(scheme))
    if not bool(runtime_supported):
        return {
            "status": "unsupported",
            "path": "provider_pair",
            "nodes": list(node_names),
            "compile_s": None,
            "generate_diagonals_s": None,
            "compile_backend_s": None,
            "warmup_s": None,
            "attach_audit": attach,
            "runtime_supported": False,
            "executor": type(executor).__name__,
            "reason": "provider executor does not support the active Lattigo scheme",
        }
    _set_compile_level(dag)
    generate_started = time.perf_counter()
    left.generate_diagonals(last=False)
    right.generate_diagonals(last=False)
    generate_diagonals_s = float(time.perf_counter() - generate_started)
    _emit_phase("provider_pair_generate_done", nodes=list(node_names), seconds=float(generate_diagonals_s))
    compile_started = time.perf_counter()
    left.compile()
    right.compile()
    compile_backend_s = float(time.perf_counter() - compile_started)
    _emit_phase("provider_pair_compile_done", nodes=list(node_names), seconds=float(compile_backend_s))
    compile_s = float(generate_diagonals_s + compile_backend_s)
    left.he_mode = True
    right.he_mode = True
    if int(repeats) <= 0:
        return {
            "status": "ok",
            "path": "provider_pair",
            "nodes": list(node_names),
            "compile_s": float(compile_s),
            "generate_diagonals_s": float(generate_diagonals_s),
            "compile_backend_s": float(compile_backend_s),
            "warmup_s": 0.0,
            "attach_audit": attach,
            "runtime_supported": True,
            "executor": type(executor).__name__,
            "rotation_stats": _provider_rotation_stats(executor),
            "source_ciphertext_count": int(executor.cols),
            "output_ciphertext_count_per_branch": int(executor.rows),
            "hot_run_s": [],
            "hot_run_median_s": None,
            "executor_last_runtime_timing": dict(executor.last_runtime_timing),
            "executor_hot_runtime_timings": [],
            "compile_only": True,
        }
    warm_source = _prebuilt_sources(left, count=int(executor.cols), repeats=1, seed=int(seed))[0]
    warm_started = time.perf_counter()
    _ = left(warm_source)
    _ = right(warm_source)
    warmup_s = float(time.perf_counter() - warm_started)
    _emit_phase("provider_pair_warmup_done", nodes=list(node_names), seconds=float(warmup_s))
    run_sources = _prebuilt_sources(left, count=int(executor.cols), repeats=int(repeats), seed=int(seed + 1))
    run_times: list[float] = []
    executor_timings: list[dict[str, float]] = []
    for source in run_sources:
        started = time.perf_counter()
        out_left = left(source)
        out_right = right(source)
        run_times.append(float(time.perf_counter() - started))
        executor_timings.append(dict(executor.last_runtime_timing))
        del out_left, out_right
    return {
        "status": "ok",
        "path": "provider_pair",
        "nodes": list(node_names),
        "compile_s": float(compile_s),
        "generate_diagonals_s": float(generate_diagonals_s),
        "compile_backend_s": float(compile_backend_s),
        "warmup_s": float(warmup_s),
        "attach_audit": attach,
        "runtime_supported": True,
        "executor": type(executor).__name__,
        "rotation_stats": _provider_rotation_stats(executor),
        "source_ciphertext_count": int(executor.cols),
        "output_ciphertext_count_per_branch": int(executor.rows),
        "hot_run_s": run_times,
        "hot_run_median_s": float(statistics.median(run_times)),
        "executor_last_runtime_timing": dict(executor.last_runtime_timing),
        "executor_hot_runtime_timings": executor_timings,
    }


def _bench_case(case_name: str, *, repeats: int) -> dict[str, Any]:
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

    if str(spec["kind"]) == "single":
        dense = _run_once(lambda: _bench_dense_single(str(spec["node"]), repeats=int(repeats), seed=int(spec["seed"])))
        provider = _run_once(lambda: _bench_provider_single(str(spec["node"]), repeats=int(repeats), seed=int(spec["seed"])))
    else:
        dense = _run_once(
            lambda: _bench_dense_pair(tuple(str(v) for v in spec["nodes"]), repeats=int(repeats), seed=int(spec["seed"]))
        )
        provider = _run_once(
            lambda: _bench_provider_pair(tuple(str(v) for v in spec["nodes"]), repeats=int(repeats), seed=int(spec["seed"]))
        )
    dense_hot = dense.get("hot_run_median_s")
    provider_hot = provider.get("hot_run_median_s")
    return {
        "case": str(case_name),
        "kind": str(spec["kind"]),
        "dense": dense,
        "provider": provider,
        "delta": {
            "compile_s": float(provider["compile_s"] - dense["compile_s"]),
            "hot_run_median_s": None
            if dense_hot is None or provider_hot is None
            else float(float(provider_hot) - float(dense_hot)),
        },
        "speedup": {
            "compile_ratio_dense_over_provider": None if float(provider["compile_s"]) == 0.0 else float(dense["compile_s"] / provider["compile_s"]),
            "hot_run_ratio_dense_over_provider": None
            if dense_hot is None or provider_hot is None or float(provider_hot) == 0.0
            else float(float(dense_hot) / float(provider_hot)),
        },
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

    if str(spec["kind"]) == "single":
        if str(path_kind) == "dense":
            result = _run_once(lambda: _bench_dense_single(str(spec["node"]), repeats=int(repeats), seed=int(spec["seed"])))
        else:
            result = _run_once(lambda: _bench_provider_single(str(spec["node"]), repeats=int(repeats), seed=int(spec["seed"])))
    else:
        if str(path_kind) == "dense":
            result = _run_once(
                lambda: _bench_dense_pair(tuple(str(v) for v in spec["nodes"]), repeats=int(repeats), seed=int(spec["seed"]))
            )
        else:
            result = _run_once(
                lambda: _bench_provider_pair(tuple(str(v) for v in spec["nodes"]), repeats=int(repeats), seed=int(spec["seed"]))
            )
    return result


def _worker_main(case_name: str, *, path_kind: str, repeats: int) -> int:
    payload = _bench_case_path(str(case_name), path_kind=str(path_kind), repeats=int(repeats))
    print(json.dumps(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark R34 provider kernels against Orion dense conv on representative shapes.")
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-case", default="")
    parser.add_argument("--worker-path", choices=("dense", "provider"), default="")
    args = parser.parse_args()

    if str(args.worker_case):
        return _worker_main(str(args.worker_case), path_kind=str(args.worker_path), repeats=int(args.repeats))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if bool(args.resume) and Path(args.out).exists():
        payload = json.loads(Path(args.out).read_text(encoding="utf-8"))
        payload["status"] = "started"
        payload.setdefault("scope", "R34 representative conv compile/runtime comparison: provider bridge vs Orion dense")
        payload.setdefault("config", _build_config())
        payload.setdefault("cases", [])
    else:
        payload = {
            "status": "started",
            "scope": "R34 representative conv compile/runtime comparison: provider bridge vs Orion dense",
            "config": _build_config(),
            "cases": [],
        }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    cases_by_name = {str(case_payload.get("case")): case_payload for case_payload in payload["cases"]}
    for case_name in args.cases:
        case_payload = cases_by_name.get(str(case_name))
        if case_payload is None:
            case_payload = {
                "case": str(case_name),
                "status": "running",
                "paths": {},
            }
            payload["cases"].append(case_payload)
            cases_by_name[str(case_name)] = case_payload
        case_payload["status"] = "running"
        case_payload.setdefault("paths", {})
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        for path_kind in ("dense", "provider"):
            existing_path = dict(case_payload["paths"].get(str(path_kind), {}))
            if bool(args.resume) and existing_path.get("status") == "ok":
                continue
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
        if (
            dense_entry.get("status") == "ok"
            and provider_entry.get("status") == "ok"
            and dict(provider_entry.get("result", {})).get("status", "ok") == "ok"
        ):
            dense = dict(dense_entry["result"])
            provider = dict(provider_entry["result"])
            dense_hot = dense.get("hot_run_median_s")
            provider_hot = provider.get("hot_run_median_s")
            case_payload.update(
                {
                    "status": "ok",
                    "kind": str(CASE_SPECS[str(case_name)]["kind"]),
                    "dense": dense,
                    "provider": provider,
                    "delta": {
                        "compile_s": float(provider["compile_s"] - dense["compile_s"]),
                        "hot_run_median_s": None
                        if dense_hot is None or provider_hot is None
                        else float(float(provider_hot) - float(dense_hot)),
                    },
                    "speedup": {
                        "compile_ratio_dense_over_provider": None if float(provider["compile_s"]) == 0.0 else float(dense["compile_s"] / provider["compile_s"]),
                        "hot_run_ratio_dense_over_provider": None
                        if dense_hot is None or provider_hot is None or float(provider_hot) == 0.0
                        else float(float(dense_hot) / float(provider_hot)),
                    },
                }
            )
        elif dense_entry.get("status") == "ok" and provider_entry.get("status") == "ok":
            dense = dict(dense_entry["result"])
            provider = dict(provider_entry["result"])
            case_payload.update(
                {
                    "status": str(provider.get("status", "partial")),
                    "kind": str(CASE_SPECS[str(case_name)]["kind"]),
                    "dense": dense,
                    "provider": provider,
                    "delta": None,
                    "speedup": None,
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
