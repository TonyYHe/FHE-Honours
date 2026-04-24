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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.backend.python.tensors import CipherTensor
from orion.core.auto_bootstrap import BootstrapSolver
from orion.core.fuser import Fuser
from orion.core.network_dag import NetworkDAG
from orion.core.orion import scheme
from orion.core.region_lowering import pack_chw_gap
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental.cir.runtime_group import RegionFirstCompileRegistry
from orion.models.resnet import ResNet18
from orion.nn.module import Module


DEFAULT_OUT = Path("/tmp/orion_r18_unique_layers_e2e_bench.json")
DEFAULT_CONFIG = REPO_ROOT / "configs" / "resnet.yml"
DEFAULT_CASES = (
    "stem_bridge",
    "stage1_same",
    "stage1_transition",
    "stage2_same",
    "stage2_transition",
    "stage3_same",
    "stage3_transition",
    "stage4_same",
)


CASE_SPECS: dict[str, dict[str, Any]] = {
    "stem_bridge": {
        "kind": "stem",
        "node": "conv1",
        "seed": 11,
        "multiplicity": 1,
    },
    "stage1_same": {
        "kind": "single",
        "node": "layers_0_0_conv1",
        "seed": 101,
        "multiplicity": 4,
    },
    "stage1_transition": {
        "kind": "transition",
        "nodes": ("layers_1_0_conv1", "layers_1_0_shortcut_0"),
        "seed": 111,
        "multiplicity": 1,
    },
    "stage2_same": {
        "kind": "single",
        "node": "layers_1_1_conv1",
        "seed": 202,
        "multiplicity": 3,
    },
    "stage2_transition": {
        "kind": "transition",
        "nodes": ("layers_2_0_conv1", "layers_2_0_shortcut_0"),
        "seed": 222,
        "multiplicity": 1,
    },
    "stage3_same": {
        "kind": "single",
        "node": "layers_2_1_conv1",
        "seed": 303,
        "multiplicity": 3,
    },
    "stage3_transition": {
        "kind": "transition",
        "nodes": ("layers_3_0_conv1", "layers_3_0_shortcut_0"),
        "seed": 333,
        "multiplicity": 1,
    },
    "stage4_same": {
        "kind": "single",
        "node": "layers_3_0_conv2",
        "seed": 404,
        "multiplicity": 3,
    },
}


def _load_config(config_path: Path, *, provider: bool) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = dict(config)
    config["ckks_params"] = dict(config.get("ckks_params", {}))
    config["orion"] = dict(config.get("orion", {}))
    config["boot_params"] = dict(config.get("boot_params", {}))
    config["orion"]["backend"] = "lattigo"
    config["orion"]["io_mode"] = "none"
    config["orion"]["experimental_region_first"] = "r18_tiny_e2e" if bool(provider) else ""
    return config


def _emit_phase(phase: str, **data: Any) -> None:
    print(json.dumps({"phase": str(phase), **data}), flush=True)


def _prepare_fused_r18_tiny_dag(*, provider: bool) -> tuple[ResNet18, NetworkDAG, dict[str, Any]]:
    torch.manual_seed(0)
    x = torch.randn((1, 3, 64, 64), dtype=torch.float32)
    net = ResNet18(dataset="tiny")
    net.eval()
    net.set_scheme(scheme)
    net.set_margin(scheme.params.get_margin())
    traced = OrionTracer().trace_model(net)
    StatsTracker(traced).propagate(x)
    for module in net.modules():
        if hasattr(module, "fit") and callable(module.fit):
            module.fit()
        if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
            module.init_orion_params()
        if hasattr(module, "update_params") and callable(module.update_params):
            module.update_params()
    dag = NetworkDAG(traced)
    dag.build_dag()
    fuser = Fuser(dag)
    fuser.fuse_modules()
    dag.remove_fused_batchnorms()
    attach_audit: dict[str, Any] = {}
    if bool(provider):
        registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
        attach_audit = registry.attach_to_dag(dag)
    dag.find_residuals()
    solver = BootstrapSolver(net, dag, l_eff=len(scheme.params.get_logq()) - 1)
    input_level, num_bootstraps, bootstrapper_slots = solver.solve()
    solver_summary = {
        "input_level": int(input_level),
        "num_bootstraps": int(num_bootstraps),
        "bootstrapper_slots": [int(value) for value in bootstrapper_slots],
    }
    return net, dag, {"attach_audit": attach_audit, "solver": solver_summary}


def _plain_blocks(*, count: int, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(int(seed))
    return [
        torch.randn((scheme.params.get_slots(),), generator=gen, dtype=torch.float32) * 0.01
        for _ in range(int(count))
    ]


def _stem_plain_blocks(*, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(int(seed))
    image = torch.randn((3, 64, 64), generator=gen, dtype=torch.float32)
    packed = pack_chw_gap(image, shape=(3, 64, 64), gap=1, slots=32768).to(dtype=torch.float32)
    return [packed]


def _encrypt_source(
    *,
    module: Any,
    plain_blocks: list[torch.Tensor],
    input_level: int,
) -> CipherTensor:
    ids: list[int] = []
    for block in plain_blocks:
        ct = scheme.encrypt(scheme.encode(block.to(dtype=torch.float32), int(input_level)))
        ids.append(int(ct.ids[0]))
        ct.ids = []
    return CipherTensor(scheme, ids, module.input_shape, module.fhe_input_shape)


def _dense_cols(module: Any) -> int:
    keys = list(getattr(module, "transform_ids", {}).keys())
    if not keys:
        return 0
    return max(int(col) for _row, col in keys) + 1


def _provider_cols(module: Any) -> int:
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None)
    if executor is None:
        return 0
    if hasattr(executor, "cols"):
        return int(executor.cols)
    if hasattr(executor, "source_pair_count"):
        if bool(getattr(executor, "requires_compact_source", False)):
            return 1
        return int(executor.source_pair_count) * 2
    return 1


def _decode(ct: CipherTensor) -> torch.Tensor:
    return ct.decrypt().decode().detach().cpu().to(dtype=torch.float32)


def _result_metrics(dense: torch.Tensor, provider: torch.Tensor) -> dict[str, float]:
    diff = provider - dense
    return {
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.pow(2))).item()),
    }


def _single_path_metrics(
    *,
    case_name: str,
    node_name: str,
    provider: bool,
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    _net, dag, setup = _prepare_fused_r18_tiny_dag(provider=bool(provider))
    module = dag.nodes[str(node_name)]["module"]

    generate_started = time.perf_counter()
    module.generate_diagonals(last=False)
    generate_diagonals_s = float(time.perf_counter() - generate_started)
    _emit_phase(
        "generate_done",
        case=str(case_name),
        path="provider" if provider else "dense",
        node=str(node_name),
        seconds=float(generate_diagonals_s),
    )

    compile_started = time.perf_counter()
    module.compile()
    compile_backend_s = float(time.perf_counter() - compile_started)
    _emit_phase(
        "compile_done",
        case=str(case_name),
        path="provider" if provider else "dense",
        node=str(node_name),
        seconds=float(compile_backend_s),
    )

    module.he_mode = True
    source_ciphertext_count = int(_provider_cols(module) if provider else _dense_cols(module))
    block_builder = _stem_plain_blocks if str(case_name) == "stem_bridge" else _plain_blocks
    compare_plain = block_builder(count=int(source_ciphertext_count), seed=int(seed)) if block_builder is _plain_blocks else block_builder(seed=int(seed))
    compare_source = _encrypt_source(
        module=module,
        plain_blocks=compare_plain,
        input_level=int(module.level),
    )
    compare_out = module(compare_source)
    compare_decoded = _decode(compare_out)

    warm_source = _encrypt_source(
        module=module,
        plain_blocks=block_builder(count=int(source_ciphertext_count), seed=int(seed + 1)) if block_builder is _plain_blocks else block_builder(seed=int(seed + 1)),
        input_level=int(module.level),
    )
    warm_started = time.perf_counter()
    warm_out = module(warm_source)
    warmup_s = float(time.perf_counter() - warm_started)
    del warm_out

    run_times: list[float] = []
    for repeat_index in range(int(repeats)):
        source = _encrypt_source(
            module=module,
            plain_blocks=block_builder(count=int(source_ciphertext_count), seed=int(seed + 10 + repeat_index)) if block_builder is _plain_blocks else block_builder(seed=int(seed + 10 + repeat_index)),
            input_level=int(module.level),
        )
        started = time.perf_counter()
        out = module(source)
        run_times.append(float(time.perf_counter() - started))
        del out

    executor = getattr(getattr(module, "region_runtime", None), "executor", None)
    return (
        {
            "path": "provider" if provider else "dense",
            "node": str(node_name),
            "input_level": int(module.level),
            "depth": int(getattr(module, "depth", 0) or 0),
            "compile_s": float(generate_diagonals_s + compile_backend_s),
            "generate_diagonals_s": float(generate_diagonals_s),
            "compile_backend_s": float(compile_backend_s),
            "warmup_s": float(warmup_s),
            "hot_run_s": [float(value) for value in run_times],
            "hot_run_median_s": float(statistics.median(run_times)),
            "source_ciphertext_count": int(source_ciphertext_count),
            "output_ciphertext_count": int(len(compare_out.ids)),
            "attach_audit": setup.get("attach_audit", {}) if provider else {},
            "solver": dict(setup["solver"]),
            "executor_last_runtime_timing": dict(getattr(executor, "last_runtime_timing", {})) if executor is not None else {},
        },
        compare_decoded,
    )


def _transition_dense_metrics(
    *,
    case_name: str,
    node_names: tuple[str, str],
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    _net, dag, setup = _prepare_fused_r18_tiny_dag(provider=False)
    left = dag.nodes[str(node_names[0])]["module"]
    right = dag.nodes[str(node_names[1])]["module"]

    generate_started = time.perf_counter()
    left.generate_diagonals(last=False)
    right.generate_diagonals(last=False)
    generate_diagonals_s = float(time.perf_counter() - generate_started)

    compile_started = time.perf_counter()
    left.compile()
    right.compile()
    compile_backend_s = float(time.perf_counter() - compile_started)

    left.he_mode = True
    right.he_mode = True
    source_ciphertext_count = int(max(_dense_cols(left), _dense_cols(right)))
    compare_plain = _plain_blocks(count=int(source_ciphertext_count), seed=int(seed))
    compare_source = _encrypt_source(module=left, plain_blocks=compare_plain, input_level=int(left.level))
    left_compare = left(compare_source)
    right_compare = right(compare_source)

    warm_source = _encrypt_source(
        module=left,
        plain_blocks=_plain_blocks(count=int(source_ciphertext_count), seed=int(seed + 1)),
        input_level=int(left.level),
    )
    warm_started = time.perf_counter()
    left_warm = left(warm_source)
    right_warm = right(warm_source)
    warmup_s = float(time.perf_counter() - warm_started)
    del left_warm, right_warm

    run_times: list[float] = []
    for repeat_index in range(int(repeats)):
        source = _encrypt_source(
            module=left,
            plain_blocks=_plain_blocks(count=int(source_ciphertext_count), seed=int(seed + 10 + repeat_index)),
            input_level=int(left.level),
        )
        started = time.perf_counter()
        out_left = left(source)
        out_right = right(source)
        run_times.append(float(time.perf_counter() - started))
        del out_left, out_right

    return (
        {
            "path": "dense",
            "nodes": [str(node_names[0]), str(node_names[1])],
            "input_level": int(left.level),
            "depths": {str(node_names[0]): int(getattr(left, "depth", 0) or 0), str(node_names[1]): int(getattr(right, "depth", 0) or 0)},
            "compile_s": float(generate_diagonals_s + compile_backend_s),
            "generate_diagonals_s": float(generate_diagonals_s),
            "compile_backend_s": float(compile_backend_s),
            "warmup_s": float(warmup_s),
            "hot_run_s": [float(value) for value in run_times],
            "hot_run_median_s": float(statistics.median(run_times)),
            "source_ciphertext_count": int(source_ciphertext_count),
            "output_ciphertext_count_per_branch": {
                str(node_names[0]): int(len(left_compare.ids)),
                str(node_names[1]): int(len(right_compare.ids)),
            },
            "solver": dict(setup["solver"]),
        },
        {
            str(node_names[0]): _decode(left_compare),
            str(node_names[1]): _decode(right_compare),
        },
    )


def _transition_provider_metrics(
    *,
    case_name: str,
    node_names: tuple[str, str],
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    _net, dag, setup = _prepare_fused_r18_tiny_dag(provider=True)
    left = dag.nodes[str(node_names[0])]["module"]
    right = dag.nodes[str(node_names[1])]["module"]
    runtime = left.region_runtime
    executor = runtime.executor

    generate_started = time.perf_counter()
    left.generate_diagonals(last=False)
    right.generate_diagonals(last=False)
    generate_diagonals_s = float(time.perf_counter() - generate_started)

    compile_started = time.perf_counter()
    left.compile()
    right.compile()
    compile_backend_s = float(time.perf_counter() - compile_started)

    left.he_mode = True
    right.he_mode = True
    source_ciphertext_count = int(_provider_cols(left))
    compare_plain = _plain_blocks(count=int(source_ciphertext_count), seed=int(seed))
    compare_source = _encrypt_source(module=left, plain_blocks=compare_plain, input_level=int(left.level))
    compare_outputs = runtime.execute(compare_source)

    warm_source = _encrypt_source(
        module=left,
        plain_blocks=_plain_blocks(count=int(source_ciphertext_count), seed=int(seed + 1)),
        input_level=int(left.level),
    )
    warm_started = time.perf_counter()
    warm_outputs = runtime.execute(warm_source)
    warmup_s = float(time.perf_counter() - warm_started)
    del warm_outputs

    run_times: list[float] = []
    executor_timings: list[dict[str, float]] = []
    for repeat_index in range(int(repeats)):
        source = _encrypt_source(
            module=left,
            plain_blocks=_plain_blocks(count=int(source_ciphertext_count), seed=int(seed + 10 + repeat_index)),
            input_level=int(left.level),
        )
        started = time.perf_counter()
        outputs = runtime.execute(source)
        run_times.append(float(time.perf_counter() - started))
        executor_timings.append(dict(getattr(executor, "last_runtime_timing", {})))
        del outputs

    return (
        {
            "path": "provider",
            "nodes": [str(node_names[0]), str(node_names[1])],
            "input_level": int(left.level),
            "depths": {str(node_names[0]): int(getattr(left, "depth", 0) or 0), str(node_names[1]): int(getattr(right, "depth", 0) or 0)},
            "compile_s": float(generate_diagonals_s + compile_backend_s),
            "generate_diagonals_s": float(generate_diagonals_s),
            "compile_backend_s": float(compile_backend_s),
            "warmup_s": float(warmup_s),
            "hot_run_s": [float(value) for value in run_times],
            "hot_run_median_s": float(statistics.median(run_times)),
            "source_ciphertext_count": int(source_ciphertext_count),
            "output_ciphertext_count_per_branch": {
                str(node_names[0]): int(len(compare_outputs[str(node_names[0])].ids)),
                str(node_names[1]): int(len(compare_outputs[str(node_names[1])].ids)),
            },
            "attach_audit": setup.get("attach_audit", {}),
            "solver": dict(setup["solver"]),
            "executor_last_runtime_timing": dict(getattr(executor, "last_runtime_timing", {})),
            "executor_hot_runtime_timings": executor_timings,
        },
        {
            str(node_names[0]): _decode(compare_outputs[str(node_names[0])]),
            str(node_names[1]): _decode(compare_outputs[str(node_names[1])]),
        },
    )


def _run_case_path(case_name: str, *, repeats: int, provider: bool) -> tuple[dict[str, Any], Any]:
    spec = dict(CASE_SPECS[str(case_name)])
    kind = str(spec["kind"])
    if kind in {"single", "stem"}:
        return _single_path_metrics(
            case_name=str(case_name),
            node_name=str(spec["node"]),
            provider=bool(provider),
            repeats=int(repeats),
            seed=int(spec["seed"]),
        )
    if kind == "transition":
        if bool(provider):
            return _transition_provider_metrics(
                case_name=str(case_name),
                node_names=tuple(spec["nodes"]),
                repeats=int(repeats),
                seed=int(spec["seed"]),
            )
        return _transition_dense_metrics(
            case_name=str(case_name),
            node_names=tuple(spec["nodes"]),
            repeats=int(repeats),
            seed=int(spec["seed"]),
        )
    raise ValueError(f"unsupported case kind {kind!r}")


def _cleanup_scheme() -> None:
    try:
        scheme.delete_scheme()
    except Exception:
        pass
    gc.collect()


def _worker_main(case_name: str, *, repeats: int, config_path: Path) -> int:
    dense_config = _load_config(Path(config_path), provider=False)
    provider_config = _load_config(Path(config_path), provider=True)

    def _run_once(provider: bool):
        config = provider_config if provider else dense_config
        scheme.init_scheme(config)
        try:
            Module.set_scheme(scheme)
            Module.set_margin(scheme.params.get_margin())
            return _run_case_path(str(case_name), repeats=int(repeats), provider=bool(provider))
        finally:
            _cleanup_scheme()

    dense_payload, dense_output = _run_once(False)
    provider_payload, provider_output = _run_once(True)
    spec = dict(CASE_SPECS[str(case_name)])
    _ = dense_output, provider_output
    correctness = {
        "status": "not_measured_in_timing_harness",
        "reason": "Dense/provider timing runs are executed under separate scheme instances; correctness must be validated in a dedicated shared-scheme or Python-backend pass.",
    }

    dense_compile = float(dense_payload["compile_s"])
    provider_compile = float(provider_payload["compile_s"])
    dense_hot = float(dense_payload["hot_run_median_s"])
    provider_hot = float(provider_payload["hot_run_median_s"])
    multiplicity = int(spec["multiplicity"])
    payload = {
        "status": "ok",
        "case": str(case_name),
        "kind": str(spec["kind"]),
        "multiplicity": int(multiplicity),
        "dense": dense_payload,
        "provider": provider_payload,
        "correctness": correctness,
        "ratios": {
            "compile_dense_over_provider": float(dense_compile / provider_compile) if provider_compile else None,
            "hot_dense_over_provider": float(dense_hot / provider_hot) if provider_hot else None,
        },
        "estimated_total_s": {
            "dense_compile": float(dense_compile * multiplicity),
            "provider_compile": float(provider_compile * multiplicity),
            "dense_hot": float(dense_hot * multiplicity),
            "provider_hot": float(provider_hot * multiplicity),
        },
        "config_dense": dense_config,
        "config_provider": provider_config,
    }
    print(json.dumps(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fair R18 unique-layer dense vs current e2e provider kernels.")
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timeout-s", type=int, default=28800)
    parser.add_argument("--worker-case", default="")
    args = parser.parse_args()

    if str(args.worker_case):
        return _worker_main(str(args.worker_case), repeats=int(args.repeats), config_path=Path(args.config))

    payload: dict[str, Any] = {
        "status": "started",
        "scope": "R18 fair unique-layer benchmark: Orion dense vs current r18_tiny_e2e provider",
        "config_path": str(Path(args.config)),
        "cases": [],
        "coverage_note": {
            "included": [
                "stem_bridge",
                "stage1_same",
                "stage1_transition",
                "stage2_same",
                "stage2_transition",
                "stage3_same",
                "stage3_transition",
                "stage4_same",
            ],
            "excluded": [
                "activations",
                "adds",
                "pools",
                "final_linear",
                "bootstrapping_overhead",
            ],
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for case_name in args.cases:
        case_payload: dict[str, Any] = {"case": str(case_name), "status": "running"}
        payload["cases"].append(case_payload)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-case",
            str(case_name),
            "--repeats",
            str(int(args.repeats)),
            "--config",
            str(Path(args.config)),
        ]
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=int(args.timeout_s),
            )
        except subprocess.TimeoutExpired as exc:
            case_payload.update(
                {
                    "status": "failed",
                    "error": f"timeout after {int(args.timeout_s)}s",
                    "stdout_tail": (exc.stdout or "")[-4000:],
                    "stderr_tail": (exc.stderr or "")[-4000:],
                    "worker_wall_s": float(time.perf_counter() - started),
                }
            )
            Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            continue

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        row: dict[str, Any] | None = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if proc.returncode != 0 or row is None:
            case_payload.update(
                {
                    "status": "failed",
                    "returncode": int(proc.returncode),
                    "stdout_tail": stdout[-4000:],
                    "stderr_tail": stderr[-4000:],
                    "worker_wall_s": float(time.perf_counter() - started),
                }
            )
            Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            continue

        case_payload.clear()
        case_payload.update(row)
        case_payload["worker_wall_s"] = float(time.perf_counter() - started)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    successful_rows = [row for row in payload["cases"] if str(row.get("status")) == "ok"]
    dense_compile_total = float(sum(float(row["estimated_total_s"]["dense_compile"]) for row in successful_rows))
    provider_compile_total = float(sum(float(row["estimated_total_s"]["provider_compile"]) for row in successful_rows))
    dense_hot_total = float(sum(float(row["estimated_total_s"]["dense_hot"]) for row in successful_rows))
    provider_hot_total = float(sum(float(row["estimated_total_s"]["provider_hot"]) for row in successful_rows))
    payload["summary"] = {
        "successful_case_count": int(len(successful_rows)),
        "requested_case_count": int(len(args.cases)),
        "estimated_conv_region_totals_s": {
            "dense_compile": float(dense_compile_total),
            "provider_compile": float(provider_compile_total),
            "dense_hot": float(dense_hot_total),
            "provider_hot": float(provider_hot_total),
        },
        "estimated_conv_region_ratios": {
            "compile_dense_over_provider": float(dense_compile_total / provider_compile_total) if provider_compile_total else None,
            "hot_dense_over_provider": float(dense_hot_total / provider_hot_total) if provider_hot_total else None,
        },
    }
    payload["status"] = "ok" if len(successful_rows) == len(args.cases) else "partial"
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
