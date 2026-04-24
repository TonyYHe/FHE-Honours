from __future__ import annotations

import argparse
import gc
import json
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
from orion.core.fuser import Fuser
from orion.core.network_dag import NetworkDAG
from orion.core.orion import scheme
from orion.core.region_lowering import pack_chw_gap
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental.cir.runtime_group import RegionFirstCompileRegistry
from orion.models.resnet import ResNet18
from orion.nn.module import Module


DEFAULT_OUT = Path("/tmp/orion_r18_unique_layers_backend_correctness.json")
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
    "stem_bridge": {"kind": "stem", "node": "conv1", "seed": 11},
    "stage1_same": {"kind": "single", "node": "layers_0_0_conv1", "seed": 101},
    "stage1_transition": {"kind": "transition", "nodes": ("layers_1_0_conv1", "layers_1_0_shortcut_0"), "seed": 111},
    "stage2_same": {"kind": "single", "node": "layers_1_1_conv1", "seed": 202},
    "stage2_transition": {"kind": "transition", "nodes": ("layers_2_0_conv1", "layers_2_0_shortcut_0"), "seed": 222},
    "stage3_same": {"kind": "single", "node": "layers_2_1_conv1", "seed": 303},
    "stage3_transition": {"kind": "transition", "nodes": ("layers_3_0_conv1", "layers_3_0_shortcut_0"), "seed": 333},
    "stage4_same": {"kind": "single", "node": "layers_3_0_conv2", "seed": 404},
}


def _load_config(config_path: Path, *, backend: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = dict(config)
    config["ckks_params"] = dict(config.get("ckks_params", {}))
    config["orion"] = dict(config.get("orion", {}))
    config["boot_params"] = dict(config.get("boot_params", {}))
    config["orion"]["backend"] = str(backend)
    config["orion"]["io_mode"] = "none"
    return config


def _prepared_dag(*, provider: bool) -> NetworkDAG:
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
        if hasattr(module, "update_params") and callable(module.update_params):
            module.update_params()
    fuser = Fuser(dag)
    fuser.fuse_modules()
    dag.remove_fused_batchnorms()
    if bool(provider):
        registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
        registry.attach_to_dag(dag)
    return dag


def _set_compile_level(dag: NetworkDAG) -> None:
    level = len(scheme.params.get_logq()) - 1
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None and hasattr(module, "set_level"):
            module.set_level(level)


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


def _plain_blocks(*, count: int, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(int(seed))
    return [torch.randn((scheme.params.get_slots(),), generator=gen, dtype=torch.float32) * 0.01 for _ in range(int(count))]


def _stem_plain_blocks(*, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(int(seed))
    image = torch.randn((3, 64, 64), generator=gen, dtype=torch.float32)
    packed = pack_chw_gap(image, shape=(3, 64, 64), gap=1, slots=32768).to(dtype=torch.float32)
    return [packed]


def _encrypt_source(module: Any, plain_blocks: list[torch.Tensor]) -> CipherTensor:
    level = len(scheme.params.get_logq()) - 1
    ids: list[int] = []
    for block in plain_blocks:
        ct = scheme.encrypt(scheme.encode(block, level))
        ids.append(int(ct.ids[0]))
        ct.ids = []
    return CipherTensor(scheme, ids, module.input_shape, module.fhe_input_shape)


def _decode(ct: CipherTensor) -> torch.Tensor:
    return ct.decrypt().decode().detach().cpu().to(dtype=torch.float32)


def _metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    diff = right - left
    return {
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.pow(2))).item()),
    }


def _validate_single(case_name: str) -> dict[str, Any]:
    spec = dict(CASE_SPECS[str(case_name)])
    dense_dag = _prepared_dag(provider=False)
    provider_dag = _prepared_dag(provider=True)
    _set_compile_level(dense_dag)
    _set_compile_level(provider_dag)
    dense = dense_dag.nodes[str(spec["node"])]["module"]
    provider = provider_dag.nodes[str(spec["node"])]["module"]
    dense.generate_diagonals(last=False)
    dense.compile()
    provider.generate_diagonals(last=False)
    provider.compile()
    dense.he_mode = True
    provider.he_mode = True
    count = max(int(_dense_cols(dense)), int(_provider_cols(provider)))
    plain_blocks = _stem_plain_blocks(seed=int(spec["seed"])) if str(case_name) == "stem_bridge" else _plain_blocks(count=int(count), seed=int(spec["seed"]))
    dense_source = _encrypt_source(dense, list(plain_blocks))
    provider_source = _encrypt_source(provider, list(plain_blocks))
    return {
        "network": "R18",
        "case": str(case_name),
        "kind": str(spec["kind"]),
        "dense_input_level": int(dense.level),
        "provider_input_level": int(provider.level),
        **_metrics(_decode(dense(dense_source)), _decode(provider(provider_source))),
    }


def _validate_transition(case_name: str) -> dict[str, Any]:
    spec = dict(CASE_SPECS[str(case_name)])
    dense_dag = _prepared_dag(provider=False)
    provider_dag = _prepared_dag(provider=True)
    _set_compile_level(dense_dag)
    _set_compile_level(provider_dag)
    left_dense = dense_dag.nodes[str(spec["nodes"][0])]["module"]
    right_dense = dense_dag.nodes[str(spec["nodes"][1])]["module"]
    left_provider = provider_dag.nodes[str(spec["nodes"][0])]["module"]
    right_provider = provider_dag.nodes[str(spec["nodes"][1])]["module"]
    for module in (left_dense, right_dense, left_provider, right_provider):
        module.generate_diagonals(last=False)
        module.compile()
        module.he_mode = True
    count = max(int(_dense_cols(left_dense)), int(_provider_cols(left_provider)))
    plain_blocks = _plain_blocks(count=int(count), seed=int(spec["seed"]))
    dense_source = _encrypt_source(left_dense, list(plain_blocks))
    provider_source = _encrypt_source(left_provider, list(plain_blocks))
    left = _metrics(_decode(left_dense(dense_source)), _decode(left_provider(provider_source)))
    right = _metrics(_decode(right_dense(dense_source)), _decode(right_provider(provider_source)))
    return {
        "network": "R18",
        "case": str(case_name),
        "kind": "transition",
        "dense_input_level": {
            str(spec["nodes"][0]): int(left_dense.level),
            str(spec["nodes"][1]): int(right_dense.level),
        },
        "provider_input_level": {
            str(spec["nodes"][0]): int(left_provider.level),
            str(spec["nodes"][1]): int(right_provider.level),
        },
        "left": left,
        "right": right,
        "mae": float((left["mae"] + right["mae"]) / 2.0),
        "max_abs": float(max(left["max_abs"], right["max_abs"])),
        "rmse": float((left["rmse"] + right["rmse"]) / 2.0),
    }


def _validate_case(case_name: str) -> dict[str, Any]:
    kind = str(CASE_SPECS[str(case_name)]["kind"])
    return _validate_transition(str(case_name)) if kind == "transition" else _validate_single(str(case_name))


def _cleanup_scheme() -> None:
    try:
        scheme.delete_scheme()
    except Exception:
        pass
    gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate R18 unique-layer provider correctness on a chosen backend.")
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backend", choices=("python", "lattigo"), default="python")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    payload = {
        "status": "running",
        "scope": "R18 unique-layer backend correctness",
        "backend": str(args.backend),
        "config_path": str(Path(args.config)),
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for case_name in args.cases:
        scheme.init_scheme(_load_config(Path(args.config), backend=str(args.backend)))
        try:
            Module.set_scheme(scheme)
            Module.set_margin(scheme.params.get_margin())
            row = _validate_case(str(case_name))
            rows.append(row)
            Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"phase": "case_done", **row}), flush=True)
        finally:
            _cleanup_scheme()

    payload["status"] = "ok"
    payload["timing_s"] = {"total_s": float(time.perf_counter() - started)}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
