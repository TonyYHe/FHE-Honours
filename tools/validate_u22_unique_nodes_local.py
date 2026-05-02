from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.network_dag import NetworkDAG
from orion.core.orion import scheme
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import U22CompileRegistry
from orion.models.unet import UNet22
from orion.nn.module import Module


DEFAULT_OUT = Path("/tmp/orion_u22_unique_nodes_local.json")
DECODER_TCONV_NODES = ("up4", "up3", "up2", "up1")
DATASET_SPECS = {
    "tiny": {"image_size": 64, "logn": 15},
    "imagenet": {"image_size": 256, "logn": 16},
}


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        raise RuntimeError("local Lattigo shared library has not been built")


def _init_scheme(*, backend: str, dataset: str) -> None:
    config = {
        "ckks_params": {
            "LogN": int(DATASET_SPECS[str(dataset)]["logn"]),
            "LogQ": [45, 30, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": str(backend),
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    Module.set_scheme(scheme)
    Module.set_margin(scheme.params.get_margin())


def _cleanup_scheme() -> None:
    try:
        scheme.delete_scheme()
    except Exception:
        pass
    gc.collect()


def _prepared_dag(*, dataset: str) -> tuple[NetworkDAG, dict[str, Any]]:
    torch.manual_seed(0)
    image_size = int(DATASET_SPECS[str(dataset)]["image_size"])
    traced = OrionTracer().trace_model(UNet22(dataset=str(dataset)))
    StatsTracker(traced).propagate(torch.randn((1, 3, int(image_size), int(image_size)), dtype=torch.float32))
    dag = NetworkDAG(traced)
    dag.build_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()
    registry = U22CompileRegistry.for_dag(dag)
    attach_audit = registry.attach_to_dag(dag)
    level = len(scheme.params.get_logq()) - 1
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None and hasattr(module, "set_level"):
            module.set_level(level)
    return dag, attach_audit


def _encode_input(module, x: torch.Tensor) -> CipherTensor:
    level = len(scheme.params.get_logq()) - 1
    packed = packing.multiplex(x.unsqueeze(0), int(module.input_gap)).squeeze(0)
    target = torch.zeros(tuple(int(v) for v in module.fhe_input_shape[1:]), dtype=torch.float32)
    target[: packed.shape[0], : packed.shape[1], : packed.shape[2]] = packed
    flat = target.flatten()
    ids: list[int] = []
    slots = int(scheme.params.get_slots())
    for start in range(0, int(flat.numel()), int(slots)):
        block = flat[int(start) : int(min(int(flat.numel()), int(start + int(slots))))]
        padded = torch.zeros((int(slots),), dtype=torch.float32)
        padded[: int(block.numel())] = block
        ct = scheme.encrypt(scheme.encode(padded, level))
        ids.append(int(ct.ids[0]))
        ct.ids = []
    return CipherTensor(scheme, ids, module.input_shape, module.fhe_input_shape)


def _decode_output(module, out: CipherTensor) -> torch.Tensor:
    decoded = out.decrypt().decode().detach().cpu()
    if torch.is_complex(decoded):
        decoded = decoded.real
    flat = decoded.to(dtype=torch.float32).flatten()
    packed_size = int(module.fhe_output_shape[1] * module.fhe_output_shape[2] * module.fhe_output_shape[3])
    packed = flat[: int(packed_size)].reshape(
        1,
        int(module.fhe_output_shape[1]),
        int(module.fhe_output_shape[2]),
        int(module.fhe_output_shape[3]),
    )
    return packing._demultiplex(
        packed,
        int(module.output_gap),
        int(module.output_shape[1]),
        int(module.output_shape[2]),
        int(module.output_shape[3]),
    )[0]


def _metrics(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    diff = other.detach().to(dtype=torch.float32) - reference.detach().to(dtype=torch.float32)
    return {
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.pow(2))).item()),
    }


def _validate_case(*, dataset: str, node_name: str, backend: str) -> dict[str, Any]:
    _init_scheme(backend=str(backend), dataset=str(dataset))
    try:
        dag, attach_audit = _prepared_dag(dataset=str(dataset))
        module = dag.nodes[str(node_name)]["module"]
        runtime = getattr(module, "region_runtime")

        module.generate_diagonals(last=False)
        module.compile()
        module.he_mode = True

        torch.manual_seed(abs(hash((str(dataset), str(node_name), str(backend)))) % (2**31))
        x = torch.randn(tuple(int(v) for v in module.input_shape[1:]), dtype=torch.float32)
        kernel_out = _decode_output(module, module(_encode_input(module, x)))

        # Orion clear path for the same node/weights.
        clear_out = F.conv_transpose2d(
            x.unsqueeze(0),
            module.on_weight.detach().to(dtype=torch.float32),
            module.on_bias.detach().to(dtype=torch.float32) if getattr(module, "on_bias", None) is not None else None,
            stride=tuple(int(v) for v in module.stride),
            padding=tuple(int(v) for v in module.padding),
            output_padding=tuple(int(v) for v in module.output_padding),
            groups=int(module.groups),
            dilation=tuple(int(v) for v in module.dilation),
        )[0]

        return {
            "status": "ok",
            "dataset": str(dataset),
            "node": str(node_name),
            "backend": str(backend),
            "local_machine": True,
            "experimental_kernel": bool(getattr(runtime.executor, "kernel_kind", "") == "tconv_k2s2_gap_halving_experimental"),
            "supports_scheme": bool(runtime.supports_scheme(scheme)),
            "strategy": str(getattr(runtime, "strategy", "")),
            "materializer": str(getattr(runtime, "materializer", "")),
            "attach_audit": dict(attach_audit),
            "new_kernel_vs_orion": _metrics(clear_out, kernel_out),
            "correctness": {
                "provider_vs_reference": _metrics(clear_out, kernel_out),
                "reference_kind": "orion_clear_conv_transpose2d",
            },
        }
    finally:
        _cleanup_scheme()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local U22 unique-node experimental-kernel vs Orion comparisons.")
    parser.add_argument("--backend", choices=("lattigo", "python"), default="lattigo")
    parser.add_argument("--datasets", nargs="*", default=["tiny", "imagenet"])
    parser.add_argument("--nodes", nargs="*", default=list(DECODER_TCONV_NODES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if str(args.backend) == "lattigo":
        _require_lattigo()

    rows: list[dict[str, Any]] = []
    payload = {
        "status": "running",
        "scope": "u22_unique_nodes_local_validation",
        "backend": str(args.backend),
        "rows": rows,
    }
    started = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for node_name in args.nodes:
            row = _validate_case(dataset=str(dataset), node_name=str(node_name), backend=str(args.backend))
            rows.append(row)
            args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"phase": "case_done", **row}), flush=True)

    payload["status"] = "ok"
    payload["timing_s"] = {"total_s": float(time.perf_counter() - started)}
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
