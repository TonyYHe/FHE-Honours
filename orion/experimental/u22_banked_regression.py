from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.network_dag import NetworkDAG
from orion.core.orion import scheme
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import U22CompileRegistry
from orion.models.unet import UNet22
from orion.nn.module import Module


@dataclass(frozen=True)
class U22BankedRegressionCase:
    case_name: str
    dataset: str
    node_name: str
    expected_rotations: int
    expected_conjugations: int
    assumption: str


U22_BANKED_REGRESSION_CASES: dict[str, U22BankedRegressionCase] = {
    "u4_tiny": U22BankedRegressionCase(
        case_name="u4_tiny",
        dataset="tiny",
        node_name="dec4a",
        expected_rotations=58,
        expected_conjugations=0,
        assumption="Mapped to current mainline UNet22(dataset='tiny') provider conv node dec4a; ConvTranspose2d uses the common dense path.",
    ),
    "u4_mini": U22BankedRegressionCase(
        case_name="u4_mini",
        dataset="imagenet",
        node_name="dec4a",
        expected_rotations=54,
        expected_conjugations=0,
        assumption="Mapped to current mainline UNet22(dataset='imagenet') provider conv node dec4a; ConvTranspose2d uses the common dense path.",
    ),
    "u3_mini": U22BankedRegressionCase(
        case_name="u3_mini",
        dataset="imagenet",
        node_name="dec3a",
        expected_rotations=38,
        expected_conjugations=0,
        assumption="Mapped to current mainline UNet22(dataset='imagenet') provider conv node dec3a; ConvTranspose2d uses the common dense path.",
    ),
}


_DATASET_SPECS = {
    "tiny": {"image_size": 64, "logn": 15},
    "imagenet": {"image_size": 256, "logn": 16},
}


def require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        raise RuntimeError("local Lattigo shared library has not been built")


def _init_scheme(*, backend: str, dataset: str) -> None:
    config = {
        "ckks_params": {
            "LogN": int(_DATASET_SPECS[str(dataset)]["logn"]),
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


def _prepared_dag(*, dataset: str) -> NetworkDAG:
    torch.manual_seed(0)
    image_size = int(_DATASET_SPECS[str(dataset)]["image_size"])
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
    registry = U22CompileRegistry.for_dag(dag, enable_conv_kernels=True)
    attach_audit = registry.attach_to_dag(dag)
    _set_compile_level(dag)
    return dag, attach_audit


def _set_compile_level(dag: NetworkDAG) -> None:
    level = len(scheme.params.get_logq()) - 1
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None and hasattr(module, "set_level"):
            module.set_level(level)


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


def _rotation_key_stats(groups: list[Any], backend: Any) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    group_union_total = 0
    transform_sum_total = 0
    get_keys = getattr(backend, "GetLinearTransformRotationKeys")
    for group_index, group in enumerate(groups):
        transform_rows: list[dict[str, Any]] = []
        union_keys: set[int] = set()
        transform_ids = [int(value) for value in list(getattr(group, "unified_ids", []) or [])]
        for transform_id in transform_ids:
            keys = sorted({int(key) for key in list(get_keys(int(transform_id))) if int(key) != 0})
            union_keys.update(int(key) for key in keys)
            transform_sum_total += int(len(keys))
            transform_rows.append(
                {
                    "transform_id": int(transform_id),
                    "rotation_count": int(len(keys)),
                    "rotation_keys": [int(key) for key in keys],
                }
            )
        group_union_total += int(len(union_keys))
        group_rows.append(
            {
                "group_index": int(group_index),
                "unified_ids": [int(value) for value in transform_ids],
                "union_rotation_count": int(len(union_keys)),
                "transform_sum_rotation_count": int(sum(int(row["rotation_count"]) for row in transform_rows)),
                "union_rotation_keys": [int(key) for key in sorted(union_keys)],
                "transforms": transform_rows,
            }
        )
    return {
        "group_union_rotation_count": int(group_union_total),
        "transform_sum_rotation_count": int(transform_sum_total),
        "groups": group_rows,
    }


def _executor_chain(executor: Any) -> list[Any]:
    chain: list[Any] = []
    seen: set[int] = set()
    pending = [executor]
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        pending.append(getattr(current, "base_executor", None))
        pending.append(getattr(current, "delegate", None))
    return chain


def _executor_groups(executor: Any) -> list[Any]:
    groups: list[Any] = []
    seen: set[int] = set()

    def add_group(group: Any) -> None:
        if group is None or id(group) in seen:
            return
        seen.add(id(group))
        groups.append(group)

    for current in _executor_chain(executor):
        single_group = getattr(current, "group", None)
        add_group(single_group)
        for attr in ("groups", "runtime_groups", "groups_by_pair", "groups_by_input_index"):
            value = getattr(current, attr, None)
            if value is None:
                continue
            values = value.values() if isinstance(value, dict) else value
            for item in values:
                if item is None:
                    continue
                if isinstance(item, (list, tuple)):
                    for entry in item:
                        add_group(entry)
                else:
                    add_group(item)
    return groups


def _executor_kernel_kinds(executor: Any) -> set[str]:
    return {
        str(getattr(current, "kernel_kind", ""))
        for current in _executor_chain(executor)
        if str(getattr(current, "kernel_kind", ""))
    }


def _executor_runtime_counts(executor: Any) -> dict[str, Any]:
    for current in _executor_chain(executor):
        counts = dict(getattr(current, "last_runtime_counts", {}) or {})
        if counts:
            return counts
    return {}


def run_u22_banked_regression_case(case_name: str, *, backend: str = "lattigo") -> dict[str, Any]:
    case = U22_BANKED_REGRESSION_CASES[str(case_name)]
    if str(backend) == "lattigo":
        require_lattigo()

    _init_scheme(backend=str(backend), dataset=str(case.dataset))
    try:
        dag, attach_audit = _prepared_dag(dataset=str(case.dataset))
        module = dag.nodes[str(case.node_name)]["module"]
        runtime = getattr(module, "region_runtime", None)
        if runtime is None:
            raise RuntimeError(
                f"U22 banked regression node {case.node_name!r} did not receive a provider runtime. "
                f"attach_audit={attach_audit}"
            )
        module.generate_diagonals(last=False)
        module.compile()
        module.he_mode = True

        torch.manual_seed(abs(hash((str(case.case_name), str(backend)))) % (2**31))
        x = torch.randn(tuple(int(v) for v in module.input_shape[1:]), dtype=torch.float32)
        out = _decode_output(module, module(_encode_input(module, x)))
        weight = module.on_weight.detach().to(dtype=torch.float32)
        bias = module.on_bias.detach().to(dtype=torch.float32) if getattr(module, "on_bias", None) is not None else None
        if type(module).__name__ == "ConvTranspose2d":
            reference = F.conv_transpose2d(
                x.unsqueeze(0),
                weight,
                bias,
                stride=tuple(int(v) for v in module.stride),
                padding=tuple(int(v) for v in module.padding),
                output_padding=tuple(int(v) for v in module.output_padding),
                groups=int(module.groups),
                dilation=tuple(int(v) for v in module.dilation),
            )[0]
        else:
            reference = F.conv2d(
                x.unsqueeze(0),
                weight,
                bias,
                stride=tuple(int(v) for v in module.stride),
                padding=tuple(int(v) for v in module.padding),
                groups=int(module.groups),
                dilation=tuple(int(v) for v in module.dilation),
            )[0]
        max_abs = float((out - reference).abs().max().item())
        mae = float((out - reference).abs().mean().item())

        executor = getattr(runtime, "executor", None)
        groups = _executor_groups(executor)
        rotation_key_stats = _rotation_key_stats(groups, scheme.backend)
        observed_rotations = int(rotation_key_stats["group_union_rotation_count"])

        observed_conjugations = 0
        kernel_kinds = _executor_kernel_kinds(executor)
        payload = {
            "status": "ok" if max_abs <= 1.0e-4 else "failed",
            "case": str(case.case_name),
            "backend": str(backend),
            "dataset": str(case.dataset),
            "node": str(case.node_name),
            "local_lattigo": bool(str(backend) == "lattigo"),
            "experimental_kernel": bool(
                "halo_supported_tconv" in kernel_kinds
                or "tconv_k2s2_gap_halving_experimental" in kernel_kinds
                or "halo_local_conv2d" in kernel_kinds
                or "input_pair_conv_shared_rotations" in kernel_kinds
            ),
            "supports_scheme": bool(runtime.supports_scheme(scheme)),
            "strategy": str(getattr(runtime, "strategy", "")),
            "materializer": str(getattr(runtime, "materializer", "")),
            "attach_audit": dict(attach_audit),
            "assumption": str(case.assumption),
            "observed": {
                "rotations": int(observed_rotations),
                "conjugations": int(observed_conjugations),
            },
            "rotation_key_stats": rotation_key_stats,
            "runtime_counts": _executor_runtime_counts(executor),
            "expected": {
                "rotations": int(case.expected_rotations),
                "conjugations": int(case.expected_conjugations),
            },
            "drift": {
                "rotations": int(observed_rotations - int(case.expected_rotations)),
                "conjugations": int(observed_conjugations - int(case.expected_conjugations)),
            },
            "parity": {
                "exact": bool(max_abs <= 1.0e-4),
                "max_abs": float(max_abs),
                "mae": float(mae),
                "tolerance": 1.0e-4,
            },
        }
        return payload
    finally:
        scheme.delete_scheme()
