from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.network_dag import NetworkDAG
from orion.core.orion import scheme
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import R34CompileRegistry
from orion.experimental.cir import r34_orion_same_shape as r34_same_shape
from orion.experimental.cir.transition_pool_provider import (
    BranchPairConvRuntimeExecutor,
    InputPairConvRuntimeExecutor,
)
from orion.models.resnet import ResNet34
from orion.nn.module import Module


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def _init_lattigo_scheme() -> None:
    config = {
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
    scheme.init_scheme(config)
    Module.set_scheme(scheme)
    Module.set_margin(scheme.params.get_margin())


def _prepared_dag() -> NetworkDAG:
    torch.manual_seed(0)
    traced = OrionTracer().trace_model(ResNet34(dataset="imagenet"))
    StatsTracker(traced).propagate(torch.randn((1, 3, 224, 224), dtype=torch.float32))
    dag = NetworkDAG(traced)
    dag.build_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()
    return dag


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
    decoded = decoded.to(dtype=torch.float32)

    output_shape = getattr(module, "output_shape", None)
    fhe_output_shape = getattr(module, "fhe_output_shape", None)
    output_gap = getattr(module, "output_gap", None)
    if (
        output_shape is not None
        and fhe_output_shape is not None
        and output_gap is not None
        and len(tuple(output_shape)) == 4
        and len(tuple(fhe_output_shape)) == 4
    ):
        flat = decoded.flatten()
        packed_size = int(fhe_output_shape[1] * fhe_output_shape[2] * fhe_output_shape[3])
        packed = flat[: int(packed_size)].reshape(
            1,
            int(fhe_output_shape[1]),
            int(fhe_output_shape[2]),
            int(fhe_output_shape[3]),
        )
        return packing._demultiplex(
            packed,
            int(output_gap),
            int(output_shape[1]),
            int(output_shape[2]),
            int(output_shape[3]),
        )[0]
    return decoded


def _compile_module(node_name: str):
    dag = _prepared_dag()
    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
    registry.attach_to_dag(dag)
    _set_compile_level(dag)
    module = dag.nodes[str(node_name)]["module"]
    module.generate_diagonals(last=False)
    module.compile()
    module.he_mode = True
    return dag, module


def _conv_reference(module, x: torch.Tensor) -> torch.Tensor:
    return F.conv2d(
        x.unsqueeze(0),
        module.on_weight.detach().to(dtype=torch.float32),
        module.on_bias.detach().to(dtype=torch.float32) if getattr(module, "on_bias", None) is not None else None,
        stride=tuple(int(v) for v in module.stride),
        padding=tuple(int(v) for v in module.padding),
        dilation=tuple(int(v) for v in module.dilation),
        groups=int(module.groups),
    )[0]


@pytest.mark.parametrize(
    ("conv_name", "shortcut_name"),
    (
        ("layers_1_0_conv1", "layers_1_0_shortcut_0"),
        ("layers_2_0_conv1", "layers_2_0_shortcut_0"),
        ("layers_3_0_conv1", "layers_3_0_shortcut_0"),
    ),
)
def test_r34_transition_lattigo_runtime_matches_reference(conv_name: str, shortcut_name: str) -> None:
    _require_lattigo()
    _init_lattigo_scheme()
    try:
        dag = _prepared_dag()
        registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
        registry.attach_to_dag(dag)
        _set_compile_level(dag)

        conv = dag.nodes[str(conv_name)]["module"]
        shortcut = dag.nodes[str(shortcut_name)]["module"]
        conv.generate_diagonals(last=False)
        shortcut.generate_diagonals(last=False)
        conv.compile()
        shortcut.compile()
        conv.he_mode = True
        shortcut.he_mode = True

        assert isinstance(conv.region_runtime.executor, BranchPairConvRuntimeExecutor)
        assert conv.region_runtime.supports_scheme(scheme) is True

        torch.manual_seed(23)
        x = torch.randn(tuple(int(v) for v in conv.input_shape[1:]), dtype=torch.float32)
        source = _encode_input(conv, x)
        conv_out = _decode_output(conv, conv(source))
        shortcut_out = _decode_output(shortcut, shortcut(source))
        conv_ref = _conv_reference(conv, x)
        shortcut_ref = _conv_reference(shortcut, x)

        assert float((conv_out - conv_ref).abs().max().item()) <= 1.0e-3
        assert float((shortcut_out - shortcut_ref).abs().max().item()) <= 1.0e-3
    finally:
        scheme.delete_scheme()


@pytest.mark.parametrize(
    ("node_name", "executor_type"),
    (
        ("layers_0_0_conv1", r34_same_shape.R34InterGroupHybridSameShapeRuntimeExecutor),
        ("layers_1_1_conv1", r34_same_shape.R34InterGroupHybridSameShapeRuntimeExecutor),
        ("layers_2_1_conv1", r34_same_shape.R34Pack2SameShapeRuntimeExecutor),
        ("layers_3_1_conv1", r34_same_shape.R34Pack2SameShapeRuntimeExecutor),
    ),
)
def test_r34_same_shape_lattigo_runtime_matches_reference_without_dense_pack(monkeypatch, node_name: str, executor_type: type) -> None:
    _require_lattigo()

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError(f"{node_name} Lattigo runtime must not call pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
    _init_lattigo_scheme()
    try:
        _dag, module = _compile_module(str(node_name))
        assert isinstance(module.region_runtime.executor, executor_type)

        torch.manual_seed(abs(hash(str(node_name))) % (2**31))
        x = torch.randn(tuple(int(v) for v in module.input_shape[1:]), dtype=torch.float32)
        source = _encode_input(module, x)
        out = _decode_output(module, module(source))
        reference = _conv_reference(module, x)

        assert float((out - reference).abs().max().item()) <= 1.0e-3
    finally:
        scheme.delete_scheme()


@pytest.mark.parametrize(("node_name",), (("conv1",), ("pool",), ("avgpool",)))
def test_r34_single_flow_lattigo_runtime_matches_reference(node_name: str) -> None:
    _require_lattigo()
    _init_lattigo_scheme()
    try:
        _dag, module = _compile_module(str(node_name))
        assert isinstance(module.region_runtime.executor, InputPairConvRuntimeExecutor)

        torch.manual_seed(abs(hash(str(node_name))) % (2**31))
        x = torch.randn(tuple(int(v) for v in module.input_shape[1:]), dtype=torch.float32)
        source = _encode_input(module, x)
        out = _decode_output(module, module(source))
        reference = _conv_reference(module, x)

        assert float((out - reference).abs().max().item()) <= 1.0e-3
    finally:
        scheme.delete_scheme()
