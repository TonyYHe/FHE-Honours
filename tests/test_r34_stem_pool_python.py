from __future__ import annotations

import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.network_dag import NetworkDAG
from orion.core.orion import scheme
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import R34CompileRegistry
from orion.nn.module import Module
from orion.models.resnet import ResNet34


def _init_python_scheme() -> None:
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
            "backend": "python",
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


def _assert_single_case(node_name: str, *, tolerance: float = 2.5e-4) -> None:
    _init_python_scheme()
    try:
        dag = _prepared_dag()
        registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
        registry.attach_to_dag(dag)
        _set_compile_level(dag)

        module = dag.nodes[str(node_name)]["module"]
        module.generate_diagonals(last=False)
        module.compile()
        module.he_mode = True

        torch.manual_seed(abs(hash(str(node_name))) % (2**31))
        x = torch.randn(tuple(int(v) for v in module.input_shape[1:]), dtype=torch.float32)
        source = _encode_input(module, x)
        out = _decode_output(module, module(source))

        reference = F.conv2d(
            x.unsqueeze(0),
            module.on_weight.detach().to(dtype=torch.float32),
            module.on_bias.detach().to(dtype=torch.float32) if getattr(module, "on_bias", None) is not None else None,
            stride=tuple(int(v) for v in module.stride),
            padding=tuple(int(v) for v in module.padding),
            dilation=tuple(int(v) for v in module.dilation),
            groups=int(module.groups),
        )[0]

        assert float((out - reference).abs().max().item()) <= float(tolerance)
    finally:
        scheme.delete_scheme()


def test_r34_stem_conv_python_runtime_matches_reference() -> None:
    _assert_single_case("conv1")


def test_r34_stem_pool_python_runtime_matches_reference() -> None:
    _assert_single_case("pool")


def test_r34_global_avgpool_exit_python_runtime_matches_reference() -> None:
    _assert_single_case("avgpool")


def test_r34_stem_conv_to_pool_chain_matches_reference() -> None:
    _init_python_scheme()
    try:
        dag = _prepared_dag()
        registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
        registry.attach_to_dag(dag)
        _set_compile_level(dag)

        conv = dag.nodes["conv1"]["module"]
        pool = dag.nodes["pool"]["module"]
        conv.generate_diagonals(last=False)
        pool.generate_diagonals(last=False)
        conv.compile()
        pool.compile()
        conv.he_mode = True
        pool.he_mode = True

        torch.manual_seed(7)
        x = torch.randn(tuple(int(v) for v in conv.input_shape[1:]), dtype=torch.float32)
        source = _encode_input(conv, x)
        out = _decode_output(pool, pool(conv(source)))

        conv_ref = F.conv2d(
            x.unsqueeze(0),
            conv.on_weight.detach().to(dtype=torch.float32),
            conv.on_bias.detach().to(dtype=torch.float32) if getattr(conv, "on_bias", None) is not None else None,
            stride=tuple(int(v) for v in conv.stride),
            padding=tuple(int(v) for v in conv.padding),
            dilation=tuple(int(v) for v in conv.dilation),
            groups=int(conv.groups),
        )[0]
        reference = F.conv2d(
            conv_ref.unsqueeze(0),
            pool.on_weight.detach().to(dtype=torch.float32),
            pool.on_bias.detach().to(dtype=torch.float32) if getattr(pool, "on_bias", None) is not None else None,
            stride=tuple(int(v) for v in pool.stride),
            padding=tuple(int(v) for v in pool.padding),
            dilation=tuple(int(v) for v in pool.dilation),
            groups=int(pool.groups),
        )[0]

        assert float((out - reference).abs().max().item()) <= 2.5e-4
    finally:
        scheme.delete_scheme()


def test_r34_avgpool_to_linear_chain_matches_reference() -> None:
    _init_python_scheme()
    try:
        dag = _prepared_dag()
        registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
        registry.attach_to_dag(dag)
        _set_compile_level(dag)

        avgpool = dag.nodes["avgpool"]["module"]
        flatten = dag.nodes["flatten"]["module"]
        linear = dag.nodes["linear"]["module"]
        avgpool.generate_diagonals(last=False)
        linear.generate_diagonals(last=True)
        avgpool.compile()
        linear.compile()
        avgpool.he_mode = True
        flatten.he_mode = True
        linear.he_mode = True

        torch.manual_seed(11)
        x = torch.randn(tuple(int(v) for v in avgpool.input_shape[1:]), dtype=torch.float32)
        source = _encode_input(avgpool, x)
        out = linear(flatten(avgpool(source))).decrypt().decode().detach().cpu().to(dtype=torch.float32)

        pooled = F.conv2d(
            x.unsqueeze(0),
            avgpool.on_weight.detach().to(dtype=torch.float32),
            avgpool.on_bias.detach().to(dtype=torch.float32) if getattr(avgpool, "on_bias", None) is not None else None,
            stride=tuple(int(v) for v in avgpool.stride),
            padding=tuple(int(v) for v in avgpool.padding),
            dilation=tuple(int(v) for v in avgpool.dilation),
            groups=int(avgpool.groups),
        )
        reference = F.linear(
            pooled.view(int(pooled.shape[0]), -1),
            linear.on_weight.detach().to(dtype=torch.float32),
            linear.on_bias.detach().to(dtype=torch.float32),
        )

        assert out.shape == reference.shape
        assert float((out - reference).abs().max().item()) <= 2.5e-4
    finally:
        scheme.delete_scheme()
