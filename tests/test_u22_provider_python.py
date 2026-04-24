from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.network_dag import NetworkDAG
from orion.core.orion import scheme
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import U22CompileRegistry
from orion.experimental.u22_phase1 import TconvK2S2PythonRuntimeExecutor
from orion.models.unet import UNet22
from orion.nn.linear import ConvTranspose2d
from orion.nn.module import Module


DATASET_SPECS = {
    "tiny": {"image_size": 64, "logn": 15},
    "imagenet": {"image_size": 256, "logn": 17},
}

DECODER_TCONV_NODES = ("up4", "up3", "up2", "up1")


def _init_python_scheme(*, logn: int) -> None:
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
            "backend": "python",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    Module.set_scheme(scheme)
    Module.set_margin(scheme.params.get_margin())


def _prepared_dag(*, dataset: str, base_channels: int | None = None) -> NetworkDAG:
    image_size = int(DATASET_SPECS[str(dataset)]["image_size"])
    torch.manual_seed(0)
    traced = OrionTracer().trace_model(UNet22(dataset=dataset, base_channels=base_channels))
    StatsTracker(traced).propagate(torch.randn((1, 3, int(image_size), int(image_size)), dtype=torch.float32))
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


@pytest.mark.parametrize("dataset", ("tiny", "imagenet"))
def test_u22_registry_attaches_all_decoder_tconvs_with_experimental_kernel(dataset: str) -> None:
    _init_python_scheme(logn=int(DATASET_SPECS[str(dataset)]["logn"]))
    try:
        dag = _prepared_dag(dataset=str(dataset))
        registry = U22CompileRegistry.for_dag(dag)
        audit = registry.attach_to_dag(dag)

        assert audit["attached_count"] == len(DECODER_TCONV_NODES)
        assert {row["node"] for row in audit["attached"]} == set(DECODER_TCONV_NODES)
        assert audit["graph_audit"]["excluded_nodes"] == []
        for node_name in DECODER_TCONV_NODES:
            module = dag.nodes[str(node_name)]["module"]
            runtime = getattr(module, "region_runtime")
            assert runtime is not None
            assert runtime.strategy == "tconv_k2s2_gap_halving_experimental"
            assert runtime.materializer == "tconv_k2s2_gap_halving_experimental"
            assert runtime.executable is True
            assert getattr(runtime.executor, "kernel_kind", "") == "tconv_k2s2_gap_halving_experimental"
    finally:
        scheme.delete_scheme()


@pytest.mark.parametrize("dataset", ("tiny", "imagenet"))
@pytest.mark.parametrize("node_name", DECODER_TCONV_NODES)
def test_u22_decoder_tconv_provider_matches_reference_on_python_backend(monkeypatch, dataset: str, node_name: str) -> None:
    def fail_pack_conv_transpose2d(*_args, **_kwargs):
        raise AssertionError("U22 decoder provider runtime must not call pack_conv_transpose2d")

    monkeypatch.setattr(packing, "pack_conv_transpose2d", fail_pack_conv_transpose2d)
    _init_python_scheme(logn=int(DATASET_SPECS[str(dataset)]["logn"]))
    try:
        dag = _prepared_dag(dataset=str(dataset))
        registry = U22CompileRegistry.for_dag(dag)
        registry.attach_to_dag(dag)
        _set_compile_level(dag)

        module = dag.nodes[str(node_name)]["module"]
        runtime = getattr(module, "region_runtime")
        assert runtime is not None
        assert runtime.supports_scheme(scheme) is True

        module.generate_diagonals(last=False)
        module.compile()
        module.he_mode = True

        torch.manual_seed(abs(hash((str(dataset), str(node_name)))) % (2**31))
        x = torch.randn(tuple(int(v) for v in module.input_shape[1:]), dtype=torch.float32)
        out = _decode_output(module, module(_encode_input(module, x)))
        reference = F.conv_transpose2d(
            x.unsqueeze(0),
            module.on_weight.detach().to(dtype=torch.float32),
            module.on_bias.detach().to(dtype=torch.float32) if getattr(module, "on_bias", None) is not None else None,
            stride=tuple(int(v) for v in module.stride),
            padding=tuple(int(v) for v in module.padding),
            output_padding=tuple(int(v) for v in module.output_padding),
            groups=int(module.groups),
            dilation=tuple(int(v) for v in module.dilation),
        )[0]

        assert float((out - reference).abs().max().item()) <= 1.0e-5
        assert runtime.execute_count == 1
    finally:
        scheme.delete_scheme()


def test_u22_decoder_tconv_provider_supports_actual_base64_tiny_at_logn16() -> None:
    _init_python_scheme(logn=16)
    try:
        dag = _prepared_dag(dataset="tiny", base_channels=64)
        registry = U22CompileRegistry.for_dag(dag)
        registry.attach_to_dag(dag)
        for node_name in DECODER_TCONV_NODES:
            module = dag.nodes[str(node_name)]["module"]
            runtime = getattr(module, "region_runtime")
            assert runtime is not None
            assert runtime.supports_scheme(scheme) is True
    finally:
        scheme.delete_scheme()


def test_u22_tconv_provider_runtime_handles_multi_input_and_output_blocks_on_python_backend() -> None:
    _init_python_scheme(logn=10)
    try:
        torch.manual_seed(11)
        module = ConvTranspose2d(
            in_channels=48,
            out_channels=24,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
        )
        module.eval()
        module.init_orion_params()
        module.input_shape = torch.Size((1, 48, 4, 4))
        module.output_shape = torch.Size((1, 24, 8, 8))
        module.input_gap = 4
        module.output_gap = 2
        module.fhe_input_shape = torch.Size((1, 3, 16, 16))
        module.fhe_output_shape = torch.Size((1, 6, 16, 16))
        module.set_level(len(scheme.params.get_logq()) - 1)

        runtime = TconvK2S2PythonRuntimeExecutor(module=module, output_node_id="synthetic_u22_tconv")
        assert runtime.supports_scheme(scheme) is True

        torch.manual_seed(19)
        x = torch.randn(tuple(int(v) for v in module.input_shape), dtype=torch.float32)
        out = runtime(_encode_input(module, x[0]))
        result = _decode_output(module, out["synthetic_u22_tconv"])
        reference = F.conv_transpose2d(
            x,
            module.on_weight.detach().to(dtype=torch.float32),
            module.on_bias.detach().to(dtype=torch.float32),
            stride=tuple(int(v) for v in module.stride),
            padding=tuple(int(v) for v in module.padding),
            output_padding=tuple(int(v) for v in module.output_padding),
            groups=int(module.groups),
            dilation=tuple(int(v) for v in module.dilation),
        )[0]

        assert runtime.input_block_count == 2
        assert runtime.output_block_count == 3
        assert float((result - reference).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()
