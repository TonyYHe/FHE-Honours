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
from orion.experimental import U22CompileRegistry
from orion.models.unet import UNet22
from orion.nn.linear import Conv2d
from orion.nn.module import Module


DECODER_TCONV_NODES = ("up4", "up3", "up2", "up1")
DATASET_SPECS = {
    "tiny": {"image_size": 64, "logn": 15},
    "imagenet": {"image_size": 256, "logn": 16},
}


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def _init_scheme(*, backend: str, logn: int) -> None:
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
    registry.attach_to_dag(dag)
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


def _run_backend(node_name: str, *, dataset: str, backend: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
    _init_scheme(backend=str(backend), logn=int(DATASET_SPECS[str(dataset)]["logn"]))
    try:
        dag = _prepared_dag(dataset=str(dataset))
        _set_compile_level(dag)
        module = dag.nodes[str(node_name)]["module"]
        module.generate_diagonals(last=False)
        module.compile()
        module.he_mode = True

        torch.manual_seed(int(seed))
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
        return out, reference, bool(module.region_runtime.supports_scheme(scheme))
    finally:
        scheme.delete_scheme()


@pytest.mark.parametrize("dataset", ("tiny", "imagenet"))
@pytest.mark.parametrize("node_name", DECODER_TCONV_NODES)
def test_u22_decoder_tconv_python_and_lattigo_match(dataset: str, node_name: str) -> None:
    _require_lattigo()

    seed = abs(hash((str(dataset), str(node_name)))) % (2**31)
    python_out, python_ref, python_supported = _run_backend(str(node_name), dataset=str(dataset), backend="python", seed=int(seed))
    lattigo_out, lattigo_ref, lattigo_supported = _run_backend(str(node_name), dataset=str(dataset), backend="lattigo", seed=int(seed))

    assert python_supported is True
    assert lattigo_supported is True
    assert float((python_out - python_ref).abs().max().item()) <= 1.0e-5
    assert float((lattigo_out - lattigo_ref).abs().max().item()) <= 1.0e-4
    assert float((python_out - lattigo_out).abs().max().item()) <= 1.0e-4


def test_native_halo_conv_lattigo_matches_reference_at_level_one() -> None:
    _require_lattigo()
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
    from orion.experimental.cir.native_halo_conv2d import native_halo_source_plaintext_blocks_from_nchw
    from orion.experimental.u22_phase1 import LayoutPolicyProviderRuntimeExecutor

    _init_scheme(backend="lattigo", logn=15)
    try:
        conv = Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        conv.weight.data = torch.tensor(
            [[[[0.0, 0.25, 0.0], [0.5, 1.0, -0.25], [0.0, -0.5, 0.125]]]],
            dtype=torch.float32,
        )
        conv.bias.data.fill_(0.125)
        conv.init_orion_params()
        conv.input_shape = torch.Size((1, 1, 4, 4))
        conv.output_shape = torch.Size((1, 1, 4, 4))
        conv.fhe_input_shape = torch.Size((1, 1, 4, 4))
        conv.fhe_output_shape = torch.Size((1, 1, 4, 4))
        conv.input_gap = 1
        conv.output_gap = 1
        conv.name = "native_halo_level_one_toy_conv"
        conv.set_level(1)
        conv.set_depth(1)

        executor = LayoutPolicyProviderRuntimeExecutor(
            base_executor=HaloLocalConvRuntimeExecutor(module=conv, output_node_id="conv"),
            output_node_id="conv",
            compile_plan={
                "policy": "dp",
                "edge_layouts": [
                    {
                        "edge": "x->conv",
                        "source": "x",
                        "target": "conv",
                        "op_kind": "conv2d",
                        "shape": [1, 1, 4, 4],
                        "fhe_shape": [1, 1, 4, 4],
                        "relayout": False,
                        "layout_mode": "native_halo_stripe",
                        "physical_layout": "native_source_stripe",
                        "source_layout": {"alpha": 1, "beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                        "selected_layout": {"alpha": 1, "beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                    }
                ],
                "node_layouts": [
                    {
                        "node": "conv",
                        "shape": [1, 1, 4, 4],
                        "fhe_shape": [1, 1, 4, 4],
                        "output_relayout": False,
                        "physical_layout": "native_source_stripe",
                        "selected_layout": {"alpha": 1, "beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                    }
                ],
            },
        )
        executor.assigned_level = 1
        executor.assigned_depth = 1

        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 10.0
        native_plan = executor._native_halo_plan()
        ids = []
        for block in native_halo_source_plaintext_blocks_from_nchw(x, native_plan):
            block_ct = scheme.encrypt(scheme.encode(block, 1))
            ids.append(int(block_ct.ids[0]))
            block_ct.ids = []
        x_ct = CipherTensor(
            scheme,
            ids,
            torch.Size(tuple(int(value) for value in x.shape)),
            torch.Size([int(len(ids)), int(scheme.params.get_slots())]),
        )
        out = executor(x_ct)["conv"]
        decoded = (
            out.decrypt()
            .decode()
            .detach()
            .cpu()
            .to(dtype=torch.float32)
            .reshape(-1)[:24]
            .reshape(1, 1, 6, 4)
        )
        x_halo = torch.cat([x[:, :, :1, :], x, x[:, :, -1:, :]], dim=2)
        reference = F.conv2d(x_halo, conv.on_weight.detach(), conv.on_bias.detach(), padding=1)

        assert out.level() == 0
        assert float((decoded - reference).abs().max().item()) <= 1.0e-4
    finally:
        scheme.delete_scheme()
