from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.network_dag import NetworkDAG
from orion.core.orion import _region_first_mode_options, scheme
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import U22CompileRegistry
from orion.experimental.u22_phase1 import TconvK2S2PythonRuntimeExecutor
from orion.models.unet import UNet22
from orion.nn.linear import ConvTranspose2d
from orion.nn.module import Module


DATASET_SPECS = {
    "tiny": {"image_size": 64, "logn": 15, "input_channels": 3},
    "imagenet": {"image_size": 256, "logn": 16, "input_channels": 3},
    "montgomery_lung_64": {"image_size": 64, "logn": 15, "input_channels": 1},
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
    spec = DATASET_SPECS[str(dataset)]
    image_size = int(spec["image_size"])
    input_channels = int(spec["input_channels"])
    torch.manual_seed(0)
    traced = OrionTracer().trace_model(UNet22(dataset=dataset, base_channels=base_channels))
    StatsTracker(traced).propagate(
        torch.randn((1, int(input_channels), int(image_size), int(image_size)), dtype=torch.float32)
    )
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
            assert getattr(runtime.executor, "use_ct_pt_hybrid_packing", False) is True
    finally:
        scheme.delete_scheme()


def test_u22_registry_can_attach_decoder_tconv_subset() -> None:
    _init_python_scheme(logn=int(DATASET_SPECS["tiny"]["logn"]))
    try:
        dag = _prepared_dag(dataset="tiny")
        registry = U22CompileRegistry.for_dag(dag, allowed_nodes=("up2", "up1"))
        audit = registry.attach_to_dag(dag)

        assert audit["attached_count"] == 2
        assert {row["node"] for row in audit["attached"]} == {"up2", "up1"}
        assert audit["graph_audit"]["allowed_nodes"] == ["up2", "up1"]
        filtered = {
            row["node"]
            for row in audit["graph_audit"]["excluded_nodes"]
            if row["reason"] == "u22_ablation_filtered_out"
        }
        assert filtered == {"up4", "up3"}
        for node_name in ("up2", "up1"):
            assert getattr(dag.nodes[str(node_name)]["module"], "region_runtime", None) is not None
        for node_name in ("up4", "up3"):
            assert getattr(dag.nodes[str(node_name)]["module"], "region_runtime", None) is None
    finally:
        scheme.delete_scheme()


def test_u22_64_default_provider_mode_selects_all_decoder_tconvs_and_conv_kernels() -> None:
    opts = _region_first_mode_options("u22_64_base32")
    assert opts["enabled"] is True
    assert opts["is_u22_phase1"] is True
    assert opts["u22_allowed_nodes"] == ("up1", "up2", "up3", "up4")
    assert opts["u22_conv_kernels"] is True

    imagenet_opts = _region_first_mode_options("u22_256_base32")
    assert imagenet_opts["enabled"] is True
    assert imagenet_opts["is_u22_phase1"] is True
    assert imagenet_opts["u22_allowed_nodes"] == ("up4", "up3")
    assert imagenet_opts["u22_conv_kernels"] is True

    ablation_opts = _region_first_mode_options("u22_64_base32_up1234_noconv")
    assert ablation_opts["u22_allowed_nodes"] == ("up1", "up2", "up3", "up4")
    assert ablation_opts["u22_conv_kernels"] is False

    up34_opts = _region_first_mode_options("u22_64_base32_up34_conv")
    assert up34_opts["u22_allowed_nodes"] == ("up3", "up4")
    assert up34_opts["u22_conv_kernels"] is True


def test_u22_registry_can_attach_up34_and_same_shape_conv_kernels() -> None:
    _init_python_scheme(logn=int(DATASET_SPECS["tiny"]["logn"]))
    try:
        dag = _prepared_dag(dataset="tiny", base_channels=32)
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=("up4", "up3"),
            enable_conv_kernels=True,
        )
        audit = registry.attach_to_dag(dag)
        attached = {row["node"] for row in audit["attached"]}

        assert {"up4", "up3", "enc1b", "dec1a", "dec2b", "pool1", "pool2", "pool3", "pool4"}.issubset(attached)
        assert "up2" not in attached
        assert "up1" not in attached
        assert "bottleneckb" in attached
        assert audit["graph_audit"]["allowed_nodes"] == ["up4", "up3"]
        assert audit["graph_audit"]["enable_conv_kernels"] is True
        assert audit["graph_audit"]["selected_tconv_count"] == 2
        assert audit["graph_audit"]["selected_conv_count"] >= 15
        assert audit["graph_audit"]["selected_pool_count"] == 4
        assert audit["graph_audit"]["selected_generic_conv_count"] >= 7

        for node_name in ("enc1b", "dec1a", "dec2b"):
            runtime = getattr(dag.nodes[str(node_name)]["module"], "region_runtime", None)
            assert runtime is not None
            assert runtime.strategy.startswith("u22_conv_same_shape_")
            assert runtime.supports_scheme(scheme) is True
        assert dag.nodes["pool1"]["module"].region_runtime.stage == "pool_downsample"
        assert dag.nodes["bottleneckb"]["module"].region_runtime.stage == "single_block_conv"
    finally:
        scheme.delete_scheme()


def test_u22_64_base32_provider_mode_attaches_all_linear_and_pool_nodes() -> None:
    _init_python_scheme(logn=int(DATASET_SPECS["montgomery_lung_64"]["logn"]))
    try:
        dag = _prepared_dag(dataset="montgomery_lung_64", base_channels=32)
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=("up1", "up2", "up3", "up4"),
            enable_conv_kernels=True,
        )
        audit = registry.attach_to_dag(dag)
        attached = {row["node"] for row in audit["attached"]}
        expected = {
            "enc1a",
            "enc1b",
            "pool1",
            "enc2a",
            "enc2b",
            "pool2",
            "enc3a",
            "enc3b",
            "pool3",
            "enc4a",
            "enc4b",
            "pool4",
            "bottlenecka",
            "bottleneckb",
            "up4",
            "dec4a",
            "dec4b",
            "up3",
            "dec3a",
            "dec3b",
            "up2",
            "dec2a",
            "dec2b",
            "up1",
            "dec1a",
            "dec1b",
        }

        assert attached == expected
        assert audit["attached_count"] == 26
        assert audit["executable_region_count"] == 26
        assert audit["graph_audit"]["excluded_nodes"] == []
        assert audit["graph_audit"]["selected_tconv_count"] == 4
        assert audit["graph_audit"]["selected_conv_count"] == 18
        assert audit["graph_audit"]["selected_pool_count"] == 4
        assert audit["graph_audit"]["selected_generic_conv_count"] == 7
        assert dag.nodes["bottleneckb"]["module"].region_runtime.stage == "single_block_conv"
        assert dag.nodes["dec1b"]["module"].region_runtime.stage == "channel_transition"
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
        no_hybrid_runtime = TconvK2S2PythonRuntimeExecutor(
            module=module,
            output_node_id="synthetic_u22_tconv_no_hybrid",
            use_ct_pt_hybrid_packing=False,
        )
        assert no_hybrid_runtime.supports_scheme(scheme) is True

        torch.manual_seed(19)
        x = torch.randn(tuple(int(v) for v in module.input_shape), dtype=torch.float32)
        source = _encode_input(module, x[0])
        out = runtime(source)
        no_hybrid_out = no_hybrid_runtime(source)
        result = _decode_output(module, out["synthetic_u22_tconv"])
        no_hybrid_result = _decode_output(module, no_hybrid_out["synthetic_u22_tconv_no_hybrid"])
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
        assert runtime.use_ct_pt_hybrid_packing is True
        assert runtime.input_block_pairs == [(0, 1)]
        assert len(runtime.groups) == 1
        assert runtime.block_evaluate_count == 1
        assert no_hybrid_runtime.input_block_pairs == [(0, None), (1, None)]
        assert len(no_hybrid_runtime.groups) == 2
        assert no_hybrid_runtime.block_evaluate_count == 2
        assert float((result - reference).abs().max().item()) <= 1.0e-5
        assert float((no_hybrid_result - reference).abs().max().item()) <= 1.0e-5
        assert float((result - no_hybrid_result).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()


def test_u22_tconv_provider_splits_one_plane_across_many_ciphertexts_on_python_backend() -> None:
    _init_python_scheme(logn=8)
    try:
        torch.manual_seed(23)
        module = ConvTranspose2d(
            in_channels=8,
            out_channels=4,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
        )
        module.eval()
        module.init_orion_params()
        module.input_shape = torch.Size((1, 8, 4, 4))
        module.output_shape = torch.Size((1, 4, 8, 8))
        module.input_gap = 4
        module.output_gap = 2
        module.fhe_input_shape = torch.Size((1, 1, 16, 16))
        module.fhe_output_shape = torch.Size((1, 1, 16, 16))
        module.set_level(len(scheme.params.get_logq()) - 1)

        runtime = TconvK2S2PythonRuntimeExecutor(module=module, output_node_id="split_plane_tconv")
        assert runtime.supports_scheme(scheme) is True

        torch.manual_seed(29)
        x = torch.randn(tuple(int(v) for v in module.input_shape), dtype=torch.float32)
        source = _encode_input(module, x[0])
        out = runtime(source)
        result = _decode_output(module, out["split_plane_tconv"])
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

        assert len(source.ids) == 2
        assert runtime.input_block_count == 2
        assert runtime.output_block_count == 2
        assert len(out["split_plane_tconv"].ids) == 2
        assert runtime.input_block_pairs == [(0, 1)]
        assert len(runtime.groups) == 1
        assert runtime.block_evaluate_count == 1
        assert float((result - reference).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()
