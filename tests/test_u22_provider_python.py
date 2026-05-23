from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.auto_bootstrap import BootstrapSolver
from orion.core.network_dag import NetworkDAG
from orion.core.orion import _region_first_mode_options, scheme
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import U22CompileRegistry
from orion.experimental.cir.hybrid_schedule import mark_hybrid_schedule_padding_allowed
from orion.experimental.u22_phase1 import LayoutPolicyProviderRuntimeExecutor, TconvK2S2PythonRuntimeExecutor
from orion.models.unet import UNet22
from orion.nn.linear import Conv2d, ConvTranspose2d
from orion.nn.module import Module


DATASET_SPECS = {
    "tiny": {"image_size": 64, "logn": 15, "input_channels": 3},
    "imagenet": {"image_size": 256, "logn": 16, "input_channels": 3},
    "montgomery_lung_64": {"image_size": 64, "logn": 15, "input_channels": 1},
    "kvasir_polyp_256": {"image_size": 256, "logn": 16, "input_channels": 3},
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
            assert getattr(runtime.executor, "use_ct_pt_hybrid_packing", False) is False
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
    assert imagenet_opts["u22_allowed_nodes"] == ("up1", "up2", "up3", "up4")
    assert imagenet_opts["u22_conv_kernels"] is True

    nohybrid_opts = _region_first_mode_options("u22_64_base32_ir_false")
    assert nohybrid_opts["enabled"] is True
    assert nohybrid_opts["is_u22_phase1"] is True
    assert nohybrid_opts["u22_allowed_nodes"] == ("up1", "up2", "up3", "up4")
    assert nohybrid_opts["u22_conv_kernels"] is True
    assert "u22_disable_real_imag_hybrid" not in nohybrid_opts

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
        assert audit["graph_audit"]["selected_generic_conv_count"] == 3

        for node_name in ("enc1a", "enc1b", "dec1a", "dec2b"):
            runtime = getattr(dag.nodes[str(node_name)]["module"], "region_runtime", None)
            assert runtime is not None
            assert runtime.strategy.startswith("u22_native_halo_stripe_no_ri_conv_same_shape")
            assert type(runtime.executor).__name__ == "LayoutPolicyProviderRuntimeExecutor"
            assert type(runtime.executor.base_executor).__name__ == "HaloLocalConvRuntimeExecutor"
            assert runtime.executor.native_halo_input is bool(runtime.plan.get("native_halo_provider", False))
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
        assert audit["graph_audit"]["selected_generic_conv_count"] == 4
        assert dag.nodes["bottleneckb"]["module"].region_runtime.stage == "single_block_conv"
        assert dag.nodes["dec1b"]["module"].region_runtime.stage == "conv_same_shape"
    finally:
        scheme.delete_scheme()


def test_u22_registry_provider_default_is_no_hybrid() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor

    opts = _region_first_mode_options("u22_64_base32_ir_false")
    _init_python_scheme(logn=int(DATASET_SPECS["tiny"]["logn"]))
    try:
        dag = _prepared_dag(dataset="tiny", base_channels=32)
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=opts["u22_allowed_nodes"],
            enable_conv_kernels=bool(opts["u22_conv_kernels"]),
            layout_policy=str(opts["u22_layout_policy"]),
        )
        audit = registry.attach_to_dag(dag)
        assert "use_real_imag_hybrid" not in audit["graph_audit"]
        assert audit["executable_region_count"] == 26
        attached_modules = [
            dag.nodes[str(row["node"])]["module"]
            for row in audit["attached"]
        ]
        assert attached_modules
        for module in attached_modules:
            runtime = getattr(module, "region_runtime", None)
            executor = getattr(runtime, "executor", None)
            assert executor is not None
            assert getattr(executor, "use_ct_pt_hybrid_packing", False) is False
    finally:
        scheme.delete_scheme()


def test_u22_256_no_hybrid_same_shape_metadata_does_not_use_r34_hardcoded_plan() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
    from orion.experimental.cir.native_halo_conv2d import NativeHaloStripeNoRIConvExecutor
    from orion.experimental.u22_phase1 import LayoutPolicyProviderRuntimeExecutor

    opts = _region_first_mode_options("u22_256_base32_nohybrid")
    _init_python_scheme(logn=int(DATASET_SPECS["kvasir_polyp_256"]["logn"]))
    try:
        dag = _prepared_dag(dataset="kvasir_polyp_256", base_channels=32)
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=opts["u22_allowed_nodes"],
            enable_conv_kernels=bool(opts["u22_conv_kernels"]),
            layout_policy=str(opts["u22_layout_policy"]),
        )
        registry.attach_to_dag(dag)

        executor = dag.nodes["enc1a"]["module"].region_runtime.executor
        assert isinstance(executor, LayoutPolicyProviderRuntimeExecutor)
        assert executor.native_halo_input is True
        assert len(executor.relayout_rows) == 0
        assert len(executor.native_physical_relayout_rows) == 0
        assert executor.native_input_rows[0]["source"] == "x"
        assert executor.native_input_rows[0]["physical_layout"] == "native_source_stripe"

        base = executor.base_executor
        assert isinstance(base, HaloLocalConvRuntimeExecutor)
        assert bool(base.use_ct_pt_hybrid_packing) is False
        assert base.force_input_pair is False
        assert isinstance(base.delegate, NativeHaloStripeNoRIConvExecutor)
        assert base.same_shape_spec is None
        assert base.delegate.spec.family_label.startswith("native_halo_")
        assert base.native_halo_input_capable is True
        assert base.native_halo_output_capable is True
        assert base.delegate.native_halo_input_capable is True
        assert base.delegate.native_halo_output_capable is True

        metadata = executor.compile_cache_metadata()
        assert metadata["layout_policy_wrapper"]["runtime_lowering"] == "provider_executable+native_halo_layout"
        assert metadata["layout_policy_wrapper"]["native_halo_provider"] is True
        assert metadata["layout_policy_wrapper"]["relayout_edge_count"] == 0
        assert metadata["layout_policy_wrapper"]["native_physical_relayout_edge_count"] == 0
        assert metadata["native_halo_conv2d_plan"]["spec"]["input_alpha"] == 1
        assert metadata["native_halo_conv2d_plan"]["spec"]["input_beta"] >= 0
        assert metadata["native_halo_conv2d_plan"]["output_storage_layout"] in {
            "native_halo_stripe",
            "tight_compact",
        }
        assert metadata["delegate_kind"] == "NativeHaloStripeNoRIConvExecutor"
        assert metadata["input_relayout"] == {}
        assert metadata["output_relayout"] == {}
        assert metadata["relayout_sparse_lt_tasks"] == 0

        enc1a_executor = dag.nodes["enc1a"]["module"].region_runtime.executor
        assert isinstance(enc1a_executor, LayoutPolicyProviderRuntimeExecutor)
        enc1a_base = enc1a_executor.base_executor
        assert isinstance(enc1a_base, HaloLocalConvRuntimeExecutor)
        assert isinstance(enc1a_base.delegate, NativeHaloStripeNoRIConvExecutor)
        enc1a_metadata = enc1a_executor.compile_cache_metadata()
        enc1a_plan = enc1a_metadata["native_halo_conv2d_plan"]
        assert enc1a_plan["source_channel_tile"] == 2
        assert enc1a_plan["target_channel_tile"] == 2
        assert enc1a_plan["input_ct_count"] == 10
        assert enc1a_plan["output_ct_count"] > 0
        assert enc1a_plan["submatrix_program_count"] >= enc1a_plan["sharing_group_count"]
        assert enc1a_plan["sharing_group_count"] > 0
        assert enc1a_plan["cb_shared_rotations"] > 0
    finally:
        scheme.delete_scheme()


def test_u22_256_base8_model_input_encodes_native_halo_source_tiles() -> None:
    from tools.run_lattigo_e2e_compare import (
        _encrypt_native_halo_model_input,
        _model_input_native_halo_plan,
    )

    opts = _region_first_mode_options("u22_256_base8")
    _init_python_scheme(logn=int(DATASET_SPECS["kvasir_polyp_256"]["logn"]))
    try:
        dag = _prepared_dag(dataset="kvasir_polyp_256", base_channels=8)
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=opts["u22_allowed_nodes"],
            enable_conv_kernels=bool(opts["u22_conv_kernels"]),
            layout_policy=str(opts["u22_layout_policy"]),
        )
        registry.attach_to_dag(dag)
        root = torch.nn.Module()
        root.enc1a = dag.nodes["enc1a"]["module"]

        native_input = _model_input_native_halo_plan(root)
        assert native_input is not None
        assert native_input["node"] == "enc1a"
        assert native_input["native_executor"] == "NativeHaloStripeNoRIConvExecutor"
        assert int(native_input["plan"].input_ct_count) == 10

        x = torch.randn((1, 3, 256, 256), dtype=torch.float32)
        ct = _encrypt_native_halo_model_input(
            x,
            len(scheme.params.get_logq()) - 1,
            native_input,
        )
        try:
            assert len(ct.ids) == 10
            assert tuple(int(value) for value in ct.on_shape) == (10, int(scheme.params.get_slots()))
        finally:
            ct.release()
    finally:
        scheme.delete_scheme()


def test_native_physical_relayout_honors_source_layout_offsets() -> None:
    from orion.experimental.cir.native_halo_conv2d import (
        NativeHaloConv2DSpec,
        NativeHaloRelayoutKernel,
        native_halo_conv2d_plan,
        native_halo_source_plaintext_blocks_from_nchw,
    )

    _init_python_scheme(logn=8)
    try:
        spec = NativeHaloConv2DSpec(
            family_label="native_relayout_source_layout_test",
            c_in=1,
            h_in=4,
            w_in=4,
            c_out=1,
            h_out=4,
            w_out=4,
            gap_in=1,
            gap_out=1,
            kernel=3,
            stride=1,
            pad=1,
            slot_count=int(scheme.params.get_slots()),
            input_alpha=1,
            input_beta=1,
        )
        plan = native_halo_conv2d_plan(spec)
        level = len(scheme.params.get_logq()) - 1
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        expected = torch.cat(native_halo_source_plaintext_blocks_from_nchw(x, plan)).to(dtype=torch.float32)

        for source_layout, source_tensor in (
            ({"alpha": 0, "beta": 0, "gap": 1, "tile_count": 1}, x),
            (
                {"alpha": 1, "beta": 1, "gap": 1, "tile_count": 1},
                torch.cat([x[:, :, :1, :], x, x[:, :, -1:, :]], dim=2),
            ),
        ):
            kernel = NativeHaloRelayoutKernel(
                plan=plan,
                direction="compact_to_native",
                name="native_relayout_source_layout_test",
                output_shape=torch.Size([int(plan.input_ct_count), int(spec.slot_count)]),
                fhe_output_shape=torch.Size([int(plan.input_ct_count), int(spec.slot_count)]),
                source_layout=dict(source_layout),
            )
            kernel.compile(scheme, level=level)
            out = kernel.apply(scheme.encrypt(scheme.encode(source_tensor.reshape(-1), level)))
            decoded = out.decrypt().decode().detach().cpu().to(dtype=torch.float32).reshape(-1)
            assert float((decoded[: expected.numel()] - expected).abs().max().item()) <= 1.0e-5
            kernel.cleanup(getattr(scheme, "backend", None))
    finally:
        scheme.delete_scheme()


def test_u22_256_base8_routes_flat_halo_producers_to_native_compact_source() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
    from orion.experimental.cir.native_halo_conv2d import NativeHaloStripeNoRIConvExecutor

    opts = _region_first_mode_options("u22_256_base8")
    _init_python_scheme(logn=int(DATASET_SPECS["kvasir_polyp_256"]["logn"]))
    try:
        dag = _prepared_dag(dataset="kvasir_polyp_256", base_channels=8)
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=opts["u22_allowed_nodes"],
            enable_conv_kernels=bool(opts["u22_conv_kernels"]),
            layout_policy=str(opts["u22_layout_policy"]),
        )
        registry.attach_to_dag(dag)

        enc1b = dag.nodes["enc1b"]["module"].region_runtime.executor
        assert isinstance(enc1b, LayoutPolicyProviderRuntimeExecutor)
        assert len(enc1b.native_physical_relayout_rows) == 0

        enc2a = dag.nodes["enc2a"]["module"].region_runtime.executor
        assert isinstance(enc2a, LayoutPolicyProviderRuntimeExecutor)
        assert len(enc2a.relayout_rows) == 0
        assert len(enc2a.native_physical_relayout_rows) == 0
        assert len(enc2a.native_input_rows) == 0
        assert len(enc2a.compact_align_shared_rows) == 1
        row = enc2a.compact_align_shared_rows[0]
        assert row["source"] == "pool1"
        assert row["target"] == "enc2a"
        assert row["layout_mode"] == "compact_align_shared"
        assert row["physical_layout"] == "packed_compact"
        assert bool(row["consumer_fused_relayout"]) is True
        assert isinstance(enc2a.base_executor, HaloLocalConvRuntimeExecutor)
        assert isinstance(enc2a.base_executor.delegate, NativeHaloStripeNoRIConvExecutor)
        assert enc2a.compact_source_rows

    finally:
        scheme.delete_scheme()


def test_native_halo_stripe_provider_honors_dp_input_and_output_halo_layout() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
    from orion.experimental.cir.native_halo_conv2d import native_halo_source_plaintext_blocks_from_nchw
    from orion.experimental.u22_phase1 import LayoutPolicyProviderRuntimeExecutor

    _init_python_scheme(logn=15)
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
        conv.name = "native_halo_toy_conv"
        level = len(scheme.params.get_logq()) - 1
        conv.set_level(level)
        conv.set_depth(2)

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
        executor.assigned_level = level
        executor.assigned_depth = 3

        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 10.0
        native_plan = executor._native_halo_plan()
        ids = []
        for block in native_halo_source_plaintext_blocks_from_nchw(x, native_plan):
            block_ct = scheme.encrypt(scheme.encode(block, level))
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

        metadata = executor.compile_cache_metadata()
        assert executor.last_runtime_io["runtime_lowering"] == "provider_executable+native_halo_layout"
        assert executor.last_runtime_io["native_halo_provider"] is True
        assert executor.last_runtime_io["output_relayout_edge_count"] == 0
        assert metadata["input_relayout"] == {}
        assert metadata["output_relayout"] == {}
        assert metadata["relayout_sparse_lt_tasks"] == 0
        assert executor.last_runtime_io["internal_input_relayout"] is False
        assert executor.last_runtime_io["internal_output_relayout"] is False
        assert tuple(int(value) for value in out.on_shape) == (1, int(scheme.params.get_slots()))
        assert float((decoded - reference).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()


def test_native_halo_provider_uses_tight_compact_output_when_output_halo_is_zero() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
    from orion.experimental.cir.native_halo_conv2d import native_halo_source_plaintext_blocks_from_nchw
    from orion.experimental.u22_phase1 import LayoutPolicyProviderRuntimeExecutor

    _init_python_scheme(logn=15)
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
        conv.name = "native_halo_compact_output_toy_conv"
        level = len(scheme.params.get_logq()) - 1
        conv.set_level(level)
        conv.set_depth(2)

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
                        "physical_layout": "packed_compact",
                        "selected_layout": {"alpha": 0, "beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
                    }
                ],
            },
        )
        executor.assigned_level = level
        executor.assigned_depth = 3

        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 10.0
        native_plan = executor._native_halo_plan()
        ids = []
        for block in native_halo_source_plaintext_blocks_from_nchw(x, native_plan):
            block_ct = scheme.encrypt(scheme.encode(block, level))
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
            .reshape(-1)[:16]
            .reshape(1, 1, 4, 4)
        )
        x_halo = torch.cat([x[:, :, :1, :], x, x[:, :, -1:, :]], dim=2)
        reference = F.conv2d(x_halo, conv.on_weight.detach(), conv.on_bias.detach(), padding=1)[:, :, 1:5, :]

        metadata = executor.compile_cache_metadata()
        assert tuple(int(value) for value in out.on_shape) == (1, 1, 4, 4)
        assert executor.last_runtime_io["native_output_storage_layout"] == "tight_compact"
        assert metadata["native_halo_conv2d_plan"]["output_storage_layout"] == "tight_compact"
        assert int(metadata["native_halo_conv2d_plan"]["output_ct_count"]) == 1
        assert float((decoded - reference).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()


def test_native_halo_provider_accepts_compact_source_layout_without_input_pair_fallback() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
    from orion.experimental.cir.native_halo_conv2d import NativeHaloStripeNoRIConvExecutor
    from orion.experimental.u22_phase1 import LayoutPolicyProviderRuntimeExecutor

    _init_python_scheme(logn=15)
    try:
        conv = Conv2d(2, 2, kernel_size=3, padding=1, bias=True)
        conv.weight.data = torch.arange(36, dtype=torch.float32).reshape(2, 2, 3, 3) / 100.0 - 0.1
        conv.bias.data = torch.tensor([0.125, -0.375], dtype=torch.float32)
        conv.init_orion_params()
        conv.input_shape = torch.Size((1, 2, 4, 4))
        conv.output_shape = torch.Size((1, 2, 4, 4))
        conv.fhe_input_shape = torch.Size((1, 2, 4, 4))
        conv.fhe_output_shape = torch.Size((1, 2, 4, 4))
        conv.input_gap = 1
        conv.output_gap = 1
        conv.name = "native_halo_compact_source_toy_conv"
        level = len(scheme.params.get_logq()) - 1
        conv.set_level(level)
        conv.set_depth(2)

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
                        "shape": [1, 2, 4, 4],
                        "fhe_shape": [1, 2, 4, 4],
                        "relayout": False,
                        "layout_mode": "halo_local",
                        "physical_layout": "packed_compact",
                        "required_layout": {"alpha": 1, "beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                        "source_layout": {"alpha": 0, "beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
                        "selected_layout": {"alpha": 0, "beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
                    }
                ],
                "node_layouts": [
                    {
                        "node": "conv",
                        "shape": [1, 2, 4, 4],
                        "fhe_shape": [1, 2, 4, 4],
                        "output_relayout": False,
                        "physical_layout": "packed_compact",
                        "selected_layout": {"alpha": 0, "beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
                    }
                ],
            },
        )
        executor.assigned_level = level
        executor.assigned_depth = 2

        x = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4) / 10.0
        out = executor(scheme.encrypt(scheme.encode(x, level)))["conv"]
        decoded = (
            out.decrypt()
            .decode()
            .detach()
            .cpu()
            .to(dtype=torch.float32)
            .reshape(-1)[:32]
            .reshape(1, 2, 4, 4)
        )
        x_halo = torch.cat([x[:, :, :1, :], x, x[:, :, -1:, :]], dim=2)
        reference = F.conv2d(x_halo, conv.on_weight.detach(), conv.on_bias.detach(), padding=1)[:, :, 1:5, :]
        metadata = executor.compile_cache_metadata()

        assert isinstance(executor.base_executor, HaloLocalConvRuntimeExecutor)
        assert isinstance(executor.base_executor.delegate, NativeHaloStripeNoRIConvExecutor)
        assert executor.last_runtime_io["runtime_lowering"] == "provider_executable+compact_layout"
        assert executor.last_runtime_io["provider_executor"] == "HaloLocalConvRuntimeExecutor"
        assert executor.last_runtime_io["delegate_executor"] == "NativeHaloStripeNoRIConvExecutor"
        assert executor.last_runtime_io["input_physical_layout"] == "packed_compact"
        assert executor.last_runtime_io["runtime_input_ct_count"] == 1
        assert executor.last_runtime_io["internal_input_relayout"] is False
        assert metadata["input_physical_layout"] == "packed_compact"
        assert metadata["runtime_input_ct_count"] == 1
        assert float((decoded - reference).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()


def test_u22_64_benchmark_no_hybrid_helper_keeps_same_shape_on_bounded_delegate() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
    from orion.experimental.cir.native_halo_conv2d import NativeHaloStripeNoRIConvExecutor
    from orion.experimental.u22_phase1 import LayoutPolicyProviderRuntimeExecutor
    from tools import benchmark_node_specific_lattigo_provider_vs_dense as bench

    bench._init_scheme("u22_64_base32", backend="python")
    try:
        dag, _audit = bench._prepare_dag("u22_64_base32", provider=True)
        module = dag.nodes["enc1b"]["module"]
        executor = module.region_runtime.executor
        assert isinstance(executor, LayoutPolicyProviderRuntimeExecutor)
        assert isinstance(executor.base_executor, HaloLocalConvRuntimeExecutor)
        assert executor.base_executor.force_input_pair is False
        assert bool(executor.base_executor.use_ct_pt_hybrid_packing) is False

        no_hybrid_audit = bench._apply_provider_no_hybrid_ablation("u22_64_base32", module)

        executor = module.region_runtime.executor
        assert isinstance(executor, LayoutPolicyProviderRuntimeExecutor)
        base = executor.base_executor
        delegate = base.delegate
        assert no_hybrid_audit["status"] == "ok"
        assert no_hybrid_audit["mode"] == "native_halo_stripe_no_ri"
        assert no_hybrid_audit["executor"] == "LayoutPolicyProviderRuntimeExecutor"
        assert no_hybrid_audit["base_executor"] == "HaloLocalConvRuntimeExecutor"
        assert no_hybrid_audit["delegate_executor"] == "NativeHaloStripeNoRIConvExecutor"
        assert bool(base.use_ct_pt_hybrid_packing) is False
        assert isinstance(delegate, NativeHaloStripeNoRIConvExecutor)
    finally:
        bench._cleanup_scheme()


@pytest.mark.parametrize("network", ("u22_64_base32", "u22_256_base32"))
def test_node_specific_benchmark_u22_provider_helper_uses_full_default_provider_mode(network: str) -> None:
    from tools import benchmark_node_specific_lattigo_provider_vs_dense as bench

    bench._init_scheme(str(network), backend="python")
    try:
        _dag, audit = bench._prepare_dag(str(network), provider=True)
        graph = dict(audit["graph_audit"])
        halo_edges = [
            row
            for row in graph["layout_policy_edge_layouts"]
            if int(dict(row.get("selected_layout", {})).get("alpha", 0)) > 0
            or int(dict(row.get("selected_layout", {})).get("beta", 0)) > 0
        ]

        assert audit["attached_count"] == 26
        assert audit["executable_region_count"] == 26
        assert graph["selected_tconv_count"] == 4
        assert graph["selected_conv_count"] == 18
        assert graph["selected_pool_count"] == 4
        assert graph["enable_conv_kernels"] is True
        assert graph["layout_policy"] == "dp"
        assert graph["layout_policy_edge_layout_count"] == 34
        expected_halo_edges = {
            "u22_64_base32": 0,
            "u22_256_base32": 4,
        }[str(network)]
        expected_relayouts = {
            "u22_64_base32": 0,
            "u22_256_base32": 0,
        }[str(network)]
        assert len(halo_edges) == expected_halo_edges
        assert int(graph["layout_policy_relayout_edge_count"]) == expected_relayouts
        assert int(graph["layout_policy_output_relayout_node_count"]) == 0
        assert int(graph["layout_policy_summary"]["relayout_depth_estimate"]) == expected_relayouts
    finally:
        bench._cleanup_scheme()


def test_u22_256_base8_provider_solver_depth_covers_native_halo_relayout() -> None:
    _init_python_scheme(logn=int(DATASET_SPECS["kvasir_polyp_256"]["logn"]))
    try:
        dag = _prepared_dag(dataset="kvasir_polyp_256", base_channels=8)
        opts = _region_first_mode_options("u22_256_base8")
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=opts["u22_allowed_nodes"],
            enable_conv_kernels=bool(opts["u22_conv_kernels"]),
            layout_policy=str(opts["u22_layout_policy"]),
        )
        registry.attach_to_dag(dag)

        dag.find_residuals()
        solver = BootstrapSolver(
            SimpleNamespace(),
            dag,
            l_eff=int(len(scheme.params.get_logq()) - 1),
        )
        solver.solve()

        mismatches = []
        for node_name in dag.topological_sort():
            module = dag.nodes[node_name].get("module")
            group = getattr(module, "region_runtime", None) if module is not None else None
            executor = getattr(group, "executor", None) if group is not None else None
            if not isinstance(executor, LayoutPolicyProviderRuntimeExecutor):
                continue
            relayout_depth = (
                (len(executor.relayout_rows) if bool(executor.native_halo_input) else 2 * len(executor.relayout_rows))
                + len(executor.native_physical_relayout_rows)
                + len(executor.output_relayout_rows)
            )
            required_level = int(relayout_depth + max(0, int(group.depth) - int(relayout_depth)))
            level = int(getattr(module, "level"))
            if int(level) < int(required_level):
                mismatches.append((str(node_name), int(level), int(required_level), int(group.depth)))

        dec1a = dag.nodes["dec1a"]["module"]
        assert not mismatches
        assert int(dec1a.level) >= int(dec1a.region_runtime.depth)
        assert int(dec1a.region_runtime.solver_depth) == int(dec1a.region_runtime.depth)
    finally:
        scheme.delete_scheme()


def test_u22_256_base32_provider_mode_skips_dense_tconv_pack_for_all_decoder_nodes(monkeypatch) -> None:
    def fail_pack_conv_transpose2d(*_args, **_kwargs):
        raise AssertionError("u22_256_base32 provider mode must not dense-pack decoder ConvTranspose2d nodes")

    monkeypatch.setattr(packing, "pack_conv_transpose2d", fail_pack_conv_transpose2d)
    opts = _region_first_mode_options("u22_256_base32")
    _init_python_scheme(logn=int(DATASET_SPECS["kvasir_polyp_256"]["logn"]))
    try:
        dag = _prepared_dag(dataset="kvasir_polyp_256", base_channels=32)
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=opts["u22_allowed_nodes"],
            enable_conv_kernels=bool(opts["u22_conv_kernels"]),
        )
        audit = registry.attach_to_dag(dag)
        _set_compile_level(dag)

        attached = {row["node"] for row in audit["attached"]}
        assert set(DECODER_TCONV_NODES).issubset(attached)
        assert audit["graph_audit"]["selected_tconv_count"] == 4
        assert audit["graph_audit"]["allowed_nodes"] == ["up1", "up2", "up3", "up4"]
        for node_name in DECODER_TCONV_NODES:
            module = dag.nodes[str(node_name)]["module"]
            assert getattr(module, "region_first_skip_dense_pack", False) is True
            module.generate_diagonals(last=False)
            assert module.diagonals == {}
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
        executor = getattr(runtime, "executor")
        if (
            int(getattr(executor, "output_block_count", 0)) == 1
            and int(getattr(executor, "output_total_slots", 0)) < int(scheme.params.get_slots())
        ):
            assert int(getattr(executor, "output_fold_rotations", 0)) > 0

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


def test_u22_tconv_provider_fuses_output_beta_relayout_on_python_backend() -> None:
    _init_python_scheme(logn=8)
    try:
        module = ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
        )
        module.eval()
        module.init_orion_params()
        module.on_weight = torch.ones_like(module.on_weight)
        module.on_bias = torch.zeros_like(module.on_bias)
        module.input_shape = torch.Size((1, 1, 2, 2))
        module.output_shape = torch.Size((1, 1, 4, 4))
        module.input_gap = 2
        module.output_gap = 1
        module.fhe_input_shape = torch.Size((1, 1, 4, 4))
        module.fhe_output_shape = torch.Size((1, 1, 5, 4))
        module.layout_policy_output_layout = {"alpha": 0, "beta": 1, "gap": 1}
        module.layout_policy_output_materialization = "fused_relayout"
        module.set_level(len(scheme.params.get_logq()) - 1)

        runtime = TconvK2S2PythonRuntimeExecutor(
            module=module,
            output_node_id="synthetic_tconv_fused_output_beta",
        )
        x = torch.arange(1, 5, dtype=torch.float32).reshape(1, 2, 2)
        out = runtime(_encode_input(module, x))["synthetic_tconv_fused_output_beta"]
        decoded = out.decrypt().decode().detach().cpu()
        if torch.is_complex(decoded):
            decoded = decoded.real
        packed = decoded.to(dtype=torch.float32).flatten()[:20].reshape(1, 1, 5, 4)
        core = F.conv_transpose2d(
            x.unsqueeze(0),
            module.on_weight.detach().to(dtype=torch.float32),
            module.on_bias.detach().to(dtype=torch.float32),
            stride=tuple(int(v) for v in module.stride),
            padding=tuple(int(v) for v in module.padding),
            output_padding=tuple(int(v) for v in module.output_padding),
            groups=int(module.groups),
            dilation=tuple(int(v) for v in module.dilation),
        )
        expected = torch.cat([core, core[:, :, -1:, :]], dim=2)

        assert tuple(int(value) for value in out.on_shape) == (1, 1, 5, 4)
        assert float((packed - expected).abs().max().item()) <= 1.0e-5
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

        runtime = TconvK2S2PythonRuntimeExecutor(
            module=module,
            output_node_id="synthetic_u22_tconv",
            use_ct_pt_hybrid_packing=True,
        )
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


def test_u22_tconv_no_hybrid_skips_empty_source_output_block_shells_on_python_backend() -> None:
    _init_python_scheme(logn=10)
    try:
        torch.manual_seed(31)
        module = ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
        )
        module.eval()
        module.init_orion_params()
        module.input_shape = torch.Size((1, 32, 8, 8))
        module.output_shape = torch.Size((1, 16, 16, 16))
        module.input_gap = 4
        module.output_gap = 2
        module.fhe_input_shape = torch.Size((1, 2, 32, 32))
        module.fhe_output_shape = torch.Size((1, 4, 32, 32))
        module.set_level(len(scheme.params.get_logq()) - 1)

        runtime = TconvK2S2PythonRuntimeExecutor(
            module=module,
            output_node_id="synthetic_u22_tconv_no_hybrid_skip_empty",
            use_ct_pt_hybrid_packing=False,
        )
        runtime.compile(scheme)

        assert runtime.input_block_count == 4
        assert runtime.output_block_count == 8
        assert runtime.compiled_transform_count == 16
        assert runtime.skipped_empty_transform_count == 16
        assert sum(len(group.transforms) for group in runtime.groups) == runtime.compiled_transform_count
    finally:
        scheme.delete_scheme()


def test_u22_tconv_no_tile_family_sharing_splits_output_tiles_on_python_backend() -> None:
    _init_python_scheme(logn=10)
    try:
        torch.manual_seed(32)
        module = ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
        )
        module.eval()
        module.init_orion_params()
        module.input_shape = torch.Size((1, 32, 8, 8))
        module.output_shape = torch.Size((1, 16, 16, 16))
        module.input_gap = 4
        module.output_gap = 2
        module.fhe_input_shape = torch.Size((1, 2, 32, 32))
        module.fhe_output_shape = torch.Size((1, 4, 32, 32))
        module.set_level(len(scheme.params.get_logq()) - 1)

        runtime = TconvK2S2PythonRuntimeExecutor(
            module=module,
            output_node_id="synthetic_u22_tconv_no_tile_family",
            use_ct_pt_hybrid_packing=True,
            disable_tile_family_sharing=True,
        )
        runtime.compile(scheme)

        assert runtime.use_ct_pt_hybrid_packing is True
        assert runtime.disable_tile_family_sharing is True
        assert runtime.input_block_count == 4
        assert runtime.output_block_count == 8
        assert runtime.compiled_transform_count == 16
        assert len(runtime.groups) == 16
        assert all(len(group.transforms) == 1 for group in runtime.groups)
    finally:
        scheme.delete_scheme()


def test_u22_tconv_hybrid_rejects_mismatched_adjacent_source_schedules() -> None:
    _init_python_scheme(logn=8)
    try:
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

        runtime = TconvK2S2PythonRuntimeExecutor(
            module=module,
            output_node_id="mismatch_tconv",
            use_ct_pt_hybrid_packing=True,
        )

        def fake_build_source_block_transforms(*, scheme, level: int, source_block: int):
            slots = int(scheme.params.get_slots())
            diag_key = 0 if int(source_block) == 0 else 1
            transforms = [
                SimpleNamespace(
                    name=f"fake_src{int(source_block)}_out{int(output_block)}",
                    diagonals={(0, 0): {int(diag_key): torch.ones((int(slots),), dtype=torch.float32)}},
                    level=int(level),
                    scheme=scheme,
                    fhe_output_shape=torch.Size([1, int(slots)]),
                    output_shape=torch.Size([1, int(slots)]),
                )
                for output_block in range(int(runtime.output_block_count))
            ]
            return transforms, 0

        runtime._build_source_block_transforms = fake_build_source_block_transforms
        runtime.compile(scheme)

        assert runtime.input_block_count == 2
        assert runtime.output_block_count == 2
        assert runtime.hybrid_pair_count == 0
        assert runtime.hybrid_pair_rejected_count == 1
        assert runtime.input_block_pairs == [(0, None), (1, None)]
        assert runtime.complex_input_block_flags == [False, False]
        assert len(runtime.groups) == 2
        metadata = runtime.compile_cache_metadata()
        assert metadata["hybrid_pair_rejected_count"] == 1
        assert all(group["complex_input_block"] is False for group in metadata["groups_by_input_unit"])
        assert all(group["hybrid_pair_reject_reason"] for group in metadata["groups_by_input_unit"])
    finally:
        scheme.delete_scheme()


def test_u22_tconv_global_layout_materializes_adjacent_source_schedules() -> None:
    _init_python_scheme(logn=8)
    try:
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

        runtime = TconvK2S2PythonRuntimeExecutor(
            module=module,
            output_node_id="global_layout_tconv",
            use_ct_pt_hybrid_packing=True,
        )

        def fake_build_source_block_transforms(*, scheme, level: int, source_block: int):
            slots = int(scheme.params.get_slots())
            diag_key = 0 if int(source_block) == 0 else 1
            transforms = [
                mark_hybrid_schedule_padding_allowed(
                    SimpleNamespace(
                        name=f"fake_src{int(source_block)}_out{int(output_block)}",
                        diagonals={(0, 0): {int(diag_key): torch.ones((int(slots),), dtype=torch.float32)}},
                        level=int(level),
                        scheme=scheme,
                        fhe_output_shape=torch.Size([1, int(slots)]),
                        output_shape=torch.Size([1, int(slots)]),
                    ),
                    family="global_layout_tconv_family",
                )
                for output_block in range(int(runtime.output_block_count))
            ]
            return transforms, 0

        runtime._build_source_block_transforms = fake_build_source_block_transforms
        runtime.compile(scheme)

        assert runtime.input_block_count == 2
        assert runtime.hybrid_pair_layout_strategy == "global_schedule_layout"
        assert runtime.hybrid_pair_count == 1
        assert runtime.hybrid_pair_rejected_count == 0
        assert runtime.hybrid_pair_schedule_padded_count == 1
        assert runtime.input_block_pairs == [(0, 1)]
        assert runtime.complex_input_block_flags == [True]
    finally:
        scheme.delete_scheme()


def test_u22_tconv_hybrid_layout_dp_shifts_pair_boundary_for_strict_schedule() -> None:
    _init_python_scheme(logn=8)
    try:
        module = ConvTranspose2d(
            in_channels=12,
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
        module.input_shape = torch.Size((1, 12, 4, 4))
        module.output_shape = torch.Size((1, 4, 8, 8))
        module.input_gap = 4
        module.output_gap = 2
        module.fhe_input_shape = torch.Size((1, 1, 24, 16))
        module.fhe_output_shape = torch.Size((1, 1, 16, 16))
        module.set_level(len(scheme.params.get_logq()) - 1)

        runtime = TconvK2S2PythonRuntimeExecutor(
            module=module,
            output_node_id="layout_dp_tconv",
            use_ct_pt_hybrid_packing=True,
        )

        def fake_build_source_block_transforms(*, scheme, level: int, source_block: int):
            slots = int(scheme.params.get_slots())
            diag_key = 0 if int(source_block) == 0 else 1
            transforms = [
                SimpleNamespace(
                    name=f"fake_src{int(source_block)}_out{int(output_block)}",
                    diagonals={(0, 0): {int(diag_key): torch.ones((int(slots),), dtype=torch.float32)}},
                    level=int(level),
                    scheme=scheme,
                    fhe_output_shape=torch.Size([1, int(slots)]),
                    output_shape=torch.Size([1, int(slots)]),
                )
                for output_block in range(int(runtime.output_block_count))
            ]
            return transforms, 0

        runtime._build_source_block_transforms = fake_build_source_block_transforms
        runtime.compile(scheme)

        assert runtime.input_block_count == 3
        assert runtime.hybrid_pair_layout_strategy == "strict_schedule_dp"
        assert runtime.hybrid_pair_layout_strict_pair_count == 1
        assert runtime.hybrid_pair_count == 1
        assert runtime.hybrid_pair_schedule_padded_count == 0
        assert runtime.input_block_pairs == [(0, None), (1, 2)]
        assert runtime.complex_input_block_flags == [False, True]
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

        runtime = TconvK2S2PythonRuntimeExecutor(
            module=module,
            output_node_id="split_plane_tconv",
            use_ct_pt_hybrid_packing=True,
        )
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
        assert runtime.complex_input_block_flags == [True]
        assert runtime.hybrid_pair_count == 1
        assert runtime.hybrid_pair_rejected_count == 0
        assert runtime.hybrid_pair_schedule_padded_count == 1
        assert len(runtime.groups) == 1
        assert runtime.block_evaluate_count == 1
        assert float((result - reference).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()
