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
from orion.experimental.u22_phase1 import (
    HaloSupportedTConvRuntimeExecutor,
    LayoutPolicyProviderRuntimeExecutor,
    TconvK2S2PythonRuntimeExecutor,
    _u22_same_shape_conv_module_supported,
    _u22_same_shape_conv_runtime_supported,
)
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


def test_u22_concat_skip_consumer_conv_has_concat_fusion_specs_in_provider_mode() -> None:
    _init_python_scheme(logn=int(DATASET_SPECS["tiny"]["logn"]))
    try:
        dag = _prepared_dag(dataset="tiny", base_channels=4)
        registry = U22CompileRegistry.for_dag(dag, enable_conv_kernels=True)
        registry.attach_to_dag(dag)
        _set_compile_level(dag)

        join = dag.nodes["cat4"]["module"]
        conv = dag.nodes["dec4a"]["module"]

        assert type(join).__name__ == "Concat"
        assert conv.in_channels == 2 * dag.nodes["up4"]["module"].out_channels
        assert len(getattr(conv, "concat_fusion_specs", ()) or ()) == 2
        assert not getattr(conv, "_concat_transform_ids_by_input", [])
    finally:
        scheme.delete_scheme()


@pytest.mark.parametrize("dataset", ("tiny", "imagenet"))
def test_u22_registry_routes_decoder_tconvs_to_halo_supported_provider(dataset: str) -> None:
    _init_python_scheme(logn=int(DATASET_SPECS[str(dataset)]["logn"]))
    try:
        dag = _prepared_dag(dataset=str(dataset))
        registry = U22CompileRegistry.for_dag(dag)
        audit = registry.attach_to_dag(dag)

        assert audit["attached_count"] == len(DECODER_TCONV_NODES)
        assert audit["graph_audit"]["selected_tconv_count"] == len(DECODER_TCONV_NODES)
        assert audit["graph_audit"]["excluded_nodes"] == []
        for node_name in DECODER_TCONV_NODES:
            module = dag.nodes[str(node_name)]["module"]
            runtime = getattr(module, "region_runtime", None)
            assert runtime is not None
            assert runtime.strategy == "halo_supported_tconv"
            assert runtime.materializer == "halo_supported_tconv"
            assert type(runtime.executor).__name__ == "HaloSupportedTConvRuntimeExecutor"
            assert isinstance(runtime.executor, HaloSupportedTConvRuntimeExecutor)
            assert isinstance(runtime.executor.delegate, TconvK2S2PythonRuntimeExecutor)
            assert getattr(module, "region_first_skip_dense_pack", False) is True
    finally:
        scheme.delete_scheme()


def test_u22_registry_can_attach_decoder_tconv_subset() -> None:
    _init_python_scheme(logn=int(DATASET_SPECS["tiny"]["logn"]))
    try:
        dag = _prepared_dag(dataset="tiny")
        registry = U22CompileRegistry.for_dag(dag, allowed_nodes=("up2", "up1"))
        audit = registry.attach_to_dag(dag)

        assert audit["attached_count"] == 2
        assert audit["graph_audit"]["selected_tconv_count"] == 2
        assert audit["graph_audit"]["allowed_nodes"] == ["up2", "up1"]
        filtered = {
            row["node"]
            for row in audit["graph_audit"]["excluded_nodes"]
            if row["reason"] == "u22_ablation_filtered_out"
        }
        assert filtered == {"up4", "up3"}
        for node_name in ("up2", "up1"):
            module = dag.nodes[str(node_name)]["module"]
            runtime = getattr(module, "region_runtime", None)
            assert runtime is not None
            assert runtime.strategy == "halo_supported_tconv"
            assert isinstance(runtime.executor, HaloSupportedTConvRuntimeExecutor)
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

        assert {"enc1b", "dec1a", "dec2b", "pool1", "pool2", "pool3", "pool4"}.issubset(attached)
        assert "up4" in attached
        assert "up3" in attached
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
        for node_name in ("up4", "up3"):
            module = dag.nodes[str(node_name)]["module"]
            runtime = getattr(module, "region_runtime", None)
            assert runtime is not None
            assert runtime.strategy.startswith("halo_supported_tconv")
            assert type(runtime.executor).__name__ == "LayoutPolicyProviderRuntimeExecutor"
            assert isinstance(runtime.executor.base_executor, HaloSupportedTConvRuntimeExecutor)
        assert dag.nodes["pool1"]["module"].region_runtime.stage == "pool_downsample"
        assert dag.nodes["bottleneckb"]["module"].region_runtime.stage == "single_block_conv"
    finally:
        scheme.delete_scheme()


def test_u22_same_shape_provider_accepts_1x1_output_head() -> None:
    conv = Conv2d(64, 1, kernel_size=1, padding=0, bias=True)
    conv.init_orion_params()
    conv.name = "output"
    conv.input_shape = torch.Size((1, 64, 224, 224))
    conv.output_shape = torch.Size((1, 1, 224, 224))
    conv.fhe_input_shape = torch.Size((1, 64, 224, 224))
    conv.fhe_output_shape = torch.Size((1, 1, 224, 224))
    conv.input_gap = 1
    conv.output_gap = 1

    assert _u22_same_shape_conv_module_supported(conv) is True
    assert _u22_same_shape_conv_runtime_supported(conv) is True


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
        for node_name in DECODER_TCONV_NODES:
            module = dag.nodes[str(node_name)]["module"]
            runtime = getattr(module, "region_runtime", None)
            assert runtime is not None
            executor = getattr(runtime, "executor", None)
            base_executor = getattr(executor, "base_executor", executor)
            assert isinstance(base_executor, HaloSupportedTConvRuntimeExecutor)
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
        assert metadata["native_halo_conv2d_plan"]["spec"]["input_top_beta"] == 1
        assert metadata["native_halo_conv2d_plan"]["spec"]["input_bottom_beta"] >= 0
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
            input_top_beta=1,
            input_bottom_beta=1,
        )
        plan = native_halo_conv2d_plan(spec)
        level = len(scheme.params.get_logq()) - 1
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        expected = torch.cat(native_halo_source_plaintext_blocks_from_nchw(x, plan)).to(dtype=torch.float32)

        for source_layout, source_tensor in (
            ({"top_beta": 0, "bottom_beta": 0, "gap": 1, "tile_count": 1}, x),
            (
                {"top_beta": 1, "bottom_beta": 1, "gap": 1, "tile_count": 1},
                torch.cat([torch.zeros_like(x[:, :, :1, :]), x, torch.zeros_like(x[:, :, -1:, :])], dim=2),
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
        assert row["layout_mode"] == "compact_halo_shared"
        assert row["physical_layout"] == "logical_halo_compact"
        assert bool(row["consumer_fused_relayout"]) is True
        assert isinstance(enc2a.base_executor, HaloLocalConvRuntimeExecutor)
        assert isinstance(enc2a.base_executor.delegate, NativeHaloStripeNoRIConvExecutor)
        assert enc2a.compact_source_rows

    finally:
        scheme.delete_scheme()


def test_consumer_fused_compact_source_uses_source_physical_layout_contract() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor

    conv = Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
    conv.init_orion_params()
    conv.input_shape = torch.Size((1, 32, 96, 96))
    conv.output_shape = torch.Size((1, 64, 96, 96))
    conv.fhe_input_shape = torch.Size((1, 8, 192, 192))
    conv.fhe_output_shape = torch.Size((1, 16, 192, 192))
    conv.input_gap = 2
    conv.output_gap = 2
    conv.name = "consumer_fused_contract_enc2a"

    selected = {
        "top_beta": 1,
        "bottom_beta": 9,
        "stride": 1,
        "gap": 2,
        "core_slots": 294912,
        "stored_slots": 325632,
        "tile_count": 10,
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=HaloLocalConvRuntimeExecutor(module=conv, output_node_id="enc2a"),
        output_node_id="enc2a",
        compile_plan={
            "policy": "dp",
            "edge_layouts": [
                {
                    "edge": "pool1->enc2a",
                    "source": "pool1",
                    "target": "enc2a",
                    "op_kind": "conv2d",
                    "shape": [1, 32, 96, 96],
                    "fhe_shape": [1, 8, 192, 192],
                    "required_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 2},
                    "source_layout": dict(selected),
                    "selected_layout": dict(selected),
                    "physical_layout": "logical_halo_compact",
                    "source_physical_layout": "packed_compact",
                    "relayout": False,
                    "layout_mode": "compact_halo_shared",
                    "consumer_fused_relayout": True,
                }
            ],
            "node_layouts": [],
        },
    )

    attrs = executor._native_halo_module_attrs()

    assert attrs["layout_policy_input_physical_layout"] == "packed_compact"
    assert attrs["layout_policy_input_layout"]["top_beta"] == 0
    assert attrs["layout_policy_input_layout"]["bottom_beta"] == 0
    assert attrs["layout_policy_selected_input_layout"]["top_beta"] == 1
    assert attrs["layout_policy_selected_input_layout"]["bottom_beta"] == 9
    assert tuple(int(value) for value in attrs["fhe_input_shape"]) == (1, 8, 192, 192)


def test_fixed_max_fused_compact_source_halo_stays_on_native_provider() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor

    _init_python_scheme(logn=int(DATASET_SPECS["kvasir_polyp_256"]["logn"]))
    try:
        torch.manual_seed(0)
        model = UNet22(
            dataset="kvasir_polyp_256",
            base_channels=8,
            activation="silu",
            silu_degree=7,
        )
        traced = OrionTracer().trace_model(model)
        StatsTracker(traced).propagate(torch.randn((1, 3, 224, 224), dtype=torch.float32))
        dag = NetworkDAG(traced)
        dag.build_dag()
        for node in dag.nodes:
            module = dag.nodes[node]["module"]
            if module is not None and hasattr(module, "init_orion_params"):
                module.init_orion_params()
            if module is not None and hasattr(module, "update_params"):
                module.update_params()

        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=None,
            enable_conv_kernels=True,
            layout_policy="fixed_max_fused",
        )
        assert registry.graph_audit["layout_policy_provider_fallback_nodes"] == []
        registry.attach_to_dag(dag)

        enc1a = dag.nodes["enc1a"]["module"].region_runtime.executor
        assert isinstance(enc1a, LayoutPolicyProviderRuntimeExecutor)
        enc1a.assigned_level = len(scheme.params.get_logq()) - 1
        enc1a.assigned_depth = 1
        enc1a.compile(scheme)
        assert enc1a.base_executor.delegate.native_plan.spec.input_bottom_beta == 0

        executor = dag.nodes["enc3a"]["module"].region_runtime.executor
        assert isinstance(executor, LayoutPolicyProviderRuntimeExecutor)
        assert isinstance(executor.base_executor, HaloLocalConvRuntimeExecutor)
        assert executor._runtime_lowering_label() == "provider_executable+native_halo_output_layout"
        assert executor.compact_source_rows
        assert not executor.native_input_rows
        assert executor.output_relayout_rows == ()
        assert executor.native_physical_relayout_rows == ()
        assert executor.compile_plan["policy"] == "fixed_max_fused"
        assert executor.base_executor.native_halo_output_capable

        executor.assigned_level = len(scheme.params.get_logq()) - 1
        executor.assigned_depth = 1
        executor.compile(scheme)
        assert executor.base_executor.rows > 0
        assert executor.base_executor.cols > 0
        assert executor.base_executor.delegate.native_plan.spec.input_bottom_beta == 22

        enc4a = dag.nodes["enc4a"]["module"].region_runtime.executor
        assert isinstance(enc4a, LayoutPolicyProviderRuntimeExecutor)
        assert isinstance(enc4a.base_executor, HaloLocalConvRuntimeExecutor)
        assert enc4a._runtime_lowering_label() == "provider_executable+native_halo_output_layout"
    finally:
        scheme.delete_scheme()


def test_u22_224_silu7_layout_policies_validate_native_provider_plans() -> None:
    _init_python_scheme(logn=int(DATASET_SPECS["kvasir_polyp_256"]["logn"]))
    try:
        torch.manual_seed(0)
        model = UNet22(
            dataset="kvasir_polyp_256",
            base_channels=8,
            activation="silu",
            silu_degree=7,
        )
        traced = OrionTracer().trace_model(model)
        StatsTracker(traced).propagate(torch.randn((1, 3, 224, 224), dtype=torch.float32))
        dag = NetworkDAG(traced)
        dag.build_dag()
        for node in dag.nodes:
            module = dag.nodes[node]["module"]
            if module is not None and hasattr(module, "init_orion_params"):
                module.init_orion_params()
            if module is not None and hasattr(module, "update_params"):
                module.update_params()

        for policy in ("fixed_max_fused", "eager_fused", "greedy_fused", "dp"):
            registry = U22CompileRegistry.for_dag(
                dag,
                allowed_nodes=None,
                enable_conv_kernels=True,
                layout_policy=policy,
            )
            assert registry.graph_audit["layout_policy_provider_fallback_nodes"] == []
    finally:
        scheme.delete_scheme()


def test_u22_224_silu7_dp_preserves_native_physical_layout_across_activation() -> None:
    from orion.experimental.layout_policy_ablation import build_layout_policy_compile_plan

    _init_python_scheme(logn=int(DATASET_SPECS["kvasir_polyp_256"]["logn"]))
    try:
        torch.manual_seed(0)
        model = UNet22(
            dataset="kvasir_polyp_256",
            base_channels=8,
            activation="silu",
            silu_degree=7,
        )
        traced = OrionTracer().trace_model(model)
        StatsTracker(traced).propagate(torch.randn((1, 3, 224, 224), dtype=torch.float32))
        dag = NetworkDAG(traced)
        dag.build_dag()
        for node in dag.nodes:
            module = dag.nodes[node]["module"]
            if module is not None and hasattr(module, "init_orion_params"):
                module.init_orion_params()
            if module is not None and hasattr(module, "update_params"):
                module.update_params()

        plan = build_layout_policy_compile_plan(dag, policy="dp")
        edges = {str(row["edge"]): dict(row) for row in plan["edge_layouts"]}
        nodes = {str(row["node"]): dict(row) for row in plan["node_layouts"]}

        assert edges["x->enc1a"]["physical_layout"] == "native_source_stripe"
        assert edges["enc1a->enc1a_act"]["physical_layout"] == "native_source_stripe"
        assert nodes["enc1a_act"]["physical_layout"] == "native_source_stripe"
        assert edges["enc1a_act->enc1b"]["layout_mode"] == "native_halo_stripe"
        assert edges["enc1a_act->enc1b"]["physical_layout"] == "native_source_stripe"
        assert edges["enc1a_act->enc1b"]["relayout"] is False
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
                        "source_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                        "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                    }
                ],
                "node_layouts": [
                    {
                        "node": "conv",
                        "shape": [1, 1, 4, 4],
                        "fhe_shape": [1, 1, 4, 4],
                        "output_relayout": False,
                        "physical_layout": "native_source_stripe",
                        "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
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
        reference = F.conv2d(
            x,
            conv.on_weight.detach(),
            conv.on_bias.detach(),
            padding=(2, 1),
        )

        metadata = executor.compile_cache_metadata()
        assert executor.last_runtime_io["runtime_lowering"] == "provider_executable+native_halo_layout"
        assert executor.last_runtime_io["native_halo_provider"] is True
        assert executor.last_runtime_io["output_relayout_edge_count"] == 0
        assert metadata["input_relayout"] == {}
        assert metadata["output_relayout"] == {}
        assert metadata["relayout_sparse_lt_tasks"] == 0
        assert executor.last_runtime_io["internal_input_relayout"] is False
        assert executor.last_runtime_io["internal_output_relayout"] is False
        timing = executor.last_runtime_timing
        assert timing["group_eval_s"] >= 0.0
        assert timing["partial_wrap_s"] >= 0.0
        assert timing["partial_rescale_s"] >= 0.0
        assert timing["partial_accumulate_s"] >= 0.0
        counts = executor.last_runtime_counts
        assert counts["partial_count"] >= counts["target_count"]
        assert counts["partial_rescale_count"] >= 0
        assert counts["partial_accumulate_count"] >= 0
        assert counts["target_count"] >= 1
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
                        "source_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                        "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                    }
                ],
                "node_layouts": [
                    {
                        "node": "conv",
                        "shape": [1, 1, 4, 4],
                        "fhe_shape": [1, 1, 4, 4],
                        "output_relayout": False,
                        "physical_layout": "packed_compact",
                        "selected_layout": {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
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
        reference = F.conv2d(x, conv.on_weight.detach(), conv.on_bias.detach(), padding=1)

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
                        "required_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                        "source_layout": {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
                        "selected_layout": {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
                    }
                ],
                "node_layouts": [
                    {
                        "node": "conv",
                        "shape": [1, 2, 4, 4],
                        "fhe_shape": [1, 2, 4, 4],
                        "output_relayout": False,
                        "physical_layout": "packed_compact",
                        "selected_layout": {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
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
        reference = F.conv2d(x, conv.on_weight.detach(), conv.on_bias.detach(), padding=1)
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


def test_native_halo_compact_source_prunes_zero_blocks_without_dense_pack_speedup() -> None:
    from orion.experimental.cir.native_halo_conv2d import (
        NativeHaloStripeNoRIConvExecutor,
        native_halo_conv2d_spec_from_module,
    )

    _init_python_scheme(logn=6)
    try:
        conv = Conv2d(4, 4, kernel_size=1, padding=0, bias=False)
        conv.weight.data.zero_()
        conv.weight.data[:2, :2, 0, 0] = torch.eye(2)
        conv.weight.data[2:, 2:, 0, 0] = torch.eye(2)
        conv.init_orion_params()
        conv.input_shape = torch.Size((1, 4, 4, 4))
        conv.output_shape = torch.Size((1, 4, 4, 4))
        conv.fhe_input_shape = torch.Size((1, 4, 4, 4))
        conv.fhe_output_shape = torch.Size((1, 4, 4, 4))
        conv.input_gap = 1
        conv.output_gap = 1
        conv.name = "native_halo_compact_source_prune_zero_blocks"
        conv.layout_policy_input_physical_layout = "packed_compact"
        conv.layout_policy_input_layout = {"top_beta": 0, "bottom_beta": 0, "gap": 1, "tile_count": 2}
        conv.layout_policy_output_layout = {"top_beta": 0, "bottom_beta": 0, "gap": 1, "tile_count": 2}
        level = len(scheme.params.get_logq()) - 1
        conv.set_level(level)
        conv.set_depth(1)

        spec = native_halo_conv2d_spec_from_module(conv, output_node_id="conv")
        assert spec is not None
        executor = NativeHaloStripeNoRIConvExecutor(module=conv, spec=spec, output_node_id="conv")
        executor.assigned_level = level
        executor.assigned_depth = 1
        try:
            executor.compile(scheme)
            transform_count = sum(
                len(group.transforms)
                for group in executor.groups_by_input_index.values()
            )

            assert executor.rows == 2
            assert executor.cols == 2
            assert sorted(executor.target_indices_by_input_index) == [0, 1]
            assert transform_count == 2
            assert executor.target_indices_by_input_index[0] == (0,)
            assert executor.target_indices_by_input_index[1] == (1,)
        finally:
            executor.cleanup(getattr(scheme, "backend", None))
    finally:
        scheme.delete_scheme()


def test_input_pair_provider_prunes_zero_blocks_without_dense_pack_speedup() -> None:
    from orion.experimental.cir.transition_pool_provider import InputPairConvRuntimeExecutor

    _init_python_scheme(logn=6)
    try:
        conv = Conv2d(4, 4, kernel_size=1, padding=0, bias=True)
        conv.weight.data.zero_()
        conv.weight.data[:2, :2, 0, 0] = torch.eye(2)
        conv.weight.data[2:, 2:, 0, 0] = torch.eye(2)
        conv.bias.data.zero_()
        conv.init_orion_params()
        conv.input_shape = torch.Size((1, 4, 4, 4))
        conv.output_shape = torch.Size((1, 4, 4, 4))
        conv.fhe_input_shape = torch.Size((1, 4, 4, 4))
        conv.fhe_output_shape = torch.Size((1, 4, 4, 4))
        conv.input_gap = 1
        conv.output_gap = 1
        conv.name = "input_pair_provider_prune_zero_blocks"
        level = len(scheme.params.get_logq()) - 1
        conv.set_level(level)
        conv.set_depth(1)

        dense_diagonals, _output_rotations = packing.pack_conv2d(conv, last=False)
        assert set(dense_diagonals) == {(0, 0), (0, 1), (1, 0), (1, 1)}

        executor = InputPairConvRuntimeExecutor(
            module=conv,
            output_node_id="conv",
            use_ct_pt_hybrid_packing=False,
        )
        executor.assigned_level = level
        executor.assigned_depth = 1
        try:
            executor.compile(scheme)
            transform_count = sum(len(group.transforms) for group in executor.groups_by_pair)

            assert executor.rows == 2
            assert executor.cols == 2
            assert transform_count == 2
            assert executor.input_block_pairs == [(0, None), (1, None)]
            assert executor.row_indices_by_pair == [(0,), (1,)]
        finally:
            executor.cleanup(getattr(scheme, "backend", None))
    finally:
        scheme.delete_scheme()


def test_native_halo_provider_uses_physical_compact_layout_for_raw_input_source() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
    from orion.experimental.cir.native_halo_conv2d import NativeHaloStripeNoRIConvExecutor

    _init_python_scheme(logn=15)
    try:
        conv = Conv2d(2, 2, kernel_size=3, padding=1, bias=True)
        conv.weight.data = torch.arange(36, dtype=torch.float32).reshape(2, 2, 3, 3) / 50.0 - 0.2
        conv.bias.data = torch.tensor([0.25, -0.125], dtype=torch.float32)
        conv.init_orion_params()
        conv.input_shape = torch.Size((1, 2, 4, 4))
        conv.output_shape = torch.Size((1, 2, 4, 4))
        conv.fhe_input_shape = torch.Size((1, 2, 4, 4))
        conv.fhe_output_shape = torch.Size((1, 2, 4, 4))
        conv.input_gap = 1
        conv.output_gap = 1
        conv.name = "native_halo_raw_input_compact_source_toy_conv"
        level = len(scheme.params.get_logq()) - 1
        conv.set_level(level)
        conv.set_depth(2)

        executor = LayoutPolicyProviderRuntimeExecutor(
            base_executor=HaloLocalConvRuntimeExecutor(module=conv, output_node_id="conv"),
            output_node_id="conv",
            compile_plan={
                "policy": "fixed_max_fused",
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
                        "required_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                        "source_layout": {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
                        "selected_layout": {"top_beta": 1, "bottom_beta": 3, "stride": 1, "gap": 1, "tile_count": 1},
                    }
                ],
                "node_layouts": [
                    {
                        "node": "conv",
                        "shape": [1, 2, 4, 4],
                        "fhe_shape": [1, 2, 4, 4],
                        "output_relayout": False,
                        "physical_layout": "packed_compact",
                        "selected_layout": {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
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
        reference = F.conv2d(x, conv.on_weight.detach(), conv.on_bias.detach(), padding=1)
        metadata = executor.compile_cache_metadata()

        assert isinstance(executor.base_executor.delegate, NativeHaloStripeNoRIConvExecutor)
        assert executor.last_runtime_io["input_physical_layout"] == "packed_compact"
        assert metadata["compact_source_layout"]["top_beta"] == 0
        assert metadata["compact_source_layout"]["bottom_beta"] == 0
        assert metadata["native_halo_conv2d_plan"]["spec"]["input_top_beta"] == 0
        assert metadata["native_halo_conv2d_plan"]["spec"]["input_bottom_beta"] == 0
        assert float((decoded - reference).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()


def test_native_halo_executor_loads_cached_manifest_groups(monkeypatch) -> None:
    from orion.experimental.cir import native_halo_conv2d

    module = SimpleNamespace(
        on_weight=torch.zeros((2, 2, 3, 3), dtype=torch.float32),
        on_bias=None,
        input_shape=torch.Size([1, 2, 4, 4]),
        output_shape=torch.Size([1, 2, 4, 4]),
        fhe_input_shape=torch.Size([1, 2, 4, 4]),
        fhe_output_shape=torch.Size([1, 2, 4, 4]),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        input_gap=1,
        output_gap=1,
        layout_policy_input_layout={},
        layout_policy_output_layout={},
    )
    spec = native_halo_conv2d.NativeHaloConv2DSpec(
        family_label="test_cached_native_halo",
        c_in=2,
        h_in=4,
        w_in=4,
        c_out=2,
        h_out=4,
        w_out=4,
        gap_in=1,
        gap_out=1,
        kernel=3,
        stride=1,
        pad=1,
        slot_count=16,
    )
    executor = native_halo_conv2d.NativeHaloStripeNoRIConvExecutor(
        module=module,
        spec=spec,
        output_node_id="conv",
    )
    executor.assigned_level = 3
    executor.load_compile_cache_metadata(
        {
            "rows": 2,
            "cols": 2,
            "groups_by_input_index": [
                {"input_index": 0, "storage_key": "group_10", "target_indices": [0, 1]},
                {"input_index": 1, "storage_key": "group_11", "target_indices": [1]},
            ],
        }
    )

    calls: list[tuple[str, int]] = []

    def _fake_compile_unified(self, _backend):
        calls.append((str(self._storage_key), int(len(self.transforms))))
        self.unified_ids = [int(1000 + len(calls) * 10 + index) for index in range(len(self.transforms))]
        self.is_compiled = True

    def _unexpected_rebuild(*_args, **_kwargs):
        raise AssertionError("load-mode native halo compile should use cached transform shells")

    monkeypatch.setattr(native_halo_conv2d.UnifiedTransformGroup, "compile_unified", _fake_compile_unified)
    monkeypatch.setattr(native_halo_conv2d, "_build_compact_source_conv_transform", _unexpected_rebuild)
    monkeypatch.setattr(native_halo_conv2d, "_build_conv_transform", _unexpected_rebuild)
    monkeypatch.setattr(native_halo_conv2d, "_build_conv_transforms_for_compact_output", _unexpected_rebuild)

    fake_scheme = SimpleNamespace(
        params=SimpleNamespace(
            get_io_mode=lambda: "load",
            get_logq=lambda: [45, 30, 30, 45],
            get_default_scale=lambda: 1 << 30,
        ),
        backend=SimpleNamespace(),
    )

    executor.compile(fake_scheme)

    assert calls == [("group_10", 2), ("group_11", 1)]
    assert sorted(executor.groups_by_input_index) == [0, 1]
    assert executor.target_indices_by_input_index[0] == (0, 1)
    assert executor.target_indices_by_input_index[1] == (1,)
    assert executor.last_runtime_timing["built_transform_count"] == 0.0
    assert executor.last_runtime_timing["compiled_group_count"] == 2.0


def test_compact_source_conv_does_not_consume_global_padding_halo_rows() -> None:
    from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor

    _init_python_scheme(logn=15)
    try:
        conv = Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        conv.weight.data = torch.tensor(
            [[[[0.25, -0.5, 0.75], [1.0, 0.5, -0.25], [0.125, -0.375, 0.625]]]],
            dtype=torch.float32,
        )
        conv.init_orion_params()
        conv.input_shape = torch.Size((1, 1, 4, 4))
        conv.output_shape = torch.Size((1, 1, 4, 4))
        conv.fhe_input_shape = torch.Size((1, 1, 4, 4))
        conv.fhe_output_shape = torch.Size((1, 1, 4, 4))
        conv.input_gap = 1
        conv.output_gap = 1
        conv.name = "native_halo_global_padding_halo_toy_conv"
        level = len(scheme.params.get_logq()) - 1
        conv.set_level(level)
        conv.set_depth(2)

        halo_layout = {"top_beta": 1, "bottom_beta": 2, "stride": 1, "gap": 1, "tile_count": 1}
        executor = LayoutPolicyProviderRuntimeExecutor(
            base_executor=HaloLocalConvRuntimeExecutor(module=conv, output_node_id="conv"),
            output_node_id="conv",
            compile_plan={
                "policy": "fixed_max_fused",
                "edge_layouts": [
                    {
                        "edge": "prev->conv",
                        "source": "prev",
                        "target": "conv",
                        "op_kind": "conv2d",
                        "shape": [1, 1, 4, 4],
                        "fhe_shape": [1, 1, 7, 4],
                        "relayout": False,
                        "layout_mode": "halo_local",
                        "physical_layout": "logical_halo_compact",
                        "required_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                        "source_layout": dict(halo_layout),
                        "selected_layout": dict(halo_layout),
                    }
                ],
                "node_layouts": [
                    {
                        "node": "conv",
                        "shape": [1, 1, 4, 4],
                        "fhe_shape": [1, 1, 4, 4],
                        "output_relayout": False,
                        "physical_layout": "packed_compact",
                        "selected_layout": {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
                    }
                ],
            },
        )
        executor.assigned_level = level
        executor.assigned_depth = 2

        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 10.0
        physical = torch.full((1, 1, 7, 4), 100.0, dtype=torch.float32)
        physical[:, :, 1:5, :] = x
        out = executor(scheme.encrypt(scheme.encode(physical, level)))["conv"]
        decoded = (
            out.decrypt()
            .decode()
            .detach()
            .cpu()
            .to(dtype=torch.float32)
            .reshape(-1)[:16]
            .reshape(1, 1, 4, 4)
        )
        reference = F.conv2d(x, conv.on_weight.detach(), None, padding=1)
        metadata = executor.compile_cache_metadata()

        assert metadata["compact_source_layout"]["top_beta"] == 1
        assert metadata["compact_source_layout"]["bottom_beta"] == 2
        assert float((decoded - reference).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()


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
            if int(dict(row.get("selected_layout", {})).get("top_beta", 0)) > 0
            or int(dict(row.get("selected_layout", {})).get("bottom_beta", 0)) > 0
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


def test_u22_256_base32_provider_mode_routes_decoder_tconv_to_halo_supported_provider() -> None:
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
            runtime = getattr(module, "region_runtime", None)
            assert runtime is not None
            assert runtime.strategy.startswith("halo_supported_tconv")
            executor = getattr(runtime, "executor", None)
            base_executor = getattr(executor, "base_executor", executor)
            assert isinstance(base_executor, HaloSupportedTConvRuntimeExecutor)
            assert isinstance(base_executor.delegate, TconvK2S2PythonRuntimeExecutor)
            assert getattr(module, "region_first_skip_dense_pack", False) is True
    finally:
        scheme.delete_scheme()


@pytest.mark.parametrize("dataset", ("tiny", "imagenet"))
@pytest.mark.parametrize("node_name", DECODER_TCONV_NODES)
def test_u22_decoder_tconv_halo_supported_provider_matches_reference_on_python_backend(dataset: str, node_name: str) -> None:
    _init_python_scheme(logn=int(DATASET_SPECS[str(dataset)]["logn"]))
    try:
        dag = _prepared_dag(dataset=str(dataset))
        registry = U22CompileRegistry.for_dag(dag)
        registry.attach_to_dag(dag)
        _set_compile_level(dag)

        module = dag.nodes[str(node_name)]["module"]
        runtime = getattr(module, "region_runtime", None)
        assert runtime is not None
        assert runtime.strategy == "halo_supported_tconv"
        assert isinstance(runtime.executor, HaloSupportedTConvRuntimeExecutor)

        runtime.assigned_level = int(getattr(module, "level"))
        runtime.assigned_depth = int(runtime.depth)
        runtime.compile(scheme)
        module.he_mode = True

        torch.manual_seed(abs(hash((str(dataset), str(node_name)))) % (2**31))
        x = torch.randn(tuple(int(v) for v in module.input_shape[1:]), dtype=torch.float32)
        out_map = runtime.execute(_encode_input(module, x))
        out = _decode_output(module, out_map[str(node_name)])
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
        assert getattr(runtime.executor, "groups", ())
    finally:
        scheme.delete_scheme()


def test_u22_decoder_tconv_halo_supported_provider_supports_actual_base64_tiny_at_logn16() -> None:
    _init_python_scheme(logn=16)
    try:
        dag = _prepared_dag(dataset="tiny", base_channels=64)
        registry = U22CompileRegistry.for_dag(dag)
        registry.attach_to_dag(dag)
        for node_name in DECODER_TCONV_NODES:
            module = dag.nodes[str(node_name)]["module"]
            runtime = getattr(module, "region_runtime", None)
            assert runtime is not None
            assert runtime.strategy == "halo_supported_tconv"
            assert isinstance(runtime.executor, HaloSupportedTConvRuntimeExecutor)
    finally:
        scheme.delete_scheme()


def test_u22_tconv_provider_fuses_output_bottom_beta_relayout_on_python_backend() -> None:
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
        module.layout_policy_output_layout = {"top_beta": 0, "bottom_beta": 1, "gap": 1}
        module.layout_policy_output_materialization = "fused_relayout"
        module.set_level(len(scheme.params.get_logq()) - 1)

        runtime = TconvK2S2PythonRuntimeExecutor(
            module=module,
            output_node_id="synthetic_tconv_fused_output_bottom_beta",
        )
        x = torch.arange(1, 5, dtype=torch.float32).reshape(1, 2, 2)
        out = runtime(_encode_input(module, x))["synthetic_tconv_fused_output_bottom_beta"]
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
