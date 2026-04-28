from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import orion.nn as on

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.auto_bootstrap import BootstrapSolver
from orion.core.fuser import Fuser
from orion.core.orion import scheme
from orion.core.region_lowering import pack_chw_gap
from orion.core.network_dag import NetworkDAG
from orion.core.region_cir_replay import _compact_stage4_source_from_regular
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental.cir.lattigo_block import (
    build_r18_stage3_shared_block_plan,
    build_r18_stage4_compact_intra_plan,
)
from orion.experimental.cir.runtime_group import (
    FullConvRegionRuntimeExecutor,
    RegionFirstCompileRegistry,
    _rescale_cipher_tensor as _runtime_rescale_cipher_tensor,
    build_r18_actual_region_first_e2e_report,
)
from orion.experimental.cir.r18_e2e_bridges import (
    R18_STAGE12_TRANSITION_SPEC,
    R18_STAGE23_TRANSITION_SPEC,
    R18_STAGE34_TRANSITION_SPEC,
    R18StemBridgeRuntimeExecutor,
    R18TransitionBridgeRuntimeExecutor,
    _rescale_cipher_tensor as _r18_bridge_rescale_cipher_tensor,
)
from orion.models.resnet import ResNet18


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def _prepared_r18_tiny_dag() -> tuple[ResNet18, NetworkDAG]:
    torch.manual_seed(0)
    net = ResNet18(dataset="tiny")
    traced = OrionTracer().trace_model(net)
    StatsTracker(traced).propagate(torch.randn((1, 3, 64, 64), dtype=torch.float32))
    dag = NetworkDAG(traced)
    dag.build_dag()
    for module in net.modules():
        if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
            module.init_orion_params()
    return net, dag


def _set_scheme_levels(dag: NetworkDAG) -> None:
    level = len(scheme.params.get_logq()) - 1
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "set_level"):
            module.set_level(level)


class _RescaleProbeEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def rescale(self, ctxt: int, in_place: bool) -> int:
        self.calls.append((int(ctxt), bool(in_place)))
        return int(ctxt) + 1000


class _RescaleProbeCipher:
    def __init__(self, scheme: SimpleNamespace, ids: list[int], shape: torch.Size, on_shape: torch.Size) -> None:
        self.scheme = scheme
        self.evaluator = scheme.evaluator
        self.ids = ids
        self.shape = shape
        self.on_shape = on_shape


@pytest.mark.parametrize("helper", [_runtime_rescale_cipher_tensor, _r18_bridge_rescale_cipher_tensor])
def test_provider_rescale_helper_honors_rescaled_backend_outputs(helper) -> None:
    evaluator = _RescaleProbeEvaluator()
    scheme_stub = SimpleNamespace(
        backend=SimpleNamespace(lt_outputs_are_rescaled=True),
        evaluator=evaluator,
    )
    ct = _RescaleProbeCipher(scheme_stub, [7], torch.Size([1, 4]), torch.Size([1, 4]))

    assert helper(ct) is ct
    assert evaluator.calls == []

    scheme_stub.backend.lt_outputs_are_rescaled = False
    out = helper(ct)

    assert out is not ct
    assert out.ids == [1007]
    assert evaluator.calls == [(7, False)]


def _chunk_stage_source(x: torch.Tensor, *, channels_per_ct: int, gap: int) -> list[torch.Tensor]:
    c, h, w = (int(v) for v in x.shape)
    chunks: list[torch.Tensor] = []
    for c0 in range(0, int(c), int(channels_per_ct)):
        c1 = min(int(c), int(c0 + int(channels_per_ct)))
        chunk = pack_chw_gap(
            x[int(c0): int(c1)].to(dtype=torch.float32),
            shape=(int(c1 - c0), int(h), int(w)),
            gap=int(gap),
            slots=32768,
        )
        chunks.append(chunk.to(dtype=torch.float32))
    return chunks


def _decode_blocks(ct: CipherTensor) -> list[torch.Tensor]:
    pt = ct.decrypt()
    return [scheme.backend._clone_values(scheme.backend._plaintexts[int(idx)].values) for idx in pt.ids]


def _make_conv_stub(
    *,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    input_gap: int,
    output_gap: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> SimpleNamespace:
    c_out = int(weight.shape[0])
    on_c = int((int(c_out) + int(output_gap * output_gap) - 1) // int(output_gap * output_gap))
    fhe_output_shape = torch.Size([1, int(on_c), int(output_shape[1] * output_gap), int(output_shape[2] * output_gap)])
    return SimpleNamespace(
        on_weight=weight.to(dtype=torch.float32),
        on_bias=bias.to(dtype=torch.float32),
        stride=tuple(int(v) for v in stride),
        padding=tuple(int(v) for v in padding),
        input_shape=torch.Size([1, int(input_shape[0]), int(input_shape[1]), int(input_shape[2])]),
        output_shape=torch.Size([1, int(output_shape[0]), int(output_shape[1]), int(output_shape[2])]),
        fhe_output_shape=fhe_output_shape,
        input_gap=int(input_gap),
        output_gap=int(output_gap),
    )


def _fitted_fused_r18_tiny_dag(config: dict) -> tuple[ResNet18, NetworkDAG]:
    torch.manual_seed(0)
    x = torch.randn((1, 3, 64, 64), dtype=torch.float32)
    net = ResNet18(dataset="tiny", activation="silu", silu_degree=7, stem_relu=True)
    net.set_scheme(scheme)
    net.set_margin(scheme.params.get_margin())
    traced = OrionTracer().trace_model(net)
    StatsTracker(traced).propagate(x)
    for module in net.modules():
        if hasattr(module, "fit") and callable(module.fit):
            module.fit()
        if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
            module.init_orion_params()
        if hasattr(module, "update_params") and callable(module.update_params):
            module.update_params()
    dag = NetworkDAG(traced)
    dag.build_dag()
    fuser = Fuser(dag)
    fuser.fuse_modules()
    dag.remove_fused_batchnorms()
    return net, dag


def _bootstrap_count_for_net(net: ResNet18) -> int:
    config = {
        "ckks_params": {"LogN": 16, "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40], "LogP": [61, 61, 61], "LogScale": 40, "H": 192, "RingType": "Standard"},
        "boot_params": {"LogP": [61, 61, 61, 61, 61, 61, 61, 61]},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none", "experimental_region_first": "r18_tiny_e2e_probe"},
    }
    torch.manual_seed(0)
    x = torch.randn((1, 3, 64, 64), dtype=torch.float32)
    scheme.init_scheme(config)
    try:
        net.set_scheme(scheme)
        net.set_margin(scheme.params.get_margin())
        traced = OrionTracer().trace_model(net)
        StatsTracker(traced).propagate(x)
        for module in net.modules():
            if hasattr(module, "fit") and callable(module.fit):
                module.fit()
        dag = NetworkDAG(traced)
        dag.build_dag()
        for module in net.modules():
            if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
                module.init_orion_params()
        for module in net.modules():
            if hasattr(module, "update_params") and callable(module.update_params):
                module.update_params()
        fuser = Fuser(dag)
        fuser.fuse_modules()
        dag.remove_fused_batchnorms()
        registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
        registry.attach_to_dag(dag)
        registry.attach_probe_dense_bypass_to_dag(dag)
        dag.find_residuals()
        solver = BootstrapSolver(net, dag, l_eff=len(scheme.params.get_logq()) - 1)
        _input_level, bootstraps, _slots = solver.solve()
        return int(bootstraps)
    finally:
        scheme.delete_scheme()


def test_r18_silu_variant_preserves_default_and_hits_bootstrap_target() -> None:
    default = ResNet18(dataset="tiny")
    assert isinstance(default.act, on.ReLU)
    assert sum(1 for _name, module in default.named_modules() if isinstance(module, on.SiLU)) == 0

    silu = ResNet18(dataset="tiny", activation="silu", silu_degree=7, stem_relu=True)
    assert isinstance(silu.act, on.ReLU)
    assert sum(1 for _name, module in silu.named_modules() if isinstance(module, on.SiLU)) == 16
    assert sum(1 for _name, module in silu.named_modules() if isinstance(module, on.ReLU)) == 1

    assert _bootstrap_count_for_net(silu) == 60


def test_r18_e2e_compile_registry_attaches_full_conv_region_nodes() -> None:
    _net, dag = _prepared_r18_tiny_dag()

    registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
    audit = registry.attach_to_dag(dag)

    assert registry.graph_audit["replacement_mode"] == "full_conv_region_nodes"
    assert audit["executable_region_count"] == len(registry.groups)
    assert len(registry.groups) == 17
    assert registry.graph_audit["excluded_nodes"] == []
    stem_groups = [group for group in registry.groups if group.stage == "stem"]
    transition_groups = [group for group in registry.groups if str(group.stage).endswith("_transition")]
    same_shape_groups = [group for group in registry.groups if group.stage in {"stage1", "stage2", "stage3", "stage4"}]
    assert len(stem_groups) == 1
    assert len(transition_groups) == 3
    assert len(same_shape_groups) == 13
    for group in registry.groups:
        assert group.executable is True
        if group.stage == "stem":
            assert len(group.conv_nodes) == 1
            assert isinstance(group.executor, R18StemBridgeRuntimeExecutor)
        elif str(group.stage).endswith("_transition"):
            assert len(group.conv_nodes) == 2
            assert isinstance(group.executor, R18TransitionBridgeRuntimeExecutor)
        else:
            assert len(group.conv_nodes) == 1
            assert isinstance(group.executor, FullConvRegionRuntimeExecutor)
        for node in group.conv_nodes:
            module = dag.nodes[node]["module"]
            assert getattr(module, "region_first_skip_dense_pack") is True
            assert getattr(module, "region_runtime") is group


def test_r18_e2e_solver_depth_matches_provider_runtime_level_cost() -> None:
    _net, dag = _prepared_r18_tiny_dag()

    registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
    registry.attach_to_dag(dag)

    depth_by_node = {
        str(node): int(getattr(dag.nodes[node]["module"], "depth"))
        for group in registry.groups
        for node in group.conv_nodes
    }

    assert depth_by_node["conv1"] == 1
    assert depth_by_node["layers_1_0_conv1"] == 1
    assert depth_by_node["layers_1_0_shortcut_0"] == 1
    assert depth_by_node["layers_0_0_conv1"] == 1
    assert depth_by_node["layers_2_1_conv2"] == 1
    assert depth_by_node["layers_3_1_conv2"] == 2


def test_r18_actual_e2e_report_builds_gate_without_dense_pack(monkeypatch) -> None:
    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("actual E2E gate must not materialize dense conv masks")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)

    payload = build_r18_actual_region_first_e2e_report()

    assert payload["status"] == "ready_for_manual_ckks_forward"
    assert payload["dense_cleartext"]["ran"] is True
    assert payload["region_first"]["replacement_mode"] == "full_conv_region_nodes"
    assert payload["region_first"]["runtime_group_count"] == 17
    assert payload["e2e_gate"]["graph_replacement_ready"] is True
    assert payload["e2e_gate"]["source_pairing_runtime_ready"] is True
    assert payload["e2e_gate"]["output_assembly_runtime_ready"] is True
    assert payload["claim"]["full_network_ckks"] is False
    assert payload["claim"]["runtime_speedup_publishable"] is False


def test_r18_actual_e2e_report_records_silu_bootstrap_reference() -> None:
    payload = build_r18_actual_region_first_e2e_report(
        activation="silu",
        silu_degree=7,
        stem_relu=True,
    )

    assert payload["activation"] == {
        "kind": "silu",
        "silu_degree": 7,
        "stem_relu": True,
        "expected_bootstraps_reference": 61,
    }


def test_r18_e2e_probe_marks_dense_fallback_bypass_nodes() -> None:
    _net, dag = _prepared_r18_tiny_dag()
    registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
    audit = registry.attach_to_dag(dag)

    bypassed = registry.attach_probe_dense_bypass_to_dag(dag)

    assert audit["executable_region_count"] == len(registry.groups)
    assert bypassed == []
    for item in bypassed:
        module = dag.nodes[item["node"]]["module"]
        assert getattr(module, "region_first_probe_dense_bypass") is True
        assert getattr(module, "region_first_skip_dense_pack", False) is not True
    for group in registry.groups:
        module = dag.nodes[group.conv_nodes[0]]["module"]
        assert getattr(module, "region_first_probe_lazy_region_compile") is True


def test_r18_e2e_probe_can_disable_lazy_region_compile() -> None:
    _net, dag = _prepared_r18_tiny_dag()
    registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
    registry.attach_to_dag(dag)

    registry.attach_probe_dense_bypass_to_dag(dag, lazy_region_compile=False)

    for group in registry.groups:
        module = dag.nodes[group.conv_nodes[0]]["module"]
        assert getattr(module, "region_first_probe_lazy_region_compile") is False


def test_r18_e2e_probe_bypasses_only_stem_relu_in_he_mode() -> None:
    net = ResNet18(dataset="tiny", activation="silu", silu_degree=7, stem_relu=True)
    traced = OrionTracer().trace_model(net)
    StatsTracker(traced).propagate(torch.randn((1, 3, 64, 64), dtype=torch.float32))
    dag = NetworkDAG(traced)
    dag.build_dag()
    registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)

    bypassed = registry.attach_probe_stem_activation_bypass(net)

    assert bypassed == [{"node": "act", "module": "ReLU", "reason": "probe_only_skip_stem_relu"}]
    assert getattr(net.act, "region_first_probe_activation_bypass") is True
    sentinel = object()
    net.act.he_mode = True
    net.act.scheme = SimpleNamespace(params=SimpleNamespace(get_debug_status=lambda: False))
    assert net.act(sentinel) is sentinel


def test_r18_e2e_compile_records_assigned_region_level() -> None:
    conv = on.Conv2d(1, 1, 3, padding=1, bias=False)
    group = RegionFirstCompileRegistry.for_r18_tiny_e2e(_prepared_r18_tiny_dag()[1]).groups[0]
    conv.region_runtime = group
    conv.region_first_skip_dense_pack = True
    conv.region_first_probe_lazy_region_compile = True
    conv.level = 6
    conv.depth = 2
    conv.scheme = SimpleNamespace()

    conv.compile()

    assert getattr(group, "assigned_level") == 6
    assert getattr(group, "assigned_depth") == 2
    assert getattr(group.executor, "assigned_level") == 6


def test_stage4_dense_transition_runtime_matches_solver_levels_after_cir_depth_fix() -> None:
    _require_lattigo()

    config = {
        "ckks_params": {"LogN": 16, "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40], "LogP": [61, 61, 61], "LogScale": 40, "H": 192, "RingType": "Standard"},
        "boot_params": {"LogP": [61, 61, 61, 61, 61, 61, 61, 61]},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none", "experimental_region_first": "r18_tiny_e2e"},
    }
    scheme.init_scheme(config)
    try:
        net, dag = _fitted_fused_r18_tiny_dag(config)
        registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
        registry.attach_to_dag(dag)
        dag.find_residuals()
        solver = BootstrapSolver(net, dag, l_eff=len(scheme.params.get_logq()) - 1)
        solver.solve()

        block = net.layers[3][0]
        assert block.conv1.level == 9
        assert block.act1.level == 7
        assert block.conv2.level == 4
        assert block.shortcut[0].level == 3
        assert block.add.level == 1

        block.conv1.generate_diagonals(last=False)
        block.conv1.compile()
        block.act1.compile()
        block.conv2.generate_diagonals(last=False)
        block.conv2.compile()
        block.shortcut[0].generate_diagonals(last=False)
        block.shortcut[0].compile()

        block.conv1.he_mode = True
        block.act1.he_mode = True
        block.conv2.he_mode = True
        block.shortcut[0].he_mode = True
        block.add.he_mode = True

        dummy = torch.randn(block.conv1.fhe_input_shape, dtype=torch.float32)
        x_ct = scheme.encrypt(scheme.encode(dummy, int(block.conv1.level)))

        main1 = block.conv1(x_ct)
        main2 = block.act1(main1)
        main3 = block.conv2(main2)
        short = block.shortcut[0](x_ct)
        summed = block.add(main3, short)

        assert main1.level() == block.conv1.level - block.conv1.depth
        assert main2.level() == block.act1.level - block.act1.depth
        assert main3.level() == block.conv2.level - block.conv2.depth
        assert short.level() == block.shortcut[0].level - block.shortcut[0].depth
        assert summed.level() == 1
    finally:
        scheme.delete_scheme()


def test_full_conv_region_executor_pairs_sources_and_assembles_output(monkeypatch) -> None:
    _require_lattigo()

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("full-conv region executor must not call dense pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
    _plan, inputs, _reference = build_r18_stage3_shared_block_plan(bank_count=2)
    executor = FullConvRegionRuntimeExecutor(
        plan_builders=(build_r18_stage3_shared_block_plan,),
        builder_kwargs=({"bank_count": 2},),
        output_node_id="layers_2_0_conv2",
        output_shape=torch.Size([1, 256, 16, 16]),
        fhe_output_shape=torch.Size([1, 16, 64, 64]),
        source_pair_count=1,
    )
    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        level = len(scheme.params.get_logq()) - 1
        left = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        right = inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        ct_left = scheme.encrypt(scheme.encode(left, level))
        ct_right = scheme.encrypt(scheme.encode(right, level))
        source = CipherTensor(
            scheme,
            [int(ct_left.ids[0]), int(ct_right.ids[0])],
            torch.Size([1, 16, 64, 64]),
            torch.Size([1, 16, 64, 64]),
        )

        outputs = executor(source)

        out = outputs["layers_2_0_conv2"]
        assert isinstance(out, CipherTensor)
        assert len(out.ids) == 2
        assert out.shape == torch.Size([1, 256, 16, 16])
        assert out.on_shape == torch.Size([1, 16, 64, 64])
        assert out.scale() == scheme.params.get_default_scale()
        assert tuple(out.decrypt().decode().shape) == (1, 16, 64, 64)
        assert executor.compile_count == 1
        assert executor.block_evaluate_count == 1
        assert executor.last_runtime_timing["compile_unified_s"] >= 0.0
        assert executor.last_runtime_timing["evaluate_unified_s"] > 0.0
        assert executor.last_runtime_timing["postprocess_s"] > 0.0
    finally:
        scheme.delete_scheme()


def test_precompiled_full_conv_executor_runtime_excludes_compile_bucket(monkeypatch) -> None:
    _require_lattigo()

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("full-conv region executor must not call dense pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
    _plan, inputs, _reference = build_r18_stage3_shared_block_plan(bank_count=2)
    executor = FullConvRegionRuntimeExecutor(
        plan_builders=(build_r18_stage3_shared_block_plan,),
        builder_kwargs=({"bank_count": 2},),
        output_node_id="layers_2_0_conv2",
        output_shape=torch.Size([1, 256, 16, 16]),
        fhe_output_shape=torch.Size([1, 16, 64, 64]),
        source_pair_count=1,
    )
    executor.assigned_level = 6
    executor.assigned_depth = 2
    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        level = len(scheme.params.get_logq()) - 1
        left = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        right = inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        ct_left = scheme.encrypt(scheme.encode(left, level))
        ct_right = scheme.encrypt(scheme.encode(right, level))
        source = CipherTensor(
            scheme,
            [int(ct_left.ids[0]), int(ct_right.ids[0])],
            torch.Size([1, 16, 64, 64]),
            torch.Size([1, 16, 64, 64]),
        )

        executor.compile(scheme)
        assert executor.compile_count == 1
        assert executor.last_runtime_timing["compile_unified_s"] > 0.0

        _outputs = executor(source)
        assert executor.compile_count == 1
        assert executor.last_runtime_timing["compile_unified_s"] == 0.0
        assert executor.last_runtime_timing["evaluate_unified_s"] > 0.0
        assert executor.last_runtime_timing["postprocess_s"] > 0.0
    finally:
        scheme.delete_scheme()


def test_stage4_full_conv_region_executor_prepacks_regular_source_on_python_backend(monkeypatch) -> None:
    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("stage4 full-conv region executor must not call dense pack_conv2d")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
    _net, dag = _prepared_r18_tiny_dag()
    registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
    registry.attach_to_dag(dag)
    module = dag.nodes["layers_3_0_conv2"]["module"]
    group = getattr(module, "region_runtime")

    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "python", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        level = len(scheme.params.get_logq()) - 1
        plan, inputs, _reference = build_r18_stage4_compact_intra_plan(
            weight_override=getattr(module, "on_weight"),
            bias_override=getattr(module, "on_bias", None),
            input_shape=(512, 8, 8),
            output_shape=(512, 8, 8),
            input_gap=8,
            output_gap=8,
        )
        regular_source = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        x_ct = scheme.encrypt(scheme.encode(regular_source, level))
        source = CipherTensor(scheme, [int(x_ct.ids[0])], module.input_shape, module.fhe_input_shape)

        compact_source = group.executor._source_ciphertexts(source)[0]
        compact_decoded = compact_source.decrypt()
        compact_raw = scheme.backend.DecodeComplex(compact_decoded.ids[0])
        compact_tensor = torch.tensor(
            [complex(compact_raw[2 * i], compact_raw[2 * i + 1]) for i in range(int(scheme.params.get_slots()))],
            dtype=torch.complex64,
        )
        reference_compact = _compact_stage4_source_from_regular(regular_source)

        assert float((compact_tensor - reference_compact).abs().max()) <= 1.0e-6
        assert getattr(compact_source, "region_first_compact_source", False) is True
        assert getattr(compact_source, "stage4_compact_source", False) is True
    finally:
        scheme.delete_scheme()


@pytest.mark.parametrize(
    ("spec", "channels_per_ct_in", "channels_per_ct_out"),
    (
        (R18_STAGE12_TRANSITION_SPEC, 8, 32),
        (R18_STAGE23_TRANSITION_SPEC, 32, 128),
        (R18_STAGE34_TRANSITION_SPEC, 128, 512),
    ),
)
def test_transition_bridge_matches_clear_reference_on_python_backend(
    spec,
    channels_per_ct_in: int,
    channels_per_ct_out: int,
) -> None:
    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "python", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        torch.manual_seed(0)
        x = torch.randn((int(spec.c_in), int(spec.h_in), int(spec.w_in)), dtype=torch.float32)
        conv_weight = torch.randn((int(spec.c_out), int(spec.c_in), 3, 3), dtype=torch.float32) * 0.05
        shortcut_weight = torch.randn((int(spec.c_out), int(spec.c_in), 1, 1), dtype=torch.float32) * 0.05
        conv_bias = torch.randn((int(spec.c_out),), dtype=torch.float32) * 0.01
        shortcut_bias = torch.randn((int(spec.c_out),), dtype=torch.float32) * 0.01
        conv_module = _make_conv_stub(
            weight=conv_weight,
            bias=conv_bias,
            input_shape=(int(spec.c_in), int(spec.h_in), int(spec.w_in)),
            output_shape=(int(spec.c_out), int(spec.h_out), int(spec.w_out)),
            input_gap=int(spec.input_gap),
            output_gap=int(spec.output_gap),
            stride=(2, 2),
            padding=(1, 1),
        )
        shortcut_module = _make_conv_stub(
            weight=shortcut_weight,
            bias=shortcut_bias,
            input_shape=(int(spec.c_in), int(spec.h_in), int(spec.w_in)),
            output_shape=(int(spec.c_out), int(spec.h_out), int(spec.w_out)),
            input_gap=int(spec.input_gap),
            output_gap=int(spec.output_gap),
            stride=(2, 2),
            padding=(0, 0),
        )
        group = SimpleNamespace(
            execute=R18TransitionBridgeRuntimeExecutor(
                conv_module=conv_module,
                shortcut_module=shortcut_module,
                spec=spec,
                output_node_ids=("main", "shortcut"),
            ).__call__
        )
        source_blocks = _chunk_stage_source(x, channels_per_ct=int(channels_per_ct_in), gap=int(spec.input_gap))
        cts = [scheme.encrypt(scheme.encode(block, len(scheme.params.get_logq()) - 1)) for block in source_blocks]
        source = CipherTensor(
            scheme,
            [int(ct.ids[0]) for ct in cts],
            conv_module.input_shape,
            torch.Size([1, int(spec.c_in // (spec.input_gap * spec.input_gap)), int(spec.h_in * spec.input_gap), int(spec.w_in * spec.input_gap)]),
        )

        outputs = group.execute(source)
        out_conv = outputs["main"]
        out_shortcut = outputs["shortcut"]

        ref_conv = torch.nn.functional.conv2d(
            x.unsqueeze(0),
            conv_module.on_weight.to(dtype=torch.float32),
            conv_module.on_bias.to(dtype=torch.float32),
            stride=conv_module.stride,
            padding=conv_module.padding,
        )[0]
        ref_shortcut = torch.nn.functional.conv2d(
            x.unsqueeze(0),
            shortcut_module.on_weight.to(dtype=torch.float32),
            shortcut_module.on_bias.to(dtype=torch.float32),
            stride=shortcut_module.stride,
            padding=shortcut_module.padding,
        )[0]
        ref_conv_blocks = _chunk_stage_source(ref_conv, channels_per_ct=int(channels_per_ct_out), gap=int(spec.output_gap))
        ref_shortcut_blocks = _chunk_stage_source(ref_shortcut, channels_per_ct=int(channels_per_ct_out), gap=int(spec.output_gap))

        out_conv_blocks = _decode_blocks(out_conv)
        out_shortcut_blocks = _decode_blocks(out_shortcut)
        assert len(out_conv_blocks) == len(ref_conv_blocks)
        assert len(out_shortcut_blocks) == len(ref_shortcut_blocks)
        for actual, expected in zip(out_conv_blocks, ref_conv_blocks):
            assert float((actual - expected).abs().max()) <= 1.0e-3
        for actual, expected in zip(out_shortcut_blocks, ref_shortcut_blocks):
            assert float((actual - expected).abs().max()) <= 1.0e-3
    finally:
        scheme.delete_scheme()


def test_stem_bridge_matches_clear_reference_on_python_backend() -> None:
    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "python", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        torch.manual_seed(0)
        weight = torch.randn((64, 3, 7, 7), dtype=torch.float32) * 0.05
        bias = torch.randn((64,), dtype=torch.float32) * 0.01
        module = _make_conv_stub(
            weight=weight,
            bias=bias,
            input_shape=(3, 64, 64),
            output_shape=(64, 64, 64),
            input_gap=1,
            output_gap=1,
            stride=(1, 1),
            padding=(3, 3),
        )
        executor = R18StemBridgeRuntimeExecutor(module=module, output_node_id="conv1")
        x = torch.randn((3, 64, 64), dtype=torch.float32)
        source_block = pack_chw_gap(x, shape=(3, 64, 64), gap=1, slots=32768).to(dtype=torch.float32)
        ct = scheme.encrypt(scheme.encode(source_block, len(scheme.params.get_logq()) - 1))
        source = CipherTensor(scheme, [int(ct.ids[0])], module.input_shape, torch.Size([1, 3, 64, 64]))

        outputs = executor(source)
        out = outputs["conv1"]

        ref = torch.nn.functional.conv2d(
            x.unsqueeze(0),
            module.on_weight.to(dtype=torch.float32),
            module.on_bias.to(dtype=torch.float32),
            stride=module.stride,
            padding=module.padding,
        )[0]
        ref_blocks = _chunk_stage_source(ref, channels_per_ct=8, gap=1)
        out_blocks = _decode_blocks(out)
        assert len(out_blocks) == len(ref_blocks)
        for actual, expected in zip(out_blocks, ref_blocks):
            assert float((actual - expected).abs().max()) <= 1.0e-3
    finally:
        scheme.delete_scheme()
