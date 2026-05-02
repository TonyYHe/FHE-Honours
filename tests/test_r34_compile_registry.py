from __future__ import annotations

from pathlib import Path

import pytest
import torch

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.orion import scheme
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental.cir import r34_orion_same_shape as r34_same_shape
from orion.experimental.cir.transition_pool_provider import (
    BranchPairConvRuntimeExecutor,
    InputPairConvRuntimeExecutor,
)
from orion.experimental.cir.ir import (
    CanonicalTemplateEntry,
    ConvSchemePlan,
    ExecutionStats,
    FamilyTemplateBank,
    LinearTransformStep,
    LinearTransformTerm,
    PreparedPlaintext,
    TensorRegion,
)
from orion.experimental import R34CompileRegistry
from orion.nn.module import Module
from orion.models.resnet import ResNet34


def _prepared_r34_imagenet_dag() -> NetworkDAG:
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
        if module is not None and hasattr(module, "update_params"):
            module.update_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()
    return dag


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


def test_r34_compile_registry_builds_groups_from_imported_layout_contracts() -> None:
    dag = _prepared_r34_imagenet_dag()

    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)

    assert registry.graph_audit["replacement_mode"] == "r34_imgnet_phase1_layout_contracts"
    assert registry.graph_audit["missing_imported_nodes"] == []
    assert registry.graph_audit["selected_region_count"] == 35
    assert sorted(registry.graph_audit["family_labels"]) == [
        "global_avgpool_exit",
        "stage1_same",
        "stage2_same",
        "stage2_transition",
        "stage3_same",
        "stage3_transition",
        "stage4_same",
        "stage4_transition",
        "stem_conv",
        "stem_pool",
    ]
    transition = next(group for group in registry.groups if group.stage == "stage3_transition")
    assert transition.conv_nodes == ("layers_2_0_conv1", "layers_2_0_shortcut_0")
    assert transition.materializer == "policy_inter_group_hybrid_transition"
    assert transition.executable is False
    assert any(group.conv_nodes == ("layers_1_0_conv1", "layers_1_0_shortcut_0") for group in registry.groups)
    assert any(group.conv_nodes == ("layers_3_0_conv1", "layers_3_0_shortcut_0") for group in registry.groups)
    assert any(group.conv_nodes == ("layers_3_2_conv2",) for group in registry.groups)


def test_r34_compile_registry_attaches_layout_metadata_and_dense_fallback() -> None:
    dag = _prepared_r34_imagenet_dag()
    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)

    audit = registry.attach_to_dag(dag)

    assert audit["attached_count"] == 38
    assert audit["executable_region_count"] == 35
    assert len(audit["fallback_layers"]) == 0

    stem_conv = dag.nodes["conv1"]["module"]
    stem_pool = dag.nodes["pool"]["module"]
    exit_pool = dag.nodes["avgpool"]["module"]
    assert getattr(stem_conv, "region_family_label") == "stem_conv"
    assert getattr(stem_conv, "region_source_group_count") == 3
    assert getattr(stem_conv, "region_kernel_policy") == "inter_group_hybrid"
    assert isinstance(stem_conv.region_runtime.executor, InputPairConvRuntimeExecutor)
    assert stem_conv.region_runtime.executable is True

    assert getattr(stem_pool, "region_family_label") == "stem_pool"
    assert getattr(stem_pool, "region_source_group_count") == 16
    assert getattr(stem_pool, "region_kernel_policy") == "inter_group_hybrid"
    assert isinstance(stem_pool.region_runtime.executor, InputPairConvRuntimeExecutor)
    assert stem_pool.region_runtime.executable is True

    assert getattr(exit_pool, "region_family_label") == "global_avgpool_exit"
    assert getattr(exit_pool, "region_source_group_count") == 1
    assert getattr(exit_pool, "region_kernel_policy") == "intra_group_pack2"
    assert isinstance(exit_pool.region_runtime.executor, InputPairConvRuntimeExecutor)
    assert exit_pool.region_runtime.executable is True

    stage2_module = dag.nodes["layers_1_1_conv1"]["module"]
    assert getattr(stage2_module, "region_runtime") is not None
    assert getattr(stage2_module, "region_output_id") == "layers_1_1_conv1"
    assert getattr(stage2_module, "region_first_skip_dense_pack") is True
    assert getattr(stage2_module, "region_layout_contract_imported") is True
    assert getattr(stage2_module, "region_input_layout")["stride"] == 8
    assert getattr(stage2_module, "region_output_layout")["stride"] == 8
    assert getattr(stage2_module, "region_family_label") == "stage2_same"
    assert getattr(stage2_module, "region_source_group_count") == 2
    assert getattr(stage2_module, "region_kernel_policy") == "inter_group_hybrid"
    assert stage2_module.region_runtime.executable is True
    assert isinstance(stage2_module.region_runtime.executor, r34_same_shape.R34InterGroupHybridSameShapeRuntimeExecutor)

    stage4_module = dag.nodes["layers_3_2_conv2"]["module"]
    assert getattr(stage4_module, "region_runtime") is not None
    assert getattr(stage4_module, "region_first_skip_dense_pack") is True
    assert getattr(stage4_module, "region_layout_contract_imported") is False
    assert getattr(stage4_module, "region_input_layout")["stride"] == 32
    assert getattr(stage4_module, "region_output_layout")["stride"] == 32
    assert getattr(stage4_module, "region_family_label") == "stage4_same"
    assert getattr(stage4_module, "region_source_group_count") == 1
    assert getattr(stage4_module, "region_kernel_policy") == "intra_group_pack2"
    assert stage4_module.region_runtime.executable is True

    transition_conv = dag.nodes["layers_2_0_conv1"]["module"]
    transition_shortcut = dag.nodes["layers_2_0_shortcut_0"]["module"]
    assert transition_conv.region_runtime is transition_shortcut.region_runtime
    assert getattr(transition_conv, "region_output_id") == "layers_2_0_conv1"
    assert getattr(transition_shortcut, "region_output_id") == "layers_2_0_shortcut_0"
    assert getattr(transition_conv, "region_first_skip_dense_pack") is True
    assert getattr(transition_conv, "region_input_layout")["stride"] == 8
    assert getattr(transition_conv, "region_output_layout")["stride"] == 16
    assert getattr(transition_conv, "region_family_kind") == "transition_family_baseline"
    assert getattr(transition_conv, "region_source_group_count") == 2
    assert getattr(transition_conv, "region_kernel_policy") == "inter_group_hybrid"
    assert transition_conv.region_runtime.executable is True
    assert transition_conv.region_runtime.executor is not None

    stage2_transition_conv = dag.nodes["layers_1_0_conv1"]["module"]
    stage2_transition_shortcut = dag.nodes["layers_1_0_shortcut_0"]["module"]
    assert stage2_transition_conv.region_runtime is stage2_transition_shortcut.region_runtime
    assert getattr(stage2_transition_conv, "region_source_group_count") == 4
    assert getattr(stage2_transition_conv, "region_kernel_policy") == "inter_group_hybrid"
    assert stage2_transition_conv.region_runtime.executable is True
    assert isinstance(stage2_transition_conv.region_runtime.executor, BranchPairConvRuntimeExecutor)

    stage4_transition_conv = dag.nodes["layers_3_0_conv1"]["module"]
    stage4_transition_shortcut = dag.nodes["layers_3_0_shortcut_0"]["module"]
    assert stage4_transition_conv.region_runtime is stage4_transition_shortcut.region_runtime
    assert isinstance(stage4_transition_conv.region_runtime.executor, BranchPairConvRuntimeExecutor)
    assert getattr(stage4_transition_conv, "region_source_group_count") == 1
    assert getattr(stage4_transition_conv, "region_kernel_policy") == "intra_group_pack2"
    assert stage4_transition_conv.region_runtime.executable is True


@pytest.mark.parametrize(
    ("conv_node", "shortcut_node"),
    (
        ("layers_1_0_conv1", "layers_1_0_shortcut_0"),
        ("layers_2_0_conv1", "layers_2_0_shortcut_0"),
        ("layers_3_0_conv1", "layers_3_0_shortcut_0"),
    ),
)
def test_r34_transition_group_compiles_and_executes_one_hybrid_runtime(monkeypatch, conv_node: str, shortcut_node: str) -> None:
    _require_lattigo()
    dag = _prepared_r34_imagenet_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()

    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
    registry.attach_to_dag(dag)
    transition_conv = dag.nodes[str(conv_node)]["module"]
    transition_shortcut = dag.nodes[str(shortcut_node)]["module"]
    group = transition_conv.region_runtime
    executor = group.executor
    assert isinstance(executor, BranchPairConvRuntimeExecutor)

    def fake_pack_conv2d(module, last):
        assert last is False
        slots = int(module.scheme.params.get_slots())
        scale = 1.0 if module is transition_conv else 2.0
        return (
            {
                (0, 0): {0: [scale] * slots},
                (0, 1): {1: [0.0] * slots},
                (1, 0): {0: [0.5 * scale] * slots},
                (1, 1): {2: [0.0] * slots},
            },
            0,
        )

    def fake_bias(module):
        slots = int(module.scheme.params.get_slots())
        return torch.zeros((2 * slots,), dtype=torch.float32)

    monkeypatch.setattr(packing, "pack_conv2d", fake_pack_conv2d)
    monkeypatch.setattr(packing, "construct_conv2d_bias", fake_bias)

    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        Module.set_scheme(scheme)
        transition_conv.level = len(scheme.params.get_logq()) - 1
        transition_shortcut.level = len(scheme.params.get_logq()) - 1
        transition_conv.depth = 2
        transition_shortcut.depth = 2

        group.compile(scheme)
        assert executor.compile_count == 1
        assert executor.cols == 2
        assert executor.rows == 2

        ids = []
        for seed in range(executor.cols):
            torch.manual_seed(int(seed))
            packed = torch.randn((scheme.params.get_slots(),), dtype=torch.float32) * 0.01
            ct = scheme.encrypt(scheme.encode(packed, len(scheme.params.get_logq()) - 1))
            ids.append(int(ct.ids[0]))
            ct.ids = []
        source = CipherTensor(
            scheme,
            ids,
            transition_conv.input_shape,
            transition_conv.fhe_input_shape,
        )

        out_conv = group.output(str(conv_node), source)
        out_shortcut = group.output(str(shortcut_node), source)

        assert isinstance(out_conv, CipherTensor)
        assert isinstance(out_shortcut, CipherTensor)
        assert len(out_conv.ids) == executor.rows
        assert len(out_shortcut.ids) == executor.rows
        assert group.execute_count == 1
    finally:
        scheme.delete_scheme()


@pytest.mark.parametrize(
    ("node_name", "executor_type"),
    (
        ("layers_1_1_conv1", r34_same_shape.R34InterGroupHybridSameShapeRuntimeExecutor),
        ("layers_2_1_conv1", r34_same_shape.R34Pack2SameShapeRuntimeExecutor),
    ),
)
def test_r34_same_shape_group_compiles_and_executes_runtime(monkeypatch, node_name: str, executor_type: type) -> None:
    _require_lattigo()
    dag = _prepared_r34_imagenet_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()

    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
    registry.attach_to_dag(dag)
    module = dag.nodes[str(node_name)]["module"]
    group = module.region_runtime
    executor = group.executor
    assert isinstance(executor, executor_type)

    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("same-shape runtime should not call pack_conv2d")

    def fake_bias(conv_layer):
        slots = int(conv_layer.scheme.params.get_slots())
        return torch.zeros((2 * slots,), dtype=torch.float32)

    def fake_same_shape_plan(**_kwargs):
        slots = 32768
        c = int(module.output_shape[1])
        h = int(module.output_shape[2])
        w = int(module.output_shape[3])
        template_entries = (
            CanonicalTemplateEntry("template_0", "family", (0, 0), 0, torch.tensor([0], dtype=torch.int64)),
            CanonicalTemplateEntry("template_1", "family", (1, 0), 0, torch.tensor([0], dtype=torch.int64)),
            CanonicalTemplateEntry("template_2", "family", (2, 1), 1, torch.tensor([1], dtype=torch.int64)),
            CanonicalTemplateEntry("template_3", "family", (3, 1), 1, torch.tensor([1], dtype=torch.int64)),
            CanonicalTemplateEntry("template_4", "family", (4, 2), 2, torch.tensor([2], dtype=torch.int64)),
        )
        prepared = (
            PreparedPlaintext("pt_0", "template_0", 0, 1.0, slots, torch.tensor([1.0], dtype=torch.float32)),
            PreparedPlaintext("pt_1", "template_1", 0, 1.0, slots, torch.tensor([1.5], dtype=torch.float32)),
            PreparedPlaintext("pt_2", "template_2", 0, 1.0, slots, torch.tensor([0.5], dtype=torch.float32)),
            PreparedPlaintext("pt_3", "template_3", 0, 1.0, slots, torch.tensor([0.75], dtype=torch.float32)),
            PreparedPlaintext("pt_4", "template_4", 0, 1.0, slots, torch.tensor([0.25], dtype=torch.float32)),
        )
        steps = (
            LinearTransformStep(
                "lt_0_0",
                "orion_source_block_0",
                0,
                0,
                (),
                (),
                (LinearTransformTerm("term_0", 0, "pt_0", "template_0", torch.tensor([0]), torch.tensor([0])),),
                (),
                ("pt_0",),
                ExecutionStats(rotations=1, ct_pt_mults=1, adds=1),
                "real_bsgs",
            ),
            LinearTransformStep(
                "lt_1_0",
                "orion_source_block_1",
                0,
                0,
                (),
                (),
                (LinearTransformTerm("term_1", 0, "pt_1", "template_1", torch.tensor([0]), torch.tensor([0])),),
                (),
                ("pt_1",),
                ExecutionStats(rotations=1, ct_pt_mults=1, adds=1),
                "real_bsgs",
            ),
            LinearTransformStep(
                "lt_0_1",
                "orion_source_block_0",
                1,
                0,
                (),
                (),
                (LinearTransformTerm("term_2", 1, "pt_2", "template_2", torch.tensor([0]), torch.tensor([1])),),
                (1,),
                ("pt_2",),
                ExecutionStats(rotations=1, ct_pt_mults=1, adds=1),
                "real_bsgs",
            ),
            LinearTransformStep(
                "lt_1_1",
                "orion_source_block_1",
                1,
                0,
                (),
                (),
                (LinearTransformTerm("term_3", 1, "pt_3", "template_3", torch.tensor([0]), torch.tensor([1])),),
                (1,),
                ("pt_3",),
                ExecutionStats(rotations=1, ct_pt_mults=1, adds=1),
                "real_bsgs",
            ),
            LinearTransformStep(
                "lt_2_0",
                "orion_source_block_2",
                0,
                0,
                (),
                (),
                (LinearTransformTerm("term_4", 2, "pt_4", "template_4", torch.tensor([0]), torch.tensor([2])),),
                (2,),
                ("pt_4",),
                ExecutionStats(rotations=1, ct_pt_mults=1, adds=1),
                "real_bsgs",
            ),
        )
        plan = ConvSchemePlan(
            case_name="fake_r34_same_shape",
            ring_slot_count=slots,
            output_regions=(
                TensorRegion(0, int(c), 0, int(h), 0, int(w)),
                TensorRegion(0, int(c), 0, int(h), 0, int(w)),
            ),
            output_active_slot_counts=(slots, slots),
            family_templates=(
                FamilyTemplateBank(
                    "family",
                    ("fake",),
                    (int(c), int(h), int(w)),
                    (int(c), int(h), int(w)),
                    (0, int(h)),
                    (0, int(h)),
                    template_entries,
                    5,
                ),
            ),
            prepared_plaintexts=prepared,
            linear_transform_steps=steps,
            expected_cost=ExecutionStats(rotations=5, ct_pt_mults=5, adds=5),
        )
        return plan, {}, torch.zeros((int(c), int(h), int(w)), dtype=torch.float32)

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
    monkeypatch.setattr(packing, "construct_conv2d_bias", fake_bias)
    monkeypatch.setattr(r34_same_shape, "build_r34_same_shape_orion_plan", fake_same_shape_plan)

    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        Module.set_scheme(scheme)
        module.level = len(scheme.params.get_logq()) - 1
        module.depth = 2

        group.compile(scheme)
        assert executor.compile_count == 1
        expected_cols = 3 if isinstance(executor, r34_same_shape.R34InterGroupHybridSameShapeRuntimeExecutor) else 2
        assert executor.cols == expected_cols
        assert executor.rows == 2
        if isinstance(executor, r34_same_shape.R34InterGroupHybridSameShapeRuntimeExecutor):
            assert executor.input_block_pairs == [(0, 1), (2, None)]
            assert executor.complex_input_block_flags == [True, False]
            assert executor.groups_by_input_index == {}
            assert len(executor.groups_by_input_block) == 2

        ids = []
        for seed in range(executor.cols):
            torch.manual_seed(int(seed))
            packed = torch.randn((scheme.params.get_slots(),), dtype=torch.float32) * 0.01
            ct = scheme.encrypt(scheme.encode(packed, len(scheme.params.get_logq()) - 1))
            ids.append(int(ct.ids[0]))
            ct.ids = []
        source = CipherTensor(
            scheme,
            ids,
            module.input_shape,
            module.fhe_input_shape,
        )

        out = group.output(str(node_name), source)

        assert isinstance(out, CipherTensor)
        assert len(out.ids) == executor.rows
        assert group.execute_count == 1
    finally:
        scheme.delete_scheme()


def test_r34_single_flow_group_compiles_and_executes_runtime(monkeypatch) -> None:
    _require_lattigo()
    dag = _prepared_r34_imagenet_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()

    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
    registry.attach_to_dag(dag)
    module = dag.nodes["avgpool"]["module"]
    group = module.region_runtime
    executor = group.executor
    assert isinstance(executor, InputPairConvRuntimeExecutor)

    def fake_pack_conv2d(conv_layer, last):
        assert last is False
        slots = int(conv_layer.scheme.params.get_slots())
        return (
            {
                (0, 0): {0: [1.0] * slots},
                (0, 1): {1: [0.0] * slots},
                (1, 0): {0: [0.5] * slots},
                (1, 1): {2: [0.0] * slots},
            },
            0,
        )

    def fake_bias(_conv_layer):
        slots = int(module.scheme.params.get_slots())
        return torch.zeros((2 * slots,), dtype=torch.float32)

    monkeypatch.setattr(packing, "pack_conv2d", fake_pack_conv2d)
    monkeypatch.setattr(packing, "construct_conv2d_bias", fake_bias)

    config = {
        "ckks_params": {"LogN": 16, "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    scheme.init_scheme(config)
    try:
        Module.set_scheme(scheme)
        module.level = len(scheme.params.get_logq()) - 1
        module.depth = 1

        group.compile(scheme)
        assert executor.compile_count == 1
        assert executor.cols == 2
        assert executor.rows == 2

        ids = []
        for seed in range(executor.cols):
            torch.manual_seed(int(seed))
            packed = torch.randn((scheme.params.get_slots(),), dtype=torch.float32) * 0.01
            ct = scheme.encrypt(scheme.encode(packed, len(scheme.params.get_logq()) - 1))
            ids.append(int(ct.ids[0]))
            ct.ids = []
        source = CipherTensor(
            scheme,
            ids,
            module.input_shape,
            module.fhe_input_shape,
        )

        out = group.output("avgpool", source)

        assert isinstance(out, CipherTensor)
        assert len(out.ids) == executor.rows
        assert group.execute_count == 1
    finally:
        scheme.delete_scheme()


def test_r34_policy_matches_simple_source_group_rule() -> None:
    assert r34_same_shape.r34_same_shape_policy(c=64, gap=4) == "inter_group_hybrid"
    assert r34_same_shape.r34_same_shape_policy(c=128, gap=8) == "inter_group_hybrid"
    assert r34_same_shape.r34_same_shape_policy(c=256, gap=16) == "intra_group_pack2"
    assert r34_same_shape.r34_same_shape_policy(c=512, gap=32) == "intra_group_pack2"


def test_r34_inter_group_policy_attaches_native_runtime_executor() -> None:
    dag = _prepared_r34_imagenet_dag()
    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
    registry.attach_to_dag(dag)

    stage1_module = dag.nodes["layers_0_0_conv1"]["module"]
    stage2_module = dag.nodes["layers_1_1_conv1"]["module"]

    assert getattr(stage1_module, "region_kernel_policy") == "inter_group_hybrid"
    assert getattr(stage2_module, "region_kernel_policy") == "inter_group_hybrid"
    assert stage1_module.region_runtime.executable is True
    assert stage2_module.region_runtime.executable is True
    assert isinstance(stage1_module.region_runtime.executor, r34_same_shape.R34InterGroupHybridSameShapeRuntimeExecutor)
    assert isinstance(stage2_module.region_runtime.executor, r34_same_shape.R34InterGroupHybridSameShapeRuntimeExecutor)
