from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.orion import scheme
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental.cir import r34_orion_same_shape as r34_same_shape
from orion.experimental.cir.halo_local_conv_provider import (
    HaloLocalBranchPairConvRuntimeExecutor,
    HaloLocalConvRuntimeExecutor,
)
from orion.experimental.cir.transition_pool_provider import (
    BranchPairConvRuntimeExecutor,
    BranchPairNoHybridConvRuntimeExecutor,
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


def _executor_delegate(executor):
    return getattr(executor, "delegate", getattr(executor, "_delegate", executor))


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
    assert isinstance(stem_conv.region_runtime.executor, HaloLocalConvRuntimeExecutor)
    assert type(_executor_delegate(stem_conv.region_runtime.executor)).__name__ == "NativeHaloStripeNoRIConvExecutor"
    assert stem_conv.region_runtime.plan["runtime_lowering"] == "provider_executable+native_halo_stripe_no_ri"
    assert stem_conv.region_runtime.plan["halo_layout_gate"]["decision"] == "skip"
    assert stem_conv.region_runtime.plan["halo_layout_gate"]["dense_shared_rotations"] == 610
    assert stem_conv.region_runtime.plan["halo_layout_gate"]["halo_total_shared_rotations"] == 1080
    assert stem_conv.region_runtime.executable is True

    assert getattr(stem_pool, "region_family_label") == "stem_pool"
    assert getattr(stem_pool, "region_source_group_count") == 16
    assert getattr(stem_pool, "region_kernel_policy") == "inter_group_hybrid"
    assert type(stem_pool.region_runtime.executor).__name__ == "R34DenseSingleFlowRuntimeExecutor"
    assert stem_pool.region_runtime.plan["runtime_lowering"] == "dense_orion_pool_shared_rotations"
    assert stem_pool.region_runtime.plan["halo_layout_gate"]["decision"] == "skip"
    assert stem_pool.region_runtime.plan["halo_layout_gate"]["dense_shared_rotations"] == 635
    assert stem_pool.region_runtime.plan["halo_layout_gate"]["halo_total_shared_rotations"] == 5343
    assert stem_pool.region_runtime.executable is True

    assert getattr(exit_pool, "region_family_label") == "global_avgpool_exit"
    assert getattr(exit_pool, "region_source_group_count") == 1
    assert getattr(exit_pool, "region_kernel_policy") == "intra_group_pack2"
    assert type(exit_pool.region_runtime.executor).__name__ == "R34DenseSingleFlowRuntimeExecutor"
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
    assert isinstance(stage2_module.region_runtime.executor, HaloLocalConvRuntimeExecutor)
    assert isinstance(
        _executor_delegate(stage2_module.region_runtime.executor),
        r34_same_shape.NativeAlignedHaloNoRIConvExecutor,
    )

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
    assert isinstance(stage2_transition_conv.region_runtime.executor, HaloLocalBranchPairConvRuntimeExecutor)
    assert isinstance(_executor_delegate(stage2_transition_conv.region_runtime.executor), BranchPairNoHybridConvRuntimeExecutor)
    assert stage2_transition_conv.region_runtime.plan["halo_layout_gate"]["decision"] == "skip"
    assert stage2_transition_conv.region_runtime.plan["halo_layout_gate"]["dense_shared_rotations"] == 1045
    assert stage2_transition_conv.region_runtime.plan["halo_layout_gate"]["halo_total_shared_rotations"] == 1196

    stage4_transition_conv = dag.nodes["layers_3_0_conv1"]["module"]
    stage4_transition_shortcut = dag.nodes["layers_3_0_shortcut_0"]["module"]
    assert stage4_transition_conv.region_runtime is stage4_transition_shortcut.region_runtime
    assert isinstance(stage4_transition_conv.region_runtime.executor, HaloLocalBranchPairConvRuntimeExecutor)
    assert isinstance(_executor_delegate(stage4_transition_conv.region_runtime.executor), BranchPairNoHybridConvRuntimeExecutor)
    assert getattr(stage4_transition_conv, "region_source_group_count") == 1
    assert getattr(stage4_transition_conv, "region_kernel_policy") == "intra_group_pack2"
    assert stage4_transition_conv.region_runtime.plan["halo_layout_gate"]["decision"] == "skip"
    assert stage4_transition_conv.region_runtime.plan["halo_layout_gate"]["blocking_stages"] == (
        "stage2_transition",
        "stage3_transition",
    )
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
    assert isinstance(executor, HaloLocalBranchPairConvRuntimeExecutor)
    assert isinstance(_executor_delegate(executor), BranchPairNoHybridConvRuntimeExecutor)

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
        ("layers_1_1_conv1", r34_same_shape.NativeAlignedHaloNoRIConvExecutor),
        ("layers_2_1_conv1", r34_same_shape.NativeAlignedHaloNoRIConvExecutor),
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
    assert isinstance(executor, HaloLocalConvRuntimeExecutor)
    assert isinstance(_executor_delegate(executor), executor_type)

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
    monkeypatch.setattr(r34_same_shape, "build_r34_native_aligned_halo_no_ri_plan", fake_same_shape_plan)

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
        delegate = _executor_delegate(executor)
        assert executor.cols == 3
        assert executor.rows == 2
        assert executor.groups_by_input_index

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
    assert type(executor).__name__ == "R34DenseSingleFlowRuntimeExecutor"

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


def test_r34_hardcoded_same_shape_relayout_plan_is_halo_local_and_reduced() -> None:
    expected = {
        "stage1_same": (8, 8, 4, 14, 47),
        "stage2_same": (4, 4, 2, 6, 14),
        "stage3_same": (2, 2, 1, 2, 4),
        "stage4_same": (3, 3, 2, 4, 4),
    }

    for family_label, (stripe_count, raw_tasks, effective_tasks, relayout_ops, legacy_tasks) in expected.items():
        plan = r34_same_shape.r34_same_shape_hardcoded_relayout_plan(family_label=family_label)
        payload = plan.to_dict()

        assert payload["runtime_layout"] == "height_stripe_halo_local_hardcoded"
        assert payload["conv_dependency"] == "current_ciphertext_materialized_halo_only"
        assert payload["stripe_count"] == stripe_count
        assert payload["raw_conv_lt_tasks"] == raw_tasks
        assert payload["effective_conv_lt_tasks_with_hybrid"] == effective_tasks
        assert payload["legacy_flat_conv_lt_tasks"] == legacy_tasks
        assert payload["legacy_flat_offdiag_tasks"] > 0
        assert payload["relayout_rotations"] == relayout_ops
        assert payload["relayout_mask_mults"] == relayout_ops
        assert payload["max_active_slots"] <= 32768
        assert payload["conv_source_target_pairs"] == [[index, index] for index in range(stripe_count)]
        assert payload["effective_conv_lt_tasks_with_hybrid"] < payload["legacy_flat_conv_lt_tasks"]
        assert all(int(stripe["core_shift_rows"]) == 0 for stripe in payload["stripes"])
        assert all(
            int(stripe["relayout_rotations"])
            == int(stripe["relayout_mask_mults"])
            == int(stripe["halo_top"] > 0 and stripe["index"] > 0)
            + int(stripe["halo_bottom"] > 0 and stripe["index"] + 1 < stripe_count)
            for stripe in payload["stripes"]
        )


def test_r34_same_shape_groups_expose_native_aligned_halo_plan() -> None:
    dag = _prepared_r34_imagenet_dag()
    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
    registry.attach_to_dag(dag)

    for node_name in ("layers_0_0_conv1", "layers_1_1_conv1", "layers_2_1_conv1", "layers_3_1_conv1"):
        module = dag.nodes[node_name]["module"]
        group = module.region_runtime
        plan = dict(group.plan)
        relayout = dict(plan["relayout_plan"])
        executor = group.executor
        metadata = executor.compile_cache_metadata()

        assert plan["runtime_lowering"] == "provider_executable+native_aligned_halo_no_ri"
        assert "native_aligned_halo_no_ri" in group.boundary_actions
        assert relayout["runtime_layout"] == "native_aligned_halo_no_ri"
        assert relayout["conv_dependency"] == "native_aligned_halo_source_tiles"
        assert plan["conv_lt_effective_submatrix_tasks"] <= plan["conv_lt_raw_submatrix_tasks"]
        assert plan["native_cb_shared_rotations"] <= plan["native_c_only_rotations"]
        assert metadata["r34_same_shape_halo_relayout_plan"]["submatrix_program_count"] == relayout[
            "submatrix_program_count"
        ]
        assert metadata["conv_lt_effective_submatrix_tasks"] == relayout["sharing_group_count"]


def test_r34_native_relayout_kernel_loads_sparse_lt_manifest() -> None:
    spec = r34_same_shape.r34_same_shape_spec_for_family_label("stage4_same")
    native_plan = r34_same_shape.r34_native_aligned_halo_plan(spec)
    seen: list[dict[tuple[int, int], tuple[int, ...]]] = []

    class _FakeLtEvaluator:
        def generate_transforms(self, kernel):
            seen.append(
                {
                    (int(row), int(col)): tuple(sorted(int(idx) for idx in block.keys()))
                    for (row, col), block in kernel.diagonals.items()
                }
            )
            return {
                (int(row), int(col)): int(index + 100)
                for index, (row, col) in enumerate(sorted(kernel.diagonals))
            }

    fake_scheme = SimpleNamespace(
        backend=SimpleNamespace(DeleteLinearTransform=lambda _transform_id: None),
        lt_evaluator=_FakeLtEvaluator(),
    )
    kernel = r34_same_shape.R34NativeAlignedRelayoutKernel(
        spec=spec,
        native_plan=native_plan,
        direction="compact_to_native",
        name="stage4_test_compact_to_native",
        output_shape=torch.Size([2, spec.slot_count]),
        fhe_output_shape=torch.Size([2, spec.slot_count]),
    )

    kernel.compile_from_cache_metadata(
        fake_scheme,
        {
            "name": "stage4_test_compact_to_native",
            "blocks": [
                {"row": 1, "col": 0, "diag_indices": [7, 3], "slot_count": spec.slot_count},
                {"row": 0, "col": 0, "diag_indices": [0], "slot_count": spec.slot_count},
            ],
        },
        level=2,
    )

    assert seen == [{(0, 0): (0,), (1, 0): (3, 7)}]
    assert kernel.transform_ids == {(0, 0): 100, (1, 0): 101}
    metadata = kernel.to_metadata()
    assert metadata["level"] == 2
    assert metadata["lt_tasks"] == 2
    assert metadata["diagonal_count"] == 3
    assert metadata["blocks"][1]["diag_indices"] == [3, 7]


def test_r34_native_executor_loads_relayout_and_conv_manifest(monkeypatch) -> None:
    spec = r34_same_shape.r34_same_shape_spec_for_family_label("stage4_same")
    relayout_seen: list[tuple[str, dict[tuple[int, int], tuple[int, ...]]]] = []

    class _FakeParams:
        def get_io_mode(self):
            return "load"

        def get_logq(self):
            return [45, 30, 30, 45]

        def get_default_scale(self):
            return 1 << 30

    class _FakeLtEvaluator:
        def generate_transforms(self, kernel):
            relayout_seen.append(
                (
                    str(kernel.name),
                    {
                        (int(row), int(col)): tuple(sorted(int(idx) for idx in block.keys()))
                        for (row, col), block in kernel.diagonals.items()
                    },
                )
            )
            return {
                (int(row), int(col)): int(index + 200)
                for index, (row, col) in enumerate(sorted(kernel.diagonals))
            }

    def _fake_compile_unified(self, _backend):
        self._compiled_by_test = True

    monkeypatch.setattr(r34_same_shape.UnifiedTransformGroup, "compile_unified", _fake_compile_unified)
    module = SimpleNamespace(
        on_weight=torch.zeros(spec.weight_shape, dtype=torch.float32),
        on_bias=torch.zeros((int(spec.c),), dtype=torch.float32),
        input_shape=torch.Size([1, int(spec.c), int(spec.h), int(spec.w)]),
        output_shape=torch.Size([1, int(spec.c), int(spec.h), int(spec.w)]),
        fhe_input_shape=torch.Size([1, 1, int(spec.h * spec.gap), int(spec.w * spec.gap)]),
        fhe_output_shape=torch.Size([1, 1, int(spec.h * spec.gap), int(spec.w * spec.gap)]),
        input_gap=int(spec.gap),
        output_gap=int(spec.gap),
        stride=(1, 1),
        padding=(1, 1),
    )
    executor = r34_same_shape.NativeAlignedHaloNoRIConvExecutor(
        module=module,
        spec=spec,
        output_node_id="stage4_test",
    )
    executor.assigned_level = 3
    executor.load_compile_cache_metadata(
        {
            "rows": 2,
            "cols": 2,
            "input_relayout": {
                "name": "stage4_test_compact_to_native_halo",
                "blocks": [{"row": 0, "col": 0, "diag_indices": [0, 1], "slot_count": spec.slot_count}],
            },
            "groups_by_input_index": [
                {"input_index": 0, "storage_key": "group_10", "target_indices": [0, 1]},
            ],
            "output_relayout": {
                "name": "stage4_test_native_halo_to_compact",
                "blocks": [{"row": 0, "col": 0, "diag_indices": [2], "slot_count": spec.slot_count}],
            },
        }
    )
    fake_scheme = SimpleNamespace(
        params=_FakeParams(),
        backend=SimpleNamespace(DeleteLinearTransform=lambda _transform_id: None),
        lt_evaluator=_FakeLtEvaluator(),
        encode=lambda values, level, scale: SimpleNamespace(values=values, level=level, scale=scale),
    )

    assert executor._compile_from_cache_metadata(fake_scheme) is True

    assert relayout_seen == [
        ("stage4_test_compact_to_native_halo", {(0, 0): (0, 1)}),
        ("stage4_test_native_halo_to_compact", {(0, 0): (2,)}),
    ]
    assert executor.input_relayout_kernel is not None
    assert executor.output_relayout_kernel is not None
    assert executor.input_relayout_kernel.transform_ids == {(0, 0): 200}
    assert executor.output_relayout_kernel.transform_ids == {(0, 0): 200}
    assert executor.target_indices_by_input_index == {0: (0, 1)}


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
    assert isinstance(stage1_module.region_runtime.executor, HaloLocalConvRuntimeExecutor)
    assert isinstance(stage2_module.region_runtime.executor, HaloLocalConvRuntimeExecutor)
    assert isinstance(
        _executor_delegate(stage1_module.region_runtime.executor),
        r34_same_shape.NativeAlignedHaloNoRIConvExecutor,
    )
    assert isinstance(
        _executor_delegate(stage2_module.region_runtime.executor),
        r34_same_shape.NativeAlignedHaloNoRIConvExecutor,
    )


def test_r34_no_hybrid_ablation_keeps_halo_local_same_shape_facade() -> None:
    from tools import benchmark_node_specific_lattigo_provider_vs_dense as bench

    dag = _prepared_r34_imagenet_dag()
    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
    registry.attach_to_dag(dag)

    stem_module = dag.nodes["conv1"]["module"]
    stem_executor = stem_module.region_runtime.executor
    assert isinstance(stem_executor, HaloLocalConvRuntimeExecutor)

    stem_audit = bench._apply_provider_no_hybrid_ablation("r34_imgnet", stem_module)

    stem_executor = stem_module.region_runtime.executor
    assert stem_audit["status"] == "ok"
    assert stem_audit["mode"] == "native_halo_stripe_no_ri"
    assert stem_audit["executor"] == "HaloLocalConvRuntimeExecutor"
    assert stem_audit["delegate_executor"] == "NativeHaloStripeNoRIConvExecutor"
    assert isinstance(stem_executor, HaloLocalConvRuntimeExecutor)

    for node_name in ("layers_0_0_conv1", "layers_2_1_conv1"):
        module = dag.nodes[node_name]["module"]
        executor = module.region_runtime.executor
        assert isinstance(executor, HaloLocalConvRuntimeExecutor)
        assert bool(executor.use_ct_pt_hybrid_packing) is False
        assert type(_executor_delegate(executor)) is not r34_same_shape.R34OrionSameShapeRuntimeExecutor

        audit = bench._apply_provider_no_hybrid_ablation("r34_imgnet", module)

        executor = module.region_runtime.executor
        assert audit["status"] == "ok"
        assert audit["mode"] == "r34_native_aligned_halo_no_ri"
        assert audit["executor"] == "HaloLocalConvRuntimeExecutor"
        assert audit["delegate_executor"] == "NativeAlignedHaloNoRIConvExecutor"
        assert audit["conv_lt_effective_submatrix_tasks"] <= audit["conv_lt_raw_submatrix_tasks"]
        assert audit["conv_lt_effective_submatrix_tasks"] < audit["legacy_flat_conv_lt_tasks"]
        assert audit["r34_same_shape_halo_relayout_plan"]["runtime_layout"] == "native_aligned_halo_no_ri"
        assert isinstance(executor, HaloLocalConvRuntimeExecutor)
        assert bool(executor.use_ct_pt_hybrid_packing) is False
        assert isinstance(_executor_delegate(executor), r34_same_shape.NativeAlignedHaloNoRIConvExecutor)
        assert str(module.region_runtime.strategy).endswith("_no_hybrid")

    transition_module = dag.nodes["layers_1_0_conv1"]["module"]
    transition_executor = transition_module.region_runtime.executor
    assert isinstance(transition_executor, HaloLocalBranchPairConvRuntimeExecutor)
    assert bool(transition_executor.use_ct_pt_hybrid_packing) is False
    assert isinstance(_executor_delegate(transition_executor), BranchPairNoHybridConvRuntimeExecutor)

    transition_audit = bench._apply_provider_no_hybrid_ablation("r34_imgnet", transition_module)

    transition_executor = transition_module.region_runtime.executor
    assert transition_audit["status"] == "ok"
    assert transition_audit["mode"] == "r34_halo_local_branch_pair_already_no_real_imag_packing"
    assert transition_audit["executor"] == "HaloLocalBranchPairConvRuntimeExecutor"
    assert transition_audit["delegate_executor"] == "BranchPairNoHybridConvRuntimeExecutor"
    assert isinstance(transition_executor, HaloLocalBranchPairConvRuntimeExecutor)
    assert bool(transition_executor.use_ct_pt_hybrid_packing) is False
    assert isinstance(_executor_delegate(transition_executor), BranchPairNoHybridConvRuntimeExecutor)
    assert "branch_pair_real_imag_hybrid" not in transition_module.region_runtime.boundary_actions
    assert "branch_pair_no_real_imag" in transition_module.region_runtime.boundary_actions


def test_r34_public_conv_provider_path_is_generic_halo_local() -> None:
    dag = _prepared_r34_imagenet_dag()
    registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
    registry.attach_to_dag(dag)

    public_executor_types = {
        type(group.executor).__name__
        for group in registry.groups
        if group.executor is not None and group.conv_nodes
    }
    assert "R34InterGroupHybridSameShapeRuntimeExecutor" not in public_executor_types
    assert "R34Pack2SameShapeRuntimeExecutor" not in public_executor_types
    assert {
        "HaloLocalConvRuntimeExecutor",
        "HaloLocalBranchPairConvRuntimeExecutor",
    }.issubset(public_executor_types)
