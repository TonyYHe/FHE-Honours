from __future__ import annotations

import torch

from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer
from orion.backend.python.parameters import NewParameters
from orion.experimental.cir.runtime_group import RegionFirstCompileRegistry, discover_r18_tiny_region_groups
from orion.models.resnet import ResNet18


def test_r18_compile_registry_attaches_region_metadata_to_stage_convs() -> None:
    torch.manual_seed(0)
    traced = OrionTracer().trace_model(ResNet18(dataset="tiny"))
    dag = NetworkDAG(traced)
    dag.build_dag()
    registry = RegionFirstCompileRegistry.for_r18_tiny(dag)

    audit = registry.attach_to_dag(dag)

    assert audit["attached_count"] > 0
    assert audit["executable_region_count"] == 0
    assert audit["fallback_layers"]
    for item in audit["attached"][:5]:
        module = dag.nodes[item["node"]]["module"]
        assert getattr(module, "region_runtime") is not None
        assert getattr(module, "region_output_id") == item["node"]
        assert getattr(module, "region_first_skip_dense_pack") is False


def test_r18_compile_registry_marks_stage1_and_stage2_executable_with_fused_weights() -> None:
    torch.manual_seed(0)
    traced = OrionTracer().trace_model(ResNet18(dataset="tiny"))
    dag = NetworkDAG(traced)
    dag.build_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
    registry = RegionFirstCompileRegistry.for_r18_tiny(dag)

    audit = registry.attach_to_dag(dag)

    assert audit["executable_region_count"] == 2
    assert len(audit["fallback_layers"]) == 8
    stage1 = [item for item in audit["attached"] if item["stage"] == "stage1"]
    stage2 = [item for item in audit["attached"] if item["stage"] == "stage2"]
    assert stage1
    assert stage2
    assert all(item["executable"] is True for item in stage1)
    assert all(item["executable"] is True for item in stage2)
    for item in stage1 + stage2:
        module = dag.nodes[item["node"]]["module"]
        assert getattr(module, "region_first_skip_dense_pack") is True
        assert getattr(module, "region_runtime").executable is True
        assert getattr(module, "region_runtime").executor is not None


def test_r18_region_discovery_depths_are_bootstrap_visible() -> None:
    groups, audit = discover_r18_tiny_region_groups()

    assert audit["selected_region_count"] == 4
    assert {group.stage: group.depth for group in groups} == {
        "stage1": 2,
        "stage2": 2,
        "stage3": 2,
        "stage4": 3,
    }


def test_experimental_region_first_config_flag_is_parsed() -> None:
    params = NewParameters(
        {
            "ckks_params": {"LogN": 12, "LogQ": [45, 30, 45], "LogP": [50], "LogScale": 30},
            "orion": {"experimental_region_first": "r18_tiny"},
        }
    )

    assert params.get_experimental_region_first() == "r18_tiny"
