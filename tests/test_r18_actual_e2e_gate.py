from __future__ import annotations

from pathlib import Path

import pytest
import torch

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.orion import scheme
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental.cir.lattigo_block import build_r18_stage3_shared_block_plan
from orion.experimental.cir.runtime_group import (
    FullConvRegionRuntimeExecutor,
    RegionFirstCompileRegistry,
    build_r18_actual_region_first_e2e_report,
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


def test_r18_e2e_compile_registry_attaches_full_conv_region_nodes() -> None:
    _net, dag = _prepared_r18_tiny_dag()

    registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
    audit = registry.attach_to_dag(dag)

    assert registry.graph_audit["replacement_mode"] == "full_conv_region_nodes"
    assert audit["executable_region_count"] == len(registry.groups)
    assert len(registry.groups) == 10
    assert registry.graph_audit["excluded_nodes"]
    for group in registry.groups:
        assert len(group.conv_nodes) == 1
        assert group.executable is True
        assert isinstance(group.executor, FullConvRegionRuntimeExecutor)
        module = dag.nodes[group.conv_nodes[0]]["module"]
        assert getattr(module, "region_first_skip_dense_pack") is True
        assert getattr(module, "region_runtime") is group


def test_r18_actual_e2e_report_builds_gate_without_dense_pack(monkeypatch) -> None:
    def fail_pack_conv2d(*_args, **_kwargs):
        raise AssertionError("actual E2E gate must not materialize dense conv masks")

    monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)

    payload = build_r18_actual_region_first_e2e_report()

    assert payload["status"] == "ready_for_manual_ckks_forward"
    assert payload["dense_cleartext"]["ran"] is True
    assert payload["region_first"]["replacement_mode"] == "full_conv_region_nodes"
    assert payload["region_first"]["runtime_group_count"] == 10
    assert payload["e2e_gate"]["graph_replacement_ready"] is True
    assert payload["e2e_gate"]["source_pairing_runtime_ready"] is True
    assert payload["e2e_gate"]["output_assembly_runtime_ready"] is True
    assert payload["claim"]["full_network_ckks"] is False
    assert payload["claim"]["runtime_speedup_publishable"] is False


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
        assert tuple(out.decrypt().decode().shape) == (1, 16, 64, 64)
        assert executor.compile_count == 1
        assert executor.block_evaluate_count == 1
    finally:
        scheme.delete_scheme()
