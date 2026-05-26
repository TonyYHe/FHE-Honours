from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import torch

from orion.core import packing
from orion.core.bootstrap_layout_compression import apply_bootstrap_layout_compression
from orion.core.orion import _region_first_mode_options
from orion.core.orion import scheme
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental.layout_policy_ablation import (
    attach_backend_runtime_anchors,
    attach_non_ckks_simulation,
    attach_runtime_anchor,
    build_planner_ablation,
    build_layout_policy_compile_plan,
    build_u22_dag,
    network_spec,
    run_backend_runtime_anchors,
    run_non_ckks_layout_simulation,
    run_runtime_anchor,
    _fill_beta_to_tile_capacity,
    _layout_for_shape,
    _runtime_config,
)
from orion.models.resnet import BasicBlock, ResNet
from orion.models.unet import UNet22
from orion.experimental.cir.runtime_group import RegionFirstRuntimeGroup
from orion.experimental.cir.transition_pool_provider import InputPairConvRuntimeExecutor
from orion.experimental.u22_phase1 import (
    LayoutPolicyAddRuntimeExecutor,
    LayoutPolicyEncryptedModuleRuntimeExecutor,
    LayoutPolicyProviderRuntimeExecutor,
    LayoutPolicyRelayoutKernel,
    U22CompileRegistry,
    _layout_policy_runtime_compile_plan,
    collect_layout_policy_provider_pressure,
)
from orion.core.auto_bootstrap import BootstrapSolver
from orion.nn.activation import SiLU
from orion.nn.linear import Conv2d, ConvTranspose2d
from orion.nn.module import Module
from orion.nn.operations import Add
from orion.nn.pooling import AvgPool2d


def _policy(payload: dict, name: str) -> dict:
    for row in payload["policies"]:
        if row["policy"] == str(name):
            return row
    raise AssertionError(f"missing policy {name}")


def test_layout_policy_runtime_config_uses_resnet_e2e_logq_budget() -> None:
    config = _runtime_config(backend="python", provider_mode="u22_64_base32_layout_eager", logn=15)

    assert config["ckks_params"]["LogQ"] == [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    assert len(config["ckks_params"]["LogQ"]) - 1 == 10
    assert config["ckks_params"]["LogP"] == [61, 61, 61]
    assert config["ckks_params"]["LogScale"] == 40
    assert config["ckks_params"]["H"] == 192
    assert config["boot_params"]["LogP"] == [61, 61, 61, 61, 61, 61, 61, 61]


def _layout_key(row: dict) -> tuple[int, int, int, int]:
    layout = row["selected_layout"]
    return int(layout["top_beta"]), int(layout["bottom_beta"]), int(layout["stride"]), int(layout["gap"])


def _layout_covers(selected: dict, required: dict) -> bool:
    return (
        int(selected["gap"]) == int(required["gap"])
        and int(selected["top_beta"]) >= int(required["top_beta"])
        and int(selected["bottom_beta"]) >= int(required["bottom_beta"])
        and int(selected["stride"]) >= int(required["stride"])
    )


def test_capacity_fill_balances_extra_halo_without_growing_tile_count() -> None:
    shape = (1, 512, 28, 28)
    required = _layout_for_shape(shape=shape, gap=8, top_beta=1, bottom_beta=1, stride=1, slots=32768)

    filled = _fill_beta_to_tile_capacity(required, shape=shape, slots=32768)

    assert int(required.tile_count) == 14
    assert int(filled.tile_count) == int(required.tile_count)
    assert (int(filled.top_beta), int(filled.bottom_beta)) == (2, 2)


def _init_python_scheme(provider_mode: str) -> None:
    config = {
        "ckks_params": {
            "LogN": 15,
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
            "experimental_region_first": str(provider_mode),
        },
    }
    scheme.init_scheme(config)
    Module.set_scheme(scheme)
    Module.set_margin(scheme.params.get_margin())


def _init_long_python_scheme(provider_mode: str = "") -> None:
    config = _runtime_config(backend="python", provider_mode=str(provider_mode), logn=16)
    scheme.init_scheme(config)
    Module.set_scheme(scheme)
    Module.set_margin(scheme.params.get_margin())


class OneDownOneUpUNet(Module):
    def __init__(self, *, base_channels: int = 8, activation_degree: int = 7) -> None:
        super().__init__()
        base = int(base_channels)
        self.enc = Conv2d(3, base, kernel_size=3, padding=1, bias=True)
        self.enc_act = SiLU(degree=int(activation_degree))
        self.pool = AvgPool2d(kernel_size=2, stride=2)
        self.mid = Conv2d(base, base * 2, kernel_size=3, padding=1, bias=True)
        self.mid_act = SiLU(degree=int(activation_degree))
        self.up = ConvTranspose2d(base * 2, base, kernel_size=2, stride=2, bias=True)
        self.add = Add()
        self.dec = Conv2d(base, base, kernel_size=3, padding=1, bias=True)
        self.dec_act = SiLU(degree=int(activation_degree))
        self.out = Conv2d(base, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc_act(self.enc(x))
        x = self.mid_act(self.mid(self.pool(skip)))
        x = self.add(self.up(x), skip)
        x = self.dec_act(self.dec(x))
        return self.out(x)


def _prepared_one_down_one_up_dag(*, image_size: int = 192, base_channels: int = 8) -> NetworkDAG:
    torch.manual_seed(0)
    model = OneDownOneUpUNet(base_channels=int(base_channels), activation_degree=7)
    model.eval()
    traced = OrionTracer().trace_model(model)
    StatsTracker(traced).propagate(
        torch.randn((1, 3, int(image_size), int(image_size)), dtype=torch.float32)
    )
    dag = NetworkDAG(traced)
    dag.build_dag()
    Module.set_margin(2)
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "fit"):
            module.fit()
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()
    return dag


def test_u22_64_layout_policy_planner_reports_all_edges_and_ordering() -> None:
    payload = build_planner_ablation(
        network="u22_64_base32",
        policies=("fixed_max", "always", "greedy", "dp"),
    )

    assert payload["graph"]["node_count"] == 31
    assert payload["graph"]["edge_count"] == 34
    assert [row["policy"] for row in payload["policies"]] == ["fixed_max", "always", "greedy", "dp"]
    for row in payload["policies"]:
        assert row["metric_source"] == "planner_estimate"
        assert len(row["edge_layouts"]) == 34

    fixed = _policy(payload, "fixed_max")
    always = _policy(payload, "always")
    greedy = _policy(payload, "greedy")
    dp = _policy(payload, "dp")
    for policy in (fixed, greedy):
        assert policy["relayouts"] == 0
        assert policy["halo_redundancy_ratio"] == 0.0
        assert all(
            int(row["required_layout"]["top_beta"]) == 0 and int(row["required_layout"]["bottom_beta"]) == 0
            for row in policy["edge_layouts"]
        )
        assert all(
            int(row["selected_layout"]["top_beta"]) == 0 and int(row["selected_layout"]["bottom_beta"]) == 0
            for row in policy["edge_layouts"]
        )
    assert always["relayouts"] == 33
    assert always["halo_redundancy_ratio"] == 0.0
    assert all(
        int(row["required_layout"]["top_beta"]) == 0 and int(row["required_layout"]["bottom_beta"]) == 0
        for row in always["edge_layouts"]
    )
    assert all(
        int(row["selected_layout"]["top_beta"]) == 0 and int(row["selected_layout"]["bottom_beta"]) == 0
        for row in always["edge_layouts"]
    )
    assert dp["relayouts"] == 0
    assert dp["halo_redundancy_ratio"] >= 0.0


def test_u22_128_layout_policy_elides_halo_when_height_strip_fits_single_ct() -> None:
    payload = build_planner_ablation(
        network="u22_128_base32",
        policies=("fixed_max", "always", "greedy", "dp"),
    )

    for policy in [row for row in payload["policies"] if row["policy"] != "dp"]:
        if policy["policy"] == "always":
            assert policy["relayouts"] == 33
        else:
            assert policy["relayouts"] == 0
        assert policy["halo_redundancy_ratio"] == 0.0
        assert all(
            int(row["required_layout"]["top_beta"]) == 0 and int(row["required_layout"]["bottom_beta"]) == 0
            for row in policy["edge_layouts"]
        )
        assert all(
            int(row["selected_layout"]["top_beta"]) == 0 and int(row["selected_layout"]["bottom_beta"]) == 0
            for row in policy["edge_layouts"]
        )
    assert _policy(payload, "dp")["relayouts"] == 0


def test_non_dp_runtime_relayout_insertion_keeps_policy_specific_producer_layouts() -> None:
    def runtime_count(policy: str) -> tuple[int, int]:
        plan = build_layout_policy_compile_plan(build_u22_dag(network_spec("u22_256_base32")), policy=policy)
        runtime_plan = _layout_policy_runtime_compile_plan(plan)
        return int(len(runtime_plan["node_layouts"])), int(runtime_plan["relayout_edge_count"])

    assert runtime_count("fixed_max") == (30, 5)
    assert runtime_count("always") == (30, 33)
    assert runtime_count("greedy") == (30, 4)
    assert runtime_count("dp") == (30, 0)


def test_u22_64_layout_policy_planner_aligns_add_inputs() -> None:
    payload = build_planner_ablation(network="u22_64_base32")

    for policy in payload["policies"]:
        rows = list(policy["edge_layouts"])
        for add_node in ("add4", "add3", "add2", "add1"):
            incoming = [row for row in rows if row["target"] == add_node]
            assert len(incoming) == 2
            assert len({_layout_key(row) for row in incoming}) == 1


def test_layout_policy_dp_allows_only_native_compact_align_shared_fallback() -> None:
    payload = build_planner_ablation(network="u22_256_base32", policies=("dp",))
    dp = _policy(payload, "dp")

    assert int(dp["compact_fallback_penalty_estimate"]) == 0
    fallback = [row for row in dp["edge_layouts"] if row.get("layout_mode") == "compact_global_fallback"]
    assert fallback == []
    uncovered = [
        row
        for row in dp["edge_layouts"]
        if row["op_kind"] in {"conv2d", "avgpool2d"}
        and not _layout_covers(row["selected_layout"], row["required_layout"])
    ]
    assert all(
        bool(row.get("consumer_fused_relayout", False))
        and row.get("layout_mode") in {"compact_align_shared", "compact_halo_shared"}
        for row in uncovered
    )
    shared_rows = [
        row
        for row in dp["edge_layouts"]
        if row.get("layout_mode") in {"compact_align_shared", "compact_halo_shared"}
    ]
    assert shared_rows
    assert int(dp["consumer_fused_relayout_count"]) == len(shared_rows)
    assert all(not bool(row.get("relayout", False)) for row in shared_rows)
    native_conv_halo = [
        row
        for row in dp["edge_layouts"]
        if row["op_kind"] == "conv2d" and row.get("layout_mode") == "native_halo_stripe"
    ]
    assert native_conv_halo
    assert all(row.get("physical_layout") == "native_source_stripe" for row in native_conv_halo)


def test_layout_policy_parser_marks_non_dp_u22_modes_as_provider_executable() -> None:
    opts = _region_first_mode_options("u22_64_base32_layout_eager")
    assert opts["u22_layout_policy"] == "eager"
    assert opts["u22_allowed_nodes"] == ("up1", "up2", "up3", "up4")
    assert _region_first_mode_options("u22_64_base32_layout_always")["u22_layout_policy"] == "always"

    dag = build_u22_dag(network_spec("u22_64_base32"))
    registry = U22CompileRegistry.for_dag(
        dag,
        allowed_nodes=opts["u22_allowed_nodes"],
        enable_conv_kernels=bool(opts["u22_conv_kernels"]),
        layout_policy=str(opts["u22_layout_policy"]),
    )
    audit = registry.attach_to_dag(dag)

    assert audit["attached_count"] > 0
    assert audit["executable_region_count"] == audit["attached_count"]
    assert audit["graph_audit"]["layout_policy"] == "eager"
    assert audit["graph_audit"]["layout_policy_runtime"] == "provider_executable_layout_policy"
    assert audit["graph_audit"]["layout_policy_runtime_lowering"] == "provider_executable+compact_layout"
    assert audit["graph_audit"]["layout_policy_compile_plan_consumed"] is True
    assert audit["graph_audit"]["layout_policy_edge_layout_count"] == 34
    assert audit["graph_audit"]["layout_policy_compile_plan_region_count"] == audit["attached_count"]
    assert audit["graph_audit"]["layout_policy_provider_executable_region_count"] == audit["attached_count"]
    assert audit["graph_audit"]["layout_policy_backend_executable_region_count"] == 0
    assert audit["graph_audit"]["layout_policy_native_halo_provider_region_count"] == 0
    assert all(group.fallback_reason == "" for group in registry.groups)
    assert all(isinstance(group.executor, LayoutPolicyProviderRuntimeExecutor) for group in registry.groups)
    assert all(group.executor.base_executor is not None for group in registry.groups)
    assert not any(group.materializer == "backend_encrypted_module" for group in registry.groups)
    runtime_lowerings = {group.plan["runtime_lowering"] for group in registry.groups}
    assert runtime_lowerings == {"provider_executable+compact_layout"}
    assert {
        group.plan["provider_materializer"]
        for group in registry.groups
    } >= {
        "halo_supported_tconv",
        "u22_input_pair_conv_shared_rotations",
        "u22_pool_input_pair_shared_rotations",
    }
    fake_lattigo_scheme = type(
        "FakeScheme",
        (),
        {"params": type("FakeParams", (), {"get_backend": lambda self: "lattigo", "get_slots": lambda self: 32768})()},
    )()
    assert all(group.supports_scheme(fake_lattigo_scheme) for group in registry.groups)


def test_non_dp_layout_policy_eager_wraps_provider_with_relayout_depth() -> None:
    dag = build_u22_dag(network_spec("u22_256_base32"))
    registry = U22CompileRegistry.for_dag(
        dag,
        allowed_nodes=("up1", "up2", "up3", "up4"),
        enable_conv_kernels=True,
        layout_policy="eager",
    )

    relayout_groups = [
        group
        for group in registry.groups
        if isinstance(group.executor, LayoutPolicyProviderRuntimeExecutor) and group.executor.relayout_rows
    ]
    assert relayout_groups
    assert all(any(str(action).startswith("relayout_kernel_depth_") for action in group.boundary_actions) for group in relayout_groups)
    assert all(group.depth >= 2 for group in relayout_groups)
    assert all(group.solver_depth == group.depth for group in relayout_groups)
    assert all(group.effective_depth() == group.depth for group in relayout_groups)
    assert {
        type(group.executor.base_executor).__name__
        for group in registry.groups
    } >= {
        "HaloSupportedTConvRuntimeExecutor",
        "InputPairConvRuntimeExecutor",
    }


def test_layout_policy_dp_registry_wraps_provider_runtime_and_materializes_halo_plan() -> None:
    opts = _region_first_mode_options("u22_64_base32_layout_dp")
    dag = build_u22_dag(network_spec("u22_64_base32"))
    registry = U22CompileRegistry.for_dag(
        dag,
        allowed_nodes=opts["u22_allowed_nodes"],
        enable_conv_kernels=bool(opts["u22_conv_kernels"]),
        layout_policy=str(opts["u22_layout_policy"]),
    )
    audit = registry.attach_to_dag(dag)

    assert audit["attached_count"] > 0
    assert audit["executable_region_count"] > 0
    assert audit["graph_audit"]["layout_policy"] == "dp"
    assert audit["graph_audit"]["layout_policy_runtime"] == "provider_executable_layout_policy"
    assert audit["graph_audit"]["layout_policy_runtime_lowering"] in {
        "provider_executable+compact_layout",
        "provider_executable+native_halo_layout",
        "provider_executable+mixed_layout_policy",
    }
    assert audit["graph_audit"]["layout_policy_compile_plan_consumed"] is True
    assert audit["graph_audit"]["layout_policy_edge_layout_count"] == 34
    assert audit["graph_audit"]["layout_policy_summary"]["relayout_depth_estimate"] >= 0
    assert all(isinstance(group.executor, LayoutPolicyProviderRuntimeExecutor) for group in registry.groups)


def test_non_dp_layout_policy_executor_uses_encrypted_module_backend() -> None:
    torch.manual_seed(0)
    _init_python_scheme("")
    try:
        conv = Conv2d(1, 1, kernel_size=1, padding=0, bias=True)
        conv.weight.data.fill_(2.0)
        conv.bias.data.fill_(0.25)
        conv.init_orion_params()
        conv.input_shape = torch.Size((1, 1, 4, 4))
        conv.output_shape = torch.Size((1, 1, 4, 4))
        conv.fhe_input_shape = torch.Size((1, 1, 4, 4))
        conv.fhe_output_shape = torch.Size((1, 1, 4, 4))
        conv.input_gap = 1
        conv.output_gap = 1
        conv.name = "layout_policy_conv"
        conv.set_level(len(scheme.params.get_logq()) - 1)
        conv.set_depth(3)
        conv.he_mode = True
        executor = LayoutPolicyEncryptedModuleRuntimeExecutor(
            module=conv,
            output_node_id="conv",
            compile_plan={
                "policy": "fixed_max",
                "edge_layouts": [
                    {
                        "edge": "input->conv",
                        "source": "input",
                        "target": "conv",
                        "shape": [1, 1, 4, 4],
                        "relayout": True,
                        "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1},
                    },
                ],
            },
        )
        group = RegionFirstRuntimeGroup(
            region_id="layout_policy_conv",
            network="U22",
            stage="layout_policy_conv",
            module_prefix="conv",
            conv_nodes=("conv",),
            strategy="layout_policy_backend_encrypted_module",
            materializer="backend_encrypted_module",
            depth=3,
            solver_depth=3,
            boundary_actions=("layout_policy_compile_plan", "backend_encrypted_module"),
            expected_stats={},
            executable=True,
            fallback_reason="",
            output_node_ids=("conv",),
            executor=executor,
        )
        group.assigned_level = int(conv.level)
        group.assigned_depth = int(conv.depth)
        conv.region_runtime = group
        conv.region_output_id = "conv"
        conv.region_first_skip_dense_pack = True
        x = torch.randn((1, 1, 4, 4), dtype=torch.float32)
        reference = torch.nn.functional.conv2d(x, conv.weight, conv.bias)
        x_ct = scheme.encrypt(scheme.encode(x, conv.level))
        out_ct = conv(x_ct)
        decoded = out_ct.decrypt().decode().detach().cpu().to(dtype=torch.float32)
        assert tuple(int(value) for value in decoded.shape) == tuple(int(value) for value in reference.shape)
        assert float((decoded - reference).abs().max().item()) <= 1.0e-4
        assert executor.compile_count == 1
        assert executor.execute_count == 1
        assert executor.last_runtime_io["runtime_lowering"] == "backend_encrypted_module"
        assert executor.last_runtime_io["backend"] == "python"
        assert executor.last_runtime_io["relayout_kernel"] is True
        assert executor.last_runtime_io["relayout_kernel_count"] == 2
        assert conv.depth == 3
    finally:
        scheme.delete_scheme()


def test_layout_policy_relayout_kernel_fills_and_roundtrips_compact_halo_layout() -> None:
    _init_python_scheme("")
    try:
        edge_row = {
            "edge": "input->conv",
            "source": "input",
            "target": "conv",
            "shape": [1, 1, 2, 3],
            "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1},
        }
        x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        level = len(scheme.params.get_logq()) - 1
        x_ct = scheme.encrypt(scheme.encode(x, level))
        pad = LayoutPolicyRelayoutKernel(
            edge_row=edge_row,
            node="conv",
            direction="compact_to_halo",
            index=0,
        )
        pad.compile(scheme, level=int(level))
        halo_ct = pad.apply(x_ct)
        halo = halo_ct.decrypt().decode().detach().cpu().to(dtype=torch.float32)
        assert tuple(int(value) for value in halo.shape) == (1, 1, 4, 3)
        assert torch.equal(halo[:, :, 1:3, :], x)
        assert torch.equal(halo[:, :, 0, :], x[:, :, 0, :])
        assert torch.equal(halo[:, :, 3, :], x[:, :, -1, :])

        trim = LayoutPolicyRelayoutKernel(
            edge_row=edge_row,
            node="conv",
            direction="halo_to_compact",
            index=1,
        )
        trim.compile(scheme, level=int(halo_ct.level()))
        compact_ct = trim.apply(halo_ct)
        compact = compact_ct.decrypt().decode().detach().cpu().to(dtype=torch.float32)
        assert tuple(int(value) for value in compact.shape) == tuple(int(value) for value in x.shape)
        assert torch.equal(compact, x)
    finally:
        scheme.delete_scheme()


def test_layout_policy_relayout_kernel_fuses_output_affine() -> None:
    _init_python_scheme("")
    try:
        edge_row = {
            "edge": "input->conv",
            "source": "input",
            "target": "conv",
            "shape": [1, 1, 2, 3],
            "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1},
        }
        x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        level = len(scheme.params.get_logq()) - 1
        x_ct = scheme.encrypt(scheme.encode(x, level))
        pad = LayoutPolicyRelayoutKernel(
            edge_row=edge_row,
            node="conv",
            direction="compact_to_halo",
            index=0,
        )
        pad.output_scale = 0.5
        pad.output_bias = 1.25
        pad.compile(scheme, level=int(level))
        halo = pad.apply(x_ct).decrypt().decode().detach().cpu().to(dtype=torch.float32)
        expected = torch.cat([x[:, :, :1, :], x, x[:, :, -1:, :]], dim=2) * 0.5 + 1.25

        assert tuple(int(value) for value in halo.shape) == (1, 1, 4, 3)
        assert float((halo - expected).abs().max().item()) <= 1.0e-4
    finally:
        scheme.delete_scheme()


def test_layout_policy_relayout_kernel_maps_packed_gap_halo_rows_from_neighbors() -> None:
    edge_row = {
        "edge": "input->conv",
        "source": "input",
        "target": "conv",
        "shape": [1, 4, 3, 2],
        "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 2},
    }
    kernel = LayoutPolicyRelayoutKernel(
        edge_row=edge_row,
        node="conv",
        direction="compact_to_halo",
        index=0,
    )
    compact_shape = (1, 1, 6, 4)
    halo_shape = (1, 1, 10, 4)
    out_to_src = {int(out): int(src) for src, out in kernel._iter_mappings(32768)}

    def idx(n: int, c: int, h: int, w: int, shape: tuple[int, int, int, int]) -> int:
        return ((int(n) * int(shape[1]) + int(c)) * int(shape[2]) + int(h)) * int(shape[3]) + int(w)

    for physical_row in range(2):
        for w in range(4):
            assert out_to_src[idx(0, 0, physical_row, w, halo_shape)] == idx(0, 0, physical_row, w, compact_shape)
            assert out_to_src[idx(0, 0, 2 + physical_row, w, halo_shape)] == idx(0, 0, physical_row, w, compact_shape)
            assert out_to_src[idx(0, 0, 8 + physical_row, w, halo_shape)] == idx(0, 0, 4 + physical_row, w, compact_shape)


def test_layout_policy_relayout_kernel_counts_one_rotation_and_mask_per_halo_side() -> None:
    edge_row = {
        "edge": "input->conv",
        "source": "input",
        "target": "conv",
        "shape": [1, 1, 4, 4],
        "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
    }
    pad = LayoutPolicyRelayoutKernel(edge_row=edge_row, node="conv", direction="compact_to_halo", index=0)
    estimate = pad.operation_estimate()

    assert estimate["kind"] == "height_stripe_halo_fill"
    assert estimate["rotation_count"] == 2
    assert estimate["mask_mult_count"] == 2
    assert estimate["sparse_lt_count"] == 0


def test_layout_policy_add_runtime_materializes_common_halo_join() -> None:
    _init_python_scheme("")
    try:
        compact = {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1}
        halo = {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1}
        rows = [
            {
                "edge": "left->add",
                "source": "left",
                "target": "add",
                "op_kind": "add",
                "shape": [1, 1, 2, 3],
                "source_layout": compact,
                "selected_layout": halo,
                "target_layout": halo,
                "relayout": True,
                "relayout_reason": "dp_add_input_alignment",
                "relayout_rotation_estimate": 2,
                "relayout_mask_mult_estimate": 2,
                "relayout_depth_estimate": 1,
            },
            {
                "edge": "right->add",
                "source": "right",
                "target": "add",
                "op_kind": "add",
                "shape": [1, 1, 2, 3],
                "source_layout": halo,
                "selected_layout": halo,
                "target_layout": halo,
                "relayout": False,
            },
        ]
        runtime = LayoutPolicyAddRuntimeExecutor(
            node="add",
            compile_plan={"policy": "dp", "edge_layouts": rows},
            input_sources=("left", "right"),
        )
        level = len(scheme.params.get_logq()) - 1
        left = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        left_halo = torch.cat([left[:, :, :1, :], left, left[:, :, -1:, :]], dim=2)
        right_core = torch.full((1, 1, 2, 3), 10.0, dtype=torch.float32)
        right_halo = torch.cat([right_core[:, :, :1, :], right_core, right_core[:, :, -1:, :]], dim=2)
        out = runtime(
            scheme.encrypt(scheme.encode(left, level)),
            scheme.encrypt(scheme.encode(right_halo, level)),
        )
        decoded = out.decrypt().decode().detach().cpu().to(dtype=torch.float32)

        assert tuple(int(value) for value in decoded.shape) == (1, 1, 4, 3)
        assert torch.equal(decoded, left_halo + right_halo)
        assert runtime.execute_count == 1
        assert runtime.last_runtime_io["runtime_lowering"] == "layout_policy_add_join"
        assert runtime.last_runtime_io["relayout_kernel_count"] == 1
        assert runtime.last_runtime_io["relayout_rotation_count"] == 2
        assert runtime.last_runtime_io["relayout_mask_mult_count"] == 2
    finally:
        scheme.delete_scheme()


def test_layout_policy_add_runtime_distributes_bootstrap_affine_over_relayout_inputs() -> None:
    _init_python_scheme("")
    try:
        compact = {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1}
        halo = {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1}
        rows = [
            {
                "edge": f"{source}->add",
                "source": source,
                "target": "add",
                "op_kind": "add",
                "shape": [1, 1, 2, 3],
                "source_layout": compact,
                "selected_layout": halo,
                "target_layout": halo,
                "relayout": True,
            }
            for source in ("left", "right")
        ]
        runtime = LayoutPolicyAddRuntimeExecutor(
            node="add",
            compile_plan={"policy": "dp", "edge_layouts": rows},
            input_sources=("left", "right"),
        )
        runtime._bootstrap_prescale_fusion = {"scale": 0.5, "bias": 1.25}
        level = len(scheme.params.get_logq()) - 1
        left = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        right = torch.full((1, 1, 2, 3), 10.0, dtype=torch.float32)
        left_halo = torch.cat([left[:, :, :1, :], left, left[:, :, -1:, :]], dim=2)
        right_halo = torch.cat([right[:, :, :1, :], right, right[:, :, -1:, :]], dim=2)

        out = runtime(
            scheme.encrypt(scheme.encode(left, level)),
            scheme.encrypt(scheme.encode(right, level)),
        )
        decoded = out.decrypt().decode().detach().cpu().to(dtype=torch.float32)
        expected = (left_halo + right_halo) * 0.5 + 1.25

        assert runtime.bootstrap_prescale_fusion_capable() is True
        assert runtime.last_runtime_io["bootstrap_prescale_fused"] is True
        assert float((decoded - expected).abs().max().item()) <= 1.0e-4
    finally:
        scheme.delete_scheme()


def test_layout_policy_provider_wrapper_fuses_bootstrap_affine_into_base_attrs() -> None:
    module = SimpleNamespace(
        on_weight=torch.ones((2, 3, 3, 3), dtype=torch.float32),
        on_bias=torch.tensor([1.0, 2.0], dtype=torch.float32),
    )
    base_executor = SimpleNamespace(module=module)
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=base_executor,
        output_node_id="conv",
        compile_plan={"policy": "dp", "edge_layouts": []},
    )
    executor._bootstrap_prescale_fusion = {"scale": 0.25, "bias": 1.5}

    def check_attrs():
        assert torch.equal(module.on_weight, torch.full((2, 3, 3, 3), 0.25, dtype=torch.float32))
        assert torch.equal(module.on_bias, torch.tensor([1.75, 2.0], dtype=torch.float32))
        return "ok"

    assert executor._with_base_module_attrs(check_attrs) == "ok"
    assert torch.equal(module.on_weight, torch.ones((2, 3, 3, 3), dtype=torch.float32))
    assert torch.equal(module.on_bias, torch.tensor([1.0, 2.0], dtype=torch.float32))


def test_layout_policy_provider_runtime_shape_reflects_output_relayout() -> None:
    module = SimpleNamespace(fhe_output_shape=torch.Size([1, 1, 4, 4]))
    base_executor = SimpleNamespace(module=module)
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=base_executor,
        output_node_id="conv",
        compile_plan={
            "policy": "dp",
            "edge_layouts": [],
            "node_layouts": [
                {
                    "node": "conv",
                    "shape": [1, 1, 4, 4],
                    "fhe_shape": [1, 1, 4, 4],
                    "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                    "output_relayout": True,
                }
            ],
        },
    )

    assert tuple(int(value) for value in executor.runtime_fhe_output_shape()) == (1, 1, 6, 4)


def test_layout_policy_provider_runtime_shape_reflects_fused_output_halo_without_input_halo() -> None:
    module = SimpleNamespace(fhe_output_shape=torch.Size([1, 1, 4, 4]))
    base_executor = SimpleNamespace(module=module, native_halo_output_capable=True)
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=base_executor,
        output_node_id="pool",
        compile_plan={
            "policy": "dp",
            "edge_layouts": [],
            "node_layouts": [
                {
                    "node": "pool",
                    "shape": [1, 1, 4, 4],
                    "fhe_shape": [1, 1, 4, 4],
                    "selected_layout": {"top_beta": 0, "bottom_beta": 1, "stride": 1, "gap": 1, "tile_count": 1},
                    "output_relayout": False,
                    "producer_materialized_halo": True,
                }
            ],
        },
    )

    assert tuple(int(value) for value in executor.runtime_fhe_output_shape()) == (1, 1, 5, 4)
    assert executor._runtime_lowering_label() == "provider_executable+native_halo_output_layout"


def test_bootstrap_layout_compression_rewrites_profitable_layout_policy_boundary() -> None:
    conv_module = SimpleNamespace(fhe_output_shape=torch.Size([1, 1, 4, 4]))
    act_module = type("SiLU", (), {})()
    next_module = SimpleNamespace(fhe_output_shape=torch.Size([1, 1, 4, 4]))

    compact = {
        "top_beta": 0,
        "bottom_beta": 0,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 16,
        "tile_count": 1,
    }
    halo = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 24,
        "tile_count": 2,
    }
    compile_plan = {
        "policy": "dp",
        "summary": {},
        "edge_layouts": [
            {
                "edge": "conv->act",
                "source": "conv",
                "target": "act",
                "op_kind": "silu",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(halo),
                "selected_layout": dict(halo),
                "target_layout": dict(halo),
                "compact_layout": dict(compact),
                "physical_layout": "logical_halo_compact",
                "layout_mode": "halo_local",
                "relayout": False,
            },
            {
                "edge": "act->next",
                "source": "act",
                "target": "next",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(halo),
                "selected_layout": dict(halo),
                "target_layout": dict(halo),
                "compact_layout": dict(compact),
                "physical_layout": "logical_halo_compact",
                "layout_mode": "compact_halo_shared",
                "relayout": False,
            },
        ],
        "node_layouts": [
            {
                "node": "conv",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 6, 4],
                "selected_layout": dict(halo),
                "compact_layout": dict(compact),
                "physical_layout": "logical_halo_compact",
            },
            {
                "node": "act",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 6, 4],
                "selected_layout": dict(halo),
                "compact_layout": dict(compact),
                "physical_layout": "logical_halo_compact",
            },
            {
                "node": "next",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            },
        ],
    }
    conv_executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=conv_module),
        output_node_id="conv",
        compile_plan=compile_plan,
    )
    next_executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=next_module),
        output_node_id="next",
        compile_plan=compile_plan,
    )
    conv_module.region_runtime = SimpleNamespace(executor=conv_executor)
    next_module.region_runtime = SimpleNamespace(executor=next_executor)

    dag = nx.DiGraph()
    dag.add_node("conv", module=conv_module, bootstrap=True)
    dag.add_node("act", module=act_module, bootstrap=False)
    dag.add_node("next", module=next_module, bootstrap=False)
    dag.add_edge("conv", "act")
    dag.add_edge("act", "next")

    audit = apply_bootstrap_layout_compression(dag)
    updated = next_executor.compile_plan
    nodes = {str(row["node"]): dict(row) for row in updated["node_layouts"]}
    edges = {str(row["edge"]): dict(row) for row in updated["edge_layouts"]}

    assert audit["enabled"] is True
    assert audit["nodes"][0]["saved_ciphertexts_per_bootstrap"] == 1
    assert nodes["conv"]["selected_layout"]["tile_count"] == 1
    assert nodes["act"]["selected_layout"]["tile_count"] == 1
    assert edges["conv->act"]["physical_layout"] == "packed_compact"
    assert edges["conv->act"]["source_physical_layout"] == "packed_compact"
    assert edges["conv->act"]["target_physical_layout"] == "packed_compact"
    assert bool(edges["conv->act"].get("consumer_fused_relayout", False)) is False
    assert edges["act->next"]["physical_layout"] == "packed_compact"
    assert edges["act->next"]["source_physical_layout"] == "packed_compact"
    assert edges["act->next"]["target_physical_layout"] == "packed_compact"
    assert bool(edges["act->next"].get("consumer_fused_relayout", False)) is False
    assert edges["act->next"]["selected_layout"]["tile_count"] == 1
    assert next_executor.native_input_rows == ()
    assert len(next_executor.compact_source_rows) == 1
    assert tuple(int(value) for value in conv_module.fhe_output_shape) == (1, 1, 4, 4)


def test_input_pair_pool_provider_fuses_output_beta_relayout() -> None:
    _init_python_scheme("")
    try:
        pool = AvgPool2d(kernel_size=2, stride=2, padding=0)
        pool.init_orion_params()
        pool.input_shape = torch.Size((1, 1, 4, 4))
        pool.output_shape = torch.Size((1, 1, 2, 2))
        pool.fhe_input_shape = torch.Size((1, 1, 4, 4))
        pool.fhe_output_shape = torch.Size((1, 1, 3, 2))
        pool.input_gap = 1
        pool.output_gap = 1
        pool.layout_policy_output_layout = {"top_beta": 0, "bottom_beta": 1, "gap": 1}
        pool.layout_policy_output_materialization = "fused_relayout"
        pool.update_params()
        pool.set_level(len(scheme.params.get_logq()) - 1)

        executor = InputPairConvRuntimeExecutor(
            module=pool,
            output_node_id="pool_fused_beta",
            use_ct_pt_hybrid_packing=False,
        )
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        out = executor(scheme.encrypt(scheme.encode(x, pool.level)))["pool_fused_beta"]
        decoded = out.decrypt().decode().detach().cpu()
        if torch.is_complex(decoded):
            decoded = decoded.real
        observed = decoded.to(dtype=torch.float32).flatten()[:6].reshape(1, 1, 3, 2)
        core = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        expected = torch.cat([core, core[:, :, -1:, :]], dim=2)

        assert tuple(int(value) for value in out.on_shape) == (1, 1, 3, 2)
        assert float((observed - expected).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()


def test_layout_policy_dp_costs_explicit_beta_growth_paths() -> None:
    dag = build_u22_dag(network_spec("u22_256_base32"))
    compile_plan = build_layout_policy_compile_plan(dag, policy="dp")

    producer_halo_nodes = [
        row
        for row in compile_plan["node_layouts"]
        if bool(row.get("producer_materialized_halo", False))
    ]
    compact_align_shared = [
        row
        for row in compile_plan["edge_layouts"]
        if row.get("layout_mode") == "compact_align_shared"
    ]
    compact_halo_shared = [
        row
        for row in compile_plan["edge_layouts"]
        if row.get("layout_mode") == "compact_halo_shared"
    ]
    native_halo_stripe = [
        row
        for row in compile_plan["edge_layouts"]
        if row.get("physical_layout") == "native_source_stripe"
    ]
    producer_fused_edges = [
        row
        for row in compile_plan["edge_layouts"]
        if bool(row.get("producer_fused_relayout", False))
    ]
    node_rows = {str(row["node"]): row for row in compile_plan["node_layouts"]}
    edge_rows = {str(row["edge"]): row for row in compile_plan["edge_layouts"]}
    assert int(compile_plan["relayout_edge_count"]) == 0
    assert int(compile_plan["output_relayout_node_count"]) == 0
    assert producer_halo_nodes
    assert producer_fused_edges
    for pool_node in ("pool1", "pool2", "pool3", "pool4"):
        pool_layout = dict(node_rows[pool_node]["selected_layout"])
        assert bool(node_rows[pool_node].get("producer_materialized_halo", False)) is True
        assert node_rows[pool_node]["producer_materialized_halo_reason"] == "dp_producer_materialized_halo"
        assert node_rows[pool_node]["physical_layout"] == "logical_halo_compact"
        assert int(pool_layout["top_beta"]) == 1
        assert int(pool_layout["bottom_beta"]) == 1
    for pool_edge in ("pool1->enc2a", "pool2->enc3a", "pool3->enc4a", "pool4->bottlenecka"):
        row = edge_rows[pool_edge]
        assert row["layout_mode"] == "compact_halo_shared"
        assert bool(row["relayout"]) is False
        assert bool(row["consumer_fused_relayout"]) is True
        assert dict(row["source_layout"]) == dict(row["selected_layout"])
    assert int(compile_plan["summary"]["producer_fused_materialization_count"]) == len(producer_halo_nodes)
    assert int(compile_plan["summary"]["producer_fused_rotation_estimate"]) >= 0
    assert compact_align_shared or compact_halo_shared
    assert int(compile_plan["summary"]["consumer_fused_relayout_count"]) == (
        len(compact_align_shared) + len(compact_halo_shared)
    )
    assert int(compile_plan["summary"]["consumer_fused_rotation_estimate"]) == sum(
        int(row.get("consumer_fused_rotation_estimate", 0) or 0)
        for row in [*compact_align_shared, *compact_halo_shared]
    )
    assert all(not bool(row["relayout"]) for row in [*compact_align_shared, *compact_halo_shared])
    assert all(not bool(row["relayout"]) for row in producer_fused_edges)
    assert "x->enc1a" in {row["edge"] for row in native_halo_stripe}
    assert int(compile_plan["summary"]["compact_fallback_penalty_estimate"]) == 0
    relayout_edge_depth = sum(int(row["depth_estimate"]) for row in compile_plan["relayout_edges"])
    assert int(compile_plan["summary"]["relayout_depth_estimate"]) == int(relayout_edge_depth) + sum(
        int(row["depth_estimate"]) for row in compile_plan["output_relayout_nodes"]
    )
    assert all(int(row["rotation_estimate"]) >= 0 for row in compile_plan["relayout_edges"])
    assert all(int(row["mask_mult_estimate"]) >= 0 for row in compile_plan["relayout_edges"])
    assert all("lt_one_channel_diagonal_estimate" not in row for row in compile_plan["edge_layouts"])
    assert all("lt_bsgs_group_count_estimate" in row for row in compile_plan["edge_layouts"])
    assert int(compile_plan["summary"]["relayout_rotation_estimate"]) == sum(
        int(row["rotation_estimate"]) for row in compile_plan["relayout_edges"]
    ) + sum(int(row["rotation_estimate"]) for row in compile_plan["output_relayout_nodes"])
    assert int(compile_plan["summary"]["relayout_mask_mult_estimate"]) == sum(
        int(row["mask_mult_estimate"]) for row in compile_plan["relayout_edges"]
    ) + sum(int(row["mask_mult_estimate"]) for row in compile_plan["output_relayout_nodes"])


def test_layout_policy_dp_treats_silu_as_layout_preserving() -> None:
    torch.manual_seed(0)
    model = UNet22(dataset="kvasir_polyp_256", base_channels=8, activation="silu", silu_degree=7)
    traced = OrionTracer().trace_model(model)
    StatsTracker(traced).propagate(torch.randn((1, 3, 256, 256), dtype=torch.float32))
    dag = NetworkDAG(traced)
    dag.build_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()

    compile_plan = build_layout_policy_compile_plan(dag, policy="dp")
    activation_rows = [row for row in compile_plan["node_layouts"] if str(row["node"]).endswith("_act")]
    node_rows = {str(row["node"]): row for row in compile_plan["node_layouts"]}
    edge_rows = {str(row["edge"]): row for row in compile_plan["edge_layouts"]}

    assert activation_rows
    assert int(compile_plan["output_relayout_node_count"]) == 0
    assert all(not bool(row.get("output_relayout", False)) for row in activation_rows)
    assert int(node_rows["enc1a"]["selected_layout"]["top_beta"]) == 1
    assert int(node_rows["enc1a"]["selected_layout"]["bottom_beta"]) == 1
    assert int(node_rows["enc1a_act"]["selected_layout"]["top_beta"]) == 1
    assert int(node_rows["enc1a_act"]["selected_layout"]["bottom_beta"]) == 1
    assert bool(node_rows["enc1a"].get("producer_materialized_halo", False)) is True
    enc1a_to_act = edge_rows["enc1a->enc1a_act"]
    enc1a_act_to_enc1b = edge_rows["enc1a_act->enc1b"]
    assert enc1a_to_act["future_layouts"]
    assert int(enc1a_to_act["required_layout"]["top_beta"]) == 0
    assert int(enc1a_to_act["selected_layout"]["top_beta"]) == 1
    assert enc1a_act_to_enc1b["layout_mode"] == "compact_halo_shared"
    assert bool(enc1a_act_to_enc1b["relayout"]) is False
    assert dict(enc1a_act_to_enc1b["source_layout"]) == dict(enc1a_act_to_enc1b["selected_layout"])


def test_layout_policy_dp_can_avoid_tconv_add_relayout_with_shared_conv_fallback() -> None:
    dag = build_u22_dag(network_spec("u22_256_base32"))
    compile_plan = build_layout_policy_compile_plan(dag, policy="dp")
    node_rows = {str(row["node"]): row for row in compile_plan["node_layouts"]}
    edge_rows = {str(row["edge"]): row for row in compile_plan["edge_layouts"]}

    for up_node, join_node, join_edge, consumer_edge in (
        ("up3", "add3", "up3->add3", "add3->dec3a"),
        ("up2", "add2", "up2->add2", "add2->dec2a"),
        ("up1", "add1", "up1->add1", "add1->dec1a"),
    ):
        up_layout = dict(node_rows[up_node]["selected_layout"])
        join_layout = dict(node_rows[join_node]["selected_layout"])
        up_join = dict(edge_rows[join_edge])
        join_dec = dict(edge_rows[consumer_edge])

        assert up_join["op_kind"] == "add"
        assert bool(up_join["relayout"]) is False
        assert int(up_layout["top_beta"]) == int(join_layout["top_beta"])
        assert int(up_layout["bottom_beta"]) == int(join_layout["bottom_beta"])
        assert int(up_layout["stride"]) == int(join_layout["stride"])
        assert int(up_layout["gap"]) == int(join_layout["gap"])
        assert int(join_layout["core_slots"]) == int(edge_rows[consumer_edge]["shape"][1]) * int(
            edge_rows[consumer_edge]["shape"][2]
        ) * int(edge_rows[consumer_edge]["shape"][3])
        assert dict(join_dec["source_layout"]) == join_layout
        assert join_dec["layout_mode"] == "compact_halo_shared"
        assert join_dec["physical_layout"] == "logical_halo_compact"
        assert bool(join_dec["relayout"]) is False
        assert bool(join_dec["consumer_fused_relayout"]) is True


def test_one_down_one_up_silu_dp_fuses_relayout_without_bootstrap_growth() -> None:
    _init_long_python_scheme("")
    try:
        greedy_dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)
        dp_dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)
        greedy_plan = build_layout_policy_compile_plan(greedy_dag, policy="greedy")
        dp_plan = build_layout_policy_compile_plan(dp_dag, policy="dp")

        assert int(dp_plan["summary"]["relayout_depth_estimate"]) == 0
        assert int(dp_plan["summary"]["relayouts"]) == 0
        assert int(dp_plan["summary"]["consumer_fused_relayout_count"]) > 0
        assert int(dp_plan["summary"]["total_ciphertext_tiles"]) < int(
            greedy_plan["summary"]["total_ciphertext_tiles"]
        )

        fused_rows = [
            row
            for row in dp_plan["edge_layouts"]
            if bool(row.get("consumer_fused_relayout", False))
        ]
        assert fused_rows
        assert all(not bool(row.get("relayout", False)) for row in fused_rows)
        assert sum(int(row.get("consumer_fused_rotation_estimate", 0) or 0) for row in fused_rows) == int(
            dp_plan["summary"]["consumer_fused_rotation_estimate"]
        )

        boot_counts: dict[str, int] = {}
        for policy, dag in (("greedy", greedy_dag), ("dp", dp_dag)):
            registry = U22CompileRegistry.for_dag(
                dag,
                allowed_nodes=None,
                enable_conv_kernels=True,
                layout_policy=str(policy),
            )
            audit = registry.attach_to_dag(dag)
            dag.find_residuals()
            _input_level, bootstraps, _slots = BootstrapSolver(
                SimpleNamespace(),
                dag,
                l_eff=int(len(scheme.params.get_logq()) - 1),
            ).solve()
            boot_counts[str(policy)] = int(bootstraps)
            if str(policy) == "dp":
                assert int(audit["graph_audit"]["layout_policy_summary"]["relayout_depth_estimate"]) == 0
                mid_executor = dag.nodes["mid"]["module"].region_runtime.executor
                assert isinstance(mid_executor, LayoutPolicyProviderRuntimeExecutor)
                assert mid_executor.compact_align_shared_rows
                assert not mid_executor.relayout_rows

        assert boot_counts["dp"] <= boot_counts["greedy"]
    finally:
        scheme.delete_scheme()


def test_one_down_one_up_fused_non_dp_policies_remove_adjacent_relayout_depth() -> None:
    fixed_dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)
    fixed_fused_dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)
    eager_dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)
    eager_fused_dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)
    greedy_dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)
    greedy_fused_dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)

    fixed = build_layout_policy_compile_plan(fixed_dag, policy="fixed_max")
    fixed_fused = build_layout_policy_compile_plan(fixed_fused_dag, policy="fixed_max_fused")
    eager = build_layout_policy_compile_plan(eager_dag, policy="eager")
    eager_fused = build_layout_policy_compile_plan(eager_fused_dag, policy="eager_fused")
    greedy = build_layout_policy_compile_plan(greedy_dag, policy="greedy")
    greedy_fused = build_layout_policy_compile_plan(greedy_fused_dag, policy="greedy_fused")

    assert int(fixed["summary"]["relayout_depth_estimate"]) > 0
    assert int(eager["summary"]["relayout_depth_estimate"]) > 0
    assert int(greedy["summary"]["relayout_depth_estimate"]) > 0
    assert int(fixed_fused["summary"]["relayout_depth_estimate"]) <= int(fixed["summary"]["relayout_depth_estimate"])
    assert int(eager_fused["summary"]["relayout_depth_estimate"]) == 0
    assert int(greedy_fused["summary"]["relayout_depth_estimate"]) == 0
    assert int(fixed_fused["summary"]["relayouts"]) <= int(fixed["summary"]["relayouts"])
    assert int(eager_fused["summary"]["relayouts"]) == 0
    assert int(greedy_fused["summary"]["relayouts"]) == 0
    assert int(eager_fused["summary"]["consumer_fused_relayout_count"]) > 0
    assert int(fixed_fused["summary"]["producer_fused_materialization_count"]) > 0


def test_orion_dense_policy_simulates_no_halo_compact_baseline() -> None:
    dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)
    plan = build_layout_policy_compile_plan(dag, policy="orion_dense")

    assert int(plan["summary"]["relayouts"]) == 0
    assert int(plan["summary"]["relayout_depth_estimate"]) == 0
    assert int(plan["summary"]["consumer_fused_relayout_count"]) == 0
    assert int(plan["summary"]["producer_fused_materialization_count"]) == 0
    assert all(
        int(row["selected_layout"]["top_beta"]) == 0
        and int(row["selected_layout"]["bottom_beta"]) == 0
        and row["physical_layout"] == "packed_compact"
        and row["layout_mode"] == "orion_dense"
        for row in plan["edge_layouts"]
    )
    assert all(
        int(row["selected_layout"]["top_beta"]) == 0
        and int(row["selected_layout"]["bottom_beta"]) == 0
        and row["physical_layout"] == "packed_compact"
        for row in plan["node_layouts"]
    )


def test_layout_policy_provider_keeps_native_halo_for_compact_halo_local_conv() -> None:
    _init_long_python_scheme("")
    try:
        dag = _prepared_one_down_one_up_dag(image_size=128, base_channels=8)
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=None,
            enable_conv_kernels=True,
            layout_policy="dp",
        )
        registry.attach_to_dag(dag)

        executor = dag.nodes["enc"]["module"].region_runtime.executor
        assert isinstance(executor, LayoutPolicyProviderRuntimeExecutor)
        assert not executor.native_input_rows
        assert not executor.relayout_rows
        assert type(executor.base_executor).__name__ == "HaloLocalConvRuntimeExecutor"
        assert type(executor.base_executor.delegate).__name__ == "NativeHaloStripeNoRIConvExecutor"
        assert executor.compact_source_rows
        assert executor._runtime_lowering_label() == "provider_executable+compact_layout"
    finally:
        scheme.delete_scheme()


def test_layout_policy_dp_plans_reduced_full_structure_r34_mock() -> None:
    torch.manual_seed(0)
    model = ResNet(
        "cifar10",
        BasicBlock,
        [3, 4, 6, 3],
        [4, 8, 16, 32],
        {"kernel_size": 3, "stride": 1, "padding": 1},
        10,
    )
    traced = OrionTracer().trace_model(model)
    StatsTracker(traced).propagate(torch.randn((1, 3, 32, 32), dtype=torch.float32))
    dag = NetworkDAG(traced)
    dag.build_dag()
    for node in dag.nodes:
        module = dag.nodes[node]["module"]
        if module is not None and hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if module is not None and hasattr(module, "update_params"):
            module.update_params()

    compile_plan = build_layout_policy_compile_plan(dag, policy="dp", slots=4096)

    conv_edges = [row for row in compile_plan["edge_layouts"] if row["op_kind"] == "conv2d"]
    add_edges = [row for row in compile_plan["edge_layouts"] if row["target"].endswith("add")]
    assert compile_plan["status"] == "ok"
    assert len(conv_edges) >= 36
    assert compile_plan["edge_layout_count"] == len(compile_plan["edge_layouts"])
    assert int(compile_plan["summary"]["relayout_mask_mult_estimate"]) >= 0
    assert add_edges


def test_input_pair_no_hybrid_compiles_singleton_source_blocks(monkeypatch) -> None:
    _init_python_scheme("")
    try:
        conv = Conv2d(1, 1, kernel_size=1, bias=True)
        conv.init_orion_params()
        conv.input_shape = torch.Size((1, 1, 1, 2))
        conv.output_shape = torch.Size((1, 1, 1, 1))
        conv.fhe_input_shape = torch.Size((1, 1, 1, 2))
        conv.fhe_output_shape = torch.Size((1, 1, 1, 1))
        conv.input_gap = 1
        conv.output_gap = 1
        conv.name = "input_pair_no_hybrid_probe"
        conv.set_level(len(scheme.params.get_logq()) - 1)
        conv.set_depth(1)

        def fake_pack_conv2d(_module, last=False):
            slots = int(scheme.params.get_slots())
            return ({(0, 0): {0: [1.0] * slots}, (0, 1): {0: [2.0] * slots}}, 0)

        def fake_bias(_module):
            return torch.zeros((int(scheme.params.get_slots()),), dtype=torch.float32)

        monkeypatch.setattr(packing, "pack_conv2d", fake_pack_conv2d)
        monkeypatch.setattr(packing, "construct_conv2d_bias", fake_bias)

        executor = InputPairConvRuntimeExecutor(
            module=conv,
            output_node_id="probe",
            use_ct_pt_hybrid_packing=False,
        )
        executor.compile(scheme)

        assert executor.input_block_pairs == [(0, None), (1, None)]
        assert executor.pair_is_complex == [False, False]
        assert executor.hybrid_pair_count == 0
        assert executor.hybrid_pair_layout_strategy == "hybrid_disabled"
    finally:
        scheme.delete_scheme()


def test_layout_policy_provider_wrapper_runs_input_pair_provider_after_relayout() -> None:
    torch.manual_seed(0)
    _init_python_scheme("")
    try:
        conv = Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        conv.weight.data.fill_(0.125)
        conv.bias.data.fill_(0.25)
        conv.init_orion_params()
        conv.input_shape = torch.Size((1, 1, 4, 4))
        conv.output_shape = torch.Size((1, 1, 4, 4))
        conv.fhe_input_shape = torch.Size((1, 1, 4, 4))
        conv.fhe_output_shape = torch.Size((1, 1, 4, 4))
        conv.input_gap = 1
        conv.output_gap = 1
        conv.name = "layout_policy_provider_conv"
        conv.set_level(len(scheme.params.get_logq()) - 1)
        conv.set_depth(2)
        base = InputPairConvRuntimeExecutor(module=conv, output_node_id="conv")
        executor = LayoutPolicyProviderRuntimeExecutor(
            base_executor=base,
            output_node_id="conv",
            compile_plan={
                "policy": "eager",
                "edge_layouts": [
                    {
                        "edge": "input->conv",
                        "source": "input",
                        "target": "conv",
                        "shape": [1, 1, 4, 4],
                        "relayout": True,
                        "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1},
                    },
                ],
                "node_layouts": [
                    {
                        "node": "conv",
                        "shape": [1, 1, 4, 4],
                        "fhe_shape": [1, 1, 4, 4],
                        "selected_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1},
                    },
                ],
            },
        )
        executor.assigned_level = int(conv.level)
        executor.assigned_depth = 3
        x = torch.randn((1, 1, 4, 4), dtype=torch.float32)
        x_height_halo = torch.cat([torch.zeros_like(x[:, :, :1, :]), x, torch.zeros_like(x[:, :, -1:, :])], dim=2)
        reference = torch.nn.functional.conv2d(x_height_halo, conv.weight, conv.bias, padding=(1, 1))
        x_ct = scheme.encrypt(scheme.encode(x, conv.level))
        out_ct = executor(x_ct)["conv"]
        decoded = out_ct.decrypt().decode().detach().cpu().to(dtype=torch.float32)
        assert tuple(int(value) for value in decoded.shape) == tuple(int(value) for value in reference.shape)
        assert float((decoded - reference).abs().max().item()) <= 1.0e-4
        assert executor.compile_count == 1
        assert executor.execute_count == 1
        assert base.compile_count == 1
        assert executor.native_halo_input is False
        assert base.assigned_depth == 1
        assert executor.last_runtime_io["runtime_lowering"] == "provider_executable+native_halo_output_layout"
        assert executor.last_runtime_io["provider_executor"] == "InputPairConvRuntimeExecutor"
        assert executor.last_runtime_io["native_halo_provider"] is False
        assert executor.last_runtime_io["relayout_kernel_count"] == 2
        assert tuple(int(value) for value in out_ct.on_shape) == (1, 1, 6, 4)
        assert tuple(int(value) for value in executor.runtime_fhe_output_shape()) == (1, 1, 6, 4)
        assert not hasattr(conv, "layout_policy_input_row_offset")
        assert not hasattr(conv, "layout_policy_output_row_offset")
        assert tuple(int(value) for value in base.fhe_output_shape) == (1, 1, 4, 4)
        metadata = executor.compile_cache_metadata()["layout_policy_wrapper"]
        assert metadata["runtime_lowering"] == "provider_executable+native_halo_output_layout"
        assert metadata["relayout_kernel_count"] == 2
        assert metadata["native_halo_provider"] is False
        group = RegionFirstRuntimeGroup(
            region_id="layout_policy_provider_conv",
            network="U22",
            stage="layout_policy_provider_conv",
            module_prefix="conv",
            conv_nodes=("conv",),
            strategy="layout_policy_provider",
            materializer="u22_input_pair_conv_shared_rotations",
            depth=3,
            solver_depth=3,
            boundary_actions=("layout_policy_compile_plan",),
            expected_stats={},
            executable=True,
            fallback_reason="",
            output_node_ids=("conv",),
            executor=executor,
        )
        pressure = collect_layout_policy_provider_pressure(
            type("Registry", (), {"groups": (group,)})(),
            backend=None,
            slots=int(scheme.params.get_slots()),
        )
        summary = pressure["summary"]
        assert summary["provider_region_count"] == 1
        assert summary["native_halo_provider_region_count"] == 0
        assert summary["relayout_kernel_count"] == 2
        assert summary["provider_input_block_cols"] >= summary["compact_input_block_cols"]
        assert pressure["regions"][0]["runtime_lowering"] == "provider_executable+native_halo_output_layout"
    finally:
        scheme.delete_scheme()


def test_layout_policy_provider_wrapper_runs_consumer_fused_compact_align_shared_without_relayout() -> None:
    torch.manual_seed(0)
    _init_python_scheme("")
    try:
        conv = Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        conv.weight.data.fill_(0.125)
        conv.bias.data.fill_(0.25)
        conv.init_orion_params()
        conv.input_shape = torch.Size((1, 1, 4, 4))
        conv.output_shape = torch.Size((1, 1, 4, 4))
        conv.fhe_input_shape = torch.Size((1, 1, 4, 4))
        conv.fhe_output_shape = torch.Size((1, 1, 4, 4))
        conv.input_gap = 1
        conv.output_gap = 1
        conv.name = "layout_policy_provider_consumer_fused_conv"
        conv.set_level(len(scheme.params.get_logq()) - 1)
        conv.set_depth(1)
        base = InputPairConvRuntimeExecutor(
            module=conv,
            output_node_id="conv",
            use_ct_pt_hybrid_packing=False,
        )
        executor = LayoutPolicyProviderRuntimeExecutor(
            base_executor=base,
            output_node_id="conv",
            compile_plan={
                "policy": "dp",
                "edge_layouts": [
                    {
                        "edge": "input->conv",
                        "source": "input",
                        "target": "conv",
                        "op_kind": "conv2d",
                        "shape": [1, 1, 4, 4],
                        "fhe_shape": [1, 1, 4, 4],
                        "required_layout": {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1},
                        "source_layout": {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
                        "selected_layout": {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "tile_count": 1},
                        "layout_mode": "compact_align_shared",
                        "physical_layout": "packed_compact",
                        "relayout": False,
                        "consumer_fused_relayout": True,
                        "consumer_fused_rotation_estimate": 0,
                    },
                ],
                "node_layouts": [
                    {
                        "node": "conv",
                        "shape": [1, 1, 4, 4],
                        "fhe_shape": [1, 1, 4, 4],
                        "selected_layout": {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1},
                        "physical_layout": "packed_compact",
                        "output_relayout": False,
                    },
                ],
            },
        )
        executor.assigned_level = int(conv.level)
        executor.assigned_depth = int(conv.depth)
        x = torch.randn((1, 1, 4, 4), dtype=torch.float32)
        reference = torch.nn.functional.conv2d(x, conv.weight, conv.bias, padding=(1, 1))
        x_ct = scheme.encrypt(scheme.encode(x, conv.level))
        out_ct = executor(x_ct)["conv"]
        decoded = out_ct.decrypt().decode().detach().cpu().to(dtype=torch.float32)

        assert float((decoded - reference).abs().max().item()) <= 1.0e-4
        assert executor.compile_count == 1
        assert executor.execute_count == 1
        assert base.compile_count == 1
        assert executor.compact_align_shared_rows
        assert not executor.relayout_rows
        assert not executor.native_physical_relayout_rows
        assert executor.last_runtime_io["runtime_lowering"] == "provider_executable+compact_align_shared"
        assert executor.last_runtime_io["relayout_kernel_count"] == 0
        assert executor.last_runtime_io["compact_align_shared_edge_count"] == 1
        assert tuple(int(value) for value in out_ct.on_shape) == (1, 1, 4, 4)
        assert tuple(int(value) for value in executor.runtime_fhe_output_shape()) == (1, 1, 4, 4)
    finally:
        scheme.delete_scheme()


def test_runtime_anchor_timeout_is_reported_without_launching_real_e2e(tmp_path: Path) -> None:
    def fake_runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["fake"], timeout=1, output="still compiling")

    anchor = run_runtime_anchor(
        network="u22_64_base32",
        backend="lattigo",
        cache_root=tmp_path,
        compile_timeout_s=1,
        runner=fake_runner,
    )

    assert anchor["status"] == "compile_timeout"
    payload = attach_runtime_anchor(build_planner_ablation(network="u22_64_base32", policies=("dp",)), anchor)
    dp = _policy(payload, "dp")
    assert dp["metric_source"] == "planner_estimate+runtime_anchor"
    assert dp["runtime_status"] == "compile_timeout"


def test_lattigo_dp_only_anchor_leaves_non_dp_runtime_unrun() -> None:
    payload = build_planner_ablation(network="u22_64_base32", policies=("fixed_max", "dp"))
    payload["backend"] = "lattigo"
    updated = attach_runtime_anchor(payload, {"status": "skipped", "reason": "mock"})

    fixed = _policy(updated, "fixed_max")
    dp = _policy(updated, "dp")
    assert fixed["runtime_status"] == "lattigo_layout_policy_runtime_anchor_not_run"
    assert "run_backend_runtime_anchors" in fixed["runtime_reason"]
    assert dp["runtime_status"] == "skipped"


def test_lattigo_backend_runtime_anchors_launch_each_policy_provider_mode(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(cmd, **_kwargs):
        calls.append([str(value) for value in cmd])
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        provider_mode = str(cmd[cmd.index("--provider-mode") + 1])
        (out_dir / "montgomery_lung_64_samples0_1_provider_fhe_figure.json").write_text(
            json_payload := (
                "{\n"
                '  "status": "ok",\n'
                '  "timing_s": {"he_forward_total": 1.25},\n'
                '  "provider_pressure": {"summary": {"provider_region_count": 3, "native_halo_provider_region_count": 2, "diagonal_key_set_mismatch_count": 5, "group_union_rotation_count": 17}},\n'
                '  "samples": [\n'
                '    {"fhe_vs_pytorch_logits": {"mae": 0.0}, "fhe_vs_reference": {"dice": 1.0}}\n'
                "  ]\n"
                "}\n"
            ),
            encoding="utf-8",
        )
        assert json_payload
        return subprocess.CompletedProcess(cmd, 0, stdout=f"ok {provider_mode}")

    payload = build_planner_ablation(network="u22_64_base32", policies=("fixed_max", "eager", "greedy", "dp"))
    payload["backend"] = "lattigo"
    anchors = run_backend_runtime_anchors(
        payload,
        backend="lattigo",
        cache_root=tmp_path,
        compile_timeout_s=10,
        runner=fake_runner,
    )
    updated = attach_backend_runtime_anchors(payload, anchors)

    provider_modes = [call[call.index("--provider-mode") + 1] for call in calls]
    assert provider_modes == [
        "u22_64_base32_layout_fixedmax",
        "u22_64_base32_layout_eager",
        "u22_64_base32_layout_greedy",
        "u22_64_base32",
    ]
    assert all(_policy(updated, policy)["runtime_status"] == "ok" for policy in ("fixed_max", "eager", "greedy", "dp"))
    eager = _policy(updated, "eager")
    assert eager["provider_region_count"] == 3
    assert eager["native_halo_provider_region_count"] == 2
    assert eager["diagonal_key_set_mismatch_count"] == 5
    assert eager["group_union_rotation_count"] == 17


def test_runtime_anchor_extracts_mocked_e2e_metrics(tmp_path: Path) -> None:
    def fake_runner(cmd, **_kwargs):
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "montgomery_lung_64_samples0_1_provider_fhe_figure.json").write_text(
            """
{
  "status": "ok",
  "timing_s": {"he_forward_total": 12.5},
  "samples": [
    {
      "fhe_vs_pytorch_logits": {"mae": 0.125},
      "fhe_vs_reference": {"dice": 0.75}
    }
  ]
}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    anchor = run_runtime_anchor(
        network="u22_64_base32",
        backend="lattigo",
        cache_root=tmp_path,
        compile_timeout_s=10,
        runner=fake_runner,
    )

    assert anchor["status"] == "ok"
    assert anchor["he_forward_s"] == 12.5
    assert anchor["mae"] == 0.125
    assert anchor["dice"] == 0.75


def test_non_ckks_layout_simulation_attaches_clear_sanity_metrics() -> None:
    payload = build_planner_ablation(network="u22_64_base32", policies=("fixed_max", "eager", "greedy", "dp"))
    simulation = run_non_ckks_layout_simulation(payload, seed=0)
    updated = attach_non_ckks_simulation(payload, simulation)

    assert simulation["status"] == "ok"
    for row in updated["policies"]:
        assert row["metric_source"] == "planner_estimate+non_ckks_sim"
        assert row["runtime_status"] == "non_ckks_sim_ok"
        assert row["layout_alignment_ok"] is True
        assert row["mae"] == 0.0
        assert row["max_abs"] == 0.0
        assert row["dice"] == 1.0


def test_layout_policy_cli_planner_smoke(tmp_path: Path) -> None:
    out_csv = tmp_path / "layout_policy_ablation_u22_64.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_layout_policy_ablation.py",
            "--network",
            "u22_64_base32",
            "--policies",
            "fixed_max",
            "always",
            "greedy",
            "dp",
            "--mode",
            "planner",
            "--out",
            str(out_csv),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout
    assert out_csv.exists()
    assert out_csv.with_suffix(".json").exists()
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["policy"] for row in rows] == ["fixed_max", "always", "greedy", "dp"]
    assert all(row["metric_source"] == "planner_estimate" for row in rows)
    assert "diagonal_key_set_mismatch_count" in rows[0]
    assert "provider_input_block_cols" in rows[0]


def test_layout_policy_cli_non_ckks_simulation_smoke(tmp_path: Path) -> None:
    out_csv = tmp_path / "layout_policy_ablation_u22_64_sim.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_layout_policy_ablation.py",
            "--network",
            "u22_64_base32",
            "--mode",
            "planner,simulate",
            "--out",
            str(out_csv),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["metric_source"] == "planner_estimate+non_ckks_sim" for row in rows)
    assert all(row["runtime_status"] == "non_ckks_sim_ok" for row in rows)
    assert all(row["layout_alignment_ok"] == "True" for row in rows)
