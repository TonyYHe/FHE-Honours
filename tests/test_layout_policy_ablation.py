from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
import torch

from orion.core import packing
from orion.core.bootstrap_layout_compression import apply_bootstrap_layout_compression
from orion.core.bootstrap_layout_compression import apply_bootstrap_aware_layout_refinement
from orion.core.bootstrap_layout_compression import apply_bootstrap_aware_layout_refinement_candidate
from orion.core.bootstrap_layout_compression import enumerate_bootstrap_aware_layout_refinement_candidates
from orion.core.bootstrap_layout_compression import restore_layout_policy_compile_plan
from orion.core.orion import _region_first_mode_options
from orion.core.orion import scheme
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental.layout_policy_ablation import (
    attach_backend_runtime_anchors,
    attach_non_ckks_simulation,
    attach_runtime_anchor,
    build_edge_infos,
    build_planner_ablation,
    build_layout_policy_compile_plan,
    build_u22_dag,
    network_spec,
    normalize_policy,
    run_backend_runtime_anchors,
    run_non_ckks_layout_simulation,
    run_runtime_anchor,
    validate_layout_policy_compile_plan,
    _fill_beta_to_tile_capacity,
    _layout_for_shape,
    _layout_physical_bottom_beta,
    _layout_physical_top_beta,
    _output_gap_for_edge,
    _runtime_config,
)
from orion.experimental.cir.native_halo_conv2d import (
    NativeHaloConv2DSpec,
    _COMPACT_OUTPUT_DIAG_SET_CACHE,
    _compact_output_diag_sets_for_task,
    _compact_output_diag_sets_for_task_torch_oracle,
    _diag_indices_for_task,
    _diag_indices_for_task_torch_oracle,
    native_halo_conv2d_compact_output_rotation_stats,
    native_halo_conv2d_plan,
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
from orion.core.auto_bootstrap import collect_bootstrap_solver_audit
from orion.core.auto_bootstrap import reset_bootstrap_solver_assignments
from orion.core.auto_bootstrap import snapshot_bootstrap_solver_assignments
from orion.nn.activation import SiLU
from orion.nn.linear import Conv2d, ConvTranspose2d
from orion.nn.module import Module
from orion.nn.operations import Add, Bootstrap, Concat
from orion.nn.pooling import AvgPool2d


def _build_u23_dim8_192_dag() -> NetworkDAG:
    from tools import generate_unet22_compile_plan_csv as gen

    saved = (
        int(gen.BASE_DIM),
        str(gen.ACTIVATION),
        int(gen.SILU_DEGREE),
    )
    gen.BASE_DIM = 8
    gen.ACTIVATION = "silu"
    gen.SILU_DEGREE = 7
    try:
        return gen._build_real_unet22_dag(
            height=192,
            width=192,
            in_channels=1,
            out_channels=4,
        )
    finally:
        gen.BASE_DIM, gen.ACTIVATION, gen.SILU_DEGREE = saved


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

    assert int(required.tile_count) == 13
    assert int(required.top_beta) == 1
    assert int(required.bottom_beta) == 1
    assert int(required.physical_top_beta) == 0
    assert int(required.physical_bottom_beta) == 0
    assert bool(required.boundary_pruned) is True
    assert int(filled.tile_count) == int(required.tile_count)
    assert (int(filled.top_beta), int(filled.bottom_beta)) == (1, 1)


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
    assert dp["halo_redundancy_ratio"] == 0.0
    assert all(
        int(row["selected_layout"]["top_beta"]) == 0 and int(row["selected_layout"]["bottom_beta"]) == 0
        for row in dp["edge_layouts"]
    )


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
    dp = _policy(payload, "dp")
    assert dp["relayouts"] == 0
    assert dp["halo_redundancy_ratio"] == 0.0
    assert all(
        int(row["selected_layout"]["top_beta"]) == 0 and int(row["selected_layout"]["bottom_beta"]) == 0
        for row in dp["edge_layouts"]
    )


def test_non_dp_runtime_relayout_insertion_keeps_policy_specific_producer_layouts() -> None:
    def runtime_count(policy: str) -> tuple[int, int]:
        plan = build_layout_policy_compile_plan(build_u22_dag(network_spec("u22_256_base32")), policy=policy)
        runtime_plan = _layout_policy_runtime_compile_plan(plan)
        return int(len(runtime_plan["node_layouts"])), int(runtime_plan["relayout_edge_count"])

    assert runtime_count("fixed_max") == (30, 0)
    assert runtime_count("always") == (30, 33)
    assert runtime_count("greedy") == (30, 4)
    assert runtime_count("dp") == (30, 0)


def test_u22_64_layout_policy_planner_aligns_concat_inputs() -> None:
    payload = build_planner_ablation(network="u22_64_base32")

    for policy in payload["policies"]:
        rows = list(policy["edge_layouts"])
        for concat_node in ("cat4", "cat3", "cat2", "cat1"):
            incoming = [row for row in rows if row["target"] == concat_node]
            assert len(incoming) == 2
            assert len({_layout_key(row) for row in incoming}) == 1
            assert len({str(row["target_physical_layout"]) for row in incoming}) == 1
            for row in incoming:
                if (
                    str(row.get("source_physical_layout", "")) == "packed_compact"
                    and str(row.get("target_physical_layout", "")) == "logical_halo_compact"
                ):
                    assert bool(row["relayout"]) is True


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


def test_layout_policy_no_share_fold_keeps_conservative_boundaries() -> None:
    plan = build_layout_policy_compile_plan(
        build_u22_dag(network_spec("u22_256_base32")),
        policy="dp_no_share_fold",
    )

    assert plan["validation"]["ok"] is True
    concat_rows = [row for row in plan["edge_layouts"] if row["op_kind"] == "concat"]
    assert concat_rows
    assert all(row["physical_layout"] == "packed_compact" for row in concat_rows)
    assert all(row["target_physical_layout"] == "packed_compact" for row in concat_rows)
    assert all(int(row["selected_layout"]["top_beta"]) == 0 for row in concat_rows)
    assert all(int(row["selected_layout"]["bottom_beta"]) == 0 for row in concat_rows)

    tconv_rows = [row for row in plan["edge_layouts"] if row["op_kind"] == "conv_transpose2d"]
    assert tconv_rows
    assert all(row["physical_layout"] != "native_source_stripe" for row in tconv_rows)

    raw_input_rows = [row for row in plan["edge_layouts"] if row["source"] == "x"]
    assert raw_input_rows
    assert all(row["physical_layout"] == "packed_compact" for row in raw_input_rows)

    native_rows = [row for row in plan["edge_layouts"] if row["physical_layout"] == "native_source_stripe"]
    assert native_rows
    assert all(row["source"] != "x" for row in native_rows)
    assert all(row["provider_lt_grouping_mode"] == "individual" for row in native_rows)
    assert all(row["native_halo_channel_fold_mode"] == "per_stripe" for row in native_rows)


def test_layout_policy_no_share_fold_native_rotation_estimate_uses_c_only_plan() -> None:
    dag = _prepared_one_down_one_up_dag(image_size=128, base_channels=8)
    edges = {
        edge.edge_id: edge
        for edge in build_edge_infos(dag, slots=4096)
    }
    plan = build_layout_policy_compile_plan(dag, policy="dp_no_share_fold", slots=4096)

    native_rows = [
        row
        for row in plan["edge_layouts"]
        if row.get("layout_mode") == "native_halo_stripe"
        and not bool(row.get("concat_fusion_runtime_estimate", False))
    ]

    assert native_rows
    for row in native_rows:
        edge = edges[str(row["edge"])]
        layout = _layout_for_shape(
            shape=edge.shape,
            gap=int(row["selected_layout"]["gap"]),
            top_beta=int(row["selected_layout"]["top_beta"]),
            bottom_beta=int(row["selected_layout"]["bottom_beta"]),
            stride=int(row["selected_layout"]["stride"]),
            slots=int(edge.slots),
            physical_top_beta=int(row["selected_layout"].get("physical_top_beta", 0)),
            physical_bottom_beta=int(row["selected_layout"].get("physical_bottom_beta", 0)),
            boundary_pruned=bool(row["selected_layout"].get("boundary_pruned", False)),
        )
        kernel_h, kernel_w = (int(value) for value in edge.kernel_size)
        assert kernel_h == kernel_w
        spec = NativeHaloConv2DSpec(
            family_label=f"test_{str(row['edge']).replace('->', '_')}",
            c_in=int(edge.input_channels or edge.shape[1]),
            h_in=int(edge.shape[2]),
            w_in=int(edge.shape[3]),
            c_out=int(edge.output_channels or edge.output_shape[1]),
            h_out=int(edge.output_shape[2]),
            w_out=int(edge.output_shape[3]),
            gap_in=int(layout.gap),
            gap_out=int(_output_gap_for_edge(edge)),
            kernel=int(kernel_h),
            stride=int(edge.stride[0]),
            pad=int(edge.padding[0]),
            dilation=int(edge.dilation[0]),
            groups=int(edge.groups),
            slot_count=int(edge.slots),
            input_top_beta=int(layout.top_beta),
            input_bottom_beta=int(layout.bottom_beta),
            output_top_beta=0,
            output_bottom_beta=0,
            input_physical_top_beta=_layout_physical_top_beta(layout),
            input_physical_bottom_beta=_layout_physical_bottom_beta(layout),
            output_physical_top_beta=0,
            output_physical_bottom_beta=0,
        )
        native_plan = native_halo_conv2d_plan(
            spec,
            require_native_target_fit=False,
            channel_fold_mode="per_stripe",
        )
        compact_stats = native_halo_conv2d_compact_output_rotation_stats(native_plan)

        assert row["provider_lt_grouping_mode"] == "individual"
        assert row["native_halo_channel_fold_mode"] == "per_stripe"
        assert row["native_halo_plan_channel_fold_mode"] == "per_stripe"
        assert row["native_halo_rotation_mode"] == "c_only"
        assert row["native_halo_rotation_exact_compact_output"] is True
        assert int(row["planner_rotation_cost_estimate"]) == int(compact_stats.c_only_rotations)
        assert int(row["lt_bsgs_rotation_estimate"]) == int(compact_stats.c_only_rotations)
        assert int(row["native_c_only_rotation_estimate"]) == int(compact_stats.c_only_rotations)
        assert int(row["native_cb_shared_rotation_estimate"]) == int(compact_stats.cb_shared_rotations)
        assert int(row["native_plan_c_only_rotation_estimate"]) == int(native_plan.c_only_rotations)
        assert int(row["native_plan_cb_shared_rotation_estimate"]) == int(native_plan.cb_shared_rotations)


def test_native_halo_closed_form_diag_sets_match_torch_oracle() -> None:
    for gap in (1, 2, 4):
        for beta in (0, 1, 2):
            spec = NativeHaloConv2DSpec(
                family_label=f"diag_oracle_g{gap}_b{beta}",
                c_in=17,
                h_in=32,
                w_in=24,
                c_out=19,
                h_out=32,
                w_out=24,
                gap_in=int(gap),
                gap_out=int(gap),
                kernel=3,
                stride=1,
                pad=1,
                dilation=1,
                slot_count=4096,
                input_top_beta=int(beta),
                input_bottom_beta=int(beta),
                output_top_beta=int(beta),
                output_bottom_beta=int(beta),
                input_physical_top_beta=int(beta),
                input_physical_bottom_beta=int(beta),
                output_physical_top_beta=int(beta),
                output_physical_bottom_beta=int(beta),
            )
            plan = native_halo_conv2d_plan(spec, require_native_target_fit=False, channel_fold_mode="per_stripe")
            selected = {
                int(stripe.index): stripe
                for stripe in (plan.stripes[0], plan.stripes[len(plan.stripes) // 2], plan.stripes[-1])
            }
            for stripe in selected.values():
                source_counts = (
                    1,
                    min(5, int(spec.c_in)),
                    int(spec.c_in) % max(1, int(plan.source_tile_for_stripe(stripe))) or int(plan.source_tile_for_stripe(stripe)),
                )
                target_counts = (
                    1,
                    min(7, int(spec.c_out)),
                    int(spec.c_out) % max(1, int(plan.target_tile_for_stripe(stripe))) or int(plan.target_tile_for_stripe(stripe)),
                )
                for source_count in source_counts:
                    for target_count in target_counts:
                        assert _diag_indices_for_task(
                            spec,
                            stripe,
                            source_channel_count=int(source_count),
                            target_channel_count=int(target_count),
                        ) == _diag_indices_for_task_torch_oracle(
                            spec,
                            stripe,
                            source_channel_count=int(source_count),
                            target_channel_count=int(target_count),
                        )
                for source_group in (0, max(0, int(plan.source_group_count_for_stripe(stripe)) - 1)):
                    for target_group in (0, max(0, int(plan.target_group_count_for_stripe(stripe)) - 1)):
                        _COMPACT_OUTPUT_DIAG_SET_CACHE.clear()
                        assert _compact_output_diag_sets_for_task(
                            spec,
                            plan,
                            stripe,
                            source_group=int(source_group),
                            target_group=int(target_group),
                        ) == _compact_output_diag_sets_for_task_torch_oracle(
                            spec,
                            plan,
                            stripe,
                            source_group=int(source_group),
                            target_group=int(target_group),
                        )


@pytest.mark.parametrize("policy", ("fixed_max_no_share_unfused", "always_no_share_unfused"))
def test_layout_policy_non_dp_no_share_variants_use_individual_native_stripe(policy: str) -> None:
    plan = build_layout_policy_compile_plan(
        _prepared_one_down_one_up_dag(image_size=128, base_channels=8),
        policy=policy,
        slots=4096,
    )

    assert plan["validation"]["ok"] is True
    native_rows = [
        row
        for row in plan["edge_layouts"]
        if row.get("layout_mode") == "native_halo_stripe"
    ]

    assert native_rows
    assert all(row["physical_layout"] == "native_source_stripe" for row in native_rows)
    assert all(row["provider_lt_grouping_mode"] == "individual" for row in native_rows)
    assert all(row["native_halo_channel_fold_mode"] == "per_stripe" for row in native_rows)
    assert all(row["native_halo_rotation_mode"] == "c_only" for row in native_rows)


@pytest.mark.parametrize(
    ("public_policy", "canonical"),
    (
        ("fixed_max_no_share", "fixed_max_no_share_fused"),
        ("fixedmax_no_share", "fixed_max_no_share_fused"),
        ("always_no_share", "always_no_share_fused"),
        ("always_relayout_no_share", "always_no_share_fused"),
    ),
)
def test_layout_policy_no_share_defaults_to_fused(public_policy: str, canonical: str) -> None:
    assert normalize_policy(public_policy) == canonical
    plan = build_layout_policy_compile_plan(
        _prepared_one_down_one_up_dag(image_size=128, base_channels=8),
        policy=public_policy,
        slots=4096,
    )
    assert plan["policy"] == canonical
    assert plan["validation"]["ok"] is True


@pytest.mark.parametrize(
    "policy",
    (
        "fixed_max_no_share_fused",
        "fixed_max_no_share_unfused",
        "always_no_share_fused",
        "always_no_share_unfused",
    ),
)
def test_layout_policy_non_dp_no_share_validator_requires_individual_stripe(policy: str) -> None:
    with pytest.raises(ValueError, match="individual LT grouping"):
        validate_layout_policy_compile_plan(
            {
                "policy": policy,
                "edge_layouts": [
                    {
                        "edge": "a->b",
                        "op_kind": "conv2d",
                        "physical_layout": "native_source_stripe",
                        "provider_lt_grouping_mode": "shared",
                        "native_halo_channel_fold_mode": "per_stripe",
                    }
                ],
                "node_layouts": [],
            }
        )
    with pytest.raises(ValueError, match="per-stripe"):
        validate_layout_policy_compile_plan(
            {
                "policy": policy,
                "edge_layouts": [
                    {
                        "edge": "a->b",
                        "op_kind": "conv2d",
                        "physical_layout": "native_source_stripe",
                        "provider_lt_grouping_mode": "individual",
                        "native_halo_channel_fold_mode": "heuristic",
                    }
                ],
                "node_layouts": [],
            }
        )


def test_layout_policy_no_share_fold_validator_rejects_double_fused_relayout() -> None:
    with pytest.raises(ValueError, match="producer and consumer fused"):
        validate_layout_policy_compile_plan(
            {
                "policy": "dp_no_share_fold",
                "edge_layouts": [
                    {
                        "edge": "a->b",
                        "op_kind": "conv2d",
                        "physical_layout": "packed_compact",
                        "consumer_fused_relayout": True,
                        "producer_fused_relayout": True,
                    }
                ],
            }
        )


def test_layout_policy_parser_marks_non_dp_u22_modes_as_provider_executable() -> None:
    opts = _region_first_mode_options("u22_64_base32_layout_eager")
    assert opts["u22_layout_policy"] == "eager"
    assert opts["u22_allowed_nodes"] == ("up1", "up2", "up3", "up4")
    assert _region_first_mode_options("u22_64_base32_layout_always")["u22_layout_policy"] == "always"
    assert _region_first_mode_options("u22_256_base32_layout_fixed_max_fused")["u22_layout_policy"] == "fixed_max_fused"
    assert _region_first_mode_options("u22_256_base32_layout_fixedmax_no_share")["u22_layout_policy"] == "fixed_max_no_share_fused"
    assert _region_first_mode_options("u22_256_base32_layout_fixedmax_no_share_fused")["u22_layout_policy"] == "fixed_max_no_share_fused"
    assert _region_first_mode_options("u22_256_base32_layout_fixedmax_no_share_unfused")["u22_layout_policy"] == "fixed_max_no_share_unfused"
    assert _region_first_mode_options("u22_256_base32_layout_always_no_share")["u22_layout_policy"] == "always_no_share_fused"
    assert _region_first_mode_options("u22_256_base32_layout_always_no_share_fused")["u22_layout_policy"] == "always_no_share_fused"
    assert _region_first_mode_options("u22_256_base32_layout_always_no_share_unfused")["u22_layout_policy"] == "always_no_share_unfused"
    assert _region_first_mode_options("u22_256_base32_layout_dp_no_share_fold")["u22_layout_policy"] == "dp_no_share_fold"
    assert _region_first_mode_options("u22_256_base32_layout_dp_noshare_fold")["u22_layout_policy"] == "dp_no_share_fold"
    assert _region_first_mode_options("generic_layout_dp_no_share_fold")["u22_layout_policy"] == "dp_no_share_fold"
    assert _region_first_mode_options("generic_layout_dp_noshare_fold")["u22_layout_policy"] == "dp_no_share_fold"

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
        "u22_input_pair_conv_shared_rotations",
        "u22_pool_input_pair_shared_rotations",
    }
    excluded = {row["node"]: row["reason"] for row in audit["graph_audit"]["excluded_nodes"]}
    assert {node: excluded[node] for node in ("up4", "up3", "up2", "up1")} == {
        node: "tconv_uses_common_dense_path" for node in ("up4", "up3", "up2", "up1")
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
    assert relayout_groups == []
    assert registry.graph_audit["layout_policy_summary"]["relayout_depth_estimate"] == 0
    assert {
        type(group.executor.base_executor).__name__
        for group in registry.groups
    } >= {
        "InputPairConvRuntimeExecutor",
        "HaloLocalConvRuntimeExecutor",
    }
    excluded = {row["node"]: row["reason"] for row in registry.graph_audit["excluded_nodes"]}
    assert {node: excluded[node] for node in ("up4", "up3", "up2", "up1")} == {
        node: "tconv_uses_common_dense_path" for node in ("up4", "up3", "up2", "up1")
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


def test_layout_policy_provider_rejects_lattigo_slot_mismatch() -> None:
    module = SimpleNamespace(fhe_output_shape=torch.Size([1, 1, 4, 4]))
    base_executor = SimpleNamespace(module=module)
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=base_executor,
        output_node_id="conv",
        compile_plan={"policy": "dp", "slots": 32768, "edge_layouts": []},
    )
    fake_scheme = SimpleNamespace(
        params=SimpleNamespace(
            get_backend=lambda: "lattigo",
            get_slots=lambda: 4096,
        ),
        backend=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="32768 slots.*4096 slots"):
        executor.compile(fake_scheme)


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


def test_no_share_fold_bootstrap_compression_forces_compact_without_tile_savings() -> None:
    conv_module = SimpleNamespace(fhe_output_shape=torch.Size([1, 1, 4, 4]))
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
        "tile_count": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [],
        "node_layouts": [
            {
                "node": "conv",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 6, 4],
                "selected_layout": dict(halo),
                "compact_layout": dict(compact),
                "physical_layout": "logical_halo_compact",
            },
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=conv_module),
        output_node_id="conv",
        compile_plan=compile_plan,
    )
    conv_module.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("conv", module=conv_module, bootstrap=True)

    audit = apply_bootstrap_layout_compression(dag)
    updated = executor.compile_plan["node_layouts"][0]

    assert audit["enabled"] is True
    assert audit["nodes"][0]["forced_compact_boundary"] is True
    assert audit["nodes"][0]["saved_ciphertexts_per_bootstrap"] == 0
    assert updated["physical_layout"] == "packed_compact"
    assert updated["selected_layout"]["tile_count"] == 1
    assert tuple(int(value) for value in conv_module.fhe_output_shape) == (1, 1, 4, 4)


def test_bootstrap_compression_does_not_materialize_lazy_delegate() -> None:
    class LazyDelegateExecutor:
        def __init__(self, child: object) -> None:
            self.base_executor = child

        @property
        def delegate(self) -> object:
            raise AssertionError("delegate property should not be materialized")

        def __getattr__(self, name: str) -> object:
            return getattr(self.delegate, name)

    compile_plan = {
        "policy": "dp_no_share_fold",
        "summary": {},
        "edge_layouts": [],
        "node_layouts": [
            {
                "node": "conv",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": {
                    "top_beta": 0,
                    "bottom_beta": 0,
                    "stride": 1,
                    "gap": 1,
                    "core_slots": 16,
                    "stored_slots": 16,
                    "tile_count": 1,
                },
                "compact_layout": {
                    "top_beta": 0,
                    "bottom_beta": 0,
                    "stride": 1,
                    "gap": 1,
                    "core_slots": 16,
                    "stored_slots": 16,
                    "tile_count": 1,
                },
                "physical_layout": "packed_compact",
            },
        ],
    }
    child = SimpleNamespace(compile_plan=compile_plan)
    module = SimpleNamespace(region_runtime=SimpleNamespace(executor=LazyDelegateExecutor(child)))
    dag = nx.DiGraph()
    dag.add_node("conv", module=module, bootstrap=True)

    audit = apply_bootstrap_layout_compression(dag)

    assert audit["enabled"] is False
    assert audit["reason"] == "no_bootstrap_ct_savings"


def test_bootstrap_compression_finds_materialized_private_delegate() -> None:
    class MaterializedDelegateExecutor:
        def __init__(self, child: object) -> None:
            self._delegate = child

        @property
        def delegate(self) -> object:
            raise AssertionError("delegate property should not be materialized")

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
        "tile_count": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "summary": {},
        "edge_layouts": [],
        "node_layouts": [
            {
                "node": "conv",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 6, 4],
                "selected_layout": dict(halo),
                "compact_layout": dict(compact),
                "physical_layout": "logical_halo_compact",
            },
        ],
    }
    child = SimpleNamespace(compile_plan=compile_plan)
    module = SimpleNamespace(region_runtime=SimpleNamespace(executor=MaterializedDelegateExecutor(child)))
    dag = nx.DiGraph()
    dag.add_node("conv", module=module, bootstrap=True)

    audit = apply_bootstrap_layout_compression(dag)

    assert audit["enabled"] is True
    assert child.compile_plan["node_layouts"][0]["physical_layout"] == "packed_compact"


def test_bootstrap_solver_audit_and_reset_allow_independent_second_solve() -> None:
    class DummyParams:
        def get_slots(self) -> int:
            return 16

    class DummyModule:
        def __init__(self, *, depth: int, slots: int) -> None:
            self.depth = int(depth)
            self.level = None
            self.scheme = SimpleNamespace(params=DummyParams())
            self.fhe_output_shape = torch.Size([1, int(slots)])

    modules = [DummyModule(depth=depth, slots=16) for depth in (1, 1, 1, 1, 0)]
    dag = NetworkDAG(SimpleNamespace(graph=SimpleNamespace(nodes=[])))
    for index, module in enumerate(modules):
        dag.add_node(f"n{index}", module=module)
        if index > 0:
            dag.add_edge(f"n{index - 1}", f"n{index}")
    dag.find_residuals()

    solver = BootstrapSolver(SimpleNamespace(), dag, l_eff=3)
    input_level, bootstraps, slots = solver._solve_once(apply_layout_compression=False)
    audit = collect_bootstrap_solver_audit(dag, l_eff=3)

    assert int(input_level) == 3
    assert int(bootstraps) == 1
    assert slots == [16]
    assert audit["bootstrap_count"] == 1
    assert audit["boot_edges"] == [
        {
            "source": "n0",
            "target": "n1",
            "source_level": 3,
            "target_level": 3,
            "source_depth": 1,
            "bootstrap_ct_count": 1,
            "bootstrapper_slots": 16,
        }
    ]

    snapshot = snapshot_bootstrap_solver_assignments(dag)
    dag.nodes["n0"]["level"] = 99
    modules[0].level = 99
    with pytest.raises(ValueError, match="Automatic bootstrap placement failed"):
        BootstrapSolver(SimpleNamespace(), dag, l_eff=3)._solve_once(apply_layout_compression=False)

    reset_bootstrap_solver_assignments(dag, snapshot)
    input_level_2, bootstraps_2, slots_2 = BootstrapSolver(
        SimpleNamespace(), dag, l_eff=3
    )._solve_once(apply_layout_compression=False)

    assert int(input_level_2) == int(input_level)
    assert int(bootstraps_2) == int(bootstraps)
    assert slots_2 == slots


def test_bootstrap_aware_refinement_finds_non_adjacent_interval_relayout() -> None:
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
        "tile_count": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "a->b",
                "source": "a",
                "target": "b",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(halo),
                "selected_layout": dict(halo),
                "target_layout": dict(halo),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "logical_halo_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
                "relayout_reason": "native_halo_physical_source_stripe_relayout",
                "relayout_depth_estimate": 0,
            },
            {
                "edge": "b->c",
                "source": "b",
                "target": "c",
                "op_kind": "silu",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(halo),
                "selected_layout": dict(halo),
                "target_layout": dict(halo),
                "compact_layout": dict(compact),
                "physical_layout": "logical_halo_compact",
                "source_physical_layout": "logical_halo_compact",
                "target_physical_layout": "logical_halo_compact",
                "relayout": False,
            },
        ],
        "node_layouts": [],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=2)),
        output_node_id="b",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][0]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    module_b = SimpleNamespace(
        depth=2,
        region_runtime=SimpleNamespace(executor=executor),
        set_depth=lambda value: setattr(module_b, "depth", int(value)),
    )
    module_a = SimpleNamespace(depth=1)
    module_c = SimpleNamespace(depth=1)

    dag = nx.DiGraph()
    dag.add_node("a", module=module_a)
    dag.add_node("b", module=module_b)
    dag.add_node("c", module=module_c)
    dag.add_node("d", module=SimpleNamespace(depth=1))
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    dag.add_edge("c", "d")
    first_pass_audit = {
        "bootstrap_count": 1,
        "boot_edges": [
            {
                "source": "c",
                "target": "d",
                "source_level": 1,
                "target_level": 1,
                "source_depth": 1,
                "bootstrap_ct_count": 1,
                "bootstrapper_slots": 16,
            }
        ],
    }

    audit = apply_bootstrap_aware_layout_refinement(dag, first_pass_audit)
    updated_edge = executor.compile_plan["edge_layouts"][0]

    assert audit["enabled"] is True
    assert audit["accepted"][0]["edge"] == "a->b"
    assert audit["accepted"][0]["boot_edge"] == {"source": "c", "target": "d"}
    assert updated_edge["physical_layout"] == "logical_halo_compact"
    assert updated_edge["boot_refinement_reason"] == "boot_interval_native_physical_relayout"
    assert bool(updated_edge["boot_refined_compact_source"]) is True


def test_bootstrap_aware_refinement_enumerates_without_applying_and_restores() -> None:
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
        "tile_count": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "a->b",
                "source": "a",
                "target": "b",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(halo),
                "selected_layout": dict(halo),
                "target_layout": dict(halo),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "logical_halo_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
                "relayout_reason": "native_halo_physical_source_stripe_relayout",
                "relayout_depth_estimate": 0,
            },
            {
                "edge": "b->c",
                "source": "b",
                "target": "c",
                "op_kind": "silu",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(halo),
                "selected_layout": dict(halo),
                "target_layout": dict(halo),
                "compact_layout": dict(compact),
                "physical_layout": "logical_halo_compact",
                "source_physical_layout": "logical_halo_compact",
                "target_physical_layout": "logical_halo_compact",
                "relayout": False,
            },
        ],
        "node_layouts": [],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=2)),
        output_node_id="b",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][0]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    module_b = SimpleNamespace(
        depth=2,
        solver_depth=2,
        region_runtime=SimpleNamespace(executor=executor),
        set_depth=lambda value: setattr(module_b, "depth", int(value)),
    )

    dag = nx.DiGraph()
    dag.add_node("a", module=SimpleNamespace(depth=1))
    dag.add_node("b", module=module_b)
    dag.add_node("c", module=SimpleNamespace(depth=1))
    dag.add_node("d", module=SimpleNamespace(depth=1))
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    dag.add_edge("c", "d")
    first_pass_audit = {
        "bootstrap_count": 1,
        "boot_edges": [{"source": "c", "target": "d"}],
    }

    original_plan = dict(executor.compile_plan)
    enumeration = enumerate_bootstrap_aware_layout_refinement_candidates(dag, first_pass_audit)

    assert enumeration["enabled"] is True
    assert enumeration["candidate_count"] == 1
    assert executor.compile_plan == original_plan

    candidate = dict(enumeration["candidates"][0])
    audit = apply_bootstrap_aware_layout_refinement_candidate(
        dag,
        candidate,
        first_pass_audit=first_pass_audit,
    )

    assert audit["enabled"] is True
    assert executor.compile_plan["edge_layouts"][0]["physical_layout"] == "logical_halo_compact"
    restore_layout_policy_compile_plan(
        dag,
        enumeration["previous_compile_plan"],
        depth_snapshot=enumeration["previous_depths"],
    )
    assert executor.compile_plan == original_plan
    assert module_b.depth == 2


def test_bootstrap_aware_refinement_prefers_interval_beta_lift() -> None:
    compact = {
        "top_beta": 0,
        "bottom_beta": 0,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 16,
        "tile_count": 1,
    }
    beta1 = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 24,
        "tile_count": 1,
        "physical_top_beta": 1,
        "physical_bottom_beta": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "pool->conv1",
                "source": "pool",
                "target": "conv1",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
                "relayout_reason": "native_halo_physical_source_stripe_relayout",
            },
            {
                "edge": "conv1->conv2",
                "source": "conv1",
                "target": "conv2",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
                "relayout_reason": "native_halo_physical_source_stripe_relayout",
            },
        ],
        "node_layouts": [
            {
                "node": "pool",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            },
            {
                "node": "conv1",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            },
            {
                "node": "conv2",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            },
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=3)),
        output_node_id="conv1",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (
        dict(compile_plan["edge_layouts"][0]),
        dict(compile_plan["edge_layouts"][1]),
    )
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    pool = AvgPool2d(kernel_size=1, stride=1)
    conv1 = Conv2d(1, 1, kernel_size=3, padding=1)
    conv2 = Conv2d(1, 1, kernel_size=3, padding=1)
    pool.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("pool", module=pool)
    dag.add_node("conv1", module=conv1)
    dag.add_node("conv2", module=conv2)
    dag.add_node("tail", module=SimpleNamespace(depth=1))
    dag.add_edge("pool", "conv1")
    dag.add_edge("conv1", "conv2")
    dag.add_edge("conv2", "tail")
    first_pass_audit = {
        "bootstrap_count": 1,
        "boot_edges": [{"source": "conv2", "target": "tail"}],
    }

    audit = apply_bootstrap_aware_layout_refinement(dag, first_pass_audit)
    updated = executor.compile_plan
    nodes = {str(row["node"]): dict(row) for row in updated["node_layouts"]}
    edges = {str(row["edge"]): dict(row) for row in updated["edge_layouts"]}

    assert audit["enabled"] is True
    assert audit["strategy"] == "producer_beta_lift"
    assert audit["accepted"][0]["producer"] == "pool"
    assert audit["accepted"][0]["covered_edge_count"] == 2
    assert nodes["pool"]["producer_materialized_halo"] is True
    assert nodes["pool"]["selected_layout"]["top_beta"] == 2
    assert nodes["pool"]["selected_layout"]["bottom_beta"] == 2
    assert edges["pool->conv1"]["relayout"] is False
    assert edges["pool->conv1"]["physical_layout"] == "logical_halo_compact"
    assert edges["conv1->conv2"]["relayout"] is False
    assert edges["conv1->conv2"]["physical_layout"] == "logical_halo_compact"


def test_bootstrap_aware_refinement_lifts_through_activation_between_convs() -> None:
    compact = {
        "top_beta": 0,
        "bottom_beta": 0,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 16,
        "tile_count": 1,
    }
    beta1 = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 24,
        "tile_count": 1,
        "physical_top_beta": 1,
        "physical_bottom_beta": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "conv1->act",
                "source": "conv1",
                "target": "act",
                "op_kind": "silu",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(compact),
                "target_layout": dict(compact),
                "required_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "packed_compact",
                "relayout": False,
            },
            {
                "edge": "act->conv2",
                "source": "act",
                "target": "conv2",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
                "relayout_reason": "native_halo_physical_source_stripe_relayout",
            },
        ],
        "node_layouts": [
            {
                "node": "conv1",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            },
            {
                "node": "act",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            },
            {
                "node": "conv2",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            },
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=3)),
        output_node_id="conv2",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][1]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv1 = Conv2d(1, 1, kernel_size=3, padding=1)
    act = SiLU(degree=7)
    conv2 = Conv2d(1, 1, kernel_size=3, padding=1)
    conv2.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("conv1", module=conv1)
    dag.add_node("act", module=act)
    dag.add_node("conv2", module=conv2)
    dag.add_node("tail", module=SimpleNamespace(depth=1))
    dag.add_edge("conv1", "act")
    dag.add_edge("act", "conv2")
    dag.add_edge("conv2", "tail")
    first_pass_audit = {
        "bootstrap_count": 1,
        "boot_edges": [{"source": "conv2", "target": "tail"}],
    }

    audit = apply_bootstrap_aware_layout_refinement(dag, first_pass_audit)
    updated = executor.compile_plan
    nodes = {str(row["node"]): dict(row) for row in updated["node_layouts"]}
    edges = {str(row["edge"]): dict(row) for row in updated["edge_layouts"]}

    assert audit["enabled"] is True
    assert audit["strategy"] == "activation_transparent_beta_lift"
    assert audit["accepted"][0]["producer"] == "conv1"
    assert audit["accepted"][0]["covered_edge_count"] == 1
    assert audit["accepted"][0]["carried_edges"][0]["edge"] == "conv1->act"
    assert nodes["conv1"]["producer_materialized_halo"] is True
    assert nodes["conv1"]["selected_layout"]["top_beta"] == beta1["top_beta"]
    assert nodes["conv1"]["selected_layout"]["bottom_beta"] == beta1["bottom_beta"]
    assert nodes["act"]["producer_materialized_halo"] is True
    assert edges["conv1->act"]["physical_layout"] == "logical_halo_compact"
    assert edges["act->conv2"]["physical_layout"] == "logical_halo_compact"
    assert edges["act->conv2"]["relayout"] is False


@pytest.mark.parametrize(
    ("producer_module", "middle_module"),
    [
        (AvgPool2d(kernel_size=1, stride=1), SiLU(degree=7)),
        (Conv2d(1, 1, kernel_size=3, padding=1), SimpleNamespace(depth=1)),
    ],
)
def test_bootstrap_aware_refinement_rejects_unsupported_single_native_activation_lift(
    producer_module,
    middle_module,
) -> None:
    compact = {
        "top_beta": 0,
        "bottom_beta": 0,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 16,
        "tile_count": 1,
    }
    beta1 = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 24,
        "tile_count": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "producer->middle",
                "source": "producer",
                "target": "middle",
                "op_kind": type(middle_module).__name__.lower(),
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(compact),
                "target_layout": dict(compact),
                "required_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "packed_compact",
                "relayout": False,
            },
            {
                "edge": "middle->conv",
                "source": "middle",
                "target": "conv",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
                "relayout_reason": "native_halo_physical_source_stripe_relayout",
            },
        ],
        "node_layouts": [
            {
                "node": node,
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            }
            for node in ("producer", "middle", "conv")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=3)),
        output_node_id="conv",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][1]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv = Conv2d(1, 1, kernel_size=3, padding=1)
    conv.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("producer", module=producer_module)
    dag.add_node("middle", module=middle_module)
    dag.add_node("conv", module=conv)
    dag.add_node("tail", module=SimpleNamespace(depth=1))
    dag.add_edge("producer", "middle")
    dag.add_edge("middle", "conv")
    dag.add_edge("conv", "tail")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "conv", "target": "tail"}]},
    )

    assert audit["enabled"] is False
    assert audit["reason"] == "no_supported_boot_interval_native_physical_relayout"
    assert executor.compile_plan["edge_layouts"][1]["physical_layout"] == "native_source_stripe"


def test_bootstrap_aware_refinement_lifts_direct_pool_to_conv_relayout() -> None:
    compact = {
        "top_beta": 0,
        "bottom_beta": 0,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 16,
        "tile_count": 1,
    }
    beta1 = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 24,
        "tile_count": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "pool->conv",
                "source": "pool",
                "target": "conv",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
                "relayout_reason": "native_halo_physical_source_stripe_relayout",
            },
        ],
        "node_layouts": [
            {
                "node": node,
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            }
            for node in ("pool", "conv")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=3)),
        output_node_id="conv",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][0]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv = Conv2d(1, 1, kernel_size=3, padding=1)
    conv.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("pool", module=AvgPool2d(kernel_size=1, stride=1))
    dag.add_node("conv", module=conv)
    dag.add_node("tail", module=SimpleNamespace(depth=1))
    dag.add_edge("pool", "conv")
    dag.add_edge("conv", "tail")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "conv", "target": "tail"}]},
    )
    updated = executor.compile_plan
    nodes = {str(row["node"]): dict(row) for row in updated["node_layouts"]}
    edges = {str(row["edge"]): dict(row) for row in updated["edge_layouts"]}

    assert audit["enabled"] is True
    assert audit["strategy"] == "pool_direct_beta_lift"
    assert audit["accepted"][0]["producer"] == "pool"
    assert audit["accepted"][0]["covered_edge_count"] == 1
    assert nodes["pool"]["producer_materialized_halo"] is True
    assert nodes["pool"]["selected_layout"]["top_beta"] == beta1["top_beta"]
    assert nodes["pool"]["selected_layout"]["bottom_beta"] == beta1["bottom_beta"]
    assert edges["pool->conv"]["physical_layout"] == "logical_halo_compact"
    assert edges["pool->conv"]["relayout"] is False


def test_bootstrap_aware_refinement_rejects_pool_activation_conv_single_native_lift() -> None:
    compact = {
        "top_beta": 0,
        "bottom_beta": 0,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 16,
        "tile_count": 1,
    }
    beta1 = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 24,
        "tile_count": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "pool->act",
                "source": "pool",
                "target": "act",
                "op_kind": "silu",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(compact),
                "target_layout": dict(compact),
                "required_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "packed_compact",
                "relayout": False,
            },
            {
                "edge": "act->conv",
                "source": "act",
                "target": "conv",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
                "relayout_reason": "native_halo_physical_source_stripe_relayout",
            },
        ],
        "node_layouts": [
            {
                "node": node,
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            }
            for node in ("pool", "act", "conv")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=3)),
        output_node_id="conv",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][1]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv = Conv2d(1, 1, kernel_size=3, padding=1)
    conv.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("pool", module=AvgPool2d(kernel_size=1, stride=1))
    dag.add_node("act", module=SiLU(degree=7))
    dag.add_node("conv", module=conv)
    dag.add_node("tail", module=SimpleNamespace(depth=1))
    dag.add_edge("pool", "act")
    dag.add_edge("act", "conv")
    dag.add_edge("conv", "tail")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "conv", "target": "tail"}]},
    )

    assert audit["enabled"] is False
    assert audit["reason"] == "no_supported_boot_interval_native_physical_relayout"
    assert executor.compile_plan["edge_layouts"][1]["physical_layout"] == "native_source_stripe"


def test_bootstrap_aware_refinement_lifts_pool_to_conv_boot_boundary() -> None:
    compact = {
        "top_beta": 0,
        "bottom_beta": 0,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 16,
        "tile_count": 1,
    }
    beta1 = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 24,
        "tile_count": 1,
        "physical_top_beta": 1,
        "physical_bottom_beta": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "pool->conv",
                "source": "pool",
                "target": "conv",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
            },
        ],
        "node_layouts": [
            {
                "node": "pool",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            },
            {
                "node": "conv",
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            },
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=2)),
        output_node_id="conv",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][0]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    pool = AvgPool2d(kernel_size=1, stride=1)
    pool.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("pool", module=pool)
    dag.add_node("conv", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_edge("pool", "conv")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "pool", "target": "conv"}]},
    )
    updated = executor.compile_plan
    nodes = {str(row["node"]): dict(row) for row in updated["node_layouts"]}
    edges = {str(row["edge"]): dict(row) for row in updated["edge_layouts"]}

    assert audit["enabled"] is True
    assert audit["strategy"] == "boot_boundary_pool_direct_beta_lift"
    assert audit["accepted"][0]["boot_edge"] == {"source": "pool", "target": "conv"}
    assert nodes["pool"]["producer_materialized_halo"] is True
    assert nodes["pool"]["selected_layout"]["top_beta"] == 1
    assert edges["pool->conv"]["physical_layout"] == "logical_halo_compact"
    assert edges["pool->conv"]["source_physical_layout"] == "logical_halo_compact"
    assert not executor.native_physical_relayout_rows


def test_bootstrap_aware_refinement_lifts_activation_to_conv_boot_interval_boundary() -> None:
    compact = {
        "top_beta": 0,
        "bottom_beta": 0,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 16,
        "tile_count": 1,
    }
    beta1 = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 24,
        "tile_count": 1,
        "physical_top_beta": 1,
        "physical_bottom_beta": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "conv1->act",
                "source": "conv1",
                "target": "act",
                "op_kind": "silu",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(compact),
                "target_layout": dict(compact),
                "required_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "packed_compact",
                "relayout": False,
            },
            {
                "edge": "act->conv2",
                "source": "act",
                "target": "conv2",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
            },
        ],
        "node_layouts": [
            {
                "node": node,
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            }
            for node in ("conv1", "act", "conv2")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=2)),
        output_node_id="conv2",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][1]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv2 = Conv2d(1, 1, kernel_size=3, padding=1)
    conv2.region_runtime = SimpleNamespace(executor=executor)
    act = SiLU(degree=7)
    act.fhe_output_shape = torch.Size((1, 1, 4, 4))

    dag = nx.DiGraph()
    dag.add_node("conv1", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_node("act", module=act)
    dag.add_node("conv2", module=conv2)
    dag.add_node("tail", module=SimpleNamespace(depth=1))
    dag.add_edge("conv1", "act")
    dag.add_edge("act", "conv2")
    dag.add_edge("conv2", "tail")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {
            "bootstrap_count": 2,
            "boot_edges": [
                {"source": "conv1", "target": "act"},
                {"source": "conv2", "target": "tail"},
            ],
        },
    )
    updated = executor.compile_plan
    nodes = {str(row["node"]): dict(row) for row in updated["node_layouts"]}
    edges = {str(row["edge"]): dict(row) for row in updated["edge_layouts"]}
    act = dag.nodes["act"]["module"]

    assert audit["enabled"] is True
    assert audit["strategy"] == "boot_boundary_activation_beta_lift"
    assert audit["accepted"][0]["producer"] == "conv1"
    assert audit["accepted"][0]["carried_edges"][0]["edge"] == "conv1->act"
    assert nodes["conv1"]["producer_materialized_halo"] is True
    assert nodes["act"]["producer_materialized_halo"] is True
    assert edges["conv1->act"]["physical_layout"] == "logical_halo_compact"
    assert edges["act->conv2"]["physical_layout"] == "logical_halo_compact"
    assert tuple(int(value) for value in act.fhe_output_shape) == (1, 1, 6, 4)
    assert act.layout_policy_output_layout["top_beta"] == 1
    assert not executor.native_physical_relayout_rows
    restore_layout_policy_compile_plan(
        dag,
        audit["previous_compile_plan"],
        depth_snapshot=audit["previous_depths"],
        module_layout_snapshot=audit["_previous_module_layouts"],
    )
    assert tuple(int(value) for value in act.fhe_output_shape) == (1, 1, 4, 4)
    assert not hasattr(act, "layout_policy_output_layout")


def test_bootstrap_aware_refinement_lifts_conv_to_conv_boot_boundary() -> None:
    compact = {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 16, "tile_count": 1}
    beta1 = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 24,
        "tile_count": 1,
        "physical_top_beta": 1,
        "physical_bottom_beta": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "conv1->conv2",
                "source": "conv1",
                "target": "conv2",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
            }
        ],
        "node_layouts": [
            {"node": node, "shape": [1, 1, 4, 4], "fhe_shape": [1, 1, 4, 4], "selected_layout": dict(compact), "compact_layout": dict(compact), "physical_layout": "packed_compact"}
            for node in ("conv1", "conv2")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=2)),
        output_node_id="conv2",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][0]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv1 = Conv2d(1, 1, kernel_size=3, padding=1)
    conv1.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("conv1", module=conv1)
    dag.add_node("conv2", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_edge("conv1", "conv2")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "conv1", "target": "conv2"}]},
    )
    updated = executor.compile_plan
    edges = {str(row["edge"]): dict(row) for row in updated["edge_layouts"]}

    assert audit["enabled"] is True
    assert audit["strategy"] == "boot_boundary_beta_lift"
    assert edges["conv1->conv2"]["physical_layout"] == "logical_halo_compact"
    assert not executor.native_physical_relayout_rows


def test_bootstrap_aware_refinement_lifts_local_activation_tail_cleanup() -> None:
    compact = {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 16, "tile_count": 1}
    beta1 = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 16,
        "stored_slots": 24,
        "tile_count": 1,
        "physical_top_beta": 1,
        "physical_bottom_beta": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "conv1->act",
                "source": "conv1",
                "target": "act",
                "op_kind": "silu",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(compact),
                "target_layout": dict(compact),
                "required_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "packed_compact",
                "relayout": False,
            },
            {
                "edge": "act->conv2",
                "source": "act",
                "target": "conv2",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
            },
        ],
        "node_layouts": [
            {
                "node": node,
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            }
            for node in ("conv1", "act", "conv2")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=2)),
        output_node_id="conv2",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][1]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv2 = Conv2d(1, 1, kernel_size=3, padding=1)
    conv2.region_runtime = SimpleNamespace(executor=executor)
    act = SiLU(degree=7)
    act.fhe_output_shape = torch.Size((1, 1, 4, 4))

    dag = nx.DiGraph()
    dag.add_node("conv1", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_node("act", module=act)
    dag.add_node("conv2", module=conv2)
    dag.add_node("boot_src", module=SimpleNamespace(depth=1))
    dag.add_node("boot_tgt", module=SimpleNamespace(depth=1))
    dag.add_edge("conv1", "act")
    dag.add_edge("act", "conv2")
    dag.add_edge("boot_src", "boot_tgt")

    enumeration = enumerate_bootstrap_aware_layout_refinement_candidates(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "boot_src", "target": "boot_tgt"}]},
    )
    local_candidates = [
        row for row in enumeration["candidates"] if row["strategy"] == "local_activation_beta_lift"
    ]

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "boot_src", "target": "boot_tgt"}]},
    )
    updated = executor.compile_plan
    nodes = {str(row["node"]): dict(row) for row in updated["node_layouts"]}
    edges = {str(row["edge"]): dict(row) for row in updated["edge_layouts"]}

    assert len(local_candidates) == 1
    assert local_candidates[0]["candidate_priority"] == 1
    assert int(local_candidates[0]["rotation_delta"]) > 0
    assert local_candidates[0]["require_bootstrap_count_unchanged"] is True
    assert local_candidates[0]["require_bootstrap_shape_nonincrease"] is True
    assert audit["enabled"] is True
    assert audit["strategy"] == "local_activation_beta_lift"
    assert int(audit["rotation_delta"]) > 0
    assert int(updated["summary"]["producer_fused_rotation_estimate"]) == int(audit["rotation_delta"])
    assert int(updated["summary"]["reported_rotation_estimate"]) == int(audit["rotation_delta"])
    assert audit["require_bootstrap_count_unchanged"] is True
    assert audit["require_bootstrap_shape_nonincrease"] is True
    assert audit["accepted"][0]["producer"] == "conv1"
    assert audit["accepted"][0]["carried_edges"][0]["edge"] == "conv1->act"
    assert nodes["conv1"]["producer_materialized_halo"] is True
    assert nodes["act"]["producer_materialized_halo"] is True
    assert edges["conv1->act"]["physical_layout"] == "logical_halo_compact"
    assert edges["act->conv2"]["physical_layout"] == "logical_halo_compact"
    assert tuple(int(value) for value in act.fhe_output_shape) == (1, 1, 6, 4)
    assert not executor.native_physical_relayout_rows


def test_bootstrap_aware_refinement_rejects_local_activation_cleanup_with_external_fanout() -> None:
    compact = {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 16, "tile_count": 1}
    beta1 = {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 24, "tile_count": 1}
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "conv1->act",
                "source": "conv1",
                "target": "act",
                "op_kind": "silu",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(compact),
                "target_layout": dict(compact),
                "required_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "packed_compact",
                "relayout": False,
            },
            {
                "edge": "act->conv2",
                "source": "act",
                "target": "conv2",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
            },
        ],
        "node_layouts": [
            {
                "node": node,
                "shape": [1, 1, 4, 4],
                "fhe_shape": [1, 1, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
            }
            for node in ("conv1", "act", "conv2")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=2)),
        output_node_id="conv2",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][1]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv2 = Conv2d(1, 1, kernel_size=3, padding=1)
    conv2.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("conv1", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_node("act", module=SiLU(degree=7))
    dag.add_node("conv2", module=conv2)
    dag.add_node("side", module=SimpleNamespace(depth=1))
    dag.add_node("boot_src", module=SimpleNamespace(depth=1))
    dag.add_node("boot_tgt", module=SimpleNamespace(depth=1))
    dag.add_edge("conv1", "act")
    dag.add_edge("act", "conv2")
    dag.add_edge("act", "side")
    dag.add_edge("boot_src", "boot_tgt")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "boot_src", "target": "boot_tgt"}]},
    )

    assert audit["enabled"] is False
    assert audit["reason"] == "no_supported_boot_interval_native_physical_relayout"
    assert executor.compile_plan["edge_layouts"][1]["physical_layout"] == "native_source_stripe"


def test_bootstrap_aware_refinement_rejects_boot_boundary_lift_with_external_fanout() -> None:
    compact = {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 16, "tile_count": 1}
    beta1 = {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 24, "tile_count": 1}
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "conv1->conv2",
                "source": "conv1",
                "target": "conv2",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
            }
        ],
        "node_layouts": [
            {"node": node, "shape": [1, 1, 4, 4], "fhe_shape": [1, 1, 4, 4], "selected_layout": dict(compact), "compact_layout": dict(compact), "physical_layout": "packed_compact"}
            for node in ("conv1", "conv2")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=2)),
        output_node_id="conv2",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][0]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv1 = Conv2d(1, 1, kernel_size=3, padding=1)
    conv1.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("conv1", module=conv1)
    dag.add_node("conv2", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_node("side", module=SimpleNamespace(depth=1))
    dag.add_edge("conv1", "conv2")
    dag.add_edge("conv1", "side")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "conv1", "target": "conv2"}]},
    )

    assert audit["enabled"] is False
    assert audit["reason"] == "no_supported_boot_interval_native_physical_relayout"
    assert executor.compile_plan["edge_layouts"][0]["physical_layout"] == "native_source_stripe"


def test_bootstrap_aware_refinement_rejects_boot_boundary_tconv_source() -> None:
    compact = {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 16, "tile_count": 1}
    beta1 = {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 24, "tile_count": 1}
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 8,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "up->conv",
                "source": "up",
                "target": "conv",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
            }
        ],
        "node_layouts": [
            {"node": node, "shape": [1, 1, 4, 4], "fhe_shape": [1, 1, 4, 4], "selected_layout": dict(compact), "compact_layout": dict(compact), "physical_layout": "packed_compact"}
            for node in ("up", "conv")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=2)),
        output_node_id="conv",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][0]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    up = ConvTranspose2d(1, 1, kernel_size=2, stride=2)
    up.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("up", module=up)
    dag.add_node("conv", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_edge("up", "conv")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "up", "target": "conv"}]},
    )

    assert audit["enabled"] is False
    assert audit["reason"] == "no_supported_boot_interval_native_physical_relayout"
    assert executor.compile_plan["edge_layouts"][0]["physical_layout"] == "native_source_stripe"


def test_bootstrap_aware_refinement_lifts_concat_output_for_local_conv_consumer() -> None:
    compact = {
        "top_beta": 0,
        "bottom_beta": 0,
        "stride": 1,
        "gap": 1,
        "core_slots": 32,
        "stored_slots": 32,
        "tile_count": 1,
    }
    beta1 = {
        "top_beta": 1,
        "bottom_beta": 1,
        "stride": 1,
        "gap": 1,
        "core_slots": 32,
        "stored_slots": 48,
        "tile_count": 1,
        "physical_top_beta": 1,
        "physical_bottom_beta": 1,
    }
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 64,
        "summary": {},
        "edge_layouts": [
            {
                "edge": "skip->cat",
                "source": "skip",
                "target": "cat",
                "op_kind": "concat",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(compact),
                "target_layout": dict(compact),
                "required_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "packed_compact",
                "relayout": False,
            },
            {
                "edge": "up->cat",
                "source": "up",
                "target": "cat",
                "op_kind": "concat",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(compact),
                "target_layout": dict(compact),
                "required_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "packed_compact",
                "relayout": False,
            },
            {
                "edge": "cat->conv",
                "source": "cat",
                "target": "conv",
                "op_kind": "conv2d",
                "shape": [1, 2, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "provider_lt_grouping_mode": "individual",
                "native_halo_channel_fold_mode": "per_stripe",
                "relayout": False,
                "relayout_reason": "native_halo_physical_source_stripe_relayout",
            },
        ],
        "node_layouts": [
            {
                "node": "cat",
                "shape": [1, 2, 4, 4],
                "fhe_shape": [1, 2, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
                "output_relayout": False,
                "producer_materialized_halo": False,
            },
            {
                "node": "conv",
                "shape": [1, 2, 4, 4],
                "fhe_shape": [1, 2, 4, 4],
                "selected_layout": dict(compact),
                "compact_layout": dict(compact),
                "physical_layout": "packed_compact",
                "output_relayout": False,
                "producer_materialized_halo": False,
            },
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=3)),
        output_node_id="conv",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][2]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv = Conv2d(2, 1, kernel_size=3, padding=1)
    conv.name = "conv"
    conv.output_shape = torch.Size((1, 1, 4, 4))
    conv.fhe_output_shape = torch.Size((1, 1, 4, 4))
    conv.output_gap = 1
    conv.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("skip", module=SimpleNamespace(depth=1))
    dag.add_node("up", module=SimpleNamespace(depth=1))
    dag.add_node("cat", module=Concat(dim=1))
    dag.add_node("conv", module=conv)
    dag.add_node("tail", module=SimpleNamespace(depth=1))
    dag.add_edge("skip", "cat")
    dag.add_edge("up", "cat")
    dag.add_edge("cat", "conv")
    dag.add_edge("conv", "tail")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "conv", "target": "tail"}]},
    )
    updated = executor.compile_plan
    nodes = {str(row["node"]): dict(row) for row in updated["node_layouts"]}
    edges = {str(row["edge"]): dict(row) for row in updated["edge_layouts"]}

    assert audit["enabled"] is True
    assert audit["strategy"] == "concat_output_beta_lift"
    assert audit["accepted"][0]["producer"] == "cat"
    assert validate_layout_policy_compile_plan(updated)["ok"] is True
    assert edges["skip->cat"]["physical_layout"] == "packed_compact"
    assert edges["up->cat"]["physical_layout"] == "packed_compact"
    assert edges["skip->cat"]["selected_layout"]["top_beta"] == 0
    assert edges["up->cat"]["selected_layout"]["bottom_beta"] == 0
    assert nodes["cat"]["physical_layout"] == "logical_halo_compact"
    assert nodes["cat"]["producer_materialized_halo"] is True
    assert nodes["cat"]["producer_materialized_halo_reason"] == "boot_interval_concat_output_beta_lift"
    assert nodes["cat"]["selected_layout"]["top_beta"] == beta1["top_beta"]
    assert edges["cat->conv"]["physical_layout"] == "logical_halo_compact"
    assert edges["cat->conv"]["source_physical_layout"] == "logical_halo_compact"
    assert edges["cat->conv"]["layout_mode"] == "concat_output_compact_halo_shared"
    assert edges["cat->conv"]["concat_output_beta_lift"] is True
    assert not executor.native_physical_relayout_rows
    assert executor.compact_source_rows

    conv.compile_plan = updated
    assert conv._concat_source_input_layout({"source": "skip", "concat_node": "cat", "gap": 1})["top_beta"] == 0
    output_attrs = conv._concat_output_layout_attrs()
    assert output_attrs["layout_policy_output_layout"]["top_beta"] == 1
    assert output_attrs["layout_policy_output_materialization"] == "fused_relayout"


def test_bootstrap_aware_refinement_rejects_concat_output_lift_with_external_fanout() -> None:
    compact = {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "core_slots": 32, "stored_slots": 32, "tile_count": 1}
    beta1 = {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "core_slots": 32, "stored_slots": 48, "tile_count": 1}
    compile_plan = {
        "policy": "dp_no_share_fold",
        "slots": 64,
        "summary": {},
        "edge_layouts": [
            {
                "edge": edge,
                "source": source,
                "target": target,
                "op_kind": op_kind,
                "shape": [1, 2 if source == "cat" else 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1 if source == "cat" else compact),
                "target_layout": dict(beta1 if source == "cat" else compact),
                "required_layout": dict(beta1 if source == "cat" else compact),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe" if source == "cat" else "packed_compact",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe" if source == "cat" else "packed_compact",
                "relayout": False,
            }
            for edge, source, target, op_kind in (
                ("skip->cat", "skip", "cat", "concat"),
                ("up->cat", "up", "cat", "concat"),
                ("cat->conv", "cat", "conv", "conv2d"),
            )
        ],
        "node_layouts": [
            {"node": "cat", "shape": [1, 2, 4, 4], "fhe_shape": [1, 2, 4, 4], "selected_layout": dict(compact), "compact_layout": dict(compact), "physical_layout": "packed_compact"},
            {"node": "conv", "shape": [1, 2, 4, 4], "fhe_shape": [1, 2, 4, 4], "selected_layout": dict(compact), "compact_layout": dict(compact), "physical_layout": "packed_compact"},
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=3)),
        output_node_id="conv",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = (dict(compile_plan["edge_layouts"][2]),)
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    conv = Conv2d(2, 1, kernel_size=3, padding=1)
    conv.name = "conv"
    conv.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("skip", module=SimpleNamespace(depth=1))
    dag.add_node("up", module=SimpleNamespace(depth=1))
    dag.add_node("cat", module=Concat(dim=1))
    dag.add_node("conv", module=conv)
    dag.add_node("side", module=SimpleNamespace(depth=1))
    dag.add_node("tail", module=SimpleNamespace(depth=1))
    dag.add_edge("skip", "cat")
    dag.add_edge("up", "cat")
    dag.add_edge("cat", "conv")
    dag.add_edge("cat", "side")
    dag.add_edge("conv", "tail")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "conv", "target": "tail"}]},
    )

    assert audit["enabled"] is False
    assert audit["reason"] == "no_supported_boot_interval_native_physical_relayout"
    assert executor.compile_plan["edge_layouts"][2]["physical_layout"] == "native_source_stripe"


def test_bootstrap_aware_refinement_rejects_beta_lift_with_external_fanout() -> None:
    compact = {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 16, "tile_count": 1}
    beta1 = {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 24, "tile_count": 1}
    compile_plan = {
        "policy": "dp_no_share_fold",
        "summary": {},
        "edge_layouts": [
            {
                "edge": edge,
                "source": source,
                "target": target,
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
            }
            for edge, source, target in (
                ("pool->conv1", "pool", "conv1"),
                ("conv1->conv2", "conv1", "conv2"),
            )
        ],
        "node_layouts": [
            {"node": node, "shape": [1, 1, 4, 4], "fhe_shape": [1, 1, 4, 4], "selected_layout": dict(compact), "compact_layout": dict(compact), "physical_layout": "packed_compact"}
            for node in ("pool", "conv1", "conv2")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=3)),
        output_node_id="conv1",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = tuple(dict(row) for row in compile_plan["edge_layouts"])
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    pool = AvgPool2d(kernel_size=1, stride=1)
    pool.region_runtime = SimpleNamespace(executor=executor)

    dag = nx.DiGraph()
    dag.add_node("pool", module=pool)
    dag.add_node("conv1", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_node("conv2", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_node("tail", module=SimpleNamespace(depth=1))
    dag.add_node("side", module=SimpleNamespace(depth=1))
    dag.add_edge("pool", "conv1")
    dag.add_edge("conv1", "conv2")
    dag.add_edge("conv1", "side")
    dag.add_edge("conv2", "tail")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "conv2", "target": "tail"}]},
    )

    assert audit["enabled"] is False
    assert audit["reason"] == "no_supported_boot_interval_native_physical_relayout"


def test_bootstrap_aware_refinement_rejects_beta_lift_non_conv_native_edge() -> None:
    compact = {"top_beta": 0, "bottom_beta": 0, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 16, "tile_count": 1}
    beta1 = {"top_beta": 1, "bottom_beta": 1, "stride": 1, "gap": 1, "core_slots": 16, "stored_slots": 24, "tile_count": 1}
    compile_plan = {
        "policy": "dp_no_share_fold",
        "summary": {},
        "edge_layouts": [
            {
                "edge": "pool->conv1",
                "source": "pool",
                "target": "conv1",
                "op_kind": "avgpool2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
            },
            {
                "edge": "conv1->conv2",
                "source": "conv1",
                "target": "conv2",
                "op_kind": "conv2d",
                "shape": [1, 1, 4, 4],
                "source_layout": dict(compact),
                "selected_layout": dict(beta1),
                "target_layout": dict(beta1),
                "required_layout": dict(beta1),
                "compact_layout": dict(compact),
                "physical_layout": "native_source_stripe",
                "source_physical_layout": "packed_compact",
                "target_physical_layout": "native_source_stripe",
                "relayout": False,
            },
        ],
        "node_layouts": [
            {"node": node, "shape": [1, 1, 4, 4], "fhe_shape": [1, 1, 4, 4], "selected_layout": dict(compact), "compact_layout": dict(compact), "physical_layout": "packed_compact"}
            for node in ("pool", "conv1", "conv2")
        ],
    }
    executor = LayoutPolicyProviderRuntimeExecutor(
        base_executor=SimpleNamespace(module=SimpleNamespace(depth=3)),
        output_node_id="conv1",
        compile_plan=compile_plan,
    )
    executor.native_physical_relayout_rows = tuple(dict(row) for row in compile_plan["edge_layouts"])
    executor.relayout_rows = ()
    executor.output_relayout_rows = ()
    pool = AvgPool2d(kernel_size=1, stride=1)
    pool.region_runtime = SimpleNamespace(executor=executor)
    dag = nx.DiGraph()
    dag.add_node("pool", module=pool)
    dag.add_node("conv1", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_node("conv2", module=Conv2d(1, 1, kernel_size=3, padding=1))
    dag.add_node("tail", module=SimpleNamespace(depth=1))
    dag.add_edge("pool", "conv1")
    dag.add_edge("conv1", "conv2")
    dag.add_edge("conv2", "tail")

    audit = apply_bootstrap_aware_layout_refinement(
        dag,
        {"bootstrap_count": 1, "boot_edges": [{"source": "conv2", "target": "tail"}]},
    )

    assert audit["enabled"] is False
    assert audit["reason"] == "no_supported_boot_interval_native_physical_relayout"


@pytest.mark.parametrize("single_slot", [False, True])
def test_input_pair_pool_provider_uses_physical_compact_output_for_pruned_halo(
    monkeypatch: pytest.MonkeyPatch,
    single_slot: bool,
) -> None:
    if single_slot:
        monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    else:
        monkeypatch.delenv("ORION_SINGLE_SLOT_LAYER_CACHE", raising=False)
    _init_python_scheme("")
    try:
        pool = AvgPool2d(kernel_size=2, stride=2, padding=0)
        pool.init_orion_params()
        pool.input_shape = torch.Size((1, 8, 8, 8))
        pool.output_shape = torch.Size((1, 8, 4, 4))
        pool.fhe_input_shape = torch.Size((1, 8, 8, 8))
        pool.fhe_output_shape = torch.Size((1, 2, 12, 8))
        pool.input_gap = 1
        pool.output_gap = 2
        pool.layout_policy_output_layout = {
            "top_beta": 1,
            "bottom_beta": 1,
            "physical_top_beta": 0,
            "physical_bottom_beta": 0,
            "gap": 2,
            "boundary_pruned": True,
        }
        pool.layout_policy_output_row_offset = 2
        pool.layout_policy_output_materialization = "fused_relayout"
        pool.update_params()
        pool.on_bias = torch.linspace(-0.125, 0.125, steps=8, dtype=torch.float32)
        pool.set_level(len(scheme.params.get_logq()) - 1)

        executor = InputPairConvRuntimeExecutor(
            module=pool,
            output_node_id="pool_pruned_physical",
            use_ct_pt_hybrid_packing=False,
        )
        x = torch.arange(512, dtype=torch.float32).reshape(1, 8, 8, 8) / 37.0 - 4.0
        out = executor(scheme.encrypt(scheme.encode(x, pool.level)))["pool_pruned_physical"]
        decoded = out.decrypt().decode().detach().cpu()
        if torch.is_complex(decoded):
            decoded = decoded.real
        observed = packing._demultiplex(decoded.to(dtype=torch.float32), 2, 8, 4, 4)
        expected = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        expected = expected + pool.on_bias.detach().reshape(1, 8, 1, 1)
        metadata = executor.compile_cache_metadata()

        assert tuple(int(value) for value in out.on_shape) == (1, 2, 8, 8)
        assert metadata["physical_output_pack"]["fhe_output_shape"] == [1, 2, 8, 8]
        assert metadata["physical_output_pack"]["row_offset"] == 0
        assert metadata["physical_output_pack"]["materialization"] == ""
        assert tuple(int(value) for value in pool.fhe_output_shape) == (1, 2, 12, 8)
        assert pool.layout_policy_output_layout["top_beta"] == 1
        assert pool.layout_policy_output_layout["physical_top_beta"] == 0
        assert pool.layout_policy_output_materialization == "fused_relayout"
        assert int(pool.layout_policy_output_row_offset) == 2
        assert float((observed - expected).abs().max().item()) <= 1.0e-5
    finally:
        scheme.delete_scheme()


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


def test_input_pair_pool_provider_bootstrap_affine_fusion_roundtrips() -> None:
    _init_python_scheme("")
    try:
        pool = AvgPool2d(kernel_size=2, stride=2, padding=0)
        pool.init_orion_params()
        pool.input_shape = torch.Size((1, 8, 8, 8))
        pool.output_shape = torch.Size((1, 8, 4, 4))
        pool.fhe_input_shape = torch.Size((1, 8, 8, 8))
        pool.fhe_output_shape = torch.Size((1, 2, 8, 8))
        pool.input_gap = 1
        pool.output_gap = 2
        pool.update_params()
        pool.set_level(len(scheme.params.get_logq()) - 1)
        pool.set_depth(1)

        base = InputPairConvRuntimeExecutor(
            module=pool,
            output_node_id="pool_boot_fused",
            use_ct_pt_hybrid_packing=False,
        )
        executor = LayoutPolicyProviderRuntimeExecutor(
            base_executor=base,
            output_node_id="pool_boot_fused",
            compile_plan={"policy": "dp_no_share_fold", "edge_layouts": []},
        )
        constant = -0.08674037456512451
        executor._bootstrap_prescale_fusion = {"scale": 1.0, "bias": constant}

        bootstrapper = Bootstrap(torch.tensor(-1.0), torch.tensor(1.0), int(pool.level) - 1)
        bootstrapper.scheme = scheme
        bootstrapper.margin = 1.0
        bootstrapper.he_mode = True
        bootstrapper.fhe_input_shape = torch.Size((1, 2, 8, 8))
        bootstrapper.fit()
        bootstrapper.constant = float(constant)
        bootstrapper.prescale = 1.0
        bootstrapper.postscale = 1.0
        bootstrapper.compile()
        bootstrapper.preprocess_fused = True
        bootstrapper.preprocess_fusion_kind = "producer_affine"

        torch.manual_seed(31)
        x = torch.randn((1, 8, 8, 8), dtype=torch.float32)
        out = executor(scheme.encrypt(scheme.encode(x, pool.level)))["pool_boot_fused"]
        pre_bootstrap = packing._demultiplex(
            out.decrypt().decode().detach().cpu().to(dtype=torch.float32),
            2,
            8,
            4,
            4,
        )
        booted = bootstrapper(out)
        decoded = booted.decrypt().decode().detach().cpu().to(dtype=torch.float32)
        observed = packing._demultiplex(decoded, 2, 8, 4, 4)
        expected = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        assert executor.last_runtime_io["bootstrap_prescale_fused"] is True
        assert float(base.bias_vector.abs().max().item()) > 0.0
        assert float((pre_bootstrap - (expected + float(constant))).abs().max().item()) <= 1.0e-4
        assert bootstrapper._bootstrap_runtime_profile[-1]["preprocess_fused"] is True
        assert float((observed - expected).abs().max().item()) <= 1.0e-4
    finally:
        scheme.delete_scheme()


def test_input_pair_pool_provider_records_cpp_diag_builder_attribution(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CPP_DIAG_BUILDER", "1")
    monkeypatch.setenv("ORION_CPP_DIAG_BUILDER_DENSE", "1")
    monkeypatch.setenv("ORION_CPP_DIAG_BUILDER_SHADOW", "1")
    monkeypatch.setenv("ORION_CPP_DIAG_BUILDER_STRICT", "1")
    _init_python_scheme("")
    try:
        pool = AvgPool2d(kernel_size=2, stride=2, padding=0)
        pool.init_orion_params()
        pool.input_shape = torch.Size((1, 2, 4, 4))
        pool.output_shape = torch.Size((1, 2, 2, 2))
        pool.fhe_input_shape = torch.Size((1, 2, 4, 4))
        pool.fhe_output_shape = torch.Size((1, 2, 4, 4))
        pool.input_gap = 1
        pool.output_gap = 2
        pool.update_params()
        pool.set_level(len(scheme.params.get_logq()) - 1)

        executor = InputPairConvRuntimeExecutor(
            module=pool,
            output_node_id="pool_cppdiag_attr",
            use_ct_pt_hybrid_packing=False,
        )
        executor.compile(scheme)
        timing = dict(executor.last_runtime_timing)

        assert timing["diag_builder_kind"] == "cpp_dense_conv2d"
        assert timing["diag_builder_source"] == "cpp"
        assert timing["diag_builder_build_s"] > 0.0
        assert timing["diag_builder_payload_count"] > 0.0
        assert getattr(pool, "_last_diag_builder_metadata", {})["diag_builder_shadow_ok"] is True
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
    assert producer_fused_edges == []
    assert compile_plan["relayout_edges"] == []
    for materialized_node in (
        "enc1a",
        "enc1b",
        "enc2a",
        "enc2b",
        "enc3a",
        "enc3b",
        "enc4a",
        "enc4b",
        "bottlenecka",
        "bottleneckb",
        "up3",
        "up2",
        "up1",
    ):
        layout = dict(node_rows[materialized_node]["selected_layout"])
        assert bool(node_rows[materialized_node].get("producer_materialized_halo", False)) is True
        assert node_rows[materialized_node]["producer_materialized_halo_reason"] == "dp_producer_materialized_halo"
        assert node_rows[materialized_node]["physical_layout"] == "logical_halo_compact"
        assert int(layout["top_beta"]) >= 1
        assert int(layout["bottom_beta"]) >= 1
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
    assert int(compile_plan["summary"]["relayout_depth_estimate"]) == 0
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


def test_layout_policy_dp_can_avoid_tconv_concat_relayout_with_shared_conv_fallback() -> None:
    dag = build_u22_dag(network_spec("u22_256_base32"))
    compile_plan = build_layout_policy_compile_plan(dag, policy="dp")
    node_rows = {str(row["node"]): row for row in compile_plan["node_layouts"]}
    edge_rows = {str(row["edge"]): row for row in compile_plan["edge_layouts"]}

    for up_node, join_node, join_edge, consumer_edge in (
        ("up3", "cat3", "up3->cat3", "cat3->dec3a"),
        ("up2", "cat2", "up2->cat2", "cat2->dec2a"),
        ("up1", "cat1", "up1->cat1", "cat1->dec1a"),
    ):
        up_layout = dict(node_rows[up_node]["selected_layout"])
        join_layout = dict(node_rows[join_node]["selected_layout"])
        up_join = dict(edge_rows[join_edge])
        join_dec = dict(edge_rows[consumer_edge])

        assert up_join["op_kind"] == "concat"
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


def test_one_down_one_up_silu_dp_accounts_for_join_relayout_without_bootstrap_growth() -> None:
    _init_long_python_scheme("")
    try:
        greedy_dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)
        dp_dag = _prepared_one_down_one_up_dag(image_size=192, base_channels=8)
        greedy_plan = build_layout_policy_compile_plan(greedy_dag, policy="greedy")
        dp_plan = build_layout_policy_compile_plan(dp_dag, policy="dp")

        assert int(dp_plan["summary"]["relayout_depth_estimate"]) == 1
        assert int(dp_plan["summary"]["relayouts"]) == 1
        assert {str(row["edge"]) for row in dp_plan["relayout_edges"]} == {"enc_act->add"}
        assert int(dp_plan["summary"]["consumer_fused_relayout_count"]) > 0
        assert int(dp_plan["summary"]["total_ciphertext_tiles"]) <= int(
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
                assert int(audit["graph_audit"]["layout_policy_summary"]["relayout_depth_estimate"]) == 1
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

    assert int(fixed["summary"]["relayout_depth_estimate"]) == 0
    assert int(eager["summary"]["relayout_depth_estimate"]) == 0
    assert int(greedy["summary"]["relayout_depth_estimate"]) > 0
    assert int(fixed_fused["summary"]["relayout_depth_estimate"]) <= int(fixed["summary"]["relayout_depth_estimate"])
    assert int(eager_fused["summary"]["relayout_depth_estimate"]) == 0
    assert int(greedy_fused["summary"]["relayout_depth_estimate"]) == int(
        greedy["summary"]["relayout_depth_estimate"]
    )
    assert int(fixed_fused["summary"]["relayouts"]) <= int(fixed["summary"]["relayouts"])
    assert int(eager_fused["summary"]["relayouts"]) == 0
    assert int(greedy_fused["summary"]["relayouts"]) == int(greedy["summary"]["relayouts"])
    assert int(eager_fused["summary"]["consumer_fused_relayout_count"]) == 0
    assert int(fixed_fused["summary"]["producer_fused_materialization_count"]) == 0


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


def test_no_share_fold_registry_sets_individual_provider_grouping() -> None:
    _init_long_python_scheme("")
    try:
        dag = _prepared_one_down_one_up_dag(image_size=128, base_channels=8)
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=None,
            enable_conv_kernels=True,
            layout_policy="dp_no_share_fold",
        )
        registry.attach_to_dag(dag)

        executors = [
            dag.nodes[node]["module"].region_runtime.executor
            for node in dag.nodes
            if getattr(dag.nodes[node].get("module"), "region_runtime", None) is not None
        ]
        attrs = [
            executor._native_halo_module_attrs()
            for executor in executors
            if isinstance(executor, LayoutPolicyProviderRuntimeExecutor)
        ]

        assert attrs
        assert all(row["layout_policy_provider_lt_grouping_mode"] == "individual" for row in attrs)
        assert all(row["layout_policy_native_halo_channel_fold_mode"] == "per_stripe" for row in attrs)
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
            "always_fused",
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
    assert [row["policy"] for row in rows] == ["fixed_max", "always_fused", "dp"]
    assert all(row["metric_source"] == "planner_estimate" for row in rows)
    assert "diagonal_key_set_mismatch_count" in rows[0]
    assert "provider_input_block_cols" in rows[0]


def test_u22_224_silu7_policy_table_uses_always_fused_and_policy_aware_bootstrap() -> None:
    from tools import generate_u22_224_silu7_policy_runtime_table as table

    rows, metadata = table._planner_rows()
    by_policy = {str(row["policy"]): row for row in rows}

    assert [str(row["policy"]) for row in rows] == ["fixed_max", "always_fused", "dp"]
    assert str(metadata["bootstrap_source"]) == "policy_aware_bootstrap_solver_after_registry_attach"
    assert int(by_policy["fixed_max"]["boot"]) == 143
    assert int(by_policy["always_fused"]["boot"]) == 169
    assert int(by_policy["dp"]["boot"]) == 169
    assert int(by_policy["always_fused"]["fused_relayout"]) > 0
    assert int(by_policy["always_fused"]["rotation"]) > int(by_policy["dp"]["rotation"])


def test_u22_224_silu7_provider_runner_accepts_always_fused_policy() -> None:
    from tools import run_u22_base32_silu7_streaming_provider_e2e as runner

    assert runner._provider_mode("always_fused") == "u22_256_base32_layout_always_fused"
    assert runner._provider_mode("fixed_max") == "u22_256_base32_layout_fixedmax_no_share"
    assert runner._provider_mode("fixed_max_no_share_fused") == "u22_256_base32_layout_fixedmax_no_share"
    assert runner._provider_mode("fixed_max_no_share_unfused") == "u22_256_base32_layout_fixedmax_no_share_unfused"
    assert runner._provider_mode("always_relayout_no_share") == "u22_256_base32_layout_always_no_share"
    assert runner._provider_mode("always_no_share_fused") == "u22_256_base32_layout_always_no_share"
    assert runner._provider_mode("always_no_share_unfused") == "u22_256_base32_layout_always_no_share_unfused"

    env = runner._apply_env_defaults(
        {
            "GOMAXPROCS": "99",
            "ORION_LATTIGO_BOOTSTRAP_MANY": "1",
            "ORION_UNIFIED_LT_INDIVIDUAL_EVAL": "0",
            "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS": "1",
            "ORION_LATTIGO_UNIFIED_NO_BSGS": "1",
        }
    )
    assert env["GOMAXPROCS"] == "1"
    assert env["ORION_LATTIGO_BOOTSTRAP_MANY"] == "0"
    assert env["ORION_UNIFIED_LT_INDIVIDUAL_EVAL"] == "1"
    assert env["ORION_UNIFIED_LT_SHARED_ROTATION_KEYS"] == "0"
    assert env["ORION_LATTIGO_UNIFIED_NO_BSGS"] == "0"


def test_u22_dim32_runners_accept_no_share_fold_policy() -> None:
    from tools import run_u22_dim32_dense_provider_e2e_matrix as matrix_runner
    from tools import run_u22_dim32_encoder4_noshare_e2e as encoder4_runner

    assert matrix_runner._provider_mode("dp_no_share_fold") == "u22_256_base32_layout_dp_no_share_fold"
    assert matrix_runner._provider_mode("dp_noshare_fold") == "u22_256_base32_layout_dp_no_share_fold"
    assert matrix_runner._provider_mode("fixed_max") == "u22_256_base32_layout_fixedmax_no_share"
    assert matrix_runner._provider_mode("fixed_max_no_share_fused") == "u22_256_base32_layout_fixedmax_no_share"
    assert matrix_runner._provider_mode("fixed_max_no_share_unfused") == "u22_256_base32_layout_fixedmax_no_share_unfused"
    assert matrix_runner._provider_mode("always_no_share") == "u22_256_base32_layout_always_no_share"
    assert matrix_runner._provider_mode("always_no_share_fused") == "u22_256_base32_layout_always_no_share"
    assert matrix_runner._provider_mode("always_no_share_unfused") == "u22_256_base32_layout_always_no_share_unfused"
    assert encoder4_runner._provider_mode("dp_no_share_fold") == "u22_256_base32_layout_dp_no_share_fold"
    assert encoder4_runner._provider_mode("noshare_fold") == "u22_256_base32_layout_dp_no_share_fold"
    assert encoder4_runner._provider_mode("fixed_max") == "u22_256_base32_layout_fixedmax_no_share"
    assert encoder4_runner._provider_mode("fixed_max_no_share_fused") == "u22_256_base32_layout_fixedmax_no_share"
    assert encoder4_runner._provider_mode("fixed_max_no_share_unfused") == "u22_256_base32_layout_fixedmax_no_share_unfused"
    assert encoder4_runner._provider_mode("always_relayout_no_share") == "u22_256_base32_layout_always_no_share"
    assert encoder4_runner._provider_mode("always_no_share_fused") == "u22_256_base32_layout_always_no_share"
    assert encoder4_runner._provider_mode("always_no_share_unfused") == "u22_256_base32_layout_always_no_share_unfused"


def test_u22_dim32_matrix_runner_forces_noshare_mainline_env() -> None:
    from tools import run_u22_dim32_dense_provider_e2e_matrix as matrix_runner
    from tools import run_u22_dim32_encoder4_noshare_e2e as encoder4_runner

    bad_env = {
            "GOMAXPROCS": "99",
            "ORION_LATTIGO_BOOTSTRAP_MANY": "1",
            "ORION_UNIFIED_LT_INDIVIDUAL_EVAL": "0",
            "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS": "1",
            "ORION_LATTIGO_UNIFIED_NO_BSGS": "1",
            "ORION_CONCAT_FUSION": "0",
            "ORION_PACK_CONV_WORKERS": "240",
    }
    for runner in (matrix_runner, encoder4_runner):
        env = runner._apply_env_defaults(dict(bad_env))
        assert env["GOMAXPROCS"] == "1"
        assert env["ORION_SINGLE_SLOT_LAYER_CACHE"] == "1"
        assert env["ORION_LATTIGO_BOOTSTRAP_MANY"] == "0"
        assert env["ORION_UNIFIED_LT_INDIVIDUAL_EVAL"] == "1"
        assert env["ORION_UNIFIED_LT_SHARED_ROTATION_KEYS"] == "0"
        assert env["ORION_LATTIGO_UNIFIED_NO_BSGS"] == "0"
        assert env["ORION_CONCAT_FUSION"] == "1"
    assert matrix_runner._apply_env_defaults(dict(bad_env))["ORION_PACK_CONV_WORKERS"] == "240"


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
