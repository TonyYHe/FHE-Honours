from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from orion.experimental import layout_policy_ablation as layout_planner
from orion.experimental.u22_phase1 import (
    _u22_input_pair_conv_module_supported,
    _u22_pool_module_supported,
    _u22_same_shape_conv_module_supported,
    _u22_tconv_module_supported,
)
from orion.core.auto_bootstrap import BootstrapSolver
from orion.core.bootstrap_fusion import module_bootstrap_ct_count, module_bootstrap_slots
from orion.core.level_dag import LevelDAG
from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer, StatsTracker
from orion.models.unet import UNet22, _apply_u22_activation, _make_u22_activation
from orion.nn.module import Module
from orion.nn.activation import SiLU
from orion.nn.linear import Conv2d, ConvTranspose2d
from orion.nn.operations import Add
from orion.nn.pooling import AvgPool2d
import orion.nn as on


SLOTS = 32768
POLICY = "dp"
ESTIMATOR = "template"
BASE_DIM = 64
ACTIVATION = "silu"
SILU_DEGREE = 7
BOOTSTRAP_L_EFF = len(layout_planner.U22_E2E_LOGQ) - 1

# Non-DP policies call the LT estimator through the planner module default in a
# few helper paths, so set it explicitly here as well as passing estimator= below.
layout_planner.LAYOUT_ESTIMATOR_DEFAULT = layout_planner.LAYOUT_ESTIMATOR_TEMPLATE


PAPER_OPERATOR_ORDER = [
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
    "output",
]


class UNet22PlusOutput(UNet22):
    """UNet22 body plus a separate logits/output layer.

    The repository's UNet22 currently folds the output projection into dec1b.
    This variant keeps dec1b as the final 64->64 decoder conv, applies SiLU7,
    and adds an explicit 1x1 output layer.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        activation: str,
        silu_degree: int,
    ) -> None:
        super().__init__(
            dataset="kvasir_polyp_256",
            in_channels=int(in_channels),
            out_channels=int(base_channels),
            base_channels=int(base_channels),
            activation=str(activation),
            silu_degree=int(silu_degree),
        )
        requested_out = int(out_channels)
        self.out_channels = requested_out
        self.dec1b_act = _make_u22_activation(self.activation, silu_degree=self.silu_degree)
        self.output = on.Conv2d(int(base_channels), requested_out, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = _apply_u22_activation(self.enc1a_act, self.enc1a(x))
        skip1 = _apply_u22_activation(self.enc1b_act, self.enc1b(x))

        x = _apply_u22_activation(self.enc2a_act, self.enc2a(self.pool1(skip1)))
        skip2 = _apply_u22_activation(self.enc2b_act, self.enc2b(x))

        x = _apply_u22_activation(self.enc3a_act, self.enc3a(self.pool2(skip2)))
        skip3 = _apply_u22_activation(self.enc3b_act, self.enc3b(x))

        x = _apply_u22_activation(self.enc4a_act, self.enc4a(self.pool3(skip3)))
        skip4 = _apply_u22_activation(self.enc4b_act, self.enc4b(x))

        x = _apply_u22_activation(self.bottlenecka_act, self.bottlenecka(self.pool4(skip4)))
        x = _apply_u22_activation(self.bottleneckb_act, self.bottleneckb(x))

        x = _apply_u22_activation(self.dec4a_act, self.dec4a(self.add4(self.up4(x), skip4)))
        x = _apply_u22_activation(self.dec4b_act, self.dec4b(x))
        x = _apply_u22_activation(self.dec3a_act, self.dec3a(self.add3(self.up3(x), skip3)))
        x = _apply_u22_activation(self.dec3b_act, self.dec3b(x))
        x = _apply_u22_activation(self.dec2a_act, self.dec2a(self.add2(self.up2(x), skip2)))
        x = _apply_u22_activation(self.dec2b_act, self.dec2b(x))
        x = _apply_u22_activation(self.dec1a_act, self.dec1a(self.add1(self.up1(x), skip1)))
        x = _apply_u22_activation(self.dec1b_act, self.dec1b(x))
        return self.output(x)


class _DummyParams:
    def get_slots(self) -> int:
        return int(SLOTS)

    def get_debug_status(self) -> bool:
        return False

    def get_io_mode(self) -> str:
        return "none"

    def get_compile_save_resume(self) -> bool:
        return False


def _ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def _shape_text(value: Any) -> str:
    return "x".join(str(int(item)) for item in tuple(value))


def _ct_count(shape: Any, *, slots: int = SLOTS) -> int:
    return max(1, _ceil_div(int(torch.Size(tuple(int(v) for v in shape)).numel()), int(slots)))


def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _tex_int(value: int | str) -> str:
    if value == "--":
        return "--"
    return f"{int(value):,}".replace(",", "{,}")


def _paper_case_label(case: str) -> str:
    labels = {
        "192x192_IBSR_BRAIN_2D": "M1 $192{\\times}192$, $1{\\to}4$ (IBSR)",
        "224x224_HanCo_Hand": "M2 $224{\\times}224$, $3{\\to}1$ (HanCo)",
        "384x288_CVC_ClinicDB": "M3 $384{\\times}288$, $3{\\to}1$ (CVC)",
        "384x384_Satellite_cloud": "M4 $384{\\times}384$, $4{\\to}1$ (Cloud)",
    }
    return labels[str(case)]


def _paper_visible_layout(row: dict[str, Any]) -> tuple[int, int]:
    layout = json.loads(str(row["selected_output_layout"])) if row.get("selected_output_layout") else {}
    width = int(str(row["image_hw"]).split("x")[1])
    gap = int(row.get("output_gap") or 1)
    tile_count = max(1, int(layout.get("tile_count", row.get("ct_count", 1)) or row.get("ct_count", 1)))
    core_slots = int(layout.get("core_slots", 0) or 0)
    if core_slots > 0:
        alpha = math.floor(float(core_slots) / float(tile_count * width * gap))
    else:
        alpha = math.floor(float(SLOTS) / float(width * gap))
    beta = (int(layout.get("top_beta", 0) or 0) + int(layout.get("bottom_beta", 0) or 0)) // 2
    return int(alpha), int(beta)


def _paper_bootstrap_for_node(rows_by_node: dict[str, dict[str, Any]], node: str) -> str:
    row = rows_by_node.get(str(node), {})
    direct = int(row.get("bootstrap_count_after_layer", 0) or 0)
    if direct > 0:
        return _tex_int(direct)
    act = rows_by_node.get(f"{node}_act", {})
    via_activation = int(act.get("bootstrap_count_after_layer", 0) or 0)
    return "--" if via_activation <= 0 else _tex_int(via_activation)


def render_operator_plan_table(rows: list[dict[str, Any]]) -> str:
    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    rows_by_case = {
        case: {str(row["node"]): row for row in rows if str(row["case"]) == case}
        for case in cases
    }
    caption = (
        "Per-operator layout and compile-plan summary for U\\text{-}Net workloads with "
        "23 learned/linear layers plus four pooling transition operators. Model groups "
        "correspond to M1: IBSR Brain 2D, $192{\\times}192$, $C_{\\mathrm{in}}{=}1$, "
        "$C_{\\mathrm{out}}{=}4$; M2: HanCo hand segmentation, $224{\\times}224$, "
        "$3{\\to}1$; M3: CVC-ClinicDB, $384{\\times}288$, $3{\\to}1$; and M4: "
        "satellite cloud segmentation, $384{\\times}384$, $4{\\to}1$. Each group "
        "reports output-layout interior rows $\\alpha$, averaged actual output-layout "
        "halo rows $\\beta$, output ciphertext count (CT), estimated rotations (R), "
        "diagonal/CT--PT multiplication count (M), and bootstrap count inserted after "
        "the operator or its following activation (B). Pooling rows expose downsampling "
        "operators into which output re-layout can be fused. Here $\\alpha$ is the "
        "per-ciphertext useful row budget after removing the planner-selected top and "
        "bottom halos; $\\beta=\\lfloor(\\beta_t+\\beta_b)/2\\rfloor$ compresses the "
        "selected top/bottom output halo into one column. Dashes indicate no bootstrap."
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\begingroup",
        "\\tiny",
        "\\setlength{\\tabcolsep}{1.05pt}",
        "\\renewcommand{\\arraystretch}{0.98}",
        f"\\caption{{{caption}}}",
        "\\label{tab:unet27-operator-plan}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lrrrrrrrrrrrrrrrrrrrrrrrr@{}}",
        "\\toprule",
        " & "
        + " & ".join(f"\\multicolumn{{6}}{{c}}{{{_paper_case_label(case)}}}" for case in cases)
        + " \\\\",
        "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\\cmidrule(lr){14-19}\\cmidrule(l){20-25}",
        "Op. & "
        + " & ".join(["$\\alpha$ & $\\beta$ & CT & R & M & B"] * len(cases))
        + " \\\\",
        "\\midrule",
    ]
    for index, node in enumerate(PAPER_OPERATOR_ORDER, start=1):
        cells = [f"{index}~\\texttt{{{node}}}"]
        for case in cases:
            by_node = rows_by_case[case]
            row = by_node[node]
            alpha, beta = _paper_visible_layout(row)
            cells.extend(
                [
                    str(alpha),
                    str(beta),
                    _tex_int(row["ct_count"]),
                    _tex_int(row["rotation"]),
                    _tex_int(row["diagonal_count"]),
                    _paper_bootstrap_for_node(by_node, node),
                ]
            )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\endgroup",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_real_unet22_dag(*, height: int, width: int, in_channels: int, out_channels: int) -> NetworkDAG:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    Module.set_margin(2)
    scheme = SimpleNamespace(params=_DummyParams())

    model = UNet22PlusOutput(
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        base_channels=int(BASE_DIM),
        activation=ACTIVATION,
        silu_degree=int(SILU_DEGREE),
    )
    model.eval()
    traced = OrionTracer().trace_model(model)
    sample = torch.randn((1, int(in_channels), int(height), int(width)), dtype=torch.float32)
    StatsTracker(traced).propagate(sample)

    for module in traced.modules():
        if isinstance(module, Module):
            module.scheme = scheme
        if hasattr(module, "fit"):
            module.fit()
    for module in traced.modules():
        if hasattr(module, "init_orion_params"):
            module.init_orion_params()
        if hasattr(module, "update_params"):
            module.update_params()

    dag = NetworkDAG(traced)
    dag.build_dag()
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None:
            module.name = str(node)
            module.scheme = scheme
    return dag


def _max_layout(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    layouts = [dict(row.get(key, {}) or {}) for row in rows if row.get(key)]
    if not layouts:
        return {}
    return {
        "top_beta": max(int(layout.get("top_beta", layout.get("alpha", 0)) or 0) for layout in layouts),
        "bottom_beta": max(int(layout.get("bottom_beta", layout.get("beta", 0)) or 0) for layout in layouts),
        "stride": max(int(layout.get("stride", 1) or 1) for layout in layouts),
        "gap": max(int(layout.get("gap", 1) or 1) for layout in layouts),
        "tile_count": max(int(layout.get("tile_count", 1) or 1) for layout in layouts),
        "stored_slots": max(int(layout.get("stored_slots", 0) or 0) for layout in layouts),
    }


def _module_params(module: Any) -> dict[str, str | int]:
    payload: dict[str, str | int] = {
        "kernel": "",
        "stride": "",
        "padding": "",
        "dilation": "",
        "groups": "",
        "in_channels": "",
        "out_channels": "",
    }
    for attr in ("kernel_size", "stride", "padding", "dilation"):
        if hasattr(module, attr):
            payload[attr.replace("_size", "")] = "x".join(str(int(v)) for v in tuple(getattr(module, attr)))
    for attr in ("groups", "in_channels", "out_channels"):
        if hasattr(module, attr):
            payload[attr] = int(getattr(module, attr))
    payload["activation_kind"] = "silu" if isinstance(module, SiLU) else ""
    payload["activation_degree"] = int(getattr(module, "degree")) if isinstance(module, SiLU) else ""
    if hasattr(module, "output_padding"):
        payload["output_padding"] = "x".join(str(int(v)) for v in tuple(getattr(module, "output_padding")))
    else:
        payload["output_padding"] = ""
    return payload


def _provider_static_support(module: Any) -> tuple[bool | str, str, str]:
    if isinstance(module, AvgPool2d):
        ok = _u22_pool_module_supported(module)
        return bool(ok), "u22_pool_provider" if ok else "", "" if ok else "u22_pool_requires_stride2_avgpool"
    if isinstance(module, ConvTranspose2d):
        ok = _u22_tconv_module_supported(module)
        return bool(ok), "u22_tconv_provider" if ok else "", "" if ok else "u22_tconv_requires_k2s2_gap_halving"
    if isinstance(module, Conv2d):
        if _u22_same_shape_conv_module_supported(module):
            return True, "u22_same_shape_conv_provider", ""
        if _u22_input_pair_conv_module_supported(module):
            return True, "u22_input_pair_conv_provider", ""
        return False, "", "u22_conv_requires_supported_same_shape_kernel_padding"
    if isinstance(module, SiLU):
        return "", "chebyshev_polynomial_activation", "not_a_provider_linear_kernel"
    if isinstance(module, Add):
        return "", "add_runtime_or_dense_add", "not_a_provider_linear_kernel"
    return "", "", "not_a_provider_linear_kernel"


def _fusion_side(in_rows: list[dict[str, Any]], node_row: dict[str, Any]) -> tuple[str, str]:
    notes: list[str] = []
    if any(bool(row.get("producer_fused_relayout", False)) for row in in_rows):
        notes.append("source:producer_fused_relayout")
    if any(bool(row.get("consumer_fused_relayout", False)) for row in in_rows):
        notes.append("target:consumer_fused_relayout")
    if bool(node_row.get("producer_materialized_halo", False)):
        notes.append("source:producer_materialized_halo")
    if any(bool(row.get("relayout", False)) for row in in_rows):
        notes.append("source_to_target:separate_relayout_edge")
    if bool(node_row.get("output_relayout", False)):
        notes.append("source_to_target:output_relayout_node")
    if not notes:
        return "none", ""
    sides = []
    if any(item.startswith("source:") for item in notes):
        sides.append("source")
    if any(item.startswith("target:") for item in notes):
        sides.append("target")
    if any(item.startswith("source_to_target:") for item in notes):
        sides.append("source_to_target")
    return "+".join(sides), ";".join(notes)


def _attach_bootstrap_dummy_scheme(dag: Any) -> None:
    scheme = SimpleNamespace(params=_DummyParams())
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None:
            module.scheme = scheme


def _bootstrap_node_map(shortest_path: set[str]) -> dict[str, str]:
    node_map: dict[str, str] = {}
    for item in shortest_path:
        name = str(item).split("@")[0]
        node_map[str(name)] = str(item)
    return node_map


def _level_from_node_token(token: str | None) -> int | None:
    if not token:
        return None
    return int(str(token).split("=")[-1])


def _expanded_bootstrap_targets(dag: Any, targets: list[str]) -> list[str]:
    expanded: list[str] = []
    for target in targets:
        module = dag.nodes[target].get("module") if target in dag.nodes else None
        if module is not None:
            expanded.append(str(target))
            continue
        successors = [str(child) for child in dag.successors(target)] if target in dag.nodes else []
        expanded.extend(successors or [str(target)])
    return list(dict.fromkeys(expanded))


def _bootstrap_plan_for_dag(dag: Any, original_nodes: list[str]) -> dict[str, Any]:
    _attach_bootstrap_dummy_scheme(dag)
    dag.find_residuals()
    solver = BootstrapSolver(SimpleNamespace(), dag, l_eff=int(BOOTSTRAP_L_EFF))
    input_level, bootstrap_count, bootstrapper_slots = solver.solve()
    node_map = _bootstrap_node_map(solver.shortest_path)
    query = LevelDAG(l_eff=int(BOOTSTRAP_L_EFF), network_dag=dag, path=None)

    nodes: dict[str, dict[str, Any]] = {}
    for node in original_nodes:
        module = dag.nodes[node].get("module")
        node_token = node_map.get(str(node))
        level = _level_from_node_token(node_token)
        depth = int(getattr(module, "depth", 0) or 0) if module is not None else 0
        output_level = "" if level is None else int(level) - int(depth)
        raw_targets: list[str] = []
        target_levels: list[str] = []
        edge_boot_count = 0
        if level is not None:
            for child in dag.successors(node):
                child = str(child)
                child_token = node_map.get(child)
                if child_token is None:
                    continue
                _latency, curr_boots = query.estimate_bootstrap_latency(str(node_token), str(child_token))
                if int(curr_boots) <= 0:
                    continue
                raw_targets.append(child)
                child_level = _level_from_node_token(child_token)
                target_levels.append("" if child_level is None else str(int(child_level)))
                if int(edge_boot_count) == 0:
                    edge_boot_count = int(curr_boots)
        bootstrap_after = bool(dag.nodes[node].get("bootstrap", False))
        bootstrap_shape = tuple(int(value) for value in getattr(module, "fhe_output_shape", ())) if module is not None else ()
        bootstrap_elements = int(torch.Size(bootstrap_shape).numel()) if bootstrap_shape else 0
        nodes[str(node)] = {
            "assigned_level": "" if level is None else int(level),
            "module_depth": int(depth),
            "output_level_before_bootstrap": output_level,
            "bootstrap_after_layer": bool(bootstrap_after),
            "bootstrap_reason": "child_level_exceeds_output_level" if bootstrap_after else "",
            "bootstrap_before_nodes": ";".join(_expanded_bootstrap_targets(dag, raw_targets)) if bootstrap_after else "",
            "bootstrap_boundary_nodes": ";".join(raw_targets) if bootstrap_after else "",
            "bootstrap_target_levels": ";".join(target_levels) if bootstrap_after else "",
            "bootstrap_input_shape_after_layer": _shape_text(bootstrap_shape) if bootstrap_after and bootstrap_shape else "",
            "bootstrap_input_elements_after_layer": int(bootstrap_elements) if bootstrap_after else 0,
            "bootstrap_count_after_layer": int(edge_boot_count) if bootstrap_after else 0,
            "bootstrap_slots_after_layer": int(module_bootstrap_slots(module)) if bootstrap_after and module is not None else 0,
            "bootstrap_ct_count_after_layer": int(module_bootstrap_ct_count(module)) if bootstrap_after and module is not None else 0,
        }
    return {
        "input_level": int(input_level),
        "bootstrap_count": int(bootstrap_count),
        "bootstrapper_slots": [int(value) for value in bootstrapper_slots],
        "nodes": nodes,
    }


def _rows_for_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    dag = _build_real_unet22_dag(
        height=int(case["height"]),
        width=int(case["width"]),
        in_channels=int(case["input_channels"]),
        out_channels=int(case["output_channels"]),
    )
    original_nodes = [
        str(node)
        for node in dag.topological_sort()
        if str(node) != "x" and dag.nodes[node].get("module") is not None
    ]
    plan = layout_planner.build_layout_policy_compile_plan(
        dag,
        policy=POLICY,
        slots=SLOTS,
        estimator=ESTIMATOR,
    )
    bootstrap_plan = _bootstrap_plan_for_dag(dag, original_nodes)
    summary = dict(plan.get("summary", {}) or {})
    edge_rows_by_target: dict[str, list[dict[str, Any]]] = {}
    for row in plan["edge_layouts"]:
        edge_rows_by_target.setdefault(str(row["target"]), []).append(dict(row))
    node_rows = {str(row["node"]): dict(row) for row in plan.get("node_layouts", [])}

    rows: list[dict[str, Any]] = []
    linear_index = 0
    linear_with_output_index = 0
    compile_index = 0
    for node in original_nodes:
        module = dag.nodes[node].get("module")
        if module is None:
            continue
        compile_index += 1
        op_type = type(module).__name__
        in_rows = edge_rows_by_target.get(str(node), [])
        selected_input = _max_layout(in_rows, "selected_layout")
        source_layout = _max_layout(in_rows, "source_layout")
        required_layout = _max_layout(in_rows, "required_layout")
        node_row = node_rows.get(str(node), {})
        output_layout = dict(node_row.get("selected_layout", {}) or {})
        if not output_layout:
            output_layout = {
                "top_beta": 0,
                "bottom_beta": 0,
                "stride": 1,
                "gap": int(getattr(module, "output_gap", 1)),
                "tile_count": _ct_count(getattr(module, "fhe_output_shape")),
                "stored_slots": int(torch.Size(getattr(module, "fhe_output_shape")).numel()),
            }

        is_output_layer = str(node) == "output"
        is_unet_linear = op_type in {"Conv2d", "ConvTranspose2d"}
        if is_unet_linear:
            linear_with_output_index += 1
            if not is_output_layer:
                linear_index += 1
        params = _module_params(module)
        bootstrap_row = dict(bootstrap_plan["nodes"].get(str(node), {}) or {})
        relayout_side, relayout_note = _fusion_side(in_rows, node_row)
        lt_rot = sum(int(row.get("lt_bsgs_rotation_estimate", 0) or 0) for row in in_rows)
        planner_lt_rot = sum(int(row.get("planner_rotation_cost_estimate", 0) or 0) for row in in_rows)
        relayout_rot = (
            sum(int(row.get("relayout_rotation_estimate", 0) or 0) for row in in_rows)
            + int(node_row.get("relayout_rotation_estimate", 0) or 0)
        )
        relayout_mask = (
            sum(int(row.get("relayout_mask_mult_estimate", 0) or 0) for row in in_rows)
            + int(node_row.get("relayout_mask_mult_estimate", 0) or 0)
        )
        relayout_sparse_lt = sum(int(row.get("relayout_sparse_lt_estimate", 0) or 0) for row in in_rows)
        relayout_depth = (
            sum(int(row.get("relayout_depth_estimate", 0) or 0) for row in in_rows)
            + int(node_row.get("relayout_depth_estimate", 0) or 0)
        )
        producer_rot = int(node_row.get("producer_fused_rotation_estimate", 0) or 0)
        consumer_rot = sum(int(row.get("consumer_fused_rotation_estimate", 0) or 0) for row in in_rows)
        diag_count = sum(int(row.get("lt_ct_pt_mult_estimate", 0) or 0) for row in in_rows)
        activation_ct_mult = sum(int(row.get("activation_ct_mult_estimate", 0) or 0) for row in in_rows)
        total_rot = int(planner_lt_rot + relayout_rot + producer_rot + consumer_rot)
        is_activation = isinstance(module, SiLU)
        if is_activation:
            lt_rot = 0
            planner_lt_rot = 0
            producer_rot = 0
            consumer_rot = 0
            diag_count = 0
            total_rot = int(relayout_rot)
        lt_estimators = sorted(
            {
                str(row.get("lt_estimator", "") or "")
                for row in in_rows
                if str(row.get("lt_estimator", "") or "")
            }
        )
        lt_estimator_text = ";".join(lt_estimators)
        if int(diag_count) == 0 and int(lt_rot) == 0:
            lt_estimator_text = "none_no_linear_transform"
        if is_activation:
            lt_estimator_text = "none_activation_polynomial"
        layout_modes = sorted(
            {
                str(row.get("layout_mode", "") or "")
                for row in in_rows
                if str(row.get("layout_mode", "") or "")
            }
        )
        input_shape = tuple(int(v) for v in getattr(module, "input_shape"))
        output_shape = tuple(int(v) for v in getattr(module, "output_shape"))
        fhe_input_shape = tuple(int(v) for v in getattr(module, "fhe_input_shape"))
        fhe_output_shape = tuple(int(v) for v in getattr(module, "fhe_output_shape"))
        provider_supported, provider_kind, provider_fallback_reason = _provider_static_support(module)
        rows.append(
            {
                "case": str(case["case"]),
                "dataset": str(case["dataset"]),
                "image_hw": f"{int(case['height'])}x{int(case['width'])}",
                "input_c": int(case["input_channels"]),
                "output_c": int(case["output_channels"]),
                "base_dim": int(BASE_DIM),
                "model_variant": "UNet22PlusOutput_real_trace_22_body_layers_plus_1x1_output",
                "compile_source": "OrionTracer+StatsTracker+fit+NetworkDAG+layout_policy+BootstrapSolver",
                "activation": ACTIVATION,
                "activation_degree": int(SILU_DEGREE),
                "activation_nominal_depth_degree7": 3,
                "activation_fit_rule": "Chebyshev.fit from StatsTracker; depth=3 plus 1 if fitted prescale!=1",
                "policy": POLICY,
                "layout_estimator": str(plan.get("layout_estimator", ESTIMATOR)),
                "rotation_estimator": "template_unweighted_diagonal_bsgs",
                "diagonal_estimator": "template_unweighted_diagonal",
                "slot_count": SLOTS,
                "compile_node_index": int(compile_index),
                "unet22_body_linear_layer_index": int(linear_index) if is_unet_linear and not is_output_layer else "",
                "linear_layer_index_including_output": int(linear_with_output_index) if is_unet_linear else "",
                "node": str(node),
                "op_type": op_type,
                "is_output_layer": is_output_layer,
                "provider_static_supported": provider_supported,
                "provider_static_kind": provider_kind,
                "provider_static_fallback_reason": provider_fallback_reason,
                "ct_count": int(output_layout.get("tile_count", _ct_count(fhe_output_shape)) or _ct_count(fhe_output_shape)),
                "input_ct_count_compact": _ct_count(fhe_input_shape),
                "output_ct_count_compact": _ct_count(fhe_output_shape),
                "selected_input_ct_count": int(selected_input.get("tile_count", _ct_count(fhe_input_shape)) or _ct_count(fhe_input_shape)),
                "selected_output_ct_count": int(output_layout.get("tile_count", _ct_count(fhe_output_shape)) or _ct_count(fhe_output_shape)),
                "alpha": int(selected_input.get("top_beta", 0) or 0),
                "beta": int(selected_input.get("bottom_beta", 0) or 0),
                "output_alpha": int(output_layout.get("top_beta", 0) or 0),
                "output_beta": int(output_layout.get("bottom_beta", 0) or 0),
                "assigned_level": bootstrap_row.get("assigned_level", ""),
                "module_depth": bootstrap_row.get("module_depth", ""),
                "output_level_before_bootstrap": bootstrap_row.get("output_level_before_bootstrap", ""),
                "bootstrap_after_layer": bool(bootstrap_row.get("bootstrap_after_layer", False)),
                "bootstrap_reason": str(bootstrap_row.get("bootstrap_reason", "")),
                "bootstrap_before_nodes": str(bootstrap_row.get("bootstrap_before_nodes", "")),
                "bootstrap_boundary_nodes": str(bootstrap_row.get("bootstrap_boundary_nodes", "")),
                "bootstrap_target_levels": str(bootstrap_row.get("bootstrap_target_levels", "")),
                "bootstrap_input_shape_after_layer": str(bootstrap_row.get("bootstrap_input_shape_after_layer", "")),
                "bootstrap_input_elements_after_layer": int(bootstrap_row.get("bootstrap_input_elements_after_layer", 0) or 0),
                "bootstrap_count_after_layer": int(bootstrap_row.get("bootstrap_count_after_layer", 0) or 0),
                "bootstrap_slots_after_layer": int(bootstrap_row.get("bootstrap_slots_after_layer", 0) or 0),
                "bootstrap_ct_count_after_layer": int(bootstrap_row.get("bootstrap_ct_count_after_layer", 0) or 0),
                "rotation": int(total_rot),
                "lt_rotation_estimate": int(lt_rot),
                "planner_rotation_cost_estimate": int(planner_lt_rot),
                "relayout_rotation_estimate": int(relayout_rot),
                "relayout_mask_mult_estimate": int(relayout_mask),
                "relayout_sparse_lt_estimate": int(relayout_sparse_lt),
                "relayout_depth_estimate": int(relayout_depth),
                "producer_fused_rotation_estimate": int(producer_rot),
                "consumer_fused_rotation_estimate": int(consumer_rot),
                "diagonal_count": int(diag_count),
                "activation_ct_mult_estimate": int(activation_ct_mult),
                "lt_transform_count_estimate": 0 if is_activation else sum(int(row.get("lt_transform_count_estimate", 0) or 0) for row in in_rows),
                "lt_bsgs_group_count_estimate": 0 if is_activation else sum(int(row.get("lt_bsgs_group_count_estimate", 0) or 0) for row in in_rows),
                "lt_baby_rotation_estimate": 0 if is_activation else sum(int(row.get("lt_baby_rotation_estimate", 0) or 0) for row in in_rows),
                "lt_giant_rotation_estimate": 0 if is_activation else sum(int(row.get("lt_giant_rotation_estimate", 0) or 0) for row in in_rows),
                "lt_input_cross_rotation_estimate": 0 if is_activation else sum(int(row.get("lt_input_cross_rotation_estimate", 0) or 0) for row in in_rows),
                "lt_local_submatrix_rotation_estimate": 0 if is_activation else sum(int(row.get("lt_local_submatrix_rotation_estimate", 0) or 0) for row in in_rows),
                "lt_output_materialize_rotation_estimate": 0 if is_activation else sum(int(row.get("lt_output_materialize_rotation_estimate", 0) or 0) for row in in_rows),
                "lt_local_program_count_estimate": 0 if is_activation else sum(int(row.get("lt_local_program_count_estimate", 0) or 0) for row in in_rows),
                "lt_recovery_program_count_estimate": 0 if is_activation else sum(int(row.get("lt_recovery_program_count_estimate", 0) or 0) for row in in_rows),
                "lt_rho_hat_per_program_estimate": 0 if is_activation else max((int(row.get("lt_rho_hat_per_program_estimate", 0) or 0) for row in in_rows), default=0),
                "lt_unfused_rotation_estimate": 0 if is_activation else sum(int(row.get("lt_unfused_rotation_estimate", 0) or 0) for row in in_rows),
                "lt_same_input_fusion_savings_estimate": 0 if is_activation else sum(int(row.get("lt_same_input_fusion_savings_estimate", 0) or 0) for row in in_rows),
                "lt_input_channel_multiplier": 0 if is_activation else max((int(row.get("lt_input_channel_multiplier", 0) or 0) for row in in_rows), default=0),
                "lt_output_channel_multiplier": 0 if is_activation else max((int(row.get("lt_output_channel_multiplier", 0) or 0) for row in in_rows), default=0),
                "lt_estimator": lt_estimator_text,
                "layout_mode": ";".join(layout_modes),
                "relayout_fusion_side": relayout_side,
                "relayout_note": relayout_note,
                "input_gap": int(getattr(module, "input_gap", 0) or 0),
                "output_gap": int(getattr(module, "output_gap", 0) or 0),
                "input_shape": _shape_text(input_shape),
                "output_shape": _shape_text(output_shape),
                "fhe_input_shape": _shape_text(fhe_input_shape),
                "fhe_output_shape": _shape_text(fhe_output_shape),
                "clear_input_min": float(getattr(module, "input_min", 0.0)),
                "clear_input_max": float(getattr(module, "input_max", 0.0)),
                "clear_output_min": float(getattr(module, "output_min", 0.0)),
                "clear_output_max": float(getattr(module, "output_max", 0.0)),
                "in_channels_layer": params["in_channels"],
                "out_channels_layer": params["out_channels"],
                "activation_kind_layer": params["activation_kind"],
                "activation_degree_layer": params["activation_degree"],
                "activation_depth_layer": int(getattr(module, "depth", 0) or 0) if is_activation else "",
                "activation_prescale_layer": float(getattr(module, "prescale", 1.0)) if is_activation else "",
                "activation_constant_layer": float(getattr(module, "constant", 0.0)) if is_activation else "",
                "activation_fit_low": float(getattr(module, "low")) if is_activation and hasattr(module, "low") else "",
                "activation_fit_high": float(getattr(module, "high")) if is_activation and hasattr(module, "high") else "",
                "kernel": params["kernel"],
                "stride": params["stride"],
                "padding": params["padding"],
                "output_padding": params["output_padding"],
                "dilation": params["dilation"],
                "groups": params["groups"],
                "source_nodes": ";".join(str(row.get("source", "")) for row in in_rows),
                "source_layout": _json_compact(source_layout),
                "required_layout": _json_compact(required_layout),
                "selected_input_layout": _json_compact(selected_input),
                "selected_output_layout": _json_compact(output_layout),
                "physical_layout": str(node_row.get("physical_layout", "")),
                "source_physical_layout": ";".join(
                    sorted(
                        {
                            str(row.get("source_physical_layout", "") or "")
                            for row in in_rows
                            if str(row.get("source_physical_layout", "") or "")
                        }
                    )
                ),
                "target_physical_layout": ";".join(
                    sorted(
                        {
                            str(row.get("target_physical_layout", "") or "")
                            for row in in_rows
                            if str(row.get("target_physical_layout", "") or "")
                        }
                    )
                ),
                "incoming_edge_count": int(len(in_rows)),
                "edge_relayout_count": sum(int(bool(row.get("relayout", False))) for row in in_rows),
                "output_relayout": bool(node_row.get("output_relayout", False)),
                "consumer_fused_relayout_count": sum(int(bool(row.get("consumer_fused_relayout", False))) for row in in_rows),
                "producer_fused_relayout_count": sum(int(bool(row.get("producer_fused_relayout", False))) for row in in_rows),
                "producer_materialized_halo": bool(node_row.get("producer_materialized_halo", False)),
                "planner_metric_source": str(plan.get("metric_source", "")),
                "case_total_ciphertext_tiles": int(summary.get("total_ciphertext_tiles", 0) or 0),
                "case_stored_slots": int(summary.get("stored_slots", 0) or 0),
                "case_relayout_count": int(summary.get("relayouts", 0) or 0),
                "case_bootstrap_l_eff": int(BOOTSTRAP_L_EFF),
                "case_input_level": int(bootstrap_plan["input_level"]),
                "case_bootstrap_count": int(bootstrap_plan["bootstrap_count"]),
                "case_bootstrapper_slots": ";".join(str(int(value)) for value in bootstrap_plan["bootstrapper_slots"]),
                "case_raw_planner_reported_rotation_estimate": int(summary.get("reported_rotation_estimate", 0) or 0),
                "case_reported_rotation_estimate": int(summary.get("reported_rotation_estimate", 0) or 0),
                "case_lt_bsgs_rotation_estimate": int(summary.get("lt_bsgs_rotation_estimate", 0) or 0),
                "case_planner_rotation_cost_estimate": int(summary.get("planner_rotation_cost_estimate", 0) or 0),
                "case_relayout_rotation_estimate": int(summary.get("relayout_rotation_estimate", 0) or 0),
                "case_relayout_mask_mult_estimate": int(summary.get("relayout_mask_mult_estimate", 0) or 0),
                "case_relayout_depth_estimate": int(summary.get("relayout_depth_estimate", 0) or 0),
                "case_diagonal_count_estimate": int(summary.get("lt_ct_pt_mult_estimate", 0) or 0),
                "case_activation_ct_mult_estimate": int(summary.get("activation_ct_mult_estimate", 0) or 0),
                "case_ct_pt_mult_estimate": int(summary.get("ct_pt_mult_estimate", 0) or 0),
                "case_halo_redundancy_ratio": float(summary.get("halo_redundancy_ratio", 0.0) or 0.0),
                "case_objective": float(summary.get("objective", 0.0) or 0.0),
                "notes": "planner-only metadata compile; real traced DAG with 22 body LT layers plus explicit 1x1 output; SiLU7 fit uses StatsTracker ranges; no dense HE transform generation or FHE forward",
            }
        )
    corrected_case_values = {
        "case_reported_rotation_estimate": int(sum(int(row["rotation"]) for row in rows)),
        "case_lt_bsgs_rotation_estimate": int(sum(int(row["lt_rotation_estimate"]) for row in rows)),
        "case_planner_rotation_cost_estimate": int(sum(int(row["planner_rotation_cost_estimate"]) for row in rows)),
        "case_relayout_rotation_estimate": int(sum(int(row["relayout_rotation_estimate"]) for row in rows)),
        "case_relayout_mask_mult_estimate": int(sum(int(row["relayout_mask_mult_estimate"]) for row in rows)),
        "case_relayout_depth_estimate": int(sum(int(row["relayout_depth_estimate"]) for row in rows)),
        "case_diagonal_count_estimate": int(sum(int(row["diagonal_count"]) for row in rows)),
        "case_activation_ct_mult_estimate": int(sum(int(row["activation_ct_mult_estimate"]) for row in rows)),
    }
    corrected_case_values["case_ct_pt_mult_estimate"] = int(
        corrected_case_values["case_diagonal_count_estimate"]
        + corrected_case_values["case_relayout_mask_mult_estimate"]
        + corrected_case_values["case_activation_ct_mult_estimate"]
    )
    corrected_case_values["case_objective"] = float(
        corrected_case_values["case_reported_rotation_estimate"]
        + corrected_case_values["case_activation_ct_mult_estimate"]
    )
    for row in rows:
        row.update(corrected_case_values)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compile-only U22 operator-plan CSV and optional LaTeX table."
    )
    parser.add_argument("--base-dim", type=int, default=BASE_DIM, help="U-Net base channel width.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults to .tmp/results/unet22_plus_output_dim{base}_real_trace_compile_plan_4cases.csv.",
    )
    parser.add_argument("--out-tex", type=Path, default=None, help="Optional LaTeX table output path.")
    return parser.parse_args()


def main() -> None:
    global BASE_DIM
    args = _parse_args()
    BASE_DIM = int(args.base_dim)
    cases = [
        {
            "case": "192x192_IBSR_BRAIN_2D",
            "dataset": "IBSR BRAIN 2D",
            "height": 192,
            "width": 192,
            "input_channels": 1,
            "output_channels": 4,
        },
        {
            "case": "224x224_HanCo_Hand",
            "dataset": "HanCo Hand Segmentation Dataset",
            "height": 224,
            "width": 224,
            "input_channels": 3,
            "output_channels": 1,
        },
        {
            "case": "384x288_CVC_ClinicDB",
            "dataset": "CVC-ClinicDB",
            "height": 384,
            "width": 288,
            "input_channels": 3,
            "output_channels": 1,
        },
        {
            "case": "384x384_Satellite_cloud",
            "dataset": "Satellite cloud segmentation",
            "height": 384,
            "width": 384,
            "input_channels": 4,
            "output_channels": 1,
        },
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(_rows_for_case(case))
    out = Path(args.out_csv) if args.out_csv is not None else Path(
        f".tmp/results/unet22_plus_output_dim{int(BASE_DIM)}_real_trace_compile_plan_4cases.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.out_tex is not None:
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text(render_operator_plan_table(rows), encoding="utf-8")
    print(out)
    print(f"rows={len(rows)}")
    if args.out_tex is not None:
        print(args.out_tex)


if __name__ == "__main__":
    main()
