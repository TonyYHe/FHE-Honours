#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.auto_bootstrap import BootstrapSolver
from orion.experimental import layout_policy_ablation as lp
from orion.experimental.u22_phase1 import U22CompileRegistry


POLICIES = ("fixed_max", "always_fused", "dp")
POLICY_TEX = {
    "fixed_max": "\\fixedmax",
    "always_fused": "\\textsc{Always+Fusion}",
    "dp": "\\textit{HaloDP}",
}


class _DummyParams:
    def get_slots(self) -> int:
        return int(lp.DEFAULT_SLOTS)

    def get_debug_status(self) -> bool:
        return False

    def get_io_mode(self) -> str:
        return "none"

    def get_compile_save_resume(self) -> bool:
        return False


def _tex_int(value: int | str) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _network_spec(network: str) -> lp.NetworkSpec:
    normalized = str(network).strip().lower()
    if normalized in {"u22_192_base32", "u22_224_base32", "u22_320_base32"}:
        image_size = int(normalized.split("_")[1])
        return lp.NetworkSpec(
            network=normalized,
            dataset="kvasir_polyp_256",
            image_size=int(image_size),
            input_channels=3,
            base_channels=32,
            provider_mode="u22_256_base32",
        )
    return lp.network_spec(normalized)


def _attach_dummy_scheme(dag: Any) -> None:
    scheme = SimpleNamespace(params=_DummyParams())
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None:
            module.scheme = scheme


def _policy_bootstrap_count(spec: lp.NetworkSpec, policy: str) -> tuple[int, str]:
    dag = lp.build_u22_dag(spec)
    _attach_dummy_scheme(dag)
    registry = U22CompileRegistry.for_dag(
        dag,
        allowed_nodes=None,
        enable_conv_kernels=True,
        layout_policy=str(policy),
    )
    audit = registry.attach_to_dag(dag)
    dag.find_residuals()
    input_level, bootstraps, bootstrapper_slots = BootstrapSolver(
        SimpleNamespace(),
        dag,
        l_eff=len(lp.U22_E2E_LOGQ) - 1,
    ).solve()
    summary = dict(audit.get("graph_audit", {}).get("layout_policy_summary", {}) or {})
    detail = (
        "policy_aware_solver_after_registry_attach;"
        f"input_level={int(input_level)};"
        f"slots={','.join(str(int(value)) for value in bootstrapper_slots)};"
        f"relayout_depth={int(summary.get('relayout_depth_estimate', 0) or 0)};"
        f"fused={int(summary.get('producer_fused_materialization_count', 0) or 0) + int(summary.get('consumer_fused_relayout_count', 0) or 0)}"
    )
    return int(bootstraps), detail


def build_rows(network: str, *, slots: int = lp.DEFAULT_SLOTS) -> list[dict[str, Any]]:
    spec = _network_spec(network)
    dag = lp.build_u22_dag(spec)
    edges = lp.build_edge_infos(dag, slots=int(slots))

    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        plan = lp.plan_policy(dag, edges, policy, slots=int(slots))
        boot_count, boot_detail = _policy_bootstrap_count(spec, str(policy))
        rows.append(
            {
                "network": spec.network,
                "dataset": spec.dataset,
                "image_size": int(spec.image_size),
                "input_channels": int(spec.input_channels),
                "base_channels": int(spec.base_channels),
                "policy": policy,
                "policy_label": plan.policy_label,
                "ct": int(plan.total_ciphertext_tiles),
                "rotation": int(plan.reported_rotation_estimate),
                "ct_pt_mult": int(plan.ct_pt_mult_estimate),
                "relayouts": int(plan.relayouts),
                "relayout_depth": int(plan.relayout_depth_estimate),
                "fused": int(plan.producer_fused_materialization_count + plan.consumer_fused_relayout_count),
                "producer_fused": int(plan.producer_fused_materialization_count),
                "consumer_fused": int(plan.consumer_fused_relayout_count),
                "bootstrap": int(boot_count),
                "bootstrap_detail": boot_detail,
            }
        )
    return rows


def render_latex(rows: list[dict[str, Any]]) -> str:
    network = str(rows[0]["network"])
    image_size = int(rows[0]["image_size"])
    min_by_column = {
        key: min(int(row[key]) for row in rows)
        for key in ("ct", "rotation", "ct_pt_mult", "relayouts", "relayout_depth", "bootstrap")
    }

    def cost_cell(row: dict[str, Any], key: str) -> str:
        value = _tex_int(row[key])
        if int(row[key]) == int(min_by_column[key]):
            return f"\\textbf{{{value}}}"
        return value

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{1}",
        f"\\caption{{Compile-plan re-layout scheduling on the ${image_size}\\times{image_size}$ U\\text{{-}}Net. "
        "\\textit{HaloDP} balances carried halos against fused layout transitions; Boot counts policy-specific "
        "ciphertext bootstraps from the depth-aware solver after applying each layout policy.}",
        "\\label{tab:relayout_ablation}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        "\\textbf{Strategy} & \\textbf{CT} & \\textbf{Rot.} & \\textbf{CT--PT Mult.} & \\textbf{RL} & \\textbf{Depth} & \\textbf{Fused} & \\textbf{Boot} \\\\",
        "\\hline",
    ]
    del network
    for row in rows:
        policy = str(row["policy"])
        lines.append(
            f"{POLICY_TEX[policy]:31s} & {cost_cell(row, 'ct')} & {cost_cell(row, 'rotation')} & "
            f"{cost_cell(row, 'ct_pt_mult')} & {cost_cell(row, 'relayouts')} & "
            f"{cost_cell(row, 'relayout_depth')} & {_tex_int(row['fused'])} & "
            f"{cost_cell(row, 'bootstrap')} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the U22 layout-policy ablation paper table.")
    parser.add_argument("--network", default="u22_320_base32")
    parser.add_argument("--slots", type=int, default=lp.DEFAULT_SLOTS)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO_ROOT / ".tmp" / "results" / "layout_policy_ablation_u22_320_base32_paper.csv",
    )
    parser.add_argument("--out-tex", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_rows(str(args.network), slots=int(args.slots))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out_csv)

    if args.out_tex is not None:
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text(render_latex(rows), encoding="utf-8")
        print(args.out_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
