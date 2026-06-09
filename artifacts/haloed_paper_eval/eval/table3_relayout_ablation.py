#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common import (
    REPO_ROOT,
    add_common_args,
    clean_int,
    ensure_layout,
    maybe_existing_artifact_root,
    print_outputs,
    read_csv,
    resolve_run_root,
    run_command,
    tex_int,
    write_csv,
    write_manifest,
)


ARTIFACT = "table3"
SOURCE_SCRIPT = "tools/run_u22_224_policy_compile_probe.py"
POLICIES = ("fixed_max_no_share", "always_no_share_producer_fused", "dp_no_share_fold")
LABELS = {
    "fixed_max_no_share": "\\fixedmax",
    "always_no_share_producer_fused": "\\eagerrelayout",
    "always_no_share_producer": "\\eagerrelayout",
    "dp_no_share_fold": "\\halodp",
}


def _layout_beta(layout: dict[str, Any]) -> int:
    return max(clean_int(layout.get("top_beta")), clean_int(layout.get("bottom_beta")))


def _raw_input_node(name: Any) -> bool:
    return str(name) in {"x", "input"}


def _fused_relayout_decisions(row: dict[str, Any]) -> list[dict[str, Any]]:
    compile_plan = row.get("compile_plan", {}) if isinstance(row, dict) else {}
    edges = compile_plan.get("edge_layouts", []) if isinstance(compile_plan, dict) else []
    decisions: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if bool(edge.get("relayout", False)):
            continue
        if _raw_input_node(edge.get("source", "")):
            continue
        if str(edge.get("layout_mode", "")) != "native_halo_stripe":
            continue
        source_beta = _layout_beta(dict(edge.get("source_layout", {}) or {}))
        selected_beta = _layout_beta(dict(edge.get("selected_layout", {}) or {}))
        if selected_beta <= source_beta:
            continue
        decisions.append(
            {
                "edge": str(edge.get("edge") or f"{edge.get('source', '')}->{edge.get('target', '')}"),
                "source": str(edge.get("source", "")),
                "target": str(edge.get("target", "")),
                "source_beta": int(source_beta),
                "selected_beta": int(selected_beta),
                "reason": "native-source fused halo expansion",
            }
        )
    return decisions


def _read_json_rows(raw_json: Path) -> dict[str, dict[str, Any]]:
    if not raw_json.exists():
        raise FileNotFoundError(f"raw JSON compile plan is required to count fused re-layout decisions: {raw_json}")
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    return {str(row.get("policy", "")): row for row in rows if isinstance(row, dict)}


def _build_command(raw_json: Path, raw_csv: Path) -> list[str | Path]:
    return [
        sys.executable,
        REPO_ROOT / SOURCE_SCRIPT,
        "--backend",
        "lattigo",
        "--image-size",
        "224",
        "--base-channels",
        "32",
        "--silu-degree",
        "7",
        "--logn",
        "16",
        "--max-rounds",
        "16",
        "--auto-target",
        "1",
        "--skip-layer-compile",
        "--policies",
        *POLICIES,
        "--out",
        raw_json,
        "--csv",
        raw_csv,
    ]


def _paper_rows(raw_csv: Path, raw_json: Path) -> list[dict[str, Any]]:
    json_rows = _read_json_rows(raw_json)
    rows: list[dict[str, Any]] = []
    for row in read_csv(raw_csv):
        policy = str(row.get("policy", ""))
        if policy not in LABELS:
            continue
        status = str(row.get("status", ""))
        if status != "ok":
            raise RuntimeError(
                f"Table 3 policy {policy!r} did not compile successfully: "
                f"status={status!r}, error={row.get('error', '')!r}"
            )
        json_row = json_rows.get(policy, {})
        fused_relayouts = _fused_relayout_decisions(json_row)
        explicit_relayouts = clean_int(row.get("summary_relayouts"))
        rows.append(
            {
                "policy": policy,
                "label": LABELS[policy],
                "rotations": clean_int(row.get("summary_reported_rotation_estimate")),
                "relayouts": int(explicit_relayouts + len(fused_relayouts)),
                "explicit_relayouts": int(explicit_relayouts),
                "fused_relayouts": int(len(fused_relayouts)),
                "fused_relayout_edges": "; ".join(str(item["edge"]) for item in fused_relayouts),
                "status": status,
                "provider_mode": row.get("provider_mode", ""),
            }
        )
    order = {policy: index for index, policy in enumerate(POLICIES)}
    return sorted(rows, key=lambda row: order.get(str(row["policy"]), 100))


def _render_tex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\smaller",
        "\\renewcommand{\\arraystretch}{1}",
        "\\caption{Comparison of re-layout policies on the $224\\times224$ \\unet.}",
        "\\label{tab:relayout_ablation}",
        "\\begin{tabular}{lrr}",
        "\\hline",
        "\\textbf{Strategy} & \\textbf{\\# Rotations} & \\textbf{\\# Re-layout decisions}\\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(f"{row['label']} & {tex_int(row['rotations'])} & {tex_int(row['relayouts'])} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HaloED paper Table 3 from Orion relayout policy probe.")
    add_common_args(parser)
    args = parser.parse_args()
    run_root = maybe_existing_artifact_root(resolve_run_root(args, ARTIFACT), ARTIFACT) if args.check_existing else resolve_run_root(args, ARTIFACT)
    dirs = ensure_layout(run_root)
    raw_json = dirs["raw"] / "u22_224_policy_compile_probe.json"
    raw_csv = dirs["raw"] / "u22_224_policy_compile_probe.csv"
    command = _build_command(raw_json, raw_csv)
    if not args.check_existing:
        rc = run_command(command, log_path=dirs["logs"] / "table3_relayout_ablation.log", dry_run=bool(args.dry_run))
        if rc != 0:
            return rc
        if args.dry_run:
            return 0
    elif not raw_csv.exists():
        matches = sorted(run_root.rglob("*policy_compile*.csv"))
        if not matches:
            raise FileNotFoundError(f"no policy compile CSV under {run_root}")
        raw_csv = matches[-1]
        candidate_json = raw_csv.with_suffix(".json")
        if candidate_json.exists():
            raw_json = candidate_json

    rows = _paper_rows(raw_csv, raw_json)
    if len(rows) != 3:
        raise SystemExit(f"expected 3 Table 3 policy rows, found {len(rows)} in {raw_csv}")
    paper_csv = dirs["paper"] / "table3_relayout_ablation.csv"
    tex_path = dirs["paper"] / "table3_relayout_ablation.tex"
    write_csv(paper_csv, rows)
    tex_path.write_text(_render_tex(rows), encoding="utf-8")
    outputs = {"raw_csv": str(raw_csv), "paper_csv": str(paper_csv), "table_tex": str(tex_path)}
    write_manifest(
        run_root,
        artifact=ARTIFACT,
        source_script=SOURCE_SCRIPT,
        command=command,
        outputs=outputs,
        measurement=(
            "compile-level 224x224 U22 policy probe; eager row uses producer-fused no-share policy, "
            "not legacy consumer-fused eager; re-layout decisions count explicit materializations plus "
            "native-source fused halo-expansion decisions"
        ),
    )
    print_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
