#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    command_text,
    tex_int,
    write_csv,
    write_manifest,
)


ARTIFACT = "table2"
SOURCE_SCRIPT = "tools/generate_unet22_compile_plan_csv.py"
CASE = "256x256_COVID19_lung"
NODE_ORDER = [
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
    "cat4",
    "dec4a",
    "dec4b",
    "up3",
    "cat3",
    "dec3a",
    "dec3b",
    "up2",
    "cat2",
    "dec2a",
    "dec2b",
    "up1",
    "cat1",
    "dec1a",
    "dec1b",
    "output",
]


def _build_command(raw_csv: Path, raw_tex: Path) -> list[str | Path]:
    return [
        sys.executable,
        REPO_ROOT / SOURCE_SCRIPT,
        "--base-dim",
        "32",
        "--out-csv",
        raw_csv,
        "--out-tex",
        raw_tex,
    ]


def _generate_raw_outputs(raw_csv: Path, raw_tex: Path) -> None:
    from tools import generate_unet22_compile_plan_csv as generator

    generator.BASE_DIM = 32
    case = {
        "case": CASE,
        "dataset": "COVID-19 lung",
        "height": 256,
        "width": 256,
        "input_channels": 1,
        "output_channels": 1,
    }
    rows = generator._rows_for_case(case)  # Paper artifact wrapper around the local Orion generator.
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    raw_tex.parent.mkdir(parents=True, exist_ok=True)
    raw_tex.write_text(generator.render_operator_plan_table(rows), encoding="utf-8")


def _selected_beta(layout_text: str, fallback: int) -> int:
    if not layout_text:
        return int(fallback)
    try:
        layout = json.loads(layout_text)
    except Exception:
        return int(fallback)
    return max(clean_int(layout.get("top_beta", fallback)), clean_int(layout.get("bottom_beta", fallback)))


def _compact_rows(full_csv: Path) -> list[dict[str, Any]]:
    source_rows = [row for row in read_csv(full_csv) if row.get("case") == CASE and row.get("node") in NODE_ORDER]
    by_node = {str(row["node"]): row for row in source_rows}
    rows: list[dict[str, Any]] = []
    for node in NODE_ORDER:
        row = by_node.get(node)
        if not row:
            continue
        beta_in = _selected_beta(str(row.get("selected_input_layout", "")), clean_int(row.get("alpha")))
        beta_out = _selected_beta(str(row.get("selected_output_layout", "")), clean_int(row.get("output_alpha")))
        rows.append(
            {
                "node": node,
                "beta_in": beta_in,
                "beta_out": beta_out,
                "haloed_ct": clean_int(row.get("selected_output_ct_count") or row.get("ct_count")),
                "orion_ct": clean_int(row.get("output_ct_count_compact") or row.get("ct_count")),
                "op_type": row.get("op_type", ""),
                "source_row": clean_int(row.get("compile_node_index")),
            }
        )
    return rows


def _render_tex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{DP-selected halo layouts for $256{\\times}256$ \\unet.}",
        "\\label{tab:e2e-layout-breakdown}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Node} & $\\beta_{\\mathrm{in}}\\!\\rightarrow\\!\\beta_{\\mathrm{out}}$ & \\textbf{\\#CT} & \\textbf{\\Orion \\#CT}\\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"\\texttt{{{row['node']}}} & ${row['beta_in']}\\!\\rightarrow\\!{row['beta_out']}$ & "
            f"{tex_int(row['haloed_ct'])} & {tex_int(row['orion_ct'])}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HaloED paper Table 2 from Orion U22 compile-plan output.")
    add_common_args(parser)
    args = parser.parse_args()

    run_root = maybe_existing_artifact_root(resolve_run_root(args, ARTIFACT), ARTIFACT) if args.check_existing else resolve_run_root(args, ARTIFACT)
    dirs = ensure_layout(run_root)
    raw_csv = dirs["raw"] / "u22_256_base32_compile_plan.csv"
    raw_tex = dirs["raw"] / "u22_256_base32_compile_plan_full.tex"
    command = _build_command(raw_csv, raw_tex)
    if not args.check_existing:
        if args.dry_run:
            print(command_text(command), flush=True)
            return 0
        _generate_raw_outputs(raw_csv, raw_tex)
    elif not raw_csv.exists():
        matches = sorted(run_root.rglob("*compile_plan*.csv"))
        if not matches:
            raise FileNotFoundError(f"no compile-plan CSV under {run_root}")
        raw_csv = matches[-1]

    rows = _compact_rows(raw_csv)
    if not rows:
        raise SystemExit(f"no compact Table 2 rows found in {raw_csv}")
    compact_csv = dirs["paper"] / "table2_dp_layout_breakdown.csv"
    tex_path = dirs["paper"] / "table2_dp_layout_breakdown.tex"
    write_csv(compact_csv, rows)
    tex_path.write_text(_render_tex(rows), encoding="utf-8")
    outputs = {"raw_csv": str(raw_csv), "compact_csv": str(compact_csv), "table_tex": str(tex_path)}
    write_manifest(
        run_root,
        artifact=ARTIFACT,
        source_script=SOURCE_SCRIPT,
        command=command,
        outputs=outputs,
        measurement="planner-only U22 base32 256x256 DP layout compile plan; compact table derived from selected layouts",
    )
    print_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
