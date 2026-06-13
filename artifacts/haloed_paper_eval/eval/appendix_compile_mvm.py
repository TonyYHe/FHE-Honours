#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    add_common_args,
    clean_int,
    clean_number,
    ensure_layout,
    maybe_existing_artifact_root,
    print_outputs,
    read_json,
    resolve_run_root,
    tex_int,
    tex_speedup,
    tex_time,
    write_csv,
    write_manifest,
)


ARTIFACT = "appendix_compile"
SOURCE_SCRIPT = "tools/run_u22_dim32_dense_provider_e2e_matrix.py"
CASES = {
    "192x192": "IBSR",
    "224x224": "HanCo",
    "256x256": "COVID",
    "384x384": "NuSetMS",
}


def _case_json(root: Path, case: str, mode: str) -> Path:
    stem = case.replace("x", "_")
    candidates = [
        root / "raw" / "e2e_matrix" / stem / f"{mode}_e2e.json",
        root / "e2e_matrix" / stem / f"{mode}_e2e.json",
        root / stem / f"{mode}_e2e.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(path for path in root.rglob(f"{mode}_e2e.json") if stem in str(path))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"missing {case} {mode} E2E JSON under {root}")


def _compile_totals(payload: dict[str, Any]) -> dict[str, Any]:
    return (payload.get("operator_breakdown_after_compile") or {}).get("totals") or {}


def _compile_time(payload: dict[str, Any], totals: dict[str, Any]) -> float:
    timing = payload.get("timing_s") or {}
    value = clean_number(timing.get("compile") if isinstance(timing, dict) else None)
    if value > 0.0:
        return value
    value = clean_number(totals.get("compile_s"))
    if value > 0.0:
        return value
    raise ValueError("missing positive compile time in fresh E2E JSON")


def _mvm_count(payload: dict[str, Any], totals: dict[str, Any]) -> int:
    value = clean_int(totals.get("transform_count"))
    if value > 0:
        return value
    # Fresh artifact runs should populate operator_breakdown_after_compile.totals.
    # This fallback only improves the error message when a developer points at an
    # incomplete root; the caller still rejects zero counts below.
    rows = (payload.get("operator_breakdown_after_compile") or {}).get("group_rows") or []
    value = sum(clean_int(row.get("transform_count")) for row in rows if isinstance(row, dict))
    if value > 0:
        return value
    raise ValueError("missing positive BSGS MVM/transform count in fresh E2E JSON")


def _build_time(totals: dict[str, Any]) -> float:
    for key in (
        "executor_build_transform_s",
        "group_backend_generate_s",
        "group_total_s",
    ):
        value = clean_number(totals.get(key))
        if value > 0.0:
            return value
    parts = [
        clean_number(totals.get("executor_prepare_s")),
        clean_number(totals.get("executor_group_compile_s")),
        clean_number(totals.get("group_flatten_s")),
        clean_number(totals.get("group_record_keys_s")),
        clean_number(totals.get("group_save_unload_s")),
    ]
    value = sum(parts)
    if value > 0.0:
        return value
    raise ValueError("missing positive MVM construction time in fresh E2E JSON")


def _extract_case(source_root: Path, case: str) -> dict[str, Any]:
    dense_path = _case_json(source_root, case, "dense")
    provider_path = _case_json(source_root, case, "provider")
    dense = read_json(dense_path)
    provider = read_json(provider_path)
    dense_totals = _compile_totals(dense)
    provider_totals = _compile_totals(provider)
    dense_count = _mvm_count(dense, dense_totals)
    provider_count = _mvm_count(provider, provider_totals)
    dense_build_s = _build_time(dense_totals)
    provider_build_s = _build_time(provider_totals)
    dense_compile_s = _compile_time(dense, dense_totals)
    provider_compile_s = _compile_time(provider, provider_totals)
    return {
        "case": case,
        "dataset": CASES[case],
        "baseline_mvm_count": dense_count,
        "haloed_mvm_count": provider_count,
        "count_reduction_pct": 100.0 * (float(dense_count) - float(provider_count)) / float(dense_count),
        "baseline_mvm_build_s": dense_build_s,
        "haloed_mvm_build_s": provider_build_s,
        "mvm_build_speedup": dense_build_s / provider_build_s if provider_build_s else 0.0,
        "baseline_compile_s": dense_compile_s,
        "haloed_compile_s": provider_compile_s,
        "compile_speedup": dense_compile_s / provider_compile_s if provider_compile_s else 0.0,
        "dense_json": str(dense_path),
        "provider_json": str(provider_path),
    }


def _render_tex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[h]",
        "\\footnotesize",
        "\\caption{Reduction in BSGS-based MVM count and compilation time. B and H denote Baseline and \\system, respectively; Count$\\downarrow$ denotes the percentage reduction in BSGS-based MVM count from Baseline to \\system; Build Spd. reports the speedup of MVM construction time; and Spd. reports the end-to-end compilation speedup.}",
        "\\label{tab:submatrix-reduction}",
        "\\setlength{\\tabcolsep}{.6ex}",
        "\\begin{tabular}{c|rrrr|rrr}",
        "\\hline",
        "\\multirow{2}{*}{\\textbf{Dataset}} &",
        "\\multicolumn{4}{c|}{\\textbf{BSGS-based MVMs}} &",
        "\\multicolumn{3}{c}{\\textbf{Compilation Time (s)}} \\\\",
        "& \\textbf{B} & \\textbf{H} & \\textbf{Count$\\downarrow$} & \\textbf{Build Spd.} &",
        "\\textbf{B} & \\textbf{H} & \\textbf{Spd.} \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {tex_int(row['baseline_mvm_count'])} & {tex_int(row['haloed_mvm_count'])} & "
            f"{clean_number(row['count_reduction_pct']):.1f}\\% & {tex_speedup(row['mvm_build_speedup'])} & "
            f"{tex_time(row['baseline_compile_s'])} & {tex_time(row['haloed_compile_s'])} & "
            f"{tex_speedup(row['compile_speedup'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_md(rows: list[dict[str, Any]], *, source_root: Path) -> str:
    lines = [
        "# Appendix Compile/MVM Table",
        "",
        f"- Source E2E root: `{source_root}`",
        "- This table is derived from the same fresh Table 1 E2E JSONs.",
        "- Historical JSON paths are intentionally not embedded as defaults.",
        "",
        "| Dataset | B MVMs | H MVMs | Count reduction | Build speedup | B compile (s) | H compile (s) | Compile speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {baseline_mvm_count:,} | {haloed_mvm_count:,} | {count_reduction_pct:.1f}% | "
            "{mvm_build_speedup:.2f}x | {baseline_compile_s:.1f} | {haloed_compile_s:.1f} | {compile_speedup:.2f}x |".format(
                **row
            )
        )
    lines.extend(["", "## Raw JSON", "", "| Dataset | Dense JSON | HaloED JSON |", "|---|---|---|"])
    for row in rows:
        lines.append(f"| {row['dataset']} | `{row['dense_json']}` | `{row['provider_json']}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive the appendix BSGS MVM/compile table from fresh Table 1 E2E JSONs."
    )
    add_common_args(parser)
    parser.add_argument(
        "--e2e-root",
        type=Path,
        default=None,
        help="Fresh Table 1 artifact root or raw e2e_matrix root. Required unless --check-existing/--run-root is that root.",
    )
    args = parser.parse_args()

    run_root = maybe_existing_artifact_root(resolve_run_root(args, ARTIFACT), ARTIFACT) if args.check_existing else resolve_run_root(args, ARTIFACT)
    dirs = ensure_layout(run_root)
    source_arg = args.e2e_root or args.check_existing or args.run_root
    if source_arg is None:
        parser.error("--e2e-root is required when no --run-root/--check-existing source is provided")
    source_root = Path(source_arg).resolve()
    if args.dry_run:
        print(f"extract appendix compile/MVM table from fresh E2E root: {source_root}", flush=True)
        return 0

    rows = [_extract_case(source_root, case) for case in CASES]
    csv_path = dirs["paper"] / "appendix_compile_mvm.csv"
    tex_path = dirs["paper"] / "appendix_compile_mvm.tex"
    md_path = dirs["paper"] / "appendix_compile_mvm.md"
    write_csv(csv_path, rows)
    tex_path.write_text(_render_tex(rows), encoding="utf-8")
    md_path.write_text(_render_md(rows, source_root=source_root), encoding="utf-8")
    outputs = {"summary_csv": str(csv_path), "table_tex": str(tex_path), "summary_markdown": str(md_path)}
    write_manifest(
        run_root,
        artifact=ARTIFACT,
        source_script=SOURCE_SCRIPT,
        command=[],
        outputs=outputs,
        measurement="derived from fresh Table 1 E2E operator_breakdown_after_compile totals; no historical JSON defaults",
        extra={"source_root": str(source_root)},
    )
    print_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
