#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import (
    REPO_ROOT,
    add_common_args,
    clean_int,
    clean_number,
    copy_doc_snapshot,
    ensure_layout,
    maybe_existing_artifact_root,
    print_outputs,
    read_json,
    resolve_run_root,
    run_command,
    tex_int,
    tex_speedup,
    tex_time,
    write_csv,
    write_manifest,
)


ARTIFACT = "table1"
SOURCE_SCRIPT = "tools/run_u22_dim32_dense_provider_e2e_matrix.py"
CASES = {
    "192x192": "IBSR ($192\\times192$)",
    "224x224": "HanCo ($224\\times224$)",
    "256x256": "COVID-19 ($256\\times256$)",
    "384x384": "NusetMS ($384\\times384$)",
}


def _build_command(raw_root: Path, doc_path: Path) -> list[str | Path]:
    return [
        sys.executable,
        REPO_ROOT / SOURCE_SCRIPT,
        "--run-root",
        raw_root,
        "--doc",
        doc_path,
        "--cases",
        *CASES.keys(),
        "--modes",
        "dense",
        "provider",
        "--backend",
        "ckks",
        "--policy",
        "dp_no_share_fold",
        "--forward-runs",
        "1",
        "--warmup-runs",
        "0",
        "--operator-breakdown",
        "--keep-going",
    ]


def _case_json(raw_root: Path, case: str, mode: str) -> Path:
    stem = case.replace("x", "_")
    patterns = [
        raw_root / stem / f"{mode}_e2e.json",
        raw_root / stem / f"{mode}_encoder4_e2e.json",
    ]
    for path in patterns:
        if path.exists():
            return path
    matches = sorted((raw_root / stem).glob(f"{mode}*_e2e.json")) if (raw_root / stem).exists() else []
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"missing {case} {mode} E2E JSON under {raw_root}")


def _total_rotations(payload: dict[str, Any]) -> int:
    report = payload.get("rotation_report_after_forward") or {}
    for key in ("total_rotation_eval_count_estimate", "total_transform_rotation_key_count", "rotation_eval_count"):
        if key in report:
            return clean_int(report.get(key))
    return clean_int(payload.get("rotation_eval_count", 0))


def _time_components(payload: dict[str, Any]) -> dict[str, float]:
    totals = ((payload.get("operator_breakdown_after_forward") or {}).get("totals") or {})
    mvm_s = clean_number(totals.get("mvm_kernel_s"))
    provider_group_eval_s = clean_number(totals.get("executor_group_eval_wall_s"))
    if mvm_s <= 0.0 and provider_group_eval_s > 0.0:
        mvm_s = provider_group_eval_s
    linear_postprocess_s = clean_number(
        totals.get("wall_linear_wrapper_postprocess_s") or totals.get("linear_wrapper_postprocess_s")
    )
    if linear_postprocess_s <= 0.0:
        linear_postprocess_s = sum(
            clean_number(totals.get(key))
            for key in (
                "linear_wrapper_rescale_s",
                "linear_wrapper_accumulate_s",
                "linear_wrapper_bias_s",
                "linear_wrapper_output_rotation_s",
            )
        )
    return {
        "mvm_s": mvm_s,
        "activation_s": clean_number(totals.get("activation_s")),
        "bootstrap_s": clean_number(totals.get("bootstrap_s")),
        "executor_rescale_s": clean_number(totals.get("executor_rescale_s")),
        "executor_accumulate_s": clean_number(totals.get("executor_accumulate_s")),
        "lt_runtime_stream_accumulate_s": clean_number(totals.get("lt_runtime_stream_accumulate_s")),
        "linear_wrapper_postprocess_s": linear_postprocess_s,
    }


def _compute_time(payload: dict[str, Any]) -> float:
    parts = _time_components(payload)
    return sum(parts.values())


def _summary_rows(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        dense_path = _case_json(raw_root, case, "dense")
        provider_path = _case_json(raw_root, case, "provider")
        dense = read_json(dense_path)
        provider = read_json(provider_path)
        dense_time = _compute_time(dense)
        provider_time = _compute_time(provider)
        dense_parts = _time_components(dense)
        provider_parts = _time_components(provider)
        rows.append(
            {
                "case": case,
                "dataset": CASES[case],
                "baseline_time_s": dense_time,
                "baseline_rotations": _total_rotations(dense),
                "haloed_time_s": provider_time,
                "haloed_rotations": _total_rotations(provider),
                "speedup": dense_time / provider_time if provider_time else 0.0,
                "baseline_mvm_s": dense_parts["mvm_s"],
                "baseline_activation_s": dense_parts["activation_s"],
                "baseline_bootstrap_s": dense_parts["bootstrap_s"],
                "baseline_rescale_accumulate_s": dense_time
                - dense_parts["mvm_s"]
                - dense_parts["activation_s"]
                - dense_parts["bootstrap_s"],
                "haloed_mvm_s": provider_parts["mvm_s"],
                "haloed_activation_s": provider_parts["activation_s"],
                "haloed_bootstrap_s": provider_parts["bootstrap_s"],
                "haloed_rescale_accumulate_s": provider_time
                - provider_parts["mvm_s"]
                - provider_parts["activation_s"]
                - provider_parts["bootstrap_s"],
                "dense_json": str(dense_path),
                "provider_json": str(provider_path),
            }
        )
    return rows


def _render_tex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[th]",
        "\\centering",
        "\\smaller",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\renewcommand{\\arraystretch}{1}",
        "\\caption{End-to-end \\unet inference under CKKS. Times report MVM, activation, bootstrap, rescale, and accumulation compute, excluding total I/O.}",
        "\\label{tab:unet_times}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\multirow{2}{*}[-0.3ex]{\\textbf{Dataset}} &",
        "\\multicolumn{2}{c}{\\textbf{Baseline}} &",
        "\\multicolumn{2}{c}{\\textbf{\\system}} & \\multirow{2}{*}[-0.3ex]{\\textbf{Speedup}} \\\\",
        "\\cline{2-5}",
        " & \\textbf{Time (s)} & \\textbf{\\# Rot.} &",
        " \\textbf{Time (s)} & \\textbf{\\# Rot.} \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {tex_time(row['baseline_time_s'])} & {tex_int(row['baseline_rotations'])} & "
            f"{tex_time(row['haloed_time_s'])} & {tex_int(row['haloed_rotations'])} & {tex_speedup(row['speedup'])}\\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\label{tab:end-to-end}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HaloED paper Table 1 from Orion E2E runs.")
    add_common_args(parser)
    args = parser.parse_args()

    run_root = maybe_existing_artifact_root(resolve_run_root(args, ARTIFACT), ARTIFACT) if args.check_existing else resolve_run_root(args, ARTIFACT)
    dirs = ensure_layout(run_root)
    raw_root = dirs["raw"] / "e2e_matrix"
    doc_path = dirs["raw"] / "u22_orion_streaming_haloed_mainline.snapshot.md"
    command = _build_command(raw_root, doc_path)
    if args.force:
        command.append("--force")

    if not args.check_existing:
        copy_doc_snapshot(doc_path)
        rc = run_command(command, log_path=dirs["logs"] / "table1_e2e_unet.log", dry_run=bool(args.dry_run))
        if rc != 0:
            return rc
        if args.dry_run:
            return 0

    rows = _summary_rows(raw_root)
    csv_path = dirs["paper"] / "table1_e2e_unet.csv"
    tex_path = dirs["paper"] / "table1_e2e_unet.tex"
    write_csv(csv_path, rows)
    tex_path.write_text(_render_tex(rows), encoding="utf-8")
    outputs = {"summary_csv": str(csv_path), "table_tex": str(tex_path)}
    write_manifest(
        run_root,
        artifact=ARTIFACT,
        source_script=SOURCE_SCRIPT,
        command=command,
        outputs=outputs,
        measurement="E2E compute = MVM + ACT + BOOT + rescale/accumulation; total I/O excluded",
    )
    print_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
