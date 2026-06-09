#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from common import (
    REPO_ROOT,
    add_common_args,
    clean_int,
    clean_number,
    command_text,
    copy_doc_snapshot,
    ensure_layout,
    maybe_existing_artifact_root,
    print_outputs,
    read_json,
    rebuild_local_cpp_libraries,
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


def _build_command(
    raw_root: Path,
    doc_path: Path,
    *,
    cases: list[str],
    modes: list[str],
    compile_only: bool = False,
) -> list[str | Path]:
    if compile_only:
        if len(cases) != 1 or len(modes) != 1:
            raise ValueError("--compile-only expects exactly one --cases value and one --modes value")
        case = str(cases[0])
        mode = str(modes[0])
        stem = case.replace("x", "_")
        return [
            sys.executable,
            REPO_ROOT / SOURCE_SCRIPT,
            "--run-one",
            "--case",
            case,
            "--mode",
            mode,
            "--backend",
            "ckks",
            "--policy",
            "dp_no_share_fold",
            "--run-root",
            raw_root,
            "--out",
            raw_root / stem / f"{mode}_compile_only.json",
            "--seed",
            "0",
            "--forward-runs",
            "1",
            "--warmup-runs",
            "0",
            "--operator-breakdown",
            "--compile-only",
        ]
    return [
        sys.executable,
        REPO_ROOT / SOURCE_SCRIPT,
        "--run-root",
        raw_root,
        "--doc",
        doc_path,
        "--cases",
        *cases,
        "--modes",
        *modes,
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


def _bootstrap_count(payload: dict[str, Any]) -> int:
    for key in ("bootstrap_report_after_forward", "bootstrap_report", "bootstrap_report_after_compile"):
        report = payload.get(key) or {}
        if "count" in report:
            return clean_int(report.get("count"))
    return clean_int(payload.get("bootstrap_count", 0))


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
                "baseline_bootstraps": _bootstrap_count(dense),
                "haloed_time_s": provider_time,
                "haloed_rotations": _total_rotations(provider),
                "haloed_bootstraps": _bootstrap_count(provider),
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


def _md_time(value: Any) -> str:
    return f"{clean_number(value):,.1f}"


def _md_int(value: Any) -> str:
    return f"{clean_int(value):,}"


def _render_markdown(rows: list[dict[str, Any]], *, run_root: Path, raw_root: Path, command: list[str | Path]) -> str:
    worker_keys = [
        "ORION_SINGLE_SLOT_ENCODE_WORKERS",
        "ORION_PACK_CONV_WORKERS",
        "ORION_DIRECT_PACK_WORKERS",
        "ORION_LT_COMPILE_WORKERS",
        "ORION_UNIFIED_COMPILE_WORKERS",
        "ORION_LATTIGO_COMPILE_WORKERS",
        "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS",
        "ORION_LATTIGO_BOOTSTRAP_WORKERS",
    ]
    worker_config = ", ".join(f"{key}={os.environ.get(key, '')}" for key in worker_keys)
    lines = [
        "# HaloED Paper Table 1 E2E U22",
        "",
        "This report is generated from the artifact Table 1 script for the current U22 Orion/HaloED mainline comparison.",
        "",
        "## Run",
        "",
        f"- Artifact run root: `{run_root}`",
        f"- Raw E2E matrix root: `{raw_root}`",
        f"- Source script: `{SOURCE_SCRIPT}`",
        f"- Command: `{command_text(command)}`",
        "",
        "## Measurement",
        "",
        "- Backend: CKKS.",
        "- Modes: dense baseline and HaloED/provider.",
        "- Policy: `dp_no_share_fold`.",
        "- Single-slot layer cache: on by default (`ORION_SINGLE_SLOT_LAYER_CACHE=1`).",
        "- Primary time column: LT/MVM or provider executor group eval + activation + bootstrap + executor/wrapper rescale/accumulate/postprocess.",
        "- Total I/O is excluded from the primary comparison.",
        "- Bootstrap counts use the artifact bootstrap report `count` field, not expanded ciphertext operation counts.",
        "- Orion dense uses the default dense lowering; HaloED/provider uses native provider lowering. Bootstrap layout refinement is disabled (`ORION_BOOTSTRAP_LAYOUT_REFINEMENT=0`), and bootstrap prescale fusion remains enabled (`ORION_DISABLE_BOOTSTRAP_PRESCALE_FUSION=0`).",
        f"- Compile/materialization worker configuration: `{worker_config}`.",
        "",
        "## Summary",
        "",
        "| Case | Dataset | Dense time (s) | Dense rotations | Dense boots | Provider time (s) | Provider rotations | Provider boots | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {dataset} | {dense_time} | {dense_rot} | {dense_boots} | "
            "{provider_time} | {provider_rot} | {provider_boots} | {speedup:.2f}x |".format(
                case=row["case"],
                dataset=row["dataset"],
                dense_time=_md_time(row["baseline_time_s"]),
                dense_rot=_md_int(row["baseline_rotations"]),
                dense_boots=_md_int(row["baseline_bootstraps"]),
                provider_time=_md_time(row["haloed_time_s"]),
                provider_rot=_md_int(row["haloed_rotations"]),
                provider_boots=_md_int(row["haloed_bootstraps"]),
                speedup=clean_number(row["speedup"]),
            )
        )

    lines.extend(
        [
            "",
            "## Raw Results",
            "",
            "| Case | Dense JSON | Provider JSON |",
            "|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(f"| {row['case']} | `{row['dense_json']}` | `{row['provider_json']}` |")
    lines.append("")
    return "\n".join(lines)


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


def _compile_only_json(raw_root: Path, case: str, mode: str) -> Path:
    stem = str(case).replace("x", "_")
    return Path(raw_root) / stem / f"{mode}_compile_only.json"


def _compile_only_summary(payload: dict[str, Any], *, case: str, mode: str, raw_json: Path) -> dict[str, Any]:
    report = payload.get("bootstrap_report_after_compile") or {}
    rotation_report = payload.get("rotation_report_after_compile") or {}
    attach_audit = payload.get("attach_audit") or {}
    bootstrap_rows = report.get("rows") or []
    concat_rows = attach_audit.get("layout_policy_concat_runtimes") or []
    return {
        "case": str(case),
        "mode": str(mode),
        "status": str(payload.get("status", "")),
        "step": str(payload.get("step", "")),
        "network": str(payload.get("network", "")),
        "input_shape": payload.get("input_shape"),
        "provider_mode": str(payload.get("provider_mode", "")),
        "input_level": clean_int(payload.get("input_level", 0)),
        "bootstrap_hooks": clean_int(report.get("count", 0)),
        "bootstrap_hook_names": [
            str(row.get("name", row.get("hook", "")))
            for row in bootstrap_rows
            if str(row.get("name", row.get("hook", "")))
        ],
        "bootstrap_by_slots": report.get("by_slots", {}),
        "rotation_eval_count": clean_int(rotation_report.get("total_rotation_eval_count_estimate", 0)),
        "shared_rotation_eval_count": clean_int(rotation_report.get("total_shared_rotation_eval_count", 0)),
        "transform_rotation_key_count": clean_int(rotation_report.get("total_transform_rotation_key_count", 0)),
        "unique_rotation_stats_row_count": clean_int(rotation_report.get("unique_rotation_stats_row_count", 0)),
        "layout_policy_concat_runtime_count": clean_int(
            attach_audit.get("layout_policy_concat_runtime_count", 0)
        ),
        "layout_policy_concat_runtime_nodes": [
            str(row.get("node", ""))
            for row in concat_rows
            if str(row.get("node", ""))
        ],
        "compile_s": clean_number((payload.get("timing_s") or {}).get("compile", 0.0)),
        "raw_json": str(raw_json),
    }


def _render_compile_only_markdown(summary: dict[str, Any], *, command: list[str | Path]) -> str:
    return "\n".join(
        [
            "# HaloED Paper Table 1 Compile-Only Probe",
            "",
            f"- Command: `{command_text(command)}`",
            f"- Case: `{summary['case']}`",
            f"- Mode: `{summary['mode']}`",
            f"- Status: `{summary['status']}`",
            f"- Network: `{summary['network']}`",
            f"- Input shape: `{summary['input_shape']}`",
            f"- Provider mode: `{summary['provider_mode']}`",
            f"- Input level: `{summary['input_level']}`",
            f"- Bootstrap hooks: `{summary['bootstrap_hooks']}`",
            f"- Bootstrap hook names: `{summary['bootstrap_hook_names']}`",
            f"- Bootstrap by slots: `{summary['bootstrap_by_slots']}`",
            f"- Rotation eval count: `{summary['rotation_eval_count']}`",
            f"- Shared rotation eval count: `{summary['shared_rotation_eval_count']}`",
            f"- Transform rotation key count: `{summary['transform_rotation_key_count']}`",
            f"- Unique rotation stats rows: `{summary['unique_rotation_stats_row_count']}`",
            f"- Layout-policy concat runtime count: `{summary['layout_policy_concat_runtime_count']}`",
            f"- Layout-policy concat runtime nodes: `{summary['layout_policy_concat_runtime_nodes']}`",
            f"- Compile time (s): `{summary['compile_s']:.3f}`",
            f"- Raw JSON: `{summary['raw_json']}`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HaloED paper Table 1 from Orion E2E runs.")
    add_common_args(parser)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--modes", nargs="+", choices=("dense", "provider"), default=["dense", "provider"])
    parser.add_argument("--compile-only", action="store_true", help="Run a single artifact Table 1 compile-only probe.")
    parser.add_argument("--skip-cpp-rebuild", action="store_true", help="Use existing local C++ components.")
    args = parser.parse_args()

    run_root = maybe_existing_artifact_root(resolve_run_root(args, ARTIFACT), ARTIFACT) if args.check_existing else resolve_run_root(args, ARTIFACT)
    dirs = ensure_layout(run_root)
    raw_root = dirs["raw"] / "e2e_matrix"
    doc_path = dirs["raw"] / "u22_orion_streaming_haloed_mainline.snapshot.md"
    cases = [str(case) for case in args.cases]
    modes = [str(mode) for mode in args.modes]
    command = _build_command(raw_root, doc_path, cases=cases, modes=modes, compile_only=bool(args.compile_only))
    if args.force:
        command.append("--force")

    cpp_builds: list[dict[str, str]] = []
    if not args.check_existing:
        if not bool(args.skip_cpp_rebuild):
            cpp_builds = rebuild_local_cpp_libraries(log_dir=dirs["logs"], dry_run=bool(args.dry_run))
        copy_doc_snapshot(doc_path)
        rc = run_command(command, log_path=dirs["logs"] / "table1_e2e_unet.log", dry_run=bool(args.dry_run))
        if rc != 0:
            return rc
        if args.dry_run:
            return 0

    if bool(args.compile_only):
        raw_json = _compile_only_json(raw_root, cases[0], modes[0])
        payload = read_json(raw_json)
        summary = _compile_only_summary(payload, case=cases[0], mode=modes[0], raw_json=raw_json)
        csv_path = dirs["paper"] / "table1_compile_only_probe.csv"
        md_path = dirs["paper"] / "table1_compile_only_probe.md"
        write_csv(csv_path, [summary])
        md_path.write_text(_render_compile_only_markdown(summary, command=command), encoding="utf-8")
        outputs = {
            "compile_only_csv": str(csv_path),
            "compile_only_markdown": str(md_path),
            "compile_only_json": str(raw_json),
        }
        write_manifest(
            run_root,
            artifact=ARTIFACT,
            source_script=SOURCE_SCRIPT,
            command=command,
            outputs=outputs,
            measurement="Artifact Table 1 single-case compile-only probe; no forward pass.",
            extra={"cpp_library_builds": cpp_builds, "compile_only_summary": summary},
        )
        print_outputs(outputs)
        return 0

    rows = _summary_rows(raw_root)
    csv_path = dirs["paper"] / "table1_e2e_unet.csv"
    tex_path = dirs["paper"] / "table1_e2e_unet.tex"
    md_path = dirs["paper"] / "table1_e2e_unet.md"
    docs_md_path = REPO_ROOT / "docs" / "haloed_paper_eval_table1_e2e_unet.md"
    write_csv(csv_path, rows)
    tex_path.write_text(_render_tex(rows), encoding="utf-8")
    markdown = _render_markdown(rows, run_root=run_root, raw_root=raw_root, command=command)
    md_path.write_text(markdown, encoding="utf-8")
    docs_md_path.parent.mkdir(parents=True, exist_ok=True)
    docs_md_path.write_text(markdown, encoding="utf-8")
    outputs = {
        "summary_csv": str(csv_path),
        "table_tex": str(tex_path),
        "summary_markdown": str(md_path),
        "docs_markdown": str(docs_md_path),
    }
    write_manifest(
        run_root,
        artifact=ARTIFACT,
        source_script=SOURCE_SCRIPT,
        command=command,
        outputs=outputs,
        measurement="E2E compute = MVM + ACT + BOOT + rescale/accumulation; total I/O excluded",
        extra={"cpp_library_builds": cpp_builds},
    )
    print_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
