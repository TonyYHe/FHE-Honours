from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.evaluation_suite import (
    BOOTSTRAP_STUDY_TARGETS,
    REPO_ROOT,
    RunFlags,
    TIGHT_BUDGET_CONFIGS,
    aggregate_results,
    bootstrap_plan_signature,
    bootstrap_study_flag_matrix,
    compare_outputs,
    compile_once,
    current_head,
    default_workloads,
    delete_scheme,
    json_default,
    run_subprocess,
    worktree_dirty,
    write_run_artifacts,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "bootstrap_placement_eval"


def candidate_budget_specs(workload_map) -> Dict[str, List[Any]]:
    budget_labels = [label for label, _ in reversed(TIGHT_BUDGET_CONFIGS)]
    return {
        base_workload: [workload_map[f"{base_workload}_{budget_label}"] for budget_label in budget_labels]
        for base_workload in BOOTSTRAP_STUDY_TARGETS
    }


def compile_probe(spec, *, profiled: bool, seed: int) -> Dict[str, Any]:
    flags = RunFlags(
        enable_global_dependency_analysis=False,
        enable_adaptive_bsgs=False,
        enable_profiled_leveldag=profiled,
    )
    try:
        result = compile_once(
            orion_module=None,
            spec=spec,
            import_root=REPO_ROOT,
            flags=flags,
            seed=seed,
            clear_cache_before_compile=profiled,
        )
    except Exception as exc:
        return {
            "success": False,
            "profiled": profiled,
            "error": str(exc),
        }

    bootstrap_summary = result.get("bootstrap_plan_summary", {})
    payload = {
        "success": True,
        "profiled": profiled,
        "cold_compile_time_s": result["compile_time_s"],
        "bootstrap_plan_summary": bootstrap_summary,
        "bootstrap_plan_signature": bootstrap_plan_signature(bootstrap_summary),
    }
    delete_scheme(result["orion"], result["scheme"])
    return payload


def pair_changes_decision(analytical: Dict[str, Any], profiled: Dict[str, Any]) -> bool:
    if not analytical.get("success") or not profiled.get("success"):
        return False
    analytical_signature = analytical["bootstrap_plan_signature"]
    profiled_signature = profiled["bootstrap_plan_signature"]
    return (
        analytical_signature["planned_bootstrap_count"]
        != profiled_signature["planned_bootstrap_count"]
        or analytical_signature["planned_bootstrap_nodes"]
        != profiled_signature["planned_bootstrap_nodes"]
        or analytical_signature["bootstrap_slots_hist"]
        != profiled_signature["bootstrap_slots_hist"]
    )


def calibrate_budgets(workload_map, *, seed: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {"selected_workloads": [], "workloads": {}}
    for base_workload, specs in candidate_budget_specs(workload_map).items():
        workload_report = {"selected_budget_labels": [], "candidates": []}
        for spec in specs:
            analytical = compile_probe(spec, profiled=False, seed=seed)
            profiled = compile_probe(spec, profiled=True, seed=seed)
            changed = pair_changes_decision(analytical, profiled)
            workload_report["candidates"].append(
                {
                    "workload": spec.name,
                    "budget_label": spec.budget_label,
                    "config_relpath": spec.config_relpath,
                    "analytical": analytical,
                    "profiled": profiled,
                    "placement_changed": changed,
                }
            )
            if changed and len(workload_report["selected_budget_labels"]) < 2:
                workload_report["selected_budget_labels"].append(spec.budget_label)
                report["selected_workloads"].append(spec.name)
        workload_report["evidentiary"] = bool(workload_report["selected_budget_labels"])
        report["workloads"][base_workload] = workload_report
    return report


def rows_evidence(rows: List[Dict[str, Any]]) -> Dict[str, bool]:
    summary = aggregate_results(rows)
    evidence: Dict[str, bool] = {}
    for pair in summary["bootstrap_pairwise"]:
        workload = pair["base_workload"]
        evidence[workload] = evidence.get(workload, False) or bool(pair["placement_changed"])
    return evidence


def write_bootstrap_artifacts(
    *,
    out_dir: Path,
    rows: List[Dict[str, Any]],
    selected_specs: List[Any],
    calibration_report: Dict[str, Any],
    current_commit: str,
    current_dirty: bool,
    repeats: int,
    warmup_runs: int,
    timed_runs: int,
) -> None:
    write_run_artifacts(
        out_dir=out_dir,
        rows=rows,
        selected=selected_specs,
        current_commit=current_commit,
        current_dirty=current_dirty,
        historical_commit=None,
        repeats=repeats,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )

    summary = aggregate_results(rows)
    pairwise = summary["bootstrap_pairwise"]
    evidentiary = rows_evidence(rows)
    for base_workload in BOOTSTRAP_STUDY_TARGETS:
        evidentiary.setdefault(base_workload, False)

    pairwise_json_path = out_dir / "bootstrap_pairwise.json"
    pairwise_json_path.write_text(json.dumps(pairwise, indent=2, default=json_default), encoding="utf-8")

    pairwise_csv_path = out_dir / "bootstrap_pairwise.csv"
    with pairwise_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "base_workload",
            "budget_label",
            "unified_flag",
            "analytical_variant",
            "profiled_variant",
            "compile_overhead_vs_analytical_s",
            "warm_latency_delta_vs_analytical_s",
            "planned_bootstrap_count_delta",
            "planned_bootstrap_nodes_changed",
            "planned_bootstrap_slots_hist_changed",
            "placement_changed",
            "runtime_bootstrap_count_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairwise:
            writer.writerow({key: row.get(key) for key in fieldnames})

    markdown_lines = [
        "# Bootstrap Placement Evaluation",
        "",
        "## Calibration",
        "",
        "| Workload | Selected Budgets | Evidentiary |",
        "| --- | --- | --- |",
    ]
    for base_workload in BOOTSTRAP_STUDY_TARGETS:
        workload_report = calibration_report.get("workloads", {}).get(base_workload, {})
        selected_labels = workload_report.get("selected_budget_labels", [])
        label_str = ", ".join(selected_labels) if selected_labels else "none"
        markdown_lines.append(
            f"| {base_workload} | {label_str} | {'yes' if evidentiary.get(base_workload) else 'no'} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Pairwise Analytical vs Profiled",
            "",
            "| Workload | Budget | U | Compile Δ (s) | Warm Δ (s) | Plan Count Δ | Nodes Changed | Slot Hist Changed | Placement Changed | Runtime Count Δ |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |",
        ]
    )
    for row in pairwise:
        markdown_lines.append(
            "| "
            f"{row['base_workload']} | "
            f"{row['budget_label']} | "
            f"{row['unified_flag']} | "
            f"{row['compile_overhead_vs_analytical_s']:.4f} | "
            f"{row['warm_latency_delta_vs_analytical_s']:.4f} | "
            f"{row['planned_bootstrap_count_delta']} | "
            f"{'yes' if row['planned_bootstrap_nodes_changed'] else 'no'} | "
            f"{'yes' if row['planned_bootstrap_slots_hist_changed'] else 'no'} | "
            f"{'yes' if row['placement_changed'] else 'no'} | "
            f"{row['runtime_bootstrap_count_delta']} |"
        )

    markdown_path = out_dir / "bootstrap_pairwise.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    calibration_payload = dict(calibration_report)
    calibration_payload["evidentiary_by_workload"] = evidentiary
    calibration_payload["non_evidentiary_workloads"] = [
        workload for workload, changed in sorted(evidentiary.items()) if not changed
    ]
    calibration_path = out_dir / "calibration.json"
    calibration_path.write_text(
        json.dumps(calibration_payload, indent=2, default=json_default),
        encoding="utf-8",
    )


def run_main_matrix(
    *,
    out_dir: Path,
    selected_specs: List[Any],
    repeats: int,
    warmup_runs: int,
    timed_runs: int,
    seed: int,
    calibration_report: Dict[str, Any],
) -> None:
    rows: List[Dict[str, Any]] = []
    current_commit = current_head(REPO_ROOT)
    current_dirty = worktree_dirty(REPO_ROOT)
    variant_flags = bootstrap_study_flag_matrix()

    for spec in selected_specs:
        for repeat_index in range(repeats):
            repeat_seed = seed + repeat_index
            baseline_flags = variant_flags[0]
            baseline_row = run_subprocess(
                tool_path=REPO_ROOT / "tools" / "evaluation_suite.py",
                import_root=REPO_ROOT,
                source="current",
                spec=spec,
                flags=baseline_flags,
                repeat_index=repeat_index,
                seed=repeat_seed,
                output_dir=out_dir,
                clear_profile_cache_before_run=False,
                warmup_runs=warmup_runs,
                timed_runs=timed_runs,
                current_repo_root=REPO_ROOT,
                commit_sha=current_commit,
                dirty=current_dirty,
            )
            reference_output = np.load(baseline_row["output_path"])
            baseline_row["drift_vs_current_disabled"] = {
                "mae": 0.0,
                "precision_bits": float("inf"),
                "passed": True,
                "tolerance": spec.tolerance,
                "reference_shape": list(reference_output.shape),
                "candidate_shape": list(reference_output.shape),
            }
            rows.append(baseline_row)
            write_bootstrap_artifacts(
                out_dir=out_dir,
                rows=rows,
                selected_specs=selected_specs,
                calibration_report=calibration_report,
                current_commit=current_commit,
                current_dirty=current_dirty,
                repeats=repeats,
                warmup_runs=warmup_runs,
                timed_runs=timed_runs,
            )

            for flags in variant_flags[1:]:
                row = run_subprocess(
                    tool_path=REPO_ROOT / "tools" / "evaluation_suite.py",
                    import_root=REPO_ROOT,
                    source="current",
                    spec=spec,
                    flags=flags,
                    repeat_index=repeat_index,
                    seed=repeat_seed,
                    output_dir=out_dir,
                    clear_profile_cache_before_run=flags.enable_profiled_leveldag,
                    warmup_runs=warmup_runs,
                    timed_runs=timed_runs,
                    current_repo_root=REPO_ROOT,
                    commit_sha=current_commit,
                    dirty=current_dirty,
                )
                row["drift_vs_current_disabled"] = compare_outputs(
                    reference_output=reference_output,
                    candidate_path=Path(row["output_path"]),
                    tolerance=spec.tolerance,
                )
                rows.append(row)
                write_bootstrap_artifacts(
                    out_dir=out_dir,
                    rows=rows,
                    selected_specs=selected_specs,
                    calibration_report=calibration_report,
                    current_commit=current_commit,
                    current_dirty=current_dirty,
                    repeats=repeats,
                    warmup_runs=warmup_runs,
                    timed_runs=timed_runs,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-calibration", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    workload_map = default_workloads()
    if args.skip_calibration:
        calibration_report = {
            "selected_workloads": [
                f"{base_workload}_{budget_label}"
                for base_workload in BOOTSTRAP_STUDY_TARGETS
                for budget_label, _ in reversed(TIGHT_BUDGET_CONFIGS)
            ],
            "workloads": {
                base_workload: {
                    "selected_budget_labels": [label for label, _ in reversed(TIGHT_BUDGET_CONFIGS)],
                    "candidates": [],
                    "evidentiary": False,
                }
                for base_workload in BOOTSTRAP_STUDY_TARGETS
            },
            "skipped": True,
        }
    else:
        calibration_report = calibrate_budgets(workload_map, seed=args.seed)
        calibration_report["skipped"] = False

    selected_names = calibration_report["selected_workloads"]
    if not selected_names:
        print("Calibration found no evidentiary tight-budget workloads; wrote calibration artifacts only.")
        write_bootstrap_artifacts(
            out_dir=out_dir,
            rows=[],
            selected_specs=[],
            calibration_report=calibration_report,
            current_commit=current_head(REPO_ROOT),
            current_dirty=worktree_dirty(REPO_ROOT),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )
        return 0

    selected_specs = [workload_map[name] for name in selected_names]
    run_main_matrix(
        out_dir=out_dir,
        selected_specs=selected_specs,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        seed=args.seed,
        calibration_report=calibration_report,
    )
    print(f"Saved bootstrap placement evaluation to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
