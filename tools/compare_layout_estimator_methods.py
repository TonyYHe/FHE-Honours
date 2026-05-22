#!/usr/bin/env python3
"""Compare lightweight layout-cost estimators against descriptor oracle rows.

This tool is intentionally offline and read-only. It takes the descriptor table
produced by ``tools/analyze_halo_bsgs_combined_descriptor.py`` and asks which
cheap proxy tracks the shared-source BSGS rotation comparison:

* task-count proxy: only compares the number of submatrix/local programs.
* no-sharing proxy: uses exact unweighted diagonals, but evaluates each program
  independently.
* shared-source BSGS: the descriptor oracle used as the reference column.

The goal is to decide which quantity is safe enough for DP ranking and which
ones should remain offline validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(".tmp/results/rho_hat_real_descriptor_check/halo_bsgs_combined_descriptor.csv")
DEFAULT_OUT_DIR = Path(".tmp/results/layout_estimator_method_compare")


def _parse_gain(value: str) -> float:
    text = str(value).strip()
    if text.endswith("x"):
        text = text[:-1]
    return float(text)


def _safe_int(value: Any) -> int:
    return int(float(str(value).strip()))


def _ratio_error(predicted: float, reference: float) -> dict[str, float]:
    if reference <= 0 or predicted <= 0:
        return {"relative_error_pct": math.inf, "abs_log2_error": math.inf}
    return {
        "relative_error_pct": (float(predicted) / float(reference) - 1.0) * 100.0,
        "abs_log2_error": abs(math.log2(float(predicted) / float(reference))),
    }


def _fmt_float(value: float, digits: int = 2) -> str:
    if math.isinf(float(value)):
        return "inf"
    return f"{float(value):.{digits}f}"


def build_rows(input_csv: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = list(csv.DictReader(input_csv.open(newline="", encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    method_errors: dict[str, list[float]] = {"task_count": [], "no_sharing": []}
    for row in source_rows:
        task_gain = _parse_gain(str(row["task_gain_orion_over_halo"]))
        no_sharing_gain = _parse_gain(str(row["no_sharing_gain_orion_over_halo"]))
        shared_gain = _parse_gain(str(row["gain_common_n1_over_halo"]))
        task_error = _ratio_error(task_gain, shared_gain)
        no_sharing_error = _ratio_error(no_sharing_gain, shared_gain)
        method_errors["task_count"].append(float(task_error["abs_log2_error"]))
        method_errors["no_sharing"].append(float(no_sharing_error["abs_log2_error"]))
        rows.append(
            {
                "case": row["case"],
                "orion_tasks": _safe_int(row["orion_tasks"]),
                "halo_tasks": _safe_int(row["halo_tasks"]),
                "task_count_proxy": float(task_gain),
                "no_sharing_diag_proxy": float(no_sharing_gain),
                "shared_bsgs_reference": float(shared_gain),
                "task_proxy_rel_error_pct": float(task_error["relative_error_pct"]),
                "no_sharing_rel_error_pct": float(no_sharing_error["relative_error_pct"]),
                "task_proxy_abs_log2_error": float(task_error["abs_log2_error"]),
                "no_sharing_abs_log2_error": float(no_sharing_error["abs_log2_error"]),
                "orion_common_n1_rot": _safe_int(row["orion_common_n1_rot"]),
                "halo_embedded_rot": _safe_int(row["halo_embedded_rot"]),
            }
        )

    summary = [
        {
            "method": "task_count",
            "description": "Counts submatrix/local programs only.",
            "mean_abs_log2_error": float(sum(method_errors["task_count"]) / max(1, len(method_errors["task_count"]))),
            "dp_use": "Only as a tie-breaker or fragmentation diagnostic; unsafe as a rotation estimate.",
        },
        {
            "method": "no_sharing_diag",
            "description": "Uses exact unweighted diagonal sets but no same-source BSGS materialization.",
            "mean_abs_log2_error": float(sum(method_errors["no_sharing"]) / max(1, len(method_errors["no_sharing"]))),
            "dp_use": "Better than task count, but misses cases where source-side materialization dominates.",
        },
        {
            "method": "shared_bsgs_reference",
            "description": "Uses exact unweighted diagonal sets and shared source-side BSGS materialization.",
            "mean_abs_log2_error": 0.0,
            "dp_use": "Best offline oracle; too slow to run inside every DP candidate.",
        },
    ]
    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "estimator_method_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = out_dir / "estimator_method_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    md_path = out_dir / "estimator_method_comparison.md"
    lines = [
        "# Layout Estimator Method Comparison",
        "",
        "Reference column: shared-source BSGS rotations from the descriptor oracle.",
        "",
        "## Per-case comparison",
        "",
        "| case | tasks O/H | task proxy | no-sharing diag proxy | shared-BSGS ref. | task err | no-sharing err |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {orion_tasks}/{halo_tasks} | {task:.2f}x | {noshare:.2f}x | {shared:.2f}x | {task_err:+.1f}% | {noshare_err:+.1f}% |".format(
                case=row["case"],
                orion_tasks=row["orion_tasks"],
                halo_tasks=row["halo_tasks"],
                task=row["task_count_proxy"],
                noshare=row["no_sharing_diag_proxy"],
                shared=row["shared_bsgs_reference"],
                task_err=row["task_proxy_rel_error_pct"],
                noshare_err=row["no_sharing_rel_error_pct"],
            )
        )
    lines.extend(
        [
            "",
            "## Method summary",
            "",
            "| method | mean abs log2 error | DP use |",
            "| --- | ---: | --- |",
        ]
    )
    for item in summary:
        lines.append(
            f"| {item['method']} | {_fmt_float(float(item['mean_abs_log2_error']), 3)} | {item['dp_use']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = out_dir / "estimator_method_comparison.json"
    json_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(md_path)
    print(csv_path)
    print(summary_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows, summary = build_rows(Path(args.input))
    write_outputs(rows, summary, Path(args.out_dir))


if __name__ == "__main__":
    main()
