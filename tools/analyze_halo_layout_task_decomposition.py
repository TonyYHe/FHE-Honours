#!/usr/bin/env python3
"""Paper-facing halo-layout LT task decomposition.

This is a descriptor-only analysis. It compares the number of independent
submatrix/local LT programs created by:

  * Orion-style active-only ciphertext partitioning, and
  * HaloED-style height stripes with top/bottom halo rows embedded.

It intentionally does not count BSGS rotations, source-side materialization
sharing, real/imaginary packing, or backend runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools import descriptor_halo_lt_counter as counter


DEFAULT_OUT_DIR = Path(".tmp/results/halo_layout_task_decomposition")


def _shape(case: dict[str, Any], *, prefix: str) -> str:
    return f"{int(case[f'c_{prefix}'])}x{int(case[f'h_{prefix}'])}x{int(case[f'w_{prefix}'])}"


def _speedup(num: int, den: int) -> str:
    if int(den) <= 0:
        return "--"
    return f"{float(num) / float(den):.2f}x"


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    case = dict(row["case"])
    orion = dict(row["orion_global_block_descriptor"])
    halo = dict(row["haloed_local_descriptor"])
    orion_tasks = int(orion["total_lt_tasks"])
    halo_tasks = int(halo["total_lt_tasks"])
    return {
        "case": str(case["name"]),
        "kind": str(case["kind"]),
        "input": _shape(case, prefix="in"),
        "output": _shape(case, prefix="out"),
        "kernel": int(case.get("kernel", 0)),
        "stride": int(case.get("stride", 0)),
        "pad": int(case.get("pad", 0)) if "pad" in case else "",
        "orion_source_blocks": int(orion.get("source_block_count", 0)),
        "orion_target_blocks": int(orion.get("target_block_count", 0)),
        "orion_submatrix_tasks": int(orion_tasks),
        "orion_spatial_boundary_tasks": int(orion.get("spatial_boundary_tasks", 0)),
        "orion_channel_factor_tasks": int(orion.get("channel_pair_factor_tasks", 0)),
        "halo_mode": str(halo.get("mode", "")),
        "halo_height_stripes": int(halo.get("height_stripe_count", 0)),
        "halo_redundant_rows": int(halo.get("halo_redundant_rows", 0)),
        "halo_local_tasks": int(halo_tasks),
        "task_reduction_orion_over_halo": _speedup(orion_tasks, halo_tasks),
        "delta_halo_minus_orion_tasks": int(halo_tasks - orion_tasks),
        "interpretation": str(row.get("interpretation", "")),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case",
        "input",
        "output",
        "orion_submatrix_tasks",
        "orion_spatial_boundary_tasks",
        "halo_height_stripes",
        "halo_redundant_rows",
        "halo_local_tasks",
        "task_reduction_orion_over_halo",
        "interpretation",
    ]
    lines = [
        "# Halo Layout Task Decomposition",
        "",
        "Descriptor-only count. BSGS sharing, real/imaginary packing, and runtime rotations are disabled.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=counter.RING_SLOT_COUNT)
    parser.add_argument("--case", action="append", default=[], help="Restrict to a named descriptor case. May be repeated.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = counter.build_payload(args.case, slots=int(args.slots))
    rows = [_flatten_row(dict(row)) for row in list(payload["rows"])]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "halo_layout_task_decomposition.json").write_text(
        json.dumps({"payload": payload, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "halo_layout_task_decomposition.csv", rows)
    _write_markdown(out_dir / "halo_layout_task_decomposition.md", rows)
    print(out_dir / "halo_layout_task_decomposition.md")
    for row in rows:
        print(
            f"{row['case']}: Orion {row['orion_submatrix_tasks']} -> "
            f"Halo {row['halo_local_tasks']} "
            f"({row['task_reduction_orion_over_halo']})"
        )


if __name__ == "__main__":
    main()
