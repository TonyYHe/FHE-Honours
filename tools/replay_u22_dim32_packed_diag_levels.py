from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.generate_unet22_compile_plan_csv as compile_plan


DEFAULT_PLAN_CSV = Path(".tmp/results/u22_dim32_4sizes_compile_plan_packed_diags.csv")
DEFAULT_OUT_CSV = Path(".tmp/results/u22_dim32_4sizes_packed_diag_level_replay.csv")
DEFAULT_OUT_JSON = Path(".tmp/results/u22_dim32_4sizes_packed_diag_level_replay.json")

RING_DEGREE = 65536
COEFF_BYTES = 8
SPECIAL_PRIME_COUNT = 3
GIB = 1024**3
TIB = 1024**4

SUMMARY_FIELDS = [
    "input",
    "diagonal_count",
    "encoded_assigned_gib",
    "encoded_assigned_tib",
]

NODE_FIELDS = [
    "input",
    "node",
    "diagonal_count",
    "replay_assigned_level",
    "encoded_assigned_gib",
]


class _DiagonalCountProxy:
    def __init__(self, count: int):
        self.count = int(count)

    def __len__(self) -> int:
        return int(self.count)


def _int_value(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    text = str(value).strip()
    if not text:
        return int(default)
    return int(float(text))


def _float_gib(bytes_count: int) -> float:
    return float(bytes_count) / float(GIB)


def _float_tib(bytes_count: int) -> float:
    return float(bytes_count) / float(TIB)


def _plaintext_bytes_at_level(level: int) -> int:
    coeff_mod_count = int(level) + 1 + int(SPECIAL_PRIME_COUNT)
    return int(RING_DEGREE) * int(COEFF_BYTES) * int(coeff_mod_count)


def _read_plan_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"plan CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"plan CSV has no rows: {path}")
    required = {"case", "image_hw", "input_c", "output_c", "node", "diagonal_count"}
    missing = sorted(required.difference(rows[0].keys()))
    if missing:
        raise ValueError(f"plan CSV missing required columns: {', '.join(missing)}")
    return rows


def _group_by_case(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case"]), []).append(row)
    return grouped


def _case_spec(case_rows: list[dict[str, str]], *, base_dim: int | None) -> dict[str, Any]:
    first = case_rows[0]
    image_hw = str(first["image_hw"])
    if "x" not in image_hw:
        raise ValueError(f"cannot parse image_hw={image_hw!r} for case={first['case']}")
    height_text, width_text = image_hw.split("x", 1)
    return {
        "case": str(first["case"]),
        "dataset": str(first.get("dataset", "")),
        "height": int(height_text),
        "width": int(width_text),
        "input_channels": _int_value(first["input_c"]),
        "output_channels": _int_value(first["output_c"]),
        "base_dim": int(base_dim if base_dim is not None else _int_value(first.get("base_dim"), 32)),
    }


def _inject_diagonal_counts(dag: Any, counts_by_node: dict[str, int]) -> None:
    for node, count in counts_by_node.items():
        if int(count) <= 0 or node not in dag.nodes:
            continue
        module = dag.nodes[node].get("module")
        if module is None:
            continue
        module.diagonals = {(0, 0): _DiagonalCountProxy(int(count))}


def _replay_case(
    case_rows: list[dict[str, str]],
    *,
    base_dim: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = _case_spec(case_rows, base_dim=base_dim)
    compile_plan.BASE_DIM = int(spec["base_dim"])
    dag = compile_plan._build_real_unet22_dag(
        height=int(spec["height"]),
        width=int(spec["width"]),
        in_channels=int(spec["input_channels"]),
        out_channels=int(spec["output_channels"]),
    )

    counts_by_node = {
        str(row["node"]): _int_value(row.get("diagonal_count"), 0)
        for row in case_rows
    }
    _inject_diagonal_counts(dag, counts_by_node)
    original_nodes = [
        str(node)
        for node in dag.topological_sort()
        if str(node) != "x" and dag.nodes[node].get("module") is not None
    ]
    bootstrap_plan = compile_plan._bootstrap_plan_for_dag(dag, original_nodes)

    node_rows: list[dict[str, Any]] = []
    encoded_assigned_bytes = 0
    total_diagonals = 0
    for node in original_nodes:
        diag_count = int(counts_by_node.get(str(node), 0))
        total_diagonals += int(diag_count)
        replay_level = bootstrap_plan["nodes"].get(str(node), {}).get("assigned_level", "")
        replay_level_text = "" if replay_level == "" else str(int(replay_level))

        assigned_bytes = 0
        if diag_count > 0 and replay_level_text:
            assigned_bytes = int(diag_count) * _plaintext_bytes_at_level(int(replay_level_text))
            encoded_assigned_bytes += int(assigned_bytes)

        node_rows.append(
            {
                "input": f"{int(spec['height'])}x{int(spec['width'])}",
                "node": str(node),
                "diagonal_count": int(diag_count),
                "replay_assigned_level": replay_level_text,
                "encoded_assigned_gib": round(_float_gib(int(assigned_bytes)), 6),
            }
        )

    summary = {
        "input": f"{int(spec['height'])}x{int(spec['width'])}",
        "diagonal_count": int(total_diagonals),
        "encoded_assigned_gib": round(_float_gib(encoded_assigned_bytes), 6),
        "encoded_assigned_tib": round(_float_tib(encoded_assigned_bytes), 6),
    }
    return summary, node_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay corrected U22 dim32 packed-diagonal levels and encoded plaintext "
            "memory from a compile-plan CSV without materializing plaintext payloads."
        )
    )
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--node-csv", type=Path, default=None, help="Optional per-node debug CSV output.")
    parser.add_argument(
        "--base-dim",
        type=int,
        default=None,
        help="Override base_dim. Defaults to the base_dim column in the input CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan_csv = Path(args.plan_csv)
    rows = _read_plan_rows(plan_csv)
    grouped = _group_by_case(rows)
    summaries: list[dict[str, Any]] = []
    all_node_rows: list[dict[str, Any]] = []
    for case_rows in grouped.values():
        summary, node_rows = _replay_case(
            case_rows,
            base_dim=args.base_dim,
        )
        summaries.append(summary)
        all_node_rows.extend(node_rows)

    _write_csv(Path(args.out_csv), summaries, SUMMARY_FIELDS)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.node_csv is not None:
        _write_csv(Path(args.node_csv), all_node_rows, NODE_FIELDS)

    print(Path(args.out_csv))
    print(Path(args.out_json))
    if args.node_csv is not None:
        print(Path(args.node_csv))
    for row in summaries:
        print(
            "{input}: D={diagonal_count} encoded_assigned={encoded_assigned_tib:.3f} TiB".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
