from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


OPERATOR_ORDER_27 = [
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
    "dec4a",
    "dec4b",
    "up3",
    "dec3a",
    "dec3b",
    "up2",
    "dec2a",
    "dec2b",
    "up1",
    "dec1a",
    "dec1b",
    "output",
]

PLOT_ORDER_WITH_CAT = [
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

CAT_TO_DECODER = {
    "cat4": "dec4a",
    "cat3": "dec3a",
    "cat2": "dec2a",
    "cat1": "dec1a",
}
DECODER_TO_CAT = {decoder: cat for cat, decoder in CAT_TO_DECODER.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build plot-ready UNet22 dim32 trace CSVs from the enriched edge "
            "compile-plan CSV."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".tmp/results/unet22_plus_output_dim32_real_trace_edge_compile_plan_4cases.csv"),
    )
    parser.add_argument(
        "--out-with-cat",
        type=Path,
        default=Path(".tmp/results/unet22_plus_output_dim32_plot_trace_with_cat_4cases.csv"),
    )
    parser.add_argument(
        "--out-27",
        type=Path,
        default=Path(".tmp/results/unet22_plus_output_dim32_plot_trace_27ops_4cases.csv"),
    )
    return parser.parse_args()


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return int(default)
    return int(float(value))


def _layout(row: dict[str, Any], key: str = "selected_output_layout") -> dict[str, Any]:
    raw = str(row.get(key, "") or "")
    return json.loads(raw) if raw else {}


def _shape_width(row: dict[str, Any]) -> int:
    return int(str(row["image_hw"]).split("x")[1])


def _paper_visible_layout(row: dict[str, Any]) -> tuple[int, int, int, int]:
    layout = _layout(row)
    width = _shape_width(row)
    gap = _int(row, "output_gap", 1)
    tile_count = max(1, int(layout.get("tile_count", row.get("ct_count", 1)) or row.get("ct_count", 1)))
    core_slots = int(layout.get("core_slots", 0) or 0)
    if core_slots > 0:
        alpha = math.floor(float(core_slots) / float(tile_count * width * gap))
    else:
        alpha = math.floor(float(_int(row, "slot_count", 32768)) / float(width * gap))
    top_beta = int(layout.get("top_beta", 0) or 0)
    bottom_beta = int(layout.get("bottom_beta", 0) or 0)
    beta = (top_beta + bottom_beta) // 2
    return int(alpha), int(beta), int(top_beta), int(bottom_beta)


def _join_nonempty(values: list[str]) -> str:
    return ";".join(value for value in values if value)


def _activation_fold(row: dict[str, Any], rows_by_node: dict[str, dict[str, Any]]) -> dict[str, Any]:
    node = str(row["node"])
    act = rows_by_node.get(f"{node}_act")
    direct_boot = _int(row, "bootstrap_count_after_layer")
    act_boot = _int(act, "bootstrap_count_after_layer") if act is not None else 0
    folded_from = ""
    if direct_boot > 0:
        boot_count = direct_boot
        boot_ct_count = _int(row, "bootstrap_ct_count_after_layer")
        source_node = node
        source_reason = str(row.get("bootstrap_reason", ""))
    elif act_boot > 0 and act is not None:
        boot_count = act_boot
        boot_ct_count = _int(act, "bootstrap_ct_count_after_layer")
        source_node = str(act["node"])
        source_reason = str(act.get("bootstrap_reason", ""))
        folded_from = str(act["node"])
    else:
        boot_count = 0
        boot_ct_count = 0
        source_node = ""
        source_reason = ""

    rows_to_fold = [row] + ([act] if act is not None else [])
    drop_count = sum(_int(item, "edge_bootstrap_drop_count") for item in rows_to_fold)
    drop_ct_saved = sum(_int(item, "edge_bootstrap_drop_ct_saved") for item in rows_to_fold)
    drop_slots_saved = sum(_int(item, "edge_bootstrap_drop_slots_saved") for item in rows_to_fold)
    drop_detail = _join_nonempty([str(item.get("edge_bootstrap_drop_detail", "")) for item in rows_to_fold])
    act_edge_kind = str(act.get("edge_kind", "")) if act is not None else ""
    act_edge_detail = str(act.get("edge_kind_detail", "")) if act is not None else ""
    return {
        "bootstrap_count": int(boot_count),
        "bootstrap_ct_count": int(boot_ct_count),
        "bootstrap_source_node": source_node,
        "bootstrap_reason_folded": source_reason,
        "folded_activation_node": folded_from,
        "folded_activation_edge_kind": act_edge_kind,
        "folded_activation_edge_detail": act_edge_detail,
        "bootstrap_drop_count": int(drop_count),
        "bootstrap_drop_ct_saved": int(drop_ct_saved),
        "bootstrap_drop_slots_saved": int(drop_slots_saved),
        "bootstrap_drop_detail": drop_detail,
    }


def _skip_merge_fold(row: dict[str, Any], rows_by_node: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cat_node = DECODER_TO_CAT.get(str(row.get("node", "")), "")
    cat = rows_by_node.get(cat_node) if cat_node else None
    if cat is None:
        return {
            "folded_skip_merge_node": "",
            "folded_skip_merge_edge_kind": "",
            "folded_skip_merge_edge_kind_counts": "",
            "folded_skip_merge_edge_detail": "",
            "folded_skip_merge_explicit_count": 0,
            "folded_skip_merge_carry_count": 0,
            "folded_skip_merge_fused_count": 0,
        }
    return {
        "folded_skip_merge_node": str(cat["node"]),
        "folded_skip_merge_edge_kind": str(cat.get("edge_kind", "")),
        "folded_skip_merge_edge_kind_counts": str(cat.get("edge_kind_counts", "")),
        "folded_skip_merge_edge_detail": str(cat.get("edge_kind_detail", "")),
        "folded_skip_merge_explicit_count": _int(cat, "edge_explicit_count"),
        "folded_skip_merge_carry_count": _int(cat, "edge_carry_count"),
        "folded_skip_merge_fused_count": _int(cat, "edge_fused_count"),
    }


def _plot_edge_label(row: dict[str, Any], skip_fold: dict[str, Any]) -> str:
    pieces: list[str] = []
    if skip_fold.get("folded_skip_merge_node"):
        pieces.append(f"{skip_fold['folded_skip_merge_node']}:{skip_fold['folded_skip_merge_edge_kind']}")
    pieces.append(f"{row['node']}:{row.get('edge_kind', '')}")
    return " | ".join(piece for piece in pieces if piece)


def _base_plot_row(
    row: dict[str, Any],
    *,
    plot_index: int,
    rows_by_node: dict[str, dict[str, Any]],
    include_skip_fold: bool,
) -> dict[str, Any]:
    alpha, beta, top_beta, bottom_beta = _paper_visible_layout(row)
    activation = _activation_fold(row, rows_by_node)
    skip = _skip_merge_fold(row, rows_by_node) if include_skip_fold else _skip_merge_fold({}, {})
    node = str(row["node"])
    row_kind = "skip_merge" if node.startswith("cat") else "operator"
    operator_index = OPERATOR_ORDER_27.index(node) + 1 if node in OPERATOR_ORDER_27 else ""
    return {
        "case": str(row["case"]),
        "dataset": str(row["dataset"]),
        "image_hw": str(row["image_hw"]),
        "input_c": _int(row, "input_c"),
        "output_c": _int(row, "output_c"),
        "base_dim": _int(row, "base_dim"),
        "plot_index": int(plot_index),
        "operator_index_27": operator_index,
        "plot_row_kind": row_kind,
        "node": node,
        "op_type": str(row["op_type"]),
        "source_nodes": str(row.get("source_nodes", "")),
        "alpha_visible": int(alpha),
        "beta_output": int(beta),
        "output_top_beta": int(top_beta),
        "output_bottom_beta": int(bottom_beta),
        "ct": _int(row, "selected_output_ct_count", _int(row, "ct_count")),
        "ct_count_raw": _int(row, "ct_count"),
        "rotation": _int(row, "rotation"),
        "diagonal_count": _int(row, "diagonal_count"),
        **activation,
        "edge_kind": str(row.get("edge_kind", "")),
        "incoming_edge_kind": str(row.get("incoming_edge_kind", "")),
        "edge_kind_counts": str(row.get("edge_kind_counts", "")),
        "edge_kind_detail": str(row.get("edge_kind_detail", "")),
        "edge_carry_count": _int(row, "edge_carry_count"),
        "edge_fused_count": _int(row, "edge_fused_count"),
        "edge_explicit_count": _int(row, "edge_explicit_count"),
        "edge_bootstrap_drop_count_raw": _int(row, "edge_bootstrap_drop_count"),
        "edge_layout_changed_count": _int(row, "edge_layout_changed_count"),
        "edge_carry_halo_count": _int(row, "edge_carry_halo_count"),
        **skip,
        "plot_edge_label": _plot_edge_label(row, skip),
        "output_gap": _int(row, "output_gap"),
        "output_shape": str(row.get("output_shape", "")),
        "fhe_output_shape": str(row.get("fhe_output_shape", "")),
        "selected_output_layout": str(row.get("selected_output_layout", "")),
        "folding_note": (
            "B/CT-saving bootstrap-drop folded from following SiLU when present; "
            "27-op table also folds cat explicit skip alignment onto decoder-a rows"
            if include_skip_fold
            else "B/CT-saving bootstrap-drop folded from following SiLU when present; cat rows kept as skip-merge nodes"
        ),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_with_cat(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    for case in cases:
        rows_by_node = {str(row["node"]): row for row in rows if str(row["case"]) == case}
        for plot_index, node in enumerate(PLOT_ORDER_WITH_CAT, start=1):
            out.append(
                _base_plot_row(
                    rows_by_node[node],
                    plot_index=plot_index,
                    rows_by_node=rows_by_node,
                    include_skip_fold=False,
                )
            )
    return out


def _build_27(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    for case in cases:
        rows_by_node = {str(row["node"]): row for row in rows if str(row["case"]) == case}
        for plot_index, node in enumerate(OPERATOR_ORDER_27, start=1):
            out.append(
                _base_plot_row(
                    rows_by_node[node],
                    plot_index=plot_index,
                    rows_by_node=rows_by_node,
                    include_skip_fold=True,
                )
            )
    return out


def main() -> None:
    args = _parse_args()
    rows = _read_rows(args.input)
    with_cat = _build_with_cat(rows)
    ops27 = _build_27(rows)
    _write_rows(args.out_with_cat, with_cat)
    _write_rows(args.out_27, ops27)
    print(args.out_with_cat)
    print(f"rows={len(with_cat)}")
    print(args.out_27)
    print(f"rows={len(ops27)}")


if __name__ == "__main__":
    main()
