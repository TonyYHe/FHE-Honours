#!/usr/bin/env python3
"""Build conv-only operator-level halo ablation rows.

The older X/A/B/C table made the components look like independent switches.
For Conv operators the cleaner story is the cumulative native path:

  original Orion -> flat+halo -> flat+halo+B -> aligned+halo+B

``flat+halo`` uses the selected halo height stripes, but packs each local stripe
as one continuous packed tensor and cuts it into ring ciphertexts.  This keeps
the top/bottom halo benefit while preserving the old channel-boundary spill.
``aligned+halo+B`` is the native no-RI path: channel tiles are aligned to the
selected local stripe geometry, and BSGS baby rotations are shared per source
tile.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_halo_bsgs_combined_descriptor as halo_desc
from tools import benchmark_node_specific_lattigo_provider_vs_dense as bench
from tools import build_native_halo_c_rows as native_rows
from tools import descriptor_halo_lt_counter as layout_counter


DEFAULT_OUT = Path(".tmp/results/conv_operator_halo_ablation")
DEFAULT_CASES = (
    "r34_stage1",
    "r34_stage2",
    "r34_stage3",
    "u22_256_enc1a",
)
ALL_CASES = (
    *DEFAULT_CASES,
    "u22_256_enc1b",
)

CONV_CASE_MAP = {
    "r34_stage1": "r34_imgnet_stage1_same_real_network",
    "r34_stage2": "r34_imgnet_stage2_same_real_network",
    "r34_stage3": "r34_imgnet_stage3_same_real_network",
    "u22_256_enc1a": "u22_256_base32_enc1a_real_network",
    "u22_256_enc1b": "u22_256_base32_enc1b_real_network",
}

R34_ORION_CASE_MAP = {
    "r34_stage1": ("r34_imgnet", "stage1_same"),
    "r34_stage2": ("r34_imgnet", "stage2_same"),
    "r34_stage3": ("r34_imgnet", "stage3_same"),
}


def _ratio(base: int, value: int) -> str:
    return "--" if int(value) <= 0 else f"{float(base) / float(value):.2f}x"


def _pct(base: int, value: int) -> str:
    return "--" if int(base) <= 0 else f"{(1.0 - float(value) / float(base)) * 100.0:.2f}%"


def _ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def _channel_bases(
    *,
    channel_count: int,
    height: int,
    width: int,
    gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    g = max(1, int(gap))
    phases = int(g * g)
    packed_w = int(width) * int(g)
    group_block = int(height) * int(g) * int(packed_w)
    bases = np.empty((int(channel_count),), dtype=np.int64)
    phase_h = np.empty((int(channel_count),), dtype=np.int64)
    phase_w = np.empty((int(channel_count),), dtype=np.int64)
    for channel in range(int(channel_count)):
        phase = int(channel) % int(phases)
        bases[int(channel)] = int(channel) // int(phases) * int(group_block)
        phase_h[int(channel)] = int(phase) // int(g)
        phase_w[int(channel)] = int(phase) % int(g)
    return bases, phase_h, phase_w, int(packed_w), int(group_block)


def _valid_positions(
    case: layout_counter.ConvCase,
    stripe: dict[str, int],
    *,
    kh: int,
    kw: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_h_start = int(stripe["source_h_start"])
    source_storage_h = int(stripe["source_h_end"]) - int(stripe["source_h_start"])
    target_storage_h = int(stripe.get("target_storage_rows", source_storage_h))
    target_local_h_start = int(stripe.get("target_local_h_start", int(stripe["target_h_start"]) - source_h_start))
    source_local_hs: list[int] = []
    source_ws: list[int] = []
    target_local_hs: list[int] = []
    target_ws: list[int] = []
    for out_h in range(int(stripe["target_h_start"]), int(stripe["target_h_end"])):
        in_h = int(out_h) * int(case.stride) - int(case.pad) + int(kh)
        if int(in_h) < 0 or int(in_h) >= int(case.h_in):
            continue
        source_local_h = int(in_h) - int(source_h_start)
        target_local_h = int(target_local_h_start) + int(out_h) - int(stripe["target_h_start"])
        if (
            int(source_local_h) < 0
            or int(source_local_h) >= int(source_storage_h)
            or int(target_local_h) < 0
            or int(target_local_h) >= int(target_storage_h)
        ):
            continue
        for out_w in range(int(case.w_out)):
            in_w = int(out_w) * int(case.stride) - int(case.pad) + int(kw)
            if int(in_w) < 0 or int(in_w) >= int(case.w_in):
                continue
            source_local_hs.append(int(source_local_h))
            source_ws.append(int(in_w))
            target_local_hs.append(int(target_local_h))
            target_ws.append(int(out_w))
    return (
        np.asarray(source_local_hs, dtype=np.int64),
        np.asarray(source_ws, dtype=np.int64),
        np.asarray(target_local_hs, dtype=np.int64),
        np.asarray(target_ws, dtype=np.int64),
    )


def _flat_halo_ablation(
    case: layout_counter.ConvCase,
    *,
    slots: int,
    stripes: list[dict[str, int]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    sharing_groups: list[list[dict[str, Any]]] = []
    stripe_rows: list[dict[str, Any]] = []
    for stripe_index, stripe in enumerate(stripes):
        source_storage_h = int(stripe["source_h_end"]) - int(stripe["source_h_start"])
        target_storage_h = int(stripe.get("target_storage_rows", source_storage_h))
        source_bases, source_phase_h, source_phase_w, source_packed_w, source_group_block = _channel_bases(
            channel_count=int(case.c_in),
            height=int(source_storage_h),
            width=int(case.w_in),
            gap=int(case.gap_in),
        )
        target_bases, target_phase_h, target_phase_w, target_packed_w, target_group_block = _channel_bases(
            channel_count=int(case.c_out),
            height=int(target_storage_h),
            width=int(case.w_out),
            gap=int(case.gap_out),
        )
        source_block_count = _ceil_div(int(source_bases[-1]) + int(source_group_block), int(slots))
        target_block_count = _ceil_div(int(target_bases[-1]) + int(target_group_block), int(slots))
        pair_diags: dict[tuple[int, int], set[int]] = {}
        pair_support_diags: dict[tuple[int, int], set[int]] = {}

        for kh in range(int(case.kernel)):
            for kw in range(int(case.kernel)):
                source_local_hs, source_ws, target_local_hs, target_ws = _valid_positions(
                    case,
                    stripe,
                    kh=int(kh),
                    kw=int(kw),
                )
                if int(source_local_hs.size) == 0:
                    continue
                for source_channel in range(int(case.c_in)):
                    source_offsets = (
                        (source_local_hs * int(case.gap_in) + int(source_phase_h[int(source_channel)]))
                        * int(source_packed_w)
                        + source_ws * int(case.gap_in)
                        + int(source_phase_w[int(source_channel)])
                    )
                    source_abs = int(source_bases[int(source_channel)]) + source_offsets
                    source_blocks = source_abs // int(slots)
                    source_mods = source_abs % int(slots)
                    for target_channel in range(int(case.c_out)):
                        target_offsets = (
                            (target_local_hs * int(case.gap_out) + int(target_phase_h[int(target_channel)]))
                            * int(target_packed_w)
                            + target_ws * int(case.gap_out)
                            + int(target_phase_w[int(target_channel)])
                        )
                        target_abs = int(target_bases[int(target_channel)]) + target_offsets
                        target_blocks = target_abs // int(slots)
                        target_mods = target_abs % int(slots)
                        # For stride-1 same-shape conv in a fixed local tile the
                        # source-target slot delta is constant over valid pixels.
                        diagonal = int((int(source_mods[0]) - int(target_mods[0])) % int(slots))
                        block_pairs = np.unique(
                            np.stack((target_blocks, source_blocks), axis=1),
                            axis=0,
                        )
                        for target_block, source_block in block_pairs.tolist():
                            key = (int(target_block), int(source_block))
                            pair_support_diags.setdefault(key, set()).add(int(diagonal))
                            if int(diagonal) != 0:
                                pair_diags.setdefault(key, set()).add(int(diagonal))

        matrix: list[list[int]] = []
        support_matrix: list[list[int]] = []
        per_source: dict[int, list[dict[str, Any]]] = {int(index): [] for index in range(int(source_block_count))}
        for target_block in range(int(target_block_count)):
            matrix_row: list[int] = []
            support_row: list[int] = []
            for source_block in range(int(source_block_count)):
                key = (int(target_block), int(source_block))
                diag_indices = set(pair_diags.get(key, set()))
                support_diags = set(pair_support_diags.get(key, set()))
                matrix_row.append(int(len(diag_indices)))
                support_row.append(int(len(support_diags)))
                if diag_indices:
                    entry = {
                        "diag_indices": diag_indices,
                        "stripe_index": int(stripe_index),
                        "target_block": int(target_block),
                        "source_block": int(source_block),
                    }
                    entries.append(entry)
                    per_source[int(source_block)].append(entry)
            matrix.append(matrix_row)
            support_matrix.append(support_row)
        sharing_groups.extend([group for group in per_source.values() if group])
        stripe_rows.append(
            {
                "stripe_index": int(stripe_index),
                "source_h_start": int(stripe["source_h_start"]),
                "source_h_end": int(stripe["source_h_end"]),
                "target_h_start": int(stripe["target_h_start"]),
                "target_h_end": int(stripe["target_h_end"]),
                "source_storage_h": int(source_storage_h),
                "target_storage_h": int(target_storage_h),
                "target_local_h_start": int(
                    stripe.get(
                        "target_local_h_start",
                        int(stripe["target_h_start"]) - int(stripe["source_h_start"]),
                    )
                ),
                "target_local_h_end": int(
                    stripe.get(
                        "target_local_h_end",
                        int(stripe.get("target_local_h_start", int(stripe["target_h_start"]) - int(stripe["source_h_start"])))
                        + int(stripe["target_h_end"])
                        - int(stripe["target_h_start"]),
                    )
                ),
                "source_block_count": int(source_block_count),
                "target_block_count": int(target_block_count),
                "rotation_diag_count_matrix": matrix,
                "support_diag_count_matrix": support_matrix,
            }
        )

    independent = 0
    independent_n1s: list[int] = []
    for entry in entries:
        n1, cost = bench._best_unified_common_n1([entry], slots=int(slots))
        independent_n1s.append(int(n1))
        independent += int(cost["actual_rotation_callback_count"])

    shared = 0
    shared_baby = 0
    shared_giant = 0
    shared_n1s: list[int] = []
    for group in sharing_groups:
        n1, cost = bench._best_unified_common_n1(group, slots=int(slots))
        shared_n1s.append(int(n1))
        shared += int(cost["actual_rotation_callback_count"])
        shared_baby += int(cost["shared_baby_rotation_count"])
        shared_giant += int(cost["giant_rotation_count_total"])

    return {
        "programs": int(len(entries)),
        "b_groups": int(len(sharing_groups)),
        "flat_halo_rotations": int(independent),
        "flat_halo_b_rotations": int(shared),
        "flat_halo_baby_rotations": int(shared_baby),
        "flat_halo_giant_rotations": int(shared_giant),
        "independent_n1s": sorted(set(int(value) for value in independent_n1s)),
        "shared_n1s": sorted(set(int(value) for value in shared_n1s)),
        "program_rotation_diag_counts": [int(len(entry["diag_indices"])) for entry in entries],
        "stripes": stripe_rows,
    }


def _orion_original(case_key: str, conv_case: layout_counter.ConvCase, *, slots: int) -> dict[str, Any]:
    if str(case_key) in R34_ORION_CASE_MAP:
        network, case_name = R34_ORION_CASE_MAP[str(case_key)]
        return halo_desc._orion_bsgs(str(network), str(case_name), slots=int(slots))
    return halo_desc._orion_gap1_same_shape_bsgs(conv_case, slots=int(slots))


def _aligned_native(case: layout_counter.ConvCase, *, slots: int) -> dict[str, Any]:
    candidates = [
        native_rows._evaluate_conv_candidate(case, slots=int(slots), channel_tile=int(channel_tile))
        for channel_tile in native_rows._channel_tile_candidates(case)
    ]
    return min(candidates, key=lambda row: (int(row["C+B"]), int(row["C"]), int(row["programs"])))


def _case_row(case_key: str, *, slots: int) -> dict[str, Any]:
    conv_case = {case.name: case for case in layout_counter.default_conv_cases()}[CONV_CASE_MAP[str(case_key)]]
    orion = _orion_original(str(case_key), conv_case, slots=int(slots))
    aligned = _aligned_native(conv_case, slots=int(slots))
    flat = _flat_halo_ablation(conv_case, slots=int(slots), stripes=list(aligned["stripe_rows"]))
    original = int(orion["independent_rotations"])
    flat_halo = int(flat["flat_halo_rotations"])
    flat_halo_b = int(flat["flat_halo_b_rotations"])
    aligned_halo_b = int(aligned["C+B"])
    return {
        "case": str(case_key),
        "input": f"{int(conv_case.c_in)}x{int(conv_case.h_in)}x{int(conv_case.w_in)}",
        "output": f"{int(conv_case.c_out)}x{int(conv_case.h_out)}x{int(conv_case.w_out)}",
        "kernel": f"{int(conv_case.kernel)}x{int(conv_case.kernel)}",
        "gap": int(conv_case.gap_in),
        "original_orion": original,
        "flat_halo": flat_halo,
        "flat_halo_b": flat_halo_b,
        "aligned_halo_b": aligned_halo_b,
        "halo_saving": int(original - flat_halo),
        "b_saving_after_halo": int(flat_halo - flat_halo_b),
        "align_saving_after_halo_b": int(flat_halo_b - aligned_halo_b),
        "total_saving": int(original - aligned_halo_b),
        "total_gain": _ratio(int(original), int(aligned_halo_b)),
        "total_reduction": _pct(int(original), int(aligned_halo_b)),
        "selected_geometry": {
            "channel_tile": int(aligned["channel_tile"]),
            "source_h": int(aligned["source_h"]),
            "target_h": int(aligned["target_h"]),
            "height_stripes": int(aligned["height_stripes"]),
            "aligned_programs": int(aligned["programs"]),
            "aligned_b_groups": int(aligned["b_groups"]),
            "flat_programs": int(flat["programs"]),
            "flat_b_groups": int(flat["b_groups"]),
        },
        "orion": orion,
        "flat_halo_details": flat,
        "aligned_details": aligned,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case",
        "input",
        "output",
        "kernel",
        "gap",
        "original_orion",
        "flat_halo",
        "flat_halo_b",
        "aligned_halo_b",
        "halo_saving",
        "b_saving_after_halo",
        "align_saving_after_halo_b",
        "total_saving",
        "total_gain",
        "total_reduction",
        "channel_tile",
        "source_h",
        "target_h",
        "height_stripes",
        "flat_programs",
        "flat_b_groups",
        "aligned_programs",
        "aligned_b_groups",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            geometry = dict(row["selected_geometry"])
            writer.writerow(
                {
                    **{field: row[field] for field in fields if field in row},
                    "channel_tile": int(geometry["channel_tile"]),
                    "source_h": int(geometry["source_h"]),
                    "target_h": int(geometry["target_h"]),
                    "height_stripes": int(geometry["height_stripes"]),
                    "flat_programs": int(geometry["flat_programs"]),
                    "flat_b_groups": int(geometry["flat_b_groups"]),
                    "aligned_programs": int(geometry["aligned_programs"]),
                    "aligned_b_groups": int(geometry["aligned_b_groups"]),
                }
            )


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "case",
        "shape",
        "geometry",
        "original Orion",
        "flat+halo",
        "flat+halo+B",
        "aligned+halo+B",
        "gain",
    ]
    lines = [
        "# Conv Operator-Level Halo Ablation",
        "",
        "Descriptor-only BSGS rotation counts. TConv rows are intentionally excluded.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        geometry = row["selected_geometry"]
        geometry_text = (
            f"{geometry['channel_tile']}ch, source H{geometry['source_h']}, "
            f"target H{geometry['target_h']}, {geometry['height_stripes']} stripes"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    f"{row['input']} -> {row['output']}",
                    geometry_text,
                    str(row["original_orion"]),
                    str(row["flat_halo"]),
                    str(row["flat_halo_b"]),
                    str(row["aligned_halo_b"]),
                    str(row["total_gain"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Incremental Savings",
            "",
            "| case | halo saving | B saving after halo | align saving after halo+B | total saving | reduction |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["halo_saving"]),
                    str(row["b_saving_after_halo"]),
                    str(row["align_saving_after_halo_b"]),
                    str(row["total_saving"]),
                    str(row["total_reduction"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Program Counts",
            "",
            "| case | flat+halo programs | flat+halo B groups | aligned programs | aligned B groups |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        geometry = row["selected_geometry"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(geometry["flat_programs"]),
                    str(geometry["flat_b_groups"]),
                    str(geometry["aligned_programs"]),
                    str(geometry["aligned_b_groups"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=ALL_CASES, default=None)
    parser.add_argument("--slots", type=int, default=layout_counter.RING_SLOT_COUNT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for case_key in list(args.case or DEFAULT_CASES):
        print(f"[conv-ablation] {case_key}: computing operator-level rows", flush=True)
        rows.append(_case_row(case_key, slots=int(args.slots)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conv_operator_halo_ablation.json").write_text(
        json.dumps({"status": "ok", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "conv_operator_halo_ablation.csv", rows)
    _write_md(out_dir / "conv_operator_halo_ablation.md", rows)
    print(out_dir / "conv_operator_halo_ablation.md")


if __name__ == "__main__":
    main()
