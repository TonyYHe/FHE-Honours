#!/usr/bin/env python3
"""Descriptor-only Orion-vs-Halo Conv task + BSGS analysis.

This script is intentionally not a runtime benchmark. It answers a narrower
question for the paper:

  If halo rows are actually embedded in the local tile, and source-side BSGS
  materialization sharing is then applied, how many callback-equivalent
  rotations remain compared with Orion submatrix tasks?

The Halo side uses an embedded-target coordinate system: active output rows are
placed at the corresponding interior coordinates of the halo tile, so the
source and output tile share the same stripe coordinate system.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from tools import benchmark_node_specific_lattigo_provider_vs_dense as bench
from tools import describe_lt_sharing_oracle as sharing
from tools import descriptor_halo_lt_counter as layout_counter


DEFAULT_OUT_DIR = Path(".tmp/results/halo_bsgs_combined_descriptor")

CASE_MAP = {
    "r34_stage1": ("r34_imgnet", "stage1_same", "r34_imgnet_stage1_same_real_network"),
    "r34_stage2": ("r34_imgnet", "stage2_same", "r34_imgnet_stage2_same_real_network"),
    "r34_stage3": ("r34_imgnet", "stage3_same", "r34_imgnet_stage3_same_real_network"),
    "r34_stage4": ("r34_imgnet", "stage4_same", "r34_imgnet_stage4_same_fixed_capacity_halo"),
}

DESCRIPTOR_ONLY_CASE_MAP = {
    "u22_256_enc1a": "u22_256_base32_enc1a_real_network",
    "u22_256_enc1b": "u22_256_base32_enc1b_real_network",
}

ALL_CASE_KEYS = tuple(sorted(set(CASE_MAP) | set(DESCRIPTOR_ONLY_CASE_MAP)))


def _slot_indices(channel_count: int, height: int, width: int, gap: int) -> torch.Tensor:
    g = max(1, int(gap))
    phases = int(g * g)
    packed_w = int(width) * int(g)
    group_block = int(height) * int(g) * int(packed_w)
    hs = torch.arange(int(height), dtype=torch.int64)[:, None]
    ws = torch.arange(int(width), dtype=torch.int64)[None, :]
    out = torch.empty((int(channel_count), int(height), int(width)), dtype=torch.int64)
    for channel in range(int(channel_count)):
        group = int(channel) // int(phases)
        phase = int(channel) % int(phases)
        phase_h = int(phase) // int(g)
        phase_w = int(phase) % int(g)
        out[int(channel)] = (
            int(group) * int(group_block)
            + (hs * int(g) + int(phase_h)) * int(packed_w)
            + ws * int(g)
            + int(phase_w)
        )
    return out


def _halo_task_diag_indices(
    case: layout_counter.ConvCase,
    stripe: dict[str, int],
    *,
    source_channel_start: int,
    source_channel_end: int,
    target_channel_start: int,
    target_channel_end: int,
    slots: int,
) -> set[int]:
    source_h_start = int(stripe["source_h_start"])
    source_h_end = int(stripe["source_h_end"])
    target_h_start = int(stripe["target_h_start"])
    target_h_end = int(stripe["target_h_end"])
    stored_h = int(source_h_end - source_h_start)
    source_slots = _slot_indices(
        int(source_channel_end - source_channel_start),
        int(stored_h),
        int(case.w_in),
        int(case.gap_in),
    )
    target_slots = _slot_indices(
        int(target_channel_end - target_channel_start),
        int(stored_h),
        int(case.w_out),
        int(case.gap_out),
    )
    shifts: set[int] = set()
    for kh in range(int(case.kernel)):
        for kw in range(int(case.kernel)):
            values: list[torch.Tensor] = []
            for out_h in range(int(target_h_start), int(target_h_end)):
                in_h = int(out_h) * int(case.stride) - int(case.pad) + int(kh)
                if int(in_h) < 0 or int(in_h) >= int(case.h_in):
                    continue
                source_local_h = int(in_h) - int(source_h_start)
                # Embedded-target coordinate: active output row stays at the
                # same local stripe coordinate as its global row.
                target_local_h = int(out_h) - int(source_h_start)
                if (
                    int(source_local_h) < 0
                    or int(source_local_h) >= int(stored_h)
                    or int(target_local_h) < 0
                    or int(target_local_h) >= int(stored_h)
                ):
                    continue
                for out_w in range(int(case.w_out)):
                    in_w = int(out_w) * int(case.stride) - int(case.pad) + int(kw)
                    if int(in_w) < 0 or int(in_w) >= int(case.w_in):
                        continue
                    diff = (
                        source_slots[:, int(source_local_h), int(in_w)][:, None]
                        - target_slots[:, int(target_local_h), int(out_w)][None, :]
                    ).reshape(-1)
                    values.append(diff.remainder(int(slots)))
            if values:
                shifts.update(int(value) for value in torch.unique(torch.cat(values)).tolist())
    shifts.discard(0)
    return shifts


def _halo_embedded_bsgs(
    case: layout_counter.ConvCase,
    *,
    slots: int,
    source_channel_cap: int | None = None,
    target_channel_cap: int | None = None,
) -> dict[str, Any]:
    descriptor = layout_counter.count_halo_conv_tasks(case, slots=int(slots))
    total = 0
    individual_total = 0
    task_count = 0
    rows: list[dict[str, Any]] = []
    diag_cache: dict[tuple[int, ...], set[int]] = {}
    for stripe_index, stripe in enumerate(list(descriptor["stripes"])):
        source_channels_per_tile = int(stripe["source_channels_per_tile"])
        target_channels_per_tile = int(stripe["target_channels_per_tile"])
        if source_channel_cap is not None:
            source_channels_per_tile = max(1, min(int(source_channels_per_tile), int(source_channel_cap)))
        if target_channel_cap is not None:
            target_channels_per_tile = max(1, min(int(target_channels_per_tile), int(target_channel_cap)))
        for source_start in range(0, int(case.c_in), int(source_channels_per_tile)):
            source_end = min(int(case.c_in), int(source_start + source_channels_per_tile))
            entries: list[dict[str, Any]] = []
            for target_start in range(0, int(case.c_out), int(target_channels_per_tile)):
                target_end = min(int(case.c_out), int(target_start + target_channels_per_tile))
                diag_key = (
                    int(stripe["source_h_start"]) == 0,
                    int(stripe["source_h_end"]) == int(case.h_in),
                    int(stripe["source_h_end"]) - int(stripe["source_h_start"]),
                    int(stripe["target_h_start"]) - int(stripe["source_h_start"]),
                    int(stripe["target_h_end"]) - int(stripe["target_h_start"]),
                    int(source_end - source_start),
                    int(target_end - target_start),
                )
                if diag_key not in diag_cache:
                    diag_cache[diag_key] = _halo_task_diag_indices(
                        case,
                        stripe,
                        source_channel_start=int(source_start),
                        source_channel_end=int(source_end),
                        target_channel_start=int(target_start),
                        target_channel_end=int(target_end),
                        slots=int(slots),
                    )
                entries.append(
                    {
                        "diag_indices": set(diag_cache[diag_key]),
                    }
                )
            n1, cost = bench._best_unified_common_n1(entries, slots=int(slots))
            task_count += int(len(entries))
            total += int(cost["actual_rotation_callback_count"])
            individual_total += int(cost["sum_per_transform_rotation_count"])
            rows.append(
                {
                    "stripe_index": int(stripe_index),
                    "source_channel_start": int(source_start),
                    "source_channel_end": int(source_end),
                    "transform_count": int(len(entries)),
                    "n1": int(n1),
                    "shared_rotations": int(cost["actual_rotation_callback_count"]),
                    "individual_rotations": int(cost["sum_per_transform_rotation_count"]),
                    "diag_counts": [int(len(entry["diag_indices"])) for entry in entries],
                }
            )
    return {
        "task_count": int(task_count),
        "height_stripe_count": int(descriptor.get("height_stripe_count", 0)),
        "halo_redundant_rows": int(descriptor.get("halo_redundant_rows", 0)),
        "shared_rotations": int(total),
        "individual_rotations": int(individual_total),
        "groups": rows,
    }


def _orion_bsgs(network: str, case_name: str, *, slots: int) -> dict[str, Any]:
    bench._init_scheme(str(network), backend="lattigo")
    dag, _audit = bench._prepare_dag(str(network), provider=False)
    case = bench._case_by_name(str(network), str(case_name))
    module = dag.nodes[str(case["node"])]["module"]
    module.generate_diagonals(last=False)
    entries = sharing._orion_transform_entries(module)
    log_ratio = bench._dense_bsgs_log_ratio(module)
    n1_by_index = {
        int(index): sharing._best_n1_for_entry(entry, slots=int(slots), log_ratio=int(log_ratio))
        for index, entry in enumerate(entries)
    }
    independent = sharing._sum_independent(entries, slots=int(slots), n1_by_index=n1_by_index)
    same_source_n1, group_count, n1s = sharing._sum_shared_signature(
        entries,
        slots=int(slots),
        n1_by_index=n1_by_index,
    )
    by_col: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        by_col.setdefault(int(entry["col"]), []).append(entry)
    common_n1_total = 0
    common_n1s: list[int] = []
    for _col, group_entries in sorted(by_col.items()):
        n1, cost = bench._best_unified_common_n1(group_entries, slots=int(slots))
        common_n1_total += int(cost["actual_rotation_callback_count"])
        common_n1s.append(int(n1))
    target_cts = max((int(entry["row"]) for entry in entries), default=-1) + 1
    return {
        "task_count": int(len(entries)),
        "source_ct_count": int(len(by_col)),
        "target_ct_count": int(target_cts),
        "per_submatrix_n1s": sorted(set(int(value) for value in n1_by_index.values())),
        "independent_rotations": int(independent),
        "same_source_same_n1_rotations": int(same_source_n1),
        "same_source_same_n1_group_count": int(group_count),
        "same_source_same_n1s": list(n1s),
        "per_source_common_n1_rotations": int(common_n1_total),
        "per_source_common_n1s": sorted(set(int(value) for value in common_n1s)),
    }


def _gap1_same_shape_chunk_diag_indices(
    case: layout_counter.ConvCase,
    *,
    target_chunk: int,
    source_chunk: int,
    slots: int,
) -> set[int]:
    if (
        int(case.gap_in) != 1
        or int(case.gap_out) != 1
        or int(case.stride) != 1
        or int(case.h_in) != int(case.h_out)
        or int(case.w_in) != int(case.w_out)
    ):
        raise ValueError(f"{case.name} is not a gap-1 aligned same-shape Conv")
    plane = int(case.h_in) * int(case.w_in)
    if int(plane) % int(slots) != 0:
        raise ValueError(f"{case.name} plane={plane} is not divisible by slots={slots}")

    t0 = int(target_chunk) * int(slots)
    t1 = min(int(plane), int(t0 + int(slots)))
    s0 = int(source_chunk) * int(slots)
    s1 = min(int(plane), int(s0 + int(slots)))
    shifts: set[int] = set()
    for out_sid in range(int(t0), int(t1)):
        oh = int(out_sid) // int(case.w_out)
        ow = int(out_sid) % int(case.w_out)
        target_local = int(out_sid) - int(t0)
        for kh in range(int(case.kernel)):
            ih = int(oh) - int(case.pad) + int(kh)
            if int(ih) < 0 or int(ih) >= int(case.h_in):
                continue
            for kw in range(int(case.kernel)):
                iw = int(ow) - int(case.pad) + int(kw)
                if int(iw) < 0 or int(iw) >= int(case.w_in):
                    continue
                source_sid = int(ih) * int(case.w_in) + int(iw)
                if int(s0) <= int(source_sid) < int(s1):
                    source_local = int(source_sid) - int(s0)
                    shifts.add(int((int(source_local) - int(target_local)) % int(slots)))
    shifts.discard(0)
    return shifts


def _orion_gap1_same_shape_bsgs(case: layout_counter.ConvCase, *, slots: int) -> dict[str, Any]:
    """Descriptor-only Orion BSGS counts for large gap-1 same-shape Conv."""

    descriptor = layout_counter.count_orion_conv_pairs(case, slots=int(slots))
    plane = int(case.h_in) * int(case.w_in)
    chunks_per_channel = int(plane) // int(slots)
    chunk_diags: dict[tuple[int, int], set[int]] = {}
    for target_chunk in range(int(chunks_per_channel)):
        for source_chunk in range(int(chunks_per_channel)):
            chunk_diags[(int(target_chunk), int(source_chunk))] = _gap1_same_shape_chunk_diag_indices(
                case,
                target_chunk=int(target_chunk),
                source_chunk=int(source_chunk),
                slots=int(slots),
            )

    nonempty_pairs = [
        (int(target_chunk), int(source_chunk))
        for target_chunk in range(int(chunks_per_channel))
        for source_chunk in range(int(chunks_per_channel))
        if chunk_diags[(int(target_chunk), int(source_chunk))]
    ]
    expected = int(descriptor["total_lt_tasks"])
    built = int(case.c_in) * int(case.c_out) * int(len(nonempty_pairs))
    if int(built) != int(expected):
        raise RuntimeError(f"{case.name}: descriptor mismatch, built {built} entries vs expected {expected}")

    pattern_n1: dict[tuple[int, int], int] = {}
    pattern_cost: dict[tuple[int, int], int] = {}
    for pair in nonempty_pairs:
        entry = {"diag_indices": set(chunk_diags[pair])}
        n1, cost = bench._best_unified_common_n1([entry], slots=int(slots))
        pattern_n1[pair] = int(n1)
        pattern_cost[pair] = int(cost["actual_rotation_callback_count"])

    independent = int(case.c_in) * int(case.c_out) * sum(int(pattern_cost[pair]) for pair in nonempty_pairs)

    same_source_n1 = 0
    group_count = 0
    n1s: list[int] = []
    common_n1_total = 0
    common_n1s: list[int] = []
    for source_chunk in range(int(chunks_per_channel)):
        per_source_entries = [
            {"diag_indices": set(chunk_diags[(int(target_chunk), int(source_chunk))])}
            for target_chunk in range(int(chunks_per_channel))
            if chunk_diags[(int(target_chunk), int(source_chunk))]
            for _oc in range(int(case.c_out))
        ]
        if not per_source_entries:
            continue

        grouped_by_n1: dict[int, list[dict[str, Any]]] = {}
        for target_chunk in range(int(chunks_per_channel)):
            pair = (int(target_chunk), int(source_chunk))
            if pair not in pattern_n1:
                continue
            grouped_by_n1.setdefault(int(pattern_n1[pair]), []).extend(
                {"diag_indices": set(chunk_diags[pair])} for _oc in range(int(case.c_out))
            )
        for n1, group_entries in sorted(grouped_by_n1.items()):
            cost = bench._shared_cache_bsgs_group_cost(
                group_entries,
                slots=int(slots),
                n1s=[int(n1)] * int(len(group_entries)),
            )
            same_source_n1 += int(case.c_in) * int(cost["actual_rotation_callback_count"])
            group_count += int(case.c_in)
            n1s.append(int(n1))

        common_n1, common_cost = bench._best_unified_common_n1(per_source_entries, slots=int(slots))
        common_n1_total += int(case.c_in) * int(common_cost["actual_rotation_callback_count"])
        common_n1s.append(int(common_n1))
    return {
        "task_count": int(built),
        "source_ct_count": int(int(case.c_in) * int(chunks_per_channel)),
        "target_ct_count": int(int(case.c_out) * int(chunks_per_channel)),
        "per_submatrix_n1s": sorted(set(int(value) for value in pattern_n1.values())),
        "independent_rotations": int(independent),
        "same_source_same_n1_rotations": int(same_source_n1),
        "same_source_same_n1_group_count": int(group_count),
        "same_source_same_n1s": sorted(set(int(value) for value in n1s)),
        "per_source_common_n1_rotations": int(common_n1_total),
        "per_source_common_n1s": sorted(set(int(value) for value in common_n1s)),
        "descriptor_source": "gap1_same_shape_descriptor",
    }


def _gain(left: int, right: int) -> str:
    return "--" if int(right) <= 0 else f"{float(left) / float(right):.2f}x"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "case",
        "input",
        "output",
        "orion_split",
        "halo_split",
        "orion_tasks",
        "halo_tasks",
        "task_gain_orion_over_halo",
        "orion_independent_rot",
        "halo_independent_rot",
        "no_sharing_gain_orion_over_halo",
        "orion_common_n1_rot",
        "halo_embedded_rot",
        "gain_common_n1_over_halo",
    ]
    lines = [
        "# Halo + BSGS Combined Descriptor",
        "",
        "Descriptor-only count. Real/imaginary packing and runtime execution are disabled.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=ALL_CASE_KEYS, default=[], help="Case key. May be repeated.")
    parser.add_argument("--slots", type=int, default=layout_counter.RING_SLOT_COUNT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(args.case or ("r34_stage1", "r34_stage2", "r34_stage4"))
    conv_cases = {case.name: case for case in layout_counter.default_conv_cases()}
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for key in selected:
        if str(key) in CASE_MAP:
            network, orion_case, halo_case_name = CASE_MAP[str(key)]
            print(f"[halo-bsgs] {key}: computing Orion descriptors", flush=True)
            orion = _orion_bsgs(network, orion_case, slots=int(args.slots))
        else:
            halo_case_name = DESCRIPTOR_ONLY_CASE_MAP[str(key)]
            print(f"[halo-bsgs] {key}: computing descriptor-only Orion descriptors", flush=True)
            orion = _orion_gap1_same_shape_bsgs(conv_cases[halo_case_name], slots=int(args.slots))
        print(f"[halo-bsgs] {key}: computing embedded-halo descriptors", flush=True)
        conv_case = conv_cases[halo_case_name]
        halo = _halo_embedded_bsgs(conv_case, slots=int(args.slots))
        orion_source_cts = int(orion.get("source_ct_count", 0))
        orion_target_cts = int(orion.get("target_ct_count", 0))
        row = {
            "case": str(key),
            "input": f"{int(conv_case.c_in)}x{int(conv_case.h_in)}x{int(conv_case.w_in)}",
            "output": f"{int(conv_case.c_out)}x{int(conv_case.h_out)}x{int(conv_case.w_out)}",
            "orion_split": f"{orion_target_cts}x{orion_source_cts}",
            "halo_split": f"{int(halo['height_stripe_count'])} height stripes",
            "orion_tasks": int(orion["task_count"]),
            "halo_tasks": int(halo["task_count"]),
            "task_gain_orion_over_halo": _gain(int(orion["task_count"]), int(halo["task_count"])),
            "orion_independent_rot": int(orion["independent_rotations"]),
            "halo_independent_rot": int(halo["individual_rotations"]),
            "no_sharing_gain_orion_over_halo": _gain(
                int(orion["independent_rotations"]),
                int(halo["individual_rotations"]),
            ),
            "orion_same_source_n1_rot": int(orion["same_source_same_n1_rotations"]),
            "orion_common_n1_rot": int(orion["per_source_common_n1_rotations"]),
            "halo_embedded_rot": int(halo["shared_rotations"]),
            "gain_common_n1_over_halo": _gain(
                int(orion["per_source_common_n1_rotations"]),
                int(halo["shared_rotations"]),
            ),
            "orion_per_submatrix_n1s": str(orion["per_submatrix_n1s"]),
            "halo_redundant_rows": int(halo["halo_redundant_rows"]),
        }
        rows.append(row)
        details[str(key)] = {"orion": orion, "halo": halo}
        print(
            f"[halo-bsgs] {key}: Orion common-n1 {row['orion_common_n1_rot']} "
            f"vs Halo {row['halo_embedded_rot']} ({row['gain_common_n1_over_halo']})",
            flush=True,
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "halo_bsgs_combined_descriptor.json").write_text(
        json.dumps({"rows": rows, "details": details}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "halo_bsgs_combined_descriptor.csv", rows)
    _write_markdown(out_dir / "halo_bsgs_combined_descriptor.md", rows)
    print(out_dir / "halo_bsgs_combined_descriptor.md")


if __name__ == "__main__":
    main()
