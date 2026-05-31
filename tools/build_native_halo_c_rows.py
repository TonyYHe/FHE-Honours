#!/usr/bin/env python3
"""Build native C-side XABC rows for the paper table.

The old XABC descriptor mixed the C-side rows with a legacy capacity proxy.  This
script keeps the Orion/X-side rows out of scope and reports only the HaloED
C-side rows under the no-real/imaginary native halo policy:

  * Conv: use the same native halo-stripe plan that the provider executor uses.
    R34 same-shape rows intentionally use the specialized native aligned
    executor plan; generic rows use the native halo-stripe Conv2d executor plan.
    C is the sum of independent per-program BSGS costs.  C+B shares source-side
    BSGS materialized rotations across target-channel programs from the same
    source tile.
  * TConv: read the current provider executor descriptor.  TConv has no halo-row
    stencil reuse; C is local output-placement programs without source sharing,
    and C+B is the provider's same-source BSGS materialization sharing.

This is a descriptor-only analysis.  Runtime execution and real/imaginary lane
packing are excluded, but Conv geometry and rotation counts are provider-plan
derived rather than independently re-optimized here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools import descriptor_halo_lt_counter as layout_counter
from tools import describe_lt_sharing_oracle as sharing

from orion.experimental.cir.native_halo_conv2d import (
    NativeHaloConv2DSpec,
    native_halo_conv2d_plan,
)
from orion.experimental.cir.r34_orion_same_shape import (
    r34_native_aligned_halo_plan,
    r34_same_shape_spec_for_family_label,
)


DEFAULT_OUT = Path(".tmp/results/native_halo_c_rows")
DEFAULT_CASES = (
    "r34_stage1",
    "r34_stage2",
    "u22_256_enc1a",
    "u22_256_up2_tconv",
)

CONV_CASE_MAP = {
    "r34_stage1": "r34_imgnet_stage1_same_real_network",
    "r34_stage2": "r34_imgnet_stage2_same_real_network",
    "u22_256_enc1a": "u22_256_base32_enc1a_real_network",
}

R34_SAME_SHAPE_FAMILY_MAP = {
    "r34_imgnet_stage1_same_real_network": "stage1_same",
    "r34_imgnet_stage2_same_real_network": "stage2_same",
    "r34_imgnet_stage3_same_real_network": "stage3_same",
}

TCONV_CASE_MAP = {
    "u22_256_up2_tconv": ("u22_256_base32", "up2"),
}


def _ceil_div(left: int, right: int) -> int:
    return -(-int(left) // int(right))


def _phase_count(gap: int) -> int:
    return max(1, int(gap) * int(gap))


def _pct(base: int, value: int) -> str:
    return "--" if int(base) <= 0 else f"{(1.0 - float(value) / float(base)) * 100.0:.2f}%"


def _ratio(base: int, value: int) -> str:
    return "--" if int(value) <= 0 else f"{float(base) / float(value):.2f}x"


def _generic_conv_spec(case: layout_counter.ConvCase, *, slots: int) -> NativeHaloConv2DSpec:
    return NativeHaloConv2DSpec(
        family_label=str(case.name),
        c_in=int(case.c_in),
        h_in=int(case.h_in),
        w_in=int(case.w_in),
        c_out=int(case.c_out),
        h_out=int(case.h_out),
        w_out=int(case.w_out),
        gap_in=int(case.gap_in),
        gap_out=int(case.gap_out),
        kernel=int(case.kernel),
        stride=int(case.stride),
        pad=int(case.pad),
        dilation=1,
        groups=1,
        slot_count=int(slots),
    )


def _native_plan_for_case(case: layout_counter.ConvCase, *, slots: int):
    r34_family = R34_SAME_SHAPE_FAMILY_MAP.get(str(case.name))
    if r34_family is not None:
        return r34_native_aligned_halo_plan(r34_same_shape_spec_for_family_label(str(r34_family)))
    return native_halo_conv2d_plan(_generic_conv_spec(case, slots=int(slots)))


def _plan_channel_tile(plan: Any) -> int:
    return int(getattr(plan, "channel_tile", getattr(plan, "source_channel_tile", 0)))


def _plan_stripe_rows(plan: Any) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for stripe in tuple(getattr(plan, "stripes", ())):
        source_start = int(getattr(stripe, "source_h_start"))
        source_end = int(getattr(stripe, "source_h_end"))
        target_start = int(getattr(stripe, "target_h_start"))
        target_end = int(getattr(stripe, "target_h_end"))
        target_local_start = int(getattr(stripe, "target_local_h_start", 0))
        target_local_end = int(
            getattr(
                stripe,
                "target_local_h_end",
                int(target_local_start) + int(target_end - target_start),
            )
        )
        target_storage_rows = int(
            getattr(
                stripe,
                "local_h",
                int(target_local_end) if int(target_local_end) > 0 else int(target_end - target_start),
            )
        )
        rows.append(
            {
                "stripe_index": int(getattr(stripe, "index")),
                "target_h_start": int(target_start),
                "target_h_end": int(target_end),
                "source_h_start": int(source_start),
                "source_h_end": int(source_end),
                "stored_source_rows": int(source_end - source_start),
                "target_rows": int(target_end - target_start),
                "target_storage_rows": int(target_storage_rows),
                "target_local_h_start": int(target_local_start),
                "target_local_h_end": int(target_local_end),
            }
        )
    return rows


def _channel_tile_candidates(case: layout_counter.ConvCase) -> list[int]:
    plan = _native_plan_for_case(case, slots=int(layout_counter.RING_SLOT_COUNT))
    return [int(_plan_channel_tile(plan))]


def _evaluate_conv_candidate(
    case: layout_counter.ConvCase,
    *,
    slots: int,
    channel_tile: int,
) -> dict[str, Any]:
    plan = _native_plan_for_case(case, slots=int(slots))
    selected_channel_tile = _plan_channel_tile(plan)
    if int(channel_tile) != int(selected_channel_tile):
        raise ValueError(
            f"{case.name} native provider plan uses {selected_channel_tile} channels per tile, "
            f"not requested {channel_tile}"
        )
    stripes = _plan_stripe_rows(plan)
    target_group_count = int(
        getattr(plan, "target_channel_group_count", _ceil_div(int(case.c_out), int(selected_channel_tile)))
    )
    group_rows: list[dict[str, Any]] = []
    group_rots = list(int(value) for value in getattr(plan, "group_shared_rotations"))
    group_baby = list(int(value) for value in getattr(plan, "group_baby_rotations"))
    group_giant = list(int(value) for value in getattr(plan, "group_giant_rotations"))
    group_n1s = list(int(value) for value in getattr(plan, "group_n1s"))
    group_index = 0
    for stripe_index, _stripe in enumerate(stripes):
        for source_group in range(int(getattr(plan, "source_channel_group_count"))):
            source_start = int(source_group) * int(selected_channel_tile)
            source_end = min(int(case.c_in), int(source_start) + int(selected_channel_tile))
            start_program = int(group_index) * int(target_group_count)
            stop_program = int(start_program) + int(target_group_count)
            group_rows.append(
                {
                    "stripe_index": int(stripe_index),
                    "source_channel_start": int(source_start),
                    "source_channel_end": int(source_end),
                    "transform_count": int(target_group_count),
                    "n1": int(group_n1s[int(group_index)]),
                    "shared_rotations": int(group_rots[int(group_index)]),
                    "baby_rotations": int(group_baby[int(group_index)]),
                    "giant_rotations": int(group_giant[int(group_index)]),
                    "individual_rotations": int(
                        sum(int(value) for value in tuple(getattr(plan, "program_rotation_counts"))[start_program:stop_program])
                    ),
                }
            )
            group_index += 1

    max_source_h = max((int(row["stored_source_rows"]) for row in stripes), default=0)
    max_target_storage_h = max((int(row["target_storage_rows"]) for row in stripes), default=0)

    return {
        "channel_tile": int(selected_channel_tile),
        "source_h": int(max_source_h),
        "target_h": int(max_target_storage_h),
        "height_stripes": int(len(stripes)),
        "programs": int(getattr(plan, "submatrix_program_count")),
        "b_groups": int(getattr(plan, "sharing_group_count")),
        "C": int(getattr(plan, "c_only_rotations")),
        "C+A": int(getattr(plan, "c_only_rotations")),
        "C+B": int(getattr(plan, "cb_shared_rotations")),
        "C+A+B+align": int(getattr(plan, "cb_shared_rotations")),
        "baby": int(getattr(plan, "shared_baby_rotations")),
        "giant": int(getattr(plan, "shared_giant_rotations")),
        "program_diagonal_counts": [int(value) for value in getattr(plan, "program_diagonal_counts")],
        "program_rotation_counts": [int(value) for value in getattr(plan, "program_rotation_counts")],
        "stripe_rows": stripes,
        "groups": group_rows,
    }


def _conv_case_row(case_key: str, *, slots: int) -> dict[str, Any]:
    conv_case = {case.name: case for case in layout_counter.default_conv_cases()}[CONV_CASE_MAP[str(case_key)]]
    candidates = [
        _evaluate_conv_candidate(conv_case, slots=int(slots), channel_tile=int(channel_tile))
        for channel_tile in _channel_tile_candidates(conv_case)
    ]
    best = min(candidates, key=lambda row: (int(row["C+B"]), int(row["C"]), int(row["programs"])))
    return {
        "case": str(case_key),
        "kind": "conv",
        "input": f"{int(conv_case.c_in)}x{int(conv_case.h_in)}x{int(conv_case.w_in)}",
        "output": f"{int(conv_case.c_out)}x{int(conv_case.h_out)}x{int(conv_case.w_out)}",
        "kernel": f"{int(conv_case.kernel)}x{int(conv_case.kernel)}",
        "stride": int(conv_case.stride),
        "gap": str(conv_case.gap_in),
        "selected": best,
        "candidate_cuts": candidates,
    }


def _tconv_case_row(case_key: str) -> dict[str, Any]:
    network, case_name = TCONV_CASE_MAP[str(case_key)]
    rows = sharing._dense_orion_descriptors(str(network), str(case_name))
    provider = sharing._provider_group_descriptors(str(network), str(case_name), full=False)
    dense = {str(row["variant"]): row for row in rows}
    return {
        "case": str(case_key),
        "kind": "tconv",
        "network": str(network),
        "case_name": str(case_name),
        "selected": {
            "programs": int(provider["lt_tasks"]),
            "b_groups": int(provider["sharing_group_count"]),
            "C": int(provider["individual_rotation_eval_count"]),
            "C+A": int(provider["individual_rotation_eval_count"]),
            "C+B": int(provider["oracle_rotation_eval_count"]),
            "C+A+B+align": int(provider["oracle_rotation_eval_count"]),
            "baby": "",
            "giant": "",
            "distinct_n1s": list(provider["distinct_bsgs_n1s"]),
        },
        "provider": provider,
        "orion_dense": {
            "X": dense["Orion independent BSGS"],
            "X+B": dense["Orion+Shared BSGS"],
            "X+B_upper": dense["Orion+Shared BSGS upper"],
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for row in rows:
        selected = dict(row["selected"])
        flat.append(
            {
                "case": row["case"],
                "kind": row["kind"],
                "input": row.get("input", ""),
                "output": row.get("output", ""),
                "channel_tile": selected.get("channel_tile", ""),
                "source_h": selected.get("source_h", ""),
                "target_h": selected.get("target_h", ""),
                "height_stripes": selected.get("height_stripes", ""),
                "programs": selected["programs"],
                "b_groups": selected["b_groups"],
                "C": selected["C"],
                "C+A": selected["C+A"],
                "C+B": selected["C+B"],
                "C+A+B+align": selected["C+A+B+align"],
                "baby": selected.get("baby", ""),
                "giant": selected.get("giant", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Native Halo C-side XABC Rows",
        "",
        "Descriptor-only count. Real/imaginary packing and runtime execution are excluded.",
        "",
        "| case | selected geometry | programs | B groups | C | C+A | C+B | C+A+B+align |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        selected = row["selected"]
        if row["kind"] == "conv":
            geometry = (
                f"{selected['channel_tile']}ch x source H{selected['source_h']}, "
                f"target H{selected['target_h']}, {selected['height_stripes']} stripes"
            )
        else:
            geometry = f"provider local placement, n1s={selected.get('distinct_n1s')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(geometry),
                    str(selected["programs"]),
                    str(selected["b_groups"]),
                    str(selected["C"]),
                    str(selected["C+A"]),
                    str(selected["C+B"]),
                    str(selected["C+A+B+align"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Candidate Conv Cuts", ""])
    for row in rows:
        if row["kind"] != "conv":
            continue
        lines.extend(
            [
                f"### {row['case']}",
                "",
                "| channel tile | source H | target H | stripes | programs | B groups | C | C+B | baby | giant |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate in row["candidate_cuts"]:
            lines.append(
                "| "
                + " | ".join(
                    str(candidate[key])
                    for key in (
                        "channel_tile",
                        "source_h",
                        "target_h",
                        "height_stripes",
                        "programs",
                        "b_groups",
                        "C",
                        "C+B",
                        "baby",
                        "giant",
                    )
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=DEFAULT_CASES, default=None)
    parser.add_argument("--slots", type=int, default=layout_counter.RING_SLOT_COUNT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for case_key in list(args.case or DEFAULT_CASES):
        if str(case_key) in CONV_CASE_MAP:
            rows.append(_conv_case_row(str(case_key), slots=int(args.slots)))
        else:
            rows.append(_tconv_case_row(str(case_key)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "native_halo_c_rows.json").write_text(
        json.dumps({"status": "ok", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "native_halo_c_rows.csv", rows)
    _write_md(out_dir / "native_halo_c_rows.md", rows)
    print(out_dir / "native_halo_c_rows.md")


if __name__ == "__main__":
    main()
