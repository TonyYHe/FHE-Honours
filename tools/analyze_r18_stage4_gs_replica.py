from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from orion.experimental.cir.lattigo_block import R18_STAGE4_SPEC, _build_compact_intra_rows


def _local_shift_families() -> list[int]:
    spec = R18_STAGE4_SPEC
    weight = torch.randn((int(spec.c), int(spec.c), 3, 3), dtype=torch.float32)
    shifts, _output_slots, _values = _build_compact_intra_rows(spec=spec, weight=weight)

    # Collapse away inter-group offsets so we only study the phase-local shift
    # families that a scratch/replica layout can plausibly affect.
    group_block = int(spec.h) * int(spec.gap) * int(spec.w) * int(spec.gap)
    half = int(group_block // 2)
    return sorted({((int(shift) + half) % group_block) - half for shift in shifts.tolist()})


def _min_cover_with_replica(local_shifts: list[int], *, gs: int) -> int:
    """Minimum residual-shift count when each shift r may use r or r-gs.

    Interpreting the extra row as a single fixed-offset input replica means one
    residual basis shift can cover either:
    - a direct term at r
    - a replica-routed term at r + gs
    """

    groups: dict[int, list[int]] = defaultdict(list)
    for shift in local_shifts:
        groups[int(shift) % int(gs)].append(int(shift))

    total = 0
    for values in groups.values():
        values.sort()
        run_len = 1
        for left, right in zip(values, values[1:]):
            if int(right - left) == int(gs):
                run_len += 1
            else:
                total += (run_len + 1) // 2
                run_len = 1
        total += (run_len + 1) // 2
    return int(total)


def _scaled_rotation_estimate(*, baseline_rotations: int, old_local_count: int, new_local_count: int) -> float:
    return float(baseline_rotations) * float(new_local_count) / float(old_local_count)


def analyze(gs_values: list[int] | None = None, *, top_k: int = 20) -> dict[str, object]:
    local_shifts = _local_shift_families()
    baseline_local = int(len(local_shifts))
    baseline_core_rotations = int(R18_STAGE4_SPEC.rotations_per_block)
    baseline_total_rotations = 158  # includes the current +2 prepack cost

    max_span = int(local_shifts[-1] - local_shifts[0])
    ranked: list[dict[str, object]] = []
    for gs in range(1, max_span + 1):
        covered = _min_cover_with_replica(local_shifts, gs=int(gs))
        ranked.append(
            {
                "gs": int(gs),
                "local_shift_cover": int(covered),
                "local_shift_ratio": float(covered) / float(baseline_local),
                "estimated_core_rotations": _scaled_rotation_estimate(
                    baseline_rotations=baseline_core_rotations,
                    old_local_count=baseline_local,
                    new_local_count=int(covered),
                ),
                "estimated_total_rotations": _scaled_rotation_estimate(
                    baseline_rotations=baseline_total_rotations,
                    old_local_count=baseline_local,
                    new_local_count=int(covered),
                ),
            }
        )

    ranked.sort(key=lambda row: (int(row["local_shift_cover"]), int(row["gs"])))
    if gs_values is None:
        requested = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512]
    else:
        requested = [int(value) for value in gs_values]
    requested_rows = [next(row for row in ranked if int(row["gs"]) == int(gs)) for gs in requested if 1 <= int(gs) <= max_span]

    return {
        "status": "ok",
        "scope": "R18 stage4 extra-gs replica estimate on phase-local shift families",
        "baseline": {
            "family": "stage4_same",
            "shape": {
                "c": int(R18_STAGE4_SPEC.c),
                "h": int(R18_STAGE4_SPEC.h),
                "w": int(R18_STAGE4_SPEC.w),
                "gap": int(R18_STAGE4_SPEC.gap),
            },
            "local_shift_count": int(baseline_local),
            "baseline_core_rotations": int(baseline_core_rotations),
            "baseline_total_rotations": int(baseline_total_rotations),
        },
        "best": dict(ranked[0]),
        "requested_gs": requested_rows,
        "top_k": ranked[: int(top_k)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate whether an extra fixed-gs replica can reduce R18 stage4 rotations.")
    parser.add_argument("--gs", nargs="*", type=int, default=None, help="Specific gs values to report.")
    parser.add_argument("--top-k", type=int, default=20, help="How many best gs values to print.")
    args = parser.parse_args()

    payload = analyze(gs_values=args.gs, top_k=int(args.top_k))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
