from __future__ import annotations

import math
import sys
from typing import Any, Iterable, Sequence


def powers_of_two_below_slots(slots: int) -> tuple[int, ...]:
    values: list[int] = []
    n1 = 1
    while int(n1) < max(1, int(slots)):
        values.append(int(n1))
        n1 <<= 1
    return tuple(values or (1,))


def normalize_diag_indices(diag_indices: Iterable[int], *, slots: int) -> tuple[int, ...]:
    slot_count = max(1, int(slots))
    return tuple(sorted({int(value) % int(slot_count) for value in diag_indices}))


def bsgs_index(
    diag_indices: Iterable[int],
    *,
    slots: int,
    n1: int,
) -> tuple[set[int], set[int]]:
    rot_n1: set[int] = set()
    rot_n2: set[int] = set()
    slot_count = max(1, int(slots))
    step = max(1, int(n1))
    for value in diag_indices:
        rot = int(value) % int(slot_count)
        idx_n1 = int(((rot // step) * step) % int(slot_count))
        idx_n2 = int(rot % step)
        rot_n1.add(int(idx_n1))
        rot_n2.add(int(idx_n2))
    return rot_n1, rot_n2


def lattigo_log_bsgs_ratio(bsgs_ratio: float) -> int:
    if not float(bsgs_ratio) > 0.0:
        return 0
    return int(math.log(float(bsgs_ratio)))


def find_best_bsgs_ratio_n1(
    diag_indices: Iterable[int],
    *,
    slots: int,
    bsgs_ratio: float,
) -> int:
    values = normalize_diag_indices(diag_indices, slots=int(slots))
    if not values:
        return 1
    max_ratio = float(1 << max(0, lattigo_log_bsgs_ratio(float(bsgs_ratio))))
    for n1 in powers_of_two_below_slots(int(slots)):
        rot_n1, rot_n2 = bsgs_index(values, slots=int(slots), n1=int(n1))
        nb_n1 = int(len(rot_n1) - 1)
        nb_n2 = int(len(rot_n2) - 1)
        if nb_n1 == 0:
            ratio = math.nan if nb_n2 == 0 else math.inf
        else:
            ratio = float(nb_n2) / float(nb_n1)
        if ratio == max_ratio:
            return int(n1)
        if ratio > max_ratio:
            return max(1, int(n1) // 2)
    return 1


def find_optimal_unified_bsgs_n1(
    diag_sets: Sequence[Iterable[int]],
    *,
    slots: int,
) -> int:
    base_sets = [
        normalize_diag_indices(diag_set, slots=int(slots))
        for diag_set in diag_sets
    ]
    base_sets = [diag_set for diag_set in base_sets if diag_set]
    if not base_sets:
        return 1

    best_n1 = 1
    best_raw = sys.maxsize
    best_giant = sys.maxsize
    best_baby = sys.maxsize
    for n1 in powers_of_two_below_slots(int(slots)):
        shared_baby: set[int] = set()
        total_giant = 0
        for diag_set in base_sets:
            rot_n1, rot_n2 = bsgs_index(diag_set, slots=int(slots), n1=int(n1))
            shared_baby.update(int(value) for value in rot_n2 if int(value) != 0)
            total_giant += sum(1 for value in rot_n1 if int(value) != 0)
        baby = int(len(shared_baby))
        raw = int(baby + total_giant)
        if (
            raw < best_raw
            or (raw == best_raw and total_giant < best_giant)
            or (raw == best_raw and total_giant == best_giant and baby < best_baby)
            or (
                raw == best_raw
                and total_giant == best_giant
                and baby == best_baby
                and int(n1) < int(best_n1)
            )
        ):
            best_n1 = int(n1)
            best_raw = int(raw)
            best_giant = int(total_giant)
            best_baby = int(baby)
    return int(best_n1)


def lattigo_galois_element(rotation: int, *, slots: int) -> int:
    nth_root = int(4 * max(1, int(slots)))
    return int(pow(5, int(rotation) & int(nth_root - 1), int(nth_root)))


def identity_galois_element(*, slots: int) -> int:
    return lattigo_galois_element(0, slots=int(slots))


def bsgs_eval_rotation_stats_for_n1(
    diag_indices: Iterable[int],
    *,
    slots: int,
    n1: int,
) -> dict[str, int]:
    values = normalize_diag_indices(diag_indices, slots=int(slots))
    if not values:
        return {"rotations": 0, "baby_rotations": 0, "giant_rotations": 0, "n1": int(n1)}
    rot_n1, rot_n2 = bsgs_index(values, slots=int(slots), n1=int(n1))
    baby = sum(1 for value in rot_n2 if int(value) != 0)
    giant = sum(1 for value in rot_n1 if int(value) != 0)
    return {
        "rotations": int(baby + giant),
        "baby_rotations": int(baby),
        "giant_rotations": int(giant),
        "n1": int(n1),
    }


def bsgs_galois_key_stats_for_n1(
    diag_indices: Iterable[int],
    *,
    slots: int,
    n1: int,
) -> dict[str, Any]:
    values = normalize_diag_indices(diag_indices, slots=int(slots))
    rot_n1, rot_n2 = bsgs_index(values, slots=int(slots), n1=int(n1))
    rotations = {int(value) for value in rot_n1}
    rotations.update(int(value) for value in rot_n2)
    keys = sorted(
        {
            lattigo_galois_element(int(rotation), slots=int(slots))
            for rotation in rotations
        }
    )
    identity = identity_galois_element(slots=int(slots))
    nonidentity = [int(key) for key in keys if int(key) != int(identity)]
    return {
        "rotation_key_requests": [int(key) for key in keys],
        "rotation_key_request_count": int(len(keys)),
        "unique_rotation_keys": [int(key) for key in nonidentity],
        "unique_rotation_key_count": int(len(nonidentity)),
        "identity_rotation_key": int(identity),
        "identity_rotation_key_requested": bool(int(identity) in set(int(key) for key in keys)),
    }


def individual_bsgs_ratio_rotation_stats(
    diag_sets: Sequence[Iterable[int]],
    *,
    slots: int,
    bsgs_ratio: float,
) -> dict[str, Any]:
    total = 0
    baby = 0
    giant = 0
    key_request_total = 0
    unique_keys: set[int] = set()
    per_transform: list[dict[str, Any]] = []
    nonempty = 0
    identity = identity_galois_element(slots=int(slots))
    for transform_index, diag_set in enumerate(diag_sets):
        values = normalize_diag_indices(diag_set, slots=int(slots))
        if not values:
            continue
        n1 = find_best_bsgs_ratio_n1(values, slots=int(slots), bsgs_ratio=float(bsgs_ratio))
        eval_stats = bsgs_eval_rotation_stats_for_n1(values, slots=int(slots), n1=int(n1))
        key_stats = bsgs_galois_key_stats_for_n1(values, slots=int(slots), n1=int(n1))
        total += int(eval_stats["rotations"])
        baby += int(eval_stats["baby_rotations"])
        giant += int(eval_stats["giant_rotations"])
        key_request_total += int(key_stats["rotation_key_request_count"])
        unique_keys.update(int(key) for key in key_stats["unique_rotation_keys"])
        per_transform.append(
            {
                "transform_index": int(transform_index),
                "n1": int(n1),
                "rotation_eval_count": int(eval_stats["rotations"]),
                "baby_rotation_eval_count": int(eval_stats["baby_rotations"]),
                "giant_rotation_eval_count": int(eval_stats["giant_rotations"]),
                "rotation_key_request_count": int(key_stats["rotation_key_request_count"]),
                "rotation_key_count": int(key_stats["unique_rotation_key_count"]),
                "identity_rotation_key_requested": bool(key_stats["identity_rotation_key_requested"]),
            }
        )
        nonempty += 1
    return {
        "rotations": int(total),
        "baby_rotations": int(baby),
        "giant_rotations": int(giant),
        "transform_rotation_eval_count_total": int(total),
        "transform_rotation_key_request_count_total": int(key_request_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "identity_rotation_key": int(identity),
        "transforms": int(nonempty),
        "bsgs_groups": int(nonempty),
        "per_transform": per_transform,
    }


def unified_bsgs_rotation_stats(
    diag_sets: Sequence[Iterable[int]],
    *,
    slots: int,
    individual_eval: bool,
) -> dict[str, Any]:
    base_sets = [
        normalize_diag_indices(diag_set, slots=int(slots))
        for diag_set in diag_sets
    ]
    base_sets = [diag_set for diag_set in base_sets if diag_set]
    identity = identity_galois_element(slots=int(slots))
    if not base_sets:
        return {
            "rotations": 0,
            "baby_rotations": 0,
            "giant_rotations": 0,
            "transform_rotation_eval_count_total": 0,
            "shared_rotation_eval_count_total": 0,
            "rotation_eval_count_estimate": 0,
            "rotation_eval_count_mode": "independent_transform_bsgs" if bool(individual_eval) else "shared_group_bsgs",
            "transform_rotation_key_request_count_total": 0,
            "unique_rotation_key_count": 0,
            "unique_rotation_keys": [],
            "identity_rotation_key": int(identity),
            "identity_rotation_key_requested": False,
            "transforms": 0,
            "bsgs_groups": 0,
            "n1": 1,
            "per_transform": [],
        }

    n1 = find_optimal_unified_bsgs_n1(base_sets, slots=int(slots))
    shared_baby: set[int] = set()
    total_baby = 0
    total_giant = 0
    key_request_total = 0
    identity_requested = False
    unique_keys: set[int] = set()
    per_transform: list[dict[str, Any]] = []
    for transform_index, diag_set in enumerate(base_sets):
        rot_n1, rot_n2 = bsgs_index(diag_set, slots=int(slots), n1=int(n1))
        shared_baby.update(int(value) for value in rot_n2 if int(value) != 0)
        eval_stats = bsgs_eval_rotation_stats_for_n1(diag_set, slots=int(slots), n1=int(n1))
        key_stats = bsgs_galois_key_stats_for_n1(diag_set, slots=int(slots), n1=int(n1))
        total_baby += int(eval_stats["baby_rotations"])
        total_giant += int(eval_stats["giant_rotations"])
        key_request_total += int(key_stats["rotation_key_request_count"])
        identity_requested = bool(identity_requested or key_stats["identity_rotation_key_requested"])
        unique_keys.update(int(key) for key in key_stats["unique_rotation_keys"])
        per_transform.append(
            {
                "transform_index": int(transform_index),
                "n1": int(n1),
                "rotation_eval_count": int(eval_stats["rotations"]),
                "baby_rotation_eval_count": int(eval_stats["baby_rotations"]),
                "giant_rotation_eval_count": int(eval_stats["giant_rotations"]),
                "rotation_key_request_count": int(key_stats["rotation_key_request_count"]),
                "rotation_key_count": int(key_stats["unique_rotation_key_count"]),
                "identity_rotation_key_requested": bool(key_stats["identity_rotation_key_requested"]),
            }
        )
    transform_total = int(total_baby + total_giant)
    shared_total = int(len(shared_baby) + total_giant)
    selected_total = int(transform_total if bool(individual_eval) else shared_total)
    return {
        "rotations": int(selected_total),
        "baby_rotations": int(total_baby if bool(individual_eval) else len(shared_baby)),
        "giant_rotations": int(total_giant),
        "transform_rotation_eval_count_total": int(transform_total),
        "shared_rotation_eval_count_total": int(shared_total),
        "rotation_eval_count_estimate": int(selected_total),
        "rotation_eval_count_mode": "independent_transform_bsgs" if bool(individual_eval) else "shared_group_bsgs",
        "transform_rotation_key_request_count_total": int(key_request_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "identity_rotation_key": int(identity),
        "identity_rotation_key_requested": bool(identity_requested),
        "transforms": int(len(base_sets)),
        "bsgs_groups": 1,
        "n1": int(n1),
        "per_transform": per_transform,
    }
