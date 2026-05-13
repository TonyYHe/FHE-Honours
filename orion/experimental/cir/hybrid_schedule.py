from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch


@dataclass(frozen=True)
class HybridScheduleSignature:
    slots: int
    normalized_diagonal_keys: tuple[int, ...]
    selected_baby_width: int | None = None
    baby_rotations: tuple[int, ...] = ()
    giant_rotations: tuple[int, ...] = ()


def _int_attr(value: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        raw = getattr(value, name, None)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _int_tuple_attr(value: Any, names: tuple[str, ...]) -> tuple[int, ...]:
    for name in names:
        raw = getattr(value, name, None)
        if raw is None:
            continue
        try:
            return tuple(sorted({int(item) for item in raw}))
        except (TypeError, ValueError):
            continue
    return ()


def _iter_diagonal_keys(diagonals: Any):
    if not isinstance(diagonals, Mapping):
        return
    for key, value in diagonals.items():
        if isinstance(value, Mapping):
            for nested_key in value.keys():
                yield nested_key
        else:
            yield key


def hybrid_schedule_signature(transform: Any, slots: int) -> HybridScheduleSignature:
    """Conservative provider-level schedule signature for real/imag packing.

    The first safety invariant is the normalized diagonal support. Optional BSGS
    planner metadata is included only when the transform already exposes it.
    """

    slot_count = int(slots)
    if slot_count <= 0:
        raise ValueError("slots must be positive")
    keys = tuple(sorted({int(key) % int(slot_count) for key in _iter_diagonal_keys(getattr(transform, "diagonals", {}))}))
    return HybridScheduleSignature(
        slots=int(slot_count),
        normalized_diagonal_keys=keys,
        selected_baby_width=_int_attr(transform, ("selected_n1", "bsgs_n1", "N1", "n1")),
        baby_rotations=_int_tuple_attr(transform, ("baby_rotations", "baby_shifts")),
        giant_rotations=_int_tuple_attr(transform, ("giant_rotations", "giant_shifts")),
    )


def _format_key_sample(keys: tuple[int, ...], *, limit: int = 8) -> str:
    shown = ", ".join(str(int(value)) for value in keys[: int(limit)])
    if len(keys) > int(limit):
        shown = f"{shown}, ..."
    return f"[{shown}]"


def hybrid_pair_schedule_reject_reason(left: Any | None, right: Any | None, slots: int) -> str:
    if left is None and right is None:
        return "missing_left_and_right_transform"
    if left is None:
        return "missing_left_transform"
    if right is None:
        return "missing_right_transform"

    left_sig = hybrid_schedule_signature(left, int(slots))
    right_sig = hybrid_schedule_signature(right, int(slots))
    if left_sig == right_sig:
        return ""

    if left_sig.slots != right_sig.slots:
        return f"slot_count_mismatch(left={left_sig.slots},right={right_sig.slots})"
    if left_sig.normalized_diagonal_keys != right_sig.normalized_diagonal_keys:
        left_keys = set(left_sig.normalized_diagonal_keys)
        right_keys = set(right_sig.normalized_diagonal_keys)
        left_only = tuple(sorted(left_keys - right_keys))
        right_only = tuple(sorted(right_keys - left_keys))
        return (
            "diagonal_key_set_mismatch("
            f"left_count={len(left_keys)},right_count={len(right_keys)},"
            f"left_only={_format_key_sample(left_only)},right_only={_format_key_sample(right_only)}"
            ")"
        )
    return "bsgs_schedule_metadata_mismatch"


def hybrid_pair_schedule_compatible(left: Any | None, right: Any | None, slots: int) -> bool:
    return hybrid_pair_schedule_reject_reason(left, right, int(slots)) == ""


def mark_hybrid_schedule_padding_allowed(transform: Any | None, *, family: str) -> Any | None:
    if transform is None:
        return None
    try:
        setattr(transform, "hybrid_schedule_padding_allowed", True)
        setattr(transform, "hybrid_schedule_family", str(family))
    except Exception:
        pass
    return transform


def hybrid_schedule_padding_allowed(left: Any | None, right: Any | None) -> bool:
    transforms = [transform for transform in (left, right) if transform is not None]
    if not transforms:
        return False
    families = {str(getattr(transform, "hybrid_schedule_family", "")) for transform in transforms}
    if len(families) != 1 or "" in families:
        return False
    return all(bool(getattr(transform, "hybrid_schedule_padding_allowed", False)) for transform in transforms)


def _primary_diagonal_map(transform: Any) -> MutableMapping[int, Any]:
    diagonals = getattr(transform, "diagonals", None)
    if not isinstance(diagonals, MutableMapping):
        raise TypeError("transform.diagonals must be mutable to pad hybrid schedule support")
    block = diagonals.setdefault((0, 0), {})
    if not isinstance(block, MutableMapping):
        raise TypeError("transform diagonal block must be mutable to pad hybrid schedule support")
    return block


def _normalized_key_map(transform: Any | None, slots: int) -> dict[int, int]:
    if transform is None:
        return {}
    block = dict(getattr(transform, "diagonals", {}).get((0, 0), {}))
    return {int(key) % int(slots): int(key) % int(slots) for key in block.keys()}


def _zero_diag_like(sample: Any | None, slots: int) -> torch.Tensor:
    if isinstance(sample, torch.Tensor):
        return torch.zeros_like(sample.detach())
    return torch.zeros((int(slots),), dtype=torch.float32)


def _sample_diag_for_key(left: Any | None, right: Any | None, key: int) -> Any | None:
    for transform in (left, right):
        if transform is None:
            continue
        block = dict(getattr(transform, "diagonals", {}).get((0, 0), {}))
        if int(key) in block:
            return block[int(key)]
    return None


def _zero_transform_like(anchor: Any, *, keys: tuple[int, ...], slots: int, name: str) -> Any:
    block = {int(key): _zero_diag_like(_sample_diag_for_key(anchor, None, int(key)), int(slots)) for key in keys}
    transform = SimpleNamespace(
        name=str(name),
        diagonals={(0, 0): block},
        level=int(getattr(anchor, "level")),
        scheme=getattr(anchor, "scheme"),
        fhe_output_shape=getattr(anchor, "fhe_output_shape"),
        output_shape=getattr(anchor, "output_shape"),
        target_index=int(getattr(anchor, "target_index", 0)),
        input_id=str(getattr(anchor, "input_id", "hybrid_zero_schedule_pad")),
    )
    family = str(getattr(anchor, "hybrid_schedule_family", ""))
    if family:
        mark_hybrid_schedule_padding_allowed(transform, family=family)
    return transform


def pad_hybrid_pair_to_common_schedule(
    left: Any | None,
    right: Any | None,
    slots: int,
    *,
    name: str,
) -> tuple[Any | None, Any | None, str]:
    """Explicitly materialize zero schedule terms for planner-approved tile pairs.

    This is intentionally opt-in: transforms must carry the same
    ``hybrid_schedule_family`` and ``hybrid_schedule_padding_allowed`` marker.
    Callers use this for geometry-derived boundary/halo padding, while arbitrary
    mismatched transforms keep falling back to non-hybrid execution.
    """

    slot_count = int(slots)
    if slot_count <= 0:
        raise ValueError("slots must be positive")
    if left is None and right is None:
        return left, right, ""
    if not hybrid_schedule_padding_allowed(left, right):
        return left, right, ""

    left_map = _normalized_key_map(left, int(slot_count))
    right_map = _normalized_key_map(right, int(slot_count))
    common_keys = tuple(sorted(set(left_map) | set(right_map)))
    if not common_keys:
        return left, right, ""

    left_missing = tuple(key for key in common_keys if int(key) not in left_map)
    right_missing = tuple(key for key in common_keys if int(key) not in right_map)
    left_was_empty = left is None
    right_was_empty = right is None

    if left is None:
        left = _zero_transform_like(right, keys=common_keys, slots=int(slot_count), name=f"{name}_left_zero_schedule")
        left_missing = common_keys
    elif left_missing:
        block = _primary_diagonal_map(left)
        for key in left_missing:
            block[int(key)] = _zero_diag_like(_sample_diag_for_key(left, right, int(key)), int(slot_count))

    if right is None:
        right = _zero_transform_like(left, keys=common_keys, slots=int(slot_count), name=f"{name}_right_zero_schedule")
        right_missing = common_keys
    elif right_missing:
        block = _primary_diagonal_map(right)
        for key in right_missing:
            block[int(key)] = _zero_diag_like(_sample_diag_for_key(left, right, int(key)), int(slot_count))

    if left_missing or right_missing or left_was_empty or right_was_empty:
        return (
            left,
            right,
            "schedule_padded("
            f"left_added={len(left_missing)},right_added={len(right_missing)},"
            f"left_zero={bool(left_was_empty)},right_zero={bool(right_was_empty)}"
            ")",
        )
    return left, right, ""
