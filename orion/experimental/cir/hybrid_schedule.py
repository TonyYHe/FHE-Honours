from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
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


@dataclass(frozen=True)
class HybridPairLayoutItem:
    left_index: int
    right_index: int | None
    strict_schedule_pair: bool
    covered_output_count: int = 0
    reject_reasons: tuple[str, ...] = ()
    schedule_materialized: bool = False


@dataclass(frozen=True)
class HybridPairLayoutPlan:
    items: tuple[HybridPairLayoutItem, ...]
    strict_pair_count: int
    covered_output_count: int
    singleton_count: int
    rejected_adjacent_pair_reasons: tuple[str, ...] = ()
    schedule_materialized_pair_count: int = 0
    schedule_materialized_output_count: int = 0

    @property
    def uses_shifted_boundaries(self) -> bool:
        if not self.items:
            return False
        first_left = int(self.items[0].left_index)
        for item in self.items:
            if item.right_index is not None and (int(item.left_index) - int(first_left)) % 2 != 0:
                return True
        return False


@dataclass(frozen=True)
class HybridScheduleMaterializationResult:
    pair_count: int
    output_count: int
    reasons: tuple[str, ...] = ()


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


def _iter_transform_diagonal_keys(transform: Any):
    indices_by_block = getattr(transform, "_single_slot_diag_indices_by_block", None)
    yielded = False
    if indices_by_block is not None:
        for diag_indices in dict(indices_by_block or {}).values():
            for value in diag_indices:
                yielded = True
                yield value
    if yielded:
        return
    yield from _iter_diagonal_keys(getattr(transform, "diagonals", {}))


def hybrid_schedule_signature(transform: Any, slots: int) -> HybridScheduleSignature:
    """Conservative provider-level schedule signature for real/imag packing.

    The first safety invariant is the normalized diagonal support. Optional BSGS
    planner metadata is included only when the transform already exposes it.
    """

    slot_count = int(slots)
    if slot_count <= 0:
        raise ValueError("slots must be positive")
    keys = tuple(sorted({int(key) % int(slot_count) for key in _iter_transform_diagonal_keys(transform)}))
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


def hybrid_transform_sequence_pair_reject_reasons(
    left_transforms: Sequence[Any | None],
    right_transforms: Sequence[Any | None],
    slots: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    width = max(len(left_transforms), len(right_transforms))
    for output_index in range(int(width)):
        left = left_transforms[int(output_index)] if int(output_index) < len(left_transforms) else None
        right = right_transforms[int(output_index)] if int(output_index) < len(right_transforms) else None
        if left is None and right is None:
            continue
        reason = hybrid_pair_schedule_reject_reason(left, right, int(slots))
        if reason:
            reasons.append(f"output={int(output_index)}:{reason}")
    return tuple(reasons)


def hybrid_transform_sequence_pair_weight(
    left_transforms: Sequence[Any | None],
    right_transforms: Sequence[Any | None],
) -> int:
    width = max(len(left_transforms), len(right_transforms))
    count = 0
    for output_index in range(int(width)):
        left = left_transforms[int(output_index)] if int(output_index) < len(left_transforms) else None
        right = right_transforms[int(output_index)] if int(output_index) < len(right_transforms) else None
        if left is not None or right is not None:
            count += 1
    return int(count)


def _schedule_reject_can_be_materialized(reason: str) -> bool:
    return bool(
        str(reason).startswith("diagonal_key_set_mismatch(")
        or str(reason) == "missing_left_transform"
        or str(reason) == "missing_right_transform"
    )


def hybrid_transform_sequence_pair_schedule_materializable(
    left_transforms: Sequence[Any | None],
    right_transforms: Sequence[Any | None],
    slots: int,
) -> tuple[bool, tuple[str, ...], int]:
    reasons: list[str] = []
    materialized_outputs = 0
    width = max(len(left_transforms), len(right_transforms))
    for output_index in range(int(width)):
        left = left_transforms[int(output_index)] if int(output_index) < len(left_transforms) else None
        right = right_transforms[int(output_index)] if int(output_index) < len(right_transforms) else None
        if left is None and right is None:
            continue
        reason = hybrid_pair_schedule_reject_reason(left, right, int(slots))
        if not reason:
            continue
        if not _schedule_reject_can_be_materialized(str(reason)):
            reasons.append(f"output={int(output_index)}:{reason}:not_materializable")
            continue
        if left is not None and right is not None:
            left_sig = hybrid_schedule_signature(left, int(slots))
            right_sig = hybrid_schedule_signature(right, int(slots))
            if (
                left_sig.selected_baby_width != right_sig.selected_baby_width
                or left_sig.baby_rotations != right_sig.baby_rotations
                or left_sig.giant_rotations != right_sig.giant_rotations
            ):
                reasons.append(f"output={int(output_index)}:bsgs_schedule_metadata_mismatch:not_materializable")
                continue
        if not hybrid_schedule_padding_allowed(left, right):
            reasons.append(f"output={int(output_index)}:{reason}:layout_padding_not_allowed")
            continue
        materialized_outputs += 1
    return bool(materialized_outputs > 0 and not reasons), tuple(reasons), int(materialized_outputs)


def optimize_hybrid_pair_layout(
    transforms_by_block: Mapping[int, Sequence[Any | None]],
    slots: int,
    *,
    allow_schedule_materialization: bool = False,
) -> HybridPairLayoutPlan:
    """Choose adjacent source-block pair boundaries that maximize strict pairs.

    This is a tiny DP analogue of the paper's layout search: at each source
    boundary, choose either a singleton block or a real/imag pair. A pair is
    admitted only when every paired output transform already has an identical
    schedule signature. The helper does not pad or mutate transforms.
    """

    block_indices = tuple(sorted(int(index) for index in transforms_by_block))
    if not block_indices:
        return HybridPairLayoutPlan((), 0, 0, 0, ())
    expected = tuple(range(int(block_indices[0]), int(block_indices[-1]) + 1))
    if block_indices != expected:
        raise ValueError("hybrid pair layout optimizer requires contiguous block indices")

    rejected_reasons: list[str] = []
    pair_cache: dict[int, tuple[bool, bool, bool, int, int, tuple[str, ...]]] = {}
    for index in block_indices[:-1]:
        left = tuple(transforms_by_block[int(index)])
        right = tuple(transforms_by_block[int(index + 1)])
        reasons = hybrid_transform_sequence_pair_reject_reasons(left, right, int(slots))
        weight = hybrid_transform_sequence_pair_weight(left, right)
        strict_compatible = bool(weight > 0 and not reasons)
        schedule_materializable = False
        materialized_outputs = 0
        materialize_reasons: tuple[str, ...] = ()
        if not strict_compatible and bool(allow_schedule_materialization) and int(weight) > 0:
            schedule_materializable, materialize_reasons, materialized_outputs = (
                hybrid_transform_sequence_pair_schedule_materializable(left, right, int(slots))
            )
        compatible = bool(strict_compatible or schedule_materializable)
        pair_cache[int(index)] = (
            bool(compatible),
            bool(strict_compatible),
            bool(schedule_materializable),
            int(weight),
            int(materialized_outputs),
            tuple(reasons if reasons else materialize_reasons),
        )
        if not compatible and (weight > 0 or reasons or materialize_reasons):
            shown = materialize_reasons if materialize_reasons else reasons
            rejected_reasons.append(f"input_pair=({int(index)},{int(index + 1)}):" + "; ".join(shown))

    # dp[position] = (pair_count, covered_outputs, -materialized_outputs, -singleton_count, items)
    dp: dict[int, tuple[int, int, int, int, tuple[HybridPairLayoutItem, ...]]] = {
        len(block_indices): (0, 0, 0, 0, ())
    }
    for pos in range(len(block_indices) - 1, -1, -1):
        left_index = int(block_indices[int(pos)])
        next_single = dp[int(pos + 1)]
        singleton_item = HybridPairLayoutItem(
            left_index=int(left_index),
            right_index=None,
            strict_schedule_pair=False,
            covered_output_count=0,
        )
        best = (
            int(next_single[0]),
            int(next_single[1]),
            int(next_single[2]),
            int(next_single[3] - 1),
            (singleton_item,) + tuple(next_single[4]),
        )
        if int(pos + 1) < len(block_indices):
            compatible, strict, materialized, weight, materialized_outputs, reasons = pair_cache.get(
                int(left_index),
                (False, False, False, 0, 0, ()),
            )
            if compatible:
                next_pair = dp[int(pos + 2)]
                pair_item = HybridPairLayoutItem(
                    left_index=int(left_index),
                    right_index=int(left_index + 1),
                    strict_schedule_pair=bool(strict),
                    covered_output_count=int(weight),
                    reject_reasons=tuple(reasons),
                    schedule_materialized=bool(materialized),
                )
                candidate = (
                    int(next_pair[0] + 1),
                    int(next_pair[1] + int(weight)),
                    int(next_pair[2] - int(materialized_outputs)),
                    int(next_pair[3]),
                    (pair_item,) + tuple(next_pair[4]),
                )
                if candidate[:4] > best[:4]:
                    best = candidate
        dp[int(pos)] = best

    pair_count, covered_count, materialized_output_score, singleton_score, items = dp[0]
    materialized_pair_count = sum(
        1
        for item in items
        if item.right_index is not None and bool(item.schedule_materialized)
    )
    return HybridPairLayoutPlan(
        items=tuple(items),
        strict_pair_count=int(pair_count),
        covered_output_count=int(covered_count),
        singleton_count=int(-singleton_score),
        rejected_adjacent_pair_reasons=tuple(rejected_reasons),
        schedule_materialized_pair_count=int(materialized_pair_count),
        schedule_materialized_output_count=int(-materialized_output_score),
    )


def materialize_hybrid_pair_layout_schedules(
    transforms_by_block: MutableMapping[int, list[Any | None]],
    plan: HybridPairLayoutPlan,
    slots: int,
    *,
    name_prefix: str,
) -> HybridScheduleMaterializationResult:
    pair_reasons: list[str] = []
    pair_count = 0
    output_count = 0
    for item in plan.items:
        if item.right_index is None or not bool(item.schedule_materialized):
            continue
        left_index = int(item.left_index)
        right_index = int(item.right_index)
        left_transforms = list(transforms_by_block[int(left_index)])
        right_transforms = list(transforms_by_block[int(right_index)])
        width = max(len(left_transforms), len(right_transforms))
        if len(left_transforms) < int(width):
            left_transforms.extend([None] * int(int(width) - len(left_transforms)))
        if len(right_transforms) < int(width):
            right_transforms.extend([None] * int(int(width) - len(right_transforms)))
        reasons: list[str] = []
        for output_index in range(int(width)):
            left = left_transforms[int(output_index)]
            right = right_transforms[int(output_index)]
            if left is None and right is None:
                continue
            before_reason = hybrid_pair_schedule_reject_reason(left, right, int(slots))
            if not before_reason:
                continue
            left, right, pad_reason = pad_hybrid_pair_to_common_schedule(
                left,
                right,
                int(slots),
                name=f"{name_prefix}_src{left_index}_{right_index}_out{int(output_index)}",
            )
            left_transforms[int(output_index)] = left
            right_transforms[int(output_index)] = right
            after_reason = hybrid_pair_schedule_reject_reason(left, right, int(slots))
            if after_reason:
                reasons.append(f"output={int(output_index)}:{after_reason}")
            elif pad_reason:
                output_count += 1
                reasons.append(f"output={int(output_index)}:{pad_reason}")
        transforms_by_block[int(left_index)] = left_transforms
        transforms_by_block[int(right_index)] = right_transforms
        if reasons:
            pair_count += 1
            pair_reasons.append(f"input_pair=({left_index},{right_index}):" + "; ".join(reasons))
    return HybridScheduleMaterializationResult(
        pair_count=int(pair_count),
        output_count=int(output_count),
        reasons=tuple(pair_reasons),
    )


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
    return {int(key) % int(slots): int(key) % int(slots) for key in _iter_transform_diagonal_keys(transform)}


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
    for attr in ("selected_n1", "bsgs_n1", "N1", "n1", "baby_rotations", "baby_shifts", "giant_rotations", "giant_shifts"):
        if hasattr(anchor, attr):
            try:
                setattr(transform, attr, getattr(anchor, attr))
            except Exception:
                pass
    family = str(getattr(anchor, "hybrid_schedule_family", ""))
    if family:
        mark_hybrid_schedule_padding_allowed(transform, family=family)
    return transform


def _sync_single_slot_diag_indices(transform: Any | None) -> None:
    if transform is None or not hasattr(transform, "_single_slot_diag_indices_by_block"):
        return
    try:
        block = dict(getattr(transform, "diagonals", {}).get((0, 0), {}) or {})
        if block:
            keys = tuple(sorted(int(key) for key in block))
        else:
            keys = tuple(sorted(int(key) for key in _iter_transform_diagonal_keys(transform)))
        setattr(transform, "_single_slot_diag_indices_by_block", {(0, 0): keys})
    except Exception:
        pass


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
        _sync_single_slot_diag_indices(left)

    if right is None:
        right = _zero_transform_like(left, keys=common_keys, slots=int(slot_count), name=f"{name}_right_zero_schedule")
        right_missing = common_keys
    elif right_missing:
        block = _primary_diagonal_map(right)
        for key in right_missing:
            block[int(key)] = _zero_diag_like(_sample_diag_for_key(left, right, int(key)), int(slot_count))
        _sync_single_slot_diag_indices(right)

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
