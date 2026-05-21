from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Any

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.experimental.cir.runtime_group import (
    _add_plaintext_for_add,
    _align_ciphertexts_for_add,
    _encode_plaintext_for_add,
    _rescale_cipher_tensor,
)
from orion.experimental.cir.hybrid_schedule import (
    hybrid_pair_schedule_compatible,
    hybrid_pair_schedule_reject_reason,
    optimize_hybrid_pair_layout,
)
from orion.nn.unified_transform import UnifiedTransformGroup


def _delete_ciphertext_ids(scheme: Any, ids: list[int]) -> None:
    delete = getattr(getattr(scheme, "backend", None), "DeleteCiphertext", None)
    if not callable(delete):
        return
    for value in ids:
        try:
            delete(int(value))
        except Exception:
            pass


def _transform_proxy(
    *,
    name: str,
    diagonals: dict[int, Any],
    level: int,
    module: Any,
    target_index: int,
    input_id: str,
    slots: int,
    scheme: Any,
) -> Any:
    return SimpleNamespace(
        name=str(name),
        diagonals={(0, 0): dict(diagonals)},
        level=int(level),
        scheme=scheme,
        fhe_output_shape=torch.Size([1, int(slots)]),
        output_shape=torch.Size([1, int(slots)]),
        target_index=int(target_index),
        input_id=str(input_id),
        bsgs_ratio=float(getattr(module, "bsgs_ratio", 2.0)),
    )


def _cached_transform_shell(*, level: int, scheme: Any) -> Any:
    return SimpleNamespace(diagonals={}, level=int(level), scheme=scheme)


def _compile_cached_group_if_available(group: UnifiedTransformGroup, scheme: Any) -> bool:
    configure = getattr(group, "_configure_io", None)
    resume_enabled = getattr(group, "_compile_save_resume_enabled", None)
    cached_descriptors = getattr(group, "_cached_transform_descriptors", None)
    if not callable(configure) or not callable(resume_enabled) or not callable(cached_descriptors):
        return False
    configure()
    if not bool(resume_enabled()):
        return False
    try:
        descriptors = cached_descriptors()
    except (KeyError, OSError, RuntimeError):
        return False
    if len(descriptors) != len(group.transforms):
        return False
    group.compile_unified(scheme.backend)
    return True


def _diag_tensor(value: Any, *, slots: int) -> torch.Tensor:
    if value is None:
        return torch.zeros((int(slots),), dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().clone().reshape(-1).to(dtype=torch.float32)
    return torch.as_tensor(list(value), dtype=torch.float32).reshape(-1)


def _merge_input_pair_to_complex(left: Any | None, right: Any | None, *, name: str) -> Any:
    if left is None and right is None:
        raise ValueError("at least one transform is required")
    anchor = left if left is not None else right
    slots = int(anchor.fhe_output_shape[-1])
    if not hybrid_pair_schedule_compatible(left, right, int(slots)):
        reason = hybrid_pair_schedule_reject_reason(left, right, int(slots))
        raise ValueError(f"input-pair hybrid merge requires identical schedules: {reason}")
    left_diags = dict(getattr(left, "diagonals", {}).get((0, 0), {})) if left is not None else {}
    right_diags = dict(getattr(right, "diagonals", {}).get((0, 0), {})) if right is not None else {}
    keys = sorted(set(int(key) for key in left_diags) | set(int(key) for key in right_diags))
    merged: dict[int, torch.Tensor] = {}
    for key in keys:
        left_tensor = _diag_tensor(left_diags.get(int(key)), slots=int(slots))
        right_tensor = _diag_tensor(right_diags.get(int(key)), slots=int(slots))
        merged[int(key)] = left_tensor.to(dtype=torch.complex64) * 0.5 - 1j * right_tensor.to(dtype=torch.complex64) * 0.5
    return SimpleNamespace(
        name=str(name),
        diagonals={(0, 0): merged},
        level=int(anchor.level),
        scheme=anchor.scheme,
        fhe_output_shape=anchor.fhe_output_shape,
        output_shape=anchor.output_shape,
        target_index=int(getattr(anchor, "target_index", 0)),
        input_id=str(getattr(anchor, "input_id", "")),
        bsgs_ratio=float(getattr(anchor, "bsgs_ratio", 2.0)),
    )


def _merge_branch_pair_to_complex(real_transform: Any | None, imag_transform: Any | None, *, name: str) -> Any:
    if real_transform is None and imag_transform is None:
        raise ValueError("at least one branch transform is required")
    anchor = real_transform if real_transform is not None else imag_transform
    slots = int(anchor.fhe_output_shape[-1])
    real_diags = dict(getattr(real_transform, "diagonals", {}).get((0, 0), {})) if real_transform is not None else {}
    imag_diags = dict(getattr(imag_transform, "diagonals", {}).get((0, 0), {})) if imag_transform is not None else {}
    keys = sorted(set(int(key) for key in real_diags) | set(int(key) for key in imag_diags))
    merged: dict[int, torch.Tensor] = {}
    for key in keys:
        real_tensor = _diag_tensor(real_diags.get(int(key)), slots=int(slots))
        imag_tensor = _diag_tensor(imag_diags.get(int(key)), slots=int(slots))
        merged[int(key)] = (real_tensor.to(dtype=torch.complex64) + 1j * imag_tensor.to(dtype=torch.complex64)) * 0.5
    return SimpleNamespace(
        name=str(name),
        diagonals={(0, 0): merged},
        level=int(anchor.level),
        scheme=anchor.scheme,
        fhe_output_shape=anchor.fhe_output_shape,
        output_shape=anchor.output_shape,
        target_index=int(getattr(anchor, "target_index", 0)),
        input_id=str(getattr(anchor, "input_id", "")),
        bsgs_ratio=float(getattr(anchor, "bsgs_ratio", 2.0)),
    )


def _apply_output_rotations(ct: Any, output_rotations: int) -> Any:
    slots = int(ct.scheme.params.get_slots())
    out = ct
    for rotation_index in range(1, int(output_rotations) + 1):
        out = out + out.roll(int(slots // (2**int(rotation_index))), in_place=False)
    return out


class _BiasCacheMixin:
    assigned_level: int | None
    assigned_depth: int | None

    def _bias_plaintext(
        self,
        ct: Any,
        *,
        bias_vector: torch.Tensor | None,
        row_index: int,
        cache: dict[tuple[int, int], Any],
    ) -> Any | None:
        if bias_vector is None:
            return None
        key = (int(row_index), int(ct.level()))
        cached = cache.get(key)
        if cached is not None:
            return cached
        slots = int(ct.slots())
        chunk = torch.zeros((int(slots),), dtype=torch.float32)
        start = int(row_index) * int(slots)
        end = min(int(start + slots), int(bias_vector.numel()))
        if end <= start:
            return None
        chunk[: int(end - start)] = bias_vector[int(start): int(end)]
        ptxt = _encode_plaintext_for_add(ct, chunk)
        cache[key] = ptxt
        return ptxt

    def _add_bias(
        self,
        ct: Any,
        *,
        bias_vector: torch.Tensor | None,
        row_index: int,
        cache: dict[tuple[int, int], Any],
    ) -> Any:
        ptxt = self._bias_plaintext(ct, bias_vector=bias_vector, row_index=int(row_index), cache=cache)
        return ct if ptxt is None else _add_plaintext_for_add(ct, ptxt)


class InputPairConvRuntimeExecutor(_BiasCacheMixin):
    """Conv2d/AvgPool provider using input-pair packing and shared rotations."""

    kernel_kind = "input_pair_conv_shared_rotations"
    native_halo_input_capable = True
    native_halo_output_capable = True
    use_ct_pt_hybrid_packing = True

    def __init__(self, *, module: Any, output_node_id: str, use_ct_pt_hybrid_packing: bool = True) -> None:
        self.module = module
        self.output_node_id = str(output_node_id)
        self.use_ct_pt_hybrid_packing = bool(use_ct_pt_hybrid_packing)
        self.output_shape = getattr(module, "output_shape")
        self.fhe_output_shape = getattr(module, "fhe_output_shape")
        self.groups_by_pair: list[Any] = []
        self.row_indices_by_pair: list[tuple[int, ...]] = []
        self.input_block_pairs: list[tuple[int, int | None]] = []
        self.pair_is_complex: list[bool] = []
        self.hybrid_group_reject_reasons: list[str] = []
        self.hybrid_pair_count = 0
        self.hybrid_pair_rejected_count = 0
        self.hybrid_pair_reject_reasons: list[str] = []
        self.hybrid_pair_layout_strategy = ""
        self.hybrid_pair_layout_strict_pair_count = 0
        self.hybrid_pair_layout_covered_output_count = 0
        self.hybrid_pair_layout_reject_reasons: list[str] = []
        self.rows = 0
        self.cols = 0
        self.output_rotations = 0
        self.bias_vector: torch.Tensor | None = None
        self._bias_plaintext_cache: dict[tuple[int, int], Any] = {}
        self.compile_count = 0
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.last_runtime_timing: dict[str, float] = {
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
            "input_pack_s": 0.0,
            "real_extract_s": 0.0,
        }
        self._compile_cache_metadata: dict[str, Any] = {}
        self._save_resume_precreated_groups_by_pair: dict[
            int,
            tuple[Any, tuple[int, ...], tuple[int, int | None], bool, bool],
        ] = {}

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        backend = str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())
        return backend in {"python", "lattigo", "cheddar"}

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        self._compile_cache_metadata = dict(metadata or {})

    def compile_cache_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kernel_kind,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "output_rotations": int(self.output_rotations),
            "use_ct_pt_hybrid_packing": bool(self.use_ct_pt_hybrid_packing),
            "hybrid_pair_count": int(self.hybrid_pair_count),
            "hybrid_pair_rejected_count": int(self.hybrid_pair_rejected_count),
            "hybrid_pair_reject_reasons": [str(value) for value in self.hybrid_pair_reject_reasons],
            "hybrid_pair_layout_strategy": str(self.hybrid_pair_layout_strategy),
            "hybrid_pair_layout_strict_pair_count": int(self.hybrid_pair_layout_strict_pair_count),
            "hybrid_pair_layout_covered_output_count": int(self.hybrid_pair_layout_covered_output_count),
            "hybrid_pair_layout_reject_reasons": [
                str(value) for value in self.hybrid_pair_layout_reject_reasons
            ],
            "groups": [
                {
                    "storage_key": str(getattr(group, "_storage_key", "")),
                    "row_indices": [int(value) for value in self.row_indices_by_pair[index]],
                    "input_block_pair": [
                        int(self.input_block_pairs[index][0]),
                        None if self.input_block_pairs[index][1] is None else int(self.input_block_pairs[index][1]),
                    ],
                    "pair_is_complex": bool(self.pair_is_complex[index]),
                    "complex_input_block": bool(self.pair_is_complex[index]),
                    "hybrid_pair_reject_reason": (
                        str(self.hybrid_group_reject_reasons[index])
                        if index < len(self.hybrid_group_reject_reasons)
                        else ""
                    ),
                }
                for index, group in enumerate(self.groups_by_pair)
            ],
        }

    def _compile_from_cache_metadata(self, scheme: Any) -> bool:
        metadata = dict(self._compile_cache_metadata or {})
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "load" or not metadata:
            return False
        self.last_runtime_timing.update({"prepare_transforms_s": 0.0, "compile_unified_s": 0.0})
        self.rows = int(metadata.get("rows", 0))
        self.cols = int(metadata.get("cols", 0))
        self.output_rotations = int(metadata.get("output_rotations", 0))
        self.use_ct_pt_hybrid_packing = bool(
            metadata.get("use_ct_pt_hybrid_packing", self.use_ct_pt_hybrid_packing)
        )
        self.hybrid_pair_count = int(metadata.get("hybrid_pair_count", 0))
        self.hybrid_pair_rejected_count = int(metadata.get("hybrid_pair_rejected_count", 0))
        self.hybrid_pair_reject_reasons = [str(value) for value in metadata.get("hybrid_pair_reject_reasons", [])]
        self.hybrid_pair_layout_strategy = str(metadata.get("hybrid_pair_layout_strategy", ""))
        self.hybrid_pair_layout_strict_pair_count = int(metadata.get("hybrid_pair_layout_strict_pair_count", 0))
        self.hybrid_pair_layout_covered_output_count = int(
            metadata.get("hybrid_pair_layout_covered_output_count", 0)
        )
        self.hybrid_pair_layout_reject_reasons = [
            str(value) for value in metadata.get("hybrid_pair_layout_reject_reasons", [])
        ]
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        level = int(self._level(scheme))
        compile_started = time.perf_counter()
        for group_meta in list(metadata.get("groups", [])):
            row_indices = tuple(int(value) for value in group_meta.get("row_indices", []))
            group = UnifiedTransformGroup(
                [_cached_transform_shell(level=int(level), scheme=scheme) for _row in row_indices]
            )
            group._storage_key = str(group_meta["storage_key"])
            group.compile_unified(scheme.backend)
            pair = list(group_meta.get("input_block_pair", []))
            self.groups_by_pair.append(group)
            self.row_indices_by_pair.append(row_indices)
            self.input_block_pairs.append((int(pair[0]), None if len(pair) < 2 or pair[1] is None else int(pair[1])))
            self.pair_is_complex.append(bool(group_meta.get("complex_input_block", group_meta.get("pair_is_complex", False))))
            self.hybrid_group_reject_reasons.append(str(group_meta.get("hybrid_pair_reject_reason", "")))
        if "hybrid_pair_count" not in metadata:
            self.hybrid_pair_count = int(sum(1 for value in self.pair_is_complex if bool(value)))
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.perf_counter() - compile_started)
        return True

    def _compile_from_save_resume_group_cache(self, scheme: Any) -> bool:
        resume_enabled = getattr(scheme.params, "get_compile_save_resume", None)
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "save":
            return False
        if not callable(resume_enabled) or not bool(resume_enabled()):
            return False
        # The legal hybrid schedule is only known after materializing the real
        # transform diagonals, so speculative shell-only resume must not decide
        # that adjacent blocks can be packed as complex inputs.
        self._save_resume_precreated_groups_by_pair = {}
        return False
        try:
            slots = int(scheme.params.get_slots())
            rows = int(math.ceil(int(getattr(self.module, "fhe_output_shape").numel()) / int(slots)))
            cols = int(math.ceil(int(getattr(self.module, "fhe_input_shape").numel()) / int(slots)))
        except Exception:
            return False
        if rows <= 0 or cols <= 0:
            return False

        level = int(self._level(scheme))
        pending: list[tuple[Any, tuple[int, ...], tuple[int, int | None], bool, bool]] = []
        for left_col in range(0, int(cols), 2):
            right_col = int(left_col + 1)
            has_right = int(right_col) < int(cols)
            row_indices = tuple(range(int(rows)))
            group = UnifiedTransformGroup(
                [_cached_transform_shell(level=int(level), scheme=scheme) for _row in row_indices]
            )
            if not _compile_cached_group_if_available(group, scheme):
                pending.append(
                    (
                        group,
                        row_indices,
                        (int(left_col), int(right_col) if has_right else None),
                        bool(has_right),
                        False,
                    )
                )
                self._save_resume_precreated_groups_by_pair = {
                    int(index): entry
                    for index, entry in enumerate(pending)
                }
                return False
            pending.append(
                (
                    group,
                    row_indices,
                    (int(left_col), int(right_col) if has_right else None),
                    bool(has_right),
                    True,
                )
            )

        self.last_runtime_timing.update({"prepare_transforms_s": 0.0, "compile_unified_s": 0.0})
        self.rows = int(rows)
        self.cols = int(cols)
        self.output_rotations = 0
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        for group, row_indices, pair, is_complex, _compiled in pending:
            self.groups_by_pair.append(group)
            self.row_indices_by_pair.append(row_indices)
            self.input_block_pairs.append(pair)
            self.pair_is_complex.append(bool(is_complex))
        self.compile_count += 1
        self._save_resume_precreated_groups_by_pair = {}
        return True

    def compile(self, scheme: Any) -> None:
        if self.groups_by_pair:
            return
        if self._compile_from_cache_metadata(scheme):
            return
        if self._compile_from_save_resume_group_cache(scheme):
            return
        self.last_runtime_timing.update({"prepare_transforms_s": 0.0, "compile_unified_s": 0.0})
        self.hybrid_pair_count = 0
        self.hybrid_pair_rejected_count = 0
        self.hybrid_pair_reject_reasons = []
        self.hybrid_pair_layout_strategy = ""
        self.hybrid_pair_layout_strict_pair_count = 0
        self.hybrid_pair_layout_covered_output_count = 0
        self.hybrid_pair_layout_reject_reasons = []
        self.hybrid_group_reject_reasons = []
        prepare_started = time.perf_counter()
        diagonals, output_rotations = packing.pack_conv2d(self.module, last=False)
        keys = sorted((int(row), int(col)) for row, col in diagonals)
        self.rows = 0 if not keys else max(row for row, _col in keys) + 1
        self.cols = 0 if not keys else max(col for _row, col in keys) + 1
        self.output_rotations = int(output_rotations)
        self.bias_vector = packing.construct_conv2d_bias(self.module).to(dtype=torch.float32)
        slots = int(scheme.params.get_slots())
        level = int(self._level(scheme))
        transforms_by_row_col: dict[tuple[int, int], Any] = {
            (int(row), int(col)): _transform_proxy(
                name=f"{self.output_node_id}_r{int(row)}_c{int(col)}",
                diagonals=diag,
                level=int(level),
                module=self.module,
                target_index=int(row),
                input_id=f"input_{int(col)}",
                slots=int(slots),
                scheme=scheme,
            )
            for (row, col), diag in diagonals.items()
        }
        diagonals.clear()
        self.last_runtime_timing["prepare_transforms_s"] = float(time.perf_counter() - prepare_started)

        compile_started = time.perf_counter()
        precreated_groups = {}
        self._save_resume_precreated_groups_by_pair = {}
        transforms_by_col: dict[int, list[Any | None]] = {
            int(col): [
                transforms_by_row_col.get((int(row), int(col)))
                for row in range(int(self.rows))
            ]
            for col in range(int(self.cols))
        }
        if bool(self.use_ct_pt_hybrid_packing):
            layout_plan = optimize_hybrid_pair_layout(transforms_by_col, int(slots))
            self.hybrid_pair_layout_strict_pair_count = int(layout_plan.strict_pair_count)
            self.hybrid_pair_layout_covered_output_count = int(layout_plan.covered_output_count)
            self.hybrid_pair_layout_reject_reasons = [
                str(value) for value in layout_plan.rejected_adjacent_pair_reasons
            ]
            use_strict_layout = int(layout_plan.strict_pair_count) > 0
        else:
            layout_plan = None
            use_strict_layout = False
            self.hybrid_pair_layout_strategy = "hybrid_disabled"
            self.hybrid_pair_layout_strict_pair_count = 0
            self.hybrid_pair_layout_covered_output_count = 0
            self.hybrid_pair_layout_reject_reasons = []
        if bool(use_strict_layout):
            assert layout_plan is not None
            self.hybrid_pair_layout_strategy = "strict_schedule_dp"
            layout_items = [
                (int(item.left_index), None if item.right_index is None else int(item.right_index), True)
                for item in layout_plan.items
            ]
        elif not bool(self.use_ct_pt_hybrid_packing):
            layout_items = [(int(col), None, False) for col in range(int(self.cols))]
        else:
            self.hybrid_pair_layout_strategy = (
                "adjacent_schedule_fallback"
                if int(layout_plan.strict_pair_count) == 0
                else "adjacent_schedule_regression_guard"
            )
            layout_items = []
            for left_col in range(0, int(self.cols), 2):
                right_col = int(left_col + 1)
                has_right = int(right_col) < int(self.cols)
                layout_items.append((int(left_col), int(right_col) if bool(has_right) else None, False))

        def compile_entries(
            entries: list[tuple[int, Any]],
            *,
            pair: tuple[int, int | None],
            is_complex: bool,
            reject_reason: str = "",
            pair_index: int = 0,
        ) -> None:
            if not entries:
                return
            precreated = precreated_groups.get(int(pair_index)) if bool(is_complex) else None
            if precreated is not None and bool(precreated[4]):
                group = precreated[0]
            else:
                group = (
                    precreated[0]
                    if precreated is not None
                    else UnifiedTransformGroup([transform for _row, transform in entries])
                )
                group.transforms = [transform for _row, transform in entries]
                group.compile_unified(scheme.backend)
            self.groups_by_pair.append(group)
            self.row_indices_by_pair.append(tuple(int(row) for row, _transform in entries))
            self.input_block_pairs.append((int(pair[0]), None if pair[1] is None else int(pair[1])))
            self.pair_is_complex.append(bool(is_complex))
            self.hybrid_group_reject_reasons.append(str(reject_reason))

        for pair_index, (left_col, maybe_right_col, _layout_pair_planned) in enumerate(layout_items):
            has_right = maybe_right_col is not None
            right_col = int(maybe_right_col) if maybe_right_col is not None else int(left_col + 1)
            if bool(has_right):
                candidates: list[tuple[int, Any | None, Any | None]] = []
                reject_reasons: list[str] = []
                for row in range(int(self.rows)):
                    left = transforms_by_col[int(left_col)][int(row)]
                    right = transforms_by_col[int(right_col)][int(row)]
                    if left is None and right is None:
                        continue
                    candidates.append((int(row), left, right))
                    reason = hybrid_pair_schedule_reject_reason(left, right, int(slots))
                    if reason:
                        reject_reasons.append(f"row={int(row)}:{reason}")
                if not candidates:
                    continue
                if reject_reasons:
                    reason = "; ".join(reject_reasons)
                    self.hybrid_pair_rejected_count += 1
                    self.hybrid_pair_reject_reasons.append(
                        f"input_pair=({int(left_col)},{int(right_col)}):{reason}"
                    )
                    left_entries = [
                        (int(row), left)
                        for row, left, _right in candidates
                        if left is not None
                    ]
                    right_entries = [
                        (int(row), right)
                        for row, _left, right in candidates
                        if right is not None
                    ]
                    compile_entries(
                        left_entries,
                        pair=(int(left_col), None),
                        is_complex=False,
                        reject_reason=reason,
                        pair_index=int(pair_index),
                    )
                    compile_entries(
                        right_entries,
                        pair=(int(right_col), None),
                        is_complex=False,
                        reject_reason=reason,
                        pair_index=int(pair_index),
                    )
                else:
                    entries = [
                        (
                            int(row),
                            _merge_input_pair_to_complex(
                                left,
                                right,
                                name=(
                                    f"{self.output_node_id}_input_pair_"
                                    f"{int(left_col)}_{int(right_col)}_r{int(row)}"
                                ),
                            ),
                        )
                        for row, left, right in candidates
                    ]
                    self.hybrid_pair_count += 1
                    compile_entries(
                        entries,
                        pair=(int(left_col), int(right_col)),
                        is_complex=True,
                        pair_index=int(pair_index),
                    )
                for row, _left, _right in candidates:
                    transforms_by_row_col.pop((int(row), int(left_col)), None)
                    transforms_by_row_col.pop((int(row), int(right_col)), None)
            else:
                entries = []
                for row in range(int(self.rows)):
                    transform = transforms_by_col[int(left_col)][int(row)]
                    if transform is not None:
                        entries.append((int(row), transform))
                compile_entries(
                    entries,
                    pair=(int(left_col), None),
                    is_complex=False,
                    pair_index=int(pair_index),
                )
                for row, _transform in entries:
                    transforms_by_row_col.pop((int(row), int(left_col)), None)
        del transforms_by_col
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.perf_counter() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing.update(
            {"evaluate_unified_s": 0.0, "postprocess_s": 0.0, "input_pack_s": 0.0, "real_extract_s": 0.0}
        )
        self.compile(scheme)
        ids = tuple(int(value) for value in getattr(source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            raise RuntimeError(f"{self.output_node_id} requires {self.cols} source ids, got {len(ids)}")

        partials: list[Any | None] = [None for _ in range(int(self.rows))]
        owned_temp_ids: list[int] = []
        evaluate_started = time.perf_counter()
        for pair_index, group in enumerate(self.groups_by_pair):
            left_col, right_col = self.input_block_pairs[int(pair_index)]
            if self.pair_is_complex[int(pair_index)]:
                if right_col is None:
                    raise RuntimeError("complex input pair is missing its imaginary lane")
                pack_started = time.perf_counter()
                imag_id = int(scheme.evaluator.mul_imaginary_unit(int(ids[int(right_col)]), +1, False))
                input_id = int(scheme.evaluator.add_ciphertext(int(ids[int(left_col)]), int(imag_id), False))
                owned_temp_ids.extend([int(imag_id), int(input_id)])
                self.last_runtime_timing["input_pack_s"] += float(time.perf_counter() - pack_started)
            else:
                input_id = int(ids[int(left_col)])
            output_ids = group.evaluate_unified(int(input_id), scheme.backend)
            for row, output_id in zip(self.row_indices_by_pair[int(pair_index)], output_ids):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(scheme.params.get_slots())]),
                    torch.Size([1, int(scheme.params.get_slots())]),
                )
                if self.pair_is_complex[int(pair_index)]:
                    extract_started = time.perf_counter()
                    conj = partial.conjugate(in_place=False)
                    partial, conj = _align_ciphertexts_for_add(partial, conj)
                    partial = partial.add(conj, in_place=True)
                    self.last_runtime_timing["real_extract_s"] += float(time.perf_counter() - extract_started)
                if partials[int(row)] is None:
                    partials[int(row)] = partial
                else:
                    lhs, rhs = _align_ciphertexts_for_add(partials[int(row)], partial)
                    partials[int(row)] = lhs + rhs
        self.last_runtime_timing["evaluate_unified_s"] = float(time.perf_counter() - evaluate_started)

        postprocess_started = time.perf_counter()
        output_ids: list[int] = []
        for row, ct in enumerate(partials):
            if ct is None:
                raise RuntimeError(f"{self.output_node_id} missing output row {row}")
            ct = _rescale_cipher_tensor(ct)
            ct = _apply_output_rotations(ct, int(self.output_rotations))
            ct = self._add_bias(
                ct,
                bias_vector=self.bias_vector,
                row_index=int(row),
                cache=self._bias_plaintext_cache,
            )
            ct.set_scale(int(scheme.params.get_default_scale()))
            output_ids.append(int(ct.ids[0]))
            ct.ids = []
        _delete_ciphertext_ids(scheme, owned_temp_ids)
        self.last_runtime_timing["postprocess_s"] = float(time.perf_counter() - postprocess_started)
        return {self.output_node_id: CipherTensor(scheme, output_ids, self.output_shape, self.fhe_output_shape)}

    def cleanup(self, backend: Any) -> None:
        for group in self.groups_by_pair:
            if hasattr(group, "cleanup"):
                group.cleanup(backend)
        self.groups_by_pair = []
        self.row_indices_by_pair = []
        self.input_block_pairs = []
        self.pair_is_complex = []
        self.hybrid_group_reject_reasons = []
        self.hybrid_pair_count = 0
        self.hybrid_pair_rejected_count = 0
        self.hybrid_pair_reject_reasons = []
        self.hybrid_pair_layout_strategy = ""
        self.hybrid_pair_layout_strict_pair_count = 0
        self.hybrid_pair_layout_covered_output_count = 0
        self.hybrid_pair_layout_reject_reasons = []
        self._bias_plaintext_cache = {}


class BranchPairConvRuntimeExecutor(_BiasCacheMixin):
    """Transition provider packing the main and shortcut branches as real/imag."""

    kernel_kind = "branch_pair_conv_shared_rotations"

    def __init__(self, *, conv_module: Any, shortcut_module: Any, output_node_ids: tuple[str, str]) -> None:
        self.conv_module = conv_module
        self.shortcut_module = shortcut_module
        self.output_node_ids = tuple(str(value) for value in output_node_ids)
        self.groups_by_input: list[Any] = []
        self.row_indices_by_input: list[tuple[int, ...]] = []
        self.rows = 0
        self.cols = 0
        self.output_rotations = 0
        self.conv_bias_vector: torch.Tensor | None = None
        self.shortcut_bias_vector: torch.Tensor | None = None
        self._conv_bias_plaintext_cache: dict[tuple[int, int], Any] = {}
        self._shortcut_bias_plaintext_cache: dict[tuple[int, int], Any] = {}
        self.output_shape = getattr(conv_module, "output_shape")
        self.fhe_output_shape = getattr(conv_module, "fhe_output_shape")
        self.compile_count = 0
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.last_runtime_timing: dict[str, float] = {
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
            "branch_extract_s": 0.0,
            "bias_s": 0.0,
        }
        self._compile_cache_metadata: dict[str, Any] = {}

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        backend = str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())
        return backend in {"python", "lattigo", "cheddar"}

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        self._compile_cache_metadata = dict(metadata or {})

    def compile_cache_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kernel_kind,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "output_rotations": int(self.output_rotations),
            "groups": [
                {
                    "storage_key": str(getattr(group, "_storage_key", "")),
                    "row_indices": [int(value) for value in self.row_indices_by_input[index]],
                    "input_index": int(index),
                }
                for index, group in enumerate(self.groups_by_input)
            ],
        }

    def _compile_from_cache_metadata(self, scheme: Any) -> bool:
        metadata = dict(self._compile_cache_metadata or {})
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "load" or not metadata:
            return False
        self.last_runtime_timing.update({"prepare_transforms_s": 0.0, "compile_unified_s": 0.0})
        self.rows = int(metadata.get("rows", 0))
        self.cols = int(metadata.get("cols", 0))
        self.output_rotations = int(metadata.get("output_rotations", 0))
        self.conv_bias_vector = packing.construct_conv2d_bias(self.conv_module).to(dtype=torch.float32)
        self.shortcut_bias_vector = packing.construct_conv2d_bias(self.shortcut_module).to(dtype=torch.float32)
        level = int(self._level(scheme))
        compile_started = time.perf_counter()
        groups = sorted(list(metadata.get("groups", [])), key=lambda item: int(item.get("input_index", 0)))
        for group_meta in groups:
            row_indices = tuple(int(value) for value in group_meta.get("row_indices", []))
            group = UnifiedTransformGroup(
                [_cached_transform_shell(level=int(level), scheme=scheme) for _row in row_indices]
            )
            group._storage_key = str(group_meta["storage_key"])
            group.compile_unified(scheme.backend)
            self.groups_by_input.append(group)
            self.row_indices_by_input.append(row_indices)
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.perf_counter() - compile_started)
        return True

    def compile(self, scheme: Any) -> None:
        if self.groups_by_input:
            return
        if self._compile_from_cache_metadata(scheme):
            return
        self.last_runtime_timing.update({"prepare_transforms_s": 0.0, "compile_unified_s": 0.0})
        prepare_started = time.perf_counter()
        conv_diags, conv_output_rotations = packing.pack_conv2d(self.conv_module, last=False)
        shortcut_diags, shortcut_output_rotations = packing.pack_conv2d(self.shortcut_module, last=False)
        if int(conv_output_rotations) != int(shortcut_output_rotations):
            raise RuntimeError("branch-pair provider requires matching output-rotation counts")
        keys = sorted(
            set((int(row), int(col)) for row, col in conv_diags)
            | set((int(row), int(col)) for row, col in shortcut_diags)
        )
        self.rows = 0 if not keys else max(row for row, _col in keys) + 1
        self.cols = 0 if not keys else max(col for _row, col in keys) + 1
        self.output_rotations = int(conv_output_rotations)
        slots = int(scheme.params.get_slots())
        level = int(self._level(scheme))
        conv_by_row_col = {
            (int(row), int(col)): _transform_proxy(
                name=f"{self.output_node_ids[0]}_r{int(row)}_c{int(col)}",
                diagonals=diag,
                level=int(level),
                module=self.conv_module,
                target_index=int(row),
                input_id=f"input_{int(col)}",
                slots=int(slots),
                scheme=scheme,
            )
            for (row, col), diag in conv_diags.items()
        }
        shortcut_by_row_col = {
            (int(row), int(col)): _transform_proxy(
                name=f"{self.output_node_ids[1]}_r{int(row)}_c{int(col)}",
                diagonals=diag,
                level=int(level),
                module=self.shortcut_module,
                target_index=int(row),
                input_id=f"input_{int(col)}",
                slots=int(slots),
                scheme=scheme,
            )
            for (row, col), diag in shortcut_diags.items()
        }
        conv_diags.clear()
        shortcut_diags.clear()
        self.conv_bias_vector = packing.construct_conv2d_bias(self.conv_module).to(dtype=torch.float32)
        self.shortcut_bias_vector = packing.construct_conv2d_bias(self.shortcut_module).to(dtype=torch.float32)
        self.last_runtime_timing["prepare_transforms_s"] = float(time.perf_counter() - prepare_started)

        compile_started = time.perf_counter()
        for col in range(int(self.cols)):
            entries: list[tuple[int, Any]] = []
            for row in range(int(self.rows)):
                conv_transform = conv_by_row_col.get((int(row), int(col)))
                shortcut_transform = shortcut_by_row_col.get((int(row), int(col)))
                if conv_transform is None and shortcut_transform is None:
                    continue
                entries.append(
                    (
                        int(row),
                        _merge_branch_pair_to_complex(
                            conv_transform,
                            shortcut_transform,
                            name=f"branch_pair_c{int(col)}_r{int(row)}",
                        ),
                    )
                )
            if not entries:
                continue
            group = UnifiedTransformGroup([transform for _row, transform in entries])
            group.compile_unified(scheme.backend)
            self.groups_by_input.append(group)
            self.row_indices_by_input.append(tuple(int(row) for row, _transform in entries))
            for row, _transform in entries:
                conv_by_row_col.pop((int(row), int(col)), None)
                shortcut_by_row_col.pop((int(row), int(col)), None)
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.perf_counter() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing.update(
            {"evaluate_unified_s": 0.0, "postprocess_s": 0.0, "branch_extract_s": 0.0, "bias_s": 0.0}
        )
        self.compile(scheme)
        ids = tuple(int(value) for value in getattr(source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            raise RuntimeError(f"branch-pair provider requires {self.cols} source ids, got {len(ids)}")

        rows: list[Any | None] = [None for _ in range(int(self.rows))]
        evaluate_started = time.perf_counter()
        for col, group in enumerate(self.groups_by_input):
            output_ids = group.evaluate_unified(int(ids[int(col)]), scheme.backend)
            for row, output_id in zip(self.row_indices_by_input[int(col)], output_ids):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(scheme.params.get_slots())]),
                    torch.Size([1, int(scheme.params.get_slots())]),
                )
                if rows[int(row)] is None:
                    rows[int(row)] = partial
                else:
                    lhs, rhs = _align_ciphertexts_for_add(rows[int(row)], partial)
                    rows[int(row)] = lhs + rhs
        self.last_runtime_timing["evaluate_unified_s"] = float(time.perf_counter() - evaluate_started)

        postprocess_started = time.perf_counter()
        conv_ids: list[int] = []
        shortcut_ids: list[int] = []
        for row, ct in enumerate(rows):
            if ct is None:
                raise RuntimeError(f"branch-pair provider missing output row {row}")
            ct = _rescale_cipher_tensor(ct)
            ct = _apply_output_rotations(ct, int(self.output_rotations))
            extract_started = time.perf_counter()
            conj = ct.conjugate(in_place=False)
            ct, conj = _align_ciphertexts_for_add(ct, conj)
            shortcut = ct - conj
            conv = ct.add(conj, in_place=True)
            shortcut = shortcut.mul_imaginary_unit(-1, in_place=True)
            self.last_runtime_timing["branch_extract_s"] += float(time.perf_counter() - extract_started)
            bias_started = time.perf_counter()
            conv = self._add_bias(
                conv,
                bias_vector=self.conv_bias_vector,
                row_index=int(row),
                cache=self._conv_bias_plaintext_cache,
            )
            shortcut = self._add_bias(
                shortcut,
                bias_vector=self.shortcut_bias_vector,
                row_index=int(row),
                cache=self._shortcut_bias_plaintext_cache,
            )
            self.last_runtime_timing["bias_s"] += float(time.perf_counter() - bias_started)
            conv.set_scale(int(scheme.params.get_default_scale()))
            shortcut.set_scale(int(scheme.params.get_default_scale()))
            conv_ids.append(int(conv.ids[0]))
            shortcut_ids.append(int(shortcut.ids[0]))
            conv.ids = []
            shortcut.ids = []
        self.last_runtime_timing["postprocess_s"] = float(time.perf_counter() - postprocess_started)
        return {
            self.output_node_ids[0]: CipherTensor(scheme, conv_ids, self.output_shape, self.fhe_output_shape),
            self.output_node_ids[1]: CipherTensor(scheme, shortcut_ids, self.output_shape, self.fhe_output_shape),
        }

    def cleanup(self, backend: Any) -> None:
        for group in self.groups_by_input:
            if hasattr(group, "cleanup"):
                group.cleanup(backend)
        self.groups_by_input = []
        self.row_indices_by_input = []
        self._conv_bias_plaintext_cache = {}
        self._shortcut_bias_plaintext_cache = {}


class BranchPairNoHybridConvRuntimeExecutor:
    """Transition provider ablation that evaluates branches without real/imag packing."""

    kernel_kind = "branch_pair_conv_no_real_imag"
    use_ct_pt_hybrid_packing = False

    def __init__(self, *, conv_module: Any, shortcut_module: Any, output_node_ids: tuple[str, str]) -> None:
        self.conv_module = conv_module
        self.shortcut_module = shortcut_module
        self.output_node_ids = tuple(str(value) for value in output_node_ids)
        self.conv_executor = InputPairConvRuntimeExecutor(
            module=conv_module,
            output_node_id=str(self.output_node_ids[0]),
            use_ct_pt_hybrid_packing=False,
        )
        self.shortcut_executor = InputPairConvRuntimeExecutor(
            module=shortcut_module,
            output_node_id=str(self.output_node_ids[1]),
            use_ct_pt_hybrid_packing=False,
        )
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.last_runtime_timing: dict[str, Any] = {}
        self.last_runtime_io: dict[str, Any] = {}
        self.compile_count = 0

    @property
    def rows(self) -> int:
        return max(int(getattr(self.conv_executor, "rows", 0)), int(getattr(self.shortcut_executor, "rows", 0)))

    @property
    def cols(self) -> int:
        return max(int(getattr(self.conv_executor, "cols", 0)), int(getattr(self.shortcut_executor, "cols", 0)))

    @property
    def groups_by_input(self) -> list[Any]:
        return list(getattr(self.conv_executor, "groups_by_pair", [])) + list(
            getattr(self.shortcut_executor, "groups_by_pair", [])
        )

    @property
    def input_block_pairs(self) -> list[tuple[int, int | None]]:
        return list(getattr(self.conv_executor, "input_block_pairs", [])) + list(
            getattr(self.shortcut_executor, "input_block_pairs", [])
        )

    def _sync_delegate_assignment(self) -> None:
        for executor in (self.conv_executor, self.shortcut_executor):
            executor.assigned_level = self.assigned_level
            executor.assigned_depth = self.assigned_depth

    def supports_scheme(self, scheme: Any | None) -> bool:
        return bool(self.conv_executor.supports_scheme(scheme) and self.shortcut_executor.supports_scheme(scheme))

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        payload = dict(metadata or {})
        conv_payload = payload.get("conv")
        shortcut_payload = payload.get("shortcut")
        if isinstance(conv_payload, dict):
            self.conv_executor.load_compile_cache_metadata(conv_payload)
        if isinstance(shortcut_payload, dict):
            self.shortcut_executor.load_compile_cache_metadata(shortcut_payload)

    def compile_cache_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kernel_kind,
            "use_ct_pt_hybrid_packing": False,
            "conv": self.conv_executor.compile_cache_metadata(),
            "shortcut": self.shortcut_executor.compile_cache_metadata(),
            "rows": int(self.rows),
            "cols": int(self.cols),
        }

    def compile(self, scheme: Any) -> None:
        self._sync_delegate_assignment()
        self.conv_executor.compile(scheme)
        self.shortcut_executor.compile(scheme)
        self.compile_count += 1

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        self._sync_delegate_assignment()
        conv_outputs = dict(self.conv_executor(source_ct))
        shortcut_outputs = dict(self.shortcut_executor(source_ct))
        self.last_runtime_timing = {
            "conv": dict(getattr(self.conv_executor, "last_runtime_timing", {}) or {}),
            "shortcut": dict(getattr(self.shortcut_executor, "last_runtime_timing", {}) or {}),
        }
        self.last_runtime_io = {
            "runtime_lowering": "branch_pair_conv_no_real_imag",
            "provider_executor": type(self).__name__,
            "conv_executor": type(self.conv_executor).__name__,
            "shortcut_executor": type(self.shortcut_executor).__name__,
            "use_ct_pt_hybrid_packing": False,
        }
        return {**conv_outputs, **shortcut_outputs}

    def cleanup(self, backend: Any | None) -> None:
        self.conv_executor.cleanup(backend)
        self.shortcut_executor.cleanup(backend)
