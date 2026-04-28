from __future__ import annotations

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
        merged[int(key)] = real_tensor.to(dtype=torch.complex64) + 1j * imag_tensor.to(dtype=torch.complex64)
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
    use_ct_pt_hybrid_packing = True

    def __init__(self, *, module: Any, output_node_id: str) -> None:
        self.module = module
        self.output_node_id = str(output_node_id)
        self.output_shape = getattr(module, "output_shape")
        self.fhe_output_shape = getattr(module, "fhe_output_shape")
        self.groups_by_pair: list[Any] = []
        self.row_indices_by_pair: list[tuple[int, ...]] = []
        self.input_block_pairs: list[tuple[int, int | None]] = []
        self.pair_is_complex: list[bool] = []
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

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        backend = str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())
        return backend in {"python", "lattigo", "cheddar"}

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def compile(self, scheme: Any) -> None:
        if self.groups_by_pair:
            return
        self.last_runtime_timing.update({"prepare_transforms_s": 0.0, "compile_unified_s": 0.0})
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
        self.last_runtime_timing["prepare_transforms_s"] = float(time.perf_counter() - prepare_started)

        compile_started = time.perf_counter()
        for left_col in range(0, int(self.cols), 2):
            right_col = int(left_col + 1)
            has_right = int(right_col) < int(self.cols)
            entries: list[tuple[int, Any]] = []
            for row in range(int(self.rows)):
                left = transforms_by_row_col.get((int(row), int(left_col)))
                right = transforms_by_row_col.get((int(row), int(right_col))) if has_right else None
                if left is None and right is None:
                    continue
                transform = (
                    _merge_input_pair_to_complex(
                        left,
                        right,
                        name=f"{self.output_node_id}_input_pair_{int(left_col)}_{int(right_col)}_r{int(row)}",
                    )
                    if has_right
                    else (left if left is not None else right)
                )
                entries.append((int(row), transform))
            if not entries:
                continue
            group = UnifiedTransformGroup([transform for _row, transform in entries])
            group.compile_unified(scheme.backend)
            self.groups_by_pair.append(group)
            self.row_indices_by_pair.append(tuple(int(row) for row, _transform in entries))
            self.input_block_pairs.append((int(left_col), int(right_col) if has_right else None))
            self.pair_is_complex.append(bool(has_right))
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
                    partial = partial + conj
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
        }

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        backend = str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())
        return backend in {"python", "lattigo", "cheddar"}

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def compile(self, scheme: Any) -> None:
        if self.groups_by_input:
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
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.perf_counter() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing.update({"evaluate_unified_s": 0.0, "postprocess_s": 0.0})
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
            conj = ct.conjugate(in_place=False)
            ct, conj = _align_ciphertexts_for_add(ct, conj)
            conv = (ct + conj) * 0.5
            shortcut = (ct - conj).mul_imaginary_unit(-1, in_place=False) * 0.5
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
