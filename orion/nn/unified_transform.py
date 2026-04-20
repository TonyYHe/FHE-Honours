"""Unified LinearTransform group support.

This is a focused port of the existing `st` branch UnifiedLinearTransform
support. It is backend support for region experiments: multiple transforms with
the same input can be compiled/evaluated through Lattigo's shared BSGS APIs.
"""

from __future__ import annotations

import ctypes
from typing import Iterable, List


class UnifiedTransformGroup:
    """Group LinearTransform-like layers that share one input ciphertext."""

    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)
        self.unified_ids: list[int] | None = None
        self.is_compiled = False

    def compile_unified(self, backend) -> None:
        if self.is_compiled:
            return
        if not self.transforms:
            raise ValueError("UnifiedTransformGroup requires at least one transform")

        diag_idxs_list: list[list[int]] = []
        diag_data_list: list[list[float]] = []
        levels: list[int] = []
        has_complex = False

        for transform in self.transforms:
            all_diagonals: dict[int, list[float]] = {}
            for _block_key, block_diags in getattr(transform, "diagonals", {}).items():
                for diag_idx, diag_values in block_diags.items():
                    values = list(diag_values)
                    if any(isinstance(value, complex) or getattr(value, "imag", 0) != 0 for value in values):
                        has_complex = True
                    all_diagonals.setdefault(int(diag_idx), values)
            if not all_diagonals:
                raise ValueError("all transforms must have generated diagonals before unified compilation")

            diag_idxs = sorted(all_diagonals.keys())
            diag_data_flat: list[float] = []
            for idx in diag_idxs:
                for value in all_diagonals[int(idx)]:
                    if has_complex:
                        diag_data_flat.extend((float(getattr(value, "real", value)), float(getattr(value, "imag", 0.0))))
                    else:
                        diag_data_flat.append(float(value))

            diag_idxs_list.append(diag_idxs)
            diag_data_list.append(diag_data_flat)

            level = getattr(transform, "level", None)
            if level is None:
                level = len(transform.scheme.params.get_logq()) - 1
            levels.append(int(level))

        num_transforms = len(self.transforms)
        diag_idxs_ptrs = (ctypes.POINTER(ctypes.c_int) * num_transforms)()
        diag_idxs_lens = (ctypes.c_int * num_transforms)()
        diag_data_ptrs = (ctypes.POINTER(ctypes.c_double if has_complex else ctypes.c_float) * num_transforms)()
        diag_data_lens = (ctypes.c_int * num_transforms)()

        # Keep arrays alive until backend call returns.
        owned_arrays: list[object] = []
        for idx, diag_idxs in enumerate(diag_idxs_list):
            array = (ctypes.c_int * len(diag_idxs))(*diag_idxs)
            owned_arrays.append(array)
            diag_idxs_ptrs[idx] = array
            diag_idxs_lens[idx] = len(diag_idxs)

        for idx, diag_data in enumerate(diag_data_list):
            array_type = ctypes.c_double if has_complex else ctypes.c_float
            array = (array_type * len(diag_data))(*diag_data)
            owned_arrays.append(array)
            diag_data_ptrs[idx] = array
            diag_data_lens[idx] = len(diag_data)

        levels_array = (ctypes.c_int * num_transforms)(*levels)
        owned_arrays.append(levels_array)

        generate = (
            backend.GenerateLinearTransformsUnifiedComplex
            if has_complex and hasattr(backend, "GenerateLinearTransformsUnifiedComplex")
            else backend.GenerateLinearTransformsUnified
        )
        self.unified_ids = list(
            generate(
                num_transforms,
                diag_idxs_ptrs,
                diag_idxs_lens,
                diag_data_ptrs,
                diag_data_lens,
                levels_array,
            )
        )
        self.is_compiled = True

        for transform_id in self.unified_ids:
            for key in backend.GetLinearTransformRotationKeys(int(transform_id)):
                backend.GenerateLinearTransformRotationKey(int(key))

    def get_transform_ids(self, transform) -> dict[tuple[int, int], int]:
        if not self.is_compiled or self.unified_ids is None:
            raise RuntimeError("UnifiedTransformGroup not compiled")
        try:
            index = self.transforms.index(transform)
        except ValueError as exc:
            raise ValueError("Transform not found in unified group") from exc
        return {(0, 0): int(self.unified_ids[index])}

    def evaluate_unified(self, ct_input_id: int, backend) -> list[int]:
        if not self.is_compiled or self.unified_ids is None:
            raise RuntimeError("UnifiedTransformGroup must be compiled before evaluation")
        transform_ids_array = (ctypes.c_int * len(self.unified_ids))(*[int(v) for v in self.unified_ids])
        return list(backend.EvaluateLinearTransformsWithSharedCache(transform_ids_array, len(self.unified_ids), int(ct_input_id)))

    def execute(self, calling_transform, ct_input):
        from orion.backend.python.tensors import CipherTensor

        if not isinstance(ct_input, CipherTensor):
            raise TypeError(f"Expected CipherTensor input, got {type(ct_input)}")
        ct_id = int(ct_input.ids[0])

        if not hasattr(self, "_result_cache") or getattr(self, "_result_cache_key") != ct_id:
            output_ids = self.evaluate_unified(ct_id, calling_transform.scheme.backend)
            self._result_cache = {}
            self._result_cache_key = ct_id
            for index, transform in enumerate(self.transforms):
                out_shape = transform.fhe_output_shape
                self._result_cache[transform] = CipherTensor(calling_transform.scheme, [output_ids[index]], out_shape, out_shape)
        return self._result_cache[calling_transform]

    def cleanup(self, backend) -> None:
        if self.unified_ids is not None:
            for transform_id in self.unified_ids:
                backend.DeleteLinearTransform(int(transform_id))
        self.unified_ids = None
        self.is_compiled = False


def can_use_unified_bsgs(layers: List) -> bool:
    from orion.nn.linear import LinearTransform

    if len(layers) < 2:
        return False
    return all(isinstance(layer, LinearTransform) and bool(getattr(layer, "diagonals", {})) for layer in layers)
