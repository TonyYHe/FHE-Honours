"""Unified LinearTransform group support.

This is a focused port of the existing `st` branch UnifiedLinearTransform
support. It is backend support for region experiments: multiple transforms with
the same input can be compiled/evaluated through Lattigo's shared BSGS APIs.
"""

from __future__ import annotations

import ctypes
from itertools import count
from typing import Iterable, List
import h5py
import numpy as np
import torch


_UNIFIED_GROUP_COUNTER = count(1)


class UnifiedTransformGroup:
    """Group LinearTransform-like layers that share one input ciphertext."""

    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)
        self.unified_ids: list[int] | None = None
        self.is_compiled = False
        self._io_mode = "none"
        self._diags_path = ""
        self._storage_key = f"group_{next(_UNIFIED_GROUP_COUNTER)}"
        self._offloaded_plaintext_diagonals = False
        self._diag_indices_by_transform: dict[int, tuple[int, ...]] = {}

    def _scheme_params(self):
        if not self.transforms:
            return None
        scheme = getattr(self.transforms[0], "scheme", None)
        return getattr(scheme, "params", None)

    def _configure_io(self) -> None:
        params = self._scheme_params()
        if params is None:
            self._io_mode = "none"
            self._diags_path = ""
            return

        get_io_mode = getattr(params, "get_io_mode", None)
        get_diags_path = getattr(params, "get_diags_path", None)
        self._io_mode = (
            str(get_io_mode()).lower()
            if callable(get_io_mode)
            else "none"
        )
        self._diags_path = (
            str(get_diags_path() or "")
            if callable(get_diags_path)
            else ""
        )

    def _should_offload_plaintext_diagonals(self) -> bool:
        return self._io_mode == "save"

    def _storage_root_name(self) -> str:
        return "__unified_transform_groups__"

    def _storage_group(self, mode: str):
        if not self._diags_path:
            raise ValueError(
                "UnifiedTransformGroup with io_mode='save' requires "
                "'orion.diags_path' to be set."
            )
        handle = h5py.File(self._diags_path, mode)
        root = handle.require_group(self._storage_root_name())
        return handle, root

    def _save_and_unload_plaintext_diagonals(self, backend) -> None:
        if self.unified_ids is None:
            raise RuntimeError("UnifiedTransformGroup must be compiled before saving diagonals")

        handle, root = self._storage_group("a")
        try:
            if self._storage_key in root:
                del root[self._storage_key]
            storage = root.create_group(self._storage_key)
            for transform_id, diag_indices in self._diag_indices_by_transform.items():
                transform_group = storage.create_group(str(transform_id))
                payload_chunks: list[np.ndarray] = []
                offsets: list[int] = []
                lengths: list[int] = []
                cursor = 0
                for diag_idx in diag_indices:
                    serial_diag, diag_ptr = backend.SerializeDiagonal(
                        int(transform_id),
                        int(diag_idx),
                    )
                    try:
                        serial_arr = np.asarray(serial_diag, dtype=np.uint8).reshape(-1).copy()
                        offsets.append(int(cursor))
                        lengths.append(int(serial_arr.size))
                        payload_chunks.append(serial_arr)
                        cursor += int(serial_arr.size)
                    finally:
                        backend.FreeCArray(diag_ptr)
                payload = (
                    np.concatenate(payload_chunks)
                    if payload_chunks
                    else np.zeros((0,), dtype=np.uint8)
                )
                transform_group.create_dataset(
                    "diag_indices",
                    data=np.asarray(diag_indices, dtype=np.int32),
                )
                transform_group.create_dataset(
                    "diag_offsets",
                    data=np.asarray(offsets, dtype=np.uint64),
                )
                transform_group.create_dataset(
                    "diag_lengths",
                    data=np.asarray(lengths, dtype=np.uint64),
                )
                transform_group.create_dataset("diag_payload", data=payload)
                backend.RemovePlaintextDiagonals(int(transform_id))
        finally:
            handle.close()
        self._offloaded_plaintext_diagonals = True

    def _load_plaintext_diagonals(self, backend) -> None:
        if not self._offloaded_plaintext_diagonals:
            return
        handle, root = self._storage_group("r")
        try:
            storage = root[self._storage_key]
            for transform_id, diag_indices in self._diag_indices_by_transform.items():
                transform_group = storage[str(transform_id)]
                payload = np.asarray(transform_group["diag_payload"][:], dtype=np.uint8)
                offsets = transform_group["diag_offsets"][:].tolist()
                lengths = transform_group["diag_lengths"][:].tolist()
                backend.LoadPlaintextDiagonalsBatch(
                    payload,
                    offsets,
                    lengths,
                    list(diag_indices),
                    int(transform_id),
                )
        finally:
            handle.close()

    def _unload_plaintext_diagonals(self, backend) -> None:
        if not self._offloaded_plaintext_diagonals:
            return
        for transform_id in self._diag_indices_by_transform:
            backend.RemovePlaintextDiagonals(int(transform_id))

    def _delete_offloaded_storage(self) -> None:
        if not self._offloaded_plaintext_diagonals or not self._diags_path:
            return
        try:
            handle, root = self._storage_group("a")
        except OSError:
            return
        try:
            if self._storage_key in root:
                del root[self._storage_key]
        finally:
            handle.close()

    def compile_unified(self, backend) -> None:
        if self.is_compiled:
            return
        if not self.transforms:
            raise ValueError("UnifiedTransformGroup requires at least one transform")

        self._configure_io()
        diag_idxs_list: list[list[int]] = []
        diag_data_list: list[list[float]] = []
        levels: list[int] = []
        has_complex = False

        for transform in self.transforms:
            all_diagonals: dict[int, torch.Tensor] = {}
            for _block_key, block_diags in getattr(transform, "diagonals", {}).items():
                for diag_idx, diag_values in block_diags.items():
                    if isinstance(diag_values, torch.Tensor):
                        values = diag_values.detach().clone().reshape(-1)
                    else:
                        values = torch.as_tensor(list(diag_values))
                    if bool(torch.is_complex(values)):
                        has_complex = True
                    all_diagonals.setdefault(int(diag_idx), values)
            if not all_diagonals:
                raise ValueError("all transforms must have generated diagonals before unified compilation")

            diag_idxs = sorted(all_diagonals.keys())
            diag_data_flat: list[float] = []
            for idx in diag_idxs:
                values = all_diagonals[int(idx)].reshape(-1)
                if has_complex:
                    if not bool(torch.is_complex(values)):
                        values = values.to(dtype=torch.complex64)
                    real = values.real.to(dtype=torch.float64).tolist()
                    imag = values.imag.to(dtype=torch.float64).tolist()
                    for real_value, imag_value in zip(real, imag):
                        diag_data_flat.extend((float(real_value), float(imag_value)))
                else:
                    diag_data_flat.extend(float(value) for value in values.to(dtype=torch.float32).tolist())

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
        self._diag_indices_by_transform = {
            int(transform_id): tuple(int(idx) for idx in diag_idxs)
            for transform_id, diag_idxs in zip(self.unified_ids, diag_idxs_list)
        }

        required_keys: set[int] = set()
        for transform_id in self.unified_ids:
            for key in backend.GetLinearTransformRotationKeys(int(transform_id)):
                required_keys.add(int(key))
        for key in sorted(required_keys):
            backend.GenerateLinearTransformRotationKey(int(key))

        if self._should_offload_plaintext_diagonals():
            self._save_and_unload_plaintext_diagonals(backend)

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
        self._load_plaintext_diagonals(backend)
        transform_ids_array = (ctypes.c_int * len(self.unified_ids))(*[int(v) for v in self.unified_ids])
        try:
            return list(
                backend.EvaluateLinearTransformsWithSharedCache(
                    transform_ids_array,
                    len(self.unified_ids),
                    int(ct_input_id),
                )
            )
        finally:
            self._unload_plaintext_diagonals(backend)

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
        self._delete_offloaded_storage()
        self.unified_ids = None
        self.is_compiled = False
        self._diag_indices_by_transform = {}
        self._offloaded_plaintext_diagonals = False


def can_use_unified_bsgs(layers: List) -> bool:
    from orion.nn.linear import LinearTransform

    if len(layers) < 2:
        return False
    return all(isinstance(layer, LinearTransform) and bool(getattr(layer, "diagonals", {})) for layer in layers)
