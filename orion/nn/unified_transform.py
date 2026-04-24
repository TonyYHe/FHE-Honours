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

from orion.backend.python.io_prefetch import (
    AsyncIOPrefetcher,
    estimate_linear_transform_device_bytes,
    should_prefetch_saved_io,
)


_UNIFIED_GROUP_COUNTER = count(1)


class UnifiedTransformGroup:
    """Group LinearTransform-like layers that share one input ciphertext."""

    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)
        self.unified_ids: list[int] | None = None
        self.is_compiled = False
        self._io_mode = "none"
        self._diags_path = ""
        self._keys_path = ""
        self._storage_key = f"group_{next(_UNIFIED_GROUP_COUNTER)}"
        self._offloaded_plaintext_diagonals = False
        self._diag_indices_by_transform: dict[int, tuple[int, ...]] = {}
        self._required_keys: tuple[tuple[int, int | None], ...] = ()
        self._io_prefetcher = AsyncIOPrefetcher()
        self._prefetch_host_bytes: int | None = None
        self._prefetch_device_bytes: int | None = None

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
        get_keys_path = getattr(params, "get_keys_path", None)
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
        self._keys_path = (
            str(get_keys_path() or "")
            if callable(get_keys_path)
            else ""
        )

    def _should_offload_plaintext_diagonals(self) -> bool:
        return self._io_mode == "save" and bool(self._diags_path)

    def _should_offload_rotation_keys(self) -> bool:
        return self._io_mode == "save" and bool(self._keys_path)

    def _storage_root_name(self) -> str:
        return "__unified_transform_groups__"

    def _rotation_key_storage_name(self, key: int, level: int | None) -> str:
        return str(int(key)) if level is None else f"{int(key)}@{int(level)}"

    def _rotation_key_requests(self, backend, transform_id: int) -> tuple[tuple[int, int | None], ...]:
        get_requests = getattr(backend, "GetLinearTransformRotationKeyRequests", None)
        if callable(get_requests):
            flat = list(get_requests(int(transform_id)))
            if len(flat) % 2 != 0:
                raise RuntimeError("backend returned malformed rotation key requests")
            requests: dict[int, int] = {}
            for index in range(0, len(flat), 2):
                key = int(flat[index])
                level = int(flat[index + 1])
                requests[key] = max(level, requests.get(key, level))
            return tuple(sorted(requests.items()))
        return tuple((int(key), None) for key in backend.GetLinearTransformRotationKeys(int(transform_id)))

    def _generate_rotation_key(self, backend, key: int, level: int | None) -> None:
        if level is not None and hasattr(backend, "GenerateLinearTransformRotationKeyAtLevel"):
            backend.GenerateLinearTransformRotationKeyAtLevel(int(key), int(level))
        else:
            backend.GenerateLinearTransformRotationKey(int(key))

    def _generate_and_serialize_rotation_key(self, backend, key: int, level: int | None):
        if level is not None and hasattr(backend, "GenerateAndSerializeRotationKeyAtLevel"):
            return backend.GenerateAndSerializeRotationKeyAtLevel(int(key), int(level))
        return backend.GenerateAndSerializeRotationKey(int(key))

    def _plaintext_payload_required(self, backend) -> bool:
        return bool(getattr(backend, "load_plaintext_diagonals_requires_payload", True))

    def _estimate_prefetch_host_bytes(self, backend) -> int:
        if self._prefetch_host_bytes is not None:
            return int(self._prefetch_host_bytes)

        total_bytes = 0
        if self._should_offload_rotation_keys():
            with h5py.File(self._keys_path, "r") as handle:
                for key, level in self._required_keys:
                    key_name = self._rotation_key_storage_name(int(key), level)
                    if key_name not in handle and str(int(key)) in handle:
                        key_name = str(int(key))
                    dataset = handle[key_name]
                    total_bytes += int(dataset.size) * int(dataset.dtype.itemsize)

        if self._offloaded_plaintext_diagonals and self._plaintext_payload_required(backend):
            handle, root = self._storage_group("r")
            try:
                storage = root[self._storage_key]
                for transform_id in self._diag_indices_by_transform:
                    transform_group = storage[str(transform_id)]
                    for name in ("diag_payload", "diag_offsets", "diag_lengths", "diag_indices"):
                        dataset = transform_group[name]
                        total_bytes += int(dataset.size) * int(dataset.dtype.itemsize)
            finally:
                handle.close()

        self._prefetch_host_bytes = int(total_bytes)
        return int(total_bytes)

    def _estimate_prefetch_device_bytes(self, backend) -> int:
        if self._prefetch_device_bytes is not None:
            return int(self._prefetch_device_bytes)

        total_bytes = 0
        if self._should_offload_rotation_keys():
            total_bytes += self._estimate_prefetch_host_bytes(backend)
        if self._offloaded_plaintext_diagonals:
            for transform_id in self._diag_indices_by_transform:
                total_bytes += estimate_linear_transform_device_bytes(backend, int(transform_id))
        self._prefetch_device_bytes = int(total_bytes)
        return int(total_bytes)

    def _read_saved_io_bundle(self, backend, *, prefetch: bool) -> dict[str, object] | None:
        if prefetch:
            if not should_prefetch_saved_io(
                self._estimate_prefetch_host_bytes(backend),
                backend=backend,
                device_bytes=self._estimate_prefetch_device_bytes(backend),
            ):
                return None

        bundle: dict[str, object] = {
            "rotation_keys": (),
            "plaintexts": {},
        }

        if self._should_offload_rotation_keys():
            rotation_keys = []
            with h5py.File(self._keys_path, "r") as handle:
                for key, level in self._required_keys:
                    key_name = self._rotation_key_storage_name(int(key), level)
                    if key_name not in handle and str(int(key)) in handle:
                        key_name = str(int(key))
                    rotation_keys.append((int(key), np.asarray(handle[key_name][()], dtype=np.uint8).copy()))
            bundle["rotation_keys"] = tuple(rotation_keys)

        if self._offloaded_plaintext_diagonals:
            if not self._plaintext_payload_required(backend):
                bundle["plaintexts"] = {
                    int(transform_id): {
                        "payload": np.zeros((0,), dtype=np.uint8),
                        "offsets": (),
                        "lengths": (),
                        "diag_indices": tuple(int(idx) for idx in diag_indices),
                    }
                    for transform_id, diag_indices in self._diag_indices_by_transform.items()
                }
                return bundle

            handle, root = self._storage_group("r")
            try:
                storage = root[self._storage_key]
                plaintexts: dict[int, dict[str, object]] = {}
                for transform_id, diag_indices in self._diag_indices_by_transform.items():
                    transform_group = storage[str(transform_id)]
                    plaintexts[int(transform_id)] = {
                        "payload": np.asarray(transform_group["diag_payload"][:], dtype=np.uint8).copy(),
                        "offsets": tuple(int(v) for v in transform_group["diag_offsets"][:].tolist()),
                        "lengths": tuple(int(v) for v in transform_group["diag_lengths"][:].tolist()),
                        "diag_indices": tuple(int(v) for v in diag_indices),
                    }
                bundle["plaintexts"] = plaintexts
            finally:
                handle.close()

        return bundle

    def _schedule_next_saved_io_prefetch(self, backend) -> None:
        if not self._should_offload_rotation_keys() and not self._offloaded_plaintext_diagonals:
            return
        self._io_prefetcher.submit(
            self._storage_key,
            lambda: self._read_saved_io_bundle(backend, prefetch=True),
        )

    def _load_rotation_keys(self, backend, bundle: dict[str, object] | None = None) -> None:
        if not self._should_offload_rotation_keys():
            return
        if bundle is not None:
            for key, serial_key in bundle.get("rotation_keys", ()):
                backend.LoadRotationKey(serial_key, int(key))
            return
        with h5py.File(self._keys_path, "r") as handle:
            for key, level in self._required_keys:
                key_name = self._rotation_key_storage_name(int(key), level)
                if key_name not in handle and str(int(key)) in handle:
                    key_name = str(int(key))
                backend.LoadRotationKey(handle[key_name][()], int(key))

    def _unload_rotation_keys(self, backend) -> None:
        if self._should_offload_rotation_keys():
            backend.RemoveRotationKeys()

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
            payload_required = self._plaintext_payload_required(backend)
            for transform_id, diag_indices in self._diag_indices_by_transform.items():
                transform_group = storage.create_group(str(transform_id))
                payload_chunks: list[np.ndarray] = []
                offsets: list[int] = []
                lengths: list[int] = []
                cursor = 0
                if payload_required:
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
        self._prefetch_host_bytes = None
        self._prefetch_device_bytes = None

    def _load_plaintext_diagonals(self, backend, bundle: dict[str, object] | None = None) -> None:
        if not self._offloaded_plaintext_diagonals:
            return
        if bundle is not None:
            for transform_id, payload in bundle.get("plaintexts", {}).items():
                backend.LoadPlaintextDiagonalsBatch(
                    payload["payload"],
                    list(payload["offsets"]),
                    list(payload["lengths"]),
                    list(payload["diag_indices"]),
                    int(transform_id),
                )
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
        self._prefetch_host_bytes = None
        self._prefetch_device_bytes = None
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

        required_keys: dict[int, int | None] = {}
        for transform_id in self.unified_ids:
            for key, level in self._rotation_key_requests(backend, int(transform_id)):
                if level is None:
                    required_keys[int(key)] = None
                else:
                    current = required_keys.get(int(key))
                    required_keys[int(key)] = int(level) if current is None else max(int(level), int(current))
        self._required_keys = tuple(sorted((int(key), level) for key, level in required_keys.items()))
        if self._should_offload_rotation_keys():
            with h5py.File(self._keys_path, "a") as handle:
                for key, level in self._required_keys:
                    key_name = self._rotation_key_storage_name(int(key), level)
                    if key_name in handle:
                        continue
                    serial_key, key_ptr = self._generate_and_serialize_rotation_key(backend, int(key), level)
                    try:
                        handle.create_dataset(key_name, data=serial_key)
                    finally:
                        backend.FreeCArray(key_ptr)
        else:
            for key, level in self._required_keys:
                self._generate_rotation_key(backend, int(key), level)

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
        bundle = self._io_prefetcher.consume(self._storage_key)
        if bundle is None and (self._should_offload_rotation_keys() or self._offloaded_plaintext_diagonals):
            bundle = self._read_saved_io_bundle(backend, prefetch=False)
        self._load_rotation_keys(backend, bundle)
        self._load_plaintext_diagonals(backend, bundle)
        self._schedule_next_saved_io_prefetch(backend)
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
            self._unload_rotation_keys(backend)

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
        self._io_prefetcher.clear(wait=True)
        if self.unified_ids is not None:
            for transform_id in self.unified_ids:
                backend.DeleteLinearTransform(int(transform_id))
        self._delete_offloaded_storage()
        self.unified_ids = None
        self.is_compiled = False
        self._diag_indices_by_transform = {}
        self._offloaded_plaintext_diagonals = False
        self._required_keys = ()
        self._prefetch_host_bytes = None
        self._prefetch_device_bytes = None


def can_use_unified_bsgs(layers: List) -> bool:
    from orion.nn.linear import LinearTransform

    if len(layers) < 2:
        return False
    return all(isinstance(layer, LinearTransform) and bool(getattr(layer, "diagonals", {})) for layer in layers)
