from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np

from orion.nn.unified_transform import UnifiedTransformGroup


class _Backend:
    def __init__(self) -> None:
        self.generated_keys: list[int] = []
        self.serialized: list[tuple[int, int]] = []
        self.encoded_serialized: list[int] = []
        self.encoded_loaded: list[tuple[int, tuple[int, ...]]] = []
        self.loaded_transform_keys: list[tuple[int, int, tuple[int, ...]]] = []
        self.rotation_keys = {11: [1, 3], 12: [1, 5]}
        self.load_plaintext_diagonals_requires_payload = False
        self.prefer_encoded_plaintext_payload_cache = True
        self.encoded_plaintext_payload_max_device_bytes = 1000
        self._next_transform_id = 11

    def GenerateLinearTransformsUnified(
        self,
        num_transforms,
        _diag_idxs_ptrs,
        _diag_idxs_lens,
        _diag_data_ptrs,
        _diag_data_lens,
        _levels_array,
    ):
        ids = [self._next_transform_id + i for i in range(int(num_transforms))]
        self._next_transform_id += int(num_transforms)
        return ids

    def GetLinearTransformRotationKeys(self, transform_id):
        return list(self.rotation_keys[int(transform_id)])

    def GenerateAndSerializeRotationKey(self, key):
        self.generated_keys.append(int(key))
        return np.asarray([int(key)], dtype=np.uint8), None

    def SerializeDiagonal(self, transform_id, diag_idx):
        self.serialized.append((int(transform_id), int(diag_idx)))
        return np.asarray([int(transform_id), int(diag_idx)], dtype=np.uint8), None

    def SerializeLinearTransformPlaintexts(self, transform_id):
        self.encoded_serialized.append(int(transform_id))
        return np.asarray([int(transform_id), 99], dtype=np.uint8), None

    def LoadLinearTransformPlaintexts(self, payload, transform_id):
        self.encoded_loaded.append(
            (int(transform_id), tuple(int(v) for v in np.asarray(payload).reshape(-1)))
        )

    def LoadLinearTransformRotationKey(self, serial_key, key, transform_id):
        self.loaded_transform_keys.append(
            (
                int(key),
                int(transform_id),
                tuple(int(v) for v in np.asarray(serial_key).reshape(-1)),
            )
        )

    def RemovePlaintextDiagonals(self, _transform_id):
        return None

    def RemoveLinearTransformRotationKeys(self, _transform_id):
        return None

    def EvaluateLinearTransformsWithSharedCache(self, transform_ids_array, num_transforms, ct_input_id):
        return [100 + i for i in range(int(num_transforms))]

    def EstimateLinearTransformDeviceBytes(self, _transform_id):
        return [90]

    def FreeCArray(self, _ptr):
        return None

    def DeleteLinearTransform(self, _transform_id):
        return None


def _transform(diagonals, *, io_mode: str, diags_path: str, keys_path: str):
    return SimpleNamespace(
        diagonals={(0, 0): dict(diagonals)},
        level=2,
        scheme=SimpleNamespace(
            params=SimpleNamespace(
                get_logq=lambda: [0, 1, 2, 3],
                get_io_mode=lambda: str(io_mode),
                get_diags_path=lambda: str(diags_path),
                get_keys_path=lambda: str(keys_path),
            )
        ),
    )


def _transforms(*, io_mode: str, diags_path: str, keys_path: str):
    return (
        _transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode=io_mode,
            diags_path=diags_path,
            keys_path=keys_path,
        ),
        _transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode=io_mode,
            diags_path=diags_path,
            keys_path=keys_path,
        ),
    )


def test_unified_transform_load_mode_backfills_missing_rotation_keys(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    keys_path = tmp_path / "unified_keys.h5"
    save_group = UnifiedTransformGroup(
        _transforms(io_mode="save", diags_path=str(diags_path), keys_path=str(keys_path))
    )
    save_group.compile_unified(_Backend())
    with h5py.File(keys_path, "a") as handle:
        del handle["5"]

    load_backend = _Backend()
    load_group = UnifiedTransformGroup(
        _transforms(io_mode="load", diags_path=str(diags_path), keys_path=str(keys_path))
    )
    load_group._storage_key = save_group._storage_key

    load_group.compile_unified(load_backend)

    assert load_backend.generated_keys == [5]
    assert load_backend.encoded_serialized == []
    assert load_backend.serialized == []
    with h5py.File(keys_path, "r") as handle:
        assert "5" in handle

    assert load_group.evaluate_unified(7, load_backend) == [100, 101]
    assert (5, 11, (5,)) in load_backend.loaded_transform_keys
    assert (5, 12, (5,)) in load_backend.loaded_transform_keys
