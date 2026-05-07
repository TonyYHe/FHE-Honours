from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.nn import unified_transform as unified_transform_module
from orion.nn.unified_transform import UnifiedTransformGroup, can_use_unified_bsgs


class _FakeBackend:
    def __init__(self) -> None:
        self.generated = []
        self.evaluated = []
        self.deleted = []
        self.serialized = []
        self.loaded_batches = []
        self.removed_plaintext_diagonals = []
        self.released_matrices = []
        self.loaded_transform_keys = []
        self.removed_transform_keys = []
        self.rotation_keys = {11: [1, 3], 12: [1, 5]}
        self.generated_keys = []
        self.next_transform_id = 11
        self.streaming_transform_ids = set()

    def GenerateLinearTransformsUnified(self, num_transforms, diag_idxs_ptrs, diag_idxs_lens, diag_data_ptrs, diag_data_lens, levels_array):
        self.generated.append(
            {
                "num_transforms": int(num_transforms),
                "diag_lengths": [int(diag_idxs_lens[i]) for i in range(int(num_transforms))],
                "data_lengths": [int(diag_data_lens[i]) for i in range(int(num_transforms))],
                "levels": [int(levels_array[i]) for i in range(int(num_transforms))],
            }
        )
        ids = [int(self.next_transform_id + i) for i in range(int(num_transforms))]
        self.next_transform_id += int(num_transforms)
        return ids

    def GetLinearTransformRotationKeys(self, transform_id):
        return list(self.rotation_keys[int(transform_id)])

    def GenerateLinearTransformRotationKey(self, key):
        self.generated_keys.append(int(key))

    def GenerateAndSerializeRotationKey(self, key):
        self.generated_keys.append(int(key))
        return np.asarray([int(key)], dtype=np.uint8), None

    def EvaluateLinearTransformsWithSharedCache(self, transform_ids_array, num_transforms, ct_input_id):
        ids = [int(transform_ids_array[i]) for i in range(int(num_transforms))]
        self.evaluated.append((ids, int(ct_input_id)))
        return [100 + i for i in range(int(num_transforms))]

    def SerializeDiagonal(self, transform_id, diag_idx):
        self.serialized.append((int(transform_id), int(diag_idx)))
        return np.asarray([int(transform_id), int(diag_idx)], dtype=np.uint8), None

    def FreeCArray(self, _ptr):
        return None

    def LoadPlaintextDiagonalsBatch(self, payload, offsets, lengths, diag_indices, transform_id):
        payload_arr = np.asarray(payload, dtype=np.uint8).reshape(-1)
        segments = []
        for offset, length in zip(offsets, lengths):
            start = int(offset)
            end = int(start + length)
            segments.append(tuple(int(v) for v in payload_arr[start:end]))
        self.loaded_batches.append(
            {
                "transform_id": int(transform_id),
                "diag_indices": [int(value) for value in diag_indices],
                "segments": segments,
            }
        )

    def RemovePlaintextDiagonals(self, transform_id):
        self.removed_plaintext_diagonals.append(int(transform_id))

    def ReleaseLinearTransformMatrix(self, transform_id):
        self.released_matrices.append(int(transform_id))

    def LoadLinearTransformRotationKey(self, serial_key, key, transform_id):
        self.loaded_transform_keys.append((int(key), int(transform_id), tuple(int(v) for v in np.asarray(serial_key).reshape(-1))))

    def RemoveLinearTransformRotationKeys(self, transform_id):
        self.removed_transform_keys.append(int(transform_id))

    def EstimateLinearTransformDeviceBytes(self, transform_id):
        return [90]

    def LinearTransformUsesStreaming(self, transform_id):
        return int(transform_id) in self.streaming_transform_ids

    def GetDeviceMemoryInfo(self):
        return [200, 400]

    def DeleteLinearTransform(self, transform_id):
        self.deleted.append(int(transform_id))


class _FakeUnifiedLoadBackend(_FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_shells = []

    def GenerateLinearTransformsUnifiedLoad(self, num_transforms, diag_idxs_ptrs, diag_idxs_lens, levels_array):
        self.loaded_shells.append(
            {
                "num_transforms": int(num_transforms),
                "diag_lengths": [int(diag_idxs_lens[i]) for i in range(int(num_transforms))],
                "levels": [int(levels_array[i]) for i in range(int(num_transforms))],
            }
        )
        ids = [int(self.next_transform_id + i) for i in range(int(num_transforms))]
        self.next_transform_id += int(num_transforms)
        return ids


class _FakeSharedRotationBackend(_FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        gib = 1024**3
        self.memory_bounded_unified_transforms = True
        self.memory_bounded_unified_evaluate = True
        self.retain_unified_rotation_keys = True
        self.unified_transform_eval_budget_bytes = 1000
        self.loaded_shared_keys = []
        self.removed_shared_keys = 0
        self._trim_seconds = 0.0
        self._profile_seconds = [0.0] * 6
        self._device_memory_info = [90 * gib, 96 * gib]

    def LoadRotationKey(self, serial_key, key):
        self.loaded_shared_keys.append((int(key), tuple(int(v) for v in np.asarray(serial_key).reshape(-1))))

    def RemoveRotationKeys(self):
        self.removed_shared_keys += 1

    def ConsumeDeviceMemoryTrimSeconds(self):
        value = float(self._trim_seconds)
        self._trim_seconds = 0.0
        return value

    def ConsumeSharedCacheEvalProfileSeconds(self):
        values = list(self._profile_seconds)
        self._profile_seconds = [0.0] * 6
        return values

    def EvaluateLinearTransformsWithSharedCache(self, transform_ids_array, num_transforms, ct_input_id):
        self._trim_seconds += 0.25
        self._profile_seconds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.25]
        return super().EvaluateLinearTransformsWithSharedCache(transform_ids_array, num_transforms, ct_input_id)

    def GetDeviceMemoryInfo(self):
        return list(self._device_memory_info)


class _FakeEncodedPayloadBackend(_FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.load_plaintext_diagonals_requires_payload = False
        self.prefer_encoded_plaintext_payload_cache = True
        self.encoded_plaintext_payload_max_device_bytes = 1000
        self.encoded_serialized = []
        self.encoded_loaded = []

    def SerializeLinearTransformPlaintexts(self, transform_id):
        self.encoded_serialized.append(int(transform_id))
        return np.asarray([int(transform_id), 99], dtype=np.uint8), None

    def LoadLinearTransformPlaintexts(self, payload, transform_id):
        self.encoded_loaded.append((int(transform_id), tuple(int(v) for v in np.asarray(payload).reshape(-1))))


class _FakeResidentEncodedPayloadBackend(_FakeEncodedPayloadBackend):
    def __init__(self) -> None:
        super().__init__()
        gib = 1024**3
        self.retain_unified_plaintexts = True
        self._device_memory_info = [80 * gib, 96 * gib]

    def GetDeviceMemoryInfo(self):
        return list(self._device_memory_info)


class _FakeForcedStreamingEncodedPayloadBackend(_FakeEncodedPayloadBackend):
    def LinearTransformUsesStreaming(self, transform_id):
        return True


class _FakeSavedIOScheduler:
    def __init__(self) -> None:
        self.registered = []
        self.unregistered = []

    def register_saved_io_prefetch_work_unit(self, key, *, loader, host_bytes, device_bytes) -> None:
        self.registered.append(
            {
                "key": key,
                "loader": loader,
                "host_bytes": host_bytes,
                "device_bytes": device_bytes,
            }
        )

    def unregister_saved_io_prefetch_work_unit(self, key) -> None:
        self.unregistered.append(key)

    def fill_saved_io_prefetch_window(self, key, *, include_current=True, scratch_reserve_bytes=0) -> bool:
        return False

    def consume_saved_io_prefetch(self, key):
        return None


def _fake_transform(
    diagonals,
    *,
    level=2,
    io_mode="none",
    diags_path="",
    keys_path="",
    compile_save_resume=False,
):
    return SimpleNamespace(
        diagonals={(0, 0): dict(diagonals)},
        level=int(level),
        scheme=SimpleNamespace(
            params=SimpleNamespace(
                get_logq=lambda: [0, 1, 2, 3],
                get_io_mode=lambda: str(io_mode),
                get_diags_path=lambda: str(diags_path),
                get_keys_path=lambda: str(keys_path),
                get_compile_save_resume=lambda: bool(compile_save_resume),
            )
        ),
    )


def test_unified_transform_group_compiles_and_evaluates_with_fake_backend() -> None:
    transforms = (
        _fake_transform({0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]}),
        _fake_transform({0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]}),
    )
    backend = _FakeBackend()
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)
    outputs = group.evaluate_unified(7, backend)

    assert group.is_compiled is True
    assert group.unified_ids == [11, 12]
    assert backend.generated == [
        {
            "num_transforms": 2,
            "diag_lengths": [2, 2],
            "data_lengths": [8, 8],
            "levels": [2, 2],
        }
    ]
    assert sorted(backend.generated_keys) == [1, 3, 5]
    assert outputs == [100, 101]
    assert backend.evaluated == [([11, 12], 7)]
    assert group.get_transform_ids(transforms[0]) == {(0, 0): 11}

    group.cleanup(backend)
    assert backend.deleted == [11, 12]
    assert group.is_compiled is False


def test_unified_transform_group_requires_diagonals() -> None:
    backend = _FakeBackend()
    group = UnifiedTransformGroup([_fake_transform({})])

    with pytest.raises(ValueError, match="generated diagonals"):
        group.compile_unified(backend)


def test_unified_transform_group_offloads_plaintext_diagonals_when_saving(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    transforms = (
        _fake_transform({0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
        _fake_transform({0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
    )
    backend = _FakeBackend()
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)

    assert group.unified_ids == [11, 12]
    assert backend.serialized == [(11, 0), (11, 1), (12, 0), (12, 2)]
    assert backend.removed_plaintext_diagonals == [11, 12]
    with h5py.File(diags_path, "r") as handle:
        root = handle["__unified_transform_groups__"]
        assert len(root.keys()) == 1

    outputs = group.evaluate_unified(7, backend)

    assert outputs == [100, 101]
    assert backend.loaded_batches == [
        {
            "transform_id": 11,
            "diag_indices": [0, 1],
            "segments": [(11, 0), (11, 1)],
        },
        {
            "transform_id": 12,
            "diag_indices": [0, 2],
            "segments": [(12, 0), (12, 2)],
        },
    ]
    assert backend.removed_plaintext_diagonals == [11, 12, 11, 12]

    group.cleanup(backend)

    with h5py.File(diags_path, "r") as handle:
        assert "__unified_transform_groups__" in handle
        assert len(handle["__unified_transform_groups__"].keys()) == 0


def test_unified_transform_group_streams_plaintext_save_without_concatenate(tmp_path, monkeypatch) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    backend = _FakeBackend()
    group = UnifiedTransformGroup([])

    def fail_concatenate(*_args, **_kwargs):
        raise AssertionError("plaintext save should not concatenate all diagonal payloads")

    monkeypatch.setattr(unified_transform_module.np, "concatenate", fail_concatenate)
    with h5py.File(diags_path, "w") as handle:
        storage = handle.create_group("payloads")
        group._save_and_unload_plaintext_diagonals_for_transform(
            backend,
            storage,
            11,
            (0, 1, 2),
        )

    assert backend.serialized == [(11, 0), (11, 1), (11, 2)]
    assert backend.removed_plaintext_diagonals == [11]
    with h5py.File(diags_path, "r") as handle:
        transform_group = handle["payloads"]["11"]
        assert transform_group["diag_indices"][:].tolist() == [0, 1, 2]
        assert transform_group["diag_offsets"][:].tolist() == [0, 2, 4]
        assert transform_group["diag_lengths"][:].tolist() == [2, 2, 2]
        assert transform_group["diag_payload"][:].tolist() == [11, 0, 11, 1, 11, 2]


def test_unified_transform_group_streams_compile_and_chunks_eval_for_memory_bounded_backend(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    keys_path = tmp_path / "unified_keys.h5"
    transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
    )
    backend = _FakeBackend()
    backend.memory_bounded_unified_transforms = True
    backend.memory_bounded_unified_evaluate = True
    backend.unified_transform_eval_budget_bytes = 100
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)

    assert group.unified_ids == [11, 12]
    assert [entry["num_transforms"] for entry in backend.generated] == [1, 1]
    assert backend.serialized == [(11, 0), (11, 1), (12, 0), (12, 2)]
    assert backend.removed_plaintext_diagonals == [11, 12]
    assert sorted(backend.generated_keys) == [1, 3, 5]

    outputs = group.evaluate_unified(7, backend)

    assert outputs == [100, 100]
    assert backend.evaluated == [([11], 7), ([12], 7)]
    assert backend.loaded_batches == [
        {
            "transform_id": 11,
            "diag_indices": [0, 1],
            "segments": [(11, 0), (11, 1)],
        },
        {
            "transform_id": 12,
            "diag_indices": [0, 2],
            "segments": [(12, 0), (12, 2)],
        },
    ]
    assert backend.removed_plaintext_diagonals == [11, 12, 11, 12]
    assert backend.removed_transform_keys == [11, 12]
    assert any(event["event"] == "after_offload_transform" for event in group.memory_trace)
    assert any(event["event"] == "before_eval_group_memory_bounded" for event in group.memory_trace)


def test_memory_bounded_compile_batches_transforms_when_workers_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_UNIFIED_COMPILE_WORKERS", "2")
    monkeypatch.setenv("ORION_UNIFIED_STREAM_COMPILE_BATCH_TRANSFORMS", "2")
    monkeypatch.setenv("ORION_UNIFIED_STREAM_COMPILE_BATCH_BYTES", "0")
    diags_path = tmp_path / "unified_diagonals.h5"
    keys_path = tmp_path / "unified_keys.h5"
    transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
    )
    backend = _FakeBackend()
    backend.memory_bounded_unified_transforms = True
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)

    assert group.unified_ids == [11, 12]
    assert [entry["num_transforms"] for entry in backend.generated] == [2]
    assert backend.serialized == [(11, 0), (11, 1), (12, 0), (12, 2)]
    assert backend.removed_plaintext_diagonals == [11, 12]
    batch_events = [
        event for event in group.memory_trace
        if event["event"] == "before_compile_transform_batch"
    ]
    assert batch_events
    assert batch_events[-1]["batch_size"] == 2


def test_memory_bounded_load_streams_compile_to_match_saved_payload_layout(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    save_transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
        ),
    )
    save_backend = _FakeBackend()
    save_backend.memory_bounded_unified_transforms = True
    save_group = UnifiedTransformGroup(save_transforms)
    save_group.compile_unified(save_backend)

    load_transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="load",
            diags_path=str(diags_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="load",
            diags_path=str(diags_path),
        ),
    )
    load_backend = _FakeBackend()
    load_backend.memory_bounded_unified_transforms = True
    load_backend.memory_bounded_unified_evaluate = True
    load_backend.unified_transform_eval_budget_bytes = 100
    load_group = UnifiedTransformGroup(load_transforms)
    load_group._storage_key = save_group._storage_key

    load_group.compile_unified(load_backend)

    assert [entry["num_transforms"] for entry in load_backend.generated] == [1, 1]
    assert load_backend.serialized == []
    assert load_backend.removed_plaintext_diagonals == [11, 12]
    assert load_group._offloaded_plaintext_diagonals is True

    assert load_group.evaluate_unified(7, load_backend) == [100, 100]
    assert load_backend.loaded_batches == [
        {
            "transform_id": 11,
            "diag_indices": [0, 1],
            "segments": [(11, 0), (11, 1)],
        },
        {
            "transform_id": 12,
            "diag_indices": [0, 2],
            "segments": [(12, 0), (12, 2)],
        },
    ]


def test_save_resume_reuses_complete_unified_group(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    save_transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
        ),
    )
    save_backend = _FakeBackend()
    save_backend.memory_bounded_unified_transforms = True
    save_group = UnifiedTransformGroup(save_transforms)
    save_group.compile_unified(save_backend)

    resume_transforms = (
        _fake_transform(
            {99: [9.0, 9.0, 9.0, 9.0]},
            io_mode="save",
            diags_path=str(diags_path),
            compile_save_resume=True,
        ),
        _fake_transform(
            {101: [10.0, 10.0, 10.0, 10.0]},
            io_mode="save",
            diags_path=str(diags_path),
            compile_save_resume=True,
        ),
    )
    resume_backend = _FakeUnifiedLoadBackend()
    resume_backend.memory_bounded_unified_transforms = True
    resume_group = UnifiedTransformGroup(resume_transforms)
    resume_group._storage_key = save_group._storage_key

    resume_group.compile_unified(resume_backend)

    assert resume_backend.loaded_shells == [
        {"num_transforms": 2, "diag_lengths": [2, 2], "levels": [2, 2]}
    ]
    assert resume_backend.generated == []
    assert resume_backend.serialized == []
    assert resume_transforms[0].diagonals == {}
    assert resume_transforms[1].diagonals == {}


def test_memory_bounded_eval_schedule_can_change_without_recompile(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    keys_path = tmp_path / "unified_keys.h5"
    transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
    )
    backend = _FakeBackend()
    backend.memory_bounded_unified_transforms = True
    backend.memory_bounded_unified_evaluate = True
    backend.unified_transform_eval_budget_bytes = 100
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)
    first_outputs = group.evaluate_unified(7, backend)
    backend.unified_transform_eval_budget_bytes = 1000
    second_outputs = group.evaluate_unified(8, backend)

    assert first_outputs == [100, 100]
    assert second_outputs == [100, 101]
    assert [entry["num_transforms"] for entry in backend.generated] == [1, 1]
    assert backend.evaluated == [([11], 7), ([12], 7), ([11, 12], 8)]


def test_unified_transform_group_preserves_streaming_backend_group_for_memory_bounded_eval(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    keys_path = tmp_path / "unified_keys.h5"
    transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
    )
    backend = _FakeBackend()
    backend.memory_bounded_unified_transforms = True
    backend.memory_bounded_unified_evaluate = True
    backend.unified_transform_eval_budget_bytes = 100
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)
    backend.streaming_transform_ids = set(group.unified_ids)
    outputs = group.evaluate_unified(7, backend)

    assert outputs == [100, 101]
    assert backend.evaluated == [([11, 12], 7)]
    assert backend.loaded_batches == []
    assert sorted(backend.loaded_transform_keys) == [
        (1, 11, (1,)),
        (1, 12, (1,)),
        (3, 11, (3,)),
        (5, 11, (5,)),
        (5, 12, (5,)),
    ]
    assert backend.removed_plaintext_diagonals == [11, 12]
    eval_events = [event for event in group.memory_trace if event["event"] == "before_eval_group_memory_bounded"]
    assert eval_events[-1]["chunk_count"] == 1


def test_memory_bounded_eval_records_timing_and_retains_shared_rotation_keys(tmp_path) -> None:
    UnifiedTransformGroup._resident_shared_rotation_keys.clear()
    diags_path = tmp_path / "unified_diagonals.h5"
    keys_path = tmp_path / "unified_keys.h5"
    transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
    )
    backend = _FakeSharedRotationBackend()
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)
    assert group.evaluate_unified(7, backend) == [100, 101]
    assert group.evaluate_unified(8, backend) == [100, 101]

    assert sorted(backend.loaded_shared_keys) == [(1, (1,)), (3, (3,)), (5, (5,))]
    assert backend.removed_shared_keys == 0
    assert backend.removed_plaintext_diagonals == [11, 12, 11, 12, 11, 12]
    eval_events = [event for event in group.memory_trace if event["event"] == "after_eval_chunk_unload"]
    assert eval_events
    timing = eval_events[-1]["timing"]
    assert set(timing) >= {
        "read_bundle_s",
        "load_keys_s",
        "load_plaintexts_s",
        "eval_s",
        "eval_total_s",
        "unload_s",
        "trim_s",
        "cpp_plan_s",
        "cpp_level_adjust_s",
        "cpp_baby_step_s",
        "cpp_giant_step_s",
        "cpp_push_s",
        "cpp_trim_s",
    }
    assert timing["trim_s"] == pytest.approx(0.25)
    assert timing["cpp_giant_step_s"] == pytest.approx(0.04)
    assert timing["cpp_trim_s"] == pytest.approx(0.25)
    group_events = [event for event in group.memory_trace if event["event"] == "after_eval_group_memory_bounded"]
    assert sum(float(event["timing"]["trim_s"]) for event in group_events) == pytest.approx(0.5)


def test_cheddar_like_backend_prefers_encoded_plaintext_payload_cache(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
        ),
    )
    backend = _FakeEncodedPayloadBackend()
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)
    assert backend.encoded_serialized == [11, 12]
    assert backend.released_matrices == [11, 12]
    assert backend.serialized == []
    with h5py.File(diags_path, "r") as handle:
        root = handle["__unified_transform_groups__"]
        storage = next(iter(root.values()))
        assert "__encoded_hoist_payload__" in storage["11"]
        assert "diag_payload" not in storage["11"]

    assert group.evaluate_unified(7, backend) == [100, 101]
    assert backend.encoded_loaded == [(11, (11, 99)), (12, (12, 99))]
    assert backend.loaded_batches == []


def test_unified_transform_save_mode_keeps_plaintexts_resident_when_memory_allows(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
        ),
    )
    backend = _FakeResidentEncodedPayloadBackend()
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)

    assert backend.encoded_serialized == [11, 12]
    assert backend.released_matrices == []
    assert group._resident_plaintext_transform_ids == {11, 12}
    assert group.evaluate_unified(7, backend) == [100, 101]
    assert backend.encoded_loaded == []
    assert backend.removed_plaintext_diagonals == []


def test_streaming_transforms_skip_encoded_payload_cache(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    transforms = (
        _fake_transform({0: [1.0, 0.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
        _fake_transform({0: [2.0, 0.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
    )
    backend = _FakeForcedStreamingEncodedPayloadBackend()
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)

    assert backend.encoded_serialized == []
    assert backend.released_matrices == []
    with h5py.File(diags_path, "r") as handle:
        storage = handle["__unified_transform_groups__"][group._storage_key]
        assert "__encoded_hoist_payload__" not in storage["11"]
        assert storage["11"]["diag_payload"].shape == (0,)


def test_unified_transform_load_mode_uses_saved_payloads_and_keys_without_regenerating(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    keys_path = tmp_path / "unified_keys.h5"
    save_transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
    )
    save_backend = _FakeEncodedPayloadBackend()
    save_group = UnifiedTransformGroup(save_transforms)
    save_group.compile_unified(save_backend)

    load_transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="load",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="load",
            diags_path=str(diags_path),
            keys_path=str(keys_path),
        ),
    )
    load_backend = _FakeEncodedPayloadBackend()
    load_group = UnifiedTransformGroup(load_transforms)
    load_group._storage_key = save_group._storage_key

    load_group.compile_unified(load_backend)

    assert load_backend.encoded_serialized == []
    assert load_backend.serialized == []
    assert load_backend.generated_keys == []
    assert load_backend.released_matrices == []
    assert load_group._offloaded_plaintext_diagonals is True

    assert load_group.evaluate_unified(7, load_backend) == [100, 101]
    assert load_backend.encoded_loaded == [(11, (11, 99)), (12, (12, 99))]
    assert sorted(load_backend.loaded_transform_keys) == [
        (1, 11, (1,)),
        (1, 12, (1,)),
        (3, 11, (3,)),
        (5, 11, (5,)),
        (5, 12, (5,)),
    ]

    load_group.cleanup(load_backend)
    with h5py.File(diags_path, "r") as handle:
        assert save_group._storage_key in handle["__unified_transform_groups__"]


def test_unified_transform_compile_only_can_release_index_only_matrices(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_UNIFIED_LT_RELEASE_INDEX_ONLY_RAW_MATRICES_AFTER_SAVE", "1")
    diags_path = tmp_path / "unified_diagonals.h5"
    transforms = (
        _fake_transform({0: [1.0, 0.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
        _fake_transform({0: [2.0, 0.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
    )
    backend = _FakeBackend()
    backend.load_plaintext_diagonals_requires_payload = False
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)

    assert backend.removed_plaintext_diagonals == [11, 12]
    assert backend.released_matrices == [11, 12]
    assert group.evaluate_unified(7, backend) == [100, 101]


def test_unified_transform_group_registers_shared_saved_io_prefetch(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    scheduler = _FakeSavedIOScheduler()
    transforms = (
        _fake_transform({0: [1.0, 0.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
        _fake_transform({0: [2.0, 0.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
    )
    for transform in transforms:
        transform.scheme.lt_evaluator = scheduler

    backend = _FakeBackend()
    group = UnifiedTransformGroup(transforms)
    group.compile_unified(backend)

    assert len(scheduler.registered) == 1
    assert scheduler.registered[0]["key"] == ("unified", group._storage_key)
    assert callable(scheduler.registered[0]["loader"])

    group.cleanup(backend)
    assert scheduler.unregistered == [("unified", group._storage_key)]


def test_memory_bounded_unified_group_skips_full_group_prefetch(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    scheduler = _FakeSavedIOScheduler()
    transforms = (
        _fake_transform({0: [1.0, 0.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
        _fake_transform({0: [2.0, 0.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
    )
    for transform in transforms:
        transform.scheme.lt_evaluator = scheduler

    backend = _FakeBackend()
    backend.memory_bounded_unified_transforms = True
    backend.memory_bounded_unified_evaluate = True
    backend.unified_transform_eval_budget_bytes = 100
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)
    outputs = group.evaluate_unified(7, backend)

    assert outputs == [100, 100]
    assert scheduler.registered == []
    assert backend.evaluated == [([11], 7), ([12], 7)]


def test_unified_transform_group_skips_payloads_when_backend_does_not_need_them(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    transforms = (
        _fake_transform({0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
        _fake_transform({0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]}, io_mode="save", diags_path=str(diags_path)),
    )
    backend = _FakeBackend()
    backend.load_plaintext_diagonals_requires_payload = False
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)

    assert backend.serialized == []
    with h5py.File(diags_path, "r") as handle:
        storage = handle["__unified_transform_groups__"][group._storage_key]
        assert storage["11"]["diag_payload"].shape == (0,)
        assert storage["12"]["diag_payload"].shape == (0,)

    outputs = group.evaluate_unified(7, backend)

    assert outputs == [100, 101]
    assert backend.loaded_batches == [
        {
            "transform_id": 11,
            "diag_indices": [0, 1],
            "segments": [],
        },
        {
            "transform_id": 12,
            "diag_indices": [0, 2],
            "segments": [],
        },
    ]


def test_can_use_unified_bsgs_rejects_non_linear_transform_instances() -> None:
    assert can_use_unified_bsgs([_fake_transform({0: [1.0]})]) is False


def test_unified_transform_group_runs_on_lattigo_backend() -> None:
    shared_library = Path("orion/backend/lattigo/lattigo-linux.so")
    if not shared_library.exists():
        pytest.skip("local Lattigo shared library has not been built")

    config = {
        "ckks_params": {
            "LogN": 12,
            "LogQ": [45, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    try:
        slots = int(scheme.params.get_slots())
        level = len(scheme.params.get_logq()) - 1
        identity = [1.0] * slots
        doubled = [2.0] * slots
        transform_a = SimpleNamespace(
            diagonals={(0, 0): {0: identity}},
            level=level,
            scheme=scheme,
            fhe_output_shape=torch.Size([1, slots]),
            output_shape=torch.Size([1, slots]),
        )
        transform_b = SimpleNamespace(
            diagonals={(0, 0): {0: doubled}},
            level=level,
            scheme=scheme,
            fhe_output_shape=torch.Size([1, slots]),
            output_shape=torch.Size([1, slots]),
        )
        group = UnifiedTransformGroup([transform_a, transform_b])
        group.compile_unified(scheme.backend)

        x = torch.zeros(slots, dtype=torch.float32)
        x[:8] = torch.linspace(0.1, 0.8, 8)
        ct = scheme.encrypt(scheme.encode(x, level))
        output_ids = group.evaluate_unified(ct.ids[0], scheme.backend)
        decoded = []
        for output_id in output_ids:
            out_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, slots]), torch.Size([1, slots]))
            decoded.append(out_ct.decrypt().decode().reshape(-1))

        assert len(decoded) == 2
        assert float((decoded[0][:8] - x[:8]).abs().max()) <= 1.0e-4
        assert float((decoded[1][:8] - 2.0 * x[:8]).abs().max()) <= 1.0e-4
    finally:
        scheme.delete_scheme()


def test_unified_transform_group_runs_on_lattigo_backend_with_saved_diagonals(tmp_path) -> None:
    shared_library = Path("orion/backend/lattigo/lattigo-linux.so")
    if not shared_library.exists():
        pytest.skip("local Lattigo shared library has not been built")

    config = {
        "ckks_params": {
            "LogN": 12,
            "LogQ": [45, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "save",
            "diags_path": str(tmp_path / "unified_diags.h5"),
            "keys_path": str(tmp_path / "unified_keys.h5"),
        },
    }
    scheme.init_scheme(config)
    try:
        slots = int(scheme.params.get_slots())
        level = len(scheme.params.get_logq()) - 1
        identity = [1.0] * slots
        doubled = [2.0] * slots
        transform_a = SimpleNamespace(
            diagonals={(0, 0): {0: identity}},
            level=level,
            scheme=scheme,
            fhe_output_shape=torch.Size([1, slots]),
            output_shape=torch.Size([1, slots]),
        )
        transform_b = SimpleNamespace(
            diagonals={(0, 0): {0: doubled}},
            level=level,
            scheme=scheme,
            fhe_output_shape=torch.Size([1, slots]),
            output_shape=torch.Size([1, slots]),
        )
        group = UnifiedTransformGroup([transform_a, transform_b])
        group.compile_unified(scheme.backend)

        x = torch.zeros(slots, dtype=torch.float32)
        x[:8] = torch.linspace(0.1, 0.8, 8)
        ct = scheme.encrypt(scheme.encode(x, level))
        output_ids = group.evaluate_unified(ct.ids[0], scheme.backend)
        decoded = []
        for output_id in output_ids:
            out_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, slots]), torch.Size([1, slots]))
            decoded.append(out_ct.decrypt().decode().reshape(-1))

        assert len(decoded) == 2
        assert float((decoded[0][:8] - x[:8]).abs().max()) <= 1.0e-4
        assert float((decoded[1][:8] - 2.0 * x[:8]).abs().max()) <= 1.0e-4
    finally:
        scheme.delete_scheme()
