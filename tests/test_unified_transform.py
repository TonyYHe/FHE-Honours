from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.experimental.cir.hybrid_schedule import (
    hybrid_pair_schedule_compatible,
    hybrid_schedule_signature,
    mark_hybrid_schedule_padding_allowed,
    materialize_hybrid_pair_layout_schedules,
    optimize_hybrid_pair_layout,
    pad_hybrid_pair_to_common_schedule,
)
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
        self.source_target_sum_evaluated = []
        self.ct_levels = {}
        self.ct_scales = {}
        self.ct_slots = {}
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

    def PlanLinearTransformsUnifiedRotationKeys(self, num_transforms, diag_idxs_ptrs, diag_idxs_lens, levels_array):
        keys = set()
        for transform_index in range(int(num_transforms)):
            for diag_index in range(int(diag_idxs_lens[transform_index])):
                keys.add(int(diag_idxs_ptrs[transform_index][diag_index]))
        return sorted(keys)

    def GenerateLinearTransformRotationKey(self, key):
        self.generated_keys.append(int(key))

    def GenerateAndSerializeRotationKey(self, key):
        self.generated_keys.append(int(key))
        return np.asarray([int(key)], dtype=np.uint8), None

    def EvaluateLinearTransformsWithSharedCache(self, transform_ids_array, num_transforms, ct_input_id):
        ids = [int(transform_ids_array[i]) for i in range(int(num_transforms))]
        self.evaluated.append((ids, int(ct_input_id)))
        return [100 + i for i in range(int(num_transforms))]

    def EvaluateLinearTransformSourcesWithSharedCacheAdd(
        self,
        ctxt_ids_array,
        num_sources,
        transform_ids_array,
        target_ids_array,
        group_offsets_array,
        num_partials,
        num_targets,
    ):
        record = {
            "ctxt_ids": [int(ctxt_ids_array[i]) for i in range(int(num_sources))],
            "transform_ids": [int(transform_ids_array[i]) for i in range(int(num_partials))],
            "target_ids": [int(target_ids_array[i]) for i in range(int(num_partials))],
            "group_offsets": [int(group_offsets_array[i]) for i in range(int(num_sources) + 1)],
            "target_count": int(num_targets),
        }
        self.source_target_sum_evaluated.append(record)
        return [200 + i for i in range(int(num_targets))]

    def GetCiphertextLevel(self, ciphertext_id):
        return int(self.ct_levels.get(int(ciphertext_id), 2))

    def GetCiphertextScale(self, ciphertext_id):
        return int(self.ct_scales.get(int(ciphertext_id), 1 << 30))

    def GetCiphertextSlots(self, ciphertext_id):
        return int(self.ct_slots.get(int(ciphertext_id), 8))

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

    def register_saved_io_prefetch_work_unit(
        self,
        key,
        *,
        loader,
        host_bytes,
        device_bytes,
        raw_loader=None,
        raw_host_bytes=None,
        predecode_loader=None,
    ) -> None:
        self.registered.append(
            {
                "key": key,
                "loader": loader,
                "host_bytes": host_bytes,
                "device_bytes": device_bytes,
                "raw_loader": raw_loader,
                "raw_host_bytes": raw_host_bytes,
                "predecode_loader": predecode_loader,
            }
        )

    def unregister_saved_io_prefetch_work_unit(self, key) -> None:
        self.unregistered.append(key)

    def fill_saved_io_prefetch_window(self, key, *, include_current=True, scratch_reserve_bytes=0) -> bool:
        return False

    def consume_saved_io_prefetch(self, key):
        return None


class _SingleSlotTrackingBackend(_FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.live_plaintext_transforms: set[int] = set()
        self.live_before_generate: list[tuple[int, ...]] = []
        self.max_live_plaintext_transforms = 0

    def GenerateLinearTransformsUnified(
        self,
        num_transforms,
        diag_idxs_ptrs,
        diag_idxs_lens,
        diag_data_ptrs,
        diag_data_lens,
        levels_array,
    ):
        self.live_before_generate.append(tuple(sorted(self.live_plaintext_transforms)))
        ids = super().GenerateLinearTransformsUnified(
            num_transforms,
            diag_idxs_ptrs,
            diag_idxs_lens,
            diag_data_ptrs,
            diag_data_lens,
            levels_array,
        )
        for transform_id in ids:
            self.live_plaintext_transforms.add(int(transform_id))
        self.max_live_plaintext_transforms = max(
            int(self.max_live_plaintext_transforms),
            int(len(self.live_plaintext_transforms)),
        )
        return ids

    def EvaluateLinearTransformsWithSharedCache(self, transform_ids_array, num_transforms, ct_input_id):
        ids = [int(transform_ids_array[i]) for i in range(int(num_transforms))]
        assert set(ids).issubset(self.live_plaintext_transforms)
        return super().EvaluateLinearTransformsWithSharedCache(transform_ids_array, num_transforms, ct_input_id)

    def DeleteLinearTransform(self, transform_id):
        super().DeleteLinearTransform(transform_id)
        self.live_plaintext_transforms.discard(int(transform_id))


def _fake_transform(
    diagonals,
    *,
    level=2,
    io_mode="none",
    diags_path="",
    keys_path="",
    compile_save_resume=False,
):
    source_diagonals = {(0, 0): dict(diagonals)}
    stored_diagonals = {(int(row), int(col)): dict(block) for (row, col), block in source_diagonals.items()}
    return SimpleNamespace(
        diagonals={(int(row), int(col)): dict(block) for (row, col), block in source_diagonals.items()},
        _single_slot_diag_indices_by_block={(0, 0): tuple(sorted(int(index) for index in dict(diagonals).keys()))},
        _single_slot_build_diagonals=lambda stored_diagonals=stored_diagonals: {
            (int(row), int(col)): dict(block)
            for (row, col), block in stored_diagonals.items()
        },
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


def test_hybrid_schedule_signature_normalizes_diagonal_keys_mod_slots() -> None:
    transform = _fake_transform({-1: [1.0], 9: [2.0], 2: [3.0]})

    signature = hybrid_schedule_signature(transform, slots=8)

    assert signature.slots == 8
    assert signature.normalized_diagonal_keys == (1, 2, 7)


def test_hybrid_pair_schedule_compatibility_requires_matching_support() -> None:
    left = _fake_transform({0: [1.0], 3: [2.0], 9: [3.0]})
    same = _fake_transform({0: [4.0], 1: [5.0], 3: [6.0]})
    different = _fake_transform({0: [1.0], 2: [2.0]})

    assert hybrid_pair_schedule_compatible(left, same, slots=8) is True
    assert hybrid_pair_schedule_compatible(left, different, slots=8) is False
    assert hybrid_pair_schedule_compatible(left, None, slots=8) is False
    assert hybrid_pair_schedule_compatible(None, same, slots=8) is False


def test_hybrid_schedule_padding_is_explicit_and_family_scoped() -> None:
    left = _fake_transform({0: torch.ones(8)})
    right = _fake_transform({1: torch.ones(8)})

    unpadded_left, unpadded_right, reason = pad_hybrid_pair_to_common_schedule(left, right, 8, name="unmarked")
    assert reason == ""
    assert hybrid_pair_schedule_compatible(unpadded_left, unpadded_right, slots=8) is False

    mark_hybrid_schedule_padding_allowed(left, family="same_tile_family")
    mark_hybrid_schedule_padding_allowed(right, family="same_tile_family")
    padded_left, padded_right, reason = pad_hybrid_pair_to_common_schedule(left, right, 8, name="marked")

    assert reason.startswith("schedule_padded(")
    assert hybrid_pair_schedule_compatible(padded_left, padded_right, slots=8) is True
    assert sorted(padded_left.diagonals[(0, 0)]) == [0, 1]
    assert sorted(padded_right.diagonals[(0, 0)]) == [0, 1]


def test_hybrid_pair_layout_optimizer_shifts_boundaries_to_maximize_strict_pairs() -> None:
    transforms_by_block = {
        0: [_fake_transform({0: torch.ones(8)})],
        1: [_fake_transform({1: torch.ones(8)})],
        2: [_fake_transform({1: torch.ones(8)})],
    }

    plan = optimize_hybrid_pair_layout(transforms_by_block, slots=8)

    assert [(item.left_index, item.right_index) for item in plan.items] == [(0, None), (1, 2)]
    assert plan.strict_pair_count == 1
    assert plan.covered_output_count == 1
    assert plan.singleton_count == 1
    assert plan.uses_shifted_boundaries is True
    assert any("input_pair=(0,1)" in reason for reason in plan.rejected_adjacent_pair_reasons)


def test_hybrid_pair_layout_materializes_planner_approved_common_schedule() -> None:
    left = _fake_transform({0: torch.ones(8)})
    right = _fake_transform({1: torch.ones(8)})
    mark_hybrid_schedule_padding_allowed(left, family="global_layout")
    mark_hybrid_schedule_padding_allowed(right, family="global_layout")
    transforms_by_block = {0: [left], 1: [right]}

    plan = optimize_hybrid_pair_layout(
        transforms_by_block,
        slots=8,
        allow_schedule_materialization=True,
    )
    result = materialize_hybrid_pair_layout_schedules(
        transforms_by_block,
        plan,
        slots=8,
        name_prefix="global_layout_test",
    )

    assert [(item.left_index, item.right_index) for item in plan.items] == [(0, 1)]
    assert plan.strict_pair_count == 1
    assert plan.schedule_materialized_pair_count == 1
    assert result.pair_count == 1
    assert result.output_count == 1
    assert hybrid_pair_schedule_compatible(transforms_by_block[0][0], transforms_by_block[1][0], slots=8)


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
    runtime = group.last_runtime_timing
    assert runtime["runtime_fairness_mode"] == "resident_compute"
    assert runtime["resident_compute_s"] == pytest.approx(runtime["eval_s"])
    assert runtime["artifact_load_s"] == pytest.approx(
        runtime["load_keys_s"] + runtime["load_plaintexts_s"]
    )
    assert runtime["artifact_unload_s"] == pytest.approx(runtime["unload_s"])
    assert runtime["serving_hot_s"] >= runtime["resident_compute_s"]

    group.cleanup(backend)
    assert backend.deleted == [11, 12]
    assert group.is_compiled is False


def test_single_slot_layer_cache_evicts_between_tiny_64_layers(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    monkeypatch.setenv("ORION_SINGLE_SLOT_ENCODE_WORKERS", "2")
    slots_64 = 64 * 64
    backend = _SingleSlotTrackingBackend()
    backend.rotation_keys = {
        11: [1],
        12: [3],
        13: [5],
    }
    groups = [
        UnifiedTransformGroup(
            (
                _fake_transform(
                    {
                        0: torch.ones(slots_64),
                        layer_index + 1: torch.ones(slots_64),
                    },
                    level=2,
                ),
            )
        )
        for layer_index in range(3)
    ]

    for group in groups:
        group.compile_unified(backend)
        assert group.is_compiled is True
        assert group.unified_ids is None
        assert group.last_compile_profile["mode"] == "single_slot_deferred"
        assert group.last_compile_profile["flatten_s"] >= 0.0
        assert group._single_slot_payloads is None
        assert group._single_slot_recipes is not None
        assert all(getattr(transform, "diagonals", None) == {} for transform in group.transforms)

    assert backend.generated == []

    for layer_index, group in enumerate(groups):
        assert group.evaluate_unified(64 + int(layer_index), backend) == [100]
        assert group.last_compile_profile["mode"] == "single_slot_materialize"
        assert group.last_compile_profile["flatten_s"] == 0.0
        runtime = group.last_runtime_timing
        assert runtime["runtime_fairness_mode"] == "single_slot_layer_cache"
        assert runtime["resident_compute_s"] == pytest.approx(runtime["eval_s"])
        assert runtime["layer_cache_encode_s"] > 0.0
        assert runtime["layer_cache_evict_s"] >= 0.0
        assert runtime["layer_cache_turnover_s"] == pytest.approx(
            runtime["layer_cache_encode_s"]
            + runtime["layer_cache_key_prepare_s"]
            + runtime["layer_cache_evict_s"]
        )

    assert backend.live_before_generate == [(), (), ()]
    assert backend.max_live_plaintext_transforms == 1
    assert backend.live_plaintext_transforms == set()
    assert backend.deleted == [11, 12, 13]
    assert [entry["num_transforms"] for entry in backend.generated] == [1, 1, 1]


def test_single_slot_requires_recipe_and_never_generates_backend_ids_at_compile(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    missing_recipe = SimpleNamespace(
        diagonals={(0, 0): {0: torch.ones(8)}},
        level=2,
        scheme=SimpleNamespace(params=SimpleNamespace(get_logq=lambda: [0, 1, 2])),
    )
    group = UnifiedTransformGroup((missing_recipe,))
    backend = _FakeBackend()

    with pytest.raises(RuntimeError, match="runtime diagonal recipe|_single_slot_build_diagonals"):
        group.compile_unified(backend)

    assert backend.generated == []
    assert backend.generated_keys == []


def test_single_slot_materializes_whole_group_once_and_evicts_once(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    monkeypatch.setenv("ORION_SINGLE_SLOT_ENCODE_WORKERS", "1")
    backend = _SingleSlotTrackingBackend()
    transforms = (
        _fake_transform({0: torch.ones(16)}, level=2),
        _fake_transform({1: torch.ones(16) * 2}, level=2),
    )
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)

    assert backend.generated == []
    assert backend.generated_keys == [0, 1]

    outputs = group.evaluate_unified(77, backend)

    assert outputs == [100, 101]
    assert [entry["num_transforms"] for entry in backend.generated] == [2]
    assert backend.evaluated == [([11, 12], 77)]
    assert backend.deleted == [11, 12]
    assert backend.live_plaintext_transforms == set()
    assert group.unified_ids is None


def test_single_slot_releases_shared_diagonal_cache_once(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    monkeypatch.setenv("ORION_SINGLE_SLOT_ENCODE_WORKERS", "1")

    class SharedCache:
        def __init__(self) -> None:
            self.blocks = None
            self.build_calls = 0
            self.release_calls = 0

        def get_required(self, row: int, col: int, *, context: str = ""):
            if self.blocks is None:
                self.build_calls += 1
                self.blocks = {
                    (0, 0): {0: torch.ones(16)},
                    (1, 0): {1: torch.ones(16) * 2},
                }
            return dict(self.blocks[(int(row), int(col))])

        def release(self) -> None:
            self.release_calls += 1
            self.blocks = None

    cache = SharedCache()

    def cached_transform(row: int, diag: int):
        transform = _fake_transform({int(diag): torch.ones(16)}, level=2)
        transform.diagonals = {}
        transform._single_slot_build_diagonals = (
            lambda row=int(row), cache=cache: {(0, 0): cache.get_required(int(row), 0)}
        )
        transform._single_slot_diagonal_cache = cache
        transform._single_slot_release_diagonal_cache = cache.release
        return transform

    group = UnifiedTransformGroup((cached_transform(0, 0), cached_transform(1, 1)))
    backend = _SingleSlotTrackingBackend()

    group.compile_unified(backend)
    group.evaluate_unified(77, backend)

    assert cache.build_calls == 1
    assert cache.release_calls == 1
    assert backend.generated[0]["num_transforms"] == 2


def test_single_slot_releases_shared_diagonal_cache_on_materialize_error(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    monkeypatch.setenv("ORION_SINGLE_SLOT_ENCODE_WORKERS", "1")
    release_calls = 0

    def release_cache() -> None:
        nonlocal release_calls
        release_calls += 1

    transform = _fake_transform({0: torch.ones(16)}, level=2)
    transform.diagonals = {}
    transform._single_slot_build_diagonals = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    transform._single_slot_diagonal_cache = object()
    transform._single_slot_release_diagonal_cache = release_cache
    group = UnifiedTransformGroup((transform,))
    backend = _SingleSlotTrackingBackend()

    group.compile_unified(backend)
    with pytest.raises(RuntimeError, match="boom"):
        group.evaluate_unified(77, backend)

    assert release_calls == 1
    assert backend.generated == []
    assert backend.deleted == []


def test_single_slot_progress_file_tracks_materialize_eval_and_evict(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    monkeypatch.setenv("ORION_SINGLE_SLOT_ENCODE_WORKERS", "1")
    progress_path = tmp_path / "progress.jsonl"
    state_path = tmp_path / "progress_state.json"
    monkeypatch.setenv("ORION_PROGRESS_JSONL", str(progress_path))
    monkeypatch.setenv("ORION_PROGRESS_STATE_JSON", str(state_path))
    monkeypatch.setenv("ORION_PROGRESS_CONTEXT", json.dumps({"network": "tiny", "mode": "provider"}))
    backend = _SingleSlotTrackingBackend()
    transform_a = _fake_transform({0: torch.ones(16)}, level=2)
    transform_b = _fake_transform({1: torch.ones(16) * 2}, level=2)
    transform_a.name = "dec1a_concat0_a"
    transform_b.name = "dec1a_concat0_b"
    group = UnifiedTransformGroup((transform_a, transform_b))

    group.compile_unified(backend)
    group.evaluate_unified(77, backend)

    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    phases = [(row["event"], row["phase"]) for row in rows]
    assert ("start", "diag_encode") in phases
    assert ("end", "diag_encode") in phases
    assert ("start", "eval") in phases
    assert ("end", "eval") in phases
    assert ("start", "evict") in phases
    assert ("end", "evict") in phases
    assert rows[0]["network"] == "tiny"
    assert rows[0]["mode"] == "provider"
    assert rows[0]["layer"] == "dec1a_concat0_a"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["event"] == "end"
    assert state["phase"] == "eval"


def test_unified_transform_sources_target_sum_uses_compatible_targets() -> None:
    backend = _FakeBackend()
    group_a = UnifiedTransformGroup((_fake_transform({0: [1.0]}, level=2), _fake_transform({0: [2.0]}, level=2)))
    group_b = UnifiedTransformGroup((_fake_transform({0: [3.0]}, level=2), _fake_transform({0: [4.0]}, level=2)))
    group_a.is_compiled = True
    group_a.unified_ids = [11, 12]
    group_b.is_compiled = True
    group_b.unified_ids = [21, 22]

    outputs = UnifiedTransformGroup.evaluate_sources_with_target_sum(
        [group_a, group_b],
        [7, 8],
        [(0, 1), (0, 1)],
        2,
        backend,
    )

    assert outputs == [200, 201]
    assert backend.source_target_sum_evaluated == [
        {
            "ctxt_ids": [7, 8],
            "transform_ids": [11, 12, 21, 22],
            "target_ids": [0, 1, 0, 1],
            "group_offsets": [0, 2, 4],
            "target_count": 2,
        }
    ]


def test_unified_transform_sources_target_sum_rejects_mismatched_target_level() -> None:
    backend = _FakeBackend()
    backend.ct_levels[8] = 1
    group_a = UnifiedTransformGroup((_fake_transform({0: [1.0]}, level=2),))
    group_b = UnifiedTransformGroup((_fake_transform({0: [3.0]}, level=2),))
    group_a.is_compiled = True
    group_a.unified_ids = [11]
    group_b.is_compiled = True
    group_b.unified_ids = [21]

    outputs = UnifiedTransformGroup.evaluate_sources_with_target_sum(
        [group_a, group_b],
        [7, 8],
        [(0,), (0,)],
        1,
        backend,
    )

    assert outputs is None
    assert backend.source_target_sum_evaluated == []


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
    assert sum(entry["num_transforms"] for entry in backend.generated) == 2
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

    assert [entry["num_transforms"] for entry in load_backend.generated] == [
        entry["num_transforms"] for entry in save_backend.generated
    ]
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


def test_save_resume_reuses_complete_unified_group(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_UNIFIED_STREAM_COMPILE_BATCH_TRANSFORMS", "2")
    monkeypatch.setenv("ORION_UNIFIED_STREAM_COMPILE_BATCH_BYTES", "0")
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


def test_save_resume_missing_unified_root_falls_back_to_compile(tmp_path) -> None:
    diags_path = tmp_path / "unified_diagonals.h5"
    with h5py.File(diags_path, "w") as handle:
        handle.create_group("unrelated")

    transforms = (
        _fake_transform(
            {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            compile_save_resume=True,
        ),
        _fake_transform(
            {0: [3.0, 0.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0, 0.0]},
            io_mode="save",
            diags_path=str(diags_path),
            compile_save_resume=True,
        ),
    )
    backend = _FakeUnifiedLoadBackend()
    backend.memory_bounded_unified_transforms = True
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)

    assert backend.loaded_shells == []
    assert backend.generated
    with h5py.File(diags_path, "r") as handle:
        assert "__unified_transform_groups__" in handle
        assert group._storage_key in handle["__unified_transform_groups__"]


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
    assert sum(entry["num_transforms"] for entry in backend.generated) == 2
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
    runtime = group.last_runtime_timing
    assert runtime["runtime_fairness_mode"] == "streaming_eval_encode"
    assert runtime["resident_compute_s"] is None
    assert runtime["serving_hot_s"] >= 0.0
    assert runtime["artifact_load_s"] == pytest.approx(
        runtime["load_keys_s"] + runtime["load_plaintexts_s"]
    )


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
    runtime = group.last_runtime_timing
    assert runtime["runtime_fairness_mode"] == "memory_bounded_load_eval"
    assert runtime["resident_compute_s"] == pytest.approx(runtime["eval_s"])
    assert runtime["artifact_read_s"] == pytest.approx(runtime["read_bundle_s"])
    assert runtime["artifact_load_s"] == pytest.approx(
        runtime["load_keys_s"] + runtime["load_plaintexts_s"]
    )
    assert runtime["artifact_unload_s"] == pytest.approx(runtime["unload_s"])
    assert runtime["trim_s"] == pytest.approx(0.25)


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


def test_single_slot_layer_cache_runs_and_evicts_on_lattigo_backend(monkeypatch) -> None:
    shared_library = Path("orion/backend/lattigo/lattigo-linux.so")
    if not shared_library.exists():
        pytest.skip("local Lattigo shared library has not been built")

    monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    monkeypatch.setenv("ORION_SINGLE_SLOT_ENCODE_WORKERS", "2")
    monkeypatch.setenv("ORION_LATTIGO_STREAMING_LT", "0")

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
        transform_a = SimpleNamespace(
            diagonals={(0, 0): {0: [1.0] * slots}},
            _single_slot_diag_indices_by_block={(0, 0): (0,)},
            _single_slot_build_diagonals=lambda slots=slots: {(0, 0): {0: [1.0] * slots}},
            level=level,
            scheme=scheme,
            fhe_output_shape=torch.Size([1, slots]),
            output_shape=torch.Size([1, slots]),
        )
        transform_b = SimpleNamespace(
            diagonals={(0, 0): {0: [2.0] * slots}},
            _single_slot_diag_indices_by_block={(0, 0): (0,)},
            _single_slot_build_diagonals=lambda slots=slots: {(0, 0): {0: [2.0] * slots}},
            level=level,
            scheme=scheme,
            fhe_output_shape=torch.Size([1, slots]),
            output_shape=torch.Size([1, slots]),
        )
        group = UnifiedTransformGroup([transform_a, transform_b])
        group.compile_unified(scheme.backend)

        assert group.unified_ids is None
        assert group.last_compile_profile["mode"] == "single_slot_deferred"
        assert transform_a.diagonals == {}
        assert transform_b.diagonals == {}

        x = torch.zeros(slots, dtype=torch.float32)
        x[:8] = torch.linspace(0.1, 0.8, 8)
        ct = scheme.encrypt(scheme.encode(x, level))
        output_ids = group.evaluate_unified(ct.ids[0], scheme.backend)

        assert group.unified_ids is None
        runtime = group.last_runtime_timing
        assert runtime["runtime_fairness_mode"] == "single_slot_layer_cache"
        assert runtime["layer_cache_encode_s"] > 0.0
        assert runtime["layer_cache_turnover_s"] == pytest.approx(
            runtime["layer_cache_encode_s"]
            + runtime["layer_cache_key_prepare_s"]
            + runtime["layer_cache_evict_s"]
        )

        decoded = []
        for output_id in output_ids:
            out_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, slots]), torch.Size([1, slots]))
            decoded.append(out_ct.decrypt().decode().reshape(-1))
        assert float((decoded[0][:8] - x[:8]).abs().max()) <= 1.0e-4
        assert float((decoded[1][:8] - 2.0 * x[:8]).abs().max()) <= 1.0e-4
    finally:
        scheme.delete_scheme()


def test_lattigo_streaming_lt_force_requires_legacy_gate() -> None:
    shared_library = Path("orion/backend/lattigo/lattigo-linux.so")
    if not shared_library.exists():
        pytest.skip("local Lattigo shared library has not been built")

    probe = """
from orion.core.orion import scheme
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
    transform_id = scheme.backend.GenerateLinearTransform(
        [0, 1],
        [1.0] * slots + [0.5] * slots,
        level,
        1.0,
        "none",
    )
    print(int(scheme.backend.LinearTransformUsesStreaming(int(transform_id))))
finally:
    scheme.delete_scheme()
"""

    def run_probe(*, legacy: bool) -> int:
        env = dict(os.environ)
        env["ORION_LATTIGO_STREAMING_LT"] = "force"
        env["ORION_SINGLE_SLOT_LAYER_CACHE"] = "0"
        if legacy:
            env["ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT"] = "1"
        else:
            env.pop("ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT", None)
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return int(completed.stdout.strip())

    assert run_probe(legacy=False) == 0
    assert run_probe(legacy=True) == 1


def test_unified_transform_sources_target_sum_runs_on_lattigo_backend() -> None:
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

        def transform(name: str, multiplier: float):
            return SimpleNamespace(
                name=name,
                diagonals={(0, 0): {0: [float(multiplier)] * slots}},
                level=level,
                scheme=scheme,
                fhe_output_shape=torch.Size([1, slots]),
                output_shape=torch.Size([1, slots]),
            )

        group_a = UnifiedTransformGroup([transform("a_to_0", 1.0), transform("a_to_1", 2.0)])
        group_b = UnifiedTransformGroup([transform("b_to_0", 3.0), transform("b_to_1", 4.0)])
        group_a.compile_unified(scheme.backend)
        group_b.compile_unified(scheme.backend)

        x = torch.zeros(slots, dtype=torch.float32)
        y = torch.zeros(slots, dtype=torch.float32)
        x[:8] = torch.linspace(0.1, 0.8, 8)
        y[:8] = torch.linspace(0.2, 0.9, 8)
        ct_x = scheme.encrypt(scheme.encode(x, level))
        ct_y = scheme.encrypt(scheme.encode(y, level))

        output_ids = UnifiedTransformGroup.evaluate_sources_with_target_sum(
            [group_a, group_b],
            [int(ct_x.ids[0]), int(ct_y.ids[0])],
            [(0, 1), (0, 1)],
            2,
            scheme.backend,
        )

        assert output_ids is not None
        decoded = []
        for output_id in output_ids:
            out_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, slots]), torch.Size([1, slots]))
            values = out_ct.decrypt().decode().reshape(-1)
            decoded.append(values.real if torch.is_complex(values) else values)

        assert len(decoded) == 2
        assert float((decoded[0][:8] - (x[:8] + 3.0 * y[:8])).abs().max()) <= 1.0e-3
        assert float((decoded[1][:8] - (2.0 * x[:8] + 4.0 * y[:8])).abs().max()) <= 1.0e-3
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
