from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion.nn.unified_transform import UnifiedTransformGroup, can_use_unified_bsgs


class _FakeBackend:
    def __init__(self) -> None:
        self.generated = []
        self.evaluated = []
        self.deleted = []
        self.rotation_keys = {11: [1, 3], 12: [1, 5]}
        self.generated_keys = []

    def GenerateLinearTransformsUnified(self, num_transforms, diag_idxs_ptrs, diag_idxs_lens, diag_data_ptrs, diag_data_lens, levels_array):
        self.generated.append(
            {
                "num_transforms": int(num_transforms),
                "diag_lengths": [int(diag_idxs_lens[i]) for i in range(int(num_transforms))],
                "data_lengths": [int(diag_data_lens[i]) for i in range(int(num_transforms))],
                "levels": [int(levels_array[i]) for i in range(int(num_transforms))],
            }
        )
        return [11 + i for i in range(int(num_transforms))]

    def GetLinearTransformRotationKeys(self, transform_id):
        return list(self.rotation_keys[int(transform_id)])

    def GenerateLinearTransformRotationKey(self, key):
        self.generated_keys.append(int(key))

    def EvaluateLinearTransformsWithSharedCache(self, transform_ids_array, num_transforms, ct_input_id):
        ids = [int(transform_ids_array[i]) for i in range(int(num_transforms))]
        self.evaluated.append((ids, int(ct_input_id)))
        return [100 + i for i in range(int(num_transforms))]

    def DeleteLinearTransform(self, transform_id):
        self.deleted.append(int(transform_id))


def _fake_transform(diagonals, *, level=2):
    return SimpleNamespace(
        diagonals={(0, 0): dict(diagonals)},
        level=int(level),
        scheme=SimpleNamespace(params=SimpleNamespace(get_logq=lambda: [0, 1, 2, 3])),
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
    assert sorted(backend.generated_keys) == [1, 1, 3, 5]
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


def test_can_use_unified_bsgs_rejects_non_linear_transform_instances() -> None:
    assert can_use_unified_bsgs([_fake_transform({0: [1.0]})]) is False
