from types import SimpleNamespace

import torch

from orion.nn.unified_transform import UnifiedTransformGroup


class _FakeBackend:
    def __init__(self) -> None:
        self.generated = []

    def GenerateLinearTransformsUnified(
        self,
        num_transforms,
        _diag_idxs_ptrs,
        diag_idxs_lens,
        _diag_data_ptrs,
        diag_data_lens,
        _levels_array,
    ):
        self.generated.append(
            {
                "num_transforms": int(num_transforms),
                "diag_lengths": [int(diag_idxs_lens[i]) for i in range(int(num_transforms))],
                "data_lengths": [int(diag_data_lens[i]) for i in range(int(num_transforms))],
            }
        )
        return [11 + i for i in range(int(num_transforms))]

    def GetLinearTransformRotationKeys(self, _transform_id):
        return []


def _transform(diagonals):
    return SimpleNamespace(
        diagonals={(0, 0): {key: torch.as_tensor(value) for key, value in diagonals.items()}},
        level=2,
    )


def test_clear_source_diagonals_after_non_streaming_compile(monkeypatch):
    monkeypatch.setenv("ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE", "1")
    transforms = (
        _transform({0: [1.0, 0.0, 0.0, 0.0]}),
        _transform({1: [0.0, 2.0, 0.0, 0.0]}),
    )
    backend = _FakeBackend()
    group = UnifiedTransformGroup(transforms)

    group.compile_unified(backend)

    assert backend.generated == [
        {
            "num_transforms": 2,
            "diag_lengths": [1, 1],
            "data_lengths": [4, 4],
        }
    ]
    assert not group._offloaded_plaintext_diagonals
    assert transforms[0].diagonals == {}
    assert transforms[1].diagonals == {}
