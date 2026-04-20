from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
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
