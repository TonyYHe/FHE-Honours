from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

import orion
from orion.backend.python.lt_evaluator import NewEvaluator
from orion.backend.python import compile_cache
from orion.core import packing
from orion.core import orion as orion_core
from orion.nn.linear import Conv2d, ConvTranspose2d
from orion.nn.module import Module


class _FakeParams:
    def __init__(self, *, slots: int, embedding_method: str = "square", io_mode: str = "none") -> None:
        self._slots = int(slots)
        self._embedding_method = str(embedding_method)
        self._io_mode = str(io_mode)

    def get_slots(self):
        return int(self._slots)

    def get_embedding_method(self):
        return str(self._embedding_method)

    def get_io_mode(self):
        return str(self._io_mode)

    def get_diags_path(self):
        return ""

    def get_keys_path(self):
        return ""


def _attach_fake_scheme(layer, *, slots: int, embedding_method: str = "square") -> None:
    layer.scheme = SimpleNamespace(params=_FakeParams(slots=slots, embedding_method=embedding_method))


def _configure_conv2d(layer: Conv2d, x: torch.Tensor, *, input_gap: int) -> None:
    y = layer(x)
    layer.input_shape = torch.Size(x.shape)
    layer.output_shape = torch.Size(y.shape)
    layer.input_gap = int(input_gap)
    layer.output_gap = int(input_gap) * int(layer.stride[0])
    layer.fhe_input_shape = torch.Size(
        (
            x.shape[0],
            (layer.in_channels + layer.input_gap**2 - 1) // layer.input_gap**2,
            x.shape[2] * layer.input_gap,
            x.shape[3] * layer.input_gap,
        )
    )
    layer.fhe_output_shape = layer.compute_fhe_output_shape(
        input_gap=layer.input_gap,
        input_shape=layer.input_shape,
        clear_output_shape=layer.output_shape,
    )


def _configure_tconv2d(layer: ConvTranspose2d, x: torch.Tensor, *, input_gap: int) -> None:
    y = layer(x)
    layer.input_shape = torch.Size(x.shape)
    layer.output_shape = torch.Size(y.shape)
    layer.input_gap = int(input_gap)
    layer.output_gap = max(1, int(input_gap) // int(layer.stride[0]))
    layer.fhe_input_shape = torch.Size(
        (
            x.shape[0],
            (layer.in_channels + layer.input_gap**2 - 1) // layer.input_gap**2,
            x.shape[2] * layer.input_gap,
            x.shape[3] * layer.input_gap,
        )
    )
    layer.fhe_output_shape = layer.compute_fhe_output_shape(
        input_gap=layer.input_gap,
        clear_output_shape=layer.output_shape,
    )


def _assert_diagonals_close(left, right) -> None:
    assert set(left.keys()) == set(right.keys())
    for block_key in left:
        assert set(left[block_key].keys()) == set(right[block_key].keys())
        for diag_idx in left[block_key]:
            lval = torch.as_tensor(left[block_key][diag_idx], dtype=torch.float32)
            rval = torch.as_tensor(right[block_key][diag_idx], dtype=torch.float32)
            assert torch.allclose(lval, rval, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.parametrize("embedding_method", ["square", "hybrid"])
@pytest.mark.parametrize(
    "spec",
    [
        {"in_channels": 2, "out_channels": 3, "groups": 1},
        {"in_channels": 4, "out_channels": 4, "groups": 2},
    ],
)
def test_conv2d_direct_diagonals_match_legacy_toeplitz(embedding_method: str, spec: dict) -> None:
    torch.manual_seed(0)
    layer = Conv2d(
        spec["in_channels"],
        spec["out_channels"],
        kernel_size=3,
        stride=1,
        padding=1,
        groups=spec["groups"],
        bias=False,
    )
    layer.init_orion_params()
    _attach_fake_scheme(layer, slots=64, embedding_method=embedding_method)
    _configure_conv2d(layer, torch.randn(1, spec["in_channels"], 3, 3), input_gap=1)

    direct, direct_rotations = packing.pack_conv2d(layer, last=False)
    weight = packing.resolve_grouped_conv(layer) if layer.groups > 1 else layer.on_weight
    toeplitz = packing.construct_conv2d_toeplitz(layer, weight)
    legacy, legacy_rotations = packing.diagonalize(
        toeplitz,
        64,
        embedding_method,
        False,
    )

    assert direct_rotations == legacy_rotations
    _assert_diagonals_close(direct, legacy)


def test_conv2d_pack_keeps_zero_blocks_for_dense_baseline() -> None:
    layer = Conv2d(1, 1, kernel_size=1, bias=False)
    layer.weight.data.zero_()
    layer.init_orion_params()
    _attach_fake_scheme(layer, slots=8, embedding_method="square")
    _configure_conv2d(layer, torch.randn(1, 1, 4, 4), input_gap=1)

    diagonals, output_rotations = packing.pack_conv2d(layer, last=False)

    assert output_rotations == 0
    assert len(diagonals) == 4
    assert set(diagonals) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert packing.prune_zero_diagonal_blocks(diagonals) == {}
    assert set(packing.prune_zero_diagonal_blocks(diagonals, preserve_empty_rows=True)) == {
        (0, 0),
        (1, 0),
    }


def test_conv_transpose2d_pack_keeps_zero_blocks_for_dense_baseline() -> None:
    layer = ConvTranspose2d(1, 1, kernel_size=2, stride=2, padding=0, bias=False)
    layer.weight.data.zero_()
    layer.init_orion_params()
    _attach_fake_scheme(layer, slots=8, embedding_method="square")
    _configure_tconv2d(layer, torch.randn(1, 1, 2, 2), input_gap=1)

    diagonals, output_rotations = packing.pack_conv_transpose2d(layer, last=False)

    assert output_rotations == 0
    assert diagonals
    assert packing.prune_zero_diagonal_blocks(diagonals) == {}


def test_prune_zero_diagonal_blocks_removes_empty_terms_and_preserves_empty_rows() -> None:
    diagonals = {
        (0, 0): {0: torch.zeros(4), 1: torch.tensor([0.0, 1.0, 0.0, 0.0])},
        (0, 1): {0: torch.zeros(4)},
        (1, 0): {0: torch.zeros(4)},
    }

    pruned = packing.prune_zero_diagonal_blocks(diagonals)
    row_preserving = packing.prune_zero_diagonal_blocks(diagonals, preserve_empty_rows=True)

    assert set(pruned) == {(0, 0)}
    assert set(pruned[(0, 0)]) == {1}
    assert torch.equal(pruned[(0, 0)][1], diagonals[(0, 0)][1])
    assert set(row_preserving) == {(0, 0), (1, 0)}
    assert set(row_preserving[(0, 0)]) == {1}


@pytest.mark.parametrize(
    "spec",
    [
        {"in_channels": 2, "out_channels": 3, "groups": 1},
        {"in_channels": 4, "out_channels": 4, "groups": 2},
    ],
)
def test_conv_transpose2d_direct_diagonals_match_legacy_toeplitz(spec: dict) -> None:
    torch.manual_seed(1)
    layer = ConvTranspose2d(
        spec["in_channels"],
        spec["out_channels"],
        kernel_size=2,
        stride=2,
        padding=0,
        groups=spec["groups"],
        bias=False,
    )
    layer.init_orion_params()
    _attach_fake_scheme(layer, slots=128, embedding_method="hybrid")
    _configure_tconv2d(layer, torch.randn(1, spec["in_channels"], 2, 2), input_gap=1)

    direct, direct_rotations = packing.pack_conv_transpose2d(layer, last=False)
    toeplitz = packing.construct_conv_transpose2d_toeplitz(layer)
    legacy, legacy_rotations = packing.diagonalize(
        toeplitz,
        128,
        "hybrid",
        False,
    )

    assert direct_rotations == legacy_rotations
    _assert_diagonals_close(direct, legacy)


def test_dense_compile_batches_independent_transforms_without_unified_api() -> None:
    class Backend:
        def __init__(self):
            self.batch_called = False

        def NewLinearTransformEvaluator(self):
            return None

        def GenerateLinearTransformsBatch(self, *args):
            self.batch_called = True
            assert len(args) == 8
            return [101, 102]

        def GenerateLinearTransformsUnified(self, *_args):
            raise AssertionError("dense baseline must not use unified transforms")

        def GetLinearTransformRotationKeys(self, _transform_id):
            return []

    backend = Backend()
    fake_scheme = SimpleNamespace(
        backend=backend,
        evaluator=SimpleNamespace(),
        params=_FakeParams(slots=4),
    )
    evaluator = NewEvaluator(fake_scheme)
    layer = SimpleNamespace(
        name="tiny",
        diagonals={
            (0, 0): {0: [1.0, 0.0, 0.0, 0.0]},
            (0, 1): {1: [0.0, 2.0, 0.0, 0.0]},
        },
        level=2,
        bsgs_ratio=2,
    )

    transform_ids = evaluator.generate_transforms(layer)

    assert backend.batch_called is True
    assert transform_ids == {(0, 0): 101, (0, 1): 102}
    profile = evaluator.get_compile_load_profile()
    assert {
        "read_s",
        "diag_generate_s",
        "encode_s",
        "decode_s",
        "serialize_s",
        "device_commit_s",
        "key_prepare_s",
        "wait_s",
        "peak_host_bytes",
        "peak_device_bytes",
    }.issubset(profile)
    assert profile["diag_generate_s"] >= 0.0
    assert profile["encode_s"] >= 0.0


def test_dense_compile_batches_are_memory_bounded(monkeypatch) -> None:
    class Backend:
        def __init__(self):
            self.batch_sizes = []
            self.next_id = 200

        def NewLinearTransformEvaluator(self):
            return None

        def GenerateLinearTransformsBatch(self, num_transforms, *_args):
            self.batch_sizes.append(int(num_transforms))
            ids = list(range(self.next_id, self.next_id + int(num_transforms)))
            self.next_id += int(num_transforms)
            return ids

        def GenerateLinearTransformsUnified(self, *_args):
            raise AssertionError("dense baseline must not use unified transforms")

        def GetLinearTransformRotationKeys(self, _transform_id):
            return []

    monkeypatch.setenv("ORION_DENSE_LT_COMPILE_BATCH_TRANSFORMS", "2")
    backend = Backend()
    fake_scheme = SimpleNamespace(
        backend=backend,
        evaluator=SimpleNamespace(),
        params=_FakeParams(slots=4),
    )
    evaluator = NewEvaluator(fake_scheme)
    layer = SimpleNamespace(
        name="bounded",
        diagonals={
            (0, index): {index: [float(index), 0.0, 0.0, 0.0]}
            for index in range(5)
        },
        level=2,
        bsgs_ratio=2,
    )

    transform_ids = evaluator.generate_transforms(layer)

    assert backend.batch_sizes == [2, 2, 1]
    assert transform_ids == {
        (0, 0): 200,
        (0, 1): 201,
        (0, 2): 202,
        (0, 3): 203,
        (0, 4): 204,
    }


def test_dense_save_load_compile_defaults_to_provider_sized_batches(monkeypatch) -> None:
    monkeypatch.setenv("ORION_COMPILE_PARALLEL_POLICY", "manual")
    for name in (
        "ORION_DENSE_LT_COMPILE_BATCH_TRANSFORMS",
        "ORION_LT_COMPILE_BATCH_TRANSFORMS",
        "ORION_UNIFIED_CACHED_LOAD_BATCH_TRANSFORMS",
        "ORION_UNIFIED_LOAD_BATCH_TRANSFORMS",
    ):
        monkeypatch.delenv(name, raising=False)
    evaluator = object.__new__(NewEvaluator)
    evaluator.io_mode = "save"

    assert evaluator._lt_compile_batch_limit(9) == 4


class _FakeSerializedDenseBackend:
    load_plaintext_diagonals_requires_payload = True

    def __init__(self) -> None:
        self.serialized = []
        self.freed = 0
        self.removed = []
        self.loaded_batches = []
        self.loaded_diagonals = []

    def SerializeDiagonal(self, transform_id, diag_idx):
        self.serialized.append((int(transform_id), int(diag_idx)))
        return np.asarray([int(transform_id), int(diag_idx), int(diag_idx) + 1], dtype=np.uint8), object()

    def FreeCArray(self, _ptr) -> None:
        self.freed += 1

    def RemovePlaintextDiagonals(self, transform_id) -> None:
        self.removed.append(int(transform_id))

    def LoadPlaintextDiagonalsBatch(self, payload, offsets, lengths, diag_indices, transform_id) -> None:
        payload_arr = np.asarray(payload, dtype=np.uint8).reshape(-1)
        self.loaded_batches.append(
            {
                "transform_id": int(transform_id),
                "diag_indices": [int(value) for value in diag_indices],
                "segments": [
                    payload_arr[int(offset): int(offset) + int(length)].tolist()
                    for offset, length in zip(offsets, lengths)
                ],
            }
        )

    def LoadPlaintextDiagonal(self, payload, transform_id, diag_idx) -> None:
        self.loaded_diagonals.append(
            (int(transform_id), int(diag_idx), np.asarray(payload, dtype=np.uint8).reshape(-1).tolist())
        )


def _dense_cache_evaluator(tmp_path, backend) -> NewEvaluator:
    evaluator = object.__new__(NewEvaluator)
    evaluator.backend = backend
    evaluator.diags_path = str(tmp_path / "diags.h5")
    evaluator.keys_path = ""
    evaluator.io_mode = "save"
    evaluator.compile_load_profile = {}
    evaluator._transform_io_size_cache = {}
    evaluator._transform_device_size_cache = {}
    return evaluator


def test_dense_plaintext_cache_saves_coarse_payload_and_loads_batch(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ORION_DENSE_LT_COARSE_ARTIFACT_IO", raising=False)
    backend = _FakeSerializedDenseBackend()
    evaluator = _dense_cache_evaluator(tmp_path, backend)

    evaluator.save_plaintext_diagonals("cached", 77, 0, 1, [5, 3])

    assert backend.serialized == [(77, 5), (77, 3)]
    assert backend.freed == 2
    assert backend.removed == [77]
    with h5py.File(tmp_path / "diags.h5", "r") as handle:
        block = handle["cached"]["plaintexts"]["0_1"]
        assert block["diag_indices"][:].tolist() == [5, 3]
        assert block["diag_offsets"][:].tolist() == [0, 3]
        assert block["diag_lengths"][:].tolist() == [3, 3]
        assert block["diag_payload"][:].tolist() == [77, 5, 6, 77, 3, 4]
        assert "5" not in block and "3" not in block

    bundle = evaluator._read_transform_io_bundle("cached", 0, 1, 88, prefetch=False)
    evaluator.load_plaintext_diagonals("cached", 0, 1, 88, bundle=bundle)

    assert backend.loaded_batches[-1] == {
        "transform_id": 88,
        "diag_indices": [5, 3],
        "segments": [[77, 5, 6], [77, 3, 4]],
    }


def test_dense_coarse_payload_streams_when_chunk_limit_is_small(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_DENSE_LT_COARSE_LOAD_CHUNK_BYTES", "4")
    backend = _FakeSerializedDenseBackend()
    evaluator = _dense_cache_evaluator(tmp_path, backend)
    evaluator.save_plaintext_diagonals("cached", 77, 0, 1, [5, 3])

    bundle = evaluator._read_transform_io_bundle("cached", 0, 1, 88, prefetch=False)
    assert bundle["coarse_stream_plaintexts"] is True

    evaluator.load_plaintext_diagonals("cached", 0, 1, 88, bundle=bundle)

    assert backend.loaded_batches[-2:] == [
        {"transform_id": 88, "diag_indices": [5], "segments": [[77, 5, 6]]},
        {"transform_id": 88, "diag_indices": [3], "segments": [[77, 3, 4]]},
    ]


def test_dense_plaintext_cache_reads_legacy_fine_diagonal_payloads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_DENSE_LT_COARSE_ARTIFACT_IO", "0")
    backend = _FakeSerializedDenseBackend()
    evaluator = _dense_cache_evaluator(tmp_path, backend)
    with h5py.File(tmp_path / "diags.h5", "w") as handle:
        block = handle.create_group("cached").create_group("plaintexts").create_group("0_1")
        block.create_dataset("5", data=np.asarray([5, 50], dtype=np.uint8))
        block.create_dataset("3", data=np.asarray([3, 30], dtype=np.uint8))

    bundle = evaluator._read_transform_io_bundle("cached", 0, 1, 88, prefetch=False)
    evaluator.load_plaintext_diagonals("cached", 0, 1, 88, bundle=bundle)

    assert backend.loaded_batches[-1] == {
        "transform_id": 88,
        "diag_indices": [3, 5],
        "segments": [[3, 30], [5, 50]],
    }


def test_dense_legacy_fine_payload_streams_when_chunk_limit_is_small(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_DENSE_LT_COARSE_LOAD_CHUNK_BYTES", "3")
    backend = _FakeSerializedDenseBackend()
    evaluator = _dense_cache_evaluator(tmp_path, backend)
    with h5py.File(tmp_path / "diags.h5", "w") as handle:
        block = handle.create_group("cached").create_group("plaintexts").create_group("0_1")
        block.create_dataset("5", data=np.asarray([5, 50], dtype=np.uint8))
        block.create_dataset("3", data=np.asarray([3, 30], dtype=np.uint8))

    bundle = evaluator._read_transform_io_bundle("cached", 0, 1, 88, prefetch=False)
    assert bundle["fine_stream_plaintexts"] is True

    evaluator.load_plaintext_diagonals("cached", 0, 1, 88, bundle=bundle)

    assert backend.loaded_batches[-2:] == [
        {"transform_id": 88, "diag_indices": [3], "segments": [[3, 30]]},
        {"transform_id": 88, "diag_indices": [5], "segments": [[5, 50]]},
    ]


def test_dense_shared_cache_is_opt_in(monkeypatch) -> None:
    class Backend:
        def EvaluateLinearTransformsWithSharedCache(self):
            raise AssertionError("only presence is checked")

    evaluator = object.__new__(NewEvaluator)
    evaluator.backend = Backend()
    evaluator.io_mode = "none"

    monkeypatch.delenv("ORION_DENSE_LT_SHARED_CACHE", raising=False)
    assert evaluator._dense_shared_cache_enabled(rows=2, cols=2) is False

    monkeypatch.setenv("ORION_DENSE_LT_SHARED_CACHE", "0")
    assert evaluator._dense_shared_cache_enabled(rows=2, cols=2) is False

    monkeypatch.setenv("ORION_DENSE_LT_SHARED_CACHE", "1")
    assert evaluator._dense_shared_cache_enabled(rows=2, cols=2) is True


def test_transform_metadata_validation_ignores_rotation_request_order() -> None:
    params = SimpleNamespace(
        get_logscale=lambda: 35,
        get_slots=lambda: 8,
        get_backend=lambda: "python",
    )
    layer = SimpleNamespace(
        name="cached_conv",
        diagonals={(0, 0): {3: [0.0, 0.0], 1: [1.0, 1.0]}},
        transform_ids={(0, 0): 42},
        level=5,
        depth=1,
        input_shape=torch.Size((1, 1, 2, 2)),
        output_shape=torch.Size((1, 1, 2, 2)),
        fhe_input_shape=torch.Size((1, 1, 2, 2)),
        fhe_output_shape=torch.Size((1, 1, 2, 2)),
        bsgs_ratio=2.0,
        output_rotations=0,
        on_weight=torch.empty(1, 1),
        scheme=SimpleNamespace(
            params=params,
            lt_evaluator=SimpleNamespace(
                get_required_rotation_key_requests=lambda _transform_id: (
                    (7, None),
                    (1, 3),
                    (5, None),
                )
            ),
        ),
    )
    metadata = compile_cache.collect_transform_metadata([layer])
    cached_metadata = json.loads(json.dumps(metadata))
    requests = cached_metadata["layers"][0]["blocks"][0]["descriptor"]["rotation_requests"]
    cached_metadata["layers"][0]["blocks"][0]["descriptor"]["rotation_requests"] = list(reversed(requests))

    compile_cache.validate_transform_metadata(
        {"transform_metadata": cached_metadata},
        [layer],
    )


def _cache_config(tmp_path, *, io_mode: str) -> dict:
    return {
        "ckks_params": {
            "LogN": 6,
            "LogQ": [45, 35, 45],
            "LogP": [50],
            "LogScale": 35,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "square",
            "backend": "python",
            "fuse_modules": True,
            "debug": False,
            "io_mode": str(io_mode),
            "diags_path": str(tmp_path / "diags.h5"),
            "keys_path": str(tmp_path / "keys.h5"),
        },
    }


def _tiny_python_config() -> dict:
    return {
        "ckks_params": {
            "LogN": 6,
            "LogQ": [45, 35, 45],
            "LogP": [50],
            "LogScale": 35,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "square",
            "backend": "python",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }


def test_tiny_conv2d_python_backend_matches_clear() -> None:
    torch.manual_seed(3)
    active_scheme = orion.init_scheme(_tiny_python_config())
    try:
        layer = Conv2d(1, 2, kernel_size=3, padding=1, bias=True)
        layer.eval()
        x = torch.randn(1, 1, 4, 4)
        clear = layer(x)

        orion.fit(layer, x)
        input_level = orion.compile(layer)
        encrypted = orion.encrypt(orion.encode(x, input_level))
        layer.he()
        fhe = layer(encrypted).decrypt().decode()

        assert fhe.shape == clear.shape
        assert torch.allclose(fhe, clear, atol=1.0e-4, rtol=1.0e-4)
    finally:
        active_scheme.delete_scheme()


def _tiny_cached_conv(weight: torch.Tensor) -> Conv2d:
    layer = Conv2d(1, 1, kernel_size=1, bias=False)
    layer.weight.data.copy_(weight)
    layer.init_orion_params()
    layer.name = "cached_conv"
    layer.input_shape = torch.Size((1, 1, 2, 2))
    layer.output_shape = torch.Size((1, 1, 2, 2))
    layer.input_gap = 1
    layer.output_gap = 1
    layer.fhe_input_shape = torch.Size((1, 1, 2, 2))
    layer.fhe_output_shape = torch.Size((1, 1, 2, 2))
    layer.input_min = torch.tensor(-1.0)
    layer.input_max = torch.tensor(1.0)
    layer.output_min = torch.tensor(-1.0)
    layer.output_max = torch.tensor(1.0)
    layer.set_level(2)
    return layer


def test_load_mode_reuses_cached_diagonals_without_repacking(tmp_path, monkeypatch) -> None:
    torch.manual_seed(2)
    weight = torch.randn(1, 1, 1, 1)

    save_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="save"))
    Module.set_scheme(save_scheme)
    Module.set_margin(save_scheme.params.get_margin())
    try:
        layer = _tiny_cached_conv(weight)
        layer.generate_diagonals(last=False)
        layer.compile()
        manifest = {
            "schema_version": compile_cache.SCHEMA_VERSION,
            "cache_format_version": compile_cache.CACHE_FORMAT_VERSION,
            "fingerprint": {},
            "bootstrap_plan": {},
            "transform_metadata": compile_cache.collect_transform_metadata([layer]),
            "provider_metadata": {"rows": [], "sha256": ""},
            "sha256": "",
        }
        compile_cache.write_manifest(str(tmp_path / "compile_manifest.json"), manifest)
    finally:
        save_scheme.delete_scheme()

    with h5py.File(tmp_path / "diags.h5", "r") as handle:
        assert "cached_conv" in handle

    load_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="load"))
    Module.set_scheme(load_scheme)
    Module.set_margin(load_scheme.params.get_margin())
    try:
        layer = _tiny_cached_conv(weight)

        def fail_pack_conv2d(*_args, **_kwargs):
            raise AssertionError("load mode should not repack cached conv diagonals")

        monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
        layer.generate_diagonals(last=False)
        assert layer.diagonals == {}
        assert layer.output_rotations == 0
        layer.compile()
        assert layer.transform_ids
    finally:
        load_scheme.delete_scheme()


def test_save_resume_reuses_cached_diagonals_without_repacking(tmp_path, monkeypatch) -> None:
    torch.manual_seed(22)
    weight = torch.randn(1, 1, 1, 1)

    save_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="save"))
    Module.set_scheme(save_scheme)
    Module.set_margin(save_scheme.params.get_margin())
    try:
        layer = _tiny_cached_conv(weight)
        layer.generate_diagonals(last=False)
        layer.compile()
    finally:
        save_scheme.delete_scheme()

    monkeypatch.setenv("ORION_COMPILE_SAVE_RESUME", "1")
    resume_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="save"))
    Module.set_scheme(resume_scheme)
    Module.set_margin(resume_scheme.params.get_margin())
    try:
        layer = _tiny_cached_conv(weight)

        def fail_pack_conv2d(*_args, **_kwargs):
            raise AssertionError("save resume should not repack cached conv diagonals")

        monkeypatch.setattr(packing, "pack_conv2d", fail_pack_conv2d)
        layer.generate_diagonals(last=False)
        assert layer.diagonals == {}
        layer.compile()
        assert layer.transform_ids
    finally:
        resume_scheme.delete_scheme()


def test_load_mode_uses_cached_compile_plan(tmp_path, monkeypatch) -> None:
    torch.manual_seed(11)
    weight = torch.randn(1, 1, 1, 1)
    x = torch.randn(1, 1, 2, 2)

    save_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="save"))
    try:
        layer = Conv2d(1, 1, kernel_size=1, bias=False)
        layer.weight.data.copy_(weight)
        orion.fit(layer, x)
        save_input_level = orion.compile(layer)
    finally:
        save_scheme.delete_scheme()

    manifest_path = tmp_path / "compile_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 3
    assert manifest["cache_format_version"] == "compile-plan-v3"
    assert manifest["bootstrap_plan"]["input_level"] == save_input_level
    assert manifest["transform_metadata"]["layers"]

    def fail_solve(*_args, **_kwargs):
        raise AssertionError("load mode should reuse the cached bootstrap plan")

    monkeypatch.setattr(orion_core.BootstrapSolver, "solve", fail_solve)
    monkeypatch.setattr(
        packing,
        "pack_conv2d",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("load mode should not repack cached conv diagonals")
        ),
    )

    load_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="load"))
    try:
        layer = Conv2d(1, 1, kernel_size=1, bias=False)
        layer.weight.data.copy_(weight)
        orion.fit(layer, x)
        assert orion.compile(layer) == save_input_level
    finally:
        load_scheme.delete_scheme()


def test_load_mode_rejects_v2_compile_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "compile_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cache_format_version": "compile-plan-v2",
                "fingerprint": {},
                "bootstrap_plan": {},
                "transform_metadata": {"layers": []},
                "provider_metadata": {"rows": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Stale compile cache manifest"):
        compile_cache.read_manifest(str(manifest_path))


def test_compile_manifest_keeps_entry_fingerprint_after_compile_mutates_model(tmp_path, monkeypatch) -> None:
    torch.manual_seed(111)
    weight = torch.randn(1, 1, 1, 1)
    x = torch.randn(1, 1, 2, 2)
    original_pack_conv2d = packing.pack_conv2d
    mutated = False

    save_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="save"))
    try:
        layer = Conv2d(1, 1, kernel_size=1, bias=False)
        layer.weight.data.copy_(weight)
        orion.fit(layer, x)
        entry_fingerprint = compile_cache.cache_fingerprint(save_scheme.params, layer)

        def mutating_pack_conv2d(*args, **kwargs):
            nonlocal mutated
            if not mutated:
                layer.add_module("_compile_time_probe", torch.nn.Identity())
                mutated = True
            return original_pack_conv2d(*args, **kwargs)

        monkeypatch.setattr(packing, "pack_conv2d", mutating_pack_conv2d)
        orion.compile(layer)
        mutated_fingerprint = compile_cache.cache_fingerprint(save_scheme.params, layer)
    finally:
        save_scheme.delete_scheme()

    manifest_path = tmp_path / "compile_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert mutated is True
    assert mutated_fingerprint != entry_fingerprint
    assert manifest["fingerprint"] == entry_fingerprint


def test_compile_plan_can_round_trip_residual_helper_nodes() -> None:
    class FakeDag:
        def __init__(self):
            self.nodes = {
                "conv": {
                    "module": SimpleNamespace(level=3, depth=1, scheme=SimpleNamespace(params=SimpleNamespace(get_slots=lambda: 16))),
                    "bootstrap": False,
                },
                "conv_fork": {"module": None, "bootstrap": False},
                "conv_join": {"module": None, "bootstrap": False},
            }

    dag = FakeDag()
    topo_sort = ["conv", "conv_fork", "conv_join"]
    plan = compile_cache.collect_bootstrap_plan(
        dag,
        topo_sort,
        input_level=4,
        bootstrap_count=0,
        bootstrapper_slots=[],
    )

    assert plan["topological_order"] == topo_sort
    assert compile_cache.apply_bootstrap_plan(dag, plan) == (4, 0, [])


def test_load_mode_rejects_compile_manifest_fingerprint_mismatch(tmp_path) -> None:
    torch.manual_seed(12)
    weight = torch.randn(1, 1, 1, 1)
    x = torch.randn(1, 1, 2, 2)

    save_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="save"))
    try:
        layer = Conv2d(1, 1, kernel_size=1, bias=False)
        layer.weight.data.copy_(weight)
        orion.fit(layer, x)
        orion.compile(layer)
    finally:
        save_scheme.delete_scheme()

    manifest_path = tmp_path / "compile_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fingerprint"]["sha256"] = "not-current"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    load_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="load"))
    try:
        layer = Conv2d(1, 1, kernel_size=1, bias=False)
        layer.weight.data.copy_(weight)
        orion.fit(layer, x)
        with pytest.raises(RuntimeError, match="fingerprint mismatch"):
            orion.compile(layer)
    finally:
        load_scheme.delete_scheme()


def test_load_mode_rejects_transform_metadata_mismatch(tmp_path) -> None:
    torch.manual_seed(13)
    weight = torch.randn(1, 1, 1, 1)
    x = torch.randn(1, 1, 2, 2)

    save_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="save"))
    try:
        layer = Conv2d(1, 1, kernel_size=1, bias=False)
        layer.weight.data.copy_(weight)
        orion.fit(layer, x)
        orion.compile(layer)
    finally:
        save_scheme.delete_scheme()

    manifest_path = tmp_path / "compile_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["transform_metadata"]["layers"][0]["level"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    load_scheme = orion.init_scheme(_cache_config(tmp_path, io_mode="load"))
    try:
        layer = Conv2d(1, 1, kernel_size=1, bias=False)
        layer.weight.data.copy_(weight)
        orion.fit(layer, x)
        with pytest.raises(RuntimeError, match="transform metadata mismatch"):
            orion.compile(layer)
    finally:
        load_scheme.delete_scheme()
