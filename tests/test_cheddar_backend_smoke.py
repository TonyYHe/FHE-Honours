from __future__ import annotations

import ctypes
from types import SimpleNamespace
from pathlib import Path

import h5py
import pytest
import torch

from orion.backend.python import compile_cache
from orion.core.orion import Scheme
from orion.nn.unified_transform import UnifiedTransformGroup


def _backend_available() -> bool:
    backend_dir = Path(__file__).resolve().parents[1] / "orion" / "backend" / "cheddar"
    return (backend_dir / "cheddar-linux.so").exists()


def _config() -> dict:
    return {
        "ckks_params": {
            "LogN": 16,
            "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            "LogP": [61, 61, 61],
            "LogScale": 40,
            "H": 192,
            "RingType": "standard",
        },
        "boot_params": {"LogP": [61] * 8},
        "orion": {
            "backend": "cheddar",
            "io_mode": "none",
            "embedding_method": "hybrid",
            "margin": 2,
            "fuse_modules": True,
            "debug": False,
        },
    }


def _save_config(tmp_path: Path) -> dict:
    config = _config()
    config["orion"] = dict(config["orion"])
    config["orion"]["io_mode"] = "save"
    config["orion"]["diags_path"] = str(tmp_path / "cheddar_diags.h5")
    config["orion"]["keys_path"] = str(tmp_path / "cheddar_keys.h5")
    return config


def _make_fake_lt_layer(scheme: Scheme, *, name: str, include_diagonals: bool) -> SimpleNamespace:
    slots = int(scheme.params.get_slots())
    return SimpleNamespace(
        name=name,
        diagonals={(0, 0): {0: [1.0] * slots, 1: [0.0] * slots}} if include_diagonals else {},
        level=13,
        bsgs_ratio=2.0,
        output_shape=torch.Size([1, slots]),
        fhe_output_shape=torch.Size([1, slots]),
        on_bias=torch.zeros(slots, dtype=torch.float32),
        output_rotations=0,
        input_shape=torch.Size([1, slots]),
        input_min=torch.tensor(-1.0),
        input_max=torch.tensor(1.0),
        output_min=torch.tensor(-1.0),
        output_max=torch.tensor(1.0),
        scheme=scheme,
    )


@pytest.mark.skipif(not _backend_available(), reason="cheddar backend shared library is not built")
def test_cheddar_roundtrip_rotate_poly_and_linear_transform() -> None:
    scheme = Scheme().init_scheme(_config())
    try:
        pt = scheme.encode(torch.tensor([1.0, 2.0, 3.0, 4.0]), level=13)
        ct = scheme.encrypt(pt)

        roundtrip = scheme.decrypt(ct).decode()
        assert torch.allclose(roundtrip[:4], torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-3, rtol=1e-3)

        rot_id = scheme.evaluator.rotate(ct.ids[0], 1, False)
        rot_pt = scheme.backend.Decrypt(rot_id)
        rotated = torch.tensor(scheme.backend.Decode(rot_pt)[:4])
        assert torch.allclose(rotated, torch.tensor([2.0, 3.0, 4.0, 0.0]), atol=1e-3, rtol=1e-3)

        neg_rot_id = scheme.evaluator.rotate(ct.ids[0], -1, False)
        neg_rot_pt = scheme.backend.Decrypt(neg_rot_id)
        neg_rotated = torch.tensor(scheme.backend.Decode(neg_rot_pt)[:4])
        assert torch.allclose(neg_rotated, torch.tensor([0.0, 1.0, 2.0, 3.0]), atol=1e-3, rtol=1e-3)

        poly = scheme.poly_evaluator.generate_monomial(torch.tensor([0.0, 0.0, 1.0]))
        poly_out = scheme.poly_evaluator.evaluate_polynomial(ct, poly)
        squared = scheme.decrypt(poly_out).decode()
        assert torch.allclose(squared[:4], torch.tensor([1.0, 4.0, 9.0, 16.0]), atol=2e-3, rtol=2e-3)

        slots = scheme.params.get_slots()
        transform_id = scheme.backend.GenerateLinearTransform(
            [0, 1],
            [1.0] * slots + [0.0] * slots,
            13,
            2.0,
            "none",
        )
        for key in scheme.backend.GetLinearTransformRotationKeys(transform_id):
            scheme.backend.GenerateLinearTransformRotationKey(int(key))
        lt_ct = scheme.backend.EvaluateLinearTransform(transform_id, ct.ids[0])
        lt_pt = scheme.backend.Decrypt(lt_ct)
        lt_values = torch.tensor(scheme.backend.Decode(lt_pt)[:4])
        assert torch.allclose(lt_values, torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-3, rtol=1e-3)

        neg_transform_id = scheme.backend.GenerateLinearTransform(
            [-1],
            [1.0] * slots,
            13,
            2.0,
            "none",
        )
        for key in scheme.backend.GetLinearTransformRotationKeys(neg_transform_id):
            scheme.backend.GenerateLinearTransformRotationKey(int(key))
        neg_lt_ct = scheme.backend.EvaluateLinearTransform(neg_transform_id, ct.ids[0])
        neg_lt_pt = scheme.backend.Decrypt(neg_lt_ct)
        neg_lt_values = torch.tensor(scheme.backend.Decode(neg_lt_pt)[:4])
        assert torch.allclose(neg_lt_values, torch.tensor([0.0, 1.0, 2.0, 3.0]), atol=1e-3, rtol=1e-3)
    finally:
        scheme.delete_scheme()


@pytest.mark.skipif(not _backend_available(), reason="cheddar backend shared library is not built")
def test_cheddar_unified_shared_cache_matches_individual_evaluation() -> None:
    scheme = Scheme().init_scheme(_config())
    try:
        pt = scheme.encode(torch.tensor([1.0, 2.0, 3.0, 4.0]), level=13)
        ct = scheme.encrypt(pt)
        slots = scheme.params.get_slots()

        transform_a = scheme.backend.GenerateLinearTransform(
            [0, 1],
            [1.0] * slots + [0.0] * slots,
            13,
            2.0,
            "none",
        )
        transform_b = scheme.backend.GenerateLinearTransform(
            [0, 1],
            [0.0] * slots + [1.0] * slots,
            13,
            2.0,
            "none",
        )

        transform_ids = [int(transform_a), int(transform_b)]
        transform_ids_array = (ctypes.c_int * len(transform_ids))(*transform_ids)
        shared_ids = scheme.backend.EvaluateLinearTransformsWithSharedCache(
            transform_ids_array, len(transform_ids), int(ct.ids[0])
        )
        direct_ids = [
            int(scheme.backend.EvaluateLinearTransform(transform_a, int(ct.ids[0]))),
            int(scheme.backend.EvaluateLinearTransform(transform_b, int(ct.ids[0]))),
        ]

        shared_a = torch.tensor(scheme.backend.Decode(scheme.backend.Decrypt(int(shared_ids[0])))[:8])
        shared_b = torch.tensor(scheme.backend.Decode(scheme.backend.Decrypt(int(shared_ids[1])))[:8])
        direct_a = torch.tensor(scheme.backend.Decode(scheme.backend.Decrypt(int(direct_ids[0])))[:8])
        direct_b = torch.tensor(scheme.backend.Decode(scheme.backend.Decrypt(int(direct_ids[1])))[:8])

        assert torch.allclose(shared_a, direct_a, atol=1e-3, rtol=1e-3)
        assert torch.allclose(shared_b, direct_b, atol=1e-3, rtol=1e-3)
    finally:
        scheme.delete_scheme()


@pytest.mark.skipif(not _backend_available(), reason="cheddar backend shared library is not built")
def test_cheddar_shared_cache_lowers_to_input_level() -> None:
    scheme = Scheme().init_scheme(_config())
    try:
        slots = int(scheme.params.get_slots())
        x = torch.zeros(slots, dtype=torch.float32)
        x[:8] = torch.arange(1, 9, dtype=torch.float32)
        ct = scheme.encrypt(scheme.encode(x, level=1))

        diag0 = torch.ones(slots, dtype=torch.float32)
        diag1 = torch.zeros(slots, dtype=torch.float32)
        diag1[:16] = 0.25
        transform_ids = []
        for scale in (1.0, 2.0):
            transform_ids.append(
                int(
                    scheme.backend.GenerateLinearTransform(
                        [0, 1],
                        (diag0 * scale).tolist() + (diag1 * scale).tolist(),
                        2,
                        2.0,
                        "none",
                    )
                )
            )

        transform_ids_array = (ctypes.c_int * len(transform_ids))(*transform_ids)
        shared_ids = scheme.backend.EvaluateLinearTransformsWithSharedCache(
            transform_ids_array, len(transform_ids), int(ct.ids[0])
        )

        expected = x + torch.roll(x, -1) * 0.25
        values_a = torch.tensor(scheme.backend.Decode(scheme.backend.Decrypt(int(shared_ids[0])))[:8])
        values_b = torch.tensor(scheme.backend.Decode(scheme.backend.Decrypt(int(shared_ids[1])))[:8])

        assert int(scheme.backend.GetCiphertextLevel(int(shared_ids[0]))) == 0
        assert int(scheme.backend.GetCiphertextLevel(int(shared_ids[1]))) == 0
        assert torch.allclose(values_a, expected[:8], atol=1e-3, rtol=1e-3)
        assert torch.allclose(values_b, (2.0 * expected)[:8], atol=2e-3, rtol=1e-3)
    finally:
        scheme.delete_scheme()


@pytest.mark.skipif(not _backend_available(), reason="cheddar backend shared library is not built")
def test_cheddar_linear_transform_uses_bsgs_and_level_specific_key_requests() -> None:
    scheme = Scheme().init_scheme(_config())
    try:
        slots = scheme.params.get_slots()
        diag_idxs = list(range(0, 17 * 64, 64))
        transform_id = scheme.backend.GenerateLinearTransform(
            diag_idxs,
            [0.0] * (len(diag_idxs) * slots),
            7,
            2.0,
            "none",
        )

        flat_requests = scheme.backend.GetLinearTransformRotationKeyRequests(transform_id)
        requests = [(int(flat_requests[i]), int(flat_requests[i + 1])) for i in range(0, len(flat_requests), 2)]

        assert requests
        assert all(level == 7 for _key, level in requests)
        assert len(requests) < len([idx for idx in diag_idxs if idx != 0])
    finally:
        scheme.delete_scheme()


@pytest.mark.skipif(not _backend_available(), reason="cheddar backend shared library is not built")
def test_cheddar_bsgs_multi_diagonal_linear_transform_matches_expected() -> None:
    scheme = Scheme().init_scheme(_config())
    try:
        slots = int(scheme.params.get_slots())
        level = 4
        x = torch.zeros(slots, dtype=torch.float32)
        x[:32] = torch.arange(1, 33, dtype=torch.float32)
        ct = scheme.encrypt(scheme.encode(x, level))

        diag_idxs = list(range(10))
        diag_data = []
        expected = torch.zeros(slots, dtype=torch.float32)
        for offset in diag_idxs:
            diag = torch.zeros(slots, dtype=torch.float32)
            diag[:64] = 1.0 / float(offset + 1)
            diag_data.extend(diag.tolist())
            expected += torch.roll(x, -offset) * diag

        transform_id = scheme.backend.GenerateLinearTransform(
            diag_idxs,
            diag_data,
            level,
            2.0,
            "none",
        )
        flat_requests = list(scheme.backend.GetLinearTransformRotationKeyRequests(transform_id))
        for index in range(0, len(flat_requests), 2):
            scheme.backend.GenerateLinearTransformRotationKeyAtLevel(
                int(flat_requests[index]),
                int(flat_requests[index + 1]),
            )

        out_id = scheme.backend.EvaluateLinearTransform(transform_id, int(ct.ids[0]))
        values = torch.tensor(scheme.backend.Decode(scheme.backend.Decrypt(int(out_id)))[:32])
        assert torch.allclose(values, expected[:32], atol=1e-3, rtol=1e-3)
    finally:
        scheme.delete_scheme()


@pytest.mark.skipif(not _backend_available(), reason="cheddar backend shared library is not built")
def test_cheddar_unified_transform_group_save_mode_offloads_and_recovers(tmp_path: Path) -> None:
    scheme = Scheme().init_scheme(_save_config(tmp_path))
    try:
        slots = int(scheme.params.get_slots())
        level = 13
        transform_a = SimpleNamespace(
            diagonals={(0, 0): {0: [1.0] * slots, 1: [0.0] * slots}},
            level=level,
            scheme=scheme,
            fhe_output_shape=torch.Size([1, slots]),
            output_shape=torch.Size([1, slots]),
        )
        transform_b = SimpleNamespace(
            diagonals={(0, 0): {0: [0.0] * slots, 1: [1.0] * slots}},
            level=level,
            scheme=scheme,
            fhe_output_shape=torch.Size([1, slots]),
            output_shape=torch.Size([1, slots]),
        )

        group = UnifiedTransformGroup([transform_a, transform_b])
        group.compile_unified(scheme.backend)

        assert Path(scheme.params.get_diags_path()).exists()
        assert Path(scheme.params.get_keys_path()).exists()
        with h5py.File(scheme.params.get_diags_path(), "r") as handle:
            storage = handle[group._storage_root_name()][group._storage_key]
            for transform_id in group.unified_ids:
                transform_storage = storage[str(int(transform_id))]
                assert (
                    "__encoded_hoist_payload__" in transform_storage
                    or "diag_payload" in transform_storage
                )

        x = torch.zeros(slots, dtype=torch.float32)
        x[:4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ct = scheme.encrypt(scheme.encode(x, level))
        output_ids = group.evaluate_unified(int(ct.ids[0]), scheme.backend)

        decoded = []
        for output_id in output_ids:
            pt_id = scheme.backend.Decrypt(int(output_id))
            decoded.append(torch.tensor(scheme.backend.Decode(pt_id)[:4]))

        assert torch.allclose(decoded[0], torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-3, rtol=1e-3)
        assert torch.allclose(decoded[1], torch.tensor([2.0, 3.0, 4.0, 0.0]), atol=1e-3, rtol=1e-3)
    finally:
        scheme.delete_scheme()


@pytest.mark.skipif(not _backend_available(), reason="cheddar backend shared library is not built")
def test_cheddar_regular_linear_transform_save_and_load_end_to_end(tmp_path: Path) -> None:
    save_scheme = Scheme().init_scheme(_save_config(tmp_path))
    save_out = None
    save_ct = None
    try:
        slots = int(save_scheme.params.get_slots())
        layer = _make_fake_lt_layer(save_scheme, name="fake_lt", include_diagonals=True)

        save_scheme.lt_evaluator.save_transforms(layer)
        diags_path = Path(save_scheme.params.get_diags_path())
        keys_path = Path(save_scheme.params.get_keys_path())
        assert diags_path.exists()
        assert keys_path.exists()

        with h5py.File(diags_path, "r") as handle:
            assert "fake_lt" in handle
            assert "diagonals" in handle["fake_lt"]

        layer.diagonals, layer.on_bias, layer.output_rotations = save_scheme.lt_evaluator.load_transforms(layer)
        layer.transform_ids = save_scheme.lt_evaluator.generate_transforms(layer)
        compile_cache.write_manifest(
            str(tmp_path / "compile_manifest.json"),
            {
                "schema_version": compile_cache.SCHEMA_VERSION,
                "cache_format_version": compile_cache.CACHE_FORMAT_VERSION,
                "fingerprint": {},
                "bootstrap_plan": {},
                "transform_metadata": compile_cache.collect_transform_metadata([layer]),
                "provider_metadata": {"rows": [], "sha256": ""},
                "sha256": "",
            },
        )

        with h5py.File(diags_path, "r") as handle:
            assert "plaintexts" in handle["fake_lt"]
            assert "0_0" in handle["fake_lt"]["plaintexts"]
            block = handle["fake_lt"]["plaintexts"]["0_0"]
            assert "__encoded_hoist_payload__" in block or len(block.keys()) > 0

        with h5py.File(keys_path, "r") as handle:
            assert len(handle.keys()) >= 2

        x = torch.zeros(slots, dtype=torch.float32)
        x[:4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        save_ct = save_scheme.encrypt(save_scheme.encode(x, 13))
        save_out = save_scheme.lt_evaluator.evaluate_transforms(layer, save_ct)
        save_decoded = save_scheme.decrypt(save_out).decode().reshape(-1)
        assert torch.allclose(save_decoded[:4], x[:4], atol=1e-3, rtol=1e-3)
    finally:
        if save_out is not None:
            del save_out
        if save_ct is not None:
            del save_ct
        save_scheme.delete_scheme()

    load_config = _save_config(tmp_path)
    load_config["orion"] = dict(load_config["orion"])
    load_config["orion"]["io_mode"] = "load"
    load_scheme = Scheme().init_scheme(load_config)
    load_out = None
    load_ct = None
    try:
        slots = int(load_scheme.params.get_slots())
        layer = _make_fake_lt_layer(load_scheme, name="fake_lt", include_diagonals=False)
        layer.diagonals, layer.on_bias, layer.output_rotations = load_scheme.lt_evaluator.load_transforms(layer)
        layer.transform_ids = load_scheme.lt_evaluator.generate_transforms(layer)

        x = torch.zeros(slots, dtype=torch.float32)
        x[:4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        load_ct = load_scheme.encrypt(load_scheme.encode(x, 13))
        load_out = load_scheme.lt_evaluator.evaluate_transforms(layer, load_ct)
        load_decoded = load_scheme.decrypt(load_out).decode().reshape(-1)
        assert torch.allclose(load_decoded[:4], x[:4], atol=1e-3, rtol=1e-3)
    finally:
        if load_out is not None:
            del load_out
        if load_ct is not None:
            del load_ct
        load_scheme.delete_scheme()
