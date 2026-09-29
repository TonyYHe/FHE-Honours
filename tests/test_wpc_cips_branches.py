from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import orion
from orion.experimental.wpc_cips_baseline import evaluate_cyclic_diagonals
from orion.experimental.wpc_cips_branches import (
    WPCCIPSConcatPlan,
    WPCCIPSResidualAddPlan,
    build_concat_transforms,
    pack_cips_groups,
    unpack_cips_groups,
)
from orion.nn import Add, Concat


def _signature(
    channels: int,
    *,
    slots: int = 32,
    height: int = 2,
    width: int = 4,
) -> tuple:
    capacity = slots // (height * width)
    ranges = tuple(
        (start, min(channels, start + capacity))
        for start in range(0, channels, capacity)
    )
    return ("wpc_cips", slots, channels, height, width, ranges)


def _producer(channels: int, *, level: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        output_shape=(1, channels, 2, 4),
        output_packing_signature=_signature(channels),
        output_level=level,
    )


def _python_config() -> dict:
    return {
        "ckks_params": {
            "LogN": 6,
            "LogQ": [50, 40, 40],
            "LogP": [50],
            "LogScale": 40,
            "H": 16,
            "RingType": "Standard",
        },
        "orion": {
            "backend": "python",
            "embedding_method": "square",
            "io_mode": "none",
            "debug": False,
        },
    }


def _encrypt(scheme, values: np.ndarray, signature: tuple, *, level: int):
    packed = pack_cips_groups(values, signature)
    plaintext = scheme.encode(torch.tensor(packed, dtype=torch.float32), level=level)
    try:
        ciphertext = scheme.encrypt(plaintext)
    finally:
        plaintext.release()
    ciphertext._wpc_cips_packing_signature = signature
    return ciphertext


def test_residual_plan_requires_identical_shape_signature_and_level() -> None:
    plan = WPCCIPSResidualAddPlan.between_plans(_producer(6), _producer(6))
    assert plan.output_shape == (1, 6, 2, 4)
    assert plan.output_level == 2
    assert plan.output_packing_signature == _signature(6)

    with pytest.raises(ValueError, match="different CKKS levels"):
        WPCCIPSResidualAddPlan.between_plans(_producer(6), _producer(6, level=1))
    with pytest.raises(ValueError, match="different logical shapes"):
        WPCCIPSResidualAddPlan.between_plans(_producer(6), _producer(5))


def test_concat_diagonals_exactly_materialize_cross_group_channel_offsets() -> None:
    rng = np.random.default_rng(20260929)
    signatures = (_signature(6), _signature(2))
    branches = (
        rng.normal(size=(6, 2, 4)),
        rng.normal(size=(2, 2, 4)),
    )
    output_signature, transforms = build_concat_transforms(signatures)
    packed_inputs = [pack_cips_groups(values, signature) for values, signature in zip(branches, signatures)]

    packed_outputs = []
    for output_group in range(len(output_signature[5])):
        accumulated = np.zeros((output_signature[1],), dtype=np.float64)
        for row in transforms.values():
            if int(row["output_group"]) != output_group:
                continue
            accumulated += evaluate_cyclic_diagonals(
                packed_inputs[int(row["branch_index"])][int(row["source_group"])],
                row["diagonals"],
            )
        packed_outputs.append(accumulated)

    expected = np.concatenate(branches, axis=0)
    actual = unpack_cips_groups(np.stack(packed_outputs), output_signature)
    assert output_signature == _signature(8)
    assert len(transforms) == 3
    assert sum(len(row["diagonals"]) for row in transforms.values()) == 3
    assert np.array_equal(actual, expected)


def test_actual_orion_add_and_concat_preserve_cips_without_online_encode() -> None:
    scheme = orion.init_scheme(_python_config())
    Add.set_scheme(scheme)
    residual_module = Add()
    concat_module = Concat(dim=1)
    left_plan = _producer(6)
    right_plan = _producer(6)
    side_plan = _producer(2)
    residual_plan = residual_module.install_wpc_cips_plan(left_plan, right_plan)
    consumer = SimpleNamespace(
        input_shape=(1, 8, 2, 4),
        input_packing_signature=_signature(8),
        level=1,
    )
    concat_plan = concat_module.install_wpc_cips_plan(
        (residual_plan, side_plan),
        consumer_plan=consumer,
    )
    residual_module.he()
    concat_module.he()

    rng = np.random.default_rng(17)
    left_values = rng.normal(0.0, 0.1, size=(6, 2, 4))
    right_values = rng.normal(0.0, 0.1, size=(6, 2, 4))
    side_values = rng.normal(0.0, 0.1, size=(2, 2, 4))
    left = _encrypt(scheme, left_values, _signature(6), level=2)
    right = _encrypt(scheme, right_values, _signature(6), level=2)
    side = _encrypt(scheme, side_values, _signature(2), level=2)
    residual = None
    concatenated = None
    try:
        online_encode_calls = 0
        original_encode = scheme.encoder.encode

        def counted_encode(*args, **kwargs):
            nonlocal online_encode_calls
            online_encode_calls += 1
            return original_encode(*args, **kwargs)

        scheme.encoder.encode = counted_encode
        try:
            residual = residual_module(left, right)
            concatenated = concat_module(residual, side)
        finally:
            scheme.encoder.encode = original_encode

        decoded = concat_plan.decrypt_unpack(concatenated)
        expected = np.concatenate((left_values + right_values, side_values), axis=0)
        assert np.allclose(decoded, expected, rtol=0.0, atol=1e-6)
        assert online_encode_calls == 0
        assert residual_plan.last_evaluation["level_consumption"] == 0
        assert concat_plan.last_evaluation["level_consumption"] == 1
        assert concat_plan.last_evaluation["input_ciphertext_group_count"] == 3
        assert concat_plan.last_evaluation["output_ciphertext_group_count"] == 2
        assert concatenated.level() == 1
    finally:
        for value in (concatenated, residual, side, right, left):
            release = getattr(value, "release", None)
            if callable(release):
                release()
        concat_module.remove_wpc_cips_plan()
        residual_module.remove_wpc_cips_plan()
        scheme.delete_scheme()


def test_concat_rejects_non_channel_dimension_and_consumer_mismatch() -> None:
    scheme = orion.init_scheme(_python_config())
    Add.set_scheme(scheme)
    concat = Concat(dim=2)
    try:
        with pytest.raises(ValueError, match="channel dim=1"):
            concat.install_wpc_cips_plan((_producer(2), _producer(2)))

        plan = WPCCIPSConcatPlan.from_producers((_producer(2), _producer(2)))
        plan.compile(scheme)
        with pytest.raises(ValueError, match="logical shape"):
            plan.validate_consumer(
                SimpleNamespace(
                    input_shape=(1, 5, 2, 4),
                    input_packing_signature=_signature(5),
                    level=1,
                )
            )
        plan.cleanup()
    finally:
        scheme.delete_scheme()
