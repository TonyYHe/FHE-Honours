from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import orion
from orion.experimental.wpc_cips_activation import WPCCIPSActivationBootstrap


def _python_config() -> dict:
    return {
        "ckks_params": {
            "LogN": 6,
            "LogQ": [50, 40, 40, 40],
            "LogP": [50],
            "LogScale": 40,
            "H": 16,
            "RingType": "Standard",
        },
        "boot_params": {"LogP": [50, 50]},
        "orion": {
            "margin": 1,
            "backend": "python",
            "embedding_method": "square",
            "io_mode": "none",
            "debug": False,
        },
    }


def _signature() -> tuple:
    return ("wpc_cips", 32, 6, 2, 4, ((0, 4), (4, 6)))


def test_bridge_requires_matching_chainable_plans() -> None:
    producer = SimpleNamespace(
        output_packing_signature=_signature(),
        output_shape=(1, 6, 2, 4),
        output_level=2,
    )
    consumer = SimpleNamespace(
        input_packing_signature=_signature(),
        input_shape=(1, 6, 2, 4),
        level=3,
    )
    bridge = WPCCIPSActivationBootstrap.between_plans(
        producer,
        consumer,
        bootstrap_bound=1.0,
    )
    assert bridge.input_level == 2
    assert bridge.activation_output_level == 1
    assert bridge.output_level == 3
    assert bridge.group_count == 2

    consumer.input_packing_signature = (
        "wpc_cips",
        32,
        5,
        2,
        4,
        ((0, 4), (4, 5)),
    )
    with pytest.raises(ValueError, match="signatures differ"):
        WPCCIPSActivationBootstrap.between_plans(
            producer,
            consumer,
            bootstrap_bound=1.0,
        )


def test_grouped_quad_bootstrap_preserves_cips_contract_without_online_encode() -> None:
    scheme = orion.init_scheme(_python_config())
    bridge = WPCCIPSActivationBootstrap(
        logical_shape=(1, 6, 2, 4),
        packing_signature=_signature(),
        input_level=2,
        output_level=3,
        bootstrap_bound=1.0,
    )
    ciphertext = None
    output = None
    try:
        summary = bridge.compile(scheme)
        assert summary["activation_output_level"] == 1
        assert summary["bootstrap_preprocess_output_level"] == 0
        assert summary["bootstrap_output_level"] == 3
        assert summary["ciphertext_group_count"] == 2
        assert summary["bootstrap_constant"] == 0.0

        values = torch.zeros((2, 32), dtype=torch.float32)
        values[0] = torch.linspace(-0.5, 0.5, 32)
        values[1].reshape(8, 4)[:, :2] = torch.linspace(-0.25, 0.25, 16).reshape(
            8, 2
        )
        plaintext = scheme.encode(values, level=2)
        try:
            ciphertext = scheme.encrypt(plaintext)
        finally:
            plaintext.release()
        ciphertext._wpc_cips_packing_signature = _signature()

        online_encode_calls = 0
        original_encode = scheme.encoder.encode

        def counted_encode(*args, **kwargs):
            nonlocal online_encode_calls
            online_encode_calls += 1
            return original_encode(*args, **kwargs)

        scheme.encoder.encode = counted_encode
        try:
            output = bridge(ciphertext)
        finally:
            scheme.encoder.encode = original_encode

        decoded = scheme.decode(scheme.decrypt(output))
        assert torch.allclose(decoded, values.square(), rtol=0.0, atol=1e-6)
        assert online_encode_calls == 0
        assert output.level() == 3
        assert output._wpc_cips_packing_signature == _signature()
        assert bridge.last_evaluation["packing_signature_preserved"] is True
        assert bridge.last_evaluation["ciphertext_group_count"] == 2
        record = bridge.last_evaluation["bootstrap_runtime_record"]
        assert record["input_before_preprocess"]["id_count"] == 2
        assert record["output_after_postprocess"]["id_count"] == 2
    finally:
        if output is not None:
            output.release()
        if ciphertext is not None:
            ciphertext.release()
        bridge.cleanup()
        scheme.delete_scheme()


def test_bridge_rejects_wrong_level_or_packing_signature() -> None:
    scheme = orion.init_scheme(_python_config())
    bridge = WPCCIPSActivationBootstrap(
        logical_shape=(1, 6, 2, 4),
        packing_signature=_signature(),
        input_level=2,
        output_level=3,
        bootstrap_bound=1.0,
    )
    ciphertext = None
    try:
        bridge.compile(scheme)
        plaintext = scheme.encode(torch.zeros((2, 32)), level=1)
        try:
            ciphertext = scheme.encrypt(plaintext)
        finally:
            plaintext.release()
        ciphertext._wpc_cips_packing_signature = _signature()
        with pytest.raises(ValueError, match="required level 2"):
            bridge(ciphertext)

        ciphertext._wpc_cips_packing_signature = ("wrong",)
        with pytest.raises(ValueError, match="required WPC CIPS packing"):
            bridge(ciphertext)
    finally:
        if ciphertext is not None:
            ciphertext.release()
        bridge.cleanup()
        scheme.delete_scheme()


@pytest.mark.parametrize("bound", [0.0, -1.0, float("inf")])
def test_bridge_rejects_invalid_bootstrap_bound(bound: float) -> None:
    with pytest.raises(ValueError, match="bootstrap_bound"):
        WPCCIPSActivationBootstrap(
            logical_shape=(1, 6, 2, 4),
            packing_signature=_signature(),
            input_level=2,
            output_level=3,
            bootstrap_bound=bound,
        )
