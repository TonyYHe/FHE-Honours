from __future__ import annotations

import math

import pytest
import torch

import orion
from orion.experimental.wpc_cips_checkpoint import (
    CheckpointChebyshevSpec,
    WPCCIPSTrainedActivationBootstrap,
)


def _state(*, alpha: float = 1.0) -> dict[str, torch.Tensor]:
    return {
        "dec1a_act.coeffs": torch.tensor([0.125, 0.5, 0.125, -0.01]),
        "dec1a_act.log_postscale": torch.tensor(math.log(8.0)),
        "dec1a_act.log_prescale": torch.tensor(math.log(0.125)),
        "dec1a_act.blend_alpha": torch.tensor(alpha),
    }


def _signature() -> tuple:
    return ("wpc_cips", 32, 6, 2, 4, ((0, 4), (4, 6)))


def _config() -> dict:
    return {
        "ckks_params": {
            "LogN": 6,
            "LogQ": [50, 40, 40, 40, 40, 40, 40],
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


def test_checkpoint_spec_reads_exact_scales_and_chebyshev_formula() -> None:
    spec = CheckpointChebyshevSpec.from_state_dict(_state(), "dec1a_act")
    assert spec.degree == 3
    assert spec.coefficients == pytest.approx((0.125, 0.5, 0.125, -0.01))
    assert spec.postscale == pytest.approx(8.0)
    assert spec.prescale == pytest.approx(0.125)

    value = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)
    z = value * 0.125
    expected = 8.0 * (
        0.125
        + 0.5 * z
        + 0.125 * (2.0 * z.square() - 1.0)
        - 0.01 * (4.0 * z.pow(3) - 3.0 * z)
    )
    assert torch.allclose(spec.evaluate(value), expected, rtol=0.0, atol=1e-7)


def test_checkpoint_spec_rejects_plaintext_reference_blending() -> None:
    with pytest.raises(ValueError, match="blend_alpha=1"):
        CheckpointChebyshevSpec.from_state_dict(_state(alpha=0.5), "dec1a_act")


def test_trained_bridge_preserves_grouped_cips_contract() -> None:
    spec = CheckpointChebyshevSpec.from_state_dict(_state(), "dec1a_act")
    bridge = WPCCIPSTrainedActivationBootstrap(
        logical_shape=(1, 6, 2, 4),
        packing_signature=_signature(),
        input_level=4,
        output_level=6,
        activation_spec=spec,
        bootstrap_bound=2.0,
    )
    assert bridge.activation.depth == 3
    assert bridge.activation_output_level == 1
    assert bridge.output_level == 6
    assert bridge.group_count == 2
    assert tuple(bridge.bootstrap.fhe_input_shape) == (2, 32)
    mask = bridge.bootstrap._bootstrap_prescale_active_mask
    assert int(mask.numel()) == 64
    assert int(mask.sum().item()) == 48
    assert mask[:32].all()
    assert mask[32:].reshape(8, 4)[:, :2].all()
    assert not mask[32:].reshape(8, 4)[:, 2:].any()


def test_trained_bridge_rejects_insufficient_levels() -> None:
    spec = CheckpointChebyshevSpec.from_state_dict(_state(), "dec1a_act")
    with pytest.raises(ValueError, match="enough levels"):
        WPCCIPSTrainedActivationBootstrap(
            logical_shape=(1, 6, 2, 4),
            packing_signature=_signature(),
            input_level=3,
            output_level=6,
            activation_spec=spec,
            bootstrap_bound=2.0,
        )
