from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion.experimental.wpc_cips_unet import WPCCIPSBootstrapRefresh


def _signature(
    channels: int = 12,
    *,
    slots: int = 512,
    height: int = 4,
    width: int = 4,
) -> tuple:
    capacity = slots // (height * width)
    ranges = tuple(
        (start, min(channels, start + capacity))
        for start in range(0, channels, capacity)
    )
    return ("wpc_cips", slots, channels, height, width, ranges)


def _producer(*, level: int = 1, channels: int = 12) -> SimpleNamespace:
    return SimpleNamespace(
        output_shape=(1, channels, 4, 4),
        output_packing_signature=_signature(channels),
        output_level=level,
    )


def _consumer(*, level: int = 5, channels: int = 12) -> SimpleNamespace:
    return SimpleNamespace(
        input_shape=(1, channels, 4, 4),
        input_packing_signature=_signature(channels),
        level=level,
    )


def test_bootstrap_refresh_contract_preserves_shape_and_cips_signature() -> None:
    refresh = WPCCIPSBootstrapRefresh.between_plans(
        _producer(),
        _consumer(),
        bootstrap_bound=2.0,
    )

    assert refresh.input_shape == refresh.output_shape == (1, 12, 4, 4)
    assert refresh.input_packing_signature == refresh.output_packing_signature
    assert refresh.input_level == 1
    assert refresh.output_level == 5
    assert refresh.group_count == 1
    assert tuple(refresh.bootstrap.fhe_input_shape) == (1, 512)


def test_bootstrap_refresh_mask_tracks_interleaved_active_channels() -> None:
    refresh = WPCCIPSBootstrapRefresh.between_plans(
        _producer(),
        _consumer(),
        bootstrap_bound=2.0,
    )
    mask = refresh.bootstrap._bootstrap_prescale_active_mask

    assert mask.numel() == 512
    assert int(mask.sum().item()) == 12 * 4 * 4
    assert mask.reshape(16, 32)[:, :12].all()
    assert not mask.reshape(16, 32)[:, 12:].any()


@pytest.mark.parametrize(
    "producer,consumer,message",
    [
        (_producer(channels=8), _consumer(channels=12), "packing signatures differ"),
        (
            _producer(),
            SimpleNamespace(
                input_shape=(1, 12, 2, 8),
                input_packing_signature=_signature(),
                level=5,
            ),
            "logical shapes differ",
        ),
    ],
)
def test_bootstrap_refresh_rejects_incompatible_plan_boundaries(
    producer, consumer, message
) -> None:
    with pytest.raises(ValueError, match=message):
        WPCCIPSBootstrapRefresh.between_plans(
            producer,
            consumer,
            bootstrap_bound=2.0,
        )


@pytest.mark.parametrize(
    "input_level,output_level,bound,message",
    [
        (0, 5, 1.0, "one preprocessing level"),
        (2, 2, 1.0, "restore a higher level"),
        (2, 1, 1.0, "restore a higher level"),
        (1, 5, 0.0, "positive finite"),
    ],
)
def test_bootstrap_refresh_rejects_invalid_levels_and_bounds(
    input_level, output_level, bound, message
) -> None:
    with pytest.raises(ValueError, match=message):
        WPCCIPSBootstrapRefresh(
            logical_shape=(1, 12, 4, 4),
            packing_signature=_signature(),
            input_level=input_level,
            output_level=output_level,
            bootstrap_bound=bound,
        )
