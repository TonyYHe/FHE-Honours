from __future__ import annotations

from types import SimpleNamespace

import torch

from orion.core.auto_bootstrap import BootstrapPlacer
from orion.nn.operations import Bootstrap


class _DummyEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def get_moduli_chain(self):
        return [101, 202, 303, 404]

    def encode(self, vec, level, scale):
        self.calls.append((int(level), int(scale), int(vec.numel())))
        return SimpleNamespace(level=int(level), scale=int(scale), size=int(vec.numel()))


def test_bootstrap_prescale_plaintexts_are_cached_per_runtime_level() -> None:
    bootstrap = Bootstrap(input_min=torch.tensor(-1.0), input_max=torch.tensor(1.0), input_level=1)
    encoder = _DummyEncoder()
    bootstrap.scheme = SimpleNamespace(encoder=encoder)
    bootstrap.fhe_input_shape = torch.Size([1, 4, 4, 4])
    bootstrap.prescale = 0.25

    bootstrap.compile()

    assert encoder.calls == [(1, 202, 64)]
    cached = bootstrap._get_prescale_ptxt(1)
    assert cached.level == 1
    assert encoder.calls == [(1, 202, 64)]

    higher_level = bootstrap._get_prescale_ptxt(2)
    assert higher_level.level == 2
    assert encoder.calls == [(1, 202, 64), (2, 303, 64)]


def test_bootstrap_placer_uses_runtime_materialized_output_shape() -> None:
    class _RuntimeExecutor:
        def runtime_fhe_output_shape(self):
            return torch.Size([1, 1, 6, 4])

    module = SimpleNamespace(
        fhe_output_shape=torch.Size([1, 1, 4, 4]),
        region_runtime=SimpleNamespace(executor=_RuntimeExecutor()),
    )
    placer = BootstrapPlacer(net=SimpleNamespace(), network_dag=None)

    assert tuple(int(value) for value in placer._runtime_fhe_output_shape(module)) == (1, 1, 6, 4)
