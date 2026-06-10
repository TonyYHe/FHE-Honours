from __future__ import annotations

from types import SimpleNamespace

import torch

from orion.core.auto_bootstrap import BootstrapPlacer
from orion.nn.activation import Chebyshev
from orion.nn.linear import Conv2d
from orion.nn.operations import Bootstrap
from tools.run_lattigo_e2e_compare import _layer_mae_target_names


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


def test_layer_mae_targets_exclude_modules_fused_after_compile() -> None:
    net = torch.nn.Sequential()
    conv = Conv2d(1, 1, kernel_size=1)
    act = Chebyshev(degree=1, fn=lambda x: x)
    net.add_module("conv", conv)
    net.add_module("act", act)

    assert _layer_mae_target_names(net) == ["conv", "act"]

    act.fused = True

    assert _layer_mae_target_names(net) == ["conv"]


def test_masked_producer_affine_matches_unfused_bootstrap_preprocess_algebra() -> None:
    values = torch.tensor(
        [
            [0.25, -0.5, 12_345_678.0, -12_345_678.0],
            [0.75, 1.25, 0.0, -50.0],
        ],
        dtype=torch.float32,
    )
    active_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
        ],
        dtype=torch.bool,
    )
    scale = 0.25
    constant = -1.5
    scalar_bias = float(scale * constant)

    unfused_preboot = (values + float(constant)) * float(scale)
    unfused_preboot = torch.where(active_mask, unfused_preboot, torch.zeros_like(unfused_preboot))

    masked_fused = torch.where(
        active_mask,
        values * float(scale) + float(scalar_bias),
        torch.zeros_like(values),
    )
    scalar_fused = values * float(scale) + float(scalar_bias)

    assert torch.equal(masked_fused, unfused_preboot)
    assert float((scalar_fused[~active_mask] - unfused_preboot[~active_mask]).abs().max().item()) > 1.0


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


def test_bootstrap_placer_marks_full_slot_prescale_fusion() -> None:
    class _Params:
        def get_slots(self):
            return 64

    class _RuntimeExecutor:
        def runtime_fhe_output_shape(self):
            return torch.Size([1, 64])

        def bootstrap_prescale_fusion_capable(self):
            return True

    encoder = _DummyEncoder()
    scheme = SimpleNamespace(encoder=encoder, params=_Params())
    module = SimpleNamespace(
        level=3,
        depth=1,
        output_min=torch.tensor(-4.0),
        output_max=torch.tensor(4.0),
        fhe_output_shape=torch.Size([1, 64]),
        scheme=scheme,
        region_runtime=SimpleNamespace(executor=_RuntimeExecutor()),
    )
    placer = BootstrapPlacer(net=SimpleNamespace(scheme=scheme, margin=1.0), network_dag=None)

    bootstrap = placer._create_bootstrapper(module)

    assert bootstrap.preprocess_fused is True
    assert module._bootstrap_prescale_fusion["scale"] == bootstrap.prescale
    assert module._bootstrap_prescale_fusion["bias"] == bootstrap.prescale * bootstrap.constant


def test_bootstrap_placer_respects_prescale_fusion_disable_env(monkeypatch) -> None:
    class _Params:
        def get_slots(self):
            return 64

    class _RuntimeExecutor:
        def runtime_fhe_output_shape(self):
            return torch.Size([1, 64])

        def bootstrap_prescale_fusion_capable(self):
            return True

    monkeypatch.setenv("ORION_DISABLE_BOOTSTRAP_PRESCALE_FUSION", "1")

    encoder = _DummyEncoder()
    scheme = SimpleNamespace(encoder=encoder, params=_Params())
    module = SimpleNamespace(
        level=3,
        depth=1,
        output_min=torch.tensor(-4.0),
        output_max=torch.tensor(4.0),
        fhe_output_shape=torch.Size([1, 64]),
        scheme=scheme,
        region_runtime=SimpleNamespace(executor=_RuntimeExecutor()),
    )
    placer = BootstrapPlacer(net=SimpleNamespace(scheme=scheme, margin=1.0), network_dag=None)

    bootstrap = placer._create_bootstrapper(module)

    assert bootstrap.preprocess_fused is False
    assert not hasattr(module, "_bootstrap_prescale_fusion")


def test_bootstrap_placer_keeps_sparse_prescale_in_bootstrap() -> None:
    class _Params:
        def get_slots(self):
            return 64

    class _RuntimeExecutor:
        def runtime_fhe_output_shape(self):
            return torch.Size([1, 16])

        def bootstrap_prescale_fusion_capable(self):
            return True

    encoder = _DummyEncoder()
    scheme = SimpleNamespace(encoder=encoder, params=_Params())
    module = SimpleNamespace(
        level=3,
        depth=1,
        output_min=torch.tensor(-4.0),
        output_max=torch.tensor(4.0),
        fhe_output_shape=torch.Size([1, 16]),
        scheme=scheme,
        region_runtime=SimpleNamespace(executor=_RuntimeExecutor()),
    )
    placer = BootstrapPlacer(net=SimpleNamespace(scheme=scheme, margin=1.0), network_dag=None)

    bootstrap = placer._create_bootstrapper(module)

    assert bootstrap.preprocess_fused is False
    assert not hasattr(module, "_bootstrap_prescale_fusion")


def test_chebyshev_compile_fuses_bootstrap_output_affine() -> None:
    class _PolyEvaluator:
        def __init__(self) -> None:
            self.coeffs = None

        def generate_chebyshev(self, coeffs):
            self.coeffs = list(coeffs)
            return list(coeffs)

    evaluator = _PolyEvaluator()
    activation = Chebyshev(degree=1, fn=lambda x: x)
    activation.coeffs = [2.0, 3.0]
    activation.scheme = SimpleNamespace(poly_evaluator=evaluator)
    activation._bootstrap_prescale_fusion = {"scale": 0.5, "bias": 1.25}

    activation.compile()

    assert evaluator.coeffs == [2.25, 1.5]


def test_chebyshev_compile_fuses_bootstrap_output_scale() -> None:
    class _PolyEvaluator:
        def __init__(self) -> None:
            self.coeffs = None

        def generate_chebyshev(self, coeffs):
            self.coeffs = list(coeffs)
            return list(coeffs)

    evaluator = _PolyEvaluator()
    activation = Chebyshev(degree=1, fn=lambda x: x)
    activation.coeffs = [2.0, 3.0]
    activation.scheme = SimpleNamespace(poly_evaluator=evaluator)
    activation._bootstrap_output_scale_fusion = 0.25

    activation.compile()

    assert evaluator.coeffs == [0.5, 0.75]
