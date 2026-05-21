from types import SimpleNamespace

import torch

from orion.core.fuser import Fuser


def test_linear_chebyshev_fusion_keeps_depth_when_prescale_is_identity() -> None:
    linear = SimpleNamespace(on_weight=torch.ones(2), on_bias=torch.zeros(2))
    cheb = SimpleNamespace(prescale=1, constant=0, depth=3, fused=False)

    Fuser(SimpleNamespace())._fuse_linear_chebyshev(linear, cheb)

    assert cheb.fused is True
    assert cheb.depth == 3
    assert torch.equal(linear.on_weight, torch.ones(2))
    assert torch.equal(linear.on_bias, torch.zeros(2))


def test_linear_chebyshev_fusion_removes_only_real_prescale_depth() -> None:
    linear = SimpleNamespace(on_weight=torch.ones(2), on_bias=torch.zeros(2))
    cheb = SimpleNamespace(prescale=0.5, constant=0.25, depth=4, fused=False)

    Fuser(SimpleNamespace())._fuse_linear_chebyshev(linear, cheb)

    assert cheb.fused is True
    assert cheb.depth == 3
    assert torch.equal(linear.on_weight, torch.full((2,), 0.5))
    assert torch.equal(linear.on_bias, torch.full((2,), 0.25))


def test_batchnorm_chebyshev_fusion_keeps_depth_when_prescale_is_identity() -> None:
    bn = SimpleNamespace(
        affine=True,
        on_weight=torch.ones(2),
        on_bias=torch.zeros(2),
        num_features=2,
    )
    cheb = SimpleNamespace(prescale=1, constant=0, depth=3, fused=False)

    Fuser(SimpleNamespace())._fuse_bn_chebyshev(bn, cheb)

    assert cheb.fused is True
    assert cheb.depth == 3
    assert torch.equal(bn.on_weight, torch.ones(2))
    assert torch.equal(bn.on_bias, torch.zeros(2))
