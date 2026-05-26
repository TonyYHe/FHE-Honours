from types import SimpleNamespace

import networkx as nx
import torch

from orion.core.level_dag import LevelDAG
from orion.core.fuser import Fuser
from orion.core.bootstrap_fusion import install_bootstrap_prescale_fusion
from orion.core.tracer import OrionTracer
from orion.nn.activation import ReLU
from orion.nn.operations import Identity


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


def test_relu_prescale_depth_tracks_actual_prescale() -> None:
    relu = ReLU()
    relu.margin = 2
    relu.mult1.input_min = torch.tensor(-0.2)
    relu.mult1.input_max = torch.tensor(0.3)
    relu.fit()

    assert relu.prescale == 1
    assert relu.mult1.depth == 0

    relu.mult1.input_min = torch.tensor(-3.0)
    relu.mult1.input_max = torch.tensor(2.0)
    relu.fit()

    assert relu.prescale != 1
    assert relu.mult1.depth == 1


def test_relu_trace_prescales_only_sign_branch() -> None:
    relu = ReLU(degrees=[3, 3, 3])
    traced = OrionTracer().trace_model(relu)
    edges = {
        (str(node.name), str(user.name))
        for node in traced.graph.nodes
        for user in node.users
    }

    assert ("x", "mult1") in edges
    assert ("x", "identity") in edges
    assert ("identity", "mult2") in edges
    assert ("x", "mult2") not in edges
    assert ("mult1", "sign_acts_0") in edges
    assert ("sign_acts_2", "mult2") in edges
    assert not any(str(node.target) == "<built-in function mul>" for node in traced.graph.nodes)


def test_orion_identity_can_anchor_relu_direct_branch_bootstrap() -> None:
    class Params:
        def get_slots(self):
            return 8

    identity = Identity()
    identity.scheme = SimpleNamespace(params=Params())
    identity.fhe_input_shape = torch.Size([8])

    network_dag = nx.DiGraph()
    network_dag.add_node("identity", module=identity)
    network_dag.add_node("join", module=None)

    cost, boots = LevelDAG(l_eff=4, network_dag=network_dag).estimate_bootstrap_latency(
        "identity@l=2", "join@l=3"
    )

    assert cost > 0
    assert boots == 1


def test_relu_output_bootstrap_fusion_scales_sign_and_biases_output_mult() -> None:
    class Params:
        def get_slots(self):
            return 8

    relu = ReLU(degrees=[3, 3, 3])
    relu.mult2.scheme = SimpleNamespace(params=Params())
    relu.mult2.fhe_output_shape = torch.Size([8])
    bootstrapper = SimpleNamespace(prescale=0.5, constant=1.25)

    assert install_bootstrap_prescale_fusion(relu.mult2, bootstrapper) is True
    assert relu.sign.acts[-1]._bootstrap_output_scale_fusion == 0.5
    assert relu.mult2._bootstrap_output_bias_fusion == 0.625
    assert bootstrapper.preprocess_fused is True

    relu.mult2.he_mode = True
    out = relu.mult2(torch.tensor([2.0]), torch.tensor([3.0]))
    assert torch.equal(out, torch.tensor([6.625]))
