from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import orion
import orion.nn as on
from orion.core import packing
from orion.core.network_dag import NetworkDAG
from orion.core.orion import scheme
from orion.core.tracer import OrionTracer, StatsTracker
from orion.backend.python.tensors import CipherTensor
from orion.models.unet import MiniUNet
from orion.nn.linear import ConvTranspose2d
from orion.nn.module import Module
from orion.nn.operations import Concat


PYTHON_BACKEND_CONFIG = {
    "ckks_params": {
        "LogN": 10,
        "LogQ": [45, 35, 35, 35, 35, 45],
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


def _python_backend_config(*, embedding_method: str = "square") -> dict:
    config = copy.deepcopy(PYTHON_BACKEND_CONFIG)
    config["orion"]["embedding_method"] = str(embedding_method)
    return config


def _pad_to_fhe(matrix: torch.Tensor, fhe_shape: torch.Size) -> torch.Tensor:
    out = torch.zeros(fhe_shape, dtype=matrix.dtype)
    out[:, : matrix.shape[1], : matrix.shape[2], : matrix.shape[3]] = matrix
    return out


def _encode_layer_input(layer: ConvTranspose2d, x: torch.Tensor) -> CipherTensor:
    level = len(scheme.params.get_logq()) - 1
    packed = packing.multiplex(x, int(layer.input_gap))
    target = torch.zeros(tuple(int(v) for v in layer.fhe_input_shape), dtype=torch.float32)
    target[:, : packed.shape[1], : packed.shape[2], : packed.shape[3]] = packed
    flat = target.flatten()
    ids: list[int] = []
    slots = int(scheme.params.get_slots())
    for start in range(0, int(flat.numel()), int(slots)):
        block = flat[int(start) : int(min(int(flat.numel()), int(start + int(slots))))]
        padded = torch.zeros((int(slots),), dtype=torch.float32)
        padded[: int(block.numel())] = block
        ct = scheme.encrypt(scheme.encode(padded, level))
        ids.append(int(ct.ids[0]))
        ct.ids = []
    return CipherTensor(scheme, ids, layer.input_shape, layer.fhe_input_shape)


def _decode_layer_output(layer: ConvTranspose2d, out: CipherTensor) -> torch.Tensor:
    decoded = out.decrypt().decode().detach().cpu()
    if torch.is_complex(decoded):
        decoded = decoded.real
    packed = decoded.to(dtype=torch.float32).reshape(tuple(int(v) for v in layer.fhe_output_shape))
    return packing._demultiplex(
        packed,
        int(layer.output_gap),
        int(layer.output_shape[1]),
        int(layer.output_shape[2]),
        int(layer.output_shape[3]),
    )


@pytest.mark.parametrize(
    ("seed", "spec"),
    [
        (
            0,
            {
                "in_channels": 4,
                "out_channels": 3,
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "output_padding": 0,
                "groups": 1,
                "input_shape": (1, 4, 4, 4),
                "input_gap": 2,
            },
        ),
        (
            1,
            {
                "in_channels": 4,
                "out_channels": 6,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 0,
                "groups": 2,
                "input_shape": (1, 4, 5, 7),
                "input_gap": 1,
            },
        ),
        (
            2,
            {
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 1,
                "groups": 1,
                "input_shape": (1, 4, 3, 5),
                "input_gap": 1,
            },
        ),
    ],
)
def test_conv_transpose2d_toeplitz_matches_pytorch_in_multiplexed_space(seed: int, spec: dict) -> None:
    torch.manual_seed(seed)

    layer = ConvTranspose2d(
        spec["in_channels"],
        spec["out_channels"],
        kernel_size=spec["kernel_size"],
        stride=spec["stride"],
        padding=spec["padding"],
        output_padding=spec["output_padding"],
        groups=spec["groups"],
        bias=True,
    )
    layer.eval()
    layer.init_orion_params()

    x = torch.randn(*spec["input_shape"])
    y = layer(x)

    layer.input_shape = x.shape
    layer.output_shape = y.shape
    layer.input_gap = int(spec["input_gap"])
    layer.output_gap = max(1, layer.input_gap // int(spec["stride"]))

    on_ci = (spec["in_channels"] + layer.input_gap ** 2 - 1) // (layer.input_gap ** 2)
    on_co = (spec["out_channels"] + layer.output_gap ** 2 - 1) // (layer.output_gap ** 2)
    layer.fhe_input_shape = torch.Size((1, on_ci, x.shape[2] * layer.input_gap, x.shape[3] * layer.input_gap))
    layer.fhe_output_shape = torch.Size((1, on_co, y.shape[2] * layer.output_gap, y.shape[3] * layer.output_gap))

    toeplitz = packing.construct_conv_transpose2d_toeplitz(layer)
    bias = packing.construct_conv_transpose2d_bias(layer)

    x_fhe = _pad_to_fhe(packing.multiplex(x, layer.input_gap), layer.fhe_input_shape)
    expected = _pad_to_fhe(packing.multiplex(y, layer.output_gap), layer.fhe_output_shape)
    actual = torch.tensor(toeplitz @ x_fhe.flatten().numpy(), dtype=torch.float32).reshape(layer.fhe_output_shape)
    actual = actual + bias.reshape(layer.fhe_output_shape)

    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.parametrize("embedding_method", ["square", "hybrid"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_mini_unet_runs_end_to_end_on_python_backend(embedding_method: str, batch_size: int) -> None:
    torch.manual_seed(0)

    scheme = orion.init_scheme(_python_backend_config(embedding_method=embedding_method))
    try:
        net = MiniUNet(in_channels=1, base_channels=4, out_channels=1)
        net.eval()

        x = torch.randn(batch_size, 1, 8, 8)
        clear = net(x)

        orion.fit(net, x)
        input_level = orion.compile(net)

        assert type(net.add).__name__ == "Add"
        assert not getattr(net.out, "_concat_transform_ids_by_input", [])

        encrypted = orion.encrypt(orion.encode(x, input_level))
        net.he()
        fhe = net(encrypted).decrypt().decode()

        assert fhe.shape == clear.shape
        assert torch.allclose(fhe, clear, atol=1.0e-4, rtol=1.0e-4)
    finally:
        scheme.delete_scheme()


def test_concat_stats_tracker_records_channel_sum_and_fusion_specs() -> None:
    class TinyConcatNet(on.Module):
        def __init__(self):
            super().__init__()
            self.left = on.Conv2d(1, 4, kernel_size=1, bias=True)
            self.right = on.Conv2d(1, 4, kernel_size=1, bias=True)
            self.add = on.Concat(dim=1)
            self.out = on.Conv2d(8, 1, kernel_size=1, bias=True)

        def forward(self, x):
            return self.out(self.add(self.left(x), self.right(x)))

    torch.manual_seed(0)
    net = TinyConcatNet()
    traced = OrionTracer().trace_model(net)
    StatsTracker(traced).propagate(torch.randn(1, 1, 8, 8))
    dag = NetworkDAG(traced)
    dag.build_dag()

    concat = dag.nodes["add"]["module"]
    out = dag.nodes["out"]["module"]

    assert type(concat).__name__ == "Concat"
    assert tuple(concat.output_shape) == (1, 8, 8, 8)
    assert tuple(concat.fhe_output_shape) == (1, 8, 8, 8)
    assert int(concat.output_gap) == 1
    assert len(getattr(out, "concat_fusion_specs", ()) or ()) == 2
    assert [spec["channel_start"] for spec in out.concat_fusion_specs] == [0, 4]
    assert [spec["channel_end"] for spec in out.concat_fusion_specs] == [4, 8]


def test_concat_fusion_uses_unified_groups_without_dense_branch_pack(monkeypatch) -> None:
    class TinyConcatNet(on.Module):
        def __init__(self):
            super().__init__()
            self.left = on.Conv2d(1, 4, kernel_size=1, bias=True)
            self.right = on.Conv2d(1, 4, kernel_size=1, bias=True)
            self.cat = on.Concat(dim=1)
            self.out = on.Conv2d(8, 2, kernel_size=1, bias=True)

        def forward(self, x):
            return self.out(self.cat(self.left(x), self.right(x)))

    active_scheme = orion.init_scheme(_python_backend_config())
    Module.set_scheme(active_scheme)
    Module.set_margin(active_scheme.params.get_margin())
    try:
        torch.manual_seed(0)
        net = TinyConcatNet()
        net.eval()
        x = torch.randn(1, 1, 8, 8)
        clear = net(x)

        original_pack_conv2d = packing.pack_conv2d

        def guarded_pack_conv2d(layer, last):
            if "_concat_source_" in str(getattr(layer, "name", "")):
                raise AssertionError("concat fusion should use unified grouped transforms, not dense branch pack")
            return original_pack_conv2d(layer, last)

        monkeypatch.setattr(packing, "pack_conv2d", guarded_pack_conv2d)
        orion.fit(net, x)
        input_level = orion.compile(net)

        assert getattr(net.out, "_concat_unified_groups_by_input", [])
        assert not getattr(net.out, "_concat_transform_ids_by_input", [])

        encrypted = orion.encrypt(orion.encode(x, input_level))
        net.he()
        fhe = net(encrypted).decrypt().decode()

        assert fhe.shape == clear.shape
        assert torch.allclose(fhe, clear, atol=1.0e-4, rtol=1.0e-4)
    finally:
        scheme.delete_scheme()


def test_concat_fusion_uses_actual_join_part_layouts_for_compact_and_halo() -> None:
    conv = on.Conv2d(4, 2, kernel_size=3, padding=1, bias=False)
    conv.name = "out"
    left_spec = {
        "concat_node": "cat",
        "source": "left",
        "shape": (1, 2, 8, 8),
        "fhe_shape": (1, 2, 8, 8),
        "gap": 1,
        "channels": 2,
        "channel_start": 0,
        "channel_end": 2,
    }
    right_spec = {
        "concat_node": "cat",
        "source": "right",
        "shape": (1, 2, 8, 8),
        "fhe_shape": (1, 2, 10, 8),
        "gap": 1,
        "channels": 2,
        "channel_start": 2,
        "channel_end": 4,
    }
    conv.concat_fusion_specs = (left_spec, right_spec)
    halo_layout = {"top_beta": 1, "bottom_beta": 1, "gap": 1, "core_slots": 128, "stored_slots": 160}
    compact_layout = {"top_beta": 0, "bottom_beta": 0, "gap": 1, "core_slots": 128, "stored_slots": 128}
    conv.region_runtime = SimpleNamespace(
        executor=SimpleNamespace(
            compile_plan={
                "edge_layouts": [
                    {
                        "edge": "left->cat",
                        "source": "left",
                        "target": "cat",
                        "op_kind": "concat",
                        "relayout": False,
                        "physical_layout": "logical_halo_compact",
                        "source_physical_layout": "packed_compact",
                        "source_layout": dict(compact_layout),
                        "selected_layout": dict(halo_layout),
                    },
                    {
                        "edge": "right->cat",
                        "source": "right",
                        "target": "cat",
                        "op_kind": "concat",
                        "relayout": False,
                        "physical_layout": "logical_halo_compact",
                        "source_physical_layout": "logical_halo_compact",
                        "source_layout": dict(halo_layout),
                        "selected_layout": dict(halo_layout),
                    },
                    {
                        "edge": "cat->out",
                        "source": "cat",
                        "target": "out",
                        "op_kind": "conv2d",
                        "selected_layout": dict(halo_layout),
                    },
                ],
            },
        ),
    )

    left_layout = conv._concat_source_input_layout(left_spec)
    right_layout = conv._concat_source_input_layout(right_spec)

    assert left_layout["top_beta"] == 0
    assert left_layout["bottom_beta"] == 0
    assert left_layout["stored_slots"] == left_layout["core_slots"]
    assert right_layout["top_beta"] == 1
    assert right_layout["bottom_beta"] == 1


def test_concat_materialization_fallback_matches_channel_cat_on_python_backend() -> None:
    active_scheme = orion.init_scheme(_python_backend_config())
    Module.set_scheme(active_scheme)
    Module.set_margin(active_scheme.params.get_margin())
    try:
        slots = int(active_scheme.params.get_slots())
        height = 16
        width = 16
        values_per_channel = int(height * width)
        left_channels = max(1, (int(slots) + int(values_per_channel) - 1) // int(values_per_channel))
        right_channels = 1
        concat = Concat(dim=1)
        concat.name = "concat_materialize_toy"
        concat.set_level(len(active_scheme.params.get_logq()) - 1)
        concat.configure_from_stats(
            input_shapes=[
                torch.Size((1, left_channels, height, width)),
                torch.Size((1, right_channels, height, width)),
            ],
            input_fhe_shapes=[
                torch.Size((1, left_channels, height, width)),
                torch.Size((1, right_channels, height, width)),
            ],
            input_gaps=[1, 1],
            output_shape=torch.Size((1, left_channels + right_channels, height, width)),
            fhe_output_shape=torch.Size((1, left_channels + right_channels, height, width)),
            output_gap=1,
        )
        concat.he_mode = True
        level = len(active_scheme.params.get_logq()) - 1
        left = torch.arange(
            left_channels * height * width,
            dtype=torch.float32,
        ).reshape(1, left_channels, height, width)
        right = (
            torch.arange(right_channels * height * width, dtype=torch.float32)
            .reshape(1, right_channels, height, width)
            + 10.0
        )
        left_ct = active_scheme.encrypt(active_scheme.encode(left, level))
        right_ct = active_scheme.encrypt(active_scheme.encode(right, level))

        out = concat(left_ct, right_ct).materialize().decrypt().decode().to(dtype=torch.float32)

        assert torch.allclose(
            out.reshape(1, left_channels + right_channels, height, width),
            torch.cat([left, right], dim=1),
            atol=1.0e-5,
        )
    finally:
        active_scheme.delete_scheme()


def test_python_encoder_preserves_explicit_level_zero() -> None:
    scheme = orion.init_scheme(_python_backend_config())
    try:
        encoded = orion.encode(torch.tensor([1.0]), level=0)
        assert encoded.level() == 0
    finally:
        scheme.delete_scheme()


def test_conv_transpose2d_dense_path_supports_multi_block_compile_and_eval() -> None:
    config = _python_backend_config(embedding_method="hybrid")
    config["ckks_params"]["LogN"] = 10

    active_scheme = orion.init_scheme(config)
    Module.set_scheme(active_scheme)
    Module.set_margin(active_scheme.params.get_margin())
    try:
        torch.manual_seed(7)
        layer = ConvTranspose2d(
            in_channels=16,
            out_channels=12,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
        )
        layer.eval()
        layer.init_orion_params()
        layer.name = "dense_tconv_multi_block"

        x = torch.randn(1, 16, 6, 6, dtype=torch.float32)
        y = layer(x)

        layer.input_shape = x.shape
        layer.output_shape = y.shape
        layer.input_gap = 2
        layer.output_gap = max(1, int(layer.input_gap) // int(layer.stride[0]))
        layer.fhe_input_shape = torch.Size((1, 4, 12, 12))
        layer.fhe_output_shape = torch.Size((1, 12, 12, 12))
        layer.set_level(len(active_scheme.params.get_logq()) - 1)

        layer.generate_diagonals(last=False)
        layer.compile()
        layer.he_mode = True

        assert len(layer.transform_ids) == 8
        assert {int(row) for row, _col in layer.transform_ids.keys()} == {0, 1, 2, 3}
        assert {int(col) for _row, col in layer.transform_ids.keys()} == {0, 1}

        source = _encode_layer_input(layer, x)
        assert len(source.ids) == 2

        out = layer(source)
        assert len(out.ids) == 4
        decoded = _decode_layer_output(layer, out)
        reference = F.conv_transpose2d(
            x,
            layer.on_weight.detach().to(dtype=torch.float32),
            layer.on_bias.detach().to(dtype=torch.float32),
            stride=tuple(int(v) for v in layer.stride),
            padding=tuple(int(v) for v in layer.padding),
            output_padding=tuple(int(v) for v in layer.output_padding),
            groups=int(layer.groups),
            dilation=tuple(int(v) for v in layer.dilation),
        )

        assert decoded.shape == reference.shape
        assert torch.allclose(decoded, reference, atol=1.0e-5, rtol=1.0e-5)
    finally:
        active_scheme.delete_scheme()
