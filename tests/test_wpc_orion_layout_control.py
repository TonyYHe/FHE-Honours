"""Small correctness tests; never execute the server-scale decoder benchmark."""
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import orion
from orion.core.packing import direct_diagonalize_conv2d, direct_diagonalize_conv_transpose2d
from orion.experimental.wpc_cips_checkpoint import CheckpointChebyshevSpec
from orion.experimental.wpc_orion_layout_control import (
    OrionLayoutConvPlan, OrionAlignedConcatPlan, OrionLayoutTrainedActivationBootstrap,
    native_signature, pack_native, unpack_native, encrypt_native,
    block_channel_pairs,
)
from orion.nn import Conv2d, ConvTranspose2d
from tools.run_wpc_cips_trained_decoder import _config, _rotation_padding_conv2d
from tools.run_wpc_cips_isolated_worker import _operation_counters


def _physical_layer(transpose=False, channels=8):
    rng = np.random.default_rng(11)
    size = 2 if transpose else 3
    layer = torch.nn.ConvTranspose2d(channels, channels, size, stride=2) if transpose else torch.nn.Conv2d(channels, channels, size, padding=1)
    layer.on_weight = torch.tensor(rng.normal(0, .02, (channels, channels, size, size)), dtype=torch.float32)
    layer.on_bias = torch.tensor(rng.normal(0, .01, channels), dtype=torch.float32)
    layer.input_shape = torch.Size((1, channels, 2 if transpose else 4, 2 if transpose else 4))
    layer.output_shape = torch.Size((1, channels, 4, 4))
    layer.input_gap, layer.output_gap = (2 if transpose else 1), 1
    layer.fhe_input_shape = torch.Size((1, channels // 4, 4, 4)) if transpose else layer.input_shape
    layer.fhe_output_shape = layer.output_shape
    return layer


def _evaluate_blocks(diagonals, messages, rows, slots):
    output = np.zeros((rows, slots), dtype=np.float64)
    for (row, col), block in diagonals.items():
        for offset, diagonal in block.items():
            output[row] += np.asarray(diagonal) * np.roll(messages[col], -int(offset))
    return output


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("channels", [8, 12])
def test_pruned_block_regeneration_is_exact_and_visits_only_selected_channel_pairs(transpose, workers, channels, monkeypatch):
    import orion.core.packing as packing
    monkeypatch.setenv("ORION_DIRECT_PACK_WORKERS", str(workers))
    layer = _physical_layer(transpose, channels)
    slots = (32 if not transpose else 16) if channels == 8 else (128 if not transpose else 64)
    input_sig = native_signature(layer.input_shape, slots, layer.input_gap)
    output_sig = native_signature(layer.output_shape, slots, layer.output_gap)
    def build(blocks=None, pairs=None):
        kwargs = dict(allow_hybrid=False, allowed_blocks=blocks, channel_pairs=pairs)
        return (direct_diagonalize_conv_transpose2d(layer, slots, "square", False, **kwargs) if transpose
                else direct_diagonalize_conv2d(layer, layer.on_weight, slots, "square", False,
                    padding_semantics="flattened_spatial_cyclic", **kwargs))[0]
    full = build()
    original = packing._packed_flat_indices
    visits = []
    def counted(channel, *args, **kwargs):
        visits.append(channel)
        return original(channel, *args, **kwargs)
    monkeypatch.setattr(packing, "_packed_flat_indices", counted)
    total_selected_pairs = 0
    for key, expected in full.items():
        pairs = block_channel_pairs(input_sig, output_sig, {key})
        visits.clear()
        result = build({key}, pairs)
        assert set(result) == {key}
        assert set(result[key]) == set(expected)
        for offset in expected:
            np.testing.assert_array_equal(result[key][offset], expected[offset])
        selected_pairs = sum(map(len, pairs.values()))
        total_selected_pairs += selected_pairs
        # Two index-vector constructions per channel pair / kernel offset.
        assert len(visits) == 2 * selected_pairs * int(np.prod(layer.kernel_size))
        assert selected_pairs < layer.in_channels * layer.out_channels
    assert total_selected_pairs == layer.in_channels * layer.out_channels
    with pytest.raises(ValueError, match="block"):
        block_channel_pairs(input_sig, output_sig, {(len(output_sig[5]), 0)})
    with pytest.raises(ValueError, match="channel"):
        build({(0, 0)}, {0: (layer.in_channels,)})


@pytest.mark.parametrize("gap", [1, 2])
def test_native_pack_round_trip_and_matches_pixel_shuffle(gap):
    source = np.arange(8 * 4 * 4, dtype=np.float64).reshape(1, 8, 4, 4)
    signature = native_signature(source.shape, 64, gap)
    messages = pack_native(source, signature)
    np.testing.assert_array_equal(unpack_native(messages, signature), source[0])
    expected = F.pixel_shuffle(torch.tensor(source), gap).numpy().reshape(-1)
    np.testing.assert_array_equal(messages.reshape(-1), expected)


@pytest.mark.parametrize("transpose", [False, True])
def test_native_diagonals_match_independent_torch_oracle(transpose):
    layer = _physical_layer(transpose)
    slots = 32
    input_sig = native_signature(layer.input_shape, slots, layer.input_gap)
    output_sig = native_signature(layer.output_shape, slots)
    source = np.zeros(layer.input_shape, dtype=np.float64)
    source[0, 0, 0, -1] = 1  # row edge; cyclic is not separate H/W toroidal wrap
    source[0, 3, -1, -1] = 2
    if transpose:
        diagonals, rotations = direct_diagonalize_conv_transpose2d(layer, slots, "square", False, allow_hybrid=False)
        reference = F.conv_transpose2d(torch.tensor(source), layer.on_weight.double(), stride=2).numpy()[0]
    else:
        diagonals, rotations = direct_diagonalize_conv2d(layer, layer.on_weight, slots, "square", False,
            allow_hybrid=False, padding_semantics="flattened_spatial_cyclic")
        reference = _rotation_padding_conv2d(torch.tensor(source), layer.on_weight.double(), torch.zeros(8, dtype=torch.float64)).numpy()[0]
        zero, _ = direct_diagonalize_conv2d(layer, layer.on_weight, slots, "square", False, allow_hybrid=False)
        zero_result = unpack_native(_evaluate_blocks(zero, pack_native(source, input_sig), len(output_sig[5]), slots), output_sig)
        np.testing.assert_allclose(zero_result, F.conv2d(torch.tensor(source), layer.on_weight.double(), padding=1).numpy()[0], rtol=0, atol=1e-8)
        assert not np.allclose(zero_result, reference)
    assert rotations == 0
    result = unpack_native(_evaluate_blocks(diagonals, pack_native(source, input_sig), len(output_sig[5]), slots), output_sig)
    np.testing.assert_allclose(result, reference, rtol=0, atol=1e-8)
    # One-block regeneration must not include other blocks or change values.
    key = sorted(diagonals)[0]
    if not transpose:
        partial, _ = direct_diagonalize_conv2d(layer, layer.on_weight, slots, "square", False,
            allow_hybrid=False, allowed_blocks={key}, padding_semantics="flattened_spatial_cyclic")
        assert set(partial) == {key}
        for offset in diagonals[key]:
            np.testing.assert_array_equal(partial[key][offset], diagonals[key][offset])


def test_rejects_unsupported_padding_and_native_geometry():
    layer = _physical_layer()
    with pytest.raises(ValueError, match="padding semantics"):
        direct_diagonalize_conv2d(layer, layer.on_weight, 32, "square", False, padding_semantics="toroidal")
    layer.stride = (2, 2)
    with pytest.raises(ValueError, match="Rotation-Padding"):
        direct_diagonalize_conv2d(layer, layer.on_weight, 32, "square", False, padding_semantics="flattened_spatial_cyclic")
    with pytest.raises(ValueError):
        native_signature((1, 7, 4, 4), 64, 2)
    with pytest.raises(ValueError):
        native_signature((1, 8, 8, 8), 32)


@pytest.mark.parametrize("transpose", [False, True])
def test_real_fhe_native_full_online_correct_and_release(monkeypatch, transpose):
    monkeypatch.setenv("ORION_LATTIGO_CLEAR_BACKEND", "0")
    monkeypatch.setenv("ORION_LATTIGO_STREAMING_LT", "0")
    monkeypatch.setenv("ORION_DIRECT_PACK_WORKERS", "1")
    scheme = orion.init_scheme(_config(10))
    rng = np.random.default_rng(113)
    source = rng.normal(0, .02, (1, 12, 4 if transpose else 8, 4 if transpose else 8))
    results, operations, byte_rows = {}, {}, {}
    try:
        kind = ConvTranspose2d if transpose else Conv2d
        kind.set_scheme(scheme)
        for mode in ("full", "online_encode"):
            layer = kind(12, 12, 2 if transpose else 3, stride=2 if transpose else 1,
                         padding=0 if transpose else 1, level=2)
            layer.name = "test_native"
            with torch.no_grad():
                layer.weight.copy_(torch.tensor(rng.normal(0, .01, tuple(layer.weight.shape)), dtype=torch.float32)) if mode == "full" else layer.weight.copy_(weight)
                layer.bias.zero_()
            weight = layer.weight.detach().clone()
            layer.init_orion_params()
            scheme.backend.ResetWPCBenchmarkEncodeCounters()
            plan = OrionLayoutConvPlan(layer, source.shape, scheme, storage_mode=mode, transpose=transpose)
            expected = plan.case.transform_count
            assert list(scheme.backend.GetWPCBenchmarkEncodeCounters()) == [expected if mode == "full" else 0, 0, 0]
            plan.record_sequence = False
            encrypted = plan.encrypt_input(source)
            try:
                for _ in range(2):
                    scheme.backend.ResetOperationCounters()
                    scheme.backend.ResetWPCBenchmarkEncodeCounters()
                    output = plan(encrypted)
                    try:
                        results[mode] = plan.decrypt_unpack(output)
                        np.testing.assert_allclose(results[mode], plan.clear_reference(source), rtol=0, atol=1e-6)
                        operations[mode] = _operation_counters(scheme.backend)
                    finally:
                        output.release()
                    assert plan.current_materialized_count == 0
                    assert plan.last_evaluation["full_transform_evaluate_call_s"] > 0
                    assert list(scheme.backend.GetWPCBenchmarkEncodeCounters()) == [expected if mode == "online_encode" else 0, 0, 0]
                byte_rows[mode] = plan.storage_summary()["full_weight_qp_payload_bytes"]
            finally:
                encrypted.release()
                plan.cleanup()
        np.testing.assert_allclose(results["full"], results["online_encode"], rtol=0, atol=1e-6)
        assert operations["full"] == operations["online_encode"]
        assert byte_rows["full"] == byte_rows["online_encode"]
    finally:
        scheme.delete_scheme()


def test_real_fhe_native_concat_bridge_chain(monkeypatch):
    monkeypatch.setenv("ORION_LATTIGO_CLEAR_BACKEND", "0")
    monkeypatch.setenv("ORION_LATTIGO_STREAMING_LT", "0")
    monkeypatch.setenv("ORION_DIRECT_PACK_WORKERS", "1")
    scheme = orion.init_scheme(_config(10))
    plans, bridge, low_ct, skip_ct, output = [], None, None, None, None
    rng = np.random.default_rng(24)
    try:
        Conv2d.set_scheme(scheme)
        ConvTranspose2d.set_scheme(scheme)
        low = rng.normal(0, .01, (1, 16, 4, 4))
        skip = rng.normal(0, .01, (1, 8, 8, 8))
        layers = [ConvTranspose2d(16, 8, 2, stride=2, level=8),
                  Conv2d(16, 8, 3, padding=1, level=6), Conv2d(8, 8, 3, padding=1, level=8)]
        for index, layer in enumerate(layers):
            layer.name = f"native_chain_{index}"
            with torch.no_grad():
                layer.weight.copy_(torch.tensor(rng.normal(0, .01, tuple(layer.weight.shape)), dtype=torch.float32))
                layer.bias.zero_()
            layer.init_orion_params()
        up = OrionLayoutConvPlan(layers[0], low.shape, scheme, transpose=True)
        plans.append(up)
        a = OrionLayoutConvPlan(layers[1], (1, 16, 8, 8), scheme)
        plans.append(a)
        b = OrionLayoutConvPlan(layers[2], (1, 8, 8, 8), scheme)
        plans.append(b)
        skip_contract = SimpleNamespace(output_shape=skip.shape, output_packing_signature=up.output_packing_signature)
        concat = OrionAlignedConcatPlan(scheme, up, skip_contract, a)
        spec = CheckpointChebyshevSpec("test", (.01, .2, .001, .0001, .00001, .000001, .0000001, .00000001), 1., 1., 1.)
        bridge = OrionLayoutTrainedActivationBootstrap(logical_shape=skip.shape, packing_signature=a.output_packing_signature,
            input_level=5, output_level=8, activation_spec=spec, bootstrap_bound=1.)
        bridge.compile(scheme)
        low_ct = up.encrypt_input(low)
        skip_ct = encrypt_native(scheme, skip, up.output_packing_signature, 7)
        skip_before = [scheme.backend.GetCiphertextLevel(i) for i in skip_ct.ids]
        from tools.run_wpc_cips_trained_isolated_worker import _run_decoder
        output = _run_decoder(up, concat, a, bridge, b, low_ct, skip_ct)
        clear_up = up.clear_reference(low)
        clear_a = a.clear_reference(np.concatenate((clear_up, skip[0]))[None])
        clear_b = b.clear_reference(spec.evaluate(torch.tensor(clear_a[None])).numpy())
        np.testing.assert_allclose(b.decrypt_unpack(output), clear_b, rtol=0, atol=2e-5)
        assert [scheme.backend.GetCiphertextLevel(i) for i in skip_ct.ids] == skip_before
        assert all(scheme.backend.GetCiphertextLevel(i) == 7 for i in output.ids)
        native_result = b.decrypt_unpack(output)
        output.release()
        output = None
        low_ct.release()
        skip_ct.release()
        low_ct = skip_ct = None
        bridge.cleanup()
        bridge = None
        for plan in plans:
            plan.cleanup()
        plans = []
        # End-to-end layout equivalence at a deliberately small geometry.
        from orion.experimental.wpc_cips_branches import WPCCIPSConcatPlan
        from orion.experimental.wpc_cips_checkpoint import WPCCIPSTrainedActivationBootstrap
        from orion.nn import Concat
        from tools.run_wpc_cips_trained_decoder import _encrypt_packed
        for mode in ("full", "online_encode", "compressed"):
            up = layers[0].install_wpc_cips_plan(low.shape, storage_mode=mode, verify_exact_qp=False)
            plans.append(up)
            a = layers[1].install_wpc_cips_plan((1, 16, 8, 8), storage_mode=mode, verify_exact_qp=False)
            plans.append(a)
            b = layers[2].install_wpc_cips_plan(skip.shape, storage_mode=mode, verify_exact_qp=False)
            plans.append(b)
            Concat.set_scheme(scheme)
            cat_module = Concat(dim=1)
            skip_contract = SimpleNamespace(output_shape=skip.shape, output_packing_signature=up.output_packing_signature, output_level=7)
            concat = cat_module.install_wpc_cips_plan((up, skip_contract), consumer_plan=a)
            assert isinstance(concat, WPCCIPSConcatPlan)
            bridge = WPCCIPSTrainedActivationBootstrap(logical_shape=skip.shape, packing_signature=a.output_packing_signature,
                input_level=5, output_level=8, activation_spec=spec, bootstrap_bound=1.)
            bridge.compile(scheme)
            low_ct = up.encrypt_input(low)
            skip_ct = _encrypt_packed(scheme, skip[0], up.output_packing_signature, level=7)
            try:
                output = _run_decoder(up.evaluate, concat.evaluate, a.evaluate, bridge, b.evaluate, low_ct, skip_ct)
                np.testing.assert_allclose(b.decrypt_unpack(output), native_result, rtol=0, atol=2e-5)
                if mode == "full":
                    assert all(plan.last_evaluation["full_transform_evaluate_call_s"] > 0 for plan in plans)
            finally:
                for value in (output, low_ct, skip_ct):
                    if value is not None:
                        value.release()
                output = low_ct = skip_ct = None
                bridge.cleanup()
                bridge = None
                cat_module.remove_wpc_cips_plan()
                for layer in layers:
                    layer.remove_wpc_cips_plan()
                plans = []
    finally:
        for value in (output, low_ct, skip_ct):
            if value is not None:
                value.release()
        if bridge is not None:
            bridge.cleanup()
        for plan in plans:
            plan.cleanup()
        scheme.delete_scheme()


@pytest.mark.parametrize("mode", ["full", "online_encode"])
def test_small_native_worker_records_real_counters_and_provenance(monkeypatch, tmp_path, mode):
    """Exercise the real worker interface, using synthetic weights and LogN=9."""
    from tools.run_wpc_cips_trained_isolated_worker import main
    rng = np.random.default_rng(19)
    state = {}
    for name, shape in {"up1.weight": (64, 32, 2, 2), "up1.bias": (32,),
                        "dec1a.weight": (32, 64, 3, 3), "dec1a.bias": (32,),
                        "dec1b.weight": (32, 32, 3, 3), "dec1b.bias": (32,)}.items():
        state[name] = torch.tensor(rng.normal(0, .001, shape), dtype=torch.float32)
    state["dec1a_act.coeffs"] = torch.tensor([.01, .2, .001, .0001, .00001, .000001, .0000001, .00000001])
    state["dec1a_act.prescale_tensor"] = torch.tensor(1.)
    state["dec1a_act.postscale_tensor"] = torch.tensor(1.)
    checkpoint = tmp_path / "synthetic_decoder.pt"
    torch.save({"state_dict": state, "model": {"architecture": "unet22-plus-output", "base_dim": 32}}, checkpoint)
    result = tmp_path / f"{mode}.json"
    # Restore the worker's explicit environment changes after this test.
    for name in ("ORION_LATTIGO_CLEAR_BACKEND", "ORION_LATTIGO_STREAMING_LT", "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT",
                 "ORION_WPC_PERIODICITY_PROFILE", "ORION_SINGLE_SLOT_LAYER_CACHE", "ORION_CPP_DIAG_BUILDER",
                 "ORION_DIRECT_PACK_WORKERS", "ORION_SINGLE_SLOT_ENCODE_WORKERS", "ORION_LATTIGO_COMPILE_WORKERS"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr("sys.argv", ["worker", "--layout", "native_orion", "--mode", mode,
        "--checkpoint", str(checkpoint), "--out", str(result), "--phase-file", str(tmp_path / "phase.json"),
        "--logn", "9", "--height", "4", "--width", "4", "--warmup-runs", "0", "--forward-runs", "1"])
    assert main() == 0
    worker = json.loads(result.read_text())
    assert worker["acceptance"] and all(v is True for v in worker["acceptance"].values())
    assert worker["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert worker["storage"]["learned_transform_count"] == 14
    assert worker["storage"]["concat"]["transform_count"] == 0
    assert worker["measurements"]["transform_evaluate_s"][0] > 0
    assert worker["backend"]["measured_transform_encode_invocations"] == {"ordinary": 14 if mode == "online_encode" else 0,
                                                                         "compressed": 0, "online_recipe": 0}
    assert worker["backend"]["native_materialized_transform_count_after_forward"] == 0
    from orion.experimental.wpc_layout_gate import _native
    _native(worker, mode, worker["experiment"]["atol"])
    output = np.asarray(worker["correctness"]["output_values"])
    clear = np.asarray(worker["correctness"]["independent_clear_output_values"])
    assert float(np.max(np.abs(output - clear))) <= worker["experiment"]["atol"]
