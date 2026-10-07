"""Small real-FHE storage-policy correctness gates, not performance tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import orion
from orion.nn import Conv2d, ConvTranspose2d
from tools.run_wpc_cips_isolated_worker import _config, _operation_counters


@pytest.mark.parametrize("operation", ["conv", "transpose", "stride2"])
def test_real_fhe_three_storage_modes_and_repeated_online_release(monkeypatch, operation):
    transpose = operation == "transpose"
    stride = 2 if operation in ("transpose", "stride2") else 1
    monkeypatch.setenv("ORION_LATTIGO_CLEAR_BACKEND", "0")
    monkeypatch.setenv("ORION_LATTIGO_STREAMING_LT", "0")
    monkeypatch.setenv("ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT", "0")
    rng = np.random.default_rng(117)
    source = rng.normal(0, .1, (1,12,4 if transpose else 8,4 if transpose else 8))
    weight = rng.normal(0, .02, (12,12,2 if transpose else 3,2 if transpose else 3))
    bias = rng.normal(0, .01, 12)
    scheme = orion.init_scheme(_config(10))
    results, operations = {}, {}
    try:
        kind = ConvTranspose2d if transpose else Conv2d
        kind.set_scheme(scheme)
        for mode in ("full", "compressed", "online_encode"):
            layer = kind(12,12,2 if transpose else 3, stride=stride,
                         padding=0 if transpose else 1, bias=True, level=2)
            with torch.no_grad():
                layer.weight.copy_(torch.tensor(weight, dtype=torch.float32))
                layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))
            layer.init_orion_params()
            scheme.backend.ResetWPCBenchmarkEncodeCounters()
            plan = layer.install_wpc_cips_plan(source.shape, storage_mode=mode, verify_exact_qp=False)
            expected = plan.case.transform_count
            compile_counts = list(scheme.backend.GetWPCBenchmarkEncodeCounters())
            assert compile_counts == ([expected,0,0] if mode=="full" else [0,expected,0] if mode=="compressed" else [0,0,0])
            plan.record_sequence = False
            encrypted = plan.encrypt_input(source)
            layer.he()
            reference = plan.clear_reference(source)
            try:
                for _ in range(2):
                    scheme.backend.ResetWPCBenchmarkEncodeCounters()
                    scheme.backend.ResetOperationCounters()
                    output = layer(encrypted)
                    try:
                        decoded = plan.decrypt_unpack(output)
                        np.testing.assert_allclose(decoded, reference, rtol=0, atol=1e-6)
                        results[mode] = decoded
                        operations[mode] = _operation_counters(scheme.backend)
                    finally:
                        output.release()
                    assert plan.last_evaluation["evaluation_sequence"] == []
                    assert list(scheme.backend.GetWPCBenchmarkEncodeCounters()) == ([0,0,expected] if mode=="online_encode" else [0,0,0])
                    online_stats = list(scheme.backend.GetWPCOnlineGlobalStats())
                    assert online_stats[1] == online_stats[3] == 0
                    if mode == "online_encode":
                        assert online_stats[0] == expected
                        assert online_stats[4] == 1
                        assert plan.storage_summary()["resident_weight_qp_payload_bytes"] == 0
                        for tid in plan.online_transform_ids.values():
                            assert len(scheme.backend.GetLinearTransformEmptyPlaintextKeys(tid)) > 0
            finally:
                encrypted.release()
                layer.remove_wpc_cips_plan()
            assert list(scheme.backend.GetWPCOnlineGlobalStats())[0] == 0
        np.testing.assert_allclose(results["online_encode"], results["full"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(results["compressed"], results["full"], rtol=0, atol=1e-6)
        assert operations["online_encode"] == operations["full"] == operations["compressed"]
    finally:
        scheme.delete_scheme()
