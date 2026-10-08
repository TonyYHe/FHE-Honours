from types import SimpleNamespace
import copy

import numpy as np
import pytest

import orion
from orion.experimental.wpc_selective_orion import (
    SelectiveOrionTransform, _buffers, selective_policy, snapshot_model, runtime_delta,
)
from tools.run_wpc_cips_isolated_worker import _config
from tools.run_wpc_selective_orion import run_gate
from tools.run_wpc_selective_orion import review_gate


def test_policy_is_opt_in_and_strict(monkeypatch):
    monkeypatch.delenv("ORION_WPC_SELECTIVE_POLICY", raising=False)
    assert selective_policy() is None
    monkeypatch.setenv("ORION_WPC_SELECTIVE_POLICY", "hybrid")
    assert selective_policy() == "hybrid"
    monkeypatch.setenv("ORION_WPC_SELECTIVE_POLICY", "true")
    with pytest.raises(ValueError):
        selective_policy()


@pytest.mark.parametrize("indices,data", [([0, -8], np.zeros(16)), ([8], np.zeros(8)),
    ([.1], np.zeros(8)), ([0], np.zeros(7)), ([0], np.full(8, np.nan))])
def test_rejects_invalid_payloads_before_ctypes(indices, data):
    with pytest.raises(ValueError):
        _buffers(indices, data, 8)


def test_stale_library_fails_closed():
    with pytest.raises(RuntimeError, match="rebuilt"):
        SelectiveOrionTransform(object(), [0], np.ones(8), slots=8, level=2, bsgs_ratio=2, policy="hybrid")


def test_step1_cannot_mislabel_selective_materialization():
    from tools.run_step1_online_encode_profile import _profile_environment
    with pytest.raises(ValueError, match="historical"):
        _profile_environment({"ORION_WPC_SELECTIVE_POLICY": "hybrid"}, encode_workers=1)


def test_real_fhe_gate_mixed_and_zero_eligible(monkeypatch):
    for name in ("ORION_LATTIGO_CLEAR_BACKEND", "ORION_LATTIGO_STREAMING_LT", "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("ORION_SINGLE_SLOT_ENCODE_WORKERS", "1")
    monkeypatch.setenv("ORION_DIRECT_PACK_WORKERS", "1")
    result = run_gate(runs=2)
    assert result["status"] == "ok"
    assert review_gate(result)["status"] == "ok"
    for case in result["cases"]:
        for policy in case["policies"].values():
            assert all(policy["acceptance"].values())
        if case["case"].startswith("native"):
            assert case["policies"]["hybrid"]["stats"]["eligible_count"] == 0
        if case["case"] == "mixed_control":
            assert case["policies"]["hybrid"]["stats"]["compressed_count"] == 2
            assert case["policies"]["hybrid"]["stats"]["online_embed_calls"] == 4
        if case["case"] == "all_periodic_control":
            assert case["policies"]["hybrid"]["stats"]["online_embed_calls"] == 0
            assert all(s["baseline_eligible_encode_time_coverage_pct"] == 100 for s in case["policies"]["online"]["samples"])
    import copy
    for field, value in (("max_abs_error_vs_clear", 999), ("backend_encode_s", -1),
                         ("exact_qp_match", False), ("encode_pct_of_lt_prepare_plus_evaluate", 101)):
        bad = copy.deepcopy(result)
        bad["cases"][0]["policies"]["online"]["samples"][0][field] = value
        with pytest.raises(ValueError):
            review_gate(bad)


@pytest.mark.parametrize("granularity", ["layer", "lt", "group"])
def test_ordinary_dense_cache_uses_selective_materialize_and_release(monkeypatch, granularity):
    monkeypatch.setenv("ORION_LATTIGO_CLEAR_BACKEND", "0")
    monkeypatch.setenv("ORION_LATTIGO_STREAMING_LT", "0")
    monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    monkeypatch.setenv("ORION_SINGLE_SLOT_ENCODE_WORKERS", "1")
    monkeypatch.setenv("ORION_WPC_SELECTIVE_POLICY", "hybrid")
    monkeypatch.setenv("ORION_DENSE_LAYER_CACHE_GRANULARITY", granularity)
    scheme = orion.init_scheme(_config(9))
    slots = scheme.params.get_slots()
    block = {0: np.full(slots, .125, dtype=np.float32), 1: np.arange(slots, dtype=np.float32)/slots}
    layer = SimpleNamespace(name="test_selective_cache", level=2, bsgs_ratio=2, output_rotations=0,
        diagonals={}, on_bias_ptxt=None, _dense_layer_cache_diag_indices_by_block={(0, 0): (0, 1)},
        _dense_layer_cache_build_diagonals=lambda: {(0, 0): block},
        _dense_layer_cache_build_block_diagonals=lambda keys: {(0, 0): block})
    model = SimpleNamespace(named_modules=lambda: [("layer", layer)])
    try:
        evaluator = scheme.lt_evaluator
        evaluator._defer_dense_layer_cache_compile(layer, {}, level=2, bsgs_ratio=2)
        plan = layer._wpc_selective_plans[(0, 0)]
        assert plan.stats()["offline_embed_calls"] == 1
        for _ in range(2):
            before = snapshot_model(model)
            if granularity == "layer":
                evaluator.materialize_dense_layer_cache(layer)
                assert evaluator._consume_dense_layer_cache_pending_timing(layer)["layer_cache_encode_s"] > 0
                evaluator.evict_dense_layer_cache(layer)
            else:
                ids, _ = evaluator._materialize_dense_layer_cache_blocks(layer, ((0, 0),))
                evaluator._evict_dense_layer_cache_transform_ids(layer, ids)
            after = snapshot_model(model)
            delta = runtime_delta(before, after, forward_s=10)
            assert delta["counter_delta"]["online_embed_calls"] == 1
            assert delta["materialized_bytes_after_forward"] == 0
            assert plan.stats()["materialized_bytes"] == 0
        mutated = np.concatenate([block[0], block[1]]); mutated[0] = .25
        with pytest.raises(RuntimeError, match="identity"):
            plan.materialize([0, 1], mutated)
        assert plan.stats()["materialized_bytes"] == 0
    finally:
        for plan in getattr(layer, "_wpc_selective_plans", {}).values():
            plan.close()
        scheme.delete_scheme()


@pytest.mark.parametrize("method", ["evict_dense_layer_cache", "_evict_dense_layer_cache_transform_ids"])
def test_selective_release_failure_is_not_silently_accepted(method):
    from orion.backend.python.lt_evaluator import NewEvaluator
    def fail():
        raise RuntimeError("release failed")
    layer = SimpleNamespace(name="failed_release", _dense_layer_cache_active_transform_ids={(0, 0): 42},
        transform_ids={(0, 0): 42}, _wpc_selective_plans={(0, 0): SimpleNamespace(id=42, release=fail)})
    evaluator = SimpleNamespace(backend=object(), dense_layer_cache_granularity=lambda: "lt")
    args = (layer,) if method == "evict_dense_layer_cache" else (layer, {(0, 0): 42})
    with pytest.raises(RuntimeError, match="release failed"):
        getattr(NewEvaluator, method)(evaluator, *args)
    assert layer.transform_ids == {(0, 0): 42}


def _snapshot(policy, *, eligible=1, count=0):
    from orion.experimental.wpc_selective_orion import FIELDS
    stats = dict.fromkeys(FIELDS, 0)
    stats.update(schema_version=1, diagonal_count=2, eligible_count=eligible, full_payload_bytes=65536,
        eligible_full_payload_bytes=32768*eligible, metadata_bytes=104, materialization_count=count)
    if policy == "hybrid":
        stats.update(compressed_count=eligible, offline_embed_calls=eligible, compressed_payload_bytes=64*eligible)
    stats.update(online_embed_calls=count*(2-stats["compressed_count"]),
        online_eligible_embed_calls=count*eligible if policy == "online" else 0,
        peak_materialized_bytes=65536 if count else 0,
        total_prepare_nanoseconds=10_000*count, total_encode_nanoseconds=20_000*count,
        total_eligible_encode_nanoseconds=10_000*count*eligible if policy == "online" else 0,
        total_decompress_nanoseconds=10_000*count*eligible if policy == "hybrid" else 0)
    row = {"module": "conv", "operator": "Conv2d", "block": [0, 0], "policy": policy,
        "slot_payload_sha256": "a"*64, "stats": stats}
    totals = {k:v for k,v in stats.items() if k not in ("schema_version", "peak_materialized_bytes") and not k.startswith("last_")}
    return {"schema_version": 1, "scope": "opted_in_dense_layer_cache_transforms_only",
        "transform_count": 1, "transforms": [row], "totals": totals}


def _workers(*, eligible=1):
    workers = {}
    for policy in ("online", "hybrid"):
        initial, previous = _snapshot(policy, eligible=eligible), _snapshot(policy, eligible=eligible)
        attempts = []
        for index, kind in enumerate(("warmup", "measured", "measured"), 1):
            current = _snapshot(policy, eligible=eligible, count=index)
            attempts.append({"kind": kind, "status": "ok", "timing_s": {"he_forward": .1},
                "decoded": {"shape": [1], "values": [.125]}, "selective_orion_snapshot": current,
                "selective_orion_profile": runtime_delta(previous, current, forward_s=.1),
                "selective_orion_operations": {"conjugation": 0, "direct_rotation": 0,
                    "linear_transform_rotation": 1, "rotation_total": 1}})
            previous = current
        workers[policy] = {"network": "test", "model": "test", "seed": 0, "config": {},
            "status": "ok", "mode": "dense", "selective_orion_policy": policy,
            "clear": {"shape": [1], "values": [.125]}, "forward_runs": 2, "warmup_runs": 1,
            "selective_orion_after_compile": initial, "forward_attempts": attempts}
    return workers


@pytest.mark.parametrize("eligible", [0, 1])
def test_model_review_recomputes_encode_time_coverage(eligible):
    from tools.run_wpc_selective_model import validate_pair
    result = validate_pair(_workers(eligible=eligible), atol=1e-6)
    assert result["baseline_eligible_encode_time_coverage_pct"] == 50*eligible
    assert result["samples"]["hybrid"][0]["baseline_eligible_encode_time_coverage_pct"] is None
    assert result["performance_claims_enabled"] is False


@pytest.mark.parametrize("mutation", ["identity", "totals", "timer", "output", "operations", "warmup", "repeat", "nonfinite"])
def test_model_review_rejects_corrupt_raw_evidence(mutation):
    from tools.run_wpc_selective_model import validate_pair
    workers = _workers()
    worker = workers["hybrid"]
    attempt = worker["forward_attempts"][1]
    if mutation == "identity":
        attempt["selective_orion_snapshot"]["transforms"][0]["slot_payload_sha256"] = "b"*64
    elif mutation == "totals":
        attempt["selective_orion_snapshot"]["totals"]["materialization_count"] += 1
    elif mutation == "timer":
        attempt["selective_orion_profile"]["backend_encode_s"] = 999
    elif mutation == "output":
        attempt["decoded"]["values"] = [1.]
    elif mutation == "operations":
        attempt["selective_orion_operations"]["rotation_total"] = 2
    elif mutation == "warmup":
        worker["forward_attempts"].pop(0)
    elif mutation == "repeat":
        worker["forward_attempts"].append(copy.deepcopy(worker["forward_attempts"][-1]))
    else:
        attempt["selective_orion_snapshot"]["transforms"][0]["stats"]["total_encode_nanoseconds"] = float("nan")
    with pytest.raises(ValueError):
        validate_pair(workers, atol=1e-6)


def test_tiny_orion_compile_forward_hooks_and_raw_review(monkeypatch, tmp_path):
    """Actual ordinary Orion compiler/forward hooks; tiny functional parameters."""
    import torch
    from tools.run_lattigo_e2e_compare import _run_forward_attempt, _tensor_payload
    from tools.run_wpc_selective_model import validate_pair
    class Tiny(orion.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = orion.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        def forward(self, x):
            return self.conv(x)
    monkeypatch.setenv("ORION_LATTIGO_CLEAR_BACKEND", "0")
    monkeypatch.setenv("ORION_LATTIGO_STREAMING_LT", "0")
    monkeypatch.setenv("ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT", "0")
    monkeypatch.setenv("ORION_SINGLE_SLOT_LAYER_CACHE", "1")
    monkeypatch.setenv("ORION_SINGLE_SLOT_ENCODE_WORKERS", "1")
    monkeypatch.setenv("ORION_DIRECT_PACK_WORKERS", "1")
    workers = {}
    for policy in ("online", "hybrid"):
        monkeypatch.setenv("ORION_WPC_SELECTIVE_POLICY", policy)
        torch.manual_seed(8)
        net = Tiny()
        net.eval()
        x = torch.randn(1, 1, 4, 4)*.02
        with torch.no_grad():
            clear = net(x)
        scheme = orion.init_scheme(_config(9))
        try:
            scheme.fit(net, x)
            level = scheme.compile(net)
            net.he()
            worker = {"status": "ok", "backend": "lattigo", "network": "tiny", "model": "Tiny",
                "mode": "dense", "seed": 8, "config": _config(9), "clear": _tensor_payload(clear),
                "selective_orion_policy": policy, "selective_orion_after_compile": snapshot_model(net),
                "warmup_runs": 0, "forward_runs": 2}
            for i in range(2):
                attempt = _run_forward_attempt(payload=worker, out_path=tmp_path/f"{policy}.json",
                    net=net, x0=x, clear=clear, input_level=level, mode="dense", attempt_index=i,
                    attempt_kind="measured", profile_modules=False, profile_lt=False,
                    trace_forward_memory=False, operator_breakdown=False, layer_mae=False,
                    layer_mae_clear_outputs=None, layer_mae_reference_transforms=None, record_primary=False)
                assert attempt["status"] == "ok"
                assert attempt["selective_orion_profile"]["materialized_bytes_after_forward"] == 0
            workers[policy] = worker
        finally:
            for plan in getattr(net.conv, "_wpc_selective_plans", {}).values():
                plan.close()
            scheme.delete_scheme()
    result = validate_pair(workers, atol=1e-6)
    assert result["status"] == "ok"
    assert result["registered_transform_count"] == 1
