from __future__ import annotations

import copy
import itertools

import pytest

from orion.experimental.wpc_cips_trained_benchmark import WORKER_GATES
from orion.experimental.wpc_online_encode_benchmark import (
    EXTRA_GATES, MODES, balanced_orders, compare_block, process_block_summary,
)


def worker(mode, pid):
    online, compressed = mode == "online_encode", mode == "compressed"
    wall = 2.0 if online else 1.2 if compressed else 1.0
    qp = 0 if online else 100 if compressed else 1960
    recipe = 10 if online else 0
    meta = 0 if mode == "full" else 20
    resident = qp + recipe + meta + 20
    counters = {"rotation_total": 4, "linear_transform_rotation": 4, "direct_rotation": 0, "conjugation": 0}
    row = {"resident_weight_qp_payload_bytes": qp, "unencoded_recipe_payload_bytes": recipe,
           "weight_metadata_bytes": meta, "uncompressed_bias_q_payload_bytes": 20,
           "full_weight_qp_payload_bytes": 1960, "stored_weight_plus_metadata_plus_bias_bytes": resident}
    comp = {"registered_transform_count": 56 if compressed else 0, "current_materialized_full_payload_bytes": 0,
            "current_materialized_transform_count": 0, "peak_materialized_transform_count": 1 if compressed else 0,
            "peak_materialized_full_payload_bytes": 100 if compressed else 0}
    return {
        "schema_version": 2, "status": "ok", "mode": mode, "seed": 17,
        "feature_sha256": "b"*64, "checkpoint": {"sha256": "a"*64}, "process": {"pid": pid},
        "shared_library": {"sha256": "d"*64},
        "experiment": {"forward_runs": 2, "atol": .002, "timed_lifecycle_tracing": False,
                       "security_scope": "small_insecure_functional_test_not_secure_deployment"},
        "measurements": {"forward_wall_s": [wall]*2, "decompression_s": [.1 if compressed else 0]*2,
                         "transform_evaluate_s": [.3]*2, "activation_s": [.05]*2, "bootstrap_s": [.4]*2,
                         "online_encode_s": [.5 if online else 0]*2, "online_prepare_s": [.1 if online else 0]*2,
                         "online_encode_pct_of_forward": [25 if online else 0]*2,
                         "online_materialization_pct_of_forward": [30 if online else 0]*2,
                         "transform_encode_invocations": [56 if online else 0]*2,
                         "measured_output_max_abs_errors": [1e-6]*2, "bootstrap_call_count": 2,
                         "timed_lifecycle_record_count": [0,0],
                         "online_python_encode_call_count": 0, "operation_counters_per_forward": counters,
                         "operation_counters_total": {k: v*2 for k,v in counters.items()}},
        "correctness": {"correct": True, "max_abs_error": 1e-6, "preflight_max_abs_error": 1e-6,
                        "untimed_lifecycle_trace": [[{"current_materialized_bytes_before": 0,
                            "current_materialized_bytes_after": 0, "current_materialized_transforms_after": 0}
                            for _ in range(56)]] if compressed else [[]],
                        "output_values": [1.,2.], "output_shape": [2]},
        "storage": {"learned_transform_count": 56, "logical_resident_total_bytes": resident+20,
                    "logical_full_reference_total_bytes": 2000, "concat_full_qp_payload_bytes": 20,
                    "concat": {"transform_count": 8}, "by_layer": {"layer": row}},
        "memory": {"pre_measured_gc": {"current_rss_bytes": 300},
                   "post_compile_gc": {"go": {"heap_inuse_bytes": 200}}},
        "backend": {"compressed_global_stats": comp,
                    "weight_plaintext_offline_encode_calls": 0 if online else 56,
                    "weight_plaintext_online_encode_calls": 112 if online else 0,
                    "expected_max_single_transform_bytes": 100,
                    "compile_transform_encode_invocations": {"ordinary": 64 if mode=="full" else 8,
                                                             "compressed": 56 if compressed else 0, "online_recipe": 0},
                    "measured_transform_encode_invocations": {"ordinary": 0, "compressed": 0, "online_recipe": 112 if online else 0},
                    "online_recipe_global_stats": {"registered_transform_count": 56 if online else 0,
                        "current_materialized_bytes": 0, "current_materialized_transforms": 0,
                        "peak_materialized_bytes": 100 if online else 0, "peak_materialized_transforms": 1 if online else 0}},
        "acceptance": dict.fromkeys(WORKER_GATES | EXTRA_GATES, True),
    }


def inputs():
    workers = {mode: worker(mode, 100+i) for i,mode in enumerate(MODES)}
    rss = {mode: {"sample_count_by_phase": {"measured": 3}, "peak_rss_by_phase": {"measured": 400}} for mode in MODES}
    return workers, rss


def test_three_way_raw_observations_and_process_block_ci():
    workers, rss = inputs()
    blocks = [compare_block(workers, rss, order=list(order), atol=.002) for order in itertools.permutations(MODES)]
    original = copy.deepcopy(blocks)
    result = process_block_summary(blocks, resamples=100, seed=3)
    assert result["independent_block_count"] == 6
    assert result["paired_ratios"]["compressed_over_online_encode_forward_ratio"]["ci95_mean"] == pytest.approx([.6,.6])
    assert result["by_mode"]["online_encode"]["online_encode_median_pct"]["mean"] == 25
    assert blocks == original


def test_balanced_orders_are_reproducible_and_balanced_in_every_cycle():
    orders = balanced_orders(12, seed=5)
    assert orders == balanced_orders(12, seed=5)
    for start in (0,6):
        assert {tuple(o) for o in orders[start:start+6]} == set(itertools.permutations(MODES))
    with pytest.raises(ValueError): balanced_orders(5)
    assert balanced_orders(1, smoke=True) == [list(MODES)]


@pytest.mark.parametrize("problem", ["count", "percent", "counter", "offline", "trace", "nan", "missing",
    "components", "leak", "peak", "registry", "resident", "features", "checkpoint", "operations", "rss", "schema", "shape", "error"])
def test_three_way_rejects_bad_raw_data(problem):
    workers, rss = inputs()
    w = workers["online_encode"]
    if problem == "count": w["measurements"]["transform_encode_invocations"][0] = 0
    elif problem == "percent": w["measurements"]["online_encode_pct_of_forward"][0] = 99
    elif problem == "counter": w["backend"]["measured_transform_encode_invocations"]["online_recipe"] = 0
    elif problem == "offline": w["backend"]["compile_transform_encode_invocations"]["ordinary"] = 64
    elif problem == "trace": w["experiment"]["timed_lifecycle_tracing"] = True
    elif problem == "nan": w["measurements"]["online_encode_s"][0] = float("nan")
    elif problem == "missing": w["acceptance"].pop("timed_lifecycle_tracing_disabled")
    elif problem == "components": w["measurements"]["online_prepare_s"][0] = 2
    elif problem == "leak": w["backend"]["online_recipe_global_stats"]["current_materialized_bytes"] = 1
    elif problem == "peak": w["backend"]["online_recipe_global_stats"]["peak_materialized_transforms"] = 2
    elif problem == "registry": w["backend"]["online_recipe_global_stats"]["registered_transform_count"] = 55
    elif problem == "resident": w["storage"]["logical_resident_total_bytes"] += 1
    elif problem == "features": w["feature_sha256"] = "c"*64
    elif problem == "checkpoint": w["checkpoint"]["sha256"] = "c"*64
    elif problem == "operations":
        w["measurements"]["operation_counters_per_forward"]["conjugation"] = 1
        w["measurements"]["operation_counters_total"]["conjugation"] = 2
    elif problem == "rss": rss["online_encode"]["sample_count_by_phase"]["measured"] = 0
    elif problem == "schema": w["schema_version"] = 1
    elif problem == "shape": w["correctness"]["output_shape"] = [1,2]
    elif problem == "error": w["measurements"]["measured_output_max_abs_errors"][0] = .1
    with pytest.raises(ValueError): compare_block(workers, rss, order=list(MODES), atol=.002)


def test_unbalanced_blocks_do_not_get_performance_confidence_intervals():
    workers, rss = inputs()
    block = compare_block(workers, rss, order=list(MODES), atol=.002)
    with pytest.raises(ValueError): process_block_summary([block], resamples=100)
    smoke = process_block_summary([block], smoke=True, resamples=100)
    assert smoke["performance_claims_enabled"] is False
    assert smoke["paired_ratios"]["compressed_over_full_forward_ratio"]["ci95_mean"] is None


def test_smoke_report_does_not_claim_balanced_orders_or_bootstrap_uncertainty():
    from tools.run_wpc_online_encode_benchmark import render_report

    workers, rss = inputs()
    block = compare_block(workers, rss, order=list(MODES), atol=.002)
    summary = process_block_summary([block], smoke=True, resamples=100)
    report = render_report({"status": "ok", "blocks": [block], "summary": summary, "smoke": True})
    assert "fixed treatment order for correctness only" in report
    assert "balanced across" not in report
    assert "percentile bootstrap" not in report
    assert "no uncertainty estimation or performance claim" in report
