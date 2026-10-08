"""Synthetic raw-evidence tests; no server-scale profiling in this suite."""
from __future__ import annotations

from collections import Counter
import copy

import pytest

from orion.experimental.wpc_cips_benchmark import summarize_samples
from orion.experimental.wpc_layout_gate import TREATMENTS
from orion.experimental.wpc_matched_layout_benchmark import balanced_orders, compare_block, summarize_blocks
from orion.experimental.wpc_orion_layout_control import NATIVE_PREPARATION_POLICY
from tests.test_wpc_layout_gate import inputs as gate_inputs


def inputs(count=3):
    workers, rss = gate_inputs()
    for treatment, worker in workers.items():
        m, exp, b = worker["measurements"], worker["experiment"], worker["backend"]
        exp.update(forward_runs=count, warmup_runs=2, retain_sample_outputs=True,
                   native_preparation_policy=NATIVE_PREPARATION_POLICY if treatment.startswith("native_orion/") else None,
                   runtime_environment={"ORION_DIRECT_PACK_WORKERS": "1"})
        worker["process"].update(python="test", platform="test", torch="test", numpy="test")
        for key, value in list(m.items()):
            if isinstance(value, list):
                m[key] = value * count
        m["operation_counters_total"] = {k: v * count for k, v in m["operation_counters_per_forward"].items()}
        m["bootstrap_call_count"] = count
        m["measured_operation_counters"] = [m["operation_counters_per_forward"].copy() for _ in range(count)]
        b["measured_transform_encode_invocations"] = {k: v * count for k, v in b["measured_transform_encode_invocations"].items()}
        b["weight_plaintext_online_encode_calls"] *= count
        worker["correctness"]["measured_output_values"] = [worker["correctness"]["output_values"].copy() for _ in range(count)]
        for key in ("forward_wall_s", "decompression_s", "transform_evaluate_s", "activation_s", "bootstrap_s"):
            m[key.replace("_s", "_summary_s")] = summarize_samples(m[key])
    return workers, rss


def block(order=None):
    workers, rss = inputs()
    return compare_block(workers, rss, order=order or list(TREATMENTS), atol=.002,
                         checkpoint_sha256="a"*64, config=workers["cips/full"]["experiment"]["ckks_config"])


def test_williams_schedule_balances_positions_and_ordered_neighbors():
    orders = balanced_orders(20, seed=7)
    assert orders == balanced_orders(20, seed=7)
    for cycle in (orders[:10], orders[10:]):
        for position in range(5):
            assert Counter(row[position] for row in cycle) == Counter(dict.fromkeys(TREATMENTS, 2))
        pairs = Counter((a, b) for row in cycle for a, b in zip(row, row[1:]))
        assert len(pairs) == 20 and set(pairs.values()) == {2}
    with pytest.raises(ValueError):
        balanced_orders(5)
    assert balanced_orders(1, smoke=True) == [list(TREATMENTS)]


def test_raw_block_recomputes_closure_and_all_sample_outputs():
    result = block()
    assert result["valid"] is True
    row = result["rows"]["native_orion/online_encode"]
    assert row["online_encode_s_pct"] == 25
    assert row["forward_mean_s"] == 2
    assert row["maximum_clear_error"] == 1e-6


@pytest.mark.parametrize("problem", ["outputs_missing", "first_output_wrong", "numeric_string", "first_error_wrong", "ops", "policy", "summary", "count", "timer", "software"])
def test_corrupt_raw_forward_samples_fail_closed(problem):
    workers, rss = inputs()
    w = workers["native_orion/online_encode"]
    if problem == "outputs_missing": w["correctness"].pop("measured_output_values")
    elif problem == "first_output_wrong": w["correctness"]["measured_output_values"][0][0] = .5
    elif problem == "numeric_string": w["correctness"]["measured_output_values"][0][0] = "0.000001"
    elif problem == "first_error_wrong": w["measurements"]["measured_output_max_abs_errors"][0] = 0.
    elif problem == "ops": w["measurements"]["measured_operation_counters"][0]["conjugation"] += 1
    elif problem == "policy": w["experiment"]["native_preparation_policy"] = "unpruned"
    elif problem == "summary": w["measurements"]["forward_wall_summary_s"]["median"] = 0.
    elif problem == "count": w["backend"]["measured_transform_encode_invocations"]["ordinary"] = 14
    elif problem == "timer": w["measurements"]["transform_evaluate_s"][0] = 5.
    else: w["process"]["torch"] = "different"
    with pytest.raises(ValueError):
        compare_block(workers, rss, order=list(TREATMENTS), atol=.002, checkpoint_sha256="a"*64,
                      config=workers["cips/full"]["experiment"]["ckks_config"])


def test_paired_block_uncertainty_and_smoke_scope():
    orders = balanced_orders(10, seed=0)
    blocks = [block(order) for order in orders]
    before = copy.deepcopy(blocks)
    summary = summarize_blocks(blocks, orders=orders, resamples=100)
    assert summary["block_count"] == 10
    assert summary["paired_latency_ratios"]["cips/compressed_over_native_orion/online_encode"]["ci95_mean"] == pytest.approx([.6, .6])
    assert blocks == before
    smoke = summarize_blocks([block()], orders=[list(TREATMENTS)], smoke=True, resamples=100)
    assert all(r["ci95_mean"] is None for r in smoke["paired_latency_ratios"].values())
    with pytest.raises(ValueError):
        summarize_blocks(blocks[:-1], orders=orders[:-1], resamples=100)
    blocks[0]["identity"]["checkpoint_sha256"] = "b"*64
    with pytest.raises(ValueError, match="changed"):
        summarize_blocks(blocks, orders=orders, resamples=100)


def test_plan_only_no_checkpoint_load_and_rejects_unbalanced_design(capsys):
    from tools.run_wpc_matched_layout_benchmark import parser, run
    args = parser().parse_args(["--plan-only", "--checkpoint", "/missing/checkpoint"])
    assert run(args) == 0
    assert "channel_pruned_block_regeneration_v1" in capsys.readouterr().out
    args.trial_blocks = 5
    with pytest.raises(ValueError, match="Williams"):
        run(args)


def write_review_fixture(root):
    from tools.run_wpc_matched_layout_benchmark import save, manifest_row, source_hashes, render_report
    workers, rss = inputs()
    exp = workers["cips/full"]["experiment"]
    orders = balanced_orders(1, smoke=True)
    request = {"orders": orders, "order_seed": 0, "smoke": True, "security_assessed": False, "bootstrap_resamples": 100, "bootstrap_seed": 0,
        "native_preparation_policy": NATIVE_PREPARATION_POLICY,
        "ckks_config": exp["ckks_config"], "configuration_source": exp["configuration_source"], "geometry": exp["geometry"],
        "resource_guards": {"max_worker_rss_mib": 8192., "worker_timeout_s": 1800.},
        "worker_request": {"seed": workers["cips/full"]["seed"], **{k: exp[k] for k in (
            "bsgs_ratio", "feature_std", "bound_headroom", "atol", "forward_runs", "warmup_runs", "verify_exact_qp", "retain_sample_outputs")}}}
    artifacts = []
    for treatment in TREATMENTS:
        layout, mode = treatment.split("/")
        directory = root / "block_001" / layout
        directory.mkdir(parents=True, exist_ok=True)
        for suffix, payload in (("worker.json", workers[treatment]), ("rss.json", rss[treatment]), ("worker.log", {}), ("phase.json", {})):
            path = directory / f"{mode}.{suffix}"
            save(path, payload)
            artifacts.append(manifest_row(path, root))
    b = compare_block(workers, rss, order=orders[0], atol=.002, checkpoint_sha256="a"*64, config=exp["ckks_config"])
    save(root / "block_001/block.json", b)
    artifacts.append(manifest_row(root / "block_001/block.json", root))
    save(root / "configuration_input.json", exp["ckks_config"])
    save(root / "requested_run.json", request)
    summary = summarize_blocks([b], orders=orders, smoke=True, resamples=100)
    data = {"schema_version": 1, "profile": "wpc_orion_matched_layout_repeated_benchmark", "status": "ok", "security_assessed": False,
        "request": request, "summary": summary, "provenance": {"checkpoint": {"sha256": "a"*64},
            "implementation_sha256": source_hashes(), "artifacts": artifacts,
            "configuration_artifact": manifest_row(root / "configuration_input.json", root)}}
    save(root / "comparison.json", data)
    (root / "comparison.md").write_text(render_report(data))
    return data


def test_frozen_review_request_cannot_promote_smoke_or_drop_guards(tmp_path):
    from tools.run_wpc_matched_layout_benchmark import validate_frozen_request
    request = write_review_fixture(tmp_path)["request"]
    assert validate_frozen_request(request) == request["orders"]
    request["worker_request"]["retain_sample_outputs"] = False
    with pytest.raises(ValueError): validate_frozen_request(request)
    request["worker_request"]["retain_sample_outputs"] = True
    request["resource_guards"]["max_worker_rss_mib"] = 0
    with pytest.raises(ValueError): validate_frozen_request(request)


@pytest.mark.parametrize("corruption", ["raw", "summary", "manifest", "report", "request"])
def test_offline_review_recomputes_and_detects_corruption(tmp_path, corruption):
    from tools.run_wpc_matched_layout_benchmark import review, save
    data = write_review_fixture(tmp_path)
    assert review(tmp_path) == 0
    if corruption == "raw": save(tmp_path / "block_001/cips/full.worker.json", {})
    elif corruption == "summary":
        data["summary"]["metrics"]["cips/full"]["forward_median_s"]["mean"] = 0.
        save(tmp_path / "comparison.json", data)
    elif corruption == "manifest":
        data["provenance"]["artifacts"].pop()
        save(tmp_path / "comparison.json", data)
    elif corruption == "report": (tmp_path / "comparison.md").write_text("incorrect")
    else: save(tmp_path / "requested_run.json", {})
    with pytest.raises(ValueError): review(tmp_path)


def test_small_real_fhe_controller_smoke_and_offline_review(tmp_path):
    """Five actual fresh workers with synthetic weights/LogN=9, not a profile."""
    import json
    import os
    import subprocess
    import sys
    import numpy as np
    import torch
    from tools.run_wpc_cips_trained_decoder import _config
    from tools.run_wpc_matched_layout_benchmark import REPO_ROOT, review
    from tools.run_wpc_cips_trained_isolated_benchmark import _read_process_memory
    if _read_process_memory(os.getpid())[0] is None:
        pytest.skip("RSS watchdog unavailable in this sandbox; Linux server preflight exercises the guarded controller")
    rng = np.random.default_rng(19)
    state = {name: torch.tensor(rng.normal(0, .001, shape), dtype=torch.float32) for name, shape in {
        "up1.weight": (64, 32, 2, 2), "up1.bias": (32,), "dec1a.weight": (32, 64, 3, 3),
        "dec1a.bias": (32,), "dec1b.weight": (32, 32, 3, 3), "dec1b.bias": (32,)}.items()}
    state["dec1a_act.coeffs"] = torch.tensor([.01, .2, .001, .0001, .00001, .000001, .0000001, .00000001])
    state["dec1a_act.prescale_tensor"] = state["dec1a_act.postscale_tensor"] = torch.tensor(1.)
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({"state_dict": state, "model": {"architecture": "unet22-plus-output", "base_dim": 32}}, checkpoint)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config(9)))
    destination = tmp_path / "run"
    command = [sys.executable, str(REPO_ROOT / "tools/run_wpc_matched_layout_benchmark.py"),
        "--smoke", "--trial-blocks", "1", "--height", "4", "--width", "4", "--ckks-config", str(config),
        "--checkpoint", str(checkpoint), "--out-dir", str(destination), "--warmup-runs", "1", "--forward-runs", "2",
        "--max-worker-rss-mib", "4096", "--worker-timeout-s", "120", "--bootstrap-resamples", "100"]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=300)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    assert review(destination) == 0
    result = json.loads((destination / "comparison.json").read_text())
    assert result["summary"]["block_count"] == 1
    assert all(r["ci95_mean"] is None for r in result["summary"]["paired_latency_ratios"].values())
    for treatment in TREATMENTS:
        layout, mode = treatment.split("/")
        worker = json.loads((destination / "block_001" / layout / f"{mode}.worker.json").read_text())
        assert len(worker["correctness"]["measured_output_values"]) == 2
        assert worker["measurements"]["measured_operation_counters"] == [worker["measurements"]["operation_counters_per_forward"]] * 2
        assert worker["backend"]["weight_plaintext_online_encode_calls"] == (28 if mode == "online_encode" else 0)
