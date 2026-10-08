from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from orion.experimental.wpc_layout_gate import TREATMENTS, validate_layout_gate
from tests.test_wpc_decoder_geometry import configurable_inputs


def inputs():
    cips, _rss = configurable_inputs()
    workers, samples = {}, {}
    for index, treatment in enumerate(TREATMENTS):
        layout, mode = treatment.split("/")
        worker = copy.deepcopy(cips[mode])
        worker.update(layout=layout, profile="wpc_native_orion_matched_decoder_worker" if layout == "native_orion" else "wpc_cips_trained_decoder_isolated_worker")
        exp, m, b, s = worker["experiment"], worker["measurements"], worker["backend"], worker["storage"]
        exp.update(forward_runs=1, warmup_runs=0, graph="up1+skip1->cat1->dec1a->Cheb7->bootstrap->dec1b",
                   feature_std=.02, bsgs_ratio=2., bound_headroom=1.25, configuration_source={"kind": "test"},
                   padding_semantics="flattened_spatial_cyclic")
        for key, values in list(m.items()):
            if isinstance(values, list):
                m[key] = values[:1]
        m["operation_counters_total"] = m["operation_counters_per_forward"].copy()
        m["bootstrap_call_count"] = 1
        b["measured_transform_encode_invocations"]["online_recipe"] = 14 if mode == "online_encode" else 0
        b["weight_plaintext_online_encode_calls"] = 14 if mode == "online_encode" else 0
        worker["bootstrap_range"] = {"activation_min": -.01, "activation_max": .01, "symmetric_bound": 1.}
        rng = np.random.default_rng(worker["seed"])
        low = rng.normal(0, exp["feature_std"], exp["low_shape"]).astype(np.float64)
        skip = rng.normal(0, exp["feature_std"], exp["high_shape"]).astype(np.float64)
        worker["feature_sha256"] = hashlib.sha256(low.tobytes() + skip.tobytes()).hexdigest()
        worker["process"]["pid"] = 100 + index
        size = np.prod(exp["high_shape"][1:])
        clear = np.zeros(size, dtype="<f8")
        worker["correctness"].update(output_shape=exp["high_shape"][1:], output_values=[1e-6] * size,
            independent_clear_output_values=clear.tolist(), independent_clear_output_sha256=hashlib.sha256(clear.tobytes()).hexdigest())
        worker["clear_oracle_max_abs_delta"] = 0.
        if layout == "native_orion":
            b["compile_transform_encode_invocations"] = {"ordinary": 14 if mode == "full" else 0, "compressed": 0, "online_recipe": 0}
            b["measured_transform_encode_invocations"] = {"ordinary": 14 if mode == "online_encode" else 0, "compressed": 0, "online_recipe": 0}
            b["online_recipe_global_stats"]["registered_transform_count"] = 0
            b["native_materialized_transform_count_after_forward"] = 0
            worker["correctness"]["untimed_lifecycle_trace"] = [[{"materialized_transforms_after": 0} for _ in range(14)]]
            # Layouts may have different rotations, storage policies may not.
            m["operation_counters_total"]["rotation_total"] = m["operation_counters_total"]["linear_transform_rotation"] = 2
            m["operation_counters_per_forward"] = m["operation_counters_total"].copy()
            s["storage_mode"] = mode
            s["concat"] = {"transform_count": 0}
            s["concat_full_qp_payload_bytes"] = 0
            b["native_transform_rows"] = {}
            total_full = total_stored = 0
            for name, row in s["by_layer"].items():
                count = row["transform_count"]
                geometry = exp["geometry"]
                level = geometry["levels"][{"checkpoint_up1": "up1", "checkpoint_dec1a": "dec1a", "checkpoint_dec1b": "dec1b"}[name]]
                q_bytes, p_bytes = 2 * 2 * geometry["slots"] * (level + 1) * 8, 2 * 2 * geometry["slots"] * 8
                per_transform = q_bytes + p_bytes
                qp = per_transform * count
                row["full_weight_qp_payload_bytes"] = qp
                b["native_transform_rows"][name] = {str(i): {"full_payload_bytes": per_transform, "diagonal_count": 2,
                    "payload_bytes_measured": mode == "full", "actual_qp_stats": {"schema_version": 1, "diagonal_count": 2,
                        "q_bytes": q_bytes, "p_bytes": p_bytes, "total_bytes": per_transform} if mode == "full" else None} for i in range(count)}
                row.update(resident_weight_qp_payload_bytes=qp if mode == "full" else 0,
                    unencoded_recipe_payload_bytes=10 if mode == "online_encode" else 0, weight_metadata_bytes=8)
                row["full_weight_plus_bias_payload_bytes"] = qp + row["uncompressed_bias_q_payload_bytes"]
                row["stored_weight_plus_metadata_plus_bias_bytes"] = sum(row[k] for k in (
                    "resident_weight_qp_payload_bytes", "unencoded_recipe_payload_bytes", "weight_metadata_bytes", "uncompressed_bias_q_payload_bytes"))
                total_full += row["full_weight_plus_bias_payload_bytes"]
                total_stored += row["stored_weight_plus_metadata_plus_bias_bytes"]
            s.update(logical_resident_total_bytes=total_stored, logical_full_reference_total_bytes=total_full)
        workers[treatment] = worker
        samples[treatment] = {"return_code": 0, "termination_reason": None, "elapsed_s": 1.,
            "sample_count_by_phase": {"measured": 3}, "peak_rss_by_phase": {"measured": 400},
            "guards": {"max_worker_rss_mib": 8192., "worker_timeout_s": 1800.}}
    return workers, samples


def test_gate_recomputes_five_worker_evidence_allows_cross_layout_operation_difference():
    workers, samples = inputs()
    before = copy.deepcopy(workers)
    data = validate_layout_gate(workers, samples, atol=.002, checkpoint_sha256="a"*64,
                                config=workers["cips/full"]["experiment"]["ckks_config"])
    assert data["acceptance"]["valid"] is True
    assert data["maximum_output_delta_vs_cips_full"] == 0
    assert len(data["rows"]) == 5
    assert workers == before


@pytest.mark.parametrize("problem", ["gate", "checkpoint", "same_wrong_features", "reference", "reported_error",
    "output_nan", "output_truncated", "scope", "timing_zero", "overlap", "native_count", "native_bytes", "leak",
    "storage", "within_layout_ops", "pid", "rss_missing", "rss_killed", "rss_limit", "timeout", "incomplete", "missing"])
def test_gate_fails_closed_on_corrupt_raw_evidence(problem):
    workers, samples = inputs()
    w = workers["native_orion/online_encode"]
    if problem == "gate": w["acceptance"]["valid"] = False
    elif problem == "checkpoint": w["checkpoint"]["sha256"] = "c"*64
    elif problem == "same_wrong_features":
        for worker in workers.values(): worker["feature_sha256"] = "c"*64
    elif problem == "reference": w["correctness"]["independent_clear_output_values"][0] = .5
    elif problem == "reported_error": w["correctness"]["max_abs_error"] = 0.
    elif problem == "output_nan": w["correctness"]["output_values"][0] = float("nan")
    elif problem == "output_truncated": w["correctness"]["output_values"].pop()
    elif problem == "scope": w["experiment"]["padding_semantics"] = "zero"
    elif problem == "timing_zero": workers["cips/full"]["measurements"]["transform_evaluate_s"] = [0.]
    elif problem == "overlap": w["measurements"]["transform_evaluate_s"] = [3.]
    elif problem == "native_count": w["backend"]["measured_transform_encode_invocations"]["ordinary"] = 0
    elif problem == "native_bytes": w["backend"]["native_transform_rows"]["checkpoint_up1"]["0"]["full_payload_bytes"] += 1
    elif problem == "leak": w["backend"]["native_materialized_transform_count_after_forward"] = 1
    elif problem == "storage": w["storage"]["logical_resident_total_bytes"] += 1
    elif problem == "within_layout_ops":
        for key in ("operation_counters_total", "operation_counters_per_forward"):
            w["measurements"][key]["conjugation"] += 1
    elif problem == "pid": w["process"]["pid"] = workers["cips/full"]["process"]["pid"]
    elif problem == "rss_missing": samples["native_orion/full"]["sample_count_by_phase"]["measured"] = 0
    elif problem == "rss_killed": samples["native_orion/full"]["termination_reason"] = "watchdog"
    elif problem == "rss_limit": samples["native_orion/full"]["peak_rss_by_phase"]["measured"] = 10**12
    elif problem == "timeout": samples["native_orion/full"]["elapsed_s"] = 2000.
    elif problem == "incomplete": workers.pop("cips/compressed")
    else: w["correctness"].pop("independent_clear_output_values")
    with pytest.raises(ValueError): validate_layout_gate(workers, samples, atol=.002)


def test_plan_only_validates_without_loading_checkpoint(monkeypatch, capsys):
    from tools.run_wpc_layout_gate import parser, run
    args = parser().parse_args(["--plan-only", "--checkpoint", "/missing/checkpoint"])
    assert run(args) == 0
    assert "native_orion/full" in capsys.readouterr().out
    args = parser().parse_args(["--plan-only", "--height", "4", "--width", "4"])
    with pytest.raises(ValueError, match="aligned"):
        run(args)


def test_transferred_review_recomputes_and_checks_manifest(tmp_path):
    from tools.run_wpc_layout_gate import review, render_report, save, source_hashes
    workers, samples = inputs()
    reference = workers["cips/full"]
    experiment = reference["experiment"]
    request = {"ckks_config": experiment["ckks_config"], "configuration_source": experiment["configuration_source"],
        "geometry": experiment["geometry"], "resource_guards": {"max_worker_rss_mib": 8192., "worker_timeout_s": 1800.},
        "worker_request": {"seed": reference["seed"], **{k: experiment[k] for k in (
            "bsgs_ratio", "feature_std", "bound_headroom", "atol", "forward_runs", "warmup_runs", "verify_exact_qp")}}}
    artifacts = []
    for treatment in TREATMENTS:
        layout, mode = treatment.split("/")
        directory = tmp_path / layout
        directory.mkdir(exist_ok=True)
        for suffix, payload in (("worker.json", workers[treatment]), ("rss.json", samples[treatment]),
                                ("worker.log", {"test": True}), ("phase.json", {"phase": "complete"})):
            path = directory / f"{mode}.{suffix}"
            save(path, payload)
            artifacts.append({"path": str(path.relative_to(tmp_path)), "size_bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    config_path = tmp_path / "configuration_input.json"
    save(config_path, request["ckks_config"])
    data = validate_layout_gate(workers, samples, atol=.002)
    data.update(schema_version=1, request=request, performance_claims_enabled=False, security_assessed=False,
        provenance={"checkpoint": {"sha256": "a"*64}, "implementation_sha256": source_hashes(), "worker_artifacts": artifacts,
            "configuration_artifact": {"path": config_path.name, "size_bytes": config_path.stat().st_size,
                                       "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}})
    save(tmp_path / "requested_run.json", request)
    save(tmp_path / "gate.json", data)
    (tmp_path / "gate.md").write_text(render_report(data))
    assert review(tmp_path) == 0
    (tmp_path / "native_orion/full.worker.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        review(tmp_path)
