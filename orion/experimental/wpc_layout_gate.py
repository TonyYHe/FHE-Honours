"""Independent raw-evidence checks for a matched-function layout gate."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from orion.experimental.wpc_cips_trained_benchmark import CONFIGURABLE_WORKER_GATES
from orion.experimental.wpc_online_encode_benchmark import EXTRA_GATES, validate_three_way_worker
from orion.experimental.wpc_decoder_geometry import expected_worker_geometry, validate_parameter_manifest
from orion.experimental.wpc_evidence_validation import EvidenceValidationError, close, finite, gates, integer, sha256

TREATMENTS = ("native_orion/full", "native_orion/online_encode", "cips/full", "cips/online_encode", "cips/compressed")
COMMON_EXPERIMENT = ("graph", "logn", "slots", "low_shape", "high_shape", "bsgs_ratio", "feature_std",
                     "bound_headroom", "warmup_runs", "forward_runs", "atol", "ckks_config",
                     "configuration_source", "geometry", "verify_exact_qp", "security_scope", "padding_semantics")


def _native(worker, mode, atol):
    gates(worker, name="native worker", required=CONFIGURABLE_WORKER_GATES | EXTRA_GATES)
    if worker.get("profile") != "wpc_native_orion_matched_decoder_worker" or worker.get("schema_version") != 3:
        raise EvidenceValidationError("native worker schema/profile mismatch")
    geometry = expected_worker_geometry(worker)
    validate_parameter_manifest(worker["runtime_parameter_manifest"], worker["experiment"]["ckks_config"], geometry["slots"])
    if worker["bootstrap_compile"]["ciphertext_group_count"] != geometry["bootstrap_ciphertext_group_count"]:
        raise EvidenceValidationError("native bootstrap group count mismatch")
    storage, m, backend = worker["storage"], worker["measurements"], worker["backend"]
    layers = storage["by_layer"]
    expected = geometry["learned_transform_count"]
    if storage["learned_transform_count"] != expected or {k: r["transform_count"] for k, r in layers.items()} != geometry["learned_transform_count_by_layer"]:
        raise EvidenceValidationError("native transform counts mismatch")
    if storage["concat"]["transform_count"] != 0 or storage["concat_full_qp_payload_bytes"] != 0:
        raise EvidenceValidationError("aligned native concat must not retain plaintext transforms")
    if storage["storage_mode"] != mode:
        raise EvidenceValidationError("native storage mode mismatch")
    resident = full = 0
    for name, row in layers.items():
        rows = backend["native_transform_rows"][name]
        if len(rows) != row["transform_count"]:
            raise EvidenceValidationError("native transform byte records incomplete")
        qp = sum(integer(r["full_payload_bytes"], name="transform Q/P bytes", minimum=1) for r in rows.values())
        if qp != row["full_weight_qp_payload_bytes"] or any(r["payload_bytes_measured"] is not (mode == "full") for r in rows.values()):
            raise EvidenceValidationError("native Q/P byte evidence mismatch")
        level = geometry["levels"][{"checkpoint_up1": "up1", "checkpoint_dec1a": "dec1a", "checkpoint_dec1b": "dec1b"}[name]]
        n = 2 * geometry["slots"]
        p_limbs = len(worker["experiment"]["ckks_config"]["ckks_params"]["LogP"])
        for observation in rows.values():
            diagonals = integer(observation["diagonal_count"], name="native diagonal count", minimum=1)
            q_bytes, p_bytes = diagonals * n * (level + 1) * 8, diagonals * n * p_limbs * 8
            if observation["full_payload_bytes"] != q_bytes + p_bytes:
                raise EvidenceValidationError("native diagonal/limb payload formula mismatch")
            actual = observation["actual_qp_stats"]
            if (mode == "full" and actual != {"schema_version": 1, "diagonal_count": diagonals,
                                               "q_bytes": q_bytes, "p_bytes": p_bytes, "total_bytes": q_bytes + p_bytes}) or (mode != "full" and actual is not None):
                raise EvidenceValidationError("native raw Q/P coefficient statistics mismatch")
        bias = integer(row["uncompressed_bias_q_payload_bytes"], name="bias bytes")
        recipe = integer(row["unencoded_recipe_payload_bytes"], name="recipe bytes")
        meta = integer(row["weight_metadata_bytes"], name="metadata bytes")
        weight = integer(row["resident_weight_qp_payload_bytes"], name="resident Q/P bytes")
        if weight != (qp if mode == "full" else 0) or (recipe > 0) != (mode == "online_encode"):
            raise EvidenceValidationError("native resident policy mismatch")
        if row["stored_weight_plus_metadata_plus_bias_bytes"] != weight + recipe + meta + bias or row["full_weight_plus_bias_payload_bytes"] != qp + bias:
            raise EvidenceValidationError("native layer byte accounting does not close")
        resident += weight + recipe + meta + bias
        full += qp + bias
    if storage["logical_resident_total_bytes"] != resident or storage["logical_full_reference_total_bytes"] != full:
        raise EvidenceValidationError("native total byte accounting does not close")
    if backend["native_materialized_transform_count_after_forward"] != 0:
        raise EvidenceValidationError("native online transform leaked")
    if backend["compile_transform_encode_invocations"] != {"ordinary": expected if mode == "full" else 0, "compressed": 0, "online_recipe": 0}:
        raise EvidenceValidationError("native compile Encode count mismatch")
    calls = expected if mode == "online_encode" else 0
    for counts in (backend["compile_transform_encode_invocations"], backend["measured_transform_encode_invocations"]):
        for value in counts.values():
            integer(value, name="native Encode counter")
    for value in m["transform_encode_invocations"]:
        integer(value, name="native per-forward Encode counter")
    if backend["measured_transform_encode_invocations"] != {"ordinary": calls, "compressed": 0, "online_recipe": 0}:
        raise EvidenceValidationError("native actual online Encode count mismatch")
    if backend["weight_plaintext_online_encode_calls"] != calls or backend["weight_plaintext_offline_encode_calls"] != (expected if mode == "full" else 0):
        raise EvidenceValidationError("native weight Encode count mismatch")
    if m["transform_encode_invocations"] != [calls] or m["online_python_encode_call_count"] != 0 or m["bootstrap_call_count"] != 1:
        raise EvidenceValidationError("native per-forward call counts mismatch")
    if m["timed_lifecycle_record_count"] != [0]:
        raise EvidenceValidationError("native timed lifecycle tracing enabled")
    traces = [r for sequence in worker["correctness"]["untimed_lifecycle_trace"] for r in sequence]
    if len(traces) != expected or any(r["materialized_transforms_after"] != 0 for r in traces):
        raise EvidenceValidationError("native preflight release trace incomplete")
    if m["decompression_s"] != [0.0]:
        raise EvidenceValidationError("native path claims WPC decompression")
    if mode == "full" and (m["online_encode_s"] != [0.0] or m["online_prepare_s"] != [0.0]):
        raise EvidenceValidationError("native full path claims online Encode")
    if mode == "online_encode" and (m["online_encode_s"][0] <= 0 or m["online_prepare_s"][0] <= 0):
        raise EvidenceValidationError("native online timers must be positive")
    close(m["online_encode_pct_of_forward"][0], 100 * m["online_encode_s"][0] / m["forward_wall_s"][0], name="native Encode share")
    close(m["online_materialization_pct_of_forward"][0], 100 * (m["online_encode_s"][0] + m["online_prepare_s"][0]) / m["forward_wall_s"][0], name="native preparation share")
    for stats in (backend["compressed_global_stats"], backend["online_recipe_global_stats"]):
        if stats["registered_transform_count"] != 0:
            raise EvidenceValidationError("native path registered CIPS recipes/compression")


def validate_layout_gate(workers, samples, *, atol, checkpoint_sha256=None, config=None):
    """Recompute correctness/closure; never trust success flags alone."""
    try:
        if set(workers) != set(TREATMENTS) or set(samples) != set(TREATMENTS):
            raise EvidenceValidationError("layout gate requires all five treatments")
        reference = workers["cips/full"]
        pids = set()
        rows = {}
        for treatment in TREATMENTS:
            layout, mode = treatment.split("/")
            worker, rss = workers[treatment], samples[treatment]
            if worker.get("status") != "ok" or worker.get("mode") != mode or worker.get("layout") != layout:
                raise EvidenceValidationError("worker identity/status mismatch")
            if layout == "cips":
                validate_three_way_worker(worker, mode=mode, atol=atol)
            else:
                _native(worker, mode, atol)
            experiment, m, correctness = worker["experiment"], worker["measurements"], worker["correctness"]
            if experiment["forward_runs"] != 1 or experiment["warmup_runs"] != 0 or experiment["atol"] != atol or experiment["padding_semantics"] != "flattened_spatial_cyclic":
                raise EvidenceValidationError("layout gate requires matched function, zero warmups and one diagnostic forward")
            if experiment["timed_lifecycle_tracing"] is not False:
                raise EvidenceValidationError("timed trace must be disabled")
            if any(experiment[key] != reference["experiment"][key] for key in COMMON_EXPERIMENT):
                raise EvidenceValidationError("layout experiment request mismatch")
            if config is not None and experiment["ckks_config"] != config:
                raise EvidenceValidationError("worker changed requested configuration")
            for key in ("checkpoint", "shared_library"):
                sha256(worker[key]["sha256"], name=key)
                if worker[key]["sha256"] != reference[key]["sha256"]:
                    raise EvidenceValidationError(f"{key} identity mismatch")
            if checkpoint_sha256 is not None and worker["checkpoint"]["sha256"] != checkpoint_sha256:
                raise EvidenceValidationError("checkpoint differs from frozen controller request")
            for key in ("feature_sha256", "seed", "runtime_parameter_manifest", "bootstrap_range"):
                if worker[key] != reference[key]:
                    raise EvidenceValidationError(f"matched input/parameters mismatch: {key}")
            rng = np.random.default_rng(worker["seed"])
            low = rng.normal(0, experiment["feature_std"], experiment["low_shape"]).astype(np.float64)
            skip = rng.normal(0, experiment["feature_std"], experiment["high_shape"]).astype(np.float64)
            if hashlib.sha256(low.tobytes() + skip.tobytes()).hexdigest() != worker["feature_sha256"]:
                raise EvidenceValidationError("exact feature digest does not match regenerated input")
            shape = experiment["high_shape"][1:]
            if correctness["output_shape"] != shape:
                raise EvidenceValidationError("output shape mismatch")
            size = math.prod(shape)
            for key in ("output_values", "independent_clear_output_values"):
                if not isinstance(correctness[key], list) or len(correctness[key]) != size:
                    raise EvidenceValidationError("output/reference missing or truncated")
                for value in correctness[key]:
                    finite(value, name=key)
            output = np.asarray(correctness["output_values"], dtype=np.float64)
            clear = np.asarray(correctness["independent_clear_output_values"], dtype="<f8")
            if output.shape != (size,) or clear.shape != (size,) or not np.isfinite(output).all() or not np.isfinite(clear).all():
                raise EvidenceValidationError("output/reference missing, non-finite or truncated")
            digest = hashlib.sha256(clear.tobytes()).hexdigest()
            if digest != correctness["independent_clear_output_sha256"] or digest != reference["correctness"]["independent_clear_output_sha256"]:
                raise EvidenceValidationError("independent clear reference digest mismatch")
            error = float(np.max(np.abs(output - clear)))
            close(correctness["max_abs_error"], error, name="reported clear error", atol=1e-10)
            if error > atol or correctness["correct"] is not True or finite(correctness["preflight_max_abs_error"], name="preflight error", minimum=0) > atol:
                raise EvidenceValidationError("FHE output exceeds clear tolerance")
            if finite(worker["clear_oracle_max_abs_delta"], name="clear oracle delta", minimum=0) > 1e-10:
                raise EvidenceValidationError("independent clear oracle mismatch")
            if len(m["measured_output_max_abs_errors"]) != 1 or finite(m["measured_output_max_abs_errors"][0], name="measured error", minimum=0) > atol:
                raise EvidenceValidationError("measured output failed")
            times = {}
            for key in ("forward_wall_s", "online_encode_s", "online_prepare_s", "decompression_s", "transform_evaluate_s", "activation_s", "bootstrap_s"):
                if not isinstance(m[key], list) or len(m[key]) != 1:
                    raise EvidenceValidationError("missing diagnostic timer sample")
                times[key] = finite(m[key][0], name=key, minimum=0)
            if min(times[k] for k in ("forward_wall_s", "transform_evaluate_s", "activation_s", "bootstrap_s")) <= 0:
                raise EvidenceValidationError("measured computation timer is not positive")
            if sum(v for k, v in times.items() if k != "forward_wall_s") > times["forward_wall_s"] + 1e-6:
                raise EvidenceValidationError("forward subtimers overlap or exceed wall time")
            operations = m["operation_counters_per_forward"]
            if operations["rotation_total"] != operations["direct_rotation"] + operations["linear_transform_rotation"] or m["operation_counters_total"] != operations:
                raise EvidenceValidationError("operation counters do not close")
            for value in operations.values():
                integer(value, name="operation count")
            pid = integer(worker["process"]["pid"], name="worker PID", minimum=1)
            if pid in pids:
                raise EvidenceValidationError("workers were not separate processes")
            pids.add(pid)
            if rss["return_code"] != 0 or rss["termination_reason"] is not None:
                raise EvidenceValidationError("worker failed or resource guard terminated it")
            if integer(rss["sample_count_by_phase"].get("measured", 0), name="measured RSS samples", minimum=1) <= 0:
                raise EvidenceValidationError("no measured RSS samples")
            guards = rss["guards"]
            limit = finite(guards["max_worker_rss_mib"], name="RSS guard", minimum=1) * 2**20
            timeout = finite(guards["worker_timeout_s"], name="timeout guard", minimum=1)
            peak = max(integer(value, name="sampled RSS", minimum=1) for value in rss["peak_rss_by_phase"].values())
            if peak > limit or finite(rss["elapsed_s"], name="worker elapsed", minimum=0) > timeout:
                raise EvidenceValidationError("resource budget exceeded")
            rows[treatment] = {"maximum_error_vs_independent_clear": error,
                               "diagnostic_forward_s": times["forward_wall_s"],
                               "transform_evaluate_s": times["transform_evaluate_s"],
                               "logical_resident_bytes": worker["storage"]["logical_resident_total_bytes"],
                               "sampled_all_phase_peak_rss_bytes": peak,
                               "sampled_measured_peak_rss_bytes": rss["peak_rss_by_phase"]["measured"],
                               "operation_counters": operations}
        for layout in ("native_orion", "cips"):
            full = workers[f"{layout}/full"]
            for treatment in (t for t in TREATMENTS if t.startswith(layout + "/")):
                worker = workers[treatment]
                if worker["measurements"]["operation_counters_total"] != full["measurements"]["operation_counters_total"]:
                    raise EvidenceValidationError("within-layout operations differ across storage policies")
                if worker["storage"]["logical_full_reference_total_bytes"] != full["storage"]["logical_full_reference_total_bytes"]:
                    raise EvidenceValidationError("within-layout full-reference storage differs")
        native_full, native_online = workers["native_orion/full"], workers["native_orion/online_encode"]
        for name, full_rows in native_full["backend"]["native_transform_rows"].items():
            online_rows = native_online["backend"]["native_transform_rows"][name]
            if {k: (v["diagonal_count"], v["full_payload_bytes"]) for k, v in full_rows.items()} != {k: (v["diagonal_count"], v["full_payload_bytes"]) for k, v in online_rows.items()}:
                raise EvidenceValidationError("native online estimates differ from measured full Q/P storage")
        output_reference = np.asarray(reference["correctness"]["output_values"])
        max_delta = max(float(np.max(np.abs(np.asarray(w["correctness"]["output_values"]) - output_reference))) for w in workers.values())
        if max_delta > atol:
            raise EvidenceValidationError("cross-layout outputs exceed recorded tolerance")
        return {"status": "ok", "rows": rows, "maximum_output_delta_vs_cips_full": max_delta,
                "acceptance": {"all_raw_evidence_checked": True, "matched_function_and_provenance": True,
                               "within_layout_operations_match": True, "resource_guards_passed": True, "valid": True}}
    except (KeyError, TypeError, IndexError) as error:
        raise EvidenceValidationError(f"missing/malformed layout evidence: {error}") from error
