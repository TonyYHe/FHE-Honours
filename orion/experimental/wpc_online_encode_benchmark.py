"""Fail-closed three-way CIPS storage comparison and process-block uncertainty.

Within-process forwards are not independent trials. Every confidence interval
below resamples whole matched process blocks, preserving the three treatments.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from typing import Any

from orion.experimental.wpc_cips_trained_benchmark import expected_transform_count, validate_worker
from orion.experimental.wpc_evidence_validation import (
    EvidenceValidationError, close, finite, gates, integer, sha256,
)

MODES = ("online_encode", "full", "compressed")
EXTRA_GATES = {
    "untimed_preflight_correct", "all_measured_outputs_correct",
    "timed_lifecycle_tracing_disabled", "actual_transform_encode_invocations_match_mode",
    "online_recipe_materialization_released",
}


def balanced_orders(blocks: int, *, seed: int = 0, smoke: bool = False) -> list[list[str]]:
    blocks = integer(blocks, name="trial blocks", minimum=1)
    if smoke:
        if blocks != 1:
            raise EvidenceValidationError("smoke mode requires exactly one trial block")
        return [list(MODES)]
    if blocks % 6:
        raise EvidenceValidationError("trial blocks must be a positive multiple of six")
    rng = random.Random(seed)
    orders = []
    for _ in range(blocks // 6):
        cycle = list(itertools.permutations(MODES))
        rng.shuffle(cycle)
        orders.extend(map(list, cycle))
    return orders


def _samples(values: Any, count: int, name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != count:
        raise EvidenceValidationError(f"{name}: expected {count} samples")
    return [finite(value, name=name, minimum=0) for value in values]


def validate_three_way_worker(worker: dict[str, Any], *, mode: str, atol: float) -> None:
    validate_worker(worker, mode=mode, atol=atol)
    gates(worker, name=f"{mode} Stage-28 worker", required=EXTRA_GATES)
    if worker.get("schema_version") not in (2, 3):
        raise EvidenceValidationError("three-way benchmark requires worker schema 2 or 3")
    if worker["experiment"].get("timed_lifecycle_tracing") is not False:
        raise EvidenceValidationError("lifecycle tracing must be outside measured forwards")
    if worker["schema_version"] == 2 and worker["experiment"].get("security_scope") != "small_insecure_functional_test_not_secure_deployment":
        raise EvidenceValidationError("functional security scope must be explicit")
    sha256(worker.get("feature_sha256"), name="exact feature identity")
    sha256(worker["shared_library"]["sha256"], name="loaded backend binary identity")
    preflight = finite(worker["correctness"]["preflight_max_abs_error"], name="preflight error", minimum=0)
    if preflight > atol:
        raise EvidenceValidationError("untimed correctness preflight exceeds tolerance")
    m = worker["measurements"]
    count = worker["experiment"]["forward_runs"]
    transforms = expected_transform_count(worker)
    encode = _samples(m.get("online_encode_s"), count, "online Encode")
    prepare = _samples(m.get("online_prepare_s"), count, "recipe preparation")
    shares = _samples(m.get("online_encode_pct_of_forward"), count, "Encode shares")
    materialization = _samples(m.get("online_materialization_pct_of_forward"), count, "materialization shares")
    calls = _samples(m.get("transform_encode_invocations"), count, "Encode invocation counts")
    errors = _samples(m.get("measured_output_max_abs_errors"), count, "measured errors")
    trace_counts = _samples(m.get("timed_lifecycle_record_count"), count, "timed lifecycle records")
    if any(trace_counts):
        raise EvidenceValidationError("lifecycle audit records were produced during timed forwards")
    expected_calls = transforms if mode == "online_encode" else 0
    for i, wall in enumerate(m["forward_wall_s"]):
        if calls[i] != expected_calls or errors[i] > atol:
            raise EvidenceValidationError("measured Encode calls or output errors do not match contract")
        if mode == "online_encode":
            if encode[i] <= 0 or prepare[i] <= 0 or m["decompression_s"][i] != 0:
                raise EvidenceValidationError("online recipe must prepare and Encode, not decompress")
        elif encode[i] != 0 or prepare[i] != 0:
            raise EvidenceValidationError("preencoded workers must not report online Encode")
        components = encode[i] + prepare[i] + sum(m[key][i] for key in (
            "decompression_s", "transform_evaluate_s", "activation_s", "bootstrap_s"))
        if components > wall + max(1e-6, wall * 1e-6):
            raise EvidenceValidationError("three-way subtimers exceed forward wall time")
        close(shares[i], 100 * encode[i] / wall, name="Encode percentage")
        close(materialization[i], 100 * (encode[i] + prepare[i]) / wall, name="materialization percentage")
    if mode == "compressed" and any(value <= 0 for value in m["decompression_s"]):
        raise EvidenceValidationError("compressed worker has no decompression measurement")
    backend = worker["backend"]
    measured_counts = backend["measured_transform_encode_invocations"]
    for counts in (measured_counts, backend["compile_transform_encode_invocations"]):
        if set(counts) != {"ordinary", "compressed", "online_recipe"}:
            raise EvidenceValidationError("missing or unknown backend Encode counter")
        for value in counts.values():
            integer(value, name="actual backend Encode invocation count")
    if measured_counts != {"ordinary": 0, "compressed": 0, "online_recipe": count * expected_calls}:
        raise EvidenceValidationError("backend Encode counters disagree with per-forward observations")
    if backend["weight_plaintext_online_encode_calls"] != count * expected_calls:
        raise EvidenceValidationError("weight Encode summary disagrees with actual calls")
    concat_count = integer(worker["storage"]["concat"]["transform_count"], name="common concat transforms")
    verification_calls = transforms if mode == "compressed" and worker["experiment"].get("verify_exact_qp", False) else 0
    compile_expected = {"ordinary": concat_count + (transforms if mode == "full" else verification_calls),
                        "compressed": transforms if mode == "compressed" else 0, "online_recipe": 0}
    if backend["compile_transform_encode_invocations"] != compile_expected:
        raise EvidenceValidationError("offline Encode counts disagree with selected storage mode")
    if backend["weight_plaintext_offline_encode_calls"] != (0 if mode == "online_encode" else transforms):
        raise EvidenceValidationError("offline learned-transform Encode count is inconsistent")
    online = backend["online_recipe_global_stats"]
    comp = backend["compressed_global_stats"]
    for key in ("current_materialized_bytes", "current_materialized_transforms"):
        if integer(online[key], name=key) != 0:
            raise EvidenceValidationError("online recipe left materialization resident")
    for key in ("current_materialized_full_payload_bytes", "current_materialized_transform_count"):
        if integer(comp[key], name=key) != 0:
            raise EvidenceValidationError("compressed path left materialization resident")
    if integer(online["registered_transform_count"], name="recipe registry") != (transforms if mode == "online_encode" else 0):
        raise EvidenceValidationError("recipe registry disagrees with storage mode")
    if integer(comp["registered_transform_count"], name="compressed registry") != (transforms if mode == "compressed" else 0):
        raise EvidenceValidationError("compressed registry disagrees with storage mode")
    if mode in ("online_encode", "compressed"):
        integer(backend["expected_max_single_transform_bytes"], name="maximum full transform bytes", minimum=1)
        peak_count = online["peak_materialized_transforms"] if mode == "online_encode" else comp["peak_materialized_transform_count"]
        peak_bytes = online["peak_materialized_bytes"] if mode == "online_encode" else comp["peak_materialized_full_payload_bytes"]
        if integer(peak_count, name="peak transform count") != 1 or peak_bytes != backend["expected_max_single_transform_bytes"]:
            raise EvidenceValidationError("materialization peak is not exactly one full transform")
    storage = worker["storage"]
    rows = storage["by_layer"].values()
    resident = 0
    full = 0
    for row in rows:
        qp = integer(row["resident_weight_qp_payload_bytes"], name="resident Q/P bytes")
        recipe = integer(row["unencoded_recipe_payload_bytes"], name="recipe bytes")
        meta = integer(row["weight_metadata_bytes"], name="metadata bytes")
        bias = integer(row["uncompressed_bias_q_payload_bytes"], name="bias bytes")
        if mode == "online_encode" and (qp != 0 or recipe <= 0):
            raise EvidenceValidationError("online worker must retain recipes, not learned Q/P")
        if mode != "online_encode" and recipe != 0:
            raise EvidenceValidationError("preencoded worker retained slot recipes")
        if mode == "full" and qp != row["full_weight_qp_payload_bytes"]:
            raise EvidenceValidationError("full worker storage is not the complete learned Q/P")
        if row["stored_weight_plus_metadata_plus_bias_bytes"] != qp + recipe + meta + bias:
            raise EvidenceValidationError("layer resident accounting does not close")
        resident += qp + recipe + meta + bias
        full += row["full_weight_qp_payload_bytes"] + bias
    trace = worker["correctness"]["untimed_lifecycle_trace"]
    if not isinstance(trace, list):
        raise EvidenceValidationError("missing untimed lifecycle trace")
    if mode == "compressed":
        records = [row for sequence in trace for row in sequence]
        if len(records) != transforms or any(row[key] != 0 for row in records for key in (
            "current_materialized_bytes_before", "current_materialized_bytes_after",
            "current_materialized_transforms_after")):
            raise EvidenceValidationError("untimed lifecycle audit does not verify sequential release")
    concat_bytes = integer(storage["concat_full_qp_payload_bytes"], name="concat bytes")
    if storage["logical_resident_total_bytes"] != resident + concat_bytes or storage["logical_full_reference_total_bytes"] != full + concat_bytes:
        raise EvidenceValidationError("logical resident storage accounting does not close")


def compare_block(workers: dict[str, dict[str, Any]], rss: dict[str, dict[str, Any]], *, order: list[str], atol: float) -> dict[str, Any]:
    if set(workers) != set(MODES) or sorted(order) != sorted(MODES):
        raise EvidenceValidationError("each process block must contain all three modes exactly once")
    for mode in MODES:
        validate_three_way_worker(workers[mode], mode=mode, atol=atol)
    reference = workers["full"]
    pids = [worker["process"]["pid"] for worker in workers.values()]
    if len(set(pids)) != 3:
        raise EvidenceValidationError("treatments must run in separate processes")
    rows = {}
    for mode in MODES:
        worker = workers[mode]
        if any(worker[key] != reference[key] for key in ("experiment", "feature_sha256", "seed")) or worker["checkpoint"]["sha256"] != reference["checkpoint"]["sha256"] or worker["shared_library"]["sha256"] != reference["shared_library"]["sha256"]:
            raise EvidenceValidationError("checkpoint, exact features, or configuration differ between treatments")
        if worker["measurements"]["operation_counters_per_forward"] != reference["measurements"]["operation_counters_per_forward"]:
            raise EvidenceValidationError("homomorphic operation counts differ between treatments")
        if worker.get("runtime_parameter_manifest") != reference.get("runtime_parameter_manifest"):
            raise EvidenceValidationError("actual residual/bootstrap parameters differ between treatments")
        if worker["correctness"]["output_shape"] != reference["correctness"]["output_shape"]:
            raise EvidenceValidationError("isolated output shapes differ")
        delta = max(abs(a-b) for a,b in zip(worker["correctness"]["output_values"], reference["correctness"]["output_values"]))
        if delta > atol:
            raise EvidenceValidationError("isolated outputs exceed comparison tolerance")
        sample = rss[mode]
        if integer(sample["sample_count_by_phase"].get("measured", 0), name="online RSS samples") <= 0:
            raise EvidenceValidationError("RSS sampling missed the measured phase")
        peak = integer(sample["peak_rss_by_phase"]["measured"], name="sampled peak RSS", minimum=1)
        m = worker["measurements"]
        rows[mode] = {
            "forward_median_s": statistics.median(m["forward_wall_s"]),
            "online_encode_median_s": statistics.median(m["online_encode_s"]),
            "online_prepare_median_s": statistics.median(m["online_prepare_s"]),
            "online_encode_median_pct": statistics.median(m["online_encode_pct_of_forward"]),
            "online_materialization_median_pct": statistics.median(m["online_materialization_pct_of_forward"]),
            "decompression_median_s": statistics.median(m["decompression_s"]),
            "decompression_median_pct": statistics.median(100*d/w for d,w in zip(m["decompression_s"], m["forward_wall_s"])),
            "activation_median_s": statistics.median(m["activation_s"]),
            "bootstrap_median_s": statistics.median(m["bootstrap_s"]),
            "logical_resident_bytes": worker["storage"]["logical_resident_total_bytes"],
            "pre_online_rss_bytes": worker["memory"]["pre_measured_gc"]["current_rss_bytes"],
            "sampled_online_peak_rss_bytes": peak,
            "max_abs_error_vs_clear": max(m["measured_output_max_abs_errors"]),
            "max_abs_delta_vs_isolated_full": delta,
        }
    result = {"order": order, "rows": rows, "checkpoint_sha256": reference["checkpoint"]["sha256"],
            "feature_sha256": reference["feature_sha256"], "shared_library_sha256": reference["shared_library"]["sha256"],
            "experiment": reference["experiment"], "valid": True}
    if "runtime_parameter_manifest" in reference:
        result["runtime_parameter_manifest"] = reference["runtime_parameter_manifest"]
    return result


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = math.floor(index)
    return values[low] + (values[math.ceil(index)] - values[low]) * (index - low)


def process_block_summary(blocks: list[dict[str, Any]], *, smoke: bool = False, resamples: int = 10000, seed: int = 0) -> dict[str, Any]:
    expected_orders = list(itertools.permutations(MODES))
    orders = [tuple(block["order"]) for block in blocks]
    if not blocks or any(block.get("valid") is not True for block in blocks):
        raise EvidenceValidationError("missing/invalid process blocks")
    if not smoke and (len(blocks) % 6 or any(orders.count(order) != len(blocks) // 6 for order in expected_orders)):
        raise EvidenceValidationError("independent process blocks are not order-balanced")
    identity = (blocks[0]["checkpoint_sha256"], blocks[0]["feature_sha256"], blocks[0]["shared_library_sha256"], blocks[0]["experiment"])
    if any((b["checkpoint_sha256"], b["feature_sha256"], b["shared_library_sha256"], b["experiment"]) != identity for b in blocks):
        raise EvidenceValidationError("checkpoint/features/configuration changed between process blocks")
    if any(b.get("runtime_parameter_manifest") != blocks[0].get("runtime_parameter_manifest") for b in blocks):
        raise EvidenceValidationError("actual runtime parameters changed between process blocks")
    count = len(blocks)
    resamples = integer(resamples, name="bootstrap resamples", minimum=100)
    rng = random.Random(seed)
    indices = [[rng.randrange(count) for _ in range(count)] for _ in range(resamples)] if not smoke else []

    def estimate(values: list[float]) -> dict[str, Any]:
        values = [finite(value, name="process-block statistic", minimum=0) for value in values]
        estimates = [statistics.mean(values[i] for i in sample) for sample in indices]
        return {"count": count, "mean": statistics.mean(values), "median": statistics.median(values),
                "stdev": statistics.stdev(values) if count > 1 else None,
                "ci95_mean": [_percentile(estimates, .025), _percentile(estimates, .975)] if estimates else None}

    metrics = {mode: {key: estimate([b["rows"][mode][key] for b in blocks])
                     for key in blocks[0]["rows"][mode]} for mode in MODES}
    ratios = {}
    for left, right in (("compressed", "online_encode"), ("compressed", "full"), ("full", "online_encode")):
        ratios[f"{left}_over_{right}_forward_ratio"] = estimate([
            b["rows"][left]["forward_median_s"] / b["rows"][right]["forward_median_s"] for b in blocks])
    return {"by_mode": metrics, "paired_ratios": ratios,
            "uncertainty_method": (
                "correctness-only smoke block; no uncertainty estimation or performance claim"
                if smoke else
                "percentile bootstrap of matched process blocks; mean of block statistics; not pooled-forward confidence intervals"
            ),
            "bootstrap_resamples": resamples if not smoke else 0, "bootstrap_seed": seed,
            "independent_block_count": count, "order_balanced": not smoke,
            "performance_claims_enabled": not smoke,
            "scope": ("same-CIPS trained decoder with synthetic features; security is not assessed; not Orion-versus-WPC layout or whole-network comparison"
                      if "geometry" in blocks[0]["experiment"] else
                      "same-CIPS trained decoder with synthetic features and insecure functional CKKS parameters; not Orion-versus-WPC layout or whole-network comparison")}
