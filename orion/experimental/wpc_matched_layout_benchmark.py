"""Stage 31: balanced, repeated matched-function layout/storage experiment.

Uncertainty resamples paired fresh-process blocks, never individual forwards.
Logical coefficient/recipe bytes and sampled process RSS are distinct metrics.
"""
from __future__ import annotations

import random
import statistics

import numpy as np

from orion.experimental.wpc_layout_gate import TREATMENTS, COMMON_EXPERIMENT, validate_layout_gate
from orion.experimental.wpc_orion_layout_control import NATIVE_PREPARATION_POLICY
from orion.experimental.wpc_cips_benchmark import summarize_samples
from orion.experimental.wpc_evidence_validation import EvidenceValidationError, close, finite, integer
from orion.experimental.wpc_online_encode_benchmark import _percentile

COMPONENTS = ("online_prepare_s", "online_encode_s", "decompression_s", "transform_evaluate_s", "activation_s", "bootstrap_s")
RATIOS = (
    ("cips/compressed", "native_orion/online_encode"),
    ("cips/compressed", "native_orion/full"),
    ("cips/full", "native_orion/full"),
    ("cips/online_encode", "native_orion/online_encode"),
    ("cips/compressed", "cips/online_encode"),
    ("cips/compressed", "cips/full"),
)


def balanced_orders(blocks, *, seed=0, smoke=False):
    blocks = integer(blocks, name="process blocks", minimum=1)
    if smoke:
        if blocks != 1:
            raise EvidenceValidationError("smoke requires exactly one block")
        return [list(TREATMENTS)]
    if blocks % 10:
        raise EvidenceValidationError("five-treatment Williams design requires a multiple of ten blocks")
    base = [0, 1, 4, 2, 3]
    cycle = [[TREATMENTS[(i + shift) % 5] for i in base] for shift in range(5)]
    cycle += [list(reversed(row)) for row in cycle]
    rng, orders = random.Random(seed), []
    for _ in range(blocks // 10):
        rows = [row.copy() for row in cycle]
        rng.shuffle(rows)
        orders.extend(rows)
    return orders


def compare_block(workers, rss, *, order, atol, checkpoint_sha256, config):
    if sorted(order) != sorted(TREATMENTS):
        raise EvidenceValidationError("block order must contain every treatment once")
    gate = validate_layout_gate(workers, rss, atol=atol, checkpoint_sha256=checkpoint_sha256,
                                config=config, repeated=True)
    reference = workers["cips/full"]
    rows = {}
    for treatment in TREATMENTS:
        worker = workers[treatment]
        exp, m, correctness = worker["experiment"], worker["measurements"], worker["correctness"]
        count = exp["forward_runs"]
        if exp.get("retain_sample_outputs") is not True:
            raise EvidenceValidationError("every measured output must be retained")
        expected_policy = NATIVE_PREPARATION_POLICY if treatment.startswith("native_orion/") else None
        if exp.get("native_preparation_policy") != expected_policy:
            raise EvidenceValidationError("native preparation policy is not the block-pruned implementation")
        if exp["runtime_environment"] != reference["experiment"]["runtime_environment"]:
            raise EvidenceValidationError("runtime worker environment differs between treatments")
        clear = np.asarray(correctness["independent_clear_output_values"], dtype=np.float64)
        raw_outputs = correctness.get("measured_output_values")
        if not isinstance(raw_outputs, list) or len(raw_outputs) != count or any(not isinstance(row, list) or len(row) != len(clear) for row in raw_outputs):
            raise EvidenceValidationError("missing/truncated measured output arrays")
        for row in raw_outputs:
            for value in row:
                finite(value, name="measured output")
        outputs = np.asarray(raw_outputs, dtype=np.float64)
        if outputs.shape != (count, len(clear)) or not np.isfinite(outputs).all():
            raise EvidenceValidationError("missing/non-finite measured output arrays")
        errors = np.max(np.abs(outputs - clear), axis=1)
        for reported, recomputed in zip(m["measured_output_max_abs_errors"], errors):
            close(reported, float(recomputed), name="per-forward clear error", atol=1e-10)
        if float(errors.max()) > atol or outputs[-1].tolist() != correctness["output_values"]:
            raise EvidenceValidationError("measured output failed tolerance/final-output consistency")
        if m.get("measured_operation_counters") != [m["operation_counters_per_forward"]] * count:
            raise EvidenceValidationError("per-forward operation counts are missing or vary")
        for counters in m["measured_operation_counters"]:
            for value in counters.values():
                integer(value, name="measured operation count")
        for key in ("forward_wall_s", "decompression_s", "transform_evaluate_s", "activation_s", "bootstrap_s"):
            summary_key = key.replace("_s", "_summary_s") if key != "forward_wall_s" else "forward_wall_summary_s"
            if m.get(summary_key) != summarize_samples(m[key]):
                raise EvidenceValidationError("worker sample summary differs from raw timings")
        wall_mean = statistics.mean(m["forward_wall_s"])
        row = {"forward_median_s": statistics.median(m["forward_wall_s"]),
               "forward_mean_s": wall_mean,
               "logical_resident_bytes": worker["storage"]["logical_resident_total_bytes"],
               "sampled_measured_peak_rss_bytes": rss[treatment]["peak_rss_by_phase"]["measured"],
               "pre_online_rss_bytes": worker["memory"]["pre_measured_gc"]["current_rss_bytes"],
               "maximum_clear_error": float(errors.max())}
        residual = []
        for key in COMPONENTS:
            row[key] = statistics.mean(m[key])
            row[key + "_pct"] = statistics.mean(100 * value / wall for value, wall in zip(m[key], m["forward_wall_s"]))
        for i, wall in enumerate(m["forward_wall_s"]):
            remainder = wall - sum(m[key][i] for key in COMPONENTS)
            if remainder < 0:
                raise EvidenceValidationError("negative residual wall time")
            residual.append(remainder)
        row["other_forward_s"] = statistics.mean(residual)
        row["other_forward_pct"] = statistics.mean(100 * value / wall for value, wall in zip(residual, m["forward_wall_s"]))
        close(sum(row[key] for key in COMPONENTS) + row["other_forward_s"], wall_mean, name="wall closure")
        close(sum(row[key + "_pct"] for key in COMPONENTS) + row["other_forward_pct"], 100, name="share closure")
        for key in ("python", "platform", "torch", "numpy"):
            if worker["process"][key] != reference["process"][key]:
                raise EvidenceValidationError("software/host differs between treatments")
        rows[treatment] = row
    identity = {key: reference[key] for key in ("feature_sha256", "seed", "runtime_parameter_manifest", "bootstrap_range")}
    identity.update(checkpoint_sha256=reference["checkpoint"]["sha256"],
                    shared_library_sha256=reference["shared_library"]["sha256"],
                    experiment={key: reference["experiment"][key] for key in COMMON_EXPERIMENT},
                    software={key: reference["process"][key] for key in ("python", "platform", "torch", "numpy")},
                    runtime_environment=reference["experiment"]["runtime_environment"])
    policies = {t: workers[t]["measurements"]["operation_counters_per_forward"] for t in TREATMENTS}
    return {"valid": True, "order": order, "identity": identity, "rows": rows,
            "operations": policies, "maximum_final_output_delta": gate["maximum_output_delta_vs_cips_full"]}


def summarize_blocks(blocks, *, orders, smoke=False, resamples=10000, seed=0):
    if not blocks or len(blocks) != len(orders) or any(b.get("valid") is not True or b["order"] != order for b, order in zip(blocks, orders)):
        raise EvidenceValidationError("missing/invalid or incorrectly ordered process blocks")
    # Check the schedule itself, not merely the number of blocks.
    canonical = balanced_orders(len(blocks), seed=0, smoke=smoke)
    if sorted(map(tuple, orders)) != sorted(map(tuple, canonical)):
        raise EvidenceValidationError("order schedule is not a complete Williams design")
    if any(b["identity"] != blocks[0]["identity"] or b["operations"] != blocks[0]["operations"] for b in blocks):
        raise EvidenceValidationError("input/software/parameters/operations changed between blocks")
    count = len(blocks)
    resamples = integer(resamples, name="bootstrap resamples", minimum=100)
    rng = random.Random(seed)
    indices = np.asarray([[rng.randrange(count) for _ in range(count)] for _ in range(resamples)], dtype=np.int64) if not smoke else None

    def estimate(values):
        values = [finite(v, name="block statistic", minimum=0) for v in values]
        draws = np.asarray(values)[indices].mean(axis=1).tolist() if indices is not None else []
        return {"count": count, "mean": statistics.mean(values), "median": statistics.median(values),
                "stdev": statistics.stdev(values) if count > 1 else None,
                "ci95_mean": [_percentile(draws, .025), _percentile(draws, .975)] if draws else None}

    metrics = {t: {key: estimate([b["rows"][t][key] for b in blocks]) for key in blocks[0]["rows"][t]} for t in TREATMENTS}
    ratios = {f"{left}_over_{right}": estimate([b["rows"][left]["forward_median_s"] / b["rows"][right]["forward_median_s"] for b in blocks])
              for left, right in RATIOS}
    return {"metrics": metrics, "paired_latency_ratios": ratios, "block_count": count,
            "identity": blocks[0]["identity"], "operations": blocks[0]["operations"],
            "maximum_final_output_delta": max(b["maximum_final_output_delta"] for b in blocks),
            "inference": "percentile bootstrap of matched fresh-process blocks; mean of block medians for latency; descriptive pointwise intervals, not multiple-comparison-adjusted",
            "resamples": resamples, "resample_seed": seed}
