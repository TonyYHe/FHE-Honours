"""Validation and synthesis helpers for the WPC--Orion trade-off study.

The helpers in this module deliberately separate three evidence scopes:

* whole-model Orion Step-1 wall-time profiles;
* read-only compatibility censuses of the unchanged Orion layouts; and
* WPC CIPS/Rotation-Padding evidence for a trained encrypted decoder stage.

Combining those scopes is useful, but none of them is silently promoted to a
complete encrypted-network WPC-versus-Orion comparison.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from orion.experimental.wpc_evidence_validation import (
    EvidenceValidationError, close as _close, finite as _strict_finite,
    gates as _gates, sha256 as _sha256,
)
from orion.experimental.wpc_cips_trained_benchmark import compare_trained_decoder_workers


EXPECTED_STEP1_MODELS = {
    "resnet20_cifar10": ("ResNet20", "dense"),
    "u22_64_base32": ("U-Net22", "provider"),
    "vgg16_imgnet": ("VGG16", "provider"),
}


ACCURACY_GATES = {
    "valid", "source_checkpoint_sha256_recorded",
    "all_18_spatial_convolutions_use_wpc_rotation_padding",
    "conversion_preserved_state_dict_keys_and_values",
    "checkpoint_cheb7_activations_remain_fully_polynomial",
    "native_validation_metrics_are_finite",
    "unfinetuned_rotation_padding_evaluation_is_fully_accounted",
    "best_rotation_padding_metrics_are_finite",
    "rotation_padding_semantic_change_was_observed",
    "best_checkpoint_is_finite_and_not_worse_when_comparable",
    "completed_epochs_cover_every_training_sample",
    "post_epoch_training_set_audits_are_fully_finite",
    "immutable_epoch_checkpoints_exist", "requested_epochs_completed",
    "fine_tuned_checkpoint_reloads_with_original_orion_schema",
    "best_and_last_checkpoints_exist", "evaluated_checkpoint_content_identity_recorded",
}
DECODER_GATES = {
    "valid", "checkpoint_architecture_and_tensor_shapes_validated",
    "checkpoint_activation_coefficients_and_scales_used_exactly",
    "independent_torch_wpc_rotation_padding_and_cips_clear_oracles_match",
    "all_compressed_weight_qp_exactly_matches_full_control",
    "full_and_compressed_encrypted_stages_match_clear",
    "compressed_final_output_exactly_matches_full_control",
    "trained_cheb7_and_real_bootstrap_executed", "ckks_level_schedule_valid",
    "cips_packing_chain_valid", "zero_online_python_encode_calls",
    "all_layout_boundaries_report_zero_clear_repack_or_encode",
    "full_and_compressed_operation_counters_match",
    "all_compressed_weight_materialization_released",
    "materialization_isolated_between_weight_transforms",
    "peak_weight_materialization_is_one_transform", "learned_weight_storage_is_compressed",
}


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _finite(value: Any, *, name: str, minimum: float | None = None) -> float:
    return _strict_finite(value, name=name, minimum=minimum)


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    number = _finite(value, name=name)
    result = int(number)
    if float(result) != number:
        raise EvidenceValidationError(f"{name} is not an integer: {value!r}")
    if minimum is not None and result < minimum:
        raise EvidenceValidationError(f"{name} is below {minimum}: {result}")
    return result


def _all_acceptance_gates(payload: Mapping[str, Any], *, name: str, required: set[str] | None = None) -> None:
    if required is not None:
        _gates(payload, name=name, required=required)
        return
    acceptance = _dict(payload.get("acceptance"))
    if not acceptance:
        raise EvidenceValidationError(f"{name} has no acceptance gates")
    failed = [key for key, value in acceptance.items() if value is not True]
    if failed:
        raise EvidenceValidationError(
            f"{name} has failed acceptance gates: {', '.join(sorted(failed))}"
        )


def _major_category(profile: Mapping[str, Any], category: str) -> dict[str, Any]:
    row = _dict(_dict(profile.get("major_wall_categories")).get(category))
    if not row:
        raise EvidenceValidationError(f"Step-1 profile is missing {category!r}")
    return row


def _validate_wall_profile(profile: Mapping[str, Any], *, name: str) -> None:
    """Recompute percentages and closure using a fixed, non-user-loosenable tolerance."""
    wall = _finite(profile.get("he_forward_s"), name=f"{name} HE forward", minimum=0)
    if wall <= 0 or profile.get("valid") is not True or profile.get("validation_errors"):
        raise EvidenceValidationError(f"{name} has an invalid wall profile")
    major = _dict(profile.get("major_wall_categories"))
    if not major:
        raise EvidenceValidationError(f"{name} has no major wall categories")
    seconds = []
    for category, value in major.items():
        row = _dict(value)
        duration = _finite(row.get("seconds"), name=f"{name}/{category} seconds", minimum=0)
        if duration > wall + max(1e-6, wall * 1e-9):
            raise EvidenceValidationError(f"{name}/{category} exceeds HE forward")
        _close(row.get("percent_of_he_forward"), 100 * duration / wall, name=f"{name}/{category} percent")
        seconds.append(duration)
    if abs(sum(seconds) - wall) > max(1e-6, wall * 1e-9):
        raise EvidenceValidationError(f"{name} major categories do not close")
    online = _major_category(profile, "online_encode")
    _close(profile.get("online_encode_s"), online["seconds"], name=f"{name} online Encode seconds")
    _close(profile.get("online_encode_pct_of_he_forward"), 100 * online["seconds"] / wall, name=f"{name} online Encode percent")


def summarize_step1(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and extract the matched three-model real-FHE Step-1 matrix."""

    _integer(payload.get("schema_version"), name="Step-1 schema", minimum=1)
    if _list(payload.get("read_errors")):
        raise EvidenceValidationError("Step-1 extraction contains read errors")

    accepted_by_network: dict[str, dict[str, Any]] = {}
    for record_value in _list(payload.get("records")):
        record = _dict(record_value)
        if record.get("accepted_as_step1") is not True:
            continue
        network = str(record.get("network", ""))
        if network in accepted_by_network:
            raise EvidenceValidationError(
                f"Step-1 extraction has duplicate accepted profile for {network}"
            )
        accepted_by_network[network] = record

    missing = sorted(set(EXPECTED_STEP1_MODELS) - set(accepted_by_network))
    extra = sorted(set(accepted_by_network) - set(EXPECTED_STEP1_MODELS))
    if missing or extra:
        raise EvidenceValidationError(
            f"Step-1 model set mismatch; missing={missing}, extra={extra}"
        )
    recorded_count = _integer(
        payload.get("accepted_real_fhe_count"),
        name="accepted_real_fhe_count",
        minimum=0,
    )
    if recorded_count != len(EXPECTED_STEP1_MODELS):
        raise EvidenceValidationError(
            "accepted_real_fhe_count does not equal the required three-model matrix"
        )

    rows: list[dict[str, Any]] = []
    for network, (display_name, expected_mode) in EXPECTED_STEP1_MODELS.items():
        record = accepted_by_network[network]
        if record.get("status") != "ok" or record.get("backend") != "lattigo" or record.get("clear_backend") is not False:
            raise EvidenceValidationError(f"{network} is not successful real-FHE evidence")
        if str(record.get("mode")) != expected_mode:
            raise EvidenceValidationError(
                f"{network} mode is {record.get('mode')!r}, expected {expected_mode!r}"
            )
        profile = _dict(record.get("profile"))
        _validate_wall_profile(profile, name=network)
        if not bool(profile.get("valid", False)):
            raise EvidenceValidationError(f"{network} canonical Step-1 profile is invalid")
        if _integer(profile.get("schema_version"), name=f"{network} profile schema") < 2:
            raise EvidenceValidationError(f"{network} profile schema is older than 2")
        accounting = _dict(profile.get("major_wall_categories_accounting"))
        if not bool(accounting.get("valid", False)):
            raise EvidenceValidationError(f"{network} major-wall accounting is invalid")
        if _dict(profile.get("major_wall_categories_metadata")).get("additive") is not True:
            raise EvidenceValidationError(f"{network} major-wall categories are not additive")
        if _dict(profile.get("operator_microprofile_metadata")).get("additive") is not False:
            raise EvidenceValidationError(
                f"{network} operator microprofile is not marked non-additive"
            )

        he_forward = _finite(
            profile.get("he_forward_s"), name=f"{network} HE forward", minimum=0.0
        )
        if he_forward <= 0.0:
            raise EvidenceValidationError(f"{network} HE-forward time is not positive")
        major = _dict(profile.get("major_wall_categories"))
        if not major:
            raise EvidenceValidationError(f"{network} has no major-wall categories")
        category_sum = sum(
            _finite(_dict(value).get("seconds"), name=f"{network}/{key} seconds", minimum=0.0)
            for key, value in major.items()
        )
        tolerance = _finite(
            accounting.get("tolerance_s", max(1e-6, he_forward * 1e-9)),
            name=f"{network} accounting tolerance",
            minimum=0.0,
        )
        if abs(category_sum - he_forward) > tolerance:
            raise EvidenceValidationError(
                f"{network} independently summed major categories do not close"
            )

        online = _major_category(profile, "online_encode")
        bootstrap = _major_category(profile, "bootstrap")
        mvm = _major_category(profile, "mvm_kernel")
        other = _major_category(profile, "other_he_forward")
        online_s = _finite(
            online.get("seconds"), name=f"{network} online Encode", minimum=0.0
        )
        online_pct = _finite(
            online.get("percent_of_he_forward"),
            name=f"{network} online Encode share",
            minimum=0.0,
        )
        canonical_online_s = _finite(
            profile.get("online_encode_s"),
            name=f"{network} canonical online Encode",
            minimum=0.0,
        )
        canonical_online_pct = _finite(
            profile.get("online_encode_pct_of_he_forward"),
            name=f"{network} canonical online Encode share",
            minimum=0.0,
        )
        if abs(online_s - canonical_online_s) > tolerance or abs(
            online_pct - canonical_online_pct
        ) > 1e-8:
            raise EvidenceValidationError(
                f"{network} canonical online Encode disagrees with major-wall category"
            )

        correctness = _dict(record.get("correctness"))
        if correctness.get("shape_match") is not True:
            raise EvidenceValidationError(f"{network} decrypted output shape did not match")
        stats = _dict(record.get("measured_attempt_statistics"))
        measured = [row for row in _list(record.get("attempts")) if _dict(row).get("kind") == "measured"]
        count = _integer(profile.get("profile_count"), name=f"{network} profile count", minimum=1)
        for key in ("requested_measured_attempt_count", "measured_attempt_count"):
            if _integer(profile.get(key), name=f"{network}/{key}", minimum=1) != count:
                raise EvidenceValidationError(f"{network}/{key} disagrees with collected profile count")
        if len(measured) != count or any(row.get("status") != "ok" for row in measured):
            raise EvidenceValidationError(f"{network} measured attempts do not match profile count")
        for index, attempt in enumerate(measured):
            _validate_wall_profile(_dict(attempt.get("profile")), name=f"{network}/attempt{index}")
            _close(_dict(attempt.get("timing_s")).get("he_forward"), attempt["profile"]["he_forward_s"], name=f"{network}/attempt{index} forward timer")
            if set(_dict(attempt["profile"].get("major_wall_categories"))) != set(major):
                raise EvidenceValidationError(f"{network} attempt category set differs from aggregate")
        for field, mean_key, std_key in (
            ("he_forward_s", "he_forward_mean_s", "he_forward_sample_std_s"),
            ("online_encode_s", "online_encode_mean_s", "online_encode_sample_std_s"),
            ("online_encode_pct_of_he_forward", "online_encode_pct_mean", "online_encode_pct_sample_std"),
        ):
            samples = [row["profile"][field] for row in measured]
            mean = statistics.mean(samples)
            _close(stats.get(mean_key), mean, name=f"{network}/{mean_key}")
            _close(stats.get(std_key), statistics.stdev(samples) if count > 1 else 0.0, name=f"{network}/{std_key}")
            # Aggregate percentage is ratio-of-means, not mean-of-ratios.
            if field != "online_encode_pct_of_he_forward":
                _close(profile.get(field), mean, name=f"{network} aggregate {field}")
        for category, row in major.items():
            mean = statistics.mean(attempt["profile"]["major_wall_categories"][category]["seconds"] for attempt in measured)
            _close(row["seconds"], mean, name=f"{network} aggregate category {category}")
        rows.append(
            {
                "network": network,
                "model": display_name,
                "mode": expected_mode,
                "he_forward_s": he_forward,
                "he_forward_sample_std_s": _finite(
                    stats.get("he_forward_sample_std_s", 0.0),
                    name=f"{network} HE-forward sample std",
                    minimum=0.0,
                ),
                "online_encode_s": online_s,
                "online_encode_sample_std_s": _finite(
                    stats.get("online_encode_sample_std_s", 0.0),
                    name=f"{network} Encode sample std",
                    minimum=0.0,
                ),
                "online_encode_pct": online_pct,
                "online_encode_pct_sample_std": _finite(
                    stats.get("online_encode_pct_sample_std", 0.0),
                    name=f"{network} Encode-share sample std",
                    minimum=0.0,
                ),
                "bootstrap_pct": _finite(
                    bootstrap.get("percent_of_he_forward"),
                    name=f"{network} bootstrap share",
                    minimum=0.0,
                ),
                "mvm_kernel_pct": _finite(
                    mvm.get("percent_of_he_forward"),
                    name=f"{network} MVM share",
                    minimum=0.0,
                ),
                "other_he_forward_pct": _finite(
                    other.get("percent_of_he_forward"),
                    name=f"{network} other share",
                    minimum=0.0,
                ),
                "mae_vs_clear": _finite(
                    correctness.get("mae"), name=f"{network} MAE", minimum=0.0
                ),
                "profile_count": _integer(
                    profile.get("profile_count"),
                    name=f"{network} profile count",
                    minimum=1,
                ),
            }
        )
    return rows


def classify_periodic_candidate(record: Mapping[str, Any]) -> str:
    """Conservatively classify a periodic Orion record by its producer."""

    module_name = str(record.get("module_name", "")).lower()
    operator_type = str(record.get("operator_type", "")).lower()
    transform_id = str(record.get("transform_id", "")).lower()
    combined = " ".join((module_name, operator_type, transform_id))
    if (
        "materialize" in combined
        and any(token in combined for token in ("cat1", "cat2", "cat3", "concat"))
    ):
        return "structural"
    operator_class = operator_type.rsplit(".", 1)[-1]
    learned_operator_classes = {"conv2d", "convtranspose2d", "linear"}
    if operator_class in learned_operator_classes:
        return "learned_weight"
    return "unclassified"


def summarize_periodicity(
    summary: Mapping[str, Any],
    jsonl_path: Path,
    *,
    network: str,
    display_name: str,
    expected_mode: str,
    require_encoded_verification: bool,
) -> tuple[dict[str, Any], str]:
    """Validate a census summary against its JSONL and classify candidates.

    The returned string is the SHA-256 digest accumulated while streaming the
    JSONL, avoiding a second pass over the large census files.
    """

    import hashlib

    schema_version = _integer(
        summary.get("schema_version"),
        name=f"{network} census schema",
        minimum=1,
    )
    if require_encoded_verification and schema_version < 2:
        raise EvidenceValidationError(
            f"{network} encoded-Q/P verification requires census schema 2 or newer"
        )
    if str(summary.get("profile")) != "wpc_orion_slot_periodicity":
        raise EvidenceValidationError(f"{network} has an unexpected census profile")
    occurrence = _dict(summary.get("occurrence"))
    unique = _dict(summary.get("unique_diagonal"))
    observed = _integer(
        occurrence.get("observed_count"), name=f"{network} observed count", minimum=1
    )
    nonzero = _integer(
        occurrence.get("nonzero_count"), name=f"{network} nonzero count", minimum=0
    )
    all_zero = _integer(
        occurrence.get("all_zero_count"), name=f"{network} all-zero count", minimum=0
    )
    periodic = _integer(
        occurrence.get("periodic_count"), name=f"{network} periodic count", minimum=0
    )
    if nonzero + all_zero != observed:
        raise EvidenceValidationError(
            f"{network} nonzero + all-zero count does not equal observed count"
        )
    if _integer(
        summary.get("metadata_complete_occurrence_count"),
        name=f"{network} complete occurrence metadata",
        minimum=0,
    ) != observed:
        raise EvidenceValidationError(f"{network} occurrence metadata is incomplete")
    unique_observed = _integer(
        unique.get("observed_count"), name=f"{network} unique count", minimum=1
    )
    if _integer(
        summary.get("metadata_complete_unique_count"),
        name=f"{network} complete unique metadata",
        minimum=0,
    ) != unique_observed:
        raise EvidenceValidationError(f"{network} unique metadata is incomplete")
    if _integer(
        summary.get("logical_diagonal_payload_change_count"),
        name=f"{network} logical payload changes",
        minimum=0,
    ) != 0:
        raise EvidenceValidationError(f"{network} census contains logical payload changes")

    count_pct = _finite(
        occurrence.get("count_coverage_pct"),
        name=f"{network} count coverage",
        minimum=0.0,
    )
    expected_count_pct = 100.0 * periodic / nonzero if nonzero else 0.0
    if not math.isclose(count_pct, expected_count_pct, rel_tol=1e-10, abs_tol=1e-12):
        raise EvidenceValidationError(f"{network} count coverage is inconsistent")

    candidate_class_counts = {
        "learned_weight": 0,
        "structural": 0,
        "unclassified": 0,
    }
    record_counts = {
        "slot_periodicity_occurrence": 0,
        "slot_periodicity_unique": 0,
    }
    candidate_encoded_passes = 0
    unique_ids: set[str] = set()
    occurrence_ids: set[str] = set()
    logical_sources: dict[str, str] = {}
    raw_totals = {kind: {"nonzero_count": 0, "all_zero_count": 0, "periodic_count": 0,
                        "full_encoded_bytes": 0, "hybrid_storage_bytes": 0,
                        "periodic_full_encoded_bytes": 0}
                  for kind in record_counts}
    digest = hashlib.sha256()
    try:
        handle = jsonl_path.open("rb")
    except OSError as exc:
        raise EvidenceValidationError(f"cannot open {jsonl_path}: {exc}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise EvidenceValidationError(
                    f"{jsonl_path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise EvidenceValidationError(
                    f"{jsonl_path}:{line_number}: record is not an object"
                )
            record_schema = _integer(
                record.get("schema_version"),
                name=f"{jsonl_path}:{line_number} schema",
                minimum=1,
            )
            if record_schema != schema_version:
                raise EvidenceValidationError(
                    f"{jsonl_path}:{line_number}: record/summary schema mismatch"
                )
            kind = str(record.get("record_type", ""))
            if kind not in record_counts:
                raise EvidenceValidationError(
                    f"{jsonl_path}:{line_number}: unexpected record type {kind!r}"
                )
            record_counts[kind] += 1
            if str(record.get("model")) != network:
                raise EvidenceValidationError(
                    f"{jsonl_path}:{line_number}: model metadata mismatch"
                )
            if str(record.get("mode")) != expected_mode:
                raise EvidenceValidationError(
                    f"{jsonl_path}:{line_number}: mode metadata mismatch"
                )
            from orion.experimental.wpc_periodicity import IDENTITY_METADATA_FIELDS
            if record.get("metadata_complete") is not True or any(record.get(key) is None for key in IDENTITY_METADATA_FIELDS):
                raise EvidenceValidationError(f"{jsonl_path}:{line_number}: incomplete identity metadata")
            diagonal_id = _sha256(record.get("diagonal_id"), name="diagonal ID")
            logical_id = _sha256(record.get("logical_diagonal_id"), name="logical diagonal ID")
            source_hash = _sha256(record.get("source_hash"), name="diagonal payload hash")
            if logical_id in logical_sources and logical_sources[logical_id] != source_hash:
                raise EvidenceValidationError(f"{network} JSONL contains logical payload changes")
            logical_sources[logical_id] = source_hash
            if kind == "slot_periodicity_unique":
                if diagonal_id in unique_ids:
                    raise EvidenceValidationError(f"{network} JSONL contains duplicate unique diagonals")
                unique_ids.add(diagonal_id)
            else:
                occurrence_ids.add(diagonal_id)
            slots = _integer(record.get("slots_n"), name="slots", minimum=1)
            period = _integer(record.get("slot_min_period_t"), name="slot period", minimum=1)
            if slots & (slots - 1) or period & (period - 1) or period > slots:
                raise EvidenceValidationError("invalid power-of-two slot period")
            zero = record.get("all_zero")
            candidate = record.get("wpc_slot_candidate")
            if not isinstance(zero, bool) or candidate is not (period < slots and not zero):
                raise EvidenceValidationError("periodic candidate classification is inconsistent")
            totals = raw_totals[kind]
            totals["all_zero_count" if zero else "nonzero_count"] += 1
            if not zero:
                full = _integer(record.get("full_encoded_bytes"), name="diagonal bytes", minimum=1)
                hybrid = _integer(record.get("compressed_bytes"), name="diagonal stored bytes", minimum=1)
                if hybrid != full * period // slots:
                    raise EvidenceValidationError("diagonal compressed bytes are inconsistent with period")
                totals["full_encoded_bytes"] += full
                totals["hybrid_storage_bytes"] += hybrid
                if candidate:
                    totals["periodic_count"] += 1
                    totals["periodic_full_encoded_bytes"] += full
            if kind != "slot_periodicity_occurrence":
                continue
            if candidate:
                classification = classify_periodic_candidate(record)
                candidate_class_counts[classification] += 1
                verification = _dict(record.get("encoded_qp_verification"))
                if verification.get("attempted") is True and verification.get("passed") is True:
                    if verification.get("status_code") != 1 or verification.get("full_polynomial_reconstruction_exact") is not True or verification.get("slot_period_t") != period or verification.get("evaluation_period_2t") != 2 * period:
                        raise EvidenceValidationError(f"{network} encoded-Q/P pass details are inconsistent")
                    candidate_encoded_passes += 1

    if record_counts["slot_periodicity_occurrence"] != observed:
        raise EvidenceValidationError(f"{network} occurrence JSONL count does not match summary")
    if record_counts["slot_periodicity_unique"] != unique_observed:
        raise EvidenceValidationError(f"{network} unique JSONL count does not match summary")
    if unique_ids != occurrence_ids:
        raise EvidenceValidationError(f"{network} unique/occurrence diagonal identities differ")
    classified_total = sum(candidate_class_counts.values())
    if classified_total != periodic:
        raise EvidenceValidationError(f"{network} candidate JSONL count does not match summary")
    for kind, aggregate in (("slot_periodicity_occurrence", occurrence), ("slot_periodicity_unique", unique)):
        totals = raw_totals[kind]
        for field, expected in totals.items():
            if _integer(aggregate.get(field), name=f"{network}/{kind}/{field}", minimum=0) != expected:
                raise EvidenceValidationError(f"{network}/{kind}/{field} disagrees with JSONL")
        full = totals["full_encoded_bytes"]
        _close(aggregate.get("byte_coverage_pct"), 100 * totals["periodic_full_encoded_bytes"] / full if full else 0,
               name=f"{network}/{kind} byte coverage")
        _close(aggregate.get("count_coverage_pct"), 100 * totals["periodic_count"] / totals["nonzero_count"] if totals["nonzero_count"] else 0,
               name=f"{network}/{kind} count coverage")
    if candidate_class_counts["unclassified"]:
        raise EvidenceValidationError(
            f"{network} has {candidate_class_counts['unclassified']} "
            "unclassified periodic candidates"
        )

    encoded = _dict(summary.get("encoded_qp_verification"))
    if require_encoded_verification:
        if not bool(encoded.get("complete", False)):
            raise EvidenceValidationError(f"{network} encoded-Q/P verification is incomplete")
        for field in (
            "candidate_occurrence_count",
            "attempted_occurrence_count",
            "passed_occurrence_count",
        ):
            if _integer(encoded.get(field), name=f"{network} {field}", minimum=0) != periodic:
                raise EvidenceValidationError(
                    f"{network} {field} does not equal periodic candidate count"
                )
        if candidate_encoded_passes != periodic:
            raise EvidenceValidationError(
                f"{network} JSONL does not contain an encoded-Q/P pass for every candidate"
            )
        if _integer(encoded.get("failed_occurrence_count"), name="failed encoded verification", minimum=0) != 0:
            raise EvidenceValidationError(f"{network} encoded-Q/P verification contains failures")

    full_bytes = _integer(
        occurrence.get("full_encoded_bytes"),
        name=f"{network} full encoded bytes",
        minimum=0,
    )
    hybrid_bytes = _integer(
        occurrence.get("hybrid_storage_bytes"),
        name=f"{network} hybrid bytes",
        minimum=0,
    )
    ratio = _finite(
        occurrence.get("partial_storage_compression_ratio"),
        name=f"{network} partial storage ratio",
        minimum=1.0,
    )
    expected_ratio = full_bytes / hybrid_bytes if hybrid_bytes else 1.0
    if not math.isclose(ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
        raise EvidenceValidationError(f"{network} partial storage ratio is inconsistent")

    return (
        {
            "network": network,
            "model": display_name,
            "mode": expected_mode,
            "observed_count": observed,
            "nonzero_count": nonzero,
            "all_zero_count": all_zero,
            "periodic_count": periodic,
            "periodic_learned_weight_count": candidate_class_counts["learned_weight"],
            "periodic_structural_count": candidate_class_counts["structural"],
            "count_coverage_pct": count_pct,
            "byte_coverage_pct": _finite(
                occurrence.get("byte_coverage_pct"),
                name=f"{network} byte coverage",
                minimum=0.0,
            ),
            "full_encoded_bytes": full_bytes,
            "hybrid_storage_bytes": hybrid_bytes,
            "partial_storage_compression_ratio": ratio,
            "encoded_qp_verified_candidate_count": candidate_encoded_passes,
        },
        digest.hexdigest(),
    )


def summarize_accuracy(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize the complete held-out Rotation-Padding result."""

    if _integer(payload.get("schema_version"), name="accuracy schema") != 5:
        raise EvidenceValidationError("full-validation requires schema 5 content-bound checkpoint provenance; rerun eval-only (no retraining)")
    if str(payload.get("status")) != "ok":
        raise EvidenceValidationError("full-validation status is not ok")
    _all_acceptance_gates(payload, name="full validation", required=ACCURACY_GATES)
    identity = _dict(payload.get("evaluated_checkpoint"))
    checkpoint_sha = _sha256(identity.get("sha256"), name="evaluated accuracy checkpoint")
    source_sha = _sha256(_dict(payload.get("source_checkpoint")).get("sha256"), name="accuracy source checkpoint")
    if identity.get("source_checkpoint_sha256") != source_sha or identity.get("padding_semantics") != "wpc_flattened_spatial_rotation_padding" or identity.get("identity_policy") != "sha256_of_exact_bytes_deserialized":
        raise EvidenceValidationError("accuracy checkpoint provenance is inconsistent")
    configuration = _dict(payload.get("configuration"))
    if configuration.get("eval_only") is not True:
        raise EvidenceValidationError("full-validation result was not produced in eval-only mode")
    dataset = _dict(payload.get("dataset"))
    count = _integer(dataset.get("validation_count"), name="validation count", minimum=1)
    metrics = _dict(payload.get("metrics"))

    def metric_row(key: str) -> dict[str, Any]:
        source = _dict(metrics.get(key))
        sample_count = _integer(
            source.get("sample_count"), name=f"{key} sample count", minimum=0
        )
        finite_count = _integer(
            source.get("finite_sample_count"), name=f"{key} finite count", minimum=0
        )
        nonfinite_count = _integer(
            source.get("nonfinite_sample_count"),
            name=f"{key} nonfinite count",
            minimum=0,
        )
        if sample_count != count or finite_count + nonfinite_count != count:
            raise EvidenceValidationError(f"{key} does not account for all validation samples")
        if source.get("fully_finite") is not (nonfinite_count == 0) or source.get("metric_scope") != ("all_samples" if nonfinite_count == 0 else "finite_samples_only"):
            raise EvidenceValidationError(f"{key} finite flag/metric scope disagrees with counts")
        indices = _list(source.get("nonfinite_sample_indices"))
        if len(indices) != nonfinite_count or len(set(indices)) != nonfinite_count or any(_integer(index, name="nonfinite index") >= count for index in indices):
            raise EvidenceValidationError(f"{key} non-finite sample indices disagree with counts")
        for metric in ("dice", "iou"):
            if _finite(source.get(metric), name=f"{key}/{metric}", minimum=0) > 1:
                raise EvidenceValidationError(f"{key}/{metric} is outside [0, 1]")
        return {
            "sample_count": sample_count,
            "finite_sample_count": finite_count,
            "nonfinite_sample_count": nonfinite_count,
            "fully_finite": bool(source.get("fully_finite", nonfinite_count == 0)),
            "dice": _finite(source.get("dice"), name=f"{key} Dice", minimum=0.0),
            "iou": _finite(source.get("iou"), name=f"{key} IoU", minimum=0.0),
            "loss": _finite(source.get("loss"), name=f"{key} loss", minimum=0.0),
        }

    native = metric_row("native_zero_padding_checkpoint")
    before = metric_row("rotation_padding_before_finetune")
    final = metric_row("rotation_padding_best")
    if not native["fully_finite"] or not final["fully_finite"]:
        raise EvidenceValidationError("native or final validation is not fully finite")
    if final["nonfinite_sample_count"] != 0:
        raise EvidenceValidationError("fine-tuned Rotation-Padding model has non-finite samples")
    if before["fully_finite"] and final["dice"] + 1e-12 < before["dice"]:
        raise EvidenceValidationError("final accuracy contradicts not-worse-when-comparable gate")
    training = _dict(payload.get("training"))
    completed = _integer(training.get("completed_epoch"), name="completed epoch", minimum=1)
    best_epoch = _integer(training.get("best_epoch"), name="best epoch", minimum=1)
    if best_epoch > completed or identity.get("epoch") != best_epoch:
        raise EvidenceValidationError("evaluated checkpoint epoch disagrees with training")
    history = _list(training.get("history"))
    if [row.get("epoch") for row in history] != list(range(1, completed + 1)):
        raise EvidenceValidationError("accuracy training history is incomplete")
    manifest = _list(payload.get("checkpoint_manifest"))
    paths = [str(_dict(row).get("path", "")) for row in manifest]
    best_path = str(_dict(payload.get("outputs")).get("best_checkpoint", ""))
    if len(manifest) != completed + 3 or len(set(paths)) != len(paths) or not all(paths) or identity.get("path") != best_path:
        raise EvidenceValidationError("accuracy checkpoint manifest is incomplete/inconsistent")
    for row in manifest:
        _sha256(row.get("sha256"), name="checkpoint manifest digest")
        _integer(row.get("bytes"), name="checkpoint manifest bytes", minimum=1)
    if not any(row.get("path") == best_path and row.get("sha256") == checkpoint_sha for row in manifest):
        raise EvidenceValidationError("accuracy evaluated checkpoint disagrees with manifest")
    train_count = _integer(dataset.get("train_count"), name="training count", minimum=1)
    for row in history:
        audit = _dict(row.get("post_epoch_training_audit"))
        if audit.get("sample_count") != train_count or audit.get("finite_sample_count") != train_count or audit.get("nonfinite_sample_count") != 0 or audit.get("fully_finite") is not True:
            raise EvidenceValidationError("accuracy training audit is incomplete/non-finite")
    return {
        "dataset": str(dataset.get("name")),
        "validation_count": count,
        "native_zero_padding": native,
        "rotation_padding_before_finetune": before,
        "rotation_padding_finetuned": final,
        "native_to_finetuned_dice_delta": final["dice"] - native["dice"],
        "native_to_finetuned_iou_delta": final["iou"] - native["iou"],
        "native_dice_retained_pct": 100.0 * final["dice"] / native["dice"],
        "numerical_stability_recovered": bool(
            before["nonfinite_sample_count"] > 0
            and final["nonfinite_sample_count"] == 0
        ),
        "best_checkpoint_path": str(_dict(payload.get("outputs")).get("best_checkpoint", "")),
        "checkpoint_sha256": checkpoint_sha,
        "source_checkpoint_sha256": source_sha,
    }


def summarize_decoder_correctness(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fine-tuned checkpoint's encrypted decoder correctness gate."""

    if _integer(payload.get("schema_version"), name="decoder correctness schema") != 2:
        raise EvidenceValidationError("decoder-correctness requires schema 2 with recorded tolerance; rerun the correctness gate")
    if str(payload.get("status")) != "ok":
        raise EvidenceValidationError("decoder-correctness status is not ok")
    _all_acceptance_gates(payload, name="decoder correctness", required=DECODER_GATES)
    full_errors = _dict(_dict(payload.get("full_control")).get("errors_vs_clear"))
    compressed_row = _dict(payload.get("compressed"))
    compressed_errors = _dict(compressed_row.get("errors_vs_clear"))
    if not full_errors or not compressed_errors:
        raise EvidenceValidationError("decoder correctness has no per-stage errors")
    max_full = max(
        _finite(value, name=f"full decoder error/{key}", minimum=0.0)
        for key, value in full_errors.items()
    )
    max_compressed = max(
        _finite(value, name=f"compressed decoder error/{key}", minimum=0.0)
        for key, value in compressed_errors.items()
    )
    storage = _dict(payload.get("storage"))
    stats = _dict(payload.get("compressed_global_stats"))
    tolerance = _finite(payload.get("correctness_atol"), name="decoder correctness tolerance", minimum=0)
    if tolerance <= 0 or max(max_full, max_compressed) > tolerance:
        raise EvidenceValidationError("decoder clear-reference errors exceed tolerance")
    if set(full_errors) != set(compressed_errors):
        raise EvidenceValidationError("decoder per-stage error sets differ")
    if set(full_errors) != {"up1", "concat", "dec1a", "activation", "dec1b"}:
        raise EvidenceValidationError("decoder correctness is missing required stages")
    if compressed_row.get("final_max_abs_delta_vs_full_control") != 0:
        raise EvidenceValidationError("compressed decoder output differs from full control")
    if stats.get("registered_transform_count") != 56 or stats.get("total_weight_plaintext_online_encode_calls") != 0 or compressed_row.get("online_python_encode_call_count") != 0:
        raise EvidenceValidationError("decoder transform/online Encode counts are inconsistent")
    if stats.get("current_materialized_full_payload_bytes") != 0 or stats.get("current_materialized_transform_count") != 0 or stats.get("peak_materialized_transform_count") != 1:
        raise EvidenceValidationError("decoder materialization lifecycle is inconsistent")
    if not _dict(payload.get("full_control")).get("operation_counters") or _dict(payload.get("full_control")).get("operation_counters") != compressed_row.get("operation_counters"):
        raise EvidenceValidationError("decoder operation counters differ")
    full_bytes = _integer(storage.get("full_weight_plus_bias_payload_bytes"), name="decoder full weight/bias bytes", minimum=1)
    stored_bytes = _integer(storage.get("stored_weight_plus_metadata_plus_bias_bytes"), name="decoder stored bytes", minimum=1)
    concat_bytes = _integer(storage.get("concat_full_qp_payload_bytes"), name="decoder concat bytes", minimum=1)
    _close(storage.get("overall_full_to_stored_including_concat_ratio"),
           (full_bytes + concat_bytes) / (stored_bytes + concat_bytes), name="decoder overall storage ratio")
    return {
        "checkpoint_sha256": _sha256(_dict(payload.get("checkpoint")).get("sha256"), name="decoder checkpoint"),
        "max_full_qp_error_vs_clear": max_full,
        "max_compressed_qp_error_vs_clear": max_compressed,
        "compressed_full_final_max_abs_delta": _finite(
            compressed_row.get("final_max_abs_delta_vs_full_control"),
            name="compressed/full decoder delta",
            minimum=0.0,
        ),
        "compressed_transform_count": _integer(
            stats.get("registered_transform_count"),
            name="compressed decoder transform count",
            minimum=1,
        ),
        "online_weight_encode_calls": _integer(
            stats.get("total_weight_plaintext_online_encode_calls"),
            name="decoder online weight Encode calls",
            minimum=0,
        ),
        "overall_storage_ratio_including_concat": _finite(
            storage.get("overall_full_to_stored_including_concat_ratio"),
            name="decoder correctness storage ratio",
            minimum=1.0,
        ),
    }


def summarize_decoder_benchmark(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize the isolated full/compressed decoder benchmark."""

    if _integer(payload.get("schema_version"), name="decoder benchmark schema") != 1:
        raise EvidenceValidationError("decoder benchmark schema is not version 1")
    if str(payload.get("status")) != "ok":
        raise EvidenceValidationError("decoder benchmark status is not ok")
    _all_acceptance_gates(payload, name="decoder benchmark")
    comparison = _dict(payload.get("comparison"))
    if not comparison:
        raise EvidenceValidationError("decoder benchmark comparison is missing")
    latency = _dict(comparison.get("latency"))
    memory = _dict(comparison.get("memory"))
    correctness = _dict(comparison.get("correctness"))
    operations = _dict(comparison.get("operations"))
    workers = _dict(payload.get("workers"))
    try:
        recomputed = compare_trained_decoder_workers(
            _dict(workers.get("full")), _dict(workers.get("compressed")),
            rss_samples=memory["rss_sampling"], atol=correctness["atol"],
        )
    except (KeyError, TypeError) as exc:
        raise EvidenceValidationError(f"decoder benchmark raw evidence is missing: {exc}") from exc
    if recomputed["acceptance"]["valid"] is not True:
        raise EvidenceValidationError("independent decoder benchmark validation failed")
    required = set(recomputed["acceptance"]) - {"raw_measurements_and_outputs_validated"}
    _all_acceptance_gates(payload, name="decoder benchmark", required=required)
    _all_acceptance_gates(comparison, name="decoder comparison", required=required)
    for mode in ("full", "compressed"):
        if workers[mode]["checkpoint"]["sha256"] != _dict(payload.get("checkpoint")).get("sha256"):
            raise EvidenceValidationError("decoder benchmark worker/checkpoint hashes differ")
    for key, value in recomputed["latency"].items():
        if isinstance(value, dict):
            recorded = _dict(latency.get(key))
            for statistic, expected in value.items():
                _close(recorded.get(statistic), expected, name=f"benchmark/{key}/{statistic}")
        else:
            _close(latency.get(key), value, name=f"benchmark/{key}")
    for key, value in recomputed["memory"].items():
        if isinstance(value, (int, float)):
            _close(memory.get(key), value, name=f"benchmark/memory/{key}", atol=0)
    for key, value in recomputed["correctness"].items():
        _close(correctness.get(key), value, name=f"benchmark/correctness/{key}")
    if operations != recomputed["operations"]:
        raise EvidenceValidationError("reported benchmark operation counters disagree with workers")
    ratio = _finite(
        latency.get("compressed_over_full_median_ratio"),
        name="compressed/full median latency ratio",
        minimum=0.0,
    )
    full_logical = _integer(
        memory.get("logical_full_resident_bytes"), name="full logical bytes", minimum=1
    )
    compressed_logical = _integer(
        memory.get("logical_compressed_resident_bytes"),
        name="compressed logical bytes",
        minimum=1,
    )
    full_pre = _integer(
        memory.get("full_pre_online_rss_bytes"), name="full pre-online RSS", minimum=1
    )
    compressed_pre = _integer(
        memory.get("compressed_pre_online_rss_bytes"),
        name="compressed pre-online RSS",
        minimum=1,
    )
    full_peak = _integer(
        memory.get("full_online_peak_rss_bytes"), name="full peak RSS", minimum=1
    )
    compressed_peak = _integer(
        memory.get("compressed_online_peak_rss_bytes"),
        name="compressed peak RSS",
        minimum=1,
    )
    result = {
        "checkpoint_sha256": str(_dict(payload.get("checkpoint")).get("sha256", "")),
        "forward_runs": _integer(
            _dict(
                _dict(_dict(payload.get("workers")).get("full")).get(
                    "experiment"
                )
            ).get("forward_runs"),
            name="decoder forward runs",
            minimum=1,
        ),
        "full_forward_median_s": _finite(
            _dict(latency.get("full_forward_s")).get("median"),
            name="full median forward",
            minimum=0.0,
        ),
        "compressed_forward_median_s": _finite(
            _dict(latency.get("compressed_forward_s")).get("median"),
            name="compressed median forward",
            minimum=0.0,
        ),
        "compressed_over_full_median_ratio": ratio,
        "latency_overhead_pct": 100.0 * (ratio - 1.0),
        "decompression_median_s": _finite(
            _dict(latency.get("compressed_decompression_s")).get("median"),
            name="median decompression",
            minimum=0.0,
        ),
        "decompression_median_pct_of_forward": _finite(
            _dict(latency.get("compressed_decompression_pct_of_forward")).get("median"),
            name="median decompression share",
            minimum=0.0,
        ),
        "bootstrap_median_s": _finite(
            _dict(latency.get("compressed_bootstrap_s")).get("median"),
            name="median bootstrap",
            minimum=0.0,
        ),
        "activation_plus_bootstrap_median_pct_of_forward": _finite(
            _dict(latency.get("compressed_activation_plus_bootstrap_pct_of_forward")).get("median"),
            name="median activation plus bootstrap share",
            minimum=0.0,
        ),
        "logical_full_resident_bytes": full_logical,
        "logical_compressed_resident_bytes": compressed_logical,
        "logical_storage_compression_ratio": _finite(
            memory.get("logical_storage_compression_ratio"),
            name="logical storage compression ratio",
            minimum=1.0,
        ),
        "logical_storage_reduction_pct": 100.0 * (1.0 - compressed_logical / full_logical),
        "full_pre_online_rss_bytes": full_pre,
        "compressed_pre_online_rss_bytes": compressed_pre,
        "pre_online_rss_reduction_pct": 100.0 * (1.0 - compressed_pre / full_pre),
        "full_online_peak_rss_bytes": full_peak,
        "compressed_online_peak_rss_bytes": compressed_peak,
        "online_peak_rss_reduction_pct": 100.0 * (1.0 - compressed_peak / full_peak),
        "full_go_heap_inuse_after_compile_gc_bytes": _integer(
            memory.get("full_go_heap_inuse_after_compile_gc_bytes"),
            name="full Go heap",
            minimum=1,
        ),
        "compressed_go_heap_inuse_after_compile_gc_bytes": _integer(
            memory.get("compressed_go_heap_inuse_after_compile_gc_bytes"),
            name="compressed Go heap",
            minimum=1,
        ),
        "max_abs_delta_between_isolated_outputs": _finite(
            correctness.get("max_abs_delta_between_isolated_outputs"),
            name="isolated output delta",
            minimum=0.0,
        ),
        "operation_counters_match": operations.get("match") is True,
    }
    if not result["operation_counters_match"]:
        raise EvidenceValidationError("decoder benchmark operation counters do not match")
    return result


def build_synthesis(
    *,
    step1: Mapping[str, Any],
    periodicity_rows: Iterable[dict[str, Any]],
    accuracy: Mapping[str, Any],
    decoder_correctness: Mapping[str, Any],
    decoder_benchmark: Mapping[str, Any],
    checkpoint_sha256: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Join already-validated evidence and enforce cross-artifact consistency."""

    profiles = summarize_step1(step1)
    censuses = list(periodicity_rows)
    if [row["network"] for row in censuses] != list(EXPECTED_STEP1_MODELS):
        raise EvidenceValidationError("periodicity rows are not the required ordered model set")
    accuracy_row = summarize_accuracy(accuracy)
    correctness_row = summarize_decoder_correctness(decoder_correctness)
    benchmark_row = summarize_decoder_benchmark(decoder_benchmark)
    expected_hashes = {
        _sha256(checkpoint_sha256, name="supplied checkpoint"),
        str(accuracy_row["checkpoint_sha256"]),
        str(correctness_row["checkpoint_sha256"]),
        str(benchmark_row["checkpoint_sha256"]),
    }
    if "" in expected_hashes or len(expected_hashes) != 1:
        raise EvidenceValidationError(
            "checkpoint file, accuracy, decoder correctness, and isolated benchmark hashes differ"
        )
    # A portable path is informational; content identity decides compatibility.

    learned_candidates = sum(
        int(row["periodic_learned_weight_count"]) for row in censuses
    )
    structural_candidates = sum(
        int(row["periodic_structural_count"]) for row in censuses
    )
    encode_rank = [
        row["network"]
        for row in sorted(profiles, key=lambda row: row["online_encode_pct"], reverse=True)
    ]
    selective_status = (
        "not_supported"
        if learned_candidates == 0
        else "candidate_support_detected_impact_not_measured"
    )
    return {
        "schema_version": 2,
        "profile": "wpc_orion_tradeoff_evidence_synthesis",
        "status": "ok",
        "model_profiles": profiles,
        "orion_layout_periodicity": censuses,
        "rotation_padding_accuracy": accuracy_row,
        "trained_decoder_fhe_correctness": correctness_row,
        "trained_decoder_isolated_benchmark": benchmark_row,
        "cross_artifact_checkpoint_sha256": str(checkpoint_sha256),
        "acceptance": {
            "step1_three_model_matrix_valid": True,
            "periodicity_censuses_and_jsonl_valid": True,
            "full_validation_valid": True,
            "trained_decoder_fhe_correctness_valid": True,
            "isolated_decoder_benchmark_valid": True,
            "checkpoint_identity_consistent": True,
            "periodic_candidates_fully_classified": True,
            "valid": True,
        },
        "derived": {
            "models_by_descending_online_encode_share": encode_rank,
            "periodic_learned_weight_candidate_count_across_models": learned_candidates,
            "periodic_structural_candidate_count_across_models": structural_candidates,
            "selective_existing_orion_layout_hypothesis": selective_status,
            "wpc_layout_trained_decoder_mechanism": "supported",
            "complete_encrypted_network_tradeoff": "not_established",
        },
        "claim_scope": {
            "step1": "whole-model Orion real-FHE forward profiles",
            "periodicity": "unchanged Orion layouts; structural clear-Lattigo audit",
            "accuracy": (
                "complete clear U-Net validation on "
                f"{accuracy_row['validation_count']:,} validation-split samples (not an untouched test set)"
            ),
            "wpc_resource_benchmark": (
                "fine-tuned encrypted U-Net decoder stage with synthetic "
                "internal features"
            ),
            "not_measured": "complete encrypted WPC U-Net or ResNet latency and memory",
        },
    }
