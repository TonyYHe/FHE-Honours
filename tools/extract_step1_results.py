from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    REPO_ROOT / ".tmp" / "results" / "honours" / "04_step1_model_matrix",
    REPO_ROOT / ".tmp" / "results" / "honours" / "05_vgg16_profile_fix",
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / ".tmp" / "results" / "honours" / "06_step1_extracted"
)

MAJOR_CATEGORY_ORDER = (
    "online_encode",
    "bootstrap",
    "mvm_kernel",
    "other_he_forward",
    "linear_wrapper_postprocess",
    "provider_executor_overhead",
    "runtime_load_trim",
    "layer_cache_key_prepare",
    "layer_cache_evict",
    "layer_cache_other",
)
MICROPROFILE_ORDER = (
    "bootstrap",
    "lt_fused_multiply_accumulate",
    "lt_rotation",
    "explicit_accumulate",
    "elementwise_add_module_wall",
    "elementwise_multiply_module_wall",
)
OPERATION_COUNT_ORDER = (
    "transform_count",
    "diag_terms",
    "q_mul",
    "qp_mul",
    "baby_rotation_count",
    "giant_rotation_count",
    "inner_reduce_count",
    "outer_reduce_count",
    "final_moddown",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> Optional[int]:
    number = _number(value)
    return int(number) if number is not None else None


def _mean_and_sample_std(values: Iterable[Any]) -> tuple[Optional[float], Optional[float]]:
    clean = [number for value in values if (number := _number(value)) is not None]
    if not clean:
        return None, None
    return statistics.mean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0


def _profile_values(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _integer(profile.get("schema_version")),
        "valid": bool(profile.get("valid", False)),
        "validation_errors": [str(value) for value in _as_list(profile.get("validation_errors"))],
        "runtime_fairness_mode": profile.get("runtime_fairness_mode"),
        "measurement_scope": profile.get("measurement_scope"),
        "denominator": profile.get("denominator"),
        "profile_count": _integer(profile.get("profile_count")),
        "measured_attempt_count": _integer(profile.get("measured_attempt_count")),
        "requested_measured_attempt_count": _integer(
            profile.get("requested_measured_attempt_count")
        ),
        "he_forward_s": _number(profile.get("he_forward_s")),
        "online_encode_s": _number(profile.get("online_encode_s")),
        "online_encode_pct_of_he_forward": _number(
            profile.get("online_encode_pct_of_he_forward")
        ),
        "major_wall_categories": _as_dict(profile.get("major_wall_categories")),
        "major_wall_categories_metadata": _as_dict(
            profile.get("major_wall_categories_metadata")
        ),
        "major_wall_categories_accounting": _as_dict(
            profile.get("major_wall_categories_accounting")
        ),
        "operator_microprofile": _as_dict(profile.get("operator_microprofile")),
        "operator_microprofile_metadata": _as_dict(
            profile.get("operator_microprofile_metadata")
        ),
        "operation_counts_mean_per_forward": _as_dict(
            profile.get("operation_counts_mean_per_forward")
            or profile.get("operation_counts")
        ),
    }


def _clear_backend(payload: dict[str, Any], profile: dict[str, Any]) -> bool:
    env_value = _as_dict(payload.get("lattigo_runtime_env")).get(
        "ORION_LATTIGO_CLEAR_BACKEND"
    )
    if env_value is not None:
        return str(env_value).strip().lower() in {"1", "true", "yes", "on"}
    errors = " ".join(str(value) for value in _as_list(profile.get("validation_errors")))
    return "CLEAR_BACKEND" in errors or "clear backend" in errors.lower()


def _acceptance_errors(
    payload: dict[str, Any], profile: dict[str, Any], *, clear_backend: bool
) -> list[str]:
    errors: list[str] = []
    runtime_env = _as_dict(payload.get("lattigo_runtime_env"))
    if str(payload.get("status", "")) != "ok":
        errors.append(f"runner status is {payload.get('status')!r}, not 'ok'")
    if str(payload.get("backend", "")) != "lattigo":
        errors.append(f"backend is {payload.get('backend')!r}, not 'lattigo'")
    if clear_backend:
        errors.append("ORION_LATTIGO_CLEAR_BACKEND is enabled")
    elif str(runtime_env.get("ORION_LATTIGO_CLEAR_BACKEND", "")) != "0":
        errors.append("ORION_LATTIGO_CLEAR_BACKEND=0 is not recorded")
    if str(runtime_env.get("ORION_SINGLE_SLOT_LAYER_CACHE", "")) != "1":
        errors.append("ORION_SINGLE_SLOT_LAYER_CACHE=1 is not recorded")
    if str(runtime_env.get("ORION_LATTIGO_STREAMING_LT", "")) != "0":
        errors.append("ORION_LATTIGO_STREAMING_LT=0 is not recorded")
    if str(runtime_env.get("ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT", "")) != "0":
        errors.append("ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT=0 is not recorded")
    if not profile:
        errors.append("canonical step1_online_encode_profile is missing")
        return errors
    if not bool(profile.get("valid", False)):
        errors.append("canonical Step 1 profile is marked invalid")
    if int(profile.get("schema_version", 0) or 0) < 2:
        errors.append("profile schema is older than version 2")
    if profile.get("runtime_fairness_mode") != "single_slot_layer_cache":
        errors.append("runtime_fairness_mode is not single_slot_layer_cache")
    accounting = _as_dict(profile.get("major_wall_categories_accounting"))
    if not bool(accounting.get("valid", False)):
        errors.append("additive major wall-category accounting is invalid")
    if not bool(_as_dict(profile.get("major_wall_categories_metadata")).get("additive", False)):
        errors.append("major wall categories are not declared additive")
    if _as_dict(profile.get("operator_microprofile_metadata")).get("additive") is not False:
        errors.append("operator microprofile is not declared non-additive")
    he_forward_s = _number(profile.get("he_forward_s"))
    major = _as_dict(profile.get("major_wall_categories"))
    missing_categories = [category for category in MAJOR_CATEGORY_ORDER if category not in major]
    if missing_categories:
        errors.append(
            "major wall categories are missing: " + ", ".join(missing_categories)
        )
    category_seconds = [
        _category_value(major, category, "seconds") for category in major
    ]
    if not major or any(value is None or value < 0.0 for value in category_seconds):
        errors.append("major wall categories contain a missing, non-finite, or negative value")
    elif he_forward_s is not None:
        recomputed_sum = sum(value for value in category_seconds if value is not None)
        tolerance = _number(accounting.get("tolerance_s"))
        if tolerance is None:
            tolerance = max(1e-6, abs(he_forward_s) * 1e-9)
        if abs(recomputed_sum - he_forward_s) > tolerance:
            errors.append("independently recomputed major categories do not close to HE-forward wall time")
        percentage_mismatches: list[str] = []
        for category in major:
            seconds = _category_value(major, category, "seconds")
            recorded_pct = _category_value(
                major, category, "percent_of_he_forward"
            )
            if seconds is None or recorded_pct is None or recorded_pct < 0.0:
                percentage_mismatches.append(category)
                continue
            expected_pct = seconds / he_forward_s * 100.0 if he_forward_s > 0.0 else 0.0
            pct_tolerance = max(1e-8, abs(expected_pct) * 1e-9)
            if abs(recorded_pct - expected_pct) > pct_tolerance:
                percentage_mismatches.append(category)
        if percentage_mismatches:
            errors.append(
                "major category percentages disagree with seconds/HE-forward for: "
                + ", ".join(percentage_mismatches)
            )
        online_entry_s = _category_value(major, "online_encode", "seconds")
        online_entry_pct = _category_value(
            major, "online_encode", "percent_of_he_forward"
        )
        canonical_online_s = _number(profile.get("online_encode_s"))
        canonical_online_pct = _number(profile.get("online_encode_pct_of_he_forward"))
        if (
            online_entry_s is None
            or canonical_online_s is None
            or abs(online_entry_s - canonical_online_s) > tolerance
            or online_entry_pct is None
            or canonical_online_pct is None
            or abs(online_entry_pct - canonical_online_pct) > 1e-8
        ):
            errors.append("canonical online Encode values disagree with the major category")
    profile_count = _integer(profile.get("profile_count"))
    measured_count = _integer(profile.get("measured_attempt_count"))
    requested_count = _integer(profile.get("requested_measured_attempt_count"))
    if profile_count is None or measured_count is None or requested_count is None:
        errors.append("measured/profile counts are incomplete")
    elif not (profile_count == measured_count == requested_count and requested_count > 0):
        errors.append(
            "requested, successful measured-attempt, and profile counts do not agree"
        )
    elif _integer(payload.get("forward_runs")) != requested_count:
        errors.append("canonical requested count does not match the runner's forward_runs")
    if (_number(profile.get("he_forward_s")) or 0.0) <= 0.0:
        errors.append("HE-forward wall time is not positive")
    if (_number(profile.get("online_encode_s")) or 0.0) <= 0.0:
        errors.append("online Encode time is not positive")
    if not bool(_as_dict(payload.get("mae_vs_clear")).get("shape_match", False)):
        errors.append("decrypted and clear output shapes do not match")
    return errors


def _error_summary(payload: dict[str, Any]) -> dict[str, Any]:
    traceback_text = str(payload.get("traceback", "") or "")
    traceback_lines = [line.strip() for line in traceback_text.splitlines() if line.strip()]
    return {
        "error_type": payload.get("error_type"),
        "error": payload.get("error"),
        "traceback_last_line": traceback_lines[-1] if traceback_lines else None,
    }


def extract_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    raw_profile = _as_dict(payload.get("step1_online_encode_profile"))
    profile = _profile_values(raw_profile)
    clear_backend = _clear_backend(payload, raw_profile)
    acceptance_errors = _acceptance_errors(
        payload, raw_profile, clear_backend=clear_backend
    )
    accepted = not acceptance_errors
    if str(payload.get("status", "")) != "ok":
        result_kind = "failed_run"
    elif clear_backend:
        result_kind = "clear_structural"
    elif accepted:
        result_kind = "accepted_real_fhe"
    else:
        result_kind = "invalid_profile"

    attempts: list[dict[str, Any]] = []
    for attempt in _as_list(payload.get("forward_attempts")):
        attempt_dict = _as_dict(attempt)
        attempt_profile = _profile_values(
            _as_dict(attempt_dict.get("step1_online_encode_profile"))
        )
        attempts.append(
            {
                "attempt_index": _integer(attempt_dict.get("attempt_index")),
                "kind": attempt_dict.get("kind"),
                "status": attempt_dict.get("status"),
                "timing_s": _as_dict(attempt_dict.get("timing_s")),
                "profile": attempt_profile,
                **_error_summary(attempt_dict),
            }
        )

    measured = [
        attempt
        for attempt in attempts
        if attempt.get("kind") == "measured" and attempt.get("status") == "ok"
    ]
    he_mean, he_std = _mean_and_sample_std(
        attempt["profile"].get("he_forward_s") for attempt in measured
    )
    encode_mean, encode_std = _mean_and_sample_std(
        attempt["profile"].get("online_encode_s") for attempt in measured
    )
    pct_mean, pct_std = _mean_and_sample_std(
        attempt["profile"].get("online_encode_pct_of_he_forward")
        for attempt in measured
    )

    runtime_env = _as_dict(payload.get("lattigo_runtime_env"))
    selected_env_names = (
        "ORION_LATTIGO_CLEAR_BACKEND",
        "ORION_SINGLE_SLOT_LAYER_CACHE",
        "ORION_LATTIGO_STREAMING_LT",
        "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT",
        "ORION_SINGLE_SLOT_ENCODE_WORKERS",
    )
    return {
        "source": str(path),
        "result_kind": result_kind,
        "accepted_as_step1": accepted,
        "acceptance_errors": acceptance_errors,
        "status": payload.get("status"),
        "network": payload.get("network"),
        "model": payload.get("model"),
        "label": payload.get("label") or payload.get("model") or payload.get("network"),
        "mode": payload.get("mode"),
        "provider_mode": payload.get("provider_mode"),
        "backend": payload.get("backend"),
        "dataset": payload.get("dataset"),
        "clear_backend": clear_backend,
        "environment": {name: runtime_env.get(name) for name in selected_env_names},
        "compile_s": _number(_as_dict(payload.get("timing_s")).get("compile")),
        "correctness": _as_dict(payload.get("mae_vs_clear")),
        "profile": profile,
        "measured_attempt_statistics": {
            "he_forward_mean_s": he_mean,
            "he_forward_sample_std_s": he_std,
            "online_encode_mean_s": encode_mean,
            "online_encode_sample_std_s": encode_std,
            "online_encode_pct_mean": pct_mean,
            "online_encode_pct_sample_std": pct_std,
        },
        "attempts": attempts,
        **_error_summary(payload),
    }


def _insert_honours_fallback(path: Path) -> Optional[Path]:
    parts = list(path.parts)
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == [".tmp", "results"]:
            if index + 2 < len(parts) and parts[index + 2] != "honours":
                candidate = Path(*parts[: index + 2], "honours", *parts[index + 2 :])
                return candidate
    return None


def resolve_input_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    if candidate.exists():
        return candidate
    fallback = _insert_honours_fallback(candidate)
    if fallback is not None and fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"input path does not exist: {path}")


def _is_primary_result(path: Path) -> bool:
    name = path.name
    return (
        path.suffix == ".json"
        and ".progress_state." not in name
        and not (".forward" in name and ".progress" in name)
        and name not in {"summary.json", "extracted_results.json"}
    )


def discover_result_files(inputs: Sequence[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw_path in inputs:
        path = resolve_input_path(raw_path)
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in seen and _is_primary_result(candidate):
                discovered.append(candidate)
                seen.add(candidate)
    return discovered


def _load_records(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    read_errors: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON value is not an object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            read_errors.append(f"{path}: {exc}")
            continue
        records.append(extract_record(path, payload))
    return records, read_errors


def _category_value(profile: dict[str, Any], category: str, field: str) -> Optional[float]:
    return _number(_as_dict(_as_dict(profile).get(category)).get(field))


def _run_row(record: dict[str, Any]) -> dict[str, Any]:
    profile = _as_dict(record.get("profile"))
    accounting = _as_dict(profile.get("major_wall_categories_accounting"))
    correctness = _as_dict(record.get("correctness"))
    stats = _as_dict(record.get("measured_attempt_statistics"))
    row: dict[str, Any] = {
        "source": record.get("source"),
        "result_kind": record.get("result_kind"),
        "accepted_as_step1": record.get("accepted_as_step1"),
        "acceptance_errors": " | ".join(record.get("acceptance_errors", [])),
        "status": record.get("status"),
        "network": record.get("network"),
        "model": record.get("model"),
        "label": record.get("label"),
        "mode": record.get("mode"),
        "backend": record.get("backend"),
        "clear_backend": record.get("clear_backend"),
        "schema_version": profile.get("schema_version"),
        "profile_valid": profile.get("valid"),
        "runtime_fairness_mode": profile.get("runtime_fairness_mode"),
        "profile_count": profile.get("profile_count"),
        "measured_attempt_count": profile.get("measured_attempt_count"),
        "requested_measured_attempt_count": profile.get(
            "requested_measured_attempt_count"
        ),
        "compile_s": record.get("compile_s"),
        "he_forward_s": profile.get("he_forward_s"),
        "he_forward_sample_std_s": stats.get("he_forward_sample_std_s"),
        "online_encode_s": profile.get("online_encode_s"),
        "online_encode_sample_std_s": stats.get("online_encode_sample_std_s"),
        "online_encode_pct_of_he_forward": profile.get(
            "online_encode_pct_of_he_forward"
        ),
        "online_encode_pct_sample_std": stats.get(
            "online_encode_pct_sample_std"
        ),
        "major_accounting_valid": accounting.get("valid"),
        "major_category_sum_s": accounting.get("category_sum_s"),
        "major_closure_error_s": accounting.get("closure_error_s"),
        "major_accounting_tolerance_s": accounting.get("tolerance_s"),
        "shape_match": correctness.get("shape_match"),
        "mae": correctness.get("mae"),
        "max_abs": correctness.get("max_abs"),
        "rmse": correctness.get("rmse"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
    }
    major = _as_dict(profile.get("major_wall_categories"))
    for category in MAJOR_CATEGORY_ORDER:
        row[f"major_{category}_s"] = _category_value(major, category, "seconds")
        row[f"major_{category}_pct"] = _category_value(
            major, category, "percent_of_he_forward"
        )
    micro = _as_dict(profile.get("operator_microprofile"))
    for category in MICROPROFILE_ORDER:
        row[f"micro_{category}_s"] = _category_value(micro, category, "seconds")
        row[f"micro_{category}_pct"] = _category_value(
            micro, category, "percent_of_he_forward"
        )
    counts = _as_dict(profile.get("operation_counts_mean_per_forward"))
    for operation in OPERATION_COUNT_ORDER:
        row[f"ops_{operation}"] = _number(counts.get(operation))
    return row


def _attempt_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for attempt in record.get("attempts", []):
            profile = _as_dict(attempt.get("profile"))
            timing = _as_dict(attempt.get("timing_s"))
            row: dict[str, Any] = {
                "source": record.get("source"),
                "label": record.get("label"),
                "network": record.get("network"),
                "mode": record.get("mode"),
                "parent_result_kind": record.get("result_kind"),
                "parent_accepted_as_step1": record.get("accepted_as_step1"),
                "attempt_index": attempt.get("attempt_index"),
                "kind": attempt.get("kind"),
                "status": attempt.get("status"),
                "profile_valid": profile.get("valid"),
                "encrypt_s": _number(timing.get("encrypt")),
                "he_forward_s": profile.get("he_forward_s")
                or _number(timing.get("he_forward")),
                "decrypt_decode_s": _number(timing.get("decrypt_decode")),
                "online_encode_s": profile.get("online_encode_s"),
                "online_encode_pct_of_he_forward": profile.get(
                    "online_encode_pct_of_he_forward"
                ),
                "error_type": attempt.get("error_type"),
                "error": attempt.get("error"),
            }
            major = _as_dict(profile.get("major_wall_categories"))
            for category in MAJOR_CATEGORY_ORDER:
                row[f"major_{category}_s"] = _category_value(
                    major, category, "seconds"
                )
                row[f"major_{category}_pct"] = _category_value(
                    major, category, "percent_of_he_forward"
                )
            rows.append(row)
    return rows


def _long_rows(
    records: Sequence[dict[str, Any]], profile_field: str, order: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        profile = _as_dict(record.get("profile"))
        values = _as_dict(profile.get(profile_field))
        keys = list(order) + sorted(set(values) - set(order))
        for key in keys:
            value = _as_dict(values.get(key))
            if not value:
                continue
            rows.append(
                {
                    "source": record.get("source"),
                    "label": record.get("label"),
                    "network": record.get("network"),
                    "mode": record.get("mode"),
                    "result_kind": record.get("result_kind"),
                    "accepted_as_step1": record.get("accepted_as_step1"),
                    "category": key,
                    "seconds": _number(value.get("seconds")),
                    "percent_of_he_forward": _number(
                        value.get("percent_of_he_forward")
                    ),
                    "additive": profile_field == "major_wall_categories",
                }
            )
    return rows


def _operation_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        counts = _as_dict(
            _as_dict(record.get("profile")).get("operation_counts_mean_per_forward")
        )
        keys = list(OPERATION_COUNT_ORDER) + sorted(
            set(counts) - set(OPERATION_COUNT_ORDER)
        )
        for operation in keys:
            if operation not in counts:
                continue
            rows.append(
                {
                    "source": record.get("source"),
                    "label": record.get("label"),
                    "network": record.get("network"),
                    "mode": record.get("mode"),
                    "result_kind": record.get("result_kind"),
                    "accepted_as_step1": record.get("accepted_as_step1"),
                    "operation": operation,
                    "mean_count_per_forward": _number(counts.get(operation)),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _md_cell(value: Any) -> str:
    return str(value if value is not None else "n/a").replace("|", "\\|").replace("\n", " ")


def build_markdown_report(
    records: Sequence[dict[str, Any]], read_errors: Sequence[str]
) -> str:
    accepted = [record for record in records if record.get("accepted_as_step1")]
    excluded = [record for record in records if not record.get("accepted_as_step1")]
    lines = [
        "# Step 1 profiling extraction",
        "",
        f"Accepted real-FHE profiles: **{len(accepted)}** of **{len(records)}** readable result files.",
        "",
        "## Accepted real-FHE results",
        "",
    ]
    if accepted:
        lines.extend(
            [
                "| Model | Mode | HE forward (s) | Online Encode (s) | Encode share | Bootstrap share | MVM share | Other share | MAE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for record in accepted:
            profile = _as_dict(record.get("profile"))
            major = _as_dict(profile.get("major_wall_categories"))
            correctness = _as_dict(record.get("correctness"))
            lines.append(
                "| "
                + " | ".join(
                    (
                        _md_cell(record.get("label")),
                        _md_cell(record.get("mode")),
                        _fmt(profile.get("he_forward_s")),
                        _fmt(profile.get("online_encode_s")),
                        f"{_fmt(profile.get('online_encode_pct_of_he_forward'))}%",
                        f"{_fmt(_category_value(major, 'bootstrap', 'percent_of_he_forward'))}%",
                        f"{_fmt(_category_value(major, 'mvm_kernel', 'percent_of_he_forward'))}%",
                        f"{_fmt(_category_value(major, 'other_he_forward', 'percent_of_he_forward'))}%",
                        _fmt(correctness.get("mae"), 8),
                    )
                )
                + " |"
            )
    else:
        lines.append("No result passed the Step 1 acceptance gate.")

    lines.extend(["", "## Excluded or diagnostic results", ""])
    if excluded:
        lines.extend(
            [
                "| Result | Classification | Reason |",
                "|---|---|---|",
            ]
        )
        for record in excluded:
            lines.append(
                "| "
                + " | ".join(
                    (
                        _md_cell(record.get("label") or Path(str(record.get("source"))).name),
                        _md_cell(record.get("result_kind")),
                        _md_cell("; ".join(record.get("acceptance_errors", []))),
                    )
                )
                + " |"
            )
    else:
        lines.append("No readable result was excluded.")

    if read_errors:
        lines.extend(["", "## Unreadable inputs", ""])
        lines.extend(f"- {_md_cell(error)}" for error in read_errors)

    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Comparative plots contain only accepted real-FHE profiles. Clear-Lattigo timing is structural evidence and is not comparable to encrypted execution.",
            "- `major_wall_categories` are additive, mutually exclusive wall-time categories. For an accepted schema-v2 profile their percentages close to 100% within the recorded tolerance.",
            "- `operator_microprofile` values are diagnostic and non-additive. They may overlap or represent parallel/nested work, so do not sum them or use them as a wall-time pie chart.",
            "- Operation counts are mean backend counts per measured forward, not seconds and not percentages.",
            "- Error bars in the Encode comparison are sample standard deviations across measured forwards. Warmups are excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_label(record: dict[str, Any]) -> str:
    return f"{record.get('model') or record.get('network')}\n({record.get('mode')})"


def write_plots(records: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    accepted = [record for record in records if record.get("accepted_as_step1")]
    if not accepted:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is unavailable; install project dependencies or pass --no-plots"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [_plot_label(record) for record in accepted]
    x = list(range(len(accepted)))
    written: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, len(accepted) * 3.1), 4.8))
    he_values = [_as_dict(record["profile"]).get("he_forward_s") or 0.0 for record in accepted]
    encode_values = [_as_dict(record["profile"]).get("online_encode_s") or 0.0 for record in accepted]
    encode_errors = [
        _as_dict(record.get("measured_attempt_statistics")).get(
            "online_encode_sample_std_s"
        )
        or 0.0
        for record in accepted
    ]
    pct_values = [
        _as_dict(record["profile"]).get("online_encode_pct_of_he_forward") or 0.0
        for record in accepted
    ]
    pct_errors = [
        _as_dict(record.get("measured_attempt_statistics")).get(
            "online_encode_pct_sample_std"
        )
        or 0.0
        for record in accepted
    ]
    width = 0.36
    axes[0].bar([value - width / 2 for value in x], he_values, width, label="HE forward")
    axes[0].bar(
        [value + width / 2 for value in x],
        encode_values,
        width,
        yerr=encode_errors,
        capsize=4,
        label="Online Encode",
    )
    axes[0].set_ylabel("Seconds per measured forward")
    axes[0].set_title("Wall time")
    axes[0].legend()
    axes[1].bar(x, pct_values, yerr=pct_errors, capsize=4, color="tab:orange")
    axes[1].set_ylabel("Percent of HE-forward wall time")
    axes[1].set_title("Online Encode proportion")
    axes[1].set_ylim(0, max(100.0, max(pct_values) * 1.12))
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Step 1 online Encode profile (accepted real FHE only)")
    fig.tight_layout()
    path = out_dir / "online_encode_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, axis = plt.subplots(figsize=(max(8.5, len(accepted) * 2.4), 5.8))
    bottoms = [0.0] * len(accepted)
    for category in MAJOR_CATEGORY_ORDER:
        values = [
            _category_value(
                _as_dict(_as_dict(record.get("profile")).get("major_wall_categories")),
                category,
                "percent_of_he_forward",
            )
            or 0.0
            for record in accepted
        ]
        if not any(value > 0.0001 for value in values):
            continue
        axis.bar(x, values, bottom=bottoms, label=category.replace("_", " "))
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_xticks(x, labels)
    axis.set_ylabel("Percent of HE-forward wall time")
    axis.set_ylim(0, 100.5)
    axis.set_title("Additive major wall categories (sum to 100%)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    path = out_dir / "major_wall_categories_pct.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, axis = plt.subplots(figsize=(max(10.0, len(MICROPROFILE_ORDER) * 1.4), 5.6))
    group_width = 0.82
    bar_width = group_width / len(accepted)
    category_x = list(range(len(MICROPROFILE_ORDER)))
    for record_index, record in enumerate(accepted):
        micro = _as_dict(_as_dict(record.get("profile")).get("operator_microprofile"))
        values = [
            _category_value(micro, category, "percent_of_he_forward") or 0.0
            for category in MICROPROFILE_ORDER
        ]
        offsets = [
            value - group_width / 2 + bar_width / 2 + record_index * bar_width
            for value in category_x
        ]
        axis.bar(offsets, values, bar_width, label=_plot_label(record).replace("\n", " "))
    axis.set_xticks(category_x, [value.replace("_", "\n") for value in MICROPROFILE_ORDER])
    axis.set_ylabel("Diagnostic seconds / HE-forward wall time (%)")
    axis.set_title("Operator microprofile (non-additive: do not sum bars)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "operator_microprofile_pct.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, axis = plt.subplots(figsize=(max(11.0, len(OPERATION_COUNT_ORDER) * 1.25), 5.8))
    group_width = 0.82
    bar_width = group_width / len(accepted)
    operation_x = list(range(len(OPERATION_COUNT_ORDER)))
    for record_index, record in enumerate(accepted):
        counts = _as_dict(
            _as_dict(record.get("profile")).get("operation_counts_mean_per_forward")
        )
        values = [_number(counts.get(operation)) or 0.0 for operation in OPERATION_COUNT_ORDER]
        offsets = [
            value - group_width / 2 + bar_width / 2 + record_index * bar_width
            for value in operation_x
        ]
        axis.bar(offsets, values, bar_width, label=_plot_label(record).replace("\n", " "))
    axis.set_xticks(
        operation_x,
        [value.replace("_count", "").replace("_", "\n") for value in OPERATION_COUNT_ORDER],
    )
    axis.set_yscale("log")
    axis.set_ylabel("Mean count per measured forward (log scale)")
    axis.set_title("Lattigo linear-transform operation counts")
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "operation_counts_log.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


def write_outputs(
    records: Sequence[dict[str, Any]],
    read_errors: Sequence[str],
    out_dir: Path,
    *,
    plots: bool,
) -> tuple[list[Path], Optional[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        out_dir / "extracted_results.json",
        out_dir / "runs.csv",
        out_dir / "attempts.csv",
        out_dir / "major_wall_categories.csv",
        out_dir / "operator_microprofile.csv",
        out_dir / "operation_counts.csv",
        out_dir / "report.md",
    ]
    summary = {
        "schema_version": 1,
        "accepted_real_fhe_count": sum(
            bool(record.get("accepted_as_step1")) for record in records
        ),
        "readable_result_count": len(records),
        "read_errors": list(read_errors),
        "records": list(records),
    }
    output_paths[0].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_paths[1], [_run_row(record) for record in records])
    _write_csv(output_paths[2], _attempt_rows(records))
    _write_csv(
        output_paths[3],
        _long_rows(records, "major_wall_categories", MAJOR_CATEGORY_ORDER),
    )
    _write_csv(
        output_paths[4],
        _long_rows(records, "operator_microprofile", MICROPROFILE_ORDER),
    )
    _write_csv(output_paths[5], _operation_rows(records))
    output_paths[6].write_text(
        build_markdown_report(records, read_errors), encoding="utf-8"
    )

    plot_error: Optional[str] = None
    if plots:
        try:
            output_paths.extend(write_plots(records, out_dir))
        except RuntimeError as exc:
            plot_error = str(exc)
    return output_paths, plot_error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract, validate, tabulate, and plot saved Step 1 online-Encode "
            "profiling results. Progress-state checkpoints are ignored."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Result JSON files or directories. Defaults to honours/04_step1_model_matrix "
            "and honours/05_vgg16_profile_fix. Paths that omit the honours directory "
            "are resolved automatically when possible."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any JSON input cannot be read.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    requested_inputs = tuple(args.inputs) if args.inputs else DEFAULT_INPUTS
    try:
        result_paths = discover_result_files(requested_inputs)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not result_paths:
        print("No primary result JSON files were found.", file=sys.stderr)
        return 2

    records, read_errors = _load_records(result_paths)
    if not records:
        print("No readable result JSON files were found.", file=sys.stderr)
        for error in read_errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir = out_dir.resolve()
    output_paths, plot_error = write_outputs(
        records, read_errors, out_dir, plots=not bool(args.no_plots)
    )

    accepted = [record for record in records if record.get("accepted_as_step1")]
    print(f"Read {len(records)} primary result(s); accepted {len(accepted)} real-FHE Step 1 profile(s).")
    for record in records:
        profile = _as_dict(record.get("profile"))
        marker = "ACCEPT" if record.get("accepted_as_step1") else "EXCLUDE"
        print(
            f"[{marker}] {record.get('label')} ({record.get('mode')}): "
            f"{record.get('result_kind')}; HE={_fmt(profile.get('he_forward_s'))} s, "
            f"Encode={_fmt(profile.get('online_encode_s'))} s "
            f"({_fmt(profile.get('online_encode_pct_of_he_forward'))}%)"
        )
        if not record.get("accepted_as_step1"):
            print(f"  reason: {'; '.join(record.get('acceptance_errors', []))}")
    print(f"Outputs: {out_dir}")
    if plot_error:
        print(f"Plot warning: {plot_error}", file=sys.stderr)
    for error in read_errors:
        print(f"Read warning: {error}", file=sys.stderr)
    if bool(args.strict) and read_errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
