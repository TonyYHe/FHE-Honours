"""Read-only slot-message periodicity analysis for WPC experiments.

This module deliberately does not modify Orion payloads or runtime state.  It
classifies the exact power-of-two period of the values that are about to cross
the Python/Lattigo boundary and provides an optional audit collector.  Passing
this slot-message check makes a diagonal a *candidate* for WPC; the encoded Q/P
polynomial repetition and reconstruction checks are still required before a
diagonal is considered safe to compress.

The global collector is disabled unless ``ORION_WPC_PERIODICITY_PROFILE`` is
truthy.  A disabled ``record_payload`` call returns immediately without
materialising or scanning the payload.
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import math
import os
import platform
import struct
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union


PROFILE_ENV = "ORION_WPC_PERIODICITY_PROFILE"
PROFILE_JSONL_ENV = "ORION_WPC_PERIODICITY_JSONL"
PROFILE_SUMMARY_ENV = "ORION_WPC_PERIODICITY_SUMMARY"
ENCODED_VERIFY_ENV = "ORION_WPC_ENCODED_QP_VERIFY"
PROFILE_SCHEMA_VERSION = 2

EncodedQPVerifier = Callable[
    [Sequence[complex], "SlotPeriodicity", str, int],
    Mapping[str, Any],
]

PAYLOAD_FORMAT_AUTO = "auto"
PAYLOAD_FORMAT_REAL = "real"
PAYLOAD_FORMAT_COMPLEX = "complex"
PAYLOAD_FORMAT_INTERLEAVED_COMPLEX = "interleaved_complex"
PAYLOAD_FORMATS = {
    PAYLOAD_FORMAT_AUTO,
    PAYLOAD_FORMAT_REAL,
    PAYLOAD_FORMAT_COMPLEX,
    PAYLOAD_FORMAT_INTERLEAVED_COMPLEX,
}

# These fields define a logical diagonal when they are available.  Collection
# remains permissive so hooks can be brought up incrementally, but summaries
# report how many records have a complete identity.
EXPECTED_METADATA_FIELDS = (
    "model",
    "mode",
    "checkpoint_hash",
    "module_name",
    "operator_type",
    "transform_id",
    "block_row",
    "block_col",
    "diagonal_index",
    "level_q",
    "level_p",
    "source_dtype",
    "bsgs_n1",
    "bsgs_rotation",
)
IDENTITY_METADATA_FIELDS = (
    "model",
    "mode",
    "checkpoint_hash",
    "module_name",
    "operator_type",
    "transform_id",
    "block_row",
    "block_col",
    "diagonal_index",
)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _scalar_number(value: Any) -> Number:
    """Return a Python/numpy numeric scalar without accepting numeric strings."""

    if isinstance(value, Number):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if isinstance(converted, Number):
            return converted
    raise TypeError(
        "slot payload values must be real or complex numeric scalars; "
        f"received {type(value).__name__}"
    )


def canonicalize_slot_value(value: Any) -> complex:
    """Convert one slot to finite complex128 semantics and canonicalise zeros.

    Orion's Python/Lattigo interface uses floating-point slot values.  Converting
    to Python ``complex`` mirrors its double-precision real/imaginary semantics.
    Both components of signed zero are normalised to positive zero so hashes are
    stable across producers.
    """

    numeric = _scalar_number(value)
    try:
        converted = complex(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"slot payload value is not representable as complex: {value!r}") from exc
    real = float(converted.real)
    imag = float(converted.imag)
    if not math.isfinite(real) or not math.isfinite(imag):
        raise ValueError("slot payload contains NaN or infinity")
    if real == 0.0:
        real = 0.0
    if imag == 0.0:
        imag = 0.0
    return complex(real, imag)


def _materialize_payload(payload: Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(payload, (str, bytes, bytearray)):
        raise TypeError("slot payload must be a one-dimensional numeric iterable")
    try:
        values = tuple(payload)
    except TypeError as exc:
        raise TypeError("slot payload must be a one-dimensional numeric iterable") from exc
    if not values:
        raise ValueError("slot payload must contain at least one slot")
    return values


def canonicalize_slot_payload(
    payload: Iterable[Any],
    *,
    payload_format: str = PAYLOAD_FORMAT_AUTO,
) -> tuple[complex, ...]:
    """Return a canonical one-dimensional sequence of complex slot values.

    Accepted payload shapes are:

    - ``real``: ``[x0, x1, ..., x(n-1)]``; every imaginary part must be zero.
    - ``complex`` or ``auto``: ``[z0, z1, ..., z(n-1)]``.
    - ``interleaved_complex``: ``[re0, im0, re1, im1, ...]``.  The raw length
      is ``2*n`` and each raw component must itself be real.

    The decoded slot count ``n`` must be a power of two, as required by the
    CKKS slot vector examined by this experiment.
    """

    normalized_format = str(payload_format).strip().lower().replace("-", "_")
    if normalized_format == "interleaved":
        normalized_format = PAYLOAD_FORMAT_INTERLEAVED_COMPLEX
    if normalized_format not in PAYLOAD_FORMATS:
        allowed = ", ".join(sorted(PAYLOAD_FORMATS))
        raise ValueError(f"unsupported payload_format={payload_format!r}; expected one of {allowed}")

    raw = _materialize_payload(payload)
    slots: list[complex] = []
    if normalized_format == PAYLOAD_FORMAT_INTERLEAVED_COMPLEX:
        if len(raw) % 2:
            raise ValueError("interleaved_complex payload length must be even")
        for offset in range(0, len(raw), 2):
            real = canonicalize_slot_value(raw[offset])
            imag = canonicalize_slot_value(raw[offset + 1])
            if real.imag != 0.0 or imag.imag != 0.0:
                raise ValueError("interleaved_complex components must be real scalars")
            slots.append(canonicalize_slot_value(complex(real.real, imag.real)))
    else:
        for raw_value in raw:
            value = canonicalize_slot_value(raw_value)
            if normalized_format == PAYLOAD_FORMAT_REAL and value.imag != 0.0:
                raise ValueError("real payload contains a value with a nonzero imaginary part")
            slots.append(value)

    if not _is_power_of_two(len(slots)):
        raise ValueError(
            "decoded slot payload length must be a positive power of two; "
            f"received {len(slots)}"
        )
    return tuple(slots)


@dataclass(frozen=True)
class SlotPeriodicity:
    """Exact power-of-two period and semantic classification of one payload."""

    slot_count: int
    minimal_period: int
    all_zero: bool
    constant_nonzero: bool

    @property
    def exact_periodic(self) -> bool:
        """Whether the payload has a proper period (``T < n``)."""

        return self.minimal_period < self.slot_count

    @property
    def wpc_candidate(self) -> bool:
        """Whether WPC may be attempted after the encoded-form verification."""

        return self.exact_periodic and not self.all_zero

    @property
    def compression_ratio(self) -> int:
        """Analytical full/period slot ratio ``n/T``."""

        return self.slot_count // self.minimal_period

    @property
    def classification(self) -> str:
        if self.all_zero:
            return "all_zero"
        if self.constant_nonzero:
            return "constant_nonzero"
        if self.exact_periodic:
            return "periodic"
        return "aperiodic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots_n": self.slot_count,
            "slot_min_period_t": self.minimal_period,
            "slot_exact_periodic": self.exact_periodic,
            "wpc_slot_candidate": self.wpc_candidate,
            "all_zero": self.all_zero,
            "constant_nonzero": self.constant_nonzero,
            "slot_compression_ratio": self.compression_ratio,
            "slot_classification": self.classification,
        }


def _analyze_canonical_slots(slots: Sequence[complex]) -> SlotPeriodicity:
    slot_count = len(slots)
    if not _is_power_of_two(slot_count):
        raise ValueError("canonical slot count must be a positive power of two")

    # Starting with the trivial period n, repeatedly test its half.  Once a
    # half fails no smaller power-of-two period can succeed.  The comparisons
    # across all levels total fewer than n operations.
    period = slot_count
    while period > 1:
        half = period // 2
        if not all(slots[index] == slots[index + half] for index in range(half)):
            break
        period = half

    all_zero = all(value.real == 0.0 and value.imag == 0.0 for value in slots)
    constant_nonzero = period == 1 and not all_zero
    return SlotPeriodicity(
        slot_count=slot_count,
        minimal_period=period,
        all_zero=all_zero,
        constant_nonzero=constant_nonzero,
    )


def analyze_slot_periodicity(
    payload: Iterable[Any],
    *,
    payload_format: str = PAYLOAD_FORMAT_AUTO,
) -> SlotPeriodicity:
    """Find the exact minimal power-of-two period of a real/complex payload."""

    return _analyze_canonical_slots(
        canonicalize_slot_payload(payload, payload_format=payload_format)
    )


def _hash_canonical_slots(slots: Sequence[complex]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack(">Q", len(slots)))
    for value in slots:
        digest.update(struct.pack(">dd", value.real, value.imag))
    return digest.hexdigest()


def slot_payload_sha256(
    payload: Iterable[Any],
    *,
    payload_format: str = PAYLOAD_FORMAT_AUTO,
) -> str:
    """Hash canonical slot values without serialising or retaining the payload."""

    slots = canonicalize_slot_payload(payload, payload_format=payload_format)
    return _hash_canonical_slots(slots)


@dataclass(frozen=True)
class PeriodicityObservation:
    """One diagonal's periodicity with optional byte and Encode-time weights."""

    periodicity: SlotPeriodicity
    full_encoded_bytes: Optional[int] = None
    baseline_encode_s: Optional[float] = None
    compressed_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.full_encoded_bytes is not None:
            if isinstance(self.full_encoded_bytes, bool) or int(self.full_encoded_bytes) != self.full_encoded_bytes:
                raise TypeError("full_encoded_bytes must be an integer or None")
            if int(self.full_encoded_bytes) < 0:
                raise ValueError("full_encoded_bytes must be nonnegative")
        if self.compressed_bytes is not None:
            if self.full_encoded_bytes is None:
                raise ValueError("compressed_bytes requires full_encoded_bytes")
            if isinstance(self.compressed_bytes, bool) or int(self.compressed_bytes) != self.compressed_bytes:
                raise TypeError("compressed_bytes must be an integer or None")
            if int(self.compressed_bytes) < 0:
                raise ValueError("compressed_bytes must be nonnegative")
        if self.baseline_encode_s is not None:
            seconds = float(self.baseline_encode_s)
            if not math.isfinite(seconds) or seconds < 0.0:
                raise ValueError("baseline_encode_s must be finite and nonnegative")

    @property
    def effective_hybrid_bytes(self) -> Optional[int]:
        """Compressed candidate bytes or full fallback bytes, excluding zeros."""

        if self.periodicity.all_zero:
            return 0
        if self.full_encoded_bytes is None:
            return None
        full = int(self.full_encoded_bytes)
        if not self.periodicity.wpc_candidate:
            return full
        if self.compressed_bytes is not None:
            return int(self.compressed_bytes)
        numerator = full * self.periodicity.minimal_period
        if numerator % self.periodicity.slot_count:
            raise ValueError(
                "full_encoded_bytes is not divisible by the analytical n/T ratio; "
                "pass measured compressed_bytes explicitly"
            )
        return numerator // self.periodicity.slot_count


def _safe_ratio(numerator: Union[int, float], denominator: Union[int, float]) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


@dataclass(frozen=True)
class PeriodicityAggregate:
    """Count-, full-byte-, and Encode-time-weighted periodicity coverage."""

    observed_count: int
    nonzero_count: int
    periodic_count: int
    all_zero_count: int
    constant_nonzero_count: int
    byte_observation_count: int
    full_encoded_bytes: int
    periodic_full_encoded_bytes: int
    hybrid_storage_bytes: int
    time_observation_count: int
    baseline_encode_s: float
    periodic_encode_s: float

    @property
    def count_coverage(self) -> Optional[float]:
        return _safe_ratio(self.periodic_count, self.nonzero_count)

    @property
    def byte_coverage(self) -> Optional[float]:
        return _safe_ratio(self.periodic_full_encoded_bytes, self.full_encoded_bytes)

    @property
    def encode_time_coverage(self) -> Optional[float]:
        return _safe_ratio(self.periodic_encode_s, self.baseline_encode_s)

    @property
    def partial_storage_compression_ratio(self) -> Optional[float]:
        return _safe_ratio(self.full_encoded_bytes, self.hybrid_storage_bytes)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "count_coverage": self.count_coverage,
                "count_coverage_pct": (
                    None if self.count_coverage is None else 100.0 * self.count_coverage
                ),
                "byte_coverage": self.byte_coverage,
                "byte_coverage_pct": (
                    None if self.byte_coverage is None else 100.0 * self.byte_coverage
                ),
                "encode_time_coverage": self.encode_time_coverage,
                "encode_time_coverage_pct": (
                    None
                    if self.encode_time_coverage is None
                    else 100.0 * self.encode_time_coverage
                ),
                "partial_storage_compression_ratio": self.partial_storage_compression_ratio,
            }
        )
        return result


def aggregate_periodicity(
    observations: Iterable[PeriodicityObservation],
) -> PeriodicityAggregate:
    """Aggregate WPC-candidate coverage while excluding all-zero diagonals.

    Count coverage uses every nonzero observation.  Byte and time coverage use
    only observations for which the corresponding weight was supplied, and the
    output exposes those denominator counts to prevent accidental overclaiming.
    """

    rows = tuple(observations)
    nonzero = [row for row in rows if not row.periodicity.all_zero]
    periodic = [row for row in nonzero if row.periodicity.wpc_candidate]
    byte_rows = [row for row in nonzero if row.full_encoded_bytes is not None]
    time_rows = [row for row in nonzero if row.baseline_encode_s is not None]

    return PeriodicityAggregate(
        observed_count=len(rows),
        nonzero_count=len(nonzero),
        periodic_count=len(periodic),
        all_zero_count=sum(row.periodicity.all_zero for row in rows),
        constant_nonzero_count=sum(row.periodicity.constant_nonzero for row in rows),
        byte_observation_count=len(byte_rows),
        full_encoded_bytes=sum(int(row.full_encoded_bytes or 0) for row in byte_rows),
        periodic_full_encoded_bytes=sum(
            int(row.full_encoded_bytes or 0)
            for row in byte_rows
            if row.periodicity.wpc_candidate
        ),
        hybrid_storage_bytes=sum(int(row.effective_hybrid_bytes or 0) for row in byte_rows),
        time_observation_count=len(time_rows),
        baseline_encode_s=sum(float(row.baseline_encode_s or 0.0) for row in time_rows),
        periodic_encode_s=sum(
            float(row.baseline_encode_s or 0.0)
            for row in time_rows
            if row.periodicity.wpc_candidate
        ),
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata contains NaN or infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return _json_safe(converted)
    raise TypeError(f"metadata value {value!r} is not JSON serialisable")


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation_from_record(record: Mapping[str, Any]) -> PeriodicityObservation:
    periodicity = SlotPeriodicity(
        slot_count=int(record["slots_n"]),
        minimal_period=int(record["slot_min_period_t"]),
        all_zero=bool(record["all_zero"]),
        constant_nonzero=bool(record["constant_nonzero"]),
    )
    return PeriodicityObservation(
        periodicity=periodicity,
        full_encoded_bytes=record.get("full_encoded_bytes"),
        baseline_encode_s=record.get("baseline_encode_s"),
        compressed_bytes=record.get("compressed_bytes"),
    )


class PeriodicityCollector:
    """Thread-safe in-memory collector for a dedicated, untimed audit pass."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_sequence = 0
        self._occurrences: list[dict[str, Any]] = []
        self._unique: dict[str, dict[str, Any]] = {}
        self._logical_sources: dict[str, set[str]] = {}

    def record_payload(
        self,
        payload: Iterable[Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        payload_format: str = PAYLOAD_FORMAT_AUTO,
        full_encoded_bytes: Optional[int] = None,
        baseline_encode_s: Optional[float] = None,
        compressed_bytes: Optional[int] = None,
        encoded_qp_verifier: Optional[EncodedQPVerifier] = None,
    ) -> dict[str, Any]:
        """Analyze one materialization occurrence and retain metadata, not values."""

        started = time.perf_counter()
        slots = canonicalize_slot_payload(payload, payload_format=payload_format)
        periodicity = _analyze_canonical_slots(slots)
        source_hash = _hash_canonical_slots(slots)
        observation = PeriodicityObservation(
            periodicity=periodicity,
            full_encoded_bytes=full_encoded_bytes,
            baseline_encode_s=baseline_encode_s,
            compressed_bytes=compressed_bytes,
        )
        effective_compressed_bytes = observation.effective_hybrid_bytes
        encoded_verification: dict[str, Any] = {}
        if periodicity.wpc_candidate and encoded_qp_verifier is not None:
            encoded_verification = dict(
                encoded_qp_verifier(
                    slots,
                    periodicity,
                    str(payload_format),
                    int(dict(metadata or {}).get("level_q", -1)),
                )
            )

        safe_metadata = _json_safe(dict(metadata or {}))
        metadata_complete = all(
            field in safe_metadata and safe_metadata[field] is not None
            for field in IDENTITY_METADATA_FIELDS
        )
        logical_identity = {
            field: safe_metadata.get(field)
            for field in IDENTITY_METADATA_FIELDS
            if field in safe_metadata
        }
        if not logical_identity:
            logical_identity = dict(safe_metadata)
        logical_id = _stable_json_hash(logical_identity)
        diagonal_id = _stable_json_hash(
            {"logical_identity": logical_identity, "source_hash": source_hash}
        )

        record: dict[str, Any] = {
            "record_type": "slot_periodicity_occurrence",
            "schema_version": PROFILE_SCHEMA_VERSION,
            "diagonal_id": diagonal_id,
            "logical_diagonal_id": logical_id,
            "metadata_complete": metadata_complete,
            "payload_format": str(payload_format),
            "source_hash": source_hash,
            **periodicity.to_dict(),
            "full_encoded_bytes": (
                None if full_encoded_bytes is None else int(full_encoded_bytes)
            ),
            "compressed_bytes": effective_compressed_bytes,
            "baseline_encode_s": (
                None if baseline_encode_s is None else float(baseline_encode_s)
            ),
            "analysis_s": float(time.perf_counter() - started),
        }
        if encoded_verification:
            record["encoded_qp_verification"] = _json_safe(encoded_verification)
        for field in EXPECTED_METADATA_FIELDS:
            record[field] = safe_metadata.get(field)
        extras = {
            key: value
            for key, value in safe_metadata.items()
            if key not in EXPECTED_METADATA_FIELDS
        }
        if extras:
            record["extra_metadata"] = extras

        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            record["sequence"] = sequence
            self._occurrences.append(record)
            if diagonal_id not in self._unique:
                unique_record = dict(record)
                unique_record["record_type"] = "slot_periodicity_unique"
                self._unique[diagonal_id] = unique_record
            self._logical_sources.setdefault(logical_id, set()).add(source_hash)
        return dict(record)

    def clear(self) -> None:
        with self._lock:
            self._next_sequence = 0
            self._occurrences.clear()
            self._unique.clear()
            self._logical_sources.clear()

    def _snapshot(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        with self._lock:
            occurrences = [dict(record) for record in self._occurrences]
            unique = [dict(record) for record in self._unique.values()]
            payload_change_count = sum(
                len(source_hashes) > 1 for source_hashes in self._logical_sources.values()
            )
        return occurrences, unique, payload_change_count

    def summary(self) -> dict[str, Any]:
        occurrences, unique, payload_change_count = self._snapshot()
        occurrence_aggregate = aggregate_periodicity(
            _observation_from_record(record) for record in occurrences
        )
        unique_aggregate = aggregate_periodicity(
            _observation_from_record(record) for record in unique
        )
        candidate_occurrences = [
            record for record in occurrences if bool(record.get("wpc_slot_candidate", False))
        ]
        verified_occurrences = [
            record
            for record in candidate_occurrences
            if isinstance(record.get("encoded_qp_verification"), Mapping)
            and bool(record["encoded_qp_verification"].get("attempted", False))
        ]
        passed_occurrences = [
            record
            for record in verified_occurrences
            if bool(record["encoded_qp_verification"].get("passed", False))
        ]
        encoded_verification_complete = bool(
            len(verified_occurrences) == len(candidate_occurrences)
            and len(passed_occurrences) == len(candidate_occurrences)
        )
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile": "wpc_orion_slot_periodicity",
            "scope": (
                "slot_message_plus_encoded_qp_candidates"
                if verified_occurrences
                else "slot_message_only"
            ),
            "encoded_representation_verification_required": not encoded_verification_complete,
            "encoded_qp_verification": {
                "candidate_occurrence_count": len(candidate_occurrences),
                "attempted_occurrence_count": len(verified_occurrences),
                "passed_occurrence_count": len(passed_occurrences),
                "failed_occurrence_count": len(verified_occurrences) - len(passed_occurrences),
                "complete": encoded_verification_complete,
            },
            "occurrence": occurrence_aggregate.to_dict(),
            "unique_diagonal": unique_aggregate.to_dict(),
            "metadata_complete_occurrence_count": sum(
                bool(record["metadata_complete"]) for record in occurrences
            ),
            "metadata_complete_unique_count": sum(
                bool(record["metadata_complete"]) for record in unique
            ),
            "logical_diagonal_payload_change_count": payload_change_count,
        }

    def flush(
        self,
        jsonl_path: Union[str, Path],
        *,
        summary_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        """Atomically write unique and occurrence JSONL plus a summary JSON."""

        destination = Path(jsonl_path)
        if summary_path is None:
            summary_destination = destination.with_suffix(".summary.json")
        else:
            summary_destination = Path(summary_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        summary_destination.parent.mkdir(parents=True, exist_ok=True)

        occurrences, unique, _ = self._snapshot()
        summary = self.summary()
        summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["jsonl_path"] = str(destination)
        summary["summary_path"] = str(summary_destination)

        def write_jsonl(handle: Any) -> None:
            for record in sorted(unique, key=lambda item: str(item["diagonal_id"])):
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False))
                handle.write("\n")
            for record in sorted(occurrences, key=lambda item: int(item["sequence"])):
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False))
                handle.write("\n")

        _atomic_text_write(destination, write_jsonl)
        _atomic_text_write(
            summary_destination,
            lambda handle: json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False),
        )
        return summary


def _atomic_text_write(path: Path, writer: Any) -> None:
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


_DISABLED = object()
_GLOBAL_LOCK = threading.Lock()
_GLOBAL_COLLECTOR: Union[PeriodicityCollector, object, None] = None


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def real_lattigo_library_path() -> Path:
    """Resolve the real Lattigo shared library even during a clear run."""

    override = str(os.environ.get("ORION_LATTIGO_LIBRARY_PATH", "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    system = platform.system()
    if system == "Linux":
        name = "lattigo-linux.so"
    elif system == "Darwin":
        name = (
            "lattigo-mac-arm64.dylib"
            if platform.machine().lower() in {"arm64", "aarch64"}
            else "lattigo-mac.dylib"
        )
    elif system == "Windows":
        name = "lattigo-windows.dll"
    else:
        raise RuntimeError(f"unsupported platform for Lattigo verifier: {system}")
    return Path(__file__).resolve().parents[1] / "backend" / "lattigo" / name


class _LattigoEncodedQPVerifier:
    """Minimal ctypes bridge used only by the untimed clear audit."""

    _STATUS = {
        1: "verified",
        0: "encoded_qp_mismatch",
        -1: "invalid_input",
        -2: "source_not_periodic",
        -3: "encode_failure",
    }

    def __init__(self, params: Any) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._logn = int(params.get_logn())
        self._logq = tuple(int(value) for value in params.get_logq())
        self._logp = tuple(int(value) for value in params.get_logp())
        self._logscale = int(params.get_logscale())
        self._hamming_weight = int(params.get_hamming_weight())
        self._ringtype = str(params.get_ringtype())
        self.signature = (
            self._logn,
            self._logq,
            self._logp,
            self._logscale,
            self._hamming_weight,
            self._ringtype,
        )

        path = real_lattigo_library_path()
        if not path.is_file():
            raise RuntimeError(
                f"real Lattigo verifier library does not exist: {path}; "
                "run python tools/build_lattigo.py first"
            )
        self.library_path = path
        self._lib = ctypes.CDLL(str(path))
        try:
            new_scheme = self._lib.NewScheme
            verify = self._lib.VerifyWPCEncodedDiagonal
            delete_scheme = self._lib.DeleteScheme
        except AttributeError as exc:
            raise RuntimeError(
                f"{path} does not expose the WPC encoded verifier; rebuild it from current source"
            ) from exc

        new_scheme.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        new_scheme.restype = None
        verify.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        verify.restype = ctypes.c_int
        delete_scheme.argtypes = []
        delete_scheme.restype = None
        self._verify = verify
        self._delete_scheme = delete_scheme

        logq_array = (ctypes.c_int * len(self._logq))(*self._logq)
        logp_array = (ctypes.c_int * len(self._logp))(*self._logp)
        new_scheme(
            self._logn,
            logq_array,
            len(self._logq),
            logp_array,
            len(self._logp),
            self._logscale,
            self._hamming_weight,
            self._ringtype.encode("utf-8"),
            b"",
            b"none",
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._delete_scheme()
            self._closed = True

    def verify(
        self,
        slots: Sequence[complex],
        periodicity: SlotPeriodicity,
        payload_format: str,
        level_q: int,
    ) -> Mapping[str, Any]:
        is_complex = str(payload_format) in {
            PAYLOAD_FORMAT_COMPLEX,
            PAYLOAD_FORMAT_INTERLEAVED_COMPLEX,
        }
        raw_values: list[float] = []
        if is_complex:
            for value in slots:
                raw_values.extend((float(value.real), float(value.imag)))
        else:
            raw_values.extend(float(value.real) for value in slots)
        values_array = (ctypes.c_double * len(raw_values))(*raw_values)
        with self._lock:
            if self._closed:
                raise RuntimeError("the WPC encoded verifier has already been closed")
            status_code = int(
                self._verify(
                    values_array,
                    len(raw_values),
                    int(is_complex),
                    int(level_q),
                    int(periodicity.minimal_period),
                )
            )
        return {
            "attempted": True,
            "passed": status_code == 1,
            "status_code": status_code,
            "status": self._STATUS.get(status_code, "unknown_status"),
            "method": "lattigo_qp_evaluation_copy_map_exact_roundtrip",
            "library_path": str(self.library_path),
            "level_q": int(level_q),
            "level_p": int(len(self._logp) - 1),
            "q_limb_count": int(level_q + 1),
            "p_limb_count": int(len(self._logp)),
            "slot_period_t": int(periodicity.minimal_period),
            "evaluation_period_2t": int(2 * periodicity.minimal_period),
            "full_polynomial_reconstruction_exact": status_code == 1,
        }


_ENCODED_VERIFIER_LOCK = threading.Lock()
_ENCODED_VERIFIER: Optional[_LattigoEncodedQPVerifier] = None


def encoded_qp_verification_enabled() -> bool:
    return _env_truthy(ENCODED_VERIFY_ENV)


def encoded_qp_verifier_for_params(params: Any) -> Optional[EncodedQPVerifier]:
    """Return a process-global verifier bound to the model's CKKS parameters."""

    global _ENCODED_VERIFIER
    if not encoded_qp_verification_enabled():
        return None
    with _ENCODED_VERIFIER_LOCK:
        candidate_signature = (
            int(params.get_logn()),
            tuple(int(value) for value in params.get_logq()),
            tuple(int(value) for value in params.get_logp()),
            int(params.get_logscale()),
            int(params.get_hamming_weight()),
            str(params.get_ringtype()),
        )
        if _ENCODED_VERIFIER is None:
            _ENCODED_VERIFIER = _LattigoEncodedQPVerifier(params)
        elif _ENCODED_VERIFIER.signature != candidate_signature:
            raise RuntimeError(
                "one WPC audit process cannot mix different CKKS parameter sets"
            )
        return _ENCODED_VERIFIER.verify


def _global_collector() -> Optional[PeriodicityCollector]:
    global _GLOBAL_COLLECTOR
    current = _GLOBAL_COLLECTOR
    if isinstance(current, PeriodicityCollector):
        return current
    if current is _DISABLED:
        return None
    with _GLOBAL_LOCK:
        if _GLOBAL_COLLECTOR is None:
            _GLOBAL_COLLECTOR = PeriodicityCollector() if _env_truthy(PROFILE_ENV) else _DISABLED
        return _GLOBAL_COLLECTOR if isinstance(_GLOBAL_COLLECTOR, PeriodicityCollector) else None


def periodicity_collection_enabled() -> bool:
    """Return whether the env-gated global collector is active."""

    return _global_collector() is not None


def record_payload(
    payload: Iterable[Any],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    payload_format: str = PAYLOAD_FORMAT_AUTO,
    full_encoded_bytes: Optional[int] = None,
    baseline_encode_s: Optional[float] = None,
    compressed_bytes: Optional[int] = None,
    encoded_qp_verifier: Optional[EncodedQPVerifier] = None,
) -> Optional[dict[str, Any]]:
    """Record one payload when profiling is enabled; otherwise do no work."""

    collector = _global_collector()
    if collector is None:
        return None
    return collector.record_payload(
        payload,
        metadata=metadata,
        payload_format=payload_format,
        full_encoded_bytes=full_encoded_bytes,
        baseline_encode_s=baseline_encode_s,
        compressed_bytes=compressed_bytes,
        encoded_qp_verifier=encoded_qp_verifier,
    )


def encoded_plaintext_bytes(
    *,
    ring_degree: int,
    level_q: int,
    level_p: int,
) -> int:
    """Return the Q/P polynomial payload size for one encoded diagonal.

    Lattigo stores one ``uint64`` coefficient per ring position in each active
    Q and P limb.  Levels are zero based, so level ``L`` contains ``L+1``
    limbs.  ``level_p=-1`` denotes an empty P basis.
    """

    n = int(ring_degree)
    q = int(level_q)
    p = int(level_p)
    if not _is_power_of_two(n):
        raise ValueError("ring_degree must be a positive power of two")
    if q < 0 or p < -1:
        raise ValueError("level_q must be nonnegative and level_p must be at least -1")
    return int(n * 8 * ((q + 1) + (p + 1)))


def record_flattened_diagonals(
    diag_indices: Iterable[Any],
    diag_data: Any,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    has_complex: bool = False,
    full_encoded_bytes: Optional[int] = None,
    encoded_qp_verifier: Optional[EncodedQPVerifier] = None,
) -> int:
    """Split one backend batch payload and record each logical diagonal.

    Orion flattens all diagonals for a linear transform before crossing the
    Python/Go boundary.  Real payloads contain ``n`` values per diagonal;
    complex payloads contain ``2*n`` interleaved real/imaginary components.
    The function validates that split before scanning any values.  It is a
    strict no-op, including no payload conversion, when profiling is disabled.
    """

    collector = _global_collector()
    if collector is None:
        return 0

    indices = tuple(int(value) for value in diag_indices)
    if not indices:
        if int(getattr(diag_data, "size", 0) or 0) != 0:
            raise ValueError("flattened diagonal data is nonempty but has no indices")
        return 0

    try:
        raw_count = int(diag_data.size)
    except (AttributeError, TypeError, ValueError):
        raw_count = len(diag_data)
    if raw_count % len(indices):
        raise ValueError(
            "flattened diagonal data length is not divisible by the diagonal count: "
            f"data={raw_count} diagonals={len(indices)}"
        )
    raw_per_diagonal = raw_count // len(indices)
    if bool(has_complex) and raw_per_diagonal % 2:
        raise ValueError("interleaved complex diagonal payload length must be even")
    slot_count = raw_per_diagonal // 2 if bool(has_complex) else raw_per_diagonal
    if not _is_power_of_two(slot_count):
        raise ValueError(
            "decoded slots per diagonal must be a positive power of two; "
            f"received {slot_count}"
        )

    base_metadata = dict(metadata or {})
    payload_format = (
        PAYLOAD_FORMAT_INTERLEAVED_COMPLEX if bool(has_complex) else PAYLOAD_FORMAT_REAL
    )
    for position, diagonal_index in enumerate(indices):
        start = int(position * raw_per_diagonal)
        stop = int(start + raw_per_diagonal)
        row_metadata = dict(base_metadata)
        row_metadata["diagonal_index"] = int(diagonal_index)
        row_metadata["diagonal_position"] = int(position)
        collector.record_payload(
            diag_data[start:stop],
            metadata=row_metadata,
            payload_format=payload_format,
            full_encoded_bytes=full_encoded_bytes,
            encoded_qp_verifier=encoded_qp_verifier,
        )
    return int(len(indices))


def flush_periodicity_profile(
    jsonl_path: Optional[Union[str, Path]] = None,
    *,
    summary_path: Optional[Union[str, Path]] = None,
) -> Optional[dict[str, Any]]:
    """Flush the enabled global collector using arguments or environment paths."""

    collector = _global_collector()
    if collector is None:
        return None
    resolved_jsonl = jsonl_path or os.environ.get(PROFILE_JSONL_ENV)
    if not resolved_jsonl:
        raise ValueError(
            f"jsonl_path is required (or set {PROFILE_JSONL_ENV}) when profiling is enabled"
        )
    resolved_summary = summary_path or os.environ.get(PROFILE_SUMMARY_ENV) or None
    return collector.flush(resolved_jsonl, summary_path=resolved_summary)


def reset_global_periodicity_collector() -> None:
    """Forget cached env state and records; intended for tests and run boundaries."""

    global _GLOBAL_COLLECTOR, _ENCODED_VERIFIER
    with _GLOBAL_LOCK:
        _GLOBAL_COLLECTOR = None
    with _ENCODED_VERIFIER_LOCK:
        if _ENCODED_VERIFIER is not None:
            _ENCODED_VERIFIER.close()
        _ENCODED_VERIFIER = None


def _flush_global_collector_at_exit() -> None:
    """Best-effort audit flush for runner processes configured by environment."""

    if not _env_truthy(PROFILE_ENV):
        return
    if not os.environ.get(PROFILE_JSONL_ENV):
        return
    try:
        flush_periodicity_profile()
    except Exception as exc:  # pragma: no cover - exercised only during shutdown
        print(f"Failed to flush WPC periodicity profile: {exc}", file=sys.stderr)


def _close_encoded_verifier_at_exit() -> None:
    global _ENCODED_VERIFIER
    with _ENCODED_VERIFIER_LOCK:
        if _ENCODED_VERIFIER is not None:
            try:
                _ENCODED_VERIFIER.close()
            except Exception as exc:  # pragma: no cover - shutdown only
                print(f"Failed to close WPC encoded verifier: {exc}", file=sys.stderr)
            _ENCODED_VERIFIER = None


atexit.register(_flush_global_collector_at_exit)
atexit.register(_close_encoded_verifier_at_exit)


__all__ = [
    "EXPECTED_METADATA_FIELDS",
    "ENCODED_VERIFY_ENV",
    "PAYLOAD_FORMAT_AUTO",
    "PAYLOAD_FORMAT_COMPLEX",
    "PAYLOAD_FORMAT_INTERLEAVED_COMPLEX",
    "PAYLOAD_FORMAT_REAL",
    "PROFILE_ENV",
    "PROFILE_JSONL_ENV",
    "PROFILE_SUMMARY_ENV",
    "PeriodicityAggregate",
    "PeriodicityCollector",
    "PeriodicityObservation",
    "SlotPeriodicity",
    "aggregate_periodicity",
    "analyze_slot_periodicity",
    "canonicalize_slot_payload",
    "canonicalize_slot_value",
    "encoded_plaintext_bytes",
    "encoded_qp_verification_enabled",
    "encoded_qp_verifier_for_params",
    "flush_periodicity_profile",
    "periodicity_collection_enabled",
    "record_flattened_diagonals",
    "record_payload",
    "real_lattigo_library_path",
    "reset_global_periodicity_collector",
    "slot_payload_sha256",
]
