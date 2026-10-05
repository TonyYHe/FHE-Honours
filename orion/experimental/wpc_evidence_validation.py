"""Fail-closed scalar and content-identity checks for experiment evidence."""

from __future__ import annotations

import hashlib
import io
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping


class EvidenceValidationError(ValueError):
    """An artifact cannot support the result claimed by its producer."""


def finite(value: Any, *, name: str, minimum: float | None = None) -> float:
    # JSON booleans and numeric strings are not measurements.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EvidenceValidationError(f"{name} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceValidationError(f"{name} is not finite")
    if minimum is not None and result < minimum:
        raise EvidenceValidationError(f"{name} is below {minimum}: {result}")
    return result


def integer(value: Any, *, name: str, minimum: int = 0) -> int:
    number = finite(value, name=name, minimum=minimum)
    if not number.is_integer():
        raise EvidenceValidationError(f"{name} is not an integer: {value!r}")
    return int(number)


def sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EvidenceValidationError(f"{name} is not a SHA-256 digest")
    return value


def gates(payload: Mapping[str, Any], *, name: str, required: set[str]) -> None:
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise EvidenceValidationError(f"{name} has no acceptance gates")
    missing = required - acceptance.keys()
    failed = {key for key, value in acceptance.items() if value is not True}
    if missing or failed:
        raise EvidenceValidationError(
            f"{name} acceptance gates: missing={sorted(missing)}, failed={sorted(failed)}"
        )


def close(actual: Any, expected: float, *, name: str, atol: float = 1e-8) -> None:
    number = finite(actual, name=name)
    if not math.isclose(number, expected, rel_tol=1e-10, abs_tol=atol):
        raise EvidenceValidationError(f"{name} is inconsistent: {number} != {expected}")


def load_checkpoint_bytes(path: Path) -> tuple[dict[str, Any], str]:
    """Hash and deserialize the SAME bytes, not two reads of a mutable path.

    Checkpoints are trusted local Torch artifacts (weights_only=False). The
    digest is content identity, not an authenticity/signature guarantee.
    """
    import torch

    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise EvidenceValidationError(f"{path}: checkpoint is not a mapping")
    return payload, digest
