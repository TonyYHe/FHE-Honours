from __future__ import annotations

import gc
import os
import time
from typing import Any


_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def host_memory_info() -> dict[str, int] | None:
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    values[parts[0][:-1]] = int(parts[1]) * 1024
    except OSError:
        return None

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None
    return {"total_bytes": int(total), "available_bytes": int(available)}


def runtime_trim_enabled(backend: Any) -> bool:
    trim = getattr(backend, "TrimRuntimeMemory", None)
    if not callable(trim):
        return False
    for name in ("ORION_FORWARD_TRIM_RUNTIME_MEMORY", "ORION_TRIM_RUNTIME_AFTER_UNLOAD"):
        raw_value = os.environ.get(name)
        if raw_value is not None:
            return raw_value.strip().lower() not in _FALSE_ENV_VALUES
    return True


def trim_runtime_memory(backend: Any, *, reason: str = "") -> dict[str, Any]:
    started = time.perf_counter()
    gc.collect()
    backend_trim_s = 0.0
    trim = getattr(backend, "TrimRuntimeMemory", None)
    if callable(trim):
        try:
            backend_trim_s = float(trim())
        except Exception:
            backend_trim_s = 0.0
    gc.collect()
    return {
        "reason": str(reason),
        "trim_s": float(time.perf_counter() - started),
        "backend_trim_s": float(backend_trim_s),
    }


def forward_min_available_bytes() -> int:
    explicit = os.environ.get("ORION_FORWARD_MIN_AVAILABLE_BYTES")
    if explicit:
        try:
            return max(0, int(explicit))
        except ValueError:
            pass
    explicit_gb = os.environ.get("ORION_FORWARD_MIN_AVAILABLE_GB")
    if explicit_gb:
        try:
            return max(0, int(float(explicit_gb) * 1024**3))
        except ValueError:
            pass
    return 0


def host_memory_guard_enabled() -> bool:
    raw_value = os.environ.get("ORION_FORWARD_HOST_MEMORY_GUARD")
    if raw_value is None:
        return True
    return raw_value.strip().lower() not in _FALSE_ENV_VALUES


def guard_host_memory(
    backend: Any,
    *,
    reason: str,
    needed_bytes: int = 0,
    force_trim: bool = False,
    raise_on_low: bool = True,
) -> dict[str, Any]:
    before = host_memory_info()
    result: dict[str, Any] = {
        "reason": str(reason),
        "needed_bytes": int(max(0, int(needed_bytes or 0))),
        "min_available_bytes": int(forward_min_available_bytes()),
        "before": before,
        "trim": None,
        "after": before,
    }
    if not host_memory_guard_enabled():
        return result

    min_available = int(result["min_available_bytes"])
    needed = int(result["needed_bytes"])

    def low(info: dict[str, int] | None) -> bool:
        if info is None or min_available <= 0:
            return False
        return int(info["available_bytes"]) - needed < min_available

    if bool(force_trim) or low(before):
        if runtime_trim_enabled(backend):
            result["trim"] = trim_runtime_memory(backend, reason=reason)
        after = host_memory_info()
        result["after"] = after
    else:
        after = before

    if bool(raise_on_low) and low(after):
        available = 0 if after is None else int(after["available_bytes"])
        raise MemoryError(
            "Orion forward host-memory guard tripped before "
            f"{reason}: available={available} needed={needed} "
            f"min_available={min_available}. Refusing to enter swap/OOM."
        )
    return result
