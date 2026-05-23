from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

from .memory_lifecycle import host_memory_info


_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}
_GIB = 1024**3


@dataclass(frozen=True)
class CompileMemoryBudget:
    total_bytes: int
    available_bytes: int
    reserve_bytes: int
    budget_bytes: int


def compile_parallel_policy() -> str:
    raw = os.environ.get("ORION_COMPILE_PARALLEL_POLICY", "auto").strip().lower()
    return "manual" if raw == "manual" else "auto"


def manual_parallel_policy_enabled() -> bool:
    return compile_parallel_policy() == "manual"


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def first_env_int(names: Sequence[str]) -> int | None:
    for name in names:
        value = _env_int(str(name))
        if value is not None:
            return int(value)
    return None


def _compile_reserve_bytes(total_bytes: int) -> int:
    explicit = _env_int("ORION_COMPILE_MEMORY_RESERVE_BYTES")
    if explicit is not None:
        return int(explicit)
    raw_gib = os.environ.get("ORION_COMPILE_MEMORY_RESERVE_GB")
    if raw_gib:
        try:
            return max(0, int(float(raw_gib) * _GIB))
        except ValueError:
            pass
    raw_fraction = os.environ.get("ORION_COMPILE_MEMORY_RESERVE_FRACTION")
    fraction = 0.12
    if raw_fraction:
        try:
            fraction = max(0.0, min(0.95, float(raw_fraction)))
        except ValueError:
            fraction = 0.12
    return max(16 * _GIB, int(int(total_bytes) * float(fraction)))


def compile_memory_budget(memory_info: dict[str, int] | None = None) -> CompileMemoryBudget:
    info = host_memory_info() if memory_info is None else memory_info
    if info is None:
        return CompileMemoryBudget(0, 0, 0, 0)
    total = int(info.get("total_bytes", 0) or 0)
    available = int(info.get("available_bytes", 0) or 0)
    reserve = int(_compile_reserve_bytes(total))
    return CompileMemoryBudget(
        total_bytes=int(total),
        available_bytes=int(available),
        reserve_bytes=int(reserve),
        budget_bytes=max(0, int(available - reserve)),
    )


def manual_worker_count(
    item_count: int,
    env_names: Sequence[str],
    *,
    default_workers: int,
    cpu_count: int | None = None,
) -> int:
    item_count = int(item_count)
    if item_count <= 1:
        return 1
    requested = first_env_int(env_names)
    if requested is None:
        requested = int(default_workers)
    cpu = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    return max(1, min(item_count, cpu, int(requested)))


def auto_worker_count(
    item_count: int,
    env_names: Sequence[str],
    *,
    default_workers: int,
    estimated_per_worker_bytes: int,
    cpu_count: int | None = None,
    memory_info: dict[str, int] | None = None,
) -> int:
    if manual_parallel_policy_enabled():
        return manual_worker_count(
            int(item_count),
            env_names,
            default_workers=int(default_workers),
            cpu_count=cpu_count,
        )
    item_count = int(item_count)
    if item_count <= 1:
        return 1
    cpu = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    hard_cap = first_env_int(env_names)
    cap = min(item_count, cpu, int(hard_cap)) if hard_cap is not None else min(item_count, cpu)
    budget = compile_memory_budget(memory_info).budget_bytes
    per_worker = max(1, int(estimated_per_worker_bytes or 1))
    if budget <= 0:
        memory_cap = 1
    else:
        memory_cap = max(1, int(budget // per_worker))
    return max(1, min(int(cap), int(memory_cap)))


def manual_batch_limit(
    item_count: int,
    env_names: Sequence[str],
    *,
    default_limit: int,
) -> int:
    item_count = int(item_count)
    if item_count <= 1:
        return 1
    requested = first_env_int(env_names)
    if requested is None:
        requested = int(default_limit)
    return max(1, min(item_count, int(requested)))


def auto_batch_limit(
    item_count: int,
    env_names: Sequence[str],
    *,
    default_limit: int,
    estimated_item_bytes: int,
    memory_info: dict[str, int] | None = None,
) -> int:
    if manual_parallel_policy_enabled():
        return manual_batch_limit(
            int(item_count),
            env_names,
            default_limit=int(default_limit),
        )
    item_count = int(item_count)
    if item_count <= 1:
        return 1
    hard_cap = first_env_int(env_names)
    cap = min(item_count, int(hard_cap)) if hard_cap is not None else int(item_count)
    budget = compile_memory_budget(memory_info).budget_bytes
    per_item = max(1, int(estimated_item_bytes or 1))
    memory_cap = max(1, int(budget // per_item)) if budget > 0 else 1
    return max(1, min(int(cap), int(memory_cap)))


def batch_limit_for_payloads(
    payload_bytes: Iterable[int],
    *,
    hard_cap: int | None = None,
    memory_info: dict[str, int] | None = None,
) -> int:
    sizes = [max(0, int(value)) for value in payload_bytes]
    if not sizes:
        return 0
    if manual_parallel_policy_enabled():
        return max(1, min(len(sizes), int(hard_cap or len(sizes))))
    budget = compile_memory_budget(memory_info).budget_bytes
    cap = max(1, min(len(sizes), int(hard_cap or len(sizes))))
    if budget <= 0:
        return 1
    total = 0
    count = 0
    for size in sizes[:cap]:
        if count and total + int(size) > budget:
            break
        total += int(size)
        count += 1
    return max(1, int(count))


def policy_audit(memory_info: dict[str, int] | None = None) -> dict[str, int | str]:
    budget = compile_memory_budget(memory_info)
    return {
        "policy": compile_parallel_policy(),
        "total_bytes": int(budget.total_bytes),
        "available_bytes": int(budget.available_bytes),
        "reserve_bytes": int(budget.reserve_bytes),
        "budget_bytes": int(budget.budget_bytes),
    }
