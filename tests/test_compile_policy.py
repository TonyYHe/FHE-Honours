from __future__ import annotations

from orion.backend.python.compile_policy import (
    auto_batch_limit,
    auto_worker_count,
    batch_limit_for_payloads,
    compile_memory_budget,
)


GIB = 1024**3


def _clear_compile_env(monkeypatch) -> None:
    for name in (
        "ORION_COMPILE_PARALLEL_POLICY",
        "ORION_COMPILE_MEMORY_RESERVE_BYTES",
        "ORION_COMPILE_MEMORY_RESERVE_GB",
        "ORION_COMPILE_MEMORY_RESERVE_FRACTION",
        "ORION_TEST_WORKERS",
        "ORION_TEST_BATCH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_auto_worker_count_low_memory_forces_one_worker(monkeypatch) -> None:
    _clear_compile_env(monkeypatch)
    memory_info = {"total_bytes": 64 * GIB, "available_bytes": 18 * GIB}

    workers = auto_worker_count(
        16,
        ("ORION_TEST_WORKERS",),
        default_workers=8,
        estimated_per_worker_bytes=4 * GIB,
        cpu_count=16,
        memory_info=memory_info,
    )

    assert compile_memory_budget(memory_info).budget_bytes == 2 * GIB
    assert workers == 1


def test_auto_worker_count_respects_memory_cpu_and_env_caps(monkeypatch) -> None:
    _clear_compile_env(monkeypatch)
    monkeypatch.setenv("ORION_TEST_WORKERS", "6")
    memory_info = {"total_bytes": 128 * GIB, "available_bytes": 80 * GIB}

    workers = auto_worker_count(
        20,
        ("ORION_TEST_WORKERS",),
        default_workers=4,
        estimated_per_worker_bytes=8 * GIB,
        cpu_count=16,
        memory_info=memory_info,
    )

    assert workers == 6


def test_manual_compile_policy_preserves_env_worker_behavior(monkeypatch) -> None:
    _clear_compile_env(monkeypatch)
    monkeypatch.setenv("ORION_COMPILE_PARALLEL_POLICY", "manual")
    monkeypatch.setenv("ORION_TEST_WORKERS", "7")
    memory_info = {"total_bytes": 64 * GIB, "available_bytes": 17 * GIB}

    workers = auto_worker_count(
        20,
        ("ORION_TEST_WORKERS",),
        default_workers=3,
        estimated_per_worker_bytes=8 * GIB,
        cpu_count=12,
        memory_info=memory_info,
    )

    assert workers == 7


def test_auto_batch_limit_uses_memory_budget_and_hard_cap(monkeypatch) -> None:
    _clear_compile_env(monkeypatch)
    monkeypatch.setenv("ORION_TEST_BATCH", "5")
    memory_info = {"total_bytes": 64 * GIB, "available_bytes": 28 * GIB}

    assert (
        auto_batch_limit(
            20,
            ("ORION_TEST_BATCH",),
            default_limit=10,
            estimated_item_bytes=3 * GIB,
            memory_info=memory_info,
        )
        == 4
    )


def test_batch_limit_for_payloads_stops_at_cumulative_memory_budget(monkeypatch) -> None:
    _clear_compile_env(monkeypatch)
    memory_info = {"total_bytes": 64 * GIB, "available_bytes": 26 * GIB}

    limit = batch_limit_for_payloads(
        [2 * GIB, 3 * GIB, 6 * GIB, 1 * GIB],
        hard_cap=4,
        memory_info=memory_info,
    )

    assert limit == 2
