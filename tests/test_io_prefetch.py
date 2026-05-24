from types import SimpleNamespace

from orion.backend.python import io_prefetch
from orion.backend.python.lt_evaluator import NewEvaluator


def test_saved_io_prefetch_uses_host_and_device_gates(monkeypatch) -> None:
    gib = 1024 * 1024 * 1024
    monkeypatch.setattr(io_prefetch, "_meminfo_bytes", lambda: (10 * gib, 9 * gib))
    backend = SimpleNamespace(
        saved_io_prefetch_requires_device_memory=True,
        GetDeviceMemoryInfo=lambda: [9 * gib, 10 * gib],
    )

    assert io_prefetch.should_prefetch_saved_io(100 * 1024, backend=backend, device_bytes=100 * 1024)

    tight_backend = SimpleNamespace(
        saved_io_prefetch_requires_device_memory=True,
        GetDeviceMemoryInfo=lambda: [400 * 1024 * 1024, 10 * gib],
    )
    assert not io_prefetch.should_prefetch_saved_io(100 * 1024, backend=tight_backend, device_bytes=100 * 1024)


def test_saved_io_prefetch_skips_when_host_ram_is_tight(monkeypatch) -> None:
    gib = 1024 * 1024 * 1024
    monkeypatch.setattr(io_prefetch, "_meminfo_bytes", lambda: (10 * gib, 400 * 1024 * 1024))
    backend = SimpleNamespace(
        saved_io_prefetch_requires_device_memory=True,
        GetDeviceMemoryInfo=lambda: [9 * gib, 10 * gib],
    )

    assert not io_prefetch.should_prefetch_saved_io(100 * 1024, backend=backend, device_bytes=100 * 1024)


def test_async_io_prefetcher_keeps_multiple_keyed_entries() -> None:
    prefetcher = io_prefetch.AsyncIOPrefetcher()
    try:
        assert prefetcher.submit("a", lambda: "first", host_bytes=11, device_bytes=101)
        assert prefetcher.submit("b", lambda: "second", host_bytes=13, device_bytes=103)
        assert not prefetcher.submit("a", lambda: "replacement")

        assert prefetcher.pending_count() == 2
        assert prefetcher.pending_host_bytes() == 24
        assert prefetcher.pending_device_bytes() == 204
        assert prefetcher.consume("a") == "first"
        assert prefetcher.pending_count() == 1
        assert prefetcher.consume("b") == "second"
        assert prefetcher.pending_count() == 0
    finally:
        prefetcher.clear(wait=True)


def test_lt_evaluator_registers_global_saved_io_work_order() -> None:
    evaluator = NewEvaluator.__new__(NewEvaluator)
    evaluator._transform_io_prefetcher = io_prefetch.AsyncIOPrefetcher()
    evaluator._saved_io_external_work_order = []
    evaluator._saved_io_external_loaders = {}
    evaluator._saved_io_external_host_bytes = {}
    evaluator._saved_io_external_device_bytes = {}

    layers = [
        SimpleNamespace(name="first", transform_ids={(0, 0): 11, (0, 1): 12}),
        SimpleNamespace(name="second", transform_ids={(0, 0): 21}),
    ]
    evaluator.register_saved_io_schedule(layers)

    assert evaluator._saved_io_work_order == (
        ("first", 0, 0, 11),
        ("first", 0, 1, 12),
        ("second", 0, 0, 21),
    )
    assert evaluator._saved_io_work_index[("second", 0, 0, 21)] == 2


def test_lt_evaluator_saved_io_prefetch_default_lookahead_is_one(monkeypatch) -> None:
    evaluator = NewEvaluator.__new__(NewEvaluator)

    monkeypatch.delenv("ORION_SAVED_IO_PREFETCH_LOOKAHEAD", raising=False)
    assert evaluator._read_transform_io_lookahead() == 1

    monkeypatch.setenv("ORION_SAVED_IO_PREFETCH_LOOKAHEAD", "not-an-int")
    assert evaluator._read_transform_io_lookahead() == 1

    monkeypatch.setenv("ORION_SAVED_IO_PREFETCH_LOOKAHEAD", "0")
    assert evaluator._read_transform_io_lookahead() == 0


def test_lt_evaluator_rebuilds_transform_matrix_from_numeric_block_keys() -> None:
    evaluator = NewEvaluator.__new__(NewEvaluator)
    transform_ids = {}
    for key in (
        [(0, 0), (0, 1), (0, 10)]
        + [(0, col) for col in range(2, 10)]
        + [(1, 0), (1, 1), (1, 10)]
        + [(1, col) for col in range(2, 10)]
    ):
        row, col = key
        transform_ids[key] = int((row + 1) * 100 + col)

    rows, cols, matrix = evaluator._transform_id_matrix(transform_ids, input_cols=11)

    assert rows == 2
    assert cols == 11
    assert matrix[0][0] == 100
    assert matrix[0][1] == 101
    assert matrix[0][2] == 102
    assert matrix[0][10] == 110
    assert matrix[1][0] == 200
    assert matrix[1][10] == 210


def test_lt_evaluator_reports_plaintext_level_mismatch() -> None:
    evaluator = NewEvaluator.__new__(NewEvaluator)
    evaluator.backend = SimpleNamespace(
        GetLinearTransformEmptyPlaintextKeys=lambda _transform_id: [],
        GetLinearTransformPlaintextLevels=lambda _transform_id: [7, 2, 9, 3],
    )

    try:
        evaluator.ensure_plaintext_diagonals_loaded(
            "cached_conv",
            0,
            1,
            42,
            expected_level=3,
        )
    except RuntimeError as exc:
        assert "plaintext levels are incompatible" in str(exc)
        assert "expected_level=3" in str(exc)
        assert "(7, 2)" in str(exc)
    else:
        raise AssertionError("expected plaintext level mismatch")
