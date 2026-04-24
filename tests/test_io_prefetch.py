from types import SimpleNamespace

from orion.backend.python import io_prefetch


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
