from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
from typing import Any, Callable


def _worker_count() -> int:
    return max(1, min(4, int(os.cpu_count() or 1)))


_PREFETCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=_worker_count(),
    thread_name_prefix="orion-prefetch",
)


def _meminfo_bytes() -> tuple[int | None, int | None]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            mem_total = None
            mem_available = None
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) * 1024
                if mem_total is not None and mem_available is not None:
                    break
    except OSError:
        return None, None

    return mem_total, mem_available


def should_prefetch_host_bytes(num_bytes: int) -> bool:
    bytes_needed = int(num_bytes or 0)
    if bytes_needed <= 0:
        return False

    mem_total, mem_available = _meminfo_bytes()
    if mem_available is None:
        return False

    reserve = 512 * 1024 * 1024
    if mem_total is not None:
        reserve = max(int(mem_total * 0.10), int(reserve))
    return int(mem_available) - int(bytes_needed) >= int(reserve)


def should_prefetch_device_bytes(backend: Any, num_bytes: int) -> bool:
    if not bool(getattr(backend, "saved_io_prefetch_requires_device_memory", False)):
        return True

    bytes_needed = int(num_bytes or 0)
    if bytes_needed <= 0:
        return False

    get_memory_info = getattr(backend, "GetDeviceMemoryInfo", None)
    if not callable(get_memory_info):
        return False

    memory_info = list(get_memory_info())
    if len(memory_info) < 2:
        return False

    free_bytes = int(memory_info[0])
    total_bytes = int(memory_info[1])
    reserve = max(512 * 1024 * 1024, int(total_bytes * 0.10))
    return free_bytes - bytes_needed >= reserve


def should_prefetch_saved_io(
    host_bytes: int,
    *,
    backend: Any | None = None,
    device_bytes: int = 0,
) -> bool:
    if not should_prefetch_host_bytes(int(host_bytes or 0)):
        return False
    if backend is None:
        return True
    return should_prefetch_device_bytes(backend, int(device_bytes or 0))


def should_prefetch_bytes(num_bytes: int) -> bool:
    return should_prefetch_host_bytes(num_bytes)


def estimate_linear_transform_device_bytes(backend: Any, transform_id: int) -> int:
    estimator = getattr(backend, "EstimateLinearTransformDeviceBytes", None)
    if not callable(estimator):
        return 0
    values = list(estimator(int(transform_id)))
    if not values:
        return 0
    return int(values[0])


class AsyncIOPrefetcher:
    def __init__(self) -> None:
        self._key: Any = None
        self._future: Future | None = None

    def submit(self, key: Any, loader: Callable[[], Any]) -> None:
        self.clear()
        self._key = key
        self._future = _PREFETCH_EXECUTOR.submit(loader)

    def consume(self, key: Any) -> Any | None:
        if self._future is None or self._key != key:
            return None
        try:
            return self._future.result()
        except Exception:
            return None
        finally:
            self._future = None
            self._key = None

    def clear(self, *, wait: bool = False) -> None:
        if self._future is not None:
            cancelled = self._future.cancel()
            if wait and not cancelled:
                try:
                    self._future.result()
                except Exception:
                    pass
        self._future = None
        self._key = None
