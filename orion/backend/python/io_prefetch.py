from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _PrefetchEntry:
    future: Future
    host_bytes: int = 0
    device_bytes: int = 0


class AsyncIOPrefetcher:
    def __init__(self) -> None:
        self._entries: dict[Any, _PrefetchEntry] = {}

    def submit(
        self,
        key: Any,
        loader: Callable[[], Any],
        *,
        host_bytes: int = 0,
        device_bytes: int = 0,
        replace: bool = False,
    ) -> bool:
        if key in self._entries:
            if not replace:
                return False
            self.discard(key)
        self._entries[key] = _PrefetchEntry(
            future=_PREFETCH_EXECUTOR.submit(loader),
            host_bytes=int(host_bytes or 0),
            device_bytes=int(device_bytes or 0),
        )
        return True

    def consume(self, key: Any) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        try:
            return entry.future.result()
        except Exception:
            return None
        finally:
            self._entries.pop(key, None)

    def discard(self, key: Any, *, wait: bool = False) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        cancelled = entry.future.cancel()
        if wait and not cancelled:
            try:
                entry.future.result()
            except Exception:
                pass

    def pending_host_bytes(self) -> int:
        return sum(int(entry.host_bytes) for entry in self._entries.values())

    def pending_device_bytes(self) -> int:
        return sum(int(entry.device_bytes) for entry in self._entries.values())

    def pending_count(self) -> int:
        return len(self._entries)

    def has_pending(self, key: Any) -> bool:
        return key in self._entries

    def clear(self, *, wait: bool = False) -> None:
        entries = list(self._entries.values())
        self._entries.clear()
        for entry in entries:
            cancelled = entry.future.cancel()
            if wait and not cancelled:
                try:
                    entry.future.result()
                except Exception:
                    pass
