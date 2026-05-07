"""Unified LinearTransform group support.

This is a focused port of the existing `st` branch UnifiedLinearTransform
support. It is backend support for region experiments: multiple transforms with
the same input can be compiled/evaluated through Lattigo's shared BSGS APIs.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from typing import Any, Iterable, List
import h5py
import numpy as np
import torch

from orion.backend.python.io_prefetch import (
    AsyncIOPrefetcher,
    estimate_linear_transform_device_bytes,
    should_prefetch_saved_io,
)
from orion.backend.python.memory_lifecycle import (
    guard_host_memory,
    host_memory_info,
)


_UNIFIED_GROUP_COUNTER = count(1)
_ENCODED_HOIST_PAYLOAD_DATASET = "__encoded_hoist_payload__"


def _unified_compile_workers(item_count: int) -> int:
    raw = os.environ.get("ORION_UNIFIED_COMPILE_WORKERS")
    if raw is None:
        raw = os.environ.get("ORION_REGION_COMPILE_WORKERS", "1")
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        requested = 1
    return max(1, min(int(item_count), int(requested)))


def _unified_stream_compile_batch_limit(item_count: int, workers: int) -> int:
    raw = os.environ.get("ORION_UNIFIED_STREAM_COMPILE_BATCH_TRANSFORMS")
    if raw is None:
        raw = os.environ.get("ORION_UNIFIED_COMPILE_BATCH_TRANSFORMS")
    if raw is None:
        requested = int(workers)
    else:
        try:
            requested = int(raw)
        except (TypeError, ValueError):
            requested = int(workers)
    return max(1, min(int(item_count), int(requested)))


def _unified_stream_compile_batch_bytes() -> int:
    raw = os.environ.get("ORION_UNIFIED_STREAM_COMPILE_BATCH_BYTES")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    raw_gib = os.environ.get("ORION_UNIFIED_STREAM_COMPILE_BATCH_GB", "4")
    try:
        return max(0, int(float(raw_gib) * 1024**3))
    except ValueError:
        return 4 * 1024**3


def _unified_compile_trace_enabled() -> bool:
    return os.environ.get("ORION_UNIFIED_COMPILE_TRACE", "0").strip().lower() not in ("", "0", "false", "no", "off")


def _unified_compile_trace(event: str, **fields: Any) -> None:
    if not _unified_compile_trace_enabled():
        return
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[unified_compile] event={event} {payload}", file=sys.stderr, flush=True)


class UnifiedTransformGroup:
    """Group LinearTransform-like layers that share one input ciphertext."""

    _resident_shared_rotation_keys: dict[int, set[tuple[int, int | None]]] = {}

    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)
        self.unified_ids: list[int] | None = None
        self.is_compiled = False
        self._io_mode = "none"
        self._diags_path = ""
        self._keys_path = ""
        self._storage_key = f"group_{next(_UNIFIED_GROUP_COUNTER)}"
        self._offloaded_plaintext_diagonals = False
        self._diag_indices_by_transform: dict[int, tuple[int, ...]] = {}
        self._required_keys: tuple[tuple[int, int | None], ...] = ()
        self._required_keys_by_transform: dict[int, tuple[tuple[int, int | None], ...]] = {}
        self._io_prefetcher = AsyncIOPrefetcher()
        self._prefetch_host_bytes: int | None = None
        self._prefetch_device_bytes: int | None = None
        self._saved_io_host_bytes_by_transform: dict[int, int] | None = None
        self._resident_plaintext_transform_ids: set[int] = set()
        self.memory_trace: list[dict[str, Any]] = []

    def _scheme_params(self):
        if not self.transforms:
            return None
        scheme = getattr(self.transforms[0], "scheme", None)
        return getattr(scheme, "params", None)

    def _configure_io(self) -> None:
        params = self._scheme_params()
        if params is None:
            self._io_mode = "none"
            self._diags_path = ""
            return

        get_io_mode = getattr(params, "get_io_mode", None)
        get_diags_path = getattr(params, "get_diags_path", None)
        get_keys_path = getattr(params, "get_keys_path", None)
        self._io_mode = (
            str(get_io_mode()).lower()
            if callable(get_io_mode)
            else "none"
        )
        self._diags_path = (
            str(get_diags_path() or "")
            if callable(get_diags_path)
            else ""
        )
        self._keys_path = (
            str(get_keys_path() or "")
            if callable(get_keys_path)
            else ""
        )

    def _should_offload_plaintext_diagonals(self) -> bool:
        return self._io_mode in ("save", "load") and bool(self._diags_path)

    def _should_save_plaintext_diagonals(self) -> bool:
        return self._io_mode == "save" and bool(self._diags_path)

    def _compile_save_resume_enabled(self) -> bool:
        params = self._scheme_params()
        enabled = getattr(params, "get_compile_save_resume", None)
        return self._io_mode == "save" and callable(enabled) and bool(enabled())

    def _should_offload_rotation_keys(self) -> bool:
        return self._io_mode in ("save", "load") and bool(self._keys_path)

    def _should_save_rotation_keys(self) -> bool:
        return self._io_mode == "save" and bool(self._keys_path)

    def _storage_root_name(self) -> str:
        return "__unified_transform_groups__"

    def _rotation_key_storage_name(self, key: int, level: int | None) -> str:
        return str(int(key)) if level is None else f"{int(key)}@{int(level)}"

    def _rotation_key_requests(self, backend, transform_id: int) -> tuple[tuple[int, int | None], ...]:
        get_requests = getattr(backend, "GetLinearTransformRotationKeyRequests", None)
        if callable(get_requests):
            flat = list(get_requests(int(transform_id)))
            if len(flat) % 2 != 0:
                raise RuntimeError("backend returned malformed rotation key requests")
            requests: dict[int, int] = {}
            for index in range(0, len(flat), 2):
                key = int(flat[index])
                level = int(flat[index + 1])
                requests[key] = max(level, requests.get(key, level))
            return tuple(sorted(requests.items()))
        return tuple((int(key), None) for key in backend.GetLinearTransformRotationKeys(int(transform_id)))

    def _generate_rotation_key(self, backend, key: int, level: int | None) -> None:
        if level is not None and hasattr(backend, "GenerateLinearTransformRotationKeyAtLevel"):
            backend.GenerateLinearTransformRotationKeyAtLevel(int(key), int(level))
        else:
            backend.GenerateLinearTransformRotationKey(int(key))

    def _generate_and_serialize_rotation_key(self, backend, key: int, level: int | None):
        if level is not None and hasattr(backend, "GenerateAndSerializeRotationKeyAtLevel"):
            return backend.GenerateAndSerializeRotationKeyAtLevel(int(key), int(level))
        return backend.GenerateAndSerializeRotationKey(int(key))

    def _rotation_key_cached(self, handle, key: int, level: int | None) -> bool:
        key_name = self._rotation_key_storage_name(int(key), level)
        if key_name in handle:
            return True
        return str(int(key)) in handle

    def _ensure_serialized_rotation_keys_available(self, backend) -> None:
        if not self._keys_path:
            return
        with h5py.File(self._keys_path, "a") as handle:
            for key, level in self._required_keys:
                if self._rotation_key_cached(handle, int(key), level):
                    continue
                key_name = self._rotation_key_storage_name(int(key), level)
                serial_key, key_ptr = self._generate_and_serialize_rotation_key(backend, int(key), level)
                try:
                    handle.create_dataset(key_name, data=serial_key)
                finally:
                    backend.FreeCArray(key_ptr)

    def _plaintext_payload_required(self, backend) -> bool:
        return bool(getattr(backend, "load_plaintext_diagonals_requires_payload", True))

    def _encoded_plaintext_payload_supported(self, backend) -> bool:
        return callable(getattr(backend, "SerializeLinearTransformPlaintexts", None)) and callable(
            getattr(backend, "LoadLinearTransformPlaintexts", None)
        )

    def _should_save_encoded_plaintext_payload(self, backend, transform_id: int) -> bool:
        if not self._encoded_plaintext_payload_supported(backend):
            return False
        if self._transform_uses_backend_streaming(backend, int(transform_id)) and not bool(
            getattr(backend, "supports_streaming_encoded_plaintext_payload_cache", False)
        ):
            return False
        override = os.environ.get("ORION_UNIFIED_LT_SAVE_ENCODED_PLAINTEXTS")
        if override is not None:
            enabled = override.lower() not in ("0", "false", "no", "off")
        else:
            enabled = bool(
                self._plaintext_payload_required(backend)
                or getattr(backend, "prefer_encoded_plaintext_payload_cache", False)
            )
        if not enabled:
            return False
        max_bytes = getattr(backend, "encoded_plaintext_payload_max_device_bytes", None)
        env_max = os.environ.get("ORION_UNIFIED_LT_ENCODED_PLAINTEXT_MAX_DEVICE_BYTES")
        if env_max:
            try:
                max_bytes = int(env_max)
            except ValueError:
                max_bytes = None
        if max_bytes is None or int(max_bytes) <= 0:
            return True
        estimated = int(estimate_linear_transform_device_bytes(backend, int(transform_id)))
        return estimated <= int(max_bytes)

    def _device_transform_prefetch_supported(self, backend) -> bool:
        if not bool(getattr(backend, "saved_io_device_prefetch_enabled", False)):
            return False
        return self._encoded_plaintext_payload_supported(backend)

    def _use_shared_rotation_key_map(self, backend, transform_ids: Iterable[int] | None = None) -> bool:
        if not callable(getattr(backend, "LoadRotationKey", None)):
            return False
        override = os.environ.get("ORION_UNIFIED_LT_SHARED_ROTATION_KEYS")
        if override is not None:
            return override.lower() not in ("0", "false", "no", "off")
        ids = [int(value) for value in (transform_ids if transform_ids is not None else (self.unified_ids or []))]
        return len(ids) > 1 and callable(getattr(backend, "LoadLinearTransformRotationKey", None))

    def _rotation_key_residency_enabled(self, backend, transform_ids: Iterable[int] | None = None) -> bool:
        override = os.environ.get("ORION_UNIFIED_LT_ROTKEY_RESIDENCY")
        if override is not None:
            enabled = override.lower() not in ("0", "false", "no", "off")
        else:
            enabled = bool(getattr(backend, "retain_unified_rotation_keys", False))
        return bool(enabled and self._use_shared_rotation_key_map(backend, transform_ids))

    def _rotation_key_residency_watermark_ok(self, backend) -> bool:
        info = self._device_memory_info(backend)
        if info is None:
            return True
        free_bytes = int(info["free_bytes"])
        total_bytes = int(info["total_bytes"])
        explicit = os.environ.get("ORION_UNIFIED_LT_ROTKEY_MIN_FREE_BYTES")
        if explicit:
            try:
                min_free = int(explicit)
            except ValueError:
                min_free = 0
        else:
            min_free = max(8 * 1024**3, int(total_bytes * 0.08))
        return free_bytes >= int(min_free)

    def _plaintext_residency_enabled(self, backend) -> bool:
        override = os.environ.get("ORION_UNIFIED_LT_PLAINTEXT_RESIDENCY")
        if override is not None:
            return override.lower() not in ("0", "false", "no", "off")
        return bool(getattr(backend, "retain_unified_plaintexts", False))

    def _plaintext_residency_min_free_bytes(self, backend) -> int:
        info = self._device_memory_info(backend)
        total_bytes = int(info["total_bytes"]) if info is not None else 0
        explicit = os.environ.get("ORION_UNIFIED_LT_PLAINTEXT_MIN_FREE_BYTES")
        if explicit:
            try:
                return max(0, int(explicit))
            except ValueError:
                return 0
        pct_raw = os.environ.get("ORION_UNIFIED_LT_PLAINTEXT_MIN_FREE_PCT", "12")
        try:
            pct = max(0.0, min(95.0, float(pct_raw)))
        except ValueError:
            pct = 12.0
        pct_bytes = int(total_bytes * pct / 100.0) if total_bytes > 0 else 0
        return max(12 * 1024**3, pct_bytes)

    def _can_keep_plaintexts_resident(self, backend, transform_id: int, *, already_loaded: bool) -> bool:
        if not self._plaintext_residency_enabled(backend):
            return False
        if self._transform_uses_backend_streaming(backend, int(transform_id)):
            info = self._host_memory_info()
            if info is None:
                return True
            available = int(info.get("available_bytes", 0))
            total = int(info.get("total_bytes", 0))
            explicit = os.environ.get("ORION_UNIFIED_LT_HOST_PLAINTEXT_MIN_FREE_BYTES")
            if explicit:
                try:
                    min_free = max(0, int(explicit))
                except ValueError:
                    min_free = 32 * 1024**3
            else:
                min_free = max(32 * 1024**3, int(total * 0.08)) if total > 0 else 0
            if bool(already_loaded):
                return available >= min_free
            estimate = int(self._estimate_saved_io_host_bytes_for_transform(backend, int(transform_id)))
            return available - estimate >= min_free
        info = self._device_memory_info(backend)
        if info is None:
            return False
        free_bytes = int(info["free_bytes"])
        min_free = int(self._plaintext_residency_min_free_bytes(backend))
        if bool(already_loaded):
            return free_bytes >= min_free
        estimate = int(self._estimate_transform_eval_resident_bytes(backend, int(transform_id)))
        return free_bytes - estimate >= min_free

    def _mark_plaintexts_resident(self, transform_id: int) -> None:
        self._resident_plaintext_transform_ids.add(int(transform_id))

    def _clear_plaintexts_resident(self, transform_id: int) -> None:
        self._resident_plaintext_transform_ids.discard(int(transform_id))

    def _resident_rotation_key_set(self, backend) -> set[tuple[int, int | None]]:
        return self._resident_shared_rotation_keys.setdefault(id(backend), set())

    def _clear_resident_rotation_keys(self, backend) -> None:
        self._resident_shared_rotation_keys.pop(id(backend), None)

    def _rotation_key_requests_to_load(
        self,
        backend,
        transform_ids: Iterable[int] | None,
        required_keys: Iterable[tuple[int, int | None]],
    ) -> tuple[tuple[int, int | None], ...]:
        requests = tuple((int(key), None if level is None else int(level)) for key, level in required_keys)
        if not self._rotation_key_residency_enabled(backend, transform_ids):
            return requests
        if not self._rotation_key_residency_watermark_ok(backend):
            return requests
        resident = self._resident_rotation_key_set(backend)
        return tuple((key, level) for key, level in requests if (int(key), level) not in resident)

    def _mark_rotation_keys_resident(
        self,
        backend,
        transform_ids: Iterable[int] | None,
        key_requests: Iterable[tuple[int, int | None]],
    ) -> None:
        if not self._rotation_key_residency_enabled(backend, transform_ids):
            return
        resident = self._resident_rotation_key_set(backend)
        for key, level in key_requests:
            key = int(key)
            # Cheddar's shared rotation-key map stores one EVK per rotation
            # index. A different level for the same rotation overwrites the
            # previous EVK, so the residency model must be exact by level.
            for existing in tuple(resident):
                if int(existing[0]) == key:
                    resident.discard(existing)
            resident.add((key, None if level is None else int(level)))

    def _consume_trim_seconds(self, backend) -> float:
        consume = getattr(backend, "ConsumeDeviceMemoryTrimSeconds", None)
        if not callable(consume):
            return 0.0
        try:
            return float(consume())
        except Exception:
            return 0.0

    def _consume_shared_cache_eval_profile(self, backend) -> dict[str, float]:
        consume = getattr(backend, "ConsumeSharedCacheEvalProfileSeconds", None)
        if not callable(consume):
            return {}
        try:
            values = list(consume())
        except Exception:
            return {}
        names = (
            (
                "cpp_plan_s",
                "cpp_level_adjust_s",
                "cpp_baby_step_s",
                "cpp_giant_step_s",
                "cpp_push_s",
                "cpp_trim_s",
            )
            if len(values) <= 6
            else (
                "cpp_plan_s",
                "cpp_level_adjust_s",
                "cpp_baby_step_s",
                "cpp_giant_step_s",
                "stream_build_map_s",
                "stream_encode_hoist_s",
                "stream_load_payload_s",
                "stream_eval_s",
                "stream_accumulate_s",
                "cpp_push_s",
                "cpp_trim_s",
            )
        )
        return {name: float(values[index]) for index, name in enumerate(names) if index < len(values)}

    def _release_source_diagonals_after_compile_enabled(self) -> bool:
        override = os.environ.get("ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE")
        if override is not None:
            return override.lower() not in ("0", "false", "no", "off")
        return self._io_mode in ("save", "load")

    def _release_index_only_backend_matrix_after_save_enabled(self) -> bool:
        return os.environ.get(
            "ORION_UNIFIED_LT_RELEASE_INDEX_ONLY_RAW_MATRICES_AFTER_SAVE",
            "0",
        ).lower() not in ("0", "false", "no", "off")

    def _force_compile_trim_each_transform_enabled(self) -> bool:
        return os.environ.get(
            "ORION_UNIFIED_LT_FORCE_COMPILE_TRIM_EACH_TRANSFORM",
            "0",
        ).strip().lower() not in ("", "0", "false", "no", "off")

    def _release_backend_matrix_after_save(
        self,
        backend,
        transform_id: int,
        *,
        encoded_payload_saved: bool = False,
        index_only_payload_saved: bool = False,
    ) -> None:
        release = getattr(backend, "ReleaseLinearTransformMatrix", None)
        if not callable(release):
            return
        if not bool(encoded_payload_saved) and not (
            bool(index_only_payload_saved)
            and self._release_index_only_backend_matrix_after_save_enabled()
        ):
            return
        try:
            release(int(transform_id))
        except Exception:
            return

    def _clear_source_diagonals_after_compile(self, transform) -> None:
        if not self._release_source_diagonals_after_compile_enabled():
            return
        try:
            getattr(transform, "diagonals", {}).clear()
        except Exception:
            pass

    def _saved_io_prefetch_key(self):
        return ("unified", str(self._storage_key))

    def _device_memory_info(self, backend) -> dict[str, int] | None:
        get_memory_info = getattr(backend, "GetDeviceMemoryInfo", None)
        if not callable(get_memory_info):
            return None
        values = list(get_memory_info())
        if len(values) < 2:
            return None
        return {
            "free_bytes": int(values[0]),
            "total_bytes": int(values[1]),
        }

    def _transform_uses_backend_streaming(self, backend, transform_id: int) -> bool:
        uses_streaming = getattr(backend, "LinearTransformUsesStreaming", None)
        if not callable(uses_streaming):
            return False
        return bool(uses_streaming(int(transform_id)))

    def _plaintext_load_transform_ids(
        self,
        backend,
        transform_ids: Iterable[int] | None = None,
    ) -> set[int]:
        selected_ids = (
            {int(value) for value in transform_ids}
            if transform_ids is not None
            else set(int(value) for value in self._diag_indices_by_transform)
        )
        return {
            int(transform_id)
            for transform_id in selected_ids
            if (
                not self._transform_uses_backend_streaming(backend, int(transform_id))
                or bool(getattr(backend, "supports_streaming_encoded_plaintext_payload_cache", False))
            )
        }

    def _streaming_payload_missing_error(self, transform_id: int) -> RuntimeError:
        return RuntimeError(
            "backend streaming linear transform requires an encoded plaintext payload "
            f"in saved IO cache for transform {int(transform_id)}"
        )

    def _estimate_transform_device_bytes_summary(
        self,
        backend,
        transform_ids: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        ids = [int(value) for value in (transform_ids if transform_ids is not None else (self.unified_ids or []))]
        estimates: list[dict[str, int]] = []
        for transform_id in ids:
            saved_host_bytes = int(self._estimate_saved_io_host_bytes_for_transform(backend, int(transform_id)))
            estimates.append(
                {
                    "transform_id": int(transform_id),
                    "device_bytes": int(self._estimate_transform_eval_resident_bytes(backend, int(transform_id))),
                    "backend_device_bytes": int(estimate_linear_transform_device_bytes(backend, int(transform_id))),
                    "saved_host_bytes": int(saved_host_bytes),
                }
            )
        estimates.sort(key=lambda item: int(item["device_bytes"]), reverse=True)
        total = int(sum(int(item["device_bytes"]) for item in estimates))
        saved_total = int(sum(int(item["saved_host_bytes"]) for item in estimates))
        return {
            "transform_count": int(len(estimates)),
            "device_bytes_total": int(total),
            "device_bytes_max": int(estimates[0]["device_bytes"]) if estimates else 0,
            "saved_host_bytes_total": int(saved_total),
            "saved_host_bytes_max": int(estimates[0]["saved_host_bytes"]) if estimates else 0,
            "top_transforms": estimates[:16],
        }

    def _host_memory_info(self) -> dict[str, int] | None:
        return host_memory_info()

    def _runtime_memory_stats(self, backend) -> dict[str, int] | None:
        get_stats = getattr(backend, "GetRuntimeMemoryStats", None)
        if not callable(get_stats):
            return None
        try:
            values = [int(value) for value in list(get_stats())]
        except Exception:
            return None
        names = (
            "alloc_bytes",
            "total_alloc_bytes",
            "sys_bytes",
            "heap_alloc_bytes",
            "heap_sys_bytes",
            "heap_idle_bytes",
            "heap_released_bytes",
            "heap_inuse_bytes",
            "stack_inuse_bytes",
            "mspan_inuse_bytes",
            "mcache_inuse_bytes",
            "num_gc",
        )
        return {name: int(values[index]) for index, name in enumerate(names) if index < len(values)}

    def _forward_memory_guard(
        self,
        backend,
        *,
        reason: str,
        transform_ids: Iterable[int] | None = None,
        needed_bytes: int = 0,
        force_trim: bool = False,
        raise_on_low: bool = True,
        stream_plaintexts: bool = False,
    ) -> dict[str, Any]:
        ids = [int(value) for value in (transform_ids if transform_ids is not None else ())]
        estimated = int(max(0, int(needed_bytes or 0)))
        if not estimated and ids:
            estimated = int(
                sum(
                    int(
                        self._estimate_transform_load_resident_bytes(
                            backend,
                            int(transform_id),
                            stream_plaintexts=bool(stream_plaintexts),
                        )
                    )
                    for transform_id in ids
                )
            )
        event = guard_host_memory(
            backend,
            reason=str(reason),
            needed_bytes=int(estimated),
            force_trim=bool(force_trim),
            raise_on_low=bool(raise_on_low),
        )
        runtime_stats = self._runtime_memory_stats(backend)
        if runtime_stats is not None:
            event["runtime_memory"] = runtime_stats
        return event

    def _host_eval_budget_bytes(self) -> int:
        info = self._host_memory_info()
        if info is None:
            return 0
        total = int(info["total_bytes"])
        available = int(info["available_bytes"])
        reserve_env = os.environ.get("ORION_UNIFIED_LT_HOST_EVAL_RESERVE_BYTES")
        if reserve_env:
            try:
                reserve = int(reserve_env)
            except ValueError:
                reserve = 0
        else:
            fraction = 0.25
            fraction_env = os.environ.get("ORION_UNIFIED_LT_HOST_EVAL_RESERVE_FRACTION")
            if fraction_env:
                try:
                    fraction = max(0.0, min(0.95, float(fraction_env)))
                except ValueError:
                    fraction = 0.25
            reserve = max(64 * 1024**3, int(total * float(fraction)))
        return max(1, int(available - reserve))

    def _saved_io_host_bytes_by_transform_map(self, backend) -> dict[int, int]:
        if self._saved_io_host_bytes_by_transform is not None:
            return dict(self._saved_io_host_bytes_by_transform)
        result: dict[int, int] = {}
        if not self._offloaded_plaintext_diagonals:
            self._saved_io_host_bytes_by_transform = result
            return result
        try:
            handle, root = self._storage_group("r")
        except Exception:
            self._saved_io_host_bytes_by_transform = result
            return result
        try:
            storage = root[self._storage_key]
            for transform_id in self._plaintext_load_transform_ids(backend):
                total = 0
                transform_group = storage.get(str(int(transform_id)))
                if transform_group is None:
                    continue
                if _ENCODED_HOIST_PAYLOAD_DATASET in transform_group:
                    dataset = transform_group[_ENCODED_HOIST_PAYLOAD_DATASET]
                    total += int(dataset.size) * int(dataset.dtype.itemsize)
                else:
                    for name in ("diag_payload", "diag_offsets", "diag_lengths", "diag_indices"):
                        if name not in transform_group:
                            continue
                        dataset = transform_group[name]
                        total += int(dataset.size) * int(dataset.dtype.itemsize)
                result[int(transform_id)] = int(total)
        finally:
            handle.close()
        self._saved_io_host_bytes_by_transform = dict(result)
        return result

    def _estimate_saved_io_host_bytes_for_transform(self, backend, transform_id: int) -> int:
        return int(self._saved_io_host_bytes_by_transform_map(backend).get(int(transform_id), 0))

    def _saved_payload_resident_multiplier(self) -> float:
        raw = os.environ.get("ORION_UNIFIED_LT_SAVED_PAYLOAD_RESIDENT_MULTIPLIER", "1.35")
        try:
            return max(1.0, float(raw))
        except ValueError:
            return 1.35

    def _stream_plaintext_diag_load_enabled(self, backend) -> bool:
        if not self._memory_bounded_eval_enabled(backend):
            return False
        if not self._plaintext_payload_required(backend):
            return False
        raw = os.environ.get("ORION_UNIFIED_LT_STREAM_LOAD_PLAINTEXTS", "1")
        return raw.strip().lower() not in ("0", "false", "no", "off")

    def _stream_plaintext_diag_load_chunk_bytes(self) -> int:
        raw = os.environ.get("ORION_UNIFIED_LT_STREAM_LOAD_CHUNK_BYTES")
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        raw_gib = os.environ.get("ORION_UNIFIED_LT_STREAM_LOAD_CHUNK_GB", "0.5")
        try:
            return max(1, int(float(raw_gib) * 1024**3))
        except ValueError:
            return 512 * 1024**2

    def _stream_plaintext_diag_save_chunk_bytes(self) -> int:
        raw = os.environ.get("ORION_UNIFIED_LT_STREAM_SAVE_CHUNK_BYTES")
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        raw_mib = os.environ.get("ORION_UNIFIED_LT_STREAM_SAVE_CHUNK_MB", "16")
        try:
            return max(1, int(float(raw_mib) * 1024**2))
        except ValueError:
            return 16 * 1024**2

    def _estimate_transform_eval_resident_bytes(self, backend, transform_id: int) -> int:
        backend_estimate = int(estimate_linear_transform_device_bytes(backend, int(transform_id)))
        if backend_estimate > 0:
            return int(backend_estimate)
        saved_bytes = int(self._estimate_saved_io_host_bytes_for_transform(backend, int(transform_id)))
        if saved_bytes > 0:
            return max(1, int(saved_bytes * self._saved_payload_resident_multiplier()))
        return 1

    def _estimate_transform_load_resident_bytes(
        self,
        backend,
        transform_id: int,
        *,
        stream_plaintexts: bool = False,
    ) -> int:
        saved_bytes = int(self._estimate_saved_io_host_bytes_for_transform(backend, int(transform_id)))
        load_bytes = int(saved_bytes)
        if bool(stream_plaintexts) and saved_bytes > 0:
            load_bytes = min(saved_bytes, int(self._stream_plaintext_diag_load_chunk_bytes()))
        return int(self._estimate_transform_eval_resident_bytes(backend, int(transform_id))) + int(load_bytes)

    def _record_memory_event(
        self,
        event: str,
        backend,
        transform_ids: Iterable[int] | None = None,
        **extra: Any,
    ) -> None:
        payload: dict[str, Any] = {"event": str(event)}
        host_info = self._host_memory_info()
        if host_info is not None:
            payload["host_memory"] = host_info
        memory_info = self._device_memory_info(backend)
        if memory_info is not None:
            payload["device_memory"] = memory_info
        runtime_stats = self._runtime_memory_stats(backend)
        if runtime_stats is not None:
            payload["runtime_memory"] = runtime_stats
        payload["linear_transform_device_bytes"] = self._estimate_transform_device_bytes_summary(
            backend,
            transform_ids,
        )
        if self._resident_plaintext_transform_ids:
            payload["resident_plaintext_transform_ids"] = sorted(
                int(value) for value in self._resident_plaintext_transform_ids
            )
        payload.update(extra)
        self.memory_trace.append(payload)

    def _memory_bounded_compile_enabled(self, backend) -> bool:
        if not bool(getattr(backend, "memory_bounded_unified_transforms", False)):
            return False
        if self._should_save_plaintext_diagonals():
            return True
        # Saved plaintext payloads are tied to how the backend constructs the
        # unified transform. If save mode compiled one transform at a time,
        # load mode must do the same before reloading those payloads.
        return self._io_mode == "load" and bool(self._diags_path)

    def _memory_bounded_eval_enabled(self, backend) -> bool:
        return (
            bool(getattr(backend, "memory_bounded_unified_evaluate", False))
            and (self._offloaded_plaintext_diagonals or self._should_offload_rotation_keys())
        )

    def _eval_budget_bytes(self, backend) -> int:
        explicit_budget = getattr(backend, "unified_transform_eval_budget_bytes", None)
        if explicit_budget is not None:
            try:
                return max(1, int(explicit_budget))
            except (TypeError, ValueError):
                pass
        memory_info = self._device_memory_info(backend)
        if memory_info is None:
            return int(self._host_eval_budget_bytes())
        free_bytes = int(memory_info["free_bytes"])
        total_bytes = int(memory_info["total_bytes"])
        reserve = max(16 * 1024**3, int(total_bytes * 0.20))
        return max(1, int(free_bytes - reserve))

    def _memory_bounded_chunks(self, backend) -> list[list[int]]:
        ids = [int(value) for value in (self.unified_ids or [])]
        if not ids:
            return []
        if all(self._transform_uses_backend_streaming(backend, int(transform_id)) for transform_id in ids):
            return [ids]
        budget = int(self._eval_budget_bytes(backend))
        if budget <= 0:
            return [ids]
        chunks: list[list[int]] = []
        current: list[int] = []
        current_bytes = 0
        for transform_id in ids:
            estimate = max(1, int(self._estimate_transform_eval_resident_bytes(backend, int(transform_id))))
            if current and current_bytes + estimate > budget:
                chunks.append(current)
                current = []
                current_bytes = 0
            current.append(int(transform_id))
            current_bytes += int(estimate)
        if current:
            chunks.append(current)
        return chunks

    def _required_keys_for_transform_ids(
        self,
        transform_ids: Iterable[int] | None,
    ) -> tuple[tuple[int, int | None], ...]:
        if transform_ids is None:
            return self._required_keys
        required: dict[int, int | None] = {}
        for transform_id in transform_ids:
            for key, level in self._required_keys_by_transform.get(int(transform_id), ()):
                if level is None:
                    required[int(key)] = None
                else:
                    current = required.get(int(key))
                    required[int(key)] = int(level) if current is None else max(int(level), int(current))
        return tuple(sorted((int(key), level) for key, level in required.items()))

    def _shared_saved_io_scheduler(self):
        if not self.transforms:
            return None
        scheme = getattr(self.transforms[0], "scheme", None)
        scheduler = getattr(scheme, "lt_evaluator", None)
        if scheduler is None:
            return None
        if not callable(getattr(scheduler, "register_saved_io_prefetch_work_unit", None)):
            return None
        if not callable(getattr(scheduler, "fill_saved_io_prefetch_window", None)):
            return None
        if not callable(getattr(scheduler, "consume_saved_io_prefetch", None)):
            return None
        return scheduler

    def _register_shared_saved_io_work_unit(self, backend) -> None:
        if self._memory_bounded_eval_enabled(backend):
            return
        scheduler = self._shared_saved_io_scheduler()
        if scheduler is None:
            return
        scheduler.register_saved_io_prefetch_work_unit(
            self._saved_io_prefetch_key(),
            loader=lambda: self._read_and_prefetch_saved_io_bundle(backend),
            host_bytes=lambda: self._estimate_prefetch_host_bytes(backend),
            device_bytes=lambda: self._estimate_prefetch_device_bytes(backend),
        )

    def _prepare_shared_cache_plans(self, backend) -> None:
        if os.environ.get("ORION_UNIFIED_LT_PREPARE_SHARED_CACHE_PLAN", "1").lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return
        prepare_plan = getattr(backend, "PrepareLinearTransformsSharedCachePlan", None)
        if not callable(prepare_plan) or not self.unified_ids:
            return
        if os.environ.get("ORION_CHEDDAR_SHARED_CACHE_PLAN_PERSIST", "").lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return
        chunks = (
            self._memory_bounded_chunks(backend)
            if self._memory_bounded_eval_enabled(backend)
            else [[int(value) for value in self.unified_ids]]
        )
        for chunk_index, chunk_ids in enumerate(chunks):
            chunk_ids = [int(value) for value in chunk_ids]
            load_started = time.perf_counter()
            required_keys = self._rotation_key_requests_to_load(
                backend,
                chunk_ids,
                self._required_keys_for_transform_ids(chunk_ids),
            )
            self._load_rotation_keys(
                backend,
                None,
                transform_ids=chunk_ids,
                required_keys=required_keys,
            )
            self._load_plaintext_diagonals(backend, None, transform_ids=chunk_ids)
            plaintexts_stable = True
            for transform_id in self._plaintext_load_transform_ids(backend, chunk_ids):
                if int(transform_id) in self._resident_plaintext_transform_ids:
                    continue
                if self._can_keep_plaintexts_resident(
                    backend,
                    int(transform_id),
                    already_loaded=True,
                ):
                    self._mark_plaintexts_resident(int(transform_id))
                else:
                    plaintexts_stable = False
            load_s = float(time.perf_counter() - load_started)
            if not plaintexts_stable:
                self._record_memory_event(
                    "skip_prepare_shared_cache_plan",
                    backend,
                    chunk_ids,
                    chunk_index=int(chunk_index),
                    chunk_transform_count=int(len(chunk_ids)),
                    timing={"prepare_load_s": load_s},
                    reason="plaintexts_not_resident_after_prepare_load",
                )
                continue
            if len(chunk_ids) <= 1:
                self._record_memory_event(
                    "after_prepare_shared_cache_plan",
                    backend,
                    chunk_ids,
                    chunk_index=int(chunk_index),
                    chunk_transform_count=int(len(chunk_ids)),
                    timing={
                        "prepare_load_s": load_s,
                        "prepare_shared_cache_plan_s": 0.0,
                    },
                    reason="single_transform_preload_only",
                )
                continue
            transform_ids_array = (ctypes.c_int * len(chunk_ids))(*chunk_ids)
            prepare_started = time.perf_counter()
            prepare_plan(transform_ids_array, len(chunk_ids))
            prepare_s = float(time.perf_counter() - prepare_started)
            self._record_memory_event(
                "after_prepare_shared_cache_plan",
                backend,
                chunk_ids,
                chunk_index=int(chunk_index),
                chunk_transform_count=int(len(chunk_ids)),
                timing={
                    "prepare_load_s": load_s,
                    "prepare_shared_cache_plan_s": prepare_s,
                },
            )

    def _read_and_prefetch_saved_io_bundle(self, backend) -> dict[str, object] | None:
        bundle = self._read_saved_io_bundle(backend, prefetch=False)
        if bundle is not None and self._device_transform_prefetch_supported(backend):
            self._prefetch_saved_io_bundle_to_device(backend, bundle)
        return bundle

    def _prefetch_saved_io_bundle_to_device(self, backend, bundle: dict[str, object]) -> None:
        load_transform_key = getattr(backend, "LoadLinearTransformRotationKey", None)
        # The shared Cheddar/Lattigo key map is process-global. Loading
        # rotation keys, and the associated plaintext hoist payloads, from the
        # async prefetch thread can overlap the current group's GPU residency.
        # Keep unified-provider prefetch host-only; eval loads to device inside
        # the memory-bounded critical section.
        if self._use_shared_rotation_key_map(backend):
            return
        if callable(load_transform_key) and self.unified_ids is not None:
            for key, serial_key in bundle.get("rotation_keys", ()):
                for transform_id in self.unified_ids:
                    load_transform_key(serial_key, int(key), int(transform_id))
            if bundle.get("rotation_keys"):
                bundle["rotation_keys_prefetched_to_device"] = True
                bundle["rotation_keys"] = ()

        if not callable(getattr(backend, "LoadLinearTransformPlaintexts", None)):
            return
        for transform_id, payload in bundle.get("plaintexts", {}).items():
            encoded_payload = payload.get("encoded_payload")
            if encoded_payload is not None:
                backend.LoadLinearTransformPlaintexts(
                    encoded_payload,
                    int(transform_id),
                )
                payload["encoded_payload"] = None
                payload["plaintexts_prefetched_to_device"] = True

    def _estimate_prefetch_host_bytes(self, backend) -> int:
        if self._prefetch_host_bytes is not None:
            return int(self._prefetch_host_bytes)

        total_bytes = 0
        if self._should_offload_rotation_keys():
            with h5py.File(self._keys_path, "r") as handle:
                for key, level in self._required_keys:
                    key_name = self._rotation_key_storage_name(int(key), level)
                    if key_name not in handle and str(int(key)) in handle:
                        key_name = str(int(key))
                    dataset = handle[key_name]
                    total_bytes += int(dataset.size) * int(dataset.dtype.itemsize)

        if self._offloaded_plaintext_diagonals and self._plaintext_payload_required(backend):
            handle, root = self._storage_group("r")
            try:
                storage = root[self._storage_key]
                for transform_id in self._plaintext_load_transform_ids(backend):
                    transform_group = storage[str(transform_id)]
                    if _ENCODED_HOIST_PAYLOAD_DATASET in transform_group:
                        dataset = transform_group[_ENCODED_HOIST_PAYLOAD_DATASET]
                        total_bytes += int(dataset.size) * int(dataset.dtype.itemsize)
                    else:
                        for name in ("diag_payload", "diag_offsets", "diag_lengths", "diag_indices"):
                            dataset = transform_group[name]
                            total_bytes += int(dataset.size) * int(dataset.dtype.itemsize)
            finally:
                handle.close()
        elif self._offloaded_plaintext_diagonals and self._encoded_plaintext_payload_supported(backend):
            handle, root = self._storage_group("r")
            try:
                storage = root[self._storage_key]
                for transform_id in self._plaintext_load_transform_ids(backend):
                    transform_group = storage[str(transform_id)]
                    if _ENCODED_HOIST_PAYLOAD_DATASET in transform_group:
                        dataset = transform_group[_ENCODED_HOIST_PAYLOAD_DATASET]
                        total_bytes += int(dataset.size) * int(dataset.dtype.itemsize)
            finally:
                handle.close()

        self._prefetch_host_bytes = int(total_bytes)
        return int(total_bytes)

    def _estimate_prefetch_device_bytes(self, backend) -> int:
        if self._prefetch_device_bytes is not None:
            return int(self._prefetch_device_bytes)

        total_bytes = 0
        if self._should_offload_rotation_keys() and not self._use_shared_rotation_key_map(backend):
            total_bytes += self._estimate_prefetch_host_bytes(backend)
        if self._offloaded_plaintext_diagonals:
            for transform_id in self._plaintext_load_transform_ids(backend):
                total_bytes += estimate_linear_transform_device_bytes(backend, int(transform_id))
        self._prefetch_device_bytes = int(total_bytes)
        return int(total_bytes)

    def _read_saved_io_bundle(
        self,
        backend,
        *,
        prefetch: bool,
        transform_ids: Iterable[int] | None = None,
        required_keys: Iterable[tuple[int, int | None]] | None = None,
        include_plaintexts: bool = True,
    ) -> dict[str, object] | None:
        if prefetch:
            if not should_prefetch_saved_io(
                self._estimate_prefetch_host_bytes(backend),
                backend=backend,
                device_bytes=self._estimate_prefetch_device_bytes(backend),
            ):
                return None

        bundle: dict[str, object] = {
            "rotation_keys": (),
            "plaintexts": {},
        }

        if self._should_offload_rotation_keys():
            rotation_keys = []
            key_requests = tuple(required_keys) if required_keys is not None else self._required_keys
            with h5py.File(self._keys_path, "r") as handle:
                for key, level in key_requests:
                    key_name = self._rotation_key_storage_name(int(key), level)
                    if key_name not in handle and str(int(key)) in handle:
                        key_name = str(int(key))
                    rotation_keys.append((int(key), np.asarray(handle[key_name][()], dtype=np.uint8)))
            bundle["rotation_keys"] = tuple(rotation_keys)

        if bool(include_plaintexts) and self._offloaded_plaintext_diagonals:
            selected_ids = self._plaintext_load_transform_ids(backend, transform_ids)
            selected_ids -= set(int(value) for value in self._resident_plaintext_transform_ids)
            handle, root = self._storage_group("r")
            try:
                storage = root[self._storage_key]
                plaintexts: dict[int, dict[str, object]] = {}
                for transform_id, diag_indices in self._diag_indices_by_transform.items():
                    if int(transform_id) not in selected_ids:
                        continue
                    transform_group = storage[str(transform_id)]
                    if _ENCODED_HOIST_PAYLOAD_DATASET in transform_group:
                        plaintexts[int(transform_id)] = {
                            "encoded_payload": np.asarray(
                                transform_group[_ENCODED_HOIST_PAYLOAD_DATASET][()],
                                dtype=np.uint8,
                            ).reshape(-1),
                            "payload": np.zeros((0,), dtype=np.uint8),
                            "offsets": (),
                            "lengths": (),
                            "diag_indices": tuple(int(idx) for idx in diag_indices),
                        }
                        continue
                    if self._transform_uses_backend_streaming(backend, int(transform_id)):
                        raise self._streaming_payload_missing_error(int(transform_id))
                    if not self._plaintext_payload_required(backend):
                        plaintexts[int(transform_id)] = {
                            "payload": np.zeros((0,), dtype=np.uint8),
                            "offsets": (),
                            "lengths": (),
                            "diag_indices": tuple(int(idx) for idx in diag_indices),
                        }
                        continue
                    plaintexts[int(transform_id)] = {
                        "payload": np.asarray(transform_group["diag_payload"][:], dtype=np.uint8),
                        "offsets": tuple(int(v) for v in transform_group["diag_offsets"][:].tolist()),
                        "lengths": tuple(int(v) for v in transform_group["diag_lengths"][:].tolist()),
                        "diag_indices": tuple(int(v) for v in diag_indices),
                    }
                bundle["plaintexts"] = plaintexts
            finally:
                handle.close()

        return bundle

    def _schedule_next_saved_io_prefetch(self, backend) -> None:
        if not self._should_offload_rotation_keys() and not self._offloaded_plaintext_diagonals:
            return
        if not should_prefetch_saved_io(
            self._estimate_prefetch_host_bytes(backend),
            backend=backend,
            device_bytes=self._estimate_prefetch_device_bytes(backend),
        ):
            return
        self._io_prefetcher.submit(
            self._storage_key,
            lambda: self._read_and_prefetch_saved_io_bundle(backend),
            host_bytes=self._estimate_prefetch_host_bytes(backend),
            device_bytes=self._estimate_prefetch_device_bytes(backend),
        )

    def _load_rotation_keys(
        self,
        backend,
        bundle: dict[str, object] | None = None,
        *,
        transform_ids: Iterable[int] | None = None,
        required_keys: Iterable[tuple[int, int | None]] | None = None,
    ) -> None:
        if not self._should_offload_rotation_keys():
            return
        target_ids = [int(value) for value in (transform_ids if transform_ids is not None else (self.unified_ids or []))]
        key_requests = tuple(required_keys) if required_keys is not None else self._required_keys
        load_transform_key = getattr(backend, "LoadLinearTransformRotationKey", None)
        use_shared_keys = self._use_shared_rotation_key_map(backend, target_ids)
        if use_shared_keys:
            key_requests = self._rotation_key_requests_to_load(backend, target_ids, key_requests)
            if not key_requests:
                return
        if bundle is not None:
            if bundle.get("rotation_keys_prefetched_to_device"):
                return
            serial_by_key = {
                int(key): serial_key
                for key, serial_key in bundle.get("rotation_keys", ())
            }
            if use_shared_keys:
                loaded_requests: list[tuple[int, int | None]] = []
                for key, _level in key_requests:
                    serial_key = serial_by_key.get(int(key))
                    if serial_key is not None:
                        backend.LoadRotationKey(serial_key, int(key))
                        loaded_requests.append((int(key), _level))
                self._mark_rotation_keys_resident(backend, target_ids, loaded_requests)
                bundle["rotation_keys"] = ()
                return
            if callable(load_transform_key) and target_ids:
                first_transform_id = int(target_ids[0])
                for key, _level in key_requests:
                    serial_key = serial_by_key.get(int(key))
                    if serial_key is not None:
                        load_transform_key(serial_key, int(key), first_transform_id)
                for transform_id in target_ids[1:]:
                    for key, _level in self._required_keys_for_transform_ids((int(transform_id),)):
                        serial_key = serial_by_key.get(int(key))
                        if serial_key is not None:
                            load_transform_key(serial_key, int(key), int(transform_id))
                bundle["rotation_keys"] = ()
                return
            for key, serial_key in bundle.get("rotation_keys", ()):
                backend.LoadRotationKey(serial_key, int(key))
            bundle["rotation_keys"] = ()
            return
        with h5py.File(self._keys_path, "r") as handle:
            if use_shared_keys:
                loaded_requests: list[tuple[int, int | None]] = []
                for key, level in key_requests:
                    key_name = self._rotation_key_storage_name(int(key), level)
                    if key_name not in handle and str(int(key)) in handle:
                        key_name = str(int(key))
                    backend.LoadRotationKey(handle[key_name][()], int(key))
                    loaded_requests.append((int(key), level))
                self._mark_rotation_keys_resident(backend, target_ids, loaded_requests)
                return
            if callable(load_transform_key) and target_ids:
                first_transform_id = int(target_ids[0])
                for key, level in key_requests:
                    key_name = self._rotation_key_storage_name(int(key), level)
                    if key_name not in handle and str(int(key)) in handle:
                        key_name = str(int(key))
                    load_transform_key(handle[key_name][()], int(key), first_transform_id)
                for transform_id in target_ids[1:]:
                    for key, level in self._required_keys_for_transform_ids((int(transform_id),)):
                        key_name = self._rotation_key_storage_name(int(key), level)
                        if key_name not in handle and str(int(key)) in handle:
                            key_name = str(int(key))
                        load_transform_key(handle[key_name][()], int(key), int(transform_id))
                return
            for key, level in key_requests:
                key_name = self._rotation_key_storage_name(int(key), level)
                if key_name not in handle and str(int(key)) in handle:
                    key_name = str(int(key))
                backend.LoadRotationKey(handle[key_name][()], int(key))

    def _unload_rotation_keys(self, backend, *, transform_ids: Iterable[int] | None = None) -> None:
        if not self._should_offload_rotation_keys():
            return
        remove_transform_keys = getattr(backend, "RemoveLinearTransformRotationKeys", None)
        target_ids = [int(value) for value in (transform_ids if transform_ids is not None else (self.unified_ids or []))]
        if self._use_shared_rotation_key_map(backend, target_ids):
            if self._rotation_key_residency_enabled(backend, target_ids) and self._rotation_key_residency_watermark_ok(backend):
                return
            backend.RemoveRotationKeys()
            self._clear_resident_rotation_keys(backend)
            return
        if callable(remove_transform_keys) and target_ids:
            for transform_id in target_ids:
                remove_transform_keys(int(transform_id))
            return
        backend.RemoveRotationKeys()

    def _storage_group(self, mode: str):
        if not self._diags_path:
            raise ValueError(
                "UnifiedTransformGroup with io_mode='save' requires "
                "'orion.diags_path' to be set."
            )
        handle = h5py.File(self._diags_path, mode)
        root = handle.require_group(self._storage_root_name())
        return handle, root

    def _save_and_unload_plaintext_diagonals(self, backend) -> None:
        if self.unified_ids is None:
            raise RuntimeError("UnifiedTransformGroup must be compiled before saving diagonals")

        handle, root = self._storage_group("a")
        try:
            if self._storage_key in root:
                del root[self._storage_key]
            storage = root.create_group(self._storage_key)
            for transform_id, diag_indices in self._diag_indices_by_transform.items():
                self._save_and_unload_plaintext_diagonals_for_transform(
                    backend,
                    storage,
                    int(transform_id),
                    tuple(int(idx) for idx in diag_indices),
                )
        finally:
            handle.close()
        self._offloaded_plaintext_diagonals = True
        self._prefetch_host_bytes = None
        self._prefetch_device_bytes = None
        self._saved_io_host_bytes_by_transform = None

    def _save_and_unload_plaintext_diagonals_for_transform(
        self,
        backend,
        storage,
        transform_id: int,
        diag_indices: Iterable[int],
    ) -> None:
        diag_indices = tuple(int(idx) for idx in diag_indices)
        if str(int(transform_id)) in storage:
            del storage[str(int(transform_id))]
        transform_group = storage.create_group(str(int(transform_id)))
        if self._should_save_encoded_plaintext_payload(backend, int(transform_id)):
            serial_payload, payload_ptr = backend.SerializeLinearTransformPlaintexts(
                int(transform_id)
            )
            try:
                transform_group.create_dataset(
                    _ENCODED_HOIST_PAYLOAD_DATASET,
                    data=serial_payload,
                )
            finally:
                backend.FreeCArray(payload_ptr)
            transform_group.create_dataset(
                "diag_indices",
                data=np.asarray(diag_indices, dtype=np.int32),
            )
            if self._can_keep_plaintexts_resident(
                backend,
                int(transform_id),
                already_loaded=True,
            ):
                self._mark_plaintexts_resident(int(transform_id))
                return
            self._release_backend_matrix_after_save(
                backend,
                int(transform_id),
                encoded_payload_saved=True,
            )
            return

        if not self._plaintext_payload_required(backend):
            transform_group.create_dataset(
                "diag_indices",
                data=np.asarray(diag_indices, dtype=np.int32),
            )
            transform_group.create_dataset("diag_offsets", data=np.zeros((0,), dtype=np.uint64))
            transform_group.create_dataset("diag_lengths", data=np.zeros((0,), dtype=np.uint64))
            transform_group.create_dataset("diag_payload", data=np.zeros((0,), dtype=np.uint8))
            if self._can_keep_plaintexts_resident(
                backend,
                int(transform_id),
                already_loaded=False,
            ):
                backend.LoadPlaintextDiagonalsBatch(
                    np.zeros((0,), dtype=np.uint8),
                    [],
                    [],
                    list(diag_indices),
                    int(transform_id),
                )
                self._mark_plaintexts_resident(int(transform_id))
                return
            backend.RemovePlaintextDiagonals(int(transform_id))
            self._release_backend_matrix_after_save(
                backend,
                int(transform_id),
                index_only_payload_saved=True,
            )
            return

        offsets: list[int] = []
        lengths: list[int] = []
        cursor = 0
        payload_ds = transform_group.create_dataset(
            "diag_payload",
            shape=(0,),
            maxshape=(None,),
            chunks=(int(self._stream_plaintext_diag_save_chunk_bytes()),),
            dtype=np.uint8,
        )
        for diag_idx in diag_indices:
            serial_diag, diag_ptr = backend.SerializeDiagonal(
                int(transform_id),
                int(diag_idx),
            )
            try:
                serial_arr = np.asarray(serial_diag, dtype=np.uint8).reshape(-1)
                length = int(serial_arr.size)
                offsets.append(int(cursor))
                lengths.append(int(length))
                if length:
                    next_cursor = int(cursor + length)
                    payload_ds.resize((next_cursor,))
                    payload_ds[int(cursor):next_cursor] = serial_arr
                    cursor = int(next_cursor)
            finally:
                backend.FreeCArray(diag_ptr)
        transform_group.create_dataset(
            "diag_indices",
            data=np.asarray(diag_indices, dtype=np.int32),
        )
        transform_group.create_dataset(
            "diag_offsets",
            data=np.asarray(offsets, dtype=np.uint64),
        )
        transform_group.create_dataset(
            "diag_lengths",
            data=np.asarray(lengths, dtype=np.uint64),
        )
        if self._can_keep_plaintexts_resident(
            backend,
            int(transform_id),
            already_loaded=False,
        ):
            payload = np.asarray(payload_ds[:], dtype=np.uint8)
            backend.LoadPlaintextDiagonalsBatch(
                payload,
                offsets,
                lengths,
                list(diag_indices),
                int(transform_id),
            )
            self._mark_plaintexts_resident(int(transform_id))
            return
        backend.RemovePlaintextDiagonals(int(transform_id))

    def _load_plaintext_diagonals(
        self,
        backend,
        bundle: dict[str, object] | None = None,
        *,
        transform_ids: Iterable[int] | None = None,
    ) -> None:
        if not self._offloaded_plaintext_diagonals:
            return
        selected_ids = self._plaintext_load_transform_ids(backend, transform_ids) - set(
            int(value) for value in self._resident_plaintext_transform_ids
        )
        if not selected_ids:
            return
        if bundle is not None and bundle.get("plaintexts"):
            for transform_id, payload in bundle.get("plaintexts", {}).items():
                if int(transform_id) not in selected_ids:
                    continue
                if payload.get("plaintexts_prefetched_to_device"):
                    continue
                encoded_payload = payload.get("encoded_payload")
                if encoded_payload is not None:
                    backend.LoadLinearTransformPlaintexts(
                        encoded_payload,
                        int(transform_id),
                    )
                    payload["encoded_payload"] = None
                    if self._can_keep_plaintexts_resident(
                        backend,
                        int(transform_id),
                        already_loaded=True,
                    ):
                        self._mark_plaintexts_resident(int(transform_id))
                    continue
                backend.LoadPlaintextDiagonalsBatch(
                    payload["payload"],
                    list(payload["offsets"]),
                    list(payload["lengths"]),
                    list(payload["diag_indices"]),
                    int(transform_id),
                )
                payload["payload"] = np.zeros((0,), dtype=np.uint8)
                payload["offsets"] = ()
                payload["lengths"] = ()
            bundle["plaintexts"] = {}
            return
        handle, root = self._storage_group("r")
        try:
            storage = root[self._storage_key]
            for transform_id, diag_indices in self._diag_indices_by_transform.items():
                if int(transform_id) not in selected_ids:
                    continue
                transform_group = storage[str(transform_id)]
                if _ENCODED_HOIST_PAYLOAD_DATASET in transform_group:
                    payload = np.asarray(
                        transform_group[_ENCODED_HOIST_PAYLOAD_DATASET][()],
                        dtype=np.uint8,
                    ).reshape(-1)
                    backend.LoadLinearTransformPlaintexts(
                        payload,
                        int(transform_id),
                    )
                    if self._can_keep_plaintexts_resident(
                        backend,
                        int(transform_id),
                        already_loaded=True,
                    ):
                        self._mark_plaintexts_resident(int(transform_id))
                    continue
                if self._transform_uses_backend_streaming(backend, int(transform_id)):
                    raise self._streaming_payload_missing_error(int(transform_id))
                offsets = [int(v) for v in transform_group["diag_offsets"][:].tolist()]
                lengths = [int(v) for v in transform_group["diag_lengths"][:].tolist()]
                stored_diag_indices = [int(v) for v in transform_group["diag_indices"][:].tolist()]
                if self._stream_plaintext_diag_load_enabled(backend):
                    payload_ds = transform_group["diag_payload"]
                    chunk_limit = int(self._stream_plaintext_diag_load_chunk_bytes())
                    index = 0
                    while index < len(stored_diag_indices):
                        start = int(offsets[index])
                        end = int(start + lengths[index])
                        next_index = int(index + 1)
                        while next_index < len(stored_diag_indices):
                            candidate_end = int(offsets[next_index] + lengths[next_index])
                            if candidate_end - start > chunk_limit:
                                break
                            end = int(candidate_end)
                            next_index += 1
                        guard = self._forward_memory_guard(
                            backend,
                            reason=(
                                f"before_plaintext_stream_load:{self._storage_key}:"
                                f"{int(transform_id)}:{int(index)}"
                            ),
                            needed_bytes=int(end - start),
                            raise_on_low=True,
                        )
                        self._record_memory_event(
                            "before_plaintext_stream_load",
                            backend,
                            (int(transform_id),),
                            stream_start_index=int(index),
                            stream_end_index=int(next_index),
                            stream_bytes=int(end - start),
                            memory_guard=guard,
                        )
                        payload = np.asarray(payload_ds[start:end], dtype=np.uint8)
                        backend.LoadPlaintextDiagonalsBatch(
                            payload,
                            [int(offsets[i] - start) for i in range(index, next_index)],
                            [int(lengths[i]) for i in range(index, next_index)],
                            [int(stored_diag_indices[i]) for i in range(index, next_index)],
                            int(transform_id),
                        )
                        del payload
                        index = int(next_index)
                    del offsets, lengths, stored_diag_indices
                    continue
                payload = np.asarray(transform_group["diag_payload"][:], dtype=np.uint8)
                backend.LoadPlaintextDiagonalsBatch(
                    payload,
                    offsets,
                    lengths,
                    stored_diag_indices or list(diag_indices),
                    int(transform_id),
                )
                del payload, offsets, lengths, stored_diag_indices
        finally:
            handle.close()

    def _unload_plaintext_diagonals(self, backend, *, transform_ids: Iterable[int] | None = None) -> None:
        if not self._offloaded_plaintext_diagonals:
            return
        selected_ids = sorted(self._plaintext_load_transform_ids(backend, transform_ids))
        for transform_id in selected_ids:
            if int(transform_id) in self._resident_plaintext_transform_ids and self._can_keep_plaintexts_resident(
                backend,
                int(transform_id),
                already_loaded=True,
            ):
                continue
            backend.RemovePlaintextDiagonals(int(transform_id))
            self._clear_plaintexts_resident(int(transform_id))

    def _delete_offloaded_storage(self) -> None:
        if self._io_mode != "save":
            return
        if not self._offloaded_plaintext_diagonals or not self._diags_path:
            return
        try:
            handle, root = self._storage_group("a")
        except OSError:
            return
        try:
            if self._storage_key in root:
                del root[self._storage_key]
        finally:
            handle.close()

    def _transforms_have_complex_diagonals(self) -> bool:
        for transform in self.transforms:
            for _block_key, block_diags in getattr(transform, "diagonals", {}).items():
                for _diag_idx, diag_values in block_diags.items():
                    if isinstance(diag_values, torch.Tensor):
                        if bool(torch.is_complex(diag_values)):
                            return True
                    elif bool(torch.is_complex(torch.as_tensor(list(diag_values)))):
                        return True
        return False

    def _flatten_transform_diagonals(self, transform, *, has_complex: bool) -> tuple[np.ndarray, np.ndarray, int]:
        all_diagonals: dict[int, torch.Tensor] = {}
        for _block_key, block_diags in getattr(transform, "diagonals", {}).items():
            for diag_idx, diag_values in block_diags.items():
                if isinstance(diag_values, torch.Tensor):
                    values = diag_values.detach().reshape(-1)
                else:
                    values = torch.as_tensor(list(diag_values))
                all_diagonals.setdefault(int(diag_idx), values)
        if not all_diagonals:
            raise ValueError("all transforms must have generated diagonals before unified compilation")

        diag_idxs = np.asarray(sorted(all_diagonals.keys()), dtype=np.int32)
        chunks: list[np.ndarray] = []
        for idx in diag_idxs:
            values = all_diagonals[int(idx)].reshape(-1)
            if has_complex:
                if not bool(torch.is_complex(values)):
                    values = values.to(dtype=torch.complex64)
                values = values.detach().cpu()
                real = values.real.to(dtype=torch.float64).numpy().reshape(-1)
                imag = values.imag.to(dtype=torch.float64).numpy().reshape(-1)
                interleaved = np.empty(int(real.size) * 2, dtype=np.float64)
                interleaved[0::2] = real
                interleaved[1::2] = imag
                chunks.append(interleaved)
            else:
                chunks.append(values.detach().cpu().to(dtype=torch.float32).numpy().reshape(-1))
        if chunks:
            diag_data_flat = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float64 if has_complex else np.float32)
        else:
            diag_data_flat = np.zeros((0,), dtype=np.float64 if has_complex else np.float32)

        level = getattr(transform, "level", None)
        if level is None:
            level = len(transform.scheme.params.get_logq()) - 1
        return diag_idxs, diag_data_flat, int(level)

    def _generate_unified_backend_batch(
        self,
        backend,
        payloads: list[tuple[np.ndarray, np.ndarray, int]],
        *,
        has_complex: bool,
    ) -> list[int]:
        num_transforms = len(payloads)
        diag_idxs_ptrs = (ctypes.POINTER(ctypes.c_int) * num_transforms)()
        diag_idxs_lens = (ctypes.c_int * num_transforms)()
        array_type = ctypes.c_double if has_complex else ctypes.c_float
        diag_data_ptrs = (ctypes.POINTER(array_type) * num_transforms)()
        diag_data_lens = (ctypes.c_int * num_transforms)()

        owned_arrays: list[object] = []
        for idx, (diag_idxs, diag_data, _level) in enumerate(payloads):
            idx_array = np.ascontiguousarray(diag_idxs, dtype=np.int32)
            data_array = np.ascontiguousarray(diag_data, dtype=np.float64 if has_complex else np.float32)
            owned_arrays.extend((idx_array, data_array))
            diag_idxs_ptrs[idx] = idx_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            diag_idxs_lens[idx] = int(idx_array.size)
            diag_data_ptrs[idx] = data_array.ctypes.data_as(ctypes.POINTER(array_type))
            diag_data_lens[idx] = int(data_array.size)

        levels_array = (ctypes.c_int * num_transforms)(*[int(level) for _diag_idxs, _diag_data, level in payloads])
        owned_arrays.append(levels_array)

        generate = (
            backend.GenerateLinearTransformsUnifiedComplex
            if has_complex and hasattr(backend, "GenerateLinearTransformsUnifiedComplex")
            else backend.GenerateLinearTransformsUnified
        )
        _unified_compile_trace(
            "backend_generate_start",
            group=self._storage_key,
            transforms=num_transforms,
            has_complex=int(bool(has_complex)),
            diag_data_total=sum(int(payload[1].size) for payload in payloads),
        )
        started = time.perf_counter()
        ids = list(
            generate(
                num_transforms,
                diag_idxs_ptrs,
                diag_idxs_lens,
                diag_data_ptrs,
                diag_data_lens,
                levels_array,
            )
        )
        _unified_compile_trace(
            "backend_generate_done",
            group=self._storage_key,
            transforms=num_transforms,
            seconds=f"{time.perf_counter() - started:.6f}",
        )
        return ids

    def _cached_transform_descriptors(self) -> list[tuple[list[int], int]]:
        handle, root = self._storage_group("r")
        try:
            if self._storage_key not in root:
                raise RuntimeError(
                    f"Missing cached unified transform group '{self._storage_key}' in {self._diags_path}."
                )
            storage = root[self._storage_key]
            transform_names = sorted(storage.keys(), key=lambda value: int(value))
            if len(transform_names) != len(self.transforms):
                raise RuntimeError(
                    "Cached unified transform group size mismatch. "
                    f"group={self._storage_key} cached={len(transform_names)} expected={len(self.transforms)}. "
                    "Re-run with io_mode='save' to rebuild the cache."
                )
            descriptors: list[tuple[list[int], int]] = []
            for transform_name, transform in zip(transform_names, self.transforms):
                transform_group = storage[str(transform_name)]
                diag_indices = [
                    int(value)
                    for value in np.asarray(transform_group["diag_indices"][:], dtype=np.int32).reshape(-1)
                ]
                level = getattr(transform, "level", None)
                if level is None:
                    level = len(transform.scheme.params.get_logq()) - 1
                descriptors.append((diag_indices, int(level)))
            return descriptors
        finally:
            handle.close()

    def _generate_unified_backend_load_batch(
        self,
        backend,
        descriptors: list[tuple[list[int], int]],
    ) -> list[int]:
        generate = getattr(backend, "GenerateLinearTransformsUnifiedLoad", None)
        if not callable(generate):
            raise RuntimeError(
                "Backend does not support cached unified transform load. "
                "Rebuild the backend shared library or run with io_mode='save'."
            )

        num_transforms = len(descriptors)
        diag_idxs_ptrs = (ctypes.POINTER(ctypes.c_int) * num_transforms)()
        diag_idxs_lens = (ctypes.c_int * num_transforms)()
        owned_arrays: list[object] = []
        for idx, (diag_idxs, _level) in enumerate(descriptors):
            idx_array = np.ascontiguousarray(diag_idxs, dtype=np.int32)
            owned_arrays.append(idx_array)
            diag_idxs_ptrs[idx] = idx_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            diag_idxs_lens[idx] = int(idx_array.size)
        levels_array = (ctypes.c_int * num_transforms)(*[int(level) for _diag_idxs, level in descriptors])
        owned_arrays.append(levels_array)
        return list(generate(num_transforms, diag_idxs_ptrs, diag_idxs_lens, levels_array))

    def _compile_unified_cached_load(self, backend, *, allow_save_resume: bool = False) -> bool:
        can_resume_save = bool(allow_save_resume and self._compile_save_resume_enabled())
        if self._io_mode != "load" and not can_resume_save:
            return False
        if not self._diags_path:
            return False
        if not callable(getattr(backend, "GenerateLinearTransformsUnifiedLoad", None)):
            return False

        try:
            descriptors = self._cached_transform_descriptors()
        except (KeyError, OSError, RuntimeError):
            if can_resume_save:
                return False
            raise
        _unified_compile_trace(
            "compile_group_cached_load",
            group=self._storage_key,
            transforms=len(descriptors),
            save_resume=int(bool(can_resume_save)),
        )
        self.unified_ids = self._generate_unified_backend_load_batch(backend, descriptors)
        if len(self.unified_ids) != len(descriptors):
            raise RuntimeError("backend returned unexpected transform count for cached unified load")
        self._diag_indices_by_transform = {
            int(transform_id): tuple(int(idx) for idx in diag_indices)
            for transform_id, (diag_indices, _level) in zip(self.unified_ids, descriptors)
        }
        self._required_keys = ()
        self._required_keys_by_transform = {}
        for transform_id in self.unified_ids:
            self._record_transform_key_requests(backend, int(transform_id))
        for transform in self.transforms:
            self._clear_source_diagonals_after_compile(transform)
        self._offloaded_plaintext_diagonals = True
        self._prefetch_host_bytes = None
        self._prefetch_device_bytes = None
        self._saved_io_host_bytes_by_transform = None
        gc.collect()
        return True

    def _record_transform_key_requests(self, backend, transform_id: int) -> None:
        requests = tuple((int(key), level) for key, level in self._rotation_key_requests(backend, int(transform_id)))
        self._required_keys_by_transform[int(transform_id)] = requests
        merged: dict[int, int | None] = {int(key): level for key, level in self._required_keys}
        for key, level in requests:
            if level is None:
                merged[int(key)] = None
            else:
                current = merged.get(int(key))
                merged[int(key)] = int(level) if current is None else max(int(level), int(current))
        self._required_keys = tuple(sorted((int(key), level) for key, level in merged.items()))

    def _compile_rotation_keys(self, backend) -> None:
        _unified_compile_trace("rotation_keys_start", group=self._storage_key, required=len(self._required_keys))
        started = time.perf_counter()
        if self._should_save_rotation_keys():
            self._ensure_serialized_rotation_keys_available(backend)
        elif self._should_offload_rotation_keys():
            self._ensure_serialized_rotation_keys_available(backend)
            _unified_compile_trace(
                "rotation_keys_done",
                group=self._storage_key,
                required=len(self._required_keys),
                seconds=f"{time.perf_counter() - started:.6f}",
            )
            return
        else:
            for key, level in self._required_keys:
                self._generate_rotation_key(backend, int(key), level)
        _unified_compile_trace(
            "rotation_keys_done",
            group=self._storage_key,
            required=len(self._required_keys),
            seconds=f"{time.perf_counter() - started:.6f}",
        )

    def _compile_unified_streaming(self, backend, *, has_complex: bool) -> None:
        _unified_compile_trace(
            "streaming_compile_start",
            group=self._storage_key,
            transforms=len(self.transforms),
            has_complex=int(bool(has_complex)),
        )
        self.unified_ids = []
        self._diag_indices_by_transform = {}
        self._required_keys = ()
        self._required_keys_by_transform = {}
        save_plaintexts = bool(self._should_save_plaintext_diagonals())
        load_plaintexts = bool(self._io_mode == "load" and self._diags_path)
        handle = None
        storage = None
        if save_plaintexts:
            handle, root = self._storage_group("a")
            if self._storage_key in root:
                del root[self._storage_key]
            storage = root.create_group(self._storage_key)
        flatten_workers = _unified_compile_workers(len(self.transforms))
        batch_limit = _unified_stream_compile_batch_limit(len(self.transforms), int(flatten_workers))
        batch_byte_limit = int(_unified_stream_compile_batch_bytes())
        flatten_executor = (
            ThreadPoolExecutor(
                max_workers=max(1, min(int(flatten_workers), int(batch_limit))),
                thread_name_prefix="orion-unified-stream-flatten",
            )
            if flatten_workers > 1 and len(self.transforms) > 1
            else None
        )
        force_compile_trim = self._force_compile_trim_each_transform_enabled()
        try:
            transform_index = 0
            while transform_index < len(self.transforms):
                batch_end = min(len(self.transforms), int(transform_index + int(batch_limit)))
                batch_indices = list(range(int(transform_index), int(batch_end)))
                _unified_compile_trace(
                    "streaming_flatten_batch_start",
                    group=self._storage_key,
                    start=int(batch_indices[0]),
                    end=int(batch_indices[-1] + 1),
                    workers=(0 if flatten_executor is None else min(int(flatten_workers), int(batch_limit))),
                )
                flatten_started = time.perf_counter()
                if flatten_executor is not None:
                    payloads = list(
                        flatten_executor.map(
                            lambda index: self._flatten_transform_diagonals(
                                self.transforms[int(index)],
                                has_complex=has_complex,
                            ),
                            batch_indices,
                        )
                    )
                else:
                    payloads = [
                        self._flatten_transform_diagonals(self.transforms[int(index)], has_complex=has_complex)
                        for index in batch_indices
                    ]
                _unified_compile_trace(
                    "streaming_flatten_batch_done",
                    group=self._storage_key,
                    start=int(batch_indices[0]),
                    end=int(batch_indices[-1] + 1),
                    transforms=len(payloads),
                    diag_count=sum(int(payload[0].size) for payload in payloads),
                    data_count=sum(int(payload[1].size) for payload in payloads),
                    seconds=f"{time.perf_counter() - flatten_started:.6f}",
                )

                queued: list[tuple[int, tuple[np.ndarray, np.ndarray, int]]] = []
                queued_bytes = 0
                for index, payload in zip(batch_indices, payloads):
                    payload_bytes = int(payload[0].nbytes + payload[1].nbytes)
                    if (
                        queued
                        and int(batch_byte_limit) > 0
                        and int(queued_bytes + payload_bytes) > int(batch_byte_limit)
                    ):
                        self._compile_unified_streaming_payload_batch(
                            backend,
                            queued,
                            has_complex=has_complex,
                            save_plaintexts=save_plaintexts,
                            load_plaintexts=load_plaintexts,
                            storage=storage,
                            force_compile_trim=force_compile_trim,
                        )
                        queued = []
                        queued_bytes = 0
                    queued.append((int(index), payload))
                    queued_bytes += int(payload_bytes)
                if queued:
                    self._compile_unified_streaming_payload_batch(
                        backend,
                        queued,
                        has_complex=has_complex,
                        save_plaintexts=save_plaintexts,
                        load_plaintexts=load_plaintexts,
                        storage=storage,
                        force_compile_trim=force_compile_trim,
                    )
                transform_index = int(batch_end)
                del payloads
                gc.collect()
        finally:
            if flatten_executor is not None:
                flatten_executor.shutdown(wait=True)
            if handle is not None:
                handle.close()
        self._offloaded_plaintext_diagonals = bool(save_plaintexts or load_plaintexts)
        _unified_compile_trace("streaming_compile_done", group=self._storage_key, transforms=len(self.unified_ids))
        self._prefetch_host_bytes = None
        self._prefetch_device_bytes = None
        self._saved_io_host_bytes_by_transform = None

    def _compile_unified_streaming_payload_batch(
        self,
        backend,
        items: list[tuple[int, tuple[np.ndarray, np.ndarray, int]]],
        *,
        has_complex: bool,
        save_plaintexts: bool,
        load_plaintexts: bool,
        storage,
        force_compile_trim: bool,
    ) -> None:
        if not items:
            return
        payloads = [payload for _index, payload in items]
        indices = [int(index) for index, _payload in items]
        needed_bytes = int(sum(int(payload[0].nbytes + payload[1].nbytes) for payload in payloads))
        self._record_memory_event(
            "before_compile_transform_batch",
            backend,
            (),
            transform_indices=tuple(indices),
            batch_size=len(items),
            payload_bytes=int(needed_bytes),
        )
        self._forward_memory_guard(
            backend,
            reason=f"before_compile_transform_batch:{self._storage_key}:{indices[0]}:{indices[-1]}",
            needed_bytes=int(needed_bytes),
            force_trim=force_compile_trim,
            raise_on_low=True,
        )
        ids = self._generate_unified_backend_batch(backend, payloads, has_complex=has_complex)
        if len(ids) != len(payloads):
            raise RuntimeError("backend returned unexpected transform count for streaming unified compile batch")
        transform_ids = [int(value) for value in ids]
        self.unified_ids.extend(transform_ids)
        for transform_id, payload in zip(transform_ids, payloads):
            self._diag_indices_by_transform[int(transform_id)] = tuple(int(idx) for idx in payload[0])
            self._record_transform_key_requests(backend, int(transform_id))
        self._record_memory_event(
            "after_compile_transform_batch",
            backend,
            transform_ids,
            transform_indices=tuple(indices),
            compiled_transform_ids=tuple(transform_ids),
            batch_size=len(items),
        )
        remove_plaintexts = getattr(backend, "RemovePlaintextDiagonals", None)
        for transform_index, transform_id, payload in zip(indices, transform_ids, payloads):
            if save_plaintexts:
                if storage is None:
                    raise RuntimeError("missing unified transform plaintext storage")
                self._save_and_unload_plaintext_diagonals_for_transform(
                    backend,
                    storage,
                    int(transform_id),
                    payload[0],
                )
            elif load_plaintexts and callable(remove_plaintexts):
                remove_plaintexts(int(transform_id))
            self._clear_source_diagonals_after_compile(self.transforms[int(transform_index)])
            self._record_memory_event(
                "after_offload_transform",
                backend,
                (int(transform_id),),
                transform_index=int(transform_index),
                transform_id=int(transform_id),
            )
        self._forward_memory_guard(
            backend,
            reason=f"after_compile_transform_batch_offload:{self._storage_key}:{indices[0]}:{indices[-1]}",
            transform_ids=transform_ids,
            force_trim=force_compile_trim,
            raise_on_low=True,
        )

    def compile_unified(self, backend) -> None:
        if self.is_compiled:
            return
        if not self.transforms:
            raise ValueError("UnifiedTransformGroup requires at least one transform")

        _unified_compile_trace("compile_group_start", group=self._storage_key, transforms=len(self.transforms))
        group_started = time.perf_counter()
        self._configure_io()
        self._prefetch_host_bytes = None
        self._prefetch_device_bytes = None
        self._saved_io_host_bytes_by_transform = None
        self._resident_plaintext_transform_ids = set()
        self.memory_trace = []
        self._record_memory_event("before_compile_group", backend, ())
        if self._compile_unified_cached_load(backend, allow_save_resume=True):
            pass
        elif self._memory_bounded_compile_enabled(backend):
            complex_started = time.perf_counter()
            has_complex = self._transforms_have_complex_diagonals()
            _unified_compile_trace(
                "detect_complex_done",
                group=self._storage_key,
                has_complex=int(bool(has_complex)),
                seconds=f"{time.perf_counter() - complex_started:.6f}",
            )
            self._compile_unified_streaming(backend, has_complex=has_complex)
        else:
            complex_started = time.perf_counter()
            has_complex = self._transforms_have_complex_diagonals()
            _unified_compile_trace(
                "detect_complex_done",
                group=self._storage_key,
                has_complex=int(bool(has_complex)),
                seconds=f"{time.perf_counter() - complex_started:.6f}",
            )
            workers = _unified_compile_workers(len(self.transforms))
            flatten_started = time.perf_counter()
            _unified_compile_trace("flatten_all_start", group=self._storage_key, transforms=len(self.transforms), workers=workers)
            if workers > 1 and len(self.transforms) > 1:
                with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="orion-unified-flatten") as executor:
                    payloads = list(
                        executor.map(
                            lambda transform: self._flatten_transform_diagonals(transform, has_complex=has_complex),
                            self.transforms,
                        )
                    )
            else:
                payloads = [
                    self._flatten_transform_diagonals(transform, has_complex=has_complex)
                    for transform in self.transforms
                ]
            _unified_compile_trace(
                "flatten_all_done",
                group=self._storage_key,
                transforms=len(payloads),
                seconds=f"{time.perf_counter() - flatten_started:.6f}",
            )
            self.unified_ids = self._generate_unified_backend_batch(backend, payloads, has_complex=has_complex)
            self._diag_indices_by_transform = {
                int(transform_id): tuple(int(idx) for idx in diag_idxs)
                for transform_id, (diag_idxs, _diag_data, _level) in zip(self.unified_ids, payloads)
            }
            self._required_keys = ()
            self._required_keys_by_transform = {}
            for transform_id in self.unified_ids:
                self._record_transform_key_requests(backend, int(transform_id))
            for transform in self.transforms:
                self._clear_source_diagonals_after_compile(transform)
            del payloads
            gc.collect()
        self.is_compiled = True

        self._compile_rotation_keys(backend)

        if self._io_mode == "load" and bool(self._diags_path):
            self._offloaded_plaintext_diagonals = True
        if self._should_save_plaintext_diagonals() and not self._offloaded_plaintext_diagonals:
            self._save_and_unload_plaintext_diagonals(backend)
        self._prepare_shared_cache_plans(backend)
        self._record_memory_event("after_compile_group", backend)
        self._register_shared_saved_io_work_unit(backend)
        _unified_compile_trace(
            "compile_group_done",
            group=self._storage_key,
            transforms=len(self.unified_ids or ()),
            seconds=f"{time.perf_counter() - group_started:.6f}",
        )

    def get_transform_ids(self, transform) -> dict[tuple[int, int], int]:
        if not self.is_compiled or self.unified_ids is None:
            raise RuntimeError("UnifiedTransformGroup not compiled")
        try:
            index = self.transforms.index(transform)
        except ValueError as exc:
            raise ValueError("Transform not found in unified group") from exc
        return {(0, 0): int(self.unified_ids[index])}

    def evaluate_unified(self, ct_input_id: int, backend) -> list[int]:
        if not self.is_compiled or self.unified_ids is None:
            raise RuntimeError("UnifiedTransformGroup must be compiled before evaluation")
        if self._memory_bounded_eval_enabled(backend):
            return self._evaluate_unified_memory_bounded(int(ct_input_id), backend)
        scheduler = self._shared_saved_io_scheduler()
        prefetch_key = self._saved_io_prefetch_key()
        using_shared_prefetch = False
        if scheduler is not None:
            using_shared_prefetch = bool(
                scheduler.fill_saved_io_prefetch_window(
                    prefetch_key,
                    include_current=True,
                    scratch_reserve_bytes=self._estimate_prefetch_device_bytes(backend),
                )
            )
        bundle = (
            scheduler.consume_saved_io_prefetch(prefetch_key)
            if using_shared_prefetch
            else self._io_prefetcher.consume(self._storage_key)
        )
        if bundle is None and (self._should_offload_rotation_keys() or self._offloaded_plaintext_diagonals):
            self._forward_memory_guard(
                backend,
                reason=f"before_unified_group_load:{self._storage_key}",
                transform_ids=self.unified_ids,
            )
            bundle = self._read_saved_io_bundle(
                backend,
                prefetch=False,
                required_keys=self._rotation_key_requests_to_load(backend, self.unified_ids, self._required_keys),
            )
        self._load_rotation_keys(backend, bundle)
        self._load_plaintext_diagonals(backend, bundle)
        if using_shared_prefetch:
            scheduler.fill_saved_io_prefetch_window(
                prefetch_key,
                include_current=False,
                scratch_reserve_bytes=self._estimate_prefetch_device_bytes(backend),
            )
        else:
            self._schedule_next_saved_io_prefetch(backend)
        transform_ids_array = (ctypes.c_int * len(self.unified_ids))(*[int(v) for v in self.unified_ids])
        try:
            self._record_memory_event("before_eval_group", backend)
            return list(
                backend.EvaluateLinearTransformsWithSharedCache(
                    transform_ids_array,
                    len(self.unified_ids),
                    int(ct_input_id),
                )
            )
        finally:
            self._record_memory_event("after_eval_group", backend)
            self._unload_plaintext_diagonals(backend)
            self._unload_rotation_keys(backend)
            if bundle is not None:
                bundle.clear()
            trim_event = self._forward_memory_guard(
                backend,
                reason=f"after_unified_group_unload:{self._storage_key}",
                transform_ids=self.unified_ids,
                force_trim=True,
                raise_on_low=False,
            )
            self._record_memory_event("after_eval_group_trim", backend, self.unified_ids, memory_guard=trim_event)

    def _evaluate_unified_memory_bounded(self, ct_input_id: int, backend) -> list[int]:
        if self.unified_ids is None:
            raise RuntimeError("UnifiedTransformGroup must be compiled before evaluation")
        chunks = self._memory_bounded_chunks(backend)
        stream_plaintexts = bool(self._stream_plaintext_diag_load_enabled(backend))
        output_id_by_transform: dict[int, int] = {}
        group_timing = {
            "read_bundle_s": 0.0,
            "load_keys_s": 0.0,
            "load_plaintexts_s": 0.0,
            "eval_s": 0.0,
            "eval_total_s": 0.0,
            "unload_s": 0.0,
            "trim_s": 0.0,
            "cpp_plan_s": 0.0,
            "cpp_level_adjust_s": 0.0,
            "cpp_baby_step_s": 0.0,
            "cpp_giant_step_s": 0.0,
            "stream_build_map_s": 0.0,
            "stream_encode_hoist_s": 0.0,
            "stream_load_payload_s": 0.0,
            "stream_eval_s": 0.0,
            "stream_accumulate_s": 0.0,
            "cpp_push_s": 0.0,
            "cpp_trim_s": 0.0,
        }
        def add_timing(chunk_timing: dict[str, float], key: str, value: float) -> None:
            chunk_timing[key] = float(chunk_timing.get(key, 0.0) + float(value))
            group_timing[key] = float(group_timing.get(key, 0.0) + float(value))

        def chunk_prefetch_key(chunk_index: int) -> tuple[str, str, int]:
            return ("unified_chunk", str(self._storage_key), int(chunk_index))

        def chunk_required_keys(chunk_ids: Iterable[int]) -> tuple[tuple[int, int | None], ...]:
            required = self._required_keys_for_transform_ids(chunk_ids)
            return self._rotation_key_requests_to_load(backend, chunk_ids, required)

        def read_chunk_bundle(chunk_ids: list[int]) -> dict[str, object] | None:
            return self._read_saved_io_bundle(
                backend,
                prefetch=False,
                transform_ids=chunk_ids,
                required_keys=chunk_required_keys(chunk_ids),
                include_plaintexts=not stream_plaintexts,
            )

        def schedule_chunk_prefetch(next_index: int) -> None:
            if next_index >= len(chunks):
                return
            next_ids = [int(value) for value in chunks[int(next_index)]]
            next_host_bytes = 0
            if not stream_plaintexts:
                next_host_bytes = int(
                    sum(
                        int(self._estimate_saved_io_host_bytes_for_transform(backend, int(transform_id)))
                        for transform_id in next_ids
                    )
                )
            if not should_prefetch_saved_io(next_host_bytes, backend=None, device_bytes=0):
                return
            memory_info = self._host_memory_info()
            if memory_info is not None:
                guard = self._forward_memory_guard(
                    backend,
                    reason=f"before_unified_chunk_prefetch:{self._storage_key}:{int(next_index)}",
                    transform_ids=next_ids,
                    needed_bytes=int(next_host_bytes),
                    raise_on_low=False,
                )
                after = guard.get("after") or guard.get("before") or {}
                if int(after.get("available_bytes", 0)) - int(next_host_bytes) < int(guard.get("min_available_bytes", 0)):
                    return
            self._io_prefetcher.submit(
                chunk_prefetch_key(int(next_index)),
                lambda ids=next_ids: read_chunk_bundle(ids),
                host_bytes=next_host_bytes,
                device_bytes=0,
            )

        self._record_memory_event(
            "before_eval_group_memory_bounded",
            backend,
            self.unified_ids,
            chunk_count=int(len(chunks)),
            eval_budget_bytes=int(self._eval_budget_bytes(backend)),
            stream_plaintexts=bool(stream_plaintexts),
            stream_plaintext_chunk_bytes=int(self._stream_plaintext_diag_load_chunk_bytes()) if stream_plaintexts else 0,
        )
        for chunk_index, chunk_ids in enumerate(chunks):
            chunk_timing = {
                "read_bundle_s": 0.0,
                "load_keys_s": 0.0,
                "load_plaintexts_s": 0.0,
                "eval_s": 0.0,
                "eval_total_s": 0.0,
                "unload_s": 0.0,
                "trim_s": 0.0,
                "cpp_plan_s": 0.0,
                "cpp_level_adjust_s": 0.0,
                "cpp_baby_step_s": 0.0,
                "cpp_giant_step_s": 0.0,
                "stream_build_map_s": 0.0,
                "stream_encode_hoist_s": 0.0,
                "stream_load_payload_s": 0.0,
                "stream_eval_s": 0.0,
                "stream_accumulate_s": 0.0,
                "cpp_push_s": 0.0,
                "cpp_trim_s": 0.0,
            }
            required_keys = self._required_keys_for_transform_ids(chunk_ids)
            required_keys_to_load = self._rotation_key_requests_to_load(backend, chunk_ids, required_keys)
            memory_guard = self._forward_memory_guard(
                backend,
                reason=f"before_unified_chunk_load:{self._storage_key}:{int(chunk_index)}",
                transform_ids=chunk_ids,
                stream_plaintexts=bool(stream_plaintexts),
            )
            self._record_memory_event(
                "before_eval_chunk_load",
                backend,
                chunk_ids,
                chunk_index=int(chunk_index),
                chunk_transform_count=int(len(chunk_ids)),
                timing=dict(chunk_timing),
                memory_guard=memory_guard,
            )
            read_started = time.perf_counter()
            bundle = self._io_prefetcher.consume(chunk_prefetch_key(int(chunk_index)))
            if bundle is None:
                bundle = self._read_saved_io_bundle(
                    backend,
                    prefetch=False,
                    transform_ids=chunk_ids,
                    required_keys=required_keys_to_load,
                    include_plaintexts=not stream_plaintexts,
                )
            add_timing(chunk_timing, "read_bundle_s", time.perf_counter() - read_started)
            schedule_chunk_prefetch(int(chunk_index) + 1)
            load_keys_started = time.perf_counter()
            self._load_rotation_keys(
                backend,
                bundle,
                transform_ids=chunk_ids,
                required_keys=required_keys_to_load,
            )
            add_timing(chunk_timing, "load_keys_s", time.perf_counter() - load_keys_started)
            load_plaintexts_started = time.perf_counter()
            self._load_plaintext_diagonals(
                backend,
                None if stream_plaintexts else bundle,
                transform_ids=chunk_ids,
            )
            add_timing(chunk_timing, "load_plaintexts_s", time.perf_counter() - load_plaintexts_started)
            self._record_memory_event(
                "after_eval_chunk_load",
                backend,
                chunk_ids,
                chunk_index=int(chunk_index),
                chunk_transform_count=int(len(chunk_ids)),
                timing=dict(chunk_timing),
            )
            transform_ids_array = (ctypes.c_int * len(chunk_ids))(*[int(v) for v in chunk_ids])
            try:
                self._consume_trim_seconds(backend)
                self._consume_shared_cache_eval_profile(backend)
                eval_started = time.perf_counter()
                output_ids = list(
                    backend.EvaluateLinearTransformsWithSharedCache(
                        transform_ids_array,
                        len(chunk_ids),
                        int(ct_input_id),
                    )
                )
                eval_total_s = float(time.perf_counter() - eval_started)
                trim_s = self._consume_trim_seconds(backend)
                cpp_profile = self._consume_shared_cache_eval_profile(backend)
                add_timing(chunk_timing, "trim_s", trim_s)
                for profile_key, profile_value in cpp_profile.items():
                    add_timing(chunk_timing, profile_key, profile_value)
                add_timing(chunk_timing, "eval_total_s", eval_total_s)
                add_timing(chunk_timing, "eval_s", max(0.0, eval_total_s - trim_s))
                for transform_id, output_id in zip(chunk_ids, output_ids):
                    output_id_by_transform[int(transform_id)] = int(output_id)
                self._record_memory_event(
                    "after_eval_chunk",
                    backend,
                    chunk_ids,
                    chunk_index=int(chunk_index),
                    chunk_transform_count=int(len(chunk_ids)),
                    output_count=int(len(output_ids)),
                    timing=dict(chunk_timing),
                )
            finally:
                unload_started = time.perf_counter()
                self._unload_plaintext_diagonals(backend, transform_ids=chunk_ids)
                self._unload_rotation_keys(backend, transform_ids=chunk_ids)
                add_timing(chunk_timing, "unload_s", time.perf_counter() - unload_started)
                if bundle is not None:
                    bundle.clear()
                trim_event = self._forward_memory_guard(
                    backend,
                    reason=f"after_unified_chunk_unload:{self._storage_key}:{int(chunk_index)}",
                    transform_ids=chunk_ids,
                    force_trim=True,
                    raise_on_low=False,
                )
                self._record_memory_event(
                    "after_eval_chunk_unload",
                    backend,
                    chunk_ids,
                    chunk_index=int(chunk_index),
                    chunk_transform_count=int(len(chunk_ids)),
                    timing=dict(chunk_timing),
                    memory_guard=trim_event,
                )
        self._record_memory_event("after_eval_group_memory_bounded", backend, self.unified_ids, timing=dict(group_timing))
        return [int(output_id_by_transform[int(transform_id)]) for transform_id in self.unified_ids]

    def execute(self, calling_transform, ct_input):
        from orion.backend.python.tensors import CipherTensor

        if not isinstance(ct_input, CipherTensor):
            raise TypeError(f"Expected CipherTensor input, got {type(ct_input)}")
        ct_id = int(ct_input.ids[0])

        if not hasattr(self, "_result_cache") or getattr(self, "_result_cache_key") != ct_id:
            output_ids = self.evaluate_unified(ct_id, calling_transform.scheme.backend)
            self._result_cache = {}
            self._result_cache_key = ct_id
            for index, transform in enumerate(self.transforms):
                out_shape = transform.fhe_output_shape
                self._result_cache[transform] = CipherTensor(calling_transform.scheme, [output_ids[index]], out_shape, out_shape)
        if calling_transform not in self._result_cache:
            output_ids = self.evaluate_unified(ct_id, calling_transform.scheme.backend)
            self._result_cache = {}
            self._result_cache_key = ct_id
            for index, transform in enumerate(self.transforms):
                out_shape = transform.fhe_output_shape
                self._result_cache[transform] = CipherTensor(calling_transform.scheme, [output_ids[index]], out_shape, out_shape)
        result = self._result_cache.pop(calling_transform)
        if not self._result_cache:
            self._result_cache_key = None
        return result

    def cleanup(self, backend) -> None:
        self._io_prefetcher.clear(wait=True)
        scheduler = self._shared_saved_io_scheduler()
        unregister = getattr(scheduler, "unregister_saved_io_prefetch_work_unit", None)
        if callable(unregister):
            unregister(self._saved_io_prefetch_key())
        self._clear_resident_rotation_keys(backend)
        if self.unified_ids is not None:
            for transform_id in self.unified_ids:
                backend.DeleteLinearTransform(int(transform_id))
            self._forward_memory_guard(
                backend,
                reason=f"after_unified_group_cleanup:{self._storage_key}",
                transform_ids=self.unified_ids,
                force_trim=True,
                raise_on_low=False,
            )
        self._delete_offloaded_storage()
        self.unified_ids = None
        self.is_compiled = False
        self._diag_indices_by_transform = {}
        self._offloaded_plaintext_diagonals = False
        self._required_keys = ()
        self._required_keys_by_transform = {}
        self._prefetch_host_bytes = None
        self._prefetch_device_bytes = None
        self._saved_io_host_bytes_by_transform = None
        self._resident_plaintext_transform_ids = set()


def can_use_unified_bsgs(layers: List) -> bool:
    from orion.nn.linear import LinearTransform

    if len(layers) < 2:
        return False
    return all(isinstance(layer, LinearTransform) and bool(getattr(layer, "diagonals", {})) for layer in layers)
