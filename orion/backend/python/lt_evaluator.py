import h5py
import ctypes
import gc
import os
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from . import compile_cache
from .io_prefetch import (
    AsyncIOPrefetcher,
    estimate_linear_transform_device_bytes,
    should_prefetch_saved_io,
)
from .memory_lifecycle import guard_host_memory
from orion.backend.python.tensors import CipherTensor


_ENCODED_HOIST_PAYLOAD_DATASET = "__encoded_hoist_payload__"


_COMPILE_PROFILE_KEYS = (
    "read_s",
    "diag_generate_s",
    "encode_s",
    "decode_s",
    "serialize_s",
    "device_commit_s",
    "key_prepare_s",
    "wait_s",
    "peak_host_bytes",
    "peak_device_bytes",
)


class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme 
        self.params = scheme.params
        self.backend = scheme.backend
        self.evaluator = scheme.evaluator

        self.embed_method = self.params.get_embedding_method()
        self.io_mode = self.params.get_io_mode()
        self.diags_path = self.params.get_diags_path()
        self.keys_path = self.params.get_keys_path()

        self.saved_rotation_keys = set()
        self.compile_manifest = None
        self.compile_load_profile = {key: 0.0 for key in _COMPILE_PROFILE_KEYS}
        self._transform_io_prefetcher = AsyncIOPrefetcher()
        self._transform_io_lookahead = self._read_transform_io_lookahead()
        self._transform_io_size_cache: dict[tuple[str, int, int, int], int] = {}
        self._transform_device_size_cache: dict[int, int] = {}
        self._saved_io_linear_work_order: tuple[tuple[str, int, int, int], ...] = ()
        self._saved_io_external_work_order: list[object] = []
        self._saved_io_external_loaders: dict[object, object] = {}
        self._saved_io_external_host_bytes: dict[object, object] = {}
        self._saved_io_external_device_bytes: dict[object, object] = {}
        self._saved_io_work_order: tuple[object, ...] = ()
        self._saved_io_work_index: dict[object, int] = {}
        self._dense_host_plaintext_payload_cache: dict[int, np.ndarray] = {}
        self.new_evaluator()

    def set_compile_manifest(self, manifest) -> None:
        self.compile_manifest = manifest

    def get_compile_load_profile(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.compile_load_profile.items()}

    def _add_profile(self, key: str, seconds: float) -> None:
        if key in self.compile_load_profile:
            self.compile_load_profile[key] += float(seconds)

    def _forward_memory_guard(
        self,
        *,
        reason: str,
        needed_bytes: int = 0,
        force_trim: bool = False,
        raise_on_low: bool = True,
    ) -> dict[str, object]:
        return guard_host_memory(
            self.backend,
            reason=str(reason),
            needed_bytes=int(max(0, int(needed_bytes or 0))),
            force_trim=bool(force_trim),
            raise_on_low=bool(raise_on_low),
        )

    def _cached_manifest(self):
        if self.compile_manifest is not None:
            return self.compile_manifest
        if self.io_mode != "load":
            return None
        manifest = compile_cache.read_manifest(compile_cache.manifest_path(self.params))
        self.compile_manifest = manifest
        return manifest

    def new_evaluator(self):
        self.backend.NewLinearTransformEvaluator()

    def _read_transform_io_lookahead(self) -> int:
        raw_value = os.environ.get("ORION_SAVED_IO_PREFETCH_LOOKAHEAD", "2")
        try:
            return max(0, int(raw_value))
        except (TypeError, ValueError):
            return 2

    def register_saved_io_schedule(self, linear_layers) -> None:
        work_order: list[tuple[str, int, int, int]] = []
        for layer in linear_layers:
            layer_name = getattr(layer, "name", None)
            if layer_name is None:
                continue
            for row, col, transform_id in self._work_items_from_transform_ids(
                getattr(layer, "transform_ids", {}) or {}
            ):
                work_order.append((str(layer_name), int(row), int(col), int(transform_id)))

        self._saved_io_linear_work_order = tuple(work_order)
        self._rebuild_saved_io_work_order()

    def register_saved_io_prefetch_work_unit(
        self,
        key,
        *,
        loader,
        host_bytes,
        device_bytes,
    ) -> None:
        if key not in self._saved_io_external_loaders:
            self._saved_io_external_work_order.append(key)
        self._saved_io_external_loaders[key] = loader
        self._saved_io_external_host_bytes[key] = host_bytes
        self._saved_io_external_device_bytes[key] = device_bytes
        self._rebuild_saved_io_work_order()

    def unregister_saved_io_prefetch_work_unit(self, key) -> None:
        if key not in self._saved_io_external_loaders:
            return
        self._transform_io_prefetcher.discard(key, wait=True)
        self._saved_io_external_loaders.pop(key, None)
        self._saved_io_external_host_bytes.pop(key, None)
        self._saved_io_external_device_bytes.pop(key, None)
        self._saved_io_external_work_order = [
            existing_key
            for existing_key in self._saved_io_external_work_order
            if existing_key != key
        ]
        self._rebuild_saved_io_work_order()

    def consume_saved_io_prefetch(self, key):
        return self._transform_io_prefetcher.consume(key)

    def fill_saved_io_prefetch_window(
        self,
        key,
        *,
        include_current: bool = True,
        scratch_reserve_bytes: int = 0,
    ) -> bool:
        if key not in self._saved_io_work_index:
            return False
        start_index = int(self._saved_io_work_index[key])
        if not bool(include_current):
            start_index += 1
        self._fill_transform_io_prefetch_window(
            self._saved_io_work_order,
            start_index,
            scratch_reserve_bytes=int(scratch_reserve_bytes or 0),
        )
        return True

    def _rebuild_saved_io_work_order(self) -> None:
        self._transform_io_prefetcher.clear(wait=True)
        self._saved_io_work_order = tuple(self._saved_io_linear_work_order) + tuple(
            self._saved_io_external_work_order
        )
        self._saved_io_work_index = {
            key: index
            for index, key in enumerate(self._saved_io_work_order)
        }

    def _work_items_from_transform_ids(self, transform_ids: dict) -> list[tuple[int, int, int]]:
        if not transform_ids:
            return []
        try:
            keys = [(int(row), int(col)) for row, col in transform_ids.keys()]
        except (TypeError, ValueError):
            return [
                (0, index, int(transform_id))
                for index, transform_id in enumerate(transform_ids.values())
            ]

        rows = max(row for row, _col in keys) + 1
        cols = max(col for _row, col in keys) + 1
        if rows * cols == len(transform_ids) and all(
            (row, col) in transform_ids
            for row in range(rows)
            for col in range(cols)
        ):
            return [
                (int(row), int(col), int(transform_ids[(row, col)]))
                for row in range(rows)
                for col in range(cols)
            ]
        return [
            (int(row), int(col), int(transform_ids[(row, col)]))
            for row, col in sorted(keys)
        ]

    def generate_transforms(self, linear_layer):
        layer_name = linear_layer.name
        diagonals = linear_layer.diagonals 
        level = linear_layer.level
        bsgs_ratio = linear_layer.bsgs_ratio
        reuse_saved_plaintexts = bool(getattr(linear_layer, "_compile_cache_reuse_saved_plaintexts", False))
        effective_io_mode = "load" if reuse_saved_plaintexts else self.io_mode
        if self.io_mode == "save":
            linear_layer._compile_cache_diag_indices_by_block = {
                (int(row), int(col)): tuple(sorted(int(idx) for idx in diags.keys()))
                for (row, col), diags in diagonals.items()
            }
            linear_layer._compile_cache_slot_count_by_block = {
                (int(row), int(col)): self._slot_count_from_diags(diags)
                for (row, col), diags in diagonals.items()
            }

        lintransf_ids = {}

        def finish_compiled_batch(batch_ids) -> None:
            for row, col, diags_idxs, lintransf_id in batch_ids:
                lintransf_ids[(int(row), int(col))] = int(lintransf_id)
                self.generate_rotation_keys(int(lintransf_id))
                self._cache_dense_host_plaintexts_if_enabled(int(lintransf_id))
                if self.io_mode == "save" and not reuse_saved_plaintexts:
                    self.save_plaintext_diagonals(
                        layer_name, int(lintransf_id), int(row), int(col), diags_idxs
                    )
            if self.io_mode != "none" and not reuse_saved_plaintexts:
                self._trim_backend_runtime_memory()

        batch_ids = self._generate_transforms_batch(
            diagonals,
            level=int(level),
            bsgs_ratio=float(bsgs_ratio),
            io_mode=str(effective_io_mode),
            on_batch_result=finish_compiled_batch,
        )
        if batch_ids is not None:
            if self.io_mode == "save":
                linear_layer.diagonals = {}
                gc.collect()
            return lintransf_ids

        # Generate all linear transforms block by block.
        for (row, col), diags in diagonals.items(): 
            diags_idxs, diags_data = [], []
            for idx, diag in diags.items(): 
                diags_idxs.append(idx)
                diags_data.extend(diag)

            lintransf_id = self.backend.GenerateLinearTransform(
                diags_idxs, diags_data, level, bsgs_ratio, effective_io_mode
            )
            lintransf_ids[(row, col)] = lintransf_id

            # Now we can generate any new rotation keys needed for
            # this linear transform.
            self.generate_rotation_keys(lintransf_id)
            self._cache_dense_host_plaintexts_if_enabled(int(lintransf_id))
            if self.io_mode == "save" and not reuse_saved_plaintexts:
                self.save_plaintext_diagonals(
                    layer_name, lintransf_id, row, col, diags_idxs
                )
            if self.io_mode != "none" and not reuse_saved_plaintexts:
                self._trim_backend_runtime_memory()

        if self.io_mode == "save":
            linear_layer.diagonals = {}
            gc.collect()
        return lintransf_ids

    def _dense_host_payload_cache_enabled(self) -> bool:
        override = os.environ.get("ORION_DENSE_LT_HOST_PAYLOAD_CACHE")
        if override is None:
            return False
        if override.lower() in ("0", "false", "no", "off"):
            return False
        if self.io_mode != "none":
            return False
        return self._encoded_plaintext_payload_supported()

    def _cache_dense_host_plaintexts_if_enabled(self, transform_id: int) -> None:
        if not self._dense_host_payload_cache_enabled():
            return
        transform_id = int(transform_id)
        if transform_id in self._dense_host_plaintext_payload_cache:
            return
        load_batch = getattr(self.backend, "LoadPlaintextDiagonalsBatch", None)
        if callable(load_batch):
            load_batch([], [], [], [], int(transform_id))
        else:
            loader = getattr(self.backend, "LoadLinearTransformPlaintexts", None)
            if not callable(loader):
                return
            loader(np.zeros((0,), dtype=np.uint8), int(transform_id))
        serial_payload, payload_ptr = self.backend.SerializeLinearTransformPlaintexts(int(transform_id))
        try:
            self._dense_host_plaintext_payload_cache[int(transform_id)] = np.asarray(
                serial_payload,
                dtype=np.uint8,
            ).copy()
        finally:
            self.backend.FreeCArray(payload_ptr)
            self.backend.RemovePlaintextDiagonals(int(transform_id))

    def _slot_count_from_diags(self, diags):
        for diag in diags.values():
            try:
                length = int(len(diag))
            except TypeError:
                continue
            if length > 0:
                return int(length)
        params = getattr(self.scheme, "params", None)
        if params is not None and hasattr(params, "get_slots"):
            return int(params.get_slots())
        return 0

    def _flatten_diagonals(self, diags):
        diags_idxs, diags_data = [], []
        for idx, diag in diags.items():
            diags_idxs.append(int(idx))
            if isinstance(diag, torch.Tensor):
                values = diag.detach().cpu().reshape(-1).tolist()
            elif isinstance(diag, np.ndarray):
                values = diag.reshape(-1).tolist()
            else:
                values = list(diag)
            diags_data.extend(float(value) for value in values)
        return diags_idxs, diags_data

    def _lt_worker_count(self, item_count: int) -> int:
        if item_count <= 1:
            return 1
        raw_value = os.environ.get("ORION_LT_COMPILE_WORKERS", "")
        try:
            requested = int(raw_value) if raw_value else min(4, int(os.cpu_count() or 1))
        except (TypeError, ValueError):
            requested = 4
        return max(1, min(int(item_count), int(requested)))

    def _lt_compile_batch_limit(self, item_count: int, *, index_only: bool = False) -> int:
        if item_count <= 1:
            return 1
        if bool(index_only):
            raw_value = os.environ.get("ORION_DENSE_LT_INDEX_ONLY_BATCH_TRANSFORMS")
            if raw_value is not None:
                try:
                    requested = int(raw_value)
                except (TypeError, ValueError):
                    requested = 0
                if requested > 0:
                    return max(1, min(int(item_count), int(requested)))
                return int(item_count)
            return max(1, min(int(item_count), 128))
        for name in (
            "ORION_DENSE_LT_COMPILE_BATCH_TRANSFORMS",
            "ORION_LT_COMPILE_BATCH_TRANSFORMS",
            # Dense save/load needs the same kind of guard that provider cached
            # load uses. Honor the existing provider knob so benchmark scripts
            # that already set it get bounded dense compilation too.
            "ORION_UNIFIED_CACHED_LOAD_BATCH_TRANSFORMS",
            "ORION_UNIFIED_LOAD_BATCH_TRANSFORMS",
        ):
            raw_value = os.environ.get(name)
            if raw_value is None:
                continue
            try:
                requested = int(raw_value)
            except (TypeError, ValueError):
                requested = 0
            if requested > 0:
                return max(1, min(int(item_count), int(requested)))
            return int(item_count)
        if self.io_mode != "none":
            return max(1, min(int(item_count), 4))
        return int(item_count)

    def _build_block_payload(self, item):
        (row, col), diags = item
        diags_idxs, diags_data = self._flatten_diagonals(diags)
        return int(row), int(col), diags_idxs, diags_data

    def _compile_save_resume_enabled_for_layer(self, linear_layer) -> bool:
        enabled = getattr(linear_layer, "_compile_save_resume_enabled", None)
        return callable(enabled) and bool(enabled())

    def _plaintext_blocks_complete(self, layer, diag_group) -> bool:
        if "plaintexts" not in layer:
            return False
        plaintext_group = layer["plaintexts"]
        for block_name in diag_group:
            if str(block_name) not in plaintext_group:
                return False
            plaintext_block = plaintext_group[str(block_name)]
            if _ENCODED_HOIST_PAYLOAD_DATASET in plaintext_block:
                continue
            expected_keys = {str(diag_idx) for diag_idx in diag_group[str(block_name)].keys()}
            if not expected_keys.issubset(set(str(key) for key in plaintext_block.keys())):
                return False
        return True

    def load_transform_shell_metadata(self, layer_name: str):
        with h5py.File(self.diags_path, "r") as handle:
            layer = handle[str(layer_name)]
            output_rotations = int(layer["output_rotations"][()])
            diagonals: dict[tuple[int, int], dict[int, list[float]]] = {}
            for block_name in layer["diagonals"].keys():
                row_raw, col_raw = str(block_name).split("_", 1)
                block = layer["diagonals"][str(block_name)]
                diagonals[(int(row_raw), int(col_raw))] = {
                    int(diag_name): []
                    for diag_name in block.keys()
                }
        return diagonals, int(output_rotations)

    def _trim_backend_runtime_memory(self) -> None:
        trim = getattr(self.backend, "TrimRuntimeMemory", None)
        if callable(trim):
            trim()

    def _generate_transforms_batch(
        self,
        diagonals,
        *,
        level: int,
        bsgs_ratio: float,
        io_mode: str,
        on_batch_result=None,
    ):
        generate_batch = getattr(self.backend, "GenerateLinearTransformsBatch", None)
        if not callable(generate_batch) or len(diagonals) <= 1:
            return None

        items = list(diagonals.items())
        batch_limit = self._lt_compile_batch_limit(
            len(items),
            index_only=(str(io_mode) == "load" and self.io_mode == "save"),
        )
        results = []
        offset = 0
        while offset < len(items):
            batch_items = items[int(offset): int(offset + batch_limit)]
            workers = self._lt_worker_count(len(batch_items))
            batch_started = time.perf_counter()
            if workers <= 1:
                block_payloads = [self._build_block_payload(item) for item in batch_items]
            else:
                futures = []
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="orion-lt-build") as pool:
                    for index, item in enumerate(batch_items):
                        futures.append((index, pool.submit(self._build_block_payload, item)))
                    ordered = [None] * len(futures)
                    wait_started = time.perf_counter()
                    for index, future in futures:
                        ordered[index] = future.result()
                    self._add_profile("wait_s", time.perf_counter() - wait_started)
                block_payloads = list(ordered)
            self._add_profile("diag_generate_s", time.perf_counter() - batch_started)

            num_transforms = len(block_payloads)
            diag_idxs_ptrs = (ctypes.POINTER(ctypes.c_int) * num_transforms)()
            diag_idxs_lens = (ctypes.c_int * num_transforms)()
            diag_data_ptrs = (ctypes.POINTER(ctypes.c_float) * num_transforms)()
            diag_data_lens = (ctypes.c_int * num_transforms)()
            levels = (ctypes.c_int * num_transforms)(*[int(level)] * num_transforms)

            owned_arrays: list[object] = [levels]
            for index, (_row, _col, diags_idxs, _diags_data) in enumerate(block_payloads):
                array = (ctypes.c_int * len(diags_idxs))(*diags_idxs)
                owned_arrays.append(array)
                diag_idxs_ptrs[index] = array
                diag_idxs_lens[index] = len(diags_idxs)
            for index, (_row, _col, _diags_idxs, diags_data) in enumerate(block_payloads):
                array = (ctypes.c_float * len(diags_data))(*diags_data)
                owned_arrays.append(array)
                diag_data_ptrs[index] = array
                diag_data_lens[index] = len(diags_data)

            encode_started = time.perf_counter()
            lintransf_ids = list(
                generate_batch(
                    int(num_transforms),
                    diag_idxs_ptrs,
                    diag_idxs_lens,
                    diag_data_ptrs,
                    diag_data_lens,
                    levels,
                    float(bsgs_ratio),
                    str(io_mode),
                )
            )
            self._add_profile("encode_s", time.perf_counter() - encode_started)
            batch_results = [
                (row, col, diags_idxs, int(lintransf_id))
                for (row, col, diags_idxs, _diags_data), lintransf_id in zip(block_payloads, lintransf_ids)
            ]
            if callable(on_batch_result):
                on_batch_result(batch_results)
            results.extend(batch_results)
            del block_payloads, owned_arrays, lintransf_ids
            gc.collect()
            offset += int(batch_limit)
        return results
    
    def get_required_rotation_keys(self, transform_id):
        return self.backend.GetLinearTransformRotationKeys(transform_id)

    def get_required_rotation_key_requests(self, transform_id):
        get_requests = getattr(self.backend, "GetLinearTransformRotationKeyRequests", None)
        if callable(get_requests):
            flat = list(get_requests(transform_id))
            if len(flat) % 2 != 0:
                raise RuntimeError("backend returned malformed rotation key requests")
            requests = {}
            for index in range(0, len(flat), 2):
                key = int(flat[index])
                level = int(flat[index + 1])
                requests[key] = max(level, requests.get(key, level))
            return tuple(sorted(requests.items()))
        return tuple((int(key), None) for key in self.get_required_rotation_keys(transform_id))

    def _rotation_key_storage_name(self, key, level):
        return str(int(key)) if level is None else f"{int(key)}@{int(level)}"

    def _generate_rotation_key(self, key, level):
        if level is not None and hasattr(self.backend, "GenerateLinearTransformRotationKeyAtLevel"):
            self.backend.GenerateLinearTransformRotationKeyAtLevel(int(key), int(level))
        else:
            self.backend.GenerateLinearTransformRotationKey(int(key))

    def _generate_and_serialize_rotation_key(self, key, level):
        if level is not None and hasattr(self.backend, "GenerateAndSerializeRotationKeyAtLevel"):
            return self.backend.GenerateAndSerializeRotationKeyAtLevel(int(key), int(level))
        return self.backend.GenerateAndSerializeRotationKey(int(key))

    def generate_rotation_keys(self, transform_id):
        started = time.perf_counter()
        curr_keys = self.get_required_rotation_key_requests(transform_id)

        # Only generate keys that don't exist yet. Depending on the I/O
        # mode, we may also save these keys immediately rather than keep
        # them in RAM.
        keys_to_gen = set(curr_keys).difference(self.saved_rotation_keys)
        self.saved_rotation_keys.update(keys_to_gen)

        if self.io_mode == "none":
            for key, level in keys_to_gen:
                self._generate_rotation_key(key, level)

        elif self.io_mode in ("save", "load"):
            with h5py.File(self.keys_path, "a") as f:
                for key, level in keys_to_gen:
                    key_str = self._rotation_key_storage_name(key, level)
                    if key_str in f or str(int(key)) in f: # don't regenerate the key
                        continue
                    
                    # We'll generate, serialize, and then save the key
                    serial_key, ptr = self._generate_and_serialize_rotation_key(key, level)
                    try:
                        f.create_dataset(key_str, data=serial_key)
                    finally:
                        self.backend.FreeCArray(ptr)
        self._add_profile("key_prepare_s", time.perf_counter() - started)

    def save_transforms(self, linear_layer):
        layer_name = linear_layer.name
        diagonals = linear_layer.diagonals 
        on_bias = linear_layer.on_bias 
        output_rotations = linear_layer.output_rotations 
        input_shape = linear_layer.input_shape 
        output_shape = linear_layer.output_shape
        input_min = linear_layer.input_min
        input_max = linear_layer.input_max
        output_min = linear_layer.output_min 
        output_max = linear_layer.output_max

        print("└── saving... ", end="", flush=True)
        with h5py.File(self.diags_path, "a") as f:
            if layer_name in f:
                del f[layer_name]
            layer = f.create_group(layer_name)

            layer.create_dataset("embedding_method", data=self.embed_method)
            layer.create_dataset("output_rotations", data=output_rotations)
            layer.create_dataset("on_bias", data=on_bias.numpy())
            layer.create_dataset("input_shape", data=list(input_shape))
            layer.create_dataset("output_shape", data=list(output_shape))
            layer.create_dataset("input_min", data=input_min.item())
            layer.create_dataset("input_max", data=input_max.item())
            layer.create_dataset("output_min", data=output_min.item())
            layer.create_dataset("output_max", data=output_max.item())

            diags_group = layer.require_group("diagonals")
            for (row, col), diags in diagonals.items():
                block_idx = f"{row}_{col}"
                block_diags_group = diags_group.create_group(block_idx)
                
                # Iterate over all diagonals in the block and save
                for diag_idx, diag_data in diags.items():
                    block_diags_group.create_dataset(str(diag_idx), data=diag_data)
                    diags[diag_idx] = [] # delete after saving

        print("done!")

    def load_transforms(self, linear_layer):
        self._verify_layer_compatibility(linear_layer)

        layer_name = linear_layer.name
        on_bias = linear_layer.on_bias
        manifest = self._cached_manifest() if self.io_mode == "load" else None
        if (
            manifest is not None
            and bool(getattr(self.backend, "supports_index_only_linear_transform_load", False))
        ):
            return (
                compile_cache.transform_blocks_from_manifest(manifest, str(layer_name)),
                on_bias,
                compile_cache.transform_output_rotations(manifest, str(layer_name)),
            )

        read_started = time.perf_counter()
        with h5py.File(self.diags_path, "r") as f:
            layer = f[layer_name]
            output_rotations = (
                compile_cache.transform_output_rotations(manifest, str(layer_name))
                if manifest is not None
                else int(layer["output_rotations"][()])
            )

            # Load the diagonals back into the correct struct
            all_diagonals = {}
            diag_group = layer["diagonals"]
            reuse_saved_plaintexts = (
                self.io_mode == "save"
                and self._compile_save_resume_enabled_for_layer(linear_layer)
                and bool(getattr(self.backend, "supports_index_only_linear_transform_load", False))
                and self._plaintext_blocks_complete(layer, diag_group)
            )
            linear_layer._compile_cache_reuse_saved_plaintexts = bool(reuse_saved_plaintexts)
            for block in diag_group:
                row, col = map(int, block.split("_")) # 0_1 -> (0,1)
                diags = {}
                block_group = diag_group[block]
                for diag_idx in block_group:
                    if (
                        (self.io_mode == "load" or bool(reuse_saved_plaintexts))
                        and bool(getattr(self.backend, "supports_index_only_linear_transform_load", False))
                    ):
                        # Load mode only needs diagonal indices to rebuild the
                        # in-process transform shell. Encoded plaintext
                        # diagonals are loaded later from the plaintext cache.
                        diags[int(diag_idx)] = []
                    else:
                        diag_data = block_group[diag_idx][:]
                        diags[int(diag_idx)] = diag_data
                all_diagonals[(row, col)] = diags

        self._add_profile("read_s", time.perf_counter() - read_started)
        return all_diagonals, on_bias, output_rotations

    def load_transform_metadata(self, linear_layer):
        manifest = self._cached_manifest() if self.io_mode == "load" else None
        if manifest is not None:
            return compile_cache.transform_output_rotations(manifest, str(linear_layer.name))
        self._verify_layer_compatibility(linear_layer, check_output_rotations=False)
        with h5py.File(self.diags_path, "r") as f:
            return int(f[linear_layer.name]["output_rotations"][()])

    def evaluate_transforms(self, linear_layer, in_ctensor):
        layer_name = linear_layer.name
        out_shape = linear_layer.output_shape
        fhe_out_shape = linear_layer.fhe_output_shape 
        skip_post_rescale = bool(getattr(self.backend, "lt_outputs_are_rescaled", False))

        rows, cols, transform_ids = self._transform_id_matrix(
            getattr(linear_layer, "transform_ids", {}) or {},
            input_cols=len(in_ctensor),
        )
        work_items = [
            (int(i), int(j), int(transform_ids[i][j]))
            for i in range(rows)
            for j in range(cols)
            if int(transform_ids[i][j]) >= 0
        ]
        dense_rectangular = int(len(work_items)) == int(rows * cols)
        use_dense_shared_cache = bool(dense_rectangular) and self._dense_shared_cache_enabled(rows=rows, cols=cols)
        linear_layer._last_dense_shared_cache_eval = {
            "enabled": bool(use_dense_shared_cache),
            "group_count": int(cols) if bool(use_dense_shared_cache) else 0,
            "group_axis": "input_column" if bool(use_dense_shared_cache) else "",
            "transform_count": int(len(work_items)),
            "io_mode": str(self.io_mode),
            "host_payload_cache": bool(self._dense_host_payload_cache_enabled()),
            "host_payload_cache_count": int(len(self._dense_host_plaintext_payload_cache)),
        }
        if bool(use_dense_shared_cache):
            return self._evaluate_transforms_with_dense_shared_cache(
                linear_layer,
                in_ctensor,
                rows=int(rows),
                cols=int(cols),
                transform_ids=transform_ids,
                out_shape=out_shape,
                fhe_out_shape=fhe_out_shape,
                skip_post_rescale=bool(skip_post_rescale),
            )
        prefetch_sequence, use_global_prefetch_sequence = self._transform_prefetch_sequence(
            layer_name,
            work_items,
        )
        if self.io_mode != "none" and work_items:
            first_key = self._transform_io_key(layer_name, *work_items[0])
            first_index = self._transform_sequence_index(
                first_key,
                fallback=0,
                use_global_sequence=use_global_prefetch_sequence,
            )
            self._fill_transform_io_prefetch_window(
                prefetch_sequence,
                first_index,
                scratch_reserve_bytes=self._estimate_transform_scratch_reserve_bytes(work_items[0][2]),
            )
        work_index = 0
        cts_out = []
        for i in range(rows):
            ct_out = None
            for j in range(cols):
                t_id = transform_ids[i][j]
                if int(t_id) < 0:
                    continue

                if self.io_mode != "none":
                    current_key = self._transform_io_key(layer_name, int(i), int(j), int(t_id))
                    self._forward_memory_guard(
                        reason=f"before_lt_block_load:{layer_name}:{int(i)}:{int(j)}",
                        needed_bytes=self._estimate_transform_device_bytes(
                            layer_name,
                            int(i),
                            int(j),
                            int(t_id),
                        ),
                    )
                    bundle = self._transform_io_prefetcher.consume(current_key)
                    if bundle is None:
                        bundle = self._read_transform_io_bundle(
                            layer_name,
                            int(i),
                            int(j),
                            int(t_id),
                            prefetch=False,
                        )
                    self.load_rotation_keys(t_id, bundle=bundle)
                    self.load_plaintext_diagonals(layer_name, i, j, t_id, bundle=bundle)
                    self.ensure_plaintext_diagonals_loaded(
                        layer_name,
                        i,
                        j,
                        t_id,
                        expected_level=int(linear_layer.level),
                    )
                    current_index = self._transform_sequence_index(
                        current_key,
                        fallback=work_index,
                        use_global_sequence=use_global_prefetch_sequence,
                    )
                    self._fill_transform_io_prefetch_window(
                        prefetch_sequence,
                        int(current_index + 1),
                        scratch_reserve_bytes=self._estimate_transform_scratch_reserve_bytes(t_id),
                    )

                res = self.backend.EvaluateLinearTransform(t_id, in_ctensor.ids[j]) 
                ct = CipherTensor(self.scheme, res, out_shape, fhe_out_shape)

                # Accumulate results across a row of blocks
                ct_out = ct if ct_out is None else ct_out + ct
                    
                if self.io_mode != "none":
                    self.remove_rotation_keys(t_id)
                    self.remove_plaintext_diagonals(t_id)
                    if bundle is not None:
                        bundle.clear()
                    self._forward_memory_guard(
                        reason=f"after_lt_block_unload:{layer_name}:{int(i)}:{int(j)}",
                        force_trim=True,
                        raise_on_low=False,
                    )
                work_index += 1

            if ct_out is None:
                raise ValueError(f"Linear transform row {int(i)} has no nonempty blocks")
            
            # We know the output of this accumulation will just be one ciphertext
            if skip_post_rescale:
                cts_out.append(int(ct_out.ids[0]))
                ct_out.ids = []
            else:
                ct_out_rescaled = self.evaluator.rescale(ct_out.ids[0], in_place=False)
                cts_out.append(ct_out_rescaled)

        return CipherTensor(self.scheme, cts_out, out_shape, fhe_out_shape)

    def _dense_shared_cache_enabled(self, *, rows: int, cols: int) -> bool:
        override = os.environ.get("ORION_DENSE_LT_SHARED_CACHE")
        if override is None or override.lower() not in ("1", "true", "yes", "on"):
            return False
        if not callable(getattr(self.backend, "EvaluateLinearTransformsWithSharedCache", None)):
            return False
        if self.io_mode != "none":
            # Save/load mode needs coordinated chunk IO and eviction. Keep the
            # established scalar path there until the cached path is taught the
            # same column-wise schedule.
            return False
        return int(rows) > 1 and int(cols) > 0

    def _evaluate_transforms_with_dense_shared_cache(
        self,
        linear_layer,
        in_ctensor,
        *,
        rows: int,
        cols: int,
        transform_ids,
        out_shape,
        fhe_out_shape,
        skip_post_rescale: bool,
    ):
        cts_out = [None for _ in range(int(rows))]
        for j in range(int(cols)):
            column_ids = [int(transform_ids[i][j]) for i in range(int(rows))]
            loaded_from_host_cache: list[int] = []
            if self._dense_host_payload_cache_enabled():
                for transform_id in column_ids:
                    payload = self._dense_host_plaintext_payload_cache.get(int(transform_id))
                    if payload is None:
                        continue
                    self.backend.LoadLinearTransformPlaintexts(payload, int(transform_id))
                    loaded_from_host_cache.append(int(transform_id))
            transform_ids_array = (ctypes.c_int * len(column_ids))(*column_ids)
            try:
                result_ids = list(
                    self.backend.EvaluateLinearTransformsWithSharedCache(
                        transform_ids_array,
                        len(column_ids),
                        int(in_ctensor.ids[j]),
                    )
                )
            finally:
                for transform_id in loaded_from_host_cache:
                    self.backend.RemovePlaintextDiagonals(int(transform_id))
                if loaded_from_host_cache:
                    self._forward_memory_guard(
                        reason="after_dense_shared_cache_unload",
                        force_trim=True,
                        raise_on_low=False,
                    )
            if len(result_ids) != len(column_ids):
                raise RuntimeError(
                    "EvaluateLinearTransformsWithSharedCache returned "
                    f"{len(result_ids)} outputs for {len(column_ids)} dense transforms"
                )
            for i, res in enumerate(result_ids):
                ct = CipherTensor(self.scheme, int(res), out_shape, fhe_out_shape)
                cts_out[int(i)] = ct if cts_out[int(i)] is None else cts_out[int(i)] + ct

        output_ids = []
        for i, ct_out in enumerate(cts_out):
            if ct_out is None:
                raise RuntimeError(f"dense shared-cache LT missing output row {int(i)}")
            if bool(skip_post_rescale):
                output_ids.append(int(ct_out.ids[0]))
                ct_out.ids = []
            else:
                output_ids.append(int(self.evaluator.rescale(ct_out.ids[0], in_place=False)))
        return CipherTensor(self.scheme, output_ids, out_shape, fhe_out_shape)

    def _transform_id_matrix(self, transform_ids: dict, *, input_cols: int):
        work_items = self._work_items_from_transform_ids(transform_ids)
        if not work_items:
            return 0, 0, np.zeros((0, 0), dtype=int)

        rows = max(int(row) for row, _col, _transform_id in work_items) + 1
        cols = max(int(col) for _row, col, _transform_id in work_items) + 1
        if int(input_cols) < int(cols):
            raise ValueError(
                f"Linear transform block column count {cols} does not match "
                f"input ciphertext count {int(input_cols)}"
            )
        cols = int(input_cols)

        matrix = np.full((int(rows), int(cols)), -1, dtype=int)
        for row, col, transform_id in work_items:
            matrix[int(row)][int(col)] = int(transform_id)
        return int(rows), int(cols), matrix
            
    def delete_transforms(self, transform_ids: dict):
        for tid in transform_ids.values():
            self.backend.DeleteLinearTransform(tid)

    def _verify_layer_compatibility(self, linear_layer, *, check_output_rotations: bool = True):
        layer_name = linear_layer.name

        # -------- Current network values -------- #

        curr_embed_method = linear_layer.scheme.params.get_embedding_method()
        curr_output_rotations = linear_layer.output_rotations
        curr_on_bias = linear_layer.on_bias
        curr_input_shape = linear_layer.input_shape 
        curr_output_shape = linear_layer.output_shape
        curr_input_min = linear_layer.input_min 
        curr_input_max = linear_layer.input_max
        curr_output_min = linear_layer.output_min
        curr_output_max = linear_layer.output_max

        # ------- Previous network values ------- #

        with h5py.File(self.diags_path, "r") as f:

            # Check if the layer exists in the h5py file
            if layer_name not in f:
                raise ValueError(
                    f"Layer '{layer_name}' not found in file {self.diags_path}. " + 
                    "First set IO mode in parameters YAML file to `save`."
                )
            
            layer = f[layer_name]
            
            last_embed_method = layer["embedding_method"][()].decode("utf-8")
            last_output_rotations = layer["output_rotations"][()]
            last_on_bias = torch.tensor(layer["on_bias"][:])
            last_input_shape = torch.Size(layer["input_shape"][:])
            last_output_shape = torch.Size(layer["output_shape"][:])
            last_input_min = layer["input_min"][()]
            last_input_max = layer["input_max"][()]
            last_output_min = layer["output_min"][()]
            last_output_max = layer["output_max"][()]

            # Check each parameter and collect mismatches
            mismatches = []
                            
            if curr_on_bias.shape != last_on_bias.shape:
                mismatches.append(f"on_bias: shape mismatch")
            elif not torch.allclose(curr_on_bias, last_on_bias):
                mismatches.append(f"on_bias: values mismatch")
            
            # Simple equality checks
            if check_output_rotations and curr_output_rotations != last_output_rotations:
                mismatches.append(f"output_rotations mismatch")

            if curr_input_shape != last_input_shape:
                mismatches.append(f"input_shape mismatch")
            
            if curr_output_shape != last_output_shape:
                mismatches.append(f"output_shape mismatch")
            
            if curr_embed_method != last_embed_method:
                mismatches.append(f"embedding_method mismatch")
            
            if curr_input_min != last_input_min:
                mismatches.append(f"input_min mismatch")
            
            if curr_input_max != last_input_max:
                mismatches.append(f"input_max mismatch")
            
            if curr_output_min != last_output_min:
                mismatches.append(f"output_min mismatch")
            
            if curr_output_max != last_output_max:
                mismatches.append(f"output_max mismatch")
            
            # If there are mismatches, raise a detailed error
            if mismatches:
                error_msg = "Saved network does not match currently instantiated network: "
                error_msg += ", ".join(mismatches)
                error_msg += ". First set IO mode in parameters YAML file to `save` to "
                error_msg += "override existing data. Then loading will work."
                
                raise ValueError(error_msg)
            
    def save_plaintext_diagonals(self, layer_name, lintransf_id, row, col, diag_idxs):
        started = time.perf_counter()
        try:
            with h5py.File(self.diags_path, "a") as f:
                layer = f.require_group(layer_name)
                plaintext_group = layer.require_group("plaintexts")
                block_idx = f"{row}_{col}"
                if block_idx in plaintext_group:
                    del plaintext_group[block_idx]
                block_group = plaintext_group.create_group(block_idx)

                if self._encoded_plaintext_payload_supported():
                    serial_payload, payload_ptr = self.backend.SerializeLinearTransformPlaintexts(
                        int(lintransf_id)
                    )
                    try:
                        block_group.create_dataset(
                            _ENCODED_HOIST_PAYLOAD_DATASET,
                            data=serial_payload,
                        )
                    finally:
                        self.backend.FreeCArray(payload_ptr)
                        self.backend.RemovePlaintextDiagonals(int(lintransf_id))
                    return

                if not self._plaintext_payload_required():
                    for diag_idx in diag_idxs:
                        block_group.create_dataset(str(int(diag_idx)), data=np.zeros((0,), dtype=np.uint8))
                    self.backend.RemovePlaintextDiagonals(int(lintransf_id))
                    return

                for diag_idx in diag_idxs:
                    diag_serial, diag_ptr = self.backend.SerializeDiagonal(lintransf_id, diag_idx)
                    block_group.create_dataset(str(diag_idx), data=diag_serial)

                    # Now that it's saved, we'll free the memory
                    self.backend.FreeCArray(diag_ptr)
                self.backend.RemovePlaintextDiagonals(int(lintransf_id))
        finally:
            self._add_profile("serialize_s", time.perf_counter() - started)

    def _plaintext_payload_required(self) -> bool:
        return bool(getattr(self.backend, "load_plaintext_diagonals_requires_payload", True))

    def _encoded_plaintext_payload_supported(self) -> bool:
        return callable(getattr(self.backend, "SerializeLinearTransformPlaintexts", None)) and callable(
            getattr(self.backend, "LoadLinearTransformPlaintexts", None)
        )

    def _device_transform_prefetch_supported(self) -> bool:
        if not bool(getattr(self.backend, "saved_io_device_prefetch_enabled", False)):
            return False
        return self._encoded_plaintext_payload_supported() and callable(
            getattr(self.backend, "LoadLinearTransformRotationKey", None)
        )

    def _transform_io_key(self, layer_name, row, col, transform_id):
        return (str(layer_name), int(row), int(col), int(transform_id))

    def _transform_prefetch_sequence(self, layer_name, work_items):
        if not work_items:
            return (), False
        first_key = self._transform_io_key(layer_name, *work_items[0])
        if first_key in self._saved_io_work_index:
            return self._saved_io_work_order, True
        return tuple(
            (str(layer_name), int(row), int(col), int(transform_id))
            for row, col, transform_id in work_items
        ), False

    def _transform_sequence_index(self, key, *, fallback: int, use_global_sequence: bool) -> int:
        if use_global_sequence:
            return int(self._saved_io_work_index.get(key, int(fallback)))
        return int(fallback)

    def _fill_transform_io_prefetch_window(
        self,
        sequence,
        start_index: int,
        *,
        scratch_reserve_bytes: int = 0,
    ) -> None:
        if self.io_mode == "none" or self._transform_io_lookahead <= 0:
            return
        if not sequence:
            return

        index = max(0, int(start_index))
        window_count = 0
        while index < len(sequence) and window_count < self._transform_io_lookahead:
            key = sequence[index]
            if self._transform_io_prefetcher.has_pending(key):
                window_count += 1
                index += 1
                continue
            submitted = self._submit_saved_io_work_prefetch(
                key,
                scratch_reserve_bytes=int(scratch_reserve_bytes or 0),
                reserved_host_bytes=self._transform_io_prefetcher.pending_host_bytes(),
                reserved_device_bytes=self._transform_io_prefetcher.pending_device_bytes(),
            )
            if not submitted:
                break
            window_count += 1
            index += 1

    def _resolve_saved_io_size(self, source, key) -> int:
        value = source.get(key, 0)
        if callable(value):
            return int(value())
        return int(value or 0)

    def _submit_saved_io_work_prefetch(
        self,
        key,
        *,
        scratch_reserve_bytes=0,
        reserved_host_bytes=0,
        reserved_device_bytes=0,
    ):
        if key in self._saved_io_external_loaders:
            return self._submit_external_saved_io_prefetch(
                key,
                scratch_reserve_bytes=int(scratch_reserve_bytes or 0),
                reserved_host_bytes=int(reserved_host_bytes or 0),
                reserved_device_bytes=int(reserved_device_bytes or 0),
            )
        layer_name, row, col, transform_id = key
        return self._submit_transform_io_prefetch(
            layer_name,
            int(row),
            int(col),
            int(transform_id),
            scratch_reserve_bytes=int(scratch_reserve_bytes or 0),
            reserved_host_bytes=int(reserved_host_bytes or 0),
            reserved_device_bytes=int(reserved_device_bytes or 0),
        )

    def _submit_external_saved_io_prefetch(
        self,
        key,
        *,
        scratch_reserve_bytes=0,
        reserved_host_bytes=0,
        reserved_device_bytes=0,
    ):
        if self.io_mode == "none":
            return False
        host_bytes = self._resolve_saved_io_size(self._saved_io_external_host_bytes, key)
        device_bytes = self._resolve_saved_io_size(self._saved_io_external_device_bytes, key)
        if not should_prefetch_saved_io(
            int(host_bytes) + int(reserved_host_bytes or 0),
            backend=self.backend,
            device_bytes=(
                int(device_bytes)
                + int(scratch_reserve_bytes or 0)
                + int(reserved_device_bytes or 0)
            ),
        ):
            return False
        return self._transform_io_prefetcher.submit(
            key,
            self._saved_io_external_loaders[key],
            host_bytes=int(host_bytes),
            device_bytes=int(device_bytes),
        )

    def _estimate_transform_io_bundle_bytes(self, layer_name, row, col, transform_id):
        cache_key = self._transform_io_key(layer_name, row, col, transform_id)
        if cache_key in self._transform_io_size_cache:
            return int(self._transform_io_size_cache[cache_key])

        total_bytes = 0
        if self.keys_path:
            with h5py.File(self.keys_path, "r") as f:
                for key_value, level in self.get_required_rotation_key_requests(transform_id):
                    key_str = self._rotation_key_storage_name(key_value, level)
                    if key_str not in f and str(int(key_value)) in f:
                        key_str = str(int(key_value))
                    dataset = f[key_str]
                    total_bytes += int(dataset.size) * int(dataset.dtype.itemsize)

        if self.diags_path and self._plaintext_payload_required():
            with h5py.File(self.diags_path, "r") as f:
                block = f[layer_name]["plaintexts"][f"{row}_{col}"]
                if _ENCODED_HOIST_PAYLOAD_DATASET in block:
                    dataset = block[_ENCODED_HOIST_PAYLOAD_DATASET]
                    total_bytes += int(dataset.size) * int(dataset.dtype.itemsize)
                else:
                    for diag_idx in block:
                        dataset = block[diag_idx]
                        total_bytes += int(dataset.size) * int(dataset.dtype.itemsize)
        elif self.diags_path and self._encoded_plaintext_payload_supported():
            with h5py.File(self.diags_path, "r") as f:
                block = f[layer_name]["plaintexts"][f"{row}_{col}"]
                if _ENCODED_HOIST_PAYLOAD_DATASET in block:
                    dataset = block[_ENCODED_HOIST_PAYLOAD_DATASET]
                    total_bytes += int(dataset.size) * int(dataset.dtype.itemsize)

        self._transform_io_size_cache[cache_key] = int(total_bytes)
        return int(total_bytes)

    def _estimate_transform_device_bytes(self, layer_name, row, col, transform_id):
        transform_key = int(transform_id)
        if transform_key in self._transform_device_size_cache:
            return int(self._transform_device_size_cache[transform_key])

        total_bytes = self._estimate_transform_io_bundle_bytes(
            layer_name,
            row,
            col,
            transform_id,
        )
        total_bytes += estimate_linear_transform_device_bytes(self.backend, int(transform_id))
        self._transform_device_size_cache[transform_key] = int(total_bytes)
        return int(total_bytes)

    def _estimate_transform_scratch_reserve_bytes(self, transform_id):
        transform_bytes = estimate_linear_transform_device_bytes(self.backend, int(transform_id))
        return max(512 * 1024 * 1024, int(transform_bytes))

    def _read_transform_io_bundle(self, layer_name, row, col, transform_id, *, prefetch: bool):
        if prefetch:
            if not should_prefetch_saved_io(
                self._estimate_transform_io_bundle_bytes(layer_name, row, col, transform_id),
                backend=self.backend,
                device_bytes=self._estimate_transform_device_bytes(layer_name, row, col, transform_id),
            ):
                return None

        bundle = {
            "rotation_keys": (),
            "diag_indices": (),
            "offsets": (),
            "lengths": (),
            "payload": np.zeros((0,), dtype=np.uint8),
            "encoded_plaintext_payload": None,
            "plaintexts": (),
        }

        if self.keys_path:
            rotation_keys = []
            with h5py.File(self.keys_path, "r") as f:
                for key_value, level in self.get_required_rotation_key_requests(transform_id):
                    key_str = self._rotation_key_storage_name(key_value, level)
                    if key_str not in f and str(int(key_value)) in f:
                        key_str = str(int(key_value))
                    rotation_keys.append((int(key_value), np.asarray(f[key_str][()], dtype=np.uint8).copy()))
            bundle["rotation_keys"] = tuple(rotation_keys)

        if not self.diags_path:
            return bundle

        with h5py.File(self.diags_path, "r") as f:
            block = f[layer_name]["plaintexts"][f"{row}_{col}"]
            if _ENCODED_HOIST_PAYLOAD_DATASET in block:
                bundle["encoded_plaintext_payload"] = np.asarray(
                    block[_ENCODED_HOIST_PAYLOAD_DATASET][()],
                    dtype=np.uint8,
                ).reshape(-1).copy()
                return bundle

        if not self._plaintext_payload_required():
            with h5py.File(self.diags_path, "r") as f:
                block = f[layer_name]["plaintexts"][f"{row}_{col}"]
                bundle["diag_indices"] = tuple(sorted(int(diag_idx) for diag_idx in block.keys()))
            return bundle

        payload_chunks = []
        offsets = []
        lengths = []
        plaintexts = []
        cursor = 0
        with h5py.File(self.diags_path, "r") as f:
            block = f[layer_name]["plaintexts"][f"{row}_{col}"]
            diag_indices = sorted(int(diag_idx) for diag_idx in block.keys())
            for diag_idx in diag_indices:
                serial_diag = np.asarray(block[str(diag_idx)][()], dtype=np.uint8).reshape(-1).copy()
                plaintexts.append((int(diag_idx), serial_diag))
                offsets.append(int(cursor))
                lengths.append(int(serial_diag.size))
                payload_chunks.append(serial_diag)
                cursor += int(serial_diag.size)
        bundle["diag_indices"] = tuple(diag_idx for diag_idx, _ in plaintexts)
        bundle["plaintexts"] = tuple(plaintexts)
        bundle["offsets"] = tuple(offsets)
        bundle["lengths"] = tuple(lengths)
        if payload_chunks:
            bundle["payload"] = np.concatenate(payload_chunks)
        return bundle

    def _submit_transform_io_prefetch(
        self,
        layer_name,
        row,
        col,
        transform_id,
        *,
        scratch_reserve_bytes=0,
        reserved_host_bytes=0,
        reserved_device_bytes=0,
    ):
        if self.io_mode == "none":
            return False
        host_bytes = self._estimate_transform_io_bundle_bytes(
            layer_name,
            row,
            col,
            transform_id,
        )
        device_bytes = self._estimate_transform_device_bytes(
            layer_name,
            row,
            col,
            transform_id,
        )
        if not should_prefetch_saved_io(
            int(host_bytes) + int(reserved_host_bytes or 0),
            backend=self.backend,
            device_bytes=(
                int(device_bytes)
                + int(scratch_reserve_bytes or 0)
                + int(reserved_device_bytes or 0)
            ),
        ):
            return False
        key = self._transform_io_key(layer_name, row, col, transform_id)

        def load_bundle():
            bundle = self._read_transform_io_bundle(
                layer_name,
                row,
                col,
                transform_id,
                prefetch=False,
            )
            if bundle is None or not self._device_transform_prefetch_supported():
                return bundle
            load_transform_key = self.backend.LoadLinearTransformRotationKey
            for key_value, serial_key in bundle.get("rotation_keys", ()):
                load_transform_key(serial_key, int(key_value), int(transform_id))
            if bundle.get("rotation_keys"):
                bundle["rotation_keys_prefetched_to_device"] = True
                bundle["rotation_keys"] = ()
            encoded_payload = bundle.get("encoded_plaintext_payload")
            if encoded_payload is not None:
                self.backend.LoadLinearTransformPlaintexts(
                    encoded_payload,
                    int(transform_id),
                )
                bundle["encoded_plaintext_payload"] = None
                bundle["plaintexts_prefetched_to_device"] = True
            return bundle

        return self._transform_io_prefetcher.submit(
            key,
            load_bundle,
            host_bytes=int(host_bytes),
            device_bytes=int(device_bytes),
        )

    def load_plaintext_diagonals(self, layer_name, row, col, transform_id, bundle=None):
        if bundle is not None:
            if bundle.get("plaintexts_prefetched_to_device"):
                return
            encoded_payload = bundle.get("encoded_plaintext_payload")
            if encoded_payload is not None:
                self.backend.LoadLinearTransformPlaintexts(
                    encoded_payload,
                    int(transform_id),
                )
                bundle["encoded_plaintext_payload"] = None
                return
            diag_indices = list(bundle.get("diag_indices", ()))
            if hasattr(self.backend, "LoadPlaintextDiagonalsBatch"):
                self.backend.LoadPlaintextDiagonalsBatch(
                    bundle.get("payload", np.zeros((0,), dtype=np.uint8)),
                    list(bundle.get("offsets", ())),
                    list(bundle.get("lengths", ())),
                    diag_indices,
                    int(transform_id),
                )
                bundle["payload"] = np.zeros((0,), dtype=np.uint8)
                bundle["offsets"] = ()
                bundle["lengths"] = ()
                bundle["plaintexts"] = ()
                return
            for diag_idx, serial_diag in bundle.get("plaintexts", ()):
                self.backend.LoadPlaintextDiagonal(
                    serial_diag,
                    transform_id,
                    int(diag_idx),
                )
            bundle["plaintexts"] = ()
            return
        with h5py.File(self.diags_path, "r") as f:
            layer = f[layer_name]
            ptxt_group = layer["plaintexts"]
            block = ptxt_group[f"{row}_{col}"]
            if _ENCODED_HOIST_PAYLOAD_DATASET in block:
                payload = np.asarray(
                    block[_ENCODED_HOIST_PAYLOAD_DATASET][()],
                    dtype=np.uint8,
                ).reshape(-1)
                self.backend.LoadLinearTransformPlaintexts(
                    payload,
                    int(transform_id),
                )
                del payload
                return
            if not self._plaintext_payload_required() and hasattr(self.backend, "LoadPlaintextDiagonalsBatch"):
                diag_indices = sorted(int(diag_idx) for diag_idx in block.keys())
                self.backend.LoadPlaintextDiagonalsBatch(
                    np.zeros((0,), dtype=np.uint8),
                    [],
                    [],
                    diag_indices,
                    int(transform_id),
                )
                return

            for diag_idx in block:
                serial_diag = block[diag_idx][()]
            self.backend.LoadPlaintextDiagonal(
                serial_diag, transform_id, int(diag_idx)
            )

    def ensure_plaintext_diagonals_loaded(self, layer_name, row, col, transform_id, expected_level=None):
        get_empty_keys = getattr(self.backend, "GetLinearTransformEmptyPlaintextKeys", None)
        if callable(get_empty_keys):
            empty_keys = [int(key) for key in list(get_empty_keys(int(transform_id)))]
            if empty_keys:
                reloaded = self._reload_plaintext_diagonals(layer_name, row, col, transform_id, empty_keys)
                remaining = [int(key) for key in list(get_empty_keys(int(transform_id)))]
                if remaining:
                    raise RuntimeError(
                        "Linear transform plaintext diagonals are not loaded "
                        f"for layer={layer_name!r} block=({int(row)}, {int(col)}) "
                        f"transform_id={int(transform_id)} empty_keys={remaining[:16]} "
                        f"reloaded_keys={reloaded[:16]}"
                    )

        self._ensure_plaintext_diagonal_levels(
            layer_name,
            row,
            col,
            transform_id,
            expected_level=expected_level,
        )

    def _ensure_plaintext_diagonal_levels(self, layer_name, row, col, transform_id, expected_level=None):
        if expected_level is None:
            return
        get_levels = getattr(self.backend, "GetLinearTransformPlaintextLevels", None)
        if not callable(get_levels):
            return
        flat = [int(value) for value in list(get_levels(int(transform_id)))]
        if len(flat) % 2 != 0:
            raise RuntimeError("backend returned malformed linear transform plaintext level data")
        too_low = []
        for index in range(0, len(flat), 2):
            diag_idx = int(flat[index])
            actual_level = int(flat[index + 1])
            if actual_level < int(expected_level):
                too_low.append((diag_idx, actual_level))
        if too_low:
            raise RuntimeError(
                "Linear transform plaintext levels are incompatible "
                f"for layer={layer_name!r} block=({int(row)}, {int(col)}) "
                f"transform_id={int(transform_id)} expected_level={int(expected_level)} "
                f"bad_diags={too_low[:16]}"
            )

    def _reload_plaintext_diagonals(self, layer_name, row, col, transform_id, diag_indices):
        if not self.diags_path:
            return []
        reloaded = []
        with h5py.File(self.diags_path, "r") as f:
            block = f[str(layer_name)]["plaintexts"][f"{int(row)}_{int(col)}"]
            for diag_idx in diag_indices:
                key = str(int(diag_idx))
                if key not in block:
                    continue
                serial_diag = block[key][()]
                self.backend.LoadPlaintextDiagonal(serial_diag, int(transform_id), int(diag_idx))
                reloaded.append(int(diag_idx))
        return reloaded
    
    def load_rotation_keys(self, transform_id, bundle=None):
        load_transform_key = getattr(self.backend, "LoadLinearTransformRotationKey", None)
        if bundle is not None:
            if bundle.get("rotation_keys_prefetched_to_device"):
                return
            for key_value, serial_key in bundle.get("rotation_keys", ()):
                if callable(load_transform_key):
                    load_transform_key(serial_key, int(key_value), int(transform_id))
                else:
                    self.backend.LoadRotationKey(serial_key, int(key_value))
            bundle["rotation_keys"] = ()
            return
        keys = self.get_required_rotation_key_requests(transform_id)

        with h5py.File(self.keys_path, "r") as f:
            for key, level in keys:
                key_str = self._rotation_key_storage_name(key, level)
                if key_str not in f and str(int(key)) in f:
                    key_str = str(int(key))
                serial_key = f[key_str][()]
                if callable(load_transform_key):
                    load_transform_key(serial_key, int(key), int(transform_id))
                else:
                    self.backend.LoadRotationKey(serial_key, int(key))

    def remove_rotation_keys(self, transform_id=None):
        remove_transform_keys = getattr(self.backend, "RemoveLinearTransformRotationKeys", None)
        if transform_id is not None and callable(remove_transform_keys):
            remove_transform_keys(int(transform_id))
            return
        self.backend.RemoveRotationKeys() 

    def remove_plaintext_diagonals(self, transform_id):
        self.backend.RemovePlaintextDiagonals(transform_id)
