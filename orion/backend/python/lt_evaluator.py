import h5py
import ctypes
import torch
import numpy as np

from .io_prefetch import (
    AsyncIOPrefetcher,
    estimate_linear_transform_device_bytes,
    should_prefetch_saved_io,
)
from orion.backend.python.tensors import CipherTensor


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
        self._transform_io_prefetcher = AsyncIOPrefetcher()
        self._transform_io_size_cache: dict[tuple[str, int, int, int], int] = {}
        self._transform_device_size_cache: dict[int, int] = {}
        self.new_evaluator()

    def new_evaluator(self):
        self.backend.NewLinearTransformEvaluator()

    def generate_transforms(self, linear_layer):
        layer_name = linear_layer.name
        diagonals = linear_layer.diagonals 
        level = linear_layer.level
        bsgs_ratio = linear_layer.bsgs_ratio

        batch_ids = self._generate_transforms_batch(
            diagonals,
            level=int(level),
            bsgs_ratio=float(bsgs_ratio),
        )
        if batch_ids is not None:
            lintransf_ids = {}
            for row, col, diags_idxs, lintransf_id in batch_ids:
                lintransf_ids[(row, col)] = int(lintransf_id)
                self.generate_rotation_keys(int(lintransf_id))
                if self.io_mode == "save":
                    self.save_plaintext_diagonals(
                        layer_name, int(lintransf_id), int(row), int(col), diags_idxs
                    )
            return lintransf_ids

        # Generate all linear transforms block by block.
        lintransf_ids = {}        
        for (row, col), diags in diagonals.items(): 
            diags_idxs, diags_data = [], []
            for idx, diag in diags.items(): 
                diags_idxs.append(idx)
                diags_data.extend(diag)

            lintransf_id = self.backend.GenerateLinearTransform(
                diags_idxs, diags_data, level, bsgs_ratio, self.io_mode
            )
            lintransf_ids[(row, col)] = lintransf_id

            # Now we can generate any new rotation keys needed for
            # this linear transform.
            self.generate_rotation_keys(lintransf_id)
            if self.io_mode == "save":
                self.save_plaintext_diagonals(
                    layer_name, lintransf_id, row, col, diags_idxs
                )

        return lintransf_ids

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

    def _generate_transforms_batch(self, diagonals, *, level: int, bsgs_ratio: float):
        generate_batch = getattr(self.backend, "GenerateLinearTransformsBatch", None)
        if not callable(generate_batch) or len(diagonals) <= 1:
            return None

        block_payloads = []
        for (row, col), diags in diagonals.items():
            diags_idxs, diags_data = self._flatten_diagonals(diags)
            block_payloads.append((int(row), int(col), diags_idxs, diags_data))

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

        lintransf_ids = list(
            generate_batch(
                int(num_transforms),
                diag_idxs_ptrs,
                diag_idxs_lens,
                diag_data_ptrs,
                diag_data_lens,
                levels,
                float(bsgs_ratio),
                str(self.io_mode),
            )
        )
        return [
            (row, col, diags_idxs, int(lintransf_id))
            for (row, col, diags_idxs, _diags_data), lintransf_id in zip(block_payloads, lintransf_ids)
        ]
    
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
            layer = f.require_group(layer_name)

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

        with h5py.File(self.diags_path, "r") as f:
            layer = f[layer_name]
            output_rotations = int(layer["output_rotations"][()])

            # Load the diagonals back into the correct struct
            all_diagonals = {}
            diag_group = layer["diagonals"]
            for block in diag_group:
                row, col = map(int, block.split("_")) # 0_1 -> (0,1)
                diags = {}
                block_group = diag_group[block]
                for diag_idx in block_group:
                    diag_data = block_group[diag_idx][:]
                    diags[int(diag_idx)] = diag_data 
                all_diagonals[(row, col)] = diags

        return all_diagonals, on_bias, output_rotations

    def load_transform_metadata(self, linear_layer):
        self._verify_layer_compatibility(
            linear_layer,
            check_output_rotations=False,
        )
        with h5py.File(self.diags_path, "r") as f:
            return int(f[linear_layer.name]["output_rotations"][()])

    def evaluate_transforms(self, linear_layer, in_ctensor):
        layer_name = linear_layer.name
        out_shape = linear_layer.output_shape
        fhe_out_shape = linear_layer.fhe_output_shape 
        skip_post_rescale = bool(getattr(self.backend, "lt_outputs_are_rescaled", False))

        # Order-preserving flatten that can be mapped back to 
        # (row, col) format in backend via len(in_ctensor.ids)
        transform_ids = np.array(list(linear_layer.transform_ids.values()))
        cols = len(in_ctensor)
        rows = len(transform_ids) // cols

        # Now we can perform a blocked linear transform
        transform_ids = transform_ids.reshape(rows, cols)
        self._transform_io_prefetcher.clear(wait=True)
        work_items = [
            (int(i), int(j), int(transform_ids[i][j]))
            for i in range(rows)
            for j in range(cols)
        ]
        work_index = 0
        cts_out = []
        for i in range(rows):
            ct_out = None
            for j in range(cols):
                t_id = transform_ids[i][j]

                if self.io_mode != "none":
                    current_key = self._transform_io_key(layer_name, int(i), int(j), int(t_id))
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
                    next_index = int(work_index + 1)
                    if next_index < len(work_items):
                        next_i, next_j, next_t_id = work_items[next_index]
                        self._submit_transform_io_prefetch(
                            layer_name,
                            int(next_i),
                            int(next_j),
                            int(next_t_id),
                        )

                res = self.backend.EvaluateLinearTransform(t_id, in_ctensor.ids[j]) 
                ct = CipherTensor(self.scheme, res, out_shape, fhe_out_shape)

                # Accumulate results across a row of blocks
                ct_out = ct if j == 0 else ct_out + ct
                    
                if self.io_mode != "none":
                    self.remove_rotation_keys()
                    self.remove_plaintext_diagonals(t_id)
                work_index += 1
            
            # We know the output of this accumulation will just be one ciphertext
            if skip_post_rescale:
                cts_out.append(int(ct_out.ids[0]))
                ct_out.ids = []
            else:
                ct_out_rescaled = self.evaluator.rescale(ct_out.ids[0], in_place=False)
                cts_out.append(ct_out_rescaled)

        self._transform_io_prefetcher.clear(wait=True)
        return CipherTensor(self.scheme, cts_out, out_shape, fhe_out_shape)
            
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
        with h5py.File(self.diags_path, "a") as f:
            layer = f[layer_name]
            plaintext_group = layer.require_group("plaintexts")
            block_idx = f"{row}_{col}"
            block_group = plaintext_group.create_group(block_idx)

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

    def _plaintext_payload_required(self) -> bool:
        return bool(getattr(self.backend, "load_plaintext_diagonals_requires_payload", True))

    def _transform_io_key(self, layer_name, row, col, transform_id):
        return (str(layer_name), int(row), int(col), int(transform_id))

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
                for diag_idx in block:
                    dataset = block[diag_idx]
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

    def _submit_transform_io_prefetch(self, layer_name, row, col, transform_id):
        if self.io_mode == "none":
            return
        key = self._transform_io_key(layer_name, row, col, transform_id)
        self._transform_io_prefetcher.submit(
            key,
            lambda: self._read_transform_io_bundle(
                layer_name,
                row,
                col,
                transform_id,
                prefetch=True,
            ),
        )

    def load_plaintext_diagonals(self, layer_name, row, col, transform_id, bundle=None):
        if bundle is not None:
            diag_indices = list(bundle.get("diag_indices", ()))
            if hasattr(self.backend, "LoadPlaintextDiagonalsBatch"):
                self.backend.LoadPlaintextDiagonalsBatch(
                    bundle.get("payload", np.zeros((0,), dtype=np.uint8)),
                    list(bundle.get("offsets", ())),
                    list(bundle.get("lengths", ())),
                    diag_indices,
                    int(transform_id),
                )
                return
            for diag_idx, serial_diag in bundle.get("plaintexts", ()):
                self.backend.LoadPlaintextDiagonal(
                    serial_diag,
                    transform_id,
                    int(diag_idx),
                )
            return
        with h5py.File(self.diags_path, "r") as f:
            layer = f[layer_name]
            ptxt_group = layer["plaintexts"]
            block = ptxt_group[f"{row}_{col}"]
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
    
    def load_rotation_keys(self, transform_id, bundle=None):
        if bundle is not None:
            for key_value, serial_key in bundle.get("rotation_keys", ()):
                self.backend.LoadRotationKey(serial_key, int(key_value))
            return
        keys = self.get_required_rotation_key_requests(transform_id)

        with h5py.File(self.keys_path, "r") as f:
            for key, level in keys:
                key_str = self._rotation_key_storage_name(key, level)
                if key_str not in f and str(int(key)) in f:
                    key_str = str(int(key))
                serial_key = f[key_str][()]
                self.backend.LoadRotationKey(serial_key, int(key))

    def remove_rotation_keys(self):
        self.backend.RemoveRotationKeys() 

    def remove_plaintext_diagonals(self, transform_id):
        self.backend.RemovePlaintextDiagonals(transform_id)
