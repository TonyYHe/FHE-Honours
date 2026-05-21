import time
import math
import os
import contextlib
import io
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Union, Dict, Any

import h5py
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler

from orion.nn.module import Module
from orion.nn.linear import LinearTransform
from orion.backend.lattigo import bindings as lgo
from orion.backend.python import (
    PythonBackend,
    compile_cache,
    parameters, 
    key_generator,
    encoder, 
    encryptor,
    evaluator, 
    poly_evaluator, 
    lt_evaluator,
    bootstrapper
)

from .tracer import StatsTracker, OrionTracer 
from .fuser import Fuser
from .network_dag import NetworkDAG
from .auto_bootstrap import BootstrapSolver, BootstrapPlacer


class _PackWorkerParams:
    def __init__(self, *, slots: int, embedding_method: str) -> None:
        self._slots = int(slots)
        self._embedding_method = str(embedding_method)

    def get_slots(self) -> int:
        return int(self._slots)

    def get_embedding_method(self) -> str:
        return str(self._embedding_method)


class _PackWorkerScheme:
    def __init__(self, *, slots: int, embedding_method: str) -> None:
        self.params = _PackWorkerParams(slots=int(slots), embedding_method=str(embedding_method))


def _dense_pack_worker_count() -> int:
    raw_value = os.environ.get("ORION_DENSE_PACK_WORKERS", "")
    try:
        return max(1, int(raw_value)) if raw_value else 1
    except (TypeError, ValueError):
        return 1


def _parse_size_bytes(raw_value: str | None, default: int) -> int:
    if raw_value is None or str(raw_value).strip() == "":
        return int(default)
    value = str(raw_value).strip().lower()
    multipliers = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
    }
    for suffix, multiplier in sorted(multipliers.items(), key=lambda item: -len(item[0])):
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)].strip()) * int(multiplier))
    return int(float(value))


def _mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def _dense_pack_worker_budget(target_workers: int) -> tuple[int, int | None, int, int]:
    reserve_raw = os.environ.get("ORION_DENSE_PACK_MEMORY_RESERVE_BYTES")
    if reserve_raw is None:
        reserve_raw = os.environ.get("ORION_DENSE_PACK_MEMORY_RESERVE_GB", "80g")
        if str(reserve_raw).strip().replace(".", "", 1).isdigit():
            reserve_raw = f"{reserve_raw}g"
    worker_raw = os.environ.get("ORION_DENSE_PACK_WORKER_MEMORY_BYTES")
    if worker_raw is None:
        worker_raw = os.environ.get("ORION_DENSE_PACK_WORKER_MEMORY_GB", "24g")
        if str(worker_raw).strip().replace(".", "", 1).isdigit():
            worker_raw = f"{worker_raw}g"
    reserve = _parse_size_bytes(
        reserve_raw,
        80 * 1024**3,
    )
    per_worker = _parse_size_bytes(
        worker_raw,
        24 * 1024**3,
    )
    available = _mem_available_bytes()
    if available is None or per_worker <= 0:
        return int(target_workers), available, int(reserve), int(per_worker)
    budget_workers = max(1, int((int(available) - int(reserve)) // int(per_worker)))
    return max(1, min(int(target_workers), int(budget_workers))), int(available), int(reserve), int(per_worker)


def _dense_pack_start_method() -> str:
    return os.environ.get("ORION_DENSE_PACK_START_METHOD", "spawn").strip() or "spawn"


def _linear_transform_pack_kind(module: Any) -> str | None:
    class_name = type(module).__name__
    if class_name == "Conv2d":
        return "conv2d"
    if class_name == "ConvTranspose2d":
        return "conv_transpose2d"
    if class_name == "Linear":
        return "linear"
    return None


def _linear_transform_uses_dense_pack(module: Any) -> bool:
    if bool(getattr(module, "region_first_probe_dense_bypass", False)):
        return False
    runtime = getattr(module, "region_runtime", None)
    runtime_supported = bool(
        runtime is not None and getattr(runtime, "supports_scheme", lambda _scheme: True)(module.scheme)
    )
    return not (
        runtime is not None
        and bool(getattr(runtime, "executable", False))
        and bool(getattr(module, "region_first_skip_dense_pack", False))
        and bool(runtime_supported)
    )


def _shape_list(value: Any) -> list[int]:
    return [int(item) for item in tuple(value)]


def _parallel_pack_snapshot(node: str, module: Any, *, last: bool, temp_dir: Path) -> dict[str, Any]:
    fields = {
        "node": str(node),
        "layer_name": str(module.name),
        "kind": _linear_transform_pack_kind(module),
        "last": bool(last),
        "temp_path": str(Path(temp_dir) / f"{len(str(node))}_{abs(hash(str(node))) & 0xffffffff:x}.h5"),
        "slots": int(module.scheme.params.get_slots()),
        "embedding_method": str(module.scheme.params.get_embedding_method()),
        "on_weight": module.on_weight.detach().cpu(),
        "on_bias": module.on_bias.detach().cpu().numpy(),
        "input_shape": _shape_list(module.input_shape),
        "output_shape": _shape_list(module.output_shape),
        "fhe_input_shape": _shape_list(module.fhe_input_shape),
        "fhe_output_shape": _shape_list(module.fhe_output_shape),
        "input_min": float(module.input_min.item() if hasattr(module.input_min, "item") else module.input_min),
        "input_max": float(module.input_max.item() if hasattr(module.input_max, "item") else module.input_max),
        "output_min": float(module.output_min.item() if hasattr(module.output_min, "item") else module.output_min),
        "output_max": float(module.output_max.item() if hasattr(module.output_max, "item") else module.output_max),
        "input_gap": int(getattr(module, "input_gap", 1)),
        "output_gap": int(getattr(module, "output_gap", 1)),
        "bsgs_ratio": float(getattr(module, "bsgs_ratio", 0.0)),
    }
    for attr in (
        "groups",
        "in_channels",
        "out_channels",
        "in_features",
        "out_features",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "output_padding",
    ):
        if hasattr(module, attr):
            fields[attr] = getattr(module, attr)
    return fields


def _module_from_pack_snapshot(snapshot: dict[str, Any]) -> Any:
    layer = SimpleNamespace()
    layer.scheme = _PackWorkerScheme(
        slots=int(snapshot["slots"]),
        embedding_method=str(snapshot["embedding_method"]),
    )
    layer.on_weight = snapshot["on_weight"]
    layer.on_bias = torch.tensor(snapshot["on_bias"], dtype=torch.float32)
    layer.input_shape = torch.Size(snapshot["input_shape"])
    layer.output_shape = torch.Size(snapshot["output_shape"])
    layer.fhe_input_shape = torch.Size(snapshot["fhe_input_shape"])
    layer.fhe_output_shape = torch.Size(snapshot["fhe_output_shape"])
    layer.input_gap = int(snapshot.get("input_gap", 1))
    layer.output_gap = int(snapshot.get("output_gap", 1))
    for attr in (
        "groups",
        "in_channels",
        "out_channels",
        "in_features",
        "out_features",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "output_padding",
    ):
        if attr in snapshot:
            setattr(layer, attr, snapshot[attr])
    return layer


def _write_packed_layer_h5(snapshot: dict[str, Any], diagonals: dict, output_rotations: int) -> None:
    path = Path(snapshot["temp_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with h5py.File(path, "w") as handle:
        layer = handle.create_group(str(snapshot["layer_name"]))
        layer.create_dataset("embedding_method", data=str(snapshot["embedding_method"]))
        layer.create_dataset("output_rotations", data=int(output_rotations))
        layer.create_dataset("on_bias", data=snapshot["on_bias"])
        layer.create_dataset("input_shape", data=list(snapshot["input_shape"]))
        layer.create_dataset("output_shape", data=list(snapshot["output_shape"]))
        layer.create_dataset("input_min", data=float(snapshot["input_min"]))
        layer.create_dataset("input_max", data=float(snapshot["input_max"]))
        layer.create_dataset("output_min", data=float(snapshot["output_min"]))
        layer.create_dataset("output_max", data=float(snapshot["output_max"]))
        diags_group = layer.create_group("diagonals")
        for row, col in sorted((int(row), int(col)) for row, col in diagonals.keys()):
            block_group = diags_group.create_group(f"{int(row)}_{int(col)}")
            for diag_idx in sorted(int(idx) for idx in diagonals[(row, col)].keys()):
                block_group.create_dataset(str(int(diag_idx)), data=diagonals[(row, col)][int(diag_idx)])


def _dense_pack_worker(snapshot: dict[str, Any]) -> dict[str, Any]:
    from orion.core import packing

    layer = _module_from_pack_snapshot(snapshot)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        if snapshot["kind"] == "conv2d":
            diagonals, output_rotations = packing.pack_conv2d(layer, bool(snapshot["last"]))
        elif snapshot["kind"] == "conv_transpose2d":
            diagonals, output_rotations = packing.pack_conv_transpose2d(layer, bool(snapshot["last"]))
        elif snapshot["kind"] == "linear":
            diagonals, output_rotations = packing.pack_linear(layer, bool(snapshot["last"]))
        else:
            raise ValueError(f"Unsupported dense pack kind: {snapshot['kind']}")
    _write_packed_layer_h5(snapshot, diagonals, int(output_rotations))
    return {
        "node": str(snapshot["node"]),
        "layer_name": str(snapshot["layer_name"]),
        "temp_path": str(snapshot["temp_path"]),
        "output_rotations": int(output_rotations),
        "log": output.getvalue(),
    }


def _commit_packed_layer(diags_path: str, result: dict[str, Any]) -> None:
    print(result.get("log", ""), end="", flush=True)
    print("└── saving... ", end="", flush=True)
    temp_path = Path(result["temp_path"])
    with h5py.File(diags_path, "a") as dst, h5py.File(temp_path, "r") as src:
        layer_name = str(result["layer_name"])
        if layer_name in dst:
            del dst[layer_name]
        src.copy(src[layer_name], dst, name=layer_name)
    temp_path.unlink(missing_ok=True)
    print("done!", flush=True)


def _region_first_mode_options(mode: str) -> dict[str, Any]:
    normalized = str(mode or "").lower()
    r18_allowed_stages = None
    r18_probe_stage_allowed_stages = None
    r18_stage_prefix = "r18_tiny_e2e_stage"
    if normalized.startswith(r18_stage_prefix):
        suffix = normalized[len(r18_stage_prefix) :]
        allowed = []
        for char in suffix:
            if char in "1234":
                allowed.append(f"stage{char}")
        if allowed:
            r18_allowed_stages = tuple(dict.fromkeys(allowed))
    r18_probe_stage_prefix = "r18_tiny_e2e_probe_precompile_stage"
    if normalized.startswith(r18_probe_stage_prefix):
        suffix = normalized[len(r18_probe_stage_prefix) :]
        allowed = []
        for char in suffix:
            if char == "0":
                allowed.append("stage1")
            elif char in "1234":
                allowed.append(f"stage{char}")
        if allowed:
            r18_probe_stage_allowed_stages = tuple(dict.fromkeys(allowed))
            r18_allowed_stages = r18_probe_stage_allowed_stages
    r18_modes = {
        "r18_tiny",
        "r18_tiny_e2e",
        "r18_tiny_e2e_probe",
        "r18_tiny_e2e_probe_precompile",
        "r18_tiny_e2e_probe_precompile_no_stem_bypass",
        "r18_tiny_e2e_probe_precompile_stage0",
    }
    u22_base_modes = {
        "u22_phase1",
        "u22_64_base8",
        "u22_64_base32",
        "u22_256_base8",
        "u22_256_base32",
    }
    u22_mode_base = next(
        (
            str(base)
            for base in sorted(u22_base_modes, key=len, reverse=True)
            if normalized == str(base) or normalized.startswith(f"{base}_")
        ),
        None,
    )
    u22_allowed_nodes = None
    u22_conv_kernels = bool(u22_mode_base is not None)
    u22_layout_policy = "dp" if u22_mode_base is not None else ""
    if u22_mode_base is not None and normalized != str(u22_mode_base):
        suffix = normalized[len(str(u22_mode_base)) + 1 :]
        allowed: list[str] = []
        tokens = suffix.split("_")
        index = 0
        while index < len(tokens):
            token = tokens[int(index)]
            if token in {"nohybrid", "nori", "noir"}:
                index += 1
                continue
            if (
                token in {"hybrid", "ir", "ri"}
                and int(index + 1) < len(tokens)
                and str(tokens[int(index + 1)]) in {"0", "false", "off", "no"}
            ):
                index += 2
                continue
            if token == "conv":
                u22_conv_kernels = True
                index += 1
                continue
            if token == "noconv":
                u22_conv_kernels = False
                index += 1
                continue
            if token == "layout" and int(index + 1) < len(tokens):
                raw_policy = str(tokens[int(index + 1)])
                if raw_policy == "fixed" and int(index + 2) < len(tokens) and str(tokens[int(index + 2)]) == "max":
                    raw_policy = "fixedmax"
                    index += 1
                policy_aliases = {
                    "fixed": "fixed_max",
                    "fixedmax": "fixed_max",
                    "eager": "eager",
                    "greedy": "greedy",
                    "dp": "dp",
                }
                if raw_policy in policy_aliases:
                    u22_layout_policy = str(policy_aliases[raw_policy])
                    index += 2
                    continue
            if token in {"fixedmax", "eager", "greedy", "dp"} and int(index) > 0 and tokens[int(index - 1)] == "layout":
                index += 1
                continue
            if not token.startswith("up"):
                index += 1
                continue
            digits = token[2:]
            if digits and all(ch in "1234" for ch in digits):
                allowed.extend(f"up{ch}" for ch in digits)
            index += 1
        if allowed:
            u22_allowed_nodes = tuple(dict.fromkeys(allowed))
    if u22_mode_base in {"u22_64_base8", "u22_64_base32", "u22_256_base8", "u22_256_base32"} and u22_allowed_nodes is None:
        u22_allowed_nodes = ("up1", "up2", "up3", "up4")
    elif u22_mode_base is not None and u22_allowed_nodes is None:
        u22_allowed_nodes = ("up4", "up3")
    is_enabled = normalized in r18_modes or r18_allowed_stages is not None or normalized == "r34_imgnet_phase1" or u22_mode_base is not None
    is_r18 = normalized in r18_modes or r18_allowed_stages is not None
    is_probe_dense_bypass = r18_probe_stage_allowed_stages is not None or normalized in {
        "r18_tiny_e2e_probe",
        "r18_tiny_e2e_probe_precompile",
        "r18_tiny_e2e_probe_precompile_no_stem_bypass",
        "r18_tiny_e2e_probe_precompile_stage0",
    }
    is_probe_stem_bypass = r18_probe_stage_allowed_stages is not None or normalized in {
        "r18_tiny_e2e_probe",
        "r18_tiny_e2e_probe_precompile",
        "r18_tiny_e2e_probe_precompile_stage0",
    }
    is_precompiled_probe = r18_probe_stage_allowed_stages is not None or normalized in {
        "r18_tiny_e2e_probe_precompile",
        "r18_tiny_e2e_probe_precompile_no_stem_bypass",
        "r18_tiny_e2e_probe_precompile_stage0",
    }
    return {
        "mode": normalized,
        "enabled": bool(is_enabled),
        "is_r18": bool(is_r18),
        "is_r34_phase1": bool(normalized == "r34_imgnet_phase1"),
        "is_u22_phase1": bool(u22_mode_base is not None),
        "u22_allowed_nodes": u22_allowed_nodes,
        "u22_conv_kernels": bool(u22_conv_kernels),
        "u22_layout_policy": str(u22_layout_policy),
        "allowed_stages": (
            r18_allowed_stages
            if r18_allowed_stages is not None
            else (("stage1",) if normalized == "r18_tiny_e2e_probe_precompile_stage0" else None)
        ),
        "attach_probe_dense_bypass": bool(is_probe_dense_bypass),
        "attach_probe_stem_activation_bypass": bool(is_probe_stem_bypass),
        "lazy_region_compile": bool(normalized == "r18_tiny_e2e_probe"),
        "probe_region_precompiled": bool(is_precompiled_probe),
        "probe_publishable": False,
    }


class Scheme:
    """
    This Scheme class drives most of the functionality in Orion. It 
    configures and manages how our framework interfaces with FHE backends, 
    and exposes this functionality to the user through attributes such as 
    the encoder, evaluators (linear transform, polynomial, etc.) and 
    bootstrappers. 

    It also serves two important purposes required before running FHE 
    inference: fitting the network and then compiling it. The fit() method 
    runs cleartext forward passes through the network to determine per-layer 
    input ranges, which are then used to fit polynomial approximations to 
    common activation functions (e.g., SiLU, ReLU). 

    The compile() function is responsible for all packing of data and 
    determines a level management policy by running our automatic bootstrap 
    placement algorithm. Once done, each Orion module is automatically 
    assigned a level that can then be used in its compilation. This primarily 
    includes generating the plaintexts needed for each linear transform. 
    """
    
    def __init__(self):
        self.backend = None
        self.traced = None

    def init_scheme(self, config: Union[str, Dict[str, Any]]):
        """Initializes the scheme."""
        if isinstance(config, str):
            try:
                with open(config, "r") as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                raise ValueError(f"Configuration file '{config}' not found.")
        elif not isinstance(config, dict):
            raise TypeError("Config must be a file path (str) or a dictionary.")
        
        self.params = parameters.NewParameters(config)
        self.backend = self.setup_backend(self.params)
        
        self.keygen = key_generator.NewKeyGenerator(self)
        self.encoder = encoder.NewEncoder(self)
        self.encryptor = encryptor.NewEncryptor(self)
        self.evaluator = evaluator.NewEvaluator(self)
        self.poly_evaluator = poly_evaluator.NewEvaluator(self)
        self.lt_evaluator = lt_evaluator.NewEvaluator(self)
        self.bootstrapper = bootstrapper.NewEvaluator(self)

        return self
    
    def delete_scheme(self):
        if self.backend:
            self.backend.DeleteScheme()
            self.backend = None
    
    def __del__(self):
        self.delete_scheme()
    
    def __str__(self):
        return str(self.params)
        
    def setup_backend(self, params):
        backend = params.get_backend()
        if backend == "lattigo":
            py_lattigo = lgo.LattigoLibrary()
            py_lattigo.setup_bindings(params)
            return py_lattigo
        elif backend == "cheddar":
            from orion.backend.cheddar import bindings as cgo

            py_cheddar = cgo.CheddarLibrary()
            py_cheddar.setup_bindings(params)
            return py_cheddar
        elif backend == "python":
            return PythonBackend(params)
        elif backend in ("heaan", "openfhe"):
            raise ValueError(f"Backend {backend} not yet supported.")
        else:
            raise ValueError(
                f"Invalid {backend}. Supported backends are: lattigo, cheddar, python."
            )

    def encode(self, tensor, level=None, scale=None):
        self._check_initialization()
        return self.encoder.encode(tensor, level, scale)

    def decode(self, ptxt):
        self._check_initialization() 
        return self.encoder.decode(ptxt)

    def encrypt(self, ptxt):
        self._check_initialization() 
        return self.encryptor.encrypt(ptxt)

    def decrypt(self, ctxt):
        self._check_initialization()
        return self.encryptor.decrypt(ctxt)
    
    def fit(self, net, input_data, batch_size=128):
        self._check_initialization()

        net.set_scheme(self)
        net.set_margin(self.params.get_margin())
        
        tracer = OrionTracer()
        traced = tracer.trace_model(net)
        self.traced = traced 

        stats_tracker = StatsTracker(traced)

        #-----------------------------------------#
        #   Populate layers with useful metadata  #
        #-----------------------------------------# 

        # Send input_data to the same device as the model.
        param = next(iter(net.parameters()), None)
        device = param.device if param is not None else torch.device("cpu")

        print("\n{1} Finding per-layer input/output ranges and shapes...", 
              flush=True)
        start = time.time()
        if isinstance(input_data, DataLoader):
            # Users often specify small batch sizes for FHE operations.
            # However, fitting statistics with large datasets would take 
            # unnecessarily long with small batches. To speed this up, we'll 
            # temporarily increase the batch size during the statistics-fitting 
            # step, and then restore the original batch size afterward.
            user_batch_size = input_data.batch_size
            if batch_size > user_batch_size:
                dataset = input_data.dataset
                shuffle = input_data.sampler is None or isinstance(input_data.sampler, RandomSampler)
                
                input_data = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=input_data.num_workers,
                    pin_memory=input_data.pin_memory,
                    drop_last=input_data.drop_last
                )

            # Use this (potentially new) dataloader
            for batch in tqdm(input_data, desc="Processing input data",
                    unit="batch", leave=True):
                stats_tracker.propagate(batch[0].to(device))

            # Now we'll reset the batch size back to what the user specified.
            stats_tracker.update_batch_size(user_batch_size)

        elif isinstance(input_data, torch.Tensor):
            stats_tracker.propagate(input_data.to(device)) 
        else:
            raise ValueError(
                "Input data must be a torch.Tensor or DataLoader, but "
                f"received {type(input_data)}."
            )

        #-------------------------------------#
        #      Fit polynomial activations     #
        #-------------------------------------#

        # Now we can use the statistics we just obtained above to fit
        # all polynomial activation functions.
        print("\n{2} Fitting polynomials... ", end="", flush=True)
        start = time.time()
        for module in net.modules():
            if hasattr(module, "fit") and callable(module.fit):
                module.fit()
        print(f"done! [{time.time()-start:.3f} secs.]")

    def _generate_matrix_diagonals_sequential(self, network_dag, topo_sort, last_linear) -> None:
        for node in topo_sort:
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform):
                print(f"\nPacking {node}:")
                module.generate_diagonals(last=(node == last_linear))

    def _generate_matrix_diagonals_parallel_save(self, network_dag, topo_sort, last_linear) -> bool:
        target_workers = _dense_pack_worker_count()
        workers, mem_available, mem_reserve, worker_memory = _dense_pack_worker_budget(int(target_workers))
        if self.params.get_io_mode() != "save" or workers <= 1:
            return False

        dense_nodes: list[str] = []
        for node in topo_sort:
            module = network_dag.nodes[node]["module"]
            if not isinstance(module, LinearTransform):
                continue
            if (
                _linear_transform_pack_kind(module) is not None
                and _linear_transform_uses_dense_pack(module)
                and not module.load_cached_transform_metadata()
            ):
                dense_nodes.append(str(node))

        if len(dense_nodes) <= 1:
            return False

        temp_dir = Path(self.lt_evaluator.diags_path).parent / ".dense_pack_tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        start_method = _dense_pack_start_method()
        mem_note = (
            "unknown"
            if mem_available is None
            else f"available={mem_available / 1024**3:.1f}GiB, reserve={mem_reserve / 1024**3:.1f}GiB, per_worker={worker_memory / 1024**3:.1f}GiB"
        )
        print(
            f"├── Parallel dense packing enabled: target_workers={target_workers}, "
            f"effective_workers={workers}, layers={len(dense_nodes)}, "
            f"start_method={start_method}, memory_budget=({mem_note})",
            flush=True,
        )

        snapshots: list[tuple[str, dict[str, Any]]] = []
        dense_node_set = set(dense_nodes)
        for node in topo_sort:
            module = network_dag.nodes[node]["module"]
            if str(node) in dense_node_set:
                snapshots.append(
                    (
                        str(node),
                        _parallel_pack_snapshot(
                            str(node),
                            module,
                            last=(node == last_linear),
                            temp_dir=temp_dir,
                        ),
                    )
                )

        futures: dict[str, Any] = {}
        snapshot_by_node = {node: snapshot for node, snapshot in snapshots}
        snapshot_nodes = [node for node, _snapshot in snapshots]
        submit_index = 0
        max_workers = min(int(workers), len(snapshot_nodes))
        context = mp.get_context(start_method)

        def submit_next(executor) -> None:
            nonlocal submit_index
            if submit_index >= len(snapshot_nodes):
                return
            submit_node = snapshot_nodes[submit_index]
            futures[submit_node] = executor.submit(_dense_pack_worker, snapshot_by_node[submit_node])
            submit_index += 1

        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
                for _ in range(max_workers):
                    submit_next(executor)

                for node in topo_sort:
                    module = network_dag.nodes[node]["module"]
                    if not isinstance(module, LinearTransform):
                        continue
                    if str(node) not in futures:
                        print(f"\nPacking {node}:")
                        module.generate_diagonals(last=(node == last_linear))
                        continue

                    print(f"\nPacking {node}:")
                    result = futures.pop(str(node)).result()
                    module.output_rotations = int(result["output_rotations"])
                    module.diagonals = {}
                    _commit_packed_layer(self.lt_evaluator.diags_path, result)
                    submit_next(executor)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return True

    def compile(self, net):
        self._check_initialization()

        if self.traced is None:
            raise ValueError(
                "Network has not been fit yet! Before running orion.compile(net) "
                "you must run orion.fit(net, input_data)."
            )
                
        #------------------------------------------------#
        #   Build DAG representation of neural network   #
        #------------------------------------------------#

        network_dag = NetworkDAG(self.traced)
        network_dag.build_dag()

        # Before fusing, we'll instantiate our own Orion parameters (e.g. 
        # weights and biases) that can be fused/modified without affecting 
        # the original network's parameters. 
        for module in net.modules():
            if (hasattr(module, "init_orion_params") and 
                    callable(module.init_orion_params)):
                module.init_orion_params()

        #-------------------------------------#
        #       Resolve pooling kernels       #
        #-------------------------------------# 
        
        # AvgPools are implemented as grouped convolutions in Orion, which
        # are not passed arguments for the number of channels for consistency
        # with PyTorch. We must resolve this after the passes above use 
        # torch.nn.functional.
        for module in net.modules():
            if hasattr(module, "update_params") and callable(module.update_params):
                module.update_params()

        #------------------------------------------#
        #   Fuse Orion modules (Conv -> BN, etc)   #
        #------------------------------------------#

        enable_fusing = self.params.get_fuse_modules()
        if enable_fusing:
            fuser = Fuser(network_dag)
            fuser.fuse_modules()
            network_dag.remove_fused_batchnorms()

        self.region_first_registry = None
        self.region_first_attach_audit = {}
        experimental_region_first = self.params.get_experimental_region_first()
        region_first_options = _region_first_mode_options(experimental_region_first)
        if bool(region_first_options["enabled"]):
            if bool(region_first_options["is_r34_phase1"]):
                from orion.experimental.r34_phase1 import R34CompileRegistry

                self.region_first_registry = R34CompileRegistry.for_r34_imgnet_phase1(network_dag)
                self.region_first_attach_audit = self.region_first_registry.attach_to_dag(network_dag)
            elif bool(region_first_options["is_u22_phase1"]):
                from orion.experimental.u22_phase1 import U22CompileRegistry

                self.region_first_registry = U22CompileRegistry.for_dag(
                    network_dag,
                    allowed_nodes=region_first_options.get("u22_allowed_nodes"),
                    enable_conv_kernels=bool(region_first_options.get("u22_conv_kernels", False)),
                    layout_policy=str(region_first_options.get("u22_layout_policy", "dp")),
                )
                self.region_first_attach_audit = self.region_first_registry.attach_to_dag(network_dag)
            else:
                from orion.experimental.cir.runtime_group import RegionFirstCompileRegistry

                allowed_stages = region_first_options["allowed_stages"]
                use_r18_e2e_registry = experimental_region_first in {
                    "r18_tiny_e2e",
                    "r18_tiny_e2e_probe",
                    "r18_tiny_e2e_probe_precompile",
                    "r18_tiny_e2e_probe_precompile_no_stem_bypass",
                    "r18_tiny_e2e_probe_precompile_stage0",
                } or allowed_stages is not None
                if use_r18_e2e_registry:
                    self.region_first_registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(
                        network_dag,
                        allowed_stages=allowed_stages,
                    )
                else:
                    self.region_first_registry = RegionFirstCompileRegistry.for_r18_tiny(network_dag)
                self.region_first_attach_audit = self.region_first_registry.attach_to_dag(network_dag)
                if bool(region_first_options["attach_probe_dense_bypass"]):
                    self.region_first_attach_audit["probe_dense_bypass"] = self.region_first_registry.attach_probe_dense_bypass_to_dag(
                        network_dag,
                        lazy_region_compile=bool(region_first_options["lazy_region_compile"]),
                    )
                if bool(region_first_options["attach_probe_stem_activation_bypass"]):
                    self.region_first_attach_audit["probe_stem_activation_bypass"] = self.region_first_registry.attach_probe_stem_activation_bypass(net)
                if bool(region_first_options["attach_probe_dense_bypass"] or region_first_options["attach_probe_stem_activation_bypass"]):
                    self.region_first_attach_audit["probe_publishable"] = bool(region_first_options["probe_publishable"])
                    self.region_first_attach_audit["probe_region_precompiled"] = bool(region_first_options["probe_region_precompiled"])

        #---------------------------------------------#
        #   Pack diagonals of all linear transforms   #
        #---------------------------------------------#

        # Then, we must ensure that there is no junk data left in the slots
        # of the final linear layer (leaking information about partials).
        # This would occur when using the hybrid embedding method. We could
        # use an additional level to zero things out, but instead, we'll
        # just force the last linear layer to use the "square" embedding 
        # method which solves this while consuming just one level (albeit 
        # usually for more ciphertext rotations).
        topo_sort = list(network_dag.topological_sort())
        io_mode = self.params.get_io_mode()
        compile_manifest_path = compile_cache.manifest_path(self.params)
        compile_manifest = None
        compile_identity_fingerprint = compile_cache.cache_fingerprint(self.params, net)
        if io_mode == "load":
            compile_manifest = compile_cache.read_manifest(compile_manifest_path)
            compile_cache.validate_manifest_identity(compile_manifest, params=self.params, net=net)
        set_compile_manifest = getattr(self.lt_evaluator, "set_compile_manifest", None)
        if callable(set_compile_manifest):
            set_compile_manifest(compile_manifest)

        last_linear = None
        for node in reversed(topo_sort):
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform):
                last_linear = node
                break

        if io_mode == "load":
            print("\n{3} Restoring cached matrix descriptors...", flush=True)
            for node in topo_sort:
                module = network_dag.nodes[node]["module"]
                if isinstance(module, LinearTransform):
                    module.load_cached_transform_metadata()
        else:
            # Now we can generate the diagonals.
            print("\n{3} Generating matrix diagonals...", flush=True)
            if not self._generate_matrix_diagonals_parallel_save(network_dag, topo_sort, last_linear):
                self._generate_matrix_diagonals_sequential(network_dag, topo_sort, last_linear)

        #------------------------------#
        #   Find and place bootstraps  # 
        #------------------------------#

        network_dag.find_residuals()
        plan_topo_sort = list(network_dag.topological_sort())
        #(save_path="network.png", figsize=(8,30)) # optional plot

        if io_mode == "load":
            compile_cache.validate_topology(compile_manifest, plan_topo_sort)
            print("\n{4} Loading cached bootstrap placement... ", end="", flush=True)
            start = time.time()
            input_level, num_bootstraps, bootstrapper_slots = compile_cache.apply_bootstrap_plan(
                network_dag,
                compile_manifest["bootstrap_plan"],
            )
            print(f"done! [{time.time()-start:.3f} secs.]", flush=True)
        else:
            print("\n{4} Running bootstrap placement... ", end="", flush=True)
            start = time.time()
            l_eff = len(self.params.get_logq()) - 1
            btp_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
            input_level, num_bootstraps, bootstrapper_slots = btp_solver.solve()
            print(f"done! [{time.time()-start:.3f} secs.]", flush=True)
        print(f"├── Network requires {num_bootstraps} bootstrap "
            f"{'operation' if num_bootstraps == 1 else 'operations'}.")

        if io_mode == "save":
            compile_cache.write_manifest(
                compile_manifest_path,
                compile_cache.build_manifest(
                    params=self.params,
                    net=net,
                    network_dag=network_dag,
                    topo_sort=plan_topo_sort,
                    input_level=int(input_level),
                    bootstrap_count=int(num_bootstraps),
                    bootstrapper_slots=list(bootstrapper_slots),
                    linear_layers=[],
                    fingerprint=compile_identity_fingerprint,
                ),
            )

        #btp_solver.plot_shortest_path(
        #    save_path="network-with-levels.png", figsize=(8,30) # optional plot
        #)

        skip_bootstrapper_generation = os.environ.get(
            "ORION_COMPILE_SKIP_BOOTSTRAPPER_GENERATION",
            "0",
        ).lower() not in ("0", "false", "no", "off")
        if bootstrapper_slots and not skip_bootstrapper_generation:
            start = time.time()
            slots_str = ", ".join([str(int(math.log2(slot))) for slot in bootstrapper_slots])
            print(f"├── Generating bootstrappers for logslots = {slots_str} ... ", 
                  end="", flush=True)
            
            # Generate the required (potentially sparse) bootstrappers.
            for slot_count in bootstrapper_slots:
                self.bootstrapper.generate_bootstrapper(slot_count)
            print(f"done! [{time.time()-start:.3f} secs.]")
        elif bootstrapper_slots:
            slots_str = ", ".join([str(int(math.log2(slot))) for slot in bootstrapper_slots])
            print(
                f"├── Skipping bootstrapper generation for logslots = {slots_str} "
                "(ORION_COMPILE_SKIP_BOOTSTRAPPER_GENERATION=1).",
                flush=True,
            )

        btp_placer = BootstrapPlacer(net, network_dag)
        btp_placer.place_bootstraps()
        if io_mode == "load" and compile_manifest is not None:
            compile_cache.apply_provider_metadata(
                network_dag,
                compile_manifest.get("provider_metadata", {}),
            )

        #------------------------------------------#
        #   Compile Orion modules in the network   #
        #------------------------------------------#

        print("\n{5} Compiling network layers...", flush=True)
        compiled_linear_layers = []
        for node in topo_sort:
            node_attrs = network_dag.nodes[node]
            module = node_attrs["module"]
            if isinstance(module, Module):
                print(f"├── {node} @ level={module.level}", flush=True)
                module.compile()
                if isinstance(module, LinearTransform):
                    compiled_linear_layers.append(module)

        register_saved_io_schedule = getattr(self.lt_evaluator, "register_saved_io_schedule", None)
        if callable(register_saved_io_schedule):
            register_saved_io_schedule(compiled_linear_layers)

        if io_mode == "load" and compile_manifest is not None:
            compile_cache.validate_transform_metadata(compile_manifest, compiled_linear_layers)
        elif io_mode == "save":
            compile_cache.write_manifest(
                compile_manifest_path,
                compile_cache.build_manifest(
                    params=self.params,
                    net=net,
                    network_dag=network_dag,
                    topo_sort=plan_topo_sort,
                    input_level=int(input_level),
                    bootstrap_count=int(num_bootstraps),
                    bootstrapper_slots=list(bootstrapper_slots),
                    linear_layers=compiled_linear_layers,
                    fingerprint=compile_identity_fingerprint,
                ),
            )
                
        return input_level # level at which to encrypt the input.

    def _check_initialization(self):
        if self.backend is None:
            raise ValueError(
                "Scheme not initialized. Call `orion.init_scheme()` first.") 
        
scheme = Scheme()
