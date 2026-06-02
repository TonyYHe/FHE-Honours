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
from .auto_bootstrap import collect_bootstrap_solver_audit
from .auto_bootstrap import reset_bootstrap_solver_assignments
from .auto_bootstrap import snapshot_bootstrap_solver_assignments
from .bootstrap_layout_compression import (
    apply_bootstrap_aware_layout_refinement_candidate,
    apply_bootstrap_layout_compression,
    bootstrap_aware_layout_refinement_applicable,
    enumerate_final_boot_boundary_halo0_cleanup_candidate,
    enumerate_bootstrap_aware_layout_refinement_candidates,
    restore_layout_policy_compile_plan,
)
from .bootstrap_trial_evaluator import ExactBootstrapTrialEvaluator


class _PackWorkerParams:
    def __init__(self, *, slots: int, embedding_method: str) -> None:
        self._slots = int(slots)
        self._embedding_method = str(embedding_method)

    def get_slots(self) -> int:
        return int(self._slots)

    def get_embedding_method(self) -> str:
        return str(self._embedding_method)

    def get_io_mode(self) -> str:
        return "none"


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


def _bootstrap_shape_nonincrease_safe(
    candidate: dict[str, Any],
    *,
    bootstrap_ct_delta: int,
    bootstrap_slots_delta: int,
    force: bool = False,
) -> bool:
    if not bool(force) and not bool(candidate.get("require_bootstrap_shape_nonincrease", False)):
        return True
    return bool(int(bootstrap_ct_delta) <= 0 and int(bootstrap_slots_delta) <= 0)


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
    layout_policy_generic_modes = {
        "generic_layout_dp",
        "generic_layout_dp_no_share_fold",
        "generic_layout_dp_noshare_fold",
        "vgg_imgnet_layout_dp",
        "r18_imgnet_layout_dp",
    }
    is_layout_policy_generic = normalized in layout_policy_generic_modes
    u22_allowed_nodes = None
    u22_conv_kernels = bool(u22_mode_base is not None or is_layout_policy_generic)
    u22_layout_policy = "dp" if bool(u22_mode_base is not None or is_layout_policy_generic) else ""
    if normalized in {"generic_layout_dp_no_share_fold", "generic_layout_dp_noshare_fold"}:
        u22_layout_policy = "dp_no_share_fold"
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
                consumed = 2
                remaining = "_".join(str(value) for value in tokens[int(index + 1) :])
                for multi_token_policy in (
                    "dp_no_share_fold",
                    "dp_noshare_fold",
                    "fixed_max_no_share",
                    "fixedmax_no_share",
                    "fixed_max_noshare",
                    "fixedmax_noshare",
                    "fixed_max_no_share_fused",
                    "fixedmax_no_share_fused",
                    "fixed_max_noshare_fused",
                    "fixedmax_noshare_fused",
                    "fixed_max_no_share_unfused",
                    "fixedmax_no_share_unfused",
                    "fixed_max_noshare_unfused",
                    "fixedmax_noshare_unfused",
                    "always_no_share",
                    "always_noshare",
                    "always_relayout_no_share",
                    "always_relayout_noshare",
                    "always_no_share_fused",
                    "always_noshare_fused",
                    "always_relayout_no_share_fused",
                    "always_relayout_noshare_fused",
                    "always_no_share_unfused",
                    "always_noshare_unfused",
                    "always_relayout_no_share_unfused",
                    "always_relayout_noshare_unfused",
                    "no_share_fold",
                    "noshare_fold",
                ):
                    if remaining == multi_token_policy or remaining.startswith(f"{multi_token_policy}_"):
                        raw_policy = str(multi_token_policy)
                        consumed = 1 + len(multi_token_policy.split("_"))
                        break
                if raw_policy == "fixed" and int(index + 2) < len(tokens) and str(tokens[int(index + 2)]) == "max":
                    raw_policy = "fixedmax"
                    consumed += 1
                fused_index = int(index + consumed)
                if fused_index < len(tokens) and str(tokens[fused_index]) == "fused":
                    raw_policy = f"{raw_policy}_fused"
                    consumed += 1
                elif fused_index < len(tokens) and str(tokens[fused_index]) == "unfused":
                    raw_policy = f"{raw_policy}_unfused"
                    consumed += 1
                policy_aliases = {
                    "fixed": "fixed_max",
                    "fixedmax": "fixed_max",
                    "fixed_fused": "fixed_max_fused",
                    "fixedmax_fused": "fixed_max_fused",
                    "fixed_max_no_share": "fixed_max_no_share_fused",
                    "fixedmax_no_share": "fixed_max_no_share_fused",
                    "fixed_max_noshare": "fixed_max_no_share_fused",
                    "fixedmax_noshare": "fixed_max_no_share_fused",
                    "fixed_max_no_share_fused": "fixed_max_no_share_fused",
                    "fixedmax_no_share_fused": "fixed_max_no_share_fused",
                    "fixed_max_noshare_fused": "fixed_max_no_share_fused",
                    "fixedmax_noshare_fused": "fixed_max_no_share_fused",
                    "fixed_max_no_share_unfused": "fixed_max_no_share_unfused",
                    "fixedmax_no_share_unfused": "fixed_max_no_share_unfused",
                    "fixed_max_noshare_unfused": "fixed_max_no_share_unfused",
                    "fixedmax_noshare_unfused": "fixed_max_no_share_unfused",
                    "eager": "eager",
                    "eager_fused": "eager_fused",
                    "greedy": "greedy",
                    "greedy_fused": "greedy_fused",
                    "always": "always",
                    "always_fused": "always_fused",
                    "always_no_share": "always_no_share_fused",
                    "always_noshare": "always_no_share_fused",
                    "always_relayout_no_share": "always_no_share_fused",
                    "always_relayout_noshare": "always_no_share_fused",
                    "always_no_share_fused": "always_no_share_fused",
                    "always_noshare_fused": "always_no_share_fused",
                    "always_relayout_no_share_fused": "always_no_share_fused",
                    "always_relayout_noshare_fused": "always_no_share_fused",
                    "always_no_share_unfused": "always_no_share_unfused",
                    "always_noshare_unfused": "always_no_share_unfused",
                    "always_relayout_no_share_unfused": "always_no_share_unfused",
                    "always_relayout_noshare_unfused": "always_no_share_unfused",
                    "orion": "orion_dense",
                    "dense": "orion_dense",
                    "oriondense": "orion_dense",
                    "orion_dense": "orion_dense",
                    "nohalo": "orion_dense",
                    "no_halo": "orion_dense",
                    "dp": "dp",
                    "dp_no_share_fold": "dp_no_share_fold",
                    "dp_noshare_fold": "dp_no_share_fold",
                    "no_share_fold": "dp_no_share_fold",
                    "noshare_fold": "dp_no_share_fold",
                    "nosharefold": "dp_no_share_fold",
                }
                if raw_policy in policy_aliases:
                    u22_layout_policy = str(policy_aliases[raw_policy])
                    index += int(consumed)
                    continue
            if token in {"fixedmax", "noshare", "no", "share", "relayout", "eager", "greedy", "always", "dp", "fused"} and int(index) > 0 and tokens[int(index - 1)] == "layout":
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
    is_enabled = (
        normalized in r18_modes
        or r18_allowed_stages is not None
        or normalized == "r34_imgnet_phase1"
        or u22_mode_base is not None
        or bool(is_layout_policy_generic)
    )
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
        "is_layout_policy_generic": bool(is_layout_policy_generic),
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
        elif backend == "clear_lattigo":
            py_clear_lattigo = lgo.ClearLattigoLibrary()
            py_clear_lattigo.setup_bindings(params)
            return py_clear_lattigo
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
                f"Invalid {backend}. Supported backends are: lattigo, clear_lattigo, cheddar, python."
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
            elif bool(region_first_options["is_u22_phase1"]) or bool(region_first_options["is_layout_policy_generic"]):
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
            network_dag.bootstrap_layout_compression_audit = apply_bootstrap_layout_compression(
                network_dag
            )
            print(f"done! [{time.time()-start:.3f} secs.]", flush=True)
        else:
            print("\n{4} Running bootstrap placement... ", end="", flush=True)
            start = time.time()
            l_eff = len(self.params.get_logq()) - 1
            if bootstrap_aware_layout_refinement_applicable(network_dag):
                level_snapshot = snapshot_bootstrap_solver_assignments(network_dag)
                btp_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                input_level, num_bootstraps, bootstrapper_slots = btp_solver._solve_once(
                    apply_layout_compression=False
                )
                initial_audit = collect_bootstrap_solver_audit(network_dag, l_eff=l_eff)
                current_audit = dict(initial_audit)
                refinement_rounds: list[dict[str, Any]] = []
                max_refinement_rounds = max(
                    1,
                    int(os.environ.get("ORION_BOOTSTRAP_LAYOUT_REFINEMENT_MAX_ROUNDS", "16") or 16),
                )
                target_bootstrap_count: int | None = None
                target_bootstrap_source = ""
                target_bootstrap_audit: dict[str, Any] | None = None
                target_bootstrap_error = ""
                target_raw = str(os.environ.get("ORION_BOOTSTRAP_LAYOUT_REFINEMENT_TARGET_BOOTSTRAPS", "") or "").strip()
                if target_raw:
                    try:
                        target_bootstrap_count = max(0, int(target_raw))
                        target_bootstrap_source = "env"
                    except (TypeError, ValueError):
                        target_bootstrap_error = f"invalid_env_target:{target_raw}"
                auto_target_enabled = (
                    target_bootstrap_count is None
                    and str(os.environ.get("ORION_BOOTSTRAP_LAYOUT_REFINEMENT_AUTO_TARGET", "1") or "1").lower()
                    not in {"0", "false", "no", "off"}
                )
                trial_evaluator_mode = str(
                    os.environ.get("ORION_BOOTSTRAP_LAYOUT_REFINEMENT_TRIAL_EVALUATOR", "legacy") or "legacy"
                ).strip().lower()
                if trial_evaluator_mode in {"", "0", "false", "no", "off"}:
                    trial_evaluator_mode = "legacy"
                trial_evaluator_enabled = trial_evaluator_mode in {
                    "exact_cached_verify_all",
                    "exact_cached_selected_verify",
                }
                trial_evaluator_fallback_reason = ""
                trial_evaluator_fallback_round: int | None = None
                trial_evaluator = (
                    ExactBootstrapTrialEvaluator(network_dag, l_eff=l_eff)
                    if trial_evaluator_enabled
                    else None
                )

                def _restore_depth_snapshot(depth_snapshot: list[dict[str, Any]]) -> None:
                    for depth_row in depth_snapshot:
                        runtime = depth_row.get("runtime")
                        module = depth_row.get("module")
                        if runtime is not None:
                            if depth_row.get("runtime_depth") is not None:
                                runtime.depth = int(depth_row["runtime_depth"])
                            if depth_row.get("runtime_solver_depth") is not None:
                                runtime.solver_depth = int(depth_row["runtime_solver_depth"])
                        if module is not None and depth_row.get("module_depth") is not None:
                            if hasattr(module, "set_depth"):
                                module.set_depth(int(depth_row["module_depth"]))
                            else:
                                module.depth = int(depth_row["module_depth"])

                def _estimate_relayout_free_bootstrap_target(
                    depth_snapshot: list[dict[str, Any]],
                ) -> tuple[int, dict[str, Any]]:
                    assignment_snapshot = snapshot_bootstrap_solver_assignments(network_dag)
                    try:
                        for depth_row in depth_snapshot:
                            runtime = depth_row.get("runtime")
                            module = depth_row.get("module")
                            raw_depth = depth_row.get("runtime_solver_depth")
                            if raw_depth is None:
                                raw_depth = depth_row.get("runtime_depth")
                            if raw_depth is None:
                                raw_depth = depth_row.get("module_depth")
                            if raw_depth is None:
                                continue
                            base_depth = max(
                                0,
                                int(raw_depth) - int(depth_row.get("relayout_depth", 0) or 0),
                            )
                            if runtime is not None:
                                runtime.depth = int(base_depth)
                                runtime.solver_depth = int(base_depth)
                            if module is not None:
                                if hasattr(module, "set_depth"):
                                    module.set_depth(int(base_depth))
                                else:
                                    module.depth = int(base_depth)
                        reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                        target_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                        _target_input_level, target_bootstraps, _target_slots = target_solver._solve_once(
                            apply_layout_compression=False
                        )
                        target_audit = collect_bootstrap_solver_audit(network_dag, l_eff=l_eff)
                        return int(target_audit.get("bootstrap_count", target_bootstraps) or 0), dict(target_audit)
                    finally:
                        _restore_depth_snapshot(depth_snapshot)
                        reset_bootstrap_solver_assignments(network_dag, assignment_snapshot)

                def _bootstrap_ct_slots(audit: dict[str, Any]) -> tuple[int, int]:
                    return (
                        int(
                            sum(
                                int(row.get("bootstrap_ct_count", 0) or 0)
                                for row in audit.get("boot_edges", [])
                            )
                        ),
                        int(
                            sum(
                                int(row.get("bootstrap_ct_count", 0) or 0)
                                * int(row.get("bootstrapper_slots", 0) or 0)
                                for row in audit.get("boot_edges", [])
                            )
                        ),
                    )

                def _cleanup_edge_ids(candidate: dict[str, Any]) -> tuple[str, ...]:
                    return tuple(
                        sorted(
                            str(row.get("edge", ""))
                            for row in candidate.get("accepted", [])
                            if str(row.get("edge", ""))
                        )
                    )

                def _strip_cleanup_private(candidate: dict[str, Any]) -> dict[str, Any]:
                    return {
                        key: value
                        for key, value in dict(candidate).items()
                        if key not in {"plan", "previous_compile_plan", "previous_depths", "_previous_module_layouts"}
                    }

                def _run_final_halo0_cleanup(
                    base_audit: dict[str, Any],
                ) -> tuple[int, int, list[int], dict[str, Any], dict[str, Any]]:
                    base_boot_count = int(base_audit.get("bootstrap_count", num_bootstraps) or 0)
                    base_bootstrap_ct_count, base_bootstrap_slots_total = _bootstrap_ct_slots(base_audit)
                    cleanup_rounds: list[dict[str, Any]] = []
                    working_audit = dict(base_audit)
                    base_compile_plan: dict[str, Any] | None = None
                    base_depths: list[dict[str, Any]] | None = None
                    max_cleanup_rounds = max(
                        1,
                        int(os.environ.get("ORION_BOOTSTRAP_LAYOUT_FINAL_HALO0_CLEANUP_MAX_ROUNDS", "8") or 8),
                    )

                    for cleanup_round_index in range(int(max_cleanup_rounds)):
                        cleanup_candidate = enumerate_final_boot_boundary_halo0_cleanup_candidate(
                            network_dag,
                            working_audit,
                        )
                        if base_compile_plan is None and isinstance(
                            cleanup_candidate.get("previous_compile_plan"), dict
                        ):
                            base_compile_plan = dict(cleanup_candidate["previous_compile_plan"])
                            base_depths = list(cleanup_candidate.get("previous_depths", []))
                        if not bool(cleanup_candidate.get("enabled", False)):
                            cleanup_rounds.append(
                                {
                                    **_strip_cleanup_private(cleanup_candidate),
                                    "round": int(cleanup_round_index + 1),
                                    "input_bootstrap_count": int(working_audit.get("bootstrap_count", 0) or 0),
                                }
                            )
                            if cleanup_round_index == 0:
                                return int(input_level), int(num_bootstraps), [int(v) for v in bootstrapper_slots], dict(base_audit), {
                                    "enabled": False,
                                    "reason": str(
                                        cleanup_candidate.get(
                                            "reason",
                                            "no_final_boot_boundary_halo0_cleanup_candidates",
                                        )
                                    ),
                                    "rounds": cleanup_rounds,
                                }
                            break

                        if base_compile_plan is None:
                            base_compile_plan = dict(cleanup_candidate.get("previous_compile_plan", {}) or {})
                            base_depths = list(cleanup_candidate.get("previous_depths", []))
                        if not base_compile_plan:
                            cleanup_rounds.append(
                                {
                                    **_strip_cleanup_private(cleanup_candidate),
                                    "round": int(cleanup_round_index + 1),
                                    "enabled": False,
                                    "rolled_back": True,
                                    "rollback_reason": "missing_base_compile_plan",
                                }
                            )
                            break

                        cleanup_edges = _cleanup_edge_ids(cleanup_candidate)
                        cleanup_module_layout_snapshot = None
                        try:
                            cleanup_apply_audit = apply_bootstrap_aware_layout_refinement_candidate(
                                network_dag,
                                dict(cleanup_candidate),
                                first_pass_audit=working_audit,
                            )
                            cleanup_module_layout_snapshot = cleanup_apply_audit.get("_previous_module_layouts")
                            cleanup_apply_public = _strip_cleanup_private(cleanup_apply_audit)
                            reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                            cleanup_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                            _cleanup_input_level, cleanup_bootstraps, _cleanup_slots = cleanup_solver._solve_once(
                                apply_layout_compression=False
                            )
                            cleanup_final_audit = collect_bootstrap_solver_audit(network_dag, l_eff=l_eff)
                        except Exception as exc:
                            restore_layout_policy_compile_plan(
                                network_dag,
                                base_compile_plan,
                                depth_snapshot=base_depths or [],
                                module_layout_snapshot=cleanup_module_layout_snapshot,
                            )
                            reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                            rollback_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                            rollback_input_level, rollback_bootstraps, rollback_slots = rollback_solver._solve_once(
                                apply_layout_compression=False
                            )
                            cleanup_rounds.append(
                                {
                                    **_strip_cleanup_private(cleanup_candidate),
                                    "round": int(cleanup_round_index + 1),
                                    "enabled": False,
                                    "rolled_back": True,
                                    "rollback_reason": "final_halo0_cleanup_trial_error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            return (
                                int(rollback_input_level),
                                int(rollback_bootstraps),
                                [int(v) for v in rollback_slots],
                                dict(base_audit),
                                {
                                    "enabled": False,
                                    "reason": "final_halo0_cleanup_trial_error",
                                    "rolled_back": True,
                                    "rounds": cleanup_rounds,
                                },
                            )
                        cleanup_boot_count = int(
                            cleanup_final_audit.get("bootstrap_count", cleanup_bootstraps) or 0
                        )
                        cleanup_bootstrap_ct_count, cleanup_bootstrap_slots_total = _bootstrap_ct_slots(
                            cleanup_final_audit
                        )
                        boot_delta = int(cleanup_boot_count - base_boot_count)
                        bootstrap_ct_delta = int(cleanup_bootstrap_ct_count - base_bootstrap_ct_count)
                        bootstrap_slots_delta = int(cleanup_bootstrap_slots_total - base_bootstrap_slots_total)
                        relayout_depth_before = int(
                            cleanup_apply_audit.get("true_relayout_kernel_depth_before", 0) or 0
                        )
                        relayout_depth_after = int(
                            cleanup_apply_audit.get("true_relayout_kernel_depth_after", relayout_depth_before) or 0
                        )
                        relayout_depth_delta = int(relayout_depth_after - relayout_depth_before)
                        bootstrap_shape_safe = _bootstrap_shape_nonincrease_safe(
                            cleanup_candidate,
                            bootstrap_ct_delta=int(bootstrap_ct_delta),
                            bootstrap_slots_delta=int(bootstrap_slots_delta),
                            force=True,
                        )
                        bootstrap_width_reduced = bool(int(bootstrap_slots_delta) < 0)
                        safe = bool(
                            int(boot_delta) <= 0
                            and bool(bootstrap_shape_safe)
                            and bool(bootstrap_width_reduced)
                            and int(relayout_depth_delta) <= 0
                        )

                        restore_layout_policy_compile_plan(
                            network_dag,
                            base_compile_plan,
                            depth_snapshot=base_depths or [],
                            module_layout_snapshot=cleanup_module_layout_snapshot,
                        )
                        reset_bootstrap_solver_assignments(network_dag, level_snapshot)

                        if not bool(safe):
                            rollback_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                            rollback_input_level, rollback_bootstraps, rollback_slots = rollback_solver._solve_once(
                                apply_layout_compression=False
                            )
                            cleanup_rounds.append(
                                {
                                    **_strip_cleanup_private(cleanup_candidate),
                                    **dict(cleanup_apply_public),
                                    "round": int(cleanup_round_index + 1),
                                    "enabled": False,
                                    "rolled_back": True,
                                    "rollback_reason": (
                                        "final_halo0_bootstrap_shape_increased"
                                        if bool(bootstrap_shape_safe) is False
                                        else "final_halo0_no_boot_width_reduction"
                                        if not bool(bootstrap_width_reduced)
                                        else "final_halo0_bootstrap_count_or_depth_increased"
                                    ),
                                    "boot_delta": int(boot_delta),
                                    "bootstrap_ct_delta": int(bootstrap_ct_delta),
                                    "bootstrap_slots_delta": int(bootstrap_slots_delta),
                                    "bootstrap_width_reduced": bool(bootstrap_width_reduced),
                                    "relayout_depth_delta": int(relayout_depth_delta),
                                    "trial_boot_edges": list(cleanup_final_audit.get("boot_edges", [])),
                                }
                            )
                            return (
                                int(rollback_input_level),
                                int(rollback_bootstraps),
                                [int(v) for v in rollback_slots],
                                dict(base_audit),
                                {
                                    "enabled": False,
                                    "reason": str(cleanup_rounds[-1]["rollback_reason"]),
                                    "rolled_back": True,
                                    "rounds": cleanup_rounds,
                                },
                            )

                        next_candidate = enumerate_final_boot_boundary_halo0_cleanup_candidate(
                            network_dag,
                            cleanup_final_audit,
                        )
                        next_edges = (
                            _cleanup_edge_ids(next_candidate)
                            if bool(next_candidate.get("enabled", False))
                            else tuple()
                        )
                        stable_edges = bool(tuple(cleanup_edges) == tuple(next_edges))
                        round_row = {
                            **_strip_cleanup_private(cleanup_candidate),
                            **dict(cleanup_apply_public),
                            "round": int(cleanup_round_index + 1),
                            "committed": bool(stable_edges),
                            "rolled_back": not bool(stable_edges),
                            "rollback_reason": ""
                            if bool(stable_edges)
                            else "revalidate_with_moved_boot_boundary",
                            "input_bootstrap_count": int(working_audit.get("bootstrap_count", 0) or 0),
                            "final_bootstrap_count": int(cleanup_boot_count),
                            "boot_delta": int(boot_delta),
                            "bootstrap_ct_delta": int(bootstrap_ct_delta),
                            "bootstrap_slots_delta": int(bootstrap_slots_delta),
                            "bootstrap_width_reduced": bool(bootstrap_width_reduced),
                            "relayout_depth_delta": int(relayout_depth_delta),
                            "cleanup_edges": list(cleanup_edges),
                            "next_cleanup_edges": list(next_edges),
                            "stable_final_boot_boundary": bool(stable_edges),
                            "trial_boot_edges": list(cleanup_final_audit.get("boot_edges", [])),
                        }
                        cleanup_rounds.append(dict(round_row))

                        if bool(stable_edges):
                            final_candidate = cleanup_candidate
                            final_module_layout_snapshot = None
                            try:
                                final_apply_audit = apply_bootstrap_aware_layout_refinement_candidate(
                                    network_dag,
                                    dict(final_candidate),
                                    first_pass_audit=working_audit,
                                )
                                final_module_layout_snapshot = final_apply_audit.get("_previous_module_layouts")
                                final_apply_public = _strip_cleanup_private(final_apply_audit)
                                reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                                final_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                                final_input_level, final_bootstraps, final_slots = final_solver._solve_once(
                                    apply_layout_compression=False
                                )
                                final_audit = collect_bootstrap_solver_audit(network_dag, l_eff=l_eff)
                            except Exception as exc:
                                restore_layout_policy_compile_plan(
                                    network_dag,
                                    base_compile_plan,
                                    depth_snapshot=base_depths or [],
                                    module_layout_snapshot=final_module_layout_snapshot,
                                )
                                reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                                rollback_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                                rollback_input_level, rollback_bootstraps, rollback_slots = rollback_solver._solve_once(
                                    apply_layout_compression=False
                                )
                                cleanup_rounds[-1] = {
                                    **dict(cleanup_rounds[-1]),
                                    "enabled": False,
                                    "committed": False,
                                    "rolled_back": True,
                                    "rollback_reason": "final_halo0_cleanup_commit_error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                                return (
                                    int(rollback_input_level),
                                    int(rollback_bootstraps),
                                    [int(v) for v in rollback_slots],
                                    dict(base_audit),
                                    {
                                        "enabled": False,
                                        "reason": "final_halo0_cleanup_commit_error",
                                        "rolled_back": True,
                                        "rounds": cleanup_rounds,
                                    },
                                )
                            final_boot_count = int(final_audit.get("bootstrap_count", final_bootstraps) or 0)
                            final_bootstrap_ct_count, final_bootstrap_slots_total = _bootstrap_ct_slots(final_audit)
                            final_boot_delta = int(final_boot_count - base_boot_count)
                            final_ct_delta = int(final_bootstrap_ct_count - base_bootstrap_ct_count)
                            final_slots_delta = int(final_bootstrap_slots_total - base_bootstrap_slots_total)
                            final_depth_before = int(
                                final_apply_audit.get("true_relayout_kernel_depth_before", 0) or 0
                            )
                            final_depth_after = int(
                                final_apply_audit.get("true_relayout_kernel_depth_after", final_depth_before) or 0
                            )
                            final_depth_delta = int(final_depth_after - final_depth_before)
                            final_shape_safe = _bootstrap_shape_nonincrease_safe(
                                final_candidate,
                                bootstrap_ct_delta=int(final_ct_delta),
                                bootstrap_slots_delta=int(final_slots_delta),
                                force=True,
                            )
                            final_width_reduced = bool(int(final_slots_delta) < 0)
                            final_safe = bool(
                                int(final_boot_delta) <= 0
                                and bool(final_shape_safe)
                                and bool(final_width_reduced)
                                and int(final_depth_delta) <= 0
                            )
                            if not bool(final_safe):
                                restore_layout_policy_compile_plan(
                                    network_dag,
                                    base_compile_plan,
                                    depth_snapshot=base_depths or [],
                                    module_layout_snapshot=final_module_layout_snapshot,
                                )
                                reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                                rollback_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                                rollback_input_level, rollback_bootstraps, rollback_slots = rollback_solver._solve_once(
                                    apply_layout_compression=False
                                )
                                reason = (
                                    "final_halo0_bootstrap_shape_increased"
                                    if not bool(final_shape_safe)
                                    else "final_halo0_no_boot_width_reduction"
                                    if not bool(final_width_reduced)
                                    else "final_halo0_bootstrap_count_or_depth_increased"
                                )
                                cleanup_rounds[-1] = {
                                    **dict(cleanup_rounds[-1]),
                                    "enabled": False,
                                    "committed": False,
                                    "rolled_back": True,
                                    "rollback_reason": str(reason),
                                    "final_boot_delta": int(final_boot_delta),
                                    "final_bootstrap_ct_delta": int(final_ct_delta),
                                    "final_bootstrap_slots_delta": int(final_slots_delta),
                                    "final_bootstrap_width_reduced": bool(final_width_reduced),
                                    "final_relayout_depth_delta": int(final_depth_delta),
                                }
                                return (
                                    int(rollback_input_level),
                                    int(rollback_bootstraps),
                                    [int(v) for v in rollback_slots],
                                    dict(base_audit),
                                    {
                                        "enabled": False,
                                        "reason": str(reason),
                                        "rolled_back": True,
                                        "rounds": cleanup_rounds,
                                    },
                                )
                            return (
                                int(final_input_level),
                                int(final_bootstraps),
                                [int(v) for v in final_slots],
                                dict(final_audit),
                                {
                                    **_strip_cleanup_private(final_candidate),
                                    **dict(final_apply_public),
                                    "enabled": True,
                                    "fixed_point": True,
                                    "rolled_back": False,
                                    "round_count": int(len(cleanup_rounds)),
                                    "rounds": cleanup_rounds,
                                    "final_bootstrap_count": int(final_boot_count),
                                    "final_boot_delta": int(final_boot_delta),
                                    "final_bootstrap_ct_delta": int(final_ct_delta),
                                    "final_bootstrap_slots_delta": int(final_slots_delta),
                                    "final_bootstrap_width_reduced": bool(final_width_reduced),
                                    "final_relayout_depth_delta": int(final_depth_delta),
                                    "final_boot_edges": list(final_audit.get("boot_edges", [])),
                                },
                            )

                        working_audit = dict(cleanup_final_audit)

                    if base_compile_plan:
                        restore_layout_policy_compile_plan(
                            network_dag,
                            base_compile_plan,
                            depth_snapshot=base_depths or [],
                        )
                    reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                    rollback_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                    rollback_input_level, rollback_bootstraps, rollback_slots = rollback_solver._solve_once(
                        apply_layout_compression=False
                    )
                    return (
                        int(rollback_input_level),
                        int(rollback_bootstraps),
                        [int(v) for v in rollback_slots],
                        dict(base_audit),
                        {
                            "enabled": False,
                            "reason": "final_halo0_cleanup_fixed_point_not_reached",
                            "rolled_back": True,
                            "rounds": cleanup_rounds,
                        },
                    )

                rollback_audit: dict[str, Any] | None = None
                stopped_reason = "max_rounds_reached"
                refinement_policy = "dp_no_share_fold"
                for round_index in range(int(max_refinement_rounds)):
                    candidate_audit = enumerate_bootstrap_aware_layout_refinement_candidates(
                        network_dag,
                        current_audit,
                    )
                    if str(candidate_audit.get("policy", "")):
                        refinement_policy = str(candidate_audit.get("policy", ""))
                    round_audit = {
                        **dict(candidate_audit),
                        "round": int(round_index + 1),
                        "input_bootstrap_count": int(current_audit.get("bootstrap_count", 0) or 0),
                    }
                    if not bool(candidate_audit.get("enabled", False)):
                        refinement_rounds.append(round_audit)
                        stopped_reason = str(candidate_audit.get("reason", "no_candidates"))
                        break

                    candidates = [dict(row) for row in candidate_audit.get("candidates", [])]
                    accepted_compile_plan = dict(candidate_audit["previous_compile_plan"])
                    accepted_depths = list(candidate_audit.get("previous_depths", []))
                    current_boot_count = int(current_audit.get("bootstrap_count", 0) or 0)
                    current_relayout_depth = int(
                        sum(int(row.get("relayout_depth", 0) or 0) for row in accepted_depths)
                    )
                    if target_bootstrap_count is None and auto_target_enabled:
                        try:
                            target_bootstrap_count, target_bootstrap_audit = _estimate_relayout_free_bootstrap_target(
                                accepted_depths
                            )
                            target_bootstrap_source = "relayout_free_depth"
                        except Exception as exc:
                            target_bootstrap_error = str(exc)
                            auto_target_enabled = False
                    if target_bootstrap_count is not None:
                        round_audit = {
                            **dict(round_audit),
                            "target_bootstrap_count": int(target_bootstrap_count),
                            "target_bootstrap_source": str(target_bootstrap_source),
                        }
                        post_target_cleanup_round = False
                        if int(current_boot_count) <= int(target_bootstrap_count):
                            post_target_candidates = [
                                dict(candidate)
                                for candidate in candidates
                                if bool(candidate.get("allow_after_bootstrap_target", False))
                                and int(candidate.get("output_tile_delta", 0) or 0) <= 0
                                and int(candidate.get("rotation_delta", 0) or 0) <= 0
                            ]
                            if not post_target_candidates:
                                round_audit = {
                                    **dict(round_audit),
                                    "enabled": False,
                                    "stopped_at_bootstrap_target": True,
                                    "trials": [],
                                }
                                refinement_rounds.append(round_audit)
                                stopped_reason = "bootstrap_target_reached"
                                break
                            candidates = post_target_candidates
                            post_target_cleanup_round = True
                            round_audit = {
                                **dict(round_audit),
                                "bootstrap_target_reached_cleanup": True,
                                "post_target_candidate_count": int(len(candidates)),
                            }
                    else:
                        post_target_cleanup_round = False
                    current_bootstrap_ct_count = int(
                        sum(
                            int(row.get("bootstrap_ct_count", 0) or 0)
                            for row in current_audit.get("boot_edges", [])
                        )
                    )
                    current_bootstrap_slots_total = int(
                        sum(
                            int(row.get("bootstrapper_slots", 0) or 0)
                            for row in current_audit.get("boot_edges", [])
                        )
                    )
                    trials: list[dict[str, Any]] = []
                    best_trial: dict[str, Any] | None = None
                    best_score: tuple[int, int, int, int, int, int, int, int, str] | None = None
                    round_trial_evaluator_audit: dict[str, Any] = {
                        "mode": str(trial_evaluator_mode),
                        "enabled": bool(trial_evaluator_enabled),
                        "rank_verification": "legacy",
                        "candidate_count": int(len(candidates)),
                    }

                    def _trial_from_result(
                        *,
                        candidate: dict[str, Any],
                        trial_apply_audit: dict[str, Any],
                        trial_input_level: int,
                        trial_bootstraps: int,
                        trial_slots: list[int],
                        trial_final_audit: dict[str, Any],
                        trial_number: int,
                    ) -> tuple[dict[str, Any], tuple[int, int, int, int, int, int, int, int, str] | None, dict[str, Any] | None]:
                        trial_relayout_depth = int(
                            trial_apply_audit.get(
                                "true_relayout_kernel_depth_after",
                                current_relayout_depth,
                            )
                            or 0
                        )
                        trial_boot_count = int(trial_final_audit.get("bootstrap_count", trial_bootstraps) or 0)
                        boot_delta = int(trial_boot_count - current_boot_count)
                        depth_delta = int(trial_relayout_depth - current_relayout_depth)
                        bootstrap_ct_count = int(
                            sum(
                                int(row.get("bootstrap_ct_count", 0) or 0)
                                for row in trial_final_audit.get("boot_edges", [])
                            )
                        )
                        bootstrap_ct_delta = int(bootstrap_ct_count - current_bootstrap_ct_count)
                        bootstrap_slots_total = int(
                            sum(
                                int(row.get("bootstrapper_slots", 0) or 0)
                                for row in trial_final_audit.get("boot_edges", [])
                            )
                        )
                        bootstrap_slots_delta = int(bootstrap_slots_total - current_bootstrap_slots_total)
                        accepted_count_for_score = int(trial_apply_audit.get("accepted_count", 0) or 0)
                        bootstrap_shape_forced = bool(post_target_cleanup_round)
                        bootstrap_shape_safe = _bootstrap_shape_nonincrease_safe(
                            candidate,
                            bootstrap_ct_delta=int(bootstrap_ct_delta),
                            bootstrap_slots_delta=int(bootstrap_slots_delta),
                            force=bool(bootstrap_shape_forced),
                        )
                        bootstrap_count_safe = bool(
                            int(boot_delta) == 0
                            if bool(candidate.get("require_bootstrap_count_unchanged", False))
                            else int(boot_delta) <= 0
                        )
                        rotation_delta = int(candidate.get("rotation_delta", 0) or 0)
                        output_tile_delta = int(candidate.get("output_tile_delta", 0) or 0)
                        allow_depth_unchanged = bool(
                            candidate.get("allow_relayout_depth_unchanged", False)
                            and int(depth_delta) == 0
                            and int(rotation_delta) <= 0
                            and int(output_tile_delta) <= 0
                        )
                        relayout_depth_safe = bool(int(depth_delta) < 0 or bool(allow_depth_unchanged))
                        acceptable = bool(
                            bool(bootstrap_count_safe)
                            and bool(relayout_depth_safe)
                            and bool(bootstrap_shape_safe)
                        )
                        trial_audit = {
                            **dict(trial_apply_audit),
                            "accepted": bool(acceptable),
                            "trial_input_level": int(trial_input_level),
                            "trial_bootstrap_count": int(trial_boot_count),
                            "trial_bootstrapper_slots": [int(value) for value in trial_slots],
                            "boot_delta": int(boot_delta),
                            "relayout_depth_delta": int(depth_delta),
                            "bootstrap_ct_delta": int(bootstrap_ct_delta),
                            "bootstrap_slots_delta": int(bootstrap_slots_delta),
                            "bootstrap_count_safe": bool(bootstrap_count_safe),
                            "bootstrap_shape_safe": bool(bootstrap_shape_safe),
                            "bootstrap_shape_forced": bool(bootstrap_shape_forced),
                            "relayout_depth_safe": bool(relayout_depth_safe),
                            "allow_relayout_depth_unchanged": bool(
                                candidate.get("allow_relayout_depth_unchanged", False)
                            ),
                            "output_tile_delta": int(output_tile_delta),
                            "trial_boot_edges": list(trial_final_audit.get("boot_edges", [])),
                        }
                        if not bool(acceptable):
                            return trial_audit, None, None
                        score = (
                            int(boot_delta),
                            int(candidate.get("candidate_priority", 0) or 0),
                            int(rotation_delta),
                            int(bootstrap_ct_count),
                            int(bootstrap_slots_total),
                            int(depth_delta),
                            -int(accepted_count_for_score),
                            int(trial_number),
                            str(candidate.get("candidate_id", "")),
                        )
                        payload = {
                            "candidate": dict(candidate),
                            "trial_audit": dict(trial_audit),
                            "final_audit": dict(trial_final_audit),
                            "input_level": int(trial_input_level),
                            "bootstrap_count": int(trial_boot_count),
                            "bootstrapper_slots": [int(value) for value in trial_slots],
                            "score": tuple(score),
                        }
                        return trial_audit, score, payload

                    def _score_official_candidate(
                        candidate: dict[str, Any],
                        trial_number: int,
                    ) -> tuple[dict[str, Any], tuple[int, int, int, int, int, int, int, int, str] | None, dict[str, Any] | None]:
                        trial_module_layout_snapshot = None
                        try:
                            trial_apply_audit = apply_bootstrap_aware_layout_refinement_candidate(
                                network_dag,
                                candidate,
                                first_pass_audit=current_audit,
                            )
                            trial_module_layout_snapshot = trial_apply_audit.get("_previous_module_layouts")
                            trial_apply_audit = {
                                key: value
                                for key, value in dict(trial_apply_audit).items()
                                if key != "_previous_module_layouts"
                            }
                            reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                            trial_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                            trial_input_level, trial_bootstraps, trial_slots = trial_solver._solve_once(
                                apply_layout_compression=False
                            )
                            trial_final_audit = collect_bootstrap_solver_audit(network_dag, l_eff=l_eff)
                            return _trial_from_result(
                                candidate=candidate,
                                trial_apply_audit=trial_apply_audit,
                                trial_input_level=int(trial_input_level),
                                trial_bootstraps=int(trial_bootstraps),
                                trial_slots=[int(value) for value in trial_slots],
                                trial_final_audit=trial_final_audit,
                                trial_number=int(trial_number),
                            )
                        except Exception as exc:
                            return (
                                {
                                    "candidate_id": str(candidate.get("candidate_id", "")),
                                    "kind": str(candidate.get("kind", candidate.get("strategy", ""))),
                                    "accepted": False,
                                    "rejected": True,
                                    "reject_reason": "trial_solve_error",
                                    "error": str(exc),
                                },
                                None,
                                None,
                            )
                        finally:
                            restore_layout_policy_compile_plan(
                                network_dag,
                                accepted_compile_plan,
                                depth_snapshot=accepted_depths,
                                module_layout_snapshot=trial_module_layout_snapshot,
                            )
                            reset_bootstrap_solver_assignments(network_dag, level_snapshot)

                    def _score_evaluator_candidate(
                        candidate: dict[str, Any],
                        trial_number: int,
                    ) -> tuple[dict[str, Any], tuple[int, int, int, int, int, int, int, int, str] | None, dict[str, Any] | None]:
                        if trial_evaluator is None:
                            raise RuntimeError("bootstrap trial evaluator is not enabled")
                        trial_module_layout_snapshot = None
                        try:
                            trial_apply_audit = apply_bootstrap_aware_layout_refinement_candidate(
                                network_dag,
                                candidate,
                                first_pass_audit=current_audit,
                            )
                            trial_module_layout_snapshot = trial_apply_audit.get("_previous_module_layouts")
                            trial_apply_audit = {
                                key: value
                                for key, value in dict(trial_apply_audit).items()
                                if key != "_previous_module_layouts"
                            }
                            reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                            trial_result = trial_evaluator.evaluate()
                            return _trial_from_result(
                                candidate=candidate,
                                trial_apply_audit=trial_apply_audit,
                                trial_input_level=int(trial_result.input_level),
                                trial_bootstraps=int(trial_result.bootstrap_count),
                                trial_slots=[int(value) for value in trial_result.bootstrapper_slots],
                                trial_final_audit=dict(trial_result.audit),
                                trial_number=int(trial_number),
                            )
                        finally:
                            restore_layout_policy_compile_plan(
                                network_dag,
                                accepted_compile_plan,
                                depth_snapshot=accepted_depths,
                                module_layout_snapshot=trial_module_layout_snapshot,
                            )
                            reset_bootstrap_solver_assignments(network_dag, level_snapshot)

                    def _select_best(
                        scored_trials: list[
                            tuple[
                                dict[str, Any],
                                tuple[int, int, int, int, int, int, int, int, str] | None,
                                dict[str, Any] | None,
                            ]
                        ],
                    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, tuple[int, int, int, int, int, int, int, int, str] | None]:
                        selected_trials: list[dict[str, Any]] = []
                        selected_best_trial: dict[str, Any] | None = None
                        selected_best_score: tuple[int, int, int, int, int, int, int, int, str] | None = None
                        for trial_audit, score, payload in scored_trials:
                            selected_trials.append(dict(trial_audit))
                            if score is None or payload is None:
                                continue
                            if selected_best_score is None or tuple(score) < tuple(selected_best_score):
                                selected_best_score = tuple(score)
                                selected_best_trial = dict(payload)
                        return selected_trials, selected_best_trial, selected_best_score

                    def _trial_signature(
                        trial_audit: dict[str, Any],
                        score: tuple[int, int, int, int, int, int, int, int, str] | None,
                    ) -> tuple[Any, ...]:
                        boot_edges = [
                            (
                                str(row.get("source", "")),
                                str(row.get("target", "")),
                                int(row.get("source_level", 0) or 0),
                                int(row.get("target_level", 0) or 0),
                                int(row.get("source_depth", 0) or 0),
                                int(row.get("bootstrap_ct_count", 0) or 0),
                                int(row.get("bootstrapper_slots", 0) or 0),
                            )
                            for row in trial_audit.get("trial_boot_edges", [])
                        ]
                        return (
                            bool(trial_audit.get("accepted", False)),
                            None if score is None else tuple(score),
                            int(trial_audit.get("trial_bootstrap_count", 0) or 0),
                            tuple(int(value) for value in trial_audit.get("trial_bootstrapper_slots", []) or []),
                            int(trial_audit.get("boot_delta", 0) or 0),
                            int(trial_audit.get("relayout_depth_delta", 0) or 0),
                            int(trial_audit.get("bootstrap_ct_delta", 0) or 0),
                            int(trial_audit.get("bootstrap_slots_delta", 0) or 0),
                            tuple(boot_edges),
                        )

                    def _score_all_official() -> tuple[list[dict[str, Any]], dict[str, Any] | None, tuple[int, int, int, int, int, int, int, int, str] | None]:
                        scored = [
                            _score_official_candidate(dict(candidate), int(index + 1))
                            for index, candidate in enumerate(candidates)
                        ]
                        return _select_best(scored)

                    try:
                        if trial_evaluator_enabled and trial_evaluator_mode == "exact_cached_verify_all":
                            evaluator_started = time.perf_counter()
                            evaluator_scored = [
                                _score_evaluator_candidate(dict(candidate), int(index + 1))
                                for index, candidate in enumerate(candidates)
                            ]
                            evaluator_s = float(time.perf_counter() - evaluator_started)
                            official_started = time.perf_counter()
                            official_scored = [
                                _score_official_candidate(dict(candidate), int(index + 1))
                                for index, candidate in enumerate(candidates)
                            ]
                            official_s = float(time.perf_counter() - official_started)
                            evaluator_trials, evaluator_best, evaluator_score = _select_best(evaluator_scored)
                            official_trials, official_best, official_score = _select_best(official_scored)
                            mismatch_reason = ""
                            for index, (eval_row, official_row) in enumerate(zip(evaluator_scored, official_scored)):
                                if _trial_signature(eval_row[0], eval_row[1]) != _trial_signature(official_row[0], official_row[1]):
                                    mismatch_reason = f"candidate_{index + 1}_signature_mismatch"
                                    break
                            if not mismatch_reason and (
                                (evaluator_best or {}).get("candidate", {}).get("candidate_id")
                                != (official_best or {}).get("candidate", {}).get("candidate_id")
                                or evaluator_score != official_score
                            ):
                                mismatch_reason = "selected_candidate_mismatch"
                            round_trial_evaluator_audit = {
                                **dict(round_trial_evaluator_audit),
                                "rank_verification": "all_candidates",
                                "evaluator_timing_s": evaluator_s,
                                "official_verifier_timing_s": official_s,
                                "selected_candidate_parity": not bool(mismatch_reason),
                                "mismatch_reason": str(mismatch_reason),
                            }
                            if mismatch_reason:
                                trial_evaluator_enabled = False
                                trial_evaluator_fallback_reason = str(mismatch_reason)
                                trial_evaluator_fallback_round = int(round_index + 1)
                            trials, best_trial, best_score = official_trials, official_best, official_score
                        elif trial_evaluator_enabled and trial_evaluator_mode == "exact_cached_selected_verify":
                            evaluator_started = time.perf_counter()
                            evaluator_scored = [
                                _score_evaluator_candidate(dict(candidate), int(index + 1))
                                for index, candidate in enumerate(candidates)
                            ]
                            evaluator_s = float(time.perf_counter() - evaluator_started)
                            evaluator_trials, evaluator_best, evaluator_score = _select_best(evaluator_scored)
                            mismatch_reason = ""
                            official_s = 0.0
                            official_trials = None
                            official_best = None
                            official_score = None
                            if evaluator_best is not None:
                                selected_index = int(evaluator_score[7]) if evaluator_score is not None else 0
                                official_started = time.perf_counter()
                                official_selected = _score_official_candidate(
                                    dict(evaluator_best["candidate"]),
                                    int(selected_index),
                                )
                                official_s = float(time.perf_counter() - official_started)
                                official_trial, official_score, official_payload = official_selected
                                if official_score is None or official_payload is None:
                                    mismatch_reason = "selected_candidate_rejected_by_official"
                                elif tuple(evaluator_score or ()) != tuple(official_score):
                                    mismatch_reason = "selected_candidate_score_mismatch"
                            else:
                                official_started = time.perf_counter()
                                official_trials, official_best, official_score = _score_all_official()
                                official_s = float(time.perf_counter() - official_started)
                                if official_best is not None:
                                    mismatch_reason = "selected_candidate_missing_from_evaluator"
                            round_trial_evaluator_audit = {
                                **dict(round_trial_evaluator_audit),
                                "rank_verification": "selected_only",
                                "evaluator_timing_s": evaluator_s,
                                "official_verifier_timing_s": official_s,
                                "selected_candidate_parity": not bool(mismatch_reason),
                                "mismatch_reason": str(mismatch_reason),
                            }
                            if mismatch_reason:
                                trial_evaluator_enabled = False
                                trial_evaluator_fallback_reason = str(mismatch_reason)
                                trial_evaluator_fallback_round = int(round_index + 1)
                                if official_trials is None:
                                    official_trials, official_best, official_score = _score_all_official()
                                trials, best_trial, best_score = official_trials, official_best, official_score
                            else:
                                trials, best_trial, best_score = evaluator_trials, evaluator_best, evaluator_score
                        else:
                            trials, best_trial, best_score = _score_all_official()
                    except Exception as exc:
                        trial_evaluator_enabled = False
                        trial_evaluator_fallback_reason = f"{type(exc).__name__}: {exc}"
                        trial_evaluator_fallback_round = int(round_index + 1)
                        round_trial_evaluator_audit = {
                            **dict(round_trial_evaluator_audit),
                            "enabled": False,
                            "fallback_reason": str(trial_evaluator_fallback_reason),
                        }
                        trials, best_trial, best_score = _score_all_official()

                    if best_trial is None:
                        reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                        restore_layout_policy_compile_plan(
                            network_dag,
                            accepted_compile_plan,
                            depth_snapshot=accepted_depths,
                        )
                        stable_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                        input_level, num_bootstraps, bootstrapper_slots = stable_solver._solve_once(
                            apply_layout_compression=False
                        )
                        rollback_audit = {
                            **dict(round_audit),
                            "enabled": False,
                            "rolled_back": True,
                            "rollback_reason": "no_candidate_improved_boot_safe_relayout_depth",
                            "trials": trials,
                            "trial_evaluator": dict(round_trial_evaluator_audit),
                        }
                        refinement_rounds.append(dict(rollback_audit))
                        stopped_reason = "no_candidate_improved_boot_safe_relayout_depth"
                        break

                    refinement_audit = apply_bootstrap_aware_layout_refinement_candidate(
                        network_dag,
                        dict(best_trial["candidate"]),
                        first_pass_audit=current_audit,
                    )
                    refinement_module_layout_snapshot = refinement_audit.get("_previous_module_layouts")
                    refinement_audit = {
                        key: value
                        for key, value in dict(refinement_audit).items()
                        if key != "_previous_module_layouts"
                    }
                    reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                    second_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                    input_level, num_bootstraps, bootstrapper_slots = second_solver._solve_once(
                        apply_layout_compression=False
                    )
                    final_audit = collect_bootstrap_solver_audit(network_dag, l_eff=l_eff)
                    final_bootstrap_ct_count = int(
                        sum(
                            int(row.get("bootstrap_ct_count", 0) or 0)
                            for row in final_audit.get("boot_edges", [])
                        )
                    )
                    final_bootstrap_slots_total = int(
                        sum(
                            int(row.get("bootstrapper_slots", 0) or 0)
                            for row in final_audit.get("boot_edges", [])
                        )
                    )
                    final_bootstrap_ct_delta = int(final_bootstrap_ct_count - current_bootstrap_ct_count)
                    final_bootstrap_slots_delta = int(
                        final_bootstrap_slots_total - current_bootstrap_slots_total
                    )
                    round_audit = {
                        **round_audit,
                        **dict(refinement_audit),
                        "candidate_count": int(candidate_audit.get("candidate_count", len(candidates)) or len(candidates)),
                        "trials": trials,
                        "trial_evaluator": dict(round_trial_evaluator_audit),
                        "target_bootstrap_count": (
                            None if target_bootstrap_count is None else int(target_bootstrap_count)
                        ),
                        "target_bootstrap_source": str(target_bootstrap_source),
                        "final_bootstrap_count": int(final_audit["bootstrap_count"]),
                        "final_bootstrap_ct_delta": int(final_bootstrap_ct_delta),
                        "final_bootstrap_slots_delta": int(final_bootstrap_slots_delta),
                        "final_boot_edges": list(final_audit.get("boot_edges", [])),
                    }
                    if int(final_audit["bootstrap_count"]) > int(current_audit["bootstrap_count"]) or (
                        bool(post_target_cleanup_round)
                        and (int(final_bootstrap_ct_delta) > 0 or int(final_bootstrap_slots_delta) > 0)
                    ):
                        restore_layout_policy_compile_plan(
                            network_dag,
                            accepted_compile_plan,
                            depth_snapshot=accepted_depths,
                            module_layout_snapshot=refinement_module_layout_snapshot,
                        )
                        reset_bootstrap_solver_assignments(network_dag, level_snapshot)
                        rollback_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
                        input_level, num_bootstraps, bootstrapper_slots = rollback_solver._solve_once(
                            apply_layout_compression=False
                        )
                        rollback_audit = {
                            **dict(round_audit),
                            "enabled": False,
                            "rolled_back": True,
                            "rollback_reason": (
                                "post_target_bootstrap_shape_increased"
                                if bool(post_target_cleanup_round)
                                and (int(final_bootstrap_ct_delta) > 0 or int(final_bootstrap_slots_delta) > 0)
                                else "bootstrap_count_increased"
                            ),
                        }
                        refinement_rounds.append(dict(rollback_audit))
                        stopped_reason = str(rollback_audit["rollback_reason"]) + "_rollback"
                        break
                    else:
                        round_audit = {
                            **dict(round_audit),
                            "rolled_back": False,
                        }
                        refinement_rounds.append(dict(round_audit))
                        current_audit = dict(final_audit)

                accepted_rounds = [
                    row
                    for row in refinement_rounds
                    if bool(row.get("enabled", False)) and not bool(row.get("rolled_back", False))
                ]
                accepted_count = int(
                    sum(int(row.get("accepted_count", 0) or 0) for row in accepted_rounds)
                )
                input_level, num_bootstraps, bootstrapper_slots, current_audit, final_halo0_cleanup_audit = (
                    _run_final_halo0_cleanup(current_audit)
                )
                final_halo0_cleanup_enabled = bool(final_halo0_cleanup_audit.get("enabled", False))
                final_halo0_cleanup_accepted_count = int(
                    final_halo0_cleanup_audit.get("accepted_count", 0) or 0
                )
                network_dag.bootstrap_layout_compression_audit = {
                    "enabled": False,
                    "reason": "staged_bootstrap_aware_refinement"
                    if accepted_count > 0 or final_halo0_cleanup_enabled
                    else "staged_bootstrap_aware_refinement_no_candidates",
                }
                network_dag.bootstrap_layout_refinement_audit = {
                    "enabled": bool(accepted_count > 0 or final_halo0_cleanup_enabled),
                    "policy": str(refinement_policy),
                    "fixed_point": bool(rollback_audit is None and stopped_reason != "max_rounds_reached"),
                    "stopped_reason": str(stopped_reason),
                    "rolled_back": bool(rollback_audit is not None),
                    "first_pass_bootstrap_count": int(initial_audit.get("bootstrap_count", 0) or 0),
                    "final_bootstrap_count": int(num_bootstraps),
                    "target_bootstrap_count": (
                        None if target_bootstrap_count is None else int(target_bootstrap_count)
                    ),
                    "target_bootstrap_source": str(target_bootstrap_source),
                    "target_bootstrap_error": str(target_bootstrap_error),
                    "target_boot_edges": list((target_bootstrap_audit or {}).get("boot_edges", [])),
                    "accepted_round_count": int(len(accepted_rounds)),
                    "accepted_count": int(accepted_count),
                    "final_halo0_cleanup_enabled": bool(final_halo0_cleanup_enabled),
                    "final_halo0_cleanup_accepted_count": int(final_halo0_cleanup_accepted_count),
                    "final_halo0_cleanup": dict(final_halo0_cleanup_audit),
                    "candidate_trial_count": int(
                        sum(len(row.get("trials", []) or []) for row in refinement_rounds)
                    ),
                    "trial_evaluator_mode": str(trial_evaluator_mode),
                    "trial_evaluator_enabled_final": bool(trial_evaluator_enabled),
                    "trial_evaluator_fallback_reason": str(trial_evaluator_fallback_reason),
                    "trial_evaluator_fallback_round": trial_evaluator_fallback_round,
                    "rounds": refinement_rounds,
                }
            else:
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
