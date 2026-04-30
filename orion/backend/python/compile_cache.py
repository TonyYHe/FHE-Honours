import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from orion.nn.linear import LinearTransform
from orion.nn.module import Module


SCHEMA_VERSION = 3
CACHE_FORMAT_VERSION = "compile-plan-v3"
MANIFEST_NAME = "compile_manifest.json"


@dataclass(frozen=True)
class BackendPayloadRef:
    kind: str
    path: str
    backend: str
    storage: str = "hdf5"


@dataclass(frozen=True)
class BackendTransformDescriptor:
    row: int
    col: int
    diag_indices: list[int]
    diag_indices_sha256: str
    level: int | None
    scale: int | None
    bsgs_ratio: float
    slot_count: int
    rotation_requests: list[dict[str, int | None]]
    transform_id: int | None
    payload_refs: list[BackendPayloadRef]


@dataclass(frozen=True)
class BackendCompileManifest:
    schema_version: int
    cache_format_version: str
    fingerprint: dict[str, Any]
    bootstrap_plan: dict[str, Any]
    transform_metadata: dict[str, Any]
    provider_metadata: dict[str, Any]
    sha256: str


def manifest_path(params) -> str:
    diags_path = str(params.get_diags_path() or "")
    if not diags_path:
        return ""
    return str(Path(diags_path).resolve().parent / MANIFEST_NAME)


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _hash_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    payload = cpu.numpy().tobytes()
    header = f"{tuple(cpu.shape)}|{cpu.dtype}".encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(payload)
    return digest.hexdigest()


def params_fingerprint(params) -> dict[str, Any]:
    return {
        "ckks": {
            "logn": int(params.get_logn()),
            "logq": [int(v) for v in params.get_logq()],
            "logp": [int(v) for v in params.get_logp()],
            "logscale": int(params.get_logscale()),
            "h": int(params.get_hamming_weight()),
            "ringtype": str(params.get_ringtype()).lower(),
            "boot_logp": [int(v) for v in params.get_boot_logp()],
        },
        "orion": {
            "backend": str(params.get_backend()).lower(),
            "margin": int(params.get_margin()),
            "embedding_method": str(params.get_embedding_method()).lower(),
            "fuse_modules": bool(params.get_fuse_modules()),
            "debug": bool(params.get_debug_status()),
            "experimental_region_first": str(params.get_experimental_region_first()).lower(),
        },
    }


def model_fingerprint(net: torch.nn.Module) -> dict[str, Any]:
    modules = [
        (str(name), type(module).__module__, type(module).__qualname__)
        for name, module in net.named_modules()
    ]
    state = []
    for name, tensor in net.state_dict().items():
        state.append(
            {
                "name": str(name),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": _hash_tensor(tensor),
            }
        )
    payload = {"modules": modules, "state": state}
    return {
        "sha256": _hash_json(payload),
        "module_count": int(len(modules)),
        "state_count": int(len(state)),
    }


def cache_fingerprint(params, net: torch.nn.Module) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "params": params_fingerprint(params),
        "model": model_fingerprint(net),
    }
    payload["sha256"] = _hash_json(payload)
    return payload


def _module_depth(module) -> int | None:
    depth = getattr(module, "depth", None)
    return None if depth is None else int(depth)


def _module_level(module) -> int | None:
    level = getattr(module, "level", None)
    return None if level is None else int(level)


def _shape(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [int(v) for v in list(value)]
    except TypeError:
        return []


def _tensor_shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    return _shape(shape)


def _module_scale(module) -> int | None:
    scheme = getattr(module, "scheme", None)
    params = getattr(scheme, "params", None)
    if params is None or not hasattr(params, "get_logscale"):
        return None
    return int(1 << int(params.get_logscale()))


def _slot_count_from_diags(module, diags: dict[int, Any]) -> int:
    for diag in diags.values():
        try:
            length = int(len(diag))
            if length > 0:
                return length
        except TypeError:
            continue
    scheme = getattr(module, "scheme", None)
    params = getattr(scheme, "params", None)
    if params is not None and hasattr(params, "get_slots"):
        return int(params.get_slots())
    return 0


def _rotation_requests_for_layer(layer: LinearTransform, transform_id: int | None) -> list[dict[str, int | None]]:
    if transform_id is None:
        return []
    evaluator = getattr(getattr(layer, "scheme", None), "lt_evaluator", None)
    get_requests = getattr(evaluator, "get_required_rotation_key_requests", None)
    if not callable(get_requests):
        return []
    return [
        {"key": int(key), "level": None if level is None else int(level)}
        for key, level in get_requests(int(transform_id))
    ]


def _payload_refs(layer: LinearTransform, row: int, col: int) -> list[BackendPayloadRef]:
    params = getattr(getattr(layer, "scheme", None), "params", None)
    backend = str(params.get_backend()).lower() if params is not None and hasattr(params, "get_backend") else ""
    layer_name = str(getattr(layer, "name", ""))
    block = f"{int(row)}_{int(col)}"
    return [
        BackendPayloadRef("raw_diagonals", f"/{layer_name}/diagonals/{block}", backend),
        BackendPayloadRef("plaintext_payload", f"/{layer_name}/plaintexts/{block}", backend),
        BackendPayloadRef("rotation_keys", "/keys", backend),
    ]


def _bootstrap_slots_for_node(network_dag, node: str) -> int | None:
    if not bool(network_dag.nodes[node].get("bootstrap", False)):
        return None
    module = network_dag.nodes[node].get("module")
    if module is None:
        return None
    max_slots = int(module.scheme.params.get_slots())
    elements = int(module.fhe_output_shape.numel())
    curr_slots = 1 << int((elements - 1).bit_length())
    return int(min(max_slots, curr_slots))


def collect_bootstrap_plan(
    network_dag,
    topo_sort: list[str],
    *,
    input_level: int,
    bootstrap_count: int,
    bootstrapper_slots: list[int],
) -> dict[str, Any]:
    nodes = []
    for node in topo_sort:
        module = network_dag.nodes[node].get("module")
        nodes.append(
            {
                "name": str(node),
                "module_type": "" if module is None else f"{type(module).__module__}.{type(module).__qualname__}",
                "level": _module_level(module) if module is not None else None,
                "depth": _module_depth(module) if module is not None else None,
                "bootstrap": bool(network_dag.nodes[node].get("bootstrap", False)),
                "bootstrap_slots": _bootstrap_slots_for_node(network_dag, str(node)),
            }
        )
    return {
        "input_level": int(input_level),
        "topological_order": [str(node) for node in topo_sort],
        "bootstrap_count": int(bootstrap_count),
        "bootstrapper_slots": [int(v) for v in bootstrapper_slots],
        "nodes": nodes,
        "sha256": _hash_json(nodes),
    }


def apply_bootstrap_plan(network_dag, plan: dict[str, Any]) -> tuple[int, int, list[int]]:
    nodes = {str(row["name"]): row for row in plan.get("nodes", [])}
    missing = [str(node) for node in network_dag.nodes if str(node) not in nodes]
    if missing:
        raise RuntimeError(f"Cached compile plan is missing nodes: {missing[:8]}")

    for node in network_dag.nodes:
        row = nodes[str(node)]
        module = network_dag.nodes[node].get("module")
        level = row.get("level")
        network_dag.nodes[node]["level"] = None if level is None else int(level)
        network_dag.nodes[node]["bootstrap"] = bool(row.get("bootstrap", False))
        if module is not None and level is not None:
            module.level = int(level)
        depth = row.get("depth")
        if module is not None and depth is not None and hasattr(module, "depth"):
            module.depth = int(depth)

    return (
        int(plan["input_level"]),
        int(plan.get("bootstrap_count", 0)),
        [int(v) for v in plan.get("bootstrapper_slots", [])],
    )


def _diag_index_digest(diags: dict) -> dict[str, Any]:
    indices = sorted(int(idx) for idx in diags.keys())
    return {
        "diag_count": int(len(indices)),
        "diag_indices_sha256": _hash_json(indices),
    }


def collect_transform_metadata(linear_layers: list[LinearTransform]) -> dict[str, Any]:
    layers = []
    for layer in linear_layers:
        blocks = []
        diagonals = getattr(layer, "diagonals", {}) or {}
        for key in sorted(diagonals):
            row, col = key
            diag_indices = sorted(int(idx) for idx in diagonals[key].keys())
            transform_id = None
            transform_ids = getattr(layer, "transform_ids", {}) or {}
            if key in transform_ids:
                transform_id = int(transform_ids[key])
            block = {
                "row": int(row),
                "col": int(col),
                "descriptor": asdict(
                    BackendTransformDescriptor(
                        row=int(row),
                        col=int(col),
                        diag_indices=list(diag_indices),
                        diag_indices_sha256=_hash_json(diag_indices),
                        level=_module_level(layer),
                        scale=_module_scale(layer),
                        bsgs_ratio=float(getattr(layer, "bsgs_ratio", 0.0)),
                        slot_count=_slot_count_from_diags(layer, diagonals[key]),
                        rotation_requests=_rotation_requests_for_layer(layer, transform_id),
                        transform_id=transform_id,
                        payload_refs=_payload_refs(layer, int(row), int(col)),
                    )
                ),
            }
            block.update(_diag_index_digest(diagonals[key]))
            blocks.append(block)
        layers.append(
            {
                "name": str(getattr(layer, "name", "")),
                "module_type": f"{type(layer).__module__}.{type(layer).__qualname__}",
                "level": _module_level(layer),
                "depth": _module_depth(layer),
                "shape": _shape(getattr(layer, "input_shape", None)),
                "on_shape": _tensor_shape(getattr(layer, "on_weight", None)),
                "input_shape": _shape(getattr(layer, "input_shape", None)),
                "output_shape": _shape(getattr(layer, "output_shape", None)),
                "fhe_input_shape": _shape(getattr(layer, "fhe_input_shape", None)),
                "fhe_output_shape": _shape(getattr(layer, "fhe_output_shape", None)),
                "bsgs_ratio": float(getattr(layer, "bsgs_ratio", 0.0)),
                "output_rotations": int(getattr(layer, "output_rotations", 0)),
                "block_count": int(len(blocks)),
                "blocks": blocks,
            }
        )
    payload = {"layers": layers}
    payload["sha256"] = _hash_json(payload)
    return payload


def collect_provider_metadata(network_dag) -> dict[str, Any]:
    rows = []
    for node in network_dag.nodes:
        module = network_dag.nodes[node].get("module")
        runtime = getattr(module, "region_runtime", None) if module is not None else None
        if runtime is None:
            continue
        executor = getattr(runtime, "executor", None)
        rows.append(
            {
                "node": str(node),
                "runtime_type": f"{type(runtime).__module__}.{type(runtime).__qualname__}",
                "executor_type": "" if executor is None else f"{type(executor).__module__}.{type(executor).__qualname__}",
                "executable": bool(getattr(runtime, "executable", False)),
                "assigned_level": None if getattr(runtime, "assigned_level", None) is None else int(runtime.assigned_level),
                "assigned_depth": None if getattr(runtime, "assigned_depth", None) is None else int(runtime.assigned_depth),
            }
        )
    payload = {"rows": rows}
    payload["sha256"] = _hash_json(payload)
    return payload


def build_manifest(
    *,
    params,
    net: torch.nn.Module,
    network_dag,
    topo_sort: list[str],
    input_level: int,
    bootstrap_count: int,
    bootstrapper_slots: list[int],
    linear_layers: list[LinearTransform] | None = None,
    fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "fingerprint": dict(fingerprint) if fingerprint is not None else cache_fingerprint(params, net),
        "bootstrap_plan": collect_bootstrap_plan(
            network_dag,
            topo_sort,
            input_level=int(input_level),
            bootstrap_count=int(bootstrap_count),
            bootstrapper_slots=list(bootstrapper_slots),
        ),
        "transform_metadata": collect_transform_metadata(linear_layers or []),
        "provider_metadata": collect_provider_metadata(network_dag),
    }
    manifest["sha256"] = _hash_json(
        {
            "fingerprint": manifest["fingerprint"],
            "bootstrap_plan": manifest["bootstrap_plan"],
            "transform_metadata": manifest["transform_metadata"],
            "provider_metadata": manifest["provider_metadata"],
        }
    )
    return manifest


def read_manifest(path: str) -> dict[str, Any]:
    if not path:
        raise RuntimeError("io_mode='load' requires a diagonals path so the compile manifest can be located")
    if not os.path.exists(path):
        raise RuntimeError(f"Missing compile cache manifest: {path}. Re-run with io_mode='save'.")
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(
            "Stale compile cache manifest. "
            f"cached_schema={manifest.get('schema_version')} expected_schema={SCHEMA_VERSION}. "
            "Re-run with io_mode='save' to rebuild v3 cache."
        )
    if str(manifest.get("cache_format_version", "")) != CACHE_FORMAT_VERSION:
        raise RuntimeError(
            "Stale compile cache manifest. "
            f"cached_format={manifest.get('cache_format_version')} expected_format={CACHE_FORMAT_VERSION}. "
            "Re-run with io_mode='save' to rebuild v3 cache."
        )
    return manifest


def write_manifest(path: str, manifest: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def validate_manifest_identity(manifest: dict[str, Any], *, params, net: torch.nn.Module) -> None:
    expected = cache_fingerprint(params, net)
    cached = manifest.get("fingerprint", {})
    if cached.get("sha256") != expected.get("sha256"):
        raise RuntimeError(
            "Compile cache fingerprint mismatch. "
            f"cached={cached.get('sha256')} current={expected.get('sha256')}"
        )


def validate_topology(manifest: dict[str, Any], topo_sort: list[str]) -> None:
    cached_order = [str(node) for node in manifest.get("bootstrap_plan", {}).get("topological_order", [])]
    current_order = [str(node) for node in topo_sort]
    if cached_order != current_order:
        raise RuntimeError(
            "Compile cache topology mismatch. "
            f"cached_nodes={len(cached_order)} current_nodes={len(current_order)}"
        )


def validate_transform_metadata(manifest: dict[str, Any], linear_layers: list[LinearTransform]) -> None:
    cached_layers = {
        str(row.get("name")): row
        for row in manifest.get("transform_metadata", {}).get("layers", [])
    }
    current = collect_transform_metadata(linear_layers)
    def comparable_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for block in blocks:
            copied = dict(block)
            descriptor = dict(copied.get("descriptor", {}))
            descriptor.pop("transform_id", None)
            copied["descriptor"] = descriptor
            out.append(copied)
        return out

    for row in current.get("layers", []):
        name = str(row.get("name"))
        cached = cached_layers.get(name)
        if cached is None:
            raise RuntimeError(f"Compile cache transform metadata missing layer {name!r}")
        comparable = {
            "level": row.get("level"),
            "depth": row.get("depth"),
            "input_shape": row.get("input_shape"),
            "output_shape": row.get("output_shape"),
            "fhe_input_shape": row.get("fhe_input_shape"),
            "fhe_output_shape": row.get("fhe_output_shape"),
            "bsgs_ratio": row.get("bsgs_ratio"),
            "output_rotations": row.get("output_rotations"),
            "block_count": row.get("block_count"),
            "blocks": comparable_blocks(row.get("blocks", [])),
        }
        cached_comparable = {
            "level": cached.get("level"),
            "depth": cached.get("depth"),
            "input_shape": cached.get("input_shape"),
            "output_shape": cached.get("output_shape"),
            "fhe_input_shape": cached.get("fhe_input_shape"),
            "fhe_output_shape": cached.get("fhe_output_shape"),
            "bsgs_ratio": cached.get("bsgs_ratio"),
            "output_rotations": cached.get("output_rotations"),
            "block_count": cached.get("block_count"),
            "blocks": comparable_blocks(cached.get("blocks", [])),
        }
        if comparable != cached_comparable:
            raise RuntimeError(
                f"Compile cache transform metadata mismatch for layer {name!r}: "
                f"cached={cached_comparable} current={comparable}"
            )


def transform_layer(manifest: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in manifest.get("transform_metadata", {}).get("layers", []):
        if str(row.get("name")) == str(name):
            return row
    return None


def transform_blocks_from_manifest(manifest: dict[str, Any], name: str) -> dict[tuple[int, int], dict[int, list]]:
    layer = transform_layer(manifest, name)
    if layer is None:
        raise RuntimeError(f"Compile cache transform metadata missing layer {name!r}")
    blocks: dict[tuple[int, int], dict[int, list]] = {}
    for block in layer.get("blocks", []):
        descriptor = block.get("descriptor", {})
        row = int(descriptor.get("row", block.get("row", 0)))
        col = int(descriptor.get("col", block.get("col", 0)))
        diag_indices = descriptor.get("diag_indices")
        if diag_indices is None:
            diag_indices = sorted(int(idx) for idx in block.get("diag_indices", []))
        blocks[(row, col)] = {int(idx): [] for idx in diag_indices}
    return blocks


def transform_output_rotations(manifest: dict[str, Any], name: str) -> int:
    layer = transform_layer(manifest, name)
    if layer is None:
        raise RuntimeError(f"Compile cache transform metadata missing layer {name!r}")
    return int(layer.get("output_rotations", 0))


def cleanup_compile_artifacts(params) -> None:
    for path, label in (
        (getattr(params, "get_keys_path")(), "keys"),
        (getattr(params, "get_diags_path")(), "diagonals"),
        (manifest_path(params), "compile manifest"),
    ):
        if not path:
            continue
        target = Path(path)
        print(f"Deleting existing {label} at {target}", flush=True)
        if target.exists():
            target.unlink()
