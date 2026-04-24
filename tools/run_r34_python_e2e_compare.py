from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import orion

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.fuser import Fuser
from orion.core.network_dag import NetworkDAG
from orion.models.resnet import BasicBlock, ResNet, get_resnet_config
from orion.nn.linear import LinearTransform
from orion.nn.module import Module
from orion.experimental import R34CompileRegistry


DEFAULT_OUT = Path("/tmp/orion_r34_python_e2e_compare.json")
CAPTURE_NODES = (
    ("avgpool", "avgpool"),
    ("linear", "linear"),
)


def _build_config(*, experimental_region_first: str) -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": 16,
            "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            "LogP": [61, 61, 61],
            "LogScale": 40,
            "H": 192,
            "RingType": "Standard",
        },
        "boot_params": {"LogP": [61, 61, 61, 61, 61, 61, 61, 61]},
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "python",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
            "experimental_region_first": str(experimental_region_first or ""),
        },
    }


def _decode_module_output(module: Any, output: Any) -> torch.Tensor:
    if not isinstance(output, CipherTensor):
        return output.detach().cpu().to(dtype=torch.float32)

    decoded = output.decrypt().decode().detach().cpu()
    if torch.is_complex(decoded):
        decoded = decoded.real
    decoded = decoded.to(dtype=torch.float32)

    output_shape = getattr(module, "output_shape", None)
    fhe_output_shape = getattr(module, "fhe_output_shape", None)
    output_gap = getattr(module, "output_gap", None)
    if (
        output_shape is not None
        and fhe_output_shape is not None
        and output_gap is not None
        and len(tuple(output_shape)) == 4
        and len(tuple(fhe_output_shape)) == 4
    ):
        flat = decoded.flatten()
        packed_size = int(fhe_output_shape[1] * fhe_output_shape[2] * fhe_output_shape[3])
        packed = flat[:packed_size].reshape(
            1,
            int(fhe_output_shape[1]),
            int(fhe_output_shape[2]),
            int(fhe_output_shape[3]),
        )
        return packing._demultiplex(
            packed,
            int(output_gap),
            int(output_shape[1]),
            int(output_shape[2]),
            int(output_shape[3]),
        )[0].detach().cpu().to(dtype=torch.float32)

    return decoded.detach().cpu().to(dtype=torch.float32)


def _build_r34_silu() -> ResNet:
    conv1_params, num_classes = get_resnet_config("imagenet")
    return ResNet(
        "imagenet",
        BasicBlock,
        [3, 4, 6, 3],
        [64, 128, 256, 512],
        conv1_params,
        num_classes,
        activation="silu",
        silu_degree=7,
        stem_relu=False,
    )


def _capture_map(net: ResNet, x: Any, *, he_mode: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    modules = dict(net.named_modules())
    missing = [module_name for module_name, _report_name in CAPTURE_NODES if module_name not in modules]
    if missing:
        raise KeyError(f"missing capture nodes: {missing}")

    def _make_hook(report_name: str):
        def _hook(module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            captured[str(report_name)] = _decode_module_output(module, output)

        return _hook

    for module_name, report_name in CAPTURE_NODES:
        handles.append(modules[str(module_name)].register_forward_hook(_make_hook(str(report_name))))

    try:
        out = net(x)
        final = _decode_module_output(modules["linear"], out)
    finally:
        for handle in handles:
            handle.remove()
    return final, captured


def _metrics(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    diff = other.detach().to(dtype=torch.float32) - reference.detach().to(dtype=torch.float32)
    return {
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.pow(2))).item()),
    }


def _run_one(*, experimental_region_first: str, model_seed: int, input_tensor: torch.Tensor) -> dict[str, Any]:
    config = _build_config(experimental_region_first=str(experimental_region_first))
    scheme = orion.init_scheme(copy.deepcopy(config))
    try:
        torch.manual_seed(int(model_seed))
        net = _build_r34_silu()
        net.eval()

        clear_started = time.perf_counter()
        clear_final, clear_nodes = _capture_map(net, input_tensor, he_mode=False)
        clear_s = float(time.perf_counter() - clear_started)

        fit_started = time.perf_counter()
        orion.fit(net, input_tensor)
        fit_s = float(time.perf_counter() - fit_started)

        compile_started = time.perf_counter()
        if str(experimental_region_first) == "r34_imgnet_phase1":
            input_level = _manual_provider_compile(scheme, net)
        else:
            input_level = orion.compile(net)
        compile_s = float(time.perf_counter() - compile_started)

        encrypted = orion.encrypt(orion.encode(input_tensor, input_level))
        net.he()

        he_started = time.perf_counter()
        he_final, he_nodes = _capture_map(net, encrypted, he_mode=True)
        he_s = float(time.perf_counter() - he_started)

        return {
            "config": config,
            "timing_s": {
                "clear": float(clear_s),
                "fit": float(fit_s),
                "compile": float(compile_s),
                "he_forward": float(he_s),
            },
            "clear_final": clear_final,
            "clear_nodes": clear_nodes,
            "he_final": he_final,
            "he_nodes": he_nodes,
        }
    finally:
        scheme.delete_scheme()


def _manual_provider_compile(scheme: Any, net: ResNet) -> int:
    if scheme.traced is None:
        raise RuntimeError("manual provider compile requires scheme.traced from orion.fit")

    network_dag = NetworkDAG(scheme.traced)
    network_dag.build_dag()

    for module in net.modules():
        if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
            module.init_orion_params()
    for module in net.modules():
        if hasattr(module, "update_params") and callable(module.update_params):
            module.update_params()

    fuser = Fuser(network_dag)
    fuser.fuse_modules()
    network_dag.remove_fused_batchnorms()

    registry = R34CompileRegistry.for_r34_imgnet_phase1(network_dag)
    scheme.region_first_registry = registry
    scheme.region_first_attach_audit = registry.attach_to_dag(network_dag)

    level = len(scheme.params.get_logq()) - 1
    topo_sort = list(network_dag.topological_sort())
    for node in topo_sort:
        module = network_dag.nodes[node]["module"]
        if isinstance(module, Module):
            module.set_level(level)

    for node in topo_sort:
        module = network_dag.nodes[node]["module"]
        if isinstance(module, Module):
            module.compile()
        elif isinstance(module, LinearTransform):
            module.compile()

    return int(level)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dense vs provider R34 end-to-end comparison on the Python backend.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    x = torch.randn((1, 3, 224, 224), dtype=torch.float32)

    provider = _run_one(experimental_region_first="r34_imgnet_phase1", model_seed=0, input_tensor=x)
    dense = None if bool(args.provider_only) else _run_one(experimental_region_first="", model_seed=0, input_tensor=x)

    node_rows = []
    for _module_name, report_name in CAPTURE_NODES:
        clear = provider["clear_nodes"][str(report_name)]
        provider_he = provider["he_nodes"][str(report_name)]
        row = {
            "node": str(report_name),
            "provider_vs_clear": _metrics(clear, provider_he),
        }
        if dense is not None:
            dense_he = dense["he_nodes"][str(report_name)]
            row["dense_vs_clear"] = _metrics(clear, dense_he)
            row["provider_vs_dense"] = _metrics(dense_he, provider_he)
        node_rows.append(row)

    payload = {
        "status": "ok",
        "scope": "R34 ImageNet Python-backend end-to-end dense vs provider comparison with captured intermediate nodes",
        "model": {
            "kind": "r34_silu_python_e2e",
            "activation": "silu",
            "silu_degree": 7,
            "stem_relu": False,
            "reason": "Python backend does not currently implement ReLU minimax sign fitting.",
        },
        "capture_nodes": [str(report_name) for _module_name, report_name in CAPTURE_NODES],
        "timing_s": {
            "provider": dict(provider["timing_s"]),
        },
        "final": {"provider_vs_clear": _metrics(provider["clear_final"], provider["he_final"])},
        "nodes": node_rows,
    }
    if dense is not None:
        payload["timing_s"]["dense"] = dict(dense["timing_s"])
        payload["final"]["dense_vs_clear"] = _metrics(dense["clear_final"], dense["he_final"])
        payload["final"]["provider_vs_dense"] = _metrics(dense["he_final"], provider["he_final"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
