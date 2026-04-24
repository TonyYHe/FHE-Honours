from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from orion.core.orion import scheme
from orion.core.tracer import OrionTracer, StatsTracker
from orion.core.network_dag import NetworkDAG
from orion.core.fuser import Fuser
from orion.core.auto_bootstrap import BootstrapSolver
from orion.experimental.cir.runtime_group import RegionFirstCompileRegistry
from orion.models.resnet import ResNet18


DEFAULT_OUT = Path("/run/media/anakano/7TB/haloed-cache/orion_runtime_only/orion_r18_stage0_region_only_silu7.json")


def _build_config() -> dict:
    return {
        "ckks_params": {
            "LogN": 16,
            "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            "LogP": [61, 61, 61],
            "LogScale": 40,
            "H": 192,
            "RingType": "Standard",
        },
        "boot_params": {"LogP": [61, 61, 61, 61, 61, 61, 61, 61]},
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }


def _timed(payload: dict, out_path: Path, step: str, fn):
    payload["step"] = step
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    started = time.time()
    value = fn()
    payload.setdefault("timing_s", {})[step] = float(time.time() - started)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return value


def _collect_executor_audit(module) -> dict:
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None)
    return {
        "node": str(getattr(module, "region_output_id", getattr(module, "name", ""))),
        "compile_count": int(getattr(executor, "compile_count", 0)) if executor is not None else 0,
        "last_runtime_timing": dict(getattr(executor, "last_runtime_timing", {})) if executor is not None else {},
        "lazy_region_compile": bool(getattr(module, "region_first_probe_lazy_region_compile", False)),
        "assigned_level": None if runtime is None else getattr(runtime, "assigned_level", None),
        "assigned_depth": None if runtime is None else getattr(runtime, "assigned_depth", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile/run a single stage0 region-only R18 SiLU7 CIR kernel.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--node", default="layers_0_0_conv1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = _build_config()
    payload = {
        "status": "started",
        "step": "init",
        "mode": "stage0_region_only",
        "config": config,
        "activation": {
            "kind": "silu",
            "silu_degree": 7,
            "stem_relu": True,
            "expected_bootstraps_reference": 61,
        },
        "target_node": str(args.node),
        "timing_s": {},
        "audit": {},
        "claim": {
            "publishable": False,
            "reason": "single-kernel region-only compile/run verification",
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    try:
        torch.manual_seed(int(args.seed))
        x = torch.randn((1, 3, 64, 64), dtype=torch.float32)
        net = ResNet18(dataset="tiny", activation="silu", silu_degree=7, stem_relu=True)
        net.eval()

        _timed(payload, out_path, "init_scheme", lambda: scheme.init_scheme(config))
        _timed(payload, out_path, "fit", lambda: scheme.fit(net, x))

        traced = OrionTracer().trace_model(net)
        StatsTracker(traced).propagate(x)
        dag = NetworkDAG(traced)
        dag.build_dag()

        for module in net.modules():
            if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
                module.init_orion_params()
        for module in net.modules():
            if hasattr(module, "update_params") and callable(module.update_params):
                module.update_params()

        if scheme.params.get_fuse_modules():
            fuser = Fuser(dag)
            fuser.fuse_modules()
            dag.remove_fused_batchnorms()

        registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag, allowed_stages=("stage1",))
        payload["audit"]["attach"] = registry.attach_to_dag(dag)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        dag.find_residuals()
        _timed(payload, out_path, "bootstrap_solver", lambda: BootstrapSolver(net, dag, l_eff=len(scheme.params.get_logq()) - 1).solve())

        module = dag.nodes[str(args.node)]["module"]
        runtime = getattr(module, "region_runtime", None)
        if runtime is None:
            raise RuntimeError(f"node {args.node} has no region_runtime")

        payload["audit"]["before_compile"] = _collect_executor_audit(module)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        _timed(payload, out_path, "region_compile", lambda: module.compile())
        payload["audit"]["after_compile"] = _collect_executor_audit(module)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        input_level = int(module.level + 4)
        stem = net.pool(net.act(net.bn1(net.conv1(x))))

        # Build the stage0 source through the real builder kwargs held by the executor.
        executor = runtime.executor
        ids = []
        for builder, kwargs in zip(getattr(executor, "_plan_builders", ()), getattr(executor, "_builder_kwargs", ())):
            plan, inputs, _reference = builder(**{**kwargs, "source_override": stem[0]})
            _ = plan
            left = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
            right = inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
            ct_left = scheme.encrypt(scheme.encode(left, input_level))
            ct_right = scheme.encrypt(scheme.encode(right, input_level))
            ids.extend(int(v) for v in ct_left.ids)
            ids.extend(int(v) for v in ct_right.ids)

        from orion.backend.python.tensors import CipherTensor

        source_ct = CipherTensor(
            scheme,
            ids,
            torch.Size([1, 64, 64, 64]),
            torch.Size([1, 64, 64, 64]),
        )
        payload["audit"]["source_ciphertext_count"] = int(len(source_ct.ids))
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        module.he_mode = True
        out = _timed(payload, out_path, "region_forward", lambda: module(source_ct))
        payload["audit"]["after_forward"] = _collect_executor_audit(module)
        payload["audit"]["output"] = {
            "ciphertext_count": int(len(out.ids)),
            "level": int(out.level()),
            "scale": float(out.scale()),
            "shape": list(out.shape),
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        payload["status"] = "ok_probe_only"
        payload["step"] = "done"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc(limit=80)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        try:
            scheme.delete_scheme()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
