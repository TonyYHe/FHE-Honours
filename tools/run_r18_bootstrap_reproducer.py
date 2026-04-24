from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from orion.core.auto_bootstrap import BootstrapPlacer
from orion.core.orion import scheme
from orion.models.resnet import ResNet18
from orion.nn.operations import Bootstrap as OrionBootstrap


DEFAULT_OUT = Path("/tmp/orion_r18_bootstrap_reproducer.json")
DEFAULT_TRACE = Path("/tmp/orion_r18_bootstrap_trace.jsonl")


def _build_config(experimental_region_first: str) -> dict:
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
            "experimental_region_first": str(experimental_region_first or ""),
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


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return out
    return out


def _ct_contract(ct: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "ids": [int(value) for value in getattr(ct, "ids", ())],
        "level": int(ct.level()) if hasattr(ct, "level") else None,
        "scale_uint64": int(ct.scale()) if hasattr(ct, "scale") else None,
        "log_scale": _safe_float(ct.log_scale()) if hasattr(ct, "log_scale") else None,
        "slots": int(ct.slots()) if hasattr(ct, "slots") else None,
        "degree": int(ct.degree()) if hasattr(ct, "degree") else None,
    }
    level = info["level"]
    if level is not None and hasattr(ct, "moduli"):
        try:
            moduli = list(ct.moduli())
            if 0 <= int(level) < len(moduli):
                q = int(moduli[int(level)])
                info["modulus_at_level"] = q
                info["log_modulus"] = _safe_float(math.log2(q))
                if info["log_scale"] not in {None, float("-inf")}:
                    info["log_q_over_scale"] = _safe_float(float(info["log_modulus"]) - float(info["log_scale"]))
        except Exception:
            pass
    return info


def _install_bootstrap_trace(trace_path: Path) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("", encoding="utf-8")

    def append(entry: dict[str, Any]) -> None:
        trace_path.write_text("", encoding="utf-8") if False else None
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
            handle.flush()

    def _module_trace_info(node: str, module: Any) -> dict[str, Any]:
        runtime = getattr(module, "region_runtime", None)
        return {
            "node": str(node),
            "module_type": type(module).__name__ if module is not None else "None",
            "module_level": int(getattr(module, "level", -1) or -1),
            "module_depth": int(getattr(module, "depth", -1) or -1),
            "is_region_runtime": bool(runtime is not None),
            "region_runtime_executable": bool(getattr(runtime, "executable", False)) if runtime is not None else False,
            "region_output_id": str(getattr(module, "region_output_id", "")) if module is not None else "",
            "boundary_actions": list(getattr(runtime, "boundary_actions", ())) if runtime is not None else [],
        }

    def _register_parent_hook(node: str, module: Any) -> None:
        if module is None or bool(getattr(module, "_bootstrap_parent_trace_installed", False)):
            return
        module._bootstrap_parent_trace_installed = True
        module._bootstrap_parent_trace_node = str(node)

        def parent_hook(mod: Any, _input: Any, output: Any) -> Any:
            append(
                {
                    "event": "parent_output",
                    "time_s": float(time.time()),
                    **_module_trace_info(str(getattr(mod, "_bootstrap_parent_trace_node", node)), mod),
                    "contract": _ct_contract(output),
                }
            )
            return output

        module.register_forward_hook(parent_hook)

    def traced_apply(self: BootstrapPlacer, module: Any, *, node_name: str | None = None, parents: list[dict[str, Any]] | None = None) -> None:
        bootstrapper = self._create_bootstrapper(module)
        bootstrapper.debug_source_node = str(node_name or getattr(module, "name", type(module).__name__))
        bootstrapper.debug_module_type = type(module).__name__
        bootstrapper.debug_module_level = int(getattr(module, "level", -1))
        bootstrapper.debug_module_depth = int(getattr(module, "depth", 0) or 0)
        bootstrapper.debug_expected_output_level = int(
            bootstrapper.debug_module_level - bootstrapper.debug_module_depth
        )
        bootstrapper.debug_parents = list(parents or [])
        module.bootstrapper = bootstrapper

        def traced_hook(mod: Any, _input: Any, output: Any) -> Any:
            append(
                {
                    "event": "hook_pre_bootstrap",
                    "time_s": float(time.time()),
                    "node": bootstrapper.debug_source_node,
                    "module_type": bootstrapper.debug_module_type,
                    "module_level": bootstrapper.debug_module_level,
                    "module_depth": bootstrapper.debug_module_depth,
                    "expected_output_level": bootstrapper.debug_expected_output_level,
                    "bootstrap_input_level_planned": int(getattr(bootstrapper, "input_level", -1)),
                    "parents": list(getattr(bootstrapper, "debug_parents", [])),
                    "contract": _ct_contract(output),
                }
            )
            return bootstrapper(output)

        module.register_forward_hook(traced_hook)

    def traced_place_bootstraps(self: BootstrapPlacer) -> None:
        for node in self.network_dag.nodes:
            if not self.network_dag.nodes[node]["bootstrap"]:
                continue
            module = self.network_dag.nodes[node]["module"]
            if module is None:
                continue
            parents: list[dict[str, Any]] = []
            for parent in self.network_dag.predecessors(node):
                parent_module = self.network_dag.nodes[parent]["module"]
                parents.append(_module_trace_info(str(parent), parent_module))
                _register_parent_hook(str(parent), parent_module)
            traced_apply(self, module, node_name=str(node), parents=parents)

    def traced_bootstrap_forward(self: OrionBootstrap, x: Any) -> Any:
        if not self.he_mode:
            return x

        base = {
            "time_s": float(time.time()),
            "node": str(getattr(self, "debug_source_node", "unknown")),
            "module_type": str(getattr(self, "debug_module_type", type(self).__name__)),
            "module_level": int(getattr(self, "debug_module_level", -1)),
            "module_depth": int(getattr(self, "debug_module_depth", -1)),
            "expected_output_level": int(getattr(self, "debug_expected_output_level", -1)),
            "bootstrap_input_level_planned": int(getattr(self, "input_level", -1)),
            "prescale": _safe_float(getattr(self, "prescale", None)),
            "postscale": _safe_float(getattr(self, "postscale", None)),
            "constant": _safe_float(getattr(self, "constant", None)),
            "parents": list(getattr(self, "debug_parents", [])),
        }
        append({**base, "event": "bootstrap_pre_constant", "contract": _ct_contract(x)})

        if self.constant != 0:
            x = x + self.constant
            append({**base, "event": "bootstrap_post_constant", "contract": _ct_contract(x)})

        x = x * self._get_prescale_ptxt(x.level())
        append({**base, "event": "bootstrap_post_prescale", "contract": _ct_contract(x)})

        x = x.bootstrap()
        append({**base, "event": "bootstrap_post_bootstrap", "contract": _ct_contract(x)})

        if self.postscale != 1:
            x = x * self.postscale
            append({**base, "event": "bootstrap_post_postscale", "contract": _ct_contract(x)})
        if self.constant != 0:
            x = x - self.constant
            append({**base, "event": "bootstrap_post_finalize", "contract": _ct_contract(x)})

        return x

    BootstrapPlacer._apply_bootstrap_hook = traced_apply
    BootstrapPlacer.place_bootstraps = traced_place_bootstraps
    OrionBootstrap.forward = traced_bootstrap_forward


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce and trace the R18 honest bootstrap failure.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--experimental-region-first", type=str, default="r18_tiny_e2e")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _install_bootstrap_trace(Path(args.trace))

    payload = {
        "status": "started",
        "step": "init",
        "mode": "bootstrap_reproducer",
        "trace_path": str(args.trace),
        "config": _build_config(args.experimental_region_first),
        "region_first_mode": str(args.experimental_region_first),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    try:
        torch.manual_seed(int(args.seed))
        net = ResNet18(dataset="tiny", activation="silu", silu_degree=7, stem_relu=True)
        net.eval()
        x0 = torch.randn((1, 3, 64, 64), dtype=torch.float32)

        _timed(payload, out_path, "init_scheme", lambda: scheme.init_scheme(payload["config"]))
        _timed(payload, out_path, "fit", lambda: scheme.fit(net, x0))
        input_level = _timed(payload, out_path, "compile", lambda: scheme.compile(net))
        payload["input_level"] = int(input_level)
        payload["attach_audit"] = getattr(scheme, "region_first_attach_audit", {})
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        net.he()
        x0_ct = _timed(payload, out_path, "encrypt_fit_input", lambda: scheme.encrypt(scheme.encode(x0, input_level)))
        _timed(payload, out_path, "he_forward_fit_input", lambda: net(x0_ct))
        payload["status"] = "ok"
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
