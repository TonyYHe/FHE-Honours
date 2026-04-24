from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.orion import scheme
from orion.models.resnet import ResNet18
from orion.nn.activation import Activation, Chebyshev, Quad
from orion.nn.operations import Bootstrap, Mult

try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:
    def _dynamo_disable(fn):
        return fn


DEFAULT_CONFIG = Path("configs/resnet.yml")
DEFAULT_OUT = Path("/tmp/orion_r18_e2e_lattigo_compare.json")


class HeForwardBreakdownCollector:
    def __init__(self, net: torch.nn.Module) -> None:
        self.net = net
        self._module_names = {
            module: str(name) if str(name) else type(module).__name__
            for name, module in net.named_modules()
        }
        self._handles: list[Any] = []
        self._stack: list[dict[str, Any]] = []
        self._stats_by_category: dict[str, dict[str, float | int]] = {}
        self._stats_by_kind: dict[str, dict[str, float | int]] = {}
        self._stats_by_module: dict[str, dict[str, Any]] = {}

    def _categorize(self, module: torch.nn.Module) -> tuple[str, str] | None:
        if isinstance(module, Bootstrap):
            return ("bootstrap", "bootstrap")
        if isinstance(module, Mult):
            return ("activation", "mult")
        if isinstance(module, Quad):
            return ("activation", "quad")
        if isinstance(module, Activation):
            return ("activation", "activation")
        if isinstance(module, Chebyshev):
            return ("activation", "chebyshev")
        return None

    @_dynamo_disable
    def _pre_hook(self, module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
        if not bool(getattr(module, "he_mode", False)):
            return
        categorized = self._categorize(module)
        if categorized is None:
            return
        category, kind = categorized
        self._stack.append(
            {
                "module_id": id(module),
                "name": self._module_names.get(module, type(module).__name__),
                "category": str(category),
                "kind": str(kind),
                "started_at": time.perf_counter(),
                "child_s": 0.0,
            }
        )

    @_dynamo_disable
    def _post_hook(self, module: torch.nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        if not bool(getattr(module, "he_mode", False)):
            return
        frame = None
        for index in range(len(self._stack) - 1, -1, -1):
            if int(self._stack[index]["module_id"]) == id(module):
                frame = self._stack.pop(index)
                break
        if frame is None:
            return

        elapsed = float(time.perf_counter() - float(frame["started_at"]))
        exclusive_elapsed = float(max(0.0, elapsed - float(frame.get("child_s", 0.0))))
        if self._stack:
            self._stack[-1]["child_s"] = float(self._stack[-1].get("child_s", 0.0)) + elapsed

        category = str(frame["category"])
        kind = str(frame["kind"])
        name = str(frame["name"])

        category_stats = self._stats_by_category.setdefault(
            str(category),
            {"exclusive_total_s": 0.0, "inclusive_total_s": 0.0, "count": 0},
        )
        category_stats["exclusive_total_s"] = float(category_stats["exclusive_total_s"]) + exclusive_elapsed
        category_stats["inclusive_total_s"] = float(category_stats["inclusive_total_s"]) + elapsed
        category_stats["count"] = int(category_stats["count"]) + 1

        kind_stats = self._stats_by_kind.setdefault(
            str(kind),
            {"exclusive_total_s": 0.0, "inclusive_total_s": 0.0, "count": 0, "category": str(category)},
        )
        kind_stats["exclusive_total_s"] = float(kind_stats["exclusive_total_s"]) + exclusive_elapsed
        kind_stats["inclusive_total_s"] = float(kind_stats["inclusive_total_s"]) + elapsed
        kind_stats["count"] = int(kind_stats["count"]) + 1

        module_stats = self._stats_by_module.setdefault(
            str(name),
            {
                "exclusive_total_s": 0.0,
                "inclusive_total_s": 0.0,
                "count": 0,
                "category": str(category),
                "kind": str(kind),
            },
        )
        module_stats["exclusive_total_s"] = float(module_stats["exclusive_total_s"]) + exclusive_elapsed
        module_stats["inclusive_total_s"] = float(module_stats["inclusive_total_s"]) + elapsed
        module_stats["count"] = int(module_stats["count"]) + 1

    def install(self) -> None:
        for module in self.net.modules():
            if self._categorize(module) is None:
                continue
            self._handles.append(module.register_forward_pre_hook(self._pre_hook))
            self._handles.append(module.register_forward_hook(self._post_hook))

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []
        self._stack = []

    def summary(self, total_he_forward_s: float) -> dict[str, Any]:
        total_he_forward_s = float(total_he_forward_s)
        attributed_exclusive_total_s = float(
            sum(float(stats.get("exclusive_total_s", 0.0)) for stats in self._stats_by_category.values())
        )
        attributed_inclusive_total_s = float(
            sum(float(stats.get("inclusive_total_s", 0.0)) for stats in self._stats_by_category.values())
        )
        categories: dict[str, Any] = {}
        for category, stats in sorted(self._stats_by_category.items()):
            module_rows = [
                {
                    "name": str(name),
                    "kind": str(values.get("kind", "")),
                    "exclusive_total_s": float(values.get("exclusive_total_s", 0.0)),
                    "inclusive_total_s": float(values.get("inclusive_total_s", 0.0)),
                    "count": int(values.get("count", 0)),
                }
                for name, values in self._stats_by_module.items()
                if str(values.get("category", "")) == str(category)
            ]
            module_rows.sort(key=lambda row: float(row["exclusive_total_s"]), reverse=True)
            kind_rows = {
                str(kind): {
                    "exclusive_total_s": float(values.get("exclusive_total_s", 0.0)),
                    "inclusive_total_s": float(values.get("inclusive_total_s", 0.0)),
                    "count": int(values.get("count", 0)),
                }
                for kind, values in self._stats_by_kind.items()
                if str(values.get("category", "")) == str(category)
            }
            categories[str(category)] = {
                "exclusive_total_s": float(stats.get("exclusive_total_s", 0.0)),
                "inclusive_total_s": float(stats.get("inclusive_total_s", 0.0)),
                "count": int(stats.get("count", 0)),
                "exclusive_share_of_he_forward": (
                    float(stats.get("exclusive_total_s", 0.0)) / total_he_forward_s
                    if total_he_forward_s > 0.0
                    else None
                ),
                "by_kind": kind_rows,
                "top_modules": module_rows[:20],
            }
        return {
            "total_he_forward_s": float(total_he_forward_s),
            "attributed_exclusive_total_s": float(attributed_exclusive_total_s),
            "attributed_inclusive_total_s": float(attributed_inclusive_total_s),
            "unattributed_total_s": float(max(0.0, total_he_forward_s - attributed_exclusive_total_s)),
            "categories": categories,
        }


def _load_config(config_path: Path, *, mode: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = dict(config)
    config["ckks_params"] = dict(config.get("ckks_params", {}))
    config["boot_params"] = dict(config.get("boot_params", {}))
    config["orion"] = dict(config.get("orion", {}))
    config["orion"]["backend"] = "lattigo"
    config["orion"]["io_mode"] = "none"
    config["orion"]["debug"] = False
    config["orion"]["experimental_region_first"] = "r18_tiny_e2e" if str(mode) == "provider" else ""
    return config


def _timed(payload: dict[str, Any], out_path: Path, step: str, fn):
    payload["step"] = str(step)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    started = time.time()
    value = fn()
    payload.setdefault("timing_s", {})[str(step)] = float(time.time() - started)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return value


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    left = reference.detach().cpu().to(dtype=torch.float32)
    right = actual.detach().cpu().to(dtype=torch.float32)
    diff = right - left
    return {
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.pow(2))).item()),
    }


def _collect_region_audit(net: torch.nn.Module) -> dict[str, Any]:
    rows = []
    for name, module in net.named_modules():
        runtime = getattr(module, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        if runtime is None or executor is None:
            continue
        rows.append(
            {
                "node": str(getattr(module, "region_output_id", name)),
                "compile_count": int(getattr(executor, "compile_count", 0)),
                "execute_count": int(getattr(runtime, "execute_count", 0)),
                "last_runtime_timing": dict(getattr(executor, "last_runtime_timing", {})),
                "lazy_region_compile": bool(getattr(module, "region_first_probe_lazy_region_compile", False)),
            }
        )
    return {
        "selected_region_count": int(len(rows)),
        "rows": rows,
        "all_precompiled": bool(rows) and all(int(row["compile_count"]) >= 1 for row in rows),
    }


def _run_one(
    *,
    mode: str,
    out_path: Path,
    config_path: Path,
    seed: int,
    profile_he_breakdown: bool = False,
) -> dict[str, Any]:
    mode = str(mode)
    config = _load_config(Path(config_path), mode=mode)
    payload: dict[str, Any] = {
        "status": "started",
        "step": "init",
        "network": "R18",
        "dataset": "tiny",
        "mode": mode,
        "config_path": str(Path(config_path)),
        "config": config,
        "seed": int(seed),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    he_breakdown_collector: HeForwardBreakdownCollector | None = None

    try:
        torch.manual_seed(int(seed))
        net = ResNet18(dataset="tiny")
        net.eval()
        x0 = torch.randn((1, 3, 64, 64), dtype=torch.float32)
        clear = _timed(payload, out_path, "clear_forward", lambda: net(x0))
        payload["clear"] = {
            "shape": list(clear.shape),
            "checksum": float(clear.detach().sum().item()),
            "l2": float(torch.linalg.vector_norm(clear.detach()).item()),
            "values": [float(v) for v in clear.detach().cpu().reshape(-1).tolist()],
        }
        Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        _timed(payload, out_path, "init_scheme", lambda: scheme.init_scheme(config))
        _timed(payload, out_path, "fit", lambda: scheme.fit(net, x0))
        input_level = _timed(payload, out_path, "compile", lambda: scheme.compile(net))
        payload["input_level"] = int(input_level)
        payload["attach_audit"] = getattr(scheme, "region_first_attach_audit", {})
        payload["region_audit_after_compile"] = _collect_region_audit(net)
        Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        net.he()
        if bool(profile_he_breakdown):
            he_breakdown_collector = HeForwardBreakdownCollector(net)
            he_breakdown_collector.install()
        x0_ct = _timed(payload, out_path, "encrypt", lambda: scheme.encrypt(scheme.encode(x0, int(input_level))))
        out_ct = _timed(payload, out_path, "he_forward", lambda: net(x0_ct))
        if he_breakdown_collector is not None:
            he_breakdown_collector.remove()
            payload["he_forward_breakdown"] = he_breakdown_collector.summary(
                float(payload.get("timing_s", {}).get("he_forward", 0.0))
            )
            Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        decoded = _timed(payload, out_path, "decrypt_decode", lambda: out_ct.decrypt().decode())
        decoded = decoded.detach().cpu().to(dtype=torch.float32)
        payload["input_ciphertext_count"] = int(len(x0_ct.ids))
        payload["output_ciphertext_count"] = int(len(out_ct.ids))
        payload["decoded"] = {
            "shape": list(decoded.shape),
            "checksum": float(decoded.sum().item()),
            "l2": float(torch.linalg.vector_norm(decoded).item()),
            "values": [float(v) for v in decoded.reshape(-1).tolist()],
        }
        payload["mae_vs_clear"] = _metrics(clear, decoded)
        payload["region_audit_after_forward"] = _collect_region_audit(net)
        payload["status"] = "ok"
        payload["step"] = "done"
        Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc(limit=120)
        Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if he_breakdown_collector is not None:
            he_breakdown_collector.remove()
        try:
            scheme.delete_scheme()
        except Exception:
            pass


def _summarize(*, dense_path: Path, provider_path: Path, out_path: Path) -> dict[str, Any]:
    dense = json.loads(Path(dense_path).read_text())
    provider = json.loads(Path(provider_path).read_text())
    summary: dict[str, Any] = {
        "status": "ok" if dense.get("status") == "ok" and provider.get("status") == "ok" else "partial",
        "network": "R18",
        "dense_path": str(Path(dense_path)),
        "provider_path": str(Path(provider_path)),
        "dense": {
            "status": dense.get("status"),
            "timing_s": dense.get("timing_s", {}),
            "mae_vs_clear": dense.get("mae_vs_clear"),
            "input_level": dense.get("input_level"),
            "output_ciphertext_count": dense.get("output_ciphertext_count"),
            "he_forward_breakdown": dense.get("he_forward_breakdown"),
        },
        "provider": {
            "status": provider.get("status"),
            "timing_s": provider.get("timing_s", {}),
            "mae_vs_clear": provider.get("mae_vs_clear"),
            "input_level": provider.get("input_level"),
            "output_ciphertext_count": provider.get("output_ciphertext_count"),
            "attach_audit": provider.get("attach_audit", {}),
            "he_forward_breakdown": provider.get("he_forward_breakdown"),
        },
    }
    if dense.get("status") == "ok" and provider.get("status") == "ok":
        dense_values = torch.tensor(dense["decoded"]["values"], dtype=torch.float32)
        provider_values = torch.tensor(provider["decoded"]["values"], dtype=torch.float32)
        clear_dense = torch.tensor(dense["clear"]["values"], dtype=torch.float32)
        clear_provider = torch.tensor(provider["clear"]["values"], dtype=torch.float32)
        summary["clear_consistency"] = _metrics(clear_dense, clear_provider)
        summary["provider_vs_dense_decoded"] = _metrics(dense_values, provider_values)
        dense_runtime = float(dense.get("timing_s", {}).get("he_forward", math.nan))
        provider_runtime = float(provider.get("timing_s", {}).get("he_forward", math.nan))
        dense_compile = float(dense.get("timing_s", {}).get("compile", math.nan))
        provider_compile = float(provider.get("timing_s", {}).get("compile", math.nan))
        summary["ratios"] = {
            "he_forward_dense_over_provider": float(dense_runtime / provider_runtime) if provider_runtime and math.isfinite(provider_runtime) else None,
            "compile_dense_over_provider": float(dense_compile / provider_compile) if provider_compile and math.isfinite(provider_compile) else None,
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or summarize R18 Lattigo E2E dense/provider comparison.")
    parser.add_argument("--mode", choices=("dense", "provider", "summarize"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dense-path", type=Path, default=Path("/tmp/orion_r18_e2e_dense.json"))
    parser.add_argument("--provider-path", type=Path, default=Path("/tmp/orion_r18_e2e_provider.json"))
    parser.add_argument("--profile-he-breakdown", action="store_true")
    args = parser.parse_args()
    if str(args.mode) == "summarize":
        _summarize(dense_path=Path(args.dense_path), provider_path=Path(args.provider_path), out_path=Path(args.out))
        return 0
    _run_one(
        mode=str(args.mode),
        out_path=Path(args.out),
        config_path=Path(args.config),
        seed=int(args.seed),
        profile_he_breakdown=bool(args.profile_he_breakdown),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
