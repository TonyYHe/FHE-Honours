from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import torch

from orion.core.orion import scheme
from orion.models.resnet import ResNet18


DEFAULT_OUT = Path("/run/media/anakano/7TB/haloed-cache/orion_runtime_only/orion_r18_runtime_only_silu7.json")


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
            # Runtime-only metrics must not use lazy region compile.
            "experimental_region_first": "r18_tiny_e2e_probe_precompile",
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


def _collect_region_compile_audit(net: torch.nn.Module) -> dict:
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
                "last_runtime_timing": dict(getattr(executor, "last_runtime_timing", {})),
                "lazy_region_compile": bool(getattr(module, "region_first_probe_lazy_region_compile", False)),
            }
        )
    return {
        "selected_region_count": int(len(rows)),
        "rows": rows,
        "all_precompiled": bool(rows) and all(int(row["compile_count"]) >= 1 for row in rows),
        "all_non_lazy": all(bool(row["lazy_region_compile"]) is False for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile-once runtime-only R18 Tiny SiLU7 probe.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--warmup-seed", type=int, default=1)
    parser.add_argument("--measured-seeds", type=int, nargs="*", default=[2, 3])
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "started",
        "step": "init",
        "mode": "runtime_only_after_precompile",
        "config": _build_config(),
        "activation": {
            "kind": "silu",
            "silu_degree": 7,
            "stem_relu": True,
            "expected_bootstraps_reference": 61,
            "runtime_metric_policy": "compile_once_then_measure_he_forward_only",
        },
        "warmup": {},
        "measured_runs": [],
        "claim": {
            "publishable": False,
            "reason": "probe-only dense fallback and stem ReLU bypass remain active; runtime-only metric excludes compile time by design",
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    try:
        torch.manual_seed(0)
        net = ResNet18(dataset="tiny", activation="silu", silu_degree=7, stem_relu=True)
        net.eval()

        x0 = torch.randn((1, 3, 64, 64), dtype=torch.float32)
        clear = _timed(payload, out_path, "clear_forward", lambda: net(x0))
        payload["clear_output_shape"] = list(clear.shape)
        payload["clear_checksum"] = float(clear.detach().sum())
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        _timed(payload, out_path, "init_scheme", lambda: scheme.init_scheme(payload["config"]))
        _timed(payload, out_path, "fit", lambda: scheme.fit(net, x0))
        input_level = _timed(payload, out_path, "compile", lambda: scheme.compile(net))
        payload["input_level"] = int(input_level)
        payload["attach_audit"] = getattr(scheme, "region_first_attach_audit", {})
        payload["runtime_contract_audit_after_compile"] = _collect_region_compile_audit(net)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        net.he()

        torch.manual_seed(int(args.warmup_seed))
        x_warm = torch.randn((1, 3, 64, 64), dtype=torch.float32)
        x_warm_ct = scheme.encrypt(scheme.encode(x_warm, input_level))
        payload["step"] = "warmup_he_forward"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        started = time.time()
        warm_out = net(x_warm_ct)
        payload["warmup"] = {
            "seed": int(args.warmup_seed),
            "he_forward_s": float(time.time() - started),
            "output_ciphertext_count": int(len(warm_out.ids)),
        }
        payload["runtime_contract_audit_after_warmup"] = _collect_region_compile_audit(net)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        del warm_out, x_warm_ct
        gc.collect()

        for index, seed in enumerate(args.measured_seeds, start=1):
            torch.manual_seed(int(seed))
            x = torch.randn((1, 3, 64, 64), dtype=torch.float32)
            x_ct = scheme.encrypt(scheme.encode(x, input_level))
            payload["step"] = f"measured_he_forward_{index}"
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            started = time.time()
            out = net(x_ct)
            elapsed = float(time.time() - started)
            payload["measured_runs"].append(
                {
                    "index": int(index),
                    "seed": int(seed),
                    "he_forward_s": elapsed,
                    "output_ciphertext_count": int(len(out.ids)),
                    "runtime_contract_audit": _collect_region_compile_audit(net),
                }
            )
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            del out, x_ct
            gc.collect()

        if payload["measured_runs"]:
            values = [float(item["he_forward_s"]) for item in payload["measured_runs"]]
            payload["runtime_only_summary"] = {
                "run_count": int(len(values)),
                "avg_he_forward_s": float(sum(values) / len(values)),
                "min_he_forward_s": float(min(values)),
                "max_he_forward_s": float(max(values)),
            }

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
