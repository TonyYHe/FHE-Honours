from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from orion.core.orion import scheme
from orion.models.resnet import ResNet18


DEFAULT_OUT = Path("/run/media/anakano/7TB/haloed-cache/orion_runtime_only/orion_r18_single_input_silu7.json")


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
    parser = argparse.ArgumentParser(description="Single-input R18 TinyImageNet SiLU7 compile/run harness.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experimental-region-first", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = str(args.experimental_region_first or "")

    payload = {
        "status": "started",
        "step": "init",
        "mode": "single_fit_input",
        "config": _build_config(mode),
        "activation": {
            "kind": "silu",
            "silu_degree": 7,
            "stem_relu": True,
            "expected_bootstraps_reference": 61,
        },
        "region_first_mode": mode,
        "claim": {
            "publishable": False,
            "reason": "manual single-input compile/run harness",
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    try:
        torch.manual_seed(int(args.seed))
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
        x0_ct = _timed(payload, out_path, "encrypt_fit_input", lambda: scheme.encrypt(scheme.encode(x0, input_level)))
        out_ct = _timed(payload, out_path, "he_forward_fit_input", lambda: net(x0_ct))

        payload["input_ciphertext_count"] = int(len(x0_ct.ids))
        payload["output_ciphertext_count"] = int(len(out_ct.ids))
        payload["runtime_contract_audit_after_forward"] = _collect_region_compile_audit(net)
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
