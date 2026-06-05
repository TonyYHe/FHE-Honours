#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import orion.nn as on


DEFAULT_RUN_ROOT_BASE = REPO_ROOT / ".tmp" / "results"
LATEST_POINTER = REPO_ROOT / ".tmp" / "latest_u22_dim32_provider_mini_192.txt"
NETWORK_NAME = "u22_dim32_provider_mini_192"
PROVIDER_MODE = "u22_256_base32_layout_dp"
CPU_COUNT = max(1, int(os.cpu_count() or 1))


def _activation_name(value: str | None) -> str:
    normalized = str(value or "none").strip().lower()
    if normalized in {"", "none", "identity", "linear"}:
        return "none"
    if normalized in {"silu", "swish"}:
        return "silu"
    if normalized == "relu":
        return "relu"
    raise ValueError(f"unsupported activation {value!r}")


def _make_activation(value: str | None, *, silu_degree: int):
    normalized = _activation_name(value)
    if normalized == "none":
        return None
    if normalized == "silu":
        return on.SiLU(degree=int(silu_degree))
    if normalized == "relu":
        return on.ReLU()
    raise AssertionError(f"unhandled activation {normalized!r}")


def _apply_activation(module, x):
    return x if module is None else module(x)


class U22ProviderMini192(on.Module):
    """One-skip U22 shard for the 192x192 provider preflight."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 32,
        activation: str | None = "silu",
        silu_degree: int = 7,
    ) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = int(base_channels) * 2
        make_act = lambda: _make_activation(activation, silu_degree=int(silu_degree))

        self.enc1a = on.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=True)
        self.enc1a_act = make_act()
        self.enc1b = on.Conv2d(c1, c1, kernel_size=3, padding=1, bias=True)
        self.enc1b_act = make_act()
        self.pool1 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc2a = on.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True)
        self.enc2a_act = make_act()
        self.enc2b = on.Conv2d(c2, c2, kernel_size=3, padding=1, bias=True)
        self.enc2b_act = make_act()

        self.up1 = on.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=True)
        self.cat1 = on.Concat(dim=1)
        self.dec1a = on.Conv2d(c1 + c1, c1, kernel_size=3, padding=1, bias=True)
        self.dec1a_act = make_act()
        self.output = on.Conv2d(c1, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = _apply_activation(self.enc1a_act, self.enc1a(x))
        skip1 = _apply_activation(self.enc1b_act, self.enc1b(x))
        x = _apply_activation(self.enc2a_act, self.enc2a(self.pool1(skip1)))
        x = _apply_activation(self.enc2b_act, self.enc2b(x))
        x = _apply_activation(self.dec1a_act, self.dec1a(self.cat1(self.up1(x), skip1)))
        return self.output(x)


def _resource_maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return int(value * 1024)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _apply_env_defaults(*, workers: int) -> dict[str, str]:
    workers = max(1, int(workers))
    defaults = {
        "PYTHONUNBUFFERED": "1",
        "MALLOC_ARENA_MAX": "2",
        "GOMAXPROCS": "1",
        "ORION_COMPILE_PARALLEL_POLICY": "manual",
        "ORION_SINGLE_SLOT_LAYER_CACHE": "1",
        "ORION_SINGLE_SLOT_ENCODE_WORKERS": str(workers),
        "ORION_LATTIGO_STREAMING_LT": "0",
        "ORION_UNIFIED_STREAM_COMPILE_IO_NONE": "0",
        "ORION_LATTIGO_MEMORY_BOUNDED_COMPILE": "0",
        "ORION_LATTIGO_MEMORY_BOUNDED_EVAL": "0",
        "ORION_LATTIGO_BOOTSTRAP_MANY": "0",
        "ORION_PACK_CONV_WORKERS": str(workers),
        "ORION_DIRECT_PACK_WORKERS": str(workers),
        "ORION_LT_COMPILE_WORKERS": str(workers),
        "ORION_UNIFIED_COMPILE_WORKERS": str(workers),
        "ORION_LATTIGO_COMPILE_WORKERS": str(workers),
        "ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS": str(workers),
        "ORION_UNIFIED_LT_OUTPUT_FUSION": "1",
        "ORION_LAYOUT_POLICY_RELAYOUT_KERNEL": "1",
        "ORION_LAYOUT_POLICY_PROVIDER_NATIVE_HALO": "1",
        "ORION_UNIFIED_LT_SHARED_ROTATION_KEYS": "1",
        "ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE": "1",
        "ORION_REGION_FIRST_CLEANUP_AFTER_OUTPUTS": "1",
        "ORION_CONCAT_FUSION": "0",
        "ORION_BOOTSTRAP_LAYOUT_REFINEMENT": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(str(key), str(value))
    os.environ["GOMAXPROCS"] = "1"
    return {key: str(os.environ.get(key, "")) for key in sorted(defaults)}


def _register_network() -> Any:
    from tools import run_lattigo_e2e_compare as base

    def _builder(*, activation: str | None = None, silu_degree: int = 7):
        return U22ProviderMini192(
            in_channels=1,
            out_channels=4,
            base_channels=32,
            activation=_activation_name(activation),
            silu_degree=int(silu_degree),
        )

    base.NETWORKS[NETWORK_NAME] = {
        "label": "U22 dim32 192 provider mini",
        "model": "U22ProviderMini192",
        "dataset": "IBSR BRAIN 2D mini",
        "input_shape": (1, 1, 192, 192),
        "provider_mode": PROVIDER_MODE,
        "config": base._r18_config,
        "builder": _builder,
        "scope": "mini_up_down_skip",
        "base_dim": 32,
        "out_channels": 4,
    }
    return base


def _annotate(
    out_path: Path,
    *,
    run_root: Path,
    mode: str,
    backend: str,
    started_at: float,
    env_snapshot: dict[str, str],
) -> None:
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    payload["runner"] = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "run_root": str(run_root),
        "mode": str(mode),
        "backend": str(backend),
        "provider_mode": PROVIDER_MODE if str(mode) == "provider" else "",
        "local_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": float(time.perf_counter() - started_at),
        "maxrss_bytes": int(_resource_maxrss_bytes()),
        "env": dict(env_snapshot),
    }
    payload["model_variant"] = {
        "name": "U22ProviderMini192",
        "purpose": "192x192 provider preflight for pool/down, up1, skip concat, dec1a, and output",
        "linear_layers": 7,
        "base_dim": 32,
        "input_shape": [1, 1, 192, 192],
        "out_channels": 4,
        "provider_mode": PROVIDER_MODE,
        "output_head": "1x1/pad0",
    }
    _write_json(out_path, payload)


def run_one(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out is not None else run_root / f"{args.mode}_mini_e2e.json"
    env_snapshot = _apply_env_defaults(workers=int(args.workers))
    base = _register_network()
    started_at = time.perf_counter()
    try:
        base._run_one(
            network=NETWORK_NAME,
            backend=str(args.backend),
            mode=str(args.mode),
            out_path=out_path,
            seed=int(args.seed),
            compile_only=bool(args.compile_only),
            forward_runs=int(args.forward_runs),
            warmup_runs=int(args.warmup_runs),
            profile_modules=False,
            profile_lt=bool(args.profile_lt),
            trace_forward_memory=bool(args.trace_forward_memory),
            operator_breakdown=bool(args.operator_breakdown),
            provider_mode_override=PROVIDER_MODE if str(args.mode) == "provider" else None,
            io_mode="none",
            io_dir=None,
            diags_path=None,
            keys_path=None,
            logn_override=None,
            activation=_activation_name(str(args.activation)),
            silu_degree=int(args.silu_degree),
            ckks_preset=str(args.ckks_preset),
        )
        return 0
    finally:
        _annotate(
            out_path,
            run_root=run_root,
            mode=str(args.mode),
            backend=str(args.backend),
            started_at=started_at,
            env_snapshot=env_snapshot,
        )
        LATEST_POINTER.parent.mkdir(parents=True, exist_ok=True)
        LATEST_POINTER.write_text(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "result": str(out_path),
                    "mode": str(args.mode),
                    "backend": str(args.backend),
                    "provider_mode": PROVIDER_MODE if str(args.mode) == "provider" else "",
                    "activation": _activation_name(str(args.activation)),
                    "silu_degree": int(args.silu_degree),
                    "local_time": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    parser = argparse.ArgumentParser(description="Run a 192x192 dim32 provider mini U-Net preflight.")
    parser.add_argument("--backend", choices=("python", "lattigo"), default="lattigo")
    parser.add_argument("--mode", choices=("provider", "dense"), default="provider")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT_BASE / f"u22_dim32_provider_mini_192_{timestamp}")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--forward-runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--activation", choices=("none", "relu", "silu"), default="silu")
    parser.add_argument("--silu-degree", type=int, default=7)
    parser.add_argument("--workers", type=int, default=CPU_COUNT)
    parser.add_argument("--profile-lt", action="store_true")
    parser.add_argument("--trace-forward-memory", action="store_true")
    parser.add_argument("--operator-breakdown", action="store_true", default=True)
    parser.add_argument("--no-operator-breakdown", dest="operator_breakdown", action="store_false")
    parser.add_argument("--ckks-preset", choices=("network-default", "resnet"), default="resnet")
    return run_one(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
