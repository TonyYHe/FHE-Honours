from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.backend.python.tensors import CipherTensor
from orion.backend.python.compile_policy import auto_worker_count, policy_audit
from orion.core.fuser import Fuser
from orion.core.network_dag import NetworkDAG
from orion.core.orion import _region_first_mode_options, scheme
from orion.core.tracer import OrionTracer, StatsTracker
from orion.experimental import R34CompileRegistry, U22CompileRegistry
from orion.experimental.cir.runtime_group import (
    RegionFirstCompileRegistry,
)
from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
from orion.experimental.cir.transition_pool_provider import InputPairConvRuntimeExecutor
from orion.models.resnet import ResNet18, ResNet34
from orion.models.unet import UNet22
from orion.nn.linear import Conv2d, ConvTranspose2d
from orion.nn.module import Module
from orion.nn.unified_transform import UnifiedTransformGroup


DEFAULT_OUT = Path("/tmp/orion_node_specific_lattigo_cheddar_provider_vs_dense.json")
DEFAULT_CSV_OUT = Path("/tmp/orion_node_specific_lattigo_cheddar_provider_vs_dense.csv")
BACKENDS = ("lattigo", "cheddar")
PATHS = (
    "dense",
    "provider",
    "provider_no_family",
    "provider_no_tile_family_sharing",
)
PROVIDER_PATHS = frozenset(path for path in PATHS if str(path) != "dense")
NO_FAMILY_PATHS = frozenset(("provider_no_family",))
LOGN_OVERRIDE_ENV = "ORION_NODE_BENCH_LOGN_OVERRIDE"
CKKS_PROFILE_ENV = "ORION_NODE_BENCH_CKKS_PROFILE"
CLEAN_CHEDDAR_IO_ENV = "ORION_NODE_BENCH_CLEAN_CHEDDAR_IO"
MIN_DISK_FREE_GB_ENV = "ORION_NODE_BENCH_MIN_DISK_FREE_GB"
DISK_WATCHDOG_INTERVAL_S_ENV = "ORION_NODE_BENCH_DISK_WATCHDOG_INTERVAL_S"
BOUNDED_LATTIGO_DENSE_ENV = "ORION_NODE_BENCH_BOUNDED_LATTIGO_DENSE"
BOUNDED_LATTIGO_DENSE_MODE_ENV = "ORION_NODE_BENCH_BOUNDED_LATTIGO_DENSE_MODE"
BOUNDED_LATTIGO_PROVIDER_ENV = "ORION_NODE_BENCH_BOUNDED_LATTIGO_PROVIDER"
BOUNDED_LATTIGO_PROVIDER_MODE_ENV = "ORION_NODE_BENCH_BOUNDED_LATTIGO_PROVIDER_MODE"
CLEAN_LATTIGO_DENSE_IO_ENV = "ORION_NODE_BENCH_CLEAN_LATTIGO_DENSE_IO"
CLEAN_LATTIGO_PROVIDER_IO_ENV = "ORION_NODE_BENCH_CLEAN_LATTIGO_PROVIDER_IO"
LATTIGO_DENSE_IO_ROOT_ENV = "ORION_LATTIGO_DENSE_IO_ROOT"
LATTIGO_PROVIDER_IO_ROOT_ENV = "ORION_LATTIGO_PROVIDER_IO_ROOT"
LATTIGO_BENCH_IO_MODE_ENV = "ORION_NODE_BENCH_LATTIGO_IO_MODE"
LATTIGO_BENCH_DIAGS_PATH_ENV = "ORION_NODE_BENCH_LATTIGO_DIAGS_PATH"
LATTIGO_BENCH_KEYS_PATH_ENV = "ORION_NODE_BENCH_LATTIGO_KEYS_PATH"
MAX_WORKER_RSS_GB_ENV = "ORION_NODE_BENCH_MAX_WORKER_RSS_GB"
BOUNDED_LATTIGO_DENSE_STREAMING_CHUNK_PLAINTEXTS = "256"
BOUNDED_LATTIGO_PROVIDER_STREAMING_CHUNK_PLAINTEXTS = "512"
_ACTIVE_CHEDDAR_BENCH_IO_DIR: Path | None = None


def _bounded_lattigo_compile_workers_default() -> str:
    cpu_count = max(1, int(os.cpu_count() or 1))
    return str(
        auto_worker_count(
            cpu_count,
            (
                "ORION_LT_COMPILE_WORKERS",
                "ORION_UNIFIED_COMPILE_WORKERS",
                "ORION_LATTIGO_COMPILE_WORKERS",
            ),
            default_workers=4,
            estimated_per_worker_bytes=24 * 1024**3,
            cpu_count=cpu_count,
        )
    )


def _bounded_lattigo_diagonal_encode_workers_default() -> str:
    cpu_count = max(1, int(os.cpu_count() or 1))
    return str(
        auto_worker_count(
            cpu_count,
            ("ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS",),
            default_workers=cpu_count,
            estimated_per_worker_bytes=4 * 1024**3,
            cpu_count=cpu_count,
        )
    )


def _bounded_lattigo_pack_workers_default() -> str:
    cpu_count = max(1, int(os.cpu_count() or 1))
    return str(
        auto_worker_count(
            cpu_count,
            ("ORION_PACK_CONV_WORKERS",),
            default_workers=8,
            estimated_per_worker_bytes=8 * 1024**3,
            cpu_count=cpu_count,
        )
    )


RESNET_LOGQ = (55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40)
RESNET_LOGP = (61, 61, 61)
RESNET_BOOT_LOGP = (61, 61, 61, 61, 61, 61, 61, 61)
RESNET_LOGSCALE = 40
RESNET_SECRET_H = 192

PATH_DESCRIPTIONS = {
    "dense": "baseline Orion dense LinearTransform path",
    "provider": "optimized provider path with backend UnifiedTransformGroup and any hybrid packing enabled",
    "provider_no_family": (
        "provider ablation with BSGS preserved but UnifiedTransformGroup family sharing disabled; "
        "each compiled LinearTransform is evaluated one by one"
    ),
    "provider_no_tile_family_sharing": (
        "U22 TConv provider ablation with halo layout, real/imag packing, and BSGS preserved, "
        "but each output tile runs as an independent UnifiedTransformGroup"
    ),
}

R18_TINY_CASES: tuple[dict[str, Any], ...] = (
    {"case": "stem_conv", "node": "conv1", "op": "conv2d", "stage": "stem", "seed": 11, "multiplicity": 1},
    {"case": "stage1_same", "node": "layers_0_0_conv1", "op": "conv2d", "stage": "stage1", "seed": 101, "multiplicity": 4},
    {"case": "stage2_same", "node": "layers_1_1_conv1", "op": "conv2d", "stage": "stage2", "seed": 202, "multiplicity": 3},
    {"case": "stage3_same", "node": "layers_2_1_conv1", "op": "conv2d", "stage": "stage3", "seed": 303, "multiplicity": 3},
    {"case": "stage4_same", "node": "layers_3_0_conv2", "op": "conv2d", "stage": "stage4", "seed": 404, "multiplicity": 3},
)

R34_IMGNET_CASES: tuple[dict[str, Any], ...] = (
    {"case": "stem_conv", "node": "conv1", "op": "conv2d", "stage": "stem", "seed": 21, "multiplicity": 1},
    {"case": "stage1_same", "node": "layers_0_0_conv1", "op": "conv2d", "stage": "stage1", "seed": 111, "multiplicity": 6},
    {"case": "stage2_same", "node": "layers_1_1_conv1", "op": "conv2d", "stage": "stage2", "seed": 222, "multiplicity": 7},
    {"case": "stage3_same", "node": "layers_2_1_conv1", "op": "conv2d", "stage": "stage3", "seed": 333, "multiplicity": 11},
    {"case": "stage4_same", "node": "layers_3_1_conv1", "op": "conv2d", "stage": "stage4", "seed": 444, "multiplicity": 5},
)

U22_TCONV_CASES: tuple[dict[str, Any], ...] = (
    {"case": "up4", "node": "up4", "op": "conv_transpose2d", "stage": "decoder", "seed": 1004, "multiplicity": 1},
    {"case": "up3", "node": "up3", "op": "conv_transpose2d", "stage": "decoder", "seed": 1003, "multiplicity": 1},
    {"case": "up2", "node": "up2", "op": "conv_transpose2d", "stage": "decoder", "seed": 1002, "multiplicity": 1},
    {"case": "up1", "node": "up1", "op": "conv_transpose2d", "stage": "decoder", "seed": 1001, "multiplicity": 1},
)

U22_ENCODER_CONV_CASES: tuple[dict[str, Any], ...] = (
    {"case": "enc1a", "node": "enc1a", "op": "conv2d", "stage": "encoder", "seed": 1101, "multiplicity": 1},
    {"case": "enc1b", "node": "enc1b", "op": "conv2d", "stage": "encoder", "seed": 1102, "multiplicity": 1},
    {"case": "enc2a", "node": "enc2a", "op": "conv2d", "stage": "encoder", "seed": 1201, "multiplicity": 1},
    {"case": "enc2b", "node": "enc2b", "op": "conv2d", "stage": "encoder", "seed": 1202, "multiplicity": 1},
    {"case": "enc3a", "node": "enc3a", "op": "conv2d", "stage": "encoder", "seed": 1301, "multiplicity": 1},
    {"case": "enc3b", "node": "enc3b", "op": "conv2d", "stage": "encoder", "seed": 1302, "multiplicity": 1},
    {"case": "enc4a", "node": "enc4a", "op": "conv2d", "stage": "encoder", "seed": 1401, "multiplicity": 1},
    {"case": "enc4b", "node": "enc4b", "op": "conv2d", "stage": "encoder", "seed": 1402, "multiplicity": 1},
    {
        "case": "bottlenecka",
        "node": "bottlenecka",
        "op": "conv2d",
        "stage": "encoder",
        "seed": 1501,
        "multiplicity": 1,
    },
    {
        "case": "bottleneckb",
        "node": "bottleneckb",
        "op": "conv2d",
        "stage": "encoder",
        "seed": 1502,
        "multiplicity": 1,
    },
)

NETWORK_SPECS: dict[str, dict[str, Any]] = {
    "r18_tiny": {
        "label": "R18 Tiny",
        "model": "ResNet18",
        "dataset": "tiny",
        "input_shape": (1, 3, 64, 64),
        "logn": 16,
        "logq": (45, 30, 30, 45),
        "logp": (50,),
        "cases": R18_TINY_CASES,
        "coverage_note": "unique optimized Conv2d representatives; transition/downsample convs excluded",
    },
    "r34_imgnet": {
        "label": "R34 Imgnet",
        "model": "ResNet34",
        "dataset": "imagenet",
        "input_shape": (1, 3, 224, 224),
        "logn": 16,
        "logq": (45, 30, 30, 45),
        "logp": (50,),
        "cases": R34_IMGNET_CASES,
        "coverage_note": "unique optimized Conv2d representatives; transition/downsample convs, pools, and avgpool exit excluded",
    },
    "u22_64_base32": {
        "label": "U-Net 22 base_dim=32 64x64",
        "model": "UNet22",
        "dataset": "montgomery_lung_64",
        "base_dim": 32,
        "input_shape": (1, 1, 64, 64),
        "logn": 16,
        "logq": RESNET_LOGQ,
        "logp": RESNET_LOGP,
        "cases": U22_TCONV_CASES,
        "coverage_note": (
            "node-specific rows cover unique ConvTranspose2d decoder representatives; provider DAG attach uses "
            "the full default U22 provider mode, including Conv2d/Pool halo layout-policy kernels"
        ),
    },
    "u22_256_base32": {
        "label": "U-Net 22 base_dim=32 256x256",
        "model": "UNet22",
        "dataset": "kvasir_polyp_256",
        "base_dim": 32,
        "input_shape": (1, 3, 256, 256),
        "logn": 16,
        "logq": RESNET_LOGQ,
        "logp": RESNET_LOGP,
        "cases": U22_TCONV_CASES,
        "coverage_note": (
            "node-specific rows cover unique ConvTranspose2d decoder representatives; provider DAG attach uses "
            "the full default U22 provider mode, including Conv2d/Pool halo layout-policy kernels"
        ),
    },
}

NETWORK_SPECS.update(
    {
        "u22_192_base32_encoder": {
            "label": "U-Net 22 base_dim=32 encoder 192x192",
            "model": "UNet22",
            "dataset": "kvasir_polyp_256",
            "base_dim": 32,
            "input_shape": (1, 3, 192, 192),
            "logn": 16,
            "logq": RESNET_LOGQ,
            "logp": RESNET_LOGP,
            "cases": U22_ENCODER_CONV_CASES,
            "coverage_note": "unique Conv2d encoder representatives only; provider attach enables Conv kernels for these nodes",
        },
        "u22_224_base32_encoder": {
            "label": "U-Net 22 base_dim=32 encoder 224x224",
            "model": "UNet22",
            "dataset": "kvasir_polyp_256",
            "base_dim": 32,
            "input_shape": (1, 3, 224, 224),
            "logn": 16,
            "logq": RESNET_LOGQ,
            "logp": RESNET_LOGP,
            "cases": U22_ENCODER_CONV_CASES,
            "coverage_note": "unique Conv2d encoder representatives only; provider attach enables Conv kernels for these nodes",
        },
        "u22_384x288_base32_encoder": {
            "label": "U-Net 22 base_dim=32 encoder 384x288",
            "model": "UNet22",
            "dataset": "kvasir_polyp_256",
            "base_dim": 32,
            "input_shape": (1, 3, 384, 288),
            "logn": 16,
            "logq": RESNET_LOGQ,
            "logp": RESNET_LOGP,
            "cases": U22_ENCODER_CONV_CASES,
            "coverage_note": "unique Conv2d encoder representatives only; provider attach enables Conv kernels for these nodes",
        },
        "u22_384_base32_encoder": {
            "label": "U-Net 22 base_dim=32 encoder 384x384",
            "model": "UNet22",
            "dataset": "kvasir_polyp_256",
            "base_dim": 32,
            "input_shape": (1, 3, 384, 384),
            "logn": 16,
            "logq": RESNET_LOGQ,
            "logp": RESNET_LOGP,
            "cases": U22_ENCODER_CONV_CASES,
            "coverage_note": "unique Conv2d encoder representatives only; provider attach enables Conv kernels for these nodes",
        },
    }
)


E2E_CKKS_SPECS: dict[str, dict[str, Any]] = {
    "r18_tiny": {
        "logn": 16,
        "logq": RESNET_LOGQ,
        "logp": RESNET_LOGP,
        "logscale": RESNET_LOGSCALE,
        "h": RESNET_SECRET_H,
        "boot_logp": RESNET_BOOT_LOGP,
    },
    "r34_imgnet": {
        "logn": 16,
        "logq": RESNET_LOGQ,
        "logp": RESNET_LOGP,
        "logscale": RESNET_LOGSCALE,
        "h": RESNET_SECRET_H,
        "boot_logp": RESNET_BOOT_LOGP,
    },
    "u22_64_base32": {
        "logn": 16,
        "logq": RESNET_LOGQ,
        "logp": RESNET_LOGP,
        "logscale": RESNET_LOGSCALE,
        "h": RESNET_SECRET_H,
        "boot_logp": RESNET_BOOT_LOGP,
    },
    "u22_256_base32": {
        "logn": 16,
        "logq": RESNET_LOGQ,
        "logp": RESNET_LOGP,
        "logscale": RESNET_LOGSCALE,
        "h": RESNET_SECRET_H,
        "boot_logp": RESNET_BOOT_LOGP,
    },
}

for _u22_dim32_encoder_network in (
    "u22_192_base32_encoder",
    "u22_224_base32_encoder",
    "u22_384x288_base32_encoder",
    "u22_384_base32_encoder",
):
    E2E_CKKS_SPECS[_u22_dim32_encoder_network] = dict(E2E_CKKS_SPECS["u22_256_base32"])

NETWORK_SPECS.update(
    {
        "u22_192_base64_encoder": {
            "label": "U-Net 22 base_dim=64 encoder 192x192",
            "model": "UNet22",
            "dataset": "kvasir_polyp_256",
            "base_dim": 64,
            "input_shape": (1, 3, 192, 192),
            "logn": 16,
            "logq": RESNET_LOGQ,
            "logp": RESNET_LOGP,
            "cases": U22_ENCODER_CONV_CASES,
            "coverage_note": "unique Conv2d encoder representatives only; provider attach enables Conv kernels for these nodes",
        },
        "u22_224_base64_encoder": {
            "label": "U-Net 22 base_dim=64 encoder 224x224",
            "model": "UNet22",
            "dataset": "kvasir_polyp_256",
            "base_dim": 64,
            "input_shape": (1, 3, 224, 224),
            "logn": 16,
            "logq": RESNET_LOGQ,
            "logp": RESNET_LOGP,
            "cases": U22_ENCODER_CONV_CASES,
            "coverage_note": "unique Conv2d encoder representatives only; provider attach enables Conv kernels for these nodes",
        },
        "u22_384x288_base64_encoder": {
            "label": "U-Net 22 base_dim=64 encoder 384x288",
            "model": "UNet22",
            "dataset": "kvasir_polyp_256",
            "base_dim": 64,
            "input_shape": (1, 3, 384, 288),
            "logn": 16,
            "logq": RESNET_LOGQ,
            "logp": RESNET_LOGP,
            "cases": U22_ENCODER_CONV_CASES,
            "coverage_note": "unique Conv2d encoder representatives only; provider attach enables Conv kernels for these nodes",
        },
        "u22_384_base64_encoder": {
            "label": "U-Net 22 base_dim=64 encoder 384x384",
            "model": "UNet22",
            "dataset": "kvasir_polyp_256",
            "base_dim": 64,
            "input_shape": (1, 3, 384, 384),
            "logn": 16,
            "logq": RESNET_LOGQ,
            "logp": RESNET_LOGP,
            "cases": U22_ENCODER_CONV_CASES,
            "coverage_note": "unique Conv2d encoder representatives only; provider attach enables Conv kernels for these nodes",
        },
    }
)

for _u22_dim64_encoder_network in (
    "u22_192_base64_encoder",
    "u22_224_base64_encoder",
    "u22_384x288_base64_encoder",
    "u22_384_base64_encoder",
):
    E2E_CKKS_SPECS[_u22_dim64_encoder_network] = dict(E2E_CKKS_SPECS["u22_256_base32"])


CHEDDAR_E2E_CKKS_OVERRIDES: dict[str, dict[str, Any]] = {
    # Cheddar's current C++ adapter only exposes its LogScale=40 preset. Keep
    # Lattigo on the U22 full-graph profile above, but replay U22 Cheddar rows
    # with the compatible preset instead of aborting at NewScheme().
    "u22_64_base32": {
        "logn": 16,
        "logq": (55, 40, 40, 40, 40),
        "logp": (61, 61, 61),
        "logscale": 40,
        "h": 192,
        "boot_logp": (),
    },
    "u22_256_base32": {
        "logn": 16,
        "logq": (55, 40, 40, 40, 40),
        "logp": (61, 61, 61),
        "logscale": 40,
        "h": 192,
        "boot_logp": (),
    },
}


def _ckks_profile() -> str:
    return str(os.environ.get(CKKS_PROFILE_ENV, "e2e")).strip().lower() or "e2e"


def _effective_logn(network: str, *, base_logn: int | None = None) -> int:
    raw_override = os.environ.get(LOGN_OVERRIDE_ENV)
    if raw_override is not None and str(raw_override).strip() != "":
        return int(raw_override)
    if base_logn is not None:
        return int(base_logn)
    return int(NETWORK_SPECS[str(network)]["logn"])


def _kernel_ckks_spec(network: str, *, backend: str) -> dict[str, Any]:
    spec = NETWORK_SPECS[str(network)]
    if str(backend) == "cheddar":
        return {
            "logn": int(_effective_logn(str(network), base_logn=16)),
            "logq": (55, 40, 40, 40, 40),
            "logp": (61, 61, 61),
            "logscale": 40,
            "h": 192,
            "boot_logp": (61, 61, 61, 61, 61, 61, 61, 61),
        }
    return {
        "logn": int(_effective_logn(str(network), base_logn=int(spec["logn"]))),
        "logq": tuple(int(v) for v in spec["logq"]),
        "logp": tuple(int(v) for v in spec["logp"]),
        "logscale": RESNET_LOGSCALE,
        "h": RESNET_SECRET_H,
        "boot_logp": RESNET_BOOT_LOGP,
    }


def _profile_ckks_spec(network: str, *, backend: str) -> dict[str, Any]:
    profile = _ckks_profile()
    if profile == "kernel":
        return _kernel_ckks_spec(str(network), backend=str(backend))
    if profile != "e2e":
        raise ValueError(f"unknown CKKS profile {profile!r}; expected 'e2e' or 'kernel'")
    if str(backend) == "cheddar" and str(network) in CHEDDAR_E2E_CKKS_OVERRIDES:
        spec = dict(CHEDDAR_E2E_CKKS_OVERRIDES[str(network)])
    else:
        spec = dict(E2E_CKKS_SPECS[str(network)])
    spec["logn"] = int(_effective_logn(str(network), base_logn=int(spec["logn"])))
    return spec


def _require_backend(backend: str) -> None:
    backend_name = str(backend)
    if backend_name not in BACKENDS:
        raise ValueError(f"unknown backend {backend_name!r}")
    lib_name = "lattigo-linux.so" if backend_name == "lattigo" else "cheddar-linux.so"
    lib_path = REPO_ROOT / "orion" / "backend" / backend_name / lib_name
    if not lib_path.exists():
        raise RuntimeError(f"local {backend_name} shared library has not been built: {lib_path}")


def _truthy_env_value(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _safe_path_fragment(value: str) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_") or "item"


def _bounded_lattigo_dense_requested(*, network: str, case_name: str, path_kind: str) -> bool:
    if str(path_kind) != "dense":
        return False
    raw = str(os.environ.get(BOUNDED_LATTIGO_DENSE_ENV, "auto")).strip().lower()
    if raw in ("0", "false", "no", "off", "none", "never"):
        return False
    if raw in ("1", "true", "yes", "on", "all", "always"):
        return True
    if raw in ("", "auto"):
        return str(network) == "r34_imgnet" and str(case_name) == "stage4_same"
    cases = {
        item.strip()
        for item in raw.replace(";", ",").split(",")
        if item.strip()
    }
    return str(case_name) in cases or f"{network}/{case_name}" in cases


def _bounded_lattigo_provider_requested(*, network: str, case_name: str, path_kind: str) -> bool:
    if str(path_kind) not in PROVIDER_PATHS:
        return False
    raw = str(os.environ.get(BOUNDED_LATTIGO_PROVIDER_ENV, "auto")).strip().lower()
    if raw in ("0", "false", "no", "off", "none", "never"):
        return False
    if raw in ("1", "true", "yes", "on", "all", "always"):
        return True
    if raw in ("", "auto"):
        return str(network) == "r34_imgnet" and str(case_name) == "stage4_same"
    cases = {
        item.strip()
        for item in raw.replace(";", ",").split(",")
        if item.strip()
    }
    return str(case_name) in cases or f"{network}/{case_name}" in cases


def _bounded_lattigo_dense_mode() -> str:
    mode = str(os.environ.get(BOUNDED_LATTIGO_DENSE_MODE_ENV, "streaming")).strip().lower()
    if mode in ("", "save", "saved", "save-load", "save_load"):
        return "save"
    if mode in ("stream", "streaming", "streaming-lt", "streaming_lt"):
        return "streaming"
    raise ValueError(
        f"unknown {BOUNDED_LATTIGO_DENSE_MODE_ENV}={mode!r}; expected 'save' or 'streaming'"
    )


def _bounded_lattigo_provider_mode() -> str:
    mode = str(os.environ.get(BOUNDED_LATTIGO_PROVIDER_MODE_ENV, "streaming")).strip().lower()
    if mode in ("", "stream", "streaming", "streaming-lt", "streaming_lt"):
        return "streaming"
    if mode in ("save", "saved", "save-load", "save_load"):
        return "save"
    raise ValueError(
        f"unknown {BOUNDED_LATTIGO_PROVIDER_MODE_ENV}={mode!r}; expected 'streaming' or 'save'"
    )


def _clean_lattigo_dense_io_enabled(env: dict[str, str]) -> bool:
    raw = env.get(CLEAN_LATTIGO_DENSE_IO_ENV)
    return True if raw is None else _truthy_env_value(raw)


def _clean_lattigo_provider_io_enabled(env: dict[str, str]) -> bool:
    raw = env.get(CLEAN_LATTIGO_PROVIDER_IO_ENV)
    return True if raw is None else _truthy_env_value(raw)


def _worker_rss_limit_bytes(env: dict[str, str], *, backend: str) -> int:
    raw = env.get(MAX_WORKER_RSS_GB_ENV)
    if raw is None:
        raw = "300" if str(backend) == "lattigo" else "0"
    try:
        gib = float(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    if gib <= 0:
        return 0
    return int(gib * (1024**3))


def _process_rss_bytes(pid: int) -> int | None:
    try:
        with Path(f"/proc/{int(pid)}/status").open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _disk_usage_payload(path: Path) -> dict[str, int | str]:
    current = Path(path)
    while not current.exists() and current.parent != current:
        current = current.parent
    usage = shutil.disk_usage(str(current))
    return {
        "path": str(current),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _read_text_tail(path: Path, *, limit: int = 4000) -> str:
    try:
        size = int(path.stat().st_size)
    except OSError:
        return ""
    try:
        with path.open("rb") as handle:
            if size > int(limit):
                handle.seek(max(0, size - int(limit)))
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _read_text_for_json(path: Path, *, limit: int = 64 * 1024 * 1024) -> str:
    try:
        size = int(path.stat().st_size)
    except OSError:
        return ""
    if size > int(limit):
        return _read_text_tail(path, limit=int(limit))
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _jsonable_shape(value: Any) -> list[int]:
    return [int(v) for v in tuple(value)]


def _backend_device_memory_info() -> dict[str, int] | None:
    backend = getattr(scheme, "backend", None)
    get_memory_info = getattr(backend, "GetDeviceMemoryInfo", None)
    if not callable(get_memory_info):
        return None
    values = list(get_memory_info())
    if len(values) < 2:
        return None
    return {"free_bytes": int(values[0]), "total_bytes": int(values[1])}


def _estimate_transform_device_bytes(transform_id: int) -> int:
    backend = getattr(scheme, "backend", None)
    estimator = getattr(backend, "EstimateLinearTransformDeviceBytes", None)
    if not callable(estimator):
        return 0
    values = list(estimator(int(transform_id)))
    return int(values[0]) if values else 0


def _compiled_transform_ids(module: Any) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()

    def append(transform_id: Any) -> None:
        value = int(transform_id)
        if value in seen:
            return
        seen.add(value)
        ids.append(value)

    for transform_id in dict(getattr(module, "transform_ids", {}) or {}).values():
        append(transform_id)
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None)
    if executor is not None:
        for transform_id in dict(getattr(executor, "transform_ids", {}) or {}).values():
            append(transform_id)
        for group in _executor_unified_groups(executor):
            for transform_id in list(getattr(group, "unified_ids", None) or []):
                append(transform_id)
    return ids


def _memory_event(label: str, module: Any | None = None, transform_ids: list[int] | None = None) -> dict[str, Any]:
    ids = list(transform_ids if transform_ids is not None else (_compiled_transform_ids(module) if module is not None else []))
    estimates = [
        {"transform_id": int(transform_id), "device_bytes": int(_estimate_transform_device_bytes(int(transform_id)))}
        for transform_id in ids
    ]
    estimates.sort(key=lambda item: int(item["device_bytes"]), reverse=True)
    payload: dict[str, Any] = {
        "event": str(label),
        "linear_transform_device_bytes": {
            "transform_count": int(len(estimates)),
            "device_bytes_total": int(sum(int(item["device_bytes"]) for item in estimates)),
            "device_bytes_max": int(estimates[0]["device_bytes"]) if estimates else 0,
            "top_transforms": estimates[:16],
        },
    }
    memory_info = _backend_device_memory_info()
    if memory_info is not None:
        payload["device_memory"] = memory_info
    return payload


def _synchronize_backend() -> None:
    sync = getattr(getattr(scheme, "backend", None), "SynchronizeDevice", None)
    if callable(sync):
        sync()


def _reset_runtime_operation_counters() -> bool:
    reset = getattr(getattr(scheme, "backend", None), "ResetOperationCounters", None)
    if not callable(reset):
        return False
    reset()
    return True


def _runtime_operation_counters() -> dict[str, int] | None:
    get_counters = getattr(getattr(scheme, "backend", None), "GetOperationCounters", None)
    if not callable(get_counters):
        return None
    values = [int(value) for value in list(get_counters())]
    if len(values) < 4:
        return None
    return {
        "rotation": int(values[0]),
        "lintrans_rotation": int(values[1]),
        "direct_rotation": int(values[2]),
        "conjugation": int(values[3]),
    }


def _mean_operation_counters(counters: list[dict[str, int]]) -> dict[str, float]:
    if not counters:
        return {}
    keys = ("rotation", "lintrans_rotation", "direct_rotation", "conjugation")
    return {
        key: float(statistics.fmean(float(item.get(key, 0)) for item in counters))
        for key in keys
    }


def _eager_materialize_dense_transforms(module: Any, *, backend: str, effective_path_kind: str) -> dict[str, Any]:
    """Move Cheddar dense LT hoist/materialization out of the first timed forward."""

    if str(backend) != "cheddar" or str(effective_path_kind) != "dense":
        return {"enabled": False, "seconds": 0.0, "transform_count": 0}
    if os.environ.get("ORION_DENSE_LT_HOST_PAYLOAD_CACHE", "").lower() not in ("", "0", "false", "no", "off"):
        return {
            "enabled": False,
            "seconds": 0.0,
            "transform_count": int(len(_compiled_transform_ids(module))),
            "reason": "dense host payload cache materializes plaintext payloads during compile",
        }

    backend_obj = getattr(scheme, "backend", None)
    load_batch = getattr(backend_obj, "LoadPlaintextDiagonalsBatch", None)
    load_one = getattr(backend_obj, "LoadPlaintextDiagonal", None)
    if not callable(load_batch) and not callable(load_one):
        return {
            "enabled": False,
            "seconds": 0.0,
            "transform_count": 0,
            "reason": "backend has no plaintext materialization entrypoint",
        }

    transform_ids = _compiled_transform_ids(module)
    started = time.perf_counter()
    for transform_id in transform_ids:
        if callable(load_batch):
            load_batch([], [], [], [], int(transform_id))
        else:
            load_one([], int(transform_id), 0)
    _synchronize_backend()
    elapsed = float(time.perf_counter() - started)
    return {
        "enabled": True,
        "seconds": elapsed,
        "transform_count": int(len(transform_ids)),
    }


def _unified_group_memory_traces(module: Any) -> list[dict[str, Any]]:
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None)
    if executor is None:
        return []
    traces: list[dict[str, Any]] = []
    for group_index, group in enumerate(_executor_unified_groups(executor)):
        for event in list(getattr(group, "memory_trace", []) or []):
            payload = dict(event)
            payload["group_index"] = int(group_index)
            traces.append(payload)
    return traces


def _build_config(network: str, *, backend: str) -> dict[str, Any]:
    global _ACTIVE_CHEDDAR_BENCH_IO_DIR
    _ACTIVE_CHEDDAR_BENCH_IO_DIR = None
    ckks_spec = _profile_ckks_spec(str(network), backend=str(backend))
    if str(backend) == "cheddar":
        io_root = Path(os.environ.get("ORION_CHEDDAR_IO_ROOT", tempfile.gettempdir()))
        io_root.mkdir(parents=True, exist_ok=True)
        io_dir = Path(tempfile.mkdtemp(prefix=f"orion_cheddar_{network}_", dir=str(io_root)))
        _ACTIVE_CHEDDAR_BENCH_IO_DIR = Path(io_dir)
        io_mode = os.environ.get("ORION_CHEDDAR_IO_MODE", "save")
        diags_path = os.environ.get("ORION_CHEDDAR_DIAGS_PATH", str(io_dir / "diagonals.h5"))
        keys_path = os.environ.get("ORION_CHEDDAR_KEYS_PATH", str(io_dir / "keys.h5"))
        config = {
            "ckks_params": {
                "LogN": int(ckks_spec["logn"]),
                "LogQ": [int(v) for v in ckks_spec["logq"]],
                "LogP": [int(v) for v in ckks_spec["logp"]],
                "LogScale": int(ckks_spec["logscale"]),
                "H": int(ckks_spec["h"]),
                "RingType": "Standard",
            },
            "orion": {
                "margin": 2,
                "embedding_method": "hybrid",
                "backend": "cheddar",
                "fuse_modules": True,
                "debug": False,
                "io_mode": str(io_mode),
                "diags_path": str(diags_path),
                "keys_path": str(keys_path),
            },
        }
        if tuple(ckks_spec.get("boot_logp", ())) != ():
            config["boot_params"] = {"LogP": [int(v) for v in ckks_spec["boot_logp"]]}
        return config
    lattigo_io_mode = str(os.environ.get(LATTIGO_BENCH_IO_MODE_ENV, "none")).strip().lower() or "none"
    config = {
        "ckks_params": {
            "LogN": int(ckks_spec["logn"]),
            "LogQ": [int(v) for v in ckks_spec["logq"]],
            "LogP": [int(v) for v in ckks_spec["logp"]],
            "LogScale": int(ckks_spec["logscale"]),
            "H": int(ckks_spec["h"]),
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": str(backend),
            "fuse_modules": True,
            "debug": False,
            "io_mode": str(lattigo_io_mode),
        },
    }
    if str(lattigo_io_mode) != "none":
        config["orion"]["diags_path"] = str(os.environ.get(LATTIGO_BENCH_DIAGS_PATH_ENV, "diagonals.h5"))
        config["orion"]["keys_path"] = str(os.environ.get(LATTIGO_BENCH_KEYS_PATH_ENV, "keys.h5"))
    if tuple(ckks_spec.get("boot_logp", ())) != ():
        config["boot_params"] = {"LogP": [int(v) for v in ckks_spec["boot_logp"]]}
    return config


def _build_config_payload(network: str, *, backend: str) -> dict[str, Any]:
    config = _build_config(str(network), backend=str(backend))
    if str(backend) == "cheddar":
        _cleanup_active_cheddar_io_dir()
    return config


def _init_scheme(network: str, *, backend: str) -> None:
    if str(backend) == "cheddar":
        os.environ.setdefault("ORION_CHEDDAR_GPU_PREFETCH", "0")
    scheme.init_scheme(_build_config(str(network), backend=str(backend)))
    Module.set_scheme(scheme)
    Module.set_margin(scheme.params.get_margin())


def _cleanup_scheme() -> None:
    try:
        scheme.delete_scheme()
    except Exception:
        pass
    gc.collect()


def _cleanup_active_cheddar_io_dir() -> None:
    global _ACTIVE_CHEDDAR_BENCH_IO_DIR
    io_dir = _ACTIVE_CHEDDAR_BENCH_IO_DIR
    _ACTIVE_CHEDDAR_BENCH_IO_DIR = None
    if io_dir is None:
        return
    if not _truthy_env_value(os.environ.get(CLEAN_CHEDDAR_IO_ENV)):
        return
    shutil.rmtree(Path(io_dir), ignore_errors=True)


def _init_orion_modules(net: torch.nn.Module, *, fit: bool = False) -> None:
    for module in net.modules():
        if bool(fit) and hasattr(module, "fit") and callable(module.fit):
            module.fit()
        if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
            module.init_orion_params()
        if hasattr(module, "update_params") and callable(module.update_params):
            module.update_params()


def _set_compile_level(dag: NetworkDAG) -> None:
    level = len(scheme.params.get_logq()) - 1
    for node in dag.nodes:
        module = dag.nodes[node].get("module")
        if module is not None and hasattr(module, "set_level"):
            module.set_level(level)


def _prepare_resnet18_tiny_dag(*, provider: bool) -> tuple[NetworkDAG, dict[str, Any]]:
    torch.manual_seed(0)
    net = ResNet18(dataset="tiny")
    net.eval()
    net.set_scheme(scheme)
    net.set_margin(scheme.params.get_margin())
    x = torch.randn(NETWORK_SPECS["r18_tiny"]["input_shape"], dtype=torch.float32)
    traced = OrionTracer().trace_model(net)
    StatsTracker(traced).propagate(x)
    _init_orion_modules(net, fit=True)
    dag = NetworkDAG(traced)
    dag.build_dag()
    Fuser(dag).fuse_modules()
    dag.remove_fused_batchnorms()
    attach_audit: dict[str, Any] = {}
    if bool(provider):
        registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
        attach_audit = registry.attach_to_dag(dag)
    _set_compile_level(dag)
    return dag, attach_audit


def _prepare_resnet34_imgnet_dag(*, provider: bool) -> tuple[NetworkDAG, dict[str, Any]]:
    torch.manual_seed(0)
    net = ResNet34(dataset="imagenet")
    net.eval()
    x = torch.randn(NETWORK_SPECS["r34_imgnet"]["input_shape"], dtype=torch.float32)
    traced = OrionTracer().trace_model(net)
    StatsTracker(traced).propagate(x)
    dag = NetworkDAG(traced)
    dag.build_dag()
    _init_orion_modules(net)
    Fuser(dag).fuse_modules()
    dag.remove_fused_batchnorms()
    attach_audit: dict[str, Any] = {}
    if bool(provider):
        registry = R34CompileRegistry.for_r34_imgnet_phase1(dag)
        attach_audit = registry.attach_to_dag(dag)
    _set_compile_level(dag)
    return dag, attach_audit


def _prepare_u22_dag(network: str, *, provider: bool) -> tuple[NetworkDAG, dict[str, Any]]:
    spec = NETWORK_SPECS[str(network)]
    dataset = str(spec["dataset"])
    input_shape = tuple(int(value) for value in spec["input_shape"])
    torch.manual_seed(0)
    net = UNet22(dataset=dataset, base_dim=int(spec["base_dim"]))
    net.eval()
    x = torch.randn(input_shape, dtype=torch.float32)
    traced = OrionTracer().trace_model(net)
    StatsTracker(traced).propagate(x)
    dag = NetworkDAG(traced)
    dag.build_dag()
    _init_orion_modules(net)
    attach_audit: dict[str, Any] = {}
    if bool(provider):
        opts = _region_first_mode_options(str(network))
        if str(network).endswith("_encoder"):
            opts = dict(opts)
            opts["u22_allowed_nodes"] = tuple(str(case["node"]) for case in spec["cases"])
            opts["u22_conv_kernels"] = True
            opts["u22_layout_policy"] = "dp"
        registry = U22CompileRegistry.for_dag(
            dag,
            allowed_nodes=opts["u22_allowed_nodes"],
            enable_conv_kernels=bool(opts["u22_conv_kernels"]),
            layout_policy=str(opts["u22_layout_policy"]),
        )
        attach_audit = registry.attach_to_dag(dag)
    _set_compile_level(dag)
    return dag, attach_audit


def _prepare_dag(network: str, *, provider: bool) -> tuple[NetworkDAG, dict[str, Any]]:
    if str(network) == "r18_tiny":
        return _prepare_resnet18_tiny_dag(provider=bool(provider))
    if str(network) == "r34_imgnet":
        return _prepare_resnet34_imgnet_dag(provider=bool(provider))
    if str(network).startswith("u22_"):
        return _prepare_u22_dag(str(network), provider=bool(provider))
    raise KeyError(f"unknown network {network!r}")


def _is_provider_path(path_kind: str) -> bool:
    return str(path_kind) in PROVIDER_PATHS


def _dense_cols(module: Any) -> int:
    keys = list(getattr(module, "transform_ids", {}).keys())
    if not keys:
        return 0
    return max(int(col) for _row, col in keys) + 1


def _dense_rows(module: Any) -> int:
    keys = list(getattr(module, "transform_ids", {}).keys())
    if not keys:
        return 0
    return max(int(row) for row, _col in keys) + 1


def _fallback_input_block_count(module: Any) -> int:
    slots = int(scheme.params.get_slots())
    total = int(torch.Size(getattr(module, "fhe_input_shape")).numel())
    return max(1, int((total + slots - 1) // slots))


def _provider_source_count(module: Any) -> int:
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None)
    if executor is None:
        return _fallback_input_block_count(module)
    if bool(getattr(executor, "requires_compact_source", False)):
        return 1
    if hasattr(executor, "source_pair_count"):
        return max(1, int(getattr(executor, "source_pair_count")) * 2)
    if hasattr(executor, "cols") and int(getattr(executor, "cols") or 0) > 0:
        return int(getattr(executor, "cols"))
    if hasattr(executor, "input_block_count") and int(getattr(executor, "input_block_count") or 0) > 0:
        return int(getattr(executor, "input_block_count"))
    groups = list(getattr(executor, "groups", []) or [])
    if groups:
        return int(len(groups))
    if getattr(executor, "group", None) is not None:
        return 1
    return _fallback_input_block_count(module)


def _source_count(module: Any, *, path_kind: str) -> int:
    if _is_provider_path(str(path_kind)):
        return _provider_source_count(module)
    return max(1, _dense_cols(module) or _fallback_input_block_count(module))


def _prebuilt_sources(module: Any, *, count: int, repeats: int, seed: int, provider: bool) -> list[CipherTensor]:
    sources: list[CipherTensor] = []
    level = len(scheme.params.get_logq()) - 1
    gen = torch.Generator().manual_seed(int(seed))
    for _repeat_index in range(int(repeats)):
        sources.append(_make_source(module, count=int(count), gen=gen, level=int(level), provider=bool(provider)))
    return sources


def _make_source(module: Any, *, count: int, gen: torch.Generator, level: int, provider: bool) -> CipherTensor:
    ids: list[int] = []
    for _source_index in range(int(count)):
        packed = torch.randn((scheme.params.get_slots(),), generator=gen, dtype=torch.float32) * 0.01
        ct = scheme.encrypt(scheme.encode(packed, int(level)))
        ids.append(int(ct.ids[0]))
        ct.ids = []
    source = CipherTensor(scheme, ids, module.input_shape, module.fhe_input_shape)
    if bool(provider):
        runtime = getattr(module, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        if bool(getattr(executor, "requires_compact_source", False)):
            source.region_first_compact_source = True
            source.stage4_compact_source = True
            source.is_region_first_compact_source = True
    return source


def _rotation_keys(transform_id: int) -> list[int]:
    try:
        keys = [int(key) for key in scheme.backend.GetLinearTransformRotationKeys(int(transform_id))]
        return sorted(int(key) for key in keys if int(key) != 0)
    finally:
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() == "save":
            remover = getattr(scheme.backend, "RemovePlaintextDiagonals", None)
            if callable(remover):
                remover(int(transform_id))


def _rotation_keys_from_group(group: Any, transform_id: int) -> list[int]:
    by_transform = getattr(group, "_required_keys_by_transform", {}) or {}
    if int(transform_id) in by_transform:
        return sorted(int(key) for key, _level in by_transform[int(transform_id)] if int(key) != 0)
    return _rotation_keys(int(transform_id))


def _nonzero_rotations(values: set[int]) -> set[int]:
    return {int(value) for value in values if int(value) != 0}


def _powers_of_two_for_slots(slots: int) -> list[int]:
    values: list[int] = []
    n1 = 1
    while n1 < int(slots):
        values.append(int(n1))
        n1 <<= 1
    return values or [1]


def _bsgs_index(diag_indices: set[int], *, slots: int, n1: int) -> tuple[set[int], set[int]]:
    rot_n1: set[int] = set()
    rot_n2: set[int] = set()
    slots = int(slots)
    n1 = max(1, int(n1))
    for value in diag_indices:
        rot = int(value) % int(slots)
        idx_n1 = int(((rot // int(n1)) * int(n1)) % int(slots))
        idx_n2 = int(rot % int(n1))
        rot_n1.add(int(idx_n1))
        rot_n2.add(int(idx_n2))
    return rot_n1, rot_n2


def _shared_cache_bsgs_group_cost(
    entries: list[dict[str, Any]],
    *,
    slots: int,
    n1s: list[int],
) -> dict[str, Any]:
    shared_baby: set[int] = set()
    reported_key_union: set[int] = set()
    giant_total = 0
    transform_cost_total = 0
    per_transform: list[dict[str, Any]] = []
    for transform_index, (entry, n1) in enumerate(zip(entries, n1s)):
        diags = {int(value) for value in entry.get("diag_indices", ())}
        rot_n1, rot_n2 = _bsgs_index(diags, slots=int(slots), n1=int(n1))
        baby = _nonzero_rotations(rot_n2)
        giant = _nonzero_rotations(rot_n1)
        shared_baby.update(baby)
        reported_key_union.update(int(value) for value in rot_n1)
        reported_key_union.update(int(value) for value in rot_n2)
        giant_total += int(len(giant))
        transform_cost_total += int(len(baby) + len(giant))
        per_transform.append(
            {
                "transform_index": int(transform_index),
                "row": entry.get("row"),
                "col": entry.get("col"),
                "transform_id": entry.get("transform_id"),
                "n1": int(n1),
                "raw_diag_count": int(len(diags)),
                "baby_rotation_count": int(len(baby)),
                "giant_rotation_count": int(len(giant)),
                "per_transform_rotation_count": int(len(baby) + len(giant)),
                "reported_unique_key_union_count": int(len(set(rot_n1) | set(rot_n2))),
            }
        )
    return {
        "shared_baby_rotation_count": int(len(shared_baby)),
        "giant_rotation_count_total": int(giant_total),
        "actual_rotation_callback_count": int(len(shared_baby) + giant_total),
        "sum_per_transform_rotation_count": int(transform_cost_total),
        "reported_unique_key_union_count": int(len(reported_key_union)),
        "per_transform_bsgs": per_transform,
    }


def _best_unified_common_n1(entries: list[dict[str, Any]], *, slots: int) -> tuple[int, dict[str, Any]]:
    best_n1 = 1
    best_cost = 10**18
    best_payload: dict[str, Any] | None = None
    for n1 in _powers_of_two_for_slots(int(slots)):
        payload = _shared_cache_bsgs_group_cost(
            entries,
            slots=int(slots),
            n1s=[int(n1)] * int(len(entries)),
        )
        cost = int(payload["actual_rotation_callback_count"])
        if cost < best_cost or (cost == best_cost and int(n1) > int(best_n1)):
            best_cost = int(cost)
            best_n1 = int(n1)
            best_payload = payload
    assert best_payload is not None
    return int(best_n1), best_payload


def _lattigo_find_best_bsgs_n1(diag_indices: set[int], *, slots: int, log_max_ratio: int) -> int:
    max_ratio = float(1 << int(log_max_ratio))
    previous = 1
    for n1 in _powers_of_two_for_slots(int(slots)):
        rot_n1, rot_n2 = _bsgs_index(set(diag_indices), slots=int(slots), n1=int(n1))
        nb_n1 = int(len(rot_n1) - 1)
        nb_n2 = int(len(rot_n2) - 1)
        if nb_n1 <= 0:
            previous = int(n1)
            continue
        ratio = float(nb_n2) / float(nb_n1)
        if ratio == max_ratio:
            return int(n1)
        if ratio > max_ratio:
            return int(previous)
        previous = int(n1)
    return 1


def _dense_bsgs_log_ratio(module: Any) -> int:
    ratio = float(getattr(module, "bsgs_ratio", 1.0) or 1.0)
    if ratio <= 1.0:
        return 0
    return max(0, int(math.log(ratio)))


def _dense_diag_entries_by_col(
    module: Any,
    transform_ids: dict[tuple[int, int], int],
    *,
    rows: int,
    cols: int,
) -> list[list[dict[str, Any]]]:
    diagonals = getattr(module, "diagonals", {}) or {}
    if not diagonals:
        return []
    groups: list[list[dict[str, Any]]] = []
    for col in range(int(cols)):
        group: list[dict[str, Any]] = []
        for row in range(int(rows)):
            transform_id = transform_ids.get((int(row), int(col)))
            if transform_id is None:
                continue
            block_diags = diagonals.get((int(row), int(col)))
            if block_diags is None:
                return []
            group.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "transform_id": int(transform_id),
                    "diag_indices": {int(value) for value in dict(block_diags).keys()},
                }
            )
        groups.append(group)
    return groups


def _unified_group_diag_entries(group: Any) -> list[dict[str, Any]]:
    ids = [int(value) for value in (getattr(group, "unified_ids", None) or [])]
    by_transform = getattr(group, "_diag_indices_by_transform", {}) or {}
    if not ids or not by_transform:
        return []
    entries: list[dict[str, Any]] = []
    for transform_index, transform_id in enumerate(ids):
        if int(transform_id) not in by_transform:
            return []
        entries.append(
            {
                "transform_index": int(transform_index),
                "transform_id": int(transform_id),
                "diag_indices": {int(value) for value in by_transform[int(transform_id)]},
            }
        )
    return entries


def _linear_transform_rotation_stats(module: Any) -> dict[str, Any]:
    transform_ids = dict(getattr(module, "transform_ids", {}) or {})
    per_transform: list[dict[str, Any]] = []
    unique_keys: set[int] = set()
    transform_rotation_total = 0
    for (row, col), transform_id in sorted(transform_ids.items()):
        nonzero_keys = _rotation_keys(int(transform_id))
        unique_keys.update(nonzero_keys)
        transform_rotation_total += int(len(nonzero_keys))
        per_transform.append(
            {
                "row": int(row),
                "col": int(col),
                "transform_id": int(transform_id),
                "rotation_keys": nonzero_keys,
                "rotation_key_count": int(len(nonzero_keys)),
            }
        )
    cols = _dense_cols(module)
    rows = _dense_rows(module)
    output_rotations = int(getattr(module, "output_rotations", 0) or 0)
    output_rotation_evals = int(rows * output_rotations)
    tconv_unified_stats = _dense_tconv_unified_stats_enabled(module)
    if _dense_shared_cache_stats_enabled(module=module, rows=int(rows), cols=int(cols)) or bool(tconv_unified_stats):
        dense_groups = _dense_diag_entries_by_col(
            module,
            transform_ids,
            rows=int(rows),
            cols=int(cols),
        )
        if dense_groups:
            slots = int(scheme.params.get_slots())
            log_ratio = _dense_bsgs_log_ratio(module)
            per_group: list[dict[str, Any]] = []
            shared_rotation_total = 0
            reported_unique_key_union_total = 0
            for col, group_entries in enumerate(dense_groups):
                if module.__class__.__name__ == "ConvTranspose2d":
                    unified_n1, cost = _best_unified_common_n1(group_entries, slots=int(slots))
                else:
                    n1s = [
                        _lattigo_find_best_bsgs_n1(
                            set(entry["diag_indices"]),
                            slots=int(slots),
                            log_max_ratio=int(log_ratio),
                        )
                        for entry in group_entries
                    ]
                    unified_n1 = 0
                    cost = _shared_cache_bsgs_group_cost(group_entries, slots=int(slots), n1s=n1s)
                group_keys = {
                    int(key)
                    for entry in group_entries
                    for key in _rotation_keys(int(entry["transform_id"]))
                }
                shared_rotation_total += int(cost["actual_rotation_callback_count"])
                reported_unique_key_union_total += int(cost["reported_unique_key_union_count"])
                per_group.append(
                    {
                        "group_index": int(col),
                        "input_col": int(col),
                        "transform_count": int(len(group_entries)),
                        "transform_rotation_key_count_total": int(
                            sum(int(item["rotation_key_count"]) for item in per_transform if int(item["col"]) == int(col))
                        ),
                        "shared_rotation_eval_count": int(cost["actual_rotation_callback_count"]),
                        "actual_rotation_callback_count": int(cost["actual_rotation_callback_count"]),
                        "shared_baby_rotation_count": int(cost["shared_baby_rotation_count"]),
                        "giant_rotation_count_total": int(cost["giant_rotation_count_total"]),
                        "sum_per_transform_rotation_count": int(cost["sum_per_transform_rotation_count"]),
                        "reported_unique_key_union_count": int(cost["reported_unique_key_union_count"]),
                        "compiled_unique_rotation_key_count": int(len(group_keys)),
                        "unique_rotation_key_count": int(len(group_keys)),
                        "unique_rotation_keys": sorted(int(key) for key in group_keys),
                        "bsgs_log_ratio": int(log_ratio),
                        "unified_n1": int(unified_n1),
                        "per_transform_bsgs": cost["per_transform_bsgs"],
                        "per_transform": [
                            item for item in per_transform if int(item["col"]) == int(col)
                        ],
                    }
                )
            actual_total = int(shared_rotation_total + output_rotation_evals)
            return {
                "source": "compiled_lattigo_transform_diagonal_indices",
                "method": (
                    "actual dense shared-cache BSGS callback-equivalent count: "
                    "unique nonzero baby rotations per input-column group plus nonzero giant rotations per transform, "
                    "then Orion hybrid output rotations"
                ),
                "dense_shared_cache": bool(_dense_shared_cache_stats_enabled(module=module, rows=int(rows), cols=int(cols))),
                "dense_unified_bsgs_stats": bool(tconv_unified_stats),
                "group_count": int(len(per_group)),
                "transform_count": int(len(transform_ids)),
                "rows": int(rows),
                "cols": int(cols),
                "transform_rotation_key_count_total": int(transform_rotation_total),
                "shared_rotation_eval_count_total": int(shared_rotation_total),
                "actual_rotation_callback_count": int(actual_total),
                "reported_unique_key_union_count": int(reported_unique_key_union_total + output_rotation_evals),
                "unique_rotation_key_count": int(len(unique_keys)),
                "output_rotations_per_output_ct": int(output_rotations),
                "output_rotation_eval_count": int(output_rotation_evals),
                "rotation_eval_count": int(actual_total),
                "unique_rotation_keys": sorted(int(key) for key in unique_keys),
                "per_transform": per_transform,
                "per_group": per_group,
            }
    return {
        "source": "compiled_lattigo_transform_galois_elements",
        "method": "sum per dense transform invocation BSGS rotation keys plus Orion hybrid output rotations",
        "transform_count": int(len(transform_ids)),
        "rows": int(rows),
        "cols": int(cols),
        "transform_rotation_eval_count": int(transform_rotation_total),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": int(output_rotations),
        "output_rotation_eval_count": int(output_rotation_evals),
        "actual_rotation_callback_count": int(transform_rotation_total + output_rotation_evals),
        "reported_unique_key_union_count": int(len(unique_keys) + output_rotation_evals),
        "rotation_eval_count": int(transform_rotation_total + output_rotation_evals),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "per_transform": per_transform,
    }


def _dense_tconv_unified_stats_enabled(module: Any) -> bool:
    if module.__class__.__name__ != "ConvTranspose2d":
        return False
    if bool(getattr(module, "dense_tconv_unified_bsgs", False)):
        return True
    return bool(os.environ.get("ORION_DENSE_LT_SHARED_CACHE", "").lower() in ("1", "true", "yes", "on"))


def _dense_shared_cache_stats_enabled(*, module: Any, rows: int, cols: int) -> bool:
    override = os.environ.get("ORION_DENSE_LT_SHARED_CACHE")
    requested = bool(override is not None and override.lower() in ("1", "true", "yes", "on"))
    requested = bool(requested or _dense_tconv_unified_stats_enabled(module))
    if not bool(requested):
        return False
    if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "none":
        return False
    if not callable(getattr(scheme.backend, "EvaluateLinearTransformsWithSharedCache", None)):
        return False
    return int(rows) > 1 and int(cols) > 0


def _append_group(groups: list[Any], seen: set[int], group: Any) -> None:
    if group is None:
        return
    marker = int(id(group))
    if marker in seen:
        return
    seen.add(marker)
    groups.append(group)


def _executor_unified_groups(executor: Any) -> list[Any]:
    groups: list[Any] = []
    seen: set[int] = set()
    _append_group(groups, seen, getattr(executor, "group", None))
    for attr in ("groups", "groups_by_input_block", "groups_by_input_chunk", "groups_by_pair"):
        value = getattr(executor, attr, None)
        if isinstance(value, dict):
            for _key, group in sorted(value.items()):
                _append_group(groups, seen, group)
        else:
            for group in list(value or []):
                _append_group(groups, seen, group)
    groups_by_input_index = getattr(executor, "groups_by_input_index", None)
    if isinstance(groups_by_input_index, dict):
        for _key, group in sorted(groups_by_input_index.items()):
            _append_group(groups, seen, group)
    return groups


_RUNTIME_FAIRNESS_NUMERIC_KEYS = (
    "resident_compute_s",
    "serving_hot_s",
    "artifact_read_s",
    "artifact_load_s",
    "artifact_unload_s",
    "trim_s",
    "read_bundle_s",
    "load_keys_s",
    "load_plaintexts_s",
    "eval_s",
    "eval_total_s",
    "unload_s",
    "cpp_plan_s",
    "cpp_level_adjust_s",
    "cpp_baby_step_s",
    "cpp_giant_step_s",
    "stream_build_map_s",
    "stream_encode_hoist_s",
    "stream_load_payload_s",
    "stream_eval_s",
    "stream_accumulate_s",
    "cpp_push_s",
    "cpp_trim_s",
)


def _runtime_fairness_mode_from_env() -> str:
    raw = os.environ.get("ORION_LATTIGO_STREAMING_LT", "")
    if str(raw).strip().lower() in {"1", "true", "yes", "on", "force", "always"}:
        return "streaming_eval_encode"
    return "unknown"


def _aggregate_runtime_fairness_timings(timings: list[dict[str, Any]], *, serving_hot_s: float) -> dict[str, Any]:
    payload: dict[str, Any] = {key: 0.0 for key in _RUNTIME_FAIRNESS_NUMERIC_KEYS}
    modes: list[str] = []
    resident_available = True
    for timing in timings:
        mode = str(timing.get("runtime_fairness_mode", "unknown") or "unknown")
        modes.append(mode)
        if mode == "streaming_eval_encode":
            resident_available = False
        for key in _RUNTIME_FAIRNESS_NUMERIC_KEYS:
            value = timing.get(key)
            if value is None:
                if key == "resident_compute_s":
                    resident_available = False
                continue
            try:
                payload[key] = float(payload.get(key, 0.0)) + float(value)
            except (TypeError, ValueError):
                if key == "resident_compute_s":
                    resident_available = False
    if not timings:
        mode = _runtime_fairness_mode_from_env()
        resident_available = False
    elif any(mode == "streaming_eval_encode" for mode in modes):
        mode = "streaming_eval_encode"
    elif any(mode == "memory_bounded_load_eval" for mode in modes):
        mode = "memory_bounded_load_eval"
    elif modes and all(mode == "resident_compute" for mode in modes):
        mode = "resident_compute"
    else:
        mode = "unknown"
        resident_available = False
    payload["serving_hot_s"] = float(serving_hot_s)
    if not resident_available:
        payload["resident_compute_s"] = None
    payload["runtime_fairness_mode"] = str(mode)
    payload["source_count"] = int(len(timings))
    return payload


def _runtime_fairness_for_module(module: Any, *, serving_hot_s: float) -> dict[str, Any]:
    timings: list[dict[str, Any]] = []
    layer_timing = getattr(module, "_last_runtime_timing", None)
    if isinstance(layer_timing, dict):
        timings.append(dict(layer_timing))
    evaluator_timing = getattr(getattr(scheme, "lt_evaluator", None), "last_runtime_timing", None)
    if isinstance(evaluator_timing, dict) and not timings:
        timings.append(dict(evaluator_timing))
    executor = getattr(getattr(module, "region_runtime", None), "executor", None)
    for group in _executor_unified_groups(executor):
        timing = getattr(group, "last_runtime_timing", None)
        if isinstance(timing, dict):
            timings.append(dict(timing))
    return _aggregate_runtime_fairness_timings(timings, serving_hot_s=float(serving_hot_s))


def _mean_runtime_fairness(timings: list[dict[str, Any]]) -> dict[str, Any]:
    if not timings:
        return _aggregate_runtime_fairness_timings([], serving_hot_s=0.0)
    result: dict[str, Any] = {}
    for key in _RUNTIME_FAIRNESS_NUMERIC_KEYS:
        values = [item.get(key) for item in timings if item.get(key) is not None]
        result[key] = _fmean([float(value) for value in values]) if values else None
    modes = [str(item.get("runtime_fairness_mode", "unknown") or "unknown") for item in timings]
    if any(mode == "streaming_eval_encode" for mode in modes):
        mode = "streaming_eval_encode"
    elif any(mode == "memory_bounded_load_eval" for mode in modes):
        mode = "memory_bounded_load_eval"
    elif modes and all(mode == "resident_compute" for mode in modes):
        mode = "resident_compute"
    else:
        mode = "unknown"
    result["runtime_fairness_mode"] = str(mode)
    result["source_count"] = int(sum(int(item.get("source_count", 0) or 0) for item in timings))
    return result


def _unified_group_rotation_stats(groups: list[Any], *, family_sharing: bool = True) -> dict[str, Any]:
    per_group: list[dict[str, Any]] = []
    all_unique_keys: set[int] = set()
    transform_rotation_total = 0
    shared_rotation_total = 0
    reported_unique_key_union_total = 0
    individual_rotation_total = 0
    transform_count = 0
    for group_index, group in enumerate(groups):
        ids = [int(value) for value in (getattr(group, "unified_ids", None) or [])]
        group_keys: set[int] = set()
        per_transform: list[dict[str, Any]] = []
        for transform_index, transform_id in enumerate(ids):
            nonzero_keys = _rotation_keys_from_group(group, int(transform_id))
            group_keys.update(nonzero_keys)
            all_unique_keys.update(nonzero_keys)
            transform_rotation_total += int(len(nonzero_keys))
            transform_count += 1
            per_transform.append(
                {
                    "transform_index": int(transform_index),
                    "transform_id": int(transform_id),
                    "rotation_keys": nonzero_keys,
                    "rotation_key_count": int(len(nonzero_keys)),
                }
            )
        diag_entries = _unified_group_diag_entries(group)
        cost: dict[str, Any] | None = None
        unified_n1: int | None = None
        if diag_entries:
            unified_n1, cost = _best_unified_common_n1(
                diag_entries,
                slots=int(scheme.params.get_slots()),
            )
            group_shared_rotation_count = int(cost["actual_rotation_callback_count"])
            group_individual_rotation_count = int(cost["sum_per_transform_rotation_count"])
            group_reported_key_union = int(cost["reported_unique_key_union_count"])
        else:
            group_shared_rotation_count = int(len(group_keys))
            group_individual_rotation_count = int(sum(int(item["rotation_key_count"]) for item in per_transform))
            group_reported_key_union = int(len(group_keys))
        shared_rotation_total += int(group_shared_rotation_count)
        individual_rotation_total += int(group_individual_rotation_count)
        reported_unique_key_union_total += int(group_reported_key_union)
        per_group.append(
            {
                "group_index": int(group_index),
                "transform_count": int(len(ids)),
                "transform_rotation_key_count_total": int(sum(int(item["rotation_key_count"]) for item in per_transform)),
                "shared_rotation_eval_count": int(group_shared_rotation_count),
                "actual_rotation_callback_count": int(
                    group_shared_rotation_count if bool(family_sharing) else group_individual_rotation_count
                ),
                "shared_baby_rotation_count": None if cost is None else int(cost["shared_baby_rotation_count"]),
                "giant_rotation_count_total": None if cost is None else int(cost["giant_rotation_count_total"]),
                "sum_per_transform_rotation_count": int(group_individual_rotation_count),
                "reported_unique_key_union_count": int(group_reported_key_union),
                "unified_n1": None if unified_n1 is None else int(unified_n1),
                "bsgs_family_sharing": bool(family_sharing),
                "unique_rotation_key_count": int(len(group_keys)),
                "unique_rotation_keys": sorted(int(key) for key in group_keys),
                "per_transform_bsgs": [] if cost is None else cost["per_transform_bsgs"],
                "per_transform": per_transform,
            }
        )
    rotation_eval_count = int(shared_rotation_total if bool(family_sharing) else individual_rotation_total)
    return {
        "source": "compiled_lattigo_unified_transform_diagonal_indices",
        "method": (
            "actual Unified shared-cache BSGS callback-equivalent count: "
            "unique nonzero baby rotations per group plus nonzero giant rotations per transform"
            if bool(family_sharing)
            else (
                "family-sharing ablation: BSGS preserved, but each compiled LinearTransform is "
                "evaluated individually; rotations are summed per transform with the compiled common BSGS split"
            )
        ),
        "group_count": int(len(groups)),
        "transform_count": int(transform_count),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "shared_rotation_eval_count_total": int(shared_rotation_total),
        "individual_rotation_eval_count_total": int(individual_rotation_total),
        "actual_rotation_callback_count": int(rotation_eval_count),
        "reported_unique_key_union_count": int(reported_unique_key_union_total),
        "unique_rotation_key_count": int(len(all_unique_keys)),
        "output_rotations_per_output_ct": 0,
        "output_rotation_eval_count": 0,
        "rotation_eval_count": int(rotation_eval_count),
        "bsgs_family_sharing": bool(family_sharing),
        "unique_rotation_keys": sorted(int(key) for key in all_unique_keys),
        "per_group": per_group,
    }


def _executor_transform_rotation_stats(executor: Any) -> dict[str, Any]:
    transform_ids = dict(getattr(executor, "transform_ids", {}) or {})
    unique_keys: set[int] = set()
    per_transform: list[dict[str, Any]] = []
    transform_rotation_total = 0
    cols = 0 if not transform_ids else max(int(col) for _row, col in transform_ids) + 1
    rows = 0 if not transform_ids else max(int(row) for row, _col in transform_ids) + 1
    for (row, col), transform_id in sorted(transform_ids.items()):
        nonzero_keys = _rotation_keys(int(transform_id))
        unique_keys.update(nonzero_keys)
        transform_rotation_total += int(len(nonzero_keys))
        per_transform.append(
            {
                "row": int(row),
                "col": int(col),
                "transform_id": int(transform_id),
                "rotation_keys": nonzero_keys,
                "rotation_key_count": int(len(nonzero_keys)),
            }
        )
    output_rotations = int(getattr(executor, "output_rotations", 0) or 0)
    output_rotation_evals = int(rows * output_rotations)
    return {
        "source": "compiled_lattigo_executor_transform_galois_elements",
        "method": "sum per executor transform invocation plus executor output rotations",
        "transform_count": int(len(transform_ids)),
        "rows": int(rows),
        "cols": int(cols),
        "transform_rotation_eval_count": int(transform_rotation_total),
        "transform_rotation_key_count_total": int(transform_rotation_total),
        "unique_rotation_key_count": int(len(unique_keys)),
        "output_rotations_per_output_ct": int(output_rotations),
        "output_rotation_eval_count": int(output_rotation_evals),
        "actual_rotation_callback_count": int(transform_rotation_total + output_rotation_evals),
        "reported_unique_key_union_count": int(len(unique_keys) + output_rotation_evals),
        "rotation_eval_count": int(transform_rotation_total + output_rotation_evals),
        "unique_rotation_keys": sorted(int(key) for key in unique_keys),
        "per_transform": per_transform,
    }


def _provider_rotation_stats(module: Any, *, path_kind: str = "provider") -> dict[str, Any]:
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None)
    if executor is None:
        return {
            "source": "no_provider_executor",
            "method": "not_available",
            "rotation_eval_count": 0,
        }
    groups = _executor_unified_groups(executor)
    if groups:
        stats = _unified_group_rotation_stats(
            groups,
            family_sharing=str(path_kind) not in NO_FAMILY_PATHS,
        )
    elif getattr(executor, "transform_ids", None):
        stats = _executor_transform_rotation_stats(executor)
    else:
        stats = {
            "source": "no_lattigo_transform_ids",
            "method": "not_available",
            "group_count": 0,
            "transform_count": 0,
            "rotation_eval_count": 0,
        }
    stats["executor"] = type(executor).__name__
    stats["runtime_stage"] = "" if runtime is None else str(getattr(runtime, "stage", ""))
    stats["compact_source_contract"] = bool(getattr(executor, "requires_compact_source", False))
    stats["ct_pt_hybrid_packing"] = bool(getattr(executor, "use_ct_pt_hybrid_packing", False))
    stats["tile_family_sharing"] = not bool(getattr(executor, "disable_tile_family_sharing", False))
    stats["bsgs_family_sharing"] = str(path_kind) not in NO_FAMILY_PATHS
    output_rotations = int(
        getattr(executor, "output_fold_rotations", getattr(executor, "output_rotations", 0)) or 0
    )
    output_ct_count = int(
        getattr(executor, "output_block_count", getattr(executor, "bank_count", 0)) or 0
    )
    output_rotation_evals = int(output_rotations * output_ct_count)
    if int(output_rotation_evals) > 0:
        stats["output_rotations_per_output_ct"] = int(output_rotations)
        stats["output_rotation_eval_count"] = int(output_rotation_evals)
        stats["actual_rotation_callback_count"] = int(stats.get("actual_rotation_callback_count", 0)) + int(
            output_rotation_evals
        )
        stats["reported_unique_key_union_count"] = int(stats.get("reported_unique_key_union_count", 0)) + int(
            output_rotation_evals
        )
        stats["rotation_eval_count"] = int(stats.get("rotation_eval_count", 0)) + int(output_rotation_evals)
        stats["method"] = f"{stats.get('method', '')}; plus provider output-fold rotations"
    stats["input_block_pairs"] = [
        [int(left), None if right is None else int(right)]
        for left, right in list(getattr(executor, "input_block_pairs", ()) or ())
    ]
    stats["manual_rotation_note"] = (
        "compact-source providers are timed with the compact-source kernel contract; "
        "source-layout prepacking rotations are not included"
    )
    return stats


def _module_metadata(module: Any) -> dict[str, Any]:
    payload = {
        "module_type": type(module).__name__,
        "input_shape": _jsonable_shape(getattr(module, "input_shape", ())),
        "output_shape": _jsonable_shape(getattr(module, "output_shape", ())),
        "fhe_input_shape": _jsonable_shape(getattr(module, "fhe_input_shape", ())),
        "fhe_output_shape": _jsonable_shape(getattr(module, "fhe_output_shape", ())),
        "input_gap": int(getattr(module, "input_gap", 0) or 0),
        "output_gap": int(getattr(module, "output_gap", 0) or 0),
        "level": int(getattr(module, "level", 0) or 0),
        "depth": int(getattr(module, "depth", 0) or 0),
    }
    if isinstance(module, (Conv2d, ConvTranspose2d)):
        payload.update(
            {
                "in_channels": int(getattr(module, "in_channels")),
                "out_channels": int(getattr(module, "out_channels")),
                "kernel_size": [int(v) for v in getattr(module, "kernel_size")],
                "stride": [int(v) for v in getattr(module, "stride")],
                "padding": [int(v) for v in getattr(module, "padding")],
                "dilation": [int(v) for v in getattr(module, "dilation")],
                "groups": int(getattr(module, "groups")),
            }
        )
    if isinstance(module, ConvTranspose2d):
        payload["output_padding"] = [int(v) for v in getattr(module, "output_padding")]
        payload["dense_tconv_unified_bsgs"] = bool(getattr(module, "dense_tconv_unified_bsgs", False))
    return payload


def _fmean(values: list[float]) -> float:
    return float(statistics.fmean(float(value) for value in values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(float(value) for value in values)) if len(values) > 1 else 0.0


def _install_unified_individual_eval_ablation():
    original = getattr(UnifiedTransformGroup, "evaluate_unified")
    if bool(getattr(UnifiedTransformGroup, "_node_bench_individual_eval_installed", False)):
        return None

    def evaluate_unified_individual(self: UnifiedTransformGroup, ct_input_id: int, backend) -> list[int]:
        if not self.is_compiled or self.unified_ids is None:
            raise RuntimeError("UnifiedTransformGroup must be compiled before evaluation")
        output_ids: list[int] = []
        group_timing = {
            "read_bundle_s": 0.0,
            "load_keys_s": 0.0,
            "load_plaintexts_s": 0.0,
            "eval_s": 0.0,
            "unload_s": 0.0,
            "trim_s": 0.0,
        }

        def add_timing(key: str, value: float) -> None:
            group_timing[key] = float(group_timing.get(key, 0.0) + float(value))

        self._record_memory_event(
            "before_eval_group_individual_no_family_sharing",
            backend,
            self.unified_ids,
            chunk_count=int(len(self.unified_ids)),
        )
        for transform_index, transform_id in enumerate([int(value) for value in self.unified_ids]):
            chunk_ids = [int(transform_id)]
            required_keys = self._required_keys_for_transform_ids(chunk_ids)
            required_keys_to_load = self._rotation_key_requests_to_load(backend, chunk_ids, required_keys)
            self._record_memory_event(
                "before_eval_individual_transform_load",
                backend,
                chunk_ids,
                chunk_index=int(transform_index),
                chunk_transform_count=1,
            )
            read_started = time.perf_counter()
            bundle = None
            if self._should_offload_rotation_keys() or self._offloaded_plaintext_diagonals:
                self._forward_memory_guard(
                    backend,
                    reason=f"before_unified_individual_load:{self._storage_key}:{int(transform_index)}",
                    transform_ids=chunk_ids,
                )
                bundle = self._read_saved_io_bundle(
                    backend,
                    prefetch=False,
                    transform_ids=chunk_ids,
                    required_keys=required_keys_to_load,
                    include_plaintexts=True,
                )
            add_timing("read_bundle_s", time.perf_counter() - read_started)
            try:
                load_keys_started = time.perf_counter()
                self._load_rotation_keys(
                    backend,
                    bundle,
                    transform_ids=chunk_ids,
                    required_keys=required_keys_to_load,
                )
                add_timing("load_keys_s", time.perf_counter() - load_keys_started)
                load_plaintexts_started = time.perf_counter()
                self._load_plaintext_diagonals(backend, bundle, transform_ids=chunk_ids)
                add_timing("load_plaintexts_s", time.perf_counter() - load_plaintexts_started)
                self._record_memory_event(
                    "after_eval_individual_transform_load",
                    backend,
                    chunk_ids,
                    chunk_index=int(transform_index),
                    chunk_transform_count=1,
                )
                eval_started = time.perf_counter()
                output_ids.append(int(backend.EvaluateLinearTransform(int(transform_id), int(ct_input_id))))
                add_timing("eval_s", time.perf_counter() - eval_started)
                self._record_memory_event(
                    "after_eval_individual_transform",
                    backend,
                    chunk_ids,
                    chunk_index=int(transform_index),
                    chunk_transform_count=1,
                    output_count=1,
                )
            finally:
                unload_started = time.perf_counter()
                self._unload_plaintext_diagonals(backend, transform_ids=chunk_ids)
                self._unload_rotation_keys(backend, transform_ids=chunk_ids)
                add_timing("unload_s", time.perf_counter() - unload_started)
                if bundle is not None:
                    bundle.clear()
                trim_started = time.perf_counter()
                trim_event = self._forward_memory_guard(
                    backend,
                    reason=f"after_unified_individual_unload:{self._storage_key}:{int(transform_index)}",
                    transform_ids=chunk_ids,
                    force_trim=True,
                    raise_on_low=False,
                )
                add_timing("trim_s", time.perf_counter() - trim_started)
                self._record_memory_event(
                    "after_eval_individual_transform_unload",
                    backend,
                    chunk_ids,
                    chunk_index=int(transform_index),
                    chunk_transform_count=1,
                    memory_guard=trim_event,
                )
        self._record_memory_event(
            "after_eval_group_individual_no_family_sharing",
            backend,
            self.unified_ids,
            timing=dict(group_timing),
        )
        return [int(value) for value in output_ids]

    setattr(UnifiedTransformGroup, "_node_bench_original_evaluate_unified", original)
    setattr(UnifiedTransformGroup, "evaluate_unified", evaluate_unified_individual)
    setattr(UnifiedTransformGroup, "_node_bench_individual_eval_installed", True)
    return original


def _restore_unified_individual_eval_ablation(original) -> None:
    if original is None:
        return
    setattr(UnifiedTransformGroup, "evaluate_unified", original)
    setattr(UnifiedTransformGroup, "_node_bench_individual_eval_installed", False)


def _bench_path(
    *,
    backend: str,
    network: str,
    case_name: str,
    path_kind: str,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    case = _case_by_name(str(network), str(case_name))
    provider = _is_provider_path(str(path_kind))
    no_family_ablation = str(path_kind) in NO_FAMILY_PATHS
    no_tile_family_ablation = str(path_kind) == "provider_no_tile_family_sharing"
    previous_no_family_env = os.environ.get("ORION_UNIFIED_LT_INDIVIDUAL_EVAL")
    previous_no_tile_family_env = os.environ.get("ORION_U22_DISABLE_TILE_FAMILY_SHARING")
    unified_eval_restore = None
    if bool(no_family_ablation):
        os.environ["ORION_UNIFIED_LT_INDIVIDUAL_EVAL"] = "1"
        unified_eval_restore = _install_unified_individual_eval_ablation()
    else:
        os.environ.pop("ORION_UNIFIED_LT_INDIVIDUAL_EVAL", None)
    if bool(no_tile_family_ablation):
        os.environ["ORION_U22_DISABLE_TILE_FAMILY_SHARING"] = "1"
    else:
        os.environ.pop("ORION_U22_DISABLE_TILE_FAMILY_SHARING", None)
    try:
        _require_backend(str(backend))
        _init_scheme(str(network), backend=str(backend))
        dag, attach_audit = _prepare_dag(str(network), provider=bool(provider))
        node_name = str(case["node"])
        if node_name not in dag.nodes:
            raise KeyError(f"node {node_name!r} not found in {network}")
        module = dag.nodes[node_name]["module"]
        runtime = getattr(module, "region_runtime", None)
        if bool(provider):
            if runtime is None:
                return {
                    "status": "unsupported",
                    "backend": str(backend),
                    "path": str(path_kind),
                    "network": str(network),
                    "case": str(case_name),
                    "node": node_name,
                    "reason": "no provider runtime attached",
                    "attach_audit": dict(attach_audit),
                    "module": _module_metadata(module),
                }
            if runtime is not None and not bool(getattr(runtime, "supports_scheme", lambda _scheme: True)(scheme)):
                return {
                    "status": "unsupported",
                    "backend": str(backend),
                    "path": str(path_kind),
                    "network": str(network),
                    "case": str(case_name),
                    "node": node_name,
                    "reason": f"provider runtime does not support active {backend} scheme",
                    "attach_audit": dict(attach_audit),
                    "module": _module_metadata(module),
                }

        stats_as_provider = bool(provider)
        source_path_kind = str(path_kind)
        memory_trace: list[dict[str, Any]] = [_memory_event("before_generate_diagonals", module)]

        generate_started = time.perf_counter()
        module.generate_diagonals(last=False)
        generate_diagonals_s = float(time.perf_counter() - generate_started)
        memory_trace.append(_memory_event("after_generate_diagonals", module))

        memory_trace.append(_memory_event("before_compile", module))
        compile_started = time.perf_counter()
        module.compile()
        compile_backend_s = float(time.perf_counter() - compile_started)
        module.he_mode = True
        memory_trace.append(_memory_event("after_compile", module))

        materialize_profile = _eager_materialize_dense_transforms(
            module,
            backend=str(backend),
            effective_path_kind=str(source_path_kind),
        )
        dense_materialize_compile_s = float(materialize_profile.get("seconds", 0.0) or 0.0)
        if dense_materialize_compile_s > 0.0:
            compile_backend_s += dense_materialize_compile_s
            memory_trace.append(_memory_event("after_dense_materialize_compile", module))

        rotation_stats = (
            _provider_rotation_stats(module, path_kind=str(path_kind))
            if bool(stats_as_provider)
            else _linear_transform_rotation_stats(module)
        )
        source_count = _source_count(module, path_kind=str(source_path_kind))
        output_ciphertext_count: int | None = None
        warmup_times: list[float] = []
        run_times: list[float] = []
        warmup_operation_counts: list[dict[str, int]] = []
        run_operation_counts: list[dict[str, int]] = []
        executor_timings: list[dict[str, float]] = []
        warmup_runtime_fairness: list[dict[str, Any]] = []
        run_runtime_fairness: list[dict[str, Any]] = []
        level = len(scheme.params.get_logq()) - 1
        gen = torch.Generator().manual_seed(int(case["seed"]) + (10000 if bool(stats_as_provider) else 0))
        for index in range(int(warmups) + int(repeats)):
            memory_trace.append(_memory_event(f"before_source_{index}", module))
            source = _make_source(
                module,
                count=int(source_count),
                gen=gen,
                level=int(level),
                provider=bool(stats_as_provider),
            )
            memory_trace.append(_memory_event(f"after_source_{index}", module))
            memory_trace.append(_memory_event(f"before_forward_{index}", module))
            counters_enabled = _reset_runtime_operation_counters()
            started = time.perf_counter()
            out = module(source)
            _synchronize_backend()
            elapsed = float(time.perf_counter() - started)
            runtime_fairness = _runtime_fairness_for_module(module, serving_hot_s=float(elapsed))
            operation_counts = _runtime_operation_counters() if bool(counters_enabled) else None
            memory_trace.append(_memory_event(f"after_forward_{index}", module))
            if output_ciphertext_count is None:
                output_ciphertext_count = int(len(getattr(out, "ids", [])))
            if int(index) < int(warmups):
                warmup_times.append(float(elapsed))
                warmup_runtime_fairness.append(dict(runtime_fairness))
                if operation_counts is not None:
                    warmup_operation_counts.append(dict(operation_counts))
            else:
                run_times.append(float(elapsed))
                run_runtime_fairness.append(dict(runtime_fairness))
                if operation_counts is not None:
                    run_operation_counts.append(dict(operation_counts))
                executor = getattr(getattr(module, "region_runtime", None), "executor", None)
                if executor is not None:
                    executor_timings.append(dict(getattr(executor, "last_runtime_timing", {})))
            del out
            del source

        first_runtime_counts = dict(run_operation_counts[0]) if run_operation_counts else {}
        runtime_operation_counts = {
            "available": bool(run_operation_counts),
            "source": "lattigo_runtime_operation_callbacks" if run_operation_counts else "not_available",
            "per_warmup": [dict(item) for item in warmup_operation_counts],
            "per_run": [dict(item) for item in run_operation_counts],
            "first_run": dict(first_runtime_counts),
            "mean_per_run": _mean_operation_counters(run_operation_counts),
        }
        runtime_fairness_timing = _mean_runtime_fairness(run_runtime_fairness)

        result = {
            "status": "ok",
            "backend": str(backend),
            "path": str(path_kind),
            "network": str(network),
            "network_label": str(NETWORK_SPECS[str(network)]["label"]),
            "case": str(case_name),
            "node": node_name,
            "op": str(case["op"]),
            "stage": str(case["stage"]),
            "multiplicity": int(case["multiplicity"]),
            "reason": "",
            "compile_once": True,
            "compile_s": float(generate_diagonals_s + compile_backend_s),
            "generate_diagonals_s": float(generate_diagonals_s),
            "compile_backend_s": float(compile_backend_s),
            "dense_materialize_compile_s": float(dense_materialize_compile_s),
            "dense_materialize_compile_profile": dict(materialize_profile),
            "warmup_count": int(warmups),
            "warmup_s": [float(value) for value in warmup_times],
            "run_count": int(repeats),
            "hot_run_s": [float(value) for value in run_times],
            "hot_run_mean_s": _fmean(run_times),
            "hot_run_stdev_s": _stdev(run_times),
            "hot_run_median_s": float(statistics.median(run_times)) if run_times else 0.0,
            "serving_hot_s": runtime_fairness_timing.get("serving_hot_s"),
            "resident_compute_s": runtime_fairness_timing.get("resident_compute_s"),
            "artifact_read_s": runtime_fairness_timing.get("artifact_read_s"),
            "artifact_load_s": runtime_fairness_timing.get("artifact_load_s"),
            "artifact_unload_s": runtime_fairness_timing.get("artifact_unload_s"),
            "trim_s": runtime_fairness_timing.get("trim_s"),
            "runtime_fairness_mode": str(runtime_fairness_timing.get("runtime_fairness_mode", "unknown")),
            "runtime_fairness_timing": dict(runtime_fairness_timing),
            "runtime_fairness_per_warmup": [dict(item) for item in warmup_runtime_fairness],
            "runtime_fairness_per_run": [dict(item) for item in run_runtime_fairness],
            "rotation_stats": rotation_stats,
            "rotation_eval_count": int(rotation_stats.get("rotation_eval_count", 0)),
            "runtime_operation_counts": runtime_operation_counts,
            "runtime_rotation_eval_count": int(first_runtime_counts.get("rotation", 0)),
            "runtime_lintrans_rotation_eval_count": int(first_runtime_counts.get("lintrans_rotation", 0)),
            "runtime_direct_rotation_eval_count": int(first_runtime_counts.get("direct_rotation", 0)),
            "runtime_conjugation_count": int(first_runtime_counts.get("conjugation", 0)),
            "source_ciphertext_count": int(source_count),
            "output_ciphertext_count": None if output_ciphertext_count is None else int(output_ciphertext_count),
            "module": _module_metadata(module),
            "attach_audit": dict(attach_audit) if bool(provider) else {},
            "no_family_audit": {
                "enabled": bool(no_family_ablation),
                "env": "ORION_UNIFIED_LT_INDIVIDUAL_EVAL=1" if bool(no_family_ablation) else "",
                "mode": (
                    "unified_lt_individual_eval_no_family_sharing_bsgs_preserved"
                    if bool(no_family_ablation)
                    else ""
                ),
            },
            "no_tile_family_audit": {
                "enabled": bool(no_tile_family_ablation),
                "env": (
                    "ORION_U22_DISABLE_TILE_FAMILY_SHARING=1"
                    if bool(no_tile_family_ablation)
                    else ""
                ),
                "mode": (
                    "u22_tconv_independent_tile_bsgs_no_family_cache"
                    if bool(no_tile_family_ablation)
                    else ""
                ),
            },
            "effective_backend_path": str(path_kind),
            "memory_trace": memory_trace,
            "unified_group_memory_trace": _unified_group_memory_traces(module),
            "orion_io_mode": str(scheme.params.get_io_mode()),
            "diags_path": str(scheme.params.get_diags_path()),
            "keys_path": str(scheme.params.get_keys_path()),
            "bounded_lattigo_dense_audit": {
                "enabled": _truthy_env_value(os.environ.get("ORION_NODE_BENCH_BOUNDED_LATTIGO_DENSE_ACTIVE")),
                "mode": str(os.environ.get("ORION_NODE_BENCH_BOUNDED_LATTIGO_DENSE_MODE_ACTIVE", "")),
                "compile_batch_transforms": str(os.environ.get("ORION_DENSE_LT_COMPILE_BATCH_TRANSFORMS", "")),
                "lt_compile_workers": str(os.environ.get("ORION_LT_COMPILE_WORKERS", "")),
                "lattigo_compile_workers": str(os.environ.get("ORION_LATTIGO_COMPILE_WORKERS", "")),
                "streaming_lt": str(os.environ.get("ORION_LATTIGO_STREAMING_LT", "")),
                "streaming_lt_chunk_plaintexts": str(
                    os.environ.get("ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS", "")
                ),
                "saved_io_prefetch_lookahead": str(os.environ.get("ORION_SAVED_IO_PREFETCH_LOOKAHEAD", "")),
            },
            "bounded_lattigo_provider_audit": {
                "enabled": _truthy_env_value(os.environ.get("ORION_NODE_BENCH_BOUNDED_LATTIGO_PROVIDER_ACTIVE")),
                "mode": str(os.environ.get("ORION_NODE_BENCH_BOUNDED_LATTIGO_PROVIDER_MODE_ACTIVE", "")),
                "compile_workers": str(os.environ.get("ORION_UNIFIED_COMPILE_WORKERS", "")),
                "stream_compile_batch_transforms": str(
                    os.environ.get("ORION_UNIFIED_STREAM_COMPILE_BATCH_TRANSFORMS", "")
                ),
                "stream_compile_batch_gb": str(os.environ.get("ORION_UNIFIED_STREAM_COMPILE_BATCH_GB", "")),
                "stream_load_plaintexts": str(os.environ.get("ORION_UNIFIED_LT_STREAM_LOAD_PLAINTEXTS", "")),
                "stream_load_chunk_gb": str(os.environ.get("ORION_UNIFIED_LT_STREAM_LOAD_CHUNK_GB", "")),
                "save_encoded_plaintexts": str(os.environ.get("ORION_UNIFIED_LT_SAVE_ENCODED_PLAINTEXTS", "")),
                "saved_io_prefetch_lookahead": str(os.environ.get("ORION_SAVED_IO_PREFETCH_LOOKAHEAD", "")),
                "eval_budget_bytes": str(os.environ.get("ORION_LATTIGO_UNIFIED_EVAL_BUDGET_BYTES", "")),
                "lattigo_streaming_lt": str(os.environ.get("ORION_LATTIGO_STREAMING_LT", "")),
                "lattigo_streaming_lt_chunk_plaintexts": str(
                    os.environ.get("ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS", "")
                ),
                "unified_stream_compile_io_none": str(
                    os.environ.get("ORION_UNIFIED_STREAM_COMPILE_IO_NONE", "")
                ),
            },
        }
        executor = getattr(getattr(module, "region_runtime", None), "executor", None)
        if executor is not None:
            result["executor"] = type(executor).__name__
            result["executor_hot_runtime_timings"] = executor_timings
            result["executor_last_runtime_timing"] = dict(getattr(executor, "last_runtime_timing", {}))
        return result
    finally:
        _cleanup_scheme()
        _cleanup_active_cheddar_io_dir()
        _restore_unified_individual_eval_ablation(unified_eval_restore)
        if previous_no_family_env is None:
            os.environ.pop("ORION_UNIFIED_LT_INDIVIDUAL_EVAL", None)
        else:
            os.environ["ORION_UNIFIED_LT_INDIVIDUAL_EVAL"] = str(previous_no_family_env)
        if previous_no_tile_family_env is None:
            os.environ.pop("ORION_U22_DISABLE_TILE_FAMILY_SHARING", None)
        else:
            os.environ["ORION_U22_DISABLE_TILE_FAMILY_SHARING"] = str(previous_no_tile_family_env)


def _case_by_name(network: str, case_name: str) -> dict[str, Any]:
    for case in NETWORK_SPECS[str(network)]["cases"]:
        if str(case["case"]) == str(case_name) or str(case["node"]) == str(case_name):
            return dict(case)
    raise KeyError(f"unknown case {case_name!r} for network {network!r}")


def _case_selected(network: str, case: dict[str, Any], filters: list[str]) -> bool:
    if not filters:
        return True
    names = {
        str(case["case"]),
        str(case["node"]),
        f"{network}:{case['case']}",
        f"{network}:{case['node']}",
    }
    return any(str(token) in names for token in filters)


def _selected_cases(network: str, filters: list[str]) -> list[dict[str, Any]]:
    return [dict(case) for case in NETWORK_SPECS[str(network)]["cases"] if _case_selected(str(network), dict(case), filters)]


def _speedup(numerator: float | int, denominator: float | int) -> float | None:
    denominator = float(denominator)
    if denominator == 0.0:
        return None
    return float(float(numerator) / denominator)


def _resident_runtime_value(result: dict[str, Any]) -> float | None:
    value = result.get("resident_compute_s")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _speedup_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return _speedup(float(left), float(right))


def _summarize_case(case_payload: dict[str, Any]) -> None:
    paths = dict(case_payload.get("paths", {}))
    dense_entry = dict(paths.get("dense", {}))
    provider_entry = dict(paths.get("provider", {}))
    if dense_entry.get("status") != "ok" or provider_entry.get("status") != "ok":
        case_payload["status"] = "partial"
        return
    dense = dict(dense_entry["result"])
    provider = dict(provider_entry["result"])
    if dense.get("status") != "ok" or provider.get("status") != "ok":
        case_payload["status"] = "partial"
        case_payload["dense"] = dense
        case_payload["provider"] = provider
        return
    dense_mean = float(dense["hot_run_mean_s"])
    provider_mean = float(provider["hot_run_mean_s"])
    dense_resident = _resident_runtime_value(dense)
    provider_resident = _resident_runtime_value(provider)
    dense_rotations = int(dense.get("rotation_eval_count", 0))
    provider_rotations = int(provider.get("rotation_eval_count", 0))
    dense_runtime_rotations = int(dense.get("runtime_rotation_eval_count", 0) or 0)
    provider_runtime_rotations = int(provider.get("runtime_rotation_eval_count", 0) or 0)

    deltas: dict[str, Any] = {
        "compile_s_provider_minus_dense": float(provider["compile_s"] - dense["compile_s"]),
        "hot_run_mean_s_provider_minus_dense": float(provider_mean - dense_mean),
        "resident_compute_s_provider_minus_dense": (
            None
            if dense_resident is None or provider_resident is None
            else float(provider_resident - dense_resident)
        ),
        "rotation_eval_count_provider_minus_dense": int(provider_rotations - dense_rotations),
        "runtime_rotation_eval_count_provider_minus_dense": int(
            provider_runtime_rotations - dense_runtime_rotations
        ),
    }
    speedups: dict[str, Any] = {
        "time_dense_over_provider": _speedup_or_none(dense_resident, provider_resident),
        "resident_compute_dense_over_provider": _speedup_or_none(dense_resident, provider_resident),
        "serving_hot_dense_over_provider": _speedup(dense_mean, provider_mean),
        "time_speedup_metric": "resident_compute_s",
        "rotation_dense_over_provider": _speedup(dense_rotations, provider_rotations),
        "runtime_rotation_dense_over_provider": _speedup(
            dense_runtime_rotations,
            provider_runtime_rotations,
        ),
        "compile_dense_over_provider": _speedup(float(dense["compile_s"]), float(provider["compile_s"])),
    }
    update_payload: dict[str, Any] = {
        "status": "ok",
        "dense": dense,
        "provider": provider,
        "delta": deltas,
        "speedup": speedups,
    }

    if "provider_no_family" in paths:
        no_family_entry = dict(paths.get("provider_no_family", {}))
        if no_family_entry.get("status") != "ok":
            case_payload.update(update_payload)
            case_payload["status"] = "partial"
            return
        no_family = dict(no_family_entry["result"])
        if no_family.get("status") != "ok":
            case_payload.update(update_payload)
            case_payload["provider_no_family"] = no_family
            case_payload["status"] = "partial"
            return
        no_family_mean = float(no_family["hot_run_mean_s"])
        no_family_resident = _resident_runtime_value(no_family)
        no_family_rotations = int(no_family.get("rotation_eval_count", 0))
        deltas.update(
            {
                "compile_s_provider_no_family_minus_dense": float(no_family["compile_s"] - dense["compile_s"]),
                "hot_run_mean_s_provider_no_family_minus_dense": float(no_family_mean - dense_mean),
                "resident_compute_s_provider_no_family_minus_dense": (
                    None
                    if dense_resident is None or no_family_resident is None
                    else float(no_family_resident - dense_resident)
                ),
                "rotation_eval_count_provider_no_family_minus_dense": int(no_family_rotations - dense_rotations),
                "compile_s_provider_no_family_minus_provider": float(no_family["compile_s"] - provider["compile_s"]),
                "hot_run_mean_s_provider_no_family_minus_provider": float(no_family_mean - provider_mean),
                "resident_compute_s_provider_no_family_minus_provider": (
                    None
                    if provider_resident is None or no_family_resident is None
                    else float(no_family_resident - provider_resident)
                ),
                "rotation_eval_count_provider_no_family_minus_provider": int(no_family_rotations - provider_rotations),
            }
        )
        speedups.update(
            {
                "time_dense_over_provider_no_family": _speedup_or_none(dense_resident, no_family_resident),
                "resident_compute_dense_over_provider_no_family": _speedup_or_none(dense_resident, no_family_resident),
                "serving_hot_dense_over_provider_no_family": _speedup(dense_mean, no_family_mean),
                "rotation_dense_over_provider_no_family": _speedup(dense_rotations, no_family_rotations),
                "compile_dense_over_provider_no_family": _speedup(float(dense["compile_s"]), float(no_family["compile_s"])),
                "time_provider_no_family_over_provider": _speedup_or_none(no_family_resident, provider_resident),
                "resident_compute_provider_no_family_over_provider": _speedup_or_none(
                    no_family_resident,
                    provider_resident,
                ),
                "serving_hot_provider_no_family_over_provider": _speedup(no_family_mean, provider_mean),
                "rotation_provider_no_family_over_provider": _speedup(no_family_rotations, provider_rotations),
                "compile_provider_no_family_over_provider": _speedup(float(no_family["compile_s"]), float(provider["compile_s"])),
            }
        )
        update_payload["provider_no_family"] = no_family

    if "provider_no_tile_family_sharing" in paths:
        no_tile_entry = dict(paths.get("provider_no_tile_family_sharing", {}))
        if no_tile_entry.get("status") != "ok":
            case_payload.update(update_payload)
            case_payload["status"] = "partial"
            return
        no_tile = dict(no_tile_entry["result"])
        if no_tile.get("status") != "ok":
            case_payload.update(update_payload)
            case_payload["provider_no_tile_family_sharing"] = no_tile
            case_payload["status"] = "partial"
            return
        no_tile_mean = float(no_tile["hot_run_mean_s"])
        no_tile_resident = _resident_runtime_value(no_tile)
        no_tile_rotations = int(no_tile.get("rotation_eval_count", 0))
        deltas.update(
            {
                "compile_s_provider_no_tile_family_sharing_minus_dense": float(no_tile["compile_s"] - dense["compile_s"]),
                "hot_run_mean_s_provider_no_tile_family_sharing_minus_dense": float(no_tile_mean - dense_mean),
                "resident_compute_s_provider_no_tile_family_sharing_minus_dense": (
                    None
                    if dense_resident is None or no_tile_resident is None
                    else float(no_tile_resident - dense_resident)
                ),
                "rotation_eval_count_provider_no_tile_family_sharing_minus_dense": int(no_tile_rotations - dense_rotations),
                "compile_s_provider_no_tile_family_sharing_minus_provider": float(no_tile["compile_s"] - provider["compile_s"]),
                "hot_run_mean_s_provider_no_tile_family_sharing_minus_provider": float(no_tile_mean - provider_mean),
                "resident_compute_s_provider_no_tile_family_sharing_minus_provider": (
                    None
                    if provider_resident is None or no_tile_resident is None
                    else float(no_tile_resident - provider_resident)
                ),
                "rotation_eval_count_provider_no_tile_family_sharing_minus_provider": int(no_tile_rotations - provider_rotations),
            }
        )
        speedups.update(
            {
                "time_dense_over_provider_no_tile_family_sharing": _speedup_or_none(dense_resident, no_tile_resident),
                "resident_compute_dense_over_provider_no_tile_family_sharing": _speedup_or_none(
                    dense_resident,
                    no_tile_resident,
                ),
                "serving_hot_dense_over_provider_no_tile_family_sharing": _speedup(dense_mean, no_tile_mean),
                "rotation_dense_over_provider_no_tile_family_sharing": _speedup(dense_rotations, no_tile_rotations),
                "compile_dense_over_provider_no_tile_family_sharing": _speedup(
                    float(dense["compile_s"]),
                    float(no_tile["compile_s"]),
                ),
                "time_provider_no_tile_family_sharing_over_provider": _speedup_or_none(
                    no_tile_resident,
                    provider_resident,
                ),
                "resident_compute_provider_no_tile_family_sharing_over_provider": _speedup_or_none(
                    no_tile_resident,
                    provider_resident,
                ),
                "serving_hot_provider_no_tile_family_sharing_over_provider": _speedup(no_tile_mean, provider_mean),
                "rotation_provider_no_tile_family_sharing_over_provider": _speedup(no_tile_rotations, provider_rotations),
                "compile_provider_no_tile_family_sharing_over_provider": _speedup(
                    float(no_tile["compile_s"]),
                    float(provider["compile_s"]),
                ),
            }
        )
        update_payload["provider_no_tile_family_sharing"] = no_tile

    case_payload.update(update_payload)


def _run_worker(
    *,
    backend: str,
    network: str,
    case_name: str,
    path_kind: str,
    repeats: int,
    warmups: int,
    timeout_s: int,
    logn_override: int | None = None,
    ckks_profile: str = "e2e",
    min_disk_free_gb: float = 0.0,
    disk_watchdog_interval_s: float = 15.0,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-backend",
        str(backend),
        "--worker-network",
        str(network),
        "--worker-case",
        str(case_name),
        "--worker-path",
        str(path_kind),
        "--repeats",
        str(int(repeats)),
        "--warmups",
        str(int(warmups)),
    ]
    if logn_override is not None:
        command.extend(["--logn-override", str(int(logn_override))])
    command.extend(["--ckks-profile", str(ckks_profile)])

    env = os.environ.copy()
    if str(path_kind) == "dense":
        env.setdefault("ORION_DENSE_LT_SHARED_CACHE", "0")
    worker_cheddar_io_root: Path | None = None
    worker_lattigo_dense_io_root: Path | None = None
    worker_lattigo_provider_io_root: Path | None = None
    log_dir: Path | None = None
    disk_watch_path: Path | None = None
    disk_before: dict[str, int | str] | None = None
    disk_after: dict[str, int | str] | None = None
    min_disk_free_bytes = 0
    max_worker_rss_bytes = int(_worker_rss_limit_bytes(env, backend=str(backend)))
    peak_worker_rss_bytes = 0
    bounded_lattigo_dense_payload: dict[str, Any] = {"enabled": False}
    bounded_lattigo_provider_payload: dict[str, Any] = {"enabled": False}
    if str(backend) == "cheddar":
        base_io_root = Path(env.get("ORION_CHEDDAR_IO_ROOT", tempfile.gettempdir()))
        base_io_root.mkdir(parents=True, exist_ok=True)
        prefix = (
            "worker_"
            f"{_safe_path_fragment(network)}_"
            f"{_safe_path_fragment(case_name)}_"
            f"{_safe_path_fragment(path_kind)}_"
        )
        worker_cheddar_io_root = Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_io_root)))
        env["ORION_CHEDDAR_IO_ROOT"] = str(worker_cheddar_io_root)
        disk_watch_path = worker_cheddar_io_root
        disk_before = _disk_usage_payload(base_io_root)
        min_disk_free_bytes = int(max(0.0, float(min_disk_free_gb)) * (1024**3))
        if min_disk_free_bytes > 0 and int(disk_before["free_bytes"]) < min_disk_free_bytes:
            shutil.rmtree(worker_cheddar_io_root, ignore_errors=True)
            return {
                "status": "failed",
                "failure_kind": "disk_watermark",
                "message": (
                    "Skipping Cheddar worker because free disk is below "
                    f"{float(min_disk_free_gb):.3g} GiB watermark."
                ),
                "worker_wall_s": 0.0,
                "command": command,
                "cheddar_io_root": str(worker_cheddar_io_root),
                "cheddar_io_root_cleaned": True,
                "disk_before": disk_before,
                "min_disk_free_bytes": int(min_disk_free_bytes),
            }

    if str(backend) == "lattigo" and _bounded_lattigo_dense_requested(
        network=str(network),
        case_name=str(case_name),
        path_kind=str(path_kind),
    ):
        try:
            bounded_mode = _bounded_lattigo_dense_mode()
        except ValueError as exc:
            return {
                "status": "failed",
                "failure_kind": "bounded_lattigo_dense_config",
                "message": str(exc),
                "worker_wall_s": 0.0,
                "command": command,
            }
        env["ORION_NODE_BENCH_BOUNDED_LATTIGO_DENSE_ACTIVE"] = "1"
        env["ORION_NODE_BENCH_BOUNDED_LATTIGO_DENSE_MODE_ACTIVE"] = str(bounded_mode)
        compile_workers = _bounded_lattigo_compile_workers_default()
        diagonal_encode_workers = _bounded_lattigo_diagonal_encode_workers_default()
        pack_workers = _bounded_lattigo_pack_workers_default()
        env.setdefault("ORION_PACK_CONV_WORKERS", pack_workers)
        env.setdefault("ORION_DENSE_LT_COMPILE_BATCH_TRANSFORMS", compile_workers)
        env.setdefault("ORION_LT_COMPILE_WORKERS", compile_workers)
        env.setdefault("ORION_LATTIGO_COMPILE_WORKERS", compile_workers)
        env.setdefault("ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS", diagonal_encode_workers)
        env.setdefault("ORION_DENSE_LT_HOST_PAYLOAD_CACHE", "0")
        bounded_lattigo_dense_payload = {
            "enabled": True,
            "mode": str(bounded_mode),
            "policy": str(env.get(BOUNDED_LATTIGO_DENSE_ENV, "auto")),
            "compile_parallel_policy": str(env.get("ORION_COMPILE_PARALLEL_POLICY", "auto")),
            "compile_parallel_policy_audit": policy_audit(),
            "compile_batch_transforms": str(env.get("ORION_DENSE_LT_COMPILE_BATCH_TRANSFORMS", "")),
            "pack_conv_workers": str(env.get("ORION_PACK_CONV_WORKERS", "")),
            "lt_compile_workers": str(env.get("ORION_LT_COMPILE_WORKERS", "")),
            "lattigo_compile_workers": str(env.get("ORION_LATTIGO_COMPILE_WORKERS", "")),
            "lattigo_diagonal_encode_workers": str(
                env.get("ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS", "")
            ),
        }
        if bounded_mode == "save":
            base_io_root = Path(env.get(LATTIGO_DENSE_IO_ROOT_ENV, str(REPO_ROOT / ".tmp" / "lattigo_dense_io")))
            base_io_root.mkdir(parents=True, exist_ok=True)
            prefix = (
                "worker_"
                f"{_safe_path_fragment(network)}_"
                f"{_safe_path_fragment(case_name)}_"
                f"{_safe_path_fragment(path_kind)}_"
            )
            worker_lattigo_dense_io_root = Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_io_root)))
            env[LATTIGO_BENCH_IO_MODE_ENV] = "save"
            env[LATTIGO_BENCH_DIAGS_PATH_ENV] = str(worker_lattigo_dense_io_root / "diagonals.h5")
            env[LATTIGO_BENCH_KEYS_PATH_ENV] = str(worker_lattigo_dense_io_root / "keys.h5")
            env.setdefault("ORION_SAVED_IO_PREFETCH_LOOKAHEAD", "1")
            env.setdefault("ORION_DENSE_LT_SHARED_CACHE", "0")
            disk_watch_path = worker_lattigo_dense_io_root
            disk_before = _disk_usage_payload(base_io_root)
            min_disk_free_bytes = int(max(0.0, float(min_disk_free_gb)) * (1024**3))
            bounded_lattigo_dense_payload.update(
                {
                    "io_root": str(worker_lattigo_dense_io_root),
                    "base_io_root": str(base_io_root),
                    "io_mode": "save",
                    "diags_path": str(worker_lattigo_dense_io_root / "diagonals.h5"),
                    "keys_path": str(worker_lattigo_dense_io_root / "keys.h5"),
                    "saved_io_prefetch_lookahead": str(env.get("ORION_SAVED_IO_PREFETCH_LOOKAHEAD", "")),
                    "dense_shared_cache": str(env.get("ORION_DENSE_LT_SHARED_CACHE", "")),
                }
            )
            if min_disk_free_bytes > 0 and int(disk_before["free_bytes"]) < min_disk_free_bytes:
                shutil.rmtree(worker_lattigo_dense_io_root, ignore_errors=True)
                return {
                    "status": "failed",
                    "failure_kind": "disk_watermark",
                    "message": (
                        "Skipping bounded Lattigo dense worker because free disk is below "
                        f"{float(min_disk_free_gb):.3g} GiB watermark."
                    ),
                    "worker_wall_s": 0.0,
                    "command": command,
                    "bounded_lattigo_dense": {
                        **bounded_lattigo_dense_payload,
                        "io_root_cleaned": True,
                    },
                    "disk_before": disk_before,
                    "min_disk_free_bytes": int(min_disk_free_bytes),
                }
        else:
            env.setdefault("ORION_LATTIGO_STREAMING_LT", "force")
            bounded_lattigo_dense_payload.update(
                {
                    "io_mode": "none",
                    "streaming_lt": str(env.get("ORION_LATTIGO_STREAMING_LT", "")),
                    "streaming_lt_chunk_plaintexts": str(
                        env.get("ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS", "")
                    ),
                    "streaming_lt_shared_transforms": str(
                        env.get("ORION_LATTIGO_STREAMING_LT_SHARED_TRANSFORMS", "")
                    ),
                }
            )

    if str(backend) == "lattigo" and _bounded_lattigo_provider_requested(
        network=str(network),
        case_name=str(case_name),
        path_kind=str(path_kind),
    ):
        try:
            provider_mode = _bounded_lattigo_provider_mode()
        except ValueError as exc:
            return {
                "status": "failed",
                "failure_kind": "bounded_lattigo_provider_config",
                "message": str(exc),
                "worker_wall_s": 0.0,
                "command": command,
            }
        base_io_root = Path(
            env.get(
                LATTIGO_PROVIDER_IO_ROOT_ENV,
                env.get(LATTIGO_DENSE_IO_ROOT_ENV, str(REPO_ROOT / ".tmp" / "lattigo_provider_io")),
            )
        )
        base_io_root.mkdir(parents=True, exist_ok=True)
        prefix = (
            "worker_"
            f"{_safe_path_fragment(network)}_"
            f"{_safe_path_fragment(case_name)}_"
            f"{_safe_path_fragment(path_kind)}_"
        )
        worker_lattigo_provider_io_root = Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_io_root)))
        env["ORION_NODE_BENCH_BOUNDED_LATTIGO_PROVIDER_ACTIVE"] = "1"
        env["ORION_NODE_BENCH_BOUNDED_LATTIGO_PROVIDER_MODE_ACTIVE"] = str(provider_mode)
        compile_workers = _bounded_lattigo_compile_workers_default()
        diagonal_encode_workers = _bounded_lattigo_diagonal_encode_workers_default()
        pack_workers = _bounded_lattigo_pack_workers_default()
        env.setdefault("ORION_PACK_CONV_WORKERS", pack_workers)
        env.setdefault("ORION_LT_COMPILE_WORKERS", compile_workers)
        env.setdefault("ORION_UNIFIED_COMPILE_WORKERS", compile_workers)
        env.setdefault("ORION_UNIFIED_STREAM_COMPILE_BATCH_TRANSFORMS", compile_workers)
        env.setdefault("ORION_UNIFIED_COMPILE_BATCH_TRANSFORMS", compile_workers)
        env.setdefault("ORION_UNIFIED_LOAD_BATCH_TRANSFORMS", compile_workers)
        env.setdefault("ORION_UNIFIED_CACHED_LOAD_BATCH_TRANSFORMS", compile_workers)
        env.setdefault("ORION_LATTIGO_COMPILE_WORKERS", compile_workers)
        env.setdefault("ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS", diagonal_encode_workers)
        env.setdefault("ORION_LATTIGO_BOOTSTRAP_WORKERS", compile_workers)
        env.setdefault("ORION_UNIFIED_STREAM_COMPILE_BATCH_GB", "2")
        env.setdefault("ORION_UNIFIED_LT_FORCE_COMPILE_TRIM_EACH_TRANSFORM", "0")
        env.setdefault("ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE", "1")
        if provider_mode == "save":
            env[LATTIGO_BENCH_IO_MODE_ENV] = "save"
            env[LATTIGO_BENCH_DIAGS_PATH_ENV] = str(worker_lattigo_provider_io_root / "unified_diagonals.h5")
            env[LATTIGO_BENCH_KEYS_PATH_ENV] = str(worker_lattigo_provider_io_root / "unified_keys.h5")
            env.setdefault("ORION_UNIFIED_LT_RELEASE_INDEX_ONLY_RAW_MATRICES_AFTER_SAVE", "1")
            env.setdefault("ORION_UNIFIED_LT_STREAM_LOAD_PLAINTEXTS", "1")
            env.setdefault("ORION_UNIFIED_LT_STREAM_LOAD_CHUNK_GB", "0.25")
            env.setdefault("ORION_UNIFIED_LT_SAVE_ENCODED_PLAINTEXTS", "0")
            env.setdefault("ORION_UNIFIED_LT_ROTKEY_RESIDENCY", "0")
            env.setdefault("ORION_UNIFIED_LT_PLAINTEXT_RESIDENCY", "0")
            env.setdefault("ORION_SAVED_IO_PREFETCH_LOOKAHEAD", "1")
            env.setdefault("ORION_LATTIGO_UNIFIED_EVAL_BUDGET_BYTES", str(48 * 1024**3))
            # Lattigo's streaming LT state is useful for dense/provider
            # io_mode=none, but save/load needs serialized plaintext payloads.
            env.setdefault("ORION_LATTIGO_STREAMING_LT", "0")
        else:
            env[LATTIGO_BENCH_IO_MODE_ENV] = "none"
            env.setdefault("ORION_UNIFIED_STREAM_COMPILE_IO_NONE", "1")
            env.setdefault("ORION_LATTIGO_STREAMING_LT", "force")
        disk_watch_path = worker_lattigo_provider_io_root
        disk_before = _disk_usage_payload(base_io_root)
        min_disk_free_bytes = int(max(0.0, float(min_disk_free_gb)) * (1024**3))
        bounded_lattigo_provider_payload = {
            "enabled": True,
            "mode": str(provider_mode),
            "policy": str(env.get(BOUNDED_LATTIGO_PROVIDER_ENV, "auto")),
            "compile_parallel_policy": str(env.get("ORION_COMPILE_PARALLEL_POLICY", "auto")),
            "compile_parallel_policy_audit": policy_audit(),
            "io_root": str(worker_lattigo_provider_io_root),
            "base_io_root": str(base_io_root),
            "io_mode": str(env.get(LATTIGO_BENCH_IO_MODE_ENV, "none")),
            "diags_path": str(env.get(LATTIGO_BENCH_DIAGS_PATH_ENV, "")),
            "keys_path": str(env.get(LATTIGO_BENCH_KEYS_PATH_ENV, "")),
            "pack_conv_workers": str(env.get("ORION_PACK_CONV_WORKERS", "")),
            "compile_workers": str(env.get("ORION_UNIFIED_COMPILE_WORKERS", "")),
            "stream_compile_batch_transforms": str(
                env.get("ORION_UNIFIED_STREAM_COMPILE_BATCH_TRANSFORMS", "")
            ),
            "stream_compile_batch_gb": str(env.get("ORION_UNIFIED_STREAM_COMPILE_BATCH_GB", "")),
            "lattigo_diagonal_encode_workers": str(
                env.get("ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS", "")
            ),
            "stream_load_plaintexts": str(env.get("ORION_UNIFIED_LT_STREAM_LOAD_PLAINTEXTS", "")),
            "stream_load_chunk_gb": str(env.get("ORION_UNIFIED_LT_STREAM_LOAD_CHUNK_GB", "")),
            "save_encoded_plaintexts": str(env.get("ORION_UNIFIED_LT_SAVE_ENCODED_PLAINTEXTS", "")),
            "saved_io_prefetch_lookahead": str(env.get("ORION_SAVED_IO_PREFETCH_LOOKAHEAD", "")),
            "eval_budget_bytes": str(env.get("ORION_LATTIGO_UNIFIED_EVAL_BUDGET_BYTES", "")),
            "lattigo_streaming_lt": str(env.get("ORION_LATTIGO_STREAMING_LT", "")),
            "lattigo_streaming_lt_chunk_plaintexts": str(
                env.get("ORION_LATTIGO_STREAMING_LT_CHUNK_PLAINTEXTS", "")
            ),
            "unified_stream_compile_io_none": str(env.get("ORION_UNIFIED_STREAM_COMPILE_IO_NONE", "")),
            "rss_limit_bytes": int(max_worker_rss_bytes),
        }
        if min_disk_free_bytes > 0 and int(disk_before["free_bytes"]) < min_disk_free_bytes:
            shutil.rmtree(worker_lattigo_provider_io_root, ignore_errors=True)
            return {
                "status": "failed",
                "failure_kind": "disk_watermark",
                "message": (
                    "Skipping bounded Lattigo provider worker because free disk is below "
                    f"{float(min_disk_free_gb):.3g} GiB watermark."
                ),
                "worker_wall_s": 0.0,
                "command": command,
                "bounded_lattigo_provider": {
                    **bounded_lattigo_provider_payload,
                    "io_root_cleaned": True,
                },
                "disk_before": disk_before,
                "min_disk_free_bytes": int(min_disk_free_bytes),
            }

    started = time.perf_counter()
    killed_reason = ""
    killed_disk: dict[str, int | str] | None = None
    killed_rss_bytes: int | None = None
    stdout = ""
    stderr = ""
    returncode: int | None = None
    try:
        if worker_cheddar_io_root is not None:
            log_parent = worker_cheddar_io_root
        elif worker_lattigo_dense_io_root is not None:
            log_parent = worker_lattigo_dense_io_root
        elif worker_lattigo_provider_io_root is not None:
            log_parent = worker_lattigo_provider_io_root
        else:
            log_parent = Path(tempfile.gettempdir())
        log_dir = Path(tempfile.mkdtemp(prefix="orion_node_worker_logs_", dir=str(log_parent)))
        stdout_path = log_dir / "stdout.txt"
        stderr_path = log_dir / "stderr.txt"
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            deadline = float(started + int(timeout_s))
            interval = max(1.0, float(disk_watchdog_interval_s))
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                now = time.perf_counter()
                if now >= deadline:
                    killed_reason = "timeout"
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    returncode = process.returncode
                    break
                if min_disk_free_bytes > 0 and disk_watch_path is not None:
                    disk_now = _disk_usage_payload(disk_watch_path)
                    if int(disk_now["free_bytes"]) < min_disk_free_bytes:
                        killed_reason = "disk_watermark"
                        killed_disk = disk_now
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        returncode = process.returncode
                        break
                if int(max_worker_rss_bytes) > 0:
                    rss_bytes = _process_rss_bytes(int(process.pid))
                    if rss_bytes is not None:
                        peak_worker_rss_bytes = max(int(peak_worker_rss_bytes), int(rss_bytes))
                        if int(rss_bytes) > int(max_worker_rss_bytes):
                            killed_reason = "rss_watermark"
                            killed_rss_bytes = int(rss_bytes)
                            process.terminate()
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            returncode = process.returncode
                            break
                sleep_s = min(interval, max(0.1, deadline - now))
                try:
                    returncode = process.wait(timeout=float(sleep_s))
                    break
                except subprocess.TimeoutExpired:
                    pass
        stdout = _read_text_for_json(stdout_path)
        stderr = _read_text_for_json(stderr_path)
        if disk_watch_path is not None:
            disk_after = _disk_usage_payload(disk_watch_path)
    finally:
        if log_dir is not None:
            shutil.rmtree(log_dir, ignore_errors=True)
        cleaned_io_root = False
        if worker_cheddar_io_root is not None and _truthy_env_value(env.get(CLEAN_CHEDDAR_IO_ENV)):
            shutil.rmtree(worker_cheddar_io_root, ignore_errors=True)
            cleaned_io_root = True
        cleaned_lattigo_dense_io_root = False
        if worker_lattigo_dense_io_root is not None and _clean_lattigo_dense_io_enabled(env):
            shutil.rmtree(worker_lattigo_dense_io_root, ignore_errors=True)
            cleaned_lattigo_dense_io_root = True
        cleaned_lattigo_provider_io_root = False
        if worker_lattigo_provider_io_root is not None and _clean_lattigo_provider_io_enabled(env):
            shutil.rmtree(worker_lattigo_provider_io_root, ignore_errors=True)
            cleaned_lattigo_provider_io_root = True

    worker_wall_s = float(time.perf_counter() - started)
    common_payload: dict[str, Any] = {
        "worker_wall_s": float(worker_wall_s),
        "command": command,
        "max_worker_rss_bytes": int(max_worker_rss_bytes),
        "peak_worker_rss_bytes": int(peak_worker_rss_bytes),
    }
    if worker_cheddar_io_root is not None:
        common_payload.update(
            {
                "cheddar_io_root": str(worker_cheddar_io_root),
                "cheddar_io_root_cleaned": bool(cleaned_io_root),
                "disk_before": disk_before,
                "disk_after": disk_after,
                "min_disk_free_bytes": int(min_disk_free_bytes),
            }
        )
    if bool(bounded_lattigo_dense_payload.get("enabled", False)):
        common_payload["bounded_lattigo_dense"] = {
            **bounded_lattigo_dense_payload,
            "io_root_cleaned": bool(cleaned_lattigo_dense_io_root),
            "disk_before": disk_before,
            "disk_after": disk_after,
            "min_disk_free_bytes": int(min_disk_free_bytes),
        }
    if bool(bounded_lattigo_provider_payload.get("enabled", False)):
        common_payload["bounded_lattigo_provider"] = {
            **bounded_lattigo_provider_payload,
            "io_root_cleaned": bool(cleaned_lattigo_provider_io_root),
            "disk_before": disk_before,
            "disk_after": disk_after,
            "min_disk_free_bytes": int(min_disk_free_bytes),
        }
    if killed_reason == "timeout":
        common_payload.update(
            {
                "status": "timeout",
                "timeout_s": int(timeout_s),
                "returncode": None if returncode is None else int(returncode),
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
        )
        return common_payload
    if killed_reason == "disk_watermark":
        common_payload.update(
            {
                "status": "failed",
                "failure_kind": "disk_watermark",
                "message": (
                    "Terminated worker because free disk dropped below "
                    f"{float(min_disk_free_gb):.3g} GiB watermark."
                ),
                "returncode": None if returncode is None else int(returncode),
                "disk_at_termination": killed_disk,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
        )
        return common_payload
    if killed_reason == "rss_watermark":
        common_payload.update(
            {
                "status": "failed",
                "failure_kind": "rss_watermark",
                "message": (
                    "Terminated worker because RSS exceeded "
                    f"{int(max_worker_rss_bytes)} bytes."
                ),
                "returncode": None if returncode is None else int(returncode),
                "rss_at_termination_bytes": 0 if killed_rss_bytes is None else int(killed_rss_bytes),
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
        )
        return common_payload
    result: dict[str, Any] | None = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if returncode != 0 or result is None:
        common_payload.update(
            {
                "status": "failed",
                "returncode": -999999 if returncode is None else int(returncode),
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
        )
        return common_payload
    common_payload.update(
        {
            "status": "ok",
            "result": result,
        }
    )
    return common_payload


def _list_cases_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "scope": "node-specific optimized CONV/TCONV cases",
        "backends": list(BACKENDS),
        "paths": {path: str(PATH_DESCRIPTIONS[path]) for path in PATHS},
        "networks": {
            network: {
                "label": str(spec["label"]),
                "dataset": str(spec["dataset"]),
                "ckks_profile": str(_ckks_profile()),
                "logn": int(_profile_ckks_spec(str(network), backend="lattigo")["logn"]),
                "default_logn": int(spec["logn"]),
                "e2e_logn": int(E2E_CKKS_SPECS[str(network)]["logn"]),
                "coverage_note": str(spec["coverage_note"]),
                "cases": [dict(case) for case in spec["cases"]],
            }
            for network, spec in NETWORK_SPECS.items()
        },
    }


def _flatten_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for network_payload in payload.get("networks", []):
        for case in network_payload.get("cases", []):
            if case.get("status") != "ok":
                continue
            dense = dict(case["dense"])
            provider = dict(case["provider"])
            row = {
                "backend": str(network_payload.get("backend", "")),
                "network": str(network_payload["network"]),
                "network_label": str(network_payload["label"]),
                "case": str(case["case"]),
                "node": str(case["node"]),
                "op": str(case["op"]),
                "dense_mean_s": float(dense["hot_run_mean_s"]),
                "provider_mean_s": float(provider["hot_run_mean_s"]),
                "dense_resident_compute_s": dense.get("resident_compute_s"),
                "provider_resident_compute_s": provider.get("resident_compute_s"),
                "dense_runtime_fairness_mode": dense.get("runtime_fairness_mode"),
                "provider_runtime_fairness_mode": provider.get("runtime_fairness_mode"),
                "time_speedup_dense_over_provider": case["speedup"]["time_dense_over_provider"],
                "serving_hot_speedup_dense_over_provider": case["speedup"].get("serving_hot_dense_over_provider"),
                "resident_compute_speedup_dense_over_provider": case["speedup"].get(
                    "resident_compute_dense_over_provider"
                ),
                "dense_rotations": int(dense["rotation_eval_count"]),
                "provider_rotations": int(provider["rotation_eval_count"]),
                "rotation_speedup_dense_over_provider": case["speedup"]["rotation_dense_over_provider"],
                "dense_runtime_rotations": int(dense.get("runtime_rotation_eval_count", 0) or 0),
                "provider_runtime_rotations": int(provider.get("runtime_rotation_eval_count", 0) or 0),
                "runtime_rotation_speedup_dense_over_provider": case["speedup"].get(
                    "runtime_rotation_dense_over_provider"
                ),
            }
            if "provider_no_family" in case:
                no_family = dict(case["provider_no_family"])
                row.update(
                    {
                        "provider_no_family_mean_s": float(no_family["hot_run_mean_s"]),
                        "time_speedup_dense_over_provider_no_family": case["speedup"][
                            "time_dense_over_provider_no_family"
                        ],
                        "time_provider_no_family_over_provider": case["speedup"][
                            "time_provider_no_family_over_provider"
                        ],
                        "provider_no_family_rotations": int(no_family["rotation_eval_count"]),
                        "rotation_speedup_dense_over_provider_no_family": case["speedup"][
                            "rotation_dense_over_provider_no_family"
                        ],
                        "rotation_provider_no_family_over_provider": case["speedup"][
                            "rotation_provider_no_family_over_provider"
                        ],
                        "provider_no_family_mode": str(no_family.get("no_family_audit", {}).get("mode", "")),
                    }
                )
            if "provider_no_tile_family_sharing" in case:
                no_tile = dict(case["provider_no_tile_family_sharing"])
                row.update(
                    {
                        "provider_no_tile_family_sharing_mean_s": float(no_tile["hot_run_mean_s"]),
                        "time_speedup_dense_over_provider_no_tile_family_sharing": case["speedup"][
                            "time_dense_over_provider_no_tile_family_sharing"
                        ],
                        "time_provider_no_tile_family_sharing_over_provider": case["speedup"][
                            "time_provider_no_tile_family_sharing_over_provider"
                        ],
                        "provider_no_tile_family_sharing_rotations": int(no_tile["rotation_eval_count"]),
                        "rotation_speedup_dense_over_provider_no_tile_family_sharing": case["speedup"][
                            "rotation_dense_over_provider_no_tile_family_sharing"
                        ],
                        "rotation_provider_no_tile_family_sharing_over_provider": case["speedup"][
                            "rotation_provider_no_tile_family_sharing_over_provider"
                        ],
                        "provider_no_tile_family_sharing_mode": str(
                            no_tile.get("no_tile_family_audit", {}).get("mode", "")
                        ),
                    }
                )
            rows.append(row)
    return rows


CSV_COLUMNS = (
    "backend",
    "network",
    "network_label",
    "model",
    "dataset",
    "base_dim",
    "ckks_profile",
    "ckks_logn",
    "ckks_logq_json",
    "ckks_logp_json",
    "ckks_logscale",
    "ckks_h",
    "boot_logp_json",
    "case",
    "node",
    "op",
    "stage",
    "multiplicity",
    "path",
    "case_status",
    "path_status",
    "reason",
    "effective_backend_path",
    "executor",
    "no_family_mode",
    "no_tile_family_mode",
    "compile_once",
    "compile_s",
    "generate_diagonals_s",
    "compile_backend_s",
    "dense_materialize_compile_s",
    "dense_materialize_compile_profile_json",
    "warmup_count",
    "run_count",
    "hot_run_mean_s",
    "hot_run_stdev_s",
    "hot_run_median_s",
    "hot_run_s_json",
    "serving_hot_s",
    "resident_compute_s",
    "artifact_read_s",
    "artifact_load_s",
    "artifact_unload_s",
    "trim_s",
    "runtime_fairness_mode",
    "runtime_fairness_timing_json",
    "rotation_eval_count",
    "runtime_rotation_eval_count",
    "runtime_lintrans_rotation_eval_count",
    "runtime_direct_rotation_eval_count",
    "runtime_conjugation_count",
    "runtime_operation_counts_json",
    "actual_rotation_callback_count",
    "reported_unique_key_union_count",
    "rotation_stats_source",
    "rotation_stats_method",
    "group_count",
    "transform_count",
    "transform_rotation_key_count_total",
    "shared_rotation_eval_count_total",
    "unique_rotation_key_count",
    "output_rotations_per_output_ct",
    "output_rotation_eval_count",
    "ct_pt_hybrid_packing",
    "tile_family_sharing",
    "bsgs_family_sharing",
    "input_block_pairs_json",
    "source_ciphertext_count",
    "output_ciphertext_count",
    "time_speedup_dense_over_provider",
    "serving_hot_speedup_dense_over_provider",
    "resident_compute_speedup_dense_over_provider",
    "rotation_speedup_dense_over_provider",
    "runtime_rotation_speedup_dense_over_provider",
    "compile_speedup_dense_over_provider",
    "time_speedup_dense_over_provider_no_family",
    "rotation_speedup_dense_over_provider_no_family",
    "compile_speedup_dense_over_provider_no_family",
    "time_provider_no_family_over_provider",
    "rotation_provider_no_family_over_provider",
    "compile_provider_no_family_over_provider",
    "time_speedup_dense_over_provider_no_tile_family_sharing",
    "rotation_speedup_dense_over_provider_no_tile_family_sharing",
    "compile_speedup_dense_over_provider_no_tile_family_sharing",
    "time_provider_no_tile_family_sharing_over_provider",
    "rotation_provider_no_tile_family_sharing_over_provider",
    "compile_provider_no_tile_family_sharing_over_provider",
    "module_input_shape",
    "module_output_shape",
    "module_fhe_input_shape",
    "module_fhe_output_shape",
    "module_input_gap",
    "module_output_gap",
    "memory_trace_json",
    "unified_group_memory_trace_json",
    "worker_wall_s",
    "command_json",
)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _flatten_csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for network_payload in payload.get("networks", []):
        backend = str(network_payload.get("backend", ""))
        config = dict(network_payload.get("config", {}) or {})
        ckks_params = dict(config.get("ckks_params", {}) or {})
        boot_params = dict(config.get("boot_params", {}) or {})
        for case in network_payload.get("cases", []):
            speedup = dict(case.get("speedup", {}) or {})
            for path_kind, path_entry in dict(case.get("paths", {}) or {}).items():
                row: dict[str, Any] = {
                    "backend": backend,
                    "network": str(network_payload.get("network", "")),
                    "network_label": str(network_payload.get("label", "")),
                    "model": str(network_payload.get("model", "")),
                    "dataset": str(network_payload.get("dataset", "")),
                    "base_dim": network_payload.get("base_dim", ""),
                    "ckks_profile": str(network_payload.get("ckks_profile", "")),
                    "ckks_logn": ckks_params.get("LogN", ""),
                    "ckks_logq_json": ckks_params.get("LogQ", ()),
                    "ckks_logp_json": ckks_params.get("LogP", ()),
                    "ckks_logscale": ckks_params.get("LogScale", ""),
                    "ckks_h": ckks_params.get("H", ""),
                    "boot_logp_json": boot_params.get("LogP", ()),
                    "case": str(case.get("case", "")),
                    "node": str(case.get("node", "")),
                    "op": str(case.get("op", "")),
                    "stage": str(case.get("stage", "")),
                    "multiplicity": case.get("multiplicity", ""),
                    "path": str(path_kind),
                    "case_status": str(case.get("status", "")),
                    "path_status": str(path_entry.get("status", "")),
                    "worker_wall_s": path_entry.get("worker_wall_s", ""),
                    "command_json": path_entry.get("command", ""),
                    "time_speedup_dense_over_provider": speedup.get("time_dense_over_provider"),
                    "serving_hot_speedup_dense_over_provider": speedup.get("serving_hot_dense_over_provider"),
                    "resident_compute_speedup_dense_over_provider": speedup.get(
                        "resident_compute_dense_over_provider"
                    ),
                    "rotation_speedup_dense_over_provider": speedup.get("rotation_dense_over_provider"),
                    "runtime_rotation_speedup_dense_over_provider": speedup.get(
                        "runtime_rotation_dense_over_provider"
                    ),
                    "compile_speedup_dense_over_provider": speedup.get("compile_dense_over_provider"),
                    "time_speedup_dense_over_provider_no_family": speedup.get("time_dense_over_provider_no_family"),
                    "rotation_speedup_dense_over_provider_no_family": speedup.get("rotation_dense_over_provider_no_family"),
                    "compile_speedup_dense_over_provider_no_family": speedup.get("compile_dense_over_provider_no_family"),
                    "time_provider_no_family_over_provider": speedup.get("time_provider_no_family_over_provider"),
                    "rotation_provider_no_family_over_provider": speedup.get("rotation_provider_no_family_over_provider"),
                    "compile_provider_no_family_over_provider": speedup.get("compile_provider_no_family_over_provider"),
                    "time_speedup_dense_over_provider_no_tile_family_sharing": speedup.get(
                        "time_dense_over_provider_no_tile_family_sharing"
                    ),
                    "rotation_speedup_dense_over_provider_no_tile_family_sharing": speedup.get(
                        "rotation_dense_over_provider_no_tile_family_sharing"
                    ),
                    "compile_speedup_dense_over_provider_no_tile_family_sharing": speedup.get(
                        "compile_dense_over_provider_no_tile_family_sharing"
                    ),
                    "time_provider_no_tile_family_sharing_over_provider": speedup.get(
                        "time_provider_no_tile_family_sharing_over_provider"
                    ),
                    "rotation_provider_no_tile_family_sharing_over_provider": speedup.get(
                        "rotation_provider_no_tile_family_sharing_over_provider"
                    ),
                    "compile_provider_no_tile_family_sharing_over_provider": speedup.get(
                        "compile_provider_no_tile_family_sharing_over_provider"
                    ),
                }
                result = dict(path_entry.get("result", {}) or {})
                if result:
                    rotation_stats = dict(result.get("rotation_stats", {}) or {})
                    module = dict(result.get("module", {}) or {})
                    no_family_audit = dict(result.get("no_family_audit", {}) or {})
                    no_tile_family_audit = dict(result.get("no_tile_family_audit", {}) or {})
                    row.update(
                        {
                            "path_status": str(result.get("status", row["path_status"])),
                            "reason": str(result.get("reason", path_entry.get("reason", ""))),
                            "effective_backend_path": str(result.get("effective_backend_path", "")),
                            "executor": str(result.get("executor", rotation_stats.get("executor", ""))),
                            "no_family_mode": str(no_family_audit.get("mode", "")),
                            "no_tile_family_mode": str(no_tile_family_audit.get("mode", "")),
                            "compile_once": result.get("compile_once", ""),
                            "compile_s": result.get("compile_s"),
                            "generate_diagonals_s": result.get("generate_diagonals_s"),
                            "compile_backend_s": result.get("compile_backend_s"),
                            "dense_materialize_compile_s": result.get("dense_materialize_compile_s"),
                            "dense_materialize_compile_profile_json": result.get("dense_materialize_compile_profile", {}),
                            "warmup_count": result.get("warmup_count"),
                            "run_count": result.get("run_count"),
                            "hot_run_mean_s": result.get("hot_run_mean_s"),
                            "hot_run_stdev_s": result.get("hot_run_stdev_s"),
                            "hot_run_median_s": result.get("hot_run_median_s"),
                            "hot_run_s_json": result.get("hot_run_s", ()),
                            "serving_hot_s": result.get("serving_hot_s"),
                            "resident_compute_s": result.get("resident_compute_s"),
                            "artifact_read_s": result.get("artifact_read_s"),
                            "artifact_load_s": result.get("artifact_load_s"),
                            "artifact_unload_s": result.get("artifact_unload_s"),
                            "trim_s": result.get("trim_s"),
                            "runtime_fairness_mode": result.get("runtime_fairness_mode"),
                            "runtime_fairness_timing_json": result.get("runtime_fairness_timing", {}),
                            "rotation_eval_count": result.get("rotation_eval_count"),
                            "runtime_rotation_eval_count": result.get("runtime_rotation_eval_count"),
                            "runtime_lintrans_rotation_eval_count": result.get(
                                "runtime_lintrans_rotation_eval_count"
                            ),
                            "runtime_direct_rotation_eval_count": result.get(
                                "runtime_direct_rotation_eval_count"
                            ),
                            "runtime_conjugation_count": result.get("runtime_conjugation_count"),
                            "runtime_operation_counts_json": result.get("runtime_operation_counts", {}),
                            "actual_rotation_callback_count": rotation_stats.get("actual_rotation_callback_count"),
                            "reported_unique_key_union_count": rotation_stats.get("reported_unique_key_union_count"),
                            "rotation_stats_source": str(rotation_stats.get("source", "")),
                            "rotation_stats_method": str(rotation_stats.get("method", "")),
                            "group_count": rotation_stats.get("group_count"),
                            "transform_count": rotation_stats.get("transform_count"),
                            "transform_rotation_key_count_total": rotation_stats.get("transform_rotation_key_count_total"),
                            "shared_rotation_eval_count_total": rotation_stats.get("shared_rotation_eval_count_total"),
                            "unique_rotation_key_count": rotation_stats.get("unique_rotation_key_count"),
                            "output_rotations_per_output_ct": rotation_stats.get("output_rotations_per_output_ct"),
                            "output_rotation_eval_count": rotation_stats.get("output_rotation_eval_count"),
                            "ct_pt_hybrid_packing": rotation_stats.get("ct_pt_hybrid_packing"),
                            "tile_family_sharing": rotation_stats.get("tile_family_sharing"),
                            "bsgs_family_sharing": rotation_stats.get("bsgs_family_sharing"),
                            "input_block_pairs_json": rotation_stats.get("input_block_pairs", ()),
                            "source_ciphertext_count": result.get("source_ciphertext_count"),
                            "output_ciphertext_count": result.get("output_ciphertext_count"),
                            "module_input_shape": module.get("input_shape", ()),
                            "module_output_shape": module.get("output_shape", ()),
                            "module_fhe_input_shape": module.get("fhe_input_shape", ()),
                            "module_fhe_output_shape": module.get("fhe_output_shape", ()),
                            "module_input_gap": module.get("input_gap"),
                            "module_output_gap": module.get("output_gap"),
                            "memory_trace_json": result.get("memory_trace", ()),
                            "unified_group_memory_trace_json": result.get("unified_group_memory_trace", ()),
                        }
                    )
                else:
                    row["reason"] = str(path_entry.get("reason", ""))
                rows.append({column: _csv_value(row.get(column, "")) for column in CSV_COLUMNS})
    return rows


def _write_csv(payload: dict[str, Any], csv_out: Path) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    rows = _flatten_csv_rows(payload)
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _worker_main(args: argparse.Namespace) -> int:
    payload = _bench_path(
        backend=str(args.worker_backend),
        network=str(args.worker_network),
        case_name=str(args.worker_case),
        path_kind=str(args.worker_path),
        repeats=int(args.repeats),
        warmups=int(args.warmups),
    )
    print(json.dumps(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Node-specific multi-backend dense/provider benchmark for unique optimized "
            "CONV/TCONV nodes in R18 Tiny, R34 ImageNet, and U22 base_dim=32."
        )
    )
    parser.add_argument("--backends", nargs="*", default=["lattigo"], choices=BACKENDS)
    parser.add_argument("--networks", nargs="*", default=list(NETWORK_SPECS.keys()), choices=tuple(NETWORK_SPECS.keys()))
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[],
        help="Optional case filters. Use case, node, network:case, or network:node.",
    )
    parser.add_argument("--paths", nargs="*", default=list(PATHS), choices=PATHS)
    parser.add_argument("--repeats", type=int, default=3, help="Timed forward runs per compiled path.")
    parser.add_argument("--warmups", type=int, default=0, help="Untimed forwards after compile, before timed repeats.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Write a long-format CSV with one row per backend/network/case/path. Defaults to --out with .csv suffix.",
    )
    parser.add_argument("--timeout-s", type=int, default=28800)
    parser.add_argument(
        "--min-disk-free-gb",
        type=float,
        default=float(os.environ.get(MIN_DISK_FREE_GB_ENV, "25")),
        help=(
            "For Cheddar workers, refuse to start or terminate the current worker when the IO filesystem "
            "drops below this free-space watermark. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--disk-watchdog-interval-s",
        type=float,
        default=float(os.environ.get(DISK_WATCHDOG_INTERVAL_S_ENV, "5")),
        help="Polling interval for the Cheddar worker disk-space watchdog.",
    )
    parser.add_argument(
        "--logn-override",
        type=int,
        default=None,
        help="Override LogN for selected node-benchmark networks and worker subprocesses.",
    )
    parser.add_argument(
        "--ckks-profile",
        choices=("e2e", "kernel"),
        default="e2e",
        help="Use full-graph E2E CKKS settings by default, or the older short-chain kernel settings.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--print-full", action="store_true", help="Print the full JSON payload to stdout instead of a compact summary.")
    parser.add_argument("--worker-backend", choices=BACKENDS, default="lattigo")
    parser.add_argument("--worker-network", choices=tuple(NETWORK_SPECS.keys()), default="")
    parser.add_argument("--worker-case", default="")
    parser.add_argument("--worker-path", choices=PATHS, default="dense")
    args = parser.parse_args()

    if args.logn_override is not None:
        os.environ[LOGN_OVERRIDE_ENV] = str(int(args.logn_override))
    os.environ[CKKS_PROFILE_ENV] = str(args.ckks_profile)

    if bool(args.list_cases):
        print(json.dumps(_list_cases_payload(), indent=2))
        return 0

    if str(args.worker_network):
        return _worker_main(args)

    selected_backends = [str(backend) for backend in args.backends]
    if not selected_backends:
        selected_backends = ["lattigo"]
    for backend in selected_backends:
        _require_backend(str(backend))
    if int(args.repeats) <= 0:
        raise ValueError("--repeats must be positive")
    if int(args.warmups) < 0:
        raise ValueError("--warmups must be non-negative")
    selected_paths = [str(path) for path in args.paths]
    if not selected_paths:
        selected_paths = list(PATHS)

    out_path = Path(args.out)
    csv_out = Path(args.csv_out) if args.csv_out is not None else out_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if bool(args.resume) and out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        payload["status"] = "running"
    else:
        payload = {
            "status": "running",
            "scope": (
                "node-specific multi-backend dense vs provider with real/imag no-hybrid and "
                "no-family-sharing ablations for unique optimized CONV/TCONV"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(REPO_ROOT),
            "python": sys.version,
            "platform": platform.platform(),
            "backends": selected_backends,
            "repeats": int(args.repeats),
            "warmups": int(args.warmups),
            "ckks_profile": str(args.ckks_profile),
            "logn_override": None if args.logn_override is None else int(args.logn_override),
            "min_disk_free_gb": float(args.min_disk_free_gb),
            "disk_watchdog_interval_s": float(args.disk_watchdog_interval_s),
            "compile_runs_per_path": 1,
            "rotation_count_note": (
                "rotation_eval_count is the callback-equivalent BSGS rotation count after planning: "
                "the default dense path counts each Orion submatrix transform independently; shared-cache "
                "dense evaluation is disabled unless ORION_DENSE_LT_SHARED_CACHE is explicitly enabled. "
                "Provider shared-cache paths count unique nonzero baby rotations per family plus nonzero "
                "giant rotations per transform, and dense output-fold rotations are added where the runtime still uses them. "
                "runtime_rotation_eval_count is measured during the timed Lattigo forward with runtime operation "
                "callbacks; reported_unique_key_union_count is retained only as an audit field for the old proxy."
            ),
            "paths": {path: str(PATH_DESCRIPTIONS[path]) for path in PATHS},
            "networks": [],
        }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    networks_by_key = {
        (str(item.get("backend", "lattigo")), str(item.get("network"))): item
        for item in payload.get("networks", [])
    }
    for backend in selected_backends:
        for network in [str(value) for value in args.networks]:
            spec = NETWORK_SPECS[str(network)]
            network_key = (str(backend), str(network))
            network_payload = networks_by_key.get(network_key)
            if network_payload is None:
                network_payload = {
                    "backend": str(backend),
                    "network": str(network),
                    "label": str(spec["label"]),
                    "model": str(spec["model"]),
                        "dataset": str(spec["dataset"]),
                        "base_dim": spec.get("base_dim"),
                        "ckks_profile": str(args.ckks_profile),
                        "logn": int(_profile_ckks_spec(str(network), backend=str(backend))["logn"]),
                        "default_logn": int(spec["logn"]),
                        "e2e_logn": int(E2E_CKKS_SPECS[str(network)]["logn"]),
                        "config": _build_config_payload(str(network), backend=str(backend)),
                    "coverage_note": str(spec["coverage_note"]),
                    "cases": [],
                }
                payload["networks"].append(network_payload)
                networks_by_key[network_key] = network_payload

            cases_by_name = {str(item.get("case")): item for item in network_payload.get("cases", [])}
            for case in _selected_cases(str(network), [str(token) for token in args.cases]):
                case_name = str(case["case"])
                case_payload = cases_by_name.get(case_name)
                if case_payload is None:
                    case_payload = {
                        "case": case_name,
                        "node": str(case["node"]),
                        "op": str(case["op"]),
                        "stage": str(case["stage"]),
                        "multiplicity": int(case["multiplicity"]),
                        "status": "running",
                        "paths": {},
                    }
                    network_payload["cases"].append(case_payload)
                    cases_by_name[case_name] = case_payload
                case_payload["status"] = "running"
                case_payload.setdefault("paths", {})
                out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                _write_csv(payload, csv_out)

                for path_kind in selected_paths:
                    existing = dict(case_payload["paths"].get(str(path_kind), {}))
                    if bool(args.resume) and existing.get("status") == "ok":
                        continue
                    case_payload["paths"][str(path_kind)] = {
                        "status": "running",
                        "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                    _write_csv(payload, csv_out)
                    case_payload["paths"][str(path_kind)] = _run_worker(
                        backend=str(backend),
                        network=str(network),
                        case_name=case_name,
                        path_kind=str(path_kind),
                        repeats=int(args.repeats),
                        warmups=int(args.warmups),
                        timeout_s=int(args.timeout_s),
                        logn_override=None if args.logn_override is None else int(args.logn_override),
                        ckks_profile=str(args.ckks_profile),
                        min_disk_free_gb=float(args.min_disk_free_gb),
                        disk_watchdog_interval_s=float(args.disk_watchdog_interval_s),
                    )
                    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                    _write_csv(payload, csv_out)

                _summarize_case(case_payload)
                out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                _write_csv(payload, csv_out)

    summary_rows = _flatten_summary(payload)
    payload["summary"] = {
        "successful_case_count": int(len(summary_rows)),
        "requested_case_count": int(
            len(selected_backends)
            * sum(len(_selected_cases(str(network), [str(token) for token in args.cases])) for network in args.networks)
        ),
        "rows": summary_rows,
    }
    payload["status"] = "ok" if int(payload["summary"]["successful_case_count"]) == int(payload["summary"]["requested_case_count"]) else "partial"
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["csv_out"] = str(csv_out)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload, csv_out)
    if bool(args.print_full):
        print(json.dumps(payload, indent=2))
    else:
        print(
            json.dumps(
                {
                    "status": str(payload["status"]),
                    "out": str(out_path),
                    "csv_out": str(csv_out),
                    "summary": payload["summary"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
