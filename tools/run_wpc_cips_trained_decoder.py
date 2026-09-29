#!/usr/bin/env python3
"""Validate a checkpoint-derived trained U-Net decoder stage in WPC CIPS.

The exact ``up1``, ``dec1a``, learned ``dec1a_act`` and ``dec1b`` parameters
are loaded from a medseg U-Net22-plus-output checkpoint.  The real encrypted
graph is:

    low-resolution feature -> up1 --+
                                  concat -> dec1a -> Cheb7 -> bootstrap -> dec1b
    high-resolution skip ---------+

Synthetic feature tensors isolate this decoder-stage correctness gate from
dataset preprocessing and earlier encoder error.  This is not an accuracy or
performance experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import orion
from orion.experimental.wpc_cips_branches import (
    WPCCIPSConcatPlan,
    pack_cips_groups,
)
from orion.experimental.wpc_cips_checkpoint import (
    CheckpointChebyshevSpec,
    WPCCIPSTrainedActivationBootstrap,
)
from orion.experimental.wpc_cips_layer import WPC_GLOBAL_STATS_FIELDS
from orion.experimental.wpc_cips_upsample import WPCCIPSConvTranspose2dPlan
from orion.nn import Concat, Conv2d, ConvTranspose2d


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/fhelipe_medseg_staged_covid19_256_scaled_silu_freeze15_cheb7_20260603/"
    "covid19_unet22_plus_output_base32_256_scaled_silu_avgpool_degree_7_"
    "rawgain_tight_g045_finetune_best.pt"
)
DEFAULT_OUT = (
    REPO_ROOT
    / ".tmp/results/honours/21_wpc_trained_decoder/"
    "trained_decoder_stage.json"
)

BOOTSTRAP_PROFILE_FIELDS = (
    "seq",
    "kind",
    "slots",
    "input_level",
    "output_level",
    "input_slots",
    "output_slots",
    "input_log_cols",
    "output_log_cols",
    "total_ns",
    "retrieve_ns",
    "copy_ns",
    "evaluator_bootstrap_ns",
    "postscale_ns",
    "push_ns",
    "input_degree",
    "output_degree",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--feature-std", type=float, default=0.02)
    parser.add_argument("--bound-headroom", type=float, default=1.25)
    parser.add_argument("--atol", type=float, default=2e-3)
    return parser


def _config(logn: int) -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [55] + [45] * 8,
            "LogP": [60],
            "LogScale": 45,
            "H": 192,
            "RingType": "Standard",
        },
        "boot_params": {"LogP": [61] * 8},
        "orion": {
            "margin": 1,
            "backend": "lattigo",
            "embedding_method": "square",
            "io_mode": "none",
            "debug": False,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model = dict(payload.get("model", {}) or {})
    expected_shapes = {
        "up1.weight": (64, 32, 2, 2),
        "up1.bias": (32,),
        "dec1a.weight": (32, 64, 3, 3),
        "dec1a.bias": (32,),
        "dec1a_act.coeffs": (8,),
        "dec1b.weight": (32, 32, 3, 3),
        "dec1b.bias": (32,),
    }
    missing = [name for name in expected_shapes if name not in state]
    wrong = {
        name: tuple(state[name].shape)
        for name, shape in expected_shapes.items()
        if name in state and tuple(state[name].shape) != shape
    }
    if missing or wrong:
        raise ValueError(
            f"checkpoint decoder tensors are incompatible: missing={missing}, wrong={wrong}"
        )
    if str(model.get("architecture")) != "unet22-plus-output":
        raise ValueError("checkpoint must use the unet22-plus-output architecture")
    if int(model.get("base_dim", -1)) != 32:
        raise ValueError("checkpoint must use base_dim=32")
    return state, model


def _make_up(
    state: dict[str, torch.Tensor], *, level: int, bsgs_ratio: float
) -> ConvTranspose2d:
    layer = ConvTranspose2d(
        64,
        32,
        2,
        stride=2,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        bsgs_ratio=float(bsgs_ratio),
        level=int(level),
    )
    layer.name = "checkpoint_up1"
    with torch.no_grad():
        layer.weight.copy_(state["up1.weight"])
        layer.bias.copy_(state["up1.bias"])
    layer.init_orion_params()
    return layer


def _make_conv(
    state: dict[str, torch.Tensor],
    name: str,
    *,
    input_channels: int,
    output_channels: int,
    level: int,
    bsgs_ratio: float,
) -> Conv2d:
    layer = Conv2d(
        int(input_channels),
        int(output_channels),
        3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        bsgs_ratio=float(bsgs_ratio),
        level=int(level),
    )
    layer.name = f"checkpoint_{name}"
    with torch.no_grad():
        layer.weight.copy_(state[f"{name}.weight"])
        layer.bias.copy_(state[f"{name}.bias"])
    layer.init_orion_params()
    return layer


def _operation_counters(backend: Any) -> dict[str, int]:
    names = (
        "rotation_total",
        "linear_transform_rotation",
        "direct_rotation",
        "conjugation",
    )
    values = [int(value) for value in backend.GetOperationCounters()]
    if len(values) != len(names):
        raise RuntimeError("unexpected operation-counter length")
    return dict(zip(names, values))


def _global_stats(backend: Any) -> dict[str, Any]:
    values = [int(value) for value in backend.GetWPCCompressedGlobalStats()]
    if len(values) != len(WPC_GLOBAL_STATS_FIELDS):
        raise RuntimeError("unexpected WPC global-statistics length")
    result = dict(zip(WPC_GLOBAL_STATS_FIELDS, values))
    full = int(result["aggregate_full_payload_bytes"])
    compressed = int(result["aggregate_compressed_payload_bytes"])
    stored = int(result["aggregate_stored_payload_plus_metadata_bytes"])
    peak = int(result["peak_materialized_full_payload_bytes"])
    result["aggregate_payload_compression_ratio"] = full / compressed
    result["aggregate_storage_compression_ratio_including_metadata"] = full / stored
    result["aggregate_full_to_sequential_peak_ratio"] = full / peak
    return result


def _bootstrap_profile(backend: Any) -> dict[str, Any]:
    values = [int(value) for value in backend.GetBootstrapProfileCounters()]
    width = len(BOOTSTRAP_PROFILE_FIELDS)
    if len(values) % width:
        raise RuntimeError("malformed Lattigo bootstrap profile")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(values), width):
        row = dict(
            zip(BOOTSTRAP_PROFILE_FIELDS, values[start : start + width])
        )
        for name in (
            "total_ns",
            "retrieve_ns",
            "copy_ns",
            "evaluator_bootstrap_ns",
            "postscale_ns",
            "push_ns",
        ):
            row[name.replace("_ns", "_s")] = row[name] / 1_000_000_000
        rows.append(row)
    return {
        "row_count": len(rows),
        "rows": rows,
        "total_s": float(sum(row["total_s"] for row in rows)),
    }


def _set_bootstrap_profile(backend: Any, enabled: bool) -> None:
    if enabled:
        backend.ResetBootstrapProfile()
        backend.EnableBootstrapProfile(1)
    else:
        backend.EnableBootstrapProfile(0)


def _release(value: Any) -> None:
    release = getattr(value, "release", None)
    if callable(release):
        release()


def _encrypt_packed(
    scheme: Any,
    values: np.ndarray,
    signature: tuple[Any, ...],
    *,
    level: int,
):
    messages = pack_cips_groups(values, signature)
    plaintext = scheme.encode(torch.tensor(messages, dtype=torch.float32), level=level)
    try:
        ciphertext = scheme.encrypt(plaintext)
    finally:
        plaintext.release()
    ciphertext._wpc_cips_packing_signature = signature
    return ciphertext


def _aggregate_storage(plans: list[Any], concat_plan: WPCCIPSConcatPlan) -> dict[str, Any]:
    rows = {plan.layer_name: plan.storage_summary() for plan in plans}
    full = sum(row["full_weight_plus_bias_payload_bytes"] for row in rows.values())
    stored = sum(
        row["stored_weight_plus_metadata_plus_bias_bytes"] for row in rows.values()
    )
    concat = concat_plan.storage_summary()
    return {
        "by_layer": rows,
        "learned_transform_count": sum(row["transform_count"] for row in rows.values()),
        "full_weight_plus_bias_payload_bytes": int(full),
        "stored_weight_plus_metadata_plus_bias_bytes": int(stored),
        "learned_weight_storage_ratio": float(full / stored),
        "concat_full_qp_payload_bytes": int(concat["full_qp_payload_bytes"]),
        "overall_full_to_stored_including_concat_ratio": float(
            full / (stored + concat["full_qp_payload_bytes"])
        ),
        "concat": concat,
    }


def _rotation_padding_conv2d(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Independent Torch implementation of WPC flattened Rotation Padding."""

    if value.ndim != 4 or weight.ndim != 4 or value.shape[0] != 1:
        raise ValueError("rotation-padding oracle expects NCHW tensors with N=1")
    _, _, height, width = value.shape
    output_channels, _, kernel_height, kernel_width = weight.shape
    flattened = value.reshape(1, value.shape[1], height * width)
    output = torch.zeros(
        (1, output_channels, height * width),
        dtype=value.dtype,
        device=value.device,
    )
    for kernel_row in range(kernel_height):
        for kernel_column in range(kernel_width):
            offset = (
                (kernel_row - kernel_height // 2) * width
                + kernel_column
                - kernel_width // 2
            )
            selected = torch.roll(flattened, shifts=-int(offset), dims=-1)
            output = output + torch.einsum(
                "oi,nih->noh",
                weight[:, :, kernel_row, kernel_column],
                selected,
            )
    return output.reshape(1, output_channels, height, width) + bias.reshape(
        1, -1, 1, 1
    )


def _torch_reference(
    state: dict[str, torch.Tensor],
    spec: CheckpointChebyshevSpec,
    low: np.ndarray,
    skip: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    dtype = torch.float64
    low_tensor = torch.tensor(low, dtype=dtype)
    skip_tensor = torch.tensor(skip, dtype=dtype)
    up = F.conv_transpose2d(
        low_tensor,
        state["up1.weight"].to(dtype),
        state["up1.bias"].to(dtype),
        stride=2,
    )
    concat = torch.cat((up, skip_tensor), dim=1)
    native_dec1a = F.conv2d(
        concat,
        state["dec1a.weight"].to(dtype),
        state["dec1a.bias"].to(dtype),
        padding=1,
    )
    native_activation = spec.evaluate(native_dec1a)
    native_dec1b = F.conv2d(
        native_activation,
        state["dec1b.weight"].to(dtype),
        state["dec1b.bias"].to(dtype),
        padding=1,
    )
    wpc_dec1a = _rotation_padding_conv2d(
        concat,
        state["dec1a.weight"].to(dtype),
        state["dec1a.bias"].to(dtype),
    )
    wpc_activation = spec.evaluate(wpc_dec1a)
    wpc_dec1b = _rotation_padding_conv2d(
        wpc_activation,
        state["dec1b.weight"].to(dtype),
        state["dec1b.bias"].to(dtype),
    )
    common = {
        "up1": up[0].numpy(),
        "concat": concat[0].numpy(),
    }
    return (
        {
            **common,
            "dec1a": wpc_dec1a[0].numpy(),
            "activation": wpc_activation[0].numpy(),
            "dec1b": wpc_dec1b[0].numpy(),
        },
        {
            **common,
            "dec1a": native_dec1a[0].numpy(),
            "activation": native_activation[0].numpy(),
            "dec1b": native_dec1b[0].numpy(),
        },
    )


def main() -> int:
    args = _parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
    if int(args.height) % 2 or int(args.width) % 2:
        raise SystemExit("height and width must be even")
    if float(args.bound_headroom) < 1.0:
        raise SystemExit("bound-headroom must be at least one")

    state, model_metadata = _load_checkpoint(checkpoint_path)
    activation_spec = CheckpointChebyshevSpec.from_state_dict(
        state, "dec1a_act"
    )
    slots = 1 << (int(args.logn) - 1)
    low_shape = (1, 64, int(args.height) // 2, int(args.width) // 2)
    high_shape = (1, 32, int(args.height), int(args.width))
    concat_shape = (1, 64, int(args.height), int(args.width))
    if slots != 512 or high_shape[2:] != (8, 8):
        raise SystemExit(
            "this checkpoint gate currently requires logn=10 and an 8x8 high-resolution stage"
        )

    rng = np.random.default_rng(int(args.seed))
    low = rng.normal(0.0, float(args.feature_std), size=low_shape).astype(np.float64)
    skip = rng.normal(0.0, float(args.feature_std), size=high_shape).astype(np.float64)
    torch_reference, native_checkpoint_reference = _torch_reference(
        state, activation_spec, low, skip
    )

    scheme = orion.init_scheme(_config(int(args.logn)))
    Conv2d.set_scheme(scheme)
    ConvTranspose2d.set_scheme(scheme)
    Concat.set_scheme(scheme)
    up_layer = _make_up(state, level=8, bsgs_ratio=float(args.bsgs_ratio))
    dec1a_layer = _make_conv(
        state,
        "dec1a",
        input_channels=64,
        output_channels=32,
        level=6,
        bsgs_ratio=float(args.bsgs_ratio),
    )
    dec1b_layer = _make_conv(
        state,
        "dec1b",
        input_channels=32,
        output_channels=32,
        level=8,
        bsgs_ratio=float(args.bsgs_ratio),
    )
    concat_module = Concat(dim=1, bsgs_ratio=float(args.bsgs_ratio))
    concat_module.name = "checkpoint_cat1"

    plans: list[Any] = []
    concat_plan: WPCCIPSConcatPlan | None = None
    bridge: WPCCIPSTrainedActivationBootstrap | None = None
    values: list[Any] = []
    payload: dict[str, Any] = {}
    profile_enabled = False
    try:
        compile_started = time.perf_counter()
        up_plan = up_layer.install_wpc_cips_plan(
            low_shape, include_full_control=True, verify_exact_qp=True
        )
        if not isinstance(up_plan, WPCCIPSConvTranspose2dPlan):
            raise RuntimeError("up1 did not install a transposed-convolution plan")
        dec1a_plan = dec1a_layer.install_wpc_cips_plan(
            concat_shape, include_full_control=True, verify_exact_qp=True
        )
        dec1b_plan = dec1b_layer.install_wpc_cips_plan(
            high_shape, include_full_control=True, verify_exact_qp=True
        )
        plans = [up_plan, dec1a_plan, dec1b_plan]
        skip_contract = SimpleNamespace(
            output_shape=high_shape,
            output_packing_signature=up_plan.output_packing_signature,
            output_level=7,
        )
        concat_plan = concat_module.install_wpc_cips_plan(
            (up_plan, skip_contract), consumer_plan=dec1a_plan
        )
        if not isinstance(concat_plan, WPCCIPSConcatPlan):
            raise RuntimeError("cat1 did not install a CIPS concat plan")

        clear_up = up_plan.clear_reference(low)
        clear_concat = np.concatenate((clear_up, skip[0]), axis=0)
        clear_dec1a = dec1a_plan.clear_reference(clear_concat[None, ...])
        clear_activation = (
            activation_spec.evaluate(torch.tensor(clear_dec1a[None], dtype=torch.float64))
            .numpy()[0]
        )
        clear_dec1b = dec1b_plan.clear_reference(clear_activation[None, ...])
        bound = max(
            1.0,
            float(np.max(np.abs(clear_activation))) * float(args.bound_headroom),
        )
        bridge = WPCCIPSTrainedActivationBootstrap(
            logical_shape=high_shape,
            packing_signature=dec1a_plan.output_packing_signature,
            input_level=int(dec1a_plan.output_level),
            output_level=int(dec1b_plan.level),
            activation_spec=activation_spec,
            bootstrap_bound=bound,
        )
        bridge_compile = bridge.compile(scheme)
        for layer in (up_layer, dec1a_layer, dec1b_layer):
            layer.he()
        concat_module.he()
        compile_s = float(time.perf_counter() - compile_started)

        oracle_errors = {
            "up1": float(np.max(np.abs(clear_up - torch_reference["up1"]))),
            "concat": float(
                np.max(np.abs(clear_concat - torch_reference["concat"]))
            ),
            "dec1a": float(
                np.max(np.abs(clear_dec1a - torch_reference["dec1a"]))
            ),
            "activation": float(
                np.max(np.abs(clear_activation - torch_reference["activation"]))
            ),
            "dec1b": float(
                np.max(np.abs(clear_dec1b - torch_reference["dec1b"]))
            ),
        }
        native_padding_drift = {
            name: float(
                np.max(
                    np.abs(
                        torch_reference[name] - native_checkpoint_reference[name]
                    )
                )
            )
            for name in ("dec1a", "activation", "dec1b")
        }

        low_ciphertext = up_plan.encrypt_input(low)
        values.append(low_ciphertext)
        skip_ciphertext = _encrypt_packed(
            scheme,
            skip[0],
            up_plan.output_packing_signature,
            level=7,
        )
        values.append(skip_ciphertext)
        online_encode_calls = 0
        original_encode = scheme.encoder.encode

        def counted_encode(*encode_args, **encode_kwargs):
            nonlocal online_encode_calls
            online_encode_calls += 1
            return original_encode(*encode_args, **encode_kwargs)

        scheme.encoder.encode = counted_encode
        try:
            bridge.clear_runtime_profile()
            _set_bootstrap_profile(scheme.backend, True)
            profile_enabled = True
            scheme.backend.ResetOperationCounters()
            full_started = time.perf_counter()
            full_up = up_plan.evaluate(low_ciphertext, compressed=False)
            values.append(full_up)
            full_concat = concat_plan.evaluate(full_up, skip_ciphertext)
            values.append(full_concat)
            full_dec1a = dec1a_plan.evaluate(full_concat, compressed=False)
            values.append(full_dec1a)
            full_activated = bridge(full_dec1a)
            values.append(full_activated)
            full_bridge_row = dict(bridge.last_evaluation)
            full_dec1b = dec1b_plan.evaluate(full_activated, compressed=False)
            values.append(full_dec1b)
            full_s = float(time.perf_counter() - full_started)
            full_counters = _operation_counters(scheme.backend)
            full_bootstrap = _bootstrap_profile(scheme.backend)

            bridge.clear_runtime_profile()
            _set_bootstrap_profile(scheme.backend, True)
            scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
            scheme.backend.ResetOperationCounters()
            compressed_started = time.perf_counter()
            compressed_up = up_layer(low_ciphertext)
            values.append(compressed_up)
            compressed_up_row = dict(up_plan.last_evaluation)
            compressed_concat = concat_module(compressed_up, skip_ciphertext)
            values.append(compressed_concat)
            compressed_concat_row = dict(concat_plan.last_evaluation)
            compressed_dec1a = dec1a_layer(compressed_concat)
            values.append(compressed_dec1a)
            compressed_dec1a_row = dict(dec1a_plan.last_evaluation)
            compressed_activated = bridge(compressed_dec1a)
            values.append(compressed_activated)
            compressed_bridge_row = dict(bridge.last_evaluation)
            compressed_dec1b = dec1b_layer(compressed_activated)
            values.append(compressed_dec1b)
            compressed_dec1b_row = dict(dec1b_plan.last_evaluation)
            compressed_s = float(time.perf_counter() - compressed_started)
            compressed_counters = _operation_counters(scheme.backend)
            compressed_bootstrap = _bootstrap_profile(scheme.backend)
        finally:
            scheme.encoder.encode = original_encode
            if profile_enabled:
                _set_bootstrap_profile(scheme.backend, False)
                profile_enabled = False

        full_outputs = {
            "up1": up_plan.decrypt_unpack(full_up),
            "concat": concat_plan.decrypt_unpack(full_concat),
            "dec1a": dec1a_plan.decrypt_unpack(full_dec1a),
            "activation": dec1a_plan.decrypt_unpack(full_activated),
            "dec1b": dec1b_plan.decrypt_unpack(full_dec1b),
        }
        compressed_outputs = {
            "up1": up_plan.decrypt_unpack(compressed_up),
            "concat": concat_plan.decrypt_unpack(compressed_concat),
            "dec1a": dec1a_plan.decrypt_unpack(compressed_dec1a),
            "activation": dec1a_plan.decrypt_unpack(compressed_activated),
            "dec1b": dec1b_plan.decrypt_unpack(compressed_dec1b),
        }
        clear_outputs = {
            "up1": clear_up,
            "concat": clear_concat,
            "dec1a": clear_dec1a,
            "activation": clear_activation,
            "dec1b": clear_dec1b,
        }
        errors = {
            path: {
                name: float(np.max(np.abs(output[name] - clear_outputs[name])))
                for name in clear_outputs
            }
            for path, output in (
                ("full_control", full_outputs),
                ("compressed", compressed_outputs),
            )
        }
        final_delta = float(
            np.max(np.abs(compressed_outputs["dec1b"] - full_outputs["dec1b"]))
        )
        global_stats = _global_stats(scheme.backend)
        storage = _aggregate_storage(plans, concat_plan)
        transform_rows = [
            row for plan in plans for row in plan.transform_rows.values()
        ]
        runtime_rows = [
            row for plan in plans for row in plan.compressed_stats().values()
        ]
        sequence = [
            *compressed_up_row["evaluation_sequence"],
            *compressed_dec1a_row["evaluation_sequence"],
            *compressed_dec1b_row["evaluation_sequence"],
        ]
        expected_transforms = sum(plan.case.transform_count for plan in plans)
        exact_qp = bool(
            len(transform_rows) == expected_transforms
            and all(row["exact_qp_match"] is True for row in transform_rows)
        )
        runtime_released = bool(
            len(runtime_rows) == expected_transforms
            and all(
                row["evaluation_count"] == 1
                and row["materialized_full_payload_bytes"] == 0
                and row["weight_plaintext_online_encode_calls"] == 0
                for row in runtime_rows
            )
        )
        peak_one = bool(
            global_stats["peak_materialized_transform_count"] == 1
            and global_stats["peak_materialized_full_payload_bytes"]
            == global_stats["max_single_transform_full_payload_bytes"]
            and global_stats["current_materialized_transform_count"] == 0
        )
        level_chain = bool(
            compressed_up_row["input_level"] == 8
            and compressed_up_row["output_level"] == 7
            and compressed_concat_row["input_level"] == 7
            and compressed_concat_row["output_level"] == 6
            and compressed_dec1a_row["input_level"] == 6
            and compressed_dec1a_row["output_level"] == 5
            and compressed_bridge_row["input_level"] == 5
            and compressed_bridge_row["activation_output_level"] == 1
            and compressed_bridge_row["bootstrap_output_level"] == 8
            and compressed_dec1b_row["input_level"] == 8
            and compressed_dec1b_row["output_level"] == 7
        )
        packing_chain = bool(
            up_plan.output_packing_signature
            == concat_plan.input_packing_signatures[0]
            == concat_plan.input_packing_signatures[1]
            and concat_plan.output_packing_signature
            == dec1a_plan.input_packing_signature
            and dec1a_plan.output_packing_signature
            == bridge.packing_signature
            == dec1b_plan.input_packing_signature
        )
        correctness = bool(
            max(max(row.values()) for row in errors.values()) <= float(args.atol)
            and final_delta == 0.0
        )
        acceptance = {
            "checkpoint_architecture_and_tensor_shapes_validated": True,
            "checkpoint_activation_coefficients_and_scales_used_exactly": True,
            "independent_torch_wpc_rotation_padding_and_cips_clear_oracles_match": max(
                oracle_errors.values()
            )
            <= 1e-10,
            "all_compressed_weight_qp_exactly_matches_full_control": exact_qp,
            "full_and_compressed_encrypted_stages_match_clear": correctness,
            "compressed_final_output_exactly_matches_full_control": final_delta == 0.0,
            "trained_cheb7_and_real_bootstrap_executed": bool(
                bridge_compile["activation_degree"] == 7
                and full_bootstrap["row_count"] == 4
                and compressed_bootstrap["row_count"] == 4
            ),
            "ckks_level_schedule_valid": level_chain,
            "cips_packing_chain_valid": packing_chain,
            "zero_online_python_encode_calls": online_encode_calls == 0,
            "all_layout_boundaries_report_zero_clear_repack_or_encode": bool(
                compressed_concat_row["clear_repack_or_encode_count"] == 0
                and compressed_bridge_row["clear_repack_or_encode_count"] == 0
            ),
            "full_and_compressed_operation_counters_match": full_counters
            == compressed_counters,
            "all_compressed_weight_materialization_released": runtime_released,
            "materialization_isolated_between_weight_transforms": bool(
                len(sequence) == expected_transforms
                and all(
                    row["current_materialized_bytes_before"] == 0
                    and row["current_materialized_bytes_after"] == 0
                    for row in sequence
                )
            ),
            "peak_weight_materialization_is_one_transform": peak_one,
            "learned_weight_storage_is_compressed": storage[
                "learned_weight_storage_ratio"
            ]
            > 1.0,
        }
        acceptance["valid"] = bool(all(acceptance.values()))

        payload = {
            "schema_version": 1,
            "profile": "wpc_cips_checkpoint_trained_unet_decoder_stage",
            "status": "ok" if acceptance["valid"] else "invalid",
            "timing_policy": "single diagnostic execution; no performance claim",
            "seed": int(args.seed),
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
                "model_metadata": model_metadata,
                "parameter_names": [
                    "up1.weight",
                    "up1.bias",
                    "dec1a.weight",
                    "dec1a.bias",
                    "dec1a_act.coeffs",
                    "dec1a_act.log_postscale",
                    "dec1a_act.log_prescale",
                    "dec1b.weight",
                    "dec1b.bias",
                ],
                "activation": bridge_compile,
            },
            "scope": {
                "graph": "up1 + skip1 -> cat1 -> dec1a -> trained dec1a_act -> bootstrap -> dec1b",
                "source_model": "UNet22PlusOutput base_dim=32",
                "spatial_shapes": {"low": list(low_shape), "high": list(high_shape)},
                "synthetic_feature_inputs": True,
                "limitations": [
                    "checkpoint-derived decoder-stage correctness gate, not the complete model",
                    "synthetic internal feature tensors rather than a dataset example",
                    "WPC flattened Rotation Padding differs from the checkpoint model's native zero padding; drift is measured explicitly",
                    "dec1b output is checked before its following activation",
                    "validation-only full transforms coexist with compressed transforms",
                    "single diagnostic timing run; no latency or accuracy claim",
                ],
            },
            "ckks": {
                "logn": int(args.logn),
                "slots": int(slots),
                "logq": [55] + [45] * 8,
                "logp": [60],
                "bootstrap_logp": [61] * 8,
                "level_schedule": {
                    "up1": [8, 7],
                    "cat1": [7, 6],
                    "dec1a": [6, 5],
                    "dec1a_act": [5, 1],
                    "bootstrap": [0, 8],
                    "dec1b": [8, 7],
                },
            },
            "compile_s": float(compile_s),
            "bootstrap_range": {
                "activation_min": float(np.min(clear_activation)),
                "activation_max": float(np.max(clear_activation)),
                "symmetric_bound": float(bound),
                "headroom": float(args.bound_headroom),
            },
            "clear_oracle_max_abs_delta": oracle_errors,
            "native_zero_padding_semantic_drift_max_abs_delta": native_padding_drift,
            "storage": storage,
            "compressed_global_stats": global_stats,
            "full_control": {
                "evaluate_s": float(full_s),
                "errors_vs_clear": errors["full_control"],
                "operation_counters": full_counters,
                "activation_bootstrap": full_bridge_row,
                "backend_bootstrap_profile": full_bootstrap,
            },
            "compressed": {
                "evaluate_s": float(compressed_s),
                "errors_vs_clear": errors["compressed"],
                "final_max_abs_delta_vs_full_control": float(final_delta),
                "operation_counters": compressed_counters,
                "operation_counters_match_full_control": full_counters
                == compressed_counters,
                "up1": compressed_up_row,
                "cat1": compressed_concat_row,
                "dec1a": compressed_dec1a_row,
                "activation_bootstrap": compressed_bridge_row,
                "dec1b": compressed_dec1b_row,
                "backend_bootstrap_profile": compressed_bootstrap,
                "online_python_encode_call_count": int(online_encode_calls),
            },
            "acceptance": acceptance,
        }
    finally:
        if profile_enabled:
            try:
                _set_bootstrap_profile(scheme.backend, False)
            except Exception:
                pass
        for value in reversed(values):
            _release(value)
        if bridge is not None:
            bridge.cleanup()
        concat_module.remove_wpc_cips_plan()
        for layer in (up_layer, dec1a_layer, dec1b_layer):
            layer.remove_wpc_cips_plan()
        scheme.delete_scheme()

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(f"\nresult: {out_path}")
    return 0 if payload["acceptance"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
