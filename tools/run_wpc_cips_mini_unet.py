#!/usr/bin/env python3
"""Validate an integrated encrypted WPC CIPS miniature U-Net block.

The graph composes all layout-changing operators required by a U-Net block:

    encoder Conv2d
      |------------------------------- skip --------------------|
      -> stride-2 Conv2d -> encrypted compact reshape            |
      -> bottleneck Conv2d -> identity bootstrap refresh          |
      -> ConvTranspose2d(2x2,stride=2) -> encrypted channel concat
      -> decoder Conv2d

Every learned transform has a full-Q/P correctness control and a compressed
Q/P execution.  The online graph performs no decrypt, clear repack, Encode, or
re-encryption.  Timings are diagnostic single observations only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import orion
from orion.experimental.wpc_cips_branches import WPCCIPSConcatPlan
from orion.experimental.wpc_cips_downsample import (
    WPCCIPSDownsampleReshapePlan,
    WPCCIPSStride2Conv2dPlan,
)
from orion.experimental.wpc_cips_layer import WPC_GLOBAL_STATS_FIELDS
from orion.experimental.wpc_cips_unet import WPCCIPSBootstrapRefresh
from orion.experimental.wpc_cips_upsample import WPCCIPSConvTranspose2dPlan
from orion.nn import Concat, Conv2d, ConvTranspose2d


DEFAULT_OUT = (
    REPO_ROOT
    / ".tmp/results/honours/20_wpc_mini_unet/"
    "mini_unet_pipeline.json"
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--logn", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--output-channels", type=int, default=8)
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--bound-headroom", type=float, default=1.25)
    parser.add_argument("--atol", type=float, default=3e-5)
    return parser


def _config(logn: int) -> dict[str, Any]:
    return {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [55, 45, 45, 45, 45, 45],
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


def _make_conv(
    *,
    name: str,
    input_channels: int,
    output_channels: int,
    stride: int,
    level: int,
    bsgs_ratio: float,
    weight: np.ndarray,
    bias: np.ndarray,
) -> Conv2d:
    layer = Conv2d(
        int(input_channels),
        int(output_channels),
        3,
        stride=int(stride),
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        bsgs_ratio=float(bsgs_ratio),
        level=int(level),
    )
    layer.name = str(name)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    layer.init_orion_params()
    return layer


def _make_upsample(
    *,
    name: str,
    channels: int,
    level: int,
    bsgs_ratio: float,
    weight: np.ndarray,
    bias: np.ndarray,
) -> ConvTranspose2d:
    layer = ConvTranspose2d(
        int(channels),
        int(channels),
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
    layer.name = str(name)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    layer.init_orion_params()
    return layer


def _operation_counters(backend: Any) -> dict[str, int]:
    values = [int(value) for value in backend.GetOperationCounters()]
    names = (
        "rotation_total",
        "linear_transform_rotation",
        "direct_rotation",
        "conjugation",
    )
    if len(values) != len(names):
        raise RuntimeError("unexpected operation-counter length")
    return {name: value for name, value in zip(names, values)}


def _global_stats(backend: Any) -> dict[str, Any]:
    values = [int(value) for value in backend.GetWPCCompressedGlobalStats()]
    if len(values) != len(WPC_GLOBAL_STATS_FIELDS):
        raise RuntimeError("unexpected WPC global-statistics length")
    result: dict[str, Any] = {
        name: int(value) for name, value in zip(WPC_GLOBAL_STATS_FIELDS, values)
    }
    full_bytes = int(result["aggregate_full_payload_bytes"])
    compressed_bytes = int(result["aggregate_compressed_payload_bytes"])
    stored_bytes = int(result["aggregate_stored_payload_plus_metadata_bytes"])
    peak_bytes = int(result["peak_materialized_full_payload_bytes"])
    result["aggregate_payload_compression_ratio"] = (
        float(full_bytes / compressed_bytes) if compressed_bytes else None
    )
    result["aggregate_storage_compression_ratio_including_metadata"] = (
        float(full_bytes / stored_bytes) if stored_bytes else None
    )
    result["aggregate_full_to_sequential_peak_ratio"] = (
        float(full_bytes / peak_bytes) if peak_bytes else None
    )
    return result


def _set_bootstrap_profile(backend: Any, enabled: bool) -> None:
    required = (
        "EnableBootstrapProfile",
        "ResetBootstrapProfile",
        "GetBootstrapProfileCounters",
    )
    if not all(callable(getattr(backend, name, None)) for name in required):
        raise RuntimeError("Lattigo backend does not expose bootstrap profiling")
    if enabled:
        backend.ResetBootstrapProfile()
        backend.EnableBootstrapProfile(1)
    else:
        backend.EnableBootstrapProfile(0)


def _bootstrap_profile(backend: Any) -> dict[str, Any]:
    values = [int(value) for value in backend.GetBootstrapProfileCounters()]
    width = len(BOOTSTRAP_PROFILE_FIELDS)
    if len(values) % width:
        raise RuntimeError("malformed Lattigo bootstrap profile")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(values), width):
        row = {
            name: int(value)
            for name, value in zip(
                BOOTSTRAP_PROFILE_FIELDS,
                values[start : start + width],
            )
        }
        for name in (
            "total_ns",
            "retrieve_ns",
            "copy_ns",
            "evaluator_bootstrap_ns",
            "postscale_ns",
            "push_ns",
        ):
            row[name.replace("_ns", "_s")] = float(row[name] / 1_000_000_000)
        rows.append(row)
    return {
        "row_count": int(len(rows)),
        "rows": rows,
        "total_s": float(sum(row["total_s"] for row in rows)),
        "evaluator_bootstrap_s": float(
            sum(row["evaluator_bootstrap_s"] for row in rows)
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if int(args.height) % 2 or int(args.width) % 2:
        raise SystemExit("height and width must be divisible by two")
    if float(args.bound_headroom) < 1.0:
        raise SystemExit("bound-headroom must be at least one")
    slots = 1 << (int(args.logn) - 1)
    high_capacity = int(slots // (int(args.height) * int(args.width)))
    low_capacity = int(slots // ((int(args.height) // 2) * (int(args.width) // 2)))
    if not (int(args.channels) > high_capacity and int(args.channels) <= low_capacity):
        raise SystemExit(
            "channels must require multiple high-resolution groups and one low-resolution group"
        )

    rng = np.random.default_rng(int(args.seed))
    input_values = rng.normal(
        0.0,
        0.15,
        size=(1, int(args.channels), int(args.height), int(args.width)),
    ).astype(np.float64)
    layer_specs = (
        ("wpc_unet_encoder", int(args.channels), int(args.channels), 1, 5),
        ("wpc_unet_downsample", int(args.channels), int(args.channels), 2, 4),
        ("wpc_unet_bottleneck", int(args.channels), int(args.channels), 1, 2),
        (
            "wpc_unet_decoder",
            2 * int(args.channels),
            int(args.output_channels),
            1,
            3,
        ),
    )
    weights = [
        rng.normal(0.0, 0.035, size=(out_c, in_c, 3, 3)).astype(np.float64)
        for _, in_c, out_c, _, _ in layer_specs
    ]
    biases = [
        rng.normal(0.0, 0.01, size=(out_c,)).astype(np.float64)
        for _, _, out_c, _, _ in layer_specs
    ]
    upsample_weight = rng.normal(
        0.0,
        0.035,
        size=(int(args.channels), int(args.channels), 2, 2),
    ).astype(np.float64)
    upsample_bias = rng.normal(
        0.0, 0.01, size=(int(args.channels),)
    ).astype(np.float64)

    scheme = orion.init_scheme(_config(int(args.logn)))
    Conv2d.set_scheme(scheme)
    ConvTranspose2d.set_scheme(scheme)
    Concat.set_scheme(scheme)
    conv_layers = [
        _make_conv(
            name=name,
            input_channels=in_c,
            output_channels=out_c,
            stride=stride,
            level=level,
            bsgs_ratio=float(args.bsgs_ratio),
            weight=weight,
            bias=bias,
        )
        for (name, in_c, out_c, stride, level), weight, bias in zip(
            layer_specs, weights, biases
        )
    ]
    upsample_layer = _make_upsample(
        name="wpc_unet_upsample",
        channels=int(args.channels),
        level=5,
        bsgs_ratio=float(args.bsgs_ratio),
        weight=upsample_weight,
        bias=upsample_bias,
    )
    concat_module = Concat(dim=1, bsgs_ratio=float(args.bsgs_ratio))
    concat_module.name = "wpc_unet_skip_concat"

    weight_plans: list[Any] = []
    reshape: WPCCIPSDownsampleReshapePlan | None = None
    refresh: WPCCIPSBootstrapRefresh | None = None
    concat_plan: WPCCIPSConcatPlan | None = None
    values_to_release: list[Any] = []
    payload: dict[str, Any] = {}
    accepted = False
    bootstrap_profile_enabled = False
    try:
        high_shape = (
            1,
            int(args.channels),
            int(args.height),
            int(args.width),
        )
        low_shape = (
            1,
            int(args.channels),
            int(args.height) // 2,
            int(args.width) // 2,
        )
        concat_shape = (
            1,
            2 * int(args.channels),
            int(args.height),
            int(args.width),
        )

        compile_started = time.perf_counter()
        encoder_plan = conv_layers[0].install_wpc_cips_plan(
            high_shape, include_full_control=True, verify_exact_qp=True
        )
        downsample_plan = conv_layers[1].install_wpc_cips_plan(
            high_shape, include_full_control=True, verify_exact_qp=True
        )
        if not isinstance(downsample_plan, WPCCIPSStride2Conv2dPlan):
            raise RuntimeError("stride-two encoder did not install its WPC plan")
        bottleneck_plan = conv_layers[2].install_wpc_cips_plan(
            low_shape, include_full_control=True, verify_exact_qp=True
        )
        upsample_plan = upsample_layer.install_wpc_cips_plan(
            low_shape, include_full_control=True, verify_exact_qp=True
        )
        if not isinstance(upsample_plan, WPCCIPSConvTranspose2dPlan):
            raise RuntimeError("decoder upsample did not install its WPC plan")
        decoder_plan = conv_layers[3].install_wpc_cips_plan(
            concat_shape, include_full_control=True, verify_exact_qp=True
        )
        weight_plans = [
            encoder_plan,
            downsample_plan,
            bottleneck_plan,
            upsample_plan,
            decoder_plan,
        ]

        reshape = WPCCIPSDownsampleReshapePlan(downsample_plan, level=3)
        reshape.compile(scheme)
        reshape.validate_consumer(bottleneck_plan)

        clear_encoder = encoder_plan.clear_reference(input_values)
        clear_downsample = downsample_plan.clear_reference(clear_encoder[None, ...])
        clear_bottleneck = bottleneck_plan.clear_reference(
            clear_downsample[None, ...]
        )
        bootstrap_bound = float(
            max(
                1.0,
                float(np.max(np.abs(clear_bottleneck)))
                * float(args.bound_headroom),
            )
        )
        refresh = WPCCIPSBootstrapRefresh.between_plans(
            bottleneck_plan,
            upsample_plan,
            bootstrap_bound=bootstrap_bound,
        )
        refresh_compile = refresh.compile(scheme)

        concat_plan = concat_module.install_wpc_cips_plan(
            (encoder_plan, upsample_plan),
            consumer_plan=decoder_plan,
        )
        if not isinstance(concat_plan, WPCCIPSConcatPlan):
            raise RuntimeError("skip concat did not install its WPC CIPS plan")
        for layer in (*conv_layers, upsample_layer):
            layer.he()
        concat_module.he()
        compile_s = float(time.perf_counter() - compile_started)

        clear_upsample = upsample_plan.clear_reference(clear_bottleneck[None, ...])
        clear_concat = np.concatenate((clear_encoder, clear_upsample), axis=0)
        clear_final = decoder_plan.clear_reference(clear_concat[None, ...])

        input_ciphertext = encoder_plan.encrypt_input(input_values)
        values_to_release.append(input_ciphertext)
        online_encode_call_count = 0
        original_encode = scheme.encoder.encode

        def counted_online_encode(*encode_args, **encode_kwargs):
            nonlocal online_encode_call_count
            online_encode_call_count += 1
            return original_encode(*encode_args, **encode_kwargs)

        scheme.encoder.encode = counted_online_encode
        try:
            refresh.clear_runtime_profile()
            _set_bootstrap_profile(scheme.backend, True)
            bootstrap_profile_enabled = True
            scheme.backend.ResetOperationCounters()
            full_started = time.perf_counter()
            full_encoder = encoder_plan.evaluate(input_ciphertext, compressed=False)
            values_to_release.append(full_encoder)
            full_encoder_output = encoder_plan.decrypt_unpack(full_encoder)
            full_downsample = downsample_plan.evaluate(full_encoder, compressed=False)
            values_to_release.append(full_downsample)
            full_downsample_output = downsample_plan.decrypt_unpack(full_downsample)
            full_compact = reshape.evaluate(full_downsample)
            values_to_release.append(full_compact)
            full_compact_output = reshape.decrypt_unpack(full_compact)
            full_bottleneck = bottleneck_plan.evaluate(full_compact, compressed=False)
            values_to_release.append(full_bottleneck)
            full_bottleneck_output = bottleneck_plan.decrypt_unpack(full_bottleneck)
            full_refreshed = refresh(full_bottleneck)
            values_to_release.append(full_refreshed)
            full_refresh_row = dict(refresh.last_evaluation)
            full_refreshed_output = refresh.decrypt_unpack(full_refreshed)
            full_upsample = upsample_plan.evaluate(full_refreshed, compressed=False)
            values_to_release.append(full_upsample)
            full_upsample_output = upsample_plan.decrypt_unpack(full_upsample)
            full_concat = concat_plan.evaluate(full_encoder, full_upsample)
            values_to_release.append(full_concat)
            full_concat_row = dict(concat_plan.last_evaluation)
            full_concat_output = concat_plan.decrypt_unpack(full_concat)
            full_final = decoder_plan.evaluate(full_concat, compressed=False)
            values_to_release.append(full_final)
            full_s = float(time.perf_counter() - full_started)
            full_output = decoder_plan.decrypt_unpack(full_final)
            full_counters = _operation_counters(scheme.backend)
            full_bootstrap_profile = _bootstrap_profile(scheme.backend)

            refresh.clear_runtime_profile()
            _set_bootstrap_profile(scheme.backend, True)
            scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()
            scheme.backend.ResetOperationCounters()
            compressed_started = time.perf_counter()
            compressed_encoder = conv_layers[0](input_ciphertext)
            values_to_release.append(compressed_encoder)
            compressed_encoder_row = dict(encoder_plan.last_evaluation)
            compressed_encoder_output = encoder_plan.decrypt_unpack(
                compressed_encoder
            )
            compressed_downsample = conv_layers[1](compressed_encoder)
            values_to_release.append(compressed_downsample)
            compressed_downsample_row = dict(downsample_plan.last_evaluation)
            compressed_downsample_output = downsample_plan.decrypt_unpack(
                compressed_downsample
            )
            compressed_compact = reshape.evaluate(compressed_downsample)
            values_to_release.append(compressed_compact)
            compressed_reshape_row = dict(reshape.last_evaluation)
            compressed_compact_output = reshape.decrypt_unpack(compressed_compact)
            compressed_bottleneck = conv_layers[2](compressed_compact)
            values_to_release.append(compressed_bottleneck)
            compressed_bottleneck_row = dict(bottleneck_plan.last_evaluation)
            compressed_bottleneck_output = bottleneck_plan.decrypt_unpack(
                compressed_bottleneck
            )
            compressed_refreshed = refresh(compressed_bottleneck)
            values_to_release.append(compressed_refreshed)
            compressed_refresh_row = dict(refresh.last_evaluation)
            compressed_refreshed_output = refresh.decrypt_unpack(
                compressed_refreshed
            )
            compressed_upsample = upsample_layer(compressed_refreshed)
            values_to_release.append(compressed_upsample)
            compressed_upsample_row = dict(upsample_plan.last_evaluation)
            compressed_upsample_output = upsample_plan.decrypt_unpack(
                compressed_upsample
            )
            compressed_concat = concat_module(
                compressed_encoder, compressed_upsample
            )
            values_to_release.append(compressed_concat)
            compressed_concat_row = dict(concat_plan.last_evaluation)
            compressed_concat_output = concat_plan.decrypt_unpack(compressed_concat)
            compressed_final = conv_layers[3](compressed_concat)
            values_to_release.append(compressed_final)
            compressed_decoder_row = dict(decoder_plan.last_evaluation)
            compressed_s = float(time.perf_counter() - compressed_started)
            compressed_output = decoder_plan.decrypt_unpack(compressed_final)
            compressed_counters = _operation_counters(scheme.backend)
            compressed_bootstrap_profile = _bootstrap_profile(scheme.backend)
        finally:
            scheme.encoder.encode = original_encode
            if bootstrap_profile_enabled:
                _set_bootstrap_profile(scheme.backend, False)
                bootstrap_profile_enabled = False

        global_stats = _global_stats(scheme.backend)
        compressed_stats = {
            plan.layer_name: plan.compressed_stats() for plan in weight_plans
        }
        runtime_rows = [
            row
            for layer_rows in compressed_stats.values()
            for row in layer_rows.values()
        ]
        transform_rows = [
            row for plan in weight_plans for row in plan.transform_rows.values()
        ]
        sequence = [
            *compressed_encoder_row["evaluation_sequence"],
            *compressed_downsample_row["evaluation_sequence"],
            *compressed_bottleneck_row["evaluation_sequence"],
            *compressed_upsample_row["evaluation_sequence"],
            *compressed_decoder_row["evaluation_sequence"],
        ]
        layer_storage = {
            plan.layer_name: plan.storage_summary() for plan in weight_plans
        }
        full_weight_bias = int(
            sum(
                row["full_weight_plus_bias_payload_bytes"]
                for row in layer_storage.values()
            )
        )
        stored_weight_bias = int(
            sum(
                row["stored_weight_plus_metadata_plus_bias_bytes"]
                for row in layer_storage.values()
            )
        )
        reshape_storage = reshape.storage_summary()
        concat_storage = concat_plan.storage_summary()
        ordinary_layout_bytes = int(
            reshape_storage["full_qp_payload_bytes"]
            + concat_storage["full_qp_payload_bytes"]
        )

        errors = {
            "full_encoder_max_abs_error": float(
                np.max(np.abs(full_encoder_output - clear_encoder))
            ),
            "compressed_encoder_max_abs_error": float(
                np.max(np.abs(compressed_encoder_output - clear_encoder))
            ),
            "full_downsample_max_abs_error": float(
                np.max(np.abs(full_downsample_output - clear_downsample))
            ),
            "compressed_downsample_max_abs_error": float(
                np.max(np.abs(compressed_downsample_output - clear_downsample))
            ),
            "full_compact_max_abs_error": float(
                np.max(np.abs(full_compact_output - clear_downsample))
            ),
            "compressed_compact_max_abs_error": float(
                np.max(np.abs(compressed_compact_output - clear_downsample))
            ),
            "full_bottleneck_max_abs_error": float(
                np.max(np.abs(full_bottleneck_output - clear_bottleneck))
            ),
            "compressed_bottleneck_max_abs_error": float(
                np.max(np.abs(compressed_bottleneck_output - clear_bottleneck))
            ),
            "full_refresh_max_abs_error": float(
                np.max(np.abs(full_refreshed_output - clear_bottleneck))
            ),
            "compressed_refresh_max_abs_error": float(
                np.max(np.abs(compressed_refreshed_output - clear_bottleneck))
            ),
            "full_upsample_max_abs_error": float(
                np.max(np.abs(full_upsample_output - clear_upsample))
            ),
            "compressed_upsample_max_abs_error": float(
                np.max(np.abs(compressed_upsample_output - clear_upsample))
            ),
            "full_concat_max_abs_error": float(
                np.max(np.abs(full_concat_output - clear_concat))
            ),
            "compressed_concat_max_abs_error": float(
                np.max(np.abs(compressed_concat_output - clear_concat))
            ),
            "full_final_max_abs_error": float(
                np.max(np.abs(full_output - clear_final))
            ),
            "compressed_final_max_abs_error": float(
                np.max(np.abs(compressed_output - clear_final))
            ),
            "compressed_vs_full_final_max_abs_delta": float(
                np.max(np.abs(compressed_output - full_output))
            ),
        }

        expected_transforms = int(
            sum(plan.case.transform_count for plan in weight_plans)
        )
        exact_qp = bool(
            len(transform_rows) == expected_transforms
            and all(row["exact_qp_match"] is True for row in transform_rows)
        )
        runtime_released = bool(
            len(runtime_rows) == expected_transforms
            and all(
                int(row["evaluation_count"]) == 1
                and int(row["materialized_full_payload_bytes"]) == 0
                and int(row["weight_plaintext_offline_encode_calls"]) == 1
                and int(row["weight_plaintext_online_encode_calls"]) == 0
                for row in runtime_rows
            )
        )
        sequence_isolated = bool(
            len(sequence) == expected_transforms
            and all(
                int(row["current_materialized_bytes_before"]) == 0
                and int(row["current_materialized_bytes_after"]) == 0
                and int(row["current_materialized_transforms_after"]) == 0
                for row in sequence
            )
        )
        peak_one = bool(
            int(global_stats["peak_materialized_transform_count"]) == 1
            and int(global_stats["peak_materialized_full_payload_bytes"])
            == int(global_stats["max_single_transform_full_payload_bytes"])
            and int(global_stats["current_materialized_transform_count"]) == 0
            and int(global_stats["current_materialized_full_payload_bytes"]) == 0
        )
        signatures_valid = bool(
            encoder_plan.output_packing_signature
            == downsample_plan.input_packing_signature
            and downsample_plan.output_packing_signature
            == reshape.input_packing_signature
            and reshape.output_packing_signature
            == bottleneck_plan.input_packing_signature
            and bottleneck_plan.output_packing_signature
            == refresh.input_packing_signature
            and refresh.output_packing_signature
            == upsample_plan.input_packing_signature
            and concat_plan.input_packing_signatures
            == (
                encoder_plan.output_packing_signature,
                upsample_plan.output_packing_signature,
            )
            and concat_plan.output_packing_signature
            == decoder_plan.input_packing_signature
        )
        levels_valid = bool(
            compressed_encoder_row["input_level"] == 5
            and compressed_encoder_row["output_level"] == 4
            and compressed_downsample_row["input_level"] == 4
            and compressed_downsample_row["output_level"] == 3
            and compressed_reshape_row["input_level"] == 3
            and compressed_reshape_row["output_level"] == 2
            and compressed_bottleneck_row["input_level"] == 2
            and compressed_bottleneck_row["output_level"] == 1
            and compressed_refresh_row["input_level"] == 1
            and compressed_refresh_row["output_level"] == 5
            and compressed_upsample_row["input_level"] == 5
            and compressed_upsample_row["output_level"] == 4
            and compressed_concat_row["input_level"] == 4
            and compressed_concat_row["output_level"] == 3
            and compressed_decoder_row["input_level"] == 3
            and compressed_decoder_row["output_level"] == 2
        )
        groups_valid = bool(
            len(input_ciphertext.ids) == 2
            and len(compressed_downsample.ids) == 2
            and len(compressed_compact.ids) == 1
            and len(compressed_refreshed.ids) == 1
            and len(compressed_upsample.ids) == 2
            and compressed_concat_row["input_ciphertext_group_count"] == 4
            and compressed_concat_row["output_ciphertext_group_count"] == 3
            and len(compressed_final.ids) == 1
        )
        bootstrap_valid = bool(
            full_bootstrap_profile["row_count"] == 1
            and compressed_bootstrap_profile["row_count"] == 1
            and full_refresh_row["packing_signature_preserved"]
            and compressed_refresh_row["packing_signature_preserved"]
            and compressed_refresh_row["clear_repack_or_encode_count"] == 0
        )
        correctness = bool(
            all(
                value <= float(args.atol)
                for key, value in errors.items()
                if "delta" not in key
            )
            and errors["compressed_vs_full_final_max_abs_delta"] == 0.0
        )
        storage_valid = bool(
            int(global_stats["registered_transform_count"]) == expected_transforms
            and int(global_stats["total_weight_plaintext_offline_encode_calls"])
            == expected_transforms
            and int(global_stats["total_weight_plaintext_online_encode_calls"]) == 0
            and stored_weight_bias < full_weight_bias
        )
        no_clear_repack = bool(
            compressed_reshape_row["clear_repack_or_encode_count"] == 0
            and compressed_refresh_row["clear_repack_or_encode_count"] == 0
            and compressed_concat_row["clear_repack_or_encode_count"] == 0
        )
        acceptance = {
            "actual_orion_encoder_decoder_modules_executed": bool(
                isinstance(downsample_plan, WPCCIPSStride2Conv2dPlan)
                and isinstance(upsample_plan, WPCCIPSConvTranspose2dPlan)
                and isinstance(concat_plan, WPCCIPSConcatPlan)
            ),
            "compressed_weight_qp_exactly_matches_full_control": exact_qp,
            "all_encrypted_intermediates_and_final_output_match_clear": correctness,
            "full_and_compressed_operation_counters_match": (
                full_counters == compressed_counters
            ),
            "packing_signatures_chain_across_complete_graph": signatures_valid,
            "ckks_level_schedule_is_valid": levels_valid,
            "ciphertext_group_transitions_are_valid": groups_valid,
            "bootstrap_refresh_preserves_cips_and_runs_once_per_path": bootstrap_valid,
            "zero_online_python_encode_calls": online_encode_call_count == 0,
            "all_layout_boundaries_report_zero_clear_repack_or_encode": no_clear_repack,
            "all_compressed_weight_materialization_released": runtime_released,
            "materialization_isolated_between_weight_transforms": sequence_isolated,
            "peak_weight_materialization_is_one_transform": peak_one,
            "storage_and_weight_encode_accounting_valid": storage_valid,
        }
        accepted = bool(all(acceptance.values()))
        acceptance["valid"] = accepted

        payload = {
            "schema_version": 1,
            "profile": "wpc_cips_integrated_mini_unet",
            "status": "ok" if accepted else "validation_failed",
            "seed": int(args.seed),
            "scope": {
                "graph": "encoder_skip_downsample_reshape_bottleneck_bootstrap_upsample_concat_decoder",
                "input_shape": list(input_values.shape),
                "low_resolution_shape": list(low_shape),
                "skip_shape": list(clear_encoder.shape),
                "concat_shape": list(clear_concat.shape),
                "output_shape": list(clear_final.shape),
                "compressed_weight_transform_count": expected_transforms,
                "limitations": [
                    "deterministic miniature graph rather than trained U-Net22",
                    "linear convolutions with an identity bootstrap refresh and no trained activation",
                    "concat and downsample reshape use ordinary offline Q/P permutation plaintexts",
                    "full-Q/P controls coexist only for correctness validation",
                    "logical plaintext accounting rather than process RSS",
                    "diagnostic single-run timing without performance claim",
                ],
            },
            "levels": {
                "encoder_convolution": [5, 4],
                "downsample_convolution": [4, 3],
                "encrypted_compact_reshape": [3, 2],
                "bottleneck_convolution": [2, 1],
                "bootstrap_refresh": [1, 5],
                "transposed_convolution": [5, 4],
                "skip_concat": [4, 3],
                "decoder_convolution": [3, 2],
            },
            "packing": {
                "encoder_output": encoder_plan.output_packing_signature,
                "sparse_downsample_output": downsample_plan.output_packing_signature,
                "compact_low_resolution": reshape.output_packing_signature,
                "refreshed_low_resolution": refresh.output_packing_signature,
                "upsample_output": upsample_plan.output_packing_signature,
                "concat_output": concat_plan.output_packing_signature,
                "decoder_input": decoder_plan.input_packing_signature,
            },
            "group_transitions": {
                "encrypted_input": len(input_ciphertext.ids),
                "sparse_downsample": len(compressed_downsample.ids),
                "compact_low_resolution": len(compressed_compact.ids),
                "refreshed_low_resolution": len(compressed_refreshed.ids),
                "upsampled_high_resolution": len(compressed_upsample.ids),
                "concat_input_total": compressed_concat_row[
                    "input_ciphertext_group_count"
                ],
                "concat_output": len(compressed_concat.ids),
                "decoder_output": len(compressed_final.ids),
            },
            "periodicity_and_exact_qp": {
                plan.layer_name: plan.transform_rows for plan in weight_plans
            },
            "layout_boundaries": {
                "downsample_reshape": {
                    "storage": reshape_storage,
                    "compressed_path_evaluation": compressed_reshape_row,
                },
                "bootstrap_refresh": {
                    "compile": refresh_compile,
                    "full_path_evaluation": full_refresh_row,
                    "compressed_path_evaluation": compressed_refresh_row,
                },
                "skip_concat": {
                    "storage": concat_storage,
                    "full_path_evaluation": full_concat_row,
                    "compressed_path_evaluation": compressed_concat_row,
                },
            },
            "storage": {
                "by_weight_layer": layer_storage,
                "full_weight_plus_bias_payload_bytes": full_weight_bias,
                "stored_weight_plus_metadata_plus_bias_bytes": stored_weight_bias,
                "weight_layer_storage_compression_ratio": float(
                    full_weight_bias / stored_weight_bias
                ),
                "ordinary_downsample_reshape_qp_payload_bytes": int(
                    reshape_storage["full_qp_payload_bytes"]
                ),
                "ordinary_concat_qp_payload_bytes": int(
                    concat_storage["full_qp_payload_bytes"]
                ),
                "ordinary_layout_qp_payload_bytes": ordinary_layout_bytes,
                "full_weight_and_layout_bytes": int(
                    full_weight_bias + ordinary_layout_bytes
                ),
                "stored_weight_and_layout_bytes": int(
                    stored_weight_bias + ordinary_layout_bytes
                ),
                "overall_logical_storage_compression_ratio": float(
                    (full_weight_bias + ordinary_layout_bytes)
                    / (stored_weight_bias + ordinary_layout_bytes)
                ),
            },
            "compressed_runtime_stats": compressed_stats,
            "global_stats_after_compressed_pipeline": global_stats,
            "bootstrap_profiles": {
                "full_control": full_bootstrap_profile,
                "compressed": compressed_bootstrap_profile,
            },
            "errors": errors,
            "operation_counters": {
                "full_control": full_counters,
                "compressed": compressed_counters,
            },
            "timings_s": {
                "compile": compile_s,
                "full_control_pipeline": full_s,
                "compressed_pipeline": compressed_s,
            },
            "online_python_encode_call_count": int(online_encode_call_count),
            "acceptance": acceptance,
            "timing_policy": "diagnostic_single_run_not_for_performance_claims",
        }
    finally:
        if bootstrap_profile_enabled:
            try:
                _set_bootstrap_profile(scheme.backend, False)
            except Exception:
                pass
        for value in reversed(values_to_release):
            release = getattr(value, "release", None)
            if callable(release):
                release()
        concat_module.remove_wpc_cips_plan()
        if refresh is not None:
            refresh.cleanup()
        if reshape is not None:
            reshape.cleanup()
        for layer in (*conv_layers, upsample_layer):
            layer.remove_wpc_cips_plan()
        scheme.delete_scheme()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(f"\nresult: {out_path}")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
