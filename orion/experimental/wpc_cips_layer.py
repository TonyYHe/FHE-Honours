"""Opt-in WPC CIPS planning for real Orion ``Conv2d`` layers.

The normal Orion Conv2d path deliberately remains unchanged.  This module
adapts one actual layer's weights and bias to the WPC CIPS/Rotation-Padding
layout, compiles one compressed transform per input/output ciphertext-group
pair, and exposes a grouped ``CipherTensor`` execution path that can be chained
across same-spatial-shape convolutions.
"""

from __future__ import annotations

import math
import weakref
from typing import Any

import numpy as np
import torch

from orion.backend.python.tensors import CipherTensor
from orion.experimental.wpc_cips_baseline import (
    CIPSConvCase,
    CIPSMultiGroupConvCase,
    LAYOUT_CIPS,
    build_cips_group_transforms,
    pack_tensor,
    rotation_padded_reference_multi_group,
    summarize_layout,
    unpack_output,
)
from orion.experimental.wpc_periodicity import analyze_slot_periodicity


WPC_COMPRESSED_STATS_FIELDS = (
    "diagonal_count",
    "full_payload_bytes",
    "compressed_payload_bytes",
    "metadata_bytes",
    "stored_payload_plus_metadata_bytes",
    "min_slot_period",
    "max_slot_period",
    "last_decompress_nanoseconds",
    "last_evaluate_nanoseconds",
    "weight_plaintext_offline_encode_calls",
    "weight_plaintext_online_encode_calls",
    "decompression_count",
    "evaluation_count",
    "materialized_full_payload_bytes",
    "backend_schema_version",
)

WPC_GLOBAL_STATS_FIELDS = (
    "registered_transform_count",
    "aggregate_full_payload_bytes",
    "aggregate_compressed_payload_bytes",
    "aggregate_metadata_bytes",
    "aggregate_stored_payload_plus_metadata_bytes",
    "current_materialized_full_payload_bytes",
    "peak_materialized_full_payload_bytes",
    "current_materialized_transform_count",
    "peak_materialized_transform_count",
    "max_single_transform_full_payload_bytes",
    "total_weight_plaintext_offline_encode_calls",
    "total_weight_plaintext_online_encode_calls",
    "backend_schema_version",
)

WPC_STORAGE_COMPRESSED = "compressed"
WPC_STORAGE_FULL = "full"
WPC_STORAGE_ONLINE = "online_encode"
WPC_STORAGE_MODES = (WPC_STORAGE_COMPRESSED, WPC_STORAGE_FULL, WPC_STORAGE_ONLINE)
WPC_ONLINE_STATS_FIELDS = (
    "diagonal_count", "full_payload_bytes", "recipe_payload_bytes", "metadata_bytes",
    "last_prepare_nanoseconds", "last_encode_nanoseconds", "last_evaluate_nanoseconds",
    "transform_encode_invocations",
)


def _tuple2(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    values = tuple(int(item) for item in value)
    if len(values) != 2:
        raise ValueError(f"expected a two-dimensional value, got {value!r}")
    return values


def _decode_stats(values: list[int], fields: tuple[str, ...]) -> dict[str, Any]:
    if len(values) != len(fields):
        raise RuntimeError(
            f"unexpected WPC statistics length: {len(values)} != {len(fields)}"
        )
    result: dict[str, Any] = {
        name: int(value) for name, value in zip(fields, values)
    }
    if "full_payload_bytes" in result and "compressed_payload_bytes" in result:
        full_bytes = int(result["full_payload_bytes"])
        compressed_bytes = int(result["compressed_payload_bytes"])
        stored_bytes = int(result["stored_payload_plus_metadata_bytes"])
        result["payload_compression_ratio"] = (
            float(full_bytes / compressed_bytes) if compressed_bytes else None
        )
        result["storage_compression_ratio_including_metadata"] = (
            float(full_bytes / stored_bytes) if stored_bytes else None
        )
        result["last_decompress_s"] = float(
            int(result["last_decompress_nanoseconds"]) / 1_000_000_000
        )
        result["last_evaluate_s"] = float(
            int(result["last_evaluate_nanoseconds"]) / 1_000_000_000
        )
    if "aggregate_full_payload_bytes" in result:
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


class WPCCIPSConv2dPlan:
    """Compiled WPC CIPS plan owned by one Orion Conv2d layer."""

    def __init__(
        self,
        layer: Any,
        *,
        input_shape: tuple[int, int, int, int] | torch.Size,
        slots: int,
    ) -> None:
        shape = tuple(int(value) for value in input_shape)
        if len(shape) != 4 or int(shape[0]) != 1:
            raise ValueError("WPC CIPS Conv2d requires input shape [1,C,H,W]")
        if int(shape[1]) != int(getattr(layer, "in_channels", -1)):
            raise ValueError("input shape channel count does not match Conv2d")
        kernel = _tuple2(getattr(layer, "kernel_size"))
        stride = _tuple2(getattr(layer, "stride"))
        padding = _tuple2(getattr(layer, "padding"))
        dilation = _tuple2(getattr(layer, "dilation"))
        if stride != (1, 1):
            raise ValueError("current WPC CIPS layer path supports stride one only")
        if dilation != (1, 1):
            raise ValueError("current WPC CIPS layer path supports dilation one only")
        if int(getattr(layer, "groups", 1)) != 1:
            raise ValueError("current WPC CIPS layer path supports groups=1 only")
        if any(int(size) % 2 == 0 for size in kernel):
            raise ValueError("current WPC Rotation Padding path requires odd kernels")
        if padding != (kernel[0] // 2, kernel[1] // 2):
            raise ValueError(
                "current WPC CIPS layer path requires same-shape symmetric padding"
            )

        height, width = int(shape[2]), int(shape[3])
        self._layer_ref = weakref.ref(layer)
        self.layer_name = str(getattr(layer, "name", layer.__class__.__name__))
        self.input_shape = shape
        self.output_shape = (
            1,
            int(getattr(layer, "out_channels")),
            height,
            width,
        )
        self.case = CIPSMultiGroupConvCase(
            slots=int(slots),
            input_channels=int(getattr(layer, "in_channels")),
            output_channels=int(getattr(layer, "out_channels")),
            height=height,
            width=width,
            kernel_height=int(kernel[0]),
            kernel_width=int(kernel[1]),
            pad_height_before=int(padding[0]),
            pad_width_before=int(padding[1]),
        )
        self.scheme: Any | None = None
        self.level: int | None = None
        self.output_level: int | None = None
        self.compressed_transform_ids: dict[str, int] = {}
        self.full_control_transform_ids: dict[str, int] = {}
        self.online_transform_ids: dict[str, int] = {}
        self.transform_rows: dict[str, dict[str, Any]] = {}
        self.bias_plaintext: Any | None = None
        self.bias_plaintext_payload_bytes = 0
        self.include_full_control = False
        self.storage_mode = WPC_STORAGE_COMPRESSED
        self.compiled = False
        self.cleaned = False
        self.last_evaluation: dict[str, Any] = {}
        self.record_sequence = True

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    @property
    def input_packing_signature(self) -> tuple[Any, ...]:
        return (
            "wpc_cips",
            int(self.case.slots),
            int(self.case.input_channels),
            int(self.case.height),
            int(self.case.width),
            tuple(tuple(value) for value in self.case.input_group_ranges),
        )

    @property
    def output_packing_signature(self) -> tuple[Any, ...]:
        return (
            "wpc_cips",
            int(self.case.slots),
            int(self.case.output_channels),
            int(self.case.height),
            int(self.case.width),
            tuple(tuple(value) for value in self.case.output_group_ranges),
        )

    def _layer(self) -> Any:
        layer = self._layer_ref()
        if layer is None:
            raise RuntimeError("the Orion Conv2d owning this WPC plan no longer exists")
        return layer

    def _weight_bias(self) -> tuple[np.ndarray, np.ndarray]:
        layer = self._layer()
        weight = getattr(layer, "on_weight", None)
        if weight is None:
            weight = getattr(layer, "weight").detach()
        bias = getattr(layer, "on_bias", None)
        if bias is None:
            raw_bias = getattr(layer, "bias", None)
            bias = (
                raw_bias.detach()
                if raw_bias is not None
                else torch.zeros(int(self.case.output_channels), dtype=torch.float32)
            )
        return (
            np.asarray(weight.detach().cpu(), dtype=np.float64),
            np.asarray(bias.detach().cpu(), dtype=np.float64),
        )

    def clear_reference(self, tensor: np.ndarray | torch.Tensor) -> np.ndarray:
        source = np.asarray(
            tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor,
            dtype=np.float64,
        )
        if tuple(source.shape) == self.input_shape:
            source = source[0]
        weight, bias = self._weight_bias()
        output = rotation_padded_reference_multi_group(source, weight, self.case)
        return output + bias[:, None, None]

    def _group_order(self) -> list[str]:
        return [
            f"out{output_group}_in{input_group}"
            for output_group in range(int(self.case.output_group_count))
            for input_group in range(int(self.case.input_group_count))
        ]

    def _build_group_transforms(
        self, weight: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        """Build the transforms used by this layout.

        The stride-two experimental planner overrides this hook because its
        output remains on the sparse down-sampled grid until the explicit
        reshape layer runs. Keeping the compilation/storage lifecycle here
        ensures both layouts use the same full-vs-compressed Q/P checks.
        """

        return build_cips_group_transforms(weight, self.case)

    @staticmethod
    def _flatten_diagonals(
        diagonals: dict[int, np.ndarray],
    ) -> tuple[list[int], list[float]]:
        indices = [int(value) for value in sorted(diagonals)]
        flattened = np.concatenate(
            [np.asarray(diagonals[index], dtype=np.float32) for index in indices]
        ).tolist()
        return indices, flattened

    @staticmethod
    def _common_period(diagonals: dict[int, np.ndarray]) -> int:
        periods = {
            int(analyze_slot_periodicity(value, payload_format="real").minimal_period)
            for value in diagonals.values()
        }
        return next(iter(periods)) if len(periods) == 1 else 0

    def _bias_messages(self, bias: np.ndarray) -> np.ndarray:
        messages: list[np.ndarray] = []
        for output_group, (output_start, output_end) in enumerate(
            self.case.output_group_ranges
        ):
            del output_group
            channel_count = int(output_end - output_start)
            bias_case = CIPSConvCase(
                slots=int(self.case.slots),
                input_channels=channel_count,
                output_channels=channel_count,
                height=int(self.case.height),
                width=int(self.case.width),
                kernel_height=1,
                kernel_width=1,
                pad_height_before=0,
                pad_width_before=0,
            )
            values = np.broadcast_to(
                bias[output_start:output_end, None, None],
                (channel_count, int(self.case.height), int(self.case.width)),
            )
            messages.append(pack_tensor(values, bias_case, LAYOUT_CIPS))
        return np.stack(messages, axis=0)

    def compile(
        self,
        scheme: Any,
        *,
        storage_mode: str = WPC_STORAGE_COMPRESSED,
        include_full_control: bool = False,
        verify_exact_qp: bool = True,
    ) -> dict[str, Any]:
        if self.compiled:
            raise RuntimeError("WPC CIPS Conv2d plan is already compiled")
        if self.cleaned:
            raise RuntimeError("a cleaned WPC CIPS Conv2d plan cannot be reused")
        storage_mode = str(storage_mode).strip().lower()
        if storage_mode not in WPC_STORAGE_MODES:
            raise ValueError(
                f"storage_mode must be one of {WPC_STORAGE_MODES}, got {storage_mode!r}"
            )
        if storage_mode != WPC_STORAGE_COMPRESSED and bool(include_full_control):
            raise ValueError("a full-storage plan is already the full control")
        if storage_mode != WPC_STORAGE_COMPRESSED and bool(verify_exact_qp):
            raise ValueError(
                "exact compressed-Q/P verification requires compressed storage"
            )
        required: list[str] = []
        if storage_mode == WPC_STORAGE_ONLINE:
            required.extend([
                "GenerateWPCOnlineLinearTransform", "EvaluateWPCOnlineLinearTransform",
                "GetWPCOnlineLinearTransformStats", "GetWPCOnlineGlobalStats",
                "ResetWPCOnlineMaterializationPeak",
            ])
        if storage_mode == WPC_STORAGE_COMPRESSED:
            required.extend(
                [
                    "GenerateWPCCompressedLinearTransform",
                    "EvaluateWPCCompressedLinearTransform",
                    "GetWPCCompressedLinearTransformStats",
                    "GetWPCCompressedGlobalStats",
                    "ResetWPCCompressedGlobalMaterializationPeak",
                ]
            )
            if bool(verify_exact_qp):
                required.extend(
                    [
                        "DecompressWPCLinearTransform",
                        "VerifyWPCDecompressedLinearTransformExact",
                        "RemoveWPCDecompressedLinearTransform",
                    ]
                )
        missing = [name for name in required if not hasattr(scheme.backend, name)]
        if missing:
            raise RuntimeError("Lattigo backend is missing WPC APIs: " + ", ".join(missing))

        layer = self._layer()
        if getattr(layer, "on_weight", None) is None:
            layer.init_orion_params()
        self.scheme = scheme
        self.level = int(
            getattr(layer, "level", None)
            if getattr(layer, "level", None) is not None
            else scheme.params.get_max_level()
        )
        if self.level <= 0:
            raise ValueError("WPC Conv2d needs at least one remaining rescale level")
        self.output_level = int(self.level - 1)
        self.include_full_control = bool(include_full_control)
        self.storage_mode = storage_mode
        weight, bias = self._weight_bias()
        group_rows = self._build_group_transforms(weight)
        full_bytes_per_diagonal = (
            int(self.level + 1 + len(scheme.params.get_logp()))
            * int(scheme.params.get_ring_degree())
            * 8
        )

        try:
            for key in self._group_order():
                diagonals = group_rows[key]["diagonals"]
                period = int(self._common_period(diagonals))
                summary = summarize_layout(
                    diagonals,
                    full_encoded_bytes_per_diagonal=int(full_bytes_per_diagonal),
                )
                if not (
                    period > 0
                    and period < int(self.case.slots)
                    and int(summary["periodicity"]["periodic_count"])
                    == int(summary["diagonal_count"])
                ):
                    raise RuntimeError(f"group transform {key} is not fully WPC-periodic")
                indices, flattened = self._flatten_diagonals(diagonals)

                full_id: int | None = None
                if (
                    storage_mode == WPC_STORAGE_FULL
                    or bool(verify_exact_qp)
                    or bool(include_full_control)
                ):
                    full_id = int(
                        scheme.backend.GenerateLinearTransform(
                            indices,
                            flattened,
                            int(self.level),
                            float(getattr(layer, "bsgs_ratio", 2.0)),
                            "none",
                        )
                    )
                    scheme.lt_evaluator.generate_rotation_keys(full_id)

                compressed_id: int | None = None
                if storage_mode == WPC_STORAGE_COMPRESSED:
                    compressed_id = int(
                        scheme.backend.GenerateWPCCompressedLinearTransform(
                            indices,
                            flattened,
                            int(self.level),
                            float(getattr(layer, "bsgs_ratio", 2.0)),
                            period,
                        )
                    )
                    scheme.lt_evaluator.generate_rotation_keys(compressed_id)
                    self.compressed_transform_ids[key] = compressed_id

                online_id: int | None = None
                if storage_mode == WPC_STORAGE_ONLINE:
                    online_id = int(scheme.backend.GenerateWPCOnlineLinearTransform(
                        indices, flattened, int(self.level),
                        float(getattr(layer, "bsgs_ratio", 2.0)), period,
                    ))
                    self.online_transform_ids[key] = online_id
                    scheme.lt_evaluator.generate_rotation_keys(online_id)

                exact_qp_match: bool | None = None
                manually_decompressed_count = 0
                if bool(verify_exact_qp):
                    if full_id is None or compressed_id is None:
                        raise RuntimeError("exact Q/P verification requires a full transform")
                    manually_decompressed_count = int(
                        scheme.backend.DecompressWPCLinearTransform(compressed_id)
                    )
                    exact_qp_match = bool(
                        int(
                            scheme.backend.VerifyWPCDecompressedLinearTransformExact(
                                full_id,
                                compressed_id,
                            )
                        )
                        == 1
                    )
                    scheme.backend.RemoveWPCDecompressedLinearTransform(compressed_id)
                    if not exact_qp_match:
                        raise RuntimeError(f"group transform {key} failed exact Q/P verification")

                if storage_mode == WPC_STORAGE_FULL or bool(include_full_control):
                    if full_id is None:
                        raise RuntimeError("full control transform was not generated")
                    self.full_control_transform_ids[key] = int(full_id)
                elif full_id is not None:
                    scheme.backend.DeleteLinearTransform(int(full_id))

                full_payload_bytes = int(
                    int(summary["diagonal_count"]) * full_bytes_per_diagonal
                )
                if compressed_id is not None:
                    stats = _decode_stats(
                        scheme.backend.GetWPCCompressedLinearTransformStats(
                            compressed_id
                        ),
                        WPC_COMPRESSED_STATS_FIELDS,
                    )
                    resident_payload_bytes = int(stats["compressed_payload_bytes"])
                    metadata_bytes = int(stats["metadata_bytes"])
                elif online_id is not None:
                    stats = _decode_stats(
                        scheme.backend.GetWPCOnlineLinearTransformStats(online_id),
                        WPC_ONLINE_STATS_FIELDS,
                    )
                    resident_payload_bytes = 0
                    metadata_bytes = int(stats["metadata_bytes"])
                else:
                    stats = {
                        "backend_schema_version": 1,
                        "diagonal_count": int(summary["diagonal_count"]),
                        "full_payload_bytes": full_payload_bytes,
                        "resident_payload_bytes": full_payload_bytes,
                        "metadata_bytes": 0,
                        "stored_payload_plus_metadata_bytes": full_payload_bytes,
                        "storage_mode": WPC_STORAGE_FULL,
                    }
                    resident_payload_bytes = full_payload_bytes
                    metadata_bytes = 0
                self.transform_rows[key] = {
                    "output_group": int(group_rows[key]["output_group"]),
                    "input_group": int(group_rows[key]["input_group"]),
                    "output_channel_range": group_rows[key]["output_channel_range"],
                    "input_channel_range": group_rows[key]["input_channel_range"],
                    "diagonal_count": int(summary["diagonal_count"]),
                    "slot_period": period,
                    "storage_mode": storage_mode,
                    "full_payload_bytes": full_payload_bytes,
                    "resident_payload_bytes": resident_payload_bytes,
                    "recipe_payload_bytes": int(stats.get("recipe_payload_bytes", 0)),
                    "metadata_bytes": metadata_bytes,
                    "exact_qp_match": exact_qp_match,
                    "manual_decompressed_diagonal_count": int(
                        manually_decompressed_count
                    ),
                    "stats_after_compile": stats,
                }

            if bool(np.any(bias != 0.0)):
                bias_messages = self._bias_messages(bias)
                self.bias_plaintext = scheme.encode(
                    torch.tensor(bias_messages, dtype=torch.float32),
                    level=int(self.output_level),
                )
                q_limb_count = int(self.output_level + 1)
                self.bias_plaintext_payload_bytes = int(
                    self.case.output_group_count
                    * q_limb_count
                    * int(scheme.params.get_ring_degree())
                    * 8
                )
            self.compiled = True
            self.cleaned = False
            return self.storage_summary()
        except Exception:
            self.cleanup()
            raise

    def storage_summary(self) -> dict[str, Any]:
        rows = list(self.transform_rows.values())
        full_weights = int(sum(int(row["full_payload_bytes"]) for row in rows))
        resident_weights = int(
            sum(int(row["resident_payload_bytes"]) for row in rows)
        )
        metadata = int(sum(int(row["metadata_bytes"]) for row in rows))
        recipes = int(sum(int(row.get("recipe_payload_bytes", 0)) for row in rows))
        full_including_bias = int(full_weights + self.bias_plaintext_payload_bytes)
        stored_including_bias = int(
            resident_weights + recipes + metadata + self.bias_plaintext_payload_bytes
        )
        return {
            "storage_mode": self.storage_mode,
            "transform_count": int(len(rows)),
            "full_weight_qp_payload_bytes": full_weights,
            "resident_weight_qp_payload_bytes": resident_weights,
            "compressed_weight_qp_payload_bytes": (
                resident_weights
                if self.storage_mode == WPC_STORAGE_COMPRESSED
                else 0
            ),
            "weight_metadata_bytes": metadata,
            "unencoded_recipe_payload_bytes": recipes,
            "uncompressed_bias_q_payload_bytes": int(
                self.bias_plaintext_payload_bytes
            ),
            "full_weight_plus_bias_payload_bytes": full_including_bias,
            "stored_weight_plus_metadata_plus_bias_bytes": stored_including_bias,
            "weight_payload_compression_ratio": (
                float(full_weights / resident_weights) if resident_weights else None
            ),
            "layer_plaintext_storage_compression_ratio": (
                float(full_including_bias / stored_including_bias)
                if stored_including_bias
                else None
            ),
        }

    def encrypt_input(self, tensor: np.ndarray | torch.Tensor) -> CipherTensor:
        if not self.compiled or self.scheme is None:
            raise RuntimeError("WPC CIPS plan must be compiled before input encryption")
        source = np.asarray(
            tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor,
            dtype=np.float64,
        )
        if tuple(source.shape) == self.input_shape:
            source = source[0]
        expected = (
            int(self.case.input_channels),
            int(self.case.height),
            int(self.case.width),
        )
        if tuple(source.shape) != expected:
            raise ValueError(f"input tensor shape is {tuple(source.shape)}, expected {expected}")
        messages = []
        for input_group, (start, end) in enumerate(self.case.input_group_ranges):
            local_case = self.case.local_case(0, input_group)
            messages.append(
                pack_tensor(source[start:end], local_case, LAYOUT_CIPS)
            )
        plaintext = self.scheme.encode(
            torch.tensor(np.stack(messages), dtype=torch.float32),
            level=int(self.level),
        )
        try:
            ciphertext = self.scheme.encrypt(plaintext)
        finally:
            plaintext.release()
        ciphertext._wpc_cips_packing_signature = self.input_packing_signature
        return ciphertext

    def _check_input_ciphertext(self, value: CipherTensor) -> None:
        if not isinstance(value, CipherTensor):
            raise TypeError("WPC CIPS Conv2d expects a grouped CipherTensor")
        if len(value.ids) != int(self.case.input_group_count):
            raise ValueError(
                f"expected {self.case.input_group_count} input ciphertexts, got {len(value.ids)}"
            )
        signature = getattr(value, "_wpc_cips_packing_signature", None)
        if signature != self.input_packing_signature:
            raise ValueError("input CipherTensor does not have the required WPC CIPS packing")
        for ciphertext_id in value.ids:
            level = int(self.scheme.backend.GetCiphertextLevel(int(ciphertext_id)))
            if level != int(self.level):
                raise ValueError(
                    f"input ciphertext level {level} does not match plan level {self.level}"
                )

    def evaluate(
        self,
        value: CipherTensor,
        *,
        compressed: bool | None = None,
        record_sequence: bool | None = None,
    ) -> CipherTensor:
        if not self.compiled or self.scheme is None:
            raise RuntimeError("WPC CIPS plan must be compiled before evaluation")
        self._check_input_ciphertext(value)
        if record_sequence is None:
            record_sequence = self.record_sequence
        if compressed is None:
            compressed = self.storage_mode == WPC_STORAGE_COMPRESSED
        transform_ids = (
            self.online_transform_ids if self.storage_mode == WPC_STORAGE_ONLINE
            else (self.compressed_transform_ids if bool(compressed) else self.full_control_transform_ids)
        )
        if len(transform_ids) != int(self.case.transform_count):
            path = "compressed" if compressed else "full-control"
            raise RuntimeError(f"{path} transforms are incomplete")

        backend = self.scheme.backend
        sequence: list[dict[str, Any]] = []
        output_ids: list[int] = []
        accumulation_add_count = 0
        for output_group in range(int(self.case.output_group_count)):
            accumulated_id: int | None = None
            for input_group in range(int(self.case.input_group_count)):
                key = f"out{output_group}_in{input_group}"
                before = None
                if bool(compressed) and bool(record_sequence):
                    before = _decode_stats(
                        backend.GetWPCCompressedGlobalStats(),
                        WPC_GLOBAL_STATS_FIELDS,
                    )
                if self.storage_mode == WPC_STORAGE_ONLINE:
                    partial_id = int(backend.EvaluateWPCOnlineLinearTransform(
                        int(transform_ids[key]), int(value.ids[input_group]),
                    ))
                elif bool(compressed):
                    partial_id = int(
                        backend.EvaluateWPCCompressedLinearTransform(
                            int(transform_ids[key]),
                            int(value.ids[input_group]),
                        )
                    )
                else:
                    partial_id = int(
                        backend.EvaluateLinearTransform(
                            int(transform_ids[key]),
                            int(value.ids[input_group]),
                        )
                    )
                if bool(compressed) and bool(record_sequence):
                    after = _decode_stats(
                        backend.GetWPCCompressedGlobalStats(),
                        WPC_GLOBAL_STATS_FIELDS,
                    )
                    sequence.append(
                        {
                            "layer": self.layer_name,
                            "group_key": key,
                            "current_materialized_bytes_before": int(
                                before["current_materialized_full_payload_bytes"]
                            ),
                            "current_materialized_bytes_after": int(
                                after["current_materialized_full_payload_bytes"]
                            ),
                            "current_materialized_transforms_after": int(
                                after["current_materialized_transform_count"]
                            ),
                        }
                    )
                if accumulated_id is None:
                    accumulated_id = partial_id
                else:
                    backend.AddCiphertext(int(accumulated_id), int(partial_id))
                    backend.DeleteCiphertext(int(partial_id))
                    accumulation_add_count += 1
            if accumulated_id is None:
                raise RuntimeError(f"output group {output_group} had no partial results")
            rescaled_id = int(self.scheme.evaluator.rescale(accumulated_id, in_place=False))
            backend.DeleteCiphertext(int(accumulated_id))
            if self.bias_plaintext is not None:
                backend.AddPlaintext(
                    int(rescaled_id),
                    int(self.bias_plaintext.ids[output_group]),
                )
            output_ids.append(rescaled_id)

        output = CipherTensor(
            self.scheme,
            output_ids,
            torch.Size([int(self.case.output_group_count), int(self.case.slots)]),
        )
        output._wpc_cips_packing_signature = self.output_packing_signature
        self.last_evaluation = {
            "path": (
                "online_encode" if self.storage_mode == WPC_STORAGE_ONLINE else "compressed"
                if compressed
                else (
                    "full"
                    if self.storage_mode == WPC_STORAGE_FULL
                    else "full_control"
                )
            ),
            "transform_evaluation_count": int(self.case.transform_count),
            "ciphertext_accumulation_add_count": int(accumulation_add_count),
            "bias_plaintext_add_count": (
                int(self.case.output_group_count)
                if self.bias_plaintext is not None
                else 0
            ),
            "input_level": int(self.level),
            "output_level": int(self.output_level),
            "evaluation_sequence": sequence,
        }
        return output

    def decrypt_unpack(self, value: CipherTensor) -> np.ndarray:
        if self.scheme is None:
            raise RuntimeError("WPC CIPS plan has no scheme")
        if getattr(value, "_wpc_cips_packing_signature", None) != self.output_packing_signature:
            raise ValueError("output CipherTensor does not match this WPC CIPS plan")
        decoded = np.asarray(self.scheme.decode(self.scheme.decrypt(value)), dtype=np.float64)
        groups: list[np.ndarray] = []
        for output_group in range(int(self.case.output_group_count)):
            local_case = self.case.local_case(output_group, 0)
            groups.append(
                unpack_output(decoded[output_group], local_case, LAYOUT_CIPS)
            )
        return np.concatenate(groups, axis=0)

    def compressed_stats(self) -> dict[str, dict[str, Any]]:
        if not self.compiled or self.scheme is None:
            return {}
        return {
            key: _decode_stats(
                self.scheme.backend.GetWPCCompressedLinearTransformStats(
                    int(transform_id)
                ),
                WPC_COMPRESSED_STATS_FIELDS,
            )
            for key, transform_id in self.compressed_transform_ids.items()
        }

    def global_stats(self) -> dict[str, Any]:
        if not self.compiled or self.scheme is None:
            return {}
        return _decode_stats(
            self.scheme.backend.GetWPCCompressedGlobalStats(),
            WPC_GLOBAL_STATS_FIELDS,
        )

    def online_stats(self) -> dict[str, dict[str, Any]]:
        if not self.compiled or self.scheme is None:
            return {}
        return {key: _decode_stats(
            self.scheme.backend.GetWPCOnlineLinearTransformStats(transform_id),
            WPC_ONLINE_STATS_FIELDS,
        ) for key, transform_id in self.online_transform_ids.items()}

    def reset_global_materialization_peak(self) -> None:
        if not self.compiled or self.scheme is None:
            raise RuntimeError("WPC CIPS plan is not compiled")
        self.scheme.backend.ResetWPCCompressedGlobalMaterializationPeak()

    def cleanup(self) -> None:
        if self.cleaned:
            return
        scheme = self.scheme
        backend = getattr(scheme, "backend", None) if scheme is not None else None
        if backend is not None:
            for transform_id in list(self.compressed_transform_ids.values()):
                backend.DeleteLinearTransform(int(transform_id))
            for transform_id in list(self.full_control_transform_ids.values()):
                backend.DeleteLinearTransform(int(transform_id))
            for transform_id in list(self.online_transform_ids.values()):
                backend.DeleteLinearTransform(int(transform_id))
        self.compressed_transform_ids = {}
        self.full_control_transform_ids = {}
        self.online_transform_ids = {}
        if self.bias_plaintext is not None:
            self.bias_plaintext.release()
            self.bias_plaintext = None
        self.compiled = False
        self.scheme = None
        self.cleaned = True


__all__ = [
    "WPCCIPSConv2dPlan",
    "WPC_COMPRESSED_STATS_FIELDS",
    "WPC_GLOBAL_STATS_FIELDS",
    "WPC_STORAGE_COMPRESSED",
    "WPC_STORAGE_FULL",
    "WPC_STORAGE_ONLINE",
    "WPC_STORAGE_MODES",
]
