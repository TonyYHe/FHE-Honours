"""Encrypted residual and concatenation joins for grouped WPC CIPS tensors."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

from orion.backend.python.tensors import CipherTensor


def _group_ranges(total: int, capacity: int) -> tuple[tuple[int, int], ...]:
    if int(total) <= 0 or int(capacity) <= 0:
        raise ValueError("channel count and capacity must be positive")
    return tuple(
        (start, min(int(total), start + int(capacity)))
        for start in range(0, int(total), int(capacity))
    )


def _normalise_shape(shape: Any) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in shape)
    if len(result) != 4 or result[0] != 1 or any(value <= 0 for value in result):
        raise ValueError("WPC CIPS branch tensors require positive shape [1,C,H,W]")
    return result


def _normalise_signature(
    signature: Any,
    *,
    logical_shape: tuple[int, int, int, int],
) -> tuple[Any, ...]:
    if not isinstance(signature, tuple) or len(signature) != 6:
        raise ValueError("invalid WPC CIPS packing signature")
    _, channels, height, width = logical_shape
    slots = int(signature[1])
    ranges = tuple(tuple(int(value) for value in row) for row in signature[5])
    if (
        signature[0] != "wpc_cips"
        or int(signature[2]) != int(channels)
        or int(signature[3]) != int(height)
        or int(signature[4]) != int(width)
    ):
        raise ValueError("packing signature does not match the logical shape")
    if slots <= 0 or slots & (slots - 1):
        raise ValueError("packing signature slot count must be a power of two")
    if slots % (height * width):
        raise ValueError("slot count must be divisible by the spatial size")
    capacity = int(slots // (height * width))
    if ranges != _group_ranges(channels, capacity):
        raise ValueError("packing signature does not use canonical CIPS groups")
    return ("wpc_cips", slots, channels, height, width, ranges)


def _producer_contract(plan: Any) -> tuple[tuple[int, int, int, int], tuple[Any, ...], int]:
    shape = _normalise_shape(getattr(plan, "output_shape"))
    signature = _normalise_signature(
        getattr(plan, "output_packing_signature"),
        logical_shape=shape,
    )
    return shape, signature, int(getattr(plan, "output_level"))


def _cipher_levels(value: CipherTensor) -> list[int]:
    return [
        int(value.backend.GetCiphertextLevel(int(ciphertext_id)))
        for ciphertext_id in value.ids
    ]


def _validate_ciphertext(
    value: CipherTensor,
    *,
    signature: tuple[Any, ...],
    level: int,
    scheme: Any,
    label: str,
) -> None:
    if not isinstance(value, CipherTensor):
        raise TypeError(f"{label} must be a CipherTensor")
    if value.scheme is not scheme:
        raise ValueError(f"{label} belongs to a different scheme")
    if getattr(value, "_wpc_cips_packing_signature", None) != signature:
        raise ValueError(f"{label} does not have the required WPC CIPS packing")
    expected_groups = len(signature[5])
    if len(value.ids) != expected_groups:
        raise ValueError(
            f"{label} has {len(value.ids)} ciphertext groups; expected {expected_groups}"
        )
    levels = _cipher_levels(value)
    if any(int(actual) != int(level) for actual in levels):
        raise ValueError(f"{label} levels {levels} do not match required level {level}")


def pack_cips_groups(values: np.ndarray, signature: tuple[Any, ...]) -> np.ndarray:
    """Pack a logical ``[C,H,W]`` tensor according to a canonical signature."""

    source = np.asarray(values, dtype=np.float64)
    slots = int(signature[1])
    channels = int(signature[2])
    height = int(signature[3])
    width = int(signature[4])
    ranges = tuple(tuple(int(value) for value in row) for row in signature[5])
    if tuple(source.shape) != (channels, height, width):
        raise ValueError(
            f"logical tensor shape is {tuple(source.shape)}, expected {(channels, height, width)}"
        )
    capacity = int(slots // (height * width))
    messages: list[np.ndarray] = []
    for start, end in ranges:
        message = np.zeros((slots,), dtype=np.float64)
        for local_channel, global_channel in enumerate(range(start, end)):
            for row in range(height):
                for column in range(width):
                    slot = int((row * width + column) * capacity + local_channel)
                    message[slot] = source[global_channel, row, column]
        messages.append(message)
    return np.stack(messages)


def unpack_cips_groups(messages: np.ndarray, signature: tuple[Any, ...]) -> np.ndarray:
    """Unpack canonical grouped CIPS messages into ``[C,H,W]``."""

    source = np.asarray(messages, dtype=np.float64)
    slots = int(signature[1])
    channels = int(signature[2])
    height = int(signature[3])
    width = int(signature[4])
    ranges = tuple(tuple(int(value) for value in row) for row in signature[5])
    expected = (len(ranges), slots)
    if tuple(source.shape) != expected:
        raise ValueError(f"packed tensor shape is {tuple(source.shape)}, expected {expected}")
    capacity = int(slots // (height * width))
    result = np.empty((channels, height, width), dtype=np.float64)
    for group, (start, end) in enumerate(ranges):
        for local_channel, global_channel in enumerate(range(start, end)):
            for row in range(height):
                for column in range(width):
                    slot = int((row * width + column) * capacity + local_channel)
                    result[global_channel, row, column] = source[group, slot]
    return result


class WPCCIPSResidualAddPlan:
    """Zero-depth residual addition for two identical CIPS contracts."""

    def __init__(
        self,
        *,
        logical_shape: Any,
        packing_signature: Any,
        level: int,
    ) -> None:
        self.output_shape = _normalise_shape(logical_shape)
        self.output_packing_signature = _normalise_signature(
            packing_signature,
            logical_shape=self.output_shape,
        )
        self.input_packing_signature = self.output_packing_signature
        self.level = int(level)
        self.output_level = int(level)
        if self.level < 0:
            raise ValueError("residual level cannot be negative")
        self.scheme: Any | None = None
        self.compiled = False
        self.cleaned = False
        self.last_evaluation: dict[str, Any] = {}

    @classmethod
    def between_plans(cls, left_plan: Any, right_plan: Any) -> "WPCCIPSResidualAddPlan":
        left_shape, left_signature, left_level = _producer_contract(left_plan)
        right_shape, right_signature, right_level = _producer_contract(right_plan)
        if left_shape != right_shape:
            raise ValueError("residual branches have different logical shapes")
        if left_signature != right_signature:
            raise ValueError("residual branches have different CIPS packing signatures")
        if left_level != right_level:
            raise ValueError("residual branches have different CKKS levels")
        return cls(
            logical_shape=left_shape,
            packing_signature=left_signature,
            level=left_level,
        )

    def compile(self, scheme: Any) -> dict[str, Any]:
        if self.compiled:
            raise RuntimeError("residual plan is already compiled")
        if self.cleaned:
            raise RuntimeError("a cleaned residual plan cannot be reused")
        if int(scheme.params.get_slots()) != int(self.output_packing_signature[1]):
            raise ValueError("residual slot count does not match the scheme")
        self.scheme = scheme
        self.compiled = True
        return {
            "operation": "ciphertext_groupwise_add",
            "ciphertext_group_count": int(len(self.output_packing_signature[5])),
            "input_level": int(self.level),
            "output_level": int(self.output_level),
            "level_consumption": 0,
        }

    def evaluate(self, left: CipherTensor, right: CipherTensor) -> CipherTensor:
        if not self.compiled or self.scheme is None:
            raise RuntimeError("compile the residual plan before evaluation")
        _validate_ciphertext(
            left,
            signature=self.input_packing_signature,
            level=self.level,
            scheme=self.scheme,
            label="left residual branch",
        )
        _validate_ciphertext(
            right,
            signature=self.input_packing_signature,
            level=self.level,
            scheme=self.scheme,
            label="right residual branch",
        )
        backend = self.scheme.backend
        if bool(getattr(backend, "align_addition_scales", False)):
            scale = max(1, int(left.scale()), int(right.scale()))
            for ciphertext_id in (*left.ids, *right.ids):
                backend.SetCiphertextScale(int(ciphertext_id), int(scale))
        output_ids = [
            int(backend.AddCiphertext(int(left_id), int(right_id)))
            for left_id, right_id in zip(left.ids, right.ids)
        ]
        output = CipherTensor(
            self.scheme,
            output_ids,
            torch.Size([len(output_ids), int(self.output_packing_signature[1])]),
        )
        output._wpc_cips_packing_signature = self.output_packing_signature
        self.last_evaluation = {
            "ciphertext_add_count": int(len(output_ids)),
            "input_level": int(self.level),
            "output_level": int(self.output_level),
            "level_consumption": 0,
            "clear_repack_or_encode_count": 0,
        }
        return output

    def decrypt_unpack(self, value: CipherTensor) -> np.ndarray:
        if self.scheme is None:
            raise RuntimeError("residual plan has no scheme")
        _validate_ciphertext(
            value,
            signature=self.output_packing_signature,
            level=self.output_level,
            scheme=self.scheme,
            label="residual output",
        )
        decoded = np.asarray(self.scheme.decode(self.scheme.decrypt(value)), dtype=np.float64)
        return unpack_cips_groups(decoded, self.output_packing_signature)

    def cleanup(self) -> None:
        self.scheme = None
        self.compiled = False
        self.cleaned = True


def build_concat_transforms(
    branch_signatures: Iterable[tuple[Any, ...]],
) -> tuple[tuple[Any, ...], dict[str, dict[str, Any]]]:
    """Build branch-group to output-group CIPS permutation transforms."""

    signatures = tuple(branch_signatures)
    if not signatures:
        raise ValueError("concat requires at least one branch")
    slots = int(signatures[0][1])
    height = int(signatures[0][3])
    width = int(signatures[0][4])
    for signature in signatures:
        if (
            int(signature[1]) != slots
            or int(signature[3]) != height
            or int(signature[4]) != width
        ):
            raise ValueError("concat branches must share slots and spatial shape")
    capacity = int(slots // (height * width))
    total_channels = int(sum(int(signature[2]) for signature in signatures))
    output_ranges = _group_ranges(total_channels, capacity)
    output_signature = (
        "wpc_cips",
        slots,
        total_channels,
        height,
        width,
        output_ranges,
    )

    transforms: dict[str, dict[str, Any]] = {}
    branch_offset = 0
    for branch_index, signature in enumerate(signatures):
        branch_ranges = tuple(tuple(int(value) for value in row) for row in signature[5])
        for source_group, (source_start, source_end) in enumerate(branch_ranges):
            global_start = int(branch_offset + source_start)
            global_end = int(branch_offset + source_end)
            for output_group, (output_start, output_end) in enumerate(output_ranges):
                overlap_start = max(global_start, int(output_start))
                overlap_end = min(global_end, int(output_end))
                if overlap_start >= overlap_end:
                    continue
                diagonals: dict[int, np.ndarray] = {}
                for global_channel in range(overlap_start, overlap_end):
                    source_local = int(global_channel - branch_offset - source_start)
                    output_local = int(global_channel - output_start)
                    for spatial in range(height * width):
                        source_slot = int(spatial * capacity + source_local)
                        output_slot = int(spatial * capacity + output_local)
                        rotation = int((source_slot - output_slot) % slots)
                        diagonal = diagonals.setdefault(
                            rotation,
                            np.zeros((slots,), dtype=np.float64),
                        )
                        diagonal[output_slot] = 1.0
                key = f"out{output_group}_branch{branch_index}_in{source_group}"
                transforms[key] = {
                    "key": key,
                    "output_group": int(output_group),
                    "branch_index": int(branch_index),
                    "source_group": int(source_group),
                    "branch_channel_offset": int(branch_offset),
                    "source_channel_range": [int(source_start), int(source_end)],
                    "global_channel_range": [int(overlap_start), int(overlap_end)],
                    "output_channel_range": [int(output_start), int(output_end)],
                    "diagonals": {
                        int(rotation): diagonal
                        for rotation, diagonal in sorted(diagonals.items())
                    },
                }
        branch_offset += int(signature[2])
    return output_signature, transforms


class WPCCIPSConcatPlan:
    """One-level encrypted channel concatenation for CIPS branch outputs."""

    def __init__(
        self,
        *,
        branch_shapes: Iterable[Any],
        branch_signatures: Iterable[Any],
        level: int,
        bsgs_ratio: float = 2.0,
    ) -> None:
        self.branch_shapes = tuple(_normalise_shape(shape) for shape in branch_shapes)
        raw_signatures = tuple(branch_signatures)
        if len(self.branch_shapes) != len(raw_signatures) or not self.branch_shapes:
            raise ValueError("concat branch shape/signature counts must match and be nonzero")
        self.branch_signatures = tuple(
            _normalise_signature(signature, logical_shape=shape)
            for signature, shape in zip(raw_signatures, self.branch_shapes)
        )
        batch, _, height, width = self.branch_shapes[0]
        if any(
            shape[0] != batch or shape[2] != height or shape[3] != width
            for shape in self.branch_shapes
        ):
            raise ValueError("concat branches must share batch and spatial dimensions")
        self.output_packing_signature, self._uncompiled_rows = build_concat_transforms(
            self.branch_signatures
        )
        self.input_packing_signatures = self.branch_signatures
        self.output_shape = (
            1,
            int(sum(shape[1] for shape in self.branch_shapes)),
            int(height),
            int(width),
        )
        self.level = int(level)
        if self.level <= 0:
            raise ValueError("concat materialization requires one rescale level")
        self.output_level = int(self.level - 1)
        self.bsgs_ratio = float(bsgs_ratio)
        self.scheme: Any | None = None
        self.transform_ids: dict[str, int] = {}
        self.transform_rows: dict[str, dict[str, Any]] = {}
        self.compiled = False
        self.cleaned = False
        self.last_evaluation: dict[str, Any] = {}

    @classmethod
    def from_producers(
        cls,
        producers: Iterable[Any],
        *,
        bsgs_ratio: float = 2.0,
    ) -> "WPCCIPSConcatPlan":
        contracts = tuple(_producer_contract(plan) for plan in producers)
        if not contracts:
            raise ValueError("concat requires at least one producer")
        levels = {int(contract[2]) for contract in contracts}
        if len(levels) != 1:
            raise ValueError("concat branches have different CKKS levels")
        return cls(
            branch_shapes=[contract[0] for contract in contracts],
            branch_signatures=[contract[1] for contract in contracts],
            level=next(iter(levels)),
            bsgs_ratio=float(bsgs_ratio),
        )

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def compile(self, scheme: Any) -> dict[str, Any]:
        if self.compiled:
            raise RuntimeError("concat plan is already compiled")
        if self.cleaned:
            raise RuntimeError("a cleaned concat plan cannot be reused")
        if int(scheme.params.get_slots()) != int(self.output_packing_signature[1]):
            raise ValueError("concat slot count does not match the scheme")
        self.scheme = scheme
        bytes_per_diagonal = int(
            (self.level + 1 + len(scheme.params.get_logp()))
            * int(scheme.params.get_ring_degree())
            * 8
        )
        try:
            for key, row in self._uncompiled_rows.items():
                diagonals = row["diagonals"]
                indices = [int(value) for value in sorted(diagonals)]
                flattened = np.concatenate(
                    [np.asarray(diagonals[index], dtype=np.float32) for index in indices]
                ).tolist()
                transform_id = int(
                    scheme.backend.GenerateLinearTransform(
                        indices,
                        flattened,
                        int(self.level),
                        float(self.bsgs_ratio),
                        "none",
                    )
                )
                scheme.lt_evaluator.generate_rotation_keys(transform_id)
                self.transform_ids[key] = transform_id
                self.transform_rows[key] = {
                    **{name: value for name, value in row.items() if name != "diagonals"},
                    "diagonal_count": int(len(diagonals)),
                    "full_qp_payload_bytes": int(len(diagonals) * bytes_per_diagonal),
                }
            self.compiled = True
            return self.storage_summary()
        except Exception:
            self.cleanup()
            raise

    def validate_consumer(self, consumer_plan: Any) -> None:
        if self.output_shape != tuple(int(value) for value in consumer_plan.input_shape):
            raise ValueError("concat output logical shape does not match consumer input")
        if self.output_packing_signature != consumer_plan.input_packing_signature:
            raise ValueError("concat output packing does not match consumer input")
        if int(self.output_level) != int(consumer_plan.level):
            raise ValueError("concat output level does not match consumer input level")

    def storage_summary(self) -> dict[str, Any]:
        return {
            "storage_mode": "ordinary_full_qp_permutation",
            "transform_count": int(len(self.transform_rows)),
            "diagonal_count": int(
                sum(int(row["diagonal_count"]) for row in self.transform_rows.values())
            ),
            "full_qp_payload_bytes": int(
                sum(
                    int(row["full_qp_payload_bytes"])
                    for row in self.transform_rows.values()
                )
            ),
        }

    def evaluate(self, *branches: CipherTensor) -> CipherTensor:
        if not self.compiled or self.scheme is None:
            raise RuntimeError("compile the concat plan before evaluation")
        if len(branches) != len(self.branch_signatures):
            raise ValueError(
                f"concat received {len(branches)} branches; expected {len(self.branch_signatures)}"
            )
        for index, (branch, signature) in enumerate(
            zip(branches, self.branch_signatures)
        ):
            _validate_ciphertext(
                branch,
                signature=signature,
                level=self.level,
                scheme=self.scheme,
                label=f"concat branch {index}",
            )

        backend = self.scheme.backend
        output_ids: list[int] = []
        evaluation_count = 0
        accumulation_add_count = 0
        output_ranges = self.output_packing_signature[5]
        for output_group in range(len(output_ranges)):
            accumulated_id: int | None = None
            for key, row in self.transform_rows.items():
                if int(row["output_group"]) != int(output_group):
                    continue
                branch_index = int(row["branch_index"])
                source_group = int(row["source_group"])
                partial_id = int(
                    backend.EvaluateLinearTransform(
                        int(self.transform_ids[key]),
                        int(branches[branch_index].ids[source_group]),
                    )
                )
                evaluation_count += 1
                if accumulated_id is None:
                    accumulated_id = partial_id
                else:
                    backend.AddCiphertext(int(accumulated_id), int(partial_id))
                    backend.DeleteCiphertext(int(partial_id))
                    accumulation_add_count += 1
            if accumulated_id is None:
                raise RuntimeError(f"concat output group {output_group} has no source")
            rescaled_id = int(self.scheme.evaluator.rescale(accumulated_id, in_place=False))
            backend.DeleteCiphertext(int(accumulated_id))
            output_ids.append(rescaled_id)

        output = CipherTensor(
            self.scheme,
            output_ids,
            torch.Size([len(output_ids), int(self.output_packing_signature[1])]),
        )
        output._wpc_cips_packing_signature = self.output_packing_signature
        self.last_evaluation = {
            "transform_evaluation_count": int(evaluation_count),
            "ciphertext_accumulation_add_count": int(accumulation_add_count),
            "input_branch_count": int(len(branches)),
            "input_ciphertext_group_count": int(sum(len(value.ids) for value in branches)),
            "output_ciphertext_group_count": int(len(output_ids)),
            "input_level": int(self.level),
            "output_level": int(self.output_level),
            "level_consumption": 1,
            "clear_repack_or_encode_count": 0,
        }
        return output

    def decrypt_unpack(self, value: CipherTensor) -> np.ndarray:
        if self.scheme is None:
            raise RuntimeError("concat plan has no scheme")
        _validate_ciphertext(
            value,
            signature=self.output_packing_signature,
            level=self.output_level,
            scheme=self.scheme,
            label="concat output",
        )
        decoded = np.asarray(self.scheme.decode(self.scheme.decrypt(value)), dtype=np.float64)
        return unpack_cips_groups(decoded, self.output_packing_signature)

    def cleanup(self) -> None:
        if self.cleaned:
            return
        backend = getattr(self.scheme, "backend", None) if self.scheme is not None else None
        if backend is not None:
            for transform_id in list(self.transform_ids.values()):
                backend.DeleteLinearTransform(int(transform_id))
        self.transform_ids = {}
        self.compiled = False
        self.cleaned = True


__all__ = [
    "WPCCIPSResidualAddPlan",
    "WPCCIPSConcatPlan",
    "build_concat_transforms",
    "pack_cips_groups",
    "unpack_cips_groups",
]
