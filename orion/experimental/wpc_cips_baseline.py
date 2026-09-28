"""Functional CIPS/Rotation-Padding baseline for the WPC experiment.

This module implements the vertically padded convolution illustrated by
Algorithms 1-2 and Figures 8-10 of the WPC paper.  It is intentionally a
correctness baseline, not an optimized homomorphic kernel:

* ``cips`` packs channels in the innermost slot dimension;
* ``channel_first`` is the matched channel-first diagonal-layout control;
* height padding uses adjacent same-channel values, i.e. circular extension;
* the width kernel is fixed to one so a single flattened rotation has exactly
  the paper's illustrated semantics without an unimplemented row-boundary
  transform.

Both layouts are lowered to cyclic diagonals and evaluated with the same clear
rotate/multiply/accumulate primitive.  The resulting output is compared with a
direct clear convolution using the same Rotation-Padding semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from orion.experimental.wpc_periodicity import (
    PeriodicityObservation,
    aggregate_periodicity,
    analyze_slot_periodicity,
)


LAYOUT_CIPS = "cips"
LAYOUT_CHANNEL_FIRST = "channel_first"
SUPPORTED_LAYOUTS = {LAYOUT_CIPS, LAYOUT_CHANNEL_FIRST}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class CIPSConvCase:
    """One single-ciphertext convolution shared by both layouts."""

    slots: int
    input_channels: int
    output_channels: int
    height: int
    width: int
    kernel_height: int
    pad_height_before: int

    def __post_init__(self) -> None:
        values = {
            "slots": self.slots,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "height": self.height,
            "width": self.width,
            "kernel_height": self.kernel_height,
        }
        if any(int(value) <= 0 for value in values.values()):
            raise ValueError(f"case dimensions must be positive: {values}")
        if not _is_power_of_two(int(self.slots)):
            raise ValueError("slots must be a power of two")
        if not _is_power_of_two(int(self.height)) or not _is_power_of_two(int(self.width)):
            raise ValueError("CIPS baseline height and width must be powers of two")
        if int(self.slots) % (int(self.height) * int(self.width)):
            raise ValueError("slots must be divisible by height*width")
        if int(self.input_channels) > self.channel_capacity:
            raise ValueError("input channels exceed the single-ciphertext CIPS capacity")
        if int(self.output_channels) > self.channel_capacity:
            raise ValueError("output channels exceed the single-ciphertext CIPS capacity")
        if not 0 <= int(self.pad_height_before) < int(self.kernel_height):
            raise ValueError("pad_height_before must lie in [0, kernel_height)")

    @property
    def channel_capacity(self) -> int:
        return int(self.slots) // (int(self.height) * int(self.width))

    @property
    def pad_height_after(self) -> int:
        return int(self.kernel_height) - 1 - int(self.pad_height_before)

    def to_dict(self) -> dict[str, int]:
        result = {key: int(value) for key, value in asdict(self).items()}
        result["channel_capacity"] = int(self.channel_capacity)
        result["pad_height_after"] = int(self.pad_height_after)
        result["kernel_width"] = 1
        return result


def slot_index(case: CIPSConvCase, layout: str, channel: int, height: int, width: int) -> int:
    """Map one logical tensor element to its ciphertext slot."""

    normalized = str(layout)
    if normalized not in SUPPORTED_LAYOUTS:
        raise ValueError(f"unsupported layout {layout!r}")
    c = int(channel)
    h = int(height)
    w = int(width)
    if not 0 <= c < int(case.channel_capacity):
        raise IndexError("channel index lies outside the packed capacity")
    if not 0 <= h < int(case.height) or not 0 <= w < int(case.width):
        raise IndexError("spatial index lies outside the packed tensor")
    if normalized == LAYOUT_CIPS:
        return int((h * int(case.width) + w) * int(case.channel_capacity) + c)
    return int((c * int(case.height) + h) * int(case.width) + w)


def pack_tensor(tensor: np.ndarray, case: CIPSConvCase, layout: str) -> np.ndarray:
    """Pack ``[C,H,W]`` into one full slot vector, zeroing spare channels."""

    values = np.asarray(tensor, dtype=np.float64)
    expected = (int(case.input_channels), int(case.height), int(case.width))
    if tuple(values.shape) != expected:
        raise ValueError(f"input tensor shape is {tuple(values.shape)}, expected {expected}")
    packed = np.zeros((int(case.slots),), dtype=np.float64)
    for channel in range(int(case.input_channels)):
        for height in range(int(case.height)):
            for width in range(int(case.width)):
                packed[slot_index(case, layout, channel, height, width)] = values[
                    channel, height, width
                ]
    return packed


def unpack_output(packed: np.ndarray, case: CIPSConvCase, layout: str) -> np.ndarray:
    """Unpack the active output channels from one slot vector."""

    values = np.asarray(packed, dtype=np.float64)
    if tuple(values.shape) != (int(case.slots),):
        raise ValueError(f"packed output length is {values.size}, expected {case.slots}")
    output = np.empty(
        (int(case.output_channels), int(case.height), int(case.width)),
        dtype=np.float64,
    )
    for channel in range(int(case.output_channels)):
        for height in range(int(case.height)):
            for width in range(int(case.width)):
                output[channel, height, width] = values[
                    slot_index(case, layout, channel, height, width)
                ]
    return output


def rotation_padded_reference(
    tensor: np.ndarray,
    weights: np.ndarray,
    case: CIPSConvCase,
) -> np.ndarray:
    """Direct convolution with WPC adjacent-neuron height padding.

    The padded rows are copied cyclically from the same channel.  This is not
    zero padding and is therefore kept separate from ordinary ``conv2d``.
    """

    source = np.asarray(tensor, dtype=np.float64)
    kernel = np.asarray(weights, dtype=np.float64)
    expected_input = (int(case.input_channels), int(case.height), int(case.width))
    expected_weight = (
        int(case.output_channels),
        int(case.input_channels),
        int(case.kernel_height),
        1,
    )
    if tuple(source.shape) != expected_input:
        raise ValueError(f"input tensor shape is {tuple(source.shape)}, expected {expected_input}")
    if tuple(kernel.shape) != expected_weight:
        raise ValueError(f"weight tensor shape is {tuple(kernel.shape)}, expected {expected_weight}")

    output = np.zeros(
        (int(case.output_channels), int(case.height), int(case.width)),
        dtype=np.float64,
    )
    for output_channel in range(int(case.output_channels)):
        for height in range(int(case.height)):
            for width in range(int(case.width)):
                total = 0.0
                for input_channel in range(int(case.input_channels)):
                    for kernel_height in range(int(case.kernel_height)):
                        source_height = (
                            height + kernel_height - int(case.pad_height_before)
                        ) % int(case.height)
                        total += (
                            source[input_channel, source_height, width]
                            * kernel[output_channel, input_channel, kernel_height, 0]
                        )
                output[output_channel, height, width] = total
    return output


def zero_padded_reference(
    tensor: np.ndarray,
    weights: np.ndarray,
    case: CIPSConvCase,
) -> np.ndarray:
    """Matched zero-padding reference used to expose the boundary change."""

    source = np.asarray(tensor, dtype=np.float64)
    kernel = np.asarray(weights, dtype=np.float64)
    output = np.zeros(
        (int(case.output_channels), int(case.height), int(case.width)),
        dtype=np.float64,
    )
    for output_channel in range(int(case.output_channels)):
        for height in range(int(case.height)):
            for width in range(int(case.width)):
                total = 0.0
                for input_channel in range(int(case.input_channels)):
                    for kernel_height in range(int(case.kernel_height)):
                        source_height = height + kernel_height - int(case.pad_height_before)
                        if 0 <= source_height < int(case.height):
                            total += (
                                source[input_channel, source_height, width]
                                * kernel[output_channel, input_channel, kernel_height, 0]
                            )
                output[output_channel, height, width] = total
    return output


def build_cyclic_diagonals(
    weights: np.ndarray,
    case: CIPSConvCase,
    layout: str,
) -> dict[int, np.ndarray]:
    """Lower the matched convolution to cyclic diagonals for one layout."""

    kernel = np.asarray(weights, dtype=np.float64)
    expected = (
        int(case.output_channels),
        int(case.input_channels),
        int(case.kernel_height),
        1,
    )
    if tuple(kernel.shape) != expected:
        raise ValueError(f"weight tensor shape is {tuple(kernel.shape)}, expected {expected}")

    diagonals: dict[int, np.ndarray] = {}
    for output_channel in range(int(case.output_channels)):
        for input_channel in range(int(case.input_channels)):
            for kernel_height in range(int(case.kernel_height)):
                weight = float(kernel[output_channel, input_channel, kernel_height, 0])
                if weight == 0.0:
                    continue
                for height in range(int(case.height)):
                    source_height = (
                        height + kernel_height - int(case.pad_height_before)
                    ) % int(case.height)
                    for width in range(int(case.width)):
                        output_slot = slot_index(
                            case,
                            layout,
                            output_channel,
                            height,
                            width,
                        )
                        input_slot = slot_index(
                            case,
                            layout,
                            input_channel,
                            source_height,
                            width,
                        )
                        rotation = int((input_slot - output_slot) % int(case.slots))
                        diagonal = diagonals.setdefault(
                            rotation,
                            np.zeros((int(case.slots),), dtype=np.float64),
                        )
                        diagonal[output_slot] += weight
    return {
        int(rotation): diagonal
        for rotation, diagonal in sorted(diagonals.items())
        if bool(np.any(diagonal != 0.0))
    }


def evaluate_cyclic_diagonals(
    packed_input: np.ndarray,
    diagonals: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Evaluate ``sum_r diag[r] * Rot(input,r)`` in clear arithmetic."""

    source = np.asarray(packed_input, dtype=np.float64)
    output = np.zeros_like(source)
    for rotation, diagonal in sorted(diagonals.items()):
        # np.roll(x, -r)[j] = x[(j+r) mod n].
        output += np.roll(source, -int(rotation)) * np.asarray(diagonal, dtype=np.float64)
    return output


def reconstruct_message_period(message: np.ndarray, period: int) -> np.ndarray:
    """Reconstruct a slot message from one exact period."""

    values = np.asarray(message, dtype=np.float64)
    t = int(period)
    if t <= 0 or int(values.size) % t:
        raise ValueError("period must be positive and divide the message length")
    return np.tile(values[:t], int(values.size) // t)


def summarize_layout(
    diagonals: Mapping[int, np.ndarray],
    *,
    full_encoded_bytes_per_diagonal: int,
) -> dict[str, Any]:
    """Return operation and slot-periodicity metrics for one layout."""

    rows: list[PeriodicityObservation] = []
    periods: dict[str, int] = {}
    message_roundtrip_failures: list[int] = []
    for rotation, diagonal in sorted(diagonals.items()):
        periodicity = analyze_slot_periodicity(diagonal, payload_format="real")
        rows.append(
            PeriodicityObservation(
                periodicity=periodicity,
                full_encoded_bytes=int(full_encoded_bytes_per_diagonal),
            )
        )
        periods[str(int(rotation))] = int(periodicity.minimal_period)
        reconstructed = reconstruct_message_period(diagonal, periodicity.minimal_period)
        if not np.array_equal(diagonal, reconstructed):
            message_roundtrip_failures.append(int(rotation))

    aggregate = aggregate_periodicity(rows)
    rotations = [int(rotation) for rotation in diagonals if int(rotation) != 0]
    return {
        "diagonal_count": int(len(diagonals)),
        "homomorphic_multiply_count": int(len(diagonals)),
        "homomorphic_rotation_count_unoptimized": int(len(rotations)),
        "homomorphic_add_count": int(max(0, len(diagonals) - 1)),
        "rotation_indices": rotations,
        "minimal_slot_period_by_rotation": periods,
        "message_period_reconstruction_exact": not message_roundtrip_failures,
        "message_period_reconstruction_failures": message_roundtrip_failures,
        "periodicity": aggregate.to_dict(),
    }


def run_clear_layout_comparison(
    tensor: np.ndarray,
    weights: np.ndarray,
    case: CIPSConvCase,
    *,
    full_encoded_bytes_per_diagonal: int,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Run the matched clear baseline and enforce layout correctness."""

    reference = rotation_padded_reference(tensor, weights, case)
    zero_reference = zero_padded_reference(tensor, weights, case)
    layouts: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    for layout in (LAYOUT_CHANNEL_FIRST, LAYOUT_CIPS):
        packed = pack_tensor(tensor, case, layout)
        diagonals = build_cyclic_diagonals(weights, case, layout)
        evaluated = evaluate_cyclic_diagonals(packed, diagonals)
        output = unpack_output(evaluated, case, layout)
        outputs[layout] = output
        error = np.abs(output - reference)
        layouts[layout] = {
            **summarize_layout(
                diagonals,
                full_encoded_bytes_per_diagonal=int(full_encoded_bytes_per_diagonal),
            ),
            "max_abs_error_vs_rotation_padding_reference": float(np.max(error)),
            "mean_abs_error_vs_rotation_padding_reference": float(np.mean(error)),
            "correct": bool(np.allclose(output, reference, rtol=0.0, atol=float(atol))),
            "diagonals": diagonals,
        }

    layout_delta = np.abs(outputs[LAYOUT_CIPS] - outputs[LAYOUT_CHANNEL_FIRST])
    boundary_delta = np.abs(reference - zero_reference)
    return {
        "case": case.to_dict(),
        "rotation_padding_semantics": "circular_adjacent_same_channel_height",
        "zero_padding_is_not_the_reference": True,
        "rotation_padding_vs_zero_padding_max_abs_delta": float(np.max(boundary_delta)),
        "layout_output_max_abs_delta": float(np.max(layout_delta)),
        "layouts": layouts,
        "valid": bool(
            all(bool(row["correct"]) for row in layouts.values())
            and np.allclose(
                outputs[LAYOUT_CIPS],
                outputs[LAYOUT_CHANNEL_FIRST],
                rtol=0.0,
                atol=float(atol),
            )
        ),
    }


__all__ = [
    "CIPSConvCase",
    "LAYOUT_CHANNEL_FIRST",
    "LAYOUT_CIPS",
    "build_cyclic_diagonals",
    "evaluate_cyclic_diagonals",
    "pack_tensor",
    "reconstruct_message_period",
    "rotation_padded_reference",
    "run_clear_layout_comparison",
    "slot_index",
    "summarize_layout",
    "unpack_output",
    "zero_padded_reference",
]
