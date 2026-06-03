import math
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from orion.core.progress import write_progress_event

from .module import Module, timer


def _align_ciphertexts_for_add(left, right):
    if bool(getattr(left.scheme.backend, "align_addition_scales", False)):
        scale = max(1, int(left.scale()))
        left.set_scale(int(scale))
        right.set_scale(int(scale))
    return left, right


class Add(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def compile(self):
        runtime = getattr(self, "layout_policy_add_runtime", None)
        if runtime is not None and callable(getattr(runtime, "compile", None)):
            runtime.assigned_level = int(self.level) if self.level is not None else None
            runtime.assigned_depth = int(self.depth or 0)
            runtime.compile(self.scheme)

    def forward(self, x, y):
        runtime = getattr(self, "layout_policy_add_runtime", None)
        if self.he_mode and runtime is not None:
            return runtime(x, y)
        return x + y


class ConcatCipherTensor:
    """Lazy HE representation of a channel concat.

    Keeping the source ciphertexts separate lets a following Conv2d split its
    weights by channel range and avoid materializing the concat boundary.
    """

    def __init__(self, module, parts, *, owned_parts=()):
        self.module = module
        self.parts = tuple(parts)
        self.owned_parts = tuple(owned_parts)
        self.scheme = self.parts[0].scheme if self.parts else None
        self.shape = module.output_shape
        self.on_shape = module.fhe_output_shape
        self._materialized = None

    def materialize(self):
        if self._materialized is None:
            self._materialized = self.module.materialize(self.parts)
        return self._materialized

    @property
    def ids(self):
        return self.materialize().ids

    @ids.setter
    def ids(self, value):
        self.materialize().ids = value

    def release(self):
        if self._materialized is not None:
            self._materialized.release()
            self._materialized = None
        self.release_owned_parts()

    def release_owned_parts(self):
        for part in self.owned_parts:
            release = getattr(part, "release", None)
            if callable(release):
                release()
        self.owned_parts = ()

    def map_parts(self, fn):
        parts = tuple(fn(int(index), part) for index, part in enumerate(self.parts))
        return ConcatCipherTensor(self.module, parts, owned_parts=parts)

    def __len__(self):
        return len(self.materialize())

    def __getattr__(self, name):
        return getattr(self.materialize(), name)

    def __add__(self, other):
        return self.materialize() + other

    def __radd__(self, other):
        return other + self.materialize()

    def __iadd__(self, other):
        materialized = self.materialize()
        materialized += other
        return materialized

    def __sub__(self, other):
        return self.materialize() - other

    def __rsub__(self, other):
        return other - self.materialize()

    def __mul__(self, other):
        return self.materialize() * other

    def __rmul__(self, other):
        return other * self.materialize()


class Concat(Module):
    def __init__(self, dim: int = 1, bsgs_ratio: int = 2):
        super().__init__()
        self.dim = int(dim)
        self.bsgs_ratio = float(bsgs_ratio)
        self.set_depth(0)
        self.concat_input_shapes = ()
        self.concat_input_fhe_shapes = ()
        self.concat_input_gaps = ()
        self.transform_ids_by_input = []
        self.transform_sources_by_input = []
        self._compiled_backend = None

    def extra_repr(self):
        return super().extra_repr() + f", dim={self.dim}"

    def configure_from_stats(
        self,
        *,
        input_shapes,
        input_fhe_shapes,
        input_gaps,
        output_shape,
        fhe_output_shape,
        output_gap,
    ) -> None:
        self.concat_input_shapes = tuple(torch.Size(shape) for shape in input_shapes)
        self.concat_input_fhe_shapes = tuple(torch.Size(shape) for shape in input_fhe_shapes)
        self.concat_input_gaps = tuple(int(gap) for gap in input_gaps)
        self.output_shape = torch.Size(output_shape)
        self.fhe_output_shape = torch.Size(fhe_output_shape)
        self.output_gap = int(output_gap)

    def compile(self):
        # Generic materialization is compiled lazily only if a non-fused
        # consumer actually asks for it.
        return

    def _flat_index(self, channel, rows, cols, *, gap, height, width, row_offset=0):
        gap = int(gap)
        phase = int(channel) % int(gap * gap)
        packed_channel = int(channel) // int(gap * gap)
        packed_rows = rows.astype(np.int64) * int(gap) + int(phase // gap) + int(row_offset)
        packed_cols = cols.astype(np.int64) * int(gap) + int(phase % gap)
        return ((int(packed_channel) * int(height) + packed_rows) * int(width) + packed_cols).astype(np.int64)

    def _add_diagonal_entries(self, diagonals, source_indices, output_indices, *, slots):
        source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        output_indices = np.asarray(output_indices, dtype=np.int64).reshape(-1)
        if source_indices.size == 0:
            return
        source_blocks = source_indices // int(slots)
        output_blocks = output_indices // int(slots)
        source_local = source_indices % int(slots)
        output_local = output_indices % int(slots)
        diag_indices = (source_local - output_local) % int(slots)
        order = np.lexsort((diag_indices, source_blocks, output_blocks))
        source_blocks = source_blocks[order]
        output_blocks = output_blocks[order]
        output_local = output_local[order]
        diag_indices = diag_indices[order]

        start = 0
        while start < int(diag_indices.size):
            end = start + 1
            while (
                end < int(diag_indices.size)
                and int(output_blocks[end]) == int(output_blocks[start])
                and int(source_blocks[end]) == int(source_blocks[start])
                and int(diag_indices[end]) == int(diag_indices[start])
            ):
                end += 1
            block = diagonals.setdefault((int(output_blocks[start]), int(source_blocks[start])), {})
            diag = block.get(int(diag_indices[start]))
            if diag is None:
                diag = np.zeros((int(slots),), dtype=np.float32)
                block[int(diag_indices[start])] = diag
            diag[output_local[start:end].astype(np.int64)] = 1.0
            start = end

    def _diagonals_for_input(self, input_index: int, *, slots: int):
        input_shape = self.concat_input_shapes[int(input_index)]
        input_fhe_shape = self.concat_input_fhe_shapes[int(input_index)]
        output_shape = self.output_shape
        output_fhe_shape = self.fhe_output_shape
        input_gap = int(self.concat_input_gaps[int(input_index)])
        output_gap = int(self.output_gap)
        if int(input_gap) != int(output_gap):
            raise ValueError("Concat materialization requires all inputs to share the same FHE gap")
        channel_offset = sum(int(shape[1]) for shape in self.concat_input_shapes[: int(input_index)])
        n, channels, height, width = (int(value) for value in input_shape)
        input_block = int(input_fhe_shape[1] * input_fhe_shape[2] * input_fhe_shape[3])
        output_block = int(output_fhe_shape[1] * output_fhe_shape[2] * output_fhe_shape[3])
        row_grid, col_grid = np.meshgrid(
            np.arange(int(height), dtype=np.int64),
            np.arange(int(width), dtype=np.int64),
            indexing="ij",
        )
        rows = row_grid.reshape(-1)
        cols = col_grid.reshape(-1)
        diagonals = {}
        for batch in range(int(n)):
            for channel in range(int(channels)):
                source = self._flat_index(
                    int(channel),
                    rows,
                    cols,
                    gap=int(input_gap),
                    height=int(input_fhe_shape[2]),
                    width=int(input_fhe_shape[3]),
                ) + int(batch) * int(input_block)
                output = self._flat_index(
                    int(channel_offset + channel),
                    rows,
                    cols,
                    gap=int(output_gap),
                    height=int(output_fhe_shape[2]),
                    width=int(output_fhe_shape[3]),
                ) + int(batch) * int(output_block)
                self._add_diagonal_entries(diagonals, source, output, slots=int(slots))
        return diagonals

    def _diag_indices_by_block(self, diagonals):
        return {
            (int(row), int(col)): tuple(sorted(int(index) for index in dict(diags).keys()))
            for (row, col), diags in dict(diagonals or {}).items()
        }

    def _diagonals_for_input_blocks(self, input_index: int, *, slots: int, blocks):
        requested = {
            (int(row), int(col))
            for row, col in tuple(blocks or ())
        }
        if not requested:
            return {}
        input_shape = self.concat_input_shapes[int(input_index)]
        input_fhe_shape = self.concat_input_fhe_shapes[int(input_index)]
        output_fhe_shape = self.fhe_output_shape
        input_gap = int(self.concat_input_gaps[int(input_index)])
        output_gap = int(self.output_gap)
        if int(input_gap) != int(output_gap):
            raise ValueError("Concat materialization requires all inputs to share the same FHE gap")
        channel_offset = sum(int(shape[1]) for shape in self.concat_input_shapes[: int(input_index)])
        n, channels, height, width = (int(value) for value in input_shape)
        input_block = int(input_fhe_shape[1] * input_fhe_shape[2] * input_fhe_shape[3])
        output_block = int(output_fhe_shape[1] * output_fhe_shape[2] * output_fhe_shape[3])
        row_grid, col_grid = np.meshgrid(
            np.arange(int(height), dtype=np.int64),
            np.arange(int(width), dtype=np.int64),
            indexing="ij",
        )
        rows = row_grid.reshape(-1)
        cols = col_grid.reshape(-1)
        diagonals = {}
        for batch in range(int(n)):
            for channel in range(int(channels)):
                source = self._flat_index(
                    int(channel),
                    rows,
                    cols,
                    gap=int(input_gap),
                    height=int(input_fhe_shape[2]),
                    width=int(input_fhe_shape[3]),
                ) + int(batch) * int(input_block)
                output = self._flat_index(
                    int(channel_offset + channel),
                    rows,
                    cols,
                    gap=int(output_gap),
                    height=int(output_fhe_shape[2]),
                    width=int(output_fhe_shape[3]),
                ) + int(batch) * int(output_block)
                source_blocks = source // int(slots)
                output_blocks = output // int(slots)
                mask = np.zeros(output.shape, dtype=bool)
                for row, col in requested:
                    mask |= (output_blocks == int(row)) & (source_blocks == int(col))
                if bool(np.any(mask)):
                    self._add_diagonal_entries(diagonals, source[mask], output[mask], slots=int(slots))
        return diagonals

    def _native_materialization_requested(self) -> bool:
        materialization = str(getattr(self, "layout_policy_output_materialization", "") or "")
        if materialization not in {"native_halo_stripe", "native_stripe", "channel_aligned_native_stripe"}:
            return False
        if not tuple(getattr(self, "layout_policy_native_output_target_signature", ()) or ()):
            raise RuntimeError(
                f"Concat {getattr(self, 'name', '')} requested native materialization without a target signature"
            )
        return True

    @staticmethod
    def _normalise_native_signature(raw, *, label: str):
        rows = []
        for item in tuple(raw or ()):
            values = tuple(int(value) for value in tuple(item))
            if len(values) != 4:
                raise RuntimeError(f"Concat native {label} signature row must have 4 integers")
            h_start, h_end, channel_start, channel_count = values
            if int(h_end) <= int(h_start) or int(channel_count) <= 0:
                continue
            if int(channel_start) < 0:
                raise RuntimeError(f"Concat native {label} signature has a negative channel start")
            rows.append((int(h_start), int(h_end), int(channel_start), int(channel_count)))
        return tuple(rows)

    def _native_input_signatures(self):
        raw = tuple(getattr(self, "layout_policy_concat_input_source_signatures", ()) or ())
        if len(raw) != len(self.concat_input_shapes):
            raise RuntimeError(
                f"Concat {getattr(self, 'name', '')} native materialization expects "
                f"{len(self.concat_input_shapes)} source signatures, got {len(raw)}"
            )
        signatures = []
        for input_index, signature in enumerate(raw):
            parsed = self._normalise_native_signature(signature, label=f"input{int(input_index)}")
            if not parsed:
                raise RuntimeError(
                    f"Concat {getattr(self, 'name', '')} native input {int(input_index)} has no storage signature"
                )
            signatures.append(parsed)
        return tuple(signatures)

    def _native_output_signature(self):
        signature = self._normalise_native_signature(
            getattr(self, "layout_policy_native_output_target_signature", ()) or (),
            label="target",
        )
        if not signature:
            raise RuntimeError(f"Concat {getattr(self, 'name', '')} native target signature is empty")
        return tuple(signature)

    def _validate_native_concat_signatures(self, input_signatures, target_signature) -> None:
        if len(self.concat_input_shapes) != len(input_signatures):
            raise RuntimeError("Concat native materialization input signature count mismatch")
        if int(getattr(self, "dim", 1)) != 1:
            raise RuntimeError("Concat native materialization only supports channel dim=1")
        if int(self.output_shape[0]) != 1:
            raise RuntimeError("Concat native materialization currently requires batch size 1")
        output_channels = int(self.output_shape[1])
        output_height = int(self.output_shape[2])
        channel_offsets = []
        offset = 0
        for shape in self.concat_input_shapes:
            if int(shape[0]) != 1:
                raise RuntimeError("Concat native materialization currently requires batch size 1")
            channel_offsets.append(int(offset))
            offset += int(shape[1])
        if int(offset) != int(output_channels):
            raise RuntimeError("Concat native materialization channel offsets do not match output channels")

        covered: set[tuple[int, int, int]] = set()
        for target_block, (target_h0, target_h1, target_c0, target_count) in enumerate(target_signature):
            if int(target_c0 + target_count) > int(output_channels):
                raise RuntimeError("Concat native target signature exceeds output channels")
            for channel in range(int(target_c0), int(target_c0 + target_count)):
                for row in range(int(target_h0), int(target_h1)):
                    covered.add((int(target_block), int(channel), int(row)))

        produced: set[tuple[int, int, int]] = set()
        for input_index, signature in enumerate(input_signatures):
            branch_offset = int(channel_offsets[int(input_index)])
            branch_channels = int(self.concat_input_shapes[int(input_index)][1])
            branch_height = int(self.concat_input_shapes[int(input_index)][2])
            for source_h0, source_h1, source_c0, source_count in signature:
                if int(source_c0 + source_count) > int(branch_channels):
                    raise RuntimeError("Concat native source signature exceeds branch channels")
                global_c0 = int(branch_offset + source_c0)
                global_c1 = int(global_c0 + source_count)
                for target_block, (target_h0, target_h1, target_c0, target_count) in enumerate(target_signature):
                    h0 = max(int(source_h0), int(target_h0))
                    h1 = min(int(source_h1), int(target_h1))
                    c0 = max(int(global_c0), int(target_c0))
                    c1 = min(int(global_c1), int(target_c0 + target_count))
                    if int(h1) <= int(h0) or int(c1) <= int(c0):
                        continue
                    for channel in range(int(c0), int(c1)):
                        for row in range(int(h0), int(h1)):
                            key = (int(target_block), int(channel), int(row))
                            if key in produced:
                                raise RuntimeError("Concat native materialization target coverage overlaps")
                            produced.add(key)
        if produced != covered:
            missing = len(covered - produced)
            extra = len(produced - covered)
            raise RuntimeError(
                f"Concat native materialization target coverage mismatch: missing={int(missing)} extra={int(extra)}"
            )

    def _native_diagonals_for_input(self, input_index: int, *, slots: int, blocks=None):
        input_signatures = self._native_input_signatures()
        target_signature = self._native_output_signature()
        source_signature = input_signatures[int(input_index)]
        requested = None
        if blocks is not None:
            requested = {(int(row), int(col)) for row, col in tuple(blocks or ())}
            if not requested:
                return {}
        channel_offset = sum(int(shape[1]) for shape in self.concat_input_shapes[: int(input_index)])
        _n, channels, height, width = (int(value) for value in self.concat_input_shapes[int(input_index)])
        output_gap = int(getattr(self, "output_gap", self.concat_input_gaps[int(input_index)]))
        input_gap = int(self.concat_input_gaps[int(input_index)])
        diagonals = {}
        for source_block, (source_h0, source_h1, source_c0, source_count) in enumerate(source_signature):
            source_height = int(source_h1) - int(source_h0)
            source_global_c0 = int(channel_offset + source_c0)
            source_global_c1 = int(source_global_c0 + source_count)
            for target_block, (target_h0, target_h1, target_c0, target_count) in enumerate(target_signature):
                if requested is not None and (int(target_block), int(source_block)) not in requested:
                    continue
                h0 = max(int(source_h0), int(target_h0))
                h1 = min(int(source_h1), int(target_h1))
                c0 = max(int(source_global_c0), int(target_c0))
                c1 = min(int(source_global_c1), int(target_c0 + target_count))
                if int(h1) <= int(h0) or int(c1) <= int(c0):
                    continue
                target_height = int(target_h1) - int(target_h0)
                source_physical_height = int(source_height) * int(input_gap)
                source_physical_width = int(width) * int(input_gap)
                target_physical_height = int(target_height) * int(output_gap)
                target_physical_width = int(width) * int(output_gap)
                row_grid, col_grid = np.meshgrid(
                    np.arange(int(h0), int(h1), dtype=np.int64),
                    np.arange(int(width), dtype=np.int64),
                    indexing="ij",
                )
                rows = row_grid.reshape(-1)
                cols = col_grid.reshape(-1)
                for global_channel in range(int(c0), int(c1)):
                    source_channel = int(global_channel) - int(channel_offset)
                    if int(source_channel) < 0 or int(source_channel) >= int(channels):
                        continue
                    source_local_channel = int(source_channel) - int(source_c0)
                    target_local_channel = int(global_channel) - int(target_c0)
                    source_indices = self._flat_index(
                        int(source_local_channel),
                        rows - int(source_h0),
                        cols,
                        gap=int(input_gap),
                        height=int(source_physical_height),
                        width=int(source_physical_width),
                    ) + int(source_block) * int(slots)
                    target_indices = self._flat_index(
                        int(target_local_channel),
                        rows - int(target_h0),
                        cols,
                        gap=int(output_gap),
                        height=int(target_physical_height),
                        width=int(target_physical_width),
                    ) + int(target_block) * int(slots)
                    self._add_diagonal_entries(diagonals, source_indices, target_indices, slots=int(slots))
        return diagonals

    def _ensure_native_materialize_transforms(self, scheme):
        input_signatures = self._native_input_signatures()
        target_signature = self._native_output_signature()
        self._validate_native_concat_signatures(input_signatures, target_signature)
        slots = int(scheme.params.get_slots())
        cache_key = (
            tuple(input_signatures),
            tuple(target_signature),
            tuple(tuple(int(value) for value in shape) for shape in self.concat_input_shapes),
            tuple(int(value) for value in tuple(self.concat_input_gaps or ())),
            int(getattr(self, "output_gap", 1) or 1),
            int(slots),
            int(self.level) if self.level is not None else int(len(scheme.params.get_logq()) - 1),
        )
        if (
            self.transform_ids_by_input
            and self._compiled_backend is getattr(scheme, "backend", None)
            and getattr(self, "_native_materialize_cache_key", None) == cache_key
        ):
            return
        self.cleanup(getattr(scheme, "backend", None))
        level = int(self.level) if self.level is not None else int(len(scheme.params.get_logq()) - 1)
        target_ct_count = int(len(target_signature))
        self.fhe_output_shape = torch.Size([int(target_ct_count), int(slots)])
        self.transform_ids_by_input = []
        self.transform_sources_by_input = []
        for input_index in range(len(self.concat_input_shapes)):
            diagonals = self._native_diagonals_for_input(int(input_index), slots=int(slots))
            if not diagonals:
                raise RuntimeError(
                    f"Concat {getattr(self, 'name', '')} native materialization produced no transforms "
                    f"for input {int(input_index)}"
                )
            proxy = SimpleNamespace(
                name=f"{getattr(self, 'name', 'concat')}_native_materialize_{int(input_index)}",
                diagonals=diagonals,
                level=int(level),
                bsgs_ratio=float(self.bsgs_ratio),
                scheme=scheme,
                output_shape=self.output_shape,
                fhe_output_shape=torch.Size([int(target_ct_count), int(slots)]),
                allow_sparse_output_rows=True,
                expected_output_rows=int(target_ct_count),
            )
            if scheme.lt_evaluator.single_slot_layer_cache_enabled():
                self._install_materialize_single_slot_recipe(
                    proxy,
                    int(input_index),
                    slots=int(slots),
                    diagonals=diagonals,
                )
                proxy._single_slot_build_diagonals = (
                    lambda input_index=int(input_index), slots=int(slots): self._native_diagonals_for_input(
                        int(input_index),
                        slots=int(slots),
                    )
                )
                proxy._single_slot_build_block_diagonals = (
                    lambda blocks, input_index=int(input_index), slots=int(slots): self._native_diagonals_for_input(
                        int(input_index),
                        slots=int(slots),
                        blocks=blocks,
                    )
                )
                proxy._dense_layer_cache_build_diagonals = proxy._single_slot_build_diagonals
                proxy._dense_layer_cache_build_block_diagonals = proxy._single_slot_build_block_diagonals
            transform_ids = dict(scheme.lt_evaluator.generate_transforms(proxy))
            proxy.transform_ids = dict(transform_ids)
            if not scheme.lt_evaluator.single_slot_layer_cache_enabled():
                proxy.diagonals = {}
            self.transform_ids_by_input.append(transform_ids)
            self.transform_sources_by_input.append(proxy)
        self._native_materialize_cache_key = cache_key
        self._compiled_backend = getattr(scheme, "backend", None)

    def _install_materialize_single_slot_recipe(self, proxy, input_index: int, *, slots: int, diagonals) -> None:
        diag_indices_by_block = self._diag_indices_by_block(diagonals)
        build_diagonals = (
            lambda input_index=int(input_index), slots=int(slots): self._diagonals_for_input(
                int(input_index),
                slots=int(slots),
            )
        )
        build_block_diagonals = (
            lambda blocks, input_index=int(input_index), slots=int(slots): self._diagonals_for_input_blocks(
                int(input_index),
                slots=int(slots),
                blocks=blocks,
            )
        )
        proxy._dense_layer_cache_diag_indices_by_block = dict(diag_indices_by_block)
        proxy._dense_layer_cache_build_diagonals = build_diagonals
        proxy._dense_layer_cache_build_block_diagonals = build_block_diagonals
        proxy._single_slot_diag_indices_by_block = dict(diag_indices_by_block)
        proxy._single_slot_build_diagonals = build_diagonals
        proxy._single_slot_build_block_diagonals = build_block_diagonals

    def _ensure_materialize_transforms(self, scheme):
        if self.transform_ids_by_input and self._compiled_backend is getattr(scheme, "backend", None):
            return
        self.cleanup(getattr(scheme, "backend", None))
        if not self.concat_input_shapes:
            raise RuntimeError("Concat shapes have not been initialized by StatsTracker")
        level = int(self.level) if self.level is not None else int(len(scheme.params.get_logq()) - 1)
        slots = int(scheme.params.get_slots())
        expected_output_rows = int(math.ceil(int(np.prod(tuple(int(v) for v in self.fhe_output_shape))) / int(slots)))
        self.transform_ids_by_input = []
        self.transform_sources_by_input = []
        for input_index in range(len(self.concat_input_shapes)):
            diagonals = self._diagonals_for_input(int(input_index), slots=int(slots))
            proxy = SimpleNamespace(
                name=f"{getattr(self, 'name', 'concat')}_materialize_{int(input_index)}",
                diagonals=diagonals,
                level=int(level),
                bsgs_ratio=float(self.bsgs_ratio),
                scheme=scheme,
                output_shape=self.output_shape,
                fhe_output_shape=self.fhe_output_shape,
                allow_sparse_output_rows=True,
                expected_output_rows=int(expected_output_rows),
            )
            if scheme.lt_evaluator.single_slot_layer_cache_enabled():
                self._install_materialize_single_slot_recipe(
                    proxy,
                    int(input_index),
                    slots=int(slots),
                    diagonals=diagonals,
                )
            transform_ids = dict(scheme.lt_evaluator.generate_transforms(proxy))
            proxy.transform_ids = dict(transform_ids)
            if not scheme.lt_evaluator.single_slot_layer_cache_enabled():
                proxy.diagonals = {}
            self.transform_ids_by_input.append(transform_ids)
            self.transform_sources_by_input.append(proxy)
        self._compiled_backend = getattr(scheme, "backend", None)

    def materialize(self, parts):
        parts = tuple(parts)
        if not parts:
            raise ValueError("Concat requires at least one input")
        scheme = parts[0].scheme
        native_materialization = bool(self._native_materialization_requested())
        if bool(native_materialization):
            self._ensure_native_materialize_transforms(scheme)
            input_signatures = self._native_input_signatures()
        else:
            self._ensure_materialize_transforms(scheme)
            input_signatures = ()
        out = None
        for input_index, source in enumerate(parts):
            if bool(native_materialization):
                expected_ct_count = int(len(input_signatures[int(input_index)]))
                actual_ct_count = int(len(getattr(source, "ids", ()) or ()))
                if int(actual_ct_count) != int(expected_ct_count):
                    raise ValueError(
                        f"Concat {getattr(self, 'name', '')} native materialization input "
                        f"{int(input_index)} expected {int(expected_ct_count)} ciphertexts from its "
                        f"source signature, got {int(actual_ct_count)}"
                    )
            proxy = self.transform_sources_by_input[int(input_index)]
            proxy.transform_ids = dict(self.transform_ids_by_input[int(input_index)])
            partial = scheme.lt_evaluator.evaluate_transforms(proxy, source)
            if out is None:
                out = partial
            else:
                lhs, rhs = _align_ciphertexts_for_add(out, partial)
                out = lhs + rhs
        if bool(native_materialization):
            target_signature = self._native_output_signature()
            input_signatures = self._native_input_signatures()
            compile_io = dict(getattr(self, "_concat_native_compile_rotation_io", {}) or {})
            self._concat_native_runtime_io = {
                **compile_io,
                "runtime_lowering": "concat_explicit_native_materialize",
                "concat_native_runtime_materializer": True,
                "native_input_signature_count": int(len(input_signatures)),
                "native_input_ct_counts": [int(len(signature)) for signature in input_signatures],
                "native_output_ct_count": int(len(target_signature)),
                "native_output_target_signature": [
                    [int(value) for value in item] for item in target_signature
                ],
                "compact_fallback": False,
            }
        return out

    def cleanup(self, backend=None):
        backend = backend if backend is not None else self._compiled_backend
        delete = getattr(backend, "DeleteLinearTransform", None)
        if callable(delete):
            for transform_ids in self.transform_ids_by_input:
                for value in dict(transform_ids).values():
                    try:
                        delete(int(value))
                    except Exception:
                        pass
            for proxy in self.transform_sources_by_input:
                for value in dict(getattr(proxy, "_dense_layer_cache_active_transform_ids", {}) or {}).values():
                    try:
                        delete(int(value))
                    except Exception:
                        pass
                proxy.transform_ids = {}
                proxy._dense_layer_cache_active_transform_ids = {}
        self.transform_ids_by_input = []
        self.transform_sources_by_input = []
        self._compiled_backend = None
        self._native_materialize_cache_key = None

    def forward(self, *xs):
        if len(xs) == 1 and isinstance(xs[0], (list, tuple)):
            xs = tuple(xs[0])
        if not xs:
            raise ValueError("Concat requires at least one input")
        if self.he_mode:
            runtime = getattr(self, "layout_policy_concat_runtime", None)
            owned_parts = ()
            if runtime is not None:
                xs, owned_parts = runtime(*xs)
            if self._native_materialization_requested():
                try:
                    return self.materialize(xs)
                finally:
                    for part in owned_parts:
                        release = getattr(part, "release", None)
                        if callable(release):
                            release()
            return ConcatCipherTensor(self, xs, owned_parts=owned_parts)
        return torch.cat(tuple(xs), dim=int(self.dim))


class Identity(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def forward(self, x):
        return x
    

class Mult(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def forward(self, x, y):
        out = x * y
        if self.he_mode:
            bias = float(getattr(self, "_bootstrap_output_bias_fusion", 0.0) or 0.0)
            if bias != 0.0:
                out += bias
        return out
    

class Bootstrap(Module):
    def __init__(self, input_min, input_max, input_level):
        super().__init__()
        self.input_min = input_min 
        self.input_max = input_max 
        self.input_level = input_level
        self.prescale = 1
        self.postscale = 1
        self.constant = 0
        self.prescale_ptxt = None
        self._prescale_vec = None
        self._prescale_ptxt_cache = {}
        self.preprocess_fused = False
        self.preprocess_fusion_kind = ""
        self._bootstrap_runtime_profile = []
        self._bootstrap_runtime_call_index = 0

    def extra_repr(self):
        l_eff = len(self.scheme.params.get_logq()) - 1
        return f"l_eff={l_eff}"

    def _resolve_margin(self):
        override = os.environ.get("ORION_BOOTSTRAP_MARGIN_OVERRIDE", "").strip()
        if not override:
            return float(self.margin)
        try:
            return float(override)
        except ValueError as exc:
            raise ValueError(
                f"invalid ORION_BOOTSTRAP_MARGIN_OVERRIDE={override!r}; expected a float"
            ) from exc

    def fit(self):
        center = (self.input_min + self.input_max) / 2 
        half_range = (self.input_max - self.input_min) / 2
        self.bootstrap_margin = self._resolve_margin()
        self.low = (center - (self.bootstrap_margin * half_range)).item()
        self.high = (center + (self.bootstrap_margin * half_range)).item()

        # We'll want to scale from [A, B] into [-1, 1] using a value of the
        # form 1 / integer, so that way our multiplication back to the range
        # [A, B] (by integer) after bootstrapping doesn't consume a level.
        if self.high - self.low > 2:
            self.postscale = math.ceil((self.high - self.low) / 2)
            self.prescale = 1 / self.postscale

        self.constant = -(self.low + self.high) / 2 

    def compile(self):
        # Precompute the sparse prescale vector once. We then lazily encode it
        # at the ciphertext's actual runtime level to keep the bootstrap
        # prescale contract aligned even if the planner's guessed input level
        # differs from the level that reaches the hook at execution time.
        elements = self.fhe_input_shape.numel()
        curr_slots = 2 ** math.ceil(math.log2(elements))
        self.bootstrap_slots = curr_slots

        prescale_vec = torch.zeros(curr_slots)
        prescale_vec[:elements] = self.prescale
        self._prescale_vec = prescale_vec
        self._prescale_ptxt_cache = {}
        self.prescale_ptxt = self._get_prescale_ptxt(self.input_level)

    def _get_prescale_ptxt(self, level):
        level = int(level)
        if level not in self._prescale_ptxt_cache:
            if self._prescale_vec is None:
                raise RuntimeError("Bootstrap prescale vector has not been compiled")
            ql = self.scheme.encoder.get_moduli_chain()[level]
            self._prescale_ptxt_cache[level] = self.scheme.encoder.encode(
                self._prescale_vec, level=level, scale=ql
            )
        return self._prescale_ptxt_cache[level]

    def _debug_cipher_stats(self, x):
        ids = [int(value) for value in getattr(x, "ids", [])]
        return {
            "id_count": int(len(ids)),
            "ids": ids[:8],
            "level": int(x.level()) if hasattr(x, "level") else None,
            "scale": int(x.scale()) if hasattr(x, "scale") else None,
            "scale_log2": float(x.scale_log2()) if hasattr(x, "scale_log2") else None,
            "slots": int(x.slots()) if hasattr(x, "slots") else None,
        }

    def _is_concat_cipher_tensor(self, x) -> bool:
        return bool(type(x).__name__ == "ConcatCipherTensor" and hasattr(x, "parts"))

    def _get_part_prescale_ptxt(self, part, *, level: int, slots: int):
        cache = getattr(self, "_prescale_part_ptxt_cache", None)
        if cache is None:
            cache = {}
            self._prescale_part_ptxt_cache = cache
        ids = tuple(int(value) for value in getattr(part, "ids", ()) or ())
        on_shape = torch.Size(getattr(part, "on_shape", ()))
        block_count = max(1, int(len(ids)))
        key = (int(level), int(slots), block_count, tuple(int(value) for value in on_shape))
        if key not in cache:
            elements = int(on_shape.numel())
            vec = torch.zeros((int(block_count) * int(slots),), dtype=torch.float32)
            vec[: min(int(elements), int(vec.numel()))] = float(self.prescale)
            ql = self.scheme.encoder.get_moduli_chain()[int(level)]
            cache[key] = self.scheme.encoder.encode(vec, level=int(level), scale=ql)
        return cache[key]

    def _bootstrap_cipher_tensor(
        self,
        x,
        *,
        profile_record: dict,
        progress_name: str,
        record_debug: bool,
        preprocess_in_place: bool = True,
        prescale_ptxt=None,
    ):
        preprocess_start = time.perf_counter()
        add_shift_s = 0.0
        prescale_mul_s = 0.0
        original_input = x
        if not bool(getattr(self, "preprocess_fused", False)):
            if self.constant != 0:
                step_start = time.perf_counter()
                x = x + self.constant if not bool(preprocess_in_place) else x.__iadd__(self.constant)
                add_shift_s = float(time.perf_counter() - step_start)
            step_start = time.perf_counter()
            ptxt = prescale_ptxt if prescale_ptxt is not None else self._get_prescale_ptxt(x.level())
            if bool(preprocess_in_place):
                x = x.__imul__(ptxt)
            else:
                previous = x
                x = x * ptxt
                if previous is not original_input:
                    release = getattr(previous, "release", None)
                    if callable(release):
                        release()
            prescale_mul_s = float(time.perf_counter() - step_start)
        profile_record["timing_s"]["preprocess_total"] = float(time.perf_counter() - preprocess_start)
        profile_record["timing_s"]["preprocess_add_shift"] = float(add_shift_s)
        profile_record["timing_s"]["preprocess_prescale_mul"] = float(prescale_mul_s)

        slots = int(min(x.slots(), self.bootstrap_slots))
        profile_record["runtime_slots"] = int(slots)
        profile_record["input_to_backend"] = self._profile_cipher_batch_stats(x)
        if bool(record_debug):
            self._write_bootstrap_debug(phase="before_bootstrap", x=x, slots=slots)
        if os.environ.get("ORION_ABORT_BEFORE_BOOTSTRAP", "0") != "0":
            raise RuntimeError(
                f"aborting before bootstrap for debug: "
                f"{getattr(self, 'bootstrap_debug_name', '')}"
            )
        write_progress_event(
            "start",
            phase="bootstrap",
            layer=progress_name,
            bootstrap_slots=int(slots),
            call_index=int(profile_record.get("call_index", 0) or 0),
        )
        backend_start = time.perf_counter()
        bootstrap_input = x
        x = x.bootstrap(slots=slots)
        if not bool(preprocess_in_place) and bootstrap_input is not original_input:
            release = getattr(bootstrap_input, "release", None)
            if callable(release):
                release()
        backend_bootstrap_s = float(time.perf_counter() - backend_start)
        profile_record["timing_s"]["backend_bootstrap_call"] = float(
            profile_record["timing_s"].get("backend_bootstrap_call", 0.0) + backend_bootstrap_s
        )
        write_progress_event(
            "end",
            phase="bootstrap",
            layer=progress_name,
            bootstrap_slots=int(slots),
            call_index=int(profile_record.get("call_index", 0) or 0),
            backend_bootstrap_s=float(backend_bootstrap_s),
        )
        profile_record["output_from_backend"] = self._profile_cipher_batch_stats(x)
        if bool(record_debug):
            self._write_bootstrap_debug(phase="after_bootstrap", x=x, slots=slots)

        postprocess_start = time.perf_counter()
        postscale_mul_s = 0.0
        post_shift_s = 0.0
        if self.postscale != 1:
            step_start = time.perf_counter()
            x *= self.postscale
            postscale_mul_s = float(time.perf_counter() - step_start)
        if self.constant != 0:
            step_start = time.perf_counter()
            x -= self.constant
            post_shift_s = float(time.perf_counter() - step_start)
        profile_record["timing_s"]["postprocess_total"] = float(time.perf_counter() - postprocess_start)
        profile_record["timing_s"]["postprocess_postscale_mul"] = float(postscale_mul_s)
        profile_record["timing_s"]["postprocess_sub_shift"] = float(post_shift_s)
        profile_record["output_after_postprocess"] = self._profile_cipher_batch_stats(x)
        return x

    def _bootstrap_concat_cipher_tensor(self, x, *, profile_record: dict, total_start: float):
        parts = tuple(getattr(x, "parts", ()) or ())
        if not parts:
            raise ValueError("Cannot bootstrap an empty ConcatCipherTensor")
        progress_base = str(getattr(self, "bootstrap_debug_name", "") or self.__class__.__name__)
        profile_record["lazy_concat"] = True
        profile_record["lazy_concat_part_count"] = int(len(parts))
        part_records = []

        def transform_part(index: int, part):
            part_record = self._new_bootstrap_runtime_record()
            part_record["parent_call_index"] = int(profile_record.get("call_index", 0) or 0)
            part_record["lazy_concat_part_index"] = int(index)
            part_record["input_before_preprocess"] = self._profile_cipher_batch_stats(part)
            part_slots = int(min(part.slots(), self.bootstrap_slots))
            prescale_ptxt = None
            if not bool(getattr(self, "preprocess_fused", False)):
                prescale_ptxt = self._get_part_prescale_ptxt(
                    part,
                    level=int(part.level()),
                    slots=int(part_slots),
                )
            result = self._bootstrap_cipher_tensor(
                part,
                profile_record=part_record,
                progress_name=f"{progress_base}.part{int(index)}",
                record_debug=False,
                preprocess_in_place=False,
                prescale_ptxt=prescale_ptxt,
            )
            part_timing = dict(part_record.get("timing_s", {}) or {})
            part_record["timing_s"]["forward_total_inner"] = float(
                float(part_timing.get("preprocess_total", 0.0) or 0.0)
                + float(part_timing.get("backend_bootstrap_call", 0.0) or 0.0)
                + float(part_timing.get("postprocess_total", 0.0) or 0.0)
            )
            part_records.append(part_record)
            return result

        out = x.map_parts(transform_part)
        release_owned = getattr(x, "release_owned_parts", None)
        if callable(release_owned):
            release_owned()
        profile_record["parts"] = part_records
        for key in (
            "preprocess_total",
            "preprocess_add_shift",
            "preprocess_prescale_mul",
            "backend_bootstrap_call",
            "postprocess_total",
            "postprocess_postscale_mul",
            "postprocess_sub_shift",
        ):
            profile_record["timing_s"][key] = float(
                sum(float(record.get("timing_s", {}).get(key, 0.0) or 0.0) for record in part_records)
            )
        profile_record["runtime_slots"] = [int(record.get("runtime_slots", 0) or 0) for record in part_records]
        profile_record["output_after_postprocess"] = {
            "lazy_concat_part_count": int(len(parts)),
            "parts": [record.get("output_after_postprocess", {}) for record in part_records],
        }
        profile_record["timing_s"]["forward_total_inner"] = float(time.perf_counter() - total_start)
        return out

    def _profile_value_summary(self, values):
        values = list(values)
        if not values:
            return {"count": 0, "distinct": [], "min": None, "max": None, "sample": []}
        numeric = [value for value in values if value is not None]
        distinct = sorted({value for value in numeric})
        return {
            "count": int(len(values)),
            "distinct": distinct[:16],
            "distinct_count": int(len(distinct)),
            "min": min(numeric) if numeric else None,
            "max": max(numeric) if numeric else None,
            "sample": values[:16],
        }

    def _profile_cipher_batch_stats(self, x):
        ids = [int(value) for value in getattr(x, "ids", [])]
        backend = getattr(x, "backend", None)
        if backend is None:
            scheme = getattr(x, "scheme", None)
            backend = getattr(scheme, "backend", None)

        def collect(method_name, convert):
            method = getattr(backend, method_name, None) if backend is not None else None
            values = []
            errors = 0
            if not callable(method):
                return {"available": False, "values": [], "errors": 0}
            for ciphertext_id in ids:
                try:
                    values.append(convert(method(int(ciphertext_id))))
                except Exception:
                    values.append(None)
                    errors += 1
            return {"available": True, "values": values, "errors": int(errors)}

        levels = collect("GetCiphertextLevel", int)
        scales = collect("GetCiphertextScale", int)
        scale_log2 = collect("GetCiphertextScaleLog2", float)
        slots = collect("GetCiphertextSlots", int)
        degrees = collect("GetCiphertextDegree", int)
        return {
            "id_count": int(len(ids)),
            "ids_sample": ids[:16],
            "ids_tail": ids[-16:],
            "level": self._profile_value_summary(levels["values"]),
            "scale": self._profile_value_summary(scales["values"]),
            "scale_log2": self._profile_value_summary(scale_log2["values"]),
            "slots": self._profile_value_summary(slots["values"]),
            "degree": self._profile_value_summary(degrees["values"]),
            "query_errors": {
                "level": int(levels["errors"]),
                "scale": int(scales["errors"]),
                "scale_log2": int(scale_log2["errors"]),
                "slots": int(slots["errors"]),
                "degree": int(degrees["errors"]),
            },
        }

    def _new_bootstrap_runtime_record(self):
        self._bootstrap_runtime_call_index += 1
        return {
            "call_index": int(self._bootstrap_runtime_call_index),
            "name": str(getattr(self, "bootstrap_debug_name", "")),
            "configured_input_level": int(self.input_level),
            "configured_bootstrap_slots": int(getattr(self, "bootstrap_slots", 0) or 0),
            "prescale": float(self.prescale),
            "postscale": float(self.postscale),
            "constant": float(self.constant),
            "preprocess_fused": bool(getattr(self, "preprocess_fused", False)),
            "preprocess_fusion_kind": str(getattr(self, "preprocess_fusion_kind", "")),
            "timing_s": {},
        }

    def _write_bootstrap_debug(self, *, phase: str, x, slots: int | None = None) -> None:
        path = os.environ.get("ORION_BOOTSTRAP_DEBUG_PATH", "")
        if not path:
            return
        row = {
            "time": float(time.time()),
            "phase": str(phase),
            "name": str(getattr(self, "bootstrap_debug_name", "")),
            "input_level": int(self.input_level),
            "bootstrap_slots": int(self.bootstrap_slots),
            "runtime_slots": None if slots is None else int(slots),
            "margin": float(getattr(self, "bootstrap_margin", self.margin)),
            "prescale": float(self.prescale),
            "postscale": float(self.postscale),
            "constant": float(self.constant),
            "preprocess_fused": bool(getattr(self, "preprocess_fused", False)),
            "preprocess_fusion_kind": str(getattr(self, "preprocess_fusion_kind", "")),
            "cipher": self._debug_cipher_stats(x),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    @timer
    def forward(self, x):
        if not self.he_mode:
            return x

        total_start = time.perf_counter()
        profile_record = self._new_bootstrap_runtime_record()
        if self._is_concat_cipher_tensor(x):
            out = self._bootstrap_concat_cipher_tensor(x, profile_record=profile_record, total_start=total_start)
        else:
            profile_record["input_before_preprocess"] = self._profile_cipher_batch_stats(x)
            out = self._bootstrap_cipher_tensor(
                x,
                profile_record=profile_record,
                progress_name=str(getattr(self, "bootstrap_debug_name", "") or self.__class__.__name__),
                record_debug=True,
            )
            profile_record["timing_s"]["forward_total_inner"] = float(time.perf_counter() - total_start)
        self._bootstrap_runtime_profile.append(profile_record)

        return out
