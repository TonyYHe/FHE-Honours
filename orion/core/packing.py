import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch 
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

#-------------------#
#   Packing Logic   #
#-------------------#

def pack_conv2d(conv_layer: nn.Module, last: bool):
    slots = conv_layer.scheme.params.get_slots()
    embed_method = conv_layer.scheme.params.get_embedding_method()

    weight = conv_layer.on_weight
    if conv_layer.groups > 1:
        weight = resolve_grouped_conv(conv_layer)

    diagonals, output_rotations = direct_diagonalize_conv2d(
        conv_layer,
        weight,
        slots,
        embed_method,
        last,
    )
    
    return diagonals, output_rotations


def pack_conv2d_blocks(conv_layer: nn.Module, last: bool, blocks):
    slots = conv_layer.scheme.params.get_slots()
    embed_method = conv_layer.scheme.params.get_embedding_method()

    weight = conv_layer.on_weight
    if conv_layer.groups > 1:
        weight = resolve_grouped_conv(conv_layer)

    return direct_diagonalize_conv2d(
        conv_layer,
        weight,
        slots,
        embed_method,
        last,
        allowed_blocks={
            (int(row), int(col))
            for row, col in blocks
        },
    )[0]

def construct_conv2d_toeplitz(conv_layer, weight):
    N, on_Ci, on_Hi, on_Wi = conv_layer.fhe_input_shape
    on_Co, on_Ho, on_Wo = conv_layer.fhe_output_shape[1:]
    Ho, Wo = conv_layer.output_shape[2:]
   
    P = conv_layer.padding[0]
    D = conv_layer.dilation[0]
    iG = conv_layer.input_gap 
    oG = conv_layer.output_gap
    kW, kH = weight.shape[2:]

    def compute_first_kernel_position():
        mpx_anchors = valid_image_indices[:, :iG, :iG].reshape(-1, 1)

        row_idxs = torch.arange(0, kH*D*iG, D*iG).reshape(-1, 1)
        col_idxs = torch.arange(0, kW*D*iG, D*iG)
        kernel_offsets = valid_image_indices[0, row_idxs, col_idxs].flatten()
        
        img_pixels_touched = mpx_anchors + kernel_offsets
        return img_pixels_touched.flatten()
    
    def compute_row_interchange_map():
        output_indices = torch.arange(on_Ho * on_Wo).reshape(on_Ho, on_Wo)
        
        start_indices = output_indices[:oG, :oG].flatten()
        corner_indices = output_indices[0:(Ho*oG):oG, 0:(Wo*oG):oG].reshape(-1, 1)
        return corner_indices + start_indices
    
    # Padded input dimensions with multiplexing
    Hi_pad = on_Hi + 2*P*iG 
    Wi_pad = on_Wi + 2*P*iG

    # Initialize our sparse Toeplitz matrix
    n_rows = on_Co * on_Ho * on_Wo
    n_cols = on_Ci * Hi_pad * Wi_pad
    toeplitz = sp.lil_matrix((n_rows, n_cols), dtype="f")

    # Create an index grid for the padded input image.
    valid_image_indices = torch.arange(n_cols).reshape(on_Ci, Hi_pad, Wi_pad)

    # Pad the kernel's input and output channels to the nearest multiple
    # of gap^2 to ensure that multiplexing works.
    kernel = torch.zeros(on_Co * oG**2, on_Ci * iG**2, kW, kH) 
    kernel[:weight.shape[0], :weight.shape[1], ...] = weight

    # All the indices the kernel initially touches
    initial_kernel_position = compute_first_kernel_position()

    # Create our row-interchange map that dictates how we permute rows for 
    # optimal packing. Also return all indices that the first top-left filter 
    # value touches throughout the convolution.
    row_map = compute_row_interchange_map()
    corner_indices = valid_image_indices[0, 0:(Ho*oG):oG, 0:(Wo*oG):oG].flatten() 

    # Create offsets for the multiplexed output channels.
    out_channels = (torch.arange(on_Co) * (on_Ho * on_Wo)).reshape(on_Co, 1)

    # Flattened kernel populates rows of our Toeplitz matrix
    kernel_flat = kernel.reshape(kernel.shape[0], -1)

    # Iterate over all positions that the top-left kernel element can touch 
    # populating the correct (permuted) rows of our Toeplitz matrix.
    for i, start_idx in enumerate(corner_indices):
        rows = (row_map[i] + out_channels).reshape(-1, 1).cpu().numpy()
        cols = (initial_kernel_position + start_idx).cpu().numpy()
        toeplitz[rows, cols] = kernel_flat.cpu().numpy()

    # Keep only the columns corresponding to the non-padded input image.
    row_idxs = torch.arange(P*iG, P*iG + on_Hi).reshape(-1, 1)
    col_idxs = torch.arange(P*iG, P*iG + on_Wi)
    image_indices = valid_image_indices[:, row_idxs, col_idxs].flatten().cpu().numpy()
    toeplitz = toeplitz.tocsc()[:, image_indices]
    
    # Support batching
    toeplitz = sp.kron(sp.eye(N, dtype="f"), toeplitz, format="csr")
    return toeplitz


def _bias_clear_extent_from_fhe(
    conv_layer,
    *,
    base_h: int,
    base_w: int,
    top_beta: int,
    bottom_beta: int,
) -> tuple[int, int]:
    gap = max(1, int(getattr(conv_layer, "output_gap", 1) or 1))
    layout_h = int(base_h) + max(0, int(top_beta)) + max(0, int(bottom_beta))
    layout_w = int(base_w)
    try:
        _, _on_c, on_h, on_w = [int(value) for value in tuple(conv_layer.fhe_output_shape)]
    except (TypeError, ValueError):
        return int(layout_h), int(layout_w)
    inferred_h = int(math.ceil(max(0, int(on_h)) / float(gap)))
    inferred_w = int(math.ceil(max(0, int(on_w)) / float(gap)))
    return max(int(layout_h), int(inferred_h)), max(int(layout_w), int(inferred_w))


def construct_conv2d_bias(conv_layer):
    N, Co, Ho, Wo = conv_layer.output_shape 
    on_Co, on_Ho, on_Wo = conv_layer.fhe_output_shape[1:]
    output_layout = dict(getattr(conv_layer, "layout_policy_output_layout", {}) or {})
    top_beta = _layout_physical_top_beta(output_layout)
    bottom_beta = _layout_physical_bottom_beta(output_layout)
    total_h, total_w = _bias_clear_extent_from_fhe(
        conv_layer,
        base_h=int(Ho),
        base_w=int(Wo),
        top_beta=int(top_beta),
        bottom_beta=int(bottom_beta),
    )

    bias = conv_layer.on_bias
    bias = bias.repeat_interleave(int(total_h) * int(total_w))
    bias = bias.reshape(1, Co, int(total_h), int(total_w))
    bias_multiplexed = multiplex(bias, conv_layer.output_gap).squeeze(0)
    
    mC, mH, mW = bias_multiplexed.shape
    bias_vector = torch.zeros(on_Co, on_Ho, on_Wo)
    bias_vector[:mC, :mH, :mW] = bias_multiplexed
    bias_vector = bias_vector.flatten().repeat(N)

    return bias_vector


def pack_conv_transpose2d(conv_layer: nn.Module, last: bool):
    slots = conv_layer.scheme.params.get_slots()
    embed_method = conv_layer.scheme.params.get_embedding_method()

    diagonals, output_rotations = direct_diagonalize_conv_transpose2d(
        conv_layer,
        slots,
        embed_method,
        last,
    )

    return diagonals, output_rotations


def pack_conv_transpose2d_blocks(conv_layer: nn.Module, last: bool, blocks):
    slots = conv_layer.scheme.params.get_slots()
    embed_method = conv_layer.scheme.params.get_embedding_method()

    return direct_diagonalize_conv_transpose2d(
        conv_layer,
        slots,
        embed_method,
        last,
        allowed_blocks={
            (int(row), int(col))
            for row, col in blocks
        },
    )[0]


def _diagonal_has_nonzero(diag) -> bool:
    if isinstance(diag, torch.Tensor):
        return bool(torch.any(diag.detach() != 0).item())
    return bool(np.any(np.asarray(diag) != 0))


def prune_zero_diagonal_blocks(diagonals, *, preserve_empty_rows: bool = False):
    pruned = {}
    empty_blocks_by_row = {}
    rows_with_nonzero = set()
    for block_key, block in sorted(dict(diagonals).items()):
        row, col = (int(value) for value in block_key)
        nonzero_block = {
            int(index): diag
            for index, diag in dict(block or {}).items()
            if _diagonal_has_nonzero(diag)
        }
        if nonzero_block:
            pruned[(int(row), int(col))] = nonzero_block
            rows_with_nonzero.add(int(row))
        else:
            empty_blocks_by_row.setdefault(int(row), []).append(((int(row), int(col)), dict(block or {})))
    if bool(preserve_empty_rows):
        for row, empty_blocks in sorted(empty_blocks_by_row.items()):
            if int(row) in rows_with_nonzero or not empty_blocks:
                continue
            block_key, block = empty_blocks[0]
            pruned[block_key] = block
    return pruned


def _prune_zero_diagonal_blocks(diagonals):
    return prune_zero_diagonal_blocks(diagonals)


def _pad_fhe_tensor(matrix: torch.Tensor, fhe_shape: torch.Size) -> torch.Tensor:
    padded = torch.zeros(fhe_shape, dtype=matrix.dtype)
    padded[
        : matrix.shape[0],
        : matrix.shape[1],
        : matrix.shape[2],
        : matrix.shape[3],
    ] = matrix
    return padded


def _demultiplex(matrix: torch.Tensor, gap: int, channels: int, height: int, width: int) -> torch.Tensor:
    if gap == 1:
        return matrix[:, :channels, :height, :width]

    demux = F.pixel_unshuffle(matrix, gap)
    return demux[:, :channels, :height, :width]


def construct_conv_transpose2d_toeplitz(conv_layer):
    """
    Build the transposed-convolution Toeplitz matrix in the multiplexed space.

    Rather than manually reasoning about gap-aligned subpixels, we probe the
    operator with basis vectors in the FHE layout, demultiplex them back to the
    clear layout, run PyTorch's reference `conv_transpose2d`, then remultiplex
    the result. This keeps the packing logic aligned with Orion's actual tensor
    layout and naturally handles stride, padding, dilation, output_padding, and
    grouped convolutions.
    """
    N, on_Ci, on_Hi, on_Wi = conv_layer.fhe_input_shape
    on_Co, on_Ho, on_Wo = conv_layer.fhe_output_shape[1:]
    Hi, Wi = conv_layer.input_shape[2:]

    n_rows = on_Co * on_Ho * on_Wo
    n_cols = on_Ci * on_Hi * on_Wi

    row_coords = []
    col_coords = []
    data_values = []

    for input_col in range(n_cols):
        basis_fhe = torch.zeros((1, on_Ci, on_Hi, on_Wi), dtype=conv_layer.on_weight.dtype)
        basis_fhe.view(-1)[input_col] = 1.0

        clear_input = _demultiplex(
            basis_fhe,
            conv_layer.input_gap,
            conv_layer.in_channels,
            Hi,
            Wi,
        )
        clear_output = F.conv_transpose2d(
            clear_input,
            conv_layer.on_weight,
            None,
            conv_layer.stride,
            conv_layer.padding,
            conv_layer.output_padding,
            conv_layer.groups,
            conv_layer.dilation,
        )
        output_fhe = _pad_fhe_tensor(
            multiplex(clear_output, conv_layer.output_gap),
            torch.Size((1, on_Co, on_Ho, on_Wo)),
        )

        flat_output = output_fhe.view(-1)
        nonzero_idx = flat_output.nonzero(as_tuple=False).flatten()
        row_coords.extend(nonzero_idx.tolist())
        col_coords.extend([input_col] * int(nonzero_idx.numel()))
        data_values.extend(flat_output[nonzero_idx].tolist())

    if row_coords:
        toeplitz = sp.coo_matrix(
            (data_values, (row_coords, col_coords)),
            shape=(n_rows, n_cols),
            dtype="f",
        ).tocsr()
    else:
        toeplitz = sp.csr_matrix((n_rows, n_cols), dtype="f")

    toeplitz = sp.kron(sp.eye(N, dtype="f"), toeplitz, format="csr")
    return toeplitz


def construct_conv_transpose2d_bias(conv_layer):
    N, Co, Ho, Wo = conv_layer.output_shape
    on_Co, on_Ho, on_Wo = conv_layer.fhe_output_shape[1:]
    output_layout = dict(getattr(conv_layer, "layout_policy_output_layout", {}) or {})
    top_beta = _layout_physical_top_beta(output_layout)
    bottom_beta = _layout_physical_bottom_beta(output_layout)
    total_h, total_w = _bias_clear_extent_from_fhe(
        conv_layer,
        base_h=int(Ho),
        base_w=int(Wo),
        top_beta=int(top_beta),
        bottom_beta=int(bottom_beta),
    )

    bias = conv_layer.on_bias
    bias = bias.repeat_interleave(int(total_h) * int(total_w))
    bias = bias.reshape(1, Co, int(total_h), int(total_w))
    bias_multiplexed = multiplex(bias, conv_layer.output_gap).squeeze(0)

    mC, mH, mW = bias_multiplexed.shape
    bias_vector = torch.zeros(on_Co, on_Ho, on_Wo)
    bias_vector[:mC, :mH, :mW] = bias_multiplexed
    bias_vector = bias_vector.flatten().repeat(N)

    return bias_vector


class _DirectDiagonalAccumulator:
    def __init__(
        self,
        matrix_shape: tuple[int, int],
        num_slots: int,
        embed_method: str,
        is_last_layer: bool,
        *,
        allow_hybrid: bool = True,
        verbose: bool = True,
        allowed_blocks: set[tuple[int, int]] | None = None,
    ) -> None:
        self.matrix_height = int(matrix_shape[0])
        self.matrix_width = int(matrix_shape[1])
        self.num_slots = int(num_slots)
        self.num_block_rows = math.ceil(self.matrix_height / self.num_slots)
        self.num_block_cols = math.ceil(self.matrix_width / self.num_slots)
        if bool(verbose):
            print(f"├── embed method: {embed_method}")
            print(f"├── original matrix shape: {matrix_shape}")
            print(f"├── # blocks (rows, cols) = {(self.num_block_rows, self.num_block_cols)}")

        if (
            allow_hybrid
            and self.num_block_rows == 1
            and embed_method == "hybrid"
            and not is_last_layer
        ):
            self.block_height = 2 ** math.ceil(math.log2(self.matrix_height))
            self.output_rotations = int(math.log2(self.num_slots // self.block_height))
        else:
            self.block_height = self.num_slots
            self.output_rotations = 0
        self.allowed_blocks = (
            {
                (int(row), int(col))
                for row, col in allowed_blocks
            }
            if allowed_blocks is not None
            else None
        )

        self.resized_shape = (
            self.num_block_rows * self.block_height,
            self.num_block_cols * self.num_slots,
        )
        if bool(verbose):
            print(f"├── resized matrix shape: {self.resized_shape}")
            print(f"├── # output rotations: {self.output_rotations}")
        self.diagonals_by_block: dict[tuple[int, int], dict[int, np.ndarray]] = {}

    def _allowed_entry_mask(self, block_rows: np.ndarray, block_cols: np.ndarray) -> np.ndarray | None:
        if self.allowed_blocks is None:
            return None
        allowed = np.zeros(np.asarray(block_rows).shape, dtype=bool)
        for row, col in self.allowed_blocks:
            allowed |= (block_rows == int(row)) & (block_cols == int(col))
        return allowed

    def add_entries(self, rows, cols, values) -> None:
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if rows.size == 0:
            return

        valid = (
            (rows >= 0)
            & (rows < self.matrix_height)
            & (cols >= 0)
            & (cols < self.matrix_width)
            & (values != 0.0)
        )
        if not bool(np.any(valid)):
            return
        rows = rows[valid]
        cols = cols[valid]
        values = values[valid]

        if self.block_height == self.num_slots:
            block_rows = rows // self.num_slots
            local_rows = rows - block_rows * self.num_slots
            block_cols = cols // self.num_slots
            local_cols = cols - block_cols * self.num_slots
            diag_idxs = (local_cols - local_rows) % self.num_slots
            positions = local_rows
        else:
            block_rows = np.zeros_like(rows)
            local_rows = rows
            block_cols = cols // self.num_slots
            local_cols = cols - block_cols * self.num_slots
            diag_idxs = (local_cols - local_rows) % self.block_height
            positions = (local_cols - diag_idxs) % self.num_slots

        allowed_mask = self._allowed_entry_mask(block_rows, block_cols)
        if allowed_mask is not None:
            if not bool(np.any(allowed_mask)):
                return
            block_rows = block_rows[allowed_mask]
            block_cols = block_cols[allowed_mask]
            diag_idxs = diag_idxs[allowed_mask]
            positions = positions[allowed_mask]
            values = values[allowed_mask]

        order = np.lexsort((diag_idxs, block_cols, block_rows))
        block_rows = block_rows[order]
        block_cols = block_cols[order]
        diag_idxs = diag_idxs[order]
        positions = positions[order]
        values = values[order]

        start = 0
        while start < int(values.size):
            end = start + 1
            while (
                end < int(values.size)
                and int(block_rows[end]) == int(block_rows[start])
                and int(block_cols[end]) == int(block_cols[start])
                and int(diag_idxs[end]) == int(diag_idxs[start])
            ):
                end += 1

            block_key = (int(block_rows[start]), int(block_cols[start]))
            diag_idx = int(diag_idxs[start])
            block = self.diagonals_by_block.setdefault(block_key, {})
            diagonal = block.get(diag_idx)
            if diagonal is None:
                diagonal = np.zeros((self.num_slots,), dtype=np.float32)
                block[diag_idx] = diagonal
            np.add.at(diagonal, positions[start:end].astype(np.int64), values[start:end])
            start = end

    def add_constant_entries(self, rows, cols, value: float) -> None:
        coeff = float(value)
        if coeff == 0.0:
            return
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        if rows.size == 0:
            return

        valid = (
            (rows >= 0)
            & (rows < self.matrix_height)
            & (cols >= 0)
            & (cols < self.matrix_width)
        )
        if not bool(np.any(valid)):
            return
        rows = rows[valid]
        cols = cols[valid]

        if self.block_height == self.num_slots:
            block_rows = rows // self.num_slots
            local_rows = rows - block_rows * self.num_slots
            block_cols = cols // self.num_slots
            local_cols = cols - block_cols * self.num_slots
            diag_idxs = (local_cols - local_rows) % self.num_slots
            positions = local_rows
        else:
            block_rows = np.zeros_like(rows)
            local_rows = rows
            block_cols = cols // self.num_slots
            local_cols = cols - block_cols * self.num_slots
            diag_idxs = (local_cols - local_rows) % self.block_height
            positions = (local_cols - diag_idxs) % self.num_slots

        allowed_mask = self._allowed_entry_mask(block_rows, block_cols)
        if allowed_mask is not None:
            if not bool(np.any(allowed_mask)):
                return
            block_rows = block_rows[allowed_mask]
            block_cols = block_cols[allowed_mask]
            diag_idxs = diag_idxs[allowed_mask]
            positions = positions[allowed_mask]

        entry_count = int(block_rows.size)
        if entry_count <= 1:
            bounds = np.array([int(entry_count)], dtype=np.int64)
        else:
            change = (
                (block_rows[1:] != block_rows[:-1])
                | (block_cols[1:] != block_cols[:-1])
                | (diag_idxs[1:] != diag_idxs[:-1])
            )
            bounds = np.concatenate(
                (
                    np.flatnonzero(change).astype(np.int64) + 1,
                    np.array([int(entry_count)], dtype=np.int64),
                )
            )

        start = 0
        for end_value in bounds:
            end = int(end_value)
            block_key = (int(block_rows[start]), int(block_cols[start]))
            diag_idx = int(diag_idxs[start])
            block = self.diagonals_by_block.setdefault(block_key, {})
            diagonal = block.get(diag_idx)
            if diagonal is None:
                diagonal = np.zeros((self.num_slots,), dtype=np.float32)
                block[diag_idx] = diagonal
            diagonal[positions[start:end].astype(np.int64)] += coeff
            start = end

    def merge_from(self, other: "_DirectDiagonalAccumulator") -> None:
        for block_key, source_block in other.diagonals_by_block.items():
            target_block = self.diagonals_by_block.setdefault(block_key, {})
            for diag_idx, source_diag in source_block.items():
                target_diag = target_block.get(int(diag_idx))
                if target_diag is None:
                    target_block[int(diag_idx)] = source_diag
                else:
                    target_diag += source_diag

    def finish(self, start_time: float) -> tuple[dict[tuple[int, int], dict[int, np.ndarray]], int]:
        out: dict[tuple[int, int], dict[int, np.ndarray]] = {}
        total_diagonals = 0
        if self.allowed_blocks is None:
            block_keys = [
                (int(block_row), int(block_col))
                for block_row in range(self.num_block_rows)
                for block_col in range(self.num_block_cols)
            ]
        else:
            block_keys = sorted(
                (int(row), int(col))
                for row, col in self.allowed_blocks
                if 0 <= int(row) < int(self.num_block_rows) and 0 <= int(col) < int(self.num_block_cols)
            )
        for block_row, block_col in block_keys:
            block = self.diagonals_by_block.get((int(block_row), int(block_col)), {})
            if not block:
                out[(int(block_row), int(block_col))] = {
                    0: np.zeros((int(self.num_slots),), dtype=np.float32)
                }
                continue
            total_diagonals += len(block)
            out[(int(block_row), int(block_col))] = {
                int(diag_idx): np.ascontiguousarray(block[int(diag_idx)], dtype=np.float32)
                for diag_idx in sorted(int(value) for value in block.keys())
            }
        elapsed_time = time.time() - start_time
        print(f"├── time to pack (s): {elapsed_time:.2f}")
        print(f"├── # diagonals = {total_diagonals}")
        return out, int(self.output_rotations)


def _direct_pack_worker_count(item_count: int) -> int:
    if int(item_count) <= 1:
        return 1
    raw = os.environ.get("ORION_PACK_CONV_WORKERS")
    if raw is None:
        raw = os.environ.get("ORION_DIRECT_PACK_WORKERS")
    if raw is None:
        return 1
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        requested = 1
    return max(1, min(int(item_count), int(requested)))


def _split_contiguous_ranges(item_count: int, workers: int) -> list[tuple[int, int]]:
    item_count = int(item_count)
    workers = max(1, min(int(item_count), int(workers)))
    base, extra = divmod(item_count, workers)
    ranges: list[tuple[int, int]] = []
    start = 0
    for worker_index in range(workers):
        end = start + int(base) + (1 if int(worker_index) < int(extra) else 0)
        if start < end:
            ranges.append((int(start), int(end)))
        start = int(end)
    return ranges


def _packed_flat_indices(
    channel: int,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    gap: int,
    height: int,
    width: int,
    row_offset: int = 0,
) -> np.ndarray:
    gap = int(gap)
    phase = int(channel) % int(gap * gap)
    packed_channel = int(channel) // int(gap * gap)
    packed_rows = rows.astype(np.int64) * int(gap) + int(phase // gap) + int(row_offset)
    packed_cols = cols.astype(np.int64) * int(gap) + int(phase % gap)
    return (
        (int(packed_channel) * int(height) + packed_rows) * int(width)
        + packed_cols
    ).astype(np.int64)


def _packed_flat_indices_for_channels(
    channels: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    gap: int,
    height: int,
    width: int,
    row_offset: int = 0,
) -> np.ndarray:
    channels = np.asarray(channels, dtype=np.int64).reshape(-1)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    gap = int(gap)
    phase = channels % int(gap * gap)
    packed_channel = channels // int(gap * gap)
    channel_base = (
        (packed_channel * int(height) + phase // int(gap) + int(row_offset)) * int(width)
        + phase % int(gap)
    ).astype(np.int64)
    spatial = (rows * int(gap) * int(width) + cols * int(gap)).astype(np.int64)
    return channel_base[:, None] + spatial[None, :]


def _diagonal_key_values(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    matrix_height: int,
    matrix_width: int,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
    allow_hybrid: bool = True,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    num_slots = int(num_slots)
    num_block_rows = math.ceil(int(matrix_height) / int(num_slots))
    num_block_cols = math.ceil(int(matrix_width) / int(num_slots))
    if (
        bool(allow_hybrid)
        and int(num_block_rows) == 1
        and str(embed_method) == "hybrid"
        and not bool(is_last_layer)
    ):
        block_height = 2 ** math.ceil(math.log2(int(matrix_height)))
    else:
        block_height = int(num_slots)

    if int(block_height) == int(num_slots):
        block_rows = rows // int(num_slots)
        local_rows = rows - block_rows * int(num_slots)
        block_cols = cols // int(num_slots)
        local_cols = cols - block_cols * int(num_slots)
        diag_idxs = (local_cols - local_rows) % int(num_slots)
    else:
        block_rows = np.zeros_like(rows)
        block_cols = cols // int(num_slots)
        local_cols = cols - block_cols * int(num_slots)
        diag_idxs = (local_cols - rows) % int(block_height)

    return ((block_rows * int(num_block_cols) + block_cols) * int(num_slots) + diag_idxs).reshape(-1)


def _append_diagonal_key_chunk(
    chunks: list[np.ndarray],
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    matrix_height: int,
    matrix_width: int,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
    allow_hybrid: bool = True,
) -> None:
    values = _diagonal_key_values(
        rows,
        cols,
        matrix_height=int(matrix_height),
        matrix_width=int(matrix_width),
        num_slots=int(num_slots),
        embed_method=str(embed_method),
        is_last_layer=bool(is_last_layer),
        allow_hybrid=bool(allow_hybrid),
    )
    if values.size:
        chunks.append(np.unique(values))


def _count_diagonal_key_chunks(chunks: list[np.ndarray]) -> int:
    if not chunks:
        return 0
    if len(chunks) == 1:
        return int(chunks[0].size)
    return int(np.unique(np.concatenate(chunks)).size)


def _diagonal_key_chunks_to_indices(
    chunks: list[np.ndarray],
    *,
    matrix_height: int,
    matrix_width: int,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
    allow_hybrid: bool = True,
) -> dict[tuple[int, int], tuple[int, ...]]:
    num_slots = int(num_slots)
    num_block_rows = math.ceil(int(matrix_height) / int(num_slots))
    num_block_cols = math.ceil(int(matrix_width) / int(num_slots))
    if (
        bool(allow_hybrid)
        and int(num_block_rows) == 1
        and str(embed_method) == "hybrid"
        and not bool(is_last_layer)
    ):
        block_height = 2 ** math.ceil(math.log2(int(matrix_height)))
    else:
        block_height = int(num_slots)

    indices: dict[tuple[int, int], set[int]] = {
        (int(row), int(col)): set()
        for row in range(int(num_block_rows))
        for col in range(int(num_block_cols))
    }
    if chunks:
        keys = np.unique(np.concatenate(chunks)) if len(chunks) > 1 else np.unique(chunks[0])
        for raw_key in np.asarray(keys, dtype=np.int64).reshape(-1):
            key = int(raw_key)
            block_linear = int(key // int(num_slots))
            diag_idx = int(key % int(num_slots))
            row = int(block_linear // int(num_block_cols))
            col = int(block_linear % int(num_block_cols))
            if 0 <= row < int(num_block_rows) and 0 <= col < int(num_block_cols):
                indices.setdefault((int(row), int(col)), set()).add(int(diag_idx))
    return {
        block: tuple(sorted(values)) if values else (0,)
        for block, values in sorted(indices.items())
    }


def _packed_output_rotations(
    *,
    matrix_height: int,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
    allow_hybrid: bool = True,
) -> int:
    num_block_rows = math.ceil(int(matrix_height) / int(num_slots))
    if (
        bool(allow_hybrid)
        and int(num_block_rows) == 1
        and str(embed_method) == "hybrid"
        and not bool(is_last_layer)
    ):
        block_height = 2 ** math.ceil(math.log2(int(matrix_height)))
        return int(math.log2(int(num_slots) // int(block_height)))
    return 0


def _channel_chunk_shape(spatial_count: int, outer_count: int, inner_count: int, max_chunk_items: int) -> tuple[int, int]:
    channel_area = max(1, int(max_chunk_items) // max(1, int(spatial_count)))
    outer = max(1, min(int(outer_count), int(math.sqrt(channel_area))))
    inner = max(1, min(int(inner_count), int(channel_area) // int(outer)))
    return int(outer), int(inner)


def _layout_top_beta(layout: dict) -> int:
    return max(0, int(layout.get("top_beta", layout.get("alpha", 0)) or 0))


def _layout_bottom_beta(layout: dict) -> int:
    return max(0, int(layout.get("bottom_beta", layout.get("beta", 0)) or 0))


def _layout_physical_top_beta(layout: dict) -> int:
    return max(
        0,
        int(
            layout.get(
                "physical_top_beta",
                layout.get("top_beta", layout.get("alpha", 0)),
            )
            or 0
        ),
    )


def _layout_physical_bottom_beta(layout: dict) -> int:
    return max(
        0,
        int(
            layout.get(
                "physical_bottom_beta",
                layout.get("bottom_beta", layout.get("beta", 0)),
            )
            or 0
        ),
    )


def _conv2d_spatial_cache(conv_layer) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    _, _, hi, wi = [int(value) for value in conv_layer.input_shape]
    _, _, ho, wo = [int(value) for value in conv_layer.output_shape]
    stride_h, stride_w = [int(value) for value in conv_layer.stride]
    pad_h, pad_w = [int(value) for value in conv_layer.padding]
    dil_h, dil_w = [int(value) for value in conv_layer.dilation]
    k_h, k_w = [int(value) for value in conv_layer.kernel_size]
    output_layout = dict(getattr(conv_layer, "layout_policy_output_layout", {}) or {})
    output_top_beta = _layout_top_beta(output_layout)
    output_bottom_beta = _layout_bottom_beta(output_layout)
    output_materialization = str(getattr(conv_layer, "layout_policy_output_materialization", "") or "")
    fuse_output_relayout = output_materialization == "fused_relayout"

    target_oh_grid, ow_grid = np.meshgrid(
        np.arange(-int(output_top_beta), int(ho + output_bottom_beta), dtype=np.int64),
        np.arange(wo, dtype=np.int64),
        indexing="ij",
    )
    if bool(fuse_output_relayout):
        op_oh_grid = target_oh_grid.copy()
        top = op_oh_grid < 0
        bottom = op_oh_grid >= int(ho)
        op_oh_grid[top] = op_oh_grid[top] + int(output_top_beta)
        op_oh_grid[bottom] = op_oh_grid[bottom] - int(output_bottom_beta)
        op_oh_grid = np.clip(op_oh_grid, 0, max(0, int(ho) - 1))
    else:
        op_oh_grid = target_oh_grid
    cache = {}
    for kh in range(k_h):
        for kw in range(k_w):
            ih_grid = op_oh_grid * int(stride_h) - int(pad_h) + int(kh) * int(dil_h)
            iw_grid = ow_grid * int(stride_w) - int(pad_w) + int(kw) * int(dil_w)
            valid = (
                (ih_grid >= 0)
                & (ih_grid < int(hi))
                & (iw_grid >= 0)
                & (iw_grid < int(wi))
            )
            cache[(int(kh), int(kw))] = (
                target_oh_grid[valid].reshape(-1),
                ow_grid[valid].reshape(-1),
                ih_grid[valid].reshape(-1),
                iw_grid[valid].reshape(-1),
            )
    return cache


def direct_diagonalize_conv2d(
    conv_layer,
    weight,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
    *,
    allow_hybrid: bool = True,
    allowed_blocks: set[tuple[int, int]] | None = None,
):
    start_time = time.time()
    matrix_shape = (
        int(torch.Size(conv_layer.fhe_output_shape).numel()),
        int(torch.Size(conv_layer.fhe_input_shape).numel()),
    )
    accumulator = _DirectDiagonalAccumulator(
        matrix_shape,
        int(num_slots),
        str(embed_method),
        bool(is_last_layer),
        allow_hybrid=bool(allow_hybrid),
        allowed_blocks=allowed_blocks,
    )

    n_batch, ci, _, _ = [int(value) for value in conv_layer.input_shape]
    _, co, _, _ = [int(value) for value in conv_layer.output_shape]
    _, on_ci, on_hi, on_wi = [int(value) for value in conv_layer.fhe_input_shape]
    _, on_co, on_ho, on_wo = [int(value) for value in conv_layer.fhe_output_shape]
    input_gap = int(conv_layer.input_gap)
    output_gap = int(conv_layer.output_gap)
    input_row_offset = max(0, int(getattr(conv_layer, "layout_policy_input_row_offset", 0) or 0))
    output_row_offset = max(0, int(getattr(conv_layer, "layout_policy_output_row_offset", 0) or 0))
    input_block_size = int(on_ci * on_hi * on_wi)
    output_block_size = int(on_co * on_ho * on_wo)

    weight_np = weight.detach().cpu().to(dtype=torch.float32).numpy()
    spatial_cache = _conv2d_spatial_cache(conv_layer)

    def fill_accumulator(
        target: _DirectDiagonalAccumulator,
        oc_start: int,
        oc_end: int,
    ) -> _DirectDiagonalAccumulator:
        for oc in range(int(oc_start), int(oc_end)):
            for ic in range(int(ci)):
                for kh in range(int(weight_np.shape[2])):
                    for kw in range(int(weight_np.shape[3])):
                        coeff = float(weight_np[int(oc), int(ic), int(kh), int(kw)])
                        if coeff == 0.0:
                            continue
                        oh, ow, ih, iw = spatial_cache[(int(kh), int(kw))]
                        if oh.size == 0:
                            continue
                        local_rows = _packed_flat_indices(
                            int(oc),
                            oh,
                            ow,
                            gap=int(output_gap),
                            height=int(on_ho),
                            width=int(on_wo),
                            row_offset=int(output_row_offset),
                        )
                        local_cols = _packed_flat_indices(
                            int(ic),
                            ih,
                            iw,
                            gap=int(input_gap),
                            height=int(on_hi),
                            width=int(on_wi),
                            row_offset=int(input_row_offset),
                        )
                        for batch in range(int(n_batch)):
                            target.add_constant_entries(
                                local_rows + int(batch) * int(output_block_size),
                                local_cols + int(batch) * int(input_block_size),
                                float(coeff),
                            )
        return target

    workers = _direct_pack_worker_count(int(co))
    ranges = _split_contiguous_ranges(int(co), int(workers))
    if len(ranges) <= 1:
        fill_accumulator(accumulator, 0, int(co))
    else:
        print(f"├── direct pack workers: {len(ranges)}")
        with ThreadPoolExecutor(max_workers=len(ranges), thread_name_prefix="orion-pack-conv2d") as executor:
            futures = []
            for oc_start, oc_end in ranges:
                local = _DirectDiagonalAccumulator(
                    matrix_shape,
                    int(num_slots),
                    str(embed_method),
                    bool(is_last_layer),
                    allow_hybrid=bool(allow_hybrid),
                    verbose=False,
                    allowed_blocks=allowed_blocks,
                )
                futures.append(executor.submit(fill_accumulator, local, int(oc_start), int(oc_end)))
            for future in futures:
                accumulator.merge_from(future.result())

    return accumulator.finish(start_time)


def _tconv2d_spatial_cache(conv_layer) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    _, _, hi, wi = [int(value) for value in conv_layer.input_shape]
    _, _, ho, wo = [int(value) for value in conv_layer.output_shape]
    stride_h, stride_w = [int(value) for value in conv_layer.stride]
    pad_h, pad_w = [int(value) for value in conv_layer.padding]
    dil_h, dil_w = [int(value) for value in conv_layer.dilation]
    k_h, k_w = [int(value) for value in conv_layer.kernel_size]

    ih_grid, iw_grid = np.meshgrid(
        np.arange(hi, dtype=np.int64),
        np.arange(wi, dtype=np.int64),
        indexing="ij",
    )
    cache = {}
    for kh in range(k_h):
        for kw in range(k_w):
            oh_grid = ih_grid * int(stride_h) - int(pad_h) + int(kh) * int(dil_h)
            ow_grid = iw_grid * int(stride_w) - int(pad_w) + int(kw) * int(dil_w)
            valid = (
                (oh_grid >= 0)
                & (oh_grid < int(ho))
                & (ow_grid >= 0)
                & (ow_grid < int(wo))
            )
            cache[(int(kh), int(kw))] = (
                ih_grid[valid].reshape(-1),
                iw_grid[valid].reshape(-1),
                oh_grid[valid].reshape(-1),
                ow_grid[valid].reshape(-1),
            )
    return cache


def direct_diagonalize_conv_transpose2d(
    conv_layer,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
    *,
    allow_hybrid: bool = True,
    allowed_blocks: set[tuple[int, int]] | None = None,
):
    start_time = time.time()
    matrix_shape = (
        int(torch.Size(conv_layer.fhe_output_shape).numel()),
        int(torch.Size(conv_layer.fhe_input_shape).numel()),
    )
    accumulator = _DirectDiagonalAccumulator(
        matrix_shape,
        int(num_slots),
        str(embed_method),
        bool(is_last_layer),
        allow_hybrid=bool(allow_hybrid),
        allowed_blocks=allowed_blocks,
    )

    n_batch, ci, _, _ = [int(value) for value in conv_layer.input_shape]
    _, co, _, _ = [int(value) for value in conv_layer.output_shape]
    _, on_ci, on_hi, on_wi = [int(value) for value in conv_layer.fhe_input_shape]
    _, on_co, on_ho, on_wo = [int(value) for value in conv_layer.fhe_output_shape]
    input_gap = int(conv_layer.input_gap)
    output_gap = int(conv_layer.output_gap)
    input_row_offset = max(0, int(getattr(conv_layer, "layout_policy_input_row_offset", 0) or 0))
    output_row_offset = max(0, int(getattr(conv_layer, "layout_policy_output_row_offset", 0) or 0))
    input_block_size = int(on_ci * on_hi * on_wi)
    output_block_size = int(on_co * on_ho * on_wo)

    weight_np = conv_layer.on_weight.detach().cpu().to(dtype=torch.float32).numpy()
    groups = int(conv_layer.groups)
    in_channels_per_group = int(conv_layer.in_channels // groups)
    out_channels_per_group = int(conv_layer.out_channels // groups)
    spatial_cache = _tconv2d_spatial_cache(conv_layer)

    def fill_accumulator(
        target: _DirectDiagonalAccumulator,
        ic_start: int,
        ic_end: int,
    ) -> _DirectDiagonalAccumulator:
        for ic in range(int(ic_start), int(ic_end)):
            group = int(ic) // int(in_channels_per_group)
            oc_offset = int(group) * int(out_channels_per_group)
            for oc_rel in range(int(out_channels_per_group)):
                oc = int(oc_offset + oc_rel)
                if int(oc) >= int(co):
                    continue
                for kh in range(int(weight_np.shape[2])):
                    for kw in range(int(weight_np.shape[3])):
                        coeff = float(weight_np[int(ic), int(oc_rel), int(kh), int(kw)])
                        if coeff == 0.0:
                            continue
                        ih, iw, oh, ow = spatial_cache[(int(kh), int(kw))]
                        if ih.size == 0:
                            continue
                        local_rows = _packed_flat_indices(
                            int(oc),
                            oh,
                            ow,
                            gap=int(output_gap),
                            height=int(on_ho),
                            width=int(on_wo),
                            row_offset=int(output_row_offset),
                        )
                        local_cols = _packed_flat_indices(
                            int(ic),
                            ih,
                            iw,
                            gap=int(input_gap),
                            height=int(on_hi),
                            width=int(on_wi),
                            row_offset=int(input_row_offset),
                        )
                        for batch in range(int(n_batch)):
                            target.add_constant_entries(
                                local_rows + int(batch) * int(output_block_size),
                                local_cols + int(batch) * int(input_block_size),
                                float(coeff),
                            )
        return target

    workers = _direct_pack_worker_count(int(ci))
    ranges = _split_contiguous_ranges(int(ci), int(workers))
    if len(ranges) <= 1:
        fill_accumulator(accumulator, 0, int(ci))
    else:
        print(f"├── direct pack workers: {len(ranges)}")
        with ThreadPoolExecutor(max_workers=len(ranges), thread_name_prefix="orion-pack-tconv2d") as executor:
            futures = []
            for ic_start, ic_end in ranges:
                local = _DirectDiagonalAccumulator(
                    matrix_shape,
                    int(num_slots),
                    str(embed_method),
                    bool(is_last_layer),
                    allow_hybrid=bool(allow_hybrid),
                    verbose=False,
                    allowed_blocks=allowed_blocks,
                )
                futures.append(executor.submit(fill_accumulator, local, int(ic_start), int(ic_end)))
            for future in futures:
                accumulator.merge_from(future.result())

    return accumulator.finish(start_time)


def estimate_direct_conv2d_diagonal_count(
    conv_layer,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
    *,
    allow_hybrid: bool = True,
    max_chunk_items: int | None = None,
    return_indices: bool = False,
) -> int:
    """Count packed block diagonals without materializing plaintext payloads."""

    if max_chunk_items is None:
        max_chunk_items = int(os.environ.get("ORION_PACK_DIAGONAL_COUNT_MAX_ITEMS", "2000000"))
    max_chunk_items = max(1, int(max_chunk_items))

    weight_np = conv_layer.on_weight.detach().cpu().to(dtype=torch.float32).numpy()

    n_batch, ci, _, _ = [int(value) for value in conv_layer.input_shape]
    _, co, _, _ = [int(value) for value in conv_layer.output_shape]
    _, on_ci, on_hi, on_wi = [int(value) for value in conv_layer.fhe_input_shape]
    _, on_co, on_ho, on_wo = [int(value) for value in conv_layer.fhe_output_shape]
    input_gap = int(conv_layer.input_gap)
    output_gap = int(conv_layer.output_gap)
    input_row_offset = max(0, int(getattr(conv_layer, "layout_policy_input_row_offset", 0) or 0))
    output_row_offset = max(0, int(getattr(conv_layer, "layout_policy_output_row_offset", 0) or 0))
    input_block_size = int(on_ci * on_hi * on_wi)
    output_block_size = int(on_co * on_ho * on_wo)
    matrix_shape = (
        int(torch.Size(conv_layer.fhe_output_shape).numel()),
        int(torch.Size(conv_layer.fhe_input_shape).numel()),
    )
    spatial_cache = _conv2d_spatial_cache(conv_layer)
    groups = int(conv_layer.groups)
    in_channels_per_group = int(conv_layer.in_channels // groups)
    out_channels_per_group = int(conv_layer.out_channels // groups)

    key_chunks: list[np.ndarray] = []
    for kh in range(int(weight_np.shape[2])):
        for kw in range(int(weight_np.shape[3])):
            oh, ow, ih, iw = spatial_cache[(int(kh), int(kw))]
            if oh.size == 0:
                continue
            for group in range(int(groups)):
                oc_group_start = int(group) * int(out_channels_per_group)
                oc_group_end = min(int(co), int(oc_group_start) + int(out_channels_per_group))
                ic_group_start = int(group) * int(in_channels_per_group)
                ic_group_end = min(int(ci), int(ic_group_start) + int(in_channels_per_group))
                output_chunk, input_chunk = _channel_chunk_shape(
                    int(oh.size),
                    int(oc_group_end - oc_group_start),
                    int(ic_group_end - ic_group_start),
                    int(max_chunk_items),
                )
                for oc_start in range(int(oc_group_start), int(oc_group_end), int(output_chunk)):
                    oc_end = min(int(oc_group_end), int(oc_start) + int(output_chunk))
                    output_channels = np.arange(int(oc_start), int(oc_end), dtype=np.int64)
                    row_indices = _packed_flat_indices_for_channels(
                        output_channels,
                        oh,
                        ow,
                        gap=int(output_gap),
                        height=int(on_ho),
                        width=int(on_wo),
                        row_offset=int(output_row_offset),
                    )[:, None, :]
                    for ic_start in range(int(ic_group_start), int(ic_group_end), int(input_chunk)):
                        ic_end = min(int(ic_group_end), int(ic_start) + int(input_chunk))
                        input_channels = np.arange(int(ic_start), int(ic_end), dtype=np.int64)
                        rel_start = int(ic_start) - int(ic_group_start)
                        rel_end = int(ic_end) - int(ic_group_start)
                        weight_block = weight_np[
                            int(oc_start):int(oc_end),
                            int(rel_start):int(rel_end),
                            int(kh),
                            int(kw),
                        ]
                        if not bool(np.any(weight_block != 0.0)):
                            continue
                        if not bool(np.all(weight_block != 0.0)):
                            for local_oc, oc in enumerate(output_channels.tolist()):
                                nonzero_rel = np.flatnonzero(weight_block[int(local_oc)] != 0.0)
                                if nonzero_rel.size == 0:
                                    continue
                                sparse_inputs = input_channels[nonzero_rel]
                                sparse_cols = _packed_flat_indices_for_channels(
                                    sparse_inputs,
                                    ih,
                                    iw,
                                    gap=int(input_gap),
                                    height=int(on_hi),
                                    width=int(on_wi),
                                    row_offset=int(input_row_offset),
                                )[None, :, :]
                                sparse_rows = row_indices[int(local_oc):int(local_oc) + 1]
                                for batch in range(int(n_batch)):
                                    _append_diagonal_key_chunk(
                                        key_chunks,
                                        sparse_rows + int(batch) * int(output_block_size),
                                        sparse_cols + int(batch) * int(input_block_size),
                                        matrix_height=int(matrix_shape[0]),
                                        matrix_width=int(matrix_shape[1]),
                                        num_slots=int(num_slots),
                                        embed_method=str(embed_method),
                                        is_last_layer=bool(is_last_layer),
                                        allow_hybrid=bool(allow_hybrid),
                                    )
                            continue
                        col_indices = _packed_flat_indices_for_channels(
                            input_channels,
                            ih,
                            iw,
                            gap=int(input_gap),
                            height=int(on_hi),
                            width=int(on_wi),
                            row_offset=int(input_row_offset),
                        )[None, :, :]
                        for batch in range(int(n_batch)):
                            _append_diagonal_key_chunk(
                                key_chunks,
                                row_indices + int(batch) * int(output_block_size),
                                col_indices + int(batch) * int(input_block_size),
                                matrix_height=int(matrix_shape[0]),
                                matrix_width=int(matrix_shape[1]),
                                num_slots=int(num_slots),
                                embed_method=str(embed_method),
                                is_last_layer=bool(is_last_layer),
                                allow_hybrid=bool(allow_hybrid),
                            )
    if bool(return_indices):
        return _diagonal_key_chunks_to_indices(
            key_chunks,
            matrix_height=int(matrix_shape[0]),
            matrix_width=int(matrix_shape[1]),
            num_slots=int(num_slots),
            embed_method=str(embed_method),
            is_last_layer=bool(is_last_layer),
            allow_hybrid=bool(allow_hybrid),
        )
    return _count_diagonal_key_chunks(key_chunks)


def estimate_direct_conv_transpose2d_diagonal_count(
    conv_layer,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
    *,
    allow_hybrid: bool = True,
    max_chunk_items: int | None = None,
    return_indices: bool = False,
) -> int:
    """Count packed block diagonals for ConvTranspose2d without payloads."""

    if max_chunk_items is None:
        max_chunk_items = int(os.environ.get("ORION_PACK_DIAGONAL_COUNT_MAX_ITEMS", "2000000"))
    max_chunk_items = max(1, int(max_chunk_items))

    weight_np = conv_layer.on_weight.detach().cpu().to(dtype=torch.float32).numpy()
    n_batch, ci, _, _ = [int(value) for value in conv_layer.input_shape]
    _, co, _, _ = [int(value) for value in conv_layer.output_shape]
    _, on_ci, on_hi, on_wi = [int(value) for value in conv_layer.fhe_input_shape]
    _, on_co, on_ho, on_wo = [int(value) for value in conv_layer.fhe_output_shape]
    input_gap = int(conv_layer.input_gap)
    output_gap = int(conv_layer.output_gap)
    input_row_offset = max(0, int(getattr(conv_layer, "layout_policy_input_row_offset", 0) or 0))
    output_row_offset = max(0, int(getattr(conv_layer, "layout_policy_output_row_offset", 0) or 0))
    input_block_size = int(on_ci * on_hi * on_wi)
    output_block_size = int(on_co * on_ho * on_wo)
    matrix_shape = (
        int(torch.Size(conv_layer.fhe_output_shape).numel()),
        int(torch.Size(conv_layer.fhe_input_shape).numel()),
    )
    groups = int(conv_layer.groups)
    in_channels_per_group = int(conv_layer.in_channels // groups)
    out_channels_per_group = int(conv_layer.out_channels // groups)
    spatial_cache = _tconv2d_spatial_cache(conv_layer)

    key_chunks: list[np.ndarray] = []
    for kh in range(int(weight_np.shape[2])):
        for kw in range(int(weight_np.shape[3])):
            ih, iw, oh, ow = spatial_cache[(int(kh), int(kw))]
            if ih.size == 0:
                continue
            for group in range(int(groups)):
                ic_group_start = int(group) * int(in_channels_per_group)
                ic_group_end = min(int(ci), int(ic_group_start) + int(in_channels_per_group))
                oc_group_start = int(group) * int(out_channels_per_group)
                oc_group_end = min(int(co), int(oc_group_start) + int(out_channels_per_group))
                input_chunk, output_chunk = _channel_chunk_shape(
                    int(ih.size),
                    int(ic_group_end - ic_group_start),
                    int(oc_group_end - oc_group_start),
                    int(max_chunk_items),
                )
                for ic_start in range(int(ic_group_start), int(ic_group_end), int(input_chunk)):
                    ic_end = min(int(ic_group_end), int(ic_start) + int(input_chunk))
                    input_channels = np.arange(int(ic_start), int(ic_end), dtype=np.int64)
                    col_indices = _packed_flat_indices_for_channels(
                        input_channels,
                        ih,
                        iw,
                        gap=int(input_gap),
                        height=int(on_hi),
                        width=int(on_wi),
                        row_offset=int(input_row_offset),
                    )[:, None, :]
                    for oc_start in range(int(oc_group_start), int(oc_group_end), int(output_chunk)):
                        oc_end = min(int(oc_group_end), int(oc_start) + int(output_chunk))
                        rel_start = int(oc_start) - int(oc_group_start)
                        rel_end = int(oc_end) - int(oc_group_start)
                        output_channels = np.arange(int(oc_start), int(oc_end), dtype=np.int64)
                        weight_block = weight_np[
                            int(ic_start):int(ic_end),
                            int(rel_start):int(rel_end),
                            int(kh),
                            int(kw),
                        ]
                        if not bool(np.any(weight_block != 0.0)):
                            continue
                        if not bool(np.all(weight_block != 0.0)):
                            for local_ic, ic in enumerate(input_channels.tolist()):
                                nonzero_rel = np.flatnonzero(weight_block[int(local_ic)] != 0.0)
                                if nonzero_rel.size == 0:
                                    continue
                                sparse_outputs = output_channels[nonzero_rel]
                                sparse_rows = _packed_flat_indices_for_channels(
                                    sparse_outputs,
                                    oh,
                                    ow,
                                    gap=int(output_gap),
                                    height=int(on_ho),
                                    width=int(on_wo),
                                    row_offset=int(output_row_offset),
                                )[None, :, :]
                                sparse_cols = col_indices[int(local_ic):int(local_ic) + 1]
                                for batch in range(int(n_batch)):
                                    _append_diagonal_key_chunk(
                                        key_chunks,
                                        sparse_rows + int(batch) * int(output_block_size),
                                        sparse_cols + int(batch) * int(input_block_size),
                                        matrix_height=int(matrix_shape[0]),
                                        matrix_width=int(matrix_shape[1]),
                                        num_slots=int(num_slots),
                                        embed_method=str(embed_method),
                                        is_last_layer=bool(is_last_layer),
                                        allow_hybrid=bool(allow_hybrid),
                                    )
                            continue
                        row_indices = _packed_flat_indices_for_channels(
                            output_channels,
                            oh,
                            ow,
                            gap=int(output_gap),
                            height=int(on_ho),
                            width=int(on_wo),
                            row_offset=int(output_row_offset),
                        )[None, :, :]
                        for batch in range(int(n_batch)):
                            _append_diagonal_key_chunk(
                                key_chunks,
                                row_indices + int(batch) * int(output_block_size),
                                col_indices + int(batch) * int(input_block_size),
                                matrix_height=int(matrix_shape[0]),
                                matrix_width=int(matrix_shape[1]),
                                num_slots=int(num_slots),
                                embed_method=str(embed_method),
                                is_last_layer=bool(is_last_layer),
                                allow_hybrid=bool(allow_hybrid),
                            )
    if bool(return_indices):
        return _diagonal_key_chunks_to_indices(
            key_chunks,
            matrix_height=int(matrix_shape[0]),
            matrix_width=int(matrix_shape[1]),
            num_slots=int(num_slots),
            embed_method=str(embed_method),
            is_last_layer=bool(is_last_layer),
            allow_hybrid=bool(allow_hybrid),
        )
    return _count_diagonal_key_chunks(key_chunks)


def pack_conv2d_diagonal_indices(conv_layer: nn.Module, last: bool, *, allow_hybrid: bool = True):
    slots = int(conv_layer.scheme.params.get_slots())
    embed_method = str(conv_layer.scheme.params.get_embedding_method())
    indices = estimate_direct_conv2d_diagonal_count(
        conv_layer,
        slots,
        embed_method,
        bool(last),
        allow_hybrid=bool(allow_hybrid),
        return_indices=True,
    )
    output_rotations = _packed_output_rotations(
        matrix_height=int(torch.Size(conv_layer.fhe_output_shape).numel()),
        num_slots=int(slots),
        embed_method=str(embed_method),
        is_last_layer=bool(last),
        allow_hybrid=bool(allow_hybrid),
    )
    return indices, int(output_rotations)


def pack_conv_transpose2d_diagonal_indices(conv_layer: nn.Module, last: bool, *, allow_hybrid: bool = True):
    slots = int(conv_layer.scheme.params.get_slots())
    embed_method = str(conv_layer.scheme.params.get_embedding_method())
    indices = estimate_direct_conv_transpose2d_diagonal_count(
        conv_layer,
        slots,
        embed_method,
        bool(last),
        allow_hybrid=bool(allow_hybrid),
        return_indices=True,
    )
    output_rotations = _packed_output_rotations(
        matrix_height=int(torch.Size(conv_layer.fhe_output_shape).numel()),
        num_slots=int(slots),
        embed_method=str(embed_method),
        is_last_layer=bool(last),
        allow_hybrid=bool(allow_hybrid),
    )
    return indices, int(output_rotations)


def estimate_packed_diagonal_count(linear_layer: nn.Module, last: bool, *, max_chunk_items: int | None = None) -> int:
    slots = linear_layer.scheme.params.get_slots()
    embed_method = linear_layer.scheme.params.get_embedding_method()
    if hasattr(linear_layer, "kernel_size") and hasattr(linear_layer, "stride"):
        if type(linear_layer).__name__ == "ConvTranspose2d":
            return estimate_direct_conv_transpose2d_diagonal_count(
                linear_layer,
                int(slots),
                str(embed_method),
                bool(last),
                max_chunk_items=max_chunk_items,
            )
        return estimate_direct_conv2d_diagonal_count(
            linear_layer,
            int(slots),
            str(embed_method),
            bool(last),
            max_chunk_items=max_chunk_items,
        )
    return 0


def pack_linear(linear_layer: nn.Module, last: bool):
    slots = linear_layer.scheme.params.get_slots()
    embed_method = linear_layer.scheme.params.get_embedding_method()

    weight = construct_linear_matrix(linear_layer)
    diagonals, output_rotations = diagonalize(weight, slots, embed_method, last)
    return diagonals, output_rotations

def construct_linear_matrix(linear_layer):
    if len(linear_layer.input_shape) == 2:
        N = linear_layer.input_shape[0]
        matrix = linear_layer.on_weight 
    else: # Prior layer was not a linear layer
        out_features = linear_layer.out_features
        input_gap = linear_layer.input_gap 
        N, Ci, Hi, Wi = linear_layer.input_shape 
        on_Ci, on_Hi, on_Wi = linear_layer.fhe_input_shape[1:]
        
        reshaped = linear_layer.on_weight.reshape(out_features, Ci, Hi, Wi)
        reshaped = multiplex(reshaped, input_gap)

        matrix = torch.zeros(out_features, on_Ci, on_Hi, on_Wi)
        matrix[..., :Hi*input_gap, :Wi*input_gap] = reshaped 
        matrix = matrix.reshape(out_features, -1)
   
    matrix = torch.kron(torch.eye(N), matrix) 
    matrix_sparse = sp.csr_matrix(matrix.cpu().numpy())
    return matrix_sparse

def construct_linear_bias(linear_layer):
    N = linear_layer.input_shape[0]
    return linear_layer.on_bias.repeat(N)

#-----------------------------#
#       Helper Functions      #
#-----------------------------#

def multiplex(matrix, gap):
    N, Ci, Hi, Wi = matrix.shape
    Co = math.ceil(Ci / (gap**2))
    
    # Pad the tensor to have channels divisible by gap^2
    padded = torch.zeros(N, Co * gap**2, Hi, Wi)
    padded[:, :Ci, ...] = matrix
    return F.pixel_shuffle(padded, gap) # multiplexed

def resolve_grouped_conv(conv_layer):
    on_weight = conv_layer.on_weight.repeat(1, conv_layer.groups, 1, 1)

    # Zero out input channels to support arbitrary groups
    mask = torch.zeros_like(on_weight)        
    Ci_per_group = conv_layer.in_channels // conv_layer.groups
    Co_per_group = conv_layer.out_channels // conv_layer.groups
    
    for i in range(conv_layer.groups):
        mask[i*Co_per_group:(i+1)*Co_per_group, 
             i*Ci_per_group:(i+1)*Ci_per_group, ...] = 1

    return on_weight * mask

def diagonalize(
    matrix: sp.csr_matrix,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
    *,
    allow_hybrid: bool = True,
):
    """
    For each (slots, slots) block of the input matrix, this function 
    extracts the generalized diagonals and stores them in a dictionary. 
    Each key ((i,j)) in the dictionary block_{i,j}, and the value is 
    another dictionary mapping diagonal indices to their values.

    Args:
        matrix (scipy.sparse.csr_matrix): A 4D tensor representing a weight matrix 
            for a fully-connected or convolutional layer. The shape must 
            conform to (num_blocks_y, num_blocks_x, slots, slots).
        slots (int): The number of SIMD plaintext slots, dictating the 
            block size.

    Returns:
        dict: A dictionary where each key is a tuple (i, j) corresponding 
              to the (i, j)th (slots, slots) block of `matrix`. The value 
              for each key is another dictionary that maps diagonal indices 
              within the block to the diagonal's tensor values.

    Examples:
        >>> matrix = torch.tensor([[[[ 0,  1,  2,  3],
                                     [ 4,  5,  6,  7],
                                     [ 8,  9, 10, 11],
                                     [12, 13, 14, 15]]]])
        >>> # Example with slots=4, showing processing of a single block
        >>> print(diagonalize(matrix, slots=4)) 
        {(0, 0): {0: [0., 5., 10., 15.], 
                  1: [1., 6., 11., 12.], 
                  2: [2., 7., 8., 13.], 
                  3: [3., 4., 9., 14.]}}

        >>> # Example with slots=2, showing processing of four blocks or 
              sub-matrices
        >>> print(diagonalize(matrix, slots=2)) 
        {(0, 0): {0: [0., 5.], 
                  1: [1., 4.]}, 
         (0, 1): {0: [2., 7.], 
                  1: [3., 6.]}, 
         (1, 0): {0: [8., 13.], 
                  1: [9., 12.]}, 
         (1, 1): {0: [10., 15.], 
                  1: [11., 14.]}}
    """

    matrix_height, matrix_width = matrix.shape
    num_block_rows = math.ceil(matrix_height / num_slots)
    num_block_cols = math.ceil(matrix_width / num_slots)
    print(f"├── embed method: {embed_method}")
    print(f"├── original matrix shape: {matrix.shape}")
    print(f"├── # blocks (rows, cols) = {(num_block_rows, num_block_cols)}")

    if allow_hybrid and num_block_rows == 1 and embed_method == "hybrid" and not is_last_layer:
        block_height = 2 ** math.ceil(math.log2(matrix_height))
        output_rotations = int(math.log2(num_slots // block_height))
    else:
        block_height = num_slots
        output_rotations = 0

    # Inflate dimensions of the sparse matrix
    matrix.resize(num_block_rows * block_height, num_block_cols * num_slots)

    print(f"├── resized matrix shape: {matrix.shape}")
    print(f"├── # output rotations: {output_rotations}")

    # Prepare indices for diagonal extraction 
    row_idx = torch.arange(block_height).repeat(num_slots // block_height)
    col_idx = torch.arange(block_height)[:, None] + torch.arange(num_slots)[None, :]
    col_idx = torch.where(col_idx >= num_slots, col_idx - num_slots, col_idx)

    diagonals_by_block = {}
    total_diagonals = 0

    # Process each block 
    progress_bar = tqdm(
        total=num_block_rows * num_block_cols,
        desc="|    Processing blocks",
        leave=False,
    )
    start_time = time.time()
    for block_row in range(num_block_rows):
        for block_col in range(num_block_cols):
            row_start = num_slots * block_row
            col_start = num_slots * block_col
            block_sparse = matrix[
                row_start: row_start + block_height,
                col_start: col_start + num_slots,
            ]
            if block_height == num_slots:
                nonzero_diagonals = _extract_square_block_diagonals(block_sparse, num_slots)
            else:
                block_dense = torch.tensor(block_sparse.todense(), dtype=torch.float32)
                block_diagonals = block_dense[row_idx, col_idx]

                # Collect non-zero diagonals
                nonzero_diagonals = {}
                for i in range(block_height):
                    if torch.any(block_diagonals[i]):
                        nonzero_diagonals[i] = block_diagonals[i].tolist()

            total_diagonals += len(nonzero_diagonals)
            diagonals_by_block[(block_row, block_col)] = (
                nonzero_diagonals or {0: [0.0] * num_slots}
            )

            progress_bar.set_postfix({
                "Current Block": f"({block_row},{block_col})",
                "Total Diagonals": total_diagonals,
            })
            progress_bar.update(1)

    progress_bar.close()
    elapsed_time = time.time() - start_time
    print(f"├── time to pack (s): {elapsed_time:.2f}")
    print(f"├── # diagonals = {total_diagonals}")

    return diagonals_by_block, output_rotations


def _extract_square_block_diagonals(block_sparse: sp.spmatrix, num_slots: int) -> dict[int, list[float]]:
    coo = block_sparse.tocoo()
    diagonals: dict[int, torch.Tensor] = {}

    for row_idx, col_idx, value in zip(coo.row, coo.col, coo.data):
        diag_idx = int((int(col_idx) - int(row_idx)) % int(num_slots))
        diagonal = diagonals.get(int(diag_idx))
        if diagonal is None:
            diagonal = torch.zeros((int(num_slots),), dtype=torch.float32)
            diagonals[int(diag_idx)] = diagonal
        diagonal[int(row_idx)] = float(value)

    return {
        int(diag_idx): diagonals[int(diag_idx)].tolist()
        for diag_idx in sorted(int(value) for value in diagonals.keys())
    }

def plot_toeplitz(matrix, save_path=""):
    if isinstance(matrix, sp.csr_matrix):
        matrix = matrix.todense()

    if matrix.ndim != 2:
        raise ValueError(f"Cannot plot matrix of dimension {matrix.ndim}")

    plt.imshow(matrix)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


#---------------------#
#   BatchNorm Logic   #
#---------------------#

def pack_bn1d(bn1d_layer):
    N = bn1d_layer.input_shape[0]
    on_running_mean = bn1d_layer.on_running_mean
    on_inv_running_std = 1 / torch.sqrt(bn1d_layer.running_var + bn1d_layer.eps)
    on_weight = bn1d_layer.on_weight if bn1d_layer.affine else None 
    on_bias = bn1d_layer.on_bias if bn1d_layer.affine else None 

    return (
        on_running_mean.flatten().repeat(N),
        on_inv_running_std.flatten().repeat(N),
        on_weight.flatten().repeat(N) if on_weight is not None else None, 
        on_bias.flatten().repeat(N) if on_bias is not None else None
    )

def pack_bn2d(bn2d_layer):
    N, Ci, Hi, Wi = bn2d_layer.input_shape
    on_Ci, on_Hi, on_Wi = bn2d_layer.fhe_input_shape[1:]

    on_running_mean = torch.zeros(on_Ci, on_Hi, on_Wi)
    on_inv_running_std = torch.zeros(on_Ci, on_Hi, on_Wi)

    mean = bn2d_layer.on_running_mean.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)
    var = bn2d_layer.on_running_var.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)

    mean_mpx = multiplex(mean, bn2d_layer.input_gap).squeeze(0)
    var_mpx = multiplex(var, bn2d_layer.input_gap).squeeze(0)

    mC, mH, mW = mean_mpx.shape
    on_running_mean[:mC, :mH, :mW] = mean_mpx 
    on_inv_running_std[:mC, :mH, :mW] = 1 / torch.sqrt(var_mpx + bn2d_layer.eps)

    on_weight = None 
    on_bias = None
    if bn2d_layer.affine:
        on_weight = torch.zeros(on_Ci, on_Hi, on_Wi)
        on_bias = torch.zeros(on_Ci, on_Hi, on_Wi)

        weight = bn2d_layer.on_weight.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)
        bias = bn2d_layer.on_bias.view(1, Ci, 1, 1).expand(1, Ci, Hi, Wi)

        weight_mpx = multiplex(weight, bn2d_layer.input_gap).squeeze(0)
        bias_mpx = multiplex(bias, bn2d_layer.input_gap).squeeze(0)

        mC, mH, mW = weight_mpx.shape
        on_weight[:mC, :mH, :mW] = weight_mpx 
        on_bias[:mC, :mH, :mW] = bias_mpx 

    return (
        on_running_mean.flatten().repeat(N),
        on_inv_running_std.flatten().repeat(N), 
        on_weight.flatten().repeat(N) if on_weight is not None else None, 
        on_bias.flatten().repeat(N) if on_bias is not None else None
    )
