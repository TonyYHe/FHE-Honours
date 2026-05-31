from __future__ import annotations

import ctypes
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}
_LIB_CACHE: ctypes.CDLL | None = None
_LOAD_ERROR: str | None = None


class _CPayload(ctypes.Structure):
    _fields_ = [
        ("row", ctypes.c_int),
        ("col", ctypes.c_int),
        ("level", ctypes.c_int),
        ("task_id", ctypes.c_char_p),
        ("diag_indices", ctypes.POINTER(ctypes.c_int)),
        ("diag_indices_len", ctypes.c_ulong),
        ("diag_data", ctypes.POINTER(ctypes.c_float)),
        ("diag_data_len", ctypes.c_ulong),
    ]


class _CPayloadBatch(ctypes.Structure):
    _fields_ = [
        ("payloads", ctypes.POINTER(_CPayload)),
        ("len", ctypes.c_ulong),
        ("output_rotations", ctypes.c_int),
        ("builder_kind", ctypes.c_char_p),
        ("fallback_reason", ctypes.c_char_p),
    ]


class _CProviderNativeSourceSpec(ctypes.Structure):
    _fields_ = [
        ("slots", ctypes.c_int),
        ("c_in", ctypes.c_int),
        ("h_in", ctypes.c_int),
        ("w_in", ctypes.c_int),
        ("c_out", ctypes.c_int),
        ("h_out", ctypes.c_int),
        ("w_out", ctypes.c_int),
        ("gap_in", ctypes.c_int),
        ("gap_out", ctypes.c_int),
        ("kernel", ctypes.c_int),
        ("stride", ctypes.c_int),
        ("pad", ctypes.c_int),
        ("dilation", ctypes.c_int),
        ("input_h_min", ctypes.c_int),
        ("input_h_max", ctypes.c_int),
        ("output_top_beta", ctypes.c_int),
        ("output_bottom_beta", ctypes.c_int),
        ("output_physical_top_beta", ctypes.c_int),
        ("output_physical_bottom_beta", ctypes.c_int),
        ("stripe_index", ctypes.c_int),
        ("stripe_source_h_start", ctypes.c_int),
        ("stripe_source_h", ctypes.c_int),
        ("stripe_target_h_start", ctypes.c_int),
        ("stripe_target_h_end", ctypes.c_int),
        ("stripe_target_h", ctypes.c_int),
        ("source_tile", ctypes.c_int),
        ("target_tile", ctypes.c_int),
        ("source_group", ctypes.c_int),
        ("target_group", ctypes.c_int),
        ("source_index", ctypes.c_int),
        ("target_index", ctypes.c_int),
        ("compact_output", ctypes.c_int),
        ("compact_target_block", ctypes.c_int),
    ]


class _CProviderCompactSourceSpec(ctypes.Structure):
    _fields_ = [
        ("slots", ctypes.c_int),
        ("c_in", ctypes.c_int),
        ("h_in", ctypes.c_int),
        ("w_in", ctypes.c_int),
        ("c_out", ctypes.c_int),
        ("h_out", ctypes.c_int),
        ("w_out", ctypes.c_int),
        ("gap_out", ctypes.c_int),
        ("kernel", ctypes.c_int),
        ("stride", ctypes.c_int),
        ("pad", ctypes.c_int),
        ("dilation", ctypes.c_int),
        ("source_top_beta", ctypes.c_int),
        ("source_bottom_beta", ctypes.c_int),
        ("source_gap", ctypes.c_int),
        ("output_top_beta", ctypes.c_int),
        ("output_bottom_beta", ctypes.c_int),
        ("output_physical_top_beta", ctypes.c_int),
        ("output_physical_bottom_beta", ctypes.c_int),
        ("stripe_index", ctypes.c_int),
        ("stripe_target_h_start", ctypes.c_int),
        ("stripe_target_h_end", ctypes.c_int),
        ("stripe_target_h", ctypes.c_int),
        ("target_tile", ctypes.c_int),
        ("source_block", ctypes.c_int),
        ("target_group", ctypes.c_int),
        ("target_index", ctypes.c_int),
        ("compact_output", ctypes.c_int),
        ("compact_target_block", ctypes.c_int),
    ]


class _CProviderStripeSpec(ctypes.Structure):
    _fields_ = [
        ("stripe_index", ctypes.c_int),
        ("target_h_start", ctypes.c_int),
        ("target_h_end", ctypes.c_int),
        ("target_h", ctypes.c_int),
        ("target_tile", ctypes.c_int),
        ("target_group_count", ctypes.c_int),
    ]


class _CProviderCompactSourceConcatIndexSpec(ctypes.Structure):
    _fields_ = [
        ("slots", ctypes.c_int),
        ("c_in", ctypes.c_int),
        ("h_in", ctypes.c_int),
        ("w_in", ctypes.c_int),
        ("c_out", ctypes.c_int),
        ("h_out", ctypes.c_int),
        ("w_out", ctypes.c_int),
        ("gap_out", ctypes.c_int),
        ("kernel", ctypes.c_int),
        ("stride", ctypes.c_int),
        ("pad", ctypes.c_int),
        ("dilation", ctypes.c_int),
        ("source_top_beta", ctypes.c_int),
        ("source_bottom_beta", ctypes.c_int),
        ("source_gap", ctypes.c_int),
        ("output_top_beta", ctypes.c_int),
        ("output_bottom_beta", ctypes.c_int),
        ("output_physical_top_beta", ctypes.c_int),
        ("output_physical_bottom_beta", ctypes.c_int),
        ("source_ct_count", ctypes.c_int),
        ("target_ct_count", ctypes.c_int),
        ("fuse_output_relayout", ctypes.c_int),
    ]


@dataclass(frozen=True)
class DenseConv2DPayload:
    row: int
    col: int
    diag_indices: np.ndarray
    diag_data: np.ndarray
    level: int = 0
    task_id: str = ""


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(str(name))
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in _FALSE_ENV_VALUES


def dense_builder_enabled() -> bool:
    return bool(_env_enabled("ORION_CPP_DIAG_BUILDER") and _env_enabled("ORION_CPP_DIAG_BUILDER_DENSE", True))


def shadow_enabled() -> bool:
    return bool(_env_enabled("ORION_CPP_DIAG_BUILDER_SHADOW"))


def strict_enabled() -> bool:
    return bool(_env_enabled("ORION_CPP_DIAG_BUILDER_STRICT"))


def _library_filename() -> str:
    system = platform.system()
    if system == "Darwin":
        return "diag_builder-mac.dylib"
    if system == "Windows":
        return "diag_builder-windows.dll"
    return "diag_builder-linux.so"


def _candidate_paths() -> list[Path]:
    override = os.environ.get("ORION_DIAG_BUILDER_LIB")
    paths: list[Path] = []
    if override:
        paths.append(Path(override).expanduser())
    here = Path(__file__).resolve().parent
    paths.append(here / _library_filename())
    return paths


def load_library() -> ctypes.CDLL | None:
    global _LIB_CACHE, _LOAD_ERROR
    if _LIB_CACHE is not None:
        return _LIB_CACHE
    for path in _candidate_paths():
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            lib.OrionDiagBuilderVersion.argtypes = []
            lib.OrionDiagBuilderVersion.restype = ctypes.c_char_p
            lib.OrionDiagBuilderLastError.argtypes = []
            lib.OrionDiagBuilderLastError.restype = ctypes.c_char_p
            lib.OrionBuildDenseConv2D.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
            ]
            lib.OrionBuildDenseConv2D.restype = _CPayloadBatch
            dense_tconv = getattr(lib, "OrionBuildDenseConvTranspose2D", None)
            if dense_tconv is not None:
                dense_tconv.argtypes = list(lib.OrionBuildDenseConv2D.argtypes)
                dense_tconv.restype = _CPayloadBatch
            dense_index = getattr(lib, "OrionBuildDenseConv2DIndexOnly", None)
            if dense_index is not None:
                dense_index.argtypes = [
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                ]
                dense_index.restype = _CPayloadBatch
            dense_tconv_index = getattr(lib, "OrionBuildDenseConvTranspose2DIndexOnly", None)
            if dense_tconv_index is not None and dense_index is not None:
                dense_tconv_index.argtypes = list(dense_index.argtypes)
                dense_tconv_index.restype = _CPayloadBatch
            lib.OrionFreeDiagPayloadBatch.argtypes = [_CPayloadBatch]
            lib.OrionFreeDiagPayloadBatch.restype = None
            provider_native = getattr(lib, "OrionBuildProviderNativeSourceConv2D", None)
            if provider_native is not None:
                provider_native.argtypes = [
                    _CProviderNativeSourceSpec,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                ]
                provider_native.restype = _CPayloadBatch
            provider_compact = getattr(lib, "OrionBuildProviderCompactSourceConv2D", None)
            if provider_compact is not None:
                provider_compact.argtypes = [
                    _CProviderCompactSourceSpec,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                ]
                provider_compact.restype = _CPayloadBatch
            provider_concat_index = getattr(lib, "OrionBuildProviderCompactSourceConcatConv2DIndexOnly", None)
            if provider_concat_index is not None:
                provider_concat_index.argtypes = [
                    _CProviderCompactSourceConcatIndexSpec,
                    ctypes.POINTER(_CProviderStripeSpec),
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                ]
                provider_concat_index.restype = _CPayloadBatch
            _LIB_CACHE = lib
            _LOAD_ERROR = None
            return _LIB_CACHE
        except OSError as exc:
            _LOAD_ERROR = str(exc)
    if _LOAD_ERROR is None:
        _LOAD_ERROR = "diag_builder shared library not found"
    return None


def load_error() -> str | None:
    return _LOAD_ERROR


def _shape4(values: Any) -> tuple[int, int, int, int]:
    shape = tuple(int(value) for value in tuple(values))
    if len(shape) != 4:
        raise ValueError(f"expected 4D shape, got {shape!r}")
    return shape


def _int_array(values: Iterable[int]):
    seq = tuple(int(value) for value in values)
    return (ctypes.c_int * len(seq))(*seq)


def _block_arrays(blocks: Iterable[tuple[int, int]] | None):
    requested = tuple((int(row), int(col)) for row, col in (blocks or ()))
    if not requested:
        return None, None, 0
    rows = (ctypes.c_int * len(requested))(*[row for row, _col in requested])
    cols = (ctypes.c_int * len(requested))(*[col for _row, col in requested])
    return rows, cols, int(len(requested))


def _layout_top_beta(layout: dict[str, Any]) -> int:
    return max(0, int(layout.get("top_beta", layout.get("alpha", 0)) or 0))


def _decode_c_string(value: bytes | None) -> str:
    return "" if value is None else value.decode("utf-8", errors="replace")


def _copy_payloads(batch: _CPayloadBatch) -> list[DenseConv2DPayload]:
    payloads: list[DenseConv2DPayload] = []
    for index in range(int(batch.len)):
        item = batch.payloads[index]
        idx_len = int(item.diag_indices_len)
        data_len = int(item.diag_data_len)
        diag_indices = (
            np.ctypeslib.as_array(item.diag_indices, shape=(idx_len,)).astype(np.int32, copy=True)
            if idx_len > 0
            else np.zeros((0,), dtype=np.int32)
        )
        diag_data = (
            np.ctypeslib.as_array(item.diag_data, shape=(data_len,)).astype(np.float32, copy=True)
            if data_len > 0
            else np.zeros((0,), dtype=np.float32)
        )
        payloads.append(
            DenseConv2DPayload(
                row=int(item.row),
                col=int(item.col),
                level=int(item.level),
                task_id=_decode_c_string(item.task_id),
                diag_indices=np.ascontiguousarray(diag_indices, dtype=np.int32),
                diag_data=np.ascontiguousarray(diag_data, dtype=np.float32),
            )
        )
    return payloads


def _dense_conv2d_weight(conv_layer: Any) -> torch.Tensor:
    weight = getattr(conv_layer, "on_weight").detach().cpu().to(dtype=torch.float32)
    if int(getattr(conv_layer, "groups", 1) or 1) > 1:
        from orion.core import packing

        weight = packing.resolve_grouped_conv(conv_layer).detach().cpu().to(dtype=torch.float32)
    return weight


def _dense_conv_transpose2d_weight(conv_layer: Any) -> torch.Tensor:
    weight = getattr(conv_layer, "on_weight").detach().cpu().to(dtype=torch.float32)
    groups = int(getattr(conv_layer, "groups", 1) or 1)
    if groups <= 1:
        return weight
    ci = int(getattr(conv_layer, "in_channels"))
    co = int(getattr(conv_layer, "out_channels"))
    ci_per_group = int(ci // groups)
    co_per_group = int(co // groups)
    expanded = torch.zeros((ci, co, int(weight.shape[2]), int(weight.shape[3])), dtype=torch.float32)
    for group in range(groups):
        ic_start = int(group) * int(ci_per_group)
        ic_end = min(int(ci), int(ic_start) + int(ci_per_group))
        oc_start = int(group) * int(co_per_group)
        oc_end = min(int(co), int(oc_start) + int(co_per_group))
        expanded[ic_start:ic_end, oc_start:oc_end, :, :] = weight[ic_start:ic_end, : int(oc_end - oc_start), :, :]
    return expanded


def _cpp_dense_conv2d_payloads(
    conv_layer: Any,
    *,
    last: bool,
    allow_hybrid: bool,
    blocks: Iterable[tuple[int, int]] | None,
) -> tuple[list[DenseConv2DPayload], int, dict[str, Any]]:
    lib = load_library()
    if lib is None:
        raise RuntimeError(load_error() or "diag_builder shared library unavailable")
    build = getattr(lib, "OrionBuildDenseConv2D")
    return _cpp_dense_layer_payloads(
        conv_layer,
        lib=lib,
        build=build,
        last=bool(last),
        allow_hybrid=bool(allow_hybrid),
        blocks=blocks,
        fallback_kind="cpp_dense_conv2d",
        transposed=False,
    )


def _cpp_dense_conv_transpose2d_payloads(
    conv_layer: Any,
    *,
    last: bool,
    allow_hybrid: bool,
    blocks: Iterable[tuple[int, int]] | None,
) -> tuple[list[DenseConv2DPayload], int, dict[str, Any]]:
    lib = load_library()
    if lib is None:
        raise RuntimeError(load_error() or "diag_builder shared library unavailable")
    build = getattr(lib, "OrionBuildDenseConvTranspose2D", None)
    if not callable(build):
        raise RuntimeError("diag_builder shared library does not export dense ConvTranspose2d builder")
    return _cpp_dense_layer_payloads(
        conv_layer,
        lib=lib,
        build=build,
        last=bool(last),
        allow_hybrid=bool(allow_hybrid),
        blocks=blocks,
        fallback_kind="cpp_dense_conv_transpose2d",
        transposed=True,
    )


def _cpp_dense_layer_payloads(
    conv_layer: Any,
    *,
    lib: ctypes.CDLL,
    build: Any,
    last: bool,
    allow_hybrid: bool,
    blocks: Iterable[tuple[int, int]] | None,
    fallback_kind: str,
    transposed: bool,
) -> tuple[list[DenseConv2DPayload], int, dict[str, Any]]:
    params = conv_layer.scheme.params
    weight = _dense_conv_transpose2d_weight(conv_layer) if bool(transposed) else _dense_conv2d_weight(conv_layer)
    weight_np = np.ascontiguousarray(weight.numpy().reshape(-1), dtype=np.float32)
    input_shape = _int_array(_shape4(conv_layer.input_shape))
    output_shape = _int_array(_shape4(conv_layer.output_shape))
    fhe_input_shape = _int_array(_shape4(conv_layer.fhe_input_shape))
    fhe_output_shape = _int_array(_shape4(conv_layer.fhe_output_shape))
    block_rows, block_cols, block_count = _block_arrays(blocks)
    output_layout = dict(getattr(conv_layer, "layout_policy_output_layout", {}) or {})
    materialization = str(getattr(conv_layer, "layout_policy_output_materialization", "") or "")
    kernel_h, kernel_w = (int(value) for value in tuple(conv_layer.kernel_size))
    stride_h, stride_w = (int(value) for value in tuple(conv_layer.stride))
    pad_h, pad_w = (int(value) for value in tuple(conv_layer.padding))
    dil_h, dil_w = (int(value) for value in tuple(conv_layer.dilation))

    started = time.perf_counter()
    batch = build(
        int(params.get_slots()),
        str(params.get_embedding_method()).encode("utf-8"),
        int(bool(last)),
        int(bool(allow_hybrid)),
        input_shape,
        output_shape,
        fhe_input_shape,
        fhe_output_shape,
        int(getattr(conv_layer, "input_gap")),
        int(getattr(conv_layer, "output_gap")),
        max(0, int(getattr(conv_layer, "layout_policy_input_row_offset", 0) or 0)),
        max(0, int(getattr(conv_layer, "layout_policy_output_row_offset", 0) or 0)),
        int(kernel_h),
        int(kernel_w),
        int(stride_h),
        int(stride_w),
        int(pad_h),
        int(pad_w),
        int(dil_h),
        int(dil_w),
        _layout_top_beta(output_layout),
        max(0, int(output_layout.get("bottom_beta", output_layout.get("beta", 0)) or 0)),
        int(materialization == "fused_relayout"),
        weight_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(weight_np.size),
        block_rows,
        block_cols,
        int(block_count),
    )
    build_s = float(time.perf_counter() - started)
    try:
        fallback_reason = _decode_c_string(batch.fallback_reason)
        if fallback_reason:
            raise RuntimeError(fallback_reason)
        payloads = _copy_payloads(batch)
        metadata = {
            "diag_builder_kind": _decode_c_string(batch.builder_kind) or str(fallback_kind),
            "diag_builder_source": "cpp",
            "diag_builder_build_s": float(build_s),
            "diag_builder_payload_count": int(len(payloads)),
            "diag_builder_fallback_reason": "",
        }
        return payloads, int(batch.output_rotations), metadata
    finally:
        lib.OrionFreeDiagPayloadBatch(batch)


def _dense_conv2d_c_args(conv_layer: Any, *, last: bool, allow_hybrid: bool, transposed: bool = False):
    params = conv_layer.scheme.params
    weight = _dense_conv_transpose2d_weight(conv_layer) if bool(transposed) else _dense_conv2d_weight(conv_layer)
    weight_np = np.ascontiguousarray(weight.numpy().reshape(-1), dtype=np.float32)
    input_shape = _int_array(_shape4(conv_layer.input_shape))
    output_shape = _int_array(_shape4(conv_layer.output_shape))
    fhe_input_shape = _int_array(_shape4(conv_layer.fhe_input_shape))
    fhe_output_shape = _int_array(_shape4(conv_layer.fhe_output_shape))
    output_layout = dict(getattr(conv_layer, "layout_policy_output_layout", {}) or {})
    materialization = str(getattr(conv_layer, "layout_policy_output_materialization", "") or "")
    kernel_h, kernel_w = (int(value) for value in tuple(conv_layer.kernel_size))
    stride_h, stride_w = (int(value) for value in tuple(conv_layer.stride))
    pad_h, pad_w = (int(value) for value in tuple(conv_layer.padding))
    dil_h, dil_w = (int(value) for value in tuple(conv_layer.dilation))
    return (
        int(params.get_slots()),
        str(params.get_embedding_method()).encode("utf-8"),
        int(bool(last)),
        int(bool(allow_hybrid)),
        input_shape,
        output_shape,
        fhe_input_shape,
        fhe_output_shape,
        int(getattr(conv_layer, "input_gap")),
        int(getattr(conv_layer, "output_gap")),
        max(0, int(getattr(conv_layer, "layout_policy_input_row_offset", 0) or 0)),
        max(0, int(getattr(conv_layer, "layout_policy_output_row_offset", 0) or 0)),
        int(kernel_h),
        int(kernel_w),
        int(stride_h),
        int(stride_w),
        int(pad_h),
        int(pad_w),
        int(dil_h),
        int(dil_w),
        _layout_top_beta(output_layout),
        max(0, int(output_layout.get("bottom_beta", output_layout.get("beta", 0)) or 0)),
        int(materialization == "fused_relayout"),
        weight_np,
    )


def _cpp_dense_conv2d_index_only(
    conv_layer: Any,
    *,
    last: bool,
    allow_hybrid: bool,
) -> tuple[dict[tuple[int, int], tuple[int, ...]], int, dict[str, Any]]:
    lib = load_library()
    if lib is None:
        raise RuntimeError(load_error() or "diag_builder shared library unavailable")
    build = getattr(lib, "OrionBuildDenseConv2DIndexOnly", None)
    if not callable(build):
        raise RuntimeError("diag_builder shared library does not export dense Conv2d index-only builder")
    return _cpp_dense_layer_index_only(
        conv_layer,
        lib=lib,
        build=build,
        last=bool(last),
        allow_hybrid=bool(allow_hybrid),
        fallback_kind="cpp_dense_conv2d:index_only",
        transposed=False,
    )


def _cpp_dense_conv_transpose2d_index_only(
    conv_layer: Any,
    *,
    last: bool,
    allow_hybrid: bool,
) -> tuple[dict[tuple[int, int], tuple[int, ...]], int, dict[str, Any]]:
    lib = load_library()
    if lib is None:
        raise RuntimeError(load_error() or "diag_builder shared library unavailable")
    build = getattr(lib, "OrionBuildDenseConvTranspose2DIndexOnly", None)
    if not callable(build):
        raise RuntimeError("diag_builder shared library does not export dense ConvTranspose2d index-only builder")
    return _cpp_dense_layer_index_only(
        conv_layer,
        lib=lib,
        build=build,
        last=bool(last),
        allow_hybrid=bool(allow_hybrid),
        fallback_kind="cpp_dense_conv_transpose2d:index_only",
        transposed=True,
    )


def _cpp_dense_layer_index_only(
    conv_layer: Any,
    *,
    lib: ctypes.CDLL,
    build: Any,
    last: bool,
    allow_hybrid: bool,
    fallback_kind: str,
    transposed: bool,
) -> tuple[dict[tuple[int, int], tuple[int, ...]], int, dict[str, Any]]:
    args = _dense_conv2d_c_args(conv_layer, last=bool(last), allow_hybrid=bool(allow_hybrid), transposed=bool(transposed))
    (
        slots,
        embed_method,
        is_last,
        allow,
        input_shape,
        output_shape,
        fhe_input_shape,
        fhe_output_shape,
        input_gap,
        output_gap,
        input_row_offset,
        output_row_offset,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        output_top_beta,
        output_bottom_beta,
        fuse_output_relayout,
        weight_np,
    ) = args
    started = time.perf_counter()
    batch = build(
        int(slots),
        embed_method,
        int(is_last),
        int(allow),
        input_shape,
        output_shape,
        fhe_input_shape,
        fhe_output_shape,
        int(input_gap),
        int(output_gap),
        int(input_row_offset),
        int(output_row_offset),
        int(kernel_h),
        int(kernel_w),
        int(stride_h),
        int(stride_w),
        int(pad_h),
        int(pad_w),
        int(dil_h),
        int(dil_w),
        int(output_top_beta),
        int(output_bottom_beta),
        int(fuse_output_relayout),
        weight_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(weight_np.size),
    )
    build_s = float(time.perf_counter() - started)
    try:
        fallback_reason = _decode_c_string(batch.fallback_reason)
        if fallback_reason:
            raise RuntimeError(fallback_reason)
        payloads = _copy_payloads(batch)
        indices = {
            (int(payload.row), int(payload.col)): tuple(int(value) for value in np.asarray(payload.diag_indices).reshape(-1))
            for payload in payloads
        }
        metadata = {
            "diag_builder_kind": _decode_c_string(batch.builder_kind) or str(fallback_kind),
            "diag_builder_source": "cpp",
            "diag_builder_build_s": float(build_s),
            "diag_builder_payload_count": int(len(payloads)),
            "diag_builder_fallback_reason": "",
        }
        return indices, int(batch.output_rotations), metadata
    finally:
        lib.OrionFreeDiagPayloadBatch(batch)


def build_dense_conv2d_payloads(
    conv_layer: Any,
    *,
    last: bool,
    allow_hybrid: bool = True,
    blocks: Iterable[tuple[int, int]] | None = None,
) -> tuple[list[DenseConv2DPayload], int, dict[str, Any]]:
    return _cpp_dense_conv2d_payloads(
        conv_layer,
        last=bool(last),
        allow_hybrid=bool(allow_hybrid),
        blocks=blocks,
    )


def build_dense_conv2d_payloads_if_enabled(
    conv_layer: Any,
    *,
    last: bool,
    allow_hybrid: bool = True,
    blocks: Iterable[tuple[int, int]] | None = None,
) -> tuple[list[DenseConv2DPayload], int, dict[str, Any]] | None:
    if not dense_builder_enabled():
        return None
    try:
        return build_dense_conv2d_payloads(
            conv_layer,
            last=bool(last),
            allow_hybrid=bool(allow_hybrid),
            blocks=blocks,
        )
    except Exception:
        if strict_enabled():
            raise
        return None


def build_dense_conv_transpose2d_payloads(
    conv_layer: Any,
    *,
    last: bool,
    allow_hybrid: bool = True,
    blocks: Iterable[tuple[int, int]] | None = None,
) -> tuple[list[DenseConv2DPayload], int, dict[str, Any]]:
    return _cpp_dense_conv_transpose2d_payloads(
        conv_layer,
        last=bool(last),
        allow_hybrid=bool(allow_hybrid),
        blocks=blocks,
    )


def build_dense_conv_transpose2d_payloads_if_enabled(
    conv_layer: Any,
    *,
    last: bool,
    allow_hybrid: bool = True,
    blocks: Iterable[tuple[int, int]] | None = None,
) -> tuple[list[DenseConv2DPayload], int, dict[str, Any]] | None:
    if not dense_builder_enabled():
        return None
    try:
        return build_dense_conv_transpose2d_payloads(
            conv_layer,
            last=bool(last),
            allow_hybrid=bool(allow_hybrid),
            blocks=blocks,
        )
    except Exception:
        if strict_enabled():
            raise
        return None


def build_dense_conv2d_index_only(
    conv_layer: Any,
    *,
    last: bool,
    allow_hybrid: bool = True,
) -> tuple[dict[tuple[int, int], tuple[int, ...]], int, dict[str, Any]]:
    return _cpp_dense_conv2d_index_only(
        conv_layer,
        last=bool(last),
        allow_hybrid=bool(allow_hybrid),
    )


def build_dense_conv_transpose2d_index_only(
    conv_layer: Any,
    *,
    last: bool,
    allow_hybrid: bool = True,
) -> tuple[dict[tuple[int, int], tuple[int, ...]], int, dict[str, Any]]:
    return _cpp_dense_conv_transpose2d_index_only(
        conv_layer,
        last=bool(last),
        allow_hybrid=bool(allow_hybrid),
    )


def build_provider_native_source_conv2d_payload(
    *,
    spec: Any,
    plan: Any,
    weight: torch.Tensor,
    weight_np: np.ndarray | None = None,
    stripe: Any,
    source_group: int,
    target_group: int,
    compact_target_block: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    lib = load_library()
    build = None if lib is None else getattr(lib, "OrionBuildProviderNativeSourceConv2D", None)
    if not callable(build):
        raise RuntimeError(load_error() or "provider native-source diag builder unavailable")
    source_tile = int(plan.source_tile_for_stripe(stripe))
    target_tile = int(plan.target_tile_for_stripe(stripe))
    target_index = (
        int(compact_target_block)
        if compact_target_block is not None
        else int(plan.target_block_index(stripe, int(target_group)))
    )
    source_index = int(plan.source_block_index(stripe, int(source_group)))
    c_spec = _CProviderNativeSourceSpec(
        int(spec.slot_count),
        int(spec.c_in),
        int(spec.h_in),
        int(spec.w_in),
        int(spec.c_out),
        int(spec.h_out),
        int(spec.w_out),
        int(spec.gap_in),
        int(spec.gap_out),
        int(spec.kernel),
        int(spec.stride),
        int(spec.pad),
        int(spec.dilation),
        int(spec.input_h_min),
        int(spec.input_h_max),
        int(spec.output_top_beta),
        int(spec.output_bottom_beta),
        int(spec.output_physical_top_beta or 0),
        int(spec.output_physical_bottom_beta or 0),
        int(stripe.index),
        int(stripe.source_h_start),
        int(stripe.source_h),
        int(stripe.target_h_start),
        int(stripe.target_h_end),
        int(stripe.target_h),
        int(source_tile),
        int(target_tile),
        int(source_group),
        int(target_group),
        int(source_index),
        int(target_index),
        int(compact_target_block is not None),
        int(-1 if compact_target_block is None else compact_target_block),
    )
    if weight_np is None:
        weight_np = np.ascontiguousarray(weight.detach().cpu().to(dtype=torch.float32).numpy().reshape(-1), dtype=np.float32)
    else:
        weight_np = np.ascontiguousarray(np.asarray(weight_np, dtype=np.float32).reshape(-1), dtype=np.float32)
    started = time.perf_counter()
    batch = build(
        c_spec,
        weight_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(weight_np.size),
    )
    build_s = float(time.perf_counter() - started)
    try:
        fallback_reason = _decode_c_string(batch.fallback_reason)
        if fallback_reason:
            raise RuntimeError(fallback_reason)
        payloads = _copy_payloads(batch)
        if not payloads:
            return None
        payload = payloads[0]
        metadata = {
            "diag_builder_kind": _decode_c_string(batch.builder_kind) or "cpp_provider_native_halo_conv2d:native_source",
            "diag_builder_source": "cpp",
            "diag_builder_build_s": float(build_s),
            "diag_builder_payload_count": 1.0,
            "diag_builder_fallback_reason": "",
        }
        return payload.diag_indices, payload.diag_data, metadata
    finally:
        lib.OrionFreeDiagPayloadBatch(batch)


def build_provider_compact_source_conv2d_payload(
    *,
    spec: Any,
    plan: Any,
    weight: torch.Tensor,
    stripe: Any,
    source_block: int,
    target_group: int,
    source_layout: dict[str, Any],
    compact_target_block: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    lib = load_library()
    build = None if lib is None else getattr(lib, "OrionBuildProviderCompactSourceConv2D", None)
    if not callable(build):
        raise RuntimeError(load_error() or "provider compact-source diag builder unavailable")
    target_tile = int(plan.target_tile_for_stripe(stripe))
    target_index = (
        int(compact_target_block)
        if compact_target_block is not None
        else int(plan.target_block_index(stripe, int(target_group)))
    )
    source_top_beta = max(
        0,
        int(
            source_layout.get(
                "physical_top_beta",
                source_layout.get("top_beta", int(getattr(spec, "input_physical_top_beta", 0) or 0)),
            )
            or 0
        ),
    )
    source_bottom_beta = max(
        0,
        int(
            source_layout.get(
                "physical_bottom_beta",
                source_layout.get("bottom_beta", int(getattr(spec, "input_physical_bottom_beta", 0) or 0)),
            )
            or 0
        ),
    )
    source_gap = max(1, int(source_layout.get("gap", spec.gap_in) or 1))
    c_spec = _CProviderCompactSourceSpec(
        int(spec.slot_count),
        int(spec.c_in),
        int(spec.h_in),
        int(spec.w_in),
        int(spec.c_out),
        int(spec.h_out),
        int(spec.w_out),
        int(spec.gap_out),
        int(spec.kernel),
        int(spec.stride),
        int(spec.pad),
        int(spec.dilation),
        int(source_top_beta),
        int(source_bottom_beta),
        int(source_gap),
        int(spec.output_top_beta),
        int(spec.output_bottom_beta),
        int(spec.output_physical_top_beta or 0),
        int(spec.output_physical_bottom_beta or 0),
        int(stripe.index),
        int(stripe.target_h_start),
        int(stripe.target_h_end),
        int(stripe.target_h),
        int(target_tile),
        int(source_block),
        int(target_group),
        int(target_index),
        int(compact_target_block is not None),
        int(-1 if compact_target_block is None else compact_target_block),
    )
    weight_np = np.ascontiguousarray(weight.detach().cpu().to(dtype=torch.float32).numpy().reshape(-1), dtype=np.float32)
    started = time.perf_counter()
    batch = build(
        c_spec,
        weight_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(weight_np.size),
    )
    build_s = float(time.perf_counter() - started)
    try:
        fallback_reason = _decode_c_string(batch.fallback_reason)
        if fallback_reason:
            raise RuntimeError(fallback_reason)
        payloads = _copy_payloads(batch)
        if not payloads:
            return None
        payload = payloads[0]
        metadata = {
            "diag_builder_kind": _decode_c_string(batch.builder_kind) or "cpp_provider_native_halo_conv2d:compact_source",
            "diag_builder_source": "cpp",
            "diag_builder_build_s": float(build_s),
            "diag_builder_payload_count": 1.0,
            "diag_builder_fallback_reason": "",
        }
        return payload.diag_indices, payload.diag_data, metadata
    finally:
        lib.OrionFreeDiagPayloadBatch(batch)


def build_provider_compact_source_concat_conv2d_indices(
    *,
    spec: Any,
    plan: Any,
    weight: torch.Tensor,
    source_layout: dict[str, Any],
    source_ct_count: int,
    target_ct_count: int,
    output_materialization: str = "",
    weight_np: np.ndarray | None = None,
) -> tuple[dict[int, list[tuple[int, tuple[int, ...]]]], dict[str, Any]] | None:
    lib = load_library()
    build = None if lib is None else getattr(lib, "OrionBuildProviderCompactSourceConcatConv2DIndexOnly", None)
    if not callable(build):
        raise RuntimeError(load_error() or "provider compact-source concat index diag builder unavailable")
    source_top_beta = max(
        0,
        int(
            source_layout.get(
                "physical_top_beta",
                source_layout.get("top_beta", int(getattr(spec, "input_physical_top_beta", 0) or 0)),
            )
            or 0
        ),
    )
    source_bottom_beta = max(
        0,
        int(
            source_layout.get(
                "physical_bottom_beta",
                source_layout.get("bottom_beta", int(getattr(spec, "input_physical_bottom_beta", 0) or 0)),
            )
            or 0
        ),
    )
    source_gap = max(1, int(source_layout.get("gap", spec.gap_in) or 1))
    c_spec = _CProviderCompactSourceConcatIndexSpec(
        int(spec.slot_count),
        int(spec.c_in),
        int(spec.h_in),
        int(spec.w_in),
        int(spec.c_out),
        int(spec.h_out),
        int(spec.w_out),
        int(spec.gap_out),
        int(spec.kernel),
        int(spec.stride),
        int(spec.pad),
        int(spec.dilation),
        int(source_top_beta),
        int(source_bottom_beta),
        int(source_gap),
        int(spec.output_top_beta),
        int(spec.output_bottom_beta),
        int(spec.output_physical_top_beta or 0),
        int(spec.output_physical_bottom_beta or 0),
        int(source_ct_count),
        int(target_ct_count),
        int(str(output_materialization or "") == "fused_relayout"),
    )
    stripe_specs = [
        _CProviderStripeSpec(
            int(stripe.index),
            int(stripe.target_h_start),
            int(stripe.target_h_end),
            int(stripe.target_h),
            int(plan.target_tile_for_stripe(stripe)),
            int(plan.target_group_count_for_stripe(stripe)),
        )
        for stripe in plan.stripes
    ]
    stripes_array = (_CProviderStripeSpec * len(stripe_specs))(*stripe_specs) if stripe_specs else None
    if weight_np is None:
        weight_np = np.ascontiguousarray(weight.detach().cpu().to(dtype=torch.float32).numpy().reshape(-1), dtype=np.float32)
    else:
        weight_np = np.ascontiguousarray(np.asarray(weight_np, dtype=np.float32).reshape(-1), dtype=np.float32)
    started = time.perf_counter()
    batch = build(
        c_spec,
        stripes_array,
        int(len(stripe_specs)),
        weight_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(weight_np.size),
    )
    build_s = float(time.perf_counter() - started)
    try:
        fallback_reason = _decode_c_string(batch.fallback_reason)
        if fallback_reason:
            raise RuntimeError(fallback_reason)
        payloads = _copy_payloads(batch)
        if not payloads:
            metadata = {
                "diag_builder_kind": _decode_c_string(batch.builder_kind) or "cpp_provider_native_halo_conv2d:compact_source_concat_index_only",
                "diag_builder_source": "cpp",
                "diag_builder_build_s": float(build_s),
                "diag_builder_payload_count": 0.0,
                "diag_builder_fallback_reason": "",
            }
            return {}, metadata
        by_source: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for payload in payloads:
            diag_tuple = tuple(int(value) for value in np.asarray(payload.diag_indices, dtype=np.int32).tolist())
            if not diag_tuple:
                continue
            by_source.setdefault(int(payload.col), []).append((int(payload.row), diag_tuple))
        ordered = {
            int(source_block): [
                (int(target_block), tuple(int(value) for value in diag_tuple))
                for target_block, diag_tuple in sorted(items, key=lambda item: int(item[0]))
            ]
            for source_block, items in sorted(by_source.items(), key=lambda item: int(item[0]))
        }
        metadata = {
            "diag_builder_kind": _decode_c_string(batch.builder_kind) or "cpp_provider_native_halo_conv2d:compact_source_concat_index_only",
            "diag_builder_source": "cpp",
            "diag_builder_build_s": float(build_s),
            "diag_builder_payload_count": float(len(payloads)),
            "diag_builder_fallback_reason": "",
        }
        return ordered, metadata
    finally:
        lib.OrionFreeDiagPayloadBatch(batch)
