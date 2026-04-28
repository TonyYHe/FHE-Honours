from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.region_lowering import (
    ConvRegionSpec,
    build_tile_local_conv_lt,
    merge_tile_lts_as_complex,
    pack_chw_gap,
    transform_from_tile_lt,
)
from orion.core.shared_lt import OutputBank, SourceTile, TargetTile
from orion.nn.unified_transform import UnifiedTransformGroup


RING_SLOT_COUNT = 32768


def _encode_plaintext_for_add(ct: Any, values: torch.Tensor) -> Any:
    scale = int(ct.scheme.params.get_default_scale())
    if bool(getattr(ct.scheme.backend, "align_addition_scales", False)):
        scale = max(1, int(ct.scale()))
        ct.set_scale(int(scale))
    return ct.scheme.encode(values, ct.level(), scale=scale)


def _add_plaintext_for_add(ct: Any, ptxt: Any) -> Any:
    if bool(getattr(ct.scheme.backend, "align_addition_scales", False)):
        scale = max(1, int(ct.scale()))
        ct.set_scale(int(scale))
        ptxt.set_scale(int(scale))
    return ct + ptxt


def _align_ciphertexts_for_add(left: Any, right: Any) -> tuple[Any, Any]:
    if bool(getattr(left.scheme.backend, "align_addition_scales", False)):
        scale = max(1, int(left.scale()))
        left.set_scale(int(scale))
        right.set_scale(int(scale))
    return left, right


@dataclass(frozen=True)
class R18TransitionBridgeSpec:
    stage: str
    c_in: int
    h_in: int
    w_in: int
    input_gap: int
    c_out: int
    h_out: int
    w_out: int
    output_gap: int

    @property
    def input_ct_channels(self) -> int:
        return int(8 * int(self.input_gap) * int(self.input_gap))

    @property
    def output_ct_channels(self) -> int:
        return int(8 * int(self.output_gap) * int(self.output_gap))

    @property
    def input_ct_count(self) -> int:
        return int(self.c_in // self.input_ct_channels)

    @property
    def output_ct_count(self) -> int:
        return int(self.c_out // self.output_ct_channels)


R18_STAGE12_TRANSITION_SPEC = R18TransitionBridgeSpec(
    stage="stage1_transition",
    c_in=64,
    h_in=64,
    w_in=64,
    input_gap=1,
    c_out=128,
    h_out=32,
    w_out=32,
    output_gap=2,
)
R18_STAGE23_TRANSITION_SPEC = R18TransitionBridgeSpec(
    stage="stage2_transition",
    c_in=128,
    h_in=32,
    w_in=32,
    input_gap=2,
    c_out=256,
    h_out=16,
    w_out=16,
    output_gap=4,
)
R18_STAGE34_TRANSITION_SPEC = R18TransitionBridgeSpec(
    stage="stage3_transition",
    c_in=256,
    h_in=16,
    w_in=16,
    input_gap=4,
    c_out=512,
    h_out=8,
    w_out=8,
    output_gap=8,
)


@dataclass(frozen=True)
class R18StemBridgeSpec:
    c_in: int = 3
    h_in: int = 64
    w_in: int = 64
    input_gap: int = 1
    c_out: int = 64
    h_out: int = 64
    w_out: int = 64
    output_gap: int = 1
    kernel: int = 7
    stride: int = 1
    pad: int = 3

    @property
    def output_ct_channels(self) -> int:
        return 8

    @property
    def output_ct_count(self) -> int:
        return int(self.c_out // self.output_ct_channels)


R18_STEM_BRIDGE_SPEC = R18StemBridgeSpec()


def _rescale_cipher_tensor(ct: Any) -> Any:
    if len(getattr(ct, "ids", ())) != 1:
        raise ValueError("bridge rescale helper expects a single-ciphertext tensor")
    if bool(getattr(ct.scheme.backend, "lt_outputs_are_rescaled", False)):
        return ct
    rescaled_id = ct.evaluator.rescale(int(ct.ids[0]), in_place=False)
    return type(ct)(ct.scheme, [int(rescaled_id)], ct.shape, ct.on_shape)


def _channel_split_source_tiles(*, c: int, h: int, w: int, gap: int, channels_per_ct: int) -> tuple[SourceTile, ...]:
    tiles = []
    for c0 in range(0, int(c), int(channels_per_ct)):
        c1 = min(int(c), int(c0 + int(channels_per_ct)))
        tiles.append(
            SourceTile(
                tile_id=f"source_c{int(c0)}_{int(c1)}_h0_{int(h)}",
                c_start=int(c0),
                c_end=int(c1),
                h_start=0,
                h_end=int(h),
                w=int(w),
                gap=int(gap),
            )
        )
    return tuple(tiles)


def _channel_split_target_tiles(*, c: int, h: int, w: int, gap: int, channels_per_ct: int) -> tuple[TargetTile, ...]:
    tiles = []
    for c0 in range(0, int(c), int(channels_per_ct)):
        c1 = min(int(c), int(c0 + int(channels_per_ct)))
        tiles.append(
            TargetTile(
                tile_id=f"target_c{int(c0)}_{int(c1)}_h0_{int(h)}",
                c_start=int(c0),
                c_end=int(c1),
                h_start=0,
                h_end=int(h),
                w=int(w),
                gap=int(gap),
            )
        )
    return tuple(tiles)


def _bias_chunk(bias_vector: torch.Tensor | None, *, row_index: int, slots: int) -> torch.Tensor | None:
    if bias_vector is None:
        return None
    out = torch.zeros((int(slots),), dtype=torch.float32)
    start = int(row_index) * int(slots)
    end = min(int(start + int(slots)), int(bias_vector.numel()))
    if end > start:
        out[: int(end - start)] = bias_vector[int(start): int(end)]
    return out


def build_r18_transition_bridge_assets(
    *,
    spec: R18TransitionBridgeSpec,
    conv_module: Any,
    shortcut_module: Any,
    level: int,
    scheme: Any,
) -> dict[str, Any]:
    weight_conv = getattr(conv_module, "on_weight")
    weight_short = getattr(shortcut_module, "on_weight")
    if weight_conv is None or weight_short is None:
        raise RuntimeError(f"{spec.stage} transition modules must expose fused weights")

    source_tiles = _channel_split_source_tiles(
        c=int(spec.c_in),
        h=int(spec.h_in),
        w=int(spec.w_in),
        gap=int(spec.input_gap),
        channels_per_ct=int(spec.input_ct_channels),
    )
    target_tiles = _channel_split_target_tiles(
        c=int(spec.c_out),
        h=int(spec.h_out),
        w=int(spec.w_out),
        gap=int(spec.output_gap),
        channels_per_ct=int(spec.output_ct_channels),
    )
    conv_spec = ConvRegionSpec(
        case_name=f"r18_{spec.stage}_main",
        c_in=int(spec.c_in),
        h_in=int(spec.h_in),
        w_in=int(spec.w_in),
        c_out=int(spec.c_out),
        h_out=int(spec.h_out),
        w_out=int(spec.w_out),
        kernel=3,
        stride=2,
        pad=1,
        input_gap=int(spec.input_gap),
        output_gap=int(spec.output_gap),
        slots=int(RING_SLOT_COUNT),
    )
    shortcut_spec = ConvRegionSpec(
        case_name=f"r18_{spec.stage}_shortcut",
        c_in=int(spec.c_in),
        h_in=int(spec.h_in),
        w_in=int(spec.w_in),
        c_out=int(spec.c_out),
        h_out=int(spec.h_out),
        w_out=int(spec.w_out),
        kernel=1,
        stride=2,
        pad=0,
        input_gap=int(spec.input_gap),
        output_gap=int(spec.output_gap),
        slots=int(RING_SLOT_COUNT),
    )
    baseline_groups: dict[int, list[tuple[int, Any, Any]]] = {}
    prototype_groups: dict[int, list[tuple[int, Any]]] = {}
    for source_index, source_tile in enumerate(source_tiles):
        for target_index, target_tile in enumerate(target_tiles):
            conv_bank = OutputBank(f"{spec.stage}_conv_bank_{int(target_index)}", str(target_tile.tile_id), "regular", "main")
            shortcut_bank = OutputBank(f"{spec.stage}_shortcut_bank_{int(target_index)}", str(target_tile.tile_id), "regular", "shortcut")
            conv_lt = build_tile_local_conv_lt(
                spec=conv_spec,
                source_tile=source_tile,
                target_tile=target_tile,
                output_bank=conv_bank,
                weight=weight_conv.to(dtype=torch.float32),
                transform_id=f"{spec.stage}_conv_src{int(source_index)}_dst{int(target_index)}",
            )
            shortcut_lt = build_tile_local_conv_lt(
                spec=shortcut_spec,
                source_tile=source_tile,
                target_tile=target_tile,
                output_bank=shortcut_bank,
                weight=weight_short.to(dtype=torch.float32),
                transform_id=f"{spec.stage}_shortcut_src{int(source_index)}_dst{int(target_index)}",
            )
            baseline_groups.setdefault(int(source_index), []).append(
                (
                    int(target_index),
                    transform_from_tile_lt(conv_lt, level=int(level), scheme=scheme, name=conv_lt.transform_id),
                    transform_from_tile_lt(shortcut_lt, level=int(level), scheme=scheme, name=shortcut_lt.transform_id),
                )
            )
            hybrid_lt = merge_tile_lts_as_complex(
                real_lt=conv_lt,
                imag_lt=shortcut_lt,
                transform_id=f"{spec.stage}_hybrid_src{int(source_index)}_dst{int(target_index)}",
            )
            prototype_groups.setdefault(int(source_index), []).append(
                (
                    int(target_index),
                    transform_from_tile_lt(hybrid_lt, level=int(level), scheme=scheme, name=hybrid_lt.transform_id),
                )
            )

    conv_bias_vector = packing.construct_conv2d_bias(conv_module).to(dtype=torch.float32)
    shortcut_bias_vector = packing.construct_conv2d_bias(shortcut_module).to(dtype=torch.float32)
    return {
        "spec": spec,
        "source_tiles": source_tiles,
        "target_tiles": target_tiles,
        "baseline_groups": {
            int(source_index): {
                "target_indices": tuple(int(target_index) for target_index, _conv_transform, _shortcut_transform in entries),
                "conv_transforms": [conv_transform for _target_index, conv_transform, _shortcut_transform in entries],
                "shortcut_transforms": [shortcut_transform for _target_index, _conv_transform, shortcut_transform in entries],
            }
            for source_index, entries in baseline_groups.items()
        },
        "prototype_groups": {
            int(source_index): {
                "target_indices": tuple(int(target_index) for target_index, _transform in entries),
                "transforms": [transform for _target_index, transform in entries],
            }
            for source_index, entries in prototype_groups.items()
        },
        "conv_bias_vector": conv_bias_vector,
        "shortcut_bias_vector": shortcut_bias_vector,
    }


class R18TransitionBridgeRuntimeExecutor:
    def __init__(
        self,
        *,
        conv_module: Any,
        shortcut_module: Any,
        spec: R18TransitionBridgeSpec,
        output_node_ids: tuple[str, str],
    ) -> None:
        self.conv_module = conv_module
        self.shortcut_module = shortcut_module
        self.spec = spec
        self.output_node_ids = tuple(str(value) for value in output_node_ids)
        self.groups_by_input_block: list[Any] = []
        self.rows = int(spec.output_ct_count)
        self.cols = int(spec.input_ct_count)
        self.conv_bias_vector: torch.Tensor | None = None
        self.shortcut_bias_vector: torch.Tensor | None = None
        self.conv_bias_plaintexts: tuple[Any | None, ...] = ()
        self.shortcut_bias_plaintexts: tuple[Any | None, ...] = ()
        self._conv_bias_plaintext_cache: dict[tuple[int, int], Any] = {}
        self._shortcut_bias_plaintext_cache: dict[tuple[int, int], Any] = {}
        self.compile_count = 0
        self.block_evaluate_count = 0
        self.output_shape = getattr(conv_module, "output_shape", None)
        self.fhe_output_shape = getattr(conv_module, "fhe_output_shape", None)
        self.last_runtime_timing: dict[str, float] = {
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self._target_indices_by_input: dict[int, tuple[int, ...]] = {}

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def _output_level(self, scheme: Any, *, extra_depth: int = 0) -> int:
        depth = int(self.assigned_depth) if self.assigned_depth is not None else 1
        return max(0, int(self._level(scheme)) - max(0, int(depth)) - max(0, int(extra_depth)))

    def _compile_bias_plaintexts(self, scheme: Any, bias_vector: torch.Tensor | None) -> tuple[Any | None, ...]:
        if bias_vector is None:
            return ()
        level = self._output_level(scheme)
        scale = int(scheme.params.get_default_scale())
        plaintexts: list[Any | None] = []
        for row_index in range(int(self.rows)):
            chunk = _bias_chunk(bias_vector, row_index=int(row_index), slots=int(RING_SLOT_COUNT))
            plaintexts.append(None if chunk is None else scheme.encode(chunk, level=int(level), scale=int(scale)))
        return tuple(plaintexts)

    def _add_cached_bias(
        self,
        ct: Any,
        *,
        row_index: int,
        bias_vector: torch.Tensor | None,
        plaintexts: tuple[Any | None, ...],
        cache: dict[tuple[int, int], Any],
    ) -> Any:
        if bias_vector is None:
            return ct
        bias_ptxt = plaintexts[int(row_index)] if int(row_index) < len(plaintexts) else None
        if bias_ptxt is None or int(bias_ptxt.level()) != int(ct.level()):
            bias_ptxt = cache.get((int(row_index), int(ct.level())))
        if bias_ptxt is None or int(bias_ptxt.level()) != int(ct.level()):
            chunk = _bias_chunk(bias_vector, row_index=int(row_index), slots=int(RING_SLOT_COUNT))
            if chunk is None:
                return ct
            bias_ptxt = _encode_plaintext_for_add(ct, chunk)
            cache[(int(row_index), int(ct.level()))] = bias_ptxt
        return _add_plaintext_for_add(ct, bias_ptxt)

    def compile(self, scheme: Any) -> None:
        if self.groups_by_input_block:
            return
        prepared_started = time.time()
        assets = build_r18_transition_bridge_assets(
            spec=self.spec,
            conv_module=self.conv_module,
            shortcut_module=self.shortcut_module,
            level=self._level(scheme),
            scheme=scheme,
        )
        self.conv_bias_vector = assets["conv_bias_vector"]
        self.shortcut_bias_vector = assets["shortcut_bias_vector"]
        self.conv_bias_plaintexts = self._compile_bias_plaintexts(scheme, self.conv_bias_vector)
        self.shortcut_bias_plaintexts = self._compile_bias_plaintexts(scheme, self.shortcut_bias_vector)
        for row_index, bias_ptxt in enumerate(self.conv_bias_plaintexts):
            if bias_ptxt is not None:
                self._conv_bias_plaintext_cache[(int(row_index), int(bias_ptxt.level()))] = bias_ptxt
        for row_index, bias_ptxt in enumerate(self.shortcut_bias_plaintexts):
            if bias_ptxt is not None:
                self._shortcut_bias_plaintext_cache[(int(row_index), int(bias_ptxt.level()))] = bias_ptxt
        self.last_runtime_timing["prepare_transforms_s"] = float(time.time() - prepared_started)

        compile_started = time.time()
        self.groups_by_input_block = []
        self._target_indices_by_input = {}
        for input_index in range(int(self.cols)):
            payload = assets["prototype_groups"][int(input_index)]
            group = UnifiedTransformGroup(list(payload["transforms"]))
            group.compile_unified(scheme.backend)
            self.groups_by_input_block.append(group)
            self._target_indices_by_input[int(input_index)] = tuple(int(value) for value in payload["target_indices"])
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing = {
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.compile(scheme)
        ids = tuple(int(value) for value in getattr(source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            raise RuntimeError(f"{self.spec.stage} transition bridge requires {self.cols} source ciphertext blocks, got {len(ids)}")

        complex_outputs: list[Any | None] = [None for _ in range(int(self.rows))]
        evaluate_started = time.time()
        for input_index, group in enumerate(self.groups_by_input_block):
            row_output_ids = group.evaluate_unified(int(ids[int(input_index)]), scheme.backend)
            self.block_evaluate_count += 1
            for target_index, output_id in zip(self._target_indices_by_input[int(input_index)], row_output_ids):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(RING_SLOT_COUNT)]),
                    torch.Size([1, int(RING_SLOT_COUNT)]),
                )
                if complex_outputs[int(target_index)] is None:
                    complex_outputs[int(target_index)] = partial
                else:
                    lhs, rhs = _align_ciphertexts_for_add(complex_outputs[int(target_index)], partial)
                    complex_outputs[int(target_index)] = lhs + rhs
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)

        postprocess_started = time.time()
        conv_ids: list[int] = []
        shortcut_ids: list[int] = []
        for row_index, row_ct in enumerate(complex_outputs):
            if row_ct is None:
                raise RuntimeError(f"missing transition output row {row_index} for {self.spec.stage}")
            complex_outputs[int(row_index)] = None
            row_ct = _rescale_cipher_tensor(row_ct)
            conj = row_ct.conjugate(in_place=False)
            row_ct, conj = _align_ciphertexts_for_add(row_ct, conj)
            conv_real = (row_ct + conj) * 0.5
            shortcut_imag = (row_ct - conj).mul_imaginary_unit(-1, in_place=False) * 0.5
            conv_real = self._add_cached_bias(
                conv_real,
                row_index=int(row_index),
                bias_vector=self.conv_bias_vector,
                plaintexts=self.conv_bias_plaintexts,
                cache=self._conv_bias_plaintext_cache,
            )
            shortcut_imag = self._add_cached_bias(
                shortcut_imag,
                row_index=int(row_index),
                bias_vector=self.shortcut_bias_vector,
                plaintexts=self.shortcut_bias_plaintexts,
                cache=self._shortcut_bias_plaintext_cache,
            )
            conv_real.set_scale(int(scheme.params.get_default_scale()))
            shortcut_imag.set_scale(int(scheme.params.get_default_scale()))
            conv_ids.append(int(conv_real.ids[0]))
            shortcut_ids.append(int(shortcut_imag.ids[0]))
            conv_real.ids = []
            shortcut_imag.ids = []
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {
            str(self.output_node_ids[0]): CipherTensor(scheme, conv_ids, self.output_shape, self.fhe_output_shape),
            str(self.output_node_ids[1]): CipherTensor(scheme, shortcut_ids, self.output_shape, self.fhe_output_shape),
        }


def build_r18_stem_bridge_assets(*, module: Any, level: int, scheme: Any) -> dict[str, Any]:
    weight = getattr(module, "on_weight")
    if weight is None:
        raise RuntimeError("stem bridge module must expose fused weights")
    spec = R18_STEM_BRIDGE_SPEC
    source_tiles = (
        SourceTile(
            tile_id="stem_source_c0_3_h0_64",
            c_start=0,
            c_end=int(spec.c_in),
            h_start=0,
            h_end=int(spec.h_in),
            w=int(spec.w_in),
            gap=int(spec.input_gap),
        ),
    )
    target_tiles = _channel_split_target_tiles(
        c=int(spec.c_out),
        h=int(spec.h_out),
        w=int(spec.w_out),
        gap=int(spec.output_gap),
        channels_per_ct=int(spec.output_ct_channels),
    )
    conv_spec = ConvRegionSpec(
        case_name="r18_stem_bridge",
        c_in=int(spec.c_in),
        h_in=int(spec.h_in),
        w_in=int(spec.w_in),
        c_out=int(spec.c_out),
        h_out=int(spec.h_out),
        w_out=int(spec.w_out),
        kernel=int(spec.kernel),
        stride=int(spec.stride),
        pad=int(spec.pad),
        input_gap=int(spec.input_gap),
        output_gap=int(spec.output_gap),
        slots=int(RING_SLOT_COUNT),
    )
    transforms: list[Any] = []
    target_indices: list[int] = []
    for target_index, target_tile in enumerate(target_tiles):
        bank = OutputBank(f"stem_bank_{int(target_index)}", str(target_tile.tile_id), "regular", "conv1")
        lt = build_tile_local_conv_lt(
            spec=conv_spec,
            source_tile=source_tiles[0],
            target_tile=target_tile,
            output_bank=bank,
            weight=weight.to(dtype=torch.float32),
            transform_id=f"stem_bridge_dst{int(target_index)}",
        )
        transforms.append(transform_from_tile_lt(lt, level=int(level), scheme=scheme, name=lt.transform_id))
        target_indices.append(int(target_index))
    return {
        "spec": spec,
        "transforms": transforms,
        "target_indices": tuple(target_indices),
        "bias_vector": packing.construct_conv2d_bias(module).to(dtype=torch.float32),
    }


class R18StemBridgeRuntimeExecutor:
    def __init__(self, *, module: Any, output_node_id: str) -> None:
        self.module = module
        self.output_node_id = str(output_node_id)
        self.group: Any | None = None
        self.target_indices: tuple[int, ...] = ()
        self.bias_vector: torch.Tensor | None = None
        self.bias_plaintexts: tuple[Any | None, ...] = ()
        self.compile_count = 0
        self.output_shape = getattr(module, "output_shape", None)
        self.fhe_output_shape = getattr(module, "fhe_output_shape", None)
        self.last_runtime_timing: dict[str, float] = {
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def _output_level(self, scheme: Any) -> int:
        depth = int(self.assigned_depth) if self.assigned_depth is not None else 1
        return max(0, int(self._level(scheme)) - max(0, int(depth)))

    def _compile_bias_plaintexts(self, scheme: Any) -> tuple[Any | None, ...]:
        if self.bias_vector is None:
            return ()
        level = self._output_level(scheme)
        scale = int(scheme.params.get_default_scale())
        plaintexts: list[Any | None] = []
        for row_index in range(len(self.target_indices)):
            chunk = _bias_chunk(self.bias_vector, row_index=int(row_index), slots=int(RING_SLOT_COUNT))
            plaintexts.append(None if chunk is None else scheme.encode(chunk, level, scale=scale))
        return tuple(plaintexts)

    def compile(self, scheme: Any) -> None:
        if self.group is not None:
            return
        prepared_started = time.time()
        assets = build_r18_stem_bridge_assets(module=self.module, level=self._level(scheme), scheme=scheme)
        self.bias_vector = assets["bias_vector"]
        self.target_indices = tuple(assets["target_indices"])
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme)
        self.last_runtime_timing["prepare_transforms_s"] = float(time.time() - prepared_started)

        compile_started = time.time()
        self.group = UnifiedTransformGroup(list(assets["transforms"]))
        self.group.compile_unified(scheme.backend)
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing = {
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.compile(scheme)
        ids = tuple(int(value) for value in getattr(source_ct, "ids", ()))
        if not ids:
            raise RuntimeError("stem bridge requires one source ciphertext")
        evaluate_started = time.time()
        output_ids = self.group.evaluate_unified(int(ids[0]), scheme.backend)
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)

        postprocess_started = time.time()
        conv_ids: list[int] = []
        for row_index, output_id in enumerate(output_ids):
            out_ct = CipherTensor(
                scheme,
                [int(output_id)],
                torch.Size([1, int(RING_SLOT_COUNT)]),
                torch.Size([1, int(RING_SLOT_COUNT)]),
            )
            out_ct = _rescale_cipher_tensor(out_ct)
            bias_ptxt = self.bias_plaintexts[int(row_index)] if int(row_index) < len(self.bias_plaintexts) else None
            if bias_ptxt is not None:
                out_ct = _add_plaintext_for_add(out_ct, bias_ptxt)
            out_ct.set_scale(int(scheme.params.get_default_scale()))
            conv_ids.append(int(out_ct.ids[0]))
            out_ct.ids = []
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {
            self.output_node_id: CipherTensor(scheme, conv_ids, self.output_shape, self.fhe_output_shape)
        }
