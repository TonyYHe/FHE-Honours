from __future__ import annotations

import math
import time
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any

import torch

from orion.core import packing
from orion.experimental.cir.lattigo_block import _slot_index
from orion.experimental.cir.runtime_group import RegionFirstRuntimeGroup
from orion.experimental.cir.transition_pool_provider import InputPairConvRuntimeExecutor
from orion.nn.linear import Conv2d, ConvTranspose2d
from orion.nn.pooling import AvgPool2d
from orion.nn.unified_transform import UnifiedTransformGroup

try:
    from orion.experimental.cir.runtime_group import _add_plaintext_for_add, _align_ciphertexts_for_add, _rescale_cipher_tensor
except ImportError:
    def _rescale_cipher_tensor(ct: Any) -> Any:
        if len(getattr(ct, "ids", ())) != 1:
            raise ValueError("region-first rescale helper expects a single-ciphertext tensor")
        if bool(getattr(ct.scheme.backend, "lt_outputs_are_rescaled", False)):
            return ct
        rescaled_id = ct.evaluator.rescale(int(ct.ids[0]), in_place=False)
        return type(ct)(ct.scheme, [int(rescaled_id)], ct.shape, ct.on_shape)

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


def _u22_tconv_module_supported(module: Any) -> bool:
    input_gap = int(getattr(module, "input_gap", -1))
    output_gap = int(getattr(module, "output_gap", -1))
    return bool(
        isinstance(module, ConvTranspose2d)
        and tuple(getattr(module, "kernel_size", ())) == (2, 2)
        and tuple(getattr(module, "stride", ())) == (2, 2)
        and tuple(getattr(module, "padding", ())) == (0, 0)
        and tuple(getattr(module, "output_padding", ())) == (0, 0)
        and tuple(getattr(module, "dilation", ())) == (1, 1)
        and int(getattr(module, "groups", 1)) == 1
        and int(input_gap) >= 2
        and int(input_gap) % 2 == 0
        and int(output_gap) * 2 == int(input_gap)
    )


def _u22_same_shape_conv_module_supported(module: Any) -> bool:
    return bool(
        isinstance(module, Conv2d)
        and tuple(getattr(module, "kernel_size", ())) == (3, 3)
        and tuple(getattr(module, "stride", ())) == (1, 1)
        and tuple(getattr(module, "padding", ())) == (1, 1)
        and tuple(getattr(module, "dilation", ())) == (1, 1)
        and int(getattr(module, "groups", 1)) == 1
        and int(getattr(module, "input_gap", -1)) == int(getattr(module, "output_gap", -2))
        and int(getattr(module, "input_shape", torch.Size([0, 0]))[1])
        == int(getattr(module, "output_shape", torch.Size([0, 1]))[1])
        and tuple(int(v) for v in getattr(module, "input_shape", torch.Size())[2:])
        == tuple(int(v) for v in getattr(module, "output_shape", torch.Size())[2:])
    )


def _u22_pool_module_supported(module: Any) -> bool:
    return bool(
        isinstance(module, AvgPool2d)
        and tuple(getattr(module, "kernel_size", ())) in {(2, 2), (3, 3)}
        and tuple(getattr(module, "stride", ())) == (2, 2)
        and tuple(getattr(module, "dilation", ())) == (1, 1)
        and int(getattr(module, "groups", 1)) == int(getattr(module, "input_shape", torch.Size([0, 0]))[1])
    )


def _u22_input_pair_conv_module_supported(module: Any) -> bool:
    return bool(
        isinstance(module, Conv2d)
        and not isinstance(module, ConvTranspose2d)
        and tuple(getattr(module, "kernel_size", ())) == (3, 3)
        and tuple(getattr(module, "stride", ())) == (1, 1)
        and tuple(getattr(module, "padding", ())) == (1, 1)
        and tuple(getattr(module, "dilation", ())) == (1, 1)
        and int(getattr(module, "groups", 1)) == 1
        and tuple(int(v) for v in getattr(module, "input_shape", torch.Size())[2:])
        == tuple(int(v) for v in getattr(module, "output_shape", torch.Size())[2:])
        and int(getattr(module, "input_gap", -1)) == int(getattr(module, "output_gap", -2))
    )


def _u22_input_pair_conv_stage(*, node: str, module: Any) -> str:
    in_channels = int(getattr(module, "input_shape")[1])
    out_channels = int(getattr(module, "output_shape")[1])
    if int(in_channels) != int(out_channels):
        return "channel_transition"
    return "single_block_conv"


def _ceil_pow2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (int(value) - 1).bit_length()


def _u22_packed_active_slots(module: Any) -> int:
    input_shape = getattr(module, "input_shape")
    c = int(input_shape[1])
    h = int(input_shape[2])
    w = int(input_shape[3])
    gap = max(1, int(getattr(module, "input_gap")))
    phase_count = int(gap * gap)
    groups = -(-int(c) // int(phase_count))
    return int(groups * h * gap * w * gap)


def _u22_same_shape_conv_runtime_supported(module: Any) -> bool:
    # The reused R34 same-shape materializer currently expects either
    # multi-block output or a single block that already fills the ring. Smaller
    # single-block outputs need Orion's output-fold rotations, which this
    # region executor does not materialize yet.
    output_length = int(_u22_packed_active_slots(module))
    slot_count = int(_u22_module_slot_count(module))
    return bool(int(output_length) > int(slot_count) or int(_ceil_pow2(output_length)) == int(slot_count))


def _u22_module_slot_count(module: Any) -> int:
    scheme = getattr(module, "scheme", None)
    params = getattr(scheme, "params", None)
    get_slots = getattr(params, "get_slots", None)
    if callable(get_slots):
        try:
            return max(1, int(get_slots()))
        except Exception:
            pass
    return 32768


def _u22_same_shape_conv_group(*, node: str, module: Any) -> RegionFirstRuntimeGroup:
    from orion.experimental.cir.r34_orion_same_shape import (
        R34InterGroupHybridSameShapeRuntimeExecutor,
        R34Pack2SameShapeRuntimeExecutor,
        R34SameShapeStageSpec,
        r34_same_shape_policy,
    )

    c = int(getattr(module, "input_shape")[1])
    h = int(getattr(module, "input_shape")[2])
    w = int(getattr(module, "input_shape")[3])
    gap = int(getattr(module, "input_gap"))
    slot_count = int(_u22_module_slot_count(module))
    policy = r34_same_shape_policy(c=int(c), gap=int(gap))
    spec = R34SameShapeStageSpec(
        family_label=f"u22_{str(node)}_same",
        stage=f"u22_{str(node)}",
        c=int(c),
        h=int(h),
        w=int(w),
        gap=int(gap),
        policy=policy,
        materializer=f"u22_same_shape_{policy}",
        slot_count=int(slot_count),
    )
    if str(policy) == "inter_group_hybrid":
        executor = R34InterGroupHybridSameShapeRuntimeExecutor(module=module, spec=spec, output_node_id=str(node))
    else:
        executor = R34Pack2SameShapeRuntimeExecutor(module=module, spec=spec, output_node_id=str(node))

    def _supports_selected_slot_count(scheme: Any | None) -> bool:
        if scheme is None:
            return False
        try:
            return int(scheme.params.get_slots()) == int(slot_count)
        except Exception:
            return False

    executor.supports_scheme = _supports_selected_slot_count
    return RegionFirstRuntimeGroup(
        region_id=f"u22_conv_{str(node)}",
        network="U22",
        stage="conv_same_shape",
        module_prefix=str(node),
        conv_nodes=(str(node),),
        strategy=f"u22_conv_same_shape_{policy}",
        materializer=f"u22_same_shape_{policy}_slots{int(slot_count)}",
        depth=1,
        solver_depth=1,
        boundary_actions=("same_shape_conv_provider",),
        expected_stats={},
        executable=True,
        fallback_reason="",
        output_node_ids=(str(node),),
        executor=executor,
        fused_weight_count=1,
    )


def _u22_input_pair_conv_group(*, node: str, module: Any, stage: str) -> RegionFirstRuntimeGroup:
    return RegionFirstRuntimeGroup(
        region_id=f"u22_{str(stage)}_{str(node)}",
        network="U22",
        stage=str(stage),
        module_prefix=str(node),
        conv_nodes=(str(node),),
        strategy="u22_input_pair_conv_shared_rotations",
        materializer="u22_input_pair_conv_shared_rotations",
        depth=1,
        solver_depth=1,
        boundary_actions=("input_pair_ctpt_hybrid", "shared_block_rotation_cache"),
        expected_stats={},
        executable=True,
        fallback_reason="",
        output_node_ids=(str(node),),
        executor=InputPairConvRuntimeExecutor(module=module, output_node_id=str(node)),
        fused_weight_count=1,
    )


def _u22_pool_group(*, node: str, module: Any) -> RegionFirstRuntimeGroup:
    return RegionFirstRuntimeGroup(
        region_id=f"u22_pool_downsample_{str(node)}",
        network="U22",
        stage="pool_downsample",
        module_prefix=str(node),
        conv_nodes=(str(node),),
        strategy="u22_pool_input_pair_shared_rotations",
        materializer="u22_pool_input_pair_shared_rotations",
        depth=1,
        solver_depth=1,
        boundary_actions=("input_pair_ctpt_hybrid", "shared_block_rotation_cache", "downsample_pool"),
        expected_stats={},
        executable=True,
        fallback_reason="",
        output_node_ids=(str(node),),
        executor=InputPairConvRuntimeExecutor(module=module, output_node_id=str(node)),
        fused_weight_count=1,
    )


def _merge_optional_tconv_source_block_transforms_to_complex(
    left: Any | None,
    right: Any | None,
    *,
    name: str,
    real_lane_input_scale: float = 1.0,
) -> Any:
    if left is None and right is None:
        raise ValueError("at least one source-block transform is required")
    anchor = left if left is not None else right
    slots = int(anchor.fhe_output_shape[-1])
    left_diags = dict(getattr(left, "diagonals", {}).get((0, 0), {})) if left is not None else {}
    right_diags = dict(getattr(right, "diagonals", {}).get((0, 0), {})) if right is not None else {}
    all_keys = sorted({int(key) for key in left_diags.keys()} | {int(key) for key in right_diags.keys()})
    merged: dict[int, torch.Tensor] = {}
    for key in all_keys:
        left_diag = left_diags.get(int(key))
        right_diag = right_diags.get(int(key))
        left_tensor = (
            left_diag.detach().clone().reshape(-1).to(dtype=torch.float32)
            if isinstance(left_diag, torch.Tensor)
            else torch.tensor(left_diag, dtype=torch.float32)
        ) if left_diag is not None else torch.zeros((int(slots),), dtype=torch.float32)
        right_tensor = (
            right_diag.detach().clone().reshape(-1).to(dtype=torch.float32)
            if isinstance(right_diag, torch.Tensor)
            else torch.tensor(right_diag, dtype=torch.float32)
        ) if right_diag is not None else torch.zeros((int(slots),), dtype=torch.float32)
        scale = float(real_lane_input_scale)
        merged[int(key)] = (
            left_tensor.to(dtype=torch.complex64) * float(scale)
            - 1j * right_tensor.to(dtype=torch.complex64) * float(scale)
        )
    return SimpleNamespace(
        name=str(name),
        diagonals={(0, 0): merged},
        level=int(getattr(anchor, "level")),
        scheme=getattr(anchor, "scheme"),
        fhe_output_shape=getattr(anchor, "fhe_output_shape"),
        output_shape=getattr(anchor, "output_shape"),
        target_index=int(getattr(anchor, "target_index", 0)),
    )


class TconvK2S2PythonRuntimeExecutor:
    kernel_kind = "tconv_k2s2_gap_halving_experimental"

    def __init__(
        self,
        *,
        module: Any,
        output_node_id: str,
        use_ct_pt_hybrid_packing: bool = True,
        project_complex_inputs_to_real: bool = True,
    ) -> None:
        if not _u22_tconv_module_supported(module):
            raise ValueError("U22 experimental tconv kernel only supports k=2, s=2, gap-halving ConvTranspose2d layers")
        self.module = module
        self.output_node_id = str(output_node_id)
        self.use_ct_pt_hybrid_packing = bool(use_ct_pt_hybrid_packing)
        self.project_complex_inputs_to_real = bool(project_complex_inputs_to_real)
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.groups: list[UnifiedTransformGroup] = []
        self.target_indices_by_input_unit: list[tuple[int, ...]] = []
        self.input_block_pairs: list[tuple[int, int | None]] = []
        self.complex_input_block_flags: list[bool] = []
        self.output_block_count = 0
        self.input_block_count = 0
        self.compile_count = 0
        self.block_evaluate_count = 0
        self.real_projection_count = 0
        self._bias_vector: torch.Tensor | None = None
        self._bias_ptxt_cache: dict[tuple[int, int], Any] = {}
        self.last_runtime_timing: dict[str, float] = {
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
            "projection_s": 0.0,
            "input_pack_s": 0.0,
            "partial_rescale_s": 0.0,
            "accumulate_s": 0.0,
            "real_extract_s": 0.0,
            "bias_s": 0.0,
            "total_call_s": 0.0,
        }
        self.last_runtime_counts: dict[str, int] = {}
        self.last_runtime_io: dict[str, Any] = {}
        self._compiled = False

    def _reset_call_observability(self) -> None:
        self.last_runtime_timing.update(
            {
                "evaluate_unified_s": 0.0,
                "postprocess_s": 0.0,
                "projection_s": 0.0,
                "input_pack_s": 0.0,
                "partial_rescale_s": 0.0,
                "accumulate_s": 0.0,
                "real_extract_s": 0.0,
                "bias_s": 0.0,
                "total_call_s": 0.0,
            }
        )
        self.last_runtime_counts = {
            "projection_count": 0,
            "rescale_count": 0,
            "conjugate_count": 0,
            "evaluate_count": 0,
            "accumulate_add_count": 0,
            "real_extract_count": 0,
            "bias_add_count": 0,
            "input_pack_count": 0,
        }
        self.last_runtime_io = {}

    def _cipher_ids_state(self, ids: list[int] | tuple[int, ...], *, scheme: Any) -> dict[str, Any]:
        levels: list[int | None] = []
        scales: list[int | None] = []
        scale_log2: list[float | None] = []
        slots: list[int | None] = []
        backend = scheme.backend
        for value in ids:
            cid = int(value)
            try:
                levels.append(int(backend.GetCiphertextLevel(cid)))
            except Exception:
                levels.append(None)
            try:
                scales.append(int(backend.GetCiphertextScale(cid)))
            except Exception:
                scales.append(None)
            try:
                scale_log2.append(float(backend.GetCiphertextScaleLog2(cid)))
            except Exception:
                scale_log2.append(None)
            try:
                slots.append(int(backend.GetCiphertextSlots(cid)))
            except Exception:
                slots.append(None)
        return {
            "id_count": int(len(ids)),
            "levels": levels,
            "scales": scales,
            "scale_log2": scale_log2,
            "slots": slots,
        }

    def _delete_temp_ciphertext_ids(self, scheme: Any, ids: list[int]) -> None:
        delete = getattr(getattr(scheme, "backend", None), "DeleteCiphertext", None)
        if not callable(delete):
            return
        for value in ids:
            try:
                delete(int(value))
            except Exception:
                pass

    def _real_lane_input_id(self, ciphertext_id: int, *, scheme: Any) -> tuple[int, list[int]]:
        if not bool(self.project_complex_inputs_to_real):
            return int(ciphertext_id), []
        owned: list[int] = []
        conj_id = int(scheme.evaluator.conjugate(int(ciphertext_id), False))
        sum_id = int(scheme.evaluator.add_ciphertext(int(ciphertext_id), int(conj_id), False))
        owned.extend([int(conj_id), int(sum_id)])
        self.real_projection_count += 1
        return int(sum_id), owned

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        backend = str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())
        if backend not in {"python", "lattigo", "cheddar"}:
            return False
        slots = int(scheme.params.get_slots())
        input_plane = int(torch.Size(getattr(self.module, "fhe_input_shape"))[2:].numel())
        output_plane = int(torch.Size(getattr(self.module, "fhe_output_shape"))[2:].numel())
        return int(max(input_plane, output_plane)) <= int(slots)

    def _block_layout(self, *, scheme: Any) -> dict[str, int]:
        slots = int(scheme.params.get_slots())
        on_ci = int(getattr(self.module, "fhe_input_shape")[1])
        on_hi = int(getattr(self.module, "fhe_input_shape")[2])
        on_wi = int(getattr(self.module, "fhe_input_shape")[3])
        on_co = int(getattr(self.module, "fhe_output_shape")[1])
        on_ho = int(getattr(self.module, "fhe_output_shape")[2])
        on_wo = int(getattr(self.module, "fhe_output_shape")[3])
        input_plane = int(on_hi * on_wi)
        output_plane = int(on_ho * on_wo)
        input_channels_per_block = max(1, int(slots // input_plane))
        output_channels_per_block = max(1, int(slots // output_plane))
        return {
            "slots": int(slots),
            "input_channels_per_block": int(input_channels_per_block),
            "output_channels_per_block": int(output_channels_per_block),
            "input_block_count": int(math.ceil(int(on_ci) / int(input_channels_per_block))),
            "output_block_count": int(math.ceil(int(on_co) / int(output_channels_per_block))),
        }

    def compile(self, scheme: Any) -> None:
        if self._compiled:
            return
        if not self.supports_scheme(scheme):
            raise RuntimeError("U22 experimental tconv kernel requires a compatible Python or Lattigo backend")
        self.last_runtime_timing = {
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
            "projection_s": 0.0,
            "input_pack_s": 0.0,
            "partial_rescale_s": 0.0,
            "accumulate_s": 0.0,
            "real_extract_s": 0.0,
            "bias_s": 0.0,
            "total_call_s": 0.0,
        }
        level = int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1
        prepare_started = time.time()
        transforms_by_source_block = self._build_transforms_by_source_block(scheme=scheme, level=int(level))
        self.last_runtime_timing["prepare_transforms_s"] = float(time.time() - prepare_started)
        self.groups = []
        self.target_indices_by_input_unit = []
        self.input_block_pairs = []
        self.complex_input_block_flags = []

        grouped_entries: list[tuple[tuple[int, int | None], bool, list[tuple[int, Any]]]] = []
        if bool(self.use_ct_pt_hybrid_packing) and int(self.input_block_count) > 1:
            for left_block in range(0, int(self.input_block_count), 2):
                right_block = int(left_block + 1)
                has_right = int(right_block) < int(self.input_block_count)
                entries: list[tuple[int, Any]] = []
                for output_block in range(int(self.output_block_count)):
                    left_transform = transforms_by_source_block[int(left_block)][int(output_block)]
                    right_transform = (
                        transforms_by_source_block[int(right_block)][int(output_block)]
                        if bool(has_right)
                        else None
                    )
                    if bool(has_right):
                        transform = _merge_optional_tconv_source_block_transforms_to_complex(
                            left_transform,
                            right_transform,
                            name=(
                                f"{self.output_node_id}_ctpt_hybrid_src"
                                f"{int(left_block)}_{int(right_block)}_out{int(output_block)}"
                            ),
                            real_lane_input_scale=(0.25 if bool(self.project_complex_inputs_to_real) else 0.5),
                        )
                    else:
                        transform = left_transform
                    entries.append((int(output_block), transform))
                grouped_entries.append(
                    (
                        (int(left_block), int(right_block) if bool(has_right) else None),
                        bool(has_right),
                        entries,
                    )
                )
        else:
            for source_block, transforms in enumerate(transforms_by_source_block):
                grouped_entries.append(
                    (
                        (int(source_block), None),
                        False,
                        [(int(output_block), transform) for output_block, transform in enumerate(transforms)],
                    )
                )

        compile_started = time.time()
        for pair, is_complex, entries in grouped_entries:
            ordered = sorted(entries, key=lambda item: int(item[0]))
            transforms = [transform for _target_index, transform in ordered]
            group = UnifiedTransformGroup(transforms)
            group.compile_unified(scheme.backend)
            self.groups.append(group)
            self.target_indices_by_input_unit.append(tuple(int(target_index) for target_index, _transform in ordered))
            self.input_block_pairs.append((int(pair[0]), None if pair[1] is None else int(pair[1])))
            self.complex_input_block_flags.append(bool(is_complex))
        bias_level = max(0, int(level) - max(0, int(self.assigned_depth) if self.assigned_depth is not None else 1))
        self._compile_bias_plaintexts(scheme=scheme, level=int(bias_level))
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)
        self._compiled = True

    def _build_transforms_by_source_block(self, *, scheme: Any, level: int) -> list[list[Any]]:
        c_in = int(getattr(self.module, "input_shape")[1])
        h_in = int(getattr(self.module, "input_shape")[2])
        w_in = int(getattr(self.module, "input_shape")[3])
        c_out = int(getattr(self.module, "output_shape")[1])
        h_out = int(getattr(self.module, "output_shape")[2])
        w_out = int(getattr(self.module, "output_shape")[3])
        input_gap = int(getattr(self.module, "input_gap"))
        weight = getattr(self.module, "on_weight").detach().to(dtype=torch.float32)
        output_gap = int(getattr(self.module, "output_gap"))
        layout = self._block_layout(scheme=scheme)
        slots = int(layout["slots"])
        self.input_block_count = int(layout["input_block_count"])
        self.output_block_count = int(layout["output_block_count"])

        diagonal_entries: list[list[dict[int, dict[int, float]]]] = [
            [
                {}
                for _output_block in range(int(self.output_block_count))
            ]
            for _input_block in range(int(self.input_block_count))
        ]
        for ic in range(int(c_in)):
            for ih in range(int(h_in)):
                for iw in range(int(w_in)):
                    src_slot = int(_slot_index(int(ic), int(ih), int(iw), h=int(h_in), w=int(w_in), gap=int(input_gap)))
                    source_block = int(src_slot // int(slots))
                    source_local = int(src_slot % int(slots))
                    for kh in range(2):
                        oh = int(ih * 2 + kh)
                        for kw in range(2):
                            ow = int(iw * 2 + kw)
                            for oc in range(int(c_out)):
                                coeff = float(weight[int(ic), int(oc), int(kh), int(kw)])
                                if coeff == 0.0:
                                    continue
                                out_slot = int(_slot_index(int(oc), int(oh), int(ow), h=int(h_out), w=int(w_out), gap=int(output_gap)))
                                output_block = int(out_slot // int(slots))
                                output_local = int(out_slot % int(slots))
                                diag_idx = (int(source_local) - int(output_local)) % int(slots)
                                row = diagonal_entries[int(source_block)][int(output_block)].setdefault(int(diag_idx), {})
                                row[int(output_local)] = float(row.get(int(output_local), 0.0) + coeff)

        transforms_by_source_block: list[list[Any]] = []
        for source_block in range(int(self.input_block_count)):
            block_transforms: list[Any] = []
            for output_block in range(int(self.output_block_count)):
                diagonals: dict[int, list[float]] = {}
                for diag_idx, slot_values in diagonal_entries[int(source_block)][int(output_block)].items():
                    diag = torch.zeros((int(slots),), dtype=torch.float32)
                    indices = torch.tensor(sorted(int(value) for value in slot_values.keys()), dtype=torch.int64)
                    values = torch.tensor([float(slot_values[int(index)]) for index in indices.tolist()], dtype=torch.float32)
                    diag.index_copy_(0, indices, values)
                    diagonals[int(diag_idx)] = diag.tolist()
                block_transforms.append(
                    SimpleNamespace(
                        name=f"{self.output_node_id}_experimental_tconv_src{int(source_block)}_out{int(output_block)}",
                        diagonals={(0, 0): diagonals or {0: [0.0] * int(slots)}},
                        level=int(level),
                        scheme=scheme,
                        fhe_output_shape=torch.Size([1, int(slots)]),
                        output_shape=torch.Size([1, int(slots)]),
                    )
                )
            transforms_by_source_block.append(block_transforms)
        return transforms_by_source_block

    def _bias_plaintext(self, *, scheme: Any, level: int, output_block: int):
        cache_key = (int(output_block), int(level))
        cached = self._bias_ptxt_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._bias_vector is None:
            self._bias_vector = packing.construct_conv_transpose2d_bias(self.module).to(dtype=torch.float32)
        slots = int(scheme.params.get_slots())
        start = int(output_block) * int(slots)
        stop = min(int(start + int(slots)), int(self._bias_vector.numel()))
        chunk = torch.zeros((int(slots),), dtype=torch.float32)
        if int(stop) > int(start):
            chunk[: int(stop - start)] = self._bias_vector[int(start) : int(stop)]
        ptxt = scheme.encode(chunk, level=int(level), scale=int(scheme.params.get_default_scale()))
        self._bias_ptxt_cache[cache_key] = ptxt
        return ptxt

    def _compile_bias_plaintexts(self, *, scheme: Any, level: int) -> None:
        for output_block in range(int(self.output_block_count)):
            self._bias_plaintext(scheme=scheme, level=int(level), output_block=int(output_block))

    def _assemble_output(self, output_ids: list[int], *, scheme: Any):
        from orion.backend.python.tensors import CipherTensor

        block_ids: list[int] = []
        for output_block, output_id in enumerate(output_ids):
            block_ct = CipherTensor(
                scheme,
                [int(output_id)],
                torch.Size([1, int(scheme.params.get_slots())]),
                torch.Size([1, int(scheme.params.get_slots())]),
            )
            bias_ptxt = self._bias_plaintext(
                scheme=scheme,
                level=int(block_ct.level()),
                output_block=int(output_block),
            )
            block_ct = _add_plaintext_for_add(block_ct, bias_ptxt)
            block_ids.append(int(block_ct.ids[0]))
            block_ct.ids = []
        return CipherTensor(
            scheme,
            block_ids,
            getattr(self.module, "output_shape"),
            getattr(self.module, "fhe_output_shape"),
        )

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        call_started = time.time()
        scheme = source_ct.scheme
        self._reset_call_observability()
        self.compile(scheme)
        ids = [int(value) for value in getattr(source_ct, "ids", ())]
        self.last_runtime_io["input"] = self._cipher_ids_state(ids, scheme=scheme)
        self.last_runtime_io["input_block_count"] = int(self.input_block_count)
        self.last_runtime_io["output_block_count"] = int(self.output_block_count)
        self.last_runtime_io["use_ct_pt_hybrid_packing"] = bool(self.use_ct_pt_hybrid_packing)
        self.last_runtime_io["project_complex_inputs_to_real"] = bool(self.project_complex_inputs_to_real)
        if len(ids) < int(self.input_block_count):
            raise RuntimeError(
                "U22 experimental tconv kernel requires one ciphertext per packed input block: "
                f"expected {self.input_block_count}, got {len(ids)}"
            )
        if len(self.groups) != int(self.input_block_count):
            expected = int(math.ceil(int(self.input_block_count) / 2)) if bool(self.use_ct_pt_hybrid_packing) else int(self.input_block_count)
            if len(self.groups) != int(expected):
                raise RuntimeError("U22 experimental tconv kernel was not compiled")
        accumulated: list[Any | None] = [None for _ in range(int(self.output_block_count))]
        real_input_ids: dict[int, int] = {}
        owned_temp_ids: list[int] = []

        def source_id_for_complex_block(block_index: int) -> int:
            block_index = int(block_index)
            if block_index not in real_input_ids:
                projection_started = time.time()
                real_id, owned = self._real_lane_input_id(int(ids[block_index]), scheme=scheme)
                self.last_runtime_timing["projection_s"] += float(time.time() - projection_started)
                if owned:
                    self.last_runtime_counts["projection_count"] += 1
                    self.last_runtime_counts["conjugate_count"] += 1
                real_input_ids[int(block_index)] = int(real_id)
                owned_temp_ids.extend(int(value) for value in owned)
            return int(real_input_ids[int(block_index)])

        for input_unit, group in enumerate(self.groups):
            left_block, right_block = self.input_block_pairs[int(input_unit)]
            if bool(self.complex_input_block_flags[int(input_unit)]):
                if right_block is None:
                    raise RuntimeError("U22 CT-PT hybrid input block pair is missing its imaginary lane")
                left_id = source_id_for_complex_block(int(left_block))
                right_id = source_id_for_complex_block(int(right_block))
                input_pack_started = time.time()
                imag_id = int(scheme.evaluator.mul_imaginary_unit(int(right_id), +1, False))
                input_id = int(scheme.evaluator.add_ciphertext(int(left_id), int(imag_id), False))
                self.last_runtime_timing["input_pack_s"] += float(time.time() - input_pack_started)
                self.last_runtime_counts["input_pack_count"] += 1
                owned_temp_ids.extend([int(imag_id), int(input_id)])
            else:
                input_id = int(ids[int(left_block)])
            evaluate_started = time.time()
            output_ids = group.evaluate_unified(int(input_id), scheme.backend)
            self.last_runtime_timing["evaluate_unified_s"] += float(time.time() - evaluate_started)
            self.last_runtime_counts["evaluate_count"] += 1
            self.block_evaluate_count += 1
            for output_block, output_id in zip(self.target_indices_by_input_unit[int(input_unit)], output_ids):
                from orion.backend.python.tensors import CipherTensor

                block_ct = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(scheme.params.get_slots())]),
                    torch.Size([1, int(scheme.params.get_slots())]),
                )
                rescale_started = time.time()
                block_ct = _rescale_cipher_tensor(block_ct)
                self.last_runtime_timing["partial_rescale_s"] += float(time.time() - rescale_started)
                self.last_runtime_counts["rescale_count"] += 1
                if accumulated[int(output_block)] is None:
                    accumulated[int(output_block)] = block_ct
                else:
                    accumulate_started = time.time()
                    lhs, rhs = _align_ciphertexts_for_add(accumulated[int(output_block)], block_ct)
                    accumulated[int(output_block)] = lhs + rhs
                    self.last_runtime_timing["accumulate_s"] += float(time.time() - accumulate_started)
                    self.last_runtime_counts["accumulate_add_count"] += 1
        postprocess_started = time.time()
        final_ids: list[int] = []
        needs_real_lane_extract = any(bool(value) for value in self.complex_input_block_flags)
        for output_block, block_ct in enumerate(accumulated):
            if block_ct is None:
                raise RuntimeError(f"U22 experimental tconv kernel missing output block {output_block}")
            if bool(needs_real_lane_extract):
                real_extract_started = time.time()
                conj = block_ct.conjugate(in_place=False)
                block_ct, conj = _align_ciphertexts_for_add(block_ct, conj)
                block_ct = block_ct + conj
                self.last_runtime_timing["real_extract_s"] += float(time.time() - real_extract_started)
                self.last_runtime_counts["conjugate_count"] += 1
                self.last_runtime_counts["real_extract_count"] += 1
            final_ids.append(int(block_ct.ids[0]))
            block_ct.ids = []
        self._delete_temp_ciphertext_ids(scheme, owned_temp_ids)
        bias_started = time.time()
        out = self._assemble_output(final_ids, scheme=scheme)
        self.last_runtime_timing["bias_s"] = float(time.time() - bias_started)
        self.last_runtime_counts["bias_add_count"] = int(len(final_ids))
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        self.last_runtime_timing["total_call_s"] = float(time.time() - call_started)
        self.last_runtime_io["output"] = self._cipher_ids_state([int(value) for value in getattr(out, "ids", ())], scheme=scheme)
        return {self.output_node_id: out}


@dataclass
class U22CompileRegistry:
    groups: tuple[RegionFirstRuntimeGroup, ...]
    graph_audit: dict[str, Any]

    @classmethod
    def for_dag(
        cls,
        dag,
        *,
        allowed_nodes: tuple[str, ...] | None = None,
        enable_conv_kernels: bool = False,
    ) -> "U22CompileRegistry":
        groups: list[RegionFirstRuntimeGroup] = []
        excluded_nodes: list[dict[str, str]] = []
        allowed = None if allowed_nodes is None else {str(value) for value in allowed_nodes}
        selected_tconv_count = 0
        selected_conv_count = 0
        selected_pool_count = 0
        selected_generic_conv_count = 0
        for node in dag.topological_sort():
            module = dag.nodes[node].get("module")
            if isinstance(module, ConvTranspose2d):
                if allowed is not None and str(node) not in allowed:
                    excluded_nodes.append(
                        {
                            "node": str(node),
                            "reason": "u22_ablation_filtered_out",
                        }
                    )
                    continue
                if not _u22_tconv_module_supported(module):
                    excluded_nodes.append(
                        {
                            "node": str(node),
                            "reason": "experimental_tconv_requires_k2s2_gap_halving",
                        }
                    )
                    continue
                groups.append(
                    RegionFirstRuntimeGroup(
                        region_id=f"u22_tconv_{node}",
                        network="U22",
                        stage="decoder_up",
                        module_prefix=str(node),
                        conv_nodes=(str(node),),
                        strategy="tconv_k2s2_gap_halving_experimental",
                        materializer="tconv_k2s2_gap_halving_experimental",
                        depth=1,
                        solver_depth=1,
                        boundary_actions=("packed_slot_gather", "phase_halving_output_repack"),
                        expected_stats={},
                        executable=True,
                        fallback_reason="",
                        output_node_ids=(str(node),),
                        executor=TconvK2S2PythonRuntimeExecutor(module=module, output_node_id=str(node)),
                        fused_weight_count=1,
                    )
                )
                selected_tconv_count += 1
            elif bool(enable_conv_kernels) and isinstance(module, AvgPool2d):
                if not _u22_pool_module_supported(module):
                    excluded_nodes.append(
                        {
                            "node": str(node),
                            "reason": "u22_pool_requires_stride2_avgpool",
                        }
                    )
                    continue
                groups.append(_u22_pool_group(node=str(node), module=module))
                selected_pool_count += 1
            elif bool(enable_conv_kernels) and isinstance(module, Conv2d):
                if not _u22_same_shape_conv_module_supported(module):
                    if _u22_input_pair_conv_module_supported(module):
                        stage = _u22_input_pair_conv_stage(node=str(node), module=module)
                        groups.append(_u22_input_pair_conv_group(node=str(node), module=module, stage=str(stage)))
                        selected_generic_conv_count += 1
                        selected_conv_count += 1
                    else:
                        excluded_nodes.append(
                            {
                                "node": str(node),
                                "reason": "u22_conv_requires_3x3_stride1_same_spatial_layout",
                            }
                        )
                    continue
                if not _u22_same_shape_conv_runtime_supported(module):
                    if _u22_input_pair_conv_module_supported(module):
                        groups.append(
                            _u22_input_pair_conv_group(
                                node=str(node),
                                module=module,
                                stage=_u22_input_pair_conv_stage(node=str(node), module=module),
                            )
                        )
                        selected_generic_conv_count += 1
                        selected_conv_count += 1
                        continue
                    excluded_nodes.append(
                        {
                            "node": str(node),
                            "reason": "u22_conv_single_block_fold_unsupported",
                        }
                    )
                    continue
                groups.append(_u22_same_shape_conv_group(node=str(node), module=module))
                selected_conv_count += 1
        return cls(
            groups=tuple(groups),
            graph_audit={
                "node_count": int(len(dag.nodes)),
                "edge_count": int(len(dag.edges)),
                "selected_region_count": int(len(groups)),
                "selected_tconv_count": int(selected_tconv_count),
                "selected_conv_count": int(selected_conv_count),
                "selected_pool_count": int(selected_pool_count),
                "selected_generic_conv_count": int(selected_generic_conv_count),
                "allowed_nodes": None if allowed_nodes is None else [str(value) for value in allowed_nodes],
                "enable_conv_kernels": bool(enable_conv_kernels),
                "excluded_nodes": excluded_nodes,
            },
        )

    def attach_to_dag(self, dag) -> dict[str, Any]:
        attached: list[dict[str, Any]] = []
        for group in self.groups:
            for node in group.conv_nodes:
                if node not in dag.nodes:
                    continue
                module = dag.nodes[node].get("module")
                if module is None:
                    continue
                module.region_runtime = group
                module.region_output_id = str(node)
                module.region_first_skip_dense_pack = bool(group.executable)
                if bool(group.executable) and hasattr(module, "set_depth"):
                    module.set_depth(int(group.effective_depth()))
                attached.append(
                    {
                        "node": str(node),
                        "stage": str(group.stage),
                        "executable": bool(group.executable),
                    }
                )
        return {
            "attached_count": int(len(attached)),
            "attached": attached,
            "executable_region_count": int(sum(1 for group in self.groups if bool(group.executable))),
            "graph_audit": dict(self.graph_audit),
        }
