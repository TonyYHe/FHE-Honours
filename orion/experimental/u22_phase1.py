from __future__ import annotations

import math
import os
import time
from types import SimpleNamespace
from dataclasses import dataclass, replace
from typing import Any

import torch

from orion.core import packing
from orion.experimental.cir.hybrid_schedule import (
    hybrid_pair_schedule_compatible,
    hybrid_pair_schedule_reject_reason,
    mark_hybrid_schedule_padding_allowed,
    materialize_hybrid_pair_layout_schedules,
    optimize_hybrid_pair_layout,
)
from orion.experimental.cir.lattigo_block import _idx_chw_gap_tensor
from orion.experimental.cir.halo_local_conv_provider import HaloLocalConvRuntimeExecutor
from orion.experimental.cir.native_halo_conv2d import NativeHaloRelayoutKernel
from orion.experimental.cir.runtime_group import RegionFirstRuntimeGroup
from orion.experimental.cir.transition_pool_provider import InputPairConvRuntimeExecutor
from orion.nn.linear import Conv2d, ConvTranspose2d
from orion.nn.pooling import AvgPool2d
from orion.nn.unified_transform import UnifiedTransformGroup


def _cached_transform_shell(*, level: int, scheme: Any) -> Any:
    return SimpleNamespace(diagonals={}, level=int(level), scheme=scheme)


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(str(name), "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

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
    # The generic halo-local same-shape materializer currently expects either
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


def _u22_same_shape_conv_group(
    *,
    node: str,
    module: Any,
) -> RegionFirstRuntimeGroup:
    slot_count = int(_u22_module_slot_count(module))
    executor = HaloLocalConvRuntimeExecutor(
        module=module,
        output_node_id=str(node),
    )

    return RegionFirstRuntimeGroup(
        region_id=f"u22_conv_{str(node)}",
        network="U22",
        stage="conv_same_shape",
        module_prefix=str(node),
        conv_nodes=(str(node),),
        strategy="u22_native_halo_stripe_no_ri_conv_same_shape",
        materializer=f"native_halo_stripe_no_ri_conv2d_slots{int(slot_count)}",
        depth=1,
        solver_depth=1,
        boundary_actions=("halo_local_conv2d_provider", "same_shape_tile_local_halo"),
        expected_stats={},
        executable=True,
        fallback_reason="",
        output_node_ids=(str(node),),
        executor=executor,
        fused_weight_count=1,
    )


def _u22_input_pair_conv_group(
    *,
    node: str,
    module: Any,
    stage: str,
) -> RegionFirstRuntimeGroup:
    boundary_actions = ("input_pair_no_real_imag", "shared_block_rotation_cache")
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
        boundary_actions=boundary_actions,
        expected_stats={},
        executable=True,
        fallback_reason="",
        output_node_ids=(str(node),),
        executor=InputPairConvRuntimeExecutor(
            module=module,
            output_node_id=str(node),
            use_ct_pt_hybrid_packing=False,
        ),
        fused_weight_count=1,
    )


def _u22_pool_group(*, node: str, module: Any) -> RegionFirstRuntimeGroup:
    boundary_actions = ("input_pair_no_real_imag", "shared_block_rotation_cache", "downsample_pool")
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
        boundary_actions=boundary_actions,
        expected_stats={},
        executable=True,
        fallback_reason="",
        output_node_ids=(str(node),),
        executor=InputPairConvRuntimeExecutor(
            module=module,
            output_node_id=str(node),
            use_ct_pt_hybrid_packing=False,
        ),
        fused_weight_count=1,
    )


def _normalize_u22_layout_policy(value: str) -> str:
    normalized = str(value or "dp").strip().lower()
    if normalized in {"fixed", "fixedmax", "fixed-max", "fixed_max"}:
        return "fixed_max"
    if normalized in {"eager", "greedy", "dp"}:
        return str(normalized)
    return "dp"


def _layout_policy_stage_for_module(module: Any) -> str:
    if isinstance(module, ConvTranspose2d):
        return "layout_policy_tconv"
    if isinstance(module, AvgPool2d):
        return "layout_policy_pool"
    if isinstance(module, Conv2d):
        return "layout_policy_conv"
    return ""


def _layout_policy_node_plan(compile_plan: dict[str, Any], *, node: str) -> dict[str, Any]:
    edge_layouts = [
        dict(row)
        for row in compile_plan.get("edge_layouts", [])
        if str(row.get("source")) == str(node) or str(row.get("target")) == str(node)
    ]
    node_layouts = [
        dict(row)
        for row in compile_plan.get("node_layouts", [])
        if str(row.get("node")) == str(node)
    ]
    return {
        "policy": str(compile_plan.get("policy", "")),
        "node": str(node),
        "runtime_lowering": "backend_encrypted_module",
        "edge_layouts": edge_layouts,
        "node_layouts": node_layouts,
        "is_last_linear": not any(str(row.get("source")) == str(node) for row in edge_layouts),
    }


def _layout_policy_relayout_enabled() -> bool:
    return str(os.environ.get("ORION_LAYOUT_POLICY_RELAYOUT_KERNEL", "1")).strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _layout_policy_incoming_relayout_rows(compile_plan: dict[str, Any], *, node: str) -> tuple[dict[str, Any], ...]:
    if not _layout_policy_relayout_enabled():
        return ()
    rows = []
    for row in compile_plan.get("edge_layouts", []):
        if str(row.get("target")) != str(node) or not bool(row.get("relayout", False)):
            continue
        if str(row.get("physical_layout", "")) == "native_source_stripe":
            continue
        layout = dict(row.get("selected_layout", {}))
        if (
            str(row.get("op_kind", "")) != "add"
            and int(layout.get("alpha", 0)) == 0
            and int(layout.get("beta", 0)) == 0
        ):
            continue
        rows.append(dict(row))
    return tuple(rows)


def _layout_policy_has_halo(layout: dict[str, Any]) -> bool:
    return bool(int(layout.get("alpha", 0)) > 0 or int(layout.get("beta", 0)) > 0)


def _layout_policy_incoming_native_rows(compile_plan: dict[str, Any], *, node: str) -> tuple[dict[str, Any], ...]:
    rows = []
    for row in compile_plan.get("edge_layouts", []):
        if str(row.get("target")) != str(node):
            continue
        if str(row.get("physical_layout", "")) != "native_source_stripe":
            continue
        if str(row.get("op_kind", "conv2d")) not in {"conv2d", "avgpool2d", "conv_transpose2d"}:
            continue
        layout = dict(row.get("selected_layout", {}))
        if not _layout_policy_has_halo(layout):
            continue
        rows.append(dict(row))
    return tuple(rows)


def _layout_policy_incoming_compact_align_shared_rows(
    compile_plan: dict[str, Any],
    *,
    node: str,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for row in compile_plan.get("edge_layouts", []):
        if str(row.get("target")) != str(node):
            continue
        if str(row.get("layout_mode", "")) not in {"compact_align_shared", "compact_halo_shared"}:
            continue
        if bool(row.get("relayout", False)):
            continue
        rows.append(dict(row))
    return tuple(rows)


def _layout_policy_incoming_compact_source_rows(
    compile_plan: dict[str, Any],
    *,
    node: str,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for row in compile_plan.get("edge_layouts", []):
        if str(row.get("target")) != str(node):
            continue
        if bool(row.get("relayout", False)):
            continue
        if str(row.get("op_kind", "conv2d")) not in {"conv2d", "avgpool2d", "conv_transpose2d"}:
            continue
        if str(row.get("physical_layout", "")) not in {"packed_compact", "logical_halo_compact"}:
            continue
        rows.append(dict(row))
    return tuple(rows)


def _layout_policy_native_output_row(compile_plan: dict[str, Any], *, node: str) -> dict[str, Any] | None:
    for row in compile_plan.get("node_layouts", []):
        if str(row.get("node")) != str(node):
            continue
        layout = dict(row.get("selected_layout", {}))
        if not _layout_policy_has_halo(layout):
            continue
        shape = [int(value) for value in row.get("shape", [])]
        if len(shape) != 4:
            continue
        return dict(row)
    return None


def _layout_policy_output_relayout_rows(compile_plan: dict[str, Any], *, node: str) -> tuple[dict[str, Any], ...]:
    if not _layout_policy_relayout_enabled():
        return ()
    rows = []
    for row in compile_plan.get("node_layouts", []):
        if str(row.get("node")) != str(node) or not bool(row.get("output_relayout", False)):
            continue
        layout = dict(row.get("selected_layout", {}))
        if not _layout_policy_has_halo(layout):
            continue
        shape = [int(value) for value in row.get("shape", [])]
        if len(shape) != 4:
            continue
        updated = {
            "edge": f"{str(node)}->__layout_policy_output__",
            "source": str(node),
            "target": str(node),
            "op_kind": "producer_output",
            "shape": shape,
            "fhe_shape": [int(value) for value in row.get("fhe_shape", shape)],
            "selected_layout": dict(layout),
            "relayout": True,
            "relayout_reason": str(row.get("output_relayout_reason", "producer_materialized_halo")),
        }
        rows.append(updated)
    return tuple(rows)


def _layout_policy_input_pair_native_halo_enabled() -> bool:
    return str(os.environ.get("ORION_LAYOUT_POLICY_PROVIDER_NATIVE_HALO", "1")).strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _layout_policy_compact_on_shape(edge_row: dict[str, Any]) -> torch.Size:
    layout = dict(edge_row.get("selected_layout", {}))
    compact_layout = {**layout, "alpha": 0, "beta": 0}
    return _layout_policy_on_shape(edge_row, compact_layout)


def _layout_policy_on_shape(edge_row: dict[str, Any], layout: dict[str, Any]) -> torch.Size:
    shape = [int(value) for value in edge_row.get("shape", [])]
    if len(shape) != 4:
        raise ValueError(f"layout-policy relayout edge has invalid shape: {shape}")
    n, channels, height, width = shape
    gap = max(1, int(layout.get("gap", 1)))
    alpha = max(0, int(layout.get("alpha", 0)))
    beta = max(0, int(layout.get("beta", 0)))
    on_channels = int(math.ceil(int(channels) / float(gap * gap)))
    return torch.Size(
        (
            int(n),
            int(on_channels),
            int(height * gap + (alpha + beta) * gap),
            int(width * gap),
        )
    )


def _layout_policy_halo_on_shape(edge_row: dict[str, Any]) -> torch.Size:
    layout = dict(edge_row.get("selected_layout", {}))
    return _layout_policy_on_shape(edge_row, layout)


def _layout_policy_native_halo_input_supported(base_executor: Any, relayout_rows: tuple[dict[str, Any], ...]) -> bool:
    if not _layout_policy_input_pair_native_halo_enabled():
        return False
    if len(relayout_rows) != 1:
        return False
    native_capable = bool(getattr(base_executor, "native_halo_input_capable", False))
    if not isinstance(base_executor, InputPairConvRuntimeExecutor) and not native_capable:
        return False
    module = getattr(base_executor, "module", None)
    if module is None:
        return False
    layout = dict(relayout_rows[0].get("selected_layout", {}))
    if int(layout.get("alpha", 0)) == 0 and int(layout.get("beta", 0)) == 0:
        return False
    return int(layout.get("gap", getattr(module, "input_gap", 1))) == int(getattr(module, "input_gap", 1))


def _layout_policy_native_module_attrs(
    compile_plan: dict[str, Any],
    *,
    node: str,
    base_executor: Any,
    native_input_rows: tuple[dict[str, Any], ...],
    compact_input_rows: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    input_rows = tuple(native_input_rows) if native_input_rows else tuple(compact_input_rows)
    if input_rows:
        row = dict(input_rows[0])
        layout = dict(row.get("selected_layout", {}))
        required_layout = dict(row.get("required_layout", layout) or layout)
        gap = max(1, int(layout.get("gap", 1)))
        alpha = max(0, int(layout.get("alpha", 0)))
        attrs.update(
            {
                "fhe_input_shape": _layout_policy_halo_on_shape(row),
                "layout_policy_input_row_offset": int(alpha * gap),
                "layout_policy_input_layout": dict(layout),
                "layout_policy_required_input_layout": dict(required_layout),
                "layout_policy_input_physical_layout": str(row.get("physical_layout", "")),
            }
        )
    if bool(getattr(base_executor, "native_halo_output_capable", False)):
        output_row = _layout_policy_native_output_row(compile_plan, node=str(node))
        if output_row is not None:
            output_layout = dict(output_row.get("selected_layout", {}))
            output_gap = max(1, int(output_layout.get("gap", 1)))
            output_alpha = max(0, int(output_layout.get("alpha", 0)))
            attrs.update(
                {
                    "fhe_output_shape": _layout_policy_halo_on_shape(output_row),
                    "layout_policy_output_row_offset": int(output_alpha * output_gap),
                    "layout_policy_output_layout": dict(output_layout),
                }
            )
            if bool(output_row.get("producer_materialized_halo", False)) or str(
                output_row.get("physical_layout", "")
            ) == "logical_halo_compact":
                attrs["layout_policy_output_materialization"] = "fused_relayout"
    return attrs


def _layout_policy_with_base_executor_attrs(base_executor: Any, attrs: dict[str, Any], callback: Any) -> Any:
    module = getattr(base_executor, "module", None)
    if not attrs or module is None:
        return callback()
    saved = {name: getattr(module, name, _MISSING) for name in attrs}
    base_saved = {
        name: getattr(base_executor, name, _MISSING)
        for name in attrs
        if hasattr(base_executor, name)
    }
    try:
        for name, value in attrs.items():
            setattr(module, name, value)
            if name in base_saved:
                setattr(base_executor, name, value)
        return callback()
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                try:
                    delattr(module, name)
                except AttributeError:
                    pass
            else:
                setattr(module, name, value)
        for name, value in base_saved.items():
            if value is _MISSING:
                try:
                    delattr(base_executor, name)
                except AttributeError:
                    pass
            else:
                setattr(base_executor, name, value)


def _layout_policy_native_halo_plan(
    base_executor: Any,
    compile_plan: dict[str, Any],
    *,
    node: str,
    native_input_rows: tuple[dict[str, Any], ...],
) -> Any | None:
    attrs = _layout_policy_native_module_attrs(
        compile_plan,
        node=str(node),
        base_executor=base_executor,
        native_input_rows=native_input_rows,
    )

    def _read_plan() -> Any | None:
        delegate = getattr(base_executor, "delegate", base_executor)
        refresh = getattr(delegate, "_refresh_runtime_plan", None)
        if callable(refresh):
            refresh()
        return getattr(delegate, "native_plan", None)

    return _layout_policy_with_base_executor_attrs(base_executor, attrs, _read_plan)


def _layout_policy_native_input_ct_count(
    base_executor: Any,
    compile_plan: dict[str, Any],
    *,
    node: str,
    native_input_rows: tuple[dict[str, Any], ...],
) -> int:
    plan = _layout_policy_native_halo_plan(
        base_executor,
        compile_plan,
        node=str(node),
        native_input_rows=native_input_rows,
    )
    try:
        return int(getattr(plan, "input_ct_count"))
    except Exception:
        return 0


def _layout_policy_native_physical_relayout_rows(
    base_executor: Any,
    compile_plan: dict[str, Any],
    *,
    node: str,
    native_input_rows: tuple[dict[str, Any], ...],
    relayout_rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    if not _layout_policy_native_halo_input_supported(base_executor, native_input_rows):
        return ()
    if len(native_input_rows) != 1:
        return ()
    row = dict(native_input_rows[0])
    if str(row.get("source", "")) == "x" and not relayout_rows:
        return ()
    source_node = str(row.get("source", ""))
    native_count = _layout_policy_native_input_ct_count(
        base_executor,
        compile_plan,
        node=str(node),
        native_input_rows=native_input_rows,
    )
    native_source_counts = dict(compile_plan.get("_native_physical_output_ct_counts", {}) or {})
    if (
        source_node
        and int(native_count) > 0
        and int(native_source_counts.get(source_node, -1)) == int(native_count)
        and not relayout_rows
    ):
        return ()
    layout = dict(row.get("selected_layout", {}))
    logical_count = max(1, int(layout.get("tile_count", 1) or 1))
    if relayout_rows or int(native_count) > 0:
        updated = dict(row)
        updated["relayout_reason"] = "native_halo_physical_source_stripe_relayout"
        updated["logical_tile_count"] = int(logical_count)
        updated["native_input_ct_count"] = int(native_count)
        return (updated,)
    return ()


def _flat_nchw_index(n: int, c: int, h: int, w: int, shape: torch.Size) -> int:
    return int(((int(n) * int(shape[1]) + int(c)) * int(shape[2]) + int(h)) * int(shape[3]) + int(w))


class LayoutPolicyRelayoutKernel:
    def __init__(self, *, edge_row: dict[str, Any], node: str, direction: str, index: int) -> None:
        direction = str(direction)
        if direction not in {"compact_to_halo", "halo_to_compact", "layout_to_layout"}:
            raise ValueError(f"unknown layout relayout direction {direction!r}")
        self.edge_row = dict(edge_row)
        self.node = str(node)
        self.direction = str(direction)
        self.index = int(index)
        self.transform_ids: dict[tuple[int, int], int] = {}
        selected_layout = dict(self.edge_row.get("selected_layout", {}) or {})
        target_layout = dict(self.edge_row.get("target_layout", {}) or selected_layout)
        source_layout = dict(self.edge_row.get("source_layout", {}) or {})
        if not source_layout:
            if str(direction) == "halo_to_compact":
                source_layout = dict(selected_layout)
            else:
                source_layout = {**target_layout, "alpha": 0, "beta": 0}
        if str(direction) == "halo_to_compact" and not dict(self.edge_row.get("target_layout", {}) or {}):
            target_layout = {**selected_layout, "alpha": 0, "beta": 0}
        if not target_layout:
            target_layout = dict(selected_layout)
        self.source_layout = dict(source_layout)
        self.target_layout = dict(target_layout)
        self.output_shape = torch.Size(int(value) for value in self.edge_row.get("shape", []))
        self.fhe_output_shape = _layout_policy_on_shape(self.edge_row, self.target_layout)
        self.input_on_shape = _layout_policy_on_shape(self.edge_row, self.source_layout)
        self.output_on_shape = self.fhe_output_shape
        self.level: int | None = None
        self.bsgs_ratio = 2.0
        self.output_scale = 1.0
        self.output_bias = 0.0
        self._bias_ptxt_cache: dict[tuple[int, int], Any] = {}
        self.name = (
            f"layout_policy_relayout_{self.node}_{self.index}_"
            f"{self.direction}_{str(self.edge_row.get('source', 'src'))}_to_{str(self.edge_row.get('target', 'dst'))}"
        )
        self.diagonals: dict[tuple[int, int], dict[int, torch.Tensor]] = {}

    def operation_estimate(self) -> dict[str, int | str]:
        source_layout = dict(self.source_layout)
        target_layout = dict(self.target_layout)
        alpha = max(0, int(target_layout.get("alpha", 0)))
        beta = max(0, int(target_layout.get("beta", 0)))
        tile_count = max(
            1,
            int(
                target_layout.get(
                    "tile_count",
                    math.ceil(int(_layout_policy_on_shape(self.edge_row, target_layout).numel()) / 32768.0),
                )
            ),
        )
        source_tile_count = max(
            1,
            int(
                source_layout.get(
                    "tile_count",
                    math.ceil(int(_layout_policy_on_shape(self.edge_row, source_layout).numel()) / 32768.0),
                )
            ),
        )
        halo_sides = int(tile_count) * (int(alpha > 0) + int(beta > 0))
        source_has_halo = bool(int(source_layout.get("alpha", 0)) > 0 or int(source_layout.get("beta", 0)) > 0)
        target_has_halo = bool(int(target_layout.get("alpha", 0)) > 0 or int(target_layout.get("beta", 0)) > 0)
        same_physical = bool(
            int(source_layout.get("alpha", 0)) == int(target_layout.get("alpha", 0))
            and int(source_layout.get("beta", 0)) == int(target_layout.get("beta", 0))
            and int(source_layout.get("gap", 1)) == int(target_layout.get("gap", 1))
            and int(source_layout.get("tile_count", tile_count)) == int(target_layout.get("tile_count", tile_count))
        )
        if bool(target_has_halo) and not bool(source_has_halo):
            return {
                "kind": "height_stripe_halo_fill",
                "rotation_count": int(halo_sides),
                "mask_mult_count": int(halo_sides),
                "sparse_lt_count": 0,
            }
        if bool(source_has_halo) and not bool(target_has_halo):
            return {
                "kind": "halo_trim",
                "rotation_count": 0,
                "mask_mult_count": int(max(1, source_tile_count)),
                "sparse_lt_count": 0,
            }
        if bool(same_physical):
            return {
                "kind": "layout_identity",
                "rotation_count": 0,
                "mask_mult_count": 0,
                "sparse_lt_count": 0,
            }
        return {
            "kind": "sparse_relayout_lt",
            "rotation_count": int(halo_sides),
            "mask_mult_count": int(halo_sides if bool(target_has_halo) else max(1, tile_count)),
            "sparse_lt_count": 1,
        }

    def _iter_mappings(self, slots: int):
        del slots
        source = _layout_policy_on_shape(self.edge_row, self.source_layout)
        target = _layout_policy_on_shape(self.edge_row, self.target_layout)
        source_gap = max(1, int(self.source_layout.get("gap", 1)))
        target_gap = max(1, int(self.target_layout.get("gap", 1)))
        if int(source_gap) != int(target_gap):
            raise ValueError(
                "layout-policy relayout only supports equal source/target packing gaps, "
                f"got source gap {int(source_gap)} and target gap {int(target_gap)}"
            )
        gap = int(target_gap)
        source_offset = int(max(0, int(self.source_layout.get("alpha", 0))) * gap)
        target_offset = int(max(0, int(self.target_layout.get("alpha", 0))) * gap)
        target_beta_rows = int(max(0, int(self.target_layout.get("beta", 0))) * gap)
        clear_shape = [int(value) for value in self.edge_row.get("shape", [])]
        if len(clear_shape) != 4:
            raise ValueError(f"layout-policy relayout edge has invalid shape: {clear_shape}")
        core_rows = int(clear_shape[2] * gap)
        if int(source[0]) != int(target[0]) or int(source[1]) != int(target[1]) or int(source[3]) != int(target[3]):
            raise ValueError(
                "layout-policy relayout requires matching batch/channel/width physical axes, "
                f"got source {tuple(int(v) for v in source)} and target {tuple(int(v) for v in target)}"
            )
        for n in range(int(target[0])):
            for c in range(int(target[1])):
                for target_h in range(int(target[2])):
                    if int(target_h) < int(target_offset):
                        source_core_h = min(int(target_h), int(core_rows) - 1)
                    elif int(target_h) >= int(target_offset + core_rows):
                        bottom_h = int(target_h) - int(target_offset + core_rows)
                        source_core_h = min(
                            max(int(core_rows - target_beta_rows + bottom_h), 0),
                            int(core_rows) - 1,
                        )
                    else:
                        source_core_h = int(target_h) - int(target_offset)
                    source_h = min(
                        max(int(source_offset + source_core_h), 0),
                        int(source[2]) - 1,
                    )
                    for w in range(int(target[3])):
                        source_index = _flat_nchw_index(int(n), int(c), int(source_h), int(w), source)
                        target_index = _flat_nchw_index(int(n), int(c), int(target_h), int(w), target)
                        yield int(source_index), int(target_index)

    def compile(self, scheme: Any, *, level: int) -> None:
        self.cleanup(getattr(scheme, "backend", None))
        slots = int(scheme.params.get_slots())
        diagonals: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
        for source_index, output_index in self._iter_mappings(int(slots)):
            input_block = int(source_index // int(slots))
            output_block = int(output_index // int(slots))
            source_local = int(source_index % int(slots))
            output_local = int(output_index % int(slots))
            diag_index = int((source_local - output_local) % int(slots))
            block = diagonals.setdefault((int(output_block), int(input_block)), {})
            diag = block.get(int(diag_index))
            if diag is None:
                diag = torch.zeros((int(slots),), dtype=torch.float32)
                block[int(diag_index)] = diag
            diag[int(output_local)] = float(self.output_scale)
        self.level = int(level)
        self.diagonals = diagonals
        self.transform_ids = {
            (int(row), int(col)): int(transform_id)
            for (row, col), transform_id in scheme.lt_evaluator.generate_transforms(self).items()
        }

    def _bias_plaintext(self, output_ct: Any) -> Any | None:
        bias = float(getattr(self, "output_bias", 0.0) or 0.0)
        if float(bias) == 0.0:
            return None
        level = int(output_ct.level())
        scale = max(1, int(output_ct.scale()))
        key = (int(level), int(scale))
        cached = self._bias_ptxt_cache.get(key)
        if cached is not None:
            return cached
        values = torch.full(tuple(int(value) for value in self.fhe_output_shape), float(bias), dtype=torch.float32)
        ptxt = output_ct.scheme.encode(values, level=int(level), scale=int(scale))
        self._bias_ptxt_cache[key] = ptxt
        return ptxt

    def apply(self, source_ct: Any) -> Any:
        if not self.transform_ids:
            raise RuntimeError(f"layout-policy relayout kernel {self.name} has not been compiled")
        out = source_ct.scheme.lt_evaluator.evaluate_transforms(self, source_ct)
        bias_ptxt = self._bias_plaintext(out)
        if bias_ptxt is not None:
            out = _add_plaintext_for_add(out, bias_ptxt)
        return out

    def cleanup(self, backend: Any | None) -> None:
        delete = getattr(backend, "DeleteLinearTransform", None)
        if callable(delete):
            for value in self.transform_ids.values():
                try:
                    delete(int(value))
                except Exception:
                    pass
        self.transform_ids = {}
        self.diagonals = {}
        for ptxt in self._bias_ptxt_cache.values():
            release = getattr(ptxt, "release", None)
            if callable(release):
                release()
        self._bias_ptxt_cache = {}


class LayoutPolicyEncryptedModuleRuntimeExecutor:
    def __init__(self, *, module: Any, output_node_id: str, compile_plan: dict[str, Any]) -> None:
        self.module = module
        self.output_node_id = str(output_node_id)
        self.compile_plan = dict(compile_plan)
        self.compile_count = 0
        self.execute_count = 0
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.last_runtime_io: dict[str, Any] = {}
        self.last_runtime_timing: dict[str, float] = {
            "compile_dense_module_s": 0.0,
            "evaluate_dense_module_s": 0.0,
        }
        self._compiled_backend = ""
        self._compiled_transform_ids: dict[Any, int] = {}
        self._compiled_output_rotations = 0
        self._compiled_on_bias_ptxt: Any | None = None
        self._compiled_transform_backend: Any | None = None
        self.relayout_rows = _layout_policy_incoming_relayout_rows(self.compile_plan, node=self.output_node_id)
        self.relayout_kernels: list[LayoutPolicyRelayoutKernel] = []

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        backend = str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())
        return bool(backend in {"python", "lattigo"})

    def _backend_name(self, scheme: Any | None) -> str:
        return str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())

    def _restore_module_attrs(self, attrs: dict[str, Any]) -> None:
        module = self.module
        for name, value in attrs.items():
            if value is _MISSING:
                try:
                    delattr(module, name)
                except AttributeError:
                    pass
            else:
                setattr(module, name, value)

    def _capture_module_attrs(self, names: tuple[str, ...]) -> dict[str, Any]:
        module = self.module
        return {name: getattr(module, name, _MISSING) for name in names}

    def compile(self, scheme: Any) -> None:
        if not self.supports_scheme(scheme):
            raise RuntimeError("layout-policy encrypted-module executor supports only Python and Lattigo backends")
        backend = self._backend_name(scheme)
        if self._compiled_backend == str(backend) and self._compiled_transform_ids:
            return
        if self._compiled_transform_ids:
            self.cleanup(self._compiled_transform_backend)
        module = self.module
        saved = self._capture_module_attrs(
            (
                "level",
                "depth",
                "region_first_skip_dense_pack",
                "transform_ids",
                "output_rotations",
                "on_bias_ptxt",
                "_transform_backend",
            )
        )
        started = time.time()
        try:
            self.relayout_kernels = []
            current_level = int(self.assigned_level) if self.assigned_level is not None else int(getattr(module, "level", len(scheme.params.get_logq()) - 1))
            relayout_depth = int(2 * len(self.relayout_rows))
            if self.assigned_depth is not None:
                module_depth = max(0, int(self.assigned_depth) - int(relayout_depth))
            else:
                module_depth = max(1, int(getattr(module, "depth", 1) or 1))
            required_level = int(relayout_depth + module_depth)
            if int(current_level) < int(required_level):
                raise RuntimeError(
                    "layout-policy encrypted-module executor needs input level "
                    f">= {int(required_level)} for {int(relayout_depth)} relayout LT depth "
                    f"+ {int(module_depth)} module LT depth, got {int(current_level)}"
                )
            for row_index, row in enumerate(self.relayout_rows):
                pad_kernel = LayoutPolicyRelayoutKernel(
                    edge_row=row,
                    node=self.output_node_id,
                    direction="compact_to_halo",
                    index=int(row_index * 2),
                )
                pad_kernel.compile(scheme, level=int(current_level))
                self.relayout_kernels.append(pad_kernel)
                current_level = max(0, int(current_level - 1))
                trim_kernel = LayoutPolicyRelayoutKernel(
                    edge_row=row,
                    node=self.output_node_id,
                    direction="halo_to_compact",
                    index=int(row_index * 2 + 1),
                )
                trim_kernel.compile(scheme, level=int(current_level))
                self.relayout_kernels.append(trim_kernel)
                current_level = max(0, int(current_level - 1))
            module.level = int(current_level)
            module.depth = int(module_depth)
            module.region_first_skip_dense_pack = False
            plan = _layout_policy_node_plan(self.compile_plan, node=self.output_node_id)
            module.generate_diagonals(last=bool(plan["is_last_linear"]))
            module.compile()
            self._compiled_transform_ids = {
                key: int(value)
                for key, value in dict(getattr(module, "transform_ids", {}) or {}).items()
            }
            self._compiled_output_rotations = int(getattr(module, "output_rotations", 0) or 0)
            self._compiled_on_bias_ptxt = getattr(module, "on_bias_ptxt", None)
            self._compiled_transform_backend = getattr(module, "_transform_backend", None)
            self._compiled_backend = str(backend)
            self.compile_count += 1
            self.last_runtime_timing["compile_dense_module_s"] = float(time.time() - started)
        finally:
            self._restore_module_attrs(saved)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = getattr(source_ct, "scheme", None)
        if not self.supports_scheme(scheme):
            raise RuntimeError("layout-policy encrypted-module executor received an unsupported backend ciphertext")
        self.compile(scheme)
        if not self._compiled_transform_ids:
            raise RuntimeError(f"layout-policy encrypted-module executor for {self.output_node_id} has no transforms")
        module = self.module
        working_ct = source_ct
        temp_cts: list[Any] = []
        relayout_started = time.time()
        for kernel in self.relayout_kernels:
            working_ct = kernel.apply(working_ct)
            temp_cts.append(working_ct)
        relayout_s = float(time.time() - relayout_started)
        saved = self._capture_module_attrs(
            (
                "region_runtime",
                "region_first_skip_dense_pack",
                "transform_ids",
                "output_rotations",
                "on_bias_ptxt",
                "_transform_backend",
            )
        )
        started = time.time()
        try:
            module.region_runtime = None
            module.region_first_skip_dense_pack = False
            module.transform_ids = dict(self._compiled_transform_ids)
            module.output_rotations = int(self._compiled_output_rotations)
            module.on_bias_ptxt = self._compiled_on_bias_ptxt
            module._transform_backend = self._compiled_transform_backend
            out = module(working_ct)
            self.execute_count += 1
            self.last_runtime_timing["evaluate_dense_module_s"] = float(time.time() - started)
            self.last_runtime_timing["relayout_s"] = float(relayout_s)
            relayout_ops = _layout_policy_relayout_operation_totals(self.relayout_kernels)
            self.last_runtime_io = {
                "policy": str(self.compile_plan.get("policy", "")),
                "runtime_lowering": "backend_encrypted_module",
                "backend": self._backend_name(scheme),
                "relayout_kernel": bool(self.relayout_kernels),
                "relayout_kernel_count": int(len(self.relayout_kernels)),
                "relayout_edge_count": int(len(self.relayout_rows)),
                "relayout_rotation_count": int(relayout_ops["rotation_count"]),
                "relayout_mask_mult_count": int(relayout_ops["mask_mult_count"]),
                "relayout_sparse_lt_count": int(relayout_ops["sparse_lt_count"]),
                "input_shape": [int(value) for value in getattr(module, "input_shape", ())],
                "output_shape": [int(value) for value in getattr(module, "output_shape", ())],
                "input_ciphertext_count": int(len(getattr(source_ct, "ids", ())) or 0),
                "output_ciphertext_count": int(len(getattr(out, "ids", ())) or 0),
                "transform_count": int(len(self._compiled_transform_ids)),
                "output_rotations": int(self._compiled_output_rotations),
            }
            return {self.output_node_id: out}
        finally:
            self._restore_module_attrs(saved)
            for temp in temp_cts:
                release = getattr(temp, "release", None)
                if callable(release):
                    release()

    def cleanup(self, backend: Any | None) -> None:
        target_backend = backend if backend is not None else self._compiled_transform_backend
        for kernel in self.relayout_kernels:
            kernel.cleanup(target_backend)
        self.relayout_kernels = []
        delete = getattr(target_backend, "DeleteLinearTransform", None)
        if callable(delete):
            for value in self._compiled_transform_ids.values():
                try:
                    delete(int(value))
                except Exception:
                    pass
        self._compiled_transform_ids = {}
        self._compiled_output_rotations = 0
        self._compiled_on_bias_ptxt = None
        self._compiled_transform_backend = None
        self._compiled_backend = ""


class LayoutPolicyProviderRuntimeExecutor:
    def __init__(self, *, base_executor: Any, output_node_id: str, compile_plan: dict[str, Any]) -> None:
        self.base_executor = base_executor
        self.output_node_id = str(output_node_id)
        self.compile_plan = dict(compile_plan)
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.compile_count = 0
        self.execute_count = 0
        self.relayout_rows = _layout_policy_incoming_relayout_rows(self.compile_plan, node=self.output_node_id)
        self.native_input_rows = _layout_policy_incoming_native_rows(self.compile_plan, node=self.output_node_id)
        self.compact_source_rows = _layout_policy_incoming_compact_source_rows(
            self.compile_plan,
            node=self.output_node_id,
        )
        self.compact_align_shared_rows = _layout_policy_incoming_compact_align_shared_rows(
            self.compile_plan,
            node=self.output_node_id,
        )
        self.output_relayout_rows = _layout_policy_output_relayout_rows(self.compile_plan, node=self.output_node_id)
        self.relayout_kernels: list[LayoutPolicyRelayoutKernel] = []
        self.output_relayout_kernels: list[LayoutPolicyRelayoutKernel] = []
        self.native_halo_input = _layout_policy_native_halo_input_supported(self.base_executor, self.native_input_rows)
        self.native_physical_relayout_rows = _layout_policy_native_physical_relayout_rows(
            self.base_executor,
            self.compile_plan,
            node=self.output_node_id,
            native_input_rows=self.native_input_rows,
            relayout_rows=self.relayout_rows,
        )
        self.native_physical_relayout_kernel: NativeHaloRelayoutKernel | None = None
        self.last_runtime_timing: dict[str, float] = {"relayout_s": 0.0}
        self.last_runtime_io: dict[str, Any] = {}
        self._compiled = False
        self._compiled_backend: Any | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_executor, name)

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        base_supports = getattr(self.base_executor, "supports_scheme", lambda _scheme: True)
        if not bool(base_supports(scheme)):
            return False
        if not self.relayout_rows and not self.output_relayout_rows and not self.native_physical_relayout_rows:
            return True
        if bool(self.native_halo_input):
            backend = str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())
            return bool(backend in {"python", "lattigo"})
        backend = str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())
        return bool(backend in {"python", "lattigo"})

    def _backend_name(self, scheme: Any | None) -> str:
        return str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())

    def _runtime_lowering_label(self) -> str:
        if bool(self.native_halo_input):
            return "provider_executable+native_halo_layout"
        if _layout_policy_native_output_row(self.compile_plan, node=self.output_node_id) is not None:
            return "provider_executable+native_halo_output_layout"
        if self.compact_align_shared_rows:
            return "provider_executable+compact_align_shared"
        if not self.relayout_rows and not self.output_relayout_rows:
            return "provider_executable+compact_layout"
        return "provider_executable+relayout_lt"

    def _bootstrap_prescale_fusion_spec(self) -> dict[str, float] | None:
        fusion = self.__dict__.get("_bootstrap_prescale_fusion", None)
        if not fusion:
            return None
        return {
            "scale": float(dict(fusion).get("scale", 1.0)),
            "bias": float(dict(fusion).get("bias", 0.0)),
        }

    def bootstrap_prescale_fusion_capable(self) -> bool:
        if self.output_relayout_rows:
            return True
        module = getattr(self.base_executor, "module", None)
        return bool(module is not None and hasattr(module, "on_weight") and hasattr(module, "on_bias"))

    def _native_halo_module_attrs(self) -> dict[str, Any]:
        return _layout_policy_native_module_attrs(
            self.compile_plan,
            node=self.output_node_id,
            base_executor=self.base_executor,
            native_input_rows=self.native_input_rows,
            compact_input_rows=tuple([*self.compact_source_rows, *self.compact_align_shared_rows]),
        )

    def _bootstrap_prescale_module_attrs(self) -> dict[str, Any]:
        fusion = self._bootstrap_prescale_fusion_spec()
        if fusion is None or self.output_relayout_rows:
            return {}
        module = getattr(self.base_executor, "module", None)
        if module is None or not hasattr(module, "on_weight") or not hasattr(module, "on_bias"):
            return {}
        scale = float(fusion["scale"])
        bias = float(fusion["bias"])
        return {
            "on_weight": getattr(module, "on_weight") * float(scale),
            "on_bias": getattr(module, "on_bias") * float(scale) + float(bias),
        }

    def _with_base_module_attrs(self, callback: Any) -> Any:
        attrs = {
            **self._native_halo_module_attrs(),
            **self._bootstrap_prescale_module_attrs(),
        }
        module = getattr(self.base_executor, "module", None)
        if not attrs or module is None:
            return callback()
        saved = {name: getattr(module, name, _MISSING) for name in attrs}
        base_saved = {
            name: getattr(self.base_executor, name, _MISSING)
            for name in attrs
            if hasattr(self.base_executor, name)
        }
        try:
            for name, value in attrs.items():
                setattr(module, name, value)
                if name in base_saved:
                    setattr(self.base_executor, name, value)
            return callback()
        finally:
            for name, value in saved.items():
                if value is _MISSING:
                    try:
                        delattr(module, name)
                    except AttributeError:
                        pass
                else:
                    setattr(module, name, value)
            for name, value in base_saved.items():
                if value is _MISSING:
                    try:
                        delattr(self.base_executor, name)
                    except AttributeError:
                        pass
                else:
                    setattr(self.base_executor, name, value)

    def _compile_base_with_optional_native_halo(self, scheme: Any) -> None:
        compile_base = getattr(self.base_executor, "compile", None)
        if not callable(compile_base):
            return
        self._with_base_module_attrs(lambda: compile_base(scheme))

    def _native_halo_plan(self) -> Any:
        plan = _layout_policy_native_halo_plan(
            self.base_executor,
            self.compile_plan,
            node=self.output_node_id,
            native_input_rows=self.native_input_rows,
        )
        if plan is None:
            raise RuntimeError(f"{self.output_node_id} has no native halo plan for physical relayout")
        return plan

    def runtime_fhe_output_shape(self) -> Any:
        if self.output_relayout_rows:
            row = dict(self.output_relayout_rows[-1])
            layout = dict(row.get("target_layout", row.get("selected_layout", {})) or {})
            return _layout_policy_on_shape(row, layout)
        attrs = self._native_halo_module_attrs()
        if "fhe_output_shape" in attrs:
            return attrs["fhe_output_shape"]
        if bool(self.native_halo_input) and bool(getattr(self.base_executor, "native_halo_output_capable", False)):
            get_native_shape = getattr(self.base_executor, "runtime_native_fhe_output_shape", None)
            if callable(get_native_shape):
                return self._with_base_module_attrs(lambda: get_native_shape())
        module = getattr(self.base_executor, "module", None)
        if module is not None:
            return getattr(module, "fhe_output_shape", None)
        return getattr(self.base_executor, "fhe_output_shape", None)

    def compile(self, scheme: Any) -> None:
        if self._compiled:
            return
        if not self.supports_scheme(scheme):
            raise RuntimeError("layout-policy provider executor received an unsupported backend")
        current_level = int(self.assigned_level) if self.assigned_level is not None else int(len(scheme.params.get_logq()) - 1)
        native_physical_relayout_depth = int(len(self.native_physical_relayout_rows))
        input_relayout_depth = int(
            (len(self.relayout_rows) if bool(self.native_halo_input) else 2 * len(self.relayout_rows))
            + int(native_physical_relayout_depth)
        )
        output_relayout_depth = int(len(self.output_relayout_rows))
        relayout_depth = int(input_relayout_depth + output_relayout_depth)
        if self.assigned_depth is not None:
            base_depth = max(0, int(self.assigned_depth) - int(relayout_depth))
        else:
            base_depth = max(1, int(getattr(self.base_executor, "assigned_depth", 1) or 1))
        required_level = int(relayout_depth + base_depth)
        if int(current_level) < int(required_level):
            raise RuntimeError(
                "layout-policy provider executor needs input level "
                f">= {int(required_level)} for {int(relayout_depth)} relayout LT depth "
                f"+ {int(base_depth)} provider depth, got {int(current_level)}"
            )
        self.relayout_kernels = []
        self.output_relayout_kernels = []
        self.native_physical_relayout_kernel = None
        for row_index, row in enumerate(self.relayout_rows):
            pad_kernel = LayoutPolicyRelayoutKernel(
                edge_row=row,
                node=self.output_node_id,
                direction="compact_to_halo",
                index=int(row_index * 2),
            )
            pad_kernel.compile(scheme, level=int(current_level))
            self.relayout_kernels.append(pad_kernel)
            current_level = max(0, int(current_level - 1))
            if bool(self.native_halo_input):
                continue
            trim_kernel = LayoutPolicyRelayoutKernel(
                edge_row=row,
                node=self.output_node_id,
                direction="halo_to_compact",
                index=int(row_index * 2 + 1),
            )
            trim_kernel.compile(scheme, level=int(current_level))
            self.relayout_kernels.append(trim_kernel)
            current_level = max(0, int(current_level - 1))
        if self.native_physical_relayout_rows:
            native_plan = self._native_halo_plan()
            slots = int(scheme.params.get_slots())
            native_kernel = NativeHaloRelayoutKernel(
                plan=native_plan,
                direction="compact_to_native",
                name=f"layout_policy_native_physical_relayout_{self.output_node_id}",
                output_shape=torch.Size([int(native_plan.input_ct_count), int(slots)]),
                fhe_output_shape=torch.Size([int(native_plan.input_ct_count), int(slots)]),
                source_layout=dict(self.native_physical_relayout_rows[0].get("source_layout", {}) or {}),
            )
            native_kernel.compile(scheme, level=int(current_level))
            self.native_physical_relayout_kernel = native_kernel
            current_level = max(0, int(current_level - 1))
        if hasattr(self.base_executor, "assigned_level"):
            self.base_executor.assigned_level = int(current_level)
        if hasattr(self.base_executor, "assigned_depth"):
            self.base_executor.assigned_depth = int(base_depth)
        self._compile_base_with_optional_native_halo(scheme)
        post_level = max(0, int(current_level) - int(base_depth))
        for row_index, row in enumerate(self.output_relayout_rows):
            post_kernel = LayoutPolicyRelayoutKernel(
                edge_row=row,
                node=self.output_node_id,
                direction="compact_to_halo",
                index=int(1000 + row_index),
            )
            fusion = self._bootstrap_prescale_fusion_spec()
            if fusion is not None and int(row_index) == int(len(self.output_relayout_rows) - 1):
                post_kernel.output_scale = float(fusion["scale"])
                post_kernel.output_bias = float(fusion["bias"])
            post_kernel.compile(scheme, level=int(post_level))
            self.output_relayout_kernels.append(post_kernel)
            post_level = max(0, int(post_level - 1))
        self.compile_count += 1
        self._compiled = True
        self._compiled_backend = getattr(scheme, "backend", None)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = getattr(source_ct, "scheme", None)
        if not self.supports_scheme(scheme):
            raise RuntimeError("layout-policy provider executor received an unsupported backend ciphertext")
        self.compile(scheme)
        working_ct = source_ct
        temp_cts: list[Any] = []
        relayout_started = time.time()
        for kernel in self.relayout_kernels:
            working_ct = kernel.apply(working_ct)
            temp_cts.append(working_ct)
        native_physical_relayout_applied = False
        if self.native_physical_relayout_kernel is not None:
            working_ct = self.native_physical_relayout_kernel.apply(working_ct)
            temp_cts.append(working_ct)
            native_physical_relayout_applied = True
        relayout_s = float(time.time() - relayout_started)
        try:
            outputs = dict(self._with_base_module_attrs(lambda: self.base_executor(working_ct)))
            output_relayout_s = 0.0
            if self.output_relayout_kernels:
                output_relayout_started = time.time()
                output_ct = outputs.get(self.output_node_id)
                if output_ct is None:
                    raise KeyError(f"layout-policy output relayout missing output {self.output_node_id!r}")
                for kernel in self.output_relayout_kernels:
                    next_ct = kernel.apply(output_ct)
                    release = getattr(output_ct, "release", None)
                    if callable(release):
                        release()
                    output_ct = next_ct
                outputs[self.output_node_id] = output_ct
                output_relayout_s = float(time.time() - output_relayout_started)
            self.execute_count += 1
            base_timing = dict(getattr(self.base_executor, "last_runtime_timing", {}) or {})
            base_io = dict(getattr(self.base_executor, "last_runtime_io", {}) or {})
            relayout_ops = _layout_policy_relayout_operation_totals(
                [
                    *self.relayout_kernels,
                    *([self.native_physical_relayout_kernel] if self.native_physical_relayout_kernel is not None else []),
                    *self.output_relayout_kernels,
                ]
            )
            self.last_runtime_timing = {
                **base_timing,
                "relayout_s": float(relayout_s + output_relayout_s),
            }
            self.last_runtime_io = {
                **base_io,
                "policy": str(self.compile_plan.get("policy", "")),
                "runtime_lowering": self._runtime_lowering_label(),
                "backend": self._backend_name(scheme),
                "provider_executor": type(self.base_executor).__name__,
                "native_halo_provider": bool(self.native_halo_input),
                "relayout_kernel": bool(
                    self.relayout_kernels
                    or self.native_physical_relayout_kernel is not None
                    or self.output_relayout_kernels
                ),
                "relayout_kernel_count": int(
                    len(self.relayout_kernels)
                    + int(self.native_physical_relayout_kernel is not None)
                    + len(self.output_relayout_kernels)
                ),
                "relayout_edge_count": int(len(self.relayout_rows)),
                "native_physical_relayout_edge_count": int(len(self.native_physical_relayout_rows)),
                "native_physical_relayout_applied": bool(native_physical_relayout_applied),
                "compact_align_shared_edge_count": int(len(self.compact_align_shared_rows)),
                "output_relayout_edge_count": int(len(self.output_relayout_rows)),
                "relayout_rotation_count": int(relayout_ops["rotation_count"]),
                "relayout_mask_mult_count": int(relayout_ops["mask_mult_count"]),
                "relayout_sparse_lt_count": int(relayout_ops["sparse_lt_count"]),
                "bootstrap_prescale_fused": bool(self._bootstrap_prescale_fusion_spec() is not None),
            }
            return outputs
        finally:
            for temp in temp_cts:
                release = getattr(temp, "release", None)
                if callable(release):
                    release()

    def cleanup(self, backend: Any | None) -> None:
        target_backend = backend if backend is not None else self._compiled_backend
        for kernel in [*self.relayout_kernels, *self.output_relayout_kernels]:
            kernel.cleanup(target_backend)
        if self.native_physical_relayout_kernel is not None:
            self.native_physical_relayout_kernel.cleanup(target_backend)
        self.relayout_kernels = []
        self.output_relayout_kernels = []
        self.native_physical_relayout_kernel = None
        cleanup_base = getattr(self.base_executor, "cleanup", None)
        if callable(cleanup_base) and target_backend is not None:
            cleanup_base(target_backend)
        self._compiled = False
        self._compiled_backend = None

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        load = getattr(self.base_executor, "load_compile_cache_metadata", None)
        if callable(load):
            load(dict(metadata or {}))

    def compile_cache_metadata(self) -> dict[str, Any]:
        metadata = {}
        get_metadata = getattr(self.base_executor, "compile_cache_metadata", None)
        if callable(get_metadata):
            metadata = dict(self._with_base_module_attrs(lambda: get_metadata()))
        metadata["layout_policy_wrapper"] = {
            "policy": str(self.compile_plan.get("policy", "")),
            "relayout_edge_count": int(len(self.relayout_rows)),
            "native_physical_relayout_edge_count": int(len(self.native_physical_relayout_rows)),
            "compact_align_shared_edge_count": int(len(self.compact_align_shared_rows)),
            "output_relayout_edge_count": int(len(self.output_relayout_rows)),
            "relayout_kernel_count": int(
                (len(self.relayout_rows) if bool(self.native_halo_input) else 2 * len(self.relayout_rows))
                + len(self.native_physical_relayout_rows)
                + len(self.output_relayout_rows)
            ),
            "relayout_rotation_count": int(
                _layout_policy_relayout_operation_totals(
                    [
                        *self.relayout_kernels,
                        *(
                            [self.native_physical_relayout_kernel]
                            if self.native_physical_relayout_kernel is not None
                            else []
                        ),
                        *self.output_relayout_kernels,
                    ]
                )["rotation_count"]
            ),
            "relayout_mask_mult_count": int(
                _layout_policy_relayout_operation_totals(
                    [
                        *self.relayout_kernels,
                        *(
                            [self.native_physical_relayout_kernel]
                            if self.native_physical_relayout_kernel is not None
                            else []
                        ),
                        *self.output_relayout_kernels,
                    ]
                )["mask_mult_count"]
            ),
            "base_executor": type(self.base_executor).__name__,
            "runtime_lowering": self._runtime_lowering_label(),
            "native_halo_provider": bool(self.native_halo_input),
            "bootstrap_prescale_fused": bool(self._bootstrap_prescale_fusion_spec() is not None),
        }
        if self.native_physical_relayout_kernel is not None:
            metadata["layout_policy_wrapper"]["native_physical_input_relayout"] = (
                self.native_physical_relayout_kernel.to_metadata()
            )
        return metadata


def _layout_policy_add_rows(compile_plan: dict[str, Any], *, node: str) -> tuple[dict[str, Any], ...]:
    if not _layout_policy_relayout_enabled():
        return ()
    rows = []
    for row in compile_plan.get("edge_layouts", []):
        if str(row.get("target")) != str(node):
            continue
        if str(row.get("op_kind", "")) != "add":
            continue
        rows.append(dict(row))
    return tuple(rows)


def _layout_policy_relayout_direction(row: dict[str, Any]) -> str:
    source_layout = dict(row.get("source_layout", {}) or {})
    target_layout = dict(row.get("target_layout", row.get("selected_layout", {})) or {})
    source_has_halo = bool(int(source_layout.get("alpha", 0)) > 0 or int(source_layout.get("beta", 0)) > 0)
    target_has_halo = bool(int(target_layout.get("alpha", 0)) > 0 or int(target_layout.get("beta", 0)) > 0)
    if bool(target_has_halo) and not bool(source_has_halo):
        return "compact_to_halo"
    if bool(source_has_halo) and not bool(target_has_halo):
        return "halo_to_compact"
    return "layout_to_layout"


class LayoutPolicyAddRuntimeExecutor:
    def __init__(
        self,
        *,
        node: str,
        compile_plan: dict[str, Any],
        input_sources: tuple[str, ...],
    ) -> None:
        self.node = str(node)
        self.compile_plan = dict(compile_plan)
        self.input_sources = tuple(str(value) for value in input_sources)
        rows_by_source = {
            str(row.get("source", "")): dict(row)
            for row in _layout_policy_add_rows(self.compile_plan, node=self.node)
        }
        self.input_rows = tuple(
            rows_by_source[str(source)]
            for source in self.input_sources
            if str(source) in rows_by_source
        )
        if len(self.input_rows) != len(self.input_sources):
            missing = [str(source) for source in self.input_sources if str(source) not in rows_by_source]
            raise ValueError(f"layout-policy Add runtime for {self.node} is missing input rows for {missing}")
        target_keys = {
            tuple(
                int(dict(row.get("selected_layout", {})).get(name, 0 if name in {"alpha", "beta"} else 1))
                for name in ("alpha", "beta", "gap")
            )
            for row in self.input_rows
        }
        if len(target_keys) > 1:
            raise ValueError(f"layout-policy Add runtime for {self.node} received non-aligned input layouts")
        self.relayout_rows = tuple(row for row in self.input_rows if bool(row.get("relayout", False)))
        self.relayout_kernels: dict[int, LayoutPolicyRelayoutKernel] = {}
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.compile_count = 0
        self.execute_count = 0
        self.last_runtime_timing: dict[str, float] = {"relayout_s": 0.0, "add_s": 0.0}
        self.last_runtime_io: dict[str, Any] = {}
        self._compiled = False
        self._compiled_backend: Any | None = None

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        backend = str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())
        return bool(backend in {"python", "lattigo"})

    def _backend_name(self, scheme: Any | None) -> str:
        return str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())

    def _bootstrap_prescale_fusion_spec(self) -> dict[str, float] | None:
        fusion = self.__dict__.get("_bootstrap_prescale_fusion", None)
        if not fusion:
            return None
        return {
            "scale": float(dict(fusion).get("scale", 1.0)),
            "bias": float(dict(fusion).get("bias", 0.0)),
        }

    def bootstrap_prescale_fusion_capable(self) -> bool:
        return bool(self.input_rows and all(bool(row.get("relayout", False)) for row in self.input_rows))

    def runtime_fhe_output_shape(self) -> Any:
        if not self.input_rows:
            return None
        row = dict(self.input_rows[0])
        layout = dict(row.get("target_layout", row.get("selected_layout", {})) or {})
        return _layout_policy_on_shape(row, layout)

    def _bias_plaintext(self, output_ct: Any) -> Any | None:
        fusion = self._bootstrap_prescale_fusion_spec()
        if fusion is None:
            return None
        bias = float(fusion["bias"])
        if float(bias) == 0.0:
            return None
        shape = self.runtime_fhe_output_shape()
        if shape is None:
            return None
        values = torch.full(tuple(int(value) for value in shape), float(bias), dtype=torch.float32)
        return output_ct.scheme.encode(
            values,
            level=int(output_ct.level()),
            scale=max(1, int(output_ct.scale())),
        )

    def compile(self, scheme: Any) -> None:
        if self._compiled:
            return
        if not self.supports_scheme(scheme):
            raise RuntimeError("layout-policy Add runtime supports only Python and Lattigo backends")
        current_level = int(self.assigned_level) if self.assigned_level is not None else int(len(scheme.params.get_logq()) - 1)
        self.relayout_kernels = {}
        for index, row in enumerate(self.input_rows):
            if not bool(row.get("relayout", False)):
                continue
            kernel = LayoutPolicyRelayoutKernel(
                edge_row=row,
                node=self.node,
                direction=_layout_policy_relayout_direction(row),
                index=int(index),
            )
            fusion = self._bootstrap_prescale_fusion_spec()
            if fusion is not None and self.bootstrap_prescale_fusion_capable():
                kernel.output_scale = float(fusion["scale"])
            kernel.compile(scheme, level=int(current_level))
            self.relayout_kernels[int(index)] = kernel
        self.compile_count += 1
        self._compiled = True
        self._compiled_backend = getattr(scheme, "backend", None)

    def __call__(self, left: Any, right: Any) -> Any:
        inputs = [left, right]
        if len(inputs) != len(self.input_rows):
            raise ValueError(f"layout-policy Add runtime for {self.node} expects {len(self.input_rows)} inputs")
        scheme = getattr(left, "scheme", None)
        if not self.supports_scheme(scheme):
            return left + right
        self.compile(scheme)
        temp_cts: list[Any] = []
        relayout_started = time.time()
        working: list[Any] = []
        for index, source_ct in enumerate(inputs):
            kernel = self.relayout_kernels.get(int(index))
            if kernel is None:
                working.append(source_ct)
                continue
            out = kernel.apply(source_ct)
            temp_cts.append(out)
            working.append(out)
        relayout_s = float(time.time() - relayout_started)
        add_started = time.time()
        lhs, rhs = _align_ciphertexts_for_add(working[0], working[1])
        out = lhs + rhs
        if self.bootstrap_prescale_fusion_capable():
            bias_ptxt = self._bias_plaintext(out)
            if bias_ptxt is not None:
                out = _add_plaintext_for_add(out, bias_ptxt)
                release = getattr(bias_ptxt, "release", None)
                if callable(release):
                    release()
        add_s = float(time.time() - add_started)
        for temp in temp_cts:
            release = getattr(temp, "release", None)
            if callable(release):
                release()
        relayout_ops = _layout_policy_relayout_operation_totals(self.relayout_kernels.values())
        self.execute_count += 1
        self.last_runtime_timing = {
            "relayout_s": float(relayout_s),
            "add_s": float(add_s),
        }
        self.last_runtime_io = {
            "policy": str(self.compile_plan.get("policy", "")),
            "runtime_lowering": "layout_policy_add_join",
            "backend": self._backend_name(scheme),
            "relayout_edge_count": int(len(self.relayout_rows)),
            "relayout_kernel_count": int(len(self.relayout_kernels)),
            "relayout_rotation_count": int(relayout_ops["rotation_count"]),
            "relayout_mask_mult_count": int(relayout_ops["mask_mult_count"]),
            "relayout_sparse_lt_count": int(relayout_ops["sparse_lt_count"]),
            "input_ciphertext_counts": [int(len(getattr(value, "ids", ())) or 0) for value in inputs],
            "output_ciphertext_count": int(len(getattr(out, "ids", ())) or 0),
            "bootstrap_prescale_fused": bool(
                self._bootstrap_prescale_fusion_spec() is not None and self.bootstrap_prescale_fusion_capable()
            ),
        }
        return out

    def cleanup(self, backend: Any | None) -> None:
        target_backend = backend if backend is not None else self._compiled_backend
        for kernel in self.relayout_kernels.values():
            kernel.cleanup(target_backend)
        self.relayout_kernels = {}
        self._compiled = False
        self._compiled_backend = None


class _Missing:
    pass


_MISSING = _Missing()


def _layout_policy_compile_plan_groups(dag: Any, compile_plan: dict[str, Any]) -> tuple[RegionFirstRuntimeGroup, ...]:
    policy = str(compile_plan.get("policy", ""))
    groups: list[RegionFirstRuntimeGroup] = []
    for node in dag.topological_sort():
        module = dag.nodes[node].get("module")
        stage = _layout_policy_stage_for_module(module)
        if not stage:
            continue
        relayout_depth = int(2 * len(_layout_policy_incoming_relayout_rows(compile_plan, node=str(node))))
        groups.append(
            RegionFirstRuntimeGroup(
                region_id=f"u22_layout_policy_{policy}_{str(node)}",
                network="U22",
                stage=str(stage),
                module_prefix=str(node),
                conv_nodes=(str(node),),
                strategy=f"u22_layout_policy_{policy}_compile_plan",
                materializer="backend_encrypted_module",
                depth=int(1 + relayout_depth),
                solver_depth=int(1 + relayout_depth),
                boundary_actions=(
                    "layout_policy_compile_plan",
                    f"layout_policy_{policy}",
                    "backend_encrypted_module",
                    f"relayout_kernel_depth_{int(relayout_depth)}",
                ),
                expected_stats={},
                executable=True,
                fallback_reason="",
                output_node_ids=(str(node),),
                executor=LayoutPolicyEncryptedModuleRuntimeExecutor(
                    module=module,
                    output_node_id=str(node),
                    compile_plan=compile_plan,
                ),
                plan=_layout_policy_node_plan(compile_plan, node=str(node)),
                fused_weight_count=0,
            )
        )
    return tuple(groups)


def _layout_policy_provider_plan(group: RegionFirstRuntimeGroup, compile_plan: dict[str, Any]) -> dict[str, Any]:
    plan = _layout_policy_node_plan(compile_plan, node=str(group.module_prefix))
    relayout_rows = _layout_policy_incoming_relayout_rows(compile_plan, node=str(group.module_prefix))
    native_rows = _layout_policy_incoming_native_rows(compile_plan, node=str(group.module_prefix))
    compact_align_shared_rows = _layout_policy_incoming_compact_align_shared_rows(
        compile_plan,
        node=str(group.module_prefix),
    )
    output_relayout_rows = _layout_policy_output_relayout_rows(compile_plan, node=str(group.module_prefix))
    native_halo = _layout_policy_native_halo_input_supported(group.executor, native_rows)
    native_output_row = _layout_policy_native_output_row(compile_plan, node=str(group.module_prefix))
    native_physical_relayout_rows = _layout_policy_native_physical_relayout_rows(
        group.executor,
        compile_plan,
        node=str(group.module_prefix),
        native_input_rows=native_rows,
        relayout_rows=relayout_rows,
    )
    if bool(native_halo):
        plan["runtime_lowering"] = "provider_executable+native_halo_layout"
    elif native_output_row is not None:
        plan["runtime_lowering"] = "provider_executable+native_halo_output_layout"
    elif compact_align_shared_rows:
        plan["runtime_lowering"] = "provider_executable+compact_align_shared"
    elif not relayout_rows and not output_relayout_rows:
        plan["runtime_lowering"] = "provider_executable+compact_layout"
    else:
        plan["runtime_lowering"] = "provider_executable+relayout_lt"
    plan["native_halo_provider"] = bool(native_halo)
    plan["native_halo_output_provider"] = bool(native_output_row is not None)
    plan["relayout_edge_count"] = int(len(relayout_rows))
    plan["native_physical_relayout_edge_count"] = int(len(native_physical_relayout_rows))
    plan["compact_align_shared_edge_count"] = int(len(compact_align_shared_rows))
    plan["output_relayout_edge_count"] = int(len(output_relayout_rows))
    plan["provider_stage"] = str(group.stage)
    plan["provider_materializer"] = str(group.materializer)
    plan["provider_strategy"] = str(group.strategy)
    return plan


def _layout_policy_wrap_provider_group(
    group: RegionFirstRuntimeGroup,
    *,
    compile_plan: dict[str, Any],
) -> RegionFirstRuntimeGroup:
    base_executor = group.executor
    plan_group = group
    relayout_rows = _layout_policy_incoming_relayout_rows(compile_plan, node=str(group.module_prefix))
    native_rows = _layout_policy_incoming_native_rows(compile_plan, node=str(group.module_prefix))
    output_relayout_rows = _layout_policy_output_relayout_rows(compile_plan, node=str(group.module_prefix))
    native_halo = _layout_policy_native_halo_input_supported(base_executor, native_rows)
    native_physical_relayout_rows = _layout_policy_native_physical_relayout_rows(
        base_executor,
        compile_plan,
        node=str(group.module_prefix),
        native_input_rows=native_rows,
        relayout_rows=relayout_rows,
    )
    relayout_depth = int(
        (len(relayout_rows) if bool(native_halo) else 2 * len(relayout_rows))
        + len(native_physical_relayout_rows)
        + len(output_relayout_rows)
    )
    policy = str(compile_plan.get("policy", ""))
    base_depth = int(group.effective_depth())
    executor = (
        LayoutPolicyProviderRuntimeExecutor(
            base_executor=base_executor,
            output_node_id=str(group.module_prefix),
            compile_plan=compile_plan,
        )
        if base_executor is not None
        else None
    )
    return replace(
        group,
        region_id=f"u22_layout_policy_{policy}_{str(group.region_id)}",
        strategy=f"{str(group.strategy)}_layout_policy_{policy}",
        materializer=f"{str(group.materializer)}_layout_policy_relayout",
        depth=int(base_depth + relayout_depth),
        solver_depth=int(base_depth + relayout_depth),
        boundary_actions=(
            *tuple(str(value) for value in group.boundary_actions),
            "layout_policy_compile_plan",
            f"layout_policy_{policy}",
            f"relayout_kernel_depth_{int(relayout_depth)}",
        ),
        executor=executor,
        plan=_layout_policy_provider_plan(plan_group, compile_plan),
        compiled=False,
    )


def _layout_policy_with_native_physical_output_counts(
    compile_plan: dict[str, Any],
    groups: tuple[RegionFirstRuntimeGroup, ...],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for group in groups:
        executor = getattr(group, "executor", None)
        if executor is None or not bool(getattr(executor, "native_halo_output_capable", False)):
            continue
        node = str(getattr(group, "module_prefix", ""))
        native_rows = _layout_policy_incoming_native_rows(compile_plan, node=str(node))
        if not _layout_policy_native_halo_input_supported(executor, native_rows):
            continue
        plan = _layout_policy_native_halo_plan(
            executor,
            compile_plan,
            node=str(node),
            native_input_rows=native_rows,
        )
        if plan is None:
            continue
        spec = getattr(plan, "spec", None)
        output_has_halo = bool(
            int(getattr(spec, "output_alpha", 0) or 0) > 0
            or int(getattr(spec, "output_beta", 0) or 0) > 0
        )
        if not bool(output_has_halo):
            continue
        counts[str(node)] = int(getattr(plan, "output_ct_count", 0) or 0)
    if not counts:
        return dict(compile_plan)
    updated = dict(compile_plan)
    updated["_native_physical_output_ct_counts"] = dict(counts)
    return updated


def _layout_policy_block_count_from_shape(shape: Any, *, slots: int) -> int:
    try:
        size = int(torch.Size(shape).numel())
    except Exception:
        return 0
    if int(size) <= 0 or int(slots) <= 0:
        return 0
    return int(math.ceil(int(size) / float(int(slots))))


def _layout_policy_relayout_operation_totals(kernels: Any) -> dict[str, int]:
    totals = {"rotation_count": 0, "mask_mult_count": 0, "sparse_lt_count": 0}
    for kernel in list(kernels or []):
        estimate = getattr(kernel, "operation_estimate", lambda: {})()
        for key in totals:
            totals[key] += int(dict(estimate).get(key, 0) or 0)
    return totals


def _layout_policy_halo_side_count_from_layout(layout: dict[str, Any]) -> int:
    alpha = max(0, int(layout.get("alpha", 0)))
    beta = max(0, int(layout.get("beta", 0)))
    tile_count = max(1, int(layout.get("tile_count", 1) or 1))
    return int(tile_count) * (int(alpha > 0) + int(beta > 0))


def _layout_policy_rebuild_relayout_edges(edge_layouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relayout_edges: list[dict[str, Any]] = []
    for row in edge_layouts:
        if not bool(row.get("relayout", False)):
            continue
        layout = dict(row.get("selected_layout", {}))
        relayout_edges.append(
            {
                "edge": str(row.get("edge", "")),
                "source": str(row.get("source", "")),
                "target": str(row.get("target", "")),
                "reason": str(row.get("relayout_reason", "")),
                "source_layout": dict(row.get("source_layout", {}) or {}),
                "selected_layout": dict(layout),
                "target_layout": dict(row.get("target_layout", layout) or layout),
                "rotation_estimate": int(row.get("relayout_rotation_estimate", 0) or 0),
                "mask_mult_estimate": int(row.get("relayout_mask_mult_estimate", 0) or 0),
                "sparse_lt_estimate": int(row.get("relayout_sparse_lt_estimate", 0) or 0),
                "depth_estimate": int(row.get("relayout_depth_estimate", 0) or 0),
            }
        )
    return relayout_edges


def _layout_policy_runtime_compile_plan(compile_plan: dict[str, Any]) -> dict[str, Any]:
    """Convert a planner layout policy into the executable provider runtime plan."""

    edge_layouts: list[dict[str, Any]] = []
    honors_producer_layouts = bool(compile_plan.get("node_layouts"))
    for row in compile_plan.get("edge_layouts", []):
        updated = dict(row)
        layout = dict(updated.get("selected_layout", {}))
        op_kind = str(updated.get("op_kind", ""))
        has_halo = _layout_policy_has_halo(layout)
        if (
            not bool(honors_producer_layouts)
            and op_kind in {"conv2d", "avgpool2d"}
            and bool(has_halo)
            and str(updated.get("source", "")) != "x"
        ):
            updated["relayout"] = True
            if not str(updated.get("relayout_reason", "")):
                updated["relayout_reason"] = "runtime_consumer_halo_fill"
            side_count = int(_layout_policy_halo_side_count_from_layout(layout))
            updated["relayout_rotation_estimate"] = int(side_count)
            updated["relayout_mask_mult_estimate"] = int(side_count)
        edge_layouts.append(updated)

    relayout_edges = _layout_policy_rebuild_relayout_edges(edge_layouts)
    output_relayout_nodes = [
        dict(row)
        for row in compile_plan.get("output_relayout_nodes", [])
        if bool(row.get("selected_layout", {}))
    ]
    summary = dict(compile_plan.get("summary", {}))
    summary["relayouts"] = int(len(relayout_edges) + len(output_relayout_nodes))
    summary["relayout_rotation_estimate"] = int(
        sum(int(row["rotation_estimate"]) for row in relayout_edges)
        + sum(int(row.get("rotation_estimate", 0)) for row in output_relayout_nodes)
    )
    summary["relayout_mask_mult_estimate"] = int(
        sum(int(row["mask_mult_estimate"]) for row in relayout_edges)
        + sum(int(row.get("mask_mult_estimate", 0)) for row in output_relayout_nodes)
    )
    summary["relayout_depth_estimate"] = int(
        sum(int(row.get("depth_estimate", 0) or 0) for row in relayout_edges)
        + sum(int(row.get("depth_estimate", 0)) for row in output_relayout_nodes)
    )
    return {
        **dict(compile_plan),
        "edge_layouts": edge_layouts,
        "relayout_edges": relayout_edges,
        "relayout_edge_count": int(len(relayout_edges)),
        "output_relayout_nodes": output_relayout_nodes,
        "output_relayout_node_count": int(len(output_relayout_nodes)),
        "summary": summary,
        "runtime_materialization": "producer_layout_propagation",
    }


def _layout_policy_transform_groups(executor: Any) -> list[Any]:
    groups: list[Any] = []
    group = getattr(executor, "group", None)
    if group is not None:
        groups.append(group)
    for attr in ("groups", "groups_by_pair", "groups_by_input"):
        value = getattr(executor, attr, None)
        if value:
            groups.extend(list(value))
    return groups


def _layout_policy_rotation_pressure(executor: Any, backend: Any | None) -> dict[str, int | bool]:
    get_keys = getattr(backend, "GetLinearTransformRotationKeys", None)
    if not callable(get_keys):
        return {
            "available": False,
            "group_union_rotation_count": 0,
            "transform_sum_rotation_count": 0,
        }
    group_union_total = 0
    transform_sum_total = 0
    for group in _layout_policy_transform_groups(executor):
        union_keys: set[int] = set()
        for transform_id in list(getattr(group, "unified_ids", []) or []):
            try:
                keys = [
                    int(key)
                    for key in list(get_keys(int(transform_id)))
                    if int(key) != 0
                ]
            except Exception:
                keys = []
            union_keys.update(int(key) for key in keys)
            transform_sum_total += int(len(keys))
        group_union_total += int(len(union_keys))
    return {
        "available": True,
        "group_union_rotation_count": int(group_union_total),
        "transform_sum_rotation_count": int(transform_sum_total),
    }


def _layout_policy_reject_reason_count(executor: Any, token: str) -> int:
    total = 0
    for attr in (
        "hybrid_pair_reject_reasons",
        "hybrid_group_reject_reasons",
        "hybrid_pair_layout_reject_reasons",
        "hybrid_pair_schedule_pad_reasons",
    ):
        for reason in list(getattr(executor, attr, []) or []):
            total += str(reason).count(str(token))
    return int(total)


def _layout_policy_region_pressure_row(group: RegionFirstRuntimeGroup, *, backend: Any | None, slots: int) -> dict[str, Any]:
    executor = getattr(group, "executor", None)
    base_executor = getattr(executor, "base_executor", executor)
    relayout_rows = tuple(getattr(executor, "relayout_rows", ()) or ())
    native_rows = tuple(getattr(executor, "native_input_rows", ()) or ())
    compact_align_shared_rows = tuple(getattr(executor, "compact_align_shared_rows", ()) or ())
    output_relayout_rows = tuple(getattr(executor, "output_relayout_rows", ()) or ())
    compact_cols = 0
    halo_cols = 0
    for row in native_rows or compact_align_shared_rows or relayout_rows:
        compact_cols += _layout_policy_block_count_from_shape(
            _layout_policy_compact_on_shape(dict(row)),
            slots=int(slots),
        )
        halo_cols += _layout_policy_block_count_from_shape(
            _layout_policy_halo_on_shape(dict(row)),
            slots=int(slots),
        )
    provider_cols = int(getattr(base_executor, "cols", 0) or 0)
    if compact_cols <= 0:
        compact_cols = int(provider_cols)
    if halo_cols <= 0:
        halo_cols = int(provider_cols)
    rotation = _layout_policy_rotation_pressure(base_executor, backend)
    plan = getattr(group, "plan", None)
    runtime_lowering = str(dict(plan or {}).get("runtime_lowering", ""))
    lowering_label = getattr(executor, "_runtime_lowering_label", None)
    if callable(lowering_label):
        runtime_lowering = str(lowering_label())
    relayout_ops = _layout_policy_relayout_operation_totals(
        [
            *(getattr(executor, "relayout_kernels", []) or []),
            *(
                [getattr(executor, "native_physical_relayout_kernel")]
                if getattr(executor, "native_physical_relayout_kernel", None) is not None
                else []
            ),
            *(getattr(executor, "output_relayout_kernels", []) or []),
        ]
    )
    diagonal_mismatch_count = _layout_policy_reject_reason_count(
        base_executor,
        "diagonal_key_set_mismatch",
    )
    return {
        "region_id": str(getattr(group, "region_id", "")),
        "module_prefix": str(getattr(group, "module_prefix", "")),
        "materializer": str(getattr(group, "materializer", "")),
        "provider_executor": type(base_executor).__name__ if base_executor is not None else "",
        "runtime_lowering": runtime_lowering,
        "native_halo_provider": bool(getattr(executor, "native_halo_input", False)),
        "relayout_edge_count": int(len(relayout_rows)),
        "native_physical_relayout_edge_count": int(
            len(getattr(executor, "native_physical_relayout_rows", ()) or ())
        ),
        "compact_align_shared_edge_count": int(len(compact_align_shared_rows)),
        "output_relayout_edge_count": int(len(output_relayout_rows)),
        "relayout_kernel_count": int(
            len(getattr(executor, "relayout_kernels", []) or [])
            + int(getattr(executor, "native_physical_relayout_kernel", None) is not None)
            + len(getattr(executor, "output_relayout_kernels", []) or [])
        ),
        "relayout_rotation_count": int(relayout_ops["rotation_count"]),
        "relayout_mask_mult_count": int(relayout_ops["mask_mult_count"]),
        "relayout_sparse_lt_count": int(relayout_ops["sparse_lt_count"]),
        "provider_output_rows": int(getattr(base_executor, "rows", 0) or 0),
        "provider_input_block_cols": int(provider_cols),
        "compact_input_block_cols": int(compact_cols),
        "halo_input_block_cols": int(halo_cols),
        "extra_input_block_cols_vs_compact": int(provider_cols - compact_cols),
        "hybrid_pair_count": int(getattr(base_executor, "hybrid_pair_count", 0) or 0),
        "hybrid_pair_rejected_count": int(getattr(base_executor, "hybrid_pair_rejected_count", 0) or 0),
        "hybrid_pair_layout_strict_pair_count": int(
            getattr(base_executor, "hybrid_pair_layout_strict_pair_count", 0) or 0
        ),
        "hybrid_pair_layout_covered_output_count": int(
            getattr(base_executor, "hybrid_pair_layout_covered_output_count", 0) or 0
        ),
        "diagonal_key_set_mismatch_count": int(diagonal_mismatch_count),
        "group_union_rotation_count": int(rotation["group_union_rotation_count"]),
        "transform_sum_rotation_count": int(rotation["transform_sum_rotation_count"]),
        "rotation_pressure_available": bool(rotation["available"]),
    }


def collect_layout_policy_provider_pressure(registry: Any, *, backend: Any | None = None, slots: int | None = None) -> dict[str, Any]:
    groups = tuple(getattr(registry, "groups", ()) or ())
    if slots is None:
        params = getattr(getattr(backend, "scheme", None), "params", None)
        slots = int(getattr(params, "get_slots", lambda: 0)() or 0)
    if int(slots or 0) <= 0:
        slots = 32768
    regions = [
        _layout_policy_region_pressure_row(group, backend=backend, slots=int(slots))
        for group in groups
        if getattr(group, "executor", None) is not None
    ]

    def total(key: str) -> int:
        return int(sum(int(row.get(key, 0) or 0) for row in regions))

    summary = {
        "provider_region_count": int(len(regions)),
        "native_halo_provider_region_count": int(sum(1 for row in regions if bool(row.get("native_halo_provider")))),
        "relayout_lt_region_count": int(
            sum(1 for row in regions if str(row.get("runtime_lowering")) == "provider_executable+relayout_lt")
        ),
        "relayout_edge_count": total("relayout_edge_count"),
        "native_physical_relayout_edge_count": total("native_physical_relayout_edge_count"),
        "compact_align_shared_edge_count": total("compact_align_shared_edge_count"),
        "output_relayout_edge_count": total("output_relayout_edge_count"),
        "relayout_kernel_count": total("relayout_kernel_count"),
        "relayout_rotation_count": total("relayout_rotation_count"),
        "relayout_mask_mult_count": total("relayout_mask_mult_count"),
        "relayout_sparse_lt_count": total("relayout_sparse_lt_count"),
        "provider_input_block_cols": total("provider_input_block_cols"),
        "compact_input_block_cols": total("compact_input_block_cols"),
        "halo_input_block_cols": total("halo_input_block_cols"),
        "extra_input_block_cols_vs_compact": total("extra_input_block_cols_vs_compact"),
        "hybrid_pair_count": total("hybrid_pair_count"),
        "hybrid_pair_rejected_count": total("hybrid_pair_rejected_count"),
        "hybrid_pair_layout_strict_pair_count": total("hybrid_pair_layout_strict_pair_count"),
        "hybrid_pair_layout_covered_output_count": total("hybrid_pair_layout_covered_output_count"),
        "diagonal_key_set_mismatch_count": total("diagonal_key_set_mismatch_count"),
        "group_union_rotation_count": total("group_union_rotation_count"),
        "transform_sum_rotation_count": total("transform_sum_rotation_count"),
        "rotation_pressure_available": bool(any(bool(row.get("rotation_pressure_available")) for row in regions)),
    }
    return {
        "summary": summary,
        "regions": regions,
    }


def _layout_policy_relayout_nodes(compile_plan: dict[str, Any]) -> list[dict[str, str]]:
    edge_nodes = [
        {
            "node": str(row.get("target", "")),
            "edge": str(row.get("edge", "")),
            "reason": str(row.get("reason", "")),
            "kind": "incoming_edge",
        }
        for row in compile_plan.get("relayout_edges", [])
    ]
    output_nodes = [
        {
            "node": str(row.get("node", "")),
            "edge": str(row.get("node", "")),
            "reason": str(row.get("reason", "")),
            "kind": "producer_output",
        }
        for row in compile_plan.get("output_relayout_nodes", [])
    ]
    return [*edge_nodes, *output_nodes]


def _layout_policy_audit_fields(
    compile_plan: dict[str, Any],
    *,
    runtime: str,
    runtime_lowering: str,
) -> dict[str, Any]:
    return {
        "layout_policy": str(compile_plan.get("policy", "dp")),
        "layout_policy_runtime": str(runtime),
        "layout_policy_runtime_lowering": str(runtime_lowering),
        "layout_policy_compile_plan_consumed": True,
        "layout_policy_metric_source": str(compile_plan.get("metric_source", "planner_estimate")),
        "layout_policy_summary": dict(compile_plan.get("summary", {})),
        "layout_policy_edge_layout_count": int(compile_plan.get("edge_layout_count", 0)),
        "layout_policy_relayout_edge_count": int(compile_plan.get("relayout_edge_count", 0)),
        "layout_policy_output_relayout_node_count": int(compile_plan.get("output_relayout_node_count", 0)),
        "layout_policy_relayout_nodes": _layout_policy_relayout_nodes(compile_plan),
        "layout_policy_relayout_edges": [dict(row) for row in compile_plan.get("relayout_edges", [])],
        "layout_policy_output_relayout_nodes": [dict(row) for row in compile_plan.get("output_relayout_nodes", [])],
        "layout_policy_edge_layouts": [dict(row) for row in compile_plan.get("edge_layouts", [])],
        "layout_policy_node_layouts": [dict(row) for row in compile_plan.get("node_layouts", [])],
    }


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
    if not hybrid_pair_schedule_compatible(left, right, int(slots)):
        reason = hybrid_pair_schedule_reject_reason(left, right, int(slots))
        raise ValueError(f"tconv source-block hybrid merge requires identical schedules: {reason}")
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
    native_halo_input_capable = True
    native_halo_output_capable = True

    def __init__(
        self,
        *,
        module: Any,
        output_node_id: str,
        use_ct_pt_hybrid_packing: bool = False,
        project_complex_inputs_to_real: bool = True,
        disable_tile_family_sharing: bool | None = None,
    ) -> None:
        if not _u22_tconv_module_supported(module):
            raise ValueError("U22 experimental tconv kernel only supports k=2, s=2, gap-halving ConvTranspose2d layers")
        self.module = module
        self.output_node_id = str(output_node_id)
        self.use_ct_pt_hybrid_packing = bool(use_ct_pt_hybrid_packing)
        self.project_complex_inputs_to_real = bool(project_complex_inputs_to_real)
        self.disable_tile_family_sharing = (
            _env_truthy("ORION_U22_DISABLE_TILE_FAMILY_SHARING")
            if disable_tile_family_sharing is None
            else bool(disable_tile_family_sharing)
        )
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.groups: list[UnifiedTransformGroup] = []
        self.target_indices_by_input_unit: list[tuple[int, ...]] = []
        self.input_block_pairs: list[tuple[int, int | None]] = []
        self.complex_input_block_flags: list[bool] = []
        self.hybrid_group_reject_reasons: list[str] = []
        self.hybrid_pair_count = 0
        self.hybrid_pair_rejected_count = 0
        self.hybrid_pair_reject_reasons: list[str] = []
        self.hybrid_pair_schedule_padded_count = 0
        self.hybrid_pair_schedule_pad_reasons: list[str] = []
        self.hybrid_pair_layout_strategy = ""
        self.hybrid_pair_layout_strict_pair_count = 0
        self.hybrid_pair_layout_covered_output_count = 0
        self.hybrid_pair_layout_reject_reasons: list[str] = []
        self.output_block_count = 0
        self.input_block_count = 0
        self.input_total_slots = 0
        self.output_total_slots = 0
        self.compiled_fhe_input_shape: tuple[int, ...] = ()
        self.compiled_fhe_output_shape: tuple[int, ...] = ()
        self.compiled_input_layout: dict[str, int] = {}
        self.compiled_output_layout: dict[str, int] = {}
        self.output_fold_block_height = 0
        self.output_fold_rotations = 0
        self.compile_count = 0
        self.block_evaluate_count = 0
        self.real_projection_count = 0
        self.compiled_transform_count = 0
        self.skipped_empty_transform_count = 0
        self._compile_cache_metadata: dict[str, Any] = {}
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
            "output_fold_s": 0.0,
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
                "output_fold_s": 0.0,
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
            "output_fold_rotation_count": 0,
            "bias_add_count": 0,
            "input_pack_count": 0,
        }
        self.last_runtime_io = {}

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        self._compile_cache_metadata = dict(metadata or {})

    def compile_cache_metadata(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "fhe_input_shape": [int(value) for value in self.compiled_fhe_input_shape],
            "fhe_output_shape": [int(value) for value in self.compiled_fhe_output_shape],
            "layout_policy_input_layout": dict(self.compiled_input_layout),
            "layout_policy_output_layout": dict(self.compiled_output_layout),
            "input_block_count": int(self.input_block_count),
            "output_block_count": int(self.output_block_count),
            "input_total_slots": int(self.input_total_slots),
            "output_total_slots": int(self.output_total_slots),
            "output_fold_block_height": int(self.output_fold_block_height),
            "output_fold_rotations": int(self.output_fold_rotations),
            "hybrid_pair_count": int(self.hybrid_pair_count),
            "hybrid_pair_rejected_count": int(self.hybrid_pair_rejected_count),
            "hybrid_pair_reject_reasons": [str(value) for value in self.hybrid_pair_reject_reasons],
            "hybrid_pair_schedule_padded_count": int(self.hybrid_pair_schedule_padded_count),
            "hybrid_pair_schedule_pad_reasons": [str(value) for value in self.hybrid_pair_schedule_pad_reasons],
            "hybrid_pair_layout_strategy": str(self.hybrid_pair_layout_strategy),
            "hybrid_pair_layout_strict_pair_count": int(self.hybrid_pair_layout_strict_pair_count),
            "hybrid_pair_layout_covered_output_count": int(self.hybrid_pair_layout_covered_output_count),
            "hybrid_pair_layout_reject_reasons": [
                str(value) for value in self.hybrid_pair_layout_reject_reasons
            ],
            "groups_by_input_unit": [
                {
                    "storage_key": str(getattr(group, "_storage_key", "")),
                    "target_indices": [int(value) for value in self.target_indices_by_input_unit[index]],
                    "input_block_pair": [
                        int(self.input_block_pairs[index][0]),
                        None if self.input_block_pairs[index][1] is None else int(self.input_block_pairs[index][1]),
                    ],
                    "complex_input_block": bool(self.complex_input_block_flags[index]),
                    "hybrid_pair_reject_reason": (
                        str(self.hybrid_group_reject_reasons[index])
                        if index < len(self.hybrid_group_reject_reasons)
                        else ""
                    ),
                }
                for index, group in enumerate(self.groups)
            ],
        }

    def _compile_cache_group_metadata_from_inferred_keys(self, storage_keys: list[str]) -> list[dict[str, Any]]:
        keys = [str(value) for value in storage_keys if str(value)]
        if not keys:
            return []

        pairs: list[tuple[tuple[int, int | None], bool]] = []
        if bool(self.use_ct_pt_hybrid_packing) and int(self.input_block_count) > 1:
            for left_block in range(0, int(self.input_block_count), 2):
                right_block = int(left_block + 1)
                has_right = int(right_block) < int(self.input_block_count)
                pairs.append(((int(left_block), int(right_block) if bool(has_right) else None), bool(has_right)))
        else:
            for source_block in range(int(self.input_block_count)):
                pairs.append(((int(source_block), None), False))

        if len(keys) == len(pairs):
            return [
                {
                    "storage_key": str(storage_key),
                    "target_indices": [int(index) for index in range(int(self.output_block_count))],
                    "input_block_pair": [int(pair[0]), None if pair[1] is None else int(pair[1])],
                    "complex_input_block": bool(is_complex),
                    "hybrid_pair_reject_reason": "",
                }
                for storage_key, (pair, is_complex) in zip(keys, pairs)
            ]

        expected_split_count = int(len(pairs) * max(1, int(self.output_block_count)))
        if len(keys) == expected_split_count:
            rows: list[dict[str, Any]] = []
            cursor = 0
            for pair, is_complex in pairs:
                for output_block in range(int(self.output_block_count)):
                    rows.append(
                        {
                            "storage_key": str(keys[int(cursor)]),
                            "target_indices": [int(output_block)],
                            "input_block_pair": [int(pair[0]), None if pair[1] is None else int(pair[1])],
                            "complex_input_block": bool(is_complex),
                            "hybrid_pair_reject_reason": "",
                        }
                    )
                    cursor += 1
            return rows

        raise RuntimeError(
            f"Cached U22 tconv metadata has {len(keys)} storage keys, expected "
            f"{len(pairs)} grouped or {expected_split_count} split keys for {self.output_node_id}."
        )

    def _compile_from_cache_metadata(self, scheme: Any) -> bool:
        metadata = dict(self._compile_cache_metadata or {})
        if str(getattr(scheme.params, "get_io_mode", lambda: "none")()).lower() != "load" or not metadata:
            return False

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
            "output_fold_s": 0.0,
            "bias_s": 0.0,
            "total_call_s": 0.0,
        }
        self._set_block_layout(scheme=scheme)
        expected_signature = self._cache_layout_signature()
        cached_signature = {
            "fhe_input_shape": [int(value) for value in metadata.get("fhe_input_shape", [])],
            "fhe_output_shape": [int(value) for value in metadata.get("fhe_output_shape", [])],
            "layout_policy_input_layout": dict(metadata.get("layout_policy_input_layout", {}) or {}),
            "layout_policy_output_layout": dict(metadata.get("layout_policy_output_layout", {}) or {}),
        }
        if cached_signature["fhe_input_shape"] and cached_signature != expected_signature:
            return False
        if not cached_signature["fhe_input_shape"] and (
            bool(expected_signature["layout_policy_input_layout"].get("alpha", 0))
            or bool(expected_signature["layout_policy_input_layout"].get("beta", 0))
            or bool(expected_signature["layout_policy_output_layout"].get("alpha", 0))
            or bool(expected_signature["layout_policy_output_layout"].get("beta", 0))
        ):
            return False
        group_rows = list(metadata.get("groups_by_input_unit", []))
        if not group_rows:
            group_rows = self._compile_cache_group_metadata_from_inferred_keys(
                [str(value) for value in metadata.get("inferred_storage_keys", [])]
            )
        if not group_rows:
            return False

        level = int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1
        self.groups = []
        self.target_indices_by_input_unit = []
        self.input_block_pairs = []
        self.complex_input_block_flags = []
        self.hybrid_group_reject_reasons = []
        self.hybrid_pair_count = int(metadata.get("hybrid_pair_count", 0))
        self.hybrid_pair_rejected_count = int(metadata.get("hybrid_pair_rejected_count", 0))
        self.hybrid_pair_reject_reasons = [str(value) for value in metadata.get("hybrid_pair_reject_reasons", [])]
        self.hybrid_pair_schedule_padded_count = int(metadata.get("hybrid_pair_schedule_padded_count", 0))
        self.hybrid_pair_schedule_pad_reasons = [
            str(value) for value in metadata.get("hybrid_pair_schedule_pad_reasons", [])
        ]
        self.hybrid_pair_layout_strategy = str(metadata.get("hybrid_pair_layout_strategy", ""))
        self.hybrid_pair_layout_strict_pair_count = int(metadata.get("hybrid_pair_layout_strict_pair_count", 0))
        self.hybrid_pair_layout_covered_output_count = int(
            metadata.get("hybrid_pair_layout_covered_output_count", 0)
        )
        self.hybrid_pair_layout_reject_reasons = [
            str(value) for value in metadata.get("hybrid_pair_layout_reject_reasons", [])
        ]
        self.compiled_transform_count = 0
        self.skipped_empty_transform_count = 0

        compile_started = time.time()
        for group_meta in group_rows:
            target_indices = tuple(int(value) for value in group_meta.get("target_indices", []))
            if not target_indices:
                continue
            group = UnifiedTransformGroup(
                [_cached_transform_shell(level=int(level), scheme=scheme) for _target in target_indices]
            )
            group._storage_key = str(group_meta["storage_key"])
            group.compile_unified(scheme.backend)
            pair = list(group_meta.get("input_block_pair", []))
            if not pair:
                raise RuntimeError(f"Cached U22 tconv metadata missing input block pair for {self.output_node_id}")
            self.groups.append(group)
            self.target_indices_by_input_unit.append(target_indices)
            self.input_block_pairs.append((int(pair[0]), None if len(pair) < 2 or pair[1] is None else int(pair[1])))
            self.complex_input_block_flags.append(bool(group_meta.get("complex_input_block", False)))
            self.hybrid_group_reject_reasons.append(str(group_meta.get("hybrid_pair_reject_reason", "")))
            self.compiled_transform_count += int(len(target_indices))
        if "hybrid_pair_count" not in metadata:
            self.hybrid_pair_count = int(sum(1 for value in self.complex_input_block_flags if bool(value)))

        bias_level = max(0, int(level) - max(0, int(self.assigned_depth) if self.assigned_depth is not None else 1))
        bias_started = time.time()
        self._compile_bias_plaintexts(scheme=scheme, level=int(bias_level))
        self.last_runtime_timing["bias_s"] = float(time.time() - bias_started)
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)
        self.compile_count += 1
        self._compiled = bool(self.groups)
        return bool(self.groups)

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
        return int(slots) > 0

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
        input_total_slots = int(on_ci * on_hi * on_wi)
        output_total_slots = int(on_co * on_ho * on_wo)
        return {
            "slots": int(slots),
            "input_channels_per_block": int(input_channels_per_block),
            "output_channels_per_block": int(output_channels_per_block),
            "input_block_count": int(math.ceil(int(input_total_slots) / int(slots))),
            "output_block_count": int(math.ceil(int(output_total_slots) / int(slots))),
            "input_total_slots": int(input_total_slots),
            "output_total_slots": int(output_total_slots),
        }

    def _input_halo_layout(self) -> dict[str, int]:
        layout = dict(getattr(self.module, "layout_policy_input_layout", {}) or {})
        return {
            "alpha": max(0, int(layout.get("alpha", 0))),
            "beta": max(0, int(layout.get("beta", 0))),
            "gap": max(1, int(layout.get("gap", getattr(self.module, "input_gap", 1)))),
        }

    def _output_halo_layout(self) -> dict[str, int]:
        layout = dict(getattr(self.module, "layout_policy_output_layout", {}) or {})
        return {
            "alpha": max(0, int(layout.get("alpha", 0))),
            "beta": max(0, int(layout.get("beta", 0))),
            "gap": max(1, int(layout.get("gap", getattr(self.module, "output_gap", 1)))),
        }

    def _cache_layout_signature(self) -> dict[str, Any]:
        return {
            "fhe_input_shape": [int(value) for value in getattr(self.module, "fhe_input_shape")],
            "fhe_output_shape": [int(value) for value in getattr(self.module, "fhe_output_shape")],
            "layout_policy_input_layout": dict(self._input_halo_layout()),
            "layout_policy_output_layout": dict(self._output_halo_layout()),
        }

    def compile(self, scheme: Any) -> None:
        if self._compiled:
            return
        if not self.supports_scheme(scheme):
            raise RuntimeError("U22 experimental tconv kernel requires a compatible Python or Lattigo backend")
        if self._compile_from_cache_metadata(scheme):
            return
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
            "output_fold_s": 0.0,
            "bias_s": 0.0,
            "total_call_s": 0.0,
        }
        level = int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1
        self._set_block_layout(scheme=scheme)
        self.groups = []
        self.target_indices_by_input_unit = []
        self.input_block_pairs = []
        self.complex_input_block_flags = []
        self.hybrid_group_reject_reasons = []
        self.hybrid_pair_count = 0
        self.hybrid_pair_rejected_count = 0
        self.hybrid_pair_reject_reasons = []
        self.hybrid_pair_schedule_padded_count = 0
        self.hybrid_pair_schedule_pad_reasons = []
        self.hybrid_pair_layout_strategy = ""
        self.hybrid_pair_layout_strict_pair_count = 0
        self.hybrid_pair_layout_covered_output_count = 0
        self.hybrid_pair_layout_reject_reasons = []
        self.compiled_transform_count = 0
        self.skipped_empty_transform_count = 0

        prepare_total = 0.0
        compile_total = 0.0
        if bool(self.use_ct_pt_hybrid_packing) and int(self.input_block_count) > 1:
            block_transforms_by_source: dict[int, list[Any | None]] = {}
            for source_block in range(int(self.input_block_count)):
                prepare_started = time.time()
                transforms, empty_count = self._build_source_block_transforms(
                    scheme=scheme,
                    level=int(level),
                    source_block=int(source_block),
                )
                prepare_total += float(time.time() - prepare_started)
                self.skipped_empty_transform_count += int(empty_count)
                block_transforms_by_source[int(source_block)] = list(transforms)

            layout_plan = optimize_hybrid_pair_layout(
                block_transforms_by_source,
                int(scheme.params.get_slots()),
                allow_schedule_materialization=True,
            )
            materialization = materialize_hybrid_pair_layout_schedules(
                block_transforms_by_source,
                layout_plan,
                int(scheme.params.get_slots()),
                name_prefix=f"{self.output_node_id}_global_hybrid_layout",
            )
            self.hybrid_pair_layout_strict_pair_count = int(layout_plan.strict_pair_count)
            self.hybrid_pair_layout_covered_output_count = int(layout_plan.covered_output_count)
            self.hybrid_pair_layout_reject_reasons = [
                str(value) for value in layout_plan.rejected_adjacent_pair_reasons
            ]
            self.hybrid_pair_schedule_padded_count += int(materialization.pair_count)
            self.hybrid_pair_schedule_pad_reasons.extend(str(value) for value in materialization.reasons)
            use_strict_layout = int(layout_plan.strict_pair_count) > 0
            if bool(use_strict_layout):
                self.hybrid_pair_layout_strategy = (
                    "global_schedule_layout"
                    if int(materialization.pair_count) > 0
                    else "strict_schedule_dp"
                )
                layout_items = [
                    (int(item.left_index), None if item.right_index is None else int(item.right_index), True)
                    for item in layout_plan.items
                ]
            else:
                self.hybrid_pair_layout_strategy = "adjacent_strict_reject_fallback"
                layout_items = []
                for left_block in range(0, int(self.input_block_count), 2):
                    right_block = int(left_block + 1)
                    has_right = int(right_block) < int(self.input_block_count)
                    layout_items.append((int(left_block), int(right_block) if bool(has_right) else None, False))

            for left_block, maybe_right_block, _layout_pair_planned in layout_items:
                left_transforms = block_transforms_by_source[int(left_block)]
                has_right = maybe_right_block is not None
                right_block = int(maybe_right_block) if maybe_right_block is not None else int(left_block + 1)
                right_transforms = block_transforms_by_source.get(int(right_block)) if bool(has_right) else None
                if bool(has_right) and right_transforms is not None:
                    candidates: list[tuple[int, Any | None, Any | None]] = []
                    reject_reasons: list[str] = []
                    pair_pad_reasons: list[str] = []
                    for output_block in range(int(self.output_block_count)):
                        left_transform = left_transforms[int(output_block)]
                        right_transform = (
                            right_transforms[int(output_block)]
                            if right_transforms is not None
                            else None
                        )
                        if left_transform is None and right_transform is None:
                            continue
                        pad_reason = ""
                        if pad_reason:
                            pair_pad_reasons.append(f"output_block={int(output_block)}:{pad_reason}")
                        candidates.append((int(output_block), left_transform, right_transform))
                        reason = hybrid_pair_schedule_reject_reason(
                            left_transform,
                            right_transform,
                            int(scheme.params.get_slots()),
                        )
                        if reason:
                            reject_reasons.append(f"output_block={int(output_block)}:{reason}")
                    if not candidates:
                        entries = []
                        reason = ""
                        left_entries = []
                        right_entries = []
                    elif reject_reasons:
                        reason = "; ".join(reject_reasons)
                        self.hybrid_pair_rejected_count += 1
                        self.hybrid_pair_reject_reasons.append(
                            f"input_pair=({int(left_block)},{int(right_block)}):{reason}"
                        )
                        left_entries = [
                            (int(output_block), left_transform)
                            for output_block, left_transform, _right_transform in candidates
                            if left_transform is not None
                        ]
                        right_entries = [
                            (int(output_block), right_transform)
                            for output_block, _left_transform, right_transform in candidates
                            if right_transform is not None
                        ]
                        entries = []
                    else:
                        entries = [
                            (
                                int(output_block),
                                _merge_optional_tconv_source_block_transforms_to_complex(
                                    left_transform,
                                    right_transform,
                                    name=(
                                        f"{self.output_node_id}_ctpt_hybrid_src"
                                        f"{int(left_block)}_{int(right_block)}_out{int(output_block)}"
                                    ),
                                    real_lane_input_scale=(0.25 if bool(self.project_complex_inputs_to_real) else 0.5),
                                ),
                            )
                            for output_block, left_transform, right_transform in candidates
                        ]
                        self.hybrid_pair_count += 1
                        if pair_pad_reasons:
                            self.hybrid_pair_schedule_padded_count += 1
                            self.hybrid_pair_schedule_pad_reasons.append(
                                f"input_pair=({int(left_block)},{int(right_block)}):" + "; ".join(pair_pad_reasons)
                            )
                        reason = ""
                        left_entries = []
                        right_entries = []
                        for _output_block, left_transform, right_transform in candidates:
                            self._clear_transform_diagonals(left_transform)
                            self._clear_transform_diagonals(right_transform)
                else:
                    entries = [
                        (int(output_block), transform)
                        for output_block, transform in enumerate(left_transforms)
                        if transform is not None
                    ]
                    reason = ""
                    left_entries = []
                    right_entries = []
                if entries:
                    compile_total += self._compile_entry_groups(
                        scheme=scheme,
                        pair=(int(left_block), int(right_block) if bool(has_right) else None),
                        is_complex=bool(has_right),
                        entries=entries,
                    )
                if left_entries:
                    compile_total += self._compile_entry_groups(
                        scheme=scheme,
                        pair=(int(left_block), None),
                        is_complex=False,
                        entries=left_entries,
                        reject_reason=reason,
                    )
                if right_entries:
                    compile_total += self._compile_entry_groups(
                        scheme=scheme,
                        pair=(int(right_block), None),
                        is_complex=False,
                        entries=right_entries,
                        reject_reason=reason,
                    )
                del left_transforms, right_transforms, entries
            del block_transforms_by_source
        else:
            for source_block in range(int(self.input_block_count)):
                prepare_started = time.time()
                transforms, empty_count = self._build_source_block_transforms(
                    scheme=scheme,
                    level=int(level),
                    source_block=int(source_block),
                )
                self.skipped_empty_transform_count += int(empty_count)
                entries = [
                    (int(output_block), transform)
                    for output_block, transform in enumerate(transforms)
                    if transform is not None
                ]
                prepare_total += float(time.time() - prepare_started)
                if not entries:
                    continue
                compile_total += self._compile_entry_groups(
                    scheme=scheme,
                    pair=(int(source_block), None),
                    is_complex=False,
                    entries=entries,
                )
                del transforms, entries
        bias_level = max(0, int(level) - max(0, int(self.assigned_depth) if self.assigned_depth is not None else 1))
        bias_started = time.time()
        self._compile_bias_plaintexts(scheme=scheme, level=int(bias_level))
        compile_total += float(time.time() - bias_started)
        self.compile_count += 1
        self.last_runtime_timing["prepare_transforms_s"] = float(prepare_total)
        self.last_runtime_timing["compile_unified_s"] = float(compile_total)
        self._compiled = True

    def _set_block_layout(self, *, scheme: Any) -> None:
        layout = self._block_layout(scheme=scheme)
        self.compiled_fhe_input_shape = tuple(int(value) for value in getattr(self.module, "fhe_input_shape"))
        self.compiled_fhe_output_shape = tuple(int(value) for value in getattr(self.module, "fhe_output_shape"))
        self.compiled_input_layout = dict(self._input_halo_layout())
        self.compiled_output_layout = dict(self._output_halo_layout())
        self.input_block_count = int(layout["input_block_count"])
        self.output_block_count = int(layout["output_block_count"])
        self.input_total_slots = int(layout["input_total_slots"])
        self.output_total_slots = int(layout["output_total_slots"])
        self.output_fold_block_height = int(layout["slots"])
        self.output_fold_rotations = 0
        embed_method = str(getattr(scheme.params, "get_embedding_method", lambda: "")())
        if (
            int(self.output_block_count) == 1
            and str(embed_method) == "hybrid"
            and int(self.output_total_slots) > 0
            and int(self.output_total_slots) < int(layout["slots"])
        ):
            block_height = int(_ceil_pow2(int(self.output_total_slots)))
            if int(block_height) < int(layout["slots"]):
                self.output_fold_block_height = int(block_height)
                self.output_fold_rotations = int(math.log2(int(layout["slots"]) // int(block_height)))

    def _compile_entry_groups(
        self,
        *,
        scheme: Any,
        pair: tuple[int, int | None],
        is_complex: bool,
        entries: list[tuple[int, Any]],
        reject_reason: str = "",
    ) -> float:
        entry_groups = (
            [[entry] for entry in sorted(entries, key=lambda item: int(item[0]))]
            if bool(self.disable_tile_family_sharing)
            else [sorted(entries, key=lambda item: int(item[0]))]
        )
        elapsed = 0.0
        for ordered in entry_groups:
            started = time.time()
            transforms = [transform for _target_index, transform in ordered]
            group = UnifiedTransformGroup(transforms)
            group.compile_unified(scheme.backend)
            elapsed += float(time.time() - started)
            self.groups.append(group)
            self.compiled_transform_count += int(len(transforms))
            self.target_indices_by_input_unit.append(tuple(int(target_index) for target_index, _transform in ordered))
            self.input_block_pairs.append((int(pair[0]), None if pair[1] is None else int(pair[1])))
            self.complex_input_block_flags.append(bool(is_complex))
            self.hybrid_group_reject_reasons.append(str(reject_reason))
        return float(elapsed)

    def _clear_transform_diagonals(self, transform: Any | None) -> None:
        if transform is None:
            return
        try:
            getattr(transform, "diagonals", {}).clear()
        except Exception:
            pass

    def cleanup(self, backend: Any) -> None:
        for group in self.groups:
            if hasattr(group, "cleanup"):
                group.cleanup(backend)
        self.groups = []
        self.target_indices_by_input_unit = []
        self.input_block_pairs = []
        self.complex_input_block_flags = []
        self.hybrid_group_reject_reasons = []
        self.hybrid_pair_count = 0
        self.hybrid_pair_rejected_count = 0
        self.hybrid_pair_reject_reasons = []
        self.hybrid_pair_layout_strategy = ""
        self.hybrid_pair_layout_strict_pair_count = 0
        self.hybrid_pair_layout_covered_output_count = 0
        self.hybrid_pair_layout_reject_reasons = []
        self.output_block_count = 0
        self.input_block_count = 0
        self.input_total_slots = 0
        self.output_total_slots = 0
        self.compiled_fhe_input_shape = ()
        self.compiled_fhe_output_shape = ()
        self.compiled_input_layout = {}
        self.compiled_output_layout = {}
        self.output_fold_block_height = 0
        self.output_fold_rotations = 0
        self._bias_ptxt_cache = {}
        self._compiled = False

    def _iter_source_positions_for_block(
        self,
        *,
        source_block: int,
        slots: int,
        c_in: int,
        h_in: int,
        w_in: int,
        input_gap: int,
    ):
        start = int(source_block) * int(slots)
        stop = min(int(start + int(slots)), int(self.input_total_slots))
        if int(stop) <= int(start):
            return
        gap = max(1, int(input_gap))
        if int(gap) == 1:
            plane = int(h_in * w_in)
            for source_slot in range(int(start), int(stop)):
                ic = int(source_slot // int(plane))
                rem = int(source_slot % int(plane))
                if int(ic) >= int(c_in):
                    continue
                yield int(source_slot), int(ic), int(rem // int(w_in)), int(rem % int(w_in))
            return

        phase_count = int(gap * gap)
        packed_w = int(w_in * gap)
        group_block = int(h_in * gap * packed_w)
        if int(group_block) <= 0:
            return
        for source_slot in range(int(start), int(stop)):
            group = int(source_slot // int(group_block))
            rem = int(source_slot % int(group_block))
            packed_h_index = int(rem // int(packed_w))
            packed_w_index = int(rem % int(packed_w))
            ih = int(packed_h_index // int(gap))
            iw = int(packed_w_index // int(gap))
            if int(ih) >= int(h_in) or int(iw) >= int(w_in):
                continue
            phase_h = int(packed_h_index % int(gap))
            phase_w = int(packed_w_index % int(gap))
            ic = int(group * phase_count + phase_h * gap + phase_w)
            if int(ic) >= int(c_in):
                continue
            yield int(source_slot), int(ic), int(ih), int(iw)

    def _source_position_tensors_for_block(
        self,
        *,
        source_block: int,
        slots: int,
        c_in: int,
        h_in: int,
        w_in: int,
        input_gap: int,
        input_alpha: int = 0,
        input_beta: int = 0,
        input_physical_h: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(source_block) * int(slots)
        stop = min(int(start + int(slots)), int(self.input_total_slots))
        if int(stop) <= int(start):
            empty = torch.empty((0,), dtype=torch.int64)
            return empty, empty, empty, empty
        source_slots = torch.arange(int(start), int(stop), dtype=torch.int64)
        gap = max(1, int(input_gap))
        alpha = max(0, int(input_alpha))
        beta = max(0, int(input_beta))
        physical_h = int(input_physical_h) if input_physical_h is not None else int((int(h_in) + int(alpha + beta)) * int(gap))
        if int(gap) == 1:
            plane = int(physical_h * w_in)
            ic = torch.div(source_slots, int(plane), rounding_mode="floor")
            rem = torch.remainder(source_slots, int(plane))
            ih = torch.div(rem, int(w_in), rounding_mode="floor") - int(alpha)
            iw = torch.remainder(rem, int(w_in))
            valid = (ic < int(c_in)) & (ih >= -int(alpha)) & (ih < int(h_in + beta))
            return source_slots[valid], ic[valid], ih[valid], iw[valid]

        packed_w = int(w_in * gap)
        group_block = int(physical_h * packed_w)
        if int(group_block) <= 0:
            empty = torch.empty((0,), dtype=torch.int64)
            return empty, empty, empty, empty
        group = torch.div(source_slots, int(group_block), rounding_mode="floor")
        rem = torch.remainder(source_slots, int(group_block))
        packed_h_index = torch.div(rem, int(packed_w), rounding_mode="floor")
        packed_w_index = torch.remainder(rem, int(packed_w))
        ih = torch.div(packed_h_index, int(gap), rounding_mode="floor") - int(alpha)
        iw = torch.div(packed_w_index, int(gap), rounding_mode="floor")
        phase_h = torch.remainder(packed_h_index, int(gap))
        phase_w = torch.remainder(packed_w_index, int(gap))
        ic = group * int(gap * gap) + phase_h * int(gap) + phase_w
        valid = (ih >= -int(alpha)) & (ih < int(h_in + beta)) & (iw < int(w_in)) & (ic < int(c_in))
        return source_slots[valid], ic[valid], ih[valid], iw[valid]

    def _build_source_block_transforms(self, *, scheme: Any, level: int, source_block: int) -> tuple[list[Any | None], int]:
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
        input_layout = self._input_halo_layout()
        output_layout = self._output_halo_layout()
        input_alpha = max(0, int(input_layout.get("alpha", 0)))
        input_beta = max(0, int(input_layout.get("beta", 0)))
        output_alpha = max(0, int(output_layout.get("alpha", 0)))
        output_beta = max(0, int(output_layout.get("beta", 0)))
        output_materialization = str(getattr(self.module, "layout_policy_output_materialization", "") or "")
        fused_output_relayout = output_materialization == "fused_relayout"
        input_physical_h = int(getattr(self.module, "fhe_input_shape")[2])
        output_total_h = int(h_out + output_alpha + output_beta)

        source_slots, ic, ih, iw = self._source_position_tensors_for_block(
            source_block=int(source_block),
            slots=int(slots),
            c_in=int(c_in),
            h_in=int(h_in),
            w_in=int(w_in),
            input_gap=int(input_gap),
            input_alpha=int(input_alpha),
            input_beta=int(input_beta),
            input_physical_h=int(input_physical_h),
        )
        diagonal_parts: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
            []
            for _output_block in range(int(self.output_block_count))
        ]
        if int(source_slots.numel()) > 0:
            source_local = torch.remainder(source_slots, int(slots))
            oc = torch.arange(int(c_out), dtype=torch.int64)
            oc_grid = oc.unsqueeze(0).expand(int(source_slots.numel()), int(c_out))
            source_grid = source_local.unsqueeze(1)
            for kh in range(2):
                oh = ih * 2 + int(kh)
                for kw in range(2):
                    ow = iw * 2 + int(kw)
                    coeff = weight[
                        ic.to(dtype=torch.int64).unsqueeze(1),
                        oc.unsqueeze(0),
                        int(kh),
                        int(kw),
                    ].to(dtype=torch.float32)
                    nonzero = coeff != 0
                    if not bool(torch.any(nonzero).item()):
                        continue
                    ow_grid = ow.unsqueeze(1).expand_as(oc_grid)
                    target_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
                    if bool(fused_output_relayout):
                        core_valid = (oh >= 0) & (oh < int(h_out))
                        target_rows.append((oh + int(output_alpha), core_valid))
                        top_valid = core_valid & (oh == 0)
                        for top_row in range(int(output_alpha)):
                            target_rows.append(
                                (
                                    torch.full_like(oh, int(top_row)),
                                    top_valid,
                                )
                            )
                        bottom_valid = core_valid & (oh == int(h_out) - 1)
                        for bottom_row in range(int(output_beta)):
                            target_rows.append(
                                (
                                    torch.full_like(oh, int(output_alpha + h_out + bottom_row)),
                                    bottom_valid,
                                )
                            )
                    else:
                        oh_total = oh + int(output_alpha)
                        target_rows.append(
                            (
                                oh_total,
                                (oh_total >= 0) & (oh_total < int(output_total_h)),
                            )
                        )
                    for target_h, row_valid in target_rows:
                        oh_grid = target_h.unsqueeze(1).expand_as(oc_grid)
                        out_slot = _idx_chw_gap_tensor(
                            oc_grid,
                            oh_grid,
                            ow_grid,
                            h=int(output_total_h),
                            w=int(w_out),
                            gap=int(output_gap),
                        )
                        output_block = torch.div(out_slot, int(slots), rounding_mode="floor")
                        output_local = torch.remainder(out_slot, int(slots))
                        if int(self.output_fold_rotations) > 0 and int(self.output_block_count) == 1:
                            diag_idx = torch.remainder(
                                source_grid - output_local,
                                int(self.output_fold_block_height),
                            )
                            output_positions = torch.remainder(source_grid - diag_idx, int(slots))
                        else:
                            diag_idx = torch.remainder(source_grid - output_local, int(slots))
                            output_positions = output_local
                        valid = (
                            nonzero
                            & row_valid.unsqueeze(1)
                            & (ow_grid >= 0)
                            & (ow_grid < int(w_out))
                            & (output_block >= 0)
                            & (output_block < int(self.output_block_count))
                        )
                        if not bool(torch.any(valid).item()):
                            continue
                        flat_blocks = output_block[valid].to(dtype=torch.int64)
                        flat_diags = diag_idx[valid].to(dtype=torch.int64)
                        flat_locals = output_positions[valid].to(dtype=torch.int64)
                        flat_values = coeff[valid].to(dtype=torch.float32)
                        for block in torch.unique(flat_blocks).tolist():
                            block_value = int(block)
                            block_mask = flat_blocks == int(block_value)
                            diagonal_parts[int(block_value)].append(
                                (
                                    flat_diags[block_mask].clone(),
                                    flat_locals[block_mask].clone(),
                                    flat_values[block_mask].clone(),
                                )
                            )

        block_transforms: list[Any | None] = []
        empty_transform_count = 0
        for output_block in range(int(self.output_block_count)):
            diagonals: dict[int, torch.Tensor] = {}
            if diagonal_parts[int(output_block)]:
                block_diag_idx = torch.cat([part[0] for part in diagonal_parts[int(output_block)]]).to(dtype=torch.int64)
                block_output_local = torch.cat([part[1] for part in diagonal_parts[int(output_block)]]).to(dtype=torch.int64)
                block_values = torch.cat([part[2] for part in diagonal_parts[int(output_block)]]).to(dtype=torch.float32)
            else:
                block_diag_idx = torch.empty((0,), dtype=torch.int64)
                block_output_local = torch.empty((0,), dtype=torch.int64)
                block_values = torch.empty((0,), dtype=torch.float32)
            for diag_idx in torch.unique(block_diag_idx).tolist():
                mask = block_diag_idx == int(diag_idx)
                diag = torch.zeros((int(slots),), dtype=torch.float32)
                diag.index_add_(0, block_output_local[mask], block_values[mask])
                diagonals[int(diag_idx)] = diag
            if not diagonals:
                empty_transform_count += 1
                block_transforms.append(None)
                continue
            block_transforms.append(
                mark_hybrid_schedule_padding_allowed(
                    SimpleNamespace(
                        name=f"{self.output_node_id}_experimental_tconv_src{int(source_block)}_out{int(output_block)}",
                        diagonals={(0, 0): diagonals},
                        level=int(level),
                        scheme=scheme,
                        fhe_output_shape=torch.Size([1, int(slots)]),
                        output_shape=torch.Size([1, int(slots)]),
                    ),
                    family=(
                        f"u22_tconv_k2s2:"
                        f"in_gap={int(input_gap)}:out_gap={int(output_gap)}:"
                        f"in={int(c_in)}x{int(h_in)}x{int(w_in)}:"
                        f"out={int(c_out)}x{int(h_out)}x{int(w_out)}:"
                        f"in_halo={int(input_alpha)},{int(input_beta)}:"
                        f"out_halo={int(output_alpha)},{int(output_beta)}"
                    ),
                )
            )
        return block_transforms, int(empty_transform_count)

    def _bias_plaintext(self, *, scheme: Any, level: int, output_block: int):
        cache_key = (int(output_block), int(level))
        cached = self._bias_ptxt_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._bias_vector is None:
            self._bias_vector = self._construct_bias_vector().to(dtype=torch.float32)
        slots = int(scheme.params.get_slots())
        start = int(output_block) * int(slots)
        stop = min(int(start + int(slots)), int(self._bias_vector.numel()))
        chunk = torch.zeros((int(slots),), dtype=torch.float32)
        if int(stop) > int(start):
            chunk[: int(stop - start)] = self._bias_vector[int(start) : int(stop)]
        ptxt = scheme.encode(chunk, level=int(level), scale=int(scheme.params.get_default_scale()))
        self._bias_ptxt_cache[cache_key] = ptxt
        return ptxt

    def _construct_bias_vector(self) -> torch.Tensor:
        n, channels, height, width = (int(value) for value in getattr(self.module, "output_shape"))
        on_channels, on_height, on_width = (int(value) for value in getattr(self.module, "fhe_output_shape")[1:])
        output_gap = max(1, int(getattr(self.module, "output_gap")))
        output_layout = self._output_halo_layout()
        alpha = max(0, int(output_layout.get("alpha", 0)))
        beta = max(0, int(output_layout.get("beta", 0)))
        total_height = int(height + alpha + beta)
        bias = getattr(self.module, "on_bias").to(dtype=torch.float32)
        clear = bias.reshape(1, int(channels), 1, 1).expand(int(n), int(channels), int(total_height), int(width))
        multiplexed = packing.multiplex(clear, int(output_gap)).squeeze(0)
        bias_vector = torch.zeros((int(on_channels), int(on_height), int(on_width)), dtype=torch.float32)
        m_channels, m_height, m_width = (int(value) for value in multiplexed.shape)
        bias_vector[: int(m_channels), : int(m_height), : int(m_width)] = multiplexed
        return bias_vector.flatten().repeat(int(n))

    def _compile_bias_plaintexts(self, *, scheme: Any, level: int) -> None:
        for output_block in range(int(self.output_block_count)):
            self._bias_plaintext(scheme=scheme, level=int(level), output_block=int(output_block))

    def _apply_output_fold_rotations(self, block_ct: Any) -> Any:
        rotations = int(self.output_fold_rotations)
        if int(rotations) <= 0:
            return block_ct
        fold_started = time.time()
        slots = int(block_ct.scheme.params.get_slots())
        out = block_ct
        for rotation_index in range(1, int(rotations) + 1):
            out = out + out.roll(int(slots // (2**int(rotation_index))), in_place=False)
            self.last_runtime_counts["output_fold_rotation_count"] += 1
        self.last_runtime_timing["output_fold_s"] += float(time.time() - fold_started)
        return out

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
        self.last_runtime_io["disable_tile_family_sharing"] = bool(self.disable_tile_family_sharing)
        self.last_runtime_io["compiled_transform_count"] = int(self.compiled_transform_count)
        self.last_runtime_io["skipped_empty_transform_count"] = int(self.skipped_empty_transform_count)
        self.last_runtime_io["hybrid_pair_count"] = int(self.hybrid_pair_count)
        self.last_runtime_io["hybrid_pair_rejected_count"] = int(self.hybrid_pair_rejected_count)
        self.last_runtime_io["hybrid_pair_reject_reasons"] = [str(value) for value in self.hybrid_pair_reject_reasons]
        self.last_runtime_io["hybrid_pair_schedule_padded_count"] = int(self.hybrid_pair_schedule_padded_count)
        self.last_runtime_io["hybrid_pair_schedule_pad_reasons"] = [
            str(value) for value in self.hybrid_pair_schedule_pad_reasons
        ]
        self.last_runtime_io["hybrid_pair_layout_strategy"] = str(self.hybrid_pair_layout_strategy)
        self.last_runtime_io["hybrid_pair_layout_strict_pair_count"] = int(
            self.hybrid_pair_layout_strict_pair_count
        )
        self.last_runtime_io["hybrid_pair_layout_covered_output_count"] = int(
            self.hybrid_pair_layout_covered_output_count
        )
        self.last_runtime_io["hybrid_pair_layout_reject_reasons"] = [
            str(value) for value in self.hybrid_pair_layout_reject_reasons
        ]
        self.last_runtime_io["complex_input_block_flags"] = [bool(value) for value in self.complex_input_block_flags]
        self.last_runtime_io["output_fold_block_height"] = int(self.output_fold_block_height)
        self.last_runtime_io["output_fold_rotations"] = int(self.output_fold_rotations)
        if len(ids) < int(self.input_block_count):
            raise RuntimeError(
                "U22 experimental tconv kernel requires one ciphertext per packed input block: "
                f"expected {self.input_block_count}, got {len(ids)}"
            )
        if not self.groups or not (
            len(self.groups)
            == len(self.target_indices_by_input_unit)
            == len(self.input_block_pairs)
            == len(self.complex_input_block_flags)
        ):
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
                if bool(self.complex_input_block_flags[int(input_unit)]):
                    real_extract_started = time.time()
                    conj = block_ct.conjugate(in_place=False)
                    block_ct, conj = _align_ciphertexts_for_add(block_ct, conj)
                    block_ct = block_ct + conj
                    self.last_runtime_timing["real_extract_s"] += float(time.time() - real_extract_started)
                    self.last_runtime_counts["conjugate_count"] += 1
                    self.last_runtime_counts["real_extract_count"] += 1
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
        for output_block, block_ct in enumerate(accumulated):
            if block_ct is None:
                raise RuntimeError(f"U22 experimental tconv kernel missing output block {output_block}")
            block_ct = self._apply_output_fold_rotations(block_ct)
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
        layout_policy: str = "dp",
        _wrap_layout_policy: bool = True,
    ) -> "U22CompileRegistry":
        groups: list[RegionFirstRuntimeGroup] = []
        excluded_nodes: list[dict[str, str]] = []
        allowed = None if allowed_nodes is None else {str(value) for value in allowed_nodes}
        normalized_layout_policy = _normalize_u22_layout_policy(str(layout_policy or "dp"))
        from orion.experimental.layout_policy_ablation import build_layout_policy_compile_plan

        layout_compile_plan = build_layout_policy_compile_plan(dag, policy=str(normalized_layout_policy))
        executable_layout_plan = _layout_policy_runtime_compile_plan(layout_compile_plan)
        if (
            bool(_wrap_layout_policy)
            and bool(enable_conv_kernels)
            and normalized_layout_policy in {"fixed_max", "eager", "greedy", "dp"}
        ):
            provider_registry = cls.for_dag(
                dag,
                allowed_nodes=allowed_nodes,
                enable_conv_kernels=enable_conv_kernels,
                layout_policy="dp",
                _wrap_layout_policy=False,
            )
            executable_layout_plan = _layout_policy_with_native_physical_output_counts(
                executable_layout_plan,
                tuple(provider_registry.groups),
            )
            groups = [
                _layout_policy_wrap_provider_group(group, compile_plan=executable_layout_plan)
                for group in provider_registry.groups
            ]
            runtime_lowerings = {
                str(group.plan.get("runtime_lowering", ""))
                for group in groups
            }
            runtime_lowering = (
                next(iter(runtime_lowerings))
                if len(runtime_lowerings) == 1
                else "provider_executable+mixed_layout_policy"
            )
            native_halo_count = sum(
                1
                for group in groups
                if bool(group.plan.get("native_halo_provider", False))
            )
            provider_audit = dict(provider_registry.graph_audit)
            selected_tconv_count = int(provider_audit.get("selected_tconv_count", 0))
            selected_conv_count = int(provider_audit.get("selected_conv_count", 0))
            selected_pool_count = int(provider_audit.get("selected_pool_count", 0))
            selected_generic_conv_count = int(provider_audit.get("selected_generic_conv_count", 0))
            excluded_nodes = [dict(row) for row in list(provider_audit.get("excluded_nodes", []))]
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
                    "layout_policy_compile_plan_region_count": int(len(groups)),
                    "layout_policy_provider_executable_region_count": int(len(groups)),
                    "layout_policy_backend_executable_region_count": 0,
                    "layout_policy_native_halo_provider_region_count": int(native_halo_count),
                    "allowed_nodes": None if allowed_nodes is None else [str(value) for value in allowed_nodes],
                    "enable_conv_kernels": bool(enable_conv_kernels),
                    "excluded_nodes": excluded_nodes,
                    **_layout_policy_audit_fields(
                        executable_layout_plan,
                        runtime="provider_executable_layout_policy",
                        runtime_lowering=str(runtime_lowering),
                    ),
                },
            )
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
                        executor=TconvK2S2PythonRuntimeExecutor(
                            module=module,
                            output_node_id=str(node),
                            use_ct_pt_hybrid_packing=False,
                        ),
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
                groups.append(
                    _u22_pool_group(
                        node=str(node),
                        module=module,
                    )
                )
                selected_pool_count += 1
            elif bool(enable_conv_kernels) and isinstance(module, Conv2d):
                if not _u22_same_shape_conv_module_supported(module):
                    if _u22_input_pair_conv_module_supported(module):
                        stage = _u22_input_pair_conv_stage(node=str(node), module=module)
                        groups.append(
                            _u22_input_pair_conv_group(
                                node=str(node),
                                module=module,
                                stage=str(stage),
                            )
                        )
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
                groups.append(
                    _u22_same_shape_conv_group(
                        node=str(node),
                        module=module,
                    )
                )
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
                "layout_policy_compile_plan_region_count": int(len(groups)),
                "allowed_nodes": None if allowed_nodes is None else [str(value) for value in allowed_nodes],
                "enable_conv_kernels": bool(enable_conv_kernels),
                "excluded_nodes": excluded_nodes,
                **_layout_policy_audit_fields(
                    layout_compile_plan,
                    runtime="provider_executable",
                    runtime_lowering="provider_executable",
                ),
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
        compile_plan: dict[str, Any] | None = None
        for group in self.groups:
            executor = getattr(group, "executor", None)
            plan = getattr(executor, "compile_plan", None)
            if isinstance(plan, dict) and plan.get("edge_layouts"):
                compile_plan = dict(plan)
                break
        add_attached: list[dict[str, Any]] = []
        if compile_plan is not None:
            for node in dag.topological_sort():
                module = dag.nodes[node].get("module")
                if type(module).__name__ != "Add":
                    continue
                rows = _layout_policy_add_rows(compile_plan, node=str(node))
                if not rows:
                    continue
                has_relayout = any(bool(row.get("relayout", False)) for row in rows)
                has_materialized_halo = any(
                    _layout_policy_has_halo(dict(row.get("selected_layout", {}) or {}))
                    for row in rows
                )
                if not bool(has_relayout or has_materialized_halo):
                    continue
                fx_node = dag.nodes[node].get("fx_node")
                fx_inputs = tuple(str(value.name) for value in getattr(fx_node, "all_input_nodes", ()) or ())
                input_sources = fx_inputs if fx_inputs else tuple(str(value) for value in dag.predecessors(node))
                runtime = LayoutPolicyAddRuntimeExecutor(
                    node=str(node),
                    compile_plan=compile_plan,
                    input_sources=tuple(input_sources),
                )
                module.layout_policy_add_runtime = runtime
                relayout_depth = max(
                    [0]
                    + [
                        int(row.get("relayout_depth_estimate", 0) or 0)
                        for row in rows
                        if bool(row.get("relayout", False))
                    ]
                )
                if hasattr(module, "set_depth"):
                    module.set_depth(max(int(getattr(module, "depth", 0) or 0), int(relayout_depth)))
                add_attached.append(
                    {
                        "node": str(node),
                        "input_sources": [str(value) for value in input_sources],
                        "relayout_edge_count": int(sum(1 for row in rows if bool(row.get("relayout", False)))),
                        "materialized_halo": bool(has_materialized_halo),
                    }
                )
        return {
            "attached_count": int(len(attached)),
            "attached": attached,
            "layout_policy_add_runtime_count": int(len(add_attached)),
            "layout_policy_add_runtimes": add_attached,
            "executable_region_count": int(sum(1 for group in self.groups if bool(group.executable))),
            "graph_audit": dict(self.graph_audit),
        }
