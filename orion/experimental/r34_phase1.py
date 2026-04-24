"""Minimal R34 ImageNet phase-1 bridge for Orion.

This module intentionally keeps two migration tracks separate:

1. Layout contracts imported from HaloED's R34 layout/planner workflow.
2. Kernel/provider bindings that tell Orion which lowering/runtime path should
   consume a given family.

Phase 1 is deliberately narrow. It does not port HaloED's full layout search
or the full scripts/cir selector/executor stack. Instead, it introduces:

* a stable imported layout registry for selected R34 representative nodes
* explicit family-to-provider bindings
* a small Orion-side report surface that shows how imported layouts and
  provider bindings meet at a narrow materializer contract
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
import json
import time

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core import packing
from orion.core.region_experiments import build_region_experiments
from orion.core.shared_lt import PackingPlanner, RegionNode, RegionPlanner
from orion.experimental.cir.r34_orion_same_shape import (
    R34InterGroupHybridSameShapeRuntimeExecutor,
    R34IntraGroupPack2SameShapeRuntimeExecutor,
    R34Pack2SameShapeRuntimeExecutor,
    R34SameShapeStageSpec,
    _maybe_set_default_scale,
    r34_same_shape_policy,
    r34_same_shape_policy_from_source_group_count,
    r34_source_group_count,
    r34_same_shape_spec_for_family_label,
)
from orion.experimental.cir.r34_inter_group_python import R34PythonTransitionFlowRuntimeExecutor
from orion.experimental.cir.region_first_data import STAGE_MATERIALIZER_REFERENCES
from orion.experimental.cir.runtime_group import RegionFirstRuntimeGroup
from orion.nn.unified_transform import UnifiedTransformGroup


ProviderKind = Literal["scripts_cir_same_shape", "tile_local_transition_bridge", "python_single_flow"]
Phase1Status = Literal["bound", "implemented"]


@dataclass(frozen=True)
class ImportedLayoutContract:
    haloed_node: str
    orion_node: str
    family_id: str
    family_kind: str
    family_label: str
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    weight_shape: tuple[int, int, int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    input_layout: dict[str, int]
    output_layout: dict[str, int]
    num_tiles: int
    exec_mode: str
    planner_diagnostics: dict[str, Any]
    planner_source: str = "haloed.scripts.cir.resnet34_imgnet_offline_cost_predictor"

    @property
    def input_gap(self) -> int:
        return int(self.input_layout.get("stride", 1))

    @property
    def output_gap(self) -> int:
        return int(self.output_layout.get("stride", 1))

    @property
    def kernel_size(self) -> tuple[int, int]:
        return int(self.weight_shape[-2]), int(self.weight_shape[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "haloed_node": str(self.haloed_node),
            "orion_node": str(self.orion_node),
            "family_id": str(self.family_id),
            "family_kind": str(self.family_kind),
            "family_label": str(self.family_label),
            "input_shape": list(int(v) for v in self.input_shape),
            "output_shape": list(int(v) for v in self.output_shape),
            "weight_shape": list(int(v) for v in self.weight_shape),
            "stride": list(int(v) for v in self.stride),
            "padding": list(int(v) for v in self.padding),
            "dilation": list(int(v) for v in self.dilation),
            "groups": int(self.groups),
            "input_layout": dict(self.input_layout),
            "output_layout": dict(self.output_layout),
            "input_gap": int(self.input_gap),
            "output_gap": int(self.output_gap),
            "num_tiles": int(self.num_tiles),
            "exec_mode": str(self.exec_mode),
            "planner_diagnostics": dict(self.planner_diagnostics),
            "planner_source": str(self.planner_source),
        }


@dataclass(frozen=True)
class KernelBinding:
    family_label: str
    provider_key: str
    provider_kind: ProviderKind
    materializer: str
    phase1_status: Phase1Status
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_label": str(self.family_label),
            "provider_key": str(self.provider_key),
            "provider_kind": str(self.provider_kind),
            "materializer": str(self.materializer),
            "phase1_status": str(self.phase1_status),
            "note": str(self.note),
        }


_R34_IMPORTED_LAYOUT_CONTRACTS: tuple[ImportedLayoutContract, ...] = (
    ImportedLayoutContract(
        haloed_node="stem_conv1_torch",
        orion_node="conv1",
        family_id="stem_conv7x7_s2_gap1_to2",
        family_kind="stem_conv_family",
        family_label="stem_conv",
        input_shape=(3, 224, 224),
        output_shape=(64, 112, 112),
        weight_shape=(64, 3, 7, 7),
        stride=(2, 2),
        padding=(3, 3),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 224, "beta": 0, "c": 3, "h": 224, "stride": 1, "w": 224},
        output_layout={"alpha": 112, "beta": 0, "c": 64, "h": 112, "stride": 2, "w": 112},
        num_tiles=16,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 0.0,
            "best_explored_mode": "halo",
            "candidate_count": 1,
            "chosen_input_layout": {"alpha": 224, "beta": 0, "stride": 1},
            "chosen_layout_cost": 0.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 112, "beta": 0, "stride": 2},
            "conv_rotation_cost": 0,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 1,
            "planned_output_stride": 2,
            "planner_cost_model": "phase1_stem_fixture",
            "planner_selected_backend": "phase1_stem_fixture",
            "regret_ratio": 1.0,
            "runtime_input_stride": 1,
            "runtime_output_stride": 2,
            "structural_oracle_used": False,
        },
        planner_source="orion.experimental.r34_stem_pool_phase1",
    ),
    ImportedLayoutContract(
        haloed_node="stem_pool_torch",
        orion_node="pool",
        family_id="stem_pool3x3_s2_gap2_to4",
        family_kind="stem_pool_family",
        family_label="stem_pool",
        input_shape=(64, 112, 112),
        output_shape=(64, 56, 56),
        weight_shape=(64, 1, 3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=64,
        input_layout={"alpha": 112, "beta": 0, "c": 64, "h": 112, "stride": 2, "w": 112},
        output_layout={"alpha": 56, "beta": 0, "c": 64, "h": 56, "stride": 4, "w": 56},
        num_tiles=8,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 0.0,
            "best_explored_mode": "halo",
            "candidate_count": 1,
            "chosen_input_layout": {"alpha": 112, "beta": 0, "stride": 2},
            "chosen_layout_cost": 0.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 56, "beta": 0, "stride": 4},
            "conv_rotation_cost": 0,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 2,
            "planned_output_stride": 4,
            "planner_cost_model": "phase1_stem_fixture",
            "planner_selected_backend": "phase1_stem_fixture",
            "regret_ratio": 1.0,
            "runtime_input_stride": 2,
            "runtime_output_stride": 4,
            "structural_oracle_used": False,
        },
        planner_source="orion.experimental.r34_stem_pool_phase1",
    ),
    ImportedLayoutContract(
        haloed_node="global_avgpool_exit_torch",
        orion_node="avgpool",
        family_id="global_avgpool_exit_gap32_to_dense",
        family_kind="global_pool_exit_family",
        family_label="global_avgpool_exit",
        input_shape=(512, 7, 7),
        output_shape=(512, 1, 1),
        weight_shape=(512, 1, 7, 7),
        stride=(7, 7),
        padding=(0, 0),
        dilation=(1, 1),
        groups=512,
        input_layout={"alpha": 2, "beta": 1, "c": 512, "h": 7, "stride": 32, "w": 7},
        output_layout={"alpha": 1, "beta": 0, "c": 512, "h": 1, "stride": 224, "w": 1},
        num_tiles=1,
        exec_mode="layout_exit",
        planner_diagnostics={
            "best_explored_layout_cost": 0.0,
            "best_explored_mode": "layout_exit",
            "candidate_count": 1,
            "chosen_input_layout": {"alpha": 2, "beta": 1, "stride": 32},
            "chosen_layout_cost": 0.0,
            "chosen_mode": "layout_exit",
            "chosen_output_layout": {"alpha": 1, "beta": 0, "stride": 224},
            "conv_rotation_cost": 0,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 32,
            "planned_output_stride": 224,
            "planner_cost_model": "phase1_stem_fixture",
            "planner_selected_backend": "phase1_stem_fixture",
            "regret_ratio": 1.0,
            "runtime_input_stride": 32,
            "runtime_output_stride": 224,
            "structural_oracle_used": False,
        },
        planner_source="orion.experimental.r34_stem_pool_phase1",
    ),
    ImportedLayoutContract(
        haloed_node="layer1_0_conv1_torch",
        orion_node="layers_0_0_conv1",
        family_id="source_pack2:64x56x56->64x56x56:k3x3:s1:gap4->4",
        family_kind="source_pack2",
        family_label="stage1_same",
        input_shape=(64, 56, 56),
        output_shape=(64, 56, 56),
        weight_shape=(64, 64, 3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 36, "beta": 0, "c": 64, "h": 56, "stride": 4, "w": 56},
        output_layout={"alpha": 36, "beta": 0, "c": 64, "h": 56, "stride": 4, "w": 56},
        num_tiles=8,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 5144.0,
            "best_explored_mode": "halo",
            "candidate_count": 12,
            "chosen_input_layout": {"alpha": 34, "beta": 1, "stride": 4},
            "chosen_layout_cost": 5144.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 36, "beta": 0, "stride": 4},
            "conv_rotation_cost": 5112,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 4,
            "planned_output_stride": 4,
            "planner_cost_model": "heuristic",
            "planner_selected_backend": "heuristic",
            "regret_ratio": 1.0,
            "runtime_input_stride": 4,
            "runtime_output_stride": 4,
            "structural_oracle_used": False,
        },
    ),
    ImportedLayoutContract(
        haloed_node="layer2_1_conv1_torch",
        orion_node="layers_1_1_conv1",
        family_id="source_pack2:128x28x28->128x28x28:k3x3:s1:gap8->8",
        family_kind="source_pack2",
        family_label="stage2_same",
        input_shape=(128, 28, 28),
        output_shape=(128, 28, 28),
        weight_shape=(128, 128, 3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 15, "beta": 0, "c": 128, "h": 28, "stride": 8, "w": 28},
        output_layout={"alpha": 15, "beta": 0, "c": 128, "h": 28, "stride": 8, "w": 28},
        num_tiles=4,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 5136.0,
            "best_explored_mode": "halo",
            "candidate_count": 13,
            "chosen_input_layout": {"alpha": 15, "beta": 1, "stride": 8},
            "chosen_layout_cost": 5136.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 15, "beta": 0, "stride": 8},
            "conv_rotation_cost": 5116,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 8,
            "planned_output_stride": 8,
            "planner_cost_model": "heuristic",
            "planner_selected_backend": "heuristic",
            "regret_ratio": 1.0,
            "runtime_input_stride": 8,
            "runtime_output_stride": 8,
            "structural_oracle_used": False,
        },
    ),
    ImportedLayoutContract(
        haloed_node="layer3_1_conv1_torch",
        orion_node="layers_2_1_conv1",
        family_id="source_pack2:256x14x14->256x14x14:k3x3:s1:gap16->16",
        family_kind="source_pack2",
        family_label="stage3_same",
        input_shape=(256, 14, 14),
        output_shape=(256, 14, 14),
        weight_shape=(256, 256, 3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 7, "beta": 0, "c": 256, "h": 14, "stride": 16, "w": 14},
        output_layout={"alpha": 7, "beta": 0, "c": 256, "h": 14, "stride": 16, "w": 14},
        num_tiles=2,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 5148.0,
            "best_explored_mode": "halo",
            "candidate_count": 10,
            "chosen_input_layout": {"alpha": 7, "beta": 1, "stride": 16},
            "chosen_layout_cost": 5148.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 7, "beta": 0, "stride": 16},
            "conv_rotation_cost": 5118,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 16,
            "planned_output_stride": 16,
            "planner_cost_model": "heuristic",
            "planner_selected_backend": "heuristic",
            "regret_ratio": 1.0,
            "runtime_input_stride": 16,
            "runtime_output_stride": 16,
            "structural_oracle_used": False,
        },
    ),
    ImportedLayoutContract(
        haloed_node="layer4_1_conv1_torch",
        orion_node="layers_3_1_conv1",
        family_id="stage4_same_3x3_s1_gap32_to32",
        family_kind="source_pack2",
        family_label="stage4_same",
        input_shape=(512, 7, 7),
        output_shape=(512, 7, 7),
        weight_shape=(512, 512, 3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 2, "beta": 1, "c": 512, "h": 7, "stride": 32, "w": 7},
        output_layout={"alpha": 2, "beta": 1, "c": 512, "h": 7, "stride": 32, "w": 7},
        num_tiles=4,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 0.0,
            "best_explored_mode": "halo",
            "candidate_count": 1,
            "chosen_input_layout": {"alpha": 2, "beta": 1, "stride": 32},
            "chosen_layout_cost": 0.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 2, "beta": 1, "stride": 32},
            "conv_rotation_cost": 0,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 32,
            "planned_output_stride": 32,
            "planner_cost_model": "pairing_search_fixture",
            "planner_selected_backend": "pairing_search_fixture",
            "regret_ratio": 1.0,
            "runtime_input_stride": 32,
            "runtime_output_stride": 32,
            "structural_oracle_used": False,
        },
        planner_source="haloed.scripts.cir.source_pack2_pairing_search",
    ),
    ImportedLayoutContract(
        haloed_node="layer2_0_conv1_torch",
        orion_node="layers_1_0_conv1",
        family_id="resnet34_layer2_0_conv1_torch",
        family_kind="transition_family_baseline",
        family_label="stage2_transition",
        input_shape=(64, 56, 56),
        output_shape=(128, 28, 28),
        weight_shape=(128, 64, 3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 36, "beta": 0, "c": 64, "h": 56, "stride": 4, "w": 56},
        output_layout={"alpha": 15, "beta": 0, "c": 128, "h": 28, "stride": 8, "w": 28},
        num_tiles=4,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 0.0,
            "best_explored_mode": "halo",
            "candidate_count": 1,
            "chosen_input_layout": {"alpha": 36, "beta": 0, "stride": 4},
            "chosen_layout_cost": 0.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 15, "beta": 0, "stride": 8},
            "conv_rotation_cost": 0,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 4,
            "planned_output_stride": 8,
            "planner_cost_model": "phase1_transition_fixture",
            "planner_selected_backend": "phase1_transition_fixture",
            "regret_ratio": 1.0,
            "runtime_input_stride": 4,
            "runtime_output_stride": 8,
            "structural_oracle_used": False,
        },
    ),
    ImportedLayoutContract(
        haloed_node="layer2_0_downsample_conv_torch",
        orion_node="layers_1_0_shortcut_0",
        family_id="resnet34_layer2_0_downsample_conv_torch",
        family_kind="transition_family_baseline",
        family_label="stage2_transition",
        input_shape=(64, 56, 56),
        output_shape=(128, 28, 28),
        weight_shape=(128, 64, 1, 1),
        stride=(2, 2),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 36, "beta": 0, "c": 64, "h": 56, "stride": 4, "w": 56},
        output_layout={"alpha": 15, "beta": 0, "c": 128, "h": 28, "stride": 8, "w": 28},
        num_tiles=4,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 0.0,
            "best_explored_mode": "halo",
            "candidate_count": 1,
            "chosen_input_layout": {"alpha": 36, "beta": 0, "stride": 4},
            "chosen_layout_cost": 0.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 15, "beta": 0, "stride": 8},
            "conv_rotation_cost": 0,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 4,
            "planned_output_stride": 8,
            "planner_cost_model": "phase1_transition_fixture",
            "planner_selected_backend": "phase1_transition_fixture",
            "regret_ratio": 1.0,
            "runtime_input_stride": 4,
            "runtime_output_stride": 8,
            "structural_oracle_used": False,
        },
    ),
    ImportedLayoutContract(
        haloed_node="layer3_0_conv1_torch",
        orion_node="layers_2_0_conv1",
        family_id="resnet34_layer3_0_conv1_torch",
        family_kind="transition_family_baseline",
        family_label="stage3_transition",
        input_shape=(128, 28, 28),
        output_shape=(256, 14, 14),
        weight_shape=(256, 128, 3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 28, "beta": 0, "c": 128, "h": 28, "stride": 8, "w": 28},
        output_layout={"alpha": 7, "beta": 0, "c": 256, "h": 14, "stride": 16, "w": 14},
        num_tiles=2,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 159998.0,
            "best_explored_mode": "no_halo_exchange",
            "candidate_count": 3,
            "chosen_input_layout": {"alpha": 15, "beta": 1, "stride": 8},
            "chosen_layout_cost": 180738.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 7, "beta": 0, "stride": 16},
            "conv_rotation_cost": 180732,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 8,
            "planned_output_stride": 16,
            "planner_cost_model": "heuristic",
            "planner_selected_backend": "heuristic",
            "regret_ratio": 1.129626620332754,
            "runtime_input_stride": 8,
            "runtime_output_stride": 16,
            "structural_oracle_used": False,
        },
    ),
    ImportedLayoutContract(
        haloed_node="layer4_0_conv1_torch",
        orion_node="layers_3_0_conv1",
        family_id="resnet34_layer4_0_conv1_torch",
        family_kind="transition_family_baseline",
        family_label="stage4_transition",
        input_shape=(256, 14, 14),
        output_shape=(512, 7, 7),
        weight_shape=(512, 256, 3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 7, "beta": 0, "c": 256, "h": 14, "stride": 16, "w": 14},
        output_layout={"alpha": 2, "beta": 1, "c": 512, "h": 7, "stride": 32, "w": 7},
        num_tiles=2,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 0.0,
            "best_explored_mode": "halo",
            "candidate_count": 1,
            "chosen_input_layout": {"alpha": 7, "beta": 0, "stride": 16},
            "chosen_layout_cost": 0.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 2, "beta": 1, "stride": 32},
            "conv_rotation_cost": 0,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 16,
            "planned_output_stride": 32,
            "planner_cost_model": "phase1_transition_fixture",
            "planner_selected_backend": "phase1_transition_fixture",
            "regret_ratio": 1.0,
            "runtime_input_stride": 16,
            "runtime_output_stride": 32,
            "structural_oracle_used": False,
        },
    ),
    ImportedLayoutContract(
        haloed_node="layer4_0_downsample_conv_torch",
        orion_node="layers_3_0_shortcut_0",
        family_id="resnet34_layer4_0_downsample_conv_torch",
        family_kind="transition_family_baseline",
        family_label="stage4_transition",
        input_shape=(256, 14, 14),
        output_shape=(512, 7, 7),
        weight_shape=(512, 256, 1, 1),
        stride=(2, 2),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 7, "beta": 0, "c": 256, "h": 14, "stride": 16, "w": 14},
        output_layout={"alpha": 2, "beta": 1, "c": 512, "h": 7, "stride": 32, "w": 7},
        num_tiles=2,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 0.0,
            "best_explored_mode": "halo",
            "candidate_count": 1,
            "chosen_input_layout": {"alpha": 7, "beta": 0, "stride": 16},
            "chosen_layout_cost": 0.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 2, "beta": 1, "stride": 32},
            "conv_rotation_cost": 0,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 16,
            "planned_output_stride": 32,
            "planner_cost_model": "phase1_transition_fixture",
            "planner_selected_backend": "phase1_transition_fixture",
            "regret_ratio": 1.0,
            "runtime_input_stride": 16,
            "runtime_output_stride": 32,
            "structural_oracle_used": False,
        },
    ),
    ImportedLayoutContract(
        haloed_node="layer3_0_downsample_conv_torch",
        orion_node="layers_2_0_shortcut_0",
        family_id="resnet34_layer3_0_downsample_conv_torch",
        family_kind="transition_family_baseline",
        family_label="stage3_transition",
        input_shape=(128, 28, 28),
        output_shape=(256, 14, 14),
        weight_shape=(256, 128, 1, 1),
        stride=(2, 2),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        input_layout={"alpha": 28, "beta": 0, "c": 128, "h": 28, "stride": 8, "w": 28},
        output_layout={"alpha": 7, "beta": 0, "c": 256, "h": 14, "stride": 16, "w": 14},
        num_tiles=2,
        exec_mode="halo",
        planner_diagnostics={
            "best_explored_layout_cost": 58366.0,
            "best_explored_mode": "halo",
            "candidate_count": 3,
            "chosen_input_layout": {"alpha": 15, "beta": 0, "stride": 8},
            "chosen_layout_cost": 58366.0,
            "chosen_mode": "halo",
            "chosen_output_layout": {"alpha": 7, "beta": 0, "stride": 16},
            "conv_rotation_cost": 58364,
            "exact_oracle_used": False,
            "manual_exchange_rotations": 0,
            "planned_input_stride": 8,
            "planned_output_stride": 16,
            "planner_cost_model": "heuristic",
            "planner_selected_backend": "heuristic",
            "regret_ratio": 1.0,
            "runtime_input_stride": 8,
            "runtime_output_stride": 16,
            "structural_oracle_used": False,
        },
    ),
)


_PHASE1_KERNEL_BINDINGS: tuple[KernelBinding, ...] = (
    KernelBinding(
        family_label="stem_conv",
        provider_key="r34_stem_conv_inter_group_hybrid_policy",
        provider_kind="python_single_flow",
        materializer="policy_inter_group_hybrid",
        phase1_status="implemented",
        note="Stem conv7x7 currently uses the Python single-flow runtime under the same source-group policy; non-Python backends still fall back.",
    ),
    KernelBinding(
        family_label="stem_pool",
        provider_key="r34_stem_pool_inter_group_hybrid_policy",
        provider_kind="python_single_flow",
        materializer="policy_inter_group_hybrid_pool",
        phase1_status="implemented",
        note="Stem pool3x3 currently uses the Python single-flow runtime under the same source-group policy; non-Python backends still fall back.",
    ),
    KernelBinding(
        family_label="global_avgpool_exit",
        provider_key="r34_global_avgpool_exit_policy",
        provider_kind="python_single_flow",
        materializer="policy_global_avgpool_exit",
        phase1_status="implemented",
        note="Global avgpool exit currently uses a Python single-flow runtime that returns to Orion's existing avgpool output layout; non-Python backends still fall back.",
    ),
    KernelBinding(
        family_label="stage1_same",
        provider_key="r34_stage1_same_inter_group_hybrid_policy",
        provider_kind="scripts_cir_same_shape",
        materializer="policy_inter_group_hybrid",
        phase1_status="implemented",
        note="Selected by policy: source_group_count > 1 uses inter_group_hybrid. Current Orion runtime implementation is enabled on the Python backend and still falls back on non-Python backends.",
    ),
    KernelBinding(
        family_label="stage2_same",
        provider_key="r34_stage2_same_inter_group_hybrid_policy",
        provider_kind="scripts_cir_same_shape",
        materializer="policy_inter_group_hybrid",
        phase1_status="implemented",
        note="Selected by policy: source_group_count > 1 uses inter_group_hybrid. Current Orion runtime implementation is enabled on the Python backend and still falls back on non-Python backends.",
    ),
    KernelBinding(
        family_label="stage3_same",
        provider_key="r34_stage3_same_intra_group_pack2_policy",
        provider_kind="scripts_cir_same_shape",
        materializer="policy_intra_group_pack2",
        phase1_status="implemented",
        note="Selected by policy: source_group_count == 1 uses intra_group_pack2.",
    ),
    KernelBinding(
        family_label="stage4_same",
        provider_key="r34_stage4_same_intra_group_pack2_policy",
        provider_kind="scripts_cir_same_shape",
        materializer="policy_intra_group_pack2",
        phase1_status="implemented",
        note="Selected by policy: source_group_count == 1 uses intra_group_pack2.",
    ),
    KernelBinding(
        family_label="stage2_transition",
        provider_key="r34_stage2_transition_inter_group_hybrid_policy",
        provider_kind="tile_local_transition_bridge",
        materializer="policy_inter_group_hybrid_transition",
        phase1_status="implemented",
        note="Selected by policy: source_group_count > 1 uses inter_group_hybrid on transition branches. Current Orion runtime implementation is enabled on the Python backend and still falls back on non-Python backends.",
    ),
    KernelBinding(
        family_label="stage3_transition",
        provider_key="r34_stage3_transition_inter_group_hybrid_policy",
        provider_kind="tile_local_transition_bridge",
        materializer="policy_inter_group_hybrid_transition",
        phase1_status="implemented",
        note="Selected by policy: source_group_count > 1 uses inter_group_hybrid on transition branches.",
    ),
    KernelBinding(
        family_label="stage4_transition",
        provider_key="r34_stage4_transition_intra_group_pack2_policy",
        provider_kind="tile_local_transition_bridge",
        materializer="policy_intra_group_pack2_transition",
        phase1_status="implemented",
        note="Selected by policy: source_group_count == 1 uses intra_group_pack2 on transition branches. Current Orion runtime implementation is enabled on the Python backend and still falls back on non-Python backends.",
    ),
)


DEFAULT_R34_PHASE1_REPORT_OUT = Path("/tmp/orion_r34_phase1_report.json")


def imported_layout_contracts() -> tuple[ImportedLayoutContract, ...]:
    return tuple(_R34_IMPORTED_LAYOUT_CONTRACTS)


def kernel_bindings() -> tuple[KernelBinding, ...]:
    return tuple(_PHASE1_KERNEL_BINDINGS)


def imported_layout_contract_by_haloed_node(node_name: str) -> ImportedLayoutContract:
    for contract in _R34_IMPORTED_LAYOUT_CONTRACTS:
        if str(contract.haloed_node) == str(node_name):
            return contract
    raise KeyError(f"unknown HaloED R34 phase1 node {node_name!r}")


def imported_layout_contract_by_orion_node(node_name: str) -> ImportedLayoutContract:
    for contract in _R34_IMPORTED_LAYOUT_CONTRACTS:
        if str(contract.orion_node) == str(node_name):
            return contract
    raise KeyError(f"unknown Orion R34 phase1 node {node_name!r}")


def imported_layout_contract_by_family_label(family_label: str) -> ImportedLayoutContract:
    for contract in _R34_IMPORTED_LAYOUT_CONTRACTS:
        if str(contract.family_label) == str(family_label):
            return contract
    raise KeyError(f"unknown R34 phase1 family label {family_label!r}")


def kernel_binding_for_family(family_label: str) -> KernelBinding:
    for binding in _PHASE1_KERNEL_BINDINGS:
        if str(binding.family_label) == str(family_label):
            return binding
    raise KeyError(f"no phase1 kernel binding for family {family_label!r}")


def materializer_attrs_from_contract(
    contract: ImportedLayoutContract,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> dict[str, Any]:
    expected_shape = tuple(int(v) for v in contract.weight_shape)
    actual_shape = tuple(int(v) for v in tuple(weight.shape))
    if actual_shape != expected_shape:
        raise ValueError(f"weight for {contract.haloed_node} has shape {actual_shape}, expected {expected_shape}")
    if bias is not None and int(bias.numel()) != int(contract.output_shape[0]):
        raise ValueError(
            f"bias for {contract.haloed_node} has {int(bias.numel())} entries, expected {int(contract.output_shape[0])}"
        )
    return {
        "weight": weight.detach().to(device="cpu"),
        "bias": None if bias is None else bias.detach().to(device="cpu"),
        "stride": tuple(int(v) for v in contract.stride),
        "padding": tuple(int(v) for v in contract.padding),
        "dilation": tuple(int(v) for v in contract.dilation),
        "groups": int(contract.groups),
        "input_layout": dict(contract.input_layout),
        "layout": dict(contract.output_layout),
        "bsgs": True,
        "bsgs_n1": 0,
        "bsgs_n1_max": 512,
        "bsgs_hoist_giant_steps": True,
        "_node_name": str(contract.haloed_node),
        "module_target": str(contract.orion_node),
    }


def _same_shape_surface(contract: ImportedLayoutContract) -> dict[str, Any]:
    try:
        target_tiles = PackingPlanner.build_target_tiles(
            c=int(contract.output_shape[0]),
            h=int(contract.output_shape[1]),
            w=int(contract.output_shape[2]),
            gap=int(contract.output_gap),
            max_slots=32768,
        )
        source_tiles = PackingPlanner.build_source_tiles_for_targets(
            target_tiles=target_tiles,
            c_in=int(contract.input_shape[0]),
            h_in=int(contract.input_shape[1]),
            w_in=int(contract.input_shape[2]),
            input_gap=int(contract.input_gap),
            kernel=int(contract.kernel_size[0]),
            stride=int(contract.stride[0]),
            pad=int(contract.padding[0]),
            max_slots=32768,
        )
        orion_surface = {
            "status": "ok",
            "source_tile_count": int(len(source_tiles)),
            "target_tile_count": int(len(target_tiles)),
            "max_source_active_slots": int(max((tile.active_slots for tile in source_tiles), default=0)),
            "max_target_active_slots": int(max((tile.active_slots for tile in target_tiles), default=0)),
            "kernel_size": list(int(v) for v in contract.kernel_size),
            "slot_budget": 32768,
            "surface_builder": "orion.PackingPlanner.build_target_tiles + build_source_tiles_for_targets",
        }
    except ValueError as exc:
        target_tiles = ()
        source_tiles = ()
        orion_surface = {
            "status": "needs_layout_aware_tiler",
            "reason": str(exc),
            "kernel_size": list(int(v) for v in contract.kernel_size),
            "slot_budget": 32768,
            "surface_builder": "orion.PackingPlanner.build_target_tiles + build_source_tiles_for_targets",
            "note": "Phase 1 keeps the imported HaloED layout contract even when Orion's simplified CH/halo tiler cannot yet replay it exactly.",
        }
    return {
        "haloed_node": str(contract.haloed_node),
        "orion_node": str(contract.orion_node),
        "family_label": str(contract.family_label),
        "provider": kernel_binding_for_family(contract.family_label).to_dict(),
        "input_gap": int(contract.input_gap),
        "output_gap": int(contract.output_gap),
        "imported_layout": {
            "input_layout": dict(contract.input_layout),
            "output_layout": dict(contract.output_layout),
            "num_tiles": int(contract.num_tiles),
            "exec_mode": str(contract.exec_mode),
        },
        "orion_surface": orion_surface,
    }


def build_r34_phase1_transition_bridge_plan() -> dict[str, Any]:
    conv = imported_layout_contract_by_haloed_node("layer3_0_conv1_torch")
    downsample = imported_layout_contract_by_haloed_node("layer3_0_downsample_conv_torch")
    if tuple(conv.input_shape) != tuple(downsample.input_shape):
        raise ValueError("transition representative inputs must share one input shape")
    if tuple(conv.output_shape) != tuple(downsample.output_shape):
        raise ValueError("transition representative outputs must share one output shape")
    if dict(conv.input_layout) != dict(downsample.input_layout):
        raise ValueError("transition representative inputs must share one imported input layout")
    if dict(conv.output_layout) != dict(downsample.output_layout):
        raise ValueError("transition representative outputs must share one imported output layout")

    region = RegionPlanner.discover_same_source_regions(
        (
            RegionNode(str(conv.orion_node), "linear", "r34_stage3_transition_input", f"{conv.orion_node}_out"),
            RegionNode(str(downsample.orion_node), "linear", "r34_stage3_transition_input", f"{downsample.orion_node}_out"),
        )
    )[0]
    lowered = PackingPlanner.lower_transition_region(
        region=region,
        c_in=int(conv.input_shape[0]),
        h_in=int(conv.input_shape[1]),
        w_in=int(conv.input_shape[2]),
        input_gap=int(conv.input_gap),
        c_out=int(conv.output_shape[0]),
        h_out=int(conv.output_shape[1]),
        w_out=int(conv.output_shape[2]),
        output_gap=int(conv.output_gap),
        kernel=int(max(conv.kernel_size[0], downsample.kernel_size[0])),
        stride=int(conv.stride[0]),
        pad=int(conv.padding[0]),
        max_slots=32768,
        use_real_imag_hybrid=True,
        strategy="real_imag_hybrid_transition_branch",
    )
    binding = kernel_binding_for_family("stage3_transition")
    return {
        "status": "ok",
        "binding": binding.to_dict(),
        "imported_contracts": [conv.to_dict(), downsample.to_dict()],
        "shared_layout_contract": {
            "input_shape": list(int(v) for v in conv.input_shape),
            "output_shape": list(int(v) for v in conv.output_shape),
            "input_layout": dict(conv.input_layout),
            "output_layout": dict(conv.output_layout),
            "source_of_truth": str(conv.planner_source),
        },
        "region": {
            "region_id": str(region.region_id),
            "source_input_id": str(region.source_input_id),
            "output_node_ids": list(str(v) for v in region.output_node_ids),
            "useful_output_banks": int(region.useful_output_banks),
        },
        "lowered": {
            "strategy": str(lowered.strategy),
            "relu_safe_boundary": bool(lowered.relu_safe_boundary),
            "boundary_actions": list(str(v) for v in lowered.boundary_actions),
            "source_packing": asdict(lowered.source_packing),
            "source_tile_count": int(len(lowered.source_tiles)),
            "target_tile_count": int(len(lowered.target_tiles)),
            "output_bank_count": int(len(lowered.output_banks)),
            "source_tiles": [
                {
                    "tile_id": str(tile.tile_id),
                    "c_start": int(tile.c_start),
                    "c_end": int(tile.c_end),
                    "h_start": int(tile.h_start),
                    "h_end": int(tile.h_end),
                    "w": int(tile.w),
                    "gap": int(tile.gap),
                    "halo_top": int(tile.halo_top),
                    "halo_bottom": int(tile.halo_bottom),
                    "active_slots": int(tile.active_slots),
                }
                for tile in lowered.source_tiles
            ],
            "target_tiles": [
                {
                    "tile_id": str(tile.tile_id),
                    "c_start": int(tile.c_start),
                    "c_end": int(tile.c_end),
                    "h_start": int(tile.h_start),
                    "h_end": int(tile.h_end),
                    "w": int(tile.w),
                    "gap": int(tile.gap),
                    "active_slots": int(tile.active_slots),
                }
                for tile in lowered.target_tiles
            ],
            "output_banks": [
                {
                    "bank_id": str(bank.bank_id),
                    "target_tile_id": str(bank.target_tile_id),
                    "kind": str(bank.kind),
                    "real_output_id": str(bank.real_output_id),
                    "imag_output_id": str(bank.imag_output_id),
                }
                for bank in lowered.output_banks
            ],
        },
    }


def _selected_r34_results() -> dict[str, Any]:
    experiments = build_region_experiments(networks=("R34",)).get("experiments", [])
    by_region = {str(row["region_id"]): dict(row) for row in experiments}
    same_shape = dict(by_region["stage1_stage2_same_shape"])
    transition = dict(by_region["stage2_stage3_stage4_transition_branch_regions"])
    stage_refs = [
        {
            "network": str(ref.network),
            "stage": str(ref.stage),
            "family": str(ref.family),
            "materializer": str(ref.materializer),
            "expected_stats": dict(ref.expected_stats),
            "source": str(ref.source),
            "status": str(ref.status),
            "note": str(ref.note),
        }
        for ref in STAGE_MATERIALIZER_REFERENCES
        if str(ref.network) == "R34" and str(ref.stage) in {"stage1", "stage2", "stage3", "stage4"}
    ]
    return {
        "same_shape_region": same_shape,
        "transition_region": transition,
        "stage_materializer_references": stage_refs,
    }


def _common_module_prefix(*node_names: str) -> str:
    if not node_names:
        return ""
    parts = [str(name).split("_") for name in node_names]
    prefix: list[str] = []
    for items in zip(*parts):
        if len(set(items)) != 1:
            break
        prefix.append(str(items[0]))
    return ".".join(prefix)


def _r34_expected_stats_by_family_label() -> dict[str, dict[str, int]]:
    refs = {
        "stem_conv": {},
        "stem_pool": {},
        "global_avgpool_exit": {},
        "stage1_same": next(
            dict(ref.expected_stats)
            for ref in STAGE_MATERIALIZER_REFERENCES
            if str(ref.network) == "R34" and str(ref.stage) == "stage1"
        ),
        "stage2_same": next(
            dict(ref.expected_stats)
            for ref in STAGE_MATERIALIZER_REFERENCES
            if str(ref.network) == "R34" and str(ref.stage) == "stage2"
        ),
        "stage3_same": next(
            dict(ref.expected_stats)
            for ref in STAGE_MATERIALIZER_REFERENCES
            if str(ref.network) == "R34" and str(ref.stage) == "stage3"
        ),
        "stage4_same": next(
            dict(ref.expected_stats)
            for ref in STAGE_MATERIALIZER_REFERENCES
            if str(ref.network) == "R34" and str(ref.stage) == "stage4"
        ),
    }
    transition_stats = dict(_selected_r34_results()["transition_region"]["candidate"])
    shared_transition = {
        key: int(transition_stats[key]) for key in ("rotations", "conjugations", "ct_pt_mults", "adds")
    }
    refs["stage2_transition"] = dict(shared_transition)
    refs["stage3_transition"] = dict(shared_transition)
    refs["stage4_transition"] = dict(shared_transition)
    return refs


def _r34_group_for_same_shape_node(*, node_name: str, family_label: str) -> RegionFirstRuntimeGroup:
    contract = imported_layout_contract_by_family_label(str(family_label))
    binding = kernel_binding_for_family(str(family_label))
    expected_stats = _r34_expected_stats_by_family_label().get(str(family_label), {})
    return RegionFirstRuntimeGroup(
        region_id=f"r34_imgnet_{str(family_label)}_{str(node_name)}",
        network="R34",
        stage=str(family_label),
        module_prefix=str(node_name).replace("_", "."),
        conv_nodes=(str(node_name),),
        strategy=str(binding.provider_key),
        materializer=str(binding.materializer),
        depth=2,
        boundary_actions=("insert_extract_before_relu_or_add", "validate_relu_safe"),
        expected_stats=dict(expected_stats),
        executable=False,
        fallback_reason="phase1_same_shape_provider_not_implemented",
    )


def _r34_group_for_direct_node(*, node_name: str, family_label: str) -> RegionFirstRuntimeGroup:
    binding = kernel_binding_for_family(str(family_label))
    expected_stats = _r34_expected_stats_by_family_label().get(str(family_label), {})
    return RegionFirstRuntimeGroup(
        region_id=f"r34_imgnet_{str(family_label)}_{str(node_name)}",
        network="R34",
        stage=str(family_label),
        module_prefix=str(node_name).replace("_", "."),
        conv_nodes=(str(node_name),),
        strategy=str(binding.provider_key),
        materializer=str(binding.materializer),
        depth=1,
        boundary_actions=(),
        expected_stats=dict(expected_stats),
        executable=False,
        fallback_reason="phase1_direct_executor_not_implemented",
    )


def _replace_r34_group(group: RegionFirstRuntimeGroup, **updates: Any) -> RegionFirstRuntimeGroup:
    payload = {
        "region_id": group.region_id,
        "network": group.network,
        "stage": group.stage,
        "module_prefix": group.module_prefix,
        "conv_nodes": group.conv_nodes,
        "strategy": group.strategy,
        "materializer": group.materializer,
        "depth": group.depth,
        "boundary_actions": group.boundary_actions,
        "expected_stats": group.expected_stats,
        "full_region": group.full_region,
        "hidden_fallback": group.hidden_fallback,
        "executable": group.executable,
        "fallback_reason": group.fallback_reason,
        "output_node_ids": group.output_node_ids,
        "executor": group.executor,
        "plan": group.plan,
        "fused_weight_count": group.fused_weight_count,
        "compiled": group.compiled,
        "execute_count": group.execute_count,
    }
    payload.update(updates)
    return RegionFirstRuntimeGroup(**payload)


def _r34_same_shape_family_label_for_module(node_name: str, module: Any) -> str | None:
    weight = getattr(module, "on_weight", None)
    if weight is None:
        return None
    if tuple(int(v) for v in getattr(module, "stride", ())) != (1, 1):
        return None
    if tuple(int(v) for v in getattr(module, "padding", ())) != (1, 1):
        return None
    input_shape = tuple(int(v) for v in getattr(module, "input_shape", torch.Size())[1:])
    output_shape = tuple(int(v) for v in getattr(module, "output_shape", torch.Size())[1:])
    if input_shape != output_shape:
        return None
    gap = int(getattr(module, "input_gap", -1))
    if int(getattr(module, "output_gap", -1)) != int(gap):
        return None
    if tuple(int(v) for v in tuple(weight.shape)) == (64, 64, 3, 3) and input_shape == (64, 56, 56) and int(gap) == 4:
        return "stage1_same"
    if tuple(int(v) for v in tuple(weight.shape)) == (128, 128, 3, 3) and input_shape == (128, 28, 28) and int(gap) == 8:
        return "stage2_same"
    if tuple(int(v) for v in tuple(weight.shape)) == (256, 256, 3, 3) and input_shape == (256, 14, 14) and int(gap) == 16:
        return "stage3_same"
    if tuple(int(v) for v in tuple(weight.shape)) == (512, 512, 3, 3) and input_shape == (512, 7, 7) and int(gap) == 32:
        return "stage4_same"
    return None


def _r34_direct_family_label_for_node(node_name: str) -> str | None:
    return {
        "conv1": "stem_conv",
        "pool": "stem_pool",
        "avgpool": "global_avgpool_exit",
    }.get(str(node_name))


def _r34_source_group_count_from_module(module: Any) -> int | None:
    input_shape = tuple(int(v) for v in getattr(module, "input_shape", torch.Size())[1:])
    if len(input_shape) != 3:
        return None
    c = int(input_shape[0])
    gap = int(getattr(module, "input_gap", -1))
    if int(gap) <= 0:
        return None
    return int(r34_source_group_count(c=int(c), gap=int(gap)))


def _r34_kernel_policy_from_module(module: Any) -> str | None:
    source_group_count = _r34_source_group_count_from_module(module)
    if source_group_count is None:
        return None
    return str(r34_same_shape_policy_from_source_group_count(int(source_group_count)))


def _merge_block_diagonals_as_complex(
    real_diagonals: dict[int, list[float]],
    imag_diagonals: dict[int, list[float]],
    *,
    slots: int,
) -> dict[int, list[complex]]:
    merged: dict[int, list[complex]] = {}
    all_indices = sorted(set(int(idx) for idx in real_diagonals).union(int(idx) for idx in imag_diagonals))
    zeros = [0.0] * int(slots)
    for idx in all_indices:
        real_values = list(real_diagonals.get(int(idx), zeros))
        imag_values = list(imag_diagonals.get(int(idx), zeros))
        merged[int(idx)] = [
            complex(float(getattr(real, "real", real)), float(getattr(imag, "real", imag)))
            for real, imag in zip(real_values, imag_values)
        ]
    return merged


class R34TransitionHybridRuntimeExecutor:
    def __init__(
        self,
        *,
        conv_module: Any,
        shortcut_module: Any,
        output_node_ids: tuple[str, str],
    ) -> None:
        self.conv_module = conv_module
        self.shortcut_module = shortcut_module
        self.output_node_ids = tuple(str(v) for v in output_node_ids)
        self.groups_by_input_block: list[Any] = []
        self.rows = 0
        self.cols = 0
        self.slots = 0
        self.output_shape = getattr(conv_module, "output_shape", None)
        self.fhe_output_shape = getattr(conv_module, "fhe_output_shape", None)
        self.conv_bias_vector: torch.Tensor | None = None
        self.shortcut_bias_vector: torch.Tensor | None = None
        self.compile_count = 0
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

    def _bias_chunk(self, bias_vector: torch.Tensor | None, *, row_index: int) -> torch.Tensor | None:
        if bias_vector is None:
            return None
        out = torch.zeros((int(self.slots),), dtype=torch.float32)
        start = int(row_index * self.slots)
        end = min(int(start + self.slots), int(bias_vector.numel()))
        if end > start:
            out[: int(end - start)] = bias_vector[start:end]
        return out

    def _add_bias(self, ct: Any, *, bias_vector: torch.Tensor | None, row_index: int) -> Any:
        chunk = self._bias_chunk(bias_vector, row_index=int(row_index))
        if chunk is None:
            return ct
        bias_pt = ct.scheme.encode(chunk, ct.level())
        return ct + bias_pt

    def compile(self, scheme: Any) -> None:
        if self.groups_by_input_block:
            return
        prepared_started = __import__("time").time()
        conv_diagonals, conv_output_rotations = packing.pack_conv2d(self.conv_module, last=False)
        shortcut_diagonals, shortcut_output_rotations = packing.pack_conv2d(self.shortcut_module, last=False)
        if int(conv_output_rotations) != 0 or int(shortcut_output_rotations) != 0:
            raise RuntimeError("r34 transition hybrid runtime currently requires zero output-rotation postprocessing")
        conv_keys = sorted((int(row), int(col)) for row, col in conv_diagonals)
        shortcut_keys = sorted((int(row), int(col)) for row, col in shortcut_diagonals)
        if conv_keys != shortcut_keys:
            raise RuntimeError("r34 transition hybrid runtime requires conv and shortcut block layouts to match")
        self.rows = 0 if not conv_keys else max(int(row) for row, _ in conv_keys) + 1
        self.cols = 0 if not conv_keys else max(int(col) for _, col in conv_keys) + 1
        self.slots = int(scheme.params.get_slots())
        level = self._level(scheme)
        self.conv_bias_vector = packing.construct_conv2d_bias(self.conv_module).to(dtype=torch.float32)
        self.shortcut_bias_vector = packing.construct_conv2d_bias(self.shortcut_module).to(dtype=torch.float32)

        groups: list[Any] = []
        for col_index in range(int(self.cols)):
            transforms: list[Any] = []
            for row_index in range(int(self.rows)):
                merged = _merge_block_diagonals_as_complex(
                    dict(conv_diagonals[(int(row_index), int(col_index))]),
                    dict(shortcut_diagonals[(int(row_index), int(col_index))]),
                    slots=int(self.slots),
                )
                transforms.append(
                    SimpleNamespace(
                        name=f"r34_transition_row{int(row_index)}_col{int(col_index)}",
                        diagonals={(0, 0): merged},
                        level=int(level),
                        scheme=scheme,
                        fhe_output_shape=torch.Size([1, int(self.slots)]),
                        output_shape=torch.Size([1, int(self.slots)]),
                    )
                )
            groups.append(UnifiedTransformGroup(transforms))
        self.last_runtime_timing["prepare_transforms_s"] = float(__import__("time").time() - prepared_started)

        compile_started = __import__("time").time()
        for group in groups:
            group.compile_unified(scheme.backend)
        self.groups_by_input_block = groups
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(__import__("time").time() - compile_started)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.last_runtime_timing = {
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.compile(scheme)
        ids = tuple(int(v) for v in getattr(source_ct, "ids", ()))
        if len(ids) < int(self.cols):
            raise RuntimeError(
                f"r34 transition hybrid runtime requires {self.cols} source ciphertext blocks, got {len(ids)}"
            )

        complex_rows: list[Any | None] = [None for _ in range(int(self.rows))]
        evaluate_started = __import__("time").time()
        for col_index, group in enumerate(self.groups_by_input_block):
            row_output_ids = group.evaluate_unified(int(ids[int(col_index)]), scheme.backend)
            for row_index, output_id in enumerate(row_output_ids):
                partial = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(self.slots)]),
                    torch.Size([1, int(self.slots)]),
                )
                if complex_rows[int(row_index)] is None:
                    complex_rows[int(row_index)] = partial
                else:
                    complex_rows[int(row_index)] = complex_rows[int(row_index)] + partial
        self.last_runtime_timing["evaluate_unified_s"] = float(__import__("time").time() - evaluate_started)

        postprocess_started = __import__("time").time()
        conv_ids: list[int] = []
        shortcut_ids: list[int] = []
        for row_index, row_ct in enumerate(complex_rows):
            if row_ct is None:
                raise RuntimeError(f"missing transition output row {row_index}")
            conj = row_ct.conjugate(in_place=False)
            conv_real = (row_ct + conj) * 0.5
            shortcut_imag = (row_ct - conj).mul_imaginary_unit(-1, in_place=False) * 0.5
            conv_real = self._add_bias(conv_real, bias_vector=self.conv_bias_vector, row_index=int(row_index))
            shortcut_imag = self._add_bias(shortcut_imag, bias_vector=self.shortcut_bias_vector, row_index=int(row_index))
            _maybe_set_default_scale(conv_real)
            _maybe_set_default_scale(shortcut_imag)
            conv_ids.append(int(conv_real.ids[0]))
            shortcut_ids.append(int(shortcut_imag.ids[0]))
            conv_real.ids = []
            shortcut_imag.ids = []
        self.last_runtime_timing["postprocess_s"] = float(__import__("time").time() - postprocess_started)

        return {
            str(self.output_node_ids[0]): CipherTensor(
                scheme,
                conv_ids,
                self.output_shape,
                self.fhe_output_shape,
            ),
            str(self.output_node_ids[1]): CipherTensor(
                scheme,
                shortcut_ids,
                self.output_shape,
                self.fhe_output_shape,
            ),
        }


class _DenseTransformProxy:
    def __init__(
        self,
        *,
        name: str,
        diagonals: dict[tuple[int, int], dict[int, Any]],
        level: int,
        bsgs_ratio: float,
        output_shape: Any,
        fhe_output_shape: Any,
    ) -> None:
        self.name = str(name)
        self.diagonals = dict(diagonals)
        self.level = int(level)
        self.bsgs_ratio = float(bsgs_ratio)
        self.output_shape = output_shape
        self.fhe_output_shape = fhe_output_shape
        self.transform_ids: dict[tuple[int, int], int] = {}


class R34DenseSingleFlowRuntimeExecutor:
    def __init__(self, *, module: Any, family_label: str, output_node_id: str) -> None:
        self.module = module
        self.family_label = str(family_label)
        self.output_node_id = str(output_node_id)
        self.output_shape = getattr(module, "output_shape", None)
        self.fhe_output_shape = getattr(module, "fhe_output_shape", None)
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.transform_ids: dict[tuple[int, int], int] = {}
        self.output_rotations = 0
        self.on_bias_ptxt: Any | None = None
        self.cols = 0
        self.rows = 0
        self.compile_count = 0
        self.last_runtime_timing: dict[str, float] = {
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self._proxy: _DenseTransformProxy | None = None

    def supports_scheme(self, scheme: Any | None) -> bool:
        return scheme is not None

    def _level(self, scheme: Any) -> int:
        return int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1

    def _bias_level(self, scheme: Any) -> int:
        depth = 0 if self.assigned_depth is None else int(self.assigned_depth)
        return max(0, int(self._level(scheme)) - int(depth))

    def compile(self, scheme: Any) -> None:
        if self._proxy is not None:
            return
        prepare_started = time.time()
        diagonals, output_rotations = packing.pack_conv2d(self.module, last=False)
        level = self._level(scheme)
        self.output_rotations = int(output_rotations)
        self._proxy = _DenseTransformProxy(
            name=f"{self.output_node_id}_dense_runtime",
            diagonals=diagonals,
            level=int(level),
            bsgs_ratio=float(getattr(self.module, "bsgs_ratio", 2.0)),
            output_shape=self.output_shape,
            fhe_output_shape=self.fhe_output_shape,
        )
        bias = packing.construct_conv2d_bias(self.module)
        self.on_bias_ptxt = scheme.encoder.encode(bias, self._bias_level(scheme))
        self.last_runtime_timing = {
            "prepare_transforms_s": float(time.time() - prepare_started),
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        compile_started = time.time()
        self.transform_ids = scheme.lt_evaluator.generate_transforms(self._proxy)
        self._proxy.transform_ids = dict(self.transform_ids)
        keys = list(self.transform_ids.keys())
        self.cols = 0 if not keys else max(int(col) for _row, col in keys) + 1
        self.rows = 0 if not keys else max(int(row) for row, _col in keys) + 1
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
        if self._proxy is None or self.on_bias_ptxt is None:
            raise RuntimeError(f"{self.output_node_id} dense single-flow runtime failed to compile")
        evaluate_started = time.time()
        out = scheme.lt_evaluator.evaluate_transforms(self._proxy, source_ct)
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)

        postprocess_started = time.time()
        slots = int(scheme.params.get_slots())
        for rotation_index in range(1, int(self.output_rotations) + 1):
            out += out.roll(int(slots // (2**int(rotation_index))))
        out += self.on_bias_ptxt
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {self.output_node_id: out}


def _r34_same_shape_module_compatible(module: Any, spec: R34SameShapeStageSpec) -> bool:
    weight = getattr(module, "on_weight", None)
    if weight is None:
        return False
    return (
        tuple(int(v) for v in tuple(weight.shape)) == tuple(int(v) for v in spec.weight_shape)
        and tuple(int(v) for v in getattr(module, "stride", ())) == (1, 1)
        and tuple(int(v) for v in getattr(module, "padding", ())) == (1, 1)
        and tuple(int(v) for v in getattr(module, "input_shape", torch.Size())[1:]) == (int(spec.c), int(spec.h), int(spec.w))
        and tuple(int(v) for v in getattr(module, "output_shape", torch.Size())[1:]) == (int(spec.c), int(spec.h), int(spec.w))
        and int(getattr(module, "input_gap", -1)) == int(spec.gap)
        and int(getattr(module, "output_gap", -1)) == int(spec.gap)
    )


def _r34_same_shape_runtime_from_modules(
    group: RegionFirstRuntimeGroup,
    modules: tuple[Any, ...],
    *,
    family_label: str,
) -> RegionFirstRuntimeGroup:
    if len(modules) != 1:
        return group
    module = modules[0]
    spec = r34_same_shape_spec_for_family_label(str(family_label))
    if not _r34_same_shape_module_compatible(module, spec):
        return group
    policy = str(_r34_kernel_policy_from_module(module) or spec.policy)
    if policy != str(spec.policy):
        return group
    if str(policy) == "inter_group_hybrid":
        executor = R34InterGroupHybridSameShapeRuntimeExecutor(module=module, spec=spec, output_node_id=str(group.conv_nodes[0]))
    else:
        executor = R34Pack2SameShapeRuntimeExecutor(module=module, spec=spec, output_node_id=str(group.conv_nodes[0]))
    return _replace_r34_group(
        group,
        executable=True,
        fallback_reason="",
        fused_weight_count=1,
        executor=executor,
    )


def _r34_transition_modules_compatible(modules: tuple[Any, ...], *, contracts: tuple[ImportedLayoutContract, ImportedLayoutContract]) -> bool:
    if len(modules) != 2:
        return False
    conv, shortcut = modules
    conv_contract, shortcut_contract = contracts
    conv_weight = getattr(conv, "on_weight", None)
    shortcut_weight = getattr(shortcut, "on_weight", None)
    return (
        conv_weight is not None
        and shortcut_weight is not None
        and tuple(int(v) for v in conv_weight.shape) == tuple(int(v) for v in conv_contract.weight_shape)
        and tuple(int(v) for v in shortcut_weight.shape) == tuple(int(v) for v in shortcut_contract.weight_shape)
        and tuple(int(v) for v in getattr(conv, "stride", ())) == tuple(int(v) for v in conv_contract.stride)
        and tuple(int(v) for v in getattr(shortcut, "stride", ())) == tuple(int(v) for v in shortcut_contract.stride)
    )


def _r34_transition_runtime_from_modules(
    group: RegionFirstRuntimeGroup,
    modules: tuple[Any, ...],
    *,
    contracts: tuple[ImportedLayoutContract, ImportedLayoutContract],
) -> RegionFirstRuntimeGroup:
    if not _r34_transition_modules_compatible(modules, contracts=contracts):
        return group
    conv, shortcut = modules
    policy = str(_r34_kernel_policy_from_module(conv) or "")
    if str(group.stage) in {"stage2_transition", "stage3_transition", "stage4_transition"}:
        executor = R34TransitionHybridRuntimeExecutor(
            conv_module=conv,
            shortcut_module=shortcut,
            output_node_ids=group.conv_nodes,
        )
    else:
        executor = R34PythonTransitionFlowRuntimeExecutor(
            conv_module=conv,
            shortcut_module=shortcut,
            family_label=str(group.stage),
            kernel_policy=str(policy),
            output_node_ids=group.conv_nodes,
        )
    return _replace_r34_group(
        group,
        executable=True,
        fallback_reason="",
        fused_weight_count=len(modules),
        executor=executor,
    )


def _r34_direct_module_compatible(module: Any, contract: ImportedLayoutContract) -> bool:
    weight = getattr(module, "on_weight", None)
    return (
        weight is not None
        and tuple(int(v) for v in weight.shape) == tuple(int(v) for v in contract.weight_shape)
        and tuple(int(v) for v in getattr(module, "stride", ())) == tuple(int(v) for v in contract.stride)
        and tuple(int(v) for v in getattr(module, "padding", ())) == tuple(int(v) for v in contract.padding)
        and tuple(int(v) for v in getattr(module, "dilation", ())) == tuple(int(v) for v in contract.dilation)
        and int(getattr(module, "groups", -1)) == int(contract.groups)
        and tuple(int(v) for v in getattr(module, "input_shape", torch.Size())[1:]) == tuple(int(v) for v in contract.input_shape)
        and tuple(int(v) for v in getattr(module, "output_shape", torch.Size())[1:]) == tuple(int(v) for v in contract.output_shape)
        and int(getattr(module, "input_gap", -1)) == int(contract.input_gap)
        and int(getattr(module, "output_gap", -1)) == int(contract.output_gap)
    )


def _r34_direct_runtime_from_modules(
    group: RegionFirstRuntimeGroup,
    modules: tuple[Any, ...],
    *,
    contract: ImportedLayoutContract,
) -> RegionFirstRuntimeGroup:
    if len(modules) != 1:
        return group
    module = modules[0]
    if not _r34_direct_module_compatible(module, contract):
        return group
    executor = R34DenseSingleFlowRuntimeExecutor(
        module=module,
        family_label=str(group.stage),
        output_node_id=str(group.conv_nodes[0]),
    )
    return _replace_r34_group(
        group,
        executable=True,
        fallback_reason="",
        fused_weight_count=1,
        executor=executor,
    )


def _r34_transition_group(*, family_label: str, conv_haloed_node: str, shortcut_haloed_node: str) -> RegionFirstRuntimeGroup:
    conv = imported_layout_contract_by_haloed_node(str(conv_haloed_node))
    downsample = imported_layout_contract_by_haloed_node(str(shortcut_haloed_node))
    binding = kernel_binding_for_family(str(family_label))
    expected_stats = _r34_expected_stats_by_family_label()[str(family_label)]
    conv_nodes = (str(conv.orion_node), str(downsample.orion_node))
    return RegionFirstRuntimeGroup(
        region_id=f"r34_imgnet_{str(family_label)}_pair",
        network="R34",
        stage=str(family_label),
        module_prefix=_common_module_prefix(*conv_nodes),
        conv_nodes=conv_nodes,
        strategy=str(binding.provider_key),
        materializer=str(binding.materializer),
        depth=2,
        boundary_actions=("insert_extract_before_relu_or_add", "validate_relu_safe"),
        expected_stats=dict(expected_stats),
        executable=False,
        fallback_reason="phase1_transition_executor_not_implemented",
    )


@dataclass
class R34CompileRegistry:
    groups: tuple[RegionFirstRuntimeGroup, ...]
    graph_audit: dict[str, Any]

    @classmethod
    def for_r34_imgnet_phase1(cls, dag) -> "R34CompileRegistry":
        wanted = {str(contract.orion_node): contract for contract in imported_layout_contracts()}
        existing_nodes = {str(node) for node in dag.nodes}
        missing_nodes = sorted(node for node in wanted if node not in existing_nodes)

        groups: list[RegionFirstRuntimeGroup] = []
        for node in sorted(existing_nodes):
            module = dag.nodes[str(node)].get("module")
            if module is None:
                continue
            direct_family = _r34_direct_family_label_for_node(str(node))
            if direct_family is not None:
                groups.append(_r34_group_for_direct_node(node_name=str(node), family_label=str(direct_family)))
                continue
            family_label = _r34_same_shape_family_label_for_module(str(node), module)
            if family_label is None:
                continue
            groups.append(_r34_group_for_same_shape_node(node_name=str(node), family_label=str(family_label)))

        transition_specs = (
            ("stage2_transition", "layer2_0_conv1_torch", "layer2_0_downsample_conv_torch"),
            ("stage3_transition", "layer3_0_conv1_torch", "layer3_0_downsample_conv_torch"),
            ("stage4_transition", "layer4_0_conv1_torch", "layer4_0_downsample_conv_torch"),
        )
        for family_label, conv_node, shortcut_node in transition_specs:
            conv_contract = imported_layout_contract_by_haloed_node(str(conv_node))
            shortcut_contract = imported_layout_contract_by_haloed_node(str(shortcut_node))
            transition_nodes = {str(conv_contract.orion_node), str(shortcut_contract.orion_node)}
            if transition_nodes.issubset(existing_nodes):
                groups.append(
                    _r34_transition_group(
                        family_label=str(family_label),
                        conv_haloed_node=str(conv_node),
                        shortcut_haloed_node=str(shortcut_node),
                    )
                )

        graph_audit = {
            "node_count": int(len(dag.nodes)),
            "edge_count": int(len(dag.edges)),
            "selected_region_count": int(len(groups)),
            "replacement_mode": "r34_imgnet_phase1_layout_contracts",
            "missing_imported_nodes": list(missing_nodes),
            "selected_nodes": [node for group in groups for node in group.conv_nodes],
            "family_labels": sorted({str(group.stage) for group in groups}),
        }
        return cls(groups=tuple(groups), graph_audit=graph_audit)

    def attach_to_dag(self, dag) -> dict[str, Any]:
        attached: list[dict[str, Any]] = []
        fallback_layers: list[dict[str, Any]] = []
        contracts_by_node = {str(contract.orion_node): contract for contract in imported_layout_contracts()}
        resolved_groups: list[RegionFirstRuntimeGroup] = []
        for group in self.groups:
            modules = tuple(dag.nodes[node].get("module") for node in group.conv_nodes if node in dag.nodes)
            if str(group.stage).endswith("_transition"):
                contracts = tuple(imported_layout_contract_by_orion_node(str(node)) for node in group.conv_nodes)
                group = _r34_transition_runtime_from_modules(group, modules, contracts=contracts)
            elif str(group.stage) in {"stem_conv", "stem_pool", "global_avgpool_exit"}:
                contract = imported_layout_contract_by_orion_node(str(group.conv_nodes[0]))
                group = _r34_direct_runtime_from_modules(group, modules, contract=contract)
            elif str(group.stage).endswith("_same") and len(group.conv_nodes) == 1:
                group = _r34_same_shape_runtime_from_modules(group, modules, family_label=str(group.stage))
            resolved_groups.append(group)
        self.groups = tuple(resolved_groups)
        group_by_node = {str(node): group for group in self.groups for node in group.conv_nodes}
        for node, group in group_by_node.items():
            if node not in dag.nodes:
                continue
            module = dag.nodes[node].get("module")
            if module is None:
                continue
            source_group_count = _r34_source_group_count_from_module(module)
            kernel_policy = _r34_kernel_policy_from_module(module)
            exact_contract = contracts_by_node.get(str(node))
            representative_contract = None
            if exact_contract is None and str(group.stage).endswith("_same"):
                representative_contract = imported_layout_contract_by_family_label(str(group.stage))
            contract = exact_contract or representative_contract
            contract_payload = None if contract is None else contract.to_dict()
            if contract_payload is not None and exact_contract is None:
                contract_payload["orion_node"] = str(node)
            module.region_runtime = group
            module.region_output_id = str(node)
            module.region_first_skip_dense_pack = bool(group.executable)
            module.region_layout_contract_imported = exact_contract is not None
            module.region_source_group_count = None if source_group_count is None else int(source_group_count)
            module.region_kernel_policy = None if kernel_policy is None else str(kernel_policy)
            if contract is not None and contract_payload is not None:
                module.region_layout_contract = dict(contract_payload)
                module.region_input_layout = dict(contract_payload["input_layout"])
                module.region_output_layout = dict(contract_payload["output_layout"])
                module.region_family_id = str(contract_payload["family_id"])
                module.region_family_kind = str(contract_payload["family_kind"])
                module.region_family_label = str(contract_payload["family_label"])
                module.region_exec_mode = str(contract_payload["exec_mode"])
                module.region_planner_diagnostics = dict(contract_payload["planner_diagnostics"])
            attached.append(
                {
                    "node": str(node),
                    "stage": str(group.stage),
                    "executable": bool(group.executable),
                    "materializer": str(group.materializer),
                    "layout_contract_imported": bool(exact_contract is not None),
                    "source_group_count": None if source_group_count is None else int(source_group_count),
                    "kernel_policy": None if kernel_policy is None else str(kernel_policy),
                }
            )
            if not bool(group.executable):
                fallback_layers.append({"node": str(node), "stage": str(group.stage), "reason": str(group.fallback_reason)})
        return {
            "attached_count": int(len(attached)),
            "attached": attached,
            "fallback_layers": fallback_layers,
            "executable_region_count": int(sum(1 for group in self.groups if bool(group.executable))),
            "graph_audit": dict(self.graph_audit),
        }


def build_r34_phase1_report() -> dict[str, Any]:
    contracts = imported_layout_contracts()
    same_shape_contracts = [contract for contract in contracts if str(contract.family_label).endswith("_same")]
    transition_plan = build_r34_phase1_transition_bridge_plan()
    selected = _selected_r34_results()
    return {
        "status": "ok",
        "scope": "R34 ImageNet phase1 layout-artifact and kernel-binding bridge for Orion",
        "phase": 1,
        "goal": "import layout contracts, bind selected families to providers, and surface an Orion-side report before full R34 CKKS integration",
        "separation_of_concerns": {
            "layout_artifact": "Imported HaloED layout contracts remain data-only in phase1. Orion does not run HaloED's layout DP search in this path.",
            "kernel_binding": "Family-to-provider bindings are kept explicit so scripts/cir kernels and Orion-native bridges can evolve independently.",
            "materializer_contract_fields": [
                "weight",
                "bias",
                "stride",
                "padding",
                "dilation",
                "groups",
                "input_layout",
                "layout",
                "_node_name",
                "module_target",
            ],
        },
        "layout_registry": {
            "selected_node_count": int(len(contracts)),
            "family_labels": sorted({str(contract.family_label) for contract in contracts}),
            "nodes": [contract.to_dict() for contract in contracts],
        },
        "kernel_bindings": [binding.to_dict() for binding in kernel_bindings()],
        "same_shape_surface": [_same_shape_surface(contract) for contract in same_shape_contracts],
        "transition_bridge": transition_plan,
        "selected_results": selected,
        "limitations": {
            "full_layout_search_ported": False,
            "full_scripts_cir_executor_ported": False,
            "production_pack_conv2d_modified": False,
            "full_model_ckks_e2e": False,
        },
    }


def write_r34_phase1_report(*, out_path: Path = DEFAULT_R34_PHASE1_REPORT_OUT) -> dict[str, Any]:
    payload = build_r34_phase1_report()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
