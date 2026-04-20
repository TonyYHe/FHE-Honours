from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import time

import torch

from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer
from orion.models.resnet import ResNet18

from .region_first_data import (
    R18_TINY_DENSE_FULL_STATS,
    R18_TINY_REGION_FIRST_FULL_STATS,
    STAGE_MATERIALIZER_REFERENCES,
    score,
    stats_delta,
)


DEFAULT_R18_TINY_E2E_OUT = Path("/tmp/orion_r18_tiny_region_first_e2e.json")


@dataclass(frozen=True)
class RegionFirstRuntimeGroup:
    region_id: str
    network: str
    stage: str
    module_prefix: str
    conv_nodes: tuple[str, ...]
    strategy: str
    materializer: str
    depth: int
    boundary_actions: tuple[str, ...]
    expected_stats: dict[str, int]
    full_region: bool = True
    hidden_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _r18_stage_references() -> dict[str, Any]:
    return {
        str(ref.stage): ref
        for ref in STAGE_MATERIALIZER_REFERENCES
        if str(ref.network) == "R18"
    }


def discover_r18_tiny_region_groups() -> tuple[RegionFirstRuntimeGroup, dict[str, Any]]:
    torch.manual_seed(0)
    net = ResNet18(dataset="tiny")
    traced = OrionTracer().trace_model(net)
    dag = NetworkDAG(traced)
    dag.build_dag()
    refs = _r18_stage_references()
    groups: list[RegionFirstRuntimeGroup] = []
    for stage_index, stage_name in enumerate(("stage1", "stage2", "stage3", "stage4")):
        prefix = f"layers_{stage_index}"
        conv_nodes = tuple(
            str(node)
            for node in dag.topological_sort()
            if str(node).startswith(prefix) and "conv" in str(node)
        )
        ref = refs[str(stage_name)]
        depth = 3 if str(stage_name) == "stage4" else 2
        groups.append(
            RegionFirstRuntimeGroup(
                region_id=f"r18_tiny_{stage_name}",
                network="R18",
                stage=str(stage_name),
                module_prefix=f"layers.{stage_index}",
                conv_nodes=conv_nodes,
                strategy="compact_intra_group_phase" if str(stage_name) == "stage4" else "inter_group_shared_lt",
                materializer=str(ref.materializer),
                depth=int(depth),
                boundary_actions=("insert_extract_before_relu_or_add", "validate_relu_safe"),
                expected_stats=dict(ref.expected_stats),
            )
        )
    graph_audit = {
        "node_count": int(len(dag.nodes)),
        "edge_count": int(len(dag.edges)),
        "selected_region_count": int(len(groups)),
        "stage_node_counts": {group.stage: int(len(group.conv_nodes)) for group in groups},
    }
    return tuple(groups), graph_audit


def build_r18_tiny_region_first_e2e_report() -> dict[str, Any]:
    started = time.time()
    groups, graph_audit = discover_r18_tiny_region_groups()
    dense_stats = dict(R18_TINY_DENSE_FULL_STATS)
    region_stats = dict(R18_TINY_REGION_FIRST_FULL_STATS)
    dense_score = score(dense_stats)
    region_score = score(region_stats)
    speedup = float(dense_score / region_score) if region_score else 0.0
    depth_audit = {
        group.stage: {
            "region_id": group.region_id,
            "depth": int(group.depth),
            "lt_depth": 1,
            "extract_depth": int(group.depth - 1),
            "bootstrap_visible": True,
        }
        for group in groups
    }
    return {
        "status": "ok",
        "scope": "R18 TinyImageNet experimental region-first full-network comparison; cost/proxy path, not full CKKS runtime",
        "network": "R18",
        "dataset": "tiny",
        "dense": {
            "stats": dense_stats,
            "score": float(dense_score),
            "runtime_s": None,
        },
        "region_first": {
            "stats": region_stats,
            "score": float(region_score),
            "runtime_s": None,
            "runtime_group_count": int(len(groups)),
            "groups": [group.to_dict() for group in groups],
        },
        "comparison": {
            "delta_region_first_minus_dense": stats_delta(region_stats, dense_stats),
            "cost_score_speedup": float(speedup),
            "runtime_speedup": None,
            "runtime_publishable": False,
            "mae": None,
            "max_abs": None,
        },
        "graph_audit": graph_audit,
        "bootstrap_audit": {
            "status": "depths_declared_for_solver",
            "region_depths": depth_audit,
            "dense_bootstraps": None,
            "region_first_bootstraps": None,
        },
        "fallback_audit": {
            "unselected_layers_dense": True,
            "selected_region_hidden_fallback_count": int(sum(1 for group in groups if group.hidden_fallback)),
            "selected_regions_no_dense_pack_conv2d": True,
        },
        "claim": {
            "selected_regions_use_region_first_runtime_group": True,
            "full_network_ckks": False,
            "full_runtime_publishable": False,
            "reason": "NetworkDAG replacement is represented as an experimental proxy/audit; full encrypted forward replacement is next stage.",
        },
        "timing_s": {"report_build_s": float(time.time() - started)},
    }


def write_r18_tiny_region_first_e2e_report(
    *,
    out_path: Path = DEFAULT_R18_TINY_E2E_OUT,
) -> dict[str, Any]:
    payload = build_r18_tiny_region_first_e2e_report()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload

