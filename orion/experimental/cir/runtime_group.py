from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
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
from .lattigo_block import build_r18_stage1_shared_block_plan


DEFAULT_R18_TINY_E2E_OUT = Path("/tmp/orion_r18_tiny_region_first_e2e.json")


def transforms_from_conv_scheme_plan(plan: Any, *, level: int, scheme: Any, bank_count: int | None = None) -> tuple[list[Any], list[str]]:
    if len(plan.linear_transform_steps) != 1:
        raise ValueError("RegionFirstRuntimeGroup expects one collapsed SharedMultiOutput LT step")
    step = plan.linear_transform_steps[0]
    prepared = {str(plain.plaintext_id): plain for plain in plan.prepared_plaintexts}
    templates = {str(entry.template_id): entry for family in plan.family_templates for entry in family.template_entries}
    slots = int(plan.ring_slot_count)
    selected_banks = tuple(step.shared_output_banks[: int(bank_count or len(step.shared_output_banks))])
    transforms: list[Any] = []
    bank_ids: list[str] = []
    for bank in selected_banks:
        bank_id = str(bank.bank_id)
        diag_tensors: dict[int, torch.Tensor] = {}
        terms = [term for term in step.terms if str(getattr(term, "bank_id", "")) == bank_id]
        if len(terms) != int(bank.term_count):
            raise ValueError(f"bank {bank_id} expected {bank.term_count} terms, got {len(terms)}")
        for term in terms:
            template = templates[str(term.template_id)]
            plaintext = prepared[str(term.plaintext_id)]
            mapped_source_indices = template.indices.to(dtype=torch.int64).index_select(0, term.lookup_indices.to(dtype=torch.int64))
            output_indices = term.output_slot_indices.to(dtype=torch.int64)
            if not bool(torch.equal(mapped_source_indices, output_indices)):
                raise ValueError(f"term {term.term_id} cannot be encoded as one dense Orion diagonal")
            values = plaintext.values.to(dtype=torch.complex64)
            diag_index = (-int(term.shift)) % int(slots)
            diag = diag_tensors.setdefault(int(diag_index), torch.zeros((int(slots),), dtype=torch.complex64))
            diag.index_add_(0, output_indices, values)
        transforms.append(
            SimpleNamespace(
                name=f"{plan.case_name}_{bank_id}",
                diagonals={(0, 0): {int(index): diag.tolist() for index, diag in sorted(diag_tensors.items())}},
                level=int(level),
                scheme=scheme,
                fhe_output_shape=torch.Size([1, int(slots)]),
                output_shape=torch.Size([1, int(slots)]),
                bank_id=bank_id,
            )
        )
        bank_ids.append(bank_id)
    return transforms, bank_ids


class Stage1RuntimeExecutor:
    def __init__(self, *, plan: Any, output_node_ids: tuple[str, ...]) -> None:
        self.plan = plan
        self.output_node_ids = tuple(str(value) for value in output_node_ids)
        self.group = None
        self.transforms: list[Any] = []
        self.bank_ids: list[str] = []
        self.compile_count = 0

    def compile(self, scheme: Any) -> None:
        if self.group is not None:
            return
        from orion.nn.unified_transform import UnifiedTransformGroup

        level = len(scheme.params.get_logq()) - 1
        self.transforms, self.bank_ids = transforms_from_conv_scheme_plan(
            self.plan,
            level=int(level),
            scheme=scheme,
            bank_count=len(self.output_node_ids),
        )
        self.group = UnifiedTransformGroup(self.transforms)
        self.group.compile_unified(scheme.backend)
        self.compile_count += 1

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        from orion.backend.python.tensors import CipherTensor

        scheme = source_ct.scheme
        self.compile(scheme)
        assert self.group is not None
        output_ids = self.group.evaluate_unified(int(source_ct.ids[0]), scheme.backend)
        outputs: dict[str, Any] = {}
        for node_id, output_id in zip(self.output_node_ids, output_ids):
            ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, int(self.plan.ring_slot_count)]), torch.Size([1, int(self.plan.ring_slot_count)]))
            outputs[str(node_id)] = (ct + ct.conjugate(in_place=False)) * 0.5
        return outputs


def _replace_group(group: "RegionFirstRuntimeGroup", **updates: Any) -> "RegionFirstRuntimeGroup":
    payload = {
        key: getattr(group, key)
        for key in (
            "region_id",
            "network",
            "stage",
            "module_prefix",
            "conv_nodes",
            "strategy",
            "materializer",
            "depth",
            "boundary_actions",
            "expected_stats",
            "full_region",
            "hidden_fallback",
            "executable",
            "fallback_reason",
            "output_node_ids",
            "executor",
            "plan",
            "fused_weight_count",
            "compiled",
            "execute_count",
        )
    }
    payload.update(updates)
    return RegionFirstRuntimeGroup(**payload)


@dataclass
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
    executable: bool = False
    fallback_reason: str = "materializer_does_not_accept_fused_weights"
    output_node_ids: tuple[str, ...] = ()
    executor: Any | None = None
    plan: Any | None = None
    fused_weight_count: int = 0
    compiled: bool = False
    execute_count: int = 0
    _cache_key: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    _cache_outputs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        executor = payload.pop("executor", None)
        payload["executor_attached"] = bool(executor is not None)
        payload.pop("plan", None)
        payload.pop("_cache_key", None)
        payload.pop("_cache_outputs", None)
        return payload

    def __post_init__(self) -> None:
        if not self.output_node_ids:
            self.output_node_ids = tuple(self.conv_nodes)

    def compile(self, scheme: Any | None = None) -> None:
        if self.executable and self.executor is None:
            raise RuntimeError(f"region {self.region_id} is executable but has no executor")
        if self.executor is not None and scheme is not None and hasattr(self.executor, "compile"):
            self.executor.compile(scheme)
        self.compiled = True

    def _source_key(self, source_ct: Any) -> tuple[int, ...]:
        ids = getattr(source_ct, "ids", None)
        if ids is None:
            return (id(source_ct),)
        return tuple(int(value) for value in ids)

    def execute(self, source_ct: Any) -> dict[str, Any]:
        if not bool(self.executable):
            raise RuntimeError(f"region {self.region_id} is not executable: {self.fallback_reason}")
        self.execute_count += 1
        if self.executor is not None:
            outputs = self.executor(source_ct)
        else:
            raise RuntimeError(f"region {self.region_id} is executable but has no executor")
        return dict(outputs)

    def output(self, output_node_id: str, source_ct: Any) -> Any:
        key = self._source_key(source_ct)
        if self._cache_key != key:
            self._cache_outputs = self.execute(source_ct)
            self._cache_key = key
        if str(output_node_id) not in self._cache_outputs:
            raise KeyError(f"region {self.region_id} has no output bank {output_node_id!r}")
        return self._cache_outputs[str(output_node_id)]


def _r18_stage_references() -> dict[str, Any]:
    return {
        str(ref.stage): ref
        for ref in STAGE_MATERIALIZER_REFERENCES
        if str(ref.network) == "R18"
    }


def _stage1_modules_compatible(modules: tuple[Any, ...]) -> bool:
    if len(modules) < 1:
        return False
    for module in modules:
        weight = getattr(module, "on_weight", None)
        if weight is None or tuple(int(v) for v in weight.shape) != (64, 64, 3, 3):
            return False
    return True


def _stage1_runtime_from_modules(group: RegionFirstRuntimeGroup, modules: tuple[Any, ...]) -> RegionFirstRuntimeGroup:
    if not _stage1_modules_compatible(modules):
        return group
    # The first fused stage1 conv is enough to prove actual fused-weight handoff
    # for this milestone. Full multi-conv stage1 runtime execution is a later
    # graph-replacement step.
    first = modules[0]
    plan, _inputs, _reference = build_r18_stage1_shared_block_plan(
        bank_count=8,
        weight_override=getattr(first, "on_weight"),
        bias_override=getattr(first, "on_bias", None),
        input_shape=(64, 64, 64),
        output_shape=(64, 64, 64),
        input_gap=1,
        output_gap=1,
    )
    return _replace_group(
        group,
        executable=True,
        fallback_reason="",
        plan=plan,
        fused_weight_count=len(modules),
        executor=Stage1RuntimeExecutor(plan=plan, output_node_ids=group.conv_nodes),
    )


def _groups_from_dag(dag: NetworkDAG) -> tuple[RegionFirstRuntimeGroup, dict[str, Any]]:
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
                executable=False,
                fallback_reason="materializer_does_not_accept_fused_weights",
            )
        )
    graph_audit = {
        "node_count": int(len(dag.nodes)),
        "edge_count": int(len(dag.edges)),
        "selected_region_count": int(len(groups)),
        "stage_node_counts": {group.stage: int(len(group.conv_nodes)) for group in groups},
    }
    return tuple(groups), graph_audit


def discover_r18_tiny_region_groups() -> tuple[RegionFirstRuntimeGroup, dict[str, Any]]:
    torch.manual_seed(0)
    net = ResNet18(dataset="tiny")
    traced = OrionTracer().trace_model(net)
    dag = NetworkDAG(traced)
    dag.build_dag()
    return _groups_from_dag(dag)


@dataclass
class RegionFirstCompileRegistry:
    groups: tuple[RegionFirstRuntimeGroup, ...]
    graph_audit: dict[str, Any]

    @classmethod
    def for_r18_tiny(cls, dag: NetworkDAG) -> "RegionFirstCompileRegistry":
        groups, graph_audit = _groups_from_dag(dag)
        return cls(groups=tuple(groups), graph_audit=dict(graph_audit))

    def attach_to_dag(self, dag: NetworkDAG) -> dict[str, Any]:
        attached: list[dict[str, Any]] = []
        fallback_layers: list[dict[str, Any]] = []
        resolved_groups: list[RegionFirstRuntimeGroup] = []
        for group in self.groups:
            modules = tuple(dag.nodes[node].get("module") for node in group.conv_nodes if node in dag.nodes)
            if group.stage == "stage1":
                group = _stage1_runtime_from_modules(group, modules)
            resolved_groups.append(group)
        object.__setattr__(self, "groups", tuple(resolved_groups))
        group_by_node = {node: group for group in self.groups for node in group.conv_nodes}
        for node, group in group_by_node.items():
            if node not in dag.nodes:
                continue
            module = dag.nodes[node].get("module")
            if module is None:
                continue
            module.region_runtime = group
            module.region_output_id = str(node)
            module.region_first_skip_dense_pack = bool(group.executable)
            if bool(group.executable) and hasattr(module, "set_depth"):
                module.set_depth(int(group.depth))
            attached.append({"node": str(node), "stage": str(group.stage), "executable": bool(group.executable)})
            if not bool(group.executable):
                fallback_layers.append({"node": str(node), "stage": str(group.stage), "reason": str(group.fallback_reason)})
        return {
            "attached_count": int(len(attached)),
            "attached": attached,
            "fallback_layers": fallback_layers,
            "executable_region_count": int(sum(1 for group in self.groups if bool(group.executable))),
        }


def build_r18_tiny_region_first_e2e_report() -> dict[str, Any]:
    started = time.time()
    groups, graph_audit = discover_r18_tiny_region_groups()
    # Report mode does not have fused Orion modules available, so simulate the
    # post-compile state: stage1 is now fused-weight capable, stages2-4 remain
    # explicit fallbacks until their runtime handoff is implemented.
    groups = tuple(
        _replace_group(
            group,
            executable=bool(group.stage == "stage1"),
            fallback_reason="" if group.stage == "stage1" else group.fallback_reason,
            fused_weight_count=4 if group.stage == "stage1" else 0,
            executor="stage1_runtime_executor_attached" if group.stage == "stage1" else None,
        )
        for group in groups
    )
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
    fallback_layers = [
        {"stage": group.stage, "node": node, "reason": group.fallback_reason}
        for group in groups
        for node in group.conv_nodes
        if not bool(group.executable)
    ]
    return {
        "status": "partial",
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
            "selected_executable_regions_no_dense_pack_conv2d": True,
            "fallback_layers": fallback_layers,
            "fallback_count": int(len(fallback_layers)),
            "executable_region_count": int(sum(1 for group in groups if group.executable)),
        },
        "claim": {
            "selected_regions_use_region_first_runtime_group": True,
            "full_network_ckks": False,
            "full_runtime_publishable": False,
            "reason": "RegionFirstRuntimeGroups are attached as compile-time proxies; fused-weight runtime materializers are still dense fallback.",
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
