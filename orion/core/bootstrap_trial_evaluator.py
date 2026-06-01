from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import networkx as nx
import torch.nn as nn

from orion.nn.linear import LinearTransform

from .bootstrap_fusion import (
    bootstrap_prescale_fusion_supported,
    module_bootstrap_ct_count,
    module_bootstrap_slots,
    runtime_fhe_output_shape,
)


@dataclass(frozen=True)
class ExactBootstrapTrialResult:
    input_level: int
    bootstrap_count: int
    bootstrapper_slots: list[int]
    audit: dict[str, Any]
    node_levels: dict[str, int]


@dataclass(frozen=True)
class _ResidualSpec:
    fork: Any
    join: Any
    paths: tuple[tuple[Any, ...], ...]
    nodes: frozenset[Any]
    path_count: int


@dataclass
class _Segment:
    head: Any
    tail: Any
    costs: list[list[float]]
    high_paths: list[list[tuple[str, ...]]]
    node_sets: list[list[frozenset[str]]]


@dataclass(frozen=True)
class _NodeInfo:
    module: Any
    depth: int | None
    forced_level: int | None
    is_identity: bool
    is_linear: bool
    linear_diag_count: int
    bootstrap_ct_count: int
    bootstrap_slots: int
    min_bootstrap_input_level: int
    shape: tuple[int, ...] | None


class ExactBootstrapTrialEvaluator:
    """Pure evaluator for BootstrapSolver trial scoring.

    The evaluator snapshots topology once, then replays LevelDAG semantics
    against the current module state without assigning levels or boot flags to
    the real DAG. It is intentionally conservative: callers still use
    BootstrapSolver as the commit/rollback source of truth.
    """

    def __init__(self, network_dag: Any, *, l_eff: int) -> None:
        self.network_dag = network_dag
        self.l_eff = int(l_eff)
        self._levels = list(range(int(l_eff) + 1))
        self._topo_order = tuple(nx.topological_sort(network_dag))
        self._residual_specs = self._snapshot_residual_specs()
        self._residual_by_fork = {spec.fork: spec for spec in self._residual_specs}
        self._node_info: dict[str, _NodeInfo] | None = None

    def evaluate(self) -> ExactBootstrapTrialResult:
        self._node_info = self._snapshot_node_info()
        try:
            solved_segments = self._solve_residual_segments()
            full_segment = self._build_full_segment(solved_segments)
            if full_segment is None:
                raise ValueError("bootstrap trial evaluator cannot solve an empty DAG")
            high_path, node_states, _latency = self._select_full_path(full_segment)
            node_levels = self._node_levels_from_states(node_states)
            input_level = self._input_level_from_high_path(high_path)
            bootstrap_nodes = self._mark_bootstrap_nodes(node_levels)
            bootstrap_count, bootstrapper_slots = self._count_marked_bootstraps(
                node_levels,
                bootstrap_nodes,
            )
            audit = self._collect_audit(node_levels, bootstrap_nodes)
            return ExactBootstrapTrialResult(
                input_level=int(input_level),
                bootstrap_count=int(bootstrap_count),
                bootstrapper_slots=[int(value) for value in bootstrapper_slots],
                audit=audit,
                node_levels=node_levels,
            )
        finally:
            self._node_info = None

    def _snapshot_node_info(self) -> dict[str, _NodeInfo]:
        info: dict[str, _NodeInfo] = {}
        for node in self.network_dag.nodes:
            module = self.network_dag.nodes[node]["module"]
            depth = int(getattr(module, "depth")) if module is not None and hasattr(module, "depth") else None
            module_level = getattr(module, "level", None) if module is not None else None
            forced_level = int(module_level) if module_level else None
            is_identity = isinstance(module, nn.Identity)
            is_linear = isinstance(module, LinearTransform)
            linear_diag_count = (
                int(sum(len(diags) for diags in module.diagonals.values()))
                if is_linear
                else 0
            )
            bootstrap_ct_count = int(module_bootstrap_ct_count(module)) if module is not None else 0
            bootstrap_slots = int(module_bootstrap_slots(module)) if module is not None else 0
            min_bootstrap_input_level = (
                1
                if module is not None
                and not is_identity
                and hasattr(module, "depth")
                and bootstrap_prescale_fusion_supported(module)
                else 2
            )
            shape = runtime_fhe_output_shape(module) if module is not None else None
            info[str(node)] = _NodeInfo(
                module=module,
                depth=depth,
                forced_level=forced_level,
                is_identity=is_identity,
                is_linear=is_linear,
                linear_diag_count=linear_diag_count,
                bootstrap_ct_count=bootstrap_ct_count,
                bootstrap_slots=bootstrap_slots,
                min_bootstrap_input_level=int(min_bootstrap_input_level),
                shape=None if shape is None else tuple(int(value) for value in tuple(shape)),
            )
        return info

    def _info(self, node: Any) -> _NodeInfo:
        if self._node_info is None:
            raise RuntimeError("bootstrap trial evaluator node cache is not initialized")
        return self._node_info[str(node)]

    def _snapshot_residual_specs(self) -> tuple[_ResidualSpec, ...]:
        specs: list[_ResidualSpec] = []
        all_subgraphs = []
        for fork in self.network_dag.residuals.keys():
            all_subgraphs.append(self.network_dag.extract_residual_subgraph(fork))
        for index, (fork, join) in enumerate(self.network_dag.residuals.items()):
            subgraph = all_subgraphs[index]
            paths = list(nx.all_simple_paths(subgraph, fork, join))
            unique_paths = []
            visited_children = set()
            for path in paths:
                if path[1] not in visited_children:
                    unique_paths.append(tuple(path))
                    visited_children.add(path[1])
            nodes = set()
            for path in paths:
                nodes.update(path)
            specs.append(
                _ResidualSpec(
                    fork=fork,
                    join=join,
                    paths=tuple(unique_paths),
                    nodes=frozenset(nodes),
                    path_count=int(len(paths)),
                )
            )
        return tuple(sorted(specs, key=lambda spec: spec.path_count))

    def _solve_residual_segments(self) -> dict[Any, _Segment]:
        solved: dict[Any, _Segment] = {}
        for spec in self._residual_specs:
            aggregate: _Segment | None = None
            for path in spec.paths:
                segment = self._build_path_segment(path, solved)
                aggregate = segment if aggregate is None else self._add_residual_segments(aggregate, segment)
            if aggregate is None:
                raise ValueError(f"residual {spec.fork!r}->{spec.join!r} has no paths")
            solved[spec.fork] = aggregate
        return solved

    def _build_full_segment(self, solved_segments: dict[Any, _Segment]) -> _Segment | None:
        full: _Segment | None = None
        visited: set[Any] = set()
        for node in self._topo_order:
            if node in visited:
                continue
            if node in solved_segments:
                segment = solved_segments[node]
                spec = self._residual_by_fork[node]
                visited.update(spec.nodes)
            else:
                segment = self._single_node_segment(node)
            full = segment if full is None else self._append_segments(full, segment)
        return full

    def _build_path_segment(
        self,
        path: tuple[Any, ...],
        solved_segments: dict[Any, _Segment],
    ) -> _Segment:
        segment: _Segment | None = None
        visited: set[Any] = set()
        index_by_node = {node: index for index, node in enumerate(path)}
        for node in path:
            if node in visited:
                continue
            if node in solved_segments:
                next_segment = solved_segments[node]
                join = self.network_dag.residuals[node]
                start_index = index_by_node[node]
                end_index = index_by_node[join]
                visited.update(path[start_index : end_index + 1])
            else:
                next_segment = self._single_node_segment(node)
            segment = next_segment if segment is None else self._append_segments(segment, next_segment)
        if segment is None:
            raise ValueError("cannot build bootstrap trial segment for an empty path")
        return segment

    def _single_node_segment(self, node: Any) -> _Segment:
        size = len(self._levels)
        costs = [[float("inf") for _ in self._levels] for _ in self._levels]
        high_paths: list[list[tuple[str, ...]]] = [[() for _ in self._levels] for _ in self._levels]
        node_sets: list[list[frozenset[str]]] = [
            [frozenset() for _ in self._levels] for _ in self._levels
        ]
        for level in self._levels:
            state = self._state(node, level)
            costs[level][level] = float(self._layer_latency(node, level))
            high_paths[level][level] = (state,)
            node_sets[level][level] = frozenset({state})
        assert len(costs) == size
        return _Segment(head=node, tail=node, costs=costs, high_paths=high_paths, node_sets=node_sets)

    def _add_residual_segments(self, left: _Segment, right: _Segment) -> _Segment:
        if left.head != right.head or left.tail != right.tail:
            raise ValueError("residual segment endpoints do not match")
        costs = [[float("inf") for _ in self._levels] for _ in self._levels]
        high_paths: list[list[tuple[str, ...]]] = [[() for _ in self._levels] for _ in self._levels]
        node_sets: list[list[frozenset[str]]] = [
            [frozenset() for _ in self._levels] for _ in self._levels
        ]
        for source_level in self._levels:
            for target_level in self._levels:
                left_cost = left.costs[source_level][target_level]
                right_cost = right.costs[source_level][target_level]
                if math.isinf(left_cost) or math.isinf(right_cost):
                    continue
                costs[source_level][target_level] = float(left_cost + right_cost)
                high_paths[source_level][target_level] = (
                    self._state(left.head, source_level),
                    self._state(left.tail, target_level),
                )
                node_sets[source_level][target_level] = (
                    left.node_sets[source_level][target_level]
                    | right.node_sets[source_level][target_level]
                )
        return _Segment(
            head=left.head,
            tail=left.tail,
            costs=costs,
            high_paths=high_paths,
            node_sets=node_sets,
        )

    def _append_segments(self, left: _Segment, right: _Segment) -> _Segment:
        costs = [[float("inf") for _ in self._levels] for _ in self._levels]
        high_paths: list[list[tuple[str, ...]]] = [[() for _ in self._levels] for _ in self._levels]
        node_sets: list[list[frozenset[str]]] = [
            [frozenset() for _ in self._levels] for _ in self._levels
        ]
        for source_level in self._levels:
            head_dist = [float("inf") for _ in self._levels]
            head_tail_level = [-1 for _ in self._levels]
            for tail_level in self._levels:
                left_cost = left.costs[source_level][tail_level]
                if math.isinf(left_cost):
                    continue
                for head_level in self._levels:
                    edge_cost, _boot_count = self._bootstrap_latency(
                        left.tail,
                        tail_level,
                        right.head,
                        head_level,
                    )
                    if math.isinf(edge_cost):
                        continue
                    candidate_cost = float(left_cost + edge_cost)
                    if candidate_cost < head_dist[head_level]:
                        head_dist[head_level] = candidate_cost
                        head_tail_level[head_level] = int(tail_level)
            for target_level in self._levels:
                for head_level in self._levels:
                    right_cost = right.costs[head_level][target_level]
                    if math.isinf(head_dist[head_level]) or math.isinf(right_cost):
                        continue
                    candidate_cost = float(head_dist[head_level] + right_cost)
                    if candidate_cost < costs[source_level][target_level]:
                        tail_level = head_tail_level[head_level]
                        costs[source_level][target_level] = candidate_cost
                        high_paths[source_level][target_level] = (
                            left.high_paths[source_level][tail_level]
                            + right.high_paths[head_level][target_level]
                        )
                        node_sets[source_level][target_level] = (
                            left.node_sets[source_level][tail_level]
                            | right.node_sets[head_level][target_level]
                            | frozenset(
                                {
                                    self._state(left.tail, tail_level),
                                    self._state(right.head, head_level),
                                }
                            )
                        )
        return _Segment(
            head=left.head,
            tail=right.tail,
            costs=costs,
            high_paths=high_paths,
            node_sets=node_sets,
        )

    def _select_full_path(self, segment: _Segment) -> tuple[tuple[str, ...], frozenset[str], float]:
        best_cost = float("inf")
        best_source_level = -1
        best_target_level = -1
        for target_level in self._levels:
            for source_level in self._levels:
                cost = segment.costs[source_level][target_level]
                if cost < best_cost:
                    best_cost = float(cost)
                    best_source_level = int(source_level)
                    best_target_level = int(target_level)
        if math.isinf(best_cost):
            raise ValueError(
                "Automatic bootstrap placement failed. First try increasing "
                "the length of your LogQ moduli chain the associated "
                "parameters YAML file. If this fails, double check that the "
                "network was instantiated properly."
            )
        return (
            segment.high_paths[best_source_level][best_target_level],
            segment.node_sets[best_source_level][best_target_level],
            best_cost,
        )

    def _layer_latency(self, node: Any, level: int) -> float:
        info = self._info(node)
        if info.is_identity:
            return 0.0
        if info.module and info.depth is None:
            raise ValueError(
                f"The multiplicative depth of the Orion module {info.module} "
                "cannot be automatically determined. Ensure it has a "
                "depth attribute."
            )
        if not info.module or info.depth is None:
            return 0.0
        if info.forced_level is not None and int(info.forced_level) != int(level):
            return float("inf")
        if int(level) < int(info.depth):
            return float("inf")
        if info.is_linear:
            return float(0.001 * int(info.linear_diag_count) * int(level))
        return 0.0

    def _bootstrap_latency(
        self,
        prev_node: Any,
        prev_level: int,
        curr_node: Any,
        curr_level: int,
    ) -> tuple[float, int]:
        prev_info = self._info(prev_node)
        if prev_info.module is None or prev_info.is_identity:
            if int(prev_level) >= int(curr_level):
                return 0.0, 0
            return float("inf"), 0
        if prev_info.depth is None:
            raise ValueError(
                f"The multiplicative depth of the Orion module {prev_info.module} "
                "cannot be automatically determined. Ensure it has a depth attribute."
            )
        prev_output_level = int(prev_level) - int(prev_info.depth)
        if int(curr_level) > prev_output_level:
            if int(prev_output_level) < int(prev_info.min_bootstrap_input_level):
                return float("inf"), 0
            t_boot = 3.41 * math.exp(0.18 * int(self.l_eff)) + 4.81
            boot_count = int(prev_info.bootstrap_ct_count)
            return float(t_boot * boot_count), int(boot_count)
        return 0.0, 0

    def _mark_bootstrap_nodes(self, node_levels: dict[str, int]) -> dict[str, bool]:
        flags: dict[str, bool] = {}
        for node in self.network_dag.nodes:
            node_key = str(node)
            node_level = int(node_levels[node_key])
            flags[node_key] = False
            for child in self.network_dag.successors(node):
                child_level = int(node_levels[str(child)])
                _latency, boot_count = self._bootstrap_latency(node, node_level, child, child_level)
                if int(boot_count) > 0:
                    flags[node_key] = True
                    break
        return flags

    def _count_marked_bootstraps(
        self,
        node_levels: dict[str, int],
        bootstrap_nodes: dict[str, bool],
    ) -> tuple[int, list[int]]:
        total_bootstraps = 0
        bootstrapper_slots: list[int] = []
        for node in self.network_dag.nodes:
            node_key = str(node)
            if not bool(bootstrap_nodes.get(node_key, False)):
                continue
            node_level = int(node_levels[node_key])
            for child in self.network_dag.successors(node):
                child_level = int(node_levels[str(child)])
                _latency, boot_count = self._bootstrap_latency(node, node_level, child, child_level)
                if int(boot_count) <= 0:
                    continue
                total_bootstraps += int(boot_count)
                slots = int(self._info(node).bootstrap_slots)
                if int(slots) not in bootstrapper_slots:
                    bootstrapper_slots.append(int(slots))
                break
        return int(total_bootstraps), [int(value) for value in bootstrapper_slots]

    def _collect_audit(
        self,
        node_levels: dict[str, int],
        bootstrap_nodes: dict[str, bool],
    ) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        boot_edges: list[dict[str, Any]] = []
        counted_boot_nodes: set[str] = set()
        counted_bootstraps = 0
        bootstrapper_slots: list[int] = []
        for node in self.network_dag.nodes:
            info = self._info(node)
            nodes.append(
                {
                    "node": str(node),
                    "level": int(node_levels[str(node)]),
                    "module_depth": None if info.depth is None else int(info.depth),
                    "bootstrap": bool(bootstrap_nodes.get(str(node), False)),
                    "bootstrap_ct_count": int(info.bootstrap_ct_count),
                    "bootstrapper_slots": int(info.bootstrap_slots),
                    "runtime_fhe_output_shape": []
                    if info.shape is None
                    else [int(value) for value in tuple(info.shape)],
                }
            )
        for source, target in self.network_dag.edges:
            source_level = int(node_levels[str(source)])
            target_level = int(node_levels[str(target)])
            _latency, boot_count = self._bootstrap_latency(source, source_level, target, target_level)
            if int(boot_count) <= 0:
                continue
            info = self._info(source)
            slots = int(info.bootstrap_slots)
            boot_edges.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "source_level": int(source_level),
                    "target_level": int(target_level),
                    "source_depth": int(info.depth or 0),
                    "bootstrap_ct_count": int(boot_count),
                    "bootstrapper_slots": int(slots),
                }
            )
            if str(source) not in counted_boot_nodes:
                counted_boot_nodes.add(str(source))
                counted_bootstraps += int(boot_count)
                if int(slots) > 0 and int(slots) not in bootstrapper_slots:
                    bootstrapper_slots.append(int(slots))
        return {
            "l_eff": int(self.l_eff),
            "nodes": nodes,
            "bootstrap_nodes": [
                str(row["node"])
                for row in nodes
                if bool(row.get("bootstrap", False))
            ],
            "boot_edges": boot_edges,
            "bootstrap_count": int(counted_bootstraps),
            "bootstrapper_slots": [int(value) for value in bootstrapper_slots],
        }

    @staticmethod
    def _state(node: Any, level: int) -> str:
        return f"{str(node)}@l={int(level)}"

    @staticmethod
    def _split_state(state: str) -> tuple[str, int]:
        node, raw_level = str(state).rsplit("@l=", 1)
        return node, int(raw_level)

    def _node_levels_from_states(self, states: frozenset[str]) -> dict[str, int]:
        levels: dict[str, int] = {}
        for state in states:
            node, level = self._split_state(state)
            levels[str(node)] = int(level)
        missing = [str(node) for node in self.network_dag.nodes if str(node) not in levels]
        if missing:
            raise ValueError(f"bootstrap trial evaluator missing levels for nodes: {missing[:4]}")
        return levels

    def _input_level_from_high_path(self, high_path: tuple[str, ...]) -> int:
        if len(high_path) > 1:
            _node, level = self._split_state(high_path[1])
            return int(level)
        if high_path:
            _node, level = self._split_state(high_path[0])
            return int(level)
        raise ValueError("bootstrap trial evaluator produced an empty path")
