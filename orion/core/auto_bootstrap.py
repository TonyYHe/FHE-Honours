import networkx as nx
import matplotlib.pyplot as plt
from typing import Any

from .level_dag import LevelDAG
from .bootstrap_fusion import (
    install_bootstrap_prescale_fusion,
    module_bootstrap_ct_count,
    module_bootstrap_slots,
    runtime_fhe_output_shape,
)
from .bootstrap_layout_compression import apply_bootstrap_layout_compression
from orion.nn.operations import Bootstrap


_MISSING = object()


def snapshot_bootstrap_solver_assignments(network_dag: Any) -> dict[str, dict[str, Any]]:
    """Capture solver-owned level/bootstrap state before a speculative solve."""

    snapshot: dict[str, dict[str, Any]] = {}
    for node in network_dag.nodes:
        attrs = network_dag.nodes[node]
        module = attrs.get("module")
        snapshot[str(node)] = {
            "dag_level": attrs.get("level", _MISSING),
            "dag_bootstrap": attrs.get("bootstrap", _MISSING),
            "module_level": getattr(module, "level", _MISSING) if module is not None else _MISSING,
        }
    return snapshot


def reset_bootstrap_solver_assignments(
    network_dag: Any,
    snapshot: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Restore or clear bootstrap-solver assignments before a second solve.

    ``LevelDAG.estimate_layer_latency`` treats an existing ``module.level`` as a
    fixed user level.  Any boot-aware layout refinement must therefore remove
    the first solver pass' assigned levels before rerunning the solver.
    """

    snapshot = snapshot or {}
    for node in network_dag.nodes:
        attrs = network_dag.nodes[node]
        saved = snapshot.get(str(node), {})
        for key, saved_key in (("level", "dag_level"), ("bootstrap", "dag_bootstrap")):
            value = saved.get(saved_key, _MISSING)
            if value is _MISSING:
                attrs.pop(key, None)
            else:
                attrs[key] = value
        module = attrs.get("module")
        if module is not None:
            value = saved.get("module_level", _MISSING)
            if value is _MISSING:
                module.level = None
            else:
                module.level = value
    for attr in (
        "bootstrap_layout_compression_audit",
        "bootstrap_layout_refinement_audit",
        "solved_residual_level_dags",
    ):
        if hasattr(network_dag, attr):
            try:
                delattr(network_dag, attr)
            except AttributeError:
                pass


def _assigned_node_level(network_dag: Any, node: str) -> int | None:
    attrs = network_dag.nodes[node]
    value = attrs.get("level", None)
    if value is None:
        module = attrs.get("module")
        value = getattr(module, "level", None) if module is not None else None
    if value is None:
        return None
    return int(value)


def collect_bootstrap_solver_audit(network_dag: Any, *, l_eff: int) -> dict[str, Any]:
    query = LevelDAG(l_eff=int(l_eff), network_dag=network_dag, path=None)
    nodes: list[dict[str, Any]] = []
    boot_edges: list[dict[str, Any]] = []
    counted_boot_nodes: set[str] = set()
    counted_bootstraps = 0
    bootstrapper_slots: list[int] = []

    for node in network_dag.nodes:
        module = network_dag.nodes[node].get("module")
        level = _assigned_node_level(network_dag, str(node))
        depth = getattr(module, "depth", None) if module is not None else None
        bootstrap = bool(network_dag.nodes[node].get("bootstrap", False))
        shape = runtime_fhe_output_shape(module) if module is not None else None
        nodes.append(
            {
                "node": str(node),
                "level": None if level is None else int(level),
                "module_depth": None if depth is None else int(depth),
                "bootstrap": bool(bootstrap),
                "bootstrap_ct_count": int(module_bootstrap_ct_count(module)) if module is not None else 0,
                "bootstrapper_slots": int(module_bootstrap_slots(module)) if module is not None else 0,
                "runtime_fhe_output_shape": []
                if shape is None
                else [int(value) for value in tuple(shape)],
            }
        )

    for source, target in network_dag.edges:
        source_level = _assigned_node_level(network_dag, str(source))
        target_level = _assigned_node_level(network_dag, str(target))
        if source_level is None or target_level is None:
            continue
        _latency, boot_count = query.estimate_bootstrap_latency(
            f"{str(source)}@l={int(source_level)}",
            f"{str(target)}@l={int(target_level)}",
        )
        if int(boot_count) <= 0:
            continue
        module = network_dag.nodes[source].get("module")
        slots = int(module_bootstrap_slots(module)) if module is not None else 0
        boot_edges.append(
            {
                "source": str(source),
                "target": str(target),
                "source_level": int(source_level),
                "target_level": int(target_level),
                "source_depth": int(getattr(module, "depth", 0) or 0) if module is not None else 0,
                "bootstrap_ct_count": int(boot_count),
                "bootstrapper_slots": int(slots),
            }
        )
        if str(source) not in counted_boot_nodes:
            counted_boot_nodes.add(str(source))
            counted_bootstraps += int(boot_count)
            if int(slots) > 0 and int(slots) not in bootstrapper_slots:
                bootstrapper_slots.append(int(slots))

    flagged_bootstrap_nodes = [
        str(row["node"])
        for row in nodes
        if bool(row.get("bootstrap", False))
    ]
    return {
        "l_eff": int(l_eff),
        "nodes": nodes,
        "bootstrap_nodes": flagged_bootstrap_nodes,
        "boot_edges": boot_edges,
        "bootstrap_count": int(counted_bootstraps),
        "bootstrapper_slots": [int(value) for value in bootstrapper_slots],
    }


class BootstrapSolver:
    def __init__(self, net, network_dag, l_eff):
        self.net = net 
        self.network_dag = network_dag 
        self.l_eff = l_eff
        self.full_level_dag = LevelDAG(l_eff=l_eff, network_dag=network_dag)

    def extract_all_residual_subgraphs(self):
        all_residual_subgraphs = []
        for fork in self.network_dag.residuals.keys():
            subgraph = self.network_dag.extract_residual_subgraph(fork)
            all_residual_subgraphs.append(subgraph)

        return all_residual_subgraphs 
    
    def sort_residual_subgraphs(self):
        # Sort the residual subgraphs by their number of paths from fork
        # to join node.
        all_residual_subgraphs = self.extract_all_residual_subgraphs()
        
        residuals = []
        for i, (fork, join) in enumerate(self.network_dag.residuals.items()):
            subgraph = all_residual_subgraphs[i]
            paths = list(nx.all_simple_paths(subgraph, fork, join))

            unique_paths = []
            visited_children = set()
            for path in paths:
                if path[1] not in visited_children:
                    unique_paths.append(path)
                    visited_children.add(path[1])

            residuals.append((fork, paths, unique_paths))

        # Sort by the number of simple paths from fork to join in the graph.
        # This way, we're guaranteed to always solve the "inner-most"
        # residual subgraph in the event it is entirely encapsulated by
        # a larger residual connection.
        sorted_subgraphs = sorted(residuals, key=lambda x: len(x[1]))
        
        return sorted_subgraphs
       
    def first_solve_residual_subgraphs(self):
        # We'll first extract all residual subgraphs in the network and create
        # their aggregate level DAGs. We'll be iterating over DAGs sorted in 
        # increasing order by the number of paths from their corresponding fork
        # and join nodes. This guarantees we solve the "inner-most" level DAGs
        # first, which can then be inserted into subsequent calls.

        sorted_residual_subgraphs = self.sort_residual_subgraphs()
        self.network_dag.solved_residual_level_dags = {}
        
        for (fork, _, paths) in sorted_residual_subgraphs:
            aggregate_level_dag = LevelDAG(
                l_eff=self.l_eff, network_dag=self.network_dag, path=None
            )
            for path in paths:
                path_dag = nx.DiGraph()

                # Then we'll just create a new DAG by extracting the 
                # subgraph along the path.
                nodes_in_path = [
                    (node, self.network_dag.nodes[node]) 
                    for node in path
                ]
                edges_in_path = [
                    (u, v, self.network_dag[u][v])
                    for u, v in zip(path[:-1], path[1:])
                ]

                path_dag.add_nodes_from(nodes_in_path)
                path_dag.add_edges_from(edges_in_path)

                # And create the level DAG based on the path.
                aggregate_level_dag += LevelDAG(
                    l_eff=self.l_eff, network_dag=self.network_dag, path=path_dag
                )

            self.network_dag.solved_residual_level_dags[fork] = aggregate_level_dag

        return self.network_dag.solved_residual_level_dags

    def then_build_full_level_dag(self, solved_residual_level_dags):
        # We can now either append our aggregate level DAGs from residual
        # connections into the network or the next layer.

        all_forks = self.network_dag.residuals.keys()

        visited = set()
        for node in nx.topological_sort(self.network_dag):
            if node not in visited:
                if node in all_forks:
                    # It is a fork node and so this subgraph has already
                    # been solved. We'll just connect it to the existing 
                    # full_level_dag.
                    next_level_dag = solved_residual_level_dags[node]
                    subgraph = self.network_dag.extract_residual_subgraph(node)
                    visited.update(subgraph.nodes)
                else:
                    node_dag = nx.DiGraph()
                    node_dag.add_nodes_from([(node, self.network_dag.nodes[node])])  
                    next_level_dag = LevelDAG(
                        l_eff=self.l_eff, network_dag=self.network_dag, path=node_dag
                    )
                    visited.update(node)
   
                self.full_level_dag.append(next_level_dag)

    def finally_solve_full_level_dag(self):
        # Now that we've built our aggregate level DAG, we can now call 
        # one final shortest path on it to determine the optimal level
        # management policy for our network.

        heads = self.full_level_dag.head()
        tails = self.full_level_dag.tail()

        self.full_level_dag.add_node("source", weight=0) 
        self.full_level_dag.add_node("target", weight=0) 

        for head, tail in zip(heads, tails):
            self.full_level_dag.add_edge("source", head, weight=0)
            self.full_level_dag.add_edge(tail, "target", weight=0)

        shortest_path, latency = self.full_level_dag.shortest_path(
            source="source", target="target"
        )

        if latency == float("inf"):
            raise ValueError(
                "Automatic bootstrap placement failed. First try increasing "
                "the length of your LogQ moduli chain the associated "
                "parameters YAML file. If this fails, double check that the "
                "network was instantiated properly."
            )

        # Just remove the source/target we added
        shortest_path = shortest_path[1:-1]

        # The shortest path above, while correct, also black-boxes the paths
        # within skip connections. We haven't lost this data, we just need
        # to access it within edge attributes designed to track it.
        reconstructed_path = set()
        for u, v in zip(shortest_path[:-1], shortest_path[1:]):
            edge = self.full_level_dag[u][v]
            reconstructed_path.update(edge["path"])

        self.shortest_path = reconstructed_path

        input_level = int(shortest_path[1].split("=")[-1])
        return input_level

    def _reset_full_level_dag(self):
        self.full_level_dag = LevelDAG(l_eff=self.l_eff, network_dag=self.network_dag)

    def _solve_once(self, *, apply_layout_compression: bool = True):
        self._reset_full_level_dag()
        solved_residual_dags = self.first_solve_residual_subgraphs()
        self.then_build_full_level_dag(solved_residual_dags)
        input_level = self.finally_solve_full_level_dag()

        self.assign_levels_to_layers()
        num_bootstraps, bootstrapper_slots = self.mark_bootstrap_locations(
            apply_layout_compression=bool(apply_layout_compression)
        )

        return input_level, num_bootstraps, bootstrapper_slots

    def solve(self):
        return self._solve_once(apply_layout_compression=True)
    
    def assign_levels_to_layers(self):
        # Set each Orion module's attribute with it's level found by this
        # algorithm. This let's linear transforms be encoded at the 
        # correct level.
        for node in self.network_dag.nodes:
            node_module = self.network_dag.nodes[node]["module"]
            for layer in self.shortest_path:
                name = layer.split("@")[0]
                level = int(layer.split("=")[-1])
                
                if node == name:
                    self.network_dag.nodes[node]["level"] = level
                    if node_module:
                        node_module.level = level
                continue

    def _mark_bootstrap_flags(self):
        # Makes things a bit easier below
        node_map = {}
        for node in self.shortest_path:
            name = node.split("@")[0]
            node_map[name] = node

        # We'll use this empty level DAG to query whether each edge crosses
        # a bootstrap boundary. The actual count is computed after optional
        # layout compression has rewritten the bootstrap input shape.
        query = LevelDAG(
            l_eff=self.l_eff, network_dag=self.network_dag, path=None
        )

        for node in self.network_dag.nodes:
            node_w_level = node_map[node]
            children = self.network_dag.successors(node)
            self.network_dag.nodes[node]["bootstrap"] = False

            # Iterate over the layer's children to determine if their assigned
            # levels necessitate a bootstrap of the current layer.
            for child in children:
                child_w_level = node_map[child]
                _, curr_boots = query.estimate_bootstrap_latency(
                    node_w_level, child_w_level)
                if curr_boots > 0:
                    self.network_dag.nodes[node]["bootstrap"] = True
                    break

        return node_map

    def _count_marked_bootstraps(self, node_map):
        query = LevelDAG(
            l_eff=self.l_eff, network_dag=self.network_dag, path=None
        )

        total_bootstraps = 0
        bootstrapper_slots = []
        for node in self.network_dag.nodes:
            if not bool(self.network_dag.nodes[node].get("bootstrap", False)):
                continue
            node_w_level = node_map[node]
            for child in self.network_dag.successors(node):
                child_w_level = node_map[child]
                _, curr_boots = query.estimate_bootstrap_latency(
                    node_w_level, child_w_level)
                if curr_boots <= 0:
                    continue
                total_bootstraps += curr_boots
                slots = self.get_bootstrap_slots(node)
                if slots not in bootstrapper_slots:
                    bootstrapper_slots.append(slots)
                break

        return total_bootstraps, bootstrapper_slots

    def mark_bootstrap_locations(self, *, apply_layout_compression: bool = True):
        node_map = self._mark_bootstrap_flags()
        self._last_bootstrap_node_map = dict(node_map)
        if bool(apply_layout_compression):
            self.network_dag.bootstrap_layout_compression_audit = apply_bootstrap_layout_compression(
                self.network_dag
            )
        return self._count_marked_bootstraps(node_map)

    def get_bootstrap_slots(self, node):
        # If we're here, then our auto-bootstrapper has determined that the 
        # output of this node will be bootstrapped. Therefore it must be an
        # Orion module, and so a module attribute exists.
        module = self.network_dag.nodes[node]["module"]
        return int(module_bootstrap_slots(module))
    
    def plot_shortest_path(self, save_path="", figsize=(10,10)):
        """Plot the network digraph. For the best visualization, please install
        Graphviz and PyGraphviz."""

        nodes = {}
        for node in self.shortest_path:
            name = node.split("@")[0]
            level = node.split("=")[-1]
            nodes[name] = level

        network = nx.DiGraph(self.network_dag)
        shortest_graph = nx.DiGraph()

        for name, level in nodes.items():
            shortest_graph.add_node(name, level=level)

        # Add edges from the original graph
        for u, v in network.edges():
            if u in nodes and v in nodes:
                shortest_graph.add_edge(u, v)

        try:
            pos = nx.nx_agraph.graphviz_layout(shortest_graph, prog='dot')
        except:
            print("Graphviz not installed. Defaulting to worse visualization.\n")
            pos = nx.kamada_kawai_layout(shortest_graph)
        
        plt.figure(figsize=figsize)
        nx.draw(
            shortest_graph, pos, with_labels=False, arrows=True, font_size=8)

        node_labels = {
            node: f"{node}\n(level: {data['level']})"
            for node, data in shortest_graph.nodes(data=True)
        }
        nx.draw_networkx_labels(
            shortest_graph, pos, labels=node_labels, font_size=8)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


class BootstrapPlacer:
    def __init__(self, net, network_dag):
        self.net = net
        self.network_dag = network_dag
    
    def place_bootstraps(self):
        for node in self.network_dag.nodes:
            if self.network_dag.nodes[node]["bootstrap"]:
                module = self.network_dag.nodes[node]["module"]
                self._apply_bootstrap_hook(module)
    
    def _apply_bootstrap_hook(self, module):
        bootstrapper = self._create_bootstrapper(module)
        module.bootstrapper = bootstrapper
        
        # Register a forward hook that applies bootstrapping to outputs
        module.register_forward_hook(lambda mod, input, output: bootstrapper(output))
    
    def _create_bootstrapper(self, module):
        # Set bootstrap statistics to scale into [-1, 1]
        btp_input_level = module.level - module.depth
        btp_input_min = module.output_min
        btp_input_max = module.output_max
        
        bootstrapper = Bootstrap(btp_input_min, btp_input_max, btp_input_level)
        
        bootstrapper.scheme = self.net.scheme
        bootstrapper.margin = self.net.margin
        bootstrapper.fhe_input_shape = self._runtime_fhe_output_shape(module)
        bootstrapper.fit()
        bootstrapper.compile()
        install_bootstrap_prescale_fusion(module, bootstrapper)
        
        return bootstrapper

    def _runtime_fhe_output_shape(self, module):
        return runtime_fhe_output_shape(module)
