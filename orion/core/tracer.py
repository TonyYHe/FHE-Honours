import warnings

import torch
import torch.nn as nn
import torch.fx as fx

import orion.nn as on
from orion.nn.module import Module
from orion.nn.linear import LinearTransform
from orion.nn.normalization import BatchNormNd


class OrionTracer(fx.Tracer):
    """
    Overrides the default fx.Tracer that does not recursively access all
    modules in the network. This is a deeper trace.
    """
    def is_leaf_module(self, m, _):
        if not isinstance(m, nn.Module):
            return False
        if isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            return False
        return not any(True for _ in m.children())
    
    def trace_model(self, model):
        # Tracing outputs are slightly different when the user provides
        # a leaf module (e.g on.Conv2d) rather than a network. We'll wrap
        # it temporarily to consistently track FHE statistics.
        if self.is_leaf_module(model, ""):
            model = ModuleWrapper(model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fx.GraphModule(model, super().trace(model))


class ModuleWrapper(on.Module):
    """Wrapper for leaf modules to make them traceable."""
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, x):
        return self.module(x)


class StatsTracker(fx.Interpreter):
    """Tracks important FHE statistics. """

    def __init__(self, module: fx.GraphModule) -> None:
        super().__init__(module)
        self._init_node_attributes()

    def _init_node_attributes(self):
        # Tracks min/max values and shapes for FHE-friendly inference
        for node in self.module.graph.nodes:
            node.input_min = float("inf")
            node.input_max = float("-inf")
            node.output_min = float("inf") 
            node.output_max = float("-inf")
            node.input_shape = None
            node.output_shape = None
            node.fhe_input_shape = None
            node.fhe_output_shape = None
            node.input_gap = 1
            node.output_gap = 1
        
    def run_node(self, node: fx.Node):
        # Run one node and track its input/output stats
        self._validate_node(node)
        
        inp = self.map_nodes_to_values(node.args, node)
        if inp: 
            self.update_input_stats(inp, node)

        result = super().run_node(node)  # Forward pass the node
        self.update_output_stats(result, node)
        
        if node.op == "call_module":
            module = self.module.get_submodule(node.target)
            if isinstance(module, Module):
                self.sync_module_attributes(node)

        return result
    
    def _validate_node(self, node):
        # Validate that the layer works under FHE
        self._validate_shapes_and_gaps(node)

        if node.op == "call_module":
            self._validate_module_properties(node)
    
    def _validate_shapes_and_gaps(self, node):
        # Ensure consistent shapes and gaps across inputs
        parents = node.all_input_nodes
        if not parents:
            return

        if self._is_concat_node(node):
            shapes = [getattr(p, "output_shape") for p in parents]
            fhe_shapes = [getattr(p, "fhe_output_shape") for p in parents]
            gaps = [getattr(p, "output_gap") for p in parents]
            if any(shape is None for shape in shapes) or any(shape is None for shape in fhe_shapes):
                return
            dim = self._concat_dim(node)
            if int(dim) != 1:
                raise ValueError(f"Concat node {node.name} only supports channel dim=1 under FHE")
            ranks = {len(shape) for shape in shapes}
            if len(ranks) > 1:
                raise ValueError(f"Inconsistent ranks for concat {node.name}: {shapes}")
            for axis in range(len(shapes[0])):
                if int(axis) == int(dim):
                    continue
                values = {int(shape[axis]) for shape in shapes}
                if len(values) > 1:
                    raise ValueError(f"Inconsistent concat non-channel shapes for {node.name}: {shapes}")
            if len({int(gap) for gap in gaps}) > 1:
                raise ValueError(f"Inconsistent concat input gaps for {node.name}: {set(gaps)}")
            for axis in (0, 2, 3):
                values = {int(shape[axis]) for shape in fhe_shapes}
                if len(values) > 1:
                    raise ValueError(f"Inconsistent concat FHE physical axes for {node.name}: {fhe_shapes}")
            return
            
        # Helper function to check consistency
        def check_consistency(attr_name, label):
            values = [getattr(p, attr_name) for p in parents 
                     if getattr(p, attr_name) is not None]
            if len(set(values)) > 1:
                raise ValueError(
                    f"Inconsistent {label} for {node.name}: {set(values)}"
                )
        
        # Check all required consistencies
        check_consistency('output_shape', 'input shapes')
        check_consistency('fhe_output_shape', 'FHE shapes')
        check_consistency('output_gap', 'input gaps')
    
    def _validate_module_properties(self, node):
        # Check module-specific FHE compatibility requirements
        submodule = self.module.get_submodule(node.target)
        
        # Check stride equality in pooling layers
        stride = getattr(submodule, "stride", None)
        if stride and len(set(stride)) > 1:
            raise ValueError(
                f"Stride for {node.name} must be equal in all directions: {stride}"
            )
        
        # Check BatchNorm parent count
        is_batchnorm = isinstance(submodule, BatchNormNd)
        has_multiple_parents = len(node.all_input_nodes) > 1
        
        if is_batchnorm and has_multiple_parents:
            raise ValueError(
                f"BatchNorm node {node} has multiple parents which prevents fusion"
            )

    def _is_concat_node(self, node: fx.Node) -> bool:
        if node.op != "call_module":
            return False
        try:
            module = self.module.get_submodule(node.target)
        except AttributeError:
            return False
        return isinstance(module, on.Concat)

    def _concat_dim(self, node: fx.Node) -> int:
        module = self.module.get_submodule(node.target)
        dim = int(getattr(module, "dim", 1))
        parents = node.all_input_nodes
        rank = len(getattr(parents[0], "output_shape", ())) if parents else 0
        if dim < 0:
            dim += int(rank)
        return int(dim)
    
    def update_input_stats(self, inp: tuple, node: fx.Node):
        # Update input statistics from actual tensor values
        min_values = []
        max_values = []
        
        for e in inp:
            if isinstance(e, torch.Tensor):
                min_values.append(e.detach().min())
                max_values.append(e.detach().max())
            else: # scalars
                scalar_tensor = torch.tensor(e)
                min_values.append(scalar_tensor)
                max_values.append(scalar_tensor)

        current_min = min(min_values)
        current_max = max(max_values)
        node.input_min = min(node.input_min, current_min)
        node.input_max = max(node.input_max, current_max)
        
        # Set input shape from parent's output shape for structure preservation
        if node.all_input_nodes:
            parent = node.all_input_nodes[0]
            node.input_shape = parent.output_shape
            node.input_gap = parent.output_gap
            node.fhe_input_shape = parent.fhe_output_shape
        else:
            # For input nodes with no parents, use actual tensor shape
            node.input_shape = inp[0].shape

    def update_output_stats(self, result: torch.Tensor, node: fx.Node):
        # Update output statistics based on actual result tensor
        node.output_min = min(node.output_min, result.min())
        node.output_max = max(node.output_max, result.max())
        
        # Determine appropriate output shape based on module type
        node.output_shape = self.compute_clear_output_shape(node, result)
        node.fhe_output_shape = self.compute_fhe_output_shape(node)
        node.output_gap = self.compute_fhe_output_gap(node)

    def compute_clear_output_shape(self, node: fx.Node, result):
        # Determine output shape, preserving structure except for transforming ops
        if not node.input_shape:
            return result.shape

        if self._is_concat_node(node):
            return result.shape
            
        # Only LinearTransform modules change the output shape
        if node.op == "call_module":
            module = self.module.get_submodule(node.target)
            if isinstance(module, LinearTransform):
                return result.shape
                
        # For all other modules, preserve the input shape
        return node.input_shape

    def compute_fhe_output_gap(self, node: fx.Node):
        if self._is_concat_node(node):
            parents = node.all_input_nodes
            return parents[0].output_gap if parents else node.input_gap
        if node.op == "call_module":
            module = self.module.get_submodule(node.target)
            if isinstance(module, LinearTransform):
                return module.compute_fhe_output_gap(
                    input_gap=node.input_gap,
                    input_shape=node.input_shape,
                    output_shape=node.output_shape,
                )
        return node.input_gap
        
    def compute_fhe_output_shape(self, node: fx.Node):
        if not node.input_shape:
            return node.output_shape

        if self._is_concat_node(node):
            parents = node.all_input_nodes
            if not parents:
                return node.output_shape
            output_shape = node.output_shape
            gap = int(parents[0].output_gap)
            n = int(output_shape[0])
            channels = int(output_shape[1])
            height = int(output_shape[2])
            width = int(output_shape[3])
            on_channels = (int(channels) + int(gap * gap) - 1) // int(gap * gap)
            return torch.Size((n, on_channels, int(height * gap), int(width * gap)))

        if node.op == "call_module":
            module = self.module.get_submodule(node.target)
            if isinstance(module, LinearTransform):
                return module.compute_fhe_output_shape(
                    input_gap=node.input_gap,
                    input_shape=node.input_shape,
                    output_shape=node.output_shape,
                    fhe_input_shape=node.fhe_input_shape,
                    output_gap=node.output_gap,
                    clear_output_shape=node.output_shape
                )
        return node.fhe_input_shape

    def sync_module_attributes(self, node: fx.Node):
        # Sync tracked node statistics to the corresponding module
        module = self.module.get_submodule(node.target)
        module.name = node.name

        # Min/max values
        module.input_min = node.input_min
        module.input_max = node.input_max
        module.output_min = node.output_min
        module.output_max = node.output_max
        
        # Shapes
        module.input_shape = node.input_shape 
        module.output_shape = node.output_shape 
        module.fhe_input_shape = node.fhe_input_shape
        module.fhe_output_shape = node.fhe_output_shape
        
        # Multiplexed aps
        module.input_gap = node.input_gap
        module.output_gap = node.output_gap

        if isinstance(module, on.Concat):
            parents = node.all_input_nodes
            module.configure_from_stats(
                input_shapes=[parent.output_shape for parent in parents],
                input_fhe_shapes=[parent.fhe_output_shape for parent in parents],
                input_gaps=[parent.output_gap for parent in parents],
                output_shape=node.output_shape,
                fhe_output_shape=node.fhe_output_shape,
                output_gap=node.output_gap,
            )

    def update_batch_size(self, batch_size):
        for node in self.module.graph.nodes:        
            if node.op == "call_module":
                module = self.module.get_submodule(node.target)

                shape_attrs = [
                    'input_shape', 
                    'output_shape', 
                    'fhe_input_shape', 
                    'fhe_output_shape'
                ]
                
                # Update only batch dimension
                for attr in shape_attrs:
                    current_shape = getattr(module, attr)
                    new_shape = torch.Size([batch_size] + list(current_shape[1:]))
                    setattr(module, attr, new_shape)

    def propagate(self, *args) -> None:
        # Run the graph with the provided inputs
        self.run(*args)
