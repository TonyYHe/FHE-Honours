import sys
import math
from types import SimpleNamespace
from abc import abstractmethod

import torch
import torch.nn as nn

from .module import Module, timer
from ..core import packing


class LinearTransform(Module):
    def __init__(self, bsgs_ratio, level) -> None:
        super().__init__()
        self.bsgs_ratio = float(bsgs_ratio)
        self.set_depth(1)
        self.set_level(level)

        self.diagonals = {} # diags[(row, col)] = {0: [...], 1: [...], ...}
        self.transform_ids = {} # ids[(row, col)] = int
        self.output_rotations = 0
        self._transform_backend = None

    def __del__(self):
        backend = getattr(self, "_transform_backend", None)
        if 'sys' in globals() and sys.modules and backend is not None:
            try:
                for tid in self.transform_ids.values():
                    backend.DeleteLinearTransform(tid)
                for transform_ids in getattr(self, "_concat_transform_ids_by_input", []) or []:
                    for tid in dict(transform_ids).values():
                        backend.DeleteLinearTransform(tid)
            except Exception:
                pass # avoids errors for GC at program termination

    def extra_repr(self):
        return super().extra_repr() + f", bsgs_ratio={self.bsgs_ratio}"
            
    def init_orion_params(self):
        # Initialize additional Orion-specific weights/biases.
        self.on_weight = self.weight.data.clone()
        self.on_bias = (self.bias.data.clone() if hasattr(self, 'bias') and 
                        self.bias is not None else torch.zeros(self.weight.shape[0]))

    @abstractmethod
    def compute_fhe_output_gap(self, **kwargs):
        """Compute the multiplexed output gap."""
        pass

    @abstractmethod
    def compute_fhe_output_shape(self, **kwargs) -> tuple:
        """Compute the FHE output dimensions after multiplexing."""
        pass

    @abstractmethod
    def generate_diagonals(self, last: bool):
        pass

    def get_io_mode(self):
        return self.scheme.params.get_io_mode()

    def _compile_save_resume_enabled(self) -> bool:
        enabled = getattr(self.scheme.params, "get_compile_save_resume", None)
        return self.get_io_mode() == "save" and callable(enabled) and bool(enabled())

    def save_transforms(self):
        self.scheme.lt_evaluator.save_transforms(self)

    def load_transforms(self):
        return self.scheme.lt_evaluator.load_transforms(self) 

    def load_cached_transform_metadata(self) -> bool:
        io_mode = self.get_io_mode()
        if io_mode != "load" and not self._compile_save_resume_enabled():
            return False
        if io_mode == "load":
            self.output_rotations = self.scheme.lt_evaluator.load_transform_metadata(self)
        else:
            try:
                self.output_rotations = self.scheme.lt_evaluator.load_transform_metadata(self)
            except (KeyError, OSError, ValueError):
                return False
        self.diagonals = {}
        return True

    def compile(self):
        self.transform_ids = self.scheme.lt_evaluator.generate_transforms(self)

    @timer
    def evaluate_transforms(self, x):
        out = self.scheme.lt_evaluator.evaluate_transforms(self, x)

        # Hybrid method's output rotations
        slots = self.scheme.params.get_slots()
        for i in range(1, self.output_rotations+1):
            out += out.roll(slots // (2**i))

        out += self.on_bias_ptxt
        return out


class Linear(LinearTransform):    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        bsgs_ratio: int = 2,
        level: int = None,
    ) -> None:
        super().__init__(bsgs_ratio, level)

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, " + 
                super().extra_repr())

    def reset_parameters(self):
        # Initialize weights and biases following standard PyTorch instantiation.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def compute_fhe_output_gap(self, **kwargs):
        return 1 # linear layers in reset the multiplexed gap to 1.

    def compute_fhe_output_shape(self, **kwargs) -> tuple:
        # Linear layers also remove any padded zeros, Therefore the output 
        # shape under FHE inference is identical to cleartext inference. 
        return kwargs["clear_output_shape"]
        
    def generate_diagonals(self, last):
        if self.load_cached_transform_metadata():
            return
        # Here, we'll apply our packing strategies to return the diagonals
        # of our linear layer. When using the "hybrid" method of packing, this
        # may also require several output rotations and summations.
        self.diagonals, self.output_rotations = packing.pack_linear(self, last)
        if self.get_io_mode() == "save":
            self.save_transforms()

    def compile(self):
        # If the user specifies an I/O mode = "save" or "load", then diagonals will
        # be temporarily stored to disk to save memory. Load right before they're 
        # needed to generate the backend transforms themselves. 
        if self.get_io_mode() != "none":
            self.diagonals, self.on_bias, self.output_rotations = self.load_transforms()

        # We delay constructing the bias until now, so that any fusing can 
        # modify the bias variable beforehand.
        bias = packing.construct_linear_bias(self)
        self.on_bias_ptxt = self.scheme.encoder.encode(bias, self.level-self.depth)
        self.transform_ids = self.scheme.lt_evaluator.generate_transforms(self)
        self._transform_backend = self.scheme.backend
    
    def forward(self, x):
        if not self.he_mode:
            if x.dim() != 2:
                extra = " Forgot to call on.Flatten() first?" if x.dim() == 4 else ""
                raise ValueError(
                    f"Expected input to {self.__class__.__name__} to have "
                    f"2 dimensions (N, in_features), but got {x.dim()} " 
                    f"dimension(s): {x.shape}." + extra
        )
            
            # If we're not in FHE inference mode, then we'll just return
            # the default PyTorch result.
            return torch.nn.functional.linear(x, self.weight, self.bias)
        
        # Otherwise, call parent evaluation for FHE.
        return self.evaluate_transforms(x) 


class Conv2d(LinearTransform):    
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            bsgs_ratio: int = 2,
            level: int = None,
    ) -> None:
        super().__init__(bsgs_ratio, level)

        # Standard PyTorch Conv2d attributes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convert int parameters to tuples
        self.kernel_size = self._make_tuple(kernel_size)
        self.stride = self._make_tuple(stride)
        self.padding = self._make_tuple(padding)
        self.dilation = self._make_tuple(dilation)
        self.groups = groups
        
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def _make_tuple(self, value):
        return (value, value) if isinstance(value, int) else value

    def reset_parameters(self):
        """Initialize parameters using PyTorch's standard approach."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, dilation={self.dilation}, "
                f"groups={self.groups}, " + super().extra_repr())

    def _is_concat_cipher_tensor(self, value) -> bool:
        return bool(type(value).__name__ == "ConcatCipherTensor" and hasattr(value, "parts"))

    def _concat_fusion_specs(self):
        return tuple(getattr(self, "concat_fusion_specs", ()) or ())

    def _concat_fusion_supported(self) -> bool:
        specs = self._concat_fusion_specs()
        return bool(specs and int(getattr(self, "groups", 1)) == 1)

    def _concat_layout_top_beta(self, layout: dict) -> int:
        return max(0, int(dict(layout).get("top_beta", dict(layout).get("alpha", 0)) or 0))

    def _concat_layout_bottom_beta(self, layout: dict) -> int:
        return max(0, int(dict(layout).get("bottom_beta", dict(layout).get("beta", 0)) or 0))

    def _concat_conv_input_layout_row(self) -> dict:
        node = str(getattr(self, "name", "") or "")
        specs = self._concat_fusion_specs()
        concat_node = str(specs[0].get("concat_node", "")) if specs else ""
        runtime = getattr(self, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        compile_plan = getattr(executor, "compile_plan", None)
        if isinstance(compile_plan, dict):
            for row in compile_plan.get("edge_layouts", []):
                if str(row.get("target", "")) != node:
                    continue
                if concat_node and str(row.get("source", "")) != concat_node:
                    continue
                if str(row.get("op_kind", "")) != "conv2d":
                    continue
                return dict(row)
        return {}

    def _concat_source_input_layout(self, spec: dict) -> dict:
        layout = dict(getattr(self, "layout_policy_input_layout", {}) or {})
        if layout:
            return layout
        row = self._concat_conv_input_layout_row()
        if row:
            return dict(row.get("selected_layout", {}) or {})
        return {}

    def _concat_source_fhe_shape(self, spec: dict, layout: dict) -> torch.Size:
        if not layout:
            return torch.Size(spec["fhe_shape"])
        n, channels, height, width = (int(value) for value in spec["shape"])
        gap = max(1, int(dict(layout).get("gap", int(spec["gap"])) or int(spec["gap"])))
        top_beta = self._concat_layout_top_beta(layout)
        bottom_beta = self._concat_layout_bottom_beta(layout)
        on_channels = int(math.ceil(int(channels) / float(gap * gap)))
        return torch.Size(
            (
                int(n),
                int(on_channels),
                int(height * gap + (top_beta + bottom_beta) * gap),
                int(width * gap),
            )
        )

    def _concat_source_proxy(self, spec: dict, *, weight: torch.Tensor, name: str):
        input_layout = self._concat_source_input_layout(spec)
        input_gap = max(1, int(dict(input_layout).get("gap", int(spec["gap"])) or int(spec["gap"])))
        input_top_beta = self._concat_layout_top_beta(input_layout)
        return SimpleNamespace(
            name=str(name),
            scheme=self.scheme,
            on_weight=weight,
            on_bias=torch.zeros(int(self.out_channels), dtype=torch.float32),
            input_shape=torch.Size(spec["shape"]),
            output_shape=self.output_shape,
            fhe_input_shape=self._concat_source_fhe_shape(spec, input_layout),
            fhe_output_shape=self.fhe_output_shape,
            input_gap=int(input_gap),
            output_gap=int(self.output_gap),
            groups=1,
            in_channels=int(spec["channels"]),
            out_channels=int(self.out_channels),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bsgs_ratio=float(self.bsgs_ratio),
            level=self.level,
            layout_policy_input_layout=dict(input_layout),
            layout_policy_input_row_offset=int(input_top_beta * input_gap),
            layout_policy_selected_input_layout=dict(input_layout),
        )

    def _generate_concat_fusion_diagonals(self, last: bool) -> bool:
        if not self._concat_fusion_supported():
            return False
        self._concat_diagonals_by_input = []
        self._concat_transform_ids_by_input = []
        self._concat_output_rotations = 0
        for input_index, spec in enumerate(self._concat_fusion_specs()):
            start = int(spec["channel_start"])
            end = int(spec["channel_end"])
            weight = self.on_weight[:, start:end, :, :].detach().clone()
            proxy = self._concat_source_proxy(
                spec,
                weight=weight,
                name=f"{getattr(self, 'name', self.__class__.__name__)}_concat_source_{int(input_index)}",
            )
            diagonals, output_rotations = packing.pack_conv2d(proxy, bool(last))
            if int(input_index) == 0:
                self._concat_output_rotations = int(output_rotations)
            elif int(output_rotations) != int(self._concat_output_rotations):
                raise RuntimeError(
                    f"concat-fused Conv2d {getattr(self, 'name', '')} produced inconsistent output rotations"
                )
            self._concat_diagonals_by_input.append(diagonals)
        self.diagonals = {}
        self.output_rotations = int(self._concat_output_rotations)
        return True

    def _compile_concat_fusion_transforms(self) -> bool:
        if not self._concat_fusion_supported():
            return False
        diagonals_by_input = list(getattr(self, "_concat_diagonals_by_input", []) or [])
        if not diagonals_by_input:
            self._generate_concat_fusion_diagonals(last=False)
            diagonals_by_input = list(getattr(self, "_concat_diagonals_by_input", []) or [])
        bias = packing.construct_conv2d_bias(self)
        self.on_bias_ptxt = self.scheme.encoder.encode(bias, self.level - self.depth)
        self._concat_transform_ids_by_input = []
        for input_index, diagonals in enumerate(diagonals_by_input):
            proxy = SimpleNamespace(
                name=f"{getattr(self, 'name', self.__class__.__name__)}_concat_source_{int(input_index)}",
                diagonals=diagonals,
                level=int(self.level),
                bsgs_ratio=float(self.bsgs_ratio),
                scheme=self.scheme,
                output_shape=self.output_shape,
                fhe_output_shape=self.fhe_output_shape,
            )
            self._concat_transform_ids_by_input.append(dict(self.scheme.lt_evaluator.generate_transforms(proxy)))
        self.transform_ids = {}
        self._transform_backend = self.scheme.backend
        return True

    def _concat_fusion_ready(self) -> bool:
        return bool(getattr(self, "_concat_transform_ids_by_input", None))

    def _evaluate_concat_fusion(self, concat_tensor):
        parts = tuple(concat_tensor.parts)
        transform_ids_by_input = list(getattr(self, "_concat_transform_ids_by_input", []) or [])
        if len(parts) != len(transform_ids_by_input):
            raise RuntimeError(
                f"concat-fused Conv2d {getattr(self, 'name', '')} expected "
                f"{len(transform_ids_by_input)} inputs, got {len(parts)}"
            )
        out = None
        for input_index, source in enumerate(parts):
            proxy = SimpleNamespace(
                name=f"{getattr(self, 'name', self.__class__.__name__)}_concat_source_{int(input_index)}",
                transform_ids=dict(transform_ids_by_input[int(input_index)]),
                level=int(self.level),
                output_shape=self.output_shape,
                fhe_output_shape=self.fhe_output_shape,
            )
            partial = self.scheme.lt_evaluator.evaluate_transforms(proxy, source)
            if out is None:
                out = partial
            else:
                if bool(getattr(out.scheme.backend, "align_addition_scales", False)):
                    scale = max(1, int(out.scale()))
                    out.set_scale(int(scale))
                    partial.set_scale(int(scale))
                out = out + partial
        slots = self.scheme.params.get_slots()
        for i in range(1, int(self.output_rotations) + 1):
            out += out.roll(slots // (2**i))
        out += self.on_bias_ptxt
        release_owned = getattr(concat_tensor, "release_owned_parts", None)
        if callable(release_owned):
            release_owned()
        return out

    def compute_fhe_output_gap(self, **kwargs):
        # Strided convolutions increase the multiplexed gap by a factor 
        # of the stride.
        input_gap = kwargs['input_gap']  
        return input_gap * self.stride[0]
    
    def compute_fhe_output_shape(self, **kwargs) -> tuple:
        input_shape = kwargs['input_shape']
        clear_output_shape = kwargs['clear_output_shape']
        input_gap = kwargs['input_gap']

        Hi, Wi = input_shape[2:]
        N, Co, Ho, Wo = clear_output_shape
        output_gap = self.compute_fhe_output_gap(input_gap=input_gap)
        
        on_Co = math.ceil(Co / (output_gap**2))
        on_Ho = max(Hi, Ho*output_gap)
        on_Wo = max(Wi, Wo*output_gap)

        return torch.Size((N, on_Co, on_Ho, on_Wo))
    
    def generate_diagonals(self, last):
        if bool(getattr(self, "region_first_probe_dense_bypass", False)):
            self.diagonals = {}
            self.output_rotations = 0
            return
        if self._generate_concat_fusion_diagonals(last=bool(last)):
            return
        runtime = getattr(self, "region_runtime", None)
        runtime_supported = bool(runtime is not None and getattr(runtime, "supports_scheme", lambda _scheme: True)(self.scheme))
        if runtime is not None and bool(getattr(runtime, "executable", False)) and bool(getattr(self, "region_first_skip_dense_pack", False)) and bool(runtime_supported):
            self.diagonals = {}
            self.output_rotations = 0
            return
        if self.load_cached_transform_metadata():
            return
        # Generate Toeplitz diagonals and determine the number of output
        # rotations if the `hybrid` packing method is used.
        self.diagonals, self.output_rotations = packing.pack_conv2d(self, last)
        if self.get_io_mode() == "save":
            self.save_transforms()

    def compile(self):
        if bool(getattr(self, "region_first_probe_dense_bypass", False)):
            self.transform_ids = {}
            return
        if self._compile_concat_fusion_transforms():
            return
        runtime = getattr(self, "region_runtime", None)
        runtime_supported = bool(runtime is not None and getattr(runtime, "supports_scheme", lambda _scheme: True)(self.scheme))
        if runtime is not None and bool(getattr(runtime, "executable", False)) and bool(getattr(self, "region_first_skip_dense_pack", False)) and bool(runtime_supported):
            runtime.assigned_level = int(self.level)
            runtime_depth = int(getattr(runtime, "depth", self.depth or 0) or 0)
            runtime.assigned_depth = int(runtime_depth)
            if getattr(runtime, "executor", None) is not None and hasattr(runtime.executor, "assigned_level"):
                runtime.executor.assigned_level = int(self.level)
            if getattr(runtime, "executor", None) is not None and hasattr(runtime.executor, "assigned_depth"):
                runtime.executor.assigned_depth = int(runtime_depth)
            if not bool(getattr(self, "region_first_probe_lazy_region_compile", False)):
                runtime.compile(self.scheme)
            self.transform_ids = {}
            return
        # If the user specifies an io mode = "save" or "load", then diagonals will
        # be temporarily stored to disk to save memory. Load right before they're 
        # needed to generate the backend transforms themselves. 
        if self.get_io_mode() != "none":
            self.diagonals, self.on_bias, self.output_rotations = self.load_transforms()

        # We delay constructing the bias until now, so that any fusing can 
        # modify the bias variable beforehand.
        bias = packing.construct_conv2d_bias(self)
        self.on_bias_ptxt = self.scheme.encoder.encode(bias, self.level-self.depth)
        self.transform_ids = self.scheme.lt_evaluator.generate_transforms(self)
        self._transform_backend = self.scheme.backend

    def forward(self, x):
        # Forward pass that handles both cleartext and FHE inference.
        if self.he_mode and self._is_concat_cipher_tensor(x):
            if self._concat_fusion_ready():
                return self._evaluate_concat_fusion(x)
            x = x.materialize()
        runtime = getattr(self, "region_runtime", None)
        runtime_supported = bool(runtime is not None and getattr(runtime, "supports_scheme", lambda _scheme: True)(self.scheme))
        if self.he_mode and runtime is not None and bool(getattr(runtime, "executable", False)) and bool(runtime_supported):
            output_id = getattr(self, "region_output_id", None) or getattr(self, "name", self.__class__.__name__)
            return runtime.output(str(output_id), x)
        if self.he_mode and bool(getattr(self, "region_first_probe_dense_bypass", False)):
            from orion.backend.python.tensors import CipherTensor

            zeros = torch.zeros(self.fhe_output_shape, dtype=torch.float32)
            input_level = x.level() if hasattr(x, "level") else len(self.scheme.params.get_logq()) - 1
            output_level = max(0, int(input_level) - int(self.depth or 0))
            ptxt = self.scheme.encode(zeros, output_level)
            ctxt = self.scheme.encrypt(ptxt)
            ids = [int(value) for value in ctxt.ids]
            ctxt.ids = []
            return CipherTensor(self.scheme, ids, self.output_shape, self.fhe_output_shape)

        if not self.he_mode: # cleartext mode
            if x.dim() != 4:
                raise ValueError(
                    f"Expected input to {self.__class__.__name__} to have "
                    f" 4 dimensions (N, C, H, W), but got {x.dim()} "
                    f"dimension(s): {x.shape}."
                )
            return torch.nn.functional.conv2d(
                x, self.weight, self.bias, self.stride, 
                self.padding, self.dilation, self.groups
            )
        
        return self.evaluate_transforms(x) # FHE mode


class ConvTranspose2d(LinearTransform):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            output_padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            bsgs_ratio: int = 2,
            level: int = None,
    ) -> None:
        super().__init__(bsgs_ratio, level)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = self._make_tuple(kernel_size)
        self.stride = self._make_tuple(stride)
        self.padding = self._make_tuple(padding)
        self.output_padding = self._make_tuple(output_padding)
        self.dilation = self._make_tuple(dilation)
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def _make_tuple(self, value):
        return (value, value) if isinstance(value, int) else value

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, output_padding={self.output_padding}, "
            f"dilation={self.dilation}, groups={self.groups}, "
            + super().extra_repr()
        )

    def init_orion_params(self):
        self.on_weight = self.weight.data.clone()
        self.on_bias = (
            self.bias.data.clone()
            if hasattr(self, "bias") and self.bias is not None
            else torch.zeros(self.out_channels)
        )

    def compute_fhe_output_gap(self, **kwargs):
        input_gap = kwargs["input_gap"]
        return max(1, input_gap // self.stride[0])

    def compute_fhe_output_shape(self, **kwargs) -> tuple:
        clear_output_shape = kwargs["clear_output_shape"]
        output_gap = self.compute_fhe_output_gap(input_gap=kwargs["input_gap"])

        N, Co, Ho, Wo = clear_output_shape
        on_Co = math.ceil(Co / (output_gap ** 2))
        on_Ho = Ho * output_gap
        on_Wo = Wo * output_gap

        return torch.Size((N, on_Co, on_Ho, on_Wo))

    def generate_diagonals(self, last):
        runtime = getattr(self, "region_runtime", None)
        runtime_supported = bool(runtime is not None and getattr(runtime, "supports_scheme", lambda _scheme: True)(self.scheme))
        if runtime is not None and bool(getattr(runtime, "executable", False)) and bool(getattr(self, "region_first_skip_dense_pack", False)) and bool(runtime_supported):
            self.diagonals = {}
            self.output_rotations = 0
            return
        if self.load_cached_transform_metadata():
            return
        self.diagonals, self.output_rotations = packing.pack_conv_transpose2d(self, last)
        if self.get_io_mode() == "save":
            self.save_transforms()

    def compile(self):
        runtime = getattr(self, "region_runtime", None)
        runtime_supported = bool(runtime is not None and getattr(runtime, "supports_scheme", lambda _scheme: True)(self.scheme))
        if runtime is not None and bool(getattr(runtime, "executable", False)) and bool(getattr(self, "region_first_skip_dense_pack", False)) and bool(runtime_supported):
            runtime.assigned_level = int(self.level)
            runtime_depth = int(getattr(runtime, "depth", self.depth or 0) or 0)
            runtime.assigned_depth = int(runtime_depth)
            if getattr(runtime, "executor", None) is not None and hasattr(runtime.executor, "assigned_level"):
                runtime.executor.assigned_level = int(self.level)
            if getattr(runtime, "executor", None) is not None and hasattr(runtime.executor, "assigned_depth"):
                runtime.executor.assigned_depth = int(runtime_depth)
            runtime.compile(self.scheme)
            self.transform_ids = {}
            return
        if self.get_io_mode() != "none":
            self.diagonals, self.on_bias, self.output_rotations = self.load_transforms()

        bias = packing.construct_conv_transpose2d_bias(self)
        self.on_bias_ptxt = self.scheme.encoder.encode(bias, self.level - self.depth)
        self.transform_ids = self.scheme.lt_evaluator.generate_transforms(self)
        self._transform_backend = self.scheme.backend

    def forward(self, x):
        runtime = getattr(self, "region_runtime", None)
        runtime_supported = bool(runtime is not None and getattr(runtime, "supports_scheme", lambda _scheme: True)(self.scheme))
        if self.he_mode and runtime is not None and bool(getattr(runtime, "executable", False)) and bool(runtime_supported):
            output_id = getattr(self, "region_output_id", None) or getattr(self, "name", self.__class__.__name__)
            return runtime.output(str(output_id), x)
        if not self.he_mode:
            if x.dim() != 4:
                raise ValueError(
                    f"Expected input to {self.__class__.__name__} to have "
                    f"4 dimensions (N, C, H, W), but got {x.dim()} "
                    f"dimension(s): {x.shape}."
                )

            return torch.nn.functional.conv_transpose2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )

        return self.evaluate_transforms(x)
