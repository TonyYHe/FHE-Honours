import sys
import math
import os
import time
from types import SimpleNamespace
from abc import abstractmethod

import torch
import torch.nn as nn

from .module import Module, timer
from ..core import packing


class _temporary_attrs:
    def __init__(self, target, **attrs) -> None:
        self.target = target
        self.attrs = dict(attrs)
        self.old_values = {}
        self.missing = set()

    def __enter__(self):
        for name, value in self.attrs.items():
            if hasattr(self.target, str(name)):
                self.old_values[str(name)] = getattr(self.target, str(name))
            else:
                self.missing.add(str(name))
            setattr(self.target, str(name), value)
        return self.target

    def __exit__(self, exc_type, exc, tb) -> None:
        for name in self.attrs:
            name = str(name)
            if name in self.missing:
                try:
                    delattr(self.target, name)
                except AttributeError:
                    pass
            else:
                setattr(self.target, name, self.old_values[name])


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
                for tid in getattr(self, "_dense_layer_cache_active_transform_ids", {}).values():
                    backend.DeleteLinearTransform(tid)
                for transform_ids in getattr(self, "_concat_transform_ids_by_input", []) or []:
                    for tid in dict(transform_ids).values():
                        backend.DeleteLinearTransform(tid)
                for groups_by_source in getattr(self, "_concat_unified_groups_by_input", []) or []:
                    for group in dict(groups_by_source).values():
                        cleanup = getattr(group, "cleanup", None)
                        if callable(cleanup):
                            cleanup(backend)
                for proxy in getattr(self, "_concat_transform_sources_by_input", []) or []:
                    for tid in getattr(proxy, "_dense_layer_cache_active_transform_ids", {}).values():
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

    def _install_single_slot_payload_recipe(
        self,
        *,
        diag_indices_by_block,
        build_diagonals,
        build_block_diagonals=None,
        build_block_payloads=None,
    ) -> None:
        indices = {
            (int(row), int(col)): tuple(int(value) for value in values)
            for (row, col), values in dict(diag_indices_by_block).items()
        }
        self._dense_layer_cache_diag_indices_by_block = indices
        self._dense_layer_cache_build_diagonals = build_diagonals
        self._dense_layer_cache_build_block_diagonals = build_block_diagonals
        self._dense_layer_cache_build_block_payloads = build_block_payloads
        self._single_slot_diag_indices_by_block = indices
        self._single_slot_build_diagonals = build_diagonals
        self._single_slot_build_block_diagonals = build_block_diagonals
        self._single_slot_build_block_payloads = build_block_payloads
        self.diagonals = {}

    def compile(self):
        self.transform_ids = self.scheme.lt_evaluator.generate_transforms(self)

    def _publish_linear_wrapper_timing(
        self,
        timing: dict,
        *,
        accumulate_s: float = 0.0,
        rescale_s: float = 0.0,
        bias_s: float = 0.0,
        output_rotation_s: float = 0.0,
        serving_extra_s: float | None = None,
    ) -> dict:
        timing = dict(timing or {})
        timing["linear_wrapper_accumulate_s"] = float(
            timing.get("linear_wrapper_accumulate_s", 0.0) + float(accumulate_s)
        )
        timing["linear_wrapper_rescale_s"] = float(
            timing.get("linear_wrapper_rescale_s", 0.0) + float(rescale_s)
        )
        timing["linear_wrapper_bias_s"] = float(timing.get("linear_wrapper_bias_s", 0.0) + float(bias_s))
        timing["linear_wrapper_output_rotation_s"] = float(
            timing.get("linear_wrapper_output_rotation_s", 0.0) + float(output_rotation_s)
        )
        timing["linear_wrapper_postprocess_s"] = float(
            timing.get("linear_wrapper_accumulate_s", 0.0)
            + timing.get("linear_wrapper_rescale_s", 0.0)
            + timing.get("linear_wrapper_bias_s", 0.0)
            + timing.get("linear_wrapper_output_rotation_s", 0.0)
        )
        if serving_extra_s is None:
            serving_extra_s = float(accumulate_s) + float(rescale_s) + float(bias_s) + float(output_rotation_s)
        timing["serving_hot_s"] = float(timing.get("serving_hot_s", 0.0) + float(serving_extra_s))
        runtime_fairness_mode = str(timing.get("runtime_fairness_mode", "") or "")
        if not runtime_fairness_mode:
            timing["runtime_fairness_mode"] = "linear_wrapper_only"
        self._last_runtime_timing = dict(timing)
        self.scheme.lt_evaluator.last_runtime_timing = dict(timing)
        return timing

    @timer
    def evaluate_transforms(self, x):
        layer_cache_active = False
        if self.scheme.lt_evaluator.dense_layer_cache_needs_materialize(self):
            self.scheme.lt_evaluator.materialize_dense_layer_cache(self)
            layer_cache_active = True
        try:
            out = self.scheme.lt_evaluator.evaluate_transforms(self, x)
            timing = dict(getattr(self, "_last_runtime_timing", {}) or {})

            # Hybrid method's output rotations
            slots = self.scheme.params.get_slots()
            output_rotation_started = time.perf_counter()
            for i in range(1, self.output_rotations + 1):
                out += out.roll(slots // (2**i))
            output_rotation_s = float(time.perf_counter() - output_rotation_started)

            bias_s = 0.0
            if self.on_bias_ptxt is not None:
                bias_started = time.perf_counter()
                out += self.on_bias_ptxt
                bias_s = float(time.perf_counter() - bias_started)
            self._publish_linear_wrapper_timing(
                timing,
                bias_s=float(bias_s),
                output_rotation_s=float(output_rotation_s),
                serving_extra_s=float(bias_s + output_rotation_s),
            )
            return out
        finally:
            if bool(layer_cache_active):
                evict_s = self.scheme.lt_evaluator.evict_dense_layer_cache(self, clear_bias=True)
                self.scheme.lt_evaluator.finish_dense_layer_cache_runtime(self, evict_s)
            elif (
                getattr(self, "_dense_layer_cache_deferred", False)
                and self.scheme.lt_evaluator.dense_layer_cache_granularity() != "layer"
                and hasattr(self, "on_bias_ptxt")
            ):
                self.on_bias_ptxt = None


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
        if self.scheme.lt_evaluator.dense_layer_cache_enabled_for(self):
            self._dense_layer_cache_bias = bias.detach().clone()
            self.on_bias_ptxt = None
        else:
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
        raw_enabled = os.environ.get("ORION_CONCAT_FUSION", "1").strip().lower()
        if raw_enabled in ("", "0", "false", "no", "off"):
            return False
        specs = self._concat_fusion_specs()
        return bool(specs and int(getattr(self, "groups", 1)) == 1)

    def _concat_fusion_unified_supported(self) -> bool:
        if not self._concat_fusion_supported():
            return False
        params = getattr(getattr(self, "scheme", None), "params", None)
        backend = getattr(getattr(self, "scheme", None), "backend", None)
        if params is None or backend is None:
            return False
        if not callable(getattr(backend, "GenerateLinearTransformsUnified", None)):
            return False
        if int(getattr(self, "output_shape", (1,))[0]) != 1:
            return False
        for spec in self._concat_fusion_specs():
            shape = tuple(int(value) for value in spec.get("shape", ()))
            if len(shape) < 4 or int(shape[0]) != 1:
                return False
            physical_layout = self._concat_source_input_physical_layout(dict(spec))
            if physical_layout and str(physical_layout) not in {"packed_compact", "logical_halo_compact"}:
                return False
        output_attrs = self._concat_output_layout_attrs()
        output_layout = dict(output_attrs.get("layout_policy_output_layout", {}) or {})
        output_has_halo = bool(
            self._concat_layout_top_beta(output_layout) > 0
            or self._concat_layout_bottom_beta(output_layout) > 0
        )
        materialization = str(output_attrs.get("layout_policy_output_materialization", "") or "")
        if bool(output_has_halo) and materialization not in {
            "fused_relayout",
            "native_halo_stripe",
            "native_stripe",
            "channel_aligned_native_stripe",
        }:
            return False
        kernel_h, kernel_w = (int(value) for value in self.kernel_size)
        stride_h, stride_w = (int(value) for value in self.stride)
        pad_h, pad_w = (int(value) for value in self.padding)
        dil_h, dil_w = (int(value) for value in self.dilation)
        return bool(
            int(kernel_h) == int(kernel_w)
            and int(stride_h) == int(stride_w)
            and int(pad_h) == int(pad_w)
            and int(dil_h) == int(dil_w)
        )

    def _concat_layout_top_beta(self, layout: dict) -> int:
        return max(0, int(dict(layout).get("top_beta", dict(layout).get("alpha", 0)) or 0))

    def _concat_layout_bottom_beta(self, layout: dict) -> int:
        return max(0, int(dict(layout).get("bottom_beta", dict(layout).get("beta", 0)) or 0))

    def _concat_layout_policy_module_attrs(self) -> dict:
        runtime = getattr(self, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        getter = getattr(executor, "_native_halo_module_attrs", None)
        if callable(getter):
            try:
                attrs = getter()
            except Exception:
                attrs = {}
            if isinstance(attrs, dict):
                return dict(attrs)
        return {}

    def _concat_layout_compile_plan(self) -> dict:
        runtime = getattr(self, "region_runtime", None)
        executor = getattr(runtime, "executor", None)
        compile_plan = getattr(executor, "compile_plan", None)
        return dict(compile_plan) if isinstance(compile_plan, dict) else {}

    def _concat_zero_halo_layout(self, layout: dict) -> dict:
        updated = dict(layout)
        updated.pop("alpha", None)
        updated.pop("beta", None)
        updated["top_beta"] = 0
        updated["bottom_beta"] = 0
        if "core_slots" in updated:
            updated["stored_slots"] = int(updated.get("core_slots", 0) or 0)
        return updated

    def _concat_conv_input_layout_row(self) -> dict:
        node = str(getattr(self, "name", "") or "")
        specs = self._concat_fusion_specs()
        concat_node = str(specs[0].get("concat_node", "")) if specs else ""
        compile_plan = self._concat_layout_compile_plan()
        for row in compile_plan.get("edge_layouts", []):
            if str(row.get("target", "")) != node:
                continue
            if concat_node and str(row.get("source", "")) != concat_node:
                continue
            if str(row.get("op_kind", "")) != "conv2d":
                continue
            return dict(row)
        return {}

    def _concat_join_input_layout_row(self, spec: dict) -> dict:
        concat_node = str(spec.get("concat_node", "") or "")
        source = str(spec.get("source", "") or "")
        if not concat_node or not source:
            return {}
        compile_plan = self._concat_layout_compile_plan()
        for row in compile_plan.get("edge_layouts", []):
            if str(row.get("target", "")) != concat_node:
                continue
            if str(row.get("source", "")) != source:
                continue
            if str(row.get("op_kind", "")) != "concat":
                continue
            return dict(row)
        return {}

    def _concat_physical_input_layout_from_join_row(self, row: dict) -> dict:
        selected_layout = dict(row.get("selected_layout", {}) or {})
        if bool(row.get("relayout", False)):
            return dict(selected_layout)
        source_layout = dict(row.get("source_layout", {}) or selected_layout)
        source_physical = str(row.get("source_physical_layout", "") or row.get("physical_layout", "") or "")
        if str(row.get("source", "")) == "x" or source_physical == "packed_compact":
            return self._concat_zero_halo_layout(source_layout or selected_layout)
        if source_physical == "logical_halo_compact":
            return dict(source_layout or selected_layout)
        return dict(source_layout or selected_layout)

    def _concat_source_input_physical_layout(self, spec: dict) -> str:
        row = self._concat_join_input_layout_row(spec)
        if row:
            if bool(row.get("relayout", False)):
                return str(row.get("physical_layout", "") or "")
            return str(row.get("source_physical_layout", "") or row.get("physical_layout", "") or "")
        attrs = self._concat_layout_policy_module_attrs()
        return str(attrs.get("layout_policy_input_physical_layout", "") or "")

    def _concat_source_input_layout(self, spec: dict) -> dict:
        join_row = self._concat_join_input_layout_row(spec)
        if join_row:
            return self._concat_physical_input_layout_from_join_row(join_row)
        layout = dict(getattr(self, "layout_policy_input_layout", {}) or {})
        if layout:
            return layout
        attrs = self._concat_layout_policy_module_attrs()
        layout = dict(attrs.get("layout_policy_input_layout", {}) or {})
        if layout:
            return layout
        row = self._concat_conv_input_layout_row()
        if row:
            return dict(row.get("selected_layout", {}) or {})
        return {}

    def _concat_output_layout_attrs(self) -> dict:
        attrs = self._concat_layout_policy_module_attrs()
        module_layout = dict(getattr(self, "layout_policy_output_layout", {}) or {})
        output_layout = dict(module_layout or attrs.get("layout_policy_output_layout", {}) or {})
        output_gap = max(1, int(output_layout.get("gap", getattr(self, "output_gap", 1)) or 1))
        output_top_beta = self._concat_layout_top_beta(output_layout)
        output_row_offset = getattr(self, "layout_policy_output_row_offset", None)
        if output_row_offset is None:
            output_row_offset = attrs.get("layout_policy_output_row_offset", int(output_top_beta * output_gap))
        output_materialization = getattr(self, "layout_policy_output_materialization", None)
        if output_materialization is None:
            output_materialization = attrs.get("layout_policy_output_materialization", "")
        result = {
            "layout_policy_output_layout": dict(output_layout),
            "layout_policy_output_row_offset": int(output_row_offset or 0),
            "layout_policy_output_materialization": str(output_materialization or ""),
        }
        if "fhe_output_shape" in attrs:
            result["fhe_output_shape"] = torch.Size(attrs["fhe_output_shape"])
        return result

    def _concat_effective_fhe_output_shape(self, output_attrs: dict | None = None) -> torch.Size:
        attrs = dict(output_attrs or self._concat_output_layout_attrs())
        return torch.Size(attrs.get("fhe_output_shape", getattr(self, "fhe_output_shape")))

    def _concat_construct_bias(self, output_attrs: dict | None = None) -> torch.Tensor:
        attrs = dict(output_attrs or self._concat_output_layout_attrs())
        effective_fhe_output_shape = self._concat_effective_fhe_output_shape(attrs)
        bias_proxy = SimpleNamespace(
            output_shape=self.output_shape,
            fhe_output_shape=effective_fhe_output_shape,
            output_gap=int(self.output_gap),
            on_bias=self.on_bias,
            layout_policy_output_layout=dict(attrs.get("layout_policy_output_layout", {}) or {}),
        )
        return packing.construct_conv2d_bias(bias_proxy)

    def _concat_bootstrap_prescale_fusion_spec(self) -> dict[str, float] | None:
        fusion = getattr(self, "_bootstrap_prescale_fusion", None)
        if not fusion:
            return None
        return {
            "scale": float(dict(fusion).get("scale", 1.0)),
            "bias": float(dict(fusion).get("bias", 0.0)),
        }

    def _concat_effective_weight_and_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.on_weight.detach().clone()
        bias = self.on_bias.detach().clone()
        fusion = self._concat_bootstrap_prescale_fusion_spec()
        if fusion is not None:
            scale = float(fusion["scale"])
            weight = weight * float(scale)
            bias = bias * float(scale) + float(fusion["bias"])
        return weight, bias

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
        output_attrs = self._concat_output_layout_attrs()
        effective_fhe_output_shape = self._concat_effective_fhe_output_shape(output_attrs)
        output_layout_attrs = dict(output_attrs)
        output_layout_attrs.pop("fhe_output_shape", None)
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
            fhe_output_shape=effective_fhe_output_shape,
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
            **output_layout_attrs,
        )

    def _concat_output_ct_count(self) -> int:
        slots = int(self.scheme.params.get_slots())
        effective_fhe_output_shape = self._concat_effective_fhe_output_shape()
        return max(1, int(math.ceil(int(effective_fhe_output_shape.numel()) / float(slots))))

    def _concat_store_compile_bias(self, bias) -> None:
        if self.scheme.lt_evaluator.single_slot_layer_cache_enabled():
            self._concat_layer_cache_bias = bias.detach().clone()
            self.on_bias_ptxt = None
        else:
            self.on_bias_ptxt = self.scheme.encoder.encode(bias, int(self.level) - int(self.depth))

    def _concat_materialize_bias_for_eval(self) -> bool:
        if self.on_bias_ptxt is not None:
            return False
        bias = getattr(self, "_concat_layer_cache_bias", None)
        if bias is None:
            return False
        self.on_bias_ptxt = self.scheme.encoder.encode(bias, int(self.level) - int(self.depth))
        return True

    def _concat_evict_bias_after_eval(self, materialized: bool) -> None:
        if bool(materialized):
            self.on_bias_ptxt = None

    def _concat_release_unified_diagonal_caches(self) -> None:
        seen: set[int] = set()
        for groups_by_source in getattr(self, "_concat_unified_groups_by_input", []) or []:
            for group in dict(groups_by_source).values():
                for transform in getattr(group, "transforms", ()) or ():
                    cache = getattr(transform, "_concat_branch_diagonal_cache", None)
                    if cache is None:
                        continue
                    cache_id = id(cache)
                    if int(cache_id) in seen:
                        continue
                    seen.add(int(cache_id))
                    release = getattr(cache, "release", None)
                    if callable(release):
                        release()

    def _compile_concat_fusion_unified_transforms(self) -> bool:
        if not self._concat_fusion_unified_supported():
            return False
        if getattr(self, "_concat_unified_groups_by_input", None):
            return True

        from concurrent.futures import ThreadPoolExecutor

        from orion.experimental.cir.native_halo_conv2d import (
            NativeHaloConv2DSpec,
            _build_compact_source_conv_transform,
            _build_compact_source_concat_transforms_single_slot,
            _layout_bottom_beta,
            _layout_top_beta,
            _retune_transform_group_bsgs,
            native_halo_conv2d_plan,
        )
        from orion.nn.unified_transform import UnifiedTransformGroup

        slots = int(self.scheme.params.get_slots())
        level = int(self.level)
        output_attrs = self._concat_output_layout_attrs()
        effective_fhe_output_shape = self._concat_effective_fhe_output_shape(output_attrs)
        target_ct_count = int(self._concat_output_ct_count())
        output_layout = dict(output_attrs.get("layout_policy_output_layout", {}) or {})
        output_top_beta = int(_layout_top_beta(output_layout))
        output_bottom_beta = int(_layout_bottom_beta(output_layout))

        self._concat_unified_groups_by_input = []
        self._concat_unified_targets_by_input = []
        self._concat_unified_output_ct_count = int(target_ct_count)
        self._concat_fusion_fhe_output_shape = torch.Size(effective_fhe_output_shape)
        self._concat_transform_ids_by_input = []
        self._concat_transform_sources_by_input = []
        self._concat_diagonals_by_input = []
        self._concat_output_rotations = 0

        effective_weight, effective_bias = self._concat_effective_weight_and_bias()
        with _temporary_attrs(self, on_bias=effective_bias):
            bias = self._concat_construct_bias(output_attrs)
        self._concat_store_compile_bias(bias)
        evaluator = getattr(self.scheme, "lt_evaluator", None)
        single_slot_layer_cache = (
            callable(getattr(evaluator, "single_slot_layer_cache_enabled", None))
            and bool(evaluator.single_slot_layer_cache_enabled())
        )

        for input_index, concat_spec in enumerate(self._concat_fusion_specs()):
            channel_start = int(concat_spec["channel_start"])
            channel_end = int(concat_spec["channel_end"])
            branch_weight = effective_weight[:, channel_start:channel_end, :, :].detach().clone()
            input_layout = self._concat_source_input_layout(concat_spec)
            input_gap = max(1, int(dict(input_layout).get("gap", int(concat_spec["gap"])) or int(concat_spec["gap"])))
            input_top_beta = int(_layout_top_beta(input_layout))
            input_bottom_beta = int(_layout_bottom_beta(input_layout))
            source_fhe_shape = self._concat_source_fhe_shape(concat_spec, input_layout)
            source_ct_count = max(1, int(math.ceil(int(torch.Size(source_fhe_shape).numel()) / float(slots))))
            n, channels, height, width = (int(value) for value in concat_spec["shape"])
            del n
            output_shape = tuple(int(value) for value in self.output_shape)
            label = (
                f"{getattr(self, 'name', self.__class__.__name__)}_concat{int(input_index)}"
                f"_{int(channels)}x{int(height)}x{int(width)}"
            )
            label = "".join(ch if ch.isalnum() else "_" for ch in str(label)).strip("_") or "concat_conv"
            native_spec = NativeHaloConv2DSpec(
                family_label=str(label),
                c_in=int(channels),
                h_in=int(height),
                w_in=int(width),
                c_out=int(output_shape[1]),
                h_out=int(output_shape[2]),
                w_out=int(output_shape[3]),
                gap_in=int(input_gap),
                gap_out=int(self.output_gap),
                kernel=int(self.kernel_size[0]),
                stride=int(self.stride[0]),
                pad=int(self.padding[0]),
                dilation=int(self.dilation[0]),
                groups=1,
                slot_count=int(slots),
                input_top_beta=int(input_top_beta),
                input_bottom_beta=int(input_bottom_beta),
                output_top_beta=int(output_top_beta),
                output_bottom_beta=int(output_bottom_beta),
            )
            plan = native_halo_conv2d_plan(native_spec, require_native_target_fit=False)

            def build_source(source_block: int) -> tuple[int, list[tuple[int, object]]]:
                ordered: list[tuple[int, object]] = []
                for stripe in plan.stripes:
                    for target_group in range(int(plan.target_group_count_for_stripe(stripe))):
                        for target_block in range(int(target_ct_count)):
                            transform = _build_compact_source_conv_transform(
                                spec=native_spec,
                                plan=plan,
                                weight=branch_weight,
                                stripe=stripe,
                                source_block=int(source_block),
                                target_group=int(target_group),
                                level=int(level),
                                scheme=self.scheme,
                                source_layout=dict(input_layout),
                                group_n1=1,
                                compact_target_block=int(target_block),
                            )
                            if transform is not None:
                                ordered.append((int(target_block), transform))
                return int(source_block), ordered

            if bool(single_slot_layer_cache):
                proxy = self._concat_source_proxy(
                    concat_spec,
                    weight=branch_weight,
                    name=f"{getattr(self, 'name', self.__class__.__name__)}_concat_source_{int(input_index)}",
                )

                def build_diagonals_by_block(proxy=proxy, branch_weight=branch_weight):
                    diagonals, output_rotations = packing.direct_diagonalize_conv2d(
                        proxy,
                        branch_weight,
                        int(slots),
                        str(self.scheme.params.get_embedding_method()),
                        False,
                        allow_hybrid=False,
                    )
                    if int(output_rotations) != 0:
                        raise RuntimeError(
                            f"concat-fused Conv2d {getattr(self, 'name', '')} single-slot "
                            "branch materializer does not support output rotations"
                        )
                    return diagonals

                built_by_source = _build_compact_source_concat_transforms_single_slot(
                    spec=native_spec,
                    plan=plan,
                    weight=branch_weight,
                    level=int(level),
                    scheme=self.scheme,
                    source_layout=dict(input_layout),
                    source_ct_count=int(source_ct_count),
                    target_ct_count=int(target_ct_count),
                    group_n1=1,
                    build_diagonals_by_block=build_diagonals_by_block,
                    output_materialization=str(output_attrs.get("layout_policy_output_materialization", "") or ""),
                )
                built = [
                    (int(source_block), list(ordered))
                    for source_block, ordered in sorted(built_by_source.items())
                ]
            else:
                raw_workers = __import__("os").environ.get("ORION_CONCAT_FUSION_BUILD_WORKERS", "1")
                try:
                    requested_workers = int(raw_workers)
                except (TypeError, ValueError):
                    requested_workers = 1
                workers = max(1, min(int(source_ct_count), int(requested_workers)))
                if int(workers) <= 1:
                    built = [build_source(int(source_block)) for source_block in range(int(source_ct_count))]
                else:
                    with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="orion-concat-fusion") as pool:
                        built = list(pool.map(build_source, range(int(source_ct_count))))

            groups_by_source: dict[int, UnifiedTransformGroup] = {}
            targets_by_source: dict[int, tuple[int, ...]] = {}
            for source_block, ordered in sorted(built, key=lambda item: int(item[0])):
                if not ordered:
                    continue
                ordered.sort(key=lambda item: int(item[0]))
                transforms = [transform for _target_index, transform in ordered]
                _retune_transform_group_bsgs(transforms, slots=int(slots))
                group = UnifiedTransformGroup(transforms)
                group.compile_unified(self.scheme.backend)
                groups_by_source[int(source_block)] = group
                targets_by_source[int(source_block)] = tuple(int(target_index) for target_index, _transform in ordered)
            if not groups_by_source:
                raise RuntimeError(
                    f"concat-fused Conv2d {getattr(self, 'name', '')} produced no unified transforms "
                    f"for concat input {int(input_index)}"
                )
            self._concat_unified_groups_by_input.append(groups_by_source)
            self._concat_unified_targets_by_input.append(targets_by_source)

        self.diagonals = {}
        self.transform_ids = {}
        self.output_rotations = 0
        self._transform_backend = self.scheme.backend
        return True

    def _generate_concat_fusion_diagonals(self, last: bool) -> bool:
        if not self._concat_fusion_supported():
            return False
        if self._concat_fusion_unified_supported():
            self._concat_diagonals_by_input = []
            self._concat_transform_ids_by_input = []
            self._concat_unified_groups_by_input = []
            self._concat_unified_targets_by_input = []
            self._concat_fusion_fhe_output_shape = self._concat_effective_fhe_output_shape()
            self._concat_output_rotations = 0
            self.diagonals = {}
            self.output_rotations = 0
            return True
        if self.scheme.lt_evaluator.single_slot_layer_cache_enabled():
            raise RuntimeError(
                "ORION_SINGLE_SLOT_LAYER_CACHE requires unified concat fusion; "
                "non-unified concat fallback would compile resident raw diagonals"
            )
        self._concat_diagonals_by_input = []
        self._concat_transform_ids_by_input = []
        self._concat_output_rotations = 0
        for input_index, spec in enumerate(self._concat_fusion_specs()):
            start = int(spec["channel_start"])
            end = int(spec["channel_end"])
            effective_weight, _effective_bias = self._concat_effective_weight_and_bias()
            weight = effective_weight[:, start:end, :, :].detach().clone()
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
        if self._compile_concat_fusion_unified_transforms():
            return True
        diagonals_by_input = list(getattr(self, "_concat_diagonals_by_input", []) or [])
        if not diagonals_by_input:
            self._generate_concat_fusion_diagonals(last=False)
            diagonals_by_input = list(getattr(self, "_concat_diagonals_by_input", []) or [])
        output_attrs = self._concat_output_layout_attrs()
        self._concat_fusion_fhe_output_shape = self._concat_effective_fhe_output_shape(output_attrs)
        _effective_weight, effective_bias = self._concat_effective_weight_and_bias()
        with _temporary_attrs(self, on_bias=effective_bias):
            bias = self._concat_construct_bias(output_attrs)
        self._concat_store_compile_bias(bias)
        self._concat_transform_ids_by_input = []
        self._concat_transform_sources_by_input = []
        for input_index, diagonals in enumerate(diagonals_by_input):
            proxy = SimpleNamespace(
                name=f"{getattr(self, 'name', self.__class__.__name__)}_concat_source_{int(input_index)}",
                diagonals=diagonals,
                level=int(self.level),
                bsgs_ratio=float(self.bsgs_ratio),
                scheme=self.scheme,
                output_shape=self.output_shape,
                fhe_output_shape=self._concat_fusion_fhe_output_shape,
            )
            proxy.transform_ids = dict(self.scheme.lt_evaluator.generate_transforms(proxy))
            self._concat_transform_sources_by_input.append(proxy)
            self._concat_transform_ids_by_input.append(dict(proxy.transform_ids))
        self.transform_ids = {}
        self._transform_backend = self.scheme.backend
        return True

    def _concat_fusion_ready(self) -> bool:
        return bool(
            getattr(self, "_concat_transform_ids_by_input", None)
            or getattr(self, "_concat_transform_sources_by_input", None)
            or getattr(self, "_concat_unified_groups_by_input", None)
        )

    def _evaluate_concat_fusion(self, concat_tensor):
        if getattr(self, "_concat_unified_groups_by_input", None):
            return self._evaluate_concat_fusion_unified(concat_tensor)
        bias_materialized = self._concat_materialize_bias_for_eval()
        parts = tuple(concat_tensor.parts)
        transform_ids_by_input = list(getattr(self, "_concat_transform_ids_by_input", []) or [])
        transform_sources = list(getattr(self, "_concat_transform_sources_by_input", []) or [])
        if not transform_sources:
            transform_sources = [
                SimpleNamespace(
                    name=f"{getattr(self, 'name', self.__class__.__name__)}_concat_source_{int(input_index)}",
                    transform_ids=dict(transform_ids_by_input[int(input_index)]),
                    level=int(self.level),
                    output_shape=self.output_shape,
                    fhe_output_shape=getattr(self, "_concat_fusion_fhe_output_shape", self.fhe_output_shape),
                )
                for input_index in range(len(transform_ids_by_input))
            ]
        try:
            if len(parts) != len(transform_sources):
                raise RuntimeError(
                    f"concat-fused Conv2d {getattr(self, 'name', '')} expected "
                    f"{len(transform_sources)} inputs, got {len(parts)}"
                )
            out = None
            concat_accumulate_s = 0.0
            for input_index, source in enumerate(parts):
                proxy = transform_sources[int(input_index)]
                partial = self.scheme.lt_evaluator.evaluate_transforms(proxy, source)
                if out is None:
                    out = partial
                else:
                    accumulate_started = time.perf_counter()
                    if bool(getattr(out.scheme.backend, "align_addition_scales", False)):
                        scale = max(1, int(out.scale()))
                        out.set_scale(int(scale))
                        partial.set_scale(int(scale))
                    out = out + partial
                    concat_accumulate_s += float(time.perf_counter() - accumulate_started)
            slots = self.scheme.params.get_slots()
            output_rotation_started = time.perf_counter()
            for i in range(1, int(self.output_rotations) + 1):
                out += out.roll(slots // (2**i))
            output_rotation_s = float(time.perf_counter() - output_rotation_started)
            bias_s = 0.0
            if self.on_bias_ptxt is not None:
                bias_started = time.perf_counter()
                out += self.on_bias_ptxt
                bias_s = float(time.perf_counter() - bias_started)
            self._publish_linear_wrapper_timing(
                {},
                accumulate_s=float(concat_accumulate_s),
                bias_s=float(bias_s),
                output_rotation_s=float(output_rotation_s),
            )
            return out
        finally:
            self._concat_evict_bias_after_eval(bias_materialized)
            release_owned = getattr(concat_tensor, "release_owned_parts", None)
            if callable(release_owned):
                release_owned()

    def _evaluate_concat_fusion_unified(self, concat_tensor):
        from orion.backend.python.tensors import CipherTensor
        from orion.experimental.cir.r34_orion_same_shape import (
            _align_ciphertexts_for_add,
            _rescale_cipher_tensor,
            _unified_output_fusion_enabled,
        )
        from orion.nn.unified_transform import UnifiedTransformGroup

        bias_materialized = self._concat_materialize_bias_for_eval()
        parts = tuple(concat_tensor.parts)
        groups_by_input = list(getattr(self, "_concat_unified_groups_by_input", []) or [])
        targets_by_input = list(getattr(self, "_concat_unified_targets_by_input", []) or [])
        if len(parts) != len(groups_by_input):
            self._concat_release_unified_diagonal_caches()
            self._concat_evict_bias_after_eval(bias_materialized)
            raise RuntimeError(
                f"concat-fused Conv2d {getattr(self, 'name', '')} expected "
                f"{len(groups_by_input)} inputs, got {len(parts)}"
            )
        slots = int(self.scheme.params.get_slots())
        target_count = int(getattr(self, "_concat_unified_output_ct_count", self._concat_output_ct_count()))
        output_blocks: list[object | None] = [None for _ in range(int(target_count))]
        fuse_output_rescale = bool(_unified_output_fusion_enabled())
        wrapper_accumulate_s = 0.0
        wrapper_rescale_s = 0.0
        wrapper_bias_s = 0.0
        wrapper_output_rotation_s = 0.0

        def add_partial(target_index: int, partial):
            nonlocal wrapper_accumulate_s
            current = output_blocks[int(target_index)]
            if current is None:
                output_blocks[int(target_index)] = partial
                return
            accumulate_started = time.perf_counter()
            lhs, rhs = _align_ciphertexts_for_add(current, partial)
            output_blocks[int(target_index)] = lhs + rhs
            wrapper_accumulate_s += float(time.perf_counter() - accumulate_started)

        try:
            for input_index, source in enumerate(parts):
                groups_by_source = dict(groups_by_input[int(input_index)])
                targets_by_source = dict(targets_by_input[int(input_index)])
                sorted_items = [
                    (int(source_index), group)
                    for source_index, group in sorted(groups_by_source.items())
                    if int(source_index) < len(getattr(source, "ids", ()))
                ]
                target_sum_ids = None
                if bool(fuse_output_rescale) and sorted_items:
                    target_sum_ids = UnifiedTransformGroup.evaluate_sources_with_target_sum(
                        [group for _source_index, group in sorted_items],
                        [int(source.ids[int(source_index)]) for source_index, _group in sorted_items],
                        [targets_by_source[int(source_index)] for source_index, _group in sorted_items],
                        int(target_count),
                        self.scheme.backend,
                    )
                if target_sum_ids is not None:
                    for target_index, output_id in enumerate(target_sum_ids):
                        partial = CipherTensor(
                            self.scheme,
                            [int(output_id)],
                            torch.Size([1, int(slots)]),
                            torch.Size([1, int(slots)]),
                        )
                        add_partial(int(target_index), partial)
                    continue
                for source_index, group in sorted_items:
                    output_ids = group.evaluate_unified(int(source.ids[int(source_index)]), self.scheme.backend)
                    for target_index, output_id in zip(targets_by_source[int(source_index)], output_ids):
                        partial = CipherTensor(
                            self.scheme,
                            [int(output_id)],
                            torch.Size([1, int(slots)]),
                            torch.Size([1, int(slots)]),
                        )
                        if not bool(fuse_output_rescale):
                            rescale_started = time.perf_counter()
                            partial = _rescale_cipher_tensor(partial)
                            wrapper_rescale_s += float(time.perf_counter() - rescale_started)
                        add_partial(int(target_index), partial)

            if bool(fuse_output_rescale):
                for target_index, block in enumerate(output_blocks):
                    if block is not None:
                        rescale_started = time.perf_counter()
                        output_blocks[int(target_index)] = _rescale_cipher_tensor(block)
                        wrapper_rescale_s += float(time.perf_counter() - rescale_started)
            output_ids: list[int] = []
            for target_index, block in enumerate(output_blocks):
                if block is None:
                    raise RuntimeError(
                        f"concat-fused Conv2d {getattr(self, 'name', '')} missing output block {int(target_index)}"
                    )
                output_ids.append(int(block.ids[0]))
                block.ids = []
            out = CipherTensor(
                self.scheme,
                output_ids,
                self.output_shape,
                getattr(self, "_concat_fusion_fhe_output_shape", self.fhe_output_shape),
            )
            output_rotation_started = time.perf_counter()
            for i in range(1, int(self.output_rotations) + 1):
                out += out.roll(int(slots) // (2**i))
            wrapper_output_rotation_s += float(time.perf_counter() - output_rotation_started)
            if self.on_bias_ptxt is not None:
                bias_started = time.perf_counter()
                out += self.on_bias_ptxt
                wrapper_bias_s += float(time.perf_counter() - bias_started)
            self._publish_linear_wrapper_timing(
                {},
                accumulate_s=float(wrapper_accumulate_s),
                rescale_s=float(wrapper_rescale_s),
                bias_s=float(wrapper_bias_s),
                output_rotation_s=float(wrapper_output_rotation_s),
            )
            return out
        finally:
            release_owned = getattr(concat_tensor, "release_owned_parts", None)
            if callable(release_owned):
                release_owned()
            self._concat_release_unified_diagonal_caches()
            self._concat_evict_bias_after_eval(bias_materialized)

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
        if self.scheme.lt_evaluator.single_slot_layer_cache_enabled():
            diag_indices, output_rotations = packing.pack_conv2d_diagonal_indices(self, bool(last))
            self.output_rotations = int(output_rotations)
            self._install_single_slot_payload_recipe(
                diag_indices_by_block=diag_indices,
                build_diagonals=lambda self=self, last=bool(last): packing.pack_conv2d(self, bool(last))[0],
                build_block_diagonals=(
                    lambda blocks, self=self, last=bool(last): packing.pack_conv2d_blocks(self, bool(last), blocks)
                ),
                build_block_payloads=(
                    lambda blocks, self=self, last=bool(last): packing.build_conv2d_block_payloads(self, bool(last), blocks)
                ),
            )
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
        if self.scheme.lt_evaluator.dense_layer_cache_enabled_for(self):
            self._dense_layer_cache_bias = bias.detach().clone()
            self.on_bias_ptxt = None
        else:
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
        if self.scheme.lt_evaluator.single_slot_layer_cache_enabled():
            diag_indices, output_rotations = packing.pack_conv_transpose2d_diagonal_indices(self, bool(last))
            self.output_rotations = int(output_rotations)
            self._install_single_slot_payload_recipe(
                diag_indices_by_block=diag_indices,
                build_diagonals=lambda self=self, last=bool(last): packing.pack_conv_transpose2d(self, bool(last))[0],
                build_block_diagonals=(
                    lambda blocks, self=self, last=bool(last): packing.pack_conv_transpose2d_blocks(self, bool(last), blocks)
                ),
                build_block_payloads=(
                    lambda blocks, self=self, last=bool(last): packing.build_conv_transpose2d_block_payloads(self, bool(last), blocks)
                ),
            )
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
        if self.scheme.lt_evaluator.dense_layer_cache_enabled_for(self):
            self._dense_layer_cache_bias = bias.detach().clone()
            self.on_bias_ptxt = None
        else:
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
