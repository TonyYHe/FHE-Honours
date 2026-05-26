import math
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from .module import Module, timer


def _align_ciphertexts_for_add(left, right):
    if bool(getattr(left.scheme.backend, "align_addition_scales", False)):
        scale = max(1, int(left.scale()))
        left.set_scale(int(scale))
        right.set_scale(int(scale))
    return left, right


class Add(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def compile(self):
        runtime = getattr(self, "layout_policy_add_runtime", None)
        if runtime is not None and callable(getattr(runtime, "compile", None)):
            runtime.assigned_level = int(self.level) if self.level is not None else None
            runtime.assigned_depth = int(self.depth or 0)
            runtime.compile(self.scheme)

    def forward(self, x, y):
        runtime = getattr(self, "layout_policy_add_runtime", None)
        if self.he_mode and runtime is not None:
            return runtime(x, y)
        return x + y


class ConcatCipherTensor:
    """Lazy HE representation of a channel concat.

    Keeping the source ciphertexts separate lets a following Conv2d split its
    weights by channel range and avoid materializing the concat boundary.
    """

    def __init__(self, module, parts, *, owned_parts=()):
        self.module = module
        self.parts = tuple(parts)
        self.owned_parts = tuple(owned_parts)
        self.scheme = self.parts[0].scheme if self.parts else None
        self.shape = module.output_shape
        self.on_shape = module.fhe_output_shape
        self._materialized = None

    def materialize(self):
        if self._materialized is None:
            self._materialized = self.module.materialize(self.parts)
        return self._materialized

    @property
    def ids(self):
        return self.materialize().ids

    @ids.setter
    def ids(self, value):
        self.materialize().ids = value

    def release(self):
        if self._materialized is not None:
            self._materialized.release()
            self._materialized = None
        self.release_owned_parts()

    def release_owned_parts(self):
        for part in self.owned_parts:
            release = getattr(part, "release", None)
            if callable(release):
                release()
        self.owned_parts = ()

    def __len__(self):
        return len(self.materialize())

    def __getattr__(self, name):
        return getattr(self.materialize(), name)

    def __add__(self, other):
        return self.materialize() + other

    def __radd__(self, other):
        return other + self.materialize()

    def __iadd__(self, other):
        materialized = self.materialize()
        materialized += other
        return materialized

    def __sub__(self, other):
        return self.materialize() - other

    def __rsub__(self, other):
        return other - self.materialize()

    def __mul__(self, other):
        return self.materialize() * other

    def __rmul__(self, other):
        return other * self.materialize()


class Concat(Module):
    def __init__(self, dim: int = 1, bsgs_ratio: int = 2):
        super().__init__()
        self.dim = int(dim)
        self.bsgs_ratio = float(bsgs_ratio)
        self.set_depth(0)
        self.concat_input_shapes = ()
        self.concat_input_fhe_shapes = ()
        self.concat_input_gaps = ()
        self.transform_ids_by_input = []
        self._compiled_backend = None

    def extra_repr(self):
        return super().extra_repr() + f", dim={self.dim}"

    def configure_from_stats(
        self,
        *,
        input_shapes,
        input_fhe_shapes,
        input_gaps,
        output_shape,
        fhe_output_shape,
        output_gap,
    ) -> None:
        self.concat_input_shapes = tuple(torch.Size(shape) for shape in input_shapes)
        self.concat_input_fhe_shapes = tuple(torch.Size(shape) for shape in input_fhe_shapes)
        self.concat_input_gaps = tuple(int(gap) for gap in input_gaps)
        self.output_shape = torch.Size(output_shape)
        self.fhe_output_shape = torch.Size(fhe_output_shape)
        self.output_gap = int(output_gap)

    def compile(self):
        # Generic materialization is compiled lazily only if a non-fused
        # consumer actually asks for it.
        return

    def _flat_index(self, channel, rows, cols, *, gap, height, width, row_offset=0):
        gap = int(gap)
        phase = int(channel) % int(gap * gap)
        packed_channel = int(channel) // int(gap * gap)
        packed_rows = rows.astype(np.int64) * int(gap) + int(phase // gap) + int(row_offset)
        packed_cols = cols.astype(np.int64) * int(gap) + int(phase % gap)
        return ((int(packed_channel) * int(height) + packed_rows) * int(width) + packed_cols).astype(np.int64)

    def _add_diagonal_entries(self, diagonals, source_indices, output_indices, *, slots):
        source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        output_indices = np.asarray(output_indices, dtype=np.int64).reshape(-1)
        if source_indices.size == 0:
            return
        source_blocks = source_indices // int(slots)
        output_blocks = output_indices // int(slots)
        source_local = source_indices % int(slots)
        output_local = output_indices % int(slots)
        diag_indices = (source_local - output_local) % int(slots)
        order = np.lexsort((diag_indices, source_blocks, output_blocks))
        source_blocks = source_blocks[order]
        output_blocks = output_blocks[order]
        output_local = output_local[order]
        diag_indices = diag_indices[order]

        start = 0
        while start < int(diag_indices.size):
            end = start + 1
            while (
                end < int(diag_indices.size)
                and int(output_blocks[end]) == int(output_blocks[start])
                and int(source_blocks[end]) == int(source_blocks[start])
                and int(diag_indices[end]) == int(diag_indices[start])
            ):
                end += 1
            block = diagonals.setdefault((int(output_blocks[start]), int(source_blocks[start])), {})
            diag = block.get(int(diag_indices[start]))
            if diag is None:
                diag = np.zeros((int(slots),), dtype=np.float32)
                block[int(diag_indices[start])] = diag
            diag[output_local[start:end].astype(np.int64)] = 1.0
            start = end

    def _diagonals_for_input(self, input_index: int, *, slots: int):
        input_shape = self.concat_input_shapes[int(input_index)]
        input_fhe_shape = self.concat_input_fhe_shapes[int(input_index)]
        output_shape = self.output_shape
        output_fhe_shape = self.fhe_output_shape
        input_gap = int(self.concat_input_gaps[int(input_index)])
        output_gap = int(self.output_gap)
        if int(input_gap) != int(output_gap):
            raise ValueError("Concat materialization requires all inputs to share the same FHE gap")
        channel_offset = sum(int(shape[1]) for shape in self.concat_input_shapes[: int(input_index)])
        n, channels, height, width = (int(value) for value in input_shape)
        input_block = int(input_fhe_shape[1] * input_fhe_shape[2] * input_fhe_shape[3])
        output_block = int(output_fhe_shape[1] * output_fhe_shape[2] * output_fhe_shape[3])
        row_grid, col_grid = np.meshgrid(
            np.arange(int(height), dtype=np.int64),
            np.arange(int(width), dtype=np.int64),
            indexing="ij",
        )
        rows = row_grid.reshape(-1)
        cols = col_grid.reshape(-1)
        diagonals = {}
        for batch in range(int(n)):
            for channel in range(int(channels)):
                source = self._flat_index(
                    int(channel),
                    rows,
                    cols,
                    gap=int(input_gap),
                    height=int(input_fhe_shape[2]),
                    width=int(input_fhe_shape[3]),
                ) + int(batch) * int(input_block)
                output = self._flat_index(
                    int(channel_offset + channel),
                    rows,
                    cols,
                    gap=int(output_gap),
                    height=int(output_fhe_shape[2]),
                    width=int(output_fhe_shape[3]),
                ) + int(batch) * int(output_block)
                self._add_diagonal_entries(diagonals, source, output, slots=int(slots))
        return diagonals

    def _ensure_materialize_transforms(self, scheme):
        if self.transform_ids_by_input and self._compiled_backend is getattr(scheme, "backend", None):
            return
        self.cleanup(getattr(scheme, "backend", None))
        if not self.concat_input_shapes:
            raise RuntimeError("Concat shapes have not been initialized by StatsTracker")
        level = int(self.level) if self.level is not None else int(len(scheme.params.get_logq()) - 1)
        slots = int(scheme.params.get_slots())
        self.transform_ids_by_input = []
        for input_index in range(len(self.concat_input_shapes)):
            diagonals = self._diagonals_for_input(int(input_index), slots=int(slots))
            proxy = SimpleNamespace(
                name=f"{getattr(self, 'name', 'concat')}_materialize_{int(input_index)}",
                diagonals=diagonals,
                level=int(level),
                bsgs_ratio=float(self.bsgs_ratio),
                scheme=scheme,
                output_shape=self.output_shape,
                fhe_output_shape=self.fhe_output_shape,
            )
            self.transform_ids_by_input.append(dict(scheme.lt_evaluator.generate_transforms(proxy)))
        self._compiled_backend = getattr(scheme, "backend", None)

    def materialize(self, parts):
        parts = tuple(parts)
        if not parts:
            raise ValueError("Concat requires at least one input")
        scheme = parts[0].scheme
        self._ensure_materialize_transforms(scheme)
        out = None
        for input_index, source in enumerate(parts):
            proxy = SimpleNamespace(
                name=f"{getattr(self, 'name', 'concat')}_materialize_{int(input_index)}",
                transform_ids=dict(self.transform_ids_by_input[int(input_index)]),
                level=int(self.level) if self.level is not None else int(len(scheme.params.get_logq()) - 1),
                output_shape=self.output_shape,
                fhe_output_shape=self.fhe_output_shape,
            )
            partial = scheme.lt_evaluator.evaluate_transforms(proxy, source)
            if out is None:
                out = partial
            else:
                lhs, rhs = _align_ciphertexts_for_add(out, partial)
                out = lhs + rhs
        return out

    def cleanup(self, backend=None):
        backend = backend if backend is not None else self._compiled_backend
        delete = getattr(backend, "DeleteLinearTransform", None)
        if callable(delete):
            for transform_ids in self.transform_ids_by_input:
                for value in dict(transform_ids).values():
                    try:
                        delete(int(value))
                    except Exception:
                        pass
        self.transform_ids_by_input = []
        self._compiled_backend = None

    def forward(self, *xs):
        if len(xs) == 1 and isinstance(xs[0], (list, tuple)):
            xs = tuple(xs[0])
        if not xs:
            raise ValueError("Concat requires at least one input")
        if self.he_mode:
            runtime = getattr(self, "layout_policy_concat_runtime", None)
            owned_parts = ()
            if runtime is not None:
                xs, owned_parts = runtime(*xs)
            return ConcatCipherTensor(self, xs, owned_parts=owned_parts)
        return torch.cat(tuple(xs), dim=int(self.dim))


class Identity(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def forward(self, x):
        return x
    

class Mult(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def forward(self, x, y):
        out = x * y
        if self.he_mode:
            bias = float(getattr(self, "_bootstrap_output_bias_fusion", 0.0) or 0.0)
            if bias != 0.0:
                out += bias
        return out
    

class Bootstrap(Module):
    def __init__(self, input_min, input_max, input_level):
        super().__init__()
        self.input_min = input_min 
        self.input_max = input_max 
        self.input_level = input_level
        self.prescale = 1
        self.postscale = 1
        self.constant = 0
        self.prescale_ptxt = None
        self._prescale_vec = None
        self._prescale_ptxt_cache = {}
        self.preprocess_fused = False
        self.preprocess_fusion_kind = ""
        self._bootstrap_runtime_profile = []
        self._bootstrap_runtime_call_index = 0

    def extra_repr(self):
        l_eff = len(self.scheme.params.get_logq()) - 1
        return f"l_eff={l_eff}"

    def _resolve_margin(self):
        override = os.environ.get("ORION_BOOTSTRAP_MARGIN_OVERRIDE", "").strip()
        if not override:
            return float(self.margin)
        try:
            return float(override)
        except ValueError as exc:
            raise ValueError(
                f"invalid ORION_BOOTSTRAP_MARGIN_OVERRIDE={override!r}; expected a float"
            ) from exc

    def fit(self):
        center = (self.input_min + self.input_max) / 2 
        half_range = (self.input_max - self.input_min) / 2
        self.bootstrap_margin = self._resolve_margin()
        self.low = (center - (self.bootstrap_margin * half_range)).item()
        self.high = (center + (self.bootstrap_margin * half_range)).item()

        # We'll want to scale from [A, B] into [-1, 1] using a value of the
        # form 1 / integer, so that way our multiplication back to the range
        # [A, B] (by integer) after bootstrapping doesn't consume a level.
        if self.high - self.low > 2:
            self.postscale = math.ceil((self.high - self.low) / 2)
            self.prescale = 1 / self.postscale

        self.constant = -(self.low + self.high) / 2 

    def compile(self):
        # Precompute the sparse prescale vector once. We then lazily encode it
        # at the ciphertext's actual runtime level to keep the bootstrap
        # prescale contract aligned even if the planner's guessed input level
        # differs from the level that reaches the hook at execution time.
        elements = self.fhe_input_shape.numel()
        curr_slots = 2 ** math.ceil(math.log2(elements))
        self.bootstrap_slots = curr_slots

        prescale_vec = torch.zeros(curr_slots)
        prescale_vec[:elements] = self.prescale
        self._prescale_vec = prescale_vec
        self._prescale_ptxt_cache = {}
        self.prescale_ptxt = self._get_prescale_ptxt(self.input_level)

    def _get_prescale_ptxt(self, level):
        level = int(level)
        if level not in self._prescale_ptxt_cache:
            if self._prescale_vec is None:
                raise RuntimeError("Bootstrap prescale vector has not been compiled")
            ql = self.scheme.encoder.get_moduli_chain()[level]
            self._prescale_ptxt_cache[level] = self.scheme.encoder.encode(
                self._prescale_vec, level=level, scale=ql
            )
        return self._prescale_ptxt_cache[level]

    def _debug_cipher_stats(self, x):
        ids = [int(value) for value in getattr(x, "ids", [])]
        return {
            "id_count": int(len(ids)),
            "ids": ids[:8],
            "level": int(x.level()) if hasattr(x, "level") else None,
            "scale": int(x.scale()) if hasattr(x, "scale") else None,
            "scale_log2": float(x.scale_log2()) if hasattr(x, "scale_log2") else None,
            "slots": int(x.slots()) if hasattr(x, "slots") else None,
        }

    def _profile_value_summary(self, values):
        values = list(values)
        if not values:
            return {"count": 0, "distinct": [], "min": None, "max": None, "sample": []}
        numeric = [value for value in values if value is not None]
        distinct = sorted({value for value in numeric})
        return {
            "count": int(len(values)),
            "distinct": distinct[:16],
            "distinct_count": int(len(distinct)),
            "min": min(numeric) if numeric else None,
            "max": max(numeric) if numeric else None,
            "sample": values[:16],
        }

    def _profile_cipher_batch_stats(self, x):
        ids = [int(value) for value in getattr(x, "ids", [])]
        backend = getattr(x, "backend", None)
        if backend is None:
            scheme = getattr(x, "scheme", None)
            backend = getattr(scheme, "backend", None)

        def collect(method_name, convert):
            method = getattr(backend, method_name, None) if backend is not None else None
            values = []
            errors = 0
            if not callable(method):
                return {"available": False, "values": [], "errors": 0}
            for ciphertext_id in ids:
                try:
                    values.append(convert(method(int(ciphertext_id))))
                except Exception:
                    values.append(None)
                    errors += 1
            return {"available": True, "values": values, "errors": int(errors)}

        levels = collect("GetCiphertextLevel", int)
        scales = collect("GetCiphertextScale", int)
        scale_log2 = collect("GetCiphertextScaleLog2", float)
        slots = collect("GetCiphertextSlots", int)
        degrees = collect("GetCiphertextDegree", int)
        return {
            "id_count": int(len(ids)),
            "ids_sample": ids[:16],
            "ids_tail": ids[-16:],
            "level": self._profile_value_summary(levels["values"]),
            "scale": self._profile_value_summary(scales["values"]),
            "scale_log2": self._profile_value_summary(scale_log2["values"]),
            "slots": self._profile_value_summary(slots["values"]),
            "degree": self._profile_value_summary(degrees["values"]),
            "query_errors": {
                "level": int(levels["errors"]),
                "scale": int(scales["errors"]),
                "scale_log2": int(scale_log2["errors"]),
                "slots": int(slots["errors"]),
                "degree": int(degrees["errors"]),
            },
        }

    def _new_bootstrap_runtime_record(self):
        self._bootstrap_runtime_call_index += 1
        return {
            "call_index": int(self._bootstrap_runtime_call_index),
            "name": str(getattr(self, "bootstrap_debug_name", "")),
            "configured_input_level": int(self.input_level),
            "configured_bootstrap_slots": int(getattr(self, "bootstrap_slots", 0) or 0),
            "prescale": float(self.prescale),
            "postscale": float(self.postscale),
            "constant": float(self.constant),
            "preprocess_fused": bool(getattr(self, "preprocess_fused", False)),
            "preprocess_fusion_kind": str(getattr(self, "preprocess_fusion_kind", "")),
            "timing_s": {},
        }

    def _write_bootstrap_debug(self, *, phase: str, x, slots: int | None = None) -> None:
        path = os.environ.get("ORION_BOOTSTRAP_DEBUG_PATH", "")
        if not path:
            return
        row = {
            "time": float(time.time()),
            "phase": str(phase),
            "name": str(getattr(self, "bootstrap_debug_name", "")),
            "input_level": int(self.input_level),
            "bootstrap_slots": int(self.bootstrap_slots),
            "runtime_slots": None if slots is None else int(slots),
            "margin": float(getattr(self, "bootstrap_margin", self.margin)),
            "prescale": float(self.prescale),
            "postscale": float(self.postscale),
            "constant": float(self.constant),
            "preprocess_fused": bool(getattr(self, "preprocess_fused", False)),
            "preprocess_fusion_kind": str(getattr(self, "preprocess_fusion_kind", "")),
            "cipher": self._debug_cipher_stats(x),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    @timer
    def forward(self, x):
        if not self.he_mode:
            return x

        total_start = time.perf_counter()
        profile_record = self._new_bootstrap_runtime_record()
        profile_record["input_before_preprocess"] = self._profile_cipher_batch_stats(x)
        
        # Shift and scale into range [-1, 1]. Important caveat -- here we first
        # shift, then scale. This lets us zero out unused slots and enables
        # sparse bootstrapping (i.e., where slots < N/2). Full-slot producers
        # that can absorb this affine set preprocess_fused and arrive here
        # already scaled, saving the plaintext-multiply level.
        preprocess_start = time.perf_counter()
        add_shift_s = 0.0
        prescale_mul_s = 0.0
        if not bool(getattr(self, "preprocess_fused", False)):
            if self.constant != 0:
                step_start = time.perf_counter()
                x += self.constant
                add_shift_s = float(time.perf_counter() - step_start)
            step_start = time.perf_counter()
            x *= self._get_prescale_ptxt(x.level())
            prescale_mul_s = float(time.perf_counter() - step_start)
        profile_record["timing_s"]["preprocess_total"] = float(time.perf_counter() - preprocess_start)
        profile_record["timing_s"]["preprocess_add_shift"] = float(add_shift_s)
        profile_record["timing_s"]["preprocess_prescale_mul"] = float(prescale_mul_s)
 
        slots = int(min(x.slots(), self.bootstrap_slots))
        profile_record["runtime_slots"] = int(slots)
        profile_record["input_to_backend"] = self._profile_cipher_batch_stats(x)
        self._write_bootstrap_debug(phase="before_bootstrap", x=x, slots=slots)
        if os.environ.get("ORION_ABORT_BEFORE_BOOTSTRAP", "0") != "0":
            raise RuntimeError(
                f"aborting before bootstrap for debug: "
                f"{getattr(self, 'bootstrap_debug_name', '')}"
            )
        backend_start = time.perf_counter()
        x = x.bootstrap(slots=slots)
        profile_record["timing_s"]["backend_bootstrap_call"] = float(time.perf_counter() - backend_start)
        profile_record["output_from_backend"] = self._profile_cipher_batch_stats(x)
        self._write_bootstrap_debug(phase="after_bootstrap", x=x, slots=slots)

        # Scale and shift back to the original range
        postprocess_start = time.perf_counter()
        postscale_mul_s = 0.0
        post_shift_s = 0.0
        if self.postscale != 1:
            step_start = time.perf_counter()
            x *= self.postscale 
            postscale_mul_s = float(time.perf_counter() - step_start)
        if self.constant != 0:
            step_start = time.perf_counter()
            x -= self.constant
            post_shift_s = float(time.perf_counter() - step_start)
        profile_record["timing_s"]["postprocess_total"] = float(time.perf_counter() - postprocess_start)
        profile_record["timing_s"]["postprocess_postscale_mul"] = float(postscale_mul_s)
        profile_record["timing_s"]["postprocess_sub_shift"] = float(post_shift_s)
        profile_record["timing_s"]["forward_total_inner"] = float(time.perf_counter() - total_start)
        profile_record["output_after_postprocess"] = self._profile_cipher_batch_stats(x)
        self._bootstrap_runtime_profile.append(profile_record)

        return x
