import math
import json
import os
import time
import torch

from .module import Module, timer

class Add(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def forward(self, x, y):
        return x + y
    

class Mult(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def forward(self, x, y):
        return x * y
    

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
            "cipher": self._debug_cipher_stats(x),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    @timer
    def forward(self, x):
        if not self.he_mode:
            return x
        
        # Shift and scale into range [-1, 1]. Important caveat -- here we first
        # shift, then scale. This let's us zero out unused slots and enables
        # sparse bootstrapping (i.e., where slots < N/2).
        if self.constant != 0:
            x += self.constant
        x *= self._get_prescale_ptxt(x.level())
 
        slots = int(min(x.slots(), self.bootstrap_slots))
        self._write_bootstrap_debug(phase="before_bootstrap", x=x, slots=slots)
        if os.environ.get("ORION_ABORT_BEFORE_BOOTSTRAP", "0") != "0":
            raise RuntimeError(
                f"aborting before bootstrap for debug: "
                f"{getattr(self, 'bootstrap_debug_name', '')}"
            )
        x = x.bootstrap(slots=slots)
        self._write_bootstrap_debug(phase="after_bootstrap", x=x, slots=slots)

        # Scale and shift back to the original range
        if self.postscale != 1:
            x *= self.postscale 
        if self.constant != 0:
            x -= self.constant

        return x
