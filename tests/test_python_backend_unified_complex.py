from __future__ import annotations

from types import SimpleNamespace

import torch

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.nn.unified_transform import UnifiedTransformGroup


def test_python_backend_unified_complex_roundtrip() -> None:
    config = {
        "ckks_params": {
            "LogN": 8,
            "LogQ": [45, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "python",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    try:
        slots = int(scheme.params.get_slots())
        level = len(scheme.params.get_logq()) - 1

        diag_identity = [complex(1.0, 0.0)] * slots
        diag_twist = [complex(0.5, -0.25)] * slots
        transform_a = SimpleNamespace(
            diagonals={(0, 0): {0: diag_identity}},
            level=level,
            scheme=scheme,
            fhe_output_shape=torch.Size([1, slots]),
            output_shape=torch.Size([1, slots]),
        )
        transform_b = SimpleNamespace(
            diagonals={(0, 0): {0: diag_twist}},
            level=level,
            scheme=scheme,
            fhe_output_shape=torch.Size([1, slots]),
            output_shape=torch.Size([1, slots]),
        )

        x_real = torch.zeros(slots, dtype=torch.float32)
        x_imag = torch.zeros(slots, dtype=torch.float32)
        x_real[:8] = torch.linspace(0.1, 0.8, 8)
        x_imag[:8] = torch.linspace(-0.2, 0.5, 8)
        ct = scheme.encrypt(scheme.encode(x_real, level)) + scheme.encrypt(scheme.encode(x_imag, level)).mul_imaginary_unit(+1, in_place=False)

        group = UnifiedTransformGroup([transform_a, transform_b])
        group.compile_unified(scheme.backend)
        output_ids = group.evaluate_unified(int(ct.ids[0]), scheme.backend)

        out_a = CipherTensor(scheme, [int(output_ids[0])], torch.Size([1, slots]), torch.Size([1, slots]))
        out_b = CipherTensor(scheme, [int(output_ids[1])], torch.Size([1, slots]), torch.Size([1, slots]))
        out_a_pt = out_a.decrypt()
        out_b_pt = out_b.decrypt()
        raw_a = scheme.backend.DecodeComplex(out_a_pt.ids[0])
        raw_b = scheme.backend.DecodeComplex(out_b_pt.ids[0])
        decoded_a = torch.tensor([complex(raw_a[2 * i], raw_a[2 * i + 1]) for i in range(slots)], dtype=torch.complex64)
        decoded_b = torch.tensor([complex(raw_b[2 * i], raw_b[2 * i + 1]) for i in range(slots)], dtype=torch.complex64)

        x = x_real.to(dtype=torch.complex64) + 1j * x_imag.to(dtype=torch.complex64)
        assert float((decoded_a[:8] - x[:8]).abs().max()) <= 1.0e-5
        assert float((decoded_b[:8] - x[:8] * complex(0.5, -0.25)).abs().max()) <= 1.0e-5
    finally:
        scheme.delete_scheme()
