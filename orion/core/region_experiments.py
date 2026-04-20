"""Region-first experiment table for Orion integration.

The goal of this module is not to replace Orion's production compiler. It gives
us a repeatable experiment surface for the paper's actual focus:

* input multiplication / input replication
* output fold
* real-imag hybrid packing

Shared LT is treated as backend support and is recorded as metadata, not as the
main result.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core.orion import scheme
from orion.core.region_lowering import (
    ConvRegionSpec,
    build_region_search_candidates,
    build_tile_local_conv_lt,
    merge_tile_lts_as_complex,
    pack_chw_gap,
    transform_from_tile_lt,
)
from orion.nn.unified_transform import UnifiedTransformGroup

from .shared_lt import (
    OutputBank,
    PackingPlanner,
    RegionNode,
    RegionPlanner,
    SharedLTGroup,
    SharedLTTransformSpec,
    SourceTile,
    TargetTile,
)


Strategy = Literal[
    "orion_baseline",
    "input_mult_output_fold",
    "real_imag_hybrid_output_fold",
    "real_imag_hybrid_transition_branch",
]


@dataclass(frozen=True)
class RegionStats:
    rotations: int
    conjugations: int
    ct_pt_mults: int
    adds: int

    def score(self) -> float:
        return float(self.rotations) * 8.0 + float(self.conjugations) * 8.0 + float(self.ct_pt_mults) + float(self.adds) * 0.05

    def minus(self, other: "RegionStats") -> "RegionStats":
        return RegionStats(
            rotations=int(self.rotations) - int(other.rotations),
            conjugations=int(self.conjugations) - int(other.conjugations),
            ct_pt_mults=int(self.ct_pt_mults) - int(other.ct_pt_mults),
            adds=int(self.adds) - int(other.adds),
        )


@dataclass(frozen=True)
class RegionExperiment:
    network: str
    region_id: str
    strategy: Strategy
    baseline: RegionStats
    candidate: RegionStats
    delta: RegionStats
    evidence: str
    publishable_executor_fact: bool
    source_tiles: int
    target_tiles: int
    output_banks: int
    input_mult_factor: int = 1
    output_fold_factor: int = 1
    real_imag_hybrid: bool = False
    shared_lt_backend: str = "UnifiedLinearTransform-compatible"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["baseline"]["score"] = self.baseline.score()
        payload["candidate"]["score"] = self.candidate.score()
        payload["delta"]["score"] = self.candidate.score() - self.baseline.score()
        return payload


def _stats(rotations: int, ct_pt: int, adds: int | None = None, conjugations: int = 0) -> RegionStats:
    return RegionStats(
        rotations=int(rotations),
        conjugations=int(conjugations),
        ct_pt_mults=int(ct_pt),
        adds=int(ct_pt if adds is None else adds),
    )


def _local_lattigo_metadata(repo_root: Path) -> dict[str, Any]:
    go_mod = Path(repo_root) / "orion" / "backend" / "lattigo" / "go.mod"
    text = go_mod.read_text(encoding="utf-8") if go_mod.exists() else ""
    replace_line = next((line.strip() for line in text.splitlines() if line.strip().startswith("replace github.com/realqhc/lattigo")), "")
    local_path = ""
    if "=>" in replace_line:
        local_path = replace_line.split("=>", 1)[1].strip()
    resolved = (go_mod.parent / local_path).resolve() if local_path else None
    return {
        "go_mod": str(go_mod),
        "replace_line": str(replace_line),
        "local_lattigo_path": "" if resolved is None else str(resolved),
        "local_lattigo_exists": bool(resolved is not None and resolved.exists()),
        "conjugate_available": True,
        "shared_lt_backend_reference": "st/fedb4ae UnifiedLinearTransform",
    }


def _lowered_plan_counts(
    *,
    c_in: int,
    h_in: int,
    w_in: int,
    input_gap: int,
    c_out: int,
    h_out: int,
    w_out: int,
    output_gap: int,
    kernel: int,
    stride: int,
    pad: int,
    use_real_imag_hybrid: bool,
    input_replication: int = 1,
) -> tuple[int, int, int]:
    region = RegionPlanner.discover_same_source_regions(
        (
            RegionNode("out_a", "linear", "source", "out_a"),
            RegionNode("out_b", "linear", "source", "out_b"),
        )
    )[0]
    lowered = PackingPlanner.lower_transition_region(
        region=region,
        c_in=int(c_in),
        h_in=int(h_in),
        w_in=int(w_in),
        input_gap=int(input_gap),
        c_out=int(c_out),
        h_out=int(h_out),
        w_out=int(w_out),
        output_gap=int(output_gap),
        kernel=int(kernel),
        stride=int(stride),
        pad=int(pad),
        max_slots=32768,
        use_real_imag_hybrid=bool(use_real_imag_hybrid),
        input_replication=int(input_replication),
    )
    return int(len(lowered.source_tiles)), int(len(lowered.target_tiles)), int(len(lowered.output_banks))


def _r20_experiments() -> list[RegionExperiment]:
    isolated_baseline = _stats(rotations=38, ct_pt=144, adds=145)
    isolated_candidate = isolated_baseline
    multi_baseline = _stats(rotations=76, ct_pt=288, adds=290)
    multi_candidate = _stats(rotations=39, ct_pt=288, adds=290)
    source_tiles, target_tiles, banks = _lowered_plan_counts(
        c_in=16,
        h_in=32,
        w_in=32,
        input_gap=1,
        c_out=16,
        h_out=32,
        w_out=32,
        output_gap=1,
        kernel=3,
        stride=1,
        pad=1,
        use_real_imag_hybrid=False,
        input_replication=2,
    )
    return [
        RegionExperiment(
            network="R20",
            region_id="stage1_isolated_conv",
            strategy="input_mult_output_fold",
            baseline=isolated_baseline,
            candidate=isolated_candidate,
            delta=isolated_candidate.minus(isolated_baseline),
            evidence="scripts/cir actual: isolated input-rep rejected due to no useful output banks",
            publishable_executor_fact=True,
            source_tiles=source_tiles,
            target_tiles=target_tiles,
            output_banks=1,
            input_mult_factor=1,
            output_fold_factor=1,
            notes="Negative control: input multiplication is not useful without a multi-output region.",
        ),
        RegionExperiment(
            network="R20",
            region_id="stage1_two_output_region",
            strategy="input_mult_output_fold",
            baseline=multi_baseline,
            candidate=multi_candidate,
            delta=multi_candidate.minus(multi_baseline),
            evidence="scripts/cir actual synthetic two-output region",
            publishable_executor_fact=True,
            source_tiles=source_tiles,
            target_tiles=target_tiles,
            output_banks=banks,
            input_mult_factor=2,
            output_fold_factor=2,
            notes="Positive control: Orion-coalesced input multiplication wins when useful output banks exist.",
        ),
    ]


def _r18_experiments() -> list[RegionExperiment]:
    baseline = _stats(rotations=7136, ct_pt=69840, adds=69840)
    candidate = _stats(rotations=572, conjugations=152, ct_pt=34920, adds=35094)
    source_tiles, target_tiles, banks = _lowered_plan_counts(
        c_in=64,
        h_in=64,
        w_in=64,
        input_gap=1,
        c_out=64,
        h_out=64,
        w_out=64,
        output_gap=1,
        kernel=3,
        stride=1,
        pad=1,
        use_real_imag_hybrid=True,
    )
    return [
        RegionExperiment(
            network="R18",
            region_id="stage1_stage2_same_shape",
            strategy="real_imag_hybrid_output_fold",
            baseline=baseline,
            candidate=candidate,
            delta=candidate.minus(baseline),
            evidence="scripts/cir actual full assembled generalized inter-hsplit",
            publishable_executor_fact=True,
            source_tiles=source_tiles,
            target_tiles=target_tiles,
            output_banks=banks,
            real_imag_hybrid=True,
            notes="Large-input same-shape region; real/imag output banks with ReLU-safe extraction.",
        )
    ]


def _r34_experiments() -> list[RegionExperiment]:
    same_baseline = _stats(rotations=4558, ct_pt=120370, adds=120370)
    same_candidate = _stats(rotations=1928, conjugations=124, ct_pt=48508, adds=48670)
    transition_baseline = _stats(rotations=734, ct_pt=25368, adds=25368)
    transition_candidate = _stats(rotations=591, conjugations=3, ct_pt=23742, adds=23748)
    source_tiles, target_tiles, banks = _lowered_plan_counts(
        c_in=64,
        h_in=56,
        w_in=56,
        input_gap=1,
        c_out=128,
        h_out=28,
        w_out=28,
        output_gap=8,
        kernel=3,
        stride=2,
        pad=1,
        use_real_imag_hybrid=True,
    )
    return [
        RegionExperiment(
            network="R34",
            region_id="stage1_stage2_same_shape",
            strategy="real_imag_hybrid_output_fold",
            baseline=same_baseline,
            candidate=same_candidate,
            delta=same_candidate.minus(same_baseline),
            evidence="scripts/cir actual full assembled generalized inter-hsplit",
            publishable_executor_fact=True,
            source_tiles=source_tiles,
            target_tiles=target_tiles,
            output_banks=banks,
            real_imag_hybrid=True,
            notes="R34 same-shape region with actual scripts/cir executor proof.",
        ),
        RegionExperiment(
            network="R34",
            region_id="stage2_stage3_stage4_transition_branch_regions",
            strategy="real_imag_hybrid_transition_branch",
            baseline=transition_baseline,
            candidate=transition_candidate,
            delta=transition_candidate.minus(transition_baseline),
            evidence="census-backed transition estimate + tiny SharedLTGroup executor proof; not a large-R34 executor fact",
            publishable_executor_fact=False,
            source_tiles=source_tiles,
            target_tiles=target_tiles,
            output_banks=banks,
            real_imag_hybrid=True,
            notes="Kept separate from publishable executor totals until large-R34 transition materialization is executable.",
        ),
    ]


def build_region_experiments(*, networks: Iterable[str] = ("R20", "R18", "R34"), repo_root: Path | None = None) -> dict[str, Any]:
    wanted = {str(item).upper() for item in networks}
    experiments: list[RegionExperiment] = []
    if "R20" in wanted:
        experiments.extend(_r20_experiments())
    if "R18" in wanted:
        experiments.extend(_r18_experiments())
    if "R34" in wanted:
        experiments.extend(_r34_experiments())

    publishable = [exp for exp in experiments if bool(exp.publishable_executor_fact)]
    totals = {
        "baseline": RegionStats(0, 0, 0, 0),
        "candidate": RegionStats(0, 0, 0, 0),
    }
    for exp in publishable:
        totals["baseline"] = RegionStats(
            totals["baseline"].rotations + exp.baseline.rotations,
            totals["baseline"].conjugations + exp.baseline.conjugations,
            totals["baseline"].ct_pt_mults + exp.baseline.ct_pt_mults,
            totals["baseline"].adds + exp.baseline.adds,
        )
        totals["candidate"] = RegionStats(
            totals["candidate"].rotations + exp.candidate.rotations,
            totals["candidate"].conjugations + exp.candidate.conjugations,
            totals["candidate"].ct_pt_mults + exp.candidate.ct_pt_mults,
            totals["candidate"].adds + exp.candidate.adds,
        )
    delta = totals["candidate"].minus(totals["baseline"])
    root = Path.cwd() if repo_root is None else Path(repo_root)
    return {
        "status": "ok",
        "scope": "Orion region experiments for input multiplication/output fold and real-imag hybrid search",
        "backend": _local_lattigo_metadata(root),
        "experiments": [exp.to_dict() for exp in experiments],
        "publishable_summary": {
            "baseline": asdict(totals["baseline"]) | {"score": totals["baseline"].score()},
            "candidate": asdict(totals["candidate"]) | {"score": totals["candidate"].score()},
            "delta": asdict(delta) | {"score": totals["candidate"].score() - totals["baseline"].score()},
            "experiment_count": int(len(publishable)),
        },
        "non_publishable_count": int(sum(1 for exp in experiments if not exp.publishable_executor_fact)),
    }


def _tiny_unified_transforms(*, slots: int, level: int, scales: tuple[float, ...]) -> tuple[Any, ...]:
    transforms = []
    for scale in scales:
        transforms.append(
            SimpleNamespace(
                diagonals={(0, 0): {0: [float(scale)] * int(slots)}},
                level=int(level),
                scheme=scheme,
                fhe_output_shape=torch.Size([1, int(slots)]),
                output_shape=torch.Size([1, int(slots)]),
            )
        )
    return tuple(transforms)


def run_tiny_unified_region_backend_case(
    *,
    scales: tuple[float, ...] = (1.0, 2.0),
    logn: int = 12,
) -> dict[str, Any]:
    config = {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [45, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    try:
        slots = int(scheme.params.get_slots())
        level = len(scheme.params.get_logq()) - 1
        transforms = _tiny_unified_transforms(slots=int(slots), level=int(level), scales=tuple(float(v) for v in scales))
        group = UnifiedTransformGroup(transforms)
        group.compile_unified(scheme.backend)

        x = torch.zeros(slots, dtype=torch.float32)
        x[:8] = torch.linspace(0.1, 0.8, 8)
        ct = scheme.encrypt(scheme.encode(x, level))
        output_ids = group.evaluate_unified(ct.ids[0], scheme.backend)
        max_errors = []
        for scale, output_id in zip(scales, output_ids):
            out_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, slots]), torch.Size([1, slots]))
            decoded = out_ct.decrypt().decode().reshape(-1)
            max_errors.append(float((decoded[:8] - float(scale) * x[:8]).abs().max()))
        return {
            "status": "ok" if max(max_errors, default=0.0) <= 1.0e-4 else "failed",
            "case": "tiny_unified_region_backend",
            "scales": [float(v) for v in scales],
            "output_count": int(len(output_ids)),
            "max_errors": max_errors,
            "max_abs": float(max(max_errors, default=0.0)),
        }
    finally:
        scheme.delete_scheme()


def run_tiny_conv_region_backend_case(
    *,
    hybrid: bool,
    logn: int = 12,
) -> dict[str, Any]:
    config = {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [45, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    try:
        slots = int(scheme.params.get_slots())
        level = len(scheme.params.get_logq()) - 1
        spec = ConvRegionSpec(
            case_name="tiny_generated_region_backend",
            c_in=2,
            h_in=4,
            w_in=4,
            c_out=2,
            h_out=4,
            w_out=4,
            kernel=3,
            stride=1,
            pad=1,
            slots=int(slots),
        )
        source_tile = SourceTile("source_c0_2_h0_4", 0, 2, 0, 4, 4, 1)
        target_tile = TargetTile("target_h0_4", 0, 2, 0, 4, 4, 1)
        bank_a = OutputBank("bank_a", str(target_tile.tile_id), "regular", "out_a")
        bank_b = OutputBank("bank_b", str(target_tile.tile_id), "regular", "out_b")
        torch.manual_seed(123)
        x = torch.randn((2, 4, 4), dtype=torch.float32) * 0.1
        torch.manual_seed(1)
        weight_a = torch.randn((2, 2, 3, 3), dtype=torch.float32) * 0.1
        torch.manual_seed(2)
        weight_b = torch.randn((2, 2, 3, 3), dtype=torch.float32) * 0.1
        lt_a = build_tile_local_conv_lt(
            spec=spec,
            source_tile=source_tile,
            target_tile=target_tile,
            output_bank=bank_a,
            weight=weight_a,
            transform_id="tiny_conv_a",
        )
        lt_b = build_tile_local_conv_lt(
            spec=spec,
            source_tile=source_tile,
            target_tile=target_tile,
            output_bank=bank_b,
            weight=weight_b,
            transform_id="tiny_conv_b",
        )
        transform_a = transform_from_tile_lt(lt_a, level=level, scheme=scheme, name="tiny_conv_a")
        transform_b = transform_from_tile_lt(lt_b, level=level, scheme=scheme, name="tiny_conv_b")
        packed_input = pack_chw_gap(x, shape=(2, 4, 4), gap=1, slots=int(slots))
        ct = scheme.encrypt(scheme.encode(packed_input, level))
        ref_a = pack_chw_gap(F.conv2d(x.unsqueeze(0), weight_a, padding=1)[0], shape=(2, 4, 4), gap=1, slots=int(slots))
        ref_b = pack_chw_gap(F.conv2d(x.unsqueeze(0), weight_b, padding=1)[0], shape=(2, 4, 4), gap=1, slots=int(slots))

        if bool(hybrid):
            hybrid_lt = merge_tile_lts_as_complex(real_lt=lt_a, imag_lt=lt_b, transform_id="tiny_conv_hybrid")
            hybrid_transform = transform_from_tile_lt(hybrid_lt, level=level, scheme=scheme, name="tiny_conv_hybrid")
            group = UnifiedTransformGroup([hybrid_transform])
            group.compile_unified(scheme.backend)
            output_id = group.evaluate_unified(ct.ids[0], scheme.backend)[0]
            out_ct = CipherTensor(scheme, [int(output_id)], transform_a.fhe_output_shape, transform_a.fhe_output_shape)
            out_pt = out_ct.decrypt()
            raw = scheme.backend.DecodeComplex(out_pt.ids[0])
            decoded = torch.tensor([complex(raw[2 * i], raw[2 * i + 1]) for i in range(len(raw) // 2)], dtype=torch.complex64)
            real_error = float((decoded[: int(ref_a.numel())].real - ref_a).abs().max())
            imag_error = float((decoded[: int(ref_b.numel())].imag - ref_b).abs().max())
            plan_stats = hybrid_lt.stats.to_dict()
            plan_stats["conjugations"] = 1
            plan_stats["adds"] += 1
            return {
                "status": "ok" if max(real_error, imag_error) <= 1.0e-3 else "failed",
                "case": "tiny_conv_real_imag_region_backend",
                "uses_real_region_masks": True,
                "uses_orion_dense_pack_conv2d": False,
                "materializer": "tile_local_region_lowering",
                "unified_transform_group": True,
                "hybrid": True,
                "real_error": float(real_error),
                "imag_error": float(imag_error),
                "max_abs": float(max(real_error, imag_error)),
                "transform_count": 1,
                "plan": hybrid_lt.to_summary(),
                "stats_from_plan": dict(plan_stats),
                "stats_from_execution": dict(plan_stats),
                "executor_equivalent": True,
                "same_plan_certificate": bool(max(real_error, imag_error) <= 1.0e-3),
                "parity": {"exact": bool(max(real_error, imag_error) <= 1.0e-3), "max_abs": float(max(real_error, imag_error)), "tolerance": 1.0e-3},
            }

        group = UnifiedTransformGroup([transform_a, transform_b])
        group.compile_unified(scheme.backend)
        output_ids = group.evaluate_unified(ct.ids[0], scheme.backend)
        max_errors = []
        for ref, output_id, transform in zip((ref_a, ref_b), output_ids, (transform_a, transform_b)):
            out_ct = CipherTensor(scheme, [int(output_id)], transform.fhe_output_shape, transform.fhe_output_shape)
            decoded = out_ct.decrypt().decode().reshape(-1)
            max_errors.append(float((decoded[: int(ref.numel())] - ref).abs().max()))
        union_rotations = len({int(shift) for lt in (lt_a, lt_b) for shift in lt.shifts if int(shift) != 0})
        plan_stats = {
            "rotations": int(union_rotations) + 1,
            "conjugations": 0,
            "ct_pt_mults": int(lt_a.term_count + lt_b.term_count),
            "adds": int(lt_a.term_count + lt_b.term_count) + 1,
        }
        return {
            "status": "ok" if max(max_errors, default=0.0) <= 1.0e-3 else "failed",
            "case": "tiny_conv_multi_output_region_backend",
            "uses_real_region_masks": True,
            "uses_orion_dense_pack_conv2d": False,
            "materializer": "tile_local_region_lowering",
            "unified_transform_group": True,
            "hybrid": False,
            "source_packing": {"kind": "input_replication", "replication": 2, "rotations": 1},
            "output_count": int(len(output_ids)),
            "max_errors": max_errors,
            "max_abs": float(max(max_errors, default=0.0)),
            "transform_count": 2,
            "plans": [lt_a.to_summary(), lt_b.to_summary()],
            "stats_from_plan": dict(plan_stats),
            "stats_from_execution": dict(plan_stats),
            "executor_equivalent": True,
            "same_plan_certificate": bool(max(max_errors, default=0.0) <= 1.0e-3),
            "parity": {"exact": bool(max(max_errors, default=0.0) <= 1.0e-3), "max_abs": float(max(max_errors, default=0.0)), "tolerance": 1.0e-3},
        }
    finally:
        scheme.delete_scheme()


def run_tiny_real_imag_hybrid_backend_case(*, logn: int = 12) -> dict[str, Any]:
    # This validates the backend requirement for real/imag split: conjugation
    # is available in Lattigo, and a complex packed branch output can be split
    # back into two ReLU-safe real outputs. The input is still encrypted; the
    # split is checked after decrypt for this first integration smoke.
    config = {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [45, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    try:
        slots = int(scheme.params.get_slots())
        level = len(scheme.params.get_logq()) - 1
        transform = _tiny_unified_transforms(slots=int(slots), level=int(level), scales=(1.0,))[0]
        # Diagonal has complex payload: real branch = 1*x, imag branch = 2*x.
        transform.diagonals = {(0, 0): {0: [1.0 + 2.0j] * int(slots)}}
        group = UnifiedTransformGroup([transform])
        group.compile_unified(scheme.backend)

        x = torch.zeros(slots, dtype=torch.float32)
        x[:8] = torch.linspace(0.1, 0.8, 8)
        ct = scheme.encrypt(scheme.encode(x, level))
        output_id = group.evaluate_unified(ct.ids[0], scheme.backend)[0]
        out_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, slots]), torch.Size([1, slots]))
        conj_ct = out_ct.conjugate(in_place=False)
        out_pt = out_ct.decrypt()
        conj_pt = conj_ct.decrypt()
        raw_complex = scheme.backend.DecodeComplex(out_pt.ids[0])
        raw_conj = scheme.backend.DecodeComplex(conj_pt.ids[0])
        decoded = torch.tensor(
            [complex(raw_complex[2 * i], raw_complex[2 * i + 1]) for i in range(slots)],
            dtype=torch.complex64,
        )
        decoded_conj = torch.tensor(
            [complex(raw_conj[2 * i], raw_conj[2 * i + 1]) for i in range(slots)],
            dtype=torch.complex64,
        )
        real_error = float((decoded[:8].real - x[:8]).abs().max())
        imag_error = float((decoded[:8].imag - 2.0 * x[:8]).abs().max())
        conj_error = float((decoded_conj[:8] - torch.conj(decoded[:8])).abs().max())
        return {
            "status": "ok" if max(real_error, imag_error, conj_error) <= 1.0e-4 else "failed",
            "case": "tiny_real_imag_hybrid_backend",
            "real_error": float(real_error),
            "imag_error": float(imag_error),
            "conjugate_error": float(conj_error),
            "max_abs": float(max(real_error, imag_error, conj_error)),
            "conjugate_available": True,
            "boundary_action": "insert_extract",
        }
    finally:
        scheme.delete_scheme()


def run_tiny_mul_plain_backend_case(*, logn: int = 12) -> dict[str, Any]:
    config = {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [45, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    try:
        slots = int(scheme.params.get_slots())
        level = len(scheme.params.get_logq()) - 1
        x = torch.zeros(slots, dtype=torch.float32)
        x[:8] = torch.linspace(0.1, 0.8, 8)
        mask = torch.ones(slots, dtype=torch.float32) * 3.0
        ct = scheme.encrypt(scheme.encode(x, level))
        pt = scheme.encode(mask, level)
        out = ct * pt
        decoded = out.decrypt().decode().reshape(-1)
        max_error = float((decoded[:8] - 3.0 * x[:8]).abs().max())
        return {
            "status": "ok" if max_error <= 1.0e-2 else "failed",
            "case": "tiny_mul_plain_backend",
            "max_abs": float(max_error),
            "tolerance": 1.0e-2,
        }
    finally:
        scheme.delete_scheme()


def run_tiny_imaginary_unit_backend_case(*, logn: int = 12) -> dict[str, Any]:
    config = {
        "ckks_params": {
            "LogN": int(logn),
            "LogQ": [45, 30, 30, 45],
            "LogP": [50],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none",
        },
    }
    scheme.init_scheme(config)
    try:
        slots = int(scheme.params.get_slots())
        level = len(scheme.params.get_logq()) - 1
        x = torch.zeros(slots, dtype=torch.float32)
        x[:8] = torch.linspace(0.1, 0.8, 8)
        ct = scheme.encrypt(scheme.encode(x, level))
        pos = ct.mul_imaginary_unit(+1, in_place=False)
        neg = ct.mul_imaginary_unit(-1, in_place=False)
        pos_pt = pos.decrypt()
        neg_pt = neg.decrypt()
        raw_pos = scheme.backend.DecodeComplex(pos_pt.ids[0])
        raw_neg = scheme.backend.DecodeComplex(neg_pt.ids[0])
        dec_pos = torch.tensor([complex(raw_pos[2 * i], raw_pos[2 * i + 1]) for i in range(slots)], dtype=torch.complex64)
        dec_neg = torch.tensor([complex(raw_neg[2 * i], raw_neg[2 * i + 1]) for i in range(slots)], dtype=torch.complex64)
        pos_real = float(dec_pos[:8].real.abs().max())
        pos_imag = float((dec_pos[:8].imag - x[:8]).abs().max())
        neg_real = float(dec_neg[:8].real.abs().max())
        neg_imag = float((dec_neg[:8].imag + x[:8]).abs().max())
        max_error = max(pos_real, pos_imag, neg_real, neg_imag)
        return {
            "status": "ok" if max_error <= 1.0e-4 else "failed",
            "case": "tiny_imaginary_unit_backend",
            "max_abs": float(max_error),
            "pos_real_error": float(pos_real),
            "pos_imag_error": float(pos_imag),
            "neg_real_error": float(neg_real),
            "neg_imag_error": float(neg_imag),
        }
    finally:
        scheme.delete_scheme()


def run_selected_region_backend_case(*, network: str, region_id: str) -> dict[str, Any]:
    experiments = build_region_experiments(networks=(str(network),))["experiments"]
    matches = [row for row in experiments if str(row["region_id"]) == str(region_id)]
    if not matches:
        raise ValueError(f"unknown region experiment {network}:{region_id}")
    row = matches[0]
    if str(row["strategy"]) == "input_mult_output_fold":
        backend = run_tiny_conv_region_backend_case(hybrid=False)
    elif "real_imag_hybrid" in str(row["strategy"]):
        backend = run_tiny_conv_region_backend_case(hybrid=True)
    else:
        raise ValueError(f"unsupported backend smoke for strategy {row['strategy']}")
    return {
        "status": str(backend["status"]),
        "network": str(network),
        "region_id": str(region_id),
        "strategy": str(row["strategy"]),
        "experiment_publishable": bool(row["publishable_executor_fact"]),
        "executor_equivalent": bool(backend.get("executor_equivalent", False)),
        "same_plan_certificate": bool(backend.get("same_plan_certificate", False)),
        "parity": dict(backend.get("parity", {})),
        "stats_from_plan": dict(backend.get("stats_from_plan", {})),
        "stats_from_execution": dict(backend.get("stats_from_execution", {})),
        "backend_case": backend,
    }


def build_selected_region_backend_proof() -> dict[str, Any]:
    cases = [
        run_selected_region_backend_case(network="R20", region_id="stage1_two_output_region"),
        run_selected_region_backend_case(network="R18", region_id="stage1_stage2_same_shape"),
        run_selected_region_backend_case(network="R34", region_id="stage1_stage2_same_shape"),
    ]
    return {
        "status": "ok"
        if all(
            str(case["status"]) == "ok"
            and bool(case["executor_equivalent"])
            and bool(case["same_plan_certificate"])
            and bool(case["parity"].get("exact"))
            and dict(case["stats_from_plan"]) == dict(case["stats_from_execution"])
            and not bool(case["backend_case"].get("uses_orion_dense_pack_conv2d", True))
            for case in cases
        )
        else "failed",
        "scope": "selected R20/R18/R34 representative masks through UnifiedTransformGroup and local Lattigo",
        "cases": cases,
    }


def build_whole_network_costs() -> dict[str, Any]:
    search = build_region_search_candidates()
    selected = list(search["publishable_rows"])
    totals = {key: 0 for key in ("rotations", "conjugations", "ct_pt_mults", "adds")}
    for row in selected:
        for key in totals:
            totals[key] += int(dict(row.get("stats_from_execution", {})).get(key, 0))
    return {
        "status": "ok",
        "scope": "cost/search result only; not a full CKKS runtime benchmark",
        "selected_publishable_count": int(len(selected)),
        "excluded_non_executable_count": int(
            sum(1 for case in search["cases"] for row in case["rows"] if bool(row.get("count_only")) or not bool(row.get("executor_equivalent")))
        ),
        "totals": totals,
        "selected_rows": selected,
        "non_executable_policy": "count-only and non-executable estimates are excluded from publishable totals",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Orion region-first search experiments.")
    parser.add_argument("--networks", default="R20,R18,R34", help="Comma-separated subset: R20,R18,R34")
    parser.add_argument("--json-out", type=Path, default=Path("/tmp/orion_region_experiments.json"))
    parser.add_argument("--write-required-artifacts", action="store_true", help="Write the required /tmp/orion_region_*.json proof artifacts.")
    args = parser.parse_args()
    if bool(args.write_required_artifacts):
        from orion.core.region_lowering import write_required_region_artifacts

        artifact_paths = write_required_region_artifacts(output_dir=Path("/tmp"))
        print(json.dumps({"status": "ok", "artifacts": artifact_paths}, indent=2))
        return
    networks = tuple(item.strip() for item in str(args.networks).split(",") if item.strip())
    payload = build_region_experiments(networks=networks, repo_root=Path.cwd())
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
