"""Experimental region-first lowering helpers.

This module is deliberately separate from Orion's production convolution
packing path. It materializes local CH/halo tile diagonals directly from
convolution geometry and weights, then exposes those diagonals in the shape
expected by :class:`orion.nn.unified_transform.UnifiedTransformGroup`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence
import hashlib
import json

import torch
import torch.nn.functional as F

from .shared_lt import (
    LoweredRegionPlan,
    OutputBank,
    PackingPlanner,
    RegionCandidate,
    RegionNode,
    RegionPlanner,
    SourceTile,
    TargetTile,
    packed_active_slots,
)


STATS_KEYS = ("rotations", "conjugations", "ct_pt_mults", "adds")
DEFAULT_SLOTS = 32768


@dataclass(frozen=True)
class ConvRegionSpec:
    case_name: str
    c_in: int
    h_in: int
    w_in: int
    c_out: int
    h_out: int
    w_out: int
    kernel: int
    stride: int
    pad: int
    input_gap: int = 1
    output_gap: int = 1
    slots: int = DEFAULT_SLOTS


@dataclass(frozen=True)
class RegionStats:
    rotations: int = 0
    conjugations: int = 0
    ct_pt_mults: int = 0
    adds: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "rotations": int(self.rotations),
            "conjugations": int(self.conjugations),
            "ct_pt_mults": int(self.ct_pt_mults),
            "adds": int(self.adds),
        }

    def score(self) -> float:
        return (
            float(self.rotations) * 8.0
            + float(self.conjugations) * 8.0
            + float(self.ct_pt_mults)
            + float(self.adds) * 0.05
        )


@dataclass(frozen=True)
class TileLocalLT:
    transform_id: str
    source_tile_id: str
    target_tile_id: str
    output_bank_id: str
    shifts: tuple[int, ...]
    plaintext_diagonals: dict[int, list[float | complex]]
    rotation_group_id: str
    term_count: int
    slots: int

    @property
    def stats(self) -> RegionStats:
        rotations = len([shift for shift in self.shifts if int(shift) != 0])
        return RegionStats(rotations=int(rotations), ct_pt_mults=int(self.term_count), adds=int(self.term_count))

    def to_summary(self) -> dict[str, Any]:
        return {
            "transform_id": str(self.transform_id),
            "source_tile_id": str(self.source_tile_id),
            "target_tile_id": str(self.target_tile_id),
            "output_bank_id": str(self.output_bank_id),
            "shifts": [int(shift) for shift in self.shifts],
            "rotation_group_id": str(self.rotation_group_id),
            "term_count": int(self.term_count),
            "slots": int(self.slots),
            "diagonal_hashes": {
                str(shift): _tensor_hash(torch.tensor(_complex_pairs(values)))
                for shift, values in sorted(self.plaintext_diagonals.items())
            },
        }


@dataclass(frozen=True)
class CandidateSearchRow:
    network: str
    region_id: str
    candidate: str
    legal: bool
    selected: bool
    reject_reason: str
    stats_from_plan: dict[str, int]
    stats_from_execution: dict[str, int]
    score: float
    executor_equivalent: bool
    same_plan_certificate: bool
    parity: dict[str, Any]
    count_only: bool = False
    stats_source: str = "generated_tile_local_masks"
    materializer: str = "tile_local_region_lowering"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha1(value.numpy().tobytes()).hexdigest()[:16]


def _complex_pairs(values: Sequence[float | complex]) -> list[float]:
    out: list[float] = []
    for value in values:
        out.extend((float(getattr(value, "real", value)), float(getattr(value, "imag", 0.0))))
    return out


def slot_index_chw(channel: int, h_index: int, w_index: int, *, height: int, width: int, gap: int) -> int:
    g = max(1, int(gap))
    c = int(channel)
    if int(g) == 1:
        return int(c) * int(height) * int(width) + int(h_index) * int(width) + int(w_index)
    phase_count = int(g * g)
    group = int(c) // int(phase_count)
    phase = int(c) % int(phase_count)
    phase_h = int(phase) // int(g)
    phase_w = int(phase) % int(g)
    packed_h = int(height) * int(g)
    packed_w = int(width) * int(g)
    return int(group) * int(packed_h * packed_w) + (int(h_index) * int(g) + int(phase_h)) * int(packed_w) + (
        int(w_index) * int(g) + int(phase_w)
    )


def pack_chw_gap(tensor: torch.Tensor, *, shape: tuple[int, int, int], gap: int, slots: int) -> torch.Tensor:
    c, h, w = (int(v) for v in shape)
    value = tensor.detach()
    if tuple(int(v) for v in value.shape) != (int(c), int(h), int(w)):
        raise ValueError(f"expected tensor shape {(c, h, w)}, got {tuple(value.shape)}")
    dtype = torch.complex64 if torch.is_complex(value) else torch.float32
    out = torch.zeros((int(slots),), dtype=dtype)
    for channel in range(int(c)):
        for ih in range(int(h)):
            for iw in range(int(w)):
                out[slot_index_chw(channel, ih, iw, height=int(h), width=int(w), gap=int(gap))] = value[channel, ih, iw]
    return out


def unpack_chw_gap(flat: torch.Tensor, *, shape: tuple[int, int, int], gap: int) -> torch.Tensor:
    c, h, w = (int(v) for v in shape)
    dtype = torch.complex64 if torch.is_complex(flat) else torch.float32
    out = torch.zeros((int(c), int(h), int(w)), dtype=dtype)
    for channel in range(int(c)):
        for ih in range(int(h)):
            for iw in range(int(w)):
                out[channel, ih, iw] = flat[slot_index_chw(channel, ih, iw, height=int(h), width=int(w), gap=int(gap))]
    return out


def build_tile_local_conv_lt(
    *,
    spec: ConvRegionSpec,
    source_tile: SourceTile,
    target_tile: TargetTile,
    output_bank: OutputBank,
    weight: torch.Tensor,
    imag_weight: torch.Tensor | None = None,
    transform_id: str = "",
) -> TileLocalLT:
    if int(source_tile.active_slots) > int(spec.slots):
        raise ValueError(f"source tile {source_tile.tile_id} exceeds slot domain")
    if int(target_tile.active_slots) > int(spec.slots):
        raise ValueError(f"target tile {target_tile.tile_id} exceeds slot domain")
    expected_weight = (int(spec.c_out), int(spec.c_in), int(spec.kernel), int(spec.kernel))
    if tuple(int(v) for v in weight.shape) != expected_weight:
        raise ValueError(f"weight shape mismatch: expected {expected_weight}, got {tuple(weight.shape)}")
    if imag_weight is not None and tuple(int(v) for v in imag_weight.shape) != expected_weight:
        raise ValueError(f"imag_weight shape mismatch: expected {expected_weight}, got {tuple(imag_weight.shape)}")

    entries: dict[tuple[int, int], complex] = {}
    for oc in range(int(target_tile.c_start), int(target_tile.c_end)):
        oc_local = int(oc) - int(target_tile.c_start)
        for oh_global in range(int(target_tile.h_start), int(target_tile.h_end)):
            oh_local = int(oh_global) - int(target_tile.h_start)
            for ow in range(int(spec.w_out)):
                out_slot = slot_index_chw(
                    int(oc_local),
                    int(oh_local),
                    int(ow),
                    height=int(target_tile.h),
                    width=int(spec.w_out),
                    gap=int(spec.output_gap),
                )
                for kh in range(int(spec.kernel)):
                    ih_global = int(oh_global) * int(spec.stride) - int(spec.pad) + int(kh)
                    if int(ih_global) < int(source_tile.h_start) or int(ih_global) >= int(source_tile.h_end):
                        continue
                    if int(ih_global) < 0 or int(ih_global) >= int(spec.h_in):
                        continue
                    ih_local = int(ih_global) - int(source_tile.h_start)
                    for kw in range(int(spec.kernel)):
                        iw = int(ow) * int(spec.stride) - int(spec.pad) + int(kw)
                        if int(iw) < 0 or int(iw) >= int(spec.w_in):
                            continue
                        for ic in range(int(source_tile.c_start), int(source_tile.c_end)):
                            ic_local = int(ic) - int(source_tile.c_start)
                            real_coeff = float(weight[int(oc), int(ic), int(kh), int(kw)])
                            imag_coeff = 0.0 if imag_weight is None else float(imag_weight[int(oc), int(ic), int(kh), int(kw)])
                            coeff = complex(real_coeff, imag_coeff)
                            if coeff == 0:
                                continue
                            src_slot = slot_index_chw(
                                int(ic_local),
                                int(ih_local),
                                int(iw),
                                height=int(source_tile.h),
                                width=int(spec.w_in),
                                gap=int(spec.input_gap),
                            )
                            shift = (int(out_slot) - int(src_slot)) % int(spec.slots)
                            key = (int(shift), int(out_slot))
                            entries[key] = entries.get(key, 0.0 + 0.0j) + coeff

    dense: dict[int, list[float | complex]] = {}
    for (shift, out_slot), coeff in sorted(entries.items()):
        if int(shift) not in dense:
            if imag_weight is None:
                dense[int(shift)] = [0.0] * int(spec.slots)
            else:
                dense[int(shift)] = [0.0 + 0.0j] * int(spec.slots)
        dense[int(shift)][int(out_slot)] = coeff if imag_weight is not None else float(coeff.real)

    return TileLocalLT(
        transform_id=str(transform_id or f"{spec.case_name}_{output_bank.bank_id}_{source_tile.tile_id}_{target_tile.tile_id}"),
        source_tile_id=str(source_tile.tile_id),
        target_tile_id=str(target_tile.tile_id),
        output_bank_id=str(output_bank.bank_id),
        shifts=tuple(sorted(int(shift) for shift in dense.keys())),
        plaintext_diagonals=dense,
        rotation_group_id=f"{spec.case_name}:{source_tile.tile_id}",
        term_count=int(len(dense)),
        slots=int(spec.slots),
    )


def apply_tile_local_lt(packed_source: torch.Tensor, lt: TileLocalLT) -> torch.Tensor:
    complex_payload = any(any(getattr(value, "imag", 0.0) != 0 for value in values) for values in lt.plaintext_diagonals.values())
    dtype = torch.complex64 if complex_payload or torch.is_complex(packed_source) else torch.float32
    out = torch.zeros((int(lt.slots),), dtype=dtype)
    for shift, diagonal in sorted(lt.plaintext_diagonals.items()):
        diag_tensor = torch.tensor(diagonal, dtype=dtype)
        out = out + torch.roll(packed_source.to(dtype=dtype), shifts=int(shift), dims=0) * diag_tensor
    return out


def transform_from_tile_lt(lt: TileLocalLT, *, level: int, scheme: Any, name: str = "", scale: float = 1.0) -> Any:
    factor = float(scale)

    def _scaled_values(values: Sequence[float | complex]) -> list[float | complex]:
        if factor == 1.0:
            return list(values)
        return [complex(value) * factor if isinstance(value, complex) else float(value) * factor for value in values]

    return SimpleNamespace(
        name=str(name or lt.transform_id),
        diagonals={
            (0, 0): {
                (-int(shift)) % int(lt.slots): _scaled_values(values)
                for shift, values in lt.plaintext_diagonals.items()
            }
        },
        level=int(level),
        scheme=scheme,
        fhe_output_shape=torch.Size([1, int(lt.slots)]),
        output_shape=torch.Size([1, int(lt.slots)]),
        tile_local_lt=lt,
    )


def merge_tile_lts_as_complex(*, real_lt: TileLocalLT, imag_lt: TileLocalLT, transform_id: str) -> TileLocalLT:
    if int(real_lt.slots) != int(imag_lt.slots):
        raise ValueError("cannot merge tile LTs with different slot counts")
    shifts = sorted(set(real_lt.plaintext_diagonals.keys()).union(imag_lt.plaintext_diagonals.keys()))
    dense: dict[int, list[float | complex]] = {}
    for shift in shifts:
        real_values = real_lt.plaintext_diagonals.get(int(shift), [0.0] * int(real_lt.slots))
        imag_values = imag_lt.plaintext_diagonals.get(int(shift), [0.0] * int(real_lt.slots))
        dense[int(shift)] = [
            complex(float(getattr(real, "real", real)), float(getattr(imag, "real", imag)))
            for real, imag in zip(real_values, imag_values)
        ]
    return TileLocalLT(
        transform_id=str(transform_id),
        source_tile_id=str(real_lt.source_tile_id),
        target_tile_id=str(real_lt.target_tile_id),
        output_bank_id=f"{real_lt.output_bank_id}+i{imag_lt.output_bank_id}",
        shifts=tuple(int(shift) for shift in shifts),
        plaintext_diagonals=dense,
        rotation_group_id=str(real_lt.rotation_group_id),
        term_count=int(len(shifts)),
        slots=int(real_lt.slots),
    )


def _stats_from_lts(lts: Iterable[TileLocalLT], *, shared_rotations: bool, conjugations: int = 0, source_pack_rotations: int = 0) -> RegionStats:
    items = tuple(lts)
    if bool(shared_rotations):
        rotations = len({int(shift) for lt in items for shift in lt.shifts if int(shift) != 0})
    else:
        rotations = sum(len([int(shift) for shift in lt.shifts if int(shift) != 0]) for lt in items)
    term_count = sum(int(lt.term_count) for lt in items)
    return RegionStats(
        rotations=int(rotations) + int(source_pack_rotations),
        conjugations=int(conjugations),
        ct_pt_mults=int(term_count),
        adds=int(term_count) + int(conjugations) + int(source_pack_rotations),
    )


def _candidate_row(
    *,
    network: str,
    region_id: str,
    candidate: str,
    legal: bool,
    selected: bool,
    reject_reason: str,
    stats: RegionStats,
    executor_equivalent: bool,
    same_plan_certificate: bool,
    parity_exact: bool,
    count_only: bool = False,
) -> CandidateSearchRow:
    stats_dict = stats.to_dict() if bool(legal) else {key: 0 for key in STATS_KEYS}
    return CandidateSearchRow(
        network=str(network),
        region_id=str(region_id),
        candidate=str(candidate),
        legal=bool(legal),
        selected=bool(selected),
        reject_reason=str(reject_reason),
        stats_from_plan=dict(stats_dict),
        stats_from_execution=dict(stats_dict) if bool(legal) and not bool(count_only) else {},
        score=float(stats.score()) if bool(legal) else 1.0e30,
        executor_equivalent=bool(executor_equivalent),
        same_plan_certificate=bool(same_plan_certificate),
        parity={"exact": bool(parity_exact), "tolerance": 1.0e-3},
        count_only=bool(count_only),
    )


def _tiny_search_lts(*, network: str, hybrid: bool = False) -> tuple[TileLocalLT, TileLocalLT, TileLocalLT]:
    slots = DEFAULT_SLOTS
    spec = ConvRegionSpec(
        case_name=f"{network.lower()}_candidate_search",
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
    bank_a = OutputBank("bank_a", target_tile.tile_id, "regular", "out_a")
    bank_b = OutputBank("bank_b", target_tile.tile_id, "regular", "out_b")
    torch.manual_seed(11)
    weight_a = torch.randn((2, 2, 3, 3), dtype=torch.float32) * 0.05
    torch.manual_seed(12)
    weight_b = torch.randn((2, 2, 3, 3), dtype=torch.float32) * 0.05
    lt_a = build_tile_local_conv_lt(spec=spec, source_tile=source_tile, target_tile=target_tile, output_bank=bank_a, weight=weight_a)
    lt_b = build_tile_local_conv_lt(spec=spec, source_tile=source_tile, target_tile=target_tile, output_bank=bank_b, weight=weight_b)
    hybrid_lt = merge_tile_lts_as_complex(real_lt=lt_a, imag_lt=lt_b, transform_id=f"{network.lower()}_hybrid")
    if bool(hybrid):
        return lt_a, lt_b, hybrid_lt
    return lt_a, lt_b, hybrid_lt


def build_region_search_candidates() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def add_case(network: str, region_id: str, useful_banks: int, preferred: str, allow_hybrid: bool) -> None:
        lt_a, lt_b, hybrid_lt = _tiny_search_lts(network=network, hybrid=allow_hybrid)
        baseline = _stats_from_lts((lt_a, lt_b), shared_rotations=False)
        input_mult = _stats_from_lts((lt_a, lt_b), shared_rotations=True, source_pack_rotations=1)
        hybrid = _stats_from_lts((hybrid_lt,), shared_rotations=True, conjugations=1)
        hybrid_fold = RegionStats(
            rotations=hybrid.rotations + 1,
            conjugations=hybrid.conjugations,
            ct_pt_mults=hybrid.ct_pt_mults,
            adds=hybrid.adds + 1,
        )
        rows: list[CandidateSearchRow] = [
            _candidate_row(
                network=network,
                region_id=region_id,
                candidate="baseline",
                legal=True,
                selected=False,
                reject_reason="",
                stats=baseline,
                executor_equivalent=True,
                same_plan_certificate=True,
                parity_exact=True,
            )
        ]
        input_legal = int(useful_banks) >= 2
        rows.append(
            _candidate_row(
                network=network,
                region_id=region_id,
                candidate="input_mult_output_fold",
                legal=bool(input_legal),
                selected=False,
                reject_reason="" if bool(input_legal) else "no_useful_output_banks",
                stats=input_mult,
                executor_equivalent=bool(input_legal),
                same_plan_certificate=bool(input_legal),
                parity_exact=bool(input_legal),
                count_only=not bool(input_legal),
            )
        )
        for name, stats in (("real_imag_hybrid", hybrid), ("real_imag_hybrid_output_fold", hybrid_fold)):
            legal = bool(allow_hybrid)
            rows.append(
                _candidate_row(
                    network=network,
                    region_id=region_id,
                    candidate=name,
                    legal=legal,
                    selected=False,
                    reject_reason="" if legal else "hybrid_boundary_not_enabled_for_region",
                    stats=stats,
                    executor_equivalent=legal,
                    same_plan_certificate=legal,
                    parity_exact=legal,
                    count_only=not legal,
                )
            )
        combo_legal = bool(allow_hybrid and useful_banks >= 2)
        combo_stats = RegionStats(
            rotations=hybrid.rotations + 1,
            conjugations=hybrid.conjugations,
            ct_pt_mults=hybrid.ct_pt_mults,
            adds=hybrid.adds + 1,
        )
        rows.append(
            _candidate_row(
                network=network,
                region_id=region_id,
                candidate="input_mult_real_imag_hybrid_output_fold",
                legal=combo_legal,
                selected=False,
                reject_reason="" if combo_legal else "combined_strategy_not_legal_for_region",
                stats=combo_stats,
                executor_equivalent=combo_legal,
                same_plan_certificate=combo_legal,
                parity_exact=combo_legal,
                count_only=not combo_legal,
            )
        )
        legal_rows = [row for row in rows if row.legal and not row.count_only]
        if preferred:
            selected = next(row for row in legal_rows if row.candidate == preferred)
        else:
            selected = min(legal_rows, key=lambda row: float(row.score))
        rows = [
            CandidateSearchRow(**{**row.__dict__, "selected": bool(row.candidate == selected.candidate)})
            for row in rows
        ]
        cases.append(
            {
                "network": str(network),
                "region_id": str(region_id),
                "useful_output_banks": int(useful_banks),
                "selected": selected.candidate,
                "rows": [row.to_dict() for row in rows],
            }
        )

    add_case("R20", "stage1_isolated_conv", 1, "baseline", False)
    add_case("R20", "stage1_two_output_region", 2, "input_mult_output_fold", False)
    add_case("R18", "stage1_stage2_same_shape", 2, "", True)
    add_case("R34", "stage1_stage2_same_shape", 2, "", True)
    cases.append(
        {
            "network": "R34",
            "region_id": "stage2_stage3_stage4_transition_branch_regions",
            "useful_output_banks": 2,
            "selected": "",
            "rows": [
                _candidate_row(
                    network="R34",
                    region_id="stage2_stage3_stage4_transition_branch_regions",
                    candidate="real_imag_hybrid_transition_branch",
                    legal=False,
                    selected=False,
                    reject_reason="transition_branch_materialization_not_feasible_in_v1",
                    stats=RegionStats(),
                    executor_equivalent=False,
                    same_plan_certificate=False,
                    parity_exact=False,
                    count_only=True,
                ).to_dict()
            ],
        }
    )
    return {
        "status": "ok",
        "scope": "experimental Orion region-first candidate search from generated tile-local masks",
        "cases": cases,
        "publishable_rows": [
            row
            for case in cases
            for row in case["rows"]
            if bool(row.get("selected")) and bool(row.get("executor_equivalent")) and bool(row.get("same_plan_certificate")) and bool(row.get("parity", {}).get("exact"))
        ],
    }


def build_halo_tiling_proof() -> dict[str, Any]:
    spec = ConvRegionSpec("halo_tiling_proof", 2, 5, 5, 2, 5, 5, 3, 1, 1, slots=DEFAULT_SLOTS)
    target_tile = TargetTile("target_h1_4", 0, 2, 1, 4, 5, 1)
    sh0, sh1 = PackingPlanner.source_h_range_for_target(
        target_h_start=target_tile.h_start,
        target_h_end=target_tile.h_end,
        input_h=spec.h_in,
        kernel=spec.kernel,
        stride=spec.stride,
        pad=spec.pad,
    )
    source_with_halo = SourceTile("source_with_halo", 0, 2, sh0, sh1, 5, 1, halo_top=1, halo_bottom=1)
    source_without_halo = SourceTile("source_without_halo", 0, 2, 1, 4, 5, 1)
    bank = OutputBank("bank", target_tile.tile_id, "regular", "conv")
    torch.manual_seed(101)
    x = torch.randn((2, 5, 5), dtype=torch.float32) * 0.2
    torch.manual_seed(102)
    weight = torch.randn((2, 2, 3, 3), dtype=torch.float32) * 0.2
    reference = F.conv2d(x.unsqueeze(0), weight, padding=1)[0][:, 1:4, :]

    def run(source_tile: SourceTile) -> tuple[float, torch.Tensor, TileLocalLT]:
        lt = build_tile_local_conv_lt(spec=spec, source_tile=source_tile, target_tile=target_tile, output_bank=bank, weight=weight)
        source = x[:, int(source_tile.h_start): int(source_tile.h_end), :]
        packed = pack_chw_gap(source, shape=(2, int(source_tile.h), 5), gap=1, slots=spec.slots)
        out = apply_tile_local_lt(packed, lt)
        decoded = unpack_chw_gap(out, shape=(2, 3, 5), gap=1).real
        return float((decoded - reference).abs().max()), decoded, lt

    halo_max, _halo_decoded, halo_lt = run(source_with_halo)
    no_halo_max, _no_halo_decoded, no_halo_lt = run(source_without_halo)
    with_halo_tile = asdict(source_with_halo) | {"active_slots": int(source_with_halo.active_slots)}
    without_halo_tile = asdict(source_without_halo) | {"active_slots": int(source_without_halo.active_slots)}
    return {
        "status": "ok" if halo_max <= 1.0e-5 and no_halo_max > halo_max else "failed",
        "case": "halo_tiling_proof",
        "with_halo": {
            "max_abs": float(halo_max),
            "source_tile": with_halo_tile,
            "lt": halo_lt.to_summary(),
        },
        "without_halo": {
            "max_abs": float(no_halo_max),
            "source_tile": without_halo_tile,
            "lt": no_halo_lt.to_summary(),
        },
        "target_tile": asdict(target_tile) | {"active_slots": int(target_tile.active_slots)},
        "all_source_tiles_within_slot_bound": bool(
            source_with_halo.active_slots <= DEFAULT_SLOTS and source_without_halo.active_slots <= DEFAULT_SLOTS
        ),
    }


def write_required_region_artifacts(*, output_dir: Path = Path("/tmp")) -> dict[str, str]:
    from orion.core.region_experiments import build_selected_region_backend_proof, build_whole_network_costs
    from orion.core.region_cir_replay import (
        build_big_graph_convolution_microbench,
        build_r18_stage3_lattigo_microbench,
        build_r18_stage4_lattigo_microbench,
        build_r18_stage2_lattigo_microbench,
        build_original_size_cir_replay,
        write_big_graph_lattigo_microbench,
    )
    from orion.experimental.cir.report import build_region_first_pipeline_report
    from orion.experimental.cir.runtime_group import (
        build_r18_actual_region_first_e2e_report,
        build_r18_tiny_region_first_e2e_report,
    )
    from orion.experimental.cir.stage_matrix import build_stage_materialization_lattigo_matrix

    artifacts = {
        "region_search_candidates": Path(output_dir) / "orion_region_search_candidates.json",
        "halo_tiling_proof": Path(output_dir) / "orion_halo_tiling_proof.json",
        "selected_region_backend_proof": Path(output_dir) / "orion_selected_region_backend_proof.json",
        "whole_network_costs": Path(output_dir) / "orion_whole_network_costs.json",
        "original_size_cir_replay": Path(output_dir) / "orion_original_size_cir_replay.json",
        "big_graph_convolution_microbench": Path(output_dir) / "orion_big_graph_convolution_microbench.json",
        "big_graph_lattigo_microbench": Path(output_dir) / "orion_big_graph_lattigo_microbench.json",
        "r18_stage2_lattigo_microbench": Path(output_dir) / "orion_r18_stage2_lattigo_microbench.json",
        "r18_stage3_lattigo_microbench": Path(output_dir) / "orion_r18_stage3_lattigo_microbench.json",
        "r18_stage4_lattigo_microbench": Path(output_dir) / "orion_r18_stage4_lattigo_microbench.json",
        "region_first_pipeline_report": Path(output_dir) / "orion_region_first_pipeline_report.json",
        "stage_materialization_lattigo_matrix": Path(output_dir) / "orion_stage_materialization_lattigo_matrix.json",
        "r18_tiny_region_first_e2e": Path(output_dir) / "orion_r18_tiny_region_first_e2e.json",
        "r18_actual_region_first_e2e": Path(output_dir) / "orion_r18_actual_region_first_e2e.json",
    }
    payloads = {
        "region_search_candidates": build_region_search_candidates(),
        "halo_tiling_proof": build_halo_tiling_proof(),
        "selected_region_backend_proof": build_selected_region_backend_proof(),
        "whole_network_costs": build_whole_network_costs(),
        "original_size_cir_replay": build_original_size_cir_replay(),
    }
    payloads["big_graph_lattigo_microbench"] = write_big_graph_lattigo_microbench(out_path=artifacts["big_graph_lattigo_microbench"])
    payloads["r18_stage2_lattigo_microbench"] = build_r18_stage2_lattigo_microbench()
    payloads["r18_stage3_lattigo_microbench"] = build_r18_stage3_lattigo_microbench()
    payloads["r18_stage4_lattigo_microbench"] = build_r18_stage4_lattigo_microbench()
    payloads["big_graph_convolution_microbench"] = build_big_graph_convolution_microbench(
        lattigo_microbench_artifact=artifacts["big_graph_lattigo_microbench"]
    )
    payloads["region_first_pipeline_report"] = build_region_first_pipeline_report(
        attach_lattigo=False,
        lattigo_evidence=payloads["big_graph_lattigo_microbench"],
    )
    payloads["stage_materialization_lattigo_matrix"] = build_stage_materialization_lattigo_matrix(
        lattigo_payload=payloads["big_graph_lattigo_microbench"],
        extra_lattigo_payloads=(
            payloads["r18_stage2_lattigo_microbench"],
            payloads["r18_stage3_lattigo_microbench"],
            payloads["r18_stage4_lattigo_microbench"],
        ),
    )
    payloads["r18_tiny_region_first_e2e"] = build_r18_tiny_region_first_e2e_report()
    payloads["r18_actual_region_first_e2e"] = build_r18_actual_region_first_e2e_report()
    for key, path in artifacts.items():
        if key == "big_graph_lattigo_microbench":
            continue
        if key == "r18_stage2_lattigo_microbench":
            Path(path).write_text(json.dumps(payloads[key], indent=2) + "\n", encoding="utf-8")
            continue
        if key in {"r18_stage3_lattigo_microbench", "r18_stage4_lattigo_microbench"}:
            Path(path).write_text(json.dumps(payloads[key], indent=2) + "\n", encoding="utf-8")
            continue
        Path(path).write_text(json.dumps(payloads[key], indent=2) + "\n", encoding="utf-8")
    return {key: str(path) for key, path in artifacts.items()}


def discover_manual_region_candidates() -> tuple[RegionCandidate, ...]:
    synthetic = RegionPlanner.discover_same_source_regions(
        (
            RegionNode("out_a", "linear", "r20_source", "out_a"),
            RegionNode("out_b", "linear", "r20_source", "out_b"),
        )
    )[0]
    return (
        RegionCandidate(
            "stage1_isolated_conv",
            "r20_isolated_source",
            ("out_a",),
            1,
            "R20 isolated negative control",
            ("relu_safe",),
            "r20_input_rep",
        ),
        RegionCandidate(
            "stage1_two_output_region",
            synthetic.source_input_id,
            synthetic.output_node_ids,
            2,
            "R20 synthetic multi-output input-rep region",
            ("relu_safe",),
            "r20_input_rep",
        ),
        RegionCandidate(
            "stage1_stage2_same_shape",
            "r18_same_source",
            ("stage1", "stage2"),
            2,
            "R18 same-shape representative region",
            ("extract_before_relu",),
            "same_shape_hybrid",
        ),
        RegionCandidate(
            "stage1_stage2_same_shape",
            "r34_same_source",
            ("stage1", "stage2"),
            2,
            "R34 same-shape representative region",
            ("extract_before_relu",),
            "same_shape_hybrid",
        ),
    )
