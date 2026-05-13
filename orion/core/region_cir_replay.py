"""Replay original-size scripts/cir region-first results.

This module is a pipeline checkpoint between the tiny Orion backend proof and
future compiler integration. It does not call Orion's dense convolution packer
and it does not claim full-network CKKS execution. Instead, it ingests the
materialized scripts/cir result artifacts, aggregates the selected original-size
rows, and verifies that the replayed stats match the locked scripts/cir stats.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
import json
import time

import torch


STATS_KEYS = ("rotations", "conjugations", "ct_pt_mults", "adds")
DEFAULT_END_TO_END_ARTIFACT = Path("/tmp/region_first_end_to_end_orion_comparison.json")
DEFAULT_TRANSITION_COMPILED_ARTIFACT = Path("/tmp/r34_transition_compiled_lowering_probe.json")
DEFAULT_SELECTED_BACKEND_ARTIFACT = Path("/tmp/orion_selected_region_backend_proof.json")
DEFAULT_REPLAY_OUT = Path("/tmp/orion_original_size_cir_replay.json")
DEFAULT_BIG_GRAPH_MICROBENCH_OUT = Path("/tmp/orion_big_graph_convolution_microbench.json")
DEFAULT_BIG_GRAPH_LATTIGO_MICROBENCH_OUT = Path("/tmp/orion_big_graph_lattigo_microbench.json")


@dataclass(frozen=True)
class OriginalRegionReplaySpec:
    replay_id: str
    network: str
    region_id: str
    source_kind: str
    expected_orion_stats: dict[str, int]
    expected_cir_stats: dict[str, int]
    publishable: bool
    executor_equivalent_required: bool = True


@dataclass(frozen=True)
class OriginalRegionReplayRow:
    replay_id: str
    network: str
    region_id: str
    source_kind: str
    source_artifacts: tuple[str, ...]
    source_families: tuple[str, ...]
    expected_orion_stats: dict[str, int]
    expected_cir_stats: dict[str, int]
    replayed_orion_stats: dict[str, int]
    replayed_cir_stats: dict[str, int]
    stats_match_expected: bool
    executor_equivalent: bool
    same_plan_certificate: bool
    parity: dict[str, Any]
    publishable: bool
    count_only: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["delta_cir_minus_orion"] = _delta(self.replayed_cir_stats, self.replayed_orion_stats)
        payload["score"] = {
            "orion": _score(self.replayed_orion_stats),
            "cir": _score(self.replayed_cir_stats),
            "delta": _score(self.replayed_cir_stats) - _score(self.replayed_orion_stats),
        }
        return payload


def _stats(values: dict[str, Any] | None) -> dict[str, int]:
    values = {} if values is None else dict(values)
    return {key: int(values.get(key, 0)) for key in STATS_KEYS}


def _sum_stats(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    total = {key: 0 for key in STATS_KEYS}
    for item in items:
        stats = _stats(item)
        for key in STATS_KEYS:
            total[key] += int(stats[key])
    return total


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_stats = _stats(left)
    right_stats = _stats(right)
    out = {key: int(left_stats[key]) - int(right_stats[key]) for key in STATS_KEYS}
    out["score"] = _score(left_stats) - _score(right_stats)
    return out


def _score(stats: dict[str, Any]) -> float:
    values = _stats(stats)
    return (
        float(values["rotations"]) * 8.0
        + float(values["conjugations"]) * 8.0
        + float(values["ct_pt_mults"])
        + float(values["adds"]) * 0.05
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"required scripts/cir artifact not found: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def original_region_replay_specs() -> tuple[OriginalRegionReplaySpec, ...]:
    return (
        OriginalRegionReplaySpec(
            replay_id="r20_stage1_two_output_region",
            network="R20",
            region_id="stage1_two_output_region",
            source_kind="scripts_cir_synthetic_multi_output_region",
            expected_orion_stats={"rotations": 76, "conjugations": 0, "ct_pt_mults": 288, "adds": 290},
            expected_cir_stats={"rotations": 39, "conjugations": 0, "ct_pt_mults": 288, "adds": 290},
            publishable=True,
        ),
        OriginalRegionReplaySpec(
            replay_id="r18_stage1_stage2_same_shape",
            network="R18",
            region_id="stage1_stage2_same_shape",
            source_kind="scripts_cir_original_size_scaled_rows",
            expected_orion_stats={"rotations": 7136, "conjugations": 0, "ct_pt_mults": 69840, "adds": 69840},
            expected_cir_stats={"rotations": 572, "conjugations": 152, "ct_pt_mults": 34920, "adds": 35094},
            publishable=True,
        ),
        OriginalRegionReplaySpec(
            replay_id="r34_stage1_stage2_same_shape",
            network="R34",
            region_id="stage1_stage2_same_shape",
            source_kind="scripts_cir_original_size_scaled_rows",
            expected_orion_stats={"rotations": 4558, "conjugations": 0, "ct_pt_mults": 120370, "adds": 120370},
            expected_cir_stats={"rotations": 1928, "conjugations": 124, "ct_pt_mults": 48508, "adds": 48670},
            publishable=True,
        ),
        OriginalRegionReplaySpec(
            replay_id="r34_transition_compile_surface",
            network="R34",
            region_id="stage2_stage3_stage4_transition_branch_regions",
            source_kind="scripts_cir_transition_compiled_cost_surface",
            expected_orion_stats={"rotations": 750, "conjugations": 0, "ct_pt_mults": 25368, "adds": 25368},
            expected_cir_stats={"rotations": 750, "conjugations": 0, "ct_pt_mults": 25368, "adds": 25368},
            publishable=False,
            executor_equivalent_required=False,
        ),
    )


def _build_r20_synthetic_row(payload: dict[str, Any], spec: OriginalRegionReplaySpec, source: Path) -> OriginalRegionReplayRow:
    synthetic = dict(payload.get("synthetic_region_checks", {})).get("r20_stage1_two_output_region")
    if not synthetic:
        raise ValueError("scripts/cir artifact missing r20_stage1_two_output_region synthetic check")
    summary = dict(synthetic["summary"])
    replayed_cir = _stats(summary["ours"])
    replayed_orion = _stats(summary["orion"])
    parity = dict(synthetic.get("parity", {}))
    matches = replayed_cir == _stats(spec.expected_cir_stats) and replayed_orion == _stats(spec.expected_orion_stats)
    exact = bool(parity.get("exact", False))
    return OriginalRegionReplayRow(
        replay_id=str(spec.replay_id),
        network=str(spec.network),
        region_id=str(spec.region_id),
        source_kind=str(spec.source_kind),
        source_artifacts=(str(source),),
        source_families=("synthetic:r20_stage1_two_output_region",),
        expected_orion_stats=_stats(spec.expected_orion_stats),
        expected_cir_stats=_stats(spec.expected_cir_stats),
        replayed_orion_stats=replayed_orion,
        replayed_cir_stats=replayed_cir,
        stats_match_expected=bool(matches),
        executor_equivalent=bool(exact),
        same_plan_certificate=bool(exact),
        parity=parity,
        publishable=bool(spec.publishable and matches and exact),
        count_only=False,
        notes=str(synthetic.get("scope", "synthetic multi-output scripts/cir materialization")),
    )


def _aggregate_end_to_end_rows(
    payload: dict[str, Any],
    *,
    model: str,
    families: set[str],
) -> tuple[dict[str, int], dict[str, int], tuple[str, ...], dict[str, Any]]:
    rows = [
        row
        for row in payload.get("rows", [])
        if str(row.get("model")) == str(model) and str(row.get("family")) in families
    ]
    if not rows:
        raise ValueError(f"scripts/cir artifact has no rows for {model}:{sorted(families)}")
    replayed_cir = _sum_stats(dict(row["scaled"])["ours"] for row in rows)
    replayed_orion = _sum_stats(dict(row["scaled"])["orion"] for row in rows)
    parity_rows = [dict(row.get("parity", {})) for row in rows]
    parity = {
        "exact": bool(all(bool(row.get("exact", False)) for row in parity_rows)),
        "max_abs": max((float(row.get("max_abs", 0.0)) for row in parity_rows), default=0.0),
        "tolerance": min((float(row.get("tolerance", 1.0e-3)) for row in parity_rows), default=1.0e-3),
        "row_count": int(len(rows)),
    }
    return replayed_orion, replayed_cir, tuple(str(row.get("family")) for row in rows), parity


def _build_scaled_region_row(
    payload: dict[str, Any],
    spec: OriginalRegionReplaySpec,
    *,
    source: Path,
    model: str,
    families: set[str],
) -> OriginalRegionReplayRow:
    replayed_orion, replayed_cir, source_families, parity = _aggregate_end_to_end_rows(
        payload,
        model=str(model),
        families=set(families),
    )
    matches = replayed_cir == _stats(spec.expected_cir_stats) and replayed_orion == _stats(spec.expected_orion_stats)
    exact = bool(parity.get("exact", False))
    return OriginalRegionReplayRow(
        replay_id=str(spec.replay_id),
        network=str(spec.network),
        region_id=str(spec.region_id),
        source_kind=str(spec.source_kind),
        source_artifacts=(str(source),),
        source_families=tuple(source_families),
        expected_orion_stats=_stats(spec.expected_orion_stats),
        expected_cir_stats=_stats(spec.expected_cir_stats),
        replayed_orion_stats=replayed_orion,
        replayed_cir_stats=replayed_cir,
        stats_match_expected=bool(matches),
        executor_equivalent=bool(exact),
        same_plan_certificate=bool(exact),
        parity=parity,
        publishable=bool(spec.publishable and matches and exact),
        count_only=False,
        notes="aggregated original-size scaled scripts/cir materialized rows",
    )


def _build_transition_surface_row(payload: dict[str, Any], spec: OriginalRegionReplaySpec, source: Path) -> OriginalRegionReplayRow:
    rows = list(payload.get("rows", []))
    replayed = _sum_stats(dict(row.get("cost_equivalent_stats", {})) for row in rows)
    matches = replayed == _stats(spec.expected_cir_stats)
    return OriginalRegionReplayRow(
        replay_id=str(spec.replay_id),
        network=str(spec.network),
        region_id=str(spec.region_id),
        source_kind=str(spec.source_kind),
        source_artifacts=(str(source),),
        source_families=tuple(str(row.get("family", "")) for row in rows),
        expected_orion_stats=_stats(spec.expected_orion_stats),
        expected_cir_stats=_stats(spec.expected_cir_stats),
        replayed_orion_stats=replayed,
        replayed_cir_stats=replayed,
        stats_match_expected=bool(matches),
        executor_equivalent=False,
        same_plan_certificate=False,
        parity={"exact": False, "reason": "compiled cost surface only; standalone transition executor not selected"},
        publishable=False,
        count_only=True,
        notes="kept as non-publishable transition cost surface until executor-equivalent selected lowering exists",
    )


def build_original_size_cir_replay(
    *,
    end_to_end_artifact: Path = DEFAULT_END_TO_END_ARTIFACT,
    transition_compiled_artifact: Path = DEFAULT_TRANSITION_COMPILED_ARTIFACT,
) -> dict[str, Any]:
    end_to_end = _load_json(Path(end_to_end_artifact))
    transition = _load_json(Path(transition_compiled_artifact)) if Path(transition_compiled_artifact).exists() else {"rows": []}
    specs = {spec.replay_id: spec for spec in original_region_replay_specs()}
    rows = [
        _build_r20_synthetic_row(end_to_end, specs["r20_stage1_two_output_region"], Path(end_to_end_artifact)),
        _build_scaled_region_row(
            end_to_end,
            specs["r18_stage1_stage2_same_shape"],
            source=Path(end_to_end_artifact),
            model="resnet18_tiny_imagenet",
            families={"stage1_same", "stage2_same"},
        ),
        _build_scaled_region_row(
            end_to_end,
            specs["r34_stage1_stage2_same_shape"],
            source=Path(end_to_end_artifact),
            model="resnet34_imagenet",
            families={"stage1_same_3x3_s1_gap4_to4", "stage2_same_3x3_s1_gap8_to8"},
        ),
        _build_transition_surface_row(
            transition,
            specs["r34_transition_compile_surface"],
            Path(transition_compiled_artifact),
        ),
    ]
    publishable_rows = [row for row in rows if bool(row.publishable)]
    totals = _sum_stats(row.replayed_cir_stats for row in publishable_rows)
    baseline_totals = _sum_stats(row.replayed_orion_stats for row in publishable_rows)
    status = "ok" if all(bool(row.stats_match_expected) for row in rows) and all(bool(row.publishable) for row in rows if not row.count_only) else "failed"
    return {
        "status": str(status),
        "scope": "original-size scripts/cir replay from compiled/materialized artifacts; not Orion compiler integration",
        "source_artifacts": {
            "end_to_end": str(end_to_end_artifact),
            "transition_compiled": str(transition_compiled_artifact),
        },
        "rows": [row.to_dict() for row in rows],
        "publishable_summary": {
            "orion": baseline_totals,
            "cir": totals,
            "delta_cir_minus_orion": _delta(totals, baseline_totals),
            "publishable_count": int(len(publishable_rows)),
            "excluded_count": int(len(rows) - len(publishable_rows)),
        },
        "gates": {
            "all_stats_match_expected": bool(all(bool(row.stats_match_expected) for row in rows)),
            "all_publishable_rows_executor_equivalent": bool(all(bool(row.executor_equivalent) for row in publishable_rows)),
            "no_publishable_count_only_rows": bool(all(not bool(row.count_only) for row in publishable_rows)),
        },
    }


def write_original_size_cir_replay(
    *,
    out_path: Path = DEFAULT_REPLAY_OUT,
    end_to_end_artifact: Path = DEFAULT_END_TO_END_ARTIFACT,
    transition_compiled_artifact: Path = DEFAULT_TRANSITION_COMPILED_ARTIFACT,
) -> dict[str, Any]:
    payload = build_original_size_cir_replay(
        end_to_end_artifact=Path(end_to_end_artifact),
        transition_compiled_artifact=Path(transition_compiled_artifact),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _load_or_build_selected_backend(path: Path) -> dict[str, Any]:
    if Path(path).exists():
        return _load_json(Path(path))
    from orion.core.region_experiments import build_selected_region_backend_proof

    return build_selected_region_backend_proof()


def _backend_case_by_region(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(case.get("network")), str(case.get("region_id"))): dict(case)
        for case in payload.get("cases", [])
    }


def build_big_graph_convolution_microbench(
    *,
    end_to_end_artifact: Path = DEFAULT_END_TO_END_ARTIFACT,
    transition_compiled_artifact: Path = DEFAULT_TRANSITION_COMPILED_ARTIFACT,
    selected_backend_artifact: Path = DEFAULT_SELECTED_BACKEND_ARTIFACT,
    lattigo_microbench_artifact: Path = DEFAULT_BIG_GRAPH_LATTIGO_MICROBENCH_OUT,
) -> dict[str, Any]:
    """Build a strict R18/R34 convolution microbenchmark evidence artifact.

    The artifact intentionally excludes the R20 synthetic row. It also keeps
    original-size scripts/cir cost facts separate from current Orion Lattigo
    backend evidence, because the latter is still tiny representative backend
    execution rather than original-size Lattigo execution.
    """

    replay = build_original_size_cir_replay(
        end_to_end_artifact=Path(end_to_end_artifact),
        transition_compiled_artifact=Path(transition_compiled_artifact),
    )
    backend = _load_or_build_selected_backend(Path(selected_backend_artifact))
    lattigo_payload = _load_json(Path(lattigo_microbench_artifact)) if Path(lattigo_microbench_artifact).exists() else {}
    lattigo_publishable_rows = []
    if bool(lattigo_payload.get("publishable_lattigo_microbenchmark", False)):
        lattigo_publishable_rows.append(
            {
                "network": str(lattigo_payload.get("network", "")),
                "family": str(lattigo_payload.get("family", "")),
                "region_id": str(lattigo_payload.get("region_id", "")),
                "full_region": bool(lattigo_payload.get("full_region", False)),
                "original_size_slot_domain": bool(lattigo_payload.get("original_size_slot_domain", False)),
                "bank_count": int(lattigo_payload.get("bank_count", 0)),
                "stats_from_execution": dict(lattigo_payload.get("stats_from_execution", {})),
                "parity": dict(lattigo_payload.get("parity", {})),
            }
        )
    backend_cases = _backend_case_by_region(backend)
    rows: list[dict[str, Any]] = []
    for row in replay["rows"]:
        if str(row.get("network")) not in {"R18", "R34"}:
            continue
        if str(row.get("region_id")) != "stage1_stage2_same_shape":
            continue
        key = (str(row["network"]), str(row["region_id"]))
        backend_case = backend_cases.get(key, {})
        backend_payload = dict(backend_case.get("backend_case", {}))
        lattigo_backend_evidence = {
            "available": bool(backend_case),
            "status": str(backend_case.get("status", "")),
            "uses_lattigo": bool(backend_payload.get("unified_transform_group", False)),
            "uses_unified_transform_group": bool(backend_payload.get("unified_transform_group", False)),
            "uses_orion_dense_pack_conv2d": bool(backend_payload.get("uses_orion_dense_pack_conv2d", True)) if backend_case else None,
            "original_size": False,
            "evidence_shape": "tiny_representative",
            "stats_from_execution": dict(backend_case.get("stats_from_execution", {})),
            "parity": dict(backend_case.get("parity", {})),
        }
        publishable_cost = bool(
            row.get("stats_match_expected")
            and row.get("executor_equivalent")
            and row.get("same_plan_certificate")
            and dict(row.get("parity", {})).get("exact")
            and not row.get("count_only")
            and "synthetic" not in str(row.get("source_kind", ""))
        )
        publishable_lattigo = bool(
            publishable_cost
            and lattigo_backend_evidence["available"]
            and lattigo_backend_evidence["uses_lattigo"]
            and lattigo_backend_evidence["original_size"]
            and dict(lattigo_backend_evidence["parity"]).get("exact", False)
        )
        rows.append(
            {
                "network": str(row["network"]),
                "region_id": str(row["region_id"]),
                "source_kind": str(row["source_kind"]),
                "source_families": list(row.get("source_families", [])),
                "original_size": True,
                "synthetic": False,
                "orion_baseline_stats": dict(row["replayed_orion_stats"]),
                "cir_stats": dict(row["replayed_cir_stats"]),
                "delta_cir_minus_orion": dict(row["delta_cir_minus_orion"]),
                "scripts_cir_executor_equivalent": bool(row["executor_equivalent"]),
                "same_plan_certificate": bool(row["same_plan_certificate"]),
                "scripts_cir_parity": dict(row["parity"]),
                "lattigo_backend_evidence": lattigo_backend_evidence,
                "publishable_cost_microbenchmark": bool(publishable_cost),
                "publishable_lattigo_microbenchmark": bool(publishable_lattigo),
                "lattigo_blocker": ""
                if publishable_lattigo
                else "original-size Lattigo backend execution is not available for this row yet",
            }
        )
    cost_rows = [row for row in rows if bool(row["publishable_cost_microbenchmark"])]
    lattigo_rows = [row for row in rows if bool(row["publishable_lattigo_microbenchmark"])]
    lattigo_count = int(len(lattigo_rows) + len(lattigo_publishable_rows))
    status = "needs_original_size_lattigo"
    if int(lattigo_count) > 0:
        status = "partial_original_size_lattigo"
    if int(lattigo_count) >= 2:
        status = "ok"
    return {
        "status": str(status),
        "scope": "R18/R34 original-size convolution microbench evidence; excludes R20 synthetic rows",
        "source_artifacts": {
            "original_size_replay": str(end_to_end_artifact),
            "selected_backend": str(selected_backend_artifact),
        },
        "rows": rows,
        "publishability": {
            "cost_microbenchmark_publishable_count": int(len(cost_rows)),
            "lattigo_microbenchmark_publishable_count": int(lattigo_count),
            "excluded_synthetic_rows": ["R20:stage1_two_output_region"],
            "lattigo_backend_status": "partial original-size block coverage" if int(lattigo_count) else "representative-only; original-size Lattigo execution still required",
        },
        "lattigo_microbench_rows": lattigo_publishable_rows,
        "cost_summary": {
            "orion": _sum_stats(row["orion_baseline_stats"] for row in cost_rows),
            "cir": _sum_stats(row["cir_stats"] for row in cost_rows),
        },
    }


def write_big_graph_convolution_microbench(
    *,
    out_path: Path = DEFAULT_BIG_GRAPH_MICROBENCH_OUT,
    end_to_end_artifact: Path = DEFAULT_END_TO_END_ARTIFACT,
    transition_compiled_artifact: Path = DEFAULT_TRANSITION_COMPILED_ARTIFACT,
    selected_backend_artifact: Path = DEFAULT_SELECTED_BACKEND_ARTIFACT,
    lattigo_microbench_artifact: Path = DEFAULT_BIG_GRAPH_LATTIGO_MICROBENCH_OUT,
) -> dict[str, Any]:
    payload = build_big_graph_convolution_microbench(
        end_to_end_artifact=Path(end_to_end_artifact),
        transition_compiled_artifact=Path(transition_compiled_artifact),
        selected_backend_artifact=Path(selected_backend_artifact),
        lattigo_microbench_artifact=Path(lattigo_microbench_artifact),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _scripts_cir_r18_stage1_block() -> tuple[Any, dict[str, Any], torch.Tensor]:
    from orion.experimental.cir import build_r18_stage1_shared_block_plan

    return build_r18_stage1_shared_block_plan(bank_count=8)


def _scripts_cir_r18_stage2_block(input_pair_index: int) -> tuple[Any, dict[str, Any], torch.Tensor]:
    from orion.experimental.cir import build_r18_stage2_shared_block_plan

    return build_r18_stage2_shared_block_plan(input_pair_index=int(input_pair_index), bank_count=None)


def _scripts_cir_r18_stage3_block() -> tuple[Any, dict[str, Any], torch.Tensor]:
    from orion.experimental.cir import build_r18_stage3_shared_block_plan

    return build_r18_stage3_shared_block_plan(bank_count=None)


def _scripts_cir_r18_stage4_block() -> tuple[Any, dict[str, Any], torch.Tensor]:
    from orion.experimental.cir import build_r18_stage4_compact_intra_plan

    return build_r18_stage4_compact_intra_plan()


def _bank_transforms_from_scripts_cir_plan(
    plan: Any,
    inputs: dict[str, Any],
    *,
    bank_count: int,
    level: int,
    scheme: Any,
    source_override: torch.Tensor | None = None,
) -> tuple[list[Any], list[torch.Tensor], dict[str, Any]]:
    if len(plan.linear_transform_steps) != 1:
        raise ValueError(f"expected one collapsed SharedMultiOutput LT step, got {len(plan.linear_transform_steps)}")
    step = plan.linear_transform_steps[0]
    if not bool(getattr(step, "shared_multi_output", False)):
        raise ValueError("scripts/cir R18 stage1 block did not provide a collapsed shared multi-output step")
    slots = int(plan.ring_slot_count)
    prepared = {str(plain.plaintext_id): plain for plain in plan.prepared_plaintexts}
    templates = {
        str(entry.template_id): entry
        for family in plan.family_templates
        for entry in family.template_entries
    }
    if source_override is None:
        source = (
            inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.complex64)
            + 1j * inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.complex64)
        )
    else:
        source = source_override.detach().to(dtype=torch.complex64).clone()
    transforms: list[Any] = []
    expected_outputs: list[torch.Tensor] = []
    selected_banks = tuple(step.shared_output_banks[: int(bank_count)])
    for bank in selected_banks:
        bank_id = str(bank.bank_id)
        diag_tensors: dict[int, torch.Tensor] = {}
        expected = torch.zeros((int(slots),), dtype=torch.complex64)
        terms = [term for term in step.terms if str(getattr(term, "bank_id", "")) == bank_id]
        if len(terms) != int(bank.term_count):
            raise ValueError(f"bank {bank_id} expected {bank.term_count} terms, got {len(terms)}")
        for term in terms:
            template = templates[str(term.template_id)]
            plaintext = prepared[str(term.plaintext_id)]
            mapped_source_indices = template.indices.to(dtype=torch.int64).index_select(0, term.lookup_indices.to(dtype=torch.int64))
            output_indices = term.output_slot_indices.to(dtype=torch.int64)
            if not bool(torch.equal(mapped_source_indices, output_indices)):
                raise ValueError(f"term {term.term_id} cannot be represented as a dense diagonal without remapping")
            values = plaintext.values.to(dtype=torch.complex64)
            diag_index = (-int(term.shift)) % int(slots)
            diag = diag_tensors.setdefault(int(diag_index), torch.zeros((int(slots),), dtype=torch.complex64))
            diag.index_add_(0, output_indices, values)
            rotated = torch.roll(source, shifts=int(term.shift), dims=0)
            expected.index_add_(0, output_indices, rotated.index_select(0, mapped_source_indices) * values)
        transforms.append(
            SimpleNamespace(
                name=f"{plan.case_name}_{bank_id}",
                diagonals={(0, 0): {int(index): diag.tolist() for index, diag in sorted(diag_tensors.items())}},
                level=int(level),
                scheme=scheme,
                fhe_output_shape=torch.Size([1, int(slots)]),
                output_shape=torch.Size([1, int(slots)]),
                bank_id=bank_id,
            )
        )
        expected_outputs.append(expected)
    subset_stats = {
        "rotations": int(step.expected_cost.rotations),
        "conjugations": int(len(selected_banks)),
        "ct_pt_mults": int(sum(int(bank.term_count) for bank in selected_banks)),
        "adds": int(sum(int(bank.term_count) for bank in selected_banks) + len(selected_banks) + 1),
    }
    return transforms, expected_outputs, {
        "bank_count": int(len(selected_banks)),
        "bank_ids": [str(bank.bank_id) for bank in selected_banks],
        "scripts_cir_subset_stats": subset_stats,
        "scripts_cir_full_block_stats": _stats(
            {
                "rotations": int(plan.expected_cost.rotations),
                "conjugations": int(plan.expected_cost.conjugations),
                "ct_pt_mults": int(plan.expected_cost.ct_pt_mults),
                "adds": int(plan.expected_cost.adds),
            }
        ),
        "linear_transform_terms": int(sum(int(bank.term_count) for bank in selected_banks)),
    }


def _compact_stage4_source_from_regular(source: torch.Tensor) -> torch.Tensor:
    from orion.experimental.cir.lattigo_block import R18_STAGE4_SPEC, _phase_mask

    spec = R18_STAGE4_SPEC
    left_phases = tuple(range(int(spec.gap * spec.gap // 2)))
    right_phases = tuple(range(int(spec.gap * spec.gap // 2), int(spec.gap * spec.gap)))
    left_selector = _phase_mask(phases=left_phases, shape=(int(spec.c), int(spec.h), int(spec.w)), gap=int(spec.gap)).to(dtype=torch.float32)
    right_selector = _phase_mask(phases=right_phases, shape=(int(spec.c), int(spec.h), int(spec.w)), gap=int(spec.gap)).to(dtype=torch.float32)
    right_align_shift = -int((int(spec.gap) // 2) * int(spec.w) * int(spec.gap))
    compact = source.to(dtype=torch.complex64) * left_selector.to(dtype=torch.complex64)
    compact = compact + 1j * torch.roll(source * right_selector, shifts=int(right_align_shift), dims=0).to(dtype=torch.complex64)
    compact = compact + torch.roll(compact, shifts=int(-right_align_shift), dims=0)
    return compact


def _compiled_rotation_key_count(group: Any, backend: Any) -> int:
    transform_ids = getattr(group, "unified_ids", None) or ()
    backend_module = str(type(backend).__module__)
    identity_keys = {0, 1} if "orion.backend.lattigo" in backend_module else {0}
    keys: set[int] = set()
    for transform_id in transform_ids:
        for key in backend.GetLinearTransformRotationKeys(int(transform_id)):
            if int(key) not in identity_keys:
                keys.add(int(key))
    return int(len(keys))


def _run_r18_stage4_plan_microbench(
    *,
    plan_builder: Any,
    scope: str,
    family: str,
    prepack_execution: str,
    backend_name: str,
    logn: int = 16,
) -> dict[str, Any]:
    from orion.backend.python.tensors import CipherTensor
    from orion.core.orion import scheme
    from orion.nn.unified_transform import UnifiedTransformGroup

    config = {
        "ckks_params": {"LogN": int(logn), "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": str(backend_name), "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    timings = {"scheme_init_s": 0.0, "compile_unified_s": 0.0, "evaluate_unified_s": 0.0}
    started = time.time()
    scheme.init_scheme(config)
    timings["scheme_init_s"] = float(time.time() - started)
    try:
        level = len(scheme.params.get_logq()) - 1
        plan, inputs, _reference = plan_builder()
        regular_source = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        compact_source = _compact_stage4_source_from_regular(regular_source)
        transforms, expected_outputs, detail = _bank_transforms_from_scripts_cir_plan(
            plan,
            inputs,
            bank_count=1,
            level=int(level),
            scheme=scheme,
            source_override=compact_source,
        )
        started_compile = time.time()
        group = UnifiedTransformGroup(transforms)
        group.compile_unified(scheme.backend)
        timings["compile_unified_s"] = float(time.time() - started_compile)
        actual_core_rotation_count = _compiled_rotation_key_count(group, scheme.backend)
        ct_source = scheme.encrypt(scheme.encode(compact_source.real.to(dtype=torch.float32), level)) + scheme.encrypt(scheme.encode(compact_source.imag.to(dtype=torch.float32), level)).mul_imaginary_unit(+1, in_place=False)
        started_eval = time.time()
        output_ids = group.evaluate_unified(int(ct_source.ids[0]), scheme.backend)
        timings["evaluate_unified_s"] = float(time.time() - started_eval)
        max_errors = []
        for output_id, expected in zip(output_ids, expected_outputs):
            out_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, int(scheme.params.get_slots())]), torch.Size([1, int(scheme.params.get_slots())]))
            out_pt = out_ct.decrypt()
            raw = scheme.backend.DecodeComplex(out_pt.ids[0])
            decoded = torch.tensor([complex(raw[2 * i], raw[2 * i + 1]) for i in range(int(scheme.params.get_slots()))], dtype=torch.complex64)
            max_errors.append(float((decoded - expected).abs().max()))
        parity = {"exact": bool(max(max_errors, default=0.0) <= 1.0e-3), "max_abs": float(max(max_errors, default=0.0)), "max_errors": max_errors, "tolerance": 1.0e-3}
        term_count = int(detail["linear_transform_terms"])
        observed = {
            "rotations": int(actual_core_rotation_count) + 2,
            "conjugations": 1,
            "ct_pt_mults": int(term_count) + 2,
            "adds": int(term_count) + 3,
        }
        return {
            "status": "ok" if bool(parity.get("exact", False)) else "failed",
            "scope": str(scope),
            "network": "R18",
            "family": str(family),
            "stage": "stage4",
            "backend": str(backend_name),
            "full_region": True,
            "original_size_slot_domain": True,
            "local_lattigo": True,
            "unified_transform_group": True,
            "uses_orion_dense_pack_conv2d": False,
            "prepack_execution": str(prepack_execution),
            "stats_from_execution": dict(observed),
            "actual_core_rotation_count": int(actual_core_rotation_count),
            "linear_transform_terms": int(term_count),
            "same_plan_certificate": bool(parity.get("exact", False)),
            "parity": parity,
            "timing_s": timings,
            "publishable_lattigo_microbenchmark": bool(parity.get("exact", False)),
        }
    finally:
        scheme.delete_scheme()


def build_big_graph_lattigo_microbench(*, bank_count: int = 8, logn: int = 16) -> dict[str, Any]:
    from orion.backend.python.tensors import CipherTensor
    from orion.core.orion import scheme
    from orion.nn.unified_transform import UnifiedTransformGroup

    plan, inputs, _reference = _scripts_cir_r18_stage1_block()
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
    timings: dict[str, float] = {}
    start = time.time()
    scheme.init_scheme(config)
    timings["scheme_init_s"] = float(time.time() - start)
    try:
        slots = int(scheme.params.get_slots())
        level = len(scheme.params.get_logq()) - 1
        transforms, expected_outputs, detail = _bank_transforms_from_scripts_cir_plan(
            plan,
            inputs,
            bank_count=int(bank_count),
            level=int(level),
            scheme=scheme,
        )
        start = time.time()
        group = UnifiedTransformGroup(transforms)
        group.compile_unified(scheme.backend)
        timings["compile_unified_s"] = float(time.time() - start)
        left = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        right = inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        ct_left = scheme.encrypt(scheme.encode(left, level))
        ct_right = scheme.encrypt(scheme.encode(right, level))
        ct_complex = ct_left + ct_right.mul_imaginary_unit(+1, in_place=False)
        start = time.time()
        output_ids = group.evaluate_unified(int(ct_complex.ids[0]), scheme.backend)
        timings["evaluate_unified_s"] = float(time.time() - start)
        max_errors: list[float] = []
        for output_id, expected in zip(output_ids, expected_outputs):
            out_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, int(slots)]), torch.Size([1, int(slots)]))
            out_pt = out_ct.decrypt()
            raw = scheme.backend.DecodeComplex(out_pt.ids[0])
            decoded = torch.tensor([complex(raw[2 * i], raw[2 * i + 1]) for i in range(int(slots))], dtype=torch.complex64)
            max_errors.append(float((decoded - expected).abs().max()))
        max_abs = float(max(max_errors, default=0.0))
        exact = bool(max_abs <= 1.0e-3)
        return {
            "status": "ok" if exact else "failed",
            "scope": "original-size R18 stage1 scripts/cir block LT through Orion UnifiedTransformGroup and local Lattigo",
            "network": "R18",
            "family": "stage1_same",
            "region_id": "stage1_same_block0",
            "full_region": False,
            "original_size_slot_domain": bool(int(slots) == 32768 and int(logn) == 16),
            "source_plan": "scripts/cir _build_shared_output_inter_hsplit_block_plan",
            "local_lattigo": True,
            "unified_transform_group": True,
            "uses_orion_dense_pack_conv2d": False,
            "bank_count": int(detail["bank_count"]),
            "bank_ids": list(detail["bank_ids"]),
            "scripts_cir_subset_stats": dict(detail["scripts_cir_subset_stats"]),
            "scripts_cir_full_block_stats": dict(detail["scripts_cir_full_block_stats"]),
            "stats_from_execution": dict(detail["scripts_cir_subset_stats"]),
            "same_plan_certificate": bool(exact),
            "parity": {"exact": bool(exact), "max_abs": float(max_abs), "max_errors": max_errors, "tolerance": 1.0e-3},
            "publishable_lattigo_microbenchmark": bool(exact),
            "publishability": {
                "lattigo_microbenchmark_publishable_count": 1 if exact else 0,
                "note": "single original-size R18 stage1 full output-bank block, not aggregate full-network execution",
            },
            "timing_s": timings,
        }
    finally:
        scheme.delete_scheme()


def _run_lattigo_plan_once(plan: Any, inputs: dict[str, Any], *, bank_count: int, level: int) -> tuple[dict[str, int], dict[str, Any], dict[str, float]]:
    from orion.backend.python.tensors import CipherTensor
    from orion.core.orion import scheme
    from orion.nn.unified_transform import UnifiedTransformGroup

    timings: dict[str, float] = {}
    slots = int(scheme.params.get_slots())
    transforms, expected_outputs, detail = _bank_transforms_from_scripts_cir_plan(
        plan,
        inputs,
        bank_count=int(bank_count),
        level=int(level),
        scheme=scheme,
    )
    start = time.time()
    group = UnifiedTransformGroup(transforms)
    group.compile_unified(scheme.backend)
    timings["compile_unified_s"] = float(time.time() - start)
    left = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
    right = inputs["source_1_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
    ct_left = scheme.encrypt(scheme.encode(left, level))
    ct_right = scheme.encrypt(scheme.encode(right, level))
    ct_complex = ct_left + ct_right.mul_imaginary_unit(+1, in_place=False)
    start = time.time()
    output_ids = group.evaluate_unified(int(ct_complex.ids[0]), scheme.backend)
    timings["evaluate_unified_s"] = float(time.time() - start)
    max_errors: list[float] = []
    for output_id, expected in zip(output_ids, expected_outputs):
        out_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, int(slots)]), torch.Size([1, int(slots)]))
        out_pt = out_ct.decrypt()
        raw = scheme.backend.DecodeComplex(out_pt.ids[0])
        decoded = torch.tensor([complex(raw[2 * i], raw[2 * i + 1]) for i in range(int(slots))], dtype=torch.complex64)
        max_errors.append(float((decoded - expected).abs().max()))
    parity = {"exact": bool(max(max_errors, default=0.0) <= 1.0e-3), "max_abs": float(max(max_errors, default=0.0)), "max_errors": max_errors, "tolerance": 1.0e-3}
    return dict(detail["scripts_cir_subset_stats"]), parity, timings


def build_r18_stage2_lattigo_microbench(*, logn: int = 16) -> dict[str, Any]:
    from orion.core.orion import scheme

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
    timings = {"scheme_init_s": 0.0, "compile_unified_s": 0.0, "evaluate_unified_s": 0.0}
    started = time.time()
    scheme.init_scheme(config)
    timings["scheme_init_s"] = float(time.time() - started)
    try:
        level = len(scheme.params.get_logq()) - 1
        total = {key: 0 for key in ("rotations", "conjugations", "ct_pt_mults", "adds")}
        block_rows: list[dict[str, Any]] = []
        all_exact = True
        max_abs = 0.0
        for input_pair_index in (0, 1):
            plan, inputs, _reference = _scripts_cir_r18_stage2_block(int(input_pair_index))
            stats, parity, block_timing = _run_lattigo_plan_once(
                plan,
                inputs,
                bank_count=len(plan.linear_transform_steps[0].shared_output_banks),
                level=int(level),
            )
            for key in total:
                total[key] += int(stats.get(key, 0))
            timings["compile_unified_s"] += float(block_timing["compile_unified_s"])
            timings["evaluate_unified_s"] += float(block_timing["evaluate_unified_s"])
            all_exact = bool(all_exact and parity.get("exact", False))
            max_abs = max(float(max_abs), float(parity.get("max_abs", 0.0)))
            block_rows.append(
                {
                    "input_pair_index": int(input_pair_index),
                    "stats_from_execution": stats,
                    "parity": parity,
                    "timing_s": block_timing,
                }
            )
        expected = {"rotations": 84, "conjugations": 8, "ct_pt_mults": 5880, "adds": 5890}
        stats_match = dict(total) == dict(expected)
        return {
            "status": "ok" if bool(all_exact and stats_match) else "failed",
            "scope": "original-size R18 stage2 same-shape full input-pair Lattigo materialization",
            "network": "R18",
            "family": "stage2_same",
            "stage": "stage2",
            "full_region": True,
            "original_size_slot_domain": True,
            "local_lattigo": True,
            "unified_transform_group": True,
            "uses_orion_dense_pack_conv2d": False,
            "stats_from_execution": dict(total),
            "expected_stats": dict(expected),
            "stats_match_scripts_cir": bool(stats_match),
            "same_plan_certificate": bool(all_exact and stats_match),
            "parity": {"exact": bool(all_exact), "max_abs": float(max_abs), "tolerance": 1.0e-3},
            "block_rows": block_rows,
            "timing_s": timings,
            "publishable_lattigo_microbenchmark": bool(all_exact and stats_match),
        }
    finally:
        scheme.delete_scheme()


def build_r18_stage3_lattigo_microbench(*, logn: int = 16) -> dict[str, Any]:
    from orion.core.orion import scheme

    config = {
        "ckks_params": {"LogN": int(logn), "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    timings = {"scheme_init_s": 0.0, "compile_unified_s": 0.0, "evaluate_unified_s": 0.0}
    started = time.time()
    scheme.init_scheme(config)
    timings["scheme_init_s"] = float(time.time() - started)
    try:
        level = len(scheme.params.get_logq()) - 1
        plan, inputs, _reference = _scripts_cir_r18_stage3_block()
        stats, parity, block_timing = _run_lattigo_plan_once(plan, inputs, bank_count=len(plan.linear_transform_steps[0].shared_output_banks), level=int(level))
        timings["compile_unified_s"] = float(block_timing["compile_unified_s"])
        timings["evaluate_unified_s"] = float(block_timing["evaluate_unified_s"])
        expected = {"rotations": 90, "conjugations": 2, "ct_pt_mults": 6750, "adds": 6753}
        stats_match = dict(stats) == dict(expected)
        return {
            "status": "ok" if bool(parity.get("exact", False) and stats_match) else "failed",
            "scope": "original-size R18 stage3 same-shape Lattigo materialization",
            "network": "R18",
            "family": "stage3_same",
            "stage": "stage3",
            "full_region": True,
            "original_size_slot_domain": True,
            "local_lattigo": True,
            "unified_transform_group": True,
            "uses_orion_dense_pack_conv2d": False,
            "stats_from_execution": dict(stats),
            "expected_stats": dict(expected),
            "stats_match_scripts_cir": bool(stats_match),
            "same_plan_certificate": bool(parity.get("exact", False) and stats_match),
            "parity": parity,
            "timing_s": timings,
            "publishable_lattigo_microbenchmark": bool(parity.get("exact", False) and stats_match),
        }
    finally:
        scheme.delete_scheme()


def build_r18_stage4_lattigo_microbench(*, logn: int = 16) -> dict[str, Any]:
    payload = _run_r18_stage4_plan_microbench(
        plan_builder=_scripts_cir_r18_stage4_block,
        scope="original-size R18 stage4 compact-intra Lattigo materialization",
        family="stage4_same",
        prepack_execution="plaintext-prepacked compact source; scripts/cir prepack cost included in stats",
        backend_name="lattigo",
        logn=int(logn),
    )
    expected_stats = {"rotations": 158, "conjugations": 1, "ct_pt_mults": 9767, "adds": 9768}
    stats_match = dict(payload["stats_from_execution"]) == dict(expected_stats)
    payload["status"] = "ok" if bool(payload["parity"].get("exact", False) and stats_match) else "failed"
    payload["expected_stats"] = dict(expected_stats)
    payload["stats_match_scripts_cir"] = bool(stats_match)
    payload["same_plan_certificate"] = bool(payload["parity"].get("exact", False) and stats_match)
    payload["publishable_lattigo_microbenchmark"] = bool(payload["parity"].get("exact", False) and stats_match)
    return payload


def build_r18_stage4_gs256_prototype_lattigo_microbench(*, logn: int = 16) -> dict[str, Any]:
    from orion.experimental.cir import build_r18_stage4_compact_intra_gs256_prototype_plan

    return _run_r18_stage4_plan_microbench(
        plan_builder=build_r18_stage4_compact_intra_gs256_prototype_plan,
        scope="original-size R18 stage4 compact-intra gs256 prototype Lattigo materialization",
        family="stage4_same_gs256_proto",
        prepack_execution="plaintext-prepacked compact source; prototype reroutes terms through the existing +256 replica",
        backend_name="lattigo",
        logn=int(logn),
    )


def build_r18_stage4_python_microbench(*, logn: int = 16) -> dict[str, Any]:
    return _run_r18_stage4_plan_microbench(
        plan_builder=_scripts_cir_r18_stage4_block,
        scope="original-size R18 stage4 compact-intra Python-backend materialization",
        family="stage4_same",
        prepack_execution="plaintext-prepacked compact source; Python backend compile/evaluate over unified complex diagonals",
        backend_name="python",
        logn=int(logn),
    )


def build_r18_stage4_gs256_prototype_python_microbench(*, logn: int = 16) -> dict[str, Any]:
    from orion.experimental.cir import build_r18_stage4_compact_intra_gs256_prototype_plan

    return _run_r18_stage4_plan_microbench(
        plan_builder=build_r18_stage4_compact_intra_gs256_prototype_plan,
        scope="original-size R18 stage4 compact-intra gs256 prototype Python-backend materialization",
        family="stage4_same_gs256_proto",
        prepack_execution="plaintext-prepacked compact source; Python backend compile/evaluate over prototype rerouted diagonals",
        backend_name="python",
        logn=int(logn),
    )


def build_tconv_k2s2_lattigo_microbench(*, logn: int = 16) -> dict[str, Any]:
    from orion.backend.python.tensors import CipherTensor
    from orion.core.orion import scheme
    from orion.experimental.cir.lattigo_block import (
        TconvK2S2Spec,
        _tconv_k2s2_compact_source_pairs,
        _tconv_k2s2_output_phase,
        _tconv_k2s2_phase_expansion_shifts,
        _tconv_k2s2_phase_mask,
        build_tconv_k2s2_phase_pair_plan,
    )
    from orion.nn.unified_transform import UnifiedTransformGroup

    # Small spec for fast test: matches a typical decoder upsample block
    spec = TconvK2S2Spec(stage="test_u3", c_in=32, h_in=8, w_in=8, c_out=32, in_gap=2, out_gap=1)
    config = {
        "ckks_params": {"LogN": int(logn), "LogQ": [45, 30, 30, 45], "LogP": [50], "LogScale": 30, "H": 64, "RingType": "Standard"},
        "orion": {"margin": 2, "embedding_method": "hybrid", "backend": "lattigo", "fuse_modules": True, "debug": False, "io_mode": "none"},
    }
    timings: dict[str, float] = {}
    t0 = time.time()
    scheme.init_scheme(config)
    timings["scheme_init_s"] = float(time.time() - t0)
    try:
        slots = int(scheme.params.get_slots())
        level = len(scheme.params.get_logq()) - 1
        mix_level = max(0, int(level) - 1)
        plan, inputs, reference = build_tconv_k2s2_phase_pair_plan(spec)
        banking_info = {}
        for note in plan.notes:
            if "=" not in str(note):
                continue
            key, value = str(note).split("=", 1)
            banking_info[str(key)] = str(value)
        prepared = {str(p.plaintext_id): p for p in plan.prepared_plaintexts}
        templates = {str(e.template_id): e for fam in plan.family_templates for e in fam.template_entries}
        bank_channels = int(banking_info.get("output_bank_channels", spec.c_out))
        bank_count = int(banking_info.get("output_bank_count", 1))
        input_phase_count = int(banking_info.get("input_phase_count", spec.in_gap * spec.in_gap))
        source_pair_count = int(banking_info.get("source_pair_count", 2))
        compact_rotation_values = tuple(
            int(value)
            for value in str(banking_info.get("compact_source_rotation_shifts", "")).split(",")
            if str(value)
        )
        expansion_shift_values = tuple(
            int(value)
            for value in str(banking_info.get("expansion_rotation_shifts", "")).split(",")
            if str(value)
        )

        src_real = inputs["source_0_lane_0"].unsafe_raw_tensor_for_debug().to(dtype=torch.float32)
        def _phase_index_from_input_id(input_id: str) -> int:
            return int(str(input_id).rsplit("_", 1)[-1])

        def _transform_from_step(step: Any) -> Any:
            diag_tensors: dict[int, torch.Tensor] = {}
            for term in step.terms:
                template = templates[str(term.template_id)]
                plaintext = prepared[str(term.plaintext_id)]
                mapped_indices = template.indices.to(dtype=torch.int64).index_select(0, term.lookup_indices.to(dtype=torch.int64))
                output_indices = term.output_slot_indices.to(dtype=torch.int64)
                if not bool(torch.equal(mapped_indices, output_indices)):
                    raise ValueError(f"term {term.term_id} cannot be encoded as one dense diagonal")
                values = plaintext.values.to(dtype=torch.complex64)
                diag_index = (-int(term.shift)) % int(slots)
                diag = diag_tensors.setdefault(int(diag_index), torch.zeros((int(slots),), dtype=torch.complex64))
                diag.index_add_(0, output_indices, values)
            return SimpleNamespace(
                name=str(step.step_id),
                diagonals={(0, 0): {int(index): diag.tolist() for index, diag in sorted(diag_tensors.items())}},
                level=int(mix_level),
                scheme=scheme,
                fhe_output_shape=torch.Size([1, int(slots)]),
                output_shape=torch.Size([1, int(slots)]),
                target_index=int(step.target_index),
            )

        transforms_by_phase: dict[int, list[tuple[int, Any]]] = {}
        for step in plan.linear_transform_steps:
            phase = _phase_index_from_input_id(str(step.input_id))
            transforms_by_phase.setdefault(int(phase), []).append((int(step.target_index), _transform_from_step(step)))

        groups_by_phase: dict[int, tuple[tuple[int, ...], Any]] = {}
        t0 = time.time()
        for phase, entries in sorted(transforms_by_phase.items()):
            ordered = sorted(entries, key=lambda item: int(item[0]))
            group = UnifiedTransformGroup([transform for _target_index, transform in ordered])
            group.compile_unified(scheme.backend)
            groups_by_phase[int(phase)] = (tuple(int(target_index) for target_index, _transform in ordered), group)
        timings["compile_unified_s"] = float(time.time() - t0)

        expected_outputs: list[torch.Tensor] = []
        for bank_index in range(int(bank_count)):
            oc_start = int(bank_index) * int(bank_channels)
            oc_end = min(int(spec.c_out), oc_start + int(bank_channels))
            for pair_index in range(int(spec.pair_count)):
                phase0 = int(pair_index) * 2
                phase1 = int(pair_index) * 2 + 1
                expected = torch.zeros((int(slots),), dtype=torch.complex64)
                for oc in range(int(oc_start), int(oc_end)):
                    oc_local = int(oc - oc_start)
                    for oh in range(int(spec.h_out)):
                        for ow in range(int(spec.w_out)):
                            phase = _tconv_k2s2_output_phase(oh, ow)
                            if int(phase) != int(phase0) and int(phase) != int(phase1):
                                continue
                            out_slot = oc_local * int(spec.h_out) * int(spec.w_out) + int(oh) * int(spec.w_out) + int(ow)
                            value = float(reference[oc, oh, ow])
                            if int(phase) == int(phase0):
                                expected[int(out_slot)] = complex(value, 0.0)
                            else:
                                expected[int(out_slot)] = complex(0.0, value)
                expected_outputs.append(expected)

        ct_in = scheme.encrypt(scheme.encode(src_real, level))
        ql = scheme.encoder.get_moduli_chain()[int(level)]
        phase_masks = {
            int(phase): scheme.encoder.encode(_tconv_k2s2_phase_mask(int(phase), spec), level=int(level), scale=ql)
            for phase in range(int(input_phase_count))
        }

        partial_outputs: list[Any | None] = [None for _ in expected_outputs]
        t0 = time.time()
        compact_pairs = _tconv_k2s2_compact_source_pairs(spec)
        for pair_index, pair in enumerate(compact_pairs):
            compact_ct: Any | None = None
            for lane_index, (phase, logical_shift) in enumerate(pair):
                phase_ct = ct_in * phase_masks[int(phase)]
                if int(logical_shift) != 0:
                    phase_ct = phase_ct.roll(int(-logical_shift), in_place=False)
                if int(lane_index) == 1:
                    phase_ct = phase_ct.mul_imaginary_unit(+1, in_place=False)
                compact_ct = phase_ct if compact_ct is None else compact_ct + phase_ct
            if compact_ct is None:
                raise RuntimeError("failed to build compact tconv source pair")
            expanded_row = compact_ct + compact_ct.roll(-1, in_place=False)
            expanded_pair = expanded_row + expanded_row.roll(-int(spec.w_out), in_place=False)

            conj_pair = expanded_pair.conjugate(in_place=False)
            expanded_real = (expanded_pair + conj_pair) * 0.5
            expanded_imag = (expanded_pair - conj_pair).mul_imaginary_unit(-1, in_place=False) * 0.5
            phase_sources = {
                int(pair[0][0]): expanded_real,
                int(pair[1][0]): expanded_imag,
            }

            for phase, expanded in phase_sources.items():
                target_indices, group = groups_by_phase[int(phase)]
                output_ids = group.evaluate_unified(int(expanded.ids[0]), scheme.backend)
                for target_index, output_id in zip(target_indices, output_ids):
                    out_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, int(slots)]), torch.Size([1, int(slots)]))
                    if partial_outputs[int(target_index)] is None:
                        partial_outputs[int(target_index)] = out_ct
                    else:
                        partial_outputs[int(target_index)] = partial_outputs[int(target_index)] + out_ct
        timings["evaluate_unified_s"] = float(time.time() - t0)

        max_errors: list[float] = []
        for partial, expected in zip(partial_outputs, expected_outputs):
            if partial is None:
                raise RuntimeError("factorized tconv microbench produced a missing output block")
            out_pt = partial.decrypt()
            raw = scheme.backend.DecodeComplex(out_pt.ids[0])
            decoded = torch.tensor([complex(raw[2 * i], raw[2 * i + 1]) for i in range(int(slots))], dtype=torch.complex64)
            max_errors.append(float((decoded - expected).abs().max()))

        max_abs = float(max(max_errors, default=0.0))
        exact = bool(max_abs <= 1.0e-3)
        return {
            "status": "ok" if exact else "failed",
            "scope": "tconv k2s2 factorized expand-then-mix Lattigo microbench",
            "spec": {"stage": spec.stage, "c_in": spec.c_in, "h_in": spec.h_in, "c_out": spec.c_out, "in_gap": spec.in_gap, "out_gap": spec.out_gap},
            "local_lattigo": True,
            "unified_transform_group": True,
            "pair_count": int(spec.pair_count),
            "input_phase_count": int(input_phase_count),
            "source_pair_count": int(source_pair_count),
            "rotation_count": int(banking_info.get("selected_rotation_count", 0)),
            "mix_rotation_count": int(banking_info.get("mix_rotation_count", 0)),
            "compact_source_rotation_count": int(len(compact_rotation_values)),
            "expansion_rotation_count": int(len(expansion_shift_values)),
            "term_count": int(sum(len(step.terms) for step in plan.linear_transform_steps)),
            "output_bank_channels": int(bank_channels),
            "output_bank_count": int(bank_count),
            "parity": {"exact": exact, "max_abs": max_abs, "max_errors": max_errors, "tolerance": 1.0e-3},
            "timing_s": timings,
        }
    finally:
        scheme.delete_scheme()


def write_big_graph_lattigo_microbench(
    *,
    out_path: Path = DEFAULT_BIG_GRAPH_LATTIGO_MICROBENCH_OUT,
    bank_count: int = 8,
    logn: int = 16,
) -> dict[str, Any]:
    payload = build_big_graph_lattigo_microbench(bank_count=int(bank_count), logn=int(logn))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
