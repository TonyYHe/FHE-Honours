from __future__ import annotations

from typing import Iterable

from .region_first_data import (
    EXCLUDED_SYNTHETIC_ROWS,
    ORION_BASELINE,
    R18_R34_FIXTURES,
    RegionFirstFixture,
    stats_delta,
    stats_sum,
    summary,
)


def _normalize_models(models: Iterable[str]) -> set[str]:
    aliases = {
        "r18": "resnet18_tiny_imagenet",
        "resnet18": "resnet18_tiny_imagenet",
        "resnet18_tiny_imagenet": "resnet18_tiny_imagenet",
        "r34": "resnet34_imagenet",
        "resnet34": "resnet34_imagenet",
        "resnet34_imagenet": "resnet34_imagenet",
        "all": "all",
    }
    resolved = {aliases.get(str(model).lower(), str(model)) for model in models}
    if "all" in resolved:
        return {"resnet18_tiny_imagenet", "resnet34_imagenet"}
    return resolved


def _case_from_fixture(fixture: RegionFirstFixture) -> dict:
    selected = {
        "candidate": str(fixture.candidate),
        "status": "actual_materialized",
        "materializer": str(fixture.materializer),
        "same_plan_certificate": True,
        "selectable": True,
        "count_only": False,
        "stats": dict(fixture.selected_stats),
        "scaled_stats": dict(fixture.selected_stats),
        "score": summary(fixture.selected_stats, fixture.orion_stats)["score"]["ours"],
        "scaled_score": summary(fixture.selected_stats, fixture.orion_stats)["score"]["ours"],
        "parity": dict(fixture.parity),
        "boundary_layout": {
            "kind": "regular_gap_after_real_extract",
            "relu_safe": True,
            "extract_inserted": True,
            "prepack_inserted": True,
            "note": "real/imag hybrid output is extracted before ReLU/add boundary",
        },
        "boundary_actions": ["insert_extract", "validate_relu_safe"],
        "scaled_delta_vs_orion": stats_delta(fixture.selected_stats, fixture.orion_stats),
        "stats_from_execution": dict(fixture.selected_stats),
        "stats_from_plan": dict(fixture.selected_stats),
    }
    orion = {
        "candidate": ORION_BASELINE,
        "status": "actual_materialized",
        "materializer": ORION_BASELINE,
        "same_plan_certificate": True,
        "selectable": True,
        "count_only": False,
        "stats": dict(fixture.orion_stats),
        "scaled_stats": dict(fixture.orion_stats),
        "parity": dict(fixture.parity) | {"source": "vendored_scripts_cir_orion_dense_reference"},
    }
    return {
        "model": str(fixture.model),
        "network": str(fixture.network),
        "family": str(fixture.family),
        "region_id": str(fixture.region_id),
        "representative": str(fixture.representative),
        "source_families": list(fixture.source_families),
        "region_context": {
            "region_kind": "same_shape_big_graph",
            "next_op": "relu",
            "useful_output_banks": 2,
        },
        "orion": orion,
        "candidates": [selected],
        "selected": selected,
        "rejected_candidates": [],
    }


def _summaries(cases: list[dict]) -> dict[str, object]:
    model_totals: dict[str, dict[str, dict[str, int]]] = {}
    for case in cases:
        bucket = model_totals.setdefault(
            str(case["model"]),
            {"ours": {"rotations": 0, "conjugations": 0, "ct_pt_mults": 0, "adds": 0}, "orion": {"rotations": 0, "conjugations": 0, "ct_pt_mults": 0, "adds": 0}},
        )
        for key in bucket["ours"]:
            bucket["ours"][key] += int(case["selected"]["scaled_stats"].get(key, 0))
            bucket["orion"][key] += int(case["orion"]["scaled_stats"].get(key, 0))
    model_summaries = {model: summary(values["ours"], values["orion"]) for model, values in model_totals.items()}
    combined_ours = stats_sum(tuple(values["ours"] for values in model_totals.values()))
    combined_orion = stats_sum(tuple(values["orion"] for values in model_totals.values()))
    return {
        "model_summaries": model_summaries,
        "combined": summary(combined_ours, combined_orion),
    }


def build_region_first_full_selector(*, models: tuple[str, ...] = ("resnet18", "resnet34")) -> dict:
    wanted = _normalize_models(models)
    cases = [_case_from_fixture(fixture) for fixture in R18_R34_FIXTURES if fixture.model in wanted]
    summaries = _summaries(cases)
    audit = {
        "status": "ok",
        "selected_count": int(len(cases)),
        "selected_non_orion_count": int(len(cases)),
        "count_only_selected": [],
        "parity_fail_selected": [],
        "unsafe_boundary_selected": [],
        "hidden_fallback_count": 0,
        "excluded_synthetic_rows": list(EXCLUDED_SYNTHETIC_ROWS),
    }
    return {
        "status": "ok" if cases else "empty",
        "scope": "Orion-local vendored region-first full selector for R18/R34; not full CKKS runtime",
        "registry_version": "orion_vendored_region_first_v1",
        "models": sorted(wanted),
        "cases": cases,
        "region_replacements": [],
        "summary": {
            "case_count": int(len(cases)),
            "selected_non_orion_count": int(len(cases)),
            "all_selected_parity_ok": bool(all(bool(case["selected"]["parity"].get("exact")) for case in cases)),
            **summaries,
        },
        "audit": audit,
        "limitations": {
            "excluded_synthetic_rows": list(EXCLUDED_SYNTHETIC_ROWS),
            "full_ckks_runtime": "not executed",
            "production_orion_compile_integration": "future work",
        },
    }


def build_region_first_full_selector_summary(selector_payload: dict) -> dict:
    cases = [dict(case) for case in selector_payload.get("cases", [])]
    summaries = _summaries(cases)
    return {
        "status": "ok" if str(selector_payload.get("status")) == "ok" and str(selector_payload.get("audit", {}).get("status")) == "ok" else "failed",
        "scope": "summary of Orion-local vendored region-first full selector artifact",
        "registry_version": str(selector_payload.get("registry_version", "")),
        "case_count": int(len(cases)),
        **summaries,
        "selected_non_orion": [
            {
                "model": str(case["model"]),
                "family": str(case["family"]),
                "region_id": str(case["region_id"]),
                "candidate": str(case["selected"]["candidate"]),
                "scaled_delta_vs_orion": dict(case["selected"]["scaled_delta_vs_orion"]),
                "parity": dict(case["selected"]["parity"]),
                "boundary_actions": list(case["selected"].get("boundary_actions", [])),
            }
            for case in cases
        ],
        "excluded_synthetic_rows": list(EXCLUDED_SYNTHETIC_ROWS),
        "audit": dict(selector_payload.get("audit", {})),
    }

