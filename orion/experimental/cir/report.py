from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .region_first_data import EXCLUDED_SYNTHETIC_ROWS
from .selector import build_region_first_full_selector, build_region_first_full_selector_summary


DEFAULT_PIPELINE_REPORT_OUT = Path("/tmp/orion_region_first_pipeline_report.json")


def _backend_evidence(attach_lattigo: bool, lattigo_evidence: dict[str, Any] | None) -> dict[str, Any]:
    if lattigo_evidence is not None:
        evidence = dict(lattigo_evidence)
    elif bool(attach_lattigo):
        from orion.core.region_cir_replay import build_big_graph_lattigo_microbench

        evidence = build_big_graph_lattigo_microbench()
    else:
        evidence = {"status": "not_run", "reason": "attach_lattigo=False"}
    return {
        "status": str(evidence.get("status", "unknown")),
        "scope": str(evidence.get("scope", "")),
        "network": str(evidence.get("network", "")),
        "family": str(evidence.get("family", "")),
        "region_id": str(evidence.get("region_id", "")),
        "full_region": bool(evidence.get("full_region", False)),
        "original_size_slot_domain": bool(evidence.get("original_size_slot_domain", False)),
        "local_lattigo": bool(evidence.get("local_lattigo", False)),
        "unified_transform_group": bool(evidence.get("unified_transform_group", False)),
        "uses_orion_dense_pack_conv2d": bool(evidence.get("uses_orion_dense_pack_conv2d", True)) if evidence.get("status") != "not_run" else None,
        "bank_count": int(evidence.get("bank_count", 0)),
        "stats_from_execution": dict(evidence.get("stats_from_execution", {})),
        "scripts_cir_full_block_stats": dict(evidence.get("scripts_cir_full_block_stats", {})),
        "parity": dict(evidence.get("parity", {})),
        "publishable_lattigo_microbenchmark": bool(evidence.get("publishable_lattigo_microbenchmark", False)),
        "note": "backend validation is original-size block-level, not full-network CKKS runtime",
    }


def build_region_first_pipeline_report(
    *,
    models: tuple[str, ...] = ("resnet18", "resnet34"),
    attach_lattigo: bool = True,
    lattigo_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selector = build_region_first_full_selector(models=tuple(models))
    summary = build_region_first_full_selector_summary(selector)
    backend = _backend_evidence(bool(attach_lattigo), lattigo_evidence)
    selected_rows = list(summary.get("selected_non_orion", []))
    hidden_fallback_count = int(dict(selector.get("audit", {})).get("hidden_fallback_count", 0))
    status = "ok" if str(selector.get("status")) == "ok" and str(summary.get("status")) == "ok" and int(hidden_fallback_count) == 0 else "failed"
    return {
        "status": str(status),
        "scope": "Orion-local experimental region-first full-network selector report for R18/R34",
        "selector": selector,
        "summary": summary,
        "method": {
            "region_discovery": "vendored R18/R34 same-shape region inventory",
            "candidate_generation": [
                "orion_dense_bsgs_output_fold_lt",
                "generalized_inter_hsplit_full_native_shared_output_collapse",
            ],
            "selected_methods": [
                "real_imag_hybrid",
                "shared_output_banks",
                "insert_extract_before_relu_or_add_boundary",
            ],
            "input_repetition_output_fold": "supported in experimental path but not a main R18/R34 selected claim",
        },
        "publishable_facts": {
            "full_network_cost": {
                "publishable": bool(str(status) == "ok"),
                "models": ["R18", "R34"],
                "combined": dict(summary.get("combined", {})),
                "selected_non_orion_count": int(dict(selector.get("summary", {})).get("selected_non_orion_count", 0)),
            },
            "lattigo_backend": {
                "publishable": bool(backend.get("publishable_lattigo_microbenchmark", False)),
                "evidence": backend,
            },
        },
        "claim_hygiene": {
            "excluded_synthetic_rows": list(EXCLUDED_SYNTHETIC_ROWS),
            "r20_in_main_claim": False,
            "transition_compiled_only_in_main_claim": False,
            "hidden_fallback_count": int(hidden_fallback_count),
            "selected_rows_not_count_only": bool(all(not bool(row.get("count_only", False)) for row in selected_rows)),
            "full_ckks_runtime_claimed": False,
        },
        "artifacts": {
            "pipeline_report": str(DEFAULT_PIPELINE_REPORT_OUT),
            "big_graph_lattigo_microbench": "/tmp/orion_big_graph_lattigo_microbench.json",
        },
    }


def write_region_first_pipeline_report(
    *,
    out_path: Path = DEFAULT_PIPELINE_REPORT_OUT,
    models: tuple[str, ...] = ("resnet18", "resnet34"),
    attach_lattigo: bool = True,
    lattigo_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_region_first_pipeline_report(
        models=tuple(models),
        attach_lattigo=bool(attach_lattigo),
        lattigo_evidence=lattigo_evidence,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload

