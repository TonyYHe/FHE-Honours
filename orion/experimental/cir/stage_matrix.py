from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .region_first_data import STAGE_MATERIALIZER_REFERENCES


DEFAULT_STAGE_MATRIX_OUT = Path("/tmp/orion_stage_materialization_lattigo_matrix.json")


def _lattigo_evidence_by_stage(*payloads: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for lattigo_payload in payloads:
        if not lattigo_payload:
            continue
        if str(lattigo_payload.get("status")) != "ok":
            continue
        if str(lattigo_payload.get("network")) != "R18":
            continue
        family = str(lattigo_payload.get("family"))
        if family == "stage1_same":
            out[("R18", "stage1")] = dict(lattigo_payload)
        elif family == "stage2_same":
            out[("R18", "stage2")] = dict(lattigo_payload)
        elif family == "stage3_same":
            out[("R18", "stage3")] = dict(lattigo_payload)
        elif family == "stage4_same":
            out[("R18", "stage4")] = dict(lattigo_payload)
    return out


def build_stage_materialization_lattigo_matrix(
    *,
    lattigo_payload: dict[str, Any] | None = None,
    extra_lattigo_payloads: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    evidence_by_stage = _lattigo_evidence_by_stage(lattigo_payload, *tuple(extra_lattigo_payloads))
    rows: list[dict[str, Any]] = []
    for ref in STAGE_MATERIALIZER_REFERENCES:
        evidence = evidence_by_stage.get((str(ref.network), str(ref.stage)))
        expected = dict(ref.expected_stats)
        if evidence is not None:
            observed = {key: int(dict(evidence.get("stats_from_execution", {})).get(key, 0)) for key in expected}
            stats_match = observed == expected
            parity = dict(evidence.get("parity", {}))
            status = "ok" if bool(stats_match and parity.get("exact", False)) else "failed"
            blocker = ""
        else:
            observed = {}
            stats_match = False
            parity = {"exact": False, "reason": "no Orion-local Lattigo materializer row yet"}
            status = "missing_materializer"
            blocker = f"port {ref.materializer} for {ref.network} {ref.stage}"
        rows.append(
            {
                "network": str(ref.network),
                "stage": str(ref.stage),
                "family": str(ref.family),
                "materializer": str(ref.materializer),
                "status": str(status),
                "expected_stats": expected,
                "lattigo_stats": observed,
                "stats_match_scripts_cir": bool(stats_match),
                "parity": parity,
                "source": str(ref.source),
                "blocker": str(blocker or ref.note),
                "publishable_lattigo_fact": bool(status == "ok"),
            }
        )
    return {
        "status": "ok" if any(row["status"] == "ok" for row in rows) else "missing_materializers",
        "scope": "R18/R34 stage1-4 materializer-to-Lattigo matrix; missing rows are explicit",
        "rows": rows,
        "summary": {
            "ok_count": int(sum(1 for row in rows if row["status"] == "ok")),
            "missing_materializer_count": int(sum(1 for row in rows if row["status"] == "missing_materializer")),
            "failed_count": int(sum(1 for row in rows if row["status"] == "failed")),
        },
    }


def write_stage_materialization_lattigo_matrix(
    *,
    out_path: Path = DEFAULT_STAGE_MATRIX_OUT,
    lattigo_payload: dict[str, Any] | None = None,
    extra_lattigo_payloads: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    payload = build_stage_materialization_lattigo_matrix(lattigo_payload=lattigo_payload, extra_lattigo_payloads=tuple(extra_lattigo_payloads))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
