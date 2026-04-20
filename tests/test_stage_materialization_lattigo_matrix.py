from __future__ import annotations

from orion.experimental.cir.stage_matrix import build_stage_materialization_lattigo_matrix


def _fake_r18_stage1_evidence() -> dict:
    return {
        "status": "ok",
        "network": "R18",
        "family": "stage1_same",
        "stats_from_execution": {"rotations": 20, "conjugations": 8, "ct_pt_mults": 1080, "adds": 1089},
        "parity": {"exact": True, "max_abs": 0.0001, "tolerance": 0.001},
    }


def _row(payload: dict, network: str, stage: str) -> dict:
    for row in payload["rows"]:
        if row["network"] == network and row["stage"] == stage:
            return row
    raise AssertionError(f"missing row {network}:{stage}")


def test_stage_matrix_has_all_r18_r34_stage_rows() -> None:
    payload = build_stage_materialization_lattigo_matrix(lattigo_payload=_fake_r18_stage1_evidence())

    assert len(payload["rows"]) == 8
    assert {(row["network"], row["stage"]) for row in payload["rows"]} == {
        ("R18", "stage1"),
        ("R18", "stage2"),
        ("R18", "stage3"),
        ("R18", "stage4"),
        ("R34", "stage1"),
        ("R34", "stage2"),
        ("R34", "stage3"),
        ("R34", "stage4"),
    }


def test_stage_matrix_marks_r18_stage1_ok_and_others_missing() -> None:
    payload = build_stage_materialization_lattigo_matrix(lattigo_payload=_fake_r18_stage1_evidence())

    assert _row(payload, "R18", "stage1")["status"] == "ok"
    assert _row(payload, "R18", "stage1")["stats_match_scripts_cir"] is True
    assert _row(payload, "R18", "stage1")["publishable_lattigo_fact"] is True
    missing = [row for row in payload["rows"] if row["status"] == "missing_materializer"]
    assert len(missing) == 7
    assert payload["summary"]["ok_count"] == 1
    assert payload["summary"]["missing_materializer_count"] == 7


def test_stage_matrix_locks_reference_stats_for_next_ports() -> None:
    payload = build_stage_materialization_lattigo_matrix(lattigo_payload=_fake_r18_stage1_evidence())

    assert _row(payload, "R18", "stage2")["expected_stats"] == {
        "rotations": 84,
        "conjugations": 8,
        "ct_pt_mults": 5880,
        "adds": 5890,
    }
    assert _row(payload, "R18", "stage3")["expected_stats"] == {
        "rotations": 90,
        "conjugations": 2,
        "ct_pt_mults": 6750,
        "adds": 6753,
    }
    assert _row(payload, "R18", "stage4")["expected_stats"] == {
        "rotations": 158,
        "conjugations": 1,
        "ct_pt_mults": 9767,
        "adds": 9768,
    }
    assert _row(payload, "R34", "stage3")["expected_stats"] == {
        "rotations": 204,
        "conjugations": 2,
        "ct_pt_mults": 7938,
        "adds": 7941,
    }
