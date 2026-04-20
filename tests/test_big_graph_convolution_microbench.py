from __future__ import annotations

from pathlib import Path

from orion.core.region_cir_replay import (
    build_big_graph_convolution_microbench,
    write_big_graph_convolution_microbench,
)


def test_big_graph_microbench_excludes_r20_synthetic_rows() -> None:
    payload = build_big_graph_convolution_microbench()

    networks = {row["network"] for row in payload["rows"]}

    assert networks == {"R18", "R34"}
    assert all(row["synthetic"] is False for row in payload["rows"])
    assert payload["publishability"]["excluded_synthetic_rows"] == ["R20:stage1_two_output_region"]


def test_big_graph_microbench_has_publishable_original_size_cost_rows() -> None:
    payload = build_big_graph_convolution_microbench()
    rows = {row["network"]: row for row in payload["rows"]}

    assert payload["publishability"]["cost_microbenchmark_publishable_count"] == 2
    assert rows["R18"]["cir_stats"] == {
        "rotations": 572,
        "conjugations": 152,
        "ct_pt_mults": 34920,
        "adds": 35094,
    }
    assert rows["R34"]["cir_stats"] == {
        "rotations": 1928,
        "conjugations": 124,
        "ct_pt_mults": 48508,
        "adds": 48670,
    }
    assert all(row["publishable_cost_microbenchmark"] is True for row in rows.values())
    assert all(row["scripts_cir_executor_equivalent"] is True for row in rows.values())
    assert all(row["scripts_cir_parity"]["exact"] is True for row in rows.values())


def test_big_graph_microbench_does_not_promote_tiny_lattigo_to_original_size_fact(tmp_path: Path) -> None:
    payload = build_big_graph_convolution_microbench(lattigo_microbench_artifact=tmp_path / "missing_lattigo.json")

    assert payload["status"] == "needs_original_size_lattigo"
    assert payload["publishability"]["lattigo_microbenchmark_publishable_count"] == 0
    for row in payload["rows"]:
        backend = row["lattigo_backend_evidence"]
        assert backend["available"] is True
        assert backend["uses_lattigo"] is True
        assert backend["original_size"] is False
        assert backend["evidence_shape"] == "tiny_representative"
        assert row["publishable_lattigo_microbenchmark"] is False
        assert "original-size Lattigo" in row["lattigo_blocker"]


def test_big_graph_microbench_counts_original_size_lattigo_artifact(tmp_path: Path) -> None:
    lattigo_artifact = tmp_path / "orion_big_graph_lattigo_microbench.json"
    lattigo_artifact.write_text(
        """
{
  "status": "ok",
  "network": "R18",
  "family": "stage1_same",
  "region_id": "stage1_same_block0",
  "full_region": false,
  "original_size_slot_domain": true,
  "bank_count": 8,
  "stats_from_execution": {"rotations": 20, "conjugations": 8, "ct_pt_mults": 1080, "adds": 1089},
  "parity": {"exact": true, "max_abs": 0.0001, "tolerance": 0.001},
  "publishable_lattigo_microbenchmark": true
}
""",
        encoding="utf-8",
    )

    payload = build_big_graph_convolution_microbench(lattigo_microbench_artifact=lattigo_artifact)

    assert payload["status"] == "partial_original_size_lattigo"
    assert payload["publishability"]["lattigo_microbenchmark_publishable_count"] == 1
    assert payload["lattigo_microbench_rows"][0]["network"] == "R18"
    assert payload["lattigo_microbench_rows"][0]["bank_count"] == 8


def test_big_graph_microbench_artifact_is_written(tmp_path: Path) -> None:
    out = tmp_path / "orion_big_graph_convolution_microbench.json"

    payload = write_big_graph_convolution_microbench(out_path=out)

    assert out.exists()
    assert payload["publishability"]["cost_microbenchmark_publishable_count"] == 2
