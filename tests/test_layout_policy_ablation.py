from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from orion.core.orion import _region_first_mode_options
from orion.experimental.layout_policy_ablation import (
    attach_non_ckks_simulation,
    attach_runtime_anchor,
    build_planner_ablation,
    build_u22_dag,
    network_spec,
    run_non_ckks_layout_simulation,
    run_runtime_anchor,
)
from orion.experimental.u22_phase1 import U22CompileRegistry


def _policy(payload: dict, name: str) -> dict:
    for row in payload["policies"]:
        if row["policy"] == str(name):
            return row
    raise AssertionError(f"missing policy {name}")


def _layout_key(row: dict) -> tuple[int, int, int, int]:
    layout = row["selected_layout"]
    return int(layout["alpha"]), int(layout["beta"]), int(layout["stride"]), int(layout["gap"])


def test_u22_64_layout_policy_planner_reports_all_edges_and_ordering() -> None:
    payload = build_planner_ablation(
        network="u22_64_base32",
        policies=("fixed_max", "eager", "greedy", "dp"),
    )

    assert payload["graph"]["node_count"] == 31
    assert payload["graph"]["edge_count"] == 34
    assert [row["policy"] for row in payload["policies"]] == ["fixed_max", "eager", "greedy", "dp"]
    for row in payload["policies"]:
        assert row["metric_source"] == "planner_estimate"
        assert len(row["edge_layouts"]) == 34

    fixed = _policy(payload, "fixed_max")
    eager = _policy(payload, "eager")
    dp = _policy(payload, "dp")
    assert fixed["relayouts"] <= eager["relayouts"]
    assert eager["halo_redundancy_ratio"] <= fixed["halo_redundancy_ratio"]
    assert dp["objective"] <= min(row["objective"] for row in payload["policies"] if row["policy"] != "dp")


def test_u22_64_layout_policy_planner_aligns_add_inputs() -> None:
    payload = build_planner_ablation(network="u22_64_base32")

    for policy in payload["policies"]:
        rows = list(policy["edge_layouts"])
        for add_node in ("add4", "add3", "add2", "add1"):
            incoming = [row for row in rows if row["target"] == add_node]
            assert len(incoming) == 2
            assert len({_layout_key(row) for row in incoming}) == 1


def test_layout_policy_parser_marks_non_dp_u22_modes_as_planner_only() -> None:
    opts = _region_first_mode_options("u22_64_base32_layout_fixedmax")
    assert opts["u22_layout_policy"] == "fixed_max"
    assert opts["u22_allowed_nodes"] == ("up1", "up2", "up3", "up4")

    dag = build_u22_dag(network_spec("u22_64_base32"))
    registry = U22CompileRegistry.for_dag(
        dag,
        allowed_nodes=opts["u22_allowed_nodes"],
        enable_conv_kernels=bool(opts["u22_conv_kernels"]),
        layout_policy=str(opts["u22_layout_policy"]),
    )
    audit = registry.attach_to_dag(dag)

    assert audit["attached_count"] == 0
    assert audit["graph_audit"]["layout_policy"] == "fixed_max"
    assert audit["graph_audit"]["layout_policy_runtime"] == "planner_only"
    assert {row["reason"] for row in audit["graph_audit"]["excluded_nodes"]} == {"u22_layout_policy_planner_only"}


def test_runtime_anchor_timeout_is_reported_without_launching_real_e2e(tmp_path: Path) -> None:
    def fake_runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["fake"], timeout=1, output="still compiling")

    anchor = run_runtime_anchor(
        network="u22_64_base32",
        backend="lattigo",
        cache_root=tmp_path,
        compile_timeout_s=1,
        runner=fake_runner,
    )

    assert anchor["status"] == "compile_timeout"
    payload = attach_runtime_anchor(build_planner_ablation(network="u22_64_base32", policies=("dp",)), anchor)
    dp = _policy(payload, "dp")
    assert dp["metric_source"] == "planner_estimate+runtime_anchor"
    assert dp["runtime_status"] == "compile_timeout"


def test_runtime_anchor_extracts_mocked_e2e_metrics(tmp_path: Path) -> None:
    def fake_runner(cmd, **_kwargs):
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "montgomery_lung_64_samples0_1_provider_fhe_figure.json").write_text(
            """
{
  "status": "ok",
  "timing_s": {"he_forward_total": 12.5},
  "samples": [
    {
      "fhe_vs_pytorch_logits": {"mae": 0.125},
      "fhe_vs_reference": {"dice": 0.75}
    }
  ]
}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    anchor = run_runtime_anchor(
        network="u22_64_base32",
        backend="lattigo",
        cache_root=tmp_path,
        compile_timeout_s=10,
        runner=fake_runner,
    )

    assert anchor["status"] == "ok"
    assert anchor["he_forward_s"] == 12.5
    assert anchor["mae"] == 0.125
    assert anchor["dice"] == 0.75


def test_non_ckks_layout_simulation_attaches_clear_sanity_metrics() -> None:
    payload = build_planner_ablation(network="u22_64_base32", policies=("fixed_max", "eager", "greedy", "dp"))
    simulation = run_non_ckks_layout_simulation(payload, seed=0)
    updated = attach_non_ckks_simulation(payload, simulation)

    assert simulation["status"] == "ok"
    for row in updated["policies"]:
        assert row["metric_source"] == "planner_estimate+non_ckks_sim"
        assert row["runtime_status"] == "non_ckks_sim_ok"
        assert row["layout_alignment_ok"] is True
        assert row["mae"] == 0.0
        assert row["max_abs"] == 0.0
        assert row["dice"] == 1.0


def test_layout_policy_cli_planner_smoke(tmp_path: Path) -> None:
    out_csv = tmp_path / "layout_policy_ablation_u22_64.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_layout_policy_ablation.py",
            "--network",
            "u22_64_base32",
            "--policies",
            "fixed_max",
            "eager",
            "greedy",
            "dp",
            "--mode",
            "planner",
            "--out",
            str(out_csv),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout
    assert out_csv.exists()
    assert out_csv.with_suffix(".json").exists()
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["policy"] for row in rows] == ["fixed_max", "eager", "greedy", "dp"]
    assert all(row["metric_source"] == "planner_estimate" for row in rows)


def test_layout_policy_cli_non_ckks_simulation_smoke(tmp_path: Path) -> None:
    out_csv = tmp_path / "layout_policy_ablation_u22_64_sim.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_layout_policy_ablation.py",
            "--network",
            "u22_64_base32",
            "--mode",
            "planner,simulate",
            "--out",
            str(out_csv),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["metric_source"] == "planner_estimate+non_ckks_sim" for row in rows)
    assert all(row["runtime_status"] == "non_ckks_sim_ok" for row in rows)
    assert all(row["layout_alignment_ok"] == "True" for row in rows)
