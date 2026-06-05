#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    REPO_ROOT,
    add_common_args,
    clean_int,
    clean_number,
    copy_doc_snapshot,
    ensure_layout,
    maybe_existing_artifact_root,
    print_outputs,
    read_csv,
    resolve_run_root,
    run_command,
    write_csv,
    write_manifest,
)


ARTIFACT = "fig7"
SOURCE_SCRIPT = "tools/run_conv_kernel_table.py"
STAGE_BY_KERNEL = {
    "Conv 32,32": "S1",
    "Conv 64,64": "S2",
    "Conv 128,128": "S3",
    "Conv 256,256": "S4",
}
SYSTEM_BY_PATH = {
    "dense": "Orion",
    "no-sharing stripe": "HaloED-beta1",
    "provider beta=2 no-share stripe": "HaloED-beta2",
}
SYSTEMS = ("Orion", "HaloED-beta1", "HaloED-beta2")


def _build_command(raw_root: Path, doc_path: Path) -> list[str | Path]:
    return [
        sys.executable,
        REPO_ROOT / SOURCE_SCRIPT,
        "--run-root",
        raw_root,
        "--doc",
        doc_path,
        "--backend",
        "lattigo",
        "--kernel-cases",
        "conv32",
        "conv64",
        "conv128",
        "conv256",
        "--channels",
        "32",
        "64",
        "128",
        "256",
        "--hw",
        "192x192",
        "224x224",
        "256x256",
        "384x384",
        "--variants",
        "orion",
        "provider_halo1_individual_lt",
        "provider_halo2_individual_lt",
        "--input-level",
        "2",
        "--provider-output-layout",
        "native_stripe",
    ]


def _csv_path(root: Path) -> Path:
    direct = root / "conv_kernel_table.csv"
    if direct.exists():
        return direct
    nested = root / "raw" / "conv_kernel_table" / "conv_kernel_table.csv"
    if nested.exists():
        return nested
    matches = sorted(root.rglob("conv_kernel_table.csv"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"missing conv_kernel_table.csv under {root}")


def _load_summary(csv_path: Path) -> list[dict[str, Any]]:
    raw = read_csv(csv_path)
    data: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for row in raw:
        if str(row.get("status", "")).strip() != "ok":
            continue
        stage = STAGE_BY_KERNEL.get(str(row.get("kernel", "")).strip())
        system = SYSTEM_BY_PATH.get(str(row.get("path_beta", "")).strip())
        if not stage or not system:
            continue
        hw = str(row.get("HW", "")).strip()
        data.setdefault((stage, hw), {})[system] = {
            "rotation": float(clean_int(row.get("rotations"))),
            "speed": float(clean_number(row.get("hot_run_s"))),
        }

    rows: list[dict[str, Any]] = []
    dims = sorted({hw for _stage, hw in data}, key=lambda item: (clean_int(item.split("x")[0]), item))
    for stage in ("S1", "S2", "S3", "S4"):
        for dim in dims:
            systems = data.get((stage, dim), {})
            if not all(system in systems for system in SYSTEMS):
                continue
            orion_speed = systems["Orion"]["speed"]
            orion_rot = systems["Orion"]["rotation"]
            rows.append(
                {
                    "type": stage,
                    "dim": dim,
                    "orion_rotation": int(systems["Orion"]["rotation"]),
                    "orion_speed_s": orion_speed,
                    "haloed_beta1_rotation": int(systems["HaloED-beta1"]["rotation"]),
                    "haloed_beta1_speed_s": systems["HaloED-beta1"]["speed"],
                    "haloed_beta2_rotation": int(systems["HaloED-beta2"]["rotation"]),
                    "haloed_beta2_speed_s": systems["HaloED-beta2"]["speed"],
                    "haloed_beta1_speedup": orion_speed / systems["HaloED-beta1"]["speed"],
                    "haloed_beta2_speedup": orion_speed / systems["HaloED-beta2"]["speed"],
                    "haloed_beta1_norm_rotation": systems["HaloED-beta1"]["rotation"] / orion_rot,
                    "haloed_beta2_norm_rotation": systems["HaloED-beta2"]["rotation"] / orion_rot,
                }
            )

    for dim in dims:
        stage_rows = [row for row in rows if row["dim"] == dim and str(row["type"]).startswith("S")]
        if len(stage_rows) != 4:
            continue
        geo: dict[str, Any] = {"type": "Geo.", "dim": dim}
        for key in (
            "orion_rotation",
            "orion_speed_s",
            "haloed_beta1_rotation",
            "haloed_beta1_speed_s",
            "haloed_beta2_rotation",
            "haloed_beta2_speed_s",
        ):
            geo[key] = math.prod(float(row[key]) for row in stage_rows) ** (1.0 / len(stage_rows))
        geo["haloed_beta1_speedup"] = geo["orion_speed_s"] / geo["haloed_beta1_speed_s"]
        geo["haloed_beta2_speedup"] = geo["orion_speed_s"] / geo["haloed_beta2_speed_s"]
        geo["haloed_beta1_norm_rotation"] = geo["haloed_beta1_rotation"] / geo["orion_rotation"]
        geo["haloed_beta2_norm_rotation"] = geo["haloed_beta2_rotation"] / geo["orion_rotation"]
        rows.append(geo)
    return rows


def _plot(rows: list[dict[str, Any]], out_pdf: Path, out_png: Path) -> None:
    stages = ["S1", "S2", "S3", "S4", "Geo."]
    dims = sorted({str(row["dim"]) for row in rows}, key=lambda item: (clean_int(item.split("x")[0]), item))
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.0), sharex=True)
    x = list(range(len(stages) * len(dims)))
    labels = []
    b1 = []
    b2 = []
    r1 = []
    r2 = []
    for stage in stages:
        for dim in dims:
            match = next((row for row in rows if row["type"] == stage and row["dim"] == dim), None)
            labels.append(f"{stage}\n{dim}")
            b1.append(float(match["haloed_beta1_speedup"]) if match else 0.0)
            b2.append(float(match["haloed_beta2_speedup"]) if match else 0.0)
            r1.append(float(match["haloed_beta1_norm_rotation"]) if match else 0.0)
            r2.append(float(match["haloed_beta2_norm_rotation"]) if match else 0.0)

    width = 0.36
    axes[0].bar([value - width / 2 for value in x], b1, width=width, label=r"HaloED $\beta=1$", color="#4C78A8")
    axes[0].bar([value + width / 2 for value in x], b2, width=width, label=r"HaloED $\beta=2$", color="#F58518")
    axes[0].axhline(1.0, color="#777777", linewidth=0.7)
    axes[0].set_ylabel("Speedup vs. Orion")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[0].legend(ncol=2, frameon=False, loc="upper left")

    axes[1].plot(x, r1, marker="o", label=r"Norm. rotations $\beta=1$", color="#1F4E79")
    axes[1].plot(x, r2, marker="s", label=r"Norm. rotations $\beta=2$", color="#B94727")
    axes[1].axhline(1.0, color="#777777", linewidth=0.7)
    axes[1].set_ylabel("Rotations / Orion")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=5)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[1].legend(ncol=2, frameon=False, loc="upper left")

    for sep in range(len(dims), len(x), len(dims)):
        for ax in axes:
            ax.axvline(sep - 0.5, color="#CCCCCC", linewidth=0.45)
    fig.tight_layout(pad=0.3)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HaloED paper Fig. 7 from Orion conv-kernel runs.")
    add_common_args(parser)
    args = parser.parse_args()

    run_root = maybe_existing_artifact_root(resolve_run_root(args, ARTIFACT), ARTIFACT) if args.check_existing else resolve_run_root(args, ARTIFACT)
    dirs = ensure_layout(run_root)
    raw_root = dirs["raw"] / "conv_kernel_table"
    doc_path = dirs["raw"] / "u22_orion_streaming_haloed_mainline.snapshot.md"
    command = _build_command(raw_root, doc_path)
    if args.force:
        command.append("--force")

    if not args.check_existing:
        copy_doc_snapshot(doc_path)
        rc = run_command(command, log_path=dirs["logs"] / "fig7_operator_conv_perf.log", dry_run=bool(args.dry_run))
        if rc != 0:
            return rc
        if args.dry_run:
            return 0

    csv_path = _csv_path(run_root)
    rows = _load_summary(csv_path)
    if not rows:
        raise SystemExit(f"no complete Fig.7 rows found in {csv_path}")
    summary_csv = dirs["paper"] / "fig7_operator_conv_perf_summary.csv"
    out_pdf = dirs["paper"] / "fig7_operator_conv_perf.pdf"
    out_png = dirs["paper"] / "fig7_operator_conv_perf.png"
    write_csv(summary_csv, rows)
    _plot(rows, out_pdf, out_png)

    outputs = {
        "raw_csv": str(csv_path),
        "summary_csv": str(summary_csv),
        "figure_pdf": str(out_pdf),
        "figure_png": str(out_png),
    }
    write_manifest(
        run_root,
        artifact=ARTIFACT,
        source_script=SOURCE_SCRIPT,
        command=command,
        outputs=outputs,
        measurement="operator-level Conv2d hot runtime on AMD/Lattigo; rotations are runtime rotation eval counts",
    )
    print_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
