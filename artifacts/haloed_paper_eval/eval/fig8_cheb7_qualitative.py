#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    REPO_ROOT,
    add_common_args,
    clean_number,
    ensure_layout,
    maybe_existing_artifact_root,
    print_outputs,
    read_json,
    resolve_run_root,
    run_command,
    write_csv,
    write_manifest,
)


ARTIFACT = "fig8"
SOURCE_SCRIPT = "tools/verify_medseg_cheb7_orion_clear_adapter.py"
DEFAULT_CASES = (("covid19", 256, 1937), ("nusetmsb", 384, 117))
ALL_SIX_CASES = (
    ("covid19", 256, 1937),
    ("covid19", 256, 1946),
    ("covid19", 256, 1948),
    ("nusetmsb", 384, 117),
    ("nusetmsb", 384, 118),
    ("nusetmsb", 384, 119),
)


def _case_name(dataset: str, size: int, val_index: int) -> str:
    return f"{dataset}_{size}_val{val_index}"


def _build_command(out_dir: Path, dataset: str, size: int, val_index: int, *, run_backend: bool) -> list[str | Path]:
    command: list[str | Path] = [
        sys.executable,
        REPO_ROOT / SOURCE_SCRIPT,
        "--dataset",
        dataset,
        "--image-size",
        str(size),
        "--val-index",
        str(val_index),
        "--threshold",
        "0.5",
        "--out-dir",
        out_dir,
        "--backend-kind",
        "ckks",
        "--backend-atol",
        "5e-3",
        "--backend-rtol",
        "1e-3",
        "--mode",
        "provider",
        "--policy",
        "dp_no_share_fold",
        "--single-slot-layer-cache",
    ]
    if run_backend:
        command.append("--run-backend")
    return command


def _find_case_files(raw_root: Path, dataset: str, size: int, val_index: int) -> tuple[Path, Path]:
    stem = f"{dataset}_{size}_val{val_index}_cheb7_orion_adapter_verify"
    json_matches = sorted(raw_root.rglob(f"{stem}.json"))
    npz_matches = sorted(raw_root.rglob(f"{stem}.npz"))
    if not json_matches or not npz_matches:
        raise FileNotFoundError(f"missing Fig.8 outputs for {dataset} {size} val{val_index} under {raw_root}")
    return json_matches[-1], npz_matches[-1]


def _image2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _render_case(npz_path: Path, out_png: Path, out_pdf: Path, title: str) -> None:
    arrays = np.load(npz_path)
    pred_key = "backend_prob" if "backend_prob" in arrays.files else "adapter_prob"
    mask_key = "backend_mask" if "backend_mask" in arrays.files else "adapter_mask"
    panels = [
        ("Input", _image2d(arrays["image"]), "gray"),
        ("Target", _image2d(arrays["target"]), "gray"),
        ("Cheb7 prob.", _image2d(arrays[pred_key]), "magma"),
        ("Prediction", _image2d(arrays[mask_key]), "gray"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.9))
    fig.suptitle(title, fontsize=9)
    for ax, (label, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(label, fontsize=7)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _render_combined(case_pngs: list[Path], out_png: Path, out_pdf: Path) -> None:
    images = [plt.imread(path) for path in case_pngs]
    fig, axes = plt.subplots(len(images), 1, figsize=(7.1, 2.0 * len(images)))
    if len(images) == 1:
        axes = [axes]
    for ax, image in zip(axes, images):
        ax.imshow(image)
        ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _summary_row(json_path: Path, dataset: str, size: int, val_index: int) -> dict[str, Any]:
    payload = read_json(json_path)
    comparisons = payload.get("comparisons") or {}
    backend = payload.get("backend") or {}
    outputs = payload.get("outputs") or {}
    metrics = outputs.get("backend") or outputs.get("adapter") or outputs.get("reference") or {}
    return {
        "dataset": dataset,
        "image_size": size,
        "val_index": val_index,
        "status": payload.get("status", ""),
        "json": str(json_path),
        "dice": metrics.get("dice", ""),
        "iou": metrics.get("iou", ""),
        "adapter_vs_backend_prob_mae": clean_number(((comparisons.get("adapter_vs_backend") or {}).get("prob") or {}).get("mae", 0.0)),
        "adapter_vs_backend_prob_max_abs": clean_number(((comparisons.get("adapter_vs_backend") or {}).get("prob") or {}).get("max_abs", 0.0)),
        "backend_timing_s": backend.get("timing_s", ""),
        "input_ciphertexts": backend.get("input_ciphertext_count", ""),
        "output_ciphertexts": backend.get("output_ciphertext_count", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HaloED paper Fig. 8 Cheb7 qualitative artifacts.")
    add_common_args(parser)
    parser.add_argument("--all-six", action="store_true", help="Run all six fixed-index audit cases instead of the two paper panels.")
    parser.add_argument("--skip-backend", action="store_true", help="Do not run CKKS backend; useful for quick adapter-only panel checks.")
    args = parser.parse_args()
    cases = ALL_SIX_CASES if args.all_six else DEFAULT_CASES

    run_root = maybe_existing_artifact_root(resolve_run_root(args, ARTIFACT), ARTIFACT) if args.check_existing else resolve_run_root(args, ARTIFACT)
    dirs = ensure_layout(run_root)
    raw_root = dirs["raw"] / "cheb7"

    case_pngs: list[Path] = []
    case_pdfs: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    commands: list[list[str | Path]] = []
    for dataset, size, val_index in cases:
        case = _case_name(dataset, size, val_index)
        out_dir = raw_root / case
        command = _build_command(out_dir, dataset, size, val_index, run_backend=not bool(args.skip_backend))
        commands.append(command)
        if not args.check_existing:
            rc = run_command(command, log_path=dirs["logs"] / f"{case}.log", dry_run=bool(args.dry_run))
            if rc != 0:
                return rc
    if args.dry_run:
        return 0

    for dataset, size, val_index in cases:
        case = _case_name(dataset, size, val_index)
        json_path, npz_path = _find_case_files(raw_root if raw_root.exists() else run_root, dataset, size, val_index)
        panel_png = dirs["paper"] / f"{case}_panel.png"
        panel_pdf = dirs["paper"] / f"{case}_panel.pdf"
        _render_case(npz_path, panel_png, panel_pdf, f"{dataset} {size} val{val_index}")
        case_pngs.append(panel_png)
        case_pdfs.append(panel_pdf)
        row = _summary_row(json_path, dataset, size, val_index)
        row["npz"] = str(npz_path)
        row["panel_png"] = str(panel_png)
        row["panel_pdf"] = str(panel_pdf)
        summary_rows.append(row)

    summary_csv = dirs["paper"] / "fig8_cheb7_metrics_summary.csv"
    combined_png = dirs["paper"] / "fig8_segmentation_results.png"
    combined_pdf = dirs["paper"] / "fig8_segmentation_results.pdf"
    write_csv(summary_csv, summary_rows)
    _render_combined(case_pngs, combined_png, combined_pdf)
    outputs = {
        "summary_csv": str(summary_csv),
        "combined_png": str(combined_png),
        "combined_pdf": str(combined_pdf),
        "panels": [str(path) for path in case_pngs],
        "panel_pdfs": [str(path) for path in case_pdfs],
    }
    write_manifest(
        run_root,
        artifact=ARTIFACT,
        source_script=SOURCE_SCRIPT,
        command=commands[0] if commands else [],
        outputs=outputs,
        measurement="fixed-index Cheb7 scale fine-tuned medseg examples; default backend is CKKS provider dp_no_share_fold single-slot",
        extra={"commands": [" ".join(str(part) for part in command) for command in commands]},
    )
    print_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
