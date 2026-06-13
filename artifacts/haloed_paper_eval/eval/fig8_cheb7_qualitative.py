#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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
FIXTURE_DATA_ROOT = REPO_ROOT / "artifacts" / "haloed_paper_eval" / "fixtures" / "fhelipe_medseg_accuracy10"
FIXTURE_FILES = {"covid19": "covid19radio_512.npz", "nusetmsb": "nuset_512.npz"}
DEFAULT_CHECKPOINTS = {
    "covid19": REPO_ROOT
    / "checkpoints"
    / "fhelipe_medseg_staged_covid19_256_scaled_silu_freeze15_cheb7_20260603"
    / "covid19_unet22_plus_output_base32_256_scaled_silu_avgpool_degree_7_rawgain_tight_g045_finetune_best.pt",
    "nusetmsb": REPO_ROOT
    / "checkpoints"
    / "fhelipe_medseg_staged_nusetmsb_384_scaled_silu_freeze15_cheb7_20260603"
    / "nusetmsb_unet22_plus_output_base32_384_scaled_silu_avgpool_degree_7_dec4a1536_rawgain_g045_finetune_best.pt",
}
DEFAULT_REPLACEMENTS = {
    "covid19": REPO_ROOT
    / "artifacts"
    / "haloed_paper_eval"
    / "fixtures"
    / "fhelipe_medseg_covid19_unet22_plus_output_256_base32_scaled_silu_freeze15_cheb7_finetune20_20260603.json",
    "nusetmsb": REPO_ROOT
    / "artifacts"
    / "haloed_paper_eval"
    / "fixtures"
    / "fhelipe_medseg_nusetmsb_unet22_plus_output_384_base32_scaled_silu_freeze15_cheb7_finetune20_20260603.json",
}
BACKEND_ENV_DEFAULTS = {
    "ORION_PROVIDER_MVM_MASKED_MATERIALIZATION": "1",
    "ORION_BOOTSTRAP_LAYOUT_REFINEMENT": "0",
    "ORION_LAYOUT_POLICY_RELAYOUT_KERNEL": "0",
    "ORION_CONCAT_FUSION": "0",
}
DEFAULT_CASES = (("covid19", 256, 1951), ("nusetmsb", 384, 120))
ALL_SIX_CASES = (
    ("covid19", 256, 1937),
    ("covid19", 256, 1946),
    ("covid19", 256, 1951),
    ("nusetmsb", 384, 117),
    ("nusetmsb", 384, 118),
    ("nusetmsb", 384, 120),
)


def _case_name(dataset: str, size: int, original_val_index: int, backend_val_index: int) -> str:
    if int(original_val_index) == int(backend_val_index):
        return f"{dataset}_{size}_val{backend_val_index}"
    return f"{dataset}_{size}_orig{original_val_index}_val{backend_val_index}"


def _resolve_backend_val_index(dataset: str, data_root: Path, original_val_index: int) -> tuple[int, bool]:
    npz_path = Path(data_root) / FIXTURE_FILES[str(dataset)]
    if not npz_path.exists():
        return int(original_val_index), False
    with np.load(npz_path) as data:
        if "accuracy_original_val_indices" not in data.files:
            return int(original_val_index), False
        original_indices = [int(value) for value in data["accuracy_original_val_indices"].astype(np.int64).tolist()]
    try:
        return int(original_indices.index(int(original_val_index))), True
    except ValueError as exc:
        raise ValueError(
            f"{dataset} original validation ID {int(original_val_index)} is not present in fixture {npz_path}"
        ) from exc


def _build_command(
    out_dir: Path,
    dataset: str,
    size: int,
    backend_val_index: int,
    *,
    data_root: Path,
    run_backend: bool,
    backend_bootstrap_many: str,
) -> list[str | Path]:
    command: list[str | Path] = [
        sys.executable,
        REPO_ROOT / SOURCE_SCRIPT,
        "--dataset",
        dataset,
        "--image-size",
        str(size),
        "--val-index",
        str(backend_val_index),
        "--threshold",
        "0.5",
        "--checkpoint",
        DEFAULT_CHECKPOINTS[dataset],
        "--replacements",
        DEFAULT_REPLACEMENTS[dataset],
        "--data-root",
        data_root,
        "--out-dir",
        out_dir,
        "--backend-kind",
        "ckks",
        "--backend-atol",
        "5e-3",
        "--backend-rtol",
        "1e-3",
        "--backend-dice-atol",
        "1e-3",
        "--backend-bootstrap-many",
        str(backend_bootstrap_many),
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


def _summary_row(
    json_path: Path,
    dataset: str,
    size: int,
    original_val_index: int,
    backend_val_index: int,
    *,
    fixture_resolved: bool,
) -> dict[str, Any]:
    payload = read_json(json_path)
    comparisons = payload.get("comparisons") or {}
    backend = payload.get("backend") or {}
    outputs = payload.get("outputs") or {}
    metrics = outputs.get("backend") or outputs.get("adapter") or outputs.get("reference") or {}
    return {
        "dataset": dataset,
        "image_size": size,
        "original_val_index": original_val_index,
        "backend_val_index": backend_val_index,
        "fixture_resolved": fixture_resolved,
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
    parser.add_argument("--data-root", type=Path, default=FIXTURE_DATA_ROOT)
    parser.add_argument(
        "--backend-bootstrap-many",
        choices=("0", "1"),
        default="1",
        help="CKKS bootstrap execution scheduling flag passed through to the verifier; it is not a correctness knob.",
    )
    args = parser.parse_args()
    cases = ALL_SIX_CASES if args.all_six else DEFAULT_CASES
    for key, value in BACKEND_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    run_root = maybe_existing_artifact_root(resolve_run_root(args, ARTIFACT), ARTIFACT) if args.check_existing else resolve_run_root(args, ARTIFACT)
    dirs = ensure_layout(run_root)
    raw_root = dirs["raw"] / "cheb7"

    case_pngs: list[Path] = []
    case_pdfs: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    commands: list[list[str | Path]] = []
    resolved_cases: list[dict[str, Any]] = []
    for dataset, size, original_val_index in cases:
        backend_val_index, fixture_resolved = _resolve_backend_val_index(
            dataset, Path(args.data_root), int(original_val_index)
        )
        resolved_cases.append(
            {
                "dataset": dataset,
                "size": int(size),
                "original_val_index": int(original_val_index),
                "backend_val_index": int(backend_val_index),
                "fixture_resolved": bool(fixture_resolved),
            }
        )
        case = _case_name(dataset, size, int(original_val_index), int(backend_val_index))
        out_dir = raw_root / case
        command = _build_command(
            out_dir,
            dataset,
            size,
            backend_val_index,
            data_root=Path(args.data_root),
            run_backend=not bool(args.skip_backend),
            backend_bootstrap_many=str(args.backend_bootstrap_many),
        )
        commands.append(command)
        if not args.check_existing:
            rc = run_command(command, log_path=dirs["logs"] / f"{case}.log", dry_run=bool(args.dry_run))
            if rc != 0:
                return rc
    if args.dry_run:
        return 0

    for resolved in resolved_cases:
        dataset = str(resolved["dataset"])
        size = int(resolved["size"])
        original_val_index = int(resolved["original_val_index"])
        backend_val_index = int(resolved["backend_val_index"])
        case = _case_name(dataset, size, original_val_index, backend_val_index)
        json_path, npz_path = _find_case_files(raw_root if raw_root.exists() else run_root, dataset, size, backend_val_index)
        panel_png = dirs["paper"] / f"{case}_panel.png"
        panel_pdf = dirs["paper"] / f"{case}_panel.pdf"
        _render_case(npz_path, panel_png, panel_pdf, f"{dataset} {size} val{original_val_index}")
        case_pngs.append(panel_png)
        case_pdfs.append(panel_pdf)
        row = _summary_row(
            json_path,
            dataset,
            size,
            original_val_index,
            backend_val_index,
            fixture_resolved=bool(resolved["fixture_resolved"]),
        )
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
        measurement=(
            "fixed-index low-postscale Cheb7 medseg examples; default backend is CKKS provider "
            "dp_no_share_fold single-slot; fixture original IDs are resolved to verifier val-index values"
        ),
        extra={
            "commands": [" ".join(str(part) for part in command) for command in commands],
            "resolved_cases": resolved_cases,
            "backend_env_defaults": BACKEND_ENV_DEFAULTS,
            "backend_bootstrap_many": str(args.backend_bootstrap_many),
            "data_root": str(Path(args.data_root)),
        },
    )
    print_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
