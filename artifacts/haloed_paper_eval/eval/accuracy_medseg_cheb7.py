#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    write_json,
    write_manifest,
)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_fhelipe_medseg_unet22 import FhelipeSegmentationDataset, SPECS, batch_metrics, segmentation_loss
from tools.train_fhelipe_medseg_staged_unet import make_stage_model
from tools.verify_medseg_cheb7_orion_clear_adapter import _build_training_reference_model


ARTIFACT = "accuracy"
SOURCE_SCRIPT = "tools/verify_medseg_cheb7_orion_clear_adapter.py"
SUBSET_VAL_LIMIT = 512
SUBSET_VAL_SEED = 1


@dataclass(frozen=True)
class AccuracyTask:
    key: str
    label: str
    dataset: str
    image_size: int
    subset_start: int
    plain_checkpoint: Path
    scaled_checkpoint: Path
    cheb_checkpoint: Path
    replacements: Path


TASKS = {
    "covid19": AccuracyTask(
        key="covid19",
        label="COVID-256",
        dataset="covid19",
        image_size=256,
        subset_start=465,
        plain_checkpoint=REPO_ROOT
        / "checkpoints"
        / "fhelipe_medseg_unet22_plus_output_relu_silu_20260603"
        / "covid19_unet22_plus_output_base32_256_silu_avgpool_retrain_best.pt",
        scaled_checkpoint=REPO_ROOT
        / "checkpoints"
        / "fhelipe_medseg_staged_covid19_256_scaled_silu_freeze15_cheb7_20260603"
        / "covid19_unet22_plus_output_base32_256_scaled_silu_avgpool_retrain_best.pt",
        cheb_checkpoint=REPO_ROOT
        / "checkpoints"
        / "fhelipe_medseg_staged_covid19_256_scaled_silu_freeze15_cheb7_20260603"
        / "covid19_unet22_plus_output_base32_256_scaled_silu_avgpool_degree_7_rawgain_tight_g045_finetune_best.pt",
        replacements=REPO_ROOT
        / "artifacts"
        / "haloed_paper_eval"
        / "fixtures"
        / "fhelipe_medseg_covid19_unet22_plus_output_256_base32_scaled_silu_freeze15_cheb7_finetune20_20260603.json",
    ),
    "nusetmsb": AccuracyTask(
        key="nusetmsb",
        label="NuSeg-384",
        dataset="nusetmsb",
        image_size=384,
        subset_start=117,
        plain_checkpoint=REPO_ROOT
        / "checkpoints"
        / "fhelipe_medseg_unet22_plus_output_relu_silu_20260603"
        / "nusetmsb_unet22_plus_output_base32_384_silu_avgpool_retrain_best.pt",
        scaled_checkpoint=REPO_ROOT
        / "checkpoints"
        / "fhelipe_medseg_staged_nusetmsb_384_scaled_silu_freeze15_cheb7_20260603"
        / "nusetmsb_unet22_plus_output_base32_384_scaled_silu_avgpool_retrain_best.pt",
        cheb_checkpoint=REPO_ROOT
        / "checkpoints"
        / "fhelipe_medseg_staged_nusetmsb_384_scaled_silu_freeze15_cheb7_20260603"
        / "nusetmsb_unet22_plus_output_base32_384_scaled_silu_avgpool_degree_7_dec4a1536_rawgain_g045_finetune_best.pt",
        replacements=REPO_ROOT
        / "artifacts"
        / "haloed_paper_eval"
        / "fixtures"
        / "fhelipe_medseg_nusetmsb_unet22_plus_output_384_base32_scaled_silu_freeze15_cheb7_finetune20_20260603.json",
    ),
}


def _select_tasks(values: list[str]) -> list[AccuracyTask]:
    if not values or "all" in values:
        return [TASKS["covid19"], TASKS["nusetmsb"]]
    return [TASKS[value] for value in values]


def _normalize_counts(values: list[int]) -> list[int]:
    counts = sorted({int(value) for value in values})
    if not counts or counts[0] <= 0:
        raise ValueError("--counts must contain positive integers")
    return counts


def _subset_dataset(task: AccuracyTask, *, data_root: Path) -> tuple[FhelipeSegmentationDataset, list[int]]:
    spec = SPECS[task.dataset]
    npz_path = Path(data_root) / spec.filename
    dataset = FhelipeSegmentationDataset(
        npz_path,
        image_key=spec.val_image_key,
        label_key=spec.val_label_key,
        image_size=int(task.image_size),
        limit=SUBSET_VAL_LIMIT,
        seed=SUBSET_VAL_SEED,
    )
    with np.load(npz_path) as data:
        full_count = int(data[spec.val_image_key].shape[0])
        fixture_indices = (
            data["accuracy_original_val_indices"].astype(np.int64).tolist()
            if "accuracy_original_val_indices" in data.files
            else None
        )
    if fixture_indices is not None:
        return dataset, [int(value) for value in fixture_indices]
    if 0 < SUBSET_VAL_LIMIT < full_count:
        rng = np.random.default_rng(SUBSET_VAL_SEED)
        original_indices = np.sort(rng.choice(full_count, size=SUBSET_VAL_LIMIT, replace=False)).astype(np.int64).tolist()
    else:
        original_indices = list(range(full_count))
    return dataset, [int(value) for value in original_indices]


def _subset_start_for(task: AccuracyTask, dataset: FhelipeSegmentationDataset) -> int:
    return int(task.subset_start) if int(task.subset_start) < len(dataset) else 0


def _load_stage_checkpoint(path: Path, *, device: torch.device | str) -> torch.nn.Module:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model_cfg = dict(checkpoint.get("model", {}) or {})
    activation = str(model_cfg.get("activation", "plain-silu")).replace("_", "-")
    model = make_stage_model(
        architecture=str(model_cfg.get("architecture", "unet22-plus-output")),
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_dim=int(model_cfg.get("base_dim", 32)),
        activation=activation,
        pool=str(model_cfg.get("pool", "avg")),
        scale_margin=2.0 if activation == "scaled-silu" else 1.0,
        silu_degree=int(model_cfg.get("silu_degree", 31)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _sample_rows(task: AccuracyTask, *, count: int, data_root: Path) -> list[dict[str, Any]]:
    dataset, original_indices = _subset_dataset(task, data_root=data_root)
    start = _subset_start_for(task, dataset)
    end = int(start) + int(count)
    if end > len(dataset):
        raise IndexError(f"{task.label} requested subset [{start}, {end}) but dataset has {len(dataset)} examples")
    rows = []
    for subset_index in range(int(start), end):
        original_val_index = int(original_indices[subset_index])
        row = {
            "task": task.label,
            "dataset": task.dataset,
            "image_size": int(task.image_size),
            "subset_index": int(subset_index),
            "original_val_index": int(original_val_index),
        }
        row["backend_val_index"] = int(subset_index if int(start) == 0 and original_val_index != subset_index else original_val_index)
        rows.append(row)
    return rows


@torch.no_grad()
def _model_metrics(model: torch.nn.Module, image: torch.Tensor, target: torch.Tensor, *, device: torch.device | str) -> dict[str, float]:
    images = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    targets = target.unsqueeze(0).to(device=device, dtype=torch.float32)
    logits = model(images)
    metrics = batch_metrics(logits, targets)
    return {
        "dice": float(metrics["dice"]),
        "iou": float(metrics["iou"]),
        "loss": float(segmentation_loss(logits, targets).detach().item()),
    }


def _add_drop_metrics(row: dict[str, Any]) -> None:
    for baseline in ("plain_silu", "scaled_silu"):
        dice_key = f"{baseline}_dice"
        iou_key = f"{baseline}_iou"
        loss_key = f"{baseline}_loss"
        if dice_key in row and row.get("cheb_dice", "") != "":
            row[f"{baseline}_dice_drop"] = float(row[dice_key]) - float(row["cheb_dice"])
        if iou_key in row and row.get("cheb_iou", "") != "":
            row[f"{baseline}_iou_drop"] = float(row[iou_key]) - float(row["cheb_iou"])
        if loss_key in row and row.get("cheb_loss", "") != "":
            row[f"{baseline}_loss_delta"] = float(row["cheb_loss"]) - float(row[loss_key])


def _evaluate_pytorch_rows(
    task: AccuracyTask,
    *,
    count: int,
    data_root: Path,
    device: torch.device | str,
    baselines: set[str],
    include_cheb: bool,
) -> list[dict[str, Any]]:
    dataset, original_indices = _subset_dataset(task, data_root=data_root)
    start = _subset_start_for(task, dataset)
    end = int(start) + int(count)
    if end > len(dataset):
        raise IndexError(f"{task.label} requested subset [{start}, {end}) but dataset has {len(dataset)} examples")

    models: dict[str, torch.nn.Module] = {}
    if "plain" in baselines:
        models["plain_silu"] = _load_stage_checkpoint(task.plain_checkpoint, device=device)
    if "scaled" in baselines:
        models["scaled_silu"] = _load_stage_checkpoint(task.scaled_checkpoint, device=device)
    if include_cheb:
        cheb_model, _meta = _build_training_reference_model(task.cheb_checkpoint, replacements_path=task.replacements, device=device)
        models["cheb"] = cheb_model

    rows: list[dict[str, Any]] = []
    for subset_index in range(int(start), end):
        image, target = dataset[subset_index]
        original_val_index = int(original_indices[subset_index])
        row: dict[str, Any] = {
            "task": task.label,
            "dataset": task.dataset,
            "image_size": int(task.image_size),
            "subset_index": int(subset_index),
            "original_val_index": int(original_val_index),
            "backend_val_index": int(subset_index if int(start) == 0 and original_val_index != subset_index else original_val_index),
        }
        for name, model in models.items():
            metrics = _model_metrics(model, image, target, device=device)
            prefix = "cheb" if name == "cheb" else name
            row[f"{prefix}_dice"] = metrics["dice"]
            row[f"{prefix}_iou"] = metrics["iou"]
            row[f"{prefix}_loss"] = metrics["loss"]
        if include_cheb:
            row["cheb_source"] = "pytorch_reference"
            _add_drop_metrics(row)
        rows.append(row)
    return rows


def _case_name(task: AccuracyTask, original_val_index: int) -> str:
    return f"{task.dataset}_{task.image_size}_val{int(original_val_index)}"


def _backend_command(
    *,
    out_dir: Path,
    task: AccuracyTask,
    data_root: Path,
    original_val_index: int,
    run_kind: str,
    threshold: float,
    backend_mode: str,
    policy: str,
    provider_mode: str | None,
    single_slot_layer_cache: bool,
    backend_atol: float | None,
    backend_rtol: float | None,
    backend_dice_atol: float | None,
) -> list[str | Path]:
    atol = float(backend_atol if backend_atol is not None else (5.0e-3 if run_kind == "ckks" else 1.0e-5))
    rtol = float(backend_rtol if backend_rtol is not None else (1.0e-3 if run_kind == "ckks" else 1.0e-4))
    command: list[str | Path] = [
        sys.executable,
        REPO_ROOT / SOURCE_SCRIPT,
        "--dataset",
        task.dataset,
        "--image-size",
        str(task.image_size),
        "--val-index",
        str(original_val_index),
        "--threshold",
        str(threshold),
        "--checkpoint",
        task.cheb_checkpoint,
        "--replacements",
        task.replacements,
        "--data-root",
        data_root,
        "--out-dir",
        out_dir,
        "--run-backend",
        "--backend-kind",
        run_kind,
        "--backend-atol",
        str(atol),
        "--backend-rtol",
        str(rtol),
        "--mode",
        backend_mode,
        "--policy",
        policy,
    ]
    if provider_mode:
        command.extend(["--provider-mode", provider_mode])
    if single_slot_layer_cache:
        command.append("--single-slot-layer-cache")
    if backend_dice_atol is not None:
        command.extend(["--backend-dice-atol", str(float(backend_dice_atol))])
    return command


def _verifier_stem(task: AccuracyTask, original_val_index: int) -> str:
    return f"{task.dataset}_{task.image_size}_val{int(original_val_index)}_cheb7_orion_adapter_verify"


def _find_verifier_output(out_dir: Path, task: AccuracyTask, original_val_index: int) -> tuple[Path, Path | None]:
    stem = _verifier_stem(task, original_val_index)
    json_path = Path(out_dir) / f"{stem}.json"
    arrays_path = Path(out_dir) / f"{stem}.npz"
    if not json_path.exists():
        matches = sorted(Path(out_dir).rglob(f"{stem}.json"))
        if not matches:
            raise FileNotFoundError(f"missing verifier JSON under {out_dir}: {stem}.json")
        json_path = matches[-1]
    if not arrays_path.exists():
        matches = sorted(Path(out_dir).rglob(f"{stem}.npz"))
        arrays_path = matches[-1] if matches else None
    return json_path, arrays_path if arrays_path and arrays_path.exists() else None


def _loss_from_npz(arrays_path: Path | None, output_name: str) -> float | str:
    if arrays_path is None:
        return ""
    arrays = np.load(arrays_path)
    key = f"{output_name}_logits"
    if key not in arrays.files or "target" not in arrays.files:
        return ""
    logits = torch.from_numpy(np.asarray(arrays[key], dtype=np.float32))
    target = torch.from_numpy(np.asarray(arrays["target"], dtype=np.float32))
    return float(segmentation_loss(logits, target).detach().item())


def _backend_metrics(out_dir: Path, task: AccuracyTask, original_val_index: int) -> dict[str, Any]:
    json_path, arrays_path = _find_verifier_output(out_dir, task, original_val_index)
    payload = read_json(json_path)
    outputs = dict(payload.get("outputs") or {})
    output_name = "backend" if "backend" in outputs else "adapter"
    segmentation = dict((outputs.get(output_name) or {}).get("segmentation") or {})
    comparisons = dict(payload.get("comparisons") or {})
    adapter_vs_backend = dict(comparisons.get("adapter_vs_backend") or {})
    prob = dict(adapter_vs_backend.get("prob") or {})
    mask = dict(adapter_vs_backend.get("mask") or {})
    segmentation_cmp = dict(adapter_vs_backend.get("segmentation") or {})
    backend = dict(payload.get("backend") or {})
    timing = backend.get("timing_s") or {}
    if isinstance(timing, (int, float)):
        timing_s: float | str = float(timing)
    else:
        timing_s = clean_number(dict(timing).get("total", dict(timing).get("total_so_far", "")), default=0.0)
    return {
        "cheb_source": output_name,
        "cheb_dice": segmentation.get("dice", ""),
        "cheb_iou": segmentation.get("iou", ""),
        "cheb_loss": _loss_from_npz(arrays_path, output_name),
        "backend_status": payload.get("status", ""),
        "backend_json": str(json_path),
        "backend_npz": str(arrays_path or ""),
        "backend_timing_s": timing_s,
        "adapter_vs_backend_prob_mae": prob.get("mae", ""),
        "adapter_vs_backend_prob_max_abs": prob.get("max_abs", ""),
        "adapter_vs_backend_mask_changed_fraction": mask.get("changed_fraction", ""),
        "adapter_vs_backend_dice_abs_delta": segmentation_cmp.get("dice_abs_delta", ""),
        "adapter_vs_backend_dice_close": segmentation_cmp.get("dice_close", ""),
        "input_ciphertexts": backend.get("input_ciphertext_count", ""),
        "output_ciphertexts": backend.get("output_ciphertext_count", ""),
    }


def _backend_dice_acceptance(payload: dict[str, Any], backend_dice_atol: float | None) -> dict[str, Any]:
    outputs = dict(payload.get("outputs") or {})
    adapter = dict((outputs.get("adapter") or {}).get("segmentation") or {})
    backend = dict((outputs.get("backend") or {}).get("segmentation") or {})
    adapter_dice = adapter.get("dice")
    backend_dice = backend.get("dice")
    metrics: dict[str, Any] = {
        "adapter_dice": adapter_dice,
        "backend_dice": backend_dice,
        "dice_atol": backend_dice_atol,
        "dice_close": False,
    }
    if adapter_dice is None or backend_dice is None or backend_dice_atol is None:
        return metrics
    dice_abs_delta = abs(float(adapter_dice) - float(backend_dice))
    metrics["dice_abs_delta"] = float(dice_abs_delta)
    metrics["dice_close"] = bool(dice_abs_delta <= float(backend_dice_atol))
    return metrics


def _mark_backend_dice_close(
    out_dir: Path,
    task: AccuracyTask,
    original_val_index: int,
    backend_dice_atol: float | None,
) -> bool:
    try:
        json_path, _ = _find_verifier_output(out_dir, task, original_val_index)
    except FileNotFoundError:
        return False
    payload = read_json(json_path)
    status = str(payload.get("status", ""))
    if status.startswith("ok"):
        return True
    comparisons = dict(payload.get("comparisons") or {})
    reference_vs_adapter = dict(comparisons.get("reference_vs_adapter") or {})
    alignment_ok = bool(dict(reference_vs_adapter.get("logits") or {}).get("allclose")) and bool(
        dict(reference_vs_adapter.get("prob") or {}).get("allclose")
    )
    acceptance = _backend_dice_acceptance(payload, backend_dice_atol)
    if not (alignment_ok and bool(acceptance.get("dice_close", False))):
        return False
    comparisons.setdefault("adapter_vs_backend", {})
    comparisons["adapter_vs_backend"]["segmentation"] = acceptance
    payload["comparisons"] = comparisons
    payload["status"] = "ok_backend_dice_close"
    write_json(json_path, payload)
    print(
        f"accepted existing backend by Dice for {_case_name(task, original_val_index)} "
        f"(abs_delta={float(acceptance['dice_abs_delta']):.6g}, atol={float(backend_dice_atol):.6g})",
        flush=True,
    )
    return True


def _run_backend_rows(
    task: AccuracyTask,
    *,
    count: int,
    data_root: Path,
    dirs: dict[str, Path],
    run_kind: str,
    threshold: float,
    backend_mode: str,
    policy: str,
    provider_mode: str | None,
    single_slot_layer_cache: bool,
    backend_atol: float | None,
    backend_rtol: float | None,
    backend_dice_atol: float | None,
    baselines: set[str],
    dry_run: bool,
    check_existing: bool,
    force: bool,
) -> tuple[list[dict[str, Any]], list[list[str | Path]]]:
    if dry_run:
        baseline_rows = _sample_rows(task, count=count, data_root=data_root)
    else:
        baseline_rows = _evaluate_pytorch_rows(
            task,
            count=count,
            data_root=data_root,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            baselines=baselines,
            include_cheb=False,
        )
    commands: list[list[str | Path]] = []
    rows: list[dict[str, Any]] = []
    raw_root = dirs["raw"] / "accuracy" / run_kind
    for baseline_row in baseline_rows:
        original_val_index = int(baseline_row["original_val_index"])
        backend_val_index = int(baseline_row.get("backend_val_index", original_val_index))
        case = _case_name(task, original_val_index)
        out_dir = raw_root / case
        command = _backend_command(
            out_dir=out_dir,
            task=task,
            data_root=Path(data_root),
            original_val_index=backend_val_index,
            run_kind=run_kind,
            threshold=float(threshold),
            backend_mode=str(backend_mode),
            policy=str(policy),
            provider_mode=provider_mode,
            single_slot_layer_cache=bool(single_slot_layer_cache),
            backend_atol=backend_atol,
            backend_rtol=backend_rtol,
            backend_dice_atol=backend_dice_atol,
        )
        commands.append(command)
        use_existing = False
        if not dry_run and not force:
            use_existing = _mark_backend_dice_close(out_dir, task, backend_val_index, backend_dice_atol)
        if not check_existing and not use_existing:
            rc = run_command(command, log_path=dirs["logs"] / f"accuracy_{run_kind}_{case}.log", dry_run=dry_run)
            if rc != 0 and not _mark_backend_dice_close(out_dir, task, backend_val_index, backend_dice_atol):
                raise RuntimeError(f"{run_kind} verifier failed for {case} with exit code {rc}")
        if dry_run:
            continue
        row = copy.deepcopy(baseline_row)
        row["run_kind"] = run_kind
        row.update(_backend_metrics(out_dir, task, backend_val_index))
        _add_drop_metrics(row)
        rows.append(row)
    return rows, commands


def _summary_rows(rows: list[dict[str, Any]], *, counts: list[int], tasks: list[AccuracyTask], run_kind: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    numeric_fields = [
        "plain_silu_dice",
        "plain_silu_iou",
        "plain_silu_loss",
        "scaled_silu_dice",
        "scaled_silu_iou",
        "scaled_silu_loss",
        "cheb_dice",
        "cheb_iou",
        "cheb_loss",
        "plain_silu_dice_drop",
        "plain_silu_iou_drop",
        "plain_silu_loss_delta",
        "scaled_silu_dice_drop",
        "scaled_silu_iou_drop",
        "scaled_silu_loss_delta",
        "backend_timing_s",
        "adapter_vs_backend_prob_mae",
        "adapter_vs_backend_prob_max_abs",
        "adapter_vs_backend_mask_changed_fraction",
        "adapter_vs_backend_dice_abs_delta",
    ]
    for task in tasks:
        task_rows = [row for row in rows if row.get("dataset") == task.dataset and int(row.get("image_size", 0)) == task.image_size]
        task_rows.sort(key=lambda row: int(row["subset_index"]))
        for count in counts:
            selected = task_rows[: int(count)]
            if len(selected) < int(count):
                continue
            summary: dict[str, Any] = {
                "run_kind": run_kind,
                "task": task.label,
                "dataset": task.dataset,
                "image_size": int(task.image_size),
                "sample_count": int(count),
                "subset_indices": ",".join(str(row["subset_index"]) for row in selected),
                "original_val_indices": ",".join(str(row["original_val_index"]) for row in selected),
            }
            for field in numeric_fields:
                values = [clean_number(row.get(field, ""), default=float("nan")) for row in selected if row.get(field, "") != ""]
                values = [value for value in values if np.isfinite(value)]
                if values:
                    summary[field] = float(sum(values) / len(values))
            summaries.append(summary)
    return summaries


def _write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# MedSeg Cheb7 Accuracy Summary",
        "",
        "Drop columns are baseline minus Cheb7; positive means Cheb7 is lower.",
        "",
        "| Run | Task | N | Plain Dice | Cheb Dice | Drop vs plain | Scaled Dice | Drop vs scaled |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {run_kind} | {task} | {sample_count} | {plain:.6f} | {cheb:.6f} | {plain_drop:.6f} | {scaled:.6f} | {scaled_drop:.6f} |".format(
                run_kind=row.get("run_kind", ""),
                task=row.get("task", ""),
                sample_count=int(row.get("sample_count", 0)),
                plain=clean_number(row.get("plain_silu_dice", "")),
                cheb=clean_number(row.get("cheb_dice", "")),
                plain_drop=clean_number(row.get("plain_silu_dice_drop", "")),
                scaled=clean_number(row.get("scaled_silu_dice", "")),
                scaled_drop=clean_number(row.get("scaled_silu_dice_drop", "")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_pytorch(
    *,
    tasks: list[AccuracyTask],
    max_count: int,
    data_root: Path,
    device: torch.device | str,
    baselines: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_rows = _evaluate_pytorch_rows(
            task,
            count=max_count,
            data_root=data_root,
            device=device,
            baselines=baselines,
            include_cheb=True,
        )
        for row in task_rows:
            row["run_kind"] = "pytorch"
        rows.extend(task_rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce fixed-index medseg Cheb7 accuracy for PyTorch, cleartext backend, or CKKS backend."
    )
    add_common_args(parser)
    parser.add_argument("--run-kind", choices=("pytorch", "clear", "ckks", "all"), default="pytorch")
    parser.add_argument("--tasks", nargs="+", choices=("all", "covid19", "nusetmsb"), default=["all"])
    parser.add_argument("--counts", nargs="+", type=int, default=[10])
    parser.add_argument("--baseline", choices=("plain", "scaled", "both"), default="both")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "fhelipe_medseg")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--backend-mode", choices=("provider", "dense"), default="provider")
    parser.add_argument("--policy", default="dp_no_share_fold")
    parser.add_argument("--provider-mode", default=None)
    parser.add_argument("--backend-atol", type=float, default=None)
    parser.add_argument("--backend-rtol", type=float, default=None)
    parser.add_argument("--backend-dice-atol", type=float, default=1.0e-3)
    parser.add_argument("--no-single-slot-layer-cache", dest="single_slot_layer_cache", action="store_false")
    parser.set_defaults(single_slot_layer_cache=True)
    args = parser.parse_args()

    selected_tasks = _select_tasks(list(args.tasks))
    counts = _normalize_counts(list(args.counts))
    max_count = max(counts)
    baselines = {"plain", "scaled"} if args.baseline == "both" else {str(args.baseline)}
    run_kinds = ["pytorch", "clear", "ckks"] if args.run_kind == "all" else [str(args.run_kind)]

    run_root = (
        maybe_existing_artifact_root(resolve_run_root(args, ARTIFACT), ARTIFACT)
        if args.check_existing
        else resolve_run_root(args, ARTIFACT)
    )
    dirs = ensure_layout(run_root)

    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    commands: list[list[str | Path]] = []
    for run_kind in run_kinds:
        if run_kind == "pytorch":
            if args.dry_run:
                continue
            rows = _run_pytorch(
                tasks=selected_tasks,
                max_count=max_count,
                data_root=Path(args.data_root),
                device=torch.device(str(args.device)),
                baselines=baselines,
            )
        else:
            rows = []
            backend_commands = []
            for task in selected_tasks:
                task_rows, task_commands = _run_backend_rows(
                    task,
                    count=max_count,
                    data_root=Path(args.data_root),
                    dirs=dirs,
                    run_kind=run_kind,
                    threshold=float(args.threshold),
                    backend_mode=str(args.backend_mode),
                    policy=str(args.policy),
                    provider_mode=args.provider_mode,
                    single_slot_layer_cache=bool(args.single_slot_layer_cache),
                    backend_atol=args.backend_atol,
                    backend_rtol=args.backend_rtol,
                    backend_dice_atol=args.backend_dice_atol,
                    baselines=baselines,
                    dry_run=bool(args.dry_run),
                    check_existing=bool(args.check_existing),
                    force=bool(args.force),
                )
                rows.extend(task_rows)
                backend_commands.extend(task_commands)
            commands.extend(backend_commands)
        if args.dry_run:
            continue
        all_rows.extend(rows)
        all_summaries.extend(_summary_rows(rows, counts=counts, tasks=selected_tasks, run_kind=run_kind))

    if args.dry_run:
        return 0

    raw_json = dirs["raw"] / "accuracy" / "accuracy_rows.json"
    summary_json = dirs["paper"] / "accuracy_summary.json"
    rows_csv = dirs["paper"] / "accuracy_by_sample.csv"
    summary_csv = dirs["paper"] / "accuracy_summary.csv"
    summary_md = dirs["paper"] / "accuracy_summary.md"
    write_json(
        raw_json,
        {
            "run_kinds": run_kinds,
            "counts": counts,
            "subset_val_limit": SUBSET_VAL_LIMIT,
            "subset_val_seed": SUBSET_VAL_SEED,
            "rows": all_rows,
        },
    )
    write_json(summary_json, {"summaries": all_summaries})
    write_csv(rows_csv, all_rows)
    write_csv(summary_csv, all_summaries)
    _write_markdown(summary_md, all_summaries)
    outputs = {
        "raw_json": str(raw_json),
        "summary_json": str(summary_json),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
    }
    write_manifest(
        run_root,
        artifact=ARTIFACT,
        source_script=SOURCE_SCRIPT,
        command=[sys.executable, Path(__file__).resolve(), "--run-kind", str(args.run_kind)],
        outputs=outputs,
        measurement=(
            "fixed validation subset medseg Cheb7 accuracy; subset uses val_limit=512 and val_seed=1; "
            "drop is baseline Dice/IoU minus Cheb7"
        ),
        extra={
            "counts": counts,
            "tasks": [task.key for task in selected_tasks],
            "run_kinds": run_kinds,
            "backend_commands": [" ".join(str(part) for part in command) for command in commands],
        },
    )
    print_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
