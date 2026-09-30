#!/usr/bin/env python3
"""Fine-tune and validate a Cheb7 U-Net under WPC Rotation Padding.

This stage starts from the trained medseg checkpoint, converts every eligible
3x3 Conv2d to differentiable flattened Rotation Padding, records the immediate
accuracy/semantic change, fine-tunes on the original segmentation data, and
compares the best WPC model with both labels and the native zero-padded model.

The output checkpoint preserves the source state-dict keys and can be loaded
back into the ordinary Orion U-Net model.  This runner performs clear PyTorch
training and validation only; it does not run FHE inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.experimental.wpc_rotation_padding_training import (
    WPCRotationPaddingConv2d,
    convert_model_to_wpc_rotation_padding,
    rotation_padding_module_names,
)
from tools.medseg_cheb7_orion_adapter import (
    CheckpointScaledChebyshevSiLU,
    build_orion_cheb7_model_from_checkpoint,
)
from tools.train_fhelipe_medseg_unet22 import (
    FhelipeSegmentationDataset,
    SPECS,
    batch_metrics,
    make_loader,
    segmentation_loss,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/fhelipe_medseg_staged_covid19_256_scaled_silu_freeze15_cheb7_20260603/"
    "covid19_unet22_plus_output_base32_256_scaled_silu_avgpool_degree_7_"
    "rawgain_tight_g045_finetune_best.pt"
)
DEFAULT_OUT_DIR = REPO_ROOT / "checkpoints/wpc_rotation_padding_covid19_cheb7"
DEFAULT_RESULT = (
    REPO_ROOT
    / ".tmp/results/honours/22_wpc_rotation_padding_finetune/"
    "rotation_padding_finetune.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", choices=sorted(SPECS), default="covid19")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data/fhelipe_medseg",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=0.05)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--resume-if-present",
        action="store_true",
        help="Resume from <out-dir>/rotation_padding_last.pt when it exists.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate <out-dir>/rotation_padding_best.pt without training.",
    )
    return parser


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_metrics(metrics: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    native_reference: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    if native_reference is not None:
        native_reference.eval()
    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "logits_mae_vs_native": 0.0,
        "prob_mae_vs_native": 0.0,
        "prediction_flip_rate_vs_native": 0.0,
    }
    maximum_logit_delta = 0.0
    items = 0
    for images, masks in tqdm(loader, desc="validate", leave=False):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        metrics = batch_metrics(logits, masks)
        batch = int(images.shape[0])
        totals["loss"] += float(loss.item()) * batch
        totals["dice"] += float(metrics["dice"]) * batch
        totals["iou"] += float(metrics["iou"]) * batch
        if native_reference is not None:
            native_logits = native_reference(images)
            delta = (logits - native_logits).abs()
            totals["logits_mae_vs_native"] += float(delta.mean().item()) * batch
            maximum_logit_delta = max(
                maximum_logit_delta,
                float(delta.max().item()),
            )
            probabilities = torch.sigmoid(logits)
            native_probabilities = torch.sigmoid(native_logits)
            totals["prob_mae_vs_native"] += float(
                (probabilities - native_probabilities).abs().mean().item()
            ) * batch
            totals["prediction_flip_rate_vs_native"] += float(
                (
                    (probabilities >= 0.5)
                    != (native_probabilities >= 0.5)
                )
                .to(torch.float32)
                .mean()
                .item()
            ) * batch
        items += batch
    result = {
        "loss": totals["loss"] / max(1, items),
        "dice": totals["dice"] / max(1, items),
        "iou": totals["iou"] / max(1, items),
        "sample_count": int(items),
    }
    if native_reference is not None:
        result.update(
            {
                "logits_mae_vs_native": totals["logits_mae_vs_native"]
                / max(1, items),
                "max_abs_logit_delta_vs_native": float(maximum_logit_delta),
                "prob_mae_vs_native": totals["prob_mae_vs_native"]
                / max(1, items),
                "prediction_flip_rate_vs_native": totals[
                    "prediction_flip_rate_vs_native"
                ]
                / max(1, items),
            }
        )
    return result


def _train_epoch(
    model: nn.Module,
    native_teacher: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
    distill_weight: float,
) -> dict[str, float]:
    model.train(True)
    native_teacher.eval()
    totals = {"loss": 0.0, "segmentation_loss": 0.0, "distillation_loss": 0.0}
    items = 0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        segmentation = segmentation_loss(logits, masks)
        if float(distill_weight) > 0.0:
            with torch.no_grad():
                native_logits = native_teacher(images)
            distillation = F.mse_loss(logits, native_logits)
        else:
            distillation = torch.zeros((), device=device, dtype=logits.dtype)
        loss = segmentation + float(distill_weight) * distillation
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        if float(grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(grad_clip_norm)
            )
        optimizer.step()
        batch = int(images.shape[0])
        totals["loss"] += float(loss.detach().item()) * batch
        totals["segmentation_loss"] += float(segmentation.detach().item()) * batch
        totals["distillation_loss"] += float(distillation.detach().item()) * batch
        items += batch
    return {
        name: value / max(1, items) for name, value in totals.items()
    } | {"sample_count": int(items)}


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_metrics: dict[str, float],
    history: list[dict[str, Any]],
    source_checkpoint: Path,
    source_sha256: str,
    conversions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_valid": dict(best_metrics),
        "history": list(history),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": str(source_sha256),
        "model": {
            "architecture": "unet22-plus-output",
            "in_channels": 1,
            "out_channels": 1,
            "base_dim": 32,
            "activation": "checkpoint-cheb7",
            "pool": "avg",
            "skip": "concat",
            "output_head": "1x1",
            "padding_semantics": "wpc_flattened_spatial_rotation_padding",
        },
        "rotation_padding_conversions": list(conversions),
    }


def _build_result(
    *,
    args: argparse.Namespace,
    source_checkpoint: Path,
    source_sha256: str,
    data_path: Path,
    train_count: int,
    val_count: int,
    conversions: list[dict[str, Any]],
    conversion_preserved_state: bool,
    activations_fully_polynomial: bool,
    native_metrics: dict[str, float],
    pre_metrics: dict[str, float],
    best_metrics: dict[str, float],
    best_epoch: int,
    completed_epoch: int,
    history: list[dict[str, Any]],
    best_path: Path,
    last_path: Path,
    checkpoint_reload_compatible: bool,
    started: float,
) -> dict[str, Any]:
    acceptance = {
        "source_checkpoint_sha256_recorded": len(source_sha256) == 64,
        "all_18_spatial_convolutions_use_wpc_rotation_padding": len(conversions)
        == 18,
        "conversion_preserved_state_dict_keys_and_values": bool(
            conversion_preserved_state
        ),
        "checkpoint_cheb7_activations_remain_fully_polynomial": bool(
            activations_fully_polynomial
        ),
        "native_validation_metrics_are_finite": _finite_metrics(native_metrics),
        "unfinetuned_rotation_padding_metrics_are_finite": _finite_metrics(
            pre_metrics
        ),
        "best_rotation_padding_metrics_are_finite": _finite_metrics(best_metrics),
        "rotation_padding_semantic_change_was_observed": float(
            pre_metrics.get("max_abs_logit_delta_vs_native", 0.0)
        )
        > 0.0,
        "best_checkpoint_not_worse_than_unfinetuned_rotation_padding": float(
            best_metrics["dice"]
        )
        + 1.0e-12
        >= float(pre_metrics["dice"]),
        "requested_epochs_completed": bool(
            args.eval_only or int(completed_epoch) >= int(args.epochs)
        ),
        "fine_tuned_checkpoint_reloads_with_original_orion_schema": bool(
            checkpoint_reload_compatible
        ),
        "best_and_last_checkpoints_exist": best_path.is_file()
        and last_path.is_file(),
    }
    acceptance["valid"] = bool(all(acceptance.values()))
    return {
        "schema_version": 1,
        "profile": "wpc_rotation_padding_checkpoint_finetune_accuracy",
        "status": "ok" if acceptance["valid"] else "invalid",
        "timing_policy": "clear training and accuracy validation; no FHE latency claim",
        "source_checkpoint": {
            "path": str(source_checkpoint),
            "sha256": str(source_sha256),
        },
        "dataset": {
            "name": str(args.dataset),
            "path": str(data_path),
            "image_size": int(args.image_size),
            "train_count": int(train_count),
            "validation_count": int(val_count),
        },
        "configuration": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip_norm": float(args.grad_clip_norm),
            "distill_weight": float(args.distill_weight),
            "train_limit": int(args.train_limit),
            "val_limit": int(args.val_limit),
            "num_workers": int(args.num_workers),
            "seed": int(args.seed),
            "device": str(args.device),
            "resume_if_present": bool(args.resume_if_present),
            "eval_only": bool(args.eval_only),
        },
        "padding": {
            "semantics": "wpc_flattened_spatial_cyclic",
            "converted_module_count": int(len(conversions)),
            "converted_modules": list(conversions),
            "native_reference_semantics": "pytorch_zero_padding",
        },
        "metrics": {
            "native_zero_padding_checkpoint": native_metrics,
            "rotation_padding_before_finetune": pre_metrics,
            "rotation_padding_best": best_metrics,
            "native_to_unfinetuned_rotation": {
                "dice_delta": float(pre_metrics["dice"] - native_metrics["dice"]),
                "iou_delta": float(pre_metrics["iou"] - native_metrics["iou"]),
            },
            "unfinetuned_to_best_rotation": {
                "dice_delta": float(best_metrics["dice"] - pre_metrics["dice"]),
                "iou_delta": float(best_metrics["iou"] - pre_metrics["iou"]),
            },
            "native_to_best_rotation": {
                "dice_delta": float(best_metrics["dice"] - native_metrics["dice"]),
                "iou_delta": float(best_metrics["iou"] - native_metrics["iou"]),
            },
        },
        "training": {
            "completed_epoch": int(completed_epoch),
            "best_epoch": int(best_epoch),
            "history": list(history),
        },
        "outputs": {
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "result": str(Path(args.result).expanduser().resolve()),
        },
        "limitations": [
            "clear PyTorch fine-tuning and validation rather than FHE inference",
            "accuracy is measured only on the selected FHELIPE dataset split",
            "the best checkpoint may remain epoch zero if fine-tuning does not improve Dice",
            "server timing is training throughput and is not an HE performance result",
        ],
        "wall_s": float(time.time() - started),
        "acceptance": acceptance,
    }


def main() -> int:
    args = _parser().parse_args()
    if int(args.epochs) < 0:
        raise SystemExit("epochs must be nonnegative")
    if int(args.batch_size) <= 0:
        raise SystemExit("batch-size must be positive")
    if float(args.distill_weight) < 0.0:
        raise SystemExit("distill-weight must be nonnegative")
    _set_seed(int(args.seed))
    started = time.time()
    source_checkpoint = args.checkpoint.expanduser().resolve()
    if not source_checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {source_checkpoint}")
    source_sha256 = _sha256(source_checkpoint)
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_state = source_payload.get("state_dict", source_payload)
    blend_rows = {
        name: float(value.detach().cpu().item())
        for name, value in source_state.items()
        if str(name).endswith(".blend_alpha")
    }
    activations_fully_polynomial = bool(
        len(blend_rows) == 18
        and all(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-7)
            for value in blend_rows.values()
        )
    )
    if not activations_fully_polynomial:
        raise RuntimeError(
            "source checkpoint must contain 18 pure-polynomial activations "
            "with blend_alpha=1"
        )
    del source_payload
    spec = SPECS[str(args.dataset)]
    data_path = (args.data_root.expanduser().resolve() / spec.filename)
    if not data_path.is_file():
        raise SystemExit(f"missing dataset: {data_path}")
    device = torch.device(args.device)

    train_set = FhelipeSegmentationDataset(
        data_path,
        image_key=spec.image_key,
        label_key=spec.label_key,
        image_size=int(args.image_size),
        limit=int(args.train_limit),
        seed=int(args.seed),
    )
    validation_set = FhelipeSegmentationDataset(
        data_path,
        image_key=spec.val_image_key,
        label_key=spec.val_label_key,
        image_size=int(args.image_size),
        limit=int(args.val_limit),
        seed=int(args.seed) + 1,
    )
    train_loader = make_loader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    validation_loader = make_loader(
        validation_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )

    native_model, adapter_metadata = build_orion_cheb7_model_from_checkpoint(
        source_checkpoint,
        device=device,
    )
    native_model.eval()
    rotation_model = copy.deepcopy(native_model).to(device)
    for parameter in native_model.parameters():
        parameter.requires_grad_(False)
    before_conversion = {
        name: value.detach().cpu().clone()
        for name, value in rotation_model.state_dict().items()
    }
    conversion_rows = convert_model_to_wpc_rotation_padding(rotation_model)
    conversions = [row.to_dict() for row in conversion_rows]
    after_conversion = rotation_model.state_dict()
    conversion_preserved_state = bool(
        set(before_conversion) == set(after_conversion)
        and all(
            torch.equal(before_conversion[name], after_conversion[name].detach().cpu())
            for name in before_conversion
        )
    )
    if len(rotation_padding_module_names(rotation_model)) != 18:
        raise RuntimeError("expected exactly 18 converted spatial convolutions")
    activation_modules = [
        module
        for module in rotation_model.modules()
        if isinstance(module, CheckpointScaledChebyshevSiLU)
    ]
    if len(activation_modules) != 18:
        raise RuntimeError("expected exactly 18 checkpoint Chebyshev activations")
    if any(parameter.requires_grad for module in activation_modules for parameter in module.parameters()):
        raise RuntimeError("checkpoint activation parameters must remain fixed")

    trainable_parameters = [
        parameter for parameter in rotation_model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    out_dir = args.out_dir.expanduser().resolve()
    best_path = out_dir / "rotation_padding_best.pt"
    last_path = out_dir / "rotation_padding_last.pt"
    result_path = args.result.expanduser().resolve()

    native_metrics = _evaluate(native_model, validation_loader, device=device)
    pre_metrics = _evaluate(
        rotation_model,
        validation_loader,
        device=device,
        native_reference=native_model,
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    completed_epoch = 0
    best_epoch = 0
    best_metrics = dict(pre_metrics)

    if bool(args.eval_only):
        if not best_path.is_file():
            raise SystemExit(f"eval-only checkpoint does not exist: {best_path}")
        saved = torch.load(best_path, map_location="cpu", weights_only=False)
        rotation_model.load_state_dict(saved["state_dict"], strict=True)
        rotation_model.to(device)
        best_epoch = int(saved.get("best_epoch", saved.get("epoch", 0)))
        completed_epoch = int(saved.get("epoch", best_epoch))
        history = list(saved.get("history", []))
        best_metrics = _evaluate(
            rotation_model,
            validation_loader,
            device=device,
            native_reference=native_model,
        )
        if not last_path.is_file():
            _atomic_torch_save(saved, last_path)
    else:
        if bool(args.resume_if_present) and last_path.is_file():
            resumed = torch.load(last_path, map_location="cpu", weights_only=False)
            if str(resumed.get("source_checkpoint_sha256")) != source_sha256:
                raise RuntimeError("resume checkpoint belongs to a different source checkpoint")
            rotation_model.load_state_dict(resumed["state_dict"], strict=True)
            rotation_model.to(device)
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            completed_epoch = int(resumed.get("epoch", 0))
            start_epoch = completed_epoch + 1
            best_epoch = int(resumed.get("best_epoch", 0))
            best_metrics = dict(resumed.get("best_valid", pre_metrics))
            history = list(resumed.get("history", []))
        else:
            initial = _checkpoint_payload(
                model=rotation_model,
                optimizer=optimizer,
                epoch=0,
                best_epoch=0,
                best_metrics=best_metrics,
                history=history,
                source_checkpoint=source_checkpoint,
                source_sha256=source_sha256,
                conversions=conversions,
            )
            _atomic_torch_save(initial, best_path)
            _atomic_torch_save(initial, last_path)

        for epoch in range(start_epoch, int(args.epochs) + 1):
            epoch_started = time.time()
            train_metrics = _train_epoch(
                rotation_model,
                native_model,
                train_loader,
                device=device,
                optimizer=optimizer,
                grad_clip_norm=float(args.grad_clip_norm),
                distill_weight=float(args.distill_weight),
            )
            validation_metrics = _evaluate(
                rotation_model,
                validation_loader,
                device=device,
                native_reference=native_model,
            )
            if not _finite_metrics(validation_metrics):
                raise RuntimeError("non-finite validation metrics")
            row = {
                "epoch": int(epoch),
                "train": train_metrics,
                "validation": validation_metrics,
                "epoch_s": float(time.time() - epoch_started),
            }
            history.append(row)
            completed_epoch = int(epoch)
            if float(validation_metrics["dice"]) >= float(best_metrics["dice"]):
                best_metrics = dict(validation_metrics)
                best_epoch = int(epoch)
                best_payload = _checkpoint_payload(
                    model=rotation_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                    history=history,
                    source_checkpoint=source_checkpoint,
                    source_sha256=source_sha256,
                    conversions=conversions,
                )
                _atomic_torch_save(best_payload, best_path)
            last_payload = _checkpoint_payload(
                model=rotation_model,
                optimizer=optimizer,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                history=history,
                source_checkpoint=source_checkpoint,
                source_sha256=source_sha256,
                conversions=conversions,
            )
            _atomic_torch_save(last_payload, last_path)
            partial_result = {
                "schema_version": 1,
                "profile": "wpc_rotation_padding_checkpoint_finetune_accuracy",
                "status": "running",
                "completed_epoch": int(completed_epoch),
                "requested_epochs": int(args.epochs),
                "best_epoch": int(best_epoch),
                "best_valid": best_metrics,
                "latest": row,
                "outputs": {
                    "best_checkpoint": str(best_path),
                    "last_checkpoint": str(last_path),
                },
            }
            _atomic_json_save(partial_result, result_path)
            print(json.dumps({"event": "epoch_complete", **row}, sort_keys=True), flush=True)

        best_saved = torch.load(best_path, map_location="cpu", weights_only=False)
        rotation_model.load_state_dict(best_saved["state_dict"], strict=True)
        rotation_model.to(device)
        best_epoch = int(best_saved.get("best_epoch", best_saved.get("epoch", 0)))
        best_metrics = _evaluate(
            rotation_model,
            validation_loader,
            device=device,
            native_reference=native_model,
        )

    reload_model, _reload_metadata = build_orion_cheb7_model_from_checkpoint(
        best_path,
        device="cpu",
    )
    checkpoint_reload_compatible = bool(
        set(reload_model.state_dict()) == set(rotation_model.state_dict())
    )
    del reload_model

    result = _build_result(
        args=args,
        source_checkpoint=source_checkpoint,
        source_sha256=source_sha256,
        data_path=data_path,
        train_count=len(train_set),
        val_count=len(validation_set),
        conversions=conversions,
        conversion_preserved_state=conversion_preserved_state,
        activations_fully_polynomial=activations_fully_polynomial,
        native_metrics=native_metrics,
        pre_metrics=pre_metrics,
        best_metrics=best_metrics,
        best_epoch=best_epoch,
        completed_epoch=completed_epoch,
        history=history,
        best_path=best_path,
        last_path=last_path,
        checkpoint_reload_compatible=checkpoint_reload_compatible,
        started=started,
    )
    result["adapter_metadata"] = adapter_metadata
    result["model_accounting"] = {
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in trainable_parameters)
        ),
        "fixed_checkpoint_activation_count": int(len(activation_modules)),
        "converted_spatial_convolution_count": int(len(conversions)),
    }
    _atomic_json_save(result, result_path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    print(f"\nresult: {result_path}", flush=True)
    return 0 if result["acceptance"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
