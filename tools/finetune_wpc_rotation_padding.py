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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.experimental.wpc_rotation_padding_training import (
    WPCRotationPaddingConv2d,
    convert_model_to_wpc_rotation_padding,
    rotation_padding_module_names,
)
from orion.experimental.wpc_evidence_validation import load_checkpoint_bytes, sha256
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-6)
    parser.add_argument(
        "--resume-lr",
        type=float,
        default=None,
        help="Override the optimizer learning rate loaded from a resume checkpoint.",
    )
    parser.add_argument("--max-epoch-retries", type=int, default=4)
    parser.add_argument("--lr-backoff-factor", type=float, default=0.25)
    parser.add_argument("--min-lr", type=float, default=1.0e-10)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=0.001)
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--val-limit", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
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


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    required = ("loss", "dice", "iou")
    return bool(
        metrics.get("fully_finite") is True
        and all(_finite_number(metrics.get(name)) for name in required)
        and 0.0 <= float(metrics["dice"]) <= 1.0
        and 0.0 <= float(metrics["iou"]) <= 1.0
        and float(metrics["loss"]) >= 0.0
        and isinstance(metrics.get("sample_count"), int)
        and not isinstance(metrics.get("sample_count"), bool)
        and metrics["sample_count"] > 0
        and metrics.get("finite_sample_count") == metrics["sample_count"]
        and metrics.get("nonfinite_sample_count") == 0
        and metrics.get("nonfinite_sample_indices") == []
        and metrics.get("metric_scope") == "all_samples"
    )


def _training_audits_complete(
    history: list[dict[str, Any]],
    *,
    completed_epoch: int,
    train_count: int,
) -> bool:
    """Return whether every completed epoch has a full finite training audit."""

    return bool(
        [int(row.get("epoch", -1)) for row in history]
        == list(range(1, int(completed_epoch) + 1))
        and all(
            _finite_metrics(dict(row.get("post_epoch_training_audit", {})))
            and int(
                row.get("post_epoch_training_audit", {}).get("sample_count", -1)
            )
            == int(train_count)
            for row in history
        )
    )


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: Path,
    train_count: int,
) -> None:
    """Reject legacy or incomplete states that are unsafe to resume."""

    schema_version = int(checkpoint.get("schema_version", 0))
    completed_epoch = int(checkpoint.get("epoch", -1))
    history = list(checkpoint.get("history", []))
    if schema_version < 4:
        raise RuntimeError(
            "resume checkpoint predates mandatory full training-set audits "
            f"(schema_version={schema_version}); restart in a fresh output directory"
        )
    if completed_epoch < 0 or not _training_audits_complete(
        history,
        completed_epoch=completed_epoch,
        train_count=int(train_count),
    ):
        raise RuntimeError(
            "resume checkpoint does not contain a complete finite training-set "
            "audit for every completed epoch; restart from an audited checkpoint"
        )
    missing_epoch_checkpoints = [
        checkpoint_path.parent / f"rotation_padding_epoch_{epoch:04d}.pt"
        for epoch in range(0, completed_epoch + 1)
        if not (
            checkpoint_path.parent / f"rotation_padding_epoch_{epoch:04d}.pt"
        ).is_file()
    ]
    if missing_epoch_checkpoints:
        raise RuntimeError(
            "resume checkpoint is missing immutable epoch evidence: "
            + ", ".join(str(path) for path in missing_epoch_checkpoints)
        )


def _validate_evaluated_checkpoint(
    saved: dict[str, Any], *, path: Path, source_sha256: str, train_count: int,
) -> None:
    """Evaluation must not bypass the provenance/audit checks used by resume."""
    _validate_resume_checkpoint(saved, checkpoint_path=path, train_count=train_count)
    if saved.get("source_checkpoint_sha256") != source_sha256:
        raise RuntimeError("evaluated checkpoint belongs to a different source checkpoint")
    if saved.get("model", {}).get("padding_semantics") != "wpc_flattened_spatial_rotation_padding":
        raise RuntimeError("evaluated checkpoint is not a WPC Rotation-Padding checkpoint")
    state = saved.get("state_dict", {})
    if not state or any(not torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("evaluated checkpoint contains missing/non-finite weights")
    # Adapted fine-tuning checkpoints store fixed coeffs and scale buffers,
    # not the source model's log-scale/blend_alpha training parameters.
    coefficients = {key.removesuffix(".coeffs"): value for key, value in state.items() if key.endswith(".coeffs")}
    if len(coefficients) != 18 or any(value.numel() != 8 for value in coefficients.values()):
        raise RuntimeError("evaluated checkpoint must have 18 degree-seven activations")
    for name in coefficients:
        for suffix in ("postscale_tensor", "prescale_tensor"):
            scale = state.get(f"{name}.{suffix}")
            if scale is None or scale.numel() != 1 or float(scale) <= 0:
                raise RuntimeError("evaluated checkpoint activation scales are invalid")
    blends = [value for key, value in state.items() if key.endswith(".blend_alpha")]
    if any(not torch.allclose(value, torch.ones_like(value), atol=1e-7, rtol=0) for value in blends):
        raise RuntimeError("evaluated checkpoint is not fully polynomial")


def _optional_delta(left: Any, right: Any) -> float | None:
    if not _finite_number(left) or not _finite_number(right):
        return None
    return float(left) - float(right)


def _should_select_candidate(
    candidate: dict[str, Any],
    current_best: dict[str, Any],
) -> bool:
    return bool(
        _finite_metrics(candidate)
        and (
            not _finite_metrics(current_best)
            or float(candidate["dice"]) + 1.0e-12 >= float(current_best["dice"])
        )
    )


class _NonFiniteEpochError(RuntimeError):
    """A recoverable numerical failure within a training epoch."""


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    learning_rates = {float(group["lr"]) for group in optimizer.param_groups}
    if len(learning_rates) != 1:
        raise RuntimeError(
            "WPC fine-tuning requires one effective learning rate; got "
            f"{sorted(learning_rates)}"
        )
    return learning_rates.pop()


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    native_reference: nn.Module | None = None,
) -> dict[str, Any]:
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
    finite_items = 0
    comparison_items = 0
    nonfinite_sample_indices: list[int] = []
    native_reference_nonfinite_sample_indices: list[int] = []
    for images, masks in tqdm(loader, desc="validate", leave=False):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        logits = model(images)
        batch = int(images.shape[0])
        finite_mask = torch.isfinite(logits).reshape(batch, -1).all(dim=1)
        nonfinite_sample_indices.extend(
            items + int(offset)
            for offset in torch.nonzero(~finite_mask, as_tuple=False).flatten().tolist()
        )
        finite_count = int(finite_mask.sum().item())
        if finite_count:
            # Compute validation reductions in float64.  A finite float32 logit can
            # still overflow a float32 reduction, which should not be mistaken for
            # a non-finite model output.
            finite_logits = logits[finite_mask].to(torch.float64)
            finite_masks = masks[finite_mask].to(torch.float64)
            loss = segmentation_loss(finite_logits, finite_masks)
            metrics = batch_metrics(finite_logits, finite_masks)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("finite logits produced a non-finite validation loss")
            totals["loss"] += float(loss.item()) * finite_count
            totals["dice"] += float(metrics["dice"]) * finite_count
            totals["iou"] += float(metrics["iou"]) * finite_count
            finite_items += finite_count
        if native_reference is not None:
            native_logits = native_reference(images)
            native_finite_mask = (
                torch.isfinite(native_logits).reshape(batch, -1).all(dim=1)
            )
            native_reference_nonfinite_sample_indices.extend(
                items + int(offset)
                for offset in torch.nonzero(
                    ~native_finite_mask, as_tuple=False
                ).flatten().tolist()
            )
            comparison_mask = finite_mask & native_finite_mask
            comparison_count = int(comparison_mask.sum().item())
            if comparison_count:
                compared_logits = logits[comparison_mask].to(torch.float64)
                compared_native = native_logits[comparison_mask].to(torch.float64)
                delta = (compared_logits - compared_native).abs()
                totals["logits_mae_vs_native"] += (
                    float(delta.mean().item()) * comparison_count
                )
                maximum_logit_delta = max(
                    maximum_logit_delta,
                    float(delta.max().item()),
                )
                probabilities = torch.sigmoid(compared_logits)
                native_probabilities = torch.sigmoid(compared_native)
                totals["prob_mae_vs_native"] += float(
                    (probabilities - native_probabilities).abs().mean().item()
                ) * comparison_count
                totals["prediction_flip_rate_vs_native"] += float(
                    (
                        (probabilities >= 0.5)
                        != (native_probabilities >= 0.5)
                    )
                    .to(torch.float32)
                    .mean()
                    .item()
                ) * comparison_count
                comparison_items += comparison_count
        items += batch
    fully_finite = bool(items > 0 and finite_items == items)
    result = {
        "loss": totals["loss"] / finite_items if finite_items else None,
        "dice": totals["dice"] / finite_items if finite_items else None,
        "iou": totals["iou"] / finite_items if finite_items else None,
        "sample_count": int(items),
        "finite_sample_count": int(finite_items),
        "nonfinite_sample_count": int(items - finite_items),
        "nonfinite_sample_indices": nonfinite_sample_indices,
        "fully_finite": fully_finite,
        "metric_scope": "all_samples" if fully_finite else "finite_samples_only",
    }
    if native_reference is not None:
        result.update(
            {
                "logits_mae_vs_native": totals["logits_mae_vs_native"]
                / comparison_items
                if comparison_items
                else None,
                "max_abs_logit_delta_vs_native": float(maximum_logit_delta)
                if comparison_items
                else None,
                "prob_mae_vs_native": totals["prob_mae_vs_native"]
                / comparison_items
                if comparison_items
                else None,
                "prediction_flip_rate_vs_native": totals[
                    "prediction_flip_rate_vs_native"
                ]
                / comparison_items
                if comparison_items
                else None,
                "comparison_sample_count": int(comparison_items),
                "native_reference_nonfinite_sample_count": int(
                    len(native_reference_nonfinite_sample_indices)
                ),
                "native_reference_nonfinite_sample_indices": (
                    native_reference_nonfinite_sample_indices
                ),
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
    for batch_index, (images, masks) in enumerate(
        tqdm(loader, desc="train", leave=False)
    ):
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
            finite_logits = bool(torch.isfinite(logits).all())
            raise _NonFiniteEpochError(
                "non-finite training loss at "
                f"batch_index={batch_index}: logits_finite={finite_logits}, "
                f"segmentation_loss={float(segmentation.detach().item())}, "
                f"distillation_loss={float(distillation.detach().item())}"
            )
        loss.backward()
        if float(grad_clip_norm) > 0.0:
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(grad_clip_norm),
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                raise _NonFiniteEpochError(
                    f"non-finite gradient norm at batch_index={batch_index}"
                ) from error
        optimizer.step()
        batch = int(images.shape[0])
        totals["loss"] += float(loss.detach().item()) * batch
        totals["segmentation_loss"] += float(segmentation.detach().item()) * batch
        totals["distillation_loss"] += float(distillation.detach().item()) * batch
        items += batch
    return {
        name: value / max(1, items) for name, value in totals.items()
    } | {"sample_count": int(items)}


def _run_epoch_with_backoff(
    *,
    model: nn.Module,
    native_teacher: nn.Module,
    train_set: Dataset,
    validation_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    grad_clip_norm: float,
    distill_weight: float,
    safe_checkpoint_path: Path,
    max_epoch_retries: int,
    lr_backoff_factor: float,
    min_lr: float,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run one epoch, rolling back and lowering LR after numerical failure."""

    epoch_started = time.time()
    failures: list[dict[str, Any]] = []
    shuffle_seed = int(seed) + int(epoch) - 1
    attempt = 0
    while True:
        attempt += 1
        attempt_started = time.time()
        effective_lr = _optimizer_lr(optimizer)
        train_loader = make_loader(
            train_set,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            seed=shuffle_seed,
        )
        try:
            train_metrics = _train_epoch(
                model,
                native_teacher,
                train_loader,
                device=device,
                optimizer=optimizer,
                grad_clip_norm=float(grad_clip_norm),
                distill_weight=float(distill_weight),
            )
            training_audit_loader = make_loader(
                train_set,
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=int(num_workers),
                seed=shuffle_seed,
            )
            post_epoch_training_audit = _evaluate(
                model,
                training_audit_loader,
                device=device,
            )
            training_audit_complete = bool(
                _finite_metrics(post_epoch_training_audit)
                and int(post_epoch_training_audit.get("sample_count", -1))
                == len(train_set)
            )
            if not training_audit_complete:
                raise _NonFiniteEpochError(
                    "post-epoch training-set audit was incomplete or found "
                    "non-finite logits: "
                    f"expected_sample_count={len(train_set)}, "
                    f"sample_count="
                    f"{post_epoch_training_audit.get('sample_count')}, "
                    f"nonfinite_sample_count="
                    f"{post_epoch_training_audit.get('nonfinite_sample_count')}, "
                    f"nonfinite_sample_indices="
                    f"{post_epoch_training_audit.get('nonfinite_sample_indices')}"
                )
            validation_metrics = _evaluate(
                model,
                validation_loader,
                device=device,
                native_reference=native_teacher,
            )
            if not _finite_metrics(validation_metrics):
                raise _NonFiniteEpochError(
                    "non-finite validation logits after epoch: "
                    f"nonfinite_sample_count="
                    f"{validation_metrics.get('nonfinite_sample_count')}"
                )
            accounting = {
                "attempt_count": int(attempt),
                "effective_lr": float(effective_lr),
                "epoch_s": float(time.time() - epoch_started),
                "failed_attempts": failures,
                "shuffle_seed": int(shuffle_seed),
            }
            return (
                train_metrics,
                post_epoch_training_audit,
                validation_metrics,
                accounting,
            )
        except _NonFiniteEpochError as error:
            failure = {
                "attempt": int(attempt),
                "attempt_s": float(time.time() - attempt_started),
                "error": str(error),
                "lr": float(effective_lr),
            }
            failures.append(failure)
            print(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "event": "nonfinite_epoch_attempt_rolled_back",
                        **failure,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if len(failures) > int(max_epoch_retries):
                raise RuntimeError(
                    f"epoch {epoch} remained non-finite after {attempt} attempts; "
                    f"last safe checkpoint remains {safe_checkpoint_path}"
                ) from error

            next_lr = max(float(min_lr), effective_lr * float(lr_backoff_factor))
            if not next_lr < effective_lr:
                raise RuntimeError(
                    f"cannot reduce learning rate below {effective_lr}; "
                    f"last safe checkpoint remains {safe_checkpoint_path}"
                ) from error
            safe = torch.load(
                safe_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            expected_safe_epoch = int(epoch) - 1
            if int(safe.get("epoch", -1)) != expected_safe_epoch:
                raise RuntimeError(
                    "safe checkpoint epoch mismatch: "
                    f"expected {expected_safe_epoch}, got {safe.get('epoch')}"
                ) from error
            model.load_state_dict(safe["state_dict"], strict=True)
            model.to(device)
            optimizer.load_state_dict(safe["optimizer_state_dict"])
            _set_optimizer_lr(optimizer, next_lr)


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_metrics: dict[str, Any],
    history: list[dict[str, Any]],
    source_checkpoint: Path,
    source_sha256: str,
    conversions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 4,
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
    native_metrics: dict[str, Any],
    pre_metrics: dict[str, Any],
    best_metrics: dict[str, Any],
    best_epoch: int,
    completed_epoch: int,
    history: list[dict[str, Any]],
    best_path: Path,
    last_path: Path,
    checkpoint_reload_compatible: bool,
    evaluated_checkpoint: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    sha256(source_sha256, name="source checkpoint")
    native_finite = _finite_metrics(native_metrics)
    pre_finite = _finite_metrics(pre_metrics)
    best_finite = _finite_metrics(best_metrics)
    pre_evaluation_complete = bool(
        int(pre_metrics.get("sample_count", -1)) == int(val_count)
        and int(pre_metrics.get("finite_sample_count", 0))
        + int(pre_metrics.get("nonfinite_sample_count", 0))
        == int(val_count)
    )
    instability_recovered = bool(not pre_finite and best_finite)
    pre_delta = pre_metrics.get("max_abs_logit_delta_vs_native")
    semantic_change_observed = bool(
        int(pre_metrics.get("nonfinite_sample_count", 0)) > 0
        or (_finite_number(pre_delta) and float(pre_delta) > 0.0)
    )
    best_not_worse_when_comparable = bool(
        best_finite
        and (
            not pre_finite
            or float(best_metrics["dice"]) + 1.0e-12 >= float(pre_metrics["dice"])
        )
    )
    history_epoch_numbers = [int(row.get("epoch", -1)) for row in history]
    training_history_complete = bool(
        history_epoch_numbers == list(range(1, int(completed_epoch) + 1))
        and all(
            int(row.get("train", {}).get("sample_count", -1)) == int(train_count)
            for row in history
        )
    )
    post_epoch_training_audits_valid = _training_audits_complete(
        history,
        completed_epoch=int(completed_epoch),
        train_count=int(train_count),
    )
    epoch_checkpoint_paths = [
        last_path.parent / f"rotation_padding_epoch_{epoch:04d}.pt"
        for epoch in range(0, int(completed_epoch) + 1)
    ]
    checkpoint_manifest = [
        {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in [best_path, last_path, *epoch_checkpoint_paths] if path.is_file()
    ]
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
        "native_validation_metrics_are_finite": native_finite,
        "unfinetuned_rotation_padding_evaluation_is_fully_accounted": (
            pre_evaluation_complete
        ),
        "best_rotation_padding_metrics_are_finite": best_finite,
        "rotation_padding_semantic_change_was_observed": semantic_change_observed,
        "best_checkpoint_is_finite_and_not_worse_when_comparable": (
            best_not_worse_when_comparable
        ),
        "completed_epochs_cover_every_training_sample": training_history_complete,
        "post_epoch_training_set_audits_are_fully_finite": (
            post_epoch_training_audits_valid
        ),
        "immutable_epoch_checkpoints_exist": all(
            path.is_file() for path in epoch_checkpoint_paths
        ),
        "requested_epochs_completed": bool(
            args.eval_only or int(completed_epoch) >= int(args.epochs)
        ),
        "fine_tuned_checkpoint_reloads_with_original_orion_schema": bool(
            checkpoint_reload_compatible
        ),
        "best_and_last_checkpoints_exist": best_path.is_file()
        and last_path.is_file(),
        "evaluated_checkpoint_content_identity_recorded": bool(
            sha256(evaluated_checkpoint.get("sha256"), name="evaluated checkpoint")
            and evaluated_checkpoint.get("source_checkpoint_sha256") == source_sha256
            and evaluated_checkpoint.get("epoch") == best_epoch
            and evaluated_checkpoint.get("path") == str(best_path)
            and evaluated_checkpoint.get("padding_semantics") == "wpc_flattened_spatial_rotation_padding"
            and evaluated_checkpoint.get("identity_policy") == "sha256_of_exact_bytes_deserialized"
            and best_path.is_file() and _sha256(best_path) == evaluated_checkpoint.get("sha256")
        ),
    }
    acceptance["valid"] = bool(all(acceptance.values()))
    return {
        "schema_version": 5,
        "profile": "wpc_rotation_padding_checkpoint_finetune_accuracy",
        "status": "ok" if acceptance["valid"] else "invalid",
        "timing_policy": "clear training and accuracy validation; no FHE latency claim",
        "source_checkpoint": {
            "path": str(source_checkpoint),
            "sha256": str(source_sha256),
        },
        "evaluated_checkpoint": dict(evaluated_checkpoint),
        "checkpoint_manifest": checkpoint_manifest,
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
            "resume_lr": float(args.resume_lr)
            if args.resume_lr is not None
            else None,
            "max_epoch_retries": int(args.max_epoch_retries),
            "lr_backoff_factor": float(args.lr_backoff_factor),
            "min_lr": float(args.min_lr),
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
            "numerical_stability_recovered_by_finetuning": instability_recovered,
            "native_to_unfinetuned_rotation": {
                "dice_delta": _optional_delta(
                    pre_metrics.get("dice"), native_metrics.get("dice")
                )
                if pre_finite
                else None,
                "iou_delta": _optional_delta(
                    pre_metrics.get("iou"), native_metrics.get("iou")
                )
                if pre_finite
                else None,
                "comparable": pre_finite and native_finite,
            },
            "unfinetuned_to_best_rotation": {
                "dice_delta": _optional_delta(
                    best_metrics.get("dice"), pre_metrics.get("dice")
                )
                if pre_finite
                else None,
                "iou_delta": _optional_delta(
                    best_metrics.get("iou"), pre_metrics.get("iou")
                )
                if pre_finite
                else None,
                "comparable": pre_finite and best_finite,
            },
            "native_to_best_rotation": {
                "dice_delta": _optional_delta(
                    best_metrics.get("dice"), native_metrics.get("dice")
                ),
                "iou_delta": _optional_delta(
                    best_metrics.get("iou"), native_metrics.get("iou")
                ),
                "comparable": best_finite and native_finite,
            },
        },
        "training": {
            "completed_epoch": int(completed_epoch),
            "best_epoch": int(best_epoch),
            "history": list(history),
            "nonfinite_retry_count": int(
                sum(len(row.get("failed_attempts", [])) for row in history)
            ),
        },
        "outputs": {
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "epoch_checkpoints": [str(path) for path in epoch_checkpoint_paths],
            "result": str(Path(args.result).expanduser().resolve()),
        },
        "limitations": [
            "clear PyTorch fine-tuning and validation rather than FHE inference",
            "accuracy is measured only on the selected FHELIPE dataset split",
            "the best checkpoint may remain epoch zero if fine-tuning does not improve Dice",
            "un-fine-tuned Rotation Padding metrics are finite-sample-only when polynomial extrapolation produces non-finite logits",
            "every accepted epoch is audited over the complete selected training set without parameter updates",
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
    if float(args.lr) <= 0.0:
        raise SystemExit("lr must be positive")
    if args.resume_lr is not None and float(args.resume_lr) <= 0.0:
        raise SystemExit("resume-lr must be positive")
    if int(args.max_epoch_retries) < 0:
        raise SystemExit("max-epoch-retries must be nonnegative")
    if not 0.0 < float(args.lr_backoff_factor) < 1.0:
        raise SystemExit("lr-backoff-factor must be between zero and one")
    if float(args.min_lr) <= 0.0:
        raise SystemExit("min-lr must be positive")
    _set_seed(int(args.seed))
    started = time.time()
    source_checkpoint = args.checkpoint.expanduser().resolve()
    if not source_checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {source_checkpoint}")
    source_payload, source_sha256 = load_checkpoint_bytes(source_checkpoint)
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
        checkpoint_payload=source_payload,
    )
    del source_payload
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
    epoch_zero_path = out_dir / "rotation_padding_epoch_0000.pt"
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
        saved, best_sha256 = load_checkpoint_bytes(best_path)
        _validate_evaluated_checkpoint(saved, path=best_path, source_sha256=source_sha256, train_count=len(train_set))
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
            raise RuntimeError("eval-only requires the existing last checkpoint; no checkpoints are written")
        best_saved = saved
    else:
        if bool(args.resume_if_present) and last_path.is_file():
            resumed = torch.load(last_path, map_location="cpu", weights_only=False)
            _validate_resume_checkpoint(
                resumed,
                checkpoint_path=last_path,
                train_count=len(train_set),
            )
            if str(resumed.get("source_checkpoint_sha256")) != source_sha256:
                raise RuntimeError("resume checkpoint belongs to a different source checkpoint")
            rotation_model.load_state_dict(resumed["state_dict"], strict=True)
            rotation_model.to(device)
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            if args.resume_lr is not None:
                _set_optimizer_lr(optimizer, float(args.resume_lr))
            completed_epoch = int(resumed.get("epoch", 0))
            start_epoch = completed_epoch + 1
            best_epoch = int(resumed.get("best_epoch", 0))
            best_metrics = dict(resumed.get("best_valid", pre_metrics))
            history = list(resumed.get("history", []))
            resumed_metrics = _evaluate(
                rotation_model,
                validation_loader,
                device=device,
                native_reference=native_model,
            )
            if _should_select_candidate(resumed_metrics, best_metrics):
                best_metrics = dict(resumed_metrics)
                best_epoch = int(completed_epoch)
                promoted_payload = _checkpoint_payload(
                    model=rotation_model,
                    optimizer=optimizer,
                    epoch=completed_epoch,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                    history=history,
                    source_checkpoint=source_checkpoint,
                    source_sha256=source_sha256,
                    conversions=conversions,
                )
                _atomic_torch_save(promoted_payload, best_path)
                print(
                    json.dumps(
                        {
                            "event": "resumed_last_promoted_to_best",
                            "epoch": int(completed_epoch),
                            "validation": resumed_metrics,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        else:
            existing_checkpoint_paths = [
                path
                for path in (
                    best_path,
                    last_path,
                    *sorted(out_dir.glob("rotation_padding_epoch_*.pt")),
                )
                if path.is_file()
            ]
            if existing_checkpoint_paths:
                raise RuntimeError(
                    "refusing to overwrite existing fine-tuning checkpoints without "
                    "--resume-if-present: "
                    + ", ".join(str(path) for path in existing_checkpoint_paths)
                )
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
            _atomic_torch_save(initial, epoch_zero_path)

        for epoch in range(start_epoch, int(args.epochs) + 1):
            (
                train_metrics,
                post_epoch_training_audit,
                validation_metrics,
                epoch_accounting,
            ) = (
                _run_epoch_with_backoff(
                    model=rotation_model,
                    native_teacher=native_model,
                    train_set=train_set,
                    validation_loader=validation_loader,
                    device=device,
                    optimizer=optimizer,
                    epoch=epoch,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    seed=int(args.seed),
                    grad_clip_norm=float(args.grad_clip_norm),
                    distill_weight=float(args.distill_weight),
                    safe_checkpoint_path=last_path,
                    max_epoch_retries=int(args.max_epoch_retries),
                    lr_backoff_factor=float(args.lr_backoff_factor),
                    min_lr=float(args.min_lr),
                )
            )
            row = {
                "epoch": int(epoch),
                "train": train_metrics,
                "post_epoch_training_audit": post_epoch_training_audit,
                "validation": validation_metrics,
                **epoch_accounting,
            }
            history.append(row)
            completed_epoch = int(epoch)
            if _should_select_candidate(validation_metrics, best_metrics):
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
            epoch_path = out_dir / f"rotation_padding_epoch_{epoch:04d}.pt"
            _atomic_torch_save(last_payload, epoch_path)
            partial_result = {
                "schema_version": 4,
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

        best_saved, best_sha256 = load_checkpoint_bytes(best_path)
        _validate_evaluated_checkpoint(best_saved, path=best_path, source_sha256=source_sha256, train_count=len(train_set))
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
        checkpoint_payload=best_saved,
    )
    checkpoint_reload_compatible = bool(
        set(reload_model.state_dict()) == set(rotation_model.state_dict())
        and all(torch.equal(value.detach().cpu(), rotation_model.state_dict()[key].detach().cpu())
                for key, value in reload_model.state_dict().items())
    )
    del reload_model
    # Refuse to emit a result if the supplied files changed during evaluation.
    if _sha256(best_path) != best_sha256 or _sha256(source_checkpoint) != source_sha256:
        raise RuntimeError("checkpoint file changed during evaluation; result not written")

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
        evaluated_checkpoint={
            "path": str(best_path),
            "sha256": best_sha256,
            "source_checkpoint_sha256": str(best_saved["source_checkpoint_sha256"]),
            "epoch": int(best_saved["epoch"]),
            "checkpoint_schema_version": int(best_saved["schema_version"]),
            "padding_semantics": str(best_saved["model"]["padding_semantics"]),
            "identity_policy": "sha256_of_exact_bytes_deserialized",
        },
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
