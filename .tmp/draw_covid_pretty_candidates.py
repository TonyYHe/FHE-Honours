#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_fhelipe_medseg_staged_unet import make_stage_model  # noqa: E402
from tools.train_fhelipe_medseg_unet22 import FhelipeSegmentationDataset, SPECS  # noqa: E402
from tools.verify_medseg_cheb7_orion_clear_adapter import _build_training_reference_model  # noqa: E402


IMAGE_SIZE = 256
THRESHOLD = 0.5
OUTPUT_DIR = Path("/home/anakano/CLionProjects/fhelipe/scripts/unet/covid_scaled_pretty_candidates")
SILU_CHECKPOINT = (
    ROOT
    / "checkpoints/fhelipe_medseg_staged_covid19_256_scaled_silu_freeze15_cheb7_20260603"
    / "covid19_unet22_plus_output_base32_256_scaled_silu_avgpool_retrain_best.pt"
)
CHEB7_CHECKPOINT = (
    ROOT
    / "checkpoints/fhelipe_medseg_staged_covid19_256_scaled_silu_freeze15_cheb7_20260603"
    / "covid19_unet22_plus_output_base32_256_scaled_silu_avgpool_degree_7_finetune_best.pt"
)
CHEB7_REPLACEMENTS = (
    ROOT
    / ".tmp/results/fhelipe_medseg_covid19_unet22_plus_output_256_base32_scaled_silu_freeze15_cheb7_finetune20_20260603.json"
)


def load_silu_model(device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(SILU_CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint.get("model", {}) or {})
    model = make_stage_model(
        architecture=str(cfg.get("architecture", "unet22-plus-output")),
        in_channels=int(cfg.get("in_channels", 1)),
        out_channels=int(cfg.get("out_channels", 1)),
        base_dim=int(cfg.get("base_dim", 32)),
        activation=str(cfg.get("activation", "scaled-silu")),
        pool="avg",
        scale_margin=2.0,
        silu_degree=7,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def make_dataset() -> FhelipeSegmentationDataset:
    spec = SPECS["covid19"]
    return FhelipeSegmentationDataset(
        ROOT / "data/fhelipe_medseg" / spec.filename,
        image_key=spec.val_image_key,
        label_key=spec.val_label_key,
        image_size=IMAGE_SIZE,
        limit=0,
        seed=0,
    )


def mask_metrics(prob: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = (prob >= THRESHOLD).to(dtype=target.dtype)
    dims = tuple(range(1, pred.dim()))
    intersection = (pred * target).sum(dim=dims)
    pred_area = pred.sum(dim=dims)
    target_area = target.sum(dim=dims)
    union = pred_area + target_area - intersection
    dice = (2.0 * intersection + 1.0e-6) / (pred_area + target_area + 1.0e-6)
    iou = (intersection + 1.0e-6) / (union + 1.0e-6)
    return dice, iou, pred_area


def normalize_display(image: torch.Tensor) -> np.ndarray:
    arr = image.squeeze(0).detach().cpu().numpy()
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1.0e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def colorize(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    rgb[mask > 0.5] = np.asarray(color, dtype=np.float32) / 255.0
    return rgb


def overlay(gray: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1)
    return np.clip((1.0 - alpha) * base + alpha * mask_rgb, 0.0, 1.0)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def shape_score(mask: np.ndarray) -> float:
    box = bbox(mask)
    if box is None:
        return -10.0
    x0, y0, x1, y1 = box
    area = float(mask.mean())
    width = float(x1 - x0 + 1)
    height = float(y1 - y0 + 1)
    center_x = float((x0 + x1) / 2.0) / float(mask.shape[1])
    aspect = height / max(width, 1.0)
    left = float(mask[:, : mask.shape[1] // 2].sum())
    right = float(mask[:, mask.shape[1] // 2 :].sum())
    balance = abs(left - right) / max(left + right, 1.0)
    score = 0.0
    score -= abs(area - 0.32) * 2.0
    score -= abs(center_x - 0.50) * 0.8
    score -= abs(aspect - 0.95) * 0.2
    score -= balance * 0.8
    score -= 0.5 if y0 < 5 or y1 > mask.shape[0] - 5 else 0.0
    return float(score)


def make_panels(image: torch.Tensor, target: torch.Tensor, silu_mask: np.ndarray, cheb_mask: np.ndarray) -> list[np.ndarray]:
    gray = normalize_display(image)
    gt = target.squeeze(0).detach().cpu().numpy()
    return [
        np.stack([gray, gray, gray], axis=-1),
        overlay(gray, colorize(gt, (0, 255, 0))),
        overlay(gray, colorize(silu_mask, (0, 128, 255))),
        overlay(gray, colorize(cheb_mask, (255, 128, 0))),
    ]


def strip_from_panels(panels: list[np.ndarray], gap_pixels: int = 4) -> np.ndarray:
    gap = np.ones((panels[0].shape[0], int(gap_pixels), 3), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for idx, panel in enumerate(panels):
        if idx:
            pieces.append(gap)
        pieces.append(np.clip(panel, 0.0, 1.0))
    return np.concatenate(pieces, axis=1)


def save_strip(base: Path, strip: np.ndarray) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(base.with_suffix(".png"), strip)
    fig = plt.figure(figsize=(12, 3), frameon=False)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(strip)
    ax.axis("off")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = make_dataset()
    silu = load_silu_model(device)
    cheb, _meta = _build_training_reference_model(CHEB7_CHECKPOINT, replacements_path=CHEB7_REPLACEMENTS, device=device)
    loader = DataLoader(ds, batch_size=24, shuffle=False, num_workers=0)
    candidates: list[dict[str, float | int]] = []
    offset = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)
            silu_prob = torch.sigmoid(silu(images))
            cheb_prob = torch.sigmoid(cheb(images))
            silu_dice, silu_iou, silu_area = mask_metrics(silu_prob, masks)
            cheb_dice, cheb_iou, cheb_area = mask_metrics(cheb_prob, masks)
            target_area = masks.flatten(1).sum(dim=1)
            cheb_masks = (cheb_prob >= THRESHOLD).detach().cpu().numpy()
            target_masks = masks.detach().cpu().numpy()
            for local in range(int(images.shape[0])):
                idx = int(offset + local)
                target_fraction = float(target_area[local].item() / (IMAGE_SIZE * IMAGE_SIZE))
                cheb_mask = cheb_masks[local, 0]
                target_mask = target_masks[local, 0]
                if not (0.22 <= target_fraction <= 0.42):
                    continue
                score = (
                    float(cheb_dice[local].item()) * 4.0
                    + float(silu_dice[local].item())
                    + shape_score(target_mask)
                    + 0.5 * shape_score(cheb_mask)
                    - abs(float(cheb_area[local].item() - target_area[local].item())) / 65536.0
                )
                candidates.append(
                    {
                        "index": idx,
                        "score": float(score),
                        "target_fraction": float(target_fraction),
                        "silu_dice": float(silu_dice[local].item()),
                        "silu_iou": float(silu_iou[local].item()),
                        "cheb_dice": float(cheb_dice[local].item()),
                        "cheb_iou": float(cheb_iou[local].item()),
                    }
                )
            offset += int(images.shape[0])
    selected = sorted(candidates, key=lambda row: float(row["score"]), reverse=True)[:24]
    rows_out = []
    strips = []
    with torch.no_grad():
        for row in selected:
            idx = int(row["index"])
            image, target = ds[idx]
            image_batch = image.unsqueeze(0).to(device=device, dtype=torch.float32)
            silu_mask = (torch.sigmoid(silu(image_batch)).detach().cpu().squeeze().numpy() >= THRESHOLD).astype(np.float32)
            cheb_mask = (torch.sigmoid(cheb(image_batch)).detach().cpu().squeeze().numpy() >= THRESHOLD).astype(np.float32)
            strip = strip_from_panels(make_panels(image, target, silu_mask, cheb_mask))
            base = OUTPUT_DIR / f"covid19_256_val{idx}_scaled_silu_cheb7_pretty"
            save_strip(base, strip)
            out_row = dict(row)
            out_row["png"] = str(base.with_suffix(".png"))
            out_row["pdf"] = str(base.with_suffix(".pdf"))
            rows_out.append(out_row)
            strips.append((out_row, strip))

    # Contact sheet with small labels above each strip.
    thumbs = []
    for row, strip in strips:
        fig = plt.figure(figsize=(12, 3.35), frameon=False)
        ax = fig.add_axes((0, 0, 1, 0.92))
        ax.imshow(strip)
        ax.axis("off")
        fig.text(
            0.01,
            0.965,
            f"val{int(row['index'])}  Cheb Dice {float(row['cheb_dice']):.4f}  IoU {float(row['cheb_iou']):.4f}",
            ha="left",
            va="top",
            fontsize=12,
        )
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].astype(np.float32) / 255.0
        plt.close(fig)
        thumbs.append(arr)
    cols = 2
    contact_rows = []
    for start in range(0, len(thumbs), cols):
        chunk = thumbs[start : start + cols]
        if len(chunk) < cols:
            chunk.append(np.ones_like(thumbs[0]))
        contact_rows.append(np.concatenate(chunk, axis=1))
    contact = np.concatenate(contact_rows, axis=0)
    contact_path = OUTPUT_DIR / "covid19_256_pretty_candidates_contact.png"
    plt.imsave(contact_path, contact)
    metrics_path = OUTPUT_DIR / "covid19_256_pretty_candidates_metrics.json"
    metrics_path.write_text(json.dumps({"rows": rows_out, "contact": str(contact_path)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metrics": str(metrics_path), "contact": str(contact_path), "rows": rows_out[:12]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
