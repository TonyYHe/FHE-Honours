#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_fhelipe_medseg_staged_unet import make_stage_model  # noqa: E402
from tools.train_fhelipe_medseg_unet22 import FhelipeSegmentationDataset, SPECS  # noqa: E402
from tools.verify_medseg_cheb7_orion_clear_adapter import _build_training_reference_model  # noqa: E402


INDICES = (1937, 1946, 1948)
IMAGE_SIZE = 256
THRESHOLD = 0.5
OUTPUT_DIR = Path("/home/anakano/CLionProjects/fhelipe/scripts/unet/covid_scaled_eval_candidates")
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
    model_cfg = dict(checkpoint.get("model", {}) or {})
    model = make_stage_model(
        architecture=str(model_cfg.get("architecture", "unet22-plus-output")),
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_dim=int(model_cfg.get("base_dim", 32)),
        activation=str(model_cfg.get("activation", "scaled-silu")),
        pool="avg",
        scale_margin=2.0,
        silu_degree=7,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def dataset() -> FhelipeSegmentationDataset:
    spec = SPECS["covid19"]
    return FhelipeSegmentationDataset(
        ROOT / "data/fhelipe_medseg" / spec.filename,
        image_key=spec.val_image_key,
        label_key=spec.val_label_key,
        image_size=IMAGE_SIZE,
        limit=0,
        seed=0,
    )


def normalize_display(image: torch.Tensor) -> np.ndarray:
    array = image.squeeze(0).detach().cpu().numpy()
    lo = float(np.min(array))
    hi = float(np.max(array))
    if hi - lo < 1.0e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - lo) / (hi - lo)).astype(np.float32)


def colorize(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    rgb[mask > 0.5] = np.asarray(color, dtype=np.float32) / 255.0
    return rgb


def overlay(gray: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1)
    return np.clip((1.0 - alpha) * base + alpha * mask_rgb, 0.0, 1.0)


def metrics(prob: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = (prob >= THRESHOLD).to(dtype=target.dtype)
    intersection = float((pred * target).sum().item())
    pred_area = float(pred.sum().item())
    target_area = float(target.sum().item())
    union = pred_area + target_area - intersection
    dice = (2.0 * intersection + 1.0e-6) / (pred_area + target_area + 1.0e-6)
    iou = (intersection + 1.0e-6) / (union + 1.0e-6)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "pred_area": float(pred_area),
        "target_area": float(target_area),
    }


def save_strip(path_base: Path, panels: list[np.ndarray], *, gap_pixels: int = 4) -> np.ndarray:
    gap = np.ones((panels[0].shape[0], int(gap_pixels), 3), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, panel in enumerate(panels):
        if index:
            pieces.append(gap)
        pieces.append(np.clip(panel, 0.0, 1.0))
    strip = np.concatenate(pieces, axis=1)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path_base.with_suffix(".png"), strip)
    fig = plt.figure(figsize=(12, 3), frameon=False)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(strip)
    ax.axis("off")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return strip


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = dataset()
    silu = load_silu_model(device)
    cheb7, cheb_meta = _build_training_reference_model(
        CHEB7_CHECKPOINT,
        replacements_path=CHEB7_REPLACEMENTS,
        device=device,
    )
    rows = []
    strips = []
    with torch.no_grad():
        for val_index in INDICES:
            image, target = val[int(val_index)]
            batch = image.unsqueeze(0).to(device=device, dtype=torch.float32)
            target_batch = target.unsqueeze(0).to(device=device, dtype=torch.float32)
            silu_prob = torch.sigmoid(silu(batch)).detach().cpu().squeeze(0)
            cheb_prob = torch.sigmoid(cheb7(batch)).detach().cpu().squeeze(0)
            gray = normalize_display(image)
            gt = target.squeeze(0).detach().cpu().numpy()
            silu_mask = (silu_prob.squeeze(0).numpy() >= THRESHOLD).astype(np.float32)
            cheb_mask = (cheb_prob.squeeze(0).numpy() >= THRESHOLD).astype(np.float32)
            panels = [
                np.stack([gray, gray, gray], axis=-1),
                overlay(gray, colorize(gt, (0, 255, 0))),
                overlay(gray, colorize(silu_mask, (0, 128, 255))),
                overlay(gray, colorize(cheb_mask, (255, 128, 0))),
            ]
            out_base = OUTPUT_DIR / f"covid19_256_val{int(val_index)}_scaled_silu_cheb7_finetune"
            strip = save_strip(out_base, panels)
            strips.append((int(val_index), strip))
            rows.append(
                {
                    "val_index": int(val_index),
                    "silu": metrics(silu_prob.unsqueeze(0), target_batch.cpu()),
                    "cheb7_finetune": metrics(cheb_prob.unsqueeze(0), target_batch.cpu()),
                    "png": str(out_base.with_suffix(".png")),
                    "pdf": str(out_base.with_suffix(".pdf")),
                }
            )
    contact_rows = []
    for val_index, strip in strips:
        label_h = 28
        label = np.ones((label_h, strip.shape[1], 3), dtype=np.float32)
        contact_rows.append(label)
        contact_rows.append(strip)
    contact = np.concatenate(contact_rows, axis=0)
    contact_path = OUTPUT_DIR / "covid19_256_eval_candidates_contact.png"
    plt.imsave(contact_path, contact)
    metrics_path = OUTPUT_DIR / "covid19_256_eval_candidates_metrics.json"
    metrics_path.write_text(
        json.dumps({"cheb7_meta": cheb_meta, "rows": rows, "contact": str(contact_path)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metrics": str(metrics_path), "contact": str(contact_path), "rows": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
