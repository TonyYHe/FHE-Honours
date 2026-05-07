#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.models.unet import get_unet22_medical_model, get_unet22_medical_spec


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    aliases: tuple[str, ...]
    archive_name: str
    url: str
    image_size: int
    in_channels: int
    description: str


@dataclass(frozen=True)
class AugmentConfig:
    hflip_p: float = 0.5
    rotate_degrees: float = 10.0
    translate: float = 0.05
    scale_min: float = 0.9
    scale_max: float = 1.1
    brightness: float = 0.1
    contrast: float = 0.1
    gamma_min: float = 0.9
    gamma_max: float = 1.1
    noise_std: float = 0.01
    blur_p: float = 0.05


DATASETS = {
    "montgomery_lung_64": DatasetConfig(
        key="montgomery_lung_64",
        aliases=("medical64", "lung64", "montgomery", "montgomery_lung"),
        archive_name="NLM-MontgomeryCXRSet.zip",
        url="https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
        image_size=64,
        in_channels=1,
        description="Montgomery County CXR lung mask segmentation resized to 64x64",
    ),
    "kvasir_polyp_256": DatasetConfig(
        key="kvasir_polyp_256",
        aliases=("medical256", "polyp256", "kvasir", "kvasir_polyp"),
        archive_name="kvasir-seg.zip",
        url="https://datasets.simula.no/downloads/kvasir-seg.zip",
        image_size=256,
        in_channels=3,
        description="Kvasir-SEG polyp segmentation resized to 256x256",
    ),
}


def normalize_dataset_key(name: str) -> str:
    name = str(name).lower()
    if name in DATASETS:
        return name
    for key, cfg in DATASETS.items():
        if name in cfg.aliases:
            return key
    supported = ", ".join(sorted(DATASETS))
    raise ValueError(f"Unknown dataset {name!r}. Supported: {supported}.")


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    context = ssl.create_default_context()
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass

    def open_response(ssl_context):
        request = urllib.request.Request(str(url), headers={"User-Agent": "orion-medical-unet22/1.0"})
        return urllib.request.urlopen(request, context=ssl_context)

    try:
        response = open_response(context)
    except urllib.error.URLError as exc:
        if not isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            raise
        print(f"warning: TLS certificate verification failed for {url}; retrying download without verification.", file=sys.stderr)
        response = open_response(ssl._create_unverified_context())

    with response:
        total = int(response.headers.get("Content-Length") or 0)
        with tmp.open("wb") as handle, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"download {dst.name}",
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))
    tmp.replace(dst)


def extract_zip(archive: Path, dst: Path) -> None:
    marker = dst / f".{archive.stem}.extracted"
    if marker.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dst)
    marker.touch()


def prepare_archive(cfg: DatasetConfig, data_root: Path, *, download: bool) -> Path:
    dataset_root = data_root / cfg.key
    archive = dataset_root / cfg.archive_name
    if download:
        download_file(cfg.url, archive)
    elif not archive.exists():
        raise FileNotFoundError(f"{archive} is missing. Re-run without --no-download or place the archive there.")
    extract_zip(archive, dataset_root)
    return dataset_root


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def random_unit_factor(strength: float) -> float:
    strength = max(0.0, float(strength))
    return random.uniform(max(0.0, 1.0 - strength), 1.0 + strength)


def open_pil_image(path: Path, *, channels: int) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L" if int(channels) == 1 else "RGB")


def pil_to_tensor(image: Image.Image, *, channels: int, size: int, mask: bool = False) -> torch.Tensor:
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    if mask:
        image = image.convert("L")
    elif channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")
    image = image.resize((int(size), int(size)), resample=resample)
    array = np.asarray(image, dtype=np.float32)
    if mask:
        array = (array > 127.0).astype(np.float32)[None, :, :]
    elif channels == 1:
        array = (array / 255.0)[None, :, :]
    else:
        array = (array / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(array.copy())


def apply_image_noise(image: torch.Tensor, *, std: float) -> torch.Tensor:
    std = max(0.0, float(std))
    if std == 0.0:
        return image
    return (image + torch.randn_like(image) * std).clamp(0.0, 1.0)


class PairedSegmentationAugment:
    def __init__(self, config: AugmentConfig):
        self.config = config

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        mask = mask.convert("L")
        image, mask = self._spatial(image, mask)
        image = self._intensity(image)
        return image, mask

    def _spatial(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        cfg = self.config
        if random.random() < clamp(cfg.hflip_p, 0.0, 1.0):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if float(cfg.rotate_degrees) > 0:
            angle = random.uniform(-float(cfg.rotate_degrees), float(cfg.rotate_degrees))
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=self._fill(image))
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

        if float(cfg.scale_min) > 0 and float(cfg.scale_max) > 0:
            scale = random.uniform(float(cfg.scale_min), float(cfg.scale_max))
            image, mask = self._scale_translate(image, mask, scale=scale, translate=float(cfg.translate))

        return image, mask

    def _scale_translate(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        scale: float,
        translate: float,
    ) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        new_width = max(1, int(round(width * float(scale))))
        new_height = max(1, int(round(height * float(scale))))
        image = image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
        mask = mask.resize((new_width, new_height), resample=Image.Resampling.NEAREST)

        max_dx = int(round(width * max(0.0, float(translate))))
        max_dy = int(round(height * max(0.0, float(translate))))
        dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
        dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0

        if new_width >= width and new_height >= height:
            left = int(clamp((new_width - width) // 2 - dx, 0, new_width - width))
            top = int(clamp((new_height - height) // 2 - dy, 0, new_height - height))
            return (
                image.crop((left, top, left + width, top + height)),
                mask.crop((left, top, left + width, top + height)),
            )

        out_image = Image.new(image.mode, (width, height), self._fill(image))
        out_mask = Image.new("L", (width, height), 0)
        left = int(clamp((width - new_width) // 2 + dx, -new_width + 1, width - 1))
        top = int(clamp((height - new_height) // 2 + dy, -new_height + 1, height - 1))
        out_image.paste(image, (left, top))
        out_mask.paste(mask, (left, top))
        return out_image, out_mask

    def _intensity(self, image: Image.Image) -> Image.Image:
        cfg = self.config
        if float(cfg.brightness) > 0:
            image = ImageEnhance.Brightness(image).enhance(random_unit_factor(float(cfg.brightness)))
        if float(cfg.contrast) > 0:
            image = ImageEnhance.Contrast(image).enhance(random_unit_factor(float(cfg.contrast)))

        gamma_min = max(0.05, float(cfg.gamma_min))
        gamma_max = max(gamma_min, float(cfg.gamma_max))
        if gamma_min != 1.0 or gamma_max != 1.0:
            gamma = random.uniform(gamma_min, gamma_max)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = np.power(np.clip(array, 0.0, 1.0), gamma)
            image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8)).convert(image.mode)

        if random.random() < clamp(cfg.blur_p, 0.0, 1.0):
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
        return image

    @staticmethod
    def _fill(image: Image.Image):
        return 0 if image.mode == "L" else tuple(0 for _ in image.getbands())


def find_one_dir(root: Path, names: tuple[str, ...]) -> Path:
    matches = []
    wanted = {name.lower() for name in names}
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() in wanted:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"Could not find any of {names!r} below {root}.")
    return sorted(matches, key=lambda p: len(p.parts))[0]


class KvasirSegDataset(Dataset):
    def __init__(self, root: Path, *, image_size: int, augment: PairedSegmentationAugment | None = None):
        self.root = root
        images_dir = find_one_dir(root, ("images",))
        masks_dir = find_one_dir(root, ("masks",))
        mask_by_stem = {path.stem: path for path in masks_dir.iterdir() if path.is_file()}
        samples = []
        for image_path in sorted(path for path in images_dir.iterdir() if path.is_file()):
            mask_path = mask_by_stem.get(image_path.stem)
            if mask_path is not None:
                samples.append((image_path, mask_path))
        if not samples:
            raise RuntimeError(f"No Kvasir image/mask pairs found below {root}.")
        self.samples = samples
        self.image_size = int(image_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[int(index)]
        image = open_pil_image(image_path, channels=3)
        mask = open_pil_image(mask_path, channels=1)
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        image = pil_to_tensor(image, channels=3, size=self.image_size)
        mask = pil_to_tensor(mask, channels=1, size=self.image_size, mask=True)
        if self.augment is not None:
            image = apply_image_noise(image, std=self.augment.config.noise_std)
        return image, mask


class MontgomeryLungDataset(Dataset):
    def __init__(self, root: Path, *, image_size: int, augment: PairedSegmentationAugment | None = None):
        self.root = root
        images_dir = find_one_dir(root, ("CXR_png",))
        left_dir = find_one_dir(root, ("leftMask", "left"))
        right_dir = find_one_dir(root, ("rightMask", "right"))

        left_by_name = {path.name: path for path in left_dir.iterdir() if path.is_file()}
        right_by_name = {path.name: path for path in right_dir.iterdir() if path.is_file()}
        samples = []
        for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() == ".png"):
            left_path = left_by_name.get(image_path.name)
            right_path = right_by_name.get(image_path.name)
            if left_path is not None and right_path is not None:
                samples.append((image_path, left_path, right_path))
        if not samples:
            raise RuntimeError(f"No Montgomery image/left-mask/right-mask triples found below {root}.")
        self.samples = samples
        self.image_size = int(image_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, left_path, right_path = self.samples[int(index)]
        image = open_pil_image(image_path, channels=1)
        left = open_pil_image(left_path, channels=1)
        right = open_pil_image(right_path, channels=1)
        mask_pil = ImageChops.lighter(left, right)
        if self.augment is not None:
            image, mask_pil = self.augment(image, mask_pil)
            image = pil_to_tensor(image, channels=1, size=self.image_size)
            mask = pil_to_tensor(mask_pil, channels=1, size=self.image_size, mask=True)
            image = apply_image_noise(image, std=self.augment.config.noise_std)
        else:
            image = pil_to_tensor(image, channels=1, size=self.image_size)
            mask = pil_to_tensor(mask_pil, channels=1, size=self.image_size, mask=True)
        return image, mask


def build_dataset(
    cfg: DatasetConfig,
    data_root: Path,
    *,
    download: bool,
    augment: PairedSegmentationAugment | None = None,
) -> Dataset:
    root = prepare_archive(cfg, data_root, download=download)
    if cfg.key == "montgomery_lung_64":
        return MontgomeryLungDataset(root, image_size=cfg.image_size, augment=augment)
    if cfg.key == "kvasir_polyp_256":
        return KvasirSegDataset(root, image_size=cfg.image_size, augment=augment)
    raise AssertionError(f"Unhandled dataset {cfg.key!r}.")


def split_indices(length: int, *, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if length < 2:
        return list(range(length)), []
    val_count = max(1, int(round(length * float(val_fraction))))
    val_count = min(val_count, length - 1)
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(length, generator=generator).tolist()
    return indices[val_count:], indices[:val_count]


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + int(worker_id)) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss_with_logits(logits, targets)


@torch.no_grad()
def batch_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0e-6) -> tuple[float, float]:
    pred = (torch.sigmoid(logits) >= 0.5).to(targets.dtype)
    dims = tuple(range(1, pred.dim()))
    intersection = (pred * targets).sum(dim=dims)
    pred_area = pred.sum(dim=dims)
    target_area = targets.sum(dim=dims)
    union = pred_area + target_area - intersection
    dice = ((2.0 * intersection + eps) / (pred_area + target_area + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return float(dice.item()), float(iou.item())


def run_epoch(model, loader: DataLoader, *, device: torch.device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_items = 0
    label = "train" if training else "valid"
    for images, masks in tqdm(loader, desc=label, leave=False):
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.float32)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = segmentation_loss(logits, masks)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = int(images.shape[0])
        dice, iou = batch_metrics(logits.detach(), masks)
        total_loss += float(loss.detach().item()) * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        total_items += batch_size

    if total_items == 0:
        return {"loss": math.nan, "dice": math.nan, "iou": math.nan}
    return {
        "loss": total_loss / total_items,
        "dice": total_dice / total_items,
        "iou": total_iou / total_items,
    }


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    cfg: DatasetConfig,
    epoch: int,
    best_val_dice: float,
    metrics: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    spec = get_unet22_medical_spec(cfg.key)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_dice": float(best_val_dice),
        "metrics": metrics,
        "model": {
            "name": cfg.key,
            "architecture": "UNet22",
            "base_dim": int(args.base_dim),
            "image_size": int(cfg.image_size),
            "in_channels": int(cfg.in_channels),
            "out_channels": int(spec.out_channels),
        },
        "dataset": asdict(spec),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNet22(base_dim=32) medical segmentation models.")
    parser.add_argument(
        "--dataset",
        default="montgomery_lung_64",
        help="Dataset/model to train: montgomery_lung_64/medical64 or kvasir_polyp_256/medical256.",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "medical")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "medical_unet22")
    parser.add_argument("--base-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for quick smoke runs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-download", action="store_true", help="Use an already downloaded archive.")
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time image/mask augmentation.")
    parser.add_argument("--hflip-p", type=float, default=0.5)
    parser.add_argument("--rotate-degrees", type=float, default=10.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--brightness", type=float, default=0.1)
    parser.add_argument("--contrast", type=float, default=0.1)
    parser.add_argument("--gamma-min", type=float, default=0.9)
    parser.add_argument("--gamma-max", type=float, default=1.1)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--blur-p", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_key = normalize_dataset_key(args.dataset)
    cfg = DATASETS[dataset_key]
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    split_source = build_dataset(cfg, args.data_root, download=not bool(args.no_download), augment=None)
    split_length = len(split_source)
    if int(args.limit) > 0:
        split_length = min(int(args.limit), split_length)

    train_augment = None
    if not bool(args.no_augment):
        train_augment = PairedSegmentationAugment(
            AugmentConfig(
                hflip_p=float(args.hflip_p),
                rotate_degrees=float(args.rotate_degrees),
                translate=float(args.translate),
                scale_min=float(args.scale_min),
                scale_max=float(args.scale_max),
                brightness=float(args.brightness),
                contrast=float(args.contrast),
                gamma_min=float(args.gamma_min),
                gamma_max=float(args.gamma_max),
                noise_std=float(args.noise_std),
                blur_p=float(args.blur_p),
            )
        )

    train_dataset = build_dataset(cfg, args.data_root, download=False, augment=train_augment)
    val_dataset = build_dataset(cfg, args.data_root, download=False, augment=None)
    train_idx, val_idx = split_indices(split_length, val_fraction=float(args.val_fraction), seed=int(args.seed))
    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(val_dataset, val_idx)
    loader_generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )

    device = torch.device(args.device)
    model = get_unet22_medical_model(cfg.key, base_dim=int(args.base_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    print(
        json.dumps(
            {
                "dataset": cfg.description,
                "train": len(train_set),
                "valid": len(val_set),
                "augmentation": None if train_augment is None else asdict(train_augment.config),
            },
            indent=2,
        )
    )
    best_val_dice = -1.0
    best_path = args.output_dir / f"{cfg.key}_unet22_base{int(args.base_dim)}_best.pt"
    last_path = args.output_dir / f"{cfg.key}_unet22_base{int(args.base_dim)}_last.pt"

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device=device, optimizer=None)
        metrics = {"train": train_metrics, "valid": val_metrics}
        print(f"epoch {epoch:03d} " + json.dumps(metrics, sort_keys=True))

        if val_metrics["dice"] >= best_val_dice:
            best_val_dice = float(val_metrics["dice"])
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                best_val_dice=best_val_dice,
                metrics=metrics,
                args=args,
            )
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
            best_val_dice=best_val_dice,
            metrics=metrics,
            args=args,
        )

    print(f"best checkpoint: {best_path}")
    print(f"last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
