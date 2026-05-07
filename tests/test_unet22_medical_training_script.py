import random

import torch
from PIL import Image, ImageDraw

from scripts.train_unet22_medical_seg import (
    AugmentConfig,
    PairedSegmentationAugment,
    apply_image_noise,
    pil_to_tensor,
)


def test_paired_segmentation_augment_preserves_training_tensor_contract() -> None:
    random.seed(0)
    torch.manual_seed(0)
    image = Image.new("RGB", (96, 80), (96, 128, 160))
    mask = Image.new("L", (96, 80), 0)
    ImageDraw.Draw(mask).rectangle((24, 20, 72, 60), fill=255)
    augment = PairedSegmentationAugment(
        AugmentConfig(
            hflip_p=1.0,
            rotate_degrees=8.0,
            translate=0.05,
            scale_min=0.95,
            scale_max=1.05,
            brightness=0.1,
            contrast=0.1,
            gamma_min=0.95,
            gamma_max=1.05,
            noise_std=0.02,
            blur_p=1.0,
        )
    )

    image, mask = augment(image, mask)
    image_tensor = apply_image_noise(pil_to_tensor(image, channels=3, size=64), std=augment.config.noise_std)
    mask_tensor = pil_to_tensor(mask, channels=1, size=64, mask=True)

    assert tuple(image_tensor.shape) == (3, 64, 64)
    assert tuple(mask_tensor.shape) == (1, 64, 64)
    assert float(image_tensor.min()) >= 0.0
    assert float(image_tensor.max()) <= 1.0
    assert set(torch.unique(mask_tensor).tolist()).issubset({0.0, 1.0})
