# UNet22 Medical Segmentation Models

This repo exposes two medical segmentation model presets for `UNet22` with
`base_dim=32`.

| Model key | Input | Channels | Public dataset | Task |
| --- | ---: | ---: | --- | --- |
| `montgomery_lung_64` | `64x64` | `1 -> 1` | Montgomery County Chest X-ray Set | binary lung field segmentation |
| `kvasir_polyp_256` | `256x256` | `3 -> 1` | Kvasir-SEG | binary polyp segmentation |

## Instantiate

```python
from orion.models.unet import (
    UNet22KvasirPolyp256,
    UNet22MontgomeryLung64,
    get_unet22_medical_model,
)

lung64 = UNet22MontgomeryLung64()          # UNet22(base_dim=32), 1x64x64 -> 1x64x64
polyp256 = UNet22KvasirPolyp256()         # UNet22(base_dim=32), 3x256x256 -> 1x256x256

same_lung64 = get_unet22_medical_model("medical64")
same_polyp256 = get_unet22_medical_model("medical256")
```

## Train

No matching pretrained `UNet22(base_dim=32)` checkpoints are bundled with this
repo. The training script downloads the public dataset archives, resizes images
and masks to the required input size, applies common train-only segmentation
augmentation, then saves `best` and `last` checkpoints.

```bash
.venv/bin/python scripts/train_unet22_medical_seg.py \
  --dataset montgomery_lung_64 \
  --epochs 30 \
  --batch-size 2

.venv/bin/python scripts/train_unet22_medical_seg.py \
  --dataset kvasir_polyp_256 \
  --epochs 30 \
  --batch-size 2
```

Checkpoints are written to `checkpoints/medical_unet22/` by default.

Default train-time augmentation includes paired image/mask horizontal flips,
small rotations, random scale/translate, brightness/contrast/gamma jitter,
light Gaussian image noise, and occasional Gaussian blur. Validation data is
not augmented. Use `--no-augment` to disable this, or tune flags such as
`--rotate-degrees`, `--scale-min`, `--scale-max`, `--brightness`, `--contrast`,
`--noise-std`, and `--blur-p`.

## Load A Checkpoint

```python
import torch
from orion.models.unet import get_unet22_medical_model

ckpt = torch.load("checkpoints/medical_unet22/montgomery_lung_64_unet22_base32_best.pt")
model = get_unet22_medical_model(ckpt["model"]["name"], base_dim=ckpt["model"]["base_dim"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

Use `torch.sigmoid(model(x)) >= 0.5` for binary masks.
