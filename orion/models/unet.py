from dataclasses import dataclass

import orion.nn as on


class MiniUNet(on.Module):
    """
    Tiny U-Net-style encoder/decoder used for transposed-conv development.

    It keeps just one downsample/upsample pair so compilation stays quick while
    still exercising the layout changes around strided conv + conv transpose.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 4, out_channels: int = 1):
        super().__init__()
        hidden = int(base_channels)
        bottleneck = hidden * 2

        self.enc = on.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=True)
        self.down = on.Conv2d(hidden, bottleneck, kernel_size=3, stride=2, padding=1, bias=True)
        self.mid = on.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1, bias=True)
        self.up = on.ConvTranspose2d(bottleneck, hidden, kernel_size=2, stride=2, bias=True)
        self.skip = on.Conv2d(hidden, hidden, kernel_size=1, bias=True)
        self.add = on.Add()
        self.out = on.Conv2d(hidden, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        skip = self.enc(x)
        x = self.down(skip)
        x = self.mid(x)
        x = self.up(x)
        x = self.add(x, self.skip(skip))
        return self.out(x)


@dataclass(frozen=True)
class UNet22MedicalSpec:
    name: str
    dataset_name: str
    task: str
    image_size: int
    in_channels: int
    out_channels: int
    base_dim: int
    source_url: str


_U22_DATASET_ALIASES = {
    "medical64": "montgomery_lung_64",
    "lung64": "montgomery_lung_64",
    "montgomery": "montgomery_lung_64",
    "montgomery_lung": "montgomery_lung_64",
    "medical256": "kvasir_polyp_256",
    "polyp256": "kvasir_polyp_256",
    "kvasir": "kvasir_polyp_256",
    "kvasir_polyp": "kvasir_polyp_256",
}


_U22_BASE_CHANNEL_DEFAULTS = {
    "tiny": 4,
    "imagenet": 1,
    "montgomery_lung_64": 32,
    "kvasir_polyp_256": 32,
}


_U22_CHANNEL_DEFAULTS = {
    "tiny": (3, 3),
    "imagenet": (3, 3),
    "montgomery_lung_64": (1, 1),
    "kvasir_polyp_256": (3, 1),
}


_U22_MEDICAL_SPECS = {
    "montgomery_lung_64": UNet22MedicalSpec(
        name="montgomery_lung_64",
        dataset_name="Montgomery County Chest X-ray Set",
        task="binary lung field segmentation",
        image_size=64,
        in_channels=1,
        out_channels=1,
        base_dim=32,
        source_url="https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
    ),
    "kvasir_polyp_256": UNet22MedicalSpec(
        name="kvasir_polyp_256",
        dataset_name="Kvasir-SEG",
        task="binary gastrointestinal polyp segmentation",
        image_size=256,
        in_channels=3,
        out_channels=1,
        base_dim=32,
        source_url="https://datasets.simula.no/downloads/kvasir-seg.zip",
    ),
}


def _normalize_u22_dataset(dataset: str) -> str:
    dataset = str(dataset).lower()
    return _U22_DATASET_ALIASES.get(dataset, dataset)


def _u22_default_base_channels(dataset: str) -> int:
    dataset = _normalize_u22_dataset(dataset)
    if dataset in _U22_BASE_CHANNEL_DEFAULTS:
        return int(_U22_BASE_CHANNEL_DEFAULTS[dataset])
    raise ValueError(f"UNet22 with dataset {dataset!r} is not supported.")


def _u22_default_channels(dataset: str) -> tuple[int, int]:
    dataset = str(dataset).lower()
    dataset = _normalize_u22_dataset(dataset)
    if dataset in _U22_CHANNEL_DEFAULTS:
        return _U22_CHANNEL_DEFAULTS[dataset]
    raise ValueError(f"UNet22 with dataset {dataset!r} is not supported.")


def get_unet22_medical_spec(name: str) -> UNet22MedicalSpec:
    name = _normalize_u22_dataset(name)
    if name not in _U22_MEDICAL_SPECS:
        supported = ", ".join(sorted(_U22_MEDICAL_SPECS))
        raise ValueError(f"Unknown UNet22 medical model {name!r}. Supported: {supported}.")
    return _U22_MEDICAL_SPECS[name]


def list_unet22_medical_specs() -> tuple[UNet22MedicalSpec, ...]:
    return tuple(_U22_MEDICAL_SPECS[name] for name in sorted(_U22_MEDICAL_SPECS))


def get_unet22_medical_model(
    name: str,
    *,
    base_channels: int | None = None,
    base_dim: int | None = None,
) -> "UNet22":
    spec = get_unet22_medical_spec(name)
    requested_base = base_channels if base_channels is not None else base_dim
    if requested_base is None:
        requested_base = spec.base_dim
    return UNet22(
        dataset=spec.name,
        in_channels=spec.in_channels,
        out_channels=spec.out_channels,
        base_channels=requested_base,
    )


class UNet22(on.Module):
    """
    A compact 22-layer U-Net-style encoder/decoder.

    The layout follows four downsample stages, a two-conv bottleneck, and four
    transposed-conv decoder stages. Skip paths use additive merges so the model
    stays compatible with Orion's current operator set.
    """

    def __init__(
        self,
        dataset: str = "tiny",
        in_channels: int | None = None,
        out_channels: int | None = None,
        base_channels: int | None = None,
        base_dim: int | None = None,
    ):
        super().__init__()
        dataset = _normalize_u22_dataset(dataset)
        if base_channels is not None and base_dim is not None and int(base_channels) != int(base_dim):
            raise ValueError(
                f"Received conflicting UNet22 base sizes: "
                f"base_channels={base_channels} and base_dim={base_dim}."
            )
        requested_base = base_channels if base_channels is not None else base_dim
        base = int(requested_base if requested_base is not None else _u22_default_base_channels(dataset))
        default_in_channels, default_out_channels = _u22_default_channels(dataset)
        in_channels = int(default_in_channels if in_channels is None else in_channels)
        out_channels = int(default_out_channels if out_channels is None else out_channels)
        self.dataset = dataset
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base

        c1 = int(base)
        c2 = int(base * 2)
        c3 = int(base * 4)
        c4 = int(base * 8)
        c5 = int(base * 16)

        self.enc1a = on.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=True)
        self.enc1b = on.Conv2d(c1, c1, kernel_size=3, padding=1, bias=True)
        self.pool1 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc2a = on.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True)
        self.enc2b = on.Conv2d(c2, c2, kernel_size=3, padding=1, bias=True)
        self.pool2 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc3a = on.Conv2d(c2, c3, kernel_size=3, padding=1, bias=True)
        self.enc3b = on.Conv2d(c3, c3, kernel_size=3, padding=1, bias=True)
        self.pool3 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc4a = on.Conv2d(c3, c4, kernel_size=3, padding=1, bias=True)
        self.enc4b = on.Conv2d(c4, c4, kernel_size=3, padding=1, bias=True)
        self.pool4 = on.AvgPool2d(kernel_size=2, stride=2)

        self.bottlenecka = on.Conv2d(c4, c5, kernel_size=3, padding=1, bias=True)
        self.bottleneckb = on.Conv2d(c5, c5, kernel_size=3, padding=1, bias=True)

        self.up4 = on.ConvTranspose2d(c5, c4, kernel_size=2, stride=2, bias=True)
        self.add4 = on.Add()
        self.dec4a = on.Conv2d(c4, c4, kernel_size=3, padding=1, bias=True)
        self.dec4b = on.Conv2d(c4, c4, kernel_size=3, padding=1, bias=True)

        self.up3 = on.ConvTranspose2d(c4, c3, kernel_size=2, stride=2, bias=True)
        self.add3 = on.Add()
        self.dec3a = on.Conv2d(c3, c3, kernel_size=3, padding=1, bias=True)
        self.dec3b = on.Conv2d(c3, c3, kernel_size=3, padding=1, bias=True)

        self.up2 = on.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=True)
        self.add2 = on.Add()
        self.dec2a = on.Conv2d(c2, c2, kernel_size=3, padding=1, bias=True)
        self.dec2b = on.Conv2d(c2, c2, kernel_size=3, padding=1, bias=True)

        self.up1 = on.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=True)
        self.add1 = on.Add()
        self.dec1a = on.Conv2d(c1, c1, kernel_size=3, padding=1, bias=True)
        self.dec1b = on.Conv2d(c1, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip1 = self.enc1b(self.enc1a(x))
        skip2 = self.enc2b(self.enc2a(self.pool1(skip1)))
        skip3 = self.enc3b(self.enc3a(self.pool2(skip2)))
        skip4 = self.enc4b(self.enc4a(self.pool3(skip3)))

        x = self.bottleneckb(self.bottlenecka(self.pool4(skip4)))

        x = self.dec4b(self.dec4a(self.add4(self.up4(x), skip4)))
        x = self.dec3b(self.dec3a(self.add3(self.up3(x), skip3)))
        x = self.dec2b(self.dec2a(self.add2(self.up2(x), skip2)))
        x = self.dec1b(self.dec1a(self.add1(self.up1(x), skip1)))
        return x


class UNet22MontgomeryLung64(UNet22):
    """UNet22(base_dim=32) for 64x64 Montgomery CXR lung segmentation."""

    def __init__(self, base_channels: int | None = None, base_dim: int | None = 32):
        super().__init__(
            dataset="montgomery_lung_64",
            in_channels=1,
            out_channels=1,
            base_channels=base_channels,
            base_dim=base_dim,
        )


class UNet22KvasirPolyp256(UNet22):
    """UNet22(base_dim=32) for 256x256 Kvasir-SEG polyp segmentation."""

    def __init__(self, base_channels: int | None = None, base_dim: int | None = 32):
        super().__init__(
            dataset="kvasir_polyp_256",
            in_channels=3,
            out_channels=1,
            base_channels=base_channels,
            base_dim=base_dim,
        )
