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


def _normalize_u22_activation(activation: str | None) -> str:
    if activation is None:
        return "none"
    normalized = str(activation).strip().lower()
    if normalized in {"", "none", "identity", "linear"}:
        return "none"
    if normalized in {"silu", "swish"}:
        return "silu"
    if normalized == "relu":
        return "relu"
    raise ValueError(f"Unsupported UNet22 activation {activation!r}")


def _make_u22_activation(activation: str | None, *, silu_degree: int):
    normalized = _normalize_u22_activation(activation)
    if normalized == "none":
        return None
    if normalized == "silu":
        return on.SiLU(degree=int(silu_degree))
    if normalized == "relu":
        return on.ReLU()
    raise AssertionError(f"unhandled UNet22 activation {normalized!r}")


def _apply_u22_activation(module, x):
    if module is None:
        return x
    return module(x)


def get_unet22_medical_model(
    name: str,
    *,
    base_channels: int | None = None,
    base_dim: int | None = None,
    activation: str | None = None,
    silu_degree: int = 31,
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
        activation=activation,
        silu_degree=int(silu_degree),
    )


class UNet22(on.Module):
    """
    A compact 22-layer U-Net-style encoder/decoder.

    The layout follows four downsample stages, a two-conv bottleneck, and four
    transposed-conv decoder stages. Skip paths use channel concatenation, the
    standard U-Net decoder merge.
    """

    def __init__(
        self,
        dataset: str = "tiny",
        in_channels: int | None = None,
        out_channels: int | None = None,
        base_channels: int | None = None,
        base_dim: int | None = None,
        activation: str | None = None,
        silu_degree: int = 31,
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
        self.activation = _normalize_u22_activation(activation)
        self.silu_degree = int(silu_degree)

        c1 = int(base)
        c2 = int(base * 2)
        c3 = int(base * 4)
        c4 = int(base * 8)
        c5 = int(base * 16)
        make_act = lambda: _make_u22_activation(self.activation, silu_degree=self.silu_degree)

        self.enc1a = on.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=True)
        self.enc1a_act = make_act()
        self.enc1b = on.Conv2d(c1, c1, kernel_size=3, padding=1, bias=True)
        self.enc1b_act = make_act()
        self.pool1 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc2a = on.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True)
        self.enc2a_act = make_act()
        self.enc2b = on.Conv2d(c2, c2, kernel_size=3, padding=1, bias=True)
        self.enc2b_act = make_act()
        self.pool2 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc3a = on.Conv2d(c2, c3, kernel_size=3, padding=1, bias=True)
        self.enc3a_act = make_act()
        self.enc3b = on.Conv2d(c3, c3, kernel_size=3, padding=1, bias=True)
        self.enc3b_act = make_act()
        self.pool3 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc4a = on.Conv2d(c3, c4, kernel_size=3, padding=1, bias=True)
        self.enc4a_act = make_act()
        self.enc4b = on.Conv2d(c4, c4, kernel_size=3, padding=1, bias=True)
        self.enc4b_act = make_act()
        self.pool4 = on.AvgPool2d(kernel_size=2, stride=2)

        self.bottlenecka = on.Conv2d(c4, c5, kernel_size=3, padding=1, bias=True)
        self.bottlenecka_act = make_act()
        self.bottleneckb = on.Conv2d(c5, c5, kernel_size=3, padding=1, bias=True)
        self.bottleneckb_act = make_act()

        self.up4 = on.ConvTranspose2d(c5, c4, kernel_size=2, stride=2, bias=True)
        self.cat4 = on.Concat(dim=1)
        self.dec4a = on.Conv2d(c4 + c4, c4, kernel_size=3, padding=1, bias=True)
        self.dec4a_act = make_act()
        self.dec4b = on.Conv2d(c4, c4, kernel_size=3, padding=1, bias=True)
        self.dec4b_act = make_act()

        self.up3 = on.ConvTranspose2d(c4, c3, kernel_size=2, stride=2, bias=True)
        self.cat3 = on.Concat(dim=1)
        self.dec3a = on.Conv2d(c3 + c3, c3, kernel_size=3, padding=1, bias=True)
        self.dec3a_act = make_act()
        self.dec3b = on.Conv2d(c3, c3, kernel_size=3, padding=1, bias=True)
        self.dec3b_act = make_act()

        self.up2 = on.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=True)
        self.cat2 = on.Concat(dim=1)
        self.dec2a = on.Conv2d(c2 + c2, c2, kernel_size=3, padding=1, bias=True)
        self.dec2a_act = make_act()
        self.dec2b = on.Conv2d(c2, c2, kernel_size=3, padding=1, bias=True)
        self.dec2b_act = make_act()

        self.up1 = on.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=True)
        self.cat1 = on.Concat(dim=1)
        self.dec1a = on.Conv2d(c1 + c1, c1, kernel_size=3, padding=1, bias=True)
        self.dec1a_act = make_act()
        self.dec1b = on.Conv2d(c1, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = _apply_u22_activation(self.enc1a_act, self.enc1a(x))
        skip1 = _apply_u22_activation(self.enc1b_act, self.enc1b(x))

        x = _apply_u22_activation(self.enc2a_act, self.enc2a(self.pool1(skip1)))
        skip2 = _apply_u22_activation(self.enc2b_act, self.enc2b(x))

        x = _apply_u22_activation(self.enc3a_act, self.enc3a(self.pool2(skip2)))
        skip3 = _apply_u22_activation(self.enc3b_act, self.enc3b(x))

        x = _apply_u22_activation(self.enc4a_act, self.enc4a(self.pool3(skip3)))
        skip4 = _apply_u22_activation(self.enc4b_act, self.enc4b(x))

        x = _apply_u22_activation(self.bottlenecka_act, self.bottlenecka(self.pool4(skip4)))
        x = _apply_u22_activation(self.bottleneckb_act, self.bottleneckb(x))

        x = _apply_u22_activation(self.dec4a_act, self.dec4a(self.cat4(self.up4(x), skip4)))
        x = _apply_u22_activation(self.dec4b_act, self.dec4b(x))
        x = _apply_u22_activation(self.dec3a_act, self.dec3a(self.cat3(self.up3(x), skip3)))
        x = _apply_u22_activation(self.dec3b_act, self.dec3b(x))
        x = _apply_u22_activation(self.dec2a_act, self.dec2a(self.cat2(self.up2(x), skip2)))
        x = _apply_u22_activation(self.dec2b_act, self.dec2b(x))
        x = _apply_u22_activation(self.dec1a_act, self.dec1a(self.cat1(self.up1(x), skip1)))
        x = self.dec1b(x)
        return x


class UNet22PlusOutput(UNet22):
    """
    UNet22 body plus a separate logits/output layer.

    The base UNet22 folds the output projection into dec1b. This variant keeps
    dec1b as a base->base decoder conv, applies the configured activation, and
    adds an explicit 1x1 output head.
    """

    def __init__(
        self,
        dataset: str = "kvasir_polyp_256",
        in_channels: int | None = None,
        out_channels: int | None = None,
        base_channels: int | None = None,
        base_dim: int | None = None,
        activation: str | None = None,
        silu_degree: int = 31,
    ):
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
        requested_out = int(default_out_channels if out_channels is None else out_channels)

        super().__init__(
            dataset=dataset,
            in_channels=in_channels,
            out_channels=base,
            base_channels=base,
            activation=activation,
            silu_degree=int(silu_degree),
        )
        self.out_channels = requested_out
        self.dec1b_act = _make_u22_activation(self.activation, silu_degree=self.silu_degree)
        self.output = on.Conv2d(base, requested_out, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = _apply_u22_activation(self.enc1a_act, self.enc1a(x))
        skip1 = _apply_u22_activation(self.enc1b_act, self.enc1b(x))

        x = _apply_u22_activation(self.enc2a_act, self.enc2a(self.pool1(skip1)))
        skip2 = _apply_u22_activation(self.enc2b_act, self.enc2b(x))

        x = _apply_u22_activation(self.enc3a_act, self.enc3a(self.pool2(skip2)))
        skip3 = _apply_u22_activation(self.enc3b_act, self.enc3b(x))

        x = _apply_u22_activation(self.enc4a_act, self.enc4a(self.pool3(skip3)))
        skip4 = _apply_u22_activation(self.enc4b_act, self.enc4b(x))

        x = _apply_u22_activation(self.bottlenecka_act, self.bottlenecka(self.pool4(skip4)))
        x = _apply_u22_activation(self.bottleneckb_act, self.bottleneckb(x))

        x = _apply_u22_activation(self.dec4a_act, self.dec4a(self.cat4(self.up4(x), skip4)))
        x = _apply_u22_activation(self.dec4b_act, self.dec4b(x))
        x = _apply_u22_activation(self.dec3a_act, self.dec3a(self.cat3(self.up3(x), skip3)))
        x = _apply_u22_activation(self.dec3b_act, self.dec3b(x))
        x = _apply_u22_activation(self.dec2a_act, self.dec2a(self.cat2(self.up2(x), skip2)))
        x = _apply_u22_activation(self.dec2b_act, self.dec2b(x))
        x = _apply_u22_activation(self.dec1a_act, self.dec1a(self.cat1(self.up1(x), skip1)))
        x = _apply_u22_activation(self.dec1b_act, self.dec1b(x))
        return self.output(x)


class UNet22Encoder(on.Module):
    """Encoder-only UNet22 prefix ending at enc4b."""

    def __init__(
        self,
        dataset: str = "kvasir_polyp_256",
        in_channels: int | None = None,
        base_channels: int | None = None,
        base_dim: int | None = None,
        activation: str | None = None,
        silu_degree: int = 31,
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
        default_in_channels, _default_out_channels = _u22_default_channels(dataset)
        in_channels = int(default_in_channels if in_channels is None else in_channels)
        self.dataset = dataset
        self.in_channels = in_channels
        self.base_channels = base
        self.activation = _normalize_u22_activation(activation)
        self.silu_degree = int(silu_degree)

        c1 = int(base)
        c2 = int(base * 2)
        c3 = int(base * 4)
        c4 = int(base * 8)
        make_act = lambda: _make_u22_activation(self.activation, silu_degree=self.silu_degree)

        self.enc1a = on.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=True)
        self.enc1a_act = make_act()
        self.enc1b = on.Conv2d(c1, c1, kernel_size=3, padding=1, bias=True)
        self.enc1b_act = make_act()
        self.pool1 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc2a = on.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True)
        self.enc2a_act = make_act()
        self.enc2b = on.Conv2d(c2, c2, kernel_size=3, padding=1, bias=True)
        self.enc2b_act = make_act()
        self.pool2 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc3a = on.Conv2d(c2, c3, kernel_size=3, padding=1, bias=True)
        self.enc3a_act = make_act()
        self.enc3b = on.Conv2d(c3, c3, kernel_size=3, padding=1, bias=True)
        self.enc3b_act = make_act()
        self.pool3 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc4a = on.Conv2d(c3, c4, kernel_size=3, padding=1, bias=True)
        self.enc4a_act = make_act()
        self.enc4b = on.Conv2d(c4, c4, kernel_size=3, padding=1, bias=True)
        self.enc4b_act = make_act()

    def forward(self, x):
        x = _apply_u22_activation(self.enc1a_act, self.enc1a(x))
        skip1 = _apply_u22_activation(self.enc1b_act, self.enc1b(x))

        x = _apply_u22_activation(self.enc2a_act, self.enc2a(self.pool1(skip1)))
        skip2 = _apply_u22_activation(self.enc2b_act, self.enc2b(x))

        x = _apply_u22_activation(self.enc3a_act, self.enc3a(self.pool2(skip2)))
        skip3 = _apply_u22_activation(self.enc3b_act, self.enc3b(x))

        x = _apply_u22_activation(self.enc4a_act, self.enc4a(self.pool3(skip3)))
        return _apply_u22_activation(self.enc4b_act, self.enc4b(x))


class UNet22MontgomeryLung64(UNet22):
    """UNet22(base_dim=32) for 64x64 Montgomery CXR lung segmentation."""

    def __init__(
        self,
        base_channels: int | None = None,
        base_dim: int | None = 32,
        activation: str | None = None,
        silu_degree: int = 31,
    ):
        super().__init__(
            dataset="montgomery_lung_64",
            in_channels=1,
            out_channels=1,
            base_channels=base_channels,
            base_dim=base_dim,
            activation=activation,
            silu_degree=int(silu_degree),
        )


class UNet22KvasirPolyp256(UNet22):
    """UNet22(base_dim=32) for 256x256 Kvasir-SEG polyp segmentation."""

    def __init__(
        self,
        base_channels: int | None = None,
        base_dim: int | None = 32,
        activation: str | None = None,
        silu_degree: int = 31,
    ):
        super().__init__(
            dataset="kvasir_polyp_256",
            in_channels=3,
            out_channels=1,
            base_channels=base_channels,
            base_dim=base_dim,
            activation=activation,
            silu_degree=int(silu_degree),
        )
