import orion.nn as on


def _normalize_activation(activation: str | None) -> str:
    if activation is None:
        return "none"
    normalized = str(activation).strip().lower()
    if normalized in {"", "none", "identity", "linear"}:
        return "none"
    if normalized in {"relu", "silu", "swish"}:
        return "silu" if normalized == "swish" else normalized
    raise ValueError(f"Unsupported TernausVGGUNet activation {activation!r}")


def _make_activation(activation: str | None, *, silu_degree: int):
    normalized = _normalize_activation(activation)
    if normalized == "none":
        return None
    if normalized == "relu":
        return on.ReLU()
    if normalized == "silu":
        return on.SiLU(degree=int(silu_degree))
    raise AssertionError(f"unhandled activation {normalized!r}")


def _apply_activation(module, x):
    if module is None:
        return x
    return module(x)


class TernausVGGUNet(on.Module):
    """VGG11/TernausNet-style U-Net assembled from Orion-supported modules."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        base_dim: int = 64,
        activation: str | None = None,
        silu_degree: int = 31,
    ):
        super().__init__()
        base = int(base_dim)
        c1 = int(base)
        c2 = int(base * 2)
        c3 = int(base * 4)
        c4 = int(base * 8)
        c5 = int(min(base * 16, 512))
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_dim = int(base)
        self.channels = (c1, c2, c3, c4, c5)
        self.activation = _normalize_activation(activation)
        self.silu_degree = int(silu_degree)
        make_act = lambda: _make_activation(self.activation, silu_degree=self.silu_degree)

        self.enc1a = on.Conv2d(int(in_channels), c1, kernel_size=3, padding=1, bias=True)
        self.enc1a_act = make_act()
        self.pool1 = on.AvgPool2d(kernel_size=2, stride=2)

        self.enc2a = on.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True)
        self.enc2a_act = make_act()
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

        self.centera = on.Conv2d(c4, c5, kernel_size=3, padding=1, bias=True)
        self.centera_act = make_act()
        self.centerb = on.Conv2d(c5, c5, kernel_size=3, padding=1, bias=True)
        self.centerb_act = make_act()

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
        self.out = on.Conv2d(c1, int(out_channels), kernel_size=1, bias=True)

    def forward(self, x):
        skip1 = _apply_activation(self.enc1a_act, self.enc1a(x))

        skip2 = _apply_activation(self.enc2a_act, self.enc2a(self.pool1(skip1)))

        x = _apply_activation(self.enc3a_act, self.enc3a(self.pool2(skip2)))
        skip3 = _apply_activation(self.enc3b_act, self.enc3b(x))

        x = _apply_activation(self.enc4a_act, self.enc4a(self.pool3(skip3)))
        skip4 = _apply_activation(self.enc4b_act, self.enc4b(x))

        x = _apply_activation(self.centera_act, self.centera(self.pool4(skip4)))
        x = _apply_activation(self.centerb_act, self.centerb(x))

        x = _apply_activation(self.dec4a_act, self.dec4a(self.cat4(self.up4(x), skip4)))
        x = _apply_activation(self.dec4b_act, self.dec4b(x))

        x = _apply_activation(self.dec3a_act, self.dec3a(self.cat3(self.up3(x), skip3)))
        x = _apply_activation(self.dec3b_act, self.dec3b(x))

        x = _apply_activation(self.dec2a_act, self.dec2a(self.cat2(self.up2(x), skip2)))
        x = _apply_activation(self.dec2b_act, self.dec2b(x))

        x = _apply_activation(self.dec1a_act, self.dec1a(self.cat1(self.up1(x), skip1)))
        return self.out(x)
