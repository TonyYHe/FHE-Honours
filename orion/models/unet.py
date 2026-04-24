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


def _u22_default_base_channels(dataset: str) -> int:
    dataset = str(dataset).lower()
    if dataset == "tiny":
        return 4
    if dataset == "imagenet":
        return 1
    raise ValueError(f"UNet22 with dataset {dataset!r} is not supported.")


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
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int | None = None,
    ):
        super().__init__()
        dataset = str(dataset).lower()
        base = int(base_channels if base_channels is not None else _u22_default_base_channels(dataset))

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
