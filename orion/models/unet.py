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
