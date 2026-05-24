import torch.nn as nn
import orion.nn as on

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _make_activation(activation="relu", silu_degree=127):
    activation = str(activation or "relu").lower()
    if activation == "relu":
        return on.ReLU(degrees=[15, 15, 27])
    if activation == "silu":
        return on.SiLU(degree=int(silu_degree))
    raise ValueError(f"Unsupported VGG activation {activation!r}")


class VGG(on.Module):
    def __init__(
        self,
        vgg_name,
        dataset="cifar10",
        activation="relu",
        silu_degree=127,
        num_classes=None,
    ):
        super().__init__()
        dataset = str(dataset or "cifar10").lower()
        if num_classes is None:
            num_classes = 1000 if dataset == "imagenet" else 10
        classifier_in = 512 * 7 * 7 if dataset == "imagenet" else 512
        self.features = self._make_layers(
            cfg[vgg_name],
            activation=activation,
            silu_degree=int(silu_degree),
        )
        self.classifier = on.Linear(classifier_in, int(num_classes))
        self.flatten = on.Flatten()

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, activation="relu", silu_degree=127):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [on.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [on.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           on.BatchNorm2d(x),
                           _make_activation(activation, silu_degree)]
                in_channels = x
        layers += [on.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    

if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    net = VGG('VGG16')
    net.eval()

    x = torch.randn(1,3,32,32)
    total_flops = FlopCountAnalysis(net, x).total()

    summary(net, (3,32,32), depth=10)
    print("Total flops: ", total_flops)
