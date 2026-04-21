import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orion
import orion.models as models


CONFIG = {
    "ckks_params": {
        "LogN": 10,
        "LogQ": [45, 35, 35, 35, 35, 45],
        "LogP": [50],
        "LogScale": 35,
        "H": 64,
        "RingType": "Standard",
    },
    "orion": {
        "margin": 2,
        "embedding_method": "square",
        "backend": "python",
        "fuse_modules": True,
        "debug": False,
        "io_mode": "none",
    },
}


def main() -> None:
    torch.manual_seed(0)

    scheme = orion.init_scheme(CONFIG)
    net = models.MiniUNet(in_channels=1, base_channels=4, out_channels=1)
    net.eval()

    inp = torch.randn(1, 1, 8, 8)
    out_clear = net(inp)

    orion.fit(net, inp)
    input_level = orion.compile(net)

    vec_ptxt = orion.encode(inp, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)

    net.he()
    out_ctxt = net(vec_ctxt)
    out_fhe = out_ctxt.decrypt().decode()

    print("clear shape:", tuple(out_clear.shape))
    print("fhe shape:", tuple(out_fhe.shape))
    print("max abs diff:", float((out_clear - out_fhe).abs().max().detach()))
    scheme.delete_scheme()


if __name__ == "__main__":
    main()
