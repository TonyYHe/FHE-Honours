from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def _output_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "diag_builder-mac.dylib"
    if system == "Windows":
        return "diag_builder-windows.dll"
    return "diag_builder-linux.so"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    src = root / "orion" / "backend" / "diag_builder" / "diag_builder.cpp"
    out = root / "orion" / "backend" / "diag_builder" / _output_name()
    cmd = [
        "c++",
        "-std=c++17",
        "-O3",
        "-shared",
        "-fPIC",
        str(src),
        "-o",
        str(out),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(root), check=True)
    print(f"built {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
