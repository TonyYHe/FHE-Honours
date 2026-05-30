#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path


def _default_output(source: Path) -> Path:
    system = platform.system()
    if system == "Linux":
        return source.with_name("clear_lattigo-linux.so")
    if system == "Darwin":
        machine = platform.machine().lower()
        if machine in {"arm64", "aarch64"}:
            return source.with_name("clear_lattigo-mac-arm64.dylib")
        return source.with_name("clear_lattigo-mac.dylib")
    if system == "Windows":
        return source.with_name("clear_lattigo-windows.dll")
    raise RuntimeError(f"unsupported platform: {system}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the clear Lattigo C ABI backend.")
    parser.add_argument(
        "--source",
        default="orion/backend/clear_lattigo/clear_lattigo.cpp",
        help="C++ source path",
    )
    parser.add_argument("--output", default="", help="Output shared library path")
    parser.add_argument("--cxx", default="c++", help="C++ compiler")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve() if args.output else _default_output(source).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.cxx),
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    print(output)


if __name__ == "__main__":
    main()
