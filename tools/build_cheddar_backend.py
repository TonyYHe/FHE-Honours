import os
import platform
import subprocess
import sys
from pathlib import Path


def build(setup_kwargs=None):
    root_dir = Path(__file__).parent.parent
    backend_dir = root_dir / "orion" / "backend" / "cheddar"
    build_dir = backend_dir / "build"

    cheddar_root = os.environ.get("CHEDDAR_ROOT", "").strip()
    if not cheddar_root:
        raise RuntimeError("Set CHEDDAR_ROOT to the Cheddar repository path before building")

    build_dir.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        "cmake",
        "-S",
        str(backend_dir),
        "-B",
        str(build_dir),
        f"-DCHEDDAR_ROOT={cheddar_root}",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), "-j"]

    try:
        print(f"Running: {' '.join(configure_cmd)}")
        subprocess.run(configure_cmd, cwd=str(backend_dir), check=True)
        print(f"Running: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, cwd=str(backend_dir), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Cheddar backend build failed with exit code {e.returncode}")
        sys.exit(1)

    return setup_kwargs or {}


if __name__ == "__main__":
    build()
    sys.exit(0)
