from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIAG_BUILDER_LIB = ROOT / "orion" / "backend" / "diag_builder" / "diag_builder-linux.so"


@pytest.fixture(autouse=True)
def _build_diag_builder_when_enabled(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    enabled = str(os.environ.get("ORION_CPP_DIAG_BUILDER", "")).strip().lower()
    name = str(getattr(getattr(request, "node", None), "name", "") or "")
    needs_builder = "diag_builder" in name or "cpp_dense_conv2d" in name
    if enabled not in {"1", "true", "yes", "on"} and not bool(needs_builder):
        return
    if not DIAG_BUILDER_LIB.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "build_diag_builder.py")],
            cwd=str(ROOT),
            check=True,
        )
    monkeypatch.setenv("ORION_DIAG_BUILDER_LIB", str(DIAG_BUILDER_LIB))
