from __future__ import annotations

import sys
from pathlib import Path


def test_experimental_cir_imports_without_haloed_bridge() -> None:
    before = list(sys.path)
    before_modules = set(sys.modules)

    from orion.experimental.cir import build_region_first_full_selector

    payload = build_region_first_full_selector()
    added_modules = set(sys.modules) - before_modules

    assert payload["status"] == "ok"
    assert sys.path == before
    assert not any("CLionProjects/HaloED" in path for path in sys.path)
    assert "haloed" not in added_modules
    assert not any(name.startswith("haloed.") for name in added_modules)
    assert "scripts" not in added_modules
    assert not any(name.startswith("scripts.cir") for name in added_modules)


def test_experimental_cir_sources_do_not_contain_bridge_imports() -> None:
    root = Path("orion/experimental/cir")
    forbidden = ("sys.path", "PYTHONPATH", "CLionProjects/HaloED", "from haloed", "import haloed", "from scripts", "import scripts")

    offenders = []
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append((str(path), token))

    assert offenders == []
