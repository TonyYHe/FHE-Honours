#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_RESULT_ROOT, timestamp


SCRIPT_DIR = Path(__file__).resolve().parent
TARGETS = {
    "fig7": "fig7_operator_conv_perf.py",
    "table1": "table1_e2e_unet.py",
    "fig8": "fig8_cheb7_qualitative.py",
    "table2": "table2_dp_layout_breakdown.py",
    "table3": "table3_relayout_ablation.py",
    "bootstrap": "bootstrap_analysis_numbers.py",
}
E2E_TARGETS = {"table1", "fig8", "bootstrap"}


def _run(command: list[str | Path]) -> int:
    print(" ".join(str(part) for part in command), flush=True)
    return int(subprocess.call([str(part) for part in command]))


def _existing_child(base: Path, target: str) -> Path:
    direct = base / target
    if direct.exists():
        return direct
    matches = sorted(path for path in base.glob(f"{target}_*") if path.is_dir())
    if matches:
        return matches[-1]
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HaloED paper evaluation artifacts from Orion workflows.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RESULT_ROOT / f"build_{timestamp()}")
    parser.add_argument("--check-existing", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="+", choices=tuple(TARGETS), default=None)
    parser.add_argument("--skip-e2e", action="store_true", help="Skip long E2E/CKKS targets: table1, fig8, bootstrap.")
    args = parser.parse_args()

    selected = list(args.only or TARGETS.keys())
    if args.skip_e2e:
        selected = [target for target in selected if target not in E2E_TARGETS]

    base = Path(args.check_existing or args.run_root).resolve()
    base.mkdir(parents=True, exist_ok=True)
    table1_root = _existing_child(base, "table1") if args.check_existing else base / "table1"

    for target in selected:
        script = SCRIPT_DIR / TARGETS[target]
        command: list[str | Path] = [sys.executable, script]
        if args.check_existing:
            command.extend(["--check-existing", _existing_child(base, target)])
        else:
            command.extend(["--run-root", base / target])
        if args.dry_run:
            command.append("--dry-run")
        if args.force:
            command.append("--force")
        if target == "bootstrap":
            command.extend(["--e2e-root", table1_root])
        rc = _run(command)
        if rc != 0:
            return rc
    print(f"artifact_root: {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
