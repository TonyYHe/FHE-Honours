from __future__ import annotations

import argparse
import json
from pathlib import Path

from orion.experimental.u22_banked_regression import (
    U22_BANKED_REGRESSION_CASES,
    run_u22_banked_regression_case,
)


DEFAULT_OUT = Path("/tmp/orion_u22_banked_regression.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inferred U22 banked regression cases on a selected backend.")
    parser.add_argument("--backend", choices=("lattigo", "python"), default="lattigo")
    parser.add_argument("--cases", nargs="*", default=list(U22_BANKED_REGRESSION_CASES.keys()))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = [run_u22_banked_regression_case(str(case_name), backend=str(args.backend)) for case_name in args.cases]
    payload = {
        "status": "ok" if all(str(row.get("status")) == "ok" for row in rows) else "failed",
        "scope": "u22_banked_regression_runner",
        "backend": str(args.backend),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
