from __future__ import annotations

from pathlib import Path

import pytest

from orion.experimental.u22_banked_regression import (
    U22_BANKED_REGRESSION_CASES,
    run_u22_banked_regression_case,
)


def _require_lattigo() -> None:
    if not Path("orion/backend/lattigo/lattigo-linux.so").exists():
        pytest.skip("local Lattigo shared library has not been built")


@pytest.mark.parametrize("case_name", tuple(U22_BANKED_REGRESSION_CASES.keys()))
def test_u22_banked_regression_runner_executes_on_lattigo(case_name: str) -> None:
    _require_lattigo()
    payload = run_u22_banked_regression_case(str(case_name), backend="lattigo")

    assert payload["case"] == str(case_name)
    assert payload["backend"] == "lattigo"
    assert payload["local_lattigo"] is True
    assert payload["experimental_kernel"] is True
    assert payload["supports_scheme"] is True
    assert payload["parity"]["exact"] is True
    assert payload["parity"]["max_abs"] <= payload["parity"]["tolerance"]
