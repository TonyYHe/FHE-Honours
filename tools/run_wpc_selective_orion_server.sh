#!/usr/bin/env bash
# Stage 32 correctness first; optional dense-model diagnostic after it passes.
set -euo pipefail
TASK_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK_REPO"
TASK_RUN=${1:-.tmp/results/honours/32_wpc_selective_orion/server_run1}
TASK_NETWORK=${2:-}
TASK_PYTHON="$TASK_REPO/.venv/bin/python"
if [[ -e "$TASK_RUN" || -e "$TASK_RUN.exit_status" || -e "$TASK_RUN.preflight.log" ]]; then
    echo "ERROR: existing evidence at $TASK_RUN; choose a fresh run name" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$TASK_RUN")"
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_RUN.exit_status"' EXIT
test -x "$TASK_PYTHON" || { echo "ERROR: missing .venv/bin/python" >&2; exit 1; }
export ORION_WPC_SELECTIVE_POLICY=off
{
    git rev-parse HEAD
    "$TASK_PYTHON" tools/build_lattigo.py
    "$TASK_PYTHON" -m pytest -q tests/test_wpc_selective_orion.py \
        tests/test_wpc_orion_layout_control.py tests/test_wpc_layout_gate.py
    (cd orion/backend/lattigo && go test ./...)
} 2>&1 | tee "$TASK_RUN.preflight.log"
PYTHONUNBUFFERED=1 "$TASK_PYTHON" tools/run_wpc_selective_orion.py --out-dir "$TASK_RUN"
"$TASK_PYTHON" tools/run_wpc_selective_orion.py --review-dir "$TASK_RUN"
if [[ -n "$TASK_NETWORK" ]]; then
    PYTHONUNBUFFERED=1 "$TASK_PYTHON" tools/run_wpc_selective_model.py \
        --network "$TASK_NETWORK" --out-dir "$TASK_RUN/model_diagnostic" \
        --warmup-runs 1 --forward-runs 3 --atol 1e-3
    "$TASK_PYTHON" tools/run_wpc_selective_model.py --review-dir "$TASK_RUN/model_diagnostic"
fi
echo "SELECTIVE ORION GATE SUCCESS: $TASK_RUN/gate.json"
