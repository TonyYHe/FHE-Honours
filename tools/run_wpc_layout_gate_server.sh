#!/usr/bin/env bash
# Stage 30 correctness/resource gate; safe under nohup, never overwrite evidence.
set -euo pipefail
TASK_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK_REPO"
TASK_RUN=${1:-.tmp/results/honours/30_wpc_orion_layout_gate/server_run1}
TASK_CHECKPOINT=${2:-checkpoints/wpc_rotation_padding_covid19_cheb7_audited_restart/rotation_padding_best.pt}
TASK_RSS_MIB=${3:-8192}
TASK_PYTHON="$TASK_REPO/.venv/bin/python"
if [[ -e "$TASK_RUN" || -e "$TASK_RUN.exit_status" || -e "$TASK_RUN.preflight.log" ]]; then
    echo "ERROR: existing evidence at $TASK_RUN; choose a fresh run name" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$TASK_RUN")"
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_RUN.exit_status"' EXIT
test -x "$TASK_PYTHON" || { echo "ERROR: missing .venv/bin/python" >&2; exit 1; }
test -f "$TASK_CHECKPOINT" || { echo "ERROR: missing checkpoint: $TASK_CHECKPOINT" >&2; exit 1; }
{
    git rev-parse HEAD
    "$TASK_PYTHON" tools/build_lattigo.py
    "$TASK_PYTHON" -m pytest -q tests/test_wpc_orion_layout_control.py \
        tests/test_wpc_layout_gate.py tests/test_wpc_online_encode_integration.py \
        tests/test_wpc_online_encode_benchmark.py tests/test_wpc_decoder_geometry.py
    (cd orion/backend/lattigo && go test ./...)
} 2>&1 | tee "$TASK_RUN.preflight.log"
PYTHONUNBUFFERED=1 "$TASK_PYTHON" tools/run_wpc_layout_gate.py \
    --checkpoint "$TASK_CHECKPOINT" --out-dir "$TASK_RUN" \
    --ckks-config configs/wpc_decoder_scale_functional.json --height 16 --width 16 \
    --max-worker-rss-mib "$TASK_RSS_MIB" --worker-timeout-s 1800
echo "MATCHED LAYOUT GATE SUCCESS: $TASK_RUN/gate.md"
