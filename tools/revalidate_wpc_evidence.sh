#!/usr/bin/env bash
# Server-only recovery of content-bound validation; never retrains a model.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TASK_OUT=${1:-.tmp/results/honours/27_wpc_validation_provenance/server_run1}
TASK_PYTHON=.venv/bin/python
TASK_CHECKPOINT_DIR=checkpoints/wpc_rotation_padding_covid19_cheb7_audited_restart
TASK_CHECKPOINT="$TASK_CHECKPOINT_DIR/rotation_padding_best.pt"

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'ERROR: missing required file: %s\n' "$1" >&2
    exit 1
  fi
}
if [[ ! -x "$TASK_PYTHON" ]]; then
  printf 'ERROR: missing virtual-environment Python: %s\n' "$TASK_PYTHON" >&2
  exit 1
fi
require_file "$TASK_CHECKPOINT"
require_file "$TASK_CHECKPOINT_DIR/rotation_padding_last.pt"
require_file data/fhelipe_medseg/covid19radio_512.npz
for TASK_MODE in full compressed; do
  require_file ".tmp/results/honours/25_wpc_finetuned_decoder_isolated_benchmark/server_run1/$TASK_MODE.worker.json"
done
# Refuse to overwrite archived/previous recovery evidence.
mkdir -p "$(dirname "$TASK_OUT")"
mkdir "$TASK_OUT"
TASK_STAGE=setup
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUT/exit_status"; if [[ "$TASK_EXIT" != 0 ]]; then printf "ERROR: %s failed; inspect logs in %s\n" "$TASK_STAGE" "$TASK_OUT" >&2; fi' EXIT
sha256sum "$TASK_CHECKPOINT" > "$TASK_OUT/checkpoint_before.sha256"
git rev-parse HEAD > "$TASK_OUT/repository_commit.txt"
git status --short > "$TASK_OUT/repository_status.txt"
sha256sum \
  tools/revalidate_wpc_evidence.sh tools/finetune_wpc_rotation_padding.py \
  tools/medseg_cheb7_orion_adapter.py tools/run_wpc_cips_trained_decoder.py \
  tools/run_wpc_cips_trained_isolated_worker.py tools/synthesize_wpc_orion_tradeoff.py \
  orion/experimental/wpc_evidence_validation.py \
  orion/experimental/wpc_cips_trained_benchmark.py \
  orion/experimental/wpc_tradeoff_synthesis.py \
  > "$TASK_OUT/implementation.sha256"

TASK_STAGE=tests
"$TASK_PYTHON" -m pytest -q \
  tests/test_wpc_evidence_validation.py \
  tests/test_wpc_tradeoff_synthesis.py \
  tests/test_wpc_cips_trained_benchmark.py \
  tests/test_wpc_rotation_padding_training.py \
  > "$TASK_OUT/tests.log" 2>&1

TASK_STAGE=cuda_preflight
"$TASK_PYTHON" -c 'import json, torch; assert torch.cuda.is_available(), "CUDA-enabled PyTorch and an available GPU are required"; print(json.dumps({"torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}))' \
  > "$TASK_OUT/runtime.json"

printf 'Re-evaluating all 2,115 validation samples (no training) ...\n'
TASK_STAGE=full_validation
"$TASK_PYTHON" tools/finetune_wpc_rotation_padding.py \
  --dataset covid19 --data-root data/fhelipe_medseg --image-size 256 \
  --epochs 5 --batch-size 1 --train-limit 2048 --val-limit 2115 \
  --seed 0 --num-workers 2 --device cuda --eval-only \
  --out-dir "$TASK_CHECKPOINT_DIR" --result "$TASK_OUT/full_validation_2115.json" \
  > "$TASK_OUT/full_validation.log" 2>&1

printf 'Re-running decoder correctness with a recorded tolerance ...\n'
TASK_STAGE=decoder_correctness
"$TASK_PYTHON" tools/run_wpc_cips_trained_decoder.py \
  --checkpoint "$TASK_CHECKPOINT" --out "$TASK_OUT/decoder_correctness.json" \
  > "$TASK_OUT/decoder_correctness.log" 2>&1

printf 'Independently validating existing benchmark workers and rebuilding synthesis ...\n'
TASK_STAGE=synthesis
"$TASK_PYTHON" tools/synthesize_wpc_orion_tradeoff.py \
  --checkpoint "$TASK_CHECKPOINT" \
  --full-validation "$TASK_OUT/full_validation_2115.json" \
  --decoder-correctness "$TASK_OUT/decoder_correctness.json" \
  --out-dir "$TASK_OUT/synthesis" \
  > "$TASK_OUT/synthesis.log" 2>&1
TASK_STAGE=checkpoint_unchanged
sha256sum -c "$TASK_OUT/checkpoint_before.sha256"
printf 'VALIDATION RECOVERY COMPLETE: %s/synthesis/report.md\n' "$TASK_OUT"
