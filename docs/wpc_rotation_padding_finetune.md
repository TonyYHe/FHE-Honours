# WPC Rotation-Padding-aware fine-tuning

## Purpose

The checkpoint-trained decoder experiment established that WPC-compressed CIPS
evaluation is correct, but also measured a semantic difference between WPC
Rotation Padding and the checkpoint model's native zero padding. This stage
provides the missing training and accuracy-validation path.

It starts from the exact COVID-19 U-Net22-plus-output Cheb7 checkpoint,
replaces all 18 spatial `3x3` convolutions with differentiable WPC Rotation
Padding, evaluates the immediate accuracy change, fine-tunes the converted
model, and reports validation Dice/IoU against both labels and the original
zero-padded model.

This is clear PyTorch training. It produces weights compatible with the WPC
CIPS FHE runtime but does not itself measure FHE latency.

## Padding operation

For an input `x` with spatial shape `H x W`, output position `(h,w)`, and
kernel position `(kh,kw)`, the converted convolution reads

```text
s = (hW + w + (kh-PH)W + (kw-PW)) mod HW.
```

The implementation flattens each channel's spatial plane, constructs the
nine cyclic shifts for a `3x3` kernel, and contracts them with the original
weights. The operation is differentiable with respect to inputs, weights, and
biases.

This is not `padding="circular"`: a horizontal offset at a row boundary moves
through the single flattened sequence rather than wrapping within that row.

## Checkpoint compatibility

Conversion preserves every state-dict key and tensor value:

- all 18 same-padded `3x3` convolutions are converted;
- four `2x2` transposed convolutions remain unchanged;
- the `1x1` output head remains unchanged;
- the 18 trained degree-7 Chebyshev activations remain fixed and require
  `blend_alpha=1`; and
- the fine-tuned state dict can be loaded strictly into the original Orion
  U-Net schema.

The source checkpoint SHA-256 is recorded in every result and training
checkpoint.

## Training and validation protocol

The runner records three validation points:

1. the original zero-padded checkpoint;
2. the same weights immediately after Rotation-Padding conversion; and
3. the best Rotation-Padding checkpoint selected by validation Dice.

Training uses the segmentation loss already used by the medseg pipeline. An
optional MSE distillation term against the frozen native checkpoint reduces
unnecessary output drift while the segmentation labels adapt the boundary
semantics.

For each validation point the result contains:

- loss, Dice, and IoU;
- mean and maximum logit delta from the native model;
- probability MAE from the native model; and
- prediction flip rate at threshold 0.5.

`rotation_padding_last.pt` is written atomically after every epoch, and
`--resume-if-present` continues from it. A checkpoint is eligible to become
the best checkpoint only when every validation sample has finite logits and
its loss, Dice, and IoU are finite. On resume, the runner re-evaluates the last
checkpoint and promotes it when it is the first fully finite checkpoint or it
improves Dice. This also recovers a completed epoch if an older runner failed
while writing its final report.

Training shuffles are derived from `seed + epoch - 1`, so a resumed epoch uses
the same sample order as an uninterrupted run. If an epoch produces a
non-finite loss, gradient, or validation output, the runner reloads the last
completed checkpoint, reduces the learning rate, and retries the entire epoch
without skipping samples. Each failed attempt, effective learning rate, retry
count, and shuffle seed is retained in the result. `--resume-lr` can lower the
optimizer rate when continuing an older checkpoint.

Before an epoch is accepted, the resulting checkpoint is evaluated without
updates over all selected training samples as well as the validation set. A
single non-finite training sample rejects and rolls back the whole epoch.
Every accepted state is also saved as an immutable
`rotation_padding_epoch_NNNN.pt` checkpoint, preventing a later apparently
successful epoch from erasing the last independently auditable state. Resume
is allowed only from schema-v4 checkpoints with complete audits and all prior
epoch files; schema-v3 checkpoints are rejected. A new non-resume run also
refuses to overwrite checkpoint files already present in its output directory.

The un-fine-tuned Rotation-Padding model can drive the fixed Chebyshev
polynomials beyond their fitted domains on some images. Validation therefore
accounts for every sample explicitly. It records the indices and counts of
non-finite samples, calculates diagnostic metrics only over the finite subset,
labels those metrics `finite_samples_only`, and writes `null` for comparisons
that are not scientifically valid. The result remains strict JSON and never
encodes `NaN` or infinity.

## Acceptance gates

The final result is valid only when:

- the source checkpoint hash is recorded;
- exactly 18 spatial convolutions use WPC Rotation Padding;
- conversion preserves all checkpoint keys and values;
- all 18 activations are pure degree-7 polynomial paths;
- native and selected-best validation metrics are fully finite;
- every pre-fine-tuning validation sample is accounted for, including any
  explicitly reported non-finite sample;
- a nonzero Rotation-Padding semantic change is observed;
- the selected best model is not worse than the un-fine-tuned model when the
  two validation results are comparable;
- every completed epoch processed every selected training sample;
- every accepted epoch passes a fully finite no-update training-set audit;
- immutable checkpoints exist for epoch zero and every completed epoch;
- all requested epochs complete, unless `--eval-only` is used;
- the best checkpoint strictly reloads through the original Orion schema; and
- both best and last checkpoints exist.

An improvement over the native zero-padded model is reported but is not an
acceptance requirement. It would be scientifically invalid to guarantee such
an improvement before running the experiment.

## Local verification

The implementation passed:

- an exact comparison with the independent NumPy WPC convolution oracle;
- an explicit boundary test showing WPC differs from zero padding;
- gradient checks for input, weights, and bias;
- a full U-Net conversion test proving 18 replacements and unchanged state;
- a one-sample training smoke run;
- resume from epoch one into epoch two; and
- evaluation-only reload of the saved best checkpoint;
- strict-JSON reporting when the pre-fine-tuning model emits non-finite logits;
- acceptance of a fully finite checkpoint that recovers from a non-finite
  epoch-zero baseline.

The synthetic smoke run validates mechanics only and is not an accuracy
result. Dataset-scale metrics must come from the server run.

## Files

- `orion/experimental/wpc_rotation_padding_training.py` implements the
  differentiable padding operation and state-compatible model conversion.
- `tools/finetune_wpc_rotation_padding.py` performs dataset loading, baseline
  evaluation, fine-tuning, resumable checkpointing, comparison, and reporting.
- `tests/test_wpc_rotation_padding_training.py` validates mathematical,
  gradient, and checkpoint-conversion behavior.

## Server execution

The bounded protocol uses the same deterministic 2,048 training and 512
validation samples selected with seed zero. The full five-epoch run is
intended for a CUDA server. First verify that the dataset exists:

```bash
ls -lh data/fhelipe_medseg/covid19radio_512.npz
```

Then run the job under `nohup`:

```bash
OUT=.tmp/results/honours/22_wpc_rotation_padding_finetune
CKPT=checkpoints/wpc_rotation_padding_covid19_cheb7_audited_restart
mkdir -p "$OUT"

if [ -e "$CKPT" ]; then
  echo "ERROR: audited restart directory already exists: $CKPT"
  exit 1
fi

mkdir -p "$CKPT"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup .venv/bin/python \
  tools/finetune_wpc_rotation_padding.py \
  --dataset covid19 \
  --data-root data/fhelipe_medseg \
  --image-size 256 \
  --epochs 1 \
  --batch-size 1 \
  --lr 1e-6 \
  --max-epoch-retries 0 \
  --lr-backoff-factor 0.25 \
  --distill-weight 0.001 \
  --train-limit 2048 \
  --val-limit 512 \
  --seed 0 \
  --num-workers 2 \
  --device cuda \
  --out-dir "$CKPT" \
  --result "$OUT/audited_epoch1_restart.json" \
  > "$OUT/audited_epoch1_restart.log" 2>&1 &

echo $! | tee "$OUT/audited_epoch1_restart.pid"
```

Monitor with:

```bash
PID=$(cat .tmp/results/honours/22_wpc_rotation_padding_finetune/audited_epoch1_restart.pid)
ps -p "$PID" -o pid,etime,stat,cmd
tail -n 80 .tmp/results/honours/22_wpc_rotation_padding_finetune/audited_epoch1_restart.log
```

The job is complete when `ps` shows no process and the log ends with the
result path. A successful result has `status: "ok"` and every acceptance gate
set to `true`.

## Server stability observation

On the P100 server, the original `lr=1e-5`, `distill_weight=0.05` run became
non-finite at training batch 1,588 of 2,048. The bounded stability protocol
above completed epoch one in 221.20 seconds with finite validation metrics:
Dice `0.909560`, IoU `0.842765`, loss `0.248910`, and prediction flip rate
`0.026200` versus the native model. Its final schema-v1 report then exposed a
separate reporting defect: a non-finite epoch-zero baseline value could not be
serialized with strict JSON. Schema v2 reports that instability explicitly
and can resume from the already-written finite epoch-one `last` checkpoint.

The recovered schema-v2 report was valid, with epoch one selected as best.
Continuing at `1e-6` then became non-finite at batch 1,588 of epoch two. The
last checkpoint was not overwritten, so epoch one remained recoverable. This
shows that `1e-6` is stable for one epoch but not for the requested five-epoch
schedule. Schema v3 adds deterministic epoch shuffling and whole-epoch
rollback with recorded learning-rate backoff. The resumed server command
starts epoch two at `2.5e-7` and backs off further only if required.

Epoch two subsequently completed at `2.5e-7`, improving validation Dice to
`0.911259`. Epoch three nevertheless failed at the identical batch 1,923 on
all five attempts from `2.5e-7` through `9.765625e-10`. A no-update probe of
the saved epoch-two checkpoint reproduced the failure on selected training
sample 308 (original dataset row 2,379). The first non-finite module was
`dec1a_act`: its finite input magnitude reached `8.661925888e9`, versus a
Chebyshev domain scale of approximately `1.138e3`. Epoch two is therefore not
a valid model despite its finite validation metrics.

Schema v4 closes this acceptance gap by auditing every selected training
sample after each epoch and retaining immutable per-epoch checkpoints. Since
the earlier epoch-one file was overwritten before this defect was known, the
next server task reconstructs epoch one in a fresh directory and audits it;
it does not continue from the invalid epoch-two state.

The one-epoch values are valid intermediate accuracy evidence; the final
five-epoch result remains pending.

## Following stage

After accuracy validation, run repeated isolated full-Q/P versus compressed-
Q/P timing and peak-RSS measurements with the Rotation-Padding-aware best
checkpoint. Those measurements, together with the earlier Orion-layout model
profiles, provide the final empirical WPC-versus-Orion trade-off comparison.
