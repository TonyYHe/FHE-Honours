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

Epoch zero is saved as the initial best checkpoint. Fine-tuning can therefore
never replace it with a lower-Dice model. `rotation_padding_last.pt` is written
atomically after every epoch, and `--resume-if-present` continues from it.
The result JSON is also updated after each completed epoch.

## Acceptance gates

The final result is valid only when:

- the source checkpoint hash is recorded;
- exactly 18 spatial convolutions use WPC Rotation Padding;
- conversion preserves all checkpoint keys and values;
- all 18 activations are pure degree-7 polynomial paths;
- native, pre-fine-tuning, and best validation metrics are finite;
- a nonzero Rotation-Padding semantic change is observed;
- the selected best model is not worse than the un-fine-tuned converted model;
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
- evaluation-only reload of the saved best checkpoint.

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

The full run is intended for a CUDA server and may take hours. First verify
that the dataset exists:

```bash
ls -lh data/fhelipe_medseg/covid19radio_512.npz
```

Then run the job under `nohup`:

```bash
OUT=.tmp/results/honours/22_wpc_rotation_padding_finetune
CKPT=checkpoints/wpc_rotation_padding_covid19_cheb7
mkdir -p "$OUT" "$CKPT"

nohup python tools/finetune_wpc_rotation_padding.py \
  --dataset covid19 \
  --data-root data/fhelipe_medseg \
  --image-size 256 \
  --epochs 5 \
  --batch-size 1 \
  --lr 1e-5 \
  --distill-weight 0.05 \
  --num-workers 2 \
  --device cuda \
  --resume-if-present \
  --out-dir "$CKPT" \
  --result "$OUT/rotation_padding_finetune.json" \
  > "$OUT/rotation_padding_finetune.log" 2>&1 &

echo $! | tee "$OUT/rotation_padding_finetune.pid"
```

Monitor with:

```bash
PID=$(cat .tmp/results/honours/22_wpc_rotation_padding_finetune/rotation_padding_finetune.pid)
ps -p "$PID" -o pid,etime,stat,cmd
tail -n 80 .tmp/results/honours/22_wpc_rotation_padding_finetune/rotation_padding_finetune.log
```

The job is complete when `ps` shows no process and the log ends with the
result path. A successful result has `status: "ok"` and every acceptance gate
set to `true`.

## Following stage

After accuracy validation, run repeated isolated full-Q/P versus compressed-
Q/P timing and peak-RSS measurements with the Rotation-Padding-aware best
checkpoint. Those measurements, together with the earlier Orion-layout model
profiles, provide the final empirical WPC-versus-Orion trade-off comparison.
