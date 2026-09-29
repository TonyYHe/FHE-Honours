# Checkpoint-trained WPC CIPS decoder stage

## Purpose

This stage replaces the random miniature-U-Net weights and identity refresh
used in the preceding functional gate with parameters from the trained
COVID-19 medseg checkpoint. It executes the final U-Net22-plus-output decoder
stage through the first following convolution:

```text
low-resolution feature -> checkpoint up1 --+
                                          cat1 -> checkpoint dec1a
checkpoint skip1 --------------------------+       -> checkpoint dec1a_act
                                                  -> bootstrap
                                                  -> checkpoint dec1b
```

The runner loads the exact `up1`, `dec1a`, `dec1a_act`, and `dec1b` tensors.
It validates their names and shapes, records the checkpoint SHA-256, and does
not refit the activation polynomial. Synthetic internal feature tensors keep
this a decoder-stage correctness experiment rather than a dataset or model-
accuracy experiment.

## Learned activation and level schedule

`dec1a_act` is the checkpoint's degree-7 scaled Chebyshev activation. For
input `x`, coefficients `a_k`, learned prescale `s_pre`, and learned postscale
`s_post`, the evaluated function is

```text
z = s_pre x
f(x) = s_post sum_{k=0}^{7} a_k T_k(z),
```

where `T_k` are Chebyshev polynomials. The checkpoint has `blend_alpha=1`, so
the function contains no plaintext-only SiLU branch. The encrypted evaluator
uses the exact stored coefficients and scales.

The CKKS schedule is:

```text
up1 transposed convolution   8 -> 7
cat1 materialization         7 -> 6
dec1a convolution            6 -> 5
prescale + degree-7 Cheb     5 -> 1
bootstrap preprocessing     1 -> 0
bootstrap refresh           0 -> 8
dec1b convolution            8 -> 7
```

The bridge bootstraps four 512-slot CIPS ciphertext groups. Its active mask
tracks exactly 32 channels at `8 x 8`, and the packing signature is restored
on the output without decrypting, repacking, encoding, or re-encrypting.

## Padding semantics

WPC uses flattened Rotation Padding, not the checkpoint model's ordinary
two-dimensional zero padding. For output position `(h,w)` and kernel offset
`(kh,kw)`, WPC reads spatial index

```text
s = (hW + w + (kh-PH)W + (kw-PW)) mod HW.
```

The runner therefore uses an independent Torch implementation of this rule as
the equality oracle. It also evaluates the native zero-padded checkpoint block
and reports the semantic delta rather than hiding it. With the seeded
synthetic features, the maximum deltas are:

| Boundary | WPC vs native zero padding |
|---|---:|
| `dec1a` | 0.077914 |
| `dec1a_act` | 0.035058 |
| `dec1b` | 0.067241 |

These are expected semantic differences, not CKKS errors. Consequently, the
current checkpoint proves trained-parameter integration but not preservation
of its original segmentation accuracy. A WPC model intended for accuracy
claims must be trained or fine-tuned with Rotation Padding.

## Local correctness result

The deterministic local Lattigo run passed every acceptance gate.

| Metric | Result |
|---|---:|
| Exact compressed/full learned transforms | 56/56 pass |
| Independent Torch-WPC vs CIPS clear oracle | max `1.42e-14` |
| Maximum full-Q/P FHE error | `6.14e-7` |
| Maximum compressed-Q/P FHE error | `6.14e-7` |
| Compressed final delta vs full control | `0` |
| Learned-transform payload compression | 13.584x |
| Learned-weight storage ratio including bias/metadata | 13.164x |
| Overall ratio including full-Q/P concat | 12.926x |
| Full learned weights plus bias | 421,855,232 B |
| Stored learned weights, metadata, and bias | 32,046,080 B |
| Ordinary concat Q/P | 589,824 B |
| Offline / online weight Encode calls | 56 / 0 |
| Online Python Encode calls | 0 |
| Peak materialized compressed transforms | 1 |
| Full / compressed rotations | 1,200 / 1,200 |
| Bootstrap calls per path | 4 |

The transposed convolution compresses by 4x because its high-resolution slot
messages have period 128. The two same-shape convolutions compress by 64x.
Their mixture explains the 13.584x aggregate payload ratio.

The local full-control and compressed paths took about 0.69 s and 0.72 s.
They are single diagnostic observations and are not a performance comparison.

## Acceptance gates

The runner exits successfully only when:

- the checkpoint architecture, tensor names, and tensor shapes match;
- exact learned activation coefficients and scales are installed;
- the independent Torch Rotation-Padding oracle matches the CIPS clear oracle;
- all 56 reconstructed compressed Q/P transforms exactly match full controls;
- every encrypted intermediate and final output matches clear computation;
- compressed and full final ciphertext results are identical;
- the degree-7 activation and four real backend bootstraps execute;
- levels and packing signatures chain across the complete stage;
- full and compressed operation counts match;
- online Python and weight `Encode` counts are zero;
- all temporary full Q/P materializations are released; and
- the peak materialization is one transform.

## Files

- `orion/experimental/wpc_cips_checkpoint.py` implements exact checkpoint
  activation extraction/evaluation and the CIPS activation-bootstrap bridge.
- `tools/run_wpc_cips_trained_decoder.py` runs the checkpoint-derived clear,
  full-Q/P, and compressed-Q/P paths and all acceptance checks.
- `tests/test_wpc_cips_checkpoint.py` checks checkpoint parsing, the exact
  Chebyshev recurrence, unsupported blending, CIPS masks, and level budgets.
- `.tmp/results/honours/21_wpc_trained_decoder/trained_decoder_stage.json`
  contains the deterministic local evidence.

## Server reproduction

```bash
cd ~/FHE-Honours
source .venv/bin/activate

git pull --ff-only
python tools/build_lattigo.py

python -m pytest -q \
  tests/test_wpc_cips_checkpoint.py \
  tests/test_wpc_cips_unet.py \
  tests/test_wpc_cips_upsample.py \
  tests/test_wpc_cips_branches.py \
  tests/test_wpc_cips_activation.py \
  tests/test_wpc_cips_layer.py \
  tests/test_wpc_cips_baseline.py \
  tests/test_wpc_periodicity.py

(
  cd orion/backend/lattigo
  go test ./...
)

OUT=.tmp/results/honours/21_wpc_trained_decoder
mkdir -p "$OUT"

python tools/run_wpc_cips_trained_decoder.py \
  --out "$OUT/trained_decoder_server.json" \
  > "$OUT/trained_decoder_server.log" 2>&1

STATUS=$?
echo "Runner exit status: $STATUS"
tail -n 120 "$OUT/trained_decoder_server.log"
```

A valid run exits with status zero, reports `status: "ok"`, and sets every
field under `acceptance` to `true`.

## Remaining work

This gate proves exact trained-parameter and learned-activation integration for
one decoder stage. It does not prove end-to-end trained-model accuracy. The
next stages are Rotation-Padding-aware fine-tuning/accuracy validation and
then repeated isolated server timing/RSS comparisons between the Orion and
WPC paths.
