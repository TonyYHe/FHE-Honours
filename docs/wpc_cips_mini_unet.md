# Integrated WPC CIPS miniature U-Net

## Purpose

This stage composes the previously isolated WPC operators into one encrypted
encoder/decoder graph with a real skip connection:

```text
input
  -> encoder Conv2d ------------------------------- skip ---------+
  -> stride-two Conv2d -> encrypted compact reshape               |
  -> bottleneck Conv2d -> identity bootstrap refresh               |
  -> ConvTranspose2d(2x2,stride=2)                                 |
  -> encrypted channel concat <------------------------------------+
  -> decoder Conv2d
```

All five learned layers are actual Orion modules. Their weights are stored as
WPC-compressed Q/P plaintexts and evaluated through the normal module
`forward` calls. The downsample reshape and channel concat are encrypted
linear transforms with ordinary offline-encoded permutation plaintexts. No
online boundary decrypts, clear-repacks, encodes, or re-encrypts data.

This is a deterministic functional graph, not a trained U-Net22 accuracy or
performance experiment.

## Level schedule and skip alignment

The encrypted graph uses this level schedule:

```text
encoder convolution       5 -> 4  (saved skip)
downsample convolution    4 -> 3
compact reshape           3 -> 2
bottleneck convolution    2 -> 1
bootstrap refresh         1 -> 5
transposed convolution    5 -> 4
skip concatenation        4 -> 3
decoder convolution       3 -> 2
```

The main branch consumes three more levels than the saved skip before
upsampling. A packing-aware identity bootstrap refresh restores the
low-resolution bottleneck to level five. The following transposed convolution
then produces level-four ciphertexts, exactly matching the saved skip.

The refresh uses symmetric bounds, so its affine shift is zero. Importantly,
it bootstraps the full physical CIPS message rather than only `C*H*W` logical
values. For the low-resolution case, 12 channels at `4 x 4` contain 192
logical values but occupy interleaved positions across all 512 slots. Using a
256-slot bootstrap loses the upper spatial positions; the implementation
therefore bootstraps all 512 physical slots and applies the exact interleaved
active-channel mask.

## Ciphertext-group transitions

With 512 slots, 12 channels, and spatial sizes `8 x 8 -> 4 x 4 -> 8 x 8`:

| Boundary | Ciphertext groups |
|---|---:|
| Encrypted input | 2 |
| Sparse stride-two output | 2 |
| Compact low-resolution representation | 1 |
| Refreshed bottleneck | 1 |
| Transposed-convolution output | 2 |
| Skip-concat inputs | 4 total |
| Concatenated representation | 3 |
| Decoder output | 1 |

Each transition is checked against an explicit CIPS packing signature and
expected CKKS level before evaluation.

## Local correctness and storage result

The seeded local Lattigo execution passed every acceptance gate.

| Metric | Result |
|---|---:|
| Exact compressed/full learned transforms | 14/14 pass |
| Weight-payload compression | 7.771x |
| Weight storage including metadata | 7.702x |
| Full weights plus bias | 68,059,136 B |
| Stored weights, metadata, and bias | 9,071,920 B |
| Ordinary reshape plus concat Q/P | 573,440 B |
| Overall logical storage compression | 7.116x |
| Full / compressed rotations | 318 / 318 |
| Full / compressed conjugations | 1 / 1 |
| Bootstrap calls per path | 1 |
| Offline / online weight Encode calls | 14 / 0 |
| Online Python Encode calls | 0 |
| Maximum intermediate error | `8.37e-9` |
| Final error versus clear | `3.25e-9` |
| Compressed final delta versus full | `0` |
| Peak materialized compressed transforms | 1 |

The aggregate ratio is lower than the 64x same-shape convolution result
because stride-two downsampling and transposed-convolution upsampling each
have period 128 and compress their payloads by 4x. The `4 x 4` bottleneck has
16x compression, while the same-shape `8 x 8` encoder and decoder transforms
have 64x compression.

The local full-control and compressed executions took approximately 131 ms
and 129 ms respectively, including one bootstrap. These are single diagnostic
observations and must not be interpreted as a performance comparison.

## Acceptance gates

The runner succeeds only when:

- actual Orion encoder, downsample, bottleneck, transposed-convolution, concat,
  and decoder modules execute;
- all 14 compressed weight transforms exactly match full Q/P controls;
- every encrypted intermediate and the final output match independent clear
  references;
- the complete packing-signature and level chain is valid;
- all ciphertext-group transitions have the expected counts;
- the identity bootstrap preserves CIPS and executes once per path;
- full and compressed operation counters match;
- online Python and weight Encode counts are zero;
- all layout boundaries report zero clear repack or Encode;
- every temporary full Q/P materialization is released before the next; and
- the materialization peak is one transform.

## Files

- `orion/experimental/wpc_cips_unet.py` implements the packing-aware identity
  bootstrap refresh used for skip-level alignment.
- `tools/run_wpc_cips_mini_unet.py` compiles and executes the complete graph,
  full-Q/P control, compressed path, correctness checks, and accounting.
- `tests/test_wpc_cips_unet.py` checks refresh contracts, the physical active
  mask, and rejection conditions.
- `.tmp/results/honours/20_wpc_mini_unet/mini_unet_pipeline.json` contains the
  deterministic local evidence.

## Server reproduction

```bash
cd ~/FHE-Honours
source .venv/bin/activate

git pull --ff-only
python tools/build_lattigo.py

python -m pytest -q \
  tests/test_wpc_cips_unet.py \
  tests/test_wpc_cips_upsample.py \
  tests/test_wpc_cips_downsample.py \
  tests/test_wpc_cips_branches.py \
  tests/test_wpc_cips_activation.py \
  tests/test_wpc_cips_layer.py \
  tests/test_wpc_cips_baseline.py \
  tests/test_wpc_periodicity.py

(
  cd orion/backend/lattigo
  go test ./...
)

OUT=.tmp/results/honours/20_wpc_mini_unet
mkdir -p "$OUT"

python tools/run_wpc_cips_mini_unet.py \
  --out "$OUT/mini_unet_server.json" \
  > "$OUT/mini_unet_server.log" 2>&1
```

A valid runner exits with status zero, reports `status: "ok"`, and sets every
field under `acceptance` to `true`.

## Remaining boundary

This stage demonstrates that the complete encrypted operator chain composes
correctly. It does not establish trained-model accuracy or end-to-end model
performance. The next implementation stage maps trained U-Net parameters and
activation/bootstrap placement onto this runtime, followed by repeated
server-scale O-online versus CIPS-WPC timing and RSS measurements.
