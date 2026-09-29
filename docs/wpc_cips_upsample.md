# WPC CIPS transposed-convolution upsampling

## Purpose

This stage implements the missing U-Net decoder upsampling boundary. An
actual Orion `ConvTranspose2d(kernel_size=2, stride=2)` now maps a compact
low-resolution CIPS input directly to high-resolution CIPS ciphertext groups
using WPC-compressed Q/P weight plaintexts. A following Orion `Conv2d` consumes
that output without decrypting, clear repacking, encoding, or re-encrypting.

The ordinary Orion `ConvTranspose2d` path remains unchanged. The CIPS path is
installed only through the explicit `install_wpc_cips_plan` method.

## Slot mapping

For `n` CKKS slots and low-resolution spatial shape `H x W`, the input channel
capacity is

```text
C_in = n / (H W).
```

The low-resolution input slot is

```text
s_in(c,h,w) = (h W + w) C_in + c.
```

The supported transposed convolution has kernel and stride `(2,2)`, so its
output has shape `2H x 2W` and channel capacity

```text
C_out = n / ((2H)(2W)) = C_in / 4.
```

For kernel coordinate `(k_h,k_w)`, input `(h,w)` contributes to

```text
h' = 2h + k_h,
w' = 2w + k_w,
s_out(c',h',w') = (h' (2W) + w') C_out + c'.
```

The cyclic-diagonal rotation is

```text
r = (s_in - s_out) mod n.
```

The diagonal stores the PyTorch transposed-convolution weight
`weight[c_in,c_out,k_h,k_w]` at `s_out`. One transform is compiled for every
input/output ciphertext-group pair. This direct mapping expands the number of
groups when the high-resolution layout has less channel capacity.

In the deterministic 512-slot case, `4 x 4` input CIPS holds 32 channels per
ciphertext while `8 x 8` output CIPS holds eight. Twelve channels therefore
expand from one ciphertext group to two. Every upsampling weight diagonal has
period 128, giving exact 4x Q/P payload compression. The following `8 x 8`
stride-one convolution has period eight and 64x payload compression.

## Validated pipeline

```text
encrypted CIPS input, 12 x 4 x 4, level 2, one group
  -> Orion ConvTranspose2d(12,12,2,stride=2), compressed weights
  -> CIPS 12 x 8 x 8, level 1, two groups
  -> Orion Conv2d(12,8,3,padding=1), compressed weights
  -> CIPS 8 x 8 x 8, level 0, one group
```

The seeded local Lattigo run passed every acceptance gate.

| Metric | Result |
|---|---:|
| Exact compressed/full Q/P transforms | 4/4 pass |
| Upsampling period / payload compression | 128 / 4.0x |
| Consumer period / payload compression | 8 / 64.0x |
| Aggregate weight payload compression | 5.734x |
| Full weights plus bias | 13,295,616 B |
| Stored weights, metadata, and bias | 2,377,568 B |
| Storage ratio including bias | 5.592x |
| Input / output upsampling groups | 1 / 2 |
| Full / compressed rotations | 84 / 84 |
| Online Python / weight Encode calls | 0 / 0 |
| Maximum upsampling error | `1.07e-8` |
| Maximum final error | `9.48e-9` |
| Compressed final delta versus full | `0` |
| Peak materialized weight transforms | 1 |

These are correctness and logical-storage results. The timings are a single
diagnostic observation and are not a performance claim.

## Acceptance gates

The runner is valid only when:

- the clear high-resolution CIPS layout round-trips exactly;
- an actual Orion `ConvTranspose2d` installs the new WPC plan;
- all compressed Q/P transforms exactly match their full controls;
- the encrypted upsample and final output match independent clear references;
- the compressed and full outputs are identical;
- the packing signatures chain directly into the consumer convolution;
- one low-resolution ciphertext group expands to two high-resolution groups;
- the two weight layers consume exactly two CKKS levels;
- online Python and weight Encode counts are zero;
- full and compressed operation counters match;
- decompressed transform payloads are released between evaluations; and
- the materialization peak is exactly one transform.

## Files

- `orion/experimental/wpc_cips_upsample.py` implements the geometry, slot
  mapping, diagonals, independent clear oracle, packing, and installed plan.
- `orion/nn/linear.py` adds explicit WPC plan installation and FHE dispatch to
  `ConvTranspose2d`.
- `tests/test_wpc_cips_upsample.py` checks the diagonal output against both the
  independent oracle and PyTorch, periodicity, packing, signature chaining,
  and rejection gates.
- `tools/run_wpc_cips_upsample_pipeline.py` executes the full-versus-compressed
  Lattigo validation.
- `.tmp/results/honours/19_wpc_upsample/conv_transpose2d_pipeline.json` is the
  deterministic local evidence.

## Server reproduction

```bash
cd ~/FHE-Honours
source .venv/bin/activate

git pull --ff-only
python tools/build_lattigo.py

python -m pytest -q \
  tests/test_wpc_cips_upsample.py \
  tests/test_wpc_cips_downsample.py \
  tests/test_wpc_cips_branches.py \
  tests/test_wpc_cips_layer.py \
  tests/test_wpc_cips_baseline.py \
  tests/test_wpc_periodicity.py

(
  cd orion/backend/lattigo
  go test ./...
)

mkdir -p .tmp/results/honours/19_wpc_upsample
python tools/run_wpc_cips_upsample_pipeline.py \
  --out .tmp/results/honours/19_wpc_upsample/conv_transpose2d_server.json \
  > .tmp/results/honours/19_wpc_upsample/conv_transpose2d_server.log 2>&1
```

A valid run exits with status zero and reports every field in `acceptance` as
`true`.

## Remaining boundary

This stage completes the isolated operator boundaries needed by a small WPC
U-Net decoder: downsampling, upsampling, residual joins, concatenation,
activation, and bootstrap are all covered. Their integrated encoder/decoder
validation is implemented in `docs/wpc_cips_mini_unet.md`; trained-model
parameter matching and server-scale repeated timing/RSS remain outstanding.
