# WPC CIPS activation and bootstrap integration

## Purpose

This stage extends the compressed CIPS convolution path across a nonlinear
refresh boundary. The implemented encrypted pipeline is

```text
compressed CIPS Conv2d -> quadratic activation -> bootstrap
                       -> compressed CIPS Conv2d
```

It uses actual `orion.nn.Conv2d`, `orion.nn.Quad`, and
`orion.nn.Bootstrap` modules. Each CIPS channel group remains a separate
ciphertext throughout the pipeline; there is no decryption, repacking, or
online weight Encode between the two convolutions.

## Level contract

The test parameter chain has maximum level 3. The pipeline consumes and
refreshes levels as follows:

| Operation | Input level | Output level |
|---|---:|---:|
| First CIPS convolution and rescale | 3 | 2 |
| `Quad`: ciphertext square and rescale | 2 | 1 |
| Bootstrap prescale plaintext multiply and rescale | 1 | 0 |
| Lattigo bootstrap | 0 | 3 |
| Second CIPS convolution and rescale | 3 | 2 |

The bridge checks every ciphertext ID at all three externally visible
boundaries. A level mismatch fails before evaluation.

## CIPS-aware bootstrap mask

For spatial position `p`, CIPS stores channel `c` at

```text
slot(p,c) = p Cmax + c,
```

where `Cmax = slots/(H W)` is the per-ciphertext channel capacity. A final
partial channel group with `Cg < Cmax` therefore occupies the first `Cg`
slots of *each spatial block*, not one contiguous prefix of the ciphertext.
Its active-slot mask is

```text
mg[p Cmax + c] = 1  if c < Cg,
                  0  otherwise.
```

Orion's bootstrap prescale is a plaintext multiplication. The bridge compiles
the exact `mg` for every group and applies it before bootstrapping. Symmetric
bounds `[-B,B]` make the bootstrap affine shift zero, so inactive slots remain
zero after postprocessing.

This detail is correctness-critical. An initial prefix mask treated the final
four-channel group as 256 consecutive active slots. CIPS actually interleaves
those four channels with four spare slots at every one of the 64 spatial
positions. That incorrect mask produced a `5.60e-2` error immediately after
the activation/bootstrap boundary. The interleaved mask reduces the same
error to `2.54e-9` in the seeded local functional run.

## Implementation

- `orion/experimental/wpc_cips_activation.py` implements
  `WPCCIPSActivationBootstrap`.
- `tests/test_wpc_cips_activation.py` checks plan compatibility, grouped
  execution, the interleaved active-slot mask, level validation, packing
  preservation, and zero online Encode calls.
- `tools/run_wpc_cips_activation_bootstrap.py` runs the real Lattigo pipeline
  and writes all correctness, bootstrap-profile, storage, and lifecycle gates
  to JSON.

The bridge can be constructed with `between_plans(producer, consumer, ...)`.
It requires the producer output signature to equal the consumer input
signature. After `Quad` and after `Bootstrap`, it restores and validates that
same signature on the new `CipherTensor`.

## Local validation result

The seeded local run uses `LogN=10`, 512 slots, 12 channels, an `8 x 8`
spatial grid, and two ciphertext groups. All acceptance gates pass.

| Metric | Result |
|---|---:|
| Exact compressed/full Q/P transforms | 8/8 |
| Backend bootstrap calls per pipeline | 2 |
| Backend bootstrap level transition | 0 -> 3 |
| First-convolution maximum error | `1.11e-8` |
| Activation/bootstrap maximum error | `2.54e-9` |
| Final maximum error versus clear | `1.06e-9` |
| Compressed/full final-output delta | `0` |
| Online Python Encode calls | `0` |
| Online weight Encode calls | `0` |
| Full weight Q/P payload | 26,050,560 B |
| Compressed weight payload | 407,040 B |
| Payload compression | 64.0x |
| Stored weights, metadata, and biases | 541,152 B |
| Full/stored ratio including bias | 48.321x |
| Peak materialized transforms | 1 |
| Materialized payload after execution | 0 B |

The result is stored at
`.tmp/results/honours/16_wpc_activation_bootstrap/`
`two_conv_quad_bootstrap.json`.

The reported wall times are one diagnostic execution and must not be used as
a performance comparison.

## Server reproduction

```bash
cd ~/FHE-Honours
source .venv/bin/activate

git pull --ff-only
python tools/build_lattigo.py

python -m pytest -q \
  tests/test_wpc_cips_activation.py \
  tests/test_wpc_cips_layer.py \
  tests/test_wpc_cips_baseline.py \
  tests/test_wpc_periodicity.py

(
  cd orion/backend/lattigo
  go test ./...
)

OUT=.tmp/results/honours/16_wpc_activation_bootstrap
mkdir -p "$OUT"

python tools/run_wpc_cips_activation_bootstrap.py \
  --out "$OUT/two_conv_quad_bootstrap_server.json" \
  > "$OUT/two_conv_quad_bootstrap_server.log" 2>&1

TASK_STATUS=$?
echo "Runner exit status: $TASK_STATUS"
tail -n 120 "$OUT/two_conv_quad_bootstrap_server.log"
```

A valid run exits with status zero, reports `status: "ok"`, and sets every
field under `acceptance` to `true`.

## Remaining boundary

This proves one multi-ciphertext activation/bootstrap boundary with a
quadratic activation. It does not yet implement stride-two CIPS reshaping,
residual or concatenation paths, a higher-degree trained-model activation, or
trained-model accuracy evaluation. It also does not make a performance claim.
