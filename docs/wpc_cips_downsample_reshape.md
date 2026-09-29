# WPC stride-two CIPS and encrypted layout reshaping

## Purpose

This stage implements the downsampling boundary that was missing from the
earlier same-shape CIPS pipelines. A real Orion `Conv2d` with stride two now
uses WPC-compressed weight Q/P plaintexts, emits the sparse layout induced by
stride-two Rotation Padding, and passes that ciphertext directly to a
one-level encrypted reshape. The reshape restores ordinary lower-resolution
CIPS and can merge ciphertext channel groups before the next convolution.

The online boundary performs no decrypt, decode, clear repack, Encode, or
re-encrypt operation.

## Layout construction

Let `n` be the slot count, let the input have spatial shape `H x W`, and let

```text
C_s = n / (H W)
```

be the number of channels that fit in each input-grid ciphertext. Ordinary
CIPS places input channel `c` at

```text
slot_in(c,h,w) = (h W + w) C_s + c.
```

For stride `(S_H,S_W)=(2,2)`, a logical output at `(c,h',w')` is initially
left on the corresponding input-grid anchor:

```text
slot_sparse(c,h',w') = ((h' S_H) W + w' S_W) C_s + c.
```

For kernel coordinate `(k_h,k_w)` and padding `(P_H,P_W)`, WPC's flattened
Rotation Padding selects the spatial source

```text
p = ((h' S_H) W + w' S_W
     + (k_h - P_H) W + (k_w - P_W)) mod (H W).
```

The source slot is `p C_s + c_in`. Therefore the cyclic-diagonal rotation at
the sparse output slot is

```text
r = (slot_source - slot_sparse) mod n.
```

This is compiled once for each input/output ciphertext-group pair. In the
default `8 x 8`, 512-slot case, every stride-two weight diagonal has period
128, so storing one exact period reduces its Q/P payload by 4x. This is smaller
than the 64x ratio in the earlier stride-one `8 x 8` stage: downsampling makes
the active output positions sparse and increases the required period, matching
the trade-off described by WPC.

After downsampling, `H'=H/2` and `W'=W/2`. The lower-resolution channel
capacity is

```text
C_c = n / (H' W').
```

The reshape maps each active sparse slot to ordinary compact CIPS:

```text
slot_compact(c,h',w') = (h' W' + w') C_c + c.
```

It is an offline-encoded permutation linear transform. Its encrypted
evaluation is followed by one rescale, so it consumes exactly one CKKS level.
When `C_c > C_s`, source channel groups that now fit together are accumulated
into one compact output ciphertext. This implementation realizes the
functional one-level reshaping boundary required by WPC; it does not claim to
reproduce an undocumented rotation schedule from the cited reshaping work.

## Implemented pipeline

The deterministic validation pipeline is:

```text
encrypted CIPS input, level 3, 2 groups
  -> Orion Conv2d(12,12,3,stride=2), WPC-compressed weights
  -> sparse 4 x 4 output, level 2, 2 groups
  -> encrypted sparse-to-compact reshape
  -> compact 4 x 4 CIPS, level 1, 1 group
  -> Orion Conv2d(12,12,3,stride=1), WPC-compressed weights
  -> compact 4 x 4 CIPS, level 0, 1 group
```

The installed `Conv2d.forward` path dispatches to the stride-two planner when
the layer stride is `(2,2)`. The sparse and compact tensors carry distinct
packing signatures, and both the reshape and consumer validate the exact
signature and expected level.

## Local correctness result

The seeded local Lattigo run passed every acceptance gate.

| Metric | Result |
|---|---:|
| Stride-two compressed weight transforms | 4 |
| Post-downsample compressed weight transforms | 1 |
| Exact compressed/full Q/P checks | 5/5 pass |
| Stride-two diagonal period / payload ratio | 128 / 4.0x |
| Post-downsample diagonal period / payload ratio | 32 / 16.0x |
| Aggregate full weight Q/P payload | 18,112,512 B |
| Aggregate compressed weight payload | 3,574,272 B |
| Aggregate weight payload ratio | 5.067x |
| Full weights plus bias | 18,169,856 B |
| Stored weights, metadata, and bias | 3,661,136 B |
| Weight-layer storage ratio including bias | 4.963x |
| Ordinary reshape Q/P payload | 262,144 B |
| Sparse / compact ciphertext groups | 2 / 1 |
| Reshape transforms / diagonals | 2 / 8 |
| Full / compressed rotations | 110 / 110 |
| Online Python Encode calls | 0 |
| Online weight Encode calls | 0 |
| Maximum sparse intermediate error | `3.09e-8` |
| Maximum reshaped intermediate error | `3.09e-8` |
| Maximum final error vs. clear | `1.91e-8` |
| Compressed final delta vs. full control | `0` |
| Peak materialized compressed-weight transforms | 1 |
| Materialized weight payload after execution | 0 B |

The ordinary reshape plaintexts are reported separately from compressed
convolution weights. Its permutation payload is encoded offline and remains
resident; it is not counted as WPC-compressed weight storage.

## Acceptance gates

The result is valid only when:

- clear sparse packing and compact packing round-trip exactly;
- a stride-two Orion `Conv2d` installs the stride-two WPC planner;
- all five compressed weight transforms exactly match full Q/P controls;
- encrypted sparse and reshaped intermediates match the clear oracle;
- the compressed final result matches the full control exactly;
- the sparse and compact packing signatures chain without clear repacking;
- the reshape changes two ciphertext groups to one and consumes one level;
- no online Python or weight Encode occurs;
- full and compressed operation counters match;
- each decompressed weight transform is released before the next transform;
- the peak materialized weight payload is exactly one transform;
- storage and Encode accounting close exactly.

## Files

- `orion/experimental/wpc_cips_downsample.py` implements the stride-two case,
  sparse weight diagonals, clear reference, sparse/compact packers, the
  installed convolution plan, and the encrypted reshape plan.
- `orion/experimental/wpc_cips_layer.py` exposes a transform-builder hook so
  stride-one and stride-two plans share the same compression lifecycle.
- `orion/nn/linear.py` dispatches explicit WPC plan installation by stride.
- `tests/test_wpc_cips_downsample.py` checks diagonal correctness,
  periodicity, exact clear reshaping, group compaction, signatures, and
  rejection gates.
- `tools/run_wpc_cips_downsample_pipeline.py` runs the real Lattigo full-vs-
  compressed validation.
- `.tmp/results/honours/17_wpc_downsample_reshape/`
  `stride2_reshape_pipeline.json` is the deterministic local evidence.

## Server reproduction

```bash
cd ~/FHE-Honours
source .venv/bin/activate

python tools/build_lattigo.py

python -m pytest -q \
  tests/test_wpc_cips_downsample.py \
  tests/test_wpc_cips_layer.py \
  tests/test_wpc_cips_baseline.py \
  tests/test_wpc_periodicity.py

(
  cd orion/backend/lattigo
  go test ./...
)

mkdir -p .tmp/results/honours/17_wpc_downsample_reshape
python tools/run_wpc_cips_downsample_pipeline.py \
  --out .tmp/results/honours/17_wpc_downsample_reshape/stride2_reshape_server.json \
  > .tmp/results/honours/17_wpc_downsample_reshape/stride2_reshape_server.log 2>&1
```

A valid runner exits with status zero and reports every field in `acceptance`
as `true`.

## Remaining boundary

This stage proves the stride-two encrypted layout transition in a functional
two-layer pipeline. Residual and channel-concat joins are implemented in
`docs/wpc_cips_branch_joins.md`, and transposed-convolution upsampling is
implemented in `docs/wpc_cips_upsample.md`. A complete trained WPC network
still requires higher-degree trained activations, trained-model
Rotation-Padding accuracy, and server-scale repeated timing/RSS measurements.
