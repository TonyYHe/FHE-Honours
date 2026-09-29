# WPC CIPS residual and concatenation branch joins

## Purpose

This stage adds encrypted branch joins to the opt-in WPC CIPS pipeline. It
supports:

- residual addition between ciphertext tensors with identical CIPS packing;
- channel concatenation between compatible CIPS branches;
- direct handoff from the concatenated ciphertext to a WPC-compressed Orion
  `Conv2d` consumer.

Both paths use actual `orion.nn.Add` and `orion.nn.Concat` modules. The online
execution performs no decrypt, decode, clear repack, Encode, or re-encrypt.
The ordinary Orion planners remain unchanged unless an explicit WPC plan is
installed.

## Residual addition

Let a canonical CIPS packing signature contain the slot count, channel count,
spatial shape, and channel-group ranges. Two residual inputs may be added only
when all of the following are identical:

- scheme instance;
- packing signature;
- logical shape;
- ciphertext-group count;
- CKKS level.

For ciphertext group `g`, the join computes

```text
y_g = x_g + z_g.
```

No plaintext multiplication or rescale is required. Therefore a residual join
with `G` channel groups performs `G` ciphertext additions and consumes zero
levels. Its output retains the input CIPS signature.

## Channel concatenation

Suppose all branches share `n` slots and spatial shape `H x W`. The canonical
CIPS channel capacity is

```text
C = n / (H W).
```

Let branch `b` have channel offset `O_b` in the concatenated logical tensor.
For a channel `c` in source group `[a,a+C)`, its global output channel is

```text
g = O_b + c.
```

At flattened spatial position `p`, its source and destination slots are

```text
source_slot = p C + (c - a),
output_group = floor(g / C),
output_slot = p C + (g mod C).
```

The corresponding cyclic-diagonal rotation is

```text
r = (source_slot - output_slot) mod n.
```

One permutation transform is generated for every nonempty intersection of a
branch source group and output group. Partial transforms targeting the same
output group are added before a single rescale. Consequently, materialized
concat consumes exactly one CKKS level even when a branch crosses an output
group boundary.

The permutation plaintexts are encoded offline as ordinary full-Q/P linear
transforms. They are reported separately and are not described as compressed
weight storage.

## Validated pipeline

The deterministic pipeline uses 512 slots, an `8 x 8` spatial shape, and CIPS
capacity eight:

```text
                         Conv 12->12 --+
encrypted 12ch input -->                 +--> residual 12ch, level 2 --+
                         Conv 12->12 --+                              |
                                                                       +--> concat 16ch
                         Conv 12->4 ----------------------------------+    level 1
                                                                            |
                                                                            v
                                                                      Conv 16->8
                                                                        level 0
```

All three branches begin at level 3 and their compressed convolutions produce
level-2 outputs. The residual retains level 2. Concatenation consumes one
level and produces the exact packing expected by the level-1 consumer.

The residual occupies two ciphertext groups. The four-channel side branch
occupies one. Concatenation maps these three source ciphertexts to two output
ciphertexts using three one-diagonal permutation transforms; the second
output group accumulates one residual partial and one side-branch partial.

## Local correctness result

The seeded local Lattigo run passed every acceptance gate.

| Metric | Result |
|---|---:|
| Compressed weight transforms | 12 |
| Exact compressed/full Q/P checks | 12/12 pass |
| Weight diagonal payload compression | 64.0x |
| Full weight Q/P payload | 36,519,936 B |
| Compressed weight payload | 570,624 B |
| Weight-transform metadata | 54,384 B |
| Full weights plus bias | 36,651,008 B |
| Stored weights, metadata, and bias | 756,080 B |
| Weight-layer ratio including bias | 48.475x |
| Ordinary concat Q/P payload | 98,304 B |
| Stored weights plus concat | 854,384 B |
| Full weight+bias / stored weight+concat | 42.898x |
| Residual ciphertext additions / levels consumed | 2 / 0 |
| Concat input / output ciphertext groups | 3 / 2 |
| Concat transforms / accumulation additions | 3 / 1 |
| Full / compressed rotations | 214 / 214 |
| Online Python Encode calls | 0 |
| Online weight Encode calls | 0 |
| Maximum residual error | `3.99e-8` |
| Maximum concat error | `3.99e-8` |
| Maximum final error vs. clear | `2.53e-8` |
| Compressed final delta vs. full control | `0` |
| Peak materialized compressed-weight transforms | 1 |
| Materialized weight payload after execution | 0 B |

## Acceptance gates

The result is valid only when:

- actual Orion `Add` and `Concat` modules execute their installed WPC plans;
- all 12 compressed weight transforms exactly match full-Q/P controls;
- encrypted residual, concatenated, and final outputs match clear references;
- the compressed final output exactly matches the full control;
- residual inputs have identical signatures and levels;
- concat output has exactly the consumer's input signature;
- residual addition preserves level 2 and concat maps level 2 to level 1;
- branch-join transform, addition, and ciphertext-group counts close exactly;
- no online Python or weight Encode occurs;
- full and compressed operation counters match;
- each materialized compressed weight transform is released before the next;
- aggregate storage and Encode accounting close exactly.

## Implementation

- `orion/experimental/wpc_cips_branches.py` implements packing validation,
  groupwise residual addition, concat diagonal construction, encrypted concat
  evaluation, decoding helpers, storage accounting, and cleanup.
- `orion/nn/operations.py` adds explicit WPC plan installation/removal and HE
  forward dispatch to `Add` and `Concat`.
- `tests/test_wpc_cips_branches.py` tests rejection gates, cross-group concat
  diagonal correctness, and actual encrypted Orion module execution.
- `tools/run_wpc_cips_branch_join_pipeline.py` runs the full-vs-compressed
  Lattigo branch pipeline.
- `.tmp/results/honours/18_wpc_branch_joins/`
  `residual_concat_pipeline.json` is the deterministic local evidence.

## Server reproduction

```bash
cd ~/FHE-Honours
source .venv/bin/activate

python tools/build_lattigo.py

python -m pytest -q \
  tests/test_wpc_cips_branches.py \
  tests/test_wpc_cips_downsample.py \
  tests/test_wpc_cips_layer.py \
  tests/test_wpc_cips_baseline.py \
  tests/test_wpc_periodicity.py

(
  cd orion/backend/lattigo
  go test ./...
)

OUT=.tmp/results/honours/18_wpc_branch_joins
mkdir -p "$OUT"

python tools/run_wpc_cips_branch_join_pipeline.py \
  --out "$OUT/residual_concat_server.json" \
  > "$OUT/residual_concat_server.log" 2>&1
```

A valid run exits with status zero, reports `status: "ok"`, and sets every
field under `acceptance` to `true`.

## Remaining boundary

This stage proves functional residual and materialized-concat handling for
same-spatial-shape CIPS branches. It is not a complete trained WPC network.
Lazy concat-to-convolution fusion, higher-degree trained activations,
trained-model Rotation-Padding accuracy, and server-scale repeated timing/RSS
remain future work. The transposed-convolution upsampling boundary is
implemented separately in `docs/wpc_cips_upsample.md`.
