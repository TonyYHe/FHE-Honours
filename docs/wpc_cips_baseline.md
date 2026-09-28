# WPC CIPS functional baseline

## Purpose

This baseline establishes that WPC-style periodic plaintexts do not arise from
compression alone: they arise after changing the tensor layout to
channel-in-the-innermost-position (CIPS) and applying the paper's Rotation
Padding semantics. It compares that layout with a channel-first control using
the same input, weights, ring size, convolution, and cyclic-diagonal evaluator.

The implementation covers a general two-dimensional, stride-one convolution
using the flattened rotation rule in WPC Algorithm 2 and the layout illustrated
by Figures 8–10. It is a correctness baseline, not a full WPC implementation
or a performance benchmark.

## Implemented construction

For a slot vector of length `n`, padded spatial dimensions `H` and `W`, and
channel capacity `C = n/(HW)`, CIPS assigns tensor element `(c,h,w)` to

```text
j = c + C(w + Wh).
```

The matched control assigns it to

```text
j = w + W(h + Hc).
```

Both layouts lower the same linear operation to cyclic diagonals. If output
slot `j` reads source slot `k`, its contribution is put in diagonal
`r = (k-j) mod n`, so the clear and encrypted evaluators compute

```text
y = sum_r d_r * Rot(x, r).
```

For output position `(h,w)` and kernel position `(kh,kw)`, the source spatial
position is

```text
s = (hW + w + (kh-PHB)W + (kw-PWB)) mod HW.
```

The source coordinates are `(floor(s/W), s mod W)`. This is the spatial part
of Algorithm 2's rotation

```text
r = khWC + kwC + channel_delta - (PHBWC + PWBC).
```

The wrap therefore applies to the single flattened spatial sequence. A width
offset from the last column advances into the next packed row; it is not
ordinary per-axis toroidal padding. Dedicated tests exercise this row-boundary
case. A separate zero-padding reference makes the semantic change explicit.

For every nonzero diagonal, the baseline finds the exact minimal power-of-two
slot period `T`. A WPC candidate must have `T < n`. It then checks two distinct
round trips:

1. reconstruct the slot message by repeating its first `T` slots;
2. encode the message with real Lattigo, retain one period of the Q/P
   evaluation representation, reconstruct all coefficients with the Lattigo
   copy map, and require exact equality in every active limb.

Finally, both layouts are compiled as real Lattigo linear transforms,
encrypted, evaluated, decrypted, and compared with the direct Rotation-Padded
convolution.

## Acceptance gates

An output is valid only if all of the following hold:

- channel-first and CIPS clear evaluations match the same direct reference;
- the two layout outputs match each other;
- every nonzero CIPS diagonal has a proper exact slot period;
- slot-message reconstruction is exact;
- exact encoded Q/P reconstruction passes for every CIPS candidate;
- both layouts match the reference after real CKKS evaluation within `1e-6`;
- a width offset exercises the flattened row-boundary wrap and differs from
  independent per-axis wrapping;
- Rotation Padding is observably different from zero padding for the test case.

`--skip-encoded-qp` and `--skip-fhe-eval` are diagnostic options. Either makes
the run fail the acceptance gate, so a clear-only result cannot be reported as
a completed baseline.

## Default test case and local result

The default test uses 512 CKKS slots, an `8 x 8` spatial grid, four input and
four output channels, and a `3 x 3` kernel. The deterministic seed is
`20260928`.

| Metric | Channel-first control | CIPS |
|---|---:|---:|
| Nonzero diagonals | 71 | 63 |
| Proper-period diagonals | 0 | 63 |
| Slot-period coverage | 0% | 100% |
| Minimal period | 512 | 8 |
| Analytical encoded storage | 2,326,528 B | 32,256 B hybrid vs. 2,064,384 B full |
| Analytical partial-storage ratio | 1.0x | 64.0x |
| Exact encoded Q/P passes | not applicable | 63/63 |
| Real-FHE maximum absolute error | `2.04e-7` | `2.04e-7` |
| Real-FHE correctness at `1e-6` | pass | pass |

The storage values are analytical Q/P payload accounting. They are not peak
resident memory measurements. The recorded timings are single diagnostic
executions without warmup and must not be used for performance comparisons.

## Reproduce on the server

From the repository root, after pulling the implementing commit:

```bash
source .venv/bin/activate
python tools/build_lattigo.py
python -m pytest -q tests/test_wpc_cips_baseline.py tests/test_wpc_periodicity.py
mkdir -p .tmp/results/honours/11_wpc_cips_baseline
python tools/run_wpc_cips_baseline.py \
  --out .tmp/results/honours/11_wpc_cips_baseline/cips_baseline.json \
  2>&1 | tee .tmp/results/honours/11_wpc_cips_baseline/cips_baseline.log
```

A successful run exits with status zero and reports:

```text
"status": "ok"
"acceptance": { ... "valid": true }
```

The result JSON is the machine-readable record. Preserve it together with the
log and the exact Git commit used for the run.

## Scope and next implementation work

This baseline deliberately does not yet implement:

- multiple ciphertext channel groups;
- WPC's downsampling reshaping layer;
- training or fine-tuning for Rotation Padding;
- the compressed online Encode/decompression path;
- matched model-level runtime and memory measurements.

Therefore, it proves that the implemented CIPS/Rotation-Padding subset creates
valid periodic Lattigo plaintexts for general two-dimensional kernels and
preserves the intended flattened-rotation computation. It does not reproduce
the WPC paper's model-level accuracy, latency, or memory results.
