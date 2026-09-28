# WPC CIPS functional baseline

## Purpose

This baseline establishes that WPC-style periodic plaintexts do not arise from
compression alone: they arise after changing the tensor layout to
channel-in-the-innermost-position (CIPS) and applying the paper's Rotation
Padding semantics. It compares that layout with a channel-first control using
the same input, weights, ring size, convolution, and cyclic-diagonal evaluator.

The implementation covers the vertical-convolution case illustrated by WPC
Algorithms 1–2 and Figures 8–10. It is a correctness baseline, not a full WPC
implementation or a performance benchmark.

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

The implemented kernel has width one, stride one, and an arbitrary vertical
kernel height. Rotation Padding supplies an out-of-range height from the
adjacent same-channel row by circular extension. A separate zero-padding
reference is evaluated to make this semantic change explicit.

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
- Rotation Padding is observably different from zero padding for the test case.

`--skip-encoded-qp` and `--skip-fhe-eval` are diagnostic options. Either makes
the run fail the acceptance gate, so a clear-only result cannot be reported as
a completed baseline.

## Default test case and local result

The default test uses 512 CKKS slots, an `8 x 8` spatial grid, four input and
four output channels, and a `3 x 1` vertical kernel. The deterministic seed is
`20260928`.

| Metric | Channel-first control | CIPS |
|---|---:|---:|
| Nonzero diagonals | 23 | 21 |
| Proper-period diagonals | 0 | 21 |
| Slot-period coverage | 0% | 100% |
| Minimal period | 512 | 8 |
| Analytical encoded storage | 753,664 B | 10,752 B hybrid vs. 688,128 B full |
| Analytical partial-storage ratio | 1.0x | 64.0x |
| Exact encoded Q/P passes | not applicable | 21/21 |
| Real-FHE maximum absolute error | `8.95e-8` | `8.95e-8` |
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

- general `K_H x K_W` convolution with width-boundary handling;
- multiple ciphertext channel groups;
- WPC's downsampling reshaping layer;
- training or fine-tuning for Rotation Padding;
- the compressed online Encode/decompression path;
- matched model-level runtime and memory measurements.

Therefore, it proves that the implemented CIPS/Rotation-Padding subset creates
valid periodic Lattigo plaintexts and preserves the intended computation. It
does not reproduce the WPC paper's model-level accuracy, latency, or memory
results.
