# Isolated WPC CIPS resource benchmark

## Result

Status: **ok**. The ordinary full-Q/P and WPC-compressed paths
ran in distinct fresh processes with identical parameters, weights, inputs,
warmups, and measured-forward counts.

| Metric | Full Q/P | WPC compressed |
|---|---:|---:|
| Compile time (s) | 0.509053 | 0.489318 |
| Median two-layer forward (ms) | 12.221 | 15.064 |
| p95 two-layer forward (ms) | 13.107 | 16.743 |
| Logical resident plaintext storage (MiB) | 17.44 | 0.35 |
| Pre-online RSS (MiB) | 379.45 | 368.73 |
| Measured-phase peak RSS (MiB) | 395.12 | 369.31 |
| Go heap-in-use after compile GC (MiB) | 24.61 | 7.61 |
| Maximum error versus clear | 3.114e-08 | 3.114e-08 |

The logical plaintext-storage ratio is **49.433x**, and the compressed/full
median-forward ratio is **1.233x**. Median online decompression takes
**1.794 ms**, or **12.038%** of the compressed forward.

The largest difference between independently encrypted full and compressed
outputs is `0.000e+00`.
Both paths perform the same recorded homomorphic operations and no online
weight Encode.

## Measurement boundary

- Forward timing includes two chained `Conv2d` layers.
- Compilation, input encryption, output decryption, and cleanup are excluded.
- An explicit Go garbage collection and unused-page release occurs between
  phases, outside timed forwards, to remove dead compilation allocations.
- RSS is sampled externally by the parent process; Go heap and logical Q/P
  storage are reported separately.
- RSS includes the Python runtime, Torch, keys, ciphertexts, and allocator
  pages. The logical storage ratio therefore must not be presented as an RSS
  ratio.
- This is a deterministic two-layer microbenchmark, not a trained-model
  latency or accuracy result.

All acceptance gates passed: **True**.
