# Three-way CIPS online-Encode benchmark

Status: ok. Independent process blocks: 6.

Same checkpoint, exact feature values, CIPS layout, CKKS configuration, and homomorphic operations in all modes.

**Functional-test parameters only:** LogN=10; not a secure-deployment or complete-network benchmark.

| Mode | Forward (s) | Encode share | Preparation + Encode share | Decompression share | Logical resident (MiB) | Sampled online peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| online_encode | 3.286573 | 26.490% | 31.258% | 0.000% | 2.097 | 1024.478 |
| full | 2.280914 | 0.000% | 0.000% | 0.000% | 402.875 | 1838.489 |
| compressed | 2.426713 | 0.000% | 0.000% | 6.406% | 31.124 | 1093.327 |

Entries are means of per-process-block statistics; shares are within-forward ratios, not ratios of aggregate medians.

| Paired latency ratio | Mean | 95% CI for mean |
|---|---:|---:|
| compressed_over_online_encode_forward_ratio | 0.738381 | [0.736169, 0.740582] |
| compressed_over_full_forward_ratio | 1.063997 | [1.057179, 1.071176] |
| full_over_online_encode_forward_ratio | 0.694030 | [0.687667, 0.699960] |

percentile bootstrap of matched process blocks; mean of block statistics; not pooled-forward confidence intervals.

Lifecycle tracing and output validation run outside timed forwards. Encode counters count actual successful linear-transform Encode invocations, not individual diagonals.

Online mode stores compact float32 slot-period recipes; it expands them and allocates Q/P online. Bias and concat plaintexts are preencoded in all modes.

Preparation and Encode are separate: neither has the same boundary as the historical Orion Step-1 layer-cache category. Do not subtract these numbers from the whole-model profiles.

RSS is an externally sampled maximum, not a guaranteed transient allocation peak. Warmups are excluded.

Fresh-process order is balanced across all six treatment permutations.

This isolates storage/materialization within CIPS. It does not establish the Orion-versus-WPC layout crossover.
