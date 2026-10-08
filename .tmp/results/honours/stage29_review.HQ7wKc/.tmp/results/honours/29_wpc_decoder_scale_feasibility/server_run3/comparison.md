# Three-way CIPS online-Encode benchmark

Status: ok. Independent process blocks: 1.

Same checkpoint, exact feature values, CIPS layout, CKKS configuration, and homomorphic operations in all modes.

**Security not assessed:** LogN=12, output shape [1, 32, 16, 16]; neither a secure-deployment nor a complete-network benchmark.

| Mode | Forward (s) | Encode share | Preparation + Encode share | Decompression share | Logical resident (MiB) | Sampled online peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| online_encode | 15.442853 | 29.900% | 34.758% | 0.000% | 7.484 | 2161.672 |
| full | 10.466436 | 0.000% | 0.000% | 0.000% | 1931.500 | 3982.852 |
| compressed | 11.129325 | 0.000% | 0.000% | 8.120% | 99.772 | 2284.316 |

Entries are means of per-process-block statistics; shares are within-forward ratios, not ratios of aggregate medians.

| Paired latency ratio | Mean | 95% CI for mean |
|---|---:|---:|
| compressed_over_online_encode_forward_ratio | 0.720678 | not estimated (smoke) |
| compressed_over_full_forward_ratio | 1.063335 | not estimated (smoke) |
| full_over_online_encode_forward_ratio | 0.677753 | not estimated (smoke) |

correctness-only smoke block; no uncertainty estimation or performance claim.

Lifecycle tracing and output validation run outside timed forwards. Encode counters count actual successful linear-transform Encode invocations, not individual diagonals.

Online mode stores compact float32 slot-period recipes; it expands them and allocates Q/P online. Bias and concat plaintexts are preencoded in all modes.

Preparation and Encode are separate: neither has the same boundary as the historical Orion Step-1 layer-cache category. Do not subtract these numbers from the whole-model profiles.

RSS is an externally sampled maximum, not a guaranteed transient allocation peak. Warmups are excluded.

This smoke block uses a fixed treatment order for correctness only.

This isolates storage/materialization within CIPS. It does not establish the Orion-versus-WPC layout crossover.

**Smoke run only: no balanced-order performance claim or confidence interval.**
