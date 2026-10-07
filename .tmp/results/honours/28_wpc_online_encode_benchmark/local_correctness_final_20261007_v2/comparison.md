# Three-way CIPS online-Encode benchmark

Status: ok. Independent process blocks: 1.

Same checkpoint, exact feature values, CIPS layout, CKKS configuration, and homomorphic operations in all modes.

**Functional-test parameters only:** LogN=10; not a secure-deployment or complete-network benchmark.

| Mode | Forward (s) | Encode share | Preparation + Encode share | Decompression share | Logical resident (MiB) | Sampled online peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| online_encode | 0.928972 | 28.080% | 30.570% | 0.000% | 2.097 | 688.188 |
| full | 0.649821 | 0.000% | 0.000% | 0.000% | 402.875 | 1108.547 |
| compressed | 0.687905 | 0.000% | 0.000% | 6.547% | 31.124 | 741.797 |

Entries are means of per-process-block statistics; shares are within-forward ratios, not ratios of aggregate medians.

| Paired latency ratio | Mean | 95% CI for mean |
|---|---:|---:|
| compressed_over_online_encode_forward_ratio | 0.740501 | not estimated (smoke) |
| compressed_over_full_forward_ratio | 1.058606 | not estimated (smoke) |
| full_over_online_encode_forward_ratio | 0.699506 | not estimated (smoke) |

correctness-only smoke block; no uncertainty estimation or performance claim.

Lifecycle tracing and output validation run outside timed forwards. Encode counters count actual successful linear-transform Encode invocations, not individual diagonals.

Online mode stores compact float32 slot-period recipes; it expands them and allocates Q/P online. Bias and concat plaintexts are preencoded in all modes.

Preparation and Encode are separate: neither has the same boundary as the historical Orion Step-1 layer-cache category. Do not subtract these numbers from the whole-model profiles.

RSS is an externally sampled maximum, not a guaranteed transient allocation peak. Warmups are excluded.

This smoke block uses a fixed treatment order for correctness only.

This isolates storage/materialization within CIPS. It does not establish the Orion-versus-WPC layout crossover.

**Smoke run only: no balanced-order performance claim or confidence interval.**
