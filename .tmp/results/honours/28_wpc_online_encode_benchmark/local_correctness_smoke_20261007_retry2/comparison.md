# Three-way CIPS online-Encode benchmark

Status: ok. Independent process blocks: 1.

Same checkpoint, exact feature values, CIPS layout, CKKS configuration, and homomorphic operations in all modes.

**Functional-test parameters only:** LogN=10; not a secure-deployment or complete-network benchmark.

| Mode | Forward (s) | Encode share | Preparation + Encode share | Decompression share | Logical resident (MiB) | Sampled online peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| online_encode | 0.930719 | 27.985% | 31.005% | 0.000% | 2.097 | 671.969 |
| full | 0.657683 | 0.000% | 0.000% | 0.000% | 402.875 | 1100.750 |
| compressed | 0.718648 | 0.000% | 0.000% | 6.093% | 31.124 | 733.188 |

Entries are means of per-process-block statistics; shares are within-forward ratios, not ratios of aggregate medians.

| Paired latency ratio | Mean | 95% CI for mean |
|---|---:|---:|
| compressed_over_online_encode_forward_ratio | 0.772143 | not estimated (smoke) |
| compressed_over_full_forward_ratio | 1.092697 | not estimated (smoke) |
| full_over_online_encode_forward_ratio | 0.706639 | not estimated (smoke) |

percentile bootstrap of matched process blocks; mean of block statistics; not pooled-forward confidence intervals.

Lifecycle tracing and output validation run outside timed forwards. Encode counters count actual successful linear-transform Encode invocations, not individual diagonals.

Online mode stores compact float32 slot-period recipes; it expands them and allocates Q/P online. Bias and concat plaintexts are preencoded in all modes.

Preparation and Encode are separate: neither has the same boundary as the historical Orion Step-1 layer-cache category. Do not subtract these numbers from the whole-model profiles.

RSS is an externally sampled maximum, not a guaranteed transient allocation peak. Warmups are excluded; fresh-process order is balanced across six permutations.

This isolates storage/materialization within CIPS. It does not establish the Orion-versus-WPC layout crossover.

**Smoke run only: no balanced-order performance claim or confidence interval.**
