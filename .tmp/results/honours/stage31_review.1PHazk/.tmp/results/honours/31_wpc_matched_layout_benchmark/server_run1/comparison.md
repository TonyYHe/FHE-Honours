# Matched-function Orion–CIPS layout/storage benchmark

Status: ok. Fresh-process blocks: 10. Smoke: False.

Same checkpoint, exact synthetic features, flattened Rotation Padding, Cheb7, bootstrap and CKKS levels. Native preparation uses channel-pruned block regeneration; Stage-30 timings are not reused.

| Treatment | Forward (s) | Prep + Encode (%) | Decompress (%) | Logical resident (MiB) | Sampled measured peak RSS (MiB) | LT rotations |
|---|---:|---:|---:|---:|---:|---:|
| native_orion/full | 11.038271 | 0.000 | 0.000 | 1118.766 | 4166.263 | 1456 |
| native_orion/online_encode | 18.926400 | 42.544 | 0.000 | 2.903 | 1916.610 | 1456 |
| cips/full | 10.396853 | 0.000 | 0.000 | 1931.500 | 5968.816 | 1288 |
| cips/online_encode | 15.264793 | 33.723 | 0.000 | 7.484 | 2187.041 | 1288 |
| cips/compressed | 10.808728 | 0.000 | 5.866 | 99.772 | 2380.951 | 1288 |

Forward entries are means of within-process medians. Component shares are means of within-forward ratios. RSS is the mean of sampled worker maxima, not an allocation ceiling.

| Paired forward ratio (numerator / denominator) | Mean | 95% CI for mean |
|---|---:|---:|
| cips/compressed / native_orion/online_encode | 0.571100 | [0.568890, 0.573328] |
| cips/compressed / native_orion/full | 0.979216 | [0.976591, 0.981629] |
| cips/full / native_orion/full | 0.941922 | [0.937806, 0.946253] |
| cips/online_encode / native_orion/online_encode | 0.806552 | [0.803652, 0.809633] |
| cips/compressed / cips/online_encode | 0.708084 | [0.705663, 0.710203] |
| cips/compressed / cips/full | 1.039632 | [1.035358, 1.043382] |

## Forward wall accounting

Mean seconds per forward; components plus residual close to the mean wall, not the median above.

| Treatment | Wall | Preparation | Encode | Decompress | LT evaluation | Activation | Bootstrap | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native_orion/full | 11.043565 | 0.000000 | 0.000000 | 0.000000 | 7.467223 | 0.076538 | 3.475283 | 0.024520 |
| native_orion/online_encode | 18.962662 | 2.732867 | 5.335256 | 0.000000 | 7.379130 | 0.073702 | 3.415786 | 0.025921 |
| cips/full | 10.387250 | 0.000000 | 0.000000 | 0.000000 | 6.732866 | 0.076530 | 3.488764 | 0.089090 |
| cips/online_encode | 15.286382 | 0.572205 | 4.583241 | 0.000000 | 6.544429 | 0.073217 | 3.423352 | 0.089938 |
| cips/compressed | 10.846372 | 0.000000 | 0.000000 | 0.637631 | 6.618083 | 0.073323 | 3.428014 | 0.089321 |

percentile bootstrap of matched fresh-process blocks; mean of block medians for latency; descriptive pointwise intervals, not multiple-comparison-adjusted.

Ten-block Williams cycles balance treatment positions and ordered immediate predecessors; they do not enumerate all 120 treatment permutations. Warmups and the correctness preflight are excluded. Output validation and lifecycle tracing are outside timed forwards. Every measured output and operation-count sample is retained and independently checked.

Native Encode includes the GenerateLinearTransform allocation/binding call; CIPS Encode is the narrower backend timer. Components are not interchangeable with historical Step-1 categories. Other includes concat, bias/rescale, wrappers and remaining call/allocation overhead; it is not a cryptographic operator measurement.

Maximum final-output delta across treatments: 2.608e-08. Within-layout storage policies have identical instrumented operations; layout operation counts may differ. Individual additions/multiplications are not separately instrumented.

**Scope:** functional, security-unassessed CKKS parameters, one checkpoint-derived decoder stage and synthetic internal tensors. This is not a complete-network crossover, secure deployment, dataset accuracy evaluation or a general claim about native Orion. The order-balanced process-block intervals describe this run and do not establish reproducibility across hosts or sessions.

Raw outputs, timings, RSS/guard observations, exact configuration bytes and source/checkpoint/binary hashes are archived. Hashes establish content identity, not authenticity; checkpoint and binary bytes are not bundled.
