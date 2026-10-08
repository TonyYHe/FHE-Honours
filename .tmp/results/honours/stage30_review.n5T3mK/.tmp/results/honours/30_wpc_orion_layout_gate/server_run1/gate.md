# Matched-function Orion–CIPS decoder gate

Status: ok.

Both layouts evaluate the same checkpoint, synthetic features, flattened Rotation Padding, trained Cheb7 and bootstrap at the same CKKS levels.

| Layout / storage | Max clear error | Diagnostic forward (s) | Logical resident (MiB) | Sampled measured RSS (MiB) | LT rotations |
|---|---:|---:|---:|---:|---:|
| native_orion/full | 6.133e-07 | 11.414637 | 1118.766 | 3040.277 | 1456 |
| native_orion/online_encode | 6.139e-07 | 59.046041 | 2.903 | 1898.363 | 1456 |
| cips/full | 6.167e-07 | 10.517926 | 1931.500 | 3981.949 | 1288 |
| cips/online_encode | 6.125e-07 | 15.551135 | 7.484 | 2186.090 | 1288 |
| cips/compressed | 6.181e-07 | 11.032202 | 99.772 | 2284.500 | 1288 |

Maximum output delta versus CIPS/full: 2.980e-08.

Instrumented rotation/conjugation counts must match across storage policies within each layout; they need not match between layouts. Individual addition/multiplication counts are not separately measured.

Native Orion uses its ordinary square-embedding diagonal packers (low gap 2, high gap 1), with an opt-in cyclic-boundary adapter. Its aligned concat copies and drops ciphertexts by one level; CIPS uses encoded permutation transforms. Native online preparation rebuilds ordinary diagonals from a float32 kernel, not CIPS periodic recipes.

Full-Q/P transform evaluation is now measured, rather than reported as an unmeasured zero. Native Encode is GenerateLinearTransform call wall time; CIPS online Encode is the narrower backend Encode timer. These shares are not interchangeable with historical Step-1 categories.

**Scope:** one fixed-order diagnostic forward per fresh process, no warmups. No latency ranking, confidence interval, secure-deployment claim or complete-network/layout-crossover conclusion. RSS is sampled, not an enforced allocation ceiling. Logical storage excludes runtime object overhead; native online Q/P estimates are checked against the native full worker's actual coefficient arrays.

Raw worker/RSS files, exact configuration bytes, source/binary hashes and checkpoint identity are retained in gate.json provenance.
