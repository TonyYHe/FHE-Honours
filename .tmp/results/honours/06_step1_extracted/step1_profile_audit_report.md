# Step 1 online Encode profiling report

Audit date: 3 September 2026

## Scope and audit outcome

This report summarizes and audits the outputs produced by:

```bash
python tools/extract_step1_results.py
```

The extractor read four primary result files from `.tmp/results/honours/04_step1_model_matrix` and `.tmp/results/honours/05_vgg16_profile_fix`. It reported no input-read errors. Three schema-v2 real-FHE profiles passed the complete Step 1 acceptance gate: ResNet20, U-Net22, and VGG16/base16. The fourth result is the VGG16 clear-Lattigo structural audit; it completed successfully but was correctly excluded because `ORION_LATTIGO_CLEAR_BACKEND=1` and its canonical Step 1 profile is invalid for real-FHE timing.

The generated JSON and CSV data were compared with all four source JSON files. Headline values, additive categories, non-additive operator microprofiles, operation counts, per-attempt values, classifications, and acceptance decisions agree with the sources. All additive wall-time categories independently sum to the corresponding HE-forward time and 100%. All four PNG files are valid, non-empty images and were visually inspected. No obvious extraction, arithmetic, missing-data, classification, or plotting errors were identified.

## Concise result summary

| Model | Mode | Compile (s) | HE forward, mean ± SD (s) | Online Encode, mean ± SD (s) | Canonical Encode share | Per-forward share, mean ± SD | Classification |
|---|---|---:|---:|---:|---:|---:|---|
| ResNet20 | dense | 198.327826 | 1530.126458 ± 13.282474 | 305.402551 ± 3.458904 | 19.959301% | 19.959162% ± 0.113770 pp | accepted real FHE |
| U-Net22 | provider | 282.060900 | 5575.576386 ± 34.695413 | 5044.205487 ± 22.326499 | 90.469669% | 90.470637% ± 0.324733 pp | accepted real FHE |
| VGG16/base16 | provider | 610.941519 | 16508.111871 ± 197.383660 | 7937.350378 ± 129.801359 | 48.081516% | 48.079899% ± 0.228297 pp | accepted real FHE |
| VGG16/base16 clear | provider | 63.135263 | 225.584192 ± 0.000000 | 177.150251 ± 0.000000 | 78.529550% | 78.529550% ± 0.000000 pp | clear structural; excluded |

The reported SD is the sample standard deviation across measured forwards; the single clear structural forward therefore has an SD of zero. The canonical Encode share is `mean(Encode seconds) / mean(HE-forward seconds) × 100`. The separately reported per-forward share mean is `mean(Encode_i / HE_i × 100)`. Their very small differences are the expected consequence of these two aggregation definitions, not an extraction discrepancy.

The main pattern is model-dependent: online Encode occupies about one fifth of ResNet20 wall time, almost all of U-Net22 wall time, and nearly half of VGG16 wall time. Bootstrapping dominates ResNet20, while VGG16 is split primarily between online Encode and bootstrapping.

## Run configuration and acceptance

| Model | `CLEAR_BACKEND` | Single-slot cache | Streaming LT | Legacy chunk streaming | Encode workers | Schema | Profiles / measured / requested | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ResNet20 | 0 | 1 | 0 | 0 | 1 | 2 | 3 / 3 / 3 | yes |
| U-Net22 | 0 | 1 | 0 | 0 | 1 | 2 | 3 / 3 / 3 | yes |
| VGG16/base16 | 0 | 1 | 0 | 0 | 1 | 2 | 3 / 3 / 3 | yes |
| VGG16/base16 clear | 1 | 1 | 0 | 0 | 1 | 2 | 1 / 1 / 1 | no: clear structural only |

All three accepted experiments used the same single Encode worker and disabled both streaming modes, so their runtime-mode settings are directly comparable. The clear result used the same cache and streaming settings but a different backend and is not timing-comparable.

## Per-forward measurements

Warmups are shown for completeness but are excluded from the means and standard deviations above.

| Model | Attempt | Kind | Status | HE forward (s) | Online Encode (s) | Encode share |
|---|---:|---|---|---:|---:|---:|
| ResNet20 | 0 | warmup | ok | 1546.656217 | 308.399064 | 19.939729% |
| ResNet20 | 1 | measured | ok | 1544.931661 | 308.253331 | 19.952554% |
| ResNet20 | 2 | measured | ok | 1526.192081 | 306.399721 | 20.076092% |
| ResNet20 | 3 | measured | ok | 1519.255631 | 301.554599 | 19.848839% |
| U-Net22 | 0 | warmup | ok | 5513.092898 | 4979.328548 | 90.318241% |
| U-Net22 | 1 | measured | ok | 5603.869819 | 5069.533098 | 90.464862% |
| U-Net22 | 2 | measured | ok | 5585.993523 | 5035.707812 | 90.148830% |
| U-Net22 | 3 | measured | ok | 5536.865815 | 5027.375552 | 90.798219% |
| VGG16/base16 | 0 | warmup | ok | 16897.043789 | 8085.672055 | 47.852584% |
| VGG16/base16 | 1 | measured | ok | 16733.601052 | 8087.220254 | 48.329228% |
| VGG16/base16 | 2 | measured | ok | 16424.112383 | 7864.045750 | 47.881100% |
| VGG16/base16 | 3 | measured | ok | 16366.622177 | 7860.785129 | 48.029368% |
| VGG16/base16 clear | 0 | measured | ok | 225.584192 | 177.150251 | 78.529550% |

Measured-run variability is small: the Encode-share SD is 0.113770 percentage points for ResNet20, 0.324733 points for U-Net22, and 0.228297 points for VGG16. No forward attempt failed.

## Correctness statistics

| Model | Reference shape | Actual shape | Compared values | MAE | RMSE | Maximum absolute error | Shape match |
|---|---|---|---:|---:|---:|---:|---|
| ResNet20 | `[1, 10]` | `[1, 10]` | 10 | 1.026004538e-4 | 1.198345781e-4 | 2.019181848e-4 | yes |
| U-Net22 | `[1, 1, 64, 64]` | `[1, 1, 64, 64]` | 4096 | 2.276418876e-8 | 2.920418574e-8 | 1.192092896e-7 | yes |
| VGG16/base16 | `[1, 1000]` | `[1, 1000]` | 1000 | 1.417963067e-4 | 1.802419720e-4 | 6.256829947e-4 | yes |
| VGG16/base16 clear | `[1, 1000]` | `[1, 1000]` | 1000 | 5.178216845e-3 | 6.486488506e-3 | 2.234802023e-2 | yes |

All output shapes and element counts match. The clear structural VGG16 result has materially larger numerical error than the accepted real-FHE VGG16 result. This value is present in the source data and was extracted correctly. It does not affect the Step 1 conclusions because the clear result is explicitly a structural execution audit, has an invalid Step 1 timing profile, and is excluded from every comparative plot. It should not be reused as numerical-accuracy evidence without a separate clear-backend investigation.

## Additive major wall-time categories

Each cell is `seconds / percent of HE-forward wall time`. These categories are mutually exclusive and can be summed.

| Category | ResNet20 | U-Net22 | VGG16/base16 | VGG16 clear (diagnostic) |
|---|---:|---:|---:|---:|
| Online Encode | 305.402551 / 19.9593013% | 5044.20549 / 90.4696688% | 7937.35038 / 48.0815156% | 177.150251 / 78.5295501% |
| Bootstrap | 989.138117 / 64.6442072% | 111.212872 / 1.99464349% | 7286.04342 / 44.1361403% | 0 / 0% |
| MVM kernel | 62.663008 / 4.09528295% | 408.378381 / 7.32441549% | 755.999395 / 4.57956307% | 13.054901 / 5.78715238% |
| Other HE forward | 169.838103 / 11.0996122% | 9.98912595 / 0.179158624% | 519.389655 / 3.14626930% | 35.3243083 / 15.6590353% |
| Linear wrapper/postprocess | 3.08281207 / 0.201474333% | 0 / 0% | 0.029268497 / 0.000177298% | 0.000436544 / 0.000193517% |
| Provider/executor overhead | 0 / 0% | 1.77262656 / 0.031792705% | 9.24657249 / 0.056012296% | 0.046057940 / 0.020417184% |
| Runtime load/trim | 0 / 0% | 0.000418577 / 7.50733761e-6% | 0.001074872 / 6.51117267e-6% | 0.000476577 / 0.000211263% |
| Layer-cache key preparation | 2.58659323e-6 / 1.69044409e-7% | 9.00899371e-6 / 1.61579594e-7% | 2.04717120e-5 / 1.24010015e-7% | 1.07152155e-5 / 4.74998507e-6% |
| Layer-cache eviction | 0.001864343 / 0.000121842% | 0.017466142 / 0.000313262% | 0.052091409 / 0.000315550% | 0.007750083 / 0.003435561% |
| Layer-cache other | 0 / 0% | 0 / 0% | 0 / 0% | 0 / 0% |

Accounting checks:

| Model | Category sum (s) | Category sum | Closure error (s) | Allowed tolerance (s) | Valid |
|---|---:|---:|---:|---:|---|
| ResNet20 | 1530.12645775 | 100% | 0 | 1.53012646e-6 | yes |
| U-Net22 | 5575.57638575 | 100% | -9.09494702e-13 | 5.57557639e-6 | yes |
| VGG16/base16 | 16508.1118707 | 100% | 0 | 1.65081119e-5 | yes |
| VGG16/base16 clear | 225.584192334 | 100% | 0 | 1.00000000e-6 | yes |

The U-Net22 closure error is approximately `9.1e-13` seconds, many orders of magnitude below tolerance and consistent with floating-point rounding.

![Online Encode comparison](online_encode_comparison.png)

The left panel compares the complete HE-forward wall time with its online Encode subset. The right panel makes the proportions directly comparable and shows the measured-forward sample-standard-deviation error bars. U-Net22 is the clearest Encode optimization target, while the smaller ResNet20 Encode share means that Encode-only optimization has a lower upper bound there.

![Additive major wall categories](major_wall_categories_pct.png)

The stacked bars contain only accepted real-FHE profiles and close to 100%. ResNet20 is bootstrap-dominated; U-Net22 is Encode-dominated; VGG16 is primarily split between Encode and bootstrap. MVM kernels are between approximately 4.10% and 7.32% for the accepted models, and all wrapper/cache-management overheads are small.

## Diagnostic operator microprofile

Each cell is `diagnostic seconds / diagnostic seconds divided by HE-forward wall time`. These timers are explicitly non-additive because they can be nested or accumulated across parallel work. They must not be summed into a wall-time partition. In particular, the ResNet20 diagnostic percentages sum above 100%; this is expected and is not an accounting error.

| Diagnostic timer | ResNet20 | U-Net22 | VGG16/base16 | VGG16 clear (diagnostic) |
|---|---:|---:|---:|---:|
| Bootstrap | 989.138117 / 64.6442072% | 111.212872 / 1.99464349% | 7286.04342 / 44.1361403% | 0 / 0% |
| LT fused multiply-accumulate | 106.093101 / 6.93361656% | 274.654414 / 4.92602728% | 923.668548 / 5.59524042% | 0 / 0% |
| LT rotation | 453.483646 / 29.6370044% | 167.910675 / 3.01153931% | 3413.11096 / 20.6753564% | 0 / 0% |
| Explicit accumulate | 0 / 0% | 0.370594422 / 0.006646746% | 6.64787590 / 0.040270359% | 0.040524078 / 0.017964059% |
| Elementwise Add module wall | 0.041114826 / 0.002687021% | 0 / 0% | 0 / 0% | 0 / 0% |
| Elementwise Multiply module wall | 576.242590 / 37.6598017% | 0 / 0% | 3292.68257 / 19.9458460% | 0.238139663 / 0.105565758% |

![Operator microprofile](operator_microprofile_pct.png)

Rotations are substantial diagnostics for ResNet20 and VGG16, whereas the U-Net22 diagnostic ratios are comparatively small because its much larger online Encode wall time dominates the denominator. Elementwise Multiply module wall time is visible for ResNet20 and VGG16. The Add timer is present for ResNet20 but is too small to be visually prominent at this scale.

## Lattigo linear-transform operation counts

Counts are means per measured forward; they are neither seconds nor percentages.

| Operation | ResNet20 | U-Net22 | VGG16/base16 | VGG16 clear |
|---|---:|---:|---:|---:|
| Transform count | 352 | 173 | 3593 | 0 |
| Diagonal terms | 17651 | 74064 | 196659 | 0 |
| Q multiply | 4102 | 2744 | 28224 | 0 |
| QP multiply | 31200 | 145384 | 365094 | 0 |
| Baby rotations | 1551 | 1703 | 11766 | 0 |
| Giant rotations | 2058 | 1533 | 18438 | 0 |
| Inner reductions | 17686 | 42012 | 169366 | 0 |
| Outer reductions | 2604 | 1328 | 24404 | 0 |
| Final modulus-down operations | 704 | 346 | 7186 | 0 |

![Linear-transform operation counts](operation_counts_log.png)

VGG16 has the largest count for every reported operation, consistent with its largest absolute HE-forward and Encode times. U-Net22 has fewer transforms than ResNet20 but many more diagonal terms and QP multiplications, which is consistent with its much larger Encode workload. The clear backend records zero Lattigo backend operation counts as expected and is omitted from the plot.

## Notable observations and caveats

1. U-Net22 is strongly Encode-bound: online Encode accounts for `90.469669%` of HE-forward time. This is the strongest candidate for an Encode-memory or Encode-latency optimization.
2. ResNet20 is bootstrap-bound: bootstrapping accounts for `64.644207%`, compared with `19.959301%` for Encode. An Encode-only optimization cannot address most of its current wall time.
3. VGG16 has two similarly important costs: Encode is `48.081516%` and bootstrapping is `44.136140%`. Improvements to either can materially affect total latency.
4. The three real-FHE results are stable across measured runs: Encode-share SD remains below `0.325` percentage points.
5. All accepted results match the clear output shape and have small reported numerical errors. U-Net22 has the smallest error of the three.
6. Microprofile percentages are diagnostic ratios, not operator proportions forming a partition. Their plot is suitable for comparing individual timers but not for deriving an “all operators sum to 100%” chart.
7. The clear VGG16 artifact is useful only as structural evidence. Its timing, zero backend counters, and larger MAE must not be mixed with real-FHE results.

## Output inventory and completeness checks

| Output | Contents | Audit result |
|---|---|---|
| [`extracted_results.json`](extracted_results.json) | Four structured records, acceptance decisions, profiles, and attempts | parsed; values match source JSON |
| [`runs.csv`](runs.csv) | Four run rows | 4/4 present |
| [`attempts.csv`](attempts.csv) | Three warmups and ten measured forwards | 13/13 present; all status `ok` |
| [`major_wall_categories.csv`](major_wall_categories.csv) | Ten categories for each of four runs | 40/40 rows present; all marked additive |
| [`operator_microprofile.csv`](operator_microprofile.csv) | Six diagnostics for each of four runs | 24/24 rows present; all marked non-additive |
| [`operation_counts.csv`](operation_counts.csv) | Nine operation counts for each of four runs | 36/36 rows present |
| [`online_encode_comparison.png`](online_encode_comparison.png) | Absolute and proportional Encode comparison | valid PNG; visually checked |
| [`major_wall_categories_pct.png`](major_wall_categories_pct.png) | Additive accepted-profile wall partition | valid PNG; visually checked |
| [`operator_microprofile_pct.png`](operator_microprofile_pct.png) | Non-additive accepted-profile diagnostics | valid PNG; visually checked |
| [`operation_counts_log.png`](operation_counts_log.png) | Accepted-profile counts on a logarithmic scale | valid PNG; visually checked |

## Final validation statement

The extracted results are complete and internally consistent with the four source artifacts and with the extractor's intended acceptance rules. No obvious errors, anomalous arithmetic, missing primary data, accidental inclusion of progress checkpoints, or misleading clear-versus-FHE plot comparisons were identified. The three accepted real-FHE profiles are suitable for the Step 1 analysis, subject to the stated distinction between additive wall categories and non-additive diagnostic microtimers.
