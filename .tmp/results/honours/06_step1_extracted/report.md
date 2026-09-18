# Step 1 profiling extraction

Accepted real-FHE profiles: **3** of **4** readable result files.

## Accepted real-FHE results

| Model | Mode | HE forward (s) | Online Encode (s) | Encode share | Bootstrap share | MVM share | Other share | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet20 CIFAR10 | dense | 1530.13 | 305.40 | 19.96% | 64.64% | 4.10% | 11.10% | 0.00010260 |
| U22 64 base32 | provider | 5575.58 | 5044.21 | 90.47% | 1.99% | 7.32% | 0.18% | 0.00000002 |
| VGG16/base16 ImageNet 224 | provider | 16508.11 | 7937.35 | 48.08% | 44.14% | 4.58% | 3.15% | 0.00014180 |

## Excluded or diagnostic results

| Result | Classification | Reason |
|---|---|---|
| VGG16/base16 ImageNet 224 | clear_structural | ORION_LATTIGO_CLEAR_BACKEND is enabled; canonical Step 1 profile is marked invalid |

## Interpretation rules

- Comparative plots contain only accepted real-FHE profiles. Clear-Lattigo timing is structural evidence and is not comparable to encrypted execution.
- `major_wall_categories` are additive, mutually exclusive wall-time categories. For an accepted schema-v2 profile their percentages close to 100% within the recorded tolerance.
- `operator_microprofile` values are diagnostic and non-additive. They may overlap or represent parallel/nested work, so do not sum them or use them as a wall-time pie chart.
- Operation counts are mean backend counts per measured forward, not seconds and not percentages.
- Error bars in the Encode comparison are sample standard deviations across measured forwards. Warmups are excluded.
