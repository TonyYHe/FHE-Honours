# FHELIPE MedSeg U-Net 22 Base32 ReLU vs SiLU Verification

Current activation-replacement verification uses concat-skip U-Net 22, base dimension 32, and two benchmark input sizes:

- COVID-19 lung: 192x192, freshly retrained for ReLU and SiLU on 2026-06-01.
- NuSegMSBench: 224x224, from the existing retrained verification run.

## Accuracy

| dataset | input | ReLU Dice | SiLU Dice | Dice change | ReLU IoU | SiLU IoU | IoU change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COVID-19 lung | 192x192 | 0.981435 | 0.981743 | +0.000308 | 0.964803 | 0.965426 | +0.000622 |
| NuSegMSBench | 224x224 | 0.944918 | 0.943935 | -0.000983 | 0.898651 | 0.897110 | -0.001541 |

## Result Files

| dataset | verification JSON |
| --- | --- |
| COVID-19 lung | `.tmp/results/fhelipe_medseg_covid19_concat_unet22_192_base32_relu_silu_20260601.json` |
| NuSegMSBench | `.tmp/results/fhelipe_medseg_nusetmsb_concat_unet22_224_base32_plain_silu_poly7_repair20_fixed_20260527.json` |

## Takeaway

Across these two benchmark settings, replacing ReLU with SiLU changes Dice by at most 0.0010 and IoU by at most 0.0016, so the activation replacement itself is negligible for the paper evaluation.
