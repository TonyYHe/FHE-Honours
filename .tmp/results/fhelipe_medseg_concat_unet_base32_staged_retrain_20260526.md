# FHELIPE MedSeg U-Net 22+Output Base32 Activation Verification

Current activation-replacement verification uses Orion's U-Net 22 body plus explicit output head, base dimension 32, and only the two retained benchmark input sizes:

- COVID-19 lung: 256x256, retrained from scratch for ReLU and SiLU.
- NuSegMSBench: 384x384, retrained from scratch for ReLU and SiLU.
- Chebyshev-SiLU d7 fine-tuning starts from the scratch-trained SiLU model and replaces SiLU with a degree-7 Chebyshev activation using per-layer pre/post scale.

## Accuracy

| dataset | input | ReLU Dice/IoU | SiLU Dice/IoU | Cheb-SiLU d7 FT Dice/IoU |
| --- | ---: | ---: | ---: | ---: |
| COVID-19 lung | 256x256 | 0.983348 / 0.968400 | 0.976484 / 0.955737 | 0.899345 / 0.826324 |
| NuSegMSBench | 384x384 | 0.946815 / 0.902100 | 0.942850 / 0.895271 | 0.435153 / 0.296983 |

## Precision Loss

| dataset | ReLU -> SiLU Dice/IoU drop | SiLU -> Cheb-SiLU d7 FT Dice/IoU drop |
| --- | ---: | ---: |
| COVID-19 lung | 0.006864 / 0.012663 | 0.077139 / 0.129413 |
| NuSegMSBench | 0.003965 / 0.006829 | 0.507696 / 0.598288 |

## Result Files

| dataset | ReLU/SiLU JSON | Cheb-SiLU d7 FT JSON |
| --- | --- | --- |
| COVID-19 lung | `.tmp/results/fhelipe_medseg_covid19_unet22_plus_output_256_base32_relu_silu_20260603.json` | `.tmp/results/fhelipe_medseg_covid19_unet22_plus_output_256_base32_silu_cheb7_margin1p5_finetune20_lr1e-6_20260603.json` |
| NuSegMSBench | `.tmp/results/fhelipe_medseg_nusetmsb_unet22_plus_output_384_base32_relu_silu_20260603.json` | `.tmp/results/fhelipe_medseg_nusetmsb_unet22_plus_output_384_base32_silu_cheb7_margin1p5_finetune20_lr5e-8_clip0p1_20260603.json` |

## Takeaway

Fresh U-Net 22+Output base32 retraining keeps the ReLU-to-SiLU activation replacement within 0.006864 Dice and 0.012663 IoU on the retained benchmarks. Degree-7 Chebyshev-SiLU with pre/post scale is recoverable by fine-tuning on COVID-19 lung, but remains far below the SiLU baseline on NuSegMSBench.
