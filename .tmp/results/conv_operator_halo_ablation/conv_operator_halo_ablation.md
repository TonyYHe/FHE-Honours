# Conv Operator-Level Halo Ablation

Descriptor-only BSGS rotation counts. TConv rows are intentionally excluded.

| case | shape | geometry | original Orion | flat+halo | flat+halo+B | aligned+halo+B | gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r34_stage1 | 64x56x56 -> 64x56x56 | 16ch, source H36, target H34, 2 stripes | 2315 | 1555 | 1094 | 904 | 2.56x |
| r34_stage2 | 128x28x28 -> 128x28x28 | 64ch, source H18, target H16, 2 stripes | 1323 | 741 | 650 | 548 | 2.41x |
| r34_stage3 | 256x14x14 -> 256x14x14 | 256ch, source H9, target H7, 2 stripes | 499 | 312 | 312 | 312 | 1.60x |
| u22_256_enc1a | 3x256x256 -> 32x256x256 | 2ch, source H64, target H62, 5 stripes | 1536 | 1280 | 240 | 240 | 6.40x |

## Incremental Savings

| case | halo saving | B saving after halo | align saving after halo+B | total saving | reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| r34_stage1 | 760 | 461 | 190 | 1411 | 60.95% |
| r34_stage2 | 582 | 91 | 102 | 775 | 58.58% |
| r34_stage3 | 187 | 0 | 0 | 187 | 37.47% |
| u22_256_enc1a | 256 | 1040 | 0 | 1296 | 84.38% |

## Program Counts

| case | flat+halo programs | flat+halo B groups | aligned programs | aligned B groups |
| --- | ---: | ---: | ---: | ---: |
| r34_stage1 | 32 | 8 | 32 | 8 |
| r34_stage2 | 8 | 4 | 8 | 4 |
| r34_stage3 | 2 | 2 | 2 | 2 |
| u22_256_enc1a | 160 | 10 | 160 | 10 |
