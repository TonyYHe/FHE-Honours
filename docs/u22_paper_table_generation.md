# U22 Paper Table Generation

This note records the compile-only commands used to generate the two U-Net
tables in the HaloED paper. Both commands run Orion's planner and metadata
estimators only; they do not generate HE plaintexts or execute encrypted
inference.

## Operator Plan Table

Generate the full 27-operator table, including the 23 learned/linear U-Net
operators, four pooling transition operators, layout rows, rotations,
diagonal/CT-PT counts, and compile-time bootstrap placement:

```bash
.venv/bin/python tools/generate_unet22_compile_plan_csv.py \
  --base-dim 64 \
  --out-csv .tmp/results/unet22_plus_output_dim64_real_trace_compile_plan_4cases.csv \
  --out-tex .tmp/results/unet22_operator_plan_dim64.tex
```

For the base-channel-32 variant, change only the dimension and output names:

```bash
.venv/bin/python tools/generate_unet22_compile_plan_csv.py \
  --base-dim 32 \
  --out-csv .tmp/results/unet22_plus_output_dim32_real_trace_compile_plan_4cases.csv \
  --out-tex .tmp/results/unet22_operator_plan_dim32.tex
```

The four built-in cases are:

- `192x192_IBSR_BRAIN_2D`: IBSR Brain 2D, `C_in=1`, `C_out=4`.
- `224x224_HanCo_Hand`: HanCo hand segmentation, `C_in=3`, `C_out=1`.
- `384x288_CVC_ClinicDB`: CVC-ClinicDB, `C_in=3`, `C_out=1`.
- `384x384_Satellite_cloud`: satellite cloud segmentation, `C_in=4`, `C_out=1`.

The LaTeX table reports output-layout interior rows `alpha`, averaged actual
output halo `beta=floor((beta_top+beta_bottom)/2)`, ciphertext count,
estimated rotations, diagonal/CT-PT multiplication count, and bootstrap count
after the operator or its following activation. Pooling rows are kept explicit
because producer-fused output re-layout can be absorbed by pooling.

## Re-Layout Ablation Table

Generate the small compile-plan ablation table with policy-specific bootstrap
counts:

```bash
.venv/bin/python tools/generate_unet22_layout_ablation_table.py \
  --network u22_320_base32 \
  --out-csv .tmp/results/layout_policy_ablation_u22_320_base32_paper.csv \
  --out-tex .tmp/results/layout_policy_ablation_u22_320_base32_paper.tex
```

The same script can be pointed at `u22_192_base32`, `u22_224_base32`, or
`u22_256_base32` if the paper should use a different ablation size:

```bash
.venv/bin/python tools/generate_unet22_layout_ablation_table.py \
  --network u22_256_base32 \
  --out-csv .tmp/results/layout_policy_ablation_u22_256_base32_paper.csv \
  --out-tex .tmp/results/layout_policy_ablation_u22_256_base32_paper.tex
```

`Boot` is computed at the compile-time bootstrap boundaries selected by
`BootstrapSolver`, then multiplied by each policy's selected output ciphertext
count at that boundary. This makes the ablation policy-aware without running
end-to-end FHE.
