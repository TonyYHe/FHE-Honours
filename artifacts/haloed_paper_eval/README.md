# HaloED Paper Evaluation Artifacts

This folder contains Orion-side artifact scripts for regenerating the paper
evaluation figures and tables.  The scripts run local Orion workflows and then
render paper-facing outputs under `.tmp/results/haloed_paper_eval/`.

The paper repository is not modified by these scripts.  Original `tools/`
scripts are also left unchanged; the `eval/` scripts provide paper defaults,
safe output roots, and rendering/check helpers.

## Common Commands

Dry-run a target:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/fig7_operator_conv_perf.py --dry-run
```

Regenerate quick compile/plot artifacts:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/build_all.py --only fig7 table2 table3
```

Re-render from an existing root without rerunning long experiments:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/build_all.py \
  --check-existing .tmp/results/haloed_paper_eval
```

## Target Mapping

- `fig7_operator_conv_perf.py` wraps `tools/run_conv_kernel_table.py`.
- `table1_e2e_unet.py` wraps `tools/run_u22_dim32_dense_provider_e2e_matrix.py`.
- `fig8_cheb7_qualitative.py` wraps `tools/verify_medseg_cheb7_orion_clear_adapter.py`.
- `table2_dp_layout_breakdown.py` wraps `tools/generate_unet22_compile_plan_csv.py`.
- `table3_relayout_ablation.py` wraps `tools/run_u22_224_policy_compile_probe.py`.
- `bootstrap_analysis_numbers.py` extracts the paper bootstrap numbers from
  E2E output breakdowns.

Each target writes:

- `raw/`: direct Orion workflow outputs.
- `paper/`: paper-facing figures/tables/summaries.
- `manifest.json`: command, host, git state, output paths, and measurement
  convention.
