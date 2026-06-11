# HaloED Paper Evaluation Artifacts

This folder contains Orion-side artifact scripts for regenerating the paper
evaluation figures and tables.  The scripts run local Orion workflows and then
render paper-facing outputs under `.tmp/results/haloed_paper_eval/`.

The paper repository is not modified by these scripts.  Original `tools/`
scripts are also left unchanged; the `eval/` scripts provide paper defaults,
safe output roots, and rendering/check helpers.

Before long E2E targets such as Table 1, the artifact runner rebuilds Orion's
C/C++ shared libraries on the host that will execute the run.  This is required
for glibc compatibility: do not rsync a `*.so` built on a newer Linux host to an
older one and reuse it for paper timings.  Go/Lattigo libraries are not rebuilt
by this preflight.

## Common Commands

Dry-run a target:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/fig7_operator_conv_perf.py --dry-run
```

Regenerate quick compile/plot artifacts:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/build_all.py --only fig7 table2 table3
```

Regenerate the MedSeg Cheb7 accuracy table in PyTorch:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/accuracy_medseg_cheb7.py --run-kind pytorch
```

The default accuracy sample count is 10 fixed validation examples per task.
MedSeg Cheb7 artifact defaults use the reduced-scale raw-gain fine-tuned
checkpoints.  COVID-256 uses the decoder-wide `rawgain_tight_g045` checkpoint;
NuSeg-384 uses the `dec4a1536_rawgain_g045` checkpoint.  The scale schedule is
saved in the checkpoint; artifact evaluation does not apply a runtime clamp.

Run the same fixed-index accuracy cases through the Orion cleartext or CKKS
backend:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/accuracy_medseg_cheb7.py --run-kind clear
.venv/bin/python artifacts/haloed_paper_eval/eval/accuracy_medseg_cheb7.py --run-kind ckks
```

Re-render from an existing root without rerunning long experiments:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/build_all.py \
  --check-existing .tmp/results/haloed_paper_eval
```

## Target Mapping

- `accuracy_medseg_cheb7.py` reproduces fixed-index MedSeg Cheb7 accuracy in
  PyTorch, cleartext backend, or CKKS backend.
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
