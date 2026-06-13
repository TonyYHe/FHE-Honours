# HaloED Paper Evaluation Artifacts

This folder contains Orion-side artifact scripts for regenerating the paper
evaluation figures and tables.  The scripts run the current Orion workflows
with the current artifact configuration and then render paper-facing outputs
under `.tmp/results/haloed_paper_eval/`.

Artifact evaluation should generate fresh results.  Historical JSON files under
`.tmp/results` are useful for development sanity checks, but they are not AE
inputs and are not embedded as defaults.  Derived paper artifacts, such as the
appendix compile/MVM table and the bootstrap summary, must be extracted from the
same fresh Table 1/E2E run root produced by the reviewer.

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

Regenerate the MedSeg scaled-SiLU reference vs Cheb7 accuracy table in PyTorch:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/accuracy_medseg_cheb7.py --run-kind pytorch
```

The default accuracy sample count is 10 fixed validation examples per task,
using the fixture data under `artifacts/haloed_paper_eval/fixtures/`.
MedSeg Cheb7 artifact defaults use the reduced-scale raw-gain fine-tuned
checkpoints.  COVID-256 uses the decoder-wide `rawgain_tight_g045` checkpoint;
NuSeg-384 uses the `dec4a1536_rawgain_g045` checkpoint.  The scale schedule is
saved in the checkpoint; artifact evaluation does not apply a runtime clamp.
The accuracy summary reports the scaled-SiLU PyTorch reference, the Cheb7
PyTorch/clear/CKKS result, and the Cheb7 drop relative to the scaled-SiLU
reference.  It does not require or report the older plain-SiLU baseline.
The CKKS backend defaults set `ORION_PROVIDER_MVM_MASKED_MATERIALIZATION=1`,
`ORION_BOOTSTRAP_LAYOUT_REFINEMENT=0`,
`ORION_LAYOUT_POLICY_RELAYOUT_KERNEL=0`, and `ORION_CONCAT_FUSION=0`.  The
`ORION_LATTIGO_BOOTSTRAP_MANY` flag is treated only as a Lattigo execution
scheduling setting.

Run the same fixed-index accuracy cases through the Orion cleartext or CKKS
backend:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/accuracy_medseg_cheb7.py --run-kind clear
.venv/bin/python artifacts/haloed_paper_eval/eval/accuracy_medseg_cheb7.py --run-kind ckks
```

Reproduce the COVID-256 single-image CKKS sanity case used for the
Concat-depth/low-postscale check:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/accuracy_medseg_cheb7.py \
  --run-kind ckks \
  --tasks covid19 \
  --counts 1 \
  --val-indices 1951 \
  --data-root artifacts/haloed_paper_eval/fixtures/fhelipe_medseg_accuracy10
```

For fixture-backed datasets, use `--original-val-indices` when you want to
select by the original validation ID recorded in `accuracy_original_val_indices`
instead of reproducing an exact verifier `--val-index` argument.

Re-render from an existing root produced by this artifact runner without
rerunning long experiments:

```bash
.venv/bin/python artifacts/haloed_paper_eval/eval/build_all.py \
  --check-existing .tmp/results/haloed_paper_eval
```

## Target Mapping

- `accuracy_medseg_cheb7.py` reproduces fixed-index MedSeg scaled-SiLU vs Cheb7
  accuracy in PyTorch, cleartext backend, or CKKS backend.
- `fig7_operator_conv_perf.py` wraps `tools/run_conv_kernel_table.py`.
- `table1_e2e_unet.py` wraps `tools/run_u22_dim32_dense_provider_e2e_matrix.py`.
- `appendix_compile_mvm.py` extracts the appendix BSGS-MVM count and compile
  table from the same fresh Table 1 E2E JSONs.
- `fig8_cheb7_qualitative.py` wraps `tools/verify_medseg_cheb7_orion_clear_adapter.py`.
- `table2_dp_layout_breakdown.py` wraps `tools/generate_unet22_compile_plan_csv.py`.
- `table3_relayout_ablation.py` wraps `tools/run_u22_224_policy_compile_probe.py`.
- `bootstrap_analysis_numbers.py` extracts the paper bootstrap numbers from
  the same fresh Table 1 E2E output breakdowns.

## Paper Evidence Ledger

- Fig. 7 convolution microbenchmarks: `fig7_operator_conv_perf.py`.
- Table 1 end-to-end speedup: `table1_e2e_unet.py`.
- End-to-end compile-time appendix table: `appendix_compile_mvm.py`, derived
  from the Table 1 run root.
- Fig. 8 qualitative examples: `fig8_cheb7_qualitative.py`, using the
  low-postscale Cheb7 checkpoints and fixture original validation IDs.
- Accuracy table/RQ3: `accuracy_medseg_cheb7.py`.
- Table 2 DP layout breakdown: `table2_dp_layout_breakdown.py`.
- Table 3 re-layout ablation: `table3_relayout_ablation.py`.
- RQ4 bootstrap and rotation analysis: `bootstrap_analysis_numbers.py`, derived
  from the Table 1 run root.

Each target writes:

- `raw/`: direct Orion workflow outputs.
- `paper/`: paper-facing figures/tables/summaries.
- `manifest.json`: command, host, git state, output paths, and measurement
  convention.
