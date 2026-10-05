# Validation and checkpoint provenance repair

## Changes

The 5 October audit reproduced cases where successful status flags concealed
inconsistent measurements. This repair changes validation, not the model,
training recipe, packing layout, or timed evaluation algorithm.

- Step 1 percentages are recomputed from seconds and HE-forward wall time.
  Major categories must close under an independent fixed tolerance. Successful
  measured attempts must match the declared count, aggregate means, category
  means and sample standard deviations. Warmups are excluded. The aggregate
  Encode percentage remains a **ratio of means**; the mean and standard
  deviation of individual percentages are checked separately.
- Census validation streams both unique and occurrence records, checks their
  identity sets, detects changed logical payload hashes, and reconstructs
  counts, byte totals and coverage from the records. Required metadata and
  encoded-Q/P pass counts cannot be replaced by a success flag. The reported
  bytes remain logical/analytical payload estimates, not measured RSS.
- Isolated decoder workers require all named gates, exactly the requested
  number of finite timing samples, positive forward/bootstrap times,
  nonnegative subtimers that fit inside the forward, finite nonempty outputs
  matching their shape, bounded clear-reference errors, and reconciled
  operation totals. Synthesis recomputes all benchmark summary statistics,
  ratios, memory differences, operation equality and output differences.
  `comparison.json`'s embedded workers must match the raw `full.worker.json`
  and `compressed.worker.json`; those files join the input-hash manifest.
- Accuracy result **schema 5** records `evaluated_checkpoint.sha256` for the
  exact bytes deserialized, its source hash, epoch and padding semantics.
  Evaluation-only mode now verifies the existing schema-4 checkpoint's source
  identity, finite weights, fixed polynomial activation representation and
  complete training audits. Adapter reload checks tensor values, not just
  names. Results include hashes of best, last and immutable epoch files.
  Checkpoint changes during evaluation prevent publication. Evaluation-only
  does not write checkpoint files or update weights.
- Decoder correctness **schema 2** records its actual error tolerance and
  checks all five stages, zero online Encode calls, 56 transforms, operation
  equality and released materialization. Its loader and isolated workers hash
  the same bytes they deserialize. The overall storage ratio now includes the
  common concat payload in **both** numerator and denominator; the historical
  correctness result included it only in the denominator (12.9261× versus
  12.9442× for the supplied server byte totals).
- Synthesis **schema 2** requires the same SHA-256 across the supplied
  checkpoint, accuracy, decoder correctness and isolated benchmark. Paths are
  informational, so identical checkpoints can be copied across machines.
  Missing gates, booleans masquerading as numbers, NaN/Infinity, stale summary
  values and same-path replacement checkpoints fail validation.

Hashes identify content, not authenticity. Torch checkpoints remain trusted
local artifacts. These checks establish artifact consistency; they do not
prove timer instrumentation is unbiased or make the small decoder CKKS test
parameters deployment-secure. Dataset-file hashing and a genuinely untouched
test split remain separate work. The 2,115-image split includes the 512 images
used to select the checkpoint and is labelled validation, not an untouched test.

## Preserve and regenerate evidence

Do not edit old JSON files to add hashes or change their schema numbers: their
evaluated-best hash cannot be recovered honestly from a path alone. Keep the
old archives unchanged. No training is required. Re-evaluate the existing
best checkpoint on all 2,115 validation images, repeat the short decoder
correctness gate with a recorded tolerance, then validate the existing Step-1,
census and isolated-worker evidence and rebuild the four figures.

After synchronizing these code changes to the server, run:

```bash
cd ~/FHE-Honours
mkdir -p .tmp/results/honours/27_wpc_validation_provenance
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup bash tools/revalidate_wpc_evidence.sh \
  > .tmp/results/honours/27_wpc_validation_provenance/server_run1.launch.log 2>&1 < /dev/null &
echo $! > .tmp/results/honours/27_wpc_validation_provenance/server_run1.pid
```

The script uses `.venv/bin/python`, runs regression tests, stops at the first
failure and refuses to overwrite an existing `server_run1` directory. A repeat
uses a fresh directory, for example:

```bash
bash tools/revalidate_wpc_evidence.sh .tmp/results/honours/27_wpc_validation_provenance/server_run2
```

Allow several minutes: the previous full validation took about 274 seconds on
this server; no exact runtime is guaranteed. The background command survives an
SSH disconnect. It reuses the built Lattigo library and existing Stage-25 raw
workers; it does not repeat the timing benchmark or modify historical results.

Check completion and inspect any failed stage with:

```bash
cd ~/FHE-Honours
tail -n 30 .tmp/results/honours/27_wpc_validation_provenance/server_run1.launch.log
cat .tmp/results/honours/27_wpc_validation_provenance/server_run1/exit_status
```

Success requires exit status `0`, `VALIDATION RECOVERY COMPLETE` and
`server_run1/synthesis/synthesis.json` with `status: ok`. Stage-specific logs
are `tests.log`, `full_validation.log`, `decoder_correctness.log` and
`synthesis.log`. If the output directory already exists or a prerequisite is
missing, the script exits before creating `exit_status`; inspect the launch log.

The final report is `server_run1/synthesis/report.md`, with four PNG figures and
the JSON/CSV evidence tables alongside it. A validation failure must be fixed
or supplied with missing raw artifacts; it must not be bypassed by copying old
acceptance flags.

## Verification

Regression tests cover the audit reproductions: mutually agreeing but false
percentages, an altered median ratio, empty/truncated timing arrays, missing
gates, false online-Encode counters, non-finite outputs, broken sample
accounting, checkpoint replacement at the same path, and a path replacement
during deserialization. Schema-4 audited checkpoints without source-model
blend parameters remain supported; legacy schema-4 **accuracy results** do not.
The available original three-model extraction passes the new independent checks.
Local verification passed 172 targeted Python tests across validation,
profiling, checkpoint, census and CIPS regression suites. Shell syntax and
`git diff --check` passed. No Go evaluator code was changed.
The server-only checkpoint, raw worker files and census files are not present
locally, so the complete server synthesis is intentionally not regenerated here.
