# WPC–Orion trade-off synthesis

## Purpose

This stage joins the completed measurements without conflating their scopes:

1. whole-model Orion real-FHE Step-1 profiles for ResNet20, U-Net22, and
   VGG16;
2. exact periodicity censuses of the unchanged Orion layouts;
3. complete clear-model U-Net accuracy under native and WPC Rotation-Padding
   semantics;
4. encrypted correctness for the fine-tuned WPC decoder stage; and
5. isolated full-Q/P versus compressed-Q/P latency and memory for that same
   decoder stage.

The runner produces a final evidence table and figures. It does not reinterpret
the decoder-stage benchmark as complete encrypted U-Net performance.

## Validation policy

`tools/synthesize_wpc_orion_tradeoff.py` fails without writing a report when
any required condition is false. It independently checks:

- exactly one accepted schema-v2 real-FHE profile for each required model;
- additive major-wall timing closure and agreement of the canonical online
  Encode value with its wall category;
- complete census identity metadata, zero logical-payload changes, and exact
  JSONL occurrence/unique counts (schema 1 is accepted for the pre-verifier
  ResNet/VGG censuses; encoded-Q/P evidence requires schema 2 or newer);
- conservative classification of every periodic occurrence as a learned
  weight or a known structural concatenation materializer;
- exact encoded-Q/P verification for all 14 U-Net candidates;
- all acceptance gates in the full validation, decoder correctness, and
  isolated benchmark results;
- complete held-out sample accounting and finite native/fine-tuned metrics;
- matching checkpoint hashes across the checkpoint file, decoder correctness
  result, and isolated benchmark; and
- matching operation counters between isolated full and compressed workers.

The large census JSONL files are streamed. Their SHA-256 hashes are calculated
during validation rather than loading them into memory.

## Outputs

The default output directory is
`.tmp/results/honours/26_wpc_orion_tradeoff_synthesis` and contains:

```text
synthesis.json
report.md
model_profiles.csv
periodicity_census.csv
rotation_padding_accuracy.csv
trained_decoder_benchmark.csv
artifact_manifest.csv
online_encode_share.png
orion_periodicity_coverage.png
trained_decoder_tradeoff.png
rotation_padding_accuracy.png
```

`artifact_manifest.csv` and `synthesis.json` record the path, size, and SHA-256
hash of every input artifact. The report uses only validated inputs.

## Server execution

Run from the repository root after synchronizing the synthesis implementation:

```bash
cd ~/FHE-Honours
source .venv/bin/activate

git pull --ff-only
git rev-parse --short HEAD

python -m pytest -q \
  tests/test_wpc_tradeoff_synthesis.py \
  tests/test_extract_step1_results.py \
  tests/test_wpc_cips_trained_benchmark.py \
  tests/test_wpc_rotation_padding_training.py

OUT=.tmp/results/honours/26_wpc_orion_tradeoff_synthesis
mkdir -p "$OUT"

set -o pipefail
python tools/synthesize_wpc_orion_tradeoff.py \
  --out-dir "$OUT" \
  2>&1 | tee "$OUT/synthesis.log"
```

This stage performs no FHE forward pass and requires no GPU. It streams several
hundred megabytes of census JSONL, hashes the checkpoint and inputs, and renders
four plots; it should normally finish in minutes rather than hours. A valid run
exits with status zero, prints `"status": "ok"`, and writes the complete output
set above.

## Interpretation boundary

The expected evidence distinguishes two hypotheses:

- **Selective WPC in unchanged Orion layouts:** unsupported if the censuses
  retain zero periodic learned-weight candidates. U-Net's 14 candidates are
  structural concatenation transforms and do not explain its high online
  Encode share.
- **WPC after adopting CIPS/Rotation Padding:** supported at the trained
  encrypted decoder-stage scope when compressed/full correctness and operation
  counts match while resident storage and RSS fall. The measured latency and
  accuracy deltas are costs of that layout choice.

A complete encrypted-network comparison remains future work and requires
matched end-to-end WPC and Orion executions for U-Net and ResNet.
