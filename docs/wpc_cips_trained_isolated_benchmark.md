# Isolated fine-tuned WPC decoder benchmark

For the subsequent three-way online-Encode comparison, balanced fresh-process
trials, and timed-tracing repair, see
[Stage 28](wpc_online_encode_benchmark.md). Historical Stage-25 evidence is
preserved; it compares two **offline-preencoded** CIPS storage policies.

## Purpose

This stage measures the resource trade-off of the exact fine-tuned decoder
graph validated by the preceding correctness gate:

```text
low-resolution feature -> up1 --+
                                  cat1 -> dec1a -> trained Cheb7
checkpoint skip1 ----------------+       -> bootstrap -> dec1b
```

The ordinary full-Q/P and WPC-compressed weight transforms run in separate
fresh processes. Each worker loads the same checkpoint and uses the same
synthetic features, CKKS parameters, packing, warmup count, measured-forward
count, and correctness tolerance. Unlike the single-run correctness gate, no
worker contains both weight-storage representations.

## Timing boundary

One measured forward contains:

1. checkpoint `up1` transposed convolution;
2. the `cat1` CIPS permutation;
3. checkpoint `dec1a` Rotation-Padding convolution;
4. the exact checkpoint degree-seven Chebyshev activation;
5. a real four-ciphertext bootstrap; and
6. checkpoint `dec1b` Rotation-Padding convolution.

Compilation, rotation-key construction, input Encode/encryption, output
decryption, and cleanup are excluded. Warmups execute the complete encrypted
graph but are excluded from the measured samples. The report separates online
compressed-weight decompression from total forward wall time and also reports
activation and bootstrap time.

## Memory boundary

The benchmark keeps three quantities distinct:

- exact logical resident plaintext storage, including learned transform Q/P
  payloads, bias plaintexts, WPC metadata, and the common full-Q/P concat
  permutation;
- Go heap statistics after compilation and garbage collection; and
- external process RSS sampled by the parent from `/proc/<pid>/status` during
  compilation and measured execution.

RSS additionally contains Python, Torch, cryptographic keys, ciphertexts,
bootstrap state, runtime metadata, and allocator pages. Logical storage
compression must therefore not be reported as RSS compression.

## Correctness and acceptance

The combined result is valid only when:

- both workers complete successfully in distinct processes;
- checkpoint hashes, graph configuration, inputs, and seeds match;
- the full worker registers zero compressed transforms;
- the compressed worker registers exactly 56 compressed learned transforms;
- both workers match the same independent Rotation-Padding clear result;
- isolated encrypted outputs agree within the configured tolerance;
- operation counters match per forward;
- every measured forward executes the trained Cheb7/bootstrap bridge;
- online Python and weight Encode counts are zero;
- compressed materialization is released and peaks at one transform;
- logical resident plaintext storage is reduced; and
- external RSS sampling covers the measured phase of both workers.

Observed latency and RSS changes are results rather than acceptance gates.
This prevents the tested hypothesis from determining validity.

## Outputs

The parent writes:

```text
comparison.json
comparison.md
full.worker.json
full.worker.log
full.phase.json
compressed.worker.json
compressed.worker.log
compressed.phase.json
```

Raw decoded values remain in the worker JSON files for independent-output
comparison and are omitted from the combined result.

## Server execution

```bash
cd ~/FHE-Honours
source .venv/bin/activate

git pull --ff-only
python tools/build_lattigo.py

python -m pytest -q \
  tests/test_wpc_cips_trained_benchmark.py \
  tests/test_wpc_cips_benchmark.py \
  tests/test_wpc_cips_checkpoint.py \
  tests/test_wpc_cips_layer.py \
  tests/test_wpc_cips_baseline.py \
  tests/test_wpc_periodicity.py

OUT=.tmp/results/honours/25_wpc_finetuned_decoder_isolated_benchmark/server_run1
CHECKPOINT=checkpoints/wpc_rotation_padding_covid19_cheb7_audited_restart/rotation_padding_best.pt
mkdir -p "$OUT"

python tools/run_wpc_cips_trained_isolated_benchmark.py \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT" \
  --warmup-runs 2 \
  --forward-runs 10 \
  --rss-sample-ms 5 \
  2>&1 | tee "$OUT/benchmark.log"
```

A valid Linux run exits with status zero, reports `status: "ok"`, and sets
every field under `acceptance` to `true`.

## Scope

This is a trained decoder-stage latency and memory experiment with synthetic
internal features. It does not measure complete encrypted U-Net latency or
dataset accuracy. The full validation result supplies accuracy evidence; this
benchmark isolates the resource consequence of compressed Q/P storage for a
checkpoint-derived FHE subgraph.
