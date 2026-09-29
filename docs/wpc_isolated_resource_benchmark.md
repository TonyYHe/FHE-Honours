# Isolated WPC CIPS resource benchmark

## Purpose

Stage 14 established that WPC-compressed Q/P plaintexts can execute two real
Orion `Conv2d` layers correctly. It did not establish a physical-memory or
repeated-latency result because full validation transforms and compressed
transforms coexisted in one process.

This stage provides a matched microbenchmark with two fresh worker processes:

- `full` stores and evaluates ordinary full-Q/P CIPS transforms only;
- `compressed` stores WPC representatives only, reconstructs one full
  transform for each evaluation, and releases it immediately.

The workers use the same deterministic input, weights, biases, CKKS
parameters, CIPS grouping, warmup count, and measured-forward count.

## Isolation and timing

The parent driver runs the workers sequentially. Each worker initializes its
own scheme and keys. The compressed worker is compiled with
`verify_exact_qp=False` and without a full control transform; exact Q/P
reconstruction was already established by Stages 12--14. The full worker does
not register any compressed transform.

One measured forward is:

```text
encrypted CIPS input -> Conv2d(level 2) -> Conv2d(level 1) -> level 0 output
```

Forward timing excludes scheme initialization, compilation, input encryption,
output decryption, and cleanup. The default run performs two warmups and ten
measured forwards per worker. It reports the minimum, maximum, mean, median,
p95, and sample standard deviation.

For the compressed path, the backend's per-transform counters are sampled
after every forward. The sum of `last_decompress_s` across the eight group
transforms is reported separately from transform evaluation and total forward
wall time.

## Memory measurements

Three different memory quantities are deliberately kept separate:

1. **Logical transform storage** is exact Q/P payload plus explicit WPC
   metadata and uncompressed bias plaintexts.
2. **Go runtime memory** comes from `runtime.ReadMemStats`, particularly
   `HeapInuse` after compilation.
3. **Process RSS** is sampled by the parent from `/proc/<pid>/status` on Linux.
   A `ps` fallback exists for local macOS validation.

The backend exposes `CollectRuntimeMemory(releaseOS)`. The worker calls it
between compilation, input preparation, warmup, and measurement phases. It
runs Go garbage collection and, when requested, returns unused pages to the
operating system. These calls are outside all timed forwards and make the
pre-online RSS boundary reflect live objects rather than dead compilation
allocations retained by the allocator.

RSS still includes Python, Torch, scheme keys, ciphertexts, runtime state, and
allocator pages. Therefore the logical storage compression ratio is not an
RSS compression ratio. Both are reported without assuming RSS must fall by the
logical payload amount.

## Acceptance gates

The comparison is valid only if:

- the workers have distinct process IDs and matching configurations;
- the full worker registers zero compressed transforms;
- the compressed worker registers eight compressed transforms;
- both outputs match the same Rotation-Padding clear reference;
- the isolated outputs agree within the configured tolerance;
- operation counters match per forward;
- no online weight or Python Encode occurs;
- compressed materializations are released after evaluation;
- compressed peak materialization is one transform;
- logical resident storage is reduced; and
- external RSS samples cover both measured phases.

Observed RSS reduction is a result, not an acceptance condition. This avoids
turning the hypothesis into a self-fulfilling test.

## Outputs

`tools/run_wpc_cips_isolated_benchmark.py` creates:

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

The worker JSON files retain decoded values so the parent can compare outputs
from independent encryptions. The combined comparison omits those raw values
while preserving correctness statistics.

## Server execution

```bash
cd ~/FHE-Honours
source .venv/bin/activate

python tools/build_lattigo.py
python -m pytest -q \
  tests/test_wpc_cips_benchmark.py \
  tests/test_wpc_cips_layer.py \
  tests/test_wpc_cips_baseline.py \
  tests/test_wpc_periodicity.py

(
  cd orion/backend/lattigo
  go test ./...
)

OUT=.tmp/results/honours/15_wpc_isolated_resource_benchmark/server_run1
mkdir -p "$OUT"

python tools/run_wpc_cips_isolated_benchmark.py \
  --out-dir "$OUT" \
  2>&1 | tee "$OUT/benchmark.log"
```

A successful run exits with status zero, writes `comparison.json` and
`comparison.md`, and reports every acceptance gate as true.

## Scope boundary

This stage makes defensible memory and repeated-latency claims for the
two-layer mechanism microbenchmark only. It does not establish trained-model
accuracy or whole-network performance. A subsequent functional stage now
integrates one quadratic activation/bootstrap boundary; see
`docs/wpc_cips_activation_bootstrap.md`. Stride-two CIPS reshaping,
residual/concatenation handling, higher-degree activations, and model-scale
evaluation remain separate stages.
