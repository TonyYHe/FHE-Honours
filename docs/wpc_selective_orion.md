# Stage 32: selective WPC storage in unchanged Orion transforms

## What this stage adds

Stage 31's repeated five-treatment matched-layout benchmark is already
implemented; its server results are still pending. This independent stage adds
the previously missing **selective storage path**, not another CIPS layout.
Both stages can be evaluated independently. Do not rebuild the shared library
while either experiment is running on the server.

`ORION_WPC_SELECTIVE_POLICY` is `off` by default. `online` and `hybrid` opt into
the same new per-diagonal backend, with the original Orion diagonal indices,
full-transform BSGS split, effective pre-rotations, Q/P levels and scale.

- Offline: scan the original float32 messages for exact proper power-of-two
  periods. In `hybrid`, Embed only those candidates, check every Q/P limb's
  encoded repetition, require exact reconstruction, and retain one evaluation
  period per limb. All-zero diagonals stay on the fallback path to preserve the
  original transform/operation set. No fallback slot vectors are retained.
- Online: the existing Python recipe regenerates the same full payload. A
  digest check rejects stale weights/indices. Eligible Q/P arrays are copied
  from stored periods; every other diagonal receives an ordinary CKKS Embed
  with the **full** transform's pre-rotation. Encoding a standalone smaller
  transform would change its BSGS parameters and is deliberately avoided.
- Eviction: release full Q/P arrays but retain the empty shell and verified
  periods. Layer-, transform- and group-granularity dense caches are supported.
  Transform deletion and scheme deletion clear the new registry.

The normal cache path is unchanged when the flag is off. Opt-in requires real
Lattigo FHE with a Standard ring and full-slot diagonals, `io-mode none`,
single-slot caching and one Encode worker. Clear
execution, stale binaries and changed payloads fail closed. Whole-model
provider/executor coverage has **not** been validated; the initial model
launcher deliberately uses dense mode only.

## Measurements and limitations

The backend reports logical full and eligible Q/P bytes, resident compressed
bytes, explicit logical metadata, actual per-diagonal offline/online Embed
counts, and separate preparation, CKKS Embed and copy-decompression timers.
The full raw recipe identity is bound to the in-memory transform and its CKKS
parameters; this is **not** a persistent on-disk compressed artifact format.

The gate evaluates ordinary zero-padded Orion Conv2d blocks, a mixed
periodic/nonperiodic/zero control and an all-periodic control. It verifies exact
Q/P equality against an ordinary full transform, decrypted outputs against an
independent clear diagonal oracle, unchanged rotation/conjugation counters,
expected fallback Encode counts and release after repeated evaluation. Raw
slot recipes, source values and decrypted outputs are retained for review.
Synthetic controls must not be described as newly discovered model candidates.

The gate uses `LogN=10` functional parameters and three repeated evaluations.
It is correctness/accounting evidence, **not a performance benchmark**. Its
Encode percentage has an LT preparation-plus-evaluation denominator, not a
complete encrypted model forward. No security or model-accuracy claim is made.
Validation full-Q/P controls coexist in this gate, so it does not measure
deployment RSS savings.

The optional model diagnostic launches two fresh processes in fixed order,
with one warmup and three measured forwards per policy. The pure backend
Encode share is `100 * sum(Embed times) / he_forward_s`; baseline Encode-time
coverage is time spent embedding eligible diagonals divided by all registered
diagonal Embed time. Coverage is measured only in the `online` policy; it is
not misreported as zero after the eligible Encodes have been eliminated.
Snapshots and model outputs are collected outside the forward timer. Raw
counter deltas, output tolerances, identity digests and operation equality are
checked independently. The diagnostic rejects a transform that is omitted or
materialized more than once in a forward rather than silently assuming its
coverage is complete.

These timers exclude Python recipe building, bias Encode and FFI overhead.
They cover only registered dense-cache transforms, not every encoding call in
the model. The legacy layer-cache category still includes broad materialization
work, including reconstruction. The historical Step-1 launcher rejects an
active selective policy; its reports must not relabel decompression as Encode.
Both selective policies use the same serial per-diagonal Embed implementation,
not the historical batched baseline. A fixed-order diagnostic does not provide
confidence intervals or a latency ranking. Logical metadata excludes allocator
overhead, and per-transform peaks are not summed into a fictitious global peak.

## First server command

Commit and push the implementation locally before running this on the server:

```bash
(
  set -e
  cd ~/FHE-Honours
  source .venv/bin/activate
  git pull --ff-only
  test -f tools/run_wpc_selective_orion_server.sh

  TASK_RUN=.tmp/results/honours/32_wpc_selective_orion/server_run1
  mkdir -p .tmp/results/honours/32_wpc_selective_orion
  test ! -e "$TASK_RUN" && test ! -e "$TASK_RUN.launch.log"
  nohup bash tools/run_wpc_selective_orion_server.sh "$TASK_RUN" \
    > "$TASK_RUN.launch.log" 2>&1 < /dev/null &
  echo $! | tee "$TASK_RUN.pid"
)
```

This builds the library, runs regression/Go tests, executes the bounded gate
and reviews its output. No dataset, GPU or trained checkpoint is required.
The gate is a short job relative to the whole-model runs; build/test duration
depends on the server and dependency caches.

Completion requires `<run>.exit_status` equal to `0` and
`SELECTIVE ORION GATE SUCCESS` in `<run>.launch.log`. The wrapper refuses
existing evidence paths. After transferring the complete directory, review
without FHE execution:

```bash
.venv/bin/python tools/run_wpc_selective_orion.py \
  --review-dir .tmp/results/honours/32_wpc_selective_orion/server_run1
```

The reviewer recomputes slot-payload hashes, the clear oracle, errors, byte
accounting, counter expectations and timing arithmetic. Exact encoded-Q/P
equality is a recorded backend gate: coefficient arrays are not bundled, and
hashes are content identities, not authentication.

Only after the gate passes, an optional **long-running** ResNet20 dense
diagnostic can be launched using a fresh name:

```bash
TASK_RUN=.tmp/results/honours/32_wpc_selective_orion/resnet20_run1
nohup bash tools/run_wpc_selective_orion_server.sh "$TASK_RUN" resnet20_cifar10 \
  > "$TASK_RUN.launch.log" 2>&1 < /dev/null &
echo $! | tee "$TASK_RUN.pid"
```

This produces `model_diagnostic/online.json`, `hybrid.json`, their logs and a
recomputable `comparison.json`. It runs eight complete encrypted forwards
including warmups, plus two compiles, so it is not the first acceptance task.
No whole-model experiment has been executed locally in this stage. With zero
eligible learned weights, do not expect learned-weight Encode savings. The
earlier U-Net structural candidates still require verified provider integration
or a separately scoped structural experiment.

## Changed files

- `orion/backend/lattigo/wpc_selective.go` and tests: exact mixed storage,
  full-BSGS Embed, payload identities, counters and cleanup APIs.
- `orion/backend/lattigo/{bindings.py,lineartransform.go,scheme.go}`: optional
  bindings and registry lifecycle hooks.
- `orion/experimental/wpc_selective_orion.py`: opt-in adapter, bounded offline
  preparation, ordinary-cache materialization/release and audit snapshots.
- `orion/backend/python/lt_evaluator.py`, `orion/nn/linear.py`: opt-in cache
  integration and cleanup; defaults preserved.
- `tools/run_lattigo_e2e_compare.py`: optional outside-forward snapshots and
  a guard against historical Encode-category mislabelling.
- `tools/run_step1_online_encode_profile.py`,
  `tools/run_wpc_periodicity_census.py`: prevent contamination of historical
  profiling and clear census runs.
- New gate/model launchers, server wrapper and Python regression tests.

## Local verification

357 WPC/Step-1 regression tests passed. Two tests were skipped on this macOS
sandbox: the actual Linux-procfs check and Stage 31's RSS-guarded five-worker
smoke. Neither skip is a successful server experiment. All Go backend tests,
Python compilation and wrapper shell syntax checks passed.

The new tests exercised exact mixed-diagonal reconstruction at multiple BSGS
splits and with/without P limbs, all three cache granularities, stale payload
rejection, repeated release and corrupted-artifact rejection. A tiny actual
Orion model (`1×1×4×4` input, one ordinary Conv2d, `LogN=9`) passed compilation
and two encrypted forwards under both policies, including the new E2E snapshot
hooks and raw model comparison. The standalone three-case gate and its
read-only JSON reviewer also passed. These are functional tests only; no
whole-model research run or remote server job was executed.

This stage does not complete WPC paper reproduction, secure-parameter testing,
the U-Net/ResNet layout crossover, or an adaptive-layout optimization.
