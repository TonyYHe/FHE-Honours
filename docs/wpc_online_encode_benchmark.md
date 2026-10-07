# Stage 28: matched three-way online-Encode benchmark

## Research question

Does compressed Q/P storage replace online weight Encode with less expensive
copy-based decompression, and what resident/online memory does each policy use?
This is a same-CIPS **storage-policy** experiment, not an Orion-versus-WPC
layout comparison. Historical Stage-25 workers preencoded both representations
offline, so that experiment could not measure removal of online Encode.

## Treatments

All workers load the same checkpoint bytes and use the same exact synthetic
features, CIPS geometry, BSGS ratio, CKKS parameters, activation, bootstrap,
warmups, and number of forwards. The graph remains:

`up1 + skip1 -> concat -> dec1a -> trained Cheb7 -> bootstrap -> dec1b`.

| Mode | Offline learned-weight representation | Online learned-transform lifecycle |
|---|---|---|
| `online_encode` | Empty transformation shells and unencoded float32 slot-period recipes | Expand recipe, allocate Q/P, call Lattigo Encode, evaluate, release |
| `full` | Complete encoded Q/P plaintexts | Evaluate resident transform |
| `compressed` | One encoded Q/P evaluation period per active limb | Copy-decompress, evaluate, release |

The recipe baseline already benefits from CIPS slot periodicity: it stores
one float32 slot period, not repeated full slot vectors. It is not the original
Orion layer-cache implementation. Rotation keys are prepared offline from empty
shells; input, bias, and common concat plaintexts are preencoded in all modes.

### Timing and counting

- `online_prepare_s` is Go recipe expansion and full-transform allocation.
- `online_encode_s` is the actual `lintrans.Encode` call, including its internal
  slot encoding/NTT work, not the Python wrapper or allocation interval.
- `online_materialization_pct_of_forward` includes preparation plus Encode.
- `decompression_s` is the compressed backend's coefficient-copy interval.
- Forward wall time also includes evaluation, accumulation, rescaling, bias,
  trained activation, bootstrap, and temporary-materialization release.
- Warmups, key preparation, input encryption, output checking, and lifecycle
  tracing are excluded from measured forward times.

These boundaries differ from historical Step-1 `lt_layer_cache_encode_s`, which
also included Python materialization/payload/GC. They must not be equated or
subtracted across the different workloads.

New backend counters increment **after successful calls** to `lintrans.Encode`
in ordinary, compressed-generation, and online-recipe transform paths. A count
is one transform-encoding invocation, not one diagonal. Input/bias and bootstrap
encoder internals are outside these counters. The common concat's eight
ordinary offline calls are recorded separately from the 56 learned transforms:

| Mode | Ordinary offline calls | Compressed offline calls | Measured online calls per forward |
|---|---:|---:|---:|
| Online Encode | 8 | 0 | 56 |
| Full Q/P | 64 | 0 | 0 |
| Compressed Q/P | 8 | 56 | 0 |

Counters reset after the untimed preflight and warmups. Every measured output
is independently decrypted and checked outside its timer. Each compressed
worker first performs a separate traced lifecycle/correctness preflight; timed
forwards disable `record_sequence`, removing the historical 112 extra global
statistics queries. Stats extraction occurs between, not inside, forward timers.

## Independent trials and uncertainty

The default is six matched process blocks, one for each permutation of the
three treatment orders. Each block launches three new processes sequentially;
each process executes two warmups and ten measured forwards. More trials must
be a multiple of six. Order randomization is reproducible via `--order-seed`.

Each block contributes its per-process median forward and median within-forward
shares. The summary reports means, medians, standard deviations, and percentile
bootstrap 95% confidence intervals for means across **whole process blocks**.
Paired latency ratios are computed within each block before aggregation. Ten
forwards in one process are not treated as ten independent experimental trials.
Six blocks are a starting design, not a guarantee of narrow confidence bounds.

`--smoke --trial-blocks 1` performs one correctness-only block and explicitly
disables performance claims and confidence intervals.

## Memory and validation

Logical resident storage includes learned Q/P or recipe data, explicit logical
metadata, resident biases, and common concat Q/P. Recipes count float32 payload
bytes; metadata excludes Go allocator/map overhead. Model objects and Python
runtime memory remain in RSS and are not included in logical plaintext counts.

Online Encode and decompression each logically materialize at most one complete
learned transform; registration leaves no learned full-Q/P arrays resident.
The backend records current/peak payload and transform counts and clears state
on transform/scheme deletion. RSS is sampled externally during measured
forwards; it is a **sampled maximum**, not a guaranteed allocation peak.

The comparison independently checks raw array lengths, finite values, timing
closure, recomputed shares, successful Encode invocation counts, all measured
output errors, configuration/feature/checkpoint identity, operation-count
equality, offline counts, logical-byte accounting, registry sizes, peak
materialization, release, and measured-phase RSS coverage. Neither a latency
improvement nor an RSS reduction is required for experimental validity.

## Scope and security

The worker retains `LogN=10`, 8x8 decoder outputs and synthetic 4x4/8x8 internal
features. These are **small insecure functional-test parameters**, not a secure
deployment benchmark. A deployment-performance claim requires a separately
security-validated residual AND bootstrapping parameter configuration, followed
by revalidation and rerunning the experiment. Increasing LogN alone is not a
security assessment. This stage does not claim whole-model U-Net accuracy,
complete encrypted model latency, or the model-family layout crossover.

## Implementation

- `orion/backend/lattigo/wpc_online_encode.go`: recipe registration, real online
  Encode/evaluate/release, observed Encode counters, logical materialization.
- `orion/backend/lattigo/bindings.py`: optional ABI bindings, rebuild required.
- `orion/experimental/wpc_cips_layer.py`: third storage mode shared by Conv2d,
  stride-two and ConvTranspose2d plans; opt-in timed tracing control.
- `tools/run_wpc_cips_trained_isolated_worker.py`: schema-2 workers, preflight,
  per-forward correctness/counters and separated preparation/Encode timings.
- `orion/experimental/wpc_online_encode_benchmark.py`: fail-closed validation,
  balanced schedules and matched-process-block uncertainty.
- `tools/run_wpc_online_encode_benchmark.py`: isolated process orchestration,
  JSON/raw artifacts, process-block CSV and concise Markdown comparison.
- `tools/run_wpc_online_encode_server.sh`: build/test gates and exit-status file.

Historical results and schemas are not rewritten. Stage-27 synthesis continues
to read the archived Stage-25 evidence; this new comparison is separate.

## Local correctness verification (2026-10-07)

- 214 targeted Python tests passed, including Step-1 extraction, evidence
  validation, historical WPC regressions, and the new three-way gates.
- `go test ./... -count=3` passed, including exact expanded-recipe versus
  ordinary Q/P encoding, invalid-recipe rejection, observed Encode counters,
  and deferred release after an evaluation panic.
- Real-FHE Conv2d, stride-two Conv2d, and ConvTranspose2d tests passed two
  consecutive forwards in each mode, with equal outputs/operation counts and
  no retained online full-Q/P payload.
- A three-process decoder smoke passed all acceptance gates and independent
  artifact/source-hash checks. Online mode recorded 56 successful weight
  Encode invocations; the preencoded modes recorded zero. Measured lifecycle
  trace counts were zero in every mode and measured-phase RSS samples existed.

The final smoke artifacts are in
`.tmp/results/honours/28_wpc_online_encode_benchmark/local_correctness_final_20261007_v2/`.
This smoke used the available original source checkpoint, **not** the
server-only audited fine-tuned checkpoint, with zero warmups and one measured
forward per process. Its timings are diagnostics only; it generated no
confidence intervals. The six-block server experiment has not been run.

## Server execution

Commit/push the implementation locally, then run on the server:

```bash
cd ~/FHE-Honours
git pull --ff-only

TASK_RUN=.tmp/results/honours/28_wpc_online_encode_benchmark/server_run1
mkdir -p .tmp/results/honours/28_wpc_online_encode_benchmark
nohup bash tools/run_wpc_online_encode_server.sh "$TASK_RUN" \
  > "$TASK_RUN.launch.log" 2>&1 < /dev/null &
echo $! | tee "$TASK_RUN.pid"
```

The wrapper uses `.venv/bin/python`, rebuilds Lattigo, runs Python/Go tests, and
executes all 18 fresh worker processes. It does not need a GPU or the dataset
NPZ. It needs the audited fine-tuned `rotation_padding_best.pt` checkpoint.
Do not estimate total runtime by multiplying the historical median alone:
imports, key/bootstrap construction, compilation, preflights, and online Encode
add work. Expect a longer run than the original two-process benchmark.

Check success using explicit paths, which remain valid after reconnecting:

```bash
cat .tmp/results/honours/28_wpc_online_encode_benchmark/server_run1.exit_status
tail -n 30 .tmp/results/honours/28_wpc_online_encode_benchmark/server_run1.launch.log
```

Success requires exit status `0`, `THREE-WAY BENCHMARK COMPLETE`, and
`comparison.json` with `status=ok` and all acceptance gates true. A missing exit
file means the wrapper has not recorded completion; process absence alone is
not proof of success. Use a fresh run name for retries; completed/partial
evidence is never silently overwritten.

Outputs include `comparison.json`, `comparison.md`, `process_blocks.csv`, and
one directory per block with all three raw worker JSON/log/phase/RSS files.
The JSON records checkpoint/feature and loaded backend-binary identity,
repository status/commit, source hashes, raw worker/RSS hashes, orders, and the
uncertainty settings. Source changes or backend/checkpoint/feature changes
during the experiment cause rejection rather than a mixed-provenance report.
