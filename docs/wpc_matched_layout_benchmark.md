# Stage 31: repeated matched-function layout/storage benchmark

## Purpose and scope

Compare the five Stage-30 treatments using repeated fresh processes, warmups and
balanced order. Both layouts evaluate the same checkpoint-derived decoder,
synthetic low/skip features, flattened Rotation Padding, trained Cheb7,
bootstrap, and CKKS level schedule. Storage policies are full Q/P or online
Encode for native Orion, and full Q/P, online Encode or compressed Q/P for CIPS.

This remains a decoder-stage experiment with **functional,
security-unassessed parameters** (`LogN=12`, high shape `16×16`). It does not
reproduce a full WPC network, establish a complete-network crossover, compare
dataset accuracy, or support secure-deployment claims.

## Native preparation correction

Stage 30's native online plan called the direct packer for each requested
matrix block, but the packer traversed all input/output channel pairs before
discarding entries outside that block. The measured preparation time therefore
included avoidable repeated work. Stage 30 remains correctness/resource
evidence; its single-forward timings are not inputs to this benchmark.

For the experimental native control, batch size is one and ciphertexts hold
complete spatial/multiplex planes. With `S` slots and logical spatial area
`A`, each input or output ciphertext holds `S/A` logical channels. A requested
block `(r,c)` selects precisely the output-channel range of row `r` and the
input-channel range of column `c`. This holds for gap-one convolution and
gap-two pixel-shuffled transposed-convolution input because complete multiplex
groups never straddle ciphertexts.

The plan now passes these channel pairs explicitly to the direct packers,
pruning before kernel/index traversal. It still generates and releases one
block at a time; it does **not** keep a float diagonal bank to accelerate the
online mode. The ordinary accumulator's entry-level block mask remains active.
General packer callers that omit `channel_pairs` retain their existing path.
Tests check bit-identical diagonals, partial final channel groups, gap-one and
gap-two geometry, threaded/single-worker builds and index-construction counts.
Across all blocks, selected channel-pair counts sum to one whole-layer
traversal, instead of one whole-layer traversal per block.

The preparation policy recorded in new raw workers is
`channel_pruned_block_regeneration_v1`.

## Experimental protocol

- Ten paired blocks; five sequential **fresh processes** per block (50 workers).
- Two requested warmups, followed by ten measured forwards per worker (500
  measured forwards in total). An additional untimed correctness/lifecycle
  preflight precedes warmups.
- The ten-row Williams design balances each treatment twice in each position
  and each ordered immediate-predecessor pair twice. Rows are shuffled with
  order seed zero. It is not an enumeration of all 120 permutations.
- Within each worker, setup, compilation, key generation, encryption,
  decryption/output validation, tracing and cleanup are outside forward time.
- The original fine-tuned checkpoint, exact feature digest, configuration
  bytes, backend hash, runtime parameters, software versions, operation counts
  and environment must agree across the matched experiment.
- Each worker is guarded by sampled RSS (8,192 MiB default) and a 1,800-second
  timeout. The watchdog is not an OS-enforced allocation ceiling. A failed or
  unobservable worker invalidates the run; completed evidence is retained.
- Exact Q/P reconstruction is checked offline for the compressed treatment.
  Temporary full weight transforms must be released, and online Encode counts
  must equal the learned-transform count times the measured-forward count.

The primary latency statistic is the **mean of process-block forward
medians**. Paired ratios divide treatment medians within each block before
averaging. Percentile confidence intervals use 10,000 resamples of whole paired
blocks, not individual forwards. These are descriptive, pointwise intervals
for this run; they are not multiple-comparison-adjusted or evidence of
reproducibility across servers/sessions.

Forward accounting uses mean seconds and means of within-forward shares for
preparation, Encode, decompression, LT evaluation, activation, bootstrap and
residual. Each raw forward must close; no sum of component medians is presented
as an additive decomposition of median wall time. Native Encode includes
allocation/binding call wall time, whereas CIPS Encode is the narrower backend
Encode timer. Do not equate these timers with historical Step-1 categories.
The residual includes concat, bias/rescale, wrapper and other call overhead;
individual cryptographic addition/multiplication counts are not measured.

Logical resident coefficient/recipe/metadata/bias bytes and process RSS are
reported separately. RSS maxima are sampled during measured phases and
averaged across blocks. Validation-only full Q/P controls are not retained by
compressed workers. Every measured decrypted output is retained outside timed
forwards, so its finite values/error and every per-forward operation count can
be recomputed independently after transfer. Retaining these small output
arrays contributes to process RSS; all treatments use the same policy.

## Run on the server

Commit and push the implementation locally first. Then run:

```bash
(
  set -e
  cd ~/FHE-Honours
  git pull --ff-only
  test -f tools/run_wpc_matched_layout_server.sh

  TASK_RUN=.tmp/results/honours/31_wpc_matched_layout_benchmark/server_run1
  mkdir -p .tmp/results/honours/31_wpc_matched_layout_benchmark
  test ! -e "$TASK_RUN" && test ! -e "$TASK_RUN.launch.log"

  nohup bash tools/run_wpc_matched_layout_server.sh "$TASK_RUN" \
    > "$TASK_RUN.launch.log" 2>&1 < /dev/null &
  echo $! | tee "$TASK_RUN.pid"
)
```

The wrapper builds Lattigo, runs the regression and small functional-FHE tests,
then starts the benchmark. Existing evidence is never overwritten. Do not edit
the implementation, binary, configuration or checkpoint while it runs. It is
a long-running 50-worker job; use the recorded events rather than extrapolating
from Stage 30's unpruned single-forward timings.

Completion checks (no shell variable from a previous login is required):

```bash
cd ~/FHE-Honours
TASK_RUN=.tmp/results/honours/31_wpc_matched_layout_benchmark/server_run1
ps -p "$(cat "$TASK_RUN.pid")" -o pid,etime,stat,cmd
tail -n 30 "$TASK_RUN.launch.log"
test ! -f "$TASK_RUN.exit_status" || cat "$TASK_RUN.exit_status"
```

Success requires exit status `0`, 50 `worker_complete` events and the final
`MATCHED LAYOUT BENCHMARK SUCCESS` line. Independently review before using any
numbers:

```bash
.venv/bin/python tools/run_wpc_matched_layout_benchmark.py \
  --review-dir .tmp/results/honours/31_wpc_matched_layout_benchmark/server_run1
```

After transferring the complete directory, the same `--review-dir` command
works locally without FHE execution or the server checkpoint/binary bytes.
Those identities are recorded as hashes, not authentication signatures.

## Artifacts and implementation changes

`comparison.json` and `comparison.md` contain checked statistics and scope.
`requested_run.json` freezes the schedule/request, and
`configuration_input.json` archives exact configuration bytes. Each of ten
`block_NNN` directories contains five raw worker JSON/log/phase files, five RSS
JSON files and a recomputable `block.json`: 210 hashed block artifacts total.
Failure writes `comparison.failed.json`; no failed run is promoted to results.
No new performance results are claimed until the server evidence is reviewed.

Changed/added code:

- `orion/core/packing.py`: explicit opt-in channel-pair pruning; unchanged
  default direct-packing behavior.
- `orion/experimental/wpc_orion_layout_control.py`: aligned block-to-channel
  selection, without resident diagonal recipes.
- `tools/run_wpc_cips_trained_isolated_worker.py`: optional retention of every
  measured output and operation-counter delta outside timed forwards.
- `orion/experimental/wpc_layout_gate.py`: explicit repeated-run validation
  mode; the default Stage-30 one-forward/zero-warmup gate stays strict.
- `orion/experimental/wpc_matched_layout_benchmark.py`: order design, raw sample
  validation, accounting and matched process-block inference.
- `tools/run_wpc_matched_layout_benchmark.py`: guarded orchestration, frozen
  hashes, manifest, report and offline raw-artifact recomputation.
- `tools/run_wpc_matched_layout_server.sh`: preflight, fresh-path protection and
  completion status compatible with `nohup`.
- `tests/test_wpc_orion_layout_control.py` and
  `tests/test_wpc_matched_layout_benchmark.py`: pruning/counter/corruption/order
  tests and a small five-worker synthetic-checkpoint functional smoke test.

No Go ABI changes, checkpoint retraining or historical result rewrites are
required. Larger/secure configurations and whole-network experiments remain
separate future stages.

## Local verification (8 October 2026)

- WPC suite: **319 passed, 1 skipped** (the actual Linux-procfs test on macOS),
  with process-statistics access for the guarded five-worker smoke test. That
  test uses synthetic checkpoint weights, `LogN=9`, `4×4` features, one warmup
  and two forwards; it is functional verification, not research timing data.
- All Go backend tests passed; shell syntax, Python compilation and diff
  whitespace checks passed.
- The transferred Stage-30 archive still passes read-only numerical/hash
  review; changed source hashes are reported rather than rewriting the archive.
- Broader default-packer/Python-backend checks: 78 passed, one unrelated
  `test_concat_fusion_uses_unified_groups_without_dense_branch_pack` failure.
  Loading the pre-change packing module from `HEAD` reproduces the same failure.
  That existing concat-fusion issue was not changed in this stage. These checks
  use the manual compilation policy to avoid the existing macOS automatic
  memory-budget fallback.

Server-scale evidence has not yet been generated for this stage.
