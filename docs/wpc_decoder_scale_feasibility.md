# Stage 29: configurable decoder correctness and resource gate

## Purpose and boundary

Stage 28 measured three storage policies inside the same CIPS decoder using
LogN=10 and 8×8 outputs. Stage 29 removes those fixed geometry assumptions and
checks a larger configuration before launching repeated timing trials.

This is the **prerequisite**, not the completed Orion-versus-WPC layout
comparison. No native Orion-layout treatment, full encrypted U-Net/ResNet,
dataset-feature extraction, or security estimate is implemented by this gate.
The existing Rotation-Padding semantics and checkpoint-derived decoder graph
are unchanged:

`up1 + skip1 → concat → dec1a → trained Cheb7 → bootstrap → dec1b`.

The next implemented correctness gate is described in
[`wpc_orion_layout_gate.md`](wpc_orion_layout_gate.md). It adds an explicit
native-layout matched-function control; Stage 29 itself remains CIPS-only.

## Changes

- `orion/experimental/wpc_decoder_geometry.py` independently derives input/output
  group counts, learned-transform counts, concat counts, bootstrap group counts,
  and the residual-level schedule. Height/width must be powers of two ≥4; one
  spatial plane must fit in one ciphertext. Spatial tiling is not implemented.
- `--ckks-config` accepts an explicit JSON configuration. Unknown/ignored keys,
  non-Standard rings, insufficient depth, conflicting `--logn`, and unsupported
  runtime settings are rejected. LogQ, LogP, LogScale, H and bootstrap LogP are
  configurable. Bootstrap parameters not exposed by this backend remain the
  bundled Lattigo defaults. At least nine residual Q primes are required; layer
  levels are derived from the actual maximum residual level rather than 8/6/8.
- The worker now writes schema **3**, including the geometry contract, exact
  config-file SHA-256, actual backend parameter manifest, and bootstrap compile
  metadata. The three-way comparison writes schema **2**. Historical schema-2
  workers and archived schema-1 comparisons retain their fixed 56-transform
  validation contract; old evidence is not rewritten.
  Exact original config-file bytes are archived as `configuration_input.json`
  with a hash/size, so evidence can be reviewed after transfer without relying
  on the original server path. Config changes between loading and archiving or
  between controller and workers cause failure.
- `GetWPCParameterManifest` exports **public metadata only**: actual residual
  and bootstrap Q/P primes, LogQP, ring size/type, default scale, secret/error
  distribution parameters, and sparse encapsulation parameters. No secret-key
  coefficients are exported. Actual values are independently cross-checked
  against the request and across treatments/blocks.
- `--verify-exact-qp` creates one temporary ordinary control for each compressed
  transform during compilation, reconstructs and compares every Q/P diagonal,
  then deletes the control. These extra ordinary Encode calls are reported as
  validation work, separately from the compressed offline weight Encodes.
  Validation is outside forward timers; its transient compile memory is included
  in the all-phase feasibility watchdog. Timed materialization peaks are reset
  after the correctness preflight and warmups.
- The isolated-process sampler adds optional sampled-RSS and wall-clock guards.
  It stops the worker's newly created process group on a violation and preserves
  the worker log and `.rss.json`, even when no worker result was produced. These
  are watchdogs, **not an OS-enforced allocation ceiling or a memory reservation**.
  Missing or lost RSS access fails closed rather than silently disabling the guard.
  If the leader's RSS becomes unreadable on Linux, the sampler probes
  `/proc/<pid>/task/<tid>/stat` and `status`. A surviving thread's RSS measures
  the shared address space; observations are combined by **maximum, not sum**.
  Only when all examined threads have `PF_EXITING` or a terminal state, with no
  unreadable or newly discovered threads, may it allow **up to 5 seconds to
  reap the process**, capped by the remaining worker deadline. Otherwise the
  original 100 ms exit check applies and persistent monitoring loss is rejected.
  The actual exit code and independent worker validation are still required.
  `.rss.json` records sample counts by source, unavailable samples, the last
  thread-state probe, and both wait limits. Missing samples are not zero-byte
  observations and do not count as valid samples.
- `tools/run_wpc_decoder_feasibility.py` runs three fresh processes, zero warmups,
  and one measured diagnostic forward per mode. It independently validates
  outputs, operation counts, lifecycle release, exact compressed Q/P, and
  resource budgets. It emits no confidence intervals or performance claim.
- The server wrapper rebuilds the shared library, runs Python/Go gates, records
  an exit status, and refuses existing output evidence. Neither the dataset NPZ
  nor a GPU is needed. The audited fine-tuned checkpoint is required on server.

## Geometry and security

For slots S=N/2 and high-resolution area A=H×W, channel capacities are
`c_high=S/A` and `c_low=4S/A`. Define
`g_high(C)=ceil(C/c_high)` and `g_low(C)=ceil(C/c_low)`. Expected learned transforms:

- up1: `g_low(64) × g_high(32)`;
- dec1a: `g_high(64) × g_high(32)`;
- dec1b: `g_high(32)²`.

The concat has `2 × g_high(32)` transforms, and the bootstrap handles
`g_high(32)` ciphertext groups. These formulas are checked against each plan's
reported transform counts, not used as replacements for observed counters.

The supplied `configs/wpc_decoder_scale_functional.json` is **not security
validated**. Its LogN=12, 16×16 smoke increases ring degree and spatial area by
4× over Stage 28, while retaining 56 learned transforms and four bootstrap
groups. It is an intermediate engineering gate, not realistic deployment
performance. A LogN=11, 8×8 correctness case instead has 14 learned transforms
and two bootstrap groups, testing that the old fixed assertions are gone.

Increasing ring degree is not a security proof. The runtime manifest deliberately
sets `security_assessed=false`. Before a secure-performance experiment, review
the **actual residual, full bootstrap, and sparse encapsulation parameters**
with the mentor and record the estimator/version, attack model, target security,
and precision requirements. All current reports explicitly retain an unassessed
security scope, even if a user supplies a large ring.

## First server command

Commit and push these implementation files locally, then on the server:

```bash
cd ~/FHE-Honours
git pull --ff-only

TASK_RUN=.tmp/results/honours/29_wpc_decoder_scale_feasibility/server_run1
mkdir -p .tmp/results/honours/29_wpc_decoder_scale_feasibility
nohup bash tools/run_wpc_decoder_feasibility_server.sh "$TASK_RUN" \
  > "$TASK_RUN.launch.log" 2>&1 < /dev/null &
echo $! | tee "$TASK_RUN.pid"
```

The default worker watchdog is **8 GiB sampled RSS and 30 minutes per worker**.
The Linux admission check refuses a budget exceeding 80% of current
`MemAvailable`; it cannot reserve those pages or prevent interference from
other jobs. The wrapper's third argument can override the MiB budget. Use a
fresh run name for retries and leave partial evidence intact.

Check completion after reconnecting, without relying on shell variables:

```bash
cd ~/FHE-Honours
cat .tmp/results/honours/29_wpc_decoder_scale_feasibility/server_run1.exit_status
tail -n 30 .tmp/results/honours/29_wpc_decoder_scale_feasibility/server_run1.launch.log
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('.tmp/results/honours/29_wpc_decoder_scale_feasibility/server_run1/feasibility.json')
d = json.loads(p.read_text())
assert d['status'] == 'ok' and d['acceptance']['valid'] is True
print(json.dumps(d['resource_observations'], indent=2))
print('FEASIBILITY VALID — security and layout comparison remain unestablished')
PY
```

Success requires exit status 0, `DECODER FEASIBILITY COMPLETE`, and valid
`feasibility.json`. Missing exit status is not success. A watchdog stop produces
a failed result and preserves partial logs/samples; do not use partial runs for
performance claims. Do not automatically launch six timing blocks after this
gate: inspect the actual parameters, correctness and available memory first.

### Server-run1 shutdown race and retry (2026-10-07)

The user-provided server diagnostic reported that the first `online_encode`
worker exited **0**, had status `ok`, passed every worker acceptance gate, and
had maximum FHE error `6.151518958802393e-7` and clear-oracle delta
`4.285634347400702e-15`. Its sampled all-phase RSS peak was `2184.7890625 MiB`,
below the 8192 MiB budget. The controller nevertheless rejected it because RSS
became unavailable just before the process exit code was visible. Neither the
full nor compressed treatment ran, so **server_run1 is incomplete evidence**,
not an accepted three-way feasibility result. These values come from the
supplied diagnostic; its raw server artifacts have not been inspected locally.

The first repair checked for an exit for up to 100 ms before treating RSS loss
as a live-worker monitoring failure. Regression tests covered successful and failed
exits observed through both `poll()` and `wait()`, actual monitoring loss while
the worker remains alive, and the existing RSS/time budget failures. The memory
budget, time budget, FHE tolerance and other acceptance gates are unchanged.
No old JSON is edited or promoted to valid.
Local repair verification passed **127 tests**: geometry/config/watchdog,
online-Encode and trained-decoder comparison, synthesis compatibility, and
three small real-FHE storage-policy integration cases. Python compilation,
server-wrapper shell syntax, and diff whitespace checks also passed. No remote
job or larger local profiling run was launched for this repair.

That repair was run as `server_run2`; its result below supersedes the earlier
retry instructions. Keep both failed runs and their sibling files intact.

### Server-run2: Linux thread-aware shutdown handling (2026-10-07)

The supplied diagnostic records commit
`d107156e93f06313770ba039f7b6c24c543826d2`. The online worker again had status
`ok`, no failed gates, and exit code **0**, but the parent rejected it after its
single unavailable RSS observation. There were 22,126 valid RSS observations,
elapsed time `121.57028893800452 s`, and sampled peak `2186.61328125 MiB`.
The recorded 100 ms guard confirms the first repair was present; that wait was
insufficient. No full or compressed worker result was produced. These are
user-supplied diagnostic values, not an independent raw-artifact audit.

Shutdown ordering is the likely explanation, but the old record does not
include thread states and cannot prove exactly which kernel cleanup stage was
observed. Linux sets `PF_EXITING` before releasing the address space and does
not reap a group leader while subthreads remain. RSS is emitted only while a
task has an address space. See the primary
[exit/reaping implementation](https://github.com/torvalds/linux/blob/master/kernel/exit.c),
[proc status/stat implementation](https://github.com/torvalds/linux/blob/master/fs/proc/array.c),
and [PF_EXITING definition](https://github.com/torvalds/linux/blob/master/include/linux/sched.h).

The second repair uses the thread-aware logic described above, not a blanket
increase in tolerance for unobservable live computation. A dead leader does
not waive the guard on live helper threads. Malformed/unreadable exit evidence
does not authorize the longer wait; a stuck shutdown or overall deadline
violation still fails. Valid thread RSS is checked against the same budget.
An exit code of 0 alone, a completed phase marker, or an `ok` worker JSON is
not sufficient to bypass these checks. The error logger now repeats the concise
failure headline **after** verbose worker context, making `tail -n 30` useful.

Local verification: **147 tests passed, 1 Linux-only test skipped** on macOS.
Tests reproduce the previous successful-exit false rejection before the repair,
check delayed zero/nonzero/signalled exits, shared-RSS recovery without double
counting, live/unreadable/new-thread rejection, RSS/reap/deadline limits,
historical comparison/synthesis compatibility, and three small real-FHE
integration cases. The Linux-only test reads actual current-process procfs and
will run in the server wrapper's preflight. No server profiling is run locally.

After committing/pushing this repair, pull it on the server and rerun all three
treatments under fresh `server_run3`. Keep `server_run1` and `server_run2`
unchanged. This remains a diagnostic gate, not an accepted performance result.

## Outputs and follow-on work

The run directory contains `configuration_input.json`, `requested_run.json`, `feasibility.json`, the smoke
`comparison.json`/`.md`/process CSV, and all three raw worker/log/phase/RSS files
under `block_001/`. Wrapper launch/preflight/exit-status files are siblings of
that directory. Source and raw-artifact hashes are recorded in the comparison.

After this gate, the next research implementation remains a genuine Orion
packing/evaluation control for the same decoder function. Simply comparing the
native zero-padded model to the Rotation-Padded model is not a pure layout
comparison. Keep semantics matched or separate the padding/accuracy factor;
the previously validated native-to-fine-tuned Dice gap must remain explicit.
Only then can paired layout timings and later complete-model runs address the
original crossover hypothesis.

## Local verification (2026-10-07)

- **250 targeted Python tests passed**, covering new geometry/config/provenance
  and watchdog failures, historical worker/synthesis compatibility, real-FHE
  three-policy integration, and the preceding WPC/Step-1 regression tests.
- `go test ./... -count=3` passed, including the public parameter-manifest test
  and existing exact compression/online-recipe backend tests. Python compilation,
  shell syntax, and whitespace/diff checks passed.
- Three-policy LogN=11, 8×8 smoke passed with 14 learned transforms and two
  bootstrap groups. A second configuration with ten Q primes passed the complete
  feasibility entry point, exercising level schedule 9/7/9 rather than 8/6/8.
- The final level-9 artifacts are in
  `.tmp/results/honours/29_wpc_decoder_scale_feasibility/local_level9_final_20261007/`.
  All worker/RSS/config/source hashes and recomputed block/smoke summaries were
  checked. These runs used the original locally available checkpoint, **not**
  the server-only audited fine-tuned checkpoint; timings are diagnostics only.
- One earlier sandboxed run passed FHE gates but lacked RSS access, so its
  comparison failed closed. It was preserved, not converted into accepted data.
  Subsequent correctness smokes ran with process-RSS access. The LogN=12, 16×16
  server gate remains unexecuted locally, and no secure/layout-crossover claim
  follows from these tests.
