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
  A missing final RSS sample triggers at most a **100 ms bounded exit check**:
  an exited worker retains its real exit code; a worker still alive without
  readable RSS is stopped and rejected. `.rss.json` records unavailable samples,
  whether exit was confirmed after one, and the exit-check timeout. Missing
  samples are never inserted as zero-byte observations or counted as valid samples.
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

The sampler now checks for an exit for up to 100 ms before treating RSS loss as
a live-worker monitoring failure. Regression tests cover successful and failed
exits observed through both `poll()` and `wait()`, actual monitoring loss while
the worker remains alive, and the existing RSS/time budget failures. The memory
budget, time budget, FHE tolerance and other acceptance gates are unchanged.
No old JSON is edited or promoted to valid.
Local repair verification passed **127 tests**: geometry/config/watchdog,
online-Encode and trained-decoder comparison, synthesis compatibility, and
three small real-FHE storage-policy integration cases. Python compilation,
server-wrapper shell syntax, and diff whitespace checks also passed. No remote
job or larger local profiling run was launched for this repair.

After committing/pushing this repair, pull it on the server and rerun the same
gate under the fresh name `server_run2`. Keep `server_run1` and all its sibling
launch/preflight/exit-status files intact. The wrapper rebuilds and tests before
starting all three workers; it refuses existing evidence. This remains a
diagnostic correctness/resource gate, not a repeated performance experiment.

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
