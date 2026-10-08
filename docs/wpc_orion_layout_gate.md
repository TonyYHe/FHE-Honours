# Stage 30: matched-function Orion–CIPS decoder gate

## What this stage adds

Stage 29 established that the larger CIPS decoder can execute all three storage
policies correctly under resource guards. It did not contain an Orion-layout
control. Stage 30 adds that control for the same decoder-stage function:

`up1 + skip1 → concat → dec1a → checkpoint Cheb7 → bootstrap → dec1b`.

The five treatments run sequentially in **separate processes**:

| Layout | Storage policies | Online weight preparation |
|---|---|---|
| Native Orion control | Full Q/P; online Encode | Regenerate ordinary block diagonals from the float32 kernel |
| WPC CIPS | Full Q/P; online Encode; compressed Q/P | Expand periodic slot recipes, or reconstruct compressed Q/P |

Checkpoint bytes, seeded feature values, CKKS configuration, actual residual and
bootstrap parameters, level schedule, activation coefficients and bootstrap
range are matched. Within each layout, storage policies must have identical
instrumented rotation/conjugation counts. **Between layouts, counts may differ.**
The arithmetic graph is unchanged within a layout; not every individual
addition/multiplication is separately counted by this backend.

## Matched function, not a padding confound

Normal Orion convolution uses zero padding; WPC Rotation Padding changes the
function. Comparing those directly would mix a layout effect with a function
change. The experimental native control therefore uses the same flattened
cyclic boundary function as CIPS. For spatial index (s=hW+w), a kernel tap

\[
(a,b)\in\{-1,0,1\}^2
\]

selects input position

\[
(s+aW+b)\bmod(HW).
\]

This is **one flattened cyclic axis**, not independent toroidal wrapping in H
and W. Corner/row-edge impulse tests check this distinction. Default Orion
zero-padding behavior is unchanged; the adapter is opt-in.

The native control uses the existing Python direct diagonal builders, square
embedding, low-resolution multiplex gap 2 and high-resolution gap 1. It does
not invoke the automatic whole-model compiler, provider, hybrid embedding,
halo layout or C++ diagonal builder. Calling it an unchanged whole-model Orion
benchmark would be incorrect.

Native channel-first branches are ciphertext-aligned, so concat copies their
groups and drops one Q level to match the CIPS permutation's output level.
Copies are independently owned: releasing concat cannot invalidate the skip
input. CIPS retains its ordinary encoded permutation transforms. Both paths
then consume the same levels, including a real activation/bootstrap bridge.

## Code and measurement changes

- `orion/core/packing.py`: an explicit `padding_semantics` keyword on the native
  Conv2d direct builder; default is still `zero`. The cyclic adapter rejects
  unsupported kernels, strides, dilation, groups and halo layouts.
- `orion/experimental/wpc_orion_layout_control.py`: native packing contracts,
  gap-preserving input encryption, full/online plans, aligned concat and the
  native trained-activation/bootstrap bridge. Online evaluation encodes,
  evaluates and deletes **one ordinary transform at a time**. It retains kernel
  values and diagonal-index metadata, not a bank of encoded learned weights.
- `orion/experimental/wpc_cips_checkpoint.py`: layout-signature hooks let the
  native bridge reuse the exact checkpoint polynomial and bootstrap evaluator
  with a channel-first active-slot mask. Existing CIPS contracts remain strict.
- `orion/backend/lattigo/wpc_layout_control.go` and `bindings.py`: actual Q/P
  coefficient-array byte counts and an owned ciphertext copy/level-drop API.
  For (D) diagonals and ring degree (N), ordinary payload bytes are checked
  against (8DN(\ell_Q+\ell_P)). This excludes object overhead and is not RSS.
- `wpc_cips_layer.py` and the isolated worker: full-Q/P learned-transform
  evaluation is actually timed. Previously its `transform_evaluate_s` was an
  **unmeasured zero**, not evidence of zero compute. Historical files are not
  rewritten. Independent clear output values and their float64 SHA-256 are
  retained, allowing raw error recomputation after transfer.
- Native Encode is `GenerateLinearTransform` call wall time, including its
  allocation/binding work. CIPS online Encode is the narrower backend Encode
  timer. Preparation is recorded separately. Neither boundary equals the old
  Step-1 layer-cache category; these percentages must not be subtracted from
  whole-model profiles. LT evaluation timers cover learned transforms, not
  concat, accumulation, bias additions or rescaling.
- Workers explicitly disable streaming, the ordinary layer cache and C++
  packing, and use one direct-pack, diagonal-Encode and compile worker. These
  settings are recorded; prior runs with different settings are not pooled.
- `orion/experimental/wpc_layout_gate.py`: independent validation of raw
  observations, reference/feature hashes, Q/P byte closure, actual Encode
  counters, per-layout operation consistency, finite outputs, timer closure and
  resource budgets. A success boolean alone is insufficient.
- `tools/run_wpc_layout_gate.py`: controller, JSON/Markdown output and read-only
  transfer review. Selected source hashes, loaded-library hashes, checkpoint
  identity, exact config bytes and all worker/RSS/log/phase hashes are retained.
  A changed source/checkpoint or a failed worker fails the gate and preserves
  partial evidence. Existing result directories are never overwritten.
- `tools/run_wpc_layout_gate_server.sh`: rebuild, Python/Go preflight, guarded
  execution and recorded exit status. The existing thread-aware RSS watchdog
  remains unchanged.

Native full Q/P bytes are measured from coefficient arrays; native online
estimates are checked against that full worker. Logical online recipe storage
counts one kernel and its numeric index/block metadata. Python/Go object
overhead, duplicate model tensors, ciphertexts and keys are not part of this
logical plaintext total; process RSS includes runtime allocations.

## Run on the server

Commit/push the implementation locally first. On the server:

```bash
(
set -e
cd ~/FHE-Honours
git pull --ff-only
test -f tools/run_wpc_layout_gate_server.sh

TASK_RUN=.tmp/results/honours/30_wpc_orion_layout_gate/server_run1
mkdir -p .tmp/results/honours/30_wpc_orion_layout_gate
test ! -e "$TASK_RUN" && test ! -e "$TASK_RUN.launch.log" || exit 1
nohup bash tools/run_wpc_layout_gate_server.sh "$TASK_RUN" \
  > "$TASK_RUN.launch.log" 2>&1 < /dev/null &
echo $! | tee "$TASK_RUN.pid"
)
```

The wrapper uses the audited fine-tuned `rotation_padding_best.pt`, LogN=12,
16×16 output, an 8192 MiB sampled-RSS budget and a 1800-second timeout **per
worker**. Optional arguments are `run_directory checkpoint_path rss_budget_mib`.
Use a fresh run name after any failed attempt; preserve old evidence.

The default absolute correctness tolerance is `1e-5`, applied to independent
clear errors and cross-layout output differences. It is recorded in every
worker and is deliberately tighter than Stage 29's `2e-3` diagnostic threshold;
the accepted Stage-29 errors were below `6.2e-7`.

```bash
cd ~/FHE-Honours
cat .tmp/results/honours/30_wpc_orion_layout_gate/server_run1.exit_status
tail -n 30 .tmp/results/honours/30_wpc_orion_layout_gate/server_run1.launch.log
```

Completion requires exit status `0`, `gate.json` status `ok`, and all acceptance
gates true. A disappeared PID or five completion messages alone is insufficient.
The wrapper includes its own build and preflight; no separate rebuild is needed.

For a configuration-only check with no checkpoint load, keys or inference:

```bash
.venv/bin/python tools/run_wpc_layout_gate.py --plan-only
```

After transferring the **complete** run directory, independently check it:

```bash
.venv/bin/python tools/run_wpc_layout_gate.py --review-dir /absolute/path/to/server_run1
```

Review recomputes the JSON summary and Markdown table from raw worker/RSS data,
verifies artifact sizes/digests and archived configuration, and lists differences
from current selected source hashes. Checkpoint and shared-library bytes are
identified but not bundled; their authenticity cannot be established by a hash
record alone. Transfer checksums should also be verified separately.

## Scope and verification

This is one fixed-order diagnostic forward per treatment, without warmups or
confidence intervals. It can establish functional equivalence and resource
feasibility at the selected shape. It cannot establish a latency ranking,
cryptographic security, dataset accuracy, a complete encrypted U-Net result, or
the ResNet/U-Net layout crossover. Repeated balanced-order matched-layout trials
and a security assessment remain later work.

Local checks use small synthetic weights/LogN=9–10: native pack/unpack and
diagonal algebra, independent cyclic/zero-padding oracles, full/online release,
the real activation/bootstrap chain, equivalence to all three CIPS policies,
native worker provenance/counters, and corruption/transfer-review regressions.
The trained LogN=12 server gate has **not** been executed locally.

Verification: the complete WPC suite passed **290 tests**, with one Linux-only
procfs test skipped on macOS; `go test ./... -count=3` passed. The 62 legacy
dense compile tests also passed with `ORION_COMPILE_PARALLEL_POLICY=manual`.
Their default-auto invocation had seven batching-expectation failures on macOS:
the unchanged existing policy reads Linux `/proc/meminfo` and falls back to
one-transform batches when unavailable. No production memory policy was changed
to accommodate those tests. Shell syntax, Python compilation and diff whitespace
checks passed. The macOS Go build used a task-local GOCACHE because the sandbox
could not write the default cache.
