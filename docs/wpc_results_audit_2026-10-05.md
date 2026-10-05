# WPC–Orion results audit — 5 October 2026

Reviewed checkout: `04eb7e0`, `tony/macos-clear-lattigo`.

The saved numbers and figures are internally consistent. The audit found
validation defects, an incomplete checkpoint identity check, and limitations
in the performance experiment that the current report does not explain
sufficiently. These findings do not establish that the saved numerical results
are wrong. They do prevent an unqualified claim that every result has been
independently verified or that the decoder benchmark proves the original
WPC-versus-Orion performance hypothesis.

## Evidence checked

- The original three-model JSON files and measured-forward records in
  `.tmp/results/honours/04_step1_model_matrix`.
- The Step-1 extraction, the transferred Stage-26 synthesis, generated report,
  and all four PNG figures.
- The profiling, periodicity, compressed Q/P, decoder benchmark, fine-tuning,
  and synthesis code; the repository experiment ledger; and the server output
  supplied in the conversation.
- The transferred archive checksum and the locally available Step-1 input's
  SHA-256 against the synthesis manifest. Both match.

Only one of the eleven raw inputs named in the synthesis manifest is available
at its recorded path locally: the Step-1 extraction. The census JSONL files,
Stage-23 accuracy JSON, Stage-24 correctness JSON, Stage-25 worker/comparison
files and logs, and fine-tuned checkpoint were not included in the transferred
synthesis archive. Their server-reported values and the relevant source code
were reviewed, but their raw contents and hashes could not be independently
rechecked locally. The manifest records hashes; it does not include those
artifacts.

## Findings

### 1. Decoder timings use small, insecure test parameters

`tools/run_wpc_cips_trained_decoder.py::_config` uses `LogN=10`,
`LogQ=[55]+[45]*8`, `LogP=[60]`, and `H=192`. Thus the residual ring degree is
1,024 and the nominal residual Q/P modulus budget is approximately 475 bits.
`orion/backend/lattigo/bootstrapper.go` also passes the residual `LogN` to the
bootstrap parameter constructor. The isolated worker explicitly requires
`LogN=10` and an 8×8 output.

These are functional test parameters, not a defensible 128-bit security
configuration. Lattigo's own [bootstrapping example](https://github.com/tuneinsight/lattigo/blob/v6.1.1/examples/singleparty/ckks_bootstrapping/basics/main.go)
labels reduced-ring runs as insecure and requires the user to check security.
The exact attack cost was not estimated in this audit.

Both decoder storage modes use the same parameters, so their comparison is
useful for this test configuration. The 2.39–2.44 second times and measured
memory reductions must be labelled accordingly. They cannot be extrapolated
to the Step-1 whole-model runs, whose saved configuration uses `LogN=16`, or
presented as secure deployment benchmarks without another experiment.

### 2. The 2.49% latency difference includes unequal instrumentation

`orion/experimental/wpc_cips_layer.py::evaluate` defaults to
`record_sequence=True`. The ordinary Orion Conv2d and ConvTranspose2d entry
points do not override it. Each compressed transform therefore performs two
`GetWPCCompressedGlobalStats` calls and constructs an audit record inside the
timed forward. The full-Q/P path does not perform this work. Across 56 learned
transforms, that is 112 extra global-statistics queries per forward.

The parent also always executes full first, compressed second, with one
process per mode and ten repetitions inside each process. Those repetitions
measure within-process variability, not variability across independent
process pairs. There is no counterbalanced execution order or confidence
interval for the reported 2.4855% difference. The effect of the extra
instrumentation was not measured, so it cannot be subtracted retrospectively.

Disable lifecycle tracing during performance measurements after a separate
correctness run, then repeat independent full/compressed process pairs in
balanced order. Keep the current latency delta as a descriptive observation.
RSS is sampled every 5 ms by default; its maximum is a sampled maximum, not
proof that every transient allocation peak was captured.

### 3. Accuracy is linked to the fine-tuned checkpoint by path, not hash

`tools/finetune_wpc_rotation_padding.py::_build_result` hashes the original
source checkpoint but records the evaluated fine-tuned checkpoint only as
`outputs.best_checkpoint`. `build_synthesis` compares that path with the
supplied checkpoint path. It checks content hashes for the supplied file,
decoder correctness, and decoder benchmark, but has no evaluated-best hash
from the accuracy result to compare with them.

A file overwritten at the same path after accuracy evaluation could therefore
pass `checkpoint_identity_consistent` while the accuracy and encrypted tests
refer to different weights. This is a provenance gap, not evidence that an
overwrite happened in the completed run.

Record the evaluated best checkpoint's SHA-256 during validation, verify its
recorded source identity in evaluation-only mode, and require that hash in the
synthesis. Preserve a dataset checksum and exact sample-selection identity as
well; the current result records a dataset path and selection configuration.

### 4. Validation helpers can accept inconsistent or missing measurements

Read-only probes, operating on in-memory copies or existing unit-test
fixtures, reproduced these failures:

| Probe | Expected | Actual |
|---|---|---|
| Replace both ResNet Encode percentage fields with 99%, leaving seconds unchanged | Reject; seconds imply 19.959301% | `summarize_step1` accepts 99% |
| Replace benchmark latency ratio with 2.0, leaving medians 2.0 s and 2.1 s | Reject; ratio must be 1.05 | `summarize_decoder_benchmark` accepts 2.0 |
| Replace correctness result's online weight Encode count with 999, retaining stored true acceptance flags | Reject the zero-online-Encode claim | `summarize_decoder_correctness` accepts 999 |
| Empty all compressed timing arrays in a worker fixture, retaining status/counters | Reject missing measured samples | `compare_trained_decoder_workers` returns `valid=True`, with zero-count latency statistics |

The synthesis frequently trusts prior acceptance flags. Its gate checker
checks whether supplied gate values are true, rather than requiring a complete
set of named gates and independently validating every relationship.

Require finite, positive forward times; exact requested array lengths; aligned
subtimer arrays; valid output shapes and error bounds; independently recomputed
percentages, ratios, and summary statistics; and explicit required gates.
Reject impossible values even when a stored acceptance flag says true.

The saved Stage-1 percentages, Stage-26 benchmark ratios, storage reductions,
and accuracy deltas passed independent arithmetic checks. These probes show
weaknesses in the validator, not demonstrated corruption of those saved values.

### 5. Some reported counters are assertions rather than measured counters

The trained worker increments `bootstrap_call_count` once per successful graph
iteration. This is a bridge invocation count, not an independent backend count
of the four ciphertext bootstraps. The compressed Go state initializes
`OfflineEncodeCalls=1` and `OnlineEncodeCalls=0`; the full worker reports 56
offline calls and zero online calls as constants. Offline calls here count
transform-generation invocations rather than individual diagonal encodings.

The current decompression source does reconstruct Q/P by coefficient copies
without calling the encoder, and the Python Encode wrapper is instrumented.
Those facts support the intended mechanism. The counter names and acceptance
wording should nevertheless distinguish measured events from structural
assertions; backend Encode/bootstrap counters would detect future regressions.

### 6. The report loses model identity and overstates the layout comparison

The raw VGG result identifies `VGG16-base16`, but synthesis relabels it `VGG16`.
The builder explicitly uses `base_dim=16`; there is a separate base64 builder.
The figure and table should preserve the narrower model's identity.

Step-1 U-Net is a 64×64 model constructed with seed zero. The later accuracy
experiment is a checkpoint-trained 256×256 COVID-19 U-Net22-plus-output model.
The decoder resource experiment uses synthetic 4×4 and 8×8 internal features.
The periodicity runner records `none:deterministic-seed=0` by default; it does
not load trained model weights. These are distinct workloads.

Stage 25 compares full and compressed Q/P under the same WPC CIPS layout.
It is not a matched comparison of Orion's original layout against CIPS, and
both storage modes already encode weights offline. It therefore isolates the
cost and storage benefit of Q/P compression within CIPS, not removal of the
90.47% online Encode time measured for the earlier Orion model. The statement
that adopting CIPS itself causes the measured memory/latency improvement goes
beyond this control experiment.

### 7. Reporting and experiment history need correction

- The fine-tuning documentation still says the five-epoch result is pending,
  and the experiment ledger says server synthesis is pending, despite the
  completed server evidence supplied here.
- `tools/synthesize_wpc_orion_tradeoff.py::_render_report` hardcodes that one
  pre-fine-tuning sample is non-finite and that the selective-layout hypothesis
  is unsupported. Those statements match the current inputs but can become
  false for another valid input set. Narrative must be derived from data.
- The README still describes completed WPC work as planned.

## Findings that are reasonable

The three whole-model means and sample standard deviations agree with the raw
three measured forwards, excluding warmup. Category percentages agree with
their seconds and denominator, and all category sums close. Closure is partly
by construction because `other_he_forward` is a residual; it alone does not
prove that every constituent interval is exclusive.

“Online Encode” measures layer materialization, payload construction, backend
transform encoding, and garbage collection inside that interval. Its 90.47%
U-Net share is valid for that defined wall-time category, but it does not
identify how much is CKKS Encode arithmetic alone.

The 14/73,077 U-Net count coverage is correctly 0.019158%; byte coverage is
0.016575%. VGG's 169 all-zero diagonals are correctly excluded from its
128,218 nonzero denominator. Census byte totals are aggregate logical payload
estimates rather than peak process memory. Raw census records still need to
be available locally for an independent reclassification audit.

The transferred summary's 12.944× logical storage ratio, 92.2745% logical
storage reduction, 40.7800% sampled peak-RSS reduction, and 2.4855% median
latency difference are arithmetically correct. The four figures agree with
their summary values.

The final clear-model Dice gap is correctly −0.033634, or 3.3634 percentage
points. The un-fine-tuned model's loss of 6,868.50 is accompanied by a known
polynomial instability and one non-finite sample; it is a finite-subset
diagnostic. It is not comparable to an all-sample result without qualification.
The 512 validation samples used for checkpoint selection are included in the
full 2,115 validation set, so the latter is validation evidence rather than an
untouched test-set evaluation. Full-model clear accuracy and synthetic-stage
encrypted correctness should remain separate claims.

## Verification and disposition

107 targeted Python tests passed across profiling, extraction, synthesis,
periodicity, training, and CIPS packing/layer/checkpoint paths. Go `TestWPC`
tests passed. The deliberate invalid-input probes above are outside those
existing test cases and explain why passing tests do not close the validation
gaps.

No implementation or saved experiment result was changed during this audit.
This file records the findings. Preserve the current evidence, harden the
validators and checkpoint provenance, correct model/security/scope labels,
and then repeat the decoder timing experiment with balanced execution and
without compressed-only lifecycle tracing. A complete encrypted-network
WPC-versus-Orion claim still requires matched workloads under both layouts.
