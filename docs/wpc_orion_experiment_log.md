# WPC–Orion experiment log

This log separates user-executed remote measurements from local implementation
and unit-test work. A slot-periodicity census identifies WPC *candidates*; it
does not demonstrate encoded Q/P eligibility or a WPC runtime speedup.

## Remote run ledger

### 2026-09-27 — ResNet20 O-online control

- Executor: user, on `tony@corg-comb`
- Repository: `tony/macos-clear-lattigo`
- Commit: `1a148e46bcf806c0a7e10a2f69b61b02c1dcc718`
- Network/mode: `resnet20_cifar10`, `dense`
- Backend: real FHE Lattigo
- Protocol: one warmup followed by three measured forwards
- Encode workers: 1
- Result: `.tmp/results/honours/08_wpc_orion_tradeoff/resnet20_o_online.json`
- Log: `.tmp/results/honours/08_wpc_orion_tradeoff/resnet20_o_online.log`

The result passed all checks exposed by the schema-v2 Step 1 runner:

| Check | Result |
|---|---:|
| Runner status | `ok` |
| Successful measured forwards | 3/3 |
| Step 1 schema | 2 or newer |
| Step 1 profile valid | yes |
| Per-forward profiles collected | 3/3 |
| Additive major-wall accounting | valid |
| Accounting closure error | 0 s |
| Decrypted output shape | matched |
| Fatal markers in saved log | none found |

Reported mean timings and shares:

| Metric | Value |
|---|---:|
| HE forward | 2,088.760 s |
| Online Encode | 435.489 s |
| Online Encode / HE forward | 20.849% |
| Bootstrap / HE forward | 66.166% |
| MVM kernel / HE forward | 4.078% |
| Other HE forward | 8.701% |

The additive major categories close to exactly 100% of HE-forward time. The
operator microprofile is diagnostic and contains nested/overlapping categories;
for example, its elementwise-multiply, rotation, and bootstrap percentages must
not be summed as though they were disjoint wall-time categories.

The full JSON and environment manifest have not yet been copied into this local
working tree. The values above are transcribed from the validation output and
the final profile summary supplied by the user.

### 2026-09-28 — Initial WPC CIPS `3 x 1` functional baseline

- Executors: local development machine and user on `tony@corg-comb`
- Remote commit: `606834a`
- Scope: WPC Algorithms 1–2 and Figures 8–10 vertical-convolution subset
- Backend: clear NumPy reference plus real Lattigo CKKS
- Workload: one ciphertext, 512 slots, `8 x 8`, 4 input/output channels,
  `3 x 1` kernel, stride one, circular same-channel height Rotation Padding
- Seed: `20260928`
- Result: `.tmp/results/honours/11_wpc_cips_baseline/cips_baseline.json`
- Result: valid

| Gate | Result |
|---|---:|
| Python/regression tests | 31 passed |
| Clear channel-first reference | pass |
| Clear CIPS reference | pass |
| CIPS periodic diagonals | 21/21 |
| CIPS minimal slot period | 8 of 512 slots |
| CIPS analytical partial-storage ratio | 64.0x |
| Exact Lattigo encoded Q/P reconstruction | 21/21 pass |
| Real-FHE channel-first maximum absolute error | `8.95e-8` |
| Real-FHE CIPS maximum absolute error | `8.95e-8` |
| Rotation Padding differs from zero padding | yes |

The matched channel-first control had 23 nonzero diagonals and no proper
periodic diagonal; CIPS had 21 nonzero diagonals and all were periodic. This is
the expected mechanism-level contrast: WPC creates periodicity through its
layout and padding construction, whereas selectively inspecting the unchanged
Orion layout finds essentially none.

The result is not a model reproduction or performance measurement. Its width
kernel is one, and it omits multi-ciphertext channel groups, downsampling
reshaping, fine-tuning, and the compressed online Encode/decompression path.
The single-run timings in the JSON are diagnostic only. See
`docs/wpc_cips_baseline.md` for the construction, acceptance gates, server
command, and remaining work.

The remote reproduction rebuilt the Linux Lattigo shared library, passed the
same 31 Python tests and Go `TestWPC` suite, passed every acceptance gate, and
exited with status zero. Its exact Q/P records resolved
`orion/backend/lattigo/lattigo-linux.so`, confirming that the server used the
rebuilt Linux backend rather than the local macOS library.

### 2026-09-28 — General `3 x 3` CIPS correctness extension

- Executor: local development machine
- Backend: clear NumPy reference plus real Lattigo CKKS
- Workload: one ciphertext, 512 slots, `8 x 8`, 4 input/output channels,
  `3 x 3` kernel, stride one
- Rotation Padding: Algorithm 2 flattened spatial cyclic rotation, including
  width offsets crossing packed row boundaries
- Seed: `20260928`
- Result: `.tmp/results/honours/11_wpc_cips_baseline/cips_baseline.json`
- Result: valid

| Gate | Result |
|---|---:|
| Python/regression tests | 32 passed |
| Explicit flattened width-boundary test | pass |
| Clear channel-first reference | pass, maximum error `6.66e-16` |
| Clear CIPS reference | pass, maximum error `8.88e-16` |
| CIPS periodic diagonals | 63/63 |
| CIPS minimal slot period | 8 of 512 slots |
| CIPS analytical partial-storage ratio | 64.0x |
| Exact Lattigo encoded Q/P reconstruction | 63/63 pass |
| Real-FHE channel-first maximum absolute error | `2.04e-7` |
| Real-FHE CIPS maximum absolute error | `2.04e-7` |
| Real-FHE correctness tolerance | `1e-6`, both pass |
| Rotation Padding differs from zero padding | yes |

The matched channel-first control had 71 nonzero diagonals and no proper
periodic diagonal. CIPS had 63 nonzero diagonals, all with slot period 8. The
flattened width-boundary test distinguishes Algorithm 2 semantics from both
zero padding and independent height/width modulo wrapping.

This extension removes the earlier width-one limitation but remains a
single-ciphertext mechanism test. Multi-ciphertext channel groups,
downsampling reshaping, model fine-tuning, compressed online decompression,
and model-level performance measurements remain out of scope.

### 2026-09-28 — Real compressed Q/P storage and online decompression

- Executor: local development machine
- Backend: real Lattigo CKKS
- Workload: the validated single-ciphertext `3 x 3` CIPS transform
- Result: `.tmp/results/honours/12_wpc_compressed_qp/cips_3x3_compressed_qp.json`
- Result schema: 3
- Result: valid

| Gate or metric | Result |
|---|---:|
| Compressed diagonals | 63 |
| Full Q/P payload | 2,064,384 B |
| Compressed Q/P payload | 32,256 B |
| Logical period/level metadata | 3,552 B |
| Stored payload plus metadata | 35,808 B |
| Payload-only compression ratio | 64.0x |
| Compression ratio including metadata | 57.651x |
| Exact decompressed-vs-full Q/P equality | pass |
| Offline weight-plaintext Encode calls | 1 |
| Online weight-plaintext Encode calls | 0 |
| Compressed-vs-full decrypted output delta | 0 |
| Full/compressed CIPS rotation counts | 17 / 17 |
| Maximum error vs. clear reference | `2.04e-7` |
| Full bytes materialized after online call | 0 B |

The online path copies stored representatives into full Q/P polynomials,
evaluates the ordinary Lattigo transform, and releases those polynomials before
returning. The default local diagnostic recorded approximately `0.24 ms` for
decompression and `1.64 ms` for evaluation. Those single-run numbers are not a
performance claim.

This result advances beyond the earlier candidate verifier: the compressed
representation is now the state retained by a real transform and is consumed
by an executable encrypted evaluation path. It still covers only one
single-ciphertext transform and temporarily materializes that transform during
evaluation; multi-group model integration and peak-memory measurement remain.

### 2026-09-27 — ResNet20 Orion slot-periodicity census

- Executor: user, on `tony@corg-comb`
- Network/mode: `resnet20_cifar10`, `dense`
- Backend: clear Lattigo; one structural forward
- Scope: slot-message periodicity only; no compression or performance timing
- Result: valid

| Census metric | Occurrence | Unique diagonal |
|---|---:|---:|
| Observed nonzero diagonals | 8,381 | 8,381 |
| Exact proper-period candidates | 0 | 0 |
| Count coverage | 0% | 0% |
| Full encoded bytes | 38,254,673,920 | 38,254,673,920 |
| Periodic full encoded bytes | 0 | 0 |
| Byte coverage | 0% | 0% |
| Selective-layout storage ratio | 1.0x | 1.0x |

All 8,381 records had complete identity metadata, occurrence and unique JSONL
counts matched their summaries, and no logical diagonal changed payload during
the run. There were no all-zero or constant-nonzero records. The 35.63 GiB
value is a sum of logical encoded Q/P plaintext sizes across diagonals at their
respective levels, not a resident- or peak-memory measurement.

For the current Orion ResNet20 layout, selective WPC has no exact slot-periodic
diagonal to compress. This result concerns WPC compatibility of the existing
Orion layout; it does not evaluate a full CIPS/Rotation-Padding WPC layout.

### 2026-09-27 — U-Net census attempt 1 (invalid metadata identity)

The first `u22_64_base32` provider census completed its clear forward and slot
scan, but validation rejected the report. It recorded 73,077 occurrences and
14 periodic slot messages, while reporting 10,095 logical IDs with multiple
payload hashes. Occurrence and unique-record counts were both 73,077, which
isolated the problem to logical identity collisions rather than duplicate
payload observations.

Cause: provider transform names and transform-local indexes are reused across
multiple unified groups, but the profiler's `transform_id` omitted the group
storage key. Records from distinct groups therefore shared a logical identity.
The fix includes `runtime_storage_key` in the provider transform identity. A
regression test now constructs two groups with the same transform name and
diagonal index but different payloads and requires zero reported payload
changes. The invalid attempt's 14/73,077 periodic count must not be used as a
final result; U-Net must be rerun after the fix.

### 2026-09-27 — U-Net census retry 2 (valid)

- Executor: user, on `tony@corg-comb`
- Network/mode: `u22_64_base32`, `provider`
- Backend: clear Lattigo; one structural forward
- Scope: slot-message periodicity only; no compression or performance timing
- Result: valid after provider group identity fix

| Census metric | Occurrence | Unique diagonal |
|---|---:|---:|
| Observed nonzero diagonals | 73,077 | 73,077 |
| Exact proper-period candidates | 14 | 14 |
| Constant-nonzero candidates | 14 | 14 |
| Count coverage | 0.019158% | 0.019158% |
| Full encoded bytes | 366,921,383,936 | 366,921,383,936 |
| Periodic full encoded bytes | 60,817,408 | 60,817,408 |
| Byte coverage | 0.016575% | 0.016575% |
| Selective-layout storage ratio | 1.000166x | 1.000166x |

All identity and correctness checks passed, including zero logical payload
changes. Encode-time coverage remains unavailable because this read-only audit
does not time each diagonal. The 14 constant candidates require layer/operator
classification before they can be described as learned-weight opportunities;
the census also includes non-weight online linear transforms.

Candidate inspection assigned all 14 records to `cat1_materialize_*`,
`cat2_materialize_*`, or `cat3_materialize_*`. These names are created by the
explicit concatenation materializers in `orion/nn/operations.py`; every record
was diagonal 0 with period 1. They are structural layout transforms, not
learned Conv, ConvTranspose, or Linear weights. Therefore U-Net has zero
periodic learned-weight candidates, while the broader online-LT scope contains
14/73,077 constant structural candidates. The broader byte coverage remains
0.016575%; periodic learned-weight byte coverage is 0%.

### 2026-09-28 — U-Net encoded Q/P candidate verification

- Executor: user, on `tony@corg-comb`
- Network/mode: `u22_64_base32`, `provider`
- Model execution: one clear-Lattigo structural forward
- Candidate verification: independent key-free real-Lattigo Encode using the
  same CKKS parameter set, followed by an exact `2T` copy-map and full Q/P
  polynomial reconstruction check
- Result directory:
  `.tmp/results/honours/10_wpc_encoded_qp_verification`
- Result: valid

| Verification metric | Result |
|---|---:|
| Observed nonzero diagonals | 73,077 |
| Slot-periodic candidates | 14 |
| Encoded Q/P attempts | 14 |
| Exact Q/P passes | 14 |
| Exact Q/P failures | 0 |
| Verification complete | yes |
| Further encoded-representation verification required | no |
| Count coverage | 0.019158% |
| Byte coverage | 0.016575% |
| Selective-layout storage ratio | 1.000166x |

The encoded-form result confirms that the 14 structural period-1 candidates
are genuinely WPC-reconstructible under Lattigo rather than false positives of
the slot-level classifier. It does not change their classification: none is a
learned-weight diagonal, and the selective storage opportunity in the existing
Orion layout remains negligible.

### 2026-09-27 — VGG16 Orion slot-periodicity census

- Executor: user, on `tony@corg-comb`
- Network/mode: `vgg16_imgnet`, `provider`
- Backend: clear Lattigo; one structural forward
- Scope: slot-message periodicity only; no compression or performance timing
- Result: valid

| Census metric | Occurrence | Unique diagonal |
|---|---:|---:|
| Observed diagonals | 128,387 | 128,387 |
| All-zero diagonals | 169 | 169 |
| Observed nonzero diagonals | 128,218 | 128,218 |
| Exact proper-period nonzero candidates | 0 | 0 |
| Count coverage | 0% | 0% |
| Full encoded nonzero bytes | 491,591,827,456 | 491,591,827,456 |
| Periodic full encoded bytes | 0 | 0 |
| Byte coverage | 0% | 0% |
| Selective-layout storage ratio | 1.0x | 1.0x |

All correctness, metadata, JSONL-count, and logical-payload checks passed.
All-zero diagonals are reported separately and excluded from the WPC candidate
denominators rather than being counted as a compression success.

## Orion-layout compatibility result

| Model | Nonzero online-LT diagonals | Periodic learned-weight candidates | Periodic structural candidates | All-LT count coverage | All-LT byte coverage |
|---|---:|---:|---:|---:|---:|
| ResNet20/dense | 8,381 | 0 | 0 | 0% | 0% |
| U-Net22/provider | 73,077 | 0 | 14 | 0.019158% | 0.016575% |
| VGG16/provider | 128,218 | 0 | 0 | 0% | 0% |

The compatibility hypothesis is not supported for the current Orion layouts:
U-Net's high online-Encode share does not correspond to naturally WPC-periodic
learned-weight diagonals. Selective compression of already-periodic Orion
payloads would have no measurable weight-storage benefit; even including
U-Net's 14 concatenation materializers gives only a 1.000166x analytical
storage ratio.

This does **not** test or refute full WPC. WPC obtains periodicity through its
CIPS/Rotation-Padding data layout. A valid layout-trade-off comparison must next
implement or reproduce that layout and compare it with Orion under matched
models, parameters, accuracy, and operation-count accounting.

## Periodicity-census implementation

The next stage is an untimed clear-Lattigo audit. It must not be mixed with the
real-FHE performance measurements above.

Current implementation changes:

- `orion/experimental/wpc_periodicity.py` classifies the exact minimal
  power-of-two period of real or interleaved-complex slot messages, excludes
  all-zero messages from WPC candidate coverage, calculates exact full Q/P
  polynomial bytes, splits flattened backend payloads into diagonals, and
  writes JSONL plus an aggregate summary.
- `orion/backend/python/lt_evaluator.py` records dense single-slot payloads
  immediately before the unchanged Lattigo generation call.
- `orion/nn/unified_transform.py` records provider payloads at the equivalent
  boundary, including grouped materialization.
- `tools/run_wpc_periodicity_census.py` configures a single clear structural
  forward and validates that it produced a complete, nonempty census.
- `orion/backend/lattigo/wpc_periodicity_test.go` verifies the library-specific
  bit-reversed NTT copy map in every Q and P limb, exact reconstruction, several
  periods, and rejection of a nonperiodic message.
- `orion/backend/lattigo/wpc_verification.go` exports an audit-only verifier
  that independently encodes one real candidate message with the model's CKKS
  parameters, checks the expected evaluation period `2T` in every active Q/P
  limb, reconstructs every coefficient from the stored representatives, and
  requires exact equality with the original polynomial.
- `--verify-encoded-qp` makes the clear census load a second, key-free real
  Lattigo instance and invoke that verifier only for candidates. This preserves
  clear execution for the model forward and avoids paying for a full encrypted
  U-Net inference merely to verify 14 structural messages.

Profiling is gated by `ORION_WPC_PERIODICITY_PROFILE`; disabled execution does
not scan or convert payloads. The audit launcher fixes both streaming flags to
zero because this is a clear census, not a real-FHE streaming run.

## Verification performed locally

```text
python -m pytest -q tests/test_wpc_periodicity.py
25 passed

go test -run 'TestWPC' -count=1
PASS
```

The encoded verifier also passed a direct Python/ctypes integration probe after
building the shared library locally. The census launcher's encoded-verification
dry run resolved the library and exported symbol and produced the intended
clear-Lattigo command and output paths. No model census or WPC performance
experiment was executed locally, following the decision to perform experiment
runs on the remote server.

## Remaining validity boundary

The U-Net candidates now pass the encoded Q/P eligibility gate. The launcher
still does not apply compression, measure decompression, or provide
Encode-time-weighted coverage. Those claims require an O-hybrid implementation
and timing run. Because the only eligible messages are 14 structural
materializers (0.016575% byte coverage), selective WPC on Orion cannot establish
the proposed layout crossover; the decisive next experiment is a matched
CIPS/Rotation-Padding reproduction versus Orion.

### 2026-09-28 — Multi-group compressed CIPS functional baseline

- Executor: local deterministic correctness run; server confirmation pending
- Layout: CIPS with flattened two-dimensional Rotation Padding
- Convolution: 12 input channels, 12 output channels, `3 x 3`, `8 x 8`
  spatial grid
- CKKS: `LogN=10`, 512 slots, channel capacity 8
- Group matrix: 2 output groups x 2 input groups, four transforms
- Result: valid
- Machine-readable result:
  `.tmp/results/honours/13_wpc_multigroup/cips_multigroup_compressed_qp.json`

| Metric | Result |
|---|---:|
| Clear grouped-vs-global maximum error | `4.44e-15` |
| Encoded Q/P diagonal checks | 318/318 pass |
| Exact compressed-vs-full transforms | 4/4 pass |
| Aggregate full Q/P payload | 10,420,224 B |
| Aggregate compressed payload | 162,816 B |
| Logical metadata | 17,904 B |
| Stored payload plus metadata | 180,720 B |
| Payload-only compression ratio | 64.0x |
| Compression ratio including metadata | 57.659x |
| Peak materialized full payload | 3,047,424 B |
| Peak materialized transforms | 1 |
| Aggregate-full / sequential-peak ratio | 3.419x |
| Offline / online weight Encode calls | 4 / 0 |
| Full / compressed LT rotations | 71 / 71 |
| Ciphertext accumulation additions | 2 per path |
| Compressed output delta vs. full | 0 |
| Maximum FHE error vs. clear | `1.81e-7` |
| Materialized payload after execution | 0 B |

The runner compiles one transform for each input/output group pair. The
compressed path synchronously reconstructs one transform, evaluates it,
releases its full Q/P representation, and only then proceeds to the next
transform. Backend-wide counters observed zero materialized bytes before and
after every call, one transform at peak, and peak bytes equal to the largest
single transform rather than aggregate full storage. This proves the intended
multi-group storage lifecycle and encrypted accumulation mechanics.

The full-Q/P transforms coexist in this runner only as an exact correctness
control. Payload counts are exact logical Q/P storage, not process RSS. The
single diagnostic timings have no warmup or repetitions and are not a
performance result. Model-layer integration, WPC downsampling reshaping,
trained Rotation-Padding accuracy, repeated latency, and process-RSS
measurements remain outstanding.

### 2026-09-28 — Orion Conv2d two-layer CIPS integration

- Executor: local deterministic correctness run; server confirmation pending
- Modules: two actual `orion.nn.Conv2d(12,12,3)` layers with bias
- Spatial shape: `8 x 8`; Rotation Padding; stride one
- CKKS: `LogN=10`, levels `2 -> 1 -> 0`
- Group matrix: 2 output x 2 input groups per layer
- Compressed transforms: eight total
- Result: valid
- Machine-readable result:
  `.tmp/results/honours/14_wpc_cnn_layer_pipeline/`
  `two_conv_cips_compressed_qp.json`

| Metric | Result |
|---|---:|
| Exact compressed-vs-full transforms | 8/8 pass |
| Aggregate full weight Q/P payload | 18,235,392 B |
| Aggregate compressed weight payload | 284,928 B |
| Weight metadata | 35,808 B |
| Uncompressed bias Q payload | 49,152 B |
| Full weights plus bias | 18,284,544 B |
| Stored weights, metadata, and bias | 369,888 B |
| Weight payload compression | 64.0x |
| Layer plaintext compression including bias | 49.433x |
| Peak materialized full payload | 3,047,424 B |
| Peak materialized transforms | 1 |
| Aggregate full weight / peak | 5.984x |
| Offline / online weight Encode calls | 8 / 0 |
| Online Python Encode calls | 0 |
| Full / compressed rotations | 142 / 142 |
| Ciphertext accumulation additions | 4 per path |
| Bias plaintext additions | 4 per path |
| Compressed output delta vs. full | 0 |
| Maximum two-layer error vs. clear | `3.11e-8` |
| Materialized payload after execution | 0 B |

The new opt-in planner extracts real Orion layer parameters, compiles the CIPS
group matrix, and returns output ciphertexts carrying an explicit packing
signature. The second layer accepts the first layer's output without repacking
and verifies its input level. The compressed validation calls the installed
`Conv2d.forward` methods, while a full-Q/P control runs through the same plan
and bias/rescale lifecycle.

This establishes same-shape convolution-layer integration, not a complete
model. PyTorch clear Conv2d still has zero-padding semantics; model training or
fine-tuning must explicitly adopt Rotation Padding. Activation/bootstrap,
stride-two reshaping, residual/concatenation paths, RSS, repeated timing, and
trained accuracy remain outside this stage.

### 2026-09-29 — Isolated full vs compressed resource benchmark

- Executor: local deterministic validation; Linux server confirmation pending
- Workers: separate fresh `full` and `compressed` processes
- Modules: two actual `orion.nn.Conv2d(12,12,3)` layers with bias
- Warmups / measured forwards: 2 / 10 per worker
- Memory sampling: external macOS `ps` RSS plus Go `runtime.MemStats`
- Result: valid; all acceptance gates pass
- Machine-readable result:
  `.tmp/results/honours/15_wpc_isolated_resource_benchmark/comparison.json`
- Generated report:
  `.tmp/results/honours/15_wpc_isolated_resource_benchmark/comparison.md`

| Metric | Full Q/P | WPC compressed |
|---|---:|---:|
| Logical resident plaintext storage | 18,284,544 B | 369,888 B |
| Go heap-in-use after compile GC | 25,804,800 B | 7,979,008 B |
| Pre-online RSS | 397,885,440 B | 386,646,016 B |
| Measured-phase peak RSS | 414,318,592 B | 387,252,224 B |
| Median two-layer forward | 12.221 ms | 15.064 ms |
| p95 two-layer forward | 13.107 ms | 16.743 ms |
| Maximum error versus clear | `3.11e-8` | `3.11e-8` |
| Rotations per forward | 142 | 142 |
| Online Encode calls | 0 | 0 |

The local logical-storage ratio is 49.433x. The compressed median forward is
1.233x the full-Q/P median; its median Q/P decompression time is 1.794 ms, or
12.038% of compressed forward wall time. The independently encrypted outputs
match exactly in this run.

These timings and RSS values are machine-specific local evidence rather than
the server result. RSS includes Python, Torch, keys, ciphertexts, and runtime
allocator pages, while the logical payload count isolates transform storage.
The Linux server run must replace these local performance observations before
they are used in a thesis table.

### 2026-09-29 — CIPS activation/bootstrap integration

- Executor: seeded local functional validation; Linux server confirmation pending
- Pipeline: compressed CIPS `Conv2d -> Quad -> Bootstrap -> Conv2d`
- Spatial shape: `8 x 8`; 12 channels; two ciphertext groups
- CKKS: `LogN=10`, levels `3 -> 2 -> 1 -> bootstrap(0 -> 3) -> 2`
- Compressed transforms: eight total
- Result: valid; all acceptance gates pass
- Machine-readable result:
  `.tmp/results/honours/16_wpc_activation_bootstrap/`
  `two_conv_quad_bootstrap.json`

| Metric | Result |
|---|---:|
| Exact compressed-vs-full transforms | 8/8 pass |
| Backend bootstrap calls per path | 2 |
| First-convolution maximum error | `1.11e-8` |
| Activation/bootstrap maximum error | `2.54e-9` |
| Final maximum error versus clear | `1.06e-9` |
| Compressed output delta vs. full | `0` |
| Full weight Q/P payload | 26,050,560 B |
| Compressed weight payload | 407,040 B |
| Weight payload compression | 64.0x |
| Stored weights, metadata, and bias | 541,152 B |
| Full/stored ratio including bias | 48.321x |
| Peak materialized transforms | 1 |
| Offline / online weight Encode calls | 8 / 0 |
| Online Python Encode calls | 0 |
| Materialized payload after execution | 0 B |

The activation/bootstrap bridge preserves the explicit CIPS packing signature
and validates levels before and after each operation. Its bootstrap prescale
mask follows the CIPS slot equation: for a partial channel group, active
channels occupy the first positions of every spatial block. A naive contiguous
prefix mask was rejected during development because it produced a `5.60e-2`
intermediate error; the corrected interleaved mask reduces that error to
`2.54e-9`.

This is a functional correctness and storage-accounting result. The quadratic
activation is not a trained-model activation, and the single-run wall times
are not a performance comparison. Stride-two reshaping,
residual/concatenation paths, higher-degree activations, and trained-model
accuracy remain outstanding.

### 2026-09-29 — Stride-two CIPS and encrypted reshaping

- Executor: seeded local functional validation; Linux server confirmation pending
- Pipeline: compressed stride-two `Conv2d -> reshape -> Conv2d`
- Input/output spatial shapes: `8 x 8 -> 4 x 4`
- Channel groups: two sparse input-grid groups -> one compact CIPS group
- CKKS levels: `3 -> 2 -> reshape(2 -> 1) -> 0`
- Compressed weight transforms: five total
- Ordinary reshape transforms: two, containing eight diagonals
- Result: valid; all acceptance gates pass
- Machine-readable result:
  `.tmp/results/honours/17_wpc_downsample_reshape/`
  `stride2_reshape_pipeline.json`

| Metric | Result |
|---|---:|
| Exact compressed-vs-full weight transforms | 5/5 pass |
| Stride-two weight period / compression | 128 / 4.0x |
| Post-downsample weight period / compression | 32 / 16.0x |
| Aggregate full / compressed weight Q/P | 18,112,512 / 3,574,272 B |
| Aggregate weight payload compression | 5.067x |
| Layer storage ratio including bias | 4.963x |
| Ordinary reshape Q/P payload | 262,144 B |
| Full / compressed rotations | 110 / 110 |
| Online Python / weight Encode calls | 0 / 0 |
| Maximum sparse and reshaped error | `3.09e-8` |
| Maximum final error versus clear | `1.91e-8` |
| Compressed final delta versus full | `0` |
| Peak materialized weight transforms | 1 |

The stride-two convolution keeps its logical outputs at every second input-grid
position. A distinct encrypted permutation transform compacts these positions
into lower-resolution CIPS, merges the two channel groups into one ciphertext,
and consumes one CKKS level. Packing signatures and level checks prevent this
sparse intermediate from being passed directly to an incompatible layer. No
decrypt, decode, clear repack, Encode, or re-encrypt occurs at the boundary.

The stride-two weights compress by 4x rather than the 64x observed for the
earlier same-shape `8 x 8` layers because their exact slot period grows to 128.
After reshaping to `4 x 4`, the next layer's period is 32 and its weight payload
compresses by 16x. These are functional local results; server timing and RSS,
trained-model accuracy, and residual/concatenation joins remain outstanding.

### 2026-09-29 — CIPS residual and concatenation branch joins

- Executor: seeded local functional validation; Linux server confirmation pending
- Pipeline: three compressed branches, residual add, channel concat, compressed consumer
- Spatial shape: `8 x 8`; CIPS channel capacity eight
- Channels: residual `12 + 12 -> 12`; concat `12 + 4 -> 16`; consumer `16 -> 8`
- Levels: branch convolutions `3 -> 2`, residual `2 -> 2`, concat `2 -> 1`, consumer `1 -> 0`
- Compressed weight transforms: 12 total
- Ordinary concat transforms: three, each with one diagonal
- Result: valid; all acceptance gates pass
- Machine-readable result:
  `.tmp/results/honours/18_wpc_branch_joins/residual_concat_pipeline.json`

| Metric | Result |
|---|---:|
| Exact compressed-vs-full weight transforms | 12/12 pass |
| Weight payload compression | 64.0x |
| Full / compressed weight Q/P | 36,519,936 / 570,624 B |
| Layer storage ratio including bias | 48.475x |
| Ordinary concat Q/P payload | 98,304 B |
| Full weight+bias / stored weight+concat | 42.898x |
| Residual additions / levels consumed | 2 / 0 |
| Concat input / output ciphertext groups | 3 / 2 |
| Full / compressed rotations | 214 / 214 |
| Online Python / weight Encode calls | 0 / 0 |
| Maximum residual and concat error | `3.99e-8` |
| Maximum final error versus clear | `2.53e-8` |
| Compressed final delta versus full | `0` |
| Peak materialized weight transforms | 1 |

The residual plan rejects different shapes, signatures, levels, schemes, or
group counts and performs a direct ciphertext addition per channel group. The
concat plan generates encrypted permutation transforms for every source/output
group intersection, so branch offsets that cross a ciphertext boundary are
handled without clear repacking. In this case three source ciphertexts become
two output ciphertexts, and concat consumes one level.

Concat plaintexts are ordinary offline-encoded Q/P permutations and are
reported separately from compressed convolution weights. These are functional
local results; lazy concat fusion, upsampling, server timing/RSS, and
trained-model accuracy remain outstanding.

### 2026-09-29 — CIPS transposed-convolution upsampling

- Executor: seeded local functional validation; Linux server confirmation pending
- Pipeline: compressed `ConvTranspose2d(2x2,stride=2) -> Conv2d(3x3)`
- Spatial shapes: `4 x 4 -> 8 x 8`
- Channel groups: one low-resolution group -> two high-resolution groups
- CKKS levels: `2 -> 1 -> 0`
- Compressed weight transforms: four total
- Result: valid; all acceptance gates pass
- Machine-readable result:
  `.tmp/results/honours/19_wpc_upsample/conv_transpose2d_pipeline.json`

| Metric | Result |
|---|---:|
| Exact compressed-vs-full weight transforms | 4/4 pass |
| Upsampling weight period / compression | 128 / 4.0x |
| Consumer weight period / compression | 8 / 64.0x |
| Aggregate weight payload compression | 5.734x |
| Full weights plus bias | 13,295,616 B |
| Stored weights, metadata, and bias | 2,377,568 B |
| Storage ratio including bias | 5.592x |
| Full / compressed rotations | 84 / 84 |
| Online Python / weight Encode calls | 0 / 0 |
| Maximum upsampling error | `1.07e-8` |
| Maximum final error versus clear | `9.48e-9` |
| Compressed final delta versus full | `0` |
| Peak materialized weight transforms | 1 |

The new plan directly maps low-resolution CIPS slots to high-resolution CIPS
slots and expands ciphertext channel groups as spatial area grows. The output
packing signature is consumed directly by the following convolution, with no
decrypt, clear repack, Encode, or re-encryption. An independent NumPy oracle
and PyTorch `conv_transpose2d` agree with the cyclic-diagonal implementation.

This completes the isolated functional operator boundaries required for a
small WPC encoder/decoder. The next stage composes them into one graph with a
skip connection; trained-model matching and repeated server timing/RSS remain
outstanding.

### 2026-09-29 — Integrated CIPS miniature U-Net

- Executor: seeded local functional validation; Linux server confirmation pending
- Graph: encoder, skip, stride-two downsample, compact reshape, bottleneck,
  identity bootstrap refresh, transposed-convolution upsample, concat, decoder
- Spatial shapes: `8 x 8 -> 4 x 4 -> 8 x 8`
- CKKS levels: `5 -> 4 -> 3 -> 2 -> 1 -> bootstrap(5) -> 4 -> 3 -> 2`
- Compressed learned transforms: 14 total
- Result: valid; all acceptance gates pass
- Machine-readable result:
  `.tmp/results/honours/20_wpc_mini_unet/mini_unet_pipeline.json`

| Metric | Result |
|---|---:|
| Exact compressed-vs-full learned transforms | 14/14 pass |
| Weight-payload compression | 7.771x |
| Weight storage including metadata | 7.702x |
| Overall storage including layout transforms | 7.116x |
| Full / stored weight+bias payload | 68,059,136 / 9,071,920 B |
| Ordinary reshape plus concat Q/P | 573,440 B |
| Full / compressed rotations | 318 / 318 |
| Bootstrap calls per path | 1 |
| Online Python / weight Encode calls | 0 / 0 |
| Maximum intermediate error | `8.37e-9` |
| Final error versus clear | `3.25e-9` |
| Compressed final delta versus full | `0` |
| Peak materialized weight transforms | 1 |

The identity refresh resolves the level mismatch between the retained
level-four skip and the deep decoder branch. It covers all 512 physical CIPS
slots; using the 256-slot logical-value power of two was rejected because
interleaved values in the upper physical slots were lost. The corrected graph
chains every packing signature and level without clear repacking.

This is a deterministic functional graph rather than a trained U-Net22. The
next stage is trained-model parameter and activation/bootstrap integration,
then repeated server O-online versus CIPS-WPC timing and RSS measurement.

### 2026-09-29 — Checkpoint-trained decoder-stage integration

- Executor: seeded local Lattigo functional validation; Linux server
  confirmation pending
- Checkpoint: COVID-19 U-Net22-plus-output base32 Cheb7 fine-tuned model
- Checkpoint SHA-256:
  `761743bc6d700ee3d9fa3d75e6204b099bb0ec46f110ce3607902ae2fe537a59`
- Graph: `up1 + skip1 -> cat1 -> dec1a -> trained dec1a_act -> bootstrap -> dec1b`
- Spatial shapes: `64 x 4 x 4` low branch and `32 x 8 x 8` skip branch
- CKKS levels: `8 -> 7 -> 6 -> 5 -> 1 -> bootstrap(8) -> 7`
- Compressed learned transforms: 56
- Result: valid; every acceptance gate passes
- Machine-readable result:
  `.tmp/results/honours/21_wpc_trained_decoder/trained_decoder_stage.json`

| Metric | Result |
|---|---:|
| Exact compressed-vs-full learned transforms | 56/56 pass |
| Independent Torch-WPC clear-oracle delta | max `1.42e-14` |
| Maximum encrypted error vs WPC clear | `6.14e-7` |
| Compressed final delta vs full | `0` |
| Aggregate learned payload compression | 13.584x |
| Learned storage ratio including bias/metadata | 13.164x |
| Overall ratio including concat Q/P | 12.926x |
| Offline / online weight Encode calls | 56 / 0 |
| Online Python Encode calls | 0 |
| Full / compressed rotations | 1,200 / 1,200 |
| Bootstrap calls per path | 4 |
| Peak materialized weight transforms | 1 |

The exact checkpoint coefficients, independent learned pre/post scales, and
all four trained layer tensors are used without refitting or slicing. The
degree-7 activation consumes four levels and a real four-ciphertext bootstrap
refreshes the result before `dec1b`.

The run also exposes the expected incompatibility between WPC flattened
Rotation Padding and the checkpoint's native zero padding. The seeded maximum
deltas are `0.077914` after `dec1a`, `0.035058` after its activation, and
`0.067241` after `dec1b`. These are layout-semantic differences rather than
FHE error. End-to-end accuracy therefore requires Rotation-Padding-aware
training or fine-tuning before a native-checkpoint accuracy claim is valid.
