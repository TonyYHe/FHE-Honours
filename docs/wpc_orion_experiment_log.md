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
