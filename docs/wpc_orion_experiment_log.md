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

Profiling is gated by `ORION_WPC_PERIODICITY_PROFILE`; disabled execution does
not scan or convert payloads. The audit launcher fixes both streaming flags to
zero because this is a clear census, not a real-FHE streaming run.

## Verification performed locally

```text
python -m pytest -q tests/test_wpc_periodicity.py
23 passed

go test -run 'TestWPC' -count=1
PASS
```

The census launcher dry run also produced the intended clear-Lattigo command
and output paths. No model census or WPC performance experiment was executed
locally, following the decision to perform all experiment runs on the remote
server.

## Remaining validity boundary

The first census reports slot-message periodicity and analytical byte coverage.
Each periodic record remains only a candidate until the actual encoded Q/P
polynomial passes the exact copy-map and reconstruction check. The current
launcher does not apply compression, does not measure decompression, and does
not provide Encode-time-weighted coverage. Those claims require later backend
instrumentation and O-hybrid timing runs.
