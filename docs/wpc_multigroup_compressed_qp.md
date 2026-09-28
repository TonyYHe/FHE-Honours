# WPC multi-group compressed Q/P baseline

## Purpose

This stage extends the validated single-transform WPC mechanism to convolution
channels that do not fit in one ciphertext. It implements a matrix of CIPS
linear transforms, evaluates one transform for every input/output channel-group
pair, and homomorphically accumulates the partial ciphertexts for each output
group.

The experiment is a functional and lifecycle-accounting baseline. It is not a
complete neural network or a performance benchmark.

## Group construction

For `n` CKKS slots and spatial dimensions `H x W`, one CIPS ciphertext has
channel capacity

```text
C = n / (H W).
```

For `Cin` input channels and `Cout` output channels, the numbers of ciphertext
groups are

```text
Gin  = ceil(Cin / C),
Gout = ceil(Cout / C).
```

The implementation compiles `Gout * Gin` transforms. Transform `(o,i)` uses
the weight slice connecting input group `i` to output group `o`. If `x_i` is
the packed ciphertext for input group `i`, output group `o` is

```text
y_o = sum_i LT[o,i](x_i).
```

Each `LT[o,i]` uses the same two-dimensional flattened Rotation Padding and
CIPS slot mapping as the single-transform baseline. The clear correctness
oracle computes the original convolution over global channel indices before
any grouping, so it independently checks weight slicing, local slot indexes,
and cross-group accumulation.

## Compressed online lifecycle

Every group-pair transform stores one exact Q/P evaluation period per active
limb. Online execution is synchronous and sequential:

1. reconstruct the full Q/P plaintext for one transform by coefficient copy;
2. evaluate that transform against its input-group ciphertext;
3. release the reconstructed Q/P polynomials immediately;
4. add the partial ciphertext into its output-group accumulator;
5. continue with the next transform.

Backend-wide counters record both current and peak materialized bytes and
transform counts. The acceptance gate requires current materialization to be
zero before and after every transform, a peak materialized transform count of
one, and peak bytes equal to the largest single transform rather than the sum
of all transforms.

## Default deterministic case

The default case uses `LogN=10`, 512 slots, an `8 x 8` spatial grid, channel
capacity 8, 12 input channels, 12 output channels, and a `3 x 3` kernel. This
gives two input groups, two output groups, and four group-pair transforms.

The validated local result is:

| Metric | Value |
|---|---:|
| Group matrix | 2 output x 2 input |
| Compressed transforms | 4 |
| Encoded Q/P diagonal checks | 318/318 pass |
| Exact full-vs-reconstructed transforms | 4/4 pass |
| Aggregate full Q/P payload | 10,420,224 B |
| Aggregate compressed Q/P payload | 162,816 B |
| Logical metadata | 17,904 B |
| Compressed payload plus metadata | 180,720 B |
| Payload-only compression ratio | 64.0x |
| Compression ratio including metadata | 57.659x |
| Largest single full transform | 3,047,424 B |
| Observed peak materialized payload | 3,047,424 B |
| Peak materialized transforms | 1 |
| Aggregate-full / sequential-peak ratio | 3.419x |
| Offline weight-plaintext Encode calls | 4 |
| Online weight-plaintext Encode calls | 0 |
| Ciphertext accumulation additions per path | 2 |
| Full/compressed LT rotations | 71 / 71 |
| Compressed output delta vs. full path | 0 |
| Maximum error vs. clear reference | `1.81e-7` |
| Materialized bytes after execution | 0 B |

The aggregate-full/peak ratio is not exactly four because the final input and
output groups contain four active channels rather than eight, so the four
transforms have different diagonal counts and full payload sizes.

The result stores diagnostic single-run timings, but they must not be used as
a latency comparison. No warmup, repetition, process-RSS sampling, or matched
server controls are part of this correctness stage.

## Acceptance gates

The result is valid only when all of the following hold:

- clear group-pair evaluation matches the ungrouped global reference;
- every group-pair diagonal message has a proper exact CIPS period;
- every candidate passes exact encoded Q/P reconstruction;
- every compressed transform exactly matches its separately encoded full
  transform;
- full and compressed FHE outputs both match the clear reference;
- full and compressed outputs and operation counters match each other;
- both paths perform the expected ciphertext accumulation additions;
- every transform is released before the next compressed transform executes;
- the global materialization peak is exactly one transform;
- all materialized bytes are zero after execution;
- there is one offline weight Encode per transform and no online weight Encode;
- aggregate compressed storage is smaller than aggregate full storage.

## Implementation

- `orion/experimental/wpc_cips_baseline.py` defines the multi-group case,
  global clear reference, group-pair transform construction, and clear
  accumulation check.
- `orion/backend/lattigo/wpc_compression.go` tracks aggregate resident storage
  plus current and peak materialization across all compressed transforms.
- `orion/backend/lattigo/bindings.py` exposes the global lifecycle counters.
- `tools/run_wpc_cips_multigroup.py` runs the matched full-Q/P and
  compressed-Q/P group matrices.
- `tests/test_wpc_cips_baseline.py` checks a 2 x 2 clear group matrix.
- `orion/backend/lattigo/wpc_periodicity_test.go` checks aggregate registration
  and a sequential one-transform materialization peak.

The machine-readable local result is
`.tmp/results/honours/13_wpc_multigroup/cips_multigroup_compressed_qp.json`.

## Reproduce on the server

```bash
cd ~/FHE-Honours
source .venv/bin/activate

python tools/build_lattigo.py
python -m pytest -q \
  tests/test_wpc_cips_baseline.py \
  tests/test_wpc_periodicity.py

(
  cd orion/backend/lattigo
  go test -run 'TestWPC' -count=1
)

mkdir -p .tmp/results/honours/13_wpc_multigroup
python tools/run_wpc_cips_multigroup.py \
  --out .tmp/results/honours/13_wpc_multigroup/cips_multigroup_server.json \
  > .tmp/results/honours/13_wpc_multigroup/cips_multigroup_server.log 2>&1
```

A valid run exits with status zero and reports `status: ok`, all acceptance
fields true, zero current materialized bytes, and one peak materialized
transform.

## Remaining boundary

This stage establishes a real multi-ciphertext compressed-transform lifecycle,
but it does not yet establish model-level WPC performance or accuracy. The
mechanism has subsequently been integrated with chained actual Orion Conv2d
layers, including bias and CKKS level consumption; see
`docs/wpc_cnn_layer_pipeline.md`. Remaining work includes activation/bootstrap
integration, downsampling reshaping, process-RSS and repeated-latency
measurement, and trained-model evaluation under Rotation Padding.
