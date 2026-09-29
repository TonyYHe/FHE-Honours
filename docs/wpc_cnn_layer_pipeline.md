# WPC CIPS integration with Orion Conv2d

## Purpose

This stage integrates the validated WPC compressed-Q/P mechanism with actual
`orion.nn.Conv2d` layers. It adds an explicit, opt-in CIPS planner while
leaving Orion's existing dense and provider planners unchanged. The planner
extracts the layer's real weights and bias, compiles a compressed transform
matrix, accepts grouped encrypted CIPS inputs, and returns grouped outputs that
can feed the next compatible WPC layer without repacking.

The validation target is a two-layer convolution block. This is the first
stage that exercises the installed `Conv2d.forward` path rather than only a
standalone transform runner.

## Supported layer subset

The current planner accepts:

- batch size one;
- two-dimensional convolutions with odd kernels;
- stride one and dilation one;
- `groups=1`;
- symmetric same-shape padding;
- power-of-two height and width;
- one or more ciphertext channel groups.

The FHE semantics use WPC flattened Rotation Padding, not PyTorch zero
padding. Training or fine-tuning under Rotation Padding remains necessary for
model-accuracy claims.

## Planner and execution

For slot capacity `n`, spatial shape `H x W`, and CIPS channel capacity

```text
C = n / (H W),
```

the planner divides input and output channels into groups of at most `C` and
compiles one compressed transform for each input/output group pair. It obtains
weights from the installed Orion layer's `on_weight` and obtains the fused
bias from `on_bias`.

Each layer executes:

```text
for each output group o:
    y_o = sum_i DecompressEvaluate(LT[o,i], x_i)
    y_o = Rescale(y_o)
    y_o = y_o + encoded_bias[o]
```

`DecompressEvaluate` reconstructs one full Q/P transform, evaluates it, and
releases it before returning. The layer consumes one CKKS level. Its output
retains a CIPS packing signature containing the slot count, spatial shape,
channel count, and group ranges. A following layer accepts the ciphertext only
when that signature and the expected CKKS level match.

Bias messages are encoded offline and kept as ordinary Q plaintexts. They are
not yet WPC-compressed, so the total layer storage ratio is reported both for
compressed weight transforms and with this uncompressed bias cost included.

## Default two-layer validation

The deterministic validation uses two real Orion `Conv2d(12,12,3)` layers on
an `8 x 8` spatial grid with bias enabled. `LogN=10` provides 512 slots and a
channel capacity of eight, so each layer has a `2 x 2` transform matrix. The
layers consume levels `2 -> 1 -> 0`.

| Metric | Result |
|---|---:|
| Orion Conv2d layers | 2 |
| Compressed group-pair transforms | 8 |
| Exact Q/P transform checks | 8/8 pass |
| Aggregate full weight Q/P payload | 18,235,392 B |
| Aggregate compressed weight payload | 284,928 B |
| Weight-transform metadata | 35,808 B |
| Uncompressed bias Q payload | 49,152 B |
| Full weights plus bias | 18,284,544 B |
| Stored weights, metadata, and bias | 369,888 B |
| Weight payload compression ratio | 64.0x |
| Weight ratio including metadata | 56.855x |
| Layer plaintext ratio including bias | 49.433x |
| Largest single full transform | 3,047,424 B |
| Observed peak materialized payload | 3,047,424 B |
| Peak materialized transforms | 1 |
| Aggregate full weight / sequential peak | 5.984x |
| Offline / online weight Encode calls | 8 / 0 |
| Online Python Encode calls during both convolutions | 0 |
| Full / compressed rotations | 142 / 142 |
| Ciphertext accumulation additions | 4 per path |
| Bias plaintext additions | 4 per path |
| Compressed output delta vs. full control | 0 |
| Maximum two-layer error vs. clear reference | `3.11e-8` |
| Materialized payload after the block | 0 B |

The local result satisfies every acceptance gate. The recorded evaluation
times are single diagnostic observations without warmup or repetitions and
must not be interpreted as a speed comparison.

## Correctness gates

The result is accepted only when:

- both layers are actual `orion.nn.Conv2d` instances with installed plans;
- the first output packing signature exactly matches the second input;
- all eight compressed transforms exactly match separately encoded full Q/P;
- both full-control and compressed two-layer outputs match the independent
  Rotation-Padding clear reference;
- the compressed output exactly equals the full-control output;
- operation, ciphertext-accumulation, and bias-addition counts match;
- the CKKS level chain is exactly `2 -> 1 -> 0`;
- no Python Encode call occurs during either online convolution;
- each compressed transform is released before the next begins;
- backend peak materialization is exactly one transform;
- all current materialized bytes are zero after execution;
- aggregate storage and Encode-call accounting close exactly.

## Implementation

- `orion/experimental/wpc_cips_layer.py` implements layer validation,
  transform planning, compilation, grouped input encryption, sequential
  execution, bias addition, signature/level checks, decoding, and cleanup.
- `orion/nn/linear.py` exposes `install_wpc_cips_plan` and dispatches an
  installed plan from `Conv2d.forward` only in HE mode.
- `tools/run_wpc_cips_layer_pipeline.py` validates a two-layer full-control and
  compressed block.
- `tests/test_wpc_cips_layer.py` covers supported geometry, rejection gates,
  chainable packing signatures, and opt-in forward dispatch.
- `.tmp/results/honours/14_wpc_cnn_layer_pipeline/`
  `two_conv_cips_compressed_qp.json` is the deterministic local evidence.

## Reproduce on the server

```bash
cd ~/FHE-Honours
source .venv/bin/activate

python tools/build_lattigo.py
python -m pytest -q \
  tests/test_wpc_cips_layer.py \
  tests/test_wpc_cips_baseline.py \
  tests/test_wpc_periodicity.py

(
  cd orion/backend/lattigo
  go test ./...
)

mkdir -p .tmp/results/honours/14_wpc_cnn_layer_pipeline
python tools/run_wpc_cips_layer_pipeline.py \
  --out .tmp/results/honours/14_wpc_cnn_layer_pipeline/two_conv_server.json \
  > .tmp/results/honours/14_wpc_cnn_layer_pipeline/two_conv_server.log 2>&1
```

A valid run exits with status zero and reports every acceptance field as true.

## Remaining boundary

This implementation proves layer planning, same-shape ciphertext handoff, bias
addition, and level chaining. The subsequent isolated full-vs-compressed
resource benchmark adds process-RSS and repeated-latency measurement without
co-resident control transforms; see `docs/wpc_isolated_resource_benchmark.md`.
The following stage now adds a real quadratic activation and Lattigo bootstrap
between compatible CIPS convolutions; see
`docs/wpc_cips_activation_bootstrap.md`. It is not yet a complete WPC network
runtime. The remaining required features are stride-two downsampling
reshaping, residual/concatenation layout handling, higher-degree trained-model
activations, and trained-model Rotation-Padding accuracy.
