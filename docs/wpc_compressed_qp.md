# WPC compressed Q/P storage and online decompression

## Scope

This stage implements a real compressed Lattigo plaintext lifecycle for the
single-ciphertext `3 x 3` CIPS baseline. It replaces the earlier
reconstructibility-only check with an executable path that stores compressed
Q/P data, reconstructs full plaintext polynomials online without Encode,
evaluates the linear transform, and immediately releases the materialization.

The mechanism has subsequently been extended to a matrix of input/output
ciphertext channel groups with backend-wide peak-materialization accounting.
See `docs/wpc_multigroup_compressed_qp.md`.
It is also integrated with chained opt-in Orion Conv2d layers, including bias
and CKKS level consumption; see `docs/wpc_cnn_layer_pipeline.md`.

It remains a mechanism-level correctness experiment. Timings are single-run
diagnostics, not model-level performance results.

## Representation

Let a CIPS weight message contain `n` slots with exact slot period `T`. WPC's
Periodic Transmit property gives an evaluation-representation period `2T` in a
ring polynomial of degree `N = 2n`. For each active Q or P limb, the backend
therefore stores only `2T` `uint64` representatives rather than all `N`
coefficients.

For `D` active Q/P limbs, the payload sizes per diagonal are

```text
full_bytes       = D * N  * 8
compressed_bytes = D * 2T * 8
payload_ratio    = N / (2T) = n / T.
```

The stored transform also retains explicit metadata: schema version, ring
degree, diagonal count, diagonal key, slot and evaluation periods, Q/P levels,
and Q/P limb counts. Reported metadata bytes are a stable logical serialized
accounting and exclude Go allocator and map overhead.

## Lifecycle

### Offline

1. Encode the CIPS diagonals normally once.
2. Verify that every active Q/P limb follows Lattigo's expected bit-reversed
   evaluation-block copy map.
3. Copy one evaluation period from every limb into compressed storage.
4. Replace every full transform polynomial with an empty `ringqp.Poly`.

### Online

1. Allocate each full Q/P polynomial.
2. Reconstruct every coefficient by copying its stored representative; do not
   invoke CKKS Encode, IDFT, or NTT.
3. Evaluate the ordinary Lattigo linear transform.
4. Replace the reconstructed polynomials with empty values before returning.

The compressed representatives remain resident throughout. The full payload
is transient during one evaluation and is reported as zero resident bytes
after the online call.

## Correctness gates

The runner accepts the compressed path only when all of these hold:

- every reconstructed Q/P coefficient exactly matches a separately encoded
  full CIPS transform;
- all 63 compressed diagonals are reconstructed;
- the transient materialization equals the expected full-payload byte count;
- the explicit release returns materialized bytes to zero;
- compressed and full CIPS decrypted outputs match exactly in the default run;
- the compressed output matches the direct clear reference within `1e-6`;
- compressed and full CIPS homomorphic operation counters are identical;
- the backend reports one offline weight-plaintext Encode and zero online
  weight-plaintext Encode calls;
- the online evaluator leaves zero full-payload bytes materialized;
- compressed payload bytes are strictly below full payload bytes.

## Default local result

The deterministic default uses `LogN=10`, 512 slots, four active Q/P limbs,
an `8 x 8` spatial grid, four input/output channels, a `3 x 3` kernel, CIPS
period `T=8`, and 63 encoded diagonals.

| Metric | Value |
|---|---:|
| Full Q/P payload | 2,064,384 B |
| Compressed Q/P payload | 32,256 B |
| Logical metadata | 3,552 B |
| Compressed payload plus metadata | 35,808 B |
| Payload-only compression ratio | 64.0x |
| Compression ratio including metadata | 57.651x |
| Exact Q/P comparison | pass |
| Offline weight-plaintext Encode calls | 1 |
| Online weight-plaintext Encode calls | 0 |
| Full CIPS rotations | 17 |
| Compressed CIPS rotations | 17 |
| Decrypted output delta vs. full CIPS | 0 |
| Maximum error vs. clear reference | `2.04e-7` |
| Materialized bytes after evaluation | 0 B |

The local run measured approximately `0.24 ms` for coefficient-copy
decompression and `1.64 ms` for evaluation, but these values are retained only
as diagnostic evidence. They were collected from one small run without warmup
or repetition.

## Implementation

- `orion/backend/lattigo/wpc_compression.go` owns compressed storage,
  reconstruction, evaluation, release, and statistics.
- `orion/backend/lattigo/bindings.py` exposes the optional real-Lattigo ABI.
- `orion/backend/lattigo/wpc_periodicity_test.go` tests exact compression,
  reconstruction, byte accounting, and invalid-period rejection.
- `tools/run_wpc_cips_baseline.py` performs the three-way full-control,
  full-CIPS, and compressed-CIPS comparison.
- `tools/run_wpc_cips_multigroup.py` applies the same storage lifecycle to
  multiple input/output ciphertext groups and accumulates partial outputs.

Compressed state is removed with its transform and cleared on scheme deletion,
so it does not outlive the Lattigo objects it references.

## Reproduce on the server

After pulling the implementing commit:

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

mkdir -p .tmp/results/honours/12_wpc_compressed_qp
python tools/run_wpc_cips_baseline.py \
  --out .tmp/results/honours/12_wpc_compressed_qp/cips_3x3_compressed_qp_server.json \
  > .tmp/results/honours/12_wpc_compressed_qp/cips_3x3_compressed_qp_server.log \
  2>&1
```

A valid run exits with status zero and reports schema version 3,
`compressed_qp_storage_and_online_decompression_valid: true`, and overall
`valid: true`.

## Limitations

- This specific experiment contains one single-ciphertext transform; the
  separate multi-group baseline covers a transform matrix but is still not a
  complete ResNet, VGG, or U-Net execution.
- It does not implement WPC downsampling reshaping or trained-model accuracy.
- At-rest payload accounting is exact, while metadata accounting is logical;
  neither is a process RSS or Go-heap measurement.
- Online evaluation temporarily materializes one full transform. Peak-memory
  and streaming decompression optimizations remain future work.
