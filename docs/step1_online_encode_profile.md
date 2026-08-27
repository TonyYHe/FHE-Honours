# Step 1: online Encode profiling

## Status

The profiling workflow is implemented and unit-tested. A real-FHE result is intentionally not included because the requested measurement will be run on the server.

Step 1 is valid only when all of the following are true:

- backend: real Lattigo FHE;
- `ORION_LATTIGO_CLEAR_BACKEND=0`;
- `ORION_SINGLE_SLOT_LAYER_CACHE=1`;
- `ORION_LATTIGO_STREAMING_LT=0`;
- `ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT=0`;
- result field `step1_online_encode_profile.valid=true`.

This follows the intended split: clear-Lattigo was the Step 0 correctness backend; Step 1 measures the real-FHE path. The old chunk-streaming path is not enabled for this experiment.

## Measurement boundary

The denominator is the wall-clock duration of one encrypted model forward pass, `he_forward_s`. It excludes clear inference, scheme setup, compilation, input encryption, and output decryption/decoding.

`online_encode_s` is the sum of `lt_layer_cache_encode_s` inside that encrypted forward. It covers materializing the current layer's diagonal plaintexts and encoding them for the Lattigo linear transform. Key preparation and eviction are measured separately.

The result also reports:

- major wall-time categories: MVM kernel, activation excluding bootstrap, bootstrap, online Encode, layer-cache key preparation and eviction, wrapper/executor work, and unattributed wall time;
- Lattigo linear-transform operation counts;
- Lattigo linear-transform microtimers for fused multiplication/accumulation and rotations;
- Orion module wall timers for explicit elementwise Add and Multiply modules.

The Lattigo kernel uses fused multiply-accumulate operations. It is therefore incorrect to claim separate primitive multiplication and fused-add times. The result names this field `lt_fused_multiply_accumulate`. Microtimers can be nested or can represent parallel work, so their percentages are diagnostic and must not be summed. The major wall-time categories are the appropriate view for the overall online Encode proportion.

## Server commands

Run from the `orion` repository root after building the real Lattigo backend and installing the Python dependencies.

Build or rebuild the server's real backend from this checkout so the profiling symbols match the Python runner:

```bash
python tools/build_lattigo.py
```

First inspect the exact configuration without running FHE:

```bash
python tools/run_step1_online_encode_profile.py \
  --network resnet20_cifar10 \
  --mode dense \
  --encode-workers 1 \
  --dry-run
```

Run a small real-FHE baseline:

```bash
python tools/run_step1_online_encode_profile.py \
  --network resnet20_cifar10 \
  --mode dense \
  --encode-workers 1 \
  --warmup-runs 1 \
  --forward-runs 3 \
  --out .tmp/results/honours/03_step1_online_encode/resnet20_cifar10_dense.json
```

Run the U-Net22 provider path when the server has enough memory:

```bash
python tools/run_step1_online_encode_profile.py \
  --network u22_64_base32 \
  --mode provider \
  --encode-workers 1 \
  --warmup-runs 1 \
  --forward-runs 3 \
  --trace-forward-memory \
  --out .tmp/results/honours/03_step1_online_encode/u22_64_base32_provider.json
```

Use the server's virtual-environment Python if `python` is not the correct interpreter. `--encode-workers` directly affects online Encode latency; record it and keep it fixed when comparing models or implementations. A later scaling experiment can repeat the same command with another worker count.

The launcher always passes these runner options:

```text
--backend lattigo
--io-mode none
--profile-modules
--profile-lt
--operator-breakdown
```

It also overrides any inherited clear/streaming variables in the child process with the valid Step 1 values listed above. It does not modify repository-wide environment defaults.

## Reading and accepting a result

The launcher prints a compact summary after the run and exits with code 2 if Step 1 validation fails. The canonical mean across measured attempts is:

```text
step1_online_encode_profile
```

The main values for the mentor discussion are:

```text
step1_online_encode_profile.he_forward_s
step1_online_encode_profile.online_encode_s
step1_online_encode_profile.online_encode_pct_of_he_forward
step1_online_encode_profile.major_wall_categories
step1_online_encode_profile.operator_microprofile
step1_online_encode_profile.operation_counts_mean_per_forward
```

Each `forward_attempts[*].step1_online_encode_profile` contains the corresponding single-pass record. Accept the experiment only if:

1. the runner exits successfully and the result has `status="ok"`;
2. `step1_online_encode_profile.valid` is `true`;
3. its runtime mode is `single_slot_layer_cache`;
4. its requested, successful measured-attempt, and profile counts all agree;
5. online Encode time is positive;
6. model correctness fields such as `mae_vs_clear.shape_match` remain successful.

## Changes made for Step 1

- `tools/run_lattigo_e2e_compare.py`
  - records the existing backend `GetLinearTransformEvaluationProfileSeconds` values with stable field names;
  - creates a validated per-forward Step 1 summary;
  - creates an arithmetic-mean summary across successful measured forwards;
  - reports online Encode seconds and percentage, major wall categories, operator microtimers, and operation counts;
  - rejects clear-Lattigo, legacy streaming, a missing LT profile, a missing operator breakdown, and a zero Encode interval as invalid Step 1 data.
- `tools/run_step1_online_encode_profile.py`
  - provides the focused server launcher and dry-run mode;
  - forces the real-FHE single-slot configuration only in the child environment;
  - runs warmup and measured forwards with all required profilers;
  - validates the output and prints the fields needed for discussion.
- `orion/core/packing.py` and `orion/nn/linear.py`
  - add index-only single-slot metadata for fully connected `Linear` layers;
  - add block-level runtime diagonal/payload reconstruction, matching the existing Conv2d and ConvTranspose2d design;
  - fix ResNet20 compilation failing at the final hybrid-packed classifier with “requires diagonal-index metadata and a runtime diagonal recipe”.
- `tests/test_step1_online_encode_profile.py`
  - tests backend timer naming, percentage calculations, rotation aggregation, invalid-mode rejection, measured-run averaging, and launcher configuration.
- `tests/test_dense_baseline_compile.py`
  - verifies Linear index metadata against full packing for square/hybrid layouts;
  - verifies block reconstruction and that compile-time single-slot setup retains no plaintext diagonal payloads.
- `FHE_compression.md`
  - records the Step 1 status, measurement definition, server command, acceptance gate, and this detailed guide.

No real-FHE timing was fabricated or inferred locally, and no global `ORION_LATTIGO_STREAMING_LT` default was changed.

## Troubleshooting history

The first real-FHE server attempt reached compilation but stopped at the final ResNet20 `linear` layer because it lacked a single-slot runtime recipe. That gap is fixed by the Linear packing changes listed above. A clear-Lattigo structural run with single-slot caching now compiles and completes the same ResNet20 path with `status="ok"`, one successful forward, and matching output shape. This structural run is not accepted as Step 1 timing because the Step 1 validator correctly rejects the clear backend.
