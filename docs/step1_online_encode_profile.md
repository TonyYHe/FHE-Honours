# Step 1: online Encode profiling

## Status

The profiling workflow is implemented, unit-tested, and has completed a real-FHE ResNet20 server run. The checked-in result is under `.tmp/results/honours/03_step1_online_encode/`. The VGG16 provider-layout failure found in the first model-matrix run is fixed and passes a complete clear-Lattigo structural forward; its real-FHE profile must be rerun on the server.

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

- additive major wall-time categories: MVM kernel, backend bootstrapping, online Encode, layer-cache key preparation and eviction, wrapper/executor work, and `other_he_forward`;
- Lattigo linear-transform operation counts;
- Lattigo linear-transform microtimers for fused multiplication/accumulation and rotations;
- Orion module wall timers for explicit elementwise Add and Multiply modules.

The Lattigo kernel uses fused multiply-accumulate operations. It is therefore incorrect to claim separate primitive multiplication and fused-add times. The result names this field `lt_fused_multiply_accumulate`. Microtimers can be nested or can represent parallel work, so their percentages are diagnostic ratios and must not be summed.

Schema version 2 makes this distinction machine-readable:

- `major_wall_categories_metadata.additive=true`;
- `major_wall_categories_accounting.valid=true` only when every category is finite and non-negative and the categories close to `he_forward_s`;
- `operator_microprofile_metadata.additive=false`;
- backend bootstrap time comes from `lattigo_bootstrap_profile_after_he_forward.totals.total_s`, whose counter is reset for each forward attempt;
- `other_he_forward` is calculated as the HE-forward wall clock minus the other non-overlapping major categories. It includes activation arithmetic and any remaining uninstrumented wall work; it is not an unknown measurement error.

Do not use the nested activation-module timers or accumulated Lattigo worker microtimers to construct a wall-time pie chart.

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

Rerun the repaired VGG16/base16 provider path for the model matrix:

```bash
python tools/run_step1_online_encode_profile.py \
  --network vgg16_imgnet \
  --mode provider \
  --encode-workers 1 \
  --warmup-runs 1 \
  --forward-runs 3 \
  --out .tmp/results/honours/04_step1_model_matrix/vgg16_imgnet_provider.json
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
step1_online_encode_profile.major_wall_categories_accounting
step1_online_encode_profile.operator_microprofile
step1_online_encode_profile.operation_counts_mean_per_forward
```

Each `forward_attempts[*].step1_online_encode_profile` contains the corresponding single-pass record. Accept the experiment only if:

1. the runner exits successfully and the result has `status="ok"`;
2. `step1_online_encode_profile.valid` is `true`;
3. `schema_version` is at least `2` and `major_wall_categories_accounting.valid` is `true`;
4. the major category sum is 100% of `he_forward_s` within the recorded tolerance;
5. its runtime mode is `single_slot_layer_cache`;
6. its requested, successful measured-attempt, and profile counts all agree;
7. online Encode time is positive;
8. model correctness fields such as `mae_vs_clear.shape_match` remain successful.

## Corrected ResNet20 result

Across the three measured forwards, encrypted inference averaged `1544.9013 s` and online Encode averaged `310.4265 s` (`20.0936%`). The reset backend bootstrap profile averaged `996.2554 s` (`64.4867%`). The earlier schema-v1 summary incorrectly reported bootstrap above 100% because Python-side Bootstrap records accumulated across the warmup and measured attempts; schema version 2 removes that cumulative source from wall accounting and rejects impossible summaries.

The checked-in artifact's canonical per-attempt and mean `step1_online_encode_profile` objects were rebuilt from the stored raw backend counters, so no FHE rerun was needed. Its historical `operator_breakdown_after_forward` captures are retained for auditability but are explicitly marked `schema_version=1` and `additive_wall_accounting=false`; do not use their old cumulative totals.

## Changes made for Step 1

- `tools/run_lattigo_e2e_compare.py`
  - records the existing backend `GetLinearTransformEvaluationProfileSeconds` values with stable field names;
  - creates a validated per-forward Step 1 summary;
  - creates an arithmetic-mean summary across successful measured forwards;
  - reports online Encode seconds and percentage, additive major wall categories, diagnostic operator microtimers, and operation counts;
  - resets Python-side Bootstrap profiles before every forward and uses the independently reset backend bootstrap wall counter in the Step 1 summary;
  - profiles only outermost activation roots and subtracts all descendant Bootstrap intervals, avoiding nested activation double-counting in the raw operator breakdown;
  - validates that major categories are finite, non-negative, and close exactly to the HE-forward wall boundary;
  - rejects clear-Lattigo, legacy streaming, a missing LT/bootstrap profile, a missing operator breakdown, zero Encode time, or impossible wall accounting as invalid Step 1 data.
- `tools/run_step1_online_encode_profile.py`
  - provides the focused server launcher and dry-run mode;
  - forces the real-FHE single-slot configuration only in the child environment;
  - runs warmup and measured forwards with all required profilers;
  - validates the output and prints the fields needed for discussion;
  - rejects obsolete schema-v1 output and any result without valid additive wall accounting.
- `orion/core/packing.py` and `orion/nn/linear.py`
  - add index-only single-slot metadata for fully connected `Linear` layers;
  - add block-level runtime diagonal/payload reconstruction, matching the existing Conv2d and ConvTranspose2d design;
  - fix ResNet20 compilation failing at the final hybrid-packed classifier with “requires diagonal-index metadata and a runtime diagonal recipe”.
- `tests/test_step1_online_encode_profile.py`
  - tests backend timer naming, percentage calculations, rotation aggregation, per-forward Bootstrap reset, backend-bootstrap source selection, negative/missing accounting rejection, nested-activation exclusion, measured-run averaging, and launcher configuration.
- `tests/test_dense_baseline_compile.py`
  - verifies Linear index metadata against full packing for square/hybrid layouts;
  - verifies block reconstruction and that compile-time single-slot setup retains no plaintext diagonal payloads.
- `orion/experimental/layout_policy_ablation.py`, `orion/experimental/u22_phase1.py`, and `orion/core/bootstrap_layout_compression.py`
  - classify the internal ReLU `Mult` and `Identity` nodes as slot-layout preserving;
  - publish each native provider's exact ciphertext count and storage signature into the executable plan;
  - propagate that metadata through a transparent activation chain only when all incoming physical signatures agree;
  - preserve the source signature at the next convolution boundary, so its provider consumes the actual native ciphertext batch and materializes the selected compact output.
- `tests/test_layout_policy_ablation.py` and `tests/test_bootstrap_prescale_cache.py`
  - regress the VGG16 first activation's 32-ciphertext native layout, the signature-aware next-provider boundary, and the Bootstrap prescale vector/mask size.
- `FHE_compression.md`
  - records the Step 1 status, measurement definition, server command, acceptance gate, and this detailed guide.

No real-FHE timing was fabricated or inferred locally, and no global `ORION_LATTIGO_STREAMING_LT` default was changed.

## Troubleshooting history

The first real-FHE server attempt reached compilation but stopped at the final ResNet20 `linear` layer because it lacked a single-slot runtime recipe. That gap is fixed by the Linear packing changes listed above. A clear-Lattigo structural run with single-slot caching now compiles and completes the same ResNet20 path with `status="ok"`, one successful forward, and matching output shape. This structural run is not accepted as Step 1 timing because the Step 1 validator correctly rejects the clear backend.

The first VGG16/base16 provider attempt failed in `features.2.mult1.bootstrapper` with `IndexError: list index out of range`. The provider emitted 32 native-stripe ciphertexts, while the nested ReLU `Mult` node was incorrectly reclassified as logical compact and compiled only 25 prescale plaintexts. `Mult` and `Identity` do not move slots, so the executable layout plan now keeps the 32-row provider signature through the entire ReLU chain. The first Bootstrap consequently compiles a `32 x 32768` active-slot mask, and the following convolution receives a signature-aware native input and materializes its selected compact output.

A full local clear-Lattigo VGG16 structural run now finishes with `status="ok"`, one successful forward, matching `[1, 1000]` output shape, and no Bootstrap batch mismatch. Its first `features.2.mult1.bootstrapper` call records 32 input ciphertexts and 32 output ciphertexts. The audit artifact is `.tmp/results/honours/05_vgg16_profile_fix/vgg16_clear_structural.json`. Its clear-backend timing is diagnostic only and is deliberately rejected by the Step 1 validator; use the server command above for the publishable real-FHE profile.
