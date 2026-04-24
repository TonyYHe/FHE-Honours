# R18 E2E Region-First TODO

This document tracks the `R18 tiny` region-first end-to-end rollout status.
The stage4 handoff, transition bridges, and stem bridge are now implemented in
the e2e registry. The remaining high-level gap is that the final classifier
still uses the ordinary Orion linear path, and no end-to-end Lattigo speedup
claim has been published yet.

## Persistent Layout Plan

- Stage 1 persistent layout: `8` real ciphertexts
- Stage 2 persistent layout: `4` real ciphertexts
- Stage 3 persistent layout: `2` real ciphertexts
- Stage 4 persistent layout: `1` real ciphertext

Kernel-internal policy:

- Multiple source groups: hybrid-pack adjacent groups into `real + imag`
- Single source group: intra-group packing, put half the channels into complex
  lanes, and reserve any remaining space for future GS scratch experiments
- Activation / residual boundaries: always return to the persistent real layout

## Current Coverage

Provider-backed end-to-end nodes in `for_r18_tiny_e2e(...)`:

- Stem bridge:
  - `conv1`
- Stage 1 same-shape full-conv:
  - `layers_0_0_conv1`
  - `layers_0_0_conv2`
  - `layers_0_1_conv1`
  - `layers_0_1_conv2`
- Stage 1 -> 2 transition bridge:
  - `layers_1_0_conv1`
  - `layers_1_0_shortcut_0`
- Stage 2 same-shape full-conv:
  - `layers_1_0_conv2`
  - `layers_1_1_conv1`
  - `layers_1_1_conv2`
- Stage 2 -> 3 transition bridge:
  - `layers_2_0_conv1`
  - `layers_2_0_shortcut_0`
- Stage 3 same-shape full-conv:
  - `layers_2_0_conv2`
  - `layers_2_1_conv1`
  - `layers_2_1_conv2`
- Stage 3 -> 4 transition bridge:
  - `layers_3_0_conv1`
  - `layers_3_0_shortcut_0`
- Stage 4 same-shape full-conv:
  - `layers_3_0_conv2`
  - `layers_3_1_conv1`
  - `layers_3_1_conv2`

The main remaining dense/original Orion path is:

- Final classifier: `linear`

## Ordered Implementation Plan

1. Stage 4 full-conv handoff
   Status: completed
   Result: the e2e registry now accepts stage4 same-shape nodes and performs
   regular-stage4 -> compact-intra prepack inside the full-conv executor.

2. Transition main + shortcut bridge
   Status: completed
   Result: stage boundary bridge executors now connect
   `stage1 -> stage2`, `stage2 -> stage3`, and `stage3 -> stage4` while keeping
   the persistent real stage layouts (`8 -> 4 -> 2 -> 1` ciphertexts).

3. Stem bridge
   Status: completed
   Result: `conv1` now writes directly into the stage1 persistent layout rather
   than falling back to the ordinary dense conv path.

4. Final classifier and end-to-end benchmark
   Status: pending
   Goal: measure a fair full-network Lattigo benchmark with the new bridge and
   same-shape coverage, and decide whether the final linear layer should remain
   on the ordinary Orion path or receive a region-first bridge.

## Validation Rule

Every newly added kernel/bridge must include a Python-backend correctness test
that compares the new runtime path against an explicit reference execution.
