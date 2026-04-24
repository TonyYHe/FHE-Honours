# R34 Stem / Pool TODO

Scope: extend the current R34 `channel-first + height-split + fixed-halo + policy`
flow so `conv1`, `pool`, and `avgpool` no longer depend on Orion dense-conv
fallbacks.

## Families

1. `stem_conv7x7_s2_gap1_to2`
   - Node: `conv1`
   - Policy: `inter_group_hybrid`
   - Python phase-1 runtime: direct single-node flow executor
   - Final target: real scripts/cir-style non-Python kernel

2. `stem_pool3x3_s2_gap2_to4`
   - Node: `pool`
   - Policy: `inter_group_hybrid`
   - Python phase-1 runtime: direct single-node flow executor
   - Final target: depthwise/pool materializer with the same geometry rules

3. `global_avgpool_exit_gap32_to_dense`
   - Node: `avgpool`
   - Policy metadata: `intra_group_pack2`
   - Python phase-1 runtime: direct single-node flow executor that exits back
     to the standard Orion `avgpool -> flatten -> linear` packed layout
   - Final target: explicit layout-exit reduction runtime

## Implementation order

1. Register imported layout contracts and kernel bindings for the three families.
2. Add a Python-only single-node flow runtime for conv/pool/avgpool.
3. Attach the three families in `R34CompileRegistry`.
4. Add focused Python-backend correctness tests for:
   - `conv1`
   - `pool`
   - `avgpool`
5. Re-run the R34 Python correctness suite.
