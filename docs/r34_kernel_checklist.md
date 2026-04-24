# R34 Kernel Checklist

This checklist tracks the deterministic R34 kernel flow we want in Orion.

## Fixed policy

- [x] Kernel selection is driven by `source_group_count = ceil(C_in / gap^2)`.
- [x] Channel-first split, then height split only if one channel partition still does not fit.
- [x] Non-`1x1` convolutions use fixed halo rows.
- [x] `source_group_count > 1` selects `inter_group_hybrid`.
- [x] `source_group_count == 1` selects `intra_group_pack2`.

## Step 1: Unified geometry helpers

- [x] Add reusable helpers for:
  - source group counting
  - channel partitions
  - target/source stripe derivation
  - fixed halo metadata
- [x] Use the same helpers for stage1/2 inter-group prototypes.
- [x] Re-run Python backend correctness after refactor.

## Step 2: Stage1/2 inter-group correctness oracle

- [x] Bridge HaloED direct inter-group plans into Orion's Python backend.
- [x] Build a generalized inter-group/inter-hsplit Python prototype for R34 stage1/2.
- [x] Verify stage1/2 full prototype parity on Python backend.
- [x] Re-enable stage1/2 runtime execution in `R34CompileRegistry` for the Python backend.
- [ ] Re-enable stage1/2 runtime execution on non-Python backends.

## Step 3: Stage3/4 pack2 path stability

- [ ] Keep stage3/4 pack2 path on the same geometry helpers.
- [ ] Re-run Python backend pack2 correctness after geometry changes.

## Step 4: Transition integration

- [x] Stage2/3 transition follows the inter-group path on the Python backend.
- [x] Stage4 transition follows the intra-group path on the Python backend.
- [x] Keep halo handling fixed and simple.
- [ ] Re-enable transition runtime on non-Python backends.

## Current Python correctness status

- [x] HaloED direct inter-group bridge test passes in Orion Python backend.
- [x] R34 stage1 generalized inter-group prototype is parity-clean on Python backend.
- [x] R34 stage2 generalized inter-group prototype is parity-clean on Python backend.
- [x] Stage1/2 Python runtime glue is enabled in `r34_phase1`.
- [x] Stage2/4 transition Python runtime glue is enabled in `r34_phase1`.
- [ ] Stage1/2 non-Python runtime glue is still falling back to Orion conv.
- [ ] Stage2/4 transition non-Python runtime glue is still falling back to Orion conv.
