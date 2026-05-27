# Agents Guide

This repository is currently on the U22 Orion streaming vs HaloED mainline.
Use this file as the top-level orientation for future agents.

## Current Source of Truth

Start here:

- `docs/u22_orion_streaming_haloed_mainline.md`

The mainline is:

1. Prove dense resident Orion is memory-prohibitive on exact U22+output worst layers.
2. Build a strong Orion dense streaming LT baseline: multicore compile, BSGS preserved inside streamed chunks, compute time reported layer-by-layer with total I/O excluded.
3. Compare that baseline against HaloED provider layer-by-layer with the same reporting convention.
4. Keep RAM and feasibility evidence separate from compute-time comparisons.

## Exact U22 Model

Use exact U22 body plus explicit output:

- `dec1b`: `64 -> 64`, `3x3/pad1`
- `output`: `64 -> 1`, `1x1/pad0`
- Base dimension: `64`
- Primary image size for the current gate: `256x256`
- Lattigo profile: `e2e`, `LogN=16`

Do not use the old stock U22 interpretation where `dec1b` is `64 -> 1`.

## Active Corg Queue

Dense resident RAM gate results are generated on `corg`:

```bash
cd ~/orion
bash .tmp/run_u22_exact_ram_gate_corg.sh
```

Remote result files:

```text
~/orion/.tmp/results/u22_exact_base64_dense_*_ram_gate_corg.json
~/orion/.tmp/results/u22_exact_base64_dense_ram_gate_logs/queue.log
```

Refresh the local docs table with:

```bash
rsync -az --checksum 'corg:/home/qihan/orion/.tmp/results/u22_exact_base64_dense_*_ram_gate_corg.json' .tmp/results/
.venv/bin/python tools/update_u22_mainline_doc.py --doc docs/u22_orion_streaming_haloed_mainline.md --result-dir .tmp/results
```

## What Not To Follow

Do not treat older kernel-rewrite campaign instructions as the default workflow
for this mainline. In particular, do not start with old banked regression,
kernel-change-verification, worker handoff, R18/R34 stage probes, or bootstrap
handoff microbench workflows unless the current task explicitly modifies those
kernel/provider code paths.

For ordinary mainline work, prefer:

- Updating the mainline doc and its tables.
- Running the exact U22 RAM gate or streaming layer-by-layer experiments.
- Recording result JSON paths and peak RSS / compile / compute timings.

If you actually change backend kernels, provider executors, slot mapping, or
planner internals, then run the relevant targeted verification for that code
change. Keep that separate from the U22 experiment bookkeeping.

## Reporting Rules

For the comparison tables:

- Sum layer compute time only.
- Exclude total I/O time from the primary comparison.
- Report compile time separately.
- Report peak RSS separately.
- State whether BSGS/sharing is preserved within each streamed chunk.
- Record the chunk/budget setting for Orion dense streaming.

For dense resident infeasibility:

- `rss_cap_exceeded` under the 900 GiB cap is a valid feasibility certificate.
- Sequential layer replay RAM is the maximum single-layer peak.
- All-resident dense cache RAM is a different, stronger memory model and should
  not be mixed into the compute-time comparison.
