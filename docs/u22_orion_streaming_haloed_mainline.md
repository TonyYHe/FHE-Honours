# U22 Orion Dense/Streaming vs HaloED Mainline

This document is the working plan for the current U-Net 22 comparison.

Current active gate, as of May 27, 2026:

- Conv kernel table on `corg:/home/qihan/orion`
- Lattigo kernel profile, `LogN=16`, default resident LT with `io_mode=none`
- `HW=4` means the four current input sizes: `192x192`, `224x224`, `384x288`, `384x384`
- Kernels: `Conv 32,32`, `Conv 64,64`, `Conv 128,128`, each `3x3/pad1/stride1`
- Variants: Orion dense LT, provider with input halo top/bottom `1/1`, provider with input halo top/bottom `2/2`
- Metrics: runtime rotations and `LT+accumulate s`; no streamed encode time in this table
- Execution containment: one worker per table row with an RSS watchdog

The old dim32 encoder dense/provider matrix is provenance for this kernel-table
gate. Reuse only exact-shape, non-stream rows; do not mix rows whose HW, channel
count, path, or halo setting differs. Older provider JSONs without an explicit
halo field are treated as provider halo `1/1`, which is the default provider
setting.

Historical dense-resident infeasibility evidence below used U22 `base_dim=64`.
That evidence is still useful as a memory caution, but it is no longer the
active baseline matrix.

The historical exact base64 model is U22 body plus an explicit output head:

- `dec1b`: `64 -> 64`, `3x3/pad1`
- `output`: `64 -> 1`, `1x1/pad0`
- Lattigo profile: `e2e`, `LogN=16`

## Thesis

The fair comparison path is:

1. For every U22 `base_dim=32` encoder unique Conv2d node, run both Orion dense LT and HaloED/provider with the same reporting convention.
2. Run each node in an isolated worker with an RSS cap and disk watermark so host availability and GPU access for other users are not put at risk.
3. If a non-streaming path exceeds the RSS cap, rerun that exact node/input/path with streamed LT and record the chunk/budget policy.
4. Compare compute time separately from memory feasibility. Do not mix resident dense RAM certificates, streamed fallback timings, and HaloED timings without labeling the path.

Streaming does not mean disabling BSGS.  The intended baseline streams LT artifacts by BSGS group / input-column group / tile-family chunk.  Within each chunk, baby-step reuse and BSGS sharing should be preserved.  Cross-chunk sharing is limited by the memory budget, and that limit must be reported.

## Step 0: Conv Kernel Table

Goal: compare default resident Orion dense LT against HaloED/provider native
halo Conv kernels for the three channel sizes and four HW sizes above.
If a resident row hits the RSS cap or OOMs, record that row as a resident
feasibility failure; do not silently replace it with a streamed result in this
table.

Run command on `corg`:

```bash
cd ~/orion
ts="$(date -u +%Y%m%dT%H%M%SZ)"
GOMAXPROCS="$(nproc)" \
ORION_LATTIGO_STREAMING_LT=0 \
ORION_COMPILE_PARALLEL_POLICY=auto \
PYTHONUNBUFFERED=1 \
nohup .venv/bin/python tools/run_conv_kernel_table.py \
  --run-root ".tmp/results/conv_kernel_table_corg_${ts}" \
  --doc docs/u22_orion_streaming_haloed_mainline.md \
  --reuse-from .tmp/results/u22_dim32_encoder_baseline \
  --max-worker-rss-gb 850 \
  > ".tmp/logs/conv_kernel_table_corg_${ts}.log" 2>&1 &
```

Refresh locally:

```bash
run_root="$(ssh corg 'cd /home/qihan/orion && awk -F= '\''$1=="run_root"{print $2}'\'' .tmp/run/conv_kernel_table.latest')"
base="$(basename "$run_root")"
mkdir -p ".tmp/results/$base"
rsync -az --checksum "corg:/home/qihan/orion/$run_root/" ".tmp/results/$base/"
.venv/bin/python tools/run_conv_kernel_table.py --update-doc-only --run-root ".tmp/results/$base" --doc docs/u22_orion_streaming_haloed_mainline.md
```

<!-- CONV_KERNEL_TABLE_START -->
| HW | kernel | path / beta | status | input halo T/B | rotations | LT+accumulate s | hot run s | compile s | input ct | output ct | peak RSS GiB | runtime mode | result file | note |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 192x192 | Conv 32,32 | Orion dense | ok |  | 15,152 | 1808.4 | 1821.6 | 76.2 | 36 | 36 |  | resident_compute | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_192x192_orion.json | reused: auto_dense_provider_corg_20260526T130718Z.json:u22_192_base32_encoder/enc1b/dense |
| 192x192 | Conv 32,32 | provider halo=1/1 | ok | 1/1 | 5,360 | 699.4 | 700.9 | 235.5 | 48 | 36 |  | unknown | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_192x192_provider_halo1.json | reused: auto_dense_provider_corg_20260526T130718Z.json:u22_192_base32_encoder/enc1b/provider |
| 192x192 | Conv 32,32 | provider halo=2/2 | running | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_192x192_provider_halo2.json |  |
| 224x224 | Conv 32,32 | Orion dense | ok |  | 22,823 | 3136.1 | 3153.1 | 161.5 | 49 | 49 |  | resident_compute | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_224x224_orion.json | reused: auto_dense_provider_corg_20260526T130718Z.json:u22_224_base32_encoder/enc1b/dense |
| 224x224 | Conv 32,32 | provider halo=1/1 | ok | 1/1 | 11,216 | 1301.7 | 1303.7 | 364.6 | 64 | 49 |  | unknown | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_224x224_provider_halo1.json | reused: auto_dense_provider_corg_20260526T130718Z.json:u22_224_base32_encoder/enc1b/provider |
| 224x224 | Conv 32,32 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_224x224_provider_halo2.json |  |
| 384x288 | Conv 32,32 | Orion dense | pending |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_384x288_orion.json |  |
| 384x288 | Conv 32,32 | provider halo=1/1 | pending | 1/1 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_384x288_provider_halo1.json |  |
| 384x288 | Conv 32,32 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_384x288_provider_halo2.json |  |
| 384x384 | Conv 32,32 | Orion dense | pending |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_384x384_orion.json |  |
| 384x384 | Conv 32,32 | provider halo=1/1 | pending | 1/1 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_384x384_provider_halo1.json |  |
| 384x384 | Conv 32,32 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv32_384x384_provider_halo2.json |  |
| 192x192 | Conv 64,64 | Orion dense | pending |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_192x192_orion.json |  |
| 192x192 | Conv 64,64 | provider halo=1/1 | pending | 1/1 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_192x192_provider_halo1.json |  |
| 192x192 | Conv 64,64 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_192x192_provider_halo2.json |  |
| 224x224 | Conv 64,64 | Orion dense | pending |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_224x224_orion.json |  |
| 224x224 | Conv 64,64 | provider halo=1/1 | pending | 1/1 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_224x224_provider_halo1.json |  |
| 224x224 | Conv 64,64 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_224x224_provider_halo2.json |  |
| 384x288 | Conv 64,64 | Orion dense | pending |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_384x288_orion.json |  |
| 384x288 | Conv 64,64 | provider halo=1/1 | pending | 1/1 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_384x288_provider_halo1.json |  |
| 384x288 | Conv 64,64 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_384x288_provider_halo2.json |  |
| 384x384 | Conv 64,64 | Orion dense | pending |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_384x384_orion.json |  |
| 384x384 | Conv 64,64 | provider halo=1/1 | pending | 1/1 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_384x384_provider_halo1.json |  |
| 384x384 | Conv 64,64 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv64_384x384_provider_halo2.json |  |
| 192x192 | Conv 128,128 | Orion dense | pending |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_192x192_orion.json |  |
| 192x192 | Conv 128,128 | provider halo=1/1 | pending | 1/1 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_192x192_provider_halo1.json |  |
| 192x192 | Conv 128,128 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_192x192_provider_halo2.json |  |
| 224x224 | Conv 128,128 | Orion dense | pending |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_224x224_orion.json |  |
| 224x224 | Conv 128,128 | provider halo=1/1 | pending | 1/1 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_224x224_provider_halo1.json |  |
| 224x224 | Conv 128,128 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_224x224_provider_halo2.json |  |
| 384x288 | Conv 128,128 | Orion dense | pending |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_384x288_orion.json |  |
| 384x288 | Conv 128,128 | provider halo=1/1 | pending | 1/1 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_384x288_provider_halo1.json |  |
| 384x288 | Conv 128,128 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_384x288_provider_halo2.json |  |
| 384x384 | Conv 128,128 | Orion dense | pending |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_384x384_orion.json |  |
| 384x384 | Conv 128,128 | provider halo=1/1 | pending | 1/1 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_384x384_provider_halo1.json |  |
| 384x384 | Conv 128,128 | provider halo=2/2 | pending | 2/2 |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_corg_20260527T020548Z/rows/conv128_384x384_provider_halo2.json |  |
<!-- CONV_KERNEL_TABLE_END -->

## Step 1: Dim32 Encoder Baseline

Goal: compare Orion dense LT and HaloED/provider on all U22 `base_dim=32` encoder unique Conv2d nodes for inputs `192x192`, `224x224`, `384x288`, and `384x384`.

Run policy:

- Prefer `corg` for the full non-streaming run because memory fit is the deciding factor.
- On `qiuchu` or other shared GPU hosts, use Docker `--memory` and `--memory-swap` caps in addition to the Python worker RSS watchdog.
- The Python worker must use an RSS watchdog, so a worker is terminated before the host is at risk.
- Use `io_mode=none`; do not write dense encoded plaintext caches to SSD.
- Start with `ORION_LATTIGO_STREAMING_LT=0`.
- Run both `--paths dense` and `--paths provider`; the comparison row is incomplete until both paths have a result or a labeled failure.
- On `rss_watermark`, OOM, or similar memory failure, rerun only the failed node/input/path with `ORION_LATTIGO_STREAMING_LT=force`.
- Record peak RSS, compile time, compute time, rotations, ciphertext counts, whether streaming was used, and exact result JSON path for both paths.

Active runs:

- Host/path: `corg:/home/qihan/orion`
- Auto dense/provider non-stream JSON: `.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json`
- Auto dense/provider non-stream log: `.tmp/logs/u22_dim32_encoder_auto_dense_provider_20260526T130718Z.log`
- Stream fallback watcher: `.tmp/logs/u22_dim32_encoder_auto_stream_fallback_watcher_20260526T130747Z.log`
- Superseded diagnostic dense-only manual-cap JSON: `.tmp/results/u22_dim32_encoder_baseline/superseded/dense_nonstream_corg_20260526T123808Z.json`
- Superseded auto attempt before encoder Conv provider attach fix: `.tmp/results/u22_dim32_encoder_baseline/superseded/auto_dense_provider_corg_20260526T125833Z.json`
- Worker RSS cap: `820 GiB`
- Compile/pack worker policy: `ORION_COMPILE_PARALLEL_POLICY=auto`; no explicit `ORION_PACK_CONV_WORKERS`, `ORION_LT_COMPILE_WORKERS`, `ORION_LATTIGO_COMPILE_WORKERS`, `ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS`, or `ORION_DENSE_LT_COMPILE_BATCH_TRANSFORMS` caps in the active run.
- Auto policy note: without explicit worker caps, the code estimates compile workers at roughly one worker per 24 GiB of memory budget, pack workers at one per 8 GiB, and diagonal encode workers at one per 4 GiB. On `corg`, this can select tens of workers; the 820 GiB worker RSS watchdog is the safety boundary.

Local refresh command:

```bash
rsync -az --checksum 'corg:/home/qihan/orion/.tmp/results/u22_dim32_encoder_baseline/*.json' .tmp/results/u22_dim32_encoder_baseline/
.venv/bin/python tools/update_u22_dim32_encoder_doc.py --doc docs/u22_orion_streaming_haloed_mainline.md --result-dir .tmp/results/u22_dim32_encoder_baseline
```

<!-- U22_DIM32_ENCODER_BASELINE_TABLE_START -->
| input | node | dense status | provider status | input -> output | dense RSS GiB | provider RSS GiB | dense compile s | provider compile s | dense compute s | provider compute s | dense/provider compute | dense rotations | provider rotations | dense ct | provider ct | dense stream | provider stream | result files | note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 192x192 | enc1a | ok | ok | 1x3x192x192 -> 1x32x192x192 | 37.8 | 36.7 | 24.2 | 54.3 | 201.8 | 47.8 | 4.22 | 1716 | 245 | 4->36 | 6->48 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 192x192 | enc1b | ok | ok | 1x32x192x192 -> 1x32x192x192 | 302.1 | 403.3 | 76.2 | 235.5 | 1808.4 | 700.9 | 2.58 | 16448 | 4720 | 36->36 | 48->36 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 192x192 | enc2a | ok | ok | 1x32x96x96 -> 1x64x96x96 | 207.6 | 340.0 | 72.4 | 133.7 | 612.5 | 421.4 | 1.45 | 5024 | 2121 | 9->18 | 10->19 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 192x192 | enc2b | ok | ok | 1x64x96x96 -> 1x64x96x96 | 411.0 | 478.2 | 113.1 | 291.6 | 1199.8 | 498.8 | 2.41 | 10048 | 2305 | 18->18 | 19->18 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 192x192 | enc3a | ok | ok | 1x64x48x48 -> 1x128x48x48 | 248.2 | 348.6 | 135.9 | 229.6 | 445.8 | 338.2 | 1.32 | 2913 | 1334 | 5->9 | 5->10 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 192x192 | enc3b | ok | ok | 1x128x48x48 -> 1x128x48x48 | 472.2 | 522.7 | 166.7 | 350.6 | 812.9 | 525.6 | 1.55 | 5480 | 2412 | 9->9 | 10->9 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 192x192 | enc4a | ok | ok | 1x128x24x24 -> 1x256x24x24 | 296.7 | 432.0 | 170.1 | 277.1 | 320.1 | 395.9 | 0.81 | 1662 | 1194 | 3->5 | 3->6 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 192x192 | enc4b | ok | ok | 1x256x24x24 -> 1x256x24x24 | 548.8 | 570.8 | 257.9 | 266.5 | 550.7 | 383.1 | 1.44 | 2953 | 1783 | 5->5 | 6->5 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 192x192 | bottlenecka | ok | ok | 1x256x12x12 -> 1x512x12x12 | 319.2 | 495.2 | 204.7 | 310.8 | 258.6 | 261.5 | 0.99 | 1029 | 879 | 2->3 | 2->3 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 192x192 | bottleneckb | ok | ok | 1x512x12x12 -> 1x512x12x12 | 618.2 | 663.3 | 350.1 | 589.9 | 512.3 | 458.4 | 1.12 | 1759 | 1205 | 3->3 | 3->3 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 224x224 | enc1a | ok | ok | 1x3x224x224 -> 1x32x224x224 | 61.2 | 51.2 | 79.7 | 49.6 | 346.7 | 49.6 | 6.99 | 2505 | 310 | 5->49 | 8->64 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 224x224 | enc1b | ok | ok | 1x32x224x224 -> 1x32x224x224 | 333.9 | 563.5 | 161.5 | 364.6 | 3136.1 | 1303.7 | 2.41 | 25224 | 9232 | 49->49 | 64->49 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 224x224 | enc2a | ok | ok | 1x32x112x112 -> 1x64x112x112 | 294.6 | 357.6 | 173.3 | 190.3 | 1081.9 | 575.5 | 1.88 | 7857 | 3615 | 13->25 | 13->26 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 224x224 | enc2b | ok | ok | 1x64x112x112 -> 1x64x112x112 | 574.9 | 584.4 | 267.3 | 289.6 | 2152.4 | 760.5 | 2.83 | 15389 | 4058 | 25->25 | 26->25 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 224x224 | enc3a | ok | ok | 1x64x56x56 -> 1x128x56x56 | 352.9 | 432.6 | 179.7 | 265.5 | 685.9 | 533.5 | 1.29 | 4519 | 2853 | 7->13 | 7->14 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 224x224 | enc3b | ok | ok | 1x128x56x56 -> 1x128x56x56 | 681.2 | 632.1 | 270.3 | 366.3 | 1248.0 | 710.3 | 1.76 | 8721 | 4427 | 13->13 | 14->13 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 224x224 | enc4a | ok | ok | 1x128x28x28 -> 1x256x28x28 | 394.8 | 434.6 | 190.2 | 286.7 | 498.9 | 293.0 | 1.70 | 2567 | 1864 | 4->7 | 4->7 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 224x224 | enc4b | ok | ok | 1x256x28x28 -> 1x256x28x28 | 767.1 | 679.1 | 304.7 | 422.8 | 936.8 | 775.0 | 1.21 | 4903 | 3240 | 7->7 | 7->7 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 224x224 | bottlenecka | ok | ok | 1x256x14x14 -> 1x512x14x14 | 386.6 | 551.5 | 230.2 | 349.2 | 306.8 | 338.0 | 0.91 | 1044 | 921 | 2->4 | 2->4 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 224x224 | bottleneckb | ok | ok | 1x512x14x14 -> 1x512x14x14 | 787.3 | 653.2 | 408.8 | 511.2 | 621.7 | 427.0 | 1.46 | 2093 | 1490 | 4->4 | 4->4 | no | no | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json; provider:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 384x288 | enc1a | running | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dense:.tmp/results/u22_dim32_encoder_baseline/auto_dense_provider_corg_20260526T130718Z.json |  |
| 384x288 | enc1b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc2a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc2b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc3a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc3b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc4a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc4b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | bottlenecka | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | bottleneckb | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc1a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc1b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc2a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc2b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc3a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc3b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc4a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc4b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | bottlenecka | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | bottleneckb | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
<!-- U22_DIM32_ENCODER_BASELINE_TABLE_END -->

## Step 1b: Dim64 Encoder Baseline

Goal: compare Orion dense LT and HaloED/provider on the same U22 encoder unique
Conv2d nodes as Step 1, but with `base_dim=64`.

Run policy:

- Use streamed LT for both dense and provider paths from the first attempt.
- Keep dense shared-cache / unified-BSGS disabled (`ORION_DENSE_LT_SHARED_CACHE=0`).
- Keep provider optimizations enabled: DP layout policy, relayout kernels, native halo,
  output fusion, shared rotation keys, compile trimming, and memory-bounded streaming.
- Use `io_mode=none`; report compute time excluding total I/O.

Active runs:

- Host/path: `qiuchu:/home/i_yuqiuchu/orion-u22-dim32`
- Docker container: `orion-u23-base64-192`, memory cap `1.5 TiB`
- Result dir: `.tmp/results/u22_dim64_encoder_baseline`
- Queue log: `.tmp/logs/u22_dim64_encoder_baseline_qiuchu_*.log`

Local refresh command:

```bash
rsync -az --checksum -e 'ssh -J root@10.10.0.1' 'qiuchu:/home/i_yuqiuchu/orion-u22-dim32/.tmp/results/u22_dim64_encoder_baseline/*.json' .tmp/results/u22_dim64_encoder_baseline/
.venv/bin/python tools/update_u22_dim64_encoder_doc.py --doc docs/u22_orion_streaming_haloed_mainline.md --result-dir .tmp/results/u22_dim64_encoder_baseline
```

<!-- U22_DIM64_ENCODER_BASELINE_TABLE_START -->
| input | node | dense status | provider status | input -> output | dense RSS GiB | provider RSS GiB | dense compile s | provider compile s | dense compute s | provider compute s | dense/provider compute | dense rotations | provider rotations | dense ct | provider ct | dense stream | provider stream | result files | note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 192x192 | enc1a | ok | ok | 1x3x192x192 -> 1x64x192x192 | 8.6 | 70.4 | 18.2 | 74.7 | 379.0 | 262.9 | 1.44 | 3432 | 1442 | 4->72 | 6->73 | yes | yes | dense:.tmp/results/u22_dim64_encoder_baseline/stream_dense_provider_qiuchu_20260526T172640Z.json; provider:.tmp/results/u22_dim64_encoder_baseline/stream_dense_provider_qiuchu_20260526T172640Z.json |  |
| 192x192 | enc1b | ok | ok | 1x64x192x192 -> 1x64x192x192 | 32.1 | 98.3 | 64.6 | 205.2 | 7414.5 | 3281.7 | 2.26 | 65792 | 18435 | 72->72 | 73->74 | yes | yes | dense:.tmp/results/u22_dim64_encoder_baseline/stream_dense_provider_qiuchu_20260526T172640Z.json; provider:.tmp/results/u22_dim64_encoder_baseline/stream_dense_provider_qiuchu_20260526T172640Z.json |  |
| 192x192 | enc2a | ok | ok | 1x64x96x96 -> 1x128x96x96 | 38.5 | 123.9 | 84.9 | 112.9 | 2518.8 | 1408.7 | 1.79 | 20096 | 4969 | 18->36 | 19->37 | yes | yes | dense:.tmp/results/u22_dim64_encoder_baseline/stream_dense_provider_qiuchu_20260526T172640Z.json; provider:.tmp/results/u22_dim64_encoder_baseline/stream_dense_provider_qiuchu_20260526T172640Z.json |  |
| 192x192 | enc2b | running | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dense:.tmp/results/u22_dim64_encoder_baseline/stream_dense_provider_qiuchu_20260526T172640Z.json |  |
| 192x192 | enc3a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 192x192 | enc3b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 192x192 | enc4a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 192x192 | enc4b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 192x192 | bottlenecka | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 192x192 | bottleneckb | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224x224 | enc1a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224x224 | enc1b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224x224 | enc2a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224x224 | enc2b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224x224 | enc3a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224x224 | enc3b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224x224 | enc4a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224x224 | enc4b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224x224 | bottlenecka | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 224x224 | bottleneckb | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc1a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc1b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc2a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc2b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc3a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc3b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc4a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | enc4b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | bottlenecka | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x288 | bottleneckb | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc1a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc1b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc2a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc2b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc3a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc3b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc4a | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | enc4b | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | bottlenecka | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 384x384 | bottleneckb | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
<!-- U22_DIM64_ENCODER_BASELINE_TABLE_END -->

## Step 2: Historical Dense Resident RAM Gate

Remote command queue:

```bash
cd ~/orion
bash .tmp/run_u22_exact_ram_gate_corg.sh
```

Remote result files:

```text
~/orion/.tmp/results/u22_exact_base64_dense_*_ram_gate_corg.json
~/orion/.tmp/results/u22_exact_base64_dense_ram_gate_logs/queue.log
```

Local refresh command:

```bash
rsync -az --checksum 'corg:/home/qihan/orion/.tmp/results/u22_exact_base64_dense_*_ram_gate_corg.json' .tmp/results/
.venv/bin/python tools/update_u22_mainline_doc.py --doc docs/u22_orion_streaming_haloed_mainline.md --result-dir .tmp/results
```

<!-- U22_RAM_GATE_TABLE_START -->
| layer | status | last event | input -> output | kernel | peak RSS GiB | cap GiB | generate s | compile s | forward s | source ct | output ct | note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| output | ok | after_forward | 1x64x256x256 -> 1x1x256x256 | 1x1/pad0x0 | 18.5 | 900.0 | 0.2 | 1.1 | 47.2 | 128 | 2 | completed |
| enc1b | rss_cap_exceeded | before_compile | 1x64x256x256 -> 1x64x256x256 | 3x3/pad1x1 | 903.3 | 900.0 | 100.6 |  |  |  |  | exceeded cap before completing layer |
| dec1a | rss_cap_exceeded | before_compile | 1x64x256x256 -> 1x64x256x256 | 3x3/pad1x1 | 901.6 | 900.0 | 100.0 |  |  |  |  | exceeded cap before completing layer |
| dec1b | rss_cap_exceeded | before_compile | 1x64x256x256 -> 1x64x256x256 | 3x3/pad1x1 | 900.2 | 900.0 | 100.5 |  |  |  |  | exceeded cap before completing layer |
| enc4b | running | before_generate_diagonals | 1x512x32x32 -> 1x512x32x32 | 3x3/pad1x1 | 8.7 | 900.0 |  |  |  |  |  | currently running / partial |
| bottleneckb | pending |  |  |  |  | 900 |  |  |  |  |  |  |
<!-- U22_RAM_GATE_TABLE_END -->

Interpretation:

- `rss_cap_exceeded` during `before_compile` means the layer exceeded the 900 GiB cap before compile completed.
- A resident dense full-network run is feasible only if every layer passes this gate.
- Sequential layer replay memory is the maximum single-layer peak, not the sum of all layer peaks.
- All-resident dense cache memory is closer to the sum of resident artifacts and is expected to be infeasible.

## Step 3: Historical Strong Orion Streaming Baseline

Goal: run Orion dense streaming LT layer-by-layer against the same exact U22 layers.

Policy to test:

- Multicore LT compile enabled.
- Streaming artifact residency enabled.
- BSGS preserved within the streamed chunk.
- Fill each chunk up to a controlled memory budget; on `corg`, test a large budget near 1 TiB to maximize baby-step sharing.
- Record chunk size / group size, because the chunk boundary determines how much sharing is preserved.

Open question for this step:

Can we use most of the 1 TiB RAM to keep the largest possible baby-step cache per chunk, while still avoiding dense resident OOM?  This should be measured by sweeping chunk budgets, not assumed.

<!-- U22_STREAMING_TABLE_START -->
| layer | backend path | chunk/budget | status | peak RSS GiB | compile s | compute s | I/O s excluded | BSGS/sharing note | result file |
|---|---|---|---|---:|---:|---:|---:|---|---|
| enc1b | Orion dense streaming | pending | pending |  |  |  | yes |  |  |
| dec1a | Orion dense streaming | pending | pending |  |  |  | yes |  |  |
| dec1b | Orion dense streaming | pending | pending |  |  |  | yes |  |  |
| output | Orion dense streaming | pending | pending |  |  |  | yes |  |  |
| enc4b | Orion dense streaming | pending | pending |  |  |  | yes |  |  |
| bottleneckb | Orion dense streaming | pending | pending |  |  |  | yes |  |  |
<!-- U22_STREAMING_TABLE_END -->

## Step 4: HaloED Layer-By-Layer

Goal: run HaloED layer-by-layer on the same exact U22 graph and use the same reporting convention.

Report:

- compile time separately
- compute time summed across layers
- total I/O excluded from the primary comparison
- peak RSS separately
- rotations / CT-PT multiplications / bootstraps from planner and runtime where available

<!-- U22_HALOED_TABLE_START -->
| layer | path | status | peak RSS GiB | compile s | compute s | I/O s excluded | rotations | CT-PT mult/diags | result file |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| enc1b | HaloED provider | pending |  |  |  | yes |  |  |  |
| dec1a | HaloED provider | pending |  |  |  | yes |  |  |  |
| dec1b | HaloED provider | pending |  |  |  | yes |  |  |  |
| output | HaloED provider | pending |  |  |  | yes |  |  |  |
| enc4b | HaloED provider | pending |  |  |  | yes |  |  |  |
| bottleneckb | HaloED provider | pending |  |  |  | yes |  |  |  |
<!-- U22_HALOED_TABLE_END -->

## Step 5: Local Full-Network E2E

Goal: run U22 plus explicit output layer (`23` linear layers), `base_dim=32`,
SiLU degree 7, full network, Lattigo `resnet` CKKS settings, across all four
segmentation shapes:

- `192x192`, IBSR BRAIN 2D, `1 -> 4`
- `224x224`, HanCo Hand, `3 -> 1`
- `384x288`, CVC-ClinicDB, `3 -> 1`
- `384x384`, Satellite cloud, `4 -> 1`

For each shape, run both Orion dense and provider paths.  Provider uses DP
layout planning, streaming LT, and all current provider optimizations.  Dense
uses streaming/memory-bounded LT, but explicitly disables the dense
unified-BSGS/shared-cache path.  Bootstrap-many is disabled for both paths
(`ORION_LATTIGO_BOOTSTRAP_MANY=0`) so the table reflects the default single
bootstrap path rather than batched bootstrap scheduling.

The table below is per-layer.  `stream+encode s` is computed per layer from
`stream_build_map_s + stream_encode_hoist_s + stream_load_payload_s`.
`LT+accum s` is computed per layer from
`stream_eval_s + stream_accumulate_s + cpp_baby_step_s + cpp_giant_step_s`
with `mvm_kernel_s` as the fallback for non-streaming dense timing.  Each
layer row also reports `boot after count` and `boot after s`: bootstraps
attached to activation/pool/cat nodes are attributed back to the preceding U22
linear layer when that is unambiguous, and otherwise emitted as `boot-only:*`
rows.  Each shape/path also has a `TOTAL` row with total runtime, rotations,
and bootstrap count.

Repro command:

```bash
.venv/bin/python tools/run_u22_dim32_dense_provider_e2e_matrix.py
```

<!-- U22_BASE32_SILU7_STREAMING_PROVIDER_E2E_TABLE_START -->
| input | dataset | I/O ch | path | status | layer | groups/transforms | stream+encode s | LT+accum s | eval total s | stream build s | encode hoist s | stream load s | LT eval s | LT accum s | baby+giant s | boot after count | boot after s | boot after nodes | compile s | HE forward s | hot E2E s | rotations | boots | runtime mode | peak RSS GiB | result file | note |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | running | TOTAL |  | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | no-row | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/192_192/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/224_224/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_288/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u22_dim32_dense_provider_e2e_matrix_nobootmany_qiuchu_20260527T014453Z/384_384/provider_e2e.json |  |
<!-- U22_BASE32_SILU7_STREAMING_PROVIDER_E2E_TABLE_END -->

## Step 6: U23 Base64 192x192 Full-Network E2E

This is a separate result from the dim32 full-network E2E above.  The model is
the exact U22 body plus explicit output head used by the compile-plan table:

- Input: IBSR BRAIN 2D, `1x1x192x192`
- Output channels: `4`
- `base_dim=64`
- `dec1b`: `64 -> 64`, `3x3/pad1`
- `output`: `64 -> 4`, `1x1/pad0`
- Activation: SiLU degree 7
- Lattigo/provider path: streaming LT forced, `ORION_LATTIGO_BOOTSTRAP_MANY=1`
- ACT cost uses the targeted activation-only hook; broad module profiling stays disabled.
- BOOT cost is read from `Bootstrap._bootstrap_runtime_profile`.

Repro command:

```bash
.venv/bin/python tools/run_u23_base64_192_silu7_streaming_provider_e2e.py
```

<!-- U23_BASE64_192_SILU7_STREAMING_PROVIDER_E2E_TABLE_START -->
| input | dataset | I/O ch | status | planner | bootmany | op breakdown | broad modules | compile/load s | compile LT group s | compile transforms | compile payload GiB | encrypt s | HE forward s | decode s | hot E2E s | MVM kernel s | ACT s | boot s | LT load/enc s | unattrib HE s | runtime mode | rotations | boot | input ct | output ct | MAE | peak RSS GiB | result file | note |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 192x192 | IBSR BRAIN 2D | 1 -> 4 | running |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_base64_192_silu7_streaming_provider_e2e_qiuchu_notrim_20260526T230724Z/provider_e2e.json |  |
<!-- U23_BASE64_192_SILU7_STREAMING_PROVIDER_E2E_TABLE_END -->

## Decision Rule

Use the strongest result that finishes by Monday, June 1, 2026:

- Best: Orion dense streaming+Bsgs vs HaloED full U22, layer-by-layer, compute time summed, I/O excluded.
- If Orion dense streaming only finishes encoder: Orion vs HaloED encoder, plus HaloED full U22.
- If Orion streaming is still not feasible on worst layers: HaloED full U22, plus dense resident infeasibility certificate and partial streaming evidence.

## Step 7: 224x224 Layer Timing Fill-In Table

This is the single example table for the current figure path: U22 `base_dim=32`,
HanCo Hand `224x224`, `3 -> 1`, SiLU degree 7. Fill the two rightmost time
columns with compute time only, excluding total I/O. Metric cells use the compact
`metric=count` form: rotations for linear operator, pooling, upsample, and
re-layout rows; CT-CT multiplications for SiLU rows; and bootstrap ciphertexts
for bootstrap rows. The dense metric column is left blank until Orion dense
layer metrics are filled in; the HaloED metric column is populated from the
current compile plan. The `beta` column records the selected output halo for
operator/activation/merge rows and the bootstrap-input halo after any drop-halo
cleanup for bootstrap rows. Downsample rows (`pool1..pool4`) and upsample rows
(`up4..up1`) are kept as their own nodes because layout transitions and
re-layout annotations can attach to those boundaries. Concat rows are also kept
so skip-merge explicit alignment does not disappear from the trace.

Source compile-plan CSV: `.tmp/results/unet22_plus_output_dim32_real_trace_edge_compile_plan_4cases.csv`.

<!-- U22_224_LAYER_TIMING_FILLIN_TABLE_START -->
| # | row | kind | Orion metric | HaloED metric | CT | beta | edge / bootstrap note | Orion dense compute s | HaloED provider compute s |
| ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | `enc1a` | Conv2d |  | rot=336 | 50 | 1 | carry |  |  |
| 2 | `enc1a_act` | SiLU act |  | ct-ct=150 | 50 | 1 | carry |  |  |
| 3 | `enc1b` | Conv2d |  | rot=3584 | 49 | 0 | fused |  |  |
| 4 | `bootstrap_after_enc1b` | Bootstrap |  | boot=49 | 49 | 0 | no halo drop; target `enc1b_act` |  |  |
| 5 | `enc1b_act` | SiLU act |  | ct-ct=147 | 49 | 0 | carry |  |  |
| 6 | `pool1` | downsample/pool |  | rot=160 | 13 | 1 | fused |  |  |
| 7 | `enc2a` | Conv2d |  | rot=6656 | 25 | 1 | fused |  |  |
| 8 | `enc2a_act` | SiLU act |  | ct-ct=75 | 25 | 1 | carry+bootstrap-drop |  |  |
| 9 | `bootstrap_after_enc2a_act` | Bootstrap |  | boot=25 | 25 | 0 | drop halo before bootstrap; target `enc2b` |  |  |
| 10 | `enc2b` | Conv2d |  | rot=13312 | 25 | 0 | fused |  |  |
| 11 | `enc2b_act` | SiLU act |  | ct-ct=75 | 25 | 0 | carry |  |  |
| 12 | `pool2` | downsample/pool |  | rot=320 | 7 | 1 | carry |  |  |
| 13 | `enc3a` | Conv2d |  | rot=17408 | 13 | 1 | fused |  |  |
| 14 | `enc3a_act` | SiLU act |  | ct-ct=39 | 13 | 1 | carry+bootstrap-drop |  |  |
| 15 | `bootstrap_after_enc3a_act` | Bootstrap |  | boot=13 | 13 | 0 | drop halo before bootstrap; target `enc3b` |  |  |
| 16 | `enc3b` | Conv2d |  | rot=34816 | 13 | 0 | fused |  |  |
| 17 | `enc3b_act` | SiLU act |  | ct-ct=39 | 13 | 0 | carry |  |  |
| 18 | `pool3` | downsample/pool |  | rot=640 | 4 | 1 | carry |  |  |
| 19 | `enc4a` | Conv2d |  | rot=67584 | 7 | 1 | fused |  |  |
| 20 | `enc4a_act` | SiLU act |  | ct-ct=21 | 7 | 1 | carry+bootstrap-drop |  |  |
| 21 | `bootstrap_after_enc4a_act` | Bootstrap |  | boot=7 | 7 | 0 | drop halo before bootstrap; target `enc4b` |  |  |
| 22 | `enc4b` | Conv2d |  | rot=135168 | 7 | 0 | fused |  |  |
| 23 | `enc4b_act` | SiLU act |  | ct-ct=21 | 7 | 0 | carry |  |  |
| 24 | `pool4` | downsample/pool |  | rot=1280 | 2 | 1 | carry |  |  |
| 25 | `bottlenecka` | Conv2d |  | rot=266240 | 4 | 1 | fused |  |  |
| 26 | `bottlenecka_act` | SiLU act |  | ct-ct=12 | 4 | 1 | carry+bootstrap-drop |  |  |
| 27 | `bootstrap_after_bottlenecka_act` | Bootstrap |  | boot=4 | 4 | 0 | drop halo before bootstrap; target `bottleneckb` |  |  |
| 28 | `bottleneckb` | Conv2d |  | rot=532480 | 4 | 1 | fused |  |  |
| 29 | `bottleneckb_act` | SiLU act |  | ct-ct=12 | 4 | 1 | carry |  |  |
| 30 | `up4` | upsample/tconv |  | rot=4096 | 7 | 2 | carry |  |  |
| 31 | `cat4` | skip merge |  | relayout-rot=14 | 14 | 2 | enc4b_act->cat4:explicit:dp_add_input_alignment:mode=halo_local;up4->cat4:carry:layout_same_carry_halo:mode=halo_local |  |  |
| 32 | `dec4a` | Conv2d |  | rot=9216 | 7 | 1 | fused |  |  |
| 33 | `dec4a_act` | SiLU act |  | ct-ct=21 | 7 | 1 | carry+bootstrap-drop |  |  |
| 34 | `bootstrap_after_dec4a_act` | Bootstrap |  | boot=7 | 7 | 0 | drop halo before bootstrap; target `dec4b` |  |  |
| 35 | `dec4b` | Conv2d |  | rot=135168 | 7 | 1 | fused |  |  |
| 36 | `dec4b_act` | SiLU act |  | ct-ct=21 | 7 | 1 | carry |  |  |
| 37 | `up3` | upsample/tconv |  | rot=2048 | 14 | 2 | carry |  |  |
| 38 | `cat3` | skip merge |  | relayout-rot=28 | 27 | 2 | enc3b_act->cat3:explicit:dp_add_input_alignment:mode=halo_local;up3->cat3:carry:layout_same_carry_halo:mode=halo_local |  |  |
| 39 | `dec3a` | Conv2d |  | rot=4608 | 13 | 1 | fused |  |  |
| 40 | `dec3a_act` | SiLU act |  | ct-ct=39 | 13 | 1 | carry+bootstrap-drop |  |  |
| 41 | `bootstrap_after_dec3a_act` | Bootstrap |  | boot=13 | 13 | 0 | drop halo before bootstrap; target `dec3b` |  |  |
| 42 | `dec3b` | Conv2d |  | rot=34816 | 13 | 1 | fused |  |  |
| 43 | `dec3b_act` | SiLU act |  | ct-ct=39 | 13 | 1 | carry |  |  |
| 44 | `up2` | upsample/tconv |  | rot=1024 | 26 | 2 | carry |  |  |
| 45 | `cat2` | skip merge |  | relayout-rot=52 | 51 | 2 | enc2b_act->cat2:explicit:dp_add_input_alignment:mode=halo_local;up2->cat2:carry:layout_same_carry_halo:mode=halo_local |  |  |
| 46 | `dec2a` | Conv2d |  | rot=2304 | 25 | 1 | fused |  |  |
| 47 | `dec2a_act` | SiLU act |  | ct-ct=75 | 25 | 1 | carry+bootstrap-drop |  |  |
| 48 | `bootstrap_after_dec2a_act` | Bootstrap |  | boot=25 | 25 | 0 | drop halo before bootstrap; target `dec2b` |  |  |
| 49 | `dec2b` | Conv2d |  | rot=13312 | 25 | 1 | fused |  |  |
| 50 | `dec2b_act` | SiLU act |  | ct-ct=75 | 25 | 1 | carry |  |  |
| 51 | `up1` | upsample/tconv |  | rot=512 | 50 | 2 | carry |  |  |
| 52 | `cat1` | skip merge |  | relayout-rot=100 | 100 | 2 | enc1b_act->cat1:explicit:dp_add_input_alignment:mode=halo_local;up1->cat1:carry:layout_same_carry_halo:mode=halo_local |  |  |
| 53 | `dec1a` | Conv2d |  | rot=1152 | 50 | 1 | fused |  |  |
| 54 | `dec1a_act` | SiLU act |  | ct-ct=150 | 50 | 1 | carry+bootstrap-drop |  |  |
| 55 | `bootstrap_after_dec1a_act` | Bootstrap |  | boot=49 | 49 | 0 | drop halo before bootstrap; saves 1 redundant CT; target `dec1b` |  |  |
| 56 | `dec1b` | Conv2d |  | rot=3584 | 49 | 0 | fused |  |  |
| 57 | `dec1b_act` | SiLU act |  | ct-ct=147 | 49 | 0 | carry |  |  |
| 58 | `output` | Conv2d |  | rot=192 | 2 | 0 | carry |  |  |
<!-- U22_224_LAYER_TIMING_FILLIN_TABLE_END -->
