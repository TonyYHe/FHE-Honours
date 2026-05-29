# U22 Orion Dense/Streaming vs HaloED Mainline

This document is the working plan for the current U-Net 22 comparison.

Current active gate, as of May 27, 2026:

- Conv kernel table on `corg:/home/qihan/orion`
- Lattigo kernel profile, `LogN=16`, default resident LT with `io_mode=none`
- `HW=4` means the four current input sizes: `192x192`, `224x224`, `384x288`, `384x384`
- Kernels: `Conv 32,32`, `Conv 64,64`, `Conv 128,128`, `Conv 256,256`, each `3x3/pad1/stride1`
- Active variants: Orion dense LT baseline plus no-sharing native-halo stripe
  provider beta `1/1` and `2/2`
- Metrics: runtime rotations and `LT+accumulate s`; no streamed encode time in this table
- Execution containment: one worker per table row with an RSS watchdog

The `HW` column is the original U-Net input size, not necessarily the logical
Conv input size.  The table simulates encoder-stage packing:

- `Conv 32,32`: logical `32 x H x W`, multiplex/input/output gap `1`, packed FHE `32 x H x W`
- `Conv 64,64`: logical `64 x H/2 x W/2`, multiplex/input/output gap `2`, four channels per packed group, packed FHE `16 x H x W`
- `Conv 128,128`: logical `128 x H/4 x W/4`, multiplex/input/output gap `4`, sixteen channels per packed group, packed FHE `8 x H x W`
- `Conv 256,256`: logical `256 x H/8 x W/8`, multiplex/input/output gap `8`, sixty-four channels per packed group, packed FHE `4 x H x W`

The completed dense resident run is the baseline provenance for this
kernel-table gate.  Reuse only exact-shape, non-stream dense rows; the active
provider rows must be measured as no-sharing native-halo stripe rows with
`native_halo_channel_fold_mode=per_stripe`.  Do not mix in older shared-provider
or non-stripe provider JSONs.

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
3. If a resident path exceeds the RSS cap, rerun that exact node/input/path with single-slot layer cache and record the encode worker count.
4. Compare compute time separately from memory feasibility. Do not mix resident dense RAM certificates, streamed fallback timings, and HaloED timings without labeling the path.

Streaming does not mean disabling BSGS.  The intended baseline is now a single-slot layer cache: compile records diagonal-index metadata, rotation-key plans, and runtime diagonal recipes without keeping raw diagonal payloads or encoded plaintexts; runtime materializes/encodes exactly the current dense op or provider group, evaluates it, then evicts those plaintexts before the next op/group.  Orion dense remains independent LT evaluation; HaloED/provider uses grouped provider kernels where the lowering exposes shared-source structure.  Report layer compute time separately from `layer_cache_turnover_s` (`layer_cache_encode_s + layer_cache_key_prepare_s + layer_cache_evict_s`).  Legacy chunked Lattigo LT streaming is not the mainline and requires the explicit `ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT=1` gate.  Kernel-level benchmarks remain resident by default; single-slot is only for full-network/E2E memory-bounded evaluation.

## Dim32 Packed-Diagonal Level Replay

Use this fixed replay script when estimating all-preencoded plaintext memory for
the four dim32 U22 input sizes:

```bash
.venv/bin/python tools/replay_u22_dim32_packed_diag_levels.py \
  --plan-csv .tmp/results/u22_dim32_4sizes_compile_plan_packed_diags.csv \
  --out-csv .tmp/results/u22_dim32_4sizes_packed_diag_level_replay.csv \
  --out-json .tmp/results/u22_dim32_4sizes_packed_diag_level_replay.json
```

The input CSV supplies `diagonal_count` only.  The replay intentionally ignores
source `assigned_level`, rebuilds the real traced U22+output DAG, applies the
same fuser as `Scheme.compile()`, injects a lightweight diagonal-count proxy
into each LT module, and reruns `BootstrapSolver`.  This prevents stale
pre-fuser or zero-LT-cost level plans from contaminating memory estimates.

Memory formula for `encoded_assigned`:

`sum_node D_node * (assigned_level_node + 1 + 3) * 65536 * 8`

Latest local replay output:

| input | diagonals | encoded assigned TiB |
| --- | ---: | ---: |
| 192x192 | 646,875 | 2.502 |
| 224x224 | 899,296 | 3.522 |
| 384x288 | 1,881,093 | 7.384 |
| 384x384 | 2,398,659 | 9.404 |

## Step 0: Conv Kernel Table

Goal: compare default resident Orion dense LT against no-sharing HaloED/provider
native-halo stripe Conv kernels for the four channel sizes and four HW sizes
above.
`Conv64`, `Conv128`, and `Conv256` use the U-Net stage-packed logical input
sizes described above, while keeping the packed FHE canvas at the original `HW`.
If a resident row hits the RSS cap or OOMs, record that row as a resident
feasibility failure; do not silently replace it with a streamed result in this
table.

Corrected main kernel profile: use the real `resnet`/`e2e` CKKS chain
(`LogN=16`, `LogQ=[55,40,40,40,40,40,40,40,40,40,40]`,
`LogP=[61,61,61]`, boot `LogP=[61,61,61,61,61,61,61,61]`,
`LogScale=40`, `H=192`). The main Conv kernel input level is `2`; a
depth-1 Conv is expected to output level `1`, and the runner records the
actual output ciphertext level for every completed row. Runtime LT evaluation
must run with `GOMAXPROCS=1` and one row worker at a time; compile and diagonal
encode worker pools may use the host CPU count. The superseded no-share stripe
run `.tmp/results/conv_kernel_table_noshare_stripe_corg_20260529T202528Z`
used the short kernel chain/max input level and is not main evidence.

Run command on `corg`:

```bash
cd ~/orion
mkdir -p .tmp/run .tmp/logs .tmp/results
cpu_count="$(nproc)"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
run_root=".tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_${ts}"
log=".tmp/logs/conv_kernel_table_noshare_stripe_level2_e2e_corg_${ts}.log"
{
  echo "run_root=$run_root"
  echo "log=$log"
  echo "started_at_utc=$(date -u +%FT%TZ)"
  echo "ckks_profile=resnet_e2e_logn16_logscale40_h192"
  echo "input_level=2"
  echo "expected_output_level=1"
} > .tmp/run/conv_kernel_table_noshare_stripe_level2_e2e.latest
GOMAXPROCS=1 \
ORION_COMPILE_PARALLEL_POLICY=manual \
ORION_LATTIGO_STREAMING_LT=0 \
ORION_LATTIGO_MEMORY_BOUNDED_COMPILE=0 \
ORION_LATTIGO_MEMORY_BOUNDED_EVAL=0 \
ORION_SINGLE_SLOT_LAYER_CACHE=0 \
ORION_SINGLE_SLOT_ENCODE_WORKERS="$cpu_count" \
ORION_PACK_CONV_WORKERS="$cpu_count" \
ORION_DIRECT_PACK_WORKERS="$cpu_count" \
ORION_LT_COMPILE_WORKERS="$cpu_count" \
ORION_UNIFIED_COMPILE_WORKERS="$cpu_count" \
ORION_LATTIGO_COMPILE_WORKERS="$cpu_count" \
ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS="$cpu_count" \
ORION_UNIFIED_LT_INDIVIDUAL_EVAL=1 \
ORION_UNIFIED_LT_SHARED_ROTATION_KEYS=0 \
ORION_LATTIGO_UNIFIED_NO_BSGS=0 \
PYTHONUNBUFFERED=1 \
nohup .venv/bin/python tools/run_conv_kernel_table.py \
  --run-root "$run_root" \
  --doc docs/u22_orion_streaming_haloed_mainline.md \
  --max-worker-rss-gb 850 \
  --channels 32 64 128 256 \
  --hw 192x192 224x224 384x288 384x384 \
  --variants orion provider_halo1_individual_lt provider_halo2_individual_lt \
  --provider-output-layout native_stripe \
  --input-level 2 \
  > "$log" 2>&1 &
echo "pid=$!" >> .tmp/run/conv_kernel_table_noshare_stripe_level2_e2e.latest
```

Refresh locally:

```bash
run_root="$(ssh corg 'cd /home/qihan/orion && awk -F= '\''$1=="run_root"{print $2}'\'' .tmp/run/conv_kernel_table_noshare_stripe_level2_e2e.latest')"
base="$(basename "$run_root")"
mkdir -p ".tmp/results/$base"
rsync -az --delete --checksum "corg:/home/qihan/orion/$run_root/" ".tmp/results/$base/"
.venv/bin/python tools/run_conv_kernel_table.py --update-doc-only \
  --run-root ".tmp/results/$base" \
  --doc docs/u22_orion_streaming_haloed_mainline.md \
  --channels 32 64 128 256 \
  --hw 192x192 224x224 384x288 384x384 \
  --variants orion provider_halo1_individual_lt provider_halo2_individual_lt \
  --provider-output-layout native_stripe \
  --input-level 2
```

<!-- CONV_KERNEL_TABLE_START -->
| HW | kernel | logical input | multiplex | channels/group | packed FHE input | path / beta | status | input level | expected output level | actual output level | input halo T/B | output layout | channel fold | LT grouping | rotations | LT+accumulate s | hot run s | compile s | input ct | output ct | peak RSS GiB | runtime mode | result file | note |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 192x192 | Conv 32,32 | 32x192x192 | 1 | 1 | 32x192x192 | Orion dense | running | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_192x192_orion.json |  |
| 192x192 | Conv 32,32 | 32x192x192 | 1 | 1 | 32x192x192 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_192x192_provider_halo1_individual_lt.json |  |
| 192x192 | Conv 32,32 | 32x192x192 | 1 | 1 | 32x192x192 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | Conv 32,32 | 32x224x224 | 1 | 1 | 32x224x224 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_224x224_orion.json |  |
| 224x224 | Conv 32,32 | 32x224x224 | 1 | 1 | 32x224x224 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_224x224_provider_halo1_individual_lt.json |  |
| 224x224 | Conv 32,32 | 32x224x224 | 1 | 1 | 32x224x224 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | Conv 32,32 | 32x384x288 | 1 | 1 | 32x384x288 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_384x288_orion.json |  |
| 384x288 | Conv 32,32 | 32x384x288 | 1 | 1 | 32x384x288 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_384x288_provider_halo1_individual_lt.json |  |
| 384x288 | Conv 32,32 | 32x384x288 | 1 | 1 | 32x384x288 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | Conv 32,32 | 32x384x384 | 1 | 1 | 32x384x384 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_384x384_orion.json |  |
| 384x384 | Conv 32,32 | 32x384x384 | 1 | 1 | 32x384x384 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_384x384_provider_halo1_individual_lt.json |  |
| 384x384 | Conv 32,32 | 32x384x384 | 1 | 1 | 32x384x384 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv32_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | Conv 64,64 | 64x96x96 | 2 | 4 | 16x192x192 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_192x192_orion.json |  |
| 192x192 | Conv 64,64 | 64x96x96 | 2 | 4 | 16x192x192 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_192x192_provider_halo1_individual_lt.json |  |
| 192x192 | Conv 64,64 | 64x96x96 | 2 | 4 | 16x192x192 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | Conv 64,64 | 64x112x112 | 2 | 4 | 16x224x224 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_224x224_orion.json |  |
| 224x224 | Conv 64,64 | 64x112x112 | 2 | 4 | 16x224x224 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_224x224_provider_halo1_individual_lt.json |  |
| 224x224 | Conv 64,64 | 64x112x112 | 2 | 4 | 16x224x224 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | Conv 64,64 | 64x192x144 | 2 | 4 | 16x384x288 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_384x288_orion.json |  |
| 384x288 | Conv 64,64 | 64x192x144 | 2 | 4 | 16x384x288 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_384x288_provider_halo1_individual_lt.json |  |
| 384x288 | Conv 64,64 | 64x192x144 | 2 | 4 | 16x384x288 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | Conv 64,64 | 64x192x192 | 2 | 4 | 16x384x384 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_384x384_orion.json |  |
| 384x384 | Conv 64,64 | 64x192x192 | 2 | 4 | 16x384x384 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_384x384_provider_halo1_individual_lt.json |  |
| 384x384 | Conv 64,64 | 64x192x192 | 2 | 4 | 16x384x384 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv64_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | Conv 128,128 | 128x48x48 | 4 | 16 | 8x192x192 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_192x192_orion.json |  |
| 192x192 | Conv 128,128 | 128x48x48 | 4 | 16 | 8x192x192 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_192x192_provider_halo1_individual_lt.json |  |
| 192x192 | Conv 128,128 | 128x48x48 | 4 | 16 | 8x192x192 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | Conv 128,128 | 128x56x56 | 4 | 16 | 8x224x224 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_224x224_orion.json |  |
| 224x224 | Conv 128,128 | 128x56x56 | 4 | 16 | 8x224x224 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_224x224_provider_halo1_individual_lt.json |  |
| 224x224 | Conv 128,128 | 128x56x56 | 4 | 16 | 8x224x224 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | Conv 128,128 | 128x96x72 | 4 | 16 | 8x384x288 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_384x288_orion.json |  |
| 384x288 | Conv 128,128 | 128x96x72 | 4 | 16 | 8x384x288 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_384x288_provider_halo1_individual_lt.json |  |
| 384x288 | Conv 128,128 | 128x96x72 | 4 | 16 | 8x384x288 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | Conv 128,128 | 128x96x96 | 4 | 16 | 8x384x384 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_384x384_orion.json |  |
| 384x384 | Conv 128,128 | 128x96x96 | 4 | 16 | 8x384x384 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_384x384_provider_halo1_individual_lt.json |  |
| 384x384 | Conv 128,128 | 128x96x96 | 4 | 16 | 8x384x384 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv128_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | Conv 256,256 | 256x24x24 | 8 | 64 | 4x192x192 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_192x192_orion.json |  |
| 192x192 | Conv 256,256 | 256x24x24 | 8 | 64 | 4x192x192 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_192x192_provider_halo1_individual_lt.json |  |
| 192x192 | Conv 256,256 | 256x24x24 | 8 | 64 | 4x192x192 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | Conv 256,256 | 256x28x28 | 8 | 64 | 4x224x224 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_224x224_orion.json |  |
| 224x224 | Conv 256,256 | 256x28x28 | 8 | 64 | 4x224x224 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_224x224_provider_halo1_individual_lt.json |  |
| 224x224 | Conv 256,256 | 256x28x28 | 8 | 64 | 4x224x224 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | Conv 256,256 | 256x48x36 | 8 | 64 | 4x384x288 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_384x288_orion.json |  |
| 384x288 | Conv 256,256 | 256x48x36 | 8 | 64 | 4x384x288 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_384x288_provider_halo1_individual_lt.json |  |
| 384x288 | Conv 256,256 | 256x48x36 | 8 | 64 | 4x384x288 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | Conv 256,256 | 256x48x48 | 8 | 64 | 4x384x384 | Orion dense | pending | 2 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_384x384_orion.json |  |
| 384x384 | Conv 256,256 | 256x48x48 | 8 | 64 | 4x384x384 | provider beta=1 no-share stripe | pending | 2 | 1 |  | 1/1 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_384x384_provider_halo1_individual_lt.json |  |
| 384x384 | Conv 256,256 | 256x48x48 | 8 | 64 | 4x384x384 | provider beta=2 no-share stripe | pending | 2 | 1 |  | 2/2 |  | per_stripe | individual |  |  |  |  |  |  |  |  | .tmp/results/conv_kernel_table_noshare_stripe_level2_e2e_corg_20260529T221125Z/rows/conv256_384x384_provider_halo2_individual_lt.json |  |
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
- On `rss_watermark`, OOM, or similar memory failure, rerun only the failed node/input/path with `ORION_SINGLE_SLOT_LAYER_CACHE=1`, `ORION_SINGLE_SLOT_ENCODE_WORKERS=<fixed worker count>`, `ORION_LATTIGO_STREAMING_LT=0`, `ORION_LATTIGO_MEMORY_BOUNDED_COMPILE=0`, and `ORION_LATTIGO_MEMORY_BOUNDED_EVAL=0`.
- Record peak RSS, compile time, compute time, layer-cache turnover time, rotations, ciphertext counts, whether single-slot layer cache was used, and exact result JSON path for both paths.

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

- Use the common single-slot layer cache for dense and provider paths from the first attempt.
- Keep ConvTranspose2d on the common dense path for both dense and provider modes.
- Keep provider optimizations enabled: DP layout policy, relayout kernels, native halo,
  output fusion, and shared rotation keys.
- Use `io_mode=none`; report compute time excluding total I/O, and report layer-cache turnover separately.

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

## Step 3: Strong Orion Single-Slot Streaming Baseline

Goal: run Orion dense LT with the single-slot layer cache against the same exact U22 layers.

Policy to test:

- Compile prepares all cleartext raw diagonal payloads for each unified BSGS group.
- Runtime materializes/encodes only the current layer plaintexts.
- BSGS/shared-cache behavior inside that layer is the normal resident path.
- After the layer evaluates, evict the current layer plaintexts before moving on.
- Record `layer_cache_turnover_s` separately from layer compute time.

Open question for this step:

What fixed `ORION_SINGLE_SLOT_ENCODE_WORKERS` value gives the best layer turnover time on the target host without increasing RSS spikes?

<!-- U22_STREAMING_TABLE_START -->
| layer | backend path | layer cache policy | status | peak RSS GiB | compile s | compute s | turnover s | BSGS/sharing note | result file |
|---|---|---|---|---:|---:|---:|---:|---|---|
| enc1b | Orion dense single-slot | pending | pending |  |  |  |  | resident layer BSGS |  |
| dec1a | Orion dense single-slot | pending | pending |  |  |  |  | resident layer BSGS |  |
| dec1b | Orion dense single-slot | pending | pending |  |  |  |  | resident layer BSGS |  |
| output | Orion dense single-slot | pending | pending |  |  |  |  | resident layer BSGS |  |
| enc4b | Orion dense single-slot | pending | pending |  |  |  |  | resident layer BSGS |  |
| bottleneckb | Orion dense single-slot | pending | pending |  |  |  |  | resident layer BSGS |  |
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

For each shape, run both Orion dense and provider paths.  The current provider
comparison target is real Lattigo/native no-share-fold stripe E2E, selected with
`--policy dp_no_share_fold`, not a compile-only or planner-only run.  Provider
LTs must run with individual evaluation and no shared rotation-key grouping:
`ORION_UNIFIED_LT_INDIVIDUAL_EVAL=1` and
`ORION_UNIFIED_LT_SHARED_ROTATION_KEYS=0`.  Keep BSGS inside each independent LT
enabled with `ORION_LATTIGO_UNIFIED_NO_BSGS=0`.  Dense and provider both use the
single-slot layer cache (`ORION_SINGLE_SLOT_LAYER_CACHE=1`), concat fusion
(`ORION_CONCAT_FUSION=1`), and the default single-bootstrap path
(`ORION_LATTIGO_BOOTSTRAP_MANY=0`).

The table below is per-layer.  `layer cache turnover s` is the single-slot
resident layer-cache overhead
(`layer_cache_encode_s + layer_cache_key_prepare_s + layer_cache_evict_s`).
`LT+accum s` is the actual linear-transform/accumulation compute time, using
`mvm_kernel_s` as the primary value.  Legacy chunk-stream load/encode columns
are retained for audit only and should be zero in the single-slot mainline.
Each layer row also reports `boot after count` and `boot after s`: bootstraps
attached to activation/pool/cat nodes are attributed back to the preceding U22
linear layer when that is unambiguous, and otherwise emitted as `boot-only:*`
rows.  Each shape/path also has a `TOTAL` row with total runtime, rotations,
and bootstrap count.

Repro command:

```bash
.venv/bin/python tools/run_u22_dim32_dense_provider_e2e_matrix.py
```

On `qiuchu`, host Python may miss `h5py`; use the container runner.  After the
dense `192x192` row is terminal and the docs have been refreshed, preserve the
existing dense result and add the provider row under the same serial run root:

```bash
docker exec -w /workspace/orion \
  -e ORION_SINGLE_SLOT_LAYER_CACHE=1 \
  -e ORION_LATTIGO_BOOTSTRAP_MANY=0 \
  -e ORION_UNIFIED_LT_INDIVIDUAL_EVAL=1 \
  -e ORION_UNIFIED_LT_SHARED_ROTATION_KEYS=0 \
  -e ORION_LATTIGO_UNIFIED_NO_BSGS=0 \
  -e ORION_CONCAT_FUSION=1 \
  -e GOMAXPROCS=1 \
  -e ORION_PACK_CONV_WORKERS=240 \
  -e ORION_LT_COMPILE_WORKERS=240 \
  -e ORION_UNIFIED_COMPILE_WORKERS=240 \
  -e ORION_LATTIGO_COMPILE_WORKERS=240 \
  -e ORION_LATTIGO_DIAGONAL_ENCODE_WORKERS=240 \
  -e ORION_SINGLE_SLOT_ENCODE_WORKERS=240 \
  -e ORION_LATTIGO_BOOTSTRAP_WORKERS=240 \
  orion-u23-base64-192 /workspace/orion/.venv/bin/python \
  tools/run_u22_dim32_dense_provider_e2e_matrix.py \
  --run-root <remote_serial_run_root> \
  --doc docs/u22_orion_streaming_haloed_mainline.md \
  --cases 192x192 \
  --modes provider \
  --policy dp_no_share_fold \
  --keep-going \
  --operator-breakdown \
  --forward-runs 1 \
  --warmup-runs 0
```

Network-level summary:

Summary split columns are sourced from
`operator_breakdown_after_forward.totals.*`.  `MVM/LT s (incl pool)` is
`mvm_kernel_s`, so it includes pool/other LT-like rows reported by the runtime
breakdown, not only named Conv2d/TConv layers; `diag+encode s` is
`lt_layer_cache_encode_s`, and `unattributed s` prefers the runtime-reported
`unattributed_he_forward_s`; when that key is absent, the runner computes the
residual from HE forward minus the displayed split columns.

Current dense rows may be retained from the serial qiuchu run root.  Provider
rows are accepted for the no-share-fold comparison only when their runner
metadata records `policy=dp_no_share_fold`,
`ORION_UNIFIED_LT_INDIVIDUAL_EVAL=1`,
`ORION_UNIFIED_LT_SHARED_ROTATION_KEYS=0`, and
`ORION_LATTIGO_UNIFIED_NO_BSGS=0`; older provider placeholders from `policy=dp`
must be replaced before drawing dense/provider conclusions.

<!-- U22_BASE32_SILU7_NETWORK_SUMMARY_TABLE_START -->
| input | dataset | I/O ch | dense status | Halo status | dense HE forward s | Halo HE forward s | dense/Halo HE | dense hot E2E s | Halo hot E2E s | dense MVM/LT s (incl pool) | Halo MVM/LT s (incl pool) | dense activation excl boot s | Halo activation excl boot s | dense bootstrap s | Halo bootstrap s | dense diag+encode s | Halo diag+encode s | dense unattributed s | Halo unattributed s | dense rotations | Halo rotations | dense boots | Halo boots | dense RSS GiB | Halo RSS GiB | runtime mode | result files | note |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 192x192 | IBSR BRAIN 2D | 1->4 | ok | started | 20018.1 |  |  | 20018.3 |  | 12214.6 |  | 97.4 |  | 2361.9 |  | 3482.1 |  | 5344.1 |  | 0 | 0 | 8 | 9 | 796.0 |  | single_slot_layer_cache | dense:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json; provider:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json | Swaps: 0 	File system inputs: 0 	File system outputs: 255864 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0;  diagonals = 26084 ├── time to pack (s): 2054.96 ├── # diagonals = 26084 ├── time to pack (s): 2078.55 ├── # diagonals = 26084 ├── time to pack (s): 2078.58 ├── # diagonals = 26084 |
| 224x224 | HanCo Hand | 3->1 | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dense:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json; provider:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dense:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json; provider:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dense:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json; provider:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
<!-- U22_BASE32_SILU7_NETWORK_SUMMARY_TABLE_END -->

<!-- U22_BASE32_SILU7_STREAMING_PROVIDER_E2E_TABLE_START -->
| input | dataset | I/O ch | path | status | layer | groups/transforms | layer cache turnover s | layer cache diag+encode s | layer cache key prep s | layer cache evict s | LT+accum s | eval total s | legacy load/encode s | stream build s | encode hoist s | stream load s | LT eval s | LT accum s | baby+giant s | boot after count | boot after s | boot after nodes | compile s | HE forward s | hot E2E s | rotations | boots | runtime mode | peak RSS GiB | result file | note |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | TOTAL | 159 | 3482.3 | 3482.1 | 0.0 | 0.1 | 12214.6 | 12264.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  | 8 | 2361.9 |  | 362.2 | 20018.1 | 20018.3 | 0 | 8 | single_slot_layer_cache | 796.0 | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json | 	Swaps: 0 	File system inputs: 0 	File system outputs: 255864 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc1a | 1 | 4.0 | 4.0 | 0.0 | 0.0 | 95.8 | 95.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc1b | 1 | 84.4 | 84.4 | 0.0 | 0.0 | 1067.5 | 1067.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 823.2 | enc1b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc2a | 1 | 55.9 | 55.9 | 0.0 | 0.0 | 569.5 | 569.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc2b | 1 | 110.7 | 110.7 | 0.0 | 0.0 | 571.0 | 571.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 408.3 | enc2b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc3a | 1 | 62.3 | 62.3 | 0.0 | 0.0 | 378.7 | 378.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc3b | 1 | 115.6 | 115.6 | 0.0 | 0.0 | 392.6 | 392.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 207.9 | enc3b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc4a | 1 | 71.5 | 71.5 | 0.0 | 0.0 | 308.4 | 308.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc4b | 1 | 131.9 | 131.9 | 0.0 | 0.0 | 336.7 | 336.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 118.9 | enc4b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | bottlenecka | 1 | 97.7 | 97.7 | 0.0 | 0.0 | 272.1 | 272.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | bottleneckb | 1 | 143.6 | 143.6 | 0.0 | 0.0 | 324.2 | 324.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 70.1 | bottleneckb_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | up4 | 1 | 87.6 | 87.6 | 0.0 | 0.0 | 307.7 | 307.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec4a | 10 | 962.0 | 962.0 | 0.0 | 0.0 | 739.3 | 742.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 739.3 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec4b | 1 | 150.4 | 150.4 | 0.0 | 0.0 | 346.5 | 346.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 114.9 | dec4b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | up3 | 1 | 71.5 | 71.5 | 0.0 | 0.0 | 407.4 | 407.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec3a | 18 | 421.1 | 421.0 | 0.0 | 0.0 | 617.5 | 623.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 617.5 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec3b | 1 | 123.7 | 123.7 | 0.0 | 0.0 | 436.6 | 436.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 210.0 | dec3b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | up2 | 1 | 57.8 | 57.8 | 0.0 | 0.0 | 614.5 | 614.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec2a | 36 | 288.5 | 288.4 | 0.0 | 0.0 | 647.9 | 660.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 647.9 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec2b | 1 | 118.2 | 118.2 | 0.0 | 0.0 | 591.8 | 591.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 408.7 | dec2b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | up1 | 1 | 41.4 | 41.4 | 0.0 | 0.0 | 1015.0 | 1015.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec1a | 72 | 190.0 | 189.9 | 0.0 | 0.0 | 659.3 | 686.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 659.3 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec1b | 1 | 85.0 | 85.0 | 0.0 | 0.0 | 1113.4 | 1113.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | output | 1 | 1.5 | 1.5 | 0.0 | 0.0 | 73.3 | 73.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | started | TOTAL |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 |  |  |  |  |  |  | 9 |  |  | 695.2 |  |  | 0 | 9 |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  diagonals = 26084 ├── time to pack (s): 2054.96 ├── # diagonals = 26084 ├── time to pack (s): 2078.55 ├── # diagonals = 26084 ├── time to pack (s): 2078.58 ├── # diagonals = 26084 |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 0.0 | enc1b_act |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 0.0 | enc2b_act |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 0.0 | enc3b_act |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 0.0 | enc4b_act |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 0.0 | bottleneckb_act |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 0.0 | dec4b |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 0.0 | dec3a_act |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 0.0 | dec2a |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 0.0 | up1 |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/192_192/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/224_224/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_288/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense_provider_e2e_matrix_qiuchu_20260527T222208Z_b6dde16_concatfast_singleslot/384_384/provider_e2e.json |  |
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
for bootstrap rows. `bootstrap-drop` is reserved for CT-saving halo drops before
bootstrap, not for cleanup that leaves the bootstrap CT count unchanged. The
dense metric column is left blank until Orion dense layer metrics are filled in;
the HaloED metric column is populated from the current compile plan. The `beta`
column records the selected output halo for operator/activation/merge rows and
the bootstrap-input halo after any CT-saving drop-halo cleanup for bootstrap
rows. Downsample rows (`pool1..pool4`) and upsample rows (`up4..up1`) are kept
as their own nodes because layout transitions and re-layout annotations can
attach to those boundaries. Concat rows are also kept
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
| 8 | `enc2a_act` | SiLU act |  | ct-ct=75 | 25 | 1 | carry |  |  |
| 9 | `bootstrap_after_enc2a_act` | Bootstrap |  | boot=25 | 25 | 1 | no CT-saving halo drop; target `enc2b` |  |  |
| 10 | `enc2b` | Conv2d |  | rot=13312 | 25 | 0 | fused |  |  |
| 11 | `enc2b_act` | SiLU act |  | ct-ct=75 | 25 | 0 | carry |  |  |
| 12 | `pool2` | downsample/pool |  | rot=320 | 7 | 1 | carry |  |  |
| 13 | `enc3a` | Conv2d |  | rot=17408 | 13 | 1 | fused |  |  |
| 14 | `enc3a_act` | SiLU act |  | ct-ct=39 | 13 | 1 | carry |  |  |
| 15 | `bootstrap_after_enc3a_act` | Bootstrap |  | boot=13 | 13 | 1 | no CT-saving halo drop; target `enc3b` |  |  |
| 16 | `enc3b` | Conv2d |  | rot=34816 | 13 | 0 | fused |  |  |
| 17 | `enc3b_act` | SiLU act |  | ct-ct=39 | 13 | 0 | carry |  |  |
| 18 | `pool3` | downsample/pool |  | rot=640 | 4 | 1 | carry |  |  |
| 19 | `enc4a` | Conv2d |  | rot=67584 | 7 | 1 | fused |  |  |
| 20 | `enc4a_act` | SiLU act |  | ct-ct=21 | 7 | 1 | carry |  |  |
| 21 | `bootstrap_after_enc4a_act` | Bootstrap |  | boot=7 | 7 | 1 | no CT-saving halo drop; target `enc4b` |  |  |
| 22 | `enc4b` | Conv2d |  | rot=135168 | 7 | 0 | fused |  |  |
| 23 | `enc4b_act` | SiLU act |  | ct-ct=21 | 7 | 0 | carry |  |  |
| 24 | `pool4` | downsample/pool |  | rot=1280 | 2 | 1 | carry |  |  |
| 25 | `bottlenecka` | Conv2d |  | rot=266240 | 4 | 1 | fused |  |  |
| 26 | `bottlenecka_act` | SiLU act |  | ct-ct=12 | 4 | 1 | carry |  |  |
| 27 | `bootstrap_after_bottlenecka_act` | Bootstrap |  | boot=4 | 4 | 1 | no CT-saving halo drop; target `bottleneckb` |  |  |
| 28 | `bottleneckb` | Conv2d |  | rot=532480 | 4 | 1 | fused |  |  |
| 29 | `bottleneckb_act` | SiLU act |  | ct-ct=12 | 4 | 1 | carry |  |  |
| 30 | `up4` | upsample/tconv |  | rot=4096 | 7 | 2 | carry |  |  |
| 31 | `cat4` | skip merge |  | relayout-rot=14 | 14 | 2 | enc4b_act->cat4:explicit:dp_add_input_alignment:mode=halo_local;up4->cat4:carry:layout_same_carry_halo:mode=halo_local |  |  |
| 32 | `dec4a` | Conv2d |  | rot=9216 | 7 | 1 | fused |  |  |
| 33 | `dec4a_act` | SiLU act |  | ct-ct=21 | 7 | 1 | carry |  |  |
| 34 | `bootstrap_after_dec4a_act` | Bootstrap |  | boot=7 | 7 | 1 | no CT-saving halo drop; target `dec4b` |  |  |
| 35 | `dec4b` | Conv2d |  | rot=135168 | 7 | 1 | fused |  |  |
| 36 | `dec4b_act` | SiLU act |  | ct-ct=21 | 7 | 1 | carry |  |  |
| 37 | `up3` | upsample/tconv |  | rot=2048 | 14 | 2 | carry |  |  |
| 38 | `cat3` | skip merge |  | relayout-rot=28 | 27 | 2 | enc3b_act->cat3:explicit:dp_add_input_alignment:mode=halo_local;up3->cat3:carry:layout_same_carry_halo:mode=halo_local |  |  |
| 39 | `dec3a` | Conv2d |  | rot=4608 | 13 | 1 | fused |  |  |
| 40 | `dec3a_act` | SiLU act |  | ct-ct=39 | 13 | 1 | carry |  |  |
| 41 | `bootstrap_after_dec3a_act` | Bootstrap |  | boot=13 | 13 | 1 | no CT-saving halo drop; target `dec3b` |  |  |
| 42 | `dec3b` | Conv2d |  | rot=34816 | 13 | 1 | fused |  |  |
| 43 | `dec3b_act` | SiLU act |  | ct-ct=39 | 13 | 1 | carry |  |  |
| 44 | `up2` | upsample/tconv |  | rot=1024 | 26 | 2 | carry |  |  |
| 45 | `cat2` | skip merge |  | relayout-rot=52 | 51 | 2 | enc2b_act->cat2:explicit:dp_add_input_alignment:mode=halo_local;up2->cat2:carry:layout_same_carry_halo:mode=halo_local |  |  |
| 46 | `dec2a` | Conv2d |  | rot=2304 | 25 | 1 | fused |  |  |
| 47 | `dec2a_act` | SiLU act |  | ct-ct=75 | 25 | 1 | carry |  |  |
| 48 | `bootstrap_after_dec2a_act` | Bootstrap |  | boot=25 | 25 | 1 | no CT-saving halo drop; target `dec2b` |  |  |
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

## Step 8: Primary Encoder4 No-Sharing E2E

Current main E2E comparison for the paper table: U22 encoder Conv2d stages only (`enc1a..enc4b`), no bottleneck and no decoder. Dense uses independent Orion LTs. HaloED provider uses encoder Conv2d provider lowering with native/halo output materialization and `ORION_UNIFIED_LT_INDIVIDUAL_EVAL=1`, so each LT preserves its own BSGS and there is no cross-LT shared-cache evaluation. Single-slot layer cache is used only for memory feasibility; `bootstrap_many` is disabled.

Results are written into the two marker-managed tables below:

- Network summary: `U22_BASE32_ENCODER4_NOSHARE_SUMMARY_TABLE`
- Per-layer timings: `U22_BASE32_ENCODER4_NOSHARE_E2E_TABLE`

Encoder summary split columns are sourced from
`operator_breakdown_after_forward.totals.*`; for older artifacts missing the
summary total, the runner falls back to the displayed per-layer/boot sums.
`MVM/LT s (incl pool)` is
`mvm_kernel_s`, so it includes the encoder pool rows reported by the runtime
breakdown; `diag+encode s` is `lt_layer_cache_encode_s`, and `unattributed s`
prefers `unattributed_he_forward_s`; when that key is absent, the runner
computes the residual from HE forward minus the displayed split columns.

<!-- U22_BASE32_ENCODER4_NOSHARE_SUMMARY_TABLE_START -->
| input | dataset | dense status | Halo status | dense HE forward s | Halo HE forward s | dense/Halo HE | dense MVM/LT s (incl pool) | Halo MVM/LT s (incl pool) | dense activation excl boot s | Halo activation excl boot s | dense bootstrap s | Halo bootstrap s | dense diag+encode s | Halo diag+encode s | dense unattributed s | Halo unattributed s | dense rotations | Halo rotations | dense diagonals | Halo diagonals | dense boots | Halo boots | dense RSS GiB | Halo RSS GiB | result files | note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 192x192 | IBSR BRAIN 2D | ok | started | 6360.7 |  |  | 3535.3 | 0.0 | 48.0 |  | 1455.2 | 0.0 | 638.3 |  | 1322.3 |  | 46,716 | 88,066 | 169,802 | 329,228 | 3 | 3 | 544.9 |  | dense:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/dense_encoder4_e2e.json; provider:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/provider_encoder4_e2e.json | Swaps: 0 	File system inputs: 0 	File system outputs: 270424 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0; blocks (rows, cols) = (47, 47) ├── resized matrix shape: (1540096, 1540096) ├── # output rotations: 0 ├── direct pack workers: 32 ├── time to pack (s): 9.16 ├── # diagonals = 52291 |
| 224x224 | HanCo Hand | ok | pending | 10110.2 |  |  | 6358.4 |  | 64.4 |  | 1956.5 |  | 940.4 |  | 1730.9 |  | 73,976 |  | 239,820 |  | 3 |  | 734.8 |  | dense:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/224_224/dense_encoder4_e2e.json; provider:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/224_224/provider_encoder4_e2e.json | Swaps: 0 	File system inputs: 0 	File system outputs: 373232 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 384x288 | CVC-ClinicDB | ok | pending | 23754.2 |  |  | 16061.4 |  | 139.1 |  | 4292.1 |  | 1970.4 |  | 3261.7 |  | 182,498 |  | 498,786 |  | 3 |  | 1327.1 |  | dense:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_288/dense_encoder4_e2e.json; provider:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_288/provider_encoder4_e2e.json | Swaps: 0 	File system inputs: 0 	File system outputs: 873024 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 384x384 | Satellite cloud | ok | pending | 29888.9 |  |  | 19801.8 |  | 185.8 |  | 5781.6 |  | 2643.0 |  | 4119.8 |  | 249,344 |  | 639,522 |  | 3 |  | 1397.6 |  | dense:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_384/dense_encoder4_e2e.json; provider:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_384/provider_encoder4_e2e.json | Swaps: 0 	File system inputs: 0 	File system outputs: 1212976 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
<!-- U22_BASE32_ENCODER4_NOSHARE_SUMMARY_TABLE_END -->

<!-- U22_BASE32_ENCODER4_NOSHARE_E2E_TABLE_START -->
| input | dataset | I/O ch | path | status | layer | groups | transforms | diagonals | rotations | diag+encode s | key prep s | evict s | turnover s | LT+accum s | eval total s | boot count | boot s | boot nodes | compile s | HE forward s | runtime mode | peak RSS GiB | result file | note |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | --- | --- |
| 192x192 | IBSR BRAIN 2D | 1->256 | dense | ok | TOTAL | 0 | 2,020 | 169,802 | 46,716 | 633.9 | 0.0 | 0.0 | 633.9 | 3308.1 | 3308.1 | 3 | 1455.2 |  | 200.2 | 6360.7 | single_slot_layer_cache | 544.9 | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/dense_encoder4_e2e.json | Swaps: 0 	File system inputs: 0 	File system outputs: 270424 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 192x192 | IBSR BRAIN 2D | 1->256 | dense | ok | enc1a | 0 | 72 | 860 | 756 | 3.9 | 0.0 | 0.0 | 3.9 | 84.7 | 84.7 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/dense_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | dense | ok | enc1b | 0 | 1,296 | 20,400 | 16,448 | 87.3 | 0.0 | 0.0 | 87.3 | 855.7 | 855.7 | 1 | 817.8 | enc1b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/dense_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | dense | ok | enc2a | 0 | 162 | 13,974 | 5,024 | 52.9 | 0.0 | 0.0 | 52.9 | 529.4 | 529.4 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/dense_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | dense | ok | enc2b | 0 | 324 | 27,948 | 10,048 | 104.7 | 0.0 | 0.0 | 104.7 | 502.6 | 502.6 | 1 | 422.5 | enc2b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/dense_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | dense | ok | enc3a | 0 | 45 | 16,857 | 2,913 | 59.5 | 0.0 | 0.0 | 59.5 | 387.4 | 387.4 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/dense_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | dense | ok | enc3b | 0 | 81 | 32,139 | 5,480 | 117.6 | 0.0 | 0.0 | 117.6 | 374.2 | 374.2 | 1 | 214.9 | enc3b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/dense_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | dense | ok | enc4a | 0 | 15 | 20,179 | 1,662 | 73.3 | 0.0 | 0.0 | 73.3 | 268.9 | 268.9 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/dense_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | dense | ok | enc4b | 0 | 25 | 37,445 | 2,953 | 134.7 | 0.0 | 0.0 | 134.8 | 305.1 | 305.1 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/dense_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | provider | started | TOTAL | 105 | 3,255 | 329,228 | 88,066 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3 | 0.0 |  | 252.3 |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/provider_encoder4_e2e.json | blocks (rows, cols) = (47, 47) ├── resized matrix shape: (1540096, 1540096) ├── # output rotations: 0 ├── direct pack workers: 32 ├── time to pack (s): 9.16 ├── # diagonals = 52291 |
| 192x192 | IBSR BRAIN 2D | 1->256 | provider | started | enc1a | 2 | 94 | 1,860 | 1,358 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/provider_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | provider | started | enc1b | 47 | 2,209 | 52,294 | 37,239 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  | pool1 |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/provider_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | provider | started | enc2a | 9 | 207 | 29,874 | 8,658 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/provider_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | provider | started | enc2b | 23 | 529 | 68,059 | 19,757 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  | pool2 |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/provider_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | provider | started | enc3a | 5 | 55 | 33,660 | 4,664 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/provider_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | provider | started | enc3b | 11 | 121 | 76,182 | 10,276 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  | pool3 |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/provider_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | provider | started | enc4a | 3 | 15 | 25,574 | 1,879 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/provider_encoder4_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->256 | provider | started | enc4b | 5 | 25 | 41,725 | 3,055 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/192_192/provider_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | dense | ok | TOTAL | 0 | 3,933 | 239,820 | 73,976 | 934.6 | 0.0 | 0.0 | 934.6 | 5844.4 | 5844.4 | 3 | 1956.5 |  | 309.8 | 10110.2 | single_slot_layer_cache | 734.8 | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/224_224/dense_encoder4_e2e.json | 	Swaps: 0 	File system inputs: 0 	File system outputs: 373232 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 224x224 | HanCo Hand | 3->256 | dense | ok | enc1a | 0 | 245 | 2,750 | 2,505 | 11.1 | 0.0 | 0.0 | 11.2 | 320.1 | 320.1 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/224_224/dense_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | dense | ok | enc1b | 0 | 2,401 | 28,155 | 25,224 | 122.3 | 0.0 | 0.0 | 122.3 | 1746.4 | 1746.4 | 1 | 1081.4 | enc1b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/224_224/dense_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | dense | ok | enc2a | 0 | 325 | 19,925 | 7,857 | 87.3 | 0.0 | 0.0 | 87.3 | 906.5 | 906.5 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/224_224/dense_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | dense | ok | enc2b | 0 | 625 | 39,109 | 15,389 | 164.1 | 0.0 | 0.0 | 164.1 | 881.5 | 881.5 | 1 | 570.0 | enc2b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/224_224/dense_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | dense | ok | enc3a | 0 | 91 | 24,009 | 4,519 | 84.6 | 0.0 | 0.0 | 84.6 | 589.1 | 589.1 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/224_224/dense_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | dense | ok | enc3b | 0 | 169 | 46,459 | 8,721 | 186.5 | 0.0 | 0.0 | 186.5 | 588.2 | 588.2 | 1 | 305.1 | enc3b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/224_224/dense_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | dense | ok | enc4a | 0 | 28 | 26,936 | 2,567 | 95.9 | 0.0 | 0.0 | 95.9 | 381.6 | 381.6 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/224_224/dense_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | dense | ok | enc4b | 0 | 49 | 52,477 | 4,903 | 182.7 | 0.0 | 0.0 | 182.7 | 430.9 | 430.9 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/224_224/dense_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | provider | pending | TOTAL | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/224_224/provider_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/224_224/provider_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/224_224/provider_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/224_224/provider_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/224_224/provider_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/224_224/provider_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/224_224/provider_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/224_224/provider_encoder4_e2e.json |  |
| 224x224 | HanCo Hand | 3->256 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/224_224/provider_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | dense | ok | TOTAL | 0 | 18,627 | 498,786 | 182,498 | 1956.0 | 0.0 | 0.1 | 1956.1 | 14567.4 | 14567.4 | 3 | 4292.1 |  | 274.4 | 23754.2 | single_slot_layer_cache | 1327.1 | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_288/dense_encoder4_e2e.json | 	Swaps: 0 	File system inputs: 0 	File system outputs: 873024 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 384x288 | CVC-ClinicDB | 3->256 | dense | ok | enc1a | 0 | 1,188 | 6,000 | 6,548 | 27.0 | 0.0 | 0.0 | 27.0 | 898.2 | 898.2 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_288/dense_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | dense | ok | enc1b | 0 | 11,664 | 61,584 | 66,720 | 252.8 | 0.0 | 0.1 | 252.9 | 5080.2 | 5080.2 | 1 | 2480.4 | enc1b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_288/dense_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | dense | ok | enc2a | 0 | 1,458 | 42,162 | 18,876 | 161.8 | 0.0 | 0.0 | 161.8 | 2177.4 | 2177.4 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_288/dense_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | dense | ok | enc2b | 0 | 2,916 | 84,324 | 37,752 | 370.0 | 0.0 | 0.0 | 370.1 | 2290.3 | 2290.3 | 1 | 1194.4 | enc2b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_288/dense_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | dense | ok | enc3a | 0 | 378 | 49,248 | 10,154 | 199.6 | 0.0 | 0.0 | 199.6 | 1248.0 | 1248.0 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_288/dense_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | dense | ok | enc3b | 0 | 729 | 96,921 | 19,974 | 408.0 | 0.0 | 0.0 | 408.0 | 1300.7 | 1300.7 | 1 | 617.4 | enc3b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_288/dense_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | dense | ok | enc4a | 0 | 98 | 52,539 | 5,305 | 170.8 | 0.0 | 0.0 | 170.8 | 707.4 | 707.4 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_288/dense_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | dense | ok | enc4b | 0 | 196 | 106,008 | 10,700 | 366.0 | 0.0 | 0.0 | 366.0 | 865.2 | 865.2 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_288/dense_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | provider | pending | TOTAL | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_288/provider_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_288/provider_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_288/provider_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_288/provider_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_288/provider_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_288/provider_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_288/provider_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_288/provider_encoder4_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->256 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_288/provider_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | dense | ok | TOTAL | 0 | 33,534 | 639,522 | 249,344 | 2623.7 | 0.0 | 0.2 | 2623.9 | 18074.5 | 18074.5 | 3 | 5781.6 |  | 228.1 | 29888.9 | single_slot_layer_cache | 1397.6 | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_384/dense_encoder4_e2e.json | 	Swaps: 0 	File system inputs: 0 	File system outputs: 1212976 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 384x384 | Satellite cloud | 4->256 | dense | ok | enc1a | 0 | 2,592 | 9,696 | 11,328 | 43.3 | 0.0 | 0.0 | 43.3 | 1399.9 | 1399.9 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_384/dense_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | dense | ok | enc1b | 0 | 20,736 | 77,568 | 90,624 | 314.9 | 0.0 | 0.1 | 315.0 | 5930.0 | 5930.0 | 1 | 3241.1 | enc1b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_384/dense_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | dense | ok | enc2a | 0 | 2,592 | 54,240 | 25,536 | 242.1 | 0.0 | 0.0 | 242.1 | 2719.4 | 2719.4 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_384/dense_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | dense | ok | enc2b | 0 | 5,184 | 108,480 | 51,072 | 470.6 | 0.0 | 0.0 | 470.6 | 2742.5 | 2742.5 | 1 | 1700.2 | enc2b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_384/dense_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | dense | ok | enc3a | 0 | 648 | 62,712 | 13,488 | 266.4 | 0.0 | 0.0 | 266.4 | 1635.9 | 1635.9 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_384/dense_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | dense | ok | enc3b | 0 | 1,296 | 125,424 | 26,976 | 548.6 | 0.0 | 0.0 | 548.6 | 1587.6 | 1587.6 | 1 | 840.3 | enc3b_act |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_384/dense_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | dense | ok | enc4a | 0 | 162 | 67,134 | 6,924 | 233.1 | 0.0 | 0.0 | 233.1 | 944.9 | 944.9 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_384/dense_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | dense | ok | enc4b | 0 | 324 | 134,268 | 13,848 | 504.7 | 0.0 | 0.0 | 504.7 | 1114.2 | 1114.2 |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_dense_only_qiuchu_20260528T184853Z/384_384/dense_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | provider | pending | TOTAL | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_384/provider_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_384/provider_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_384/provider_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_384/provider_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_384/provider_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_384/provider_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_384/provider_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_384/provider_encoder4_e2e.json |  |
| 384x384 | Satellite cloud | 4->256 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_encoder4_noshare_e2e_qiuchu_20260528T161318Z_noshare/384_384/provider_encoder4_e2e.json |  |
<!-- U22_BASE32_ENCODER4_NOSHARE_E2E_TABLE_END -->
