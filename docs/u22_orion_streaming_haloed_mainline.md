# U22 Orion Dense/Streaming vs HaloED Mainline

This document is the working plan for the current U-Net 22 comparison.

Current active gate, as of May 30, 2026:

- Conv kernel table on `amd:/home/abc/orion`
- Lattigo kernel profile, `LogN=16`, default resident LT with `io_mode=none`
- `HW=4` means the four current input sizes: `192x192`, `224x224`, `384x288`, `384x384`
- Kernels: the four existing stage-packed kernels plus U22 `base_dim=32`
  same-in/out `dec1b`, `dec2b`, `dec3b`, `dec4b`, and `bottleneckb`
  convs, each `3x3/pad1/stride1`
- Active variants: dense, beta=1 no-sharing, and beta=1 no-sharing stripe.
  Do not run beta=2 in this table.
- Metrics: runtime rotations and `LT+accumulate s`; no streamed encode time in this table
- Execution containment: one worker per table row with an RSS watchdog

The `HW` column is the original U-Net input size, not necessarily the logical
Conv input size. The existing four synthetic kernels simulate encoder-stage
packing:

- `Conv 32,32`: logical `32 x H x W`, multiplex/input/output gap `1`, packed FHE `32 x H x W`
- `Conv 64,64`: logical `64 x H/2 x W/2`, multiplex/input/output gap `2`, four channels per packed group, packed FHE `16 x H x W`
- `Conv 128,128`: logical `128 x H/4 x W/4`, multiplex/input/output gap `4`, sixteen channels per packed group, packed FHE `8 x H x W`
- `Conv 256,256`: logical `256 x H/8 x W/8`, multiplex/input/output gap `8`, sixty-four channels per packed group, packed FHE `4 x H x W`

The U22 `base_dim=32` same-in/out layer cases use the matching decoder and
bottleneck packing:

- `dec1b`: logical `32 x H x W`, gap `1`, packed FHE `32 x H x W`
- `dec2b`: logical `64 x H/2 x W/2`, gap `2`, packed FHE `16 x H x W`
- `dec3b`: logical `128 x H/4 x W/4`, gap `4`, packed FHE `8 x H x W`
- `dec4b`: logical `256 x H/8 x W/8`, gap `8`, packed FHE `4 x H x W`
- `bottleneckb`: logical `512 x H/16 x W/16`, gap `16`, packed FHE `2 x H x W`

The completed dense resident run is the baseline provenance for this
kernel-table gate.  Reuse only exact-shape, non-stream dense rows; the active
provider rows must be measured as beta=1 no-sharing rows. The non-stripe
provider row uses `native_halo_channel_fold_mode=heuristic`; the stripe row
uses `native_halo_channel_fold_mode=per_stripe`. Do not mix in older shared
provider, beta=2 provider, or stale corg JSONs.

For the kernel table only, use `amd:/home/abc/orion`; later dense/provider
E2E bookkeeping sections may still reference older corg jobs.

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

Streaming does not mean disabling BSGS.  The intended baseline is now a single-slot layer cache: compile records diagonal-index metadata, rotation-key plans, and runtime diagonal recipes without keeping raw diagonal payloads or encoded plaintexts; memory-bounded E2E runtime should materialize/encode only the current dense LT or bounded dense LT group, evaluate it, and evict it before the next LT/group.  The default `ORION_DENSE_LAYER_CACHE_GRANULARITY=layer` preserves old layer-at-a-time behavior; use `ORION_DENSE_LAYER_CACHE_GRANULARITY=lt` or `group` plus `ORION_DENSE_LAYER_CACHE_GROUP_TRANSFORMS=N` for fixed high-watermark U-Net runs, or `ORION_DENSE_LAYER_CACHE_GROUP_TRANSFORMS=auto` for the E2E single-slot default.  `auto` batches adjacent independent dense LTs, and no-sharing provider `UnifiedTransformGroup`s, up to a conservative host-memory budget so Lattigo can encode several transforms per batch instead of paying full per-LT overhead.  It is a runtime materialization batching policy, not a new dense/provider strategy: execution order is unchanged, cross-block sharing is not introduced, and BSGS stays inside each materialized LT/group.  If a single LT/group already exceeds the auto group budget, it is treated as an unsplittable singleton and may run by itself; the budget limits only additional co-materialization.  Orion dense remains independent LT evaluation with BSGS only inside each materialized LT; HaloED/provider uses grouped provider kernels where the lowering exposes shared-source structure.  Report layer compute time separately from `layer_cache_turnover_s` (`layer_cache_encode_s + layer_cache_key_prepare_s + layer_cache_evict_s`).  Legacy chunked Lattigo LT streaming is not the mainline and requires the explicit `ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT=1` gate.  Kernel-level benchmarks remain resident by default; single-slot is only for full-network/E2E memory-bounded evaluation.

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

Goal: compare default resident Orion dense LT against beta=1 no-sharing
HaloED/provider Conv kernels for the four existing stage-packed kernels plus
U22 `base_dim=32` same-in/out `dec1b..dec4b` and `bottleneckb`, across the
four HW sizes above. Each provider row disables cross-LT shared rotations and
preserves BSGS within each individual LT; the stripe row additionally uses
per-stripe native-halo channel fold.
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
used the short kernel chain/max input level and is not main evidence. The
superseded corg beta=2 retry/audit runs are not part of this AMD table.

Run command on `amd`:

```bash
cd /home/abc/orion
mkdir -p .tmp/run .tmp/logs .tmp/results
cpu_count="$(nproc)"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
run_root=".tmp/results/conv_kernel_table_base32_threeway_level2_e2e_amd_${ts}"
log=".tmp/logs/conv_kernel_table_base32_threeway_level2_e2e_amd_${ts}.log"
{
  echo "run_root=$run_root"
  echo "log=$log"
  echo "started_at_utc=$(date -u +%FT%TZ)"
  echo "ckks_profile=resnet_e2e_logn16_logscale40_h192"
  echo "input_level=2"
  echo "expected_output_level=1"
} > .tmp/run/conv_kernel_table_base32_threeway_level2_e2e.latest
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
ORION_CPP_DIAG_BUILDER=1 \
ORION_CPP_DIAG_BUILDER_DENSE=1 \
ORION_CPP_DIAG_BUILDER_PROVIDER=1 \
ORION_CPP_DIAG_BUILDER_PROVIDER_NATIVE_SOURCE=1 \
ORION_CPP_DIAG_BUILDER_PROVIDER_COMPACT_SOURCE=1 \
ORION_CPP_DIAG_BUILDER_STRICT=1 \
ORION_DIAG_BUILDER_LIB=/home/abc/orion/orion/backend/diag_builder/diag_builder-linux.so \
PYTHONUNBUFFERED=1 \
nohup .venv/bin/python tools/run_conv_kernel_table.py \
  --run-root "$run_root" \
  --doc docs/u22_orion_streaming_haloed_mainline.md \
  --backend lattigo \
  --max-worker-rss-gb 1000 \
  --kernel-cases conv32 conv64 conv128 conv256 dec1b dec2b dec3b dec4b bottleneckb \
  --hw 192x192 224x224 384x288 384x384 \
  --variants orion provider_halo1_no_share provider_halo1_individual_lt \
  --provider-output-layout native_stripe \
  --input-level 2 \
  > "$log" 2>&1 &
echo "pid=$!" >> .tmp/run/conv_kernel_table_base32_threeway_level2_e2e.latest
```

Refresh locally:

```bash
run_root="$(ssh amd 'cd /home/abc/orion && awk -F= '\''$1=="run_root"{print $2}'\'' .tmp/run/conv_kernel_table_base32_threeway_level2_e2e.latest')"
base="$(basename "$run_root")"
mkdir -p ".tmp/results/$base"
rsync -az --delete --checksum "amd:/home/abc/orion/$run_root/" ".tmp/results/$base/"
.venv/bin/python tools/run_conv_kernel_table.py --update-doc-only \
  --run-root ".tmp/results/$base" \
  --doc docs/u22_orion_streaming_haloed_mainline.md \
  --kernel-cases conv32 conv64 conv128 conv256 dec1b dec2b dec3b dec4b bottleneckb \
  --hw 192x192 224x224 384x288 384x384 \
  --variants orion provider_halo1_no_share provider_halo1_individual_lt \
  --provider-output-layout native_stripe \
  --input-level 2
```

Rotation/runtime audit: speedup in this kernel table should be interpreted as
primarily rotation-driven.  To check apparent counterexamples, all rows whose
beta=2-vs-beta=1 no-sharing stripe hot runtime direction disagreed with the
rotation direction were rerun on AMD under
`.tmp/repro/mismatch_rotation_runtime_amd_20260531T020037Z` with three hot
runs per row.  With those reruns overriding the stale candidates, the 36
matched beta pairs show `rotation_eval_count` as the strongest runtime-ratio
predictor (`Pearson log-ratio=0.766`, `Spearman=0.663`); transform-key total is
similar (`0.765` / `0.674`).  The prior large `dec3b 192x192` beta=2 speedup is
not reproducible and should not be cited.

| row | beta2/beta1 rotations | beta2/beta1 hot runtime | audit outcome |
| --- | ---: | ---: | --- |
| `bottleneckb 192x192` | 1.051 | 0.898 | residual opposite-sign case, but hot runs are highly variable (`52.2/37.5/37.7` vs `49.4/31.0/34.0`), so treat as noise pending more repeats |
| `bottleneckb 384x384` | 1.088 | 1.157 | aligned with rotations |
| `conv128 192x192` | 1.033 | 1.124 | aligned with rotations |
| `dec1b 192x192` | 1.021 | 1.004 | effectively flat/noise |
| `dec3b 192x192` | 1.033 | 1.201 | aligned with rotations; no large beta=2 speedup reproduced |

### AMD 256x256 internal-halo beta sweep

Run root:
`.tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z`.
This 256x256 operator-level sweep uses input level 2, `native_halo_stripe`
provider output layout, `per_stripe` channel fold, individual LT grouping, and
shared provider rotations disabled.  Provider rows keep the requested beta label
but clip global top/bottom boundary input halo (`input halo T/B = 0/0`), so the
reported halo/ct counts reflect internal stripe halo only.

| kernel | path | rotations | LT+acc s | hot s | speed vs dense (LT) | speed vs dense (hot) | input/output ct | input halo T/B | result file |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| conv1 (Conv 32,32) | dense | 16,384 | 221.0 | 227.0 | 1.00x | 1.00x | 64/64 | - | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv32_256x256_orion.json |
| conv1 (Conv 32,32) | beta=1 stripe | 10,278 | 123.5 | 132.9 | 1.79x | 1.71x | 65/65 | 0/0 (beta=1, clipped=True) | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv32_256x256_provider_halo1_individual_lt.json |
| conv1 (Conv 32,32) | beta=2 stripe | 10,272 | 113.5 | 122.9 | 1.95x | 1.85x | 65/65 | 0/0 (beta=2, clipped=True) | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv32_256x256_provider_halo2_individual_lt.json |
| conv2 (Conv 64,64) | dense | 11,264 | 173.6 | 175.1 | 1.00x | 1.00x | 32/32 | - | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv64_256x256_orion.json |
| conv2 (Conv 64,64) | beta=1 stripe | 6,714 | 89.1 | 91.3 | 1.95x | 1.92x | 33/33 | 0/0 (beta=1, clipped=True) | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv64_256x256_provider_halo1_individual_lt.json |
| conv2 (Conv 64,64) | beta=2 stripe | 6,714 | 92.3 | 94.9 | 1.88x | 1.85x | 33/33 | 0/0 (beta=2, clipped=True) | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv64_256x256_provider_halo2_individual_lt.json |
| conv3 (Conv 128,128) | dense | 6,400 | 112.1 | 112.3 | 1.00x | 1.00x | 16/16 | - | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv128_256x256_orion.json |
| conv3 (Conv 128,128) | beta=1 stripe | 3,802 | 66.4 | 67.0 | 1.69x | 1.68x | 17/17 | 0/0 (beta=1, clipped=True) | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv128_256x256_provider_halo1_individual_lt.json |
| conv3 (Conv 128,128) | beta=2 stripe | 3,802 | 72.6 | 73.2 | 1.54x | 1.53x | 17/17 | 0/0 (beta=2, clipped=True) | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv128_256x256_provider_halo2_individual_lt.json |
| conv4 (Conv 256,256) | dense | 3,392 | 96.0 | 96.1 | 1.00x | 1.00x | 8/8 | - | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv256_256x256_orion.json |
| conv4 (Conv 256,256) | beta=1 stripe | 2,076 | 52.8 | 52.9 | 1.82x | 1.82x | 9/9 | 0/0 (beta=1, clipped=True) | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv256_256x256_provider_halo1_individual_lt.json |
| conv4 (Conv 256,256) | beta=2 stripe | 2,076 | 55.6 | 55.7 | 1.73x | 1.73x | 9/9 | 0/0 (beta=2, clipped=True) | .tmp/results/conv_kernel_table_256_dense_beta1_beta2_internal_halo_amd_20260602T055202Z/rows/conv256_256x256_provider_halo2_individual_lt.json |

<!-- CONV_KERNEL_TABLE_START -->
| HW | kernel | logical input | multiplex | channels/group | packed FHE input | path / beta | status | input level | expected output level | actual output level | input halo T/B | output layout | channel fold | LT grouping | rotations | LT+accumulate s | hot run s | compile s | diag build s | diag shadow s | input ct | output ct | peak RSS GiB | runtime mode | result file | note |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 192x192 | Conv 32,32 | 32x192x192 | 1 | 1 | 32x192x192 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 8,140 | 107.9 | 113.2 | 70.0 | 4.0 | 0.0 | 38 | 38 | 114.0 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv32_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | Conv 32,32 | 32x224x224 | 1 | 1 | 32x224x224 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 12,288 | 167.6 | 178.8 | 88.7 | 5.8 | 0.0 | 64 | 64 | 115.8 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv32_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | Conv 32,32 | 32x384x288 | 1 | 1 | 32x384x288 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 23,552 | 298.8 | 313.4 | 169.6 | 11.5 | 0.0 | 112 | 112 | 228.4 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv32_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | Conv 32,32 | 32x384x384 | 1 | 1 | 32x384x384 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 30,720 | 402.5 | 426.0 | 240.8 | 15.8 | 0.0 | 160 | 160 | 282.9 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv32_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | Conv 64,64 | 64x96x96 | 2 | 4 | 16x192x192 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 5,536 | 81.3 | 82.5 | 60.4 | 3.8 | 0.0 | 20 | 20 | 141.9 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv64_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | Conv 64,64 | 64x112x112 | 2 | 4 | 16x224x224 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 8,192 | 119.5 | 121.9 | 65.4 | 4.2 | 0.0 | 32 | 32 | 150.9 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv64_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | Conv 64,64 | 64x192x144 | 2 | 4 | 16x384x288 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 16,384 | 226.1 | 230.3 | 126.2 | 8.9 | 0.0 | 64 | 64 | 301.9 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv64_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | Conv 64,64 | 64x192x192 | 2 | 4 | 16x384x384 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 20,480 | 301.2 | 307.9 | 151.4 | 11.4 | 0.0 | 80 | 80 | 373.9 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv64_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | Conv 128,128 | 128x48x48 | 4 | 16 | 8x192x192 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 3,124 | 61.9 | 62.2 | 69.1 | 4.1 | 0.0 | 11 | 10 | 160.9 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv128_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | Conv 128,128 | 128x56x56 | 4 | 16 | 8x224x224 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 4,608 | 86.0 | 86.6 | 67.6 | 3.9 | 0.0 | 16 | 16 | 166.4 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv128_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | Conv 128,128 | 128x96x72 | 4 | 16 | 8x384x288 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 9,216 | 167.3 | 168.7 | 138.5 | 8.3 | 0.0 | 32 | 32 | 323.2 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv128_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | Conv 128,128 | 128x96x96 | 4 | 16 | 8x384x384 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 12,008 | 222.1 | 223.6 | 195.0 | 12.0 | 0.0 | 42 | 42 | 446.2 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv128_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | Conv 256,256 | 256x24x24 | 8 | 64 | 4x192x192 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 1,948 | 59.6 | 59.7 | 101.9 | 5.7 | 0.0 | 6 | 6 | 211.4 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv256_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | Conv 256,256 | 256x28x28 | 8 | 64 | 4x224x224 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 2,432 | 65.6 | 65.7 | 67.1 | 6.5 | 0.0 | 8 | 8 | 180.5 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv256_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | Conv 256,256 | 256x48x36 | 8 | 64 | 4x384x288 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 5,384 | 148.6 | 149.0 | 181.8 | 14.7 | 0.0 | 18 | 17 | 391.4 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv256_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | Conv 256,256 | 256x48x48 | 8 | 64 | 4x384x384 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 7,944 | 202.5 | 202.9 | 225.4 | 20.0 | 0.0 | 26 | 26 | 560.8 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/conv256_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | dec1b Conv 32,32 | 32x192x192 | 1 | 1 | 32x192x192 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 8,140 | 105.0 | 109.6 | 69.0 | 4.0 | 0.0 | 38 | 38 | 113.9 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec1b_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | dec1b Conv 32,32 | 32x224x224 | 1 | 1 | 32x224x224 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 12,288 | 159.2 | 169.6 | 87.5 | 5.6 | 0.0 | 64 | 64 | 118.5 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec1b_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | dec1b Conv 32,32 | 32x384x288 | 1 | 1 | 32x384x288 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 23,552 | 293.0 | 308.8 | 128.2 | 11.2 | 0.0 | 112 | 112 | 227.4 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec1b_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | dec1b Conv 32,32 | 32x384x384 | 1 | 1 | 32x384x384 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 30,720 | 394.5 | 418.3 | 186.5 | 15.4 | 0.0 | 160 | 160 | 285.1 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec1b_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | dec2b Conv 64,64 | 64x96x96 | 2 | 4 | 16x192x192 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 5,536 | 81.1 | 82.3 | 64.7 | 3.6 | 0.0 | 20 | 20 | 142.7 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec2b_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | dec2b Conv 64,64 | 64x112x112 | 2 | 4 | 16x224x224 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 8,192 | 115.2 | 117.5 | 65.4 | 4.2 | 0.0 | 32 | 32 | 160.0 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec2b_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | dec2b Conv 64,64 | 64x192x144 | 2 | 4 | 16x384x288 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 16,384 | 224.1 | 229.0 | 132.3 | 8.2 | 0.0 | 64 | 64 | 298.5 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec2b_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | dec2b Conv 64,64 | 64x192x192 | 2 | 4 | 16x384x384 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 20,480 | 280.2 | 285.6 | 159.4 | 11.3 | 0.0 | 80 | 80 | 372.3 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec2b_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | dec3b Conv 128,128 | 128x48x48 | 4 | 16 | 8x192x192 | no-sharing stripe | ok | 2 | 1 | 1 | 1/1 | native_halo_stripe | per_stripe | individual | 3,024 | 48.3 | 44.2 | 277.8 | 3.9 | 0.0 | 10 | 10 | 165.4 | resident_compute | .tmp/repro/dec3b_beta_lt_profile_amd_20260531T013434Z/rows/dec3b_192x192_provider_halo1_individual_lt.json | profile rerun; hot runs 48.6/40.9/43.0; output halo 0/0 |
| 192x192 | dec3b Conv 128,128 | 128x48x48 | 4 | 16 | 8x192x192 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 3,124 | 47.9 | 50.5 | 282.4 | 4.4 | 0.0 | 11 | 10 | 178.1 | resident_compute | .tmp/repro/dec3b_beta_lt_profile_amd_20260531T013434Z/rows/dec3b_192x192_provider_halo2_individual_lt.json | profile rerun; hot runs 48.2/48.6/54.8; output halo 1/1; no large speedup reproduced |
| 224x224 | dec3b Conv 128,128 | 128x56x56 | 4 | 16 | 8x224x224 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 4,608 | 86.0 | 86.6 | 65.4 | 4.0 | 0.0 | 16 | 16 | 169.5 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec3b_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | dec3b Conv 128,128 | 128x96x72 | 4 | 16 | 8x384x288 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 9,216 | 167.6 | 168.8 | 141.1 | 8.3 | 0.0 | 32 | 32 | 337.1 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec3b_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | dec3b Conv 128,128 | 128x96x96 | 4 | 16 | 8x384x384 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 12,008 | 219.4 | 221.0 | 191.0 | 11.6 | 0.0 | 42 | 42 | 445.6 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec3b_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | dec4b Conv 256,256 | 256x24x24 | 8 | 64 | 4x192x192 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 1,948 | 54.1 | 54.2 | 100.9 | 6.0 | 0.0 | 6 | 6 | 210.8 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec4b_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | dec4b Conv 256,256 | 256x28x28 | 8 | 64 | 4x224x224 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 2,432 | 65.3 | 65.4 | 67.3 | 6.4 | 0.0 | 8 | 8 | 180.5 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec4b_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | dec4b Conv 256,256 | 256x48x36 | 8 | 64 | 4x384x288 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 5,384 | 147.3 | 147.7 | 178.5 | 14.5 | 0.0 | 18 | 17 | 352.8 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec4b_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | dec4b Conv 256,256 | 256x48x48 | 8 | 64 | 4x384x384 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 7,944 | 224.3 | 224.8 | 223.2 | 19.4 | 0.0 | 26 | 26 | 563.8 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/dec4b_384x384_provider_halo2_individual_lt.json |  |
| 192x192 | bottleneckb Conv 512,512 | 512x12x12 | 16 | 256 | 2x192x192 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 1,248 | 48.7 | 48.8 | 73.2 | 6.5 | 0.0 | 4 | 4 | 193.1 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/bottleneckb_192x192_provider_halo2_individual_lt.json |  |
| 224x224 | bottleneckb Conv 512,512 | 512x14x14 | 16 | 256 | 2x224x224 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 1,704 | 71.5 | 71.5 | 125.8 | 9.3 | 0.0 | 6 | 5 | 269.2 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/bottleneckb_224x224_provider_halo2_individual_lt.json |  |
| 384x288 | bottleneckb Conv 512,512 | 512x24x18 | 16 | 256 | 2x384x288 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 3,408 | 138.3 | 138.4 | 218.3 | 19.1 | 0.0 | 12 | 10 | 541.7 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/bottleneckb_384x288_provider_halo2_individual_lt.json |  |
| 384x384 | bottleneckb Conv 512,512 | 512x24x24 | 16 | 256 | 2x384x384 | provider beta=2 no-share stripe | ok | 2 | 1 | 1 | 2/2 | native_halo_stripe | per_stripe | individual | 5,432 | 221.0 | 221.1 | 313.9 | 25.4 | 0.0 | 18 | 17 | 690.6 | resident_compute | .tmp/results/conv_kernel_table_base32_provider_halo2_stripe_level2_e2e_amd_20260530T121801Z/rows/bottleneckb_384x384_provider_halo2_individual_lt.json |  |
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

- Compile prepares only diagonal-index metadata, rotation-key plans, and runtime recipes.
- Runtime with `ORION_DENSE_LAYER_CACHE_GRANULARITY=lt` materializes one dense LT at a time; `group` materializes a bounded number of independent dense LTs.
- BSGS is preserved inside each materialized dense LT; dense does not adopt provider shared-cache semantics.
- Accumulate all input-column LT results for an output row, then rescale once and apply bias/output rotations as before.
- After each LT/group evaluates, remove plaintext diagonals and delete the backend transform before moving on.
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

E2E single-slot runs should leave `ORION_DENSE_LAYER_CACHE_GROUP_TRANSFORMS=auto`
for both dense and provider modes unless a task explicitly asks for a fixed
stress probe.  The auto group policy exists to reduce the `diag_encode`/LT
materialization bottleneck by batching several independent transforms into one
Lattigo generation call when the estimated encoded-plaintext footprint fits the
current host-memory budget.  Dense still evaluates independent LTs in row/column
order, and provider no-share-fold still evaluates each `UnifiedTransformGroup`
in the original runtime-group order; auto only decides how many of those
already-independent materializations are encoded before the next evaluation
sequence begins.  On no-sharing provider runs, this matters because each
provider runtime group commonly contains a single LT, so provider otherwise pays
the same one-LT-at-a-time encode overhead as dense.  If one LT/group by itself is
larger than the auto budget, do not split it further and do not classify that as
a policy failure: it is an unsplittable singleton and may bypass the auto budget
while still being measured as a single materialized LT/group.  Report the
resulting `layer_cache_group_policy`, `layer_cache_group_budget_bytes`,
`layer_cache_group_count`, and max estimated group bytes when present; keep
`layer_cache_encode_s`, `layer_cache_key_prepare_s`, and `layer_cache_evict_s`
separate from strict compute time.

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
breakdown, not only named Conv2d/TConv layers.  Provider native executor
`group_eval_s`/`evaluate_unified_s` are inclusive diagnostics only; provider
MVM is sourced from the underlying runtime groups' `eval_s`/stream counters.
`MVM+act+boot s` is the strict compute-comparison sum
`mvm_kernel_s + activation_s + bootstrap_s`; single-slot `layer-cache turnover
s`, executor overhead, and runtime load/trim are real `he_forward_s` wall time
but are reported separately from that compute metric.
`wall residual s` prefers `wall_unattributed_he_forward_s`; when that key is
absent, the runner computes the residual from HE forward minus strict compute
plus the displayed wall auxiliary columns.

Current dense rows may be retained from the serial qiuchu run root.  Provider
rows are accepted for the no-share-fold comparison only when their runner
metadata records `policy=dp_no_share_fold`,
`ORION_UNIFIED_LT_INDIVIDUAL_EVAL=1`,
`ORION_UNIFIED_LT_SHARED_ROTATION_KEYS=0`, and
`ORION_LATTIGO_UNIFIED_NO_BSGS=0`; older provider placeholders from `policy=dp`
must be replaced before drawing dense/provider conclusions.

<!-- U22_BASE32_SILU7_NETWORK_SUMMARY_TABLE_START -->
| input | dataset | I/O ch | dense status | Halo status | dense HE forward s | Halo HE forward s | dense/Halo HE | dense hot E2E s | Halo hot E2E s | dense MVM/LT s (incl pool) | Halo MVM/LT s (incl pool) | dense activation excl boot s | Halo activation excl boot s | dense bootstrap s | Halo bootstrap s | dense MVM+act+boot s | Halo MVM+act+boot s | dense diag+encode s | Halo diag+encode s | dense layer-cache turnover s | Halo layer-cache turnover s | dense executor overhead s | Halo executor overhead s | dense load/trim s | Halo load/trim s | dense wall residual s | Halo wall residual s | dense rotations | Halo rotations | dense boots | Halo boots | dense RSS GiB | Halo RSS GiB | runtime mode | result files | note |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 192x192 | IBSR BRAIN 2D | 1->4 | ok | running | 21447.7 |  |  | 21447.9 |  | 13783.0 |  | 97.4 |  | 2399.1 |  | 16279.5 |  | 3390.3 |  | 3390.4 |  | 0.0 |  | 0.0 |  | 1777.7 |  | 102,699 |  | 8 |  | 699.7 |  | single_slot_layer_cache | dense:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json; provider:.tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json | Swaps: 0 	File system inputs: 0 	File system outputs: 369632 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0; 	Swaps: 0 	File system inputs: 0 	File system outputs: 860072 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 224x224 | HanCo Hand | 3->1 | ok | pending | 4286.4 |  |  | 4286.5 |  | 346.6 |  | 0.2 |  | 0.0 |  | 346.8 |  | 3910.9 |  | 3918.4 |  | 0.0 |  | 0.0 |  | 21.3 |  | 571,703 |  | 8 |  | 258.9 |  | single_slot_layer_cache | dense:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json; provider:/home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json | Swaps: 0 	File system inputs: 16 	File system outputs: 699192 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 384x288 | CVC-ClinicDB | 3->1 | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dense:.tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json; provider:.tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | pending | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dense:.tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json; provider:.tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
<!-- U22_BASE32_SILU7_NETWORK_SUMMARY_TABLE_END -->

<!-- U22_BASE32_SILU7_STREAMING_PROVIDER_E2E_TABLE_START -->
| input | dataset | I/O ch | path | status | layer | groups/transforms | layer cache turnover s | layer cache diag+encode s | layer cache key prep s | layer cache evict s | LT+accum s | eval total s | legacy load/encode s | stream build s | encode hoist s | stream load s | LT eval s | LT accum s | baby+giant s | boot after count | boot after s | boot after nodes | compile s | HE forward s | hot E2E s | rotations | boots | runtime mode | peak RSS GiB | result file | note |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | TOTAL | 159 | 3390.4 | 3390.3 | 0.0 | 0.1 | 13783.0 | 13783.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  | 8 | 2399.1 |  | 317.4 | 21447.7 | 21447.9 | 102,699 | 8 | single_slot_layer_cache | 699.7 | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json | Swaps: 0 	File system inputs: 0 	File system outputs: 369632 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc1a | 1 | 4.1 | 4.1 | 0.0 | 0.0 | 69.1 | 69.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc1b | 1 | 83.9 | 83.8 | 0.0 | 0.0 | 633.7 | 633.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 837.3 | enc1b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc2a | 1 | 49.9 | 49.8 | 0.0 | 0.0 | 495.4 | 495.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc2b | 1 | 109.8 | 109.8 | 0.0 | 0.0 | 437.0 | 437.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 420.9 | enc2b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc3a | 1 | 60.6 | 60.6 | 0.0 | 0.0 | 380.3 | 380.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc3b | 1 | 115.0 | 115.0 | 0.0 | 0.0 | 376.4 | 376.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 217.3 | enc3b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc4a | 1 | 72.5 | 72.5 | 0.0 | 0.0 | 322.6 | 322.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | enc4b | 1 | 146.5 | 146.5 | 0.0 | 0.0 | 333.3 | 333.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 112.2 | enc4b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | bottlenecka | 1 | 93.5 | 93.5 | 0.0 | 0.0 | 266.5 | 266.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | bottleneckb | 1 | 157.0 | 157.0 | 0.0 | 0.0 | 317.0 | 317.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 69.0 | bottleneckb_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | up4 | 1 | 75.5 | 75.5 | 0.0 | 0.0 | 305.6 | 305.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec4a | 10 | 904.8 | 904.7 | 0.0 | 0.0 | 952.9 | 952.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec4b | 1 | 132.1 | 132.1 | 0.0 | 0.0 | 334.7 | 334.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 117.1 | dec4b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | up3 | 1 | 64.5 | 64.5 | 0.0 | 0.0 | 382.0 | 382.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec3a | 18 | 430.7 | 430.7 | 0.0 | 0.0 | 1370.2 | 1370.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec3b | 1 | 120.1 | 120.1 | 0.0 | 0.0 | 379.1 | 379.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 205.4 | dec3b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | up2 | 1 | 56.6 | 56.6 | 0.0 | 0.0 | 524.6 | 524.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec2a | 36 | 284.2 | 284.2 | 0.0 | 0.0 | 1621.5 | 1621.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec2b | 1 | 115.7 | 115.7 | 0.0 | 0.0 | 479.5 | 479.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 419.9 | dec2b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | up1 | 1 | 39.1 | 39.1 | 0.0 | 0.0 | 793.7 | 793.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec1a | 72 | 188.6 | 188.6 | 0.0 | 0.0 | 2180.7 | 2180.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | dec1b | 1 | 78.9 | 78.9 | 0.0 | 0.0 | 659.7 | 659.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | dense | ok | output | 1 | 0.9 | 0.9 | 0.0 | 0.0 | 7.1 | 7.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/192_192/dense_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | running | TOTAL |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 192x192 | IBSR BRAIN 2D | 1->4 | provider | no-row | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_e2e_192_local_20260530T101914Z/192_192/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | TOTAL | 211 | 3918.4 | 3910.9 | 0.0 | 7.5 | 346.6 | 346.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  | 8 | 0.0 |  | 547.1 | 4286.4 | 4286.5 | 571,703 | 8 | single_slot_layer_cache | 258.9 | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json | Swaps: 0 	File system inputs: 16 	File system outputs: 699192 	Socket messages sent: 0 	Socket messages received: 0 	Signals delivered: 0 	Page size (bytes): 4096 	Exit status: 0 |
| 224x224 | HanCo Hand | 3->1 | dense | ok | enc1a | 1 | 42.6 | 42.6 | 0.0 | 0.0 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | enc1b | 1 | 593.8 | 593.7 | 0.0 | 0.1 | 5.4 | 5.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | enc1b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | enc2a | 1 | 84.7 | 84.7 | 0.0 | 0.0 | 6.6 | 6.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | enc2b | 1 | 214.1 | 214.1 | 0.0 | 0.0 | 12.7 | 12.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | enc2b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | enc3a | 1 | 41.3 | 40.9 | 0.0 | 0.3 | 9.6 | 9.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | enc3b | 1 | 101.3 | 101.0 | 0.0 | 0.3 | 18.8 | 18.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | enc3b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | enc4a | 1 | 28.9 | 28.5 | 0.0 | 0.4 | 11.5 | 11.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | enc4b | 1 | 66.7 | 66.0 | 0.0 | 0.7 | 22.6 | 22.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | enc4b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | bottlenecka | 1 | 28.9 | 28.5 | 0.0 | 0.4 | 12.0 | 12.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | bottleneckb | 1 | 62.1 | 61.5 | 0.0 | 0.6 | 25.5 | 25.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | bottleneckb_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | up4 | 1 | 23.5 | 23.2 | 0.0 | 0.3 | 11.3 | 11.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | dec4a | 14 | 824.4 | 820.5 | 0.0 | 3.8 | 52.4 | 52.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | dec4b | 1 | 63.2 | 63.2 | 0.0 | 0.0 | 24.1 | 24.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | dec4b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | up3 | 1 | 36.0 | 36.0 | 0.0 | 0.0 | 9.5 | 9.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | dec3a | 26 | 168.0 | 167.8 | 0.0 | 0.2 | 45.2 | 45.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | dec3b | 1 | 101.2 | 101.2 | 0.0 | 0.0 | 17.7 | 17.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | dec3b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | up2 | 1 | 85.8 | 85.8 | 0.0 | 0.0 | 6.9 | 6.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | dec2a | 50 | 65.0 | 64.9 | 0.0 | 0.1 | 19.6 | 19.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | dec2b | 1 | 220.8 | 220.8 | 0.0 | 0.0 | 13.7 | 13.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | dec2b_act |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | up1 | 1 | 256.6 | 256.6 | 0.0 | 0.0 | 2.5 | 2.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | dec1a | 98 | 38.0 | 37.9 | 0.0 | 0.1 | 12.8 | 12.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | dec1b | 1 | 601.6 | 601.5 | 0.0 | 0.1 | 4.9 | 4.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | dense | ok | output | 1 | 16.7 | 16.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  | single_slot_layer_cache |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_dense224_cppdiag_single_slot_qiuchu_20260530T085900Z/224_224/dense_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 224x224 | HanCo Hand | 3->1 | provider | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | /home/anakano/PycharmProjects/orion/.tmp/results/u22_dim32_full_u22_dense_provider_serial_cachelt_qiuchu_20260529T181331Z/224_224/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | dense | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/dense_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x288 | CVC-ClinicDB | 3->1 | provider | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_288/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | dense | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/dense_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | TOTAL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | enc4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | bottlenecka |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | bottleneckb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec4a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec4b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec3a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec3b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec2a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec2b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | up1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec1a |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | dec1b |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
| 384x384 | Satellite cloud | 4->1 | provider | pending | output |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .tmp/results/u23_dim32_clear_stream_cppdiag_layer_mae_20260530T054538Z/384_384/provider_e2e.json |  |
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

## DP layout-planner generalization stress case

This section records a planner-only stress case for reviewer discussion about
why the layout planner is formulated as a graph-level DP rather than a
U-Net-specific heuristic.  The example is deliberately not U-Net22 and does
not use U-Net layer names.  It is a pure convolutional multi-branch DAG with
Inception-style fanout: the same source feature is consumed by `1x1`, `3x3`,
`5x5`, `7x7`, and pooling branches, then later by another multi-kernel branch
after downsample/upsample exchange and an `Add`.  Such Inception-style
multi-branch convolutional blocks are a common pattern inside U-Net encoders
and U-Net variants for multi-scale feature extraction; the purpose here is to
show that the same DP planner consumes generic graph/operator metadata
(`Conv2d`, `AvgPool2d`, `ConvTranspose2d`, `Concat`, `Add`, shape, gap, and
required halo beta), not architecture-specific U-Net rules.

Rebuttal phrasing:

> The DP planner is not introduced because the U-Net topology is uniquely
> complex. It is introduced because halo layout selection is a graph-level
> optimization problem. Even in U-Net, skip connections create non-local
> dependencies between producer materialization, consumer-fused re-layout, and
> later concat alignment. We use U-Net as the primary medical workload, and
> additionally verify the same planner on nested/multi-branch convolutional DAGs
> without architecture-specific rules. The planner consumes only traced graph
> metadata and operator layout requirements, so it generalizes to any supported
> convolutional DAG.

Generator:

```bash
.venv/bin/python tools/generate_conv_dag_stress_layout.py
```

Artifacts:

```text
tools/generate_conv_dag_stress_layout.py
.tmp/results/convdag_stress_base8_256_silu7_dp_detail.csv
.tmp/results/convdag_stress_base8_256_silu7_dp_layout.md
.tmp/results/convdag_stress_base8_256_silu7_dp_summary.json
```

Planner-only summary:

| model | image | base | policy | edges | nodes | CT | rotations | CT-PT mult | re-layouts | relayout depth | consumer-fused | producer halo | required beta distribution |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvDAGStressNet | 256x256 | 8 | dp | 37 | 27 | 766 | 16,750 | 111,816 | 5 | 5 | 6 | 3 | beta0: 25; beta1: 5; beta2: 4; beta3: 3 |

DP detail table:

| # | node | op | sources | out | gap | CT | req beta | in beta | out beta | transition | mode | physical | R | M |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| 1 | stem | Conv2d | x | 8x256x256 | 1 | 16 | 1 | 1 | 0 | - | native_halo_stripe | native_source_stripe | 102 | 576 |
| 2 | stem_act | SiLU | stem | 8x256x256 | 1 | 16 | 0 | 0 | 0 | - | halo_local | native_source_stripe | 0 | 48 |
| 3 | b1 | Conv2d | stem_act | 8x256x256 | 1 | 16 | 0 | 0 | 0 | - | halo_local | native_source_stripe | 272 | 128 |
| 4 | b3 | Conv2d | stem_act | 8x256x256 | 1 | 16 | 1 | 1 | 0 | edge:1 | halo_local | packed_compact | 272 | 1,552 |
| 5 | b5 | Conv2d | stem_act | 8x256x256 | 1 | 16 | 2 | 2 | 0 | edge:1 | halo_local | packed_compact | 496 | 4,496 |
| 6 | b7 | Conv2d | stem_act | 8x256x256 | 1 | 16 | 3 | 3 | 0 | edge:1 | halo_local | packed_compact | 624 | 8,976 |
| 7 | bp | AvgPool2d | stem_act | 8x256x256 | 1 | 16 | 2 | 2 | 0 | edge:1 | halo_local | packed_compact | 104 | 208 |
| 8 | cat0 | Concat | b1;b3;b5;b7;bp | 40x256x256 | 1 | 80 | 0 | 0 | 0 | - | halo_local;native_halo_stripe | packed_compact | 0 | 0 |
| 9 | mix0 | Conv2d | cat0 | 16x256x256 | 1 | 32 | 0 | 0 | 0 | - | halo_local | packed_compact | 2,640 | 1,280 |
| 10 | mix0_act | SiLU | mix0 | 16x256x256 | 1 | 32 | 0 | 0 | 0 | - | halo_local | packed_compact | 0 | 96 |
| 11 | down | AvgPool2d | mix0_act | 16x128x128 | 2 | 8 | 0 | 0 | 3 | producer-halo | halo_local | logical_halo_compact | 64 | 128 |
| 12 | l3 | Conv2d | down | 16x128x128 | 2 | 8 | 1 | 3 | 2 | consumer-fused:1 | compact_halo_shared | logical_halo_compact | 928 | 6,144 |
| 13 | l5 | Conv2d | down | 16x128x128 | 2 | 8 | 2 | 3 | 2 | consumer-fused:1;producer-halo | compact_halo_shared | logical_halo_compact | 1,376 | 17,920 |
| 14 | l7 | Conv2d | down | 16x128x128 | 2 | 8 | 3 | 3 | 2 | consumer-fused:1;producer-halo | compact_halo_shared | logical_halo_compact | 1,632 | 35,840 |
| 15 | cat1 | Concat | l3;l5;l7 | 48x128x128 | 2 | 24 | 0 | 2 | 2 | - | halo_local | logical_halo_compact | 0 | 0 |
| 16 | mix1 | Conv2d | cat1 | 16x128x128 | 2 | 8 | 0 | 2 | 2 | - | halo_local | logical_halo_compact | 3,168 | 1,536 |
| 17 | mix1_act | SiLU | mix1 | 16x128x128 | 2 | 8 | 1 | 2 | 2 | - | halo_local | logical_halo_compact | 0 | 24 |
| 18 | up | ConvTranspose2d | mix1_act | 16x256x256 | 1 | 32 | 0 | 2 | 4 | - | halo_local | logical_halo_compact | 608 | 2,048 |
| 19 | add | Add | mix0_act;up | 16x256x256 | 1 | 32 | 0 | 4 | 4 | edge:1 | halo_local | logical_halo_compact | 0 | 32 |
| 20 | c1 | Conv2d | add | 8x256x256 | 1 | 16 | 0 | 4 | 2 | - | halo_local | logical_halo_compact | 544 | 256 |
| 21 | c3 | Conv2d | add | 8x256x256 | 1 | 16 | 1 | 4 | 0 | consumer-fused:1 | compact_halo_shared | packed_compact | 544 | 3,072 |
| 22 | c5 | Conv2d | add | 8x256x256 | 1 | 16 | 2 | 4 | 0 | consumer-fused:1 | compact_halo_shared | packed_compact | 992 | 8,960 |
| 23 | c7 | Conv2d | add | 8x256x256 | 1 | 16 | 3 | 4 | 0 | consumer-fused:1 | compact_halo_shared | packed_compact | 1,248 | 17,920 |
| 24 | cat2 | Concat | c1;c3;c5;c7 | 32x256x256 | 1 | 64 | 0 | 0 | 0 | - | halo_local | packed_compact | 0 | 0 |
| 25 | mix2 | Conv2d | cat2 | 8x256x256 | 1 | 16 | 0 | 0 | 0 | - | halo_local | packed_compact | 1,088 | 512 |
| 26 | mix2_act | SiLU | mix2 | 8x256x256 | 1 | 16 | 0 | 0 | 0 | - | halo_local | packed_compact | 0 | 48 |
| 27 | out | Conv2d | mix2_act | 1x256x256 | 1 | 2 | 0 | 0 | 0 | - | halo_local | packed_compact | 48 | 16 |

## Legacy 224x224 Re-Layout Ablation Provenance

The earlier 224x224 re-layout scheduling comparison with rotations
`7,507,030`, `22,050,490`, and `135,484` came from the compile-lite policy
probe, not from the older `generate_unet22_layout_ablation_table.py` paper
table generator.

Script:

```text
tools/run_u22_224_policy_compile_probe.py
```

Equivalent command:

```bash
mkdir -p .tmp/results .tmp/logs
env ORION_COMPILE_SKIP_BOOTSTRAPPER_GENERATION=1 \
  timeout 900s .venv/bin/python tools/run_u22_224_policy_compile_probe.py \
  --backend lattigo \
  --image-size 224 \
  --base-channels 32 \
  --silu-degree 7 \
  --logn 16 \
  --max-rounds 16 \
  --auto-target 1 \
  --skip-layer-compile \
  --policies dp_no_share_fold fixed_max_no_share always_no_share \
  --out .tmp/results/u22_224_policy_compile_lite_current_after_fixedmax_clamp.json \
  --csv .tmp/results/u22_224_policy_compile_lite_current_after_fixedmax_clamp.csv \
  | tee .tmp/logs/u22_224_policy_compile_lite_current_after_fixedmax_clamp.log
```

Historical note: this artifact predates the default eager baseline switch.  At
that time, `always_no_share` normalized to the consumer-fused eager policy.  In
the current code, bare `always_no_share` normalizes to the producer-fused eager
policy; the legacy consumer-fused row is available only by explicitly selecting
`always_no_share_fused`.

Artifacts:

```text
.tmp/results/u22_224_policy_compile_lite_current_after_fixedmax_clamp.csv
.tmp/results/u22_224_policy_compile_lite_current_after_fixedmax_clamp.json
.tmp/logs/u22_224_policy_compile_lite_current_after_fixedmax_clamp.log
```

Recorded compile-lite rows:

| policy | status | rotations | re-layouts | relayout depth | relayout mask mult | consumer-fused relayouts | native physical relayout edges | provider mode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fixed_max_no_share` | ok | 7,507,030 | 12 | 12 | 328 | 0 | 5 | `u22_256_base32_layout_fixedmax_no_share` |
| `always_no_share` (legacy consumer-fused) | ok | 22,050,490 | 30 | 30 | 666 | 22 | 0 | `u22_256_base32_layout_always_no_share` |
| `dp_no_share_fold` | ok | 135,484 | 4 | 4 | 0 | 0 | 0 | `u22_256_base32_layout_dp_no_share_fold` |

Notes:

- The probe model is `UNet22PlusOutput` with a 22-layer body plus explicit
  `1x1` output, `224x224`, base channels `32`, SiLU degree `7`, LogN `16`.
- The run used layout-policy provider native halo and re-layout kernels, with
  `ORION_UNIFIED_LT_INDIVIDUAL_EVAL=1`,
  `ORION_UNIFIED_LT_SHARED_ROTATION_KEYS=0`, and
  `ORION_LATTIGO_UNIFIED_NO_BSGS=0`.
- The legacy consumer-fused eager row should not be used as the default
  `\eagerrelayout` baseline.  Current default probes use producer-fused eager
  for bare `always_no_share`; the old row is only a debug/provenance reference.
