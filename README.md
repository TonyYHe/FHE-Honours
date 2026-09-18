# Orion

> **FHE compression and online encoding overhead for privacy-preserving neural-network inference**

[![Build Status](https://img.shields.io/badge/build-research%20prototype-lightgrey)](#)
[![Python](https://img.shields.io/badge/python-3.9--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Go](https://img.shields.io/badge/go-1.24-00ADD8?logo=go&logoColor=white)](https://go.dev/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository contains a computer science Honours thesis project built upon the existing **Orion** encrypted-inference framework. The research investigates the memory and runtime bottlenecks of weight-plaintext encoding in fully homomorphic encryption (FHE). 

To support this investigation, this project extends Orion's core architecture with validated profiling pipelines, memory-bounded execution models, and the foundational infrastructure necessary to evaluate compressed CKKS plaintext representations.

## 2. Project Overview

Modern FHE inference systems can evaluate neural networks without decrypting their inputs, but their plaintext model parameters must first be converted into representation-specific polynomial objects. For convolutional and linear layers, this includes constructing packed matrix diagonals, CKKS slot encoding, residue number system (RNS) decomposition, and conversion to the number theoretic transform (NTT) representation used by the evaluator. Retaining every encoded plaintext reduces latency but creates a large memory footprint; regenerating them layer by layer saves memory but introduces substantial online Encode overhead.

This project studies that latency-memory trade-off in Orion. Its principal objectives are to:

- quantify online weight encoding as a proportion of real-FHE inference time;
- separate additive wall-time categories from overlapping operator diagnostics;
- maintain correctness under memory-bounded, single-slot layer caching;
- identify layout, packing, tiling, batching, and caching opportunities that reduce encoding work;
- evaluate whether periodic Weight Plaintext Compression (WPC) can be adapted to Orion's diagonal layouts; and
- preserve reproducible evidence across clear-backend correctness tests and real-Lattigo performance experiments.

### Technical stack

| Component | Role |
|---|---|
| Python 3.9–3.12 | Model definitions, graph lowering, packing, experiment orchestration, validation, and analysis |
| PyTorch | Clear reference models and tensor-level correctness checks |
| Go 1.24 | Native backend build and high-performance cryptographic execution |
| Lattigo v6.1.1 | RNS-CKKS encoding, linear transforms, key switching, rotations, and bootstrapping |
| NumPy / SciPy | Numerical processing and reference calculations |
| Matplotlib | Reproducible benchmark figures |

The Python frontend lowers neural-network operators into encrypted linear transforms and activation circuits. The Go/Lattigo backend performs the corresponding RNS-CKKS operations. A clear-Lattigo mode executes the same structural path without cryptographic arithmetic for local correctness validation; it does not model CKKS security, noise growth, or real-FHE runtime. Publishable performance results use the real-FHE backend.

### Experimental boundary

The primary Step 1 denominator, `he_forward_s`, is one encrypted model forward pass. It includes linear-transform/MVM kernels, online plaintext encoding, bootstrapping, layer-cache management, activation arithmetic, and runtime orchestration. It excludes compilation, cryptographic setup and key generation, clear inference, input encryption, and output decryption/decoding.

The principal metric is

```text
online_encode_pct_of_he_forward = 100 × online_encode_s / he_forward_s
```

The current experiments use `--io-mode none`; consequently, the reported shares measure local execution rather than network communication.

### Reproducibility entry points

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python tools/build_lattigo.py
```

The native backend requires a working Go toolchain. Real-FHE model runs are resource intensive and should be executed on a suitably provisioned Linux server. See [`docs/step1_online_encode_profile.md`](docs/step1_online_encode_profile.md) for exact commands, environment constraints, and acceptance criteria.

## 3. Current Progress

- **Step 0 — clear-Lattigo validation: complete.**
  - Established clear-Lattigo as the structural correctness backend before running expensive real-FHE experiments.
  - Verified ResNet20/CIFAR-10 and the U-Net22 provider path with matching output shapes and no threshold failures.
  - Confirmed that clear execution does not require linear-transform streaming; the legacy chunk-streaming path remains disabled.

- **Memory-bounded execution path: complete for the profiled models.**
  - Enabled single-slot layer caching so full encoded diagonal payloads do not remain resident for the entire network.
  - Added index-only metadata and runtime diagonal reconstruction for hybrid-packed `Linear` layers.
  - Preserved native physical-layout signatures through layout-transparent `Mult` and `Identity` operations.
  - Repaired the VGG16 provider path so its 32-ciphertext native layout is retained through the first activation and bootstrap boundary.

- **Step 1 — real-FHE online Encode profiling: complete.**
  - Collected one warmup and three measured forwards for ResNet20, U-Net22, and VGG16/base16 using one Encode worker.
  - Introduced the validated schema-v2 `step1_online_encode_profile` report.
  - Reset bootstrap counters per forward and sourced bootstrap wall time from the independently reset backend profile.
  - Separated additive wall-clock categories from nested or parallel operator microtimers.
  - Added accounting checks that reject missing, non-finite, negative, or non-closing category reports.
  - Added a schema-aware extractor that produces compact JSON, CSV, Markdown, and Matplotlib artifacts.

- **Step 2 — WPC analysis: planned.**
  - Analyse how CIPS and Rotation Padding expose periodic encoded weight plaintexts.
  - Derive WPC's per-plaintext and model-level compression factors and documented its implications for Orion.
  - Distinguish persistent encoded-plaintext storage savings from transient reconstruction buffers.
  - Document the implications of the [WPC paper](https://dl.acm.org/doi/10.1145/3719027.3765022) for Orion's memory and execution model.

- **Step 3 — WPC reproduction and Orion adaptation: planned.**
  - Reproduce WPC-style compression for VGG and U-Net workloads.
  - Measure encoded-plaintext storage, reconstruction latency, peak memory, and end-to-end HE inference.
  - Investigate a periodic or otherwise compressed representation tailored to Orion's diagonal layout.

## 4. Results & Benchmarks

### Step 0: clear-Lattigo correctness

| Model | Input / mode | Status | Maximum layer MAE | Maximum absolute error | Structural observation |
|:---|:---|:---:|---:|---:|:---|
| ResNet20 | CIFAR-10, `32×32`, dense | Pass | `2.20e-6` | `1.10e-5` | Zero threshold failures |
| U-Net22 | `64×64`, provider | Pass | `7.85e-8` | `1.19e-6` | `enc1a` and `enc1b` each use four compact ciphertexts |

Clear-Lattigo results validate graph structure and numerical behavior but are not used as FHE performance evidence.

### Step 1: validated real-FHE profiles

| Model | Mode | `he_forward_s` | `online_encode_s` | `online_encode_pct_of_he_forward` | Bootstrap share | MVM share | Output MAE |
|:---|:---:|---:|---:|---:|---:|---:|---:|
| ResNet20 / CIFAR-10 | Dense | `1530.126 ± 13.282` | `305.403 ± 3.459` | **19.959%** | **64.644%** | 4.095% | `1.026e-4` |
| U-Net22 / 64 / base32 | Provider | `5575.576 ± 34.695` | `5044.205 ± 22.326` | **90.470%** | 1.995% | 7.324% | `2.276e-8` |
| VGG16 / base16 / ImageNet-224 | Provider | `16508.112 ± 197.384` | `7937.350 ± 129.801` | **48.082%** | **44.136%** | 4.580% | `1.418e-4` |

Values are means ± sample standard deviations over three measured forwards; one warmup is excluded. Every accepted run used real Lattigo FHE, single-slot layer caching, one Encode worker, and disabled streaming modes. All decrypted output shapes matched their clear references, and each schema-v2 additive category partition closed to the measured HE-forward wall time.

### Bottleneck classification

| Model | Dominant cost | Evidence | Immediate research implication |
|:---|:---|:---|:---|
| ResNet20 | Bootstrapping | 64.644% of HE-forward time | Prioritise bootstrap count, scheduling, level management, and ciphertext count at refresh boundaries |
| U-Net22 | Online Encode | 90.470% of HE-forward time | Prioritise encoded-weight reuse, compression, parallel encoding, and layout-aware reduction of diagonal plaintexts |
| VGG16/base16 | Mixed Encode and Bootstrap | 48.082% Encode; 44.136% Bootstrap | Evaluate joint layout/cache changes and bootstrap-aware packing |

The low 4.1–7.3% MVM shares limit the benefit of MVM-kernel-only optimisation under the measured configurations. The model matrix should not be interpreted as a controlled Dense-versus-Provider comparison because architecture and execution mode vary together; such a comparison requires paired runs of the same model and parameters.

The additive `major_wall_categories` values are suitable for wall-time partitions. The `operator_microprofile` timers are diagnostic and non-additive because nested or parallel measurements may overlap; they must not be summed into a runtime total. Full-precision results and provenance are available in [the extracted Step 1 report](.tmp/results/honours/06_step1_extracted/step1_profile_report.md).

## 5. Visuals

### Online Encode latency and HE-forward proportion

![Online Encode time and percentage of HE-forward time for ResNet20, U-Net22, and VGG16](.tmp/results/honours/06_step1_extracted/online_encode_comparison.png)

This figure compares absolute HE-forward and online Encode latency while also exposing the different Encode proportions across the three validated real-FHE models.

### Additive HE-forward wall-time composition

![Additive HE-forward wall-time categories for the validated real-FHE model matrix](.tmp/results/honours/06_step1_extracted/major_wall_categories_pct.png)

The stacked categories form a validated partition of each model's HE-forward wall time. They show the transition from bootstrap-dominated ResNet20 to Encode-dominated U-Net22 and the mixed VGG16 profile.

### Non-additive operator diagnostics

![Diagnostic operator microprofile percentages for linear transforms, rotations, elementwise operations, and bootstrapping](.tmp/results/honours/06_step1_extracted/operator_microprofile_pct.png)

These values provide operator-level diagnostic evidence but may overlap. They are intentionally presented separately from the additive wall-time accounting.

### Linear-transform operation counts

![Log-scale Lattigo linear-transform operation counts across the validated real-FHE models](.tmp/results/honours/06_step1_extracted/operation_counts_log.png)

The log-scale plot compares transform, diagonal, modular multiplication, rotation, reduction, and final modulus-down counts. VGG16 has the largest count for every reported operation, while U-Net22's high diagonal and QP-multiplication counts are consistent with its large Encode workload.

Detailed methodology and interpretation guidance are maintained in [`docs/step1_online_encode_profile.md`](docs/step1_online_encode_profile.md). This research code is distributed under the [MIT License](LICENSE).
