#!/bin/bash
set -euo pipefail

PROFILE_OUT_DIR=.tmp/results/honours/04_step1_model_matrix
mkdir -p "$PROFILE_OUT_DIR"

for spec in \
  resnet20_cifar10:dense \
  u22_64_base32:provider \
  vgg16_imgnet:provider
do
  network="${spec%%:*}"
  mode="${spec##*:}"

  echo "Profiling ${network} (${mode})"

  python tools/run_step1_online_encode_profile.py \
    --network "$network" \
    --mode "$mode" \
    --encode-workers 1 \
    --warmup-runs 1 \
    --forward-runs 3 \
    --out "${PROFILE_OUT_DIR}/${network}_${mode}.json" \
    2>&1 | tee "${PROFILE_OUT_DIR}/${network}_${mode}.log"
done
