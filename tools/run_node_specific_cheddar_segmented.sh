#!/usr/bin/env bash
set -euo pipefail

ROOT="${ORION_NODE_BENCH_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

PYTHON_BIN="${ORION_NODE_BENCH_PYTHON:-python3}"
TAG="${ORION_NODE_BENCH_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RESULT_DIR="${ORION_NODE_BENCH_RESULT_DIR:-$ROOT/.tmp/results}"
IO_PARENT="${ORION_CHEDDAR_IO_PARENT:-${ORION_CHEDDAR_IO_ROOT:-$ROOT/.tmp/cheddar_node_io_$TAG}}"
NETWORKS="${ORION_NODE_BENCH_NETWORKS:-r18_tiny r34_imgnet u22_64_base32 u22_256_base32}"
PATHS="${ORION_NODE_BENCH_PATHS:-dense provider provider_no_hybrid provider_no_family}"
REPEATS="${ORION_NODE_BENCH_REPEATS:-3}"
WARMUPS="${ORION_NODE_BENCH_WARMUPS:-0}"
TIMEOUT_S="${ORION_NODE_BENCH_TIMEOUT_S:-7200}"
CKKS_PROFILE="${ORION_NODE_BENCH_CKKS_PROFILE:-e2e}"
MIN_DISK_FREE_GB="${ORION_NODE_BENCH_MIN_DISK_FREE_GB:-25}"
DISK_WATCHDOG_INTERVAL_S="${ORION_NODE_BENCH_DISK_WATCHDOG_INTERVAL_S:-5}"

mkdir -p "$RESULT_DIR" "$IO_PARENT"

export ORION_NODE_BENCH_CLEAN_CHEDDAR_IO="${ORION_NODE_BENCH_CLEAN_CHEDDAR_IO:-1}"
export ORION_FORWARD_HOST_MEMORY_GUARD="${ORION_FORWARD_HOST_MEMORY_GUARD:-1}"

MANIFEST="$RESULT_DIR/node_specific_cheddar_segmented_${TAG}.manifest.txt"
{
  echo "tag=$TAG"
  echo "root=$ROOT"
  echo "python=$PYTHON_BIN"
  echo "result_dir=$RESULT_DIR"
  echo "io_parent=$IO_PARENT"
  echo "networks=$NETWORKS"
  echo "paths=$PATHS"
  echo "repeats=$REPEATS"
  echo "warmups=$WARMUPS"
  echo "timeout_s=$TIMEOUT_S"
  echo "ckks_profile=$CKKS_PROFILE"
  echo "min_disk_free_gb=$MIN_DISK_FREE_GB"
  echo "disk_watchdog_interval_s=$DISK_WATCHDOG_INTERVAL_S"
  echo "clean_cheddar_io=$ORION_NODE_BENCH_CLEAN_CHEDDAR_IO"
  echo
} > "$MANIFEST"

for network in $NETWORKS; do
  network_io_root="$IO_PARENT/$network"
  mkdir -p "$network_io_root"
  export ORION_CHEDDAR_IO_ROOT="$network_io_root"

  out_json="$RESULT_DIR/node_specific_cheddar_${TAG}_${network}.json"
  out_csv="$RESULT_DIR/node_specific_cheddar_${TAG}_${network}.csv"
  out_log="$RESULT_DIR/node_specific_cheddar_${TAG}_${network}.log"

  {
    echo "[$(date -Is)] start network=$network"
    df -h "$network_io_root"
    "$PYTHON_BIN" tools/benchmark_node_specific_lattigo_provider_vs_dense.py \
      --backends cheddar \
      --networks "$network" \
      --paths $PATHS \
      --repeats "$REPEATS" \
      --warmups "$WARMUPS" \
      --timeout-s "$TIMEOUT_S" \
      --ckks-profile "$CKKS_PROFILE" \
      --min-disk-free-gb "$MIN_DISK_FREE_GB" \
      --disk-watchdog-interval-s "$DISK_WATCHDOG_INTERVAL_S" \
      --out "$out_json" \
      --csv-out "$out_csv"
    df -h "$network_io_root"
    echo "[$(date -Is)] finish network=$network json=$out_json csv=$out_csv"
  } 2>&1 | tee "$out_log"

  if [[ "$ORION_NODE_BENCH_CLEAN_CHEDDAR_IO" =~ ^(1|true|yes|on)$ ]]; then
    rm -rf "$network_io_root"
  fi

  {
    echo "network=$network"
    echo "json=$out_json"
    echo "csv=$out_csv"
    echo "log=$out_log"
    echo
  } >> "$MANIFEST"
done

echo "manifest=$MANIFEST"
