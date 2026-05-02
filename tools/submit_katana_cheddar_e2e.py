from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


MODULE_COMMANDS = """module purge || true
module load gcc/12.2.0 cmake/3.31.8 cuda/13.0.0 gmp/6.2.1 python/3.10.8 pytorch/1.13.1 go/1.25.1
"""


def _pbs_resource_line(*, gpu_model: str, ncpus: int, mem: str) -> str:
    gpu_model = str(gpu_model)
    if gpu_model.upper() == "GH200":
        return "#PBS -l select=1:cpu_arch=aarch64:ngpus=1:ncpus=72:mem=480gb"
    if gpu_model.lower() in {"", "none"}:
        return f"#PBS -l select=1:ncpus={int(ncpus)}:mem={mem}:ngpus=1"
    return f"#PBS -l select=1:ncpus={int(ncpus)}:mem={mem}:ngpus=1:gpu_model={gpu_model}"


def _render_pbs(args: argparse.Namespace) -> str:
    repo = Path(args.repo)
    run_root = Path(args.run_root)
    cache_root = Path(args.cache_root)
    job_name = str(args.job_name)
    network = str(args.network)
    mode = str(args.mode)
    phase = str(args.phase)
    provider_mode_line = (
        f"  --provider-mode {shlex.quote(str(args.provider_mode))} \\\n"
        if args.provider_mode
        else ""
    )
    run_dir = run_root / job_name
    cache_dir = cache_root / network / mode
    pbs_stdout = run_root / f"{job_name}.pbs.out"
    out_compile = run_dir / f"{mode}_compile_save.json"
    out_forward = run_dir / f"{mode}_load_forward.json"
    dmon = run_dir / "dmon.csv"
    log = run_dir / "run.log"

    if phase == "cpu_compile":
        body = f"""echo "Cheddar compile-save is not CPU-only today."
echo "The heavy artifacts in io_mode=save are Cheddar encoded plaintexts/keys and require a CUDA backend context."
echo "Use phase=compile_forward or phase=compile_only on a GPU node."
exit 2
"""
    else:
        compile_block = ""
        if phase in {"compile_forward", "compile_only"}:
            compile_payload_env = ""
            if phase == "compile_only":
                compile_payload_env = """  ORION_CHEDDAR_SAVE_PLAINTEXT_PAYLOADS=0 \\
  ORION_UNIFIED_LT_SAVE_ENCODED_PLAINTEXTS=0 \\
"""
            compile_block = f"""env \\
  ORION_COMPILE_SKIP_BOOTSTRAPPER_GENERATION=1 \\
  ORION_UNIFIED_LT_PREPARE_SHARED_CACHE_PLAN=0 \\
  ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE=1 \\
  ORION_UNIFIED_LT_RELEASE_INDEX_ONLY_RAW_MATRICES_AFTER_SAVE=1 \\
{compile_payload_env}\
  /usr/bin/time -v "${{UV_RUN[@]}}" python tools/run_lattigo_e2e_compare.py \\
  --backend cheddar \\
  --network {network} \\
  --mode {mode} \\
  --io-mode save \\
  --io-dir "{cache_dir}" \\
  --compile-only \\
{provider_mode_line}\
  --out "{out_compile}"
"""

        forward_block = ""
        if phase in {"compile_forward", "forward_only"}:
            forward_block = f"""env \\
  ORION_UNIFIED_LT_CLEAR_SOURCE_DIAGONALS_AFTER_COMPILE=1 \\
  /usr/bin/time -v "${{UV_RUN[@]}}" python tools/run_lattigo_e2e_compare.py \\
  --backend cheddar \\
  --network {network} \\
  --mode {mode} \\
  --io-mode load \\
  --io-dir "{cache_dir}" \\
  --warmup-runs {int(args.warmup_runs)} \\
  --forward-runs {int(args.forward_runs)} \\
  --profile-modules \\
  --trace-forward-memory \\
{provider_mode_line}\
  --out "{out_forward}"
"""

        body = f""""${{UV_RUN[@]}}" python tools/build_cheddar_backend.py
nvidia-smi
DMON="{dmon}"
LOG="{log}"
rm -f "$DMON" "$LOG"
DMON_ARGS=()
if [ -n "${{CUDA_VISIBLE_DEVICES:-}}" ]; then
  DMON_ARGS=(-i "${{CUDA_VISIBLE_DEVICES%%,*}}")
fi
nvidia-smi dmon "${{DMON_ARGS[@]}}" -s pucmt -d 1 -o DT > "$DMON" &
DMON_PID=$!
set +e
(
  set -euo pipefail
  {compile_block}
  {forward_block}
) > "$LOG" 2>&1
RC=$?
set -e
kill "$DMON_PID" 2>/dev/null || true
wait "$DMON_PID" 2>/dev/null || true
echo "rc=$RC"
echo "run_dir={run_dir}"
echo "cache_dir={cache_dir}"
tail -120 "$LOG" || true
exit "$RC"
"""

    return f"""#!/bin/bash
#PBS -N {job_name}
{_pbs_resource_line(gpu_model=str(args.gpu_model), ncpus=int(args.ncpus), mem=str(args.mem))}
#PBS -l walltime={args.walltime}
#PBS -j oe
#PBS -o {pbs_stdout}

set -euo pipefail
cd "{repo}"
{MODULE_COMMANDS}
export PYTHONUSERBASE="{args.python_userbase}"
export UV_VENV="{args.venv}"
export UV_CACHE_DIR="{args.uv_cache_dir}"
export PATH="$HOME/.local/bin:$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.10/site-packages:$PWD:${{PYTHONPATH:-}}"
export CHEDDAR_ROOT="{args.cheddar_root}"
export LD_LIBRARY_PATH="{args.cheddar_root}/build:{args.cheddar_root}/build/_deps/rmm-build:{args.cheddar_root}/build/_deps/rapids_logger-build:${{LD_LIBRARY_PATH:-}}"
export TMPDIR="{args.tmpdir}"
export ORION_CHEDDAR_IO_ROOT="{cache_root}"
export ORION_CHEDDAR_LT_STREAMING=auto
export ORION_UNIFIED_LT_ROTKEY_RESIDENCY=1
export ORION_UNIFIED_LT_PLAINTEXT_RESIDENCY=1
export ORION_CHEDDAR_SHARED_CACHE_PLAN_PERSIST=1
export ORION_CHEDDAR_GPU_PREFETCH=0
export ORION_LATTIGO_BOOTSTRAP_MANY=0
export ORION_CHEDDAR_SAVE_PLAINTEXT_PAYLOAD_MAX_DEVICE_BYTES="{args.payload_max_device_bytes}"
export ORION_UNIFIED_LT_ENCODED_PLAINTEXT_MAX_DEVICE_BYTES="{args.payload_max_device_bytes}"
mkdir -p "{run_dir}" "{cache_dir}" "$TMPDIR" "$UV_CACHE_DIR"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -x "$UV_VENV/bin/python" ]; then
  uv venv --python "$(command -v python3)" --system-site-packages "$UV_VENV"
fi
source "$UV_VENV/bin/activate"
UV_RUN=(uv run --active --no-sync)
echo "uv=$(command -v uv)"
uv --version
echo "python=$("${{UV_RUN[@]}}" python -c 'import sys; print(sys.executable)')"
echo "job_id=${{PBS_JOBID:-manual}}"
echo "host=$(hostname)"
echo "repo={repo}"
echo "run_dir={run_dir}"
echo "cache_dir={cache_dir}"
date --iso-8601=seconds
{body}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and optionally submit a Katana OpenPBS Cheddar E2E job.")
    parser.add_argument("--repo", default="/srv/scratch/z5601763/orion")
    parser.add_argument("--run-root", default="/srv/scratch/z5601763/orion_runs/cheddar_e2e")
    parser.add_argument("--cache-root", default="/srv/scratch/z5601763/haloed-cache/cheddar_e2e")
    parser.add_argument("--tmpdir", default="/srv/scratch/z5601763/tmp")
    parser.add_argument("--cheddar-root", default="/srv/scratch/z5601763/cheddar-fhe-src")
    parser.add_argument("--python-userbase", default="/srv/scratch/z5601763/python-userbase")
    parser.add_argument("--venv", default="/srv/scratch/z5601763/orion/.venv-katana")
    parser.add_argument("--uv-cache-dir", default="/srv/scratch/z5601763/uv-cache")
    parser.add_argument("--network", choices=("r18_tiny", "r34_imgnet", "u22_64_base32", "u22_256_base32"), default="r18_tiny")
    parser.add_argument("--mode", choices=("dense", "provider"), default="provider")
    parser.add_argument("--phase", choices=("compile_forward", "compile_only", "forward_only", "cpu_compile"), default="compile_forward")
    parser.add_argument("--provider-mode", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--gpu-model", default="H200")
    parser.add_argument("--ncpus", type=int, default=8)
    parser.add_argument("--mem", default="96gb")
    parser.add_argument("--walltime", default="02:00:00")
    parser.add_argument("--payload-max-device-bytes", default=str(4 * 1024**3))
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--forward-runs", type=int, default=3)
    parser.add_argument("--script-out", type=Path, default=None)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.job_name is None:
        args.job_name = f"chd_{args.network}_{args.mode}_{args.phase}"
    script_out = Path(args.script_out or (Path(args.run_root) / f"{args.job_name}.pbs"))
    script_out.parent.mkdir(parents=True, exist_ok=True)
    script_out.write_text(_render_pbs(args), encoding="utf-8")
    print(f"pbs_script={script_out}")
    print(f"output_log={Path(args.run_root) / (str(args.job_name) + '.pbs.out')}")
    if args.submit:
        result = subprocess.run(["qsub", str(script_out)], check=True, text=True, stdout=subprocess.PIPE)
        print(f"job_id={result.stdout.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
