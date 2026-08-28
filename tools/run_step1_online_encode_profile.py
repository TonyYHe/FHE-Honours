from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
E2E_RUNNER = REPO_ROOT / "tools" / "run_lattigo_e2e_compare.py"


def _profile_environment(base: dict[str, str], *, encode_workers: int) -> dict[str, str]:
    env = dict(base)
    env.update(
        {
            "ORION_LATTIGO_CLEAR_BACKEND": "0",
            "ORION_SINGLE_SLOT_LAYER_CACHE": "1",
            "ORION_LATTIGO_STREAMING_LT": "0",
            "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT": "0",
            "ORION_SINGLE_SLOT_ENCODE_WORKERS": str(max(1, int(encode_workers))),
        }
    )
    return env


def _default_out(network: str, mode: str) -> Path:
    return (
        REPO_ROOT
        / ".tmp"
        / "results"
        / "honours"
        / "03_step1_online_encode"
        / f"{str(network)}_{str(mode)}.json"
    )


def _runner_command(args: argparse.Namespace, out_path: Path) -> list[str]:
    command = [
        str(sys.executable),
        str(E2E_RUNNER),
        "--mode",
        str(args.mode),
        "--backend",
        "lattigo",
        "--network",
        str(args.network),
        "--out",
        str(out_path),
        "--seed",
        str(int(args.seed)),
        "--forward-runs",
        str(max(1, int(args.forward_runs))),
        "--warmup-runs",
        str(max(0, int(args.warmup_runs))),
        "--io-mode",
        "none",
        "--profile-modules",
        "--profile-lt",
        "--operator-breakdown",
    ]
    if args.provider_mode:
        command.extend(("--provider-mode", str(args.provider_mode)))
    if args.activation:
        command.extend(("--activation", str(args.activation)))
    if args.ckks_preset:
        command.extend(("--ckks-preset", str(args.ckks_preset)))
    if args.logn_override is not None:
        command.extend(("--logn-override", str(int(args.logn_override))))
    if bool(args.trace_forward_memory):
        command.append("--trace-forward-memory")
    return command


def _result_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if str(payload.get("status", "")) != "ok":
        errors.append(f"runner status is {payload.get('status')!r}, expected 'ok'")
    profile = dict(payload.get("step1_online_encode_profile", {}) or {})
    if not profile:
        errors.append("step1_online_encode_profile is missing")
        return errors
    if not bool(profile.get("valid", False)):
        validation = list(profile.get("validation_errors", []) or [])
        errors.extend(str(value) for value in validation)
        if not validation:
            errors.append("Step 1 profile is marked invalid")
    if int(profile.get("schema_version", 0) or 0) < 2:
        errors.append("Step 1 profile uses the obsolete pre-accounting schema; expected version 2+")
    accounting = dict(profile.get("major_wall_categories_accounting", {}) or {})
    if not accounting or not bool(accounting.get("valid", False)):
        errors.append("major wall categories failed additive wall-time accounting validation")
    if int(profile.get("profile_count", 0) or 0) != int(
        profile.get("measured_attempt_count", 0) or 0
    ):
        errors.append("not every measured attempt produced a Step 1 profile")
    if int(profile.get("measured_attempt_count", 0) or 0) != int(
        payload.get("forward_runs", 0) or 0
    ):
        errors.append("not every requested measured forward succeeded")
    correctness = dict(payload.get("mae_vs_clear", {}) or {})
    if not bool(correctness.get("shape_match", False)):
        errors.append("final decrypted output does not match the clear output shape")
    return errors


def _print_result_summary(payload: dict[str, Any], out_path: Path) -> None:
    profile = dict(payload.get("step1_online_encode_profile", {}) or {})
    categories = dict(profile.get("major_wall_categories", {}) or {})
    micro = dict(profile.get("operator_microprofile", {}) or {})
    report = {
        "result": str(out_path),
        "valid": bool(profile.get("valid", False)),
        "runtime_fairness_mode": profile.get("runtime_fairness_mode"),
        "measured_attempt_count": profile.get("measured_attempt_count"),
        "requested_measured_attempt_count": profile.get("requested_measured_attempt_count"),
        "mean_he_forward_s": profile.get("he_forward_s"),
        "mean_online_encode_s": profile.get("online_encode_s"),
        "online_encode_pct_of_he_forward": profile.get("online_encode_pct_of_he_forward"),
        "major_wall_categories": categories,
        "major_wall_categories_accounting": profile.get("major_wall_categories_accounting"),
        "operator_microprofile": micro,
        "validation_errors": profile.get("validation_errors", []),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-FHE, single-slot-layer-cache profile required by "
            "Step 1 of FHE_compression.md."
        )
    )
    parser.add_argument("--network", default="resnet20_cifar10")
    parser.add_argument("--mode", choices=("dense", "provider"), default="dense")
    parser.add_argument("--provider-mode", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--encode-workers",
        type=int,
        default=1,
        help="Online diagonal-Encode workers. Record and keep this fixed when comparing runs.",
    )
    parser.add_argument("--activation", choices=("relu", "silu"), default=None)
    parser.add_argument("--ckks-preset", choices=("network-default", "resnet"), default=None)
    parser.add_argument("--logn-override", type=int, default=None)
    parser.add_argument("--trace-forward-memory", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact environment and command without running real FHE.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if int(args.forward_runs) < 1:
        raise SystemExit("--forward-runs must be at least 1")
    if int(args.warmup_runs) < 0:
        raise SystemExit("--warmup-runs cannot be negative")
    if int(args.encode_workers) < 1:
        raise SystemExit("--encode-workers must be at least 1")

    out_path = Path(args.out) if args.out is not None else _default_out(args.network, args.mode)
    out_path = out_path.expanduser().resolve()
    env = _profile_environment(dict(os.environ), encode_workers=int(args.encode_workers))
    command = _runner_command(args, out_path)
    selected_env = {
        name: env[name]
        for name in (
            "ORION_LATTIGO_CLEAR_BACKEND",
            "ORION_SINGLE_SLOT_LAYER_CACHE",
            "ORION_LATTIGO_STREAMING_LT",
            "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT",
            "ORION_SINGLE_SLOT_ENCODE_WORKERS",
        )
    }
    print(json.dumps({"environment": selected_env, "command": shlex.join(command)}, indent=2))
    if bool(args.dry_run):
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if int(completed.returncode) != 0:
        print(f"Step 1 runner failed with exit code {completed.returncode}", file=sys.stderr)
        return int(completed.returncode)
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Cannot read Step 1 result {out_path}: {exc}", file=sys.stderr)
        return 2
    _print_result_summary(payload, out_path)
    errors = _result_errors(payload)
    if errors:
        print("Step 1 profile validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
