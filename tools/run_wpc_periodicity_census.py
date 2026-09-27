#!/usr/bin/env python3
"""Run an untimed clear-Lattigo audit of Orion diagonal periodicity.

This launcher deliberately performs one clear structural forward.  It records
the exact slot messages passed to Lattigo but does not apply compression and
must not be used as an end-to-end timing result.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / ".tmp/results/honours/09_wpc_periodicity_census"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _slug(network: str, mode: str) -> str:
    return f"{str(network).replace('/', '_')}_{str(mode)}"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON output {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one clear-Lattigo forward and census the exact power-of-two "
            "periods of Orion linear-transform diagonals."
        )
    )
    parser.add_argument("--network", required=True)
    parser.add_argument("--mode", choices=("dense", "provider"), required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encode-workers", type=int, default=1)
    parser.add_argument("--checkpoint-hash", default=None)
    parser.add_argument(
        "--verify-encoded-qp",
        action="store_true",
        help=(
            "Independently encode each slot-periodic candidate with the real "
            "Lattigo library and require exact Q/P copy-map reconstruction."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if int(args.encode_workers) < 1:
        raise SystemExit("--encode-workers must be at least 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(args.network, args.mode)
    result_path = out_dir / f"{stem}.clear.json"
    jsonl_path = out_dir / f"{stem}.periodicity.jsonl"
    summary_path = out_dir / f"{stem}.periodicity.summary.json"

    env = dict(os.environ)
    env.update(
        {
            "ORION_LATTIGO_CLEAR_BACKEND": "1",
            "ORION_SINGLE_SLOT_LAYER_CACHE": "1",
            "ORION_LATTIGO_STREAMING_LT": "0",
            "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT": "0",
            "ORION_SINGLE_SLOT_ENCODE_WORKERS": str(int(args.encode_workers)),
            "ORION_WPC_PERIODICITY_PROFILE": "1",
            "ORION_WPC_ENCODED_QP_VERIFY": "1" if args.verify_encoded_qp else "0",
            "ORION_WPC_PERIODICITY_JSONL": str(jsonl_path),
            "ORION_WPC_PERIODICITY_SUMMARY": str(summary_path),
            "ORION_WPC_MODEL": str(args.network),
            "ORION_WPC_MODE": str(args.mode),
            "ORION_WPC_CHECKPOINT_HASH": (
                str(args.checkpoint_hash)
                if args.checkpoint_hash
                else f"none:deterministic-seed={int(args.seed)}"
            ),
        }
    )
    if bool(args.verify_encoded_qp):
        from orion.experimental.wpc_periodicity import real_lattigo_library_path

        verifier_library = real_lattigo_library_path()
        if not verifier_library.is_file():
            raise SystemExit(
                f"real Lattigo verifier library is missing: {verifier_library}; "
                "run python tools/build_lattigo.py first"
            )
        try:
            verifier_cdll = ctypes.CDLL(str(verifier_library))
            getattr(verifier_cdll, "VerifyWPCEncodedDiagonal")
        except (OSError, AttributeError) as exc:
            raise SystemExit(
                f"real Lattigo library is missing the encoded-Q/P verifier: {exc}; "
                "rebuild it from the current source"
            ) from exc
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/run_lattigo_e2e_compare.py"),
        "--backend",
        "lattigo",
        "--network",
        str(args.network),
        "--mode",
        str(args.mode),
        "--out",
        str(result_path),
        "--seed",
        str(int(args.seed)),
        "--forward-runs",
        "1",
        "--warmup-runs",
        "0",
        "--io-mode",
        "none",
    ]

    manifest = {
        "profile": "wpc_orion_slot_periodicity_census",
        "timing_policy": "audit_only_not_for_performance_claims",
        "environment": {
            key: env[key]
            for key in (
                "ORION_LATTIGO_CLEAR_BACKEND",
                "ORION_SINGLE_SLOT_LAYER_CACHE",
                "ORION_LATTIGO_STREAMING_LT",
                "ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT",
                "ORION_SINGLE_SLOT_ENCODE_WORKERS",
                "ORION_WPC_PERIODICITY_PROFILE",
                "ORION_WPC_ENCODED_QP_VERIFY",
                "ORION_WPC_MODEL",
                "ORION_WPC_MODE",
                "ORION_WPC_CHECKPOINT_HASH",
            )
        },
        "command": shlex.join(command),
        "outputs": {
            "clear_result": str(result_path),
            "periodicity_jsonl": str(jsonl_path),
            "periodicity_summary": str(summary_path),
        },
    }
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if bool(args.dry_run):
        return 0

    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if int(completed.returncode) != 0:
        print(f"Periodicity census runner exited with {completed.returncode}", file=sys.stderr)
        return int(completed.returncode)

    result = _load_json(result_path)
    summary = _load_json(summary_path)
    errors: list[str] = []
    if str(result.get("status", "")) != "ok":
        errors.append(f"clear runner status is {result.get('status')!r}, expected 'ok'")
    if int(result.get("measured_forward_ok_count", 0) or 0) != 1:
        errors.append("the one requested clear forward did not succeed")
    if not bool(dict(result.get("mae_vs_clear", {}) or {}).get("shape_match", False)):
        errors.append("clear output shape does not match the reference")
    expected_scope = (
        "slot_message_plus_encoded_qp_candidates"
        if bool(args.verify_encoded_qp)
        else "slot_message_only"
    )
    if str(summary.get("scope", "")) != expected_scope:
        errors.append(
            f"periodicity summary scope is {summary.get('scope')!r}, "
            f"expected {expected_scope!r}"
        )
    occurrence = dict(summary.get("occurrence", {}) or {})
    if int(occurrence.get("observed_count", 0) or 0) <= 0:
        errors.append("periodicity census recorded no diagonal occurrences")
    if int(summary.get("metadata_complete_occurrence_count", 0) or 0) != int(
        occurrence.get("observed_count", 0) or 0
    ):
        errors.append("one or more periodicity records has incomplete identity metadata")
    encoded_verification = dict(summary.get("encoded_qp_verification", {}) or {})
    if bool(args.verify_encoded_qp):
        candidate_count = int(encoded_verification.get("candidate_occurrence_count", 0) or 0)
        attempted_count = int(encoded_verification.get("attempted_occurrence_count", 0) or 0)
        passed_count = int(encoded_verification.get("passed_occurrence_count", 0) or 0)
        if candidate_count <= 0:
            errors.append("encoded-Q/P verification was requested but found no candidates")
        if attempted_count != candidate_count:
            errors.append(
                "encoded-Q/P verification did not attempt every slot-periodic candidate"
            )
        if passed_count != candidate_count or not bool(encoded_verification.get("complete", False)):
            errors.append("one or more encoded-Q/P candidate failed exact reconstruction")

    if errors:
        print("WPC periodicity census validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    report = {
        "result": "valid_slot_message_census",
        "encoded_representation_verification_required": bool(
            summary.get("encoded_representation_verification_required", True)
        ),
        "encoded_qp_verification": encoded_verification,
        "observed_count": occurrence.get("observed_count"),
        "nonzero_count": occurrence.get("nonzero_count"),
        "periodic_count": occurrence.get("periodic_count"),
        "all_zero_count": occurrence.get("all_zero_count"),
        "count_coverage_pct": occurrence.get("count_coverage_pct"),
        "byte_coverage_pct": occurrence.get("byte_coverage_pct"),
        "partial_storage_compression_ratio": occurrence.get(
            "partial_storage_compression_ratio"
        ),
        "outputs": manifest["outputs"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
