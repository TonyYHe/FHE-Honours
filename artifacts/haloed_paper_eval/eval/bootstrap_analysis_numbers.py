#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    add_common_args,
    clean_int,
    clean_number,
    ensure_layout,
    maybe_existing_artifact_root,
    print_outputs,
    read_json,
    resolve_run_root,
    tex_int,
    write_csv,
    write_manifest,
)


ARTIFACT = "bootstrap"
SOURCE_SCRIPT = "tools/run_u22_dim32_dense_provider_e2e_matrix.py"


def _case_json(root: Path, case: str, mode: str) -> Path:
    stem = case.replace("x", "_")
    candidates = [
        root / "raw" / "e2e_matrix" / stem / f"{mode}_e2e.json",
        root / stem / f"{mode}_e2e.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(root.rglob(f"{mode}_e2e.json"))
    matches = [path for path in matches if stem in str(path)]
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"missing {case} {mode} E2E JSON under {root}")


def _totals(payload: dict[str, Any]) -> dict[str, Any]:
    return ((payload.get("operator_breakdown_after_forward") or {}).get("totals") or {})


def _rotations(payload: dict[str, Any]) -> int:
    return clean_int((payload.get("rotation_report_after_forward") or {}).get("total_rotation_eval_count_estimate", 0))


def _without_bootstrap(totals: dict[str, Any]) -> float:
    return sum(
        clean_number(totals.get(key))
        for key in (
            "mvm_kernel_s",
            "activation_s",
            "executor_rescale_s",
            "executor_accumulate_s",
            "lt_runtime_stream_accumulate_s",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract HaloED paper bootstrap analysis numbers from E2E output breakdowns.")
    add_common_args(parser)
    parser.add_argument("--e2e-root", type=Path, default=None, help="Existing Table 1/E2E run root. Defaults to --check-existing/run-root.")
    args = parser.parse_args()
    run_root = maybe_existing_artifact_root(resolve_run_root(args, ARTIFACT), ARTIFACT) if args.check_existing else resolve_run_root(args, ARTIFACT)
    dirs = ensure_layout(run_root)
    source_root = Path(args.e2e_root or args.check_existing or args.run_root or run_root).resolve()
    if args.dry_run:
        print(f"extract bootstrap analysis from E2E root: {source_root}", flush=True)
        return 0

    dense = read_json(_case_json(source_root, "256x256", "dense"))
    provider = read_json(_case_json(source_root, "256x256", "provider"))
    dense_totals = _totals(dense)
    provider_totals = _totals(provider)

    dense_without_boot = _without_bootstrap(dense_totals)
    provider_without_boot = _without_bootstrap(provider_totals)
    dense_boot = clean_number(dense_totals.get("bootstrap_s"))
    provider_boot = clean_number(provider_totals.get("bootstrap_s"))
    dense_total = dense_without_boot + dense_boot
    provider_total = provider_without_boot + provider_boot
    row = {
        "case": "256x256",
        "dense_bootstrap_s": dense_boot,
        "haloed_bootstrap_s": provider_boot,
        "bootstrap_delta_s": provider_boot - dense_boot,
        "dense_without_bootstrap_s": dense_without_boot,
        "haloed_without_bootstrap_s": provider_without_boot,
        "dense_compute_s": dense_total,
        "haloed_compute_s": provider_total,
        "dense_rotations": _rotations(dense),
        "haloed_rotations": _rotations(provider),
        "rotation_savings": _rotations(dense) - _rotations(provider),
        "compute_speedup": dense_total / provider_total if provider_total else 0.0,
    }
    csv_path = dirs["paper"] / "bootstrap_analysis_numbers.csv"
    md_path = dirs["paper"] / "bootstrap_analysis_numbers.md"
    write_csv(csv_path, [row])
    md_path.write_text(
        "\n".join(
            [
                "# Bootstrap Analysis Numbers",
                "",
                f"- Bootstrap delta: {row['bootstrap_delta_s']:.1f}s",
                f"- Rotation savings: {tex_int(row['rotation_savings'])}",
                f"- Dense eval without bootstrap: {row['dense_without_bootstrap_s']:.1f}s",
                f"- HaloED eval without bootstrap: {row['haloed_without_bootstrap_s']:.1f}s",
                f"- Compute speedup: {row['compute_speedup']:.2f}x",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs = {"summary_csv": str(csv_path), "summary_md": str(md_path)}
    write_manifest(
        run_root,
        artifact=ARTIFACT,
        source_script=SOURCE_SCRIPT,
        command=[],
        outputs=outputs,
        measurement="derived from 256x256 dense/provider E2E operator_breakdown_after_forward totals",
        extra={"source_root": str(source_root)},
    )
    print_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
