#!/usr/bin/env python3
"""Validate and synthesize the collected WPC--Orion trade-off evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.experimental.wpc_tradeoff_synthesis import (  # noqa: E402
    EXPECTED_STEP1_MODELS,
    EvidenceValidationError,
    build_synthesis,
    summarize_periodicity,
)


DEFAULT_ROOT = REPO_ROOT / ".tmp/results/honours"
DEFAULT_OUT = DEFAULT_ROOT / "26_wpc_orion_tradeoff_synthesis"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step1",
        type=Path,
        default=DEFAULT_ROOT / "06_step1_extracted/extracted_results.json",
    )
    parser.add_argument(
        "--resnet-summary",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "09_wpc_periodicity_census/resnet20_cifar10_dense.periodicity.summary.json"
        ),
    )
    parser.add_argument(
        "--resnet-jsonl",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "09_wpc_periodicity_census/resnet20_cifar10_dense.periodicity.jsonl"
        ),
    )
    parser.add_argument(
        "--unet-summary",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "10_wpc_encoded_qp_verification/u22_64_base32_provider.periodicity.summary.json"
        ),
    )
    parser.add_argument(
        "--unet-jsonl",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "10_wpc_encoded_qp_verification/u22_64_base32_provider.periodicity.jsonl"
        ),
    )
    parser.add_argument(
        "--vgg-summary",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "09_wpc_periodicity_census/vgg16_run1/vgg16_imgnet_provider.periodicity.summary.json"
        ),
    )
    parser.add_argument(
        "--vgg-jsonl",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "09_wpc_periodicity_census/vgg16_run1/vgg16_imgnet_provider.periodicity.jsonl"
        ),
    )
    parser.add_argument(
        "--full-validation",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "23_wpc_rotation_padding_full_validation/full_validation_2115.json"
        ),
    )
    parser.add_argument(
        "--decoder-correctness",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "24_wpc_finetuned_decoder_fhe/finetuned_decoder_fhe.json"
        ),
    )
    parser.add_argument(
        "--decoder-benchmark",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "25_wpc_finetuned_decoder_isolated_benchmark/server_run1/comparison.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / "checkpoints/wpc_rotation_padding_covid19_cheb7_audited_restart"
            / "rotation_padding_best.pt"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write validated JSON/CSV/Markdown but skip Matplotlib figures.",
    )
    return parser


def _resolve(path: Path, *, name: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise EvidenceValidationError(f"missing {name}: {candidate}")
    return candidate


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceValidationError(f"{name} is not a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _artifact(path: Path, *, digest: str | None = None) -> dict[str, Any]:
    return {
        "path": _portable(path),
        "sha256": digest or _sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvidenceValidationError(f"refusing to write empty table: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _mib(value: int | float) -> float:
    return float(value) / (1024.0**2)


def _accuracy_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    accuracy = dict(summary["rotation_padding_accuracy"])
    rows = []
    for key, label in (
        ("native_zero_padding", "Native zero padding"),
        ("rotation_padding_before_finetune", "Rotation Padding before fine-tuning"),
        ("rotation_padding_finetuned", "Rotation Padding after fine-tuning"),
    ):
        row = dict(accuracy[key])
        row = {"configuration": label, **row}
        rows.append(row)
    return rows


def _benchmark_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    row = dict(summary["trained_decoder_isolated_benchmark"])
    return [
        {
            "representation": "Full Q/P",
            "median_forward_s": row["full_forward_median_s"],
            "logical_resident_bytes": row["logical_full_resident_bytes"],
            "pre_online_rss_bytes": row["full_pre_online_rss_bytes"],
            "online_peak_rss_bytes": row["full_online_peak_rss_bytes"],
            "go_heap_inuse_after_compile_gc_bytes": row[
                "full_go_heap_inuse_after_compile_gc_bytes"
            ],
        },
        {
            "representation": "WPC compressed Q/P",
            "median_forward_s": row["compressed_forward_median_s"],
            "logical_resident_bytes": row["logical_compressed_resident_bytes"],
            "pre_online_rss_bytes": row["compressed_pre_online_rss_bytes"],
            "online_peak_rss_bytes": row["compressed_online_peak_rss_bytes"],
            "go_heap_inuse_after_compile_gc_bytes": row[
                "compressed_go_heap_inuse_after_compile_gc_bytes"
            ],
        },
    ]


def _write_plots(summary: Mapping[str, Any], out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise EvidenceValidationError(
            "Matplotlib is unavailable; install project dependencies or use --no-plots"
        ) from exc

    written: list[str] = []
    profiles = list(summary["model_profiles"])
    labels = [str(row["model"]) for row in profiles]
    x = list(range(len(labels)))

    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    values = [float(row["online_encode_pct"]) for row in profiles]
    errors = [float(row["online_encode_pct_sample_std"]) for row in profiles]
    bars = axis.bar(x, values, yerr=errors, capsize=4, color="#4c78a8")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Percent of HE-forward wall time")
    axis.set_ylim(0.0, 100.0)
    axis.set_title("Orion online Encode share")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=3)
    fig.tight_layout()
    path = out_dir / "online_encode_share.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    censuses = list(summary["orion_layout_periodicity"])
    coverage = [float(row["count_coverage_pct"]) for row in censuses]
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    bars = axis.bar(x, coverage, color="#f58518")
    maximum = max(coverage) if coverage else 0.0
    axis.set_ylim(0.0, max(0.001, maximum * 1.35))
    axis.set_xticks(x, [str(row["model"]) for row in censuses])
    axis.set_ylabel("Periodic nonzero diagonals (%)")
    axis.set_title("WPC periodicity in unchanged Orion layouts")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.6f}%" for value in coverage], padding=3)
    fig.tight_layout()
    path = out_dir / "orion_periodicity_coverage.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    benchmark = dict(summary["trained_decoder_isolated_benchmark"])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4))
    pair_labels = ["Full Q/P", "WPC compressed"]
    colors = ["#9c9c9c", "#54a24b"]
    pairs = (
        (
            "Median forward",
            "Seconds",
            [
                benchmark["full_forward_median_s"],
                benchmark["compressed_forward_median_s"],
            ],
            ".3f",
        ),
        (
            "Logical resident plaintexts",
            "MiB",
            [
                _mib(benchmark["logical_full_resident_bytes"]),
                _mib(benchmark["logical_compressed_resident_bytes"]),
            ],
            ".1f",
        ),
        (
            "Measured online peak RSS",
            "MiB",
            [
                _mib(benchmark["full_online_peak_rss_bytes"]),
                _mib(benchmark["compressed_online_peak_rss_bytes"]),
            ],
            ".1f",
        ),
    )
    for axis, (title, ylabel, values_pair, number_format) in zip(axes, pairs):
        bars = axis.bar([0, 1], values_pair, color=colors)
        axis.set_xticks([0, 1], pair_labels, rotation=12, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(
            bars,
            labels=[format(float(value), number_format) for value in values_pair],
            padding=3,
        )
        axis.set_ylim(0.0, max(values_pair) * 1.18)
    fig.suptitle("Fine-tuned encrypted decoder-stage trade-off")
    fig.tight_layout()
    path = out_dir / "trained_decoder_tradeoff.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    accuracy = dict(summary["rotation_padding_accuracy"])
    accuracy_labels = ["Native zero padding", "Fine-tuned Rotation Padding"]
    dice = [
        accuracy["native_zero_padding"]["dice"],
        accuracy["rotation_padding_finetuned"]["dice"],
    ]
    iou = [
        accuracy["native_zero_padding"]["iou"],
        accuracy["rotation_padding_finetuned"]["iou"],
    ]
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    width = 0.34
    positions = [0, 1]
    left = [value - width / 2 for value in positions]
    right = [value + width / 2 for value in positions]
    dice_bars = axis.bar(left, dice, width, label="Dice", color="#4c78a8")
    iou_bars = axis.bar(right, iou, width, label="IoU", color="#e45756")
    axis.set_xticks(positions, accuracy_labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title(
        f"Full {int(accuracy['validation_count']):,}-sample U-Net validation"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.bar_label(dice_bars, labels=[f"{value:.4f}" for value in dice], padding=3)
    axis.bar_label(iou_bars, labels=[f"{value:.4f}" for value in iou], padding=3)
    fig.tight_layout()
    path = out_dir / "rotation_padding_accuracy.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)
    return written


def _render_report(summary: Mapping[str, Any], *, plots: Sequence[str]) -> str:
    profiles = list(summary["model_profiles"])
    censuses = list(summary["orion_layout_periodicity"])
    accuracy = dict(summary["rotation_padding_accuracy"])
    correctness = dict(summary["trained_decoder_fhe_correctness"])
    benchmark = dict(summary["trained_decoder_isolated_benchmark"])
    derived = dict(summary["derived"])
    unet_census = next(
        row for row in censuses if row["network"] == "u22_64_base32"
    )
    lines = [
        "# WPC–Orion layout trade-off",
        "",
        "## Result",
        "",
        "The unchanged Orion layouts do not expose periodic learned-weight diagonals: the "
        "three-model census found **"
        f"{derived['periodic_learned_weight_candidate_count_across_models']}** across "
        "ResNet20, U-Net22, and VGG16. U-Net's "
        "high online Encode share therefore does not make selective WPC compression of its "
        "existing Orion representation effective. The "
        f"{unet_census['periodic_structural_count']} U-Net candidates are verified "
        "structural concatenation transforms and cover only "
        f"{unet_census['byte_coverage_pct']:.6f}% of encoded bytes.",
        "",
        "Changing the decoder to WPC CIPS/Rotation Padding produces a different result. In "
        "the isolated fine-tuned encrypted decoder stage, compressed Q/P storage reduced "
        f"logical resident plaintexts by **{benchmark['logical_storage_reduction_pct']:.2f}%** "
        f"({benchmark['logical_storage_compression_ratio']:.2f}x), reduced measured online "
        f"peak RSS by **{benchmark['online_peak_rss_reduction_pct']:.2f}%**, and increased "
        f"median forward latency by **{benchmark['latency_overhead_pct']:.2f}%**. This is a "
        "decoder-stage result, not a complete encrypted U-Net result.",
        "",
        "## Orion whole-model profiles",
        "",
        (
            "| Model | Mode | HE forward (s) | Online Encode (s) | Encode "
            "share | Bootstrap share | MVM share |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in profiles:
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['he_forward_s']:.2f} | "
            f"{row['online_encode_s']:.2f} | {row['online_encode_pct']:.2f}% | "
            f"{row['bootstrap_pct']:.2f}% | {row['mvm_kernel_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "These are accepted real-FHE schema-v2 profiles. Major wall categories are "
            "additive and close to HE-forward time; non-additive operator microtimers are "
            "not used in this comparison.",
            "",
        ]
    )
    if "online_encode_share.png" in plots:
        lines.extend(["![Orion online Encode share](online_encode_share.png)", ""])

    lines.extend(
        [
            "## Periodicity of unchanged Orion layouts",
            "",
            (
                "| Model | Nonzero diagonals | Periodic learned weights | Periodic "
                "structural | Count coverage | Byte coverage | Selective ratio |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in censuses:
        lines.append(
            f"| {row['model']} | {row['nonzero_count']:,} | "
            f"{row['periodic_learned_weight_count']:,} | "
            f"{row['periodic_structural_count']:,} | "
            f"{row['count_coverage_pct']:.6f}% | {row['byte_coverage_pct']:.6f}% | "
            f"{row['partial_storage_compression_ratio']:.6f}x |"
        )
    lines.extend(
        [
            "",
            "The U-Net structural candidates passed exact encoded-Q/P reconstruction; "
            "they are not learned convolution, transposed-convolution, or linear weights.",
            "",
        ]
    )
    if "orion_periodicity_coverage.png" in plots:
        lines.extend(
            [
                (
                    "![WPC periodicity coverage in unchanged Orion layouts]"
                    "(orion_periodicity_coverage.png)"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Fine-tuned WPC decoder stage",
            "",
            "| Metric | Full Q/P | WPC compressed Q/P |",
            "|---|---:|---:|",
            (
                f"| Median forward | {benchmark['full_forward_median_s']:.6f} s | "
                f"{benchmark['compressed_forward_median_s']:.6f} s |"
            ),
            (
                "| Logical resident plaintexts | "
                f"{_mib(benchmark['logical_full_resident_bytes']):.2f} MiB | "
                f"{_mib(benchmark['logical_compressed_resident_bytes']):.2f} MiB |"
            ),
            (
                "| Pre-online RSS | "
                f"{_mib(benchmark['full_pre_online_rss_bytes']):.2f} MiB | "
                f"{_mib(benchmark['compressed_pre_online_rss_bytes']):.2f} MiB |"
            ),
            (
                "| Measured online peak RSS | "
                f"{_mib(benchmark['full_online_peak_rss_bytes']):.2f} MiB | "
                f"{_mib(benchmark['compressed_online_peak_rss_bytes']):.2f} MiB |"
            ),
            "",
            f"Median decompression was {benchmark['decompression_median_s'] * 1000.0:.3f} ms "
            f"({benchmark['decompression_median_pct_of_forward']:.2f}% of compressed "
            "forward time). Both paths performed identical operations, and the maximum "
            f"isolated output delta was {benchmark['max_abs_delta_between_isolated_outputs']:.3e}.",
            "",
            f"The separate FHE correctness gate used {correctness['compressed_transform_count']} "
            "compressed learned transforms, made zero online weight Encode calls, matched "
            f"the full-Q/P output within {correctness['compressed_full_final_max_abs_delta']:.3e}, "
            f"and had maximum error {correctness['max_compressed_qp_error_vs_clear']:.3e} "
            "against the independent clear reference.",
            "",
        ]
    )
    if "trained_decoder_tradeoff.png" in plots:
        lines.extend(
            [
                "![Fine-tuned encrypted decoder-stage trade-off](trained_decoder_tradeoff.png)",
                "",
            ]
        )

    native = accuracy["native_zero_padding"]
    before = accuracy["rotation_padding_before_finetune"]
    final = accuracy["rotation_padding_finetuned"]
    lines.extend(
        [
            "## Rotation-Padding accuracy",
            "",
            "| Configuration | Finite samples | Dice | IoU | Loss |",
            "|---|---:|---:|---:|---:|",
            (
                f"| Native zero padding | {native['finite_sample_count']:,}/"
                f"{native['sample_count']:,} | {native['dice']:.6f} | "
                f"{native['iou']:.6f} | {native['loss']:.6f} |"
            ),
            (
                "| Rotation Padding before fine-tuning | "
                f"{before['finite_sample_count']:,}/{before['sample_count']:,} | "
                f"{before['dice']:.6f} | {before['iou']:.6f} | "
                f"{before['loss']:.6f} |"
            ),
            (
                "| Rotation Padding after fine-tuning | "
                f"{final['finite_sample_count']:,}/{final['sample_count']:,} | "
                f"{final['dice']:.6f} | {final['iou']:.6f} | "
                f"{final['loss']:.6f} |"
            ),
            "",
            "Before fine-tuning, one sample was non-finite, so its aggregate metrics are "
            "finite-sample diagnostics rather than a directly comparable full-set result. "
            f"Fine-tuning restored finite outputs for all {accuracy['validation_count']:,} "
            "samples. The final Dice gap "
            f"to native zero padding was {accuracy['native_to_finetuned_dice_delta']:.6f} "
            f"({accuracy['native_dice_retained_pct']:.2f}% of native Dice retained).",
            "",
        ]
    )
    if "rotation_padding_accuracy.png" in plots:
        lines.extend(
            [
                (
                    "![Full validation accuracy after Rotation-Padding fine-tuning]"
                    "(rotation_padding_accuracy.png)"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Conclusion and scope",
            "",
            "The proposed explanation—high online Encode share implies that WPC can "
            "selectively compress many diagonals in the existing Orion U-Net layout—is not "
            "supported. Encode share and exact periodic learned-weight coverage are separate "
            "properties. WPC's benefit appears only after adopting its CIPS/Rotation-Padding "
            "layout: the trained decoder stage shows a large memory reduction with a small "
            "latency cost, while fine-tuning recovers numerical stability but leaves an "
            "accuracy gap to native zero padding.",
            "",
            "The experiment does not yet establish a complete encrypted-network WPC-versus-"
            "Orion comparison. Such a claim requires matched end-to-end encrypted U-Net and "
            "ResNet executions under both layouts.",
            "",
            "## Validation",
            "",
            "All input schemas, acceptance gates, model identities, accounting closures, "
            "census JSONL counts, candidate classifications, encoded-Q/P verification, and "
            "cross-artifact checkpoint hashes were checked. `artifact_manifest.csv` and "
            "`synthesis.json` record the exact input paths, sizes, and SHA-256 hashes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    try:
        paths = {
            "step1": _resolve(args.step1, name="Step-1 extraction"),
            "resnet_summary": _resolve(args.resnet_summary, name="ResNet census summary"),
            "resnet_jsonl": _resolve(args.resnet_jsonl, name="ResNet census JSONL"),
            "unet_summary": _resolve(args.unet_summary, name="U-Net census summary"),
            "unet_jsonl": _resolve(args.unet_jsonl, name="U-Net census JSONL"),
            "vgg_summary": _resolve(args.vgg_summary, name="VGG census summary"),
            "vgg_jsonl": _resolve(args.vgg_jsonl, name="VGG census JSONL"),
            "full_validation": _resolve(args.full_validation, name="full validation"),
            "decoder_correctness": _resolve(
                args.decoder_correctness, name="decoder correctness"
            ),
            "decoder_benchmark": _resolve(args.decoder_benchmark, name="decoder benchmark"),
            "checkpoint": _resolve(args.checkpoint, name="fine-tuned checkpoint"),
        }
        payloads = {
            key: _load_json(path, name=key)
            for key, path in paths.items()
            if key not in {"resnet_jsonl", "unet_jsonl", "vgg_jsonl", "checkpoint"}
        }
        census_specs = (
            (
                "resnet20_cifar10",
                "ResNet20",
                "dense",
                paths["resnet_summary"],
                paths["resnet_jsonl"],
                False,
            ),
            (
                "u22_64_base32",
                "U-Net22",
                "provider",
                paths["unet_summary"],
                paths["unet_jsonl"],
                True,
            ),
            (
                "vgg16_imgnet",
                "VGG16",
                "provider",
                paths["vgg_summary"],
                paths["vgg_jsonl"],
                False,
            ),
        )
        periodicity_rows: list[dict[str, Any]] = []
        streamed_hashes: dict[Path, str] = {}
        for network, display, mode, summary_path, jsonl_path, require_encoded in census_specs:
            row, jsonl_sha = summarize_periodicity(
                _load_json(summary_path, name=f"{network} census summary"),
                jsonl_path,
                network=network,
                display_name=display,
                expected_mode=mode,
                require_encoded_verification=require_encoded,
            )
            periodicity_rows.append(row)
            streamed_hashes[jsonl_path] = jsonl_sha

        checkpoint_sha = _sha256(paths["checkpoint"])
        streamed_hashes[paths["checkpoint"]] = checkpoint_sha
        summary = build_synthesis(
            step1=payloads["step1"],
            periodicity_rows=periodicity_rows,
            accuracy=payloads["full_validation"],
            decoder_correctness=payloads["decoder_correctness"],
            decoder_benchmark=payloads["decoder_benchmark"],
            checkpoint_sha256=checkpoint_sha,
            checkpoint_path=paths["checkpoint"],
        )
        artifacts = {
            key: _artifact(path, digest=streamed_hashes.get(path))
            for key, path in paths.items()
        }
        summary["input_artifacts"] = artifacts

        out_dir = args.out_dir.expanduser()
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        plots = [] if args.no_plots else _write_plots(summary, out_dir)
        summary["outputs"] = {
            "report": "report.md",
            "model_profiles_csv": "model_profiles.csv",
            "periodicity_census_csv": "periodicity_census.csv",
            "rotation_padding_accuracy_csv": "rotation_padding_accuracy.csv",
            "trained_decoder_benchmark_csv": "trained_decoder_benchmark.csv",
            "artifact_manifest_csv": "artifact_manifest.csv",
            "plots": plots,
        }

        _write_csv(out_dir / "model_profiles.csv", summary["model_profiles"])
        _write_csv(
            out_dir / "periodicity_census.csv", summary["orion_layout_periodicity"]
        )
        _write_csv(out_dir / "rotation_padding_accuracy.csv", _accuracy_rows(summary))
        _write_csv(out_dir / "trained_decoder_benchmark.csv", _benchmark_rows(summary))
        _write_csv(
            out_dir / "artifact_manifest.csv",
            [{"artifact": name, **row} for name, row in artifacts.items()],
        )
        (out_dir / "synthesis.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (out_dir / "report.md").write_text(
            _render_report(summary, plots=plots), encoding="utf-8"
        )
    except EvidenceValidationError as exc:
        print(f"EVIDENCE INVALID: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "result": str(out_dir / "synthesis.json"),
                "report": str(out_dir / "report.md"),
                "plots": [str(out_dir / name) for name in plots],
                "hypothesis_assessment": summary["derived"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
