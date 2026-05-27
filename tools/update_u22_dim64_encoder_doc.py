from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NETWORKS = (
    "u22_192_base64_encoder",
    "u22_224_base64_encoder",
    "u22_384x288_base64_encoder",
    "u22_384_base64_encoder",
)

CASES = (
    "enc1a",
    "enc1b",
    "enc2a",
    "enc2b",
    "enc3a",
    "enc3b",
    "enc4a",
    "enc4b",
    "bottlenecka",
    "bottleneckb",
)

INPUT_LABELS = {
    "u22_192_base64_encoder": "192x192",
    "u22_224_base64_encoder": "224x224",
    "u22_384x288_base64_encoder": "384x288",
    "u22_384_base64_encoder": "384x384",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {"status": "bad_json", "error": f"{type(exc).__name__}: {exc}"}


def _seconds(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return ""


def _gib(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value) / (1024 ** 3):.1f}"
    except (TypeError, ValueError):
        return ""


def _shape(value: Any) -> str:
    if not value:
        return ""
    try:
        return "x".join(str(int(v)) for v in value)
    except Exception:
        return str(value)


def _module_summary(result: dict[str, Any] | None) -> str:
    if not result:
        return ""
    module = result.get("module") or {}
    left = _shape(module.get("input_shape"))
    right = _shape(module.get("output_shape"))
    if left or right:
        return f"{left} -> {right}"
    return ""


def _status(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "pending"
    if entry.get("status") == "ok":
        result = entry.get("result") or {}
        return str(result.get("status", "ok"))
    return str(entry.get("failure_kind") or entry.get("status") or "failed")


def _is_streaming(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return ""
    result = entry.get("result") or {}
    bounded = entry.get("bounded_lattigo_dense") or entry.get("bounded_lattigo_provider") or {}
    audit = result.get("bounded_lattigo_dense_audit") or result.get("bounded_lattigo_provider_audit") or {}
    mode = str(bounded.get("mode") or audit.get("mode") or "")
    flag = str(
        bounded.get("streaming_lt")
        or bounded.get("lattigo_streaming_lt")
        or audit.get("streaming_lt")
        or audit.get("lattigo_streaming_lt")
        or ""
    )
    if mode == "streaming" or flag.lower() in {"1", "true", "force", "always", "on"}:
        return "yes"
    if flag == "0":
        return "no"
    return ""


def _compute_s(result: dict[str, Any]) -> Any:
    compute = result.get("resident_compute_s")
    if compute is None:
        compute = result.get("hot_run_mean_s")
    return compute


def _ratio(numerator: Any, denominator: Any) -> str:
    try:
        left = float(numerator)
        right = float(denominator)
    except (TypeError, ValueError):
        return ""
    if right <= 0:
        return ""
    return f"{left / right:.2f}"


def _ct_summary(result: dict[str, Any]) -> str:
    if result.get("source_ciphertext_count") is None and result.get("output_ciphertext_count") is None:
        return ""
    return f"{result.get('source_ciphertext_count', '')}->{result.get('output_ciphertext_count', '')}"


def _note(*entries: dict[str, Any] | None) -> str:
    notes = []
    for label, entry in zip(("dense", "provider"), entries):
        if entry is None or entry.get("status") == "ok":
            continue
        message = str(entry.get("message") or entry.get("stderr_tail") or "").replace("\n", " ")
        if message:
            notes.append(f"{label}: {message[:100]}")
    return "; ".join(notes)


def _entry_row(
    network: str,
    case: str,
    dense_entry: dict[str, Any] | None,
    dense_file: Path | None,
    provider_entry: dict[str, Any] | None,
    provider_file: Path | None,
) -> list[str]:
    dense_result = (dense_entry or {}).get("result") or {}
    provider_result = (provider_entry or {}).get("result") or {}
    dense_compute = _compute_s(dense_result)
    provider_compute = _compute_s(provider_result)
    result_files = []
    if dense_file is not None:
        result_files.append(f"dense:{dense_file}")
    if provider_file is not None:
        result_files.append(f"provider:{provider_file}")
    shape = _module_summary(dense_result) or _module_summary(provider_result)
    return [
        INPUT_LABELS.get(str(network), str(network)),
        str(case),
        _status(dense_entry),
        _status(provider_entry),
        shape,
        _gib((dense_entry or {}).get("peak_worker_rss_bytes")),
        _gib((provider_entry or {}).get("peak_worker_rss_bytes")),
        _seconds(dense_result.get("compile_s")),
        _seconds(provider_result.get("compile_s")),
        _seconds(dense_compute),
        _seconds(provider_compute),
        _ratio(dense_compute, provider_compute),
        "" if dense_result.get("rotation_eval_count") is None else str(dense_result.get("rotation_eval_count")),
        "" if provider_result.get("rotation_eval_count") is None else str(provider_result.get("rotation_eval_count")),
        _ct_summary(dense_result),
        _ct_summary(provider_result),
        _is_streaming(dense_entry),
        _is_streaming(provider_entry),
        "; ".join(result_files),
        _note(dense_entry, provider_entry),
    ]


def _iter_result_files(result_dir: Path) -> list[Path]:
    if result_dir.is_file():
        return [result_dir]
    return sorted(
        (path for path in result_dir.glob("*.json") if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
    )


def _collect_entries(result_dir: Path) -> dict[tuple[str, str, str], tuple[dict[str, Any], Path]]:
    entries: dict[tuple[str, str, str], tuple[dict[str, Any], Path]] = {}
    for path in _iter_result_files(result_dir):
        payload = _load_json(path)
        if not payload:
            continue
        for network_payload in payload.get("networks", []) or []:
            network = str(network_payload.get("network", ""))
            for case_payload in network_payload.get("cases", []) or []:
                case = str(case_payload.get("case", ""))
                paths = case_payload.get("paths") or {}
                for path_kind in ("dense", "provider"):
                    entry = paths.get(path_kind)
                    if isinstance(entry, dict):
                        entries[(network, case, path_kind)] = (entry, path)
    return entries


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        safe = [str(cell).replace("\n", " ").replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(safe) + " |")
    return "\n".join(lines)


def _replace_block(text: str, marker: str, body: str) -> str:
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    left, sep, rest = text.partition(start)
    if not sep:
        raise ValueError(f"missing marker {start}")
    _old, sep, right = rest.partition(end)
    if not sep:
        raise ValueError(f"missing marker {end}")
    return f"{left}{start}\n{body}\n{end}{right}"


def update_doc(doc_path: Path, result_dir: Path) -> None:
    entries = _collect_entries(result_dir)
    headers = [
        "input",
        "node",
        "dense status",
        "provider status",
        "input -> output",
        "dense RSS GiB",
        "provider RSS GiB",
        "dense compile s",
        "provider compile s",
        "dense compute s",
        "provider compute s",
        "dense/provider compute",
        "dense rotations",
        "provider rotations",
        "dense ct",
        "provider ct",
        "dense stream",
        "provider stream",
        "result files",
        "note",
    ]
    rows = []
    for network in NETWORKS:
        for case in CASES:
            dense_entry, dense_path = entries.get((network, case, "dense"), (None, None))
            provider_entry, provider_path = entries.get((network, case, "provider"), (None, None))
            rows.append(_entry_row(network, case, dense_entry, dense_path, provider_entry, provider_path))
    text = doc_path.read_text(encoding="utf-8")
    text = _replace_block(text, "U22_DIM64_ENCODER_BASELINE_TABLE", _markdown_table(headers, rows))
    doc_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=Path, default=Path("docs/u22_orion_streaming_haloed_mainline.md"))
    parser.add_argument("--result-dir", type=Path, default=Path(".tmp/results/u22_dim64_encoder_baseline"))
    args = parser.parse_args()
    update_doc(Path(args.doc), Path(args.result_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
