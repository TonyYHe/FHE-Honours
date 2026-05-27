from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RAM_GATE_NODES = ("output", "enc1b", "dec1a", "dec1b", "enc4b", "bottleneckb")


def _gib(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value) / (1024 ** 3):.1f}"
    except (TypeError, ValueError):
        return ""


def _seconds(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return ""


def _shape(value: Any) -> str:
    if not value:
        return ""
    return "x".join(str(int(v)) for v in value)


def _kernel(module: dict[str, Any]) -> str:
    kernel = module.get("kernel_size")
    padding = module.get("padding")
    if not kernel:
        return ""
    k = "x".join(str(int(v)) for v in kernel)
    if padding:
        p = "x".join(str(int(v)) for v in padding)
        return f"{k}/pad{p}"
    return k


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "bad_json", "result_file": str(path)}


def _ram_gate_row(node: str, result_dir: Path) -> list[str]:
    path = result_dir / f"u22_exact_base64_dense_{node}_ram_gate_corg.json"
    data = _load_json(path)
    if data is None:
        return [node, "pending", "", "", "", "", "900", "", "", "", "", "", ""]
    module = data.get("module") or {}
    summary = data.get("memory_summary") or {}
    peak = summary.get("peak_maxrss_bytes") or data.get("watchdog_max_rss_bytes")
    note = ""
    if data.get("status") == "rss_cap_exceeded":
        note = "exceeded cap before completing layer"
    elif data.get("status") == "running":
        note = "currently running / partial"
    elif data.get("status") == "ok":
        note = "completed"
    return [
        node,
        str(data.get("status", "")),
        str(data.get("last_event", "")),
        f"{_shape(module.get('input_shape'))} -> {_shape(module.get('output_shape'))}",
        _kernel(module),
        _gib(peak),
        str(data.get("rss_cap_gib", "900") or "900"),
        _seconds(data.get("generate_diagonals_s")),
        _seconds(data.get("compile_s")),
        _seconds(data.get("forward_s")),
        "" if data.get("source_ciphertext_count") is None else str(data.get("source_ciphertext_count")),
        "" if data.get("output_ciphertext_count") is None else str(data.get("output_ciphertext_count")),
        note,
    ]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
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


def update_ram_gate(doc_path: Path, result_dir: Path) -> None:
    headers = [
        "layer",
        "status",
        "last event",
        "input -> output",
        "kernel",
        "peak RSS GiB",
        "cap GiB",
        "generate s",
        "compile s",
        "forward s",
        "source ct",
        "output ct",
        "note",
    ]
    rows = [_ram_gate_row(node, result_dir) for node in RAM_GATE_NODES]
    table = _markdown_table(headers, rows)
    text = doc_path.read_text(encoding="utf-8")
    text = _replace_block(text, "U22_RAM_GATE_TABLE", table)
    doc_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=Path, default=Path("docs/u22_orion_streaming_haloed_mainline.md"))
    parser.add_argument("--result-dir", type=Path, default=Path(".tmp/results"))
    args = parser.parse_args()
    update_ram_gate(Path(args.doc), Path(args.result_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
