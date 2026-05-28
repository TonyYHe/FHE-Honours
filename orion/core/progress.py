from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _progress_context() -> dict[str, Any]:
    raw = os.environ.get("ORION_PROGRESS_CONTEXT", "")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _progress_record(event: str, fields: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "time_s": float(time.time()),
        "pid": int(os.getpid()),
        "event": str(event),
    }
    record.update(_progress_context())
    record.update(fields)
    return record


def write_progress_event(event: str, **fields: Any) -> None:
    jsonl_path = os.environ.get("ORION_PROGRESS_JSONL", "")
    state_path = os.environ.get("ORION_PROGRESS_STATE_JSON", "")
    if not jsonl_path and not state_path:
        return
    try:
        record = _progress_record(str(event), dict(fields))
        if jsonl_path:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        if state_path:
            path = Path(state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(path)
    except Exception:
        return
