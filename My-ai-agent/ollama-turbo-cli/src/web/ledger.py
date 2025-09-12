from __future__ import annotations

import os
import json
from typing import Any, Dict
from datetime import datetime, timezone


def log_veracity(cache_root: str, entry: Dict[str, Any]) -> None:
    """Append a JSONL entry to the veracity ledger under the cache root.

    Best-effort: swallow all exceptions. Intended for debug/telemetry only in PR5.
    """
    try:
        path = os.path.join(cache_root or ".", "veracity_ledger.jsonl")
        e = dict(entry or {})
        e.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except Exception:
        return
