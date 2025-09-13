from __future__ import annotations

import json
import os
from datetime import datetime, timezone
import pytest

from src.debug.ledger import log_veracity


@pytest.mark.debug
def test_ledger_scrubs_headers_and_indexes(tmp_path, monkeypatch):
    cache_root = tmp_path / ".webcache"
    cache_root.mkdir(parents=True, exist_ok=True)

    run_id = "run_test_1234"
    # Build entry with sensitive headers
    entry = {
        "run_id": run_id,
        "headers": {
            "Content-Type": "text/html",
            "Cookie": "SESSION=abc",
            "Authorization": "Bearer secret",
            "Nested": {"Set-Cookie": "x=y"},
        },
        "query": "q",
    }
    log_veracity(str(cache_root), entry)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    ledger_path = cache_root / f"veracity_ledger.{today}.jsonl"
    idx_path = cache_root / "veracity_ledger.index.json"

    assert ledger_path.exists()
    assert idx_path.exists()

    # Read last line and ensure headers scrubbed
    with open(ledger_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines, "ledger should contain at least one entry"
    last = json.loads(lines[-1])
    hdrs = last.get("headers") or {}
    assert "cookie" not in {k.lower() for k in hdrs.keys()}
    assert "authorization" not in {k.lower() for k in hdrs.keys()}
    # Nested removal
    nested = (hdrs.get("Nested") or {})
    assert "set-cookie" not in {k.lower() for k in nested.keys()}

    # Index must contain run_id mapping
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    ent = idx.get(run_id)
    assert ent and ent.get("file") == ledger_path.name and isinstance(ent.get("offset"), int)
