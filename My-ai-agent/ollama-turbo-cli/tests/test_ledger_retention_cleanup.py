from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from src.web.ledger import log_veracity


def test_ledger_retention_removes_old_files(tmp_path, monkeypatch):
    cache_root = tmp_path / ".webcache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Create two old ledgers
    old1 = cache_root / "veracity_ledger.20000101.jsonl"
    old2 = cache_root / "veracity_ledger.19990101.jsonl"
    for p in (old1, old2):
        with open(p, "w", encoding="utf-8") as f:
            f.write("{\"ts\":\"2000-01-01T00:00:00Z\"}\n")

    # Set retention to 0 days to force delete of old ledgers on next write
    monkeypatch.setenv("VERACITY_LEDGER_RETENTION_DAYS", "0")

    # Write a fresh entry to trigger rotation/cleanup
    run_id = "retention_test"
    log_veracity(str(cache_root), {"run_id": run_id, "headers": {}})

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    today_path = cache_root / f"veracity_ledger.{today}.jsonl"
    assert today_path.exists(), "today's ledger should exist"

    # Old files should be removed
    assert not old1.exists()
    assert not old2.exists()

    # Index should exist and point to today's file
    idx_path = cache_root / "veracity_ledger.index.json"
    assert idx_path.exists()
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    assert idx.get(run_id, {}).get("file") == today_path.name
