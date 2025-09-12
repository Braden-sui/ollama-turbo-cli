from __future__ import annotations

import os
import json
from typing import Any, Dict
from datetime import datetime, timedelta, timezone


_SCRUB_KEYS = {"cookie", "authorization", "set-cookie", "proxy-authorization"}


def _scrub_headers(obj: Any) -> Any:
    try:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                kl = str(k).lower()
                if kl in _SCRUB_KEYS:
                    continue
                out[k] = _scrub_headers(v)
            return out
        if isinstance(obj, list):
            return [_scrub_headers(v) for v in obj]
        return obj
    except Exception:
        return obj


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _rotate_and_index(cache_root: str, run_id: str, payload: Dict[str, Any]) -> None:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    base = cache_root or "."
    _ensure_dir(base)
    ledger_path = os.path.join(base, f"veracity_ledger.{today}.jsonl")
    idx_path = os.path.join(base, "veracity_ledger.index.json")
    # Cap entry size
    try:
        max_bytes = int(os.getenv("VERACITY_LEDGER_MAX_BYTES", "65536"))
    except Exception:
        max_bytes = 65536
    line = json.dumps(payload, ensure_ascii=False)
    if len(line.encode("utf-8")) > max_bytes:
        # simple truncation with marker
        trunc = max(0, max_bytes - 16)
        line = line.encode("utf-8")[:trunc].decode("utf-8", errors="ignore") + "…"
    # Append and capture offset
    try:
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)
            offset = f.tell()
            f.write(line + "\n")
    except Exception:
        return
    # Update index
    try:
        idx = {}
        if os.path.isfile(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                try:
                    idx = json.load(f) or {}
                except Exception:
                    idx = {}
        idx[str(run_id or "")] = {"file": os.path.basename(ledger_path), "offset": int(offset)}
        tmp = idx_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(idx, f)
        os.replace(tmp, idx_path)
    except Exception:
        pass
    # Retention cleanup
    try:
        days = int(os.getenv("VERACITY_LEDGER_RETENTION_DAYS", "14"))
        # days >= 0 means enforce retention; 0 keeps only today's file
        if days >= 0:
            today_dt = datetime.now(timezone.utc)
            cutoff_date = (today_dt - timedelta(days=days)).date()
            for fn in os.listdir(base):
                if not fn.startswith("veracity_ledger.") or not fn.endswith(".jsonl"):
                    continue
                try:
                    stamp = fn.split(".")[1]
                    dt = datetime.strptime(stamp, "%Y%m%d").date()
                    if dt < cutoff_date:
                        try:
                            os.remove(os.path.join(base, fn))
                        except Exception:
                            pass
                except Exception:
                    continue
    except Exception:
        pass


def log_veracity(cache_root: str, entry: Dict[str, Any]) -> None:
    """Append a JSONL entry to the veracity ledger with scrubbed data and daily rotation.

    Best-effort; swallows exceptions.
    """
    try:
        e = dict(entry or {})
        e.setdefault("ts", datetime.now(timezone.utc).isoformat())
        # Scrub any headers-like fields
        if isinstance(e.get("headers"), (dict, list)):
            e["headers"] = _scrub_headers(e["headers"])
        run_id = e.get("run_id") or os.getenv("WEB_RUN_ID") or ""
        _rotate_and_index(cache_root, str(run_id), e)
    except Exception:
        return
