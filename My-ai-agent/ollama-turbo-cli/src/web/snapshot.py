from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import hashlib
from datetime import datetime, timezone


@dataclass
class Snapshot:
    """Immutable snapshot of a fetched page (normalized).

    Fields mirror SPEC.schemas.snapshot. This module is scaffolding for PR1 and
    is not wired into the pipeline yet.
    """
    id: str
    url: str
    fetched_at: str  # ISO8601
    headers: Dict[str, Any]
    normalized_content_hash: str
    normalized_content: str


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def make_snapshot_id(url: str, fetched_at: str, normalized_hash: str) -> str:
    """Deterministic ID for the snapshot based on URL + fetched_at + content hash."""
    base = (url or "") + "\n" + (fetched_at or "") + "\n" + (normalized_hash or "")
    return _sha256_hex(base.encode("utf-8", "ignore"))


def build_snapshot(url: str, headers: Optional[Dict[str, Any]], normalized_content: str, *, fetched_at: Optional[str] = None) -> Snapshot:
    headers = dict(headers or {})
    fetched_at_iso = fetched_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    norm_hash = _sha256_hex((normalized_content or "").encode("utf-8", "ignore"))
    sid = make_snapshot_id(url, fetched_at_iso, norm_hash)
    return Snapshot(
        id=sid,
        url=url,
        fetched_at=fetched_at_iso,
        headers=headers,
        normalized_content_hash=norm_hash,
        normalized_content=normalized_content or "",
    )
