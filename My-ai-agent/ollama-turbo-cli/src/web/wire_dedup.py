from __future__ import annotations

from typing import Dict, Any, List, Tuple
from urllib.parse import urlparse
import hashlib


WIRE_HOSTS = {
    'reuters.com',
    'apnews.com',
    'associatedpress.com',
    'bloomberg.com',
    'afp.com',
    'upi.com',
}


def _apex(host: str) -> str:
    if not host:
        return ''
    parts = host.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return host


def _fingerprint(c: Dict[str, Any]) -> str:
    fp = c.get('content_fingerprint') or ''
    if fp:
        return fp
    # Fallback: approximate by title + first line hash if fingerprint is absent
    title = (c.get('title') or '').strip().lower()
    lines = c.get('lines') or []
    first_line = ''
    if lines and isinstance(lines, list):
        try:
            first_line = (lines[0].get('text') or lines[0].get('quote') or '').strip().lower()
        except Exception:
            first_line = ''
    key = f"{title}\n{first_line}"
    h = hashlib.md5()
    h.update(key.encode('utf-8', 'ignore'))
    return h.hexdigest()


def collapse_citations(citations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collapse near-duplicate wire/syndication articles for analysis.

    Returns (collapsed_list, meta). Meta contains groups and counts for telemetry.
    This scaffold intentionally does not alter pipeline behavior unless enabled.
    """
    groups: Dict[str, List[int]] = {}
    meta_groups: List[Dict[str, Any]] = []

    for idx, cit in enumerate(citations or []):
        try:
            host = (urlparse(cit.get('canonical_url') or cit.get('url') or '').hostname or '').lower()
        except Exception:
            host = ''
        apex = _apex(host)
        is_wire = apex in WIRE_HOSTS
        fp = _fingerprint(cit)
        # One group per content fingerprint; include apex/wire tag in meta only
        fkey = fp
        groups.setdefault(fkey, []).append(idx)

    collapsed = []
    kept_indices = set()
    for fkey, members in groups.items():
        if not members:
            continue
        # Choose representative deterministically
        def _tier_val(c: Dict[str, Any]) -> int:
            try:
                tv = c.get('tier')
                return int(tv) if tv is not None else 3
            except Exception:
                return 3
        def _body_len(c: Dict[str, Any]) -> int:
            try:
                return int(c.get('body_char_count') or 0)
            except Exception:
                return 0
        def _date_ts(c: Dict[str, Any]) -> float:
            try:
                return float(c.get('date_ts') or 0.0) or 0.0
            except Exception:
                return 0.0
        def _has_canonical(c: Dict[str, Any]) -> int:
            try:
                return 1 if bool(c.get('has_canonical')) else 0
            except Exception:
                return 0
        def _url(c: Dict[str, Any]) -> str:
            return (c.get('canonical_url') or c.get('url') or '')
        # Preference: lower tier (0 best), then longer body, then earliest date, then canonical rel present, finally URL tie-break
        members_sorted = sorted(members, key=lambda i: (
            _tier_val(citations[i]),
            -_body_len(citations[i]),
            _date_ts(citations[i]) if _date_ts(citations[i]) > 0 else float('inf'),
            -_has_canonical(citations[i]),
            _url(citations[i])
        ))
        canonical_idx = members_sorted[0]
        kept_indices.add(canonical_idx)
        # others considered duplicates in meta only
        meta_groups.append({
            'wire_group_id': fkey,
            'canonical': canonical_idx,
            'duplicates': [i for i in members if i != canonical_idx],
            'size': len(members),
            'chosen_reason': {
                'tier': _tier_val(citations[canonical_idx]),
                'body_chars': _body_len(citations[canonical_idx]),
                'date_ts': _date_ts(citations[canonical_idx]) or None,
                'has_canonical': bool(_has_canonical(citations[canonical_idx])),
            }
        })

    for i, cit in enumerate(citations or []):
        if i in kept_indices:
            collapsed.append(cit)

    meta: Dict[str, Any] = {
        'total': len(citations or []),
        'kept': len(collapsed),
        'collapsed_count': max(0, len(citations or []) - len(collapsed)),
        'groups': meta_groups,
    }
    return collapsed, meta
