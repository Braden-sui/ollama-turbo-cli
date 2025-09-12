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


def _text_key(c: Dict[str, Any]) -> str:
    title = (c.get('title') or '').strip().lower()
    # Include top lines text if available
    lines = c.get('lines') or []
    first_line = ''
    if lines and isinstance(lines, list):
        try:
            first_line = (lines[0].get('text') or lines[0].get('quote') or '').strip().lower()
        except Exception:
            first_line = ''
    key = f"{title}\n{first_line}"
    h = hashlib.sha256()
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
        tkey = _text_key(cit)
        fkey = f"{tkey}:{apex}:{'wire' if is_wire else 'pub'}"
        groups.setdefault(fkey, []).append(idx)

    collapsed = []
    kept_indices = set()
    for fkey, members in groups.items():
        # keep the first as canonical
        if not members:
            continue
        canonical_idx = members[0]
        kept_indices.add(canonical_idx)
        # others considered duplicates in meta only
        meta_groups.append({
            'fingerprint': fkey,
            'canonical': canonical_idx,
            'duplicates': [i for i in members[1:]],
            'size': len(members),
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
