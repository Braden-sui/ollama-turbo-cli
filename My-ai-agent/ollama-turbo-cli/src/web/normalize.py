from __future__ import annotations

from typing import Dict, Any, List
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


def canonicalize(url: str) -> str:
    if not url:
        return url
    p = urlparse(url)
    scheme = 'https'
    host = (p.hostname or '').lower()
    # Strip default ports and normalize netloc
    netloc = host
    try:
        if p.port and p.port not in (80, 443):
            netloc = f"{host}:{p.port}"
    except Exception:
        netloc = host
    path = p.path or '/'
    if path != '/' and path.endswith('/'):
        path = path[:-1]
    # Clean query
    bad_keys = {"ref", "source", "icmpid"}
    qs = []
    for k, v in parse_qsl(p.query, keep_blank_values=False):
        kl = (k or '').lower()
        if kl.startswith('utm_') or kl in bad_keys:
            continue
        qs.append((k, v))
    query = urlencode(sorted(qs))
    return urlunparse((scheme, netloc, path, '', query, ''))


def _norm_title(t: str) -> str:
    return ' '.join((t or '').strip().lower().split())


def dedupe_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for c in citations:
        url = canonicalize(c.get('canonical_url') or c.get('url') or '')
        title = _norm_title(c.get('title') or '')
        key = (url, title)
        if key in seen:
            continue
        seen.add(key)
        c2 = dict(c)
        c2['canonical_url'] = url
        out.append(c2)
    return out

