from __future__ import annotations

from typing import Dict, Any, List
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import hashlib
import re


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
    bad_keys = {"gclid", "fbclid", "ref", "mc_cid", "mc_eid", "igshid", "source", "icmpid"}
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


def content_fingerprint(text: str) -> str:
    """Compute a stable 128-bit fingerprint for normalized paragraph shingles.

    - Lowercase, collapse whitespace inside paragraphs.
    - Split into paragraphs by 2+ newlines; fallback to line-based if needed.
    - Use 3-shingle windows over paragraphs; if <3 paragraphs, hash available tokens.
    - Return an md5 hex digest (128-bit) for stability and speed.
    """
    s = (text or '').strip().lower()
    # Split into paragraphs; treat 2+ newlines as paragraph boundaries
    paras_raw = [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n{2,}", s) if p and p.strip()]
    # Fallback to line-based when no paragraph breaks present
    if not paras_raw:
        paras_raw = [re.sub(r"\s+", " ", ln.strip()) for ln in s.splitlines() if ln.strip()]
    # Drop obvious boilerplate and very short lines
    boilerplate_patterns = (
        r"cookie", r"subscribe", r"newsletter", r"privacy", r"terms", r"advertisement",
        r"sign up", r"accept", r"all rights reserved", r"©", r"related articles", r"more stories"
    )
    bp = re.compile("|".join(boilerplate_patterns))
    paras = []
    seen = set()
    for p in paras_raw:
        if len(p) < 20:
            continue
        if bp.search(p):
            continue
        if p in seen:
            continue
        seen.add(p)
        paras.append(p)
    m = hashlib.md5()
    if len(paras) >= 3:
        for i in range(len(paras) - 2):
            shingle = "\n".join(paras[i:i+3]).encode('utf-8', 'ignore')
            m.update(shingle)
    else:
        for p in paras:
            m.update(p.encode('utf-8', 'ignore'))
    return m.hexdigest()


def dedupe_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for c in citations:
        url = canonicalize(c.get('canonical_url') or c.get('url') or '')
        title = _norm_title(c.get('title') or '')
        fp = c.get('content_fingerprint') or ''
        key = (url, title, fp)
        if key in seen:
            continue
        seen.add(key)
        c2 = dict(c)
        c2['canonical_url'] = url
        out.append(c2)
    return out

