from __future__ import annotations
import time
import re
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional
from urllib.parse import urlparse

from .config import WebConfig


@dataclass
class _DomainStats:
    sum_scores: float = 0.0
    count: int = 0
    last_ts: float = 0.0


@dataclass
class TrustCache:
    max_items: int = 512
    ttl_seconds: int = 3600
    _store: Dict[str, _DomainStats] = field(default_factory=dict)

    def update(self, host: str, score: float) -> float:
        now = time.time()
        st = self._store.get(host)
        if st is None:
            st = _DomainStats(sum_scores=score, count=1, last_ts=now)
        else:
            if now - st.last_ts > self.ttl_seconds:
                st = _DomainStats(sum_scores=score, count=1, last_ts=now)
            else:
                st.sum_scores += score
                st.count += 1
                st.last_ts = now
        self._store[host] = st
        # LRU-like trim by dropping the stalest entries
        if len(self._store) > self.max_items:
            oldest = sorted(self._store.items(), key=lambda kv: kv[1].last_ts)[:int(self.max_items * 0.1) or 1]
            for k, _ in oldest:
                self._store.pop(k, None)
        return st.sum_scores / max(1, st.count)

    def get(self, host: str) -> Optional[float]:
        st = self._store.get(host)
        if not st:
            return None
        if time.time() - st.last_ts > self.ttl_seconds:
            self._store.pop(host, None)
            return None
        return st.sum_scores / max(1, st.count)


_CACHE = TrustCache()


def _logistic(x: float) -> float:
    # squash to 0..1, centered near 0.5
    import math
    return 1.0 / (1.0 + math.exp(-4.0 * (x - 0.5)))


def _tld_weight(host: str) -> float:
    h = (host or '').lower()
    if h.endswith('.gov') or h.endswith('.edu') or h.endswith('.ac.uk'):
        return 0.25
    if h.endswith('.org'):
        return 0.05
    if any(h.endswith(t) for t in ('.top', '.xyz', '.click', '.zip', '.review')):
        return -0.2
    return 0.0


def _url_quality(u: str) -> float:
    try:
        p = urlparse(u)
        score = 0.0
        if p.scheme == 'https':
            score += 0.05
        depth = len([x for x in (p.path or '/').split('/') if x])
        if 1 <= depth <= 6:
            score += 0.05
        # penalize excessive query params
        if p.query and len(p.query) > 160:
            score -= 0.05
        if any(s in (p.netloc or '').lower() for s in ('bit.ly', 't.co', 'tinyurl', 'goo.gl')):
            score -= 0.2
        # clickbait-ish slug patterns
        if re.search(r'\b(all\-you\-need|everything\-you\-need|shocking|must\-see)\b', (p.path or '').lower()):
            score -= 0.05
        return score
    except Exception:
        return 0.0


def _meta_signals(extracted: Dict[str, Any]) -> float:
    s = 0.0
    if extracted.get('date'):
        s += 0.15
    title = (extracted.get('title') or '').strip()
    if 8 <= len(title) <= 140:
        s += 0.05
    lang = str((extracted.get('meta', {}) or {}).get('lang') or '').lower()
    if lang.startswith('en'):
        s += 0.02
    md_len = len((extracted.get('markdown') or '')[:20000])
    if md_len > 2000:
        s += 0.12
    elif md_len > 800:
        s += 0.06
    # simple link density heuristic
    body = extracted.get('markdown') or ''
    links = len(re.findall(r'https?://', body))
    if md_len > 0:
        ratio = links / max(1, md_len / 1000.0)
        if 0.1 <= ratio <= 1.5:
            s += 0.03
        elif ratio > 3.0:
            s -= 0.05
    return s


def compute_heuristic_trust(url: str, extracted: Dict[str, Any], cfg: WebConfig) -> Tuple[float, Dict[str, float]]:
    try:
        host = (urlparse(url).hostname or '').lower().strip('.')
    except Exception:
        host = ''
    base = 0.5
    signals: Dict[str, float] = {}
    # TLD and url quality
    tw = _tld_weight(host)
    uq = _url_quality(url)
    ms = _meta_signals(extracted)
    signals['tld'] = tw
    signals['url'] = uq
    signals['meta'] = ms
    raw = base + tw + uq + ms
    # squash
    score = max(0.0, min(1.0, _logistic(raw)))
    # stabilize with domain cache
    avg = _CACHE.update(host, score) if host else score
    signals['domain_avg'] = avg
    return score, signals


def trust_score(url: str, extracted: Dict[str, Any], cfg: WebConfig) -> Tuple[float, Dict[str, float]]:
    mode = (getattr(cfg, 'trust_mode', 'allowlist') or 'allowlist').strip().lower()
    # For now, ML shares the same heuristic path until an ML backend is configured
    if mode in {'heuristic', 'ml', 'open'}:
        return compute_heuristic_trust(url, extracted, cfg)
    # allowlist mode does not use trust scoring
    return 0.0, {}
