from __future__ import annotations

from typing import Any, Dict, Tuple
from urllib.parse import urlparse


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or '').lower().strip('.')
    except Exception:
        return ''


def compute_prior(snapshot, claim=None, *, tiered=None) -> Tuple[float, Dict[str, Any]]:
    """Compute a simple source prior based on allowlist tier/category (scaffold).

    Returns (prior, features). Prior is in [0,1]. Defaults to 0.5 when unknown.
    """
    url = getattr(snapshot, 'url', '') or ''
    h = _host(url)
    tier = None
    cat = None
    try:
        if tiered is not None and h:
            tier = tiered.tier_for_host(h)
            cat = tiered.category_for_host(h)
    except Exception:
        tier = None
        cat = None
    # Simple mapping with mild spread; tune later with calibration
    prior = 0.5
    try:
        t = int(tier) if tier is not None else None
        if t == 0:
            prior = 0.6
        elif t == 1:
            prior = 0.5
        elif t == 2:
            prior = 0.45
    except Exception:
        prior = 0.5
    feats = {
        'host': h,
        'tier': tier if tier is not None else 'unknown',
        'category': cat if cat else 'unknown',
    }
    return prior, feats
