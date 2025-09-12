from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse


def _apex(host: str) -> str:
    if not host:
        return ''
    parts = host.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return host


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or '').lower().strip('.')
    except Exception:
        return ''


def _pick_sites_from_categories(tiered, categories_seen: Dict[str, int], *, max_sites: int) -> List[str]:
    # Prefer categories already seen; fall back to Tier 0 seeds
    sites: List[str] = []
    if tiered is None:
        return sites
    # Collect seeds for top categories by frequency
    try:
        top_cats = sorted((categories_seen or {}).items(), key=lambda kv: kv[1], reverse=True)
        # seed list from seeds_by_cat filtered by those categories
        for cat, _ in top_cats:
            for sd, _tier, c in getattr(tiered, 'seeds_by_cat', []) or []:
                if c == cat:
                    s = (sd or '').strip().lower()
                    if s and '/' in s:
                        s = s.split('/', 1)[0]
                    if s and s not in sites:
                        sites.append(s)
                    if len(sites) >= max_sites:
                        return sites
    except Exception:
        pass
    # Fallback to Tier 0 global seeds
    try:
        for sd, tval in getattr(tiered, 'seeds_by_tier', []) or []:
            try:
                if int(tval) == 0:
                    s = (sd or '').strip().lower()
                    if s and '/' in s:
                        s = s.split('/', 1)[0]
                    if s and s not in sites:
                        sites.append(s)
                    if len(sites) >= max_sites:
                        return sites
            except Exception:
                continue
    except Exception:
        pass
    return sites


def adaptive_rescue(query: str, *, cfg, tiered, categories_seen: Dict[str, int], freshness_days: Optional[int], risk: str = 'low', max_sites_low: int = 8, max_sites_high: int = 16, early_exit: bool = True) -> Tuple[List[Any], Dict[str, Any]]:
    """Adaptive rescue sweep seeded from allowlist JSON (seeds only).

    Returns (additional_results, meta). Caller is responsible for dedupe.
    """
    from .search import search  # local import to avoid cycles

    cap = max_sites_low if (str(risk).lower() != 'high') else max_sites_high
    sites = _pick_sites_from_categories(tiered, categories_seen, max_sites=cap)

    added: List[Any] = []
    seen_urls: set[str] = set()
    sites_considered: List[str] = []

    for site in sites:
        sites_considered.append(site)
        try:
            sr_list = search(query, cfg=cfg, site=site, freshness_days=freshness_days)
        except Exception:
            sr_list = []
        for sr in sr_list:
            u = getattr(sr, 'url', '')
            if not u or u in seen_urls:
                continue
            seen_urls.add(u)
            added.append(sr)
        if early_exit and added:
            break

    meta: Dict[str, Any] = {
        'sites_considered': sites_considered,
        'added_count': len(added),
        'risk_mode': risk,
        'cap': cap,
    }
    return added, meta
