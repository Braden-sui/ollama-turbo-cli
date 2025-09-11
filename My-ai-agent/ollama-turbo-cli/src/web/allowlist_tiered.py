"""
Tiered allowlist loader and matcher for research-mode reliability.

- Loads structured JSON from WEB_TIERED_ALLOWLIST_FILE (preferred), otherwise falls back
  to `resources/research_allowlist.json` colocated with this module if present. Returns None
  when neither is available.
- Provides a simple wildcard matcher where '*' matches any subdomain path segment
  (host-level only; path components in seeds are ignored for matching).
- Returns tiers: 0 (primary), 1 (highly reliable), 2 (reputable with corroboration), None (unmatched).
- Also exposes discouraged patterns for exclusion, and exposes category lookup and
  policy staleness windows by category name.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse


@dataclass
class TieredAllowlist:
    patterns_by_tier: List[Tuple[str, int]]
    seeds_by_tier: List[Tuple[str, int]]
    discouraged: List[str]
    policy: Dict[str, Any]
    # Extended indices carrying category names for better policy mapping
    patterns_by_cat: List[Tuple[str, int, str]]
    seeds_by_cat: List[Tuple[str, int, str]]

    def tier_for_host(self, host: str) -> Optional[int]:
        h = (host or '').lower().strip('.')
        if not h:
            return None
        # Seeds first (cheap suffix match)
        for seed, tier in self.seeds_by_tier:
            s = seed.lower().strip()
            if not s:
                continue
            # Drop any path from seed
            if '/' in s:
                s = s.split('/', 1)[0]
            if h == s or h.endswith('.' + s):
                return tier
        # Wildcard patterns next
        for pat, tier in self.patterns_by_tier:
            if _match_host_pattern(h, pat):
                return tier
        return None

    def category_for_host(self, host: str) -> Optional[str]:
        """Return the first matching category name for a host based on seeds first, then patterns."""
        h = (host or '').lower().strip('.')
        if not h:
            return None
        for seed, tier, cat in self.seeds_by_cat:
            s = seed.lower().strip()
            if not s:
                continue
            if '/' in s:
                s = s.split('/', 1)[0]
            if h == s or h.endswith('.' + s):
                return cat
        for pat, tier, cat in self.patterns_by_cat:
            if _match_host_pattern(h, pat):
                return cat
        return None

    def staleness_days_for_category(self, category_name: Optional[str]) -> Optional[int]:
        """Map category names to policy staleness buckets from the loaded policy.
        Uses heuristics based on category name. Returns None if no suitable bucket found.
        """
        if not category_name:
            return None
        name = category_name.lower()
        buckets = (self.policy or {}).get('staleness_defaults_days') or {}
        def _get(key: str) -> Optional[int]:
            try:
                v = buckets.get(key)
                return int(v) if v is not None else None
            except Exception:
                return None
        # Heuristic mapping
        if ('law' in name) or ('court' in name) or ('legislation' in name) or ('elections' in name):
            return _get('gov_law')
        if ('stat' in name) or ('census' in name):
            return _get('gov_stats')
        if 'health' in name:
            return _get('health')
        if ('science' in name) or ('academia' in name) or ('libraries' in name) or ('archives' in name):
            return _get('science')
        if ('news' in name) or ('press' in name) or ('media' in name):
            return _get('news_wires')
        if ('finance' in name) or ('market' in name) or ('regulator' in name) or ('company_registries' in name):
            return _get('finance')
        if ('weather' in name) or ('hazard' in name) or ('geology' in name) or ('geography' in name):
            return _get('weather_hazards')
        if 'standards' in name:
            return _get('standards')
        if ('tech_docs' in name) or ('developer' in name) or ('web_standards' in name):
            return _get('tech_docs')
        return None

    def discouraged_host(self, host: str) -> bool:
        h = (host or '').lower().strip('.')
        for pat in self.discouraged or []:
            if _match_host_pattern(h, pat):
                return True
        return False


def _match_host_pattern(host: str, pattern: str) -> bool:
    """
    Simple wildcard matcher for hostnames.
    - '*' matches any subdomain segment(s), but we only operate on host (no path).
    - Examples:
      *.gov           -> any host ending with .gov
      *.sos.state.*.us -> host contains 'sos.state.' and ends with '.us'
      supreme court hosts (no wildcard) -> equality or suffix.
    """
    if not pattern:
        return False
    pat = pattern.strip().lower()
    # Ignore scheme and path in pattern if present
    if '://' in pat:
        try:
            pat = urlparse(pat).hostname or pat
        except Exception:
            pass
    if '/' in pat:
        pat = pat.split('/', 1)[0]
    # No wildcard -> suffix match
    if '*' not in pat:
        return host == pat or host.endswith('.' + pat)
    # Wildcard handling: split by '*', ensure fragments appear in order, and last fragment aligns as suffix
    parts = [p for p in pat.split('*') if p]
    idx = 0
    for i, frag in enumerate(parts):
        pos = host.find(frag, idx)
        if pos == -1:
            return False
        idx = pos + len(frag)
        # For last fragment, prefer suffix alignment when pattern does not end with '*'
    if not pat.endswith('*'):
        last = parts[-1]
        if not (host.endswith(last)):
            return False
    return True


def _normalize_seed(seed: str) -> Optional[str]:
    if not seed:
        return None
    s = seed.strip().lower()
    try:
        if '://' in s:
            s = urlparse(s).hostname or s
        if '/' in s:
            s = s.split('/', 1)[0]
        return s.strip('.')
    except Exception:
        return s.strip('.')


def load_tiered_allowlist() -> Optional[TieredAllowlist]:
    path = os.getenv('WEB_TIERED_ALLOWLIST_FILE') or os.getenv('WEB_ALLOWLIST_TIERED_FILE')
    if not path:
        # Default to the packaged resource if present
        try:
            here = os.path.dirname(__file__)
            default_path = os.path.join(here, 'resources', 'research_allowlist.json')
            if os.path.isfile(default_path):
                path = default_path
        except Exception:
            path = None
    if not path:
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
    try:
        cats = (data.get('categories') or {}) if isinstance(data, dict) else {}
        patterns_by_tier: List[Tuple[str, int]] = []
        seeds_by_tier: List[Tuple[str, int]] = []
        patterns_by_cat: List[Tuple[str, int, str]] = []
        seeds_by_cat: List[Tuple[str, int, str]] = []
        for name, obj in cats.items():
            try:
                tier = int(obj.get('tier')) if obj.get('tier') is not None else None
            except Exception:
                tier = None
            if tier is None:
                continue
            for pat in (obj.get('patterns') or []) or []:
                ps = str(pat).strip()
                if ps:
                    patterns_by_tier.append((ps, tier))
                    patterns_by_cat.append((ps, tier, str(name)))
            for sd in (obj.get('seeds') or []) or []:
                ns = _normalize_seed(str(sd))
                if ns:
                    seeds_by_tier.append((ns, tier))
                    seeds_by_cat.append((ns, tier, str(name)))
        discouraged = []
        try:
            discouraged = list(data.get('discouraged_patterns') or [])
        except Exception:
            discouraged = []
        policy = (data.get('policy') or {}) if isinstance(data, dict) else {}
        return TieredAllowlist(
            patterns_by_tier=patterns_by_tier,
            seeds_by_tier=seeds_by_tier,
            discouraged=discouraged,
            policy=policy,
            patterns_by_cat=patterns_by_cat,
            seeds_by_cat=seeds_by_cat,
        )
    except Exception:
        return None
