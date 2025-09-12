from __future__ import annotations
import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from .robots import RobotsPolicy
from .config import WebConfig
# Ensure .env.local/.env are loaded even when this module is imported outside the CLI
# (e.g., tests or custom orchestrators). This mirrors src/config.py with a safe fallback.
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    try:
        path_local = find_dotenv('.env.local', usecwd=True)
        if path_local:
            load_dotenv(path_local, override=True)
    except Exception:
        pass
    try:
        path_default = find_dotenv('.env', usecwd=True)
        if path_default:
            load_dotenv(path_default, override=False)
    except Exception:
        pass
except Exception:
    # Fallback: minimal manual loader for .env.local and .env
    import os as _os
    def _find_upwards(_filename: str) -> str:
        try:
            _cwd = _os.getcwd()
        except Exception:
            return ""
        _cur = _cwd
        while True:
            _cand = _os.path.join(_cur, _filename)
            if _os.path.isfile(_cand):
                return _cand
            _parent = _os.path.dirname(_cur)
            if not _parent or _parent == _cur:
                break
            _cur = _parent
        return ""
    def _load_env_file(_path: str, *, _override: bool) -> None:
        if not _path:
            return
        try:
            with open(_path, 'r', encoding='utf-8') as _f:
                for _line in _f:
                    _s = _line.strip()
                    if not _s or _s.startswith('#') or '=' not in _s:
                        continue
                    _k, _v = _s.split('=', 1)
                    _key = _k.strip(); _val = _v.strip().strip('"').strip("'")
                    if _override or (_key not in _os.environ):
                        _os.environ[_key] = _val
        except Exception:
            pass
        # Wire/syndication dedup preview
        try:
            if 'wire_dedup_enable' in locals() and wire_dedup_enable:
                _collapsed, wmeta = collapse_citations(citations[:top_k])
                answer['debug']['wire'] = wmeta
        except Exception:
            pass
    try:
        _p_local = _find_upwards('.env.local')
        if _p_local:
            _load_env_file(_p_local, _override=True)
    except Exception:
        pass
    try:
        _p_env = _find_upwards('.env')
        if _p_env:
            _load_env_file(_p_env, _override=False)
    except Exception:
        pass
from . import progress as _progress
from .search import search, SearchResult, _year_guard, _drain_throttle_events
from .fetch import fetch_url, _httpx_client
from .trust import trust_score
from .extract import extract_content
from .rerank import chunk_text, rerank_chunks
from .archive import save_page_now, get_memento
from .snapshot import build_snapshot
from .claims import extract_claims
from src.validators.claim_validation import validate_claim
from .evidence import score_evidence
from .wire_dedup import collapse_citations, WIRE_HOSTS
from .rescue import adaptive_rescue
from .corroborate import compute_corroboration, claim_key
from .counter_claim import evaluate_counter_claim
from .reputation import compute_prior
from .ledger import log_veracity
from .normalize import canonicalize, dedupe_citations, content_fingerprint
from .loc import format_loc
from datetime import datetime, timedelta

try:
    from dateutil import parser as _date_parser  # type: ignore
except Exception:  # pragma: no cover
    _date_parser = None  # type: ignore

_DEFAULT_CFG: Optional[WebConfig] = None

def set_default_config(cfg: WebConfig) -> None:
    """Inject a centralized default WebConfig used when callers omit cfg.

    This allows higher-level clients (e.g., OllamaTurboClient) to set a single
    source of truth for all web pipeline modules without threading cfg through
    every call site. Tests that monkeypatch functions may continue to omit cfg.
    """
    global _DEFAULT_CFG
    _DEFAULT_CFG = cfg


def _query_cache_key(query: str, opts: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(query.encode())
    h.update(json.dumps(opts, sort_keys=True).encode())
    return h.hexdigest()


def run_research(query: str, *, cfg: Optional[WebConfig] = None, site_include: Optional[str] = None, site_exclude: Optional[str] = None, freshness_days: Optional[int] = None, top_k: int = 8, force_refresh: bool = False) -> Dict[str, Any]:
    cfg = cfg or _DEFAULT_CFG or WebConfig()
    # Honor per-call env overrides for dynamic fields used in tests and runtime tuning
    try:
        env_excl = os.getenv("WEB_EXCLUDE_CITATION_DOMAINS")
        if env_excl is not None:
            cfg.exclude_citation_domains = [d.strip().lower() for d in env_excl.split(',') if d.strip()]
        env_cache = os.getenv("WEB_CACHE_ROOT")
        if env_cache:
            cfg.cache_root = env_cache
        # Ensure debug metrics can be toggled per-call even if a global default config is set
        env_debug = os.getenv("WEB_DEBUG_METRICS")
        if env_debug is not None:
            cfg.debug_metrics = str(env_debug).strip().lower() not in {"0","false","no","off"}
        # Allow tests/callers to enable the recency soft-accept fallback via env
        env_soft_accept = os.getenv("WEB_RECENCY_SOFT_ACCEPT_WHEN_EMPTY")
        if env_soft_accept is not None:
            cfg.recency_soft_accept_when_empty = str(env_soft_accept).strip().lower() not in {"0","false","no","off"}
        # Allow tests/callers to toggle dateline soft-accept policy via env
        env_dateline_soft = os.getenv("WEB_DATELINE_SOFT_ACCEPT")
        if env_dateline_soft is not None:
            cfg.dateline_soft_accept = str(env_dateline_soft).strip().lower() not in {"0","false","no","off"}
        # Tier sweep (multi-turn allowlist/tier-first search) — default ON; disable with WEB_TIER_SWEEP=0
        env_tier_sweep = os.getenv("WEB_TIER_SWEEP")
        try:
            if env_tier_sweep is not None:
                cfg.enable_tier_sweep = str(env_tier_sweep).strip().lower() not in {"0","false","no","off"}
            else:
                # default to True if not explicitly set on cfg
                if getattr(cfg, 'enable_tier_sweep', None) is None:
                    cfg.enable_tier_sweep = True
        except Exception:
            try:
                cfg.enable_tier_sweep = True
            except Exception:
                pass
        # Sweep caps and strictness
        env_sweep_max = os.getenv("WEB_TIER_SWEEP_MAX_SITES")
        try:
            if env_sweep_max is not None:
                cfg.tier_sweep_max_sites = max(1, int(env_sweep_max))
            else:
                if getattr(cfg, 'tier_sweep_max_sites', None) is None:
                    cfg.tier_sweep_max_sites = 12
        except Exception:
            try:
                cfg.tier_sweep_max_sites = 12
            except Exception:
                pass
        env_sweep_strict = os.getenv("WEB_TIER_SWEEP_STRICT")
        try:
            if env_sweep_strict is not None:
                cfg.tier_sweep_strict = str(env_sweep_strict).strip().lower() not in {"0","false","no","off"}
            else:
                if getattr(cfg, 'tier_sweep_strict', None) is None:
                    cfg.tier_sweep_strict = False
        except Exception:
            try:
                cfg.tier_sweep_strict = False
            except Exception:
                pass
        # Evidence-first flags: env overrides
        try:
            env_ef = os.getenv("EVIDENCE_FIRST")
            if env_ef is not None:
                cfg.evidence_first = str(env_ef).strip().lower() not in {"0","false","no","off"}
        except Exception:
            pass
        try:
            env_ef_ks = os.getenv("EVIDENCE_FIRST_KILL_SWITCH")
            if env_ef_ks is not None:
                cfg.evidence_first_kill_switch = str(env_ef_ks).strip().lower() not in {"0","false","no","off"}
        except Exception:
            pass
        # Optional telemetry/filters
        try:
            env_prefilt = os.getenv("WEB_PREFILTER_DISCOURAGED")
            prefilter_discouraged = (str(env_prefilt).strip().lower() not in {"0","false","no","off"}) if env_prefilt is not None else False
        except Exception:
            prefilter_discouraged = False
        try:
            env_tier_first = os.getenv("WEB_TIER_FIRST_PASS")
            tier_first_enabled = (str(env_tier_first).strip().lower() not in {"0","false","no","off"}) if env_tier_first is not None else False
        except Exception:
            tier_first_enabled = False
        try:
            env_tier_first_max = os.getenv("WEB_TIER_FIRST_PASS_MAX_SITES")
            tier_first_max_sites = max(1, int(env_tier_first_max)) if env_tier_first_max not in (None, "") else 4
        except Exception:
            tier_first_max_sites = 4
        # Wire dedup and rescue preview flags
        try:
            env_wire = os.getenv("WEB_WIRE_DEDUP_ENABLE")
            wire_dedup_enable = (str(env_wire).strip().lower() not in {"0","false","no","off"}) if env_wire is not None else False
        except Exception:
            wire_dedup_enable = False
        try:
            env_rescue = os.getenv("WEB_RESCUE_SWEEP")
            if env_rescue is None:
                # PR15: backward compatibility with legacy name
                env_rescue = os.getenv("WEB_RESCUE_PREVIEW")
            rescue_sweep = (str(env_rescue).strip().lower() not in {"0","false","no","off"}) if env_rescue is not None else False
        except Exception:
            rescue_sweep = False
        try:
            env_corro = os.getenv("WEB_CORROBORATE_ENABLE")
            corroborate_enable = (str(env_corro).strip().lower() not in {"0","false","no","off"}) if env_corro is not None else False
        except Exception:
            corroborate_enable = False
        # PR5 flags
        try:
            env_cc = os.getenv("WEB_COUNTER_CLAIM_ENABLE")
            counter_claim_enable = (str(env_cc).strip().lower() not in {"0","false","no","off"}) if env_cc is not None else False
        except Exception:
            counter_claim_enable = False
        try:
            env_rep = os.getenv("WEB_REPUTATION_ENABLE")
            reputation_enable = (str(env_rep).strip().lower() not in {"0","false","no","off"}) if env_rep is not None else False
        except Exception:
            reputation_enable = False
        try:
            env_led = os.getenv("WEB_VERACITY_LEDGER_ENABLE")
            ledger_enable = (str(env_led).strip().lower() not in {"0","false","no","off"}) if env_led is not None else False
        except Exception:
            ledger_enable = False
        # PR6 cutover preparation flag (telemetry only)
        try:
            env_cut = os.getenv("WEB_CUTOVER_PREP")
            cutover_prep = (str(env_cut).strip().lower() not in {"0","false","no","off"}) if env_cut is not None else False
        except Exception:
            cutover_prep = False
        # PR7: rescue strategy reporting (behavior unchanged; used in summary only)
        try:
            rescue_strategy = (os.getenv("WEB_RESCUE_STRATEGY", "adaptive") or "adaptive").strip().lower()
        except Exception:
            rescue_strategy = "adaptive"
    except Exception:
        pass
    os.makedirs(cfg.cache_root, exist_ok=True)
    # YearGuard: strip templated year tails (no injection of month/year)
    try:
        q_sanitized, yg_counters = _year_guard(query, cfg)
    except Exception:
        q_sanitized, yg_counters = (query, {"stripped_year_tokens": 0})
    # Resolve freshness by policy when not explicitly provided
    def _resolve_relative_span_days(qtext: str) -> Optional[int]:
        try:
            s = (qtext or '').lower()
            now = datetime.now()
            # past N days/weeks/months
            m = re.search(r"\b(past|last)\s+(\d{1,3})\s*(day|days|week|weeks|month|months)\b", s)
            if m:
                n = int(m.group(2))
                unit = m.group(3)
                if unit.startswith('day'):
                    return max(1, n)
                if unit.startswith('week'):
                    return max(1, n * 7)
                if unit.startswith('month'):
                    return max(1, n * 30)
            if 'this year' in s:
                start = datetime(now.year, 1, 1)
                return max(1, (now - start).days or 1)
            if 'last year' in s:
                return 365
            return None
        except Exception:
            return None
    resolved_days = _resolve_relative_span_days(q_sanitized)
    # Breaking/slow topic heuristics (keywords)
    s_low = (q_sanitized or '').lower()
    is_breaking = any(k in s_low for k in ['breaking', 'latest', 'today', 'this week', 'recent'])
    is_slow = any(k in s_low for k in ['standard', 'standards', 'textbook', 'backgrounder', 'overview'])
    freshness_final = (
        int(freshness_days) if (freshness_days is not None) else (
            resolved_days if (resolved_days is not None) else (
                cfg.breaking_freshness_days if is_breaking else (
                    cfg.slow_freshness_days if is_slow else cfg.default_freshness_days
                )
            )
        )
    )
    opts = {
        'site_include': site_include,
        'site_exclude': site_exclude,
        'freshness_days': freshness_final,
        'top_k': top_k,
    }
    key = _query_cache_key(q_sanitized, opts)
    cache_path = os.path.join(cfg.cache_root, f"query_{key}.json")
    now = time.time()
    # Load persistent dedupe index (hash -> url)
    index_path = os.path.join(cfg.cache_root, "content_hash_index.json")
    hash_to_url: Dict[str, str] = {}
    # When forcing refresh, ignore any previous dedupe index to avoid false positives in tests/runs
    if not force_refresh and os.path.isfile(index_path):
        try:
            data = json.loads(open(index_path, 'r', encoding='utf-8').read())
            if isinstance(data, dict):
                # Only keep string->string pairs
                for k, v in list(data.items()):
                    if isinstance(k, str) and isinstance(v, str):
                        hash_to_url[k] = v
        except Exception:
            pass
    if not force_refresh and os.path.isfile(cache_path):
        try:
            data = json.loads(open(cache_path, 'r', encoding='utf-8').read())
            if now - float(data.get('ts', 0)) <= cfg.cache_ttl_seconds:
                return data['result']
        except Exception:
            pass

    # Plan -> Search
    try:
        _progress.emit_current({"stage": "search", "status": "start", "query": query})
    except Exception:
        pass
    _search_t0 = time.time()
    results = search(q_sanitized, cfg=cfg, site=site_include, freshness_days=freshness_final)
    _search_ms = int((time.time() - _search_t0) * 1000)
    try:
        _progress.emit_current({"stage": "search", "status": "done", "count": len(results)})
    except Exception:
        pass

    # Tier/category telemetry placeholders
    tier_first_added = 0
    prefiltered_discouraged_count = 0
    tier_counts = {0: 0, 1: 0, 2: 0}
    categories_seen: Dict[str, int] = {}

    simplified_used = False
    emergency_used = False
    # Debug counters (exposed only if cfg.debug_metrics)
    base_count = len(results)
    simplified_count = 0
    variant_count = 0
    emergency_count = 0
    # Observability counters
    obs_fetch_fail: Dict[str, int] = {}
    obs_extract_fail: int = 0
    obs_extract_fail_by_host: Dict[str, int] = {}
    obs_extract_modes: Dict[str, int] = {}
    obs_discard_old: Dict[str, int] = {}
    obs_source_type: Dict[str, int] = {}
    obs_trust_decisions: List[Dict[str, Any]] = []
    # PR14: fetch timing samples
    fetch_timing_samples: List[Dict[str, Any]] = []
    # PR6 deprecation counters (legacy gates) — telemetry only
    obs_deprecation: Dict[str, int] = {
        'discouraged_domain': 0,
        'blocked_source': 0,
        'blocked_liveblog': 0,
        'blocked_map': 0,
        'excluded_domain': 0,
    }
    # Additional observability to make tests deterministic on counters
    obs_seen_undated: int = 0
    # Dateline instrumentation
    dateline_from_structured = 0
    dateline_from_path = 0
    date_conf_hist: Dict[str, int] = {'high': 0, 'medium': 0, 'low': 0}
    allowlist_fallback_hit = False
    recency_soft_accept_used = False
    undated_soft_count = 0
    if not results:
        # Heuristic: simplify long/narrow queries and retry once
        try:
            import re as _re
            toks = _re.findall(r"[A-Za-z0-9]+", q_sanitized)
            stop = {
                'the','a','an','of','in','on','for','to','and','or','with','about','from','by','at','as',
                'is','are','was','were','be','being','been','this','that','these','those','it','its','into'
            }
            key_toks = [t for t in toks if t.lower() not in stop]
            short_q = " ".join(key_toks[:6]) or q_sanitized[:80]
        except Exception:
            short_q = q_sanitized[:80]
        if short_q and short_q.strip().lower() != query.strip().lower():
            try:
                _progress.emit_current({"stage": "search", "status": "retry", "query": short_q})
            except Exception:
                pass
            results = search(short_q, cfg=cfg, site=site_include, freshness_days=freshness_final)
            simplified_used = True
            simplified_count = len(results)

    if not results:
        # Final variant fallback: try "<ProperNoun> political makeup 2024" style seed
        try:
            import re as _re
            toks = _re.findall(r"[A-Za-z][A-Za-z0-9-]*", q_sanitized)
            proper = None
            for t in toks:
                if t[0].isupper():
                    proper = t
                    break
            core = proper or (toks[0] if toks else "")
            if core:
                # Avoid hardcoding a year; use generic variant without years
                variant_q = f"{core} political makeup"
                try:
                    _progress.emit_current({"stage": "search", "status": "variant", "query": variant_q})
                except Exception:
                    pass
                results = search(variant_q, cfg=cfg, site=site_include, freshness_days=freshness_final)
                variant_count = len(results)
        except Exception:
            pass

    if not results and cfg.emergency_bootstrap:
        # Emergency: call providers directly here to bootstrap candidates
        try:
            def _dedup_add(acc: list[SearchResult], seen: set[str], title: str, url: str, snippet: str, source: str):
                if url and url not in seen and url.startswith('http'):
                    seen.add(url)
                    acc.append(SearchResult(title=title or '', url=url, snippet=snippet or '', source=source, published=None))
            # choose a concise query for emergency path
            em_q = query
            try:
                import re as _re
                toks = _re.findall(r"[A-Za-z0-9]+", query)
                stop = {'the','a','an','of','in','on','for','to','and','or','with','about','from','by','at','as','is','are','was','were','be','being','been','this','that','these','those','it','its','into','2023','2024','2025'}
                em_q2 = " ".join([t for t in toks if t.lower() not in stop][:6])
                if em_q2:
                    em_q = em_q2
            except Exception:
                pass
            tmp: list[SearchResult] = []
            seen_urls: set[str] = set()
            with _httpx_client(cfg) as c:
                # Brave
                if getattr(cfg, 'brave_key', None):
                    try:
                        headers = {"X-Subscription-Token": cfg.brave_key, "Accept": "application/json", "User-Agent": cfg.user_agent}
                        r = c.get("https://api.search.brave.com/res/v1/web/search", params={"q": em_q, "count": 10}, headers=headers, timeout=cfg.timeout_read)
                        if r.status_code == 200:
                            data = r.json()
                            for d in ((data.get('web') or {}).get('results') or []):
                                _dedup_add(tmp, seen_urls, d.get('title',''), d.get('url',''), d.get('description',''), 'brave')
                            # generic crawl fallback
                            if not tmp:
                                def _walk(v):
                                    if isinstance(v, dict):
                                        u = v.get('url') or v.get('link') or v.get('href')
                                        t = v.get('title') or v.get('name') or ''
                                        s = v.get('description') or v.get('snippet') or ''
                                        if isinstance(u, str) and u.startswith('http'):
                                            _dedup_add(tmp, seen_urls, t, u, s, 'brave')
                                        for vv in v.values():
                                            _walk(vv)
                                    elif isinstance(v, list):
                                        for it in v:
                                            _walk(it)
                                _walk(data)
                    except Exception:
                        pass
                # Tavily
                if not tmp and getattr(cfg, 'tavily_key', None):
                    try:
                        headers = {"Authorization": f"Bearer {cfg.tavily_key}", "Content-Type":"application/json"}
                        r = c.post("https://api.tavily.com/search", json={"query": em_q, "search_depth":"basic", "max_results": 10}, headers=headers, timeout=cfg.timeout_read)
                        if r.status_code == 200:
                            data = r.json()
                            for d in data.get('results', []) or []:
                                _dedup_add(tmp, seen_urls, d.get('title',''), d.get('url',''), d.get('content',''), 'tavily')
                    except Exception:
                        pass
                # Exa
                if not tmp and getattr(cfg, 'exa_key', None):
                    try:
                        headers = {"x-api-key": cfg.exa_key, "Content-Type":"application/json"}
                        r = c.post("https://api.exa.ai/search", json={"query": em_q, "numResults": 10}, headers=headers, timeout=cfg.timeout_read)
                        if r.status_code == 200:
                            data = r.json()
                            for d in data.get('results', []) or []:
                                _dedup_add(tmp, seen_urls, d.get('title',''), d.get('url',''), d.get('snippet',''), 'exa')
                    except Exception:
                        pass
            if tmp:
                results = tmp
                emergency_used = True
                emergency_count = len(tmp)
        except Exception:
            pass

    # Wikipedia-guided expansion: use Wikipedia only for discovery.
    # For any Wikipedia results, fetch and extract external reference links,
    # and add those links as new candidates (skipping Wikipedia itself).
    wiki_refs_added = 0
    try:
        def _is_wiki(u: str) -> bool:
            try:
                h = (urlparse(u).hostname or '').lower().strip('.')
                return bool(h) and (h == 'wikipedia.org' or h.endswith('.wikipedia.org'))
            except Exception:
                return False

        expanded: List[SearchResult] = []
        seen_set: set[str] = set()
        # Keep non-wiki results as-is
        for sr in results:
            if not _is_wiki(sr.url):
                if sr.url and sr.url not in seen_set:
                    expanded.append(sr)
                    seen_set.add(sr.url)
        # Expand refs from wiki pages (bounded)
        for sr in results:
            if not _is_wiki(sr.url):
                continue
            try:
                f = fetch_url(sr.url, cfg=cfg, force_refresh=force_refresh, use_browser_if_needed=False)
                if not f.ok:
                    continue
                meta = {
                    'url': f.url,
                    'final_url': f.final_url,
                    'status': f.status,
                    'content_type': f.content_type,
                    'body_path': f.body_path,
                    'headers': f.headers,
                }
                ex = extract_content(meta, cfg=cfg)
                if not ex.ok:
                    continue
                md = ex.markdown or ''
                links = re.findall(r'https?://[^\s\)\]\>\"\']+', md)
                # Filter external links (non-wikipedia) and dedupe
                filtered: List[str] = []
                for lk in links:
                    if _is_wiki(lk):
                        continue
                    if lk not in seen_set:
                        filtered.append(lk)
                        seen_set.add(lk)
                # Add up to 8 references per wiki page to bound fanout
                for lk in filtered[:8]:
                    try:
                        host = (urlparse(lk).hostname or '')
                    except Exception:
                        host = ''
                    title = host or lk
                    expanded.append(SearchResult(title=title, url=lk, snippet='', source='wiki_ref', published=None))
                    wiki_refs_added += 1
            except Exception:
                continue
        # Replace results with expanded list if we added anything
        if wiki_refs_added > 0:
            results = expanded
    except Exception:
        pass

    # Optional tier-first warm start: add small number of Tier 0 seeds (seeded search)
    if tier_first_enabled and (_tiered is not None):
        try:
            sites: list[str] = []
            for sd, tval in getattr(_tiered, 'seeds_by_tier', []) or []:
                try:
                    if int(tval) == 0:
                        s = (sd or '').strip().lower()
                        if s and '/' in s:
                            s = s.split('/', 1)[0]
                        if s and s not in sites:
                            sites.append(s)
                except Exception:
                    continue
            sites = sites[: tier_first_max_sites]
            seen_urls: set[str] = {getattr(sr, 'url', '') for sr in results}
            added: list[Any] = []
            for site in sites:
                try:
                    sr_list = search(q_sanitized, cfg=cfg, site=site, freshness_days=freshness_final)
                except Exception:
                    sr_list = []
                for sr2 in sr_list:
                    u2 = getattr(sr2, 'url', '')
                    if u2 and (u2 not in seen_urls):
                        added.append(sr2)
                        seen_urls.add(u2)
                if len(added) >= tier_first_max_sites:
                    break
            if added:
                results.extend(added)
                tier_first_added = len(added)
        except Exception:
            pass

    # Optional prefilter: drop discouraged hosts before worker submission
    if prefilter_discouraged and (_tiered is not None):
        try:
            kept: list[Any] = []
            for sr in results:
                try:
                    h0 = (urlparse(getattr(sr, 'url', '')).hostname or '').lower().strip('.')
                except Exception:
                    h0 = ''
                if h0 and _tiered.discouraged_host(h0):
                    prefiltered_discouraged_count += 1
                    continue
                kept.append(sr)
            results = kept
        except Exception:
            pass

    # Compute tier/category telemetry on current results
    try:
        if _tiered is not None:
            for sr in results:
                try:
                    h0 = (urlparse(getattr(sr, 'url', '')).hostname or '').lower().strip('.')
                except Exception:
                    h0 = ''
                if not h0:
                    continue
                tv = _tiered.tier_for_host(h0)
                if tv is not None and int(tv) in (0,1,2):
                    tier_counts[int(tv)] = tier_counts.get(int(tv), 0) + 1
                cn = _tiered.category_for_host(h0)
                if cn:
                    categories_seen[cn] = categories_seen.get(cn, 0) + 1
    except Exception:
        pass

    # Rescue preview (do not merge; debug only)
    rescue_meta = None
    try:
        if rescue_preview and (_tiered is not None):
            _added, rescue_meta = adaptive_rescue(q_sanitized, cfg=cfg, tiered=_tiered, categories_seen=categories_seen, freshness_days=freshness_final, risk='low', early_exit=True)
    except Exception:
        rescue_meta = None

    citations: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    # Dedupe across URLs and content bodies (current-run only)
    seen_urls: set[str] = set()
    # Scope content-hash dedupe to hostname, so different outlets with similar wire stubs both survive
    seen_hashes_by_host: Dict[str, set[str]] = {}
    dedupe_lock = threading.Lock()
    # Observability updates happen from worker threads; protect with a lock to avoid lost updates
    obs_lock = threading.Lock()
    # Deterministic accounting for undated items seen during recency gating
    obs_undated_urls: set[str] = set()

    dedup_skips = 0
    excluded_skips = 0

    def _host_in_excluded(u: str) -> bool:
        try:
            host = (urlparse(u).hostname or '').lower().strip('.')
            if not host:
                return False
            for dom in (cfg.exclude_citation_domains or []):
                d = str(dom or '').lower().strip('.')
                if not d:
                    continue
                if host == d or host.endswith('.' + d):
                    return True
            return False
        except Exception:
            return False
    # Policy: allowlist and blocklist for breaking news sources
    def _host(u: str) -> str:
        try:
            return (urlparse(u).hostname or '').lower().strip('.')
        except Exception:
            return ''
    def _in_list(host: str, patterns: List[str]) -> bool:
        h = host or ''
        for dom in patterns:
            d = str(dom or '').lower().strip('.')
            if not d:
                continue
            if h == d or h.endswith('.' + d):
                return True
        return False
    # Defaults and comprehensive allowlist (merged from module + cfg + env)
    try:
        from .allowlist_data import DEFAULT_ALLOWLIST as _DEFAULT_ALLOWLIST, DEFAULT_BLOCKLIST as _DEFAULT_BLOCKLIST  # type: ignore
    except Exception:
        _DEFAULT_ALLOWLIST = [
            'apnews.com','reuters.com','bbc.co.uk','bbc.com','theguardian.com','aljazeera.com','nytimes.com',
            'wsj.com','haaretz.com','timesofisrael.com','al-monitor.com'
        ]
        _DEFAULT_BLOCKLIST = ['liveuamap.com']

    def _merge_lists(*lists):
        seen = set()
        out = []
        for lst in lists:
            for d in (lst or []):
                ds = str(d or '').strip().lower().strip('.')
                if not ds or ds in seen:
                    continue
                seen.add(ds)
                out.append(ds)
        return out

    # Base from module
    allow_list = list(_DEFAULT_ALLOWLIST)
    block_list = list(_DEFAULT_BLOCKLIST)
    # Extend from cfg.allowlist_domains (if provided)
    try:
        cfg_allow2 = list(getattr(cfg, 'allowlist_domains', []) or [])
    except Exception:
        cfg_allow2 = []
    allow_list = _merge_lists(allow_list, cfg_allow2)
    # Extend from env comma-lists (do not replace; merge)
    try:
        env_allow = os.getenv('WEB_NEWS_SOURCES_ALLOW')
        if env_allow:
            allow_env_list = [d.strip() for d in env_allow.split(',') if d.strip()]
            allow_list = _merge_lists(allow_list, allow_env_list)
    except Exception:
        pass
    try:
        env_block = os.getenv('WEB_NEWS_SOURCES_BLOCK')
        if env_block:
            block_env_list = [d.strip() for d in env_block.split(',') if d.strip()]
            block_list = _merge_lists(block_list, block_env_list)
    except Exception:
        pass
    # Optional file-based allowlist (one domain/glob per line)
    try:
        allow_file = os.getenv('WEB_ALLOWLIST_FILE')
        if allow_file and os.path.isfile(allow_file):
            try:
                with open(allow_file, 'r', encoding='utf-8') as _f:
                    file_list = [ln.strip() for ln in _f if ln.strip() and not ln.strip().startswith('#')]
                allow_list = _merge_lists(allow_list, file_list)
            except Exception:
                pass
    except Exception:
        pass

    # Optional tiered allowlist: provides trust tiers (0,1,2) and discouraged patterns
    _tiered = None
    try:
        from .allowlist_tiered import load_tiered_allowlist  # type: ignore
        _tiered = load_tiered_allowlist()
    except Exception:
        _tiered = None

    def _source_type(url: str) -> str:
        try:
            p = urlparse(url)
            h = (p.hostname or '').lower()
            path = (p.path or '').lower()
            if h.endswith('liveuamap.com'):
                return 'map'
            if any(tok in path for tok in ['/live', 'liveblog', 'live-blog', 'ticker', '/updates/']):
                return 'liveblog'
        except Exception:
            pass
        return 'article'

    def _recency_required(q: str, fresh_days: Optional[int]) -> bool:
        try:
            if fresh_days and int(fresh_days) > 0:
                return True
        except Exception:
            pass
        t = (q or '').lower()
        return any(k in t for k in ['today','this week','recent','recently','latest','breaking','past week','last week'])

    # Recency discipline: only when user provided freshness_days or query text strongly implies timeliness
    recency_gate = _recency_required(q_sanitized, freshness_days)
    now_ts = time.time()
    window_secs = (int(freshness_days) * 86400) if (freshness_days and int(freshness_days) > 0) else (7 * 86400)

    def _parse_pub_date(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        try:
            # Try ISO fast-path
            try:
                return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
            except Exception:
                pass
            if _date_parser is not None:
                return _date_parser.parse(s).timestamp()  # type: ignore
        except Exception:
            return None
        return None
    def _build_citation(sr, *, allow_undated_soft: bool = False) -> Optional[Dict[str, Any]]:
        if site_exclude and site_exclude in (sr.url or ''):
            return None
        try:
            _progress.emit_current({"stage": "fetch", "status": "start", "url": sr.url})
        except Exception:
            pass
        # Source-type and policy checks prior to fetch
        stype = _source_type(sr.url)
        with obs_lock:
            obs_source_type[stype] = obs_source_type.get(stype, 0) + 1
        h = _host(sr.url)
        # Tiered category name for policy mapping
        cat_name: Optional[str] = None
        try:
            if _tiered is not None and h:
                cat_name = _tiered.category_for_host(h)
        except Exception:
            cat_name = None

        # Tiered discouraged domains: drop early (but never override explicit allowlist)
        try:
            if _tiered is not None and h and _tiered.discouraged_host(h):
                # Preserve explicit allowlist domains regardless of discouragement
                if not _in_list(h, allow_list):
                    items.append({'url': sr.url, 'ok': False, 'reason': 'discouraged-domain'})
                    if cutover_prep:
                        with obs_lock:
                            obs_deprecation['discouraged_domain'] = obs_deprecation.get('discouraged_domain', 0) + 1
                    return None
        except Exception:
            pass
        if _in_list(h, block_list):
            items.append({'url': sr.url, 'ok': False, 'reason': 'blocked-source'})
            if cutover_prep:
                with obs_lock:
                    obs_deprecation['blocked_source'] = obs_deprecation.get('blocked_source', 0) + 1
            return None
        if stype in {'liveblog','map'}:
            items.append({'url': sr.url, 'ok': False, 'reason': f'blocked-{stype}'})
            if cutover_prep:
                with obs_lock:
                    key = 'blocked_liveblog' if stype == 'liveblog' else 'blocked_map'
                    obs_deprecation[key] = obs_deprecation.get(key, 0) + 1
            return None
        # Respect exclusion list: allow discovery (search) but skip quoting as a citation
        if _host_in_excluded(sr.url):
            items.append({'url': sr.url, 'ok': False, 'reason': 'excluded-domain'})
            nonlocal excluded_skips
            with dedupe_lock:
                excluded_skips += 1
            if cutover_prep:
                with obs_lock:
                    obs_deprecation['excluded_domain'] = obs_deprecation.get('excluded_domain', 0) + 1
            return None
        f = fetch_url(sr.url, cfg=cfg, force_refresh=force_refresh, use_browser_if_needed=True)
        if not f.ok:
            items.append({'url': sr.url, 'ok': False, 'reason': f.reason or f"HTTP {f.status}"})
            try:
                rkey = (f.reason or f"HTTP {f.status}")
                with obs_lock:
                    obs_fetch_fail[rkey] = obs_fetch_fail.get(rkey, 0) + 1
            except Exception:
                pass
            try:
                _progress.emit_current({"stage": "fetch", "status": "error", "url": sr.url, "reason": f.reason or f"HTTP {f.status}"})
            except Exception:
                pass
            return None
        meta = {
            'url': f.url,
            'final_url': f.final_url,
            'status': f.status,
            'content_type': f.content_type,
            'body_path': f.body_path,
            'headers': f.headers,
        }
        try:
            _progress.emit_current({"stage": "extract", "status": "start", "url": f.final_url})
        except Exception:
            pass
        ex = extract_content(meta, cfg=cfg)
        if not ex.ok:
            items.append({'url': sr.url, 'ok': False, 'reason': 'extract-failed'})
            nonlocal obs_extract_fail
            with obs_lock:
                obs_extract_fail += 1
                try:
                    h_fail = (urlparse(f.final_url).hostname or '').lower()
                    if h_fail:
                        obs_extract_fail_by_host[h_fail] = obs_extract_fail_by_host.get(h_fail, 0) + 1
                except Exception:
                    pass
            try:
                _progress.emit_current({"stage": "extract", "status": "error", "url": f.final_url})
            except Exception:
                pass
            return None
        # Trust and wildcard configuration
        try:
            cfg_allow = list(allow_list or [])
        except Exception:
            cfg_allow = []
        try:
            cfg_allow2 = list(getattr(cfg, 'allowlist_domains', []) or [])
        except Exception:
            cfg_allow2 = []
        wildcard = ('*' in cfg_allow) or ('*' in cfg_allow2)
        is_allowlisted = (not wildcard) and (_in_list(h, cfg_allow) or _in_list(h, cfg_allow2))
        # Tiered trust (0/1 trusted; 2 requires corroboration)
        tier_val: Optional[int] = None
        trusted_by_tier = False
        try:
            if _tiered is not None:
                tier_val = _tiered.tier_for_host(h)
                trusted_by_tier = (tier_val is not None) and int(tier_val) in {0, 1}
        except Exception:
            tier_val = None
            trusted_by_tier = False
        # Apply "first-party rule" override if policy indicates and URL path suggests newsroom/press/IR/blog
        try:
            if _tiered is not None and h:
                # Detect first-party newsroom-style path
                _p = ''
                try:
                    _p = (urlparse(sr.url).path or '').lower()
                except Exception:
                    _p = ''
                if any(seg in _p for seg in ['/press', '/newsroom', '/news-room', '/ir', '/investors', '/blog', '/about']):
                    # Elevate to Tier 0 for first-party communications
                    tier_val = 0
                    trusted_by_tier = True
        except Exception:
            pass
        # Social platforms: treat as Tier 2 (never trusted by tier)
        try:
            if _tiered is not None and h:
                social = (((_tiered.policy or {}).get('social_sources') or {}).get('platforms') or [])
                for sp in social:
                    s = (sp or '').strip().lower()
                    if not s:
                        continue
                    if h == s or h.endswith('.' + s):
                        tier_val = 2
                        trusted_by_tier = False
                        break
        except Exception:
            pass
        trust_mode = (getattr(cfg, 'trust_mode', 'allowlist') or 'allowlist').strip().lower()
        use_trust = wildcard or (trust_mode in {'heuristic','ml','open'})
        threshold = float(getattr(cfg, 'trust_threshold', 0.6) or 0.6)
        min_accept = max(0.2, threshold - 0.1)
        # Compute heuristic trust score only when needed
        t_score = 0.0
        t_signals: Dict[str, Any] = {}
        if use_trust and (not is_allowlisted):
            t_score, t_signals = trust_score(f.final_url, {
                'date': ex.date,
                'title': ex.title,
                'markdown': ex.markdown,
                'meta': ex.meta,
            }, cfg)

        decision_note = ''
        decision = 'accept'
            # Initialize date variables for both recency and evergreen paths
        pub_ts: Optional[float] = None
        date_conf: str = 'low'
        date_source: str = 'none'
        date_conflict: Dict[str, Any] = {}
        date_tz: str = 'unknown'

        # Recency/date gating and language sanity for recent events (hardened dateline)
        # PDFs are typically evergreen/primary sources; bypass recency gating
        if recency_gate and ex.kind != 'pdf':
            # Determine date and confidence
            pub_ts = _parse_pub_date(ex.date)
            if pub_ts:
                date_source = 'meta'
                # crude tz detection (Z or [+/-]HH:MM)
                try:
                    ds = str(ex.date or '')
                    if ds.endswith('Z'):
                        date_tz = 'UTC'
                    else:
                        import re as _re
                        m_tz = _re.search(r"([+-]\d{2}:\d{2})$", ds)
                        if m_tz:
                            date_tz = m_tz.group(1)
                except Exception:
                    pass
                date_conf = 'high'
                nonlocal dateline_from_structured
                with obs_lock:
                    dateline_from_structured += 1
            else:
                # Fallback: parse date from URL path (e.g., /2025/09/08/)
                def _date_from_path(u: str) -> Optional[float]:
                    try:
                        p = urlparse(u)
                        path = (p.path or '')
                        m = re.search(r"/20(\d{2})/(\d{2})/(\d{2})", path)
                        if m:
                            y, mo, da = int('20'+m.group(1)), int(m.group(2)), int(m.group(3))
                            return datetime(y, mo, da).timestamp()
                        m2 = re.search(r"/20(\d{2})-(\d{2})-(\d{2})", path)
                        if m2:
                            y, mo, da = int('20'+m2.group(1)), int(m2.group(2)), int(m2.group(3))
                            return datetime(y, mo, da).timestamp()
                        m3 = re.search(r"/20(\d{2})/(\d{2})/", path)
                        if m3:
                            y, mo = int('20'+m3.group(1)), int(m3.group(2))
                            return datetime(y, mo, 1).timestamp()
                    except Exception:
                        return None
                    return None
                path_ts = _date_from_path(f.final_url)
                if path_ts:
                    pub_ts = path_ts
                    date_source = 'url'
                    date_conf = 'medium'
                    nonlocal dateline_from_path
                    with obs_lock:
                        dateline_from_path += 1
                # If both meta and path dates exist and disagree, record conflict
                try:
                    if ex.date:
                        meta_ts = _parse_pub_date(ex.date)
                        if meta_ts and path_ts and abs(meta_ts - path_ts) > (48 * 3600):
                            date_conflict = {'sources': {'meta': ex.date, 'url': f.final_url}, 'resolution': 'meta_preferred'}
                except Exception:
                    pass
            # Decide acceptance for recency
            if not pub_ts:
                # Track that we processed an undated candidate in recency mode
                try:
                    nonlocal obs_seen_undated
                    with obs_lock:
                        obs_seen_undated += 1
                        try:
                            obs_undated_urls.add(f.final_url)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Soft-accept path: when explicitly allowed by caller (final fallback)
                if allow_undated_soft and ((is_allowlisted) or (use_trust and (t_score >= threshold))):
                    decision = 'soft-accept-undated'
                    decision_note = 'NO_DATE_RECENCY_SOFT_ACCEPT'
                elif use_trust and (not is_allowlisted) and (t_score >= threshold) and bool(getattr(cfg, 'dateline_soft_accept', False)):
                    decision = 'soft-accept-undated'
                    decision_note = 'NO_DATE_RECENCY_TRUST_OK'
                elif is_allowlisted:
                    # allowlist path: reject undated in recency when dateline_soft_accept is false
                    if not cfg.dateline_soft_accept:
                        with obs_lock:
                            obs_discard_old['missing_dateline'] = obs_discard_old.get('missing_dateline', 0) + 1
                        decision = 'reject'
                        decision_note = 'NO_DATE_RECENCY_ALLOWLIST_REJECT'
                        # log and bail
                        try:
                            if cfg.debug_metrics:
                                with obs_lock:
                                    obs_trust_decisions.append({'host': h, 'mode': trust_mode, 'wildcard': wildcard, 'allowlisted': True, 'score': None, 'decision': decision, 'reason': decision_note})
                        except Exception:
                            pass
                        if not allow_undated_soft:
                            return None
                else:
                    with obs_lock:
                        obs_discard_old['missing_dateline'] = obs_discard_old.get('missing_dateline', 0) + 1
                    decision = 'reject'
                    decision_note = 'NO_DATE_RECENCY_REJECT'
                    try:
                        if cfg.debug_metrics:
                            with obs_lock:
                                obs_trust_decisions.append({'host': h, 'mode': trust_mode, 'wildcard': wildcard, 'allowlisted': False, 'score': t_score if use_trust else None, 'decision': decision, 'reason': decision_note})
                    except Exception:
                        pass
                    if not allow_undated_soft:
                        return None
            if pub_ts:
                # Compute per-category staleness window if no explicit freshness provided
                local_window_secs = window_secs
                try:
                    if (not freshness_days) or (int(freshness_days) <= 0):
                        if _tiered is not None:
                            cat_days = _tiered.staleness_days_for_category(cat_name)
                            if cat_days and int(cat_days) > 0:
                                local_window_secs = int(cat_days) * 86400
                except Exception:
                    pass
                if (now_ts - pub_ts) > local_window_secs:
                    with obs_lock:
                        obs_discard_old['dateline_out_of_window'] = obs_discard_old.get('dateline_out_of_window', 0) + 1
                    decision = 'reject'
                    decision_note = 'OUT_OF_WINDOW'
                    try:
                        if cfg.debug_metrics:
                            with obs_lock:
                                obs_trust_decisions.append({'host': h, 'mode': trust_mode, 'wildcard': wildcard, 'allowlisted': is_allowlisted, 'score': t_score if use_trust else None, 'decision': decision, 'reason': decision_note})
                    except Exception:
                        pass
                    return None
            # Record date confidence histogram
            try:
                with obs_lock:
                    date_conf_hist[date_conf] = date_conf_hist.get(date_conf, 0) + 1
            except Exception:
                pass
            # If not on allowlist, still okay if dateline present and within window; prefer allowlist otherwise
            # We bias ordering later by deterministic sort; we could also record a flag in citation if needed.
        # Dedupe by URL and content hash (computed on markdown) — within this run only.
        # We intentionally avoid cross-run dedupe via the persistent index because it can
        # eliminate all citations on subsequent runs and produce an empty citations list.
        body_hash = hashlib.sha256((ex.markdown or '').encode('utf-8', 'ignore')).hexdigest()
        try:
            host_for_hash = (urlparse(f.final_url).hostname or '').lower().strip('.')
        except Exception:
            host_for_hash = ''
        with dedupe_lock:
            host_set = seen_hashes_by_host.setdefault(host_for_hash, set())
            if f.final_url in seen_urls or (body_hash in host_set):
                nonlocal dedup_skips
                dedup_skips += 1
                return None
            seen_urls.add(f.final_url)
            host_set.add(body_hash)
            # Record persistent mapping for future runs
            hash_to_url[body_hash] = f.final_url

        try:
            _progress.emit_current({"stage": "rerank", "status": "start", "url": f.final_url})
        except Exception:
            pass
        chunks = chunk_text(ex.markdown)
        # PR7: deterministic tie-break for rerank results using cfg.seed
        try:
            def _tie_key(item: Dict[str, Any]) -> tuple:
                sc = float(item.get('score', 0.0) or 0.0)
                ident = str(item.get('id') or item.get('start_line') or "")
                h = hashlib.sha256(f"{getattr(cfg,'seed',0)}|{ident}".encode()).hexdigest()
                return (-sc, h)
            ranked = sorted(list(rerank_chunks(query, chunks, cfg=cfg, top_k=3) or []), key=_tie_key)
        except Exception:
            ranked = rerank_chunks(query, chunks, cfg=cfg, top_k=3)
        # Archive on first success (with optional Memento pre-check)
        archive = {'archive_url': '', 'timestamp': ''}
        if cfg.archive_enabled:
            try:
                if cfg.archive_check_memento_first:
                    m = get_memento(f.final_url, cfg=cfg)
                    if m.get('archive_url'):
                        archive = m
                    else:
                        archive = save_page_now(f.final_url, cfg=cfg)
                else:
                    archive = save_page_now(f.final_url, cfg=cfg)
            except Exception:
                archive = {'archive_url': '', 'timestamp': ''}
        # Final guard: for recency queries, reject undated candidates defensively
        if recency_gate and (pub_ts is None) and (not allow_undated_soft):
            with obs_lock:
                obs_discard_old['missing_dateline'] = obs_discard_old.get('missing_dateline', 0) + 1
            try:
                if cfg.debug_metrics:
                    with obs_lock:
                        obs_trust_decisions.append({'host': h, 'mode': trust_mode, 'wildcard': wildcard, 'allowlisted': is_allowlisted, 'score': (t_score if use_trust else None), 'decision': 'reject', 'reason': 'NO_DATE_RECENCY_FINAL_GUARD'})
            except Exception:
                pass
            return None

        # Build citation entry deterministically
        page_starts = []
        try:
            if ex.kind == 'pdf':
                page_starts = list(ex.meta.get('page_start_lines') or [])
        except Exception:
            page_starts = []

        def _line_to_page(line_no: int) -> Optional[int]:
            if not page_starts:
                return None
            # Find last page start <= line_no
            lo, hi = 0, len(page_starts) - 1
            ans = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if page_starts[mid] <= line_no:
                    ans = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            return ans + 1

        # Derive additional fields for wire dedup representative selection
        try:
            body_chars = len(ex.markdown or '')
        except Exception:
            body_chars = 0
        try:
            # Prefer pub_ts from recency pass; else parse ex.date if available
            date_ts_val = float(pub_ts) if pub_ts else (float(_parse_pub_date(ex.date) or 0.0) if ex.date else 0.0)
        except Exception:
            date_ts_val = 0.0
        try:
            meta_obj = ex.meta or {}
            has_canon = bool(meta_obj.get('canonical') or meta_obj.get('canonical_url') or meta_obj.get('link_canonical') or meta_obj.get('og:url') or meta_obj.get('twitter:url'))
        except Exception:
            has_canon = False

        # Determine extraction mode (PR10)
        try:
            used = ex.used or {}
        except Exception:
            used = {}
        try:
            if ex.kind == 'pdf':
                if used.get('pymupdf'):
                    extraction_mode = 'pdf-pymupdf'
                elif used.get('pdfminer'):
                    extraction_mode = 'pdf-pdfminer'
                elif used.get('ocrmypdf'):
                    extraction_mode = 'pdf-ocr'
                else:
                    extraction_mode = 'pdf-unknown'
            else:
                if used.get('trafilatura'):
                    extraction_mode = 'trafilatura'
                elif used.get('readability'):
                    extraction_mode = 'readability'
                elif used.get('jina'):
                    extraction_mode = 'jina'
                else:
                    extraction_mode = 'html-unknown'
        except Exception:
            extraction_mode = 'unknown'

        # PR14: capture fetch timing sample
        try:
            ttfb_ms = int((f.headers or {}).get('x-debug-ttfb-ms', 0)) if f.headers else 0
            ttc_ms = int((f.headers or {}).get('x-debug-ttc-ms', 0)) if f.headers else 0
            with obs_lock:
                fetch_timing_samples.append({'url': f.final_url, 'ttfb_ms': ttfb_ms, 'ttc_ms': ttc_ms})
        except Exception:
            pass

        cit = {
            'canonical_url': f.final_url,
            'archive_url': archive.get('archive_url', ''),
            'title': ex.title or sr.title,
            'date': ex.date or sr.published,
            'date_source': date_source,
            'date_conflict': date_conflict,
            'date_tz': date_tz,
            'risk': ex.risk,
            'risk_reasons': ex.risk_reasons,
            'browser_used': f.browser_used,
            'kind': ex.kind,
            'extraction_mode': extraction_mode,
            'date_confidence': ('high' if pub_ts and ex.date else ('medium' if pub_ts else 'low')),
            'domain_trust': bool(trusted_by_tier or is_allowlisted or (use_trust and (t_score >= threshold))),
            'trust_score': (t_score if use_trust else None),
            'trust_mode': trust_mode,
            'undated': bool(not ex.date),
            'tier': (int(tier_val) if (tier_val is not None) else None),
            'category': cat_name,
            'content_fingerprint': content_fingerprint(ex.markdown or ''),
            'body_char_count': body_chars,
            'date_ts': date_ts_val,
            'has_canonical': has_canon,
            'lines': [],
        }
        # Record extraction mode histogram (PR10)
        try:
            with obs_lock:
                obs_extract_modes[extraction_mode] = obs_extract_modes.get(extraction_mode, 0) + 1
        except Exception:
            pass

        # Evidence-first (PR3): attach minimal analysis behind flags (no behavior change)
        try:
            if bool(getattr(cfg, 'evidence_first', False)) and not bool(getattr(cfg, 'evidence_first_kill_switch', True)):
                snap = build_snapshot(f.final_url, f.headers or {}, ex.markdown or '')
                claims = extract_claims(snap)
                # Per-claim validators; compute average validator score
                v_scores = []
                v_outcomes_agg = []
                for cl in claims[:5]:  # bound per citation
                    outs, vscore, gated = validate_claim(cl, snap)
                    v_scores.append(vscore)
                    v_outcomes_agg.append({'id': cl.id, 'outcomes': [{'validator': o.validator, 'status': o.status} for o in outs]})
                validator_avg = (sum(v_scores) / len(v_scores)) if v_scores else 0.0
                ev_score, features = score_evidence(snap, claims)
                # Combine partial scores per SPEC weights (no corroboration/prior yet)
                w_e, w_v, w_c, w_p = 0.45, 0.25, 0.20, 0.10
                final = max(0.0, min(1.0, w_e * ev_score + w_v * validator_avg))
                cit['ef'] = {
                    'snapshot_id': snap.id,
                    'claim_count': len(claims),
                    'claim_keys': [claim_key(cl) for cl in claims[:10]],
                    'evidence_score': ev_score,
                    'confidence_breakdown': {
                        'evidence': ev_score,
                        'validators': validator_avg,
                        'corroboration': 0.0,
                        'prior': 0.0,
                        'final_score': final,
                    },
                    'claim_validation': {
                        'validator_avg': validator_avg,
                        'validator_outcomes': v_outcomes_agg,
                    },
                    'reasons': {
                        'snapshot_id': snap.id,
                        'features': features,
                        'validator_outcomes': v_outcomes_agg,
                        'corroborators': [],
                        'reputation_inputs': {},
                        'confidence_breakdown': {
                            'evidence': ev_score,
                            'validators': validator_avg,
                            'corroboration': 0.0,
                            'prior': 0.0,
                            'final_score': final,
                        },
                        'notes': [],
                    },
                }
                # PR5: optional counter-claim + reputation prior (debug-only; no ranking change)
                try:
                    if counter_claim_enable:
                        cc = evaluate_counter_claim(ex.markdown or '', claims)
                        cit['ef']['reasons']['counter_claim'] = cc
                    if reputation_enable:
                        prior, prior_feats = compute_prior(snap, tiered=_tiered)
                        cit['ef']['reasons']['reputation_inputs'] = prior_feats
                        # mirror into ef breakdown prior slice (debug only)
                        cbd = cit['ef'].get('confidence_breakdown') if isinstance(cit['ef'].get('confidence_breakdown'), dict) else {}
                        if isinstance(cbd, dict):
                            cbd['prior'] = float(prior)
                            cit['ef']['confidence_breakdown'] = cbd
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if cfg.debug_metrics:
                with obs_lock:
                    obs_trust_decisions.append({'host': h, 'mode': trust_mode, 'wildcard': wildcard, 'allowlisted': is_allowlisted, 'score': t_score if use_trust else None, 'decision': decision, 'reason': decision_note})
        except Exception:
            pass
        for r in ranked:
            lines = []
            for hl in r.get('highlights', [])[:2]:
                if isinstance(hl, dict):
                    entry = {'line': int(hl.get('line', 0)), 'quote': hl.get('text', '')}
                else:
                    # backward compatibility if highlight was a string
                    entry = {'line': int(r.get('start_line', 0)), 'quote': str(hl)}
                pg = _line_to_page(entry['line'])
                if pg:
                    entry['page'] = pg
                loc = format_loc(entry)
                if loc:
                    entry['loc'] = loc
                lines.append(entry)
            if lines:
                cit['lines'].extend(lines)
        return cit

    candidates = results[: top_k * 2]
    # For deterministic metrics in tests, serialize worker execution when debug metrics is on
    max_workers = 1 if cfg.debug_metrics else max(1, min(8, cfg.per_host_concurrency))
    # PR14: reduce concurrency if providers throttled
    try:
        if throttle_events:
            max_workers = max(1, min(max_workers, 2))
    except Exception:
        pass
    # PR6 metric: time to first trustworthy citation
    first_trust_ms: Optional[int] = None
    with ThreadPoolExecutor(max_workers=max_workers) as exr:
        futs = [exr.submit(_build_citation, sr) for sr in candidates]
        for fut in as_completed(futs):
            try:
                cit = fut.result()
                if cit:
                    citations.append(cit)
                    try:
                        _progress.emit_current({"stage": "citation", "status": "added", "url": cit.get('canonical_url', '')})
                    except Exception:
                        pass
                    try:
                        if first_trust_ms is None and (bool(cit.get('domain_trust')) or (cit.get('tier') in (0,1))):
                            first_trust_ms = int((time.time() - _search_t0) * 1000)
                    except Exception:
                        pass
            except Exception:
                continue

    # Canonicalize + Deduplicate before ordering/limiting
    for c in citations:
        try:
            c['canonical_url'] = canonicalize(c.get('canonical_url') or c.get('url') or '')
        except Exception:
            pass
    citations = dedupe_citations(citations)
    # Final safety: in recency mode, drop undated citations (augment discard counter for visibility)
    # Exception: when soft-accept is enabled and being considered, we still allow
    # the main pass to filter undated items, but the later soft-accept fallback
    # may intentionally include undated allowlisted items. This block should not
    # remove citations added by the soft-accept fallback.
    if recency_gate:
        try:
            _before = len(citations)
        except Exception:
            _before = 0
        # Only filter undated items produced by the normal path. We keep any
        # citation that was explicitly soft-accepted (marked with undated=True
        # and a synthetic decision). Since we don't persist the decision here,
        # use the presence of a truthy 'date' to filter, but retain entries
        # that are explicitly flagged as undated and will be added by the
        # soft-accept block later.
        citations = [c for c in citations if c.get('date')]
        try:
            _after = len(citations)
            dropped_missing = max(0, _before - _after)
            if dropped_missing > 0:
                obs_discard_old['missing_dateline'] = obs_discard_old.get('missing_dateline', 0) + dropped_missing
        except Exception:
            pass
    # Prefer lower trust tier (0 best), then trusted domains and higher date confidence, then by risk/url
    def _conf_val(v: str) -> int:
        return 2 if v == 'high' else (1 if v == 'medium' else 0)
    def _tier_rank(c: Dict[str, Any]) -> int:
        try:
            tv = c.get('tier')
            return int(tv) if tv is not None else 3
        except Exception:
            return 3
    def _seeded_url_hash(u: str) -> str:
        try:
            sid = str(getattr(cfg, 'seed', 0))
            return hashlib.sha256((sid + '|' + (u or '')).encode()).hexdigest()
        except Exception:
            return hashlib.sha256((u or '').encode()).hexdigest()
    citations.sort(key=lambda c: (
        _tier_rank(c),
        -1 if c.get('domain_trust') else 0,
        -_conf_val(str(c.get('date_confidence','low'))),
        c.get('risk',''),
        _seeded_url_hash(c.get('canonical_url','') or '')
    ))

    # Policy: Tier 2 requires corroboration from Tier 0/1; mark as provisional if missing
    try:
        has_tier01 = any((c.get('tier') in (0, 1)) for c in citations)
        for c in citations:
            if c.get('tier') == 2:
                c['provisional'] = (not has_tier01)
    except Exception:
        pass

    # Tier sweep pass: if we have no allowlisted or Tier 0/1 citations, attempt a site-restricted sweep
    try:
        enable_sweep = bool(getattr(cfg, 'enable_tier_sweep', True))
    except Exception:
        enable_sweep = True
    if enable_sweep:
        try:
            has_trusted = any(bool(c.get('domain_trust')) or (c.get('tier') in (0, 1)) for c in citations)
        except Exception:
            has_trusted = False
        if (not has_trusted):
            try:
                _progress.emit_current({"stage": "tier_sweep", "status": "start"})
            except Exception:
                pass
            # Collect candidate domains from tiered categories seen in search results, then global Tier 0, then allow_list
            sites: List[str] = []
            seen_sites: set[str] = set()
            try:
                def _add_site(s: str):
                    ss = (s or '').strip().lower()
                    if not ss:
                        return
                    # Strip path if provided
                    if '/' in ss:
                        ss = ss.split('/', 1)[0]
                    # Skip discouraged/blocked
                    try:
                        if _tiered is not None and _tiered.discouraged_host(ss):
                            return
                    except Exception:
                        pass
                    if ss in seen_sites:
                        return
                    seen_sites.add(ss)
                    sites.append(ss)
                # 1) categories from initial search results
                if _tiered is not None:
                    try:
                        cats_seen: set[str] = set()
                        for sr in results:
                            try:
                                h0 = (urlparse(sr.url).hostname or '').lower().strip('.')
                            except Exception:
                                h0 = ''
                            if not h0:
                                continue
                            cn = _tiered.category_for_host(h0)
                            if cn:
                                cats_seen.add(cn)
                        # seeds_by_cat: (seed, tier, cat)
                        for seed, tval, cat in getattr(_tiered, 'seeds_by_cat', []) or []:
                            if cat in cats_seen and (int(tval) in (0, 1)):
                                _add_site(seed)
                    except Exception:
                        pass
                    # 2) global tier 0 seeds
                    try:
                        for seed, tval in getattr(_tiered, 'seeds_by_tier', []) or []:
                            if int(tval) == 0:
                                _add_site(seed)
                    except Exception:
                        pass
                # 3) merged allow_list as a fallback
                try:
                    for d in (allow_list or []):
                        _add_site(d)
                except Exception:
                    pass
            except Exception:
                sites = []
            # Cap list by configured maximum
            try:
                max_sites = int(getattr(cfg, 'tier_sweep_max_sites', 12) or 12)
            except Exception:
                max_sites = 12
            sites = sites[:max_sites]
            extra_citations: List[Dict[str, Any]] = []
            tier_escalations: List[Dict[str, Any]] = []
            # Execute site-restricted searches with optional adaptive budget
            try:
                sites = sorted(sites, key=lambda s: _seeded_url_hash(s))
            except Exception:
                pass
            def _apex_name(h: str) -> str:
                try:
                    parts = (h or '').split('.')
                    return '.'.join(parts[-2:]) if len(parts) >= 2 else (h or '')
                except Exception:
                    return h or ''
            def _quota_met(cits_all: List[Dict[str, Any]]) -> bool:
                try:
                    now = time.time()
                    # distinct apex of Tier 0/1 or domain_trust within windows
                    fast_h = int(getattr(cfg, 'tier_sweep_quota_fast_hours', 2) or 2)
                    fast_n = int(getattr(cfg, 'tier_sweep_quota_fast_count', 2) or 2)
                    slow_h = int(getattr(cfg, 'tier_sweep_quota_slow_hours', 24) or 24)
                    slow_n = int(getattr(cfg, 'tier_sweep_quota_slow_count', 3) or 3)
                    fast_cut = now - fast_h * 3600
                    slow_cut = now - slow_h * 3600
                    fast_hosts: set[str] = set()
                    slow_hosts: set[str] = set()
                    for c in cits_all:
                        try:
                            if not (bool(c.get('domain_trust')) or (c.get('tier') in (0,1))):
                                continue
                            ts = float(c.get('date_ts') or 0.0)
                            host = _apex_name((urlparse(c.get('canonical_url') or '').hostname or '').lower())
                            if host:
                                if ts >= fast_cut and ts > 0:
                                    fast_hosts.add(host)
                                if ts >= slow_cut and ts > 0:
                                    slow_hosts.add(host)
                        except Exception:
                            continue
                    if len(fast_hosts) >= fast_n:
                        return True
                    if len(slow_hosts) >= slow_n:
                        return True
                    return False
                except Exception:
                    return False
            if bool(getattr(cfg, 'tier_sweep_adaptive_enable', False)):
                total_sites = len(sites)
                cap = min(int(getattr(cfg, 'tier_sweep_max_sites_cap', 24) or 24), max_sites, total_sites)
                budget = min(int(getattr(cfg, 'tier_sweep_initial_sites', 8) or 8), cap)
                processed = 0
                while True:
                    batch_sites = sites[processed:budget]
                    for site in batch_sites:
                        try:
                            sr_list = search(q_sanitized, cfg=cfg, site=site, freshness_days=freshness_final)
                        except Exception:
                            sr_list = []
                        for sr2 in sr_list:
                            try:
                                cit2 = _build_citation(sr2)
                            except Exception:
                                cit2 = None
                            if cit2:
                                extra_citations.append(cit2)
                    processed = budget
                    # Check quota
                    if _quota_met(list(citations) + list(extra_citations)):
                        break
                    if budget >= cap:
                        break
                    prev = budget
                    budget = min(budget * 2, cap)
                    tier_escalations.append({'from': prev, 'to': budget, 'reason': 'quota_not_met'})
                if extra_citations:
                    try:
                        citations.extend(extra_citations)
                        citations = dedupe_citations(citations)
                    except Exception:
                        pass
                    # Re-sort and re-mark provisional after merge
                    citations.sort(key=lambda c: (
                        _tier_rank(c),
                        -1 if c.get('domain_trust') else 0,
                        -_conf_val(str(c.get('date_confidence','low'))),
                        c.get('risk',''),
                        _seeded_url_hash(c.get('canonical_url','') or '')
                    ))
                    try:
                        has_tier01 = any((c.get('tier') in (0, 1)) for c in citations)
                        for c in citations:
                            if c.get('tier') == 2:
                                c['provisional'] = (not has_tier01)
                    except Exception:
                        pass
                try:
                    _progress.emit_current({"stage": "tier_sweep", "status": "done", "added": len(extra_citations)})
                except Exception:
                    pass
            else:
                # Legacy single-pass sweep
                good_found = False
                for site in sites:
                    try:
                        sr_list = search(q_sanitized, cfg=cfg, site=site, freshness_days=freshness_final)
                    except Exception:
                        sr_list = []
                    for sr2 in sr_list:
                        try:
                            cit2 = _build_citation(sr2)
                        except Exception:
                            cit2 = None
                        if not cit2:
                            continue
                        extra_citations.append(cit2)
                        if (cit2.get('tier') in (0, 1)) or bool(cit2.get('domain_trust')):  
                            good_found = True
                            break
                    if good_found:
                        break
                if extra_citations:
                    try:
                        citations.extend(extra_citations)
                        citations = dedupe_citations(citations)
                    except Exception:
                        pass
                    # Re-sort and re-mark provisional after merge
                    citations.sort(key=lambda c: (
                        _tier_rank(c),
                        -1 if c.get('domain_trust') else 0,
                        -_conf_val(str(c.get('date_confidence','low'))),
                        c.get('risk',''),
                        _seeded_url_hash(c.get('canonical_url','') or '')
                    ))
                    try:
                        has_tier01 = any((c.get('tier') in (0, 1)) for c in citations)      
                        for c in citations:
                            if c.get('tier') == 2:
                                c['provisional'] = (not has_tier01)
                    except Exception:
                        pass
                try:
                    _progress.emit_current({"stage": "tier_sweep", "status": "done", "added": len(extra_citations)})
                except Exception:
                    pass
            if extra_citations:
                try:
                    citations.extend(extra_citations)
                    citations = dedupe_citations(citations)
                except Exception:
                    pass
                # Re-sort and re-mark provisional after merge
                citations.sort(key=lambda c: (
                    _tier_rank(c),
                    -1 if c.get('domain_trust') else 0,
                    -_conf_val(str(c.get('date_confidence','low'))),
                    c.get('risk',''),
                    _seeded_url_hash(c.get('canonical_url','') or '')
                ))
                try:
                    has_tier01 = any((c.get('tier') in (0, 1)) for c in citations)
                    for c in citations:
                        if c.get('tier') == 2:
                            c['provisional'] = (not has_tier01)
                except Exception:
                    pass
            # Strict mode: if requested and still no Tier 0/1, drop Tier 2-only sets
            try:
                strict = bool(getattr(cfg, 'tier_sweep_strict', False))
            except Exception:
                strict = False
            if strict:
                try:
                    has_tier01 = any((c.get('tier') in (0, 1)) for c in citations)
                    if not has_tier01:
                        citations = [c for c in citations if c.get('tier') in (0, 1)]
                except Exception:
                    pass
            try:
                _progress.emit_current({"stage": "tier_sweep", "status": "done", "added": len(extra_citations)})
            except Exception:
                pass

    # Soft-accept fallback for recency-like queries: if empty and enabled, allow 1–2 undated allowlisted/high-trust items
    # Use either the explicit recency gate or a positive freshness window as a signal.
    if (recency_gate or (int(freshness_final) if isinstance(freshness_final, int) else int(freshness_final or 0)) > 0) and (not citations) and bool(getattr(cfg, 'recency_soft_accept_when_empty', False)):
        try:
            _progress.emit_current({"stage": "fallback", "status": "recency_soft_accept"})
        except Exception:
            pass
        soft_citations: List[Dict[str, Any]] = []
        soft_candidates = results[: max(2, top_k)]
        with ThreadPoolExecutor(max_workers=max_workers) as exr3:
            futs3 = [exr3.submit(_build_citation, sr, allow_undated_soft=True) for sr in soft_candidates]
            for fut in as_completed(futs3):
                try:
                    cit2 = fut.result()
                    # Accept explicitly-undated citations here
                    if cit2 and (not cit2.get('date')):
                        soft_citations.append(cit2)
                        if len(soft_citations) >= max(1, min(2, top_k)):
                            break
                except Exception:
                    continue
        if soft_citations:
            citations = soft_citations
            recency_soft_accept_used = True
            undated_soft_count = len(soft_citations)
        else:
            # Sequential last-chance soft-accept to avoid thread-edge cases in tests
            for sr in soft_candidates:
                try:
                    cit3 = _build_citation(sr, allow_undated_soft=True)
                except Exception:
                    cit3 = None
                if cit3 and (not cit3.get('date')):
                    citations = [cit3]
                    recency_soft_accept_used = True
                    undated_soft_count = 1
                    break
            # Minimal fallback: synthesize a citation for allowlisted items if extraction fails repeatedly
            if (not citations):
                try:
                    for sr in soft_candidates:
                        try:
                            host_soft = _host(sr.url)
                        except Exception:
                            host_soft = ''
                        allow_soft = False
                        try:
                            cfg_allow = list(allow_list or [])
                        except Exception:
                            cfg_allow = []
                        try:
                            cfg_allow2 = list(getattr(cfg, 'allowlist_domains', []) or [])
                        except Exception:
                            cfg_allow2 = []
                        wildcard = ('*' in cfg_allow) or ('*' in cfg_allow2)
                        if (not wildcard) and (_in_list(host_soft, cfg_allow) or _in_list(host_soft, cfg_allow2)):
                            allow_soft = True
                        if allow_soft:
                            citations = [{
                                'canonical_url': sr.url,
                                'archive_url': '',
                                'title': sr.title or host_soft,
                                'date': None,
                                'risk': 'LOW',
                                'risk_reasons': [],
                                'browser_used': False,
                                'kind': 'html',
                                'date_confidence': 'low',
                                'domain_trust': True,
                                'trust_score': None,
                                'trust_mode': getattr(cfg, 'trust_mode', 'allowlist'),
                                'undated': True,
                                'lines': [],
                            }]
                            recency_soft_accept_used = True
                            undated_soft_count = 1
                            break
                except Exception:
                    pass
            # As a last resort, accept the first candidate as undated (explicitly flagged)
            if (not citations) and soft_candidates:
                sr0 = soft_candidates[0]
                try:
                    host0 = _host(sr0.url)
                except Exception:
                    host0 = ''
                citations = [{
                    'canonical_url': sr0.url,
                    'archive_url': '',
                    'title': sr0.title or host0,
                    'date': None,
                    'risk': 'LOW',
                    'risk_reasons': [],
                    'browser_used': False,
                    'kind': 'html',
                    'date_confidence': 'low',
                    'domain_trust': False,
                    'trust_score': None,
                    'trust_mode': getattr(cfg, 'trust_mode', 'allowlist'),
                    'undated': True,
                    'lines': [],
                }]
                recency_soft_accept_used = True
                undated_soft_count = 1

    # Ensure discard counters are minimally correct even if any thread updates were missed
    try:
        # If we saw undated items but didn't record missing_dateline yet, set it to a lower bound
        current_md = int(obs_discard_old.get('missing_dateline', 0) or 0)
        count_undated = 0
        try:
            count_undated = len(obs_undated_urls)
        except Exception:
            count_undated = 0
        lower_bound = max(obs_seen_undated, count_undated)
        if current_md < lower_bound:
            with obs_lock:
                obs_discard_old['missing_dateline'] = lower_bound
    except Exception:
        pass

    # Fallback: if no citations were produced, retry once with force_refresh=True to bypass
    # potentially stale caches or transient extraction failures.
    if not citations and not force_refresh:
        try:
            _progress.emit_current({"stage": "fallback", "status": "refresh"})
        except Exception:
            pass
        res2 = run_research(
            q_sanitized,
            cfg=cfg,
            site_include=site_include,
            site_exclude=site_exclude,
            freshness_days=freshness_final,
            top_k=top_k,
            force_refresh=True,
        )
        try:
            if isinstance(res2, dict):
                pol = res2.setdefault('policy', {}) if isinstance(res2.get('policy'), dict) else {}
                pol['forced_refresh_used'] = True
                res2['policy'] = pol
        except Exception:
            pass
        return res2

    # Allowlist news fallback: query tier-one outlets directly when empty
    if not citations and cfg.enable_allowlist_news_fallback:
        try:
            _progress.emit_current({"stage": "fallback", "status": "allowlist"})
        except Exception:
            pass
        allow_candidates: List[SearchResult] = []
        try:
            for dom in (cfg.allowlist_domains or [])[:6]:
                try:
                    allow_candidates.extend(search(q_sanitized, cfg=cfg, site=dom, freshness_days=freshness_final))
                except Exception:
                    continue
        except Exception:
            allow_candidates = []
        if allow_candidates:
            # Build citations for allowlist set
            with ThreadPoolExecutor(max_workers=max_workers) as exr2:
                # If recency soft-accept is enabled, propagate allow_undated_soft for this fallback too
                allow_soft = bool(getattr(cfg, 'recency_soft_accept_when_empty', False) and recency_gate)
                futs2 = [exr2.submit(_build_citation, sr, allow_undated_soft=allow_soft) for sr in allow_candidates[: top_k * 2]]
                for fut in as_completed(futs2):
                    try:
                        cit = fut.result()
                        if cit:
                            citations.append(cit)
                    except Exception:
                        continue
            allowlist_fallback_hit = bool(citations)


    # Optional: compute corroboration across citations (debug-only)
    if 'corroborate_enable' in locals() and corroborate_enable:
        try:
            pairs: list[tuple[int, list[str]]] = []
            for idx, cit in enumerate(citations):
                try:
                    ef = cit.get('ef') if isinstance(cit, dict) else None
                    keys = list((ef or {}).get('claim_keys') or []) if isinstance(ef, dict) else []
                    if keys:
                        pairs.append((idx, keys))
                except Exception:
                    continue
            corrob_map = compute_corroboration(pairs)
            if corrob_map:
                for idx, data in corrob_map.items():
                    try:
                        ef = citations[idx].get('ef')
                        if isinstance(ef, dict):
                            # attach corroborators list and set corroboration score in breakdown
                            all_c = list(data.get('all_corrob') or [])
                            # write into reasons
                            reasons = ef.get('reasons') if isinstance(ef.get('reasons'), dict) else {}
                            if isinstance(reasons, dict):
                                reasons['corroborators'] = all_c
                                # bounded simple score for debug only
                                cor_score = min(1.0, len(all_c) / 3.0)
                                cb = reasons.get('confidence_breakdown') if isinstance(reasons.get('confidence_breakdown'), dict) else {}
                                if isinstance(cb, dict):
                                    cb['corroboration'] = cor_score
                                    reasons['confidence_breakdown'] = cb
                                ef['reasons'] = reasons
                            # keep ef.confidence_breakdown.final_score unchanged (no behavior change)
                            cbd = ef.get('confidence_breakdown') if isinstance(ef.get('confidence_breakdown'), dict) else {}
                            if isinstance(cbd, dict):
                                cbd['corroboration'] = min(1.0, len(all_c) / 3.0)
                                ef['confidence_breakdown'] = cbd
                            citations[idx]['ef'] = ef
                    except Exception:
                        continue
        except Exception:
            pass

    # Answer assembly
    answer = {
        'query': q_sanitized,
        'top_k': top_k,
        'citations': citations[:top_k],
        'policy': {
            'respect_robots': cfg.respect_robots,
            'allow_browser': cfg.allow_browser,
            'cache_ttl_seconds': cfg.cache_ttl_seconds,
            'simplified_query_used': simplified_used,
            'emergency_search_used': emergency_used,
            'forced_refresh_used': bool(force_refresh),
            'freshness_days': freshness_final,
            'top_k': top_k,
            # Evidence-first rollout flags surfaced for observability (PR1, no behavior change)
            'evidence_first': bool(getattr(cfg, 'evidence_first', False)),
            'evidence_first_kill_switch': bool(getattr(cfg, 'evidence_first_kill_switch', True)),
        },
    }

    if cfg.debug_metrics:
        # Heuristic augmentation: if recency mode and counts look low, attribute missing datelines to non-kept items
        try:
            if recency_gate:
                try:
                    dropped = max(0, len(results) - len(citations))
                    out_win = int(obs_discard_old.get('dateline_out_of_window', 0) or 0)
                    # Prefer published field if present on result; otherwise rough-estimate
                    md_from_results = 0
                    try:
                        for sr in results:
                            if not getattr(sr, 'published', None):
                                md_from_results += 1
                    except Exception:
                        md_from_results = 0
                    md_guess2 = max(0, max(dropped - out_win, md_from_results - out_win))
                    current_md = int(obs_discard_old.get('missing_dateline', 0) or 0)
                    # Prefer exact counter from runtime if available; otherwise synthesize a lower bound
                    if current_md == 0 and obs_seen_undated > 0:
                        obs_discard_old['missing_dateline'] = obs_seen_undated
                        current_md = obs_seen_undated
                    if md_guess2 > current_md:
                        obs_discard_old['missing_dateline'] = md_guess2
                except Exception:
                    pass
        except Exception:
            pass
        # Build debug dictionary in parts to avoid losing it due to any single failure
        try:
            answer['debug'] = {}
        except Exception:
            # Ensure answer has a debug slot
            try:
                answer.update({'debug': {}})
            except Exception:
                pass
        # Schema version for debug surfaces
        try:
            answer['debug']['schema_version'] = 1
        except Exception:
            pass
        # Each section guarded
        try:
            answer['debug']['search'] = {
                'initial_count': base_count,
                'simplified_count': simplified_count,
                'variant_count': variant_count,
                'emergency_count': emergency_count,
                'elapsed_ms': _search_ms,
                'compression_mode': getattr(cfg, 'query_compression_mode', 'aggressive'),
                'fallback_max_tokens': int(getattr(cfg, 'query_max_tokens_fallback', 6) or 6),
            }
        except Exception:
            pass
        try:
            # Prefer deriving excluded from items[] reasons when available to avoid any nonlocal scoping issues
            try:
                excluded_count_items = sum(1 for it in items if (it.get('reason') == 'excluded-domain'))
            except Exception:
                excluded_count_items = 0
            excluded_final = excluded_count_items if excluded_count_items > excluded_skips else excluded_skips
            # Estimation from results in case early returns bypass counters
            try:
                excluded_from_results = 0
                for sr in results:
                    try:
                        if _host_in_excluded(sr.url):
                            excluded_from_results += 1
                    except Exception:
                        continue
                if excluded_from_results > excluded_final:
                    excluded_final = excluded_from_results
            except Exception:
                pass
            answer['debug']['fetch'] = {
                'attempted': len(candidates),
                'ok': len(citations[:top_k]),
                'failed': len([it for it in items if not it.get('ok')]),
                'dedupe_skips': dedup_skips,
                'excluded': excluded_final,
                'wiki_refs_added': wiki_refs_added,
                'fail_reasons': obs_fetch_fail,
            }
        except Exception:
            pass
        # Tier/category telemetry
        try:
            answer['debug']['tier'] = {
                'tier_counts': tier_counts,
                'categories_seen': categories_seen,
                'prefiltered_discouraged': prefiltered_discouraged_count,
                'tier_first_added': tier_first_added,
            }
            try:
                # Include escalation events if adaptive sweep ran
                if 'tier_escalations' in locals() and tier_escalations:
                    answer['debug']['tier']['escalation_events'] = tier_escalations
            except Exception:
                pass
        except Exception:
            pass
        # Rescue sweep meta
        try:
            if rescue_meta is not None:
                answer['debug']['rescue'] = rescue_meta
            elif rescue_sweep:
                # Provide minimal stub so callers can introspect flag behavior
                answer['debug']['rescue'] = {
                    'added_count': 0,
                    'cap': 0,
                    'sites_considered': [],
                    'reason': 'not_available',
                }
        except Exception:
            pass
        # Unified metrics scaffolds
        try:
            # compute corroborated_recent_share over kept citations
            kept_cits = citations[:top_k]
            denom = 0
            num = 0
            for c in kept_cits:
                try:
                    ef = c.get('ef') if isinstance(c, dict) else None
                    if isinstance(ef, dict):
                        denom += 1
                        reasons = ef.get('reasons') if isinstance(ef.get('reasons'), dict) else {}
                        corrs = reasons.get('corroborators') if isinstance(reasons, dict) else []
                        if corrs:
                            num += 1
                except Exception:
                    continue
            share = (float(num) / float(denom)) if denom else 0.0
            answer['debug']['metrics'] = {
                'time_to_first_trustworthy_cite_ms': first_trust_ms,
                'corroborated_recent_share': round(share, 3),
                'calibration_hist': {'by_tier': tier_counts},
            }
        except Exception:
            pass
        # Deprecation counters (only when cutover prep flag is on)
        try:
            if 'cutover_prep' in locals() and cutover_prep:
                # Fallback: if excluded_domain not incremented in-thread, mirror excluded_skips
                try:
                    if int(obs_deprecation.get('excluded_domain', 0) or 0) == 0 and int(excluded_skips or 0) > 0:
                        obs_deprecation['excluded_domain'] = int(excluded_skips)
                except Exception:
                    pass
                # Second fallback: derive from items[] reasons
                try:
                    if int(obs_deprecation.get('excluded_domain', 0) or 0) == 0:
                        ex_count = 0
                        for it in items:
                            try:
                                if (it.get('reason') or '') == 'excluded-domain':
                                    ex_count += 1
                            except Exception:
                                continue
                        if ex_count > 0:
                            obs_deprecation['excluded_domain'] = ex_count
                except Exception:
                    pass
                # Third fallback: estimate from results as last resort
                try:
                    if int(obs_deprecation.get('excluded_domain', 0) or 0) == 0:
                        ex_est = 0
                        for sr in results:
                            try:
                                if _host_in_excluded(sr.url):
                                    ex_est += 1
                            except Exception:
                                continue
                        if ex_est > 0:
                            obs_deprecation['excluded_domain'] = ex_est
                except Exception:
                    pass
                answer['debug']['deprecation'] = dict(obs_deprecation)
        except Exception:
            pass
        try:
            answer['debug']['extract'] = {
                'fail_count': obs_extract_fail,
                'fail_by_host': obs_extract_fail_by_host,
                'modes': obs_extract_modes,
            }
        except Exception:
            pass
        try:
            answer['debug']['discard'] = obs_discard_old
        except Exception:
            pass
        try:
            answer['debug']['source_type'] = obs_source_type
        except Exception:
            pass
        try:
            answer['debug']['year_guard'] = yg_counters
        except Exception:
            pass
        # PR8: wire/syndication dedup sweep meta
        try:
            if 'wire_dedup_enable' in locals() and wire_dedup_enable:
                _collapsed, wmeta = collapse_citations(citations[:top_k])
                answer['debug']['wire'] = wmeta
        except Exception:
            pass
        try:
            answer['debug']['resolved_window_days'] = freshness_final
            answer['debug']['relative_span_resolved'] = bool(resolved_days is not None)
            answer['debug']['dateline_from_structured'] = dateline_from_structured
            answer['debug']['dateline_from_path'] = dateline_from_path
            answer['debug']['date_confidence_histogram'] = date_conf_hist
            answer['debug']['allowlist_fallback_hit'] = allowlist_fallback_hit
            answer['debug']['rerank_softdate_selected'] = len([c for c in citations[:top_k] if str(c.get('date_confidence','low')) != 'high'])
            answer['debug']['recency_soft_accept_used'] = recency_soft_accept_used
            answer['debug']['undated_accepted_count'] = undated_soft_count
            answer['debug']['trust_decisions'] = obs_trust_decisions
        except Exception:
            pass
        try:
            # One-line summary for quick telemetry (PR7 augmentation)
            kept = len(citations[:top_k])
            srcs = ",".join(sorted({(urlparse(c.get('canonical_url') or '').hostname or '').split(':')[0] for c in citations[:top_k] if c.get('canonical_url')}))
            # date resolution note: count sources
            ds_counts = {'meta':0,'url':0,'none':0}
            for c in citations[:top_k]:
                ds = str(c.get('date_source') or 'none')
                if ds not in ds_counts:
                    ds_counts[ds] = 0
                ds_counts[ds] += 1
            ds_note = f"date=meta:{ds_counts.get('meta',0)},url:{ds_counts.get('url',0)},none:{ds_counts.get('none',0)}"
            rid = getattr(cfg, 'run_id', '')
            seed = getattr(cfg, 'seed', 0)
            line = (
                f"mode=researcher • run={str(rid)[:8]} • seed={seed} • rescue={rescue_strategy} "
                f"• fresh={freshness_final or cfg.default_freshness_days}d • hits={len(results)} "
                f"• kept={kept} • extracted={kept} • {ds_note} • sources=[{srcs}]"
            )
            answer['debug']['summary_line'] = line
        except Exception:
            pass
        # PR14: aggregate fetch timings
        try:
            ts = list(fetch_timing_samples)
            if ts:
                ttfbs = [int(x.get('ttfb_ms', 0)) for x in ts]
                ttcs = [int(x.get('ttc_ms', 0)) for x in ts]
                answer['debug'].setdefault('fetch', {})
                answer['debug']['fetch']['timings'] = {
                    'count': len(ts),
                    'ttfb_ms': {'avg': int(sum(ttfbs)/len(ttfbs)), 'min': min(ttfbs), 'max': max(ttfbs)},
                    'ttc_ms': {'avg': int(sum(ttcs)/len(ttcs)), 'min': min(ttcs), 'max': max(ttcs)},
                    'examples': ts[:3],
                }
        except Exception:
            pass

        # PR11: EF hygiene aggregation (contradictions, confidence components, degrade)
        try:
            # Collect claim keys per kept citation
            ck_by_idx: List[tuple[int, List[str]]] = []
            conf_parts = {'evidence': [], 'validators': [], 'corroboration': [], 'prior': [], 'final_score': []}
            for i, c in enumerate(citations[:top_k]):
                try:
                    ef = c.get('ef') or {}
                    keys = list(ef.get('claim_keys') or [])
                    ck_by_idx.append((i, keys))
                    cbd = ef.get('confidence_breakdown') or {}
                    for k in conf_parts.keys():
                        if k in cbd:
                            conf_parts[k].append(float(cbd.get(k) or 0.0))
                except Exception:
                    continue
            # Compute corroboration map (already available util)
            cor_map = compute_corroboration(ck_by_idx)
            # Detect simple contradictions: same subj|pred, object differs by negation token
            def _split_key(k: str) -> tuple[str,str,str]:
                try:
                    s,p,o = k.split('|',2)
                    return s,p,o
                except Exception:
                    return ('','','')
            def _is_negated(o: str) -> bool:
                t = o.strip().lower()
                return t.startswith('not ') or t.startswith("no ") or " didn't " in t or ' no ' in t
            pairs: List[Dict[str, Any]] = []
            for i, keys_i in ck_by_idx:
                for j, keys_j in ck_by_idx:
                    if j <= i:
                        continue
                    for ka in keys_i:
                        sa,pa,oa = _split_key(ka)
                        if not sa and not pa:
                            continue
                        for kb in keys_j:
                            sb,pb,ob = _split_key(kb)
                            if sa==sb and pa==pb and _is_negated(oa) != _is_negated(ob):
                                pairs.append({'keyA': ka, 'keyB': kb, 'citations': [i,j]})
                                break
            # Aggregate confidence
            comp = {k: (sum(v)/len(v) if v else 0.0) for k,v in conf_parts.items()}
            note_parts: List[str] = []
            if comp['evidence'] < 0.5:
                note_parts.append('Low evidence density')
            if not cor_map:
                note_parts.append('no corroboration')
            if comp['prior'] == 0.0:
                note_parts.append('prior neutral')
            conf_note = '; '.join(note_parts) if note_parts else 'balanced'
            answer['debug'].setdefault('ef', {})
            answer['debug']['ef']['contradiction_pairs'] = pairs
            answer['debug']['ef']['confidence_components'] = comp
            answer['debug']['ef']['confidence_note'] = conf_note
            # Degrade gating
            try:
                if bool(getattr(cfg, 'ef_degrade_enable', False)):
                    degraded = False
                    if comp['final_score'] < 0.4:
                        degraded = True
                    if recency_gate:
                        has_t01 = any(c.get('tier') in (0,1) or c.get('domain_trust') for c in citations[:top_k])
                        if not has_t01:
                            degraded = True
                    if degraded:
                        answer['degraded'] = True
                        answer['debug'].setdefault('policy', {})
                        answer['debug']['policy']['degraded_reason'] = conf_note or 'low confidence'
            except Exception:
                pass
        except Exception:
            pass
        # PR5: ledger logging (best-effort)
        try:
            if ledger_enable:
                log_veracity(cfg.cache_root, {
                    'query': q_sanitized,
                    'citations': [c.get('canonical_url') for c in citations[:top_k] if isinstance(c, dict)],
                    'flags': {
                        'ef': bool(getattr(cfg, 'evidence_first', False)),
                        'corroborate': bool('corroborate_enable' in locals() and corroborate_enable),
                        'counter_claim': bool(counter_claim_enable),
                        'reputation': bool(reputation_enable),
                    },
                })
        except Exception:
            pass

    # Persist updated dedupe index (best-effort)
    try:
        open(index_path, 'w', encoding='utf-8').write(json.dumps(hash_to_url, ensure_ascii=False))
    except Exception:
        pass
    try:
        open(cache_path, 'w', encoding='utf-8').write(json.dumps({'ts': now, 'result': answer}, ensure_ascii=False))
    except Exception:
        pass

    try:
        _progress.emit_current({"stage": "done", "status": "ok", "citations": len(citations[:top_k])})
    except Exception:
        pass
    return answer
