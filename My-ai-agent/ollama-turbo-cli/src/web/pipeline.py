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
from .search import search, SearchResult, _year_guard
from .fetch import fetch_url, _httpx_client
from .trust import trust_score
from .extract import extract_content
from .rerank import chunk_text, rerank_chunks
from .archive import save_page_now, get_memento
from .normalize import canonicalize, dedupe_citations
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


def run_research(query: str, *, cfg: Optional[WebConfig] = None, site_include: Optional[str] = None, site_exclude: Optional[str] = None, freshness_days: Optional[int] = None, top_k: int = 5, force_refresh: bool = False) -> Dict[str, Any]:
    cfg = cfg or _DEFAULT_CFG or WebConfig()
    # Honor per-call env overrides for dynamic fields used in tests and runtime tuning
    try:
        env_excl = os.getenv("WEB_EXCLUDE_CITATION_DOMAINS")
        if env_excl is not None:
            cfg.exclude_citation_domains = [d.strip().lower() for d in env_excl.split(',') if d.strip()]
        env_cache = os.getenv("WEB_CACHE_ROOT")
        if env_cache:
            cfg.cache_root = env_cache
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
    results = search(q_sanitized, cfg=cfg, site=site_include, freshness_days=freshness_final)
    try:
        _progress.emit_current({"stage": "search", "status": "done", "count": len(results)})
    except Exception:
        pass

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
    obs_discard_old: Dict[str, int] = {}
    obs_source_type: Dict[str, int] = {}
    obs_trust_decisions: List[Dict[str, Any]] = []
    # Dateline instrumentation
    dateline_from_structured = 0
    dateline_from_path = 0
    date_conf_hist: Dict[str, int] = {'high': 0, 'medium': 0, 'low': 0}
    allowlist_fallback_hit = False
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

    citations: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    # Dedupe across URLs and content bodies (current-run only)
    seen_urls: set[str] = set()
    # Scope content-hash dedupe to hostname, so different outlets with similar wire stubs both survive
    seen_hashes_by_host: Dict[str, set[str]] = {}
    dedupe_lock = threading.Lock()

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
    # Defaults (override via WEB_NEWS_SOURCES_ALLOW / WEB_NEWS_SOURCES_BLOCK comma lists)
    allow_defaults = [
        'apnews.com','reuters.com','bbc.co.uk','bbc.com','theguardian.com','aljazeera.com','nytimes.com',
        'wsj.com','haaretz.com','timesofisrael.com','al-monitor.com'
    ]
    block_defaults = [
        'liveuamap.com'
    ]
    try:
        env_allow = os.getenv('WEB_NEWS_SOURCES_ALLOW')
        if env_allow is not None:
            allow_list = [d.strip() for d in env_allow.split(',') if d.strip()]
        else:
            allow_list = allow_defaults
    except Exception:
        allow_list = allow_defaults
    try:
        env_block = os.getenv('WEB_NEWS_SOURCES_BLOCK')
        if env_block is not None:
            block_list = [d.strip() for d in env_block.split(',') if d.strip()]
        else:
            block_list = block_defaults
    except Exception:
        block_list = block_defaults

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
    def _build_citation(sr) -> Optional[Dict[str, Any]]:
        if site_exclude and site_exclude in (sr.url or ''):
            return None
        try:
            _progress.emit_current({"stage": "fetch", "status": "start", "url": sr.url})
        except Exception:
            pass
        # Source-type and policy checks prior to fetch
        stype = _source_type(sr.url)
        obs_source_type[stype] = obs_source_type.get(stype, 0) + 1
        h = _host(sr.url)
        if _in_list(h, block_list):
            items.append({'url': sr.url, 'ok': False, 'reason': 'blocked-source'})
            return None
        if stype in {'liveblog','map'}:
            items.append({'url': sr.url, 'ok': False, 'reason': f'blocked-{stype}'})
            return None
        # Respect exclusion list: allow discovery (search) but skip quoting as a citation
        if _host_in_excluded(sr.url):
            items.append({'url': sr.url, 'ok': False, 'reason': 'excluded-domain'})
            nonlocal excluded_skips
            with dedupe_lock:
                excluded_skips += 1
            return None
        f = fetch_url(sr.url, cfg=cfg, force_refresh=force_refresh, use_browser_if_needed=True)
        if not f.ok:
            items.append({'url': sr.url, 'ok': False, 'reason': f.reason or f"HTTP {f.status}"})
            try:
                rkey = (f.reason or f"HTTP {f.status}")
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
            obs_extract_fail += 1
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

        # Recency/date gating and language sanity for recent events (hardened dateline)
        # PDFs are typically evergreen/primary sources; bypass recency gating
        if recency_gate and ex.kind != 'pdf':
            # Determine date and confidence
            pub_ts = _parse_pub_date(ex.date)
            if pub_ts:
                date_conf = 'high'
                nonlocal dateline_from_structured
                dateline_from_structured += 1
            else:
                # Fallback: parse date from URL path (e.g., /2025/09/08/)
                def _date_from_path(u: str) -> Optional[float]:
                    try:
                        p = urlparse(u)
                        path = (p.path or '')
                        m = re.search(r"/20(\d{2})/(\d{2})/(\d{2})/", path)
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
                    date_conf = 'medium'
                    nonlocal dateline_from_path
                    dateline_from_path += 1
            # Decide acceptance for recency
            if not pub_ts:
                if use_trust and (not is_allowlisted) and (t_score >= threshold) and bool(getattr(cfg, 'dateline_soft_accept', False)):
                    decision = 'soft-accept-undated'
                    decision_note = 'NO_DATE_RECENCY_TRUST_OK'
                elif is_allowlisted:
                    # allowlist path: reject undated in recency when dateline_soft_accept is false
                    if not cfg.dateline_soft_accept:
                        obs_discard_old['missing_dateline'] = obs_discard_old.get('missing_dateline', 0) + 1
                        decision = 'reject'
                        decision_note = 'NO_DATE_RECENCY_ALLOWLIST_REJECT'
                        # log and bail
                        try:
                            if cfg.debug_metrics:
                                obs_trust_decisions.append({'host': h, 'mode': trust_mode, 'wildcard': wildcard, 'allowlisted': True, 'score': None, 'decision': decision, 'reason': decision_note})
                        except Exception:
                            pass
                        return None
                else:
                    obs_discard_old['missing_dateline'] = obs_discard_old.get('missing_dateline', 0) + 1
                    decision = 'reject'
                    decision_note = 'NO_DATE_RECENCY_REJECT'
                    try:
                        if cfg.debug_metrics:
                            obs_trust_decisions.append({'host': h, 'mode': trust_mode, 'wildcard': wildcard, 'allowlisted': False, 'score': t_score if use_trust else None, 'decision': decision, 'reason': decision_note})
                    except Exception:
                        pass
                    return None
            if pub_ts:
                if (now_ts - pub_ts) > window_secs:
                    obs_discard_old['dateline_out_of_window'] = obs_discard_old.get('dateline_out_of_window', 0) + 1
                    decision = 'reject'
                    decision_note = 'OUT_OF_WINDOW'
                    try:
                        if cfg.debug_metrics:
                            obs_trust_decisions.append({'host': h, 'mode': trust_mode, 'wildcard': wildcard, 'allowlisted': is_allowlisted, 'score': t_score if use_trust else None, 'decision': decision, 'reason': decision_note})
                    except Exception:
                        pass
                    return None
            # Record date confidence histogram
            try:
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
        if recency_gate and (pub_ts is None):
            obs_discard_old['missing_dateline'] = obs_discard_old.get('missing_dateline', 0) + 1
            try:
                if cfg.debug_metrics:
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

        cit = {
            'canonical_url': f.final_url,
            'archive_url': archive.get('archive_url', ''),
            'title': ex.title or sr.title,
            'date': ex.date or sr.published,
            'risk': ex.risk,
            'risk_reasons': ex.risk_reasons,
            'browser_used': f.browser_used,
            'kind': ex.kind,
            'date_confidence': ('high' if pub_ts and ex.date else ('medium' if pub_ts else 'low')),
            'domain_trust': bool(is_allowlisted or (use_trust and (t_score >= threshold))),
            'trust_score': (t_score if use_trust else None),
            'trust_mode': trust_mode,
            'undated': bool(not ex.date),
            'lines': [],
        }
        try:
            if cfg.debug_metrics:
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
    max_workers = max(1, min(8, cfg.per_host_concurrency))
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
            except Exception:
                continue

    # Canonicalize + Deduplicate before ordering/limiting
    for c in citations:
        try:
            c['canonical_url'] = canonicalize(c.get('canonical_url') or c.get('url') or '')
        except Exception:
            pass
    citations = dedupe_citations(citations)
    # Final safety: in recency mode, drop undated citations
    if recency_gate:
        citations = [c for c in citations if c.get('date')]
    # Prefer trusted domains and higher date confidence, then by risk/url
    def _conf_val(v: str) -> int:
        return 2 if v == 'high' else (1 if v == 'medium' else 0)
    citations.sort(key=lambda c: (
        -1 if c.get('domain_trust') else 0,
        -_conf_val(str(c.get('date_confidence','low'))),
        c.get('risk',''),
        hashlib.sha256((c.get('canonical_url','') or '').encode()).hexdigest()
    ))

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
                futs2 = [exr2.submit(_build_citation, sr) for sr in allow_candidates[: top_k * 2]]
                for fut in as_completed(futs2):
                    try:
                        cit = fut.result()
                        if cit:
                            citations.append(cit)
                    except Exception:
                        continue
            allowlist_fallback_hit = bool(citations)


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
        },
    }

    if cfg.debug_metrics:
        try:
            answer['debug'] = {
                'search': {
                    'initial_count': base_count,
                    'simplified_count': simplified_count,
                    'variant_count': variant_count,
                    'emergency_count': emergency_count,
                },
                'fetch': {
                    'attempted': len(candidates),
                    'ok': len(citations[:top_k]),
                    'failed': len([it for it in items if not it.get('ok')]),
                    'dedupe_skips': dedup_skips,
                    'excluded': excluded_skips,
                    'wiki_refs_added': wiki_refs_added,
                    'fail_reasons': obs_fetch_fail,
                },
                'extract': {
                    'fail_count': obs_extract_fail,
                },
                'discard': obs_discard_old,
                'source_type': obs_source_type,
                'year_guard': yg_counters,
                'resolved_window_days': freshness_final,
                'relative_span_resolved': bool(resolved_days is not None),
                'dateline_from_structured': dateline_from_structured,
                'dateline_from_path': dateline_from_path,
                'date_confidence_histogram': date_conf_hist,
                'allowlist_fallback_hit': allowlist_fallback_hit,
                'rerank_softdate_selected': len([c for c in citations[:top_k] if str(c.get('date_confidence','low')) != 'high']),
                'trust_decisions': obs_trust_decisions,
            }
            # One-line summary for quick telemetry
            kept = len(citations[:top_k])
            line = (
                f"mode=researcher • fresh={freshness_final or cfg.default_freshness_days}d • hits={len(results)} "
                f"• kept={kept} • extracted={kept} • sources=[" + ",".join(
                    sorted({(urlparse(c.get('canonical_url') or '').hostname or '').split(':')[0] for c in citations[:top_k] if c.get('canonical_url')})
                ) + "]"
            )
            answer['debug']['summary_line'] = line
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
