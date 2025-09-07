from __future__ import annotations
import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import WebConfig
from . import progress as _progress
from .search import search, SearchResult
from .fetch import fetch_url, _httpx_client
from .extract import extract_content
from .rerank import chunk_text, rerank_chunks
from .archive import save_page_now, get_memento


def _query_cache_key(query: str, opts: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(query.encode())
    h.update(json.dumps(opts, sort_keys=True).encode())
    return h.hexdigest()


def run_research(query: str, *, cfg: Optional[WebConfig] = None, site_include: Optional[str] = None, site_exclude: Optional[str] = None, freshness_days: Optional[int] = None, top_k: int = 5, force_refresh: bool = False) -> Dict[str, Any]:
    cfg = cfg or WebConfig()
    os.makedirs(cfg.cache_root, exist_ok=True)
    opts = {
        'site_include': site_include,
        'site_exclude': site_exclude,
        'freshness_days': freshness_days,
        'top_k': top_k,
    }
    key = _query_cache_key(query, opts)
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
    results = search(query, cfg=cfg, site=site_include, freshness_days=freshness_days)
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
    if not results:
        # Heuristic: simplify long/narrow queries and retry once
        try:
            import re as _re
            toks = _re.findall(r"[A-Za-z0-9]+", query)
            stop = {
                'the','a','an','of','in','on','for','to','and','or','with','about','from','by','at','as',
                'is','are','was','were','be','being','been','this','that','these','those','it','its','into',
                '2023','2024','2025'
            }
            key_toks = [t for t in toks if t.lower() not in stop]
            short_q = " ".join(key_toks[:6]) or query[:80]
        except Exception:
            short_q = query[:80]
        if short_q and short_q.strip().lower() != query.strip().lower():
            try:
                _progress.emit_current({"stage": "search", "status": "retry", "query": short_q})
            except Exception:
                pass
            results = search(short_q, cfg=cfg, site=site_include, freshness_days=freshness_days)
            simplified_used = True
            simplified_count = len(results)

    if not results:
        # Final variant fallback: try "<ProperNoun> political makeup 2024" style seed
        try:
            import re as _re
            toks = _re.findall(r"[A-Za-z][A-Za-z0-9-]*", query)
            proper = None
            for t in toks:
                if t[0].isupper():
                    proper = t
                    break
            core = proper or (toks[0] if toks else "")
            if core:
                variant_q = f"{core} political makeup 2024"
                try:
                    _progress.emit_current({"stage": "search", "status": "variant", "query": variant_q})
                except Exception:
                    pass
                results = search(variant_q, cfg=cfg, site=site_include, freshness_days=freshness_days)
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

    citations: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    # Dedupe across URLs and content bodies
    seen_urls: set[str] = set()
    # Start with any known hashes from previous runs
    seen_hashes: set[str] = set(hash_to_url.keys())
    dedupe_lock = threading.Lock()

    dedup_skips = 0
    def _build_citation(sr) -> Optional[Dict[str, Any]]:
        if site_exclude and site_exclude in (sr.url or ''):
            return None
        try:
            _progress.emit_current({"stage": "fetch", "status": "start", "url": sr.url})
        except Exception:
            pass
        f = fetch_url(sr.url, cfg=cfg, force_refresh=force_refresh, use_browser_if_needed=True)
        if not f.ok:
            items.append({'url': sr.url, 'ok': False, 'reason': f.reason or f"HTTP {f.status}"})
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
            try:
                _progress.emit_current({"stage": "extract", "status": "error", "url": f.final_url})
            except Exception:
                pass
            return None
        # Dedupe by URL and content hash (computed on markdown)
        body_hash = hashlib.sha256((ex.markdown or '').encode('utf-8', 'ignore')).hexdigest()
        with dedupe_lock:
            if f.final_url in seen_urls or body_hash in seen_hashes or body_hash in hash_to_url:
                nonlocal dedup_skips
                dedup_skips += 1
                return None
            seen_urls.add(f.final_url)
            seen_hashes.add(body_hash)
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
            'lines': [],
        }
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

    # Deterministic order: by score if available then by url hash
    citations.sort(key=lambda c: (c.get('risk', ''), hashlib.sha256(c.get('canonical_url','').encode()).hexdigest()))

    answer = {
        'query': query,
        'top_k': top_k,
        'citations': citations[:top_k],
        'policy': {
            'respect_robots': cfg.respect_robots,
            'allow_browser': cfg.allow_browser,
            'cache_ttl_seconds': cfg.cache_ttl_seconds,
            'simplified_query_used': simplified_used,
            'emergency_search_used': emergency_used,
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
                },
            }
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
