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
from .search import search
from .fetch import fetch_url
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

    citations: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    # Dedupe across URLs and content bodies
    seen_urls: set[str] = set()
    # Start with any known hashes from previous runs
    seen_hashes: set[str] = set(hash_to_url.keys())
    dedupe_lock = threading.Lock()

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
        },
    }

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
