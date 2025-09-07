from __future__ import annotations
import os
import re
import time
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from urllib.parse import quote_plus, urlparse, urljoin
import xml.etree.ElementTree as ET
import gzip

from .config import WebConfig
from .fetch import _httpx_client


@dataclass
class SearchQuery:
    query: str
    site: Optional[str] = None
    freshness_days: Optional[int] = None


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    published: Optional[str]


def _norm(sr: Dict[str, Any], source: str) -> SearchResult:
    title = sr.get('title') or sr.get('name') or sr.get('title_full') or ''
    url = sr.get('url') or sr.get('link') or sr.get('href') or ''
    snippet = sr.get('snippet') or sr.get('description') or sr.get('summary') or ''
    published = sr.get('date') or sr.get('published') or sr.get('published_at')
    return SearchResult(title=title, url=url, snippet=snippet, source=source, published=published)


def _hash_url(u: str) -> str:
    return hashlib.sha256(u.encode()).hexdigest()[:16]


def _title_trigrams(t: str) -> set:
    toks = re.findall(r"\w+", t.lower())
    grams = set()
    for i in range(len(toks)-2):
        grams.add(" ".join(toks[i:i+3]))
    return grams


def _dedupe(items: List[SearchResult]) -> List[SearchResult]:
    seen: set = set()
    out: List[SearchResult] = []
    title_keys: list[set] = []
    for it in items:
        key = _hash_url(it.url)
        tkey = _title_trigrams(it.title)
        if key in seen:
            continue
        # basic title trigram de-dupe
        dup = False
        for k in title_keys:
            if len(k & tkey) >= 2:  # two common trigrams => duplicate-ish
                dup = True
                break
        if not dup:
            seen.add(key)
            out.append(it)
            title_keys.append(tkey)
    return out


def _norm_host(site: str) -> str:
    """Normalize a site string to host (netloc)."""
    try:
        p = urlparse(site if '://' in site else f"https://{site}")
        host = p.netloc or p.path
        return host.strip('/')
    except Exception:
        return site


def _discover_sitemaps_for_site(site: str, cfg: WebConfig) -> List[str]:
    host = _norm_host(site)
    base = f"https://{host}"
    sitemaps: List[str] = []
    try:
        headers = {"User-Agent": cfg.user_agent, "Accept": "text/plain, */*"}
        timeout = cfg.sitemap_timeout_s
        with _httpx_client(cfg) as c:
            # robots.txt discovery
            try:
                r = c.get(urljoin(base, "/robots.txt"), headers=headers, timeout=timeout)
                if r.status_code == 200 and isinstance(r.text, str):
                    for line in r.text.splitlines():
                        if line.lower().startswith("sitemap:"):
                            url = line.split(":", 1)[1].strip()
                            if url:
                                sitemaps.append(url)
            except Exception:
                pass
            # common default
            try:
                r2 = c.get(urljoin(base, "/sitemap.xml"), headers=headers, timeout=timeout)
                if r2.status_code == 200:
                    sitemaps.append(str(r2.request.url))
            except Exception:
                pass
    except Exception:
        return []
    # de-dup preserve order
    seen: set[str] = set()
    out: List[str] = []
    for u in sitemaps:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _parse_sitemap_urls(sitemap_url: str, cfg: WebConfig, *, limit: int, include_subs: bool, visited: set[str]) -> List[str]:
    """Parse a sitemap or sitemap index and return up to limit URLs.
    Avoid cycles via visited set.
    """
    if sitemap_url in visited or limit <= 0:
        return []
    visited.add(sitemap_url)
    urls: List[str] = []
    try:
        headers = {"User-Agent": cfg.user_agent, "Accept": "application/xml,text/xml,application/rss+xml,application/gzip"}
        with _httpx_client(cfg) as c:
            r = c.get(sitemap_url, headers=headers, timeout=cfg.sitemap_timeout_s)
            if r.status_code != 200:
                return []
            content: bytes
            try:
                if str(sitemap_url).endswith(".gz"):
                    content = gzip.decompress(r.content)
                else:
                    # httpx auto-decodes Content-Encoding; ensure bytes
                    content = r.content
            except Exception:
                content = r.content
            # Parse XML
            try:
                root = ET.fromstring(content)
            except Exception:
                return []
            tag = root.tag.lower()
            # Strip namespaces for robust matching
            def _strip_ns(t: str) -> str:
                return t.split('}')[-1] if '}' in t else t
            tag = _strip_ns(tag)
            if tag == 'urlset':
                # Iterate over all descendants and pick <loc> regardless of namespace
                for uel in root.iter():
                    if _strip_ns(uel.tag) == 'loc' and uel.text:
                        urls.append(uel.text.strip())
                        if len(urls) >= limit:
                            break
            elif tag == 'sitemapindex' and include_subs:
                for sm in root.iter():
                    if _strip_ns(sm.tag) == 'loc' and sm.text:
                        sub = sm.text.strip()
                        if sub and sub not in visited and len(urls) < limit:
                            sub_urls = _parse_sitemap_urls(sub, cfg, limit=limit - len(urls), include_subs=include_subs, visited=visited)
                            urls.extend(sub_urls)
                            if len(urls) >= limit:
                                break
    except Exception:
        return urls
    return urls[:limit]


def _sitemap_search(q: SearchQuery, cfg: WebConfig) -> List[SearchResult]:
    if not cfg.sitemap_enabled or not q.site:
        return []
    try:
        sitemaps = _discover_sitemaps_for_site(q.site, cfg)
        if not sitemaps:
            return []
        visited: set[str] = set()
        limit = max(1, int(cfg.sitemap_max_urls))
        include_subs = bool(cfg.sitemap_include_subs)
        agg: List[str] = []
        for sm in sitemaps:
            if len(agg) >= limit:
                break
            urls = _parse_sitemap_urls(sm, cfg, limit=limit - len(agg), include_subs=include_subs, visited=visited)
            agg.extend(urls)
            if len(agg) >= limit:
                break
        # Map to SearchResult
        items: List[SearchResult] = []
        src = 'sitemap'
        for u in agg:
            try:
                title = u
                # Derive a simple title from path
                pu = urlparse(u)
                if pu.path and pu.path.strip('/'):
                    title = pu.path.strip('/').split('/')[-1].replace('-', ' ').replace('_', ' ')
                items.append(SearchResult(title=title or u, url=u, snippet='', source=src, published=None))
            except Exception:
                continue
        return items
    except Exception:
        return []


def _search_brave(q: SearchQuery, cfg: WebConfig) -> List[SearchResult]:
    key = cfg.brave_key
    if not key:
        return []
    try:
        params = {"q": q.query, "count": 10}
        if q.site:
            params["q"] = f"site:{q.site} {q.query}"
        headers = {"X-Subscription-Token": key, "Accept": "application/json", "User-Agent": cfg.user_agent}
        with _httpx_client(cfg) as c:
            r = c.get("https://api.search.brave.com/res/v1/web/search", params=params, headers=headers, timeout=cfg.timeout_read)
            data = r.json()
            items = []
            for d in data.get('web', {}).get('results', []):
                items.append(_norm({'title': d.get('title'), 'url': d.get('url'), 'snippet': d.get('description'), 'published': d.get('page_age')}, 'brave'))
            return items
    except Exception:
        return []


def _search_tavily(q: SearchQuery, cfg: WebConfig) -> List[SearchResult]:
    key = cfg.tavily_key
    if not key:
        return []
    try:
        payload = {"query": q.query, "search_depth": "basic", "max_results": 10}
        if q.site:
            payload["query"] = f"site:{q.site} {q.query}"
        headers = {"Content-Type":"application/json", "Authorization": f"Bearer {key}"}
        with _httpx_client(cfg) as c:
            r = c.post("https://api.tavily.com/search", json=payload, headers=headers, timeout=cfg.timeout_read)
            data = r.json()
            items = []
            for d in data.get('results', []):
                items.append(_norm({'title': d.get('title'), 'url': d.get('url'), 'snippet': d.get('content')}, 'tavily'))
            return items
    except Exception:
        return []


def _search_exa(q: SearchQuery, cfg: WebConfig) -> List[SearchResult]:
    key = cfg.exa_key
    if not key:
        return []
    try:
        payload = {"query": q.query, "numResults": 10}
        if q.site:
            payload["query"] = f"site:{q.site} {q.query}"
        headers = {"Content-Type":"application/json", "x-api-key": key}
        with _httpx_client(cfg) as c:
            r = c.post("https://api.exa.ai/search", json=payload, headers=headers, timeout=cfg.timeout_read)
            data = r.json()
            items = []
            for d in data.get('results', []):
                items.append(_norm({'title': d.get('title'), 'url': d.get('url'), 'snippet': d.get('snippet')}, 'exa'))
            return items
    except Exception:
        return []


def _search_google_pse(q: SearchQuery, cfg: WebConfig) -> List[SearchResult]:
    key = cfg.google_pse_key; cx = cfg.google_pse_cx
    if not key or not cx:
        return []
    try:
        qtext = f"site:{q.site} {q.query}" if q.site else q.query
        with _httpx_client(cfg) as c:
            r = c.get("https://www.googleapis.com/customsearch/v1", params={"key": key, "cx": cx, "q": qtext}, timeout=cfg.timeout_read)
            data = r.json()
            items = []
            for d in data.get('items', []) or []:
                items.append(_norm({'title': d.get('title'), 'url': d.get('link'), 'snippet': d.get('snippet')}, 'google_pse'))
            return items
    except Exception:
        return []


def search(query: str, *, cfg: Optional[WebConfig] = None, site: Optional[str] = None, freshness_days: Optional[int] = None) -> List[SearchResult]:
    cfg = cfg or WebConfig()
    q = SearchQuery(query=query, site=site, freshness_days=freshness_days)
    candidates: List[SearchResult] = []
    # Provider rotation: Brave -> Tavily/Exa -> Google PSE
    candidates.extend(_search_brave(q, cfg))
    if not candidates:
        candidates.extend(_search_tavily(q, cfg))
        if not candidates:
            candidates.extend(_search_exa(q, cfg))
    if not candidates:
        candidates.extend(_search_google_pse(q, cfg))
    # Optional sitemap ingestion (augment results for site-restricted queries)
    if cfg.sitemap_enabled and q.site:
        try:
            candidates.extend(_sitemap_search(q, cfg))
        except Exception:
            pass
    # Deduplicate
    out = _dedupe([it for it in candidates if it.url])
    return out[:15]
