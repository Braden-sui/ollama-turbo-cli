from __future__ import annotations
import os
import re
import time
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from urllib.parse import quote_plus, urlparse, urljoin, parse_qs, unquote
import xml.etree.ElementTree as ET
import gzip
import httpx
from datetime import datetime

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


def _is_recency_trigger(text: str, freshness_days: Optional[int]) -> bool:
    try:
        if freshness_days and int(freshness_days) > 0:
            return True
    except Exception:
        pass
    t = (text or "").lower()
    return any(k in t for k in ["today", "this week", "recent", "recently", "latest", "breaking", "past week", "last week"])


def _year_guard(q: str, cfg: WebConfig) -> tuple[str, Dict[str, int]]:
    """Strip templated year artifacts without touching genuine user constraints.

    Minimal policy:
    - Remove a trailing "Mon YYYY" or "YYYY" token often appended by templates.
    - Do nothing if feature is disabled.
    Returns sanitized query and counters.
    """
    counters = {"stripped_year_tokens": 0}
    try:
        if not getattr(cfg, 'year_guard_enabled', True):
            return q, counters
        s = q.strip()
        # Strip trailing "Mon YYYY" (e.g., "Sep 2025")
        if re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+20\d{2}$", s, flags=re.I):
            s = re.sub(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+20\d{2}$", "", s, flags=re.I).strip()
            counters["stripped_year_tokens"] += 1
        # Strip trailing standalone year token (e.g., "2024")
        if re.search(r"\b20\d{2}$", s):
            s = re.sub(r"\b20\d{2}$", "", s).strip()
            counters["stripped_year_tokens"] += 1
        return s, counters
    except Exception:
        return q, counters


def _recency_augmented_query(q: SearchQuery, cfg: WebConfig) -> str:
    """Return the query without injecting years; apply YearGuard to remove templated tails."""
    base = (q.query or "").strip()
    try:
        sanitized, _ = _year_guard(base, cfg)
        return sanitized
    except Exception:
        return base


def _discover_sitemaps_for_site(site: str, cfg: WebConfig) -> List[str]:
    host = _norm_host(site)
    base = f"https://{host}"
    sitemaps: List[str] = []
    try:
        headers = {"User-Agent": cfg.user_agent, "Accept": "text/plain, */*"}
        with httpx.Client(timeout=cfg.sitemap_timeout_s, headers=headers, follow_redirects=cfg.follow_redirects) as c:
            # robots.txt discovery
            try:
                r = c.get(urljoin(base, "/robots.txt"))
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
                r2 = c.get(urljoin(base, "/sitemap.xml"))
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
        with httpx.Client(timeout=cfg.sitemap_timeout_s, headers=headers, follow_redirects=cfg.follow_redirects) as c:
            r = c.get(sitemap_url)
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


def _search_duckduckgo_fallback(q: SearchQuery, cfg: WebConfig) -> List[SearchResult]:
    """Keyless fallback using DuckDuckGo Instant Answer API with HTML fallback.

    This avoids external API keys and keeps network policy centralized via WebConfig.
    """
    try:
        aug = _recency_augmented_query(q, cfg)
        qtext = f"site:{q.site} {aug}" if q.site else aug
        headers_json = {"Accept": "application/json", "User-Agent": cfg.user_agent}
        params = {"q": qtext, "format": "json", "no_html": "1", "no_redirect": "1", "t": "ollama-turbo-cli"}
        results: List[SearchResult] = []
        with _httpx_client(cfg) as c:
            try:
                r = c.get("https://api.duckduckgo.com/", params=params, headers=headers_json, timeout=cfg.timeout_read)
                if r.status_code == 200:
                    data = r.json()
                    # Prefer Abstract if present
                    abstract = (data.get("AbstractText") or data.get("Abstract") or "").strip()
                    abstract_url = (data.get("AbstractURL") or "").strip()
                    if abstract and abstract_url:
                        results.append(_norm({"title": data.get("Heading") or "Instant Answer", "url": abstract_url, "snippet": abstract}, "duckduckgo"))
                    # RelatedTopics (flatten minimal subset)
                    def _flatten(items):
                        out = []
                        for it in items or []:
                            if isinstance(it, dict) and it.get("FirstURL"):
                                out.append({
                                    "title": (it.get("Text") or "").split(" - ")[0][:120],
                                    "url": it.get("FirstURL"),
                                    "snippet": it.get("Text") or "",
                                })
                            elif isinstance(it, dict) and it.get("Topics"):
                                out.extend(_flatten(it.get("Topics")))
                        return out
                    for it in _flatten(data.get("RelatedTopics")):
                        results.append(_norm(it, "duckduckgo"))
            except Exception:
                results = []
            if results:
                # Deduplicate by URL
                seen = set()
                uniq: List[SearchResult] = []
                for r0 in results:
                    if r0.url and r0.url not in seen:
                        seen.add(r0.url)
                        uniq.append(r0)
                return uniq[:10]
            # HTML fallback
            headers_html = {
                "Accept": "text/html",
                "User-Agent": cfg.user_agent,
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://duckduckgo.com/",
            }
            collected: List[SearchResult] = []
            for base in ("https://duckduckgo.com/lite/", "https://html.duckduckgo.com/html/"):
                try:
                    # Add locale and web intent to reduce gating and region variance
                    params_html = {"q": qtext, "kl": "wt-wt", "ia": "web", "t": "ollama-turbo-cli"}
                    r2 = c.get(base, params=params_html, headers=headers_html, timeout=cfg.timeout_read)
                except Exception:
                    continue
                if r2.status_code != 200:
                    continue
                html = r2.text or ""
                links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.I)
                for href, text in links:
                    try:
                        absolute = href if href.lower().startswith("http") else urljoin(base, href)
                        parsed = urlparse(absolute)
                        netloc = parsed.netloc or ""
                    except Exception:
                        continue
                    resolved_url = None
                    if netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
                        # decode external URL from uddg param
                        try:
                            qs = parse_qs(parsed.query or "")
                            uddg = (qs.get("uddg") or [None])[0]
                            if uddg:
                                resolved_url = unquote(uddg)
                        except Exception:
                            pass
                    elif netloc.endswith("duckduckgo.com") or netloc.endswith("duck.com"):
                        continue
                    else:
                        resolved_url = absolute if absolute.lower().startswith("http") else None
                    if not resolved_url:
                        continue
                    title_text = re.sub(r"<[^>]+>", "", text)[:120].strip() or "(no title)"
                    snippet = re.sub(r"<[^>]+>", "", text)[:160].strip()
                    sr = SearchResult(title=title_text, url=resolved_url, snippet=snippet, source="duckduckgo", published=None)
                    if sr.url not in [c.url for c in collected]:
                        collected.append(sr)
                    if len(collected) >= 10:
                        break
                if collected:
                    break
            return collected[:10]
    except Exception:
        return []


def _search_brave(q: SearchQuery, cfg: WebConfig) -> List[SearchResult]:
    key = cfg.brave_key
    if not key:
        return []
    try:
        aug = _recency_augmented_query(q, cfg)
        params = {"q": aug, "count": 10}
        if q.site:
            params["q"] = f"site:{q.site} {aug}"
        headers = {"X-Subscription-Token": key, "Accept": "application/json", "User-Agent": cfg.user_agent}
        with _httpx_client(cfg) as c:
            r = c.get("https://api.search.brave.com/res/v1/web/search", params=params, headers=headers, timeout=cfg.timeout_read)
            try:
                data = r.json()
            except Exception:
                return []
            items: List[SearchResult] = []
            # Normal, typed mapping
            try:
                for d in (data.get('web', {}) or {}).get('results', []) or []:
                    items.append(_norm({'title': d.get('title'), 'url': d.get('url'), 'snippet': d.get('description'), 'published': d.get('page_age')}, 'brave'))
            except Exception:
                items = []
            # Fallback: generic crawl for url/title pairs anywhere in the payload
            if not items:
                try:
                    def _walk(v, out: List[SearchResult]):
                        if isinstance(v, dict):
                            u = v.get('url') or v.get('link') or v.get('href')
                            t = v.get('title') or v.get('name') or v.get('heading') or ''
                            s = v.get('description') or v.get('snippet') or ''
                            if isinstance(u, str) and u.startswith('http'):
                                out.append(_norm({'title': t, 'url': u, 'snippet': s}, 'brave'))
                            for vv in v.values():
                                _walk(vv, out)
                        elif isinstance(v, list):
                            for it in v:
                                _walk(it, out)
                    tmp: List[SearchResult] = []
                    _walk(data, tmp)
                    # Deduplicate by URL preserving order
                    seen: set[str] = set()
                    items = []
                    for it in tmp:
                        if it.url and it.url not in seen:
                            seen.add(it.url)
                            items.append(it)
                except Exception:
                    pass
            return items
    except Exception:
        return []


def _search_tavily(q: SearchQuery, cfg: WebConfig) -> List[SearchResult]:
    key = cfg.tavily_key
    if not key:
        return []
    try:
        aug = _recency_augmented_query(q, cfg)
        payload = {"query": aug, "search_depth": "basic", "max_results": 10}
        if q.site:
            payload["query"] = f"site:{q.site} {aug}"
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
        aug = _recency_augmented_query(q, cfg)
        payload = {"query": aug, "numResults": 10}
        if q.site:
            payload["query"] = f"site:{q.site} {aug}"
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
        aug = _recency_augmented_query(q, cfg)
        qtext = f"site:{q.site} {aug}" if q.site else aug
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
    # Keyless fallback to DuckDuckGo if no API-backed engines succeeded
    if not candidates:
        candidates.extend(_search_duckduckgo_fallback(q, cfg))
    # Heuristic fallback: simplify long/narrow queries and retry
    if not candidates:
        try:
            toks = re.findall(r"[A-Za-z0-9]+", query)
            # Build dynamic year stop set excluding current year
            try:
                now_y = datetime.now().year
            except Exception:
                now_y = 2025
            years = {str(y) for y in range(2010, max(2010, now_y))}
            stop = {
                'the','a','an','of','in','on','for','to','and','or','with','about','from','by','at','as',
                'is','are','was','were','be','being','been','this','that','these','those','it','its','into',
            } | years
            key_toks = [t for t in toks if t.lower() not in stop]
            short_q = " ".join(key_toks[:6]) or query[:80]
        except Exception:
            short_q = query[:80]
        if short_q and short_q.strip().lower() != query.strip().lower():
            q2 = SearchQuery(query=short_q, site=site, freshness_days=freshness_days)
            # Retry full rotation
            items2: List[SearchResult] = []
            items2.extend(_search_brave(q2, cfg))
            if not items2:
                items2.extend(_search_tavily(q2, cfg))
                if not items2:
                    items2.extend(_search_exa(q2, cfg))
            if not items2:
                items2.extend(_search_google_pse(q2, cfg))
            if not items2:
                items2.extend(_search_duckduckgo_fallback(q2, cfg))
            candidates.extend(items2)
    # Optional sitemap ingestion (augment results for site-restricted queries)
    if cfg.sitemap_enabled and q.site:
        try:
            candidates.extend(_sitemap_search(q, cfg))
        except Exception:
            pass
    # Deduplicate
    out = _dedupe([it for it in candidates if it.url])
    return out[:15]
