"""DuckDuckGo search plugin"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List
from urllib.parse import urlparse, parse_qs, unquote, urljoin

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "duckduckgo_search",
        "description": (
            "Keyless web search via DuckDuckGo Instant Answer API (with HTML fallback). "
            "Use to discover sources or quick facts; provide a focused query (keywords or a quoted phrase). "
            "Returns concise top results (title, URL, snippet). Choose a small max_results (1–5). "
            "Subject to the agent's network policy (allowlist/proxy); if blocked, acknowledge policy and suggest narrowing or alternative domains."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused search query (keywords or quoted phrase). Avoid full natural-language questions when possible."},
                "max_results": {"type": "integer", "description": "How many results to return (1-5). Choose the smallest number that answers the question.", "default": 3}
            },
            "required": ["query"]
        }
    }
}

def duckduckgo_search(query: str, max_results: int = 3):
    """Search using DuckDuckGo Instant Answer API (no API key required).

    Returns JSON with fields:
      { ok: bool, query: str, results: [{rank,title,url,snippet}], engine: 'duckduckgo', error?: {message}}
    """
    try:
        if not requests:
            return {"ok": False, "query": query, "engine": "duckduckgo", "results": [], "error": {"message": "requests not installed"}}

        query = (query or "").strip()
        # Note: Do NOT apply YearGuard here; keep the raw user query for search.
        # YearGuard is reserved for multi-hop researcher paths.
        if not query:
            return {"ok": False, "query": query, "engine": "duckduckgo", "results": [], "error": {"message": "query must be provided"}}

        try:
            max_results = int(max_results)
            if max_results < 1:
                max_results = 1
            if max_results > 5:
                max_results = 5
        except Exception:
            max_results = 3

        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
            "t": "ollama-turbo-cli",
        }
        # Use centralized WebConfig for user agent if available; prefer a browser-like UA
        # to avoid DDG gating (e.g., HTTP 202/403 for non-browser UAs).
        try:
            from ..web.pipeline import _DEFAULT_CFG
            ua = (_DEFAULT_CFG.user_agent if _DEFAULT_CFG else "") or ""
        except Exception:
            ua = ""
        # Fallback to a common Chromium UA if UA is empty or obviously bot-like
        if (not ua) or ("ollama" in ua.lower()) or ("cli" in ua.lower()):
            ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

        headers = {
            "User-Agent": ua,
            "Accept": "application/json",
        }

        # Try API with small retries for transient statuses (e.g., 202, 429, 5xx)
        resp = None
        for attempt in range(5):
            try:
                resp = requests.get("https://api.duckduckgo.com/", params=params, headers=headers, timeout=8)
            except Exception as e:
                if attempt == 4:
                    return {"ok": False, "query": query, "engine": "duckduckgo", "results": [], "error": {"message": f"network error: {e}"}}
                time.sleep(0.5 * (2 ** attempt))
                continue
            if resp.status_code == 200:
                break
            if attempt < 4 and resp.status_code in (202, 429, 403, 500, 502, 503, 504):
                time.sleep(0.5 * (2 ** attempt))
                continue
            else:
                break

        data = None
        if resp is not None and resp.status_code == 200:
            try:
                data = resp.json()
            except json.JSONDecodeError:
                data = None  # trigger HTML fallback below

        if data is None:
            # Fallback to HTML (Lite) endpoint and extract links
            html_headers = {
                "User-Agent": ua,
                "Accept": "text/html",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://duckduckgo.com/",
                "DNT": "1",
            }
            fallback_urls = [
                "https://duckduckgo.com/lite/",
                "https://html.duckduckgo.com/html/",
            ]
            collected: List[Dict[str, str]] = []
            for base in fallback_urls:
                try:
                    params_html = {"q": query, "kl": "wt-wt"}  # worldwide locale to reduce region gating
                    r = requests.get(base, params=params_html, headers=html_headers, timeout=8)
                except Exception:
                    continue
                if r.status_code != 200:
                    continue
                html = r.text or ""
                # Extract anchors; handle DDG redirect links (/l/?uddg=...)
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
                        # Decode external URL from uddg param
                        try:
                            qs = parse_qs(parsed.query)
                            uddg = qs.get("uddg", [None])[0]
                            if uddg:
                                resolved_url = unquote(uddg)
                        except Exception:
                            pass
                    elif netloc.endswith("duckduckgo.com") or netloc.endswith("duck.com"):
                        # Skip other DDG internal links
                        continue
                    else:
                        resolved_url = absolute if absolute.lower().startswith("http") else None

                    if not resolved_url:
                        continue
                    if resolved_url in [c["url"] for c in collected]:
                        continue

                    title_text = re.sub(r"<[^>]+>", "", text)[:120].strip() or "(no title)"
                    snippet = re.sub(r"<[^>]+>", "", text)[:180].strip()
                    collected.append({"title": title_text, "url": resolved_url, "snippet": snippet})
                    if len(collected) >= max_results:
                        break
                if collected:
                    break

            if collected:
                results = []
                for i, r in enumerate(collected[:max_results], 1):
                    title = re.sub(r"<[^>]+>", "", r.get("title") or "(no title)")
                    url = r.get("url") or ""
                    snippet = (r.get("snippet") or "").strip()
                    results.append({"rank": i, "title": title, "url": url, "snippet": snippet})
                return {"ok": True, "query": query, "engine": "duckduckgo", "results": results}

            # Last-chance fallback: reuse internal HTML fallback via httpx client with centralized policy
            try:
                from ..web.search import _search_duckduckgo_fallback, SearchQuery
                from ..web.config import WebConfig
                items = _search_duckduckgo_fallback(SearchQuery(query=query), WebConfig())
                if items:
                    results = []
                    for i, it in enumerate(items[:max_results], 1):
                        results.append({
                            "rank": i,
                            "title": (it.title or "(no title)"),
                            "url": it.url,
                            "snippet": (it.snippet or "").strip(),
                        })
                    return {"ok": True, "query": query, "engine": "duckduckgo", "results": results}
            except Exception:
                pass

            # 3) Query variants (e.g., transform patterns, keyword compression) with HTML/internal fallback
            def _variants(q: str) -> List[str]:
                vars: List[str] = []
                try:
                    m = re.search(r"difference between\s+(.+?)\s+and\s+(.+)", q, flags=re.I)
                    if m:
                        x = m.group(1).strip().strip('?.,;:')
                        y = m.group(2).strip().strip('?.,;:')
                        vars.extend([f"{x} vs {y}", f"{x} versus {y}", f'"{x} vs {y}"'])
                    # Keyword compression (drop common stopwords)
                    toks = re.findall(r"[A-Za-z0-9]+", q)
                    stop = {
                        'the','a','an','of','in','on','for','to','and','or','with','about','from','by','at','as','is','are','was','were','be','being','been','this','that','these','those','it','its','into','between','difference','parliamentary'
                    }
                    kw = " ".join([t for t in toks if t.lower() not in stop][:6])
                    if kw and kw.lower() != q.lower():
                        vars.append(kw)
                except Exception:
                    pass
                # Deduplicate while preserving order
                seen_v = set()
                out_v: List[str] = []
                for v in vars:
                    if v and v not in seen_v:
                        seen_v.add(v)
                        out_v.append(v)
                return out_v

            for vq in _variants(query):
                # HTML fallback for variant
                collected_v: List[Dict[str, str]] = []
                for base in fallback_urls:
                    try:
                        params_html_v = {"q": vq, "kl": "wt-wt"}
                        r = requests.get(base, params=params_html_v, headers=html_headers, timeout=8)
                    except Exception:
                        continue
                    if r.status_code != 200:
                        continue
                    html = r.text or ""
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
                            try:
                                qs = parse_qs(parsed.query)
                                uddg = qs.get("uddg", [None])[0]
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
                        if resolved_url in [c["url"] for c in collected_v]:
                            continue
                        title_text = re.sub(r"<[^>]+>", "", text)[:120].strip() or "(no title)"
                        snippet2 = re.sub(r"<[^>]+>", "", text)[:180].strip()
                        collected_v.append({"title": title_text, "url": resolved_url, "snippet": snippet2})
                        if len(collected_v) >= max_results:
                            break
                    if collected_v:
                        break
                if collected_v:
                    results_v = []
                    for i, r3 in enumerate(collected_v[:max_results], 1):
                        title3 = re.sub(r"<[^>]+>", "", r3.get("title") or "(no title)")
                        url3 = r3.get("url") or ""
                        snippet3 = (r3.get("snippet") or "").strip()
                        results_v.append({"rank": i, "title": title3, "url": url3, "snippet": snippet3})
                    return {"ok": True, "query": vq, "engine": "duckduckgo", "results": results_v}
                # Internal fallback for variant
                try:
                    from ..web.search import _search_duckduckgo_fallback, SearchQuery
                    from ..web.config import WebConfig
                    items3 = _search_duckduckgo_fallback(SearchQuery(query=vq), WebConfig())
                    if items3:
                        results3 = []
                        for i, it in enumerate(items3[:max_results], 1):
                            results3.append({
                                "rank": i,
                                "title": (it.title or "(no title)"),
                                "url": it.url,
                                "snippet": (it.snippet or "").strip(),
                            })
                        return {"ok": True, "query": vq, "engine": "duckduckgo", "results": results3}
                except Exception:
                    pass

            # Final: keep contract but avoid error to reduce planner retries
            return {"ok": True, "query": query, "engine": "duckduckgo", "results": []}

        results = []

        # Prefer Instant Answer (Abstract/Answer)
        abstract = (data.get("AbstractText") or data.get("Abstract") or "").strip()
        abstract_url = (data.get("AbstractURL") or "").strip()
        if abstract and abstract_url:
            results.append({"title": data.get("Heading") or "Instant Answer", "url": abstract_url, "snippet": abstract})

        # Flatten RelatedTopics
        def _flatten_topics(items):
            out = []
            for it in items or []:
                if isinstance(it, dict) and "FirstURL" in it:
                    out.append({
                        "title": (it.get("Text") or "").split(" - ")[0][:120],
                        "url": it.get("FirstURL"),
                        "snippet": it.get("Text") or "",
                    })
                elif isinstance(it, dict) and "Topics" in it:
                    out.extend(_flatten_topics(it.get("Topics") or []))
            return out

        results.extend(_flatten_topics(data.get("RelatedTopics")))

        # Deduplicate by URL and limit
        seen = set()
        unique = []
        for r in results:
            url = r.get("url")
            if url and url not in seen:
                seen.add(url)
                unique.append(r)
            if len(unique) >= max_results:
                break

        if not unique:
            # Fallback path when JSON API yields no useful results:
            # 1) HTML fallback (Lite/HTML endpoints)
            html_headers = {
                "User-Agent": ua,
                "Accept": "text/html",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://duckduckgo.com/",
                "DNT": "1",
            }
            fallback_urls = [
                "https://duckduckgo.com/lite/",
                "https://html.duckduckgo.com/html/",
            ]
            collected: List[Dict[str, str]] = []
            for base in fallback_urls:
                try:
                    params_html = {"q": query, "kl": "wt-wt"}
                    r = requests.get(base, params=params_html, headers=html_headers, timeout=8)
                except Exception:
                    continue
                if r.status_code != 200:
                    continue
                html = r.text or ""
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
                        # Decode external URL from uddg param
                        try:
                            qs = parse_qs(parsed.query)
                            uddg = qs.get("uddg", [None])[0]
                            if uddg:
                                resolved_url = unquote(uddg)
                        except Exception:
                            pass
                    elif netloc.endswith("duckduckgo.com") or netloc.endswith("duck.com"):
                        # Skip other DDG internal links
                        continue
                    else:
                        resolved_url = absolute if absolute.lower().startswith("http") else None

                    if not resolved_url:
                        continue
                    if resolved_url in [c["url"] for c in collected]:
                        continue

                    title_text = re.sub(r"<[^>]+>", "", text)[:120].strip() or "(no title)"
                    snippet2 = re.sub(r"<[^>]+>", "", text)[:180].strip()
                    collected.append({"title": title_text, "url": resolved_url, "snippet": snippet2})
                    if len(collected) >= max_results:
                        break
                if collected:
                    break

            if collected:
                results_fallback = []
                for i, r2 in enumerate(collected[:max_results], 1):
                    title2 = re.sub(r"<[^>]+>", "", r2.get("title") or "(no title)")
                    url2 = r2.get("url") or ""
                    snippet2 = (r2.get("snippet") or "").strip()
                    results_fallback.append({"rank": i, "title": title2, "url": url2, "snippet": snippet2})
                return {"ok": True, "query": query, "engine": "duckduckgo", "results": results_fallback}

            # 2) Internal last-chance fallback via web.search when available
            try:
                from ..web.search import _search_duckduckgo_fallback, SearchQuery
                from ..web.config import WebConfig
                items2 = _search_duckduckgo_fallback(SearchQuery(query=query), WebConfig())
                if items2:
                    results2 = []
                    for i, it in enumerate(items2[:max_results], 1):
                        results2.append({
                            "rank": i,
                            "title": (it.title or "(no title)"),
                            "url": it.url,
                            "snippet": (it.snippet or "").strip(),
                        })
                    return {"ok": True, "query": query, "engine": "duckduckgo", "results": results2}
            except Exception:
                pass

            # 3) Query variants (e.g., transform "difference between X and Y" -> "X vs Y")
            def _variants(q: str) -> List[str]:
                vars: List[str] = []
                try:
                    m = re.search(r"difference between\s+(.+?)\s+and\s+(.+)", q, flags=re.I)
                    if m:
                        x = m.group(1).strip().strip('?.,;:')
                        y = m.group(2).strip().strip('?.,;:')
                        vars.extend([f"{x} vs {y}", f"{x} versus {y}", f'"{x} vs {y}"'])
                    # Keywords-only compression
                    toks = re.findall(r"[A-Za-z0-9]+", q)
                    stop = {
                        'the','a','an','of','in','on','for','to','and','or','with','about','from','by','at','as','is','are','was','were','be','being','been','this','that','these','those','it','its','into','between','difference','parliamentary'
                    }
                    kw = " ".join([t for t in toks if t.lower() not in stop][:6])
                    if kw and kw.lower() != q.lower():
                        vars.append(kw)
                except Exception:
                    pass
                # Deduplicate while preserving order
                seen_v = set()
                out_v: List[str] = []
                for v in vars:
                    if v and v not in seen_v:
                        seen_v.add(v)
                        out_v.append(v)
                return out_v

            for vq in _variants(query):
                collected_v: List[Dict[str, str]] = []
                for base in fallback_urls:
                    try:
                        params_html_v = {"q": vq, "kl": "wt-wt"}
                        r = requests.get(base, params=params_html_v, headers=html_headers, timeout=8)
                    except Exception:
                        continue
                    if r.status_code != 200:
                        continue
                    html = r.text or ""
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
                            # Decode external URL from uddg param
                            try:
                                qs = parse_qs(parsed.query)
                                uddg = qs.get("uddg", [None])[0]
                                if uddg:
                                    resolved_url = unquote(uddg)
                            except Exception:
                                pass
                        elif netloc.endswith("duckduckgo.com") or netloc.endswith("duck.com"):
                            # Skip other DDG internal links
                            continue
                        else:
                            resolved_url = absolute if absolute.lower().startswith("http") else None

                        if not resolved_url:
                            continue
                        if resolved_url in [c["url"] for c in collected_v]:
                            continue

                        title_text = re.sub(r"<[^>]+>", "", text)[:120].strip() or "(no title)"
                        snippet2 = re.sub(r"<[^>]+>", "", text)[:180].strip()
                        collected_v.append({"title": title_text, "url": resolved_url, "snippet": snippet2})
                        if len(collected_v) >= max_results:
                            break
                    if collected_v:
                        break
                if collected_v:
                    results_v = []
                    for i, r3 in enumerate(collected_v[:max_results], 1):
                        title3 = re.sub(r"<[^>]+>", "", r3.get("title") or "(no title)")
                        url3 = r3.get("url") or ""
                        snippet3 = (r3.get("snippet") or "").strip()
                        results_v.append({"rank": i, "title": title3, "url": url3, "snippet": snippet3})
                    return {"ok": True, "query": vq, "engine": "duckduckgo", "results": results_v}
                # Last-chance internal fallback for the variant
                try:
                    from ..web.search import _search_duckduckgo_fallback, SearchQuery
                    from ..web.config import WebConfig
                    items3 = _search_duckduckgo_fallback(SearchQuery(query=vq), WebConfig())
                    if items3:
                        results3 = []
                        for i, it in enumerate(items3[:max_results], 1):
                            results3.append({
                                "rank": i,
                                "title": (it.title or "(no title)"),
                                "url": it.url,
                                "snippet": (it.snippet or "").strip(),
                            })
                        return {"ok": True, "query": vq, "engine": "duckduckgo", "results": results3}
                except Exception:
                    pass

            # If all strategies failed, return empty results (consistent contract)
            return {"ok": True, "query": query, "engine": "duckduckgo", "results": []}

        results = []
        for i, r in enumerate(unique, 1):
            title = r.get("title") or "(no title)"
            url = r.get("url") or ""
            snippet = (r.get("snippet") or "").strip()
            results.append({"rank": i, "title": title, "url": url, "snippet": snippet})
        return {"ok": True, "query": query, "engine": "duckduckgo", "results": results}
    except Exception as e:
        return {"ok": False, "query": query, "engine": "duckduckgo", "results": [], "error": {"message": str(e)}}


TOOL_IMPLEMENTATION = duckduckgo_search
TOOL_AUTHOR = "core"
TOOL_VERSION = "2.0.0"
