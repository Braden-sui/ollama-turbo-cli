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
        # Apply YearGuard to strip template-injected years/month-years without touching user quotes
        try:
            from ..web.pipeline import _DEFAULT_CFG  # type: ignore
            from ..web.search import _year_guard  # type: ignore
            cfg = _DEFAULT_CFG
            if cfg is not None:
                sanitized, _ = _year_guard(query, cfg)
                query = sanitized or query
        except Exception:
            pass
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

            # If still nothing useful, report the last status
            code = resp.status_code if (resp is not None and hasattr(resp, 'status_code')) else 'unknown'
            return {"ok": False, "query": query, "engine": "duckduckgo", "results": [], "error": {"message": f"HTTP {code} and no fallback results"}}

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
TOOL_VERSION = "1.0.0"
