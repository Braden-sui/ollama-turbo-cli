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
        "description": "Keyless web search via DuckDuckGo Instant Answer API. Use to discover sources or quick facts when external information is required. Provide a focused query (keywords or quoted phrase). The tool returns concise top results (title, URL, snippet). Choose a small max_results (1-5) to minimize noise.",
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

def duckduckgo_search(query: str, max_results: int = 3) -> str:
    """Search using DuckDuckGo Instant Answer API (no API key required).

    Returns top results with title, URL, and snippet when available.
    """
    try:
        if not requests:
            return "Error: Python 'requests' library is not installed. Install it to use duckduckgo_search."

        query = (query or "").strip()
        if not query:
            return "Error: query must be provided"

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
        headers = {
            "User-Agent": "ollama-turbo-cli/1.0 (+https://ollama.com)",
            "Accept": "application/json",
        }

        # Try API with small retries for transient statuses (e.g., 202, 429, 5xx)
        resp = None
        for attempt in range(5):
            try:
                resp = requests.get("https://api.duckduckgo.com/", params=params, headers=headers, timeout=8)
            except Exception as e:
                if attempt == 4:
                    return f"Error performing DuckDuckGo search: network error: {e}"
                time.sleep(0.5 * (2 ** attempt))
                continue
            if resp.status_code == 200:
                break
            if attempt < 4 and resp.status_code in (202, 429, 403, 500, 502, 503, 504):
                time.sleep(0.5 * (2 ** attempt))
                continue
            else:
                break

        if resp is not None and resp.status_code == 200:
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return "Error performing DuckDuckGo search: invalid JSON response"
        else:
            # Fallback to HTML (Lite) endpoint and extract links
            html_headers = {
                "User-Agent": "ollama-turbo-cli/1.0 (+https://ollama.com)",
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
                    r = requests.get(base, params={"q": query}, headers=html_headers, timeout=8)
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
                out_lines = [f"DuckDuckGo (HTML fallback): Top {len(collected)} results for '{query}':"]
                for i, r in enumerate(collected[:max_results], 1):
                    title = re.sub(r"<[^>]+>", "", r.get("title") or "(no title)")
                    url = r.get("url") or ""
                    snippet = r.get("snippet") or ""
                    out_lines.append(f"{i}. {title} - {url}")
                    if snippet:
                        out_lines.append(f"   {snippet}")
                return "\n".join(out_lines)
            # If still nothing useful, report the last status
            code = resp.status_code if (resp is not None and hasattr(resp, 'status_code')) else 'unknown'
            return f"Error performing DuckDuckGo search: HTTP {code} and no fallback results"

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
            return f"DuckDuckGo: No results for '{query}'"

        out_lines = [f"DuckDuckGo: Top {len(unique)} results for '{query}':"]
        for i, r in enumerate(unique, 1):
            title = r.get("title") or "(no title)"
            url = r.get("url") or ""
            snippet = r.get("snippet") or ""
            out_lines.append(f"{i}. {title} - {url}")
            if snippet:
                out_lines.append(f"   {snippet}")
        return "\n".join(out_lines)
    except Exception as e:
        return f"Error performing DuckDuckGo search: {str(e)}"


TOOL_IMPLEMENTATION = duckduckgo_search
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
