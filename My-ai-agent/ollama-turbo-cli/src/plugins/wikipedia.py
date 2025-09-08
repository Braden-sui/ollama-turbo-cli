"""Wikipedia search plugin providing MediaWiki API results (no API key)."""
from __future__ import annotations

import json
import re
from typing import Any

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wikipedia_search",
        "description": (
            "Keyless Wikipedia search via the MediaWiki API. Use for factual background and canonical topic pages. "
            "Provide a focused query (entity or concept). Returns concise top results (title, URL, snippet). "
            "To read the page content, call web_fetch on a returned URL. Subject to the agent's network policy; "
            "if access is blocked, acknowledge policy and propose alternative sources or queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused Wikipedia query (entity/term). Avoid full natural-language questions."},
                "limit": {"type": "integer", "description": "How many results to return (1-5). Choose the smallest number sufficient to answer.", "default": 3}
            },
            "required": ["query"]
        }
    }
}

def wikipedia_search(query: str, limit: int = 3):
    """Search Wikipedia (MediaWiki API) and return JSON.

    Shape:
      { ok: bool, query: str, results: [{rank,title,url,snippet}], engine: 'wikipedia', error?: {message}}
    """
    try:
        if not requests:
            return {"ok": False, "query": query, "engine": "wikipedia", "results": [], "error": {"message": "requests not installed"}}

        query = (query or "").strip()
        if not query:
            return {"ok": False, "query": query, "engine": "wikipedia", "results": [], "error": {"message": "query must be provided"}}

        try:
            limit = int(limit)
            if limit < 1:
                limit = 1
            if limit > 5:
                limit = 5
        except Exception:
            limit = 3

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": "1",
            "srlimit": str(limit),
        }
        # Use centralized WebConfig for user agent if available
        try:
            from ..web.pipeline import _DEFAULT_CFG
            user_agent = _DEFAULT_CFG.user_agent if _DEFAULT_CFG else "ollama-turbo-cli/1.0 (+https://ollama.com)"
        except Exception:
            user_agent = "ollama-turbo-cli/1.0 (+https://ollama.com)"
        
        headers = {"User-Agent": user_agent}
        try:
            resp = requests.get("https://en.wikipedia.org/w/api.php", params=params, headers=headers, timeout=8)
        except Exception as e:
            return {"ok": False, "query": query, "engine": "wikipedia", "results": [], "error": {"message": f"network error: {e}"}}

        if resp.status_code != 200:
            return {"ok": False, "query": query, "engine": "wikipedia", "results": [], "error": {"message": f"HTTP {resp.status_code}"}}

        try:
            data = resp.json()
        except json.JSONDecodeError:
            return {"ok": False, "query": query, "engine": "wikipedia", "results": [], "error": {"message": "invalid JSON response"}}

        search_results = (data.get("query") or {}).get("search") or []
        if not search_results:
            return {"ok": True, "query": query, "engine": "wikipedia", "results": []}

        def _strip_html(s: str) -> str:
            try:
                return re.sub(r"<[^>]+>", "", s)
            except Exception:
                return s

        results = []
        for i, item in enumerate(search_results[:limit], 1):
            title = item.get("title") or "(no title)"
            pageid = item.get("pageid")
            url = f"https://en.wikipedia.org/?curid={pageid}" if pageid else ""
            snippet = _strip_html(item.get("snippet") or "").replace("\n", " ")
            results.append({"rank": i, "title": title, "url": url, "snippet": snippet})
        return {"ok": True, "query": query, "engine": "wikipedia", "results": results}
    except Exception as e:
        return {"ok": False, "query": query, "engine": "wikipedia", "results": [], "error": {"message": str(e)}}


TOOL_IMPLEMENTATION = wikipedia_search
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
