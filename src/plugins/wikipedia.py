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
        "description": "Search Wikipedia and return top results with title, URL, and snippet (no API key).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Number of results to return (1-5)", "default": 3}
            },
            "required": ["query"]
        }
    }
}

def wikipedia_search(query: str, limit: int = 3) -> str:
    """Search Wikipedia (MediaWiki API) and return formatted results."""
    try:
        if not requests:
            return "Error: Python 'requests' library is not installed. Install it to use wikipedia_search."

        query = (query or "").strip()
        if not query:
            return "Error: query must be provided"

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
        headers = {"User-Agent": "ollama-turbo-cli/1.0 (+https://ollama.com)"}
        try:
            resp = requests.get("https://en.wikipedia.org/w/api.php", params=params, headers=headers, timeout=8)
        except Exception as e:
            return f"Error performing Wikipedia search: network error: {e}"

        if resp.status_code != 200:
            return f"Error performing Wikipedia search: HTTP {resp.status_code}"

        try:
            data = resp.json()
        except json.JSONDecodeError:
            return "Error performing Wikipedia search: invalid JSON response"

        search_results = (data.get("query") or {}).get("search") or []
        if not search_results:
            return f"Wikipedia: No results for '{query}'"

        def _strip_html(s: str) -> str:
            try:
                return re.sub(r"<[^>]+>", "", s)
            except Exception:
                return s

        lines = [f"Wikipedia: Top {min(len(search_results), limit)} results for '{query}':"]
        for i, item in enumerate(search_results[:limit], 1):
            title = item.get("title") or "(no title)"
            pageid = item.get("pageid")
            url = f"https://en.wikipedia.org/?curid={pageid}" if pageid else ""
            snippet = _strip_html(item.get("snippet") or "").replace("\n", " ")
            lines.append(f"{i}. {title} - {url}")
            if snippet:
                lines.append(f"   {snippet}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error performing Wikipedia search: {str(e)}"


TOOL_IMPLEMENTATION = wikipedia_search
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
