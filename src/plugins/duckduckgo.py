"""DuckDuckGo search plugin wrapping src.tools.duckduckgo_search"""
from __future__ import annotations

from ..tools import duckduckgo_search as _impl

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "duckduckgo_search",
        "description": "Search the web using DuckDuckGo Instant Answer API (no API key). Returns top results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Number of results to return (1-5)", "default": 3}
            },
            "required": ["query"]
        }
    }
}

TOOL_IMPLEMENTATION = _impl
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
