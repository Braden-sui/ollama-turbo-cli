"""Wikipedia search plugin wrapping src.tools.wikipedia_search"""
from __future__ import annotations

from ..tools import wikipedia_search as _impl

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

TOOL_IMPLEMENTATION = _impl
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
