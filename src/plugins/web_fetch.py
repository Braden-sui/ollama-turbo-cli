"""Web fetch tool plugin wrapping src.tools.web_fetch"""
from __future__ import annotations

from ..tools import web_fetch as _impl

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch live web content from a given URL (GET/HEAD). Returns status, headers, and a body snippet or JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL to fetch"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "HEAD"],
                    "default": "GET"
                },
                "params": {
                    "type": "object",
                    "description": "Optional query parameters as key/value"
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers as key/value"
                },
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds (1-60)",
                    "default": 10
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Maximum number of body characters to return (256-1048576)",
                    "default": 8192
                },
                "allow_redirects": {
                    "type": "boolean",
                    "description": "Whether to follow redirects",
                    "default": True
                },
                "as_json": {
                    "type": "boolean",
                    "description": "If true, try to parse the response as JSON and return it",
                    "default": False
                }
            },
            "required": ["url"]
        }
    }
}

TOOL_IMPLEMENTATION = _impl
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
