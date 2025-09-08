"""Web fetch tool plugin"""
from __future__ import annotations
import json
from typing import Any, Dict, Optional

from ..sandbox.net_proxy import fetch_via_policy
from ..web.pipeline import _DEFAULT_CFG

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch web content through a sandboxed allowlist proxy with SSRF protection and size caps. Use when you need to read a specific page or API without credentials. Prefer HTTPS URLs, small timeouts, and minimal content. Only a compact summary ('inject') is added to model context and may be truncated; avoid fetching large pages.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "url": {"type": "string", "description": "Absolute HTTP(S) URL. Prefer HTTPS. Do not include credentials. Internal/private addresses are blocked by policy."},
                "method": {"type": "string", "default": "GET", "description": "HTTP method. Policy allows only GET/HEAD/POST; other methods will be rejected by the network policy layer."},
                "headers": {"type": "object", "description": "Optional request headers. Never include Authorization or cookies."},
                "body": {"type": "string", "description": "Optional request body, used only when method is POST."},
                "timeout_s": {"type": "integer", "default": 15, "maximum": 30, "description": "Request timeout in seconds (1-30). Use small values to reduce latency."},
                "max_bytes": {"type": "integer", "description": "Maximum bytes to download. Smaller caps reduce cost and truncation."},
                "extract": {"type": "string", "enum": ["auto", "text", "json", "html_readability", "headers", "bytes"], "default": "auto", "description": "How to extract the response. Prefer 'text' or 'json' for small responses; 'html_readability' extracts main content."},
                "cache_bypass": {"type": "boolean", "default": False, "description": "If true, skip cache reads and perform a fresh network fetch (subject to policy)."},
            },
            "required": ["url"],
        },
    },
}


def web_fetch(url: str, method: str = 'GET', headers: Optional[Dict[str, str]] = None, body: Optional[str] = None, timeout_s: int = 15, max_bytes: Optional[int] = None, extract: str = 'auto', cache_bypass: bool = False) -> str:
    # Enforce HTTPS default inside fetch_via_policy
    data = body.encode() if body is not None else None
    res = fetch_via_policy(url, method=method or 'GET', headers=headers or {}, body=data, timeout_s=int(timeout_s), max_bytes=max_bytes, cache_bypass=bool(cache_bypass), cfg=_DEFAULT_CFG)

    # Build safe injection summary; never include large bodies
    inject = res.get('inject') or ''
    out = {
        'tool': 'web_fetch',
        'ok': bool(res.get('ok')),
        'status': int(res.get('status', 0) or 0),
        'url': res.get('url', url),
        'bytes': int(res.get('bytes', 0)),
        'cached': bool(res.get('cached', False)),
        'redirects': int(res.get('redirects', 0)),
        'truncated': bool(res.get('truncated', False)),
        'inject': inject,
        'sensitive': _looks_sensitive(inject),
        'debug': res.get('debug', {}),
        'net': {
            'status': int(res.get('status', 0) or 0),
            'bytes': int(res.get('bytes', 0)),
            'cached': bool(res.get('cached', False)),
            'redirects': int(res.get('redirects', 0)),
            'url': res.get('url', url),
        }
    }
    if not res.get('ok') and res.get('error'):
        out['error'] = res.get('error')
    return json.dumps(out)


def _looks_sensitive(text: str) -> bool:
    import re
    return bool(re.search(r"(?i)(api[_-]?key|secret|token|authorization)\s*[:=]\s*([^\s'\"]+)", text or ''))

TOOL_IMPLEMENTATION = web_fetch
TOOL_AUTHOR = "core"
TOOL_VERSION = "2.0.0"
