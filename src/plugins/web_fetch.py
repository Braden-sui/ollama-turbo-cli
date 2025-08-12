"""Web fetch tool plugin"""
from __future__ import annotations
import json
from typing import Any, Dict, Optional

from ..sandbox.net_proxy import fetch_via_policy

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch web content via a controlled allowlist proxy with SSRF protection and size caps.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "enum": ["GET", "HEAD", "POST"], "default": "GET"},
                "headers": {"type": "object"},
                "body": {"type": "string"},
                "timeout_s": {"type": "integer", "default": 15, "maximum": 30},
                "max_bytes": {"type": "integer"},
                "extract": {"type": "string", "enum": ["auto", "text", "json", "html_readability", "headers", "bytes"], "default": "auto"},
            },
            "required": ["url"],
        },
    },
}


def web_fetch(url: str, method: str = 'GET', headers: Optional[Dict[str, str]] = None, body: Optional[str] = None, timeout_s: int = 15, max_bytes: Optional[int] = None, extract: str = 'auto') -> str:
    # Enforce HTTPS default inside fetch_via_policy
    data = body.encode() if body is not None else None
    res = fetch_via_policy(url, method=method or 'GET', headers=headers or {}, body=data, timeout_s=int(timeout_s), max_bytes=max_bytes)

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
