"""Web fetch tool plugin"""
from __future__ import annotations
import json
from typing import Any, Dict, Optional

from ..sandbox.net_proxy import fetch_via_policy
from ..web.pipeline import _DEFAULT_CFG
from ..web.config import WebConfig

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Fetch a single HTTPS URL through the agent's sandboxed web client. "
            "Behavior is policy‑aware: by default cfg enables a liberal allowlist ('*') and proxies are permitted, "
            "but explicit SANDBOX_* environment variables may override this at runtime. Private/loopback IPs are always blocked. "
            "Use this to read a specific page or small API response (no credentials). Prefer HTTPS, short timeouts, and small max_bytes; "
            "only a compact summary ('inject') is added to context and may be truncated. If access is blocked by policy, report that briefly instead of retrying."
        ),
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
    # Prefer centralized cfg when set by the client; if explicit sandbox policy envs are present,
    # defer to environment by passing cfg=None to preserve test/user overrides.
    import os as _os
    _policy_env_present = any((_os.getenv(k) is not None) for k in (
        'SANDBOX_NET','SANDBOX_NET_ALLOW','SANDBOX_NET_BLOCK','SANDBOX_ALLOW_HTTP','SANDBOX_ALLOW_PROXIES','SANDBOX_BLOCK_PRIVATE_IPS'
    ))
    cfg = None if _policy_env_present else (_DEFAULT_CFG or WebConfig())
    res = fetch_via_policy(
        url,
        method=method or 'GET',
        headers=headers or {},
        body=data,
        timeout_s=int(timeout_s),
        max_bytes=max_bytes,
        cache_bypass=bool(cache_bypass),
        cfg=cfg,
    )

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
