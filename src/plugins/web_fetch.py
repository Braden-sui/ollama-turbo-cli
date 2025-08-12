"""Web fetch tool plugin"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

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

def web_fetch(
    url: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10,
    max_bytes: int = 8192,
    allow_redirects: bool = True,
    as_json: bool = False,
) -> str:
    """Fetch live web content from an HTTP/HTTPS URL.

    - Supports GET/HEAD with optional params and headers
    - Returns status, final URL, content type, and a truncated body snippet
    - If as_json is True, attempts to parse and return JSON
    """
    try:
        if not requests:
            return "Error: Python 'requests' library is not installed. Install it to use web_fetch."

        url = (url or "").strip()
        if not url:
            return "Error: url must be provided"
        if not (url.startswith("http://") or url.startswith("https://")):
            return "Error: url must start with http:// or https://"

        method_u = (method or "GET").upper()
        if method_u not in ("GET", "HEAD"):
            return "Error: method must be 'GET' or 'HEAD'"

        # Validate numeric bounds
        try:
            timeout = float(timeout)
            if timeout < 1 or timeout > 60:
                return "Error: timeout must be between 1 and 60 seconds"
        except Exception:
            return "Error: timeout must be a number"

        try:
            max_bytes = int(max_bytes)
            if max_bytes < 256 or max_bytes > 1048576:
                return "Error: max_bytes must be between 256 and 1048576"
        except Exception:
            return "Error: max_bytes must be an integer"

        # Merge headers with a sensible User-Agent
        hdrs = {
            "User-Agent": "ollama-turbo-cli/1.0 (+https://ollama.com)",
        }
        if isinstance(headers, dict):
            # Keep only str->str
            for k, v in headers.items():
                try:
                    hdrs[str(k)] = str(v)
                except Exception:
                    continue

        try:
            resp = requests.request(
                method=method_u,
                url=url,
                params=params if isinstance(params, dict) else None,
                headers=hdrs,
                timeout=timeout,
                allow_redirects=bool(allow_redirects),
            )
        except Exception as e:  # requests.RequestException
            return f"Error fetching URL '{url}': network error: {e}"

        status = resp.status_code
        final_url = resp.url
        ctype = resp.headers.get("Content-Type", "")

        # HEAD has no body
        if method_u == "HEAD":
            return (
                f"HTTP {status}\n"
                f"Final URL: {final_url}\n"
                f"Content-Type: {ctype or 'unknown'}\n"
                f"Note: HEAD request has no body"
            )

        if as_json:
            try:
                data = resp.json()
                # Truncate JSON string representation if large
                body_str = json.dumps(data, ensure_ascii=False)[:max_bytes]
                if len(json.dumps(data, ensure_ascii=False)) > max_bytes:
                    body_str += f"... [truncated]"
                return (
                    f"HTTP {status}\n"
                    f"Final URL: {final_url}\n"
                    f"Content-Type: {ctype or 'unknown'}\n"
                    f"--- JSON Body (first {max_bytes} chars) ---\n"
                    f"{body_str}"
                )
            except ValueError:
                return (
                    f"HTTP {status}\n"
                    f"Final URL: {final_url}\n"
                    f"Content-Type: {ctype or 'unknown'}\n"
                    f"Error: Response is not valid JSON"
                )

        # Text mode
        body = resp.text or ""
        if len(body) > max_bytes:
            body = body[:max_bytes] + "... [truncated]"

        return (
            f"HTTP {status}\n"
            f"Final URL: {final_url}\n"
            f"Content-Type: {ctype or 'unknown'}\n"
            f"--- Body (first {max_bytes} chars) ---\n"
            f"{body}"
        )
    except Exception as e:
        return f"Error fetching URL '{url}': {str(e)}"


TOOL_IMPLEMENTATION = web_fetch
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
