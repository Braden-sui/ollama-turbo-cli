"""
Tool implementations and schemas for Ollama Turbo CLI.
"""

from typing import Dict, Any, Optional

__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_FUNCTIONS",
    "reload_plugins",
    "get_manager",
    "get_current_weather",
    "calculate_math",
    "list_files",
    "get_system_info",
    "web_fetch",
    "duckduckgo_search",
    "wikipedia_search",
]

# Legacy tool implementations removed — this module now serves as a thin
# backward-compatibility shim that re-exports dynamic plugin functions.

# Compatibility shim: re-export dynamic plugin aggregates
from . import plugin_loader as _plugin_loader

# Expose the plugin-managed tool schemas and functions for backward compatibility
TOOL_SCHEMAS = _plugin_loader.TOOL_SCHEMAS
TOOL_FUNCTIONS = _plugin_loader.TOOL_FUNCTIONS

# Convenience re-exports
reload_plugins = _plugin_loader.reload_plugins
get_manager = _plugin_loader.get_manager

def _plugin_fn(name: str):
    funcs = _plugin_loader.TOOL_FUNCTIONS
    if name not in funcs:
        raise KeyError(f"Tool '{name}' is not loaded")
    return funcs[name]

# Backward-compatible wrappers calling dynamic plugin implementations
def get_current_weather(city: str, unit: str = "celsius") -> str:  # type: ignore[override]
    return _plugin_fn("get_current_weather")(city=city, unit=unit)


def calculate_math(expression: str) -> str:  # type: ignore[override]
    return _plugin_fn("calculate_math")(expression=expression)


def list_files(directory: str = ".", extension: Optional[str] = None) -> str:  # type: ignore[override]
    return _plugin_fn("list_files")(directory=directory, extension=extension)


def get_system_info() -> str:  # type: ignore[override]
    return _plugin_fn("get_system_info")()


# Updated to match plugin schema while accepting legacy args for compatibility
# Plugin schema: url, method, headers, body, timeout_s, max_bytes, extract
# Legacy shim accepted: url, method, params, headers, timeout, max_bytes, allow_redirects, as_json
# We map: timeout -> timeout_s (if provided), ignore params/allow_redirects/as_json

def web_fetch(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout_s: int = 15,
    max_bytes: Optional[int] = None,
    extract: str = "auto",
    # legacy-only args (ignored):
    timeout: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
    allow_redirects: Optional[bool] = None,
    as_json: Optional[bool] = None,
) -> str:  # type: ignore[override]
    # Map legacy timeout -> timeout_s if explicitly provided
    if timeout is not None and timeout_s == 15:
        try:
            timeout_s = int(timeout)
        except Exception:
            pass
    # params/allow_redirects/as_json are intentionally ignored in the secure proxy path
    return _plugin_fn("web_fetch")(
        url=url,
        method=method,
        headers=headers,
        body=body,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        extract=extract,
    )


def duckduckgo_search(query: str, max_results: int = 3) -> str:  # type: ignore[override]
    return _plugin_fn("duckduckgo_search")(query=query, max_results=max_results)


def wikipedia_search(query: str, limit: int = 3) -> str:  # type: ignore[override]
    return _plugin_fn("wikipedia_search")(query=query, limit=limit)
