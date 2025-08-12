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


def web_fetch(
    url: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10,
    max_bytes: int = 8192,
    allow_redirects: bool = True,
    as_json: bool = False,
) -> str:  # type: ignore[override]
    return _plugin_fn("web_fetch")(
        url=url,
        method=method,
        params=params,
        headers=headers,
        timeout=timeout,
        max_bytes=max_bytes,
        allow_redirects=allow_redirects,
        as_json=as_json,
    )


def duckduckgo_search(query: str, max_results: int = 3) -> str:  # type: ignore[override]
    return _plugin_fn("duckduckgo_search")(query=query, max_results=max_results)


def wikipedia_search(query: str, limit: int = 3) -> str:  # type: ignore[override]
    return _plugin_fn("wikipedia_search")(query=query, limit=limit)
