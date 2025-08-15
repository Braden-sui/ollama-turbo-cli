"""
Ollama Turbo CLI - A production-ready CLI application for gpt-oss:120b via Ollama Turbo.
"""

__version__ = "1.0.0"
__author__ = "Ollama Turbo CLI Team"

__all__ = [
    "OllamaTurboClient",
    "get_current_weather",
    "calculate_math",
    "list_files",
    "get_system_info",
    "duckduckgo_search",
    "wikipedia_search",
]

# Lazy attribute access to avoid importing heavy modules at package import time.
# This keeps `import src.domain...` safe during test collection.
def __getattr__(name: str):  # pragma: no cover - simple lazy loader
    if name == "OllamaTurboClient":
        from .client import OllamaTurboClient as _C
        return _C
    if name in {
        "get_current_weather",
        "calculate_math",
        "list_files",
        "get_system_info",
        "duckduckgo_search",
        "wikipedia_search",
    }:
        from .tools import (
            get_current_weather,
            calculate_math,
            list_files,
            get_system_info,
            duckduckgo_search,
            wikipedia_search,
        )
        return {
            "get_current_weather": get_current_weather,
            "calculate_math": calculate_math,
            "list_files": list_files,
            "get_system_info": get_system_info,
            "duckduckgo_search": duckduckgo_search,
            "wikipedia_search": wikipedia_search,
        }[name]
    raise AttributeError(f"module 'src' has no attribute {name!r}")
