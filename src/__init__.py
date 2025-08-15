"""
Ollama Turbo CLI - A production-ready CLI application for gpt-oss:120b via Ollama Turbo.
"""

__version__ = "1.0.0"
__author__ = "Ollama Turbo CLI Team"

from .client import OllamaTurboClient
from .tools import (
    get_current_weather,
    calculate_math,
    list_files,
    get_system_info,
    duckduckgo_search,
    wikipedia_search,
)

__all__ = [
    "OllamaTurboClient",
    "get_current_weather",
    "calculate_math",
    "list_files",
    "get_system_info",
    "duckduckgo_search",
    "wikipedia_search",
]
