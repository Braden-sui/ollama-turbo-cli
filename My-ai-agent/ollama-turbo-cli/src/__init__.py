"""
Ollama Turbo CLI - A production-ready CLI application for gpt-oss:120b via Ollama Turbo.
"""

__version__ = "1.1.0"
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
# Ensure subpackage re-export so `import src.plugins.web_research` works
from . import plugins as plugins

__all__ = [
    "OllamaTurboClient",
    "get_current_weather",
    "calculate_math",
    "list_files",
    "get_system_info",
    "duckduckgo_search",
    "wikipedia_search",
    "plugins",
]
