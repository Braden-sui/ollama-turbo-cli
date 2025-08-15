"""
Ollama Turbo client - Clean hexagonal architecture implementation.
Legacy compatibility maintained through facade pattern.
"""

from __future__ import annotations

# Export for backward compatibility
__all__ = ['OllamaTurboClient']


class OllamaTurboClient:  # type: ignore[override]
    """Lazy proxy that defers importing the heavy legacy facade until first use.

    This preserves import-time lightness so tests importing `src.client` don't
    trigger configuration or third-party imports prematurely.
    """

    def __new__(cls, *args, **kwargs):
        try:
            from .legacy.client_facade import OllamaTurboClient as _Real
        except Exception:
            # Fallback if imported as a script without package context
            import os, sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from legacy.client_facade import OllamaTurboClient as _Real  # type: ignore
        instance = _Real(*args, **kwargs)
        return instance


def __getattr__(name: str):  # pragma: no cover
    if name == 'OllamaTurboClient':
        return OllamaTurboClient
    raise AttributeError(name)
