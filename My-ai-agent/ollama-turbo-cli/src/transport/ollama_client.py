from __future__ import annotations

"""
A minimal wrapper for `ollama.Client` with explicit timeouts.

Phase A scaffolding; actual behavior remains in `src/client.py` until extraction.
"""

from typing import Any, Dict, Optional


class OllamaClient:
    def __init__(
        self,
        *,
        host: str,
        headers: Optional[Dict[str, str]] = None,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 600.0,
    ) -> None:
        self.host = host
        self.headers = dict(headers or {})
        self.connect_timeout_s = float(connect_timeout_s)
        self.read_timeout_s = float(read_timeout_s)

    def chat(self, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - placeholder
        raise NotImplementedError("OllamaClient.chat() will delegate to ollama.Client in a later phase")
