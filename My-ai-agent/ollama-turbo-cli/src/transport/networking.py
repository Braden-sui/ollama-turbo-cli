from __future__ import annotations

"""
Networking helpers for host resolution, keep-alive policy, and idempotency headers.

Phase B will move implementations from `src/client.py` here. This file contains
extracted implementations used by the client. Behavior is preserved.
"""

from typing import Any, Callable, Dict, Optional, Union
import os
import re
import logging


class IdempotencyHeaders:
    """
    Helper to manage an Idempotency-Key header lifecycle per turn.
    Not wired yet; mirrors the behavior in `OllamaTurboClient`.
    """

    def __init__(self) -> None:
        self._key: Optional[str] = None

    def set(self, key: Optional[str]) -> None:
        self._key = str(key) if key else None

    def clear(self) -> None:
        self._key = None

    def as_headers(self) -> Dict[str, str]:
        return {"Idempotency-Key": self._key} if self._key else {}


# ----------------------- Extracted helpers -----------------------

def set_idempotency_key(client: Any, key: Optional[str], trace_hook: Optional[Callable[[str], None]] = None) -> None:
    """Set Idempotency-Key header on the underlying HTTP client.

    Matches behavior of `OllamaTurboClient._set_idempotency_key`.
    """
    try:
        if not key:
            return
        try:
            if getattr(client, "_client", None) and getattr(client._client, "headers", None):  # type: ignore[attr-defined]
                client._client.headers["Idempotency-Key"] = key  # type: ignore[attr-defined]
        except Exception:
            pass
        if trace_hook:
            try:
                trace_hook(f"idempotency:set {key}")
            except Exception:
                pass
    except Exception:
        pass


def clear_idempotency_key(client: Any) -> None:
    """Remove Idempotency-Key header after request completion.

    Matches behavior of `OllamaTurboClient._clear_idempotency_key`.
    """
    try:
        try:
            if client and getattr(client, "_client", None):
                client._client.headers.pop("Idempotency-Key", None)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def resolve_host(engine: Optional[str]) -> str:
    """Resolve the Ollama host based on engine flag or env.

    Priority:
      1) Explicit --engine flag
         - 'cloud' -> https://ollama.com
         - 'local' -> http://localhost:11434
         - Full URL (http/https) -> use as-is
         - Bare hostname -> prefix with https://
      2) OLLAMA_HOST env var
      3) Default https://ollama.com
    """
    try:
        if engine:
            e = engine.strip()
            el = e.lower()
            if el in {"cloud", "default"}:
                return "https://ollama.com"
            if el == "local":
                return "http://localhost:11434"
            if e.startswith("http://") or e.startswith("https://"):
                return e
            # Fallback: treat as hostname
            return f"https://{e}"
        host = os.getenv("OLLAMA_HOST")
        if host and str(host).strip() != "":
            return str(host).strip()
        return "https://ollama.com"
    except Exception:
        return "https://ollama.com"


def resolve_keep_alive(
    *,
    warm_models: bool,
    host: Optional[str],
    keep_alive_raw: Optional[Union[str, float, int]],
    logger: Optional[logging.Logger] = None,
) -> Optional[Union[float, str]]:
    """Resolve a valid keep_alive value or None.

    Accepts env `OLLAMA_KEEP_ALIVE`-like raw values:
    - duration string with units, e.g., '10m', '1h', '30s'
    - numeric seconds (int/float), converted to '<seconds>s'
    If unset and warming is enabled, defaults to '10m'.

    Suppresses keep-alive when using Ollama Cloud.
    """
    try:
        if not warm_models:
            return None
        try:
            if host and ("ollama.com" in host):
                return None
        except Exception:
            pass
        raw = keep_alive_raw
        if raw is None or str(raw).strip() == "":
            return "10m"
        s = str(raw).strip()
        # If purely numeric (int/float), treat as seconds
        if re.fullmatch(r"\d+(?:\.\d+)?", s):
            if "." in s:
                return f"{float(s)}s"
            return f"{int(s)}s"
        # If duration with unit
        if re.fullmatch(r"\d+(?:\.\d+)?[smhdw]", s, flags=0):
            return s
        if logger is not None:
            try:
                logger.debug(f"Invalid OLLAMA_KEEP_ALIVE '{s}', falling back to 10m")
            except Exception:
                pass
        return "10m"
    except Exception:
        return "10m"
