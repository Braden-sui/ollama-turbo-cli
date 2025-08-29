from __future__ import annotations

from typing import Dict, Any, Optional


class DocCache:
    """In-memory document cache stub keyed by normalized URL or source id."""

    def __init__(self) -> None:
        self._mem: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._mem.get(key)

    def set(self, key: str, value: Any) -> None:
        self._mem[key] = value
