from __future__ import annotations

"""
Mem0 service scaffolding.

Phase E will extract initialization, context injection, and persistence here
with strict fidelity to existing environment semantics.
"""

from typing import Any, Dict, List, Optional


class Mem0Service:
    def __init__(self, config: Any) -> None:  # pragma: no cover - placeholder
        self.config = config

    def initialize(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError("Mem0Service.initialize() will be implemented in Phase E")

    def inject_context(self, conversation: List[Dict[str, Any]], user_message: str) -> List[Dict[str, Any]]:  # pragma: no cover - placeholder
        raise NotImplementedError("Mem0Service.inject_context() will be implemented in Phase E")

    def persist_turn(self, user_text: str, assistant_text: str) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError("Mem0Service.persist_turn() will be implemented in Phase E")
