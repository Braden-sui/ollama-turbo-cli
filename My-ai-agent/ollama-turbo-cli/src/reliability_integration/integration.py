from __future__ import annotations

"""
Reliability facade scaffolding.

Phase F will move `_prepare_reliability_context` and `_load_system_cited` here
with a small wrapper to the existing reliability modules.
"""

from typing import Any, Dict, List, Optional


class ReliabilityFacade:
    def __init__(self) -> None:  # pragma: no cover - placeholder
        pass

    def prepare_context(self, conversation: List[Dict[str, Any]], user_message: str, opts: Dict[str, Any]) -> List[Dict[str, Any]]:  # pragma: no cover - placeholder
        raise NotImplementedError("Will be implemented in Phase F")

    def validate(self, final_text: str, context_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:  # pragma: no cover - placeholder
        raise NotImplementedError("Will be implemented in Phase F")
