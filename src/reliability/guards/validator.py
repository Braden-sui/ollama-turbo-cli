from __future__ import annotations

from typing import List, Dict, Any


class Validator:
    """Simple guard stub to check citation presence and unsupported claims.

    Modes: off | warn | enforce (behavior left to caller; this stub only reports).
    """

    def __init__(self, mode: str = "off") -> None:
        self.mode = mode if mode in {"off", "warn", "enforce"} else "off"

    def validate(self, answer: str, context_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a report dict with minimal fields used by the client/UI."""
        _ = context_blocks
        return {
            "mode": self.mode,
            "status": "ok",
            "citations_present": "[" in answer and "]" in answer,
            "unsupported_claims": [],
        }
