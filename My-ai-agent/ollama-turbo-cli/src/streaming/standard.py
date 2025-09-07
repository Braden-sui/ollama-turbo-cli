"""
Standard (non-streaming) chat path scaffolding.

Phase C extracted `_handle_standard_chat` here without changing behavior.
Phase G delegates to `ChatTurnOrchestrator` to keep logic centralized.
"""

from typing import Any, Dict, List
from ..orchestration.context import OrchestrationContext

from ..orchestration.orchestrator import ChatTurnOrchestrator


def handle_standard_chat(ctx: OrchestrationContext, *, _suppress_errors: bool = False) -> str:
    """Handle non-streaming chat interaction (delegates to orchestrator)."""
    orch = ChatTurnOrchestrator()
    return orch.handle_standard_chat(ctx, _suppress_errors=_suppress_errors)
