"""
ToolExecutor scaffold.

Placeholder for future centralized tool execution logic. Not used by the
client yet; current execution stays in `OllamaTurboClient._execute_tool_calls`.
This class exists to satisfy imports and support incremental refactoring.
"""
from __future__ import annotations

from typing import Any, Dict, List


class ToolExecutor:
    """Minimal stub for future tool execution orchestration."""

    def __init__(self) -> None:
        pass

    def execute(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        """Placeholder execute path.
        Returns an empty list to preserve current behavior (no external usage yet).
        """
        return []
