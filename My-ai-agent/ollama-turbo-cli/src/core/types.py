from __future__ import annotations

"""
Thin DTOs and typed aliases used by the modularized client orchestrator.

Note: The existing `src/types.py` remains the source of truth for adapter-
level types. These DTOs are intentionally minimal and decoupled to avoid
cycles and keep boundaries clean.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Union, Literal


# ----------------------- Tool call/result DTOs -----------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Union[str, Dict[str, Any]]


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    ok: bool = True
    sensitive: bool = False
    truncated: bool = False
    metadata: Optional[Dict[str, Any]] = None


# ----------------------- Streaming event shapes -----------------------

class StreamToken(TypedDict):
    type: Literal["token"]
    content: str


class StreamFinal(TypedDict, total=False):
    type: Literal["final"]
    content: str
    tool_calls: List[Dict[str, Any]]
    finish_reason: Optional[str]


class StreamSummary(TypedDict, total=False):
    type: Literal["summary"]
    citations: Optional[List[Dict[str, Any]]]
    consensus: Optional[Dict[str, Any]]
    validator: Optional[Dict[str, Any]]


StreamEvent = Union[StreamToken, StreamFinal, StreamSummary]


__all__ = [
    "ToolCall",
    "ToolResult",
    "StreamToken",
    "StreamFinal",
    "StreamSummary",
    "StreamEvent",
]
