from __future__ import annotations

from typing import Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field


class ChatOptions(BaseModel):
    tool_results_format: Literal["string", "object"] = Field(
        default="object",
        description="How to return tool results: legacy 'string' for backward-compat, or 'object' for structured results.",
    )
    # Optional web retrieval profile mapping (Commit 1 – non-breaking)
    web_profile: Optional[Literal['quick', 'balanced', 'rigorous']] = Field(
        default=None,
        description="Optional retrieval profile: quick|balanced|rigorous. Defaults to rigorous behavior (unchanged).",
    )


class ChatRequest(BaseModel):
    message: str
    options: Optional[ChatOptions] = None
    # Reliability mode flags (optional; default to no-op behavior)
    ground: Optional[bool] = Field(default=False, description="Enable retrieval + grounding context injection")
    k: Optional[int] = Field(default=None, description="General 'k' parameter: retrieval top-k and/or consensus runs")
    cite: Optional[bool] = Field(default=False, description="If true, instruct model to include inline citations when grounded context exists")
    check: Optional[Literal['off', 'warn', 'enforce']] = Field(default='off', description="Validator mode: off|warn|enforce")
    consensus: Optional[bool] = Field(default=False, description="Enable multi-run consensus voting (non-streaming replaces final; streaming is trace-only)")
    engine: Optional[str] = Field(default=None, description="Backend engine host alias or URL to target")
    eval_corpus: Optional[str] = Field(default=None, description="Optional eval corpus identifier for micro-eval harness")


class ToolError(BaseModel):
    code: Optional[str] = None
    message: str
    details: Optional[dict] = None


class ToolResultObject(BaseModel):
    tool: str
    status: Literal["ok", "error"] = "ok"
    content: Optional[Any] = None
    metadata: Optional[dict] = None
    error: Optional[ToolError] = None


ToolResult = Union[str, ToolResultObject]


class ChatResponse(BaseModel):
    content: str
    tool_results: Optional[List[ToolResult]] = None


class ErrorResponse(BaseModel):
    error: str
    code: Optional[str] = None
