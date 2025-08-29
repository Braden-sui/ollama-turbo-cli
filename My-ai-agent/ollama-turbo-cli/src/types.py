"""
Shared type definitions for the Ollama Turbo client protocol adapters.

This module centralizes data structures used across protocol implementations
(Harmony, DeepSeek, etc.) to avoid circular imports and keep adapters decoupled
from the core client.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

# Protocol selection
ProtocolName = Literal["harmony", "deepseek", "auto"]

# Chat roles
Role = Literal["system", "user", "assistant", "tool"]


class ToolParamSchema(TypedDict, total=False):
    """JSONSchema-like structure for tool parameters.

    We do not enforce a strict schema here to keep flexibility across models.
    """
    type: str
    properties: Dict[str, Any]
    required: List[str]
    additionalProperties: bool
    description: str


class ToolSpec(TypedDict):
    """Represents a callable tool exposed to the model."""
    name: str
    description: str
    parameters: ToolParamSchema


class ToolCall(TypedDict):
    """A single tool call emitted by a model (normalized)."""
    id: str
    name: str
    arguments: Union[str, Dict[str, Any]]  # raw JSON string or parsed dict


class ChatMessage(TypedDict, total=False):
    """A chat message following the common schema across adapters."""
    role: Role
    content: str
    name: str
    tool_call_id: str
    tool_calls: List[ToolCall]


# Streaming shapes expected by the backend SSE contract.
class StreamToken(TypedDict):
    type: Literal["token"]
    content: str


class StreamToolCall(TypedDict):
    type: Literal["tool_call"]
    id: str
    name: str
    arguments: Union[str, Dict[str, Any]]


class StreamFinal(TypedDict, total=False):
    type: Literal["final"]
    content: str
    tool_calls: List[ToolCall]
    finish_reason: Optional[str]


class StreamSummary(TypedDict, total=False):
    type: Literal["summary"]
    citations: Optional[List[Dict[str, Any]]]
    consensus: Optional[Dict[str, Any]]
    validator: Optional[Dict[str, Any]]


NormalizedStreamEvent = Union[StreamToken, StreamToolCall, StreamFinal, StreamSummary]


class NonStreamResponse(TypedDict, total=False):
    """Normalized shape for non-streaming responses returned by adapters."""
    content: str
    tool_calls: List[ToolCall]
    usage: Dict[str, Any]
    raw: Any  # original provider payload for debugging/telemetry


class AdapterCapabilities(TypedDict):
    """Feature flags per adapter so the client can branch safely."""
    supports_tools: bool
    supports_reasoning: bool
    first_system_only: bool  # whether the provider only honors the first system msg


class AdapterOptions(TypedDict, total=False):
    """Common generation options; adapters may down-map to provider-specific keys."""
    reasoning: Optional[bool]
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    stop: Optional[List[str]]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    system: Optional[str]  # allow override of system field when applicable
    # Provider-specific passthrough (kept generic)
    extra: Dict[str, Any]
