from __future__ import annotations

"""
Formal ChatAdapter interface for protocol adapters.

This ABC freezes the callable surface that concrete adapters must implement.
It aligns with the existing adapter responsibilities used by the client:
- Map generic options to provider-specific payload
- Format initial messages and reprompts
- Parse streaming and non-streaming responses
- Extract normalized tool calls

Concrete adapters should implement this ABC. The existing ProtocolAdapter in
`src/protocols/base.py` will inherit from this ChatAdapter to guarantee
conformance while preserving current adapter implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

from ..types import (
    AdapterCapabilities,
    AdapterOptions,
    ChatMessage,
    NonStreamResponse,
    NormalizedStreamEvent,
    ToolSpec,
)


class NonStreamParse(TypedDict, total=False):
    """Minimal non-streaming parse contract.

    We keep this as the minimal, stable contract (content + tool_calls). Adapters
    may return a richer `NonStreamResponse` in practice, and the client already
    supports that superset. Tests verify at least these two fields exist.
    """

    content: str
    tool_calls: List[Dict[str, Any]]


class ChatAdapter(ABC):
    """Adapter contract that all protocol adapters must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover - abstract signature only
        """Machine-friendly adapter name (e.g., 'harmony', 'deepseek')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:  # pragma: no cover
        """Feature flags for this adapter."""
        raise NotImplementedError

    # --- Options mapping ---
    @abstractmethod
    def map_options(self, opts: Optional[AdapterOptions]) -> Dict[str, Any]:  # pragma: no cover
        """Map generic options to provider-specific request fields."""
        raise NotImplementedError

    # --- Prompt formatting ---
    @abstractmethod
    def format_initial_messages(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolSpec]] = None,
        options: Optional[AdapterOptions] = None,
        mem0_block: Optional[str] = None,
    ) -> Tuple[List[ChatMessage], Dict[str, Any]]:  # pragma: no cover
        """Prepare first turn messages and payload overrides."""
        raise NotImplementedError

    @abstractmethod
    def format_reprompt_after_tools(
        self,
        history: List[ChatMessage],
        tool_results: List[Dict[str, Any]],
        options: Optional[AdapterOptions] = None,
    ) -> Tuple[List[ChatMessage], Dict[str, Any]]:  # pragma: no cover
        """Format follow-up messages after tools have executed."""
        raise NotImplementedError

    # --- Parsing ---
    @abstractmethod
    def parse_non_stream_response(self, resp: Any) -> NonStreamResponse:  # pragma: no cover
        """Normalize a provider non-streaming response (at least content/tool_calls)."""
        raise NotImplementedError

    @abstractmethod
    def parse_stream_events(self, raw_chunk: Any) -> Iterable[NormalizedStreamEvent]:  # pragma: no cover
        """Parse a provider-specific streaming chunk into normalized events."""
        raise NotImplementedError

    # --- Tooling ---
    @abstractmethod
    def extract_tool_calls(self, raw_response: Any) -> List[Dict[str, Any]]:  # pragma: no cover
        """Extract normalized tool calls from provider responses."""
        raise NotImplementedError
