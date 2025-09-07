"""
Base protocol adapter interface for Ollama Turbo client.

Each concrete adapter (e.g., HarmonyAdapter, DeepSeekAdapter) implements
this interface to encapsulate all protocol-specific formatting and parsing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..types import (
    AdapterCapabilities,
    AdapterOptions,
    ChatMessage,
    NonStreamResponse,
    NormalizedStreamEvent,
    ProtocolName,
    ToolSpec,
)


class ProtocolAdapter(ABC):
    """Strategy interface for model protocol handling.

    Responsibilities:
    - Map generic options to provider-specific payload.
    - Ensure Mem0/system content is correctly placed (some providers only honor first system).
    - Format prompts (initial turn and post-tool reprompts).
    - Parse streaming and non-streaming responses into normalized shapes.
    - Extract normalized tool calls.
    """

    def __init__(self, model: str, protocol: ProtocolName = "auto") -> None:
        self.model = model
        self.protocol = protocol

    # --- Identity & capabilities ---
    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-friendly adapter name (e.g., 'harmony', 'deepseek')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        """Feature flags for this adapter."""
        raise NotImplementedError

    # --- Prompt formatting ---
    @abstractmethod
    def format_initial_messages(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolSpec]] = None,
        options: Optional[AdapterOptions] = None,
        mem0_block: Optional[str] = None,
    ) -> Tuple[List[ChatMessage], Dict[str, Any]]:
        """Prepare the first turn messages and provider payload extras.

        Returns a tuple of (normalized_messages, payload_overrides).
        The adapter may inject/merge system instructions and mem0_block into the
        first system message as required by the provider.
        """
        raise NotImplementedError

    @abstractmethod
    def format_reprompt_after_tools(
        self,
        messages: List[ChatMessage],
        tool_results: List[Dict[str, Any]],
        options: Optional[AdapterOptions] = None,
    ) -> Tuple[List[ChatMessage], Dict[str, Any]]:
        """Format messages for the follow-up turn after tool execution."""
        raise NotImplementedError

    # --- Options mapping ---
    @abstractmethod
    def map_options(self, options: Optional[AdapterOptions]) -> Dict[str, Any]:
        """Map generic options to provider-specific request fields."""
        raise NotImplementedError

    # --- Parsing: streaming and non-streaming ---
    @abstractmethod
    def parse_stream_events(self, raw_chunk: Any) -> Iterable[NormalizedStreamEvent]:
        """Parse a provider-specific streaming chunk into normalized events.

        Adapters may yield multiple events per chunk. The client will preserve
        the backend SSE contract by relaying these events as-is.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_non_stream_response(self, raw_response: Any) -> NonStreamResponse:
        """Normalize a provider non-streaming response."""
        raise NotImplementedError

    # --- Tooling ---
    @abstractmethod
    def extract_tool_calls(self, raw_response: Any) -> List[Dict[str, Any]]:
        """Extract normalized tool calls (non-streaming or final streaming state)."""
        raise NotImplementedError
