"""
LLM client protocol interface.
Defines the contract for LLM service implementations.
"""

from __future__ import annotations
from typing import Protocol, List, Dict, Any, Optional, AsyncIterator, Iterator
from ..models.conversation import Message, ConversationResult, ConversationContext
from ..models.tool import ToolSchema


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        context: ConversationContext,
        tools: Optional[List[ToolSchema]] = None
    ) -> ConversationResult:
        """Send chat request to LLM."""
        ...
    
    def chat_stream(
        self,
        messages: List[Dict[str, Any]], 
        context: ConversationContext,
        tools: Optional[List[ToolSchema]] = None
    ) -> Iterator[str]:
        """Send streaming chat request to LLM."""
        ...
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        ...


class StreamingHandler(Protocol):
    """Protocol for handling streaming responses."""
    
    def process_chunk(self, chunk: str) -> Optional[str]:
        """Process a single chunk from stream."""
        ...
    
    def handle_tool_call(self, tool_call_data: Dict[str, Any]) -> bool:
        """Handle tool call detected in stream."""
        ...
    
    def finalize_stream(self) -> str:
        """Finalize streaming and return complete response."""
        ...


class RetryPolicy(Protocol):
    """Protocol for retry policies."""
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if operation should be retried."""
        ...
    
    def get_delay(self, attempt: int) -> float:
        """Get delay before retry attempt."""
        ...
    
    def get_max_attempts(self) -> int:
        """Get maximum retry attempts."""
        ...
