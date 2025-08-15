"""
Conversation domain models - Pure business logic for chat interactions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ConversationState(Enum):
    """Conversation states."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    STREAMING = "streaming"


@dataclass
class Message:
    """Represents a single message in conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API calls."""
        return {
            "role": self.role.value,
            "content": self.content,
            **self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Create Message from dictionary."""
        role_str = data.get("role", "user")
        try:
            role = MessageRole(role_str)
        except ValueError:
            role = MessageRole.USER
        
        return cls(
            role=role,
            content=data.get("content", ""),
            timestamp=datetime.now(),
            metadata={k: v for k, v in data.items() if k not in ["role", "content"]}
        )


@dataclass
class ConversationHistory:
    """Manages conversation history with constraints."""
    messages: List[Message] = field(default_factory=list)
    max_history: int = 10
    system_prompt: Optional[str] = None
    
    def add_message(self, message: Message) -> None:
        """Add message and maintain history limits."""
        self.messages.append(message)
        self._trim_history()
    
    def add_user_message(self, content: str) -> None:
        """Convenience method to add user message."""
        message = Message(role=MessageRole.USER, content=content)
        self.add_message(message)
    
    def add_assistant_message(self, content: str) -> None:
        """Convenience method to add assistant message."""
        message = Message(role=MessageRole.ASSISTANT, content=content)
        self.add_message(message)
    
    def add_system_message(self, content: str) -> None:
        """Convenience method to add system message."""
        message = Message(role=MessageRole.SYSTEM, content=content)
        self.add_message(message)
    
    def _trim_history(self) -> None:
        """Trim history while preserving system messages."""
        # Separate system and non-system messages
        system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in self.messages if m.role != MessageRole.SYSTEM]
        
        # Keep only recent non-system messages
        if len(other_messages) > self.max_history:
            other_messages = other_messages[-self.max_history:]
        
        # Reconstruct with system messages first, then others
        self.messages = system_messages + other_messages
    
    def clear(self) -> None:
        """Clear all messages except initial system prompt."""
        if self.system_prompt:
            self.messages = [Message(role=MessageRole.SYSTEM, content=self.system_prompt)]
        else:
            self.messages = []
    
    def to_api_format(self) -> List[Dict[str, Any]]:
        """Convert to format expected by LLM API."""
        return [message.to_dict() for message in self.messages]
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message."""
        for message in reversed(self.messages):
            if message.role == MessageRole.USER:
                return message.content
        return None
    
    def get_formatted_history(self) -> str:
        """Get formatted string representation of history."""
        lines = []
        for message in self.messages:
            role_emoji = {
                MessageRole.SYSTEM: "🤖",
                MessageRole.USER: "👤", 
                MessageRole.ASSISTANT: "🤖",
                MessageRole.TOOL: "🔧"
            }.get(message.role, "❓")
            
            lines.append(f"{role_emoji} {message.role.value}: {message.content}")
        
        return "\n".join(lines)


@dataclass
class ConversationContext:
    """Context information for conversation processing."""
    model: str
    enable_tools: bool = True
    stream: bool = False
    max_output_tokens: Optional[int] = None
    ctx_size: Optional[int] = None
    reasoning: str = "high"
    show_trace: bool = False
    quiet: bool = False
    
    def to_generation_options(self) -> Dict[str, Any]:
        """Convert to options for model generation."""
        options = {}
        if self.max_output_tokens is not None:
            options['num_predict'] = self.max_output_tokens
        if self.ctx_size is not None:
            options['num_ctx'] = self.ctx_size
        return options


@dataclass
class ConversationResult:
    """Result of a conversation interaction."""
    success: bool
    content: str
    state: ConversationState = ConversationState.COMPLETED
    tool_calls_made: int = 0
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def failed(self) -> bool:
        """Check if conversation failed."""
        return not self.success or self.state == ConversationState.ERROR
