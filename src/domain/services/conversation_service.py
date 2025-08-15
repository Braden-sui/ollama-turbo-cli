"""
Conversation service - Domain service managing conversation state and flow.
Pure business logic with no external dependencies.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models.conversation import (
    ConversationHistory, ConversationContext, ConversationResult, 
    ConversationState, Message, MessageRole
)
from ..models.tool import ToolCall, ToolResult
from ..models.memory import MemoryContext


class ConversationService:
    """Domain service for managing conversation flow and state."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)
        self._active_conversations: Dict[str, ConversationHistory] = {}
    
    def create_conversation(
        self,
        conversation_id: str,
        system_prompt: Optional[str] = None,
        max_history: int = 10
    ) -> ConversationHistory:
        """Create a new conversation with optional system prompt."""
        history = ConversationHistory(
            max_history=max_history,
            system_prompt=system_prompt
        )
        
        if system_prompt:
            history.add_system_message(system_prompt)
        
        self._active_conversations[conversation_id] = history
        
        self._logger.debug(f"Created conversation {conversation_id} with max_history={max_history}")
        return history
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Get existing conversation by ID."""
        return self._active_conversations.get(conversation_id)
    
    def add_user_message(
        self,
        conversation_id: str,
        content: str
    ) -> Optional[ConversationHistory]:
        """Add user message to conversation."""
        history = self.get_conversation(conversation_id)
        if not history:
            return None
        
        history.add_user_message(content)
        self._logger.debug(f"Added user message to conversation {conversation_id}")
        return history
    
    def add_assistant_message(
        self,
        conversation_id: str,
        content: str
    ) -> Optional[ConversationHistory]:
        """Add assistant message to conversation."""
        history = self.get_conversation(conversation_id)
        if not history:
            return None
        
        history.add_assistant_message(content)
        self._logger.debug(f"Added assistant message to conversation {conversation_id}")
        return history
    
    def add_system_message(
        self,
        conversation_id: str,
        content: str
    ) -> Optional[ConversationHistory]:
        """Add system message to conversation."""
        history = self.get_conversation(conversation_id)
        if not history:
            return None
        
        history.add_system_message(content)
        self._logger.debug(f"Added system message to conversation {conversation_id}")
        return history
    
    def inject_memory_context(
        self,
        conversation_id: str,
        memory_content: List[str]
    ) -> bool:
        """Inject memory context into conversation."""
        if not memory_content:
            return False
        
        history = self.get_conversation(conversation_id)
        if not history:
            return False
        
        # Create memory context injection
        context_text = (
            "Relevant memories from our previous conversations:\n" + 
            "\n".join(f"- {memory}" for memory in memory_content) + 
            "\n\nPlease consider this context when responding.\n"
        )
        
        history.add_system_message(context_text)
        self._logger.debug(f"Injected memory context into conversation {conversation_id}")
        return True
    
    def add_tool_results(
        self,
        conversation_id: str,
        tool_results: List[ToolResult]
    ) -> bool:
        """Add tool execution results to conversation."""
        history = self.get_conversation(conversation_id)
        if not history:
            return False
        
        for result in tool_results:
            # Add tool result as tool message
            tool_message = Message(
                role=MessageRole.TOOL,
                content=result.content,
                metadata={
                    "tool_name": result.tool_call.name,
                    "tool_call_id": result.tool_call.call_id,
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms
                }
            )
            history.add_message(tool_message)
        
        self._logger.debug(f"Added {len(tool_results)} tool results to conversation {conversation_id}")
        return True
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history while preserving system prompt."""
        history = self.get_conversation(conversation_id)
        if not history:
            return False
        
        history.clear()
        self._logger.debug(f"Cleared conversation {conversation_id}")
        return True
    
    def get_conversation_for_api(
        self,
        conversation_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get conversation in API format for LLM calls."""
        history = self.get_conversation(conversation_id)
        if not history:
            return None
        
        return history.to_api_format()
    
    def get_last_user_message(self, conversation_id: str) -> Optional[str]:
        """Get the last user message from conversation."""
        history = self.get_conversation(conversation_id)
        if not history:
            return None
        
        return history.get_last_user_message()
    
    def get_formatted_history(self, conversation_id: str) -> Optional[str]:
        """Get formatted conversation history for display."""
        history = self.get_conversation(conversation_id)
        if not history:
            return None
        
        return history.get_formatted_history()
    
    def create_memory_context_from_conversation(
        self,
        conversation_id: str,
        user_id: str,
        agent_id: Optional[str] = None
    ) -> Optional[MemoryContext]:
        """Create memory context from conversation for storage."""
        history = self.get_conversation(conversation_id)
        if not history:
            return None
        
        return MemoryContext(
            user_id=user_id,
            agent_id=agent_id,
            session_id=conversation_id,
            metadata={
                "conversation_length": len(history.messages),
                "created_at": datetime.now().isoformat()
            }
        )
    
    def extract_conversation_messages_for_memory(
        self,
        conversation_id: str,
        include_system: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract messages suitable for memory storage."""
        history = self.get_conversation(conversation_id)
        if not history:
            return []
        
        messages = []
        for message in history.messages:
            # Skip system messages unless requested
            if message.role == MessageRole.SYSTEM and not include_system:
                continue
            
            # Skip tool messages (they're not conversational)
            if message.role == MessageRole.TOOL:
                continue
            
            messages.append(message.to_dict())
        
        return messages
    
    def get_conversation_statistics(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics about the conversation."""
        history = self.get_conversation(conversation_id)
        if not history:
            return None
        
        role_counts = {}
        total_length = 0
        
        for message in history.messages:
            role = message.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
            total_length += len(message.content)
        
        return {
            "message_count": len(history.messages),
            "role_counts": role_counts,
            "total_character_length": total_length,
            "max_history_limit": history.max_history,
            "has_system_prompt": history.system_prompt is not None
        }
    
    def cleanup_old_conversations(self, max_age_hours: int = 24) -> int:
        """Clean up old inactive conversations."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        # Simple cleanup - in production would track last activity
        old_conversations = []
        for conv_id, history in self._active_conversations.items():
            # Check if conversation has recent activity (rough heuristic)
            if history.messages:
                last_message_time = history.messages[-1].timestamp.timestamp()
                if last_message_time < cutoff_time:
                    old_conversations.append(conv_id)
        
        for conv_id in old_conversations:
            del self._active_conversations[conv_id]
        
        self._logger.debug(f"Cleaned up {len(old_conversations)} old conversations")
        return len(old_conversations)
