"""
Tool domain models - Pure business logic for tool operations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime
from enum import Enum
import json


class ToolCallStatus(Enum):
    """Status of tool call execution."""
    PENDING = "pending"
    EXECUTING = "executing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ToolCall:
    """Represents a tool call request."""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    status: ToolCallStatus = ToolCallStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_harmony_format(self) -> Dict[str, Any]:
        """Convert to Harmony tool call format."""
        return {
            "name": self.name,
            "arguments": self.arguments
        }
    
    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> ToolCall:
        """Create ToolCall from model response."""
        # Handle different response formats
        if "function" in data:
            # OpenAI format
            func_data = data["function"]
            name = func_data.get("name", "")
            try:
                arguments = json.loads(func_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
        else:
            # Direct format
            name = data.get("name", "")
            arguments = data.get("arguments", {})
        
        return cls(
            name=name,
            arguments=arguments,
            call_id=data.get("id")
        )


@dataclass
class ToolResult:
    """Result of tool execution."""
    tool_call: ToolCall
    success: bool
    content: str
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_message_format(self) -> Dict[str, Any]:
        """Convert to message format for conversation."""
        return {
            "role": "tool",
            "content": self.content,
            "tool_call_id": self.tool_call.call_id,
            "name": self.tool_call.name
        }
    
    @property
    def truncated_content(self) -> str:
        """Get truncated content for display."""
        max_length = 200  # Default truncation
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."


@dataclass
class ToolSchema:
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    implementation: Optional[Callable] = None
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert to API format for model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> bool:
        """Validate arguments against schema."""
        # Basic validation - in production would use jsonschema
        required = self.parameters.get("required", [])
        for req_param in required:
            if req_param not in arguments:
                return False
        return True


@dataclass
class ToolExecutionContext:
    """Context for tool execution."""
    user_message: Optional[str] = None
    conversation_id: Optional[str] = None
    max_rounds: int = 6
    current_round: int = 1
    print_limit: int = 200
    
    def should_continue(self) -> bool:
        """Check if tool execution should continue."""
        return self.current_round < self.max_rounds
    
    def next_round(self) -> None:
        """Move to next execution round."""
        self.current_round += 1


@dataclass
class ToolOrchestrationResult:
    """Result of complete tool orchestration."""
    success: bool
    final_response: str
    tool_results: List[ToolResult] = field(default_factory=list)
    rounds_executed: int = 0
    total_execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def tools_used(self) -> List[str]:
        """Get list of tool names used."""
        return [result.tool_call.name for result in self.tool_results]
    
    @property
    def successful_calls(self) -> int:
        """Count successful tool calls."""
        return sum(1 for result in self.tool_results if result.success)
    
    @property
    def failed_calls(self) -> int:
        """Count failed tool calls."""
        return sum(1 for result in self.tool_results if not result.success)


def canonicalize_tool_calls(tool_calls: Any) -> List[ToolCall]:
    """Convert various tool call formats to standardized ToolCall objects."""
    if not tool_calls:
        return []
    
    canonical_calls = []
    
    # Handle different input formats
    if isinstance(tool_calls, list):
        for call_data in tool_calls:
            if isinstance(call_data, dict):
                try:
                    tool_call = ToolCall.from_response(call_data)
                    canonical_calls.append(tool_call)
                except Exception:
                    continue
    
    return canonical_calls


def extract_tool_calls_from_response(response: Dict[str, Any]) -> List[ToolCall]:
    """Extract tool calls from model response in various formats."""
    tool_calls = []
    
    # Check for direct tool_calls field
    if "tool_calls" in response:
        tool_calls.extend(canonicalize_tool_calls(response["tool_calls"]))
    
    # Check for function_call (legacy OpenAI format)
    if "function_call" in response:
        func_call = response["function_call"]
        try:
            tool_call = ToolCall(
                name=func_call.get("name", ""),
                arguments=json.loads(func_call.get("arguments", "{}"))
            )
            tool_calls.append(tool_call)
        except (json.JSONDecodeError, KeyError):
            pass
    
    return tool_calls
