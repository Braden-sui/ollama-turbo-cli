"""
Tool plugin protocol interface.
Defines the contract for tool plugin implementations.
"""

from __future__ import annotations
from typing import Protocol, List, Dict, Any, Optional, Callable
from ..models.tool import ToolSchema, ToolCall, ToolResult, ToolExecutionContext


class ToolPlugin(Protocol):
    """Protocol for tool plugin implementations."""
    
    def get_schema(self) -> ToolSchema:
        """Get the tool schema definition."""
        ...
    
    async def execute(
        self,
        tool_call: ToolCall,
        context: ToolExecutionContext
    ) -> ToolResult:
        """Execute the tool with given arguments."""
        ...
    
    def is_available(self) -> bool:
        """Check if tool is available for use."""
        ...


class ToolRegistry(Protocol):
    """Protocol for tool registry implementations."""
    
    def register_tool(self, plugin: ToolPlugin) -> None:
        """Register a tool plugin."""
        ...
    
    def get_tool(self, name: str) -> Optional[ToolPlugin]:
        """Get tool plugin by name."""
        ...
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        ...
    
    def get_schemas(self) -> List[ToolSchema]:
        """Get all tool schemas for API."""
        ...
    
    async def execute_tool(
        self,
        tool_call: ToolCall,
        context: ToolExecutionContext
    ) -> ToolResult:
        """Execute a tool call."""
        ...


class ToolValidator(Protocol):
    """Protocol for tool argument validation."""
    
    def validate_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> bool:
        """Validate tool arguments against schema."""
        ...
    
    def get_validation_error(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[str]:
        """Get detailed validation error message."""
        ...
