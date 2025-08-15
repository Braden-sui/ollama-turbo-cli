"""
Tool registry implementation - Infrastructure component managing tool plugins.
Integrates with existing plugin loader while implementing domain interfaces.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Dict, List, Optional

from ...domain.models.tool import ToolCall, ToolResult, ToolSchema, ToolExecutionContext
from ...domain.interfaces.tool_plugin import ToolRegistry, ToolPlugin
from ...plugin_loader import PluginManager


class PluginToolAdapter(ToolPlugin):
    """Adapter to wrap existing plugin functions as ToolPlugin interface."""
    
    def __init__(
        self,
        name: str,
        schema: Dict,
        implementation: callable,
        logger: Optional[logging.Logger] = None
    ):
        self._name = name
        self._schema = schema
        self._implementation = implementation
        self._logger = logger or logging.getLogger(__name__)
    
    def get_schema(self) -> ToolSchema:
        """Get the tool schema definition."""
        return ToolSchema(
            name=self._name,
            description=self._schema.get('function', {}).get('description', ''),
            parameters=self._schema.get('function', {}).get('parameters', {}),
            implementation=self._implementation
        )
    
    async def execute(
        self,
        tool_call: ToolCall,
        context: ToolExecutionContext
    ) -> ToolResult:
        """Execute the tool with given arguments."""
        start_time = time.time()
        
        try:
            # Execute the tool function
            if asyncio.iscoroutinefunction(self._implementation):
                result_content = await self._implementation(**tool_call.arguments)
            else:
                # Run sync function in thread pool to avoid blocking
                result_content = await asyncio.to_thread(
                    self._implementation, **tool_call.arguments
                )
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                tool_call=tool_call,
                success=True,
                content=str(result_content),
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            self._logger.error(f"Tool {self._name} execution failed: {e}")
            
            return ToolResult(
                tool_call=tool_call,
                success=False,
                content="",
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if tool is available for use."""
        return self._implementation is not None


class DefaultToolRegistry(ToolRegistry):
    """Default implementation of tool registry using existing plugin system."""
    
    def __init__(
        self,
        plugin_manager: Optional[PluginManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        self._plugin_manager = plugin_manager or PluginManager()
        self._logger = logger or logging.getLogger(__name__)
        self._tools: Dict[str, ToolPlugin] = {}
        
        # Initialize tools from plugin manager
        self._load_tools_from_plugins()
    
    def _load_tools_from_plugins(self) -> None:
        """Load tools from existing plugin system."""
        try:
            # Get schemas and functions from plugin manager
            schemas = self._plugin_manager.tool_schemas
            functions = self._plugin_manager.tool_functions
            
            for schema in schemas:
                tool_name = schema.get('function', {}).get('name')
                if not tool_name:
                    continue
                
                implementation = functions.get(tool_name)
                if not implementation:
                    self._logger.warning(f"No implementation found for tool: {tool_name}")
                    continue
                
                # Create adapter and register
                adapter = PluginToolAdapter(
                    name=tool_name,
                    schema=schema,
                    implementation=implementation,
                    logger=self._logger
                )
                
                self._tools[tool_name] = adapter
                self._logger.debug(f"Registered tool: {tool_name}")
            
            self._logger.info(f"Loaded {len(self._tools)} tools from plugins")
            
        except Exception as e:
            self._logger.error(f"Failed to load tools from plugins: {e}")
    
    def register_tool(self, plugin: ToolPlugin) -> None:
        """Register a tool plugin."""
        schema = plugin.get_schema()
        self._tools[schema.name] = plugin
        self._logger.debug(f"Registered tool: {schema.name}")
    
    def get_tool(self, name: str) -> Optional[ToolPlugin]:
        """Get tool plugin by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_schemas(self) -> List[ToolSchema]:
        """Get all tool schemas for API."""
        schemas = []
        for tool in self._tools.values():
            try:
                schema = tool.get_schema()
                schemas.append(schema)
            except Exception as e:
                self._logger.warning(f"Failed to get schema for tool: {e}")
        
        return schemas
    
    async def execute_tool(
        self,
        tool_call: ToolCall,
        context: ToolExecutionContext
    ) -> ToolResult:
        """Execute a tool call."""
        tool = self.get_tool(tool_call.name)
        if not tool:
            return ToolResult(
                tool_call=tool_call,
                success=False,
                content="",
                error=f"Tool '{tool_call.name}' not found"
            )
        
        if not tool.is_available():
            return ToolResult(
                tool_call=tool_call,
                success=False,
                content="",
                error=f"Tool '{tool_call.name}' not available"
            )
        
        try:
            result = await tool.execute(tool_call, context)
            
            # Apply print limit if configured
            if context.print_limit and len(result.content) > context.print_limit:
                original_content = result.content
                result.content = result.content[:context.print_limit] + "..."
                result.metadata["truncated"] = True
                result.metadata["original_length"] = len(original_content)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                tool_call=tool_call,
                success=False,
                content="",
                error=f"Tool execution error: {e}"
            )
    
    def reload_tools(self) -> None:
        """Reload tools from plugin system."""
        self._tools.clear()
        self._load_tools_from_plugins()
        self._logger.info("Tools reloaded from plugins")
    
    def get_tool_statistics(self) -> Dict[str, any]:
        """Get statistics about registered tools."""
        available_count = sum(1 for tool in self._tools.values() if tool.is_available())
        
        return {
            "total_tools": len(self._tools),
            "available_tools": available_count,
            "unavailable_tools": len(self._tools) - available_count,
            "tool_names": list(self._tools.keys())
        }
