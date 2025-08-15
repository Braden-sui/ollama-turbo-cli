"""
Tool orchestrator service - Domain service managing tool execution flow.
Handles multi-round tool calling with proper state management.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from ..models.tool import (
    ToolCall, ToolResult, ToolExecutionContext, ToolOrchestrationResult,
    ToolCallStatus, extract_tool_calls_from_response, canonicalize_tool_calls
)
from ..interfaces.tool_plugin import ToolRegistry
from .harmony_formatter import HarmonyFormatter


class ToolOrchestrator:
    """Domain service for orchestrating tool execution across multiple rounds."""
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        logger: Optional[logging.Logger] = None
    ):
        self._registry = tool_registry
        self._logger = logger or logging.getLogger(__name__)
    
    async def execute_tool_calls(
        self,
        tool_calls: List[ToolCall],
        context: ToolExecutionContext
    ) -> List[ToolResult]:
        """Execute a list of tool calls with proper error handling."""
        results = []
        
        for tool_call in tool_calls:
            try:
                tool_call.status = ToolCallStatus.EXECUTING
                start_time = time.time()
                
                result = await self._registry.execute_tool(tool_call, context)
                result.execution_time_ms = (time.time() - start_time) * 1000
                
                tool_call.status = ToolCallStatus.COMPLETED if result.success else ToolCallStatus.FAILED
                results.append(result)
                
                self._logger.debug(
                    f"Tool {tool_call.name} {'succeeded' if result.success else 'failed'} "
                    f"in {result.execution_time_ms:.1f}ms"
                )
                
            except Exception as e:
                tool_call.status = ToolCallStatus.FAILED
                
                error_result = ToolResult(
                    tool_call=tool_call,
                    success=False,
                    content="",
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000 if 'start_time' in locals() else None
                )
                results.append(error_result)
                
                self._logger.error(f"Tool {tool_call.name} execution failed: {e}")
        
        return results
    
    def extract_tool_calls_from_llm_response(
        self,
        response: Dict[str, Any]
    ) -> List[ToolCall]:
        """Extract and canonicalize tool calls from LLM response."""
        # First check for Harmony format in content
        if "message" in response:
            message = response["message"]
            content = message.get("content", "")
            
            # Try to extract Harmony-formatted tool calls from content
            harmony_calls = HarmonyFormatter.extract_tool_calls_from_harmony(content)
            if harmony_calls:
                tool_calls = []
                for call_data in harmony_calls:
                    tool_call = ToolCall(
                        name=call_data["name"],
                        arguments=call_data["arguments"]
                    )
                    tool_calls.append(tool_call)
                self._logger.debug(f"Extracted {len(tool_calls)} Harmony tool calls from content")
                return tool_calls
            
            # Direct tool_calls field
            if "tool_calls" in message:
                tool_calls = canonicalize_tool_calls(message["tool_calls"])
                if tool_calls:
                    self._logger.debug(f"Extracted {len(tool_calls)} tool calls from response")
                    return tool_calls
            
            # Legacy function_call field
            if "function_call" in message:
                try:
                    import json
                    func_call = message["function_call"]
                    tool_call = ToolCall(
                        name=func_call.get("name", ""),
                        arguments=json.loads(func_call.get("arguments", "{}"))
                    )
                    return [tool_call]
                except Exception as e:
                    self._logger.warning(f"Failed to parse function_call: {e}")
        
        # Direct response format
        tool_calls = extract_tool_calls_from_response(response)
        if tool_calls:
            self._logger.debug(f"Extracted {len(tool_calls)} tool calls from direct response")
        
        return tool_calls
    
    def prepare_tools_for_api(self) -> List[Dict[str, Any]]:
        """Prepare tool schemas in API format."""
        schemas = self._registry.get_schemas()
        api_tools = [schema.to_api_format() for schema in schemas]
        
        self._logger.debug(f"Prepared {len(api_tools)} tools for API")
        return api_tools
    
    def should_continue_tool_execution(
        self,
        context: ToolExecutionContext,
        results: List[ToolResult]
    ) -> bool:
        """Determine if tool execution should continue for another round."""
        # Check round limits
        if not context.should_continue():
            self._logger.debug(f"Tool execution stopped: reached max rounds ({context.max_rounds})")
            return False
        
        # Check if any tools succeeded (basic continuation logic)
        successful_tools = [r for r in results if r.success]
        if not successful_tools:
            self._logger.debug("Tool execution stopped: no successful tool calls")
            return False
        
        return True
    
    def format_tool_results_for_conversation(
        self,
        results: List[ToolResult],
        context: ToolExecutionContext
    ) -> List[Dict[str, Any]]:
        """Format tool results as conversation messages in Harmony format."""
        messages = []
        
        for result in results:
            # Format as Harmony tool response
            content = HarmonyFormatter.format_tool_response_message(
                tool_name=result.tool_call.name,
                result=result.content
            )
            
            message = {
                "role": "tool",
                "name": result.tool_call.name,
                "content": content
            }
            
            # Add tool call ID if available
            if result.tool_call.call_id:
                message["tool_call_id"] = result.tool_call.call_id
            
            messages.append(message)
        
        return messages
    
    def create_tool_execution_summary(
        self,
        results: List[ToolResult],
        total_rounds: int
    ) -> Dict[str, Any]:
        """Create summary of tool execution for logging/debugging."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        tools_used = {}
        total_execution_time = 0.0
        
        for result in results:
            tool_name = result.tool_call.name
            tools_used[tool_name] = tools_used.get(tool_name, 0) + 1
            
            if result.execution_time_ms:
                total_execution_time += result.execution_time_ms
        
        return {
            "total_calls": len(results),
            "successful_calls": len(successful),
            "failed_calls": len(failed),
            "rounds_executed": total_rounds,
            "tools_used": tools_used,
            "total_execution_time_ms": total_execution_time,
            "average_execution_time_ms": total_execution_time / len(results) if results else 0.0
        }
    
    def handle_tool_execution_errors(
        self,
        results: List[ToolResult]
    ) -> Optional[str]:
        """Handle and format tool execution errors."""
        failed_results = [r for r in results if not r.success]
        
        if not failed_results:
            return None
        
        error_messages = []
        for result in failed_results:
            tool_name = result.tool_call.name
            error = result.error or "Unknown error"
            error_messages.append(f"Tool '{tool_name}' failed: {error}")
        
        return "; ".join(error_messages)
    
    def validate_tool_availability(self, tool_names: List[str]) -> Dict[str, bool]:
        """Validate that requested tools are available."""
        available_tools = {}
        registered_tools = self._registry.list_tools()
        
        for tool_name in tool_names:
            available_tools[tool_name] = tool_name in registered_tools
        
        return available_tools
    
    async def orchestrate_multi_round_execution(
        self,
        initial_tool_calls: List[ToolCall],
        context: ToolExecutionContext,
        llm_callback: callable
    ) -> ToolOrchestrationResult:
        """
        Orchestrate multi-round tool execution with LLM feedback.
        
        Args:
            initial_tool_calls: First round of tool calls
            context: Execution context with limits
            llm_callback: Function to call LLM for next round (messages) -> (response, tool_calls)
        """
        start_time = time.time()
        all_results = []
        current_tool_calls = initial_tool_calls
        
        try:
            while current_tool_calls and context.should_continue():
                self._logger.debug(f"Executing tool round {context.current_round}/{context.max_rounds}")
                
                # Execute current round of tools
                round_results = await self.execute_tool_calls(current_tool_calls, context)
                all_results.extend(round_results)
                
                # Format tool results for conversation
                tool_messages = self.format_tool_results_for_conversation(round_results, context)
                
                # Call LLM with tool results to potentially get more tool calls
                try:
                    response, next_tool_calls = await llm_callback(tool_messages)
                    
                    if not next_tool_calls:
                        # LLM provided final response, no more tool calls
                        final_response = response
                        break
                    
                    # Continue with next round
                    current_tool_calls = next_tool_calls
                    context.next_round()
                    
                except Exception as e:
                    self._logger.error(f"LLM callback failed in round {context.current_round}: {e}")
                    final_response = f"Error in tool execution: {e}"
                    break
            
            else:
                # Loop ended due to limits or no more tool calls
                final_response = "Tool execution completed."
                if not context.should_continue():
                    final_response += f" (Reached maximum rounds: {context.max_rounds})"
            
            total_time = (time.time() - start_time) * 1000
            
            return ToolOrchestrationResult(
                success=True,
                final_response=final_response,
                tool_results=all_results,
                rounds_executed=context.current_round,
                total_execution_time_ms=total_time
            )
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            
            return ToolOrchestrationResult(
                success=False,
                final_response=f"Tool orchestration failed: {e}",
                tool_results=all_results,
                rounds_executed=context.current_round,
                total_execution_time_ms=total_time,
                error=str(e)
            )
