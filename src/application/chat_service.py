"""
Chat service - Application service orchestrating complete chat interactions.
Coordinates conversation, tools, memory, and streaming.
"""

from __future__ import annotations
import asyncio
import logging
import uuid
import json
from typing import Optional, List, Dict, Any, Iterator

from ..domain.models.conversation import ConversationContext, ConversationResult, ConversationState
from ..domain.models.tool import ToolExecutionContext, ToolCall
from ..domain.models.memory import MemoryContext
from ..domain.services.conversation_service import ConversationService
from ..domain.services.tool_orchestrator import ToolOrchestrator
from ..domain.services.stream_processor import StreamProcessor, StreamState
from ..domain.services.harmony_formatter import HarmonyFormatter
from ..domain.interfaces.llm_client import LLMClient
from .memory_service import MemoryService


class ChatService:
    """Application service orchestrating complete chat interactions."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        conversation_service: ConversationService,
        tool_orchestrator: Optional[ToolOrchestrator] = None,
        memory_service: Optional[MemoryService] = None,
        stream_processor: Optional[StreamProcessor] = None,
        logger: Optional[logging.Logger] = None
    ):
        self._llm_client = llm_client
        self._conversation_service = conversation_service
        self._tool_orchestrator = tool_orchestrator
        self._memory_service = memory_service
        self._stream_processor = stream_processor or StreamProcessor()
        self._logger = logger or logging.getLogger(__name__)
    
    async def chat(
        self,
        message: str,
        context: ConversationContext,
        conversation_id: Optional[str] = None,
        memory_context: Optional[MemoryContext] = None
    ) -> ConversationResult:
        """Process a complete chat interaction."""
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        try:
            # Initialize or get conversation
            history = self._conversation_service.get_conversation(conversation_id)
            if not history:
                # Create with system prompt if available
                system_prompt = self._get_system_prompt(context)
                history = self._conversation_service.create_conversation(
                    conversation_id, 
                    system_prompt=system_prompt
                )
            
            # Inject memory context if available
            if self._memory_service and memory_context:
                await self._inject_memory_context(conversation_id, message, memory_context)
            
            # Add user message to conversation
            self._conversation_service.add_user_message(conversation_id, message)
            
            # Get conversation in API format
            api_messages = self._conversation_service.get_conversation_for_api(conversation_id)
            if not api_messages:
                return ConversationResult(
                    success=False,
                    content="Failed to prepare conversation",
                    state=ConversationState.ERROR,
                    error="Conversation preparation failed"
                )
            
            # Execute chat with tools if enabled
            if context.enable_tools and self._tool_orchestrator:
                result = await self._chat_with_tools(api_messages, context, conversation_id)
            else:
                result = await self._chat_simple(api_messages, context)
            
            # Add assistant response to conversation
            if result.success:
                self._conversation_service.add_assistant_message(conversation_id, result.content)
                
                # Store conversation in memory if available
                if self._memory_service and memory_context:
                    await self._store_conversation_memory(
                        conversation_id, message, result.content, memory_context
                    )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Chat service error: {e}")
            return ConversationResult(
                success=False,
                content=f"Chat service error: {e}",
                state=ConversationState.ERROR,
                error=str(e)
            )
    
    async def chat_stream(
        self,
        message: str,
        context: ConversationContext,
        conversation_id: Optional[str] = None,
        memory_context: Optional[MemoryContext] = None
    ) -> Iterator[str]:
        """Process streaming chat interaction."""
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        try:
            # Initialize or get conversation
            history = self._conversation_service.get_conversation(conversation_id)
            if not history:
                system_prompt = self._get_system_prompt(context)
                history = self._conversation_service.create_conversation(
                    conversation_id,
                    system_prompt=system_prompt
                )
            
            # Inject memory context if available
            if self._memory_service and memory_context:
                await self._inject_memory_context(conversation_id, message, memory_context)
            
            # Add user message
            self._conversation_service.add_user_message(conversation_id, message)
            
            # Get conversation for API
            api_messages = self._conversation_service.get_conversation_for_api(conversation_id)
            if not api_messages:
                yield "Error: Failed to prepare conversation"
                return
            
            # Execute streaming chat
            if context.enable_tools and self._tool_orchestrator:
                async for chunk in self._chat_stream_with_tools(api_messages, context, conversation_id):
                    yield chunk
            else:
                async for chunk in self._chat_stream_simple(api_messages, context):
                    yield chunk
            
        except Exception as e:
            # Silent fallback to non-streaming path (per user preference)
            self._logger.debug(f"Chat stream error; falling back to non-streaming: {e}")
            try:
                result = await self.chat(message, context, conversation_id, memory_context)
                if result and result.success and result.content:
                    yield result.content
            except Exception as e2:
                self._logger.debug(f"Non-streaming fallback also failed: {e2}")
    
    async def _chat_simple(
        self,
        messages: List[Dict[str, Any]],
        context: ConversationContext
    ) -> ConversationResult:
        """Execute simple chat without tools."""
        try:
            response = self._llm_client.chat(messages, context)
            if response and response.success and response.content:
                # Clean to Harmony final-only content
                state = self._stream_processor.create_stream_state()
                self._stream_processor.process_stream_chunk(response.content, state)
                final_result = self._stream_processor.finalize_stream(state)
                return ConversationResult(
                    success=True,
                    content=(final_result.get("content") or response.content),
                    state=ConversationState.COMPLETED,
                    metadata=response.metadata
                )
            return response
            
        except Exception as e:
            self._logger.error(f"Simple chat error: {e}")
            return ConversationResult(
                success=False,
                content=f"LLM error: {e}",
                state=ConversationState.ERROR,
                error=str(e)
            )
    
    async def _chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        context: ConversationContext,
        conversation_id: str
    ) -> ConversationResult:
        """Execute chat with tool calling support."""
        try:
            # Prepare tools for API
            tools = self._tool_orchestrator.prepare_tools_for_api()
            
            # Create tool execution context
            tool_context = ToolExecutionContext(
                user_message=self._conversation_service.get_last_user_message(conversation_id),
                conversation_id=conversation_id,
                max_rounds=6,  # From environment or config
                print_limit=200
            )
            
            # Initial LLM call with tools
            response = self._llm_client.chat(messages, context, tools)
            if not response.success:
                return response
            
            # Extract tool calls from response
            raw_msg = None
            try:
                raw_msg = response.metadata.get('message') if isinstance(response.metadata, dict) else None
            except Exception:
                raw_msg = None
            payload = {"message": raw_msg or {"content": response.content}}
            # Also surface top-level tool_calls if adapter provided them
            if isinstance(response.metadata, dict) and response.metadata.get('tool_calls'):
                payload["tool_calls"] = response.metadata.get('tool_calls')
            tool_calls = self._tool_orchestrator.extract_tool_calls_from_llm_response(payload)
            
            if not tool_calls:
                # No tools called, clean to Harmony final-only and return
                try:
                    state = self._stream_processor.create_stream_state()
                    self._stream_processor.process_stream_chunk(response.content or "", state)
                    final_result = self._stream_processor.finalize_stream(state)
                    cleaned = final_result.get("content") or response.content
                except Exception:
                    cleaned = response.content
                return ConversationResult(
                    success=True,
                    content=cleaned,
                    state=ConversationState.COMPLETED,
                    metadata=response.metadata
                )
            
            # Execute tools and continue conversation
            async def llm_callback(tool_messages):
                # Add tool results to conversation
                # Prefer the raw assistant message (with tool_calls) to preserve tool_call ids
                assistant_msg = raw_msg if isinstance(raw_msg, dict) else {"role": "assistant", "content": response.content}
                current_messages = messages + [assistant_msg] + tool_messages
                
                # Call LLM again without tools for final response
                final_response = self._llm_client.chat(current_messages, context)
                
                # Extract any additional tool calls from raw metadata if present (defensive)
                fm = None
                try:
                    fm = final_response.metadata.get('message') if isinstance(final_response.metadata, dict) else None
                except Exception:
                    fm = None
                next_payload = {"message": fm or {"content": final_response.content}}
                if isinstance(final_response.metadata, dict) and final_response.metadata.get('tool_calls'):
                    next_payload["tool_calls"] = final_response.metadata.get('tool_calls')
                next_tool_calls = self._tool_orchestrator.extract_tool_calls_from_llm_response(next_payload)
                
                return final_response.content, next_tool_calls
            
            # Orchestrate multi-round tool execution
            orchestration_result = await self._tool_orchestrator.orchestrate_multi_round_execution(
                tool_calls, tool_context, llm_callback
            )
            
            # Clean final response content to ensure only final channel emits
            try:
                state2 = self._stream_processor.create_stream_state()
                self._stream_processor.process_stream_chunk(orchestration_result.final_response or "", state2)
                final2 = self._stream_processor.finalize_stream(state2)
                cleaned_final = final2.get("content") or orchestration_result.final_response
            except Exception:
                cleaned_final = orchestration_result.final_response

            return ConversationResult(
                success=orchestration_result.success,
                content=cleaned_final,
                tool_calls_made=len(orchestration_result.tool_results),
                error=orchestration_result.error
            )
            
        except Exception as e:
            self._logger.error(f"Tool-enabled chat error: {e}")
            return ConversationResult(
                success=False,
                content=f"Tool execution error: {e}",
                state=ConversationState.ERROR,
                error=str(e)
            )
    
    async def _chat_stream_simple(
        self,
        messages: List[Dict[str, Any]],
        context: ConversationContext
    ) -> Iterator[str]:
        """Execute simple streaming chat."""
        try:
            # Buffer stream and emit only the cleaned final content to avoid Harmony analysis/commentary leakage
            stream = self._llm_client.chat_stream(messages, context)
            stream_state = self._stream_processor.create_stream_state()
            for chunk in stream:
                # Accumulate chunks; do not emit partials
                self._stream_processor.process_stream_chunk(chunk, stream_state)
            # Finalize and yield only the final channel content
            final_result = self._stream_processor.finalize_stream(stream_state)
            if final_result.get("content"):
                yield final_result["content"]
            elif stream_state.accumulated_content:
                # Fallback: emit raw accumulated content if no final-channel text was found
                yield stream_state.accumulated_content.strip()
                
        except Exception as e:
            # Silent non-streaming fallback
            self._logger.debug(f"Simple stream error; falling back to non-streaming: {e}")
            try:
                resp = self._llm_client.chat(messages, context)
                if resp and resp.success and resp.content:
                    yield resp.content
            except Exception as e2:
                self._logger.debug(f"Simple stream fallback failed: {e2}")
    
    async def _chat_stream_with_tools(
        self,
        messages: List[Dict[str, Any]],
        context: ConversationContext,
        conversation_id: str
    ) -> Iterator[str]:
        """Execute streaming chat with tool support."""
        try:
            tools = self._tool_orchestrator.prepare_tools_for_api()
            stream_state = self._stream_processor.create_stream_state()
            
            # Track accumulated response for tool detection
            accumulated_response = ""
            
            def on_content(chunk: str):
                nonlocal accumulated_response
                accumulated_response += chunk
            
            def on_tool_call(tool_call_data: Dict[str, Any]):
                # Tool call detected - will be handled after stream completes
                pass
            
            # Process stream
            stream = self._llm_client.chat_stream(messages, context, tools)
            for chunk in stream:
                self._stream_processor.process_stream_chunk(
                    chunk, stream_state, on_content, on_tool_call
                )
                # Do not emit initial chunks; buffer only. If tools are detected,
                # we will stream the final response after tool execution. If no tools
                # are detected, we'll emit the buffered content once at the end.
            
            # Finalize stream and check for tool calls
            final_result = self._stream_processor.finalize_stream(stream_state)
            
            if final_result["tool_calls"]:
                # Convert to ToolCall objects
                tool_calls = []
                for call_data in final_result["tool_calls"]:
                    try:
                        tool_call = ToolCall(
                            name=call_data.get("name", ""),
                            arguments=call_data.get("arguments", {}),
                            call_id=(call_data.get("id") or call_data.get("tool_call_id") or call_data.get("call_id"))
                        )
                        tool_calls.append(tool_call)
                    except Exception:
                        continue
                
                if tool_calls:
                    # Execute tools and continue streaming
                    tool_context = ToolExecutionContext(conversation_id=conversation_id)
                    tool_results = await self._tool_orchestrator.execute_tool_calls(
                        tool_calls, tool_context
                    )
                    
                    # Add tool results and get final response
                    tool_messages = self._tool_orchestrator.format_tool_results_for_conversation(
                        tool_results, tool_context
                    )
                    # Reconstruct assistant tool_call message so LLM can correlate tool outputs
                    assistant_tool_msg = {
                        "role": "assistant",
                        # Use cleaned content from finalize_stream to avoid leaking commentary tokens
                        "content": (final_result.get("content") or ""),
                        "tool_calls": [
                            {
                                "id": tc.call_id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments) if not isinstance(tc.arguments, str) else tc.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                    final_messages = messages + [assistant_tool_msg] + tool_messages
                    
                    # Stream final response, but only emit the cleaned final content once
                    final_stream = self._llm_client.chat_stream(final_messages, context)
                    stream_state2 = self._stream_processor.create_stream_state()
                    for chunk in final_stream:
                        self._stream_processor.process_stream_chunk(chunk, stream_state2)
                    final2 = self._stream_processor.finalize_stream(stream_state2)
                    if final2.get("content"):
                        yield final2["content"]
                    elif stream_state2.accumulated_content:
                        # Fallback: emit raw accumulated content if no final-channel text was found
                        yield stream_state2.accumulated_content.strip()
            else:
                # No tools detected: emit the buffered content once
                if final_result["content"]:
                    yield final_result["content"]
                elif accumulated_response:
                    # Fallback: emit the buffered raw content
                    yield accumulated_response.strip()
                        
        except Exception as e:
            # Silent non-streaming fallback for tools path
            self._logger.debug(f"Tool stream error; falling back to non-streaming: {e}")
            try:
                result = await self._chat_with_tools(messages, context, conversation_id)
                if result and result.success and result.content:
                    yield result.content
            except Exception as e2:
                self._logger.debug(f"Tool stream fallback failed: {e2}")
    
    async def _inject_memory_context(
        self,
        conversation_id: str,
        user_message: str,
        memory_context: MemoryContext
    ) -> None:
        """Inject relevant memories into conversation."""
        try:
            search_result = await self._memory_service.search_memories(
                user_message, memory_context, limit=3
            )
            
            if search_result.success and search_result.entries:
                memory_texts = [entry.memory for entry in search_result.entries]
                self._conversation_service.inject_memory_context(
                    conversation_id, memory_texts
                )
                self._logger.debug(f"Injected {len(memory_texts)} memories into conversation")
                
        except Exception as e:
            self._logger.debug(f"Memory injection failed: {e}")
    
    async def _store_conversation_memory(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        memory_context: MemoryContext
    ) -> None:
        """Store conversation in memory for future reference."""
        try:
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response}
            ]
            
            await self._memory_service.add_memory(
                messages, memory_context, background=True
            )
            self._logger.debug("Stored conversation in memory")
            
        except Exception as e:
            self._logger.debug(f"Memory storage failed: {e}")
    
    def _get_system_prompt(self, context: ConversationContext) -> str:
        """Generate system prompt in OpenHarmony format."""
        # Get tools if available
        tools = []
        if context.enable_tools and self._tool_orchestrator:
            tools = self._tool_orchestrator.prepare_tools_for_api()
        
        # Custom instructions
        reasoning = context.reasoning if context.reasoning in {"high", "low"} else "high"
        instructions = (
            "You are a helpful AI assistant with access to tools. "
            f"Reasoning level: {reasoning}.\n\n"
            "## Guidelines\n"
            "- Analyze the user's request to determine if tools are needed\n"
            "- For factual queries requiring current data, use appropriate tools\n"
            "- For computational tasks, use the calculator tool\n"
            "- Keep your thinking process internal - only show final answers\n"
            "- When using web tools, cite sources briefly\n"
            "- Be concise but complete in your responses"
        )
        
        return HarmonyFormatter.format_system_prompt_with_tools(
            tools=tools,
            reasoning=reasoning,
            custom_instructions=instructions
        )
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation's history."""
        return self._conversation_service.clear_conversation(conversation_id)
    
    def get_conversation_history(self, conversation_id: str) -> Optional[str]:
        """Get formatted conversation history."""
        return self._conversation_service.get_formatted_history(conversation_id)
    
    async def process_memory_command(
        self,
        command: str,
        memory_context: MemoryContext
    ) -> Optional[str]:
        """Process natural language memory commands."""
        if not self._memory_service:
            return None
        
        result = await self._memory_service.process_natural_language_command(
            command, memory_context
        )
        
        if not result:
            return None
        
        if result.success:
            # Format response based on operation type
            if result.operation_type.value == "search":
                if result.entries:
                    lines = ["🧠 Here's what I recall:"]
                    for i, entry in enumerate(result.entries[:5], 1):
                        lines.append(f"  {i}. {entry.memory}")
                    return "\n".join(lines)
                else:
                    return "🤔 I don't have anything saved yet."
            elif result.operation_type.value == "add":
                return "✅ Remembered."
            elif result.operation_type.value == "delete":
                count = result.metadata.get("deleted_count", 0)
                return "🗑️ Forgotten." if count > 0 else "ℹ️ Nothing to forget matched."
            else:
                return "✅ Memory operation completed."
        else:
            return f"❌ Memory operation failed: {result.error}"
