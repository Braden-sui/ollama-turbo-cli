"""
Ollama Turbo client implementation with tool calling and streaming support.
"""

import json
import sys
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import ollama
from ollama import Client
import threading
import queue
import time
import atexit
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from .plugin_loader import TOOL_SCHEMAS, TOOL_FUNCTIONS
from .utils import with_retry, RetryableError, OllamaAPIError, truncate_text, format_conversation_history


class OllamaTurboClient:
    """Client for interacting with gpt-oss:120b via Ollama Turbo."""
    
    def __init__(self, api_key: str, model: str = "gpt-oss:120b", enable_tools: bool = True, show_trace: bool = False, reasoning: str = "high", quiet: bool = False, max_output_tokens: Optional[int] = None, ctx_size: Optional[int] = None, tool_print_limit: int = 200):
        """Initialize Ollama Turbo client.
        
        Args:
            api_key: Ollama API key for authentication
            model: Model name to use (default: gpt-oss:120b)
            enable_tools: Whether to enable tool calling capabilities
            show_trace: Whether to collect and print a separated reasoning trace
            reasoning: Reasoning effort directive ('low' | 'medium' | 'high')
            quiet: Reduce CLI noise (suppress helper prints)
            max_output_tokens: Limit on tokens to generate (maps to options.num_predict)
            ctx_size: Context window size (maps to options.num_ctx)
            tool_print_limit: CLI print truncation for tool outputs (characters)
        """
        self.api_key = api_key
        self.model = model
        self.enable_tools = enable_tools
        self.show_trace = show_trace
        self.quiet = quiet
        self.reasoning = reasoning if reasoning in {"low", "medium", "high"} else "high"
        self.trace: List[str] = []
        self.logger = logging.getLogger(__name__)
        self.max_output_tokens = max_output_tokens
        self.ctx_size = ctx_size
        self.tool_print_limit = tool_print_limit
        self._last_user_message: Optional[str] = None
        self._mem0_notice_shown: bool = False
        # Mem0 runtime flags/state (set in _init_mem0)
        self.mem0_enabled: bool = False
        self.mem0_debug: bool = False
        self.mem0_max_hits: int = 3
        self.mem0_search_timeout_ms: int = 200
        self.mem0_timeout_connect_ms: int = 1000
        self.mem0_timeout_read_ms: int = 2000
        self.mem0_add_queue_max: int = 256
        self._mem0_add_queue: Optional["queue.Queue"] = None
        self._mem0_worker: Optional[threading.Thread] = None
        self._mem0_worker_stop: threading.Event = threading.Event()
        self._mem0_last_sat_log: float = 0.0
        self._mem0_fail_count: int = 0
        self._mem0_breaker_threshold: int = 3
        self._mem0_breaker_cooldown_ms: int = 60000
        self._mem0_down_until_ms: int = 0
        self._mem0_breaker_tripped_logged: bool = False
        self._mem0_breaker_recovered_logged: bool = False
        self._last_mem_hash: Optional[str] = None
        self._mem0_search_workers = int(os.getenv('MEM0_SEARCH_WORKERS', '2') or '2')
        self._mem0_search_pool: Optional[ThreadPoolExecutor] = None
        
        # Initialize Ollama client with authentication
        # Note: Ollama Turbo uses Authorization header without 'Bearer' prefix
        self.client = Client(
            host='https://ollama.com',
            headers={'Authorization': api_key}
        )
        
        # Initialize conversation history with a system directive for reasoning effort
        self.conversation_history = [
            {
                'role': 'system',
                'content': (
                    f"Reasoning effort: {self.reasoning}. "
                    "Use tools when helpful. Do not expose chain-of-thought. Provide a direct answer, and include brief source URLs when tools are used."
                )
            }
        ]
        # Enforce local history window <= 10 turns (excluding initial system)
        try:
            raw_hist = os.getenv('MAX_CONVERSATION_HISTORY', '10')
            parsed_hist = int(raw_hist) if str(raw_hist).isdigit() else 10
        except Exception:
            parsed_hist = 10
        self.max_history = max(2, min(parsed_hist, 10))
        
        # Set up tools if enabled
        if enable_tools:
            self.tools = TOOL_SCHEMAS
            self.tool_functions = TOOL_FUNCTIONS
        else:
            self.tools = []
            self.tool_functions = {}
        
        # Initialize Mem0 memory system (optional)
        self._init_mem0()

        self.logger.info(f"Initialized client with model: {model}, tools enabled: {enable_tools}, reasoning={self.reasoning}, quiet={self.quiet}")
        # Initial trace state
        if self.show_trace:
            self.trace.append(f"client:init model={model} tools={'on' if enable_tools else 'off'} reasoning={self.reasoning} quiet={'on' if self.quiet else 'off'}")

    def _trace(self, event: str):
        """Record a structured, non-sensitive trace event."""
        if self.show_trace:
            self.trace.append(event)

    def _print_trace(self):
        """Print a separated reasoning trace section."""
        if self.show_trace and self.trace:
            # Send to stderr to avoid mixing with streamed stdout
            print("\n\n--- Reasoning Trace ---", file=sys.stderr, flush=True)
            for item in self.trace:
                print(f" • {item}", file=sys.stderr, flush=True)
            # Clear after printing to avoid duplication on next call
            self.trace = []
    
    def chat(self, message: str, stream: bool = False) -> str:
        """Send a message to the model and get a response.
        
        Args:
            message: User message to send
            stream: Whether to stream the response
            
        Returns:
            Model response as string
        """
        # Reset trace for this turn
        self.trace = [] if self.show_trace else []
        self._trace(f"chat:start stream={'on' if stream else 'off'}")

        # Inject relevant memories BEFORE user message (one system block per turn)
        self._inject_mem0_context(message)

        # Add user message to history
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
        # Track last user message for Mem0 capture
        self._last_user_message = message
        
        # Trim history if needed
        self._trim_history()
        
        try:
            if stream:
                result = self._handle_streaming_chat()
            else:
                result = self._handle_standard_chat()
            # Print separated trace after output
            self._print_trace()
            return result
        except Exception as e:
            self.logger.error(f"Chat error: {e}")
            error_msg = f"Error during chat: {str(e)}"
            self.conversation_history.append({
                'role': 'assistant',
                'content': error_msg
            })
            self._trace(f"chat:error {type(e).__name__}")
            self._print_trace()
            return error_msg
    
    @with_retry(max_retries=3)
    def _handle_standard_chat(self) -> str:
        """Handle non-streaming chat interaction."""
        try:
            # Make API call with tools if enabled
            kwargs = {
                'model': self.model,
                'messages': self.conversation_history,
            }
            # Generation options
            options = {}
            if self.max_output_tokens is not None:
                options['num_predict'] = self.max_output_tokens
            if self.ctx_size is not None:
                options['num_ctx'] = self.ctx_size
            if options:
                kwargs['options'] = options
            
            if self.enable_tools and self.tools:
                kwargs['tools'] = self.tools
            
            self._trace("request:standard")
            response = self.client.chat(**kwargs)
            
            # Extract message from response
            message = response.get('message', {})
            content = message.get('content', '')
            tool_calls = message.get('tool_calls', [])
            
            # Process tool calls if present
            if tool_calls and self.enable_tools:
                # Add assistant message with tool calls
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': content,
                    'tool_calls': tool_calls
                })
                names = [tc.get('function', {}).get('name') for tc in tool_calls]
                self._trace(f"tools:detected {len(tool_calls)} -> {', '.join(n for n in names if n)}")
                
                # Execute tools
                tool_results = self._execute_tool_calls(tool_calls)
                self._trace(f"tools:executed {len(tool_results)}")
                
                # Add tool results to history
                self.conversation_history.append({
                    'role': 'tool',
                    'content': '\n'.join(tool_results)
                })
                
                # Reprompt model to synthesize an answer using tool details
                self._trace("reprompt:after-tools")
                self.conversation_history.append({
                    'role': 'user',
                    'content': "Please use these details to answer the user's original question."
                })
                
                # Get final response after tool execution
                self._trace("request:final-after-tools")
                # Final response after tools with same options
                final_kwargs = {
                    'model': self.model,
                    'messages': self.conversation_history,
                }
                # Do not include tools in the final call; force a textual answer
                if options:
                    final_kwargs['options'] = options
                final_response = self.client.chat(**final_kwargs)
                
                final_content = final_response.get('message', {}).get('content', '')
                
                # Add final response to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': final_content
                })
                # Persist memory to Mem0
                self._mem0_add_after_response(self._last_user_message, final_content)
                
                # Return combined response
                return f"{content}\n\n[Tool Results]\n" + '\n'.join(tool_results) + f"\n\n{final_content}"
            else:
                # No tool calls, just add response to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': content
                })
                self._trace("tools:none")
                # Persist memory to Mem0
                self._mem0_add_after_response(self._last_user_message, content)
                return content
                
        except Exception as e:
            self.logger.error(f"Standard chat error: {e}")
            self._trace(f"standard:error {type(e).__name__}")
            raise RetryableError(f"API request failed: {e}")
    
    def _handle_streaming_chat(self) -> str:
        """Handle streaming chat interaction with tool support."""
        return self.handle_streaming_response(
            self._create_streaming_response(),
            tools_enabled=self.enable_tools
        )
    
    @with_retry(max_retries=3)
    def _create_streaming_response(self):
        """Create a streaming response from the API."""
        try:
            kwargs = {
                'model': self.model,
                'messages': self.conversation_history,
                'stream': True
            }
            # Generation options
            options = {}
            if self.max_output_tokens is not None:
                options['num_predict'] = self.max_output_tokens
            if self.ctx_size is not None:
                options['num_ctx'] = self.ctx_size
            if options:
                kwargs['options'] = options
            
            if self.enable_tools and self.tools:
                kwargs['tools'] = self.tools
            
            self._trace("request:stream:start")
            return self.client.chat(**kwargs)
        except Exception as e:
            self.logger.error(f"Streaming creation error: {e}")
            self._trace(f"stream:init:error {type(e).__name__}")
            raise RetryableError(f"Failed to create streaming response: {e}")
    
    def handle_streaming_response(self, response_stream, tools_enabled: bool = True) -> str:
        """Complete streaming response handler with tool call support."""
        full_content = ""
        tool_calls = []
        
        # For cleaner UX: if tools are enabled, buffer initial stream and only print
        # the final answer after tools (to avoid duplicate messages). If tools are
        # disabled, stream directly to stdout.
        if not self.quiet and not tools_enabled:
            print("🤖 Assistant: ", end="", flush=True)
        
        try:
            for chunk in response_stream:
                message = chunk.get('message', {})
                
                # Handle content streaming
                if message.get('content'):
                    content = message['content']
                    # Stream directly only when tools are not enabled; otherwise buffer
                    if not tools_enabled:
                        print(content, end="", flush=True)
                    full_content += content
                
                # Collect tool calls
                if message.get('tool_calls') and tools_enabled:
                    for tool_call in message['tool_calls']:
                        # Find if this tool call already exists (streaming may send in parts)
                        existing = False
                        for tc in tool_calls:
                            if tc.get('id') == tool_call.get('id'):
                                # Update existing tool call
                                tc.update(tool_call)
                                existing = True
                                break
                        if not existing:
                            tool_calls.append(tool_call)
            
            if not tools_enabled:
                print()  # New line after content
            
            # Process tool calls if any
            if tool_calls:
                if not self.show_trace and not self.quiet:
                    print("\n🔧 Processing tool calls...")
                names = [tc.get('function', {}).get('name') for tc in tool_calls]
                self._trace(f"tools:detected {len(tool_calls)} -> {', '.join(n for n in names if n)}")
                
                # Add assistant message with tool calls to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': full_content,
                    'tool_calls': tool_calls
                })
                
                # Execute tools and collect results
                tool_results = self._execute_tool_calls(tool_calls)
                self._trace(f"tools:executed {len(tool_results)}")
                
                # Add tool results to conversation
                tool_message = {
                    'role': 'tool',
                    'content': '\n'.join(tool_results)
                }
                self.conversation_history.append(tool_message)
                
                # Reprompt model to synthesize an answer using tool details
                self._trace("reprompt:after-tools")
                self.conversation_history.append({
                    'role': 'user',
                    'content': "Please use these details to answer the user's original question."
                })
                
                # Get final response after tool execution
                if not self.quiet:
                    # Print the assistant prefix only for the final visible response
                    print("\n🤖 Assistant: ", end="", flush=True)
                self._trace("request:final-after-tools:stream")
                final_stream_kwargs = {
                    'model': self.model,
                    'messages': self.conversation_history,
                    'stream': True
                }
                # Do not include tools in the final call; force a textual answer
                final_options = {}
                if self.max_output_tokens is not None:
                    final_options['num_predict'] = self.max_output_tokens
                if self.ctx_size is not None:
                    final_options['num_ctx'] = self.ctx_size
                if final_options:
                    final_stream_kwargs['options'] = final_options
                final_response_stream = self.client.chat(**final_stream_kwargs)
                
                final_content = ""
                for chunk in final_response_stream:
                    if chunk.get('message', {}).get('content'):
                        content = chunk['message']['content']
                        print(content, end="", flush=True)
                        final_content += content
                
                print()
                
                # Add final response to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': final_content
                })
                # Persist memory to Mem0
                self._mem0_add_after_response(self._last_user_message, final_content)
                
                return full_content + "\n\n[Tool Results]\n" + '\n'.join(tool_results) + "\n\n" + final_content
            
            else:
                # No tool calls
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': full_content
                })
                self._trace("tools:none")
                # If tools are disabled we already streamed to stdout; if enabled, we
                # buffered and should now print once to stdout.
                if tools_enabled:
                    if not self.quiet:
                        print("🤖 Assistant: ", end="", flush=True)
                    if full_content:
                        print(full_content)
                # Persist memory to Mem0
                self._mem0_add_after_response(self._last_user_message, full_content)
                return full_content
                
        except KeyboardInterrupt:
            print("\n⚠️ Streaming interrupted by user")
            self._trace("stream:interrupted")
            return full_content + "\n[Interrupted]"
        except Exception as e:
            # Fallback: if streaming fails (e.g., validation errors from client),
            # degrade gracefully to non-streaming standard chat without noisy CLI output.
            self.logger.debug(f"Streaming error encountered; falling back to non-streaming: {e}")
            self._trace("stream:error -> fallback")
            try:
                final = self._handle_standard_chat()
                # Print only the final response so the user still sees a complete answer
                if final:
                    print(final)
                self._trace("fallback:success")
                return (full_content + "\n" + final) if full_content else final
            except Exception as e2:
                # Silent CLI per preference; log details at DEBUG only
                self.logger.debug(f"Non-streaming fallback also failed: {e2}")
                self._trace("fallback:error")
                return full_content or ""
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[str]:
        """Execute tool calls and return results.
        
        Args:
            tool_calls: List of tool call dictionaries
            
        Returns:
            List of tool execution results
        """
        tool_results = []
        
        for i, tool_call in enumerate(tool_calls, 1):
            function_name = tool_call.get('function', {}).get('name')
            function_args = tool_call.get('function', {}).get('arguments', {})
            
            # Handle arguments that might be JSON strings
            if isinstance(function_args, str):
                try:
                    function_args = json.loads(function_args)
                except json.JSONDecodeError:
                    function_args = {}
            
            if not self.show_trace and not self.quiet:
                print(f"   {i}. Executing {function_name}({', '.join(f'{k}={v}' for k, v in function_args.items())})")
            self._trace(f"tool:exec {function_name}")
            
            if function_name in self.tool_functions:
                try:
                    result = self.tool_functions[function_name](**function_args)
                    tool_results.append(result)
                    if not self.show_trace and not self.quiet:
                        print(f"      ✅ Result: {truncate_text(result, self.tool_print_limit)}")
                    self._trace(f"tool:ok {function_name}")
                except Exception as e:
                    error_result = f"Error executing {function_name}: {str(e)}"
                    tool_results.append(error_result)
                    if not self.show_trace and not self.quiet:
                        print(f"      ❌ {error_result}")
                    self._trace(f"tool:error {function_name}")
            else:
                error_result = f"Unknown tool: {function_name}"
                tool_results.append(error_result)
                if not self.show_trace and not self.quiet:
                    print(f"      ❌ {error_result}")
                self._trace(f"tool:unknown {function_name}")
        
        return tool_results
    
    def _trim_history(self):
        """Trim conversation history while preserving key system blocks.

        Guarantees:
        - Preserve the very first system directive (if present)
        - Preserve the latest Mem0 "Relevant information:" system message (if present)
        - Preserve the last N messages (N = self.max_history, capped at 10)
        """
        if len(self.conversation_history) <= self.max_history:
            return

        first_system = self.conversation_history[0] if self.conversation_history and self.conversation_history[0].get('role') == 'system' else None
        # Find latest memory system block
        latest_mem_idx = None
        for i in range(len(self.conversation_history) - 1, -1, -1):
            msg = self.conversation_history[i]
            if msg.get('role') == 'system' and isinstance(msg.get('content'), str) and msg['content'].startswith("Relevant information:"):
                latest_mem_idx = i
                break

        last_N = self.conversation_history[-self.max_history:]
        new_hist: List[Dict[str, Any]] = []
        if first_system is not None:
            new_hist.append(first_system)
        if latest_mem_idx is not None and (latest_mem_idx < len(self.conversation_history)):
            mem_msg = self.conversation_history[latest_mem_idx]
            # Avoid duplication if already included in last_N or identical to first_system
            if mem_msg is not first_system and mem_msg not in last_N:
                new_hist.append(mem_msg)
        # Extend with last N (may include the memory block already)
        new_hist.extend(last_N)
        # De-duplicate while preserving order
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for m in new_hist:
            key = id(m)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(m)
        self.conversation_history = deduped
    
    def clear_history(self):
        """Clear conversation history."""
        # Preserve the system reasoning directive when clearing
        self.conversation_history = [
            {
                'role': 'system',
                'content': (
                    f"Reasoning effort: {self.reasoning}. "
                    "Use tools when helpful. Do not expose chain-of-thought. Output only final answers."
                )
            }
        ]
        self.logger.info("Conversation history cleared")
    
    def get_history(self) -> str:
        """Get formatted conversation history.
        
        Returns:
            Formatted conversation history string
        """
        return format_conversation_history(self.conversation_history)
    
    # ------------------ Mem0 Integration ------------------
    def _init_mem0(self) -> None:
        """Initialize Mem0 client and runtime settings from environment variables if available."""
        self.mem0_client = None
        self.mem0_enabled = False
        try:
            # Lazy import to avoid hard crash if package isn't installed yet
            try:
                from mem0 import MemoryClient  # type: ignore
            except Exception as ie:
                self.logger.debug(f"Mem0 client import skipped: {ie}")
                return
            # Feature flags and config
            mem0_enabled_flag = os.getenv('MEM0_ENABLED', '1').strip()
            if mem0_enabled_flag in {'0', 'false', 'False'}:
                return

            api_key = os.getenv('MEM0_API_KEY')
            if not api_key:
                # Mem0 is optional; proceed without it
                return
            # Runtime knobs
            self.mem0_debug = os.getenv('MEM0_DEBUG', '0').strip() in {'1', 'true', 'True'}
            self.mem0_max_hits = int(os.getenv('MEM0_MAX_HITS', '3') or '3')
            self.mem0_search_timeout_ms = int(os.getenv('MEM0_SEARCH_TIMEOUT_MS', '200') or '200')
            self.mem0_timeout_connect_ms = int(os.getenv('MEM0_TIMEOUT_CONNECT_MS', '1000') or '1000')
            self.mem0_timeout_read_ms = int(os.getenv('MEM0_TIMEOUT_READ_MS', '2000') or '2000')
            self.mem0_add_queue_max = int(os.getenv('MEM0_ADD_QUEUE_MAX', '256') or '256')
            self._mem0_breaker_threshold = int(os.getenv('MEM0_BREAKER_THRESHOLD', '3') or '3')
            self._mem0_breaker_cooldown_ms = int(os.getenv('MEM0_BREAKER_COOLDOWN_MS', '60000') or '60000')
            self._mem0_shutdown_flush_ms = int(os.getenv('MEM0_SHUTDOWN_FLUSH_MS', '3000') or '3000')

            org_id = os.getenv('MEM0_ORG_ID')
            project_id = os.getenv('MEM0_PROJECT_ID')
            if org_id or project_id:
                self.mem0_client = MemoryClient(api_key=api_key, org_id=org_id, project_id=project_id)
            else:
                self.mem0_client = MemoryClient(api_key=api_key)
            self.mem0_user_id = os.getenv('MEM0_USER_ID', 'Braden')
            self.mem0_agent_id = os.getenv('MEM0_AGENT_ID')
            self.mem0_app_id = os.getenv('MEM0_APP_ID')
            self.mem0_enabled = True
            self.logger.info("Mem0 initialized and enabled")

            # Initialize background worker for async adds
            self._mem0_add_queue = queue.Queue(maxsize=max(1, self.mem0_add_queue_max))
            self._mem0_worker_stop.clear()
            self._mem0_worker = threading.Thread(target=self._mem0_worker_loop, name="mem0-add-worker", daemon=True)
            self._mem0_worker.start()
            # Ensure graceful shutdown
            atexit.register(self._mem0_shutdown)
            # Create tiny pool for Mem0 searches
            self._mem0_search_pool = ThreadPoolExecutor(
                max_workers=max(1, self._mem0_search_workers),
                thread_name_prefix="mem0-search"
            )
            atexit.register(lambda: self._mem0_search_pool.shutdown(wait=False) if self._mem0_search_pool else None)
        except Exception as e:
            self.logger.error(f"Mem0 initialization failed: {e}")
            self.mem0_client = None
            self.mem0_enabled = False

    def _inject_mem0_context(self, user_message: str) -> None:
        """Search Mem0 for relevant memories and inject as a system message.

        - Minimal filters: only user_id until recall proven
        - Time-boxed search with hard deadline
        - Inject a single "Relevant information:" block per turn
        - Limit to top-K hits and total char budget
        """
        if not getattr(self, 'mem0_enabled', False) or not self.mem0_client:
            return
        # Circuit breaker: skip if down
        now_ms = int(time.time() * 1000)
        if self._mem0_down_until_ms and now_ms < self._mem0_down_until_ms:
            return
        start = time.time()
        try:
            # Remove any previous Mem0 injection blocks
            for idx in range(len(self.conversation_history) - 1, -1, -1):
                msg = self.conversation_history[idx]
                if msg.get('role') == 'system':
                    c = msg.get('content') or ''
                    if c.startswith("Relevant information:") or c.startswith("Relevant user memories"):
                        self.conversation_history.pop(idx)

            # Minimal filter: just user_id
            filters = {"user_id": getattr(self, 'mem0_user_id', 'Braden')}
            def _do_search():
                return self.mem0_client.search(
                    user_message,
                    version="v2",
                    filters=filters,
                    limit=self.mem0_max_hits  # server-side limit to shrink payload
                )

            try:
                if self._mem0_search_pool:
                    fut = self._mem0_search_pool.submit(_do_search)
                    related = fut.result(timeout=max(0.05, self.mem0_search_timeout_ms / 1000.0))
                else:
                    # very unlikely fallback
                    related = self.mem0_client.search(user_message, version="v2", filters=filters, limit=self.mem0_max_hits)
            except FuturesTimeout:
                # Treat timeout like a failure for breaker accounting
                self._trace("mem0:search:timeout")
                self._mem0_fail_count += 1
                if self.mem0_debug and not self.quiet:
                    dt_ms = int((time.time() - start) * 1000)
                    print(f"[mem0] search dt={dt_ms}ms hits=0 (timeout)")
                if self._mem0_fail_count >= self._mem0_breaker_threshold:
                    self._mem0_down_until_ms = int(time.time() * 1000) + self._mem0_breaker_cooldown_ms
                    if not self._mem0_breaker_tripped_logged:
                        self.logger.warning("Mem0 circuit breaker tripped; skipping calls temporarily")
                        self._mem0_breaker_tripped_logged = True
                        self._mem0_breaker_recovered_logged = False
                return
            # Take top-K and enforce char budget
            def _memtxt(m: Dict[str, Any]) -> str:
                return (
                    (m.get('memory') if isinstance(m, dict) else None)
                    or (m.get('text') if isinstance(m, dict) else None)
                    or (m.get('content') if isinstance(m, dict) else None)
                    or str(m)
                )
            texts = []
            for m in related or []:
                try:
                    txt = _memtxt(m)
                    if txt:
                        texts.append(str(txt))
                except Exception:
                    continue
            top_texts = texts[: max(1, self.mem0_max_hits)]
            # Truncate to ~800 chars total
            budget = 800
            acc = []
            used = 0
            for ttxt in top_texts:
                remaining = max(0, budget - used)
                if remaining <= 0:
                    break
                # leave room for bullets and newlines
                slice_len = min(len(ttxt), remaining)
                snip = ttxt[:slice_len]
                if len(snip) < len(ttxt):
                    snip = snip.rstrip() + "…"
                acc.append(f"- {snip}")
                used += len(snip) + 2
            if not acc:
                dt_ms = int((time.time() - start) * 1000)
                if self.mem0_debug and not self.quiet:
                    print(f"[mem0] search dt={dt_ms}ms hits=0")
                return
            context = "Relevant information:\n" + "\n".join(acc)
            self.conversation_history.append({'role': 'system', 'content': context})
            self._trace(f"mem0:inject {len(acc)}")
            # Success -> reset breaker
            if self._mem0_fail_count >= self._mem0_breaker_threshold and not self._mem0_breaker_recovered_logged:
                self.logger.info("Mem0 recovered; resuming calls")
                self._mem0_breaker_recovered_logged = True
                self._mem0_breaker_tripped_logged = False
            self._mem0_fail_count = 0
            self._mem0_down_until_ms = 0
            dt_ms = int((time.time() - start) * 1000)
            if self.mem0_debug and not self.quiet:
                print(f"[mem0] search dt={dt_ms}ms hits={len(acc)}")
        except Exception as e:
            # Failure path: log at debug and advance breaker
            self.logger.debug(f"Mem0 search failed: {e}")
            self._trace("mem0:inject:fail")
            self._mem0_fail_count += 1
            if self._mem0_fail_count >= self._mem0_breaker_threshold:
                self._mem0_down_until_ms = int(time.time() * 1000) + self._mem0_breaker_cooldown_ms
                if not self._mem0_breaker_tripped_logged:
                    self.logger.warning("Mem0 circuit breaker tripped; skipping calls temporarily")
                    self._mem0_breaker_tripped_logged = True
                    self._mem0_breaker_recovered_logged = False
            if not self._mem0_notice_shown and not self.quiet:
                print("⚠️ Mem0 unavailable; continuing without memory for this session.")
                self._mem0_notice_shown = True

    def _mem0_add_after_response(self, user_message: Optional[str], assistant_message: Optional[str]) -> None:
        """Queue the interaction to Mem0 for persistence (fire-and-forget)."""
        if not getattr(self, 'mem0_enabled', False) or not self.mem0_client:
            return
        if not user_message and not assistant_message:
            return
        messages: List[Dict[str, str]] = []
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})
        metadata: Dict[str, Any] = {
            "source": "chat",
            "category": "inferred",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if getattr(self, 'mem0_app_id', None):
            metadata["app_id"] = self.mem0_app_id
        if getattr(self, 'mem0_agent_id', None):
            metadata["agent_id"] = self.mem0_agent_id
        self._mem0_enqueue_add(messages, metadata)
        self._trace("mem0:add:queued")

    def _mem0_enqueue_add(self, messages: List[Dict[str, str]], metadata: Dict[str, Any]) -> None:
        """Enqueue an add job; drop oldest if saturated and log once per minute."""
        try:
            if not self._mem0_add_queue:
                return
            job = {
                "messages": messages,
                "metadata": metadata,
            }
            try:
                self._mem0_add_queue.put_nowait(job)
            except queue.Full:
                # Drop oldest
                try:
                    _ = self._mem0_add_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._mem0_add_queue.put_nowait(job)
                except Exception:
                    pass
                now = time.time()
                if now - self._mem0_last_sat_log > 60:
                    self.logger.warning("mem0 add queue saturated; dropping oldest job")
                    self._mem0_last_sat_log = now
        except Exception as e:
            self.logger.debug(f"Mem0 enqueue failed: {e}")

    def _mem0_worker_loop(self) -> None:
        while not self._mem0_worker_stop.is_set():
            try:
                job = None
                try:
                    job = self._mem0_add_queue.get(timeout=0.25) if self._mem0_add_queue else None
                except queue.Empty:
                    continue
                if not job:
                    continue
                start = time.time()
                try:
                    ids = self._mem0_execute_add(job.get("messages") or [], job.get("metadata") or {})
                    # Enforce metadata after add
                    self._mem0_enforce_metadata(ids, job.get("metadata") or {})
                    if self.mem0_debug:
                        dt_ms = int((time.time() - start) * 1000)
                        self.logger.debug(f"[mem0] add dt={dt_ms}ms ids={ids}")
                    # Success -> reset breaker
                    self._mem0_fail_count = 0
                    self._mem0_down_until_ms = 0
                except Exception as we:
                    self.logger.debug(f"Mem0 add worker error: {we}")
                    self._mem0_fail_count += 1
                    if self._mem0_fail_count >= self._mem0_breaker_threshold:
                        self._mem0_down_until_ms = int(time.time() * 1000) + self._mem0_breaker_cooldown_ms
                        if not self._mem0_breaker_tripped_logged:
                            self.logger.warning("Mem0 circuit breaker tripped; skipping calls temporarily")
                            self._mem0_breaker_tripped_logged = True
                            self._mem0_breaker_recovered_logged = False
            except Exception:
                # Never crash the worker
                continue

    def _mem0_execute_add(self, messages: List[Dict[str, str]], metadata: Dict[str, Any]) -> List[str]:
        if not self.mem0_client:
            return []
        user_id = getattr(self, 'mem0_user_id', 'Braden')

        # First attempt: full-feature call
        kwargs = {"messages": messages, "user_id": user_id, "version": "v2", "metadata": metadata}
        if getattr(self, 'mem0_agent_id', None):
            kwargs["agent_id"] = self.mem0_agent_id
        try:
            res = self.mem0_client.add(**kwargs)
        except TypeError:
            # Second attempt: drop agent_id, keep version/metadata
            try:
                kwargs.pop("agent_id", None)
                res = self.mem0_client.add(**kwargs)
            except TypeError:
                # Third attempt: minimal signature (no version/metadata); enforce metadata later via update()
                kwargs_min = {"messages": messages, "user_id": user_id}
                try:
                    res = self.mem0_client.add(**kwargs_min)
                except Exception as e:
                    raise e

        # Extract IDs (best-effort)
        ids: List[str] = []
        try:
            if isinstance(res, dict):
                if "id" in res:
                    ids.append(str(res["id"]))
                elif "data" in res and isinstance(res["data"], dict) and "id" in res["data"]:
                    ids.append(str(res["data"]["id"]))
                elif "items" in res and isinstance(res["items"], list):
                    for it in res["items"]:
                        if isinstance(it, dict) and "id" in it:
                            ids.append(str(it["id"]))
            elif isinstance(res, list):
                for it in res:
                    if isinstance(it, dict) and "id" in it:
                        ids.append(str(it["id"]))
        except Exception:
            pass
        return ids

    def _mem0_enforce_metadata(self, ids: List[str], metadata: Dict[str, Any]) -> None:
        if not ids or not self.mem0_client:
            return
        backoffs = [0.5, 1.0, 2.0, 2.0, 2.0]
        for mid in ids:
            ok = False
            for delay in backoffs:
                start_meta = time.time()
                try:
                    # Some SDKs may accept metadata kw; others may nest under data
                    try:
                        self.mem0_client.update(memory_id=mid, metadata=metadata)
                    except TypeError:
                        # Fallback to text/data param-less update with metadata if supported
                        self.mem0_client.update(memory_id=mid, **{"metadata": metadata})
                    ok = True
                    if self.mem0_debug:
                        dt_ms = int((time.time() - start_meta) * 1000)
                        self.logger.debug(f"[mem0] update(meta) dt={dt_ms}ms id={mid}")
                    break
                except Exception:
                    time.sleep(delay)
            if not ok:
                self.logger.warning(f"Mem0 metadata enforcement failed for id={mid}")

    def _normalize_fact(self, text: str) -> str:
        t = ' '.join(text.strip().split())
        return t.lower()

    def _mem0_handle_command(self, cmdline: str) -> None:
        """Handle /mem CLI commands."""
        if not cmdline.startswith('/mem'):
            return
        parts = cmdline.split()
        if len(parts) == 1:
            print("ℹ️ Usage: /mem [list|search|add|get|update|delete|clear|link|export|import] ...")
            return
        if not getattr(self, 'mem0_enabled', False) or not self.mem0_client:
            print("⚠️ Mem0 is not configured. Set MEM0_API_KEY to enable.")
            return
        sub = parts[1].lower()
        try:
            if sub == 'list':
                # Optional inline query to filter display
                query = ' '.join(parts[2:]).strip() if len(parts) > 2 else ''
                filters = {"user_id": self.mem0_user_id}
                items = self.mem0_client.get_all(filters=filters, version="v2")
                if not items:
                    print("📭 No memories found.")
                    return
                print("🧠 Memories:")
                shown = 0
                for i, it in enumerate(items, 1):
                    mem_text = it.get('memory') or (it.get('data') or {}).get('memory')
                    if query and mem_text and query.lower() not in mem_text.lower():
                        continue
                    print(f"  {i}. {it.get('id')}: {truncate_text(mem_text or '', 200)}")
                    shown += 1
                if shown == 0:
                    print("  (no items match your query)")
            elif sub == 'search':
                query = ' '.join(parts[2:]).strip()
                if not query:
                    print("Usage: /mem search <query>")
                    return
                filters = {"user_id": self.mem0_user_id}
                results = self.mem0_client.search(query, version="v2", filters=filters)
                if not results:
                    print("🔍 No matching memories.")
                    return
                print("🔍 Top matches:")
                for i, it in enumerate(results[:10], 1):
                    mem_text = it.get('memory') or (it.get('data') or {}).get('memory')
                    print(f"  {i}. {it.get('id')}: {truncate_text(mem_text or '', 200)}")
            elif sub == 'add':
                text = ' '.join(parts[2:]).strip()
                if not text:
                    print("Usage: /mem add <text>")
                    return
                self.mem0_client.add([{"role": "user", "content": text}], user_id=self.mem0_user_id, version="v2", metadata={"source": "cli", "category": "manual"})
                print("✅ Memory added.")
            elif sub == 'get':
                mem_id = (parts[2] if len(parts) > 2 else '').strip()
                if not mem_id:
                    print("Usage: /mem get <memory_id>")
                    return
                item = self.mem0_client.get(memory_id=mem_id)
                print(json.dumps(item, indent=2))
            elif sub == 'update':
                if len(parts) < 4:
                    print("Usage: /mem update <memory_id> <new text>")
                    return
                mem_id = parts[2]
                new_text = ' '.join(parts[3:])
                # Support both 'text' (platform docs) and 'data' (OSS quickstart) parameter names
                try:
                    self.mem0_client.update(memory_id=mem_id, text=new_text)
                except TypeError:
                    self.mem0_client.update(memory_id=mem_id, data=new_text)
                print("✅ Memory updated.")
            elif sub == 'delete':
                mem_id = (parts[2] if len(parts) > 2 else '').strip()
                if not mem_id:
                    print("Usage: /mem delete <memory_id>")
                    return
                self.mem0_client.delete(memory_id=mem_id)
                print("🗑️ Memory deleted.")
            elif sub == 'link':
                if len(parts) < 4:
                    print("Usage: /mem link <id1> <id2>")
                    return
                id1, id2 = parts[2], parts[3]
                try:
                    # Some SDKs may require user_id; include when available
                    self.mem0_client.link(memory1_id=id1, memory2_id=id2, user_id=self.mem0_user_id)
                    print("🔗 Memories linked.")
                except Exception as e:
                    print("ℹ️ Linking not available in this plan/SDK.")
            elif sub == 'export':
                # Optional path
                out_path = (parts[2] if len(parts) > 2 else '').strip()
                if not out_path:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = f"mem0_export_{ts}.json"
                filters = {"user_id": self.mem0_user_id}
                items = self.mem0_client.get_all(filters=filters, version="v2")
                payload = []
                for it in items or []:
                    payload.append({
                        "id": it.get("id"),
                        "memory": it.get("memory") or (it.get('data') or {}).get('memory'),
                        "metadata": it.get("metadata") or {},
                        "created_at": it.get("created_at"),
                        "updated_at": it.get("updated_at"),
                    })
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"📦 Exported {len(payload)} memories to {out_path}")
            elif sub == 'import':
                if len(parts) < 3:
                    print("Usage: /mem import <path.json>")
                    return
                in_path = parts[2]
                if not os.path.exists(in_path):
                    print("❌ File not found.")
                    return
                with open(in_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                count = 0
                for item in data if isinstance(data, list) else []:
                    text = item.get('memory')
                    if not text:
                        continue
                    meta = item.get('metadata') or {}
                    # Preserve app/agent ids
                    if getattr(self, 'mem0_app_id', None):
                        meta.setdefault('app_id', self.mem0_app_id)
                    if getattr(self, 'mem0_agent_id', None):
                        meta.setdefault('agent_id', self.mem0_agent_id)
                    try:
                        self.mem0_client.add([{"role": "user", "content": text}], user_id=self.mem0_user_id, version="v2", metadata=meta)
                        count += 1
                    except Exception:
                        continue
                print(f"✅ Imported {count} memories.")
            elif sub in ('clear', 'delete-all'):
                self.mem0_client.delete_all(user_id=self.mem0_user_id)
                print("🧹 All memories for user cleared.")
            else:
                print("Unknown /mem subcommand. Use list|search|add|get|update|delete|clear|link|export|import")
        except Exception as e:
            print(f"❌ Mem0 command error: {e}")

    def _handle_mem0_nlu(self, text: str) -> bool:
        """Very simple natural-language detection for memory actions.
        Returns True if handled and chat should be skipped.
        """
        if not getattr(self, 'mem0_enabled', False) or not self.mem0_client:
            return False
        lower = text.lower().strip()
        try:
            # Add memory
            if lower.startswith("remember that ") or lower.startswith("remember ") or lower.startswith("please remember ") or lower.startswith("save this:"):
                # Remove leading directive words
                prefixes = ["remember that ", "remember ", "please remember ", "save this:"]
                mem_text = text
                for p in prefixes:
                    if lower.startswith(p):
                        mem_text = text[len(p):].strip()
                        break
                if mem_text:
                    # De-duplicate simple repeats using a short hash of normalized fact
                    try:
                        h = hashlib.sha1(self._normalize_fact(mem_text).encode('utf-8')).hexdigest()[:10]
                        if self._last_mem_hash != h:
                            metadata = {"source": "nlu", "category": "manual", "timestamp": datetime.utcnow().isoformat() + "Z"}
                            if getattr(self, 'mem0_app_id', None):
                                metadata["app_id"] = self.mem0_app_id
                            if getattr(self, 'mem0_agent_id', None):
                                metadata["agent_id"] = self.mem0_agent_id
                            self._mem0_enqueue_add([{"role": "user", "content": mem_text}], metadata)
                            self._last_mem_hash = h
                    except Exception:
                        # Fallback enqueue without dedupe
                        metadata = {"source": "nlu", "category": "manual", "timestamp": datetime.utcnow().isoformat() + "Z"}
                        if getattr(self, 'mem0_app_id', None):
                            metadata["app_id"] = self.mem0_app_id
                        if getattr(self, 'mem0_agent_id', None):
                            metadata["agent_id"] = self.mem0_agent_id
                        self._mem0_enqueue_add([{"role": "user", "content": mem_text}], metadata)
                    print("✅ I'll remember that.")
                    return True
            # Forget memory
            if lower.startswith("forget that ") or lower.startswith("forget "):
                prefixes = ["forget that ", "forget "]
                q = text
                for p in prefixes:
                    if lower.startswith(p):
                        q = text[len(p):].strip()
                        break
                if q:
                    filters = {"user_id": self.mem0_user_id}
                    results = self.mem0_client.search(q, version="v2", filters=filters)
                    deleted = 0
                    for it in results or []:
                        mem_text = it.get('memory') or (it.get('data') or {}).get('memory') or ''
                        if q.lower() in mem_text.lower():
                            try:
                                self.mem0_client.delete(memory_id=it.get('id'))
                                deleted += 1
                            except Exception:
                                pass
                    print("🗑️ Forgotten." if deleted else "ℹ️ Nothing to forget matched.")
                    return True
            # Update memory (pattern: update <subject> to <new text>)
            if lower.startswith("update ") and " to " in lower:
                body = text[7:].strip()
                parts2 = body.split(" to ", 1)
                if len(parts2) == 2:
                    subject, new_text = parts2[0].strip(), parts2[1].strip()
                    filters = {"user_id": self.mem0_user_id}
                    results = self.mem0_client.search(subject, version="v2", filters=filters)
                    if results:
                        mem_id = results[0].get('id')
                        try:
                            self.mem0_client.update(memory_id=mem_id, text=new_text)
                        except TypeError:
                            self.mem0_client.update(memory_id=mem_id, data=new_text)
                        print("✅ Updated.")
                    else:
                        # If no match, add new
                        self.mem0_client.add([{"role": "user", "content": new_text}], user_id=self.mem0_user_id, version="v2", metadata={"source": "nlu", "category": "manual"})
                        print("✅ Not found; added as new.")
                    return True
            # List memories
            if lower.startswith("list memories") or lower.startswith("show memories"):
                filters = {"user_id": self.mem0_user_id}
                items = self.mem0_client.get_all(filters=filters, version="v2")
                if not items:
                    print("📭 No memories found.")
                    return True
                print("🧠 Memories:")
                for i, it in enumerate(items[:10], 1):
                    mem_text = it.get('memory') or (it.get('data') or {}).get('memory')
                    print(f"  {i}. {mem_text}")
                return True
            # Link memories (requires IDs)
            if lower.startswith("link "):
                parts2 = text.split()
                if len(parts2) >= 3:
                    id1 = parts2[1]
                    id2 = parts2[2] if len(parts2) >= 3 else None
                    if id1 and id2:
                        try:
                            self.mem0_client.link(memory1_id=id1, memory2_id=id2, user_id=self.mem0_user_id)
                            print("🔗 Linked.")
                        except Exception:
                            print("ℹ️ Linking not available in this plan/SDK.")
                        return True
            # Search memories (explicit)
            if lower.startswith("search memories for "):
                q = text[len("search memories for "):].strip()
                if q:
                    filters = {"user_id": self.mem0_user_id}
                    results = self.mem0_client.search(q, version="v2", filters=filters)
                    if not results:
                        print("🔍 No matching memories.")
                        return True
                    print("🔍 Matches:")
                    for i, it in enumerate(results[:5], 1):
                        mem_text = it.get('memory') or (it.get('data') or {}).get('memory')
                        print(f"  {i}. {mem_text}")
                    return True
            # Query memories
            queries = {"what do you know about me", "what did i tell you", "do you remember", "what do you remember about me"}
            if any(q in lower for q in queries):
                filters = {"user_id": self.mem0_user_id}
                results = self.mem0_client.search(text, version="v2", filters=filters)
                if not results:
                    print("🤔 I don't have anything saved yet.")
                    return True
                print("🧠 Here's what I recall:")
                for i, it in enumerate(results[:5], 1):
                    mem_text = it.get('memory') or (it.get('data') or {}).get('memory')
                    print(f"  {i}. {mem_text}")
                return True
        except Exception as e:
            self.logger.debug(f"Mem0 NLU handler error: {e}")
            return False
        return False
    
    def interactive_mode(self):
        """Run interactive chat mode."""
        if not self.quiet:
            print("🚀 Ollama Turbo CLI - Interactive Mode")
            print(f"📝 Model: {self.model}")
            print(f"🔧 Tools: {'Enabled' if self.enable_tools else 'Disabled'}")
            print("💡 Commands: 'quit'/'exit' to exit, 'clear' to clear history, 'history' to show history, '/mem ...' for memory ops")
            if not getattr(self, 'mem0_enabled', False):
                print("Mem0: disabled (no key)")
            else:
                print(f"Mem0: enabled (user: {self.mem0_user_id})")
            print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle /mem commands
                if user_input.lower().startswith('/mem'):
                    self._mem0_handle_command(user_input)
                    continue

                # Natural language memory handlers
                if self._handle_mem0_nlu(user_input):
                    continue

                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    if not self.quiet:
                        print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    if not self.quiet:
                        print("✅ History cleared")
                    continue
                elif user_input.lower() == 'history':
                    if not self.quiet:
                        print("\n📜 Conversation History:")
                        print(self.get_history())
                    continue
                
                # Send message to model
                if not self.quiet:
                    print()  # Empty line for better formatting
                response = self.chat(user_input, stream=True)
                
                # Response is already printed during streaming
                
            except KeyboardInterrupt:
                if not self.quiet:
                    print("\n\n⚠️ Use 'quit' or 'exit' to leave the chat")
                continue
            except Exception as e:
                self.logger.error(f"Interactive mode error: {e}")
                if not self.quiet:
                    print(f"\n❌ Error: {str(e)}")
                continue

    # ----- Mem0 shutdown -----
    def _mem0_shutdown(self) -> None:
        try:
            if not self._mem0_add_queue:
                return
            # Signal stop and drain for a bounded duration
            self._mem0_worker_stop.set()
            deadline = time.time() + max(0.0, self._mem0_shutdown_flush_ms / 1000.0) if hasattr(self, '_mem0_shutdown_flush_ms') else time.time() + 3.0
            while time.time() < deadline:
                try:
                    job = self._mem0_add_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    ids = self._mem0_execute_add(job.get("messages") or [], job.get("metadata") or {})
                    self._mem0_enforce_metadata(ids, job.get("metadata") or {})
                except Exception:
                    pass
            # Let worker exit naturally
        except Exception:
            pass
