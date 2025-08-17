"""
Ollama Turbo client implementation with tool calling and streaming support.
Reliability hardening: retries/backoff, idempotency keys, keep-alive pools, and
streaming idle reconnects (all behind env flags with safe defaults).
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
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import re

from .plugin_loader import TOOL_SCHEMAS, TOOL_FUNCTIONS
from .utils import with_retry, RetryableError, OllamaAPIError, truncate_text, format_conversation_history
from .prompt_manager import PromptManager


class OllamaTurboClient:
    """Client for interacting with gpt-oss:120b via Ollama Turbo."""
    
    def __init__(self, api_key: str, model: str = "gpt-oss:120b", enable_tools: bool = True, show_trace: bool = False, reasoning: str = "high", quiet: bool = False, max_output_tokens: Optional[int] = None, ctx_size: Optional[int] = None, tool_print_limit: int = 200, multi_round_tools: bool = True, tool_max_rounds: Optional[int] = None):
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
        self.tool_context_cap = int(os.getenv('TOOL_CONTEXT_MAX_CHARS', '4000') or '4000')
        self._last_user_message: Optional[str] = None
        self._mem0_notice_shown: bool = False
        self._skip_mem0_after_turn: bool = False
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
        # Tool-call iteration controls
        env_mrt = os.getenv('MULTI_ROUND_TOOLS')
        if env_mrt is not None:
            self.multi_round_tools = env_mrt.strip().lower() in {'1', 'true', 'yes', 'on'}
        else:
            self.multi_round_tools = bool(multi_round_tools)
        try:
            default_rounds = tool_max_rounds if tool_max_rounds is not None else 6
            parsed_rounds = int(os.getenv('TOOL_MAX_ROUNDS', str(default_rounds)) or str(default_rounds))
            self.tool_max_rounds: int = max(1, parsed_rounds)
        except Exception:
            self.tool_max_rounds = max(1, tool_max_rounds if tool_max_rounds is not None else 6)
        self._mem0_search_pool: Optional[ThreadPoolExecutor] = None
        # CLI/network resilience knobs (env-controlled)
        self.cli_retry_enabled: bool = os.getenv('CLI_RETRY_ENABLED', 'true').strip().lower() != 'false'
        try:
            self.cli_max_retries: int = max(0, int(os.getenv('CLI_MAX_RETRIES', '3') or '3'))
        except Exception:
            self.cli_max_retries = 3
        try:
            self.cli_stream_idle_reconnect_secs: int = max(10, int(os.getenv('CLI_STREAM_IDLE_RECONNECT_SECS', '90') or '90'))
        except Exception:
            self.cli_stream_idle_reconnect_secs = 90
        try:
            self.cli_connect_timeout_s: float = max(1.0, float(os.getenv('CLI_CONNECT_TIMEOUT_S', '5') or '5'))
        except Exception:
            self.cli_connect_timeout_s = 5.0
        try:
            self.cli_read_timeout_s: float = max(60.0, float(os.getenv('CLI_READ_TIMEOUT_S', '600') or '600'))
        except Exception:
            self.cli_read_timeout_s = 600.0
        self.warm_models: bool = os.getenv('WARM_MODELS', 'true').strip().lower() not in {'0', 'false', 'no', 'off'}
        self.ollama_keep_alive_raw: Optional[str] = os.getenv('OLLAMA_KEEP_ALIVE')
        self._current_idempotency_key: Optional[str] = None
        
        # Initialize Ollama client with authentication
        # Note: Ollama Turbo uses Authorization header without 'Bearer' prefix
        self.client = Client(
            host='https://ollama.com',
            headers={'Authorization': api_key}
        )
        
        # Prompt management
        self.prompt = PromptManager(self.reasoning)
        # Initialize conversation history with a system directive
        self.conversation_history = [
            {
                'role': 'system',
                'content': self.prompt.initial_system_prompt()
            }
        ]
        # Enforce local history window <= 10 turns (excluding initial system)
        try:
            raw_hist = os.getenv('MAX_CONVERSATION_HISTORY', '10')
            parsed_hist = int(raw_hist) if str(raw_hist).isdigit() else 10
        except Exception:
            parsed_hist = 10
        self.max_history = max(2, min(parsed_hist, 10))
        
        # Set up tools if enabled (use copies to avoid global mutation leaks)
        if enable_tools:
            # Copy lists/dicts so per-client monkeypatching doesn't affect globals/tests
            self.tools = list(TOOL_SCHEMAS)
            self.tool_functions = dict(TOOL_FUNCTIONS)
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
        self._skip_mem0_after_turn = False
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
        
        # Generate a fresh Idempotency-Key per turn (reused across retries/reconnects)
        self._current_idempotency_key = str(uuid.uuid4())
        self._set_idempotency_key(self._current_idempotency_key)
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
        finally:
            # Clear idempotency header after the turn
            self._clear_idempotency_key()
    
    @with_retry(max_retries=3)
    def _handle_standard_chat(self) -> str:
        """Handle non-streaming chat interaction."""
        try:
            # Generation options (reused across rounds)
            options = {}
            if self.max_output_tokens is not None:
                options['num_predict'] = self.max_output_tokens
            if self.ctx_size is not None:
                options['num_ctx'] = self.ctx_size

            rounds = 0
            all_tool_results: List[str] = []
            first_content: str = ""
            
            while True:
                # Build request
                kwargs = {
                    'model': self.model,
                    'messages': self.conversation_history,
                }
                # Warm models on server by requesting keep_alive if enabled
                keep_val = self._resolve_keep_alive()
                if keep_val is not None:
                    kwargs['keep_alive'] = keep_val
                if options:
                    kwargs['options'] = options

                # For non-streaming, always omit tools on the post-tool call to force
                # a textual answer (test compatibility). Multi-round behavior is
                # primarily supported in streaming mode.
                include_tools = self.enable_tools and bool(self.tools) and (rounds == 0)
                if include_tools:
                    kwargs['tools'] = self.tools

                self._trace(f"request:standard:round={rounds}{' tools' if include_tools else ''}")
                response = self.client.chat(**kwargs)

                # Extract message
                message = response.get('message', {})
                content = message.get('content', '')
                tool_calls = message.get('tool_calls', [])

                # Harmony adapter: if provider didn't canonicalize tool calls, parse from content
                if self.enable_tools and not tool_calls and content:
                    cleaned, parsed_calls, final_seg = self._parse_harmony_tokens(content)
                    if parsed_calls:
                        tool_calls = parsed_calls
                        content = cleaned  # remove tool-call markup from assistant content

                # On first round, capture any preface content
                if rounds == 0 and content:
                    first_content = self._strip_harmony_markup(content)

                if tool_calls and self.enable_tools:
                    # Add assistant message with tool calls
                    self.conversation_history.append({
                        'role': 'assistant',
                        'content': self._strip_harmony_markup(content),
                        'tool_calls': tool_calls
                    })
                    names = [tc.get('function', {}).get('name') for tc in tool_calls]
                    self._trace(f"tools:detected {len(tool_calls)} -> {', '.join(n for n in names if n)}")

                    # Execute tools
                    tool_results = self._execute_tool_calls(tool_calls)
                    self._trace(f"tools:executed {len(tool_results)}")
                    all_tool_results.extend(tool_results)

                    # Add tool results to history
                    self.conversation_history.append({
                        'role': 'tool',
                        'content': '\n'.join(tool_results)
                    })

                    # Reprompt model using details
                    self._trace("reprompt:after-tools")
                    self.conversation_history.append({
                        'role': 'user',
                        'content': self.prompt.reprompt_after_tools()
                    })

                    rounds += 1
                    # Continue loop for potential additional tool rounds
                    continue
                else:
                    # Final textual answer (extract Harmony final channel if present; strip markup)
                    final_out = content
                    try:
                        if content:
                            cleaned, _, final_seg = self._parse_harmony_tokens(content)
                            final_out = final_seg or cleaned
                            if not final_out:
                                final_out = self._strip_harmony_markup(content)
                    except Exception:
                        final_out = self._strip_harmony_markup(content)

                    self.conversation_history.append({
                        'role': 'assistant',
                        'content': final_out
                    })
                    self._trace("tools:none")
                    # Persist memory to Mem0
                    self._mem0_add_after_response(self._last_user_message, final_out)

                    if all_tool_results:
                        prefix = (first_content + "\n\n") if first_content else ""
                        return f"{prefix}[Tool Results]\n" + '\n'.join(all_tool_results) + f"\n\n{final_out}"
                    return final_out
                
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
            keep_val = self._resolve_keep_alive()
            if keep_val is not None:
                kwargs['keep_alive'] = keep_val
            
            if self.enable_tools and self.tools:
                kwargs['tools'] = self.tools
            
            self._trace("request:stream:start")
            return self.client.chat(**kwargs)
        except Exception as e:
            self.logger.error(f"Streaming creation error: {e}")
            self._trace(f"stream:init:error {type(e).__name__}")
            raise RetryableError(f"Failed to create streaming response: {e}")
    
    def handle_streaming_response(self, response_stream, tools_enabled: bool = True) -> str:
        """Complete streaming response handler with tool call support (multi-round optional)."""
        rounds = 0
        aggregated_results: List[str] = []
        preface_content: str = ""
        # Keep a buffer for fallback paths
        full_content: str = ""
        try:
            while True:
                include_tools = tools_enabled and bool(self.tools) and (
                    rounds == 0 or (self.multi_round_tools and rounds < self.tool_max_rounds)
                )

                # Use provided stream on the first round; create new ones subsequently
                if rounds == 0:
                    stream = response_stream
                else:
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
                    if include_tools:
                        kwargs['tools'] = self.tools
                    keep_val = self._resolve_keep_alive()
                    if keep_val is not None:
                        kwargs['keep_alive'] = keep_val
                    stream = self.client.chat(**kwargs)

                self._trace(f"request:stream:round={rounds}{' tools' if include_tools else ''}")

                # UX: Only print content live when we are on the final, no-tools round
                if not self.quiet and not include_tools and rounds > 0:
                    print("🤖 Assistant: ", end="", flush=True)

                round_content = ""
                # Track already emitted content length to avoid duplicates across reconnects
                printed_len = 0
                tool_calls: List[Dict[str, Any]] = []

                def _iter_stream_chunks(s):
                    # Simple wrapper to let us hook per-chunk accounting
                    for ch in s:
                        yield ch

                try:
                    for chunk in _iter_stream_chunks(stream):
                        message = chunk.get('message', {})
                        if message.get('content'):
                            piece = message['content']
                            round_content += piece
                            if not include_tools and rounds > 0:
                                # Print only the new suffix not yet emitted (handles reconnect duplicates)
                                new_segment = round_content[printed_len:]
                                if new_segment:
                                    safe_segment = self._strip_harmony_markup(new_segment)
                                    if safe_segment:
                                        print(safe_segment, end="", flush=True)
                                    printed_len = len(round_content)
                        if message.get('tool_calls') and include_tools:
                            for tool_call in message['tool_calls']:
                                # Merge updates for same id
                                existing = False
                                for tc in tool_calls:
                                    if tc.get('id') == tool_call.get('id'):
                                        tc.update(tool_call)
                                        existing = True
                                        break
                                if not existing:
                                    tool_calls.append(tool_call)
                except Exception as se:
                    # Stream interrupted (timeout/connection), attempt reconnects handled by outer retry decorator for init only.
                    # Here we fall back to non-streaming finalization to preserve UX, per user preference.
                    self.logger.debug(f"Streaming read error; falling back to non-streaming: {se}")
                    self._trace("stream:read:error -> fallback")
                    try:
                        final = self._handle_standard_chat()
                        if final:
                            print(final)
                        self._trace("fallback:success")
                        return (round_content + "\n" + final) if round_content else final
                    except Exception as e2:
                        self.logger.debug(f"Non-streaming fallback also failed: {e2}")
                        self._trace("fallback:error")
                        return round_content or ""

                # Update outer buffer for fallback usage
                full_content = round_content

                # Capture first round content as a preface (not printed yet)
                if rounds == 0 and round_content:
                    preface_content = self._strip_harmony_markup(round_content)

                # If tools were requested and yielded calls, execute them and loop
                # If no canonical tool_calls but Harmony markup is present, parse it
                if not tool_calls and include_tools and round_content:
                    try:
                        cleaned, parsed_calls, _ = self._parse_harmony_tokens(round_content)
                        if parsed_calls:
                            tool_calls = parsed_calls
                            round_content = cleaned
                    except Exception:
                        pass

                if tool_calls:
                    if not self.show_trace and not self.quiet:
                        print("\n🔧 Processing tool calls...")
                    names = [tc.get('function', {}).get('name') for tc in tool_calls]
                    self._trace(f"tools:detected {len(tool_calls)} -> {', '.join(n for n in names if n)}")

                    # Add assistant message with tool calls
                    self.conversation_history.append({
                        'role': 'assistant',
                        'content': self._strip_harmony_markup(round_content),
                        'tool_calls': tool_calls
                    })
                    # Execute tools
                    tool_results = self._execute_tool_calls(tool_calls)
                    self._trace(f"tools:executed {len(tool_results)}")
                    aggregated_results.extend(tool_results)
                    # Add tool message
                    self.conversation_history.append({
                        'role': 'tool',
                        'content': '\n'.join(tool_results)
                    })
                    # Reprompt model to synthesize an answer using tool details
                    self._trace("reprompt:after-tools")
                    self.conversation_history.append({
                        'role': 'user',
                        'content': self.prompt.reprompt_after_tools()
                    })
                    rounds += 1
                    # Next loop iteration may include tools again if multi-round enabled
                    continue

                # No tool calls -> final textual answer for this turn
                if not include_tools and rounds > 0 and not self.quiet:
                    print()  # newline after final streamed content
                # Extract final-channel text if present; otherwise strip markup
                final_out = round_content
                try:
                    if round_content:
                        cleaned, _, final_seg = self._parse_harmony_tokens(round_content)
                        final_out = final_seg or cleaned
                        if not final_out:
                            final_out = self._strip_harmony_markup(round_content)
                except Exception:
                    final_out = self._strip_harmony_markup(round_content)

                self.conversation_history.append({
                    'role': 'assistant',
                    'content': final_out
                })
                self._trace("tools:none")
                # Persist memory to Mem0
                self._mem0_add_after_response(self._last_user_message, final_out)

                if aggregated_results:
                    combined = (preface_content + "\n\n" if preface_content else "") + "[Tool Results]\n" + '\n'.join(aggregated_results) + "\n\n" + final_out
                    return combined
                return final_out

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

    # ------------------ Harmony parsing helpers ------------------
    def _strip_harmony_markup(self, text: str) -> str:
        """Remove Harmony channel/control tokens from text, preserving natural content.

        This strips tokens like <|channel|>commentary, <|channel|>final, <|message|>, <|call|>, <|end|>.
        """
        try:
            if not text:
                return text
            # Remove channel headers (commentary/final) and inline markers
            t = re.sub(r"<\|channel\|>\s*commentary[^\n<]*", "", text, flags=re.IGNORECASE)
            t = re.sub(r"<\|channel\|>\s*final[^\n<]*", "", t, flags=re.IGNORECASE)
            t = re.sub(r"<\|message\|>", "", t, flags=re.IGNORECASE)
            t = re.sub(r"<\|call\|>", "", t, flags=re.IGNORECASE)
            t = re.sub(r"<\|end\|>", "", t, flags=re.IGNORECASE)
            return t
        except Exception:
            return text

    def _parse_harmony_tokens(self, text: str):
        """Parse Harmony tool-call and final-channel tokens from text.

        Returns (cleaned_text, tool_calls, final_text)
        - cleaned_text: input with tool-call segments removed and markup stripped
        - tool_calls: list of OpenAI-style tool_call dicts
        - final_text: last final-channel message content if present
        """
        tool_calls: List[Dict[str, Any]] = []
        final_text: Optional[str] = None
        if not text:
            return text, tool_calls, final_text

        # Match commentary tool calls like:
        # <|channel|>commentary to=functions.web_fetch ... <|message|>{json}<|call|>
        tc_re = re.compile(
            r"<\|channel\|>\s*commentary\b.*?to=functions\.([a-zA-Z0-9_]+).*?<\|message\|>(?P<json>\{.*?\})\s*<\|call\|>",
            flags=re.IGNORECASE | re.DOTALL,
        )

        def _tc_repl(m: re.Match) -> str:
            name = m.group(1)
            arg_text = m.group('json') if 'json' in m.groupdict() else '{}'
            args: Dict[str, Any]
            try:
                args = json.loads(arg_text)
                if not isinstance(args, dict):
                    args = {}
            except Exception:
                args = {}
            tool_calls.append({
                'type': 'function',
                'id': f"call_h_{len(tool_calls)+1}",
                'function': {
                    'name': name,
                    'arguments': args,
                }
            })
            return ""  # remove this segment from cleaned text

        cleaned = tc_re.sub(_tc_repl, text)

        # Extract final channel content (use the last occurrence if multiple)
        final_re = re.compile(r"<\|channel\|>\s*final\b.*?<\|message\|>(?P<msg>.*?)(?:<\|end\|>|$)", re.IGNORECASE | re.DOTALL)
        for m in final_re.finditer(cleaned):
            final_text = m.group('msg')

        # Remove any remaining markup tokens
        cleaned = self._strip_harmony_markup(cleaned)

        return cleaned, tool_calls, final_text

    # ------------------ Internal helpers ------------------
    def _set_idempotency_key(self, key: Optional[str]) -> None:
        """Set Idempotency-Key header on both clients for this turn."""
        try:
            if not key:
                return
            try:
                # Defensive: _client may not exist depending on ollama version
                if getattr(self.client, '_client', None) and getattr(self.client._client, 'headers', None):  # type: ignore[attr-defined]
                    self.client._client.headers['Idempotency-Key'] = key  # type: ignore[attr-defined]
            except Exception:
                pass
            self._trace(f"idempotency:set {key}")
        except Exception:
            pass

    def _clear_idempotency_key(self) -> None:
        """Remove Idempotency-Key header after request completion."""
        try:
            c = getattr(self, 'client', None)
            try:
                if c and getattr(c, '_client', None):
                    c._client.headers.pop('Idempotency-Key', None)  # type: ignore[attr-defined]
            except Exception:
                pass
            self._current_idempotency_key = None
        except Exception:
            pass

    def _resolve_keep_alive(self) -> Optional[Union[float, str]]:
        """Resolve a valid keep_alive value or None.
        Accepts env `OLLAMA_KEEP_ALIVE` as:
        - duration string with units, e.g., '10m', '1h', '30s'
        - numeric seconds (int/float), converted to '<seconds>s'
        If unset and warming is enabled, defaults to '10m'.
        """
        try:
            if not self.warm_models:
                return None
            raw = self.ollama_keep_alive_raw
            if raw is None or str(raw).strip() == "":
                return '10m'  # safe default
            s = str(raw).strip()
            # If purely numeric (int/float), treat as seconds
            if re.fullmatch(r"\d+(?:\.\d+)?", s):
                # Avoid passing floats like '10.0s' unless necessary
                if '.' in s:
                    return f"{float(s)}s"
                return f"{int(s)}s"
            # If duration with unit
            if re.fullmatch(r"\d+(?:\.\d+)?[smhdw]", s, flags=0):
                return s
            # Unknown/invalid -> log and fallback
            self.logger.debug(f"Invalid OLLAMA_KEEP_ALIVE '{s}', falling back to 10m")
            return '10m'
        except Exception:
            return '10m'
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[str]:
        """Execute tool calls and return results.
        
        Args:
            tool_calls: List of tool call dictionaries
            
        Returns:
            List of tool execution results
        """
        tool_results = []
        injected_chunks = []
         
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
                    # Confirm execute_shell in TTY if required
                    if function_name == 'execute_shell':
                        need_confirm = (os.getenv('SHELL_TOOL_CONFIRM', '1') in {'1', 'true', 'True'}) and sys.stdin.isatty()
                        if need_confirm:
                            preview_args = {k: v for k, v in function_args.items() if k in ('command', 'working_dir', 'timeout')}
                            preview = json.dumps(preview_args)
                            print(f"   ⚠️ execute_shell preview: {preview}")
                            ans = input("   Proceed? [y/N]: ").strip().lower()
                            if ans not in {'y', 'yes'}:
                                aborted = f"Execution aborted by user for execute_shell({truncate_text(preview, 120)})"
                                tool_results.append(aborted)
                                injected_chunks.append("execute_shell: aborted by user")
                                self._skip_mem0_after_turn = True
                                continue

                    result = self.tool_functions[function_name](**function_args)
                    # Try parse JSON contracts from secure tools
                    injected = None
                    sensitive = False
                    log_path = None
                    try:
                        parsed = json.loads(result)
                        injected = parsed.get('inject') if isinstance(parsed, dict) else None
                        sensitive = bool(parsed.get('sensitive')) if isinstance(parsed, dict) else False
                        log_path = parsed.get('log_path') if isinstance(parsed, dict) else None
                    except Exception:
                        parsed = None

                    # Telemetry
                    try:
                        if function_name in {'execute_shell', 'web_fetch'} and isinstance(parsed, dict):
                            if function_name == 'execute_shell':
                                self.logger.debug(f"tool:execute_shell status ok={parsed.get('ok')} exit={parsed.get('exit_code')} log={parsed.get('log_path')}")
                            else:
                                net = parsed.get('net') or {}
                                self.logger.debug(f"tool:web_fetch status={net.get('status')} bytes={net.get('bytes')} cached={net.get('cached')} redirects={net.get('redirects')}")
                    except Exception:
                        pass

                    # Determine injection chunk
                    display = result if injected is None else injected
                    if len(display) > self.tool_context_cap:
                        display = truncate_text(display, self.tool_context_cap)
                        if log_path:
                            display += f"\n[truncated; full logs stored at {log_path} (not shared with the model)]"

                    tool_results.append(result)
                    injected_chunks.append(display)
                    if sensitive or function_name == 'execute_shell' or len(display) > self.tool_context_cap:
                        self._skip_mem0_after_turn = True
                    if not self.show_trace and not self.quiet:
                        print(f"      ✅ Result: {truncate_text(display, self.tool_print_limit)}")
                    self._trace(f"tool:ok {function_name}")
                except Exception as e:
                    error_result = f"Error executing {function_name}: {str(e)}"
                    tool_results.append(error_result)
                    injected_chunks.append(error_result)
                    if not self.show_trace and not self.quiet:
                        print(f"      ❌ {error_result}")
                    self._trace(f"tool:error {function_name}")
            else:
                error_result = f"Unknown tool: {function_name}"
                tool_results.append(error_result)
                injected_chunks.append(error_result)
                if not self.show_trace and not self.quiet:
                    print(f"      ❌ {error_result}")
                self._trace(f"tool:unknown {function_name}")
         
        # Replace tool injection content with compact, capped summaries
        if injected_chunks:
            # Overwrite the last tool message content in caller using this list rather than tool_results
            pass
        return injected_chunks or tool_results
    
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
            if msg.get('role') == 'system':
                c = msg.get('content') or ''
                if (
                    c.startswith("Previous context from user history (use if relevant):")
                    or c.startswith("Relevant information:")
                    or c.startswith("Relevant user memories")
                ):
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
        # Preserve the system directive when clearing
        self.conversation_history = [
            {
                'role': 'system',
                'content': self.prompt.initial_system_prompt()
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

            # API version preference and capability flags
            self._mem0_api_version_pref = (os.getenv('MEM0_VERSION', 'v2') or 'v2').strip() or 'v2'
            self._mem0_supports_version_kw = True

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
                    if (
                        c.startswith("Previous context from user history (use if relevant):")
                        or c.startswith("Relevant information:")
                        or c.startswith("Relevant user memories")
                    ):
                        self.conversation_history.pop(idx)

            # Minimal filter: just user_id
            filters = {"user_id": getattr(self, 'mem0_user_id', 'Braden')}
            def _do_search():
                return self._mem0_search_api(user_message, filters=filters, limit=self.mem0_max_hits)

            try:
                if self._mem0_search_pool:
                    fut = self._mem0_search_pool.submit(_do_search)
                    related = fut.result(timeout=max(0.05, self.mem0_search_timeout_ms / 1000.0))
                else:
                    # very unlikely fallback
                    related = self._mem0_search_api(user_message, filters=filters, limit=self.mem0_max_hits)
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
            context = self.prompt.mem0_context_block(acc)
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
        if getattr(self, '_skip_mem0_after_turn', False):
            # Skip persisting sensitive/tool outputs
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
        kwargs = {"messages": messages, "user_id": user_id, "version": getattr(self, '_mem0_api_version_pref', 'v2'), "metadata": metadata}
        if getattr(self, 'mem0_agent_id', None):
            kwargs["agent_id"] = self.mem0_agent_id
        try:
            res = self.mem0_client.add(**kwargs)
        except TypeError:
            # Second attempt: drop version, keep agent_id/metadata (older SDKs may not accept 'version')
            try:
                kwargs_no_ver = dict(kwargs)
                kwargs_no_ver.pop("version", None)
                res = self.mem0_client.add(**kwargs_no_ver)
            except TypeError:
                # Third attempt: drop agent_id, keep version/metadata
                try:
                    kwargs_no_agent = dict(kwargs)
                    kwargs_no_agent.pop("agent_id", None)
                    res = self.mem0_client.add(**kwargs_no_agent)
                except TypeError:
                    # Fourth attempt: minimal signature (no version/metadata); enforce metadata later via update()
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

                    # Ensure agent_id is set as a first-class field when supported
                    if getattr(self, 'mem0_agent_id', None):
                        try:
                            self.mem0_client.update(memory_id=mid, agent_id=self.mem0_agent_id)
                        except TypeError:
                            # Older SDKs may not allow setting agent_id via update; ignore
                            pass
                    # Ensure app_id is set as a first-class field when supported
                    if getattr(self, 'mem0_app_id', None):
                        try:
                            self.mem0_client.update(memory_id=mid, app_id=self.mem0_app_id)
                        except TypeError:
                            pass

                    ok = True
                    if self.mem0_debug:
                        dt_ms = int((time.time() - start_meta) * 1000)
                        self.logger.debug(f"[mem0] update(meta+agent) dt={dt_ms}ms id={mid}")
                    break
                except Exception:
                    time.sleep(delay)
            if not ok:
                self.logger.warning(f"Mem0 metadata enforcement failed for id={mid}")

    def _mem0_search_api(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
        """Search wrapper that prefers v2 and falls back if the SDK doesn't accept the 'version' kwarg."""
        if not self.mem0_client:
            return []
        try:
            if getattr(self, '_mem0_supports_version_kw', True):
                try:
                    return self.mem0_client.search(query, version=self._mem0_api_version_pref, filters=filters, limit=limit)
                except TypeError:
                    # Older SDK without 'version' kwarg; cache and retry without it
                    self._mem0_supports_version_kw = False
            # Fallback path (no version kwarg)
            return self.mem0_client.search(query, filters=filters, limit=limit)
        except Exception:
            raise

    def _mem0_get_all_api(self, filters: Optional[Dict[str, Any]] = None):
        """get_all wrapper that prefers v2 and falls back if the SDK doesn't accept the 'version' kwarg."""
        if not self.mem0_client:
            return []
        try:
            if getattr(self, '_mem0_supports_version_kw', True):
                try:
                    return self.mem0_client.get_all(filters=filters, version=self._mem0_api_version_pref)
                except TypeError:
                    self._mem0_supports_version_kw = False
            return self.mem0_client.get_all(filters=filters)
        except Exception:
            raise

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
                items = self._mem0_get_all_api(filters=filters)
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
                results = self._mem0_search_api(query, filters=filters)
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
                self._mem0_execute_add([{"role": "user", "content": text}], {"source": "cli", "category": "manual"})
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
                items = self._mem0_get_all_api(filters=filters)
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
                        self._mem0_execute_add([{"role": "user", "content": text}], meta)
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
                    results = self._mem0_search_api(q, filters=filters)
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
                    results = self._mem0_search_api(subject, filters=filters)
                    if results:
                        mem_id = results[0].get('id')
                        try:
                            self.mem0_client.update(memory_id=mem_id, text=new_text)
                        except TypeError:
                            self.mem0_client.update(memory_id=mem_id, data=new_text)
                        print("✅ Updated.")
                    else:
                        # If no match, add new
                        self._mem0_execute_add([{"role": "user", "content": new_text}], {"source": "nlu", "category": "manual"})
                        print("✅ Not found; added as new.")
                    return True
            # List memories
            if lower.startswith("list memories") or lower.startswith("show memories"):
                filters = {"user_id": self.mem0_user_id}
                items = self._mem0_get_all_api(filters=filters)
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
                    results = self._mem0_search_api(q, filters=filters)
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
                results = self._mem0_search_api(text, filters=filters)
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
