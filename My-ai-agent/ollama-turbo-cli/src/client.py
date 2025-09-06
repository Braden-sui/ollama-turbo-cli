"""
Ollama Turbo client implementation with tool calling and streaming support.
Reliability hardening: retries/backoff, idempotency keys, keep-alive pools, and
streaming idle reconnects (all behind env flags with safe defaults).
"""

import json
import sys
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
from ollama import Client
import threading
import queue
import time
import atexit
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from . import plugin_loader as _plugin_loader
from .utils import with_retry, RetryableError, OllamaAPIError, truncate_text, format_conversation_history
from .prompt_manager import PromptManager
from .harmony_processor import HarmonyProcessor
from .reliability.retrieval.pipeline import RetrievalPipeline
from .reliability.grounding.context_builder import ContextBuilder
from .protocols import get_adapter
from .transport import networking as _net
from .streaming import runner as _runner, standard as _standard
from .tools_runtime.executor import ToolRuntimeExecutor


class OllamaTurboClient:
    """Client for interacting with gpt-oss:120b via Ollama Turbo."""
    
    def __init__(self, api_key: str, model: str = "gpt-oss:120b", enable_tools: bool = True, show_trace: bool = False, reasoning: str = "high", quiet: bool = False, max_output_tokens: Optional[int] = None, ctx_size: Optional[int] = None, tool_print_limit: int = 200, multi_round_tools: bool = True, tool_max_rounds: Optional[int] = None, *, ground: bool = False, k: Optional[int] = None, cite: bool = False, check: str = 'off', consensus: bool = False, engine: Optional[str] = None, eval_corpus: Optional[str] = None, reasoning_mode: str = 'system', protocol: str = 'auto', temperature: Optional[float] = None, top_p: Optional[float] = None, presence_penalty: Optional[float] = None, frequency_penalty: Optional[float] = None, 
                 # Mem0 configuration
                 mem0_enabled: bool = True,
                 mem0_local: bool = False,
                 mem0_vector_provider: str = 'chroma',
                 mem0_vector_host: str = ':memory:',
                 mem0_vector_port: int = 0,
                 mem0_ollama_url: Optional[str] = None,
                 mem0_llm_model: Optional[str] = None,
                 mem0_embedder_model: str = 'nomic-embed-text',
                 mem0_user_id: str = 'cli-user'):
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
        # How to send reasoning effort to provider: 'system' | 'request:top' | 'request:options'
        rm = str(reasoning_mode or 'system').strip().lower()
        self.reasoning_mode = rm if rm in {'system', 'request:top', 'request:options'} else 'system'
        self.trace: List[str] = []
        self.logger = logging.getLogger(__name__)
        self.max_output_tokens = max_output_tokens
        self.ctx_size = ctx_size
        # Sampling parameters (may be None; adapter- or model-specific defaults can be applied later)
        self.temperature: Optional[float] = temperature
        self.top_p: Optional[float] = top_p
        self.presence_penalty: Optional[float] = presence_penalty
        self.frequency_penalty: Optional[float] = frequency_penalty
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
        # Mem0 search timeout unified here (ms)
        try:
            self.mem0_search_timeout_ms: int = int(os.getenv('MEM0_SEARCH_TIMEOUT_MS', '500') or '500')
        except Exception:
            self.mem0_search_timeout_ms = 500
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
        # Tool results return format (for future API use). Default preserves v1 behavior (strings)
        trf = (os.getenv('TOOL_RESULTS_FORMAT') or 'string').strip().lower()
        self.tool_results_format: str = 'object' if trf == 'object' else 'string'
        # Reliability mode configuration (no-op placeholders until wired)
        self.engine: Optional[str] = engine
        self.reliability = {
            'ground': bool(ground),
            'k': k,
            'cite': bool(cite),
            'check': check if check in {'off', 'warn', 'enforce'} else 'off',
            'consensus': bool(consensus),
            'eval_corpus': eval_corpus,
        }
        # Split retrieval vs consensus k to avoid coupling
        try:
            rag_k_env = os.getenv('RAG_TOPK', '5')
            cons_k_env = os.getenv('CONSENSUS_K', '')
            rag_k_val = int(self.reliability.pop('k', None) or (rag_k_env if rag_k_env.isdigit() else 5))
        except Exception:
            rag_k_val = 5
        try:
            consensus_k_val = int(cons_k_env) if cons_k_env.isdigit() else None
        except Exception:
            consensus_k_val = None
        self.reliability.update({'rag_k': rag_k_val, 'consensus_k': consensus_k_val})
        # Reliability runtime state
        self._last_context_blocks: List[Dict[str, Any]] = []
        self._last_citations_map: Dict[str, Any] = {}
        self._system_cited_cache: Optional[str] = None

        # Initialize Ollama client with authentication
        # Note: Ollama Turbo uses Authorization header without 'Bearer' prefix
        resolved_host = self._resolve_host(self.engine)
        self.host = resolved_host
        self.client = Client(
            host=resolved_host,
            headers={'Authorization': api_key}
        )
        
        # Prompt management
        self.prompt = PromptManager(self.reasoning)
        # Harmony parsing/markup processing
        self.harmony = HarmonyProcessor()
        # Protocol adapter selection (default: auto -> harmony unless detected otherwise)
        try:
            self.protocol = str(protocol or os.getenv('OLLAMA_PROTOCOL') or 'auto').strip().lower()
        except Exception:
            self.protocol = 'auto'
        self.adapter = get_adapter(model=self.model, protocol=self.protocol)
        # Apply DeepSeek-specific defaults and minimal system prompt
        try:
            adapter_name = getattr(self.adapter, 'name', '')
        except Exception:
            adapter_name = ''
        # Resolve DeepSeek defaults only if not provided explicitly
        if adapter_name == 'deepseek':
            def _env_float(name: str, default: float) -> float:
                try:
                    v = os.getenv(name)
                    return float(v) if v is not None and str(v).strip() != '' else default
                except Exception:
                    return default
            if self.temperature is None:
                self.temperature = _env_float('DEEPSEEK_TEMP', 0.6)
            if self.top_p is None:
                self.top_p = _env_float('DEEPSEEK_TOP_P', 0.95)
            if self.presence_penalty is None:
                self.presence_penalty = _env_float('DEEPSEEK_PRESENCE_PENALTY', 0.0)
            if self.frequency_penalty is None:
                self.frequency_penalty = _env_float('DEEPSEEK_FREQUENCY_PENALTY', 0.2)
            sys_prompt = self.prompt.deepseek_system_prompt()
        else:
            sys_prompt = self.prompt.initial_system_prompt()
        # Initialize conversation history with a system directive
        self.conversation_history = [
            {
                'role': 'system',
                'content': sys_prompt
            }
        ]
        # Enforce local history window <= 10 turns (excluding initial system)
        try:
            raw_hist = os.getenv('MAX_CONVERSATION_HISTORY', '10')
            parsed_hist = int(raw_hist) if str(raw_hist).isdigit() else 10
        except Exception:
            parsed_hist = 10
        self.max_history = max(2, min(parsed_hist, 10))
        
        # Set up tools if enabled (use copies to avoid global mutation leaks).
        # Access plugin aggregates lazily to avoid import-time plugin loading.
        if enable_tools:
            schemas = _plugin_loader.TOOL_SCHEMAS  # triggers load only now
            funcs = _plugin_loader.TOOL_FUNCTIONS
            self.tools = list(schemas)
            self.tool_functions = dict(funcs)
        else:
            self.tools = []
            self.tool_functions = {}
        
        # Mem0 configuration
        self.mem0_config = {
            'enabled': mem0_enabled,
            'local': mem0_local,
            'vector_provider': mem0_vector_provider,
            'vector_host': mem0_vector_host,
            'vector_port': mem0_vector_port,
            'ollama_url': mem0_ollama_url or self.host,  # Default to main client's host
            'llm_model': mem0_llm_model or model,  # Default to main model
            'embedder_model': mem0_embedder_model,
            'user_id': mem0_user_id,
        }
        
        # Initialize Mem0 memory system (optional)
        self._init_mem0()

        self.logger.info(f"Initialized client with model: {model}, host: {self.host}, tools enabled: {enable_tools}, reasoning={self.reasoning}, mode={self.reasoning_mode}, quiet={self.quiet}")
        # Initial trace state
        if self.show_trace:
            self.trace.append(f"client:init model={model} host={self.host} tools={'on' if enable_tools else 'off'} reasoning={self.reasoning} mode={self.reasoning_mode} quiet={'on' if self.quiet else 'off'}")

    # ---------- Reasoning Injection Helpers ----------
    def _nested_set(self, d: Dict[str, Any], path: str, value: Any) -> None:
        try:
            parts = [p for p in str(path).split('.') if p]
            cur = d
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = value
        except Exception:
            # Do not fail request due to optional reasoning injection
            pass

    def _maybe_inject_reasoning(self, kwargs: Dict[str, Any]) -> None:
        """Optionally inject request-level reasoning effort based on configuration.

        Controlled by self.reasoning_mode. Field name and style are env-configurable:
        - REASONING_FIELD_PATH: dot-path for target (default depends on mode)
        - REASONING_FIELD_STYLE: 'string' (default) or 'object'
        - REASONING_OBJECT_KEY: key name when style is 'object' (default: 'effort')
        """
        try:
            if self.reasoning_mode == 'system':
                return
            # Resolve defaults by mode
            default_path = 'options.reasoning_effort' if self.reasoning_mode == 'request:options' else 'reasoning'
            field_path = (os.getenv('REASONING_FIELD_PATH') or default_path).strip()
            style = (os.getenv('REASONING_FIELD_STYLE') or 'string').strip().lower()
            obj_key = (os.getenv('REASONING_OBJECT_KEY') or 'effort').strip()

            if style == 'object':
                value: Any = {obj_key: self.reasoning}
            else:
                value = self.reasoning

            self._nested_set(kwargs, field_path, value)
            self._trace(f"reasoning:inject path={field_path} style={style} val={self.reasoning}")
        except Exception:
            # Never raise on optional reasoning injection
            pass

    def _prepare_initial_messages_for_adapter(self, *, include_tools: bool, adapter_opts: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Prepare initial-turn messages via adapter, merging Mem0 into first system if needed.

        - If Mem0 is already merged into the first system message, pass through.
        - If Mem0 exists as a separate system block, strip it from the outgoing
          message list and pass it to the adapter via mem0_block for safe merge.
        - Returns (normalized_messages, payload_overrides) where overrides may include
          provider-specific options and tools.
        """
        try:
            msgs: List[Dict[str, Any]] = list(self.conversation_history or [])
            # Determine Mem0 prefixes
            try:
                prefixes = self.prompt.mem0_prefixes()
            except Exception:
                prefixes = [
                    "Previous context from user history (use if relevant):",
                    "Relevant information:",
                    "Relevant user memories",
                ]
            # Check if first system already contains Mem0
            mem0_in_first = False
            if msgs and (msgs[0] or {}).get('role') == 'system':
                first_c = str((msgs[0] or {}).get('content') or '')
                mem0_in_first = any((p and (p in first_c)) for p in prefixes)

            mem0_block: Optional[str] = None
            # Respect env flag: only merge into first system when explicitly enabled
            prefer_merge = str(os.getenv('MEM0_IN_FIRST_SYSTEM', '0')).strip().lower() in {'1', 'true', 'yes', 'on'}
            if prefer_merge and (not mem0_in_first):
                # Find the latest Mem0 system block and extract it
                latest_idx: Optional[int] = None
                for i in range(len(msgs) - 1, -1, -1):
                    m = msgs[i]
                    if m.get('role') == 'system':
                        c = str(m.get('content') or '')
                        if any(c.startswith(p) for p in prefixes):
                            latest_idx = i
                            break
                if latest_idx is not None:
                    mem0_block = str((msgs[latest_idx] or {}).get('content') or '')
                    # Remove that block so the adapter can merge into first system
                    msgs = msgs[:latest_idx] + msgs[latest_idx + 1:]

            # Delegate to adapter for initial formatting
            norm_msgs, overrides = self.adapter.format_initial_messages(
                messages=msgs,
                tools=(self.tools if (include_tools and bool(self.tools)) else None),
                options=(adapter_opts or None),
                mem0_block=mem0_block,
            )
            return norm_msgs, (overrides or {})
        except Exception:
            # Fallback: pass-through and attempt minimal option/tool mapping
            out_msgs = list(self.conversation_history or [])
            overrides: Dict[str, Any] = {}
            try:
                mapped = self.adapter.map_options(adapter_opts) if adapter_opts else {}
                if mapped:
                    overrides['options'] = mapped
            except Exception:
                # Secondary fallback to Ollama-compatible fields
                opts: Dict[str, Any] = {}
                if self.max_output_tokens is not None:
                    opts['num_predict'] = self.max_output_tokens
                if self.ctx_size is not None:
                    opts['num_ctx'] = self.ctx_size
                if opts:
                    overrides['options'] = opts
            if include_tools and self.tools:
                overrides['tools'] = self.tools
            return out_msgs, overrides

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
    
    def _trace_mem0_presence(self, messages: Optional[List[Dict[str, Any]]], where: str) -> None:
        """Record whether Mem0 context is present in the first system message or as a separate system block.

        The trace is non-sensitive and only records booleans/counts. Used to verify that providers that
        only honor the first system message still receive Mem0 context when enabled.
        """
        if not self.show_trace:
            return
        try:
            msgs = messages or []
            prefixes: List[str] = []
            try:
                prefixes = self.prompt.mem0_prefixes()
            except Exception:
                prefixes = [
                    "Previous context from user history (use if relevant):",
                    "Relevant information:",
                    "Relevant user memories",
                ]
            in_first = False
            if msgs and (msgs[0] or {}).get('role') == 'system':
                first_c = str((msgs[0] or {}).get('content') or '')
                in_first = any((p and (p in first_c)) for p in prefixes)
            blocks = 0
            for m in msgs:
                if m.get('role') == 'system':
                    c = str(m.get('content') or '')
                    if any((p and (p in c)) for p in prefixes):
                        blocks += 1
            self._trace(f"mem0:present:{where} first={'1' if in_first else '0'} blocks={blocks}")
        except Exception:
            # Never fail request due to tracing
            pass
    
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
        # Reliability: clear per-request state to avoid cross-call bleed
        self._last_context_blocks = []
        self._last_citations_map = {}
        # Reliability: optional retrieval/grounding/citation system additions
        if self.reliability.get('ground'):
            try:
                self._prepare_reliability_context(message)
            except Exception as e:
                self.logger.debug(f"reliability:context skipped: {e}")
        
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
    
    def _handle_standard_chat(self, *, _suppress_errors: bool = False) -> str:
        """Handle non-streaming chat interaction (delegated) with dynamic retries."""
        if not self.cli_retry_enabled:
            return _standard.handle_standard_chat(self, _suppress_errors=_suppress_errors)
        @with_retry(max_retries=self.cli_max_retries)
        def _call():
            return _standard.handle_standard_chat(self, _suppress_errors=_suppress_errors)
        return _call()
    
    def _handle_streaming_chat(self) -> str:
        """Handle streaming chat interaction with tool support."""
        try:
            init_stream = self._create_streaming_response()
        except Exception as e:
            # Silent fallback: do not surface errors to CLI output
            self.logger.debug(f"Streaming init failed; falling back to non-streaming: {e}")
            self._trace("stream:init:error -> fallback")
            try:
                final = self._handle_standard_chat(_suppress_errors=True)
                # Print final response so the user sees output even when streaming init fails
                if final and not str(final).startswith("Error during chat:") and not self.quiet:
                    print(final)
                self._trace("stream:init:fallback:success")
                return final
            except Exception as e2:
                # Still suppress to avoid leaking error text in streaming mode
                self.logger.debug(f"Non-streaming fallback also failed: {e2}")
                self._trace("stream:init:fallback:error")
                return ""
        return self.handle_streaming_response(
            init_stream,
            tools_enabled=self.enable_tools
        )
    
    def _create_streaming_response(self):
        """Create a streaming response from the API with dynamic retries."""
        if not self.cli_retry_enabled:
            return _runner.create_streaming_response(self)
        @with_retry(max_retries=self.cli_max_retries)
        def _call():
            return _runner.create_streaming_response(self)
        return _call()
    
    def handle_streaming_response(self, response_stream, tools_enabled: bool = True) -> str:
        """Complete streaming response handler with tool call support (delegated)."""
        return _runner.handle_streaming_response(self, response_stream, tools_enabled=tools_enabled)

    # ------------------ Harmony parsing helpers ------------------
    def _strip_harmony_markup(self, text: str) -> str:
        """Remove Harmony channel/control tokens from text, preserving natural content.

        This strips tokens like <|channel|>commentary, <|channel|>final, <|message|>, <|call|>, <|end|>.
        """
        try:
            return self.harmony.strip_markup(text)
        except Exception:
            return text

    def _parse_harmony_tokens(self, text: str):
        """Parse Harmony tool-call and final-channel tokens from text.

        Returns (cleaned_text, tool_calls, final_text)
        - cleaned_text: input with tool-call segments removed and markup stripped
        - tool_calls: list of OpenAI-style tool_call dicts
        - final_text: last final-channel message content if present
        """
        return self.harmony.parse_tokens(text)

    # ------------------ Internal helpers ------------------
    def _set_idempotency_key(self, key: Optional[str]) -> None:
        """Set Idempotency-Key header on both clients for this turn."""
        try:
            _net.set_idempotency_key(self.client, key, trace_hook=self._trace)
        except Exception:
            # Never fail request due to optional header injection
            pass

    def _clear_idempotency_key(self) -> None:
        """Remove Idempotency-Key header after request completion."""
        try:
            _net.clear_idempotency_key(getattr(self, 'client', None))
            self._current_idempotency_key = None
        except Exception:
            # Never fail cleanup
            pass

    def _resolve_host(self, engine: Optional[str]) -> str:
        """Resolve the Ollama host to use based on engine flag or env.

        Priority:
        1) Explicit --engine flag
           - 'cloud' -> https://ollama.com
           - 'local' -> http://localhost:11434
           - Full URL (http/https) -> use as-is
           - Bare hostname -> prefix with https://
        2) OLLAMA_HOST env var
        3) Default https://ollama.com
        """
        return _net.resolve_host(engine)

    def _resolve_keep_alive(self) -> Optional[Union[float, str]]:
        """Resolve a valid keep_alive value or None.
        Accepts env `OLLAMA_KEEP_ALIVE` as:
        - duration string with units, e.g., '10m', '1h', '30s'
        - numeric seconds (int/float), converted to '<seconds>s'
        If unset and warming is enabled, defaults to '10m'.
        """
        try:
            return _net.resolve_keep_alive(
                warm_models=bool(getattr(self, 'warm_models', True)),
                host=getattr(self, 'host', None),
                keep_alive_raw=getattr(self, 'ollama_keep_alive_raw', None),
                logger=getattr(self, 'logger', None),
            )
        except Exception:
            # match docstring + avoid cold starts when warming is on
            return '10m' if bool(getattr(self, 'warm_models', True)) else None
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results (delegated to ToolRuntimeExecutor)."""
        return ToolRuntimeExecutor.execute(self, tool_calls)
    
    def _serialize_tool_result_to_string(self, tr: Dict[str, Any]) -> str:
        """Serialize a structured tool result to a safe string (delegated)."""
        return ToolRuntimeExecutor.serialize_to_string(self, tr)
    
    def _payload_for_tools(self, tool_results: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]]):
        """
        Returns a tuple (payload_for_adapter, prebuilt_tool_messages_or_None) based on self.tool_results_format.
        - If 'object': (tool_results, None) — adapters receive list[dict].
        - If 'string': (tool_strings, prebuilt_msgs) — adapters receive list[str]; fallback always uses strings.
          prebuilt_msgs is a list of {'role': 'tool', 'tool_call_id': <matching id if available>, 'content': <string>}
          one per tool call/result (mapped by position when present).
        Also updates:
          - self._last_tool_results_structured
          - self._last_tool_results_strings
        """
        tool_strings = [self._serialize_tool_result_to_string(tr) for tr in tool_results]
        # Bookkeep both views for diagnostics/tests
        try:
            self._last_tool_results_structured = tool_results
            self._last_tool_results_strings = tool_strings
        except Exception:
            pass
        fmt = getattr(self, 'tool_results_format', 'string')
        if fmt == 'object':
            return tool_results, None
        # Build per-call tool messages with tool_call_id mapping by position
        prebuilt_msgs: List[Dict[str, Any]] = []
        for i, s in enumerate(tool_strings):
            tc_id = None
            try:
                if i < len(tool_calls):
                    tc = tool_calls[i] or {}
                    raw = tc.get('id') or ((tc.get('function') or {}).get('id'))
                    # Coerce to string for strict adapters; treat empty string as None
                    tc_id = (str(raw).strip() or None) if raw is not None else None
            except Exception:
                tc_id = None
            msg: Dict[str, Any] = {'role': 'tool', 'content': s}
            if tc_id:
                msg['tool_call_id'] = tc_id
            prebuilt_msgs.append(msg)
        return tool_strings, prebuilt_msgs
    
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
        # De-duplicate while preserving order. For Mem0 system blocks, dedupe by content.
        seen = set()
        deduped: List[Dict[str, Any]] = []
        # Prepare optional Mem0 prefixes
        mem0_prefixes: List[str] = []
        try:
            mem0_prefixes = self.prompt.mem0_prefixes()
        except Exception:
            mem0_prefixes = [
                "Previous context from user history (use if relevant):",
                "Relevant information:",
                "Relevant user memories",
            ]
        def _is_mem0_block(msg: Dict[str, Any]) -> bool:
            try:
                if msg.get('role') != 'system':
                    return False
                c = str(msg.get('content') or '')
                return any(c.startswith(p) for p in mem0_prefixes if p)
            except Exception:
                return False
        for m in new_hist:
            try:
                if _is_mem0_block(m):
                    c = str(m.get('content') or '')
                    key = ("mem0_block", hash(c))
                else:
                    key = ("id", id(m))
            except Exception:
                key = ("id", id(m))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(m)
        self.conversation_history = deduped
    
    def clear_history(self):
        """Clear conversation history."""
        # Preserve the system directive when clearing, respecting adapter-specific defaults
        try:
            adapter_name = getattr(self.adapter, 'name', '')
        except Exception:
            adapter_name = ''
        sys_prompt = self.prompt.deepseek_system_prompt() if adapter_name == 'deepseek' else self.prompt.initial_system_prompt()
        self.conversation_history = [
            {
                'role': 'system',
                'content': sys_prompt
            }
        ]
        self.logger.info("Conversation history cleared")
    
    def get_history(self) -> str:
        """Get formatted conversation history.
        
        Returns:
            Formatted conversation history string
        """
        return format_conversation_history(self.conversation_history)
    
    # ------------------ Reliability integration helpers ------------------
    def _load_system_cited(self) -> str:
        try:
            if getattr(self, '_system_cited_cache', None):
                return self._system_cited_cache or ""
            base = os.path.dirname(__file__)
            path = os.path.join(base, 'reliability', 'prompts', 'system_cited.md')
            text = ""
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception:
                text = "When citing sources, include inline citations like [1], [2] mapped to provided context."
            self._system_cited_cache = text
            return text
        except Exception:
            return ""

    def _prepare_reliability_context(self, user_message: str) -> None:
        """Optionally run retrieval + grounding and inject a system addition for this turn."""
        try:
            rp = RetrievalPipeline()
            try:
                topk = int(self.reliability.get('rag_k') or int(os.getenv('RAG_TOPK', '5') or '5'))
            except Exception:
                topk = 5
            docs = rp.run(user_message, k=topk)
            cb = ContextBuilder()
            try:
                max_tokens = int(os.getenv('RAG_MAX_TOKENS', '1200') or '1200')
            except Exception:
                max_tokens = 1200
            ctx = cb.build(self.conversation_history, docs, max_tokens=max_tokens)
            self._last_context_blocks = ctx.get('context_blocks') or []
            self._last_citations_map = ctx.get('citations_map') or {}
            system_add = ctx.get('system_prompt_addition') or ""
            if self.reliability.get('cite'):
                # Only add strict citation instructions if we actually have grounded context
                if self._last_context_blocks:
                    cited = self._load_system_cited()
                    if cited:
                        system_add = (system_add + "\n" + cited).strip() if system_add else cited
                else:
                    # Soft rule when no sources are provided
                    soft_rule = "If no sources are provided, avoid specific figures (numbers/dates) or mark uncertainty clearly."
                    system_add = (system_add + "\n" + soft_rule).strip() if system_add else soft_rule
            if system_add:
                self.conversation_history.append({'role': 'system', 'content': system_add})
            self._trace(f"reliability:context blocks={len(self._last_context_blocks)}")
        except Exception as e:
            self.logger.debug(f"reliability context error: {e}")
    
    # ------------------ Mem0 Integration ------------------
    def _init_mem0(self) -> None:
        """Initialize Mem0 client and runtime settings from configuration."""
        self.mem0_client = None
        self.mem0_enabled = False
        
        if not self.mem0_config.get('enabled', True):
            self.logger.debug("Mem0 is disabled via configuration")
            return
            
        try:
            # Lazy import to avoid hard crash if package isn't installed yet
            try:
                if self.mem0_config.get('local', False):
                    # OpenMemory MCP / OSS client
                    from mem0 import Memory as _Mem0Impl  # type: ignore
                else:
                    # Mem0 Platform client
                    from mem0 import MemoryClient as _Mem0Impl  # type: ignore
            except Exception as ie:
                self.logger.debug(f"Mem0 import skipped: {ie}")
                # One-time user-visible notice so users know why Mem0 is disabled
                try:
                    if not getattr(self, '_mem0_notice_shown', False) and not self.quiet:
                        mode = "local" if self.mem0_config.get('local', False) else "cloud"
                        if mode == "local":
                            print("Mem0 disabled: mem0 package not installed for local mode. Install with: pip install mem0 chromadb or qdrant-client; set MEM0_USE_LOCAL=1")
                        else:
                            print("Mem0 disabled: mem0 package not installed. For cloud, set MEM0_API_KEY or install mem0.")
                        self._mem0_notice_shown = True
                except Exception:
                    pass
                return

            # Runtime knobs with defaults
            self.mem0_debug = False
            self.mem0_max_hits = 3
            # mem0_search_timeout_ms is configured at init from MEM0_SEARCH_TIMEOUT_MS
            self.mem0_timeout_connect_ms = 1000
            self.mem0_timeout_read_ms = 2000
            self.mem0_add_queue_max = 256
            self._mem0_breaker_threshold = 3
            self._mem0_breaker_cooldown_ms = 60000
            self._mem0_shutdown_flush_ms = 3000

            # API version preference and capability flags
            self._mem0_api_version_pref = 'v2'
            self._mem0_supports_version_kw = True

            # Optional: use an internal proxy LLM (e.g., gpt-oss:20b) for reranking Mem0 search results
            try:
                self.mem0_proxy_model = os.getenv('MEM0_PROXY_MODEL') or None
                self.mem0_proxy_timeout_ms = int(str(os.getenv('MEM0_PROXY_TIMEOUT_MS', '1200')).strip())
                self._mem0_proxy_enabled = bool(self.mem0_proxy_model)
            except Exception:
                self.mem0_proxy_model = None
                self.mem0_proxy_timeout_ms = 1200
                self._mem0_proxy_enabled = False

            if self.mem0_config.get('local', False):
                # Build configuration from provided parameters
                cfg = {
                    "vector_store": {
                        "provider": self.mem0_config['vector_provider'],
                        "config": {
                            "host": self.mem0_config['vector_host'],
                            "port": self.mem0_config['vector_port'],
                        },
                    },
                    "llm": {
                        "provider": "ollama",
                        "config": {
                            "model": self.mem0_config['llm_model'],
                            "ollama_base_url": self.mem0_config['ollama_url']
                        },
                    },
                    "embedder": {
                        "provider": "ollama",
                        "config": {
                            "model": self.mem0_config['embedder_model'],
                            "ollama_base_url": self.mem0_config['ollama_url']
                        },
                    },
                    "version": self._mem0_api_version_pref,
                }
                
                # Only add port to config if it's non-zero
                if not self.mem0_config['vector_port']:
                    del cfg["vector_store"]["config"]["port"]
                    
                # Use in-memory storage if host is ':memory:'
                if self.mem0_config['vector_host'] == ':memory:':
                    cfg["vector_store"]["config"]["in_memory"] = True
                    
                # Set user ID for memory isolation
                if 'user_id' in self.mem0_config:
                    cfg["user_id"] = self.mem0_config['user_id']

                # Instantiate local Memory client
                try:
                    if hasattr(_Mem0Impl, 'from_config'):
                        self.mem0_client = _Mem0Impl.from_config(cfg)  # type: ignore[arg-type]
                    else:
                        self.mem0_client = _Mem0Impl()  # type: ignore[call-arg]
                    self.mem0_mode = 'local'
                    self.mem0_enabled = True
                    self.logger.info(f"Initialized local Mem0 with config: {json.dumps(cfg, indent=2, default=str)}")
                except Exception as e:
                    self.logger.error(f"Mem0 local initialization failed: {e}")
                    try:
                        if not getattr(self, '_mem0_notice_shown', False) and not self.quiet:
                            print(f"Mem0 local initialization failed: {e}")
                            self._mem0_notice_shown = True
                    except Exception:
                        pass
                    return
            else:
                # Remote Mem0 client (cloud)
                try:
                    api_key = os.getenv('MEM0_API_KEY')
                    if not api_key:
                        self.logger.warning("MEM0_API_KEY not set, disabling Mem0")
                        try:
                            if not getattr(self, '_mem0_notice_shown', False) and not self.quiet:
                                print("Mem0 disabled: MEM0_API_KEY not set. Export MEM0_API_KEY or use --mem0-local for OSS mode.")
                                self._mem0_notice_shown = True
                        except Exception:
                            pass
                        return
                    
                    # Check for org and project IDs for cloud version
                    org_id = os.getenv('MEM0_ORG_ID')
                    project_id = os.getenv('MEM0_PROJECT_ID')
                    
                    if org_id or project_id:
                        self.mem0_client = _Mem0Impl(api_key=api_key, org_id=org_id, project_id=project_id)  # type: ignore[call-arg]
                    else:
                        self.mem0_client = _Mem0Impl(api_key=api_key)  # type: ignore[call-arg]
                        
                    self.mem0_mode = 'cloud'
                    self.mem0_enabled = True
                    self.logger.info("Initialized Mem0 cloud client")
                except Exception as e:
                    self.logger.error(f"Mem0 cloud initialization failed: {e}")
                    try:
                        if not getattr(self, '_mem0_notice_shown', False) and not self.quiet:
                            print(f"Mem0 cloud initialization failed: {e}")
                            self._mem0_notice_shown = True
                    except Exception:
                        pass
                    return

            # Initialize background worker for async operations
            self._mem0_add_queue = queue.Queue(maxsize=max(1, self.mem0_add_queue_max))
            self._mem0_worker_stop = threading.Event()
            self._mem0_worker = threading.Thread(
                target=self._mem0_worker_loop,
                name="mem0-worker",
                daemon=True
            )
            self._mem0_worker.start()
            atexit.register(self._mem0_shutdown)
            
            # Initialize search thread pool
            self._mem0_search_pool = ThreadPoolExecutor(
                max_workers=max(1, self._mem0_search_workers),
                thread_name_prefix="mem0-search"
            )
            atexit.register(lambda: self._mem0_search_pool.shutdown(wait=False) if self._mem0_search_pool else None)
            
            # Set common fields
            self.mem0_user_id = os.getenv('MEM0_USER_ID', str(self.mem0_config.get('user_id', 'cli-user')))
            self.mem0_agent_id = os.getenv('MEM0_AGENT_ID')
            self.mem0_app_id = os.getenv('MEM0_APP_ID')
            
            # Log successful initialization
            mode_str = getattr(self, 'mem0_mode', 'unknown')
            self.logger.info(f"Mem0 initialized and enabled in {mode_str} mode")
            
        except Exception as e:
            self.logger.error(f"Mem0 initialization failed: {e}")
            self.mem0_client = None
            self.mem0_enabled = False
            try:
                if not getattr(self, '_mem0_notice_shown', False) and not self.quiet:
                    print(f"Mem0 initialization failed: {e}")
                    self._mem0_notice_shown = True
            except Exception:
                pass

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

            # If Mem0 was previously merged into the first system message, strip it
            try:
                if self.conversation_history and (self.conversation_history[0] or {}).get('role') == 'system':
                    first_c = str((self.conversation_history[0] or {}).get('content') or '')
                    # Find any known Mem0 prefix and truncate content at that point
                    prefixes = []
                    try:
                        prefixes = self.prompt.mem0_prefixes()
                    except Exception:
                        prefixes = ["Previous context from user history (use if relevant):", "Relevant information:", "Relevant user memories"]
                    cut = -1
                    for p in prefixes:
                        if not p:
                            continue
                        pos = first_c.find(p)
                        if pos != -1:
                            cut = pos if cut == -1 else min(cut, pos)
                    if cut != -1:
                        self.conversation_history[0]['content'] = first_c[:cut].rstrip()
            except Exception:
                # Non-fatal; proceed without stripping
                pass

            # Minimal filter: just user_id
            filters = {"user_id": getattr(self, 'mem0_user_id', str(self.mem0_config.get('user_id', 'cli-user')))}
            def _do_search():
                # If reranker is enabled, pull a larger candidate set first
                search_limit = self.mem0_max_hits
                try:
                    if getattr(self, '_mem0_proxy_enabled', False) and self.mem0_proxy_model:
                        # Default 10; overridable via MEM0_RERANK_SEARCH_LIMIT
                        search_limit = max(self.mem0_max_hits, int(str(os.getenv('MEM0_RERANK_SEARCH_LIMIT', '10')).strip() or '10'))
                except Exception:
                    pass
                return self._mem0_search_api(user_message, filters=filters, limit=search_limit)

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
            texts: List[str] = []
            aug_texts: List[str] = []
            for m in related or []:
                try:
                    txt = _memtxt(m)
                    if txt:
                        s_txt = str(txt)
                        texts.append(s_txt)
                        # Build a compact metadata suffix purely for reranker signal; not injected to user
                        try:
                            md = (m.get('metadata') if isinstance(m, dict) else None) or {}
                            ts = str(md.get('timestamp') or md.get('ts') or '')
                            src = str(md.get('source') or md.get('app_id') or md.get('agent_id') or '')
                            cat = str(md.get('category') or '')
                            conf = str(md.get('confidence') or '')
                            meta_suffix = f"\n[meta: ts={ts} src={src} cat={cat} conf={conf}]".rstrip()
                        except Exception:
                            meta_suffix = ''
                        aug_texts.append((s_txt + meta_suffix) if meta_suffix else s_txt)
                except Exception:
                    continue
            # Optionally rerank with proxy model if configured
            top_texts = texts[: max(1, self.mem0_max_hits)]
            try:
                if getattr(self, '_mem0_proxy_enabled', False) and self.mem0_proxy_model and texts:
                    order = self._mem0_rerank_with_proxy(user_message, aug_texts or texts, model=self.mem0_proxy_model, k=self.mem0_max_hits)
                    if order:
                        # Map back to original, un-augmented texts
                        top_texts = [texts[i] for i in order if 0 <= int(i) < len(texts)]
                        # Safety cap to K
                        top_texts = top_texts[: max(1, self.mem0_max_hits)]
            except Exception as _re:
                # Non-fatal: fall back to original order on any rerank error
                pass
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
            # Merge into first system message if enabled via env
            in_first = str(os.getenv('MEM0_IN_FIRST_SYSTEM', '0')).strip().lower() in {"1", "true", "yes", "on"}
            if in_first and self.conversation_history and (self.conversation_history[0] or {}).get('role') == 'system':
                try:
                    base = str((self.conversation_history[0] or {}).get('content') or '').rstrip()
                    new_content = (base + "\n\n" + context).strip()
                    self.conversation_history[0]['content'] = new_content
                    self._trace(f"mem0:inject:first {len(acc)}")
                except Exception:
                    # Fallback to separate system message on any error
                    self.conversation_history.append({'role': 'system', 'content': context})
                    self._trace(f"mem0:inject {len(acc)}")
            else:
                # Default: inject as a separate system message before the user turn
                self.conversation_history.append({'role': 'system', 'content': context})
                self._trace(f"mem0:inject {len(acc)}")
            # Success -> reset breaker
            if self._mem0_fail_count >= self._mem0_breaker_threshold and not self._mem0_breaker_recovered_logged:
                self.logger.info("Mem0 recovered; resuming calls")
                self._mem0_breaker_recovered_logged = True
                self._mem0_breaker_tripped_logged = False
            # Reset failure counters and emit debug timing
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

    def _mem0_llm_generate(self, *, model: str, system: str, user: str) -> str:
        """Internal low-level generation that bypasses our higher-level chat to avoid recursion.

        Always sends a Harmony-compliant system message so gpt-oss operates properly.
        """
        try:
            # Build a Harmony-compliant system prompt. Start from the standard one and
            # append a brief reranker instruction that constrains output to JSON in <|final|>.
            try:
                base_sys = self.prompt.initial_system_prompt()
            except Exception:
                base_sys = (
                    "You are GPT-OSS running with Harmony channels.\n\n"
                    "— Harmony I/O Protocol —\n"
                    "• Always end with: <|channel|>final then <|message|>...<|end|>\n"
                )
            rerank_sys = (
                "\nFor this task, return only a JSON array of 0-based indices for the most relevant items, "
                "inside the Harmony final channel exactly as:\n"
                "<|channel|>final\n"
                "<|message|>[1,0]\n"
                "<|end|>\n"
                "No other channels or text."
            )
            msgs = [
                {'role': 'system', 'content': (base_sys + rerank_sys)},
                {'role': 'user', 'content': user},
            ]
            kwargs: Dict[str, Any] = {
                'model': model,
                'messages': msgs,
            }
            # Deterministic, short, and cheap
            options: Dict[str, Any] = {'temperature': 0, 'top_p': 0}
            if options:
                kwargs['options'] = options
            keep_val = self._resolve_keep_alive()
            if keep_val is not None:
                kwargs['keep_alive'] = keep_val
            # Do not inject reasoning or tools; call provider directly
            self._trace('mem0:proxy:call')
            # Apply a coarse timeout by using underlying retry wrapper at a higher level if present
            resp = self.client.chat(**kwargs)
            msg = resp.get('message', {}) if isinstance(resp, dict) else {}
            content = msg.get('content') or ''
            try:
                cleaned, _, final = self._parse_harmony_tokens(content)
                return (final or cleaned or self._strip_harmony_markup(content)) or ''
            except Exception:
                return self._strip_harmony_markup(content)
        except Exception as e:
            self.logger.debug(f"mem0 proxy generate failed: {e}")
            return ''

    def _mem0_rerank_with_proxy(self, query: str, candidates: List[str], *, model: str, k: Optional[int] = None) -> List[int]:
        """Rerank candidate memory snippets for the current query using a proxy LLM.

        Returns an ordered list of candidate indices (subset of range(N)). Empty list on failure.
        """
        try:
            N = len(candidates)
            if N == 0:
                return []
            K = max(1, int(k) if k is not None else int(self.mem0_max_hits))

            # Build user instructions with stronger criteria
            items = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
            usr_prompt = (
                "Rerank the candidates for the query. Criteria (in order):\n"
                "1) Semantic relevance to the query intent.\n"
                "2) Specificity and user-identifying detail over generic content.\n"
                "3) Recency if a timestamp is present in [meta: ts=...].\n"
                "4) De-duplicate near-identical content; pick the best representative.\n"
                "5) If candidates contradict, prefer the more specific or recent one.\n\n"
                f"Query: {query}\n\n"
                f"Candidates:\n{items}\n\n"
                f"Return a JSON array of unique 0-based indices, sorted by relevance, length <= {K}."
            )

            # Enforce timeout on the proxy call
            raw: str = ''
            try:
                with ThreadPoolExecutor(max_workers=1) as _ex:
                    fut = _ex.submit(self._mem0_llm_generate, model=model, system="rerank", user=usr_prompt)
                    raw = fut.result(timeout=max(0.1, (getattr(self, 'mem0_proxy_timeout_ms', 1200) or 1200) / 1000.0))
            except FuturesTimeout:
                self.logger.debug("mem0 rerank proxy timeout")
                return []

            # Extract JSON array of indices
            start = raw.find('[')
            end = raw.rfind(']')
            if start == -1 or end == -1 or end <= start:
                return []
            try:
                arr = json.loads(raw[start:end+1])
            except Exception:
                return []
            picked = [int(i) for i in arr if isinstance(i, (int, float)) and 0 <= int(i) < N]
            if not picked:
                return []
            # Deduplicate and cap to K
            seen: set = set()
            order: List[int] = []
            for i in picked:
                if i not in seen:
                    seen.add(i)
                    order.append(i)
                if len(order) >= K:
                    break
            return order
        except Exception as e:
            self.logger.debug(f"mem0 proxy rerank failed: {e}")
            return []

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
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
        user_id = getattr(self, 'mem0_user_id', str(self.mem0_config.get('user_id', 'cli-user')))

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
        user_id = None
        try:
            user_id = (filters or {}).get('user_id') if isinstance(filters, dict) else None
        except Exception:
            user_id = None
        # Try a series of signatures to support both Platform and OSS
        attempts = []
        if getattr(self, '_mem0_supports_version_kw', True):
            attempts.append(lambda: self.mem0_client.search(query, version=self._mem0_api_version_pref, filters=filters, limit=limit))
        attempts.append(lambda: self.mem0_client.search(query, filters=filters, limit=limit))
        if user_id is not None:
            if getattr(self, '_mem0_supports_version_kw', True):
                attempts.append(lambda: self.mem0_client.search(query=query, user_id=user_id, version=self._mem0_api_version_pref))
            attempts.append(lambda: self.mem0_client.search(query=query, user_id=user_id))
            if limit is not None:
                attempts.append(lambda: self.mem0_client.search(query=query, user_id=user_id, limit=limit))
        # Last resort: minimal
        attempts.append(lambda: self.mem0_client.search(query))
        last_type_error: Optional[Exception] = None
        for call in attempts:
            try:
                return call()
            except TypeError as te:
                # Signature mismatch; try next
                last_type_error = te
                self._mem0_supports_version_kw = False
                continue
        # If all attempts failed due to signature mismatches, raise one for outer handler
        if last_type_error is not None:
            raise last_type_error
        # Otherwise, generic failure
        raise RuntimeError("Mem0 search failed with all known signatures")

    def _mem0_get_all_api(self, filters: Optional[Dict[str, Any]] = None):
        """get_all wrapper that prefers v2 and falls back if the SDK doesn't accept the 'version' kwarg."""
        if not self.mem0_client:
            return []
        user_id = None
        try:
            user_id = (filters or {}).get('user_id') if isinstance(filters, dict) else None
        except Exception:
            user_id = None
        attempts = []
        if getattr(self, '_mem0_supports_version_kw', True):
            attempts.append(lambda: self.mem0_client.get_all(filters=filters, version=self._mem0_api_version_pref))
        attempts.append(lambda: self.mem0_client.get_all(filters=filters))
        if user_id is not None:
            if getattr(self, '_mem0_supports_version_kw', True):
                attempts.append(lambda: self.mem0_client.get_all(user_id=user_id, version=self._mem0_api_version_pref))
            attempts.append(lambda: self.mem0_client.get_all(user_id=user_id))
        for call in attempts:
            try:
                return call()
            except TypeError:
                self._mem0_supports_version_kw = False
                continue
        raise RuntimeError("Mem0 get_all failed with all known signatures")

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
            print("⚠️ Mem0 is not configured. Enable with MEM0_USE_LOCAL=1 (local OSS) or set MEM0_API_KEY (remote platform).")
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
        """NL memory helper: only respond to 'list memories'. Everything else goes to chat."""
        if not getattr(self, 'mem0_enabled', False) or not self.mem0_client:
            return False
        try:
            import re
            lower = text.lower()
            # normalize internal whitespace, trim and strip simple trailing punctuation
            norm = re.sub(r"\s+", " ", lower).strip().rstrip(".!?")
            if norm == "list memories":
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
            return False
        except Exception as e:
            self.logger.debug(f"Mem0 NLU list error: {e}")
            return False
    
    def interactive_mode(self):
        """Run interactive chat mode."""
        if not self.quiet:
            print("🚀 Ollama Turbo CLI - Interactive Mode")
            print(f"📝 Model: {self.model}")
            print(f"🔧 Tools: {'Enabled' if self.enable_tools else 'Disabled'}")
            print("💡 Commands: 'quit'/'exit' to exit, 'clear' to clear history, 'history' to show history, '/mem ...' for memory ops")
            if not getattr(self, 'mem0_enabled', False):
                print("Mem0: disabled (set MEM0_USE_LOCAL=1 for local OSS or provide MEM0_API_KEY for remote)")
            else:
                mode = getattr(self, 'mem0_mode', 'unknown')
                print(f"Mem0: enabled ({mode}, user: {self.mem0_user_id})")
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
