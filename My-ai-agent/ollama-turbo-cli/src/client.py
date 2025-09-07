"""
Ollama Turbo client implementation with tool calling and streaming support.
Reliability hardening: retries/backoff, idempotency keys, keep-alive pools, and
streaming idle reconnects (all behind env flags with safe defaults).
"""

import json
import sys
import logging
import os
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
from .reliability_integration.integration import ReliabilityIntegration
from .protocols import get_adapter
from .transport import networking as _net
from .streaming import runner as _runner, standard as _standard
from .tools_runtime.executor import ToolRuntimeExecutor
from .memory.mem0 import Mem0Service


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
                 mem0_embedder_model: str = 'embeddinggemma',
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
        # Reliability integration facade (Phase F)
        self.reliability_integration = ReliabilityIntegration()
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
        # Default: when using local Mem0 and no explicit Mem0 Ollama URL is provided,
        # point the embedder at local Ollama by default.
        mem0_ollama_default = os.getenv('MEM0_OLLAMA_URL') or mem0_ollama_url
        if mem0_local and not mem0_ollama_default:
            mem0_ollama_default = 'http://localhost:11434'
        self.mem0_config = {
            'enabled': mem0_enabled,
            'local': mem0_local,
            'vector_provider': mem0_vector_provider,
            'vector_host': mem0_vector_host,
            'vector_port': mem0_vector_port,
            # Embedder base URL. Note: In local mode, Mem0Service ignores this for the embedder
            # (it defaults to MEM0_EMBEDDER_OLLAMA_URL or http://localhost:11434) and selects the
            # LLM base from MEM0_LLM_OLLAMA_URL or ctx.host. This field remains for compatibility.
            'ollama_url': (mem0_ollama_default or self.host),
            'llm_model': mem0_llm_model or model,  # Default to main model
            'embedder_model': (mem0_embedder_model or ('embeddinggemma' if mem0_local else mem0_embedder_model)),
            'user_id': mem0_user_id,
        }
        
        # Initialize Mem0 memory system (optional)
        self.mem0_service = Mem0Service(self.mem0_config)
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
            # Do NOT clear here; chat() resets at the start of each turn.
            # self.trace = []
    
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
        """Delegated to ReliabilityIntegration (Phase F)."""
        return self.reliability_integration.load_system_cited(self)

    def _prepare_reliability_context(self, user_message: str) -> None:
        """Delegated to ReliabilityIntegration (Phase F)."""
        return self.reliability_integration.prepare_context(self, user_message)
    
    # ------------------ Mem0 Integration ------------------
    def _init_mem0(self) -> None:
        """Initialize Mem0 client and runtime settings from configuration (delegated)."""
        return self.mem0_service.initialize(self)

    def _inject_mem0_context(self, user_message: str) -> None:
        """Search Mem0 for relevant memories and inject as a system message (delegated)."""
        return self.mem0_service.inject_context(self, user_message)

    def _mem0_llm_generate(self, *, model: str, system: str, user: str) -> str:
        """Low-level reranker generation (delegated)."""
        return self.mem0_service.llm_generate(self, model=model, system=system, user=user)

    def _mem0_rerank_with_proxy(self, query: str, candidates: List[str], *, model: str, k: Optional[int] = None) -> List[int]:
        """Rerank candidate memory snippets for the current query (delegated)."""
        return self.mem0_service.rerank_with_proxy(self, query, candidates, model=model, k=k)

    def _mem0_add_after_response(self, user_message: Optional[str], assistant_message: Optional[str]) -> None:
        """Queue the interaction to Mem0 for persistence (delegated)."""
        return self.mem0_service.persist_turn(self, user_message, assistant_message)

    def _mem0_enqueue_add(self, messages: List[Dict[str, str]], metadata: Dict[str, Any]) -> None:
        """Enqueue an add job (delegated)."""
        return self.mem0_service.enqueue_add(self, messages, metadata)

    def _mem0_worker_loop(self) -> None:
        return self.mem0_service.worker_loop(self)

    def _mem0_execute_add(self, messages: List[Dict[str, str]], metadata: Dict[str, Any]) -> List[str]:
        return self.mem0_service.execute_add(self, messages, metadata)

    def _mem0_enforce_metadata(self, ids: List[str], metadata: Dict[str, Any]) -> None:
        return self.mem0_service.enforce_metadata(self, ids, metadata)

    def _mem0_search_api(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
        """Search wrapper (delegated)."""
        return self.mem0_service.search_api(self, query, filters=filters, limit=limit)

    def _mem0_get_all_api(self, filters: Optional[Dict[str, Any]] = None):
        """get_all wrapper (delegated)."""
        return self.mem0_service.get_all_api(self, filters=filters)

    def _normalize_fact(self, text: str) -> str:
        t = ' '.join(text.strip().split())
        return t.lower()
    
    
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
                    self.mem0_service.handle_command(self, user_input)
                    continue

                # Natural language memory handlers
                if self.mem0_service.handle_nlu(self, user_input):
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
        """Legacy shutdown hook. Mem0Service owns flushing and atexit now."""
        try:
            self.mem0_service.shutdown(self)
        except Exception:
            pass
