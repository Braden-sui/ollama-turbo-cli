from __future__ import annotations

"""
Configuration dataclasses for the modularized Ollama Turbo CLI backend.

Phase A scaffolding: these types mirror the existing runtime knobs in
`src/client.py`. They are not yet wired into the client, but provide a
stable place for future dependency injection without behavior changes.

IMPORTANT: Defaults here match current behavior as implemented in
`OllamaTurboClient`. Environment parsing helpers are provided to allow
non-invasive adoption later.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os


# ----------------------------- Helpers -----------------------------

def _env_bool(name: str, default: bool) -> bool:
    try:
        v = os.getenv(name)
        if v is None:
            return default
        return str(v).strip().lower() not in {"0", "false", "no", "off"}
    except Exception:
        return default


def _env_int(name: str, default: int, *, min_value: Optional[int] = None) -> int:
    try:
        v = os.getenv(name)
        if v is None or str(v).strip() == "":
            return default
        val = int(v)
        if min_value is not None:
            val = max(min_value, val)
        return val
    except Exception:
        return default


def _env_float(name: str, default: float, *, min_value: Optional[float] = None) -> float:
    try:
        v = os.getenv(name)
        if v is None or str(v).strip() == "":
            return default
        val = float(v)
        if min_value is not None:
            val = max(min_value, val)
        return val
    except Exception:
        return default


# ----------------------------- Dataclasses -----------------------------

@dataclass
class RetryConfig:
    enabled: bool = True  # CLI_RETRY_ENABLED
    max_retries: int = 3  # CLI_MAX_RETRIES


@dataclass
class TransportConfig:
    # Engine and host resolution
    engine: Optional[str] = None
    host: Optional[str] = None  # resolved by networking layer
    # Keep-alive
    keep_alive_raw: Optional[str] = field(default_factory=lambda: os.getenv("OLLAMA_KEEP_ALIVE") or None)
    warm_models: bool = field(default_factory=lambda: _env_bool("WARM_MODELS", True))
    # HTTP timeouts
    connect_timeout_s: float = field(default_factory=lambda: _env_float("CLI_CONNECT_TIMEOUT_S", 5.0, min_value=1.0))
    read_timeout_s: float = field(default_factory=lambda: _env_float("CLI_READ_TIMEOUT_S", 600.0, min_value=60.0))


@dataclass
class StreamingConfig:
    idle_reconnect_secs: int = field(default_factory=lambda: _env_int("CLI_STREAM_IDLE_RECONNECT_SECS", 90, min_value=10))


@dataclass
class SamplingConfig:
    reasoning: str = "high"  # low|medium|high
    reasoning_mode: str = "system"  # 'system' | 'request:top' | 'request:options'
    max_output_tokens: Optional[int] = None
    ctx_size: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


@dataclass
class ToolingConfig:
    enabled: bool = True
    print_limit: int = 200
    context_cap: int = field(default_factory=lambda: _env_int("TOOL_CONTEXT_MAX_CHARS", 4000))
    multi_round: bool = field(default_factory=lambda: _env_bool("MULTI_ROUND_TOOLS", True))
    max_rounds: int = field(default_factory=lambda: max(1, _env_int("TOOL_MAX_ROUNDS", 6)))
    results_format: str = field(default_factory=lambda: (os.getenv("TOOL_RESULTS_FORMAT") or "string").strip().lower())


@dataclass
class Mem0Config:
    # Static config
    enabled: bool = True
    local: bool = False  # MEM0_USE_LOCAL
    vector_provider: str = "chroma"
    vector_host: str = ":memory:"
    vector_port: int = 0
    ollama_url: Optional[str] = None
    llm_model: Optional[str] = None
    embedder_model: str = "nomic-embed-text"
    user_id: str = "cli-user"  # unified default across client and config
    agent_id: Optional[str] = None
    app_id: Optional[str] = None
    api_key: Optional[str] = None
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    # Runtime knobs
    debug: bool = False
    max_hits: int = 3
    search_timeout_ms: int = 200
    timeout_connect_ms: int = 1000
    timeout_read_ms: int = 2000
    add_queue_max: int = 256
    breaker_threshold: int = 3
    breaker_cooldown_ms: int = 60000
    search_workers: int = field(default_factory=lambda: _env_int("MEM0_SEARCH_WORKERS", 2))
    in_first_system: bool = field(default_factory=lambda: _env_bool("MEM0_IN_FIRST_SYSTEM", False))
    # Proxy / reranker controls
    proxy_model: Optional[str] = None  # MEM0_PROXY_MODEL
    proxy_timeout_ms: int = 1200      # MEM0_PROXY_TIMEOUT_MS
    rerank_search_limit: int = 10     # MEM0_RERANK_SEARCH_LIMIT


@dataclass
class ReliabilityConfig:
    ground: bool = False
    k: Optional[int] = None
    cite: bool = False
    check: str = "off"  # off|warn|enforce
    consensus: bool = False
    eval_corpus: Optional[str] = None


@dataclass
class HistoryConfig:
    max_history: int = field(default_factory=lambda: max(2, min(_env_int("MAX_CONVERSATION_HISTORY", 10), 10)))


@dataclass
class ClientRuntimeConfig:
    model: str = "gpt-oss:120b"
    protocol: str = "auto"
    quiet: bool = False
    show_trace: bool = False

    retry: RetryConfig = field(default_factory=RetryConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    tooling: ToolingConfig = field(default_factory=ToolingConfig)
    mem0: Mem0Config = field(default_factory=Mem0Config)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)

    @classmethod
    def from_env(
        cls,
        *,
        model: Optional[str] = None,
        protocol: Optional[str] = None,
        quiet: Optional[bool] = None,
        show_trace: Optional[bool] = None,
        engine: Optional[str] = None,
    ) -> "ClientRuntimeConfig":
        """
        Construct a runtime config from environment variables, mirroring
        the defaults used by the current `OllamaTurboClient`.
        """
        cfg = cls()
        if model is not None:
            cfg.model = model
        if protocol is not None:
            cfg.protocol = protocol
        if quiet is not None:
            cfg.quiet = quiet
        if show_trace is not None:
            cfg.show_trace = show_trace
        if engine is not None:
            cfg.transport.engine = engine

        # Retry
        cfg.retry.enabled = _env_bool("CLI_RETRY_ENABLED", True)
        cfg.retry.max_retries = _env_int("CLI_MAX_RETRIES", 3, min_value=0)

        # Transport
        cfg.transport.keep_alive_raw = os.getenv("OLLAMA_KEEP_ALIVE") or None
        cfg.transport.warm_models = _env_bool("WARM_MODELS", True)
        cfg.transport.connect_timeout_s = _env_float("CLI_CONNECT_TIMEOUT_S", 5.0, min_value=1.0)
        cfg.transport.read_timeout_s = _env_float("CLI_READ_TIMEOUT_S", 600.0, min_value=60.0)

        # Streaming
        cfg.streaming.idle_reconnect_secs = _env_int("CLI_STREAM_IDLE_RECONNECT_SECS", 90, min_value=10)

        # Tooling
        cfg.tooling.context_cap = _env_int("TOOL_CONTEXT_MAX_CHARS", 4000)
        cfg.tooling.multi_round = _env_bool("MULTI_ROUND_TOOLS", True)
        cfg.tooling.max_rounds = max(1, _env_int("TOOL_MAX_ROUNDS", 6))
        trf = (os.getenv("TOOL_RESULTS_FORMAT") or "string").strip().lower()
        cfg.tooling.results_format = "object" if trf == "object" else "string"

        # Mem0
        cfg.mem0.enabled = _env_bool("MEM0_ENABLED", cfg.mem0.enabled)
        cfg.mem0.local = _env_bool("MEM0_USE_LOCAL", cfg.mem0.local)
        cfg.mem0.search_workers = _env_int("MEM0_SEARCH_WORKERS", 2)
        cfg.mem0.in_first_system = _env_bool("MEM0_IN_FIRST_SYSTEM", False)
        cfg.mem0.user_id = os.getenv("MEM0_USER_ID", cfg.mem0.user_id)
        cfg.mem0.agent_id = os.getenv("MEM0_AGENT_ID") or None
        cfg.mem0.app_id = os.getenv("MEM0_APP_ID") or None
        cfg.mem0.api_key = os.getenv("MEM0_API_KEY") or None
        cfg.mem0.org_id = os.getenv("MEM0_ORG_ID") or None
        cfg.mem0.project_id = os.getenv("MEM0_PROJECT_ID") or None
        cfg.mem0.proxy_model = os.getenv("MEM0_PROXY_MODEL") or None
        try:
            cfg.mem0.proxy_timeout_ms = _env_int("MEM0_PROXY_TIMEOUT_MS", cfg.mem0.proxy_timeout_ms, min_value=100)
        except Exception:
            pass
        try:
            cfg.mem0.rerank_search_limit = _env_int("MEM0_RERANK_SEARCH_LIMIT", cfg.mem0.rerank_search_limit, min_value=1)
        except Exception:
            pass

        # History window
        cfg.history.max_history = max(2, min(_env_int("MAX_CONVERSATION_HISTORY", 10), 10))

        return cfg
