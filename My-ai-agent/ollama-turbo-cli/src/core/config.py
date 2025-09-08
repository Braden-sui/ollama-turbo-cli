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
class ReasoningInjectionConfig:
    """Controls request-level reasoning injection placement and style.

    field_path: dot-path in the request payload to place reasoning effort.
      - Typical: 'options.reasoning_effort' (for request:options mode)
      - Or: 'reasoning' (for request:top mode)
      - Leave empty to auto-pick based on reasoning_mode.
    field_style: 'string' | 'object'
      - 'string' → injects a simple string (e.g., 'high').
      - 'object' → injects {object_key: 'high'}.
    object_key: key to use when style is 'object' (default: 'effort').
    """
    field_path: str = ""  # default: auto by mode
    field_style: str = field(default_factory=lambda: (os.getenv("REASONING_FIELD_STYLE") or "string").strip().lower())
    object_key: str = field(default_factory=lambda: os.getenv("REASONING_OBJECT_KEY", "effort"))


@dataclass
class PromptConfig:
    """Prompt presentation preferences.

    verbosity: 'concise' | 'detailed' — controls style line in system prompts.
    fewshots: include few-shot section (disabled by default to preserve behavior).
    verbose_after_tools: use the more verbose post-tool reprompt (off by default).
    """
    verbosity: str = field(default_factory=lambda: (os.getenv("PROMPT_VERBOSITY", "concise") or "concise").lower())
    fewshots: bool = field(default_factory=lambda: _env_bool("PROMPT_FEWSHOTS", False))
    verbose_after_tools: bool = field(default_factory=lambda: _env_bool("PROMPT_VERBOSE_AFTER_TOOLS", False))

@dataclass
class ToolingConfig:
    enabled: bool = True
    print_limit: int = 2000
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
    ollama_url: Optional[str] = None  # embedder base (compat)
    # Optional explicit bases
    llm_base_url: Optional[str] = None
    embedder_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    embedder_model: str = "embeddinggemma"
    user_id: str = "cli-user"  # unified default across client and config
    agent_id: Optional[str] = None
    app_id: Optional[str] = None
    api_key: Optional[str] = None
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    # Runtime knobs
    debug: bool = False
    max_hits: int = 10
    search_timeout_ms: int = 700
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
    # Context construction
    context_budget_chars: int = 1200   # MEM0_CONTEXT_BUDGET_CHARS
    # Output format for mem0 client responses (avoid deprecation of v1.0)
    output_format: str = field(default_factory=lambda: os.getenv("MEM0_OUTPUT_FORMAT", "v1.1"))


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
class RerankProviderSpec:
    name: str = "cohere"  # cohere|voyage
    model: str = ""
    base_url: Optional[str] = None
    top_n: int = 20
    weight: float = 1.0


@dataclass
class WebConfig:
    # Identity
    user_agent: str = field(default_factory=lambda: os.getenv("WEB_UA", "ollama-turbo-cli-web/1.0 (+https://github.com)"))
    # Timeouts (seconds)
    timeout_connect: float = field(default_factory=lambda: _env_float("WEB_TIMEOUT_CONNECT", 5.0))
    timeout_read: float = field(default_factory=lambda: _env_float("WEB_TIMEOUT_READ", 15.0))
    timeout_write: float = field(default_factory=lambda: _env_float("WEB_TIMEOUT_WRITE", 10.0))
    # Retries
    retry_attempts: int = field(default_factory=lambda: _env_int("WEB_RETRY_ATTEMPTS", 3, min_value=0))
    retry_backoff_base: float = field(default_factory=lambda: _env_float("WEB_RETRY_BACKOFF_BASE", 0.4))
    retry_backoff_max: float = field(default_factory=lambda: _env_float("WEB_RETRY_BACKOFF_MAX", 6.0))
    # Concurrency
    max_connections: int = field(default_factory=lambda: _env_int("WEB_MAX_CONNECTIONS", 20, min_value=1))
    max_keepalive: int = field(default_factory=lambda: _env_int("WEB_MAX_KEEPALIVE", 10, min_value=0))
    per_host_concurrency: int = field(default_factory=lambda: _env_int("WEB_PER_HOST_CONCURRENCY", 4, min_value=1))
    # Fetch behavior
    follow_redirects: bool = field(default_factory=lambda: _env_bool("WEB_FOLLOW_REDIRECTS", True))
    head_gating_enabled: bool = field(default_factory=lambda: _env_bool("WEB_HEAD_GATING", True))
    max_download_bytes: int = field(default_factory=lambda: _env_int("WEB_MAX_DOWNLOAD_BYTES", 10 * 1024 * 1024, min_value=1024))
    accept_header_override: str = field(default_factory=lambda: os.getenv("WEB_ACCEPT_HEADER", ""))
    client_pool_size: int = field(default_factory=lambda: _env_int("WEB_CLIENT_POOL_SIZE", 16, min_value=1))
    # Caching and robots
    cache_ttl_seconds: int = field(default_factory=lambda: _env_int("WEB_CACHE_TTL_SECONDS", 86400, min_value=0))
    robots_ttl_seconds: int = field(default_factory=lambda: _env_int("WEB_ROBOTS_TTL_SECONDS", 3600, min_value=0))
    max_crawl_delay_s: int = field(default_factory=lambda: _env_int("WEB_MAX_CRAWL_DELAY_S", 20, min_value=0))
    # Provider keys
    brave_key: Optional[str] = field(default_factory=lambda: os.getenv("BRAVE_API_KEY"))
    tavily_key: Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    exa_key: Optional[str] = field(default_factory=lambda: os.getenv("EXA_API_KEY"))
    google_pse_cx: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_PSE_CX"))
    google_pse_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_PSE_KEY"))
    # Rerank
    cohere_key: Optional[str] = field(default_factory=lambda: os.getenv("COHERE_API_KEY"))
    voyage_key: Optional[str] = field(default_factory=lambda: os.getenv("VOYAGE_API_KEY"))
    rerank_enabled: bool = field(default_factory=lambda: _env_bool("WEB_RERANK_ENABLED", True))
    rerank_mode: str = field(default_factory=lambda: os.getenv("WEB_RERANK_MODE", "sdk"))  # sdk|rest
    rerank_timeout_ms: int = field(default_factory=lambda: _env_int("WEB_RERANK_TIMEOUT_MS", 2000, min_value=100))
    rerank_cache_ttl_s: int = field(default_factory=lambda: _env_int("WEB_RERANK_CACHE_TTL_S", 300, min_value=0))
    rerank_breaker_threshold: int = field(default_factory=lambda: _env_int("WEB_RERANK_BREAKER_THRESHOLD", 3, min_value=1))
    rerank_breaker_cooldown_ms: int = field(default_factory=lambda: _env_int("WEB_RERANK_BREAKER_COOLDOWN_MS", 60000, min_value=1000))
    rerank_providers: list[RerankProviderSpec] = field(default_factory=lambda: [
        RerankProviderSpec(name="cohere", model="rerank-english-v3.0"),
        RerankProviderSpec(name="voyage", model="rerank-2"),
    ])
    # Policies
    respect_robots: bool = field(default_factory=lambda: _env_bool("WEB_RESPECT_ROBOTS", True))
    allow_browser: bool = field(default_factory=lambda: _env_bool("WEB_ALLOW_BROWSER", True))
    # Debugging / fallbacks
    emergency_bootstrap: bool = field(default_factory=lambda: _env_bool("WEB_EMERGENCY_BOOTSTRAP", True))
    debug_metrics: bool = field(default_factory=lambda: _env_bool("WEB_DEBUG_METRICS", False))
    # Rate limiting
    rate_tokens_per_host: int = field(default_factory=lambda: _env_int("WEB_RATE_TOKENS_PER_HOST", 4, min_value=1))
    rate_refill_per_sec: float = field(default_factory=lambda: _env_float("WEB_RATE_REFILL_PER_SEC", 0.5, min_value=0.01))
    respect_retry_after: bool = field(default_factory=lambda: _env_bool("WEB_RESPECT_RETRY_AFTER", True))
    # Allowlist integration (reuse sandbox policy)
    sandbox_allow: Optional[str] = field(default_factory=lambda: os.getenv("SANDBOX_NET_ALLOW", ""))
    sandbox_allow_http: bool = field(default_factory=lambda: _env_bool("SANDBOX_ALLOW_HTTP", False))
    sandbox_allow_proxies: bool = field(default_factory=lambda: _env_bool("SANDBOX_ALLOW_PROXIES", False))
    # Proxy environment (centralized)
    http_proxy: Optional[str] = field(default_factory=lambda: (os.getenv("HTTP_PROXY") or os.getenv("http_proxy")))
    https_proxy: Optional[str] = field(default_factory=lambda: (os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")))
    all_proxy: Optional[str] = field(default_factory=lambda: (os.getenv("ALL_PROXY") or os.getenv("all_proxy")))
    no_proxy: Optional[str] = field(default_factory=lambda: (os.getenv("NO_PROXY") or os.getenv("no_proxy")))
    # Storage locationsq
    cache_root: str = field(default_factory=lambda: os.getenv("WEB_CACHE_ROOT", ".sandbox/webcache"))
    archive_enabled: bool = field(default_factory=lambda: _env_bool("WEB_ARCHIVE_ENABLED", True))
    archive_check_memento_first: bool = field(default_factory=lambda: _env_bool("WEB_ARCHIVE_CHECK_FIRST", False))
    archive_retry_on_429: bool = field(default_factory=lambda: _env_bool("WEB_ARCHIVE_RETRY_ON_429", True))
    # Browser limits
    browser_max_pages: int = field(default_factory=lambda: _env_int("WEB_BROWSER_MAX_PAGES", 10, min_value=1))
    browser_wait_ms: int = field(default_factory=lambda: _env_int("WEB_BROWSER_WAIT_MS", 1200, min_value=0))
    browser_block_resources: str = field(default_factory=lambda: os.getenv("WEB_BROWSER_BLOCK_RESOURCES", "image,font,media"))
    browser_stealth_light: bool = field(default_factory=lambda: _env_bool("WEB_BROWSER_STEALTH_LIGHT", False))
    # Content post-processing toggles
    clean_wiki_edit_anchors: bool = field(default_factory=lambda: _env_bool("WEB_CLEAN_WIKI_EDIT_ANCHORS", True))
    # Sitemaps
    sitemap_enabled: bool = field(default_factory=lambda: _env_bool("WEB_SITEMAP_ENABLED", False))
    sitemap_max_urls: int = field(default_factory=lambda: _env_int("WEB_SITEMAP_MAX_URLS", 50, min_value=1))
    sitemap_timeout_s: float = field(default_factory=lambda: _env_float("WEB_SITEMAP_TIMEOUT_S", 5.0, min_value=1.0))
    sitemap_include_subs: bool = field(default_factory=lambda: _env_bool("WEB_SITEMAP_INCLUDE_SUBS", True))

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
    reasoning_injection: ReasoningInjectionConfig = field(default_factory=ReasoningInjectionConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    mem0: Mem0Config = field(default_factory=Mem0Config)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    web: WebConfig = field(default_factory=WebConfig)

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
        # Model and protocol: CLI overrides first, then env, then defaults
        if model is not None:
            cfg.model = model
        else:
            cfg.model = os.getenv("OLLAMA_MODEL", cfg.model)
        if protocol is not None:
            cfg.protocol = protocol
        else:
            cfg.protocol = os.getenv("OLLAMA_PROTOCOL", cfg.protocol)
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

        # Sampling: reasoning, mode, caps, and penalties
        try:
            r_env = os.getenv("REASONING")
            if r_env:
                r = r_env.strip().lower()
                if r in {"low", "medium", "high"}:
                    cfg.sampling.reasoning = r
        except Exception:
            pass
        try:
            rm_env = os.getenv("REASONING_MODE")
            if rm_env:
                rm = rm_env.strip().lower()
                if rm in {"system", "request:top", "request:options"}:
                    cfg.sampling.reasoning_mode = rm
        except Exception:
            pass
        # Numeric sampling envs (optional)
        try:
            v = os.getenv("MAX_OUTPUT_TOKENS")
            if v and v.strip().isdigit():
                cfg.sampling.max_output_tokens = int(v)
        except Exception:
            pass
        try:
            v = os.getenv("CTX_SIZE")
            if v and v.strip().isdigit():
                cfg.sampling.ctx_size = int(v)
        except Exception:
            pass
        try:
            v = os.getenv("TEMPERATURE")
            if v not in (None, ""):
                cfg.sampling.temperature = float(v)
        except Exception:
            pass
        try:
            v = os.getenv("TOP_P")
            if v not in (None, ""):
                cfg.sampling.top_p = float(v)
        except Exception:
            pass
        try:
            v = os.getenv("PRESENCE_PENALTY")
            if v not in (None, ""):
                cfg.sampling.presence_penalty = float(v)
        except Exception:
            pass
        try:
            v = os.getenv("FREQUENCY_PENALTY")
            if v not in (None, ""):
                cfg.sampling.frequency_penalty = float(v)
        except Exception:
            pass

        # Mem0
        cfg.mem0.enabled = _env_bool("MEM0_ENABLED", cfg.mem0.enabled)
        cfg.mem0.local = _env_bool("MEM0_USE_LOCAL", cfg.mem0.local)
        # Static/local embeddings bases
        cfg.mem0.ollama_url = os.getenv("MEM0_OLLAMA_URL") or os.getenv("MEM0_OLLAMA_BASE_URL") or cfg.mem0.ollama_url
        cfg.mem0.llm_base_url = os.getenv("MEM0_LLM_OLLAMA_URL") or cfg.mem0.llm_base_url
        cfg.mem0.embedder_base_url = os.getenv("MEM0_EMBEDDER_OLLAMA_URL") or cfg.mem0.embedder_base_url
        # Vector store settings
        cfg.mem0.vector_provider = os.getenv("MEM0_VECTOR_PROVIDER", cfg.mem0.vector_provider)
        cfg.mem0.vector_host = os.getenv("MEM0_VECTOR_HOST", cfg.mem0.vector_host)
        try:
            cfg.mem0.vector_port = _env_int("MEM0_VECTOR_PORT", cfg.mem0.vector_port)
        except Exception:
            pass
        # Models and identity
        cfg.mem0.llm_model = os.getenv("MEM0_LLM_MODEL") or cfg.mem0.llm_model
        cfg.mem0.embedder_model = os.getenv("MEM0_EMBEDDER_MODEL", cfg.mem0.embedder_model)
        cfg.mem0.search_workers = _env_int("MEM0_SEARCH_WORKERS", 2)
        # Mem0 search timeout (ms)
        try:
            cfg.mem0.search_timeout_ms = _env_int("MEM0_SEARCH_TIMEOUT_MS", cfg.mem0.search_timeout_ms, min_value=50)
        except Exception:
            pass
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
        try:
            cfg.mem0.context_budget_chars = _env_int("MEM0_CONTEXT_BUDGET_CHARS", cfg.mem0.context_budget_chars, min_value=100)
        except Exception:
            pass

        # History window
        cfg.history.max_history = max(2, min(_env_int("MAX_CONVERSATION_HISTORY", 10), 10))

        # Reasoning injection (override field_path only; style/object_key picked up via defaults)
        try:
            fp = os.getenv("REASONING_FIELD_PATH")
            if fp is not None:
                cfg.reasoning_injection.field_path = fp.strip()
        except Exception:
            pass

        return cfg
