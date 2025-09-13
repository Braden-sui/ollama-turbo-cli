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
import logging
import warnings
import uuid
import random

_LOG_SILENCERS = (os.getenv("LOG_SILENCERS", "1").strip().lower() not in {"0","false","no","off"})
_BACKOFF_SILENCE = (os.getenv("BACKOFF_SILENCE", "1").strip().lower() not in {"0","false","no","off"})

# Debug banner (emits only when LOG_LEVEL=DEBUG)
_BANNER_EMITTED = False
def _emit_debug_banner_if_needed() -> None:
    global _BANNER_EMITTED
    try:
        lvl = (os.getenv("LOG_LEVEL", "INFO") or "INFO").strip().upper()
        if lvl != "DEBUG" or _BANNER_EMITTED:
            return
        worker = os.getenv("PYTEST_XDIST_WORKER") or os.getenv("WORKER_ID") or ""
        cache_root = os.getenv("WEB_CACHE_ROOT", ".sandbox/webcache")
        per_worker = (os.getenv("WEB_CACHE_PER_WORKER", "0").strip().lower() not in {"0","false","no","off"})
        effective_cache_root = os.path.join(cache_root, worker) if (per_worker and worker) else cache_root
        lines = [
            "=== debug: core.config ===",
            f"LOG_LEVEL={lvl}",
            f"LOG_SILENCERS={'on' if _LOG_SILENCERS else 'off'}",
            f"BACKOFF_SILENCE={'on' if _BACKOFF_SILENCE else 'off'}",
            f"WEB_CACHE_PER_WORKER={'on' if per_worker else 'off'}",
            f"PYTEST_XDIST_WORKER={worker or 'n/a'}",
            f"effective_cache_root={effective_cache_root}",
        ]
        try:
            import sys as _sys
            print("\n".join(lines), file=_sys.stderr)
        except Exception:
            pass
        _BANNER_EMITTED = True
    except Exception:
        pass

# Configure backoff to be less noisy (if enabled)
if _BACKOFF_SILENCE:
    logging.getLogger('backoff').addHandler(logging.NullHandler())
    logging.getLogger('backoff').propagate = False
_emit_debug_banner_if_needed()


# Global silencers for noisy third-party loggers and known deprecations.
# Keep behavior backward compatible with previous scattered locations.
if _LOG_SILENCERS:
    try:
        # Only dampen noise when not in DEBUG
        if (os.getenv("LOG_LEVEL", "INFO") or "INFO").strip().upper() != "DEBUG":
            for name in ("httpx", "ollama", "urllib3", "requests", "trafilatura", "readability.readability"):
                try:
                    logging.getLogger(name).setLevel(logging.WARNING)
                except Exception:
                    pass
    except Exception:
        pass

# Warning filters that were previously applied in CLI; relocate here to centralize behavior.
if _LOG_SILENCERS:
    try:
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*SwigPy.*has no __module__ attribute.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*swigvarlink has no __module__ attribute.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*has no __module__ attribute.*",
        )
        # Silence Pydantic v2.11 deprecation from ollama client (model_fields on instance)
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r"Accessing the 'model_fields' attribute on the instance is deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"ollama\._types",
            message=r".*model_fields.*deprecated.*",
        )
        # Silence Pydantic V2 deprecation about class-based `config` (originates in third-party libs)
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r"Support for class-based `config` is deprecated, use ConfigDict instead.*",
        )
        # Same message, but pydantic uses a custom warning type; match by module + generic Warning
        warnings.filterwarnings(
            "ignore",
            message=r"Support for class-based `config` is deprecated, use ConfigDict instead.*",
            category=Warning,
            module=r"pydantic\._internal\._config",
        )
    except Exception:
        pass


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
    max_rounds: int = field(default_factory=lambda: max(1, _env_int("TOOL_MAX_ROUNDS",10)))
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
    search_timeout_ms: int = 800
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
    context_budget_chars: int = 12000   # MEM0_CONTEXT_BUDGET_CHARS
    # Output format for mem0 client responses (avoid deprecation of v1.0)
    output_format: str = field(default_factory=lambda: os.getenv("MEM0_OUTPUT_FORMAT", "v1.1"))


@dataclass
class ReliabilityConfig:
    ground: bool = True
    k: Optional[int] = None
    cite: bool = True
    check: str = "enforce"  # off|warn|enforce
    consensus: bool = False
    eval_corpus: Optional[str] = None
    # Default research fallback behavior when local retrieval is insufficient
    # Values: 'web' (use web_research), 'off' (no fallback)
    ground_fallback: str = field(default_factory=lambda: (os.getenv("RAG_GROUND_FALLBACK", "web").strip().lower()))


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
    max_connections: int = field(default_factory=lambda: _env_int("WEB_MAX_CONNECTIONS", 32, min_value=1))
    max_keepalive: int = field(default_factory=lambda: _env_int("WEB_MAX_KEEPALIVE", 20, min_value=0))
    per_host_concurrency: int = field(default_factory=lambda: _env_int("WEB_PER_HOST_CONCURRENCY", 6, min_value=1))
    # Fetch behavior
    follow_redirects: bool = field(default_factory=lambda: _env_bool("WEB_FOLLOW_REDIRECTS", True))
    head_gating_enabled: bool = field(default_factory=lambda: _env_bool("WEB_HEAD_GATING", True))
    max_download_bytes: int = field(default_factory=lambda: _env_int("WEB_MAX_DOWNLOAD_BYTES", 10 * 1024 * 1024, min_value=1024))
    accept_header_override: str = field(default_factory=lambda: os.getenv("WEB_ACCEPT_HEADER", ""))
    client_pool_size: int = field(default_factory=lambda: _env_int("WEB_CLIENT_POOL_SIZE", 32, min_value=1))
    # Caching and robots
    cache_ttl_seconds: int = field(default_factory=lambda: _env_int("WEB_CACHE_TTL_SECONDS", 86400, min_value=0))
    # Short TTL for raw fetch artifacts (HTML/JS). Processed markdown should be persisted longer by callers.
    raw_artifacts_ttl_hours: int = field(default_factory=lambda: _env_int("WEB_RAW_ARTIFACTS_TTL_HOURS", 24, min_value=1))
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
    # Query policy hardening
    year_guard_enabled: bool = field(default_factory=lambda: _env_bool("WEB_YEAR_GUARD_ENABLED", True))
    default_freshness_days: int = field(default_factory=lambda: _env_int("WEB_DEFAULT_FRESHNESS_DAYS", 90, min_value=1))
    breaking_freshness_days: int = field(default_factory=lambda: _env_int("WEB_BREAKING_FRESHNESS_DAYS", 30, min_value=1))
    slow_freshness_days: int = field(default_factory=lambda: _env_int("WEB_SLOW_FRESHNESS_DAYS", 365, min_value=1))
    allowlist_domains: list[str] = field(default_factory=lambda: [
        d.strip().lower() for d in (
            os.getenv("WEB_ALLOWLIST_DOMAINS", "reuters.com,apnews.com,ft.com,bloomberg.com,wsj.com,spaceflightnow.com,space.com").split(",")
        ) if d.strip()
    ])
    dateline_soft_accept: bool = field(default_factory=lambda: _env_bool("WEB_DATELINE_SOFT_ACCEPT", False))
    enable_allowlist_news_fallback: bool = field(default_factory=lambda: _env_bool("WEB_ENABLE_ALLOWLIST_NEWS_FALLBACK", True))
    # Trust subsystem
    trust_mode: str = field(default_factory=lambda: (os.getenv("WEB_TRUST_MODE", "allowlist").strip().lower()))
    trust_threshold: float = field(default_factory=lambda: _env_float("WEB_TRUST_THRESHOLD", 0.6))
    # Citation policy
    exclude_citation_domains: list[str] = field(default_factory=lambda: [
        d.strip().lower() for d in (
            os.getenv("WEB_EXCLUDE_CITATION_DOMAINS", "wikipedia.org,reddit.com").split(",")
        ) if d.strip()
    ])
    # Debugging / fallbacks
    emergency_bootstrap: bool = field(default_factory=lambda: _env_bool("WEB_EMERGENCY_BOOTSTRAP", True))
    debug_metrics: bool = field(default_factory=lambda: _env_bool("WEB_DEBUG_METRICS", False))
    # Tier sweep (multi-turn allowlist/tier-first pass)
    enable_tier_sweep: bool = field(default_factory=lambda: _env_bool("WEB_TIER_SWEEP", True))
    tier_sweep_max_sites: int = field(default_factory=lambda: _env_int("WEB_TIER_SWEEP_MAX_SITES", 12, min_value=1))
    tier_sweep_strict: bool = field(default_factory=lambda: _env_bool("WEB_TIER_SWEEP_STRICT", False))
    # PR9 adaptive tier sweep
    tier_sweep_adaptive_enable: bool = field(default_factory=lambda: _env_bool("WEB_TIER_SWEEP_ADAPTIVE_ENABLE", False))
    tier_sweep_initial_sites: int = field(default_factory=lambda: _env_int("WEB_TIER_SWEEP_INITIAL_SITES", 12, min_value=1))
    tier_sweep_max_sites_cap: int = field(default_factory=lambda: _env_int("WEB_TIER_SWEEP_MAX_SITES_CAP", 24, min_value=1))
    tier_sweep_quota_fast_count: int = field(default_factory=lambda: _env_int("WEB_TIER_SWEEP_QUOTA_FAST_COUNT", 2, min_value=1))
    tier_sweep_quota_fast_hours: int = field(default_factory=lambda: _env_int("WEB_TIER_SWEEP_QUOTA_FAST_HOURS", 2, min_value=1))
    tier_sweep_quota_slow_count: int = field(default_factory=lambda: _env_int("WEB_TIER_SWEEP_QUOTA_SLOW_COUNT", 3, min_value=1))
    tier_sweep_quota_slow_hours: int = field(default_factory=lambda: _env_int("WEB_TIER_SWEEP_QUOTA_SLOW_HOURS", 24, min_value=1))
    # Rate limiting
    rate_tokens_per_host: int = field(default_factory=lambda: _env_int("WEB_RATE_TOKENS_PER_HOST", 4, min_value=1))
    rate_refill_per_sec: float = field(default_factory=lambda: _env_float("WEB_RATE_REFILL_PER_SEC", 0.5, min_value=0.01))
    respect_retry_after: bool = field(default_factory=lambda: _env_bool("WEB_RESPECT_RETRY_AFTER", True))
    # Allowlist integration (reuse sandbox policy)
    sandbox_allow: Optional[str] = field(default_factory=lambda: os.getenv("SANDBOX_NET_ALLOW", "*"))
    sandbox_allow_http: bool = field(default_factory=lambda: _env_bool("SANDBOX_ALLOW_HTTP", False))
    sandbox_allow_proxies: bool = field(default_factory=lambda: _env_bool("SANDBOX_ALLOW_PROXIES", False))
    # Network safety policy
    block_private_ips: bool = field(default_factory=lambda: _env_bool("SANDBOX_BLOCK_PRIVATE_IPS", True))
    # Extractor backend gating and log silencing (PR: extractor hygiene)
    extract_trafilatura_enabled: bool = field(default_factory=lambda: _env_bool("WEB_EXTRACT_TRAFILATURA", True))
    extract_readability_enabled: bool = field(default_factory=lambda: _env_bool("WEB_EXTRACT_READABILITY", True))
    extract_jina_enabled: bool = field(default_factory=lambda: _env_bool("WEB_EXTRACT_JINA", True))
    extract_pymupdf_enabled: bool = field(default_factory=lambda: _env_bool("WEB_EXTRACT_PYMUPDF", True))
    extract_pdfminer_enabled: bool = field(default_factory=lambda: _env_bool("WEB_EXTRACT_PDFMINER", True))
    extract_ocr_enabled: bool = field(default_factory=lambda: _env_bool("WEB_EXTRACT_OCR", False))
    extract_silence_warnings: bool = field(default_factory=lambda: _env_bool("WEB_EXTRACT_SILENCE_WARNINGS", True))
    # Proxy environment (centralized)
    http_proxy: Optional[str] = field(default_factory=lambda: (os.getenv("HTTP_PROXY") or os.getenv("http_proxy")))
    https_proxy: Optional[str] = field(default_factory=lambda: (os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")))
    all_proxy: Optional[str] = field(default_factory=lambda: (os.getenv("ALL_PROXY") or os.getenv("all_proxy")))
    no_proxy: Optional[str] = field(default_factory=lambda: (os.getenv("NO_PROXY") or os.getenv("no_proxy")))
    # Storage locations
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
    # Query handling knobs (Commit 1 – defaults preserve current behavior)
    query_compression_mode: str = field(
        default_factory=lambda: (os.getenv("WEB_QUERY_COMPRESSION_MODE", "aggressive") or "aggressive").strip().lower()
    )
    query_max_tokens_fallback: int = field(
        default_factory=lambda: _env_int("WEB_QUERY_MAX_TOKENS_FALLBACK", 12, min_value=1)
    )
    stopword_profile: str = field(
        default_factory=lambda: (os.getenv("WEB_STOPWORD_PROFILE", "standard") or "standard").strip().lower()
    )  # minimal | standard
    recency_soft_accept_when_empty: bool = field(
        default_factory=lambda: _env_bool("WEB_RECENCY_SOFT_ACCEPT_WHEN_EMPTY", False)
    )
    variant_parallel: bool = field(
        default_factory=lambda: _env_bool("WEB_VARIANT_PARALLEL", False)
    )
    variant_max: int = field(
        default_factory=lambda: _env_int("WEB_VARIANT_MAX", 3, min_value=0)
    )
    # Evidence-first rollout flags (no behavior change in PR1)
    evidence_first: bool = field(default_factory=lambda: _env_bool("EVIDENCE_FIRST", False))
    evidence_first_kill_switch: bool = field(default_factory=lambda: _env_bool("EVIDENCE_FIRST_KILL_SWITCH", True))
    ef_degrade_enable: bool = field(default_factory=lambda: _env_bool("EF_DEGRADE", False))
    # Determinism (PR7)
    run_id: str = field(default_factory=lambda: (os.getenv("WEB_RUN_ID") or uuid.uuid4().hex))
    seed: int = field(default_factory=lambda: (int(os.getenv("WEB_RUN_SEED")) if (os.getenv("WEB_RUN_SEED") not in (None, "")) else random.SystemRandom().randint(1, 2**31 - 1)))

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
        cfg.tooling.context_cap = _env_int("TOOL_CONTEXT_MAX_CHARS", 12000)
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
