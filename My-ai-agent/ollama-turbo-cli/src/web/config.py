from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WebConfig:
    # Identity
    user_agent: str = os.getenv("WEB_UA", "ollama-turbo-cli-web/1.0 (+https://github.com)" )

    # Timeouts (seconds)
    timeout_connect: float = float(os.getenv("WEB_TIMEOUT_CONNECT", "5"))
    timeout_read: float = float(os.getenv("WEB_TIMEOUT_READ", "15"))
    timeout_write: float = float(os.getenv("WEB_TIMEOUT_WRITE", "10"))

    # Retries
    retry_attempts: int = int(os.getenv("WEB_RETRY_ATTEMPTS", "3"))
    retry_backoff_base: float = float(os.getenv("WEB_RETRY_BACKOFF_BASE", "0.4"))
    retry_backoff_max: float = float(os.getenv("WEB_RETRY_BACKOFF_MAX", "6.0"))

    # Concurrency
    max_connections: int = int(os.getenv("WEB_MAX_CONNECTIONS", "20"))
    max_keepalive: int = int(os.getenv("WEB_MAX_KEEPALIVE", "10"))
    per_host_concurrency: int = int(os.getenv("WEB_PER_HOST_CONCURRENCY", "4"))

    # Fetch behavior
    follow_redirects: bool = os.getenv("WEB_FOLLOW_REDIRECTS", "1") in {"1","true","True"}
    head_gating_enabled: bool = os.getenv("WEB_HEAD_GATING", "1") in {"1","true","True"}
    max_download_bytes: int = int(os.getenv("WEB_MAX_DOWNLOAD_BYTES", "10485760"))  # 10 MB
    accept_header_override: str = os.getenv("WEB_ACCEPT_HEADER", "")
    client_pool_size: int = int(os.getenv("WEB_CLIENT_POOL_SIZE", "16"))

    # Caching and robots
    cache_ttl_seconds: int = int(os.getenv("WEB_CACHE_TTL_SECONDS", "86400"))  # 24h
    robots_ttl_seconds: int = int(os.getenv("WEB_ROBOTS_TTL_SECONDS", "3600"))
    max_crawl_delay_s: int = int(os.getenv("WEB_MAX_CRAWL_DELAY_S", "20"))

    # Provider keys
    brave_key: Optional[str] = os.getenv("BRAVE_API_KEY")
    tavily_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    exa_key: Optional[str] = os.getenv("EXA_API_KEY")
    google_pse_cx: Optional[str] = os.getenv("GOOGLE_PSE_CX")
    google_pse_key: Optional[str] = os.getenv("GOOGLE_PSE_KEY")

    # Rerank
    cohere_key: Optional[str] = os.getenv("COHERE_API_KEY")
    voyage_key: Optional[str] = os.getenv("VOYAGE_API_KEY")

    # Policies
    respect_robots: bool = os.getenv("WEB_RESPECT_ROBOTS", "1") in {"1","true","True"}
    allow_browser: bool = os.getenv("WEB_ALLOW_BROWSER", "1") in {"1","true","True"}

    # Rate limiting
    rate_tokens_per_host: int = int(os.getenv("WEB_RATE_TOKENS_PER_HOST", "4"))
    rate_refill_per_sec: float = float(os.getenv("WEB_RATE_REFILL_PER_SEC", "0.5"))
    respect_retry_after: bool = os.getenv("WEB_RESPECT_RETRY_AFTER", "1") in {"1","true","True"}

    # Allowlist integration (reuse sandbox policy)
    sandbox_allow: Optional[str] = os.getenv("SANDBOX_NET_ALLOW", "")
    sandbox_allow_http: bool = os.getenv("SANDBOX_ALLOW_HTTP", "0") in {"1","true","True"}
    sandbox_allow_proxies: bool = os.getenv("SANDBOX_ALLOW_PROXIES", "0") in {"1","true","True"}

    # Storage locations
    cache_root: str = os.getenv("WEB_CACHE_ROOT", ".sandbox/webcache")
    archive_enabled: bool = os.getenv("WEB_ARCHIVE_ENABLED", "1") in {"1","true","True"}
    archive_check_memento_first: bool = os.getenv("WEB_ARCHIVE_CHECK_FIRST", "0") in {"1","true","True"}
    archive_retry_on_429: bool = os.getenv("WEB_ARCHIVE_RETRY_ON_429", "1") in {"1","true","True"}

    # Browser limits
    browser_max_pages: int = int(os.getenv("WEB_BROWSER_MAX_PAGES", "10"))
    browser_wait_ms: int = int(os.getenv("WEB_BROWSER_WAIT_MS", "1200"))
    browser_block_resources: str = os.getenv("WEB_BROWSER_BLOCK_RESOURCES", "image,font,media")
    browser_stealth_light: bool = os.getenv("WEB_BROWSER_STEALTH_LIGHT", "0") in {"1","true","True"}

    # Sitemaps
    sitemap_enabled: bool = os.getenv("WEB_SITEMAP_ENABLED", "0") in {"1","true","True"}
    sitemap_max_urls: int = int(os.getenv("WEB_SITEMAP_MAX_URLS", "50"))
    sitemap_timeout_s: float = float(os.getenv("WEB_SITEMAP_TIMEOUT_S", "5"))
    sitemap_include_subs: bool = os.getenv("WEB_SITEMAP_INCLUDE_SUBS", "1") in {"1","true","True"}

    def as_headers(self) -> dict:
        return {"User-Agent": self.user_agent}

    def __post_init__(self) -> None:
        """Re-read environment at instantiation time so tests that tweak env are respected.
        Because the dataclass is frozen, we assign via object.__setattr__.
        """
        e = os.getenv
        # Identity
        object.__setattr__(self, 'user_agent', e("WEB_UA", "ollama-turbo-cli-web/1.0 (+https://github.com)"))
        # Timeouts
        object.__setattr__(self, 'timeout_connect', float(e("WEB_TIMEOUT_CONNECT", "5")))
        object.__setattr__(self, 'timeout_read', float(e("WEB_TIMEOUT_READ", "15")))
        object.__setattr__(self, 'timeout_write', float(e("WEB_TIMEOUT_WRITE", "10")))
        # Retries
        object.__setattr__(self, 'retry_attempts', int(e("WEB_RETRY_ATTEMPTS", "3")))
        object.__setattr__(self, 'retry_backoff_base', float(e("WEB_RETRY_BACKOFF_BASE", "0.4")))
        object.__setattr__(self, 'retry_backoff_max', float(e("WEB_RETRY_BACKOFF_MAX", "6.0")))
        # Concurrency
        object.__setattr__(self, 'max_connections', int(e("WEB_MAX_CONNECTIONS", "20")))
        object.__setattr__(self, 'max_keepalive', int(e("WEB_MAX_KEEPALIVE", "10")))
        object.__setattr__(self, 'per_host_concurrency', int(e("WEB_PER_HOST_CONCURRENCY", "4")))
        # Fetch behavior
        object.__setattr__(self, 'follow_redirects', e("WEB_FOLLOW_REDIRECTS", "1") in {"1","true","True"})
        object.__setattr__(self, 'head_gating_enabled', e("WEB_HEAD_GATING", "1") in {"1","true","True"})
        object.__setattr__(self, 'max_download_bytes', int(e("WEB_MAX_DOWNLOAD_BYTES", "10485760")))
        object.__setattr__(self, 'accept_header_override', e("WEB_ACCEPT_HEADER", ""))
        object.__setattr__(self, 'client_pool_size', int(e("WEB_CLIENT_POOL_SIZE", "16")))
        # Caching and robots
        object.__setattr__(self, 'cache_ttl_seconds', int(e("WEB_CACHE_TTL_SECONDS", "86400")))
        object.__setattr__(self, 'robots_ttl_seconds', int(e("WEB_ROBOTS_TTL_SECONDS", "3600")))
        object.__setattr__(self, 'max_crawl_delay_s', int(e("WEB_MAX_CRAWL_DELAY_S", "20")))
        # Provider keys
        object.__setattr__(self, 'brave_key', e("BRAVE_API_KEY"))
        object.__setattr__(self, 'tavily_key', e("TAVILY_API_KEY"))
        object.__setattr__(self, 'exa_key', e("EXA_API_KEY"))
        object.__setattr__(self, 'google_pse_cx', e("GOOGLE_PSE_CX"))
        object.__setattr__(self, 'google_pse_key', e("GOOGLE_PSE_KEY"))
        # Rerank
        object.__setattr__(self, 'cohere_key', e("COHERE_API_KEY"))
        object.__setattr__(self, 'voyage_key', e("VOYAGE_API_KEY"))
        # Policies
        object.__setattr__(self, 'respect_robots', e("WEB_RESPECT_ROBOTS", "1") in {"1","true","True"})
        object.__setattr__(self, 'allow_browser', e("WEB_ALLOW_BROWSER", "1") in {"1","true","True"})
        # Rate limiting
        object.__setattr__(self, 'rate_tokens_per_host', int(e("WEB_RATE_TOKENS_PER_HOST", "4")))
        object.__setattr__(self, 'rate_refill_per_sec', float(e("WEB_RATE_REFILL_PER_SEC", "0.5")))
        object.__setattr__(self, 'respect_retry_after', e("WEB_RESPECT_RETRY_AFTER", "1") in {"1","true","True"})
        # Allowlist
        object.__setattr__(self, 'sandbox_allow', e("SANDBOX_NET_ALLOW", ""))
        object.__setattr__(self, 'sandbox_allow_http', e("SANDBOX_ALLOW_HTTP", "0") in {"1","true","True"})
        object.__setattr__(self, 'sandbox_allow_proxies', e("SANDBOX_ALLOW_PROXIES", "0") in {"1","true","True"})
        # Storage and archiving
        object.__setattr__(self, 'cache_root', e("WEB_CACHE_ROOT", ".sandbox/webcache"))
        object.__setattr__(self, 'archive_enabled', e("WEB_ARCHIVE_ENABLED", "1") in {"1","true","True"})
        object.__setattr__(self, 'archive_check_memento_first', e("WEB_ARCHIVE_CHECK_FIRST", "0") in {"1","true","True"})
        object.__setattr__(self, 'archive_retry_on_429', e("WEB_ARCHIVE_RETRY_ON_429", "1") in {"1","true","True"})
        # Browser limits
        object.__setattr__(self, 'browser_max_pages', int(e("WEB_BROWSER_MAX_PAGES", "10")))
        object.__setattr__(self, 'browser_wait_ms', int(e("WEB_BROWSER_WAIT_MS", "1200")))
        object.__setattr__(self, 'browser_block_resources', e("WEB_BROWSER_BLOCK_RESOURCES", "image,font,media"))
        object.__setattr__(self, 'browser_stealth_light', e("WEB_BROWSER_STEALTH_LIGHT", "0") in {"1","true","True"})
        # Sitemaps
        object.__setattr__(self, 'sitemap_enabled', e("WEB_SITEMAP_ENABLED", "0") in {"1","true","True"})
        object.__setattr__(self, 'sitemap_max_urls', int(e("WEB_SITEMAP_MAX_URLS", "50")))
        object.__setattr__(self, 'sitemap_timeout_s', float(e("WEB_SITEMAP_TIMEOUT_S", "5")))
        object.__setattr__(self, 'sitemap_include_subs', e("WEB_SITEMAP_INCLUDE_SUBS", "1") in {"1","true","True"})
