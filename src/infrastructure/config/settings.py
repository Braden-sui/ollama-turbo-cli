"""
Configuration settings - Infrastructure component for managing application configuration.
Uses Pydantic for validation and environment variable loading.
"""

from __future__ import annotations
import os
from typing import Optional, List, Dict, Any

# Pydantic v1/v2 compatibility:
# - In v2, BaseSettings moved to 'pydantic-settings'
# - 'validator' became 'field_validator'
_USING_MINIMAL_SETTINGS = False
try:  # Prefer Pydantic v2 style
    from pydantic_settings import BaseSettings  # type: ignore
    from pydantic import Field  # type: ignore
    try:
        from pydantic import field_validator as _validator  # type: ignore
    except Exception:  # pragma: no cover - older v2 builds
        _validator = None  # type: ignore
except Exception:
    try:  # Fallback to Pydantic v1
        from pydantic import BaseSettings, Field, validator as _validator  # type: ignore
    except Exception:
        # Minimal no-deps fallback for environments without pydantic
        _USING_MINIMAL_SETTINGS = True
        class BaseSettings:  # type: ignore
            pass
        def Field(default=None, *_, **__):  # type: ignore
            return default
        _validator = None  # type: ignore

# Uniform validator decorator across pydantic versions
if _validator is None:
    def validator(*args, **kwargs):  # pragma: no cover - unlikely path
        def _noop(fn):
            return fn
        return _noop
else:
    def validator(*args, **kwargs):  # type: ignore[misc]
        # Map v1 'pre=True' to v2 'mode="before"' when using field_validator
        if getattr(_validator, "__name__", "") == "field_validator":
            pre = bool(kwargs.pop("pre", False))
            mode = "before" if pre else "after"
            return _validator(*args, mode=mode, **kwargs)  # type: ignore
        # v1 path
        return _validator(*args, **kwargs)  # type: ignore


class OllamaSettings(BaseSettings):
    """Ollama service configuration."""
    
    api_key: str = Field(..., env='OLLAMA_API_KEY')
    host: str = Field('https://ollama.com', env='OLLAMA_HOST')
    model: str = Field('gpt-oss:120b', env='OLLAMA_MODEL')
    keep_alive: bool = Field(True, env='OLLAMA_KEEP_ALIVE')
    
    # Generation settings
    max_output_tokens: Optional[int] = Field(None, env='OLLAMA_MAX_OUTPUT_TOKENS')
    ctx_size: Optional[int] = Field(None, env='OLLAMA_CTX_SIZE')
    
    class Config:
        env_file = '.env'
        case_sensitive = False


class RetrySettings(BaseSettings):
    """Retry and resilience configuration."""
    
    enabled: bool = Field(True, env='CLI_RETRY_ENABLED')
    max_retries: int = Field(3, env='CLI_MAX_RETRIES')
    backoff_base: float = Field(1.0, env='CLI_RETRY_BACKOFF_BASE')
    jitter_max: float = Field(0.1, env='CLI_RETRY_JITTER_MAX')
    
    # Timeout settings
    connect_timeout_s: float = Field(5.0, env='CLI_CONNECT_TIMEOUT_S')
    read_timeout_s: float = Field(600.0, env='CLI_READ_TIMEOUT_S')
    stream_idle_reconnect_secs: int = Field(90, env='CLI_STREAM_IDLE_RECONNECT_SECS')
    
    # Retryable HTTP status codes
    retryable_status_codes: str = Field(
        '500,502,503,504,429,408',
        env='CLI_RETRYABLE_STATUS_CODES'
    )
    
    @validator('retryable_status_codes')
    def parse_status_codes(cls, v):
        """Parse comma-separated status codes into list."""
        try:
            return [int(code.strip()) for code in v.split(',')]
        except ValueError:
            return [500, 502, 503, 504, 429, 408]
    
    class Config:
        env_file = '.env'


class MemorySettings(BaseSettings):
    """Memory service configuration."""
    
    api_key: Optional[str] = Field(None, env='MEM0_API_KEY')
    user_id: str = Field('default-user', env='MEM0_USER_ID')
    
    # Search settings
    max_hits: int = Field(3, env='MEM0_MAX_HITS')
    search_timeout_ms: int = Field(200, env='MEM0_SEARCH_TIMEOUT_MS')
    search_workers: int = Field(2, env='MEM0_SEARCH_WORKERS')
    
    # Connection timeouts
    timeout_connect_ms: int = Field(1000, env='MEM0_TIMEOUT_CONNECT_MS')
    timeout_read_ms: int = Field(2000, env='MEM0_TIMEOUT_READ_MS')
    
    # Background processing
    add_queue_max: int = Field(256, env='MEM0_ADD_QUEUE_MAX')
    
    # Circuit breaker
    breaker_threshold: int = Field(3, env='MEM0_BREAKER_THRESHOLD')
    breaker_cooldown_ms: int = Field(60000, env='MEM0_BREAKER_COOLDOWN_MS')
    
    class Config:
        env_file = '.env'


class ConversationSettings(BaseSettings):
    """Conversation management configuration."""
    
    max_history: int = Field(10, env='MAX_CONVERSATION_HISTORY')
    reasoning: str = Field('high', env='REASONING_LEVEL')
    
    @validator('max_history')
    def validate_max_history(cls, v):
        """Ensure max history is reasonable."""
        return max(2, min(v, 10))
    
    @validator('reasoning')
    def validate_reasoning(cls, v):
        """Ensure reasoning level is valid."""
        if v.lower() not in ['low', 'medium', 'high']:
            return 'high'
        return v.lower()
    
    class Config:
        env_file = '.env'


class ToolSettings(BaseSettings):
    """Tool system configuration."""
    
    enabled: bool = Field(True, env='TOOLS_ENABLED')
    multi_round_tools: bool = Field(True, env='MULTI_ROUND_TOOLS')
    max_rounds: int = Field(6, env='TOOL_MAX_ROUNDS')
    print_limit: int = Field(200, env='TOOL_PRINT_LIMIT')
    
    # Plugin paths
    plugin_paths: str = Field('', env='OLLAMA_TOOLS_DIR')
    
    @validator('plugin_paths')
    def parse_plugin_paths(cls, v):
        """Parse comma-separated plugin paths."""
        if not v:
            return []
        return [path.strip() for path in v.split(',') if path.strip()]
    
    class Config:
        env_file = '.env'


class AppSettings(BaseSettings):
    """Main application settings."""
    
    # Sub-configurations
    ollama: OllamaSettings = OllamaSettings()
    retry: RetrySettings = RetrySettings()
    memory: MemorySettings = MemorySettings()
    conversation: ConversationSettings = ConversationSettings()
    tools: ToolSettings = ToolSettings()
    
    # CLI settings
    quiet: bool = Field(False, env='CLI_QUIET')
    show_trace: bool = Field(False, env='CLI_SHOW_TRACE')
    
    # Logging
    log_level: str = Field('INFO', env='LOG_LEVEL')
    log_format: str = Field(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        env='LOG_FORMAT'
    )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Ensure log level is valid."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            return 'INFO'
        return v.upper()
    
    class Config:
        env_file = '.env'
        case_sensitive = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for serialization."""
        def _dump(model: Any) -> Dict[str, Any]:
            if hasattr(model, 'dict'):
                return model.dict()  # pydantic v1
            if hasattr(model, 'model_dump'):
                return model.model_dump()  # pydantic v2
            try:
                return dict(model)
            except Exception:
                try:
                    return vars(model)
                except Exception:
                    return {}
        return {
            'ollama': _dump(self.ollama),
            'retry': _dump(self.retry),
            'memory': _dump(self.memory),
            'conversation': _dump(self.conversation),
            'tools': _dump(self.tools),
            'quiet': self.quiet,
            'show_trace': self.show_trace,
            'log_level': self.log_level
        }
    
    @property
    def memory_enabled(self) -> bool:
        """Check if memory service is enabled."""
        # Consider runtime env var even if BaseSettings didn't populate (minimal mode)
        api = getattr(self.memory, 'api_key', None) or os.getenv('MEM0_API_KEY')
        if api is None:
            return False
        # Treat empty/false-like values as disabled
        return str(api).strip().lower() not in ('', '0', 'false')
    
    @property
    def tools_enabled(self) -> bool:
        """Check if tool system is enabled."""
        return self.tools.enabled
    
    def validate_required_settings(self) -> List[str]:
        """Validate required settings and return list of missing ones."""
        missing = []
        
        if not self.ollama.api_key:
            missing.append('OLLAMA_API_KEY')
        
        return missing


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reload_settings() -> AppSettings:
    """Reload settings from environment (for testing)."""
    global _settings
    _settings = AppSettings()
    return _settings

# Backward-compatible export alias expected by some import sites
Settings = AppSettings
