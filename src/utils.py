"""
Utility functions for Ollama Turbo CLI.
"""

import logging
import re
import sys
from typing import Optional
import os
import random


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )
    
    # Reduce noise from third-party libraries unless debugging
    if level.upper() != "DEBUG":
        for name in ("httpx", "ollama", "urllib3", "requests"):
            logging.getLogger(name).setLevel(logging.WARNING)


def validate_api_key(api_key: str) -> bool:
    """Validate Ollama API key format."""
    if not api_key:
        return False
    
    # Ollama API keys should be valid strings
    if len(api_key) < 10:
        return False
    
    return True


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def format_conversation_history(history: list, max_entries: int = 10) -> str:
    """Format conversation history for display."""
    if not history:
        return "No conversation history"
    
    result = []
    recent_history = history[-max_entries:] if len(history) > max_entries else history
    
    for i, message in enumerate(recent_history, 1):
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        # Truncate long messages
        display_content = truncate_text(content, 150)
        
        # Format role with emoji
        role_emoji = {
            'user': '👤',
            'assistant': '🤖',
            'tool': '🔧',
            'system': '⚙️'
        }.get(role, '❓')
        
        result.append(f"  {i}. {role_emoji} {role}: {display_content}")
    
    if len(history) > max_entries:
        result.insert(0, f"  ... (showing last {max_entries} of {len(history)} messages)")
    
    return "\n".join(result)


def safe_filename(text: str) -> str:
    """Convert text to safe filename."""
    # Remove invalid characters
    safe_text = re.sub(r'[<>:"/\\|?*]', '_', text)
    # Limit length
    safe_text = safe_text[:100]
    return safe_text.strip()


# Error handling classes and decorators
class RetryableError(Exception):
    """Exception for errors that can be retried."""
    pass


class OllamaAPIError(Exception):
    """Exception for Ollama API errors."""
    pass


import time
from typing import Any, Callable


def with_retry(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for retry logic with exponential backoff and jitter, env-controlled.

    Env vars:
    - CLI_RETRY_ENABLED: true/false (default: true)
    - CLI_MAX_RETRIES: int (default: max_retries)
    - CLI_RETRY_BACKOFF_BASE: float (default: backoff_factor)
    - CLI_RETRY_JITTER_MAX: float seconds (default: 0.4)
    - CLI_RETRYABLE_STATUS_CODES: comma list (default: 502,503,504,408)
    """
    def _is_retryable_http_error(exc: Exception) -> bool:
        # Check for HTTP status codes on exceptions from HTTP libs (e.g., httpx)
        try:
            raw_codes = os.getenv('CLI_RETRYABLE_STATUS_CODES', '502,503,504,408')
            codes = {int(x.strip()) for x in raw_codes.split(',') if x.strip().isdigit()}
        except Exception:
            codes = {502, 503, 504, 408}

        # Prefer response.status_code if present
        resp = getattr(exc, 'response', None)
        if resp is not None:
            status = getattr(resp, 'status_code', None)
            if isinstance(status, int) and status in codes:
                return True

        # Some exceptions may carry a direct status_code attribute
        status_attr = getattr(exc, 'status_code', None)
        if isinstance(status_attr, int) and status_attr in codes:
            return True
        return False

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Resolve env-controlled switches and tuning
            retry_enabled = os.getenv('CLI_RETRY_ENABLED', 'true').strip().lower() != 'false'
            try:
                effective_retries = max(1, int(os.getenv('CLI_MAX_RETRIES', str(max_retries)) or str(max_retries)))
            except Exception:
                effective_retries = max_retries
            try:
                base = float(os.getenv('CLI_RETRY_BACKOFF_BASE', str(backoff_factor)) or str(backoff_factor))
            except Exception:
                base = backoff_factor
            try:
                jitter_max = max(0.0, float(os.getenv('CLI_RETRY_JITTER_MAX', '0.4') or '0.4'))
            except Exception:
                jitter_max = 0.4

            last_exception: Optional[Exception] = None
            for attempt in range(effective_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError, RetryableError) as e:
                    last_exception = e
                    if not retry_enabled or attempt == effective_retries - 1:
                        raise e
                    wait_time = base ** attempt + (random.uniform(0, jitter_max) if jitter_max > 0 else 0)
                    logging.debug(f"Retry attempt {attempt + 1}/{effective_retries} after {wait_time:.2f}s: {e}")
                    time.sleep(wait_time)
                except Exception as e:
                    # Retry HTTP 5xx/408 if detectable from exception
                    if _is_retryable_http_error(e) and retry_enabled and attempt < effective_retries - 1:
                        last_exception = e
                        wait_time = base ** attempt + (random.uniform(0, jitter_max) if jitter_max > 0 else 0)
                        logging.debug(f"Retry attempt {attempt + 1}/{effective_retries} (HTTP) after {wait_time:.2f}s: {e}")
                        time.sleep(wait_time)
                        continue
                    # Non-retryable errors
                    raise e
            if last_exception is not None:
                raise last_exception
            # Should not reach here
            return func(*args, **kwargs)
        return wrapper
    return decorator
