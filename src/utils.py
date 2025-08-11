"""
Utility functions for Ollama Turbo CLI.
"""

import logging
import re
import sys
from typing import Optional


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
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError, RetryableError) as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = backoff_factor ** attempt
                    logging.debug(f"Retry attempt {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                except Exception as e:
                    # Non-retryable errors
                    raise e
            raise last_exception
        return wrapper
    return decorator
