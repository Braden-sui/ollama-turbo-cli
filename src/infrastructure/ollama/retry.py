"""
Ollama retry logic - Infrastructure component for handling request failures.
Implements exponential backoff and circuit breaker patterns.
"""

from __future__ import annotations
import logging
import random
import time
import os
from typing import Any, Dict, List, Optional, Iterator
from dataclasses import dataclass

import ollama
from ollama import Client


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.1
    retryable_status_codes: List[int] = None
    
    def __post_init__(self):
        if self.retryable_status_codes is None:
            # HTTP 5xx server errors, 408 timeout, 429 rate limit
            self.retryable_status_codes = [500, 502, 503, 504, 508, 429, 408]


class RetryableOllamaClient:
    """Wrapper for Ollama client with retry logic and resilience patterns."""
    
    def __init__(
        self,
        client: Client,
        config: Optional[RetryConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self._client = client
        self._config = config or self._load_config_from_env()
        self._logger = logger or logging.getLogger(__name__)
    
    def _load_config_from_env(self) -> RetryConfig:
        """Load retry configuration from environment variables."""
        try:
            max_retries = int(os.getenv('CLI_MAX_RETRIES', '3'))
            base_delay = float(os.getenv('CLI_RETRY_BACKOFF_BASE', '1.0'))
            jitter = float(os.getenv('CLI_RETRY_JITTER_MAX', '0.1'))
            
            # Parse retryable status codes
            status_codes_str = os.getenv('CLI_RETRYABLE_STATUS_CODES', '500,502,503,504,429,408')
            status_codes = [int(code.strip()) for code in status_codes_str.split(',')]
            
            return RetryConfig(
                max_retries=max_retries,
                base_delay=base_delay,
                jitter=jitter,
                retryable_status_codes=status_codes
            )
        except (ValueError, TypeError):
            self._logger.warning("Failed to parse retry config from env, using defaults")
            return RetryConfig()
    
    def chat(self, **kwargs) -> Any:
        """Execute chat request with retry logic."""
        return self._execute_with_retry(lambda: self._client.chat(**kwargs))
    
    def chat_stream(self, **kwargs) -> Iterator[Any]:
        """Execute streaming chat request with retry and fallback logic."""
        try:
            # Try streaming first
            for chunk in self._client.chat(**kwargs):
                yield chunk
        except Exception as e:
            self._logger.debug(f"Streaming failed, falling back to non-streaming: {e}")
            
            # Fallback to non-streaming
            try:
                kwargs['stream'] = False
                response = self._execute_with_retry(lambda: self._client.chat(**kwargs))
                
                # Simulate streaming by yielding the complete response
                if hasattr(response, 'message') and response.message:
                    content = response.message.get('content', '')
                elif isinstance(response, dict):
                    content = response.get('message', {}).get('content', '')
                else:
                    content = str(response)
                
                # Yield in small chunks to simulate streaming
                chunk_size = 50
                for i in range(0, len(content), chunk_size):
                    yield content[i:i + chunk_size]
                    time.sleep(0.01)  # Small delay to simulate streaming
                    
            except Exception as fallback_error:
                # Log at DEBUG to avoid leaking errors to CLI output
                self._logger.debug(f"Both streaming and fallback failed: {fallback_error}")
                # Do not yield error text into the stream; raise to allow caller to handle fallback
                raise fallback_error
    
    def _execute_with_retry(self, operation: callable) -> Any:
        """Execute operation with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(self._config.max_retries + 1):
            try:
                return operation()
                
            except Exception as e:
                last_exception = e
                
                # Log attempt
                if attempt < self._config.max_retries:
                    self._logger.debug(f"Attempt {attempt + 1} failed: {e}")
                else:
                    self._logger.error(f"Final attempt {attempt + 1} failed: {e}")
                
                # Check if we should retry
                if not self._should_retry(e, attempt):
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self._config.base_delay * (2 ** attempt),
                    self._config.max_delay
                )
                
                # Add jitter to prevent thundering herd
                jitter = random.uniform(0, self._config.jitter * delay)
                total_delay = delay + jitter
                
                self._logger.debug(f"Retrying in {total_delay:.2f}s...")
                time.sleep(total_delay)
        
        # All retries exhausted
        raise last_exception or Exception("Retry logic failed")
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if the exception warrants a retry."""
        # Don't retry if we've reached max attempts
        if attempt >= self._config.max_retries:
            return False
        
        # Check for HTTP status code errors (direct attribute)
        if hasattr(exception, 'status_code'):
            try:
                code = int(getattr(exception, 'status_code'))
                if code in self._config.retryable_status_codes:
                    return True
            except Exception:
                pass
        
        # Try to extract status code from known shapes
        try:
            code = self._extract_status_code(exception)
            if code is not None and code in self._config.retryable_status_codes:
                return True
        except Exception:
            pass
        
        # Check for specific exception types that are retryable
        retryable_exceptions = (
            ConnectionError,
            TimeoutError,
            ollama.ResponseError,
        )
        
        # Handle different ollama exception types
        if hasattr(ollama, 'RequestError'):
            retryable_exceptions += (ollama.RequestError,)
        
        return isinstance(exception, retryable_exceptions)
    
    def _extract_status_code(self, exception: Exception) -> Optional[int]:
        """Extract HTTP status code from exception if available."""
        # Try different ways to get status code
        for attr_name in ['status_code', 'code', 'response_code']:
            if hasattr(exception, attr_name):
                try:
                    return int(getattr(exception, attr_name))
                except (ValueError, TypeError):
                    continue
        
        # Try to extract from response object
        if hasattr(exception, 'response') and exception.response:
            response = exception.response
            if hasattr(response, 'status_code'):
                try:
                    return int(response.status_code)
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get statistics about retry behavior (for monitoring)."""
        return {
            "max_retries": self._config.max_retries,
            "base_delay": self._config.base_delay,
            "max_delay": self._config.max_delay,
            "jitter": self._config.jitter,
            "retryable_status_codes": self._config.retryable_status_codes
        }
