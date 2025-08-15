"""
Circuit breaker implementation for memory operations.
Provides resilience patterns to handle service failures gracefully.
"""

from __future__ import annotations
import logging
import os
import threading
from datetime import datetime
from typing import Optional

from ...domain.models.memory import CircuitBreakerState
from ...domain.interfaces.memory_store import MemoryCircuitBreaker


class Mem0CircuitBreaker(MemoryCircuitBreaker):
    """Circuit breaker implementation for Mem0 service calls."""
    
    def __init__(
        self,
        failure_threshold: Optional[int] = None,
        cooldown_ms: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        self._logger = logger or logging.getLogger(__name__)
        
        # Configuration from environment with defaults
        self._failure_threshold = failure_threshold or int(os.getenv('MEM0_BREAKER_THRESHOLD', '3'))
        self._cooldown_ms = cooldown_ms or int(os.getenv('MEM0_BREAKER_COOLDOWN_MS', '60000'))
        
        # State management with thread safety
        self._lock = threading.RLock()
        self._state = CircuitBreakerState()
        
        # Logging state to prevent spam
        self._tripped_logged = False
        self._recovered_logged = False
    
    def is_request_allowed(self) -> bool:
        """Check if request should be allowed through circuit breaker."""
        with self._lock:
            allowed = self._state.should_allow_request(self._failure_threshold, self._cooldown_ms)
            
            # Log state changes
            if self._state.is_open and not self._tripped_logged:
                self._logger.warning(
                    f"Memory circuit breaker OPEN - blocking requests "
                    f"(failures: {self._state.failure_count}/{self._failure_threshold})"
                )
                self._tripped_logged = True
                self._recovered_logged = False
            
            # Check if we're recovering from open state
            if not self._state.is_open and self._tripped_logged and not self._recovered_logged:
                self._logger.info("Memory circuit breaker CLOSED - allowing requests")
                self._recovered_logged = True
                self._tripped_logged = False
            
            return allowed
    
    def record_success(self) -> None:
        """Record successful operation."""
        with self._lock:
            was_open = self._state.is_open
            self._state.record_success()
            
            # Log recovery if transitioning from open to closed
            if was_open and not self._state.is_open:
                self._logger.info(
                    f"Memory circuit breaker recovered after {self._state.success_count} success(es)"
                )
    
    def record_failure(self, error: Exception) -> None:
        """Record failed operation."""
        with self._lock:
            was_open = self._state.is_open
            self._state.record_failure(self._failure_threshold)
            
            self._logger.debug(f"Memory operation failure recorded: {error}")
            
            # Log when circuit breaker trips
            if not was_open and self._state.is_open:
                self._logger.warning(
                    f"Memory circuit breaker tripped after {self._failure_threshold} failures. "
                    f"Next retry in {self._cooldown_ms}ms"
                )
    
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        with self._lock:
            if self._state.is_open:
                return "OPEN"
            elif self._state.failure_count > 0:
                return "HALF_OPEN"
            else:
                return "CLOSED"
    
    def get_statistics(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self.get_state(),
                "is_open": self._state.is_open,
                "failure_count": self._state.failure_count,
                "success_count": self._state.success_count,
                "failure_threshold": self._failure_threshold,
                "cooldown_ms": self._cooldown_ms,
                "last_failure_time": (
                    self._state.last_failure_time.isoformat() 
                    if self._state.last_failure_time else None
                )
            }
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitBreakerState()
            self._tripped_logged = False
            self._recovered_logged = False
            self._logger.info("Memory circuit breaker reset")
    
    def force_open(self) -> None:
        """Force circuit breaker into open state (for testing/maintenance)."""
        with self._lock:
            self._state.is_open = True
            self._state.failure_count = self._failure_threshold
            self._state.last_failure_time = datetime.now()
            self._logger.warning("Memory circuit breaker forced OPEN")
    
    def force_close(self) -> None:
        """Force circuit breaker into closed state (for testing/recovery)."""
        with self._lock:
            self._state.is_open = False
            self._state.failure_count = 0
            self._state.success_count = 0
            self._logger.info("Memory circuit breaker forced CLOSED")
