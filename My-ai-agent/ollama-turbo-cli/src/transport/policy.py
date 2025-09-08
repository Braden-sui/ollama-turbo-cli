from __future__ import annotations

"""
Retry/backoff policy for transport.

This module centralizes retry classification and backoff timing so transports can
remain thin and the client does not micromanage transient failures.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import random

try:
    # Prefer the project-local RetryableError if available
    from ..utils import RetryableError  # type: ignore
except Exception:  # pragma: no cover - tests will not rely on import failure
    class RetryableError(Exception):  # type: ignore
        pass


Classifier = Callable[[BaseException], bool]


@dataclass
class RetryPolicy:
    max_retries: int = 2
    backoff_base_s: float = 0.2
    backoff_max_s: float = 6.0
    jitter_ratio: float = 0.2  # +- 20% jitter
    classify_exception: Optional[Classifier] = None

    def should_retry(self, exc: BaseException, attempt_index: int) -> bool:
        """
        Return True when we should retry the given exception.
        attempt_index is zero-based (0 == first retry attempt).
        """
        if attempt_index >= max(0, int(self.max_retries)):
            return False
        cls = self.classify_exception or _default_exception_classifier
        try:
            return bool(cls(exc))
        except Exception:
            return False

    def backoff_seconds(self, attempt_index: int) -> float:
        """
        Exponential backoff with optional jitter; attempt_index is zero-based.
        0 -> base, 1 -> base*2, 2 -> base*4, ... up to max.
        """
        try:
            base = max(0.0, float(self.backoff_base_s))
            cap = max(base, float(self.backoff_max_s))
            expo = base * (2 ** attempt_index)
            dur = min(cap, expo)
            if self.jitter_ratio > 0:
                jitter = max(0.0, float(self.jitter_ratio))
                lo = dur * (1.0 - jitter)
                hi = dur * (1.0 + jitter)
                return random.uniform(lo, hi)
            return dur
        except Exception:
            return max(0.0, float(self.backoff_base_s))


def _default_exception_classifier(exc: BaseException) -> bool:
    """Conservative retry classification for transport-level exceptions."""
    try:
        # Explicit project error used for transient failures
        if isinstance(exc, RetryableError):
            return True
        # Timeouts and connection resets should be retried
        if isinstance(exc, TimeoutError):  # built-in
            return True
        # Common network family — avoid importing requests/httpx types to keep deps light
        net_names = (
            'ConnectionError', 'ConnectTimeout', 'ReadTimeout', 'TimeoutException',
            'HTTPError', 'ChunkedEncodingError', 'ProtocolError', 'RemoteProtocolError',
            'ServerDisconnectedError'
        )
        etype = type(exc).__name__
        if etype in net_names:
            return True
        # Some libraries wrap network issues in OSError; allow ECONNRESET/ETIMEDOUT by message
        msg = str(exc).lower()
        if any(k in msg for k in ('timed out', 'timeout', 'connection reset', 'connection aborted', 'broken pipe')):
            return True
        return False
    except Exception:
        return False
