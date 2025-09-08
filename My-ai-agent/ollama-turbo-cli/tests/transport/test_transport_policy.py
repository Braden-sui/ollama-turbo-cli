from __future__ import annotations

import types
import builtins

from src.transport.policy import RetryPolicy, _default_exception_classifier as classify
from src.utils import RetryableError  # present in repo


def test_retry_policy_backoff_deterministic():
    rp = RetryPolicy(max_retries=3, backoff_base_s=0.1, backoff_max_s=0.5, jitter_ratio=0.0)
    # attempt_index 0..n
    assert rp.backoff_seconds(0) == 0.1
    assert rp.backoff_seconds(1) == 0.2
    assert rp.backoff_seconds(2) == 0.4
    # capped at max 0.5
    assert rp.backoff_seconds(3) == 0.5


def test_default_classifier_common_cases():
    # Explicit project retryable
    assert classify(RetryableError("transient")) is True
    # Builtin timeout
    assert classify(TimeoutError("timeout")) is True

    # Exceptions by name
    class ConnectionError(Exception):
        pass
    assert classify(ConnectionError("conn reset")) is True

    # OSError-like message matching
    assert classify(OSError("timed out")) is True

    # Non-retryable by default
    class ValueErrorCustom(ValueError):
        pass
    assert classify(ValueErrorCustom("bad value")) is False


def test_retry_policy_should_retry_respects_max():
    rp = RetryPolicy(max_retries=2, jitter_ratio=0.0)
    # attempt_index is zero-based; indices 0 and 1 should be allowed, 2 denied
    assert rp.should_retry(RetryableError("x"), attempt_index=0) is True
    assert rp.should_retry(RetryableError("x"), attempt_index=1) is True
    assert rp.should_retry(RetryableError("x"), attempt_index=2) is False
