from __future__ import annotations

"""
HTTP transport wrapper around the SDK's client.

Owns retries, timeouts, idempotency, keep-alive resolution, and streaming reconnection.
This keeps the higher-level client free from transport micromanagement.
"""

from typing import Any, Dict, Iterable, Iterator, Optional, Callable, Union
import time
import uuid

from . import networking as _net
from .policy import RetryPolicy


class TransportHttpClient:
    """
    Thin wrapper around an SDK client that exposes a .chat(**kwargs) method.
    Responsibilities:
      - Set/Clear Idempotency-Key per call (or per user-provided key)
      - Resolve keep_alive consistently (parity across reconnect attempts)
      - Retry transient errors with backoff
      - Reconnect streaming if iteration fails mid-stream
    """

    def __init__(
        self,
        base_client: Any,
        *,
        host: str,
        headers: Optional[Dict[str, str]] = None,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 600.0,
        warm_models: bool = True,
        keep_alive_raw: Optional[Union[str, float, int]] = None,
        retry_policy: Optional[RetryPolicy] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        logger: Optional[Any] = None,
        trace_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.base_client = base_client
        self.host = host
        self.headers = dict(headers or {})
        self.connect_timeout_s = float(connect_timeout_s)
        self.read_timeout_s = float(read_timeout_s)
        self.warm_models = bool(warm_models)
        self.keep_alive_raw = keep_alive_raw
        self.retry_policy = retry_policy or RetryPolicy()
        self.sleep_fn = sleep_fn
        self.logger = logger
        self._trace = trace_hook or (lambda _e: None)

    # ---------------- Internal helpers ----------------
    def _resolve_keep_alive(self) -> Optional[Union[float, str]]:
        try:
            return _net.resolve_keep_alive(
                warm_models=self.warm_models,
                host=self.host,
                keep_alive_raw=self.keep_alive_raw,
                logger=self.logger,
            )
        except Exception:
            return None

    def _call_chat_once(self, kwargs: Dict[str, Any]) -> Any:
        return self.base_client.chat(**kwargs)

    def _retry_loop(self, func: Callable[[], Any]) -> Any:
        attempt = 0
        while True:
            try:
                return func()
            except Exception as e:  # noqa: BLE001 - we evaluate via policy
                if not self.retry_policy.should_retry(e, attempt_index=attempt):
                    raise
                delay = max(0.0, float(self.retry_policy.backoff_seconds(attempt)))
                try:
                    self._trace(f"transport:retry attempt={attempt+1} delay={delay:.2f}s")
                except Exception:
                    pass
                try:
                    self.sleep_fn(delay)
                except Exception:
                    pass
                attempt += 1

    # ---------------- Public API ----------------
    def chat(self, **kwargs: Any) -> Any:
        """
        Delegate to the underlying client's chat, with transport policy applied.

        Behavior:
          - If 'idempotency_key' kwarg provided, use it; else generate a UUID4 per call.
          - If 'keep_alive' not in kwargs, resolve via policy and inject if available.
          - If stream=True, return a generator that yields chunks. If iteration fails
            with a retryable error, we reconnect by reissuing the same request kwargs,
            preserving keep_alive and idempotency semantics.
        """
        # Keep-alive (parity across reconnects): compute once and reuse
        keep_alive = kwargs.get('keep_alive', self._resolve_keep_alive())
        if keep_alive is not None:
            kwargs['keep_alive'] = keep_alive

        # Idempotency-Key lifecycle (per call)
        idem = kwargs.pop('idempotency_key', None) or str(uuid.uuid4())
        try:
            _net.set_idempotency_key(self.base_client, idem, trace_hook=self._trace)
        except Exception:
            pass

        stream = bool(kwargs.get('stream', False))
        if not stream:
            try:
                return self._retry_loop(lambda: self._call_chat_once(kwargs))
            finally:
                try:
                    _net.clear_idempotency_key(getattr(self, 'base_client', None))
                except Exception:
                    pass

        # Streaming: wrap iterator with reconnection
        def _iter_with_reconnect(initial_kwargs: Dict[str, Any]) -> Iterator[Any]:
            attempts = 0
            local_kwargs = dict(initial_kwargs)
            try:
                while True:
                    # Always reuse identical kwargs (keep_alive parity)
                    resp = self._retry_loop(lambda: self._call_chat_once(local_kwargs))
                    try:
                        for item in resp:
                            yield item
                        # Normal completion
                        break
                    except Exception as e:  # noqa: BLE001
                        # Evaluate for reconnect
                        if not self.retry_policy.should_retry(e, attempt_index=attempts):
                            raise
                        delay = max(0.0, float(self.retry_policy.backoff_seconds(attempts)))
                        try:
                            self._trace(f"transport:stream:reconnect attempt={attempts+1} delay={delay:.2f}s")
                        except Exception:
                            pass
                        try:
                            self.sleep_fn(delay)
                        except Exception:
                            pass
                        attempts += 1
                        continue
                # end while
            finally:
                try:
                    _net.clear_idempotency_key(getattr(self, 'base_client', None))
                except Exception:
                    pass
        return _iter_with_reconnect(kwargs)
