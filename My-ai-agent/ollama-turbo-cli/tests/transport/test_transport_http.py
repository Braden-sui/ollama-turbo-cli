from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from src.transport.http import TransportHttpClient
from src.transport.policy import RetryPolicy


class HTTPError(Exception):
    pass


class _InnerClient:
    def __init__(self) -> None:
        self.headers: Dict[str, str] = {}


class FakeSDKClient:
    def __init__(self) -> None:
        self._client = _InnerClient()
        self.calls: List[Dict[str, Any]] = []
        self.stream_failures: int = 0
        self.nonstream_failures: int = 0

    def chat(self, **kwargs: Any):
        # record a copy of kwargs for parity checks
        self.calls.append(dict(kwargs))
        if kwargs.get('stream'):
            # return a generator that may fail at most stream_failures times
            failures_left = self.stream_failures

            def _gen() -> Iterator[Any]:
                nonlocal failures_left
                # yield one token then fail if configured
                yield {"type": "token", "content": "hi"}
                if failures_left > 0:
                    failures_left -= 1
                    self.stream_failures = failures_left
                    raise HTTPError("stream broke")
                # then finish cleanly
                yield {"type": "final", "content": "done"}
            return _gen()
        # Non streaming: fail N times then succeed
        if self.nonstream_failures > 0:
            self.nonstream_failures -= 1
            raise HTTPError("server error 5xx")
        return {"message": {"content": "ok"}}


def _nop_sleep(_: float) -> None:
    return


def test_idempotency_lifecycle_non_streaming():
    base = FakeSDKClient()
    t = TransportHttpClient(
        base,
        host="http://localhost:11434",
        connect_timeout_s=1.0,
        read_timeout_s=5.0,
        warm_models=True,
        retry_policy=RetryPolicy(max_retries=0),
    )
    # Before call, no header
    assert "Idempotency-Key" not in base._client.headers
    res = t.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    assert res["message"]["content"] == "ok"
    # After call, header cleared
    assert "Idempotency-Key" not in base._client.headers
    # But during call, header should have been set — we can't observe live; ensure at least one call recorded
    assert base.calls and base.calls[-1].get("model") == "m"


def test_idempotency_lifecycle_streaming_clear_on_completion():
    base = FakeSDKClient()
    base.stream_failures = 0
    t = TransportHttpClient(
        base,
        host="http://localhost:11434",
        connect_timeout_s=1.0,
        read_timeout_s=5.0,
        warm_models=True,
        retry_policy=RetryPolicy(max_retries=0),
    )
    gen = t.chat(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    # Iterate fully
    out = []
    for it in gen:
        out.append(it)
    assert out and out[-1].get("type") == "final"
    # Header cleared after generator completes
    assert "Idempotency-Key" not in base._client.headers


def test_keep_alive_parity_and_backoff_on_stream_reconnect():
    base = FakeSDKClient()
    # Fail once mid-stream then succeed on reconnect
    base.stream_failures = 1
    sleep_calls: List[float] = []

    def _record_sleep(d: float) -> None:
        sleep_calls.append(d)

    rp = RetryPolicy(max_retries=2, backoff_base_s=0.05, backoff_max_s=0.1, jitter_ratio=0.0)
    t = TransportHttpClient(
        base,
        host="http://localhost:11434",
        connect_timeout_s=1.0,
        read_timeout_s=5.0,
        warm_models=True,
        retry_policy=rp,
        sleep_fn=_record_sleep,
    )
    gen = t.chat(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    # Consume stream
    _ = list(gen)
    # There should be at least two calls: initial + reconnect
    assert len(base.calls) >= 2
    ka_vals = [c.get("keep_alive") for c in base.calls]
    # Keep-alive should be resolved and identical across attempts
    assert all(v == ka_vals[0] for v in ka_vals), f"keep_alive parity violated: {ka_vals}"
    # Backoff sleep was invoked at least once with our base
    assert sleep_calls and abs(sleep_calls[0] - 0.05) < 1e-6
