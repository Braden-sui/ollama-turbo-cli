import json
import types
import requests
import pytest

from src import plugin_loader


@pytest.fixture(autouse=True)
def _reload_plugins():
    plugin_loader.reload_plugins()
    yield


def _get_rc_fn():
    fns = plugin_loader.TOOL_FUNCTIONS
    assert 'reliable_chat' in fns, "reliable_chat tool not discovered"
    return fns['reliable_chat']


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None):
        self.status_code = status_code
        self._json = json_body or {"content": "ok"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200):
        self._lines = list(lines)
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def iter_lines(self, decode_unicode=True):
        for l in self._lines:
            yield l


def test_reliable_chat_non_stream_success(monkeypatch):
    calls = {}

    def fake_post(url, headers=None, data=None, timeout=None, stream=False):
        calls['url'] = url
        calls['data'] = data
        assert stream is False
        assert url.endswith('/v1/chat')
        body = json.loads(data)
        assert 'message' in body
        return _FakeResponse(200, {"content": "Hello"})

    monkeypatch.setattr(requests, 'post', fake_post)
    fn = _get_rc_fn()
    out_s = fn(message="hi", stream=False)
    out = json.loads(out_s)
    assert out.get('ok') is True
    assert out.get('tool') == 'reliable_chat'
    assert out.get('content') == 'Hello'
    assert isinstance(out.get('summary'), dict)


def test_reliable_chat_stream_success(monkeypatch):
    # Simulate SSE: two tokens, final, then trailing summary event
    sse_lines = [
        # token 1
        "data: {\"type\": \"token\", \"content\": \"Hel\"}",
        # token 2
        "data: {\"type\": \"token\", \"content\": \"lo\"}",
        # final event
        "data: {\"type\": \"final\", \"content\": \"Hello\"}",
        # summary trailer
        "event: summary",
        "data: {\"grounded\": true, \"citations\": [], \"validator\": null, \"consensus\": {\"k\": 1, \"agree_rate\": 1.0}}",
    ]

    def fake_post(url, headers=None, data=None, timeout=None, stream=False):
        assert stream is True
        assert url.endswith('/v1/chat/stream')
        return _FakeStreamResponse(sse_lines, status_code=200)

    monkeypatch.setattr(requests, 'post', fake_post)
    fn = _get_rc_fn()
    out_s = fn(message="hi", stream=True)
    out = json.loads(out_s)
    assert out.get('ok') is True
    assert out.get('content') == 'Hello'
    summ = out.get('summary')
    assert isinstance(summ, dict)
    assert 'consensus' in summ


def test_reliable_chat_timeout(monkeypatch):
    def fake_post(url, headers=None, data=None, timeout=None, stream=False):
        raise requests.exceptions.Timeout("sim timeout")

    monkeypatch.setattr(requests, 'post', fake_post)
    fn = _get_rc_fn()
    out = json.loads(fn(message="hi", stream=False, timeout_s=1))
    assert out.get('ok') is False
    err = out.get('error') or {}
    assert err.get('code') == 'timeout'


def test_reliable_chat_http_error(monkeypatch):
    def fake_post(url, headers=None, data=None, timeout=None, stream=False):
        # context manager not used for non-stream
        return _FakeResponse(status_code=500)

    monkeypatch.setattr(requests, 'post', fake_post)
    fn = _get_rc_fn()
    out = json.loads(fn(message="hi", stream=False))
    assert out.get('ok') is False
    err = out.get('error') or {}
    assert err.get('code') == 'http_error'


def test_reliable_chat_auto_flags_heuristics(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, data=None, timeout=None, stream=False):
        captured['url'] = url
        captured['headers'] = headers
        captured['data'] = json.loads(data)
        # Return minimal ok
        if stream:
            return _FakeStreamResponse([
                "data: {\"type\": \"final\", \"content\": \"X\"}",
                "event: summary",
                "data: {\"grounded\": true, \"citations\": [1], \"validator\": {\"ok\": true}, \"consensus\": {\"k\": 3, \"agree_rate\": 1.0}}",
            ], status_code=200)
        return _FakeResponse(200, {"content": "X"})

    monkeypatch.setattr(requests, 'post', fake_post)
    fn = _get_rc_fn()
    msg = "Please cite sources and verify. Aim for consensus, k=4."
    _ = json.loads(fn(message=msg, stream=False))
    body = captured['data']
    # Heuristics should have enabled these flags in the payload
    assert body.get('ground') is True
    assert body.get('cite') is True
    assert body.get('check') in ('warn', 'strict')
    assert body.get('consensus') is True
    assert body.get('k') == 4
