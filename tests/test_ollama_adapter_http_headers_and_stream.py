import json
import pytest

from src.infrastructure.ollama.client import OllamaAdapter
from src.domain.models.conversation import ConversationContext
from src.infrastructure.ollama.retry import RetryableOllamaClient


@pytest.fixture(autouse=True)
def _patch_ollama_client(monkeypatch):
    """Ensure adapter initialization does not require real networked client."""
    class _DummyClient:
        def __init__(self, host=None, headers=None, **kwargs):
            self.headers = dict(headers or {})
            # Bind an inner structure to mimic httpx.Client headers access
            self._client = type('Inner', (), {'headers': self.headers})()
    monkeypatch.setattr('src.infrastructure.ollama.client.Client', _DummyClient)
    yield


class _FakeRetryClient:
    def __init__(self, adapter, expect_keep_alive=None):
        self.adapter = adapter
        self.expect_keep_alive = expect_keep_alive
        self.seen_kwargs = None
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        self.seen_kwargs = kwargs
        # Idempotency header must be present during call
        assert 'Idempotency-Key' in self.adapter._client.headers
        if self.expect_keep_alive is not None:
            assert kwargs.get('keep_alive') == self.expect_keep_alive
        else:
            assert 'keep_alive' not in kwargs
        return {'message': {'content': 'ok'}}


class _FakeStreamingRetryClient:
    def __init__(self, adapter, expect_keep_alive=None):
        self.adapter = adapter
        self.expect_keep_alive = expect_keep_alive
        self.calls = 0

    def chat_stream(self, **kwargs):
        self.calls += 1
        # Idempotency header must be present for streaming
        assert 'Idempotency-Key' in self.adapter._client.headers
        if self.expect_keep_alive is not None:
            assert kwargs.get('keep_alive') == self.expect_keep_alive
        else:
            assert 'keep_alive' not in kwargs

        # Yield three chunks: cumulative text, cumulative text, tool call
        yield {'message': {'content': 'Hello'}}
        yield {'message': {'content': 'Hello world'}}
        yield {
            'message': {
                'content': '',
                'tool_calls': [
                    {
                        'type': 'function',
                        'id': 'call_a',
                        'function': {
                            'name': 'test_tool',
                            'arguments': '{"x": 1}'
                        }
                    }
                ]
            }
        }


class _FailingClient:
    def chat(self, **kwargs):
        raise RuntimeError('boom')


def test_idempotency_header_and_keep_alive_in_chat(monkeypatch):
    monkeypatch.setenv('OLLAMA_KEEP_ALIVE', '5m')
    adapter = OllamaAdapter(api_key='test', model='gpt-oss:120b')
    fake = _FakeRetryClient(adapter, expect_keep_alive='5m')
    adapter._retry_client = fake  # type: ignore[attr-defined]

    messages = [{'role': 'user', 'content': 'hi'}]
    ctx = ConversationContext(model='gpt-oss:120b')

    res = adapter.chat(messages, ctx)
    assert res.success is True
    assert res.content == 'ok'
    # Header should be cleared after request completes
    assert 'Idempotency-Key' not in adapter._client.headers
    # Ensure messages passed through
    assert fake.seen_kwargs['messages'] == messages
    assert fake.seen_kwargs['model'] == 'gpt-oss:120b'


def test_keep_alive_suppressed_when_false(monkeypatch):
    monkeypatch.setenv('OLLAMA_KEEP_ALIVE', '0')
    adapter = OllamaAdapter(api_key='test', model='gpt-oss:120b')
    fake = _FakeRetryClient(adapter, expect_keep_alive=None)
    adapter._retry_client = fake  # type: ignore[attr-defined]

    messages = [{'role': 'user', 'content': 'hello'}]
    ctx = ConversationContext(model='gpt-oss:120b')

    res = adapter.chat(messages, ctx)
    assert res.success is True
    assert 'Idempotency-Key' not in adapter._client.headers


def test_chat_stream_delta_and_tool_sentinel(monkeypatch):
    monkeypatch.setenv('OLLAMA_KEEP_ALIVE', '1')
    adapter = OllamaAdapter(api_key='test', model='gpt-oss:120b')
    fake = _FakeStreamingRetryClient(adapter, expect_keep_alive='1s')
    adapter._retry_client = fake  # type: ignore[attr-defined]

    messages = [{'role': 'user', 'content': 'stream please'}]
    ctx = ConversationContext(model='gpt-oss:120b', enable_tools=True, stream=True)

    outputs = list(adapter.chat_stream(messages, ctx))
    # Expect first full, then delta, then tool sentinel
    expected_sentinel = '<tool_call>' + json.dumps({
        'name': 'test_tool',
        'arguments': {'x': 1},
        'id': 'call_a',
    }) + '</tool_call>'
    assert outputs == ['Hello', ' world', expected_sentinel]
    # Header should be cleared after stream completes
    assert 'Idempotency-Key' not in adapter._client.headers


def test_retry_stream_raises_on_fallback_failure():
    failing = _FailingClient()
    rc = RetryableOllamaClient(client=failing)
    with pytest.raises(RuntimeError):
        # Advance the generator; should raise rather than yielding error text
        next(rc.chat_stream(model='gpt-oss:120b', messages=[{'role': 'user', 'content': 'hi'}], stream=True))


def test_tool_call_normalization_function_shape(monkeypatch):
    adapter = OllamaAdapter(api_key='test', model='gpt-oss:120b')
    class _FakeReturnTools:
        def chat(self, **kwargs):
            return {
                'message': {
                    'content': '',
                    'tool_calls': [
                        {
                            'type': 'function',
                            'id': 'call_t',
                            'function': {
                                'name': 'web_fetch',
                                'arguments': '{"x": 2}'
                            }
                        }
                    ]
                }
            }
    adapter._retry_client = _FakeReturnTools()  # type: ignore[attr-defined]

    messages = [{'role': 'user', 'content': 'hi'}]
    ctx = ConversationContext(model='gpt-oss:120b')
    res = adapter.chat(messages, ctx)
    tc = res.metadata.get('tool_calls')
    assert isinstance(tc, list) and tc == [{'name': 'web_fetch', 'arguments': {'x': 2}, 'id': 'call_t'}]


def test_tool_call_normalization_direct_shape(monkeypatch):
    adapter = OllamaAdapter(api_key='test', model='gpt-oss:120b')
    class _FakeReturnTools:
        def chat(self, **kwargs):
            return {
                'message': {
                    'content': '',
                    'tool_calls': [
                        {
                            'id': 'call_b',
                            'name': 'other',
                            'arguments': {'y': 3}
                        }
                    ]
                }
            }
    adapter._retry_client = _FakeReturnTools()  # type: ignore[attr-defined]

    messages = [{'role': 'user', 'content': 'hi'}]
    ctx = ConversationContext(model='gpt-oss:120b')
    res = adapter.chat(messages, ctx)
    tc = res.metadata.get('tool_calls')
    assert isinstance(tc, list) and tc == [{'name': 'other', 'arguments': {'y': 3}, 'id': 'call_b'}]
