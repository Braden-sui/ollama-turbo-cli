import json
import pytest
import types

from src.client import OllamaTurboClient


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    # Keep Mem0 disabled for unit tests
    monkeypatch.setenv('MEM0_ENABLED', '0')
    # Small cap to exercise truncation path (kept consistent with other tests)
    monkeypatch.setenv('TOOL_CONTEXT_MAX_CHARS', '20')
    yield


class DummyHarmonyClientNonStreaming:
    def __init__(self):
        self.calls = []
        self._step = 0

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._step == 0:
            self._step += 1
            # Harmony commentary tool-call tokens without provider-canonicalized tool_calls
            content = (
                "<|channel|>commentary to=functions.web_fetch\n"
                "<|message|>{\"url\": \"https://example.com\"}<|call|>"
            )
            return {"message": {"content": content}}
        # Final answer on Harmony final channel
        return {
            "message": {
                "content": "<|channel|>final\n<|message|>Final from harmony<|end|>"
            }
        }


def test_standard_chat_harmony_tool_call_and_final_channel(monkeypatch):
    client = OllamaTurboClient(api_key='test', enable_tools=True, quiet=True)
    dummy = DummyHarmonyClientNonStreaming()
    client.client = dummy

    # Provide tool function implementation
    def fake_web_fetch(**kwargs):
        return json.dumps({
            'tool': 'web_fetch', 'ok': True,
            'inject': 'ok', 'sensitive': False
        })
    client.tool_functions['web_fetch'] = fake_web_fetch

    out = client.chat('Use web', stream=False)
    assert 'Final from harmony' in out
    # Ensure two calls were made
    assert len(dummy.calls) == 2
    first, second = dummy.calls[0], dummy.calls[1]
    # First request should include tools
    assert 'tools' in first
    # Final request must not include tools
    assert 'tools' not in second


# ---------- Streaming ----------
class _GenWrap:
    def __init__(self, items):
        self._items = list(items)
    def __iter__(self):
        for it in self._items:
            yield it


class DummyStreamingProvider:
    def __init__(self):
        self.calls = []
        self._step = 0
    def chat(self, **kwargs):
        # Subsequent rounds return a final-channel stream
        self.calls.append(kwargs)
        self._step += 1
        return _GenWrap([
            {"message": {"content": "<|channel|>final\n<|message|>Final streaming<|end|>"}}
        ])


def test_streaming_chat_harmony_tool_call_and_final_channel(monkeypatch):
    client = OllamaTurboClient(api_key='test', enable_tools=True, quiet=True)
    # First stream (round 0) yields a commentary tool-call in Harmony format
    def initial_stream(self):
        return _GenWrap([
            {"message": {"content": (
                "<|channel|>commentary to=functions.web_fetch\n"
                "<|message|>{\\\"url\\\": \\\"https://example.com\\\"}<|call|>"
            )}}
        ])
    # Subsequent streams via client.client.chat will be provided by DummyStreamingProvider
    provider = DummyStreamingProvider()
    client.client = provider

    # Monkeypatch the initial streaming response creator
    monkeypatch.setattr(client, '_create_streaming_response', types.MethodType(initial_stream, client))

    # Provide tool implementation
    def fake_web_fetch(**kwargs):
        return json.dumps({
            'tool': 'web_fetch', 'ok': True,
            'inject': 'ok', 'sensitive': False
        })
    client.tool_functions['web_fetch'] = fake_web_fetch

    out = client.chat('Use web', stream=True)
    assert 'Final streaming' in out
