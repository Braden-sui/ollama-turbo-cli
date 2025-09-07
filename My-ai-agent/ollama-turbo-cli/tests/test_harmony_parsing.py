import json
import pytest
import types

from src.client import OllamaTurboClient
from src.harmony_processor import HarmonyProcessor


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


# ---------- Analysis channel handling ----------
def test_harmony_processor_captures_analysis_and_removes_from_cleaned():
    hp = HarmonyProcessor()
    text = (
        "<|channel|>analysis\n<|message|>Internal reasoning XYZ<|end|>\n"
        "<|channel|>final\n<|message|>Visible answer<|end|>"
    )
    cleaned, tool_calls, final_text = hp.parse_tokens(text)
    assert tool_calls == []
    assert final_text == "Visible answer"
    # Captured for tracing, not leaked in cleaned text
    assert getattr(hp, 'last_analysis', None) is not None
    assert "Internal reasoning XYZ" in hp.last_analysis
    assert "Internal reasoning XYZ" not in cleaned


class DummyClientWithAnalysis:
    def __init__(self):
        self.calls = []
        self._step = 0
    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._step == 0:
            self._step += 1
            # Include analysis + a harmony commentary tool call
            content = (
                "<|channel|>analysis\n<|message|>THINKING-SECRET 123<|end|>\n"
                "<|channel|>commentary to=functions.web_fetch\n"
                "<|message|>{\"url\": \"https://example.com\"}<|call|>"
            )
            return {"message": {"content": content}}
        # Second step: final with another analysis block up front
        return {
            "message": {
                "content": (
                    "<|channel|>analysis\n<|message|>ANOTHER SECRET STEP<|end|>\n"
                    "<|channel|>final\n<|message|>Answer only<|end|>"
                )
            }
        }


def test_non_streaming_chat_does_not_leak_analysis(monkeypatch):
    client = OllamaTurboClient(api_key='test', enable_tools=True, quiet=True)
    dummy = DummyClientWithAnalysis()
    client.client = dummy

    def fake_web_fetch(**kwargs):
        return json.dumps({
            'tool': 'web_fetch', 'ok': True,
            'inject': 'ok', 'sensitive': False
        })
    client.tool_functions['web_fetch'] = fake_web_fetch

    out = client.chat('Use web', stream=False)
    assert 'Answer only' in out
    assert 'THINKING-SECRET' not in out
    assert 'ANOTHER SECRET STEP' not in out
