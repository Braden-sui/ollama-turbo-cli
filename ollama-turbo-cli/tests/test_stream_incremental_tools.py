import json
import types
import pytest

from src.client import OllamaTurboClient


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    # Keep Mem0 disabled for unit tests
    monkeypatch.setenv('MEM0_ENABLED', '0')
    # Small cap to exercise truncation path (kept consistent with other tests)
    monkeypatch.setenv('TOOL_CONTEXT_MAX_CHARS', '20')
    yield


class CountingGen:
    def __init__(self, items):
        self._items = list(items)
        self.yielded = 0
    def __iter__(self):
        for it in self._items:
            self.yielded += 1
            yield it


class DummyStreamingProvider:
    def __init__(self):
        self.calls = []
        self._step = 0
    def chat(self, **kwargs):
        self.calls.append(kwargs)
        self._step += 1
        # Always return a final-channel stream for subsequent rounds
        return CountingGen([
            {"message": {"content": "<|channel|>final\n<|message|>Final streaming<|end|>"}}
        ])


def test_streaming_early_tool_detection_aborts_stream(monkeypatch):
    client = OllamaTurboClient(api_key='test', enable_tools=True, quiet=True)

    # Create a counting generator for the initial stream (round 0):
    # First chunk contains a Harmony commentary tool call; second chunk should not be consumed
    initial_gen = CountingGen([
        {"message": {"content": (
            "<|channel|>commentary to=functions.web_fetch\n"
            "<|message|>{\\\"url\\\": \\\"https://example.com\\\"}<|call|>"
        )}},
        # This must not be read once tool call is detected
        "SHOULD_NOT_BE_READ"
    ])

    # Monkeypatch the initial streaming response creator to return our counting generator
    def initial_stream(self):
        return initial_gen
    monkeypatch.setattr(client, '_create_streaming_response', types.MethodType(initial_stream, client))

    # Subsequent streams via client.client.chat will be provided by DummyStreamingProvider
    provider = DummyStreamingProvider()
    client.client = provider

    # Provide tool implementation
    def fake_web_fetch(**kwargs):
        return json.dumps({
            'tool': 'web_fetch', 'ok': True,
            'inject': 'ok', 'sensitive': False
        })
    client.tool_functions['web_fetch'] = fake_web_fetch

    out = client.chat('Use web', stream=True)

    # Ensure we ended the first stream early (only first chunk consumed)
    assert initial_gen.yielded == 1
    # Ensure subsequent provider stream was used exactly once
    assert len(provider.calls) == 1
    # Ensure the final content is present and that the sentinel was not leaked
    assert 'Final streaming' in out
    assert 'SHOULD_NOT_BE_READ' not in out
