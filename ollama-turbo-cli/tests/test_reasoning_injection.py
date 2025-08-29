import os
import types
import pytest

from src.client import OllamaTurboClient


class DummyOllamaClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {
            'message': {
                'content': 'ok'
            }
        }


class DummyStreamingClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        # Return a simple iterator that yields a couple of content chunks then stops
        def _iter():
            yield {'message': {'content': 'Hello'}}
            yield {'message': {'content': ' world'}}
        return _iter()


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Clear env customizations used by reasoning injection to avoid test leakage
    for k in [
        'REASONING_FIELD_PATH',
        'REASONING_FIELD_STYLE',
        'REASONING_OBJECT_KEY',
    ]:
        monkeypatch.delenv(k, raising=False)
    yield


def test_reasoning_system_mode_no_request_param(monkeypatch):
    client = OllamaTurboClient(api_key='x', enable_tools=False, quiet=True, reasoning='medium', reasoning_mode='system')
    dummy = DummyOllamaClient()
    client.client = dummy

    out = client.chat('hi', stream=False)
    assert 'ok' in out
    assert len(dummy.calls) == 1
    call = dummy.calls[0]
    # No top-level reasoning
    assert 'reasoning' not in call
    # And no default options path either
    opts = call.get('options') or {}
    assert 'reasoning_effort' not in opts
    assert 'reasoning' not in opts


def test_reasoning_request_top_injection(monkeypatch):
    client = OllamaTurboClient(api_key='x', enable_tools=False, quiet=True, reasoning='medium', reasoning_mode='request:top')
    dummy = DummyOllamaClient()
    client.client = dummy

    _ = client.chat('go', stream=False)
    call = dummy.calls[0]
    assert call.get('reasoning') == 'medium'


def test_reasoning_request_options_default_path(monkeypatch):
    client = OllamaTurboClient(api_key='x', enable_tools=False, quiet=True, reasoning='medium', reasoning_mode='request:options')
    dummy = DummyOllamaClient()
    client.client = dummy

    _ = client.chat('go', stream=False)
    call = dummy.calls[0]
    opts = call.get('options') or {}
    assert opts.get('reasoning_effort') == 'medium'


def test_reasoning_env_override_object_style(monkeypatch):
    monkeypatch.setenv('REASONING_FIELD_PATH', 'options.reasoning')
    monkeypatch.setenv('REASONING_FIELD_STYLE', 'object')
    monkeypatch.setenv('REASONING_OBJECT_KEY', 'level')

    client = OllamaTurboClient(api_key='x', enable_tools=False, quiet=True, reasoning='low', reasoning_mode='request:options')
    dummy = DummyOllamaClient()
    client.client = dummy

    _ = client.chat('custom', stream=False)
    call = dummy.calls[0]
    opts = call.get('options') or {}
    assert isinstance(opts.get('reasoning'), dict)
    assert opts['reasoning'].get('level') == 'low'


def test_reasoning_injection_streaming_top_level(monkeypatch):
    # Verify we inject for streaming initial request as well
    client = OllamaTurboClient(api_key='x', enable_tools=False, quiet=True, reasoning='high', reasoning_mode='request:top')
    dummy = DummyStreamingClient()
    client.client = dummy

    out = client.chat('stream this', stream=True)
    assert isinstance(out, str)
    # First streaming call should include top-level reasoning
    assert len(dummy.calls) >= 1
    first = dummy.calls[0]
    assert first.get('reasoning') == 'high'
