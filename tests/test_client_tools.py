import os
import json
import types
import pytest

from src.client import OllamaTurboClient


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    # Keep Mem0 disabled for unit tests
    monkeypatch.setenv('MEM0_ENABLED', '0')
    # Small cap to exercise truncation path
    monkeypatch.setenv('TOOL_CONTEXT_MAX_CHARS', '20')
    yield


def _make_tool_call(name: str, args: dict) -> dict:
    return {
        'type': 'function',
        'id': 'call_1',
        'function': {
            'name': name,
            'arguments': args,
        }
    }


def test_execute_tool_calls_injection_cap_and_memskip(monkeypatch):
    client = OllamaTurboClient(api_key='test', enable_tools=True, quiet=True)
    # Monkeypatch the web_fetch implementation on the instance
    long_inject = 'A' * 100
    def fake_web_fetch(**kwargs):
        payload = {
            'tool': 'web_fetch',
            'ok': True,
            'inject': long_inject,
            'sensitive': False,
            'log_path': 'sandbox://sessions/123/full.log'
        }
        return json.dumps(payload)
    client.tool_functions['web_fetch'] = fake_web_fetch

    tool_calls = [_make_tool_call('web_fetch', {'url': 'https://example.com'})]
    injected = client._execute_tool_calls(tool_calls)

    assert isinstance(injected, list) and len(injected) == 1
    disp = injected[0]
    # Should be capped and include truncation notice with log_path
    assert len(disp) <= client.tool_context_cap + 120  # allow for notice text
    assert 'truncated; full logs stored at sandbox://sessions/123/full.log' in disp
    # Mem0 should be skipped after sensitive or large outputs
    assert client._skip_mem0_after_turn is True


class DummyOllamaClient:
    def __init__(self):
        self.calls = []
        self._step = 0

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        # First call: return a tool call to web_fetch
        if self._step == 0:
            self._step += 1
            return {
                'message': {
                    'content': '',
                    'tool_calls': [
                        {
                            'type': 'function',
                            'id': 'call_1',
                            'function': {
                                'name': 'web_fetch',
                                'arguments': {'url': 'https://example.com'}
                            }
                        }
                    ]
                }
            }
        # Second (final) call: textual content only
        return {
            'message': {
                'content': 'Final answer'
            }
        }


def test_standard_chat_final_call_omits_tools(monkeypatch):
    client = OllamaTurboClient(api_key='test', enable_tools=True, quiet=True)
    # Replace underlying Ollama client
    dummy = DummyOllamaClient()
    client.client = dummy

    # Monkeypatch web_fetch to return small safe payload
    def fake_web_fetch(**kwargs):
        return json.dumps({'tool': 'web_fetch', 'ok': True, 'inject': 'ok', 'sensitive': False})
    client.tool_functions['web_fetch'] = fake_web_fetch

    out = client.chat('Use web', stream=False)
    assert 'Final answer' in out
    # Ensure two calls were made
    assert len(dummy.calls) == 2
    first, second = dummy.calls[0], dummy.calls[1]
    # First request should include tools
    assert 'tools' in first
    # Final request must not include tools
    assert 'tools' not in second
