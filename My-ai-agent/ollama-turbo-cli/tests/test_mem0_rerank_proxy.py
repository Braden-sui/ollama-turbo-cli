import os
import sys
import types
import pytest

from src.client import OllamaTurboClient


class DummyOllamaClient:
    def __init__(self):
        self.calls = []
        self._count = 0

    def chat(self, **kwargs):
        # Record the call
        self.calls.append(kwargs)
        self._count += 1
        # First call is expected to be the reranker proxy
        if self._count == 1:
            return {'message': {'content': '[1,0]'}}
        # Subsequent call is the main chat
        return {'message': {'content': 'Final answer'}}


def _install_fake_mem0(monkeypatch) -> None:
    mod = types.ModuleType("mem0")

    class MemoryClient:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, version='v2', filters=None, limit=None):
            return []

        def add(self, *args, **kwargs):
            return {'id': 'mem-1'}

        def update(self, *args, **kwargs):
            return True

        def get_all(self, *args, **kwargs):
            return []

        def delete_all(self, *args, **kwargs):
            return None

        def link(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            return {}

    mod.MemoryClient = MemoryClient
    monkeypatch.setitem(sys.modules, 'mem0', mod)


@pytest.mark.parametrize('stream', [False])
def test_mem0_proxy_reranks_injected_context(monkeypatch, stream):
    # Enable Mem0 and proxy reranker
    monkeypatch.setenv('MEM0_ENABLED', '1')
    monkeypatch.setenv('MEM0_API_KEY', 'test-key')
    monkeypatch.setenv('MEM0_PROXY_MODEL', 'gpt-oss:20b')

    _install_fake_mem0(monkeypatch)

    client = OllamaTurboClient(api_key='test', enable_tools=False, quiet=True)

    # Mem0 search will return two memories A then B
    def fake_search(query, version='v2', filters=None, limit=None):
        return [
            {'memory': 'Alpha fact A'},
            {'memory': 'Bravo fact B'},
        ]

    monkeypatch.setattr(client.mem0_client, 'search', fake_search, raising=False)

    # Replace underlying client with our stub that returns a rerank order [1,0]
    dummy = DummyOllamaClient()
    client.client = dummy

    out = client.chat('Hi there', stream=stream)
    assert 'Final answer' in out

    # We expect two calls: first reranker, then main chat
    assert len(dummy.calls) >= 2

    # Inspect main chat payload to find Mem0 injected system message
    main_payload = dummy.calls[-1]
    msgs = main_payload.get('messages') or []
    prefix = client.prompt.mem0_prefix()
    mem_blocks = [m for m in msgs if m.get('role') == 'system' and str(m.get('content','')).startswith(prefix)]
    assert len(mem_blocks) == 1
    mem_text = str(mem_blocks[0].get('content',''))

    # Ensure B appears before A due to rerank [1,0]
    idx_b = mem_text.find('Bravo fact B')
    idx_a = mem_text.find('Alpha fact A')
    assert idx_b != -1 and idx_a != -1 and idx_b < idx_a
