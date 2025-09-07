import os
import sys
import types
import pytest

from src.client import OllamaTurboClient


class DummyOllamaClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        # Record the full kwargs to inspect 'messages' ordering/content
        self.calls.append(kwargs)
        return {
            'message': {
                'content': 'Final answer'
            }
        }


def _install_fake_mem0(monkeypatch) -> None:
    """Install a minimal fake 'mem0' module into sys.modules before client init."""
    mod = types.ModuleType("mem0")

    class MemoryClient:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        # Default no-op methods; tests will monkeypatch 'search' per case
        def search(self, query, version='v2', filters=None, limit=None):
            return []

        def add(self, *args, **kwargs):
            # Return a shape compatible with _mem0_execute_add extractors
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
def test_mem0_system_injection_present_and_before_user(monkeypatch, stream):
    # Enable Mem0 and make sure our fake module is used
    monkeypatch.setenv('MEM0_ENABLED', '1')
    monkeypatch.setenv('MEM0_API_KEY', 'test-key')
    monkeypatch.setenv('MEM0_SEARCH_TIMEOUT_MS', '1000')

    _install_fake_mem0(monkeypatch)

    client = OllamaTurboClient(api_key='test', enable_tools=False, quiet=True)

    # Patch Mem0 search to return two short memories
    def fake_search(query, version='v2', filters=None, limit=None):
        return [
            {'memory': 'User likes Go.'},
            {'memory': 'Name is Braden.'},
        ]

    monkeypatch.setattr(client.mem0_client, 'search', fake_search, raising=False)

    # Replace underlying network client with our recorder
    dummy = DummyOllamaClient()
    client.client = dummy

    out = client.chat('Hi there', stream=stream)
    assert 'Final answer' in out
    assert len(dummy.calls) >= 1

    call = dummy.calls[0]
    msgs = call.get('messages') or []
    assert isinstance(msgs, list) and len(msgs) >= 3

    # Locate mem0 system block and the user message
    prefix = client.prompt.mem0_prefix()
    mem_idxs = [i for i, m in enumerate(msgs) if m.get('role') == 'system' and str(m.get('content', '')).startswith(prefix)]
    assert len(mem_idxs) == 1
    mem_idx = mem_idxs[0]

    user_idxs = [i for i, m in enumerate(msgs) if m.get('role') == 'user' and m.get('content') == 'Hi there']
    assert len(user_idxs) == 1
    user_idx = user_idxs[0]

    # Mem0 block must appear before the user message
    assert mem_idx < user_idx

    # Content should include bullet points for each memory and tail note
    mem_content = msgs[mem_idx]['content']
    assert '- User likes Go.' in mem_content
    assert '- Name is Braden.' in mem_content
    assert 'Integrate this context naturally into your response only where it adds value.' in mem_content


def test_mem0_injection_replaced_each_turn(monkeypatch):
    # Enable Mem0 and install fake SDK
    monkeypatch.setenv('MEM0_ENABLED', '1')
    monkeypatch.setenv('MEM0_API_KEY', 'test-key')
    _install_fake_mem0(monkeypatch)

    client = OllamaTurboClient(api_key='test', enable_tools=False, quiet=True)

    # First search returns memory A
    def search_a(query, version='v2', filters=None, limit=None):
        return [{'memory': 'First fact A'}]

    # Second search returns memory B
    def search_b(query, version='v2', filters=None, limit=None):
        return [{'memory': 'Second fact B'}]

    # Replace underlying network client
    dummy = DummyOllamaClient()
    client.client = dummy

    # First turn
    monkeypatch.setattr(client.mem0_client, 'search', search_a, raising=False)
    _ = client.chat('Turn 1', stream=False)

    # Second turn -> swap search function
    monkeypatch.setattr(client.mem0_client, 'search', search_b, raising=False)
    _ = client.chat('Turn 2', stream=False)

    # After second turn, only the latest mem0 system block should remain
    prefix = client.prompt.mem0_prefix()
    mem_blocks = [m for m in client.conversation_history if m.get('role') == 'system' and str(m.get('content', '')).startswith(prefix)]
    assert len(mem_blocks) == 1
    assert 'Second fact B' in mem_blocks[0]['content']
    assert 'First fact A' not in mem_blocks[0]['content']


def test_mem0_injection_can_affect_model_output(monkeypatch):
    # Enable Mem0 and install fake SDK
    monkeypatch.setenv('MEM0_ENABLED', '1')
    monkeypatch.setenv('MEM0_API_KEY', 'test-key')
    _install_fake_mem0(monkeypatch)

    client = OllamaTurboClient(api_key='test', enable_tools=False, quiet=True)

    # Mem0 returns a name fact
    def search_name(query, version='v2', filters=None, limit=None):
        return [{'memory': 'Name is Braden.'}]

    monkeypatch.setattr(client.mem0_client, 'search', search_name, raising=False)

    class InfluenceClient(DummyOllamaClient):
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            msgs = kwargs.get('messages') or []
            prefix = client.prompt.mem0_prefix()
            saw_braden = any(m.get('role') == 'system' and str(m.get('content','')).startswith(prefix) and 'Braden' in str(m.get('content','')) for m in msgs)
            content = 'Hello, Braden!' if saw_braden else 'Hello.'
            return {'message': {'content': content}}

    dummy = InfluenceClient()
    client.client = dummy

    out = client.chat('Greet me', stream=False)
    assert 'Braden' in out


def test_mem0_merge_first_system_and_replace_next_turn(monkeypatch):
    # Enable Mem0 and force merge into first system message
    monkeypatch.setenv('MEM0_ENABLED', '1')
    monkeypatch.setenv('MEM0_API_KEY', 'test-key')
    monkeypatch.setenv('MEM0_IN_FIRST_SYSTEM', '1')

    _install_fake_mem0(monkeypatch)

    client = OllamaTurboClient(api_key='test', enable_tools=False, quiet=True)

    # First search returns memory A
    def search_a(query, version='v2', filters=None, limit=None):
        return [{'memory': 'First fact A'}]

    # Second search returns memory B
    def search_b(query, version='v2', filters=None, limit=None):
        return [{'memory': 'Second fact B'}]

    # Replace underlying network client
    dummy = DummyOllamaClient()
    client.client = dummy

    # Turn 1
    monkeypatch.setattr(client.mem0_client, 'search', search_a, raising=False)
    _ = client.chat('Turn 1', stream=False)

    # First system message should include Mem0 prefix and the A fact
    first_sys = client.conversation_history[0]
    prefix = client.prompt.mem0_prefix()
    assert first_sys.get('role') == 'system'
    assert prefix in str(first_sys.get('content', ''))
    assert 'First fact A' in str(first_sys.get('content', ''))

    # There should be no separate Mem0 system message (since merged into first)
    separate_blocks = [m for m in client.conversation_history[1:] if m.get('role') == 'system' and str(m.get('content','')).startswith(prefix)]
    assert len(separate_blocks) == 0

    # Turn 2 -> swap search function
    monkeypatch.setattr(client.mem0_client, 'search', search_b, raising=False)
    _ = client.chat('Turn 2', stream=False)

    # The first system message Mem0 section should now reflect B, not A
    first_sys2 = client.conversation_history[0]
    content2 = str(first_sys2.get('content', ''))
    assert prefix in content2
    assert 'Second fact B' in content2
    assert 'First fact A' not in content2
    # Still no separate blocks
    separate_blocks2 = [m for m in client.conversation_history[1:] if m.get('role') == 'system' and str(m.get('content','')).startswith(prefix)]
    assert len(separate_blocks2) == 0
