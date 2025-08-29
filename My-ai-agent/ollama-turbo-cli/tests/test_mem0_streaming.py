import os
import sys
import types
from typing import Any, Dict, Iterable, List, Optional

import pytest

from src.client import OllamaTurboClient


class DummyStreamChunk(dict):
    pass


class DummyStreamOllamaClient:
    """Records chat() calls and returns iterables for streaming.

    Behaviors:
    - Normal streaming: yields token chunks with content only.
    - Tool streaming: on first round yields a chunk with tool_calls; subsequent round yields content.
    - Error streaming: yields a partial then raises to trigger fallback path.
    - Non-streaming: returns a simple final message.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._round: int = 0

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        is_stream = bool(kwargs.get("stream"))
        if is_stream:
            # Inspect last user message to decide behavior for this turn/round
            msgs = kwargs.get("messages") or []
            last_user = ""
            for m in reversed(msgs):
                if m.get("role") == "user":
                    last_user = str(m.get("content", ""))
                    break

            # Error path: trigger read error
            if "raise" in last_user:
                def _gen_err():
                    yield DummyStreamChunk({"message": {"content": "Partial"}})
                    raise RuntimeError("stream error")
                return _gen_err()

            # Tool first round -> emit tool_calls once
            if "use-tools" in last_user and self._round == 0:
                self._round += 1
                def _gen_tools():
                    yield DummyStreamChunk({
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": "call_1",
                                    "function": {"name": "echo", "arguments": {"text": "hi"}},
                                }
                            ],
                        }
                    })
                return _gen_tools()

            # Normal token stream
            def _gen_tokens():
                for part in ["Hello", " world"]:
                    yield DummyStreamChunk({"message": {"content": part}})
            return _gen_tokens()

        # Non-streaming response (used by fallback)
        return {"message": {"content": "Final via fallback"}}


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


@pytest.mark.parametrize('stream', [True])
def test_mem0_streaming_injection_and_persistence_no_tools(monkeypatch, stream):
    # Enable Mem0 and ensure fake SDK is used
    monkeypatch.setenv('MEM0_ENABLED', '1')
    monkeypatch.setenv('MEM0_API_KEY', 'test-key')
    monkeypatch.setenv('MEM0_SEARCH_TIMEOUT_MS', '1000')

    _install_fake_mem0(monkeypatch)

    client = OllamaTurboClient(api_key='test', enable_tools=False, quiet=True)

    # Patch Mem0 search to return a short memory
    def fake_search(query, version='v2', filters=None, limit=None):
        return [
            {'memory': 'User likes Go.'},
        ]
    monkeypatch.setattr(client.mem0_client, 'search', fake_search, raising=False)

    # Replace underlying network client with our streaming dummy
    dummy = DummyStreamOllamaClient()
    client.client = dummy

    # Capture Mem0 persistence enqueue
    added: List[Dict[str, Any]] = []
    def fake_enqueue(messages, metadata):
        added.append({"messages": messages, "metadata": metadata})
    monkeypatch.setattr(client, '_mem0_enqueue_add', fake_enqueue)

    out = client.chat('hello', stream=stream)
    assert isinstance(out, str)
    assert out  # some content produced

    # Verify first streamed request includes Mem0 system content before user
    assert len(dummy.calls) >= 1
    call0 = dummy.calls[0]
    msgs = call0.get('messages') or []
    assert isinstance(msgs, list) and len(msgs) >= 2

    prefix = client.prompt.mem0_prefix()
    mem_idxs = [i for i, m in enumerate(msgs) if m.get('role') == 'system' and prefix in str(m.get('content', ''))]
    assert len(mem_idxs) >= 1
    user_idxs = [i for i, m in enumerate(msgs) if m.get('role') == 'user' and m.get('content') == 'hello']
    assert len(user_idxs) == 1
    assert min(mem_idxs) < user_idxs[0]

    # Verify Mem0 persistence was enqueued with user + assistant
    assert len(added) == 1
    persisted_msgs = added[0]['messages']
    roles = [m.get('role') for m in persisted_msgs]
    assert roles == ['user', 'assistant']
    assert persisted_msgs[0]['content'] == 'hello'
    assert isinstance(persisted_msgs[1]['content'], str) and persisted_msgs[1]['content']


def test_mem0_streaming_with_tools_then_final_persists(monkeypatch):
    # Enable Mem0 and fake SDK
    monkeypatch.setenv('MEM0_ENABLED', '1')
    monkeypatch.setenv('MEM0_API_KEY', 'test-key')
    monkeypatch.setenv('MEM0_SEARCH_TIMEOUT_MS', '1000')

    _install_fake_mem0(monkeypatch)

    client = OllamaTurboClient(api_key='test', enable_tools=True, quiet=True)

    # Patch Mem0 search
    monkeypatch.setattr(
        client.mem0_client,
        'search',
        lambda q, version='v2', filters=None, limit=None: [{'memory': 'Name is Braden.'}],
        raising=False,
    )

    # Provide a simple echo tool to satisfy tool execution
    def fake_echo(**kwargs):
        import json
        return json.dumps({
            'tool': 'echo',
            'ok': True,
            'inject': 'ok',
            'sensitive': False,
        })
    client.tool_functions['echo'] = fake_echo

    dummy = DummyStreamOllamaClient()
    client.client = dummy

    # Capture Mem0 persistence enqueue
    added: List[Dict[str, Any]] = []
    monkeypatch.setattr(client, '_mem0_enqueue_add', lambda messages, metadata: added.append({"messages": messages, "metadata": metadata}))

    out = client.chat('please use-tools', stream=True)
    assert isinstance(out, str)
    assert 'Tool Results' in out or out

    # Expect at least two calls: first with tool_calls, second for final
    assert len(dummy.calls) >= 2
    kwargs0, kwargs1 = dummy.calls[0], dummy.calls[1]

    # Mem0 content present in first request before user
    msgs0 = kwargs0.get('messages') or []
    prefix = client.prompt.mem0_prefix()
    mem_idxs0 = [i for i, m in enumerate(msgs0) if m.get('role') == 'system' and prefix in str(m.get('content',''))]
    user_idx0 = next(i for i, m in enumerate(msgs0) if m.get('role') == 'user' and 'use-tools' in str(m.get('content','')))
    assert mem_idxs0 and min(mem_idxs0) < user_idx0

    # Persistence enqueued once with user + assistant
    assert len(added) == 1
    roles = [m.get('role') for m in added[0]['messages']]
    assert roles == ['user', 'assistant']


def test_mem0_streaming_read_error_fallback_preserves_mem_and_persists(monkeypatch):
    # Enable Mem0 and fake SDK
    monkeypatch.setenv('MEM0_ENABLED', '1')
    monkeypatch.setenv('MEM0_API_KEY', 'test-key')
    monkeypatch.setenv('MEM0_SEARCH_TIMEOUT_MS', '1000')

    _install_fake_mem0(monkeypatch)

    client = OllamaTurboClient(api_key='test', enable_tools=False, quiet=True)

    monkeypatch.setattr(
        client.mem0_client,
        'search',
        lambda q, version='v2', filters=None, limit=None: [{'memory': 'Prefers concise answers.'}],
        raising=False,
    )

    dummy = DummyStreamOllamaClient()
    client.client = dummy

    added: List[Dict[str, Any]] = []
    monkeypatch.setattr(client, '_mem0_enqueue_add', lambda messages, metadata: added.append({"messages": messages, "metadata": metadata}))

    # Trigger a streaming read error -> fallback to non-streaming path
    out = client.chat('please raise', stream=True)
    assert isinstance(out, str)

    # First call was streaming; second call should be non-streaming fallback
    assert len(dummy.calls) >= 2
    first = dummy.calls[0]
    second = dummy.calls[1]

    # Verify Mem0 presence in the (fallback) non-streaming request too
    prefix = client.prompt.mem0_prefix()
    for call in (first, second):
        msgs = call.get('messages') or []
        mem_idxs = [i for i, m in enumerate(msgs) if m.get('role') == 'system' and prefix in str(m.get('content',''))]
        user_idxs = [i for i, m in enumerate(msgs) if m.get('role') == 'user']
        assert mem_idxs and user_idxs and min(mem_idxs) < user_idxs[0]

    # Persistence enqueued with user + assistant once
    assert len(added) == 1
    roles = [m.get('role') for m in added[0]['messages']]
    assert roles == ['user', 'assistant']
