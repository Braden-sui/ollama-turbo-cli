import sys
import types
import re
from typing import Any, Dict, Iterable, List

import pytest

from src.client import OllamaTurboClient


class DummyOllamaClient:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "ok"}}


class DummyStreamOllamaClient:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self._round = 0

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        is_stream = bool(kwargs.get("stream"))
        if not is_stream:
            return {"message": {"content": "final"}}

        # Decide behavior based on last user content
        msgs = kwargs.get("messages") or []
        last_user = next((str(m.get("content","")) for m in reversed(msgs) if m.get("role") == "user"), "")

        if "use-tools" in last_user and self._round == 0:
            self._round += 1
            def _gen_tools():
                yield {"message": {"content": "", "tool_calls": [
                    {"type": "function", "id": "call_1", "function": {"name": "echo", "arguments": {"text": "hi"}}}
                ]}}
            return _gen_tools()

        def _gen_tokens():
            for part in ["Hello", " world"]:
                yield {"message": {"content": part}}
        return _gen_tokens()


def _install_fake_mem0(monkeypatch) -> None:
    mod = types.ModuleType("mem0")
    class MemoryClient:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def search(self, query, version='v2', filters=None, limit=None):
            return []
        def add(self, *args, **kwargs):
            return {"id": "mem-1"}
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


def _capture_print_trace(client: OllamaTurboClient):
    captured: List[str] = []
    def _fake_print_trace():
        # Snapshot without clearing to allow assertions
        captured[:] = list(client.trace)
    return captured, _fake_print_trace


def _has_mem0_present(lines: List[str], where_prefix: str, *, first_expected: int = 1, min_blocks: int = 1) -> bool:
    for ln in lines:
        if ln.startswith(f"mem0:present:{where_prefix}"):
            if f"first={first_expected}" not in ln:
                return False
            m = re.search(r"blocks=(\d+)", ln)
            if not m:
                return False
            return int(m.group(1)) >= min_blocks
    return False


def test_trace_mem0_presence_standard_merge_first(monkeypatch):
    monkeypatch.setenv('MEM0_ENABLED', '1')
    monkeypatch.setenv('MEM0_API_KEY', 'test-key')
    monkeypatch.setenv('MEM0_IN_FIRST_SYSTEM', '1')
    monkeypatch.setenv('MEM0_SEARCH_TIMEOUT_MS', '1000')

    _install_fake_mem0(monkeypatch)

    client = OllamaTurboClient(api_key='test', enable_tools=False, show_trace=True, quiet=True)

    # Mem0 returns one fact
    monkeypatch.setattr(
        client.mem0_client,
        'search',
        lambda q, version='v2', filters=None, limit=None: [{"memory": "Name is Braden."}],
        raising=False,
    )

    dummy = DummyOllamaClient()
    client.client = dummy

    captured, fake_printer = _capture_print_trace(client)
    monkeypatch.setattr(client, '_print_trace', fake_printer)

    _ = client.chat('hello', stream=False)

    assert any(s.startswith('mem0:inject:first') for s in captured)
    assert _has_mem0_present(captured, 'standard:r0', first_expected=1, min_blocks=1)


def test_trace_mem0_presence_streaming_init_and_round(monkeypatch):
    monkeypatch.setenv('MEM0_ENABLED', '1')
    monkeypatch.setenv('MEM0_API_KEY', 'test-key')
    monkeypatch.setenv('MEM0_SEARCH_TIMEOUT_MS', '1000')

    _install_fake_mem0(monkeypatch)

    client = OllamaTurboClient(api_key='test', enable_tools=True, show_trace=True, quiet=True)
    # Mem0 returns one fact
    monkeypatch.setattr(
        client.mem0_client,
        'search',
        lambda q, version='v2', filters=None, limit=None: [{"memory": "Prefers concise answers."}],
        raising=False,
    )

    # Provide echo tool
    def fake_echo(**kwargs):
        import json
        return json.dumps({"tool": "echo", "ok": True})
    client.tool_functions['echo'] = fake_echo

    dummy = DummyStreamOllamaClient()
    client.client = dummy

    captured, fake_printer = _capture_print_trace(client)
    monkeypatch.setattr(client, '_print_trace', fake_printer)

    _ = client.chat('please use-tools', stream=True)

    # Trace should include presence at stream init and round 0
    assert _has_mem0_present(captured, 'stream:init', first_expected=1, min_blocks=1) or _has_mem0_present(captured, 'stream:init', first_expected=0, min_blocks=1)
    assert _has_mem0_present(captured, 'stream:r0', first_expected=1, min_blocks=1) or _has_mem0_present(captured, 'stream:r0', first_expected=0, min_blocks=1)
