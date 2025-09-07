import json
import os
from typing import Any, Dict, List

import pytest

from src.client import OllamaTurboClient


class DummyClient:
    """Minimal underlying client that simulates both streaming and non-streaming flows.

    Behavior switches on kwargs:
    - stream=True:
      - If last user contains 'use-tools' on first call, emit a chunk with tool_calls; next call emits content tokens.
      - Else emit two content token chunks.
    - stream=False:
      - If last user contains 'use-tools', return a single response with tool_calls to trigger tool execution.
      - Else return a simple final message.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.round_counter: int = 0

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        is_stream = bool(kwargs.get("stream"))
        msgs = kwargs.get("messages") or []
        last_user = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break

        if is_stream:
            # First streaming round with tools
            if "use-tools" in last_user and self.round_counter == 0:
                self.round_counter += 1
                def _with_tools():
                    yield {"message": {"content": "", "tool_calls": [
                        {"type": "function", "id": "call_1", "function": {"name": "echo", "arguments": {"text": "hi"}}}
                    ]}}
                return _with_tools()
            # Subsequent streaming or no-tools: emit two content token chunks
            def _tokens():
                for part in ["Hello", " world"]:
                    yield {"message": {"content": part}}
            return _tokens()
        else:
            # Non-streaming
            if "use-tools" in last_user:
                return {"message": {"content": "", "tool_calls": [
                    {"type": "function", "id": "call_1", "function": {"name": "echo", "arguments": {"text": "hi"}}}
                ]}}
            return {"message": {"content": "Hello world"}}


@pytest.fixture(autouse=True)
def _no_mem0(monkeypatch):
    # Ensure Mem0 is disabled for these parity tests to avoid extra system messages
    monkeypatch.setenv('MEM0_ENABLED', '0')


def make_client(enable_tools: bool) -> OllamaTurboClient:
    client = OllamaTurboClient(api_key='test', enable_tools=enable_tools, quiet=True)
    client.client = DummyClient()
    client.show_trace = True
    return client


@pytest.mark.parametrize("mem0_enabled", [False, True])
def test_standard_vs_streaming_no_tools_parity(mem0_enabled, monkeypatch):
    # Toggle Mem0 per parameter to harden against default changes
    monkeypatch.setenv('MEM0_ENABLED', '1' if mem0_enabled else '0')
    # Standard
    c_std = make_client(enable_tools=False)
    out_std = c_std.chat('hello', stream=False)
    # Streaming
    c_str = make_client(enable_tools=False)
    out_str = c_str.chat('hello', stream=True)

    assert isinstance(out_std, str) and isinstance(out_str, str)
    assert out_std.strip() == out_str.strip() == 'Hello world'

    # Trace parity: standard r0 and streaming r0 traces present
    assert any(ev.startswith('request:standard:round=0') for ev in c_std.trace)
    # Streaming asserts start marker and r0 round marker
    assert any(ev == 'request:stream:start' for ev in c_str.trace)
    assert any(ev.startswith('request:stream:round=0') for ev in c_str.trace)
    # Require Mem0 presence breadcrumb only when Mem0 is enabled
    if getattr(c_str, 'mem0_enabled', False):
        assert any('mem0:present:stream:r0' in ev for ev in c_str.trace)
    assert any(ev.startswith('request:stream:round=0') for ev in c_str.trace)


def test_single_round_tool_loop_parity(monkeypatch):
    # Provide a simple echo tool
    def fake_echo(**kwargs):
        return json.dumps({"tool": "echo", "ok": True})

    # Standard path with tools
    c_std = make_client(enable_tools=True)
    c_std.tool_functions['echo'] = fake_echo
    out_std = c_std.chat('please use-tools', stream=False)

    # Streaming path with tools
    c_str = make_client(enable_tools=True)
    c_str.tool_functions['echo'] = fake_echo
    out_str = c_str.chat('please use-tools', stream=True)

    for out in (out_std, out_str):
        assert isinstance(out, str)
        assert 'Tool Results' in out

    # Reprompt trace and tools used count present
    assert any(ev == 'reprompt:after-tools' for ev in c_std.trace)
    assert any(ev == 'reprompt:after-tools' for ev in c_str.trace)
    assert any(ev.startswith('tools:used=') for ev in c_std.trace)
    assert any(ev.startswith('tools:used=') for ev in c_str.trace)


def test_streaming_reprompt_adapter_fallback_trace(monkeypatch):
    # Force adapter.format_reprompt_after_tools to raise so we exercise fallback + trace
    c = make_client(enable_tools=True)

    def fake_echo(**kwargs):
        return json.dumps({"tool": "echo", "ok": True})
    c.tool_functions['echo'] = fake_echo

    def raise_formatter(msgs, payload, options=None):
        raise RuntimeError('formatter error')

    monkeypatch.setattr(c.adapter, 'format_reprompt_after_tools', raise_formatter)

    out = c.chat('please use-tools', stream=True)
    assert isinstance(out, str)
    # Breadcrumb from orchestrator fallback
    assert 'reprompt:after-tools:fallback' in c.trace
    # Still produces a final output
    assert out
