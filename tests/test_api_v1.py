from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from fastapi.testclient import TestClient

from src.api.app import app
from src.api import router_v1


class FakeStreamChunk(dict):
    pass


class FakeOllamaClient:
    def __init__(self, api_key: str = "", enable_tools: bool = True, quiet: bool = True, **_: Any) -> None:
        self.api_key = api_key
        self.enable_tools = enable_tools
        self.quiet = quiet
        self.conversation_history = []
        self._last_tool_results_structured: Optional[list] = None
        self._last_tool_results_strings: Optional[list] = None

    # ----- Non-streaming path -----
    def chat(self, message: str, stream: bool = False) -> str:
        # Simulate tool results availability even in non-streaming
        self._last_tool_results_structured = [
            {
                "tool": "echo",
                "status": "ok",
                "content": "ok",
                "metadata": {"args": {"text": "ok"}},
                "error": None,
            }
        ]
        self._last_tool_results_strings = ["echo: ok"]
        return f"Echo: {message}"

    # ----- Streaming path helpers -----
    def _inject_mem0_context(self, user_message: str) -> None:
        return None

    def _trim_history(self) -> None:
        return None

    def _set_idempotency_key(self, key: Optional[str]) -> None:
        return None

    def _clear_idempotency_key(self) -> None:
        return None

    def _strip_harmony_markup(self, text: str) -> str:
        return text

    def _parse_harmony_tokens(self, text: str):
        # cleaned, tool_calls, final_text
        return text, None, None

    def _mem0_add_after_response(self, user_message: Optional[str], assistant_message: Optional[str]) -> None:
        return None

    def _handle_standard_chat(self) -> str:
        # Set example tool results for fallback path
        self._last_tool_results_structured = [
            {
                "tool": "calc",
                "status": "ok",
                "content": 42,
                "metadata": {"args": {"x": 40, "y": 2}},
                "error": None,
            }
        ]
        self._last_tool_results_strings = ["calc: 42"]
        return "Final via fallback"

    def _create_streaming_response(self) -> Iterable[Dict[str, Any]]:
        # Inspect last user msg to decide behavior
        msg = self.conversation_history[-1]["content"] if self.conversation_history else ""
        if "use-tools" in msg:
            # Emit a chunk that includes tool_calls to trigger fallback
            yield FakeStreamChunk({"message": {"content": "", "tool_calls": [{"name": "calc", "arguments": {"x": 1}}]}})
            return
        # Normal token streaming path (no tools)
        for part in ["Hel", "lo"]:
            yield FakeStreamChunk({"message": {"content": part}})


def _patch_client(monkeypatch):
    monkeypatch.setattr(router_v1, "OllamaTurboClient", FakeOllamaClient)


def test_chat_non_streaming_tool_results_object(monkeypatch):
    _patch_client(monkeypatch)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat",
        json={"message": "hi", "options": {"tool_results_format": "object"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"].startswith("Echo: hi")
    assert isinstance(data.get("tool_results"), list)
    assert data["tool_results"][0]["tool"] == "echo"


def _read_sse_events(resp) -> list[dict]:
    events: list[dict] = []
    for chunk in resp.iter_lines():
        if not chunk:
            continue
        # httpx/testclient may yield str or bytes depending on version
        if isinstance(chunk, bytes):
            line = chunk.decode("utf-8", errors="ignore")
        else:
            line = chunk
        if line.startswith("data: "):
            payload = line[len("data: "):]
            try:
                events.append(json.loads(payload))
            except Exception:
                # best effort
                pass
    return events


def test_chat_stream_tokens_and_final_no_tools(monkeypatch):
    _patch_client(monkeypatch)
    client = TestClient(app)
    with client.stream("POST", "/v1/chat/stream", json={"message": "hello"}) as resp:
        assert resp.status_code == 200
        events = _read_sse_events(resp)
    # Expect at least one token and a final
    assert any(e.get("type") == "token" for e in events)
    finals = [e for e in events if e.get("type") == "final"]
    assert len(finals) == 1
    assert finals[0].get("content")
    # No tool_results for pure text path
    assert "tool_results" not in finals[0]


def test_chat_stream_fallback_on_tool_calls_with_tool_results(monkeypatch):
    _patch_client(monkeypatch)
    client = TestClient(app)
    body = {"message": "please use-tools", "options": {"tool_results_format": "object"}}
    with client.stream("POST", "/v1/chat/stream", json=body) as resp:
        assert resp.status_code == 200
        events = _read_sse_events(resp)
    # Should emit only final event (fallback), with tool_results present
    tokens = [e for e in events if e.get("type") == "token"]
    finals = [e for e in events if e.get("type") == "final"]
    assert len(tokens) == 0
    assert len(finals) == 1
    final = finals[0]
    assert final.get("content") == "Final via fallback"
    assert isinstance(final.get("tool_results"), list)
    assert final["tool_results"][0]["tool"] == "calc"
