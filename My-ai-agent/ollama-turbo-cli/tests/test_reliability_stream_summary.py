from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, List

from fastapi.testclient import TestClient

from src.api.app import app
from src.api import router_v1


class _FakeInternalClient:
    def chat(self, **kwargs) -> Dict[str, Any]:
        # Deterministic generation for consensus runs
        return {"message": {"content": "X"}}


class FakeStreamChunk(dict):
    pass


class FakeOllamaClient:
    def __init__(self, api_key: str = "", enable_tools: bool = True, quiet: bool = True) -> None:
        self.api_key = api_key
        self.enable_tools = enable_tools
        self.quiet = quiet
        # Attributes referenced by router during consensus generation
        self.model = "gpt-oss:120b"
        self.max_output_tokens: Optional[int] = None
        self.ctx_size: Optional[int] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self._last_tool_results_structured: Optional[list] = None
        self._last_tool_results_strings: Optional[list] = None
        self._last_context_blocks: List[Dict[str, Any]] = []
        self._last_citations_map: Dict[str, Any] = {}
        # Provide an internal client for router-level consensus runs
        self.client = _FakeInternalClient()

    # ----- Non-streaming helpers used by the router -----
    def _inject_mem0_context(self, user_message: str) -> None:
        return None

    def _trim_history(self) -> None:
        return None

    def _set_idempotency_key(self, key: Optional[str]) -> None:
        return None

    def _clear_idempotency_key(self) -> None:
        return None

    def _resolve_keep_alive(self):
        # Router may call this during consensus generate_once; return None to skip header
        return None

    def _strip_harmony_markup(self, text: str) -> str:
        return text

    def _parse_harmony_tokens(self, text: str):
        # cleaned, tool_calls, final_text
        return text, None, None

    def _mem0_add_after_response(self, user_message: Optional[str], assistant_message: Optional[str]) -> None:
        return None

    def _prepare_reliability_context(self, user_message: str) -> None:
        # Simulate grounded context and a citations map
        self._last_context_blocks = [{"id": "doc-1", "tokens": 10}]
        self._last_citations_map = {"http://example.com": 1}

    # ----- Non-streaming fallback path -----
    def _handle_standard_chat(self) -> str:
        # Provide example tool results for fallback path
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

    # ----- Streaming path -----
    def _create_streaming_response(self) -> Iterable[Dict[str, Any]]:
        # Inspect last user msg to decide behavior
        msg = self.conversation_history[-1]["content"] if self.conversation_history else ""
        if "use-tools" in msg:
            # Emit a chunk that includes tool_calls to trigger router fallback
            yield FakeStreamChunk({
                "message": {
                    "content": "",
                    "tool_calls": [{"name": "calc", "arguments": {"x": 1}}],
                }
            })
            return
        # Normal token streaming path (no tools)
        for part in ["Hel", "lo"]:
            yield FakeStreamChunk({"message": {"content": part}})


def _patch_client(monkeypatch):
    monkeypatch.setattr(router_v1, "OllamaTurboClient", FakeOllamaClient)


def _read_sse_events_with_summary(resp) -> tuple[list[dict], Optional[dict]]:
    events: list[dict] = []
    summary: Optional[dict] = None
    expect_summary_data = False
    for chunk in resp.iter_lines():
        if not chunk:
            continue
        if isinstance(chunk, bytes):
            line = chunk.decode("utf-8", errors="ignore")
        else:
            line = chunk
        if line.startswith("event: "):
            # Named event, we care about 'summary'
            name = line[len("event: "):].strip()
            expect_summary_data = name == "summary"
            continue
        if line.startswith("data: "):
            payload = line[len("data: "):]
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if expect_summary_data:
                summary = obj
                expect_summary_data = False
            else:
                events.append(obj)
    return events, summary


def test_stream_tokens_final_and_summary_with_consensus_and_validator(monkeypatch):
    _patch_client(monkeypatch)
    client = TestClient(app)
    body = {
        "message": "hello",
        "ground": True,
        "check": "warn",
        "consensus": True,
        "k": 3,
    }
    with client.stream("POST", "/v1/chat/stream", json=body) as resp:
        assert resp.status_code == 200
        events, summary = _read_sse_events_with_summary(resp)

    # Expect at least one token and a final
    assert any(e.get("type") == "token" for e in events)
    finals = [e for e in events if e.get("type") == "final"]
    assert len(finals) == 1
    assert finals[0].get("content")

    # Summary expectations
    assert summary is not None
    assert isinstance(summary.get("grounded"), bool) and summary["grounded"] is True
    assert isinstance(summary.get("citations"), list) and len(summary["citations"]) >= 1
    # Validator wired with requested mode
    val = summary.get("validator")
    assert isinstance(val, dict) and val.get("mode") == "warn"
    # Consensus ran with k=3 and deterministic agree_rate
    cns = summary.get("consensus")
    assert isinstance(cns, dict) and cns.get("k") == 3
    assert cns.get("agree_rate") == 1.0


def test_stream_tool_calls_fallback_summary_skips_consensus(monkeypatch):
    _patch_client(monkeypatch)
    client = TestClient(app)
    body = {
        "message": "please use-tools",
        "ground": True,
        "check": "warn",
        "options": {"tool_results_format": "object"},
        "consensus": True,
        "k": 5,
    }
    with client.stream("POST", "/v1/chat/stream", json=body) as resp:
        assert resp.status_code == 200
        events, summary = _read_sse_events_with_summary(resp)

    # Should emit only final (fallback), with tool_results present
    tokens = [e for e in events if e.get("type") == "token"]
    finals = [e for e in events if e.get("type") == "final"]
    assert len(tokens) == 0
    assert len(finals) == 1
    assert finals[0].get("content") == "Final via fallback"
    assert isinstance(finals[0].get("tool_results"), list)

    # Summary expectations: consensus skipped (k=1, agree_rate=1.0)
    assert summary is not None
    cns = summary.get("consensus")
    assert isinstance(cns, dict)
    assert cns.get("k") == 1
    assert cns.get("agree_rate") == 1.0
    # Grounding fields still present
    assert isinstance(summary.get("grounded"), bool)
    assert isinstance(summary.get("citations"), list)
