import json
from typing import Any, Dict, Optional, Tuple

import pytest

from src.plugins.reliable_chat import execute, _parse_stream


class DummyResp:
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._json


class FakeStreamResp:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode: bool = True):
        for line in self._lines:
            yield line

    # for parity with requests.Response context manager usage (not used here)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_consensus_normalization_various_inputs(monkeypatch):
    posted: Dict[str, Any] = {}

    def fake_post(url: str, headers: Dict[str, str], data: str, timeout: Tuple[int, int]):
        nonlocal posted
        posted = json.loads(data)
        return DummyResp({"content": "ok", "summary": {"status": "ok"}})

    import src.plugins.reliable_chat as rc
    monkeypatch.setattr(rc.requests, "post", fake_post)

    # A) consensus=true, k=None => consensus:3, no k
    execute("x", stream=False, auto=False, ground=False, cite=False, check="off", consensus=True, k=None)
    assert posted.get("consensus") == 3
    assert "k" not in posted

    # B) consensus=true, k=5 => consensus:5 and k:5
    execute("x", stream=False, auto=False, ground=False, cite=False, check="off", consensus=True, k=5)
    assert posted.get("consensus") == 5
    assert posted.get("k") == 5

    # C) consensus=4 => consensus:4, no k
    execute("x", stream=False, auto=False, ground=False, cite=False, check="off", consensus=4, k=None)
    assert posted.get("consensus") == 4
    assert "k" not in posted


def test_sse_parser_handles_event_message_and_default_channel():
    # event: message {"text":"ab"} then default-channel token {"type":"token","content":"c"}
    lines = [
        "event: message",
        'data: {"text":"ab"}',
        "",
        'data: {"type":"token","content":"c"}',
        "",
        "event: summary",
        'data: {"status":"ok","grounded":true,"citations":[1]}',
        "",
    ]
    resp = FakeStreamResp(lines)
    content, summary = _parse_stream(resp)  # type: ignore[arg-type]
    assert content == "abc"
    assert isinstance(summary, dict)
    assert summary.get("status") == "ok"
    assert summary.get("grounded") is True
    assert summary.get("citations") == [1]


def test_fail_closed_on_no_docs_when_cite_and_ground(monkeypatch):
    def fake_post(url: str, headers: Dict[str, str], data: str, timeout: Tuple[int, int]):
        return DummyResp({
            "content": "from memory",
            "summary": {"status": "no_docs", "grounded": False, "citations": []},
        })

    import src.plugins.reliable_chat as rc
    monkeypatch.setattr(rc.requests, "post", fake_post)

    out = execute("q", stream=False, auto=False, ground=True, cite=True, check="off")
    obj = json.loads(out)
    assert obj["ok"] is True
    assert obj["summary"]["grounded"] is False
    assert obj["summary"]["status"] == "no_docs"
    assert obj["summary"]["citations"] == []
    assert obj["content"].startswith("I can’t provide a cited answer")


def test_non_stream_prefers_server_summary(monkeypatch):
    summary = {"status": "ok", "grounded": True, "citations": ["doc"], "consensus": {"k": 1, "agree_rate": 1.0}}

    def fake_post(url: str, headers: Dict[str, str], data: str, timeout: Tuple[int, int]):
        return DummyResp({
            "content": "ok",
            "summary": summary,
        })

    import src.plugins.reliable_chat as rc
    monkeypatch.setattr(rc.requests, "post", fake_post)

    out = execute("q", stream=False, auto=False)
    obj = json.loads(out)
    assert obj["ok"] is True
    assert obj["content"] == "ok"
    assert obj["summary"] == summary
