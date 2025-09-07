import json
from typing import Dict

import pytest

from src.protocols.harmony import HarmonyAdapter


def make_adapter() -> HarmonyAdapter:
    return HarmonyAdapter(model="gpt-oss:120b")


def test_strip_markup_and_analysis_removed():
    hp = make_adapter()._hp
    text = (
        "<|channel|>analysis<|message|>secret chain-of-thought<|end|>"
        "Intro text."
        "<|channel|>final<|message|>Visible answer<|end|>"
    )
    cleaned = hp.strip_markup(text)
    # analysis content must be removed entirely
    assert "secret chain-of-thought" not in cleaned
    # final markup stripped, user-visible text remains
    assert "Visible answer" in cleaned
    assert "<|channel|>" not in cleaned


def test_parse_tokens_extracts_tool_and_final():
    hp = make_adapter()._hp
    text = (
        "<|channel|>commentary to=functions.web_fetch\n"
        "<|message|>{\"url\": \"https://example.com\"}<|call|>"
        "<|channel|>final\n<|message|>Answer body<|end|>"
    )
    cleaned, calls, final_seg = hp.parse_tokens(text)
    assert isinstance(calls, list) and len(calls) == 1
    tc = calls[0]
    assert tc.get("type") == "function"
    fn = tc.get("function") or {}
    assert fn.get("name") == "web_fetch"
    assert fn.get("arguments", {}).get("url") == "https://example.com"
    assert final_seg == "Answer body"
    # cleaned text should not contain commentary/tool JSON
    assert "functions.web_fetch" not in cleaned


def test_parse_non_stream_response_without_tool_calls():
    adapter = make_adapter()
    raw = {
        "message": {
            # No canonical tool_calls in payload; must be parsed from content
            "content": (
                "<|channel|>commentary to=functions.calc\n<|message|>{\"x\":1,\"y\":2}<|call|>"
                "<|channel|>final\n<|message|>3<|end|>"
            )
        },
        "usage": {"prompt": 10, "completion": 2},
    }
    out = adapter.parse_non_stream_response(raw)
    assert out["content"] == "3"
    assert isinstance(out["tool_calls"], list) and len(out["tool_calls"]) == 1
    tc = out["tool_calls"][0]
    assert tc["name"] == "calc"
    assert tc["arguments"] == {"x": 1, "y": 2}


def test_parse_stream_events_emits_tokens_and_tool_calls():
    adapter = make_adapter()
    chunk: Dict = {
        "message": {
            "content": "Hello <|channel|>final<|message|>World<|end|>",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {"name": "ping", "arguments": {"x": 1}},
                }
            ],
        }
    }
    events = list(adapter.parse_stream_events(chunk))
    # Should include a token event with markup stripped ("Hello World")
    token_evs = [e for e in events if e["type"] == "token"]
    assert token_evs and "Hello" in token_evs[0]["content"]
    assert "<|channel|>" not in token_evs[0]["content"]
    # And a tool_call event
    tc_evs = [e for e in events if e["type"] == "tool_call"]
    assert tc_evs and tc_evs[0]["name"] == "ping"
    assert tc_evs[0]["arguments"] == {"x": 1}
