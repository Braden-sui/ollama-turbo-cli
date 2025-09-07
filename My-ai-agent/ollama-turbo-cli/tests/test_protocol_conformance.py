import types
from typing import Any, Dict, List, Tuple, Iterable

import pytest

from src.protocols.abc import ChatAdapter
from src.protocols.harmony import HarmonyAdapter
from src.protocols.deepseek import DeepSeekAdapter


ADAPTER_CASES = [
    (HarmonyAdapter, {"model": "gpt-oss:120b"}),
    (DeepSeekAdapter, {"model": "deepseek-chat"}),
]


def _sample_options() -> Dict[str, Any]:
    return {
        "max_tokens": 64,
        "temperature": 0.4,
        "top_p": 0.9,
        "stop": ["STOP"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "extra": {"num_ctx": 2048},
    }


def _sample_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo input",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
    ]


@pytest.mark.parametrize("adapter_cls,kwargs", ADAPTER_CASES)
def test_adapter_is_chatadapter(adapter_cls, kwargs):
    assert issubclass(adapter_cls, ChatAdapter)
    inst = adapter_cls(**kwargs)
    assert isinstance(inst.name, str) and inst.name
    caps = inst.capabilities
    assert isinstance(caps, dict)
    for key in ("supports_tools", "supports_reasoning", "first_system_only"):
        assert key in caps


@pytest.mark.parametrize("adapter_cls,kwargs", ADAPTER_CASES)
def test_map_options_and_format_initial(adapter_cls, kwargs):
    adapter: ChatAdapter = adapter_cls(**kwargs)
    opts = _sample_options()
    mapped = adapter.map_options(opts)  # type: ignore[arg-type]
    assert isinstance(mapped, dict)

    messages = [{"role": "user", "content": "hi"}]
    mem0_block = "Previous context from user history (use if relevant):\n- hello"
    out_msgs, overrides = adapter.format_initial_messages(
        messages, tools=_sample_tools(), options=opts, mem0_block=mem0_block
    )
    assert isinstance(out_msgs, list) and len(out_msgs) >= 1
    assert out_msgs[0].get("role") == "system"
    assert mem0_block in (out_msgs[0].get("content") or "")
    assert isinstance(overrides, dict)
    # When tools are provided, overrides should include them
    assert "tools" in overrides


@pytest.mark.parametrize("adapter_cls,kwargs", ADAPTER_CASES)
def test_format_reprompt_after_tools(adapter_cls, kwargs):
    adapter: ChatAdapter = adapter_cls(**kwargs)
    msgs = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "id": "call_1", "function": {"name": "echo", "arguments": {"x": 1}}},
                {"type": "function", "id": "call_2", "function": {"name": "sum", "arguments": {"a": 1, "b": 2}}},
            ],
        },
    ]
    tool_results = [
        {"tool": "echo", "status": "ok", "content": "ok", "metadata": {}, "error": None},
        {"tool": "sum", "status": "ok", "content": "3", "metadata": {}, "error": None},
    ]
    new_msgs, overrides = adapter.format_reprompt_after_tools(msgs, tool_results, options=_sample_options())
    assert isinstance(new_msgs, list) and len(new_msgs) >= len(msgs)
    tail = new_msgs[-2:]
    assert tail[0].get("role") == "tool" and tail[0].get("tool_call_id")
    assert tail[1].get("role") == "tool" and tail[1].get("tool_call_id")
    assert isinstance(overrides, dict)


@pytest.mark.parametrize("adapter_cls,kwargs", ADAPTER_CASES)
def test_parse_and_extract(adapter_cls, kwargs):
    adapter: ChatAdapter = adapter_cls(**kwargs)
    # Non-stream shape varies slightly per provider; supply both and accept either
    raw_harmony = {
        "message": {
            "content": "Hello",
            "tool_calls": [
                {"type": "function", "id": "call_1", "function": {"name": "echo", "arguments": {"x": 1}}}
            ],
        },
        "usage": {"prompt": 1, "completion": 1},
    }
    raw_deepseek = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "World", "tool_calls": []}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    out = adapter.parse_non_stream_response(raw_harmony)
    if not out.get("content"):
        out = adapter.parse_non_stream_response(raw_deepseek)
    assert "content" in out and "tool_calls" in out

    # Streaming: token-only minimal chunk
    token_chunk_h = {"message": {"content": "Hello"}}
    token_chunk_d = {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
    }
    evs = list(adapter.parse_stream_events(token_chunk_h))
    evs = evs or list(adapter.parse_stream_events(token_chunk_d))
    assert isinstance(evs, list)
    if evs:
        assert "type" in evs[0]

    # extract_tool_calls should return list
    calls = adapter.extract_tool_calls(raw_harmony) or adapter.extract_tool_calls(raw_deepseek)
    assert isinstance(calls, list)
