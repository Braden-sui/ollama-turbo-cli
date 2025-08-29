import pytest

from src.protocols.deepseek import DeepSeekAdapter
from src.protocols.factory import get_adapter


def make_adapter() -> DeepSeekAdapter:
    return DeepSeekAdapter(model="deepseek-chat")


def test_format_initial_messages_merges_mem0_into_first_system():
    adapter = make_adapter()
    messages = [
        {"role": "user", "content": "hi"},
    ]
    mem0_block = "[mem0] instructions here"
    tools = [
        {
            "name": "hello",
            "description": "say hi",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        }
    ]
    out_msgs, overrides = adapter.format_initial_messages(messages, tools=tools, mem0_block=mem0_block)
    assert out_msgs[0]["role"] == "system"
    assert mem0_block in (out_msgs[0]["content"] or "")
    # Tools should be forwarded in overrides for the initial request
    assert "tools" in overrides


def test_map_options_basic():
    adapter = make_adapter()
    opts = {
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["STOP"],
        "extra": {"num_ctx": 2048},
    }
    mapped = adapter.map_options(opts)  # type: ignore[arg-type]
    assert mapped["num_predict"] == 128
    assert mapped["temperature"] == pytest.approx(0.2)
    assert mapped["top_p"] == pytest.approx(0.9)
    assert mapped["stop"] == ["STOP"]
    assert mapped["num_ctx"] == 2048


def test_capabilities_support_tools():
    adapter = make_adapter()
    caps = adapter.capabilities
    assert caps["supports_tools"] is True


def test_stream_assembly_tool_call_arguments():
    adapter = make_adapter()
    # Simulate two streaming chunks where arguments arrive in pieces
    chunk1 = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{\"x\":"}
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    chunk2 = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "1}"}
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    ev1 = list(adapter.parse_stream_events(chunk1))
    ev2 = list(adapter.parse_stream_events(chunk2))
    # First event yields a partial tool_call string
    t1 = [e for e in ev1 if e.get("type") == "tool_call"][0]
    assert t1["id"] == "call_1"
    assert t1["name"] == "echo"
    assert isinstance(t1["arguments"], str)
    # Second event should have joined arguments and attempted to parse JSON
    t2 = [e for e in ev2 if e.get("type") == "tool_call"][0]
    assert t2["id"] == "call_1"
    assert t2["name"] == "echo"
    # Accept either parsed dict or full joined string if not parseable yet
    assert (isinstance(t2["arguments"], dict) and t2["arguments"].get("x") == 1) or (t2["arguments"] == '{"x":1}')


def test_non_stream_extract_tool_calls():
    adapter = make_adapter()
    raw = {
        "id": "cmpl-2",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_2", "type": "function", "function": {"name": "sum", "arguments": {"a": 1, "b": 2}}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    out = adapter.parse_non_stream_response(raw)
    tcs = out["tool_calls"]
    assert len(tcs) == 1 and tcs[0]["name"] == "sum"
    # Also test extract_tool_calls
    ext = adapter.extract_tool_calls(raw)
    assert len(ext) == 1 and ext[0]["id"] == "call_2"


def test_format_reprompt_after_tools_appends_tool_messages_with_ids():
    adapter = make_adapter()
    # last assistant message with tool_calls
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "id": "call_1", "function": {"name": "echo", "arguments": {"x": 1}}},
            {"type": "function", "id": "call_2", "function": {"name": "sum", "arguments": {"a": 1, "b": 2}}},
        ]},
    ]
    tool_results = [
        {"tool": "echo", "status": "ok", "content": "echo:1", "metadata": {}, "error": None},
        {"tool": "sum", "status": "ok", "content": "3", "metadata": {}, "error": None},
    ]
    new_msgs, _ = adapter.format_reprompt_after_tools(msgs, tool_results)
    # Expect two tool messages appended with matching tool_call_id
    tail = new_msgs[-2:]
    assert tail[0]["role"] == "tool" and tail[0]["tool_call_id"] == "call_1"
    assert tail[1]["role"] == "tool" and tail[1]["tool_call_id"] == "call_2"


def test_parse_stream_events_tokens_only():
    adapter = make_adapter()
    chunk = {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "Hello DeepSeek"}, "finish_reason": None}
        ],
    }
    events = list(adapter.parse_stream_events(chunk))
    assert events and events[0]["type"] == "token"
    assert "Hello" in events[0]["content"]
    # No tool_call events in this chunk
    assert not [e for e in events if e["type"] == "tool_call"]


def test_parse_non_stream_response_basic():
    adapter = make_adapter()
    raw = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Final answer"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    out = adapter.parse_non_stream_response(raw)
    assert out["content"] == "Final answer"
    assert out["tool_calls"] == []
    assert out["usage"]["total_tokens"] == 3


def test_factory_returns_deepseek_adapter():
    adapter = get_adapter(model="deepseek-chat", protocol="auto")
    assert isinstance(adapter, DeepSeekAdapter)


def test_parse_stream_events_legacy_message_shape():
    adapter = make_adapter()
    # Legacy chunk shape used by some proxies/backends
    chunk = {
        "message": {
            "content": "Legacy token",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "foo", "arguments": "{\"x\":1}"},
                }
            ],
        }
    }
    events = list(adapter.parse_stream_events(chunk))
    assert any(e.get("type") == "token" and "Legacy" in e.get("content", "") for e in events)
    # Tool call forwarded in legacy mode
    tc_events = [e for e in events if e.get("type") == "tool_call"]
    assert tc_events and tc_events[0]["name"] == "foo"
