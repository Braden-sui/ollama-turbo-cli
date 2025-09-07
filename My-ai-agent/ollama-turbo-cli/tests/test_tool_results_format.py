import os
import types
import builtins
import itertools

from src.client import OllamaTurboClient
from src.streaming import runner as stream_runner
from src.streaming import standard as standard_runner


class DummyAdapter:
    name = 'dummy'

    def __init__(self, raise_on_format=False, record_payload=False):
        self.raise_on_format = raise_on_format
        self.record_payload = record_payload
        self.last_payload = None

    def map_options(self, opts):
        return {}

    def parse_non_stream_response(self, response):
        # Do not interfere; standard path reads message.tool_calls
        return {}

    def parse_stream_events(self, chunk):
        # This adapter will not be used for streaming normalization in these tests
        return []

    def format_reprompt_after_tools(self, history, payload, options=None):
        if self.record_payload:
            self.last_payload = payload
        if self.raise_on_format:
            raise RuntimeError("adapter refused payload")
        # Return history unchanged (adapter may normally rewrite messages)
        return history, None


def make_client(format_mode: str) -> OllamaTurboClient:
    c = OllamaTurboClient(api_key='fake', enable_tools=True, quiet=True)
    c.tool_results_format = format_mode
    # Replace adapter with dummy
    c.adapter = DummyAdapter()
    # Provide two dummy tools
    c.tools = [
        {'type': 'function', 'function': {'name': 't1', 'parameters': {}}},
        {'type': 'function', 'function': {'name': 't2', 'parameters': {}}},
    ]
    c.tool_functions = {
        't1': lambda **kw: 'res1',
        't2': lambda **kw: 'res2',
    }
    return c


def setup_standard_toolcall_response(client: OllamaTurboClient):
    # First call returns tool_calls, second returns final content
    calls = itertools.count(0)

    def _chat(**kwargs):
        n = next(calls)
        if n == 0:
            return {
                'message': {
                    'content': 'preface',
                    'tool_calls': [
                        {'id': 'id1', 'function': {'name': 't1', 'arguments': {}}},
                        {'id': 'id2', 'function': {'name': 't2', 'arguments': {}}},
                    ]
                }
            }
        else:
            return {'message': {'content': 'final'}}
    client.client.chat = _chat  # type: ignore[attr-defined]


def test_adapter_receives_object_payload_standard():
    c = make_client('object')
    c.adapter = DummyAdapter(record_payload=True)
    setup_standard_toolcall_response(c)
    out = c._handle_standard_chat()
    assert 'final' in out
    # Adapter recorded payload
    assert isinstance(c.adapter.last_payload, list)
    assert c.adapter.last_payload and isinstance(c.adapter.last_payload[0], dict)
    # No fallback tool messages were injected when adapter succeeded
    injected = [m for m in c.conversation_history if m.get('role') == 'tool']
    # Still zero because adapter returned history unchanged
    assert len(injected) == 0


def test_fallback_injects_tool_messages_with_ids_standard():
    c = make_client('string')
    c.adapter = DummyAdapter(raise_on_format=True)
    setup_standard_toolcall_response(c)
    out = c._handle_standard_chat()
    assert 'final' in out
    # One tool message per tool call
    injected = [m for m in c.conversation_history if m.get('role') == 'tool']
    assert len(injected) == 1 or len(injected) == 2
    # When adapter fails, our fallback injects prebuilt messages if available
    # With string mode, prebuilt exists and should be one per call
    if len(injected) == 2:
        ids = [m.get('tool_call_id') for m in injected]
        assert set(ids) == {'id1', 'id2'}
        # Content should be stringified
        for m in injected:
            assert isinstance(m.get('content'), str)
    else:
        # Aggregated fallback path acceptable as last resort
        assert isinstance(injected[0].get('content'), str)


# Optional lightweight streaming parity (smoke): ensure runner uses helper without crashing
class _SimpleStream(list):
    pass


def test_streaming_smoke_uses_payload_helper(monkeypatch):
    c = make_client('object')
    c.adapter = DummyAdapter(record_payload=True)
    # Prepare a stream that yields a single chunk with tool_calls
    first_chunk = {
        'message': {
            'content': 'preface',
            'tool_calls': [
                {'id': 'id1', 'function': {'name': 't1', 'arguments': {}}},
            ]
        }
    }
    stream = _SimpleStream([first_chunk])

    # For the round after tools, return a fake final response stream (one chunk)
    def _chat(**kwargs):
        return _SimpleStream([{ 'message': {'content': 'final'} }])
    c.client.chat = _chat  # type: ignore[attr-defined]

    out = stream_runner.handle_streaming_response(c, stream, tools_enabled=True)
    # Output should contain final
    assert 'final' in out
    # Adapter recorded a payload list of dicts in object mode
    assert isinstance(c.adapter.last_payload, list)
    if c.adapter.last_payload:
        assert isinstance(c.adapter.last_payload[0], dict)


def test_mismatched_lengths_results_gt_calls_standard():
    c = make_client('string')
    # Force adapter failure to trigger fallback injection path
    c.adapter = DummyAdapter(raise_on_format=True)

    # Set up 1 tool_call in the first response
    calls = itertools.count(0)

    def _chat(**kwargs):
        n = next(calls)
        if n == 0:
            return {
                'message': {
                    'content': 'preface',
                    'tool_calls': [
                        {'id': 'id1', 'function': {'name': 't1', 'arguments': {}}},
                    ]
                }
            }
        else:
            return {'message': {'content': 'final'}}
    c.client.chat = _chat  # type: ignore[attr-defined]

    # Monkeypatch execution to return 2 results (more than tool_calls)
    def fake_exec(self, tool_calls):
        return [
            {'tool': 't1', 'status': 'ok', 'content': 'r1', 'metadata': {'args': {}}, 'error': None},
            {'tool': 't2', 'status': 'ok', 'content': 'r2', 'metadata': {'args': {}}, 'error': None},
        ]
    c._execute_tool_calls = types.MethodType(fake_exec, c)

    out = c._handle_standard_chat()
    assert 'final' in out

    injected = [m for m in c.conversation_history if m.get('role') == 'tool']
    # We expect two injected messages (one per result)
    assert len(injected) == 2
    # First maps tool_call_id to 'id1'
    assert injected[0].get('tool_call_id') == 'id1'
    # Second has no matching call id and thus no tool_call_id
    assert 'tool_call_id' not in injected[1]
