import json
import types

from src.client import OllamaTurboClient
from src.tools_runtime.executor import ToolRuntimeExecutor


def test_json_string_arguments_are_parsed():
    c = OllamaTurboClient(api_key='fake', enable_tools=True, quiet=True)
    received = {}

    def tjson(**kwargs):
        received.update(kwargs)
        return 'ok'

    c.tool_functions = {'tjson': tjson}
    tool_calls = [{
        'type': 'function',
        'function': {
            'name': 'tjson',
            'arguments': json.dumps({'a': 1, 'b': 'x'})
        }
    }]
    results = c._execute_tool_calls(tool_calls)
    assert results and results[0]['status'] == 'ok'
    # Metadata.args should be the parsed dict
    md = results[0].get('metadata') or {}
    args = md.get('args') or {}
    assert args == {'a': 1, 'b': 'x'}
    # And the function saw parsed kwargs
    assert received == {'a': 1, 'b': 'x'}


def test_serialize_to_string_truncates_cli_output():
    class Ctx:
        tool_print_limit = 10
    long = 'a' * 100
    tr = {'tool': 'demo', 'status': 'ok', 'content': long, 'metadata': {}, 'error': None}
    out = ToolRuntimeExecutor.serialize_to_string(Ctx(), tr)
    assert out.startswith('demo: ')
    # Should be shorter than the full content and include a prefix of the long string
    assert len(out) < len('demo: ' + long)
    assert 'a' in out


def test_metadata_has_index_and_duration():
    c = OllamaTurboClient(api_key='fake', enable_tools=True, quiet=True)
    c.tool_functions = {
        't1': lambda **kw: 'r1',
        't2': lambda **kw: 'r2',
    }
    tool_calls = [
        {'type': 'function', 'function': {'name': 't1', 'arguments': {}}},
        {'type': 'function', 'function': {'name': 't2', 'arguments': {}}},
    ]
    results = c._execute_tool_calls(tool_calls)
    assert len(results) == 2
    for idx, r in enumerate(results, start=1):
        md = r.get('metadata') or {}
        assert md.get('index') == idx
        assert isinstance(md.get('duration_ms'), int)
        assert md.get('duration_ms') >= 0
