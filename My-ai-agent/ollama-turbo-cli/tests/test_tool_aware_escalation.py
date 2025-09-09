from types import SimpleNamespace

from src.orchestration.orchestrator import ChatTurnOrchestrator


class DummyCtx:
    def __init__(self):
        self.model = 'gpt-oss:120b'
        self.conversation_history = [{'role': 'user', 'content': 'research this'}]
        self.enable_tools = True
        self.tools = []
        self.tool_print_limit = 200
        self.tool_context_cap = 4000
        self.quiet = True
        self.show_trace = False
        self.logger = SimpleNamespace(debug=lambda *a, **k: None, error=lambda *a, **k: None)
        self.max_output_tokens = None
        self.ctx_size = None
        self.temperature = None
        self.top_p = None
        self.presence_penalty = None
        self.frequency_penalty = None
        self.adapter = SimpleNamespace(
            map_options=lambda opts: {},
            format_reprompt_after_tools=lambda hist, payload, options=None: (hist, {}),
            parse_non_stream_response=lambda resp: {'content': '', 'tool_calls': []},
        )
        self.client = SimpleNamespace(chat=lambda **k: {'message': {'content': ''}})
        self.prompt = SimpleNamespace(reprompt_after_tools=lambda: 'reprompt')
        self.multi_round_tools = False
        self.tool_max_rounds = 1
        self.harmony = SimpleNamespace()
        self.reliability = {'ground': False, 'cite': False, 'check': 'off'}
        self._last_user_message = 'research this'
        self._last_sent_messages = None
        self._resolve_keep_alive = lambda: None
        self._maybe_inject_reasoning = lambda kwargs: None
        self._parse_harmony_tokens = lambda text: (text, [], text)
        self._strip_harmony_markup = lambda text: text
        self._trace = lambda ev: None
        self._trace_mem0_presence = lambda messages, where: None
        self._prepare_initial_messages_for_adapter = lambda include_tools, adapter_opts: (self.conversation_history, {})
        self._execute_tool_calls = lambda tool_calls: [{'tool': 'web_research', 'status': 'ok', 'content': {'citations': []}, 'metadata': {}}]
        self._payload_for_tools = lambda tool_results, tool_calls: (tool_results, None)
        self._mem0_add_after_response = lambda u, a: None


def test_tool_aware_escalation_sets_flag():
    ctx = DummyCtx()
    orch = ChatTurnOrchestrator()
    # Simulate a tool call to web_research to trigger escalation path
    ctx.adapter.parse_non_stream_response = lambda resp: {'content': '', 'tool_calls': [
        {'type': 'function', 'id': '1', 'function': {'name': 'web_research', 'arguments': {}}}
    ]}
    # Run one loop; handle_standard_chat expects a non-empty response after tools; our dummy returns '' so it will finalize with ''
    out = orch.handle_standard_chat(ctx, _suppress_errors=True)
    meta = getattr(ctx, '_turn_mode_meta', {})
    assert meta.get('mode_forced_by_tools') is True

