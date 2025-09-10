import os
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.orchestration.orchestrator import ChatTurnOrchestrator


class FakeAdapter:
    def format_reprompt_after_tools(self, messages, payload, options=None):
        # Echo through unchanged; orchestrator wrapper will append as needed
        return list(messages or []), {}


class FakeLogger:
    def debug(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass


class FakeCtx:
    def __init__(self):
        self.model = "test-model"
        self.conversation_history: List[Dict[str, Any]] = []
        self.enable_tools = False
        self.tools: List[Dict[str, Any]] = []
        self.tool_print_limit = 0
        self.tool_context_cap = 0
        self.quiet = True
        self.show_trace = False
        self.logger = FakeLogger()
        self.max_output_tokens = None
        self.ctx_size = None
        self.temperature = None
        self.top_p = None
        self.presence_penalty = None
        self.frequency_penalty = None
        self.adapter = FakeAdapter()
        self.client = types.SimpleNamespace(chat=lambda **kwargs: {'message': {'content': 'ok'}})
        self.prompt = types.SimpleNamespace(reprompt_after_tools=lambda: "reprompt")
        self.multi_round_tools = False
        self.tool_max_rounds = 1
        self.harmony = types.SimpleNamespace(last_analysis=None)
        self.reliability: Dict[str, Any] = {'ground': False, 'cite': False, 'check': 'off'}
        self._last_user_message: Optional[str] = None
        self._last_sent_messages: Optional[List[Dict[str, Any]]] = None
        self._last_context_blocks: List[Dict[str, Any]] = []
        self._last_citations_map: Dict[str, Any] = {}
        self.flags: Dict[str, Any] = {'ground_degraded': False}
        self.show_snippets = False

    # Methods required by Protocol
    def _resolve_keep_alive(self) -> Optional[Any]:
        return None
    def _maybe_inject_reasoning(self, kwargs: Dict[str, Any]) -> None:
        return None
    def _parse_harmony_tokens(self, text: str) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
        return text, [], None
    def _strip_harmony_markup(self, text: str) -> str:
        return text
    def _trace(self, event: str) -> None:
        return None
    def _trace_mem0_presence(self, messages: Optional[List[Dict[str, Any]]], where: str) -> None:
        return None
    def _prepare_initial_messages_for_adapter(self, *, include_tools: bool, adapter_opts: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        return list(self.conversation_history), {}
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []
    def _payload_for_tools(self, tool_results: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]]):
        return {}, None
    def _mem0_add_after_response(self, user_message: Optional[str], assistant_message: Optional[str]) -> None:
        return None
    def _handle_standard_chat(self, *, _suppress_errors: bool = False) -> str:
        return ""


def test_date_prefix_on_time_sensitive():
    ctx = FakeCtx()
    ctx._last_user_message = "latest CPI this week"
    orch = ChatTurnOrchestrator()
    os.environ['CLI_EXPERIMENTAL_VIS'] = '1'
    out = orch.finalize_reliability_streaming(ctx, "The CPI is ...", tools_used=False)
    assert out.lower().startswith("as of ")


def test_date_prefix_off_static():
    ctx = FakeCtx()
    ctx._last_user_message = "Explain photosynthesis"
    orch = ChatTurnOrchestrator()
    os.environ['CLI_EXPERIMENTAL_VIS'] = '1'
    out = orch.finalize_reliability_streaming(ctx, "Plants convert light...", tools_used=False)
    assert not out.lower().startswith("as of ")


def test_ground_auto_degrade_notice(capsys):
    ctx = FakeCtx()
    ctx.reliability.update({'ground': True, 'cite': True, 'check': 'off'})
    ctx.quiet = False
    # Simulate structured tool result: web_research with no citations
    ctx._last_tool_results_structured = [
        {'tool': 'web_research', 'status': 'ok', 'content': {'citations': []}}
    ]
    orch = ChatTurnOrchestrator()
    os.environ['CLI_EXPERIMENTAL_VIS'] = '1'
    orch.format_reprompt_after_tools(ctx, payload={}, prebuilt_msgs=None, streaming=False)
    captured = capsys.readouterr()
    assert "No grounding context found" in captured.out
    assert ctx.flags.get('ground_degraded') is True
    assert ctx.reliability.get('ground') is False
