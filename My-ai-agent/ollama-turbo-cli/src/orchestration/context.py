from __future__ import annotations

"""
Orchestration context protocol used by orchestration/ and streaming/ modules.

This formalizes the attributes and methods the orchestrator/runner rely on,
reducing duck-typing drift while remaining compatible with the existing
`OllamaTurboClient` implementation.
"""
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable

from typing import Any as _Any


@runtime_checkable
class OrchestrationContext(Protocol):
    # --- Core chat config/state ---
    model: str
    conversation_history: List[Dict[str, Any]]
    enable_tools: bool
    tools: List[Dict[str, Any]]
    tool_print_limit: int
    tool_context_cap: int
    quiet: bool
    show_trace: bool
    logger: Any

    # --- Sampling / options ---
    max_output_tokens: Optional[int]
    ctx_size: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]

    # --- Components ---
    adapter: _Any
    client: Any  # must expose chat(**kwargs)
    prompt: Any  # must expose reprompt_after_tools()

    # --- Streaming/tooling controls ---
    multi_round_tools: bool
    tool_max_rounds: int
    harmony: Any

    # --- Reliability state ---
    reliability: Dict[str, Any]

    # --- Optional last user message (used for Mem0 persistence) ---
    _last_user_message: Optional[str]
    _last_sent_messages: Optional[List[Dict[str, Any]]]

    # --- Methods used by orchestration/runner ---
    def _resolve_keep_alive(self) -> Optional[Any]: ...
    def _maybe_inject_reasoning(self, kwargs: Dict[str, Any]) -> None: ...
    def _parse_harmony_tokens(self, text: str) -> Tuple[str, List[Dict[str, Any]], Optional[str]]: ...
    def _strip_harmony_markup(self, text: str) -> str: ...
    def _trace(self, event: str) -> None: ...
    def _trace_mem0_presence(self, messages: Optional[List[Dict[str, Any]]], where: str) -> None: ...
    def _prepare_initial_messages_for_adapter(
        self,
        *,
        include_tools: bool,
        adapter_opts: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]: ...
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]: ...
    def _payload_for_tools(
        self, tool_results: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]]
    ) -> Tuple[Any, Optional[List[Dict[str, Any]]]]: ...
    def _mem0_add_after_response(self, user_message: Optional[str], assistant_message: Optional[str]) -> None: ...
    def _handle_standard_chat(self, *, _suppress_errors: bool = False) -> str: ...
