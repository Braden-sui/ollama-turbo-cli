from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..harmony_processor import HarmonyProcessor
from ..types import (
    AdapterCapabilities,
    AdapterOptions,
    ChatMessage,
    NonStreamResponse,
    NormalizedStreamEvent,
    ToolSpec,
)
from .base import ProtocolAdapter


class HarmonyAdapter(ProtocolAdapter):
    """Adapter for the Harmony protocol (gpt-oss:120b default).

    - Injects Mem0/system as needed in the first system message.
    - Parses Harmony markup and tool-call commentary/final channels via HarmonyProcessor.
    - Maps generic options to Ollama-compatible request options.
    """

    def __init__(self, model: str, protocol: str = "harmony") -> None:
        super().__init__(model=model, protocol="harmony")
        self._hp = HarmonyProcessor()

    # --- Identity & capabilities ---
    @property
    def name(self) -> str:
        return "harmony"

    @property
    def capabilities(self) -> AdapterCapabilities:
        # Providers using Harmony often only honor the first system message for instruction blocks
        return {
            "supports_tools": True,
            "supports_reasoning": True,
            "first_system_only": True,
        }

    # --- Prompt formatting ---
    def format_initial_messages(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolSpec]] = None,
        options: Optional[AdapterOptions] = None,
        mem0_block: Optional[str] = None,
    ) -> Tuple[List[ChatMessage], Dict[str, Any]]:
        # Ensure there is a first system message
        out: List[ChatMessage] = list(messages or [])
        if not out or (out[0] or {}).get("role") != "system":
            out.insert(0, {"role": "system", "content": ""})
        # Merge Mem0 block into the first system message content if provided
        if mem0_block:
            first = out[0]
            first["content"] = (first.get("content") or "") + ("\n\n" if first.get("content") else "") + mem0_block
        # Map generic options to provider options
        payload_overrides: Dict[str, Any] = {}
        mapped = self.map_options(options)
        if mapped:
            payload_overrides["options"] = mapped
        if tools:
            payload_overrides["tools"] = tools
        return out, payload_overrides

    def format_reprompt_after_tools(
        self,
        messages: List[ChatMessage],
        tool_results: List[Dict[str, Any]],
        options: Optional[AdapterOptions] = None,
    ) -> Tuple[List[ChatMessage], Dict[str, Any]]:
        out: List[ChatMessage] = list(messages or [])
        # For Harmony, the caller typically appends a user reprompt already; adapter may pass through
        payload_overrides: Dict[str, Any] = {}
        mapped = self.map_options(options)
        if mapped:
            payload_overrides["options"] = mapped
        return out, payload_overrides

    # --- Options mapping ---
    def map_options(self, options: Optional[AdapterOptions]) -> Dict[str, Any]:
        if not options:
            return {}
        mapped: Dict[str, Any] = {}
        if options.get("max_tokens") is not None:
            mapped["num_predict"] = int(options["max_tokens"])  # Ollama-compatible
        if options.get("temperature") is not None:
            mapped["temperature"] = float(options["temperature"])  # 0..1
        if options.get("top_p") is not None:
            mapped["top_p"] = float(options["top_p"])  # 0..1
        if options.get("stop") is not None:
            mapped["stop"] = list(options["stop"] or [])
        # Pass-through any additional fields for future compatibility
        extra = options.get("extra") or {}
        if isinstance(extra, dict):
            mapped.update(extra)
        return mapped

    # --- Parsing: streaming and non-streaming ---
    def parse_stream_events(self, raw_chunk: Any) -> Iterable[NormalizedStreamEvent]:
        # Expecting Ollama-style streaming chunk: { 'message': { 'content': '...', 'tool_calls': [...] }, ... }
        if not isinstance(raw_chunk, dict):
            return []
        msg = raw_chunk.get("message") or {}
        events: List[NormalizedStreamEvent] = []
        content_piece = msg.get("content")
        if content_piece:
            cleaned_piece = self._hp.strip_markup(str(content_piece))
            if cleaned_piece:
                events.append({"type": "token", "content": cleaned_piece})
        # Tool calls may be streamed incrementally; emit a normalized tool_call event for each update
        tc_list = msg.get("tool_calls") or []
        for tc in tc_list:
            # tc shape: { 'type': 'function', 'id': '...', 'function': { 'name': str, 'arguments': dict|str } }
            fn = (tc or {}).get("function") or {}
            events.append({
                "type": "tool_call",
                "id": str(tc.get("id") or ""),
                "name": str(fn.get("name") or ""),
                "arguments": fn.get("arguments"),
            })
        return events

    def parse_non_stream_response(self, raw_response: Any) -> NonStreamResponse:
        # Expecting Ollama non-streaming response: { 'message': { 'content': str, 'tool_calls': [...] }, 'usage': {...} }
        try:
            msg = (raw_response or {}).get("message") or {}
            content = str(msg.get("content") or "")
            tool_calls_raw = msg.get("tool_calls") or []
            # If provider did not canonicalize tool calls, parse from content via HarmonyProcessor
            cleaned = content
            parsed_calls: List[Dict[str, Any]] = []
            try:
                if content and not tool_calls_raw:
                    cleaned, hp_calls, final_seg = self._hp.parse_tokens(content)
                    # Normalize hp_calls (OpenAI style) to our ToolCall list below
                    tool_calls_raw = hp_calls or []
                    if final_seg:
                        cleaned = final_seg or cleaned
            except Exception:
                pass

            norm_calls = self._normalize_openai_style_tool_calls(tool_calls_raw)
            return {
                "content": self._hp.strip_markup(cleaned),
                "tool_calls": norm_calls,
                "usage": (raw_response or {}).get("usage") or {},
                "raw": raw_response,
            }
        except Exception:
            return {"content": "", "tool_calls": [], "usage": {}, "raw": raw_response}

    # --- Tooling ---
    def extract_tool_calls(self, raw_response: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_response, dict):
            return []
        msg = raw_response.get("message") or {}
        tool_calls_raw = msg.get("tool_calls") or []
        if tool_calls_raw:
            return self._normalize_openai_style_tool_calls(tool_calls_raw)
        content = msg.get("content") or ""
        try:
            if content:
                _, hp_calls, _ = self._hp.parse_tokens(str(content))
                return self._normalize_openai_style_tool_calls(hp_calls or [])
        except Exception:
            pass
        return []

    # --- Helpers ---
    def _normalize_openai_style_tool_calls(self, tool_calls_openai: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, tc in enumerate(tool_calls_openai or [], 1):
            fn = (tc or {}).get("function") or {}
            out.append({
                "id": str(tc.get("id") or f"call_h_{i}"),
                "name": str(fn.get("name") or ""),
                "arguments": fn.get("arguments"),
            })
        return out
