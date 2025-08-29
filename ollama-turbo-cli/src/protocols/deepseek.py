from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import json

from ..types import (
    AdapterCapabilities,
    AdapterOptions,
    ChatMessage,
    NonStreamResponse,
    NormalizedStreamEvent,
    ToolSpec,
)
from .base import ProtocolAdapter


class DeepSeekAdapter(ProtocolAdapter):
    """Minimal DeepSeek protocol adapter.

    Phase 4 scope:
    - Chat + streaming token parsing only (no tools).
    - Options mapping mirrors Harmony for now (will refine with official docs).
    - Mem0/system merge into first system message when provided.
    """

    def __init__(self, model: str, protocol: str = "deepseek") -> None:
        super().__init__(model=model, protocol="deepseek")
        # Accumulator for streaming tool call argument chunks keyed by tool_call id
        self._stream_tool_acc: Dict[str, Dict[str, Any]] = {}

    # --- Identity & capabilities ---
    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def capabilities(self) -> AdapterCapabilities:
        return {
            "supports_tools": True,
            "supports_reasoning": True,   # conservative default
            "first_system_only": True,    # keep Mem0/system in first system for consistency
        }

    # --- Prompt formatting ---
    def format_initial_messages(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolSpec]] = None,
        options: Optional[AdapterOptions] = None,
        mem0_block: Optional[str] = None,
    ) -> Tuple[List[ChatMessage], Dict[str, Any]]:
        out: List[ChatMessage] = list(messages or [])
        if not out or (out[0] or {}).get("role") != "system":
            out.insert(0, {"role": "system", "content": ""})
        if mem0_block:
            first = out[0]
            existing = first.get("content") or ""
            first["content"] = (existing + ("\n\n" if existing else "") + mem0_block)

        payload_overrides: Dict[str, Any] = {}
        mapped = self.map_options(options)
        if mapped:
            payload_overrides["options"] = mapped
        # Reset streaming tool-call accumulator for a fresh turn
        self._stream_tool_acc = {}
        # Include tools in provider payload when supplied
        if tools:
            payload_overrides["tools"] = tools
        return out, payload_overrides

    def format_reprompt_after_tools(
        self,
        messages: List[ChatMessage],
        tool_results: List[Dict[str, Any]],
        options: Optional[AdapterOptions] = None,
    ) -> Tuple[List[ChatMessage], Dict[str, Any]]:
        # Append OpenAI-style tool messages mapped to tool_call ids from the last assistant tool_calls
        out: List[ChatMessage] = list(messages or [])
        # Find last assistant message with tool_calls
        last_idx = None
        last_calls: List[Dict[str, Any]] = []
        for i in range(len(out) - 1, -1, -1):
            m = out[i]
            if (m.get("role") == "assistant") and (m.get("tool_calls")):
                last_idx = i
                last_calls = list(m.get("tool_calls") or [])
                break
        if last_idx is not None and last_calls:
            # Map tool_results by index to the corresponding tool_call id
            for i, tc in enumerate(last_calls):
                tc_id = str((tc or {}).get("id") or f"call_{i+1}")
                # Stringify tool result similar to client serialization rules
                content_str = ""
                if i < len(tool_results):
                    tr = tool_results[i] or {}
                    status = tr.get("status") or "ok"
                    if status == "error" and tr.get("error"):
                        err = tr.get("error") or {}
                        msg = err.get("message") or "error"
                        content_str = f"{tr.get('tool', 'tool')}: ERROR - {msg}"
                    else:
                        cont = tr.get("content")
                        if isinstance(cont, (dict, list)):
                            try:
                                content_str = json.dumps(cont, ensure_ascii=False)
                            except Exception:
                                content_str = str(cont)
                        else:
                            content_str = str(cont) if cont is not None else ""
                        tool_name = tr.get("tool", "tool")
                        content_str = f"{tool_name}: {content_str}"
                out.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": content_str,
                })
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
            mapped["num_predict"] = int(options["max_tokens"])  # temporary parity with Harmony
        if options.get("temperature") is not None:
            mapped["temperature"] = float(options["temperature"])  # 0..1
        if options.get("top_p") is not None:
            mapped["top_p"] = float(options["top_p"])  # 0..1
        if options.get("stop") is not None:
            mapped["stop"] = list(options["stop"] or [])
        extra = options.get("extra") or {}
        if isinstance(extra, dict):
            mapped.update(extra)
        return mapped

    # --- Parsing: streaming and non-streaming ---
    def parse_stream_events(self, raw_chunk: Any) -> Iterable[NormalizedStreamEvent]:
        """Parse OpenAI-compatible streaming chunk from DeepSeek.

        Expected shape (subset):
        {
          "id": "...",
          "object": "chat.completion.chunk",
          "choices": [
            {
              "delta": {
                "role": "assistant",
                "content": "text piece",
                "tool_calls": [
                  {"id": "call_1", "type": "function", "function": {"name": "fn", "arguments": "{...partial...}"}}
                ]
              },
              "index": 0,
              "finish_reason": None
            }
          ]
        }
        """
        if not isinstance(raw_chunk, dict):
            return []
        emitted = False
        choices = raw_chunk.get("choices") or []
        for ch in choices:
            delta = (ch or {}).get("delta") or {}
            # Token pieces
            piece = delta.get("content")
            if piece:
                emitted = True
                yield {"type": "token", "content": str(piece)}
            # Tool call deltas (Phase 5 will be fully supported; we forward what we get)
            for tc in (delta.get("tool_calls") or []) or []:
                if (tc or {}).get("type") == "function":
                    emitted = True
                    fn = (tc or {}).get("function") or {}
                    tc_id = str(tc.get("id") or "")
                    name = str(fn.get("name") or "")
                    arg_piece = fn.get("arguments")
                    # Normalize piece to string for accumulation
                    if isinstance(arg_piece, (dict, list)):
                        try:
                            piece_str = json.dumps(arg_piece, ensure_ascii=False)
                        except Exception:
                            piece_str = str(arg_piece)
                    else:
                        piece_str = str(arg_piece) if arg_piece is not None else ""
                    acc = self._stream_tool_acc.get(tc_id)
                    if not acc:
                        acc = {"name": name, "args_chunks": []}
                        self._stream_tool_acc[tc_id] = acc
                    if name and not acc.get("name"):
                        acc["name"] = name
                    if piece_str:
                        acc["args_chunks"].append(piece_str)
                    # Join accumulated argument fragments
                    args_joined = "".join(acc.get("args_chunks") or [])
                    # Try to parse JSON if complete; otherwise pass the concatenated string
                    args_value: Any = args_joined
                    if args_joined:
                        try:
                            args_value = json.loads(args_joined)
                        except Exception:
                            # keep as string until it becomes valid JSON
                            args_value = args_joined
                    yield {
                        "type": "tool_call",
                        "id": tc_id,
                        "name": acc.get("name") or name,
                        "arguments": args_value,
                    }
        # Legacy shape fallback: {"message": {"content": "...", "tool_calls": [...]}}
        if not emitted:
            msg = raw_chunk.get("message") or {}
            piece = msg.get("content")
            if piece:
                emitted = True
                yield {"type": "token", "content": str(piece)}
            for tc in (msg.get("tool_calls") or []) or []:
                if (tc or {}).get("type") == "function":
                    fn = (tc or {}).get("function") or {}
                    yield {
                        "type": "tool_call",
                        "id": str(tc.get("id") or ""),
                        "name": str(fn.get("name") or ""),
                        "arguments": fn.get("arguments"),
                    }
        return []

    def parse_non_stream_response(self, raw_response: Any) -> NonStreamResponse:
        """Parse OpenAI-compatible non-stream response.

        Expected shape (subset):
        {
          "id": "...",
          "object": "chat.completion",
          "choices": [
            { "index": 0, "message": {"role": "assistant", "content": "...", "tool_calls": [...]}, "finish_reason": "stop" }
          ],
          "usage": {...}
        }
        """
        try:
            choices = (raw_response or {}).get("choices") or []
            msg = (choices[0] or {}).get("message") if choices else {}
            msg = msg or {}
            content = str(msg.get("content") or "")
            tool_calls_raw = msg.get("tool_calls") or []
            tool_calls_norm: List[Dict[str, Any]] = []
            for tc in tool_calls_raw:
                if (tc or {}).get("type") == "function":
                    fn = (tc or {}).get("function") or {}
                    tool_calls_norm.append({
                        "id": str(tc.get("id") or ""),
                        "name": str(fn.get("name") or ""),
                        "arguments": fn.get("arguments"),
                    })
            return {
                "content": content,
                "tool_calls": tool_calls_norm,
                "usage": (raw_response or {}).get("usage") or {},
                "raw": raw_response,
            }
        except Exception:
            return {"content": "", "tool_calls": [], "usage": {}, "raw": raw_response}

    # --- Tooling ---
    def extract_tool_calls(self, raw_response: Any) -> List[Dict[str, Any]]:
        """Extract normalized tool calls from DeepSeek responses.

        Supports both OpenAI-compatible choices[].message.tool_calls and legacy
        {"message": {"tool_calls": [...]}} shapes.
        """
        try:
            if not isinstance(raw_response, dict):
                return []
            # Preferred OpenAI-compatible location
            choices = raw_response.get("choices") or []
            if choices:
                msg = (choices[0] or {}).get("message") or {}
                tc_list = msg.get("tool_calls") or []
                out: List[Dict[str, Any]] = []
                for tc in tc_list:
                    if (tc or {}).get("type") == "function":
                        fn = (tc or {}).get("function") or {}
                        out.append({
                            "id": str(tc.get("id") or ""),
                            "name": str(fn.get("name") or ""),
                            "arguments": fn.get("arguments"),
                        })
                if out:
                    return out
            # Legacy Ollama-style location
            msg2 = raw_response.get("message") or {}
            tc_list2 = msg2.get("tool_calls") or []
            out2: List[Dict[str, Any]] = []
            for tc in tc_list2:
                if (tc or {}).get("type") == "function":
                    fn = (tc or {}).get("function") or {}
                    out2.append({
                        "id": str(tc.get("id") or ""),
                        "name": str(fn.get("name") or ""),
                        "arguments": fn.get("arguments"),
                    })
            return out2
        except Exception:
            return []
