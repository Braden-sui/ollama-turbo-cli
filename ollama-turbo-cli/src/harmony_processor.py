"""
HarmonyProcessor encapsulates Harmony token parsing and markup stripping logic.

This preserves the exact behavior that previously lived in
`OllamaTurboClient._strip_harmony_markup` and `_parse_harmony_tokens`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class HarmonyProcessor:
    """Parse Harmony tool-call tokens and strip Harmony markup.

    Exposed methods:
    - strip_markup(text): remove Harmony control/channel tokens while preserving content
      (analysis blocks are removed entirely to avoid leaking chain-of-thought)
    - parse_tokens(text): extract tool calls and final-channel text; remove analysis blocks
      from cleaned text and store their content in `self.last_analysis` for tracing.
    """

    def strip_markup(self, text: str) -> str:
        """Remove Harmony channel/control tokens from text, preserving natural content.

        This strips tokens like <|channel|>commentary, <|channel|>final, <|message|>, <|call|>, <|end|>.
        """
        try:
            if not text:
                return text
            # Remove entire analysis blocks to prevent chain-of-thought leakage
            t = re.sub(
                r"<\|channel\|>\s*analysis\b.*?<\|message\|>.*?(?:<\|end\|>|<\|channel\|>|$)",
                "",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            # Remove channel headers (commentary/final/analysis) and inline markers
            t = re.sub(r"<\|channel\|>\s*commentary[^\n<]*", "", t, flags=re.IGNORECASE)
            t = re.sub(r"<\|channel\|>\s*final[^\n<]*", "", t, flags=re.IGNORECASE)
            t = re.sub(r"<\|channel\|>\s*analysis[^\n<]*", "", t, flags=re.IGNORECASE)
            t = re.sub(r"<\|message\|>", "", t, flags=re.IGNORECASE)
            t = re.sub(r"<\|call\|>", "", t, flags=re.IGNORECASE)
            t = re.sub(r"<\|end\|>", "", t, flags=re.IGNORECASE)
            return t
        except Exception:
            return text

    def parse_tokens(self, text: str) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
        """Parse Harmony tool-call and final-channel tokens from text.

        Returns (cleaned_text, tool_calls, final_text)
        - cleaned_text: input with tool-call segments removed and markup stripped
        - tool_calls: list of OpenAI-style tool_call dicts
        - final_text: last final-channel message content if present
        """
        tool_calls: List[Dict[str, Any]] = []
        final_text: Optional[str] = None
        # Reset captured analysis for this call
        try:
            self.last_analysis = None  # type: ignore[attr-defined]
        except Exception:
            pass
        if not text:
            return text, tool_calls, final_text

        # More permissive matcher for commentary tool calls. Capture fname and the opening brace.
        # Examples accepted:
        #   to=functions.web_search
        #   to=functions.web.run.search_query
        #   to=web-search
        tc_re = re.compile(
            r"<\|channel\|>\s*commentary\b[^<]*?to=(?:functions\.)?(?P<fname>[a-zA-Z0-9_.:-]+)[^<]*?<\|message\|>(?P<json>\{)",
            flags=re.IGNORECASE | re.DOTALL,
        )

        def _extract_balanced_json(s: str, start_idx: int) -> Tuple[Optional[str], int]:
            depth = 0
            i = start_idx
            in_str = False
            esc = False
            while i < len(s):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            return s[start_idx:i+1], i + 1
                i += 1
            return None, start_idx

        # Manually scan and remove matched commentary tool segments while extracting JSON args
        cleaned_parts: List[str] = []
        idx = 0
        for m in tc_re.finditer(text):
            # Append any literal text before the match
            cleaned_parts.append(text[idx:m.start()])
            name = m.group('fname')
            json_start = m.start('json')
            json_blob, after = _extract_balanced_json(text, json_start)
            args: Dict[str, Any] = {}
            if json_blob:
                try:
                    parsed = json.loads(json_blob)
                    if isinstance(parsed, dict):
                        args = parsed
                except Exception:
                    args = {}
            # Consume an optional trailing <|call|> after the JSON
            end_pos = after
            call_m = re.match(r"\s*<\|call\|>", text[after:], flags=re.IGNORECASE)
            if call_m:
                end_pos = after + call_m.end()
            # Record tool call
            tool_calls.append({
                'type': 'function',
                'id': f"call_h_{len(tool_calls)+1}",
                'function': {
                    'name': name,
                    'arguments': args,
                }
            })
            # Skip the entire matched commentary segment
            idx = end_pos
        # Append the remainder and join all pieces
        cleaned_parts.append(text[idx:])
        cleaned = ''.join(cleaned_parts)

        # Extract and remove analysis channel content (accumulate all segments)
        analysis_segments: List[str] = []
        analysis_re = re.compile(
            r"<\|channel\|>\s*analysis\b.*?<\|message\|>(?P<msg>.*?)(?:<\|end\|>|<\|channel\|>|$)",
            re.IGNORECASE | re.DOTALL,
        )

        def _analysis_repl(m):
            msg = m.group('msg') if 'msg' in m.groupdict() else ''
            if msg:
                analysis_segments.append(msg)
            return ""  # remove this segment from cleaned text

        cleaned = analysis_re.sub(_analysis_repl, cleaned)

        # Extract final channel content (use the last occurrence if multiple)
        final_re = re.compile(r"<\|channel\|>\s*final\b.*?<\|message\|>(?P<msg>.*?)(?:<\|end\|>|$)", re.IGNORECASE | re.DOTALL)
        for m in final_re.finditer(cleaned):
            final_text = m.group('msg')

        # Persist captured analysis for tracing purposes
        try:
            joined = "\n".join(s.strip() for s in analysis_segments if s.strip())
            self.last_analysis = joined or None  # type: ignore[attr-defined]
        except Exception:
            pass

        # Remove any remaining markup tokens
        cleaned = self.strip_markup(cleaned)

        return cleaned, tool_calls, final_text
