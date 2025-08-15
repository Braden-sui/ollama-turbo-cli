"""
Stream processor service - Domain service for handling streaming responses.
Pure business logic for stream processing with tool call detection.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Iterator, Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

from ..models.tool import ToolCall, extract_tool_calls_from_response
from .harmony_formatter import HarmonyFormatter


@dataclass
class StreamState:
    """Tracks state during streaming processing."""
    accumulated_content: str = ""
    tool_calls_detected: List[Dict[str, Any]] = field(default_factory=list)
    in_tool_call: bool = False
    current_tool_buffer: str = ""
    complete: bool = False
    error: Optional[str] = None


class StreamProcessor:
    """Domain service for processing streaming LLM responses."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)
        
        # Patterns for detecting tool calls in streams
        self._tool_call_patterns = [
            re.compile(r'"tool_calls":\s*\['),
            re.compile(r'"function_call":\s*\{'),
            re.compile(r'<tool_call>'),
            re.compile(r'{"name":\s*"[^"]+",\s*"arguments":\s*\{'),
            # Harmony commentary tool-call header, e.g. <|channel|>commentary to=functions.get_weather ... <|message|>{...}<|call|>
            re.compile(r'<\|channel\|>\s*commentary.*?to=functions\.[\w\.]+', re.IGNORECASE)
        ]
    
    def process_stream_chunk(
        self,
        chunk: str,
        state: StreamState,
        on_content: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> StreamState:
        """Process a single chunk from the stream."""
        try:
            # Accumulate content
            state.accumulated_content += chunk
            
            # Check for tool call patterns
            if self._detect_tool_call_start(chunk):
                state.in_tool_call = True
                state.current_tool_buffer = chunk
                self._logger.debug("Tool call pattern detected in stream")
                return state
            
            # If we're in a tool call, accumulate the buffer
            if state.in_tool_call:
                state.current_tool_buffer += chunk
                
                # Try to parse complete tool call
                tool_call = self._try_parse_tool_call(state.current_tool_buffer)
                if tool_call:
                    state.tool_calls_detected.append(tool_call)
                    if on_tool_call:
                        on_tool_call(tool_call)
                    
                    # Reset tool call state
                    state.in_tool_call = False
                    state.current_tool_buffer = ""
                    
                    self._logger.debug(f"Parsed tool call: {tool_call.get('name', 'unknown')}")
                
                return state
            
            # Regular content - emit for display
            if chunk and on_content:
                on_content(chunk)
            
            return state
            
        except Exception as e:
            self._logger.error(f"Error processing stream chunk: {e}")
            state.error = str(e)
            return state
    
    def _detect_tool_call_start(self, chunk: str) -> bool:
        """Detect if chunk contains start of tool call."""
        for pattern in self._tool_call_patterns:
            if pattern.search(chunk):
                return True
        return False
    
    def _try_parse_tool_call(self, buffer: str) -> Optional[Dict[str, Any]]:
        """Try to parse complete tool call from buffer."""
        # Try JSON parsing first
        try:
            # Look for complete JSON objects
            json_match = re.search(r'\{[^}]+\}', buffer)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                if self._is_valid_tool_call(data):
                    return data
        except json.JSONDecodeError:
            pass
        
        # Try XML-style parsing
        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', buffer, re.DOTALL)
        if tool_match:
            try:
                tool_content = tool_match.group(1).strip()
                # Parse as JSON within XML tags
                data = json.loads(tool_content)
                if self._is_valid_tool_call(data):
                    return data
            except json.JSONDecodeError:
                pass
        
        # Try Harmony-style commentary tool-call parsing using token scanner
        try:
            calls = HarmonyFormatter.extract_tool_calls_from_harmony(buffer)
            if calls:
                return calls[0]
        except Exception:
            pass
        
        return None
    
    def _is_valid_tool_call(self, data: Dict[str, Any]) -> bool:
        """Check if data represents a valid tool call."""
        # Must have name and arguments
        if "name" not in data:
            return False
        
        # Arguments can be dict or string
        if "arguments" not in data:
            return False
        
        return True
    
    def finalize_stream(
        self,
        state: StreamState
    ) -> Dict[str, Any]:
        """Finalize stream processing and extract results."""
        state.complete = True
        
        # Extract any remaining tool calls from accumulated content
        remaining_tool_calls = self._extract_final_tool_calls(state.accumulated_content)
        state.tool_calls_detected.extend(remaining_tool_calls)
        
        # Clean content and keep only Harmony final-channel content
        clean_content = self._clean_content(state.accumulated_content)
        
        result = {
            "content": clean_content,
            "tool_calls": state.tool_calls_detected,
            "success": state.error is None,
            "error": state.error
        }
        
        self._logger.debug(
            f"Stream finalized - Content: {len(clean_content)} chars, "
            f"Tool calls: {len(state.tool_calls_detected)}"
        )
        
        return result
    
    def _extract_final_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract any tool calls from final accumulated content."""
        tool_calls = []
        
        # Try to parse as JSON response
        try:
            data = json.loads(content)
            extracted_calls = extract_tool_calls_from_response(data)
            for call in extracted_calls:
                tool_calls.append(call.to_harmony_format())
        except json.JSONDecodeError:
            pass
        
        # Look for structured patterns
        patterns = [
            r'"tool_calls":\s*(\[.*?\])',
            r'"function_call":\s*(\{.*?\})',
            r'<tool_call>(.*?)</tool_call>'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    tool_data = json.loads(match.group(1))
                    if self._is_valid_tool_call(tool_data):
                        tool_calls.append(tool_data)
                except (json.JSONDecodeError, IndexError):
                    continue
        
        # Harmony-style commentary tool-call extraction via token scanner
        try:
            for call in HarmonyFormatter.extract_tool_calls_from_harmony(content):
                tool_calls.append(call)
        except Exception:
            pass
        
        return tool_calls
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing tool call artifacts."""
        # Prefer extracting only the Harmony final-channel text if present
        final_only = self._extract_final_channel_text(content)
        if final_only is not None and final_only.strip():
            return final_only.strip()
        
        # Remove JSON-style tool calls
        content = re.sub(r'"tool_calls":\s*\[.*?\]', '', content, flags=re.DOTALL)
        content = re.sub(r'"function_call":\s*\{.*?\}', '', content, flags=re.DOTALL)
        
        # Remove XML-style tool calls
        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
        
        # Remove Harmony-style commentary tool-call segments including assistant wrapper
        # Pattern 1: with assistant envelope
        content = re.sub(r'<\|start\|>assistant\s*<\|channel\|>\s*commentary.*?<\|call\|>\s*<\|end\|>', '', content, flags=re.DOTALL|re.IGNORECASE)
        # Pattern 2: without assistant envelope (older outputs)
        content = re.sub(r'<\|channel\|>\s*commentary.*?<\|call\|>', '', content, flags=re.DOTALL|re.IGNORECASE)
        
        # Remove Harmony analysis blocks entirely
        # With assistant envelope
        content = re.sub(r'<\|start\|>assistant\s*<\|channel\|>\s*analysis.*?<\|end\|>', '', content, flags=re.DOTALL|re.IGNORECASE)
        # Without assistant envelope
        content = re.sub(r'<\|channel\|>\s*analysis.*?(?=(<\|channel\|>|<\|end\|>|$))', '', content, flags=re.DOTALL|re.IGNORECASE)
        # Also remove any tool response wrappers if leaked into content
        content = re.sub(r'<\|start\|>functions\.[^<]*?<\|end\|>', '', content, flags=re.DOTALL|re.IGNORECASE)
        
        # Remove common JSON wrapper artifacts
        content = re.sub(r'^\s*\{.*?"content":\s*"', '', content)
        content = re.sub(r'"\s*\}\s*$', '', content)
        
        # Clean up whitespace
        content = re.sub(r'\n\s*\n', '\n', content)
        content = content.strip()
        
        return content

    def _extract_final_channel_text(self, content: str) -> Optional[str]:
        """Extract only the Harmony final-channel message content using a token scanner.
        Returns concatenated final text if found, otherwise None.
        """
        if not content:
            return None
        CHANNEL = "<|channel|>"
        MESSAGE = "<|message|>"
        END = "<|end|>"
        final_segments: List[str] = []
        idx = 0
        n = len(content)
        while idx < n:
            ch_pos = content.find(CHANNEL, idx)
            if ch_pos == -1:
                break
            header_start = ch_pos + len(CHANNEL)
            msg_pos = content.find(MESSAGE, header_start)
            if msg_pos == -1:
                break
            header = content[ch_pos:msg_pos].lower()
            if "final" not in header:
                idx = msg_pos + len(MESSAGE)
                continue
            seg_start = msg_pos + len(MESSAGE)
            end_pos = content.find(END, seg_start)
            if end_pos == -1:
                # Take until next channel or end
                next_ch = content.find(CHANNEL, seg_start)
                end_pos = next_ch if next_ch != -1 else n
            segment = content[seg_start:end_pos]
            if segment:
                final_segments.append(segment)
            idx = end_pos + len(END)
        if final_segments:
            # Join with single newline and strip artifacts
            joined = "\n".join(s.strip() for s in final_segments if s.strip())
            return joined.strip()
        return None
    
    def create_stream_state(self) -> StreamState:
        """Create a new stream state for processing."""
        return StreamState()
    
    def handle_stream_error(
        self,
        error: Exception,
        state: StreamState,
        fallback_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle streaming errors gracefully."""
        self._logger.error(f"Stream processing error: {error}")
        
        # Try to salvage any accumulated content
        content = fallback_content or state.accumulated_content or "Error processing stream"
        
        return {
            "content": content,
            "tool_calls": state.tool_calls_detected,
            "success": False,
            "error": str(error),
            "fallback_used": fallback_content is not None
        }
    
    def estimate_completion_progress(self, state: StreamState) -> float:
        """Estimate completion progress based on stream state."""
        # Simple heuristic based on content accumulation
        if state.complete:
            return 1.0
        
        # If we have tool calls detected, we're likely near completion
        if state.tool_calls_detected:
            return 0.8
        
        # If we're in a tool call, we're partially through
        if state.in_tool_call:
            return 0.6
        
        # Otherwise, estimate based on content length (rough heuristic)
        content_length = len(state.accumulated_content)
        if content_length < 100:
            return 0.2
        elif content_length < 500:
            return 0.4
        else:
            return 0.6
