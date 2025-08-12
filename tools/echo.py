"""Example third-party plugin: echo

Echo back provided text with optional transformations.
"""
from __future__ import annotations

from typing import Optional

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo back the provided text with optional uppercase or reverse transformations.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo back"},
                "uppercase": {"type": "boolean", "description": "Convert to uppercase", "default": False},
                "reverse": {"type": "boolean", "description": "Reverse the text", "default": False}
            },
            "required": ["text"]
        }
    }
}


def execute(text: str, uppercase: Optional[bool] = False, reverse: Optional[bool] = False) -> str:
    try:
        out = text
        if uppercase:
            out = out.upper()
        if reverse:
            out = out[::-1]
        return out
    except Exception as e:
        return f"Error in echo: {e}"
