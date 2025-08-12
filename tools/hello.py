"""Example third-party plugin: hello

Returns a friendly greeting. Demonstrates minimalist plugin structure.
"""
from __future__ import annotations

from typing import Optional

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "hello",
        "description": "Return a friendly greeting message.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet", "default": "world"},
            },
            "required": []
        }
    }
}


def execute(name: Optional[str] = None) -> str:
    try:
        who = (name or "world").strip()
        if not who:
            who = "world"
        return f"Hello, {who}! 👋"
    except Exception as e:
        return f"Error generating greeting: {e}"
