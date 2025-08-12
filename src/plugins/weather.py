"""Weather tool plugin wrapping src.tools.get_current_weather"""
from __future__ import annotations

from ..tools import get_current_weather as _impl

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get current weather information for a specific city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for (e.g., London, Paris, Tokyo)"
                },
                "unit": {
                    "type": "string",
                    "description": "Temperature unit - celsius or fahrenheit",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["city"]
        }
    }
}

TOOL_IMPLEMENTATION = _impl
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
