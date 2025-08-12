"""Calculator tool plugin wrapping src.tools.calculate_math"""
from __future__ import annotations

from ..tools import calculate_math as _impl

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate_math",
        "description": "Evaluate mathematical expressions including basic operations and functions like sin, cos, sqrt, log, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sin(pi/2)', 'sqrt(16)')"
                }
            },
            "required": ["expression"]
        }
    }
}

TOOL_IMPLEMENTATION = _impl
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
