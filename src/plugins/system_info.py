"""System info tool plugin wrapping src.tools.get_system_info"""
from __future__ import annotations

from ..tools import get_system_info as _impl

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_system_info",
        "description": "Get comprehensive system information including OS, CPU, memory, disk usage, and Python environment",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

TOOL_IMPLEMENTATION = _impl
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
