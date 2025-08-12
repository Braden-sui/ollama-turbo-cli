"""File listing tool plugin wrapping src.tools.list_files"""
from __future__ import annotations

from ..tools import list_files as _impl

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files and directories in a specified directory with optional extension filtering",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list contents of",
                    "default": "."
                },
                "extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt')"
                }
            },
            "required": []
        }
    }
}

TOOL_IMPLEMENTATION = _impl
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
