"""File listing tool plugin"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

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

def list_files(directory: str = ".", extension: Optional[str] = None) -> str:
    """List files in directory with optional extension filter."""
    try:
        # Resolve directory path
        directory = os.path.abspath(directory)
        
        if not os.path.exists(directory):
            return f"Error: Directory '{directory}' does not exist"
        
        if not os.path.isdir(directory):
            return f"Error: '{directory}' is not a directory"
        
        # Get all items in directory
        try:
            items = os.listdir(directory)
        except PermissionError:
            return f"Error: Permission denied accessing '{directory}'"
        
        files = []
        dirs = []
        
        for item in items:
            item_path = os.path.join(directory, item)
            try:
                if os.path.isfile(item_path):
                    # Check extension filter
                    if extension is None or item.lower().endswith(extension.lower()):
                        # Get file size and modification time
                        size = os.path.getsize(item_path)
                        mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                        
                        # Format size
                        if size < 1024:
                            size_str = f"{size} B"
                        elif size < 1024*1024:
                            size_str = f"{size/1024:.1f} KB"
                        else:
                            size_str = f"{size/(1024*1024):.1f} MB"
                        
                        files.append(f"{item} ({size_str}, {mtime.strftime('%Y-%m-%d %H:%M')})")
                elif os.path.isdir(item_path):
                    if extension is None:  # Only show directories if no extension filter
                        dirs.append(f"{item}/")
            except (OSError, PermissionError):
                continue  # Skip inaccessible items
        
        # Sort files and directories
        files.sort()
        dirs.sort()
        
        result_items = dirs + files
        
        if not result_items:
            filter_msg = f" with extension '{extension}'" if extension else ""
            return f"No files found in '{directory}'{filter_msg}"
        
        # Limit output to prevent overwhelming response
        total_items = len(result_items)
        display_items = result_items[:15]
        
        result = f"Contents of '{directory}':\n"
        result += "\n".join(f"  {item}" for item in display_items)
        
        if total_items > 15:
            result += f"\n  ... and {total_items - 15} more items"
        
        return result
        
    except Exception as e:
        return f"Error listing files in '{directory}': {str(e)}"


TOOL_IMPLEMENTATION = list_files
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
