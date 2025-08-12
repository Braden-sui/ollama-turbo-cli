"""System info tool plugin"""
from __future__ import annotations

import os
import platform
import sys
from datetime import datetime

try:  # Optional psutil
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

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

def get_system_info() -> str:
    """Get comprehensive system information."""
    try:
        # Basic system info
        system_info = {
            "OS": platform.system(),
            "OS Version": platform.release(),
            "Architecture": platform.machine(),
            "Processor": platform.processor() or "Unknown",
            "Python Version": platform.python_version(),
            "Python Executable": sys.executable,
            "Current Directory": os.getcwd(),
            "User": os.getenv('USER') or os.getenv('USERNAME') or "Unknown",
            "Hostname": platform.node(),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if psutil:
            try:
                memory = psutil.virtual_memory()
                system_info["Total RAM"] = f"{memory.total / (1024**3):.1f} GB"
                system_info["Available RAM"] = f"{memory.available / (1024**3):.1f} GB"
                system_info["RAM Usage"] = f"{memory.percent}%"
            except Exception:
                system_info["Memory"] = "Unable to retrieve"

            try:
                disk = psutil.disk_usage('/')
                system_info["Total Disk"] = f"{disk.total / (1024**3):.1f} GB"
                system_info["Free Disk"] = f"{disk.free / (1024**3):.1f} GB"
                system_info["Disk Usage"] = f"{(disk.used / disk.total) * 100:.1f}%"
            except Exception:
                system_info["Disk"] = "Unable to retrieve"

            try:
                system_info["CPU Cores"] = psutil.cpu_count(logical=False)
                system_info["CPU Threads"] = psutil.cpu_count(logical=True)
                system_info["CPU Usage"] = f"{psutil.cpu_percent(interval=1)}%"
            except Exception:
                system_info["CPU"] = "Unable to retrieve"
        else:
            system_info["System Stats"] = "psutil not installed - install for detailed system info"

        result = "System Information:\n"
        for key, value in system_info.items():
            result += f"  {key}: {value}\n"
        return result.strip()
    except Exception as e:
        return f"Error retrieving system information: {str(e)}"


TOOL_IMPLEMENTATION = get_system_info
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
