from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional
from pathlib import Path

from ..sandbox.runner import run_in_sandbox
from ..utils_scratchpad import get_scratch_host_dir, sanitize_filename, get_scratch_file_path

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "scratchpad",
        "description": "Persistent scratch-pad stored on host via sandbox RW mount. Actions: read, append, write, clear, search.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["read", "append", "write", "clear", "search"]},
                "filename": {"type": "string", "default": "scratchpad.txt"},
                "text": {"type": "string", "description": "Text to append or write"},
                "query": {"type": "string", "description": "Search query (substring match)"},
                "limit_lines": {"type": "integer", "minimum": 0, "default": 200},
            },
            "required": ["action"],
        },
    },
}


def _result(ok: bool, **extra: Any) -> str:
    d: Dict[str, Any] = {"tool": "scratchpad", "ok": bool(ok)}
    d.update(extra)
    # compose a compact inject summary
    act = extra.get("action")
    fname = extra.get("filename")
    preview = extra.get("preview")
    if act and fname:
        d["inject"] = f"scratchpad {act} {fname}: {str(preview)[:160]}" if preview is not None else f"scratchpad {act} {fname}"
    return json.dumps(d)


def _host_read(path: Path, limit_lines: int) -> Dict[str, Any]:
    if not path.exists():
        return {"ok": True, "bytes": 0, "lines": 0, "preview": ""}
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    lines = data.splitlines()
    preview = "\n".join(lines[:max(0, limit_lines)])
    return {"ok": True, "bytes": len(data.encode("utf-8", errors="replace")), "lines": len(lines), "preview": preview}


def _host_write(path: Path, text: str, append: bool) -> Dict[str, Any]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as f:
            f.write(text)
            if append and not text.endswith("\n"):
                f.write("\n")
        return {"ok": True, "bytes": path.stat().st_size}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _host_clear(path: Path) -> Dict[str, Any]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return {"ok": True, "bytes": 0}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _host_search(path: Path, query: str, limit_lines: int) -> Dict[str, Any]:
    if not path.exists():
        return {"ok": True, "matches": []}
    try:
        out: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if query in line:
                    out.append(f"{i}: {line.rstrip()} ")
                    if 0 < limit_lines <= len(out):
                        break
        return {"ok": True, "matches": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}


_DEF_SCRIPT = r"""
import os, sys, json, pathlib
base = pathlib.Path('/workspace/scratch')
name = os.environ.get('SP_FILENAME', 'scratchpad.txt')
name = name.replace('\\\\','_').replace('/', '_')[:128] or 'scratchpad.txt'
fp = base / name
act = os.environ.get('SP_ACTION')
text = os.environ.get('SP_TEXT', '')
query = os.environ.get('SP_QUERY', '')
try:
    limit = int(os.environ.get('SP_LIMIT') or '200')
except Exception:
    limit = 200
base.mkdir(parents=True, exist_ok=True)
res = {"ok": False}
try:
    if act == 'read':
        if fp.exists():
            data = fp.read_text(encoding='utf-8', errors='replace')
            lines = data.splitlines()
            prev = "\n".join(lines[:max(0, limit)])
            res = {"ok": True, "bytes": len(data.encode('utf-8', 'replace')), "lines": len(lines), "preview": prev}
        else:
            res = {"ok": True, "bytes": 0, "lines": 0, "preview": ""}
    elif act in ('append','write'):
        mode = 'a' if act == 'append' else 'w'
        with fp.open(mode, encoding='utf-8') as f:
            f.write(text)
            if act == 'append' and not text.endswith('\n'):
                f.write('\n')
        res = {"ok": True, "bytes": fp.stat().st_size}
    elif act == 'clear':
        fp.write_text('', encoding='utf-8')
        res = {"ok": True, "bytes": 0}
    elif act == 'search':
        found = []
        if fp.exists():
            with fp.open('r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f, 1):
                    if query in line:
                        found.append(f"{i}: {line.rstrip()} ")
                        if 0 < limit <= len(found):
                            break
        res = {"ok": True, "matches": found}
    else:
        res = {"ok": False, "error": 'unknown action'}
except Exception as e:
    res = {"ok": False, "error": str(e)}
print(json.dumps(res))
"""


def _via_sandbox(action: str, filename: str, text: Optional[str], query: Optional[str], limit_lines: int) -> Dict[str, Any]:
    host_dir = get_scratch_host_dir()
    env_allow = {
        'SP_ACTION': action,
        'SP_FILENAME': filename,
        'SP_TEXT': text or '',
        'SP_QUERY': query or '',
        'SP_LIMIT': str(int(limit_lines)),
    }
    cmd = "bash -lc \"python - <<'PY'\n" + _DEF_SCRIPT + "\nPY\n\""
    res = run_in_sandbox(
        command=["bash", "-lc", cmd],
        cwd=os.getcwd(),
        timeout_s=20,
        network='deny',
        cpu_quota=float(os.getenv('SANDBOX_CPU', '0.25') or 0.25),
        memory_mb=int(os.getenv('SANDBOX_MEM_MB', '256') or 256),
        pids_max=int(os.getenv('SANDBOX_PIDS', '64') or 64),
        disk_write_mb=int(os.getenv('SANDBOX_DISK_MB', '128') or 128),
        env_allowlist=env_allow,
        mount_project_ro=True,
        extra_mounts=[(str(host_dir), "/workspace/scratch", "rw")],
        user='sandbox',
        summary_max_bytes=131072,
        redact_patterns=None,
    )
    out: Dict[str, Any]
    if res.get('ok') and res.get('stdout'):
        try:
            out = json.loads(res['stdout'])
        except Exception:
            out = {"ok": False, "error": "failed to parse sandbox output"}
    else:
        out = {"ok": False, "error": res.get('stderr') or 'sandbox unavailable'}
    return out


def _via_host(action: str, filename: str, text: Optional[str], query: Optional[str], limit_lines: int) -> Dict[str, Any]:
    path = get_scratch_file_path(filename)
    if action == 'read':
        return _host_read(path, limit_lines)
    if action == 'append':
        return _host_write(path, text or '', append=True)
    if action == 'write':
        return _host_write(path, text or '', append=False)
    if action == 'clear':
        return _host_clear(path)
    if action == 'search':
        return _host_search(path, query or '', limit_lines)
    return {"ok": False, "error": "unknown action"}


def scratchpad(action: str, filename: Optional[str] = None, text: Optional[str] = None, query: Optional[str] = None, limit_lines: int = 200) -> str:
    """Tool entrypoint. Attempts sandbox write with host-mounted persistence. Falls back to host ops if sandbox is unavailable or disabled.
    """
    action = (action or '').strip()
    fname = sanitize_filename(filename)

    prefer_host = os.getenv('SANDBOX_USE_HOST_FALLBACK', '1') in {'1', 'true', 'True', 'yes', 'on'}

    result = _via_sandbox(action, fname, text, query, limit_lines)
    persisted_path = str(get_scratch_file_path(fname))

    if not result.get('ok') and prefer_host:
        result = _via_host(action, fname, text, query, limit_lines)
        result['persisted_via'] = 'host'
    else:
        result['persisted_via'] = 'sandbox'

    result['action'] = action
    result['filename'] = fname
    result['persisted_path'] = persisted_path

    return _result(bool(result.get('ok')), **result)


TOOL_IMPLEMENTATION = scratchpad
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
