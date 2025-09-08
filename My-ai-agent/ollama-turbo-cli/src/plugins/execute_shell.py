from __future__ import annotations
import os
import json
import re
import shlex
from typing import Any, Dict
from pathlib import Path

from ..sandbox.runner import run_in_sandbox

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_shell",
        "description": (
            "Execute a short shell command inside a Docker-based sandbox. By default it's locked down: "
            "read-only rootfs, project mounted read-only, path confinement to SHELL_TOOL_ROOT, allowlisted command prefixes, and network disabled. "
            "Use sparingly for diagnostics (e.g., 'git status', 'ls'). Requires SHELL_TOOL_ALLOW=1. "
            "To allow egress from the container, explicitly opt in with SANDBOX_PERMISSIVE=1 or SANDBOX_NET=bridge (and adjust SHELL_TOOL_ALLOWLIST as needed). "
            "This is separate from web_fetch, which uses the agent's policy-aware web client; prefer web_fetch for HTTP(S) reads."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "command": {"type": "string", "description": "Exact command string. In default mode it must start with an allowlisted prefix. Do not include secrets. In permissive mode (SANDBOX_PERMISSIVE=1) allowlist may be wildcarded (e.g., '*')."},
                "working_dir": {"type": "string", "default": ".", "description": "Directory to run the command in. Must be within SHELL_TOOL_ROOT."},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 300, "default": 30, "description": "Timeout in seconds (1–300). Prefer small values."},
                "shell": {"type": "boolean", "default": False, "description": "If true, run via a shell (e.g., bash -lc). Prefer false unless shell features are needed."},
                "capture_stderr": {"type": "boolean", "default": True, "description": "Include a short stderr summary in injected output."},
                "env_vars": {"type": "object", "default": {}, "description": "Subset of environment variables to pass through. Never include credentials or secrets."},
            },
            "required": ["command"],
        },
    },
}

_ALLOWLIST_DEFAULT = [x.strip() for x in (os.getenv('SHELL_TOOL_ALLOWLIST', 'git status,git log,git diff,ls,dir,cat,type,python -V,python --version,pip list,pip install, pip show').split(',') if os.getenv('SHELL_TOOL_ALLOWLIST') is not None else 'git status,git log,git diff,ls,dir,cat,type,python -V,python --version,pip list'.split(','))]


def _allowed_command(cmd: str) -> bool:
    base = cmd.strip()
    # Wildcard allow: if '*' present, allow any command
    for item in _ALLOWLIST_DEFAULT:
        if not item:
            continue
        item_s = item.strip()
        if item_s == '*':
            return True
        if base.startswith(item_s):
            return True
    return False


def _within_root(path: Path, root: Path) -> bool:
    try:
        path = path.resolve()
        root = root.resolve()
        return str(path).startswith(str(root))
    except Exception:
        return False


def execute_shell(command: str, working_dir: str = '.', timeout: int = 30, shell: bool = False, capture_stderr: bool = True, env_vars: Dict[str, str] | None = None) -> str:
    # Global kill switch
    allow = os.getenv('SHELL_TOOL_ALLOW', '0') in {'1', 'true', 'True'}
    confirm_default = (os.getenv('SHELL_TOOL_CONFIRM') or ('1' if os.isatty(0) else '0')) in {'1', 'true', 'True'}
    root = Path(os.getenv('SHELL_TOOL_ROOT', os.getcwd())).resolve()
    permissive = os.getenv('SANDBOX_PERMISSIVE', '0') in {'1', 'true', 'True'}

    cmd_str = (command or '').strip()
    # Build redaction patterns early to protect blocked responses as well
    redact = [r"(?i)(api[_-]?key|secret|token|authorization)\s*[:=]\s*([^\s'\"]+)"]
    for v in (env_vars or {}).values():
        if not v:
            continue
        try:
            s = re.escape(str(v))
            redact.append(s)
        except Exception:
            continue
    def _apply_redactions(text: str) -> str:
        try:
            import re as _re
            s = text
            for pat in redact:
                try:
                    s = _re.sub(pat, '[REDACTED]', s)
                except _re.error:
                    continue
            return s
        except Exception:
            return text

    if not cmd_str:
        return json.dumps({"ok": False, "blocked": True, "timed_out": False, "reason": "Empty command", "would_run": None, "how_to_enable": "Set SHELL_TOOL_ALLOW=1 and provide a command.", "inject": "blocked: empty command"})

    wd = Path(working_dir or '.').resolve()
    if not wd.exists() or not wd.is_dir():
        return json.dumps({"ok": False, "blocked": True, "reason": f"Working directory does not exist: {wd}", "would_run": None})

    if (not permissive) and (not _within_root(wd, root)):
        return json.dumps({"ok": False, "blocked": True, "timed_out": False, "reason": "Working directory is outside SHELL_TOOL_ROOT", "cwd": str(wd), "root": str(root), "inject": "blocked: path confinement"})

    # Policy
    if (not allow) or ((not _allowed_command(cmd_str)) and (not permissive)):
        hint = "Set SHELL_TOOL_ALLOW=1 and add your command prefix to SHELL_TOOL_ALLOWLIST."
        safe_cmd = _apply_redactions(cmd_str)
        return json.dumps({
            "ok": False,
            "blocked": True,
            "timed_out": False,
            "reason": "Command not allowed by policy",
            "cwd": str(wd),
            "would_run": safe_cmd,
            "how_to_enable": hint,
            "inject": "blocked: command not allowed by policy",
        })

    # Prepare sandbox call
    env_allow = dict(env_vars or {})
    # Add proxy envs if present
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY'):
        v = os.getenv(k)
        if v:
            env_allow[k] = v

    # Redaction patterns already built above (redact)

    # Compose container command
    if shell:
        container_cmd = ["bash", "-lc", cmd_str]
    else:
        # split safely
        try:
            container_cmd = shlex.split(cmd_str)
        except Exception:
            container_cmd = ["bash", "-lc", cmd_str]

    res = run_in_sandbox(
        command=container_cmd,
        cwd=str(wd),
        timeout_s=int(timeout),
        network=os.getenv('SANDBOX_NET', 'bridge' if permissive else 'deny'),
        allow_net_to=os.getenv('SANDBOX_NET_ALLOW', '').split(',') if os.getenv('SANDBOX_NET_ALLOW') else None,
        cpu_quota=float(os.getenv('SANDBOX_CPU', '0.5') or 0.5),
        memory_mb=int(os.getenv('SANDBOX_MEM_MB', '512') or 512),
        pids_max=int(os.getenv('SANDBOX_PIDS', '64') or 64),
        disk_write_mb=int(os.getenv('SANDBOX_DISK_MB', '128') or 128),
        env_allowlist=env_allow,
        mount_project_ro=not permissive,
        user=('root' if permissive else 'sandbox'),
        summary_max_bytes=int(os.getenv('SHELL_TOOL_MAX_OUTPUT', '131072') or 131072),
        redact_patterns=redact,
    )

    # Summarize for injection
    summary_lines = [
        f"ok={res.get('ok')} exit={res.get('exit_code')} timed_out={res.get('timed_out')} truncated={res.get('truncated')}",
        f"stdout_bytes={res.get('bytes_stdout')} stderr_bytes={res.get('bytes_stderr')}",
    ]
    # Include first and last lines of stdout/stderr
    def _first_last(s: str) -> str:
        lines = (s or '').splitlines()
        if not lines:
            return ''
        if len(lines) == 1:
            return lines[0]
        return lines[0] + (" … ") + lines[-1]

    summary_lines.append("stdout: " + _first_last(res.get('stdout', '')))
    if capture_stderr:
        summary_lines.append("stderr: " + _first_last(res.get('stderr', '')))

    inject = "\n".join(summary_lines)

    out = {
        "tool": "execute_shell",
        "ok": bool(res.get('ok')),
        "blocked": False,
        "timed_out": bool(res.get('timed_out')),
        "exit_code": int(res.get('exit_code', 0)),
        "truncated": bool(res.get('truncated')),
        "bytes_stdout": int(res.get('bytes_stdout', 0)),
        "bytes_stderr": int(res.get('bytes_stderr', 0)),
        "usage": res.get('usage', {}),
        "log_path": res.get('log_path'),
        "inject": inject,
        "sensitive": True,  # mark as sensitive by default
    }
    return json.dumps(out)

# Optional plugin metadata for loader and diagnostics
TOOL_IMPLEMENTATION = execute_shell
TOOL_AUTHOR = "core"
TOOL_VERSION = "2.0.0"
