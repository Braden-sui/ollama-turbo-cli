import os
import sys
import json
import uuid
import time
import shutil
import platform
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import subprocess  # Used only to invoke Docker CLI, never user commands directly

logger = logging.getLogger(__name__)

SESSIONS_ROOT = Path('.sandbox') / 'sessions'
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

@dataclass
class SandboxResult:
    ok: bool
    exit_code: int
    timed_out: bool
    stdout: str
    stderr: str
    bytes_stdout: int
    bytes_stderr: int
    truncated: bool
    usage: Dict[str, Any]
    log_path: str
    net: Optional[Dict[str, Any]] = None


def _truncate_and_redact(text: bytes, max_bytes: int, redact_patterns: Optional[List[str]]) -> Tuple[str, bool, int]:
    b = text or b""
    truncated = False
    if len(b) > max_bytes:
        b = b[:max_bytes]
        truncated = True
    s = b.decode(errors='replace')
    if redact_patterns:
        import re
        for pat in redact_patterns:
            try:
                s = re.sub(pat, '[REDACTED]', s)
            except re.error:
                continue
    return s, truncated, len(text or b"")


def _docker_available() -> bool:
    try:
        completed = subprocess.run(['docker', 'version', '--format', '{{.Server.Version}}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        return completed.returncode == 0
    except Exception:
        return False


def _build_docker_cmd(image: str, command: List[str], *, cwd: str, timeout_s: int, network: str, cpu_quota: float,
                       memory_mb: int, pids_max: int, disk_write_mb: int, env_allowlist: Dict[str, str],
                       mount_project_ro: bool, extra_mounts: Optional[List[Tuple[str, str, str]]], user: str,
                       session_dir: Path) -> List[str]:
    # Base flags
    cpus = max(0.1, float(cpu_quota))  # map 0.5 -> --cpus 0.5
    mem = f"{int(memory_mb)}m"

    docker_cmd = [
        'docker', 'run', '--rm',
        '--security-opt', 'no-new-privileges',
        '--cap-drop', 'ALL',
        '--pids-limit', str(int(pids_max)),
        '--cpus', str(cpus),
        '--memory', mem,
        '--read-only',
        # tmpfs for writable workspace
        '--tmpfs', f"/workspace:rw,size={int(disk_write_mb)}m,nosuid,nodev,noexec",
        # mount session logs dir (persisted on host)
        '-v', f"{session_dir.as_posix()}:/workspace/logs:rw",
    ]

    # Network flags
    if network == 'deny':
        docker_cmd += ['--network', 'none']
    else:
        # default bridge; actual egress is controlled by proxy env if provided
        pass

    # Env allowlist only
    for k, v in (env_allowlist or {}).items():
        docker_cmd += ['-e', f"{k}={v}"]

    # Mount project read-only
    if mount_project_ro:
        project_root = Path(os.getenv('SHELL_TOOL_ROOT', os.getcwd())).resolve()
        docker_cmd += ['-v', f"{project_root.as_posix()}:/project:ro"]

    # Extra mounts
    if extra_mounts:
        for src, dst, mode in extra_mounts:
            docker_cmd += ['-v', f"{Path(src).resolve().as_posix()}:{dst}:{mode}"]

    # Working directory inside container
    docker_cmd += ['-w', '/workspace']

    # User: try non-root user if exists; fallback to default container user
    # Many minimal images have nobody (65534)
    docker_cmd += ['-u', '65534:65534']

    docker_cmd += [image]
    docker_cmd += command
    return docker_cmd


def run_in_sandbox(
    command: Union[str, List[str]],
    *,
    cwd: str,
    timeout_s: int = 15,
    network: str = 'deny',
    allow_net_to: Optional[List[str]] = None,
    cpu_quota: float = 0.5,
    memory_mb: int = 512,
    pids_max: int = 64,
    disk_write_mb: int = 128,
    env_allowlist: Dict[str, str] = {},
    mount_project_ro: bool = True,
    extra_mounts: Optional[List[Tuple[str, str, str]]] = None,
    user: str = 'sandbox',
    summary_max_bytes: int = 131072,
    redact_patterns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Execute a command in a Docker-based sandbox when available. Fails closed otherwise.

    No host subprocess execution of untrusted commands occurs. Only the Docker CLI is invoked.
    """
    start = time.time()
    session_id = str(uuid.uuid4())
    session_dir = (SESSIONS_ROOT / session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    full_log_path = session_dir / 'full.log'

    # Normalize command for container shell if needed upstream
    if isinstance(command, str):
        cmd_list = ['bash', '-lc', command]
    else:
        cmd_list = list(command)

    # Enforce working_dir exists
    working_dir = Path(cwd).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        return {
            'ok': False,
            'exit_code': 127,
            'timed_out': False,
            'stdout': '',
            'stderr': f"Working directory does not exist: {working_dir}",
            'bytes_stdout': 0,
            'bytes_stderr': 0,
            'truncated': False,
            'usage': {'cpu_s': 0.0, 'wall_s': 0.0, 'max_rss_mb': 0},
            'log_path': f"sandbox://sessions/{session_id}/full.log",
        }

    # Detect Docker availability
    if not _docker_available():
        # Windows guidance
        if platform.system().lower().startswith('win'):
            reason = 'Sandbox unavailable: Docker Desktop/WSL2 not detected. Install Docker Desktop and enable WSL2 integration.'
        else:
            reason = 'Sandbox unavailable: Docker is not installed or not running.'
        with open(full_log_path, 'wb') as f:
            f.write(reason.encode())
        return {
            'ok': False,
            'exit_code': 126,
            'timed_out': False,
            'stdout': '',
            'stderr': reason,
            'bytes_stdout': 0,
            'bytes_stderr': len(reason),
            'truncated': False,
            'usage': {'cpu_s': 0.0, 'wall_s': 0.0, 'max_rss_mb': 0},
            'log_path': f"sandbox://sessions/{session_id}/full.log",
        }

    image = os.getenv('SANDBOX_IMAGE', 'python:3.11-slim')

    # Proxy env for allowlist-mode egress (net_proxy provides the proxy and envs)
    if network == 'allowlist':
        # net_proxy will pass HTTP(S)_PROXY via env_allowlist
        pass

    docker_cmd = _build_docker_cmd(
        image=image,
        command=cmd_list,
        cwd=str(working_dir),
        timeout_s=timeout_s,
        network=network,
        cpu_quota=cpu_quota,
        memory_mb=memory_mb,
        pids_max=pids_max,
        disk_write_mb=disk_write_mb,
        env_allowlist=env_allowlist,
        mount_project_ro=mount_project_ro,
        extra_mounts=extra_mounts,
        user=user,
        session_dir=session_dir,
    )

    # Run the container
    timed_out = False
    try:
        proc = subprocess.Popen(
            docker_cmd,
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = proc.communicate(timeout=max(1, int(timeout_s)))
        except subprocess.TimeoutExpired:
            proc.kill()
            timed_out = True
            stdout_b, stderr_b = proc.communicate()
    except Exception as e:
        err = f"Failed to start sandbox container: {e}"
        with open(full_log_path, 'wb') as f:
            f.write(err.encode())
        return {
            'ok': False,
            'exit_code': 126,
            'timed_out': False,
            'stdout': '',
            'stderr': err,
            'bytes_stdout': 0,
            'bytes_stderr': len(err),
            'truncated': False,
            'usage': {'cpu_s': 0.0, 'wall_s': time.time() - start, 'max_rss_mb': 0},
            'log_path': f"sandbox://sessions/{session_id}/full.log",
        }

    # Write full log
    try:
        with open(full_log_path, 'wb') as f:
            f.write(b"[STDOUT]\n")
            f.write(stdout_b or b"")
            f.write(b"\n[STDERR]\n")
            f.write(stderr_b or b"")
    except Exception:
        pass

    # Summarize
    stdout_s, trunc_out, bytes_stdout = _truncate_and_redact(stdout_b or b"", summary_max_bytes, redact_patterns)
    stderr_s, trunc_err, bytes_stderr = _truncate_and_redact(stderr_b or b"", summary_max_bytes, redact_patterns)

    usage = {
        'cpu_s': 0.0,  # Not available via Docker CLI without stats; left 0.0
        'wall_s': time.time() - start,
        'max_rss_mb': 0,
    }

    return {
        'ok': proc.returncode == 0 and not timed_out,
        'exit_code': int(proc.returncode),
        'timed_out': bool(timed_out),
        'stdout': stdout_s,
        'stderr': stderr_s,
        'bytes_stdout': int(bytes_stdout),
        'bytes_stderr': int(bytes_stderr),
        'truncated': bool(trunc_out or trunc_err),
        'usage': usage,
        'log_path': f"sandbox://sessions/{session_id}/full.log",
    }
