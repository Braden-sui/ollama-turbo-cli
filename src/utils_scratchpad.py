from __future__ import annotations
import os
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    """Return repository root directory.
    This file lives in `src/`, so the repo root is the parent of `src/`.
    """
    return Path(__file__).resolve().parents[1]


def get_scratch_host_dir() -> Path:
    """Return the host directory used for persistent scratch storage.

    Respects SANDBOX_SCRATCH_HOST_DIR; defaults to <repo>/.sandbox/scratch.
    Ensures the directory exists.
    """
    env = os.getenv("SANDBOX_SCRATCH_HOST_DIR")
    if env:
        p = Path(env).expanduser().resolve()
    else:
        p = _repo_root() / ".sandbox" / "scratch"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_filename(name: Optional[str]) -> str:
    name = (name or "scratchpad.txt").strip()
    # very simple sanitization: no path separators, limit length
    name = name.replace("\\", "_").replace("/", "_")
    if not name:
        name = "scratchpad.txt"
    if len(name) > 128:
        name = name[:128]
    return name


def get_scratch_file_path(filename: Optional[str]) -> Path:
    return get_scratch_host_dir() / sanitize_filename(filename)
