from __future__ import annotations

"""
Minimal progress bus for web_research to emit progress events that can be
forwarded as SSE from the API without invasive changes.

Usage:
  - API route registers a channel id -> queue and runs the chat turn in a
    background thread.
  - The background thread attaches to the same channel id before executing
    tools. The web_research pipeline emits progress via emit_current().
  - The API thread polls the queue and relays events as SSE 'progress'.
"""

import queue
import threading
from typing import Dict, Optional, Any

_channels: Dict[str, "queue.Queue[dict]"] = {}
_lock = threading.Lock()
_tls = threading.local()


def register(channel_id: str) -> "queue.Queue[dict]":
    q: "queue.Queue[dict]" = queue.Queue()
    with _lock:
        _channels[channel_id] = q
    return q


def unregister(channel_id: str) -> None:
    with _lock:
        _channels.pop(channel_id, None)


def attach_current(channel_id: str) -> None:
    setattr(_tls, "channel_id", channel_id)


def detach_current() -> None:
    if hasattr(_tls, "channel_id"):
        delattr(_tls, "channel_id")


def emit(channel_id: str, event: dict) -> None:
    with _lock:
        q = _channels.get(channel_id)
    if q is not None:
        try:
            q.put_nowait(event)
        except Exception:
            pass


def emit_current(event: dict) -> None:
    cid = getattr(_tls, "channel_id", None)
    if isinstance(cid, str) and cid:
        emit(cid, event)

