from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import threading


@dataclass
class SessionState:
    last_mode: str = "standard"
    last_score: float = 0.0
    lock_remaining: int = 0


_SESSIONS: Dict[str, SessionState] = {}
_LOCK = threading.Lock()


def get_session(session_id: str) -> SessionState:
    sid = session_id or "default"
    with _LOCK:
        st = _SESSIONS.get(sid)
        if st is None:
            st = SessionState()
            _SESSIONS[sid] = st
        return st

