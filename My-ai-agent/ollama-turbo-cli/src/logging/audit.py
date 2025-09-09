from __future__ import annotations

from typing import Any, Dict, List
import os
import json
from datetime import datetime, timezone


def _runs_dir() -> str:
    base = os.path.join(os.getcwd(), 'runs')
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return base


def write_audit_line(
    *,
    mode: str,
    query: str,
    answer: str,
    citations: List[Dict[str, Any]] | None,
    metrics: Dict[str, Any] | None,
    router: Dict[str, Any] | None,
) -> None:
    line = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'mode': mode,
        'query': query,
        'answer': answer,
        'citations': citations or [],
        'metrics': metrics or {},
        'router': router or {},
    }
    path = os.path.join(_runs_dir(), 'audit.jsonl')
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    except Exception:
        pass

