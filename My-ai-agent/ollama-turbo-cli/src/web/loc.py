from __future__ import annotations

from typing import Dict, Any


def format_loc(h: Dict[str, Any]) -> str:
    # preference: explicit loc
    if 'loc' in h and h['loc']:
        return str(h['loc'])
    pg = h.get('page')
    ln = h.get('line')
    lstart = h.get('line_start')
    lend = h.get('line_end')
    if pg and lstart and lend:
        return f"p.{int(pg)} L{int(lstart)}-{int(lend)}"
    if pg and ln:
        return f"p.{int(pg)} L{int(ln)}"
    if pg:
        return f"p.{int(pg)}"
    if ln:
        return f"line {int(ln)}"
    return ""

