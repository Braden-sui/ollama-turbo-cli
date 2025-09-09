from __future__ import annotations

from typing import Tuple
import re


def defaults_for_query(query: str) -> Tuple[int, int]:
    s = (query or '').lower()
    newsy = any(k in s for k in ("today", "this week", "latest"))
    newsy = newsy or bool(re.search(r"\b(2024|2025|2026|2027|2028|2029|2030|2031|2032|2033|2034|2035|2036|2037|2038|2039)\b", s))
    newsy = newsy or bool(re.search(r"\bq[1-4]\s*20\d{2}\b", s))
    if newsy:
        return 5, 90
    return 7, 365

