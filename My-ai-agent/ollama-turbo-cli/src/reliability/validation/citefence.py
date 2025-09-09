from __future__ import annotations

from typing import Dict, Any, List, Set
import re


FACTY_HINTS = {"study", "report", "announced", "launched", "filed", "revenue", "patient", "patients", "evidence"}


def _has_numerals_or_facty(text: str) -> bool:
    s = text.lower()
    if any(t in s for t in FACTY_HINTS):
        return True
    return bool(re.search(r"(\d|%|\$|€|£|¥)", s))


def validate(answer: str, citations_map: Dict[str, Any]) -> Dict[str, Any]:
    used: Set[str] = set(re.findall(r"\[(\d+)\]", answer or ""))
    # Check unknown markers
    unknown = [n for n in used if n not in citations_map]
    issues: List[str] = []
    for n in unknown:
        issues.append(f"unknown-citation-[{n}]")

    # Scan paragraphs for facty/numerals without citations
    paras = re.split(r"\n{2,}", answer or "")
    for p in paras:
        if not p.strip():
            continue
        if _has_numerals_or_facty(p):
            if not re.search(r"\[\d+\]\s*$", p.strip()) and not re.search(r"\[\d+\]", p):
                issues.append("facty-without-citation")

    ok = len(issues) == 0
    return {"ok": ok, "issues": issues, "used": sorted(list(used), key=lambda x: int(x))}

