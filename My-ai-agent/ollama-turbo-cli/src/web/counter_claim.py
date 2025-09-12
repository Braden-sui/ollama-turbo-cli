from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re


_NEGATION_PATTERNS = [
    r"\b(denied|denies|deny)\b",
    r"\b(refuted|refute|refutes)\b",
    r"\b(false|not\s+true|no\s+evidence)\b",
    r"\b(disputed|disputes|contest|contested)\b",
    r"\b(allegedly|allegation)\b",
]


def evaluate_counter_claim(text: str, claims: List[Any]) -> Dict[str, Any]:
    """Heuristic counter-claim evaluator (scaffold).

    Returns a dict with score 0..1, signals, and contested boolean (debug only).
    """
    t = (text or "").lower()
    signals: List[str] = []
    score = 0.0
    for pat in _NEGATION_PATTERNS:
        try:
            if re.search(pat, t, flags=re.I):
                signals.append(pat)
        except Exception:
            continue
    if signals:
        # Bounded simple scoring proportional to matched families
        score = min(1.0, len(signals) / 3.0)
    contested = bool(score >= 0.7)
    return {"score": score, "signals": signals, "contested": contested}
