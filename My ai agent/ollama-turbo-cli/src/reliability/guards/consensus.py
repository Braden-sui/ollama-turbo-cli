from __future__ import annotations

from typing import Callable, Dict, Any, List


def run_consensus(generate_once: Callable[[], str], k: int = 3) -> Dict[str, Any]:
    """Run k generations and compute a simple majority vote.

    This is a stub: it returns empty candidates if k <= 0 or generate_once is not wired.
    """
    candidates: List[str] = []
    try:
        for _ in range(max(0, int(k))):
            candidates.append(generate_once())
    except Exception:
        candidates = []

    if not candidates:
        return {"final": None, "agree_rate": 0.0, "candidates": []}

    # Majority vote by exact string equality (placeholder logic)
    counts: Dict[str, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    final = max(counts.items(), key=lambda kv: kv[1])[0]
    agree_rate = counts[final] / len(candidates)
    return {"final": final, "agree_rate": agree_rate, "candidates": candidates}
