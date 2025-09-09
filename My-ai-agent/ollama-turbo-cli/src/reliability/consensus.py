from __future__ import annotations

from typing import List


def aggregate_by_consensus(outputs: List[str], *, min_count: int = 2) -> str:
    """Very small consensus aggregator: keep sentences appearing in ≥ min_count outputs."""
    from collections import Counter
    sent_lists = []
    for out in outputs or []:
        parts = [s.strip() for s in out.split('.') if s.strip()]
        sent_lists.append(parts)
    flat = [s for parts in sent_lists for s in parts]
    counts = Counter(flat)
    kept = [s for s, c in counts.items() if c >= min_count]
    return '. '.join(kept) + ('.' if kept else '')

