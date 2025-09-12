from __future__ import annotations

from typing import Dict, List, Any, Tuple, Set


def claim_key(claim: Any) -> str:
    """Build a simple, normalized key for a claim.

    This is intentionally coarse for the initial scaffold and can be upgraded
    later to include qualifiers and fuzzy matching.
    """
    try:
        s = (getattr(claim, 'subject', '') or '').strip().lower()
        p = (getattr(claim, 'predicate', '') or '').strip().lower()
        o = (getattr(claim, 'object', '') or '').strip().lower()
        return f"{s}|{p}|{o}"
    except Exception:
        return ""


def compute_corroboration(claim_keys_by_citation: List[Tuple[int, List[str]]]) -> Dict[int, Dict[str, Any]]:
    """Compute corroboration groups across citations.

    Returns mapping: citation_idx -> { 'by_key': {key: [other_indices]}, 'all_corrob': [idx, ...] }
    """
    key_to_indices: Dict[str, List[int]] = {}
    for idx, keys in (claim_keys_by_citation or []):
        for k in (keys or []):
            if not k:
                continue
            key_to_indices.setdefault(k, []).append(idx)

    out: Dict[int, Dict[str, Any]] = {}
    for k, indices in key_to_indices.items():
        if len(indices) <= 1:
            continue
        for idx in indices:
            ent = out.setdefault(idx, {'by_key': {}, 'all_corrob': []})
            others = [i for i in indices if i != idx]
            ent['by_key'][k] = others
    # Populate all_corrob de-duplicated
    for idx, ent in out.items():
        agg: Set[int] = set()
        for arr in ent['by_key'].values():
            for j in arr:
                agg.add(j)
        ent['all_corrob'] = sorted(agg)
    return out
