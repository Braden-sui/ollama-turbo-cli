from __future__ import annotations

from typing import Dict, Any, List, Tuple
import re


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def score_evidence(snapshot, claims: List[Any]) -> Tuple[float, Dict[str, Any]]:
    """Minimal evidence scoring scaffold (PR3).

    Returns (score, features). Everything is best-effort and bounded 0..1.
    This is intentionally simple and will be replaced/expanded in later PRs.
    """
    text = getattr(snapshot, "normalized_content", "") or ""

    # Primary link count and quality (https share)
    links = re.findall(r"https?://[\w\-._~:/%?#\[\]@!$&'()*+,;=]+", text)
    https_links = [u for u in links if u.lower().startswith("https://")]
    primary_link_count = len(links)
    primary_link_quality = (len(https_links) / primary_link_count) if primary_link_count else 0.0
    primary_link_count_norm = _clamp01(primary_link_count / 10.0)

    # Named entity resolution proxy: fraction of claims having a non-empty subject
    total_claims = len(claims) if claims else 0
    ner_rate = 0.0
    if total_claims:
        ner_rate = sum(1 for c in claims if getattr(c, "subject", "").strip()) / total_claims

    # Quote to paraphrase proxy: fraction of quoted characters up to a cap
    quotes = text.count('"') + text.count("'")
    qpp_ratio = _clamp01(quotes / max(1, len(text) // 80))  # coarse, bounded

    # Date internal consistency: presence of ISO-like dates
    has_date = 1.0 if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text) else 0.0

    # Method transparency and corrections indicators (light proxies)
    method_transparency = 1.0 if re.search(r"\b(method|protocol|methods)\b", text, re.I) else 0.0
    corrections_updates = 1.0 if re.search(r"\b(correction|updated)\b", text, re.I) else 0.0

    # Combine a small set of features
    feats = {
        "primary_link_count": primary_link_count,
        "primary_link_quality": round(primary_link_quality, 3),
        "primary_link_count_norm": round(primary_link_count_norm, 3),
        "named_entity_resolution_rate": round(ner_rate, 3),
        "quote_to_paraphrase_ratio": round(qpp_ratio, 3),
        "date_internal_consistency": has_date,
        "method_transparency": method_transparency,
        "on_page_corrections_or_updates": corrections_updates,
    }

    # Simple average of bounded features as an initial evidence_score
    parts = [
        primary_link_count_norm,
        primary_link_quality,
        ner_rate,
        qpp_ratio,
        has_date,
        method_transparency,
        corrections_updates,
    ]
    evidence_score = _clamp01(sum(parts) / len(parts))
    return evidence_score, feats
