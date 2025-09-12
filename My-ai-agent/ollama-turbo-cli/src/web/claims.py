from __future__ import annotations

from dataclasses import dataclass
from typing import List
import hashlib
import re


@dataclass
class TextSpan:
    start: int
    end: int


@dataclass
class Qualifiers:
    time: str = ""
    place: str = ""
    scope: str = ""


@dataclass
class Claim:
    id: str
    subject: str
    predicate: str
    object: str
    qualifiers: Qualifiers
    claim_type: str  # law | science | health | finance | market | crypto | safety | policy | general
    source_snapshot_id: str
    text_span: TextSpan
    extraction_confidence: float  # 0..1


def _sha256_hex(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", "ignore"))
    return h.hexdigest()


_KEYWORD_TO_TYPE = {
    # law/policy
    "filed": "law",
    "complaint": "law",
    "lawsuit": "law",
    "charged": "law",
    "convicted": "law",
    "ruled": "law",
    "regulation": "policy",
    "bill": "policy",
    "legislation": "policy",
    # finance/market
    "acquired": "finance",
    "acquire": "finance",
    "acquisition": "finance",
    "merger": "finance",
    "ipo": "market",
    "raised": "finance",
    "earnings": "market",
    "revenue": "market",
    "stock": "market",
    # science/health
    "trial": "health",
    "fda": "health",
    "study": "science",
    "published": "science",
    # crypto/safety generic keywords (light)
    "blockchain": "crypto",
    "vulnerability": "safety",
}


def _heuristic_claim_type(subject: str, predicate: str, obj: str) -> str:
    t = f"{subject} {predicate} {obj}".lower()
    for k, v in _KEYWORD_TO_TYPE.items():
        if k in t:
            return v
    return "general"


_predicates = (
    "announced|launched|filed|acquired|acquire|acquisition|merged|raised|sued|charged|convicted|ruled|approved|recalled|published|reported"
)
# Optional article (The/A/An), then a capitalized subject, then predicate, then the rest until sentence end.
_SUBJ_VERB_RE = re.compile(
    rf"\b(?:(?:T|t)he\s+|(?:A|a)n?\s+)?([A-Z][\w&\.\- ]{{1,80}}?)\s+({ _predicates })\b(.*?)([\.!?]|$)",
    re.DOTALL,
)
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2}|(19|20)\d{2})\b")
_PLACE_RE = re.compile(r"\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b")
_SCOPE_RE = re.compile(r"\bagainst\s+([\w&\.\- ]{1,80})\b")


def _bounded_confidence(has_time: bool, has_place: bool, subj_len: int) -> float:
    base = 0.5
    if has_time:
        base += 0.15
    if has_place:
        base += 0.10
    if subj_len >= 3:
        base += 0.05
    return max(0.0, min(1.0, base))


def extract_claims_from_text(text: str, snapshot_id: str) -> List[Claim]:
    claims: List[Claim] = []
    if not text:
        return claims
    for m in _SUBJ_VERB_RE.finditer(text):
        start, end = m.start(0), m.end(0)
        subj = (m.group(1) or "").strip()
        pred = (m.group(2) or "").strip().lower()
        obj = (m.group(3) or "").strip()
        # Qualifiers via lightweight heuristics on the matched span
        span_text = text[start:end]
        tm = _DATE_RE.search(span_text)
        pl = _PLACE_RE.search(span_text)
        sc = _SCOPE_RE.search(span_text)
        qualifiers = Qualifiers(
            time=(tm.group(1) if tm else ""),
            place=(pl.group(1) if pl else ""),
            scope=(sc.group(1).strip() if sc else ""),
        )
        ctype = _heuristic_claim_type(subj, pred, obj)
        conf = _bounded_confidence(bool(tm), bool(pl), len(subj))
        cid = _sha256_hex(f"{snapshot_id}:{start}:{end}")
        claims.append(
            Claim(
                id=cid,
                subject=subj,
                predicate=pred,
                object=obj,
                qualifiers=qualifiers,
                claim_type=ctype,
                source_snapshot_id=snapshot_id,
                text_span=TextSpan(start=start, end=end),
                extraction_confidence=conf,
            )
        )
    return claims


def extract_claims(snapshot) -> List[Claim]:
    """Extract claims from a Snapshot without importing at module top (avoid cycles)."""
    # Snapshot has fields: id, normalized_content
    return extract_claims_from_text(getattr(snapshot, "normalized_content", ""), getattr(snapshot, "id", ""))
