import os
import json
import math
import re
from typing import List, Dict, Iterable, Optional, Tuple

import pytest

from src.reliability.retrieval.pipeline import RetrievalPipeline


# ---------------- Helpers: matching and metrics ----------------

_WORD = re.compile(r"\b\w+\b", flags=re.IGNORECASE)

def _norm(s: str) -> str:
    s = (s or "").lower()
    # normalize common punctuation variants so "retry-after" ~= "retry after"
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(s: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD.finditer(_norm(s))]

def _contains_phrase(hay: str, needle: str) -> bool:
    # phrase match with word boundaries after normalization
    h = _norm(hay)
    n = _norm(needle)
    if not n:
        return False
    # exact phrase search bounded by spaces or string ends
    pattern = r"(?:^|\s)" + re.escape(n) + r"(?:\s|$)"
    return re.search(pattern, h) is not None

def _variants(term: str) -> List[str]:
    t = _norm(term)
    # generate a few cheap variants: with/without hyphen, plural naive
    candidates = {t}
    if " " in t:
        candidates.add(t.replace(" ", ""))  # e.g., "retry after" -> "retryafter"
    if t.endswith("s"):
        candidates.add(t[:-1])
    else:
        candidates.add(t + "s")
    return sorted(candidates)

# A "concept" is a set of synonyms/variants; matching any variant satisfies the concept once.
# Example: ["retry-after", "retry after"], ["etag", "entity tag"]
def count_concepts_supported(row: Dict[str, str], concept_groups: List[List[str]]) -> int:
    text = f"{row.get('title') or ''} {row.get('text') or ''}"
    supported = 0
    for group in concept_groups:
        # expand minor variants automatically so input can stay simple
        expanded: List[str] = []
        for g in group:
            expanded.extend(_variants(g))
        if any(_contains_phrase(text, v) for v in expanded):
            supported += 1
    return supported

def rr_at_k(bools: Iterable[bool], k: int) -> float:
    for idx, hit in enumerate(bools):
        if idx >= k:
            break
        if hit:
            return 1.0 / (idx + 1)
    return 0.0

def dcg_at_k(gains: Iterable[float], k: int) -> float:
    dcg = 0.0
    for i, g in enumerate(gains):
        if i >= k:
            break
        denom = math.log2(i + 2)  # log2(rank+1)
        dcg += g / denom
    return dcg

# -------------- Golden set (two modes): qrels or heuristic --------------

def _load_qrels(path: str) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Optional qrels JSON: { "query_id": { "doc_id": gain, ... }, ... }
    Gain can be 1 or graded relevance like 2/3.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _doc_id(row: Dict[str, str]) -> str:
    # prefer explicit id if your pipeline returns it; otherwise derive a stable hash-like id
    if "doc_id" in row and row["doc_id"]:
        return str(row["doc_id"])
    title = row.get("title") or ""
    text = row.get("text") or ""
    # cheap stable fingerprint without importing hashlib
    return f"id::{len(title)}::{len(text)}::{(title[:16] + text[:16]).lower()}"

# -------------- Test data --------------

# Keep your existing queries but treat each term as its own concept group by default.
GOLDEN_QUERIES: List[Tuple[str, List[List[str]]]] = [
    ("how to handle 429 too many requests with retry-after?",
     [["429"], ["too many requests"], ["retry-after", "retry after"]]),
    ("what are oauth scopes and tokens best practices?",
     [["oauth"], ["scope", "scopes"], ["token", "tokens"]]),
    ("how to paginate large collections?",
     [["cursor"], ["next_page_token", "next page token"], ["pagination", "paginate"]]),
    ("how to use etag and last-modified for caching?",
     [["etag", "entity tag"], ["last-modified", "last modified"], ["cache-control", "cache control"]]),
    ("what is exponential backoff with jitter?",
     [["exponential backoff"], ["jitter"], ["thundering herds", "thundering herd"]]),
    ("how do idempotency keys help with retries?",
     [["idempotency", "idempotent"], ["idempotency-key", "idempotency key"], ["unsafe http methods", "unsafe methods"]]),
    ("how to deal with partial failures in bulk operations?",
     [["partial"], ["per-item status", "per item status"], ["retry queues", "dead letter", "dlq"]]),
    ("validate inputs using json schema?",
     [["json schema", "json-schema"], ["draft-07", "draft 07", "draft7"], ["required properties", "required fields"]]),
    ("what does tls ensure and what to verify?",
     [["tls"], ["certificate", "cert"], ["hostname", "host name", "sni"]]),
    ("how to verify webhook signatures?",
     [["webhook"], ["hmac"], ["shared secret", "secret header", "signature header"]]),
]

DOCS_PATH = os.path.join("tests", "data", "golden_corpus.jsonl")
QRELS_PATH = os.path.join("tests", "data", "golden_qrels.json")

MIN_CONCEPTS = int(os.getenv("RAG_MIN_CONCEPTS", "2"))
TOPK = int(os.getenv("RAG_TOPK", "3"))
REQ_HIT_RATE = float(os.getenv("RAG_MIN_HIT_RATE", "0.8"))
REQ_MRR = float(os.getenv("RAG_MIN_MRR", "0.6"))
REQ_NDCG = float(os.getenv("RAG_MIN_NDCG", "0.6"))


# -------------- Tests --------------

@pytest.mark.parametrize("query,concepts", GOLDEN_QUERIES)
def test_topk_has_support(query: str, concepts: List[List[str]]):
    rp = RetrievalPipeline()
    out = rp.run(query, k=TOPK, docs_glob=DOCS_PATH, min_score=None)
    assert out, f"no retrieval results for query: {query}"
    hits = [count_concepts_supported(r, concepts) >= MIN_CONCEPTS for r in out[:TOPK]]
    assert any(hits), f"no relevant support in top-{TOPK} for query: {query}"

def test_suite_quality_thresholds():
    rp = RetrievalPipeline()
    qrels = _load_qrels(QRELS_PATH)

    hit_rate_hits = 0
    rr_values: List[float] = []
    ndcg_values: List[float] = []

    for query, concepts in GOLDEN_QUERIES:
        out = rp.run(query, k=TOPK, docs_glob=DOCS_PATH, min_score=None) or []
        # Heuristic relevance for hit-rate & MRR
        bools = [count_concepts_supported(r, concepts) >= MIN_CONCEPTS for r in out[:TOPK]]
        if any(bools):
            hit_rate_hits += 1
        rr_values.append(rr_at_k(bools, TOPK))

        # If qrels present, compute graded nDCG; else approximate from heuristic (gains 1 for True)
        if qrels is not None:
            # You may key qrels by the literal query; adapt to query_id if you prefer
            relmap = qrels.get(query, {})
            gains = [float(relmap.get(_doc_id(r), 0.0)) for r in out[:TOPK]]
            ideal = sorted(relmap.values(), reverse=True)[:TOPK]
            ndcg = 0.0
            if any(ideal):
                ndcg = dcg_at_k(gains, TOPK) / max(1e-9, dcg_at_k(ideal, TOPK))
            ndcg_values.append(ndcg)
        else:
            gains = [1.0 if b else 0.0 for b in bools]
            ideal = sorted(gains, reverse=True)
            ndcg = 0.0 if not any(ideal) else dcg_at_k(gains, TOPK) / max(1e-9, dcg_at_k(ideal, TOPK))
            ndcg_values.append(ndcg)

    total = len(GOLDEN_QUERIES)
    hit_rate = hit_rate_hits / max(1, total)
    mean_rr = sum(rr_values) / max(1, total)
    mean_ndcg = sum(ndcg_values) / max(1, total)

    assert hit_rate >= REQ_HIT_RATE, f"Hit-rate@{TOPK} {hit_rate_hits}/{total} < {REQ_HIT_RATE:.2f}"
    assert mean_rr >= REQ_MRR, f"Mean RR@{TOPK} {mean_rr:.3f} < {REQ_MRR:.2f}"
    assert mean_ndcg >= REQ_NDCG, f"Mean nDCG@{TOPK} {mean_ndcg:.3f} < {REQ_NDCG:.2f}"
