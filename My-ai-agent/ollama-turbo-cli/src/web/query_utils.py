from __future__ import annotations

import re
from typing import List, Set


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')

# Minimal stopword sets for "minimal" vs "standard"
_STOPWORDS_MINIMAL: Set[str] = {
    "the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "or", "vs", "versus"
}
_STOPWORDS_STANDARD: Set[str] = _STOPWORDS_MINIMAL.union({
    "is", "are", "does", "do", "what", "when", "where", "why", "how", "from", "by", "with", "that"
})


def preserve_quoted_and_numerals(q: str) -> List[str]:
    """Return a list of preserved tokens found in the query: quoted phrases and numerals (years)."""
    preserved: List[str] = []
    for m in _QUOTED_RE.finditer(q):
        phrase = m.group(1) or m.group(2)
        if phrase:
            preserved.append(phrase.strip())
    for m in _YEAR_RE.finditer(q):
        preserved.append(m.group(0))
    return preserved


def extract_phrases(q: str) -> List[str]:
    """
    Light-weight phrase extraction for patterns we care about.
    Returns phrases like 'X vs Y', 'difference between X and Y', 'X in Y'.
    """
    q_lower = q.lower()
    phrases: List[str] = []
    # common patterns using simple regexes
    m = re.search(r'([^,]+?)\s+(?:vs|versus)\s+([^,]+)', q_lower)
    if m:
        phrases.append((m.group(1).strip() + " vs " + m.group(2).strip()))
    m = re.search(r'difference between\s+(.+?)\s+(?:and)\s+(.+)', q_lower)
    if m:
        phrases.append("difference between " + m.group(1).strip() + " and " + m.group(2).strip())
    m = re.search(r'([^,]+?)\s+(?:in|for|of)\s+([^,]+)', q_lower)
    if m:
        phrases.append(m.group(1).strip() + " in " + m.group(2).strip())
    # dedupe and return
    unique: List[str] = []
    for p in phrases:
        if p not in unique:
            unique.append(p)
    return unique


def _tokenize(q: str) -> List[str]:
    return [t for t in re.split(r'\s+', (q or '').strip()) if t]


def generate_variants(q: str, mode: str = "aggressive", max_tokens: int = 6, stopword_profile: str = "standard") -> List[str]:
    """
    Generate fallback variants for a query. Returns list starting with the raw query.
    Modes:
      - off: only raw query
      - soft: preserve quotes/numerals, extract phrases, gentle pruning up to max_tokens
      - aggressive: legacy behavior (strip many stopwords, strip years, cap tokens)
    This function remains conservative for Commit 1; logic will be reused/expanded in Commit 2.
    """
    variants: List[str] = []
    raw = (q or '').strip()
    if not raw:
        return variants
    variants.append(raw)  # always include raw first

    if mode == "off":
        return variants

    # stopword selection
    stopwords = _STOPWORDS_STANDARD if (stopword_profile == "standard") else _STOPWORDS_MINIMAL

    preserved = preserve_quoted_and_numerals(q)

    if mode == "soft":
        # phrase extraction first
        for p in extract_phrases(q):
            if p and p not in variants:
                variants.append(p)
        # Next, construct token-preserving gentle prune:
        toks = _tokenize(q)
        # preserve quoted tokens and numerals
        preserved_tokens = set()
        for p in preserved:
            for tok in p.split():
                preserved_tokens.add(tok.lower())
        kept: List[str] = []
        for tok in toks:
            if tok.lower() in preserved_tokens or (tok.lower() not in stopwords):
                kept.append(tok)
            if len(kept) >= max_tokens:
                break
        candidate = " ".join(kept)
        if candidate and candidate not in variants:
            variants.append(candidate)
        return variants

    # aggressive (legacy): emulate current behavior — strip years + strong stopwords, cap tokens
    toks = [t for t in _tokenize(q) if not _YEAR_RE.match(t)]
    pruned = [t for t in toks if t.lower() not in stopwords]
    pruned = pruned[:max_tokens]
    candidate = " ".join(pruned)
    if candidate and candidate not in variants:
        variants.append(candidate)
    return variants

