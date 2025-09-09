from __future__ import annotations

from typing import List, Dict, Any, Tuple
import re
import os


# Keep domain stopwords; consider down-weighting later instead of dropping.
STOPWORDS = {
    'the','and','but','with','from','into','that','this','those','these','were','was','are','is','for','have','has',
    'company','shipped','announced','launched','reported','reports','report'
}

# Allowlist short-but-meaningful tokens
def _short_allow() -> List[str]:
    env = os.getenv('OVERLAP_SHORT_ALLOW')
    if env:
        return [t.strip().lower() for t in env.split(',') if t.strip()]
    return ['ai','uk','us','eu','fda','sec','q1','q2','q3','q4','ios','gpu','api','llm']

_NUM_RE = re.compile(r"[$€£]?\d{1,3}(?:[\u00A0\u202F ,]\d{3})*(?:[\.,]\d+)?%?|\d+(?:[\.,]\d+)?%?")
_WORD_RE = re.compile(r"[a-z]{3,}")
_SHORT_RE = re.compile(r"\b(?:ai|uk|us|eu|fda|sec|q[1-4]|ios|gpu|api|llm)\b", re.I)
_QY_RE = re.compile(r"\bq([1-4])\s*20(\d{2})\b", re.I)
_QY2_RE = re.compile(r"\b(?:f?y\s*)?q([1-4])\s*['’]?(\d{2})\b", re.I)


def _normalize_numeric(tok: str) -> str:
    t = tok.lower()
    # strip currency sign if present
    if t and t[0] in {'$', '€', '£'}:
        t = t[1:]
    pct = t.endswith('%')
    if pct:
        t = t[:-1]
    # remove thin spaces/non-breaking spaces and normal spaces inside numbers
    t = t.replace('\u00A0', '').replace('\u202F', '').replace(' ', '')
    # comma decimal handling (e.g., 3,1%)
    if ',' in t and '.' not in t and os.getenv('OVERLAP_ALLOW_COMMA_DECIMALS', '1').strip().lower() not in {'0','false','no','off'}:
        # if there is exactly one comma, treat as decimal separator
        if t.count(',') == 1:
            t = t.replace(',', '.')
        else:
            t = t.replace(',', '')
    else:
        t = t.replace(',', '')
    # optional suffix multipliers
    if os.getenv('OVERLAP_ENABLE_SUFFIX_NORMALIZATION', '0').strip().lower() in {'1','true','yes','on'}:
        mult = 1
        if t.endswith('bn') and t[:-2].replace('.', '', 1).isdigit():
            mult = 1_000_000_000
            t = t[:-2]
        elif t.endswith('b') and t[:-1].replace('.', '', 1).isdigit():
            mult = 1_000_000_000
            t = t[:-1]
        elif t.endswith('m') and t[:-1].replace('.', '', 1).isdigit():
            mult = 1_000_000
            t = t[:-1]
        elif t.endswith('k') and t[:-1].replace('.', '', 1).isdigit():
            mult = 1_000
            t = t[:-1]
        try:
            if mult != 1:
                v = float(t)
                t = str(int(v * mult))
        except Exception:
            pass
    # Optional: collapse 400k/4k style if used in your data
    # if t.endswith('k') and t[:-1].isdigit():
    #     t = str(int(t[:-1]) * 1000)
    return (t + '%') if pct else t


def _tokens(s: str) -> List[str]:
    s = (s or '').lower()
    out: List[str] = []
    # normalized quarter-year composite
    for m in _QY_RE.finditer(s):
        out.append(f"q{m.group(1)}-20{m.group(2)}")
    for m in _QY2_RE.finditer(s):
        yy = m.group(2)
        out.append(f"q{m.group(1)}-20{yy}")
    # numbers
    for n in _NUM_RE.findall(s):
        out.append(_normalize_numeric(n))
    # long words
    out.extend(_WORD_RE.findall(s))
    # short allowlist
    # allowlist from env overrides
    out.extend(t.lower() for t in _SHORT_RE.findall(s))
    for tshort in _short_allow():
        if tshort in s:
            out.append(tshort)
    # stopword filter
    return [t for t in out if t not in STOPWORDS]


def _split_num_text(tokens: List[str]) -> Tuple[set, set]:
    nums, text = set(), set()
    for t in tokens:
        (nums if any(ch.isdigit() for ch in t) else text).add(t)
    return nums, text


def sentence_quote_overlap(claim_sentence: str, highlights: List[Dict[str, Any]], *, threshold: float = 0.18) -> float:
    """
    Returns the best token-recall style overlap between the claim and any quote:
        overlap = |claim ∩ quote| / |claim|
    Numerics are normalized (commas, percents). The 'threshold' is exposed for callers
    but not enforced here to preserve backward compatibility—use overlap_ok(...) for gating.
    """
    ctoks = set(_tokens(claim_sentence))
    if not ctoks:
        return 0.0

    best = 0.0
    cnum, _ = _split_num_text(list(ctoks))

    for h in (highlights or []):
        qtoks = set(_tokens(h.get('quote') or ''))
        if not qtoks:
            continue
        inter = len(ctoks & qtoks)
        score = inter / max(1, len(ctoks))
        # Prefer quotes that also match at least one numeric if the claim has numerics
        if cnum:
            qnum, _ = _split_num_text(list(qtoks))
            if not (cnum & qnum):
                # numeric mismatch: discount so word-only matches rarely pass
                score *= 0.25
        if score > best:
            best = score
    return best


def overlap_ok(claim_sentence: str, highlights: List[Dict[str, Any]], *, threshold: float = 0.18) -> bool:
    """Gate helper: returns True iff overlap >= threshold."""
    return sentence_quote_overlap(claim_sentence, highlights, threshold=threshold) >= threshold
