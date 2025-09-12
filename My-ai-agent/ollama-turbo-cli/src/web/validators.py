from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import re

from .reasons import ValidatorOutcome


_HIGH_IMPACT_TYPES = {"health", "law", "finance", "market", "safety", "policy"}


def _has(pattern: str, text: str) -> bool:
    try:
        return bool(re.search(pattern, text, flags=re.IGNORECASE))
    except Exception:
        return False


@dataclass
class ValidatorConfig:
    high_impact_types: set[str] = None  # type: ignore

    def __post_init__(self):
        if self.high_impact_types is None:
            self.high_impact_types = set(_HIGH_IMPACT_TYPES)


def validate_claim(claim, snapshot, cfg: ValidatorConfig | None = None) -> Tuple[List[ValidatorOutcome], float, bool]:
    """
    Return (outcomes, validator_score, high_impact_failed).
    - High-impact failure gates score to 0.0 per SPEC.
    - Otherwise, validator_score is average of non-neutral outcomes.
    """
    cfg = cfg or ValidatorConfig()
    text = getattr(snapshot, "normalized_content", "") or ""
    ctype = (getattr(claim, "claim_type", "") or "").strip().lower()

    outcomes: List[ValidatorOutcome] = []

    # Law / policy
    if ctype in {"law", "policy"}:
        pass_signals = [
            _has(r"\b(docket|case|v\.)\b", text),
            _has(r"\b(sec|doj|fcc|epa)\b", text),
            _has(r"\b(filed|ruling|order|opinion)\b", text),
        ]
        status = "pass" if any(pass_signals) else "needs_human"
        outcomes.append(ValidatorOutcome(validator="law_policy_presence", status=status))

    # Finance / market
    if ctype in {"finance", "market"}:
        pass_signals = [
            _has(r"\b(form\s*10-\w+|8-k|s-1|20-f)\b", text),
            _has(r"\b(nasdaq:|nyse:|\$[A-Z]{1,5})\b", text),
            _has(r"\b(sec\s+filing|edgar|cik)\b", text),
        ]
        status = "pass" if any(pass_signals) else "needs_human"
        outcomes.append(ValidatorOutcome(validator="finance_market_presence", status=status))

    # Health / science
    if ctype in {"health", "science"}:
        pass_signals = [
            _has(r"\b(doi:|doi\.org/|pubmed|pmid)\b", text),
            _has(r"\b(clinicaltrials\.gov|nct\d{8})\b", text),
            _has(r"\b(preprint|methods|protocol)\b", text),
        ]
        status = "pass" if any(pass_signals) else "needs_human"
        outcomes.append(ValidatorOutcome(validator="health_science_presence", status=status))

    # Crypto
    if ctype == "crypto":
        pass_signals = [
            _has(r"\b0x[0-9a-fA-F]{6,}\b", text),  # EVM-like tx or address
            _has(r"\b(block\s*height|txid|etherscan|mempool)\b", text),
        ]
        status = "pass" if any(pass_signals) else "needs_human"
        outcomes.append(ValidatorOutcome(validator="crypto_presence", status=status))

    # Safety / OSINT
    if ctype == "safety":
        pass_signals = [
            _has(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}\s*(AM|PM)?)\b", text),  # time/date present
            _has(r"\bin\s+[A-Z][a-zA-Z]+\b", text),  # coarse place
            _has(r"\b(geolocat|coordinates|lat|long)\b", text),
        ]
        status = "pass" if any(pass_signals) else "needs_human"
        outcomes.append(ValidatorOutcome(validator="safety_presence", status=status))

    # If we have no specific rules, mark as neutral
    if not outcomes:
        outcomes.append(ValidatorOutcome(validator="generic", status="needs_human"))

    # Compute score: pass=1, fail=0, needs_human=0.5; gate for high-impact failures (if any explicit 'fail')
    # In this initial pass, we only produce pass or needs_human. Keep fail path for future expansion.
    score_map = {"pass": 1.0, "needs_human": 0.5, "fail": 0.0}
    scores = [score_map.get(o.status, 0.5) for o in outcomes]
    avg_score = sum(scores) / len(scores) if scores else 0.5

    high_impact = ctype in cfg.high_impact_types
    any_fail = any(o.status == "fail" for o in outcomes)
    gated = high_impact and any_fail

    final_score = 0.0 if gated else avg_score
    return outcomes, final_score, gated
