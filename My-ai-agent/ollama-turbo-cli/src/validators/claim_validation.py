from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Set
import os
import re

from src.web.reasons import ValidatorOutcome


_DEFAULT_HIGH_IMPACT: Set[str] = {"health", "law", "finance", "market", "safety", "policy"}


@dataclass
class ValidatorConfig:
    high_impact_types: Set[str]

    @classmethod
    def from_yaml(cls) -> "ValidatorConfig":
        """Load config from the same family as the overlap checker (reliability.yaml + profile overlays).
        Falls back to sensible defaults when missing.
        """
        ycfg: Dict[str, Any] = {}
        try:
            import yaml  # type: ignore
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
            base_path = os.getenv('RELIABILITY_CFG') or os.path.join(base_dir, 'reliability.yaml')
            if os.path.isfile(base_path):
                with open(base_path, 'r', encoding='utf-8') as f:
                    ycfg = yaml.safe_load(f) or {}
            # Optional profile overlay
            profile = (os.getenv('GOVERNANCE_PROFILE') or os.getenv('CLAIM_VALIDATION_PROFILE') or '').strip().lower()
            if profile:
                prof_path = os.path.join(base_dir, 'profiles', f'{profile}.yaml')
                if os.path.isfile(prof_path):
                    with open(prof_path, 'r', encoding='utf-8') as f:
                        prof = yaml.safe_load(f) or {}
                        if isinstance(prof, dict):
                            # shallow merge under reliability
                            for k, v in (prof or {}).items():
                                if isinstance(v, dict) and isinstance(ycfg.get(k), dict):
                                    ycfg[k].update(v)
                                else:
                                    ycfg[k] = v
        except Exception:
            ycfg = {}
        rcfg = ((ycfg or {}).get('reliability') or {}).get('claim_validation') or {}
        hi = rcfg.get('high_impact_types') if isinstance(rcfg, dict) else None
        high_impact_types = set(hi) if isinstance(hi, (list, set, tuple)) else set(_DEFAULT_HIGH_IMPACT)
        return cls(high_impact_types=high_impact_types)


def _has(pattern: str, text: str) -> bool:
    try:
        return bool(re.search(pattern, text, flags=re.IGNORECASE))
    except Exception:
        return False


def validate_claim(claim, snapshot, cfg: ValidatorConfig | None = None) -> Tuple[List[ValidatorOutcome], float, bool]:
    """
    Claim validation registry entrypoint.
    Returns (outcomes, validator_score, gated_by_high_impact_fail).

    Semantics:
    - Use status 'heuristic_presence' for regex/content presence signals.
    - Use status 'needs_human' when insufficient signals are present.
    - Reserve 'fail' only for explicit authority mismatches (future PRs).
    - Only explicit 'fail' on a high-impact claim triggers gating.
    """
    cfg = cfg or ValidatorConfig.from_yaml()
    text = getattr(snapshot, "normalized_content", "") or ""
    ctype = (getattr(claim, "claim_type", "") or "").strip().lower()

    outcomes: List[ValidatorOutcome] = []

    # Law / policy
    if ctype in {"law", "policy"}:
        signals = [
            _has(r"\b(docket|case|v\.)\b", text),
            _has(r"\b(sec|doj|fcc|epa)\b", text),
            _has(r"\b(filed|ruling|order|opinion)\b", text),
        ]
        status = "heuristic_presence" if any(signals) else "needs_human"
        outcomes.append(ValidatorOutcome(validator="law_policy_presence", status=status))

    # Finance / market
    if ctype in {"finance", "market"}:
        signals = [
            _has(r"\b(form\s*10-\w+|8-k|s-1|20-f)\b", text),
            _has(r"\b(nasdaq:|nyse:|\$[A-Z]{1,5})\b", text),
            _has(r"\b(sec\s+filing|edgar|cik)\b", text),
        ]
        status = "heuristic_presence" if any(signals) else "needs_human"
        outcomes.append(ValidatorOutcome(validator="finance_market_presence", status=status))

    # Health / science
    if ctype in {"health", "science"}:
        signals = [
            _has(r"\b(doi:|doi\.org/|pubmed|pmid)\b", text),
            _has(r"\b(clinicaltrials\.gov|nct\d{8})\b", text),
            _has(r"\b(preprint|methods|protocol)\b", text),
        ]
        status = "heuristic_presence" if any(signals) else "needs_human"
        outcomes.append(ValidatorOutcome(validator="health_science_presence", status=status))

    # Crypto
    if ctype == "crypto":
        signals = [
            _has(r"\b0x[0-9a-fA-F]{6,}\b", text),
            _has(r"\b(block\s*height|txid|etherscan|mempool)\b", text),
        ]
        status = "heuristic_presence" if any(signals) else "needs_human"
        outcomes.append(ValidatorOutcome(validator="crypto_presence", status=status))

    # Safety / OSINT
    if ctype == "safety":
        signals = [
            _has(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}\s*(AM|PM)?)\b", text),
            _has(r"\bin\s+[A-Z][a-zA-Z]+\b", text),
            _has(r"\b(geolocat|coordinates|lat|long)\b", text),
        ]
        status = "heuristic_presence" if any(signals) else "needs_human"
        outcomes.append(ValidatorOutcome(validator="safety_presence", status=status))

    # Neutral when no rules matched
    if not outcomes:
        outcomes.append(ValidatorOutcome(validator="generic", status="needs_human"))

    # Score map aligned to semantics; can be tuned via calibration later
    score_map = {"heuristic_presence": 0.6, "needs_human": 0.5, "fail": 0.0}
    scores = [score_map.get(o.status, 0.5) for o in outcomes]
    avg_score = sum(scores) / len(scores) if scores else 0.5

    high_impact = ctype in cfg.high_impact_types
    any_fail = any(o.status == "fail" for o in outcomes)
    gated = high_impact and any_fail

    final_score = 0.0 if gated else avg_score
    return outcomes, final_score, gated
