from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class ValidatorOutcome:
    validator: str
    status: str  # pass|fail|needs_human
    notes: str = ""


@dataclass
class Reasons:
    snapshot_id: str
    features: Dict[str, Any] = field(default_factory=dict)
    validator_outcomes: List[ValidatorOutcome] = field(default_factory=list)
    corroborators: List[Dict[str, Any]] = field(default_factory=list)
    reputation_inputs: Dict[str, Any] = field(default_factory=dict)
    confidence_breakdown: Dict[str, float] = field(default_factory=lambda: {
        "evidence": 0.0,
        "validators": 0.0,
        "corroboration": 0.0,
        "prior": 0.0,
        "final_score": 0.0,
    })
    notes: List[str] = field(default_factory=list)


def make_empty_reasons(snapshot_id: str) -> Reasons:
    return Reasons(snapshot_id=snapshot_id)
