from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CheckMode = Literal["off", "warn", "enforce"]


@dataclass(frozen=True)
class Contract:
    ground: bool
    cite: bool
    check: CheckMode


# Dual-mode singletons
STANDARD = Contract(ground=False, cite=False, check="off")
RESEARCHER = Contract(ground=True, cite=True, check="enforce")


def to_contract(name: str) -> Contract:
    key = (name or "").strip().lower()
    if key == "researcher":
        return RESEARCHER
    return STANDARD


def classify_contract(message: str, forced: str | None = None) -> Contract:
    """Legacy keyword classifier preserved for back-compat.

    If `forced` is provided, returns that mapping via to_contract().
    Otherwise, a simple heuristic on the message.
    """
    if forced:
        return to_contract(forced)
    msg = (message or "").lower()
    risky = any(t in msg for t in (
        "today", "latest", "breaking", "2024", "2025", "fda", "sec", "lawsuit", "recall",
        "evidence", "report", "study", "revenue", "patients", "cite", "sources"
    ))
    digits = any(ch.isdigit() for ch in msg)
    if risky or digits:
        return RESEARCHER
    return STANDARD

