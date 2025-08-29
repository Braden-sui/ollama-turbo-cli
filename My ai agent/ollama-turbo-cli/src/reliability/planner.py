from __future__ import annotations

from typing import Dict, Optional
import re


def auto_reliability_flags(message: str, *, overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Heuristically derive reliability flags from a user message.

    This shim is intentionally small and deterministic. It favors clarity over
    sophistication and is safe to call anywhere. Callers can pass `overrides`
    to pre-set specific flags.

    Rules (applied in order):
    - If the message suggests citations/sources, enable ground+cite.
    - If it suggests verification/fact-check, set validator to 'warn'.
    - If it suggests voting/consensus, enable consensus with k=3.
    - Numeric hints like "k=5" will be parsed when present.
    - Engine/eval_corpus are never inferred here.
    """
    text = (message or "").lower()

    flags: Dict[str, object] = {
        "ground": False,
        "k": None,
        "cite": False,
        "check": "off",
        "consensus": False,
        "engine": None,
        "eval_corpus": None,
    }

    # Citations / sources
    if any(w in text for w in ["cite", "citation", "citations", "source", "sources", "references", "reference"]):
        flags["ground"] = True
        flags["cite"] = True

    # Verification / validator
    if any(w in text for w in ["verify", "verification", "fact-check", "fact check", "validate", "double-check", "double check"]):
        flags["check"] = "warn"

    # Consensus / vote
    if any(w in text for w in ["consensus", "vote", "majority", "ensemble"]):
        flags["consensus"] = True
        # Default to k=3 unless explicitly provided
        flags["k"] = 3

    # Parse explicit k hint like `k=5` or `k: 4`
    m = re.search(r"\bk\s*[:=]\s*(\d{1,2})\b", text)
    if m:
        try:
            k_val = int(m.group(1))
            if 1 <= k_val <= 16:
                flags["k"] = k_val
        except Exception:
            pass

    # Apply overrides last
    if overrides:
        for k, v in overrides.items():
            if k in flags:
                flags[k] = v  # type: ignore[assignment]

    return flags
