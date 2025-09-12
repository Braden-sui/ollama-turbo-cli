from __future__ import annotations

from src.validators.claim_validation import validate_claim, ValidatorConfig
from src.reliability.guards.validator import Validator as CitationValidator
from src.web.snapshot import build_snapshot
from src.web.claims import extract_claims_from_text


def test_dual_validators_coexist_without_interference():
    # Build a small snapshot and a claim
    text = "The SEC filed a complaint against ACME on 2025-01-01."
    snap = build_snapshot(
        url="https://example.com/legal/2",
        headers={"content-type": "text/html"},
        normalized_content=text,
        fetched_at="2025-01-01T00:00:00Z",
    )
    claims = extract_claims_from_text(text, snap.id)
    assert claims

    # Run claim validation
    outs, vscore, gated = validate_claim(claims[0], snap, ValidatorConfig.from_yaml())
    assert isinstance(outs, list)
    assert 0.0 <= vscore <= 1.0
    assert gated in (True, False)

    # Run citation overlap validator independently on a trivial sentence
    ov = CitationValidator(mode="warn")
    res = ov.validate("The court ruled [1].", context_blocks=[], citations_map={"1": {"highlights": [{"quote": "The court ruled"}]}})
    assert isinstance(res, dict)
    assert "details" in res and isinstance(res["details"], list)
