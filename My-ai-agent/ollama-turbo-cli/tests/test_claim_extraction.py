from __future__ import annotations

from src.web.snapshot import build_snapshot
from src.web.claims import extract_claims, extract_claims_from_text


def test_extracts_basic_claims_and_spans():
    text = (
        "The Company announced a new product on 2024-03-15 in San Francisco. "
        "SEC filed a complaint against XYZ Corp on 2025-02-01 in Washington. "
        "Acme Corp raised funding."
    )
    snap = build_snapshot(
        url="https://example.com/news/1",
        headers={"content-type": "text/html"},
        normalized_content=text,
        fetched_at="2025-01-01T00:00:00Z",
    )

    claims = extract_claims(snap)
    assert isinstance(claims, list) and len(claims) >= 2

    for c in claims:
        # Schema surface checks
        assert c.id and isinstance(c.id, str)
        assert c.subject and isinstance(c.subject, str)
        assert c.predicate and isinstance(c.predicate, str)
        assert isinstance(c.object, str)
        assert c.source_snapshot_id == snap.id
        assert 0 <= c.extraction_confidence <= 1
        # Span bounds
        assert 0 <= c.text_span.start <= c.text_span.end <= len(text)


def test_reproducible_ids_and_fields():
    text = (
        "The Court ruled on 2024-12-31 in New York. "
        "MegaCorp acquired StartUp Inc."
    )
    # Run twice with the same snapshot id to ensure identical claim ids and fields
    s_id = "SNAP-1"
    c1 = extract_claims_from_text(text, s_id)
    c2 = extract_claims_from_text(text, s_id)

    assert len(c1) == len(c2) >= 1

    for a, b in zip(c1, c2):
        assert a.id == b.id
        assert a.subject == b.subject
        assert a.predicate == b.predicate
        assert a.object == b.object
        assert a.qualifiers.time == b.qualifiers.time
        assert a.qualifiers.place == b.qualifiers.place
        assert a.qualifiers.scope == b.qualifiers.scope
        assert a.claim_type == b.claim_type
        assert a.text_span.start == b.text_span.start
        assert a.text_span.end == b.text_span.end
        assert a.extraction_confidence == b.extraction_confidence

    # Different snapshot id should change claim ids but keep fields equal
    s_id2 = "SNAP-2"
    c3 = extract_claims_from_text(text, s_id2)
    assert len(c1) == len(c3)
    for a, b in zip(c1, c3):
        assert a.id != b.id
        assert a.subject == b.subject
        assert a.predicate == b.predicate
        assert a.object == b.object
        assert a.qualifiers.time == b.qualifiers.time
        assert a.qualifiers.place == b.qualifiers.place
        assert a.qualifiers.scope == b.qualifiers.scope
        assert a.claim_type == b.claim_type
        assert a.text_span.start == b.text_span.start
        assert a.text_span.end == b.text_span.end
        assert a.extraction_confidence == b.extraction_confidence
