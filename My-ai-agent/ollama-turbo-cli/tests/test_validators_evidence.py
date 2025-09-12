from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.web.pipeline as pipeline_mod
from src.web.pipeline import run_research
from src.web.snapshot import build_snapshot
from src.web.claims import extract_claims_from_text
from src.web.validators import validate_claim
from src.web.evidence import score_evidence


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_ALLOW_BROWSER", "0")
    monkeypatch.setenv("WEB_EMERGENCY_BOOTSTRAP", "0")
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    # Keep ef flags off unless a test turns them on
    monkeypatch.setenv("EVIDENCE_FIRST", "0")
    monkeypatch.setenv("EVIDENCE_FIRST_KILL_SWITCH", "1")
    yield


def test_validator_outcomes_law_policy_pass():
    text = "SEC filed a complaint against Example Corp on 2025-01-02."
    snap = build_snapshot(
        url="https://example.com/legal/1",
        headers={"content-type": "text/html"},
        normalized_content=text,
        fetched_at="2025-01-02T00:00:00Z",
    )
    claims = extract_claims_from_text(text, snap.id)
    assert claims, "expected at least one claim extracted"
    c = claims[0]
    outs, vscore, gated = validate_claim(c, snap)
    # Expect at least a needs_human or pass; and in this text, law/policy presence likely passes
    assert isinstance(outs, list) and outs
    assert vscore >= 0.5
    assert gated is False


def test_evidence_scoring_features():
    text = (
        '"Quote" and another "Quote" — see https://example.org/ref and http://example.com/a. '
        'Methods described in the protocol. Updated on 2025-06-01.'
    )
    snap = build_snapshot(
        url="https://example.com/research/1",
        headers={"content-type": "text/html"},
        normalized_content=text,
        fetched_at="2025-01-02T00:00:00Z",
    )
    score, feats = score_evidence(snap, [])
    assert 0.0 <= score <= 1.0
    assert feats.get("primary_link_count", 0) >= 2
    assert feats.get("primary_link_quality", 0) >= 0.5
    assert feats.get("date_internal_consistency", 0) in (0, 1)


def test_pipeline_attaches_ef_block_when_flags_enabled(monkeypatch):
    # Turn on evidence-first but disable kill switch to allow ef block
    monkeypatch.setenv("EVIDENCE_FIRST", "1")
    monkeypatch.setenv("EVIDENCE_FIRST_KILL_SWITCH", "0")

    # Stub search/fetch/extract/rerank to produce a single citation with simple content
    u = "https://www.reuters.com/world/example"
    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [SimpleNamespace(title="Reuters", url=u, snippet="", source="seed", published=None)],
        raising=True,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "fetch_url",
        lambda url, **k: SimpleNamespace(
            ok=True,
            status=200,
            url=url,
            final_url=url,
            headers={"content-type": "text/html"},
            content_type="text/html",
            body_path=None,
            meta_path=None,
            cached=False,
            browser_used=False,
            reason=None,
        ),
        raising=True,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "extract_content",
        lambda meta, **k: SimpleNamespace(
            ok=True,
            kind="html",
            markdown="The Company announced a product on 2025-01-01.",
            title="T",
            date="2025-01-01T00:00:00Z",
            meta={"lang": "en"},
            used={},
            risk="LOW",
            risk_reasons=[],
        ),
        raising=True,
    )
    monkeypatch.setattr(pipeline_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )

    out = run_research("test", top_k=1, force_refresh=True)
    cits = out.get("citations", [])
    assert cits and isinstance(cits[0], dict)
    ef = cits[0].get("ef")
    assert isinstance(ef, dict)
    assert "snapshot_id" in ef and "confidence_breakdown" in ef
