from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.web.pipeline as pipeline_mod
from src.web.pipeline import run_research


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_ALLOW_BROWSER", "0")
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    # Enable EF and disable kill switch to produce EF data
    monkeypatch.setenv("EVIDENCE_FIRST", "1")
    monkeypatch.setenv("EVIDENCE_FIRST_KILL_SWITCH", "0")
    # Stabilize
    monkeypatch.setenv("WEB_TIER_SWEEP", "0")
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    yield


def _mk_sr(url: str) -> SimpleNamespace:
    return SimpleNamespace(title=url.split("//",1)[-1], url=url, snippet="", source="seed", published=None)


def test_contradiction_pairs_and_confidence_components(monkeypatch):
    u1 = "https://example.com/ann"
    u2 = "https://example.net/ann"
    monkeypatch.setattr(pipeline_mod, "search", lambda *a, **k: [_mk_sr(u1), _mk_sr(u2)], raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        "fetch_url",
        lambda url, **k: SimpleNamespace(
            ok=True,
            status=200,
            url=url,
            final_url=url,
            headers={"content-type": "text/html", "x-debug-ttfb-ms": "5", "x-debug-ttc-ms": "10"},
            content_type="text/html",
            body_path=None,
            meta_path=None,
            cached=False,
            browser_used=False,
            reason=None,
        ),
        raising=True,
    )

    def _extract(meta, **k):
        url = meta.get("final_url")
        if url == u1:
            md = "Acme announced merger."
        else:
            md = "Acme announced not merger."
        return SimpleNamespace(
            ok=True,
            kind="html",
            markdown=md,
            title="T",
            date="2025-01-01T00:00:00Z",
            meta={"lang": "en"},
            used={"trafilatura": True},
            risk="LOW",
            risk_reasons=[],
        )

    monkeypatch.setattr(pipeline_mod, "extract_content", _extract, raising=True)
    monkeypatch.setattr(pipeline_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )

    out = run_research("ef hygiene", top_k=2, force_refresh=True)
    ef = (out.get("debug", {}).get("ef") or {})
    pairs = ef.get("contradiction_pairs") or []
    assert pairs, "expected contradiction pairs to be detected"
    comp = ef.get("confidence_components") or {}
    assert set(comp.keys()) >= {"evidence", "validators", "corroboration", "prior", "final_score"}
    note = ef.get("confidence_note")
    assert isinstance(note, str)
