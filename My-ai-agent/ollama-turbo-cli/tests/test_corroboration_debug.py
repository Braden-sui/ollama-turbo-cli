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
    # Enable EF and corroboration
    monkeypatch.setenv("EVIDENCE_FIRST", "1")
    monkeypatch.setenv("EVIDENCE_FIRST_KILL_SWITCH", "0")
    monkeypatch.setenv("WEB_CORROBORATE_ENABLE", "1")
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    yield


def test_corroborators_attached_in_ef(monkeypatch):
    # Two distinct URLs that produce the same claim key
    u1 = "https://www.reuters.com/world/example-1"
    u2 = "https://apnews.com/article/example-1"
    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [
            SimpleNamespace(title="T1", url=u1, snippet="", source="seed", published=None),
            SimpleNamespace(title="T2", url=u2, snippet="", source="seed", published=None),
        ],
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
    text = "The Court ruled on 2025-01-01 in Washington."
    monkeypatch.setattr(
        pipeline_mod,
        "extract_content",
        lambda meta, **k: SimpleNamespace(
            ok=True,
            kind="html",
            markdown=text,
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
        lambda q, c, **k: [
            {"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}
        ],
        raising=True,
    )

    out = run_research("q", top_k=2, force_refresh=True)
    cits = out.get("citations", [])
    assert len(cits) == 2
    ef0 = cits[0].get("ef", {})
    ef1 = cits[1].get("ef", {})
    # Corroborators should reference the other index
    cor0 = (ef0.get("reasons") or {}).get("corroborators") or []
    cor1 = (ef1.get("reasons") or {}).get("corroborators") or []
    assert any(i in (0,1) for i in cor0)
    assert any(i in (0,1) for i in cor1)
