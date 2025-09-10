import os
from types import SimpleNamespace

import src.web.pipeline as pipeline_mod
from src.web.pipeline import run_research


def _mk_search_result(title, url, snippet="", source="seed", published=None):
    return SimpleNamespace(title=title, url=url, snippet=snippet, source=source, published=published)


def test_recency_soft_accept_when_empty(monkeypatch, tmp_path):
    # Enable soft-accept via env
    monkeypatch.setenv("WEB_RECENCY_SOFT_ACCEPT_WHEN_EMPTY", "1")
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_HEAD_GATING", "0")
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))

    # Search returns allowlisted Reuters URL with no published date
    u1 = "https://www.reuters.com/world/middle-east/example-soft-accept"
    monkeypatch.setattr(pipeline_mod, "search", lambda *a, **k: [_mk_search_result("Reuters", u1)], raising=True)

    # fetch: minimal OK result
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

    # extract: no date present (undated)
    monkeypatch.setattr(
        pipeline_mod,
        "extract_content",
        lambda meta, **k: SimpleNamespace(
            ok=True,
            kind="html",
            markdown="x\n",
            title="T",
            date=None,
            meta={"lang": "en"},
            used={},
            risk="LOW",
            risk_reasons=[],
        ),
        raising=True,
    )

    # rerank: trivial
    monkeypatch.setattr(
        pipeline_mod,
        "chunk_text",
        lambda s: [s],
        raising=True,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )

    out = run_research("recent events", top_k=2, freshness_days=7, force_refresh=True)
    cits = out.get("citations", [])
    assert isinstance(cits, list)
    # Soft-accepted undated Reuters citation should be present
    assert any((c.get("canonical_url") or "").startswith("https://www.reuters.com/") for c in cits)
    # Verify undated flag is set
    assert any(c.get("undated") for c in cits)
    # Debug flag indicating soft-accept path used
    dbg = out.get("debug", {})
    assert dbg.get("recency_soft_accept_used") is True
    assert dbg.get("undated_accepted_count", 0) >= 1
