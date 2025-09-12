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
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    monkeypatch.setenv("WEB_CUTOVER_PREP", "1")
    yield


def test_debug_schema_has_version_and_sections(monkeypatch):
    u = "https://example.com/a"
    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [SimpleNamespace(title="T1", url=u, snippet="", source="seed", published=None)],
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
            markdown="x",
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

    out = run_research("q", top_k=1, force_refresh=True)
    dbg = out.get("debug", {})
    assert isinstance(dbg.get("schema_version"), int)
    # Presence checks only
    for key in ("search", "fetch", "tier", "metrics"):
        assert key in dbg
    # cutover prep emits deprecation counter dict
    assert isinstance(dbg.get("deprecation"), dict)
