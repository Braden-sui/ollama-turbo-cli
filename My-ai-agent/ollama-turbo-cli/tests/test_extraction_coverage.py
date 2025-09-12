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
    # Stabilize
    monkeypatch.setenv("WEB_TIER_SWEEP", "0")
    yield


def _mk_search_result(title, url, snippet="", source="seed", published=None):
    return SimpleNamespace(title=title, url=url, snippet=snippet, source=source, published=published)


def test_extraction_modes_histogram(monkeypatch):
    u1 = "https://site1.example.com/a"
    u2 = "https://site2.example.net/b"

    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [_mk_search_result("A", u1), _mk_search_result("B", u2)],
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

    def _extract(meta, **k):
        url = meta.get("final_url")
        if url == u1:
            return SimpleNamespace(
                ok=True,
                kind="html",
                markdown="Para1\n\nPara2\n\nPara3",
                title="T1",
                date="2025-01-01T00:00:00Z",
                meta={"lang": "en"},
                used={"trafilatura": True},
                risk="LOW",
                risk_reasons=[],
            )
        else:
            return SimpleNamespace(
                ok=True,
                kind="html",
                markdown="ParaA\n\nParaB\n\nParaC",
                title="T2",
                date="2025-01-01T00:00:00Z",
                meta={"lang": "en"},
                used={"readability": True},
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

    out = run_research("q-extract-modes", top_k=2, force_refresh=True)
    cits = out.get("citations", [])
    assert len(cits) == 2
    modes = [c.get("extraction_mode") for c in cits]
    assert set(modes) == {"trafilatura", "readability"}
    dbg_modes = (out.get("debug", {}).get("extract", {}).get("modes") or {})
    assert dbg_modes.get("trafilatura", 0) == 1
    assert dbg_modes.get("readability", 0) == 1


def test_extract_fail_by_host(monkeypatch):
    good = "https://ok.example.com/a"
    bad = "https://bad.example.net/b"

    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [_mk_search_result("G", good), _mk_search_result("B", bad)],
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

    def _extract(meta, **k):
        url = meta.get("final_url")
        if url == bad:
            return SimpleNamespace(ok=False)
        return SimpleNamespace(
            ok=True,
            kind="html",
            markdown="P1\n\nP2\n\nP3",
            title="TG",
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

    out = run_research("q-extract-fail", top_k=2, force_refresh=True)
    dbg = out.get("debug", {}).get("extract", {})
    assert dbg.get("fail_count", 0) >= 1
    fb = dbg.get("fail_by_host") or {}
    assert fb.get("bad.example.net", 0) >= 1
