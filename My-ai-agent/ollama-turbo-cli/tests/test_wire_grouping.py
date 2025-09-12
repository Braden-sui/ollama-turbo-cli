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
    # Enable wire dedup preview
    monkeypatch.setenv("WEB_WIRE_DEDUP_ENABLE", "1")
    # Stabilize ordering
    monkeypatch.setenv("WEB_TIER_SWEEP", "0")
    yield


def _mk_search_result(title, url, snippet="", source="seed", published=None):
    return SimpleNamespace(title=title, url=url, snippet=snippet, source=source, published=published)


def test_wire_grouping_prefers_longer_body_when_fingerprint_same(monkeypatch):
    u1 = "https://news.example.com/a"
    u2 = "https://news.example.net/a"

    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [_mk_search_result("A", u1), _mk_search_result("B", u2)],
        raising=True,
    )

    # Fetch returns HTML stub
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

    base = "Para one.\n\nPara two content.\n\nPara three content."
    longer = base + "\n\nadvertisement\n\ncookie notice"  # filtered from fingerprint but counts to body length

    def _fake_extract(meta, **k):
        url = meta.get("final_url")
        md = base if url == u1 else longer
        return SimpleNamespace(
            ok=True,
            kind="html",
            markdown=md,
            title="T",
            date="2025-01-01T00:00:00Z",
            meta={"lang": "en"},
            used={},
            risk="LOW",
            risk_reasons=[],
        )

    monkeypatch.setattr(pipeline_mod, "extract_content", _fake_extract, raising=True)
    monkeypatch.setattr(pipeline_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )

    out = run_research("wire grouping", top_k=2, force_refresh=True)
    dbg = out.get("debug", {})
    wire = dbg.get("wire", {})
    assert isinstance(wire, dict)
    groups = wire.get("groups") or []
    assert groups, "expected at least one wire group"
    g0 = groups[0]
    canonical_idx = g0.get("canonical")
    cits = out.get("citations", [])
    assert cits and isinstance(canonical_idx, int)
    chosen_url = cits[canonical_idx].get("canonical_url")
    # longer body (u2) should be chosen when fingerprint matches
    assert chosen_url == u2
