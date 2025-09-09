import os
import json
from types import SimpleNamespace
from datetime import datetime, timedelta

import pytest

import src.web.pipeline as pipeline_mod
from src.web.pipeline import run_research
from src.web.config import WebConfig


def _mk_search_result(title, url, snippet="", source="seed", published=None):
    return SimpleNamespace(title=title, url=url, snippet=snippet, source=source, published=published)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    # Isolate cache, quiet robots/archive
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_HEAD_GATING", "0")
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    monkeypatch.setenv("WEB_ARCHIVE_ENABLED", "0")
    monkeypatch.setenv("WEB_EXCLUDE_CITATION_DOMAINS", "")
    yield


def test_recent_window_filters_by_date(monkeypatch):
    now = datetime.utcnow()
    within = (now - timedelta(days=2)).isoformat() + "Z"
    outside = (now - timedelta(days=10)).isoformat() + "Z"

    u1 = "https://www.reuters.com/world/middle-east/example-1"
    u2 = "https://www.bbc.com/news/example-2"
    u3 = "https://www.theguardian.com/world/example-3"

    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [
            _mk_search_result("Reuters", u1),
            _mk_search_result("BBC", u2),
            _mk_search_result("Guardian", u3),
        ],
        raising=True,
    )

    # fetch returns HTML stub
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

    def _fake_extract(meta, **k):
        url = meta.get("final_url")
        if url == u1:
            return SimpleNamespace(ok=True, kind="html", markdown="x\n", title="T1", date=within, meta={"lang": "en"}, used={}, risk="LOW", risk_reasons=[])
        if url == u2:
            return SimpleNamespace(ok=True, kind="html", markdown="x\n", title="T2", date=outside, meta={"lang": "en"}, used={}, risk="LOW", risk_reasons=[])
        return SimpleNamespace(ok=True, kind="html", markdown="x\n", title="T3", date=None, meta={"lang": "en"}, used={}, risk="LOW", risk_reasons=[])

    monkeypatch.setattr(pipeline_mod, "extract_content", _fake_extract, raising=True)
    monkeypatch.setattr(pipeline_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )

    out = run_research("What happened recently in Gaza?", top_k=5, freshness_days=7, force_refresh=True)
    cits = out.get("citations", [])
    urls = [c.get("canonical_url") for c in cits]
    # Only within-window URL should remain (u1)
    assert u1 in urls
    assert u2 not in urls
    assert u3 not in urls
    dbg = out.get("debug", {})
    assert isinstance(dbg, dict)
    # Discards should be recorded
    disc = dbg.get("discard", {})
    assert disc.get("missing_dateline", 0) >= 1
    assert disc.get("dateline_out_of_window", 0) >= 1


def test_liveblog_page_is_discarded_pre_fetch(monkeypatch):
    live = "https://www.bbc.com/news/live/world-middle-east-123456"
    art = "https://www.reuters.com/world/middle-east/example-4"

    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [_mk_search_result("BBC Live", live), _mk_search_result("Reuters", art)],
        raising=True,
    )

    # Only article will be fetched and extracted
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
        lambda meta, **k: SimpleNamespace(ok=True, kind="html", markdown="x\n", title="T", date=datetime.utcnow().isoformat()+"Z", meta={"lang": "en"}, used={}, risk="LOW", risk_reasons=[]),
        raising=True,
    )
    monkeypatch.setattr(pipeline_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )

    out = run_research("latest updates this week", top_k=5, freshness_days=7, force_refresh=True)
    urls = [c.get("canonical_url") for c in out.get("citations", [])]
    assert live not in urls
    assert art in urls


def test_js_map_source_is_discarded(monkeypatch):
    m = "https://liveuamap.com/"
    art = "https://www.aljazeera.com/news/example-5"

    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [_mk_search_result("Map", m), _mk_search_result("AJ", art)],
        raising=True,
    )

    # Fetch/extract only used for article
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
        lambda meta, **k: SimpleNamespace(ok=True, kind="html", markdown="x\n", title="T", date=datetime.utcnow().isoformat()+"Z", meta={"lang": "en"}, used={}, risk="LOW", risk_reasons=[]),
        raising=True,
    )
    monkeypatch.setattr(pipeline_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )

    out = run_research("recent events", top_k=5, freshness_days=7, force_refresh=True)
    urls = [c.get("canonical_url") for c in out.get("citations", [])]
    assert m not in urls
    assert art in urls


def test_only_articles_with_valid_dates_survive(monkeypatch):
    now = datetime.utcnow()
    within = (now - timedelta(days=3)).isoformat() + "Z"

    u1 = "https://www.haaretz.com/middle-east-news/example-6"
    u2 = "https://www.timesofisrael.com/example-7"
    u3 = "https://www.wsj.com/world/middle-east/example-8"

    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [
            _mk_search_result("Haaretz", u1),
            _mk_search_result("TOI", u2),
            _mk_search_result("WSJ", u3),
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

    def _fake_extract2(meta, **k):
        url = meta.get("final_url")
        if url in {u1, u2}:
            return SimpleNamespace(ok=True, kind="html", markdown="x\n", title="T", date=within, meta={"lang": "en"}, used={}, risk="LOW", risk_reasons=[])
        return SimpleNamespace(ok=True, kind="html", markdown="x\n", title="T3", date=None, meta={"lang": "en"}, used={}, risk="LOW", risk_reasons=[])

    monkeypatch.setattr(pipeline_mod, "extract_content", _fake_extract2, raising=True)
    monkeypatch.setattr(pipeline_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )

    out = run_research("recent developments in Israel", top_k=5, freshness_days=7, force_refresh=True)
    urls = [c.get("canonical_url") for c in out.get("citations", [])]
    assert u1 in urls and u2 in urls
    assert u3 not in urls
