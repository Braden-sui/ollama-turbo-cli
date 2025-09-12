from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.web.search as search_mod
import src.web.pipeline as pipeline_mod
from src.web.pipeline import run_research


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_ALLOW_BROWSER", "0")
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    monkeypatch.setenv("WEB_TIER_SWEEP", "0")
    yield


def test_throttle_events_reduce_concurrency(monkeypatch):
    # Force drain to start empty
    search_mod._drain_throttle_events()

    # Stub search._search_duckduckgo_fallback to simulate a throttle event via recorder
    def _fake_search(query, *, cfg=None, site=None, freshness_days=None):
        search_mod._record_throttle('duckduckgo', 429, 1.0)
        return [SimpleNamespace(title='A', url='https://example.com/a', snippet='', source='throttled', published=None)]

    monkeypatch.setattr(pipeline_mod, 'search', _fake_search, raising=True)
    # Fetch and extract standard stubs
    monkeypatch.setattr(
        pipeline_mod,
        "fetch_url",
        lambda url, **k: SimpleNamespace(
            ok=True,
            status=200,
            url=url,
            final_url=url,
            headers={"content-type": "text/html", "x-debug-ttfb-ms": "1", "x-debug-ttc-ms": "2"},
            content_type="text/html",
            body_path=None,
            meta_path=None,
            cached=False,
            browser_used=False,
            reason=None,
        ),
        raising=True,
    )
    monkeypatch.setattr(pipeline_mod, "extract_content", lambda meta, **k: SimpleNamespace(ok=True, kind="html", markdown="P1\n\nP2\n\nP3", title="T", date="2025-01-01T00:00:00Z", meta={}, used={"trafilatura": True}, risk="LOW", risk_reasons=[]), raising=True)
    monkeypatch.setattr(pipeline_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )

    out = run_research("throttle", top_k=1, force_refresh=True)
    dbg_search = (out.get("debug", {}).get("search") or {})
    events = dbg_search.get("provider_throttle_events") or []
    knobs = dbg_search.get("knob_changes") or {}
    assert events and any(e.get("provider") == 'duckduckgo' for e in events)
    assert knobs.get("reduced_concurrency_to") in (1, 2)
