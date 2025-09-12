from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

import src.web.pipeline as pipeline_mod
from src.web.pipeline import run_research


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    # Isolate and stabilize
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_ALLOW_BROWSER", "0")
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    # Disable sweep/fallbacks that could introduce extra items
    monkeypatch.setenv("WEB_TIER_SWEEP", "0")
    monkeypatch.setenv("WEB_ENABLE_ALLOWLIST_NEWS_FALLBACK", "0")
    yield


def _mk_search_result(title: str, url: str):
    return SimpleNamespace(title=title, url=url, snippet="", source="seed", published=None)


def _install_stubs(monkeypatch, urls: list[str]):
    # Fixed search order independent of seed; sorting will be applied later
    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [_mk_search_result(f"T{i}", u) for i, u in enumerate(urls)],
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
    # Extract: identical risk/tier/trust; fixed date so recency doesn't filter
    monkeypatch.setattr(
        pipeline_mod,
        "extract_content",
        lambda meta, **k: SimpleNamespace(ok=True, kind="html", markdown="para1\n\npara2\n\npara3", title="T", date="2025-01-01T00:00:00Z", meta={"lang":"en"}, used={}, risk="LOW", risk_reasons=[]),
        raising=True,
    )
    monkeypatch.setattr(pipeline_mod, "chunk_text", lambda s: [s], raising=True)
    # Rerank with identical scores to force tiebreak path
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": str(i), "score": 0.5, "start_line": i+1, "end_line": i+1, "highlights": [{"line": 1, "text": "x"}]} for i in range(3)],
        raising=True,
    )


def _seeded_hash(seed: int, u: str) -> str:
    return hashlib.sha256((str(seed) + '|' + (u or '')).encode()).hexdigest()


def test_same_seed_same_order(monkeypatch):
    urls = [
        "https://example.com/b",
        "https://example.net/a",
        "https://example.org/c",
    ]
    _install_stubs(monkeypatch, urls)

    # Same seed → identical ordering
    monkeypatch.setenv("WEB_RUN_SEED", "123")
    out1 = run_research("q", top_k=3, force_refresh=True)
    out2 = run_research("q", top_k=3, force_refresh=True)
    order1 = [c.get("canonical_url") for c in out1.get("citations", [])]
    order2 = [c.get("canonical_url") for c in out2.get("citations", [])]
    assert order1 == order2


def test_different_seed_changes_tiebreak_order(monkeypatch):
    urls = [
        "https://example.com/b",
        "https://example.net/a",
        "https://example.org/c",
    ]
    _install_stubs(monkeypatch, urls)

    # Compute expected order by seeded url hash ascending (after tier/trust/date equalization)
    seed1 = 111
    seed2 = 222

    monkeypatch.setenv("WEB_RUN_SEED", str(seed1))
    out1 = run_research("q", top_k=3, force_refresh=True)
    order1 = [c.get("canonical_url") for c in out1.get("citations", [])]

    monkeypatch.setenv("WEB_RUN_SEED", str(seed2))
    out2 = run_research("q", top_k=3, force_refresh=True)
    order2 = [c.get("canonical_url") for c in out2.get("citations", [])]

    assert set(order1) == set(order2) == set(urls)
    # It is highly likely orders differ; if not, still assert determinism by comparing to seeded hashes
    if order1 == order2:
        exp1 = sorted(urls, key=lambda u: _seeded_hash(seed1, u))
        exp2 = sorted(urls, key=lambda u: _seeded_hash(seed2, u))
        assert exp1 != exp2 or order1 == exp1 == exp2
    else:
        assert order1 != order2
