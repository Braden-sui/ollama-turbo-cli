from types import SimpleNamespace
import os

from src.web.pipeline import run_research


def _mk_sr(url: str, title: str = "Title"):
    return SimpleNamespace(title=title, url=url, snippet="", source="fake", published=None)


def _mk_ok_fetch(url: str):
    return SimpleNamespace(ok=True, url=url, final_url=url, status=200, content_type='text/html', body_path=None, headers={}, browser_used=False, reason=None)


def _mk_ok_extract(title: str, kind: str = 'html', date: str | None = None, markdown: str = 'body'):
    return SimpleNamespace(
        ok=True,
        kind=kind,
        markdown=markdown,
        title=title,
        date=date,
        meta={},
        used={},
        risk='LOW',
        risk_reasons=[],
    )


def test_tier_sweep_finds_trusted_seed(monkeypatch):
    # Ensure sweep is enabled and not strict for this test
    monkeypatch.setenv('WEB_TIER_SWEEP', '1')
    monkeypatch.setenv('WEB_TIER_SWEEP_MAX_SITES', '20')
    monkeypatch.setenv('WEB_TIER_SWEEP_STRICT', '0')
    monkeypatch.setenv('WEB_RECENCY_SOFT_ACCEPT_WHEN_EMPTY', '0')

    # Initial search returns only a social platform (Tier 2 provisional)
    def fake_search(query, *, cfg=None, site=None, freshness_days=None):
        if site is None:
            return [_mk_sr("https://twitter.com/org/status/1", "Tweet")]  # Tier 2
        # During sweep, return a trusted result when site is a Tier 0/1 seed
        if site in ("reuters.com", "sec.gov"):
            return [_mk_sr(f"https://{site}/good", "Trusted")]  # Tier 1 or 0
        return []
    monkeypatch.setattr('src.web.pipeline.search', fake_search)
    monkeypatch.setattr('src.web.pipeline.fetch_url', lambda url, **kwargs: _mk_ok_fetch(url))
    # Keep extraction simple (no date to avoid recency gating)
    monkeypatch.setattr('src.web.pipeline.extract_content', lambda *a, **k: _mk_ok_extract("Doc"))
    monkeypatch.setattr('src.web.pipeline.rerank_chunks', lambda *a, **k: [])
    monkeypatch.setattr('src.web.pipeline.save_page_now', lambda *a, **k: {'archive_url': '', 'timestamp': ''})

    out = run_research('test query', top_k=3, force_refresh=True)
    cits = out.get('citations') or []
    assert len(cits) >= 1
    # Expect at least one Tier 0/1 or allowlisted after sweep
    assert any((c.get('tier') in (0, 1)) or bool(c.get('domain_trust')) for c in cits)
    # If any Tier 2 citations remain, they should no longer be provisional once trusted present
    if any(c.get('tier') == 2 for c in cits):
        assert all(c.get('provisional') is False for c in cits if c.get('tier') == 2)


def test_tier_sweep_strict_drops_tier2_when_unresolved(monkeypatch):
    # Strict mode: if sweep cannot find Tier 0/1, drop Tier 2-only results
    monkeypatch.setenv('WEB_TIER_SWEEP', '1')
    monkeypatch.setenv('WEB_TIER_SWEEP_MAX_SITES', '4')
    monkeypatch.setenv('WEB_TIER_SWEEP_STRICT', '1')
    monkeypatch.setenv('WEB_RECENCY_SOFT_ACCEPT_WHEN_EMPTY', '0')

    def fake_search(query, *, cfg=None, site=None, freshness_days=None):
        if site is None:
            return [_mk_sr("https://twitter.com/org/status/2", "Tweet2")]
        # For sweep, return nothing to force unresolved trusted citations
        return []
    monkeypatch.setattr('src.web.pipeline.search', fake_search)
    monkeypatch.setattr('src.web.pipeline.fetch_url', lambda url, **kwargs: _mk_ok_fetch(url))
    monkeypatch.setattr('src.web.pipeline.extract_content', lambda *a, **k: _mk_ok_extract("Doc2"))
    monkeypatch.setattr('src.web.pipeline.rerank_chunks', lambda *a, **k: [])
    monkeypatch.setattr('src.web.pipeline.save_page_now', lambda *a, **k: {'archive_url': '', 'timestamp': ''})

    out = run_research('test query', top_k=3, force_refresh=True)
    # With strict on and no Tier 0/1 found, we should drop Tier 2-only citations
    assert out.get('citations') == []
