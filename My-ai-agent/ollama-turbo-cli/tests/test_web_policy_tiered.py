from types import SimpleNamespace
import os

from src.web.pipeline import run_research


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


def test_discouraged_domains_dropped(monkeypatch):
    # Search returns a discouraged source (dailymail)
    sr = SimpleNamespace(title="Bad", url="https://www.dailymail.co.uk/some/story", snippet="", source="fake", published=None)
    monkeypatch.setattr('src.web.pipeline.search', lambda *args, **kwargs: [sr])
    monkeypatch.setattr('src.web.pipeline.fetch_url', lambda url, **kwargs: _mk_ok_fetch(url))
    monkeypatch.setattr('src.web.pipeline.extract_content', lambda *a, **k: _mk_ok_extract("Bad"))
    monkeypatch.setattr('src.web.pipeline.rerank_chunks', lambda *a, **k: [])
    monkeypatch.setattr('src.web.pipeline.save_page_now', lambda *a, **k: {'archive_url': '', 'timestamp': ''})

    out = run_research('query', top_k=1, force_refresh=True)
    assert out.get('citations') == []


def test_first_party_newsroom_elevated_tier0(monkeypatch):
    # First-party newsroom path elevates to Tier 0 even if not on non-tier allowlist
    url = "https://www.microsoft.com/press/something"
    sr = SimpleNamespace(title="MS Press", url=url, snippet="", source="fake", published=None)
    monkeypatch.setattr('src.web.pipeline.search', lambda *args, **kwargs: [sr])
    monkeypatch.setattr('src.web.pipeline.fetch_url', lambda url, **kwargs: _mk_ok_fetch(url))
    monkeypatch.setattr('src.web.pipeline.extract_content', lambda *a, **k: _mk_ok_extract("MS Press"))
    monkeypatch.setattr('src.web.pipeline.rerank_chunks', lambda *a, **k: [])
    monkeypatch.setattr('src.web.pipeline.save_page_now', lambda *a, **k: {'archive_url': '', 'timestamp': ''})

    out = run_research('newsroom policy', top_k=1, force_refresh=True)
    cits = out.get('citations') or []
    assert len(cits) == 1
    cit = cits[0]
    assert cit.get('tier') == 0
    assert bool(cit.get('domain_trust')) is True


def test_social_sources_demoted_tier2_provisional(monkeypatch):
    # Social platform should be Tier 2 and provisional if alone
    url = "https://twitter.com/org/status/123"
    sr = SimpleNamespace(title="Tweet", url=url, snippet="", source="fake", published=None)
    monkeypatch.setattr('src.web.pipeline.search', lambda *args, **kwargs: [sr])
    monkeypatch.setattr('src.web.pipeline.fetch_url', lambda url, **kwargs: _mk_ok_fetch(url))
    monkeypatch.setattr('src.web.pipeline.extract_content', lambda *a, **k: _mk_ok_extract("Tweet"))
    monkeypatch.setattr('src.web.pipeline.rerank_chunks', lambda *a, **k: [])
    monkeypatch.setattr('src.web.pipeline.save_page_now', lambda *a, **k: {'archive_url': '', 'timestamp': ''})

    out = run_research('announcement', top_k=1, force_refresh=True)
    cits = out.get('citations') or []
    assert len(cits) == 1
    cit = cits[0]
    assert cit.get('tier') == 2
    assert cit.get('provisional') is True


def test_category_staleness_for_news_recency(monkeypatch):
    # Recency query with old Reuters article should be OUT_OF_WINDOW and dropped
    url = "https://www.reuters.com/world/example-old"
    sr = SimpleNamespace(title="Old Reuters", url=url, snippet="", source="fake", published=None)
    monkeypatch.setattr('src.web.pipeline.search', lambda *args, **kwargs: [sr])
    monkeypatch.setattr('src.web.pipeline.fetch_url', lambda url, **kwargs: _mk_ok_fetch(url))
    # Old date; category mapping for 'news_wires' default window is 7 days
    monkeypatch.setattr('src.web.pipeline.extract_content', lambda *a, **k: _mk_ok_extract("Old Reuters", date="2019-01-01T00:00:00Z"))
    monkeypatch.setattr('src.web.pipeline.rerank_chunks', lambda *a, **k: [])
    monkeypatch.setattr('src.web.pipeline.save_page_now', lambda *a, **k: {'archive_url': '', 'timestamp': ''})

    out = run_research('latest updates', top_k=1, force_refresh=True)
    assert out.get('citations') == []
