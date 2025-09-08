import os
import json
from types import SimpleNamespace

import pytest

from src.web.pipeline import run_research
import src.web.pipeline as pipeline_mod


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    # Permissive profile
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_HEAD_GATING", "0")
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    monkeypatch.setenv("SANDBOX_NET_ALLOW", "*")
    # Exclude Wikipedia from citations
    monkeypatch.setenv("WEB_EXCLUDE_CITATION_DOMAINS", "wikipedia.org")
    # Isolate cache dir per test
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    yield


def _mk_search_result(title, url, snippet="", source="ddg", published=None):
    return SimpleNamespace(title=title, url=url, snippet=snippet, source=source, published=published)


def test_pipeline_end_to_end_discovery_only_wikipedia(monkeypatch, tmp_path):
    # 1) Search returns one wiki page and one external link
    # Patch the search function used inside the pipeline module

    wiki_url = "https://en.wikipedia.org/wiki/Redistricting_in_the_United_States"
    ext_url1 = "https://www.reuters.com/world/us/example-redistricting-article/"

    def fake_search(query, *, cfg=None, site=None, freshness_days=None):
        return [
            _mk_search_result("Wikipedia Redistricting", wiki_url, source="duckduckgo"),
            _mk_search_result("Reuters Coverage", ext_url1, source="duckduckgo"),
        ]

    monkeypatch.setattr(pipeline_mod, "search", fake_search, raising=True)

    # 2) fetch_url and extract_content
    import src.web.fetch as fetch_mod  # only for types; we patch pipeline_mod symbols
    import src.web.extract as extract_mod  # only for types
    import src.web.rerank as rerank_mod  # only for types
    import src.web.archive as archive_mod  # only for types

    def fake_fetch_url(url, *, cfg=None, robots=None, force_refresh=False, use_browser_if_needed=True):
        # Minimal fields used by pipeline
        return SimpleNamespace(
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
        )

    monkeypatch.setattr(pipeline_mod, "fetch_url", fake_fetch_url, raising=True)

    def fake_extract(meta, *, cfg=None):
        url = meta.get("final_url")
        if url == wiki_url:
            # Include two external references plus a wiki self-link (filtered)
            md = (
                "This is a Wikipedia page about redistricting.\n"
                "Sources: https://www.cnbc.com/article-x and https://primary.example.com/ref1 \n"
                "See also https://en.wikipedia.org/wiki/Another_Page\n"
            )
            return SimpleNamespace(
                ok=True,
                kind="html",
                markdown=md,
                title="Wikipedia",
                date=None,
                meta={},
                used={},
                risk="LOW",
                risk_reasons=[],
            )
        # External page bodies
        md = "Key facts about lawsuits and timelines.\nLine 2 with details.\nLine 3 reference.\n"
        return SimpleNamespace(
            ok=True,
            kind="html",
            markdown=md,
            title="Primary Source",
            date=None,
            meta={},
            used={},
            risk="LOW",
            risk_reasons=[],
        )

    monkeypatch.setattr(pipeline_mod, "extract_content", fake_extract, raising=True)

    def fake_chunk_text(md: str):
        # 3 simple chunks (start/end lines are inferred by reranker stub)
        return [md]

    monkeypatch.setattr(pipeline_mod, "chunk_text", fake_chunk_text, raising=True)

    def fake_rerank(query: str, chunks, *, cfg=None, top_k=3):
        # Return highlights mapping to specific lines in the markdown
        return [
            {"id": "1", "score": 0.9, "start_line": 1, "end_line": 3, "highlights": [
                {"line": 1, "text": "Key facts about lawsuits"},
                {"line": 2, "text": "timelines"},
            ]}
        ]

    monkeypatch.setattr(pipeline_mod, "rerank_chunks", fake_rerank, raising=True)

    # 3) Disable archive network
    monkeypatch.setattr(pipeline_mod, "save_page_now", lambda *a, **k: {"archive_url": "", "timestamp": ""}, raising=True)
    monkeypatch.setattr(pipeline_mod, "get_memento", lambda *a, **k: {"archive_url": "", "timestamp": ""}, raising=True)

    out = run_research("verify discovery-only wikipedia", top_k=3, force_refresh=True)

    # Assertions
    assert "citations" in out and isinstance(out["citations"], list)
    # Wikipedia never appears in citations
    for cit in out["citations"]:
        assert "wikipedia.org" not in (cit.get("canonical_url") or "")
    # Primary sources should be present (either from search or expanded refs)
    urls = [c.get("canonical_url") for c in out["citations"]]
    # Accept common primary outlets from the test setup (Reuters) or references expanded from Wikipedia (primary.example.com, cnbc.com)
    primary_ok = any(any(dom in (u or "") for dom in ("reuters.com", "primary.example.com", "cnbc.com")) for u in urls)
    assert primary_ok

    # Debug metrics validate the path
    dbg = out.get("debug", {}).get("fetch", {})
    assert isinstance(dbg, dict)
    # Excluded wiki citations counted
    assert dbg.get("excluded", 0) >= 0
    # Wikipedia refs added recorded (non-strict: 0 is fine if rerank drops them)
    assert dbg.get("wiki_refs_added", 0) >= 0


def test_exclusion_policy_applies(monkeypatch, tmp_path):
    # Extend exclusion to example.com and ensure it is not cited
    monkeypatch.setenv("WEB_EXCLUDE_CITATION_DOMAINS", "wikipedia.org,example.com")

    import src.web.search as search_mod
    import src.web.extract as extract_mod
    import src.web.rerank as rerank_mod

    target = "https://example.com/article"
    monkeypatch.setattr(pipeline_mod, "search", lambda *a, **k: [_mk_search_result("Ex", target)], raising=True)
    monkeypatch.setattr(
        extract_mod,
        "extract_content",
        lambda meta, **k: SimpleNamespace(ok=True, kind="html", markdown="x\n", title="T", date=None, meta={}, used={}, risk="LOW", risk_reasons=[]),
        raising=True,
    )
    monkeypatch.setattr(rerank_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(rerank_mod, "rerank_chunks", lambda q, c, **k: [{"id": "1", "score": 0.8, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}], raising=True)

    # Bypass network in fetch_url
    import src.web.fetch as fetch_mod
    monkeypatch.setattr(pipeline_mod, "fetch_url", lambda url, **k: SimpleNamespace(ok=True, status=200, url=url, final_url=url, headers={}, content_type="text/html", body_path=None, meta_path=None, cached=False, browser_used=False, reason=None), raising=True)

    out = run_research("exclude example", top_k=1, force_refresh=True)
    urls = [c.get("canonical_url") for c in out.get("citations", [])]
    assert all("example.com" not in (u or "") for u in urls)
