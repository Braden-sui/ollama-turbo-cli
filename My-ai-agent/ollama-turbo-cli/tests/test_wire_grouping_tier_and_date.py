from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.web.pipeline as pipeline_mod
from src.web.pipeline import run_research
import src.web.allowlist_tiered as tier_mod


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_ALLOW_BROWSER", "0")
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    monkeypatch.setenv("WEB_WIRE_DEDUP_ENABLE", "1")
    monkeypatch.setenv("WEB_TIER_SWEEP", "0")
    yield


def _mk_search_result(title, url, snippet="", source="seed", published=None):
    return SimpleNamespace(title=title, url=url, snippet=snippet, source=source, published=published)


class DummyTiered:
    def __init__(self, mapping: dict[str, int]):
        self._map = mapping
        self.policy = {}
        self.seeds_by_cat = []
        self.seeds_by_tier = []

    def tier_for_host(self, h: str):
        h = (h or "").strip().lower()
        # match by suffix for convenience
        for dom, tv in self._map.items():
            if h == dom or h.endswith("." + dom):
                return tv
        return None

    def category_for_host(self, h: str):
        return None

    def discouraged_host(self, h: str) -> bool:
        return False


def _install_common_stubs(monkeypatch, urls: list[str], extract_pairs: dict[str, dict]):
    # search returns the provided urls
    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [_mk_search_result(f"T{i}", u) for i, u in enumerate(urls)],
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
    # extraction pairs controls markdown and date
    def _fake_extract(meta, **k):
        url = meta.get("final_url")
        pair = extract_pairs[url]
        return SimpleNamespace(
            ok=True,
            kind="html",
            markdown=pair["markdown"],
            title="T",
            date=pair["date"],
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


def test_wire_grouping_prefers_tier0_over_tier2(monkeypatch):
    u_t0 = "https://t0.example.com/a"
    u_t2 = "https://t2.example.net/a"
    urls = [u_t2, u_t0]

    # same content to ensure same fingerprint
    md = "Para1 text.\n\nPara two.\n\nPara three."
    pairs = {
        u_t0: {"markdown": md, "date": "2025-01-01T00:00:00Z"},
        u_t2: {"markdown": md, "date": "2025-01-01T00:00:00Z"},
    }
    _install_common_stubs(monkeypatch, urls, pairs)

    # Patch tiered allowlist to return tier 0 for t0.example.com, 2 for t2.example.net
    monkeypatch.setattr(tier_mod, "load_tiered_allowlist", lambda: DummyTiered({"t0.example.com": 0, "t2.example.net": 2}), raising=True)

    out = run_research("wire grouping tier pref", top_k=2, force_refresh=True)
    dbg = out.get("debug", {})
    wire = dbg.get("wire", {})
    groups = wire.get("groups") or []
    assert groups, "expected at least one wire group"
    g = groups[0]
    cits = out.get("citations", [])
    chosen = cits[g["canonical"]].get("canonical_url")
    assert chosen == u_t0


def test_wire_grouping_prefers_earliest_date_when_tie(monkeypatch):
    u1 = "https://t1.example.org/a"
    u2 = "https://t1.example.net/a"
    urls = [u1, u2]

    md = "Para1 text.\n\nPara two.\n\nPara three."
    # Equal tiers (both tier 1), equal body lengths, different dates
    pairs = {
        u1: {"markdown": md, "date": "2025-01-02T00:00:00Z"},  # later
        u2: {"markdown": md, "date": "2025-01-01T00:00:00Z"},  # earlier -> should win
    }
    _install_common_stubs(monkeypatch, urls, pairs)

    # Patch tier mapping - both tier 1
    monkeypatch.setattr(tier_mod, "load_tiered_allowlist", lambda: DummyTiered({"t1.example.org": 1, "t1.example.net": 1}), raising=True)

    out = run_research("wire grouping earliest date", top_k=2, force_refresh=True)
    dbg = out.get("debug", {})
    wire = dbg.get("wire", {})
    groups = wire.get("groups") or []
    assert groups, "expected at least one wire group"
    g = groups[0]
    cits = out.get("citations", [])
    chosen = cits[g["canonical"]].get("canonical_url")
    assert chosen == u2
