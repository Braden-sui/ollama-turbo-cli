from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

import src.web.pipeline as pipeline_mod
from src.web.pipeline import run_research
import src.web.allowlist_tiered as tier_mod


class DummyTiered:
    def __init__(self, seeds_by_tier: list[tuple[str, int]]):
        self.seeds_by_tier = seeds_by_tier
        self.seeds_by_cat = []
        self.patterns_by_tier = []
        self.patterns_by_cat = []
        self.discouraged = []
        self.policy = {}

    def tier_for_host(self, h: str):
        h = (h or "").strip().lower()
        for seed, tier in self.seeds_by_tier:
            s = seed.strip().lower()
            if h == s or h.endswith("." + s):
                return tier
        return None

    def category_for_host(self, h: str):
        return None

    def discouraged_host(self, h: str) -> bool:
        return False


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    # Enable tier sweep and adaptive mode with a small cap for speed and determinism
    monkeypatch.setenv("WEB_TIER_SWEEP", "1")
    monkeypatch.setenv("WEB_TIER_SWEEP_ADAPTIVE_ENABLE", "1")
    monkeypatch.setenv("WEB_TIER_SWEEP_INITIAL_SITES", "1")
    monkeypatch.setenv("WEB_TIER_SWEEP_MAX_SITES", "2")
    monkeypatch.setenv("WEB_TIER_SWEEP_MAX_SITES_CAP", "2")
    # Fast quota: just 1 host within 24h window
    monkeypatch.setenv("WEB_TIER_SWEEP_QUOTA_FAST_COUNT", "1")
    monkeypatch.setenv("WEB_TIER_SWEEP_QUOTA_FAST_HOURS", "24")
    # General env stabilization
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_ALLOW_BROWSER", "0")
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    # Deterministic seed for site ordering
    monkeypatch.setenv("WEB_RUN_SEED", "12345")
    yield


def _seeded_hash(seed: int, s: str) -> str:
    return hashlib.sha256((str(seed) + '|' + (s or '')).encode()).hexdigest()


def _install_search_stub(monkeypatch, mapping: dict[str, dict]):
    # mapping: host -> {tier: int, date: str}
    def _search(q, *, cfg=None, site: str | None = None, freshness_days=None):
        if site:
            u = f"https://{site}/a"
            return [SimpleNamespace(title=site, url=u, snippet="", source="site", published=None)]
        # initial search: empty results to force sweep path
        return []

    def _extract(meta: dict, **k):
        url = meta.get("final_url") or meta.get("url")
        host = url.split("//", 1)[-1].split("/", 1)[0]
        # strip subdomain to apex match
        hh = host.strip().lower()
        # identify mapping by exact site key (we set seeds as hosts)
        params = mapping[hh]
        return SimpleNamespace(
            ok=True,
            kind="html",
            markdown="Para one.\n\nPara two.\n\nPara three.",
            title="T",
            date=params["date"],
            meta={"lang": "en"},
            used={},
            risk="LOW",
            risk_reasons=[],
        )

    monkeypatch.setattr(pipeline_mod, "search", _search, raising=True)
    monkeypatch.setattr(pipeline_mod, "extract_content", _extract, raising=True)
    monkeypatch.setattr(pipeline_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        pipeline_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )


def test_adaptive_sweep_escalates_then_meets_quota(monkeypatch):
    seed = 12345
    # Two candidate seeds; decide order by seeded hash to ensure first is 'bad', second 'good'
    a = "badseed.example.com"
    b = "goodseed.example.com"
    ordered = sorted([a, b], key=lambda s: _seeded_hash(seed, s))
    first, second = ordered[0], ordered[1]

    # First processed host is tier 2 (not counting toward quota); second is tier 1 (counts)
    seeds = [(a, 2), (b, 1)]
    monkeypatch.setattr(tier_mod, "load_tiered_allowlist", lambda: DummyTiered(seeds), raising=True)

    # Dates far in the future to satisfy fast window
    mapping = {
        a: {"tier": 2, "date": "2100-01-01T00:00:00Z"},
        b: {"tier": 1, "date": "2100-01-01T00:00:00Z"},
    }
    _install_search_stub(monkeypatch, mapping)

    out = run_research("q", top_k=3, force_refresh=True)
    dbg = out.get("debug", {})
    tier_dbg = (dbg.get("tier") or {})
    escalations = tier_dbg.get("escalation_events") or []
    # Expect one escalation from 1 -> 2
    assert escalations and escalations[-1].get("from") == 1 and escalations[-1].get("to") == 2


def test_adaptive_sweep_no_escalation_when_quota_initial(monkeypatch):
    # Two tier 1 seeds; initial budget 1 won't meet fast_count=1 unless the first is tier 1.
    # Set initial sites to 2 to satisfy fast quota in the first pass (no escalation).
    monkeypatch.setenv("WEB_TIER_SWEEP_INITIAL_SITES", "2")
    monkeypatch.setenv("WEB_TIER_SWEEP_QUOTA_FAST_COUNT", "2")

    s1 = "press.example.org"
    s2 = "news.example.net"
    seeds = [(s1, 1), (s2, 1)]
    monkeypatch.setattr(tier_mod, "load_tiered_allowlist", lambda: DummyTiered(seeds), raising=True)

    mapping = {
        s1: {"tier": 1, "date": "2100-01-01T00:00:00Z"},
        s2: {"tier": 1, "date": "2100-01-01T00:00:00Z"},
    }
    _install_search_stub(monkeypatch, mapping)

    out = run_research("q2", top_k=3, force_refresh=True)
    tier_dbg = (out.get("debug", {}).get("tier") or {})
    escalations = tier_dbg.get("escalation_events") or []
    assert escalations == []
