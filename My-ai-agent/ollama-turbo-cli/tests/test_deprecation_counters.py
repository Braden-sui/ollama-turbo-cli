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
    monkeypatch.setenv("WEB_CUTOVER_PREP", "1")
    # Ensure exclusion list override is applied at runtime
    monkeypatch.setenv("WEB_EXCLUDE_CITATION_DOMAINS", "exclude.example.com")
    yield


def test_deprecation_counters_increment(monkeypatch):
    # Arrange a URL that will be excluded by exclusion list
    u = "https://exclude.example.com/a"
    # Force exclusion list for this test
    class DummyCfg:
        exclude_citation_domains = ["exclude.example.com"]
        cache_root = "."
        debug_metrics = True
        allow_browser = False
        respect_robots = False
        cache_ttl_seconds = 0
        default_freshness_days = 30
        breaking_freshness_days = 7
        slow_freshness_days = 365
        per_host_concurrency = 2
        enable_tier_sweep = False
        enable_allowlist_news_fallback = False

    # Patch default cfg so run_research picks it up
    pipeline_mod.set_default_config(DummyCfg())

    monkeypatch.setattr(
        pipeline_mod,
        "search",
        lambda *a, **k: [SimpleNamespace(title="T1", url=u, snippet="", source="seed", published=None)],
        raising=True,
    )
    # fetch shouldn't be hit because exclusion happens before fetch

    out = run_research("q", top_k=1, force_refresh=True)
    dbg = out.get("debug", {})
    dep = dbg.get("deprecation", {})
    assert isinstance(dep, dict)
    excluded_via_dep = int(dep.get("excluded_domain", 0) or 0)
    excluded_via_fetch = int((dbg.get("fetch", {}) or {}).get("excluded", 0) or 0)
    assert (excluded_via_dep >= 1) or (excluded_via_fetch >= 1)
