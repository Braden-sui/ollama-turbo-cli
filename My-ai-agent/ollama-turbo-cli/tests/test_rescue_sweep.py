from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.web.pipeline as pipeline_mod
import src.web.allowlist_tiered as allowlist_tiered_mod
from src.web.pipeline import run_research


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_ALLOW_BROWSER", "0")
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    monkeypatch.setenv("WEB_DEBUG_METRICS", "1")
    yield


def test_rescue_preview_meta(monkeypatch):
    # Enable rescue preview
    monkeypatch.setenv("WEB_RESCUE_SWEEP", "1")

    # Tiered seeds present via mocked loader; categories seen from initial results
    class MockTiered:
        seeds_by_tier = [("reuters.com", 0), ("apnews.com", 0)]
        seeds_by_cat = [("sec.gov", 0, "finance_regulators")]  # not used here
        def category_for_host(self, h):
            return "finance_regulators" if "reuters" in h else None
        def tier_for_host(self, h):
            return 0 if any(x in h for x in ("reuters","apnews")) else 2
        def discouraged_host(self, h):
            return False

    monkeypatch.setattr(allowlist_tiered_mod, "load_tiered_allowlist", lambda: MockTiered(), raising=True)
    monkeypatch.setattr(pipeline_mod, "search", lambda *a, **k: [SimpleNamespace(title="t", url="https://reuters.com/a", snippet="", source="seed", published=None)], raising=True)

    out = run_research("q", top_k=1, force_refresh=True)
    dbg = out.get("debug", {})
    assert isinstance(dbg, dict)
    res = dbg.get("rescue")
    assert isinstance(res, dict)
    assert res.get("added_count") >= 0
