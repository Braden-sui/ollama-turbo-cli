import os

from src.web.allowlist_tiered import load_tiered_allowlist


def test_tiered_loader_fallback_and_matching(monkeypatch):
    # Ensure no env override; loader should fall back to packaged resource file
    monkeypatch.delenv("WEB_TIERED_ALLOWLIST_FILE", raising=False)
    monkeypatch.delenv("WEB_ALLOWLIST_TIERED_FILE", raising=False)

    t = load_tiered_allowlist()
    assert t is not None, "Tiered allowlist should load from packaged resource"

    # Known Tier 0 regulator
    assert t.tier_for_host("sec.gov") == 0
    # Known Tier 1 wire
    assert t.tier_for_host("reuters.com") in (1,)
    # Discouraged pattern sample
    assert t.discouraged_host("www.dailymail.co.uk") is True

    # Category mapping sanity
    cat = t.category_for_host("sec.gov")
    assert isinstance(cat, str) and "finance" in cat

    # Staleness policy buckets exist
    p = t.policy.get("staleness_defaults_days", {})
    assert all(k in p for k in [
        "gov_law", "gov_stats", "health", "science", "news_wires", "finance", "tech_docs", "standards", "weather_hazards"
    ])
