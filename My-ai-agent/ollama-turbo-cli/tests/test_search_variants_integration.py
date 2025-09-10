from types import SimpleNamespace

def test_search_uses_generate_variants_when_empty(monkeypatch):
    # Target module
    import src.web.search as search_mod
    from src.web.config import WebConfig

    # Force providers to return empty for raw query, non-empty for 'ALT'
    calls = {"queries": []}

    def _mk_result(title: str, url: str):
        return search_mod.SearchResult(title=title, url=url, snippet="", source="fake", published=None)

    def fake_provider(q, cfg):
        # record query
        calls["queries"].append(q.query)
        if q.query == "ALT":
            return [_mk_result("Alt Hit", "https://example.com/alt")]
        return []

    # Patch all providers used by rotation to our fake
    monkeypatch.setattr(search_mod, "_search_brave", fake_provider, raising=True)
    monkeypatch.setattr(search_mod, "_search_tavily", fake_provider, raising=True)
    monkeypatch.setattr(search_mod, "_search_exa", fake_provider, raising=True)
    monkeypatch.setattr(search_mod, "_search_google_pse", fake_provider, raising=True)
    monkeypatch.setattr(search_mod, "_search_duckduckgo_fallback", fake_provider, raising=True)

    # Patch variant generator to a deterministic sequence
    monkeypatch.setattr(search_mod, "generate_variants", lambda q, mode, max_tokens, stopword_profile: [q, "ALT"], raising=True)

    cfg = WebConfig()
    cfg.query_compression_mode = "soft"
    cfg.query_max_tokens_fallback = 12
    cfg.stopword_profile = "minimal"
    cfg.variant_parallel = False
    cfg.variant_max = 2

    out = search_mod.search("RAW", cfg=cfg)
    assert out and isinstance(out, list)
    assert any(r.url == "https://example.com/alt" for r in out)
    # Ensure raw was tried first (providers saw RAW before ALT)
    assert calls["queries"][0] == "RAW"
    assert "ALT" in calls["queries"]


def test_search_variants_off_mode_yields_empty(monkeypatch):
    import src.web.search as search_mod
    from src.web.config import WebConfig

    # Providers always empty
    monkeypatch.setattr(search_mod, "_search_brave", lambda q, cfg: [], raising=True)
    monkeypatch.setattr(search_mod, "_search_tavily", lambda q, cfg: [], raising=True)
    monkeypatch.setattr(search_mod, "_search_exa", lambda q, cfg: [], raising=True)
    monkeypatch.setattr(search_mod, "_search_google_pse", lambda q, cfg: [], raising=True)
    monkeypatch.setattr(search_mod, "_search_duckduckgo_fallback", lambda q, cfg: [], raising=True)
    # Even if generator returns extra, mode=off should cause no additional variants to be used
    monkeypatch.setattr(search_mod, "generate_variants", lambda q, mode, max_tokens, stopword_profile: [q, "ALT"], raising=True)

    cfg = WebConfig()
    cfg.query_compression_mode = "off"
    cfg.variant_max = 2
    out = search_mod.search("RAW", cfg=cfg)
    assert out == []

