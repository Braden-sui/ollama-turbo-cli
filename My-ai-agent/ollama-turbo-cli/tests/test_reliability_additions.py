from types import SimpleNamespace

from src.prompt_manager import PromptManager
from src.web.pipeline import run_research, set_default_config
from src.web.config import WebConfig
import src.web.pipeline as pipeline_mod


def _mk_search_result(title, url, snippet="", source="ddg", published=None):
    return SimpleNamespace(title=title, url=url, snippet=snippet, source=source, published=published)


def test_post_tool_reprompt_is_strict_cited():
    pm = PromptManager("high")
    t = pm.reprompt_after_tools()
    assert "Synthesize an answer only from context.docs" in t
    assert "Use inline [n]" in t


def test_web_research_policy_forced_refresh_flag(monkeypatch, tmp_path):
    # Ensure cache isolation
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    target = "https://primary.example.com/article"
    monkeypatch.setattr(pipeline_mod, "search", lambda *a, **k: [_mk_search_result("Ex", target)], raising=True)

    import src.web.extract as extract_mod
    import src.web.rerank as rerank_mod

    # Minimal no-op fetch
    monkeypatch.setattr(
        pipeline_mod,
        "fetch_url",
        lambda url, **k: SimpleNamespace(ok=True, status=200, url=url, final_url=url, headers={}, content_type="text/html", body_path=None, meta_path=None, cached=False, browser_used=False, reason=None),
        raising=True,
    )
    # Extract returns text
    monkeypatch.setattr(
        extract_mod,
        "extract_content",
        lambda meta, **k: SimpleNamespace(ok=True, kind="html", markdown="a\n", title="T", date=None, meta={}, used={}, risk="LOW", risk_reasons=[]),
        raising=True,
    )
    # Rerank emits highlights
    monkeypatch.setattr(rerank_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        rerank_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.9, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "a"}]}],
        raising=True,
    )

    out = run_research("force", top_k=1, force_refresh=True)
    assert out.get("policy", {}).get("forced_refresh_used") is True


def test_env_override_applies_with_preexisting_default_cfg(monkeypatch, tmp_path):
    # Step 1: establish a default cfg with only wikipedia excluded
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    monkeypatch.setenv("WEB_EXCLUDE_CITATION_DOMAINS", "wikipedia.org")
    set_default_config(WebConfig())

    # Step 2: later in runtime, extend exclusion to include example.com
    monkeypatch.setenv("WEB_EXCLUDE_CITATION_DOMAINS", "wikipedia.org,example.com")

    # Patch pipeline to target example.com
    target = "https://example.com/article"
    monkeypatch.setattr(pipeline_mod, "search", lambda *a, **k: [_mk_search_result("Ex", target)], raising=True)

    import src.web.extract as extract_mod
    import src.web.rerank as rerank_mod

    # Bypass fetch
    monkeypatch.setattr(
        pipeline_mod,
        "fetch_url",
        lambda url, **k: SimpleNamespace(ok=True, status=200, url=url, final_url=url, headers={}, content_type="text/html", body_path=None, meta_path=None, cached=False, browser_used=False, reason=None),
        raising=True,
    )
    monkeypatch.setattr(
        extract_mod,
        "extract_content",
        lambda meta, **k: SimpleNamespace(ok=True, kind="html", markdown="x\n", title="T", date=None, meta={}, used={}, risk="LOW", risk_reasons=[]),
        raising=True,
    )
    monkeypatch.setattr(rerank_mod, "chunk_text", lambda s: [s], raising=True)
    monkeypatch.setattr(
        rerank_mod,
        "rerank_chunks",
        lambda q, c, **k: [{"id": "1", "score": 0.8, "start_line": 1, "end_line": 1, "highlights": [{"line": 1, "text": "x"}]}],
        raising=True,
    )

    out = run_research("exclude example", top_k=1, force_refresh=True)
    urls = [c.get("canonical_url") for c in out.get("citations", [])]
    assert all("example.com" not in (u or "") for u in urls)


def test_auto_force_refresh_sets_policy_when_no_citations(monkeypatch, tmp_path):
    # Build a cfg that disables emergency bootstrap so search=[] stays empty
    cfg = WebConfig()
    cfg.emergency_bootstrap = False
    cfg.cache_root = str(tmp_path / ".webcache")

    # Force empty search results so the first run yields no citations
    monkeypatch.setattr(pipeline_mod, "search", lambda *a, **k: [], raising=True)

    out = run_research("no docs fallback", top_k=1, cfg=cfg)
    # The pipeline should auto-retry with force_refresh=True and surface that in policy
    assert out.get("policy", {}).get("forced_refresh_used") is True
