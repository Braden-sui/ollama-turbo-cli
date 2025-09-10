import json
import types
import pytest

from src.client import OllamaTurboClient
from tests.orchestration.test_standard_streaming_parity import DummyClient


@pytest.fixture(autouse=True)
def _no_mem0(monkeypatch):
    # Keep messages minimal for parity and avoid extra system noise
    monkeypatch.setenv('MEM0_ENABLED', '0')


def _install_fake_web():
    mod = types.ModuleType('src.plugins.web_research')
    def fake_web_research(query, top_k=5, **kwargs):
        # Provide quotes that overlap with the query terms to ensure BM25 > 0
        return json.dumps({
            'citations': [
                {'title': 'A', 'url': 'https://a', 'lines': [{'quote': 'rate limit 429 retry-after'}]},
                {'title': 'B', 'url': 'https://b', 'lines': [{'quote': 'Too Many Requests 429'}]},
            ]
        })
    mod.web_research = fake_web_research
    return mod


def _install_empty_web():
    mod = types.ModuleType('src.plugins.web_research')
    def fake_web_research(query, top_k=5, **kwargs):
        return json.dumps({'citations': []})
    mod.web_research = fake_web_research
    return mod


def _make_client(enable_tools: bool) -> OllamaTurboClient:
    c = OllamaTurboClient(api_key='test', enable_tools=enable_tools, quiet=True)
    c.client = DummyClient()
    c.show_trace = True
    return c


def test_streaming_vs_standard_research_fallback_parity(monkeypatch):
    # Force fallback to web and ensure injected context on both paths
    monkeypatch.setenv('RAG_TOPK', '5')  # default k = 5
    monkeypatch.setenv('RAG_MIN_SCORE', '9999999')  # force fallback path

    mod = _install_fake_web()
    import sys
    sys.modules['src.plugins.web_research'] = mod

    # Standard
    c_std = _make_client(enable_tools=False)
    out_std = c_std.chat('rate limit 429', stream=False)
    # Streaming
    c_str = _make_client(enable_tools=False)
    out_str = c_str.chat('rate limit 429', stream=True)

    # Parity assertions
    assert isinstance(out_std, str) and isinstance(out_str, str)
    assert len(c_std._last_context_blocks) >= 1
    assert len(c_str._last_context_blocks) >= 1
    # No degrade flag when fallback provided context
    assert not getattr(c_std, 'flags', {}).get('ground_degraded')
    assert not getattr(c_str, 'flags', {}).get('ground_degraded')

    # Trace keys emitted on both paths
    for c in (c_std, c_str):
        joined = "\n".join(getattr(c, 'trace', []))
        for key in [
            'retrieval.topk=',
            'retrieval.avg_score=',
            'retrieval.hit_rate=',
            'retrieval.fallback_used=1',
            'retrieval.latency_ms=',
            'web.latency_ms=',
            'citations.count=',
        ]:
            assert key in joined


def test_streaming_vs_standard_degrade_flag_parity(monkeypatch):
    # Empty web path -> no context => degrade flag must be set on both paths
    monkeypatch.setenv('RAG_TOPK', '5')
    monkeypatch.delenv('RAG_MIN_SCORE', raising=False)

    mod = _install_empty_web()
    import sys
    sys.modules['src.plugins.web_research'] = mod

    # Standard
    c_std = _make_client(enable_tools=False)
    out_std = c_std.chat('unanswerable topic', stream=False)
    # Streaming
    c_str = _make_client(enable_tools=False)
    out_str = c_str.chat('unanswerable topic', stream=True)

    assert isinstance(out_std, str) and isinstance(out_str, str)
    assert getattr(c_std, 'flags', {}).get('ground_degraded') is True
    assert getattr(c_str, 'flags', {}).get('ground_degraded') is True
    assert not c_std._last_context_blocks
    assert not c_str._last_context_blocks
