import os
import json
import types
from pathlib import Path
import pytest

from src.client import OllamaTurboClient
from tests.orchestration.test_standard_streaming_parity import DummyClient


@pytest.mark.skipif(os.getenv('GENERATE_METRICS_SNAPSHOT', '0') in {'0', 'false', 'no', 'off'}, reason='Set GENERATE_METRICS_SNAPSHOT=1 to generate snapshot')
def test_generate_metrics_snapshot(tmp_path, monkeypatch):
    # Deterministic env
    monkeypatch.setenv('MEM0_ENABLED', '0')
    monkeypatch.setenv('RAG_TOPK', '5')
    monkeypatch.setenv('RAG_MIN_SCORE', '9999999')  # force web fallback

    # Fake web_research with overlapping quotes
    mod = types.ModuleType('src.plugins.web_research')
    def fake_web_research(query, top_k=5, **kwargs):
        return json.dumps({
            'citations': [
                {'title': 'A', 'url': 'https://a', 'lines': [{'quote': 'rate limit 429 retry-after'}]},
                {'title': 'B', 'url': 'https://b', 'lines': [{'quote': 'Too Many Requests 429'}]},
            ]
        })
    mod.web_research = fake_web_research
    import sys
    sys.modules['src.plugins.web_research'] = mod

    # Standard and streaming runs
    def _make():
        c = OllamaTurboClient(api_key='test', enable_tools=False, quiet=True)
        c.client = DummyClient()
        c.show_trace = True
        return c

    c_std = _make()
    _ = c_std.chat('rate limit 429', stream=False)

    c_str = _make()
    _ = c_str.chat('rate limit 429', stream=True)

    # Collect metrics lines
    def _collect(c):
        keys = (
            'retrieval.topk=',
            'retrieval.avg_score=',
            'retrieval.hit_rate=',
            'retrieval.fallback_used',
            'retrieval.latency_ms=',
            'web.latency_ms=',
            'citations.count=',
            'citations.coverage_pct=',
        )
        lines = []
        for ev in getattr(c, 'trace', []):
            if any(k in ev for k in keys):
                lines.append(ev)
        return sorted(set(lines))

    std_lines = _collect(c_std)
    str_lines = _collect(c_str)

    # Write snapshot file
    out_dir = Path('tests') / 'artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'metrics_snapshot.txt'
    with out_path.open('w', encoding='utf-8') as f:
        f.write('# Standard\n')
        for ln in std_lines:
            f.write(ln + '\n')
        f.write('\n# Streaming\n')
        for ln in str_lines:
            f.write(ln + '\n')

    # Assert written
    assert out_path.is_file()
