import json
import sys
from pathlib import Path
import types

from src.reliability.retrieval.research_ingest import citations_to_docs
from src.reliability.retrieval.pipeline import RetrievalPipeline
from src.reliability_integration.integration import ReliabilityIntegration


class FakeLogger:
    def debug(self, *a, **k):
        pass


class FakeCtx:
    def __init__(self):
        self.reliability = {
            'ground': True,
            'cite': True,
            'check': 'warn',
            'eval_corpus': None,
            'rag_k': 3,
            'rag_min_score': 9999999.0,  # force fallback
            'ground_fallback': 'web',
        }
        self.conversation_history = []
        self._last_context_blocks = []
        self._last_citations_map = {}
        self.flags = {}
        self.trace = []
        self.logger = FakeLogger()

    def _trace(self, s: str):
        self.trace.append(s)


def _fake_cits_for_query(q: str):
    # Build overlapping citations across variants to exercise merge/dedupe
    def cit(url, title, *quotes):
        return {
            'canonical_url': url,
            'title': title,
            'date': None,
            'lines': [{'quote': q} for q in quotes],
        }
    base = [
        cit('https://a', 'A', 'API rate limit is 60 rpm', 'HTTP 429 Too Many Requests'),
        cit('https://b', 'B', 'rate limit handling via Retry-After'),
    ]
    if q.endswith('overview'):
        return {'citations': [cit('https://b', 'B', 'rate limit handling via Retry-After'), cit('https://c', 'C', 'OAuth tokens and scopes')]}
    if q.endswith('latest'):
        return {'citations': [cit('https://d', 'D', 'gardening tip: soil PH')]}
    if q.endswith('explained'):
        return {'citations': [cit('https://e', 'E', '429 status indicates too many requests'), cit('https://a', 'A', 'API rate limit is 60 rpm')]}
    return {'citations': base}


def test_citations_to_docs_merge_dedup():
    cits = []
    for suf in ['', ' overview', ' latest', ' explained']:
        obj = _fake_cits_for_query('q' + suf)
        cits.extend(obj['citations'])
    docs = citations_to_docs(cits, max_docs=100)
    urls = {d['url'] for d in docs}
    # Expect merged unique URLs: https://a, https://b, https://c, https://d, https://e
    assert urls == {'https://a', 'https://b', 'https://c', 'https://d', 'https://e'}
    # Ensure provenance fields are present
    for d in docs:
        assert d.get('id') and d.get('title') and d.get('url') and d.get('timestamp') is not None


def test_research_integration_topk_dedupe_and_provenance(monkeypatch):
    # Monkeypatch web_research tool
    mod = types.ModuleType('src.plugins.web_research')
    def fake_web_research(query, top_k=5, **kwargs):
        return json.dumps(_fake_cits_for_query(query))
    mod.web_research = fake_web_research
    sys.modules['src.plugins.web_research'] = mod

    ctx = FakeCtx()
    ri = ReliabilityIntegration()
    ri.prepare_context(ctx, "rate limit 429")

    # Retrieval reduced to top-k, unique by URL
    blocks = ctx._last_context_blocks
    assert 1 <= len(blocks) <= ctx.reliability['rag_k']
    urls = [b.get('url') for b in blocks]
    assert len(urls) == len(set(urls))
    # Provenance present
    for b in blocks:
        assert b.get('url') and b.get('title')
    # Observability
    joined = "\n".join(ctx.trace)
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


def test_ephemeral_cleanup_after_run():
    # Build a few ephemeral docs and run retrieval with ephemeral=True
    docs = [
        {'id': 'm1', 'title': 'T1', 'url': 'u1', 'timestamp': '2025-01-01', 'text': 'rate limit 429 retry-after'},
        {'id': 'm2', 'title': 'T2', 'url': 'u2', 'timestamp': '2025-01-02', 'text': 'unrelated gardening soil'},
    ]
    rp = RetrievalPipeline()
    out = rp.run('rate limit 429', k=2, docs_in_memory=docs, ephemeral=True)
    assert len(out) >= 1
    # After ephemeral run, index meta should be cleared
    assert rp.get_index_meta() == {}
