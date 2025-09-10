import os
import io
import sys
import json
import types
import time
import tempfile
from pathlib import Path

import pytest

from src.reliability.retrieval.pipeline import RetrievalPipeline, DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_TOKENS
from src.tools_runtime.args import normalize_args
from src.reliability_integration.integration import ReliabilityIntegration


FIXTURE = Path(__file__).parent / "data" / "retrieval_corpus.jsonl"


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
            'rag_k': 4,
        }
        self.conversation_history = []
        self._last_context_blocks = []
        self._last_citations_map = {}
        self.flags = {}
        self.trace = []
        self.logger = FakeLogger()

    def _trace(self, s: str):
        self.trace.append(s)


def test_deterministic_fingerprint_and_mtime_change(tmp_path):
    # Copy fixture to temp so we can touch mtime
    dst = tmp_path / "corpus.jsonl"
    dst.write_bytes(FIXTURE.read_bytes())

    rp = RetrievalPipeline()
    docs = rp.run("rate limit 429", k=4, docs_glob=str(dst))
    meta1 = rp.get_index_meta()
    assert 'fingerprint' in meta1 and meta1['num_chunks'] >= 1

    # Touch mtime without changing content; fingerprint should remain identical after rebuild
    old_fp = meta1['fingerprint']
    os.utime(dst, None)
    # Force rebuild by changing chunk params
    docs = rp.run("rate limit 429", k=4, docs_glob=str(dst), chunk_tokens=DEFAULT_CHUNK_TOKENS, overlap_tokens=DEFAULT_OVERLAP_TOKENS)
    meta2 = rp.get_index_meta()
    assert meta2['fingerprint'] == old_fp  # content unchanged → same fingerprint

    # Now change content; fingerprint must change
    with dst.open('a', encoding='utf-8') as f:
        f.write("\n{" + '"id":"extra","title":"Delta","url":"u","timestamp":"2024-05-01","text":"extra"' + "}\n")
    docs = rp.run("rate limit 429", k=4, docs_glob=str(dst))
    meta3 = rp.get_index_meta()
    assert meta3['fingerprint'] != old_fp


def test_dedupe_bucket_by_region(tmp_path):
    dst = tmp_path / "corpus.jsonl"
    dst.write_bytes(FIXTURE.read_bytes())
    rp = RetrievalPipeline()
    res = rp.run("rate limit 429", k=4, docs_glob=str(dst), chunk_tokens=50, overlap_tokens=10)
    # Expect only one bucket for docA despite near-duplicate text
    buckets = set()
    for r in res:
        if r.get('doc_id') == 'docA':
            ch = str(r.get('chunk_id', ''))
            try:
                off = int(ch.split('#', 1)[1])
            except Exception:
                off = 0
            buckets.add(off // 400)
    assert len(buckets) <= 1


def test_threshold_fallback_web(monkeypatch):
    # Prepare fake web_research module
    mod = types.ModuleType('src.plugins.web_research')
    def fake_web_research(query, top_k=4, **kwargs):
        return json.dumps({
            'citations': [
                {'title': 'Web A', 'url': 'https://a', 'lines': [{'quote': 'quote A'}]},
                {'title': 'Web B', 'url': 'https://b', 'lines': [{'quote': 'quote B'}]},
            ]
        })
    mod.web_research = fake_web_research
    sys.modules['src.plugins.web_research'] = mod

    ctx = FakeCtx()
    ctx.reliability.update({
        'rag_min_score': 1e9,  # force fallback
        'ground_fallback': 'web',
    })
    ri = ReliabilityIntegration()
    ri.prepare_context(ctx, "rate limit 429")
    # Web context injected
    assert ctx._last_context_blocks, "fallback web should inject context blocks"
    # Strict citation prompt only when context exists (we appended system text)
    assert any(isinstance(m, dict) and m.get('role') == 'system' for m in ctx.conversation_history)
    # Trace keys present
    joined = "\n".join(ctx.trace)
    assert 'retrieval.topk=' in joined and 'web.latency_ms=' in joined and 'citations.count=' in joined


def test_degrade_flag_when_no_context(monkeypatch):
    # Make retrieval return nothing and disable web fallback
    ctx = FakeCtx()
    ctx.reliability.update({'ground_fallback': 'off'})
    ri = ReliabilityIntegration()
    # Point to an empty corpus path
    ctx.reliability['eval_corpus'] = str(Path(__file__).parent / 'data' / 'nonexistent.jsonl')
    ri.prepare_context(ctx, "unanswerable")
    assert ctx.flags.get('ground_degraded') is True


def test_optional_numeric_zero_preserved():
    schema = {
        'type': 'object',
        'properties': {
            'count': {'type': 'number'},
            'name': {'type': 'string'},
        },
        'required': [],
        'additionalProperties': False,
    }
    raw = json.dumps({'count': 0, 'name': ''})
    out = normalize_args(schema, raw)
    # count==0 must be preserved; empty string is allowed per schema
    assert 'count' in out and out['count'] == 0
