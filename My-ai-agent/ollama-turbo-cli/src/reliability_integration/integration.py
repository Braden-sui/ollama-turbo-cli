from __future__ import annotations

"""
Reliability integration facade (Phase F).

Moves the client helpers `_prepare_reliability_context` and `_load_system_cited`
behind a small class without changing behavior.
"""

from typing import Any, Dict
import os

# Import existing reliability pipeline/builders
from ..reliability.retrieval.pipeline import RetrievalPipeline
from ..reliability.grounding.context_builder import ContextBuilder


class ReliabilityIntegration:
    def __init__(self) -> None:
        pass

    # Mirrors client._load_system_cited behavior
    def load_system_cited(self, ctx) -> str:
        try:
            if getattr(ctx, '_system_cited_cache', None):
                return ctx._system_cited_cache or ""
            # Base relative to src/
            base_src = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(base_src, 'reliability', 'prompts', 'system_cited.md')
            text = ""
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception:
                text = "When citing sources, include inline citations like [1], [2] mapped to provided context."
            ctx._system_cited_cache = text
            return text
        except Exception:
            return ""

    # Mirrors client._prepare_reliability_context behavior
    def prepare_context(self, ctx, user_message: str) -> None:
        try:
            import time as _time
            t0 = _time.perf_counter()
            rp = RetrievalPipeline()
            try:
                topk = int(ctx.reliability.get('rag_k') or int(os.getenv('RAG_TOPK', '5') or '5'))
            except Exception:
                topk = 5
            # Retrieval knobs
            docs_glob = None
            try:
                docs_glob = ctx.reliability.get('docs_glob')
            except Exception:
                docs_glob = None
            eval_corpus = None
            try:
                eval_corpus = ctx.reliability.get('eval_corpus')
            except Exception:
                eval_corpus = None
            try:
                min_score_env = os.getenv('RAG_MIN_SCORE')
                min_score = ctx.reliability.get('rag_min_score') if ctx.reliability.get('rag_min_score') is not None else (float(min_score_env) if min_score_env else None)
            except Exception:
                min_score = None

            docs = rp.run(user_message, k=topk, docs_glob=docs_glob, eval_corpus=eval_corpus, min_score=min_score)
            # Retrieval metrics
            try:
                avg_score = 0.0
                if docs:
                    avg_score = sum(float((d or {}).get('score') or 0.0) for d in docs) / max(1, len(docs))
                hit_rate = 0.0
                if docs:
                    if min_score is not None:
                        hits = sum(1 for d in docs if float((d or {}).get('score') or 0.0) >= float(min_score))
                        hit_rate = hits / max(1, topk)
                    else:
                        hit_rate = min(1.0, len(docs) / max(1, topk))
                latency_ms = int((_time.perf_counter() - t0) * 1000)
                ctx._trace(f"retrieval.topk={len(docs)}")
                ctx._trace(f"retrieval.avg_score={avg_score:.4f}")
                ctx._trace(f"retrieval.hit_rate={hit_rate:.4f}")
                ctx._trace(f"retrieval.latency_ms={latency_ms}")
            except Exception:
                pass
            cb = ContextBuilder()
            try:
                max_tokens = int(os.getenv('RAG_MAX_TOKENS', '1200') or '1200')
            except Exception:
                max_tokens = 1200
            # Decide if local retrieval is sufficient; if not and fallback configured, try web_research
            fallback_used = False
            try:
                need_fallback = (not docs) or (bool(min_score) and (float((docs[0] or {}).get('score') or 0.0) < float(min_score)))
            except Exception:
                need_fallback = (not docs)
            fallback_mode = str(ctx.reliability.get('ground_fallback') or os.getenv('RAG_GROUND_FALLBACK') or 'off').strip().lower()
            if need_fallback and fallback_mode == 'web':
                try:
                    from ..plugins.web_research import web_research  # type: ignore
                except Exception:
                    web_research = None
                if web_research is not None:
                    try:
                        tw0 = _time.perf_counter()
                        raw = web_research(user_message, top_k=int(topk or 4))
                        import json as _json
                        obj = _json.loads(raw) if isinstance(raw, str) else raw
                        cits = obj.get('citations') if isinstance(obj, dict) else []
                        context_blocks = []
                        citations_map = {}
                        if isinstance(cits, list):
                            for i, c in enumerate(cits[:topk], 1):
                                try:
                                    title = str(c.get('title') or '')
                                    url = str(c.get('canonical_url') or c.get('url') or '')
                                    lines = c.get('lines') or []
                                    quotes = []
                                    if isinstance(lines, list):
                                        for hl in lines[:3]:
                                            q = (hl or {}).get('quote') or ''
                                            if q:
                                                quotes.append(q)
                                    block = {
                                        'id': str(i),
                                        'title': title or (url or f'source {i}'),
                                        'url': url,
                                        'source': 'web',
                                        'text': ('\n'.join(quotes) if quotes else ''),
                                    }
                                    context_blocks.append(block)
                                    citations_map[str(i)] = {'url': url, 'title': (title or url)}
                                except Exception:
                                    continue
                        ctx._last_context_blocks = context_blocks
                        ctx._last_citations_map = citations_map
                        system_add = ''
                        if ctx.reliability.get('cite') and context_blocks:
                            cited = self.load_system_cited(ctx)
                            system_add = cited or ''
                        if system_add:
                            ctx.conversation_history.append({'role': 'system', 'content': system_add})
                        web_latency_ms = int((_time.perf_counter() - tw0) * 1000)
                        ctx._trace(f"reliability:context fallback=web sources={len(context_blocks)}")
                        ctx._trace(f"web.latency_ms={web_latency_ms}")
                        fallback_used = True
                    except Exception as fe:
                        try:
                            ctx._trace(f"reliability:fallback:web:error {fe}")
                        except Exception:
                            pass
            system_add = ''
            if not fallback_used:
                built = cb.build(ctx.conversation_history, docs, max_tokens=max_tokens)
                ctx._last_context_blocks = built.get('context_blocks') or []
                ctx._last_citations_map = built.get('citations_map') or {}
                system_add = built.get('system_prompt_addition') or ""
            try:
                ctx._trace(f"citations.count={len(ctx._last_context_blocks or [])}")
            except Exception:
                pass
            if ctx.reliability.get('cite'):
                # Only add strict citation instructions if we actually have grounded context
                if ctx._last_context_blocks:
                    cited = self.load_system_cited(ctx)
                    if cited:
                        system_add = (system_add + "\n" + cited).strip() if system_add else cited
                else:
                    # Soft rule when no sources are provided
                    soft_rule = "If no sources are provided, avoid specific figures (numbers/dates) or mark uncertainty clearly."
                    system_add = (system_add + "\n" + soft_rule).strip() if system_add else soft_rule
            if system_add:
                ctx.conversation_history.append({'role': 'system', 'content': system_add})
            # Degrade when ground requested but no context available
            try:
                if bool(ctx.reliability.get('ground')) and not ctx._last_context_blocks:
                    flags = getattr(ctx, 'flags', {}) or {}
                    flags['ground_degraded'] = True
                    setattr(ctx, 'flags', flags)
                    ctx._trace("reliability:ground:degraded")
            except Exception:
                pass
            ctx._trace(f"reliability:context blocks={len(ctx._last_context_blocks)}")
        except Exception as e:
            try:
                ctx.logger.debug(f"reliability context error: {e}")
            except Exception:
                pass
