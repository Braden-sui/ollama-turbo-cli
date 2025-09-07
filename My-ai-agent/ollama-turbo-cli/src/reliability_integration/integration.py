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
            rp = RetrievalPipeline()
            try:
                topk = int(ctx.reliability.get('rag_k') or int(os.getenv('RAG_TOPK', '5') or '5'))
            except Exception:
                topk = 5
            docs = rp.run(user_message, k=topk)
            cb = ContextBuilder()
            try:
                max_tokens = int(os.getenv('RAG_MAX_TOKENS', '1200') or '1200')
            except Exception:
                max_tokens = 1200
            built = cb.build(ctx.conversation_history, docs, max_tokens=max_tokens)
            ctx._last_context_blocks = built.get('context_blocks') or []
            ctx._last_citations_map = built.get('citations_map') or {}
            system_add = built.get('system_prompt_addition') or ""
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
            ctx._trace(f"reliability:context blocks={len(ctx._last_context_blocks)}")
        except Exception as e:
            try:
                ctx.logger.debug(f"reliability context error: {e}")
            except Exception:
                pass
