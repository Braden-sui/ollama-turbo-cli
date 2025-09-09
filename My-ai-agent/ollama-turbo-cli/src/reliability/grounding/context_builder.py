from __future__ import annotations

from typing import List, Dict, Any


class ContextBuilder:
    """Builds grounding context blocks and citation scaffolding.

    No-op skeleton: returns empty context unless wired to a retrieval pipeline.
    """

    def build(self, messages: List[Dict[str, Any]], retrieved_docs: List[Dict[str, Any]], max_tokens: int = 1200) -> Dict[str, Any]:
        """Return a dict with fields for downstream prompting and validation.

        Keys:
        - system_prompt_addition: str (optional system text to enforce citations)
        - context_blocks: List[Dict] numbered and size-capped
        - citations_map: Dict[str, Any] mapping index -> {url,title}
        """
        _ = messages
        # Very rough token->char estimate to cap bloat
        char_budget = max(400, int(max_tokens) * 4)
        used = 0
        context_blocks: List[Dict[str, Any]] = []
        citations_map: Dict[str, Any] = {}
        for i, d in enumerate(retrieved_docs or [], 1):
            try:
                text = str(d.get('text') or d.get('content') or '')
                if not text:
                    continue
                title = str(d.get('title') or '')
                url = str(d.get('url') or '')
                source = str(d.get('source') or 'private')
                # Reserve ~10% for overhead
                remaining = max(0, int(char_budget - used - 200))
                if remaining <= 0:
                    break
                snippet = text[: remaining]
                block = {
                    'id': str(i),
                    'title': title or (url or f'doc {i}'),
                    'url': url,
                    'source': source,
                    'text': snippet,
                }
                context_blocks.append(block)
                citations_map[str(i)] = {'url': url, 'title': (title or url)}
                used += len(snippet)
                if used >= char_budget:
                    break
            except Exception:
                continue
        return {
            'system_prompt_addition': '',
            'context_blocks': context_blocks,
            'citations_map': citations_map,
        }
