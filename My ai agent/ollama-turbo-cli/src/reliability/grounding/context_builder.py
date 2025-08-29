from __future__ import annotations

from typing import List, Dict, Any


class ContextBuilder:
    """Builds grounding context blocks and citation scaffolding.

    No-op skeleton: returns empty context unless wired to a retrieval pipeline.
    """

    def build(self, messages: List[Dict[str, Any]], retrieved_docs: List[Dict[str, Any]], max_tokens: int = 1200) -> Dict[str, Any]:
        """Return a dict with fields for downstream prompting and validation.

        Expected keys:
        - system_prompt_addition: str (optional system text to enforce citations)
        - context_blocks: List[Dict] (chunked, token-compressed docs)
        - citations_map: Dict[source_id -> url/title]
        """
        _ = (messages, retrieved_docs, max_tokens)
        return {
            "system_prompt_addition": "",
            "context_blocks": [],
            "citations_map": {},
        }
