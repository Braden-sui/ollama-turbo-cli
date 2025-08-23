from __future__ import annotations

from typing import List, Dict, Any


class RetrievalPipeline:
    """Retrieval pipeline stub: query → dedupe → chunk → compress → id assign.

    Returns empty list by default to preserve behavior until wired.
    """

    def run(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        _ = (query, k)
        # Expected document schema (when implemented):
        # {"id": str, "title": str, "url": str, "content": str, "tokens": int}
        return []
