from __future__ import annotations

import os
from typing import List, Dict, Any, Optional


class ExaClient:
    """Thin wrapper for an external search API (e.g., Exa).

    This is a no-op skeleton. When EXA_API_KEY is not set or the integration is
    disabled, methods return empty results to preserve backward compatibility.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EXA_API_KEY")

    def search(self, query: str, k: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """Perform a search and return a list of documents.

        Returns a list of dicts with minimal fields: {"id","title","url","snippet"}.
        This skeleton returns an empty list by default.
        """
        # TODO: Implement actual API integration with rate limits and retries.
        _ = (query, k, kwargs)
        return []
