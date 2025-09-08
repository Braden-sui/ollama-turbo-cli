from __future__ import annotations
import json
from typing import Optional
from ..web.pipeline import run_research

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_research",
        "description": "Use for multi-hop web research that requires up-to-date facts and citations. Orchestrates Plan→Search→Fetch (robots.txt + crawl-delay enforced; per-host concurrency bounded)→Extract (HTML/PDF)→Chunk→Rerank→Cite (exact quotes; PDF page mapping)→Cache. Prefer this over raw web_fetch when you need sourcing and synthesis across multiple pages. Parameters: site_include/site_exclude narrow scope; top_k controls breadth; freshness_days limits recency; force_refresh=true bypasses caches. Returns compact JSON with results, citations, and archive URLs.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string", "description": "User question or query to research."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "site_include": {"type": "string", "description": "Optional site/domain to include (e.g., 'site:arxiv.org')."},
                "site_exclude": {"type": "string", "description": "Optional domain snippet to exclude."},
                "freshness_days": {"type": "integer", "minimum": 1, "maximum": 3650},
                "force_refresh": {"type": "boolean", "default": False}
            },
            "required": ["query"],
        },
    },
}


def web_research(query: str, top_k: int = 5, site_include: Optional[str] = None, site_exclude: Optional[str] = None, freshness_days: Optional[int] = None, force_refresh: bool = False) -> str:
    # Backward-compatible call shape (tests patch run_research without cfg)
    res = run_research(
        query,
        top_k=int(top_k or 5),
        site_include=site_include,
        site_exclude=site_exclude,
        freshness_days=freshness_days,
        force_refresh=bool(force_refresh),
    )
    # Return compact JSON for injection-safe integration
    return json.dumps(res, ensure_ascii=False)

TOOL_IMPLEMENTATION = web_research
TOOL_AUTHOR = "platform-core"
TOOL_VERSION = "1.0.0"
