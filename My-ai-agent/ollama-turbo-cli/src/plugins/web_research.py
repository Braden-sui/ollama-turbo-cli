from __future__ import annotations
import json
from typing import Optional
try:
    # Import lazily-safe: on plugin loader eager import failures, keep a placeholder
    from ..web.pipeline import run_research as _pipeline_run_research
except Exception:
    _pipeline_run_research = None  # type: ignore

# Expose a module-level symbol that tests can monkeypatch regardless of import timing
run_research = _pipeline_run_research  # type: ignore

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
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 8},
                "site_include": {"type": "string", "description": "Optional site/domain to include (e.g., 'site:arxiv.org')."},
                "site_exclude": {"type": "string", "description": "Optional domain snippet to exclude."},
                "freshness_days": {"type": "integer", "minimum": 1, "maximum": 3650},
                "force_refresh": {"type": "boolean", "default": False}
            },
            "required": ["query"],
        },
    },
}


def web_research(query: str, top_k: int = 8, site_include: Optional[str] = None, site_exclude: Optional[str] = None, freshness_days: Optional[int] = None, force_refresh: bool = False) -> str:
    # Relax strict date requirements by not defaulting freshness_days from query_profile.
    # Only default top_k when not provided. If callers want recency gating, they pass freshness_days explicitly.
    if not top_k:
        # Keep a simple, robust default locally; planners/callers may override
        top_k = 8
    # Use the module-level symbol so tests can monkeypatch it
    if run_research is None:  # type: ignore
        raise RuntimeError("run_research pipeline not available")
    res = run_research(
        query,
        top_k=int(top_k or 8),
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
