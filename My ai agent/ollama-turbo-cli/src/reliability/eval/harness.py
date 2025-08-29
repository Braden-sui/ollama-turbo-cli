from __future__ import annotations

from typing import Iterable, Dict, Any, List, Optional, Callable


class EvalHarness:
    """Micro-eval skeleton: runs baseline vs. reliability mode and reports metrics.

    This is a no-op stub; metrics are computed on provided candidates if any.
    """

    def __init__(self) -> None:
        pass

    def run(self,
            corpus: Iterable[Dict[str, Any]],
            generate_baseline: Callable[[Dict[str, Any]], str],
            generate_reliable: Optional[Callable[[Dict[str, Any]], str]] = None) -> Dict[str, Any]:
        """Run evaluation over a JSONL-like corpus iterator.

        Returns a report dict with minimal fields.
        """
        n = 0
        results: List[Dict[str, Any]] = []
        for item in corpus:
            n += 1
            prompt = item.get("prompt") or item.get("input") or ""
            ref = item.get("reference")
            base = generate_baseline({"prompt": prompt})
            rel = generate_reliable({"prompt": prompt}) if generate_reliable else None
            results.append({"prompt": prompt, "baseline": base, "reliable": rel, "reference": ref})
        return {
            "count": n,
            "results": results,
            "metrics": {
                "citation_rate": 0.0,
                "unsupported_claims": 0,
                "agree_rate": 0.0,
                "latency_ms": None,
                "exact_match": None,
            },
        }
