from __future__ import annotations
from typing import List
from .providers import RerankBackend, RerankRequest, RerankResult
from ..config import WebConfig, RerankProviderSpec


class CohereSDK(RerankBackend):
    def __init__(self) -> None:
        import cohere  # type: ignore
        from ..config import WebConfig as _W  # avoid unused import lint
        # The key is read at call time from cfg to allow hot-reload/env override per request
        self._cohere = cohere

    def rerank(self, cfg: WebConfig, spec: RerankProviderSpec, req: RerankRequest) -> List[RerankResult]:
        if not cfg.cohere_key:
            raise RuntimeError("Cohere key missing")
        client = self._cohere.Client(api_key=cfg.cohere_key)
        r = client.rerank(
            query=req.query,
            documents=list(req.documents),
            top_n=min(req.top_n, spec.top_n),
            model=spec.model or "rerank-english-v3.0",
        )
        return [RerankResult(index=int(it.index), score=float(getattr(it, "relevance_score", 0.0))) for it in getattr(r, "results", [])]


class VoyageSDK(RerankBackend):
    def __init__(self) -> None:
        import voyageai  # type: ignore
        self._voyageai = voyageai

    def rerank(self, cfg: WebConfig, spec: RerankProviderSpec, req: RerankRequest) -> List[RerankResult]:
        if not cfg.voyage_key:
            raise RuntimeError("Voyage key missing")
        client = self._voyageai.Client(api_key=cfg.voyage_key)
        r = client.rerank(
            query=req.query,
            documents=list(req.documents),
            top_k=min(req.top_n, spec.top_n),
            model=spec.model or "rerank-2",
        )
        out: List[RerankResult] = []
        for it in getattr(r, "data", []) or []:
            try:
                idx = int(getattr(it, "index", -1))
                score = float(getattr(it, "relevance_score", getattr(it, "score", 0.0)))
                if idx >= 0:
                    out.append(RerankResult(index=idx, score=score))
            except Exception:
                continue
        return out
