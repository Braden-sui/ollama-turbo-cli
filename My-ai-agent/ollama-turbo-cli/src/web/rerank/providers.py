from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List

from ..fetch import _httpx_client
from ..config import WebConfig, RerankProviderSpec


@dataclass
class RerankRequest:
    query: str
    documents: Sequence[str]
    top_n: int


@dataclass
class RerankResult:
    index: int
    score: float


class RerankBackend:
    def rerank(self, cfg: WebConfig, spec: RerankProviderSpec, req: RerankRequest) -> List[RerankResult]:
        raise NotImplementedError


class CohereREST(RerankBackend):
    def rerank(self, cfg: WebConfig, spec: RerankProviderSpec, req: RerankRequest) -> List[RerankResult]:
        if not cfg.cohere_key:
            # No key -> fail so router can try next
            raise RuntimeError("Cohere key missing")
        url = (spec.base_url or "https://api.cohere.com") + "/v1/rerank"
        payload = {
            "model": spec.model or "rerank-english-v3.0",
            "query": req.query,
            "documents": [{"text": d} for d in req.documents],
            "top_n": min(req.top_n, spec.top_n),
        }
        headers = {
            "Authorization": f"Bearer {cfg.cohere_key}",
            "Content-Type": "application/json",
            "User-Agent": cfg.user_agent,
        }
        with _httpx_client(cfg) as client:
            r = client.post(url, json=payload, headers=headers, timeout=cfg.rerank_timeout_ms / 1000.0)
            r.raise_for_status()
            data = r.json()
        items = data.get("results", []) or []
        out: List[RerankResult] = []
        for it in items:
            try:
                out.append(RerankResult(index=int(it.get("index")), score=float(it.get("relevance_score", 0.0))))
            except Exception:
                continue
        return out


class VoyageREST(RerankBackend):
    def rerank(self, cfg: WebConfig, spec: RerankProviderSpec, req: RerankRequest) -> List[RerankResult]:
        if not cfg.voyage_key:
            raise RuntimeError("Voyage key missing")
        url = (spec.base_url or "https://api.voyageai.com") + "/v1/rerank"
        payload = {
            "model": spec.model or "rerank-2",
            "query": req.query,
            "documents": list(req.documents),
            "top_k": min(req.top_n, spec.top_n),
        }
        headers = {
            "Authorization": f"Bearer {cfg.voyage_key}",
            "Content-Type": "application/json",
            "User-Agent": cfg.user_agent,
        }
        with _httpx_client(cfg) as client:
            r = client.post(url, json=payload, headers=headers, timeout=cfg.rerank_timeout_ms / 1000.0)
            r.raise_for_status()
            data = r.json()
        out: List[RerankResult] = []
        for it in data.get("data", []) or []:
            try:
                idx = it.get("index")
                score = float(it.get("relevance_score", it.get("score", 0.0)))
                if idx is not None:
                    out.append(RerankResult(index=int(idx), score=score))
            except Exception:
                continue
        return out
