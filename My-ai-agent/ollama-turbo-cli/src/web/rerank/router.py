from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List
import time
import hashlib
import json

from ..config import WebConfig, RerankProviderSpec
from .providers import RerankRequest, RerankResult, CohereREST, VoyageREST

# Simple debug counter hook; no-op unless cfg.debug_metrics is used by caller

def _debug_increment(metrics: dict, key: str, value: int = 1) -> None:
    try:
        metrics[key] = int(metrics.get(key, 0)) + int(value)
    except Exception:
        pass

_backends_rest = {
    "cohere": CohereREST(),
    "voyage": VoyageREST(),
}

_breakers: dict[str, dict[str, float]] = {}
_cache: dict[str, tuple[float, List[RerankResult]]] = {}


def _cache_key(query: str, docs: Sequence[str], top_n: int) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8", "ignore"))
    for d in docs:
        h.update(b"\x00")
        h.update((d or "").encode("utf-8", "ignore"))
    h.update(f"|{top_n}".encode("ascii"))
    return h.hexdigest()


def _is_open(provider: str, now: float, cfg: WebConfig) -> bool:
    st = _breakers.get(provider)
    return bool(st and now < st.get("opened_until", 0))


def _trip(provider: str, now: float, cfg: WebConfig) -> None:
    st = _breakers.setdefault(provider, {"fail_count": 0, "opened_until": 0})
    st["fail_count"] += 1
    if st["fail_count"] >= cfg.rerank_breaker_threshold:
        st["opened_until"] = now + cfg.rerank_breaker_cooldown_ms / 1000.0


def _reset(provider: str) -> None:
    _breakers[provider] = {"fail_count": 0, "opened_until": 0}


def _sdk_backend(name: str):
    if name == "cohere":
        from .sdk_backends import CohereSDK
        return CohereSDK()
    if name == "voyage":
        from .sdk_backends import VoyageSDK
        return VoyageSDK()
    raise KeyError(name)


def rerank(cfg: WebConfig, query: str, documents: Sequence[str], top_n: int = 10) -> List[int]:
    if not cfg.rerank_enabled or not documents:
        return list(range(min(top_n, len(documents))))

    ck = _cache_key(query, documents, top_n)
    now = time.time()
    cached = _cache.get(ck)
    if cached and now - cached[0] <= cfg.rerank_cache_ttl_s:
        return [r.index for r in cached[1][:top_n]]

    specs: List[RerankProviderSpec] = sorted(cfg.rerank_providers, key=lambda s: -float(s.weight))
    last_err: Exception | None = None
    results: List[RerankResult] | None = None
    had_error = False

    for spec in specs:
        if _is_open(spec.name, now, cfg):
            continue
        try:
            backend = _backends_rest[spec.name] if cfg.rerank_mode == "rest" else _sdk_backend(spec.name)
            res = backend.rerank(cfg, spec, RerankRequest(query=query, documents=documents, top_n=top_n))
            res.sort(key=lambda r: r.score, reverse=True)
            results = res
            if not had_error:
                _cache[ck] = (now, results)
            _reset(spec.name)
            break
        except Exception as e:
            last_err = e
            _trip(spec.name, now, cfg)
            had_error = True
            continue

    n = min(top_n, len(documents))
    if not results:
        # identity fallback on total failure
        return list(range(n))

    # Clamp, dedupe, and pad with identity. If any invalid index was seen
    # before a first valid pick, accept that first valid pick then stop scanning
    # and pad with identity order.
    safe: List[int] = []
    seen: set[int] = set()
    invalid_seen = False
    for r in results:
        try:
            idx = int(r.index)
        except Exception:
            invalid_seen = True
            continue
        if 0 <= idx < len(documents) and idx not in seen:
            safe.append(idx)
            seen.add(idx)
            if len(safe) == n:
                break
            if invalid_seen:
                break
    if len(safe) < n:
        for i in range(len(documents)):
            if i not in seen:
                safe.append(i)
            if len(safe) == n:
                break
    return safe
