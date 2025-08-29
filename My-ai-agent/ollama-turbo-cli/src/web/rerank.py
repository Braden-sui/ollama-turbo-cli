from __future__ import annotations
import re
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from .config import WebConfig


@dataclass
class Chunk:
    id: str
    text: str
    start_line: int
    end_line: int


def _estimate_tokens(text: str) -> int:
    # Simple approximation: 1 token ~ 4 chars avg
    return max(1, int(len(text) / 4))


def chunk_text(markdown: str, *, target_tokens: int = 768) -> List[Chunk]:
    lines = markdown.splitlines()
    chunks: List[Chunk] = []
    buf: list[str] = []
    start_line = 1
    cur_tokens = 0
    def flush(end_line: int):
        nonlocal buf, start_line, cur_tokens
        if not buf:
            return
        txt = "\n".join(buf)
        cid = f"{start_line}-{end_line}"
        chunks.append(Chunk(id=cid, text=txt, start_line=start_line, end_line=end_line))
        buf = []
        cur_tokens = 0
        start_line = end_line + 1
    for i, line in enumerate(lines, start=1):
        # Prefer breaks at headings or blank lines
        if line.strip().startswith('#') and buf:
            flush(i-1)
        buf.append(line)
        cur_tokens += _estimate_tokens(line + '\n')
        if cur_tokens >= target_tokens and (line.strip() == '' or line.strip().startswith('#')):
            flush(i)
    flush(len(lines))
    return chunks


def _simple_rerank(query: str, chunks: List[Chunk]) -> List[Tuple[Chunk, float]]:
    q = re.findall(r"\w+", query.lower())
    out: List[Tuple[Chunk, float]] = []
    for ch in chunks:
        t = ch.text.lower()
        score = sum(t.count(w) for w in q)
        out.append((ch, float(score)))
    out.sort(key=lambda x: (-x[1], x[0].start_line))
    return out


def rerank_chunks(query: str, chunks: List[Chunk], *, cfg: Optional[WebConfig] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    cfg = cfg or WebConfig()
    results: List[Tuple[Chunk, float]] = []
    used = None
    # Cohere first if configured
    if cfg.cohere_key:
        try:
            import cohere  # type: ignore
            co = cohere.Client(api_key=cfg.cohere_key)
            docs = [ch.text[:4000] for ch in chunks]
            r = co.rerank(model='rerank-english-v3.0', query=query, documents=docs, top_n=min(top_k, len(docs)))
            idx_to_score = {it.index: float(it.relevance_score) for it in r}
            order = sorted(idx_to_score.items(), key=lambda x: -x[1])
            for idx, score in order:
                ch = chunks[idx]
                results.append((ch, score))
            used = 'cohere'
        except Exception:
            results = []
            used = None
    # Voyage fallback
    if not results and cfg.voyage_key:
        try:
            import voyageai  # type: ignore
            vo = voyageai.Client(api_key=cfg.voyage_key)
            docs = [ch.text for ch in chunks]
            r = vo.rerank(query=query, documents=docs, model='rerank-2')
            for i, score in enumerate(r.get('scores', [])[:top_k]):
                results.append((chunks[i], float(score)))
            used = 'voyage'
        except Exception:
            results = []
            used = None
    # Simple fallback
    if not results:
        results = _simple_rerank(query, chunks)[:top_k]
        used = 'simple'
    out: List[Dict[str, Any]] = []
    for ch, score in results[:top_k]:
        # Highlight first matching snippet lines with absolute line numbers
        qwords = re.findall(r"\w+", query.lower())
        lines = ch.text.splitlines()
        highlights: list[Dict[str, Any]] = []
        for idx, ln in enumerate(lines, start=0):
            if any(w in ln.lower() for w in qwords):
                abs_line = ch.start_line + idx
                highlights.append({'line': abs_line, 'text': ln.strip()})
            if len(highlights) >= 3:
                break
        out.append({
            'id': ch.id,
            'score': float(score),
            'start_line': ch.start_line,
            'end_line': ch.end_line,
            'highlights': highlights,
            'preview': lines[:3],
        })
    return out
